import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_requirement_cache import build_canonical_args, load_dataset, resolve_save_dir
from eval_causal_qwen3 import (
    apply_setwise_selector,
    build_config,
    build_doc_text_to_chunk_id,
    compute_slice_metrics,
    get_gold_answers,
    get_gold_docs,
    min_max_normalize_array,
    predict_binary_scores,
    resolve_reserved_positions,
    resolve_selection_target_k,
)
from requirement_beam_utils import (
    DEFAULT_REQUIREMENT_CF_TAU,
    DEFAULT_REQUIREMENT_SMOOTH_TAU,
    REQUIREMENT_MATCHER_FEATURE_NAMES,
    compute_requirement_candidate_feature_rows,
    load_requirement_cache,
    requirement_feature_rows_to_matrix,
    resolve_requirement_cache_entry,
)
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.hipporag.utils.misc_utils import QuerySolution, string_to_bool


def split_indices(total_count: int, train_queries: int, eval_queries: int, split_seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(split_seed)
    indices = np.arange(total_count, dtype=int)
    rng.shuffle(indices)
    train_count = min(max(train_queries, 0), total_count)
    remaining = max(0, total_count - train_count)
    eval_count = min(max(eval_queries, 0), remaining)
    train_indices = indices[:train_count].tolist()
    eval_indices = indices[train_count:train_count + eval_count].tolist()
    return train_indices, eval_indices


def subset_list(values, indices: list[int]) -> list:
    return [values[idx] for idx in indices]


def subset_query_solutions(query_solutions: list[QuerySolution], indices: list[int]) -> list[QuerySolution]:
    return [query_solutions[idx] for idx in indices]


def instantiate_model(model_type: str, random_seed: int):
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_seed,
        )
    if model_type == "hist_gbdt":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            min_samples_leaf=20,
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    metrics = {}
    if labels.size == 0:
        return {"row_count": 0}
    metrics["row_count"] = int(labels.size)
    metrics["positive_rate"] = round(float(np.mean(labels)), 6)
    if len(np.unique(labels)) < 2:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        return metrics
    metrics["roc_auc"] = round(float(roc_auc_score(labels, probabilities)), 4)
    metrics["average_precision"] = round(float(average_precision_score(labels, probabilities)), 4)
    return metrics


def compute_full_support_rate(query_solutions: list[QuerySolution],
                              gold_docs: list[list[str]],
                              k: int) -> float:
    supported = 0
    for q_idx, qs in enumerate(query_solutions):
        if set(gold_docs[q_idx]).issubset(set(qs.docs[:k])):
            supported += 1
    return float(supported / max(1, len(query_solutions)))


def compute_bucket_qa_summary(config,
                              baseline_solutions: list[QuerySolution],
                              selector_solutions: list[QuerySolution],
                              gold_docs: list[list[str]],
                              gold_answers: list[list[str]]) -> dict:
    qa_em_metric = QAExactMatch(global_config=config)
    qa_f1_metric = QAF1Score(global_config=config)
    baseline_answers = [qs.answer or "" for qs in baseline_solutions]
    selector_answers = [qs.answer or "" for qs in selector_solutions]
    _, baseline_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, baseline_answers)
    _, baseline_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, baseline_answers)
    _, selector_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, selector_answers)
    _, selector_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, selector_answers)

    bucket_stats: dict[int, dict] = {}
    for q_idx in range(len(gold_docs)):
        bucket = len(set(gold_docs[q_idx]))
        bucket_stats.setdefault(bucket, {
            "count": 0,
            "baseline_em": [],
            "selector_em": [],
            "baseline_f1": [],
            "selector_f1": [],
        })
        bucket_stats[bucket]["count"] += 1
        bucket_stats[bucket]["baseline_em"].append(baseline_per_query_em[q_idx]["ExactMatch"])
        bucket_stats[bucket]["selector_em"].append(selector_per_query_em[q_idx]["ExactMatch"])
        bucket_stats[bucket]["baseline_f1"].append(baseline_per_query_f1[q_idx]["F1"])
        bucket_stats[bucket]["selector_f1"].append(selector_per_query_f1[q_idx]["F1"])

    summary = {}
    for bucket, stats in sorted(bucket_stats.items()):
        summary[f"{bucket}-doc"] = {
            "count": int(stats["count"]),
            "baseline_EM": round(float(np.mean(stats["baseline_em"])), 4),
            "selector_EM": round(float(np.mean(stats["selector_em"])), 4),
            "EM_delta": round(float(np.mean(stats["selector_em"])) - float(np.mean(stats["baseline_em"])), 4),
            "baseline_F1": round(float(np.mean(stats["baseline_f1"])), 4),
            "selector_F1": round(float(np.mean(stats["selector_f1"])), 4),
            "F1_delta": round(float(np.mean(stats["selector_f1"])) - float(np.mean(stats["baseline_f1"])), 4),
        }
    return summary


def build_requirement_training_rows(query_solutions: list[QuerySolution],
                                    gold_docs: list[list[str]],
                                    requirement_cache: dict,
                                    pool_k: int,
                                    qa_top_k: int,
                                    anchor_count: int,
                                    reserve_top_m: int,
                                    max_bridge_slots: int,
                                    smooth_tau: float,
                                    counterfactual_tau: float,
                                    training_focus: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    positive_rows = 0
    state_count = 0
    query_with_positive_state = 0
    skipped_easy_queries = 0
    skipped_queries_without_promotable_gold = 0

    for q_idx, qs in enumerate(query_solutions):
        cache_entry = resolve_requirement_cache_entry(requirement_cache, qs.question)
        effective_pool_limit = min(
            len(qs.docs),
            max(int(pool_k), int(qa_top_k)),
            int(cache_entry.get("annotation_pool_k", 0) or 0),
        )
        if effective_pool_limit <= 0:
            continue
        pool_docs = list(qs.docs[:effective_pool_limit])
        gold_set = set(gold_docs[q_idx])
        full_support_in_topk = gold_set.issubset(set(pool_docs[:qa_top_k]))
        if training_focus in {"missing_topk", "promote_missing_only"} and full_support_in_topk:
            skipped_easy_queries += 1
            continue
        promotable_gold_positions = {
            pos for pos in range(effective_pool_limit)
            if pool_docs[pos] in gold_set and pos >= qa_top_k
        }
        if training_focus == "promote_missing_only" and not promotable_gold_positions:
            skipped_queries_without_promotable_gold += 1
            continue

        if qs.doc_scores is not None and len(qs.doc_scores) >= effective_pool_limit:
            normalized_base_scores = min_max_normalize_array(np.asarray(qs.doc_scores[:effective_pool_limit], dtype=float))
        else:
            normalized_base_scores = min_max_normalize_array(
                np.linspace(effective_pool_limit, 1, effective_pool_limit, dtype=float)
            )

        candidate_count = effective_pool_limit
        target_k = min(candidate_count, qa_top_k)
        anchor_positions, reserved_positions = resolve_reserved_positions(
            candidate_count=candidate_count,
            target_k=target_k,
            anchor_count=anchor_count,
            reserve_top_m=reserve_top_m,
        )
        selection_target_k = resolve_selection_target_k(
            target_k=target_k,
            reserved_count=len(reserved_positions),
            max_bridge_slots=max_bridge_slots,
        )
        selected_positions = list(reserved_positions)
        query_had_positive_state = False

        while len(selected_positions) < selection_target_k:
            remaining_positions = [
                pos for pos in range(candidate_count)
                if pos not in set(selected_positions)
            ]
            if training_focus == "promote_missing_only":
                positive_positions = [pos for pos in remaining_positions if pos in promotable_gold_positions]
            else:
                positive_positions = [pos for pos in remaining_positions if pool_docs[pos] in gold_set]
            if not positive_positions:
                break

            query_had_positive_state = True
            state_count += 1
            feature_rows = compute_requirement_candidate_feature_rows(
                cache_entry=cache_entry,
                selected_positions=selected_positions,
                candidate_positions=remaining_positions,
                normalized_base_scores=normalized_base_scores,
                qa_top_k=qa_top_k,
                smooth_tau=smooth_tau,
                counterfactual_tau=counterfactual_tau,
            )
            feature_rows_by_position = {
                int(row["pool_position"]): row
                for row in feature_rows
            }
            for row in feature_rows:
                label = int(int(row["pool_position"]) in positive_positions)
                positive_rows += label
                rows.append({
                    "query_index": q_idx,
                    "question": qs.question,
                    "step_index": len(selected_positions),
                    "label": label,
                    **row,
                })

            oracle_next = max(
                positive_positions,
                key=lambda pos: (
                    float(feature_rows_by_position[pos]["utility_margin_after"]),
                    float(feature_rows_by_position[pos]["support_completeness_after"]),
                    -float(feature_rows_by_position[pos]["counterfactual_leakage_after"]),
                    -int(pos),
                ),
            )
            selected_positions.append(int(oracle_next))

        if query_had_positive_state:
            query_with_positive_state += 1

    summary = {
        "row_count": len(rows),
        "positive_row_count": int(positive_rows),
        "positive_rate": round(float(positive_rows / max(1, len(rows))), 6),
        "state_count": int(state_count),
        "queries_with_positive_state": int(query_with_positive_state),
        "training_focus": training_focus,
        "skipped_easy_queries": int(skipped_easy_queries),
        "skipped_queries_without_promotable_gold": int(skipped_queries_without_promotable_gold),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight matcher for requirement_beam.")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--train_queries", type=int, default=160)
    parser.add_argument("--eval_queries", type=int, default=40)
    parser.add_argument("--split_seed", type=int, default=13)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general")
    parser.add_argument("--requirement_cache_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="research_memory/emnlp_expand_then_compose/models")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_type", choices=["logistic_regression", "hist_gbdt"], default="hist_gbdt")
    parser.add_argument("--training_focus", choices=["all", "missing_topk", "promote_missing_only"], default="missing_topk")
    parser.add_argument("--setwise_pool_k", type=int, default=100)
    parser.add_argument("--setwise_anchor_count", type=int, default=1)
    parser.add_argument("--setwise_reserve_top_m", type=int, default=1)
    parser.add_argument("--setwise_max_bridge_slots", type=int, default=0)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--smooth_tau", type=float, default=DEFAULT_REQUIREMENT_SMOOTH_TAU)
    parser.add_argument("--counterfactual_tau", type=float, default=DEFAULT_REQUIREMENT_CF_TAU)
    parser.add_argument("--eval_with_qa", type=str, default="true")
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    resolved_save_dir = resolve_save_dir(args.save_dir, args.dataset)
    corpus, samples = load_dataset(args.dataset, args.limit)
    if len(samples) < 2:
        raise ValueError("Need at least two samples to build a train/eval split")

    requirement_cache = load_requirement_cache(args.requirement_cache_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path else (
        output_dir / f"{args.dataset}_requirement_beam_{args.model_type}_pool{args.setwise_pool_k}_limit{len(samples)}.joblib"
    )
    output_json = Path(args.output_json) if args.output_json else model_path.with_suffix(".json")

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_docs = get_gold_docs(samples, args.dataset, corpus=corpus)
    gold_answers = get_gold_answers(samples)
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)

    train_indices, eval_indices = split_indices(len(samples), args.train_queries, args.eval_queries, args.split_seed)
    if not train_indices or not eval_indices:
        raise ValueError("Train/eval split is empty; increase --limit or adjust --train_queries/--eval_queries")

    config_args = build_canonical_args(args, resolved_save_dir)
    config = build_config(config_args, corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)

    logger.info(
        "Retrieving %d queries for requirement matcher training/eval on %s (train=%d, eval=%d)",
        len(queries),
        args.dataset,
        len(train_indices),
        len(eval_indices),
    )
    query_solutions, retrieval_metrics = hipporag.retrieve(
        queries=queries,
        gold_docs=gold_docs,
    )

    train_query_solutions = subset_query_solutions(query_solutions, train_indices)
    eval_query_solutions = subset_query_solutions(query_solutions, eval_indices)
    train_gold_docs = subset_list(gold_docs, train_indices)
    eval_gold_docs = subset_list(gold_docs, eval_indices)
    eval_gold_answers = subset_list(gold_answers, eval_indices)

    train_rows, train_row_summary = build_requirement_training_rows(
        query_solutions=train_query_solutions,
        gold_docs=train_gold_docs,
        requirement_cache=requirement_cache,
        pool_k=int(args.setwise_pool_k),
        qa_top_k=int(args.qa_top_k),
        anchor_count=int(args.setwise_anchor_count),
        reserve_top_m=int(args.setwise_reserve_top_m),
        max_bridge_slots=int(args.setwise_max_bridge_slots),
        smooth_tau=float(args.smooth_tau),
        counterfactual_tau=float(args.counterfactual_tau),
        training_focus=args.training_focus,
    )
    if not train_rows:
        raise ValueError("No requirement training rows were generated; check the cache and split sizes")

    train_matrix = requirement_feature_rows_to_matrix(train_rows)
    train_labels = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    model = instantiate_model(args.model_type, args.split_seed)
    model.fit(train_matrix, train_labels)
    train_probabilities = predict_binary_scores({"model": model}, train_matrix)
    train_metrics = compute_binary_metrics(train_labels, train_probabilities)

    model_bundle = {
        "model": model,
        "feature_names": list(REQUIREMENT_MATCHER_FEATURE_NAMES),
        "model_type": args.model_type,
        "dataset": args.dataset,
        "save_dir": resolved_save_dir,
        "pool_k": int(args.setwise_pool_k),
        "qa_top_k": int(args.qa_top_k),
        "anchor_count": int(args.setwise_anchor_count),
        "reserve_top_m": int(args.setwise_reserve_top_m),
        "max_bridge_slots": int(args.setwise_max_bridge_slots),
        "training_focus": args.training_focus,
        "split_seed": int(args.split_seed),
        "smooth_tau": float(args.smooth_tau),
        "counterfactual_tau": float(args.counterfactual_tau),
        "requirement_cache_path": str(args.requirement_cache_path),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_row_summary": train_row_summary,
        "train_metrics": train_metrics,
    }
    joblib.dump(model_bundle, model_path)
    logger.info("Saved requirement matcher bundle to %s", model_path)

    baseline_retrieval = compute_slice_metrics(
        config=config,
        query_solutions=eval_query_solutions,
        gold_docs=eval_gold_docs,
        gold_answers=None,
    )
    requirement_selector_bundle = {
        "cache": requirement_cache,
        "cache_path": str(args.requirement_cache_path),
        "mode": "learned",
        "model_bundle": model_bundle,
        "model_path": str(model_path),
        "smooth_tau": float(args.smooth_tau),
        "counterfactual_tau": float(args.counterfactual_tau),
    }
    selected_eval_solutions, selector_summary = apply_setwise_selector(
        hipporag=hipporag,
        query_solutions=eval_query_solutions,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
        pool_k=int(args.setwise_pool_k),
        qa_top_k=int(args.qa_top_k),
        selector_name="requirement_beam",
        score_mode="bridge",
        anchor_count=int(args.setwise_anchor_count),
        reserve_top_m=int(args.setwise_reserve_top_m),
        max_bridge_slots=int(args.setwise_max_bridge_slots),
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        learned_model_bundle=None,
        beam_width=4,
        beam_expand_per_state=4,
        beam_projected_shortlist_factor=1,
        non_anchor_title_dedup=True,
        query_entity_source="seed",
        gate_mode="none",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.0,
        requirement_selector_bundle=requirement_selector_bundle,
    )
    selector_retrieval = compute_slice_metrics(
        config=config,
        query_solutions=selected_eval_solutions,
        gold_docs=eval_gold_docs,
        gold_answers=None,
    )

    baseline_fs5 = compute_full_support_rate(eval_query_solutions, eval_gold_docs, k=int(args.qa_top_k))
    selector_fs5 = compute_full_support_rate(selected_eval_solutions, eval_gold_docs, k=int(args.qa_top_k))

    baseline_qa_results = None
    selector_qa_results = None
    bucket_qa_summary = {}
    if string_to_bool(args.eval_with_qa):
        baseline_eval_solutions, _, _, _, baseline_qa_results = hipporag.rag_qa(
            queries=[QuerySolution(
                question=qs.question,
                docs=list(qs.docs),
                doc_scores=np.asarray(qs.doc_scores, dtype=float) if qs.doc_scores is not None else None,
                gold_docs=qs.gold_docs,
                gold_answers=qs.gold_answers,
                retrieval_trace=qs.retrieval_trace,
                qa_trace=qs.qa_trace,
            ) for qs in eval_query_solutions],
            gold_docs=eval_gold_docs,
            gold_answers=eval_gold_answers,
        )
        selected_eval_solutions, _, _, _, selector_qa_results = hipporag.rag_qa(
            queries=selected_eval_solutions,
            gold_docs=eval_gold_docs,
            gold_answers=eval_gold_answers,
        )
        bucket_qa_summary = compute_bucket_qa_summary(
            config=config,
            baseline_solutions=baseline_eval_solutions,
            selector_solutions=selected_eval_solutions,
            gold_docs=eval_gold_docs,
            gold_answers=eval_gold_answers,
        )

    report = {
        "dataset": args.dataset,
        "limit": len(samples),
        "resolved_save_dir": resolved_save_dir,
        "requirement_cache_path": str(args.requirement_cache_path),
        "retrieval_metrics_full_run": retrieval_metrics,
        "split": {
            "train_query_count": len(train_indices),
            "eval_query_count": len(eval_indices),
            "split_seed": int(args.split_seed),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
        },
        "selector_config": {
            "selector": "requirement_beam",
            "mode": "learned",
            "model_type": args.model_type,
            "model_path": str(model_path),
            "pool_k": int(args.setwise_pool_k),
            "qa_top_k": int(args.qa_top_k),
            "anchor_count": int(args.setwise_anchor_count),
            "reserve_top_m": int(args.setwise_reserve_top_m),
            "max_bridge_slots": int(args.setwise_max_bridge_slots),
            "training_focus": args.training_focus,
            "smooth_tau": float(args.smooth_tau),
            "counterfactual_tau": float(args.counterfactual_tau),
        },
        "training": {
            **train_row_summary,
            "train_metrics": train_metrics,
        },
        "eval_retrieval": {
            "baseline": baseline_retrieval["overall"],
            "requirement_beam_learned": selector_retrieval["overall"],
            "baseline_FS@5": round(float(baseline_fs5), 4),
            "requirement_beam_learned_FS@5": round(float(selector_fs5), 4),
            "FS@5_delta": round(float(selector_fs5 - baseline_fs5), 4),
            "selector_summary": selector_summary,
        },
        "eval_qa": None,
    }

    if baseline_qa_results is not None and selector_qa_results is not None:
        baseline_em = float(baseline_qa_results.get("ExactMatch", 0.0))
        baseline_f1 = float(baseline_qa_results.get("F1", 0.0))
        selector_em = float(selector_qa_results.get("ExactMatch", 0.0))
        selector_f1 = float(selector_qa_results.get("F1", 0.0))
        report["eval_qa"] = {
            "baseline": baseline_qa_results,
            "requirement_beam_learned": selector_qa_results,
            "EM_delta": round(selector_em - baseline_em, 4),
            "F1_delta": round(selector_f1 - baseline_f1, 4),
            "per_bucket": bucket_qa_summary,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    logger.info(
        "Requirement matcher report saved to %s | eval Recall@5 %.4f -> %.4f",
        output_json,
        float(baseline_retrieval["overall"].get("Recall@5", 0.0)),
        float(selector_retrieval["overall"].get("Recall@5", 0.0)),
    )


if __name__ == "__main__":
    main()
