import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.hipporag.utils.misc_utils import QuerySolution, string_to_bool

from eval_causal_qwen3 import (
    LEARNED_SETWISE_FEATURE_NAMES,
    apply_setwise_selector,
    build_config,
    build_doc_text_to_chunk_id,
    collect_lexical_query_seed_entities,
    collect_query_seed_entities,
    compute_candidate_feature_rows,
    compute_slice_metrics,
    feature_rows_to_matrix,
    get_gold_answers,
    get_gold_docs,
)


def resolve_save_dir(save_dir: str, dataset: str) -> str:
    if save_dir == "outputs":
        return str(Path(save_dir) / dataset)
    return f"{save_dir}_{dataset}"


def load_dataset(dataset: str, limit: int) -> tuple[list[dict], list[dict]]:
    corpus_path = Path(f"reproduce/dataset/{dataset}_corpus.json")
    sample_path = Path(f"reproduce/dataset/{dataset}.json")
    corpus = json.load(corpus_path.open())
    samples = json.load(sample_path.open())
    if limit and limit > 0:
        samples = samples[:limit]
    return corpus, samples


def build_canonical_args(cli_args: argparse.Namespace, save_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        save_dir=save_dir,
        llm_base_url=cli_args.llm_base_url,
        llm_name=cli_args.llm_name,
        embedding_base_url=cli_args.embedding_base_url,
        dataset=cli_args.dataset,
        embedding_name=cli_args.embedding_name,
        force_index_from_scratch=cli_args.force_index_from_scratch,
        force_openie_from_scratch=cli_args.force_openie_from_scratch,
        max_retry_attempts=cli_args.max_retry_attempts,
        openie_mode=cli_args.openie_mode,
        planner_enabled="false",
        planner_mode="none",
        planner_max_steps=3,
        retrieval_top_k=cli_args.retrieval_top_k,
        linking_top_k=cli_args.linking_top_k,
        qa_top_k=cli_args.qa_top_k,
        max_qa_steps=cli_args.max_qa_steps,
        embedding_batch_size=cli_args.embedding_batch_size,
        causal_enabled="false",
        causal_query_only="true",
        causal_gate_mode="hard",
        causal_seed_top_k=20,
        causal_confidence_threshold=0.5,
        causal_damping=0.7,
        causal_blend_dense_weight=0.35,
        causal_blend_fact_weight=0.15,
        causal_blend_graph_weight=0.50,
        causal_margin_gate_enabled="false",
        causal_margin_threshold=0.02,
        causal_blend_top_k=0,
        causal_engine_version="v2",
        causal_v2_probe_mode="router",
        causal_v2_graph_mode="causal",
        causal_v2_base_retrieval_mode="legacy_fact_graph",
        general_graph_related_to_weight=0.3,
        general_graph_seed_top_k=10,
        causal_v2_extraction_max_tokens=768,
        causal_v2_extraction_retry_attempts=2,
        causal_v2_extraction_workers=4,
        causal_event_top_k=8,
        causal_v2_max_hops=2,
        causal_chain_top_k=6,
        causal_context_max_items=0,
        causal_er_similarity_threshold=0.92,
        causal_er_text_threshold=0.55,
        causal_v2_min_edge_confidence=0.7,
        structure_rerank_enabled="true",
        structure_rerank_top_n=40,
        structure_rerank_bonus_weight=0.08,
        structure_rerank_min_edge_support=2,
        structure_rerank_max_top5_swaps=2,
        structure_rerank_seed_top_k=4,
        structure_rerank_max_hops=2,
        structure_rerank_margin_threshold=0.02,
        rerank_require_non_empty="true",
    )


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


def subset_list(values: Sequence, indices: Sequence[int]) -> list:
    return [values[idx] for idx in indices]


def subset_query_solutions(query_solutions: Sequence[QuerySolution], indices: Sequence[int]) -> list[QuerySolution]:
    return [query_solutions[idx] for idx in indices]


def map_pool_doc_ids(pool_docs: Sequence[str],
                     doc_text_to_chunk_id: Dict[str, str],
                     text_to_hash_id: Dict[str, str],
                     hipporag: HippoRAG) -> list[int | None]:
    pool_doc_ids: list[int | None] = []
    for doc_text in pool_docs:
        chunk_id = doc_text_to_chunk_id.get(doc_text) or text_to_hash_id.get(doc_text)
        mapped_doc_id = hipporag.passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
        pool_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)
    return pool_doc_ids


def choose_oracle_next_position(pool_docs: Sequence[str],
                                gold_set: set[str],
                                remaining_positions: Sequence[int]) -> int | None:
    for pos in remaining_positions:
        if pool_docs[pos] in gold_set:
            return int(pos)
    return None


def build_training_rows(query_solutions: Sequence[QuerySolution],
                        gold_docs: Sequence[Sequence[str]],
                        hipporag: HippoRAG,
                        doc_text_to_chunk_id: Dict[str, str],
                        pool_k: int,
                        qa_top_k: int,
                        anchor_count: int,
                        structure_max_hops: int,
                        training_focus: str) -> tuple[list[dict], dict]:
    text_to_hash_id = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}
    rows: list[dict] = []
    positive_rows = 0
    state_count = 0
    query_with_positive_state = 0
    skipped_easy_queries = 0
    skipped_queries_without_promotable_gold = 0

    for q_idx, qs in enumerate(query_solutions):
        pool_limit = min(len(qs.docs), max(pool_k, qa_top_k))
        pool_docs = list(qs.docs[:pool_limit])
        gold_set = set(gold_docs[q_idx])
        full_support_in_topk = gold_set.issubset(set(pool_docs[:qa_top_k]))
        if training_focus in {"missing_topk", "promote_missing_only"} and full_support_in_topk:
            skipped_easy_queries += 1
            continue
        promotable_gold_positions = {
            pos
            for pos in range(pool_limit)
            if pool_docs[pos] in gold_set and pos >= qa_top_k
        }
        if training_focus == "promote_missing_only" and not promotable_gold_positions:
            skipped_queries_without_promotable_gold += 1
            continue

        if qs.doc_scores is not None and len(qs.doc_scores) >= pool_limit:
            pool_scores = np.asarray(qs.doc_scores[:pool_limit], dtype=float)
        else:
            pool_scores = np.linspace(pool_limit, 1, pool_limit, dtype=float)
        pool_doc_ids = map_pool_doc_ids(pool_docs, doc_text_to_chunk_id, text_to_hash_id, hipporag)

        seed_entities = collect_query_seed_entities(hipporag, qs.question)
        if not seed_entities:
            seed_entities = collect_lexical_query_seed_entities(
                query=qs.question,
                pool_doc_ids=pool_doc_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )

        selected_positions = list(range(min(max(anchor_count, 0), min(pool_limit, qa_top_k))))
        query_had_positive_state = False

        while len(selected_positions) < min(pool_limit, qa_top_k):
            remaining_positions = [pos for pos in range(pool_limit) if pos not in selected_positions]
            if training_focus == "promote_missing_only":
                positive_positions = [pos for pos in remaining_positions if pos in promotable_gold_positions]
            else:
                positive_positions = [pos for pos in remaining_positions if pool_docs[pos] in gold_set]
            if not positive_positions:
                break

            query_had_positive_state = True
            state_count += 1
            feature_rows = compute_candidate_feature_rows(
                query=qs.question,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                pool_doc_scores=pool_scores,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                adjacency=hipporag.structure_graph_out,
                qa_top_k=qa_top_k,
                selected_positions=selected_positions,
                seed_entities=seed_entities,
                structure_max_hops=structure_max_hops,
                candidate_positions=remaining_positions,
            )

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

            if training_focus == "promote_missing_only":
                oracle_next = min(positive_positions)
            else:
                oracle_next = choose_oracle_next_position(pool_docs, gold_set, remaining_positions)
            if oracle_next is None:
                break
            selected_positions.append(oracle_next)

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


def compute_full_support_rate(query_solutions: Sequence[QuerySolution],
                              gold_docs: Sequence[Sequence[str]],
                              k: int) -> float:
    supported = 0
    for q_idx, qs in enumerate(query_solutions):
        if set(gold_docs[q_idx]).issubset(set(qs.docs[:k])):
            supported += 1
    return float(supported / max(1, len(query_solutions)))


def compute_bucket_qa_summary(config,
                              baseline_solutions: Sequence[QuerySolution],
                              selector_solutions: Sequence[QuerySolution],
                              gold_docs: Sequence[Sequence[str]],
                              gold_answers: Sequence[Sequence[str]]) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description="Train a minimal learned setwise selector on retrieval pools.")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--train_queries", type=int, default=160)
    parser.add_argument("--eval_queries", type=int, default=40)
    parser.add_argument("--split_seed", type=int, default=13)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general")
    parser.add_argument("--output_dir", type=str, default="research_memory/emnlp_expand_then_compose/models")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_type", choices=["logistic_regression", "hist_gbdt"], default="logistic_regression")
    parser.add_argument("--training_focus", choices=["all", "missing_topk", "promote_missing_only"], default="missing_topk")
    parser.add_argument("--setwise_pool_k", type=int, default=100)
    parser.add_argument("--setwise_anchor_count", type=int, default=1)
    parser.add_argument("--setwise_structure_max_hops", type=int, default=2)
    parser.add_argument("--qa_top_k", type=int, default=5)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path else (
        output_dir / f"{args.dataset}_learned_greedy_{args.model_type}_pool{args.setwise_pool_k}_limit{len(samples)}.joblib"
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
        "Retrieving %d queries for learned setwise training/eval on %s (train=%d, eval=%d)",
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

    train_rows, train_row_summary = build_training_rows(
        query_solutions=train_query_solutions,
        gold_docs=train_gold_docs,
        hipporag=hipporag,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
        pool_k=int(args.setwise_pool_k),
        qa_top_k=int(args.qa_top_k),
        anchor_count=int(args.setwise_anchor_count),
        structure_max_hops=int(args.setwise_structure_max_hops),
        training_focus=args.training_focus,
    )
    if not train_rows:
        raise ValueError("No training rows were generated; check the retrieval pool and split sizes")

    train_matrix = feature_rows_to_matrix(train_rows)
    train_labels = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    model = instantiate_model(args.model_type, args.split_seed)
    model.fit(train_matrix, train_labels)

    if hasattr(model, "predict_proba"):
        train_probabilities = np.asarray(model.predict_proba(train_matrix), dtype=float)[:, 1]
    else:
        decision_scores = np.asarray(model.decision_function(train_matrix), dtype=float).reshape(-1)
        train_probabilities = 1.0 / (1.0 + np.exp(-decision_scores))
    train_metrics = compute_binary_metrics(train_labels, train_probabilities)

    model_bundle = {
        "model": model,
        "feature_names": list(LEARNED_SETWISE_FEATURE_NAMES),
        "model_type": args.model_type,
        "dataset": args.dataset,
        "save_dir": resolved_save_dir,
        "pool_k": int(args.setwise_pool_k),
        "qa_top_k": int(args.qa_top_k),
        "anchor_count": int(args.setwise_anchor_count),
        "structure_max_hops": int(args.setwise_structure_max_hops),
        "training_focus": args.training_focus,
        "split_seed": int(args.split_seed),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_row_summary": train_row_summary,
        "train_metrics": train_metrics,
    }
    joblib.dump(model_bundle, model_path)
    logger.info("Saved learned setwise model bundle to %s", model_path)

    baseline_retrieval = compute_slice_metrics(
        config=config,
        query_solutions=eval_query_solutions,
        gold_docs=eval_gold_docs,
        gold_answers=None,
    )
    selected_eval_solutions, selector_summary = apply_setwise_selector(
        hipporag=hipporag,
        query_solutions=eval_query_solutions,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
        pool_k=int(args.setwise_pool_k),
        qa_top_k=int(args.qa_top_k),
        selector_name="learned_greedy",
        anchor_count=int(args.setwise_anchor_count),
        reserve_top_m=0,
        structure_max_hops=int(args.setwise_structure_max_hops),
        base_weight=0.0,
        structure_weight=0.0,
        novelty_weight=0.0,
        learned_model_bundle=model_bundle,
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
        "retrieval_metrics_full_run": retrieval_metrics,
        "split": {
            "train_query_count": len(train_indices),
            "eval_query_count": len(eval_indices),
            "split_seed": int(args.split_seed),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
        },
        "selector_config": {
            "selector": "learned_greedy",
            "model_type": args.model_type,
            "model_path": str(model_path),
            "pool_k": int(args.setwise_pool_k),
            "qa_top_k": int(args.qa_top_k),
            "anchor_count": int(args.setwise_anchor_count),
            "structure_max_hops": int(args.setwise_structure_max_hops),
            "training_focus": args.training_focus,
        },
        "training": {
            **train_row_summary,
            "train_metrics": train_metrics,
        },
        "eval_retrieval": {
            "baseline": baseline_retrieval["overall"],
            "learned_greedy": selector_retrieval["overall"],
            "baseline_FS@5": round(float(baseline_fs5), 4),
            "learned_greedy_FS@5": round(float(selector_fs5), 4),
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
            "learned_greedy": selector_qa_results,
            "EM_delta": round(selector_em - baseline_em, 4),
            "F1_delta": round(selector_f1 - baseline_f1, 4),
            "per_bucket": bucket_qa_summary,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "Learned setwise report saved to %s | eval Recall@5 %.4f -> %.4f",
        output_json,
        float(baseline_retrieval["overall"].get("Recall@5", 0.0)),
        float(selector_retrieval["overall"].get("Recall@5", 0.0)),
    )


if __name__ == "__main__":
    main()
