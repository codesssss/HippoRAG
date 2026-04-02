import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

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
)
from requirement_beam_utils import (
    DEFAULT_REQUIREMENT_CF_TAU,
    DEFAULT_REQUIREMENT_SMOOTH_TAU,
    NEED_UNIT_CACHE_VERSION,
    NEED_UNIT_MATCHER_FEATURE_NAMES,
    get_requirement_cache_version,
    load_requirement_cache,
    requirement_feature_rows_to_matrix,
)
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import QuerySolution, string_to_bool
from train_requirement_setwise import (
    build_requirement_training_rows,
    compute_binary_metrics,
    compute_bucket_qa_summary,
    compute_full_support_rate,
    instantiate_model,
    split_indices,
    subset_list,
    subset_query_solutions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight V2 need-unit matcher for requirement_beam.")
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
    if get_requirement_cache_version(requirement_cache) != NEED_UNIT_CACHE_VERSION:
        raise ValueError(
            f"train_need_unit_scorer expects a V2 need-unit cache, got {requirement_cache.get('version')}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path else (
        output_dir / f"{args.dataset}_need_unit_beam_{args.model_type}_pool{args.setwise_pool_k}_limit{len(samples)}.joblib"
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
        "Retrieving %d queries for need-unit scorer training/eval on %s (train=%d, eval=%d)",
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
        raise ValueError("No need-unit training rows were generated; check the cache and split sizes")

    train_matrix = requirement_feature_rows_to_matrix(
        train_rows,
        feature_names=NEED_UNIT_MATCHER_FEATURE_NAMES,
    )
    train_labels = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    model = instantiate_model(args.model_type, args.split_seed)
    model.fit(train_matrix, train_labels)
    train_probabilities = model.predict_proba(train_matrix)[:, 1]
    train_metrics = compute_binary_metrics(train_labels, train_probabilities)

    model_bundle = {
        "model": model,
        "feature_names": list(NEED_UNIT_MATCHER_FEATURE_NAMES),
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
    logger.info("Saved need-unit matcher bundle to %s", model_path)

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
        "need_unit_cache_path": str(args.requirement_cache_path),
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
            "schema_version": NEED_UNIT_CACHE_VERSION,
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
            "need_unit_beam_learned": selector_retrieval["overall"],
            "baseline_FS@5": round(float(baseline_fs5), 4),
            "need_unit_beam_learned_FS@5": round(float(selector_fs5), 4),
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
            "need_unit_beam_learned": selector_qa_results,
            "EM_delta": round(selector_em - baseline_em, 4),
            "F1_delta": round(selector_f1 - baseline_f1, 4),
            "per_bucket": bucket_qa_summary,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    logger.info(
        "Need-unit matcher report saved to %s | eval Recall@5 %.4f -> %.4f",
        output_json,
        float(baseline_retrieval["overall"].get("Recall@5", 0.0)),
        float(selector_retrieval["overall"].get("Recall@5", 0.0)),
    )


if __name__ == "__main__":
    main()
