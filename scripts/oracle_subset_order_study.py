#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_requirement_cache import load_dataset, resolve_save_dir  # noqa: E402
from eval_causal_qwen3 import (  # noqa: E402
    build_config,
    build_doc_text_to_chunk_id,
    compute_slice_metrics,
    extract_doc_title,
    get_gold_answers,
    get_gold_docs,
)
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score  # noqa: E402
from src.hipporag.utils.misc_utils import QuerySolution  # noqa: E402


LOGGER = logging.getLogger(__name__)


def build_protocol_args(args: argparse.Namespace) -> SimpleNamespace:
    resolved_save_dir = resolve_save_dir(args.save_dir, args.dataset)
    return SimpleNamespace(
        save_dir=resolved_save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        llm_request_name=args.llm_request_name,
        embedding_base_url=args.embedding_base_url,
        dataset=args.dataset,
        embedding_name=args.embedding_name,
        force_index_from_scratch=args.force_index_from_scratch,
        force_openie_from_scratch=args.force_openie_from_scratch,
        max_retry_attempts=args.max_retry_attempts,
        openie_mode=args.openie_mode,
        planner_enabled="false",
        planner_mode="none",
        planner_max_steps=3,
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        qa_top_k=args.qa_top_k,
        max_qa_steps=args.max_qa_steps,
        embedding_batch_size=args.embedding_batch_size,
        causal_enabled=args.causal_enabled,
        causal_query_only=args.causal_query_only,
        causal_gate_mode=args.causal_gate_mode,
        causal_seed_top_k=args.causal_seed_top_k,
        causal_confidence_threshold=args.causal_confidence_threshold,
        causal_damping=args.causal_damping,
        causal_blend_dense_weight=args.causal_blend_dense_weight,
        causal_blend_fact_weight=args.causal_blend_fact_weight,
        causal_blend_graph_weight=args.causal_blend_graph_weight,
        causal_margin_gate_enabled=args.causal_margin_gate_enabled,
        causal_margin_threshold=args.causal_margin_threshold,
        causal_blend_top_k=args.causal_blend_top_k,
        causal_engine_version=args.causal_engine_version,
        causal_v2_probe_mode=args.causal_v2_probe_mode,
        causal_v2_graph_mode=args.causal_v2_graph_mode,
        causal_v2_base_retrieval_mode=args.causal_v2_base_retrieval_mode,
        general_graph_related_to_weight=args.general_graph_related_to_weight,
        general_graph_seed_top_k=args.general_graph_seed_top_k,
        causal_v2_extraction_max_tokens=args.causal_v2_extraction_max_tokens,
        causal_v2_extraction_retry_attempts=args.causal_v2_extraction_retry_attempts,
        causal_v2_extraction_workers=args.causal_v2_extraction_workers,
        causal_event_top_k=args.causal_event_top_k,
        causal_v2_max_hops=args.causal_v2_max_hops,
        causal_chain_top_k=args.causal_chain_top_k,
        causal_context_max_items=args.causal_context_max_items,
        causal_er_similarity_threshold=args.causal_er_similarity_threshold,
        causal_er_text_threshold=args.causal_er_text_threshold,
        causal_v2_min_edge_confidence=args.causal_v2_min_edge_confidence,
        structure_rerank_enabled=args.structure_rerank_enabled,
        structure_rerank_top_n=args.structure_rerank_top_n,
        structure_rerank_bonus_weight=args.structure_rerank_bonus_weight,
        structure_rerank_min_edge_support=args.structure_rerank_min_edge_support,
        structure_rerank_max_top5_swaps=args.structure_rerank_max_top5_swaps,
        structure_rerank_seed_top_k=args.structure_rerank_seed_top_k,
        structure_rerank_max_hops=args.structure_rerank_max_hops,
        structure_relation_probe_mode="off",
        structure_continuity_probe_mode="off",
        structure_seed_target_bridge_mode="off",
        structure_rerank_margin_threshold=args.structure_rerank_margin_threshold,
        rerank_require_non_empty=args.rerank_require_non_empty,
    )


def ordered_support_titles(sample: Dict[str, Any]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for item in list(sample.get("supporting_facts", []) or []):
        if not item:
            continue
        title = str(item[0]).strip()
        if not title or title in seen:
            continue
        seen.add(title)
        ordered.append(title)
    return ordered


def make_variant_query_solution(question: str,
                                docs: Sequence[str],
                                gold_doc_list: Sequence[str]) -> QuerySolution:
    return QuerySolution(
        question=question,
        docs=list(docs),
        doc_scores=np.ones(len(docs), dtype=float),
        gold_docs=list(gold_doc_list),
    )


def build_variant_inputs(samples: Sequence[Dict[str, Any]],
                         baseline_solutions: Sequence[QuerySolution],
                         gold_docs: Sequence[Sequence[str]],
                         pool_k: int,
                         random_seed: int) -> Dict[str, Any]:
    eligible_indices: List[int] = []
    variant_inputs = {
        "oracle_subset_pool_order": [],
        "oracle_subset_random_order": [],
        "oracle_subset_chain_order": [],
    }
    records: List[Dict[str, Any]] = []

    for q_idx, qs in enumerate(baseline_solutions):
        pool_docs = list(qs.docs[:pool_k])
        gold_doc_list = list(gold_docs[q_idx])
        gold_set = set(gold_doc_list)
        pool_set = set(pool_docs)
        full_support_in_pool = gold_set.issubset(pool_set)
        gold_in_pool_order = [doc_text for doc_text in pool_docs if doc_text in gold_set]
        gold_title_to_doc = {extract_doc_title(doc_text): doc_text for doc_text in gold_doc_list}
        support_title_order = ordered_support_titles(samples[q_idx])

        chain_docs: List[str] = []
        seen_docs: set[str] = set()
        for title in support_title_order:
            doc_text = gold_title_to_doc.get(title)
            if doc_text is None or doc_text not in pool_set or doc_text in seen_docs:
                continue
            seen_docs.add(doc_text)
            chain_docs.append(doc_text)
        for doc_text in gold_in_pool_order:
            if doc_text in seen_docs:
                continue
            seen_docs.add(doc_text)
            chain_docs.append(doc_text)

        rng = np.random.default_rng(int(random_seed) + int(q_idx))
        random_docs = list(chain_docs)
        if len(random_docs) > 1:
            rng.shuffle(random_docs)

        records.append({
            "question": qs.question,
            "baseline_top_titles": [extract_doc_title(doc_text) for doc_text in qs.docs[:5]],
            "support_titles": support_title_order,
            "gold_titles": [extract_doc_title(doc_text) for doc_text in gold_doc_list],
            "gold_titles_in_pool_order": [extract_doc_title(doc_text) for doc_text in gold_in_pool_order],
            "full_support_in_pool": bool(full_support_in_pool),
        })

        if not full_support_in_pool:
            continue

        eligible_indices.append(q_idx)
        variant_inputs["oracle_subset_pool_order"].append(
            make_variant_query_solution(
                question=qs.question,
                docs=gold_in_pool_order,
                gold_doc_list=gold_doc_list,
            )
        )
        variant_inputs["oracle_subset_random_order"].append(
            make_variant_query_solution(
                question=qs.question,
                docs=random_docs,
                gold_doc_list=gold_doc_list,
            )
        )
        variant_inputs["oracle_subset_chain_order"].append(
            make_variant_query_solution(
                question=qs.question,
                docs=chain_docs,
                gold_doc_list=gold_doc_list,
            )
        )

    return {
        "eligible_indices": eligible_indices,
        "variant_inputs": variant_inputs,
        "query_records": records,
    }


def subset_list(values: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    return [values[idx] for idx in indices]


def compute_per_query_scores(gold_answers: Sequence[Sequence[str]],
                             query_solutions: Sequence[QuerySolution]) -> List[Dict[str, float]]:
    qa_em = QAExactMatch(global_config=None)
    qa_f1 = QAF1Score(global_config=None)
    predicted_answers = [qs.answer or "" for qs in query_solutions]
    _, em_rows = qa_em.calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        aggregation_fn=np.max,
    )
    _, f1_rows = qa_f1.calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        aggregation_fn=np.max,
    )
    return [
        {
            "ExactMatch": float(em_rows[idx]["ExactMatch"]),
            "F1": float(f1_rows[idx]["F1"]),
        }
        for idx in range(len(query_solutions))
    ]


def summarize_condition(config,
                        gold_docs: Sequence[Sequence[str]],
                        gold_answers: Sequence[Sequence[str]],
                        query_solutions: Sequence[QuerySolution]) -> Dict[str, Any]:
    slice_metrics = compute_slice_metrics(
        config=config,
        query_solutions=list(query_solutions),
        gold_docs=list(gold_docs),
        gold_answers=list(gold_answers),
    )
    per_query_scores = compute_per_query_scores(gold_answers=gold_answers, query_solutions=query_solutions)
    return {
        "overall": slice_metrics["overall"],
        "per_query_scores": per_query_scores,
    }


def compare_against_baseline(baseline_scores: Sequence[Dict[str, float]],
                             condition_scores: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    delta_f1 = [
        float(condition_scores[idx]["F1"]) - float(baseline_scores[idx]["F1"])
        for idx in range(len(baseline_scores))
    ]
    delta_em = [
        float(condition_scores[idx]["ExactMatch"]) - float(baseline_scores[idx]["ExactMatch"])
        for idx in range(len(baseline_scores))
    ]
    improved = sum(1 for delta in delta_f1 if delta > 1e-9)
    worsened = sum(1 for delta in delta_f1 if delta < -1e-9)
    unchanged = len(delta_f1) - improved - worsened
    return {
        "mean_F1_delta": round(float(np.mean(delta_f1)) if delta_f1 else 0.0, 4),
        "mean_EM_delta": round(float(np.mean(delta_em)) if delta_em else 0.0, 4),
        "improved_count": int(improved),
        "worsened_count": int(worsened),
        "unchanged_count": int(unchanged),
    }


def attach_query_results(query_records: Sequence[Dict[str, Any]],
                         eligible_indices: Sequence[int],
                         baseline_subset: Sequence[QuerySolution],
                         baseline_scores: Sequence[Dict[str, float]],
                         condition_runs: Dict[str, Sequence[QuerySolution]],
                         condition_summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_index = {idx: dict(query_records[idx]) for idx in eligible_indices}
    local_idx_by_global = {global_idx: local_idx for local_idx, global_idx in enumerate(eligible_indices)}

    for global_idx in eligible_indices:
        local_idx = local_idx_by_global[global_idx]
        record = by_index[global_idx]
        record["baseline_answer"] = str(baseline_subset[local_idx].answer or "")
        record["baseline_metrics"] = baseline_scores[local_idx]
        for condition_name, run_solutions in condition_runs.items():
            record[f"{condition_name}_titles"] = [
                extract_doc_title(doc_text) for doc_text in run_solutions[local_idx].docs
            ]
            record[f"{condition_name}_answer"] = str(run_solutions[local_idx].answer or "")
            record[f"{condition_name}_metrics"] = condition_summaries[condition_name]["per_query_scores"][local_idx]
            record[f"{condition_name}_delta_F1"] = round(
                float(condition_summaries[condition_name]["per_query_scores"][local_idx]["F1"]) -
                float(baseline_scores[local_idx]["F1"]),
                4,
            )
        record["random_equals_chain"] = (
            record.get("oracle_subset_random_order_titles", []) ==
            record.get("oracle_subset_chain_order_titles", [])
        )
    return [by_index[idx] for idx in eligible_indices]


def build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# Oracle Subset Order Study: {report['dataset']}",
        "",
        f"- Queries: `{report['num_queries']}`",
        f"- Eligible (full support in top-{report['pool_k']}): `{report['eligible_count']}` / `{report['num_queries']}` "
        f"(`{report['eligible_rate']:.1%}`)",
        f"- QA top-k: `{report['qa_top_k']}`",
        "",
        "## Comparable Slice",
        "",
        "| Condition | EM | F1 | ΔEM | ΔF1 |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = report["conditions"]["baseline_top5"]["overall"]
    lines.append(
        f"| baseline_top5 | {baseline.get('ExactMatch', 0.0):.4f} | {baseline.get('F1', 0.0):.4f} | +0.0000 | +0.0000 |"
    )
    for condition_name in [
        "oracle_subset_pool_order",
        "oracle_subset_random_order",
        "oracle_subset_chain_order",
    ]:
        overall = report["conditions"][condition_name]["overall"]
        deltas = report["pairwise_vs_baseline"][condition_name]
        lines.append(
            f"| {condition_name} | {overall.get('ExactMatch', 0.0):.4f} | {overall.get('F1', 0.0):.4f} | "
            f"{deltas.get('mean_EM_delta', 0.0):+.4f} | {deltas.get('mean_F1_delta', 0.0):+.4f} |"
        )

    lines.extend([
        "",
        "## Pairwise Summary",
        "",
    ])
    for condition_name, summary in report["pairwise_vs_baseline"].items():
        lines.append(
            f"- `{condition_name}`: mean ΔF1 `{summary['mean_F1_delta']:+.4f}`, "
            f"improved `{summary['improved_count']}`, worsened `{summary['worsened_count']}`, "
            f"unchanged `{summary['unchanged_count']}`"
        )

    lines.extend([
        "",
        "## Example Queries",
        "",
    ])
    for row in list(report.get("query_records", []) or [])[: min(8, len(report.get("query_records", []) or []))]:
        lines.append(f"- Q: {row['question']}")
        lines.append(f"  baseline top5: {row['baseline_top_titles']}")
        lines.append(f"  gold/pool order: {row['oracle_subset_pool_order_titles']}")
        lines.append(f"  random order: {row['oracle_subset_random_order_titles']}")
        lines.append(f"  chain order: {row['oracle_subset_chain_order_titles']}")
        lines.append(
            f"  baseline F1={row['baseline_metrics']['F1']:.4f}, "
            f"pool ΔF1={row['oracle_subset_pool_order_delta_F1']:+.4f}, "
            f"random ΔF1={row['oracle_subset_random_order_delta_F1']:+.4f}, "
            f"chain ΔF1={row['oracle_subset_chain_order_delta_F1']:+.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-pool oracle subset and ordering study.")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general")
    parser.add_argument("--report_json", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--output_md", type=str, default="")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8043/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--llm_request_name", type=str, default="qwen3-8b-train")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--causal_enabled", type=str, default="true")
    parser.add_argument("--causal_query_only", type=str, default="true")
    parser.add_argument("--causal_gate_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--causal_seed_top_k", type=int, default=20)
    parser.add_argument("--causal_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--causal_damping", type=float, default=0.7)
    parser.add_argument("--causal_blend_dense_weight", type=float, default=0.35)
    parser.add_argument("--causal_blend_fact_weight", type=float, default=0.15)
    parser.add_argument("--causal_blend_graph_weight", type=float, default=0.50)
    parser.add_argument("--causal_margin_gate_enabled", type=str, default="false")
    parser.add_argument("--causal_margin_threshold", type=float, default=0.02)
    parser.add_argument("--causal_blend_top_k", type=int, default=0)
    parser.add_argument("--causal_engine_version", choices=["legacy", "v2"], default="v2")
    parser.add_argument("--causal_v2_probe_mode", choices=["router", "always"], default="router")
    parser.add_argument("--causal_v2_graph_mode", choices=["causal", "general"], default="causal")
    parser.add_argument("--causal_v2_base_retrieval_mode", choices=["dense", "legacy_fact_graph", "general_relation_graph"], default="legacy_fact_graph")
    parser.add_argument("--general_graph_related_to_weight", type=float, default=0.3)
    parser.add_argument("--general_graph_seed_top_k", type=int, default=10)
    parser.add_argument("--causal_v2_extraction_max_tokens", type=int, default=768)
    parser.add_argument("--causal_v2_extraction_retry_attempts", type=int, default=2)
    parser.add_argument("--causal_v2_extraction_workers", type=int, default=4)
    parser.add_argument("--causal_event_top_k", type=int, default=8)
    parser.add_argument("--causal_v2_max_hops", type=int, default=2)
    parser.add_argument("--causal_chain_top_k", type=int, default=6)
    parser.add_argument("--causal_context_max_items", type=int, default=0)
    parser.add_argument("--causal_er_similarity_threshold", type=float, default=0.92)
    parser.add_argument("--causal_er_text_threshold", type=float, default=0.55)
    parser.add_argument("--causal_v2_min_edge_confidence", type=float, default=0.7)
    parser.add_argument("--structure_rerank_enabled", type=str, default="true")
    parser.add_argument("--structure_rerank_top_n", type=int, default=40)
    parser.add_argument("--structure_rerank_bonus_weight", type=float, default=0.08)
    parser.add_argument("--structure_rerank_min_edge_support", type=int, default=2)
    parser.add_argument("--structure_rerank_max_top5_swaps", type=int, default=2)
    parser.add_argument("--structure_rerank_seed_top_k", type=int, default=4)
    parser.add_argument("--structure_rerank_max_hops", type=int, default=2)
    parser.add_argument("--structure_rerank_margin_threshold", type=float, default=0.02)
    parser.add_argument("--rerank_require_non_empty", type=str, default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    corpus, samples = load_dataset(args.dataset, args.limit)
    all_gold_docs = get_gold_docs(samples, args.dataset, corpus=corpus)
    all_gold_answers = get_gold_answers(samples)
    sample_by_question = {str(sample["question"]): sample for sample in samples}
    gold_docs_by_question = {
        str(samples[idx]["question"]): list(all_gold_docs[idx])
        for idx in range(len(samples))
    }
    gold_answers_by_question = {
        str(samples[idx]["question"]): list(all_gold_answers[idx])
        for idx in range(len(samples))
    }

    config = build_config(build_protocol_args(args), corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)

    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    chunk_id_to_doc_text = {
        chunk_id: doc_text
        for doc_text, chunk_id in doc_text_to_chunk_id.items()
    }

    if args.report_json:
        report_path = Path(args.report_json)
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_examples = list(report_payload.get("examples", []) or [])
        if args.limit and args.limit > 0:
            report_examples = report_examples[: int(args.limit)]

        baseline_solutions: List[QuerySolution] = []
        selected_samples: List[Dict[str, Any]] = []
        gold_docs: List[List[str]] = []
        gold_answers: List[List[str]] = []
        for example in report_examples:
            question = str(example.get("question", ""))
            if question not in gold_docs_by_question or question not in sample_by_question:
                continue
            retrieved_doc_ids = list(example.get("retrieved_doc_ids", []) or [])
            reconstructed_docs = [
                chunk_id_to_doc_text[chunk_id]
                for chunk_id in retrieved_doc_ids
                if isinstance(chunk_id, str) and chunk_id in chunk_id_to_doc_text
            ]
            if not reconstructed_docs:
                reconstructed_docs = list(example.get("docs", []) or [])
            selected_samples.append(sample_by_question[question])
            gold_doc_list = list(gold_docs_by_question[question])
            gold_answer_list = list(gold_answers_by_question[question])
            gold_docs.append(gold_doc_list)
            gold_answers.append(gold_answer_list)
            baseline_solutions.append(
                QuerySolution(
                    question=question,
                    docs=reconstructed_docs,
                    doc_scores=np.ones(len(reconstructed_docs), dtype=float),
                    gold_answers=list(example.get("gold_answers", []) or gold_answer_list),
                    gold_docs=gold_doc_list,
                )
            )

        LOGGER.info(
            "Running baseline QA from frozen report docs on %s (%d queries)",
            args.dataset,
            len(baseline_solutions),
        )
        baseline_solutions, _, _, _, baseline_metrics = hipporag.rag_qa(
            queries=baseline_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )
    else:
        docs = [f"{row['title']}\n{row['text']}" for row in corpus]
        queries = [sample["question"] for sample in samples]
        gold_docs = list(all_gold_docs)
        gold_answers = list(all_gold_answers)
        selected_samples = list(samples)

        hipporag.index(docs)

        LOGGER.info("Running baseline retrieval+QA on %s (%d queries)", args.dataset, len(queries))
        baseline_solutions, _, _, _, baseline_metrics = hipporag.rag_qa(
            queries=queries,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )

    variant_payload = build_variant_inputs(
        samples=selected_samples,
        baseline_solutions=baseline_solutions,
        gold_docs=gold_docs,
        pool_k=int(args.pool_k),
        random_seed=int(args.random_seed),
    )
    eligible_indices = list(variant_payload["eligible_indices"])
    eligible_gold_docs = subset_list(gold_docs, eligible_indices)
    eligible_gold_answers = subset_list(gold_answers, eligible_indices)
    baseline_subset = subset_list(baseline_solutions, eligible_indices)

    LOGGER.info(
        "Eligible queries with full support in top-%d: %d / %d",
        int(args.pool_k),
        len(eligible_indices),
        len(baseline_solutions),
    )

    if not eligible_indices:
        report = {
            "dataset": args.dataset,
            "num_queries": len(baseline_solutions),
            "pool_k": int(args.pool_k),
            "qa_top_k": int(args.qa_top_k),
            "eligible_count": 0,
            "eligible_rate": 0.0,
            "baseline_pipeline_overall": baseline_metrics,
            "conditions": {},
            "pairwise_vs_baseline": {},
            "query_records": [],
            "config": {
                "llm_name": args.llm_name,
                "llm_request_name": args.llm_request_name,
                "llm_base_url": args.llm_base_url,
                "embedding_name": args.embedding_name,
                "embedding_base_url": args.embedding_base_url,
                "causal_engine_version": args.causal_engine_version,
                "causal_v2_base_retrieval_mode": args.causal_v2_base_retrieval_mode,
                "structure_rerank_enabled": args.structure_rerank_enabled,
            },
            "source_report": str(args.report_json or ""),
        }
        output_json = Path(args.output_json) if args.output_json else (
            ROOT_DIR / "outputs_smoke" / f"{args.dataset}_oracle_subset_order_study.json"
        )
        output_md = Path(args.output_md) if args.output_md else (
            ROOT_DIR / "outputs_smoke" / f"{args.dataset}_oracle_subset_order_study.md"
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        output_md.write_text(
            "\n".join(
                [
                    f"# Oracle Subset Order Study: {args.dataset}",
                    "",
                    f"- Queries: `{len(baseline_solutions)}`",
                    f"- Eligible (full support in top-{int(args.pool_k)}): `0 / {len(baseline_solutions)}`",
                    "",
                    "No comparable queries in this slice.",
                ]
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output_json": str(output_json),
                    "output_md": str(output_md),
                    "eligible_count": 0,
                    "num_queries": len(baseline_solutions),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    baseline_summary = summarize_condition(
        config=config,
        gold_docs=eligible_gold_docs,
        gold_answers=eligible_gold_answers,
        query_solutions=baseline_subset,
    )

    condition_runs: Dict[str, List[QuerySolution]] = {}
    condition_summaries: Dict[str, Dict[str, Any]] = {
        "baseline_top5": {
            "overall": baseline_summary["overall"],
            "per_query_scores": baseline_summary["per_query_scores"],
        }
    }

    for condition_name, variant_queries in variant_payload["variant_inputs"].items():
        LOGGER.info("Running QA for %s on %d eligible queries", condition_name, len(variant_queries))
        run_solutions, _, _, _, _ = hipporag.rag_qa(
            queries=variant_queries,
            gold_docs=eligible_gold_docs,
            gold_answers=eligible_gold_answers,
        )
        condition_runs[condition_name] = run_solutions
        condition_summaries[condition_name] = summarize_condition(
            config=config,
            gold_docs=eligible_gold_docs,
            gold_answers=eligible_gold_answers,
            query_solutions=run_solutions,
        )

    pairwise_vs_baseline = {
        condition_name: compare_against_baseline(
            baseline_scores=baseline_summary["per_query_scores"],
            condition_scores=condition_summaries[condition_name]["per_query_scores"],
        )
        for condition_name in variant_payload["variant_inputs"].keys()
    }

    query_records = attach_query_results(
        query_records=variant_payload["query_records"],
        eligible_indices=eligible_indices,
        baseline_subset=baseline_subset,
        baseline_scores=baseline_summary["per_query_scores"],
        condition_runs=condition_runs,
        condition_summaries=condition_summaries,
    )

    random_equals_chain_count = sum(1 for row in query_records if row.get("random_equals_chain"))
    report = {
        "dataset": args.dataset,
        "num_queries": len(baseline_solutions),
        "pool_k": int(args.pool_k),
        "qa_top_k": int(args.qa_top_k),
        "eligible_count": len(eligible_indices),
        "eligible_rate": (len(eligible_indices) / len(baseline_solutions)) if baseline_solutions else 0.0,
        "random_equals_chain_count": int(random_equals_chain_count),
        "random_equals_chain_rate": (random_equals_chain_count / len(query_records)) if query_records else 0.0,
        "baseline_pipeline_overall": baseline_metrics,
        "conditions": {
            name: {"overall": summary["overall"]}
            for name, summary in condition_summaries.items()
        },
        "pairwise_vs_baseline": pairwise_vs_baseline,
        "query_records": query_records,
        "config": {
            "llm_name": args.llm_name,
            "llm_request_name": args.llm_request_name,
            "llm_base_url": args.llm_base_url,
            "embedding_name": args.embedding_name,
            "embedding_base_url": args.embedding_base_url,
            "causal_engine_version": args.causal_engine_version,
            "causal_v2_base_retrieval_mode": args.causal_v2_base_retrieval_mode,
            "structure_rerank_enabled": args.structure_rerank_enabled,
        },
        "source_report": str(args.report_json or ""),
    }

    output_json = Path(args.output_json) if args.output_json else (
        ROOT_DIR / "outputs_smoke" / f"{args.dataset}_oracle_subset_order_study.json"
    )
    output_md = Path(args.output_md) if args.output_md else (
        ROOT_DIR / "outputs_smoke" / f"{args.dataset}_oracle_subset_order_study.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")

    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(output_md),
        "eligible_count": len(eligible_indices),
        "num_queries": len(baseline_solutions),
        "baseline_overall": condition_summaries["baseline_top5"]["overall"],
        "oracle_subset_chain_overall": condition_summaries["oracle_subset_chain_order"]["overall"],
        "pairwise_vs_baseline": pairwise_vs_baseline,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
