#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
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


def subset_list(values: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    return [values[idx] for idx in indices]


def make_query_solution(question: str,
                        docs: Sequence[str],
                        gold_doc_list: Sequence[str]) -> QuerySolution:
    return QuerySolution(
        question=question,
        docs=list(docs),
        doc_scores=np.ones(len(docs), dtype=float),
        gold_docs=list(gold_doc_list),
    )


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


def reconstruct_baseline_solutions(args: argparse.Namespace,
                                   corpus: Sequence[Dict[str, Any]],
                                   samples: Sequence[Dict[str, Any]]) -> tuple[List[QuerySolution], List[List[str]], List[List[str]], List[Dict[str, Any]], Dict[str, Any]]:
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

    report_path = Path(args.report_json)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_examples = list(report_payload.get("examples", []) or [])
    if args.limit and args.limit > 0 and len(report_examples) > int(args.limit):
        if int(args.random_subset_seed) >= 0:
            rng = np.random.default_rng(int(args.random_subset_seed))
            selected_indices = sorted(
                rng.choice(len(report_examples), size=int(args.limit), replace=False).tolist()
            )
            report_examples = [report_examples[idx] for idx in selected_indices]
        else:
            report_examples = report_examples[: int(args.limit)]

    baseline_solutions: List[QuerySolution] = []
    selected_samples: List[Dict[str, Any]] = []
    gold_docs: List[List[str]] = []
    gold_answers: List[List[str]] = []
    for example in report_examples:
        question = str(example.get("question", ""))
        if question not in sample_by_question or question not in gold_docs_by_question:
            continue
        retrieved_doc_ids = list(example.get("retrieved_doc_ids", []) or [])
        reconstructed_docs = [
            chunk_id_to_doc_text[chunk_id]
            for chunk_id in retrieved_doc_ids
            if isinstance(chunk_id, str) and chunk_id in chunk_id_to_doc_text
        ]
        if not reconstructed_docs:
            reconstructed_docs = list(example.get("docs", []) or [])
        gold_doc_list = list(gold_docs_by_question[question])
        gold_answer_list = list(gold_answers_by_question[question])
        selected_samples.append(sample_by_question[question])
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
    return baseline_solutions, gold_docs, gold_answers, selected_samples, {
        "hipporag": hipporag,
        "config": config,
        "baseline_metrics": baseline_metrics,
    }


def build_padded_curves(samples: Sequence[Dict[str, Any]],
                        baseline_solutions: Sequence[QuerySolution],
                        gold_docs: Sequence[Sequence[str]],
                        pool_k: int,
                        max_curve_size: int) -> Dict[str, Any]:
    eligible_indices: List[int] = []
    conditions = {
        "baseline_prefix": {size: [] for size in range(1, max_curve_size + 1)},
        "oracle_pool_padded": {size: [] for size in range(1, max_curve_size + 1)},
        "oracle_chain_padded": {size: [] for size in range(1, max_curve_size + 1)},
    }
    records: List[Dict[str, Any]] = []

    for q_idx, qs in enumerate(baseline_solutions):
        pool_docs = list(qs.docs[:pool_k])
        gold_doc_list = list(gold_docs[q_idx])
        gold_set = set(gold_doc_list)
        pool_set = set(pool_docs)
        full_support_in_pool = gold_set.issubset(pool_set)
        support_title_order = ordered_support_titles(samples[q_idx])
        gold_title_to_doc = {extract_doc_title(doc_text): doc_text for doc_text in gold_doc_list}
        gold_in_pool_order = [doc_text for doc_text in pool_docs if doc_text in gold_set]

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

        if not full_support_in_pool:
            records.append(
                {
                    "question": qs.question,
                    "eligible": False,
                    "baseline_top_titles": [extract_doc_title(doc_text) for doc_text in qs.docs[:max_curve_size]],
                    "gold_titles": [extract_doc_title(doc_text) for doc_text in gold_doc_list],
                    "pool_k": int(pool_k),
                }
            )
            continue

        eligible_indices.append(q_idx)
        padding_docs = [doc_text for doc_text in pool_docs if doc_text not in gold_set]
        baseline_prefix = list(qs.docs[:max_curve_size])
        oracle_pool_full = list(gold_in_pool_order) + list(padding_docs)
        oracle_chain_full = list(chain_docs) + [doc_text for doc_text in padding_docs if doc_text not in set(chain_docs)]

        condition_full_docs = {
            "baseline_prefix": baseline_prefix,
            "oracle_pool_padded": oracle_pool_full,
            "oracle_chain_padded": oracle_chain_full,
        }
        support_complete_size = {
            "baseline_prefix": next(
                (
                    size
                    for size in range(1, max_curve_size + 1)
                    if gold_set.issubset(set(baseline_prefix[:size]))
                ),
                None,
            ),
            "oracle_pool_padded": len(gold_in_pool_order),
            "oracle_chain_padded": len(chain_docs),
        }

        for condition_name, doc_list in condition_full_docs.items():
            for size in range(1, max_curve_size + 1):
                conditions[condition_name][size].append(
                    make_query_solution(
                        question=qs.question,
                        docs=doc_list[:size],
                        gold_doc_list=gold_doc_list,
                    )
                )

        records.append(
            {
                "question": qs.question,
                "eligible": True,
                "gold_doc_count": len(gold_doc_list),
                "gold_in_pool_count": len(gold_in_pool_order),
                "baseline_top_titles": [extract_doc_title(doc_text) for doc_text in baseline_prefix],
                "oracle_pool_titles_full": [extract_doc_title(doc_text) for doc_text in oracle_pool_full[:max_curve_size]],
                "oracle_chain_titles_full": [extract_doc_title(doc_text) for doc_text in oracle_chain_full[:max_curve_size]],
                "support_complete_size": support_complete_size,
            }
        )

    return {
        "eligible_indices": eligible_indices,
        "conditions": conditions,
        "query_records": records,
    }


def summarize_monotonicity(query_records: Sequence[Dict[str, Any]],
                           condition_scores_by_size: Dict[str, Dict[int, Sequence[Dict[str, float]]]],
                           eligible_indices: Sequence[int]) -> Dict[str, Any]:
    local_idx_by_global = {global_idx: local_idx for local_idx, global_idx in enumerate(eligible_indices)}
    summary: Dict[str, Any] = {}
    for condition_name, scores_by_size in condition_scores_by_size.items():
        post_support_gain = 0
        flat_after_support = 0
        support_not_reached = 0
        max_after_support_deltas: List[float] = []
        best_size_hist: Dict[int, int] = {}

        for global_idx in eligible_indices:
            row = query_records[global_idx]
            local_idx = local_idx_by_global[global_idx]
            support_size = row["support_complete_size"].get(condition_name)
            if support_size is None:
                support_not_reached += 1
                continue

            support_score = float(scores_by_size[int(support_size)][local_idx]["F1"])
            best_size = int(support_size)
            best_score = float(support_score)
            for size, per_query_scores in scores_by_size.items():
                if int(size) < int(support_size):
                    continue
                f1_value = float(per_query_scores[local_idx]["F1"])
                if f1_value > best_score + 1e-9:
                    best_score = f1_value
                    best_size = int(size)

            delta = best_score - support_score
            max_after_support_deltas.append(delta)
            best_size_hist[best_size] = best_size_hist.get(best_size, 0) + 1
            if delta > 1e-9:
                post_support_gain += 1
            else:
                flat_after_support += 1

        summary[condition_name] = {
            "support_not_reached_count": int(support_not_reached),
            "post_support_gain_count": int(post_support_gain),
            "flat_after_support_count": int(flat_after_support),
            "mean_best_minus_support_F1": round(
                float(np.mean(max_after_support_deltas)) if max_after_support_deltas else 0.0,
                4,
            ),
            "best_size_histogram": {str(size): int(count) for size, count in sorted(best_size_hist.items())},
        }
    return summary


def attach_query_curves(query_records: Sequence[Dict[str, Any]],
                        eligible_indices: Sequence[int],
                        condition_scores_by_size: Dict[str, Dict[int, Sequence[Dict[str, float]]]]) -> List[Dict[str, Any]]:
    local_idx_by_global = {global_idx: local_idx for local_idx, global_idx in enumerate(eligible_indices)}
    enriched: List[Dict[str, Any]] = []
    for global_idx, row in enumerate(query_records):
        record = dict(row)
        if not row.get("eligible"):
            enriched.append(record)
            continue
        local_idx = local_idx_by_global[global_idx]
        record["curve_metrics"] = {
            condition_name: {
                str(size): condition_scores_by_size[condition_name][size][local_idx]
                for size in sorted(condition_scores_by_size[condition_name].keys())
            }
            for condition_name in condition_scores_by_size.keys()
        }
        enriched.append(record)
    return enriched


def build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# Oracle Subset Size Curve: {report['dataset']}",
        "",
        f"- Queries: `{report['num_queries']}`",
        f"- Eligible (full support in top-{report['pool_k']}): `{report['eligible_count']}` / `{report['num_queries']}` "
        f"(`{report['eligible_rate']:.1%}`)",
        f"- Curve sizes: `{report.get('selected_curve_sizes', list(range(1, report['max_curve_size'] + 1)))}`",
        "",
        "## Overall Curves",
        "",
    ]

    for condition_name, curve in report["curves"].items():
        lines.extend([
            f"### {condition_name}",
            "",
            "| |S| | EM | F1 | support-complete rate |",
            "|---:|---:|---:|---:|---:|",
        ])
        for size in sorted(curve["sizes"].keys(), key=int):
            entry = curve["sizes"][size]
            overall = entry["overall"]
            lines.append(
                f"| {size} | {overall.get('ExactMatch', 0.0):.4f} | {overall.get('F1', 0.0):.4f} | "
                f"{entry.get('support_complete_rate', 0.0):.4f} |"
            )
        mono = report["monotonicity"][condition_name]
        lines.extend([
            "",
            f"- post-support gain count: `{mono['post_support_gain_count']}`",
            f"- flat-after-support count: `{mono['flat_after_support_count']}`",
            f"- mean best-minus-support F1: `{mono['mean_best_minus_support_F1']:+.4f}`",
            f"- best size histogram: `{mono['best_size_histogram']}`",
            "",
        ])

    lines.extend([
        "## Example Queries",
        "",
    ])
    shown = 0
    for row in report.get("query_records", []):
        if not row.get("eligible"):
            continue
        lines.append(f"- Q: {row['question']}")
        lines.append(f"  baseline top titles: {row['baseline_top_titles']}")
        lines.append(f"  oracle pool padded: {row['oracle_pool_titles_full']}")
        lines.append(f"  oracle chain padded: {row['oracle_chain_titles_full']}")
        lines.append(f"  support complete sizes: {row['support_complete_size']}")
        for condition_name in report.get("selected_conditions", list(report["curves"].keys())):
            curve = row["curve_metrics"][condition_name]
            compact = {size: round(float(metrics['F1']), 4) for size, metrics in curve.items()}
            lines.append(f"  {condition_name} F1 by |S|: {compact}")
        shown += 1
        if shown >= min(8, report["eligible_count"]):
            break
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-pool reader accuracy vs subset-size curves.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--random_subset_seed", type=int, default=-1)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--max_curve_size", type=int, default=5)
    parser.add_argument("--curve_sizes", type=str, default="")
    parser.add_argument("--conditions", type=str, default="baseline_prefix,oracle_pool_padded,oracle_chain_padded")
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general")
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

    selected_sizes = [
        int(part.strip())
        for part in str(args.curve_sizes).split(",")
        if str(part).strip()
    ]
    if not selected_sizes:
        selected_sizes = list(range(1, int(args.max_curve_size) + 1))
    selected_sizes = sorted({size for size in selected_sizes if size > 0})
    if not selected_sizes:
        raise ValueError("No valid curve sizes were selected.")

    selected_conditions = [
        part.strip()
        for part in str(args.conditions).split(",")
        if part.strip()
    ]
    if not selected_conditions:
        raise ValueError("No valid conditions were selected.")

    max_selected_size = max(selected_sizes)
    if int(args.max_curve_size) < int(max_selected_size):
        args.max_curve_size = int(max_selected_size)
    if int(args.qa_top_k) < int(max_selected_size):
        args.qa_top_k = int(max_selected_size)

    dataset_limit = 0 if args.report_json else int(args.limit)
    corpus, samples = load_dataset(args.dataset, dataset_limit)
    baseline_solutions, gold_docs, gold_answers, selected_samples, runtime = reconstruct_baseline_solutions(
        args=args,
        corpus=corpus,
        samples=samples,
    )
    hipporag = runtime["hipporag"]
    config = runtime["config"]

    curve_payload = build_padded_curves(
        samples=selected_samples,
        baseline_solutions=baseline_solutions,
        gold_docs=gold_docs,
        pool_k=int(args.pool_k),
        max_curve_size=int(args.max_curve_size),
    )
    eligible_indices = list(curve_payload["eligible_indices"])
    eligible_gold_docs = subset_list(gold_docs, eligible_indices)
    eligible_gold_answers = subset_list(gold_answers, eligible_indices)

    LOGGER.info(
        "Eligible queries with full support in top-%d: %d / %d",
        int(args.pool_k),
        len(eligible_indices),
        len(baseline_solutions),
    )

    condition_runs: Dict[str, Dict[int, List[QuerySolution]]] = {
        condition_name: {}
        for condition_name in selected_conditions
    }
    condition_scores_by_size: Dict[str, Dict[int, Sequence[Dict[str, float]]]] = {
        condition_name: {}
        for condition_name in selected_conditions
    }
    curves: Dict[str, Dict[str, Any]] = {}

    for condition_name, variants_by_size in curve_payload["conditions"].items():
        if condition_name not in selected_conditions:
            continue
        curves[condition_name] = {"sizes": {}}
        for size, variant_queries in variants_by_size.items():
            if int(size) not in selected_sizes:
                continue
            LOGGER.info("Running QA for %s size=%d on %d eligible queries", condition_name, int(size), len(variant_queries))
            run_solutions, _, _, _, _ = hipporag.rag_qa(
                queries=variant_queries,
                gold_docs=eligible_gold_docs,
                gold_answers=eligible_gold_answers,
            )
            summary = summarize_condition(
                config=config,
                gold_docs=eligible_gold_docs,
                gold_answers=eligible_gold_answers,
                query_solutions=run_solutions,
            )
            condition_runs[condition_name][int(size)] = run_solutions
            condition_scores_by_size[condition_name][int(size)] = summary["per_query_scores"]

            support_complete_count = 0
            for global_idx in eligible_indices:
                support_size = curve_payload["query_records"][global_idx]["support_complete_size"].get(condition_name)
                if support_size is not None and int(support_size) <= int(size):
                    support_complete_count += 1
            support_complete_rate = (
                support_complete_count / len(eligible_indices)
                if eligible_indices else 0.0
            )
            curves[condition_name]["sizes"][str(size)] = {
                "overall": summary["overall"],
                "support_complete_rate": round(float(support_complete_rate), 4),
                "support_complete_count": int(support_complete_count),
            }

    monotonicity = summarize_monotonicity(
        query_records=curve_payload["query_records"],
        condition_scores_by_size=condition_scores_by_size,
        eligible_indices=eligible_indices,
    )
    query_records = attach_query_curves(
        query_records=curve_payload["query_records"],
        eligible_indices=eligible_indices,
        condition_scores_by_size=condition_scores_by_size,
    )

    report = {
        "dataset": args.dataset,
        "source_report": str(args.report_json),
        "num_queries": len(baseline_solutions),
        "eligible_count": len(eligible_indices),
        "eligible_rate": (len(eligible_indices) / len(baseline_solutions)) if baseline_solutions else 0.0,
        "pool_k": int(args.pool_k),
        "max_curve_size": int(args.max_curve_size),
        "selected_curve_sizes": list(selected_sizes),
        "selected_conditions": list(selected_conditions),
        "qa_top_k": int(args.qa_top_k),
        "baseline_pipeline_overall": runtime["baseline_metrics"],
        "curves": curves,
        "monotonicity": monotonicity,
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
    }

    output_json = Path(args.output_json) if args.output_json else (
        ROOT_DIR / "outputs_smoke" / f"{args.dataset}_oracle_subset_size_curve.json"
    )
    output_md = Path(args.output_md) if args.output_md else (
        ROOT_DIR / "outputs_smoke" / f"{args.dataset}_oracle_subset_size_curve.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "eligible_count": len(eligible_indices),
                "num_queries": len(baseline_solutions),
                "monotonicity": monotonicity,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
