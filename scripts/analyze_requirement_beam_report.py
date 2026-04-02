#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_requirement_cache import build_canonical_args, load_dataset, resolve_save_dir
from eval_causal_qwen3 import build_config, build_doc_text_to_chunk_id, get_gold_docs
from requirement_beam_utils import align_requirement_cache_entry_to_pool, load_requirement_cache, resolve_requirement_cache_entry
from src.hipporag.HippoRAG import HippoRAG


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def rounded_dict(payload: Dict[str, Any], digits: int = 4) -> Dict[str, Any]:
    rounded: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            rounded[key] = round(value, digits)
        else:
            rounded[key] = value
    return rounded


def score_requirement_coverage_for_positions(scores: Sequence[float]) -> float:
    residual = 1.0
    for score in scores:
        residual *= max(0.0, 1.0 - float(score))
    return float(1.0 - residual)


def smooth_min(values: Sequence[float], tau: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    clipped_tau = max(float(tau), 1e-6)
    max_neg = max(-float(value) / clipped_tau for value in values)
    mean_exp = sum(math.exp((-float(value) / clipped_tau) - max_neg) for value in values) / float(len(values))
    return float(-clipped_tau * (math.log(max(mean_exp, 1e-12)) + max_neg))


def soft_worst_case(values: Sequence[float], tau: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    clipped_tau = max(float(tau), 1e-6)
    max_value = max(float(value) for value in values)
    mean_exp = sum(math.exp((float(value) - max_value) / clipped_tau) for value in values) / float(len(values))
    return float(max_value + clipped_tau * math.log(max(mean_exp, 1e-12)))


def pairwise_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    if not positive_scores or not negative_scores:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positive_scores:
        for neg in negative_scores:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return safe_div(wins, total)


def threshold_rates(scores: Sequence[float], thresholds: Sequence[float]) -> Dict[str, float]:
    return {
        f"rate_ge_{str(threshold).replace('.', '_')}": safe_div(sum(score >= threshold for score in scores), len(scores))
        for threshold in thresholds
    }


def suspicious_anchor_rate(cache_entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    wh_tokens = {"when", "where", "who", "what", "which", "how", "whom", "why"}
    anchor_texts: List[str] = []
    suspicious = 0
    for entry in cache_entries:
        for requirement in entry.get("positive_requirements", []):
            if str(requirement.get("type", "")) != "anchor":
                continue
            text = str(requirement.get("normalized_text", "")).strip().lower()
            if not text:
                continue
            anchor_texts.append(text)
            if text in wh_tokens:
                suspicious += 1
    return rounded_dict({
        "anchor_requirement_count": len(anchor_texts),
        "suspicious_wh_anchor_count": suspicious,
        "suspicious_wh_anchor_rate": safe_div(suspicious, len(anchor_texts)),
    })


def build_runtime_args(report: Dict[str, Any], dataset: str, save_dir_root: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        save_dir=resolve_save_dir(save_dir_root, dataset),
        llm_base_url=report.get("llm_base_url", "http://localhost:8039/v1"),
        llm_name=report.get("llm_name", "qwen3-8b"),
        embedding_base_url=report.get("embedding_base_url", "http://localhost:8018/v1/embeddings"),
        embedding_name=report.get("embedding_name", "VLLM//mnt/nvme/Qwen3-Embedding-8B"),
        force_index_from_scratch="false",
        force_openie_from_scratch="false",
        max_retry_attempts=5,
        openie_mode="online",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=int(((report.get("config") or {}).get("qa_top_k", 5) or 5)),
        max_qa_steps=3,
        embedding_batch_size=8,
    )


def build_runtime_context(report: Dict[str, Any], dataset: str, limit: int, save_dir_root: str) -> Dict[str, Any]:
    runtime_args = build_runtime_args(report=report, dataset=dataset, save_dir_root=save_dir_root)
    corpus, samples = load_dataset(dataset, limit)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_docs = get_gold_docs(samples, dataset, corpus=corpus)
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)

    config_args = build_canonical_args(runtime_args, runtime_args.save_dir)
    config = build_config(config_args, corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    query_solutions, _ = hipporag.retrieve(
        queries=queries,
        gold_docs=gold_docs,
    )
    chunk_text_to_hash = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}

    runtime_by_question: Dict[str, Dict[str, Any]] = {}
    for q_idx, query_solution in enumerate(query_solutions):
        pool_docs = list(query_solution.docs)
        pool_titles = [str(doc_text).split("\n", 1)[0].strip() for doc_text in pool_docs]
        pool_doc_ids: List[int | None] = []
        for doc_text in pool_docs:
            chunk_id = doc_text_to_chunk_id.get(doc_text) or chunk_text_to_hash.get(doc_text)
            mapped_doc_id = hipporag.passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
            pool_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)
        pool_doc_entities = [
            hipporag.doc_idx_to_structure_entities.get(int(doc_id), set()) if doc_id is not None else set()
            for doc_id in pool_doc_ids
        ]
        runtime_by_question[query_solution.question] = {
            "query_index": q_idx,
            "query_solution": query_solution,
            "pool_docs": pool_docs,
            "pool_titles": pool_titles,
            "pool_doc_ids": pool_doc_ids,
            "pool_doc_entities": pool_doc_entities,
            "gold_docs": list(gold_docs[q_idx]),
        }
    return {
        "corpus": corpus,
        "samples": samples,
        "query_solutions": query_solutions,
        "runtime_by_question": runtime_by_question,
        "config": config,
    }


def compute_state_detail(cache_entry: Dict[str, Any],
                         selected_positions: Sequence[int],
                         smooth_tau: float,
                         counterfactual_tau: float) -> Dict[str, Any]:
    annotations_by_position = {
        int(annotation["pool_position"]): annotation
        for annotation in cache_entry.get("doc_annotations", [])
    }
    unique_positions = sorted({int(pos) for pos in selected_positions if int(pos) >= 0})

    positive_requirement_coverages: Dict[str, float] = {}
    group_values: Dict[str, List[float]] = {}
    for requirement in cache_entry.get("positive_requirements", []):
        requirement_id = str(requirement.get("requirement_id", ""))
        requirement_type = str(requirement.get("type", "bridge"))
        scores = [
            float(
                annotations_by_position[pos]
                .get("positive_requirement_scores", {})
                .get(requirement_id, 0.0)
            )
            for pos in unique_positions
            if pos in annotations_by_position
        ]
        coverage = score_requirement_coverage_for_positions(scores)
        positive_requirement_coverages[requirement_id] = float(coverage)
        group_values.setdefault(requirement_type, []).append(float(coverage))

    group_supports = {
        group_name: float(mean(values))
        for group_name, values in group_values.items()
    }
    support_completeness = smooth_min(list(group_supports.values()), tau=smooth_tau) if group_supports else 0.0

    counterfactual_set_coverages: Dict[str, float] = {}
    counterfactual_requirement_coverages: Dict[str, Dict[str, float]] = {}
    requirement_ranges: List[float] = []
    for cf_set in cache_entry.get("counterfactual_sets", []):
        cf_id = str(cf_set.get("cf_id", ""))
        requirement_scores: Dict[str, float] = {}
        for requirement in cf_set.get("requirements", []):
            requirement_id = str(requirement.get("requirement_id", ""))
            scores = [
                float(
                    annotations_by_position[pos]
                    .get("counterfactual_requirement_scores", {})
                    .get(cf_id, {})
                    .get(requirement_id, 0.0)
                )
                for pos in unique_positions
                if pos in annotations_by_position
            ]
            requirement_scores[requirement_id] = score_requirement_coverage_for_positions(scores)
        values = list(requirement_scores.values())
        counterfactual_requirement_coverages[cf_id] = requirement_scores
        counterfactual_set_coverages[cf_id] = float(mean(values)) if values else 0.0
        if values:
            requirement_ranges.append(float(max(values) - min(values)))

    cf_values = list(counterfactual_set_coverages.values())
    counterfactual_leakage = soft_worst_case(cf_values, tau=counterfactual_tau) if cf_values else 0.0
    utility_margin = float(support_completeness - counterfactual_leakage)
    utopia_distance = float(math.sqrt((1.0 - support_completeness) ** 2 + counterfactual_leakage ** 2))
    max_cf = float(max(cf_values)) if cf_values else 0.0
    mean_cf = float(mean(cf_values)) if cf_values else 0.0
    top2_cf_mean = float(mean(sorted(cf_values, reverse=True)[:2])) if cf_values else 0.0

    return {
        "selected_positions": unique_positions,
        "positive_requirement_coverages": positive_requirement_coverages,
        "group_supports": group_supports,
        "support_completeness": float(support_completeness),
        "counterfactual_set_coverages": counterfactual_set_coverages,
        "counterfactual_requirement_coverages": counterfactual_requirement_coverages,
        "counterfactual_leakage": float(counterfactual_leakage),
        "utility_margin": float(utility_margin),
        "utopia_distance": float(utopia_distance),
        "counterfactual_max": max_cf,
        "counterfactual_mean": mean_cf,
        "counterfactual_top2_mean": top2_cf_mean,
        "counterfactual_query_range": float(max_cf - min(cf_values)) if cf_values else 0.0,
        "counterfactual_query_std": float(std(cf_values)) if cf_values else 0.0,
        "negative_requirement_range_mean": float(mean(requirement_ranges)) if requirement_ranges else 0.0,
    }


def build_selected_doc_metrics(selected_positions: Sequence[int],
                               pool_docs: Sequence[str],
                               gold_docs: Sequence[str]) -> Dict[str, Any]:
    selected_docs = [
        str(pool_docs[pos])
        for pos in selected_positions
        if 0 <= int(pos) < len(pool_docs)
    ]
    selected_set = set(selected_docs)
    gold_set = set(gold_docs)
    gold_found = len(selected_set & gold_set)
    return {
        "selected_docs": selected_docs,
        "gold_found_count": int(gold_found),
        "gold_doc_count": int(len(gold_set)),
        "gold_recall": safe_div(gold_found, len(gold_set)),
        "full_support": bool(gold_set.issubset(selected_set)),
    }


def aggregator_scores(detail: Dict[str, Any]) -> Dict[str, float]:
    support = float(detail["support_completeness"])
    leakage = float(detail["counterfactual_leakage"])
    return {
        "current_utopia": -float(detail["utopia_distance"]),
        "support_only": support,
        "margin_softworst": float(detail["utility_margin"]),
        "margin_maxcf": support - float(detail["counterfactual_max"]),
        "margin_meancf": support - float(detail["counterfactual_mean"]),
        "margin_top2cfmean": support - float(detail["counterfactual_top2_mean"]),
        "leak_only": -leakage,
    }


def select_best_finalist(finalists: Sequence[Dict[str, Any]], aggregator_name: str) -> Dict[str, Any]:
    return max(
        finalists,
        key=lambda finalist: (
            float(finalist["aggregator_scores"][aggregator_name]),
            float(finalist["detail"]["support_completeness"]),
            -float(finalist["detail"]["counterfactual_leakage"]),
            -float(finalist["doc_metrics"]["gold_recall"]),
            tuple(int(pos) for pos in finalist["selected_positions"]),
        ),
    )


def build_report_analysis(report: Dict[str, Any],
                          cache_payload: Dict[str, Any],
                          runtime_context: Dict[str, Any],
                          report_label: str) -> Dict[str, Any]:
    qa_block = report.get("setwise_selector_qa", {}) or {}
    summary_block = qa_block.get("selector_summary", {}) or {}
    traces = report.get("setwise_selector_query_traces", []) or []
    runtime_by_question = runtime_context["runtime_by_question"]
    smooth_tau = float(qa_block.get("setwise_requirement_smooth_tau", 0.15) or 0.15)
    counterfactual_tau = float(qa_block.get("setwise_requirement_counterfactual_tau", 0.10) or 0.10)
    pool_k = int(qa_block.get("pool_k", 100) or 100)

    query_rows: List[Dict[str, Any]] = []
    all_positive_scores: List[float] = []
    all_negative_scores: List[float] = []
    selected_positive_scores: List[float] = []
    selected_negative_scores: List[float] = []
    cf_query_ranges: List[float] = []
    cf_query_stds: List[float] = []
    neg_requirement_ranges: List[float] = []
    finalist_support_ranges: List[float] = []
    finalist_leak_ranges: List[float] = []
    alignment_mismatch_count = 0
    aggregator_names = list(aggregator_scores({
        "support_completeness": 0.0,
        "counterfactual_leakage": 0.0,
        "utility_margin": 0.0,
        "utopia_distance": 0.0,
        "counterfactual_max": 0.0,
        "counterfactual_mean": 0.0,
        "counterfactual_top2_mean": 0.0,
    }).keys())
    aggregator_eval: Dict[str, Dict[str, float]] = {
        name: {
            "query_count": 0.0,
            "gold_recall_sum": 0.0,
            "full_support_sum": 0.0,
            "support_sum": 0.0,
            "leakage_sum": 0.0,
            "picked_non_current_count": 0.0,
        }
        for name in aggregator_names
    }

    cache_entries: List[Dict[str, Any]] = []
    for trace in traces:
        question = str(trace["question"])
        runtime_row = runtime_by_question.get(question)
        if runtime_row is None:
            continue
        cache_entry = resolve_requirement_cache_entry(cache_payload, question)
        aligned_entry = align_requirement_cache_entry_to_pool(
            cache_entry=cache_entry,
            pool_titles=runtime_row["pool_titles"][:pool_k],
            pool_docs=runtime_row["pool_docs"][:pool_k],
            pool_doc_entities=runtime_row["pool_doc_entities"][:pool_k],
        )
        cache_entries.append(aligned_entry)

        current_selected_positions = list((trace.get("selector_trace") or {}).get("selected_pool_positions", []))
        current_selected_titles = list(trace.get("selector_top_titles") or [])
        runtime_selected_titles = [
            runtime_row["pool_titles"][pos]
            for pos in current_selected_positions
            if 0 <= int(pos) < len(runtime_row["pool_titles"])
        ]
        if runtime_selected_titles[:len(current_selected_titles)] != current_selected_titles[:len(runtime_selected_titles)]:
            alignment_mismatch_count += 1

        finalists: List[Dict[str, Any]] = []
        finalist_rows = list((trace.get("selector_trace") or {}).get("beam_finalists") or [])
        for finalist_index, finalist in enumerate(finalist_rows):
            positions = [int(pos) for pos in finalist.get("selected_positions", [])]
            detail = compute_state_detail(
                cache_entry=aligned_entry,
                selected_positions=positions,
                smooth_tau=smooth_tau,
                counterfactual_tau=counterfactual_tau,
            )
            doc_metrics = build_selected_doc_metrics(
                selected_positions=positions,
                pool_docs=runtime_row["pool_docs"],
                gold_docs=runtime_row["gold_docs"],
            )
            finalists.append({
                "index": finalist_index,
                "selected_positions": positions,
                "selected_titles": list(finalist.get("selected_titles", [])),
                "detail": detail,
                "doc_metrics": doc_metrics,
                "aggregator_scores": aggregator_scores(detail),
            })

        if finalists:
            support_values = [float(finalist["detail"]["support_completeness"]) for finalist in finalists]
            leakage_values = [float(finalist["detail"]["counterfactual_leakage"]) for finalist in finalists]
            finalist_support_ranges.append(float(max(support_values) - min(support_values)))
            finalist_leak_ranges.append(float(max(leakage_values) - min(leakage_values)))
            current_finalist = finalists[0]
        else:
            current_finalist = {
                "index": -1,
                "selected_positions": current_selected_positions,
                "selected_titles": current_selected_titles,
                "detail": compute_state_detail(
                    cache_entry=aligned_entry,
                    selected_positions=current_selected_positions,
                    smooth_tau=smooth_tau,
                    counterfactual_tau=counterfactual_tau,
                ),
                "doc_metrics": build_selected_doc_metrics(
                    selected_positions=current_selected_positions,
                    pool_docs=runtime_row["pool_docs"],
                    gold_docs=runtime_row["gold_docs"],
                ),
                "aggregator_scores": {},
            }

        detail = current_finalist["detail"]
        cf_query_ranges.append(float(detail["counterfactual_query_range"]))
        cf_query_stds.append(float(detail["counterfactual_query_std"]))
        neg_requirement_ranges.append(float(detail["negative_requirement_range_mean"]))

        for annotation in aligned_entry.get("doc_annotations", []):
            all_positive_scores.extend(float(score) for score in (annotation.get("positive_requirement_scores") or {}).values())
            for cf_scores in (annotation.get("counterfactual_requirement_scores") or {}).values():
                all_negative_scores.extend(float(score) for score in (cf_scores or {}).values())

        selected_annotation_positions = {
            int(pos)
            for pos in current_finalist["selected_positions"]
            if int(pos) >= 0
        }
        for annotation in aligned_entry.get("doc_annotations", []):
            if int(annotation.get("pool_position", -1)) not in selected_annotation_positions:
                continue
            selected_positive_scores.extend(float(score) for score in (annotation.get("positive_requirement_scores") or {}).values())
            for cf_scores in (annotation.get("counterfactual_requirement_scores") or {}).values():
                selected_negative_scores.extend(float(score) for score in (cf_scores or {}).values())

        for aggregator_name in aggregator_names:
            if finalists:
                picked = select_best_finalist(finalists, aggregator_name)
            else:
                picked = current_finalist
            aggregator_eval[aggregator_name]["query_count"] += 1.0
            aggregator_eval[aggregator_name]["gold_recall_sum"] += float(picked["doc_metrics"]["gold_recall"])
            aggregator_eval[aggregator_name]["full_support_sum"] += float(1.0 if picked["doc_metrics"]["full_support"] else 0.0)
            aggregator_eval[aggregator_name]["support_sum"] += float(picked["detail"]["support_completeness"])
            aggregator_eval[aggregator_name]["leakage_sum"] += float(picked["detail"]["counterfactual_leakage"])
            if picked["index"] != 0:
                aggregator_eval[aggregator_name]["picked_non_current_count"] += 1.0

        query_rows.append({
            "question": question,
            "gold_doc_count": int(trace.get("gold_doc_count", 0) or 0),
            "selector_em": float((trace.get("selector_metrics") or {}).get("ExactMatch", 0.0) or 0.0),
            "selector_f1": float((trace.get("selector_metrics") or {}).get("F1", 0.0) or 0.0),
            "current_support": float(detail["support_completeness"]),
            "current_leakage": float(detail["counterfactual_leakage"]),
            "current_cf_query_range": float(detail["counterfactual_query_range"]),
            "current_negative_requirement_range_mean": float(detail["negative_requirement_range_mean"]),
            "current_gold_recall": float(current_finalist["doc_metrics"]["gold_recall"]),
            "current_full_support": bool(current_finalist["doc_metrics"]["full_support"]),
            "baseline_titles": list(trace.get("baseline_top_titles") or []),
            "selector_titles": list(trace.get("selector_top_titles") or []),
            "current_selected_positions": list(current_finalist["selected_positions"]),
            "current_counterfactual_set_coverages": detail["counterfactual_set_coverages"],
            "current_group_supports": detail["group_supports"],
            "finalist_count": len(finalists),
        })

    overlap_cache_stats = suspicious_anchor_rate(cache_entries)
    all_positive_scores = [float(score) for score in all_positive_scores]
    all_negative_scores = [float(score) for score in all_negative_scores]
    selected_positive_scores = [float(score) for score in selected_positive_scores]
    selected_negative_scores = [float(score) for score in selected_negative_scores]

    aggregator_summary = {
        name: rounded_dict({
            "query_count": values["query_count"],
            "avg_gold_recall": safe_div(values["gold_recall_sum"], values["query_count"]),
            "full_support_rate": safe_div(values["full_support_sum"], values["query_count"]),
            "avg_support": safe_div(values["support_sum"], values["query_count"]),
            "avg_leakage": safe_div(values["leakage_sum"], values["query_count"]),
            "picked_non_current_rate": safe_div(values["picked_non_current_count"], values["query_count"]),
        })
        for name, values in aggregator_eval.items()
    }

    return {
        "label": report_label,
        "report_path": report_label,
        "report_summary": rounded_dict({
            "selector_EM": float(qa_block.get("selector_EM", 0.0) or 0.0),
            "selector_F1": float(qa_block.get("selector_F1", 0.0) or 0.0),
            "EM_delta": float(qa_block.get("EM_delta", 0.0) or 0.0),
            "F1_delta": float(qa_block.get("F1_delta", 0.0) or 0.0),
            "Recall@5": float((qa_block.get("selector_retrieval_metrics") or {}).get("Recall@5", 0.0) or 0.0),
            "avg_support": float(summary_block.get("avg_requirement_support_completeness", 0.0) or 0.0),
            "avg_leakage": float(summary_block.get("avg_requirement_counterfactual_leakage", 0.0) or 0.0),
            "avg_frontier_size": float(summary_block.get("avg_requirement_frontier_size", 0.0) or 0.0),
            "avg_reserved_count": float(summary_block.get("avg_requirement_runtime_reserved_count", 0.0) or 0.0),
        }),
        "bucket_summary": qa_block.get("per_bucket", {}) or {},
        "trace_summary": rounded_dict({
            "query_count": len(query_rows),
            "alignment_mismatch_count": alignment_mismatch_count,
            "changed_from_baseline_rate": safe_div(sum(
                1 for trace in traces if bool(trace.get("changed_from_baseline", False))
            ), len(traces)),
            "avg_finalist_count": mean([row["finalist_count"] for row in query_rows]),
            "avg_finalist_support_range": mean(finalist_support_ranges),
            "median_finalist_support_range": median(finalist_support_ranges),
            "avg_finalist_leakage_range": mean(finalist_leak_ranges),
            "median_finalist_leakage_range": median(finalist_leak_ranges),
            "avg_cf_query_range": mean(cf_query_ranges),
            "median_cf_query_range": median(cf_query_ranges),
            "avg_cf_query_std": mean(cf_query_stds),
            "avg_negative_requirement_range": mean(neg_requirement_ranges),
        }),
        "score_overlap": {
            "cache_all_docs": rounded_dict({
                "positive_mean": mean(all_positive_scores),
                "positive_std": std(all_positive_scores),
                "positive_p90": percentile(all_positive_scores, 90),
                "negative_mean": mean(all_negative_scores),
                "negative_std": std(all_negative_scores),
                "negative_p90": percentile(all_negative_scores, 90),
                "separation_auc": pairwise_auc(all_positive_scores, all_negative_scores),
                **threshold_rates(all_positive_scores, [0.5, 0.8, 0.95]),
                **{f"negative_{k}": v for k, v in threshold_rates(all_negative_scores, [0.5, 0.8, 0.95]).items()},
            }),
            "selected_docs_only": rounded_dict({
                "positive_mean": mean(selected_positive_scores),
                "positive_std": std(selected_positive_scores),
                "negative_mean": mean(selected_negative_scores),
                "negative_std": std(selected_negative_scores),
                "separation_auc": pairwise_auc(selected_positive_scores, selected_negative_scores),
                **threshold_rates(selected_positive_scores, [0.5, 0.8, 0.95]),
                **{f"negative_{k}": v for k, v in threshold_rates(selected_negative_scores, [0.5, 0.8, 0.95]).items()},
            }),
            "requirement_parser": overlap_cache_stats,
        },
        "aggregator_ablation": aggregator_summary,
        "query_rows": query_rows,
    }


def build_pairwise_comparison(analysis_a: Dict[str, Any], analysis_b: Dict[str, Any]) -> Dict[str, Any]:
    rows_a = {row["question"]: row for row in analysis_a["query_rows"]}
    rows_b = {row["question"]: row for row in analysis_b["query_rows"]}
    shared_questions = [question for question in rows_a if question in rows_b]

    support_deltas: List[float] = []
    leakage_deltas: List[float] = []
    f1_deltas: List[float] = []
    gold_recall_deltas: List[float] = []
    bucket_summary: Dict[int, Dict[str, float]] = {}
    regression_rows: List[Dict[str, Any]] = []

    for question in shared_questions:
        row_a = rows_a[question]
        row_b = rows_b[question]
        support_delta = float(row_b["current_support"] - row_a["current_support"])
        leakage_delta = float(row_b["current_leakage"] - row_a["current_leakage"])
        f1_delta = float(row_b["selector_f1"] - row_a["selector_f1"])
        gold_recall_delta = float(row_b["current_gold_recall"] - row_a["current_gold_recall"])
        support_deltas.append(support_delta)
        leakage_deltas.append(leakage_delta)
        f1_deltas.append(f1_delta)
        gold_recall_deltas.append(gold_recall_delta)

        bucket = int(row_a["gold_doc_count"])
        bucket_summary.setdefault(bucket, {
            "count": 0.0,
            "support_delta_sum": 0.0,
            "leakage_delta_sum": 0.0,
            "f1_delta_sum": 0.0,
            "gold_recall_delta_sum": 0.0,
            "worse_count": 0.0,
            "better_count": 0.0,
        })
        bucket_summary[bucket]["count"] += 1.0
        bucket_summary[bucket]["support_delta_sum"] += support_delta
        bucket_summary[bucket]["leakage_delta_sum"] += leakage_delta
        bucket_summary[bucket]["f1_delta_sum"] += f1_delta
        bucket_summary[bucket]["gold_recall_delta_sum"] += gold_recall_delta
        if f1_delta < -1e-9:
            bucket_summary[bucket]["worse_count"] += 1.0
        elif f1_delta > 1e-9:
            bucket_summary[bucket]["better_count"] += 1.0

        if f1_delta < -1e-9:
            regression_rows.append({
                "question": question,
                "gold_doc_count": bucket,
                "f1_delta": f1_delta,
                "support_delta": support_delta,
                "leakage_delta": leakage_delta,
                "gold_recall_delta": gold_recall_delta,
                "titles_a": row_a["selector_titles"],
                "titles_b": row_b["selector_titles"],
            })

    def corr(xs: Sequence[float], ys: Sequence[float]) -> float:
        if not xs or len(xs) != len(ys):
            return 0.0
        mean_x = mean(xs)
        mean_y = mean(ys)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denom_x = sum((x - mean_x) ** 2 for x in xs)
        denom_y = sum((y - mean_y) ** 2 for y in ys)
        if denom_x <= 0.0 or denom_y <= 0.0:
            return 0.0
        return float(numerator / math.sqrt(denom_x * denom_y))

    return {
        "pair_labels": [analysis_a["label"], analysis_b["label"]],
        "summary": rounded_dict({
            "query_count": len(shared_questions),
            "support_delta_mean": mean(support_deltas),
            "support_delta_std": std(support_deltas),
            "leakage_delta_mean": mean(leakage_deltas),
            "f1_delta_mean": mean(f1_deltas),
            "gold_recall_delta_mean": mean(gold_recall_deltas),
            "corr_support_delta_vs_f1_delta": corr(support_deltas, f1_deltas),
            "corr_leakage_delta_vs_f1_delta": corr(leakage_deltas, f1_deltas),
            "support_up_count": sum(delta > 1e-9 for delta in support_deltas),
            "support_down_count": sum(delta < -1e-9 for delta in support_deltas),
            "f1_worse_count": sum(delta < -1e-9 for delta in f1_deltas),
            "f1_better_count": sum(delta > 1e-9 for delta in f1_deltas),
        }),
        "bucket_summary": {
            f"{bucket}_doc": rounded_dict({
                "count": values["count"],
                "support_delta_mean": safe_div(values["support_delta_sum"], values["count"]),
                "leakage_delta_mean": safe_div(values["leakage_delta_sum"], values["count"]),
                "f1_delta_mean": safe_div(values["f1_delta_sum"], values["count"]),
                "gold_recall_delta_mean": safe_div(values["gold_recall_delta_sum"], values["count"]),
                "worse_rate": safe_div(values["worse_count"], values["count"]),
                "better_rate": safe_div(values["better_count"], values["count"]),
            })
            for bucket, values in sorted(bucket_summary.items())
        },
        "top_regressions": sorted(regression_rows, key=lambda row: row["f1_delta"])[:10],
    }


def build_markdown_report(payload: Dict[str, Any]) -> str:
    analyses = payload["analyses"]
    lines = ["# Requirement Beam Diagnostics", ""]
    lines.append("## Raw Data Table")
    lines.append("")
    lines.append("| Label | EM | ΔEM | F1 | ΔF1 | R@5 | Avg Support | Avg Leakage | Avg Finalist Leak Range |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for analysis in analyses:
        summary = analysis["report_summary"]
        trace_summary = analysis["trace_summary"]
        lines.append(
            f"| {analysis['label']} | {summary['selector_EM']:.4f} | {summary['EM_delta']:+.4f} | "
            f"{summary['selector_F1']:.4f} | {summary['F1_delta']:+.4f} | {summary['Recall@5']:.4f} | "
            f"{summary['avg_support']:.4f} | {summary['avg_leakage']:.4f} | {trace_summary['avg_finalist_leakage_range']:.4f} |"
        )
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    for idx, analysis in enumerate(analyses, start=1):
        overlap = analysis["score_overlap"]["cache_all_docs"]
        trace_summary = analysis["trace_summary"]
        lines.append(
            f"{idx}. `{analysis['label']}`: avg support={analysis['report_summary']['avg_support']:.4f}, "
            f"avg leakage={analysis['report_summary']['avg_leakage']:.4f}, finalist leak range={trace_summary['avg_finalist_leakage_range']:.4f}, "
            f"positive-vs-negative separation AUC={overlap['separation_auc']:.4f}."
        )
    if payload.get("pairwise"):
        pairwise = payload["pairwise"]["summary"]
        lines.append(
            f"{len(analyses) + 1}. Pairwise `{payload['pairwise']['pair_labels'][0]}` -> `{payload['pairwise']['pair_labels'][1]}`: "
            f"support mean delta={pairwise['support_delta_mean']:+.4f}, F1 mean delta={pairwise['f1_delta_mean']:+.4f}, "
            f"gold-recall mean delta={pairwise['gold_recall_delta_mean']:+.4f}, corr(supportΔ,F1Δ)={pairwise['corr_support_delta_vs_f1_delta']:+.4f}."
        )
    lines.append("")

    lines.append("## Suggested Next Experiments")
    lines.append("")
    lines.append("- If finalist leak range stays tiny while cache-level positive/negative separation is healthy, change the leakage aggregation before changing counterfactual generation.")
    lines.append("- If cache-level positive/negative separation is already poor, prioritize counterfactual/query-requirement construction or coverage scoring before more beam tuning.")
    lines.append("- If alternative aggregator rankings recover higher gold recall from the same finalists, test that aggregator in retrieval-only mode before paying reader cost.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze requirement_beam eval reports without rerunning QA.")
    parser.add_argument("--reports", nargs="+", required=True, help="One or two requirement_beam eval reports.")
    parser.add_argument("--cache_path", required=True, help="Requirement cache used by the reports.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. musique.")
    parser.add_argument("--limit", type=int, required=True, help="Number of samples in the report.")
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general", help="Root save_dir used to build the retrieval index.")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--output_md", type=str, default="")
    args = parser.parse_args()

    if len(args.reports) not in {1, 2}:
        raise ValueError("--reports accepts one or two report paths.")

    report_payloads = [json.loads(Path(path).read_text()) for path in args.reports]
    cache_payload = load_requirement_cache(args.cache_path)
    runtime_context = build_runtime_context(
        report=report_payloads[0],
        dataset=args.dataset,
        limit=int(args.limit),
        save_dir_root=str(args.save_dir),
    )

    analyses = []
    for report_path, report_payload in zip(args.reports, report_payloads):
        analyses.append(build_report_analysis(
            report=report_payload,
            cache_payload=cache_payload,
            runtime_context=runtime_context,
            report_label=str(report_path),
        ))

    payload: Dict[str, Any] = {
        "dataset": args.dataset,
        "limit": int(args.limit),
        "cache_path": str(args.cache_path),
        "reports": list(args.reports),
        "analyses": analyses,
    }
    if len(analyses) == 2:
        payload["pairwise"] = build_pairwise_comparison(analyses[0], analyses[1])

    markdown = build_markdown_report(payload)

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_md:
        output_md_path = Path(args.output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(markdown)

    print(json.dumps({
        "reports": list(args.reports),
        "output_json": str(args.output_json) if args.output_json else None,
        "output_md": str(args.output_md) if args.output_md else None,
        "pairwise_summary": payload.get("pairwise", {}).get("summary"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
