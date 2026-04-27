#!/usr/bin/env python3
"""Hard-negative v2 diagnostics and oracle edit opportunity analysis for D-PathRAG."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dpathrag_analyze_hard_negatives import (  # noqa: E402
    categorize_candidate,
    doc_features,
    is_gold,
    load_embedding_payload,
    load_jsonl,
    row_qid,
)
from src.dpathrag.io import write_json
from src.dpathrag.selector_data import SelectorExample, featurize_selector_record, summarize_selector_metrics, support_metrics_for_indices


RANK_BUCKETS = ("6-10", "11-20", "21-50", "51-100")
HARD_NEGATIVE_TYPES = (
    "answer_string_distractor",
    "bridge_entity_distractor",
    "lexical_hard_negative",
    "semantic_hard_negative",
    "low_signal_deep_negative",
)
FEATURE_ORDER = (
    "rank",
    "retriever_score",
    "title_question_jaccard",
    "body_question_jaccard",
    "question_token_coverage",
    "q_doc_cosine",
    "answer_in_doc",
    "bridge_entity_in_doc",
    "log_doc_chars",
)


def unique_valid_indices(indices: Sequence[Any], *, candidate_count: int, top_k: int) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for item in indices:
        index = int(item)
        if index < 0 or index >= int(candidate_count) or index in seen:
            continue
        seen.add(index)
        output.append(index)
        if len(output) >= int(top_k):
            break
    return output


def rank_bucket(rank: float | int) -> str:
    value = int(rank)
    if 6 <= value <= 10:
        return "6-10"
    if 11 <= value <= 20:
        return "11-20"
    if 21 <= value <= 50:
        return "21-50"
    if 51 <= value <= 100:
        return "51-100"
    return "outside"


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def classify_hard_negative(features: dict[str, float], *, semantic_threshold: float) -> str:
    if float(features.get("answer_in_doc") or 0.0) > 0.0:
        return "answer_string_distractor"
    if float(features.get("bridge_entity_in_doc") or 0.0) > 0.0:
        return "bridge_entity_distractor"
    if (
        float(features.get("question_token_coverage") or 0.0) >= 0.35
        or float(features.get("title_question_jaccard") or 0.0) >= 0.08
        or float(features.get("body_question_jaccard") or 0.0) >= 0.08
    ):
        return "lexical_hard_negative"
    if float(features.get("q_doc_cosine") or 0.0) >= float(semantic_threshold):
        return "semantic_hard_negative"
    return "low_signal_deep_negative"


def mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def summarize_docs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"docs": 0, "queries": 0, "features": {}}
    features: dict[str, dict[str, float]] = {}
    for key in FEATURE_ORDER:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            features[key] = {
                "mean": round(mean(values), 6),
                "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
            }
    return {
        "docs": len(rows),
        "queries": len({str(row.get("qid")) for row in rows}),
        "features": features,
    }


def group_summary(rows: Sequence[dict[str, Any]], key: str, allowed: Sequence[str] | None = None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {item: [] for item in allowed or []}
    for row in rows:
        value = str(row.get(key) or "unknown")
        grouped.setdefault(value, []).append(row)
    return {name: summarize_docs(items) for name, items in grouped.items()}


def summarize_metric_rows(rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    return summarize_selector_metrics(rows)


def set_metrics(
    record: dict[str, Any],
    indices: Sequence[int],
    *,
    max_candidates: int,
    top_k: int,
    example: SelectorExample | None = None,
) -> dict[str, float]:
    if example is None:
        example = featurize_selector_record(record, max_candidates=int(max_candidates), path_len=int(top_k))
    return support_metrics_for_indices(example, indices)


def edit_objective(metrics: dict[str, float], *, added_non_gold: int) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("support_complete") or 0.0),
        float(metrics.get("support_recall") or 0.0),
        float(metrics.get("selected_gold_count") or 0.0),
        float(metrics.get("bridge_entity_recall") or 0.0),
        -float(added_non_gold),
    )


def replace_index(indices: Sequence[int], remove_index: int, add_index: int) -> list[int]:
    return [int(add_index) if int(index) == int(remove_index) else int(index) for index in indices]


def best_oracle_edit(
    record: dict[str, Any],
    base_indices: Sequence[int],
    *,
    candidate_pool_size: int,
    max_candidates: int,
    top_k: int,
    example: SelectorExample | None = None,
) -> dict[str, Any]:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    usable_pool = min(int(candidate_pool_size), len(candidates), int(max_candidates))
    base = list(base_indices)
    base_set = set(base)
    base_metrics = set_metrics(record, base, max_candidates=max_candidates, top_k=top_k, example=example)
    base_objective = edit_objective(base_metrics, added_non_gold=0)
    best = {
        "indices": list(base),
        "remove_index": None,
        "add_index": None,
        "metrics": base_metrics,
        "objective": base_objective,
        "stopped": True,
        "added_gold": 0,
        "added_non_gold": 0,
        "removed_gold": 0,
        "removed_non_gold": 0,
    }
    for remove_index in base:
        for add_index in range(usable_pool):
            if add_index in base_set:
                continue
            edited = replace_index(base, remove_index, add_index)
            added_non_gold = 0 if is_gold(candidates[add_index]) else 1
            metrics = set_metrics(record, edited, max_candidates=max_candidates, top_k=top_k, example=example)
            objective = edit_objective(metrics, added_non_gold=added_non_gold)
            if objective > best["objective"]:
                best = {
                    "indices": edited,
                    "remove_index": int(remove_index),
                    "add_index": int(add_index),
                    "metrics": metrics,
                    "objective": objective,
                    "stopped": False,
                    "added_gold": 1 if is_gold(candidates[add_index]) else 0,
                    "added_non_gold": added_non_gold,
                    "removed_gold": 1 if is_gold(candidates[int(remove_index)]) else 0,
                    "removed_non_gold": 0 if is_gold(candidates[int(remove_index)]) else 1,
                }
    return best


def oracle_edit_sequence(
    record: dict[str, Any],
    *,
    top_k: int,
    candidate_pool_size: int,
    max_candidates: int,
    steps: int,
    example: SelectorExample | None = None,
) -> dict[str, Any]:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    indices = list(range(min(int(top_k), len(candidates))))
    edits: list[dict[str, Any]] = []
    totals = {"added_gold": 0, "added_non_gold": 0, "removed_gold": 0, "removed_non_gold": 0}
    for _ in range(int(steps)):
        edit = best_oracle_edit(
            record,
            indices,
            candidate_pool_size=int(candidate_pool_size),
            max_candidates=int(max_candidates),
            top_k=int(top_k),
            example=example,
        )
        if bool(edit["stopped"]):
            break
        indices = list(edit["indices"])
        edits.append(edit)
        for key in totals:
            totals[key] += int(edit.get(key) or 0)
    metrics = set_metrics(record, indices, max_candidates=max_candidates, top_k=top_k, example=example)
    return {
        "indices": indices,
        "metrics": metrics,
        "edits": edits,
        "stopped": len(edits) < int(steps),
        **totals,
    }


def summarize_oracle(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    metric_rows = [dict(row[key]["metrics"]) for row in rows]
    summary = summarize_metric_rows(metric_rows)
    denom = float(max(1, len(rows)))
    return {
        **summary,
        "queries_with_edit": sum(1 for row in rows if row[key]["edits"]) ,
        "stop_rate": round(sum(1 for row in rows if row[key]["stopped"]) / denom, 4),
        "added_gold": sum(int(row[key].get("added_gold") or 0) for row in rows),
        "added_non_gold": sum(int(row[key].get("added_non_gold") or 0) for row in rows),
        "removed_gold": sum(int(row[key].get("removed_gold") or 0) for row in rows),
        "removed_non_gold": sum(int(row[key].get("removed_non_gold") or 0) for row in rows),
    }


def analyze_v2(
    cache_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    max_candidates: int,
    oracle_pool_size: int,
    embedding_by_qid: dict[str, Any] | None = None,
    embedding_feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    embedding_by_qid = embedding_by_qid or {}
    embedding_feature_names = list(embedding_feature_names or [])
    cache_by_qid = {row_qid(row): row for row in cache_rows}
    doc_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    for prediction in prediction_rows:
        qid = row_qid(prediction)
        if qid not in cache_by_qid:
            raise KeyError(f"Missing cache row for qid={qid}")
        record = cache_by_qid[qid]
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        example = featurize_selector_record(record, max_candidates=int(max_candidates), path_len=int(top_k))
        rank_indices = list(range(min(int(top_k), len(candidates))))
        selector_indices = unique_valid_indices(
            prediction.get("selected_indices") or [],
            candidate_count=len(candidates),
            top_k=int(top_k),
        )
        rank_set = set(rank_indices)
        selector_set = set(selector_indices)
        rank_metrics = set_metrics(record, rank_indices, max_candidates=max_candidates, top_k=top_k, example=example)
        selector_metrics = set_metrics(record, selector_indices, max_candidates=max_candidates, top_k=top_k, example=example)
        selector_added_gold = 0
        selector_added_non_gold = 0
        for index in sorted(rank_set | selector_set):
            category = categorize_candidate(index, rank_set=rank_set, selector_set=selector_set, gold=is_gold(candidates[index]))
            if category is None:
                continue
            features = doc_features(
                record,
                candidates[index],
                candidate_index=index,
                embedding_by_qid=embedding_by_qid,
                embedding_feature_names=embedding_feature_names,
            )
            if category == "selector_added_gold":
                selector_added_gold += 1
            if category == "selector_added_non_gold":
                selector_added_non_gold += 1
            doc_rows.append(
                {
                    "qid": qid,
                    "query_idx": record.get("query_idx"),
                    "query_type": str(record.get("type") or "unknown"),
                    "category": category,
                    "candidate_index": int(index),
                    "title": str(candidates[index].get("title") or ""),
                    "rank_bucket": rank_bucket(features["rank"]),
                    **features,
                }
            )
        rank_complete = float(rank_metrics.get("support_complete") or 0.0) >= 1.0
        selector_changed = selector_set != rank_set
        support_delta = float(selector_metrics.get("support_complete") or 0.0) - float(rank_metrics.get("support_complete") or 0.0)
        query_rows.append(
            {
                "qid": qid,
                "query_idx": record.get("query_idx"),
                "query_type": str(record.get("type") or "unknown"),
                "gold_support_count": len(record.get("gold_titles") or []),
                "rank_complete": rank_complete,
                "selector_changed": selector_changed,
                "selector_added_gold": selector_added_gold,
                "selector_added_non_gold": selector_added_non_gold,
                "rank_support_complete": float(rank_metrics.get("support_complete") or 0.0),
                "selector_support_complete": float(selector_metrics.get("support_complete") or 0.0),
                "support_complete_delta": support_delta,
                "support_recall_delta": float(selector_metrics.get("support_recall") or 0.0) - float(rank_metrics.get("support_recall") or 0.0),
            }
        )
        oracle_rows.append(
            {
                "qid": qid,
                "rank": {"metrics": rank_metrics, "indices": rank_indices},
                "selector": {"metrics": selector_metrics, "indices": selector_indices},
                "oracle_edit1": oracle_edit_sequence(
                    record,
                    top_k=int(top_k),
                    candidate_pool_size=int(oracle_pool_size),
                    max_candidates=int(max_candidates),
                    steps=1,
                    example=example,
                ),
                "oracle_edit2": oracle_edit_sequence(
                    record,
                    top_k=int(top_k),
                    candidate_pool_size=int(oracle_pool_size),
                    max_candidates=int(max_candidates),
                    steps=2,
                    example=example,
                ),
            }
        )

    selector_added_non_gold = [row for row in doc_rows if row["category"] == "selector_added_non_gold"]
    semantic_threshold = percentile([float(row.get("q_doc_cosine") or 0.0) for row in selector_added_non_gold], 0.75)
    for row in doc_rows:
        if row["category"] == "selector_added_non_gold":
            row["hard_negative_type"] = classify_hard_negative(row, semantic_threshold=semantic_threshold)
        else:
            row["hard_negative_type"] = ""

    rank_distribution = {
        category: group_summary([row for row in doc_rows if row["category"] == category], "rank_bucket", RANK_BUCKETS)
        for category in ("selector_added_non_gold", "selector_added_gold", "rank_removed_non_gold")
    }
    hard_negative_types = group_summary(selector_added_non_gold, "hard_negative_type", HARD_NEGATIVE_TYPES)
    query_buckets = summarize_query_buckets(query_rows)
    oracle = summarize_oracle_opportunity(oracle_rows)
    decision = decide_cee_feasibility(
        hard_negative_types=hard_negative_types,
        oracle=oracle,
        query_buckets=query_buckets,
    )
    return {
        "rows": len(prediction_rows),
        "top_k": int(top_k),
        "max_candidates": int(max_candidates),
        "oracle_pool_size": int(oracle_pool_size),
        "semantic_threshold_q_doc_cosine_p75": round(float(semantic_threshold), 6),
        "rank_distribution": rank_distribution,
        "hard_negative_type_breakdown": hard_negative_types,
        "query_level_failure": query_buckets,
        "oracle_edit_opportunity": oracle,
        "cee_feasibility": decision,
        "case_rows": doc_rows,
        "query_rows": query_rows,
    }


def summarize_query_buckets(query_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets = {
        "rank_complete_and_selector_changed": [],
        "rank_incomplete_and_selector_added_gold": [],
        "rank_incomplete_but_selector_added_only_non_gold": [],
        "selector_improves_support_complete": [],
        "selector_neutral_support_complete": [],
        "selector_hurts_support_complete": [],
    }
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in query_rows:
        by_type.setdefault(str(row.get("query_type") or "unknown"), []).append(row)
        if row["rank_complete"] and row["selector_changed"]:
            buckets["rank_complete_and_selector_changed"].append(row)
        if not row["rank_complete"] and int(row["selector_added_gold"]) > 0:
            buckets["rank_incomplete_and_selector_added_gold"].append(row)
        if not row["rank_complete"] and int(row["selector_added_gold"]) == 0 and int(row["selector_added_non_gold"]) > 0:
            buckets["rank_incomplete_but_selector_added_only_non_gold"].append(row)
        if float(row["support_complete_delta"]) > 0.0:
            buckets["selector_improves_support_complete"].append(row)
        elif float(row["support_complete_delta"]) < 0.0:
            buckets["selector_hurts_support_complete"].append(row)
        else:
            buckets["selector_neutral_support_complete"].append(row)
    return {
        "buckets": {name: summarize_query_rows(rows) for name, rows in buckets.items()},
        "by_query_type": {name: summarize_query_rows(rows) for name, rows in sorted(by_type.items())},
    }


def summarize_query_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "queries": 0,
            "rank_complete_rate": 0.0,
            "selector_changed_rate": 0.0,
            "avg_added_gold": 0.0,
            "avg_added_non_gold": 0.0,
            "avg_support_complete_delta": 0.0,
            "avg_support_recall_delta": 0.0,
        }
    denom = float(len(rows))
    return {
        "queries": len(rows),
        "rank_complete_rate": round(sum(1 for row in rows if row["rank_complete"]) / denom, 4),
        "selector_changed_rate": round(sum(1 for row in rows if row["selector_changed"]) / denom, 4),
        "avg_added_gold": round(sum(float(row["selector_added_gold"]) for row in rows) / denom, 4),
        "avg_added_non_gold": round(sum(float(row["selector_added_non_gold"]) for row in rows) / denom, 4),
        "avg_support_complete_delta": round(sum(float(row["support_complete_delta"]) for row in rows) / denom, 4),
        "avg_support_recall_delta": round(sum(float(row["support_recall_delta"]) for row in rows) / denom, 4),
    }


def summarize_oracle_opportunity(oracle_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rank_summary = summarize_metric_rows([row["rank"]["metrics"] for row in oracle_rows])
    selector_summary = summarize_metric_rows([row["selector"]["metrics"] for row in oracle_rows])
    edit1 = summarize_oracle(oracle_rows, "oracle_edit1")
    edit2 = summarize_oracle(oracle_rows, "oracle_edit2")
    return {
        "rank_topk": rank_summary,
        "selector_v1": selector_summary,
        "oracle_edit1": edit1,
        "oracle_edit2": edit2,
        "deltas": {
            "selector_v1_support_complete_minus_rank": round(
                float(selector_summary.get("support_complete") or 0.0) - float(rank_summary.get("support_complete") or 0.0),
                6,
            ),
            "oracle_edit1_support_complete_minus_rank": round(
                float(edit1.get("support_complete") or 0.0) - float(rank_summary.get("support_complete") or 0.0),
                6,
            ),
            "oracle_edit2_support_complete_minus_rank": round(
                float(edit2.get("support_complete") or 0.0) - float(rank_summary.get("support_complete") or 0.0),
                6,
            ),
        },
    }


def import_ratio(added_gold: float, added_non_gold: float) -> float:
    return float(added_non_gold) / max(1.0, float(added_gold))


def decide_cee_feasibility(
    *,
    hard_negative_types: dict[str, Any],
    oracle: dict[str, Any],
    query_buckets: dict[str, Any],
) -> dict[str, Any]:
    oracle_gain = float(oracle["deltas"]["oracle_edit1_support_complete_minus_rank"])
    if float(oracle["deltas"]["oracle_edit2_support_complete_minus_rank"]) > oracle_gain:
        oracle_gain = float(oracle["deltas"]["oracle_edit2_support_complete_minus_rank"])
    edit2 = oracle["oracle_edit2"]
    oracle_ratio = import_ratio(float(edit2.get("added_gold") or 0), float(edit2.get("added_non_gold") or 0))
    v1_ratio = 15.8
    dominant_type_share = 0.0
    total_hard = sum(int(item.get("docs") or 0) for item in hard_negative_types.values())
    if total_hard:
        dominant_type_share = max(int(item.get("docs") or 0) for item in hard_negative_types.values()) / total_hard
    over_edit_queries = int(query_buckets["buckets"]["rank_complete_and_selector_changed"]["queries"])
    passes = {
        "oracle_support_complete_gain_ge_2pp": oracle_gain >= 0.02,
        "oracle_import_ratio_3x_better_than_v1": oracle_ratio <= (v1_ratio / 3.0),
        "dominant_hard_negative_type_ge_30pct": dominant_type_share >= 0.30,
        "over_edit_rate_measurable": over_edit_queries > 0,
    }
    passed_count = sum(1 for value in passes.values() if value)
    if passed_count >= 3:
        recommendation = "CEE recommended"
    elif passed_count >= 2:
        recommendation = "CEE risky but possible"
    else:
        recommendation = "CEE not recommended"
    return {
        "recommendation": recommendation,
        "passes": passes,
        "oracle_best_support_complete_gain": round(oracle_gain, 6),
        "oracle_edit2_non_gold_per_gold": round(oracle_ratio, 4),
        "v1_non_gold_per_gold_reference": v1_ratio,
        "dominant_hard_negative_type_share": round(dominant_type_share, 4),
        "over_edit_complete_queries": over_edit_queries,
    }


def write_case_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qid",
        "query_idx",
        "query_type",
        "category",
        "hard_negative_type",
        "candidate_index",
        "title",
        "rank",
        "rank_bucket",
        "q_doc_cosine",
        "answer_in_doc",
        "bridge_entity_in_doc",
        "question_token_coverage",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def table_value(summary: dict[str, Any], feature: str) -> str:
    item = summary.get("features", {}).get(feature)
    return f"{float(item['mean']):.4f}" if item else ""


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# D-PathRAG Hard-Negative Characterization v2",
        "",
        f"- Rows: `{payload['rows']}`",
        f"- Top-k: `{payload['top_k']}`",
        f"- Max candidates: `{payload['max_candidates']}`",
        f"- Oracle pool size: `{payload['oracle_pool_size']}`",
        f"- Semantic hard-negative threshold: `{payload['semantic_threshold_q_doc_cosine_p75']}`",
        "",
        "## CEE Feasibility Decision",
        "",
        f"- Recommendation: **{payload['cee_feasibility']['recommendation']}**",
        f"- Oracle best support-complete gain: `{payload['cee_feasibility']['oracle_best_support_complete_gain']:+.4f}`",
        f"- Oracle Edit@2 non-gold per gold: `{payload['cee_feasibility']['oracle_edit2_non_gold_per_gold']}`",
        f"- Dominant hard-negative type share: `{payload['cee_feasibility']['dominant_hard_negative_type_share']}`",
        f"- Over-edited complete queries: `{payload['cee_feasibility']['over_edit_complete_queries']}`",
        "",
        "## Rank Bucket Distribution",
        "",
        "| Category | Bucket | Docs | Queries | q_doc_cosine | answer_in_doc | bridge_entity_in_doc | question_coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, buckets in payload["rank_distribution"].items():
        for bucket in RANK_BUCKETS:
            summary = buckets.get(bucket, {"docs": 0, "queries": 0, "features": {}})
            lines.append(
                f"| {category} | {bucket} | {summary['docs']} | {summary['queries']} | "
                f"{table_value(summary, 'q_doc_cosine')} | {table_value(summary, 'answer_in_doc')} | "
                f"{table_value(summary, 'bridge_entity_in_doc')} | {table_value(summary, 'question_token_coverage')} |"
            )
    lines.extend(
        [
            "",
            "## Hard-Negative Type Breakdown",
            "",
            "| Type | Docs | Queries | Mean Rank | q_doc_cosine | answer_in_doc | bridge_entity_in_doc | question_coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in HARD_NEGATIVE_TYPES:
        summary = payload["hard_negative_type_breakdown"].get(name, {"docs": 0, "queries": 0, "features": {}})
        lines.append(
            f"| {name} | {summary['docs']} | {summary['queries']} | {table_value(summary, 'rank')} | "
            f"{table_value(summary, 'q_doc_cosine')} | {table_value(summary, 'answer_in_doc')} | "
            f"{table_value(summary, 'bridge_entity_in_doc')} | {table_value(summary, 'question_token_coverage')} |"
        )
    lines.extend(
        [
            "",
            "## Query-Level Failure Buckets",
            "",
            "| Bucket | Queries | Rank Complete Rate | Selector Changed Rate | Avg Added Gold | Avg Added Non-Gold | Avg Support Complete Delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in payload["query_level_failure"]["buckets"].items():
        lines.append(
            f"| {name} | {summary['queries']} | {summary['rank_complete_rate']:.4f} | "
            f"{summary['selector_changed_rate']:.4f} | {summary['avg_added_gold']:.4f} | "
            f"{summary['avg_added_non_gold']:.4f} | {summary['avg_support_complete_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Oracle Edit Opportunity",
            "",
            "| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Queries With Edit | Stop Rate | Added Gold | Added Non-Gold |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    oracle = payload["oracle_edit_opportunity"]
    for name in ("rank_topk", "selector_v1", "oracle_edit1", "oracle_edit2"):
        summary = oracle[name]
        lines.append(
            f"| {name} | {summary.get('support_recall', 0.0):.4f} | {summary.get('support_complete', 0.0):.4f} | "
            f"{summary.get('selected_gold_count', 0.0):.4f} | {summary.get('bridge_entity_recall', 0.0):.4f} | "
            f"{summary.get('queries_with_edit', '')} | {summary.get('stop_rate', '')} | "
            f"{summary.get('added_gold', '')} | {summary.get('added_non_gold', '')} |"
        )
    lines.extend(["", "## Gate Checks", "", "| Gate | Pass |", "|---|---:|"])
    for name, value in payload["cee_feasibility"]["passes"].items():
        lines.append(f"| {name} | {str(bool(value))} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--embedding_npz", default="")
    parser.add_argument("--cache_limit", type=int, default=1000)
    parser.add_argument("--prediction_limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--oracle_pool_size", type=int, default=20)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_cases_csv", default="")
    args = parser.parse_args()

    cache_rows = load_jsonl(args.cache_jsonl, limit=int(args.cache_limit))
    prediction_rows = load_jsonl(args.predictions_jsonl, limit=int(args.prediction_limit))
    embedding_by_qid, embedding_feature_names = load_embedding_payload(args.embedding_npz, limit=int(args.cache_limit))
    output = analyze_v2(
        cache_rows,
        prediction_rows,
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        oracle_pool_size=int(args.oracle_pool_size),
        embedding_by_qid=embedding_by_qid,
        embedding_feature_names=embedding_feature_names,
    )
    output["cache_jsonl"] = str(args.cache_jsonl)
    output["predictions_jsonl"] = str(args.predictions_jsonl)
    output["embedding_npz"] = str(args.embedding_npz)
    output["embedding_feature_names"] = list(embedding_feature_names)
    case_rows = list(output.pop("case_rows"))
    output.pop("query_rows", None)
    write_json(output, args.output_json)
    write_markdown(output, args.output_md)
    if args.output_cases_csv:
        write_case_csv(case_rows, args.output_cases_csv)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
