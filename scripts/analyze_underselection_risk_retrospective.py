#!/usr/bin/env python3
"""Retrospective sanity check for selection-independent under-selection risk.

This analysis is reader-free and LLM-free. It uses existing PropRAG full1000
DBEC/DAEC-selective traces and SetR-faithful selected-only outputs to test
whether a pre-reader decomposition feature, requirement_count >= 3, predicts:

1. higher SetR-faithful count-based under-selection rate, and
2. larger DBEC-selective minus SetR-faithful answer F1.

The script intentionally avoids weighted risk scores. The risk split is a
single frozen feature so it can be reused for a later held-out preregistered
predictive-validity experiment if the retrospective trend is sane.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score


REPORT_DIR = Path("reports/underselection_risk_retrospective_20260507")
DBEC_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")
SETR_RUN_DIR = Path("run_logs/setr_faithful_proprag_full1000_20260507")
SETR_REPORT_DIR = Path("reports/setr_faithful_proprag_full1000_20260507")

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260507

DATASETS: tuple[tuple[str, str], ...] = (
    ("2Wiki", "2wikimultihopqa"),
    ("HotpotQA", "hotpotqa"),
    ("MuSiQue", "musique"),
)

TARGET_DATASETS = {"2Wiki", "MuSiQue"}
RISK_THRESHOLD = 3


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_multiset_recall(gold_titles: list[Any], selected_titles: list[Any]) -> float:
    gold = [normalize_title(title) for title in gold_titles if normalize_title(title)]
    selected = [normalize_title(title) for title in selected_titles if normalize_title(title)]
    if not gold:
        return 0.0
    used = [False] * len(selected)
    hits = 0
    for gold_title in gold:
        for index, selected_title in enumerate(selected):
            if used[index]:
                continue
            if selected_title == gold_title:
                used[index] = True
                hits += 1
                break
    return float(hits / len(gold))


def list_items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def mapping_items(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def occurrence_keys(rows: list[Mapping[str, Any]]) -> list[tuple[str, int]]:
    counts: defaultdict[str, int] = defaultdict(int)
    keys: list[tuple[str, int]] = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        occurrence = counts[question]
        counts[question] += 1
        keys.append((question, occurrence))
    return keys


def align_to_reference(
    dataset: str,
    reference_rows: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    *,
    name: str,
) -> list[Mapping[str, Any]]:
    reference_keys = occurrence_keys(reference_rows)
    keys = occurrence_keys(rows)
    if len(keys) != len(reference_keys):
        raise ValueError(f"{dataset}: {name} length={len(keys)}, expected={len(reference_keys)}")
    keyed = {key: row for key, row in zip(keys, rows)}
    missing = [key for key in reference_keys if key not in keyed]
    if missing:
        raise ValueError(f"{dataset}: {name} missing {len(missing)} occurrence-aligned questions")
    return [keyed[key] for key in reference_keys]


def compute_depths(requirements: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    by_id = {str(req.get("unit_id") or ""): req for req in requirements}
    visiting: set[str] = set()
    memo: dict[str, int] = {}

    def depth(unit_id: str) -> int:
        if unit_id in memo:
            return memo[unit_id]
        if unit_id in visiting:
            return 0
        visiting.add(unit_id)
        req = by_id.get(unit_id) or {}
        deps = [str(dep) for dep in list_items(req.get("depends_on")) if str(dep)]
        value = 0 if not deps else 1 + max(depth(dep) for dep in deps)
        visiting.discard(unit_id)
        memo[unit_id] = value
        return value

    return {unit_id: depth(unit_id) for unit_id in by_id}


def dbec_path(slug: str) -> Path:
    return DBEC_DIR / f"{slug}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def setr_selected_pool_path(slug: str) -> Path:
    return SETR_RUN_DIR / f"{slug}_proprag_setr_k20_doc768_faithful.selected_pool.json"


def setr_eval_path(slug: str) -> Path:
    return SETR_REPORT_DIR / f"{slug}_proprag_setr_k20_doc768_faithful.eval.json"


def dbec_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_index, trace in enumerate(list_items(data.get("setwise_selector_query_traces"))):
        if not isinstance(trace, Mapping):
            continue
        selector_trace = mapping_items(trace.get("selector_trace"))
        requirements = [
            req for req in list_items(selector_trace.get("requirements"))
            if isinstance(req, Mapping)
        ]
        dependent_requirements = [req for req in requirements if list_items(req.get("depends_on"))]
        dependent_req_ids = [str(req.get("unit_id") or "") for req in dependent_requirements]
        candidates_by_req = mapping_items(selector_trace.get("binding_candidates_by_requirement"))
        binding_candidate_counts = [
            len(list_items(candidates_by_req.get(req_id)))
            for req_id in dependent_req_ids
        ]
        depths = compute_depths(requirements)
        metrics = mapping_items(trace.get("selector_metrics"))
        gold_titles = list_items(trace.get("gold_titles"))
        top_titles = list_items(trace.get("selector_top_titles"))
        rows.append({
            "query_index": query_index,
            "question": str(trace.get("question") or ""),
            "gold_answers": list_items(trace.get("gold_answers")),
            "gold_titles": gold_titles,
            "gold_doc_count": safe_int(trace.get("gold_doc_count"), len(gold_titles)),
            "dbec_answer": str(trace.get("selector_answer") or ""),
            "dbec_em": safe_float(metrics.get("ExactMatch")),
            "dbec_f1": safe_float(metrics.get("F1")),
            "dbec_top_titles": top_titles,
            "dbec_r5_title": title_multiset_recall(gold_titles, top_titles[:5]),
            "requirement_count": len(requirements),
            "dependent_req_count": len(dependent_requirements),
            "max_dependency_depth": max(depths.values()) if depths else 0,
            "binding_candidate_count_total": sum(binding_candidate_counts),
            "binding_candidate_count_mean": mean(binding_candidate_counts) if binding_candidate_counts else 0.0,
            "binding_candidate_count_max": max(binding_candidate_counts) if binding_candidate_counts else 0,
            "selective_binding_decision": str(selector_trace.get("selective_binding_decision") or ""),
        })
    return rows


def setr_pool_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_index, record in enumerate(list_items(data.get("records"))):
        if not isinstance(record, Mapping):
            continue
        trace = mapping_items(record.get("setr_selection_trace"))
        selected_count = safe_int(trace.get("reader_pool_size"), len(list_items(record.get("pool_titles"))))
        selected_positions = list_items(trace.get("selected_positions"))
        gold_titles = list_items(record.get("gold_titles"))
        rows.append({
            "query_index": query_index,
            "question": str(record.get("question") or ""),
            "gold_titles": gold_titles,
            "gold_doc_count": len(gold_titles),
            "setr_selected_count": selected_count,
            "setr_selected_positions": selected_positions,
            "setr_parse_success": bool(trace.get("parse_success")),
            "setr_empty_fallback_used": bool(trace.get("empty_fallback_used")),
            "setr_top_titles": list_items(record.get("pool_titles"))[:selected_count],
        })
    return rows


def setr_eval_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples = [row for row in list_items(data.get("examples")) if isinstance(row, Mapping)]
    gold_answers = [list_items(row.get("gold_answers")) for row in examples]
    predictions = [str(row.get("answer") or "") for row in examples]
    _, per_query_em = QAExactMatch(global_config=None).calculate_metric_scores(gold_answers, predictions)
    _, per_query_f1 = QAF1Score(global_config=None).calculate_metric_scores(gold_answers, predictions)

    rows: list[dict[str, Any]] = []
    for query_index, (example, em_row, f1_row) in enumerate(zip(examples, per_query_em, per_query_f1)):
        retrieval_trace = mapping_items(example.get("retrieval_trace"))
        top_titles = list_items(retrieval_trace.get("external_pool_titles"))[:5]
        rows.append({
            "query_index": query_index,
            "question": str(example.get("question") or ""),
            "setr_answer": str(example.get("answer") or ""),
            "setr_em": safe_float(em_row.get("ExactMatch")),
            "setr_f1": safe_float(f1_row.get("F1")),
            "setr_eval_top_titles": top_titles,
        })
    return rows


def build_query_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, slug in DATASETS:
        dbec = dbec_rows(read_json(dbec_path(slug)))
        setr_pool = align_to_reference(
            dataset,
            dbec,
            setr_pool_rows(read_json(setr_selected_pool_path(slug))),
            name="SetR-faithful selected_pool",
        )
        setr_eval = align_to_reference(
            dataset,
            dbec,
            setr_eval_rows(read_json(setr_eval_path(slug))),
            name="SetR-faithful eval",
        )
        for dbec_row, pool_row, eval_row in zip(dbec, setr_pool, setr_eval):
            gold_titles = list(dbec_row.get("gold_titles") or pool_row.get("gold_titles") or [])
            gold_doc_count = max(
                safe_int(dbec_row.get("gold_doc_count")),
                safe_int(pool_row.get("gold_doc_count")),
                len(gold_titles),
            )
            requirement_count = safe_int(dbec_row.get("requirement_count"))
            risk_group = "high" if requirement_count >= RISK_THRESHOLD else "low"
            setr_top_titles = list(pool_row.get("setr_top_titles") or eval_row.get("setr_eval_top_titles") or [])
            setr_r5_title = title_multiset_recall(gold_titles, setr_top_titles[:5])
            setr_selected_count = safe_int(pool_row.get("setr_selected_count"))
            rows.append({
                "dataset": dataset,
                "query_index": safe_int(dbec_row.get("query_index")),
                "question": str(dbec_row.get("question") or ""),
                "risk_group": risk_group,
                "risk_high": int(risk_group == "high"),
                "requirement_count": requirement_count,
                "dependent_req_count": safe_int(dbec_row.get("dependent_req_count")),
                "max_dependency_depth": safe_int(dbec_row.get("max_dependency_depth")),
                "binding_candidate_count_total": safe_float(dbec_row.get("binding_candidate_count_total")),
                "binding_candidate_count_mean": safe_float(dbec_row.get("binding_candidate_count_mean")),
                "binding_candidate_count_max": safe_float(dbec_row.get("binding_candidate_count_max")),
                "gold_doc_count": gold_doc_count,
                "setr_selected_count": setr_selected_count,
                "setr_under_selected": int(setr_selected_count < gold_doc_count),
                "dbec_em": safe_float(dbec_row.get("dbec_em")),
                "dbec_f1": safe_float(dbec_row.get("dbec_f1")),
                "setr_em": safe_float(eval_row.get("setr_em")),
                "setr_f1": safe_float(eval_row.get("setr_f1")),
                "delta_em_dbec_minus_setr": safe_float(dbec_row.get("dbec_em")) - safe_float(eval_row.get("setr_em")),
                "delta_f1_dbec_minus_setr": safe_float(dbec_row.get("dbec_f1")) - safe_float(eval_row.get("setr_f1")),
                "dbec_r5_title": safe_float(dbec_row.get("dbec_r5_title")),
                "setr_r5_title": setr_r5_title,
                "delta_r5_title_dbec_minus_setr": safe_float(dbec_row.get("dbec_r5_title")) - setr_r5_title,
                "setr_parse_success": int(bool(pool_row.get("setr_parse_success"))),
                "setr_empty_fallback_used": int(bool(pool_row.get("setr_empty_fallback_used"))),
                "selective_binding_decision": str(dbec_row.get("selective_binding_decision") or ""),
            })
    return rows


def paired_bootstrap_delta(
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if len(left_values) != len(right_values):
        raise ValueError("paired bootstrap requires equal-length inputs")
    if not left_values:
        return empty_ci()
    deltas = np.asarray(left_values, dtype=float) - np.asarray(right_values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = deltas[rng.integers(0, deltas.size, size=deltas.size)]
        boot[index] = float(np.mean(sampled))
    return ci_payload(deltas, boot)


def independent_bootstrap_mean_diff(
    high_values: Sequence[float],
    low_values: Sequence[float],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if not high_values or not low_values:
        return empty_ci()
    high = np.asarray(high_values, dtype=float)
    low = np.asarray(low_values, dtype=float)
    observed = float(np.mean(high) - np.mean(low))
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        high_sample = high[rng.integers(0, high.size, size=high.size)]
        low_sample = low[rng.integers(0, low.size, size=low.size)]
        boot[index] = float(np.mean(high_sample) - np.mean(low_sample))
    return {
        "n_high": int(high.size),
        "n_low": int(low.size),
        "delta_mean": observed,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_delta_gt_0": float(np.mean(boot > 0.0)),
        "ci_excludes_zero": bool(np.percentile(boot, 2.5) > 0.0 or np.percentile(boot, 97.5) < 0.0),
    }


def ci_payload(observed_values: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(observed_values.size),
        "delta_mean": float(np.mean(observed_values)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_delta_gt_0": float(np.mean(boot > 0.0)),
        "ci_excludes_zero": bool(np.percentile(boot, 2.5) > 0.0 or np.percentile(boot, 97.5) < 0.0),
    }


def empty_ci() -> dict[str, Any]:
    return {
        "n": 0,
        "delta_mean": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "p_delta_gt_0": 0.0,
        "ci_excludes_zero": False,
    }


def mean_float(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(mean([safe_float(row.get(field)) for row in rows])) if rows else 0.0


def subset_rows(rows: Sequence[Mapping[str, Any]], dataset: str, risk_group: str | None = None) -> list[Mapping[str, Any]]:
    output = [row for row in rows if str(row.get("dataset")) == dataset]
    if risk_group is not None:
        output = [row for row in output if str(row.get("risk_group")) == risk_group]
    return output


def grouped_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        all_rows = subset_rows(rows, dataset)
        for risk_group in ("low", "high"):
            group_rows = [row for row in all_rows if str(row.get("risk_group")) == risk_group]
            stats_f1 = paired_bootstrap_delta(
                [safe_float(row.get("dbec_f1")) for row in group_rows],
                [safe_float(row.get("setr_f1")) for row in group_rows],
                seed=BOOTSTRAP_SEED + len(output) * 17,
            )
            stats_r5 = paired_bootstrap_delta(
                [safe_float(row.get("dbec_r5_title")) for row in group_rows],
                [safe_float(row.get("setr_r5_title")) for row in group_rows],
                seed=BOOTSTRAP_SEED + 1000 + len(output) * 17,
            )
            output.append({
                "dataset": dataset,
                "risk_group": risk_group,
                "n": len(group_rows),
                "share": len(group_rows) / len(all_rows) if all_rows else 0.0,
                "mean_requirement_count": mean_float(group_rows, "requirement_count"),
                "mean_dependent_req_count": mean_float(group_rows, "dependent_req_count"),
                "mean_gold_doc_count": mean_float(group_rows, "gold_doc_count"),
                "mean_setr_selected_count": mean_float(group_rows, "setr_selected_count"),
                "setr_under_selected": int(sum(safe_int(row.get("setr_under_selected")) for row in group_rows)),
                "setr_under_select_rate": mean_float(group_rows, "setr_under_selected"),
                "dbec_f1": mean_float(group_rows, "dbec_f1"),
                "setr_f1": mean_float(group_rows, "setr_f1"),
                "delta_f1_dbec_minus_setr": stats_f1["delta_mean"],
                "delta_f1_ci_low": stats_f1["ci_low"],
                "delta_f1_ci_high": stats_f1["ci_high"],
                "delta_f1_ci_excludes_zero": stats_f1["ci_excludes_zero"],
                "dbec_r5_title": mean_float(group_rows, "dbec_r5_title"),
                "setr_r5_title": mean_float(group_rows, "setr_r5_title"),
                "delta_r5_title_dbec_minus_setr": stats_r5["delta_mean"],
                "delta_r5_ci_low": stats_r5["ci_low"],
                "delta_r5_ci_high": stats_r5["ci_high"],
                "delta_r5_ci_excludes_zero": stats_r5["ci_excludes_zero"],
            })
    return output


def interaction_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = [(dataset, subset_rows(rows, dataset)) for dataset, _ in DATASETS]
    target_rows = [row for row in rows if str(row.get("dataset")) in TARGET_DATASETS]
    groups.append(("2Wiki+MuSiQue", target_rows))
    for dataset, dataset_rows in groups:
        high = [row for row in dataset_rows if str(row.get("risk_group")) == "high"]
        low = [row for row in dataset_rows if str(row.get("risk_group")) == "low"]
        output.append({
            "dataset": dataset,
            "metric": "SetR under-select rate: high - low",
            **independent_bootstrap_mean_diff(
                [safe_float(row.get("setr_under_selected")) for row in high],
                [safe_float(row.get("setr_under_selected")) for row in low],
                seed=BOOTSTRAP_SEED + 2000 + len(output) * 19,
            ),
        })
        output.append({
            "dataset": dataset,
            "metric": "DBEC-SetR dF1 interaction: high - low",
            **independent_bootstrap_mean_diff(
                [safe_float(row.get("delta_f1_dbec_minus_setr")) for row in high],
                [safe_float(row.get("delta_f1_dbec_minus_setr")) for row in low],
                seed=BOOTSTRAP_SEED + 3000 + len(output) * 19,
            ),
        })
        output.append({
            "dataset": dataset,
            "metric": "DBEC-SetR dR5_TITLE interaction: high - low",
            **independent_bootstrap_mean_diff(
                [safe_float(row.get("delta_r5_title_dbec_minus_setr")) for row in high],
                [safe_float(row.get("delta_r5_title_dbec_minus_setr")) for row in low],
                seed=BOOTSTRAP_SEED + 4000 + len(output) * 19,
            ),
        })
    return output


def gold_slice_specs(dataset: str) -> tuple[tuple[str, int, str], ...]:
    if dataset == "HotpotQA":
        return (("gold_doc_count=2", 2, "eq"),)
    return (
        ("gold_doc_count=2", 2, "eq"),
        ("gold_doc_count>=3", 3, "ge"),
        ("gold_doc_count>=4", 4, "ge"),
    )


def in_gold_slice(row: Mapping[str, Any], threshold: int, mode: str) -> bool:
    count = safe_int(row.get("gold_doc_count"))
    if mode == "eq":
        return count == threshold
    if mode == "ge":
        return count >= threshold
    raise ValueError(f"unknown gold slice mode: {mode}")


def gold_control_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        dataset_rows = subset_rows(rows, dataset)
        for slice_name, threshold, mode in gold_slice_specs(dataset):
            slice_rows = [row for row in dataset_rows if in_gold_slice(row, threshold, mode)]
            if not slice_rows:
                continue
            for risk_group in ("low", "high"):
                group_rows = [row for row in slice_rows if str(row.get("risk_group")) == risk_group]
                if not group_rows:
                    continue
                stats_f1 = paired_bootstrap_delta(
                    [safe_float(row.get("dbec_f1")) for row in group_rows],
                    [safe_float(row.get("setr_f1")) for row in group_rows],
                    seed=BOOTSTRAP_SEED + 5000 + len(output) * 23,
                )
                output.append({
                    "dataset": dataset,
                    "gold_slice": slice_name,
                    "risk_group": risk_group,
                    "n": len(group_rows),
                    "share_within_gold_slice": len(group_rows) / len(slice_rows),
                    "mean_requirement_count": mean_float(group_rows, "requirement_count"),
                    "mean_gold_doc_count": mean_float(group_rows, "gold_doc_count"),
                    "mean_setr_selected_count": mean_float(group_rows, "setr_selected_count"),
                    "setr_under_select_rate": mean_float(group_rows, "setr_under_selected"),
                    "delta_f1_dbec_minus_setr": stats_f1["delta_mean"],
                    "delta_f1_ci_low": stats_f1["ci_low"],
                    "delta_f1_ci_high": stats_f1["ci_high"],
                    "delta_f1_ci_excludes_zero": stats_f1["ci_excludes_zero"],
                })
    return output


def gold_control_interactions(gold_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_slice: defaultdict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in gold_rows:
        by_slice[(str(row["dataset"]), str(row["gold_slice"]))][str(row["risk_group"])] = row

    output: list[dict[str, Any]] = []
    for (dataset, gold_slice), grouped in by_slice.items():
        low = grouped.get("low")
        high = grouped.get("high")
        if not low or not high:
            continue
        output.append({
            "dataset": dataset,
            "gold_slice": gold_slice,
            "n_high": safe_int(high.get("n")),
            "n_low": safe_int(low.get("n")),
            "under_select_high_minus_low": safe_float(high.get("setr_under_select_rate")) - safe_float(low.get("setr_under_select_rate")),
            "df1_high_minus_low": safe_float(high.get("delta_f1_dbec_minus_setr")) - safe_float(low.get("delta_f1_dbec_minus_setr")),
        })
    return output


def scout_rule_specs() -> tuple[tuple[str, str], ...]:
    return (
        ("requirement_count>=3", "requirement_count >= 3"),
        (
            "requirement_count>=3_and_mean_binding_candidates>=2",
            "requirement_count >= 3 and binding_candidate_count_mean >= 2",
        ),
        (
            "requirement_count>=3_and_max_binding_candidates>=2",
            "requirement_count >= 3 and binding_candidate_count_max >= 2",
        ),
        (
            "requirement_count>=3_and_dependent_req_count>=2",
            "requirement_count >= 3 and dependent_req_count >= 2",
        ),
        ("requirement_count>=4", "requirement_count >= 4"),
    )


def scout_rule_match(row: Mapping[str, Any], rule_name: str) -> bool:
    requirement_count = safe_int(row.get("requirement_count"))
    if rule_name == "requirement_count>=3":
        return requirement_count >= 3
    if rule_name == "requirement_count>=3_and_mean_binding_candidates>=2":
        return requirement_count >= 3 and safe_float(row.get("binding_candidate_count_mean")) >= 2.0
    if rule_name == "requirement_count>=3_and_max_binding_candidates>=2":
        return requirement_count >= 3 and safe_float(row.get("binding_candidate_count_max")) >= 2.0
    if rule_name == "requirement_count>=3_and_dependent_req_count>=2":
        return requirement_count >= 3 and safe_int(row.get("dependent_req_count")) >= 2
    if rule_name == "requirement_count>=4":
        return requirement_count >= 4
    raise ValueError(f"unknown scout rule: {rule_name}")


def scout_dataset_groups(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    groups = [(dataset, subset_rows(rows, dataset)) for dataset, _ in DATASETS]
    groups.append(("2Wiki+MuSiQue", [row for row in rows if str(row.get("dataset")) in TARGET_DATASETS]))
    return groups


def exploratory_scout_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rule_name, rule_description in scout_rule_specs():
        for dataset, dataset_rows in scout_dataset_groups(rows):
            high = [row for row in dataset_rows if scout_rule_match(row, rule_name)]
            low = [row for row in dataset_rows if not scout_rule_match(row, rule_name)]
            if not high or not low:
                continue
            under_stats = independent_bootstrap_mean_diff(
                [safe_float(row.get("setr_under_selected")) for row in high],
                [safe_float(row.get("setr_under_selected")) for row in low],
                seed=BOOTSTRAP_SEED + 7000 + len(output) * 29,
            )
            f1_stats = independent_bootstrap_mean_diff(
                [safe_float(row.get("delta_f1_dbec_minus_setr")) for row in high],
                [safe_float(row.get("delta_f1_dbec_minus_setr")) for row in low],
                seed=BOOTSTRAP_SEED + 8000 + len(output) * 29,
            )
            output.append({
                "rule": rule_name,
                "rule_description": rule_description,
                "dataset": dataset,
                "n_high": len(high),
                "n_low": len(low),
                "high_share": len(high) / len(dataset_rows) if dataset_rows else 0.0,
                "mean_gold_doc_count_high": mean_float(high, "gold_doc_count"),
                "mean_gold_doc_count_low": mean_float(low, "gold_doc_count"),
                "under_select_rate_high": mean_float(high, "setr_under_selected"),
                "under_select_rate_low": mean_float(low, "setr_under_selected"),
                "under_select_high_minus_low": under_stats["delta_mean"],
                "under_select_ci_low": under_stats["ci_low"],
                "under_select_ci_high": under_stats["ci_high"],
                "under_select_ci_excludes_zero": under_stats["ci_excludes_zero"],
                "df1_high": mean_float(high, "delta_f1_dbec_minus_setr"),
                "df1_low": mean_float(low, "delta_f1_dbec_minus_setr"),
                "df1_high_minus_low": f1_stats["delta_mean"],
                "df1_ci_low": f1_stats["ci_low"],
                "df1_ci_high": f1_stats["ci_high"],
                "df1_ci_excludes_zero": f1_stats["ci_excludes_zero"],
            })
    return output


def exploratory_gold_control_scout_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rules_to_check = (
        "requirement_count>=3_and_max_binding_candidates>=2",
        "requirement_count>=3_and_dependent_req_count>=2",
        "requirement_count>=4",
    )
    for rule_name in rules_to_check:
        for dataset, _ in DATASETS:
            dataset_rows = subset_rows(rows, dataset)
            for slice_name, threshold, mode in gold_slice_specs(dataset):
                slice_rows = [row for row in dataset_rows if in_gold_slice(row, threshold, mode)]
                high = [row for row in slice_rows if scout_rule_match(row, rule_name)]
                low = [row for row in slice_rows if not scout_rule_match(row, rule_name)]
                if not high or not low:
                    continue
                output.append({
                    "rule": rule_name,
                    "dataset": dataset,
                    "gold_slice": slice_name,
                    "n_high": len(high),
                    "n_low": len(low),
                    "under_select_high_minus_low": mean_float(high, "setr_under_selected") - mean_float(low, "setr_under_selected"),
                    "df1_high_minus_low": mean_float(high, "delta_f1_dbec_minus_setr") - mean_float(low, "delta_f1_dbec_minus_setr"),
                    "df1_high": mean_float(high, "delta_f1_dbec_minus_setr"),
                    "df1_low": mean_float(low, "delta_f1_dbec_minus_setr"),
                })
    return output


def format_signed(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):+.{digits}f}"


def format_float(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):.{digits}f}"


def format_percent(value: Any, digits: int = 1) -> str:
    return f"{100.0 * safe_float(value):.{digits}f}%"


def ci_text(low: Any, high: Any, *, digits: int = 4) -> str:
    return f"[{format_signed(low, digits)}, {format_signed(high, digits)}]"


def decision_text(interactions: Sequence[Mapping[str, Any]]) -> list[str]:
    target = {row["metric"]: row for row in interactions if row.get("dataset") == "2Wiki+MuSiQue"}
    under = target.get("SetR under-select rate: high - low", {})
    f1 = target.get("DBEC-SetR dF1 interaction: high - low", {})
    under_ok = safe_float(under.get("delta_mean")) > 0.0
    f1_ok = safe_float(f1.get("delta_mean")) > 0.0
    under_strong = under_ok and bool(under.get("ci_excludes_zero"))
    f1_strong = f1_ok and bool(f1.get("ci_excludes_zero"))
    if under_strong and f1_strong:
        verdict = "strong retrospective pass"
    elif under_ok and f1_ok:
        verdict = "partial retrospective pass"
    elif under_ok:
        verdict = "weak retrospective pass"
    else:
        verdict = "retrospective fail"
    return [
        f"M0 verdict: **{verdict}**.",
        (
            "Gold-controlled diagnostics weaken the interpretation: the frozen "
            "risk split partly proxies latent support depth, and DBEC F1-gain "
            "concentration is not stable within fixed gold-doc-count slices."
        ),
        (
            "Proceed to held-out preregistration only if we accept point-estimate "
            "under-selection prediction as enough for a low-cost follow-up. Do "
            "not frame this M0 result as strong predictive evidence for answer "
            "gain without a better risk feature or a gold-controlled success "
            "criterion."
        ),
    ]


def build_markdown(
    query_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    interactions: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
    gold_interactions: Sequence[Mapping[str, Any]],
    scout_rows: Sequence[Mapping[str, Any]],
    scout_gold_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = [
        "# Under-Selection Risk Retrospective Sanity",
        "",
        "Date: 2026-05-07",
        "",
        "This is an M0 retrospective check for a later preregistered predictive-validity experiment. It uses existing PropRAG full1000 DBEC-selective traces and SetR-faithful selected-only outputs; it makes no new LLM or reader calls.",
        "",
        f"Frozen risk split: **high-risk** if DBEC decomposition has `requirement_count >= {RISK_THRESHOLD}`, otherwise **low-risk**. This signal is computed before SetR selection and before answer evaluation.",
        "",
        "`under-select` is count-based: `SetR-faithful selected passage count < gold_doc_count`. It does not assert that selected passages are the correct supports.",
        "",
        "## Raw Risk-Slice Table",
        "",
        "| Dataset | Risk | N | Share | Demand count | Gold docs | SetR passages | SetR under-select | DBEC F1 | SetR F1 | dF1 | 95% CI | dR5_TITLE | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {risk_group} | {n} | {share} | {demands} | {gold} | {selected} | "
            "{under} | {dbec_f1} | {setr_f1} | {df1} | {df1_ci} | {dr5} | {dr5_ci} |".format(
                dataset=row["dataset"],
                risk_group=row["risk_group"],
                n=row["n"],
                share=format_percent(row["share"]),
                demands=format_float(row["mean_requirement_count"], 2),
                gold=format_float(row["mean_gold_doc_count"], 2),
                selected=format_float(row["mean_setr_selected_count"], 2),
                under=format_percent(row["setr_under_select_rate"]),
                dbec_f1=format_float(row["dbec_f1"]),
                setr_f1=format_float(row["setr_f1"]),
                df1=format_signed(row["delta_f1_dbec_minus_setr"]),
                df1_ci=ci_text(row["delta_f1_ci_low"], row["delta_f1_ci_high"]),
                dr5=format_signed(row["delta_r5_title_dbec_minus_setr"]),
                dr5_ci=ci_text(row["delta_r5_ci_low"], row["delta_r5_ci_high"]),
            )
        )

    lines.extend([
        "",
        "## High-Low Interactions",
        "",
        "Bootstrap CIs resample high-risk and low-risk rows independently. Positive values mean the high-risk slice has more under-selection or a larger DBEC-vs-SetR gap.",
        "",
        "| Dataset | Metric | N high | N low | High-low delta | 95% CI | P(delta > 0) | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in interactions:
        lines.append(
            "| {dataset} | {metric} | {n_high} | {n_low} | {delta} | {ci} | {prob} | {excludes} |".format(
                dataset=row["dataset"],
                metric=row["metric"],
                n_high=row.get("n_high", 0),
                n_low=row.get("n_low", 0),
                delta=format_signed(row.get("delta_mean")),
                ci=ci_text(row.get("ci_low"), row.get("ci_high")),
                prob=format_float(row.get("p_delta_gt_0"), 3),
                excludes="yes" if row.get("ci_excludes_zero") else "no",
            )
        )

    lines.extend([
        "",
        "## Gold-Controlled Diagnostic",
        "",
        "This table checks whether `requirement_count >= 3` still separates risk after conditioning on the number of annotated gold support documents. These gold counts are not available at inference time; this is only a retrospective confound diagnostic.",
        "",
        "| Dataset | Gold slice | Risk | N | Share in slice | Demand count | SetR passages | Under-select | dF1 | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in gold_rows:
        lines.append(
            "| {dataset} | {gold_slice} | {risk_group} | {n} | {share} | {demands} | "
            "{selected} | {under} | {df1} | {ci} |".format(
                dataset=row["dataset"],
                gold_slice=row["gold_slice"],
                risk_group=row["risk_group"],
                n=row["n"],
                share=format_percent(row["share_within_gold_slice"]),
                demands=format_float(row["mean_requirement_count"], 2),
                selected=format_float(row["mean_setr_selected_count"], 2),
                under=format_percent(row["setr_under_select_rate"]),
                df1=format_signed(row["delta_f1_dbec_minus_setr"]),
                ci=ci_text(row["delta_f1_ci_low"], row["delta_f1_ci_high"]),
            )
        )

    lines.extend([
        "",
        "| Dataset | Gold slice | N high | N low | Under-select high-low | dF1 high-low |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in gold_interactions:
        lines.append(
            "| {dataset} | {gold_slice} | {n_high} | {n_low} | {under} | {df1} |".format(
                dataset=row["dataset"],
                gold_slice=row["gold_slice"],
                n_high=row["n_high"],
                n_low=row["n_low"],
                under=format_signed(row["under_select_high_minus_low"]),
                df1=format_signed(row["df1_high_minus_low"]),
            )
        )

    lines.extend([
        "",
        "## Exploratory Feature Scout",
        "",
        "The following rules are exploratory and were checked after the frozen M0 split. They should not be presented as preregistered evidence. Their purpose is to decide whether a better selection-independent risk rule is worth preregistering for a held-out suffix.",
        "",
        "| Rule | Dataset | N high | Share high | Gold high/low | Under high-low | 95% CI | dF1 high | dF1 low | dF1 high-low | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in scout_rows:
        if row["dataset"] not in {"2Wiki", "MuSiQue", "2Wiki+MuSiQue", "HotpotQA"}:
            continue
        lines.append(
            "| {rule} | {dataset} | {n_high} | {share} | {gold_high}/{gold_low} | "
            "{under_delta} | {under_ci} | {df1_high} | {df1_low} | {df1_delta} | {df1_ci} |".format(
                rule=row["rule"],
                dataset=row["dataset"],
                n_high=row["n_high"],
                share=format_percent(row["high_share"]),
                gold_high=format_float(row["mean_gold_doc_count_high"], 2),
                gold_low=format_float(row["mean_gold_doc_count_low"], 2),
                under_delta=format_signed(row["under_select_high_minus_low"]),
                under_ci=ci_text(row["under_select_ci_low"], row["under_select_ci_high"]),
                df1_high=format_signed(row["df1_high"]),
                df1_low=format_signed(row["df1_low"]),
                df1_delta=format_signed(row["df1_high_minus_low"]),
                df1_ci=ci_text(row["df1_ci_low"], row["df1_ci_high"]),
            )
        )

    lines.extend([
        "",
        "Gold-controlled spot check for exploratory rules:",
        "",
        "| Rule | Dataset | Gold slice | N high | N low | Under high-low | dF1 high-low | dF1 high | dF1 low |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in scout_gold_rows:
        lines.append(
            "| {rule} | {dataset} | {gold_slice} | {n_high} | {n_low} | {under} | {df1_delta} | {df1_high} | {df1_low} |".format(
                rule=row["rule"],
                dataset=row["dataset"],
                gold_slice=row["gold_slice"],
                n_high=row["n_high"],
                n_low=row["n_low"],
                under=format_signed(row["under_select_high_minus_low"]),
                df1_delta=format_signed(row["df1_high_minus_low"]),
                df1_high=format_signed(row["df1_high"]),
                df1_low=format_signed(row["df1_low"]),
            )
        )

    lines.extend(["", "## Key Findings", ""])
    for index, finding in enumerate(
        key_findings(summary_rows, interactions, gold_interactions, scout_rows, scout_gold_rows),
        start=1,
    ):
        lines.append(f"{index}. {finding}")

    lines.extend(["", "## M0 Decision", ""])
    lines.extend(decision_text(interactions))

    lines.extend([
        "",
        "## Suggested Next Experiments",
        "",
        "1. Do not preregister `requirement_count >= 3` as a strong DBEC-gain predictor. At most, preregister it as an under-selection-risk predictor and use answer gain as a secondary endpoint.",
        "2. Do not switch directly to the candidate-multiplicity conjunction either: the exploratory scout is mixed and gold-controlled checks still look dataset-dependent.",
        "3. Keep HotpotQA as a retrospective negative-control diagnostic only unless a nontrivial high-risk slice exists in the held-out suffix.",
        "",
        "## Output Files",
        "",
        f"- Query rows: `{REPORT_DIR / 'risk_rows.csv'}`",
        f"- Slice summary: `{REPORT_DIR / 'risk_slice_summary.csv'}`",
        f"- Interaction statistics: `{REPORT_DIR / 'risk_interactions.csv'}`",
        f"- Gold-controlled diagnostics: `{REPORT_DIR / 'gold_controlled_risk_summary.csv'}`",
        f"- Exploratory feature scout: `{REPORT_DIR / 'exploratory_feature_scout.csv'}`",
        f"- Exploratory gold-controlled scout: `{REPORT_DIR / 'exploratory_gold_controlled_scout.csv'}`",
        f"- JSON payload: `{REPORT_DIR / 'summary.json'}`",
    ])
    return "\n".join(lines) + "\n"


def key_findings(
    summary_rows: Sequence[Mapping[str, Any]],
    interactions: Sequence[Mapping[str, Any]],
    gold_interactions: Sequence[Mapping[str, Any]],
    scout_rows: Sequence[Mapping[str, Any]],
    scout_gold_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    by_dataset_risk = {
        (str(row["dataset"]), str(row["risk_group"])): row
        for row in summary_rows
    }
    by_interaction = {
        (str(row["dataset"]), str(row["metric"])): row
        for row in interactions
    }

    findings: list[str] = []
    for dataset in ("2Wiki", "MuSiQue", "HotpotQA"):
        low = by_dataset_risk.get((dataset, "low"), {})
        high = by_dataset_risk.get((dataset, "high"), {})
        if not high or not low:
            findings.append(f"{dataset}: one risk group is empty, so interaction evidence is unavailable.")
            continue
        findings.append(
            f"{dataset}: high-risk share is {format_percent(high.get('share'))}; SetR under-selection is "
            f"{format_percent(high.get('setr_under_select_rate'))} high vs "
            f"{format_percent(low.get('setr_under_select_rate'))} low, and DBEC-SetR dF1 is "
            f"{format_signed(high.get('delta_f1_dbec_minus_setr'))} high vs "
            f"{format_signed(low.get('delta_f1_dbec_minus_setr'))} low."
        )

    under = by_interaction.get(("2Wiki+MuSiQue", "SetR under-select rate: high - low"), {})
    f1 = by_interaction.get(("2Wiki+MuSiQue", "DBEC-SetR dF1 interaction: high - low"), {})
    findings.append(
        "2Wiki+MuSiQue combined interaction: SetR under-selection high-low delta is "
        f"{format_signed(under.get('delta_mean'))} with CI {ci_text(under.get('ci_low'), under.get('ci_high'))}; "
        "DBEC-SetR dF1 high-low delta is "
        f"{format_signed(f1.get('delta_mean'))} with CI {ci_text(f1.get('ci_low'), f1.get('ci_high'))}."
    )
    gold_notes = [
        f"{row['dataset']} {row['gold_slice']}: under high-low {format_signed(row['under_select_high_minus_low'])}, dF1 high-low {format_signed(row['df1_high_minus_low'])}"
        for row in gold_interactions
        if row["gold_slice"] in {"gold_doc_count=2", "gold_doc_count>=3"}
    ]
    if gold_notes:
        findings.append(
            "Gold-controlled check: " + "; ".join(gold_notes)
            + ". This suggests the unconditioned risk signal is largely a support-depth proxy, not yet a clean answer-gain predictor."
        )
    scout_by_key = {
        (str(row["rule"]), str(row["dataset"])): row
        for row in scout_rows
    }
    max_rule = scout_by_key.get(("requirement_count>=3_and_max_binding_candidates>=2", "2Wiki+MuSiQue"), {})
    dep_rule = scout_by_key.get(("requirement_count>=3_and_dependent_req_count>=2", "2Wiki+MuSiQue"), {})
    findings.append(
        "Exploratory rules remain mixed: `requirement_count>=3_and_max_binding_candidates>=2` gives combined dF1 high-low "
        f"{format_signed(max_rule.get('df1_high_minus_low'))}, while `requirement_count>=3_and_dependent_req_count>=2` gives "
        f"{format_signed(dep_rule.get('df1_high_minus_low'))}; neither resolves the gold-controlled confound consistently across datasets."
    )
    return findings


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    query_rows = build_query_rows()
    summary_rows = grouped_summary_rows(query_rows)
    interactions = interaction_rows(query_rows)
    gold_rows = gold_control_rows(query_rows)
    gold_interactions = gold_control_interactions(gold_rows)
    scout_rows = exploratory_scout_rows(query_rows)
    scout_gold_rows = exploratory_gold_control_scout_rows(query_rows)

    write_csv(query_rows, REPORT_DIR / "risk_rows.csv")
    write_csv(summary_rows, REPORT_DIR / "risk_slice_summary.csv")
    write_csv(interactions, REPORT_DIR / "risk_interactions.csv")
    write_csv(gold_rows, REPORT_DIR / "gold_controlled_risk_summary.csv")
    write_csv(gold_interactions, REPORT_DIR / "gold_controlled_interactions.csv")
    write_csv(scout_rows, REPORT_DIR / "exploratory_feature_scout.csv")
    write_csv(scout_gold_rows, REPORT_DIR / "exploratory_gold_controlled_scout.csv")

    payload = {
        "risk_definition": {
            "feature": "requirement_count",
            "threshold": RISK_THRESHOLD,
            "high_risk": f"requirement_count >= {RISK_THRESHOLD}",
            "low_risk": f"requirement_count < {RISK_THRESHOLD}",
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "query_rows": query_rows,
        "risk_slice_summary": summary_rows,
        "risk_interactions": interactions,
        "gold_controlled_risk_summary": gold_rows,
        "gold_controlled_interactions": gold_interactions,
        "exploratory_feature_scout": scout_rows,
        "exploratory_gold_controlled_scout": scout_gold_rows,
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md = build_markdown(
        query_rows,
        summary_rows,
        interactions,
        gold_rows,
        gold_interactions,
        scout_rows,
        scout_gold_rows,
    )
    (REPORT_DIR / "summary.md").write_text(summary_md, encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT_DIR / "summary.md"),
        "risk_rows": str(REPORT_DIR / "risk_rows.csv"),
        "risk_slice_summary": str(REPORT_DIR / "risk_slice_summary.csv"),
        "risk_interactions": str(REPORT_DIR / "risk_interactions.csv"),
        "gold_controlled_risk_summary": str(REPORT_DIR / "gold_controlled_risk_summary.csv"),
        "gold_controlled_interactions": str(REPORT_DIR / "gold_controlled_interactions.csv"),
        "exploratory_feature_scout": str(REPORT_DIR / "exploratory_feature_scout.csv"),
        "exploratory_gold_controlled_scout": str(REPORT_DIR / "exploratory_gold_controlled_scout.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
