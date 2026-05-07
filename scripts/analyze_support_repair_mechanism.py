#!/usr/bin/env python3
"""Support-repair mechanism audit for DBEC vs SetR-faithful.

This is a reader-free, LLM-free diagnostic. It asks whether DBEC's answer gains
on prompt-only under-selection cases correspond to concrete support-chain repair:

* Does SetR-faithful omit annotated gold support titles?
* Does DBEC select those omitted supports?
* Is DBEC's answer gain concentrated in transitions from incomplete SetR support
  to complete/improved DBEC support?

Gold support titles are used only for analysis, not by either method.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score


REPORT_DIR = Path("reports/support_repair_mechanism_20260507")
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


def list_items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def mapping_items(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def dbec_path(slug: str) -> Path:
    return DBEC_DIR / f"{slug}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def setr_selected_pool_path(slug: str) -> Path:
    return SETR_RUN_DIR / f"{slug}_proprag_setr_k20_doc768_faithful.selected_pool.json"


def setr_eval_path(slug: str) -> Path:
    return SETR_REPORT_DIR / f"{slug}_proprag_setr_k20_doc768_faithful.eval.json"


def support_match(gold_titles: Sequence[Any], selected_titles: Sequence[Any]) -> dict[str, Any]:
    """Multiset title match from gold supports to selected titles."""
    gold_pairs = [
        (str(title), normalize_title(title))
        for title in gold_titles
        if normalize_title(title)
    ]
    selected_counts = Counter(
        normalize_title(title)
        for title in selected_titles
        if normalize_title(title)
    )

    matched: list[str] = []
    missing: list[str] = []
    for original_title, normalized in gold_pairs:
        if selected_counts[normalized] > 0:
            selected_counts[normalized] -= 1
            matched.append(original_title)
        else:
            missing.append(original_title)
    total = len(gold_pairs)
    hit_count = len(matched)
    return {
        "gold_count": total,
        "hit_count": hit_count,
        "missing_count": total - hit_count,
        "recall": float(hit_count / total) if total else 0.0,
        "complete": bool(total > 0 and hit_count == total),
        "matched_titles": matched,
        "missing_titles": missing,
    }


def recovered_missing_titles(
    setr_missing_titles: Sequence[Any],
    dbec_titles: Sequence[Any],
) -> list[str]:
    dbec_counts = Counter(
        normalize_title(title)
        for title in dbec_titles
        if normalize_title(title)
    )
    recovered: list[str] = []
    for title in setr_missing_titles:
        normalized = normalize_title(title)
        if normalized and dbec_counts[normalized] > 0:
            dbec_counts[normalized] -= 1
            recovered.append(str(title))
    return recovered


def dbec_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_index, trace in enumerate(list_items(data.get("setwise_selector_query_traces"))):
        if not isinstance(trace, Mapping):
            continue
        metrics = mapping_items(trace.get("selector_metrics"))
        selector_trace = mapping_items(trace.get("selector_trace"))
        requirements = [
            req for req in list_items(selector_trace.get("requirements"))
            if isinstance(req, Mapping)
        ]
        rows.append({
            "query_index": query_index,
            "question": str(trace.get("question") or ""),
            "gold_answers": list_items(trace.get("gold_answers")),
            "gold_titles": list_items(trace.get("gold_titles")),
            "gold_doc_count": safe_int(trace.get("gold_doc_count"), len(list_items(trace.get("gold_titles")))),
            "dbec_answer": str(trace.get("selector_answer") or ""),
            "dbec_em": safe_float(metrics.get("ExactMatch")),
            "dbec_f1": safe_float(metrics.get("F1")),
            "dbec_titles": list_items(trace.get("selector_top_titles")),
            "requirement_count": len(requirements),
            "dependent_req_count": sum(1 for req in requirements if list_items(req.get("depends_on"))),
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
        rows.append({
            "query_index": query_index,
            "question": str(record.get("question") or ""),
            "gold_titles": list_items(record.get("gold_titles")),
            "gold_doc_count": len(list_items(record.get("gold_titles"))),
            "setr_selected_count": selected_count,
            "setr_titles": list_items(record.get("pool_titles"))[:selected_count],
            "setr_selected_positions": list_items(trace.get("selected_positions")),
            "setr_parse_success": bool(trace.get("parse_success")),
            "setr_empty_fallback_used": bool(trace.get("empty_fallback_used")),
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
        rows.append({
            "query_index": query_index,
            "question": str(example.get("question") or ""),
            "setr_answer": str(example.get("answer") or ""),
            "setr_em": safe_float(em_row.get("ExactMatch")),
            "setr_f1": safe_float(f1_row.get("F1")),
        })
    return rows


def classify_recovery(setr_missing_count: int, recovered_count: int) -> str:
    if setr_missing_count <= 0:
        return "setr_no_missing_support"
    if recovered_count <= 0:
        return "dbec_recovers_none"
    if recovered_count >= setr_missing_count:
        return "dbec_recovers_all_missing"
    return "dbec_recovers_some_missing"


def transition_label(setr_complete: bool, dbec_complete: bool) -> str:
    left = "SetR complete" if setr_complete else "SetR incomplete"
    right = "DBEC complete" if dbec_complete else "DBEC incomplete"
    return f"{left} -> {right}"


def support_delta_label(setr_hit_count: int, dbec_hit_count: int) -> str:
    if dbec_hit_count > setr_hit_count:
        return "DBEC improves support coverage"
    if dbec_hit_count < setr_hit_count:
        return "DBEC loses support coverage"
    return "support coverage tied"


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
            setr_titles = list(pool_row.get("setr_titles") or [])
            dbec_titles = list(dbec_row.get("dbec_titles") or [])
            setr_support = support_match(gold_titles, setr_titles)
            dbec_support = support_match(gold_titles, dbec_titles)
            recovered_titles = recovered_missing_titles(setr_support["missing_titles"], dbec_titles)
            recovered_count = len(recovered_titles)
            setr_missing_count = safe_int(setr_support.get("missing_count"))
            setr_selected_count = safe_int(pool_row.get("setr_selected_count"))
            count_under_selected = setr_selected_count < gold_doc_count
            setr_complete = bool(setr_support.get("complete"))
            dbec_complete = bool(dbec_support.get("complete"))
            dbec_f1 = safe_float(dbec_row.get("dbec_f1"))
            setr_f1 = safe_float(eval_row.get("setr_f1"))
            rows.append({
                "dataset": dataset,
                "query_index": safe_int(dbec_row.get("query_index")),
                "question": str(dbec_row.get("question") or ""),
                "gold_doc_count": gold_doc_count,
                "gold_titles_json": json.dumps(gold_titles, ensure_ascii=False),
                "setr_selected_count": setr_selected_count,
                "count_under_selected": int(count_under_selected),
                "dbec_selected_count": len(dbec_titles),
                "setr_support_hit_count": safe_int(setr_support.get("hit_count")),
                "dbec_support_hit_count": safe_int(dbec_support.get("hit_count")),
                "setr_support_missing_count": setr_missing_count,
                "dbec_support_missing_count": safe_int(dbec_support.get("missing_count")),
                "setr_support_recall": safe_float(setr_support.get("recall")),
                "dbec_support_recall": safe_float(dbec_support.get("recall")),
                "delta_support_recall_dbec_minus_setr": (
                    safe_float(dbec_support.get("recall")) - safe_float(setr_support.get("recall"))
                ),
                "setr_support_complete": int(setr_complete),
                "dbec_support_complete": int(dbec_complete),
                "support_transition": transition_label(setr_complete, dbec_complete),
                "support_delta_label": support_delta_label(
                    safe_int(setr_support.get("hit_count")),
                    safe_int(dbec_support.get("hit_count")),
                ),
                "dbec_recovered_missing_count": recovered_count,
                "dbec_recovered_any_missing": int(recovered_count > 0),
                "dbec_recovered_all_missing": int(setr_missing_count > 0 and recovered_count >= setr_missing_count),
                "recovery_label": classify_recovery(setr_missing_count, recovered_count),
                "setr_missing_titles_json": json.dumps(setr_support["missing_titles"], ensure_ascii=False),
                "dbec_missing_titles_json": json.dumps(dbec_support["missing_titles"], ensure_ascii=False),
                "dbec_recovered_titles_json": json.dumps(recovered_titles, ensure_ascii=False),
                "setr_titles_json": json.dumps(setr_titles, ensure_ascii=False),
                "dbec_titles_json": json.dumps(dbec_titles, ensure_ascii=False),
                "setr_em": safe_float(eval_row.get("setr_em")),
                "setr_f1": setr_f1,
                "dbec_em": safe_float(dbec_row.get("dbec_em")),
                "dbec_f1": dbec_f1,
                "delta_em_dbec_minus_setr": safe_float(dbec_row.get("dbec_em")) - safe_float(eval_row.get("setr_em")),
                "delta_f1_dbec_minus_setr": dbec_f1 - setr_f1,
                "requirement_count": safe_int(dbec_row.get("requirement_count")),
                "dependent_req_count": safe_int(dbec_row.get("dependent_req_count")),
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
        return empty_stats()
    deltas = np.asarray(left_values, dtype=float) - np.asarray(right_values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = deltas[rng.integers(0, deltas.size, size=deltas.size)]
        boot[index] = float(np.mean(sampled))
    return {
        "n": int(deltas.size),
        "delta_mean": float(np.mean(deltas)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_delta_gt_0": float(np.mean(boot > 0.0)),
        "ci_excludes_zero": bool(np.percentile(boot, 2.5) > 0.0 or np.percentile(boot, 97.5) < 0.0),
    }


def empty_stats() -> dict[str, Any]:
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


def sum_int(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return int(sum(safe_int(row.get(field)) for row in rows))


def slice_predicates(dataset: str) -> tuple[tuple[str, Any], ...]:
    if dataset == "2Wiki":
        return (
            ("all", lambda row: True),
            ("gold_doc_count=2", lambda row: safe_int(row.get("gold_doc_count")) == 2),
            ("gold_doc_count>=4", lambda row: safe_int(row.get("gold_doc_count")) >= 4),
            ("SetR count-underselected", lambda row: bool(safe_int(row.get("count_under_selected")))),
            (
                "gold_doc_count>=4 and SetR count-underselected",
                lambda row: safe_int(row.get("gold_doc_count")) >= 4
                and bool(safe_int(row.get("count_under_selected"))),
            ),
        )
    if dataset == "MuSiQue":
        return (
            ("all", lambda row: True),
            ("gold_doc_count=2", lambda row: safe_int(row.get("gold_doc_count")) == 2),
            ("gold_doc_count>=3", lambda row: safe_int(row.get("gold_doc_count")) >= 3),
            ("SetR count-underselected", lambda row: bool(safe_int(row.get("count_under_selected")))),
            (
                "gold_doc_count>=3 and SetR count-underselected",
                lambda row: safe_int(row.get("gold_doc_count")) >= 3
                and bool(safe_int(row.get("count_under_selected"))),
            ),
        )
    return (
        ("all", lambda row: True),
        ("SetR count-underselected", lambda row: bool(safe_int(row.get("count_under_selected")))),
    )


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    stats = paired_bootstrap_delta(
        [safe_float(row.get("dbec_f1")) for row in rows],
        [safe_float(row.get("setr_f1")) for row in rows],
        seed=seed,
    )
    return {
        "n": len(rows),
        "mean_gold_doc_count": mean_float(rows, "gold_doc_count"),
        "mean_setr_selected_count": mean_float(rows, "setr_selected_count"),
        "count_under_select_rate": mean_float(rows, "count_under_selected"),
        "setr_support_recall": mean_float(rows, "setr_support_recall"),
        "dbec_support_recall": mean_float(rows, "dbec_support_recall"),
        "delta_support_recall": mean_float(rows, "delta_support_recall_dbec_minus_setr"),
        "setr_complete_rate": mean_float(rows, "setr_support_complete"),
        "dbec_complete_rate": mean_float(rows, "dbec_support_complete"),
        "setr_incomplete_to_dbec_complete_rate": mean_float(
            rows,
            "_setr_incomplete_to_dbec_complete",
        ),
        "dbec_recovered_any_missing_rate": mean_float(rows, "dbec_recovered_any_missing"),
        "dbec_recovered_all_missing_rate": mean_float(rows, "dbec_recovered_all_missing"),
        "mean_setr_missing_count": mean_float(rows, "setr_support_missing_count"),
        "mean_dbec_recovered_missing_count": mean_float(rows, "dbec_recovered_missing_count"),
        "setr_f1": mean_float(rows, "setr_f1"),
        "dbec_f1": mean_float(rows, "dbec_f1"),
        "delta_f1": stats["delta_mean"],
        "delta_f1_ci_low": stats["ci_low"],
        "delta_f1_ci_high": stats["ci_high"],
        "delta_f1_ci_excludes_zero": stats["ci_excludes_zero"],
    }


def add_derived_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["_setr_incomplete_to_dbec_complete"] = int(
            not bool(safe_int(row.get("setr_support_complete")))
            and bool(safe_int(row.get("dbec_support_complete")))
        )
    return rows


def build_slice_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for slice_name, predicate in slice_predicates(dataset):
            slice_rows = [row for row in dataset_rows if predicate(row)]
            if not slice_rows:
                continue
            output.append({
                "dataset": dataset,
                "slice": slice_name,
                **summarize_rows(slice_rows, seed=BOOTSTRAP_SEED + len(output) * 17),
            })
    return output


def build_transition_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for slice_name, predicate in slice_predicates(dataset):
            slice_rows = [row for row in dataset_rows if predicate(row)]
            if not slice_rows:
                continue
            transitions = sorted({str(row.get("support_transition")) for row in slice_rows})
            for transition in transitions:
                group_rows = [row for row in slice_rows if str(row.get("support_transition")) == transition]
                output.append({
                    "dataset": dataset,
                    "slice": slice_name,
                    "support_transition": transition,
                    **summarize_rows(group_rows, seed=BOOTSTRAP_SEED + 1000 + len(output) * 19),
                })
    return output


def build_recovery_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for slice_name, predicate in slice_predicates(dataset):
            slice_rows = [
                row for row in dataset_rows
                if predicate(row) and safe_int(row.get("setr_support_missing_count")) > 0
            ]
            if not slice_rows:
                continue
            for label in (
                "dbec_recovers_none",
                "dbec_recovers_some_missing",
                "dbec_recovers_all_missing",
            ):
                group_rows = [row for row in slice_rows if str(row.get("recovery_label")) == label]
                if not group_rows:
                    continue
                output.append({
                    "dataset": dataset,
                    "slice": slice_name,
                    "recovery_label": label,
                    **summarize_rows(group_rows, seed=BOOTSTRAP_SEED + 2000 + len(output) * 23),
                })
    return output


def build_count_underselected_recovery_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        dataset_rows = [
            row for row in rows
            if str(row.get("dataset")) == dataset and bool(safe_int(row.get("count_under_selected")))
        ]
        if not dataset_rows:
            continue
        for label in (
            "dbec_recovers_none",
            "dbec_recovers_some_missing",
            "dbec_recovers_all_missing",
            "setr_no_missing_support",
        ):
            group_rows = [row for row in dataset_rows if str(row.get("recovery_label")) == label]
            if not group_rows:
                continue
            output.append({
                "dataset": dataset,
                "slice": "SetR count-underselected",
                "recovery_label": label,
                **summarize_rows(group_rows, seed=BOOTSTRAP_SEED + 3000 + len(output) * 29),
            })
    return output


def build_case_rows(rows: Sequence[Mapping[str, Any]], *, limit_per_dataset: int = 12) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        dataset_rows = [
            row for row in rows
            if str(row.get("dataset")) == dataset
            and safe_int(row.get("setr_support_missing_count")) > 0
            and safe_int(row.get("dbec_recovered_missing_count")) > 0
        ]
        dataset_rows = sorted(
            dataset_rows,
            key=lambda row: (
                safe_float(row.get("delta_f1_dbec_minus_setr")),
                safe_int(row.get("dbec_recovered_missing_count")),
            ),
            reverse=True,
        )
        for row in dataset_rows[:limit_per_dataset]:
            output.append({
                "dataset": row["dataset"],
                "query_index": row["query_index"],
                "delta_f1_dbec_minus_setr": row["delta_f1_dbec_minus_setr"],
                "gold_doc_count": row["gold_doc_count"],
                "setr_selected_count": row["setr_selected_count"],
                "setr_support_recall": row["setr_support_recall"],
                "dbec_support_recall": row["dbec_support_recall"],
                "support_transition": row["support_transition"],
                "recovery_label": row["recovery_label"],
                "question": row["question"],
                "gold_titles_json": row["gold_titles_json"],
                "setr_missing_titles_json": row["setr_missing_titles_json"],
                "dbec_recovered_titles_json": row["dbec_recovered_titles_json"],
                "setr_titles_json": row["setr_titles_json"],
                "dbec_titles_json": row["dbec_titles_json"],
            })
    return output


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key.startswith("_"):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):.{digits}f}"


def sign_fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):+.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * safe_float(value):.{digits}f}%"


def ci_text(row: Mapping[str, Any], prefix: str = "delta_f1") -> str:
    return f"[{sign_fmt(row.get(prefix + '_ci_low'))}, {sign_fmt(row.get(prefix + '_ci_high'))}]"


def row_by(summary: Sequence[Mapping[str, Any]], dataset: str, slice_name: str) -> Mapping[str, Any]:
    for row in summary:
        if row.get("dataset") == dataset and row.get("slice") == slice_name:
            return row
    return {}


def build_markdown(
    slice_summary: Sequence[Mapping[str, Any]],
    transition_summary: Sequence[Mapping[str, Any]],
    recovery_summary: Sequence[Mapping[str, Any]],
    underselected_recovery_summary: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Support-Repair Mechanism Audit",
        "",
        "Date: 2026-05-07",
        "",
        "This diagnostic uses existing PropRAG full1000 DBEC-selective and SetR-faithful selected-only outputs. It makes no new LLM or reader calls. Gold support titles are used only to audit mechanism behavior.",
        "",
        "Core question: when SetR-faithful omits annotated support titles, does DBEC recover those supports, and is the answer gain concentrated in those repair transitions?",
        "",
        "## Dataset / Slice Summary",
        "",
        "| Dataset | Slice | N | Count under-select | SetR complete | DBEC complete | SetR R | DBEC R | dR | DBEC recovers any | DBEC F1 | SetR F1 | dF1 | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in slice_summary:
        lines.append(
            "| {dataset} | {slice} | {n} | {under} | {setr_complete} | {dbec_complete} | "
            "{setr_r} | {dbec_r} | {dr} | {recover_any} | {dbec_f1} | {setr_f1} | {df1} | {ci} |".format(
                dataset=row["dataset"],
                slice=row["slice"],
                n=row["n"],
                under=pct(row["count_under_select_rate"]),
                setr_complete=pct(row["setr_complete_rate"]),
                dbec_complete=pct(row["dbec_complete_rate"]),
                setr_r=fmt(row["setr_support_recall"], 3),
                dbec_r=fmt(row["dbec_support_recall"], 3),
                dr=sign_fmt(row["delta_support_recall"], 3),
                recover_any=pct(row["dbec_recovered_any_missing_rate"]),
                dbec_f1=fmt(row["dbec_f1"]),
                setr_f1=fmt(row["setr_f1"]),
                df1=sign_fmt(row["delta_f1"]),
                ci=ci_text(row),
            )
        )

    lines.extend([
        "",
        "## Support Completeness Transitions",
        "",
        "| Dataset | Slice | Transition | N | SetR R | DBEC R | dR | DBEC F1 | SetR F1 | dF1 | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    primary_transition_rows = [
        row for row in transition_summary
        if row["slice"] in {
            "all",
            "gold_doc_count>=4",
            "gold_doc_count>=3",
            "SetR count-underselected",
            "gold_doc_count>=4 and SetR count-underselected",
            "gold_doc_count>=3 and SetR count-underselected",
        }
    ]
    for row in primary_transition_rows:
        lines.append(
            "| {dataset} | {slice} | {transition} | {n} | {setr_r} | {dbec_r} | {dr} | {dbec_f1} | {setr_f1} | {df1} | {ci} |".format(
                dataset=row["dataset"],
                slice=row["slice"],
                transition=row["support_transition"],
                n=row["n"],
                setr_r=fmt(row["setr_support_recall"], 3),
                dbec_r=fmt(row["dbec_support_recall"], 3),
                dr=sign_fmt(row["delta_support_recall"], 3),
                dbec_f1=fmt(row["dbec_f1"]),
                setr_f1=fmt(row["setr_f1"]),
                df1=sign_fmt(row["delta_f1"]),
                ci=ci_text(row),
            )
        )

    lines.extend([
        "",
        "## Recovery Among SetR Support-Incomplete Cases",
        "",
        "| Dataset | Slice | Recovery label | N | Mean SetR missing | Mean DBEC recovered | DBEC complete | DBEC F1 | SetR F1 | dF1 | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in recovery_summary:
        if row["slice"] not in {"all", "gold_doc_count>=4", "gold_doc_count>=3", "SetR count-underselected"}:
            continue
        lines.append(
            "| {dataset} | {slice} | {label} | {n} | {missing} | {recovered} | {complete} | {dbec_f1} | {setr_f1} | {df1} | {ci} |".format(
                dataset=row["dataset"],
                slice=row["slice"],
                label=row["recovery_label"],
                n=row["n"],
                missing=fmt(row["mean_setr_missing_count"], 2),
                recovered=fmt(row["mean_dbec_recovered_missing_count"], 2),
                complete=pct(row["dbec_complete_rate"]),
                dbec_f1=fmt(row["dbec_f1"]),
                setr_f1=fmt(row["setr_f1"]),
                df1=sign_fmt(row["delta_f1"]),
                ci=ci_text(row),
            )
        )

    lines.extend([
        "",
        "## Count-Underselected Repair Categories",
        "",
        "| Dataset | Recovery label | N | Mean SetR missing | Mean DBEC recovered | DBEC complete | DBEC F1 | SetR F1 | dF1 | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in underselected_recovery_summary:
        lines.append(
            "| {dataset} | {label} | {n} | {missing} | {recovered} | {complete} | {dbec_f1} | {setr_f1} | {df1} | {ci} |".format(
                dataset=row["dataset"],
                label=row["recovery_label"],
                n=row["n"],
                missing=fmt(row["mean_setr_missing_count"], 2),
                recovered=fmt(row["mean_dbec_recovered_missing_count"], 2),
                complete=pct(row["dbec_complete_rate"]),
                dbec_f1=fmt(row["dbec_f1"]),
                setr_f1=fmt(row["setr_f1"]),
                df1=sign_fmt(row["delta_f1"]),
                ci=ci_text(row),
            )
        )

    lines.extend(["", "## Key Findings", ""])
    lines.extend(key_findings(slice_summary, transition_summary, recovery_summary))

    lines.extend([
        "",
        "## Paper-Facing Interpretation",
        "",
        "DBEC's gains are better described as support-chain repair than as universal reranking superiority: the strongest gains occur when SetR-faithful omits annotated supports and DBEC recovers them. However, this audit is still diagnostic rather than causal because it does not intervene on the reader context. The next stronger test is a counterfactual fill experiment: add oracle missing supports or DBEC repair documents to SetR's context and compare against rank-fill controls.",
        "",
        "## Example Repaired Cases",
        "",
        "| Dataset | Query | dF1 | Transition | Recovered titles |",
        "|---|---:|---:|---|---|",
    ])
    for row in case_rows[:12]:
        recovered_titles = ", ".join(json.loads(str(row["dbec_recovered_titles_json"])))
        question = str(row["question"]).replace("|", "\\|")
        if len(question) > 120:
            question = question[:117] + "..."
        lines.append(
            f"| {row['dataset']} | {row['query_index']} | {sign_fmt(row['delta_f1_dbec_minus_setr'])} | "
            f"{row['support_transition']} | {recovered_titles} |"
        )

    lines.extend([
        "",
        "## Output Files",
        "",
        f"- Query rows: `{REPORT_DIR / 'support_repair_rows.csv'}`",
        f"- Slice summary: `{REPORT_DIR / 'slice_summary.csv'}`",
        f"- Transition summary: `{REPORT_DIR / 'transition_summary.csv'}`",
        f"- Recovery summary: `{REPORT_DIR / 'recovery_summary.csv'}`",
        f"- Count-underselected recovery summary: `{REPORT_DIR / 'count_underselected_recovery_summary.csv'}`",
        f"- Case examples: `{REPORT_DIR / 'case_examples.csv'}`",
        f"- JSON payload: `{REPORT_DIR / 'summary.json'}`",
    ])
    return "\n".join(lines) + "\n"


def key_findings(
    slice_summary: Sequence[Mapping[str, Any]],
    transition_summary: Sequence[Mapping[str, Any]],
    recovery_summary: Sequence[Mapping[str, Any]],
) -> list[str]:
    findings: list[str] = []
    for dataset, slice_name in (
        ("2Wiki", "gold_doc_count>=4"),
        ("MuSiQue", "gold_doc_count>=3"),
        ("HotpotQA", "all"),
    ):
        row = row_by(slice_summary, dataset, slice_name)
        if not row:
            continue
        findings.append(
            f"{dataset} {slice_name}: SetR support-complete {pct(row.get('setr_complete_rate'))}, "
            f"DBEC support-complete {pct(row.get('dbec_complete_rate'))}, dR {sign_fmt(row.get('delta_support_recall'), 3)}, "
            f"dF1 {sign_fmt(row.get('delta_f1'))} with CI {ci_text(row)}."
        )

    for dataset, slice_name in (
        ("2Wiki", "gold_doc_count>=4"),
        ("MuSiQue", "gold_doc_count>=3"),
    ):
        repaired = [
            row for row in transition_summary
            if row.get("dataset") == dataset
            and row.get("slice") == slice_name
            and row.get("support_transition") == "SetR incomplete -> DBEC complete"
        ]
        if repaired:
            row = repaired[0]
            findings.append(
                f"{dataset} {slice_name}: the support-repair transition "
                f"`SetR incomplete -> DBEC complete` has N={row['n']} and dF1 {sign_fmt(row.get('delta_f1'))} "
                f"with CI {ci_text(row)}."
            )

    for dataset, slice_name in (
        ("2Wiki", "SetR count-underselected"),
        ("MuSiQue", "SetR count-underselected"),
    ):
        rows = [
            row for row in recovery_summary
            if row.get("dataset") == dataset
            and row.get("slice") == slice_name
            and row.get("recovery_label") in {"dbec_recovers_some_missing", "dbec_recovers_all_missing"}
        ]
        total_n = sum_int(rows, "n")
        if rows:
            weighted_df1 = sum(safe_float(row.get("delta_f1")) * safe_int(row.get("n")) for row in rows) / total_n
            findings.append(
                f"{dataset} count-underselected: DBEC recovers at least one SetR-missing support in N={total_n} "
                f"support-incomplete cases, with weighted dF1 {sign_fmt(weighted_df1)}."
            )

    findings.append(
        "This is stronger than the earlier under-selection slice because it verifies the concrete object being repaired: annotated support coverage. It is still not a causal intervention; use it to motivate counterfactual fill, not to claim proof."
    )
    return findings


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    query_rows = add_derived_flags(build_query_rows())
    slice_summary = build_slice_summary(query_rows)
    transition_summary = build_transition_summary(query_rows)
    recovery_summary = build_recovery_summary(query_rows)
    underselected_recovery_summary = build_count_underselected_recovery_summary(query_rows)
    case_rows = build_case_rows(query_rows)

    write_csv(query_rows, REPORT_DIR / "support_repair_rows.csv")
    write_csv(slice_summary, REPORT_DIR / "slice_summary.csv")
    write_csv(transition_summary, REPORT_DIR / "transition_summary.csv")
    write_csv(recovery_summary, REPORT_DIR / "recovery_summary.csv")
    write_csv(underselected_recovery_summary, REPORT_DIR / "count_underselected_recovery_summary.csv")
    write_csv(case_rows, REPORT_DIR / "case_examples.csv")

    payload = {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "slice_summary": slice_summary,
        "transition_summary": transition_summary,
        "recovery_summary": recovery_summary,
        "count_underselected_recovery_summary": underselected_recovery_summary,
        "case_examples": case_rows,
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md = build_markdown(
        slice_summary,
        transition_summary,
        recovery_summary,
        underselected_recovery_summary,
        case_rows,
    )
    (REPORT_DIR / "summary.md").write_text(summary_md, encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT_DIR / "summary.md"),
        "support_repair_rows": str(REPORT_DIR / "support_repair_rows.csv"),
        "slice_summary": str(REPORT_DIR / "slice_summary.csv"),
        "transition_summary": str(REPORT_DIR / "transition_summary.csv"),
        "recovery_summary": str(REPORT_DIR / "recovery_summary.csv"),
        "count_underselected_recovery_summary": str(REPORT_DIR / "count_underselected_recovery_summary.csv"),
        "case_examples": str(REPORT_DIR / "case_examples.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
