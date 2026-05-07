#!/usr/bin/env python3
"""Analyze counterfactual fill interventions for DBEC support repair.

This diagnostic tests whether SetR-faithful under-selection failures are mainly
reader-context budget failures or missing-support failures:

* rank_fill5: SetR selected docs plus rank-order filler to five docs.
* oracle_fill5: SetR selected docs plus SetR-missing gold supports, then rank fill.
* dbec_repair_fill5: SetR selected docs plus DBEC-selected repair docs, then rank fill.

The oracle condition is analysis-only and uses gold support titles after method
selection. It is not a deployable method.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score


REPORT_DIR = Path("reports/counterfactual_fill_mechanism_20260507")
RUN_DIR = Path("run_logs/counterfactual_fill_mechanism_20260507")
MANIFEST_PATH = RUN_DIR / "manifest.json"
SUPPORT_ROWS_PATH = Path("reports/support_repair_mechanism_20260507/support_repair_rows.csv")

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260507

BASELINE_METHODS = ("SetR-faithful", "DBEC-selective")
COUNTERFACTUAL_METHODS = ("rank_fill5", "oracle_fill5", "dbec_repair_fill5")
ALL_METHODS = BASELINE_METHODS + COUNTERFACTUAL_METHODS


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


def list_from_json(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else []


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def support_match(gold_titles: Sequence[Any], selected_titles: Sequence[Any]) -> dict[str, Any]:
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
    missing: list[str] = []
    hit_count = 0
    for original_title, normalized in gold_pairs:
        if selected_counts[normalized] > 0:
            selected_counts[normalized] -= 1
            hit_count += 1
        else:
            missing.append(original_title)
    total = len(gold_pairs)
    return {
        "hit_count": hit_count,
        "missing_count": total - hit_count,
        "recall": float(hit_count / total) if total else 0.0,
        "complete": int(total > 0 and hit_count == total),
        "missing_titles": missing,
    }


def read_support_rows() -> list[dict[str, Any]]:
    with SUPPORT_ROWS_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def eval_metric_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples = [row for row in list_items(data.get("examples")) if isinstance(row, Mapping)]
    gold_answers = [list_items(row.get("gold_answers")) for row in examples]
    predictions = [str(row.get("answer") or "") for row in examples]
    _, per_query_em = QAExactMatch(global_config=None).calculate_metric_scores(gold_answers, predictions)
    _, per_query_f1 = QAF1Score(global_config=None).calculate_metric_scores(gold_answers, predictions)

    rows: list[dict[str, Any]] = []
    for example, em_row, f1_row in zip(examples, per_query_em, per_query_f1):
        answer = str(example.get("answer") or "")
        rows.append({
            "question": str(example.get("question") or ""),
            "answer": answer,
            "em": safe_float(em_row.get("ExactMatch")),
            "f1": safe_float(f1_row.get("F1")),
            "empty_answer": int(not answer.strip()),
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
        return {
            "n": 0,
            "delta_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_delta_gt_0": 0.0,
            "ci_excludes_zero": False,
        }
    deltas = np.asarray(left_values, dtype=float) - np.asarray(right_values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = deltas[rng.integers(0, deltas.size, size=deltas.size)]
        boot[index] = float(np.mean(sampled))
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return {
        "n": int(deltas.size),
        "delta_mean": float(np.mean(deltas)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_delta_gt_0": float(np.mean(boot > 0.0)),
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def mean_float(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(mean([safe_float(row.get(field)) for row in rows])) if rows else 0.0


def method_prefix(method: str) -> str:
    if method == "SetR-faithful":
        return "setr"
    if method == "DBEC-selective":
        return "dbec"
    return method


def metric_value(row: Mapping[str, Any], method: str, metric: str) -> float:
    prefix = method_prefix(method)
    return safe_float(row.get(f"{prefix}_{metric}"))


def available_counterfactual_variants(manifest: Mapping[str, Any]) -> list[str]:
    variants: list[str] = []
    for dataset_entry in manifest.get("datasets", []) or []:
        for variant_entry in dataset_entry.get("variants", []) or []:
            variant = str(variant_entry.get("variant") or "")
            if variant and variant not in variants:
                variants.append(variant)
    return variants


def selected_support_rows(
    support_rows: Sequence[Mapping[str, Any]],
    dataset_label: str,
    selected_indices: Sequence[int],
) -> list[Mapping[str, Any]]:
    by_index = {
        safe_int(row.get("query_index")): row
        for row in support_rows
        if str(row.get("dataset")) == dataset_label
    }
    missing = [int(index) for index in selected_indices if int(index) not in by_index]
    if missing:
        raise ValueError(f"{dataset_label}: support rows missing selected indices {missing[:10]}")
    return [by_index[int(index)] for index in selected_indices]


def add_counterfactual_variant(
    base_rows: list[dict[str, Any]],
    *,
    variant: str,
    eval_json: Path,
    pool_json: Path,
) -> None:
    eval_rows = eval_metric_rows(read_json(eval_json))
    pool_records = list_items(read_json(pool_json).get("records"))
    if len(eval_rows) != len(base_rows):
        raise ValueError(f"{eval_json}: eval rows={len(eval_rows)}, expected={len(base_rows)}")
    if len(pool_records) != len(base_rows):
        raise ValueError(f"{pool_json}: pool records={len(pool_records)}, expected={len(base_rows)}")

    for row, eval_row, pool_record in zip(base_rows, eval_rows, pool_records):
        question_left = normalize_text(row.get("question"))
        question_right = normalize_text(eval_row.get("question"))
        if question_left != question_right:
            raise ValueError(
                f"{variant}: question mismatch for query_index={row.get('query_index')}: "
                f"{row.get('question')} != {eval_row.get('question')}"
            )
        titles = list_items(pool_record.get("pool_titles"))
        support = support_match(list_from_json(row.get("gold_titles_json")), titles)
        trace = pool_record.get("counterfactual_fill_trace")
        trace = trace if isinstance(trace, Mapping) else {}
        row[f"{variant}_answer"] = str(eval_row.get("answer") or "")
        row[f"{variant}_em"] = safe_float(eval_row.get("em"))
        row[f"{variant}_f1"] = safe_float(eval_row.get("f1"))
        row[f"{variant}_empty_answer"] = safe_int(eval_row.get("empty_answer"))
        row[f"{variant}_selected_count"] = len(titles)
        row[f"{variant}_support_hit_count"] = safe_int(support.get("hit_count"))
        row[f"{variant}_support_missing_count"] = safe_int(support.get("missing_count"))
        row[f"{variant}_support_recall"] = safe_float(support.get("recall"))
        row[f"{variant}_support_complete"] = safe_int(support.get("complete"))
        row[f"{variant}_actual_added_count"] = len(list_items(trace.get("actual_added_positions")))
        row[f"{variant}_rank_fill_count"] = len(list_items(trace.get("rank_fill_positions")))
        row[f"{variant}_titles_json"] = json.dumps(titles, ensure_ascii=False)


def build_counterfactual_rows() -> list[dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    support_rows = read_support_rows()
    all_rows: list[dict[str, Any]] = []
    for dataset_entry in manifest.get("datasets", []) or []:
        dataset_label = str(dataset_entry["label"])
        subset_dataset = str(dataset_entry["subset_dataset"])
        selected_indices = [int(index) for index in list_items(dataset_entry.get("selected_indices"))]
        support_subset = selected_support_rows(support_rows, dataset_label, selected_indices)

        base_rows: list[dict[str, Any]] = []
        for subset_index, support_row in enumerate(support_subset):
            gold_titles = list_from_json(support_row.get("gold_titles_json"))
            base_rows.append({
                "dataset": dataset_label,
                "subset_dataset": subset_dataset,
                "subset_index": subset_index,
                "query_index": safe_int(support_row.get("query_index")),
                "question": str(support_row.get("question") or ""),
                "gold_doc_count": safe_int(support_row.get("gold_doc_count")),
                "gold_titles_json": json.dumps(gold_titles, ensure_ascii=False),
                "setr_selected_count": safe_int(support_row.get("setr_selected_count")),
                "setr_em": safe_float(support_row.get("setr_em")),
                "setr_f1": safe_float(support_row.get("setr_f1")),
                "setr_support_recall": safe_float(support_row.get("setr_support_recall")),
                "setr_support_complete": safe_int(support_row.get("setr_support_complete")),
                "setr_support_missing_count": safe_int(support_row.get("setr_support_missing_count")),
                "setr_titles_json": str(support_row.get("setr_titles_json") or "[]"),
                "dbec_selected_count": safe_int(support_row.get("dbec_selected_count")),
                "dbec_em": safe_float(support_row.get("dbec_em")),
                "dbec_f1": safe_float(support_row.get("dbec_f1")),
                "dbec_support_recall": safe_float(support_row.get("dbec_support_recall")),
                "dbec_support_complete": safe_int(support_row.get("dbec_support_complete")),
                "dbec_support_missing_count": safe_int(support_row.get("dbec_support_missing_count")),
                "dbec_recovered_missing_count": safe_int(support_row.get("dbec_recovered_missing_count")),
                "dbec_titles_json": str(support_row.get("dbec_titles_json") or "[]"),
                "requirement_count": safe_int(support_row.get("requirement_count")),
                "dependent_req_count": safe_int(support_row.get("dependent_req_count")),
                "selective_binding_decision": str(support_row.get("selective_binding_decision") or ""),
            })

        for variant_entry in dataset_entry.get("variants", []) or []:
            variant = str(variant_entry["variant"])
            eval_json = REPORT_DIR / f"{subset_dataset}_{variant}.eval.json"
            pool_json = Path(str(variant_entry["pool_json"]))
            if not eval_json.exists():
                raise FileNotFoundError(f"Missing eval output: {eval_json}")
            if not pool_json.exists():
                raise FileNotFoundError(f"Missing pool output: {pool_json}")
            add_counterfactual_variant(
                base_rows,
                variant=variant,
                eval_json=eval_json,
                pool_json=pool_json,
            )
        all_rows.extend(base_rows)
    return all_rows


def build_method_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    datasets = sorted({str(row.get("dataset")) for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for method in ALL_METHODS:
            prefix = method_prefix(method)
            if not all(f"{prefix}_f1" in row for row in dataset_rows):
                continue
            output.append({
                "dataset": dataset,
                "method": method,
                "n": len(dataset_rows),
                "em": mean_float(dataset_rows, f"{prefix}_em"),
                "f1": mean_float(dataset_rows, f"{prefix}_f1"),
                "support_recall": mean_float(dataset_rows, f"{prefix}_support_recall"),
                "support_complete": mean_float(dataset_rows, f"{prefix}_support_complete"),
                "selected_count": mean_float(dataset_rows, f"{prefix}_selected_count"),
                "empty_answer_rate": (
                    mean_float(dataset_rows, f"{prefix}_empty_answer")
                    if f"{prefix}_empty_answer" in dataset_rows[0]
                    else ""
                ),
            })
    return output


def build_paired_ci(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("rank_fill5", "SetR-faithful"),
        ("oracle_fill5", "SetR-faithful"),
        ("dbec_repair_fill5", "SetR-faithful"),
        ("DBEC-selective", "SetR-faithful"),
        ("oracle_fill5", "rank_fill5"),
        ("dbec_repair_fill5", "rank_fill5"),
        ("oracle_fill5", "DBEC-selective"),
        ("dbec_repair_fill5", "DBEC-selective"),
    ]
    metrics = ("em", "f1", "support_recall", "support_complete")
    output: list[dict[str, Any]] = []
    datasets = sorted({str(row.get("dataset")) for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for left, right in comparisons:
            for metric in metrics:
                if not dataset_rows:
                    continue
                if not all(f"{method_prefix(left)}_{metric}" in row for row in dataset_rows):
                    continue
                if not all(f"{method_prefix(right)}_{metric}" in row for row in dataset_rows):
                    continue
                left_values = [metric_value(row, left, metric) for row in dataset_rows]
                right_values = [metric_value(row, right, metric) for row in dataset_rows]
                stats = paired_bootstrap_delta(
                    left_values,
                    right_values,
                    seed=BOOTSTRAP_SEED + len(output) * 31,
                )
                output.append({
                    "dataset": dataset,
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "left_mean": float(mean(left_values)) if left_values else 0.0,
                    "right_mean": float(mean(right_values)) if right_values else 0.0,
                    **stats,
                })
    return output


def build_gap_decomposition(method_summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    datasets = sorted({str(row.get("dataset")) for row in method_summary})
    for dataset in datasets:
        by_method = {
            str(row.get("method")): row
            for row in method_summary
            if str(row.get("dataset")) == dataset
        }
        setr = by_method.get("SetR-faithful")
        dbec = by_method.get("DBEC-selective")
        if not setr or not dbec:
            continue
        dbec_delta = safe_float(dbec.get("f1")) - safe_float(setr.get("f1"))
        for method in ("rank_fill5", "dbec_repair_fill5", "oracle_fill5"):
            row = by_method.get(method)
            if not row:
                continue
            method_delta = safe_float(row.get("f1")) - safe_float(setr.get("f1"))
            output.append({
                "dataset": dataset,
                "method": method,
                "method_minus_setr_f1": method_delta,
                "dbec_minus_setr_f1": dbec_delta,
                "share_of_dbec_gap": method_delta / dbec_delta if abs(dbec_delta) > 1e-12 else "",
            })
    return output


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value == "":
        return ""
    return f"{safe_float(value):.{digits}f}"


def sign_fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    return f"{number:+.{digits}f}"


def pct(value: Any) -> str:
    return f"{safe_float(value) * 100:.1f}%"


def ci_text(row: Mapping[str, Any]) -> str:
    return f"[{sign_fmt(row.get('ci_low'))}, {sign_fmt(row.get('ci_high'))}]"


def find_row(rows: Sequence[Mapping[str, Any]], *, dataset: str, method: str) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get("dataset")) == dataset and str(row.get("method")) == method:
            return row
    return {}


def build_markdown(
    rows: Sequence[Mapping[str, Any]],
    method_summary: Sequence[Mapping[str, Any]],
    paired_ci: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
) -> str:
    datasets = sorted({str(row.get("dataset")) for row in rows})
    lines: list[str] = [
        "# Counterfactual Fill Mechanism",
        "",
        "This analysis isolates whether SetR-faithful under-selection failures are caused by missing support documents rather than only by using fewer reader-context slots.",
        "",
        "- `rank_fill5`: SetR selected docs plus rank-order filler to five docs.",
        "- `dbec_repair_fill5`: SetR selected docs plus DBEC-selected repair docs, then rank fill.",
        "- `oracle_fill5`: SetR selected docs plus SetR-missing gold supports, then rank fill; diagnostic upper bound only.",
        "",
        "## Method Means",
        "",
        "| Dataset | Method | N | EM | F1 | Support recall | Support complete | Selected count | Empty answer |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        for method in ALL_METHODS:
            row = find_row(method_summary, dataset=dataset, method=method)
            if not row:
                continue
            empty = row.get("empty_answer_rate")
            lines.append(
                f"| {dataset} | {method} | {row['n']} | {fmt(row['em'])} | {fmt(row['f1'])} | "
                f"{fmt(row['support_recall'])} | {pct(row['support_complete'])} | "
                f"{fmt(row['selected_count'], 2)} | {pct(empty) if empty != '' else ''} |"
            )

    lines.extend([
        "",
        "## Paired F1 Deltas",
        "",
        "| Dataset | Comparison | dF1 | 95% CI | p(delta>0) |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in paired_ci:
        if str(row.get("metric")) != "f1":
            continue
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {sign_fmt(row['delta_mean'])} | "
            f"{ci_text(row)} | {fmt(row['p_delta_gt_0'], 3)} |"
        )

    lines.extend([
        "",
        "## Support-Repair Gap Accounting",
        "",
        "| Dataset | Method | dF1 vs SetR | DBEC dF1 vs SetR | Share of DBEC gap |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in gap_rows:
        share = row.get("share_of_dbec_gap")
        lines.append(
            f"| {row['dataset']} | {row['method']} | {sign_fmt(row['method_minus_setr_f1'])} | "
            f"{sign_fmt(row['dbec_minus_setr_f1'])} | {fmt(share) if share != '' else ''} |"
        )

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- If `rank_fill5` is close to SetR but `dbec_repair_fill5` improves, the failure is not just fewer documents; it is which missing documents are inserted.",
        "- If `oracle_fill5` is much higher than SetR, missing gold support is a real reader bottleneck on this slice.",
        "- If `dbec_repair_fill5` recovers a large share of the DBEC-selective gain, the DBEC advantage is largely explained by support-chain repair.",
        "- If `rank_fill5` also recovers most of the gain, the mechanism should be framed as budget under-fill rather than dependency-aware support repair.",
        "",
        "## Files",
        "",
        f"- Per-query rows: `{REPORT_DIR / 'counterfactual_rows.csv'}`",
        f"- Method means: `{REPORT_DIR / 'method_summary.csv'}`",
        f"- Paired CIs: `{REPORT_DIR / 'paired_ci.csv'}`",
        f"- Gap accounting: `{REPORT_DIR / 'gap_decomposition.csv'}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_counterfactual_rows()
    method_summary = build_method_summary(rows)
    paired_ci = build_paired_ci(rows)
    gap_rows = build_gap_decomposition(method_summary)

    write_csv(rows, REPORT_DIR / "counterfactual_rows.csv")
    write_csv(method_summary, REPORT_DIR / "method_summary.csv")
    write_csv(paired_ci, REPORT_DIR / "paired_ci.csv")
    write_csv(gap_rows, REPORT_DIR / "gap_decomposition.csv")
    write_json_payload = {
        "rows": rows,
        "method_summary": method_summary,
        "paired_ci": paired_ci,
        "gap_decomposition": gap_rows,
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(write_json_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "summary.md").write_text(
        build_markdown(rows, method_summary, paired_ci, gap_rows),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
