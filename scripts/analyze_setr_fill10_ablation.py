#!/usr/bin/env python3
"""Analyze the SetR Fill@10 conversion-style ablation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_reviewer_baseline_paired_ci import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    metric_mean,
    occurrence_keys,
    paired_bootstrap,
    safe_float,
    title_multiset_recall,
    trace_metric_rows,
)
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score  # noqa: E402


OUT_DIR = Path("reports/setr_fill10_proprag_full1000_20260507")
DAEC_SELECTIVE_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")
SETR_FAITHFUL_DIR = Path("reports/setr_faithful_proprag_full1000_20260507")
SETR_FILL5_DIR = Path("reports/setr_full1000_20260503")
SETR_FILL10_RUN_DIR = Path("run_logs/setr_fill10_proprag_full1000_20260507")
TOP10_RETRIEVER_DIR = Path("reports/top10_retriever_proprag_full1000_20260507")

METHODS = (
    "DAEC-selective",
    "SetR-faithful",
    "SetR-Fill@5",
    "SetR-Fill@10",
    "Top-10 retriever",
)

COMPARISONS = (
    ("DAEC-selective", "SetR-faithful"),
    ("DAEC-selective", "SetR-Fill@5"),
    ("DAEC-selective", "SetR-Fill@10"),
    ("DAEC-selective", "Top-10 retriever"),
    ("SetR-faithful", "SetR-Fill@5"),
    ("SetR-faithful", "SetR-Fill@10"),
    ("SetR-Fill@5", "SetR-Fill@10"),
    ("SetR-Fill@10", "Top-10 retriever"),
)

METRICS = ("EM", "F1", "R5_TITLE", "R10_TITLE")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def daec_selective_path(slug: str) -> Path:
    return DAEC_SELECTIVE_DIR / f"{slug}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def setr_faithful_path(slug: str) -> Path:
    return SETR_FAITHFUL_DIR / f"{slug}_proprag_setr_k20_doc768_faithful.eval.json"


def setr_fill5_path(slug: str) -> Path:
    return SETR_FILL5_DIR / f"{slug}_proprag_setr_k20_doc768.eval.json"


def setr_fill10_path(slug: str) -> Path:
    return OUT_DIR / f"{slug}_proprag_setr_k20_doc768_fill10.eval.json"


def top10_retriever_path(slug: str) -> Path:
    return TOP10_RETRIEVER_DIR / f"{slug}_proprag_top10_retriever.eval.json"


def setr_fill10_pool_path(slug: str) -> Path:
    return SETR_FILL10_RUN_DIR / f"{slug}_proprag_setr_k20_doc768_fill10.selected_pool.json"


def examples_metric_rows_with_titles(data: Mapping[str, Any], *, reader_top_k: int) -> list[dict[str, Any]]:
    examples = [row for row in data.get("examples", []) if isinstance(row, Mapping)]
    gold_answers = [list(row.get("gold_answers") or []) for row in examples]
    predictions = [str(row.get("answer") or "") for row in examples]
    _, per_query_em = QAExactMatch(global_config=None).calculate_metric_scores(gold_answers, predictions)
    _, per_query_f1 = QAF1Score(global_config=None).calculate_metric_scores(gold_answers, predictions)
    rows: list[dict[str, Any]] = []
    for example, em_row, f1_row in zip(examples, per_query_em, per_query_f1):
        retrieval_trace = example.get("retrieval_trace") or {}
        titles = list(retrieval_trace.get("external_pool_titles") or [])[:reader_top_k]
        rows.append({
            "question": str(example.get("question") or ""),
            "EM": safe_float(em_row.get("ExactMatch")),
            "F1": safe_float(f1_row.get("F1")),
            "gold_titles": list(example.get("gold_titles") or []),
            "top_titles": titles[: min(5, reader_top_k)],
            "top10_titles": titles[: min(10, reader_top_k)],
        })
    return rows


def daec_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = trace_metric_rows(data, field="selector_metrics")
    for row in rows:
        row["top10_titles"] = list(row.get("top_titles") or [])
    return rows


def align_methods_local(dataset: str, methods: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    reference_keys = occurrence_keys(methods["DAEC-selective"])
    aligned: dict[str, list[dict[str, Any]]] = {}
    for name, rows in methods.items():
        keys = occurrence_keys(rows)
        if len(keys) != len(reference_keys):
            raise ValueError(f"{dataset}: {name} length={len(keys)}, expected={len(reference_keys)}")
        keyed = {key: row for key, row in zip(keys, rows)}
        missing = [key for key in reference_keys if key not in keyed]
        if missing:
            raise ValueError(f"{dataset}: {name} missing {len(missing)} occurrence-aligned questions")
        aligned[name] = [keyed[key] for key in reference_keys]

    reference_gold = [list(row.get("gold_titles") or []) for row in aligned["DAEC-selective"]]
    for rows in aligned.values():
        for row, gold_titles in zip(rows, reference_gold):
            if not row.get("gold_titles"):
                row["gold_titles"] = list(gold_titles)
            row["R5_TITLE"] = title_multiset_recall(list(gold_titles), list(row.get("top_titles") or []))
            top10_titles = list(row.get("top10_titles") or row.get("top_titles") or [])
            row["R10_TITLE"] = title_multiset_recall(list(gold_titles), top10_titles)
    return aligned


def dataset_methods(slug: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "DAEC-selective": daec_rows(read_json(daec_selective_path(slug))),
        "SetR-faithful": examples_metric_rows_with_titles(read_json(setr_faithful_path(slug)), reader_top_k=5),
        "SetR-Fill@5": examples_metric_rows_with_titles(read_json(setr_fill5_path(slug)), reader_top_k=5),
        "SetR-Fill@10": examples_metric_rows_with_titles(read_json(setr_fill10_path(slug)), reader_top_k=10),
        "Top-10 retriever": examples_metric_rows_with_titles(read_json(top10_retriever_path(slug)), reader_top_k=10),
    }


def gold_doc_count(row: Mapping[str, Any]) -> int:
    return len(list(row.get("gold_titles") or []))


def subset_rows(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [rows[index] for index in indices]


def build_mean_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, slug in DATASETS:
        methods = align_methods_local(dataset, dataset_methods(slug))
        for method in METHODS:
            rows.append({
                "dataset": dataset,
                "method": method,
                "n": len(methods[method]),
                "EM": metric_mean(methods[method], "EM"),
                "F1": metric_mean(methods[method], "F1"),
                "R5_TITLE": metric_mean(methods[method], "R5_TITLE"),
                "R10_TITLE": metric_mean(methods[method], "R10_TITLE"),
            })
    return rows


def paired_rows_for_methods(
    dataset: str,
    methods: Mapping[str, list[dict[str, Any]]],
    *,
    slice_name: str,
    seed_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in COMPARISONS:
        for metric in METRICS:
            stats = paired_bootstrap(
                [safe_float(row.get(metric)) for row in methods[left]],
                [safe_float(row.get(metric)) for row in methods[right]],
                seed=BOOTSTRAP_SEED + seed_offset + len(rows) * 41,
            )
            rows.append({
                "dataset": dataset,
                "slice": slice_name,
                "comparison": f"{left} - {right}",
                "metric": metric,
                "left_mean": metric_mean(methods[left], metric),
                "right_mean": metric_mean(methods[right], metric),
                **stats,
            })
    return rows


def build_paired_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, slug in DATASETS:
        methods = align_methods_local(dataset, dataset_methods(slug))
        rows.extend(paired_rows_for_methods(dataset, methods, slice_name="all", seed_offset=1009 + len(rows)))
        if dataset == "2Wiki":
            gold_counts = [gold_doc_count(row) for row in methods["DAEC-selective"]]
            hard_indices = [index for index, count in enumerate(gold_counts) if count >= 3]
            hard_methods = {
                name: subset_rows(method_rows, hard_indices)
                for name, method_rows in methods.items()
            }
            rows.extend(
                paired_rows_for_methods(
                    dataset,
                    hard_methods,
                    slice_name="gold_doc_count>=3",
                    seed_offset=9001 + len(rows),
                )
            )
    return rows


def build_fill10_selection_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, slug in DATASETS:
        data = read_json(setr_fill10_pool_path(slug))
        meta = data.get("setr_selection") or {}
        rows.append({
            "dataset": dataset,
            "records": int(meta.get("records_reordered") or 0),
            "parse_success": int(meta.get("parse_success_count") or 0),
            "parse_failure": int(meta.get("parse_failure_count") or 0),
            "empty_fallback": int(meta.get("empty_fallback_count") or 0),
            "avg_selected_count": safe_float(meta.get("avg_selected_count")),
            "avg_fallback_count": safe_float(meta.get("avg_fallback_count")),
            "avg_reader_pool_size": safe_float(meta.get("avg_reader_pool_size")),
            "fill_to_k": int(meta.get("fill_to_k") or 0),
        })
    return rows


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):.{digits}f}"


def sign_fmt(value: Any, digits: int = 4) -> str:
    numeric = safe_float(value)
    return f"{numeric:+.{digits}f}"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def find_row(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    slice_name: str,
    comparison: str,
    metric: str,
) -> Mapping[str, Any]:
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["slice"] == slice_name
            and row["comparison"] == comparison
            and row["metric"] == metric
        ):
            return row
    raise KeyError((dataset, slice_name, comparison, metric))


def build_markdown(
    mean_rows: list[Mapping[str, Any]],
    paired_rows: list[Mapping[str, Any]],
    selection_rows: list[Mapping[str, Any]],
) -> str:
    lines = [
        "# SetR-Fill@10 PropRAG Full1000",
        "",
        "Date: 2026-05-07",
        "",
        "Purpose: add a bounded `SetR-Fill@10` ablation for the official SetR repository's conversion-style behavior. The official `convert_rankify.py` uses `k=10`: parse ranks from `Final Selection`, then fill missing slots from the original top-20 in rank order until 10 contexts are written. Because the SetR README evaluation section is `TBD`, this is reported as an official conversion-style ablation, not as proof of the paper-faithful reader setting.",
        "",
        "All rows reuse the existing PropRAG full1000 SetR selection JSONL from `run_logs/setr_full1000_20260503`; no selector calls were rerun. Fill@10 changes only the conversion into the reader pool and the reader budget (`qa_top_k=10`).",
        "",
        "## Fill@10 Selection Sanity",
        "",
        "| Dataset | Records | Parse success | Parse failure | Empty fallback | Avg selected | Avg rank-fill | Avg reader passages |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selection_rows:
        lines.append(
            f"| {row['dataset']} | {row['records']} | {row['parse_success']} | {row['parse_failure']} | "
            f"{row['empty_fallback']} | {fmt(row['avg_selected_count'], 3)} | "
            f"{fmt(row['avg_fallback_count'], 3)} | {fmt(row['avg_reader_pool_size'], 1)} |"
        )

    lines.extend([
        "",
        "## Mean Metrics",
        "",
        "EM/F1 are answer metrics. R@5/R@10 are unified title-multiset support recall computed from each method's final reader titles. DAEC-selective has a strict 5-document reader set, so its R@10 equals R@5.",
        "",
        "| Dataset | Method | EM | F1 | R@5 title | R@10 title |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for dataset, _ in DATASETS:
        for method in METHODS:
            row = next(item for item in mean_rows if item["dataset"] == dataset and item["method"] == method)
            lines.append(
                f"| {dataset} | {method} | {fmt(row['EM'])} | {fmt(row['F1'])} | "
                f"{fmt(row['R5_TITLE'])} | {fmt(row['R10_TITLE'])} |"
            )

    lines.extend([
        "",
        "## DAEC-Selective vs SetR-Fill@10",
        "",
        "Query-paired percentile bootstrap with 10,000 resamples. Delta is DAEC-selective minus SetR-Fill@10.",
        "",
        "| Dataset | Metric | DAEC-selective | SetR-Fill@10 | Delta | 95% CI | P(delta > 0) | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for dataset, _ in DATASETS:
        for metric in ("EM", "F1", "R5_TITLE", "R10_TITLE"):
            row = find_row(
                paired_rows,
                dataset=dataset,
                slice_name="all",
                comparison="DAEC-selective - SetR-Fill@10",
                metric=metric,
            )
            lines.append(
                f"| {dataset} | {metric} | {fmt(row['left_mean'])} | {fmt(row['right_mean'])} | "
                f"{sign_fmt(row['delta_mean'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | "
                f"{fmt(row['p_delta_gt_0'], 3)} | {yes_no(row['ci_excludes_zero'])} |"
            )

    lines.extend([
        "",
        "## Top-10 Retriever Budget Probe",
        "",
        "`Top-10 retriever` gives the reader the original PropRAG rank top-10 with no LLM selection. This isolates the effect of larger reader budget plus retriever recall from SetR's selected-plus-fallback conversion.",
        "",
        "| Dataset | DAEC F1 | Top-10 F1 | SetR-Fill@10 F1 | dF1 DAEC-Top10 | 95% CI | dF1 Fill@10-Top10 | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, _ in DATASETS:
        daec_top10 = find_row(
            paired_rows,
            dataset=dataset,
            slice_name="all",
            comparison="DAEC-selective - Top-10 retriever",
            metric="F1",
        )
        fill10_top10 = find_row(
            paired_rows,
            dataset=dataset,
            slice_name="all",
            comparison="SetR-Fill@10 - Top-10 retriever",
            metric="F1",
        )
        lines.append(
            f"| {dataset} | {fmt(daec_top10['left_mean'])} | {fmt(daec_top10['right_mean'])} | "
            f"{fmt(fill10_top10['left_mean'])} | {sign_fmt(daec_top10['delta_mean'])} | "
            f"[{fmt(daec_top10['ci_low'])}, {fmt(daec_top10['ci_high'])}] | "
            f"{sign_fmt(fill10_top10['delta_mean'])} | "
            f"[{fmt(fill10_top10['ci_low'])}, {fmt(fill10_top10['ci_high'])}] |"
        )

    lines.extend([
        "",
        "## SetR Variant Deltas",
        "",
        "| Dataset | Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    variant_rows = [
        row for row in paired_rows
        if row["slice"] == "all"
        and row["metric"] == "F1"
        and row["comparison"] in {
            "SetR-faithful - SetR-Fill@5",
            "SetR-faithful - SetR-Fill@10",
            "SetR-Fill@5 - SetR-Fill@10",
            "SetR-Fill@10 - Top-10 retriever",
        }
    ]
    for row in variant_rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['metric']} | {fmt(row['left_mean'])} | "
            f"{fmt(row['right_mean'])} | {sign_fmt(row['delta_mean'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {yes_no(row['ci_excludes_zero'])} |"
        )

    lines.extend([
        "",
        "## 2Wiki 4-Doc Hard Slice",
        "",
        "The hard slice is the same pre-specified `gold_doc_count>=3` slice used in the SetR-faithful report. In the aligned 2Wiki full1000 split this is the 4-document subset (`N=235`).",
        "",
        "| Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    hard_comparisons = (
        "DAEC-selective - SetR-faithful",
        "DAEC-selective - SetR-Fill@5",
        "DAEC-selective - SetR-Fill@10",
        "SetR-Fill@5 - SetR-Fill@10",
    )
    for comparison in hard_comparisons:
        for metric in ("EM", "F1", "R5_TITLE", "R10_TITLE"):
            row = find_row(
                paired_rows,
                dataset="2Wiki",
                slice_name="gold_doc_count>=3",
                comparison=comparison,
                metric=metric,
            )
            lines.append(
                f"| {comparison} | {metric} | {fmt(row['left_mean'])} | {fmt(row['right_mean'])} | "
                f"{sign_fmt(row['delta_mean'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | "
                f"{yes_no(row['ci_excludes_zero'])} |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `SetR-Fill@10` closes the full-set 2Wiki answer gap almost completely: DAEC-selective F1 0.7118 vs Fill@10 F1 0.7114, paired CI crosses zero. This should be framed as DAEC top-5 matching the larger official-conversion-style SetR reader budget, not as a significant DAEC win.",
        "- On HotpotQA, DAEC-selective is higher than Fill@10 on F1 by +0.0073, but the paired CI crosses zero. Fill@10 is not monotonically stronger than Fill@5 here.",
        "- On MuSiQue, pure Top-10 retriever F1 is 0.4828, between DAEC-selective (0.4548) and SetR-Fill@10 (0.5035). Thus the Fill@10 advantage is partly a larger-reader-budget/retriever-recall effect, but not fully explained by raw Top-10 rank order.",
        "- SetR-Fill@10 beats Top-10 retriever by +0.0207 F1 on MuSiQue, showing that selected-plus-fallback ordering adds value under a 10-document budget. This does not change the fairness interpretation: the relevant DAEC-matched comparison remains Fill@5.",
        "- The 2Wiki 4-doc hard slice remains the strongest DAEC evidence against the faithful adaptive SetR setting and the budget-matched Fill@5 setting. Against Fill@10, the answer gap is no longer significant, but Fill@10 uses twice DAEC's reader budget.",
        "- Paper framing: primary SetR baseline is `SetR-faithful`; `SetR-Fill@5` is DAEC-budget-matched; `SetR-Fill@10` is official conversion-style and should be reported to close the repo-code attack point.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_rows = build_mean_rows()
    paired_rows = build_paired_rows()
    selection_rows = build_fill10_selection_rows()

    write_csv(mean_rows, OUT_DIR / "mean_metrics.csv")
    write_csv(paired_rows, OUT_DIR / "paired_ci.csv")
    write_csv(selection_rows, OUT_DIR / "fill10_selection_sanity.csv")
    (OUT_DIR / "paired_ci.json").write_text(
        json.dumps(paired_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.md").write_text(
        build_markdown(mean_rows, paired_rows, selection_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "summary": str(OUT_DIR / "summary.md"),
        "mean_metrics": str(OUT_DIR / "mean_metrics.csv"),
        "paired_ci": str(OUT_DIR / "paired_ci.csv"),
        "selection_sanity": str(OUT_DIR / "fill10_selection_sanity.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
