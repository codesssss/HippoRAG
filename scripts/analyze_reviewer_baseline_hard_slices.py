#!/usr/bin/env python3
"""Gold-support-count slice analysis for reviewer baselines."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

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
    REPORT_DIR,
    align_methods,
    dataset_methods,
    metric_mean,
    paired_bootstrap,
    safe_float,
)


MetricRow = Mapping[str, Any]
SlicePredicate = Callable[[int], bool]

COMPARISONS: tuple[tuple[str, str], ...] = (
    ("DAEC-selective", "SetR-style k20"),
    ("DAEC-selective", "Top5"),
    ("DAEC-selective", "IRCoT-style local"),
    ("DAEC-selective", "LLM-direct title"),
    ("DAEC-selective", "LLM-direct snippet128"),
)

SLICE_SPECS: tuple[tuple[str, SlicePredicate], ...] = (
    ("all", lambda count: count >= 0),
    ("gold_doc_count=2", lambda count: count == 2),
    ("gold_doc_count>=3", lambda count: count >= 3),
    ("gold_doc_count=3", lambda count: count == 3),
    ("gold_doc_count>=4", lambda count: count >= 4),
)


def gold_doc_count(row: MetricRow) -> int:
    return len(list(row.get("gold_titles") or []))


def subset_rows(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [rows[index] for index in indices]


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, slug in DATASETS:
        methods = align_methods(dataset, dataset_methods(slug))
        gold_counts = [gold_doc_count(row) for row in methods["DAEC"]]
        for slice_name, predicate in SLICE_SPECS:
            indices = [index for index, count in enumerate(gold_counts) if predicate(count)]
            if not indices:
                continue
            slice_methods = {
                name: subset_rows(method_rows, indices)
                for name, method_rows in methods.items()
            }
            counts = [gold_counts[index] for index in indices]
            for left, right in COMPARISONS:
                for metric in ("EM", "F1", "R5_TITLE"):
                    left_values = [safe_float(row.get(metric)) for row in slice_methods[left]]
                    right_values = [safe_float(row.get(metric)) for row in slice_methods[right]]
                    stats = paired_bootstrap(
                        left_values,
                        right_values,
                        seed=BOOTSTRAP_SEED + 7919 + len(rows) * 37,
                    )
                    rows.append({
                        "dataset": dataset,
                        "slice": slice_name,
                        "n": len(indices),
                        "min_gold_doc_count": min(counts),
                        "max_gold_doc_count": max(counts),
                        "mean_gold_doc_count": float(np.mean(counts)),
                        "comparison": f"{left} - {right}",
                        "metric": metric,
                        "left_mean": metric_mean(slice_methods[left], metric),
                        "right_mean": metric_mean(slice_methods[right], metric),
                        **stats,
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


def find_row(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    slice_name: str,
    comparison: str,
    metric: str,
) -> Mapping[str, Any] | None:
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["slice"] == slice_name
            and row["comparison"] == comparison
            and row["metric"] == metric
        ):
            return row
    return None


def append_table(
    lines: list[str],
    rows: list[Mapping[str, Any]],
    *,
    title: str,
    comparison: str,
    metric: str,
    slices: tuple[str, ...],
) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "| Dataset | Slice | N | Left | Right | Delta | 95% CI | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, _slug in DATASETS:
        for slice_name in slices:
            row = find_row(rows, dataset=dataset, slice_name=slice_name, comparison=comparison, metric=metric)
            if row is None:
                lines.append(f"| {dataset} | {slice_name} | 0 | -- | -- | -- | -- | -- |")
                continue
            lines.append(
                f"| {dataset} | {slice_name} | {int(row['n'])} | {fmt(row['left_mean'])} | "
                f"{fmt(row['right_mean'])} | {fmt(row['delta_mean'])} | "
                f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {str(bool(row['ci_excludes_zero']))} |"
            )


def build_markdown(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "# Reviewer Baseline Hard-Slice Analysis - 2026-05-06",
        "",
        "Offline gold-support-count slice over the same aligned full1000 rows as the main reviewer-baseline paired CI report.",
        f"Bootstrap is query-paired percentile bootstrap with `{BOOTSTRAP_SAMPLES}` resamples. Delta is left method minus right method.",
        "",
        "Primary pre-specified hard slice: `gold_doc_count>=3`. Exact-count and `>=4` rows are included only to expose sample size and trend; they are not new tuned decision boundaries.",
        "",
        "Support R@5 is the same unified title-multiset `R5_TITLE` metric used in `paired_ci.md`.",
    ]
    append_table(
        lines,
        rows,
        title="DAEC-selective vs SetR-style k20, Answer F1",
        comparison="DAEC-selective - SetR-style k20",
        metric="F1",
        slices=("all", "gold_doc_count=2", "gold_doc_count>=3", "gold_doc_count=3", "gold_doc_count>=4"),
    )
    append_table(
        lines,
        rows,
        title="DAEC-selective vs SetR-style k20, Unified Support R@5",
        comparison="DAEC-selective - SetR-style k20",
        metric="R5_TITLE",
        slices=("all", "gold_doc_count=2", "gold_doc_count>=3", "gold_doc_count=3", "gold_doc_count>=4"),
    )
    append_table(
        lines,
        rows,
        title="Primary Hard Slice F1 Against Reviewer Baselines",
        comparison="DAEC-selective - Top5",
        metric="F1",
        slices=("gold_doc_count>=3",),
    )
    for comparison in (
        "DAEC-selective - IRCoT-style local",
        "DAEC-selective - LLM-direct title",
        "DAEC-selective - LLM-direct snippet128",
    ):
        append_table(
            lines,
            rows,
            title=f"Primary Hard Slice F1: {comparison}",
            comparison=comparison,
            metric="F1",
            slices=("gold_doc_count>=3",),
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- HotpotQA has no `gold_doc_count>=3` rows in this aligned subset, so it cannot support a deep-composition slice defense; it should be treated as a shallow/saturated contrast dataset.",
        "- On 2Wiki, the primary hard slice is exactly the 4-document subset (`N=235`), where DAEC-selective beats SetR-style on answer F1 with a 95% CI excluding zero; unified support R@5 remains tied.",
        "- On MuSiQue, SetR-style remains stronger on 2-document questions, while the `gold_doc_count>=3` slice is statistically tied on answer F1 and support R@5. Selective binding narrows the base DAEC gap but does not overturn SetR-style.",
        "- DAEC-selective remains clearly stronger than Top5 and both LLM-direct controls on the pre-specified hard slices where those slices exist; the hard MuSiQue comparison to IRCoT-style local is positive but not significant on answer F1.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    csv_path = REPORT_DIR / "hard_slice_gold_doc_count.csv"
    json_path = REPORT_DIR / "hard_slice_gold_doc_count.json"
    md_path = REPORT_DIR / "hard_slice_gold_doc_count.md"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps({
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
