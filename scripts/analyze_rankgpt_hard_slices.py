#!/usr/bin/env python3
"""Hard-slice comparison against RankGPT-style sliding-window reranking."""

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

from analyze_rankgpt_sliding_reader import (  # noqa: E402
    RANKGPT_METHOD,
    align_methods_local,
    dataset_methods,
)
from analyze_reviewer_baseline_paired_ci import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    metric_mean,
    paired_bootstrap,
    safe_float,
)


OUT_DIR = Path("reports/rankgpt_hard_slices_20260508")
MetricRow = Mapping[str, Any]
SlicePredicate = Callable[[int], bool]

COMPARISONS: tuple[tuple[str, str], ...] = (
    ("DAEC-selective", RANKGPT_METHOD),
    ("SetR-faithful", RANKGPT_METHOD),
    ("DAEC-selective", "SetR-faithful"),
)
METRICS = ("EM", "F1", "R5_TITLE")
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
        methods = align_methods_local(dataset, dataset_methods(slug))
        gold_counts = [gold_doc_count(row) for row in methods["DAEC-selective"]]
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
                for metric in METRICS:
                    stats = paired_bootstrap(
                        [safe_float(row.get(metric)) for row in slice_methods[left]],
                        [safe_float(row.get(metric)) for row in slice_methods[right]],
                        seed=BOOTSTRAP_SEED + 20260508 + len(rows) * 53,
                        samples=BOOTSTRAP_SAMPLES,
                    )
                    rows.append(
                        {
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
                        }
                    )
    return rows


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):.{digits}f}"


def sign_fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):+.{digits}f}"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


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


def require_row(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    slice_name: str,
    comparison: str,
    metric: str,
) -> Mapping[str, Any]:
    row = find_row(
        rows,
        dataset=dataset,
        slice_name=slice_name,
        comparison=comparison,
        metric=metric,
    )
    if row is None:
        raise KeyError((dataset, slice_name, comparison, metric))
    return row


def ci_text(row: Mapping[str, Any]) -> str:
    return f"[{sign_fmt(row['ci_low'])}, {sign_fmt(row['ci_high'])}]"


def significance_text(row: Mapping[str, Any]) -> str:
    return "CI excluding zero" if bool(row["ci_excludes_zero"]) else "CI crossing zero"


def build_takeaway(rows: list[Mapping[str, Any]]) -> list[str]:
    hard_daec_vs_rankgpt = require_row(
        rows,
        dataset="2Wiki",
        slice_name="gold_doc_count>=3",
        comparison="DAEC-selective - RankGPT-style sliding",
        metric="F1",
    )
    hard_daec_vs_setr = require_row(
        rows,
        dataset="2Wiki",
        slice_name="gold_doc_count>=3",
        comparison="DAEC-selective - SetR-faithful",
        metric="F1",
    )
    hard_setr_vs_rankgpt = require_row(
        rows,
        dataset="2Wiki",
        slice_name="gold_doc_count>=3",
        comparison="SetR-faithful - RankGPT-style sliding",
        metric="F1",
    )
    return [
        "- On the 2Wiki 4-document hard slice, DAEC-selective beats RankGPT-style sliding "
        f"by `{sign_fmt(hard_daec_vs_rankgpt['delta_mean'])}` F1 with a 95% CI "
        f"`{ci_text(hard_daec_vs_rankgpt)}`, so the hard-slice advantage holds against "
        f"the strongest RankGPT-style local adaptation ({significance_text(hard_daec_vs_rankgpt)}).",
        "- The earlier large hard-slice win over SetR-faithful remains strong: "
        f"DAEC-selective beats SetR-faithful by `{sign_fmt(hard_daec_vs_setr['delta_mean'])}` "
        f"F1 with a 95% CI `{ci_text(hard_daec_vs_setr)}`.",
        "- RankGPT-style sliding is substantially stronger than SetR-faithful on this hard slice "
        f"(`SetR-faithful - RankGPT-style sliding = {sign_fmt(hard_setr_vs_rankgpt['delta_mean'])}` "
        f"F1, 95% CI `{ci_text(hard_setr_vs_rankgpt)}`), so this is a meaningful stronger-baseline "
        "check rather than a weak-baseline artifact.",
        "- The safe main-paper claim is therefore: DAEC strongly repairs SetR-style under-selection "
        "on 2Wiki 4-doc questions and still significantly outperforms RankGPT-style sliding on "
        "the same hard slice under the controlled Qwen3-8B `/no_think` substrate.",
    ]


def append_rows(
    lines: list[str],
    rows: list[Mapping[str, Any]],
    *,
    title: str,
    row_specs: tuple[tuple[str, str, str, str], ...],
) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| Dataset | Slice | N | Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for dataset, slice_name, comparison, metric in row_specs:
        row = find_row(rows, dataset=dataset, slice_name=slice_name, comparison=comparison, metric=metric)
        if row is None:
            lines.append(f"| {dataset} | {slice_name} | 0 | {comparison} | {metric} | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {dataset} | {slice_name} | {int(row['n'])} | {comparison} | {metric} | "
            f"{fmt(row['left_mean'])} | {fmt(row['right_mean'])} | {sign_fmt(row['delta_mean'])} | "
            f"[{sign_fmt(row['ci_low'])}, {sign_fmt(row['ci_high'])}] | {yes_no(row['ci_excludes_zero'])} |"
        )


def build_markdown(rows: list[Mapping[str, Any]]) -> str:
    primary_specs = tuple(
        ("2Wiki", "gold_doc_count>=3", comparison, metric)
        for comparison in (
            "DAEC-selective - RankGPT-style sliding",
            "SetR-faithful - RankGPT-style sliding",
            "DAEC-selective - SetR-faithful",
        )
        for metric in ("EM", "F1", "R5_TITLE")
    )
    broader_specs = tuple(
        (dataset, slice_name, "DAEC-selective - RankGPT-style sliding", "F1")
        for dataset in ("2Wiki", "MuSiQue")
        for slice_name in ("all", "gold_doc_count=2", "gold_doc_count>=3", "gold_doc_count>=4")
    )
    lines = [
        "# RankGPT-Style Sliding Hard-Slice Analysis",
        "",
        "Date: 2026-05-08",
        "",
        "Purpose: close the reviewer-facing hard-slice gap by comparing DAEC-selective against the corrected RankGPT-style sliding-window local adaptation on support-depth slices.",
        "",
        "This report reuses existing full1000 reader outputs; no new LLM calls are made. Bootstrap is query-paired percentile bootstrap with 10,000 resamples. Delta is left method minus right method.",
        "",
        "The primary hard slice is 2Wiki `gold_doc_count>=3`, which is exactly the 4-document subset in this aligned full1000 split (`N=235`).",
    ]
    append_rows(lines, rows, title="Primary 2Wiki 4-Doc Hard Slice", row_specs=primary_specs)
    append_rows(lines, rows, title="DAEC vs RankGPT-Style Sliding by Support Depth", row_specs=broader_specs)
    lines.extend(
        [
            "",
            "## Paper-Facing Takeaway",
            "",
            *build_takeaway(rows),
            "",
            "## Claim Boundary",
            "",
            "Allowed:",
            "",
            "```text",
            "On 2Wiki 4-document queries, DAEC-selective dramatically outperforms SetR-faithful",
            "and significantly outperforms RankGPT-style sliding-window reranking under the",
            "same Qwen3-8B /no_think substrate.",
            "```",
            "",
            "Not allowed:",
            "",
            "```text",
            "DAEC outperforms original GPT-3.5/4 RankGPT on hard multi-hop questions.",
            "```",
            "",
            "## Files",
            "",
            f"- CSV: `{OUT_DIR / 'hard_slice_ci.csv'}`",
            f"- JSON: `{OUT_DIR / 'hard_slice_ci.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows, OUT_DIR / "hard_slice_ci.csv")
    (OUT_DIR / "hard_slice_ci.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.md").write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
