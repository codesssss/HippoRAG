#!/usr/bin/env python3
"""Cross-pool paired-CI comparison between DAEC-selective and SetR-style."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_daec_selective_cross_pool import (  # noqa: E402
    POOLS,
    daec_path,
    metric_rows,
    read_json,
    selective_path,
    write_csv,
)
from analyze_reviewer_baseline_paired_ci import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    examples_metric_rows,
    metric_mean,
    occurrence_keys,
    paired_bootstrap,
    safe_float,
    title_multiset_recall,
)


OUT_DIR = Path("reports/daec_setr_cross_pool_ci_20260508")
SETR_DIR = Path("reports/setr_full1000_20260503")
METHODS = ("Top-5", "DAEC", "DAEC-selective", "SetR-style k20")
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("DAEC-selective", "SetR-style k20"),
    ("DAEC", "SetR-style k20"),
    ("SetR-style k20", "Top-5"),
    ("DAEC-selective", "Top-5"),
)
METRICS = ("EM", "F1", "R5_TITLE")


def setr_path(slug: str, pool_slug: str) -> Path:
    return SETR_DIR / f"{slug}_{pool_slug}_setr_k20_doc768.eval.json"


def align_methods(
    *,
    pool: str,
    dataset: str,
    methods: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    reference_keys = occurrence_keys(methods["DAEC-selective"])
    aligned: dict[str, list[dict[str, Any]]] = {}
    for name, rows in methods.items():
        keys = occurrence_keys(rows)
        if len(keys) != len(reference_keys):
            raise ValueError(f"{pool}/{dataset}: {name} length={len(keys)}, expected={len(reference_keys)}")
        keyed = {key: row for key, row in zip(keys, rows)}
        missing = [key for key in reference_keys if key not in keyed]
        if missing:
            raise ValueError(f"{pool}/{dataset}: {name} missing {len(missing)} occurrence-aligned questions")
        aligned[name] = [keyed[key] for key in reference_keys]
    fill_reference_gold_and_recompute_r5(aligned)
    return aligned


def fill_reference_gold_and_recompute_r5(methods: Mapping[str, list[dict[str, Any]]]) -> None:
    reference_gold = [list(row.get("gold_titles") or []) for row in methods["DAEC-selective"]]
    for rows in methods.values():
        for row, gold_titles in zip(rows, reference_gold):
            if not row.get("gold_titles"):
                row["gold_titles"] = list(gold_titles)
            row["R5_TITLE"] = title_multiset_recall(list(gold_titles), list(row.get("top_titles") or []))


def load_pool_dataset(pool_name: str, pool_slug: str, selective_dir: Path, daec_dir: Path, dataset: str, slug: str) -> dict[str, list[dict[str, Any]]]:
    selective_data = read_json(selective_path(selective_dir, slug, pool_slug))
    daec_data = read_json(daec_path(daec_dir, slug, pool_slug))
    return align_methods(
        pool=pool_name,
        dataset=dataset,
        methods={
            "Top-5": metric_rows(selective_data, metric_field="baseline_metrics", title_field="baseline_top_titles"),
            "DAEC": metric_rows(daec_data, metric_field="selector_metrics", title_field="selector_top_titles"),
            "DAEC-selective": metric_rows(selective_data, metric_field="selector_metrics", title_field="selector_top_titles"),
            "SetR-style k20": examples_metric_rows(read_json(setr_path(slug, pool_slug))),
        },
    )


def build_method_summary(pool: str, dataset: str, methods: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows.append(
            {
                "pool": pool,
                "dataset": dataset,
                "method": method,
                "n": len(methods[method]),
                "EM": metric_mean(list(methods[method]), "EM"),
                "F1": metric_mean(list(methods[method]), "F1"),
                "R5_TITLE": metric_mean(list(methods[method]), "R5_TITLE"),
            }
        )
    return rows


def build_ci_rows(pool: str, dataset: str, methods: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in COMPARISONS:
        for metric in METRICS:
            stats = paired_bootstrap(
                [safe_float(row.get(metric)) for row in methods[left]],
                [safe_float(row.get(metric)) for row in methods[right]],
                seed=BOOTSTRAP_SEED + 20260508 + len(rows) * 59 + len(pool) * 997 + len(dataset) * 211,
                samples=BOOTSTRAP_SAMPLES,
            )
            rows.append(
                {
                    "pool": pool,
                    "dataset": dataset,
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "left_mean": metric_mean(list(methods[left]), metric),
                    "right_mean": metric_mean(list(methods[right]), metric),
                    **stats,
                }
            )
    return rows


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    for pool_name, pool_slug, selective_dir, daec_dir in POOLS:
        for dataset, slug in DATASETS:
            methods = load_pool_dataset(pool_name, pool_slug, selective_dir, daec_dir, dataset, slug)
            summary_rows.extend(build_method_summary(pool_name, dataset, methods))
            ci_rows.extend(build_ci_rows(pool_name, dataset, methods))
    return summary_rows, ci_rows


def fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):.{digits}f}"


def sign_fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):+.{digits}f}"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def ci_text(row: Mapping[str, Any]) -> str:
    return f"[{sign_fmt(row['ci_low'])}, {sign_fmt(row['ci_high'])}]"


def find_summary(
    rows: list[Mapping[str, Any]],
    *,
    pool: str,
    dataset: str,
    method: str,
) -> Mapping[str, Any]:
    for row in rows:
        if row["pool"] == pool and row["dataset"] == dataset and row["method"] == method:
            return row
    raise KeyError((pool, dataset, method))


def find_ci(
    rows: list[Mapping[str, Any]],
    *,
    pool: str,
    dataset: str,
    comparison: str,
    metric: str,
) -> Mapping[str, Any]:
    for row in rows:
        if row["pool"] == pool and row["dataset"] == dataset and row["comparison"] == comparison and row["metric"] == metric:
            return row
    raise KeyError((pool, dataset, comparison, metric))


def build_markdown(summary_rows: list[Mapping[str, Any]], ci_rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "# DAEC-Selective vs SetR-Style Cross-Pool Paired CI",
        "",
        "Date: 2026-05-08",
        "",
        "Purpose: close the reviewer-facing cross-pool baseline-CI gap by comparing DAEC-selective against the existing SetR-style `k20_doc768` full1000 runs on Dense, HippoRAG, and PropRAG pools.",
        "",
        "This report reuses existing outputs only; no new LLM calls are made. EM/F1 are answer metrics. `R5_TITLE` is recomputed uniformly as title-multiset support recall from final reader top-5 titles. Delta is left method minus right method; CIs use query-paired percentile bootstrap with 10,000 resamples.",
        "",
        "## Main F1 Summary",
        "",
        "| Pool | Dataset | Top-5 F1 | SetR-style F1 | DAEC-selective F1 | DAEC-selective - SetR F1 | 95% CI | SetR - Top-5 F1 | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pool, _, _, _ in POOLS:
        for dataset, _ in DATASETS:
            top5 = find_summary(summary_rows, pool=pool, dataset=dataset, method="Top-5")
            setr = find_summary(summary_rows, pool=pool, dataset=dataset, method="SetR-style k20")
            selective = find_summary(summary_rows, pool=pool, dataset=dataset, method="DAEC-selective")
            vs_setr = find_ci(ci_rows, pool=pool, dataset=dataset, comparison="DAEC-selective - SetR-style k20", metric="F1")
            setr_vs_top5 = find_ci(ci_rows, pool=pool, dataset=dataset, comparison="SetR-style k20 - Top-5", metric="F1")
            lines.append(
                f"| {pool} | {dataset} | {fmt(top5['F1'])} | {fmt(setr['F1'])} | {fmt(selective['F1'])} | "
                f"{sign_fmt(vs_setr['delta_mean'])} | {ci_text(vs_setr)} | "
                f"{sign_fmt(setr_vs_top5['delta_mean'])} | {ci_text(setr_vs_top5)} |"
            )

    lines.extend(
        [
            "",
            "## Detailed F1 CI",
            "",
            "| Pool | Dataset | Comparison | Left | Right | Delta | 95% CI | P(delta > 0) | Excludes 0 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ci_rows:
        if row["metric"] != "F1":
            continue
        lines.append(
            f"| {row['pool']} | {row['dataset']} | {row['comparison']} | "
            f"{fmt(row['left_mean'])} | {fmt(row['right_mean'])} | {sign_fmt(row['delta_mean'])} | "
            f"{ci_text(row)} | {fmt(row['p_delta_gt_0'])} | {yes_no(row['ci_excludes_zero'])} |"
        )

    vs_setr_rows = [
        row for row in ci_rows
        if row["comparison"] == "DAEC-selective - SetR-style k20" and row["metric"] == "F1"
    ]
    positive_vs_setr = sum(safe_float(row["delta_mean"]) > 0 for row in vs_setr_rows)
    significant_positive_vs_setr = sum(
        bool(row["ci_excludes_zero"]) and safe_float(row["delta_mean"]) > 0
        for row in vs_setr_rows
    )
    significant_negative_vs_setr = sum(
        bool(row["ci_excludes_zero"]) and safe_float(row["delta_mean"]) < 0
        for row in vs_setr_rows
    )

    r5_vs_setr_rows = [
        row for row in ci_rows
        if row["comparison"] == "DAEC-selective - SetR-style k20" and row["metric"] == "R5_TITLE"
    ]
    positive_r5_vs_setr = sum(safe_float(row["delta_mean"]) > 0 for row in r5_vs_setr_rows)
    significant_positive_r5_vs_setr = sum(
        bool(row["ci_excludes_zero"]) and safe_float(row["delta_mean"]) > 0
        for row in r5_vs_setr_rows
    )

    lines.extend(
        [
            "",
            "## Paper-Facing Takeaway",
            "",
            f"- DAEC-selective is higher than SetR-style k20 by mean F1 in `{positive_vs_setr}/9` pool-dataset cells.",
            f"- Significant positive F1 wins over SetR-style occur in `{significant_positive_vs_setr}/9` cells; significant negative F1 losses occur in `{significant_negative_vs_setr}/9` cells.",
            f"- DAEC-selective has higher title-multiset support R@5 than SetR-style in `{positive_r5_vs_setr}/9` cells, with significant positive R@5 differences in `{significant_positive_r5_vs_setr}/9` cells.",
            "- The correct cross-pool message is competitive/tie-range answer F1 against SetR-style, plus stronger same-pool Top-5 improvements for both methods. It is not a dominance result.",
            "",
            "## Claim Boundary",
            "",
            "Allowed:",
            "",
            "```text",
            "Across three retrieval pools, DAEC-selective and SetR-style k20 are in",
            "answer-F1 tie range in all 9 pool-dataset cells; DAEC-selective is",
            "mean-higher on 2Wiki, while SetR-style is mean-higher on most HotpotQA",
            "and MuSiQue cells.",
            "```",
            "",
            "Not allowed:",
            "",
            "```text",
            "DAEC-selective uniformly outperforms SetR-style across all pools and datasets.",
            "```",
            "",
            "Also not allowed:",
            "",
            "```text",
            "DAEC-selective consistently improves support R@5 over SetR-style.",
            "```",
            "",
            "## Files",
            "",
            f"- Summary CSV: `{OUT_DIR / 'method_summary.csv'}`",
            f"- Paired CI CSV: `{OUT_DIR / 'paired_ci.csv'}`",
            f"- JSON: `{OUT_DIR / 'summary.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, ci_rows = build_rows()
    write_csv(summary_rows, OUT_DIR / "method_summary.csv")
    write_csv(ci_rows, OUT_DIR / "paired_ci.csv")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "method_summary": summary_rows,
                "paired_ci": ci_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.md").write_text(
        build_markdown(summary_rows, ci_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(OUT_DIR), "summary_rows": len(summary_rows), "ci_rows": len(ci_rows)}, indent=2))


if __name__ == "__main__":
    main()
