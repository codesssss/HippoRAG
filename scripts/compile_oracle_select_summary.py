#!/usr/bin/env python3
"""Compile cross-dataset oracle-select sweep reports into a markdown summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _format_depth(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _extract_bucket_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    oracle = report.get("oracle_select_qa") or {}
    sweep = oracle.get("sweep") or {}
    rows: list[dict[str, Any]] = []
    for sweep_key, sweep_value in sorted(sweep.items(), key=lambda item: item[1].get("pool_k", 0)):
        row = {
            "K": sweep_value.get("pool_k"),
            "EM": sweep_value.get("oracle_select_EM"),
            "EM_delta": sweep_value.get("EM_delta"),
            "F1": sweep_value.get("oracle_select_F1"),
            "F1_delta": sweep_value.get("F1_delta"),
            "FS_in_pool": sweep_value.get("full_support_in_pool_rate"),
            "buckets": sweep_value.get("bucket_breakdown") or {},
        }
        rows.append(row)
    return rows


def _make_dataset_section(report_path: Path, report: dict[str, Any]) -> list[str]:
    dataset = report.get("dataset", report_path.stem)
    overall = report.get("overall_recomputed") or {}
    oracle = report.get("oracle_select_qa") or {}
    support_depth = oracle.get("support_depth") or {}
    sweep_rows = _extract_bucket_table(report)

    lines = [
        f"## {dataset}",
        "",
        f"- Report: `{report_path}`",
        f"- Queries: `{overall.get('num_queries', report.get('limit', '-'))}`",
        f"- Baseline EM/F1: `{_format_float(overall.get('ExactMatch'))}` / `{_format_float(overall.get('F1'))}`",
        f"- Baseline Recall@20 / Recall@100: `{_format_float(overall.get('Recall@20'))}` / `{_format_float(overall.get('Recall@100'))}`",
        "",
    ]

    if support_depth:
        lines.extend(["### Support Depth", "", "| Bucket | Count | Fully Supported | Median | Mean | P90 | Max |", "|---|---:|---:|---:|---:|---:|---:|"])
        for bucket_name, bucket_stats in support_depth.items():
            lines.append(
                "| {bucket} | {count} | {supported} | {median} | {mean} | {p90} | {max_depth} |".format(
                    bucket=bucket_name,
                    count=bucket_stats.get("count", "-"),
                    supported=f"{bucket_stats.get('fully_supported', '-')}/{bucket_stats.get('count', '-')}",
                    median=_format_depth(bucket_stats.get("median_depth")),
                    mean=_format_depth(bucket_stats.get("mean_depth")),
                    p90=_format_depth(bucket_stats.get("p90_depth")),
                    max_depth=_format_depth(bucket_stats.get("max_depth")),
                )
            )
        lines.append("")

    if sweep_rows:
        lines.extend(["### Oracle Select Sweep", "", "| K | EM | ΔEM | F1 | ΔF1 | FS@K |", "|---:|---:|---:|---:|---:|---:|"])
        for row in sweep_rows:
            lines.append(
                f"| {row['K']} | {_format_float(row['EM'])} | {_format_float(row['EM_delta'])} | "
                f"{_format_float(row['F1'])} | {_format_float(row['F1_delta'])} | {_format_float(row['FS_in_pool'])} |"
            )
        lines.append("")

        bucket_names = sorted({bucket_name for row in sweep_rows for bucket_name in row["buckets"].keys()})
        for bucket_name in bucket_names:
            lines.extend(
                [
                    f"### {bucket_name} Breakdown",
                    "",
                    "| K | Count | Avg Gold Found | FS | EM | F1 |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in sweep_rows:
                bucket = row["buckets"].get(bucket_name) or {}
                lines.append(
                    f"| {row['K']} | {bucket.get('count', '-')} | {_format_float(bucket.get('avg_gold_found_in_pool'), 3)} | "
                    f"{_format_float(bucket.get('full_support_rate'))} | {_format_float(bucket.get('EM'))} | {_format_float(bucket.get('F1'))} |"
                )
            lines.append("")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile oracle-select sweep reports into one markdown summary.")
    parser.add_argument("--reports", nargs="+", required=True, help="Paths to eval JSON reports.")
    parser.add_argument("--output_md", required=True, help="Markdown summary output path.")
    parser.add_argument("--output_json", default=None, help="Optional machine-readable aggregate JSON output.")
    args = parser.parse_args()

    report_paths = [Path(path) for path in args.reports]
    aggregate = []
    lines = ["# Oracle Select Cross-Dataset Summary", ""]

    for report_path in report_paths:
        report = _load_json(report_path)
        aggregate.append(
            {
                "report": str(report_path),
                "dataset": report.get("dataset", report_path.stem),
                "overall_recomputed": report.get("overall_recomputed") or {},
                "oracle_select_qa": report.get("oracle_select_qa") or {},
            }
        )
        lines.extend(_make_dataset_section(report_path, report))

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"reports": aggregate}, indent=2, ensure_ascii=False))

    print(
        json.dumps(
            {
                "output_md": str(output_md),
                "output_json": args.output_json,
                "num_reports": len(report_paths),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
