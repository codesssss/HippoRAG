"""Compare high-score oracle reports against direct reproduction reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .report_summary import DEFAULT_DATASETS, DEFAULT_REPORT_ROOT, format_table, summarize_reports


DEFAULT_REPRO_ROOT = Path("outputs_v18_repro_limit100_20260506")


def _row_by_dataset(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset")): row for row in payload.get("rows", []) or []}


def compare_reproduction(
    *,
    oracle_root: Path,
    repro_root: Path,
    datasets: Sequence[str],
    limit: int,
) -> Dict[str, Any]:
    oracle_payload = summarize_reports(report_root=oracle_root, datasets=datasets, limit=limit)
    repro_payload = summarize_reports(report_root=repro_root, datasets=datasets, limit=limit)
    oracle_rows = _row_by_dataset(oracle_payload)
    repro_rows = _row_by_dataset(repro_payload)

    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        oracle = oracle_rows[str(dataset)]
        repro = repro_rows[str(dataset)]
        oracle_r5 = float(oracle.get("source_authorized_vocab_strict_r5") or 0.0)
        repro_r5 = float(repro.get("source_authorized_vocab_strict_r5") or 0.0)
        rows.append(
            {
                "dataset": str(dataset),
                "oracle_variant": oracle.get("target_variant"),
                "repro_variant": repro.get("target_variant"),
                "hipporagv2_r5": repro.get("baseline_r5"),
                "oracle_r5": oracle_r5,
                "repro_r5": repro_r5,
                "delta_r5": round(repro_r5 - oracle_r5, 4),
                "oracle_contract_matches": bool(oracle.get("contract_matches")),
                "repro_contract_matches": bool(repro.get("contract_matches")),
            }
        )
    return {
        "schema_version": 1,
        "oracle_root": str(oracle_root),
        "repro_root": str(repro_root),
        "limit": int(limit),
        "rows": rows,
    }


def format_gap_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Dataset",
        "HippoRAGv2 R@5",
        "Oracle R@5",
        "Repro R@5",
        "Delta",
        "Oracle Variant",
        "Repro Variant",
        "Contracts",
    ]
    body: List[List[str]] = []
    for row in rows:
        body.append(
            [
                str(row.get("dataset", "")),
                f"{float(row.get('hipporagv2_r5') or 0.0):.4f}",
                f"{float(row.get('oracle_r5') or 0.0):.4f}",
                f"{float(row.get('repro_r5') or 0.0):.4f}",
                f"{float(row.get('delta_r5') or 0.0):+.4f}",
                str(row.get("oracle_variant", "")),
                str(row.get("repro_variant", "")),
                (
                    "match"
                    if row.get("oracle_contract_matches") and row.get("repro_contract_matches")
                    else "mismatch"
                ),
            ]
        )
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in body))
        for col in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[col].ljust(widths[col]) for col in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row[col].ljust(widths[col]) for col in range(len(headers))) + " |")
    return "\n".join(lines)


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Source-Authorized Vocab-Strict Reproduction Gap",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| oracle_root | {payload.get('oracle_root')} |",
        f"| repro_root | {payload.get('repro_root')} |",
        f"| limit | {payload.get('limit')} |",
        "",
        format_gap_table(payload.get("rows", []) or []),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--repro-root", default=str(DEFAULT_REPRO_ROOT))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    datasets = tuple(item.strip() for item in str(args.datasets).split(",") if item.strip())
    payload = compare_reproduction(
        oracle_root=Path(args.oracle_root),
        repro_root=Path(args.repro_root),
        datasets=datasets,
        limit=int(args.limit),
    )
    print(format_gap_table(payload["rows"]))
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        write_markdown(payload, Path(args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
