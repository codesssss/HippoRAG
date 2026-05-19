"""Summarize source-authorized vocab-strict reports.

This is an oracle/checkpoint utility, not the retrieval implementation.  It
keeps the high-score target explicit while the v18 branch is migrated out of
the legacy compare harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .contract import HIGH_SCORE_CONTRACT, HIGH_SCORE_LEGACY_VARIANT, PUBLIC_REPORT_ALIAS


DEFAULT_REPORT_ROOT = Path("outputs_source_authorized_vocab_strict_limit100_20260505")
DEFAULT_DATASETS = ("2wikimultihopqa", "musique", "hotpotqa")
CONTRACT_METADATA_KEYS = {
    "paper_facing_method_name",
    "public_report_alias",
    "legacy_variant",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(report: Mapping[str, Any], variant: str, key: str) -> float | None:
    variants = report.get("variants", {}) or {}
    row = variants.get(variant)
    if not isinstance(row, Mapping):
        return None
    retrieval = row.get("retrieval", {}) or {}
    value = retrieval.get(key)
    return None if value is None else float(value)


def target_variant_name(report: Mapping[str, Any]) -> str:
    variants = report.get("variants", {}) or {}
    if PUBLIC_REPORT_ALIAS in variants:
        return PUBLIC_REPORT_ALIAS
    if HIGH_SCORE_LEGACY_VARIANT in variants:
        return HIGH_SCORE_LEGACY_VARIANT
    return PUBLIC_REPORT_ALIAS


def target_runtime_profile(report: Mapping[str, Any]) -> Mapping[str, Any]:
    variants = report.get("variants", {}) or {}
    row = variants.get(target_variant_name(report))
    if not isinstance(row, Mapping):
        return {}
    runtime = (row.get("diagnostics", {}) or {}).get("runtime_profile", {}) or {}
    return runtime if isinstance(runtime, Mapping) else {}


def contract_value(report: Mapping[str, Any], key: str) -> Any:
    runtime = target_runtime_profile(report)
    if key == "composition":
        return runtime.get("composition")
    contract = runtime.get("source_contract", {}) or {}
    return contract.get(key) if isinstance(contract, Mapping) else None


def contract_mismatches(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []
    for key, expected in HIGH_SCORE_CONTRACT.items():
        if key in CONTRACT_METADATA_KEYS:
            continue
        actual = contract_value(report, str(key))
        if actual != expected:
            mismatches.append(
                {
                    "key": str(key),
                    "expected": expected,
                    "actual": actual,
                }
            )
    return mismatches


def report_path(report_root: Path, dataset: str, limit: int) -> Path:
    return (
        report_root
        / dataset
        / "reports"
        / f"{dataset}_graph_compare_limit{int(limit)}_retrieval.json"
    )


def summarize_reports(
    *,
    report_root: Path,
    datasets: Sequence[str],
    limit: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        path = report_path(report_root, dataset, limit)
        report = load_json(path)
        target_variant = target_variant_name(report)
        method_object = contract_value(report, "method_object")
        candidate_entrance = contract_value(report, "candidate_entrance")
        composition = target_runtime_profile(report).get("composition")
        mismatches = contract_mismatches(report)
        rows.append(
            {
                "dataset": dataset,
                "report": str(path),
                "target_variant": target_variant,
                "baseline_r5": metric(report, "hipporag_v2", "Recall@5"),
                "source_authorized_vocab_strict_r5": metric(report, target_variant, "Recall@5"),
                "source_authorized_vocab_strict_r10": metric(report, target_variant, "Recall@10"),
                "method_object": method_object,
                "candidate_entrance": candidate_entrance,
                "composition": composition,
                "contract_matches": not mismatches,
                "contract_mismatch_count": len(mismatches),
                "contract_mismatches": mismatches,
            }
        )
    return {
        "schema_version": 1,
        "public_report_alias": PUBLIC_REPORT_ALIAS,
        "high_score_legacy_variant": HIGH_SCORE_LEGACY_VARIANT,
        "high_score_contract": dict(HIGH_SCORE_CONTRACT),
        "rows": rows,
    }


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Dataset",
        "Target Variant",
        "HippoRAGv2 R@5",
        "Vocab-Strict R@5",
        "Vocab-Strict R@10",
        "Contract",
        "Mismatches",
    ]
    body: List[List[str]] = []
    for row in rows:
        body.append(
            [
                str(row.get("dataset", "")),
                str(row.get("target_variant", "")),
                f"{float(row.get('baseline_r5') or 0.0):.4f}",
                f"{float(row.get('source_authorized_vocab_strict_r5') or 0.0):.4f}",
                f"{float(row.get('source_authorized_vocab_strict_r10') or 0.0):.4f}",
                "match" if row.get("contract_matches") else "mismatch",
                str(int(row.get("contract_mismatch_count") or 0)),
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
        "# Source-Authorized Vocab-Strict Report Summary",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| public_report_alias | {payload.get('public_report_alias')} |",
        f"| high_score_legacy_variant | {payload.get('high_score_legacy_variant')} |",
        "",
        format_table(payload.get("rows", []) or []),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    datasets = tuple(item.strip() for item in str(args.datasets).split(",") if item.strip())
    payload = summarize_reports(
        report_root=Path(args.report_root),
        datasets=datasets,
        limit=int(args.limit),
    )
    print(format_table(payload["rows"]))
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        write_markdown(payload, Path(args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
