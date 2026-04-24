#!/usr/bin/env python3
"""Summarize Layer-1 fixed-pool DAEC experiment reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _get_float(mapping: Dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def summarize_report(path: Path) -> Dict[str, Any]:
    data = json.load(path.open())
    overall = data.get("overall_recomputed") or {}
    oracle = data.get("oracle_select_qa") or {}
    selector = data.get("setwise_selector_qa") or {}
    external_pool = data.get("external_pool") or {}
    payload_retrieval = external_pool.get("payload_retrieval") or {}
    oracle_k100 = (oracle.get("sweep") or {}).get("K=100") or {}

    baseline_em = _get_float(overall, "ExactMatch")
    baseline_f1 = _get_float(overall, "F1")
    selector_em = _get_float(selector, "selector_EM")
    selector_f1 = _get_float(selector, "selector_F1")
    oracle_em = _get_float(oracle_k100, "oracle_select_EM")
    oracle_f1 = _get_float(oracle_k100, "oracle_select_F1")

    return {
        "dataset": data.get("dataset") or path.stem,
        "path": str(path),
        "source": external_pool.get("source") or "",
        "baseline_em": baseline_em,
        "baseline_f1": baseline_f1,
        "selector_em": selector_em,
        "selector_f1": selector_f1,
        "selector_delta_em": None if selector_em is None or baseline_em is None else selector_em - baseline_em,
        "selector_delta_f1": None if selector_f1 is None or baseline_f1 is None else selector_f1 - baseline_f1,
        "oracle_em": oracle_em,
        "oracle_f1": oracle_f1,
        "oracle_delta_em": None if oracle_em is None or baseline_em is None else oracle_em - baseline_em,
        "oracle_delta_f1": None if oracle_f1 is None or baseline_f1 is None else oracle_f1 - baseline_f1,
        "recall5": _get_float(overall, "Recall@5"),
        "recall20": _get_float(overall, "Recall@20"),
        "recall100": _get_float(overall, "Recall@100"),
        "payload_recall5": _get_float(payload_retrieval, "Recall@5"),
        "payload_recall20": _get_float(payload_retrieval, "Recall@20"),
        "payload_recall100": _get_float(payload_retrieval, "Recall@100"),
        "exact_doc_text_match_count": external_pool.get("exact_doc_text_match_count"),
        "title_fallback_match_count": external_pool.get("title_fallback_match_count"),
        "unmatched_doc_count": external_pool.get("unmatched_doc_count"),
    }


def iter_paths(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        else:
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Report JSON files or directories.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a markdown table.")
    args = parser.parse_args()

    rows = [summarize_report(path) for path in iter_paths(args.reports) if path.exists()]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    headers = [
        "Dataset", "Source", "Base EM", "Base F1", "DAEC EM", "DAEC F1",
        "Delta EM", "Delta F1", "Oracle F1", "Oracle Gap", "R@5", "R@20", "R@100",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            str(row["dataset"]),
            str(row["source"]),
            _fmt(row["baseline_em"]),
            _fmt(row["baseline_f1"]),
            _fmt(row["selector_em"]),
            _fmt(row["selector_f1"]),
            _fmt(row["selector_delta_em"]),
            _fmt(row["selector_delta_f1"]),
            _fmt(row["oracle_f1"]),
            _fmt(row["oracle_delta_f1"]),
            _fmt(row["recall5"]),
            _fmt(row["recall20"]),
            _fmt(row["recall100"]),
        ]
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
