#!/usr/bin/env python3
"""Compute fixed-pool support ceiling for DAEC-DAPG diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.daec_dapg.metrics import summarize_numeric_rows
from src.dpathrag.io import read_json, write_json, write_jsonl
from src.dpathrag.metrics import recall_at_k, support_complete_at_k


def load_pool(path: str, limit: int) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        data = list(data.get("records") or data.get("rows") or data.get("data") or [])
    rows = list(data)
    return rows[: int(limit)] if int(limit) > 0 else rows


def hop_bucket(record: dict[str, Any]) -> str:
    if record.get("hop_bucket") is not None:
        return str(record["hop_bucket"])
    gold_titles = list(record.get("gold_titles") or [])
    return str(len(gold_titles)) if gold_titles else "unknown"


def build_rows(records: list[dict[str, Any]], *, dataset: str, k_values: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        gold_titles = list(record.get("gold_titles") or [])
        pool_titles = list(record.get("pool_titles") or [])
        for k in k_values:
            rows.append(
                {
                    "qid": str(record.get("qid") or record.get("id") or record.get("query_idx") or idx),
                    "dataset": dataset,
                    "hop_bucket": hop_bucket(record),
                    "k": int(k),
                    "gold_count": len(gold_titles),
                    "pool_size": len(pool_titles),
                    "support_recall": recall_at_k(gold_titles, pool_titles, int(k)),
                    "support_complete": support_complete_at_k(gold_titles, pool_titles, int(k)),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["hop_bucket"]), int(row["k"]))].append(row)
    metrics = ["support_recall", "support_complete"]
    return {
        "rows": len(rows),
        "by_hop_k": {
            f"{hop}:k{k}": summarize_numeric_rows(items, metrics) | {"rows": len(items)}
            for (hop, k), items in sorted(grouped.items())
        },
    }


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# DAEC-DAPG Pool Ceiling",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Pool: `{payload['pool_json']}`",
        f"- Records: `{payload['record_count']}`",
        "",
        "| Hop:K | Rows | Support Recall | Support Complete |",
        "|---|---:|---:|---:|",
    ]
    for key, value in payload["summary"]["by_hop_k"].items():
        lines.append(f"| {key} | {value['rows']} | {value['support_recall']:.4f} | {value['support_complete']:.4f} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_k_values(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--k_values", default="5,20,100")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_json", default="reports/dpathrag/daec_dapg_pool_ceiling_20260428.json")
    parser.add_argument("--rows_jsonl", default="reports/dpathrag/daec_dapg_pool_ceiling_rows_20260428.jsonl")
    parser.add_argument("--output_md", default="reports/dpathrag/daec_dapg_pool_ceiling_20260428.md")
    args = parser.parse_args()

    records = load_pool(args.pool_json, int(args.limit))
    rows = build_rows(records, dataset=str(args.dataset), k_values=parse_k_values(args.k_values))
    payload = {
        "pool_json": str(args.pool_json),
        "dataset": str(args.dataset),
        "record_count": len(records),
        "k_values": parse_k_values(args.k_values),
        "summary": summarize(rows),
    }
    write_jsonl(rows, args.rows_jsonl)
    write_json(payload, args.output_json)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
