#!/usr/bin/env python3
"""Run DAEC-DAPG Phase 2 same-substrate aggregation variants."""

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

from scripts.daec_dapg_phase1_local_absorption import load_pool
from src.dpathrag.daec_dapg.local import select_variant
from src.dpathrag.daec_dapg.metrics import noise_rate, summarize_numeric_rows
from src.dpathrag.data import normalize_text
from src.dpathrag.io import write_json, write_jsonl
from src.dpathrag.metrics import recall_at_k, support_complete_at_k


VARIANTS = ["query_level_ppr", "mixed_source_absorption", "multi_channel_sum", "local_absorption_kappa"]
VARIANT_LABEL = {"local_absorption_kappa": "multi_channel_noisy_or"}


def hop_bucket(record: dict[str, Any]) -> str:
    if record.get("hop_bucket") is not None:
        return str(record["hop_bucket"])
    return str(len(record.get("gold_titles") or [])) or "unknown"


def source_definition(variant: str) -> str:
    return {
        "query_level_ppr": "source=g(q,v), no decomposition, no binding",
        "mixed_source_absorption": "source=sum_i pi_i s_i^0 before absorption",
        "multi_channel_sum": "score(d)=sum_i pi_i phi[i,d] before selection",
        "local_absorption_kappa": "preserve phi[i,d] until noisy-OR projection",
    }[variant]


def aggregation_timing(variant: str) -> str:
    return "delayed" if variant == "local_absorption_kappa" else "early"


def run(records: list[dict[str, Any]], *, dataset: str, top_k: int, max_docs: int, alpha: float, horizon: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        gold_titles = list(record.get("gold_titles") or [])
        for variant in VARIANTS:
            selected = select_variant(record, variant=variant, top_k=top_k, max_docs=max_docs, alpha=alpha, horizon=horizon)
            label = VARIANT_LABEL.get(variant, variant)
            selected_titles = list(selected["selected_titles"])
            selected_gold = len({normalize_text(title) for title in selected_titles} & {normalize_text(title) for title in gold_titles})
            rows.append(
                {
                    "qid": record.get("qid") or record.get("id") or record.get("query_idx") or idx,
                    "dataset": dataset,
                    "hop_bucket": hop_bucket(record),
                    "variant": label,
                    "source_definition": source_definition(variant),
                    "aggregation_timing": aggregation_timing(variant),
                    "selected_titles": selected_titles,
                    "selected_doc_count": len(selected_titles),
                    "support_recall": recall_at_k(gold_titles, selected_titles, top_k),
                    "support_complete": support_complete_at_k(gold_titles, selected_titles, top_k),
                    "selected_gold_count": selected_gold,
                    "noise_rate": noise_rate(selected_gold, len(selected_titles)),
                    "em": 0.0,
                    "f1": 0.0,
                    "latency_ms": 0.0,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["hop_bucket"]), str(row["variant"]))].append(row)
    metrics = ["support_recall", "support_complete", "noise_rate", "em", "f1", "latency_ms"]
    return {
        "rows": len(rows),
        "by_hop_variant": {
            f"{hop}:{variant}": summarize_numeric_rows(items, metrics) | {"rows": len(items)}
            for (hop, variant), items in sorted(grouped.items())
        },
    }


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# DAEC-DAPG Phase 2 Same-Substrate Mechanism Test",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Rows: `{payload['summary']['rows']}`",
        "",
        "| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in payload["summary"]["by_hop_variant"].items():
        lines.append(f"| {key} | {value['rows']} | {value['support_recall']:.4f} | {value['support_complete']:.4f} | {value['noise_rate']:.4f} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_docs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--output_md", default="reports/dpathrag/daec_dapg_phase2_same_substrate_20260428.md")
    parser.add_argument("--output_json", default="reports/dpathrag/daec_dapg_phase2_same_substrate_20260428.json")
    parser.add_argument("--rows_jsonl", default="reports/dpathrag/daec_dapg_phase2_rows_20260428.jsonl")
    args = parser.parse_args()

    records = load_pool(args.pool_json, int(args.limit))
    rows = run(records, dataset=str(args.dataset), top_k=int(args.top_k), max_docs=int(args.max_docs), alpha=float(args.alpha), horizon=int(args.horizon))
    payload = {
        "pool_json": str(args.pool_json),
        "dataset": str(args.dataset),
        "limit": int(args.limit),
        "top_k": int(args.top_k),
        "max_docs": int(args.max_docs),
        "alpha": float(args.alpha),
        "horizon": int(args.horizon),
        "summary": summarize(rows),
    }
    write_jsonl(rows, args.rows_jsonl)
    write_json(payload, args.output_json)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
