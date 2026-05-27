#!/usr/bin/env python3
"""Day-1 AREC generated-obligation closure smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.arec.smoke import closure_row, load_jsonl_map, load_pool_records, summarize_by_hop, summarize_numeric
from src.dpathrag.arec.verifier import build_verifier
from src.dpathrag.io import write_json, write_jsonl


def write_markdown(payload: dict, path: str | Path) -> None:
    summary = payload.get("summary") or {}
    lines = [
        "# AREC-RAG Day 1 Closure Smoke",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Dataset: `{payload.get('dataset')}`",
        f"- Rows: `{summary.get('rows', 0)}`",
        f"- Verifier: `{payload.get('verifier_model')}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| closure_score_initial | {summary.get('closure_score_initial', 0.0):.4f} |",
        f"| oracle_closure_score_initial | {summary.get('oracle_closure_score_initial', 0.0):.4f} |",
        f"| initial_support_complete | {summary.get('initial_support_complete', 0.0):.4f} |",
        f"| active_obligation_count | {summary.get('active_obligation_count', 0.0):.4f} |",
        "",
        "## By Hop",
        "",
        "| Hop | Rows | Closure | Support Complete | Active Obligations |",
        "|---|---:|---:|---:|---:|",
    ]
    for hop, row in (payload.get("by_hop") or {}).items():
        lines.append(
            f"| {hop} | {row.get('rows', 0)} | {row.get('closure_score_initial', 0.0):.4f} | {row.get('initial_support_complete', 0.0):.4f} | {row.get('active_obligation_count', 0.0):.4f} |"
        )
    if payload.get("reason"):
        lines.extend(["", f"Reason: {payload['reason']}"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--generated_obligations_jsonl", required=True)
    parser.add_argument("--oracle_obligations_jsonl", default="")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_docs", type=int, default=100)
    parser.add_argument("--mock_reader_from_gold", action="store_true")
    parser.add_argument("--verifier_backend", choices=["nli", "lexical_smoke"], default="nli")
    parser.add_argument("--verifier_model", default="microsoft/deberta-v3-base-mnli")
    parser.add_argument("--output_md", default="reports/dpathrag/arec_rag_closure_musique_limit100_20260428.md")
    parser.add_argument("--output_json", default="reports/dpathrag/arec_rag_closure_musique_limit100_20260428.json")
    parser.add_argument("--rows_jsonl", default="reports/dpathrag/arec_rag_closure_musique_limit100_rows_20260428.jsonl")
    args = parser.parse_args()

    generated_path = Path(args.generated_obligations_jsonl)
    if not generated_path.exists():
        payload = {
            "status": "needs_generated_obligations",
            "dataset": str(args.dataset),
            "pool_json": str(args.pool_json),
            "generated_obligations_jsonl": str(generated_path),
            "reason": "Generated obligations are required for Day-1 closure smoke.",
            "summary": {"rows": 0},
        }
        write_json(payload, args.output_json)
        write_markdown(payload, args.output_md)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    records = load_pool_records(args.pool_json, int(args.limit))
    obligation_map = load_jsonl_map(generated_path)
    oracle_map = load_jsonl_map(args.oracle_obligations_jsonl) if args.oracle_obligations_jsonl else None
    verifier = build_verifier(args.verifier_backend, model_name=str(args.verifier_model))
    rows = [
        closure_row(
            record,
            index=idx,
            dataset=str(args.dataset),
            top_k=int(args.top_k),
            max_docs=int(args.max_docs),
            obligation_map=obligation_map,
            oracle_map=oracle_map,
            verifier=verifier,
            mock_reader_from_gold=bool(args.mock_reader_from_gold),
        )
        for idx, record in enumerate(records)
    ]
    keys = ["closure_score_initial", "oracle_closure_score_initial", "initial_support_complete", "initial_support_recall", "active_obligation_count"]
    payload = {
        "status": "completed",
        "dataset": str(args.dataset),
        "pool_json": str(args.pool_json),
        "generated_obligations_jsonl": str(generated_path),
        "oracle_obligations_jsonl": str(args.oracle_obligations_jsonl),
        "verifier_model": getattr(verifier, "name", str(args.verifier_model)),
        "verifier_backend": str(args.verifier_backend),
        "top_k": int(args.top_k),
        "summary": summarize_numeric(rows, keys) | {"rows": len(rows)},
        "by_hop": summarize_by_hop(rows, keys),
    }
    write_jsonl(rows, args.rows_jsonl)
    write_json(payload, args.output_json)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

