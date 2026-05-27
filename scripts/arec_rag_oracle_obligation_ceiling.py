#!/usr/bin/env python3
"""Day-0 AREC oracle-obligation ceiling and IRCoT baseline gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.arec.smoke import (
    load_jsonl_map,
    load_pool_records,
    make_oracle_template,
    residual_row,
    summarize_numeric,
)
from src.dpathrag.arec.verifier import build_verifier
from src.dpathrag.io import write_json, write_jsonl


def write_markdown(payload: dict, path: str | Path) -> None:
    summary = payload.get("summary") or {}
    lines = [
        "# AREC-RAG Day 0 Oracle Obligation Ceiling",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Dataset: `{payload.get('dataset')}`",
        f"- Rows: `{summary.get('rows', 0)}`",
        f"- Verifier: `{payload.get('verifier_model')}`",
        f"- IRCoT prompt: `{payload.get('official_ircot_prompt_path_or_fallback_reason')}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| oracle_arec_missing_hit_rate | {summary.get('arec_missing_hit_rate', 0.0):.4f} |",
        f"| raw_question_missing_hit_rate | {summary.get('raw_question_missing_hit_rate', 0.0):.4f} |",
        f"| cot_missing_hit_rate | {summary.get('cot_missing_hit_rate', 0.0):.4f} |",
        f"| final_support_complete | {summary.get('final_support_complete', 0.0):.4f} |",
    ]
    if payload.get("reason"):
        lines.extend(["", f"Reason: {payload['reason']}"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_docs", type=int, default=100)
    parser.add_argument("--per_query_k", type=int, default=10)
    parser.add_argument("--oracle_obligations_jsonl", default="data/dpathrag/arec/oracle_obligations_musique_limit100.jsonl")
    parser.add_argument("--cot_queries_jsonl", default="")
    parser.add_argument("--ircot_prompt_path", default="")
    parser.add_argument("--verifier_backend", choices=["nli", "lexical_smoke"], default="nli")
    parser.add_argument("--verifier_model", default="microsoft/deberta-v3-base-mnli")
    parser.add_argument("--output_md", default="reports/dpathrag/arec_rag_oracle_ceiling_musique_limit100_20260428.md")
    parser.add_argument("--output_json", default="reports/dpathrag/arec_rag_oracle_ceiling_musique_limit100_20260428.json")
    parser.add_argument("--rows_jsonl", default="reports/dpathrag/arec_rag_oracle_ceiling_musique_limit100_rows_20260428.jsonl")
    args = parser.parse_args()

    records = load_pool_records(args.pool_json, int(args.limit))
    oracle_path = Path(args.oracle_obligations_jsonl)
    if not oracle_path.exists():
        template = make_oracle_template(records, limit=int(args.limit))
        write_jsonl(template, oracle_path)
        payload = {
            "status": "needs_oracle_obligations",
            "dataset": str(args.dataset),
            "pool_json": str(args.pool_json),
            "oracle_template_jsonl": str(oracle_path),
            "reason": "Oracle obligations are required before Day-0 ceiling can run. A template was written.",
            "summary": {"rows": 0},
        }
        write_json(payload, args.output_json)
        write_markdown(payload, args.output_md)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if not args.ircot_prompt_path and not args.cot_queries_jsonl:
        payload = {
            "status": "needs_ircot_prompt_or_queries",
            "dataset": str(args.dataset),
            "pool_json": str(args.pool_json),
            "oracle_obligations_jsonl": str(oracle_path),
            "reason": "Official IRCoT prompt/config or frozen CoT query JSONL is required before AREC comparison.",
            "summary": {"rows": 0},
        }
        write_json(payload, args.output_json)
        write_markdown(payload, args.output_md)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    verifier = build_verifier(args.verifier_backend, model_name=str(args.verifier_model))
    oracle_map = load_jsonl_map(oracle_path)
    cot_map = load_jsonl_map(args.cot_queries_jsonl) if args.cot_queries_jsonl else {}
    rows = [
        residual_row(
            record,
            index=idx,
            dataset=str(args.dataset),
            top_k=int(args.top_k),
            max_docs=int(args.max_docs),
            per_query_k=int(args.per_query_k),
            obligation_map=oracle_map,
            cot_query_map=cot_map,
            verifier=verifier,
            mock_reader_from_gold=True,
            obligation_source="oracle",
        )
        for idx, record in enumerate(records)
    ]
    for row in rows:
        row["oracle_obligations"] = row.pop("obligations")
        row["oracle_arec_missing_hits"] = row.get("arec_missing_hits", 0)
    metrics = [
        "arec_missing_hit_rate",
        "raw_question_missing_hit_rate",
        "cot_missing_hit_rate",
        "final_support_complete",
        "final_support_recall",
    ]
    payload = {
        "status": "completed",
        "dataset": str(args.dataset),
        "pool_json": str(args.pool_json),
        "oracle_obligations_jsonl": str(oracle_path),
        "official_ircot_prompt_path_or_fallback_reason": str(args.ircot_prompt_path or args.cot_queries_jsonl),
        "verifier_model": getattr(verifier, "name", str(args.verifier_model)),
        "verifier_backend": str(args.verifier_backend),
        "top_k": int(args.top_k),
        "per_query_k": int(args.per_query_k),
        "summary": summarize_numeric(rows, metrics) | {"rows": len(rows)},
    }
    write_jsonl(rows, args.rows_jsonl)
    write_json(payload, args.output_json)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
