#!/usr/bin/env python3
"""Build reader-baseline JSONL records for FiD/Flan experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import read_json, write_json, write_jsonl
from src.dpathrag.reader_data import build_gold_reader_record, build_pool_reader_record, summarize_reader_records


def load_samples(path: str | Path, limit: int) -> list[dict[str, Any]]:
    rows = list(read_json(path))
    return rows[: int(limit)] if int(limit) > 0 else rows


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = load_samples(args.samples_json, int(args.limit))
    if args.mode == "gold":
        return [
            build_gold_reader_record(
                sample,
                split=str(args.split),
                top_k=int(args.top_k),
                max_doc_chars=int(args.max_doc_chars),
            )
            for sample in samples
        ]
    if args.mode == "pool":
        if not args.pool_json:
            raise ValueError("--pool_json is required when --mode pool")
        pool_payload = read_json(args.pool_json)
        records = list(pool_payload.get("records") or [])
        if int(args.limit) > 0:
            records = records[: int(args.limit)]
        if len(records) < len(samples):
            raise ValueError(f"Pool has fewer records ({len(records)}) than samples ({len(samples)})")
        return [
            build_pool_reader_record(
                sample,
                record,
                split=str(args.split),
                source=str(args.source),
                top_k=int(args.top_k),
                max_doc_chars=int(args.max_doc_chars),
                strict_question=not bool(args.no_strict_question),
            )
            for sample, record in zip(samples, records)
        ]
    raise ValueError(f"Unsupported mode: {args.mode}")


def default_manifest_path(output_jsonl: str | Path) -> Path:
    path = Path(output_jsonl)
    return path.with_suffix(path.suffix + ".manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_json", default="data/dpathrag/full_2wiki/2wikimultihopqa_validation.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--mode", choices=["gold", "pool"], default="gold")
    parser.add_argument("--pool_json", default="")
    parser.add_argument("--source", default="gold")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--output_jsonl", default="data/dpathrag/reader_baselines/2wiki_validation_gold_k5.jsonl")
    parser.add_argument("--manifest_json", default="")
    parser.add_argument("--no_strict_question", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args)
    write_jsonl(rows, args.output_jsonl)
    manifest = {
        "samples_json": str(args.samples_json),
        "pool_json": str(args.pool_json) if args.pool_json else "",
        "split": str(args.split),
        "mode": str(args.mode),
        "source": str(args.source),
        "top_k": int(args.top_k),
        "limit": int(args.limit),
        "max_doc_chars": int(args.max_doc_chars),
        "output_jsonl": str(args.output_jsonl),
        "summary": summarize_reader_records(rows),
        "schema": {
            "selected_docs": ["doc_id", "rank", "title", "text", "source", "retriever_score", "gold_support"],
            "metrics": ["support_recall", "support_complete"],
        },
    }
    manifest_path = Path(args.manifest_json) if args.manifest_json else default_manifest_path(args.output_jsonl)
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
