#!/usr/bin/env python3
"""Export external-pool JSON into SetR-style selector inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def split_title_text(doc: str, fallback_title: str = "") -> tuple[str, str]:
    value = str(doc or "")
    if "\n" in value:
        title, text = value.split("\n", 1)
        return title.strip() or fallback_title, text.strip()
    return str(fallback_title or "").strip(), value.strip()


def truncate_text(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rsplit(" ", 1)[0].rstrip() or value[:max_chars].rstrip()


def build_setr_record(record: dict[str, Any], *, pool_k: int, doc_max_chars: int, source_pool_path: str) -> dict[str, Any]:
    pool_docs = list(record.get("pool_docs") or [])[:pool_k]
    pool_titles = list(record.get("pool_titles") or [])[:pool_k]
    pool_scores = list(record.get("pool_doc_scores") or [])[:pool_k]
    pool_doc_ids = list(record.get("pool_doc_ids") or [])[:pool_k]
    gold_titles = {str(title).strip() for title in (record.get("gold_titles") or []) if str(title).strip()}

    contexts = []
    for idx, doc in enumerate(pool_docs):
        fallback_title = str(pool_titles[idx]) if idx < len(pool_titles) else ""
        title, text = split_title_text(str(doc), fallback_title=fallback_title)
        contexts.append(
            {
                "position": idx + 1,
                "pool_position": idx,
                "doc_id": pool_doc_ids[idx] if idx < len(pool_doc_ids) else None,
                "title": title,
                "text": truncate_text(text, doc_max_chars),
                "score": pool_scores[idx] if idx < len(pool_scores) else None,
                "has_gold_title": bool(title and title in gold_titles),
            }
        )

    return {
        "query_idx": record.get("query_idx"),
        "question": record.get("question"),
        "answers": record.get("gold_answers") or [],
        "gold_titles": record.get("gold_titles") or [],
        "source_pool_path": source_pool_path,
        "pool_k": len(contexts),
        "contexts": contexts,
    }


def iter_records(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    records = list(payload.get("records") or [])
    if limit > 0:
        records = records[:limit]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True, help="External-pool JSON from dense/PropRAG exporters.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--doc_max_chars", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pool_path = Path(args.pool_json)
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    records = iter_records(payload, args.limit)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    build_setr_record(
                        record,
                        pool_k=int(args.pool_k),
                        doc_max_chars=int(args.doc_max_chars),
                        source_pool_path=str(pool_path),
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(records)} SetR input records to {out}")


if __name__ == "__main__":
    main()
