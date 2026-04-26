"""Fixed-pool cache builders for D-PathRAG smoke experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.dpathrag.data import extract_title, normalize_text, support_titles
from src.dpathrag.io import read_json, write_jsonl


def gold_support_indicators(pool_titles: Sequence[str], gold_titles: Sequence[str]) -> list[int]:
    normalized_gold = {normalize_text(title) for title in gold_titles}
    return [1 if normalize_text(title) in normalized_gold else 0 for title in pool_titles]


def build_cache_record(sample: dict[str, Any], pool_record: dict[str, Any], *, source: str) -> dict[str, Any]:
    pool_docs = list(pool_record.get("pool_docs") or [])
    pool_titles = list(pool_record.get("pool_titles") or [extract_title(doc) for doc in pool_docs])
    scores = list(pool_record.get("pool_doc_scores") or [])
    gold_titles = support_titles(sample) or list(pool_record.get("gold_titles") or [])
    indicators = gold_support_indicators(pool_titles, gold_titles)
    return {
        "qid": sample.get("_id", pool_record.get("query_idx")),
        "query_idx": int(pool_record.get("query_idx", 0)),
        "source": source,
        "question": sample.get("question", pool_record.get("question")),
        "answer": sample.get("answer"),
        "type": sample.get("type"),
        "gold_titles": gold_titles,
        "evidences": sample.get("evidences") or [],
        "candidates": [
            {
                "doc_id": pool_record.get("pool_doc_ids", [None] * len(pool_docs))[idx]
                if idx < len(pool_record.get("pool_doc_ids", []))
                else None,
                "rank": idx + 1,
                "title": pool_titles[idx] if idx < len(pool_titles) else extract_title(pool_docs[idx]),
                "text": pool_docs[idx],
                "retriever_score": float(scores[idx]) if idx < len(scores) else None,
                "gold_support": int(indicators[idx]),
                "features": {
                    "rank": float(idx + 1),
                    "retriever_score": float(scores[idx]) if idx < len(scores) else 0.0,
                },
            }
            for idx in range(len(pool_docs))
        ],
    }


def build_cache_rows(samples: Sequence[dict[str, Any]], pool_payload: dict[str, Any], *, source: str, limit: int = 0) -> list[dict[str, Any]]:
    records = list(pool_payload.get("records") or [])
    if limit and limit > 0:
        samples = list(samples)[:limit]
        records = records[:limit]
    rows: list[dict[str, Any]] = []
    for sample, record in zip(samples, records):
        rows.append(build_cache_record(sample, record, source=source))
    return rows


def build_cache_jsonl(samples: Sequence[dict[str, Any]], pool_json: str | Path, output_jsonl: str | Path, *, source: str, limit: int = 0) -> int:
    payload = read_json(pool_json)
    rows = build_cache_rows(samples, payload, source=source, limit=limit)
    write_jsonl(rows, output_jsonl)
    return len(rows)
