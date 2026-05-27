#!/usr/bin/env python3
"""Controlled bridge-conditioned fallback baseline for DAEC-DAPG audits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.data import normalize_text
from src.dpathrag.io import read_json, write_json, write_jsonl
from src.dpathrag.metrics import recall_at_k, support_complete_at_k
from src.dpathrag.reader_data import selected_doc_payload


def tokens(text: Any) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", normalize_text(text)) if tok}


def token_cosine(left: Any, right: Any) -> float:
    a = tokens(left)
    b = tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def pool_docs(record: dict[str, Any]) -> list[dict[str, Any]]:
    docs = list(record.get("pool_docs") or record.get("docs") or [])
    titles = list(record.get("pool_titles") or [])
    scores = list(record.get("pool_doc_scores") or [])
    ids = list(record.get("pool_doc_ids") or [])
    output: list[dict[str, Any]] = []
    for idx, raw in enumerate(docs):
        text = str(raw)
        title = titles[idx] if idx < len(titles) else text.split("\n", 1)[0]
        body = text.split("\n", 1)[1] if "\n" in text else text
        output.append(
            {
                "doc_id": ids[idx] if idx < len(ids) else idx,
                "rank": idx + 1,
                "title": str(title),
                "text": body,
                "retriever_score": float(scores[idx]) if idx < len(scores) and scores[idx] is not None else 0.0,
            }
        )
    return output


def select_bridge_conditioned(record: dict[str, Any], *, top_k: int, bridge_seed_count: int) -> list[dict[str, Any]]:
    docs = pool_docs(record)
    if not docs:
        return []
    question = str(record.get("question") or "")
    seeds = sorted(
        docs,
        key=lambda doc: (-token_cosine(question, f"{doc['title']} {doc['text']}"), int(doc["rank"])),
    )[: max(1, int(bridge_seed_count))]
    seed_keys = {(doc["doc_id"], normalize_text(doc["title"])) for doc in seeds}
    bridge_context = question + " " + " ".join(str(doc["title"]) for doc in seeds)
    remaining = [
        doc for doc in docs
        if (doc["doc_id"], normalize_text(doc["title"])) not in seed_keys
    ]
    reranked = sorted(
        remaining,
        key=lambda doc: (-token_cosine(bridge_context, f"{doc['title']} {doc['text']}"), int(doc["rank"])),
    )
    return (seeds + reranked)[: max(1, int(top_k))]


def build_reader_rows(pool_records: list[dict[str, Any]], *, top_k: int, bridge_seed_count: int, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(pool_records):
        selected = select_bridge_conditioned(record, top_k=int(top_k), bridge_seed_count=int(bridge_seed_count))
        gold_titles = list(record.get("gold_titles") or [])
        selected_titles = [doc["title"] for doc in selected]
        selected_docs = [
            selected_doc_payload(
                title=str(doc["title"]),
                text=str(doc["text"]),
                rank=rank,
                source=source,
                gold_titles=gold_titles,
                doc_id=doc.get("doc_id"),
                score=float(doc.get("retriever_score") or 0.0),
            )
            for rank, doc in enumerate(selected, start=1)
        ]
        rows.append(
            {
                "qid": record.get("qid") or record.get("id") or record.get("query_idx") or idx,
                "query_idx": record.get("query_idx", idx),
                "source": source,
                "top_k": int(top_k),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "gold_titles": gold_titles,
                "selected_docs": selected_docs,
                "support_recall": recall_at_k(gold_titles, selected_titles, int(top_k)),
                "support_complete": support_complete_at_k(gold_titles, selected_titles, int(top_k)),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "support_recall": 0.0, "support_complete": 0.0}
    denom = float(len(rows))
    return {
        "rows": len(rows),
        "support_recall": round(sum(float(row["support_recall"]) for row in rows) / denom, 4),
        "support_complete": round(sum(float(row["support_complete"]) for row in rows) / denom, 4),
        "avg_selected_docs": round(sum(len(row.get("selected_docs") or []) for row in rows) / denom, 4),
    }


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# DAEC-DAPG Controlled Bridge Baseline",
        "",
        f"- Pool: `{payload['pool_json']}`",
        f"- Rows: `{payload['summary']['rows']}`",
        f"- Top-k: `{payload['top_k']}`",
        f"- Bridge seed count: `{payload['bridge_seed_count']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| support_recall | {payload['summary']['support_recall']:.4f} |",
        f"| support_complete | {payload['summary']['support_complete']:.4f} |",
        f"| avg_selected_docs | {payload['summary']['avg_selected_docs']:.4f} |",
        "",
        "This is a controlled bridge-conditioned fallback baseline, not official BridgeRAG.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--bridge_seed_count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_jsonl", default="reports/dpathrag/daec_dapg_bridge_baseline_rows_20260428.jsonl")
    parser.add_argument("--output_json", default="reports/dpathrag/daec_dapg_bridge_baseline_20260428.json")
    parser.add_argument("--output_md", default="reports/dpathrag/daec_dapg_bridge_baseline_20260428.md")
    args = parser.parse_args()

    records = read_json(args.pool_json)
    if isinstance(records, dict):
        records = list(records.get("rows") or records.get("records") or records.get("data") or [])
    if int(args.limit) > 0:
        records = list(records)[: int(args.limit)]
    rows = build_reader_rows(
        list(records),
        top_k=int(args.top_k),
        bridge_seed_count=int(args.bridge_seed_count),
        source="bridge_conditioned_fallback",
    )
    payload = {
        "pool_json": str(args.pool_json),
        "top_k": int(args.top_k),
        "bridge_seed_count": int(args.bridge_seed_count),
        "summary": summarize(rows),
        "notes": "Controlled fallback baseline; not official BridgeRAG.",
    }
    write_jsonl(rows, args.output_jsonl)
    write_json(payload, args.output_json)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
