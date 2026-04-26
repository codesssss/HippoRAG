#!/usr/bin/env python3
"""Convert D-PathRAG selector predictions into reader-baseline JSONL records."""

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

from src.dpathrag.io import write_json, write_jsonl
from src.dpathrag.reader_data import selected_doc_payload, summarize_reader_records
from src.dpathrag.selector_data import featurize_selector_record, support_metrics_for_indices


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= int(limit):
                    break
    return rows


def _candidate_body(candidate: dict[str, Any]) -> str:
    text = str(candidate.get("text") or "")
    if "\n" in text:
        return text.split("\n", 1)[1]
    return text


def build_reader_rows(
    cache_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    mode: str,
    source: str,
    max_doc_chars: int,
    sort_selected_by_rank: bool,
) -> list[dict[str, Any]]:
    cache_by_qid = {str(row.get("qid")): row for row in cache_rows}
    reader_rows: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        qid = str(prediction.get("qid"))
        if qid not in cache_by_qid:
            raise KeyError(f"Missing cache row for qid={qid}")
        cache_row = cache_by_qid[qid]
        candidates = list(cache_row.get("candidates") or [])
        if mode == "model":
            indices = [int(index) for index in prediction.get("selected_indices") or []]
        elif mode == "rank_topk":
            indices = list(range(len(prediction.get("rank_topk_titles") or [])))
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if sort_selected_by_rank:
            indices = sorted(indices)
        selected_docs: list[dict[str, Any]] = []
        for rank, index in enumerate(indices, start=1):
            if index < 0 or index >= len(candidates):
                continue
            candidate = candidates[index]
            selected_docs.append(
                selected_doc_payload(
                    title=str(candidate.get("title") or ""),
                    text=_candidate_body(candidate),
                    rank=rank,
                    source=source,
                    gold_titles=list(cache_row.get("gold_titles") or []),
                    doc_id=candidate.get("doc_id"),
                    score=candidate.get("retriever_score"),
                    max_chars=int(max_doc_chars),
                )
            )
        example = featurize_selector_record(cache_row, max_candidates=max(len(candidates), len(selected_docs)), path_len=len(indices))
        metrics = support_metrics_for_indices(example, indices)
        reader_rows.append(
            {
                "qid": qid,
                "query_idx": cache_row.get("query_idx"),
                "split": "selector_eval",
                "source": source,
                "top_k": len(selected_docs),
                "question": cache_row.get("question"),
                "answer": cache_row.get("answer"),
                "type": cache_row.get("type"),
                "gold_titles": list(cache_row.get("gold_titles") or []),
                "selected_docs": selected_docs,
                "support_recall": metrics["support_recall"],
                "support_complete": metrics["support_complete"],
            }
        )
    return reader_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--manifest_json", default="")
    parser.add_argument("--cache_limit", type=int, default=0)
    parser.add_argument("--mode", choices=["model", "rank_topk"], default="model")
    parser.add_argument("--source", default="")
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--sort_selected_by_rank", action="store_true")
    args = parser.parse_args()

    cache_rows = load_jsonl(args.cache_jsonl, limit=int(args.cache_limit))
    prediction_rows = load_jsonl(args.predictions_jsonl)
    source = str(args.source or f"selector_{args.mode}")
    reader_rows = build_reader_rows(
        cache_rows,
        prediction_rows,
        mode=str(args.mode),
        source=source,
        max_doc_chars=int(args.max_doc_chars),
        sort_selected_by_rank=bool(args.sort_selected_by_rank),
    )
    write_jsonl(reader_rows, args.output_jsonl)
    manifest = {
        "cache_jsonl": str(args.cache_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "mode": str(args.mode),
        "source": source,
        "sort_selected_by_rank": bool(args.sort_selected_by_rank),
        "summary": summarize_reader_records(reader_rows),
    }
    if args.manifest_json:
        write_json(manifest, args.manifest_json)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
