#!/usr/bin/env python3
"""Build rank-anchor blend reader inputs for D-PathRAG selector diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

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


def candidate_body(candidate: dict[str, Any]) -> str:
    text = str(candidate.get("text") or "")
    return text.split("\n", 1)[1] if "\n" in text else text


def candidate_is_gold(candidate: dict[str, Any]) -> bool:
    return int(candidate.get("gold_support") or 0) == 1


def unique_indices(indices: Sequence[int], *, max_candidates: int, top_k: int) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for index in indices:
        value = int(index)
        if value < 0 or value >= int(max_candidates) or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= int(top_k):
            break
    return output


def blend_indices(
    *,
    rank_anchor_m: int,
    selector_indices: Sequence[int],
    candidate_count: int,
    top_k: int,
    max_candidates: int,
) -> list[int]:
    usable_count = min(int(candidate_count), int(max_candidates))
    rank_anchor = list(range(min(int(rank_anchor_m), int(top_k), usable_count)))
    selector_fill = [int(index) for index in selector_indices if int(index) not in set(rank_anchor)]
    rank_backfill = [idx for idx in range(usable_count) if idx not in set(rank_anchor) and idx not in set(selector_fill)]
    return unique_indices(rank_anchor + selector_fill + rank_backfill, max_candidates=usable_count, top_k=int(top_k))


def blend_diagnostics(
    *,
    candidates: Sequence[dict[str, Any]],
    selected_indices: Sequence[int],
    rank_anchor_m: int,
    top_k: int,
) -> dict[str, float]:
    rank_topk = set(range(min(int(top_k), len(candidates))))
    rank_anchor = set(range(min(int(rank_anchor_m), int(top_k), len(candidates))))
    final_set = {int(index) for index in selected_indices}
    selector_added = [index for index in selected_indices if int(index) not in rank_anchor]
    rank_tail_removed = sorted(rank_topk - final_set)
    selector_added_gold = sum(1 for index in selector_added if candidate_is_gold(candidates[int(index)]))
    rank_tail_gold_removed = sum(1 for index in rank_tail_removed if candidate_is_gold(candidates[int(index)]))
    return {
        "rank_anchor_m": float(rank_anchor_m),
        "avg_rank_anchor_docs": float(len(rank_anchor)),
        "avg_selector_added_docs": float(len(selector_added)),
        "avg_selector_added_gold": float(selector_added_gold),
        "avg_rank_tail_gold_removed": float(rank_tail_gold_removed),
        "avg_net_gold_gain": float(selector_added_gold - rank_tail_gold_removed),
    }


def summarize_diagnostics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "rank_anchor_m": 0.0,
            "avg_rank_anchor_docs": 0.0,
            "avg_selector_added_docs": 0.0,
            "avg_selector_added_gold": 0.0,
            "avg_rank_tail_gold_removed": 0.0,
            "avg_net_gold_gain": 0.0,
        }
    denom = float(len(rows))
    keys = list(rows[0].keys())
    return {key: round(sum(float(row.get(key) or 0.0) for row in rows) / denom, 4) for key in keys}


def build_reader_rows(
    cache_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    *,
    rank_anchor_m: int,
    source: str,
    top_k: int,
    max_candidates: int,
    max_doc_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cache_by_qid = {str(row.get("qid")): row for row in cache_rows}
    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, float]] = []
    for prediction in prediction_rows:
        qid = str(prediction.get("qid"))
        if qid not in cache_by_qid:
            raise KeyError(f"Missing cache row for qid={qid}")
        cache_row = cache_by_qid[qid]
        candidates = list(cache_row.get("candidates") or [])[: int(max_candidates)]
        selector_indices = [int(index) for index in prediction.get("selected_indices") or []]
        indices = blend_indices(
            rank_anchor_m=int(rank_anchor_m),
            selector_indices=selector_indices,
            candidate_count=len(candidates),
            top_k=int(top_k),
            max_candidates=int(max_candidates),
        )
        diagnostics = blend_diagnostics(
            candidates=candidates,
            selected_indices=indices,
            rank_anchor_m=int(rank_anchor_m),
            top_k=int(top_k),
        )
        diagnostic_rows.append(diagnostics)
        selected_docs: list[dict[str, Any]] = []
        gold_titles = list(cache_row.get("gold_titles") or [])
        for prompt_rank, candidate_index in enumerate(indices, start=1):
            candidate = candidates[candidate_index]
            selected_docs.append(
                selected_doc_payload(
                    title=str(candidate.get("title") or ""),
                    text=candidate_body(candidate),
                    rank=prompt_rank,
                    source=source,
                    gold_titles=gold_titles,
                    doc_id=candidate.get("doc_id"),
                    score=float(candidate.get("retriever_score") or 0.0),
                    max_chars=int(max_doc_chars),
                )
            )
        example = featurize_selector_record(cache_row, max_candidates=max(len(candidates), int(top_k)), path_len=int(top_k))
        metrics = support_metrics_for_indices(example, indices)
        rows.append(
            {
                "qid": qid,
                "query_idx": cache_row.get("query_idx"),
                "split": "selector_eval",
                "source": source,
                "rank_anchor_m": int(rank_anchor_m),
                "top_k": len(selected_docs),
                "question": cache_row.get("question"),
                "answer": cache_row.get("answer"),
                "type": cache_row.get("type"),
                "gold_titles": gold_titles,
                "selected_docs": selected_docs,
                "support_recall": metrics["support_recall"],
                "support_complete": metrics["support_complete"],
            }
        )
    return rows, summarize_diagnostics(diagnostic_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--pool_name", default="proprag")
    parser.add_argument("--cache_limit", type=int, default=1000)
    parser.add_argument("--prediction_limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--anchor_values", default="0,1,2,3,4,5")
    parser.add_argument("--output_dir", default="data/dpathrag/reader_rank_anchor")
    parser.add_argument("--manifest_json", default="reports/dpathrag/rank_anchor_blend_proprag_eval200_manifest.json")
    args = parser.parse_args()

    cache_rows = load_jsonl(args.cache_jsonl, limit=int(args.cache_limit))
    prediction_rows = load_jsonl(args.predictions_jsonl, limit=int(args.prediction_limit))
    anchor_values = [int(value.strip()) for value in str(args.anchor_values).split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "cache_jsonl": str(args.cache_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "pool_name": str(args.pool_name),
        "cache_limit": int(args.cache_limit),
        "prediction_rows": len(prediction_rows),
        "top_k": int(args.top_k),
        "max_candidates": int(args.max_candidates),
        "configs": {},
    }
    for rank_anchor_m in anchor_values:
        config = f"rank_anchor_m{rank_anchor_m}"
        source = f"{args.pool_name}_rank_anchor_m{rank_anchor_m}"
        rows, diagnostics = build_reader_rows(
            cache_rows,
            prediction_rows,
            rank_anchor_m=int(rank_anchor_m),
            source=source,
            top_k=int(args.top_k),
            max_candidates=int(args.max_candidates),
            max_doc_chars=int(args.max_doc_chars),
        )
        output_jsonl = output_dir / f"2wiki_{source}.jsonl"
        write_jsonl(rows, output_jsonl)
        manifest["configs"][config] = {
            "source": source,
            "rank_anchor_m": int(rank_anchor_m),
            "output_jsonl": str(output_jsonl),
            "summary": summarize_reader_records(rows),
            "diagnostics": diagnostics,
        }
    write_json(manifest, args.manifest_json)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
