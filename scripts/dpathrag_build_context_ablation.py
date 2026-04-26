#!/usr/bin/env python3
"""Build reader inputs for D-PathRAG context/order/distractor ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

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


def gold_indices(candidates: Sequence[dict[str, Any]], *, max_candidates: int) -> list[int]:
    return [
        idx
        for idx, candidate in enumerate(list(candidates)[: int(max_candidates)])
        if int(candidate.get("gold_support") or 0) == 1
    ]


def unique_indices(indices: Sequence[int], *, max_candidates: int, top_k: int) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for index in indices:
        value = int(index)
        if value < 0 or value >= int(max_candidates) or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if int(top_k) > 0 and len(output) >= int(top_k):
            break
    return output


def select_indices(
    config: str,
    candidates: Sequence[dict[str, Any]],
    prediction: dict[str, Any],
    *,
    top_k: int,
    max_candidates: int,
) -> list[int]:
    selected = [int(index) for index in prediction.get("selected_indices") or []]
    selected = unique_indices(selected, max_candidates=max_candidates, top_k=top_k)
    gold = gold_indices(candidates, max_candidates=max_candidates)
    if config == "rank_top5":
        return list(range(min(int(top_k), len(candidates), int(max_candidates))))
    if config == "selector_ar_order":
        return selected
    if config == "selector_rank_order":
        return sorted(selected)
    if config == "selector_gold_first":
        gold_set = set(gold)
        return sorted(selected, key=lambda idx: (0 if idx in gold_set else 1, selected.index(idx)))
    if config == "gold_support_only":
        return gold[: int(top_k)]
    if config == "gold_plus_selector_distractors":
        gold_set = set(gold)
        distractors = [idx for idx in selected if idx not in gold_set]
        backfill = [idx for idx in range(min(len(candidates), int(max_candidates))) if idx not in gold_set]
        return unique_indices(gold + distractors + backfill, max_candidates=max_candidates, top_k=top_k)
    raise ValueError(f"Unsupported ablation config: {config}")


def build_reader_rows(
    cache_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    *,
    config: str,
    source: str,
    top_k: int,
    max_candidates: int,
    max_doc_chars: int,
) -> list[dict[str, Any]]:
    cache_by_qid = {str(row.get("qid")): row for row in cache_rows}
    rows: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        qid = str(prediction.get("qid"))
        if qid not in cache_by_qid:
            raise KeyError(f"Missing cache row for qid={qid}")
        cache_row = cache_by_qid[qid]
        candidates = list(cache_row.get("candidates") or [])[: int(max_candidates)]
        indices = select_indices(config, candidates, prediction, top_k=int(top_k), max_candidates=int(max_candidates))
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
                "ablation_config": config,
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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--predictions_jsonl", default="reports/dpathrag/selector_warmstart_proprag_embed_best_local1000.predictions.jsonl")
    parser.add_argument("--cache_limit", type=int, default=1000)
    parser.add_argument("--prediction_limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--output_dir", default="data/dpathrag/reader_ablation")
    parser.add_argument("--manifest_json", default="reports/dpathrag/context_ablation_proprag_eval200_manifest.json")
    args = parser.parse_args()

    configs = [
        "rank_top5",
        "selector_ar_order",
        "selector_rank_order",
        "selector_gold_first",
        "gold_support_only",
        "gold_plus_selector_distractors",
    ]
    cache_rows = load_jsonl(args.cache_jsonl, limit=int(args.cache_limit))
    prediction_rows = load_jsonl(args.predictions_jsonl, limit=int(args.prediction_limit))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "cache_jsonl": str(args.cache_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "cache_limit": int(args.cache_limit),
        "prediction_rows": len(prediction_rows),
        "top_k": int(args.top_k),
        "max_candidates": int(args.max_candidates),
        "configs": {},
    }
    for config in configs:
        source = f"proprag_context_ablation_{config}"
        rows = build_reader_rows(
            cache_rows,
            prediction_rows,
            config=config,
            source=source,
            top_k=int(args.top_k),
            max_candidates=int(args.max_candidates),
            max_doc_chars=int(args.max_doc_chars),
        )
        output_jsonl = output_dir / f"2wiki_{source}.jsonl"
        write_jsonl(rows, output_jsonl)
        manifest["configs"][config] = {
            "source": source,
            "output_jsonl": str(output_jsonl),
            "summary": summarize_reader_records(rows),
        }
    write_json(manifest, args.manifest_json)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
