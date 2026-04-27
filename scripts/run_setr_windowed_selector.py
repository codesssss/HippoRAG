#!/usr/bin/env python3
"""Run a windowed two-stage SetR-style selector over widened pools.

This is a fairer widened-pool adaptation than direct top-100 prompting: each
stage-1 call preserves SetR's top-20 operating regime, then a second SetR-style
call selects from the union of window candidates.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import json
from math import ceil
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_setr_inputs import split_title_text, truncate_text
from run_setr_style_selector import build_openai_client, call_selector


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_existing_keys(path: str | Path) -> set[str]:
    out = Path(path)
    if not out.exists():
        return set()
    keys = set()
    with out.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("query_idx") is not None:
                keys.add(str(row.get("query_idx")))
    return keys


def make_contexts(
    *,
    pool_docs: Sequence[Any],
    pool_titles: Sequence[Any],
    pool_doc_scores: Sequence[Any],
    pool_doc_ids: Sequence[Any],
    global_positions: Sequence[int],
    doc_max_chars: int,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for local_idx, global_pos in enumerate(global_positions, start=1):
        doc = str(pool_docs[global_pos] if global_pos < len(pool_docs) else "")
        fallback_title = str(pool_titles[global_pos] if global_pos < len(pool_titles) else "")
        title, text = split_title_text(doc, fallback_title=fallback_title)
        contexts.append(
            {
                "position": local_idx,
                "pool_position": int(global_pos),
                "doc_id": pool_doc_ids[global_pos] if global_pos < len(pool_doc_ids) else None,
                "title": title,
                "text": truncate_text(text, doc_max_chars),
                "score": pool_doc_scores[global_pos] if global_pos < len(pool_doc_scores) else None,
            }
        )
    return contexts


def make_selector_record(
    *,
    source_record: dict[str, Any],
    global_positions: Sequence[int],
    doc_max_chars: int,
) -> dict[str, Any]:
    return {
        "query_idx": source_record.get("query_idx"),
        "question": source_record.get("question"),
        "answers": source_record.get("gold_answers") or [],
        "gold_titles": source_record.get("gold_titles") or [],
        "pool_k": len(global_positions),
        "contexts": make_contexts(
            pool_docs=list(source_record.get("pool_docs") or []),
            pool_titles=list(source_record.get("pool_titles") or []),
            pool_doc_scores=list(source_record.get("pool_doc_scores") or []),
            pool_doc_ids=list(source_record.get("pool_doc_ids") or []),
            global_positions=global_positions,
            doc_max_chars=doc_max_chars,
        ),
    }


def unique_in_order(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def mock_select(record: dict[str, Any], max_count: int) -> dict[str, Any]:
    selected = list(range(min(max_count, len(record.get("contexts") or []))))
    return {
        "raw_output": "### Final Selection: " + " ".join(f"[{pos + 1}]" for pos in selected),
        "selected_positions": selected,
        "selected_1based": [pos + 1 for pos in selected],
        "parse_success": True,
    }


def safe_call_selector(
    *,
    client: Any,
    record: dict[str, Any],
    model: str,
    max_tokens: int,
    temperature: float,
    append_no_think: bool,
    mock_rank_order: bool,
    mock_max_count: int,
) -> dict[str, Any]:
    if mock_rank_order:
        return mock_select(record, mock_max_count)
    try:
        return call_selector(
            client,
            record,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            append_no_think=append_no_think,
        )
    except Exception as exc:
        return {
            "raw_output": "",
            "selected_positions": [],
            "selected_1based": [],
            "parse_success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }


def stage1_windows(pool_k: int, window_size: int) -> list[tuple[int, int, list[int]]]:
    windows = []
    for window_id in range(ceil(pool_k / window_size)):
        start = window_id * window_size
        end = min(pool_k, start + window_size)
        windows.append((start, end, list(range(start, end))))
    return windows


def run_windowed_record(
    record: dict[str, Any],
    *,
    client: Any,
    pool_k: int,
    window_size: int,
    stage1_max_per_window: int,
    stage2_max_candidates: int,
    final_max_count: int,
    doc_max_chars: int,
    model: str,
    max_tokens: int,
    temperature: float,
    append_no_think: bool,
    candidate_order: str,
    mock_rank_order: bool,
) -> dict[str, Any]:
    actual_pool_k = min(pool_k, len(record.get("pool_docs") or []))
    stage1_rows: list[dict[str, Any]] = []
    stage1_global_in_selection_order: list[int] = []

    for window_id, start, end_positions in stage1_windows(actual_pool_k, window_size):
        selector_record = make_selector_record(
            source_record=record,
            global_positions=end_positions,
            doc_max_chars=doc_max_chars,
        )
        selection = safe_call_selector(
            client=client,
            record=selector_record,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            append_no_think=append_no_think,
            mock_rank_order=mock_rank_order,
            mock_max_count=stage1_max_per_window,
        )
        selected_local = [int(pos) for pos in (selection.get("selected_positions") or [])[:stage1_max_per_window]]
        selected_global = [
            end_positions[pos]
            for pos in selected_local
            if 0 <= pos < len(end_positions)
        ]
        stage1_global_in_selection_order.extend(selected_global)
        stage1_rows.append(
            {
                "window_id": window_id,
                "start": start,
                "end": start + len(end_positions),
                "selected_local_positions": selected_local,
                "selected_global_positions": selected_global,
                "parse_success": bool(selection.get("parse_success")),
                "error_type": selection.get("error_type"),
                "error": selection.get("error"),
                "raw_output": selection.get("raw_output", ""),
            }
        )

    if candidate_order == "stage1":
        candidates = unique_in_order(stage1_global_in_selection_order)
    else:
        candidates = sorted(unique_in_order(stage1_global_in_selection_order))
    candidates = candidates[:stage2_max_candidates]

    if candidates:
        stage2_record = make_selector_record(
            source_record=record,
            global_positions=candidates,
            doc_max_chars=doc_max_chars,
        )
        stage2 = safe_call_selector(
            client=client,
            record=stage2_record,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            append_no_think=append_no_think,
            mock_rank_order=mock_rank_order,
            mock_max_count=final_max_count,
        )
        stage2_local = [int(pos) for pos in (stage2.get("selected_positions") or [])[:final_max_count]]
        selected_global = [
            candidates[pos]
            for pos in stage2_local
            if 0 <= pos < len(candidates)
        ]
    else:
        stage2 = {
            "raw_output": "",
            "selected_positions": [],
            "selected_1based": [],
            "parse_success": False,
            "error": "no_stage1_candidates",
        }
        stage2_local = []
        selected_global = []

    stage1_success_count = sum(1 for row in stage1_rows if row.get("parse_success"))
    return {
        "query_idx": record.get("query_idx"),
        "question": record.get("question"),
        "pool_k": actual_pool_k,
        "window_size": window_size,
        "stage1_max_per_window": stage1_max_per_window,
        "stage2_max_candidates": stage2_max_candidates,
        "candidate_order": candidate_order,
        "stage1": stage1_rows,
        "candidate_global_positions": candidates,
        "candidate_count": len(candidates),
        "stage2_selected_candidate_positions": stage2_local,
        "stage2_raw_output": stage2.get("raw_output", ""),
        "stage2_parse_success": bool(stage2.get("parse_success")),
        "selected_positions": selected_global,
        "selected_1based": [pos + 1 for pos in selected_global],
        "parse_success": bool(selected_global),
        "stage1_parse_success_rate": stage1_success_count / len(stage1_rows) if stage1_rows else 0.0,
        "prompt_mode": "selection_IRI_windowed_2stage",
        "append_no_think": append_no_think,
        "model": model,
        "stage2_error_type": stage2.get("error_type"),
        "stage2_error": stage2.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--output_selection_jsonl", required=True)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stage1_max_per_window", type=int, default=5)
    parser.add_argument("--stage2_max_candidates", type=int, default=25)
    parser.add_argument("--final_max_count", type=int, default=5)
    parser.add_argument("--doc_max_chars", type=int, default=768)
    parser.add_argument("--llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--model", default="qwen3-8b-train")
    parser.add_argument("--api_key", default="sk-")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate_order", choices=["global_rank", "stage1"], default="global_rank")
    parser.add_argument("--no_append_no_think", action="store_true")
    parser.add_argument("--mock_rank_order", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.pool_json)
    records = list(payload.get("records") or [])
    if args.limit > 0:
        records = records[: args.limit]
    if args.resume:
        existing = load_existing_keys(args.output_selection_jsonl)
        records = [
            record for record in records
            if str(record.get("query_idx")) not in existing
        ]

    out = Path(args.output_selection_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        client = None if args.mock_rank_order else build_openai_client(args.llm_base_url, args.api_key)
        return run_windowed_record(
            record,
            client=client,
            pool_k=int(args.pool_k),
            window_size=int(args.window_size),
            stage1_max_per_window=int(args.stage1_max_per_window),
            stage2_max_candidates=int(args.stage2_max_candidates),
            final_max_count=int(args.final_max_count),
            doc_max_chars=int(args.doc_max_chars),
            model=str(args.model),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            append_no_think=not bool(args.no_append_no_think),
            candidate_order=str(args.candidate_order),
            mock_rank_order=bool(args.mock_rank_order),
        )

    mode = "a" if args.resume else "w"
    with out.open(mode, encoding="utf-8") as handle:
        if int(args.num_workers) <= 1:
            for record in records:
                handle.write(json.dumps(run_one(record), ensure_ascii=False) + "\n")
                handle.flush()
        else:
            with futures.ThreadPoolExecutor(max_workers=int(args.num_workers)) as executor:
                for row in executor.map(run_one, records):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
    print(f"Wrote {len(records)} windowed SetR rows to {out}")


if __name__ == "__main__":
    main()
