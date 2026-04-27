#!/usr/bin/env python3
"""Apply SetR-style selected positions to an external-pool JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from run_setr_style_selector import parse_setr_selection


POOL_LIST_KEYS = ("pool_docs", "pool_titles", "pool_doc_scores", "pool_doc_ids")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_selection_map(selection_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key = {}
    for row in selection_rows:
        key = str(row.get("query_idx")) if row.get("query_idx") is not None else str(row.get("question") or "")
        by_key[key] = row
    return by_key


def selected_positions_from_row(row: dict[str, Any], pool_size: int) -> list[int]:
    raw_positions = row.get("selected_positions")
    if isinstance(raw_positions, list):
        positions = []
        seen = set()
        for raw in raw_positions:
            try:
                pos = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= pos < pool_size and pos not in seen:
                positions.append(pos)
                seen.add(pos)
        return positions
    return parse_setr_selection(str(row.get("raw_output") or ""), max_position=pool_size)


def reorder_positions(selected_positions: Sequence[int], pool_size: int) -> list[int]:
    seen = set()
    ordered = []
    for pos in selected_positions:
        if 0 <= int(pos) < pool_size and int(pos) not in seen:
            ordered.append(int(pos))
            seen.add(int(pos))
    ordered.extend(pos for pos in range(pool_size) if pos not in seen)
    return ordered


def reorder_list(values: list[Any], order: Sequence[int]) -> list[Any]:
    return [values[pos] for pos in order if pos < len(values)]


def apply_selection_to_record(record: dict[str, Any], selection_row: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    pool_size = len(record.get("pool_docs") or [])
    selected = selected_positions_from_row(selection_row or {}, pool_size=pool_size) if selection_row else []
    order = reorder_positions(selected, pool_size=pool_size)
    out = dict(record)
    for key in POOL_LIST_KEYS:
        values = list(record.get(key) or [])
        if values:
            out[key] = reorder_list(values, order)
    trace = {
        "query_idx": record.get("query_idx"),
        "parse_success": bool(selected),
        "selected_positions": selected,
        "selected_1based": [pos + 1 for pos in selected],
        "fallback_count": max(0, min(5, pool_size) - min(len(selected), 5)),
        "pool_size": pool_size,
    }
    out.setdefault("setr_selection_trace", trace)
    out["setr_selection_trace"] = trace
    return out, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--selection_jsonl", required=True)
    parser.add_argument("--output_pool_json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pool_path = Path(args.pool_json)
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    if args.limit > 0:
        records = records[: args.limit]

    selection_rows = load_jsonl(args.selection_jsonl)
    selection_by_key = build_selection_map(selection_rows)
    out_records = []
    traces = []
    for idx, record in enumerate(records):
        key = str(record.get("query_idx")) if record.get("query_idx") is not None else str(record.get("question") or "")
        selected_record, trace = apply_selection_to_record(record, selection_by_key.get(key))
        out_records.append(selected_record)
        traces.append(trace)

    out_payload = dict(payload)
    out_payload["source"] = f"{payload.get('source', 'external_pool')}_setr_style"
    out_payload["setr_selection"] = {
        "selection_jsonl": str(args.selection_jsonl),
        "records_reordered": len(out_records),
        "parse_success_count": sum(1 for trace in traces if trace["parse_success"]),
        "parse_failure_count": sum(1 for trace in traces if not trace["parse_success"]),
        "avg_selected_count": (
            sum(len(trace["selected_positions"]) for trace in traces) / len(traces) if traces else 0.0
        ),
    }
    out_payload["records"] = out_records

    out = Path(args.output_pool_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote SetR-reordered pool with {len(out_records)} records to {out}")


if __name__ == "__main__":
    main()
