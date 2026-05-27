#!/usr/bin/env python3
"""Export a reader-facing external pool from selector-frontier traces.

The selector-only frontier stores the selected pool positions in each query's
trace.  This script materializes those choices into an external-pool JSON whose
first five documents are the frozen selector output.  Downstream QA can then run
with ``--setwise_selector none`` without recomputing DBEC or demand bindings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ordered_unique_positions(values: Sequence[Any], *, pool_size: int) -> list[int]:
    positions: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            pos = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= pos < pool_size and pos not in seen:
            seen.add(pos)
            positions.append(pos)
    return positions


def selected_positions_from_trace(trace: Mapping[str, Any], *, pool_size: int, qa_top_k: int) -> list[int]:
    selector_trace = dict(trace.get("selector_trace", {}) or {})
    candidates = [
        selector_trace.get("final_front_pool_positions"),
        selector_trace.get("selected_pool_positions"),
        selector_trace.get("selected_positions"),
    ]
    selected: list[int] = []
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            selected = ordered_unique_positions(candidate, pool_size=pool_size)
            if selected:
                break
    for pos in range(pool_size):
        if len(selected) >= qa_top_k:
            break
        if pos not in selected:
            selected.append(pos)
    return selected[:qa_top_k]


def reorder_record(record: Mapping[str, Any], trace: Mapping[str, Any], *, qa_top_k: int) -> dict[str, Any]:
    pool_docs = list(record.get("pool_docs", []) or [])
    pool_titles = list(record.get("pool_titles", []) or [])
    pool_scores = list(record.get("pool_doc_scores", []) or [])
    pool_doc_ids = list(record.get("pool_doc_ids", []) or [])
    pool_size = len(pool_docs)
    selected = selected_positions_from_trace(trace, pool_size=pool_size, qa_top_k=qa_top_k)
    tail = [pos for pos in range(pool_size) if pos not in set(selected)]
    order = selected + tail

    def reorder(values: Sequence[Any]) -> list[Any]:
        return [values[pos] for pos in order if pos < len(values)]

    output = dict(record)
    output["pool_docs"] = reorder(pool_docs)
    output["pool_titles"] = reorder(pool_titles)
    output["pool_doc_scores"] = reorder(pool_scores)
    output["pool_doc_ids"] = reorder(pool_doc_ids)
    output["pool_k"] = int(len(output["pool_docs"]))
    output["selector_frontier_reader_pool"] = {
        "selected_positions": selected,
        "source_query_idx": int(record.get("query_idx", len(selected))),
        "source_selector": str((trace.get("selector_trace", {}) or {}).get("selector", "")),
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool-json", type=Path, required=True)
    parser.add_argument("--selector-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--source-name", default="")
    args = parser.parse_args()

    source_pool = read_json(Path(args.source_pool_json))
    selector = read_json(Path(args.selector_json))
    records = list(source_pool.get("records", []) or [])
    traces = list(selector.get("setwise_selector_query_traces", []) or [])
    if len(records) != len(traces):
        raise ValueError(f"Record/trace count mismatch: {len(records)} vs {len(traces)}")

    out_records = [
        reorder_record(record, trace, qa_top_k=int(args.qa_top_k))
        for record, trace in zip(records, traces)
    ]
    output = {
        "dataset": source_pool.get("dataset") or selector.get("dataset"),
        "limit": int(source_pool.get("limit") or selector.get("limit") or len(out_records)),
        "pool_k": int(max((len(row.get("pool_docs", []) or []) for row in out_records), default=0)),
        "source": str(args.source_name or f"{selector.get('dataset')}_{selector.get('variant')}_reader_pool"),
        "source_pool_json": str(args.source_pool_json),
        "selector_json": str(args.selector_json),
        "selector_variant": selector.get("variant"),
        "qa_top_k": int(args.qa_top_k),
        "retrieval": source_pool.get("retrieval", {}),
        "records": out_records,
    }
    write_json(output, Path(args.output_json))
    print(f"Wrote {args.output_json} records={len(out_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
