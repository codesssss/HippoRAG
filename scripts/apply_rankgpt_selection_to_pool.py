#!/usr/bin/env python3
"""Apply RankGPT-style selected positions to an external-pool JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


POOL_LIST_KEYS = ("pool_docs", "pool_titles", "pool_doc_scores", "pool_doc_ids")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_dataset(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "").replace("-", "")


def row_matches_dataset(row: Mapping[str, Any], dataset: str) -> bool:
    if not dataset:
        return True
    candidates = [row.get("dataset"), row.get("base_dataset")]
    present = [normalize_dataset(value) for value in candidates if str(value or "").strip()]
    if not present:
        return True
    target = normalize_dataset(dataset)
    return any(value == target for value in present)


def load_selector_rows(path: str | Path, *, method: str, dataset: str = "") -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("method")) != str(method):
                continue
            if not row_matches_dataset(row, dataset):
                continue
            rows[safe_int(row.get("query_index"))] = dict(row)
    return rows


def parse_positions(row: Mapping[str, Any] | None, *, pool_size: int) -> list[int]:
    if not row:
        return []
    try:
        raw_positions = json.loads(str(row.get("selected_positions_json") or "[]"))
    except json.JSONDecodeError:
        raw_positions = []
    output: list[int] = []
    seen: set[int] = set()
    for raw_pos in raw_positions:
        pos = safe_int(raw_pos, default=-1)
        if 0 <= pos < int(pool_size) and pos not in seen:
            output.append(pos)
            seen.add(pos)
    return output


def fill_source_order(prefix: Sequence[int], *, pool_size: int) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for raw_pos in prefix:
        pos = int(raw_pos)
        if 0 <= pos < int(pool_size) and pos not in seen:
            output.append(pos)
            seen.add(pos)
    for pos in range(int(pool_size)):
        if pos not in seen:
            output.append(pos)
            seen.add(pos)
    return output


def reorder_list(values: Sequence[Any], order: Sequence[int]) -> list[Any]:
    return [values[pos] for pos in order if 0 <= int(pos) < len(values)]


def apply_order_to_record(record: Mapping[str, Any], order: Sequence[int], *, method: str) -> dict[str, Any]:
    output = dict(record)
    for key in POOL_LIST_KEYS:
        values = list(record.get(key) or [])
        if values:
            output[key] = reorder_list(values, order)
    output["rankgpt_selection_trace"] = {
        "method": method,
        "selected_positions": list(order)[:5],
        "selected_1based": [pos + 1 for pos in list(order)[:5]],
        "pool_size": len(record.get("pool_docs") or []),
        "reader_pool_size": len(order),
    }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    pool_payload = json.loads(Path(args.pool_json).read_text(encoding="utf-8"))
    records = list(pool_payload.get("records") or [])
    if int(args.limit) > 0:
        records = records[: int(args.limit)]
    dataset = str(args.dataset or pool_payload.get("dataset") or "")
    selector_rows = load_selector_rows(args.selector_rows_csv, method=str(args.method), dataset=dataset)
    output_records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for query_index, record in enumerate(records):
        pool_size = len(record.get("pool_docs") or [])
        selected = parse_positions(selector_rows.get(query_index), pool_size=pool_size)
        order = fill_source_order(selected, pool_size=pool_size)
        output_records.append(apply_order_to_record(record, order, method=str(args.method)))
        traces.append(
            {
                "query_index": query_index,
                "query_idx": record.get("query_idx"),
                "selected_count": len(selected),
                "selected_positions": selected,
                "fallback_count": max(0, min(5, pool_size) - min(len(selected), 5)),
            }
        )

    output_payload = dict(pool_payload)
    output_payload["source"] = f"{pool_payload.get('source', 'external_pool')}_rankgpt_style"
    output_payload["rankgpt_selection"] = {
        "selector_rows_csv": str(args.selector_rows_csv),
        "method": str(args.method),
        "dataset_filter": dataset,
        "records_reordered": len(output_records),
        "parse_success_count": sum(1 for trace in traces if int(trace["selected_count"]) > 0),
        "parse_failure_count": sum(1 for trace in traces if int(trace["selected_count"]) == 0),
        "avg_selected_count": (
            sum(int(trace["selected_count"]) for trace in traces) / len(traces) if traces else 0.0
        ),
    }
    output_payload["records"] = output_records

    output_path = Path(args.output_pool_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if str(args.trace_json):
        trace_path = Path(args.trace_json)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(traces, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_payload["rankgpt_selection"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--selector_rows_csv", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output_pool_json", required=True)
    parser.add_argument("--trace_json", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset", default="")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
