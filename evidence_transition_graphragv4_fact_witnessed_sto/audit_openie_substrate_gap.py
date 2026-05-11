#!/usr/bin/env python3
"""Trace-level comparison between two V4 OpenIE/STO substrates.

This diagnostic answers a narrow question: when the base substrate misses a
gold document, does the reference substrate recover it because the document
enters the candidate universe, the STO-admitted frontier, the graph order, or
the final top5? It reads only retrieval traces and does not change retrieval.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _unique_ints(values: Sequence[Any] | None) -> list[int]:
    seen: set[int] = set()
    clean: list[int] = []
    for value in values or []:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        clean.append(item)
    return clean


def _retrieval_trace(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return ((row.get("route_trace", {}) or {}).get("retrieval", {}) or {})


def _candidate_trace(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return ((row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {})


def _selected(row: Mapping[str, Any]) -> list[int]:
    return _unique_ints(row.get("retrieved_doc_indices_top5", []) or [])


def _gold(row: Mapping[str, Any]) -> list[int]:
    return _unique_ints(row.get("gold_doc_indices", []) or [])


def _recall(row: Mapping[str, Any]) -> float:
    return float(row.get("query_grounded_sto_recall_at5", 0.0) or 0.0)


def _all_gold(row: Mapping[str, Any]) -> bool:
    return bool(row.get("query_grounded_sto_all_gold_at5", False))


def _doc_position(doc_idx: int, values: Sequence[Any] | None) -> int:
    for idx, value in enumerate(_unique_ints(values)):
        if int(value) == int(doc_idx):
            return idx + 1
    return 0


def _doc_stage(row: Mapping[str, Any], doc_idx: int) -> str:
    doc = int(doc_idx)
    selected = set(_selected(row))
    candidate = _candidate_trace(row)
    retrieval = _retrieval_trace(row)
    selection = retrieval.get("selection", {}) or {}
    source_prior = set(_unique_ints(candidate.get("source_prior_prefix_doc_indices", []) or []))
    graph_order = _unique_ints(selection.get("graph_order_doc_indices", []) or [])
    raw_graph_order = _unique_ints(selection.get("raw_graph_order_doc_indices", []) or [])
    graph_tail = set(_unique_ints(candidate.get("agsto_graph_tail_doc_indices", []) or []))
    admitted = set(_unique_ints(candidate.get("admissible_doc_indices", []) or []))
    candidates = set(_unique_ints(candidate.get("candidate_doc_indices", []) or []))
    certified = set(_unique_ints(retrieval.get("certified_doc_indices", []) or []))

    if doc in selected:
        return "selected_top5"
    if doc in graph_order:
        return "graph_order_not_selected"
    if doc in raw_graph_order:
        return "raw_graph_order_filtered"
    if doc in certified:
        return "certified_not_ordered"
    if doc in source_prior:
        return "source_prior_not_selected"
    if doc in graph_tail:
        return "graph_tail_not_ordered"
    if doc in admitted:
        return "admitted_not_ordered"
    if doc in candidates:
        return "candidate_not_admitted"
    return "candidate_missing"


def _doc_snapshot(row: Mapping[str, Any], doc_idx: int) -> Mapping[str, Any]:
    doc = int(doc_idx)
    candidate = _candidate_trace(row)
    retrieval = _retrieval_trace(row)
    selection = retrieval.get("selection", {}) or {}
    return {
        "stage": _doc_stage(row, doc),
        "selected_pos": _doc_position(doc, _selected(row)),
        "candidate_pos": _doc_position(doc, candidate.get("candidate_doc_indices", []) or []),
        "source_prior_pos": _doc_position(doc, candidate.get("source_prior_prefix_doc_indices", []) or []),
        "admissible_pos": _doc_position(doc, candidate.get("admissible_doc_indices", []) or []),
        "graph_tail_pos": _doc_position(doc, candidate.get("agsto_graph_tail_doc_indices", []) or []),
        "raw_graph_order_pos": _doc_position(doc, selection.get("raw_graph_order_doc_indices", []) or []),
        "graph_order_pos": _doc_position(doc, selection.get("graph_order_doc_indices", []) or []),
        "certified_pos": _doc_position(doc, retrieval.get("certified_doc_indices", []) or []),
    }


def run_audit(
    *,
    base_report: Path,
    reference_report: Path,
    max_queries: int,
) -> Mapping[str, Any]:
    base_payload = _load_json(base_report)
    reference_payload = _load_json(reference_report)
    reference_rows = {
        int(row.get("query_index", idx)): row
        for idx, row in enumerate(reference_payload.get("rows", []) or [])
    }
    base_rows = list(base_payload.get("rows", []) or [])
    if max_queries > 0:
        base_rows = base_rows[:max_queries]

    query_delta_counts: Counter = Counter()
    base_missing_outcome_counts: Counter = Counter()
    stage_pair_counts: Counter = Counter()
    recovered_by_depth: Counter = Counter()
    endpoint_delta_counts: Counter = Counter()
    examples: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows: list[Mapping[str, Any]] = []

    for fallback_idx, base_row in enumerate(base_rows):
        query_index = int(base_row.get("query_index", fallback_idx))
        reference_row = reference_rows.get(query_index)
        if reference_row is None:
            continue

        base_recall = _recall(base_row)
        reference_recall = _recall(reference_row)
        if reference_recall > base_recall:
            query_delta_counts["reference_better"] += 1
        elif reference_recall < base_recall:
            query_delta_counts["base_better"] += 1
        else:
            query_delta_counts["tie"] += 1

        base_endpoints = set(
            str(value)
            for value in ((_retrieval_trace(base_row).get("selection", {}) or {}).get("query_endpoints", []) or [])
            if str(value).strip()
        )
        reference_endpoints = set(
            str(value)
            for value in ((_retrieval_trace(reference_row).get("selection", {}) or {}).get("query_endpoints", []) or [])
            if str(value).strip()
        )
        if reference_endpoints != base_endpoints:
            endpoint_delta_counts["changed"] += 1
        else:
            endpoint_delta_counts["same"] += 1

        base_selected = set(_selected(base_row))
        reference_selected = set(_selected(reference_row))
        gold_docs = _gold(base_row)
        missing_docs: list[Mapping[str, Any]] = []
        for gold_doc in gold_docs:
            if int(gold_doc) in base_selected:
                continue
            base_snapshot = _doc_snapshot(base_row, int(gold_doc))
            reference_snapshot = _doc_snapshot(reference_row, int(gold_doc))
            base_stage = str(base_snapshot.get("stage"))
            reference_stage = str(reference_snapshot.get("stage"))
            outcome = "reference_selected" if int(gold_doc) in reference_selected else reference_stage
            base_missing_outcome_counts[outcome] += 1
            stage_pair_counts[f"{base_stage} -> {outcome}"] += 1
            if int(gold_doc) in reference_selected:
                recovered_by_depth[str(len(gold_docs))] += 1

            row = {
                "query_index": query_index,
                "question": str(base_row.get("question") or ""),
                "gold_doc": int(gold_doc),
                "gold_doc_count": len(gold_docs),
                "base_recall_at5": base_recall,
                "reference_recall_at5": reference_recall,
                "base_top5": _selected(base_row),
                "reference_top5": _selected(reference_row),
                "base_stage": base_stage,
                "reference_stage": reference_stage,
                "reference_outcome": outcome,
                "base_snapshot": dict(base_snapshot),
                "reference_snapshot": dict(reference_snapshot),
                "base_query_endpoints": sorted(base_endpoints),
                "reference_query_endpoints": sorted(reference_endpoints),
            }
            missing_docs.append(row)
            key = f"{base_stage} -> {outcome}"
            if len(examples[key]) < 5:
                examples[key].append(row)
        if missing_docs:
            rows.append(
                {
                    "query_index": query_index,
                    "question": str(base_row.get("question") or ""),
                    "gold_doc_indices": gold_docs,
                    "base_top5": _selected(base_row),
                    "reference_top5": _selected(reference_row),
                    "base_recall_at5": base_recall,
                    "reference_recall_at5": reference_recall,
                    "base_all_gold_at5": _all_gold(base_row),
                    "reference_all_gold_at5": _all_gold(reference_row),
                    "base_query_endpoints": sorted(base_endpoints),
                    "reference_query_endpoints": sorted(reference_endpoints),
                    "missing_gold_docs": missing_docs,
                }
            )

    return {
        "audit": "openie_substrate_trace_gap",
        "dataset": str(base_payload.get("dataset") or ""),
        "row_count": len(base_rows),
        "base_report": str(base_report),
        "reference_report": str(reference_report),
        "base_openie_path": str(base_payload.get("openie_path") or ""),
        "reference_openie_path": str(reference_payload.get("openie_path") or ""),
        "base_metrics": dict(base_payload.get("metrics", {}) or {}),
        "reference_metrics": dict(reference_payload.get("metrics", {}) or {}),
        "query_delta_counts": dict(query_delta_counts),
        "endpoint_delta_counts": dict(endpoint_delta_counts),
        "base_missing_reference_outcome_counts": dict(base_missing_outcome_counts),
        "stage_pair_counts": dict(stage_pair_counts),
        "recovered_by_depth": dict(recovered_by_depth),
        "examples": {key: value for key, value in sorted(examples.items())},
        "rows": rows,
    }


def _format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def _format_counter_table(lines: list[str], title: str, counter: Mapping[str, Any], *, numeric_left: bool = False) -> None:
    lines.extend(["", f"## {title}", ""])
    if numeric_left:
        lines.extend(["| key | count |", "| ---: | ---: |"])
    else:
        lines.extend(["| key | count |", "| --- | ---: |"])
    for key, count in sorted(counter.items()):
        lines.append(f"| {key} | {int(count)} |")


def write_markdown(payload: Mapping[str, Any], output_md: Path) -> None:
    base_metrics = payload.get("base_metrics", {}) or {}
    reference_metrics = payload.get("reference_metrics", {}) or {}
    lines = [
        "# OpenIE/STO Trace Gap Audit",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| rows | {payload.get('row_count', 0)} |",
        f"| base report | `{payload.get('base_report', '')}` |",
        f"| reference report | `{payload.get('reference_report', '')}` |",
        f"| base OpenIE | `{payload.get('base_openie_path', '')}` |",
        f"| reference OpenIE | `{payload.get('reference_openie_path', '')}` |",
        "",
        "| substrate | R@5 | all-gold@5 | mean certified docs@5 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| base | {_format_float(base_metrics.get('r5'))} | "
            f"{_format_float(base_metrics.get('all_gold_at5'))} | "
            f"{_format_float(base_metrics.get('mean_certified_doc_count_top5'))} |"
        ),
        (
            f"| reference | {_format_float(reference_metrics.get('r5'))} | "
            f"{_format_float(reference_metrics.get('all_gold_at5'))} | "
            f"{_format_float(reference_metrics.get('mean_certified_doc_count_top5'))} |"
        ),
    ]
    _format_counter_table(lines, "Query Delta", payload.get("query_delta_counts", {}) or {})
    _format_counter_table(lines, "Query Endpoint Delta", payload.get("endpoint_delta_counts", {}) or {})
    _format_counter_table(
        lines,
        "Base-Missing Gold Outcome On Reference",
        payload.get("base_missing_reference_outcome_counts", {}) or {},
    )
    _format_counter_table(lines, "Base Stage -> Reference Outcome", payload.get("stage_pair_counts", {}) or {})
    _format_counter_table(lines, "Reference Recovered Gold By Depth", payload.get("recovered_by_depth", {}) or {}, numeric_left=True)

    lines.extend(["", "## Examples"])
    for key, rows in sorted((payload.get("examples", {}) or {}).items()):
        lines.extend(
            [
                "",
                f"### {key}",
                "",
                "| query | gold_doc | base_top5 | reference_top5 | base_R@5 | reference_R@5 |",
                "| ---: | ---: | --- | --- | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {int(row.get('query_index', -1))} | {int(row.get('gold_doc', -1))} | "
                f"{row.get('base_top5', [])} | {row.get('reference_top5', [])} | "
                f"{_format_float(row.get('base_recall_at5'))} | "
                f"{_format_float(row.get('reference_recall_at5'))} |"
            )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args(argv)

    payload = run_audit(
        base_report=args.base_report,
        reference_report=args.reference_report,
        max_queries=int(args.max_queries),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
