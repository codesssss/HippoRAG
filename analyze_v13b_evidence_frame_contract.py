#!/usr/bin/env python3
"""Audit whether gold OpenIE satisfies V13B evidence-frame contracts.

This script is diagnostic-only.  It does not change retrieval, graph
construction, grounding, PPR, or fallback behavior.  It separates two questions:

1. Did the structural source report include the gold documents in the candidate
   universe?
2. If we look directly at the gold documents, did Qwen OpenIE express the
   retrieval-critical query demand as a role-aligned evidence frame?

The second question is the clean substrate check: if gold documents are present
but the gold OpenIE frame is missing or relation-mismatched, selector tuning
cannot fix the root cause.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from analyze_v13b_gold_openie_audit import classify_gold_openie_status, representative_fact_examples
from analyze_v13b_grounding_failures import (
    build_units_for_docs,
    cache_triples_by_query,
    load_json,
    merge_match_maps,
    openie_docs_from_payload,
    retrieval_critical_obligations_for_analysis,
    selector_rows,
    source_rows_by_query,
    write_json,
)
from build_query_obligation_units import build_query_obligation_units, match_query_obligations_to_sto_facts
from evaluate_obligation_closed_sto_local_ppr import query_triples_from_row
from evaluate_obligation_closed_sto_selector import candidate_doc_indices_from_row, unique_ints
from query_obligation_support_grounding import variable_values_from_matches
from query_obligation_typing import lower_query_triples_for_typed_program
from summarize_v13b_gold_openie_schema_gaps import endpoint_shape, relation_family
from v13b_graph_normalization import obligation_role_keys, safe_int


def candidate_coverage_status(*, candidate_doc_indices: Sequence[int], gold_doc_indices: Sequence[int]) -> str:
    gold_docs = set(unique_ints(gold_doc_indices))
    if not gold_docs:
        return "gold_docs_missing"
    if gold_docs <= set(unique_ints(candidate_doc_indices)):
        return "gold_docs_covered"
    return "candidate_missing"


def bound_endpoint_surfaces_for_obligation(obligation: Mapping[str, Any]) -> List[str]:
    ob = obligation_role_keys(obligation)
    surfaces: List[str] = []
    if not ob["subject_is_variable"] and ob["subject_surface"].strip():
        surfaces.append(ob["subject_surface"])
    if not ob["object_is_variable"] and ob["object_surface"].strip():
        surfaces.append(ob["object_surface"])
    return surfaces


def obligation_endpoint_shape(obligation: Mapping[str, Any]) -> str:
    surfaces = bound_endpoint_surfaces_for_obligation(obligation)
    if not surfaces:
        return "all_variable_obligation"
    shapes = [endpoint_shape(surface) for surface in surfaces]
    for shape in (
        "descriptive_bound_endpoint",
        "typed_title_endpoint",
        "title_qualified_endpoint",
        "long_bound_endpoint",
        "named_endpoint",
        "empty_endpoint",
    ):
        if shape in shapes:
            return shape
    return shapes[0]


def capped_append(bucket: Dict[str, List[Dict[str, Any]]], key: str, row: Dict[str, Any], *, limit: int) -> None:
    if len(bucket[key]) < limit:
        bucket[key].append(row)


def audit_evidence_frame_contract(
    *,
    dataset: str,
    source_report_path: Path,
    selector_json_path: Path,
    query_obligation_cache_path: Path,
    openie_json_path: Path,
    max_queries: int = 0,
    max_examples_per_bucket: int = 5,
) -> Dict[str, Any]:
    source_report = load_json(source_report_path)
    selector_payload = load_json(selector_json_path)
    cache_payload = load_json(query_obligation_cache_path)
    openie_docs = openie_docs_from_payload(load_json(openie_json_path))
    source_by_query = source_rows_by_query(source_report, dataset)
    cache_by_query = cache_triples_by_query(cache_payload, dataset)
    rows = selector_rows(selector_payload, dataset)
    if int(max_queries) > 0:
        rows = rows[: int(max_queries)]

    candidate_status_counts: Counter[str] = Counter()
    frame_status_counts: Counter[str] = Counter()
    status_by_relation_family: Dict[str, Counter[str]] = defaultdict(Counter)
    status_by_endpoint_shape: Dict[str, Counter[str]] = defaultdict(Counter)
    relation_key_counts: Counter[str] = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_obligations = 0

    for row in rows:
        query_index = safe_int(row.get("query_index"))
        selector = row.get("obligation_closed_sto_local_ppr_selector", {}) or {}
        source_row = source_by_query.get(query_index, row)
        candidate_docs = candidate_doc_indices_from_row(source_row)
        gold_docs = unique_ints(source_row.get("gold_doc_indices", []) or [])
        gold_units = build_units_for_docs(doc_indices=gold_docs, openie_docs=openie_docs)
        candidate_units = build_units_for_docs(doc_indices=candidate_docs, openie_docs=openie_docs)
        query_triples = query_triples_from_row(source_row, query_obligation_cache=cache_by_query)
        program_triples = lower_query_triples_for_typed_program(
            query=str(source_row.get("question") or source_row.get("query") or ""),
            query_triples=query_triples,
        )
        obligations = selector.get("query_obligations") or build_query_obligation_units(
            query=str(source_row.get("question") or source_row.get("query") or ""),
            query_triples=program_triples,
        )
        obligations = retrieval_critical_obligations_for_analysis(obligations=obligations, selector=selector)
        exact_candidate_matches = match_query_obligations_to_sto_facts(obligations=obligations, candidate_units=candidate_units)
        support_matches = selector.get("support_grounded_obligation_matches", {}) or {}
        source_span_matches = selector.get("source_span_grounded_obligation_matches", {}) or {}
        grounded_variable_values = variable_values_from_matches(
            obligations=obligations,
            matches=merge_match_maps(exact_candidate_matches, support_matches, source_span_matches),
        )

        coverage_status = candidate_coverage_status(candidate_doc_indices=candidate_docs, gold_doc_indices=gold_docs)
        candidate_status_counts[coverage_status] += len(obligations)

        for obligation in obligations:
            total_obligations += 1
            ob = obligation_role_keys(obligation)
            family = relation_family(ob["relation_key"])
            shape = obligation_endpoint_shape(obligation)
            relation_key_counts[ob["relation_key"]] += 1
            status = classify_gold_openie_status(
                obligation=obligation,
                gold_units=gold_units,
                candidate_doc_indices=gold_docs,
                gold_doc_indices=gold_docs,
                grounded_variable_values=grounded_variable_values,
            )
            frame_status = str(status.get("status") or "unknown")
            frame_status_counts[frame_status] += 1
            status_by_relation_family[family][frame_status] += 1
            status_by_endpoint_shape[shape][frame_status] += 1

            if frame_status != "gold_exact_openie_match_available":
                capped_append(
                    examples,
                    f"{family}::{frame_status}",
                    {
                        "query_index": query_index,
                        "question": str(source_row.get("question") or source_row.get("query") or ""),
                        "raw_triple": ob["raw_triple"],
                        "relation_key": ob["relation_key"],
                        "relation_family": family,
                        "endpoint_shape": shape,
                        "bound_endpoints": bound_endpoint_surfaces_for_obligation(obligation),
                        "gold_doc_indices": gold_docs,
                        "candidate_coverage_status": coverage_status,
                        "frame_status": frame_status,
                        "status_details": status,
                        "gold_fact_examples": representative_fact_examples(gold_units, limit=3),
                    },
                    limit=max_examples_per_bucket,
                )

    return {
        "dataset": dataset,
        "source_report_path": str(source_report_path),
        "selector_json_path": str(selector_json_path),
        "query_obligation_cache_path": str(query_obligation_cache_path),
        "openie_json_path": str(openie_json_path),
        "max_queries": int(max_queries),
        "retrieval_critical_obligation_count": total_obligations,
        "candidate_coverage_status_counts": dict(candidate_status_counts.most_common()),
        "gold_openie_frame_status_counts": dict(frame_status_counts.most_common()),
        "gold_openie_frame_status_fractions": {
            key: round(value / max(1, total_obligations), 6)
            for key, value in frame_status_counts.most_common()
        },
        "status_by_relation_family": {
            key: dict(counter.most_common())
            for key, counter in sorted(status_by_relation_family.items())
        },
        "status_by_endpoint_shape": {
            key: dict(counter.most_common())
            for key, counter in sorted(status_by_endpoint_shape.items())
        },
        "relation_key_counts": dict(relation_key_counts.most_common(40)),
        "representative_failures": dict(sorted(examples.items())),
        "interpretation": interpret_contract_counts(total_obligations=total_obligations, frame_status_counts=frame_status_counts),
    }


def interpret_contract_counts(*, total_obligations: int, frame_status_counts: Counter[str]) -> Dict[str, Any]:
    exact = frame_status_counts.get("gold_exact_openie_match_available", 0)
    relation_mismatch = frame_status_counts.get("gold_endpoint_relation_mismatch", 0)
    endpoint_missing = frame_status_counts.get("gold_endpoint_missing", 0)
    coverage = exact / max(1, total_obligations)
    schema_gap = (relation_mismatch + endpoint_missing) / max(1, total_obligations)
    if schema_gap >= 0.5:
        bottleneck = "gold_openie_schema_contract"
    elif coverage >= 0.5:
        bottleneck = "selector_or_candidate_use_after_schema"
    else:
        bottleneck = "mixed"
    return {
        "gold_openie_exact_frame_rate": round(coverage, 6),
        "endpoint_relation_or_endpoint_missing_rate": round(schema_gap, 6),
        "main_bottleneck": bottleneck,
        "recommended_next_step": recommended_next_step(bottleneck),
    }


def recommended_next_step(bottleneck: str) -> str:
    if bottleneck == "gold_openie_schema_contract":
        return "Create and test a Qwen OpenIE evidence-frame schema contract before tuning selector parameters."
    if bottleneck == "selector_or_candidate_use_after_schema":
        return "Inspect why existing gold OpenIE frames are not consumed by grounding/selection."
    return "Split by relation family and endpoint shape before changing method behavior."


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" if idx == 0 else "---:" for idx, _ in enumerate(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    frame_statuses = sorted((report.get("gold_openie_frame_status_counts", {}) or {}).keys())
    family_rows = []
    for family, counts in (report.get("status_by_relation_family", {}) or {}).items():
        total = sum(int(value or 0) for value in (counts or {}).values())
        family_rows.append([family, total] + [counts.get(status, 0) for status in frame_statuses])
    shape_rows = []
    for shape, counts in (report.get("status_by_endpoint_shape", {}) or {}).items():
        total = sum(int(value or 0) for value in (counts or {}).values())
        shape_rows.append([shape, total] + [counts.get(status, 0) for status in frame_statuses])

    lines = [
        f"# V13B Evidence Frame Contract Audit: {report.get('dataset')}",
        "",
        "This report is diagnostic-only. It separates candidate coverage from gold-document OpenIE frame compliance.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| retrieval_critical_obligation_count | {report.get('retrieval_critical_obligation_count', 0)} |",
    ]
    for key, value in (report.get("candidate_coverage_status_counts", {}) or {}).items():
        lines.append(f"| candidate_coverage:{key} | {value} |")
    for key, value in (report.get("gold_openie_frame_status_counts", {}) or {}).items():
        lines.append(f"| gold_openie_frame:{key} | {value} |")

    lines.extend(["", "## Interpretation", "", "| item | value |", "|---|---|"])
    for key, value in (report.get("interpretation", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Status By Relation Family", ""])
    lines.extend(markdown_table(["relation_family", "total"] + frame_statuses, family_rows))
    lines.extend(["", "## Status By Endpoint Shape", ""])
    lines.extend(markdown_table(["endpoint_shape", "total"] + frame_statuses, shape_rows))

    lines.extend(["", "## Top Query Relation Keys", "", "| relation_key | count |", "|---|---:|"])
    for key, value in (report.get("relation_key_counts", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Representative Failures", ""])
    for bucket, cases in (report.get("representative_failures", {}) or {}).items():
        lines.append(f"### {bucket}")
        for case in cases[:5]:
            lines.append(
                f"- q={case.get('query_index')} relation={case.get('relation_key')} "
                f"endpoint_shape={case.get('endpoint_shape')} triple={case.get('raw_triple')} "
                f"coverage={case.get('candidate_coverage_status')}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit V13B gold OpenIE evidence-frame contract compliance.")
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--selector-json", required=True)
    parser.add_argument("--query-obligation-cache", required=True)
    parser.add_argument("--openie-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-examples-per-bucket", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = audit_evidence_frame_contract(
        dataset=str(args.dataset),
        source_report_path=Path(args.source_report).resolve(),
        selector_json_path=Path(args.selector_json).resolve(),
        query_obligation_cache_path=Path(args.query_obligation_cache).resolve(),
        openie_json_path=Path(args.openie_json).resolve(),
        max_queries=max(int(args.max_queries), 0),
        max_examples_per_bucket=max(int(args.max_examples_per_bucket), 1),
    )
    write_json(Path(args.output_json), report)
    write_markdown(Path(args.output_md), report)
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "retrieval_critical_obligation_count": report["retrieval_critical_obligation_count"],
                "candidate_coverage_status_counts": report["candidate_coverage_status_counts"],
                "gold_openie_frame_status_counts": report["gold_openie_frame_status_counts"],
                "interpretation": report["interpretation"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
