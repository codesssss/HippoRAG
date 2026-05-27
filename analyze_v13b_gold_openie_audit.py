#!/usr/bin/env python3
"""Audit whether Qwen OpenIE gold documents contain V13B-groundable facts.

This script is diagnostic-only.  It does not change retrieval, matching, PPR,
or fallback behavior.  It answers a narrower question than
``analyze_v13b_grounding_failures.py``:

When a retrieval-critical query obligation is not grounded, do the gold
documents contain a role-aligned OpenIE fact for that obligation, or is the
corpus evidence graph itself missing/misrepresenting the needed fact?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from analyze_v13b_grounding_failures import (
    build_units_for_docs,
    cache_triples_by_query,
    load_json,
    merge_match_maps,
    obligation_endpoint_values,
    openie_docs_from_payload,
    retrieval_critical_obligations_for_analysis,
    role_endpoint_matches,
    selector_rows,
    source_rows_by_query,
    write_json,
)
from build_query_obligation_units import build_query_obligation_units, match_query_obligations_to_sto_facts
from evaluate_obligation_closed_sto_local_ppr import query_triples_from_row
from evaluate_obligation_closed_sto_selector import candidate_doc_indices_from_row, unique_ints
from query_obligation_support_grounding import variable_values_from_matches
from query_obligation_typing import lower_query_triples_for_typed_program
from v13b_graph_normalization import fact_role_keys, obligation_role_keys, safe_int


AUDIT_STATUSES = (
    "candidate_missing",
    "gold_exact_openie_match_available",
    "gold_endpoint_relation_mismatch",
    "gold_endpoint_missing",
    "gold_relation_present_without_endpoint_binding",
    "gold_openie_no_facts",
    "gold_docs_missing",
    "unknown",
)


def fact_rows_for_units(units: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [fact_role_keys(unit) for unit in units if str(unit.get("unit_type") or "") == "openie_fact"]


def relation_keys_for_rows(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("relation_key") or "<empty>") for row in rows)


def classify_gold_openie_status(
    *,
    obligation: Mapping[str, Any],
    gold_units: Sequence[Mapping[str, Any]],
    candidate_doc_indices: Sequence[int],
    gold_doc_indices: Sequence[int],
    grounded_variable_values: Mapping[str, Iterable[str]],
) -> Dict[str, Any]:
    gold_docs = unique_ints(gold_doc_indices)
    candidate_docs = set(unique_ints(candidate_doc_indices))
    if not gold_docs:
        return {"status": "gold_docs_missing"}
    if not set(gold_docs) <= candidate_docs:
        return {
            "status": "candidate_missing",
            "missing_gold_doc_indices": sorted(set(gold_docs) - candidate_docs),
        }

    exact = match_query_obligations_to_sto_facts(obligations=[obligation], candidate_units=gold_units)
    obligation_id = str(obligation.get("obligation_id") or "")
    if exact.get(obligation_id):
        return {
            "status": "gold_exact_openie_match_available",
            "matched_doc_indices": unique_ints(match.get("doc_index") for match in exact.get(obligation_id, []) or []),
        }

    fact_rows = fact_rows_for_units(gold_units)
    if not fact_rows:
        return {"status": "gold_openie_no_facts"}

    ob = obligation_role_keys(obligation)
    subject_values = obligation_endpoint_values(
        obligation=obligation,
        role="subject",
        grounded_variable_values=grounded_variable_values,
    )
    object_values = obligation_endpoint_values(
        obligation=obligation,
        role="object",
        grounded_variable_values=grounded_variable_values,
    )
    relation_hits = [row for row in fact_rows if row.get("relation_key") == ob["relation_key"]]
    endpoint_hits = [
        row
        for row in fact_rows
        if (not subject_values or role_endpoint_matches(endpoint_values=subject_values, fact_row=row, role="subject"))
        and (not object_values or role_endpoint_matches(endpoint_values=object_values, fact_row=row, role="object"))
    ]
    if endpoint_hits:
        return {
            "status": "gold_endpoint_relation_mismatch",
            "endpoint_hit_relation_counts": dict(relation_keys_for_rows(endpoint_hits).most_common(20)),
        }
    if relation_hits:
        return {
            "status": "gold_relation_present_without_endpoint_binding",
            "relation_hit_doc_indices": unique_ints(row.get("doc_index") for row in relation_hits),
        }
    return {
        "status": "gold_endpoint_missing",
        "gold_relation_key_counts": dict(relation_keys_for_rows(fact_rows).most_common(20)),
    }


def representative_fact_examples(units: Sequence[Mapping[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in units:
        if str(unit.get("unit_type") or "") != "openie_fact":
            continue
        rows.append(
            {
                "doc_index": safe_int(unit.get("doc_index")),
                "title": str(unit.get("title") or ""),
                "fact": list(unit.get("fact", []) or []),
                "relation": str(unit.get("relation") or ""),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def audit_gold_openie(
    *,
    dataset: str,
    source_report_path: Path,
    selector_json_path: Path,
    query_obligation_cache_path: Path,
    openie_json_path: Path,
    max_queries: int = 0,
    max_cases_per_status: int = 10,
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

    status_counts: Counter[str] = Counter()
    failure_type_by_status: Dict[str, Counter[str]] = defaultdict(Counter)
    relation_pair_counts: Counter[str] = Counter()
    representatives: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    audited_count = 0

    for row in rows:
        query_index = safe_int(row.get("query_index"))
        selector = row.get("obligation_closed_sto_local_ppr_selector", {}) or {}
        source_row = source_by_query.get(query_index, row)
        candidate_docs = candidate_doc_indices_from_row(source_row)
        gold_docs = unique_ints(source_row.get("gold_doc_indices", []) or [])
        candidate_units = build_units_for_docs(doc_indices=candidate_docs, openie_docs=openie_docs)
        gold_units = build_units_for_docs(doc_indices=gold_docs, openie_docs=openie_docs)
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
        exact_matches = match_query_obligations_to_sto_facts(obligations=obligations, candidate_units=candidate_units)
        support_matches = selector.get("support_grounded_obligation_matches", {}) or {}
        source_span_matches = selector.get("source_span_grounded_obligation_matches", {}) or {}
        grounded_ids = {
            str(obligation_id)
            for match_map in (support_matches, source_span_matches)
            for obligation_id, matches in match_map.items()
            if matches
        }
        grounded_variable_values = variable_values_from_matches(
            obligations=obligations,
            matches=merge_match_maps(exact_matches, support_matches, source_span_matches),
        )
        for obligation in obligations:
            obligation_id = str(obligation.get("obligation_id") or "")
            if exact_matches.get(obligation_id) or obligation_id in grounded_ids:
                continue
            audited_count += 1
            status = classify_gold_openie_status(
                obligation=obligation,
                gold_units=gold_units,
                candidate_doc_indices=candidate_docs,
                gold_doc_indices=gold_docs,
                grounded_variable_values=grounded_variable_values,
            )
            status_name = str(status.get("status") or "unknown")
            if status_name not in AUDIT_STATUSES:
                status_name = "unknown"
            status_counts[status_name] += 1
            failure_type = str(selector.get("abstention_reason") or "")
            failure_type_by_status[status_name][failure_type] += 1
            ob = obligation_role_keys(obligation)
            if status_name == "gold_endpoint_relation_mismatch":
                for relation_key, count in (status.get("endpoint_hit_relation_counts", {}) or {}).items():
                    relation_pair_counts[f"{ob['relation_key']} -> {relation_key}"] += int(count or 0)
            if len(representatives[status_name]) < max_cases_per_status:
                representatives[status_name].append(
                    {
                        "query_index": query_index,
                        "question": str(source_row.get("question") or source_row.get("query") or ""),
                        "raw_triple": ob["raw_triple"],
                        "normalized_obligation": ob,
                        "gold_doc_indices": gold_docs,
                        "candidate_doc_count": len(unique_ints(candidate_docs)),
                        **status,
                        "gold_fact_examples": representative_fact_examples(gold_units),
                    }
                )

    return {
        "dataset": dataset,
        "source_report_path": str(source_report_path),
        "selector_json_path": str(selector_json_path),
        "query_obligation_cache_path": str(query_obligation_cache_path),
        "openie_json_path": str(openie_json_path),
        "max_queries": int(max_queries),
        "audited_ungrounded_obligation_count": audited_count,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_type_by_status": {key: dict(counter) for key, counter in sorted(failure_type_by_status.items())},
        "relation_mismatch_pair_counts": dict(relation_pair_counts.most_common(40)),
        "representative_cases": dict(representatives),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"# V13B Gold OpenIE Audit: {report.get('dataset')}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| audited_ungrounded_obligation_count | {report.get('audited_ungrounded_obligation_count', 0)} |",
        "",
        "## Gold OpenIE Status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, value in (report.get("status_counts", {}) or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Relation Mismatch Pairs On Gold Docs", "", "| query_relation -> gold_fact_relation | count |", "|---|---:|"])
    for key, value in (report.get("relation_mismatch_pair_counts", {}) or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Representative Cases", ""])
    for status, cases in (report.get("representative_cases", {}) or {}).items():
        lines.append(f"### {status}")
        for case in cases[:5]:
            lines.append(f"- q={case.get('query_index')} triple={case.get('raw_triple')} gold={case.get('gold_doc_indices')}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit gold-doc OpenIE evidence for V13B grounding failures.")
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--selector-json", required=True)
    parser.add_argument("--query-obligation-cache", required=True)
    parser.add_argument("--openie-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-cases-per-status", type=int, default=10)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = audit_gold_openie(
        dataset=str(args.dataset),
        source_report_path=Path(args.source_report).resolve(),
        selector_json_path=Path(args.selector_json).resolve(),
        query_obligation_cache_path=Path(args.query_obligation_cache).resolve(),
        openie_json_path=Path(args.openie_json).resolve(),
        max_queries=max(int(args.max_queries), 0),
        max_cases_per_status=max(int(args.max_cases_per_status), 1),
    )
    write_json(Path(args.output_json), report)
    write_markdown(Path(args.output_md), report)
    print(json.dumps({k: report[k] for k in ("dataset", "audited_ungrounded_obligation_count", "status_counts")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
