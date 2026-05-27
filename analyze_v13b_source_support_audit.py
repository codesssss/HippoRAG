#!/usr/bin/env python3
"""Audit schema-light source support for V13B query demands.

This script is diagnostic-only.  It does not change retrieval, graph
construction, grounding, PPR, OpenIE prompts, or fallback behavior.

It asks a narrower question than exact OpenIE-frame audits:

When an exact OpenIE grounding fails, does the gold document source text still
contain anchored, predicate-bearing evidence that appears to support the query
demand?

The support check is intentionally conservative and schema-light.  It does not
use relation synonym tables or benchmark-specific relation inventories.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

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
from build_query_obligation_units import (
    build_query_obligation_units,
    endpoint_signature_tokens,
    match_query_obligations_to_sto_facts,
)
from evaluate_obligation_closed_sto_local_ppr import query_triples_from_row
from evaluate_obligation_closed_sto_selector import candidate_doc_indices_from_row, unique_ints
from query_obligation_support_grounding import (
    content_relation_tokens_for_obligation,
    content_relation_tokens_for_text,
    normalized_token_set,
    obligation_anchor_values,
    simple_source_sentences,
    variable_values_from_matches,
)
from query_obligation_typing import lower_query_triples_for_typed_program
from summarize_v13b_gold_openie_schema_gaps import endpoint_shape
from v13b_graph_normalization import obligation_role_keys, safe_int


SOURCE_SUPPORT_STATUSES = (
    "openie_exact_available",
    "openie_failed_but_source_supports",
    "source_support_missing",
    "descriptive_endpoint_query_issue",
    "unresolved_variable_issue",
    "gold_docs_missing",
)


def doc_title(doc: Mapping[str, Any]) -> str:
    passage = str(doc.get("passage") or "")
    if passage:
        return passage.split("\n", 1)[0].strip()
    return str(doc.get("title") or doc.get("idx") or "")


def doc_sentences(*, doc_index: int, doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    title = doc_title(doc)
    passage = str(doc.get("passage") or "")
    sentences: List[Dict[str, Any]] = []
    for sentence_index, sentence in enumerate(simple_source_sentences(passage)):
        sentences.append(
            {
                "doc_index": int(doc_index),
                "title": title,
                "sentence_index": int(sentence_index),
                "sentence": sentence,
                "sentence_tokens": normalized_token_set(sentence),
                "title_tokens": normalized_token_set(title),
            }
        )
    return sentences


def source_sentences_for_docs(
    *,
    doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc_index in unique_ints(doc_indices):
        if 0 <= doc_index < len(openie_docs):
            rows.extend(doc_sentences(doc_index=doc_index, doc=openie_docs[doc_index]))
    return rows


def endpoint_tokens(value: str) -> Set[str]:
    return {token for token in endpoint_signature_tokens(value) if token}


def anchor_hits_for_tokens(anchors: Iterable[str], tokens: Set[str]) -> List[str]:
    hits: List[str] = []
    for anchor in sorted({str(anchor) for anchor in anchors if str(anchor)}):
        anchor_tokens = endpoint_tokens(anchor)
        if anchor_tokens and anchor_tokens <= tokens:
            hits.append(anchor)
    return hits


def bound_endpoint_shape_for_obligation(obligation: Mapping[str, Any]) -> str:
    ob = obligation_role_keys(obligation)
    surfaces: List[str] = []
    if not ob["subject_is_variable"] and ob["subject_surface"].strip():
        surfaces.append(ob["subject_surface"])
    if not ob["object_is_variable"] and ob["object_surface"].strip():
        surfaces.append(ob["object_surface"])
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


def source_support_candidates_for_obligation(
    *,
    obligation: Mapping[str, Any],
    source_sentences: Sequence[Mapping[str, Any]],
    variable_values: Mapping[str, Set[str]],
    max_examples: int = 5,
) -> List[Dict[str, Any]]:
    anchors = obligation_anchor_values(obligation=obligation, variable_values=variable_values)
    if not anchors["any"]:
        return []
    relation_tokens = content_relation_tokens_for_obligation(obligation)
    if not relation_tokens:
        return []

    rows: List[Dict[str, Any]] = []
    for row in source_sentences:
        sentence_tokens = set(row.get("sentence_tokens", set()) or set())
        title_tokens = set(row.get("title_tokens", set()) or set())
        subject_sentence_hits = anchor_hits_for_tokens(anchors["subject"], sentence_tokens)
        object_sentence_hits = anchor_hits_for_tokens(anchors["object"], sentence_tokens)
        subject_title_hits = anchor_hits_for_tokens(anchors["subject"], title_tokens)
        object_title_hits = anchor_hits_for_tokens(anchors["object"], title_tokens)

        subject_satisfied = not anchors["subject"] or bool(subject_sentence_hits) or bool(subject_title_hits)
        object_satisfied = not anchors["object"] or bool(object_sentence_hits) or bool(object_title_hits)
        if not (subject_satisfied and object_satisfied):
            continue
        if not (subject_sentence_hits or object_sentence_hits or subject_title_hits or object_title_hits):
            continue

        relation_hits = sorted(relation_tokens & (sentence_tokens | content_relation_tokens_for_text(row.get("sentence"))))
        if not relation_hits:
            continue

        rows.append(
            {
                "doc_index": safe_int(row.get("doc_index")),
                "title": str(row.get("title") or ""),
                "sentence_index": safe_int(row.get("sentence_index")),
                "sentence": str(row.get("sentence") or "")[:500],
                "relation_token_hits": relation_hits,
                "subject_sentence_anchor_hits": subject_sentence_hits,
                "object_sentence_anchor_hits": object_sentence_hits,
                "subject_title_anchor_hits": subject_title_hits,
                "object_title_anchor_hits": object_title_hits,
            }
        )

    rows.sort(
        key=lambda item: (
            not bool(item["subject_sentence_anchor_hits"] or item["object_sentence_anchor_hits"]),
            -len(item["relation_token_hits"]),
            int(item["doc_index"]),
            int(item["sentence_index"]),
        )
    )
    return rows[: max(int(max_examples), 1)]


def classify_source_support_status(
    *,
    obligation: Mapping[str, Any],
    exact_gold_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    source_support_candidates: Sequence[Mapping[str, Any]],
    gold_doc_indices: Sequence[int],
    variable_values: Mapping[str, Set[str]],
) -> str:
    if not unique_ints(gold_doc_indices):
        return "gold_docs_missing"
    obligation_id = str(obligation.get("obligation_id") or "")
    if exact_gold_matches.get(obligation_id):
        return "openie_exact_available"
    if source_support_candidates:
        return "openie_failed_but_source_supports"
    if not obligation_anchor_values(obligation=obligation, variable_values=variable_values)["any"]:
        return "unresolved_variable_issue"
    if bound_endpoint_shape_for_obligation(obligation) == "descriptive_bound_endpoint":
        return "descriptive_endpoint_query_issue"
    return "source_support_missing"


def capped_append(bucket: Dict[str, List[Dict[str, Any]]], key: str, value: Dict[str, Any], *, limit: int) -> None:
    if len(bucket[key]) < limit:
        bucket[key].append(value)


def audit_source_support(
    *,
    dataset: str,
    source_report_path: Path,
    selector_json_path: Path,
    query_obligation_cache_path: Path,
    openie_json_path: Path,
    max_queries: int = 0,
    max_examples_per_status: int = 8,
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
    candidate_coverage_counts: Counter[str] = Counter()
    status_by_endpoint_shape: Dict[str, Counter[str]] = defaultdict(Counter)
    status_by_relation_token_count: Counter[str] = Counter()
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
        gold_sentences = source_sentences_for_docs(doc_indices=gold_docs, openie_docs=openie_docs)

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
        exact_gold_matches = match_query_obligations_to_sto_facts(obligations=obligations, candidate_units=gold_units)
        support_matches = selector.get("support_grounded_obligation_matches", {}) or {}
        source_span_matches = selector.get("source_span_grounded_obligation_matches", {}) or {}
        variable_values = variable_values_from_matches(
            obligations=obligations,
            matches=merge_match_maps(exact_candidate_matches, support_matches, source_span_matches),
        )

        coverage_status = "gold_docs_covered" if set(gold_docs) <= set(unique_ints(candidate_docs)) else "candidate_missing"
        if not gold_docs:
            coverage_status = "gold_docs_missing"
        candidate_coverage_counts[coverage_status] += len(obligations)

        for obligation in obligations:
            total_obligations += 1
            shape = bound_endpoint_shape_for_obligation(obligation)
            relation_tokens = content_relation_tokens_for_obligation(obligation)
            support_candidates = source_support_candidates_for_obligation(
                obligation=obligation,
                source_sentences=gold_sentences,
                variable_values=variable_values,
                max_examples=max_examples_per_status,
            )
            status = classify_source_support_status(
                obligation=obligation,
                exact_gold_matches=exact_gold_matches,
                source_support_candidates=support_candidates,
                gold_doc_indices=gold_docs,
                variable_values=variable_values,
            )
            status_counts[status] += 1
            status_by_endpoint_shape[shape][status] += 1
            status_by_relation_token_count[f"{len(relation_tokens)}_content_relation_tokens"] += 1
            if status != "openie_exact_available":
                ob = obligation_role_keys(obligation)
                capped_append(
                    examples,
                    status,
                    {
                        "query_index": query_index,
                        "question": str(source_row.get("question") or source_row.get("query") or ""),
                        "raw_triple": ob["raw_triple"],
                        "relation_key": ob["relation_key"],
                        "relation_tokens": sorted(relation_tokens),
                        "endpoint_shape": shape,
                        "gold_doc_indices": gold_docs,
                        "candidate_coverage_status": coverage_status,
                        "source_support_candidates": list(support_candidates[:3]),
                    },
                    limit=max_examples_per_status,
                )

    return {
        "dataset": dataset,
        "source_report_path": str(source_report_path),
        "selector_json_path": str(selector_json_path),
        "query_obligation_cache_path": str(query_obligation_cache_path),
        "openie_json_path": str(openie_json_path),
        "max_queries": int(max_queries),
        "retrieval_critical_obligation_count": total_obligations,
        "candidate_coverage_status_counts": dict(candidate_coverage_counts.most_common()),
        "source_support_status_counts": dict(status_counts.most_common()),
        "source_support_status_fractions": {
            key: round(value / max(1, total_obligations), 6)
            for key, value in status_counts.most_common()
        },
        "status_by_endpoint_shape": {
            key: dict(counter.most_common())
            for key, counter in sorted(status_by_endpoint_shape.items())
        },
        "relation_token_count_distribution": dict(status_by_relation_token_count.most_common()),
        "representative_cases": dict(sorted(examples.items())),
        "interpretation": interpret_counts(total_obligations=total_obligations, status_counts=status_counts),
    }


def interpret_counts(*, total_obligations: int, status_counts: Counter[str]) -> Dict[str, Any]:
    exact = status_counts.get("openie_exact_available", 0)
    source_support = status_counts.get("openie_failed_but_source_supports", 0)
    unresolved = status_counts.get("unresolved_variable_issue", 0)
    descriptive = status_counts.get("descriptive_endpoint_query_issue", 0)
    return {
        "openie_exact_rate": round(exact / max(1, total_obligations), 6),
        "openie_failed_but_source_supports_rate": round(source_support / max(1, total_obligations), 6),
        "unresolved_or_descriptive_query_issue_rate": round((unresolved + descriptive) / max(1, total_obligations), 6),
        "main_signal": main_signal(
            total_obligations=total_obligations,
            exact=exact,
            source_support=source_support,
            unresolved=unresolved,
            descriptive=descriptive,
        ),
    }


def main_signal(*, total_obligations: int, exact: int, source_support: int, unresolved: int, descriptive: int) -> str:
    if total_obligations <= 0:
        return "no_obligations"
    if source_support / total_obligations >= 0.2:
        return "source_support_exceeds_openie_exact_gap"
    if (unresolved + descriptive) / total_obligations >= 0.3:
        return "query_compiler_or_binding_bottleneck"
    if exact / total_obligations >= 0.5:
        return "openie_exact_often_available"
    return "source_support_not_established_by_conservative_audit"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" if index == 0 else "---:" for index, _ in enumerate(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    statuses = sorted((report.get("source_support_status_counts", {}) or {}).keys())
    shape_rows = []
    for shape, counts in (report.get("status_by_endpoint_shape", {}) or {}).items():
        total = sum(int(value or 0) for value in (counts or {}).values())
        shape_rows.append([shape, total] + [counts.get(status, 0) for status in statuses])

    lines = [
        f"# V13B Source-Support Audit: {report.get('dataset')}",
        "",
        "Diagnostic-only schema-light audit. It checks whether gold source text contains anchored predicate evidence for retrieval-critical demands.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| retrieval_critical_obligation_count | {report.get('retrieval_critical_obligation_count', 0)} |",
    ]
    for key, value in (report.get("candidate_coverage_status_counts", {}) or {}).items():
        lines.append(f"| candidate_coverage:{key} | {value} |")
    for key, value in (report.get("source_support_status_counts", {}) or {}).items():
        lines.append(f"| source_support:{key} | {value} |")

    lines.extend(["", "## Interpretation", "", "| item | value |", "|---|---|"])
    for key, value in (report.get("interpretation", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Status By Endpoint Shape", ""])
    lines.extend(markdown_table(["endpoint_shape", "total"] + statuses, shape_rows))

    lines.extend(["", "## Relation Token Count Distribution", "", "| bucket | count |", "|---|---:|"])
    for key, value in (report.get("relation_token_count_distribution", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Representative Cases", ""])
    for status, cases in (report.get("representative_cases", {}) or {}).items():
        lines.append(f"### {status}")
        for case in cases[:5]:
            lines.append(
                f"- q={case.get('query_index')} relation={case.get('relation_key')} "
                f"endpoint_shape={case.get('endpoint_shape')} triple={case.get('raw_triple')} "
                f"support_candidates={len(case.get('source_support_candidates', []) or [])}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit schema-light source support for V13B demands.")
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--selector-json", required=True)
    parser.add_argument("--query-obligation-cache", required=True)
    parser.add_argument("--openie-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-examples-per-status", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = audit_source_support(
        dataset=str(args.dataset),
        source_report_path=Path(args.source_report).resolve(),
        selector_json_path=Path(args.selector_json).resolve(),
        query_obligation_cache_path=Path(args.query_obligation_cache).resolve(),
        openie_json_path=Path(args.openie_json).resolve(),
        max_queries=max(int(args.max_queries), 0),
        max_examples_per_status=max(int(args.max_examples_per_status), 1),
    )
    write_json(Path(args.output_json), report)
    write_markdown(Path(args.output_md), report)
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "retrieval_critical_obligation_count": report["retrieval_critical_obligation_count"],
                "source_support_status_counts": report["source_support_status_counts"],
                "interpretation": report["interpretation"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
