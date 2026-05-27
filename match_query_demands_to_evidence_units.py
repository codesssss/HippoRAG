#!/usr/bin/env python3
"""Diagnostic source-grounded matching for V13B query demands.

This probe checks whether retrieval-critical query demands can be matched to
schema-light evidence units when exact OpenIE relation grounding fails.  It is
diagnostic-only: it does not change retrieval, selection, fallback, PPR, or
OpenIE prompts.
"""

from __future__ import annotations

import argparse
import json
import re
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
from build_minimal_evidence_units import build_minimal_evidence_units_for_docs
from build_query_obligation_units import (
    build_query_obligation_units,
    endpoint_alias_signatures,
    endpoint_signature,
    endpoint_signature_tokens,
    match_query_obligations_to_sto_facts,
    soften_relation_token,
)
from evaluate_obligation_closed_sto_local_ppr import query_triples_from_row
from evaluate_obligation_closed_sto_selector import candidate_doc_indices_from_row, unique_ints
from query_obligation_support_grounding import variable_values_from_matches
from query_obligation_typing import lower_query_triples_for_typed_program
from v13b_graph_normalization import safe_int


TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}

GENERIC_MENTION_TOKENS = {
    "album",
    "answer",
    "award",
    "book",
    "city",
    "country",
    "date",
    "entity",
    "film",
    "group",
    "person",
    "place",
    "river",
    "song",
    "state",
    "team",
    "thing",
    "town",
    "year",
}


def normalized_tokens(value: Any) -> Set[str]:
    return {
        soften_relation_token(token)
        for token in re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
        if len(token) >= 3 and token not in TOKEN_STOPWORDS
    }


def predicate_tokens_for_obligation(obligation: Mapping[str, Any]) -> Set[str]:
    # Use the raw query predicate.  Do not use relation_signature aliases here;
    # this probe is specifically testing whether source text can replace brittle
    # OpenIE frame labels without a hand-written schema.
    return normalized_tokens(obligation.get("raw_relation") or obligation.get("relation"))


def evidence_unit_tokens(unit: Mapping[str, Any]) -> Set[str]:
    values: List[Any] = [
        unit.get("title"),
        unit.get("span_text"),
        unit.get("predicate_text"),
        " ".join(str(item) for item in (unit.get("argument_texts", []) or [])),
    ]
    values.extend(unit.get("mention_surfaces", []) or [])
    tokens: Set[str] = set()
    for value in values:
        tokens.update(normalized_tokens(value))
    return tokens


def endpoint_tokens(value: Any) -> Set[str]:
    return set(endpoint_signature_tokens(value))


def anchor_aliases(value: Any) -> Set[str]:
    aliases = endpoint_alias_signatures(value)
    if not aliases and endpoint_signature(value):
        aliases = {endpoint_signature(value)}
    return aliases


def aliases_visible_in_text(aliases: Iterable[str], text: str) -> List[str]:
    text_tokens = endpoint_signature_tokens(text)
    hits: List[str] = []
    for alias in sorted({str(alias) for alias in aliases if str(alias)}):
        alias_tokens = endpoint_signature_tokens(alias)
        if not alias_tokens or len(alias_tokens) > len(text_tokens):
            continue
        width = len(alias_tokens)
        if any(text_tokens[start : start + width] == alias_tokens for start in range(0, len(text_tokens) - width + 1)):
            hits.append(alias)
    return hits


def anchored_role_values(
    *,
    obligation: Mapping[str, Any],
    variable_values: Mapping[str, Set[str]],
    role: str,
) -> Set[str]:
    if role == "subject":
        is_variable = bool(obligation.get("subject_is_variable", False))
        variable = str(obligation.get("subject_variable") or "")
        value = obligation.get("raw_subject") or obligation.get("subject")
    elif role == "object":
        is_variable = bool(obligation.get("object_is_variable", False))
        variable = str(obligation.get("object_variable") or "")
        value = obligation.get("raw_object") or obligation.get("object")
    else:
        return set()
    if is_variable:
        return set(variable_values.get(variable, set()) or set())
    return {str(value)} if str(value or "").strip() else set()


def visible_anchor_hits(unit: Mapping[str, Any], anchors: Iterable[str]) -> List[str]:
    text = "\n".join(
        [
            str(unit.get("title") or ""),
            str(unit.get("span_text") or ""),
            " ".join(str(item) for item in (unit.get("mention_surfaces", []) or [])),
        ]
    )
    hits: List[str] = []
    for anchor in anchors:
        hits.extend(aliases_visible_in_text(anchor_aliases(anchor), text))
    return sorted(set(hits))


def useful_mention_surface(surface: Any, *, excluded_aliases: Set[str]) -> bool:
    signature = endpoint_signature(surface)
    if not signature:
        return False
    if signature in excluded_aliases:
        return False
    tokens = set(endpoint_signature_tokens(signature))
    if not tokens or tokens <= GENERIC_MENTION_TOKENS:
        return False
    if len(tokens) == 1 and next(iter(tokens)).isdigit() and len(next(iter(tokens))) < 3:
        return False
    return True


def mention_compatible_with_variable(surface: str, variable: str) -> bool:
    variable_key = endpoint_signature(variable)
    tokens = set(endpoint_signature_tokens(surface))
    if variable_key in {"date", "day", "month", "time", "year"}:
        return any(token.isdigit() for token in tokens)
    if tokens and all(token.isdigit() for token in tokens):
        return False
    return True


def binding_candidates_from_unit(
    *,
    obligation: Mapping[str, Any],
    unit: Mapping[str, Any],
    subject_anchor_hits: Sequence[str],
    object_anchor_hits: Sequence[str],
) -> Dict[str, List[str]]:
    excluded_aliases: Set[str] = set(subject_anchor_hits) | set(object_anchor_hits)
    for value in subject_anchor_hits + object_anchor_hits:
        excluded_aliases.update(anchor_aliases(value))

    candidate_mentions = [
        endpoint_signature(surface)
        for surface in unit.get("mention_surfaces", []) or []
        if useful_mention_surface(surface, excluded_aliases=excluded_aliases)
    ]
    candidate_mentions = sorted({value for value in candidate_mentions if value})
    bindings: Dict[str, List[str]] = {}
    if bool(obligation.get("subject_is_variable", False)) and not subject_anchor_hits:
        variable = str(obligation.get("subject_variable") or "")
        if variable and candidate_mentions:
            bindings[variable] = [
                mention for mention in candidate_mentions if mention_compatible_with_variable(mention, variable)
            ][:4]
    if bool(obligation.get("object_is_variable", False)) and not object_anchor_hits:
        variable = str(obligation.get("object_variable") or "")
        if variable and candidate_mentions:
            bindings[variable] = [
                mention for mention in candidate_mentions if mention_compatible_with_variable(mention, variable)
            ][:4]
    bindings = {variable: values for variable, values in bindings.items() if values}
    return bindings


def source_grounded_matches_for_obligation(
    *,
    obligation: Mapping[str, Any],
    evidence_units: Sequence[Mapping[str, Any]],
    variable_values: Mapping[str, Set[str]],
    max_matches: int = 3,
) -> List[Dict[str, Any]]:
    relation_tokens = predicate_tokens_for_obligation(obligation)
    if not relation_tokens:
        return []
    subject_anchors = anchored_role_values(obligation=obligation, variable_values=variable_values, role="subject")
    object_anchors = anchored_role_values(obligation=obligation, variable_values=variable_values, role="object")
    if not subject_anchors and not object_anchors:
        return []

    rows: List[Dict[str, Any]] = []
    for unit in evidence_units:
        unit_tokens = evidence_unit_tokens(unit)
        relation_hits = sorted(relation_tokens & unit_tokens)
        if not relation_hits:
            continue
        subject_hits = visible_anchor_hits(unit, subject_anchors)
        object_hits = visible_anchor_hits(unit, object_anchors)
        if subject_anchors and not subject_hits:
            continue
        if object_anchors and not object_hits:
            continue

        variable_bindings = binding_candidates_from_unit(
            obligation=obligation,
            unit=unit,
            subject_anchor_hits=subject_hits,
            object_anchor_hits=object_hits,
        )
        unanchored_variables = []
        if bool(obligation.get("subject_is_variable", False)) and not subject_hits:
            unanchored_variables.append(str(obligation.get("subject_variable") or ""))
        if bool(obligation.get("object_is_variable", False)) and not object_hits:
            unanchored_variables.append(str(obligation.get("object_variable") or ""))
        if any(variable and variable not in variable_bindings for variable in unanchored_variables):
            continue

        rows.append(
            {
                "obligation_id": str(obligation.get("obligation_id") or ""),
                "unit_id": str(unit.get("unit_id") or ""),
                "doc_index": int(unit.get("doc_index", -1) or -1),
                "title": str(unit.get("title") or ""),
                "source": str(unit.get("source") or ""),
                "span_text": str(unit.get("span_text") or "")[:500],
                "predicate_text": str(unit.get("predicate_text") or ""),
                "mention_surfaces": list(unit.get("mention_surfaces", []) or [])[:12],
                "relation_token_hits": relation_hits,
                "subject_anchor_hits": subject_hits,
                "object_anchor_hits": object_hits,
                "variable_bindings": variable_bindings,
                "match_reasons": [
                    "source_grounded_span",
                    "literal_predicate_token_hit",
                    *("subject_anchor_visible" for _ in [0] if subject_hits),
                    *("object_anchor_visible" for _ in [0] if object_hits),
                    *("explicit_variable_binding_candidate" for _ in [0] if variable_bindings),
                ],
            }
        )

    rows.sort(
        key=lambda row: (
            row["source"] != "sentence",
            -len(row["relation_token_hits"]),
            not bool(row["subject_anchor_hits"]),
            not bool(row["object_anchor_hits"]),
            int(row["doc_index"]),
            str(row["title"]),
        )
    )
    return rows[: max(int(max_matches), 1)]


def source_grounded_matches(
    *,
    obligations: Sequence[Mapping[str, Any]],
    evidence_units: Sequence[Mapping[str, Any]],
    exact_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    max_matches_per_obligation: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    variable_values = variable_values_from_matches(obligations=obligations, matches=exact_matches)
    matches: Dict[str, List[Dict[str, Any]]] = {}
    for obligation in obligations:
        obligation_id = str(obligation.get("obligation_id") or "")
        if not obligation_id:
            continue
        rows = source_grounded_matches_for_obligation(
            obligation=obligation,
            evidence_units=evidence_units,
            variable_values=variable_values,
            max_matches=max_matches_per_obligation,
        )
        if rows:
            matches[obligation_id] = rows
    return matches


def build_obligations_for_row(
    *,
    row: Mapping[str, Any],
    selector: Mapping[str, Any],
    cache_by_query: Mapping[int, Sequence[Sequence[Any]]],
) -> List[Mapping[str, Any]]:
    query_triples = query_triples_from_row(row, query_obligation_cache=cache_by_query)
    program_triples = lower_query_triples_for_typed_program(
        query=str(row.get("question") or row.get("query") or ""),
        query_triples=query_triples,
    )
    obligations = selector.get("query_obligations") or build_query_obligation_units(
        query=str(row.get("question") or row.get("query") or ""),
        query_triples=program_triples,
    )
    return retrieval_critical_obligations_for_analysis(obligations=obligations, selector=selector)


def selected_doc_indices(row: Mapping[str, Any], *, doc_scope: str) -> List[int]:
    if doc_scope == "gold":
        return unique_ints(row.get("gold_doc_indices", []) or [])
    if doc_scope == "candidate":
        return candidate_doc_indices_from_row(row)
    raise ValueError(f"Unsupported doc_scope: {doc_scope}")


def match_query_demands_to_evidence_units(
    *,
    dataset: str,
    source_report_path: Path,
    selector_json_path: Path,
    query_obligation_cache_path: Path,
    openie_json_path: Path,
    doc_scope: str = "gold",
    include_selector_pseudo_seeds: bool = False,
    max_queries: int = 0,
    max_examples: int = 10,
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
    source_counts: Counter[str] = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_obligations = 0
    exact_count = 0
    source_count = 0
    new_count = 0

    for row in rows:
        query_index = safe_int(row.get("query_index"))
        source_row = source_by_query.get(query_index, row)
        selector = row.get("obligation_closed_sto_local_ppr_selector", {}) or {}
        obligations = build_obligations_for_row(row=source_row, selector=selector, cache_by_query=cache_by_query)
        doc_indices = selected_doc_indices(source_row, doc_scope=doc_scope)
        exact_units = build_units_for_docs(doc_indices=doc_indices, openie_docs=openie_docs)
        exact_matches = match_query_obligations_to_sto_facts(obligations=obligations, candidate_units=exact_units)
        if include_selector_pseudo_seeds:
            selector_support = selector.get("support_grounded_obligation_matches", {}) or {}
            selector_source_span = selector.get("source_span_grounded_obligation_matches", {}) or {}
            variable_seed_matches = merge_match_maps(exact_matches, selector_support, selector_source_span)
        else:
            variable_seed_matches = exact_matches
        evidence_units = build_minimal_evidence_units_for_docs(doc_indices=doc_indices, openie_docs=openie_docs)
        source_matches = source_grounded_matches(
            obligations=obligations,
            evidence_units=evidence_units,
            exact_matches=variable_seed_matches,
        )

        for obligation in obligations:
            obligation_id = str(obligation.get("obligation_id") or "")
            if not obligation_id:
                continue
            total_obligations += 1
            has_exact = bool(exact_matches.get(obligation_id))
            has_source = bool(source_matches.get(obligation_id))
            exact_count += int(has_exact)
            source_count += int(has_source)
            new_count += int(has_source and not has_exact)
            if has_exact:
                status = "openie_exact_available"
            elif has_source:
                status = "source_grounded_new_match"
            else:
                status = "unmatched"
            status_counts[status] += 1
            for match in source_matches.get(obligation_id, [])[:1]:
                source_counts[str(match.get("source") or "")] += 1
            if len(examples[status]) < int(max_examples):
                examples[status].append(
                    {
                        "query_index": query_index,
                        "question": str(source_row.get("question") or source_row.get("query") or ""),
                        "obligation": {
                            "raw_subject": obligation.get("raw_subject"),
                            "raw_relation": obligation.get("raw_relation"),
                            "raw_object": obligation.get("raw_object"),
                            "subject_is_variable": obligation.get("subject_is_variable"),
                            "object_is_variable": obligation.get("object_is_variable"),
                        },
                        "exact_matches": list(exact_matches.get(obligation_id, []) or [])[:2],
                        "source_grounded_matches": list(source_matches.get(obligation_id, []) or [])[:2],
                    }
                )

    exact_or_source = sum(
        count for key, count in status_counts.items() if key in {"openie_exact_available", "source_grounded_new_match"}
    )
    return {
        "config": {
            "dataset": dataset,
            "doc_scope": doc_scope,
            "include_selector_pseudo_seeds": bool(include_selector_pseudo_seeds),
            "max_queries": int(max_queries),
            "source_report_path": str(source_report_path),
            "selector_json_path": str(selector_json_path),
            "query_obligation_cache_path": str(query_obligation_cache_path),
            "openie_json_path": str(openie_json_path),
            "diagnostic_only": True,
            "uses_fixed_relation_schema": False,
            "uses_relation_synonym_table": False,
        },
        "summary": {
            "query_count": len(rows),
            "retrieval_critical_obligation_count": total_obligations,
            "openie_exact_count": exact_count,
            "source_grounded_match_count": source_count,
            "new_matches_over_openie_exact": new_count,
            "exact_or_source_grounded_count": exact_or_source,
            "openie_exact_rate": exact_count / total_obligations if total_obligations else 0.0,
            "source_grounded_new_rate": new_count / total_obligations if total_obligations else 0.0,
            "exact_or_source_rate": exact_or_source / total_obligations if total_obligations else 0.0,
            "status_counts": dict(status_counts),
            "source_counts_for_source_grounded_matches": dict(source_counts),
        },
        "examples": dict(examples),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    config = report.get("config", {}) or {}
    summary = report.get("summary", {}) or {}
    lines = [
        f"# Source-Grounded Demand Matcher: {config.get('dataset')}",
        "",
        "This is a diagnostic-only probe. It does not change retrieval or selector outputs.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "query_count",
        "retrieval_critical_obligation_count",
        "openie_exact_count",
        "source_grounded_match_count",
        "new_matches_over_openie_exact",
        "exact_or_source_grounded_count",
        "openie_exact_rate",
        "source_grounded_new_rate",
        "exact_or_source_rate",
    ):
        value = summary.get(key)
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Status Counts", "", "| Status | Count |", "|---|---:|"])
    for key, value in sorted((summary.get("status_counts", {}) or {}).items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "If `new_matches_over_openie_exact` is non-trivial, the next clean step is a minimal connected evidence cover over source-grounded units, not OpenIE prompt/schema tuning."
    )
    lines.append(
        "If it is small, this line should be downgraded because source text matching does not recover the OpenIE grounding gap."
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Match V13B query demands to schema-light evidence units.")
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--selector-json", required=True)
    parser.add_argument("--query-obligation-cache", required=True)
    parser.add_argument("--openie-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--doc-scope", choices=["gold", "candidate"], default="gold")
    parser.add_argument("--include-selector-pseudo-seeds", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = match_query_demands_to_evidence_units(
        dataset=str(args.dataset),
        source_report_path=Path(args.source_report).resolve(),
        selector_json_path=Path(args.selector_json).resolve(),
        query_obligation_cache_path=Path(args.query_obligation_cache).resolve(),
        openie_json_path=Path(args.openie_json).resolve(),
        doc_scope=str(args.doc_scope),
        include_selector_pseudo_seeds=bool(args.include_selector_pseudo_seeds),
        max_queries=max(int(args.max_queries), 0),
        max_examples=max(int(args.max_examples), 1),
    )
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    write_json(output_json, report)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=True, sort_keys=True))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
