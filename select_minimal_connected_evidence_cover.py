#!/usr/bin/env python3
"""Diagnostic minimal connected evidence cover over source-grounded units.

This script is diagnostic-only.  It tests whether source-grounded demand
matches can form a small, binding-consistent, connected evidence set.  It does
not tune PPR, add fallback, call an LLM, change OpenIE prompts, or mutate the
V13B selector.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from analyze_v13b_grounding_failures import (
    build_units_for_docs,
    cache_triples_by_query,
    load_json,
    openie_docs_from_payload,
    retrieval_critical_obligations_for_analysis,
    selector_rows,
    source_rows_by_query,
    write_json,
)
from build_minimal_evidence_units import build_minimal_evidence_units_for_docs
from build_query_obligation_units import (
    build_query_obligation_units,
    endpoint_signature,
    match_query_obligations_to_sto_facts,
)
from evaluate_obligation_closed_sto_local_ppr import query_triples_from_row
from evaluate_obligation_closed_sto_selector import candidate_doc_indices_from_row, unique_ints
from evaluate_transition_component_retriever import all_gold_at_k, recall_at_k
from match_query_demands_to_evidence_units import (
    source_grounded_matches,
)
from query_obligation_typing import lower_query_triples_for_typed_program
from v13b_graph_normalization import safe_int


GENERIC_KEYS = {
    "american",
    "answer",
    "book",
    "city",
    "country",
    "date",
    "entity",
    "film",
    "person",
    "place",
    "state",
    "thing",
    "year",
}


def unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def usable_key(value: Any) -> str:
    key = endpoint_signature(value)
    if not key or key in GENERIC_KEYS:
        return ""
    tokens = key.split()
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return ""
    if len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) < 3:
        return ""
    return key


def binding_values(match: Mapping[str, Any]) -> Dict[str, Set[str]]:
    values: Dict[str, Set[str]] = {}
    for variable, raw_values in (match.get("variable_bindings", {}) or {}).items():
        clean = {usable_key(value) for value in raw_values or []}
        clean.discard("")
        if clean:
            values[str(variable)] = clean
    return values


def mention_keys(match: Mapping[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    for field in ("title", "subject", "object"):
        key = usable_key(match.get(field))
        if key:
            keys.add(key)
    for value in match.get("mention_surfaces", []) or []:
        key = usable_key(value)
        if key:
            keys.add(key)
    for value in match.get("subject_anchor_hits", []) or []:
        key = usable_key(value)
        if key:
            keys.add(key)
    for value in match.get("object_anchor_hits", []) or []:
        key = usable_key(value)
        if key:
            keys.add(key)
    for values in binding_values(match).values():
        keys.update(values)
    return keys


def exact_match_to_cover_candidate(
    *,
    obligation_id: str,
    match: Mapping[str, Any],
    rank_by_doc: Mapping[int, int],
) -> Dict[str, Any]:
    doc_index = safe_int(match.get("doc_index"))
    return {
        "obligation_id": obligation_id,
        "match_id": f"exact:{obligation_id}:{match.get('unit_id')}",
        "unit_id": str(match.get("unit_id") or ""),
        "doc_index": doc_index,
        "title": str(match.get("title") or ""),
        "source": "openie_exact",
        "span_text": "",
        "subject": match.get("subject"),
        "predicate_text": str(match.get("relation") or ""),
        "object": match.get("object"),
        "mention_surfaces": unique_strings([match.get("subject"), match.get("object")]),
        "variable_bindings": {key: sorted(value) for key, value in binding_values(match).items()},
        "source_rank": int(rank_by_doc.get(doc_index, 10**9)),
    }


def source_match_to_cover_candidate(
    *,
    obligation_id: str,
    match: Mapping[str, Any],
    rank_by_doc: Mapping[int, int],
) -> Dict[str, Any]:
    doc_index = safe_int(match.get("doc_index"))
    return {
        "obligation_id": obligation_id,
        "match_id": f"source:{obligation_id}:{match.get('unit_id')}",
        "unit_id": str(match.get("unit_id") or ""),
        "doc_index": doc_index,
        "title": str(match.get("title") or ""),
        "source": str(match.get("source") or "source_grounded"),
        "span_text": str(match.get("span_text") or ""),
        "subject": None,
        "predicate_text": str(match.get("predicate_text") or ""),
        "object": None,
        "mention_surfaces": list(match.get("mention_surfaces", []) or []),
        "subject_anchor_hits": list(match.get("subject_anchor_hits", []) or []),
        "object_anchor_hits": list(match.get("object_anchor_hits", []) or []),
        "relation_token_hits": list(match.get("relation_token_hits", []) or []),
        "variable_bindings": {key: sorted(value) for key, value in binding_values(match).items()},
        "source_rank": int(rank_by_doc.get(doc_index, 10**9)),
    }


def build_cover_candidates_by_obligation(
    *,
    obligations: Sequence[Mapping[str, Any]],
    exact_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    source_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    rank_by_doc: Mapping[int, int],
    max_exact_matches_per_obligation: int,
    max_source_matches_per_obligation: int,
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for obligation in obligations:
        obligation_id = str(obligation.get("obligation_id") or "")
        if not obligation_id:
            continue
        exact_rows = [
            exact_match_to_cover_candidate(obligation_id=obligation_id, match=match, rank_by_doc=rank_by_doc)
            for match in exact_matches.get(obligation_id, []) or []
        ]
        source_rows = [
            source_match_to_cover_candidate(obligation_id=obligation_id, match=match, rank_by_doc=rank_by_doc)
            for match in source_matches.get(obligation_id, []) or []
        ]
        exact_rows.sort(key=lambda row: (int(row["source_rank"]), int(row["doc_index"]), str(row["unit_id"])))
        source_rows.sort(
            key=lambda row: (
                row["source"] != "sentence",
                int(row["source_rank"]),
                int(row["doc_index"]),
                str(row["unit_id"]),
            )
        )
        rows = exact_rows[: max(int(max_exact_matches_per_obligation), 0)] + source_rows[
            : max(int(max_source_matches_per_obligation), 0)
        ]
        if rows:
            result[obligation_id] = rows
    return result


def binding_consistent(matches: Sequence[Mapping[str, Any]]) -> Tuple[bool, Dict[str, List[str]]]:
    variable_domains: Dict[str, Set[str]] = {}
    for match in matches:
        for variable, values in binding_values(match).items():
            if variable in variable_domains:
                overlap = variable_domains[variable] & values
                if not overlap:
                    return False, {}
                variable_domains[variable] = overlap
            else:
                variable_domains[variable] = set(values)
    return True, {variable: sorted(values) for variable, values in variable_domains.items()}


def matches_connected(matches: Sequence[Mapping[str, Any]]) -> bool:
    if len(matches) <= 1:
        return bool(matches)
    key_sets = [mention_keys(match) for match in matches]
    doc_indices = [safe_int(match.get("doc_index")) for match in matches]
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for left, right in itertools.combinations(range(len(matches)), 2):
        same_doc = doc_indices[left] >= 0 and doc_indices[left] == doc_indices[right]
        shared_key = bool(key_sets[left] & key_sets[right])
        if same_doc or shared_key:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for nxt in adjacency.get(current, set()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(matches)


def doc_indices_for_matches(matches: Sequence[Mapping[str, Any]]) -> List[int]:
    return unique_ints(match.get("doc_index") for match in matches)


def cover_sort_key(
    *,
    matches: Sequence[Mapping[str, Any]],
    total_obligations: int,
    candidate_doc_indices: Sequence[int],
    binding_ok: bool,
    connected: bool,
) -> Tuple[Any, ...]:
    docs = doc_indices_for_matches(matches)
    rank_by_doc = {doc: rank for rank, doc in enumerate(candidate_doc_indices)}
    covered = len({str(match.get("obligation_id") or "") for match in matches})
    return (
        -covered,
        covered != total_obligations,
        not binding_ok,
        not connected,
        len(docs),
        sum(rank_by_doc.get(doc, 10**6) for doc in docs),
        docs,
    )


def select_minimal_connected_cover(
    *,
    obligations: Sequence[Mapping[str, Any]],
    candidates_by_obligation: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_doc_indices: Sequence[int],
    evidence_set_size: int = 5,
    max_combinations: int = 25000,
) -> Dict[str, Any]:
    obligation_ids = [str(obligation.get("obligation_id") or "") for obligation in obligations if str(obligation.get("obligation_id") or "")]
    if not obligation_ids:
        return empty_cover_result(
            candidate_doc_indices=candidate_doc_indices,
            evidence_set_size=evidence_set_size,
            total_obligation_count=0,
        )

    candidate_lists: List[List[Dict[str, Any]]] = []
    for obligation_id in obligation_ids:
        rows = [dict(row) for row in candidates_by_obligation.get(obligation_id, []) or []]
        if rows:
            candidate_lists.append(rows)
    if not candidate_lists:
        return empty_cover_result(
            candidate_doc_indices=candidate_doc_indices,
            evidence_set_size=evidence_set_size,
            total_obligation_count=len(obligation_ids),
        )

    best_matches: List[Dict[str, Any]] = []
    best_binding: Dict[str, List[str]] = {}
    best_binding_ok = False
    best_connected = False
    best_key: Tuple[Any, ...] | None = None
    checked = 0

    # First try full-cover combinations.  If any required obligation is missing
    # candidates, this naturally falls through to partial greedy selection.
    if len(candidate_lists) == len(obligation_ids):
        for combination in itertools.product(*candidate_lists):
            checked += 1
            if checked > int(max_combinations):
                break
            docs = doc_indices_for_matches(combination)
            if len(docs) > int(evidence_set_size):
                continue
            binding_ok, bindings = binding_consistent(combination)
            if not binding_ok:
                continue
            connected = matches_connected(combination)
            key = cover_sort_key(
                matches=combination,
                total_obligations=len(obligation_ids),
                candidate_doc_indices=candidate_doc_indices,
                binding_ok=binding_ok,
                connected=connected,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_matches = [dict(match) for match in combination]
                best_binding = bindings
                best_binding_ok = binding_ok
                best_connected = connected

    if not best_matches:
        best_matches, best_binding, best_binding_ok, best_connected = greedy_partial_cover(
            obligation_ids=obligation_ids,
            candidates_by_obligation=candidates_by_obligation,
            candidate_doc_indices=candidate_doc_indices,
            evidence_set_size=evidence_set_size,
        )

    selected_docs = doc_indices_for_matches(best_matches)
    filled_docs = selected_docs + [doc for doc in unique_ints(candidate_doc_indices) if doc not in set(selected_docs)]
    covered_ids = sorted({str(match.get("obligation_id") or "") for match in best_matches if str(match.get("obligation_id") or "")})
    return {
        "selected_doc_indices": filled_docs[: int(evidence_set_size)],
        "cover_doc_indices": selected_docs,
        "covered_obligation_ids": covered_ids,
        "covered_obligation_count": len(covered_ids),
        "total_obligation_count": len(obligation_ids),
        "full_cover_feasible": len(covered_ids) == len(obligation_ids) and bool(best_binding_ok),
        "binding_consistent": bool(best_binding_ok),
        "connected": bool(best_connected),
        "variable_bindings": best_binding,
        "selected_matches": best_matches,
        "combination_count_checked": checked,
        "selection_policy": "coverage_then_binding_consistency_then_connectivity_then_source_rank_then_doc_order",
    }


def greedy_partial_cover(
    *,
    obligation_ids: Sequence[str],
    candidates_by_obligation: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_doc_indices: Sequence[int],
    evidence_set_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], bool, bool]:
    selected: List[Dict[str, Any]] = []
    covered: Set[str] = set()
    for obligation_id in obligation_ids:
        best: Dict[str, Any] | None = None
        best_key: Tuple[Any, ...] | None = None
        for candidate in candidates_by_obligation.get(obligation_id, []) or []:
            proposal = selected + [dict(candidate)]
            if len(doc_indices_for_matches(proposal)) > int(evidence_set_size):
                continue
            binding_ok, _bindings = binding_consistent(proposal)
            if not binding_ok:
                continue
            connected = matches_connected(proposal)
            key = cover_sort_key(
                matches=proposal,
                total_obligations=len(obligation_ids),
                candidate_doc_indices=candidate_doc_indices,
                binding_ok=binding_ok,
                connected=connected,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = dict(candidate)
        if best is not None:
            selected.append(best)
            covered.add(obligation_id)
    binding_ok, bindings = binding_consistent(selected)
    return selected, bindings, binding_ok, matches_connected(selected)


def empty_cover_result(
    *,
    candidate_doc_indices: Sequence[int],
    evidence_set_size: int,
    total_obligation_count: int,
) -> Dict[str, Any]:
    docs = unique_ints(candidate_doc_indices)[: int(evidence_set_size)]
    return {
        "selected_doc_indices": docs,
        "cover_doc_indices": [],
        "covered_obligation_ids": [],
        "covered_obligation_count": 0,
        "total_obligation_count": int(total_obligation_count),
        "full_cover_feasible": False,
        "binding_consistent": False,
        "connected": False,
        "variable_bindings": {},
        "selected_matches": [],
        "combination_count_checked": 0,
        "selection_policy": "empty_or_no_candidate_cover",
    }


def build_obligations_for_row(
    *,
    row: Mapping[str, Any],
    selector: Mapping[str, Any],
    cache_by_query: Mapping[int, Sequence[Sequence[Any]]],
    obligation_scope: str = "retrieval_critical",
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
    if obligation_scope == "materialized":
        materialized = selector.get("materialized_query_obligations") or []
        return list(materialized) if materialized else list(obligations)
    return retrieval_critical_obligations_for_analysis(obligations=obligations, selector=selector)


def evaluate_minimal_connected_evidence_cover(
    *,
    dataset: str,
    source_report_path: Path,
    selector_json_path: Path,
    query_obligation_cache_path: Path,
    openie_json_path: Path,
    max_queries: int = 0,
    evidence_set_size: int = 5,
    max_exact_matches_per_obligation: int = 3,
    max_source_matches_per_obligation: int = 3,
    obligation_scope: str = "retrieval_critical",
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
    needed_doc_indices: Set[int] = set()
    for row in rows:
        query_index = safe_int(row.get("query_index"))
        source_row = source_by_query.get(query_index, row)
        needed_doc_indices.update(candidate_doc_indices_from_row(source_row))
    exact_units_by_doc = {
        doc_index: build_units_for_docs(doc_indices=[doc_index], openie_docs=openie_docs)
        for doc_index in sorted(needed_doc_indices)
        if 0 <= int(doc_index) < len(openie_docs)
    }
    evidence_units_by_doc = {
        doc_index: build_minimal_evidence_units_for_docs(doc_indices=[doc_index], openie_docs=openie_docs)
        for doc_index in sorted(needed_doc_indices)
        if 0 <= int(doc_index) < len(openie_docs)
    }

    summary: Counter[str] = Counter()
    recall_sum = 0.0
    source_recall_sum = 0.0
    selector_recall_sum = 0.0
    all_gold_count = 0
    source_all_gold_count = 0
    selector_all_gold_count = 0
    gains = 0
    losses = 0
    gains_vs_selector = 0
    losses_vs_selector = 0
    admitted_recall_sum = 0.0
    admitted_all_gold_count = 0
    admitted_count = 0
    admitted_gains_vs_selector = 0
    admitted_losses_vs_selector = 0
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    output_rows: List[Dict[str, Any]] = []

    for row in rows:
        query_index = safe_int(row.get("query_index"))
        source_row = source_by_query.get(query_index, row)
        selector = row.get("obligation_closed_sto_local_ppr_selector", {}) or {}
        candidate_docs = candidate_doc_indices_from_row(source_row)
        rank_by_doc = {doc: rank for rank, doc in enumerate(candidate_docs)}
        gold_docs = unique_ints(source_row.get("gold_doc_indices", []) or [])
        obligations = build_obligations_for_row(
            row=source_row,
            selector=selector,
            cache_by_query=cache_by_query,
            obligation_scope=obligation_scope,
        )
        exact_units = [
            unit
            for doc_index in candidate_docs
            for unit in exact_units_by_doc.get(int(doc_index), [])
        ]
        exact_matches = match_query_obligations_to_sto_facts(obligations=obligations, candidate_units=exact_units)
        evidence_units = [
            unit
            for doc_index in candidate_docs
            for unit in evidence_units_by_doc.get(int(doc_index), [])
        ]
        source_matches = source_grounded_matches(
            obligations=obligations,
            evidence_units=evidence_units,
            exact_matches=exact_matches,
        )
        cover_candidates = build_cover_candidates_by_obligation(
            obligations=obligations,
            exact_matches=exact_matches,
            source_matches=source_matches,
            rank_by_doc=rank_by_doc,
            max_exact_matches_per_obligation=max_exact_matches_per_obligation,
            max_source_matches_per_obligation=max_source_matches_per_obligation,
        )
        cover = select_minimal_connected_cover(
            obligations=obligations,
            candidates_by_obligation=cover_candidates,
            candidate_doc_indices=candidate_docs,
            evidence_set_size=evidence_set_size,
        )
        source_top5 = unique_ints(source_row.get("retrieved_doc_indices_top5", []) or candidate_docs)[:evidence_set_size]
        selector_top5 = unique_ints(row.get("obligation_closed_sto_local_ppr_doc_indices_top5", []) or [])[:evidence_set_size]
        selected_top5 = unique_ints(cover.get("selected_doc_indices", []) or [])[:evidence_set_size]

        recall = recall_at_k(gold_docs, selected_top5, evidence_set_size)
        source_recall = recall_at_k(gold_docs, source_top5, evidence_set_size)
        selector_recall = recall_at_k(gold_docs, selector_top5, evidence_set_size)
        recall_sum += recall
        source_recall_sum += source_recall
        selector_recall_sum += selector_recall
        all_gold = all_gold_at_k(gold_docs, selected_top5, evidence_set_size)
        source_all_gold = all_gold_at_k(gold_docs, source_top5, evidence_set_size)
        selector_all_gold = all_gold_at_k(gold_docs, selector_top5, evidence_set_size)
        all_gold_count += int(all_gold)
        source_all_gold_count += int(source_all_gold)
        selector_all_gold_count += int(selector_all_gold)
        selected_gold_count = len(set(gold_docs) & set(selected_top5))
        source_gold_count = len(set(gold_docs) & set(source_top5))
        selector_gold_count = len(set(gold_docs) & set(selector_top5))
        if selected_gold_count > source_gold_count:
            gains += 1
        elif selected_gold_count < source_gold_count:
            losses += 1
        if selected_gold_count > selector_gold_count:
            gains_vs_selector += 1
        elif selected_gold_count < selector_gold_count:
            losses_vs_selector += 1

        admitted = bool(cover.get("full_cover_feasible", False)) and bool(cover.get("connected", False))
        admitted_top5 = selected_top5 if admitted else selector_top5
        admitted_count += int(admitted)
        admitted_recall_sum += recall_at_k(gold_docs, admitted_top5, evidence_set_size)
        admitted_all_gold_count += int(all_gold_at_k(gold_docs, admitted_top5, evidence_set_size))
        admitted_gold_count = len(set(gold_docs) & set(admitted_top5))
        if admitted_gold_count > selector_gold_count:
            admitted_gains_vs_selector += 1
        elif admitted_gold_count < selector_gold_count:
            admitted_losses_vs_selector += 1

        summary["query_count"] += 1
        summary["retrieval_critical_obligation_count"] += int(cover.get("total_obligation_count", 0))
        summary["covered_obligation_count"] += int(cover.get("covered_obligation_count", 0))
        summary["full_cover_feasible_count"] += int(bool(cover.get("full_cover_feasible", False)))
        summary["connected_cover_count"] += int(bool(cover.get("connected", False)))
        summary["binding_consistent_cover_count"] += int(bool(cover.get("binding_consistent", False)))
        if any(str(match.get("source") or "") in {"sentence", "openie"} for match in cover.get("selected_matches", []) or []):
            summary["uses_source_grounded_match_count"] += 1

        status = "full_cover" if cover.get("full_cover_feasible") else "partial_cover"
        if len(examples[status]) < 8:
            examples[status].append(
                {
                    "query_index": query_index,
                    "question": str(source_row.get("question") or source_row.get("query") or ""),
                    "gold_doc_indices": gold_docs,
                    "source_top5": source_top5,
                    "selector_top5": selector_top5,
                    "cover_top5": selected_top5,
                    "cover": {
                        "covered_obligation_count": cover.get("covered_obligation_count"),
                        "total_obligation_count": cover.get("total_obligation_count"),
                        "full_cover_feasible": cover.get("full_cover_feasible"),
                        "connected": cover.get("connected"),
                        "binding_consistent": cover.get("binding_consistent"),
                        "selected_matches": cover.get("selected_matches", [])[:5],
                    },
                }
            )
        output_rows.append(
            {
                "query_index": query_index,
                "gold_doc_indices": gold_docs,
                "source_top5": source_top5,
                "selector_top5": selector_top5,
                "minimal_cover_top5": selected_top5,
                "minimal_cover_recall_at5": recall,
                "source_recall_at5": source_recall,
                "selector_recall_at5": selector_recall,
                "minimal_cover_all_gold_at5": all_gold,
                "certified_admitted": admitted,
                "full_cover_feasible": bool(cover.get("full_cover_feasible", False)),
                "connected": bool(cover.get("connected", False)),
                "binding_consistent": bool(cover.get("binding_consistent", False)),
                "covered_obligation_count": int(cover.get("covered_obligation_count", 0)),
                "total_obligation_count": int(cover.get("total_obligation_count", 0)),
            }
        )

    query_count = max(int(summary["query_count"]), 1)
    total_obligations = max(int(summary["retrieval_critical_obligation_count"]), 1)
    return {
        "config": {
            "dataset": dataset,
            "max_queries": int(max_queries),
            "evidence_set_size": int(evidence_set_size),
            "obligation_scope": obligation_scope,
            "source_report_path": str(source_report_path),
            "selector_json_path": str(selector_json_path),
            "query_obligation_cache_path": str(query_obligation_cache_path),
            "openie_json_path": str(openie_json_path),
            "diagnostic_only": True,
            "uses_fixed_relation_schema": False,
            "uses_relation_synonym_table": False,
            "uses_weighted_rank_fusion": False,
        },
        "summary": {
            **dict(summary),
            "covered_obligation_rate": int(summary["covered_obligation_count"]) / total_obligations,
            "full_cover_feasible_rate": int(summary["full_cover_feasible_count"]) / query_count,
            "connected_cover_rate": int(summary["connected_cover_count"]) / query_count,
            "minimal_cover_recall_at5": recall_sum / query_count,
            "source_recall_at5": source_recall_sum / query_count,
            "selector_recall_at5": selector_recall_sum / query_count,
            "minimal_cover_all_gold_at5": all_gold_count / query_count,
            "source_all_gold_at5": source_all_gold_count / query_count,
            "selector_all_gold_at5": selector_all_gold_count / query_count,
            "gold_count_gains_vs_source_top5": gains,
            "gold_count_losses_vs_source_top5": losses,
            "gold_count_gains_vs_selector_top5": gains_vs_selector,
            "gold_count_losses_vs_selector_top5": losses_vs_selector,
            "certified_admission_count": admitted_count,
            "certified_admission_recall_at5": admitted_recall_sum / query_count,
            "certified_admission_all_gold_at5": admitted_all_gold_count / query_count,
            "certified_admission_gains_vs_selector_top5": admitted_gains_vs_selector,
            "certified_admission_losses_vs_selector_top5": admitted_losses_vs_selector,
            "certified_admission_policy": "admit_cover_only_when_full_cover_feasible_and_connected_else_keep_selector",
        },
        "examples": dict(examples),
        "rows": output_rows,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    config = report.get("config", {}) or {}
    summary = report.get("summary", {}) or {}
    lines = [
        f"# Minimal Connected Evidence Cover: {config.get('dataset')}",
        "",
        "This is a diagnostic-only source-grounded evidence cover. It does not modify the V13B selector.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "query_count",
        "retrieval_critical_obligation_count",
        "covered_obligation_count",
        "covered_obligation_rate",
        "full_cover_feasible_count",
        "full_cover_feasible_rate",
        "connected_cover_count",
        "connected_cover_rate",
        "uses_source_grounded_match_count",
        "minimal_cover_recall_at5",
        "source_recall_at5",
        "selector_recall_at5",
        "minimal_cover_all_gold_at5",
        "source_all_gold_at5",
        "selector_all_gold_at5",
        "gold_count_gains_vs_source_top5",
        "gold_count_losses_vs_source_top5",
        "gold_count_gains_vs_selector_top5",
        "gold_count_losses_vs_selector_top5",
        "certified_admission_count",
        "certified_admission_recall_at5",
        "certified_admission_all_gold_at5",
        "certified_admission_gains_vs_selector_top5",
        "certified_admission_losses_vs_selector_top5",
    ):
        value = summary.get(key)
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "A useful signal requires both higher demand coverage and non-negative retrieval behavior versus the source top5."
    )
    lines.append(
        "If recall drops, the cover objective is not selector-ready even if source-grounded demand coverage is high."
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate diagnostic minimal connected evidence cover.")
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--selector-json", required=True)
    parser.add_argument("--query-obligation-cache", required=True)
    parser.add_argument("--openie-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--evidence-set-size", type=int, default=5)
    parser.add_argument("--max-exact-matches-per-obligation", type=int, default=3)
    parser.add_argument("--max-source-matches-per-obligation", type=int, default=3)
    parser.add_argument("--obligation-scope", choices=["retrieval_critical", "materialized"], default="retrieval_critical")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = evaluate_minimal_connected_evidence_cover(
        dataset=str(args.dataset),
        source_report_path=Path(args.source_report).resolve(),
        selector_json_path=Path(args.selector_json).resolve(),
        query_obligation_cache_path=Path(args.query_obligation_cache).resolve(),
        openie_json_path=Path(args.openie_json).resolve(),
        max_queries=max(int(args.max_queries), 0),
        evidence_set_size=max(int(args.evidence_set_size), 1),
        max_exact_matches_per_obligation=max(int(args.max_exact_matches_per_obligation), 0),
        max_source_matches_per_obligation=max(int(args.max_source_matches_per_obligation), 0),
        obligation_scope=str(args.obligation_scope),
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
