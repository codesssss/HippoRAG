#!/usr/bin/env python3
"""Evaluate local PPR over obligation-closed STO units.

This is a clean selector prototype. It keeps the construction object from
``build_obligation_closed_sto_units.py`` and replaces lexical coverage with a
single graph operation: compute query-seed local Personalized PageRank over
the STO fact graph and project that distribution onto candidate closed units.

The implementation keeps both source-side and target-side local-push routines.
The selector uses the source-side batch form so each query needs one local PPR
push, not one push per candidate unit. It is intentionally not HippoRAG v2's
global passage/entity diffusion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from build_obligation_closed_sto_units import (
    build_endpoint_adjacency,
    build_endpoint_index,
    endpoint_key,
    fact_units,
    unit_endpoint_entries,
)
from build_query_obligation_units import (
    build_query_obligation_units,
    match_query_obligations_to_sto_facts,
    obligation_seed_unit_ids,
)
from evaluate_obligation_closed_sto_selector import (
    DEFAULT_REPORT,
    QUERY_FUNCTION_TOKENS,
    build_candidate_sto_units,
    build_seed_closed_unit_store,
    candidate_doc_indices_from_row,
    content_tokens,
    doc_signature_tokens,
    frontier_query_tokens,
    load_json,
    row_doc_indices,
    seed_unit_ids_for_query,
    unique_ints,
)
from evaluate_transition_component_retriever import all_gold_at_k, recall_at_k
from query_obligation_support_grounding import (
    filter_matches_by_variable_values,
    merge_exact_and_support_matches,
    source_span_grounded_obligation_matches,
    support_grounded_obligation_matches,
    variable_values_from_matches,
)
from query_obligation_typing import (
    looks_like_temporal_value,
    lower_query_triples_for_typed_program,
    typed_query_program,
)


DEFAULT_PPR_ALPHA = 0.2
DEFAULT_PPR_RESIDUAL_EPSILON = 1e-6


def parse_query_indices(spec: str) -> Set[int]:
    indices: Set[int] = set()
    for part in str(spec or "").split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                start, end = end, start
            indices.update(range(start, end + 1))
            continue
        indices.add(int(token))
    return indices


def unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        item = str(value or "")
        if not item or item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def raw_triple_key(values: Sequence[Any]) -> Tuple[str, str, str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 3:
        return ("", "", "")
    return tuple(str(value or "") for value in values[:3])  # type: ignore[return-value]


def obligation_raw_triple_key(obligation: Mapping[str, Any]) -> Tuple[str, str, str]:
    return raw_triple_key(
        [
            obligation.get("raw_subject", ""),
            obligation.get("raw_relation", ""),
            obligation.get("raw_object", ""),
        ]
    )


def row_query_index(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("query_index", -1))
    except (TypeError, ValueError):
        return -1


def query_triples_from_row(
    row: Mapping[str, Any],
    head_traces: Sequence[Mapping[str, Any]] | None = None,
    query_obligation_cache: Mapping[int, Sequence[Sequence[Any]]] | None = None,
) -> List[Sequence[Any]]:
    """Resolve query triples from row fields, obligation cache, or head traces."""

    for key in ("query_triples", "query_facts", "top_k_facts"):
        triples = row.get(key)
        if isinstance(triples, Sequence) and not isinstance(triples, (str, bytes)):
            return [triple for triple in triples if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes))]

    trace_index = row_query_index(row)
    if trace_index < 0:
        return []
    if query_obligation_cache is not None:
        triples = query_obligation_cache.get(trace_index, [])
        if isinstance(triples, Sequence) and not isinstance(triples, (str, bytes)):
            clean_triples = [
                triple
                for triple in triples
                if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes))
            ]
            if clean_triples:
                return clean_triples
    if head_traces is None:
        return []
    if trace_index < 0 or trace_index >= len(head_traces):
        return []
    trace = head_traces[trace_index]
    triples = trace.get("top_k_facts") if isinstance(trace, Mapping) else None
    if not isinstance(triples, Sequence) or isinstance(triples, (str, bytes)):
        return []
    return [triple for triple in triples if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes))]


def typed_execution_plan(
    *,
    query: str,
    query_triples: Sequence[Sequence[Any]],
    materialized_obligations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Map typed query-program requirements onto materialized obligations."""

    program = typed_query_program(query=query, query_triples=query_triples)
    materialized_by_raw = {
        obligation_raw_triple_key(obligation): obligation
        for obligation in materialized_obligations
    }
    retrieval_critical_ids: List[str] = []
    retrieval_critical_unmaterialized: List[Dict[str, Any]] = []
    materialized_non_retrieval_ids: List[str] = []
    typed_rows: List[Dict[str, Any]] = []
    type_counts: Counter[str] = Counter()

    for typed_obligation in program.get("typed_obligations", []) or []:
        obligation_type = str(typed_obligation.get("obligation_type") or "unknown")
        type_counts[obligation_type] += 1
        retrieval_critical = bool(typed_obligation.get("retrieval_critical", False))
        materialized = materialized_by_raw.get(raw_triple_key(typed_obligation.get("raw_triple", []) or []))
        materialized_id = str((materialized or {}).get("obligation_id") or "")
        typed_rows.append(
            {
                **dict(typed_obligation),
                "materialized_obligation_id": materialized_id,
                "materialized_by_obligation_builder": bool(materialized_id),
            }
        )
        if retrieval_critical:
            if materialized_id:
                retrieval_critical_ids.append(materialized_id)
            else:
                retrieval_critical_unmaterialized.append(
                    {
                        "obligation_type": obligation_type,
                        "reason": str(typed_obligation.get("reason") or ""),
                        "raw_triple": list(typed_obligation.get("raw_triple", []) or []),
                    }
                )
            continue
        if materialized_id:
            materialized_non_retrieval_ids.append(materialized_id)

    return {
        "query_answer_shape": program.get("query_answer_shape", "unknown"),
        "type_counts": dict(sorted(type_counts.items())),
        "typed_obligations": typed_rows,
        "retrieval_critical_obligation_ids": unique_strings(retrieval_critical_ids),
        "retrieval_critical_unmaterialized": retrieval_critical_unmaterialized,
        "retrieval_critical_unmaterialized_count": len(retrieval_critical_unmaterialized),
        "materialized_non_retrieval_obligation_ids": unique_strings(materialized_non_retrieval_ids),
        "materialized_non_retrieval_obligation_count": len(unique_strings(materialized_non_retrieval_ids)),
        "variable_type_constraints": dict(program.get("variable_type_constraints", {}) or {}),
        "variable_type_constraint_reasons": dict(program.get("variable_type_constraint_reasons", {}) or {}),
        "retrieval_critical_count": int(program.get("retrieval_critical_count", 0) or 0),
        "triple_count": int(program.get("triple_count", 0) or 0),
    }


def match_satisfies_variable_type_constraints(
    *,
    match: Mapping[str, Any],
    variable_type_constraints: Mapping[str, str],
) -> bool:
    if not variable_type_constraints:
        return True
    bindings = match.get("variable_bindings", {}) or {}
    for variable, expected_type in variable_type_constraints.items():
        values = bindings.get(str(variable), []) or []
        if not values:
            continue
        if str(expected_type) == "temporal" and not any(looks_like_temporal_value(value) for value in values):
            return False
    return True


def filter_obligation_matches_by_variable_type_constraints(
    *,
    matches: Mapping[str, Sequence[Mapping[str, Any]]],
    variable_type_constraints: Mapping[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    if not variable_type_constraints:
        return {
            str(obligation_id): [dict(match) for match in obligation_matches]
            for obligation_id, obligation_matches in matches.items()
        }
    return {
        str(obligation_id): [
            dict(match)
            for match in obligation_matches
            if match_satisfies_variable_type_constraints(
                match=match,
                variable_type_constraints=variable_type_constraints,
            )
        ]
        for obligation_id, obligation_matches in matches.items()
    }


def load_head_traces(path: Path | None) -> List[Mapping[str, Any]]:
    if path is None:
        return []
    payload = load_json(path)
    traces = payload.get("head_traces", []) if isinstance(payload, Mapping) else []
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
        return []
    return [trace for trace in traces if isinstance(trace, Mapping)]


def load_query_obligation_cache(path: Path | None, dataset_name: str) -> Dict[int, List[Sequence[Any]]]:
    if path is None:
        return {}
    payload = load_json(path)
    rows: Sequence[Any] = []
    if isinstance(payload, Mapping) and isinstance(payload.get("datasets"), Sequence):
        for dataset_payload in payload.get("datasets", []) or []:
            if not isinstance(dataset_payload, Mapping):
                continue
            if str(dataset_payload.get("dataset") or "") != str(dataset_name):
                continue
            rows = dataset_payload.get("rows", []) or []
            break
    elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), Sequence):
        rows = payload.get("rows", []) or []
    elif isinstance(payload, Mapping):
        result: Dict[int, List[Sequence[Any]]] = {}
        for key, triples in payload.items():
            try:
                query_index = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(triples, Sequence) and not isinstance(triples, (str, bytes)):
                result[query_index] = [
                    triple
                    for triple in triples
                    if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes))
                ]
        return result

    result: Dict[int, List[Sequence[Any]]] = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return result
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        try:
            query_index = int(row.get("query_index", ordinal))
        except (TypeError, ValueError):
            query_index = ordinal
        triples = row.get("query_triples", [])
        if not isinstance(triples, Sequence) or isinstance(triples, (str, bytes)):
            continue
        clean_triples = [
            triple
            for triple in triples
            if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes))
        ]
        if clean_triples:
            result[query_index] = clean_triples
    return result


def build_fact_out_neighbors(
    *,
    candidate_units: Sequence[Mapping[str, Any]],
    max_endpoint_degree: int,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Build the fact-level STO transition graph used by local PPR."""

    facts = fact_units(candidate_units)
    units_by_id = {str(unit.get("unit_id") or ""): unit for unit in facts}
    endpoint_to_unit_ids, _endpoint_labels, endpoint_doc_degrees = build_endpoint_index(facts)
    adjacency, adjacency_stats = build_endpoint_adjacency(
        units_by_id,
        endpoint_to_unit_ids,
        endpoint_doc_degrees,
        max_endpoint_degree=max_endpoint_degree,
    )

    out_neighbors: Dict[str, List[str]] = {unit_id: [] for unit_id in units_by_id}
    for unit_id, edges in adjacency.items():
        neighbors = unique_strings(edge.get("unit_id") for edge in edges)
        out_neighbors[str(unit_id)] = sorted(neighbor for neighbor in neighbors if neighbor in units_by_id)

    stats = {
        "fact_node_count": len(units_by_id),
        "directed_fact_edge_count": sum(len(neighbors) for neighbors in out_neighbors.values()),
        "endpoint_edge_count": int(adjacency_stats["endpoint_edge_count"]),
        "source_edge_count": int(adjacency_stats["source_edge_count"]),
        "hub_endpoint_skipped_count": int(adjacency_stats["hub_endpoint_skipped"]),
    }
    return out_neighbors, stats


def reverse_target_set_local_ppr(
    *,
    out_neighbors: Mapping[str, Sequence[str]],
    target_node_ids: Sequence[str],
    alpha: float = DEFAULT_PPR_ALPHA,
    residual_epsilon: float = DEFAULT_PPR_RESIDUAL_EPSILON,
    max_pushes: int = 1_000_000,
) -> Dict[str, Any]:
    """Approximate PPR contribution from every source node to a target set.

    For a target distribution q, this computes x_s ~= PPR(s -> q), where

        x_s = alpha * q_s + (1 - alpha) * sum P(s,u) * x_u.

    The implementation is target-side reverse local push: residual starts at
    the target set and is pushed only through reverse neighbors that can
    contribute to it. This is the single-target/target-set PPR shape we want:
    candidate evidence units ask "which query seeds can reach me?", rather than
    globally diffusing query mass over all graph nodes.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if residual_epsilon < 0.0:
        raise ValueError("residual_epsilon must be >= 0")

    nodes: Set[str] = set(out_neighbors)
    clean_out: Dict[str, List[str]] = {}
    for source, neighbors in out_neighbors.items():
        clean_neighbors = unique_strings(neighbors)
        clean_out[str(source)] = clean_neighbors
        nodes.update(clean_neighbors)

    target_ids = [node for node in unique_strings(target_node_ids) if node in nodes]
    if not target_ids:
        return {
            "estimate": {},
            "target_node_count": 0,
            "push_count": 0,
            "remaining_residual_mass": 0.0,
            "truncated": False,
        }

    reverse_neighbors: Dict[str, List[str]] = {node: [] for node in nodes}
    out_degree: Dict[str, int] = {node: len(clean_out.get(node, [])) for node in nodes}
    for source, neighbors in clean_out.items():
        for target in neighbors:
            reverse_neighbors.setdefault(target, []).append(source)

    initial_mass = 1.0 / float(len(target_ids))
    residual: Dict[str, float] = {node: initial_mass for node in target_ids}
    estimate: Dict[str, float] = defaultdict(float)
    queue: Deque[str] = deque(target_ids)
    queued: Set[str] = set(target_ids)
    push_count = 0
    truncated = False

    while queue:
        node = queue.popleft()
        queued.discard(node)
        mass = float(residual.pop(node, 0.0))
        if mass <= residual_epsilon:
            continue

        estimate[node] += float(alpha) * mass
        propagated = (1.0 - float(alpha)) * mass
        if propagated > residual_epsilon:
            for predecessor in reverse_neighbors.get(node, []) or []:
                degree = int(out_degree.get(predecessor, 0))
                if degree <= 0:
                    continue
                increment = propagated / float(degree)
                if increment <= 0.0:
                    continue
                next_mass = float(residual.get(predecessor, 0.0)) + increment
                residual[predecessor] = next_mass
                if next_mass > residual_epsilon and predecessor not in queued:
                    queue.append(predecessor)
                    queued.add(predecessor)

        push_count += 1
        if push_count >= max_pushes:
            truncated = bool(queue)
            break

    return {
        "estimate": dict(estimate),
        "target_node_count": len(target_ids),
        "push_count": push_count,
        "remaining_residual_mass": round(sum(residual.values()), 12),
        "truncated": truncated,
    }


def source_set_local_ppr(
    *,
    out_neighbors: Mapping[str, Sequence[str]],
    source_node_ids: Sequence[str],
    alpha: float = DEFAULT_PPR_ALPHA,
    residual_epsilon: float = DEFAULT_PPR_RESIDUAL_EPSILON,
    max_pushes: int = 1_000_000,
) -> Dict[str, Any]:
    """Compute local PPR from a query-source set with deterministic push."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if residual_epsilon < 0.0:
        raise ValueError("residual_epsilon must be >= 0")

    nodes: Set[str] = set(out_neighbors)
    clean_out: Dict[str, List[str]] = {}
    for source, neighbors in out_neighbors.items():
        clean_neighbors = unique_strings(neighbors)
        clean_out[str(source)] = clean_neighbors
        nodes.update(clean_neighbors)

    source_ids = [node for node in unique_strings(source_node_ids) if node in nodes]
    if not source_ids:
        return {
            "estimate": {},
            "source_node_count": 0,
            "push_count": 0,
            "remaining_residual_mass": 0.0,
            "truncated": False,
        }

    initial_mass = 1.0 / float(len(source_ids))
    residual: Dict[str, float] = {node: initial_mass for node in source_ids}
    estimate: Dict[str, float] = defaultdict(float)
    queue: Deque[str] = deque(source_ids)
    queued: Set[str] = set(source_ids)
    push_count = 0
    truncated = False

    while queue:
        node = queue.popleft()
        queued.discard(node)
        mass = float(residual.pop(node, 0.0))
        if mass <= residual_epsilon:
            continue

        estimate[node] += float(alpha) * mass
        neighbors = clean_out.get(node, []) or []
        propagated = (1.0 - float(alpha)) * mass
        if neighbors and propagated > residual_epsilon:
            increment = propagated / float(len(neighbors))
            for neighbor in neighbors:
                next_mass = float(residual.get(neighbor, 0.0)) + increment
                residual[neighbor] = next_mass
                if next_mass > residual_epsilon and neighbor not in queued:
                    queue.append(neighbor)
                    queued.add(neighbor)

        push_count += 1
        if push_count >= max_pushes:
            truncated = bool(queue)
            break

    return {
        "estimate": dict(estimate),
        "source_node_count": len(source_ids),
        "push_count": push_count,
        "remaining_residual_mass": round(sum(residual.values()), 12),
        "truncated": truncated,
    }


def closed_unit_target_node_ids(unit: Mapping[str, Any], graph_nodes: Set[str]) -> List[str]:
    return [node for node in unique_strings(unit.get("path_unit_ids", []) or []) if node in graph_nodes]


def score_closed_units_by_source_ppr(
    *,
    closed_units: Sequence[Mapping[str, Any]],
    seed_unit_ids: Sequence[str],
    out_neighbors: Mapping[str, Sequence[str]],
    obligation_matches: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    alpha: float = DEFAULT_PPR_ALPHA,
    residual_epsilon: float = DEFAULT_PPR_RESIDUAL_EPSILON,
) -> List[Dict[str, Any]]:
    """Score closed units by projected query-source local PPR mass."""

    graph_nodes = set(out_neighbors)
    seed_ids = [unit_id for unit_id in unique_strings(seed_unit_ids) if unit_id in graph_nodes]
    if not seed_ids:
        return []

    local = source_set_local_ppr(
        out_neighbors=out_neighbors,
        source_node_ids=seed_ids,
        alpha=alpha,
        residual_epsilon=residual_epsilon,
    )
    estimate = local.get("estimate", {}) or {}
    obligation_match_unit_ids = {
        str(obligation_id): {
            str(match.get("unit_id") or "")
            for match in matches
            if str(match.get("unit_id") or "")
        }
        for obligation_id, matches in (obligation_matches or {}).items()
    }
    scored_units: List[Dict[str, Any]] = []
    for unit in closed_units:
        target_node_ids = closed_unit_target_node_ids(unit, graph_nodes)
        if not target_node_ids:
            continue
        target_node_set = set(target_node_ids)
        covered_obligation_ids = [
            obligation_id
            for obligation_id, match_unit_ids in sorted(obligation_match_unit_ids.items())
            if target_node_set & match_unit_ids
        ]
        unit_score = sum(float(estimate.get(node_id, 0.0)) for node_id in target_node_ids) / float(len(target_node_ids))
        scored = dict(unit)
        scored["closed_unit_local_ppr_score"] = unit_score
        scored["covered_query_obligation_ids"] = covered_obligation_ids
        scored["covered_query_obligation_count"] = len(covered_obligation_ids)
        scored["closed_unit_local_ppr"] = {
            "alpha": float(alpha),
            "residual_epsilon": float(residual_epsilon),
            "source_node_count": local.get("source_node_count", 0),
            "target_node_count": len(target_node_ids),
            "push_count": local.get("push_count", 0),
            "remaining_residual_mass": local.get("remaining_residual_mass", 0.0),
            "truncated": bool(local.get("truncated", False)),
            "target_mass": {node_id: round(float(estimate.get(node_id, 0.0)), 12) for node_id in target_node_ids},
        }
        scored_units.append(scored)
    return scored_units


def closed_unit_ppr_rank_key(unit: Mapping[str, Any]) -> Tuple[int, float, int, int, int, str]:
    return (
        -int(unit.get("covered_query_obligation_count", 0) or 0),
        -float(unit.get("closed_unit_local_ppr_score", 0.0) or 0.0),
        int(unit.get("path_edge_count", 0) or 0),
        int(unit.get("source_edge_count", 0) or 0),
        len(unique_ints(unit.get("emitted_doc_indices", []) or [])),
        str(unit.get("closed_evidence_unit_id") or ""),
    )


def closed_unit_incremental_rank_key(
    unit: Mapping[str, Any],
    covered_obligation_ids: Set[str],
) -> Tuple[int, int, float, int, int, int, str]:
    unit_obligation_ids = {
        str(obligation_id)
        for obligation_id in unit.get("covered_query_obligation_ids", []) or []
        if str(obligation_id)
    }
    new_obligation_count = len(unit_obligation_ids - covered_obligation_ids)
    return (
        -new_obligation_count,
        -int(unit.get("covered_query_obligation_count", 0) or 0),
        -float(unit.get("closed_unit_local_ppr_score", 0.0) or 0.0),
        int(unit.get("path_edge_count", 0) or 0),
        int(unit.get("source_edge_count", 0) or 0),
        len(unique_ints(unit.get("emitted_doc_indices", []) or [])),
        str(unit.get("closed_evidence_unit_id") or ""),
    )


def build_program_fact_graph(
    *,
    candidate_units: Sequence[Mapping[str, Any]],
    obligation_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    max_endpoint_degree: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    facts = fact_units(candidate_units)
    units_by_id = {str(unit.get("unit_id") or ""): unit for unit in facts}
    endpoint_to_unit_ids, _endpoint_labels, endpoint_doc_degrees = build_endpoint_index(facts)
    adjacency, adjacency_stats = build_endpoint_adjacency(
        units_by_id,
        endpoint_to_unit_ids,
        endpoint_doc_degrees,
        max_endpoint_degree=max_endpoint_degree,
    )
    constrained_endpoint_keys = source_span_transport_endpoint_keys_by_unit(obligation_matches)
    adjacency, source_span_filtered_edge_count = filter_source_span_transport_edges(
        units_by_id=units_by_id,
        adjacency=adjacency,
        constrained_endpoint_keys=constrained_endpoint_keys,
    )
    support_edge_count = 0
    edge_keys: Set[Tuple[str, str, str]] = set()
    doc_to_anchor_units: Dict[Tuple[int, str], Set[str]] = defaultdict(set)
    for unit_id, unit in units_by_id.items():
        doc_index = int(unit.get("doc_index", -1) or -1)
        for endpoint, _label in unit_endpoint_entries(unit):
            doc_to_anchor_units[(doc_index, endpoint)].add(unit_id)
    for matches in obligation_matches.values():
        for match in matches:
            if not bool(match.get("support_grounding", False)):
                continue
            source_id = str(match.get("unit_id") or "")
            source_unit = units_by_id.get(source_id)
            if not source_unit:
                continue
            doc_index = int(source_unit.get("doc_index", -1) or -1)
            for anchor in match.get("text_anchor_hits", []) or []:
                for target_id in sorted(doc_to_anchor_units.get((doc_index, str(anchor)), set())):
                    if target_id == source_id:
                        continue
                    key = tuple(sorted([source_id, target_id]) + [f"support:{doc_index}:{anchor}"])
                    if key in edge_keys:
                        continue
                    edge_keys.add(key)
                    adjacency[source_id].append(
                        {
                            "unit_id": target_id,
                            "edge_kind": "support_grounding_anchor",
                            "source_doc_index": doc_index,
                            "anchor_key": str(anchor),
                        }
                    )
                    adjacency[target_id].append(
                        {
                            "unit_id": source_id,
                            "edge_kind": "support_grounding_anchor",
                            "source_doc_index": doc_index,
                            "anchor_key": str(anchor),
                        }
                    )
                    support_edge_count += 1
    for unit_id in list(adjacency):
        adjacency[unit_id] = sorted(
            adjacency[unit_id],
            key=lambda edge: (
                str(edge.get("edge_kind") or ""),
                str(edge.get("endpoint_key") or edge.get("anchor_key") or edge.get("source_doc_index") or ""),
                str(edge.get("unit_id") or ""),
            ),
        )
    graph_stats = {
        "fact_node_count": len(units_by_id),
        "endpoint_edge_count": int(adjacency_stats["endpoint_edge_count"]),
        "source_edge_count": int(adjacency_stats["source_edge_count"]),
        "support_grounding_anchor_edge_count": int(support_edge_count),
        "source_span_transport_constrained_unit_count": len(constrained_endpoint_keys),
        "source_span_transport_filtered_edge_count": int(source_span_filtered_edge_count),
        "hub_endpoint_skipped_count": int(adjacency_stats["hub_endpoint_skipped"]),
    }
    return units_by_id, {str(key): list(edges) for key, edges in adjacency.items()}, graph_stats


def obligation_match_unit_ids_by_obligation(
    obligation_matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Set[str]]:
    return {
        str(obligation_id): {
            str(match.get("unit_id") or "")
            for match in matches
            if str(match.get("unit_id") or "")
        }
        for obligation_id, matches in obligation_matches.items()
    }


def source_span_transport_endpoint_keys_by_unit(
    obligation_matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Set[str]]:
    """Return transport constraints for units used only as source-span evidence.

    A source-span match says "this source sentence grounds the obligation"; it
    does not license every endpoint of the carrier OpenIE tuple for graph
    transport. If the same unit is also an exact/support match, keep the normal
    fact graph behavior because the tuple itself is then part of the grounding.
    """

    constrained: Dict[str, Set[str]] = defaultdict(set)
    unconstrained: Set[str] = set()
    for matches in obligation_matches.values():
        for match in matches or []:
            unit_id = str(match.get("unit_id") or "")
            if not unit_id:
                continue
            if bool(match.get("source_span_grounding", False)):
                constrained[unit_id].update(
                    endpoint_key(anchor)
                    for anchor in match.get("source_span_transport_endpoint_keys", []) or []
                    if endpoint_key(anchor)
                )
            else:
                unconstrained.add(unit_id)
    return {
        unit_id: keys
        for unit_id, keys in constrained.items()
        if unit_id not in unconstrained
    }


def source_span_transport_edge_allowed(
    *,
    source_id: str,
    target_id: str,
    edge: Mapping[str, Any],
    units_by_id: Mapping[str, Mapping[str, Any]],
    constrained_endpoint_keys: Mapping[str, Set[str]],
) -> bool:
    allowed = constrained_endpoint_keys.get(source_id)
    if allowed is None:
        return True
    edge_kind = str(edge.get("edge_kind") or "endpoint_transfer")
    if edge_kind == "endpoint_transfer":
        return str(edge.get("endpoint_key") or "") in allowed
    if edge_kind == "support_grounding_anchor":
        return str(edge.get("anchor_key") or "") in allowed
    return False


def filter_source_span_transport_edges(
    *,
    units_by_id: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[Mapping[str, Any]]],
    constrained_endpoint_keys: Mapping[str, Set[str]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    if not constrained_endpoint_keys:
        return {
            str(unit_id): [dict(edge) for edge in adjacency.get(str(unit_id), []) or []]
            for unit_id in units_by_id
        }, 0

    filtered: Dict[str, List[Dict[str, Any]]] = {}
    skipped = 0
    for unit_id in units_by_id:
        edges = adjacency.get(str(unit_id), []) or []
        clean_edges: List[Dict[str, Any]] = []
        for edge in edges or []:
            target_id = str(edge.get("unit_id") or "")
            if not target_id:
                continue
            forward_allowed = source_span_transport_edge_allowed(
                source_id=str(unit_id),
                target_id=target_id,
                edge=edge,
                units_by_id=units_by_id,
                constrained_endpoint_keys=constrained_endpoint_keys,
            )
            reverse_allowed = source_span_transport_edge_allowed(
                source_id=target_id,
                target_id=str(unit_id),
                edge=edge,
                units_by_id=units_by_id,
                constrained_endpoint_keys=constrained_endpoint_keys,
            )
            if not (forward_allowed and reverse_allowed):
                skipped += 1
                continue
            clean_edges.append(dict(edge))
        filtered[str(unit_id)] = clean_edges
    return filtered, skipped


def matches_by_unit(
    obligation_matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obligation_id, matches in obligation_matches.items():
        for match in matches or []:
            unit_id = str(match.get("unit_id") or "")
            if not unit_id:
                continue
            row = dict(match)
            row.setdefault("obligation_id", str(obligation_id))
            result[unit_id].append(row)
    return {unit_id: rows for unit_id, rows in result.items()}


def match_semantic_rank(match: Mapping[str, Any], unit: Mapping[str, Any] | None = None) -> Tuple[int, int, int, int, str]:
    if bool(match.get("source_span_grounding", False)):
        grounding_rank = 1 if bool(match.get("source_span_tuple_relation_aligned", False)) else 3
    elif bool(match.get("support_grounding", False)):
        grounding_rank = 2
    else:
        grounding_rank = 0
    anchor_count = len(match.get("source_span_transport_endpoint_keys", []) or [])
    return (
        grounding_rank,
        -anchor_count,
        int(match.get("doc_index", (unit or {}).get("doc_index", -1)) or -1),
        len(str((unit or {}).get("fact", ""))),
        str(match.get("unit_id") or (unit or {}).get("unit_id") or ""),
    )


def best_match_for_unit(
    *,
    unit_id: str,
    unit_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    unit: Mapping[str, Any],
) -> Dict[str, Any]:
    matches = [dict(match) for match in unit_matches.get(unit_id, []) or []]
    if not matches:
        return {}
    return sorted(matches, key=lambda match: match_semantic_rank(match, unit))[0]


def path_fact_row(
    *,
    unit: Mapping[str, Any],
    covered_query_obligation_ids: Sequence[str],
    unit_matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    unit_id = str(unit.get("unit_id") or "")
    best_match = best_match_for_unit(unit_id=unit_id, unit_matches=unit_matches, unit=unit)
    row: Dict[str, Any] = {
        "unit_id": unit_id,
        "doc_index": int(unit.get("doc_index", -1) or -1),
        "title": str(unit.get("title") or ""),
        "fact": list(unit.get("fact", []) or []),
        "covered_query_obligation_ids": sorted(covered_query_obligation_ids),
    }
    if bool(best_match.get("source_span_grounding", False)):
        row["evidence_grounding"] = "source_span"
        row["source_span_sentence"] = str(best_match.get("source_span_sentence") or "")
        row["source_span_tuple_relation_aligned"] = bool(
            best_match.get("source_span_tuple_relation_aligned", False)
        )
        row["source_span_transport_endpoint_keys"] = list(
            best_match.get("source_span_transport_endpoint_keys", []) or []
        )
    elif bool(best_match.get("support_grounding", False)):
        row["evidence_grounding"] = "support"
    else:
        row["evidence_grounding"] = "exact_fact"
    return row


def assemble_typed_component_cover(
    *,
    required_ids: Sequence[str],
    obligation_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    units_by_id: Mapping[str, Mapping[str, Any]],
    unit_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence_set_size: int,
    graph_stats: Mapping[str, Any],
    obligation_index: Mapping[str, int],
    visited_state_count: int,
    max_program_edges: int,
    disconnected_component_reason: str,
) -> Dict[str, Any]:
    selected_matches: List[Dict[str, Any]] = []
    selected_unit_ids: List[str] = []
    selected_unit_set: Set[str] = set()
    for obligation_id in required_ids:
        ranked_matches = sorted(
            [
                match
                for match in obligation_matches.get(obligation_id, []) or []
                if str(match.get("unit_id") or "") in units_by_id
            ],
            key=lambda match: match_semantic_rank(
                match,
                units_by_id.get(str(match.get("unit_id") or ""), {}),
            ),
        )
        if not ranked_matches:
            continue
        selected = dict(ranked_matches[0])
        selected_matches.append(selected)
        unit_id = str(selected.get("unit_id") or "")
        if unit_id and unit_id not in selected_unit_set:
            selected_unit_ids.append(unit_id)
            selected_unit_set.add(unit_id)
    covered_ids = {
        str(match.get("obligation_id") or "")
        for match in selected_matches
        if str(match.get("obligation_id") or "")
    }
    path_units = [units_by_id[unit_id] for unit_id in selected_unit_ids if unit_id in units_by_id]
    emitted_doc_indices = unique_ints(unit.get("doc_index") for unit in path_units)
    if list(required_ids) != sorted(covered_ids, key=lambda item: obligation_index[item]):
        return {
            "feasible": False,
            "reason": "typed_component_cover_missing_obligations",
            "required_query_obligation_ids": list(required_ids),
            "missing_query_obligation_ids": [
                obligation_id for obligation_id in required_ids if obligation_id not in covered_ids
            ],
            "fact_graph_summary": dict(graph_stats),
            "visited_state_count": visited_state_count,
            "max_program_edges": max_program_edges,
        }
    if len(emitted_doc_indices) > evidence_set_size:
        return {
            "feasible": False,
            "reason": "typed_component_cover_exceeds_evidence_set_size",
            "required_query_obligation_ids": list(required_ids),
            "covered_query_obligation_ids": sorted(covered_ids, key=lambda item: obligation_index[item]),
            "emitted_doc_indices": emitted_doc_indices,
            "fact_graph_summary": dict(graph_stats),
            "visited_state_count": visited_state_count,
            "max_program_edges": max_program_edges,
        }
    selected_obligations_by_unit: Dict[str, List[str]] = defaultdict(list)
    for match in selected_matches:
        selected_obligations_by_unit[str(match.get("unit_id") or "")].append(
            str(match.get("obligation_id") or "")
        )
    path_facts = [
        path_fact_row(
            unit=unit,
            covered_query_obligation_ids=selected_obligations_by_unit.get(str(unit.get("unit_id") or ""), []),
            unit_matches=unit_matches,
        )
        for unit in path_units
    ]
    return {
        "feasible": True,
        "reason": "covered_all_query_obligations_as_typed_components",
        "assembly_mode": "typed_operator_component_cover",
        "typed_component_reason": str(disconnected_component_reason or "typed_operator"),
        "required_query_obligation_ids": list(required_ids),
        "covered_query_obligation_ids": sorted(covered_ids, key=lambda item: obligation_index[item]),
        "missing_query_obligation_ids": [
            obligation_id for obligation_id in required_ids if obligation_id not in covered_ids
        ],
        "path_unit_ids": selected_unit_ids,
        "path_edges": [],
        "path_facts": path_facts,
        "emitted_doc_indices": emitted_doc_indices,
        "fact_graph_summary": dict(graph_stats),
        "visited_state_count": visited_state_count,
        "max_program_edges": max_program_edges,
    }


def assemble_program_evidence_walk(
    *,
    query_obligations: Sequence[Mapping[str, Any]],
    obligation_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_units: Sequence[Mapping[str, Any]],
    evidence_set_size: int,
    max_path_edges: int,
    max_endpoint_degree: int,
    allow_disconnected_components: bool = False,
    disconnected_component_reason: str = "",
) -> Dict[str, Any]:
    """Find a small connected STO fact walk covering every query obligation."""

    required_ids = [
        str(obligation.get("obligation_id") or "")
        for obligation in query_obligations
        if str(obligation.get("obligation_id") or "")
    ]
    if not required_ids:
        return {"feasible": False, "reason": "no_query_obligations"}
    if len(required_ids) > 15:
        return {"feasible": False, "reason": "too_many_query_obligations_for_exact_product_search"}

    match_unit_ids = obligation_match_unit_ids_by_obligation(obligation_matches)
    missing_match_ids = [obligation_id for obligation_id in required_ids if not match_unit_ids.get(obligation_id)]
    if missing_match_ids:
        return {
            "feasible": False,
            "reason": "missing_obligation_matches",
            "missing_query_obligation_ids": missing_match_ids,
        }

    units_by_id, adjacency, graph_stats = build_program_fact_graph(
        candidate_units=candidate_units,
        obligation_matches=obligation_matches,
        max_endpoint_degree=max_endpoint_degree,
    )
    unit_matches = matches_by_unit(obligation_matches)
    obligation_index = {obligation_id: index for index, obligation_id in enumerate(required_ids)}
    full_mask = (1 << len(required_ids)) - 1
    node_masks: Dict[str, int] = {}
    node_obligations: Dict[str, List[str]] = {}
    for obligation_id, unit_ids in match_unit_ids.items():
        if obligation_id not in obligation_index:
            continue
        bit = 1 << obligation_index[obligation_id]
        for unit_id in unit_ids:
            if unit_id not in units_by_id:
                continue
            node_masks[unit_id] = node_masks.get(unit_id, 0) | bit
            node_obligations.setdefault(unit_id, []).append(obligation_id)
    if not node_masks:
        return {"feasible": False, "reason": "no_matched_units_in_fact_graph", "fact_graph_summary": graph_stats}

    max_program_edges = max(int(max_path_edges), 1) * max(len(required_ids) - 1, 1)
    if allow_disconnected_components:
        component_program = assemble_typed_component_cover(
            required_ids=required_ids,
            obligation_matches=obligation_matches,
            units_by_id=units_by_id,
            unit_matches=unit_matches,
            evidence_set_size=evidence_set_size,
            graph_stats=graph_stats,
            obligation_index=obligation_index,
            visited_state_count=0,
            max_program_edges=max_program_edges,
            disconnected_component_reason=disconnected_component_reason,
        )
        return component_program
    queue: Deque[Tuple[str, int, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]]] = deque()
    visited: Set[Tuple[str, int]] = set()
    for unit_id in sorted(node_masks):
        mask = node_masks[unit_id]
        state = (unit_id, mask)
        visited.add(state)
        queue.append((unit_id, mask, (unit_id,), tuple()))

    best: Dict[str, Any] | None = None
    while queue:
        current_id, mask, path_unit_ids, path_edges = queue.popleft()
        if mask == full_mask:
            best = {
                "path_unit_ids": list(path_unit_ids),
                "path_edges": [
                    {"edge_kind": edge[0], "from_unit_id": edge[1], "to_unit_id": edge[2]}
                    for edge in path_edges
                ],
            }
            break
        if len(path_edges) >= max_program_edges:
            continue
        for edge in adjacency.get(current_id, []) or []:
            next_id = str(edge.get("unit_id") or "")
            if not next_id:
                continue
            next_mask = mask | node_masks.get(next_id, 0)
            state = (next_id, next_mask)
            if state in visited:
                continue
            visited.add(state)
            edge_kind = str(edge.get("edge_kind") or "endpoint_transfer")
            next_edge = (edge_kind, current_id, next_id)
            queue.append((next_id, next_mask, path_unit_ids + (next_id,), path_edges + (next_edge,)))

    if best is None:
        if allow_disconnected_components:
            selected_matches: List[Dict[str, Any]] = []
            selected_unit_ids: List[str] = []
            selected_unit_set: Set[str] = set()
            for obligation_id in required_ids:
                ranked_matches = sorted(
                    [
                        match
                        for match in obligation_matches.get(obligation_id, []) or []
                        if str(match.get("unit_id") or "") in units_by_id
                    ],
                    key=lambda match: match_semantic_rank(
                        match,
                        units_by_id.get(str(match.get("unit_id") or ""), {}),
                    ),
                )
                if not ranked_matches:
                    continue
                selected = dict(ranked_matches[0])
                selected_matches.append(selected)
                unit_id = str(selected.get("unit_id") or "")
                if unit_id and unit_id not in selected_unit_set:
                    selected_unit_ids.append(unit_id)
                    selected_unit_set.add(unit_id)
            covered_ids = {
                str(match.get("obligation_id") or "")
                for match in selected_matches
                if str(match.get("obligation_id") or "")
            }
            path_units = [units_by_id[unit_id] for unit_id in selected_unit_ids if unit_id in units_by_id]
            emitted_doc_indices = unique_ints(unit.get("doc_index") for unit in path_units)
            if required_ids == sorted(covered_ids, key=lambda item: obligation_index[item]) and len(emitted_doc_indices) <= evidence_set_size:
                selected_obligations_by_unit: Dict[str, List[str]] = defaultdict(list)
                for match in selected_matches:
                    selected_obligations_by_unit[str(match.get("unit_id") or "")].append(
                        str(match.get("obligation_id") or "")
                    )
                path_facts = [
                    path_fact_row(
                        unit=unit,
                        covered_query_obligation_ids=selected_obligations_by_unit.get(
                            str(unit.get("unit_id") or ""), []
                        ),
                        unit_matches=unit_matches,
                    )
                    for unit in path_units
                ]
                return {
                    "feasible": True,
                    "reason": "covered_all_query_obligations_as_typed_components",
                    "assembly_mode": "typed_operator_component_cover",
                    "typed_component_reason": str(disconnected_component_reason or "typed_operator"),
                    "required_query_obligation_ids": required_ids,
                    "covered_query_obligation_ids": sorted(covered_ids, key=lambda item: obligation_index[item]),
                    "missing_query_obligation_ids": [
                        obligation_id for obligation_id in required_ids if obligation_id not in covered_ids
                    ],
                    "path_unit_ids": selected_unit_ids,
                    "path_edges": [],
                    "path_facts": path_facts,
                    "emitted_doc_indices": emitted_doc_indices,
                    "fact_graph_summary": graph_stats,
                    "visited_state_count": len(visited),
                    "max_program_edges": max_program_edges,
                }
        return {
            "feasible": False,
            "reason": "no_connected_fact_walk_covering_all_obligations",
            "required_query_obligation_ids": required_ids,
            "fact_graph_summary": graph_stats,
            "visited_state_count": len(visited),
            "max_program_edges": max_program_edges,
        }

    path_units = [units_by_id[unit_id] for unit_id in best["path_unit_ids"] if unit_id in units_by_id]
    emitted_doc_indices = unique_ints(unit.get("doc_index") for unit in path_units)
    if len(emitted_doc_indices) > evidence_set_size:
        return {
            "feasible": False,
            "reason": "connected_fact_walk_exceeds_evidence_set_size",
            "required_query_obligation_ids": required_ids,
            "covered_query_obligation_ids": required_ids,
            "emitted_doc_indices": emitted_doc_indices,
            "fact_graph_summary": graph_stats,
            "max_program_edges": max_program_edges,
        }

    covered_ids: Set[str] = set()
    for unit_id in best["path_unit_ids"]:
        covered_ids.update(node_obligations.get(unit_id, []) or [])
    path_facts = [
        path_fact_row(
            unit=unit,
            covered_query_obligation_ids=node_obligations.get(str(unit.get("unit_id") or ""), []) or [],
            unit_matches=unit_matches,
        )
        for unit in path_units
    ]
    return {
        "feasible": required_ids == sorted(covered_ids, key=lambda item: obligation_index[item]),
        "reason": "covered_all_query_obligations",
        "assembly_mode": "connected_fact_walk",
        "required_query_obligation_ids": required_ids,
        "covered_query_obligation_ids": sorted(covered_ids, key=lambda item: obligation_index[item]),
        "missing_query_obligation_ids": [
            obligation_id for obligation_id in required_ids if obligation_id not in covered_ids
        ],
        "path_unit_ids": best["path_unit_ids"],
        "path_edges": best["path_edges"],
        "path_facts": path_facts,
        "emitted_doc_indices": emitted_doc_indices,
        "fact_graph_summary": graph_stats,
        "visited_state_count": len(visited),
        "max_program_edges": max_program_edges,
    }


def select_obligation_closed_local_ppr_evidence(
    *,
    query: str,
    candidate_doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
    query_triples: Sequence[Sequence[Any]] | None = None,
    fallback_doc_indices: Sequence[int] | None = None,
    allow_lexical_fallback: bool = False,
    allow_partial_obligation_grounding: bool = False,
    allow_support_obligation_grounding: bool = False,
    allow_source_span_obligation_grounding: bool = False,
    allow_program_evidence_assembler: bool = False,
    use_typed_retrieval_critical_obligations: bool = False,
    evidence_set_size: int = 5,
    max_path_edges: int = 3,
    max_endpoint_degree: int = 30,
    alpha: float = DEFAULT_PPR_ALPHA,
    residual_epsilon: float = DEFAULT_PPR_RESIDUAL_EPSILON,
    doc_units_by_index: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> Dict[str, Any]:
    query_tokens = content_tokens(query) - QUERY_FUNCTION_TOKENS
    candidate_docs = unique_ints(candidate_doc_indices)
    candidate_units = build_candidate_sto_units(
        candidate_doc_indices=candidate_docs,
        openie_docs=openie_docs,
        doc_units_by_index=doc_units_by_index,
    )
    program_query_triples = lower_query_triples_for_typed_program(
        query=query,
        query_triples=query_triples or [],
    )
    query_obligations = build_query_obligation_units(
        query=query,
        query_triples=program_query_triples,
    )
    typed_plan = typed_execution_plan(
        query=query,
        query_triples=program_query_triples,
        materialized_obligations=query_obligations,
    ) if use_typed_retrieval_critical_obligations else {}
    if typed_plan:
        required_obligation_ids = set(typed_plan.get("retrieval_critical_obligation_ids", []) or [])
        execution_query_obligations = [
            obligation
            for obligation in query_obligations
            if str(obligation.get("obligation_id") or "") in required_obligation_ids
        ]
    else:
        execution_query_obligations = list(query_obligations)
    exact_obligation_matches = match_query_obligations_to_sto_facts(
        obligations=execution_query_obligations,
        candidate_units=candidate_units,
    )
    exact_obligation_matches = filter_obligation_matches_by_variable_type_constraints(
        matches=exact_obligation_matches,
        variable_type_constraints=(typed_plan.get("variable_type_constraints", {}) or {}) if typed_plan else {},
    )
    support_obligation_matches = (
        support_grounded_obligation_matches(
            obligations=execution_query_obligations,
            candidate_units=candidate_units,
            exact_matches=exact_obligation_matches,
        )
        if allow_support_obligation_grounding
        else {}
    )
    support_obligation_matches = filter_obligation_matches_by_variable_type_constraints(
        matches=support_obligation_matches,
        variable_type_constraints=(typed_plan.get("variable_type_constraints", {}) or {}) if typed_plan else {},
    )
    exact_and_support_matches = (
        merge_exact_and_support_matches(
            exact_matches=exact_obligation_matches,
            support_matches=support_obligation_matches,
        )
        if support_obligation_matches
        else {str(obligation_id): [dict(match) for match in matches] for obligation_id, matches in exact_obligation_matches.items()}
    )
    if support_obligation_matches:
        exact_obligation_matches = match_query_obligations_to_sto_facts(
            obligations=execution_query_obligations,
            candidate_units=candidate_units,
            seed_matches=exact_and_support_matches,
        )
        exact_obligation_matches = filter_obligation_matches_by_variable_type_constraints(
            matches=exact_obligation_matches,
            variable_type_constraints=(typed_plan.get("variable_type_constraints", {}) or {}) if typed_plan else {},
        )
        closure_variable_values = variable_values_from_matches(
            obligations=execution_query_obligations,
            matches=exact_obligation_matches,
        )
        support_obligation_matches = filter_matches_by_variable_values(
            matches=support_obligation_matches,
            variable_values=closure_variable_values,
        )
        exact_and_support_matches = merge_exact_and_support_matches(
            exact_matches=exact_obligation_matches,
            support_matches=support_obligation_matches,
        )
    source_span_obligation_matches = (
        source_span_grounded_obligation_matches(
            obligations=execution_query_obligations,
            candidate_units=candidate_units,
            exact_matches=exact_and_support_matches,
        )
        if allow_source_span_obligation_grounding
        else {}
    )
    source_span_obligation_matches = filter_obligation_matches_by_variable_type_constraints(
        matches=source_span_obligation_matches,
        variable_type_constraints=(typed_plan.get("variable_type_constraints", {}) or {}) if typed_plan else {},
    )
    obligation_matches = (
        merge_exact_and_support_matches(
            exact_matches=exact_and_support_matches,
            support_matches=source_span_obligation_matches,
        )
        if support_obligation_matches or source_span_obligation_matches
        else {str(obligation_id): [dict(match) for match in matches] for obligation_id, matches in exact_obligation_matches.items()}
    )
    exact_grounded_obligation_ids = [
        str(obligation.get("obligation_id") or "")
        for obligation in execution_query_obligations
        if exact_obligation_matches.get(str(obligation.get("obligation_id") or ""), [])
    ]
    support_grounded_obligation_ids = [
        str(obligation_id)
        for obligation_id, matches in support_obligation_matches.items()
        if matches
    ]
    source_span_grounded_obligation_ids = [
        str(obligation_id)
        for obligation_id, matches in source_span_obligation_matches.items()
        if matches
    ]
    grounded_obligation_ids = [
        str(obligation.get("obligation_id") or "")
        for obligation in execution_query_obligations
        if obligation_matches.get(str(obligation.get("obligation_id") or ""), [])
    ]
    retrieval_critical_unmaterialized_count = int(typed_plan.get("retrieval_critical_unmaterialized_count", 0) or 0)
    typed_non_retrieval_materialized_count = int(typed_plan.get("materialized_non_retrieval_obligation_count", 0) or 0) if typed_plan else 0
    grounding_report = {
        "query_obligation_count": len(execution_query_obligations),
        "query_obligation_grounded_count": len(grounded_obligation_ids),
        "query_obligation_exact_grounded_count": len(exact_grounded_obligation_ids),
        "query_obligation_support_grounded_count": len(support_grounded_obligation_ids),
        "query_obligation_source_span_grounded_count": len(source_span_grounded_obligation_ids),
        "query_obligation_match_count": sum(len(matches) for matches in obligation_matches.values()),
        "query_obligations": execution_query_obligations,
        "program_query_triples": [list(triple) for triple in program_query_triples],
        "program_query_triple_count": len(program_query_triples),
        "materialized_query_obligation_count": len(query_obligations),
        "materialized_query_obligations": query_obligations,
        "use_typed_retrieval_critical_obligations": bool(use_typed_retrieval_critical_obligations),
        "typed_query_program": typed_plan,
        "typed_variable_type_constraints": dict(typed_plan.get("variable_type_constraints", {}) or {})
        if typed_plan
        else {},
        "typed_retrieval_critical_count": int(typed_plan.get("retrieval_critical_count", len(execution_query_obligations)) or 0)
        if typed_plan
        else len(execution_query_obligations),
        "typed_retrieval_critical_grounded_count": len(grounded_obligation_ids)
        if typed_plan
        else 0,
        "typed_retrieval_critical_unmaterialized_count": retrieval_critical_unmaterialized_count,
        "typed_materialized_non_retrieval_obligation_count": typed_non_retrieval_materialized_count,
        "support_grounded_obligation_matches": support_obligation_matches,
        "source_span_grounded_obligation_matches": source_span_obligation_matches,
        "allow_support_obligation_grounding": bool(allow_support_obligation_grounding),
        "allow_source_span_obligation_grounding": bool(allow_source_span_obligation_grounding),
    }
    if typed_plan and not execution_query_obligations and retrieval_critical_unmaterialized_count <= 0:
        prior_docs = unique_ints(list(fallback_doc_indices or []) + candidate_docs)
        return {
            "retrieved_doc_indices": prior_docs[:evidence_set_size],
            "selected_closed_units": [],
            "query_tokens": sorted(query_tokens),
            "frontier_query_tokens": [],
            "seed_source": "typed_no_retrieval_critical_obligation_abstain_to_prior",
            "abstention_reason": "no_retrieval_critical_obligations",
            **grounding_report,
            "candidate_doc_count": len(candidate_docs),
            "seed_unit_count": 0,
            "candidate_closed_unit_count": 0,
            "positive_scored_closed_unit_count": 0,
            "closed_store_summary": {},
            "fact_graph_summary": {},
            "selection_policy": "typed_program_abstain_to_prior_without_retrieval_critical_obligations",
        }
    if typed_plan and retrieval_critical_unmaterialized_count > 0:
        prior_docs = unique_ints(list(fallback_doc_indices or []) + candidate_docs)
        return {
            "retrieved_doc_indices": prior_docs[:evidence_set_size],
            "selected_closed_units": [],
            "query_tokens": sorted(query_tokens),
            "frontier_query_tokens": [],
            "seed_source": "typed_retrieval_critical_unmaterialized_abstain_to_prior",
            "abstention_reason": "retrieval_critical_unmaterialized",
            **grounding_report,
            "candidate_doc_count": len(candidate_docs),
            "seed_unit_count": 0,
            "candidate_closed_unit_count": 0,
            "positive_scored_closed_unit_count": 0,
            "closed_store_summary": {},
            "fact_graph_summary": {},
            "selection_policy": "typed_program_abstain_to_prior_with_unmaterialized_retrieval_critical_obligations",
        }
    if (
        execution_query_obligations
        and not allow_partial_obligation_grounding
        and len(grounded_obligation_ids) < len(execution_query_obligations)
    ):
        prior_docs = unique_ints(list(fallback_doc_indices or []) + candidate_docs)
        return {
            "retrieved_doc_indices": prior_docs[:evidence_set_size],
            "selected_closed_units": [],
            "query_tokens": sorted(query_tokens),
            "frontier_query_tokens": [],
            "seed_source": "query_obligation_incomplete_grounding_abstain_to_prior",
            "abstention_reason": "retrieval_critical_ungrounded"
            if typed_plan
            else "materialized_obligation_ungrounded",
            **grounding_report,
            "candidate_doc_count": len(candidate_docs),
            "seed_unit_count": 0,
            "candidate_closed_unit_count": 0,
            "positive_scored_closed_unit_count": 0,
            "closed_store_summary": {},
            "fact_graph_summary": {},
            "selection_policy": "obligation_only_local_ppr_abstain_to_prior_without_full_query_grounding",
        }
    obligation_seed_ids = obligation_seed_unit_ids(obligation_matches)
    if obligation_seed_ids:
        frontier_tokens: Set[str] = set()
        seed_unit_ids = obligation_seed_ids
        seed_source = (
            "query_obligation_bound_facts_with_source_span_grounding"
            if source_span_grounded_obligation_ids
            else "query_obligation_bound_facts_with_support_grounding"
            if support_grounded_obligation_ids
            else "query_obligation_bound_facts"
        )
    elif allow_lexical_fallback:
        frontier_tokens = frontier_query_tokens(query_tokens=query_tokens, candidate_units=candidate_units)
        seed_unit_ids = seed_unit_ids_for_query(
            query_tokens=frontier_tokens,
            candidate_units=candidate_units,
        )
        seed_source = "lexical_frontier_fallback"
    else:
        prior_docs = unique_ints(list(fallback_doc_indices or []) + candidate_docs)
        return {
            "retrieved_doc_indices": prior_docs[:evidence_set_size],
            "selected_closed_units": [],
            "query_tokens": sorted(query_tokens),
            "frontier_query_tokens": [],
            "seed_source": "no_query_obligation_abstain_to_prior",
            "abstention_reason": "no_bound_obligations",
            **grounding_report,
            "candidate_doc_count": len(candidate_docs),
            "seed_unit_count": 0,
            "candidate_closed_unit_count": 0,
            "positive_scored_closed_unit_count": 0,
            "closed_store_summary": {},
            "fact_graph_summary": {},
            "selection_policy": "obligation_only_local_ppr_abstain_to_prior_without_bound_obligations",
        }
    if allow_program_evidence_assembler and execution_query_obligations:
        typed_component_operator = ""
        if typed_plan and int((typed_plan.get("type_counts", {}) or {}).get("comparison", 0) or 0) > 0:
            typed_component_operator = "comparison"
        program = assemble_program_evidence_walk(
            query_obligations=execution_query_obligations,
            obligation_matches=obligation_matches,
            candidate_units=candidate_units,
            evidence_set_size=evidence_set_size,
            max_path_edges=max_path_edges,
            max_endpoint_degree=max_endpoint_degree,
            allow_disconnected_components=bool(typed_component_operator),
            disconnected_component_reason=typed_component_operator,
        )
        if bool(program.get("feasible", False)):
            selected_docs = unique_ints(program.get("emitted_doc_indices", []) or [])
            selected_doc_set = set(selected_docs)
            for doc in unique_ints(list(fallback_doc_indices or []) + candidate_docs):
                if len(selected_docs) >= evidence_set_size:
                    break
                if doc in selected_doc_set:
                    continue
                selected_docs.append(doc)
                selected_doc_set.add(doc)
            return {
                "retrieved_doc_indices": selected_docs[:evidence_set_size],
                "selected_closed_units": [],
                "program_evidence_assembly": program,
                "query_tokens": sorted(query_tokens),
                "frontier_query_tokens": sorted(frontier_tokens),
                "seed_source": "query_obligation_program_evidence_assembler",
                "abstention_reason": "",
                "typed_unblock_reason": "non_retrieval_materialized_obligations_excluded"
                if typed_plan and typed_non_retrieval_materialized_count > 0
                else "",
                **grounding_report,
                "candidate_doc_count": len(candidate_docs),
                "seed_unit_count": len(seed_unit_ids),
                "candidate_closed_unit_count": 0,
                "positive_scored_closed_unit_count": 0,
                "closed_store_summary": {},
                "fact_graph_summary": program.get("fact_graph_summary", {}),
                "selection_policy": "program_level_connected_fact_walk_covering_all_obligations_then_prior_fill",
            }
        if execution_query_obligations and not allow_partial_obligation_grounding:
            prior_docs = unique_ints(list(fallback_doc_indices or []) + candidate_docs)
            return {
                "retrieved_doc_indices": prior_docs[:evidence_set_size],
                "selected_closed_units": [],
                "program_evidence_assembly": program,
                "query_tokens": sorted(query_tokens),
                "frontier_query_tokens": sorted(frontier_tokens),
                "seed_source": "query_obligation_program_evidence_assembler_abstain_to_prior",
                "abstention_reason": "no_connected_feasible_evidence_subgraph",
                **grounding_report,
                "candidate_doc_count": len(candidate_docs),
                "seed_unit_count": len(seed_unit_ids),
                "candidate_closed_unit_count": 0,
                "positive_scored_closed_unit_count": 0,
                "closed_store_summary": {},
                "fact_graph_summary": program.get("fact_graph_summary", {}),
                "selection_policy": "program_level_connected_fact_walk_abstain_to_prior_without_feasible_full_coverage",
            }
    store = build_seed_closed_unit_store(
        candidate_units=candidate_units,
        seed_unit_ids=seed_unit_ids,
        max_path_edges=max_path_edges,
        max_endpoint_degree=max_endpoint_degree,
    )
    closed_units = [
        unit
        for unit in store.get("closed_units", []) or []
        if 0 < len(unique_ints(unit.get("emitted_doc_indices", []) or [])) <= evidence_set_size
    ]
    if not closed_units:
        prior_docs = unique_ints(list(fallback_doc_indices or []) + candidate_docs)
        return {
            "retrieved_doc_indices": prior_docs[:evidence_set_size],
            "selected_closed_units": [],
            "query_tokens": sorted(query_tokens),
            "frontier_query_tokens": sorted(frontier_tokens),
            "seed_source": "query_obligation_no_closed_unit_abstain_to_prior",
            "abstention_reason": "no_closed_evidence_units",
            **grounding_report,
            "candidate_doc_count": len(candidate_docs),
            "seed_unit_count": len(seed_unit_ids),
            "candidate_closed_unit_count": 0,
            "positive_scored_closed_unit_count": 0,
            "closed_store_summary": store.get("summary", {}),
            "fact_graph_summary": {},
            "selection_policy": "obligation_only_local_ppr_abstain_to_prior_without_closed_units",
        }
    out_neighbors, graph_stats = build_fact_out_neighbors(
        candidate_units=candidate_units,
        max_endpoint_degree=max_endpoint_degree,
    )
    scored_units = score_closed_units_by_source_ppr(
        closed_units=closed_units,
        seed_unit_ids=seed_unit_ids,
        out_neighbors=out_neighbors,
        obligation_matches=obligation_matches,
        alpha=alpha,
        residual_epsilon=residual_epsilon,
    )
    positive_scored_count = sum(
        1 for unit in scored_units if float(unit.get("closed_unit_local_ppr_score", 0.0) or 0.0) > 0.0
    )
    if positive_scored_count <= 0:
        prior_docs = unique_ints(list(fallback_doc_indices or []) + candidate_docs)
        return {
            "retrieved_doc_indices": prior_docs[:evidence_set_size],
            "selected_closed_units": [],
            "query_tokens": sorted(query_tokens),
            "frontier_query_tokens": sorted(frontier_tokens),
            "seed_source": "query_obligation_no_positive_ppr_abstain_to_prior",
            "abstention_reason": "no_positive_local_ppr",
            **grounding_report,
            "candidate_doc_count": len(candidate_docs),
            "seed_unit_count": len(seed_unit_ids),
            "candidate_closed_unit_count": len(closed_units),
            "positive_scored_closed_unit_count": 0,
            "closed_store_summary": store.get("summary", {}),
            "fact_graph_summary": graph_stats,
            "selection_policy": "obligation_only_local_ppr_abstain_to_prior_without_positive_ppr",
        }

    selected_docs: List[int] = []
    selected_doc_set: Set[int] = set()
    selected_units: List[Dict[str, Any]] = []
    selected_unit_ids: Set[str] = set()
    selected_obligation_ids: Set[str] = set()
    required_obligation_ids = {
        str(obligation.get("obligation_id") or "")
        for obligation in execution_query_obligations
        if str(obligation.get("obligation_id") or "")
    }
    positive_units = [
        unit
        for unit in scored_units
        if float(unit.get("closed_unit_local_ppr_score", 0.0) or 0.0) > 0.0
    ]

    while len(selected_docs) < evidence_set_size:
        ranked_units = [
            unit
            for unit in positive_units
            if str(unit.get("closed_evidence_unit_id") or "") not in selected_unit_ids
        ]
        if not ranked_units:
            break
        unit = min(
            ranked_units,
            key=lambda item: closed_unit_incremental_rank_key(item, selected_obligation_ids),
        )
        unit_id = str(unit.get("closed_evidence_unit_id") or "")
        emitted_docs = unique_ints(unit.get("emitted_doc_indices", []) or [])
        new_docs = [doc for doc in emitted_docs if doc not in selected_doc_set]
        if not new_docs or len(selected_docs) + len(new_docs) > evidence_set_size:
            selected_unit_ids.add(unit_id)
            continue
        unit_obligation_ids = {
            str(obligation_id)
            for obligation_id in unit.get("covered_query_obligation_ids", []) or []
            if str(obligation_id)
        }
        if required_obligation_ids and required_obligation_ids - selected_obligation_ids and not (unit_obligation_ids - selected_obligation_ids):
            selected_unit_ids.add(unit_id)
            continue
        for doc in new_docs:
            selected_docs.append(doc)
            selected_doc_set.add(doc)
        selected_unit_ids.add(unit_id)
        selected_obligation_ids.update(unit_obligation_ids)
        selected_units.append(
            {
                "closed_evidence_unit_id": unit.get("closed_evidence_unit_id"),
                "emitted_doc_indices": emitted_docs,
                "path_doc_indices": unit.get("path_doc_indices", []),
                "path_endpoint_keys": unit.get("path_endpoint_keys", []),
                "path_titles": unit.get("path_titles", []),
                "path_edge_count": unit.get("path_edge_count"),
                "source_edge_count": unit.get("source_edge_count"),
                "closed_unit_local_ppr_score": round(float(unit.get("closed_unit_local_ppr_score", 0.0) or 0.0), 12),
                "covered_query_obligation_ids": unit.get("covered_query_obligation_ids", []),
                "covered_query_obligation_count": unit.get("covered_query_obligation_count", 0),
                "closed_unit_local_ppr": unit.get("closed_unit_local_ppr", {}),
            }
        )

    if (
        execution_query_obligations
        and not allow_partial_obligation_grounding
        and required_obligation_ids
        and not required_obligation_ids <= selected_obligation_ids
    ):
        prior_docs = unique_ints(list(fallback_doc_indices or []) + candidate_docs)
        return {
            "retrieved_doc_indices": prior_docs[:evidence_set_size],
            "selected_closed_units": [],
            "diagnostic_selected_closed_units": selected_units,
            "selected_query_obligation_ids": sorted(selected_obligation_ids),
            "missing_selected_query_obligation_ids": sorted(required_obligation_ids - selected_obligation_ids),
            "query_tokens": sorted(query_tokens),
            "frontier_query_tokens": sorted(frontier_tokens),
            "seed_source": "query_obligation_no_full_coverage_abstain_to_prior",
            "abstention_reason": "closed_unit_selection_missing_required_obligations",
            **grounding_report,
            "candidate_doc_count": len(candidate_docs),
            "seed_unit_count": len(seed_unit_ids),
            "candidate_closed_unit_count": len(closed_units),
            "positive_scored_closed_unit_count": positive_scored_count,
            "closed_store_summary": store.get("summary", {}),
            "fact_graph_summary": graph_stats,
            "selection_policy": "obligation_only_local_ppr_abstain_to_prior_without_full_closed_unit_coverage",
        }

    remaining_docs = [
        doc
        for doc in unique_ints(list(fallback_doc_indices or []) + candidate_docs)
        if doc not in selected_doc_set
    ]
    uncovered_tokens = set(query_tokens)
    for doc in selected_docs:
        uncovered_tokens -= doc_signature_tokens(openie_docs, doc)
    for doc in remaining_docs:
        if len(selected_docs) >= evidence_set_size:
            break
        selected_docs.append(doc)
        selected_doc_set.add(doc)
        uncovered_tokens -= doc_signature_tokens(openie_docs, doc)

    return {
        "retrieved_doc_indices": selected_docs[:evidence_set_size],
        "selected_closed_units": selected_units,
        "query_tokens": sorted(query_tokens),
        "frontier_query_tokens": sorted(frontier_tokens),
        "seed_source": seed_source,
        "abstention_reason": "",
        "typed_unblock_reason": "non_retrieval_materialized_obligations_excluded"
        if typed_plan and typed_non_retrieval_materialized_count > 0
        else "",
        **grounding_report,
        "selected_query_obligation_ids": sorted(selected_obligation_ids),
        "missing_selected_query_obligation_ids": sorted(required_obligation_ids - selected_obligation_ids),
        "candidate_doc_count": len(candidate_docs),
        "seed_unit_count": len(seed_unit_ids),
        "candidate_closed_unit_count": len(closed_units),
        "positive_scored_closed_unit_count": positive_scored_count,
        "closed_store_summary": store.get("summary", {}),
        "fact_graph_summary": graph_stats,
        "selection_policy": "query_source_local_ppr_projection_over_closed_sto_units_then_prior_fill",
    }


def evaluate_dataset(
    *,
    dataset_payload: Mapping[str, Any],
    max_queries: int,
    evidence_set_size: int,
    max_path_edges: int,
    max_endpoint_degree: int,
    alpha: float,
    residual_epsilon: float,
    query_indices: Set[int] | None = None,
    head_traces_cache_path: Path | None = None,
    query_obligation_cache_path: Path | None = None,
    allow_lexical_fallback: bool = False,
    allow_partial_obligation_grounding: bool = False,
    allow_support_obligation_grounding: bool = False,
    allow_source_span_obligation_grounding: bool = False,
    allow_program_evidence_assembler: bool = False,
    use_typed_retrieval_critical_obligations: bool = False,
) -> Dict[str, Any]:
    openie_docs = list(load_json(Path(str(dataset_payload["openie_path"]))).get("docs", []) or [])
    head_traces = load_head_traces(head_traces_cache_path)
    dataset_name = str(dataset_payload.get("dataset") or "")
    query_obligation_cache = load_query_obligation_cache(query_obligation_cache_path, dataset_name)
    rows = list(dataset_payload.get("rows", []) or [])
    if query_indices:
        selected_indices = set(query_indices)
        rows = [
            row
            for row in rows
            if row_query_index(row) in selected_indices
        ]
    if max_queries > 0:
        rows = rows[:max_queries]

    needed_doc_indices: Set[int] = set()
    for row in rows:
        needed_doc_indices.update(candidate_doc_indices_from_row(row))
    doc_units_by_index = {
        doc_index: build_candidate_sto_units(candidate_doc_indices=[doc_index], openie_docs=openie_docs)
        for doc_index in sorted(needed_doc_indices)
        if 0 <= int(doc_index) < len(openie_docs)
    }

    output_rows: List[Dict[str, Any]] = []
    sums: Counter[str] = Counter()
    candidate_counts: List[int] = []
    closed_unit_counts: List[int] = []
    positive_unit_counts: List[int] = []
    changed_count = 0
    improved_gold_count = 0
    regressed_gold_count = 0
    obligation_query_count = 0
    exact_full_grounded_count = 0
    effective_full_grounded_count = 0
    support_grounding_used_count = 0
    source_span_grounding_used_count = 0
    program_assembler_attempted_count = 0
    program_assembler_feasible_count = 0
    program_assembler_changed_count = 0
    program_assembler_improved_count = 0
    program_assembler_regressed_count = 0
    program_reason_counts: Counter[str] = Counter()
    abstention_reason_counts: Counter[str] = Counter()
    typed_unblock_reason_counts: Counter[str] = Counter()
    typed_full_grounded_count = 0
    typed_retrieval_critical_unmaterialized_query_count = 0
    typed_materialized_non_retrieval_obligation_count = 0
    typed_executed_with_non_retrieval_materialized_count = 0

    for row in rows:
        query = str(row.get("question") or row.get("query") or "")
        gold_doc_indices = unique_ints(row.get("gold_doc_indices", []) or [])
        candidate_doc_indices = candidate_doc_indices_from_row(row)
        source_prior = row_doc_indices(
            row,
            "anchor_guided_evidence_doc_indices_top10",
            "retrieved_doc_indices_top10",
            "anchor_guided_evidence_doc_indices_top5",
            "retrieved_doc_indices_top5",
            limit=10,
        )
        result = select_obligation_closed_local_ppr_evidence(
            query=query,
            candidate_doc_indices=candidate_doc_indices,
            openie_docs=openie_docs,
            query_triples=query_triples_from_row(
                row,
                head_traces=head_traces,
                query_obligation_cache=query_obligation_cache,
            ),
            fallback_doc_indices=source_prior,
            allow_lexical_fallback=allow_lexical_fallback,
            allow_partial_obligation_grounding=allow_partial_obligation_grounding,
            allow_support_obligation_grounding=allow_support_obligation_grounding,
            allow_source_span_obligation_grounding=allow_source_span_obligation_grounding,
            allow_program_evidence_assembler=allow_program_evidence_assembler,
            use_typed_retrieval_critical_obligations=use_typed_retrieval_critical_obligations,
            evidence_set_size=evidence_set_size,
            max_path_edges=max_path_edges,
            max_endpoint_degree=max_endpoint_degree,
            alpha=alpha,
            residual_epsilon=residual_epsilon,
            doc_units_by_index=doc_units_by_index,
        )
        retrieved = unique_ints(result.get("retrieved_doc_indices", []) or [])
        source_top5 = source_prior[:5]
        old_gold_count = len(set(source_top5[:5]) & set(gold_doc_indices))
        new_gold_count = len(set(retrieved[:5]) & set(gold_doc_indices))
        if retrieved[:5] != source_top5[:5]:
            changed_count += 1
        if new_gold_count > old_gold_count:
            improved_gold_count += 1
        elif new_gold_count < old_gold_count:
            regressed_gold_count += 1
        query_obligation_count = int(result.get("query_obligation_count", 0) or 0)
        exact_grounded_count = int(result.get("query_obligation_exact_grounded_count", 0) or 0)
        support_grounded_count = int(result.get("query_obligation_support_grounded_count", 0) or 0)
        source_span_grounded_count = int(result.get("query_obligation_source_span_grounded_count", 0) or 0)
        effective_grounded_count = int(result.get("query_obligation_grounded_count", 0) or 0)
        abstention_reason_counts[str(result.get("abstention_reason") or "executed")] += 1
        typed_unblock_reason_counts[str(result.get("typed_unblock_reason") or "none")] += 1
        typed_retrieval_critical_count = int(result.get("typed_retrieval_critical_count", 0) or 0)
        typed_retrieval_critical_grounded_count = int(result.get("typed_retrieval_critical_grounded_count", 0) or 0)
        typed_retrieval_critical_unmaterialized_count = int(result.get("typed_retrieval_critical_unmaterialized_count", 0) or 0)
        typed_materialized_non_retrieval_count = int(result.get("typed_materialized_non_retrieval_obligation_count", 0) or 0)
        typed_materialized_non_retrieval_obligation_count += typed_materialized_non_retrieval_count
        if not str(result.get("abstention_reason") or "").strip() and typed_materialized_non_retrieval_count > 0:
            typed_executed_with_non_retrieval_materialized_count += 1
        if typed_retrieval_critical_unmaterialized_count > 0:
            typed_retrieval_critical_unmaterialized_query_count += 1
        if (
            use_typed_retrieval_critical_obligations
            and typed_retrieval_critical_count > 0
            and typed_retrieval_critical_unmaterialized_count == 0
            and typed_retrieval_critical_grounded_count == typed_retrieval_critical_count
        ):
            typed_full_grounded_count += 1
        if query_obligation_count > 0:
            obligation_query_count += 1
            if exact_grounded_count == query_obligation_count:
                exact_full_grounded_count += 1
            if effective_grounded_count == query_obligation_count:
                effective_full_grounded_count += 1
            if support_grounded_count > 0:
                support_grounding_used_count += 1
            if source_span_grounded_count > 0:
                source_span_grounding_used_count += 1
        program = result.get("program_evidence_assembly")
        if isinstance(program, Mapping):
            program_assembler_attempted_count += 1
            program_reason_counts[str(program.get("reason") or "unknown")] += 1
            if bool(program.get("feasible", False)):
                program_assembler_feasible_count += 1
                if retrieved[:5] != source_top5[:5]:
                    program_assembler_changed_count += 1
                if new_gold_count > old_gold_count:
                    program_assembler_improved_count += 1
                elif new_gold_count < old_gold_count:
                    program_assembler_regressed_count += 1

        r5 = recall_at_k(gold_doc_indices, retrieved, 5)
        r10 = recall_at_k(gold_doc_indices, retrieved, 10)
        sums["r5"] += r5
        sums["r10"] += r10
        sums["all_gold_at5"] += 1.0 if all_gold_at_k(gold_doc_indices, retrieved, 5) else 0.0
        sums["all_gold_at10"] += 1.0 if all_gold_at_k(gold_doc_indices, retrieved, 10) else 0.0
        candidate_counts.append(int(result.get("candidate_doc_count", 0) or 0))
        closed_unit_counts.append(int(result.get("candidate_closed_unit_count", 0) or 0))
        positive_unit_counts.append(int(result.get("positive_scored_closed_unit_count", 0) or 0))

        out = dict(row)
        out["obligation_closed_sto_local_ppr_doc_indices_top5"] = retrieved[:5]
        out["obligation_closed_sto_local_ppr_doc_indices_top10"] = retrieved[:10]
        out["obligation_closed_sto_local_ppr_recall_at5"] = round(float(r5), 6)
        out["obligation_closed_sto_local_ppr_recall_at10"] = round(float(r10), 6)
        out["obligation_closed_sto_local_ppr_all_gold_at5"] = bool(all_gold_at_k(gold_doc_indices, retrieved, 5))
        out["obligation_closed_sto_local_ppr_all_gold_at10"] = bool(all_gold_at_k(gold_doc_indices, retrieved, 10))
        out["obligation_closed_sto_local_ppr_selector"] = result
        out["obligation_closed_sto_local_ppr_gold_count_delta_vs_source_top5"] = new_gold_count - old_gold_count
        output_rows.append(out)

    denom = max(len(output_rows), 1)
    metrics = {
        "obligation_closed_sto_local_ppr_r5": round(sums["r5"] / float(denom), 6),
        "obligation_closed_sto_local_ppr_r10": round(sums["r10"] / float(denom), 6),
        "obligation_closed_sto_local_ppr_all_gold_at5": round(sums["all_gold_at5"] / float(denom), 6),
        "obligation_closed_sto_local_ppr_all_gold_at10": round(sums["all_gold_at10"] / float(denom), 6),
        "mean_candidate_doc_count": round(sum(candidate_counts) / float(max(len(candidate_counts), 1)), 6),
        "mean_candidate_closed_unit_count": round(sum(closed_unit_counts) / float(max(len(closed_unit_counts), 1)), 6),
        "mean_positive_scored_closed_unit_count": round(sum(positive_unit_counts) / float(max(len(positive_unit_counts), 1)), 6),
        "head_trace_count": len(head_traces),
        "query_obligation_cache_row_count": len(query_obligation_cache),
        "obligation_query_count": obligation_query_count,
        "exact_full_grounded_query_count": exact_full_grounded_count,
        "effective_full_grounded_query_count": effective_full_grounded_count,
        "support_grounding_used_query_count": support_grounding_used_count,
        "source_span_grounding_used_query_count": source_span_grounding_used_count,
        "program_assembler_attempted_query_count": program_assembler_attempted_count,
        "program_assembler_feasible_query_count": program_assembler_feasible_count,
        "program_assembler_changed_top5_queries": program_assembler_changed_count,
        "program_assembler_gold_count_improved_queries": program_assembler_improved_count,
        "program_assembler_gold_count_regressed_queries": program_assembler_regressed_count,
        "program_assembler_reason_counts": dict(sorted(program_reason_counts.items())),
        "abstention_reason_counts": dict(sorted(abstention_reason_counts.items())),
        "typed_unblock_reason_counts": dict(sorted(typed_unblock_reason_counts.items())),
        "typed_retrieval_critical_full_grounded_query_count": typed_full_grounded_count,
        "typed_retrieval_critical_unmaterialized_query_count": typed_retrieval_critical_unmaterialized_query_count,
        "typed_materialized_non_retrieval_obligation_count": typed_materialized_non_retrieval_obligation_count,
        "typed_executed_with_non_retrieval_materialized_query_count": typed_executed_with_non_retrieval_materialized_count,
        "changed_top5_queries": changed_count,
        "gold_count_improved_queries": improved_gold_count,
        "gold_count_regressed_queries": regressed_gold_count,
    }

    payload = dict(dataset_payload)
    payload["metrics"] = {**dict(dataset_payload.get("metrics", {}) or {}), **metrics}
    payload["rows"] = output_rows
    return payload


def write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Obligation-Closed STO Local PPR",
        "",
        "This is a diagnostic selector over existing candidate documents. It computes local PPR from query-seed facts over the STO fact graph and projects that mass onto closed STO units.",
        "",
        "## Config",
        "",
        "| parameter | value |",
        "| --- | ---: |",
    ]
    for key, value in summary.get("config", {}).items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| dataset | rows | R@5 | all-gold@5 | exact full | effective full | program feasible | changed | gains | losses |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in summary.get("datasets", []) or []:
        metrics = dataset.get("metrics", {}) or {}
        lines.append(
            f"| {dataset.get('dataset')} | {len(dataset.get('rows', []) or [])} | "
            f"{metrics.get('obligation_closed_sto_local_ppr_r5', 0.0):.4f} | "
            f"{metrics.get('obligation_closed_sto_local_ppr_all_gold_at5', 0.0):.4f} | "
            f"{metrics.get('exact_full_grounded_query_count', 0)} | "
            f"{metrics.get('effective_full_grounded_query_count', 0)} | "
            f"{metrics.get('program_assembler_feasible_query_count', 0)} | "
            f"{metrics.get('changed_top5_queries', 0)} | "
            f"{metrics.get('gold_count_improved_queries', 0)} | "
            f"{metrics.get('gold_count_regressed_queries', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Method Boundary",
            "",
            "- Uses local PPR over the STO fact graph, not global HippoRAG v2 diffusion.",
            "- Uses no gold labels, no QA outcomes, no SFB outputs, no answer-type rules, and no weighted late fusion.",
            "- Still uses the source report's candidate document universe, so this is not yet a standalone retriever.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def local_ppr_metric_view(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "obligation_closed_sto_local_ppr_r5",
        "obligation_closed_sto_local_ppr_r10",
        "obligation_closed_sto_local_ppr_all_gold_at5",
        "obligation_closed_sto_local_ppr_all_gold_at10",
        "mean_candidate_doc_count",
        "mean_candidate_closed_unit_count",
        "mean_positive_scored_closed_unit_count",
        "obligation_query_count",
        "exact_full_grounded_query_count",
        "effective_full_grounded_query_count",
        "support_grounding_used_query_count",
        "source_span_grounding_used_query_count",
        "program_assembler_attempted_query_count",
        "program_assembler_feasible_query_count",
        "program_assembler_changed_top5_queries",
        "program_assembler_gold_count_improved_queries",
        "program_assembler_gold_count_regressed_queries",
        "program_assembler_reason_counts",
        "abstention_reason_counts",
        "typed_unblock_reason_counts",
        "typed_retrieval_critical_full_grounded_query_count",
        "typed_retrieval_critical_unmaterialized_query_count",
        "typed_materialized_non_retrieval_obligation_count",
        "typed_executed_with_non_retrieval_materialized_query_count",
        "changed_top5_queries",
        "gold_count_improved_queries",
        "gold_count_regressed_queries",
    ]
    return {key: metrics.get(key) for key in keys}


def evaluate_obligation_closed_sto_local_ppr(
    *,
    report_path: Path,
    max_queries: int,
    evidence_set_size: int,
    max_path_edges: int,
    max_endpoint_degree: int,
    alpha: float,
    residual_epsilon: float,
    query_indices: Set[int] | None,
    head_traces_cache_paths: Mapping[str, Path] | None,
    query_obligation_cache_path: Path | None,
    output_json_path: Path,
    output_md_path: Path,
    allow_lexical_fallback: bool = False,
    allow_partial_obligation_grounding: bool = False,
    allow_support_obligation_grounding: bool = False,
    allow_source_span_obligation_grounding: bool = False,
    allow_program_evidence_assembler: bool = False,
    use_typed_retrieval_critical_obligations: bool = False,
) -> Dict[str, Any]:
    report = load_json(report_path)
    datasets = [
        evaluate_dataset(
            dataset_payload=dataset,
            max_queries=max_queries,
            evidence_set_size=evidence_set_size,
            max_path_edges=max_path_edges,
            max_endpoint_degree=max_endpoint_degree,
            alpha=alpha,
            residual_epsilon=residual_epsilon,
            query_indices=query_indices or set(),
            head_traces_cache_path=(
                head_traces_cache_paths.get(str(dataset.get("dataset") or ""))
                if head_traces_cache_paths
                else None
            ),
            query_obligation_cache_path=query_obligation_cache_path,
            allow_lexical_fallback=allow_lexical_fallback,
            allow_partial_obligation_grounding=allow_partial_obligation_grounding,
            allow_support_obligation_grounding=allow_support_obligation_grounding,
            allow_source_span_obligation_grounding=allow_source_span_obligation_grounding,
            allow_program_evidence_assembler=allow_program_evidence_assembler,
            use_typed_retrieval_critical_obligations=use_typed_retrieval_critical_obligations,
        )
        for dataset in report.get("datasets", []) or []
    ]
    summary = {
        "source_report_path": str(report_path),
        "method": "obligation_closed_sto_query_local_ppr_diagnostic",
        "config": {
            "max_queries": int(max_queries),
            "evidence_set_size": int(evidence_set_size),
            "max_path_edges": int(max_path_edges),
            "max_endpoint_degree": int(max_endpoint_degree),
            "alpha": float(alpha),
            "residual_epsilon": float(residual_epsilon),
            "query_indices": sorted(query_indices or set()),
            "head_traces_cache_paths": {
                dataset: str(path) for dataset, path in (head_traces_cache_paths or {}).items()
            },
            "query_obligation_cache_path": str(query_obligation_cache_path) if query_obligation_cache_path else "",
            "allow_lexical_fallback": bool(allow_lexical_fallback),
            "allow_partial_obligation_grounding": bool(allow_partial_obligation_grounding),
            "allow_support_obligation_grounding": bool(allow_support_obligation_grounding),
            "allow_source_span_obligation_grounding": bool(allow_source_span_obligation_grounding),
            "allow_program_evidence_assembler": bool(allow_program_evidence_assembler),
            "use_typed_retrieval_critical_obligations": bool(use_typed_retrieval_critical_obligations),
        },
        "datasets": datasets,
    }
    output_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, output_md_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate target local PPR over closed STO units.")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--evidence-set-size", type=int, default=5)
    parser.add_argument("--query-indices", default="", help="Optional comma/range query indices, for example 714 or 0-4,714.")
    parser.add_argument("--max-path-edges", type=int, default=3)
    parser.add_argument("--max-endpoint-degree", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=DEFAULT_PPR_ALPHA)
    parser.add_argument("--residual-epsilon", type=float, default=DEFAULT_PPR_RESIDUAL_EPSILON)
    parser.add_argument(
        "--head-traces-cache-map-json",
        default="",
        help="Optional JSON mapping from dataset name to baseline retrieval cache containing head_traces.",
    )
    parser.add_argument(
        "--query-obligation-cache-json",
        default="",
        help="Optional query-obligation cache produced by build_query_obligation_cache.py.",
    )
    parser.add_argument(
        "--allow-lexical-fallback",
        action="store_true",
        help="Diagnostic only: fall back to lexical seeds when no query obligation binds.",
    )
    parser.add_argument(
        "--allow-partial-obligation-grounding",
        action="store_true",
        help="Diagnostic only: allow graph execution when only part of the query obligation program grounds.",
    )
    parser.add_argument(
        "--allow-support-obligation-grounding",
        action="store_true",
        help="Ablation only: admit source-supported obligations after exact endpoint grounding.",
    )
    parser.add_argument(
        "--allow-source-span-obligation-grounding",
        action="store_true",
        help="Ablation only: admit role-aware source-span grounding with explicit provenance.",
    )
    parser.add_argument(
        "--allow-program-evidence-assembler",
        action="store_true",
        help="Ablation only: assemble a typed evidence program that covers every grounded query obligation.",
    )
    parser.add_argument(
        "--use-typed-retrieval-critical-obligations",
        action="store_true",
        help="Ablation only: require only typed retrieval-critical obligations, while abstaining on retrieval-critical builder omissions.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    head_traces_cache_paths: Dict[str, Path] = {}
    if str(args.head_traces_cache_map_json or "").strip():
        raw_map = load_json(Path(args.head_traces_cache_map_json).resolve())
        if not isinstance(raw_map, Mapping):
            raise ValueError("--head-traces-cache-map-json must point to a JSON object")
        head_traces_cache_paths = {
            str(dataset): Path(str(path)).resolve()
            for dataset, path in raw_map.items()
            if str(dataset).strip() and str(path).strip()
        }

    summary = evaluate_obligation_closed_sto_local_ppr(
        report_path=Path(args.report).resolve(),
        max_queries=max(int(args.max_queries), 0),
        evidence_set_size=max(int(args.evidence_set_size), 1),
        max_path_edges=max(int(args.max_path_edges), 1),
        max_endpoint_degree=max(int(args.max_endpoint_degree), 2),
        alpha=float(args.alpha),
        residual_epsilon=max(float(args.residual_epsilon), 0.0),
        query_indices=parse_query_indices(args.query_indices),
        head_traces_cache_paths=head_traces_cache_paths,
        query_obligation_cache_path=(
            Path(args.query_obligation_cache_json).resolve()
            if str(args.query_obligation_cache_json or "").strip()
            else None
        ),
        allow_lexical_fallback=bool(args.allow_lexical_fallback),
        allow_partial_obligation_grounding=bool(args.allow_partial_obligation_grounding),
        allow_support_obligation_grounding=bool(args.allow_support_obligation_grounding),
        allow_source_span_obligation_grounding=bool(args.allow_source_span_obligation_grounding),
        allow_program_evidence_assembler=bool(args.allow_program_evidence_assembler),
        use_typed_retrieval_critical_obligations=bool(args.use_typed_retrieval_critical_obligations),
        output_json_path=Path(args.output_json).resolve(),
        output_md_path=Path(args.output_md).resolve(),
    )
    compact = {
        dataset["dataset"]: local_ppr_metric_view(dataset.get("metrics", {}))
        for dataset in summary.get("datasets", []) or []
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
