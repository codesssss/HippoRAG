#!/usr/bin/env python3
"""Evaluate a clean selector over obligation-closed STO units.

This is a diagnostic selector over an existing candidate universe. It does not
use SFB, PPR scores, QA outcomes, gold labels, weighted fusion, or answer-type
rules. The selection objective is greedy query-token coverage by closed
evidence units under the reader document budget.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from build_obligation_closed_sto_units import (
    build_closed_unit,
    build_endpoint_adjacency,
    build_endpoint_index,
    closed_unit_canonical_id,
    closed_unit_preference_key,
    fact_units,
    shortest_endpoint_paths_from,
    unit_endpoint_entries,
)
from build_source_title_openie_substrate import build_units_for_doc, content_tokens
from evaluate_transition_component_retriever import all_gold_at_k, recall_at_k, unique_ranked


DEFAULT_REPORT = (
    "outputs_anchor_guided_evidence_20260426/reports/"
    "anchor_guided_evidence_full_from_transition_v13_answer_role_gap_prefixguard.json"
)

QUERY_FUNCTION_TOKENS = {
    "did",
    "does",
    "get",
    "got",
    "had",
    "has",
    "have",
    "how",
    "known",
    "one",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_ints(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    seen: Set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def row_doc_indices(row: Mapping[str, Any], *keys: str, limit: int | None = None) -> List[int]:
    for key in keys:
        values = row.get(key)
        if values:
            docs = unique_ints(values)
            return docs[:limit] if limit is not None else docs
    return []


def candidate_doc_indices_from_row(row: Mapping[str, Any]) -> List[int]:
    docs: List[int] = []
    for key in (
        "candidate_doc_indices",
        "anchor_guided_evidence_doc_indices_top10",
        "retrieved_doc_indices_top10",
        "support_set_doc_indices_top10",
        "neighborhood_doc_indices_top10",
        "specificity_pair_doc_indices_top10",
        "endpoint_transition_doc_indices_top10",
        "hybrid_residual_pair_doc_indices_top10",
        "native_dense_doc_indices_top10",
        "bm25_doc_indices_top10",
    ):
        docs.extend(row_doc_indices(row, key))
    return unique_ints(docs)


def doc_title(openie_docs: Sequence[Mapping[str, Any]], doc_index: int) -> str:
    if not 0 <= doc_index < len(openie_docs):
        return ""
    passage = str(openie_docs[doc_index].get("passage") or "")
    return passage.splitlines()[0].strip() if passage else ""


def doc_signature_tokens(openie_docs: Sequence[Mapping[str, Any]], doc_index: int) -> Set[str]:
    if not 0 <= doc_index < len(openie_docs):
        return set()
    doc = openie_docs[doc_index]
    values = [doc_title(openie_docs, doc_index)]
    for triple in doc.get("extracted_triples", []) or []:
        if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes)):
            values.extend(str(item) for item in triple[:3])
    return content_tokens(" ".join(values))


def closed_unit_signature_tokens(unit: Mapping[str, Any]) -> Set[str]:
    values: List[str] = []
    values.extend(str(value) for value in unit.get("path_titles", []) or [])
    values.extend(str(value) for value in unit.get("path_endpoint_keys", []) or [])
    values.extend(str(value) for value in unit.get("path_endpoints", []) or [])
    for role in unit.get("relation_roles", []) or []:
        if not isinstance(role, Mapping):
            continue
        values.extend(
            [
                str(role.get("title") or ""),
                str(role.get("subject") or ""),
                str(role.get("relation") or ""),
                str(role.get("object") or ""),
            ]
        )
    return content_tokens(" ".join(values))


def build_candidate_sto_units(
    *,
    candidate_doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
    doc_units_by_index: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for doc_index in candidate_doc_indices:
        if not 0 <= int(doc_index) < len(openie_docs):
            continue
        if doc_units_by_index is not None and int(doc_index) in doc_units_by_index:
            units.extend(dict(unit) for unit in doc_units_by_index[int(doc_index)])
        else:
            units.extend(
                build_units_for_doc(
                    openie_docs[int(doc_index)],
                    doc_index=int(doc_index),
                    include_source_spans=False,
                )
            )
    return units


def unit_signature_tokens(unit: Mapping[str, Any]) -> Set[str]:
    values = [
        unit.get("title", ""),
        unit.get("subject", ""),
        unit.get("relation", ""),
        unit.get("object", ""),
        unit.get("grounded_subject", ""),
        unit.get("grounded_object", ""),
    ]
    values.extend(str(value) for _key, value in unit_endpoint_entries(unit))
    return content_tokens(" ".join(str(value or "") for value in values))


def unit_activation_tokens(unit: Mapping[str, Any]) -> Set[str]:
    values = [
        unit.get("title", ""),
        unit.get("subject", ""),
        unit.get("object", ""),
        unit.get("grounded_subject", ""),
        unit.get("grounded_object", ""),
    ]
    values.extend(str(value) for _key, value in unit_endpoint_entries(unit))
    return content_tokens(" ".join(str(value or "") for value in values))


def seed_unit_ids_for_query(
    *,
    query_tokens: Set[str],
    candidate_units: Sequence[Mapping[str, Any]],
) -> List[str]:
    facts = fact_units(candidate_units)
    return [
        str(unit.get("unit_id") or "")
        for unit in facts
        if unit_activation_tokens(unit) & query_tokens
    ]


def frontier_query_tokens(
    *,
    query_tokens: Set[str],
    candidate_units: Sequence[Mapping[str, Any]],
) -> Set[str]:
    doc_sets: Dict[str, Set[int]] = {token: set() for token in query_tokens}
    for unit in fact_units(candidate_units):
        doc_index = int(unit.get("doc_index", -1))
        for token in unit_activation_tokens(unit) & query_tokens:
            doc_sets.setdefault(token, set()).add(doc_index)
    positive = {token: len(docs) for token, docs in doc_sets.items() if docs}
    if not positive:
        return set(query_tokens)
    ordered_freqs = sorted(positive.values())
    median_freq = ordered_freqs[(len(ordered_freqs) - 1) // 2]
    frontier = {token for token, freq in positive.items() if freq <= median_freq}
    return frontier or set(query_tokens)


def build_seed_closed_unit_store(
    *,
    candidate_units: Sequence[Mapping[str, Any]],
    seed_unit_ids: Sequence[str],
    max_path_edges: int,
    max_endpoint_degree: int,
) -> Dict[str, Any]:
    facts = fact_units(candidate_units)
    units_by_id = {str(unit.get("unit_id") or ""): unit for unit in facts}
    seed_set = {str(unit_id) for unit_id in seed_unit_ids if str(unit_id) in units_by_id}
    endpoint_to_unit_ids, endpoint_labels, endpoint_doc_degrees = build_endpoint_index(facts)
    unit_to_endpoints = {
        str(unit.get("unit_id") or ""): [key for key, _label in unit_endpoint_entries(unit)]
        for unit in facts
    }
    transfer_bearing_unit_ids = {
        unit_id
        for endpoint, unit_ids in endpoint_to_unit_ids.items()
        if 2 <= int(endpoint_doc_degrees.get(endpoint, 0)) <= max_endpoint_degree
        for unit_id in unit_ids
    }
    doc_to_transfer_units: Dict[int, Set[str]] = {}
    for unit_id in transfer_bearing_unit_ids:
        unit = units_by_id.get(str(unit_id))
        if not unit:
            continue
        doc_to_transfer_units.setdefault(int(unit.get("doc_index", -1)), set()).add(str(unit_id))

    active_unit_ids: Set[str] = set(seed_set)
    for unit_id in list(active_unit_ids):
        unit = units_by_id.get(unit_id)
        if unit:
            active_unit_ids.update(doc_to_transfer_units.get(int(unit.get("doc_index", -1)), set()))
    frontier: Set[str] = set(active_unit_ids)
    for _hop in range(max(int(max_path_edges), 0)):
        next_frontier: Set[str] = set()
        for unit_id in sorted(frontier):
            for endpoint in unit_to_endpoints.get(unit_id, []) or []:
                if int(endpoint_doc_degrees.get(endpoint, 0)) > max_endpoint_degree:
                    continue
                for neighbor_id in endpoint_to_unit_ids.get(endpoint, []) or []:
                    neighbor_id = str(neighbor_id)
                    if neighbor_id in active_unit_ids:
                        continue
                    active_unit_ids.add(neighbor_id)
                    next_frontier.add(neighbor_id)
                    neighbor = units_by_id.get(neighbor_id)
                    if neighbor:
                        same_doc_units = doc_to_transfer_units.get(int(neighbor.get("doc_index", -1)), set())
                        for same_doc_id in same_doc_units:
                            if same_doc_id not in active_unit_ids:
                                active_unit_ids.add(same_doc_id)
                                next_frontier.add(same_doc_id)
        if not next_frontier:
            break
        frontier = next_frontier

    active_facts = [units_by_id[unit_id] for unit_id in sorted(active_unit_ids) if unit_id in units_by_id]
    endpoint_to_unit_ids, endpoint_labels, endpoint_doc_degrees = build_endpoint_index(active_facts)
    units_by_id = {str(unit.get("unit_id") or ""): unit for unit in active_facts}
    adjacency, adjacency_stats = build_endpoint_adjacency(
        units_by_id,
        endpoint_to_unit_ids,
        endpoint_doc_degrees,
        max_endpoint_degree=max_endpoint_degree,
    )

    closed_units_by_id: Dict[str, Dict[str, Any]] = {}
    raw_path_count = 0
    for source_id in sorted(seed_set):
        source_unit = units_by_id[source_id]
        paths = shortest_endpoint_paths_from(source_id, adjacency, max_path_edges=max_path_edges)
        for target_id in sorted(seed_set):
            if target_id == source_id or target_id not in paths:
                continue
            target_unit = units_by_id[target_id]
            if int(source_unit.get("doc_index", -1)) == int(target_unit.get("doc_index", -1)):
                continue
            path = paths[target_id]
            closed_unit = build_closed_unit(
                source_unit=source_unit,
                target_unit=target_unit,
                path_unit_ids=path["path_unit_ids"],
                path_endpoint_keys=path["path_endpoint_keys"],
                path_edges=path["path_edges"],
                units_by_id=units_by_id,
                endpoint_labels=endpoint_labels,
                max_path_edges=max_path_edges,
                max_endpoint_degree=max_endpoint_degree,
            )
            raw_path_count += 1
            canonical_id = closed_unit_canonical_id(closed_unit)
            closed_unit["closed_evidence_unit_id"] = canonical_id
            closed_unit["canonicalization"] = {
                "key": "seed_directed_doc_path_plus_endpoint_transfers",
                "preference": "fewest_endpoint_transfers_then_fewest_source_edges_then_deterministic_path",
            }
            existing = closed_units_by_id.get(canonical_id)
            if existing is None or closed_unit_preference_key(closed_unit) < closed_unit_preference_key(existing):
                closed_units_by_id[canonical_id] = closed_unit

    closed_units = [closed_units_by_id[key] for key in sorted(closed_units_by_id)]
    return {
        "summary": {
            "fact_unit_count": len(facts),
            "active_fact_unit_count": len(active_facts),
            "seed_unit_count": len(seed_set),
            "endpoint_count": len(endpoint_to_unit_ids),
            "endpoint_edge_count": int(adjacency_stats["endpoint_edge_count"]),
            "source_edge_count": int(adjacency_stats["source_edge_count"]),
            "raw_seed_path_count": raw_path_count,
            "closed_unit_count": len(closed_units),
        },
        "closed_units": closed_units,
    }


def closed_unit_rank_key(
    *,
    unit: Mapping[str, Any],
    query_tokens: Set[str],
    uncovered_tokens: Set[str],
    selected_docs: Set[int],
) -> Tuple[int, int, int, int, int, str]:
    signature = closed_unit_signature_tokens(unit)
    emitted_docs = unique_ints(unit.get("emitted_doc_indices", []) or [])
    new_docs = [doc for doc in emitted_docs if doc not in selected_docs]
    new_coverage = len(signature & uncovered_tokens)
    total_coverage = len(signature & query_tokens)
    return (
        -new_coverage,
        -total_coverage,
        len(new_docs),
        int(unit.get("path_edge_count", 0) or 0),
        int(unit.get("source_edge_count", 0) or 0),
        str(unit.get("closed_evidence_unit_id") or ""),
    )


def fill_rank_key(
    *,
    doc_index: int,
    openie_docs: Sequence[Mapping[str, Any]],
    query_tokens: Set[str],
    uncovered_tokens: Set[str],
    prior_rank: Mapping[int, int],
) -> Tuple[int, int, int, int]:
    signature = doc_signature_tokens(openie_docs, doc_index)
    return (
        -len(signature & uncovered_tokens),
        -len(signature & query_tokens),
        int(prior_rank.get(int(doc_index), 10**9)),
        int(doc_index),
    )


def select_obligation_closed_evidence(
    *,
    query: str,
    candidate_doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
    evidence_set_size: int = 5,
    max_path_edges: int = 3,
    max_endpoint_degree: int = 30,
    doc_units_by_index: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> Dict[str, Any]:
    query_tokens = content_tokens(query) - QUERY_FUNCTION_TOKENS
    candidate_docs = unique_ints(candidate_doc_indices)
    prior_rank = {doc_index: rank for rank, doc_index in enumerate(candidate_docs)}
    candidate_units = build_candidate_sto_units(
        candidate_doc_indices=candidate_docs,
        openie_docs=openie_docs,
        doc_units_by_index=doc_units_by_index,
    )
    frontier_tokens = frontier_query_tokens(query_tokens=query_tokens, candidate_units=candidate_units)
    seed_unit_ids = seed_unit_ids_for_query(
        query_tokens=frontier_tokens,
        candidate_units=candidate_units,
    )
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

    selected_docs: List[int] = []
    selected_doc_set: Set[int] = set()
    selected_units: List[Dict[str, Any]] = []
    uncovered_tokens = set(query_tokens)

    while len(selected_docs) < evidence_set_size:
        eligible_units = []
        for unit in closed_units:
            emitted_docs = unique_ints(unit.get("emitted_doc_indices", []) or [])
            new_docs = [doc for doc in emitted_docs if doc not in selected_doc_set]
            if not new_docs or len(selected_docs) + len(new_docs) > evidence_set_size:
                continue
            signature = closed_unit_signature_tokens(unit)
            if not (signature & query_tokens):
                continue
            eligible_units.append(unit)
        if not eligible_units:
            break
        best = min(
            eligible_units,
            key=lambda unit: closed_unit_rank_key(
                unit=unit,
                query_tokens=query_tokens,
                uncovered_tokens=uncovered_tokens,
                selected_docs=selected_doc_set,
            ),
        )
        best_signature = closed_unit_signature_tokens(best)
        if not (best_signature & uncovered_tokens) and selected_units:
            break
        emitted_docs = unique_ints(best.get("emitted_doc_indices", []) or [])
        new_docs = [doc for doc in emitted_docs if doc not in selected_doc_set]
        if not new_docs:
            break
        for doc in new_docs:
            selected_docs.append(doc)
            selected_doc_set.add(doc)
        uncovered_tokens -= best_signature
        selected_units.append(
            {
                "closed_evidence_unit_id": best.get("closed_evidence_unit_id"),
                "emitted_doc_indices": emitted_docs,
                "path_doc_indices": best.get("path_doc_indices", []),
                "path_endpoint_keys": best.get("path_endpoint_keys", []),
                "path_titles": best.get("path_titles", []),
                "path_edge_count": best.get("path_edge_count"),
                "source_edge_count": best.get("source_edge_count"),
                "query_token_coverage": sorted(best_signature & query_tokens),
            }
        )

    remaining_docs = [doc for doc in candidate_docs if doc not in selected_doc_set]
    for doc in sorted(
        remaining_docs,
        key=lambda doc_index: fill_rank_key(
            doc_index=doc_index,
            openie_docs=openie_docs,
            query_tokens=query_tokens,
            uncovered_tokens=uncovered_tokens,
            prior_rank=prior_rank,
        ),
    ):
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
        "uncovered_query_tokens": sorted(uncovered_tokens),
        "candidate_doc_count": len(candidate_docs),
        "seed_unit_count": len(seed_unit_ids),
        "candidate_closed_unit_count": len(closed_units),
        "closed_store_summary": store.get("summary", {}),
        "selection_policy": "greedy_query_token_coverage_over_closed_units_then_singleton_fill",
    }


def evaluate_dataset(
    *,
    dataset_payload: Mapping[str, Any],
    max_queries: int,
    evidence_set_size: int,
    max_path_edges: int,
    max_endpoint_degree: int,
) -> Dict[str, Any]:
    openie_docs = list(load_json(Path(str(dataset_payload["openie_path"]))).get("docs", []) or [])
    rows = list(dataset_payload.get("rows", []) or [])
    if max_queries > 0:
        rows = rows[:max_queries]
    needed_doc_indices: Set[int] = set()
    for row in rows:
        needed_doc_indices.update(candidate_doc_indices_from_row(row))
    doc_units_by_index = {
        doc_index: build_units_for_doc(openie_docs[doc_index], doc_index=doc_index, include_source_spans=False)
        for doc_index in sorted(needed_doc_indices)
        if 0 <= int(doc_index) < len(openie_docs)
    }

    output_rows: List[Dict[str, Any]] = []
    sums: Counter[str] = Counter()
    candidate_counts: List[int] = []
    closed_unit_counts: List[int] = []
    changed_count = 0
    improved_gold_count = 0
    regressed_gold_count = 0

    for row in rows:
        query = str(row.get("question") or row.get("query") or "")
        gold_doc_indices = unique_ints(row.get("gold_doc_indices", []) or [])
        candidate_doc_indices = candidate_doc_indices_from_row(row)
        result = select_obligation_closed_evidence(
            query=query,
            candidate_doc_indices=candidate_doc_indices,
            openie_docs=openie_docs,
            evidence_set_size=evidence_set_size,
            max_path_edges=max_path_edges,
            max_endpoint_degree=max_endpoint_degree,
            doc_units_by_index=doc_units_by_index,
        )
        retrieved = unique_ints(result.get("retrieved_doc_indices", []) or [])
        source_top5 = row_doc_indices(row, "anchor_guided_evidence_doc_indices_top5", "retrieved_doc_indices_top5", limit=5)
        old_gold_count = len(set(source_top5[:5]) & set(gold_doc_indices))
        new_gold_count = len(set(retrieved[:5]) & set(gold_doc_indices))
        if retrieved[:5] != source_top5[:5]:
            changed_count += 1
        if new_gold_count > old_gold_count:
            improved_gold_count += 1
        elif new_gold_count < old_gold_count:
            regressed_gold_count += 1

        r5 = recall_at_k(gold_doc_indices, retrieved, 5)
        r10 = recall_at_k(gold_doc_indices, retrieved, 10)
        sums["r5"] += r5
        sums["r10"] += r10
        sums["all_gold_at5"] += 1.0 if all_gold_at_k(gold_doc_indices, retrieved, 5) else 0.0
        sums["all_gold_at10"] += 1.0 if all_gold_at_k(gold_doc_indices, retrieved, 10) else 0.0
        candidate_counts.append(int(result.get("candidate_doc_count", 0) or 0))
        closed_unit_counts.append(int(result.get("candidate_closed_unit_count", 0) or 0))

        out = dict(row)
        out["obligation_closed_sto_doc_indices_top5"] = retrieved[:5]
        out["obligation_closed_sto_doc_indices_top10"] = retrieved[:10]
        out["obligation_closed_sto_recall_at5"] = round(float(r5), 6)
        out["obligation_closed_sto_recall_at10"] = round(float(r10), 6)
        out["obligation_closed_sto_all_gold_at5"] = bool(all_gold_at_k(gold_doc_indices, retrieved, 5))
        out["obligation_closed_sto_all_gold_at10"] = bool(all_gold_at_k(gold_doc_indices, retrieved, 10))
        out["obligation_closed_sto_selector"] = result
        out["obligation_closed_sto_gold_count_delta_vs_source_top5"] = new_gold_count - old_gold_count
        output_rows.append(out)

    denom = max(len(output_rows), 1)
    metrics = {
        "obligation_closed_sto_r5": round(sums["r5"] / float(denom), 6),
        "obligation_closed_sto_r10": round(sums["r10"] / float(denom), 6),
        "obligation_closed_sto_all_gold_at5": round(sums["all_gold_at5"] / float(denom), 6),
        "obligation_closed_sto_all_gold_at10": round(sums["all_gold_at10"] / float(denom), 6),
        "mean_candidate_doc_count": round(sum(candidate_counts) / float(max(len(candidate_counts), 1)), 6),
        "mean_candidate_closed_unit_count": round(sum(closed_unit_counts) / float(max(len(closed_unit_counts), 1)), 6),
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
        "# Obligation-Closed STO Selector",
        "",
        "This is a diagnostic selector over existing candidate documents. It selects construction-time closed STO units by greedy query-token coverage under the reader budget.",
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
            "| dataset | rows | R@5 | all-gold@5 | mean closed units | changed | gains | losses |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in summary.get("datasets", []) or []:
        metrics = dataset.get("metrics", {}) or {}
        lines.append(
            f"| {dataset.get('dataset')} | {len(dataset.get('rows', []) or [])} | "
            f"{metrics.get('obligation_closed_sto_r5', 0.0):.4f} | "
            f"{metrics.get('obligation_closed_sto_all_gold_at5', 0.0):.4f} | "
            f"{metrics.get('mean_candidate_closed_unit_count', 0.0):.1f} | "
            f"{metrics.get('changed_top5_queries', 0)} | "
            f"{metrics.get('gold_count_improved_queries', 0)} | "
            f"{metrics.get('gold_count_regressed_queries', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Method Boundary",
            "",
            "- Uses existing candidate documents from the source report, so this is not yet a standalone retriever.",
            "- Uses no gold labels, no QA outcomes, no SFB outputs, no PPR scores, no answer-type rules, and no weighted fusion.",
            "- The next standalone method must replace the candidate universe with a clean closed-unit activation step.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selector_metric_view(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "obligation_closed_sto_r5",
        "obligation_closed_sto_r10",
        "obligation_closed_sto_all_gold_at5",
        "obligation_closed_sto_all_gold_at10",
        "mean_candidate_doc_count",
        "mean_candidate_closed_unit_count",
        "changed_top5_queries",
        "gold_count_improved_queries",
        "gold_count_regressed_queries",
    ]
    return {key: metrics.get(key) for key in keys}


def evaluate_obligation_closed_sto_selector(
    *,
    report_path: Path,
    max_queries: int,
    evidence_set_size: int,
    max_path_edges: int,
    max_endpoint_degree: int,
    output_json_path: Path,
    output_md_path: Path,
) -> Dict[str, Any]:
    report = load_json(report_path)
    datasets = [
        evaluate_dataset(
            dataset_payload=dataset,
            max_queries=max_queries,
            evidence_set_size=evidence_set_size,
            max_path_edges=max_path_edges,
            max_endpoint_degree=max_endpoint_degree,
        )
        for dataset in report.get("datasets", []) or []
    ]
    summary = {
        "source_report_path": str(report_path),
        "method": "obligation_closed_sto_selector_diagnostic",
        "config": {
            "max_queries": int(max_queries),
            "evidence_set_size": int(evidence_set_size),
            "max_path_edges": int(max_path_edges),
            "max_endpoint_degree": int(max_endpoint_degree),
        },
        "datasets": datasets,
    }
    output_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, output_md_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate closed-unit STO selector over existing candidate documents.")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--evidence-set-size", type=int, default=5)
    parser.add_argument("--max-path-edges", type=int, default=3)
    parser.add_argument("--max-endpoint-degree", type=int, default=30)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = evaluate_obligation_closed_sto_selector(
        report_path=Path(args.report).resolve(),
        max_queries=max(int(args.max_queries), 0),
        evidence_set_size=max(int(args.evidence_set_size), 1),
        max_path_edges=max(int(args.max_path_edges), 1),
        max_endpoint_degree=max(int(args.max_endpoint_degree), 2),
        output_json_path=Path(args.output_json).resolve(),
        output_md_path=Path(args.output_md).resolve(),
    )
    compact = {
        dataset["dataset"]: selector_metric_view(dataset.get("metrics", {}))
        for dataset in summary.get("datasets", []) or []
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
