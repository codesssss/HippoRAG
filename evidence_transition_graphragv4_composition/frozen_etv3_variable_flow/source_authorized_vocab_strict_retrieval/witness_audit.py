"""Offline feasibility audit for source-warranted witness subgraphs.

This audit uses gold support labels only to measure whether the current OpenIE
substrate can support a witness-style method.  It is not a retriever.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .candidate_expansion import (
    SourceTextCandidateExpansionIndex,
    build_source_text_candidate_expansion_index,
    expand_source_text_candidates,
)
from .certificate_graph import EvidenceNode
from .evaluate_report import (
    DEFAULT_CANDIDATE_FIELDS,
    candidate_doc_indices_from_row,
    load_candidate_cache_doc_indices,
    load_json,
    load_nodes,
    rows_for_variant,
    unique_ints,
)
from .frontier import build_source_text_frontier
from .certificate_graph import build_source_text_certificate_graph
from .witness_atoms import (
    WITNESS_ATOM_CONTRACT,
    SourceWarrantedAtom,
    atom_entities_for_doc,
    source_warranted_atoms_from_node,
    title_entities_for_doc,
)


WITNESS_AUDIT_CONTRACT: Mapping[str, bool | str] = {
    "audit": "offline_source_warranted_witness_feasibility",
    "uses_gold_support_labels_for_audit_only": True,
    "uses_llm_generated_propositions": False,
    "uses_role_vocabulary": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "is_retriever": False,
}


@dataclass(frozen=True)
class WitnessDocPath:
    doc_indices: Tuple[int, ...]
    connector_entities: Tuple[str, ...]
    connector_support_counts: Tuple[int, ...]
    connector_binding_types: Tuple[str, ...]

    @property
    def max_connector_support_count(self) -> int:
        if not self.connector_support_counts:
            return 0
        return max(int(count) for count in self.connector_support_counts)


@dataclass(frozen=True)
class WitnessLink:
    neighbor_doc_index: int
    connector_entity: str
    connector_support_count: int
    binding_type: str


def audit_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    nodes: Sequence[EvidenceNode],
    candidate_fields: Sequence[str] = DEFAULT_CANDIDATE_FIELDS,
    candidate_field_mode: str = "first",
    top_k: int = 5,
    candidate_pool_k: int = 200,
    candidate_expansion_index: SourceTextCandidateExpansionIndex | None = None,
    entry_source_policy: str = "initial_topk",
    link_policy: str = "entity_overlap",
) -> Dict[str, Any]:
    """Audit whether missing gold supports are reachable by warranted atoms."""

    top_k = max(int(top_k), 1)
    candidate_pool_k = max(int(candidate_pool_k), top_k)
    if candidate_expansion_index is None:
        candidate_expansion_index = build_source_text_candidate_expansion_index(nodes)
    node_lookup = {int(node.doc_index): node for node in nodes}

    rows_out: List[Dict[str, Any]] = []
    totals: Dict[str, int] = {
        "queries": 0,
        "queries_with_missing_gold_in_pool": 0,
        "queries_with_witness_reachable_missing_gold": 0,
        "queries_with_budget_feasible_missing_gold": 0,
        "missing_gold_in_pool_docs": 0,
        "missing_gold_with_warranted_atom_docs": 0,
        "witness_reachable_missing_gold_docs": 0,
        "budget_feasible_missing_gold_docs": 0,
        "necessary_but_no_warranted_atom_docs": 0,
        "candidate_warranted_atom_count": 0,
        "candidate_docs_with_warranted_atoms": 0,
        "witness_path_doc_count_sum": 0,
        "witness_path_doc_count_observations": 0,
        "self_start_witness_docs": 0,
        "query_mentioned_missing_gold_docs": 0,
        "witness_connector_support_count_sum": 0,
        "witness_connector_support_count_observations": 0,
        "witness_max_connector_support_count_sum": 0,
        "witness_link_count": 0,
        "witness_entity_overlap_link_count": 0,
        "witness_target_title_link_count": 0,
        "witness_target_atom_link_count": 0,
    }

    for fallback_idx, row in enumerate(rows):
        query = str(row.get("question") or row.get("query") or "")
        query_index = int(row.get("query_index", fallback_idx))
        gold = set(unique_ints(row.get("gold_doc_indices", []) or []))
        candidates = candidate_doc_indices_from_row(
            row,
            candidate_fields,
            field_mode=candidate_field_mode,
        )
        initial_pool = tuple(unique_ints(candidates))[:candidate_pool_k]
        candidate_expansion = expand_source_text_candidates(
            query=query,
            nodes=nodes,
            candidate_doc_indices=initial_pool,
            top_k=top_k,
            expansion_index=candidate_expansion_index,
        )
        pool = tuple(candidate_expansion.candidate_doc_indices)
        initial_topk = tuple(pool[:top_k])
        graph = build_source_text_certificate_graph(
            query=query,
            nodes=nodes,
            candidate_doc_indices=pool,
        )
        frontier = build_source_text_frontier(
            query=query,
            nodes=nodes,
            candidate_doc_indices=pool,
            graph=graph,
            top_k=top_k,
        )
        entry_doc_indices = _entry_doc_indices(
            policy=entry_source_policy,
            initial_topk=initial_topk,
            frontier_query_mentions=frontier.query_mentioned_doc_indices,
        )

        atoms_by_doc = _atoms_by_doc(pool, node_lookup)
        entities_by_doc = {
            int(doc_index): atom_entities_for_doc(
                node=node_lookup[int(doc_index)],
                atoms=atoms_by_doc.get(int(doc_index), ()),
            )
            for doc_index in pool
            if int(doc_index) in node_lookup
        }
        title_entities_by_doc = {
            int(doc_index): title_entities_for_doc(node_lookup[int(doc_index)])
            for doc_index in pool
            if int(doc_index) in node_lookup
        }
        entity_to_docs = _entity_to_docs(entities_by_doc)
        title_entity_to_docs = _entity_to_docs(title_entities_by_doc)
        missing_gold_in_pool = set(pool) & gold - set(initial_topk)
        query_mentioned_missing_gold = missing_gold_in_pool & set(
            frontier.query_mentioned_doc_indices
        )

        reachable_gold = set()
        budget_feasible_gold = set()
        witness_paths: Dict[int, WitnessDocPath] = {}
        for gold_doc_index in sorted(missing_gold_in_pool):
            if not atoms_by_doc.get(int(gold_doc_index)):
                continue
            path = _shortest_witness_doc_path(
                start_doc_indices=entry_doc_indices,
                target_doc_index=int(gold_doc_index),
                link_policy=link_policy,
                atoms_by_doc=atoms_by_doc,
                entities_by_doc=entities_by_doc,
                title_entities_by_doc=title_entities_by_doc,
                entity_to_docs=entity_to_docs,
                title_entity_to_docs=title_entity_to_docs,
            )
            if not path:
                continue
            reachable_gold.add(int(gold_doc_index))
            witness_paths[int(gold_doc_index)] = path
            totals["witness_path_doc_count_sum"] += len(path.doc_indices)
            totals["witness_path_doc_count_observations"] += 1
            totals["self_start_witness_docs"] += int(len(path.doc_indices) == 1)
            totals["witness_connector_support_count_sum"] += sum(
                int(count) for count in path.connector_support_counts
            )
            totals["witness_connector_support_count_observations"] += len(
                path.connector_support_counts
            )
            totals["witness_max_connector_support_count_sum"] += int(
                path.max_connector_support_count
            )
            totals["witness_link_count"] += len(path.connector_binding_types)
            for binding_type in path.connector_binding_types:
                if binding_type == "entity_overlap":
                    totals["witness_entity_overlap_link_count"] += 1
                if binding_type == "target_title":
                    totals["witness_target_title_link_count"] += 1
                if binding_type == "target_atom":
                    totals["witness_target_atom_link_count"] += 1
            if len(path.doc_indices) <= top_k:
                budget_feasible_gold.add(int(gold_doc_index))

        missing_with_atoms = {
            int(doc_index)
            for doc_index in missing_gold_in_pool
            if atoms_by_doc.get(int(doc_index))
        }
        necessary_but_no_atom = set(missing_gold_in_pool) - missing_with_atoms
        warranted_atom_count = sum(len(atoms) for atoms in atoms_by_doc.values())
        docs_with_atoms = sum(1 for atoms in atoms_by_doc.values() if atoms)

        totals["queries"] += 1
        totals["queries_with_missing_gold_in_pool"] += int(bool(missing_gold_in_pool))
        totals["queries_with_witness_reachable_missing_gold"] += int(bool(reachable_gold))
        totals["queries_with_budget_feasible_missing_gold"] += int(bool(budget_feasible_gold))
        totals["missing_gold_in_pool_docs"] += len(missing_gold_in_pool)
        totals["query_mentioned_missing_gold_docs"] += len(query_mentioned_missing_gold)
        totals["missing_gold_with_warranted_atom_docs"] += len(missing_with_atoms)
        totals["witness_reachable_missing_gold_docs"] += len(reachable_gold)
        totals["budget_feasible_missing_gold_docs"] += len(budget_feasible_gold)
        totals["necessary_but_no_warranted_atom_docs"] += len(necessary_but_no_atom)
        totals["candidate_warranted_atom_count"] += warranted_atom_count
        totals["candidate_docs_with_warranted_atoms"] += docs_with_atoms

        rows_out.append(
            {
                "query_index": query_index,
                "question": query,
                "gold_doc_indices": tuple(sorted(gold)),
                "candidate_doc_count": len(pool),
                "initial_topk_doc_indices": initial_topk,
                "entry_doc_indices": entry_doc_indices,
                "missing_gold_in_pool_doc_indices": tuple(sorted(missing_gold_in_pool)),
                "query_mentioned_missing_gold_doc_indices": tuple(
                    sorted(query_mentioned_missing_gold)
                ),
                "missing_gold_with_warranted_atom_doc_indices": tuple(sorted(missing_with_atoms)),
                "witness_reachable_missing_gold_doc_indices": tuple(sorted(reachable_gold)),
                "budget_feasible_missing_gold_doc_indices": tuple(sorted(budget_feasible_gold)),
                "necessary_but_no_warranted_atom_doc_indices": tuple(sorted(necessary_but_no_atom)),
                "candidate_warranted_atom_count": warranted_atom_count,
                "candidate_docs_with_warranted_atoms": docs_with_atoms,
                "witness_paths_by_gold_doc": {
                    str(doc_index): path.doc_indices
                    for doc_index, path in sorted(witness_paths.items())
                },
                "witness_path_details_by_gold_doc": {
                    str(doc_index): _witness_path_detail(path)
                    for doc_index, path in sorted(witness_paths.items())
                },
                "witness_atom_preview": _atom_preview(atoms_by_doc, limit=6),
            }
        )

    return {
        **dict(WITNESS_AUDIT_CONTRACT),
        **dict(WITNESS_ATOM_CONTRACT),
        "row_count": len(rows_out),
        "entry_source_policy": str(entry_source_policy),
        "link_policy": str(link_policy),
        "metrics": _metrics_from_totals(totals),
        "totals": totals,
        "rows": rows_out,
    }


def _entry_doc_indices(
    *,
    policy: str,
    initial_topk: Sequence[int],
    frontier_query_mentions: Sequence[int],
) -> Tuple[int, ...]:
    normalized = str(policy or "initial_topk").strip().lower()
    if normalized == "anchor_or_top1":
        return tuple(unique_ints(frontier_query_mentions or initial_topk[:1]))
    return tuple(unique_ints([*initial_topk, *frontier_query_mentions]))


def _atoms_by_doc(
    candidate_doc_indices: Sequence[int],
    node_lookup: Mapping[int, EvidenceNode],
) -> Dict[int, Tuple[SourceWarrantedAtom, ...]]:
    return {
        int(doc_index): source_warranted_atoms_from_node(node_lookup[int(doc_index)])
        for doc_index in candidate_doc_indices
        if int(doc_index) in node_lookup
    }


def _entity_to_docs(
    entities_by_doc: Mapping[int, Sequence[str]],
) -> Dict[str, Tuple[int, ...]]:
    mutable: Dict[str, List[int]] = {}
    for doc_index, entities in entities_by_doc.items():
        for entity in entities:
            if not _is_specific_entity(entity):
                continue
            mutable.setdefault(str(entity), []).append(int(doc_index))
    return {
        entity: tuple(unique_ints(doc_indices))
        for entity, doc_indices in mutable.items()
    }


def _shortest_witness_doc_path(
    *,
    start_doc_indices: Sequence[int],
    target_doc_index: int,
    link_policy: str,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    entities_by_doc: Mapping[int, Sequence[str]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
    entity_to_docs: Mapping[str, Sequence[int]],
    title_entity_to_docs: Mapping[str, Sequence[int]],
) -> WitnessDocPath | None:
    target = int(target_doc_index)
    starts = tuple(
        int(doc_index)
        for doc_index in unique_ints(start_doc_indices)
        if int(doc_index) in entities_by_doc
    )
    if target in starts:
        return WitnessDocPath(
            doc_indices=(target,),
            connector_entities=(),
            connector_support_counts=(),
            connector_binding_types=(),
        )
    queue = []
    best_by_doc: Dict[int, Tuple[int, int, int]] = {}
    for start in starts:
        start_path = (int(start),)
        heapq.heappush(
            queue,
            (
                1,
                0,
                0,
                start_path,
                (),
                (),
                (),
                int(start),
            ),
        )
        best_by_doc[int(start)] = (1, 0, 0)
    while queue:
        (
            path_doc_count,
            max_connector_support_count,
            connector_support_sum,
            path,
            connector_entities,
            connector_support_counts,
            connector_binding_types,
            doc_index,
        ) = heapq.heappop(queue)
        best_key = best_by_doc.get(int(doc_index))
        current_key = (
            int(path_doc_count),
            int(max_connector_support_count),
            int(connector_support_sum),
        )
        if best_key is not None and current_key > best_key:
            continue
        if int(doc_index) == target:
            return WitnessDocPath(
                doc_indices=path,
                connector_entities=connector_entities,
                connector_support_counts=connector_support_counts,
                connector_binding_types=connector_binding_types,
            )
        for link in _witness_neighbors(
            doc_index=doc_index,
            link_policy=link_policy,
            atoms_by_doc=atoms_by_doc,
            entities_by_doc=entities_by_doc,
            title_entities_by_doc=title_entities_by_doc,
            entity_to_docs=entity_to_docs,
            title_entity_to_docs=title_entity_to_docs,
        ):
            neighbor = int(link.neighbor_doc_index)
            if int(neighbor) in path:
                continue
            next_path = (*path, int(neighbor))
            next_connector_entities = (*connector_entities, str(link.connector_entity))
            next_connector_support_counts = (
                *connector_support_counts,
                int(link.connector_support_count),
            )
            next_connector_binding_types = (
                *connector_binding_types,
                str(link.binding_type),
            )
            next_path_doc_count = len(next_path)
            next_max_connector_support_count = max(
                int(max_connector_support_count),
                int(link.connector_support_count),
            )
            next_connector_support_sum = int(connector_support_sum) + int(
                link.connector_support_count
            )
            next_key = (
                int(next_path_doc_count),
                int(next_max_connector_support_count),
                int(next_connector_support_sum),
            )
            previous_key = best_by_doc.get(int(neighbor))
            if previous_key is not None and next_key >= previous_key:
                continue
            best_by_doc[int(neighbor)] = next_key
            heapq.heappush(
                queue,
                (
                    int(next_path_doc_count),
                    int(next_max_connector_support_count),
                    int(next_connector_support_sum),
                    next_path,
                    next_connector_entities,
                    next_connector_support_counts,
                    next_connector_binding_types,
                    int(neighbor),
                )
            )
    return None


def _witness_neighbors(
    *,
    doc_index: int,
    link_policy: str,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    entities_by_doc: Mapping[int, Sequence[str]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
    entity_to_docs: Mapping[str, Sequence[int]],
    title_entity_to_docs: Mapping[str, Sequence[int]],
) -> Tuple[WitnessLink, ...]:
    neighbors: List[WitnessLink] = []
    seen = set()
    normalized_policy = str(link_policy or "entity_overlap").strip().lower()
    if normalized_policy in {
        "title_bound",
        "target_atom_bound",
        "anchored_target_atom_bound",
    }:
        source_entities = _source_endpoint_entities_for_title_bound_link(
            atoms_by_doc.get(int(doc_index), ())
        )
    else:
        source_entities = tuple(entities_by_doc.get(int(doc_index), ()))
    for entity in source_entities:
        if not _is_specific_entity(entity):
            continue
        support_docs = tuple(entity_to_docs.get(str(entity), ()))
        support_count = len(support_docs)
        for neighbor in support_docs:
            if int(neighbor) == int(doc_index):
                continue
            binding_type = _witness_link_binding_type(
                entity=entity,
                doc_index=int(neighbor),
                link_policy=normalized_policy,
                atoms_by_doc=atoms_by_doc,
                title_entities_by_doc=title_entities_by_doc,
                title_entity_to_docs=title_entity_to_docs,
            )
            if not binding_type:
                continue
            key = (int(neighbor), str(entity), str(binding_type))
            if key in seen:
                continue
            seen.add(key)
            neighbors.append(
                WitnessLink(
                    neighbor_doc_index=int(neighbor),
                    connector_entity=str(entity),
                    connector_support_count=int(support_count),
                    binding_type=str(binding_type),
                )
            )
    return tuple(
        sorted(
            neighbors,
            key=lambda link: (
                int(link.connector_support_count),
                int(link.neighbor_doc_index),
                str(link.connector_entity),
                str(link.binding_type),
            ),
        )
    )


def _source_endpoint_entities_for_title_bound_link(
    atoms: Sequence[SourceWarrantedAtom],
) -> Tuple[str, ...]:
    entities = []
    for atom in atoms:
        for entity in atom.endpoint_entities:
            if entity and entity not in entities:
                entities.append(entity)
    return tuple(entities)


def _entity_binds_doc_title(
    *,
    entity: str,
    doc_index: int,
    title_entities_by_doc: Mapping[int, Sequence[str]],
) -> bool:
    entity_text = str(entity or "")
    return entity_text in set(str(item) for item in title_entities_by_doc.get(int(doc_index), ()))


def _witness_link_binding_type(
    *,
    entity: str,
    doc_index: int,
    link_policy: str,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
    title_entity_to_docs: Mapping[str, Sequence[int]],
) -> str:
    normalized_policy = str(link_policy or "entity_overlap").strip().lower()
    if normalized_policy == "entity_overlap":
        return "entity_overlap"
    if _entity_binds_doc_title(
        entity=entity,
        doc_index=int(doc_index),
        title_entities_by_doc=title_entities_by_doc,
    ):
        return "target_title"
    if normalized_policy in {
        "target_atom_bound",
        "anchored_target_atom_bound",
    } and _entity_binds_target_atom(
        entity=entity,
        doc_index=int(doc_index),
        atoms_by_doc=atoms_by_doc,
        title_entities_by_doc=title_entities_by_doc,
    ):
        if normalized_policy == "anchored_target_atom_bound" and not title_entity_to_docs.get(
            str(entity or "")
        ):
            return ""
        return "target_atom"
    return ""


def _entity_binds_target_atom(
    *,
    entity: str,
    doc_index: int,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
) -> bool:
    entity_text = str(entity or "")
    title_entities = set(str(item) for item in title_entities_by_doc.get(int(doc_index), ()))
    if not entity_text or not title_entities:
        return False
    for atom in atoms_by_doc.get(int(doc_index), ()):
        endpoints = set(str(item) for item in atom.endpoint_entities)
        if entity_text in endpoints and endpoints & title_entities:
            return True
    return False


def _is_specific_entity(entity: object) -> bool:
    item = str(entity or "")
    entity_tokens = item.split()
    if len(entity_tokens) >= 2:
        return True
    return bool(entity_tokens and len(entity_tokens[0]) >= 5 and not entity_tokens[0].isdigit())


def _metrics_from_totals(totals: Mapping[str, int]) -> Dict[str, float]:
    return {
        "missing_gold_atom_coverage": _safe_div(
            totals.get("missing_gold_with_warranted_atom_docs", 0),
            totals.get("missing_gold_in_pool_docs", 0),
        ),
        "witness_reachable_missing_gold_coverage": _safe_div(
            totals.get("witness_reachable_missing_gold_docs", 0),
            totals.get("missing_gold_in_pool_docs", 0),
        ),
        "budget_feasible_missing_gold_coverage": _safe_div(
            totals.get("budget_feasible_missing_gold_docs", 0),
            totals.get("missing_gold_in_pool_docs", 0),
        ),
        "query_witness_reachable_rate": _safe_div(
            totals.get("queries_with_witness_reachable_missing_gold", 0),
            totals.get("queries_with_missing_gold_in_pool", 0),
        ),
        "query_budget_feasible_rate": _safe_div(
            totals.get("queries_with_budget_feasible_missing_gold", 0),
            totals.get("queries_with_missing_gold_in_pool", 0),
        ),
        "necessary_but_no_warranted_atom_rate": _safe_div(
            totals.get("necessary_but_no_warranted_atom_docs", 0),
            totals.get("missing_gold_in_pool_docs", 0),
        ),
        "self_start_witness_rate": _safe_div(
            totals.get("self_start_witness_docs", 0),
            totals.get("witness_reachable_missing_gold_docs", 0),
        ),
        "query_mentioned_missing_gold_rate": _safe_div(
            totals.get("query_mentioned_missing_gold_docs", 0),
            totals.get("missing_gold_in_pool_docs", 0),
        ),
        "mean_warranted_atom_count": _safe_div(
            totals.get("candidate_warranted_atom_count", 0),
            totals.get("queries", 0),
        ),
        "mean_docs_with_warranted_atoms": _safe_div(
            totals.get("candidate_docs_with_warranted_atoms", 0),
            totals.get("queries", 0),
        ),
        "mean_witness_path_doc_count": _safe_div(
            totals.get("witness_path_doc_count_sum", 0),
            totals.get("witness_path_doc_count_observations", 0),
        ),
        "mean_connector_support_count": _safe_div(
            totals.get("witness_connector_support_count_sum", 0),
            totals.get("witness_connector_support_count_observations", 0),
        ),
        "mean_max_connector_support_count": _safe_div(
            totals.get("witness_max_connector_support_count_sum", 0),
            totals.get("witness_path_doc_count_observations", 0),
        ),
        "entity_overlap_witness_link_fraction": _safe_div(
            totals.get("witness_entity_overlap_link_count", 0),
            totals.get("witness_link_count", 0),
        ),
        "target_title_witness_link_fraction": _safe_div(
            totals.get("witness_target_title_link_count", 0),
            totals.get("witness_link_count", 0),
        ),
        "target_atom_witness_link_fraction": _safe_div(
            totals.get("witness_target_atom_link_count", 0),
            totals.get("witness_link_count", 0),
        ),
    }


def _safe_div(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _atom_preview(
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    *,
    limit: int,
) -> Tuple[Mapping[str, object], ...]:
    output = []
    for doc_index in sorted(atoms_by_doc):
        for atom in atoms_by_doc[int(doc_index)]:
            output.append(
                {
                    "source_doc_index": int(atom.source_doc_index),
                    "source_title": str(atom.source_title),
                    "subject": str(atom.subject),
                    "predicate": str(atom.predicate),
                    "object": str(atom.object),
                    "source_sentence": str(atom.source_sentence),
                }
            )
            if len(output) >= max(int(limit), 0):
                return tuple(output)
    return tuple(output)


def _witness_path_detail(path: WitnessDocPath) -> Mapping[str, object]:
    return {
        "doc_indices": tuple(path.doc_indices),
        "connector_entities": tuple(path.connector_entities),
        "connector_support_counts": tuple(path.connector_support_counts),
        "connector_binding_types": tuple(path.connector_binding_types),
        "max_connector_support_count": int(path.max_connector_support_count),
    }


def audit_report(args: argparse.Namespace) -> Dict[str, Any]:
    report_path = Path(args.report).expanduser().resolve()
    report = load_json(report_path)
    openie_path = Path(args.openie_path or report.get("analysis_openie_path") or "").expanduser()
    if not openie_path:
        raise ValueError("--openie-path is required when report has no analysis_openie_path")
    if not openie_path.is_absolute():
        openie_path = (report_path.parent / openie_path).resolve()
    nodes = load_nodes(openie_path)
    rows = rows_for_variant(report, str(args.variant))
    candidate_cache_path_text = str(
        args.candidate_cache_path
        or (report.get("config", {}) or {}).get("baseline_retrieval_cache_path")
        or ""
    )
    candidate_cache_path = Path(candidate_cache_path_text).expanduser()
    candidate_cache_docs = (
        load_candidate_cache_doc_indices(candidate_cache_path)
        if candidate_cache_path_text
        else {}
    )
    if candidate_cache_docs:
        enriched_rows: List[Mapping[str, Any]] = []
        for fallback_idx, row in enumerate(rows):
            query_index = int(row.get("query_index", fallback_idx))
            copied = dict(row)
            copied["candidate_cache_doc_indices"] = candidate_cache_docs.get(query_index, [])
            enriched_rows.append(copied)
        rows = enriched_rows
    max_queries = max(int(args.max_queries), 0)
    if max_queries:
        rows = rows[:max_queries]
    candidate_fields = tuple(
        field.strip()
        for field in str(args.candidate_fields).split(",")
        if field.strip()
    )
    payload = audit_rows(
        rows=rows,
        nodes=nodes,
        candidate_fields=candidate_fields or DEFAULT_CANDIDATE_FIELDS,
        candidate_field_mode=str(args.candidate_field_mode),
        top_k=max(int(args.top_k), 1),
        candidate_pool_k=max(int(args.candidate_pool_k), 1),
        entry_source_policy=str(args.entry_source_policy),
        link_policy=str(args.link_policy),
    )
    return {
        "dataset": str(report.get("dataset") or args.dataset or ""),
        "input_report": str(report_path),
        "openie_path": str(openie_path),
        "source_variant": str(args.variant),
        "candidate_fields": list(candidate_fields or DEFAULT_CANDIDATE_FIELDS),
        "config": {
            "top_k": max(int(args.top_k), 1),
            "candidate_pool_k": max(int(args.candidate_pool_k), 1),
            "candidate_field_mode": str(args.candidate_field_mode),
            "candidate_cache_path": str(candidate_cache_path) if candidate_cache_path_text else "",
            "entry_source_policy": str(args.entry_source_policy),
            "link_policy": str(args.link_policy),
            "max_queries": max_queries,
        },
        **payload,
    }


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    metrics = payload.get("metrics", {}) or {}
    lines = [
        "# Source-Warranted Witness Feasibility Audit",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| rows | {payload.get('row_count', 0)} |",
        f"| source variant | {payload.get('source_variant', '')} |",
        f"| entry source policy | {payload.get('entry_source_policy', '')} |",
        f"| link policy | {payload.get('link_policy', '')} |",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| missing gold atom coverage | {float(metrics.get('missing_gold_atom_coverage', 0.0)):.4f} |",
        f"| witness reachable missing-gold coverage | {float(metrics.get('witness_reachable_missing_gold_coverage', 0.0)):.4f} |",
        f"| budget feasible missing-gold coverage | {float(metrics.get('budget_feasible_missing_gold_coverage', 0.0)):.4f} |",
        f"| query witness reachable rate | {float(metrics.get('query_witness_reachable_rate', 0.0)):.4f} |",
        f"| query budget feasible rate | {float(metrics.get('query_budget_feasible_rate', 0.0)):.4f} |",
        f"| necessary but no warranted atom rate | {float(metrics.get('necessary_but_no_warranted_atom_rate', 0.0)):.4f} |",
        f"| self-start witness rate | {float(metrics.get('self_start_witness_rate', 0.0)):.4f} |",
        f"| query-mentioned missing-gold rate | {float(metrics.get('query_mentioned_missing_gold_rate', 0.0)):.4f} |",
        f"| mean warranted atom count | {float(metrics.get('mean_warranted_atom_count', 0.0)):.2f} |",
        f"| mean witness path doc count | {float(metrics.get('mean_witness_path_doc_count', 0.0)):.2f} |",
        f"| mean connector support count | {float(metrics.get('mean_connector_support_count', 0.0)):.2f} |",
        f"| mean max connector support count | {float(metrics.get('mean_max_connector_support_count', 0.0)):.2f} |",
        f"| entity-overlap witness link fraction | {float(metrics.get('entity_overlap_witness_link_fraction', 0.0)):.4f} |",
        f"| target-title witness link fraction | {float(metrics.get('target_title_witness_link_fraction', 0.0)):.4f} |",
        f"| target-atom witness link fraction | {float(metrics.get('target_atom_witness_link_fraction', 0.0)):.4f} |",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--openie-path", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--variant", default="hipporag_v2")
    parser.add_argument("--candidate-fields", default=",".join(DEFAULT_CANDIDATE_FIELDS))
    parser.add_argument("--candidate-field-mode", choices=("first", "union"), default="first")
    parser.add_argument("--candidate-cache-path", default="")
    parser.add_argument(
        "--entry-source-policy",
        choices=("initial_topk", "anchor_or_top1"),
        default="initial_topk",
    )
    parser.add_argument(
        "--link-policy",
        choices=(
            "entity_overlap",
            "title_bound",
            "target_atom_bound",
            "anchored_target_atom_bound",
        ),
        default="entity_overlap",
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-k", type=int, default=200)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = audit_report(args)
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        write_markdown(payload, Path(args.output_md).expanduser())
    print(json.dumps({payload["dataset"]: payload["metrics"]}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
