"""Offline audit for source-warranted evidence transitions.

This audit measures whether a source-warranted navigation edge can be upgraded
to an evidence transition: the source side exposes a connector entity, and the
target side contains a source-warranted atom that exposes query relation surface.
Gold support labels are used only for feasibility measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from .query_conditioning import (
    QuerySignatures,
    extract_query_signatures,
    query_relation_overlap,
    token_stems_for_text,
)
from .witness_atoms import (
    WITNESS_ATOM_CONTRACT,
    SourceWarrantedAtom,
    atom_entities_for_doc,
    source_warranted_atoms_from_node,
    title_entities_for_doc,
)


TRANSITION_WITNESS_AUDIT_CONTRACT: Mapping[str, bool | str] = {
    "audit": "source_warranted_evidence_transition_feasibility",
    "transition_unit": "source_connector_to_target_query_atom",
    "uses_gold_support_labels_for_audit_only": True,
    "uses_llm_generated_propositions": False,
    "uses_role_vocabulary": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "is_retriever": False,
}


@dataclass(frozen=True)
class TransitionWitness:
    source_doc_index: int
    target_doc_index: int
    connector_entity: str
    binding_type: str
    source_atom_id: str
    target_atom_id: str
    target_query_overlap_tokens: Tuple[str, ...]
    target_signal_type: str


@dataclass(frozen=True)
class RelationAliasIndex:
    aliases_by_stem: Mapping[str, Tuple[str, ...]]
    trace: Mapping[str, int]


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
    link_policy: str = "anchored_target_atom_bound",
    relation_signal_policy: str = "surface",
) -> Dict[str, Any]:
    """Audit navigation edges that also have target-side query atom evidence."""

    top_k = max(int(top_k), 1)
    candidate_pool_k = max(int(candidate_pool_k), top_k)
    if candidate_expansion_index is None:
        candidate_expansion_index = build_source_text_candidate_expansion_index(nodes)
    node_lookup = {int(node.doc_index): node for node in nodes}

    rows_out: List[Dict[str, Any]] = []
    totals: Dict[str, int] = {
        "queries": 0,
        "queries_with_missing_gold_in_pool": 0,
        "queries_with_navigation_witness_missing_gold": 0,
        "queries_with_transition_witness_missing_gold": 0,
        "missing_gold_in_pool_docs": 0,
        "navigation_witness_missing_gold_docs": 0,
        "transition_witness_missing_gold_docs": 0,
        "necessary_but_no_navigation_witness_docs": 0,
        "navigation_target_docs": 0,
        "transition_target_docs": 0,
        "navigation_legal_but_unnecessary_docs": 0,
        "transition_legal_but_unnecessary_docs": 0,
        "transition_witness_count": 0,
        "transition_target_title_binding_count": 0,
        "transition_target_atom_binding_count": 0,
        "transition_predicate_signal_count": 0,
        "transition_sentence_signal_count": 0,
        "transition_endpoint_signal_count": 0,
        "transition_pool_alias_signal_count": 0,
        "relation_alias_pair_count": 0,
        "relation_alias_stem_count": 0,
        "baseline_query_atom_surface_covered_stem_count": 0,
        "query_relation_stem_count": 0,
        "queries_with_full_baseline_query_atom_surface_coverage": 0,
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
        signatures = extract_query_signatures(
            query=query,
            nodes=nodes,
            candidate_doc_indices=pool,
            query_mentioned_doc_indices=frontier.query_mentioned_doc_indices,
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
        relation_alias_index = _relation_alias_index(atoms_by_doc)
        transition_witnesses = _transition_witnesses(
            entry_doc_indices=entry_doc_indices,
            link_policy=link_policy,
            atoms_by_doc=atoms_by_doc,
            entities_by_doc=entities_by_doc,
            title_entities_by_doc=title_entities_by_doc,
            entity_to_docs=entity_to_docs,
            title_entity_to_docs=title_entity_to_docs,
            signatures=signatures,
            relation_alias_index=relation_alias_index,
            relation_signal_policy=relation_signal_policy,
        )
        navigation_targets = _navigation_targets(
            entry_doc_indices=entry_doc_indices,
            link_policy=link_policy,
            atoms_by_doc=atoms_by_doc,
            entities_by_doc=entities_by_doc,
            title_entities_by_doc=title_entities_by_doc,
            entity_to_docs=entity_to_docs,
            title_entity_to_docs=title_entity_to_docs,
        )
        transition_targets = {int(item.target_doc_index) for item in transition_witnesses}
        missing_gold_in_pool = set(pool) & gold - set(initial_topk)
        navigation_gold = navigation_targets & missing_gold_in_pool
        transition_gold = transition_targets & missing_gold_in_pool
        navigation_unnecessary = navigation_targets - gold
        transition_unnecessary = transition_targets - gold
        baseline_covered_stems = _covered_query_relation_stems(
            doc_indices=initial_topk,
            atoms_by_doc=atoms_by_doc,
            signatures=signatures,
        )
        query_relation_stems = tuple(signatures.relation_stems)
        full_baseline_surface_coverage = bool(query_relation_stems) and set(
            query_relation_stems
        ).issubset(set(baseline_covered_stems))

        totals["queries"] += 1
        totals["queries_with_missing_gold_in_pool"] += int(bool(missing_gold_in_pool))
        totals["queries_with_navigation_witness_missing_gold"] += int(bool(navigation_gold))
        totals["queries_with_transition_witness_missing_gold"] += int(bool(transition_gold))
        totals["missing_gold_in_pool_docs"] += len(missing_gold_in_pool)
        totals["navigation_witness_missing_gold_docs"] += len(navigation_gold)
        totals["transition_witness_missing_gold_docs"] += len(transition_gold)
        totals["necessary_but_no_navigation_witness_docs"] += len(
            missing_gold_in_pool - navigation_targets
        )
        totals["navigation_target_docs"] += len(navigation_targets)
        totals["transition_target_docs"] += len(transition_targets)
        totals["navigation_legal_but_unnecessary_docs"] += len(navigation_unnecessary)
        totals["transition_legal_but_unnecessary_docs"] += len(transition_unnecessary)
        totals["transition_witness_count"] += len(transition_witnesses)
        totals["transition_target_title_binding_count"] += sum(
            int(item.binding_type == "target_title") for item in transition_witnesses
        )
        totals["transition_target_atom_binding_count"] += sum(
            int(item.binding_type == "target_atom") for item in transition_witnesses
        )
        totals["transition_predicate_signal_count"] += sum(
            int(item.target_signal_type == "predicate") for item in transition_witnesses
        )
        totals["transition_sentence_signal_count"] += sum(
            int(item.target_signal_type == "sentence") for item in transition_witnesses
        )
        totals["transition_endpoint_signal_count"] += sum(
            int(item.target_signal_type == "endpoint") for item in transition_witnesses
        )
        totals["transition_pool_alias_signal_count"] += sum(
            int(item.target_signal_type.startswith("pool_alias_"))
            for item in transition_witnesses
        )
        totals["relation_alias_pair_count"] += int(
            relation_alias_index.trace.get("relation_alias_pair_count", 0)
        )
        totals["relation_alias_stem_count"] += int(
            relation_alias_index.trace.get("relation_alias_stem_count", 0)
        )
        totals["baseline_query_atom_surface_covered_stem_count"] += len(baseline_covered_stems)
        totals["query_relation_stem_count"] += len(query_relation_stems)
        totals["queries_with_full_baseline_query_atom_surface_coverage"] += int(
            full_baseline_surface_coverage
        )
        closure_prefix = "closed" if full_baseline_surface_coverage else "deficit"
        _accumulate_closure_totals(
            totals=totals,
            prefix=closure_prefix,
            missing_gold_in_pool=missing_gold_in_pool,
            navigation_gold=navigation_gold,
            transition_gold=transition_gold,
            navigation_targets=navigation_targets,
            transition_targets=transition_targets,
            navigation_unnecessary=navigation_unnecessary,
            transition_unnecessary=transition_unnecessary,
            transition_witnesses=transition_witnesses,
        )

        rows_out.append(
            {
                "query_index": query_index,
                "question": query,
                "gold_doc_indices": tuple(sorted(gold)),
                "initial_topk_doc_indices": initial_topk,
                "entry_doc_indices": entry_doc_indices,
                "candidate_doc_count": len(pool),
                "query_relation_stems": query_relation_stems,
                "expanded_query_relation_stems": _expanded_query_relation_stems(
                    signatures=signatures,
                    relation_alias_index=relation_alias_index,
                    relation_signal_policy=relation_signal_policy,
                ),
                "baseline_covered_query_relation_stems": baseline_covered_stems,
                "baseline_query_atom_closure_state": (
                    "closed" if full_baseline_surface_coverage else "deficit"
                ),
                "full_baseline_query_atom_surface_coverage": bool(
                    full_baseline_surface_coverage
                ),
                "missing_gold_in_pool_doc_indices": tuple(sorted(missing_gold_in_pool)),
                "navigation_witness_missing_gold_doc_indices": tuple(sorted(navigation_gold)),
                "transition_witness_missing_gold_doc_indices": tuple(sorted(transition_gold)),
                "necessary_but_no_navigation_witness_doc_indices": tuple(
                    sorted(missing_gold_in_pool - navigation_targets)
                ),
                "navigation_target_count": len(navigation_targets),
                "transition_target_count": len(transition_targets),
                "navigation_legal_but_unnecessary_count": len(navigation_unnecessary),
                "transition_legal_but_unnecessary_count": len(transition_unnecessary),
                "transition_witness_preview": _transition_witness_preview(
                    transition_witnesses
                ),
            }
        )

    return {
        **dict(TRANSITION_WITNESS_AUDIT_CONTRACT),
        **dict(WITNESS_ATOM_CONTRACT),
        "row_count": len(rows_out),
        "entry_source_policy": str(entry_source_policy),
        "link_policy": str(link_policy),
        "relation_signal_policy": str(relation_signal_policy),
        "metrics": _metrics_from_totals(totals),
        "totals": totals,
        "rows": rows_out,
    }


def _accumulate_closure_totals(
    *,
    totals: Dict[str, int],
    prefix: str,
    missing_gold_in_pool: set[int],
    navigation_gold: set[int],
    transition_gold: set[int],
    navigation_targets: set[int],
    transition_targets: set[int],
    navigation_unnecessary: set[int],
    transition_unnecessary: set[int],
    transition_witnesses: Sequence[TransitionWitness],
) -> None:
    totals[f"{prefix}_queries"] = totals.get(f"{prefix}_queries", 0) + 1
    totals[f"{prefix}_queries_with_missing_gold_in_pool"] = totals.get(
        f"{prefix}_queries_with_missing_gold_in_pool",
        0,
    ) + int(bool(missing_gold_in_pool))
    totals[f"{prefix}_queries_with_navigation_witness_missing_gold"] = totals.get(
        f"{prefix}_queries_with_navigation_witness_missing_gold",
        0,
    ) + int(bool(navigation_gold))
    totals[f"{prefix}_queries_with_transition_witness_missing_gold"] = totals.get(
        f"{prefix}_queries_with_transition_witness_missing_gold",
        0,
    ) + int(bool(transition_gold))
    totals[f"{prefix}_missing_gold_in_pool_docs"] = totals.get(
        f"{prefix}_missing_gold_in_pool_docs",
        0,
    ) + len(missing_gold_in_pool)
    totals[f"{prefix}_navigation_witness_missing_gold_docs"] = totals.get(
        f"{prefix}_navigation_witness_missing_gold_docs",
        0,
    ) + len(navigation_gold)
    totals[f"{prefix}_transition_witness_missing_gold_docs"] = totals.get(
        f"{prefix}_transition_witness_missing_gold_docs",
        0,
    ) + len(transition_gold)
    totals[f"{prefix}_navigation_target_docs"] = totals.get(
        f"{prefix}_navigation_target_docs",
        0,
    ) + len(navigation_targets)
    totals[f"{prefix}_transition_target_docs"] = totals.get(
        f"{prefix}_transition_target_docs",
        0,
    ) + len(transition_targets)
    totals[f"{prefix}_navigation_legal_but_unnecessary_docs"] = totals.get(
        f"{prefix}_navigation_legal_but_unnecessary_docs",
        0,
    ) + len(navigation_unnecessary)
    totals[f"{prefix}_transition_legal_but_unnecessary_docs"] = totals.get(
        f"{prefix}_transition_legal_but_unnecessary_docs",
        0,
    ) + len(transition_unnecessary)
    totals[f"{prefix}_transition_witness_count"] = totals.get(
        f"{prefix}_transition_witness_count",
        0,
    ) + len(transition_witnesses)


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


def _transition_witnesses(
    *,
    entry_doc_indices: Sequence[int],
    link_policy: str,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    entities_by_doc: Mapping[int, Sequence[str]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
    entity_to_docs: Mapping[str, Sequence[int]],
    title_entity_to_docs: Mapping[str, Sequence[int]],
    signatures: QuerySignatures,
    relation_alias_index: RelationAliasIndex,
    relation_signal_policy: str,
) -> Tuple[TransitionWitness, ...]:
    output: List[TransitionWitness] = []
    seen = set()
    for source_doc_index in unique_ints(entry_doc_indices):
        for source_atom in atoms_by_doc.get(int(source_doc_index), ()):
            for connector in source_atom.endpoint_entities:
                for target_doc_index, binding_type in _bound_targets_for_connector(
                    source_doc_index=int(source_doc_index),
                    connector=connector,
                    link_policy=link_policy,
                    atoms_by_doc=atoms_by_doc,
                    entities_by_doc=entities_by_doc,
                    title_entities_by_doc=title_entities_by_doc,
                    entity_to_docs=entity_to_docs,
                    title_entity_to_docs=title_entity_to_docs,
                ):
                    target_signal = _target_query_atom_signal(
                        atoms=atoms_by_doc.get(int(target_doc_index), ()),
                        signatures=signatures,
                        relation_alias_index=relation_alias_index,
                        relation_signal_policy=relation_signal_policy,
                    )
                    if not target_signal:
                        continue
                    target_atom, overlap_tokens, signal_type = target_signal
                    key = (
                        int(source_doc_index),
                        int(target_doc_index),
                        str(connector),
                        str(target_atom.evidence_unit_id),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    output.append(
                        TransitionWitness(
                            source_doc_index=int(source_doc_index),
                            target_doc_index=int(target_doc_index),
                            connector_entity=str(connector),
                            binding_type=str(binding_type),
                            source_atom_id=str(source_atom.evidence_unit_id),
                            target_atom_id=str(target_atom.evidence_unit_id),
                            target_query_overlap_tokens=tuple(overlap_tokens),
                            target_signal_type=str(signal_type),
                        )
                    )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                int(item.source_doc_index),
                int(item.target_doc_index),
                str(item.connector_entity),
                str(item.target_atom_id),
            ),
        )
    )


def _navigation_targets(
    *,
    entry_doc_indices: Sequence[int],
    link_policy: str,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    entities_by_doc: Mapping[int, Sequence[str]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
    entity_to_docs: Mapping[str, Sequence[int]],
    title_entity_to_docs: Mapping[str, Sequence[int]],
) -> set[int]:
    targets: set[int] = set()
    for source_doc_index in unique_ints(entry_doc_indices):
        for source_atom in atoms_by_doc.get(int(source_doc_index), ()):
            for connector in source_atom.endpoint_entities:
                targets.update(
                    int(target_doc_index)
                    for target_doc_index, _binding_type in _bound_targets_for_connector(
                        source_doc_index=int(source_doc_index),
                        connector=connector,
                        link_policy=link_policy,
                        atoms_by_doc=atoms_by_doc,
                        entities_by_doc=entities_by_doc,
                        title_entities_by_doc=title_entities_by_doc,
                        entity_to_docs=entity_to_docs,
                        title_entity_to_docs=title_entity_to_docs,
                    )
                )
    return targets


def _bound_targets_for_connector(
    *,
    source_doc_index: int,
    connector: str,
    link_policy: str,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    entities_by_doc: Mapping[int, Sequence[str]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
    entity_to_docs: Mapping[str, Sequence[int]],
    title_entity_to_docs: Mapping[str, Sequence[int]],
) -> Tuple[Tuple[int, str], ...]:
    connector_text = str(connector or "")
    if not _is_specific_entity(connector_text):
        return ()
    normalized_policy = str(link_policy or "anchored_target_atom_bound").strip().lower()
    output = []
    for target_doc_index in entity_to_docs.get(connector_text, ()):
        target = int(target_doc_index)
        if target == int(source_doc_index):
            continue
        binding_type = _binding_type(
            connector=connector_text,
            target_doc_index=target,
            link_policy=normalized_policy,
            atoms_by_doc=atoms_by_doc,
            title_entities_by_doc=title_entities_by_doc,
            title_entity_to_docs=title_entity_to_docs,
        )
        if not binding_type:
            continue
        output.append((target, binding_type))
    return tuple(dict.fromkeys(output))


def _binding_type(
    *,
    connector: str,
    target_doc_index: int,
    link_policy: str,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
    title_entity_to_docs: Mapping[str, Sequence[int]],
) -> str:
    if connector in set(str(item) for item in title_entities_by_doc.get(int(target_doc_index), ())):
        return "target_title"
    if str(link_policy) == "title_bound":
        return ""
    if not _connector_binds_target_atom(
        connector=connector,
        target_doc_index=int(target_doc_index),
        atoms_by_doc=atoms_by_doc,
        title_entities_by_doc=title_entities_by_doc,
    ):
        return ""
    if str(link_policy) == "anchored_target_atom_bound" and not title_entity_to_docs.get(
        str(connector)
    ):
        return ""
    return "target_atom"


def _connector_binds_target_atom(
    *,
    connector: str,
    target_doc_index: int,
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    title_entities_by_doc: Mapping[int, Sequence[str]],
) -> bool:
    title_entities = set(
        str(item) for item in title_entities_by_doc.get(int(target_doc_index), ())
    )
    if not connector or not title_entities:
        return False
    for atom in atoms_by_doc.get(int(target_doc_index), ()):
        endpoints = set(str(item) for item in atom.endpoint_entities)
        if connector in endpoints and endpoints & title_entities:
            return True
    return False


def _target_query_atom_signal(
    *,
    atoms: Sequence[SourceWarrantedAtom],
    signatures: QuerySignatures,
    relation_alias_index: RelationAliasIndex,
    relation_signal_policy: str,
) -> Tuple[SourceWarrantedAtom, Tuple[str, ...], str] | None:
    for signal_type in ("predicate", "sentence", "endpoint"):
        for atom in atoms:
            text = _atom_signal_text(atom=atom, signal_type=signal_type)
            overlap = query_relation_overlap(signatures, text)
            if overlap:
                return atom, tuple(overlap), signal_type
    if str(relation_signal_policy) == "pool_alias":
        expanded_relation_stems = _expanded_query_relation_stems(
            signatures=signatures,
            relation_alias_index=relation_alias_index,
            relation_signal_policy=relation_signal_policy,
        )
        for atom in atoms:
            overlap = _relation_overlap_with_stems(expanded_relation_stems, atom.predicate)
            if overlap:
                return atom, tuple(overlap), "pool_alias_predicate"
    return None


def _atom_signal_text(*, atom: SourceWarrantedAtom, signal_type: str) -> str:
    if signal_type == "predicate":
        return str(atom.predicate)
    if signal_type == "endpoint":
        return " ".join((str(atom.subject), str(atom.object)))
    return str(atom.source_sentence)


def _covered_query_relation_stems(
    *,
    doc_indices: Sequence[int],
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
    signatures: QuerySignatures,
) -> Tuple[str, ...]:
    covered = []
    for doc_index in unique_ints(doc_indices):
        for atom in atoms_by_doc.get(int(doc_index), ()):
            for signal_type in ("predicate", "sentence", "endpoint"):
                covered.extend(
                    query_relation_overlap(
                        signatures,
                        _atom_signal_text(atom=atom, signal_type=signal_type),
                    )
                )
    return tuple(dict.fromkeys(str(item) for item in covered if item))


def _relation_alias_index(
    atoms_by_doc: Mapping[int, Sequence[SourceWarrantedAtom]],
) -> RelationAliasIndex:
    predicates_by_endpoint_pair: Dict[Tuple[str, str], List[str]] = {}
    for atoms in atoms_by_doc.values():
        for atom in atoms:
            endpoints = tuple(sorted(str(item) for item in atom.endpoint_entities if item))
            if len(endpoints) != 2:
                continue
            predicate_stems = tuple(token_stems_for_text(atom.predicate))
            if not predicate_stems:
                continue
            predicates_by_endpoint_pair.setdefault(endpoints, []).extend(predicate_stems)

    mutable_aliases: Dict[str, set[str]] = {}
    alias_pair_count = 0
    for predicate_stems in predicates_by_endpoint_pair.values():
        unique_stems = tuple(dict.fromkeys(str(item) for item in predicate_stems if item))
        if len(unique_stems) < 2:
            continue
        for source_stem in unique_stems:
            for target_stem in unique_stems:
                if source_stem == target_stem:
                    continue
                mutable_aliases.setdefault(source_stem, set()).add(target_stem)
                alias_pair_count += 1
    aliases_by_stem = {
        stem: tuple(sorted(aliases))
        for stem, aliases in sorted(mutable_aliases.items())
    }
    return RelationAliasIndex(
        aliases_by_stem=aliases_by_stem,
        trace={
            "relation_alias_stem_count": len(aliases_by_stem),
            "relation_alias_pair_count": alias_pair_count,
        },
    )


def _expanded_query_relation_stems(
    *,
    signatures: QuerySignatures,
    relation_alias_index: RelationAliasIndex,
    relation_signal_policy: str,
) -> Tuple[str, ...]:
    output = list(str(item) for item in signatures.relation_stems if item)
    if str(relation_signal_policy) != "pool_alias":
        return tuple(dict.fromkeys(output))
    seen = set(output)
    for stem in tuple(output):
        for alias in relation_alias_index.aliases_by_stem.get(str(stem), ()):
            alias_text = str(alias)
            if not alias_text or alias_text in seen:
                continue
            seen.add(alias_text)
            output.append(alias_text)
    return tuple(dict.fromkeys(output))


def _relation_overlap_with_stems(
    relation_stems: Sequence[str],
    text: object,
) -> Tuple[str, ...]:
    text_stems = set(token_stems_for_text(text))
    return tuple(str(stem) for stem in relation_stems if str(stem) in text_stems)


def _is_specific_entity(entity: object) -> bool:
    item = str(entity or "")
    entity_tokens = item.split()
    if len(entity_tokens) >= 2:
        return True
    return bool(entity_tokens and len(entity_tokens[0]) >= 5 and not entity_tokens[0].isdigit())


def _metrics_from_totals(totals: Mapping[str, int]) -> Dict[str, float]:
    metrics = {
        "navigation_witness_missing_gold_coverage": _safe_div(
            totals.get("navigation_witness_missing_gold_docs", 0),
            totals.get("missing_gold_in_pool_docs", 0),
        ),
        "transition_witness_missing_gold_coverage": _safe_div(
            totals.get("transition_witness_missing_gold_docs", 0),
            totals.get("missing_gold_in_pool_docs", 0),
        ),
        "transition_gold_recall_over_navigation_gold": _safe_div(
            totals.get("transition_witness_missing_gold_docs", 0),
            totals.get("navigation_witness_missing_gold_docs", 0),
        ),
        "query_navigation_witness_rate": _safe_div(
            totals.get("queries_with_navigation_witness_missing_gold", 0),
            totals.get("queries_with_missing_gold_in_pool", 0),
        ),
        "query_transition_witness_rate": _safe_div(
            totals.get("queries_with_transition_witness_missing_gold", 0),
            totals.get("queries_with_missing_gold_in_pool", 0),
        ),
        "necessary_but_no_navigation_witness_rate": _safe_div(
            totals.get("necessary_but_no_navigation_witness_docs", 0),
            totals.get("missing_gold_in_pool_docs", 0),
        ),
        "navigation_legal_precision": _safe_div(
            totals.get("navigation_witness_missing_gold_docs", 0),
            totals.get("navigation_target_docs", 0),
        ),
        "transition_legal_precision": _safe_div(
            totals.get("transition_witness_missing_gold_docs", 0),
            totals.get("transition_target_docs", 0),
        ),
        "navigation_legal_but_unnecessary_rate": _safe_div(
            totals.get("navigation_legal_but_unnecessary_docs", 0),
            totals.get("navigation_target_docs", 0),
        ),
        "transition_legal_but_unnecessary_rate": _safe_div(
            totals.get("transition_legal_but_unnecessary_docs", 0),
            totals.get("transition_target_docs", 0),
        ),
        "mean_transition_witness_count": _safe_div(
            totals.get("transition_witness_count", 0),
            totals.get("queries", 0),
        ),
        "transition_target_title_binding_fraction": _safe_div(
            totals.get("transition_target_title_binding_count", 0),
            totals.get("transition_witness_count", 0),
        ),
        "transition_target_atom_binding_fraction": _safe_div(
            totals.get("transition_target_atom_binding_count", 0),
            totals.get("transition_witness_count", 0),
        ),
        "transition_predicate_signal_fraction": _safe_div(
            totals.get("transition_predicate_signal_count", 0),
            totals.get("transition_witness_count", 0),
        ),
        "transition_sentence_signal_fraction": _safe_div(
            totals.get("transition_sentence_signal_count", 0),
            totals.get("transition_witness_count", 0),
        ),
        "transition_endpoint_signal_fraction": _safe_div(
            totals.get("transition_endpoint_signal_count", 0),
            totals.get("transition_witness_count", 0),
        ),
        "transition_pool_alias_signal_fraction": _safe_div(
            totals.get("transition_pool_alias_signal_count", 0),
            totals.get("transition_witness_count", 0),
        ),
        "mean_relation_alias_pair_count": _safe_div(
            totals.get("relation_alias_pair_count", 0),
            totals.get("queries", 0),
        ),
        "mean_relation_alias_stem_count": _safe_div(
            totals.get("relation_alias_stem_count", 0),
            totals.get("queries", 0),
        ),
        "baseline_query_atom_surface_coverage": _safe_div(
            totals.get("baseline_query_atom_surface_covered_stem_count", 0),
            totals.get("query_relation_stem_count", 0),
        ),
        "query_full_baseline_query_atom_surface_coverage_rate": _safe_div(
            totals.get("queries_with_full_baseline_query_atom_surface_coverage", 0),
            totals.get("queries", 0),
        ),
    }
    metrics.update(_closure_metrics("closed", totals))
    metrics.update(_closure_metrics("deficit", totals))
    return metrics


def _closure_metrics(prefix: str, totals: Mapping[str, int]) -> Dict[str, float]:
    return {
        f"{prefix}_query_rate": _safe_div(
            totals.get(f"{prefix}_queries", 0),
            totals.get("queries", 0),
        ),
        f"{prefix}_query_missing_gold_rate": _safe_div(
            totals.get(f"{prefix}_queries_with_missing_gold_in_pool", 0),
            totals.get(f"{prefix}_queries", 0),
        ),
        f"{prefix}_navigation_witness_missing_gold_coverage": _safe_div(
            totals.get(f"{prefix}_navigation_witness_missing_gold_docs", 0),
            totals.get(f"{prefix}_missing_gold_in_pool_docs", 0),
        ),
        f"{prefix}_transition_witness_missing_gold_coverage": _safe_div(
            totals.get(f"{prefix}_transition_witness_missing_gold_docs", 0),
            totals.get(f"{prefix}_missing_gold_in_pool_docs", 0),
        ),
        f"{prefix}_transition_gold_recall_over_navigation_gold": _safe_div(
            totals.get(f"{prefix}_transition_witness_missing_gold_docs", 0),
            totals.get(f"{prefix}_navigation_witness_missing_gold_docs", 0),
        ),
        f"{prefix}_query_navigation_witness_rate": _safe_div(
            totals.get(f"{prefix}_queries_with_navigation_witness_missing_gold", 0),
            totals.get(f"{prefix}_queries_with_missing_gold_in_pool", 0),
        ),
        f"{prefix}_query_transition_witness_rate": _safe_div(
            totals.get(f"{prefix}_queries_with_transition_witness_missing_gold", 0),
            totals.get(f"{prefix}_queries_with_missing_gold_in_pool", 0),
        ),
        f"{prefix}_navigation_legal_precision": _safe_div(
            totals.get(f"{prefix}_navigation_witness_missing_gold_docs", 0),
            totals.get(f"{prefix}_navigation_target_docs", 0),
        ),
        f"{prefix}_transition_legal_precision": _safe_div(
            totals.get(f"{prefix}_transition_witness_missing_gold_docs", 0),
            totals.get(f"{prefix}_transition_target_docs", 0),
        ),
        f"{prefix}_navigation_legal_but_unnecessary_rate": _safe_div(
            totals.get(f"{prefix}_navigation_legal_but_unnecessary_docs", 0),
            totals.get(f"{prefix}_navigation_target_docs", 0),
        ),
        f"{prefix}_transition_legal_but_unnecessary_rate": _safe_div(
            totals.get(f"{prefix}_transition_legal_but_unnecessary_docs", 0),
            totals.get(f"{prefix}_transition_target_docs", 0),
        ),
        f"{prefix}_mean_transition_target_count": _safe_div(
            totals.get(f"{prefix}_transition_target_docs", 0),
            totals.get(f"{prefix}_queries", 0),
        ),
        f"{prefix}_mean_transition_witness_count": _safe_div(
            totals.get(f"{prefix}_transition_witness_count", 0),
            totals.get(f"{prefix}_queries", 0),
        ),
    }


def _safe_div(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _transition_witness_preview(
    witnesses: Sequence[TransitionWitness],
    *,
    limit: int = 8,
) -> Tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "source_doc_index": int(item.source_doc_index),
            "target_doc_index": int(item.target_doc_index),
            "connector_entity": str(item.connector_entity),
            "binding_type": str(item.binding_type),
            "source_atom_id": str(item.source_atom_id),
            "target_atom_id": str(item.target_atom_id),
            "target_query_overlap_tokens": tuple(item.target_query_overlap_tokens),
            "target_signal_type": str(item.target_signal_type),
        }
        for item in witnesses[: max(int(limit), 0)]
    )


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
        relation_signal_policy=str(args.relation_signal_policy),
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
            "relation_signal_policy": str(args.relation_signal_policy),
            "max_queries": max_queries,
        },
        **payload,
    }


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    metrics = payload.get("metrics", {}) or {}
    lines = [
        "# Source-Warranted Transition Witness Audit",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| rows | {payload.get('row_count', 0)} |",
        f"| source variant | {payload.get('source_variant', '')} |",
        f"| entry source policy | {payload.get('entry_source_policy', '')} |",
        f"| link policy | {payload.get('link_policy', '')} |",
        f"| relation signal policy | {payload.get('relation_signal_policy', '')} |",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| navigation witness missing-gold coverage | {float(metrics.get('navigation_witness_missing_gold_coverage', 0.0)):.4f} |",
        f"| transition witness missing-gold coverage | {float(metrics.get('transition_witness_missing_gold_coverage', 0.0)):.4f} |",
        f"| transition gold recall over navigation gold | {float(metrics.get('transition_gold_recall_over_navigation_gold', 0.0)):.4f} |",
        f"| query navigation witness rate | {float(metrics.get('query_navigation_witness_rate', 0.0)):.4f} |",
        f"| query transition witness rate | {float(metrics.get('query_transition_witness_rate', 0.0)):.4f} |",
        f"| navigation legal precision | {float(metrics.get('navigation_legal_precision', 0.0)):.4f} |",
        f"| transition legal precision | {float(metrics.get('transition_legal_precision', 0.0)):.4f} |",
        f"| navigation legal-but-unnecessary rate | {float(metrics.get('navigation_legal_but_unnecessary_rate', 0.0)):.4f} |",
        f"| transition legal-but-unnecessary rate | {float(metrics.get('transition_legal_but_unnecessary_rate', 0.0)):.4f} |",
        f"| mean transition witness count | {float(metrics.get('mean_transition_witness_count', 0.0)):.2f} |",
        f"| transition target-title binding fraction | {float(metrics.get('transition_target_title_binding_fraction', 0.0)):.4f} |",
        f"| transition target-atom binding fraction | {float(metrics.get('transition_target_atom_binding_fraction', 0.0)):.4f} |",
        f"| transition predicate signal fraction | {float(metrics.get('transition_predicate_signal_fraction', 0.0)):.4f} |",
        f"| transition sentence signal fraction | {float(metrics.get('transition_sentence_signal_fraction', 0.0)):.4f} |",
        f"| transition endpoint signal fraction | {float(metrics.get('transition_endpoint_signal_fraction', 0.0)):.4f} |",
        f"| transition pool-alias signal fraction | {float(metrics.get('transition_pool_alias_signal_fraction', 0.0)):.4f} |",
        f"| mean relation alias pair count | {float(metrics.get('mean_relation_alias_pair_count', 0.0)):.2f} |",
        f"| mean relation alias stem count | {float(metrics.get('mean_relation_alias_stem_count', 0.0)):.2f} |",
        f"| baseline query-atom surface coverage | {float(metrics.get('baseline_query_atom_surface_coverage', 0.0)):.4f} |",
        f"| query full baseline query-atom surface coverage rate | {float(metrics.get('query_full_baseline_query_atom_surface_coverage_rate', 0.0)):.4f} |",
        f"| closed query rate | {float(metrics.get('closed_query_rate', 0.0)):.4f} |",
        f"| closed transition coverage | {float(metrics.get('closed_transition_witness_missing_gold_coverage', 0.0)):.4f} |",
        f"| closed transition legal precision | {float(metrics.get('closed_transition_legal_precision', 0.0)):.4f} |",
        f"| closed mean transition target count | {float(metrics.get('closed_mean_transition_target_count', 0.0)):.2f} |",
        f"| deficit query rate | {float(metrics.get('deficit_query_rate', 0.0)):.4f} |",
        f"| deficit transition coverage | {float(metrics.get('deficit_transition_witness_missing_gold_coverage', 0.0)):.4f} |",
        f"| deficit transition legal precision | {float(metrics.get('deficit_transition_legal_precision', 0.0)):.4f} |",
        f"| deficit mean transition target count | {float(metrics.get('deficit_mean_transition_target_count', 0.0)):.2f} |",
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
        choices=("title_bound", "target_atom_bound", "anchored_target_atom_bound"),
        default="anchored_target_atom_bound",
    )
    parser.add_argument(
        "--relation-signal-policy",
        choices=("surface", "pool_alias"),
        default="surface",
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
