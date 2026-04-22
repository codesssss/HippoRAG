from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .requirements import (
    build_requirement_probe_text,
    extract_requirement_units,
    extract_requirement_units_llm_closed_grounded,
    extract_requirement_units_llm,
    extract_requirement_units_llm_grounded,
    fillers_equivalent,
    is_role_slot,
    slot_family_lexical_cues,
    slot_family_lexical_cues_extended,
)
from .traces import load_instruction_aware_query_embeddings, normalize_entity_text
from .types import (
    RASReadoutTrace,
    RequirementConflictTrace,
    RequirementCoverageTrace,
    RequirementUnit,
    StrongestConfig,
    StrongestTraceState,
)


def normalize_relation_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_positive_distribution(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    if values.size == 0:
        return values
    total = float(np.sum(values))
    if total <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values / total).astype(np.float32, copy=False)


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        denom = float(np.linalg.norm(values))
        if denom <= 0.0:
            return values.astype(np.float32, copy=False)
        return (values / denom).astype(np.float32, copy=False)
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (values / denom).astype(np.float32, copy=False)


def _encode_probe_text_embeddings(embedding_model, texts: Sequence[str]) -> np.ndarray:
    if embedding_model is None:
        return np.zeros((0, 0), dtype=np.float32)
    normalized_texts = [str(text or "").strip() for text in texts if str(text or "").strip()]
    if not normalized_texts:
        return np.zeros((0, 0), dtype=np.float32)
    encoded = load_instruction_aware_query_embeddings(embedding_model, normalized_texts)
    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.ndim == 1:
        encoded = encoded.reshape(1, -1)
    return _l2_normalize_rows(encoded)


def _compute_embedding_requirement_support_hints(
    *,
    requirement_units: Sequence[RequirementUnit],
    metadata: Dict[str, object],
    num_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_scores = np.zeros((num_candidates, len(requirement_units)), dtype=np.float32)
    slot_scores = np.zeros((num_candidates, len(requirement_units)), dtype=np.float32)
    if num_candidates <= 0 or not requirement_units:
        return anchor_scores, slot_scores

    support_mode = str(metadata.get("ras_support_mode", "lexical")).strip().lower()
    if support_mode != "embedding_probe":
        return anchor_scores, slot_scores

    embedding_model = metadata.get("_ras_embedding_model")
    passage_embeddings = np.asarray(metadata.get("_ras_passage_embeddings", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    if passage_embeddings.ndim != 2 or passage_embeddings.shape[0] < num_candidates or passage_embeddings.shape[1] <= 0:
        return anchor_scores, slot_scores
    passage_embeddings = _l2_normalize_rows(passage_embeddings[:num_candidates])

    query_text = str(metadata.get("query_text", "") or "")
    probe_texts = [build_requirement_probe_text(query_text, unit) for unit in requirement_units]
    probe_embeddings = _encode_probe_text_embeddings(embedding_model, probe_texts)
    if probe_embeddings.ndim != 2 or probe_embeddings.shape[0] != len(requirement_units):
        return anchor_scores, slot_scores
    if probe_embeddings.shape[1] != passage_embeddings.shape[1]:
        return anchor_scores, slot_scores

    slot_scores = np.clip(passage_embeddings @ probe_embeddings.T, 0.0, 1.0).astype(np.float32, copy=False)

    for unit_idx, unit in enumerate(requirement_units):
        normalized_anchors = [
            normalize_entity_text(anchor)
            for anchor in unit.anchor_entities
            if normalize_entity_text(anchor)
        ]
        if not normalized_anchors:
            if len(unit.anchor_entities) == 0:
                anchor_scores[:, unit_idx] = np.float32(1.0)
            continue
        anchor_embeddings = _encode_probe_text_embeddings(embedding_model, normalized_anchors)
        if anchor_embeddings.ndim != 2 or anchor_embeddings.shape[0] == 0:
            continue
        if anchor_embeddings.shape[1] != passage_embeddings.shape[1]:
            continue
        anchor_scores[:, unit_idx] = np.max(
            np.clip(passage_embeddings @ anchor_embeddings.T, 0.0, 1.0),
            axis=1,
        ).astype(np.float32, copy=False)
    return anchor_scores, slot_scores


def select_protected_anchor_local_indices(
    local_passage_prior: np.ndarray,
    local_reset_scores: np.ndarray,
    protected_anchor_k: int,
) -> np.ndarray:
    num_candidates = len(local_reset_scores)
    if num_candidates == 0 or int(protected_anchor_k) <= 0:
        return np.zeros(0, dtype=np.int64)

    protected_anchor_k = min(max(int(protected_anchor_k), 1), num_candidates)
    anchor_source = np.clip(np.asarray(local_passage_prior, dtype=np.float32), 0.0, None)
    if np.any(anchor_source > 0):
        order = np.argsort(anchor_source)[::-1]
    else:
        order = np.argsort(np.asarray(local_reset_scores, dtype=np.float32))[::-1]
    return np.asarray(order[:protected_anchor_k], dtype=np.int64)


def build_protected_readout_scores(
    base_scores: np.ndarray,
    protected_local_indices: Sequence[int],
) -> np.ndarray:
    readout_scores = np.asarray(base_scores, dtype=np.float32).copy()
    if readout_scores.size == 0:
        return readout_scores

    protected_order: List[int] = []
    seen = set()
    for local_idx in protected_local_indices:
        local_idx = int(local_idx)
        if local_idx < 0 or local_idx >= len(readout_scores) or local_idx in seen:
            continue
        protected_order.append(local_idx)
        seen.add(local_idx)
    if not protected_order:
        return readout_scores

    base_ceiling = float(np.max(np.clip(readout_scores, 0.0, None)))
    promotion_step = max(float(np.finfo(np.float32).eps) * max(base_ceiling, 1.0) * 32.0, 1e-6)
    promoted_ceiling = base_ceiling + promotion_step * (len(protected_order) + 1)
    for offset, local_idx in enumerate(protected_order):
        readout_scores[local_idx] = np.float32(promoted_ceiling - promotion_step * offset)
    return readout_scores


def select_boundary_frontier_bonus_local_indices(
    strongest_order: np.ndarray,
    selected_local_indices: np.ndarray,
    protected_local_indices: np.ndarray,
    final_k: int,
    frontier_bonus_top_k: int,
) -> np.ndarray:
    strongest_order = np.asarray(strongest_order, dtype=np.int64)
    selected_local_indices = np.asarray(selected_local_indices, dtype=np.int64)
    protected_local_indices = np.asarray(protected_local_indices, dtype=np.int64)

    if strongest_order.size == 0 or selected_local_indices.size == 0 or int(frontier_bonus_top_k) <= 0:
        return np.zeros(0, dtype=np.int64)

    num_candidates = int(np.max(strongest_order)) + 1
    selected_local_indices = selected_local_indices[
        (selected_local_indices >= 0) & (selected_local_indices < num_candidates)
    ]
    protected_local_indices = protected_local_indices[
        (protected_local_indices >= 0) & (protected_local_indices < num_candidates)
    ]
    if selected_local_indices.size == 0:
        return np.zeros(0, dtype=np.int64)

    selected_set = {int(local_idx) for local_idx in selected_local_indices.tolist()}
    protected_set = {int(local_idx) for local_idx in protected_local_indices.tolist()}
    strongest_topk_set = {
        int(local_idx)
        for local_idx in strongest_order[: min(max(int(final_k), 0), len(strongest_order))].tolist()
    }
    boundary_candidates = [
        int(local_idx)
        for local_idx in strongest_order.tolist()
        if int(local_idx) in selected_set
        and int(local_idx) not in protected_set
        and int(local_idx) not in strongest_topk_set
    ][: max(int(frontier_bonus_top_k), 1)]
    return np.asarray(boundary_candidates, dtype=np.int64)


def _normalize_triples(raw_triples: Sequence[object]) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for triple in raw_triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject = normalize_entity_text(str(triple[0]))
        relation = normalize_relation_text(str(triple[1]))
        obj = normalize_entity_text(str(triple[2]))
        if not subject or not relation or not obj:
            continue
        triples.append((subject, relation, obj))
    return triples


def build_query_fact_units(
    trace_state: StrongestTraceState,
    metadata: Dict[str, object],
) -> List[Dict[str, object]]:
    query_entities = {
        normalize_entity_text(str(entity))
        for entity in metadata.get("query_entities", []) or []
        if normalize_entity_text(str(entity))
    }
    seed_entities = {
        normalize_entity_text(str(entity))
        for entity in metadata.get("seed_entities", []) or []
        if normalize_entity_text(str(entity))
    }
    focus_entities = query_entities | seed_entities
    if not focus_entities:
        focus_entities = {
            normalize_entity_text(str(entity))
            for entity in metadata.get("entity_vocab", []) or []
            if normalize_entity_text(str(entity))
        }

    candidate_indices = [
        int(local_idx)
        for local_idx in np.asarray(metadata.get("selected_local_indices", []), dtype=np.int64).tolist()
        if int(local_idx) >= 0
    ]
    passage_chunk_ids = metadata.get("passage_chunk_ids", []) or []
    chunk_triples_map = metadata.get("chunk_triples_map", {}) or {}

    dedup: Dict[Tuple[str, str, str], float] = {}
    for local_idx in candidate_indices:
        if local_idx >= len(passage_chunk_ids):
            continue
        chunk_id = passage_chunk_ids[local_idx]
        if chunk_id is None:
            continue
        for subject, relation, obj in _normalize_triples(chunk_triples_map.get(str(chunk_id), [])):
            participants = {subject, obj}
            overlap = len(participants & focus_entities)
            if overlap <= 0:
                continue
            signature = (subject, relation, obj)
            base_weight = 1.0 + 0.35 * float(overlap)
            if subject in query_entities or obj in query_entities:
                base_weight += 0.25
            dedup[signature] = max(dedup.get(signature, 0.0), base_weight)

    return [
        {
            "subject": subject,
            "relation": relation,
            "object": obj,
            "participants": {subject, obj},
            "weight": float(weight),
        }
        for (subject, relation, obj), weight in dedup.items()
    ]


def build_local_query_fact_support_tensor(
    passage_chunk_ids: Sequence[str | None],
    candidate_indices: np.ndarray,
    selected_local_indices: np.ndarray,
    chunk_triples_map: Dict[str, Sequence[object]],
    query_fact_units: Sequence[Dict[str, object]],
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    selected_local_indices = np.asarray(selected_local_indices, dtype=np.int64)
    num_candidates = int(candidate_indices.size)
    num_facts = len(query_fact_units)
    support_tensor = np.zeros((num_candidates, num_facts, 3), dtype=np.float32)
    if num_candidates == 0 or num_facts == 0:
        return support_tensor

    valid_selected = {
        int(local_idx)
        for local_idx in selected_local_indices.tolist()
        if 0 <= int(local_idx) < num_candidates
    }
    for local_idx in valid_selected:
        if local_idx >= len(passage_chunk_ids):
            continue
        chunk_id = passage_chunk_ids[local_idx]
        if chunk_id is None:
            continue
        passage_triples = _normalize_triples(chunk_triples_map.get(str(chunk_id), []))
        if not passage_triples:
            continue
        for fact_idx, fact in enumerate(query_fact_units):
            subject = normalize_entity_text(str(fact.get("subject", "")))
            obj = normalize_entity_text(str(fact.get("object", "")))
            relation = normalize_relation_text(str(fact.get("relation", "")))
            if not subject or not obj:
                continue
            subject_support = 0.0
            object_support = 0.0
            pair_support = 0.0
            for triple_subject, triple_relation, triple_object in passage_triples:
                triple_entities = {triple_subject, triple_object}
                if subject in triple_entities:
                    subject_support = 1.0
                if obj in triple_entities:
                    object_support = 1.0
                if relation == triple_relation and {subject, obj} == triple_entities:
                    pair_support = 1.0
                if subject_support > 0.0 and object_support > 0.0 and pair_support > 0.0:
                    break
            support_tensor[local_idx, fact_idx, 0] = np.float32(subject_support)
            support_tensor[local_idx, fact_idx, 1] = np.float32(object_support)
            support_tensor[local_idx, fact_idx, 2] = np.float32(pair_support)
    return support_tensor


def build_frontier_headcomp_route_passage_scores(
    support_tensor: np.ndarray,
    query_fact_units: Sequence[Dict[str, object]],
    core_local_indices: np.ndarray,
    frontier_local_indices: np.ndarray,
) -> Tuple[np.ndarray, float]:
    if support_tensor.ndim != 3:
        return np.zeros(0, dtype=np.float32), 0.0

    num_candidates = int(support_tensor.shape[0])
    passage_scores = np.zeros(num_candidates, dtype=np.float32)
    if num_candidates == 0 or not query_fact_units:
        return passage_scores, 0.0

    core_local_indices = np.asarray(core_local_indices, dtype=np.int64)
    core_local_indices = core_local_indices[
        (core_local_indices >= 0) & (core_local_indices < num_candidates)
    ]
    frontier_local_indices = np.asarray(frontier_local_indices, dtype=np.int64)
    frontier_local_indices = frontier_local_indices[
        (frontier_local_indices >= 0) & (frontier_local_indices < num_candidates)
    ]
    if core_local_indices.size == 0 or frontier_local_indices.size == 0:
        return passage_scores, 0.0

    query_fact_weights = np.asarray(
        [max(float(fact.get("weight", 0.0)), 0.0) for fact in query_fact_units],
        dtype=np.float32,
    )
    if not np.any(query_fact_weights > 0):
        return passage_scores, 0.0

    head_subject_coverage = np.max(support_tensor[core_local_indices, :, 0], axis=0).astype(np.float32, copy=False)
    head_object_coverage = np.max(support_tensor[core_local_indices, :, 1], axis=0).astype(np.float32, copy=False)
    head_pair_coverage = np.max(support_tensor[core_local_indices, :, 2], axis=0).astype(np.float32, copy=False)
    head_fact_coverage = np.maximum(
        np.minimum(head_subject_coverage, head_object_coverage),
        head_pair_coverage,
    ).astype(np.float32, copy=False)

    fact_deficits = (
        query_fact_weights * np.clip(1.0 - head_fact_coverage, 0.0, 1.0)
    ).astype(np.float32, copy=False)
    total_fact_weight = float(np.sum(query_fact_weights))
    if total_fact_weight <= 1e-8 or not np.any(fact_deficits > 0):
        return passage_scores, 0.0

    query_deficit_mass = float(np.clip(np.sum(fact_deficits) / total_fact_weight, 0.0, 1.0))
    if query_deficit_mass <= 1e-8:
        return passage_scores, query_deficit_mass

    for local_idx in frontier_local_indices.tolist():
        subject_support = support_tensor[local_idx, :, 0].astype(np.float32, copy=False)
        object_support = support_tensor[local_idx, :, 1].astype(np.float32, copy=False)
        pair_support = support_tensor[local_idx, :, 2].astype(np.float32, copy=False)
        if not (np.any(subject_support > 0) or np.any(object_support > 0) or np.any(pair_support > 0)):
            continue

        cross_completion = np.maximum(
            np.minimum(head_subject_coverage, object_support),
            np.minimum(head_object_coverage, subject_support),
        ).astype(np.float32, copy=False)
        completion_utility = np.maximum(cross_completion, pair_support).astype(np.float32, copy=False)
        passage_scores[local_idx] = np.float32(np.dot(fact_deficits, completion_utility))

    return np.clip(passage_scores, 0.0, None).astype(np.float32, copy=False), query_deficit_mass


def _extract_doc_title(doc_text: str) -> str:
    lines = str(doc_text or "").splitlines()
    return lines[0].strip() if lines else str(doc_text or "").strip()


def build_requirement_units(
    *,
    query: str,
    metadata: Dict[str, object],
    max_units: int,
) -> tuple[List[RequirementUnit], Dict[str, object]]:
    passage_titles = [str(title) for title in metadata.get("passage_titles", []) or []]
    baseline_titles = passage_titles[:5]
    seed_entities = [str(entity) for entity in metadata.get("seed_entities", []) or []]
    query_entities = [str(entity) for entity in metadata.get("query_entities", []) or []]

    extractor_mode = str(metadata.get("ras_extractor_mode", "rule")).strip().lower()
    if extractor_mode == "llm":
        return extract_requirement_units_llm(
            query=query,
            seed_entities=seed_entities,
            query_entities=query_entities,
            baseline_titles=baseline_titles,
            max_units=max_units,
        )
    if extractor_mode in {"llm_closed_grounded", "llm_parser_closed_grounded"}:
        return extract_requirement_units_llm_closed_grounded(
            query=query,
            seed_entities=seed_entities,
            query_entities=query_entities,
            baseline_titles=baseline_titles,
            max_units=max_units,
        )
    if extractor_mode in {"llm_grounded", "llm_parser_grounded"}:
        return extract_requirement_units_llm_grounded(
            query=query,
            seed_entities=seed_entities,
            query_entities=query_entities,
            baseline_titles=baseline_titles,
            max_units=max_units,
        )

    return extract_requirement_units(
        query=query,
        seed_entities=seed_entities,
        query_entities=query_entities,
        baseline_titles=baseline_titles,
        max_units=max_units,
    )


def _entity_contains(entity: str, anchor: str) -> bool:
    if not entity or not anchor:
        return False
    return anchor in entity or entity in anchor


def _anchor_matches_entity(anchor: str, entities: set | frozenset, title: str, text: str) -> bool:
    if anchor in entities:
        return True
    if title and (anchor in title or title in anchor):
        return True
    if text and anchor in text:
        return True
    for entity in entities:
        if _entity_contains(entity, anchor):
            return True
    return False


def _build_local_requirement_support_tensor(
    *,
    candidate_indices: np.ndarray,
    selected_local_indices: np.ndarray,
    requirement_units: Sequence[RequirementUnit],
    metadata: Dict[str, object],
) -> tuple[np.ndarray, List[List[str]]]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    selected_local_indices = np.asarray(selected_local_indices, dtype=np.int64)
    num_candidates = int(candidate_indices.size)
    num_units = len(requirement_units)
    support_tensor = np.zeros((num_candidates, num_units, 3), dtype=np.float32)
    filler_values: List[List[str]] = [["" for _ in range(num_units)] for _ in range(num_candidates)]
    if num_candidates == 0 or num_units == 0:
        return support_tensor, filler_values

    support_mode = str(metadata.get("ras_support_mode", "lexical")).strip().lower()
    probe_threshold = max(float(metadata.get("ras_embedding_probe_threshold", 0.35) or 0.35), 0.0)
    semantic_anchor_scores, semantic_slot_scores = _compute_embedding_requirement_support_hints(
        requirement_units=requirement_units,
        metadata=metadata,
        num_candidates=num_candidates,
    )
    passage_chunk_ids = metadata.get("passage_chunk_ids", []) or []
    chunk_triples_map = metadata.get("chunk_triples_map", {}) or {}
    passage_titles = [str(title) for title in metadata.get("passage_titles", []) or []]
    passage_texts = [str(text) for text in metadata.get("passage_texts", []) or []]
    passage_structure_entities = metadata.get("passage_structure_entities", []) or []
    valid_selected = {
        int(local_idx)
        for local_idx in selected_local_indices.tolist()
        if 0 <= int(local_idx) < num_candidates
    }

    for local_idx in valid_selected:
        passage_title = normalize_entity_text(
            passage_titles[local_idx] if local_idx < len(passage_titles) else ""
        )
        passage_text = normalize_entity_text(
            passage_texts[local_idx] if local_idx < len(passage_texts) else passage_title
        )
        structure_entities = {
            normalize_entity_text(str(entity))
            for entity in (
                passage_structure_entities[local_idx]
                if local_idx < len(passage_structure_entities)
                else []
            )
            if normalize_entity_text(str(entity))
        }
        chunk_id = passage_chunk_ids[local_idx] if local_idx < len(passage_chunk_ids) else None
        triples = _normalize_triples(chunk_triples_map.get(str(chunk_id), [])) if chunk_id is not None else []
        triple_entities = {entity for triple in triples for entity in (triple[0], triple[2])}
        passage_entities = structure_entities | triple_entities
        if passage_title:
            passage_entities.add(passage_title)

        for unit_idx, unit in enumerate(requirement_units):
            anchors = {
                normalize_entity_text(anchor)
                for anchor in unit.anchor_entities
                if normalize_entity_text(anchor)
            }
            is_bridge_ref = len(anchors) == 0

            anchor_hit = 0.0
            if anchors:
                anchor_hit = float(
                    any(
                        _anchor_matches_entity(anchor, passage_entities, passage_title, passage_text)
                        for anchor in anchors
                    )
                )
            elif is_bridge_ref:
                anchor_hit = 1.0
            semantic_anchor_hit = (
                float(semantic_anchor_scores[local_idx, unit_idx])
                if semantic_anchor_scores.shape == (num_candidates, num_units)
                else 0.0
            )
            if semantic_anchor_hit >= probe_threshold:
                anchor_hit = max(anchor_hit, semantic_anchor_hit)

            lexical_cues = [normalize_relation_text(cue) for cue in slot_family_lexical_cues_extended(unit.slot_family)]
            if not lexical_cues:
                lexical_cues = [normalize_relation_text(unit.slot_family)]
            slot_hit = float(
                any(cue and (cue in passage_title or cue in passage_text) for cue in lexical_cues)
            )
            semantic_slot_hit = (
                float(semantic_slot_scores[local_idx, unit_idx])
                if semantic_slot_scores.shape == (num_candidates, num_units)
                else 0.0
            )
            if semantic_slot_hit >= probe_threshold:
                slot_hit = max(slot_hit, semantic_slot_hit)
            filler_candidates: List[str] = []
            for subject, relation, obj in triples:
                relation_hit = any(
                    cue and (cue in relation or relation in cue) for cue in lexical_cues
                )
                if relation_hit:
                    slot_hit = 1.0
                if is_bridge_ref:
                    if relation_hit:
                        if subject:
                            filler_candidates.append(subject)
                        if obj:
                            filler_candidates.append(obj)
                    continue
                if not anchors:
                    continue
                triple_entities_local = {subject, obj}
                anchor_in_triple = any(
                    _anchor_matches_entity(a, triple_entities_local, "", "")
                    for a in anchors
                )
                if not relation_hit and not anchor_in_triple:
                    continue
                for a in anchors:
                    if _entity_contains(subject, a) and not _entity_contains(obj, a):
                        filler_candidates.append(obj)
                    elif _entity_contains(obj, a) and not _entity_contains(subject, a):
                        filler_candidates.append(subject)
                    elif relation_hit:
                        for entity in triple_entities_local:
                            if not _entity_contains(entity, a):
                                filler_candidates.append(entity)

            if not filler_candidates and slot_hit > 0.0 and not is_bridge_ref:
                filler_candidates = [
                    entity
                    for entity in sorted(passage_entities)
                    if entity and entity not in anchors
                ]
            if (
                not filler_candidates
                and is_bridge_ref
                and support_mode == "embedding_probe"
                and semantic_slot_hit >= probe_threshold
            ):
                bridge_entities: List[str] = []
                if passage_title:
                    bridge_entities.append(passage_title)
                bridge_entities.extend(
                    entity
                    for entity in sorted(passage_entities)
                    if entity and entity != passage_title
                )
                filler_candidates = bridge_entities
            filler_hit = float(len(filler_candidates) > 0)
            filler_values[local_idx][unit_idx] = filler_candidates[0] if filler_candidates else ""
            support_tensor[local_idx, unit_idx, 0] = np.float32(anchor_hit)
            support_tensor[local_idx, unit_idx, 1] = np.float32(slot_hit)
            support_tensor[local_idx, unit_idx, 2] = np.float32(filler_hit)
    return support_tensor, filler_values


def compute_requirement_support(
    support_tensor: np.ndarray,
    requirement_units: Sequence[RequirementUnit],
    strict_eligibility: bool = True,
) -> np.ndarray:
    if support_tensor.ndim != 3:
        return np.zeros((0, 0), dtype=np.float32)
    num_candidates, num_units, _ = support_tensor.shape
    support_scores = np.zeros((num_candidates, num_units), dtype=np.float32)
    for unit_idx, unit in enumerate(requirement_units):
        anchor_hit = support_tensor[:, unit_idx, 0]
        slot_hit = support_tensor[:, unit_idx, 1]
        filler_hit = support_tensor[:, unit_idx, 2]
        is_bridge = len(unit.anchor_entities) == 0
        eligible = np.minimum(anchor_hit, slot_hit)
        if is_role_slot(unit.slot_family) or is_bridge:
            eligible = np.minimum(eligible, filler_hit)
        elif not strict_eligibility:
            eligible = np.maximum(eligible, filler_hit)
        support_scores[:, unit_idx] = eligible.astype(np.float32, copy=False)
    return support_scores


def collect_head_fillers_by_unit(
    *,
    requirement_units: Sequence[RequirementUnit],
    support_scores: np.ndarray,
    filler_values: Sequence[Sequence[str]],
    head_local_indices: np.ndarray,
) -> Dict[str, List[str]]:
    if support_scores.ndim != 2 or not requirement_units:
        return {}
    num_candidates, num_units = support_scores.shape
    head_local_indices = np.asarray(head_local_indices, dtype=np.int64)
    head_local_indices = head_local_indices[
        (head_local_indices >= 0) & (head_local_indices < num_candidates)
    ]
    if head_local_indices.size == 0:
        return {}

    fillers_by_unit: Dict[str, List[str]] = {}
    for unit_idx, unit in enumerate(requirement_units[:num_units]):
        collected: List[str] = []
        for local_idx in head_local_indices.tolist():
            if float(support_scores[local_idx, unit_idx]) <= 0.0:
                continue
            filler = (
                str(filler_values[local_idx][unit_idx]).strip()
                if local_idx < len(filler_values) and unit_idx < len(filler_values[local_idx])
                else ""
            )
            if not filler:
                continue
            if any(fillers_equivalent(filler, prev) for prev in collected):
                continue
            collected.append(filler)
        if collected:
            fillers_by_unit[unit.unit_id] = collected
    return fillers_by_unit


def compute_requirement_set_coverage(
    *,
    requirement_units: Sequence[RequirementUnit],
    support_scores: np.ndarray,
    local_indices: np.ndarray,
    filler_values: Sequence[Sequence[str]] | None = None,
) -> np.ndarray:
    num_units = len(requirement_units)
    if num_units == 0 or support_scores.ndim != 2 or support_scores.shape[1] != num_units:
        return np.zeros(num_units, dtype=np.float32)

    num_candidates = int(support_scores.shape[0])
    local_indices = np.asarray(local_indices, dtype=np.int64)
    local_indices = local_indices[
        (local_indices >= 0) & (local_indices < num_candidates)
    ]
    if local_indices.size == 0:
        return np.zeros(num_units, dtype=np.float32)

    if filler_values is None:
        return np.max(support_scores[local_indices, :], axis=0).astype(np.float32, copy=False)

    coverage_scores = np.zeros(num_units, dtype=np.float32)
    fillers_by_unit = collect_head_fillers_by_unit(
        requirement_units=requirement_units,
        support_scores=support_scores,
        filler_values=filler_values,
        head_local_indices=local_indices,
    )

    for unit_idx, unit in enumerate(requirement_units):
        if not unit.bridge_targets:
            coverage_scores[unit_idx] = float(np.max(support_scores[local_indices, unit_idx]))
            continue

        resolved_target_fillers: List[str] = []
        prereq_covered = True
        for bridge_target in unit.bridge_targets:
            bridge_fillers = fillers_by_unit.get(bridge_target, [])
            if not bridge_fillers:
                prereq_covered = False
                break
            for filler in bridge_fillers:
                if not any(fillers_equivalent(filler, prev) for prev in resolved_target_fillers):
                    resolved_target_fillers.append(filler)
        if not prereq_covered or not resolved_target_fillers:
            continue

        for local_idx in local_indices.tolist():
            candidate_score = float(support_scores[local_idx, unit_idx])
            if candidate_score <= 0.0:
                continue
            candidate_filler = (
                str(filler_values[local_idx][unit_idx]).strip()
                if local_idx < len(filler_values) and unit_idx < len(filler_values[local_idx])
                else ""
            )
            if not candidate_filler:
                continue
            if any(
                fillers_equivalent(candidate_filler, target_filler)
                for target_filler in resolved_target_fillers
            ):
                coverage_scores[unit_idx] = max(coverage_scores[unit_idx], np.float32(candidate_score))
    return coverage_scores.astype(np.float32, copy=False)


def compute_head_requirement_coverage(
    *,
    requirement_units: Sequence[RequirementUnit],
    support_scores: np.ndarray,
    core_local_indices: np.ndarray,
    final_local_indices: np.ndarray,
    filler_values: Sequence[Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, List[RequirementCoverageTrace]]:
    num_units = len(requirement_units)
    if num_units == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
        )
    if support_scores.ndim != 2 or support_scores.shape[1] != num_units:
        return (
            np.zeros(num_units, dtype=np.float32),
            np.zeros(num_units, dtype=np.float32),
            [],
        )

    num_candidates = int(support_scores.shape[0])
    core_local_indices = np.asarray(core_local_indices, dtype=np.int64)
    core_local_indices = core_local_indices[
        (core_local_indices >= 0) & (core_local_indices < num_candidates)
    ]
    final_local_indices = np.asarray(final_local_indices, dtype=np.int64)
    final_local_indices = final_local_indices[
        (final_local_indices >= 0) & (final_local_indices < num_candidates)
    ]
    head_cov = compute_requirement_set_coverage(
        requirement_units=requirement_units,
        support_scores=support_scores,
        local_indices=core_local_indices,
        filler_values=filler_values,
    )
    final_cov = compute_requirement_set_coverage(
        requirement_units=requirement_units,
        support_scores=support_scores,
        local_indices=final_local_indices,
        filler_values=filler_values,
    )
    traces = [
        RequirementCoverageTrace(
            unit_id=unit.unit_id,
            tier=unit.tier,
            slot_family=unit.slot_family,
            head_coverage=float(head_cov[idx]),
            final_coverage=float(final_cov[idx]),
            unmet_mass=float(max(0.0, 1.0 - head_cov[idx])),
        )
        for idx, unit in enumerate(requirement_units)
    ]
    return head_cov, final_cov, traces


def compute_unmet_requirement_gain(
    *,
    requirement_units: Sequence[RequirementUnit],
    support_scores: np.ndarray,
    head_coverage: np.ndarray,
    frontier_local_indices: np.ndarray,
    filler_values: Sequence[Sequence[str]] | None = None,
    head_local_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_candidates = int(support_scores.shape[0]) if support_scores.ndim == 2 else 0
    g_core = np.zeros(num_candidates, dtype=np.float32)
    g_support = np.zeros(num_candidates, dtype=np.float32)
    unmet_mass = np.clip(1.0 - np.asarray(head_coverage, dtype=np.float32), 0.0, 1.0)
    if num_candidates == 0 or not requirement_units or unmet_mass.size == 0:
        return g_core, g_support, unmet_mass

    unit_id_to_idx = {unit.unit_id: idx for idx, unit in enumerate(requirement_units)}
    head_fillers_by_unit = (
        collect_head_fillers_by_unit(
            requirement_units=requirement_units,
            support_scores=support_scores,
            filler_values=filler_values or [],
            head_local_indices=np.asarray(head_local_indices if head_local_indices is not None else [], dtype=np.int64),
        )
        if filler_values is not None and head_local_indices is not None
        else {}
    )

    frontier_local_indices = np.asarray(frontier_local_indices, dtype=np.int64)
    frontier_local_indices = frontier_local_indices[
        (frontier_local_indices >= 0) & (frontier_local_indices < num_candidates)
    ]
    for local_idx in frontier_local_indices.tolist():
        unit_support = support_scores[local_idx, :]
        if not np.any(unit_support > 0):
            continue
        for unit_idx, unit in enumerate(requirement_units):
            gain = float(unmet_mass[unit_idx]) * float(unit_support[unit_idx])
            if gain <= 0:
                continue
            if unit.tier == "core":
                if unit.bridge_targets:
                    resolved_target_fillers: List[str] = []
                    prereq_indices = [unit_id_to_idx[bt] for bt in unit.bridge_targets if bt in unit_id_to_idx]
                    prereq_covered = bool(prereq_indices) and all(
                        float(head_coverage[bt_idx]) > 0.0 for bt_idx in prereq_indices
                    )
                    if not prereq_covered:
                        continue
                    for bridge_target in unit.bridge_targets:
                        for filler in head_fillers_by_unit.get(bridge_target, []):
                            if not any(fillers_equivalent(filler, prev) for prev in resolved_target_fillers):
                                resolved_target_fillers.append(filler)
                    candidate_filler = (
                        str(filler_values[local_idx][unit_idx]).strip()
                        if filler_values is not None
                        and local_idx < len(filler_values)
                        and unit_idx < len(filler_values[local_idx])
                        else ""
                    )
                    if (
                        not candidate_filler
                        or not resolved_target_fillers
                        or not any(
                            fillers_equivalent(candidate_filler, target_filler)
                            for target_filler in resolved_target_fillers
                        )
                    ):
                        continue
                g_core[local_idx] += np.float32(gain)
            else:
                g_support[local_idx] += np.float32(gain)
    return g_core, g_support, unmet_mass.astype(np.float32, copy=False)


def detect_requirement_conflicts(
    *,
    requirement_units: Sequence[RequirementUnit],
    support_scores: np.ndarray,
    filler_values: List[List[str]],
    core_local_indices: np.ndarray,
    frontier_local_indices: np.ndarray,
) -> Dict[int, List[RequirementConflictTrace]]:
    if support_scores.ndim != 2 or not requirement_units:
        return {}
    num_candidates = int(support_scores.shape[0])
    core_local_indices = np.asarray(core_local_indices, dtype=np.int64)
    core_local_indices = core_local_indices[
        (core_local_indices >= 0) & (core_local_indices < num_candidates)
    ]
    frontier_local_indices = np.asarray(frontier_local_indices, dtype=np.int64)
    frontier_local_indices = frontier_local_indices[
        (frontier_local_indices >= 0) & (frontier_local_indices < num_candidates)
    ]
    if core_local_indices.size == 0 or frontier_local_indices.size == 0:
        return {}

    head_fillers: Dict[int, str] = {}
    for unit_idx, unit in enumerate(requirement_units):
        if not unit.is_single_valued:
            continue
        for local_idx in core_local_indices.tolist():
            if float(support_scores[local_idx, unit_idx]) <= 0.0:
                continue
            filler = filler_values[local_idx][unit_idx] if local_idx < len(filler_values) else ""
            if filler:
                head_fillers[unit_idx] = filler
                break

    conflicts_by_local_idx: Dict[int, List[RequirementConflictTrace]] = {}
    for local_idx in frontier_local_indices.tolist():
        for unit_idx, unit in enumerate(requirement_units):
            if not unit.is_single_valued or float(support_scores[local_idx, unit_idx]) <= 0.0:
                continue
            head_filler = head_fillers.get(unit_idx, "")
            candidate_filler = filler_values[local_idx][unit_idx] if local_idx < len(filler_values) else ""
            if not head_filler or not candidate_filler:
                continue
            if fillers_equivalent(head_filler, candidate_filler):
                continue
            conflicts_by_local_idx.setdefault(int(local_idx), []).append(
                RequirementConflictTrace(
                    local_idx=int(local_idx),
                    unit_id=unit.unit_id,
                    slot_family=unit.slot_family,
                    head_filler=head_filler,
                    candidate_filler=candidate_filler,
                )
            )
    return conflicts_by_local_idx


def apply_prefix_guard(
    *,
    ranked_local_indices: List[int],
    baseline_prefix_local_indices: Sequence[int],
    g_core_scores: np.ndarray,
    prefix_guard_k: int,
) -> tuple[List[int], bool, List[int], List[int], Dict[int, float]]:
    prefix_guard_k = max(int(prefix_guard_k), 0)
    ranked_local_index_set = {int(local_idx) for local_idx in ranked_local_indices}
    prefix_candidates: List[int] = []
    for local_idx in list(baseline_prefix_local_indices)[:prefix_guard_k]:
        local_idx = int(local_idx)
        if local_idx in ranked_local_index_set and local_idx not in prefix_candidates:
            prefix_candidates.append(local_idx)
    if not prefix_candidates:
        return list(ranked_local_indices), False, [], [], {}

    g_core_scores = np.asarray(g_core_scores, dtype=np.float32)
    prefix_set = set(prefix_candidates)
    positive_intruders = [
        int(local_idx)
        for local_idx in ranked_local_indices
        if int(local_idx) not in prefix_set
        and int(local_idx) < len(g_core_scores)
        and float(g_core_scores[int(local_idx)]) > 0.0
    ]
    remainder = [
        int(local_idx)
        for local_idx in ranked_local_indices
        if int(local_idx) not in set(positive_intruders) and int(local_idx) not in prefix_set
    ]
    guarded_order = positive_intruders + prefix_candidates + remainder
    prefix_disruption = {
        int(local_idx): 1.0
        for local_idx in ranked_local_indices
        if int(local_idx) not in prefix_set
        and int(local_idx) < len(g_core_scores)
        and float(g_core_scores[int(local_idx)]) <= 0.0
    }
    prefix_before = prefix_candidates
    prefix_after = guarded_order[: len(prefix_candidates)]
    return guarded_order, guarded_order != list(ranked_local_indices), prefix_before, prefix_after, prefix_disruption


def rank_frontier_with_ras(
    *,
    config: StrongestConfig,
    strongest_scores: np.ndarray,
    strongest_order: np.ndarray,
    candidate_indices: np.ndarray,
    final_k: int,
    local_passage_prior: np.ndarray,
    protected_local_indices: np.ndarray,
    selected_local_indices: np.ndarray,
    boundary_local_indices: np.ndarray,
    trace_state: StrongestTraceState,
    metadata: Dict[str, object],
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    diagnostics: Dict[str, object] = {
        "ras_enabled": True,
        "ras_status": "fallback_standard",
        "support_mode": str(metadata.get("ras_support_mode", "lexical")).strip().lower(),
        "embedding_probe_threshold": float(metadata.get("ras_embedding_probe_threshold", 0.35) or 0.35),
    }
    requirement_units, extractor_trace = build_requirement_units(
        query=trace_state.query,
        metadata=metadata,
        max_units=int(config.ras_requirement_max_units),
    )
    diagnostics["extractor_trace"] = dict(extractor_trace)
    diagnostics["requirement_units"] = [asdict(unit) for unit in requirement_units]
    if not requirement_units:
        diagnostics["ras_status"] = "fallback_no_requirements"
        return strongest_order, strongest_scores, diagnostics

    support_tensor, filler_values = _build_local_requirement_support_tensor(
        candidate_indices=candidate_indices,
        selected_local_indices=selected_local_indices,
        requirement_units=requirement_units,
        metadata=metadata,
    )
    support_scores = compute_requirement_support(
        support_tensor=support_tensor,
        requirement_units=requirement_units,
        strict_eligibility=bool(config.ras_core_support_min_eligible),
    )
    core_local_indices = (
        protected_local_indices
        if protected_local_indices.size > 0
        else strongest_order[: min(max(int(config.gbc_head_coverage_k), 1), len(candidate_indices))]
    )
    head_coverage = compute_requirement_set_coverage(
        requirement_units=requirement_units,
        support_scores=support_scores,
        local_indices=core_local_indices,
        filler_values=filler_values,
    )
    g_core_scores, g_support_scores, unmet_mass = compute_unmet_requirement_gain(
        requirement_units=requirement_units,
        support_scores=support_scores,
        head_coverage=head_coverage,
        frontier_local_indices=boundary_local_indices,
        filler_values=filler_values,
        head_local_indices=core_local_indices,
    )
    diagnostics["queries_with_nonzero_g_core"] = bool(np.any(g_core_scores > 0))
    if not np.any((g_core_scores + g_support_scores) > 0):
        diagnostics["ras_status"] = "fallback_zero_requirement_gain"
        return strongest_order, strongest_scores, diagnostics

    strongest_scale = float(np.max(np.clip(strongest_scores, 0.0, None)))
    bonus_signal = np.clip(g_core_scores + g_support_scores, 0.0, None)
    bonus_distribution = normalize_positive_distribution(bonus_signal)
    unmet_scale = float(np.mean(unmet_mass)) if unmet_mass.size > 0 else 0.0
    bonus_scale = strongest_scale * unmet_scale
    final_scores = strongest_scores + (bonus_scale * bonus_distribution).astype(np.float32, copy=False)
    protected_sorted = [
        int(local_idx)
        for local_idx in strongest_order.tolist()
        if int(local_idx) in {int(idx) for idx in protected_local_indices.tolist()}
    ][: min(len(protected_local_indices), max(int(config.gbc_protected_anchor_k), 0), int(final_k))]
    readout_scores = build_protected_readout_scores(final_scores, protected_sorted)

    conflict_map = detect_requirement_conflicts(
        requirement_units=requirement_units,
        support_scores=support_scores,
        filler_values=filler_values,
        core_local_indices=core_local_indices,
        frontier_local_indices=boundary_local_indices,
    )
    vetoed_local_indices = (
        {int(local_idx) for local_idx in conflict_map}
        if bool(config.ras_enable_conflict_veto)
        else set()
    )

    anchor_support = np.clip(np.asarray(local_passage_prior, dtype=np.float32), 0.0, None)
    baseline_prefix_local_indices = strongest_order[: min(max(int(config.ras_prefix_guard_k), 0), len(strongest_order))]
    baseline_prefix_set = {int(idx) for idx in np.asarray(baseline_prefix_local_indices, dtype=np.int64).tolist()}
    protected_sorted_set = {int(idx) for idx in protected_sorted}
    sortable: List[tuple[tuple[float, ...], int]] = []
    for local_idx in strongest_order.tolist():
        local_idx = int(local_idx)
        if local_idx in vetoed_local_indices:
            continue
        prefix_penalty = 1.0 if local_idx not in baseline_prefix_set and float(g_core_scores[local_idx]) <= 0.0 else 0.0
        sortable.append(
            (
                (
                    1.0 if float(g_core_scores[local_idx]) > 0.0 else 0.0,
                    float(g_core_scores[local_idx]),
                    float(g_support_scores[local_idx]),
                    1.0 if local_idx in protected_sorted_set else 0.0,
                    float(anchor_support[local_idx]) if local_idx < len(anchor_support) else 0.0,
                    -prefix_penalty,
                    float(readout_scores[local_idx]),
                ),
                local_idx,
            )
        )
    sortable.sort(key=lambda item: item[0], reverse=True)
    ranked_local_indices = [local_idx for _, local_idx in sortable]
    ranked_local_indices, prefix_guard_triggered, prefix_before, prefix_after, prefix_disruption = apply_prefix_guard(
        ranked_local_indices=ranked_local_indices,
        baseline_prefix_local_indices=baseline_prefix_local_indices,
        g_core_scores=g_core_scores,
        prefix_guard_k=int(config.ras_prefix_guard_k),
    )
    for local_idx in strongest_order.tolist():
        local_idx = int(local_idx)
        if local_idx not in ranked_local_indices:
            ranked_local_indices.append(local_idx)

    final_local_indices = np.asarray(ranked_local_indices[: min(len(ranked_local_indices), 5)], dtype=np.int64)
    head_cov, final_cov, coverage_traces = compute_head_requirement_coverage(
        requirement_units=requirement_units,
        support_scores=support_scores,
        core_local_indices=core_local_indices,
        final_local_indices=final_local_indices,
        filler_values=filler_values,
    )
    protected_set = {int(idx) for idx in protected_sorted}
    final_top3 = ranked_local_indices[:3]
    final_top5 = ranked_local_indices[:5]
    protected_anchor_retention_at3 = (
        float(sum(1 for idx in final_top3 if idx in protected_set)) / float(max(len(protected_set), 1))
        if protected_set
        else 0.0
    )
    protected_anchor_retention_at5 = (
        float(sum(1 for idx in final_top5 if idx in protected_set)) / float(max(len(protected_set), 1))
        if protected_set
        else 0.0
    )
    prefix_disruption_count_at3 = int(sum(1 for pos, idx in enumerate(prefix_before[:3]) if pos >= len(final_top3) or final_top3[pos] != idx))
    prefix_disruption_count_at5 = int(sum(1 for pos, idx in enumerate(prefix_before[:5]) if pos >= len(final_top5) or final_top5[pos] != idx))
    final_conflict_rate_at5 = (
        float(sum(1 for idx in final_top5 if idx in vetoed_local_indices)) / float(max(len(final_top5), 1))
        if final_top5
        else 0.0
    )
    ras_trace = RASReadoutTrace(
        extractor_trace=dict(extractor_trace),
        requirement_units=list(requirement_units),
        coverage=coverage_traces,
        conflicts=[trace for traces in conflict_map.values() for trace in traces],
        g_core_by_local_idx={int(idx): float(g_core_scores[int(idx)]) for idx in range(len(g_core_scores)) if float(g_core_scores[int(idx)]) > 0.0},
        g_support_by_local_idx={int(idx): float(g_support_scores[int(idx)]) for idx in range(len(g_support_scores)) if float(g_support_scores[int(idx)]) > 0.0},
        anchor_support_by_local_idx={int(idx): float(anchor_support[int(idx)]) for idx in range(len(anchor_support)) if float(anchor_support[int(idx)]) > 0.0},
        base_score_by_local_idx={int(idx): float(readout_scores[int(idx)]) for idx in range(len(readout_scores)) if float(readout_scores[int(idx)]) > 0.0},
        prefix_disruption_by_local_idx={int(idx): float(score) for idx, score in prefix_disruption.items()},
        prefix_guard_triggered=bool(prefix_guard_triggered),
        protected_prefix_before=prefix_before,
        protected_prefix_after=prefix_after,
        query_level_repair=bool(prefix_guard_triggered or vetoed_local_indices or np.any(g_core_scores > 0.0)),
    )
    diagnostics.update(
        {
            "ras_status": "applied",
            "coverage": [asdict(trace) for trace in coverage_traces],
            "conflict_pairs": [asdict(trace) for trace in ras_trace.conflicts],
            "g_core_by_local_idx": dict(ras_trace.g_core_by_local_idx),
            "g_support_by_local_idx": dict(ras_trace.g_support_by_local_idx),
            "anchor_support_by_local_idx": dict(ras_trace.anchor_support_by_local_idx),
            "base_score_by_local_idx": dict(ras_trace.base_score_by_local_idx),
            "prefix_disruption_by_local_idx": dict(ras_trace.prefix_disruption_by_local_idx),
            "prefix_guard_triggered": bool(prefix_guard_triggered),
            "protected_prefix_before": list(prefix_before),
            "protected_prefix_after": list(prefix_after),
            "query_level_repair": bool(ras_trace.query_level_repair),
            "core_requirement_coverage_at5": round(
                float(
                    np.mean(
                        [
                            final_cov[idx]
                            for idx, unit in enumerate(requirement_units)
                            if unit.tier == "core"
                        ]
                    )
                )
                if any(unit.tier == "core" for unit in requirement_units)
                else 0.0,
                4,
            ),
            "support_requirement_coverage_at5": round(
                float(
                    np.mean(
                        [
                            final_cov[idx]
                            for idx, unit in enumerate(requirement_units)
                            if unit.tier == "support"
                        ]
                    )
                )
                if any(unit.tier == "support" for unit in requirement_units)
                else 0.0,
                4,
            ),
            "unmet_core_mass_reduction": round(
                float(
                    np.sum(
                        [
                            max(0.0, final_cov[idx] - head_cov[idx])
                            for idx, unit in enumerate(requirement_units)
                            if unit.tier == "core"
                        ]
                    )
                ),
                4,
            ),
            "protected_anchor_retention_at3": round(float(protected_anchor_retention_at3), 4),
            "protected_anchor_retention_at5": round(float(protected_anchor_retention_at5), 4),
            "prefix_disruption_count_at3": int(prefix_disruption_count_at3),
            "prefix_disruption_count_at5": int(prefix_disruption_count_at5),
            "single_value_conflict_rate_at5": round(float(final_conflict_rate_at5), 4),
            "new_conflict_introduced_rate": round(float(1.0 if any(idx in vetoed_local_indices for idx in final_top5) else 0.0), 4),
            "query_level_repair_count": int(1 if ras_trace.query_level_repair else 0),
        }
    )
    return np.asarray(ranked_local_indices, dtype=np.int64), readout_scores, diagnostics


def run_clean_gbc_readout(
    *,
    config: StrongestConfig,
    strongest_scores: np.ndarray,
    strongest_order: np.ndarray,
    candidate_indices: np.ndarray,
    final_k: int,
    protected_local_indices: np.ndarray,
    selected_local_indices: np.ndarray,
    boundary_local_indices: np.ndarray,
    trace_state: StrongestTraceState,
    metadata: Dict[str, object],
    diagnostics: Dict[str, object],
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    query_fact_units = build_query_fact_units(
        trace_state=trace_state,
        metadata={
            **dict(metadata or {}),
            "selected_local_indices": selected_local_indices.tolist(),
        },
    )
    diagnostics["query_fact_count"] = int(len(query_fact_units))
    if not query_fact_units:
        diagnostics["gbc_status"] = "fallback_no_query_facts"
        readout_scores = build_protected_readout_scores(strongest_scores, protected_local_indices.tolist())
        rerank_local_order = np.asarray(np.argsort(readout_scores)[::-1], dtype=np.int64)
        return rerank_local_order, readout_scores, diagnostics

    passage_chunk_ids = metadata.get("passage_chunk_ids", []) or []
    chunk_triples_map = metadata.get("chunk_triples_map", {}) or {}
    if not passage_chunk_ids or not chunk_triples_map:
        diagnostics["gbc_status"] = "fallback_no_openie"
        readout_scores = build_protected_readout_scores(strongest_scores, protected_local_indices.tolist())
        rerank_local_order = np.asarray(np.argsort(readout_scores)[::-1], dtype=np.int64)
        return rerank_local_order, readout_scores, diagnostics

    support_tensor = build_local_query_fact_support_tensor(
        passage_chunk_ids=passage_chunk_ids,
        candidate_indices=candidate_indices,
        selected_local_indices=selected_local_indices,
        chunk_triples_map=chunk_triples_map,
        query_fact_units=query_fact_units,
    )
    core_local_indices = (
        protected_local_indices
        if protected_local_indices.size > 0
        else strongest_order[: min(max(int(config.gbc_head_coverage_k), 1), len(candidate_indices))]
    )
    completion_scores, query_deficit_mass = build_frontier_headcomp_route_passage_scores(
        support_tensor=support_tensor,
        query_fact_units=query_fact_units,
        core_local_indices=core_local_indices,
        frontier_local_indices=boundary_local_indices,
    )
    diagnostics["query_deficit_mass"] = float(query_deficit_mass)
    if query_deficit_mass <= 1e-8 or not np.any(completion_scores > 0):
        diagnostics["gbc_status"] = "fallback_zero_completion"
        readout_scores = build_protected_readout_scores(strongest_scores, protected_local_indices.tolist())
        rerank_local_order = np.asarray(np.argsort(readout_scores)[::-1], dtype=np.int64)
        return rerank_local_order, readout_scores, diagnostics

    gated_completion_scores = np.zeros_like(completion_scores, dtype=np.float32)
    gated_completion_scores[boundary_local_indices] = completion_scores[boundary_local_indices]
    if protected_local_indices.size > 0:
        gated_completion_scores[protected_local_indices] = 0.0
    completion_distribution = normalize_positive_distribution(gated_completion_scores)
    strongest_scale = float(np.max(np.clip(strongest_scores, 0.0, None)))
    bonus_scale = float(query_deficit_mass) * strongest_scale * float(max(config.gbc_bonus_weight, 0.0))
    if bonus_scale <= 0.0 or not np.any(completion_distribution > 0):
        diagnostics["gbc_status"] = "fallback_zero_bonus"
        readout_scores = build_protected_readout_scores(strongest_scores, protected_local_indices.tolist())
        rerank_local_order = np.asarray(np.argsort(readout_scores)[::-1], dtype=np.int64)
        return rerank_local_order, readout_scores, diagnostics

    final_scores = strongest_scores + (bonus_scale * completion_distribution).astype(np.float32, copy=False)
    protected_sorted = [
        int(local_idx)
        for local_idx in strongest_order.tolist()
        if int(local_idx) in {int(idx) for idx in protected_local_indices.tolist()}
    ][: min(len(protected_local_indices), max(int(config.gbc_protected_anchor_k), 0), int(final_k))]
    readout_scores = build_protected_readout_scores(final_scores, protected_sorted)
    rerank_local_order = np.asarray(np.argsort(readout_scores)[::-1], dtype=np.int64)
    diagnostics["gbc_status"] = "applied"
    diagnostics["bonus_scale"] = float(bonus_scale)
    diagnostics["completion_local_indices"] = [
        int(local_idx)
        for local_idx in np.argsort(gated_completion_scores)[::-1].tolist()
        if float(gated_completion_scores[int(local_idx)]) > 0.0
    ][: min(10, len(gated_completion_scores))]
    return rerank_local_order, readout_scores, diagnostics


def apply_clean_gbc_rerank(
    *,
    config: StrongestConfig,
    strongest_scores: np.ndarray,
    strongest_order: np.ndarray,
    candidate_indices: np.ndarray,
    final_k: int,
    local_passage_prior: np.ndarray,
    local_reset_scores: np.ndarray,
    trace_state: StrongestTraceState,
    metadata: Dict[str, object],
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    strongest_scores = np.asarray(strongest_scores, dtype=np.float32)
    strongest_order = np.asarray(strongest_order, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    diagnostics: Dict[str, object] = {
        "rerank_mode": "gbc",
        "gbc_status": "fallback_standard",
        "query_deficit_mass": 0.0,
        "protected_local_indices": [],
        "boundary_local_indices": [],
        "selected_local_indices": [],
        "query_fact_count": 0,
    }

    if strongest_scores.size == 0 or strongest_order.size == 0:
        return strongest_order, strongest_scores, diagnostics
    if trace_state.retrieval_mode != "graph":
        diagnostics["gbc_status"] = "skip_non_graph"
        return strongest_order, strongest_scores, diagnostics

    protected_local_indices = select_protected_anchor_local_indices(
        local_passage_prior=local_passage_prior,
        local_reset_scores=local_reset_scores,
        protected_anchor_k=int(config.gbc_protected_anchor_k),
    )
    protected_local_indices = np.unique(protected_local_indices.astype(np.int64, copy=False))
    selected_local_indices = strongest_order[
        : min(max(int(config.gbc_top_passage_pool_k), 1), len(candidate_indices))
    ]
    diagnostics["selected_local_indices"] = [int(idx) for idx in selected_local_indices.tolist()]
    diagnostics["protected_local_indices"] = [int(idx) for idx in protected_local_indices.tolist()]

    boundary_local_indices = select_boundary_frontier_bonus_local_indices(
        strongest_order=strongest_order,
        selected_local_indices=selected_local_indices,
        protected_local_indices=protected_local_indices,
        final_k=final_k,
        frontier_bonus_top_k=int(config.gbc_frontier_bonus_k),
    )
    diagnostics["boundary_local_indices"] = [int(idx) for idx in boundary_local_indices.tolist()]
    if boundary_local_indices.size == 0:
        diagnostics["gbc_status"] = "fallback_no_boundary"
        readout_scores = build_protected_readout_scores(strongest_scores, protected_local_indices.tolist())
        rerank_local_order = np.asarray(np.argsort(readout_scores)[::-1], dtype=np.int64)
        return rerank_local_order, readout_scores, diagnostics

    if bool(config.ras_enabled):
        rerank_local_order, final_candidate_scores, ras_diagnostics = rank_frontier_with_ras(
            config=config,
            strongest_scores=strongest_scores,
            strongest_order=strongest_order,
            candidate_indices=candidate_indices,
            final_k=final_k,
            local_passage_prior=local_passage_prior,
            protected_local_indices=protected_local_indices,
            selected_local_indices=selected_local_indices,
            boundary_local_indices=boundary_local_indices,
            trace_state=trace_state,
            metadata=metadata,
        )
        diagnostics["ras"] = ras_diagnostics
        if ras_diagnostics.get("ras_status") == "applied":
            diagnostics["gbc_status"] = "applied"
            return rerank_local_order, final_candidate_scores, diagnostics
        diagnostics["ras_fallback_used"] = True

    return run_clean_gbc_readout(
        config=config,
        strongest_scores=strongest_scores,
        strongest_order=strongest_order,
        candidate_indices=candidate_indices,
        final_k=final_k,
        protected_local_indices=protected_local_indices,
        selected_local_indices=selected_local_indices,
        boundary_local_indices=boundary_local_indices,
        trace_state=trace_state,
        metadata=metadata,
        diagnostics=diagnostics,
    )
