import re
from collections import defaultdict
from typing import Dict, Iterable, List, Literal, Sequence, Set, Tuple

import numpy as np

from .misc_utils import CausalRelation, text_processing


CAUSAL_RELATION_TYPES = ("causes", "enables", "prevents")
CausalQueryType = Literal["cause", "effect", "prevention", "non_causal"]
STRUCTURE_RELATION_TYPES = CAUSAL_RELATION_TYPES + ("state_transition",)

PREVENTION_PATTERNS = (
    r"\bprevent\b",
    r"\bprevents\b",
    r"\bavoid\b",
    r"\bstop(?:s|ped|ping)?\b.+?\bfrom\b",
    r"\binhibit\b",
    r"\breduce risk\b",
)
EFFECT_PATTERNS = (
    r"\bwhat (?:does|did|can|could|will|would)\b.+?\b(?:cause|causes|lead to|leads to|result in|results in|enable|enables)\b",
    r"\bwhat (?:was|were|is|are)\b.+?\bcaused by\b",
    r"\bcaused by\b",
    r"\bresults? from\b",
    r"\beffects? of\b",
    r"\bconsequences? of\b",
    r"\beffect\b",
    r"\beffects\b",
    r"\bconsequence\b",
    r"\bwhat happens if\b",
)
CAUSE_PATTERNS = (
    r"\bwhy\b",
    r"\bwhat caused\b",
    r"\bwhat causes\b",
    r"\bwhat leads to\b",
    r"\bwhat results in\b",
    r"\breason\b",
    r"\breason for\b",
    r"\bbecause\b",
    r"\bdue to\b",
    r"\bstems? from\b",
)
WEAK_DIRECTION_PATTERNS = (
    r"\bafter\b",
    r"\bfollowing\b",
    r"\bresulted\b",
    r"\bresulting\b",
    r"\bled to\b",
    r"\bleads to\b",
    r"\bconsequence\b",
    r"\bconsequences\b",
    r"\boutcome\b",
    r"\boutcomes\b",
    r"\bimpact\b",
)
NON_CAUSAL_FACT_PATTERNS = (
    r"^\s*who is\b",
    r"^\s*where is\b",
    r"^\s*when was\b",
    r"^\s*how many\b",
    r"^\s*what is the capital\b",
)


def canonicalize_relation_type(relation_type: str) -> str | None:
    normalized = str(relation_type).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None

    if any(token in normalized for token in ("prevent", "avoid", "stop", "reduce_risk", "inhibit")):
        return "prevents"
    if any(token in normalized for token in ("enable", "allow", "facilitate", "trigger_condition")):
        return "enables"
    if any(token in normalized for token in ("cause", "lead", "result", "because", "due_to", "reason")):
        return "causes"
    if normalized in CAUSAL_RELATION_TYPES:
        return normalized
    return None


def sanitize_causal_relations(raw_relations: Sequence[dict],
                              valid_fact_ids: Iterable[str],
                              confidence_threshold: float = 0.5) -> List[CausalRelation]:
    valid_fact_ids = set(valid_fact_ids)
    dedup: Dict[Tuple[str, str, str], CausalRelation] = {}

    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue

        source_fact_id = str(relation.get("source_fact_id", "")).strip()
        target_fact_id = str(relation.get("target_fact_id", "")).strip()
        relation_type = canonicalize_relation_type(str(relation.get("relation_type", "")))

        try:
            confidence = float(relation.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if source_fact_id not in valid_fact_ids or target_fact_id not in valid_fact_ids:
            continue
        if source_fact_id == target_fact_id or relation_type is None:
            continue
        if confidence < confidence_threshold:
            continue

        edge_key = (source_fact_id, target_fact_id, relation_type)
        existing = dedup.get(edge_key)
        if existing is None or confidence > existing.confidence:
            dedup[edge_key] = CausalRelation(
                source_fact_id=source_fact_id,
                target_fact_id=target_fact_id,
                relation_type=relation_type,
                confidence=confidence,
            )

    return list(dedup.values())


def route_query_type(query: str) -> CausalQueryType:
    text = re.sub(r"\s+", " ", query.lower()).strip()

    if any(re.search(pattern, text) for pattern in PREVENTION_PATTERNS):
        return "prevention"
    if any(re.search(pattern, text) for pattern in EFFECT_PATTERNS):
        return "effect"
    if any(re.search(pattern, text) for pattern in CAUSE_PATTERNS):
        return "cause"
    return "non_causal"


def score_query_causal_intent(query: str) -> float:
    text = re.sub(r"\s+", " ", query.lower()).strip()

    if any(re.search(pattern, text) for pattern in PREVENTION_PATTERNS + EFFECT_PATTERNS + CAUSE_PATTERNS):
        return 1.0

    if any(re.search(pattern, text) for pattern in WEAK_DIRECTION_PATTERNS):
        return 0.6

    if any(re.search(pattern, text) for pattern in NON_CAUSAL_FACT_PATTERNS):
        return 0.0

    return 0.1


def normalize_structure_text(text: str) -> str:
    normalized = text_processing(text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def classify_directed_predicate(predicate: str) -> Tuple[str, bool, float] | None:
    normalized = normalize_structure_text(predicate)
    if not normalized:
        return None

    reverse_specs = (
        (r"\bcaused by\b|\bresult(?:ed|s)? from\b|\bstems? from\b|\bdue to\b", "causes", 1.0),
        (r"\benabled by\b|\ballowed by\b|\bfacilitated by\b", "enables", 0.95),
        (r"\bprevented by\b|\bstopped by\b|\bblocked by\b|\binhibited by\b|\bavoided by\b", "prevents", 0.95),
        (r"\brenamed from\b", "state_transition", 0.80),
    )
    forward_specs = (
        (r"\bcause(?:s|d|ing)?\b|\blead(?:s|ing)? to\b|\bresult(?:s|ed|ing)? in\b|\btrigger(?:s|ed|ing)?\b|\binduce(?:s|d|ing)?\b", "causes", 1.0),
        (r"\benable(?:s|d|ing)?\b|\ballow(?:s|ed|ing)?\b|\bfacilitate(?:s|d|ing)?\b|\bsupport(?:s|ed|ing)?\b", "enables", 0.90),
        (r"\bprevent(?:s|ed|ing)?\b|\bavoid(?:s|ed|ing)?\b|\bstop(?:s|ped|ping)?\b|\bblock(?:s|ed|ing)?\b|\binhibit(?:s|ed|ing)?\b", "prevents", 0.90),
        (r"\bborn in\b|\bdied in\b|\bmoved to\b|\bjoined\b|\bleft\b|\bwon\b|\bcrashed\b|\bfounded\b|\bacquired\b|\bappointed\b|\belected\b|\brenamed to\b", "state_transition", 0.80),
    )

    for pattern, relation_type, confidence in reverse_specs:
        if re.search(pattern, normalized):
            return relation_type, True, confidence
    for pattern, relation_type, confidence in forward_specs:
        if re.search(pattern, normalized):
            return relation_type, False, confidence
    return None


def derive_directed_structure_edge(subject: str,
                                   predicate: str,
                                   object_: str) -> Tuple[str, str, str, float] | None:
    normalized_subject = normalize_structure_text(subject)
    normalized_object = normalize_structure_text(object_)
    if not normalized_subject or not normalized_object or normalized_subject == normalized_object:
        return None

    predicate_info = classify_directed_predicate(predicate)
    if predicate_info is None:
        return None

    relation_type, reverse_direction, confidence = predicate_info
    source = normalized_object if reverse_direction else normalized_subject
    target = normalized_subject if reverse_direction else normalized_object
    if source == target:
        return None
    return source, target, relation_type, confidence


def derive_composed_structure_edges(source_triple: Sequence[str],
                                    target_triple: Sequence[str],
                                    relation_type: str,
                                    confidence: float) -> List[Tuple[str, str, str, float]]:
    canonical_relation = canonicalize_relation_type(relation_type)
    if canonical_relation is None or confidence <= 0:
        return []
    if len(source_triple) != 3 or len(target_triple) != 3:
        return []

    source_subject, _, source_object = [normalize_structure_text(part) for part in source_triple]
    target_subject, _, target_object = [normalize_structure_text(part) for part in target_triple]
    if not all((source_subject, source_object, target_subject, target_object)):
        return []

    composed_edges = []

    if source_object == target_subject and source_subject != target_object:
        composed_edges.append((source_subject, target_object, canonical_relation, min(1.0, confidence)))
    if source_subject == target_object and source_object != target_subject:
        composed_edges.append((source_object, target_subject, canonical_relation, min(1.0, confidence)))
    if source_object == target_object and source_subject != target_subject:
        composed_edges.append((source_subject, target_subject, canonical_relation, min(1.0, confidence * 0.8)))
    if source_subject == target_subject and source_object != target_object:
        composed_edges.append((source_object, target_object, canonical_relation, min(1.0, confidence * 0.8)))

    deduped_edges = {}
    for source, target, edge_relation, edge_confidence in composed_edges:
        if not source or not target or source == target:
            continue
        edge_key = (source, target, edge_relation)
        deduped_edges[edge_key] = max(deduped_edges.get(edge_key, 0.0), edge_confidence)

    return [
        (source, target, edge_relation, edge_confidence)
        for (source, target, edge_relation), edge_confidence in deduped_edges.items()
    ]


def expand_directed_entities(seed_entities: Set[str],
                             adjacency: Dict[str, List[Tuple[str, float, str]]],
                             max_hops: int = 2,
                             max_frontier_size: int = 32) -> Dict[str, float]:
    if not seed_entities or not adjacency or max_hops <= 0:
        return {}

    normalized_seeds = {
        normalize_structure_text(entity)
        for entity in seed_entities
        if normalize_structure_text(entity)
    }
    if not normalized_seeds:
        return {}

    reached_scores: Dict[str, float] = {}
    frontier = {entity: 1.0 for entity in normalized_seeds}
    visited = set(normalized_seeds)

    for hop in range(1, max_hops + 1):
        hop_decay = 1.0 / (hop + 0.25)
        next_frontier: Dict[str, float] = {}
        for source_entity, source_score in frontier.items():
            for target_entity, edge_weight, _ in adjacency.get(source_entity, []):
                normalized_target = normalize_structure_text(target_entity)
                if not normalized_target or normalized_target in normalized_seeds:
                    continue
                candidate_score = source_score * max(edge_weight, 0.0) * hop_decay
                if candidate_score <= 0:
                    continue
                if candidate_score > reached_scores.get(normalized_target, 0.0):
                    reached_scores[normalized_target] = candidate_score
                if normalized_target in visited:
                    continue
                if candidate_score > next_frontier.get(normalized_target, 0.0):
                    next_frontier[normalized_target] = candidate_score

        if not next_frontier:
            break

        limited_frontier = sorted(next_frontier.items(), key=lambda item: item[1], reverse=True)[:max_frontier_size]
        frontier = dict(limited_frontier)
        visited.update(frontier)

    return reached_scores


def score_candidate_docs_by_structure(candidate_doc_ids: Sequence[int],
                                      doc_idx_to_entities: Dict[int, Set[str]],
                                      doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                      seed_entities: Set[str],
                                      adjacency: Dict[str, List[Tuple[str, float, str]]],
                                      max_hops: int = 2,
                                      return_details: bool = False) -> Dict[int, float] | Tuple[Dict[int, float], Dict[str, object]]:
    reachable_scores = expand_directed_entities(seed_entities, adjacency, max_hops=max_hops)
    if not reachable_scores:
        empty = {}
        if return_details:
            return empty, {"total_bridge_edges": 0, "doc_bridge_edge_counts": {}, "scored_doc_count": 0}
        return empty

    normalized_seeds = {
        normalize_structure_text(entity)
        for entity in seed_entities
        if normalize_structure_text(entity)
    }
    if not normalized_seeds:
        empty = {}
        if return_details:
            return empty, {"total_bridge_edges": 0, "doc_bridge_edge_counts": {}, "scored_doc_count": 0}
        return empty

    max_local_structure = max(
        (
            sum(max(edge_weight, 0.0) for _, _, edge_weight, _ in doc_idx_to_edges.get(int(doc_id), []))
            for doc_id in candidate_doc_ids
        ),
        default=0.0,
    )

    raw_scores: Dict[int, float] = {}
    doc_bridge_edge_counts: Dict[int, int] = {}
    for raw_doc_id in candidate_doc_ids:
        doc_id = int(raw_doc_id)
        entities = {
            normalize_structure_text(entity)
            for entity in doc_idx_to_entities.get(doc_id, set())
            if normalize_structure_text(entity)
        }
        if not entities:
            continue

        seed_hits = entities & normalized_seeds
        reachable_hits = {
            entity: reachable_scores[entity]
            for entity in entities
            if entity in reachable_scores
        }
        if not seed_hits and not reachable_hits:
            continue

        seed_support = min(1.0, len(seed_hits) / max(1.0, min(2.0, float(len(normalized_seeds)))))
        reachable_support = 0.0

        bridge_support = 0.0
        local_structure = 0.0
        explicit_bridge_edge_count = 0
        for source, target, edge_weight, _ in doc_idx_to_edges.get(doc_id, []):
            normalized_source = normalize_structure_text(source)
            normalized_target = normalize_structure_text(target)
            positive_weight = max(edge_weight, 0.0)
            if positive_weight <= 0:
                continue
            local_structure += positive_weight

            source_support = 1.0 if normalized_source in normalized_seeds else reachable_scores.get(normalized_source, 0.0)
            target_support = reachable_scores.get(normalized_target, 0.0)
            if source_support > 0 and target_support > 0:
                explicit_bridge_edge_count += 1
                bridge_support = max(bridge_support, min(1.0, positive_weight * max(source_support, target_support)))
                reachable_support = min(1.0, max(reachable_support, sum(reachable_hits.values())))

        if explicit_bridge_edge_count <= 0:
            continue

        structure_density = min(1.0, local_structure / max_local_structure) if max_local_structure > 0 else 0.0
        doc_bridge_edge_counts[doc_id] = explicit_bridge_edge_count
        raw_scores[doc_id] = (
            0.10 * reachable_support
            + 0.80 * bridge_support
            + 0.10 * structure_density
        )

    if not raw_scores:
        empty = {}
        if return_details:
            return empty, {"total_bridge_edges": 0, "doc_bridge_edge_counts": {}, "scored_doc_count": 0}
        return empty

    doc_ids = np.array(list(raw_scores.keys()), dtype=int)
    score_values = np.array([raw_scores[doc_id] for doc_id in doc_ids], dtype=float)
    max_score = float(np.max(score_values))
    if max_score > 0:
        score_values = score_values / max_score
    else:
        score_values = np.zeros_like(score_values)

    normalized_scores = {
        int(doc_id): float(score)
        for doc_id, score in zip(doc_ids.tolist(), score_values.tolist())
        if score > 0
    }
    if not return_details:
        return normalized_scores

    return normalized_scores, {
        "total_bridge_edges": int(sum(doc_bridge_edge_counts.values())),
        "doc_bridge_edge_counts": doc_bridge_edge_counts,
        "scored_doc_count": len(normalized_scores),
    }


def build_transition_matrix(num_nodes: int,
                            adjacency: Dict[int, List[Tuple[int, float]]]) -> Dict[int, List[Tuple[int, float]]]:
    transitions: Dict[int, List[Tuple[int, float]]] = {}
    for source_idx in range(num_nodes):
        neighbors = adjacency.get(source_idx, [])
        if not neighbors:
            continue
        total_weight = sum(max(weight, 0.0) for _, weight in neighbors)
        if total_weight <= 0:
            continue
        transitions[source_idx] = [
            (target_idx, max(weight, 0.0) / total_weight)
            for target_idx, weight in neighbors
            if weight > 0
        ]
    return transitions


def run_personalized_pagerank(num_nodes: int,
                              adjacency: Dict[int, List[Tuple[int, float]]],
                              reset_prob: np.ndarray,
                              damping: float,
                              max_iter: int = 50,
                              tol: float = 1e-6) -> np.ndarray:
    if num_nodes == 0:
        return np.array([], dtype=float)

    scores = np.array(reset_prob, dtype=float)
    scores = np.where(np.isnan(scores) | (scores < 0), 0.0, scores)
    if scores.sum() <= 0:
        return np.zeros(num_nodes, dtype=float)
    scores = scores / scores.sum()
    reset = scores.copy()
    transitions = build_transition_matrix(num_nodes, adjacency)

    for _ in range(max_iter):
        next_scores = (1.0 - damping) * reset
        for source_idx, neighbors in transitions.items():
            source_score = scores[source_idx]
            if source_score <= 0:
                continue
            for target_idx, probability in neighbors:
                next_scores[target_idx] += damping * source_score * probability

        if np.linalg.norm(next_scores - scores, ord=1) < tol:
            scores = next_scores
            break
        scores = next_scores

    if scores.sum() <= 0:
        return np.zeros(num_nodes, dtype=float)
    return scores / scores.sum()


def aggregate_doc_scores(num_docs: int,
                         fact_scores: np.ndarray,
                         fact_id_by_index: Sequence[str],
                         fact_id_to_doc_idxs: Dict[str, Sequence[int]]) -> np.ndarray:
    doc_scores = np.zeros(num_docs, dtype=float)
    for fact_idx, fact_score in enumerate(fact_scores):
        if fact_score <= 0:
            continue
        fact_id = fact_id_by_index[fact_idx]
        for doc_idx in fact_id_to_doc_idxs.get(fact_id, []):
            doc_scores[doc_idx] = max(doc_scores[doc_idx], float(fact_score))
    return doc_scores


def filter_adjacency_by_relation(adjacency: Dict[int, List[Tuple[int, float, str]]],
                                 allowed_relations: Iterable[str]) -> Dict[int, List[Tuple[int, float]]]:
    allowed_relations = set(allowed_relations)
    filtered: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for source_idx, neighbors in adjacency.items():
        for target_idx, weight, relation_type in neighbors:
            if relation_type in allowed_relations:
                filtered[source_idx].append((target_idx, weight))
    return dict(filtered)
