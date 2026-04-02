from __future__ import annotations

import hashlib
import json
import math
import copy
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag.utils.causal_utils import normalize_structure_text


REQUIREMENT_CACHE_VERSION = "pcrs_rag_v1"
DEFAULT_REQUIREMENT_ANNOTATION_POOL_K = 50
DEFAULT_REQUIREMENT_SMOOTH_TAU = 0.15
DEFAULT_REQUIREMENT_CF_TAU = 0.10
DEFAULT_REQUIREMENT_MAX_COUNTERFACTUALS = 3
REQUIREMENT_TYPES = ("anchor", "bridge", "decision")
REQUIREMENT_MATCHER_FEATURE_NAMES = [
    "base_score",
    "rank_fraction",
    "reciprocal_rank",
    "selected_count_fraction",
    "anchor_support_after",
    "bridge_support_after",
    "decision_support_after",
    "support_completeness_after",
    "support_completeness_gain",
    "counterfactual_leakage_after",
    "counterfactual_leakage_gain",
    "utility_margin_after",
    "utility_margin_gain",
    "doc_positive_max",
    "doc_positive_mean",
    "doc_counterfactual_max",
    "doc_counterfactual_mean",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAPITALIZED_SPAN_RE = re.compile(r"\b(?:[A-Z][\w'.-]*)(?:\s+(?:[A-Z][\w'.-]*))*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "why", "with",
}


def tokenize_text(text: str | None) -> List[str]:
    normalized = normalize_structure_text(text or "")
    if not normalized:
        return []
    return [token for token in _TOKEN_RE.findall(normalized) if len(token) > 1]


def unique_ordered_texts(values: Sequence[str] | Set[str] | None) -> List[str]:
    deduped: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        normalized = normalize_structure_text(str(value))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def canonicalize_requirement_positions(selected_positions: Sequence[int] | None) -> tuple[int, ...]:
    return tuple(sorted({
        int(pos)
        for pos in (selected_positions or [])
        if int(pos) >= 0
    }))


def stable_question_key(question: str) -> str:
    normalized = normalize_structure_text(question or "")
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def extract_question_entities(question: str,
                              fallback_entities: Sequence[str] | Set[str] | None = None,
                              max_entities: int = 4) -> List[str]:
    entities: List[str] = []
    seen: Set[str] = set()
    for match in _CAPITALIZED_SPAN_RE.finditer(question or ""):
        candidate = normalize_structure_text(match.group(0))
        if not candidate or candidate in seen or len(candidate) < 3:
            continue
        seen.add(candidate)
        entities.append(candidate)
        if len(entities) >= max_entities:
            return entities

    for entity in unique_ordered_texts(fallback_entities):
        if entity in seen or len(entity) < 3:
            continue
        entities.append(entity)
        if len(entities) >= max_entities:
            break
    return entities


def extract_question_focus_terms(question: str,
                                 anchor_entities: Sequence[str] | Set[str] | None = None,
                                 max_terms: int = 6) -> List[str]:
    tokens = tokenize_text(question)
    entity_tokens: Set[str] = set()
    for entity in anchor_entities or []:
        entity_tokens.update(tokenize_text(entity))

    filtered = [
        token for token in tokens
        if token not in _STOPWORDS and token not in entity_tokens
    ]
    return filtered[:max_terms]


def guess_answer_type_tokens(question: str) -> List[str]:
    lowered = normalize_structure_text(question or "")
    if not lowered:
        return []
    if lowered.startswith("where"):
        return ["location", "place", "city", "country"]
    if lowered.startswith("when"):
        return ["date", "year", "time"]
    if lowered.startswith("who") or lowered.startswith("whom"):
        return ["person", "name"]
    if lowered.startswith("how many") or lowered.startswith("how much"):
        return ["number", "count", "amount"]
    if lowered.startswith("which"):
        return extract_question_focus_terms(question, max_terms=4)
    if lowered.startswith("what year"):
        return ["year", "date", "time"]
    if lowered.startswith("what city"):
        return ["city", "location", "place"]
    if lowered.startswith("what country"):
        return ["country", "location", "place"]
    return extract_question_focus_terms(question, max_terms=4)


def build_requirement(requirement_id: str,
                      requirement_type: str,
                      text: str,
                      entities: Sequence[str] | Set[str] | None = None,
                      answer_type_tokens: Sequence[str] | None = None) -> Dict[str, Any]:
    normalized_text = normalize_structure_text(text or "")
    normalized_entities = unique_ordered_texts(entities)
    tokens = [
        token for token in tokenize_text(normalized_text)
        if token not in _STOPWORDS
    ]
    return {
        "requirement_id": str(requirement_id),
        "type": str(requirement_type),
        "text": str(text),
        "normalized_text": normalized_text,
        "tokens": tokens,
        "entities": normalized_entities,
        "answer_type_tokens": list(answer_type_tokens or []),
    }


def build_positive_requirements(question: str,
                                seed_entities: Sequence[str] | Set[str] | None,
                                question_entities: Sequence[str] | Set[str] | None) -> List[Dict[str, Any]]:
    anchor_entities = extract_question_entities(
        question=question,
        fallback_entities=question_entities or seed_entities,
        max_entities=3,
    )
    focus_terms = extract_question_focus_terms(question, anchor_entities=anchor_entities)
    answer_type_tokens = guess_answer_type_tokens(question)

    requirements: List[Dict[str, Any]] = []
    for idx, entity in enumerate(anchor_entities[:2]):
        requirements.append(build_requirement(
            requirement_id=f"anchor_{idx}",
            requirement_type="anchor",
            text=entity,
            entities=[entity],
        ))

    bridge_entities = anchor_entities[:2]
    bridge_terms = list(dict.fromkeys([*focus_terms[:4], *answer_type_tokens[:2]]))
    bridge_text_parts = [part for part in [*bridge_entities, *bridge_terms] if part]
    if bridge_text_parts:
        requirements.append(build_requirement(
            requirement_id="bridge_0",
            requirement_type="bridge",
            text=" bridge ".join(bridge_text_parts),
            entities=bridge_entities,
        ))

    decision_text_parts = list(dict.fromkeys([*answer_type_tokens[:4], *focus_terms[:4], *anchor_entities[:1]]))
    if decision_text_parts:
        requirements.append(build_requirement(
            requirement_id="decision_0",
            requirement_type="decision",
            text=" ".join(decision_text_parts),
            entities=anchor_entities[:1],
            answer_type_tokens=answer_type_tokens,
        ))
    else:
        requirements.append(build_requirement(
            requirement_id="decision_0",
            requirement_type="decision",
            text=question,
            entities=anchor_entities[:1],
            answer_type_tokens=answer_type_tokens,
        ))

    return requirements


def replace_anchor_in_requirement(requirement: Dict[str, Any],
                                  source_anchor: str | None,
                                  replacement_anchor: str) -> Dict[str, Any]:
    normalized_source = normalize_structure_text(source_anchor or "")
    normalized_replacement = normalize_structure_text(replacement_anchor)
    text = str(requirement.get("text", ""))
    if normalized_source:
        text = re.sub(
            re.escape(source_anchor or ""),
            replacement_anchor,
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        text = f"{replacement_anchor} {text}".strip()

    updated_entities = [
        normalized_replacement if normalize_structure_text(entity) == normalized_source else entity
        for entity in requirement.get("entities", [])
    ]
    if requirement.get("type") == "anchor":
        updated_entities = [normalized_replacement]
        text = replacement_anchor
    return build_requirement(
        requirement_id=str(requirement.get("requirement_id", "req")),
        requirement_type=str(requirement.get("type", "bridge")),
        text=text,
        entities=updated_entities or [normalized_replacement],
        answer_type_tokens=requirement.get("answer_type_tokens", []),
    )


def build_counterfactual_requirement_sets(question: str,
                                          positive_requirements: Sequence[Dict[str, Any]],
                                          pool_entities: Sequence[str] | Set[str] | None,
                                          max_sets: int = DEFAULT_REQUIREMENT_MAX_COUNTERFACTUALS) -> List[Dict[str, Any]]:
    anchor_entities = [
        str(requirement.get("entities", [""])[0]).strip()
        for requirement in positive_requirements
        if requirement.get("type") == "anchor" and requirement.get("entities")
    ]
    source_anchor = anchor_entities[0] if anchor_entities else None
    positive_entity_set = set(unique_ordered_texts(anchor_entities))

    candidate_counter = Counter(unique_ordered_texts(pool_entities))
    replacements: List[str] = []
    for entity, _ in candidate_counter.most_common():
        if entity in positive_entity_set or len(entity) < 3:
            continue
        replacements.append(entity)
        if len(replacements) >= max(int(max_sets), 1):
            break

    if not replacements:
        fallback_terms = extract_question_focus_terms(question, anchor_entities=anchor_entities)
        if fallback_terms:
            replacements = [f"counterfactual {' '.join(fallback_terms[:2])}".strip()]

    counterfactual_sets: List[Dict[str, Any]] = []
    for idx, replacement in enumerate(replacements[:max(int(max_sets), 1)]):
        requirements = [
            replace_anchor_in_requirement(requirement, source_anchor=source_anchor, replacement_anchor=replacement)
            for requirement in positive_requirements
        ]
        counterfactual_sets.append({
            "cf_id": f"cf_{idx}",
            "transform": "entity_swap",
            "replacement_anchor": replacement,
            "requirements": requirements,
        })
    return counterfactual_sets


def score_requirement_coverage(requirement: Dict[str, Any],
                               doc_title: str,
                               doc_body: str,
                               doc_entities: Sequence[str] | Set[str] | None) -> float:
    req_tokens = set(requirement.get("tokens", []))
    req_entities = set(unique_ordered_texts(requirement.get("entities", [])))
    title_tokens = set(tokenize_text(doc_title))
    body_tokens = set(tokenize_text(doc_body))
    doc_entity_set = set(unique_ordered_texts(doc_entities))
    normalized_doc_text = normalize_structure_text(f"{doc_title} {doc_body}")

    phrase_match = 0.0
    normalized_requirement_text = str(requirement.get("normalized_text", "")).strip()
    if normalized_requirement_text and normalized_requirement_text in normalized_doc_text:
        phrase_match = 1.0

    entity_overlap = (
        len(req_entities & doc_entity_set) / float(max(1, len(req_entities)))
        if req_entities else 0.0
    )
    title_overlap = (
        len(req_tokens & title_tokens) / float(max(1, len(req_tokens)))
        if req_tokens else 0.0
    )
    body_overlap = (
        len(req_tokens & body_tokens) / float(max(1, len(req_tokens)))
        if req_tokens else 0.0
    )

    answer_type_tokens = set(requirement.get("answer_type_tokens", []))
    answer_type_overlap = (
        len(answer_type_tokens & (title_tokens | body_tokens)) / float(max(1, len(answer_type_tokens)))
        if answer_type_tokens else 0.0
    )
    lexical_score = 0.65 * title_overlap + 0.35 * body_overlap
    if requirement.get("type") == "decision":
        lexical_score = min(1.0, lexical_score + 0.20 * answer_type_overlap)

    return round(float(min(1.0, max(phrase_match, entity_overlap, lexical_score))), 6)


def build_requirement_doc_annotation(pool_position: int,
                                     doc_text: str,
                                     doc_entities: Sequence[str] | Set[str] | None,
                                     positive_requirements: Sequence[Dict[str, Any]],
                                     counterfactual_sets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    doc_title, doc_body = (str(doc_text).split("\n", 1) + [""])[:2]
    normalized_doc_entities = unique_ordered_texts(doc_entities)

    positive_scores = {
        str(requirement["requirement_id"]): score_requirement_coverage(
            requirement=requirement,
            doc_title=doc_title,
            doc_body=doc_body,
            doc_entities=normalized_doc_entities,
        )
        for requirement in positive_requirements
    }

    counterfactual_scores: Dict[str, Dict[str, float]] = {}
    counterfactual_set_scores: Dict[str, float] = {}
    for cf_set in counterfactual_sets:
        requirement_scores = {
            str(requirement["requirement_id"]): score_requirement_coverage(
                requirement=requirement,
                doc_title=doc_title,
                doc_body=doc_body,
                doc_entities=normalized_doc_entities,
            )
            for requirement in cf_set.get("requirements", [])
        }
        counterfactual_scores[str(cf_set["cf_id"])] = requirement_scores
        counterfactual_set_scores[str(cf_set["cf_id"])] = round(
            float(np.mean(list(requirement_scores.values()))) if requirement_scores else 0.0,
            6,
        )

    return {
        "pool_position": int(pool_position),
        "doc_title": doc_title,
        "positive_requirement_scores": positive_scores,
        "counterfactual_requirement_scores": counterfactual_scores,
        "counterfactual_set_scores": counterfactual_set_scores,
    }


def build_requirement_cache_entry(query_index: int,
                                  question: str,
                                  pool_docs: Sequence[str],
                                  pool_doc_entities: Sequence[Sequence[str] | Set[str]],
                                  seed_entities: Sequence[str] | Set[str] | None,
                                  question_entities: Sequence[str] | Set[str] | None,
                                  annotation_pool_k: int = DEFAULT_REQUIREMENT_ANNOTATION_POOL_K) -> Dict[str, Any]:
    positive_requirements = build_positive_requirements(
        question=question,
        seed_entities=seed_entities,
        question_entities=question_entities,
    )
    top_pool_entities: List[str] = []
    for entities in pool_doc_entities[:max(int(annotation_pool_k), 0)]:
        top_pool_entities.extend(unique_ordered_texts(entities))
    counterfactual_sets = build_counterfactual_requirement_sets(
        question=question,
        positive_requirements=positive_requirements,
        pool_entities=top_pool_entities,
    )

    doc_annotations: List[Dict[str, Any]] = []
    effective_annotation_pool_k = min(len(pool_docs), max(int(annotation_pool_k), 0))
    for pool_position in range(effective_annotation_pool_k):
        doc_annotations.append(build_requirement_doc_annotation(
            pool_position=pool_position,
            doc_text=str(pool_docs[pool_position]),
            doc_entities=pool_doc_entities[pool_position] if pool_position < len(pool_doc_entities) else [],
            positive_requirements=positive_requirements,
            counterfactual_sets=counterfactual_sets,
        ))

    return {
        "query_index": int(query_index),
        "question": question,
        "question_key": stable_question_key(question),
        "annotation_pool_k": int(effective_annotation_pool_k),
        "seed_entities": list(unique_ordered_texts(seed_entities)),
        "question_entities": list(unique_ordered_texts(question_entities)),
        "positive_requirements": positive_requirements,
        "counterfactual_sets": counterfactual_sets,
        "pool_titles": [
            str(doc_text).split("\n", 1)[0].strip()
            for doc_text in pool_docs[:effective_annotation_pool_k]
        ],
        "doc_annotations": doc_annotations,
        "diagnostics": {
            "positive_requirement_count": int(len(positive_requirements)),
            "counterfactual_set_count": int(len(counterfactual_sets)),
            "nonempty_counterfactual_rate": round(float(len(counterfactual_sets) > 0), 4),
        },
    }


def save_requirement_cache(path: str | Path, payload: Dict[str, Any]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp_path.replace(target_path)


def load_requirement_cache(path: str | Path) -> Dict[str, Any]:
    cache_payload = json.loads(Path(path).read_text())
    if str(cache_payload.get("version", "")).strip() != REQUIREMENT_CACHE_VERSION:
        raise ValueError(
            f"Unsupported requirement cache version at {path}: {cache_payload.get('version')}"
        )
    entries = cache_payload.get("queries")
    if not isinstance(entries, list):
        raise ValueError(f"Invalid requirement cache payload at {path}: missing queries")

    entries_by_question: Dict[str, Dict[str, Any]] = {}
    entries_by_index: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        question = str(entry.get("question", "")).strip()
        query_index = int(entry.get("query_index", -1))
        if question:
            if question in entries_by_question:
                raise ValueError(f"Duplicate question in requirement cache: {question[:120]}")
            entries_by_question[question] = entry
        if query_index >= 0:
            entries_by_index[query_index] = entry

    cache_payload["entries_by_question"] = entries_by_question
    cache_payload["entries_by_index"] = entries_by_index
    return cache_payload


def resolve_requirement_cache_entry(cache_payload: Dict[str, Any],
                                    question: str,
                                    query_index: int | None = None) -> Dict[str, Any]:
    entry = cache_payload.get("entries_by_question", {}).get(str(question))
    if entry is not None:
        return entry
    if query_index is not None and int(query_index) in cache_payload.get("entries_by_index", {}):
        return cache_payload["entries_by_index"][int(query_index)]
    raise KeyError(f"Missing requirement cache entry for question: {str(question)[:160]}")


def validate_requirement_cache_entry(cache_entry: Dict[str, Any],
                                     pool_titles: Sequence[str]) -> None:
    cached_titles = list(cache_entry.get("pool_titles", []))
    compare_length = min(len(cached_titles), len(pool_titles))
    for idx in range(compare_length):
        if str(cached_titles[idx]).strip() != str(pool_titles[idx]).strip():
            raise ValueError(
                "Requirement cache pool mismatch at position "
                f"{idx}: cache={cached_titles[idx]!r} current={pool_titles[idx]!r}"
            )


def align_requirement_cache_entry_to_pool(cache_entry: Dict[str, Any],
                                          pool_titles: Sequence[str],
                                          pool_docs: Sequence[str] | None = None,
                                          pool_doc_entities: Sequence[Sequence[str] | Set[str]] | None = None) -> Dict[str, Any]:
    cached_titles = list(cache_entry.get("pool_titles", []))
    cached_annotation_pool_k = int(cache_entry.get("annotation_pool_k", 0) or 0)
    if pool_docs is not None:
        effective_length = min(len(pool_titles), len(pool_docs))
        if pool_doc_entities is not None:
            effective_length = min(effective_length, len(pool_doc_entities))
    else:
        effective_length = min(
            cached_annotation_pool_k,
            len(cached_titles),
            len(pool_titles),
        )
    if effective_length <= 0:
        return cache_entry

    normalized_pool_titles = [str(title).strip() for title in pool_titles[:effective_length]]
    if (
        effective_length <= cached_annotation_pool_k
        and cached_titles[:effective_length] == normalized_pool_titles
    ):
        return cache_entry

    annotations_by_title: Dict[str, List[Dict[str, Any]]] = {}
    for annotation in cache_entry.get("doc_annotations", []):
        title = str(annotation.get("doc_title", "")).strip()
        annotations_by_title.setdefault(title, []).append(annotation)

    aligned_annotations: List[Dict[str, Any]] = []
    missing_titles: List[str] = []
    rebuilt_titles: List[str] = []
    positive_requirements = list(cache_entry.get("positive_requirements", []))
    counterfactual_sets = list(cache_entry.get("counterfactual_sets", []))
    for pool_position, raw_title in enumerate(pool_titles[:effective_length]):
        title = str(raw_title).strip()
        candidates = annotations_by_title.get(title, [])
        if candidates:
            aligned_annotation = copy.deepcopy(candidates.pop(0))
            aligned_annotation["pool_position"] = int(pool_position)
            aligned_annotation["doc_title"] = title
        elif pool_docs is not None and pool_position < len(pool_docs):
            doc_entities = []
            if pool_doc_entities is not None and pool_position < len(pool_doc_entities):
                doc_entities = pool_doc_entities[pool_position]
            aligned_annotation = build_requirement_doc_annotation(
                pool_position=pool_position,
                doc_text=str(pool_docs[pool_position]),
                doc_entities=doc_entities,
                positive_requirements=positive_requirements,
                counterfactual_sets=counterfactual_sets,
            )
            rebuilt_titles.append(title)
        else:
            missing_titles.append(title)
            continue
        aligned_annotations.append(aligned_annotation)

    if missing_titles:
        raise ValueError(
            "Requirement cache title alignment failed; missing "
            f"{len(missing_titles)} titles, first={missing_titles[0]!r}"
        )

    aligned_entry = copy.deepcopy(cache_entry)
    aligned_entry["pool_titles"] = normalized_pool_titles
    aligned_entry["doc_annotations"] = aligned_annotations
    aligned_entry["annotation_pool_k"] = int(effective_length)
    diagnostics = dict(aligned_entry.get("diagnostics", {}) or {})
    diagnostics["title_aligned_from_cache"] = True
    diagnostics["title_alignment_size"] = int(effective_length)
    diagnostics["title_alignment_rebuilt_count"] = int(len(rebuilt_titles))
    if rebuilt_titles:
        diagnostics["title_alignment_rebuilt_first_title"] = rebuilt_titles[0]
    aligned_entry["diagnostics"] = diagnostics
    return aligned_entry


def _smooth_min(values: Sequence[float], tau: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    clipped_tau = max(float(tau), 1e-6)
    max_neg = max(-float(value) / clipped_tau for value in values)
    mean_exp = sum(math.exp((-float(value) / clipped_tau) - max_neg) for value in values) / float(len(values))
    return float(-clipped_tau * (math.log(max(mean_exp, 1e-12)) + max_neg))


def _soft_worst_case(values: Sequence[float], tau: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    clipped_tau = max(float(tau), 1e-6)
    max_value = max(float(value) for value in values)
    mean_exp = sum(math.exp((float(value) - max_value) / clipped_tau) for value in values) / float(len(values))
    return float(max_value + clipped_tau * math.log(max(mean_exp, 1e-12)))


def _requirement_coverage_for_positions(requirement_scores: Sequence[float]) -> float:
    if not requirement_scores:
        return 0.0
    residual = 1.0
    for score in requirement_scores:
        residual *= max(0.0, 1.0 - float(score))
    return float(1.0 - residual)


def compute_requirement_state_metrics(cache_entry: Dict[str, Any],
                                      selected_positions: Sequence[int],
                                      smooth_tau: float = DEFAULT_REQUIREMENT_SMOOTH_TAU,
                                      counterfactual_tau: float = DEFAULT_REQUIREMENT_CF_TAU) -> Dict[str, float]:
    annotations_by_position = {
        int(annotation["pool_position"]): annotation
        for annotation in cache_entry.get("doc_annotations", [])
    }
    unique_positions = list(canonicalize_requirement_positions(selected_positions))

    group_values: Dict[str, List[float]] = {group: [] for group in REQUIREMENT_TYPES}
    for requirement in cache_entry.get("positive_requirements", []):
        requirement_id = str(requirement.get("requirement_id", ""))
        scores = [
            float(
                annotations_by_position[pos]
                .get("positive_requirement_scores", {})
                .get(requirement_id, 0.0)
            )
            for pos in unique_positions
            if pos in annotations_by_position
        ]
        group_values[str(requirement.get("type", "bridge"))].append(
            _requirement_coverage_for_positions(scores)
        )

    group_means: Dict[str, float] = {}
    for group_name in REQUIREMENT_TYPES:
        values = group_values.get(group_name, [])
        group_means[group_name] = float(np.mean(values)) if values else 0.0

    active_group_values = [
        group_means[group_name]
        for group_name in REQUIREMENT_TYPES
        if group_values.get(group_name)
    ]
    support_completeness = _smooth_min(active_group_values, tau=smooth_tau) if active_group_values else 0.0

    counterfactual_supports: List[float] = []
    for cf_set in cache_entry.get("counterfactual_sets", []):
        requirement_coverages: List[float] = []
        for requirement in cf_set.get("requirements", []):
            requirement_id = str(requirement.get("requirement_id", ""))
            scores = [
                float(
                    annotations_by_position[pos]
                    .get("counterfactual_requirement_scores", {})
                    .get(str(cf_set.get("cf_id", "")), {})
                    .get(requirement_id, 0.0)
                )
                for pos in unique_positions
                if pos in annotations_by_position
            ]
            requirement_coverages.append(_requirement_coverage_for_positions(scores))
        counterfactual_supports.append(
            float(np.mean(requirement_coverages)) if requirement_coverages else 0.0
        )

    counterfactual_leakage = _soft_worst_case(counterfactual_supports, tau=counterfactual_tau) if counterfactual_supports else 0.0
    utility_margin = float(support_completeness - counterfactual_leakage)
    utopia_distance = float(math.sqrt((1.0 - support_completeness) ** 2 + counterfactual_leakage ** 2))

    return {
        "anchor_support": float(group_means["anchor"]),
        "bridge_support": float(group_means["bridge"]),
        "decision_support": float(group_means["decision"]),
        "support_completeness": float(support_completeness),
        "counterfactual_leakage": float(counterfactual_leakage),
        "utility_margin": float(utility_margin),
        "utopia_distance": float(utopia_distance),
        "active_group_count": float(len(active_group_values)),
        "counterfactual_count": float(len(counterfactual_supports)),
        "selected_annotation_count": float(sum(1 for pos in unique_positions if pos in annotations_by_position)),
    }


def is_pareto_dominated(candidate_metrics: Dict[str, float],
                        other_metrics: Dict[str, float],
                        eps: float = 1e-9) -> bool:
    other_support = float(other_metrics.get("support_completeness", 0.0))
    other_leakage = float(other_metrics.get("counterfactual_leakage", 0.0))
    candidate_support = float(candidate_metrics.get("support_completeness", 0.0))
    candidate_leakage = float(candidate_metrics.get("counterfactual_leakage", 0.0))
    no_worse = other_support >= candidate_support - float(eps) and other_leakage <= candidate_leakage + float(eps)
    strictly_better = other_support > candidate_support + float(eps) or other_leakage < candidate_leakage - float(eps)
    return bool(no_worse and strictly_better)


def compute_requirement_candidate_feature_rows(cache_entry: Dict[str, Any],
                                               selected_positions: Sequence[int],
                                               candidate_positions: Sequence[int],
                                               normalized_base_scores: np.ndarray,
                                               qa_top_k: int,
                                               smooth_tau: float = DEFAULT_REQUIREMENT_SMOOTH_TAU,
                                               counterfactual_tau: float = DEFAULT_REQUIREMENT_CF_TAU) -> List[Dict[str, float | int]]:
    current_metrics = compute_requirement_state_metrics(
        cache_entry=cache_entry,
        selected_positions=selected_positions,
        smooth_tau=smooth_tau,
        counterfactual_tau=counterfactual_tau,
    )
    annotations_by_position = {
        int(annotation["pool_position"]): annotation
        for annotation in cache_entry.get("doc_annotations", [])
    }
    pool_size = len(annotations_by_position)
    rows: List[Dict[str, float | int]] = []
    for pos in candidate_positions:
        annotation = annotations_by_position.get(int(pos), {})
        next_positions = list(selected_positions) + [int(pos)]
        next_metrics = compute_requirement_state_metrics(
            cache_entry=cache_entry,
            selected_positions=next_positions,
            smooth_tau=smooth_tau,
            counterfactual_tau=counterfactual_tau,
        )
        positive_scores = list((annotation.get("positive_requirement_scores") or {}).values())
        counterfactual_set_scores = list((annotation.get("counterfactual_set_scores") or {}).values())
        rows.append({
            "pool_position": int(pos),
            "base_score": float(normalized_base_scores[pos]) if 0 <= int(pos) < len(normalized_base_scores) else 0.0,
            "rank_fraction": 1.0 - (float(pos) / max(1, pool_size - 1)) if pool_size > 1 else 1.0,
            "reciprocal_rank": 1.0 / float(int(pos) + 1),
            "selected_count_fraction": len(selected_positions) / float(max(1, qa_top_k)),
            "anchor_support_after": float(next_metrics["anchor_support"]),
            "bridge_support_after": float(next_metrics["bridge_support"]),
            "decision_support_after": float(next_metrics["decision_support"]),
            "support_completeness_after": float(next_metrics["support_completeness"]),
            "support_completeness_gain": float(next_metrics["support_completeness"] - current_metrics["support_completeness"]),
            "counterfactual_leakage_after": float(next_metrics["counterfactual_leakage"]),
            "counterfactual_leakage_gain": float(next_metrics["counterfactual_leakage"] - current_metrics["counterfactual_leakage"]),
            "utility_margin_after": float(next_metrics["utility_margin"]),
            "utility_margin_gain": float(next_metrics["utility_margin"] - current_metrics["utility_margin"]),
            "doc_positive_max": float(max(positive_scores) if positive_scores else 0.0),
            "doc_positive_mean": float(np.mean(positive_scores) if positive_scores else 0.0),
            "doc_counterfactual_max": float(max(counterfactual_set_scores) if counterfactual_set_scores else 0.0),
            "doc_counterfactual_mean": float(np.mean(counterfactual_set_scores) if counterfactual_set_scores else 0.0),
        })
    return rows


def requirement_feature_rows_to_matrix(feature_rows: Sequence[Dict[str, float | int]]) -> np.ndarray:
    if not feature_rows:
        return np.zeros((0, len(REQUIREMENT_MATCHER_FEATURE_NAMES)), dtype=float)
    return np.asarray([
        [float(row.get(feature_name, 0.0) or 0.0) for feature_name in REQUIREMENT_MATCHER_FEATURE_NAMES]
        for row in feature_rows
    ], dtype=float)


def load_requirement_model_bundle(model_path: str | Path) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid requirement matcher bundle at {model_path}")
    feature_names = list(bundle.get("feature_names") or [])
    if feature_names and feature_names != REQUIREMENT_MATCHER_FEATURE_NAMES:
        raise ValueError(
            "Requirement matcher feature mismatch: "
            f"expected {REQUIREMENT_MATCHER_FEATURE_NAMES}, got {feature_names}"
        )
    bundle.setdefault("feature_names", list(REQUIREMENT_MATCHER_FEATURE_NAMES))
    bundle.setdefault("model_path", str(model_path))
    return bundle
