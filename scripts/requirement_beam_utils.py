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
NEED_UNIT_CACHE_VERSION = "pcrs_rag_v2_need_units"
SUPPORTED_REQUIREMENT_CACHE_VERSIONS = {
    REQUIREMENT_CACHE_VERSION,
    NEED_UNIT_CACHE_VERSION,
}
DEFAULT_REQUIREMENT_ANNOTATION_POOL_K = 50
DEFAULT_REQUIREMENT_SMOOTH_TAU = 0.15
DEFAULT_REQUIREMENT_CF_TAU = 0.10
DEFAULT_REQUIREMENT_MAX_COUNTERFACTUALS = 3
DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS = 6
REQUIREMENT_TYPES = ("anchor", "bridge", "decision")
NEED_UNIT_TYPES = ("entity_locator", "relation_hop", "constraint_check", "answer_slot")
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
NEED_UNIT_MATCHER_FEATURE_NAMES = [
    "base_score",
    "rank_fraction",
    "reciprocal_rank",
    "selected_count_fraction",
    "entity_locator_support_after",
    "relation_hop_support_after",
    "constraint_check_support_after",
    "answer_slot_support_after",
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
    "doc_contradiction_max",
    "doc_contradiction_mean",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAPITALIZED_SPAN_RE = re.compile(r"\b(?:[A-Z][\w'.-]*)(?:\s+(?:[A-Z][\w'.-]*))*")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "why", "with",
}
_WH_WORDS = {"who", "what", "when", "where", "which", "how", "whom", "why"}
_TEMPORAL_KEYWORDS = {"before", "after", "earlier", "later", "first", "last"}
_NEGATION_KEYWORDS = {"not", "except", "excluding", "without"}
_COMPARATIVE_KEYWORDS = {"larger", "smaller", "higher", "lower", "largest", "smallest"}
_COMMON_ROLE_PREDICATE_MAP = {
    "director": "actor",
    "actor": "director",
    "mascot": "founder",
    "founder": "mascot",
    "capital": "birthplace",
    "birthplace": "capital",
    "president": "founder",
    "parent_company": "headquartered_in",
    "headquartered_in": "parent_company",
    "author": "editor",
    "editor": "author",
    "coach": "captain",
    "captain": "coach",
}
_PREDICATE_SHIFT_MAP = {
    "director": "writer",
    "actor": "producer",
    "mascot": "nickname",
    "founder": "headquartered_in",
    "birthplace": "residence",
    "capital": "largest_city",
    "headquartered_in": "founded_in",
    "parent_company": "subsidiary",
    "author": "publication_year",
}
_CONSTRAINT_FLIP_MAP = {
    "before": "after",
    "after": "before",
    "earlier": "later",
    "later": "earlier",
    "first": "last",
    "last": "first",
    "largest": "smallest",
    "smallest": "largest",
    "higher": "lower",
    "lower": "higher",
    "not": "including",
    "except": "including",
}
_RELATION_PATTERN_MAP = {
    "directed by": "director",
    "director of": "director",
    "written by": "author",
    "founded by": "founder",
    "born in": "birthplace",
    "headquartered in": "headquartered_in",
    "capital of": "capital",
    "mascot of": "mascot",
    "president of": "president",
    "parent company": "parent_company",
    "starring": "actor",
    "played by": "actor",
    "coach of": "coach",
    "captain of": "captain",
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
        first_token = candidate.split(" ", 1)[0] if candidate else ""
        if (
            not candidate
            or candidate in seen
            or len(candidate) < 3
            or first_token in _WH_WORDS
        ):
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


def get_requirement_cache_version(payload_or_entry: Dict[str, Any] | None) -> str:
    version = str((payload_or_entry or {}).get("version", "")).strip()
    if version:
        return version
    if payload_or_entry and "positive_need_units" in payload_or_entry:
        return NEED_UNIT_CACHE_VERSION
    return REQUIREMENT_CACHE_VERSION


def get_requirement_group_types(cache_entry: Dict[str, Any]) -> tuple[str, ...]:
    if get_requirement_cache_version(cache_entry) == NEED_UNIT_CACHE_VERSION:
        return NEED_UNIT_TYPES
    return REQUIREMENT_TYPES


def get_positive_units(cache_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "positive_need_units" in cache_entry:
        return list(cache_entry.get("positive_need_units", []))
    return list(cache_entry.get("positive_requirements", []))


def get_counterfactual_sets(cache_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    counterfactual_sets = list(cache_entry.get("counterfactual_sets", []))
    normalized_sets: List[Dict[str, Any]] = []
    for cf_set in counterfactual_sets:
        if "requirements" in cf_set:
            normalized_sets.append(cf_set)
            continue
        if "need_units" in cf_set:
            normalized_cf_set = dict(cf_set)
            normalized_cf_set["requirements"] = list(cf_set.get("need_units", []))
            normalized_sets.append(normalized_cf_set)
            continue
        normalized_sets.append(cf_set)
    return normalized_sets


def get_positive_score_map(annotation: Dict[str, Any]) -> Dict[str, float]:
    return dict(
        annotation.get("positive_need_unit_scores")
        or annotation.get("positive_requirement_scores")
        or {}
    )


def get_counterfactual_score_map(annotation: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return dict(annotation.get("counterfactual_requirement_scores") or {})


def get_positive_contradiction_map(annotation: Dict[str, Any]) -> Dict[str, float]:
    return dict(annotation.get("positive_contradiction_scores") or {})


def get_counterfactual_set_score_map(annotation: Dict[str, Any]) -> Dict[str, float]:
    return dict(annotation.get("counterfactual_set_scores") or {})


def normalize_predicate_text(text: str) -> str:
    return normalize_structure_text(str(text or "")).replace(" ", "_")


def build_need_unit(unit_id: str,
                    unit_type: str,
                    subject: str,
                    predicate: str,
                    object_value: str,
                    constraints: Sequence[Dict[str, Any]] | None = None,
                    target_variable: str = "",
                    answer_relevance: bool = False,
                    source_step_ids: Sequence[str] | None = None) -> Dict[str, Any]:
    normalized_subject = normalize_structure_text(subject or "")
    normalized_predicate = normalize_predicate_text(predicate or "")
    normalized_object = normalize_structure_text(object_value or "")
    normalized_constraints = [
        {
            "kind": str(constraint.get("kind", "")).strip(),
            "value": str(constraint.get("value", "")).strip(),
        }
        for constraint in (constraints or [])
        if str(constraint.get("kind", "")).strip() and str(constraint.get("value", "")).strip()
    ]
    raw_text_parts = [
        normalized_subject,
        normalized_predicate.replace("_", " "),
        normalized_object,
        " ".join(f"{item['kind']} {item['value']}" for item in normalized_constraints),
    ]
    normalized_text = normalize_structure_text(" ".join(part for part in raw_text_parts if part))
    tokens = [token for token in tokenize_text(normalized_text) if token not in _STOPWORDS]
    entities = unique_ordered_texts([
        normalized_subject if normalized_subject and not normalized_subject.startswith("?") else "",
        normalized_object if normalized_object and not normalized_object.startswith("?") else "",
    ])
    return {
        "unit_id": str(unit_id),
        "requirement_id": str(unit_id),
        "unit_type": str(unit_type),
        "type": str(unit_type),
        "target_variable": str(target_variable or ""),
        "subject": str(subject or ""),
        "predicate": str(predicate or ""),
        "object": str(object_value or ""),
        "constraints": normalized_constraints,
        "answer_relevance": bool(answer_relevance),
        "source_step_ids": list(source_step_ids or []),
        "text": normalized_text,
        "normalized_text": normalized_text,
        "tokens": tokens,
        "entities": entities,
        "answer_type_tokens": [],
    }


def _extract_question_constraints(question: str) -> List[Dict[str, Any]]:
    normalized_question = normalize_structure_text(question or "")
    constraints: List[Dict[str, Any]] = []
    for match in _YEAR_RE.findall(question or ""):
        constraints.append({"kind": "year", "value": str(match)})
    for token in tokenize_text(normalized_question):
        if token in _TEMPORAL_KEYWORDS:
            constraints.append({"kind": "temporal", "value": token})
        elif token in _NEGATION_KEYWORDS:
            constraints.append({"kind": "polarity", "value": token})
        elif token in _COMPARATIVE_KEYWORDS:
            constraints.append({"kind": "comparison", "value": token})
    deduped: List[Dict[str, Any]] = []
    seen: Set[tuple[str, str]] = set()
    for constraint in constraints:
        key = (constraint["kind"], constraint["value"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(constraint)
    return deduped[:2]


def _extract_relation_predicates(question: str,
                                 anchor_entities: Sequence[str],
                                 answer_type_tokens: Sequence[str]) -> List[str]:
    normalized_question = normalize_structure_text(question or "")
    predicates: List[str] = []
    for raw_phrase, predicate in _RELATION_PATTERN_MAP.items():
        if raw_phrase in normalized_question:
            predicates.append(predicate)

    if not predicates:
        focus_terms = extract_question_focus_terms(question, anchor_entities=anchor_entities, max_terms=8)
        filtered_focus = [
            term for term in focus_terms
            if term not in answer_type_tokens
            and term not in _TEMPORAL_KEYWORDS
            and term not in _NEGATION_KEYWORDS
            and term not in _COMPARATIVE_KEYWORDS
            and term not in _WH_WORDS
        ]
        if filtered_focus:
            predicates.append("_".join(filtered_focus[:2]))
        elif answer_type_tokens:
            predicates.append("_".join(list(answer_type_tokens)[:2]))

    deduped: List[str] = []
    seen: Set[str] = set()
    for predicate in predicates:
        normalized_predicate = normalize_predicate_text(predicate)
        if not normalized_predicate or normalized_predicate in seen:
            continue
        seen.add(normalized_predicate)
        deduped.append(normalized_predicate)
    return deduped[:2]


def build_qdmr_steps_and_need_units(question: str,
                                    seed_entities: Sequence[str] | Set[str] | None,
                                    question_entities: Sequence[str] | Set[str] | None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    anchor_entities = extract_question_entities(
        question=question,
        fallback_entities=question_entities or seed_entities,
        max_entities=2,
    )
    answer_type_tokens = guess_answer_type_tokens(question)
    predicates = _extract_relation_predicates(question, anchor_entities=anchor_entities, answer_type_tokens=answer_type_tokens)
    constraints = _extract_question_constraints(question)

    steps: List[Dict[str, Any]] = []
    units: List[Dict[str, Any]] = []
    current_subject = anchor_entities[0] if anchor_entities else "?focus"
    current_variable = "?x"

    if anchor_entities:
        locator_step_id = "s1"
        steps.append({
            "step_id": locator_step_id,
            "text": f"Locate the primary entity {anchor_entities[0]}",
        })
        units.append(build_need_unit(
            unit_id="u_entity_locator_0",
            unit_type="entity_locator",
            subject=anchor_entities[0],
            predicate="locate",
            object_value=anchor_entities[0],
            target_variable=current_variable,
            answer_relevance=False,
            source_step_ids=[locator_step_id],
        ))

    variable_names = ["?x", "?y", "?z"]
    hop_count = min(max(len(predicates), 1), 2)
    for hop_idx in range(hop_count):
        predicate = predicates[hop_idx] if hop_idx < len(predicates) else "related_to"
        next_variable = variable_names[min(hop_idx, len(variable_names) - 1)]
        step_id = f"s_relation_{hop_idx}"
        steps.append({
            "step_id": step_id,
            "text": f"Resolve {predicate.replace('_', ' ')} for {current_subject}",
        })
        units.append(build_need_unit(
            unit_id=f"u_relation_hop_{hop_idx}",
            unit_type="relation_hop",
            subject=current_subject,
            predicate=predicate,
            object_value=next_variable,
            target_variable=next_variable,
            answer_relevance=False,
            source_step_ids=[step_id],
        ))
        current_subject = next_variable
        current_variable = next_variable

    for constraint_idx, constraint in enumerate(constraints[:2]):
        step_id = f"s_constraint_{constraint_idx}"
        steps.append({
            "step_id": step_id,
            "text": f"Apply {constraint['kind']} constraint {constraint['value']}",
        })
        units.append(build_need_unit(
            unit_id=f"u_constraint_{constraint_idx}",
            unit_type="constraint_check",
            subject=current_subject,
            predicate="constraint",
            object_value=current_variable or "?x",
            constraints=[constraint],
            target_variable=current_variable or "?x",
            answer_relevance=False,
            source_step_ids=[step_id],
        ))

    answer_predicate = predicates[-1] if predicates else "_".join(answer_type_tokens[:2]) if answer_type_tokens else "answer"
    answer_step_id = "s_answer"
    steps.append({
        "step_id": answer_step_id,
        "text": f"Return the answer slot for {answer_predicate.replace('_', ' ')}",
    })
    units.append(build_need_unit(
        unit_id="u_answer_slot_0",
        unit_type="answer_slot",
        subject=current_subject,
        predicate=answer_predicate,
        object_value="?ans",
        target_variable="?ans",
        answer_relevance=True,
        source_step_ids=[answer_step_id],
    ))

    return steps, units[:6]


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


def _mutate_need_unit(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    unit_type = str(unit.get("unit_type", unit.get("type", ""))).strip()
    predicate = normalize_predicate_text(unit.get("predicate", ""))
    constraints = list(unit.get("constraints", []))
    mutated_units: List[Dict[str, Any]] = []

    if unit_type in {"relation_hop", "entity_locator"}:
        role_swap_predicate = _COMMON_ROLE_PREDICATE_MAP.get(predicate)
        if role_swap_predicate:
            mutated_units.append({
                "transform": "role_swap",
                "unit": build_need_unit(
                    unit_id=f"{unit['unit_id']}_role_swap",
                    unit_type=unit_type,
                    subject=str(unit.get("subject", "")),
                    predicate=role_swap_predicate,
                    object_value=str(unit.get("object", "")),
                    constraints=constraints,
                    target_variable=str(unit.get("target_variable", "")),
                    answer_relevance=bool(unit.get("answer_relevance", False)),
                    source_step_ids=unit.get("source_step_ids", []),
                ),
            })
        predicate_shift = _PREDICATE_SHIFT_MAP.get(predicate)
        if predicate_shift:
            mutated_units.append({
                "transform": "predicate_shift",
                "unit": build_need_unit(
                    unit_id=f"{unit['unit_id']}_predicate_shift",
                    unit_type=unit_type,
                    subject=str(unit.get("subject", "")),
                    predicate=predicate_shift,
                    object_value=str(unit.get("object", "")),
                    constraints=constraints,
                    target_variable=str(unit.get("target_variable", "")),
                    answer_relevance=bool(unit.get("answer_relevance", False)),
                    source_step_ids=unit.get("source_step_ids", []),
                ),
            })

    if unit_type == "constraint_check":
        for constraint in constraints:
            if constraint.get("kind") == "year":
                value = str(constraint.get("value", "")).strip()
                if value.isdigit():
                    shifted_value = str(int(value) - 1)
                    mutated_units.append({
                        "transform": "temporal_shift",
                        "unit": build_need_unit(
                            unit_id=f"{unit['unit_id']}_temporal_shift",
                            unit_type=unit_type,
                            subject=str(unit.get("subject", "")),
                            predicate=str(unit.get("predicate", "constraint")),
                            object_value=str(unit.get("object", "")),
                            constraints=[{"kind": "year", "value": shifted_value}],
                            target_variable=str(unit.get("target_variable", "")),
                            answer_relevance=bool(unit.get("answer_relevance", False)),
                            source_step_ids=unit.get("source_step_ids", []),
                        ),
                    })
            flipped = _CONSTRAINT_FLIP_MAP.get(str(constraint.get("value", "")).strip())
            if flipped:
                mutated_units.append({
                    "transform": "constraint_flip",
                    "unit": build_need_unit(
                        unit_id=f"{unit['unit_id']}_constraint_flip",
                        unit_type=unit_type,
                        subject=str(unit.get("subject", "")),
                        predicate=str(unit.get("predicate", "constraint")),
                        object_value=str(unit.get("object", "")),
                        constraints=[{"kind": str(constraint.get("kind", "")), "value": flipped}],
                        target_variable=str(unit.get("target_variable", "")),
                        answer_relevance=bool(unit.get("answer_relevance", False)),
                        source_step_ids=unit.get("source_step_ids", []),
                    ),
                })

    deduped: List[Dict[str, Any]] = []
    seen: Set[tuple[str, str]] = set()
    for item in mutated_units:
        transform = str(item["transform"])
        mutated_unit = item["unit"]
        key = (transform, str(mutated_unit.get("normalized_text", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_counterfactual_need_unit_sets(question: str,
                                        positive_need_units: Sequence[Dict[str, Any]],
                                        max_sets: int = DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS) -> List[Dict[str, Any]]:
    del question
    counterfactual_sets: List[Dict[str, Any]] = []
    for unit in positive_need_units:
        unit_type = str(unit.get("unit_type", unit.get("type", ""))).strip()
        if unit_type == "answer_slot":
            continue
        for item in _mutate_need_unit(unit):
            mutated_unit = item["unit"]
            cf_units = []
            for positive_unit in positive_need_units:
                if str(positive_unit.get("unit_id", positive_unit.get("requirement_id", ""))) == str(unit.get("unit_id", unit.get("requirement_id", ""))):
                    cf_units.append(mutated_unit)
                else:
                    cf_units.append(dict(positive_unit))
            cf_id = f"cf_{len(counterfactual_sets)}"
            counterfactual_sets.append({
                "cf_id": cf_id,
                "transform": str(item["transform"]),
                "source_unit_id": str(unit.get("unit_id", unit.get("requirement_id", ""))),
                "requirements": cf_units,
                "need_units": cf_units,
            })
            if len(counterfactual_sets) >= max(int(max_sets), 1):
                return counterfactual_sets
    return counterfactual_sets


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


def _token_overlap_ratio(left_tokens: Sequence[str] | Set[str],
                         right_tokens: Sequence[str] | Set[str]) -> float:
    left = set(left_tokens or [])
    right = set(right_tokens or [])
    if not left:
        return 0.0
    return float(len(left & right) / float(max(1, len(left))))


def _constraint_alignment_score(constraints: Sequence[Dict[str, Any]], normalized_doc_text: str) -> float:
    if not constraints:
        return 0.5
    hits = 0
    for constraint in constraints:
        value = normalize_structure_text(constraint.get("value", ""))
        if value and value in normalized_doc_text:
            hits += 1
    return float(hits / max(1, len(constraints)))


def _opposing_predicate_score(predicate: str,
                              normalized_doc_text: str) -> float:
    normalized_predicate = normalize_predicate_text(predicate)
    opposing_candidates = [
        _COMMON_ROLE_PREDICATE_MAP.get(normalized_predicate, ""),
        _PREDICATE_SHIFT_MAP.get(normalized_predicate, ""),
    ]
    best = 0.0
    for candidate in opposing_candidates:
        candidate_text = str(candidate or "").replace("_", " ").strip()
        if candidate_text and candidate_text in normalized_doc_text:
            best = max(best, 1.0)
    return best


def score_need_unit_support(unit: Dict[str, Any],
                            doc_title: str,
                            doc_body: str,
                            doc_entities: Sequence[str] | Set[str] | None) -> Dict[str, float]:
    normalized_doc_entities = set(unique_ordered_texts(doc_entities))
    normalized_doc_text = normalize_structure_text(f"{doc_title} {doc_body}")
    title_tokens = set(tokenize_text(doc_title))
    body_tokens = set(tokenize_text(doc_body))
    doc_tokens = title_tokens | body_tokens

    subject = normalize_structure_text(unit.get("subject", ""))
    predicate = normalize_predicate_text(unit.get("predicate", ""))
    object_value = normalize_structure_text(unit.get("object", ""))
    subject_tokens = set(tokenize_text(subject))
    predicate_tokens = set(tokenize_text(predicate.replace("_", " ")))
    object_tokens = set(tokenize_text(object_value)) if object_value and not object_value.startswith("?") else set()
    constraint_score = _constraint_alignment_score(unit.get("constraints", []), normalized_doc_text)

    subject_alignment = 0.5 if not subject or subject.startswith("?") else max(
        1.0 if subject in normalized_doc_text else 0.0,
        _token_overlap_ratio(subject_tokens, doc_tokens),
        1.0 if subject in normalized_doc_entities else 0.0,
    )
    predicate_alignment = max(
        0.0 if not predicate_tokens else _token_overlap_ratio(predicate_tokens, doc_tokens),
        1.0 if predicate and predicate.replace("_", " ") in normalized_doc_text else 0.0,
    )
    object_alignment = 0.5 if not object_tokens else max(
        _token_overlap_ratio(object_tokens, doc_tokens),
        1.0 if object_value in normalized_doc_text else 0.0,
        1.0 if object_value in normalized_doc_entities else 0.0,
    )
    alignment_score = float(np.clip(
        0.40 * subject_alignment + 0.35 * predicate_alignment + 0.15 * object_alignment + 0.10 * constraint_score,
        0.0,
        1.0,
    ))
    support_prob = float(np.clip(
        0.50 * predicate_alignment + 0.25 * subject_alignment + 0.15 * object_alignment + 0.10 * constraint_score,
        0.0,
        1.0,
    ))
    contradiction_prob = float(np.clip(
        0.65 * _opposing_predicate_score(predicate, normalized_doc_text)
        + 0.35 * max(0.0, 1.0 - constraint_score) * float(bool(unit.get("constraints"))),
        0.0,
        1.0,
    ))
    nei_prob = float(np.clip(1.0 - max(support_prob, contradiction_prob), 0.0, 1.0))
    coverage_score = float(np.clip(alignment_score * max(0.0, support_prob - contradiction_prob), 0.0, 1.0))
    return {
        "alignment_score": round(alignment_score, 6),
        "support_prob": round(support_prob, 6),
        "contradiction_prob": round(contradiction_prob, 6),
        "nei_prob": round(nei_prob, 6),
        "coverage_score": round(coverage_score, 6),
    }


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


def build_need_unit_doc_annotation(pool_position: int,
                                   doc_text: str,
                                   doc_entities: Sequence[str] | Set[str] | None,
                                   positive_need_units: Sequence[Dict[str, Any]],
                                   counterfactual_sets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    doc_title, doc_body = (str(doc_text).split("\n", 1) + [""])[:2]
    normalized_doc_entities = unique_ordered_texts(doc_entities)

    positive_scores: Dict[str, float] = {}
    positive_alignments: Dict[str, float] = {}
    positive_support_probs: Dict[str, float] = {}
    positive_contradiction_probs: Dict[str, float] = {}
    positive_nei_probs: Dict[str, float] = {}
    for unit in positive_need_units:
        unit_id = str(unit.get("unit_id", unit.get("requirement_id", "")))
        scores = score_need_unit_support(
            unit=unit,
            doc_title=doc_title,
            doc_body=doc_body,
            doc_entities=normalized_doc_entities,
        )
        positive_scores[unit_id] = float(scores["coverage_score"])
        positive_alignments[unit_id] = float(scores["alignment_score"])
        positive_support_probs[unit_id] = float(scores["support_prob"])
        positive_contradiction_probs[unit_id] = float(scores["contradiction_prob"])
        positive_nei_probs[unit_id] = float(scores["nei_prob"])

    counterfactual_scores: Dict[str, Dict[str, float]] = {}
    counterfactual_alignments: Dict[str, Dict[str, float]] = {}
    counterfactual_support_probs: Dict[str, Dict[str, float]] = {}
    counterfactual_contradiction_probs: Dict[str, Dict[str, float]] = {}
    counterfactual_nei_probs: Dict[str, Dict[str, float]] = {}
    counterfactual_set_scores: Dict[str, float] = {}
    for cf_set in counterfactual_sets:
        cf_id = str(cf_set.get("cf_id", ""))
        need_units = list(cf_set.get("requirements", cf_set.get("need_units", [])))
        cf_cov_scores: Dict[str, float] = {}
        cf_align_scores: Dict[str, float] = {}
        cf_support_scores: Dict[str, float] = {}
        cf_contradiction_scores: Dict[str, float] = {}
        cf_nei_scores: Dict[str, float] = {}
        for unit in need_units:
            unit_id = str(unit.get("unit_id", unit.get("requirement_id", "")))
            scores = score_need_unit_support(
                unit=unit,
                doc_title=doc_title,
                doc_body=doc_body,
                doc_entities=normalized_doc_entities,
            )
            cf_cov_scores[unit_id] = float(scores["coverage_score"])
            cf_align_scores[unit_id] = float(scores["alignment_score"])
            cf_support_scores[unit_id] = float(scores["support_prob"])
            cf_contradiction_scores[unit_id] = float(scores["contradiction_prob"])
            cf_nei_scores[unit_id] = float(scores["nei_prob"])
        counterfactual_scores[cf_id] = cf_cov_scores
        counterfactual_alignments[cf_id] = cf_align_scores
        counterfactual_support_probs[cf_id] = cf_support_scores
        counterfactual_contradiction_probs[cf_id] = cf_contradiction_scores
        counterfactual_nei_probs[cf_id] = cf_nei_scores
        counterfactual_set_scores[cf_id] = round(
            float(np.mean(list(cf_cov_scores.values()))) if cf_cov_scores else 0.0,
            6,
        )

    return {
        "pool_position": int(pool_position),
        "doc_title": doc_title,
        "positive_need_unit_scores": positive_scores,
        "positive_requirement_scores": positive_scores,
        "positive_alignment_scores": positive_alignments,
        "positive_support_probs": positive_support_probs,
        "positive_contradiction_scores": positive_contradiction_probs,
        "positive_nei_probs": positive_nei_probs,
        "counterfactual_requirement_scores": counterfactual_scores,
        "counterfactual_alignment_scores": counterfactual_alignments,
        "counterfactual_support_probs": counterfactual_support_probs,
        "counterfactual_contradiction_scores": counterfactual_contradiction_probs,
        "counterfactual_nei_probs": counterfactual_nei_probs,
        "counterfactual_set_scores": counterfactual_set_scores,
    }


def build_cache_doc_annotation(pool_position: int,
                               doc_text: str,
                               doc_entities: Sequence[str] | Set[str] | None,
                               cache_entry: Dict[str, Any]) -> Dict[str, Any]:
    if get_requirement_cache_version(cache_entry) == NEED_UNIT_CACHE_VERSION or "positive_need_units" in cache_entry:
        return build_need_unit_doc_annotation(
            pool_position=pool_position,
            doc_text=doc_text,
            doc_entities=doc_entities,
            positive_need_units=get_positive_units(cache_entry),
            counterfactual_sets=get_counterfactual_sets(cache_entry),
        )
    return build_requirement_doc_annotation(
        pool_position=pool_position,
        doc_text=doc_text,
        doc_entities=doc_entities,
        positive_requirements=list(cache_entry.get("positive_requirements", [])),
        counterfactual_sets=get_counterfactual_sets(cache_entry),
    )


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


def build_need_unit_cache_entry(query_index: int,
                                question: str,
                                pool_docs: Sequence[str],
                                pool_doc_entities: Sequence[Sequence[str] | Set[str]],
                                seed_entities: Sequence[str] | Set[str] | None,
                                question_entities: Sequence[str] | Set[str] | None,
                                annotation_pool_k: int = DEFAULT_REQUIREMENT_ANNOTATION_POOL_K,
                                max_counterfactual_sets: int = DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS) -> Dict[str, Any]:
    qdmr_steps, positive_need_units = build_qdmr_steps_and_need_units(
        question=question,
        seed_entities=seed_entities,
        question_entities=question_entities,
    )
    counterfactual_sets = build_counterfactual_need_unit_sets(
        question=question,
        positive_need_units=positive_need_units,
        max_sets=max_counterfactual_sets,
    )
    effective_annotation_pool_k = min(len(pool_docs), max(int(annotation_pool_k), 0))
    cache_entry: Dict[str, Any] = {
        "version": NEED_UNIT_CACHE_VERSION,
        "query_index": int(query_index),
        "question": question,
        "question_key": stable_question_key(question),
        "annotation_pool_k": int(effective_annotation_pool_k),
        "seed_entities": list(unique_ordered_texts(seed_entities)),
        "question_entities": list(unique_ordered_texts(question_entities)),
        "qdmr_steps": qdmr_steps,
        "positive_need_units": positive_need_units,
        "counterfactual_sets": counterfactual_sets,
        "pool_titles": [
            str(doc_text).split("\n", 1)[0].strip()
            for doc_text in pool_docs[:effective_annotation_pool_k]
        ],
        "doc_annotations": [],
        "diagnostics": {
            "positive_need_unit_count": int(len(positive_need_units)),
            "counterfactual_set_count": int(len(counterfactual_sets)),
            "nonempty_counterfactual_rate": round(float(len(counterfactual_sets) > 0), 4),
            "qdmr_step_count": int(len(qdmr_steps)),
        },
    }
    for pool_position in range(effective_annotation_pool_k):
        cache_entry["doc_annotations"].append(build_cache_doc_annotation(
            pool_position=pool_position,
            doc_text=str(pool_docs[pool_position]),
            doc_entities=pool_doc_entities[pool_position] if pool_position < len(pool_doc_entities) else [],
            cache_entry=cache_entry,
        ))
    return cache_entry


def save_requirement_cache(path: str | Path, payload: Dict[str, Any]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp_path.replace(target_path)


def load_requirement_cache(path: str | Path) -> Dict[str, Any]:
    cache_payload = json.loads(Path(path).read_text())
    cache_version = str(cache_payload.get("version", "")).strip()
    if cache_version not in SUPPORTED_REQUIREMENT_CACHE_VERSIONS:
        raise ValueError(
            f"Unsupported requirement cache version at {path}: {cache_payload.get('version')}"
        )
    entries = cache_payload.get("queries")
    if not isinstance(entries, list):
        raise ValueError(f"Invalid requirement cache payload at {path}: missing queries")

    entries_by_question: Dict[str, Dict[str, Any]] = {}
    entries_by_index: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        entry.setdefault("version", cache_version)
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
            aligned_annotation = build_cache_doc_annotation(
                pool_position=pool_position,
                doc_text=str(pool_docs[pool_position]),
                doc_entities=doc_entities,
                cache_entry=cache_entry,
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

    group_names = get_requirement_group_types(cache_entry)
    group_values: Dict[str, List[float]] = {group: [] for group in group_names}
    for requirement in get_positive_units(cache_entry):
        requirement_id = str(requirement.get("unit_id", requirement.get("requirement_id", "")))
        scores = [
            float(
                get_positive_score_map(annotations_by_position[pos]).get(requirement_id, 0.0)
            )
            for pos in unique_positions
            if pos in annotations_by_position
        ]
        requirement_type = str(requirement.get("unit_type", requirement.get("type", "bridge")))
        group_values.setdefault(requirement_type, [])
        group_values[requirement_type].append(
            _requirement_coverage_for_positions(scores)
        )

    group_means: Dict[str, float] = {}
    for group_name in group_names:
        values = group_values.get(group_name, [])
        group_means[group_name] = float(np.mean(values)) if values else 0.0

    active_group_values = [
        group_means[group_name]
        for group_name in group_names
        if group_values.get(group_name)
    ]
    support_completeness = _smooth_min(active_group_values, tau=smooth_tau) if active_group_values else 0.0

    counterfactual_supports: List[float] = []
    for cf_set in get_counterfactual_sets(cache_entry):
        requirement_coverages: List[float] = []
        for requirement in cf_set.get("requirements", []):
            requirement_id = str(requirement.get("unit_id", requirement.get("requirement_id", "")))
            scores = [
                float(
                    get_counterfactual_score_map(annotations_by_position[pos])
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
        "anchor_support": float(group_means.get("anchor", 0.0)),
        "bridge_support": float(group_means.get("bridge", 0.0)),
        "decision_support": float(group_means.get("decision", 0.0)),
        "entity_locator_support": float(group_means.get("entity_locator", 0.0)),
        "relation_hop_support": float(group_means.get("relation_hop", 0.0)),
        "constraint_check_support": float(group_means.get("constraint_check", 0.0)),
        "answer_slot_support": float(group_means.get("answer_slot", 0.0)),
        "group_supports": dict(group_means),
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
        positive_scores = list(get_positive_score_map(annotation).values())
        counterfactual_set_scores = list(get_counterfactual_set_score_map(annotation).values())
        contradiction_scores = list(get_positive_contradiction_map(annotation).values())
        rows.append({
            "pool_position": int(pos),
            "base_score": float(normalized_base_scores[pos]) if 0 <= int(pos) < len(normalized_base_scores) else 0.0,
            "rank_fraction": 1.0 - (float(pos) / max(1, pool_size - 1)) if pool_size > 1 else 1.0,
            "reciprocal_rank": 1.0 / float(int(pos) + 1),
            "selected_count_fraction": len(selected_positions) / float(max(1, qa_top_k)),
            "anchor_support_after": float(next_metrics["anchor_support"]),
            "bridge_support_after": float(next_metrics["bridge_support"]),
            "decision_support_after": float(next_metrics["decision_support"]),
            "entity_locator_support_after": float(next_metrics["entity_locator_support"]),
            "relation_hop_support_after": float(next_metrics["relation_hop_support"]),
            "constraint_check_support_after": float(next_metrics["constraint_check_support"]),
            "answer_slot_support_after": float(next_metrics["answer_slot_support"]),
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
            "doc_contradiction_max": float(max(contradiction_scores) if contradiction_scores else 0.0),
            "doc_contradiction_mean": float(np.mean(contradiction_scores) if contradiction_scores else 0.0),
        })
    return rows


def requirement_feature_rows_to_matrix(feature_rows: Sequence[Dict[str, float | int]],
                                       feature_names: Sequence[str] | None = None) -> np.ndarray:
    resolved_feature_names = list(feature_names or REQUIREMENT_MATCHER_FEATURE_NAMES)
    if not feature_rows:
        return np.zeros((0, len(resolved_feature_names)), dtype=float)
    return np.asarray([
        [float(row.get(feature_name, 0.0) or 0.0) for feature_name in resolved_feature_names]
        for row in feature_rows
    ], dtype=float)


def load_requirement_model_bundle(model_path: str | Path) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid requirement matcher bundle at {model_path}")
    feature_names = list(bundle.get("feature_names") or [])
    supported_feature_sets = [
        list(REQUIREMENT_MATCHER_FEATURE_NAMES),
        list(NEED_UNIT_MATCHER_FEATURE_NAMES),
    ]
    if feature_names and feature_names not in supported_feature_sets:
        raise ValueError(
            "Requirement matcher feature mismatch: "
            f"expected one of {supported_feature_sets}, got {feature_names}"
        )
    bundle.setdefault("feature_names", list(REQUIREMENT_MATCHER_FEATURE_NAMES))
    bundle.setdefault("model_path", str(model_path))
    return bundle
