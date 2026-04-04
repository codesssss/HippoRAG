from __future__ import annotations

import ast
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
from src.hipporag.utils.llm_utils import fix_broken_generated_json


REQUIREMENT_CACHE_VERSION = "pcrs_rag_v1"
LEGACY_NEED_UNIT_CACHE_VERSION = "pcrs_rag_v2_need_units"
NEED_UNIT_CACHE_VERSION = "pcrs_rag_v2_need_units_qdmr_v1"
NEED_UNIT_CACHE_VERSIONS = {
    LEGACY_NEED_UNIT_CACHE_VERSION,
    NEED_UNIT_CACHE_VERSION,
}
SUPPORTED_REQUIREMENT_CACHE_VERSIONS = {
    REQUIREMENT_CACHE_VERSION,
    *NEED_UNIT_CACHE_VERSIONS,
}
DEFAULT_REQUIREMENT_ANNOTATION_POOL_K = 50
DEFAULT_REQUIREMENT_SMOOTH_TAU = 0.15
DEFAULT_REQUIREMENT_CF_TAU = 0.10
DEFAULT_REQUIREMENT_MAX_COUNTERFACTUALS = 3
DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS = 6
DEFAULT_NEED_UNIT_MAX_RELATION_HOPS = 2
DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE = "fixed"
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
NEED_UNIT_ATOMIC_SCORER_VERSION = "need_unit_atomic_v1"
NEED_UNIT_ATOMIC_LABELS = (
    "full_support",
    "bridge_support",
    "contradiction",
    "nei",
)
DEFAULT_NEED_UNIT_BRIDGE_ALPHA_BY_TYPE = {
    "entity_locator": 0.35,
    "relation_hop": 0.65,
    "constraint_check": 0.40,
    "answer_slot": 0.25,
}
DEFAULT_NEED_UNIT_CONTRADICTION_BETA = 1.0
NEED_UNIT_ATOMIC_FEATURE_NAMES = [
    "is_entity_locator",
    "is_relation_hop",
    "is_constraint_check",
    "is_answer_slot",
    "answer_relevance",
    "has_constraints",
    "constraint_count",
    "subject_is_variable",
    "object_is_variable",
    "subject_in_title",
    "subject_in_body",
    "subject_token_overlap",
    "subject_entity_match",
    "object_in_title",
    "object_in_body",
    "object_token_overlap",
    "object_entity_match",
    "predicate_in_title",
    "predicate_in_body",
    "predicate_token_overlap",
    "opposing_predicate_in_text",
    "constraint_alignment",
    "constraint_temporal_score",
    "constraint_role_score",
    "constraint_comparative_score",
    "constraint_negation_score",
    "constraint_scope_score",
    "title_bridge_alignment",
    "entity_bridge_alignment",
    "alias_or_variable_bridge_alignment",
    "bridge_feature_count",
    "subject_alignment",
    "predicate_alignment",
    "object_alignment",
]
ATOMIC_ANNOTATION_SCORE_MODES = {
    "heuristic",
    "hybrid",
}
REQUIREMENT_BRIDGE_BONUS_MODES = (
    "off",
    "variable_binding",
)
DEFAULT_QDMR_PARSER_MAX_COMPLETION_TOKENS = 1024
DEFAULT_QDMR_POOL_TITLE_HEAD = 8
_QDMR_ALLOWED_OPERATIONS = {
    "locate_entity",
    "relation_lookup",
    "constraint_check",
    "answer",
}
_RAW_TO_CANONICAL_VARIABLE = {
    "X": "?x",
    "Y": "?y",
    "Z": "?z",
    "ANSWER": "?ans",
}
_CANONICAL_TO_RAW_VARIABLE = {
    canonical: raw
    for raw, canonical in _RAW_TO_CANONICAL_VARIABLE.items()
}
_CONSTRAINT_PRIORITY = {
    "role": 0,
    "temporal": 1,
    "comparative": 2,
    "negation": 3,
    "scope": 4,
    "answer_type": 5,
}

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
_GENERIC_PREDICATE_TOKENS = {
    "person",
    "thing",
    "place",
    "many",
    "part",
    "type",
    "kind",
    "item",
    "object",
    "name",
}
_COMMON_ROLE_PREDICATE_MAP = {
    "directed_by": "starred_in",
    "starred_in": "directed_by",
    "played_by": "directed_by",
    "mascot": "founded_by",
    "founded_by": "mascot",
    "written_by": "published_by",
    "published_by": "written_by",
    "capital": "birth_place",
    "birth_place": "capital",
    "parent_company": "headquartered_in",
    "headquartered_in": "parent_company",
}
_PREDICATE_SHIFT_MAP = {
    "birth_place": "nationality",
    "nationality": "birth_place",
    "death_place": "birth_place",
    "won": "nominated_for",
    "nominated_for": "won",
    "associated_university": "educated_at",
    "educated_at": "associated_university",
    "publisher": "owner_of",
    "owner_of": "publisher",
    "start_date": "end_date",
    "end_date": "start_date",
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
_CANONICAL_PREDICATES = {
    "identify",
    "associated_university",
    "mascot",
    "founded_by",
    "directed_by",
    "starred_in",
    "played_by",
    "written_by",
    "published_by",
    "author_of",
    "publisher",
    "signed_by",
    "scored_goal",
    "birth_place",
    "death_place",
    "date_of_birth",
    "nationality",
    "capital",
    "county",
    "north_of",
    "located_in",
    "headquartered_in",
    "parent_company",
    "owner_of",
    "child_of",
    "spouse_of",
    "educated_at",
    "won",
    "nominated_for",
    "release_date",
    "start_date",
    "end_date",
    "population",
    "empties_into",
    "governor_of",
    "screenwriter_of",
    "composer_of",
    "designer_of",
    "named_after",
    "return_as_answer",
    "satisfy_constraint",
}
_PREDICATE_ALIAS_FAMILIES = {
    "mascot": ("mascot of", "mascot for", "mascot"),
    "associated_university": (
        "university associated with",
        "college associated with",
        "university related to",
        "university related with",
    ),
    "directed_by": ("director of", "directed by", "directed"),
    "starred_in": ("actor in", "star in", "starred in", "starring"),
    "played_by": ("played by", "portrayed by"),
    "written_by": ("written by", "writer of", "author of"),
    "published_by": ("publisher of", "published by"),
    "publisher": ("publisher", "record label", "record label of"),
    "signed_by": ("signed by", "signed with"),
    "scored_goal": ("goal by", "goals by", "goal scored by", "goals scored by", "scored by"),
    "birth_place": ("birthplace of", "born in", "place of birth", "is from", "from"),
    "death_place": ("death place of", "place of death", "died in", "death place"),
    "date_of_birth": ("date of birth", "birth date"),
    "nationality": ("nationality of", "citizen of"),
    "capital": ("capital of", "capital"),
    "county": ("county of", "county", "part of"),
    "north_of": ("north of", "immediately north of"),
    "located_in": ("located in", "located at", "in the state of"),
    "headquartered_in": ("headquartered in", "headquarters in", "headquarters of", "headquarters"),
    "parent_company": ("parent company of", "parent company"),
    "owner_of": ("owner of", "owned by"),
    "child_of": ("child of", "son of", "daughter of"),
    "spouse_of": ("spouse of", "married to"),
    "educated_at": ("educated at", "studied at"),
    "won": ("won", "received"),
    "nominated_for": ("nominated for",),
    "release_date": ("released in", "release date of", "released before", "released after"),
    "start_date": ("began in", "started in", "start date of", "start year of", "start of", "began", "begin", "started", "created", "created in"),
    "end_date": ("ended in", "ceased in", "end date of", "end year of", "end year", "ended", "ceased", "abolished", "dissolved"),
    "population": ("population of", "population"),
    "empties_into": ("empties into", "empty into", "flows into"),
    "governor_of": ("governor of",),
    "screenwriter_of": ("screenwriter of",),
    "composer_of": ("composer of",),
    "designer_of": ("designer of", "designed by"),
    "named_after": ("named after",),
}
_PREDICATE_ALIAS_TO_CANONICAL = {
    normalize_structure_text(alias): canonical
    for canonical, aliases in _PREDICATE_ALIAS_FAMILIES.items()
    for alias in aliases
}
_PREDICATE_ALIASES_BY_LENGTH = sorted(
    _PREDICATE_ALIAS_TO_CANONICAL.items(),
    key=lambda item: len(item[0]),
    reverse=True,
)
_DIRECT_REJECT_PREDICATES = {
    "person_goals",
    "publisher_end",
    "birthplace_abolished",
    "region_immediately",
    "body_water",
    "many_times",
    "de_la",
}
_ROLE_CONSTRAINT_ALIASES = {
    "played by": "played_by",
    "portrayed by": "played_by",
    "directed by": "directed_by",
    "starred in": "starred_in",
    "written by": "written_by",
    "founded by": "founded_by",
}
_ANSWER_TYPE_PREFIX_NORMALIZATIONS = {
    "character": "person_or_character",
    "city": "city",
    "country": "country",
    "county": "county",
    "date": "date",
    "film": "film",
    "location": "location",
    "month": "month",
    "name": "person_or_character",
    "number": "number",
    "person": "person_or_character",
    "place": "location",
    "time": "date",
    "year": "date",
}
_QUESTION_RELATION_HINTS = tuple(sorted((
    ("record label", "publisher"),
    ("headquarters of", "headquartered_in"),
    ("headquarters", "headquartered_in"),
    ("headquartered", "headquartered_in"),
    ("birthplace", "birth_place"),
    ("birth place", "birth_place"),
    ("died", "death_place"),
    ("county", "county"),
    ("north of", "north_of"),
    ("immediately north of", "north_of"),
    ("mascot", "mascot"),
    ("director", "directed_by"),
    ("directed", "directed_by"),
    ("played by", "played_by"),
    ("designer", "designer_of"),
    ("publisher", "publisher"),
    ("signed by", "signed_by"),
    ("signed", "signed_by"),
    ("goals scored", "scored_goal"),
    ("goals", "scored_goal"),
    ("goal", "scored_goal"),
    ("empties into", "empties_into"),
    ("empty into", "empties_into"),
    ("flows into", "empties_into"),
    ("began", "start_date"),
    ("begin", "start_date"),
    ("started", "start_date"),
    ("created", "start_date"),
    ("abolished", "end_date"),
    ("ended", "end_date"),
    ("ceased", "end_date"),
), key=lambda item: len(item[0]), reverse=True))
_QDMR_PARSER_SYSTEM_PROMPT = """You are a query decomposition planner for multi-hop retrieval.

Your job is to decompose a question into a small sequence of semantic reasoning steps.
The steps must describe what information is needed to answer the question, not just repeat keywords.

Requirements:
1. Output 2 to 4 steps only.
2. Each step must be necessary for solving the question.
3. Use explicit intermediate variables such as X, Y, Z, ANSWER when needed.
4. Distinguish:
   - locating an entity/event,
   - retrieving a relation value,
   - checking a constraint,
   - producing the final answer.
5. Do NOT output irrelevant topical phrases.
6. Do NOT use wh-words (who/what/when/where/which/how/why/whom/whose) as entities.
7. Do NOT directly answer the question.
8. Prefer relation- and constraint-centered steps over keyword overlap.
9. Prefer steps that identify:
   - the bridge variable needed for the next hop,
   - the relation needed to retrieve it,
   - any role/time/comparison constraint required to avoid confusable distractors.
10. Avoid steps that only restate surface words from the question.

Return JSON only."""
_QDMR_PARSER_FEW_SHOTS = """Example A
Question:
Who is the mascot of the university related to Randy Conrads?
Output:
{"question_id":"fs_a","answer_type":"person_or_character","qdmr_steps":[{"step_id":"s1","operation":"relation_lookup","description":"Find the university associated with Randy Conrads.","inputs":["Randy Conrads"],"output_variable":"X"},{"step_id":"s2","operation":"relation_lookup","description":"Find the mascot of X.","inputs":["X"],"output_variable":"ANSWER"},{"step_id":"s3","operation":"answer","description":"Return ANSWER as the final answer.","inputs":["ANSWER"],"output_variable":"ANSWER"}]}

Example B
Question:
Which film directed by Christopher Nolan was released before 2010?
Output:
{"question_id":"fs_b","answer_type":"film","qdmr_steps":[{"step_id":"s1","operation":"relation_lookup","description":"Find films directed by Christopher Nolan.","inputs":["Christopher Nolan"],"output_variable":"X"},{"step_id":"s2","operation":"constraint_check","description":"Keep the film X whose release date is before 2010.","inputs":["X"],"output_variable":"X"},{"step_id":"s3","operation":"answer","description":"Return X as the final answer.","inputs":["X"],"output_variable":"ANSWER"}]}

Example C
Question:
Which city has the larger population, the birthplace of Author A or the birthplace of Author B?
Output:
{"question_id":"fs_c","answer_type":"city","qdmr_steps":[{"step_id":"s1","operation":"relation_lookup","description":"Find the birthplace of Author A.","inputs":["Author A"],"output_variable":"X"},{"step_id":"s2","operation":"relation_lookup","description":"Find the birthplace of Author B.","inputs":["Author B"],"output_variable":"Y"},{"step_id":"s3","operation":"constraint_check","description":"Compare the populations of X and Y and keep the city with the larger population.","inputs":["X","Y"],"output_variable":"ANSWER"},{"step_id":"s4","operation":"answer","description":"Return ANSWER as the final answer.","inputs":["ANSWER"],"output_variable":"ANSWER"}]}

Example D
Question:
Who is played by the director of The Good Shepherd in The Godfather?
Output:
{"question_id":"fs_d","answer_type":"person_or_character","qdmr_steps":[{"step_id":"s1","operation":"locate_entity","description":"Identify the film The Good Shepherd.","inputs":["The Good Shepherd"],"output_variable":"X"},{"step_id":"s2","operation":"relation_lookup","description":"Find the director of X.","inputs":["X"],"output_variable":"Y"},{"step_id":"s3","operation":"relation_lookup","description":"Find the character played by Y in The Godfather.","inputs":["Y","The Godfather"],"output_variable":"ANSWER"},{"step_id":"s4","operation":"answer","description":"Return ANSWER as the final answer.","inputs":["ANSWER"],"output_variable":"ANSWER"}]}

Example E
Question:
When was the region immediately north of the region where Israel is located created?
Output:
{"question_id":"fs_e","answer_type":"date","qdmr_steps":[{"step_id":"s1","operation":"locate_entity","description":"Identify the region where Israel is located.","inputs":["Israel"],"output_variable":"X"},{"step_id":"s2","operation":"relation_lookup","description":"Find the region immediately north of X.","inputs":["X"],"output_variable":"Y"},{"step_id":"s3","operation":"relation_lookup","description":"Find the start date of Y.","inputs":["Y"],"output_variable":"ANSWER"},{"step_id":"s4","operation":"answer","description":"Return ANSWER as the final answer.","inputs":["ANSWER"],"output_variable":"ANSWER"}]}

Example F
Question:
Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?
Output:
{"question_id":"fs_f","answer_type":"location","qdmr_steps":[{"step_id":"s1","operation":"relation_lookup","description":"Find the designer of Southeast Library.","inputs":["Southeast Library"],"output_variable":"X"},{"step_id":"s2","operation":"relation_lookup","description":"Find the death place of X.","inputs":["X"],"output_variable":"Y"},{"step_id":"s3","operation":"relation_lookup","description":"Find the body of water near Y that empties into the Gulf of Mexico.","inputs":["Y","Gulf of Mexico"],"output_variable":"ANSWER"},{"step_id":"s4","operation":"answer","description":"Return ANSWER as the final answer.","inputs":["ANSWER"],"output_variable":"ANSWER"}]}"""


def tokenize_text(text: str | None) -> List[str]:
    normalized = normalize_structure_text(text or "")
    if not normalized:
        return []
    return [token for token in _TOKEN_RE.findall(normalized) if len(token) > 1]


def normalize_entity_text(text: str | None) -> str:
    normalized = normalize_structure_text(text or "")
    if not normalized:
        return ""
    normalized = re.sub(r"\bs\b$", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def canonicalize_unit_variable_reference(value: Any) -> str:
    normalized = normalize_structure_text(str(value or ""))
    if normalized in {"x", "?x"}:
        return "?x"
    if normalized in {"y", "?y"}:
        return "?y"
    if normalized in {"z", "?z"}:
        return "?z"
    if normalized in {"ans", "answer", "?ans"}:
        return "?ans"
    return ""


def normalize_need_unit_slot_value(value: Any) -> str:
    canonical_variable = canonicalize_unit_variable_reference(value)
    if canonical_variable:
        return canonical_variable
    return normalize_structure_text(str(value or ""))


def unique_ordered_texts(values: Sequence[str] | Set[str] | None) -> List[str]:
    deduped: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        normalized = normalize_entity_text(str(value))
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
        candidate = normalize_entity_text(match.group(0))
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
    if lowered.startswith("what month"):
        return ["month", "date", "time"]
    if lowered.startswith("what county"):
        return ["county", "location", "region"]
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


def guess_answer_type_label(question: str) -> str:
    lowered = normalize_structure_text(question or "")
    if not lowered:
        return "answer"
    if lowered.startswith("who") or lowered.startswith("whom"):
        return "person_or_character"
    if lowered.startswith("where"):
        return "location"
    if lowered.startswith("what month"):
        return "month"
    if lowered.startswith("what county"):
        return "county"
    if lowered.startswith("what city"):
        return "city"
    if lowered.startswith("what country"):
        return "country"
    if lowered.startswith("when") or lowered.startswith("what year"):
        return "date"
    if lowered.startswith("how many") or lowered.startswith("how much"):
        return "number"
    focus_terms = extract_question_focus_terms(question, max_terms=2)
    if focus_terms:
        return "_".join(focus_terms)
    return "answer"


def normalize_answer_type_label(value: str | Sequence[str] | None, question: str = "") -> str:
    if isinstance(value, str):
        normalized = normalize_structure_text(value)
        if normalized:
            prefix = normalized.split(" ", 1)[0]
            if prefix in _ANSWER_TYPE_PREFIX_NORMALIZATIONS:
                return _ANSWER_TYPE_PREFIX_NORMALIZATIONS[prefix]
            return normalized.replace(" ", "_")
    elif value:
        tokens = [normalize_structure_text(item) for item in value if normalize_structure_text(item)]
        if tokens:
            prefix = tokens[0].split(" ", 1)[0]
            if prefix in _ANSWER_TYPE_PREFIX_NORMALIZATIONS:
                return _ANSWER_TYPE_PREFIX_NORMALIZATIONS[prefix]
            return "_".join(tokens[:2]).replace(" ", "_")
    return guess_answer_type_label(question)


def is_need_unit_cache_version(value: str | Dict[str, Any] | None) -> bool:
    if isinstance(value, dict):
        resolved = get_requirement_cache_version(value)
    else:
        resolved = str(value or "").strip()
    return resolved in NEED_UNIT_CACHE_VERSIONS


def _normalize_llm_infer_result(infer_result: Any) -> tuple[str, Dict[str, Any]]:
    if isinstance(infer_result, tuple):
        if len(infer_result) == 3:
            response_text, metadata, cache_hit = infer_result
            resolved_metadata = dict(metadata or {})
            resolved_metadata["cache_hit"] = cache_hit
            return str(response_text or ""), resolved_metadata
        if len(infer_result) == 2:
            response_text, metadata = infer_result
            return str(response_text or ""), dict(metadata or {})
    raise ValueError(f"Unexpected LLM infer result: {type(infer_result)!r}")


def _clean_structured_payload_text(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return cleaned
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        return cleaned[json_start:json_end + 1]
    return cleaned


def _safe_json_object_from_text(raw_text: str) -> Dict[str, Any]:
    cleaned = _clean_structured_payload_text(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        fixed = fix_broken_generated_json(cleaned)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(fixed)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _is_variable_token(value: str | None) -> bool:
    return bool(canonicalize_unit_variable_reference(value))


def _canonicalize_variable_token(value: str | None) -> str:
    return canonicalize_unit_variable_reference(value)


def _build_constraint(constraint_type: str, value: str) -> Dict[str, Any]:
    normalized_type = str(constraint_type or "").strip()
    normalized_value = normalize_structure_text(value or "")
    return {
        "type": normalized_type,
        "kind": normalized_type,
        "value": normalized_value,
    }


def _normalize_constraint_list(constraints: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen: Set[tuple[str, str]] = set()
    for constraint in constraints or []:
        constraint_type = str(constraint.get("type", constraint.get("kind", ""))).strip()
        value = normalize_structure_text(constraint.get("value", ""))
        if not constraint_type or not value:
            continue
        key = (constraint_type, value)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(_build_constraint(constraint_type, value))
    normalized.sort(key=lambda item: (_CONSTRAINT_PRIORITY.get(str(item.get("type", "")), 99), str(item.get("value", ""))))
    return normalized[:2]


def get_requirement_cache_version(payload_or_entry: Dict[str, Any] | None) -> str:
    version = str((payload_or_entry or {}).get("version", "")).strip()
    if version:
        return version
    if payload_or_entry and "positive_need_units" in payload_or_entry:
        return NEED_UNIT_CACHE_VERSION
    return REQUIREMENT_CACHE_VERSION


def get_requirement_group_types(cache_entry: Dict[str, Any]) -> tuple[str, ...]:
    if is_need_unit_cache_version(cache_entry):
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
                    object_value: str | None,
                    constraints: Sequence[Dict[str, Any]] | None = None,
                    target_variable: str = "",
                    answer_relevance: bool = False,
                    source_step_ids: Sequence[str] | None = None,
                    raw_predicate_text: str = "",
                    confidence: str = "high",
                    selector_enabled: bool = True,
                    debug: Dict[str, Any] | None = None) -> Dict[str, Any]:
    normalized_subject = normalize_need_unit_slot_value(subject or "")
    normalized_predicate = normalize_predicate_text(predicate or "")
    normalized_object = normalize_need_unit_slot_value(object_value or "") if object_value is not None else ""
    normalized_target_variable = _canonicalize_variable_token(target_variable) or normalize_structure_text(str(target_variable or ""))
    normalized_constraints = _normalize_constraint_list(constraints)
    raw_text_parts = [
        normalized_subject,
        normalized_predicate.replace("_", " "),
        normalized_object,
        " ".join(f"{item['type']} {item['value']}" for item in normalized_constraints),
    ]
    normalized_text = normalize_structure_text(" ".join(part for part in raw_text_parts if part))
    tokens = [token for token in tokenize_text(normalized_text) if token not in _STOPWORDS]
    entities = unique_ordered_texts([
        normalized_subject if normalized_subject and not normalized_subject.startswith("?") else "",
        normalized_object if normalized_object and not normalized_object.startswith("?") else "",
    ])
    answer_type_tokens = [
        token
        for constraint in normalized_constraints
        if str(constraint.get("type", "")) == "answer_type"
        for token in tokenize_text(constraint.get("value", ""))
    ]
    return {
        "unit_id": str(unit_id),
        "requirement_id": str(unit_id),
        "unit_type": str(unit_type),
        "type": str(unit_type),
        "target_variable": str(normalized_target_variable or ""),
        "subject": str(normalized_subject or subject or ""),
        "predicate": str(normalized_predicate or predicate or ""),
        "raw_predicate_text": str(raw_predicate_text or ""),
        "object": str(normalized_object) if object_value is not None else None,
        "constraints": normalized_constraints,
        "answer_relevance": bool(answer_relevance),
        "source_step_ids": list(source_step_ids or []),
        "confidence": str(confidence or "high"),
        "selector_enabled": bool(selector_enabled),
        "debug": dict(debug or {}),
        "text": normalized_text,
        "normalized_text": normalized_text,
        "tokens": tokens,
        "entities": entities,
        "answer_type_tokens": answer_type_tokens,
    }


def _extract_question_constraints(question: str) -> List[Dict[str, Any]]:
    normalized_question = normalize_structure_text(question or "")
    constraints: List[Dict[str, Any]] = []
    for pattern in (
        r"\bbefore\s+\d{4}\b",
        r"\bafter\s+\d{4}\b",
        r"\bin\s+\d{4}\b",
        r"\bduring\s+\d{4}\b",
        r"\bfirst term\b",
        r"\bsecond term\b",
    ):
        for match in re.finditer(pattern, normalized_question):
            constraints.append(_build_constraint("temporal", match.group(0)))
    for pattern in (
        r"\blargest\b",
        r"\bsmallest\b",
        r"\bhigher\b",
        r"\blower\b",
        r"\bearlier\b",
        r"\blater\b",
        r"\bgreater population\b",
        r"\bless population\b",
        r"\bfirst\b",
        r"\bsecond\b",
    ):
        for match in re.finditer(pattern, normalized_question):
            constraints.append(_build_constraint("comparative", match.group(0)))
    for pattern in (r"\bnot\b", r"\bwithout\b", r"\bexcluding\b"):
        for match in re.finditer(pattern, normalized_question):
            constraints.append(_build_constraint("negation", match.group(0)))
    for pattern in (r"\bexcept\b", r"\bincluding\b", r"\bamong\b"):
        for match in re.finditer(pattern, normalized_question):
            constraints.append(_build_constraint("scope", match.group(0)))
    return _normalize_constraint_list(constraints)


def _extract_relation_predicates(question: str,
                                 anchor_entities: Sequence[str],
                                 answer_type_tokens: Sequence[str]) -> List[str]:
    normalized_question = normalize_structure_text(question or "")
    matched_predicates: List[tuple[int, str]] = []
    seen_matches: Set[tuple[int, str]] = set()
    for raw_phrase, predicate in _PREDICATE_ALIASES_BY_LENGTH:
        match_index = normalized_question.find(raw_phrase)
        if match_index == -1:
            continue
        match_key = (match_index, predicate)
        if match_key in seen_matches:
            continue
        seen_matches.add(match_key)
        matched_predicates.append((match_index, predicate))
    for raw_phrase, predicate in _QUESTION_RELATION_HINTS:
        match_index = normalized_question.find(raw_phrase)
        if match_index == -1:
            continue
        match_key = (match_index, predicate)
        if match_key in seen_matches:
            continue
        seen_matches.add(match_key)
        matched_predicates.append((match_index, predicate))

    predicates = [predicate for _, predicate in sorted(matched_predicates, key=lambda item: item[0])]

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
            candidate = normalize_predicate_text("_".join(filtered_focus[:2]))
            if candidate in _CANONICAL_PREDICATES:
                predicates.append(candidate)
        elif answer_type_tokens:
            answer_tokens = [token for token in answer_type_tokens if token in _CANONICAL_PREDICATES]
            predicates.extend(answer_tokens[:1])

    deduped: List[str] = []
    seen: Set[str] = set()
    for predicate in predicates:
        normalized_predicate = normalize_predicate_text(predicate)
        if not normalized_predicate or normalized_predicate in seen:
            continue
        seen.add(normalized_predicate)
        deduped.append(normalized_predicate)
    return deduped[:2]


def _default_need_unit_diagnostics(parser_status: str = "ok") -> Dict[str, Any]:
    return {
        "schema_version": NEED_UNIT_CACHE_VERSION,
        "parser_status": parser_status,
        "compiler_status": "ok",
        "raw_step_count": 0,
        "normalized_step_count": 0,
        "malformed_step_count": 0,
        "malformed_unit_count": 0,
        "dropped_step_count": 0,
        "dropped_unit_count": 0,
        "low_confidence_unit_count": 0,
        "low_confidence_relation_hop_count": 0,
        "wh_repair_count": 0,
        "answer_step_inserted": False,
        "positive_need_unit_count": 0,
        "counterfactual_set_count": 0,
        "selector_enabled_unit_count": 0,
        "relation_hop_cap": DEFAULT_NEED_UNIT_MAX_RELATION_HOPS,
        "relation_hop_cap_mode": DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE,
        "pre_cap_need_unit_count": 0,
        "post_cap_need_unit_count": 0,
        "pre_cap_relation_hop_count": 0,
        "post_cap_relation_hop_count": 0,
        "conditional_third_hop_allowed": False,
        "conditional_third_hop_reasons": [],
        "dropped_unit_predicates": [],
        "dropped_unit_types": [],
        "notes": [],
    }


def _attach_parser_trace_to_diagnostics(diagnostics: Dict[str, Any],
                                        parser_trace: Dict[str, Any] | None) -> Dict[str, Any]:
    resolved = dict(diagnostics or {})
    if not parser_trace:
        return resolved
    if parser_trace.get("error"):
        resolved["parser_error"] = str(parser_trace.get("error", ""))
    if parser_trace.get("raw_response_preview"):
        resolved["parser_raw_response_preview"] = str(parser_trace.get("raw_response_preview", ""))
    if parser_trace.get("metadata"):
        resolved["parser_metadata"] = dict(parser_trace.get("metadata", {}) or {})
    return resolved


def build_qdmr_parser_messages(question: str,
                               question_entities: Sequence[str] | Set[str] | None,
                               seed_entities: Sequence[str] | Set[str] | None,
                               pool_titles: Sequence[str] | None,
                               predicted_answer_type: str,
                               question_id: str) -> List[Dict[str, str]]:
    pool_titles_blob = json.dumps(list(pool_titles or []), ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": _QDMR_PARSER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"{_QDMR_PARSER_FEW_SHOTS}\n\n"
                f"Question:\n{question}\n\n"
                f"Question entities:\n{json.dumps(list(question_entities or []), ensure_ascii=False)}\n\n"
                f"Seed entities:\n{json.dumps(list(seed_entities or []), ensure_ascii=False)}\n\n"
                f"Candidate pool titles (for grounding only, do not copy blindly):\n{pool_titles_blob}\n\n"
                f"Predicted answer type:\n{predicted_answer_type}\n\n"
                "Return a JSON object with:\n"
                "{\n"
                '  "question_id": "...",\n'
                '  "answer_type": "...",\n'
                '  "qdmr_steps": [\n'
                "    {\n"
                '      "step_id": "s1",\n'
                '      "operation": "<one of: locate_entity, relation_lookup, constraint_check, answer>",\n'
                '      "description": "...",\n'
                '      "inputs": ["..."],\n'
                '      "output_variable": "X"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Guidelines:\n"
                '- "locate_entity": identify a core entity/event if disambiguation is needed.\n'
                '- "relation_lookup": retrieve a relation value from an entity or variable.\n'
                '- "constraint_check": apply time, comparison, role, negation, or scope constraints.\n'
                '- "answer": specify which variable or relation becomes the final answer.\n'
                "- Use at most one output variable per step.\n"
                "- If the question is 2-hop, usually use 3 steps.\n"
                "- If no disambiguation is needed, skip locate_entity.\n"
                f'- Set "question_id" to "{question_id}".'
            ),
        },
    ]


def infer_qdmr_step_plan(question: str,
                         question_entities: Sequence[str] | Set[str] | None,
                         seed_entities: Sequence[str] | Set[str] | None,
                         pool_titles: Sequence[str] | None,
                         predicted_answer_type: str,
                         llm_infer_fn: Any,
                         model_name: str | None = None,
                         max_completion_tokens: int = DEFAULT_QDMR_PARSER_MAX_COMPLETION_TOKENS) -> Dict[str, Any]:
    question_id = stable_question_key(question)
    messages = build_qdmr_parser_messages(
        question=question,
        question_entities=question_entities,
        seed_entities=seed_entities,
        pool_titles=pool_titles,
        predicted_answer_type=predicted_answer_type,
        question_id=question_id,
    )
    try:
        infer_result = llm_infer_fn(
            messages=messages,
            model=model_name,
            response_format={"type": "json_object"},
            max_completion_tokens=int(max_completion_tokens),
            temperature=0,
            top_p=1,
        )
        raw_response, metadata = _normalize_llm_infer_result(infer_result)
        cleaned_response = raw_response
        if metadata.get("finish_reason") == "length":
            cleaned_response = fix_broken_generated_json(raw_response)
        payload = _safe_json_object_from_text(cleaned_response)
        return {
            "payload": payload,
            "metadata": metadata,
            "error": "",
            "raw_response_preview": cleaned_response[:300],
        }
    except Exception as exc:
        return {
            "payload": None,
            "metadata": {},
            "error": str(exc),
            "raw_response_preview": "",
        }


def _heuristic_relation_description(predicate: str, subject_input: str) -> str:
    if predicate == "associated_university":
        return f"Find the university associated with {subject_input}."
    if predicate == "death_place":
        return f"Find the death place of {subject_input}."
    if predicate == "north_of":
        return f"Find the region immediately north of {subject_input}."
    if predicate == "start_date":
        return f"Find the start date of {subject_input}."
    if predicate == "end_date":
        return f"Find the end date of {subject_input}."
    if predicate == "empties_into":
        return f"Find where {subject_input} empties into."
    alias_candidates = _PREDICATE_ALIAS_FAMILIES.get(predicate, ())
    preferred_phrase = next(iter(alias_candidates), predicate.replace("_", " "))
    if preferred_phrase.endswith((" of", " for", " by", " in", " with")):
        return f"Find the {preferred_phrase} {subject_input}."
    return f"Find the {preferred_phrase} of {subject_input}."


def build_heuristic_qdmr_step_plan_payload(question: str,
                                           seed_entities: Sequence[str] | Set[str] | None,
                                           question_entities: Sequence[str] | Set[str] | None,
                                           predicted_answer_type: str | None = None) -> Dict[str, Any]:
    anchor_entities = extract_question_entities(
        question=question,
        fallback_entities=question_entities or seed_entities,
        max_entities=2,
    )
    answer_type_label = normalize_answer_type_label(predicted_answer_type, question=question)
    answer_type_tokens = tokenize_text(answer_type_label.replace("_", " "))
    predicates = _extract_relation_predicates(
        question=question,
        anchor_entities=anchor_entities,
        answer_type_tokens=answer_type_tokens,
    )
    constraints = _extract_question_constraints(question)
    steps: List[Dict[str, Any]] = []
    relation_subject = anchor_entities[0] if anchor_entities else (unique_ordered_texts(question_entities or seed_entities)[:1] or ["focus entity"])[0]
    relation_outputs = ["X", "Y"]
    for hop_idx in range(min(max(len(predicates), 1), 2)):
        predicate = predicates[hop_idx] if hop_idx < len(predicates) else "located_in"
        output_variable = relation_outputs[min(hop_idx, len(relation_outputs) - 1)]
        subject_input = relation_subject if hop_idx == 0 else relation_outputs[hop_idx - 1]
        description = _heuristic_relation_description(predicate=predicate, subject_input=subject_input)
        steps.append({
            "step_id": f"s{len(steps) + 1}",
            "operation": "relation_lookup",
            "description": description,
            "inputs": [subject_input],
            "output_variable": output_variable,
        })
    current_variable = relation_outputs[min(max(len(steps), 1) - 1, len(relation_outputs) - 1)]
    if constraints:
        first_constraint = constraints[0]
        steps.append({
            "step_id": f"s{len(steps) + 1}",
            "operation": "constraint_check",
            "description": f"Keep {current_variable} satisfying {first_constraint['value']}.",
            "inputs": [current_variable],
            "output_variable": current_variable,
        })
    answer_source = current_variable if steps else "X"
    steps.append({
        "step_id": f"s{len(steps) + 1}",
        "operation": "answer",
        "description": f"Return {answer_source} as the final answer.",
        "inputs": [answer_source],
        "output_variable": "ANSWER",
    })
    return {
        "question_id": stable_question_key(question),
        "answer_type": answer_type_label,
        "qdmr_steps": steps[:4],
    }


def _resolve_description_entities(description: str,
                                  candidates: Sequence[str] | Set[str] | None) -> List[str]:
    normalized_description = normalize_structure_text(description or "")
    matched: List[str] = []
    for candidate in unique_ordered_texts(candidates):
        if candidate and candidate in normalized_description:
            matched.append(candidate)
    return matched


def _repair_wh_value(value: str,
                     inputs: Sequence[str],
                     question_entities: Sequence[str] | Set[str] | None,
                     seed_entities: Sequence[str] | Set[str] | None,
                     diagnostics: Dict[str, Any]) -> str:
    normalized_value = normalize_entity_text(value or "")
    if normalized_value not in _WH_WORDS:
        return normalized_value
    for item in inputs:
        normalized_item = normalize_entity_text(item)
        if not _is_variable_token(item) and normalized_item not in _WH_WORDS:
            diagnostics["wh_repair_count"] = int(diagnostics.get("wh_repair_count", 0)) + 1
            return normalized_item
    for candidate in unique_ordered_texts(question_entities):
        if candidate not in _WH_WORDS:
            diagnostics["wh_repair_count"] = int(diagnostics.get("wh_repair_count", 0)) + 1
            return candidate
    for candidate in unique_ordered_texts(seed_entities):
        if candidate not in _WH_WORDS:
            diagnostics["wh_repair_count"] = int(diagnostics.get("wh_repair_count", 0)) + 1
            return candidate
    return ""


def _extract_constraints_from_text(description: str,
                                   extra_inputs: Sequence[str] | None = None,
                                   excluded_phrases: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    normalized_description = normalize_structure_text(description or "")
    for phrase in excluded_phrases or []:
        normalized_phrase = normalize_structure_text(phrase)
        if normalized_phrase:
            normalized_description = normalized_description.replace(normalized_phrase, " ")
    constraints: List[Dict[str, Any]] = []
    for pattern in (
        r"\bbefore\s+\d{4}\b",
        r"\bafter\s+\d{4}\b",
        r"\bin\s+\d{4}\b",
        r"\bduring\s+\d{4}\b",
        r"\bfirst term\b",
        r"\bsecond term\b",
    ):
        for match in re.finditer(pattern, normalized_description):
            constraints.append(_build_constraint("temporal", match.group(0)))
    for pattern in (
        r"\blargest\b",
        r"\bsmallest\b",
        r"\bhigher\b",
        r"\blower\b",
        r"\bearlier\b",
        r"\blater\b",
        r"\bgreater population\b",
        r"\bless population\b",
        r"\bfirst\b",
        r"\bsecond\b",
    ):
        for match in re.finditer(pattern, normalized_description):
            constraints.append(_build_constraint("comparative", match.group(0)))
    for raw_phrase, canonical_role in _ROLE_CONSTRAINT_ALIASES.items():
        if raw_phrase in normalized_description:
            constraints.append(_build_constraint("role", canonical_role))
    for pattern in (r"\bnot\b", r"\bwithout\b", r"\bexcluding\b"):
        for match in re.finditer(pattern, normalized_description):
            constraints.append(_build_constraint("negation", match.group(0)))
    for pattern in (r"\bexcept\b", r"\bincluding\b", r"\bamong\b"):
        for match in re.finditer(pattern, normalized_description):
            constraints.append(_build_constraint("scope", match.group(0)))
    for item in list(extra_inputs or [])[1:]:
        if _is_variable_token(item):
            continue
        normalized_item = normalize_structure_text(item)
        if normalized_item:
            constraints.append(_build_constraint("scope", normalized_item))
    return _normalize_constraint_list(constraints)


def _extract_raw_predicate_candidate(description: str,
                                     inputs: Sequence[str] | None = None) -> str:
    normalized_description = normalize_structure_text(description or "")
    for item in inputs or []:
        normalized_item = normalize_structure_text(item)
        if normalized_item:
            normalized_description = normalized_description.replace(normalized_item, " ")
    for boilerplate in (
        "find the",
        "find",
        "identify the",
        "identify",
        "keep the",
        "keep",
        "return",
        "as the final answer",
        "final answer",
        "compare the",
        "compare",
        "whose",
    ):
        normalized_description = normalized_description.replace(boilerplate, " ")
    normalized_description = re.sub(r"\s+", " ", normalized_description).strip()
    pattern_candidates = [
        re.search(r"\b([a-z ]+?)\s+of\b", normalized_description),
        re.search(r"\b([a-z ]+?)\s+for\b", normalized_description),
        re.search(r"\b([a-z ]+?)\s+associated with\b", normalized_description),
        re.search(r"\b([a-z ]+?)\s+related to\b", normalized_description),
    ]
    for match in pattern_candidates:
        if match:
            return normalize_predicate_text(match.group(1))
    tokens = [
        token for token in tokenize_text(normalized_description)
        if token not in _STOPWORDS and token not in _WH_WORDS
    ]
    return normalize_predicate_text(" ".join(tokens[:3]))


def _is_direct_reject_predicate(predicate_text: str) -> bool:
    normalized = normalize_predicate_text(predicate_text)
    if not normalized:
        return True
    if normalized in _DIRECT_REJECT_PREDICATES:
        return True
    tokens = [token for token in tokenize_text(normalized.replace("_", " ")) if token not in _STOPWORDS]
    if not tokens:
        return True
    if all(token in _WH_WORDS or token in _GENERIC_PREDICATE_TOKENS for token in tokens):
        return True
    return False


def _normalize_predicate_from_description(description: str,
                                          inputs: Sequence[str] | None = None) -> Dict[str, Any]:
    normalized_description = normalize_structure_text(description or "")
    pattern_overrides = (
        (r"\b(city|place|location)\s+where\b.*\bdied\b", "death_place", "city where died"),
        (r"\b(city|place|location)\s+where\b.*\bwas born\b", "birth_place", "city where born"),
        (r"\b(city|place|location)\s+where\b.*\bis from\b", "birth_place", "city where from"),
        (r"\b(immediately\s+)?north of\b", "north_of", "north of"),
    )
    for pattern, canonical, raw_text in pattern_overrides:
        if re.search(pattern, normalized_description):
            return {
                "predicate": canonical,
                "raw_predicate_text": raw_text,
                "confidence": "high",
                "selector_enabled": True,
            }
    for canonical in sorted(_CANONICAL_PREDICATES, key=len, reverse=True):
        canonical_text = canonical.replace("_", " ")
        if canonical_text and canonical_text in normalized_description:
            return {
                "predicate": canonical,
                "raw_predicate_text": canonical_text,
                "confidence": "high",
                "selector_enabled": True,
            }
    for alias, canonical in _PREDICATE_ALIASES_BY_LENGTH:
        if alias in normalized_description:
            return {
                "predicate": canonical,
                "raw_predicate_text": alias,
                "confidence": "high",
                "selector_enabled": True,
            }
    raw_candidate = _extract_raw_predicate_candidate(description=description, inputs=inputs)
    return {
        "predicate": raw_candidate or "unmapped_relation",
        "raw_predicate_text": raw_candidate,
        "confidence": "low",
        "selector_enabled": False,
    }


def _normalize_step_plan_payload(step_plan_payload: Dict[str, Any] | None,
                                 question: str,
                                 predicted_answer_type: str) -> Dict[str, Any]:
    diagnostics = _default_need_unit_diagnostics(parser_status="ok" if step_plan_payload else "heuristic")
    if not step_plan_payload:
        return {
            "question_id": stable_question_key(question),
            "answer_type": normalize_answer_type_label(predicted_answer_type, question=question),
            "qdmr_steps": [],
            "diagnostics": diagnostics,
        }
    raw_steps = list(step_plan_payload.get("qdmr_steps", []) or [])
    diagnostics["raw_step_count"] = int(len(raw_steps))
    normalized_steps: List[Dict[str, Any]] = []
    for raw_index, raw_step in enumerate(raw_steps[:4]):
        if not isinstance(raw_step, dict):
            diagnostics["malformed_step_count"] = int(diagnostics.get("malformed_step_count", 0)) + 1
            diagnostics["dropped_step_count"] = int(diagnostics.get("dropped_step_count", 0)) + 1
            diagnostics["notes"].append(f"step_{raw_index}_not_object")
            continue
        operation = str(raw_step.get("operation", "") or "").strip().lower()
        if operation not in _QDMR_ALLOWED_OPERATIONS:
            diagnostics["malformed_step_count"] = int(diagnostics.get("malformed_step_count", 0)) + 1
            diagnostics["dropped_step_count"] = int(diagnostics.get("dropped_step_count", 0)) + 1
            diagnostics["notes"].append(f"step_{raw_index}_unknown_operation")
            continue
        description = re.sub(r"[ ]{2,}", " ", str(raw_step.get("description", "") or "").strip())
        raw_inputs = raw_step.get("inputs", [])
        if not isinstance(raw_inputs, list) or not raw_inputs:
            diagnostics["malformed_step_count"] = int(diagnostics.get("malformed_step_count", 0)) + 1
            diagnostics["dropped_step_count"] = int(diagnostics.get("dropped_step_count", 0)) + 1
            diagnostics["notes"].append(f"step_{raw_index}_empty_inputs")
            continue
        normalized_inputs: List[str] = []
        for item in raw_inputs[:3]:
            item_text = str(item or "").strip()
            canonical_variable = _canonicalize_variable_token(item_text)
            normalized_inputs.append(canonical_variable or item_text)
        output_variable = _canonicalize_variable_token(raw_step.get("output_variable", ""))
        if operation == "answer" and output_variable != "?ans":
            output_variable = "?ans"
            diagnostics["notes"].append(f"step_{raw_index}_answer_output_repaired")
        if not output_variable:
            diagnostics["malformed_step_count"] = int(diagnostics.get("malformed_step_count", 0)) + 1
            diagnostics["dropped_step_count"] = int(diagnostics.get("dropped_step_count", 0)) + 1
            diagnostics["notes"].append(f"step_{raw_index}_invalid_output_variable")
            continue
        normalized_steps.append({
            "step_id": f"s{len(normalized_steps) + 1}",
            "operation": operation,
            "description": description,
            "inputs": normalized_inputs,
            "output_variable": output_variable,
            "debug": {
                "raw_step_id": str(raw_step.get("step_id", "")),
                "raw_output_variable": str(raw_step.get("output_variable", "")),
                "raw_description": str(raw_step.get("description", "")),
            },
        })
    diagnostics["normalized_step_count"] = int(len(normalized_steps))
    return {
        "question_id": str(step_plan_payload.get("question_id") or stable_question_key(question)),
        "answer_type": normalize_answer_type_label(step_plan_payload.get("answer_type"), question=question) or normalize_answer_type_label(predicted_answer_type, question=question),
        "qdmr_steps": normalized_steps,
        "diagnostics": diagnostics,
    }


def _resolve_relation_subject(step: Dict[str, Any],
                              question_entities: Sequence[str] | Set[str] | None,
                              seed_entities: Sequence[str] | Set[str] | None,
                              diagnostics: Dict[str, Any]) -> str:
    inputs = list(step.get("inputs", []))
    if inputs:
        first_input = str(inputs[0])
        if _is_variable_token(first_input):
            return _canonicalize_variable_token(first_input)
        repaired = _repair_wh_value(first_input, inputs, question_entities, seed_entities, diagnostics)
        if repaired:
            return repaired
        return normalize_entity_text(first_input)
    description_entities = _resolve_description_entities(step.get("description", ""), question_entities)
    if description_entities:
        return description_entities[0]
    seed_matches = _resolve_description_entities(step.get("description", ""), seed_entities)
    if seed_matches:
        return seed_matches[0]
    return ""


def _compile_entity_locator(step: Dict[str, Any],
                            question_entities: Sequence[str] | Set[str] | None,
                            seed_entities: Sequence[str] | Set[str] | None,
                            diagnostics: Dict[str, Any],
                            unit_index: int) -> Dict[str, Any] | None:
    inputs = list(step.get("inputs", []))
    subject = ""
    for item in inputs:
        if not _is_variable_token(item):
            subject = _repair_wh_value(str(item), inputs, question_entities, seed_entities, diagnostics)
            break
    if not subject:
        description_entities = _resolve_description_entities(step.get("description", ""), question_entities or seed_entities)
        subject = description_entities[0] if description_entities else ""
    if not subject:
        diagnostics["malformed_unit_count"] = int(diagnostics.get("malformed_unit_count", 0)) + 1
        diagnostics["notes"].append("missing_subject_entity_locator")
        return None
    confidence = "high" if subject in unique_ordered_texts(question_entities) or subject in unique_ordered_texts(seed_entities) else "medium"
    return build_need_unit(
        unit_id=f"u_entity_locator_{unit_index}",
        unit_type="entity_locator",
        subject=subject,
        predicate="identify",
        raw_predicate_text="identify",
        object_value=str(step.get("output_variable", "?x")),
        target_variable=str(step.get("output_variable", "?x")),
        answer_relevance=False,
        source_step_ids=[str(step.get("step_id", ""))],
        confidence=confidence,
        selector_enabled=True,
        debug={"parser_description": str(step.get("description", ""))},
    )


def _compile_relation_hop(step: Dict[str, Any],
                          question_entities: Sequence[str] | Set[str] | None,
                          seed_entities: Sequence[str] | Set[str] | None,
                          diagnostics: Dict[str, Any],
                          unit_index: int) -> Dict[str, Any] | None:
    subject = _resolve_relation_subject(step, question_entities, seed_entities, diagnostics)
    if not subject:
        diagnostics["malformed_unit_count"] = int(diagnostics.get("malformed_unit_count", 0)) + 1
        diagnostics["notes"].append("missing_subject_relation_hop")
        return None
    predicate_info = _normalize_predicate_from_description(
        description=str(step.get("description", "")),
        inputs=step.get("inputs", []),
    )
    constraints = _extract_constraints_from_text(
        description=str(step.get("description", "")),
        extra_inputs=step.get("inputs", []),
        excluded_phrases=[str(predicate_info.get("raw_predicate_text", ""))],
    )
    confidence = str(predicate_info.get("confidence", "low"))
    selector_enabled = bool(predicate_info.get("selector_enabled", False))
    if confidence == "low":
        diagnostics["low_confidence_unit_count"] = int(diagnostics.get("low_confidence_unit_count", 0)) + 1
        diagnostics["low_confidence_relation_hop_count"] = int(diagnostics.get("low_confidence_relation_hop_count", 0)) + 1
    return build_need_unit(
        unit_id=f"u_relation_hop_{unit_index}",
        unit_type="relation_hop",
        subject=subject,
        predicate=str(predicate_info.get("predicate", "unmapped_relation")),
        raw_predicate_text=str(predicate_info.get("raw_predicate_text", "")),
        object_value=str(step.get("output_variable", "?x")),
        constraints=constraints,
        target_variable=str(step.get("output_variable", "?x")),
        answer_relevance=False,
        source_step_ids=[str(step.get("step_id", ""))],
        confidence=confidence,
        selector_enabled=selector_enabled,
        debug={"parser_description": str(step.get("description", ""))},
    )


def _compile_constraint_check(step: Dict[str, Any],
                              last_live_variable: str,
                              diagnostics: Dict[str, Any],
                              unit_index: int) -> Dict[str, Any] | None:
    inputs = list(step.get("inputs", []))
    subject = ""
    for item in inputs:
        if _is_variable_token(item):
            subject = _canonicalize_variable_token(item)
            break
    if not subject:
        subject = last_live_variable or (normalize_structure_text(inputs[0]) if inputs else "")
    constraints = _extract_constraints_from_text(
        description=str(step.get("description", "")),
        extra_inputs=inputs,
    )
    if not subject or not constraints:
        diagnostics["malformed_unit_count"] = int(diagnostics.get("malformed_unit_count", 0)) + 1
        diagnostics["notes"].append("empty_constraints_constraint_check")
        return None
    target_variable = str(step.get("output_variable", "")) or (subject if subject.startswith("?") else last_live_variable or "?x")
    return build_need_unit(
        unit_id=f"u_constraint_check_{unit_index}",
        unit_type="constraint_check",
        subject=subject,
        predicate="satisfy_constraint",
        raw_predicate_text="satisfy constraint",
        object_value=None,
        constraints=constraints,
        target_variable=target_variable,
        answer_relevance=False,
        source_step_ids=[str(step.get("step_id", ""))],
        confidence="high" if constraints else "medium",
        selector_enabled=True,
        debug={"parser_description": str(step.get("description", ""))},
    )


def _compile_answer_slot(step: Dict[str, Any],
                         answer_type: str,
                         last_live_variable: str,
                         diagnostics: Dict[str, Any],
                         unit_index: int) -> Dict[str, Any] | None:
    inputs = list(step.get("inputs", []))
    subject = ""
    for item in inputs:
        if _is_variable_token(item):
            subject = _canonicalize_variable_token(item)
            break
    if not subject and inputs:
        subject = normalize_structure_text(inputs[0])
    subject = subject or last_live_variable
    if not subject:
        diagnostics["malformed_unit_count"] = int(diagnostics.get("malformed_unit_count", 0)) + 1
        diagnostics["notes"].append("missing_answer_source")
        return None
    return build_need_unit(
        unit_id=f"u_answer_slot_{unit_index}",
        unit_type="answer_slot",
        subject=subject,
        predicate="return_as_answer",
        raw_predicate_text="return as answer",
        object_value="?ans",
        constraints=[_build_constraint("answer_type", answer_type)],
        target_variable="?ans",
        answer_relevance=True,
        source_step_ids=[str(step.get("step_id", ""))],
        confidence="high",
        selector_enabled=True,
        debug={"parser_description": str(step.get("description", ""))},
    )


def _ensure_answer_slot(units: Sequence[Dict[str, Any]],
                        answer_type: str,
                        diagnostics: Dict[str, Any]) -> List[Dict[str, Any]]:
    if any(str(unit.get("unit_type", "")) == "answer_slot" for unit in units):
        return list(units)
    resolved_units = list(units)
    fallback_subject = ""
    for unit in reversed(resolved_units):
        target_variable = str(unit.get("target_variable", "") or "")
        if target_variable:
            fallback_subject = target_variable
            break
        object_value = unit.get("object")
        if isinstance(object_value, str) and object_value:
            fallback_subject = object_value
            break
    if not fallback_subject:
        diagnostics["notes"].append("missing_answer_slot_fallback_subject")
        return resolved_units
    diagnostics["answer_step_inserted"] = True
    resolved_units.append(build_need_unit(
        unit_id=f"u_answer_slot_{len([unit for unit in resolved_units if unit.get('unit_type') == 'answer_slot'])}",
        unit_type="answer_slot",
        subject=fallback_subject,
        predicate="return_as_answer",
        raw_predicate_text="return as answer",
        object_value="?ans",
        constraints=[_build_constraint("answer_type", answer_type)],
        target_variable="?ans",
        answer_relevance=True,
        source_step_ids=[],
        confidence="high",
        selector_enabled=True,
        debug={"auto_inserted": True},
    ))
    return resolved_units


def _apply_need_unit_caps(units: Sequence[Dict[str, Any]],
                          diagnostics: Dict[str, Any],
                          max_relation_hops: int = DEFAULT_NEED_UNIT_MAX_RELATION_HOPS,
                          relation_hop_cap_mode: str = DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE,
                          max_total_units: int = 6) -> List[Dict[str, Any]]:
    relation_hop_cap = max(int(max_relation_hops), 1)
    cap_mode = str(relation_hop_cap_mode or DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE).strip().lower()
    if cap_mode not in {"fixed", "conditional"}:
        cap_mode = DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE
    limits = {
        "entity_locator": 1,
        "relation_hop": relation_hop_cap,
        "constraint_check": 2,
        "answer_slot": 1,
    }
    diagnostics["relation_hop_cap"] = int(relation_hop_cap)
    diagnostics["relation_hop_cap_mode"] = cap_mode
    diagnostics["pre_cap_need_unit_count"] = int(len(units))
    diagnostics["pre_cap_relation_hop_count"] = int(sum(1 for unit in units if str(unit.get("unit_type", "")) == "relation_hop"))
    diagnostics["conditional_third_hop_allowed"] = False
    diagnostics["conditional_third_hop_reasons"] = []
    kept: List[Dict[str, Any]] = []
    counts = {key: 0 for key in limits}
    dropped_unit_types: List[str] = list(diagnostics.get("dropped_unit_types", []) or [])
    dropped_unit_predicates: List[str] = list(diagnostics.get("dropped_unit_predicates", []) or [])

    if cap_mode == "conditional" and relation_hop_cap == 2:
        relation_units = [
            unit for unit in units
            if str(unit.get("unit_type", "")) == "relation_hop"
        ]
        reasons: List[str] = []
        if len(relation_units) < 3:
            reasons.append("fewer_than_three_relation_hops")
        else:
            first_three = relation_units[:3]
            if not all(
                str(unit.get("confidence", "")).strip().lower() == "high"
                and bool(unit.get("selector_enabled", True))
                for unit in first_three
            ):
                reasons.append("third_relation_chain_not_high_confidence")
            predicates = [
                normalize_predicate_text(unit.get("predicate", ""))
                for unit in first_three
            ]
            if len(set(predicate for predicate in predicates if predicate)) < len([predicate for predicate in predicates if predicate]):
                reasons.append("duplicate_relation_predicates")
            for prev_unit, next_unit in zip(first_three, first_three[1:]):
                prev_target = canonicalize_unit_variable_reference(prev_unit.get("target_variable")) or canonicalize_unit_variable_reference(prev_unit.get("object"))
                next_subject = canonicalize_unit_variable_reference(next_unit.get("subject"))
                if not prev_target or next_subject != prev_target:
                    reasons.append("non_contiguous_relation_chain")
                    break
            third_target = canonicalize_unit_variable_reference(first_three[2].get("target_variable")) or canonicalize_unit_variable_reference(first_three[2].get("object"))
            downstream_units = units[units.index(first_three[2]) + 1:] if first_three[2] in units else []
            if not third_target:
                reasons.append("third_relation_missing_target_variable")
            else:
                has_downstream_consumer = False
                for later_unit in downstream_units:
                    refs = {
                        canonicalize_unit_variable_reference(later_unit.get("subject")),
                        canonicalize_unit_variable_reference(later_unit.get("object")),
                        canonicalize_unit_variable_reference(later_unit.get("target_variable")),
                    }
                    if third_target in refs:
                        has_downstream_consumer = True
                        break
                if not has_downstream_consumer:
                    reasons.append("third_relation_not_connected_to_answer_chain")
        diagnostics["conditional_third_hop_reasons"] = list(reasons)
        if not reasons:
            diagnostics["conditional_third_hop_allowed"] = True
            limits["relation_hop"] = relation_hop_cap + 1

    def _record_drop(unit: Dict[str, Any], reason: str) -> None:
        diagnostics["dropped_unit_count"] = int(diagnostics.get("dropped_unit_count", 0)) + 1
        diagnostics["notes"].append(reason)
        unit_type = str(unit.get("unit_type", "") or "")
        predicate = str(unit.get("predicate", "") or "")
        if unit_type and unit_type not in dropped_unit_types:
            dropped_unit_types.append(unit_type)
        if predicate and predicate not in dropped_unit_predicates:
            dropped_unit_predicates.append(predicate)

    for unit in units:
        unit_type = str(unit.get("unit_type", ""))
        if counts.get(unit_type, 0) >= limits.get(unit_type, 99):
            _record_drop(unit, f"dropped_extra_{unit_type}")
            continue
        counts[unit_type] = counts.get(unit_type, 0) + 1
        kept.append(unit)
    if len(kept) > max_total_units:
        overflow_units = kept[max_total_units:]
        for unit in overflow_units:
            _record_drop(unit, "dropped_total_unit_budget")
        kept = kept[:max_total_units]
    diagnostics["dropped_unit_types"] = dropped_unit_types[:6]
    diagnostics["dropped_unit_predicates"] = dropped_unit_predicates[:8]
    diagnostics["post_cap_need_unit_count"] = int(len(kept))
    diagnostics["post_cap_relation_hop_count"] = int(sum(1 for unit in kept if str(unit.get("unit_type", "")) == "relation_hop"))
    return kept


def _compiled_need_unit_fallback_reasons(units: Sequence[Dict[str, Any]],
                                         diagnostics: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    relation_units = [unit for unit in units if str(unit.get("unit_type", "")) == "relation_hop"]
    selector_enabled_relation_units = [unit for unit in relation_units if bool(unit.get("selector_enabled", True))]
    if not units:
        reasons.append("empty_positive_need_units")
    if not selector_enabled_relation_units:
        reasons.append("no_selector_enabled_relation_hop")
    if not any(str(unit.get("unit_type", "")) == "answer_slot" for unit in units):
        reasons.append("missing_answer_slot")
    raw_step_count = int(diagnostics.get("raw_step_count", 0))
    malformed_step_count = int(diagnostics.get("malformed_step_count", 0))
    if raw_step_count > 0 and (float(malformed_step_count) / float(raw_step_count)) > 0.34:
        reasons.append("high_malformed_step_rate")
    if relation_units and all(str(unit.get("confidence", "")) == "low" for unit in relation_units):
        reasons.append("all_relation_hops_low_confidence")
    return reasons


def compile_qdmr_to_need_units(question: str,
                               qdmr_steps: Sequence[Dict[str, Any]] | None,
                               answer_type: str,
                               question_entities: Sequence[str] | Set[str] | None,
                               seed_entities: Sequence[str] | Set[str] | None,
                               parser_status: str = "ok",
                               max_counterfactual_sets: int = DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS,
                               max_relation_hops: int = DEFAULT_NEED_UNIT_MAX_RELATION_HOPS,
                               relation_hop_cap_mode: str = DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE) -> Dict[str, Any]:
    normalized_payload = _normalize_step_plan_payload(
        step_plan_payload={
            "question_id": stable_question_key(question),
            "answer_type": answer_type,
            "qdmr_steps": list(qdmr_steps or []),
        } if qdmr_steps is not None else None,
        question=question,
        predicted_answer_type=answer_type,
    )
    diagnostics = dict(normalized_payload["diagnostics"])
    diagnostics["parser_status"] = parser_status
    units: List[Dict[str, Any]] = []
    last_live_variable = ""
    for step in normalized_payload["qdmr_steps"]:
        operation = str(step.get("operation", ""))
        if operation == "locate_entity":
            unit = _compile_entity_locator(step, question_entities, seed_entities, diagnostics, len(units))
        elif operation == "relation_lookup":
            unit = _compile_relation_hop(step, question_entities, seed_entities, diagnostics, len(units))
        elif operation == "constraint_check":
            unit = _compile_constraint_check(step, last_live_variable, diagnostics, len(units))
        elif operation == "answer":
            unit = _compile_answer_slot(step, answer_type, last_live_variable, diagnostics, len(units))
        else:
            unit = None
        if unit is None:
            continue
        units.append(unit)
        target_variable = str(unit.get("target_variable", "") or "")
        if target_variable:
            last_live_variable = target_variable
    units = _ensure_answer_slot(units, answer_type=answer_type, diagnostics=diagnostics)
    units = _apply_need_unit_caps(
        units,
        diagnostics,
        max_relation_hops=max_relation_hops,
        relation_hop_cap_mode=relation_hop_cap_mode,
    )
    counterfactual_sets = build_counterfactual_need_unit_sets(
        question=question,
        positive_need_units=units,
        max_sets=max_counterfactual_sets,
    )
    diagnostics["positive_need_unit_count"] = int(len(units))
    diagnostics["selector_enabled_unit_count"] = int(sum(1 for unit in units if bool(unit.get("selector_enabled", True))))
    diagnostics["counterfactual_set_count"] = int(len(counterfactual_sets))
    diagnostics["compiler_fallback_reasons"] = _compiled_need_unit_fallback_reasons(units, diagnostics)
    return {
        "qdmr_steps": normalized_payload["qdmr_steps"],
        "positive_need_units": units,
        "counterfactual_sets": counterfactual_sets,
        "diagnostics": diagnostics,
        "answer_type": answer_type,
    }


def build_qdmr_steps_and_need_units(question: str,
                                    seed_entities: Sequence[str] | Set[str] | None,
                                    question_entities: Sequence[str] | Set[str] | None,
                                    max_relation_hops: int = DEFAULT_NEED_UNIT_MAX_RELATION_HOPS,
                                    relation_hop_cap_mode: str = DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload = build_heuristic_qdmr_step_plan_payload(
        question=question,
        seed_entities=seed_entities,
        question_entities=question_entities,
        predicted_answer_type=guess_answer_type_label(question),
    )
    compiled = compile_qdmr_to_need_units(
        question=question,
        qdmr_steps=payload.get("qdmr_steps", []),
        answer_type=str(payload.get("answer_type") or guess_answer_type_label(question)),
        question_entities=question_entities,
        seed_entities=seed_entities,
        parser_status="heuristic",
        max_counterfactual_sets=DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS,
        max_relation_hops=max_relation_hops,
        relation_hop_cap_mode=relation_hop_cap_mode,
    )
    return list(compiled["qdmr_steps"]), list(compiled["positive_need_units"])


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
    if not bool(unit.get("selector_enabled", True)) or str(unit.get("confidence", "")) == "low":
        return []
    constraints = list(unit.get("constraints", []))
    mutated_units: List[Dict[str, Any]] = []

    if unit_type == "relation_hop":
        role_swap_predicate = _COMMON_ROLE_PREDICATE_MAP.get(predicate)
        if role_swap_predicate:
            mutated_units.append({
                "transform": "role_swap",
                "unit": build_need_unit(
                    unit_id=f"{unit['unit_id']}_role_swap",
                    unit_type=unit_type,
                    subject=str(unit.get("subject", "")),
                    predicate=role_swap_predicate,
                    raw_predicate_text=role_swap_predicate.replace("_", " "),
                    object_value=unit.get("object"),
                    constraints=constraints,
                    target_variable=str(unit.get("target_variable", "")),
                    answer_relevance=bool(unit.get("answer_relevance", False)),
                    source_step_ids=unit.get("source_step_ids", []),
                    confidence="high",
                    selector_enabled=True,
                    debug={"counterfactual_transform": "role_swap"},
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
                    raw_predicate_text=predicate_shift.replace("_", " "),
                    object_value=unit.get("object"),
                    constraints=constraints,
                    target_variable=str(unit.get("target_variable", "")),
                    answer_relevance=bool(unit.get("answer_relevance", False)),
                    source_step_ids=unit.get("source_step_ids", []),
                    confidence="high",
                    selector_enabled=True,
                    debug={"counterfactual_transform": "predicate_shift"},
                ),
            })

    if unit_type == "constraint_check":
        for constraint in constraints:
            constraint_type = str(constraint.get("type", constraint.get("kind", ""))).strip()
            constraint_value = str(constraint.get("value", "")).strip()
            shifted_value = ""
            if constraint_type == "temporal":
                if re.fullmatch(r"in \d{4}", constraint_value):
                    shifted_value = f"in {int(constraint_value.split()[-1]) - 1}"
                elif constraint_value.startswith("before "):
                    shifted_value = constraint_value.replace("before ", "after ", 1)
                elif constraint_value.startswith("after "):
                    shifted_value = constraint_value.replace("after ", "before ", 1)
                elif constraint_value == "earlier":
                    shifted_value = "later"
                elif constraint_value == "later":
                    shifted_value = "earlier"
                elif constraint_value == "first term":
                    shifted_value = "second term"
                elif constraint_value == "second term":
                    shifted_value = "first term"
            if shifted_value:
                mutated_units.append({
                    "transform": "temporal_shift",
                    "unit": build_need_unit(
                        unit_id=f"{unit['unit_id']}_temporal_shift",
                        unit_type=unit_type,
                        subject=str(unit.get("subject", "")),
                        predicate=str(unit.get("predicate", "satisfy_constraint")),
                        raw_predicate_text=str(unit.get("raw_predicate_text", unit.get("predicate", ""))),
                        object_value=unit.get("object"),
                        constraints=[_build_constraint("temporal", shifted_value)],
                        target_variable=str(unit.get("target_variable", "")),
                        answer_relevance=bool(unit.get("answer_relevance", False)),
                        source_step_ids=unit.get("source_step_ids", []),
                        confidence="high",
                        selector_enabled=True,
                        debug={"counterfactual_transform": "temporal_shift"},
                    ),
                })
            flipped = _CONSTRAINT_FLIP_MAP.get(constraint_value)
            if flipped:
                mutated_units.append({
                    "transform": "constraint_flip",
                    "unit": build_need_unit(
                        unit_id=f"{unit['unit_id']}_constraint_flip",
                        unit_type=unit_type,
                        subject=str(unit.get("subject", "")),
                        predicate=str(unit.get("predicate", "satisfy_constraint")),
                        raw_predicate_text=str(unit.get("raw_predicate_text", unit.get("predicate", ""))),
                        object_value=unit.get("object"),
                        constraints=[_build_constraint(constraint_type or "comparative", flipped)],
                        target_variable=str(unit.get("target_variable", "")),
                        answer_relevance=bool(unit.get("answer_relevance", False)),
                        source_step_ids=unit.get("source_step_ids", []),
                        confidence="high",
                        selector_enabled=True,
                        debug={"counterfactual_transform": "constraint_flip"},
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
    return deduped[:2]


def build_counterfactual_need_unit_sets(question: str,
                                        positive_need_units: Sequence[Dict[str, Any]],
                                        max_sets: int = DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS) -> List[Dict[str, Any]]:
    del question
    counterfactual_sets: List[Dict[str, Any]] = []
    for unit in positive_need_units:
        unit_type = str(unit.get("unit_type", unit.get("type", ""))).strip()
        if unit_type == "answer_slot" or not bool(unit.get("selector_enabled", True)):
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


def _text_or_entity_match(value: str,
                          normalized_doc_title: str,
                          normalized_doc_body: str,
                          normalized_doc_entities: Set[str]) -> Dict[str, float]:
    normalized_value = normalize_structure_text(value)
    if not normalized_value or normalized_value.startswith("?"):
        return {
            "in_title": 0.0,
            "in_body": 0.0,
            "entity_match": 0.0,
        }
    return {
        "in_title": float(normalized_value in normalized_doc_title),
        "in_body": float(normalized_value in normalized_doc_body),
        "entity_match": float(normalized_value in normalized_doc_entities),
    }


def _constraint_alignment_details(constraints: Sequence[Dict[str, Any]],
                                  normalized_doc_text: str) -> tuple[float, Dict[str, float]]:
    type_scores = {
        "temporal": 0.0,
        "role": 0.0,
        "comparative": 0.0,
        "negation": 0.0,
        "scope": 0.0,
    }
    if not constraints:
        return 0.5, type_scores

    per_constraint_scores: List[float] = []
    grouped_scores: Dict[str, List[float]] = {key: [] for key in type_scores}
    for constraint in constraints:
        constraint_type = str(constraint.get("type", constraint.get("kind", ""))).strip()
        if constraint_type == "answer_type":
            continue
        value = normalize_structure_text(constraint.get("value", ""))
        score = float(bool(value and value in normalized_doc_text))
        per_constraint_scores.append(score)
        if constraint_type in grouped_scores:
            grouped_scores[constraint_type].append(score)
    for constraint_type, values in grouped_scores.items():
        type_scores[constraint_type] = float(np.mean(values)) if values else 0.0
    if not per_constraint_scores:
        return 0.5, type_scores
    return float(np.mean(per_constraint_scores)), type_scores


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


def extract_need_unit_atomic_features(unit: Dict[str, Any],
                                      doc_title: str,
                                      doc_body: str,
                                      doc_entities: Sequence[str] | Set[str] | None) -> Dict[str, float]:
    normalized_doc_title = normalize_structure_text(doc_title)
    normalized_doc_body = normalize_structure_text(doc_body)
    normalized_doc_text = normalize_structure_text(f"{doc_title} {doc_body}")
    normalized_doc_entities = set(unique_ordered_texts(doc_entities))
    title_tokens = set(tokenize_text(doc_title))
    body_tokens = set(tokenize_text(doc_body))
    doc_tokens = title_tokens | body_tokens

    subject = normalize_need_unit_slot_value(unit.get("subject", ""))
    predicate = normalize_predicate_text(unit.get("predicate", ""))
    object_value = normalize_need_unit_slot_value(unit.get("object", ""))
    unit_type = str(unit.get("unit_type", unit.get("type", ""))).strip()
    has_constraints = float(bool(unit.get("constraints")))
    subject_is_variable = float(bool(subject and subject.startswith("?")))
    object_is_variable = float(bool(object_value and object_value.startswith("?")))

    subject_tokens = set(tokenize_text(subject))
    predicate_text = predicate.replace("_", " ").strip()
    predicate_tokens = set(tokenize_text(predicate_text))
    object_tokens = set(tokenize_text(object_value)) if object_value and not object_value.startswith("?") else set()

    subject_match = _text_or_entity_match(subject, normalized_doc_title, normalized_doc_body, normalized_doc_entities)
    object_match = _text_or_entity_match(object_value, normalized_doc_title, normalized_doc_body, normalized_doc_entities)
    predicate_in_title = float(bool(predicate_text and predicate_text in normalized_doc_title))
    predicate_in_body = float(bool(predicate_text and predicate_text in normalized_doc_body))
    predicate_token_overlap = (
        max(
            _token_overlap_ratio(predicate_tokens, title_tokens),
            _token_overlap_ratio(predicate_tokens, body_tokens),
        )
        if predicate_tokens else 0.0
    )
    opposing_predicate_in_text = float(_opposing_predicate_score(predicate, normalized_doc_text))
    constraint_alignment, constraint_type_scores = _constraint_alignment_details(unit.get("constraints", []), normalized_doc_text)

    subject_alignment = 0.5 if subject_is_variable or not subject else max(
        subject_match["in_title"],
        subject_match["in_body"],
        subject_match["entity_match"],
        _token_overlap_ratio(subject_tokens, doc_tokens),
    )
    object_alignment = 0.5 if object_is_variable or not object_tokens else max(
        object_match["in_title"],
        object_match["in_body"],
        object_match["entity_match"],
        _token_overlap_ratio(object_tokens, doc_tokens),
    )
    predicate_alignment = max(
        predicate_in_title,
        predicate_in_body,
        predicate_token_overlap,
    )

    title_bridge_alignment = max(
        float(subject_match["in_title"] > 0.0 and (predicate_in_body > 0.0 or object_match["in_body"] > 0.0 or constraint_alignment > 0.5)),
        float(object_match["in_title"] > 0.0 and (subject_match["in_body"] > 0.0 or predicate_in_body > 0.0)),
        float(predicate_in_title > 0.0 and (subject_match["in_body"] > 0.0 or object_match["in_body"] > 0.0)),
    )
    entity_bridge_alignment = max(
        float(subject_match["entity_match"] > 0.0 and (predicate_in_body > 0.0 or object_match["in_body"] > 0.0 or object_match["entity_match"] > 0.0)),
        float(object_match["entity_match"] > 0.0 and (subject_match["in_body"] > 0.0 or predicate_in_body > 0.0 or subject_match["entity_match"] > 0.0)),
    )
    alias_or_variable_bridge_alignment = max(
        entity_bridge_alignment,
        float(subject_is_variable > 0.0 and (predicate_in_body > 0.0 or object_match["in_body"] > 0.0 or object_match["entity_match"] > 0.0)),
        float(object_is_variable > 0.0 and (subject_match["in_title"] > 0.0 or subject_match["in_body"] > 0.0 or subject_match["entity_match"] > 0.0)),
    )
    bridge_feature_count = float(sum(
        1
        for value in (
            title_bridge_alignment,
            entity_bridge_alignment,
            alias_or_variable_bridge_alignment,
        )
        if value >= 0.5
    ))

    return {
        "is_entity_locator": float(unit_type == "entity_locator"),
        "is_relation_hop": float(unit_type == "relation_hop"),
        "is_constraint_check": float(unit_type == "constraint_check"),
        "is_answer_slot": float(unit_type == "answer_slot"),
        "answer_relevance": float(bool(unit.get("answer_relevance", False))),
        "has_constraints": has_constraints,
        "constraint_count": float(len(list(unit.get("constraints", [])))),
        "subject_is_variable": subject_is_variable,
        "object_is_variable": object_is_variable,
        "subject_in_title": float(subject_match["in_title"]),
        "subject_in_body": float(subject_match["in_body"]),
        "subject_token_overlap": float(_token_overlap_ratio(subject_tokens, doc_tokens)) if subject_tokens else 0.0,
        "subject_entity_match": float(subject_match["entity_match"]),
        "object_in_title": float(object_match["in_title"]),
        "object_in_body": float(object_match["in_body"]),
        "object_token_overlap": float(_token_overlap_ratio(object_tokens, doc_tokens)) if object_tokens else 0.0,
        "object_entity_match": float(object_match["entity_match"]),
        "predicate_in_title": float(predicate_in_title),
        "predicate_in_body": float(predicate_in_body),
        "predicate_token_overlap": float(predicate_token_overlap),
        "opposing_predicate_in_text": float(opposing_predicate_in_text),
        "constraint_alignment": float(constraint_alignment),
        "constraint_temporal_score": float(constraint_type_scores["temporal"]),
        "constraint_role_score": float(constraint_type_scores["role"]),
        "constraint_comparative_score": float(constraint_type_scores["comparative"]),
        "constraint_negation_score": float(constraint_type_scores["negation"]),
        "constraint_scope_score": float(constraint_type_scores["scope"]),
        "title_bridge_alignment": float(title_bridge_alignment),
        "entity_bridge_alignment": float(entity_bridge_alignment),
        "alias_or_variable_bridge_alignment": float(alias_or_variable_bridge_alignment),
        "bridge_feature_count": float(bridge_feature_count),
        "subject_alignment": float(subject_alignment),
        "predicate_alignment": float(predicate_alignment),
        "object_alignment": float(object_alignment),
    }


def compute_need_unit_alignment_prior(feature_row: Dict[str, float | int]) -> float:
    return float(np.clip(
        0.30 * float(feature_row.get("subject_alignment", 0.0))
        + 0.25 * float(feature_row.get("predicate_alignment", 0.0))
        + 0.10 * float(feature_row.get("object_alignment", 0.0))
        + 0.15 * float(feature_row.get("constraint_alignment", 0.0))
        + 0.10 * float(feature_row.get("title_bridge_alignment", 0.0))
        + 0.10 * float(feature_row.get("alias_or_variable_bridge_alignment", 0.0)),
        0.0,
        1.0,
    ))


def _normalize_atomic_probability_map(probability_map: Dict[str, float]) -> Dict[str, float]:
    cleaned = {
        label: max(float(probability_map.get(label, 0.0) or 0.0), 0.0)
        for label in NEED_UNIT_ATOMIC_LABELS
    }
    total = float(sum(cleaned.values()))
    if total <= 0.0:
        return {
            "full_support": 0.0,
            "bridge_support": 0.0,
            "contradiction": 0.0,
            "nei": 1.0,
        }
    return {
        label: float(cleaned[label] / total)
        for label in NEED_UNIT_ATOMIC_LABELS
    }


def _predict_need_unit_atomic_probs_heuristic(unit: Dict[str, Any],
                                              feature_row: Dict[str, float | int],
                                              alignment_score: float) -> Dict[str, float]:
    unit_type = str(unit.get("unit_type", unit.get("type", ""))).strip()
    subject_alignment = float(feature_row.get("subject_alignment", 0.0))
    predicate_alignment = float(feature_row.get("predicate_alignment", 0.0))
    object_alignment = float(feature_row.get("object_alignment", 0.0))
    constraint_alignment = float(feature_row.get("constraint_alignment", 0.0))
    title_bridge_alignment = float(feature_row.get("title_bridge_alignment", 0.0))
    entity_bridge_alignment = float(feature_row.get("entity_bridge_alignment", 0.0))
    alias_or_variable_bridge_alignment = float(feature_row.get("alias_or_variable_bridge_alignment", 0.0))
    contradiction_prior = float(feature_row.get("opposing_predicate_in_text", 0.0))
    has_constraints = float(feature_row.get("has_constraints", 0.0))

    full_support_raw = np.clip(
        0.42 * predicate_alignment
        + 0.22 * subject_alignment
        + 0.12 * object_alignment
        + 0.14 * constraint_alignment
        + 0.10 * title_bridge_alignment
        - 0.35 * contradiction_prior,
        0.0,
        1.0,
    )
    if unit_type == "answer_slot":
        full_support_raw = max(
            full_support_raw,
            np.clip(
                0.55 * subject_alignment
                + 0.25 * object_alignment
                + 0.20 * constraint_alignment,
                0.0,
                1.0,
            ),
        )

    bridge_support_raw = np.clip(
        0.30 * title_bridge_alignment
        + 0.25 * entity_bridge_alignment
        + 0.25 * alias_or_variable_bridge_alignment
        + 0.10 * subject_alignment
        + 0.10 * predicate_alignment
        - 0.20 * max(predicate_alignment, object_alignment)
        - 0.15 * max(0.0, full_support_raw - 0.4)
        - 0.20 * contradiction_prior,
        0.0,
        1.0,
    )
    if unit_type == "relation_hop":
        bridge_support_raw = min(
            1.0,
            bridge_support_raw + 0.05 * float(feature_row.get("bridge_feature_count", 0.0) > 0.0),
        )
    if unit_type == "answer_slot":
        bridge_support_raw *= 0.6

    contradiction_raw = np.clip(
        0.70 * contradiction_prior
        + 0.30 * max(0.0, 1.0 - constraint_alignment) * has_constraints * max(subject_alignment, predicate_alignment),
        0.0,
        1.0,
    )
    nei_raw = np.clip(
        1.0 - max(full_support_raw, bridge_support_raw, contradiction_raw) + 0.20 * (1.0 - alignment_score),
        0.05,
        1.0,
    )
    return _normalize_atomic_probability_map({
        "full_support": float(full_support_raw),
        "bridge_support": float(bridge_support_raw),
        "contradiction": float(contradiction_raw),
        "nei": float(nei_raw),
    })


def _predict_need_unit_atomic_probs_model(feature_row: Dict[str, float | int],
                                          atomic_scorer_bundle: Dict[str, Any]) -> Dict[str, float]:
    model = atomic_scorer_bundle.get("model")
    if model is None:
        raise ValueError("Atomic scorer bundle is missing a model")
    feature_names = list(atomic_scorer_bundle.get("feature_names") or NEED_UNIT_ATOMIC_FEATURE_NAMES)
    feature_matrix = requirement_feature_rows_to_matrix([feature_row], feature_names=feature_names)
    probability_matrix = model.predict_proba(feature_matrix)
    if probability_matrix.ndim != 2 or probability_matrix.shape[0] != 1:
        raise ValueError("Atomic scorer predict_proba returned an unexpected shape")
    class_names = list(getattr(model, "classes_", atomic_scorer_bundle.get("labels") or []))
    probability_map = {
        str(class_name): float(probability_matrix[0][class_idx])
        for class_idx, class_name in enumerate(class_names)
    }
    return _normalize_atomic_probability_map(probability_map)


def predict_need_unit_atomic_probs(unit: Dict[str, Any],
                                   feature_row: Dict[str, float | int],
                                   alignment_score: float,
                                   atomic_scorer_bundle: Dict[str, Any] | None = None,
                                   score_mode: str = "heuristic") -> Dict[str, float]:
    normalized_score_mode = str(score_mode or "heuristic").strip().lower()
    if normalized_score_mode not in ATOMIC_ANNOTATION_SCORE_MODES:
        raise ValueError(f"Unsupported atomic score_mode: {score_mode}")
    if normalized_score_mode == "hybrid":
        if atomic_scorer_bundle is None:
            raise ValueError("Hybrid atomic scoring requires an atomic_scorer_bundle")
        return _predict_need_unit_atomic_probs_model(
            feature_row=feature_row,
            atomic_scorer_bundle=atomic_scorer_bundle,
        )
    return _predict_need_unit_atomic_probs_heuristic(
        unit=unit,
        feature_row=feature_row,
        alignment_score=alignment_score,
    )


def score_need_unit_support(unit: Dict[str, Any],
                            doc_title: str,
                            doc_body: str,
                            doc_entities: Sequence[str] | Set[str] | None,
                            atomic_scorer_bundle: Dict[str, Any] | None = None,
                            score_mode: str = "heuristic") -> Dict[str, float | str]:
    feature_row = extract_need_unit_atomic_features(
        unit=unit,
        doc_title=doc_title,
        doc_body=doc_body,
        doc_entities=doc_entities,
    )
    alignment_score = compute_need_unit_alignment_prior(feature_row)
    probability_map = predict_need_unit_atomic_probs(
        unit=unit,
        feature_row=feature_row,
        alignment_score=alignment_score,
        atomic_scorer_bundle=atomic_scorer_bundle,
        score_mode=score_mode,
    )
    bridge_alpha_by_type = dict(
        atomic_scorer_bundle.get("bridge_alpha_by_unit_type", {})
        if atomic_scorer_bundle else {}
    )
    resolved_bridge_alpha = float(
        bridge_alpha_by_type.get(
            str(unit.get("unit_type", unit.get("type", ""))).strip(),
            DEFAULT_NEED_UNIT_BRIDGE_ALPHA_BY_TYPE.get(
                str(unit.get("unit_type", unit.get("type", ""))).strip(),
                0.5,
            ),
        )
    )
    contradiction_beta = float(
        atomic_scorer_bundle.get("beta_contradiction", DEFAULT_NEED_UNIT_CONTRADICTION_BETA)
        if atomic_scorer_bundle else DEFAULT_NEED_UNIT_CONTRADICTION_BETA
    )
    support_prob = float(np.clip(
        probability_map["full_support"] + resolved_bridge_alpha * probability_map["bridge_support"],
        0.0,
        1.0,
    ))
    contradiction_prob = float(np.clip(probability_map["contradiction"], 0.0, 1.0))
    nei_prob = float(np.clip(probability_map["nei"], 0.0, 1.0))
    coverage_score = float(np.clip(
        alignment_score * max(0.0, support_prob - contradiction_beta * contradiction_prob),
        0.0,
        1.0,
    ))
    return {
        "alignment_score": round(alignment_score, 6),
        "full_support_prob": round(float(probability_map["full_support"]), 6),
        "bridge_support_prob": round(float(probability_map["bridge_support"]), 6),
        "support_prob": round(support_prob, 6),
        "contradiction_prob": round(contradiction_prob, 6),
        "nei_prob": round(nei_prob, 6),
        "coverage_score": round(coverage_score, 6),
        "score_source": "model" if str(score_mode or "").strip().lower() == "hybrid" else "heuristic_fallback",
        "scorer_version": str(
            atomic_scorer_bundle.get("scorer_version", NEED_UNIT_ATOMIC_SCORER_VERSION)
            if atomic_scorer_bundle else NEED_UNIT_ATOMIC_SCORER_VERSION
        ),
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
                                   counterfactual_sets: Sequence[Dict[str, Any]],
                                   atomic_scorer_bundle: Dict[str, Any] | None = None,
                                   score_mode: str = "heuristic") -> Dict[str, Any]:
    doc_title, doc_body = (str(doc_text).split("\n", 1) + [""])[:2]
    normalized_doc_entities = unique_ordered_texts(doc_entities)

    positive_scores: Dict[str, float] = {}
    positive_alignments: Dict[str, float] = {}
    positive_atomic_scores: Dict[str, Dict[str, Any]] = {}
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
            atomic_scorer_bundle=atomic_scorer_bundle,
            score_mode=score_mode,
        )
        positive_scores[unit_id] = float(scores["coverage_score"])
        positive_alignments[unit_id] = float(scores["alignment_score"])
        positive_support_probs[unit_id] = float(scores["support_prob"])
        positive_contradiction_probs[unit_id] = float(scores["contradiction_prob"])
        positive_nei_probs[unit_id] = float(scores["nei_prob"])
        positive_atomic_scores[unit_id] = {
            "alignment_score": float(scores["alignment_score"]),
            "full_support_prob": float(scores["full_support_prob"]),
            "bridge_support_prob": float(scores["bridge_support_prob"]),
            "contradiction_prob": float(scores["contradiction_prob"]),
            "nei_prob": float(scores["nei_prob"]),
            "support_prob": float(scores["support_prob"]),
            "coverage_score": float(scores["coverage_score"]),
            "score_source": str(scores["score_source"]),
            "scorer_version": str(scores["scorer_version"]),
        }

    counterfactual_scores: Dict[str, Dict[str, float]] = {}
    counterfactual_alignments: Dict[str, Dict[str, float]] = {}
    counterfactual_atomic_scores: Dict[str, Dict[str, Dict[str, Any]]] = {}
    counterfactual_support_probs: Dict[str, Dict[str, float]] = {}
    counterfactual_contradiction_probs: Dict[str, Dict[str, float]] = {}
    counterfactual_nei_probs: Dict[str, Dict[str, float]] = {}
    counterfactual_set_scores: Dict[str, float] = {}
    for cf_set in counterfactual_sets:
        cf_id = str(cf_set.get("cf_id", ""))
        need_units = list(cf_set.get("requirements", cf_set.get("need_units", [])))
        cf_cov_scores: Dict[str, float] = {}
        cf_align_scores: Dict[str, float] = {}
        cf_atomic_scores: Dict[str, Dict[str, Any]] = {}
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
                atomic_scorer_bundle=atomic_scorer_bundle,
                score_mode=score_mode,
            )
            cf_cov_scores[unit_id] = float(scores["coverage_score"])
            cf_align_scores[unit_id] = float(scores["alignment_score"])
            cf_support_scores[unit_id] = float(scores["support_prob"])
            cf_contradiction_scores[unit_id] = float(scores["contradiction_prob"])
            cf_nei_scores[unit_id] = float(scores["nei_prob"])
            cf_atomic_scores[unit_id] = {
                "alignment_score": float(scores["alignment_score"]),
                "full_support_prob": float(scores["full_support_prob"]),
                "bridge_support_prob": float(scores["bridge_support_prob"]),
                "contradiction_prob": float(scores["contradiction_prob"]),
                "nei_prob": float(scores["nei_prob"]),
                "support_prob": float(scores["support_prob"]),
                "coverage_score": float(scores["coverage_score"]),
                "score_source": str(scores["score_source"]),
                "scorer_version": str(scores["scorer_version"]),
            }
        counterfactual_scores[cf_id] = cf_cov_scores
        counterfactual_alignments[cf_id] = cf_align_scores
        counterfactual_atomic_scores[cf_id] = cf_atomic_scores
        counterfactual_support_probs[cf_id] = cf_support_scores
        counterfactual_contradiction_probs[cf_id] = cf_contradiction_scores
        counterfactual_nei_probs[cf_id] = cf_nei_scores
        counterfactual_set_scores[cf_id] = round(
            float(np.mean(list(cf_cov_scores.values()))) if cf_cov_scores else 0.0,
            6,
        )

    bridge_alpha_by_type = dict(DEFAULT_NEED_UNIT_BRIDGE_ALPHA_BY_TYPE)
    if atomic_scorer_bundle:
        bridge_alpha_by_type.update(dict(atomic_scorer_bundle.get("bridge_alpha_by_unit_type", {}) or {}))
    annotation_metadata = {
        "score_mode": str(score_mode or "heuristic"),
        "scorer_version": str(
            atomic_scorer_bundle.get("scorer_version", NEED_UNIT_ATOMIC_SCORER_VERSION)
            if atomic_scorer_bundle else NEED_UNIT_ATOMIC_SCORER_VERSION
        ),
        "model_path": str(atomic_scorer_bundle.get("model_path", "")) if atomic_scorer_bundle else "",
        "bridge_alpha_by_unit_type": bridge_alpha_by_type,
        "beta_contradiction": float(
            atomic_scorer_bundle.get("beta_contradiction", DEFAULT_NEED_UNIT_CONTRADICTION_BETA)
            if atomic_scorer_bundle else DEFAULT_NEED_UNIT_CONTRADICTION_BETA
        ),
    }
    return {
        "pool_position": int(pool_position),
        "doc_title": doc_title,
        "doc_entities": normalized_doc_entities,
        "positive_atomic_scores": positive_atomic_scores,
        "positive_need_unit_scores": positive_scores,
        "positive_requirement_scores": positive_scores,
        "positive_alignment_scores": positive_alignments,
        "positive_support_probs": positive_support_probs,
        "positive_contradiction_scores": positive_contradiction_probs,
        "positive_nei_probs": positive_nei_probs,
        "counterfactual_atomic_scores": counterfactual_atomic_scores,
        "counterfactual_requirement_scores": counterfactual_scores,
        "counterfactual_alignment_scores": counterfactual_alignments,
        "counterfactual_support_probs": counterfactual_support_probs,
        "counterfactual_contradiction_scores": counterfactual_contradiction_probs,
        "counterfactual_nei_probs": counterfactual_nei_probs,
        "counterfactual_set_scores": counterfactual_set_scores,
        "annotation_metadata": annotation_metadata,
    }


def build_cache_doc_annotation(pool_position: int,
                               doc_text: str,
                               doc_entities: Sequence[str] | Set[str] | None,
                               cache_entry: Dict[str, Any],
                               atomic_scorer_bundle: Dict[str, Any] | None = None,
                               score_mode: str = "heuristic") -> Dict[str, Any]:
    if is_need_unit_cache_version(cache_entry) or "positive_need_units" in cache_entry:
        return build_need_unit_doc_annotation(
            pool_position=pool_position,
            doc_text=doc_text,
            doc_entities=doc_entities,
            positive_need_units=get_positive_units(cache_entry),
            counterfactual_sets=get_counterfactual_sets(cache_entry),
            atomic_scorer_bundle=atomic_scorer_bundle,
            score_mode=score_mode,
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
                                max_counterfactual_sets: int = DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS,
                                max_relation_hops: int = DEFAULT_NEED_UNIT_MAX_RELATION_HOPS,
                                relation_hop_cap_mode: str = DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE,
                                atomic_scorer_bundle: Dict[str, Any] | None = None,
                                score_mode: str = "heuristic",
                                predicted_answer_type: str | None = None,
                                step_plan_payload: Dict[str, Any] | None = None,
                                parser_trace: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resolved_answer_type = normalize_answer_type_label(predicted_answer_type, question=question)
    active_step_plan_payload = step_plan_payload
    parser_status = "ok" if step_plan_payload else "heuristic"
    if active_step_plan_payload is None:
        active_step_plan_payload = build_heuristic_qdmr_step_plan_payload(
            question=question,
            seed_entities=seed_entities,
            question_entities=question_entities,
            predicted_answer_type=resolved_answer_type,
        )
        parser_status = "heuristic"
    normalized_payload = _normalize_step_plan_payload(
        step_plan_payload=active_step_plan_payload,
        question=question,
        predicted_answer_type=resolved_answer_type,
    )
    compiled = compile_qdmr_to_need_units(
        question=question,
        qdmr_steps=normalized_payload.get("qdmr_steps", []),
        answer_type=str(normalized_payload.get("answer_type") or resolved_answer_type),
        question_entities=question_entities,
        seed_entities=seed_entities,
        parser_status=parser_status,
        max_counterfactual_sets=max_counterfactual_sets,
        max_relation_hops=max_relation_hops,
        relation_hop_cap_mode=relation_hop_cap_mode,
    )
    diagnostics = _attach_parser_trace_to_diagnostics(
        diagnostics=dict(compiled.get("diagnostics", {}) or {}),
        parser_trace=parser_trace,
    )
    fallback_reasons = list(diagnostics.get("compiler_fallback_reasons", []) or [])
    if fallback_reasons and parser_status != "heuristic":
        pre_fallback_diagnostics = dict(diagnostics)
        fallback_payload = build_heuristic_qdmr_step_plan_payload(
            question=question,
            seed_entities=seed_entities,
            question_entities=question_entities,
            predicted_answer_type=resolved_answer_type,
        )
        fallback_compiled = compile_qdmr_to_need_units(
            question=question,
            qdmr_steps=fallback_payload.get("qdmr_steps", []),
            answer_type=str(fallback_payload.get("answer_type") or resolved_answer_type),
            question_entities=question_entities,
            seed_entities=seed_entities,
            parser_status="fallback",
            max_counterfactual_sets=max_counterfactual_sets,
            max_relation_hops=max_relation_hops,
            relation_hop_cap_mode=relation_hop_cap_mode,
        )
        compiled = fallback_compiled
        diagnostics = _attach_parser_trace_to_diagnostics(
            diagnostics=dict(fallback_compiled.get("diagnostics", {}) or {}),
            parser_trace=parser_trace,
        )
        diagnostics["parser_status"] = "fallback"
        diagnostics["compiler_status"] = "fallback"
        diagnostics["fallback_reason"] = ",".join(fallback_reasons)
        diagnostics["pre_fallback_compiler_fallback_reasons"] = list(fallback_reasons)
        diagnostics["pre_fallback_positive_need_unit_count"] = int(pre_fallback_diagnostics.get("positive_need_unit_count", 0))
        diagnostics["pre_fallback_selector_enabled_unit_count"] = int(pre_fallback_diagnostics.get("selector_enabled_unit_count", 0))
        diagnostics["pre_fallback_notes"] = list(pre_fallback_diagnostics.get("notes", []) or [])
        diagnostics["fallback_compiler_fallback_reasons"] = list(diagnostics.get("compiler_fallback_reasons", []) or [])
    effective_annotation_pool_k = min(len(pool_docs), max(int(annotation_pool_k), 0))
    cache_entry: Dict[str, Any] = {
        "version": NEED_UNIT_CACHE_VERSION,
        "query_index": int(query_index),
        "question": question,
        "question_key": stable_question_key(question),
        "predicted_answer_type": resolved_answer_type,
        "max_relation_hops": int(max_relation_hops),
        "relation_hop_cap_mode": str(relation_hop_cap_mode or DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE),
        "annotation_pool_k": int(effective_annotation_pool_k),
        "seed_entities": list(unique_ordered_texts(seed_entities)),
        "question_entities": list(unique_ordered_texts(question_entities)),
        "qdmr_steps": list(compiled.get("qdmr_steps", [])),
        "positive_need_units": list(compiled.get("positive_need_units", [])),
        "counterfactual_sets": list(compiled.get("counterfactual_sets", [])),
        "pool_titles": [
            str(doc_text).split("\n", 1)[0].strip()
            for doc_text in pool_docs[:effective_annotation_pool_k]
        ],
        "doc_annotations": [],
        "diagnostics": dict(diagnostics),
    }
    cache_entry["diagnostics"]["nonempty_counterfactual_rate"] = round(
        float(len(cache_entry["counterfactual_sets"]) > 0),
        4,
    )
    cache_entry["diagnostics"]["qdmr_step_count"] = int(len(cache_entry["qdmr_steps"]))
    cache_entry["diagnostics"]["annotation_score_mode"] = str(score_mode or "heuristic")
    cache_entry["diagnostics"]["annotation_model_path"] = (
        str(atomic_scorer_bundle.get("model_path", "")) if atomic_scorer_bundle else ""
    )
    for pool_position in range(effective_annotation_pool_k):
        cache_entry["doc_annotations"].append(build_cache_doc_annotation(
            pool_position=pool_position,
            doc_text=str(pool_docs[pool_position]),
            doc_entities=pool_doc_entities[pool_position] if pool_position < len(pool_doc_entities) else [],
            cache_entry=cache_entry,
            atomic_scorer_bundle=atomic_scorer_bundle,
            score_mode=score_mode,
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
                                          pool_doc_entities: Sequence[Sequence[str] | Set[str]] | None = None,
                                          atomic_scorer_bundle: Dict[str, Any] | None = None,
                                          score_mode: str = "heuristic") -> Dict[str, Any]:
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
                atomic_scorer_bundle=atomic_scorer_bundle,
                score_mode=str(score_mode),
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
    diagnostics["title_alignment_score_mode"] = str(score_mode or "heuristic")
    if atomic_scorer_bundle is not None:
        diagnostics["title_alignment_atomic_scorer_version"] = str(
            atomic_scorer_bundle.get("scorer_version", "")
        )
        diagnostics["title_alignment_atomic_model_path"] = str(
            atomic_scorer_bundle.get("model_path", "")
        )
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


def normalize_requirement_bridge_bonus_mode(bridge_bonus_mode: str | None) -> str:
    normalized = str(bridge_bonus_mode or "off").strip().lower()
    if normalized not in REQUIREMENT_BRIDGE_BONUS_MODES:
        raise ValueError(f"Unsupported requirement bridge bonus mode: {bridge_bonus_mode}")
    return normalized


def get_positive_score_map_with_overrides(annotation: Dict[str, Any],
                                          pool_position: int,
                                          positive_score_overrides_by_position: Dict[int, Dict[str, float]] | None = None) -> Dict[str, float]:
    score_map = get_positive_score_map(annotation)
    override_map = dict(
        (positive_score_overrides_by_position or {}).get(int(pool_position), {}) or {}
    )
    if not override_map:
        return score_map
    merged = dict(score_map)
    for requirement_id, score in override_map.items():
        merged[str(requirement_id)] = float(score)
    return merged


def merge_positive_score_overrides_by_position(
    positive_score_overrides_by_position: Dict[int, Dict[str, float]] | None,
    pool_position: int,
    override_map: Dict[str, float] | None,
) -> Dict[int, Dict[str, float]]:
    merged: Dict[int, Dict[str, float]] = {
        int(raw_position): {
            str(requirement_id): float(score)
            for requirement_id, score in dict(position_override or {}).items()
        }
        for raw_position, position_override in dict(positive_score_overrides_by_position or {}).items()
        if dict(position_override or {})
    }
    resolved_override_map = {
        str(requirement_id): float(score)
        for requirement_id, score in dict(override_map or {}).items()
    }
    if not resolved_override_map:
        return merged
    candidate_position = int(pool_position)
    merged_candidate = dict(merged.get(candidate_position, {}))
    merged_candidate.update(resolved_override_map)
    merged[candidate_position] = merged_candidate
    return merged


def _ordered_unique_positions(selected_positions: Sequence[int] | None) -> List[int]:
    ordered: List[int] = []
    seen: Set[int] = set()
    for raw_pos in selected_positions or []:
        pos = int(raw_pos)
        if pos < 0 or pos in seen:
            continue
        seen.add(pos)
        ordered.append(pos)
    return ordered


def _get_active_positive_units(cache_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        requirement
        for requirement in get_positive_units(cache_entry)
        if (not is_need_unit_cache_version(cache_entry)) or bool(requirement.get("selector_enabled", True))
    ]


def _compute_positive_unit_coverages(cache_entry: Dict[str, Any],
                                     selected_positions: Sequence[int],
                                     annotations_by_position: Dict[int, Dict[str, Any]],
                                     positive_score_overrides_by_position: Dict[int, Dict[str, float]] | None = None) -> Dict[str, float]:
    positive_coverages: Dict[str, float] = {}
    ordered_positions = _ordered_unique_positions(selected_positions)
    for requirement in _get_active_positive_units(cache_entry):
        requirement_id = str(requirement.get("unit_id", requirement.get("requirement_id", "")))
        if not requirement_id:
            continue
        scores = [
            float(
                get_positive_score_map_with_overrides(
                    annotations_by_position[pos],
                    pos,
                    positive_score_overrides_by_position=positive_score_overrides_by_position,
                ).get(requirement_id, 0.0)
            )
            for pos in ordered_positions
            if pos in annotations_by_position
        ]
        positive_coverages[requirement_id] = _requirement_coverage_for_positions(scores)
    return positive_coverages


def _resolve_positive_unit_binding_predecessors(active_positive_units: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    predecessor_by_unit_id: Dict[str, str] = {}
    producer_by_variable: Dict[str, str] = {}
    for requirement in active_positive_units:
        requirement_id = str(requirement.get("unit_id", requirement.get("requirement_id", "")))
        if not requirement_id:
            continue
        subject_variable = canonicalize_unit_variable_reference(requirement.get("subject", ""))
        predecessor_by_unit_id[requirement_id] = producer_by_variable.get(subject_variable, "") if subject_variable else ""
        target_variable = canonicalize_unit_variable_reference(requirement.get("target_variable", ""))
        if target_variable:
            producer_by_variable[target_variable] = requirement_id
    return predecessor_by_unit_id


def _resolve_requirement_candidate_bridge_bonus(annotation: Dict[str, Any],
                                                cache_entry: Dict[str, Any],
                                                current_positive_coverages: Dict[str, float],
                                                bridge_bonus_binding_entities: Sequence[str] | Set[str] | None,
                                                bridge_bonus_mode: str = "off",
                                                bridge_bonus_weight: float = 0.0,
                                                bridge_bonus_alignment_threshold: float = 0.25,
                                                bridge_bonus_activation_threshold: float = 0.25,
                                                bridge_bonus_current_coverage_cap: float = 0.70) -> tuple[Dict[str, float], Dict[str, Any]]:
    normalized_mode = normalize_requirement_bridge_bonus_mode(bridge_bonus_mode)
    metadata: Dict[str, Any] = {
        "applied": False,
        "mode": normalized_mode,
        "unit_id": "",
        "unit_type": "",
        "unit_predicate": "",
        "score": 0.0,
        "base_score": 0.0,
        "alignment_score": 0.0,
        "predecessor_id": "",
        "predecessor_coverage": 0.0,
        "overlap_entity_count": 0,
        "novel_entity_count": 0,
    }
    if (
        normalized_mode == "off"
        or float(bridge_bonus_weight) <= 0.0
        or not is_need_unit_cache_version(cache_entry)
    ):
        return {}, metadata

    doc_entities = set(unique_ordered_texts(annotation.get("doc_entities", [])))
    binding_entities = set(unique_ordered_texts(bridge_bonus_binding_entities))
    overlap_entities = doc_entities & binding_entities
    novel_entities = doc_entities - binding_entities
    metadata["overlap_entity_count"] = int(len(overlap_entities))
    metadata["novel_entity_count"] = int(len(novel_entities))
    if not overlap_entities or not novel_entities:
        return {}, metadata

    active_positive_units = _get_active_positive_units(cache_entry)
    predecessor_by_unit_id = _resolve_positive_unit_binding_predecessors(active_positive_units)
    base_score_map = get_positive_score_map(annotation)
    alignment_score_map = dict(annotation.get("positive_alignment_scores", {}) or {})

    override_map: Dict[str, float] = {}
    best_key: tuple[float, float, float, str] | None = None
    for requirement in active_positive_units:
        requirement_id = str(requirement.get("unit_id", requirement.get("requirement_id", "")))
        if not requirement_id:
            continue
        requirement_type = str(requirement.get("unit_type", requirement.get("type", "")) or "")
        if requirement_type != "relation_hop":
            continue
        subject_variable = canonicalize_unit_variable_reference(requirement.get("subject", ""))
        if not subject_variable:
            continue
        predecessor_id = str(predecessor_by_unit_id.get(requirement_id, "") or "")
        if not predecessor_id:
            continue
        predecessor_coverage = float(current_positive_coverages.get(predecessor_id, 0.0) or 0.0)
        current_coverage = float(current_positive_coverages.get(requirement_id, 0.0) or 0.0)
        if predecessor_coverage < float(bridge_bonus_activation_threshold):
            continue
        if current_coverage >= float(bridge_bonus_current_coverage_cap):
            continue
        alignment_score = float(alignment_score_map.get(requirement_id, 0.0) or 0.0)
        if alignment_score < float(bridge_bonus_alignment_threshold):
            continue
        base_score = float(base_score_map.get(requirement_id, 0.0) or 0.0)
        activation_scale = min(
            1.0,
            predecessor_coverage / max(float(bridge_bonus_activation_threshold), 1e-6),
        )
        candidate_score = float(np.clip(
            max(base_score, float(bridge_bonus_weight) * alignment_score * activation_scale),
            0.0,
            1.0,
        ))
        if candidate_score <= base_score + 1e-9:
            continue
        override_map[requirement_id] = candidate_score
        candidate_key = (
            float(candidate_score - base_score),
            float(alignment_score),
            float(predecessor_coverage),
            requirement_id,
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            metadata.update({
                "applied": True,
                "unit_id": requirement_id,
                "unit_type": requirement_type,
                "unit_predicate": str(requirement.get("predicate", "") or ""),
                "score": float(candidate_score),
                "base_score": float(base_score),
                "alignment_score": float(alignment_score),
                "predecessor_id": predecessor_id,
                "predecessor_coverage": float(predecessor_coverage),
            })

    return override_map, metadata


def compute_requirement_state_metrics(cache_entry: Dict[str, Any],
                                      selected_positions: Sequence[int],
                                      positive_score_overrides_by_position: Dict[int, Dict[str, float]] | None = None,
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
        if is_need_unit_cache_version(cache_entry) and not bool(requirement.get("selector_enabled", True)):
            continue
        requirement_id = str(requirement.get("unit_id", requirement.get("requirement_id", "")))
        scores = [
            float(
                get_positive_score_map_with_overrides(
                    annotations_by_position[pos],
                    pos,
                    positive_score_overrides_by_position=positive_score_overrides_by_position,
                ).get(requirement_id, 0.0)
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
            if is_need_unit_cache_version(cache_entry) and not bool(requirement.get("selector_enabled", True)):
                continue
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
        if not requirement_coverages:
            continue
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
                                               positive_score_overrides_by_position: Dict[int, Dict[str, float]] | None = None,
                                               bridge_bonus_binding_entities: Sequence[str] | Set[str] | None = None,
                                               bridge_bonus_mode: str = "off",
                                               bridge_bonus_weight: float = 0.0,
                                               smooth_tau: float = DEFAULT_REQUIREMENT_SMOOTH_TAU,
                                               counterfactual_tau: float = DEFAULT_REQUIREMENT_CF_TAU) -> List[Dict[str, Any]]:
    current_metrics = compute_requirement_state_metrics(
        cache_entry=cache_entry,
        selected_positions=selected_positions,
        positive_score_overrides_by_position=positive_score_overrides_by_position,
        smooth_tau=smooth_tau,
        counterfactual_tau=counterfactual_tau,
    )
    annotations_by_position = {
        int(annotation["pool_position"]): annotation
        for annotation in cache_entry.get("doc_annotations", [])
    }
    active_positive_ids = {
        str(requirement.get("unit_id", requirement.get("requirement_id", "")))
        for requirement in get_positive_units(cache_entry)
        if not is_need_unit_cache_version(cache_entry) or bool(requirement.get("selector_enabled", True))
    }
    active_counterfactual_ids = {
        str(cf_set.get("cf_id", ""))
        for cf_set in get_counterfactual_sets(cache_entry)
        if any(
            (not is_need_unit_cache_version(cache_entry)) or bool(requirement.get("selector_enabled", True))
            for requirement in cf_set.get("requirements", [])
        )
    }
    current_positive_coverages = _compute_positive_unit_coverages(
        cache_entry=cache_entry,
        selected_positions=selected_positions,
        annotations_by_position=annotations_by_position,
        positive_score_overrides_by_position=positive_score_overrides_by_position,
    )
    pool_size = len(annotations_by_position)
    rows: List[Dict[str, Any]] = []
    for pos in candidate_positions:
        annotation = annotations_by_position.get(int(pos), {})
        candidate_positive_score_override_map, bridge_bonus_metadata = _resolve_requirement_candidate_bridge_bonus(
            annotation=annotation,
            cache_entry=cache_entry,
            current_positive_coverages=current_positive_coverages,
            bridge_bonus_binding_entities=bridge_bonus_binding_entities,
            bridge_bonus_mode=bridge_bonus_mode,
            bridge_bonus_weight=bridge_bonus_weight,
        )
        next_positive_score_overrides = merge_positive_score_overrides_by_position(
            positive_score_overrides_by_position=positive_score_overrides_by_position,
            pool_position=int(pos),
            override_map=candidate_positive_score_override_map,
        )
        next_positions = list(selected_positions) + [int(pos)]
        next_metrics = compute_requirement_state_metrics(
            cache_entry=cache_entry,
            selected_positions=next_positions,
            positive_score_overrides_by_position=next_positive_score_overrides,
            smooth_tau=smooth_tau,
            counterfactual_tau=counterfactual_tau,
        )
        positive_score_map = get_positive_score_map_with_overrides(
            annotation,
            int(pos),
            positive_score_overrides_by_position=next_positive_score_overrides,
        )
        contradiction_score_map = get_positive_contradiction_map(annotation)
        counterfactual_score_map = get_counterfactual_set_score_map(annotation)
        positive_scores = [
            float(score)
            for req_id, score in positive_score_map.items()
            if req_id in active_positive_ids
        ]
        counterfactual_set_scores = [
            float(score)
            for cf_id, score in counterfactual_score_map.items()
            if cf_id in active_counterfactual_ids
        ]
        contradiction_scores = [
            float(score)
            for req_id, score in contradiction_score_map.items()
            if req_id in active_positive_ids
        ]
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
            "bridge_bonus_applied": float(1.0 if bridge_bonus_metadata.get("applied") else 0.0),
            "bridge_bonus_score": float(bridge_bonus_metadata.get("score", 0.0) or 0.0),
            "bridge_bonus_base_score": float(bridge_bonus_metadata.get("base_score", 0.0) or 0.0),
            "bridge_bonus_alignment_score": float(bridge_bonus_metadata.get("alignment_score", 0.0) or 0.0),
            "bridge_bonus_predecessor_coverage": float(bridge_bonus_metadata.get("predecessor_coverage", 0.0) or 0.0),
            "bridge_bonus_overlap_entity_count": int(bridge_bonus_metadata.get("overlap_entity_count", 0) or 0),
            "bridge_bonus_novel_entity_count": int(bridge_bonus_metadata.get("novel_entity_count", 0) or 0),
            "bridge_bonus_mode": str(bridge_bonus_metadata.get("mode", bridge_bonus_mode) or bridge_bonus_mode),
            "bridge_bonus_unit_id": str(bridge_bonus_metadata.get("unit_id", "") or ""),
            "bridge_bonus_unit_type": str(bridge_bonus_metadata.get("unit_type", "") or ""),
            "bridge_bonus_unit_predicate": str(bridge_bonus_metadata.get("unit_predicate", "") or ""),
            "bridge_bonus_predecessor_id": str(bridge_bonus_metadata.get("predecessor_id", "") or ""),
            "positive_score_override_map": dict(candidate_positive_score_override_map),
            "positive_score_overrides_after": dict(next_positive_score_overrides),
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


def load_need_unit_atomic_model_bundle(model_path: str | Path) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid atomic need-unit scorer bundle at {model_path}")
    if str(bundle.get("task", "")).strip() not in {"", "atomic_multiclass"}:
        raise ValueError(
            f"Unsupported atomic scorer task at {model_path}: {bundle.get('task')}"
        )
    feature_names = list(bundle.get("feature_names") or [])
    if feature_names and feature_names != list(NEED_UNIT_ATOMIC_FEATURE_NAMES):
        raise ValueError(
            "Atomic scorer feature mismatch: "
            f"expected {list(NEED_UNIT_ATOMIC_FEATURE_NAMES)}, got {feature_names}"
        )
    bundle.setdefault("task", "atomic_multiclass")
    bundle.setdefault("feature_names", list(NEED_UNIT_ATOMIC_FEATURE_NAMES))
    bundle.setdefault("labels", list(NEED_UNIT_ATOMIC_LABELS))
    bundle.setdefault("bridge_alpha_by_unit_type", dict(DEFAULT_NEED_UNIT_BRIDGE_ALPHA_BY_TYPE))
    bundle.setdefault("beta_contradiction", float(DEFAULT_NEED_UNIT_CONTRADICTION_BETA))
    bundle.setdefault("scorer_version", NEED_UNIT_ATOMIC_SCORER_VERSION)
    bundle.setdefault("model_path", str(model_path))
    return bundle


def derive_need_unit_atomic_weak_label(unit: Dict[str, Any],
                                       feature_row: Dict[str, float | int],
                                       heuristic_scores: Dict[str, float | str],
                                       is_gold_doc: bool,
                                       is_counterfactual: bool = False) -> tuple[str | None, float]:
    subject_alignment = float(feature_row.get("subject_alignment", 0.0))
    predicate_alignment = float(feature_row.get("predicate_alignment", 0.0))
    object_alignment = float(feature_row.get("object_alignment", 0.0))
    constraint_alignment = float(feature_row.get("constraint_alignment", 0.0))
    bridge_feature_count = float(feature_row.get("bridge_feature_count", 0.0))
    contradiction_prior = float(feature_row.get("opposing_predicate_in_text", 0.0))
    alignment_score = float(heuristic_scores.get("alignment_score", 0.0) or 0.0)
    full_support_prob = float(heuristic_scores.get("full_support_prob", 0.0) or 0.0)
    bridge_support_prob = float(heuristic_scores.get("bridge_support_prob", 0.0) or 0.0)
    contradiction_prob = float(heuristic_scores.get("contradiction_prob", 0.0) or 0.0)
    nei_prob = float(heuristic_scores.get("nei_prob", 0.0) or 0.0)

    if contradiction_prior >= 0.6 or contradiction_prob >= 0.7:
        return "contradiction", 0.7
    if alignment_score < 0.35 and max(full_support_prob, bridge_support_prob, contradiction_prob) < 0.35 and nei_prob >= 0.5:
        return "nei", 0.7
    if is_counterfactual:
        return None, 0.0
    if is_gold_doc and predicate_alignment >= 0.7 and subject_alignment >= 0.7:
        object_ok = object_alignment >= 0.5 or float(feature_row.get("object_is_variable", 0.0)) > 0.0 or float(feature_row.get("has_constraints", 0.0)) == 0.0
        constraint_ok = constraint_alignment >= 0.5 or float(feature_row.get("has_constraints", 0.0)) == 0.0
        if object_ok and constraint_ok:
            return "full_support", 0.7
    if is_gold_doc and bridge_feature_count >= 1.0 and subject_alignment >= 0.5 and contradiction_prior < 0.2:
        return "bridge_support", 0.7
    return None, 0.0


def derive_need_unit_atomic_pseudo_label(heuristic_scores: Dict[str, float | str]) -> str:
    ranking = [
        ("full_support", float(heuristic_scores.get("full_support_prob", 0.0) or 0.0)),
        ("bridge_support", float(heuristic_scores.get("bridge_support_prob", 0.0) or 0.0)),
        ("contradiction", float(heuristic_scores.get("contradiction_prob", 0.0) or 0.0)),
        ("nei", float(heuristic_scores.get("nei_prob", 0.0) or 0.0)),
    ]
    ranking.sort(key=lambda item: (item[1], item[0] == "nei"), reverse=True)
    return str(ranking[0][0])
