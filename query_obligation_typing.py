#!/usr/bin/env python3
"""Typed query-program diagnostics for query obligations.

This module classifies generated query triples into coarse program roles. It
does not decide retrieval ranking and does not change grounding behavior. The
purpose is to make obligation semantics explicit before adding new grounding
rules.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from build_query_obligation_units import (
    build_query_obligation_units,
    endpoint_signature,
    is_variable_endpoint,
    relation_signature,
    variable_name,
)


OBLIGATION_TYPES = {
    "retrieval_evidence",
    "bridge",
    "constraint",
    "answer_target",
    "aggregation",
    "comparison",
    "unknown",
}

ANSWER_VARIABLES = {"answer"}
AGGREGATION_VARIABLES = {"count", "date", "month", "number", "time", "year"}
TEMPORAL_VARIABLES = {"date", "month", "time", "year"}
MONTH_TOKENS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}
PERSON_COMPARISON_CUES = {
    "actor",
    "actress",
    "artist",
    "author",
    "composer",
    "director",
    "filmmaker",
    "man",
    "musician",
    "person",
    "player",
    "singer",
    "woman",
    "who",
    "writer",
}
BIRTH_EVENT_CUES = {
    "birth",
    "born",
}
AGE_COMPARISON_CUES = {
    "older",
    "oldest",
    "younger",
    "youngest",
}
WORK_RELEASE_CUES = {
    "album",
    "book",
    "episode",
    "film",
    "movie",
    "novel",
    "play",
    "record",
    "released",
    "song",
    "track",
    "work",
}
RELEASE_EVENT_CUES = {
    "came",
    "out",
    "release",
    "released",
}
DEATH_EVENT_CUES = {
    "dead",
    "death",
    "died",
    "die",
}
LIFESPAN_EVENT_CUES = {
    "live",
    "lived",
    "living",
    "liv",
}
LIFESPAN_COMPARISON_CUES = {
    "longer",
    "shorter",
}
CREATION_EVENT_TO_RELATION = {
    "built": "built in",
    "constructed": "constructed in",
    "created": "created in",
    "established": "established in",
    "formed": "formed in",
    "founded": "founded in",
}
GENERIC_VARIABLES = {
    "city",
    "country",
    "entity",
    "group",
    "location",
    "one",
    "person",
    "place",
    "region",
    "state",
    "thing",
    "x",
}

COMPARISON_RELATION_MARKERS = {
    "after",
    "before",
    "between",
    "compar",
    "earliest",
    "greater",
    "larger",
    "largest",
    "latest",
    "north",
    "oldest",
    "same",
    "smaller",
    "youngest",
}

COMPARISON_THAN_RELATION_MARKERS = {
    "earlier",
    "later",
    "less",
    "longer",
    "more",
    "older",
    "shorter",
    "younger",
}

TEMPORAL_COMPARISON_RELATION_MARKERS = {
    "after",
    "before",
    "earlier",
    "earliest",
    "later",
    "latest",
    "older",
    "oldest",
    "younger",
    "youngest",
}

CONSTRAINT_RELATION_MARKERS = {
    "contain",
    "located_between",
    "locat_between",
    "near",
    "part",
    "record_label",
    "related",
    "relat",
}

DERIVED_ATTRIBUTE_RELATION_MARKERS = {
    "age",
}

DESCRIPTOR_RELATION_MARKERS = {
    "batting_style",
    "country",
    "from_country",
    "nationality",
    "size",
    "style",
    "type",
}

ANSWER_QUERY_PREFIXES = (
    "who ",
    "what ",
    "when ",
    "where ",
    "which ",
    "how ",
)


def unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def relation_tokens(relation: str) -> Set[str]:
    return {token for token in str(relation or "").replace("_", " ").split() if token}


def variable_base_name(name: Any) -> str:
    return re.sub(r"\d+$", "", str(name or "").strip().lower())


def is_temporal_variable_name(name: Any) -> bool:
    return variable_base_name(name) in TEMPORAL_VARIABLES


def looks_like_temporal_value(value: Any) -> bool:
    normalized = endpoint_signature(value)
    if not normalized:
        return False
    tokens = set(normalized.split())
    if tokens & MONTH_TOKENS:
        return True
    if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", normalized):
        return True
    if re.search(r"\b[0-9]{1,2}(?:st|nd|rd|th)?\s+century\b", normalized):
        return True
    return False


def query_answer_shape(query: str) -> str:
    normalized = endpoint_signature(query)
    if normalized.startswith("how many") or normalized.startswith("how much"):
        return "count"
    if normalized.startswith("what month"):
        return "month"
    if normalized.startswith("what year") or normalized.startswith("when") or normalized.endswith(" when"):
        return "date"
    if normalized.startswith("where"):
        return "location"
    if normalized.startswith("who"):
        return "entity"
    return "unknown"


def raw_endpoint_role(value: Any) -> Dict[str, Any]:
    is_variable = is_variable_endpoint(value)
    name = variable_name(value) if is_variable else ""
    return {
        "raw": "" if value is None else str(value),
        "normalized": endpoint_signature(value),
        "is_variable": bool(is_variable),
        "variable": name,
    }


def variable_names_for_raw_triple(triple: Sequence[Any]) -> List[str]:
    if len(triple) != 3:
        return []
    names = []
    for endpoint in (triple[0], triple[2]):
        if is_variable_endpoint(endpoint):
            names.append(variable_name(endpoint))
    return unique_strings(names)


def comparison_operand_date_variable(index: int) -> str:
    return f"?date{max(int(index), 1)}"


def is_temporal_variable_endpoint(value: Any) -> bool:
    return is_variable_endpoint(value) and is_temporal_variable_name(variable_name(value))


def variable_endpoint(name: Any) -> str:
    clean = variable_name(name) if is_variable_endpoint(name) else str(name or "").strip().lstrip("?")
    return f"?{clean or 'x'}"


def add_unique_lowered_triple(
    *,
    lowered: List[List[str]],
    seen: Set[tuple],
    row: Sequence[Any],
) -> None:
    clean = ["" if value is None else str(value) for value in row[:3]]
    key = tuple(clean)
    if key in seen:
        return
    seen.add(key)
    lowered.append(clean)


def normalize_temporal_event_relation(row: Sequence[Any]) -> List[str]:
    clean = ["" if value is None else str(value) for value in row[:3]]
    if len(clean) != 3:
        return clean
    if not is_temporal_variable_endpoint(clean[2]):
        return clean
    tokens = relation_tokens(relation_signature(clean[1]))
    if tokens & BIRTH_EVENT_CUES:
        clean[1] = "born on"
    elif tokens & DEATH_EVENT_CUES:
        clean[1] = "died on"
    return clean


def has_relation_for_subject(
    *,
    triples: Sequence[Sequence[Any]],
    subject: Any,
    relation_markers: Set[str],
) -> bool:
    subject_key = endpoint_signature(subject)
    for triple in triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        if endpoint_signature(triple[0]) != subject_key:
            continue
        if relation_tokens(relation_signature(triple[1])) & relation_markers:
            return True
    return False


def has_temporal_operand_relation_for_subject(
    *,
    triples: Sequence[Sequence[Any]],
    subject: Any,
    relation_markers: Set[str],
) -> bool:
    subject_key = endpoint_signature(subject)
    for triple in triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        if endpoint_signature(triple[0]) != subject_key:
            continue
        if not is_variable_endpoint(triple[2]):
            continue
        if not is_temporal_variable_name(variable_name(triple[2])):
            continue
        if relation_tokens(relation_signature(triple[1])) & relation_markers:
            return True
    return False


def temporal_operand_variable_for_subject(
    *,
    triples: Sequence[Sequence[Any]],
    subject: Any,
    relation_markers: Set[str],
) -> str:
    subject_key = endpoint_signature(subject)
    for triple in triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        if endpoint_signature(triple[0]) != subject_key:
            continue
        if not is_variable_endpoint(triple[2]):
            continue
        if not is_temporal_variable_name(variable_name(triple[2])):
            continue
        if relation_tokens(relation_signature(triple[1])) & relation_markers:
            return variable_endpoint(triple[2])
    return ""


def comparison_operator_operands(triple: Sequence[Any]) -> List[Any]:
    if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
        return []
    relation = relation_signature(triple[1])
    tokens = relation_tokens(relation)
    if not is_comparison_relation(relation):
        return []
    if not (
        tokens & TEMPORAL_COMPARISON_RELATION_MARKERS
        or tokens & {"older", "oldest", "younger", "youngest"}
        or tokens & LIFESPAN_COMPARISON_CUES
    ):
        return []
    return [triple[0], triple[2]]


def variable_occurs_outside_comparison(*, query_triples: Sequence[Sequence[Any]], variable: str) -> bool:
    variable_key = str(variable or "").strip().lower().lstrip("?")
    if not variable_key:
        return False
    for triple in query_triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        if is_comparison_relation(relation_signature(triple[1])):
            continue
        for endpoint in (triple[0], triple[2]):
            if is_variable_endpoint(endpoint) and variable_name(endpoint) == variable_key:
                return True
    return False


def repair_comparison_branch_aliases(query_triples: Sequence[Sequence[Any]]) -> List[List[str]]:
    """Split duplicated branch variables when a comparison already declares two operands.

    Query parsers sometimes emit both branch anchors into the first comparison
    variable, e.g. Film A -> ?x1, Film B -> ?x1, ?x1 born later than ?x2.
    This is a program-normalization step: it materializes the missing second
    branch operand only when ?x2 has no retrieval branch elsewhere.
    """

    repaired: List[List[str]] = []
    valid_triples: List[List[str]] = []
    for triple in query_triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        valid_triples.append(["" if value is None else str(value) for value in triple[:3]])
    if not valid_triples:
        return []

    comparison_pairs: List[tuple[str, str]] = []
    for triple in valid_triples:
        operands = comparison_operator_operands(triple)
        if len(operands) != 2:
            continue
        if not (is_variable_endpoint(operands[0]) and is_variable_endpoint(operands[1])):
            continue
        left = variable_name(operands[0])
        right = variable_name(operands[1])
        if left and right and left != right:
            comparison_pairs.append((left, right))

    branch_rewrites: Dict[int, str] = {}
    for first_operand, second_operand in comparison_pairs:
        if variable_occurs_outside_comparison(query_triples=valid_triples, variable=second_operand):
            continue
        grouped_indices: Dict[str, List[int]] = {}
        for index, triple in enumerate(valid_triples):
            subject, relation, obj = triple
            if is_variable_endpoint(subject) or not is_variable_endpoint(obj):
                continue
            if variable_name(obj) != first_operand:
                continue
            normalized_relation = relation_signature(relation)
            if is_comparison_relation(normalized_relation):
                continue
            if relation_tokens(normalized_relation) & (
                BIRTH_EVENT_CUES
                | DEATH_EVENT_CUES
                | RELEASE_EVENT_CUES
                | set(CREATION_EVENT_TO_RELATION)
            ):
                continue
            grouped_indices.setdefault(normalized_relation, []).append(index)
        for indices in grouped_indices.values():
            if len(indices) != 2:
                continue
            branch_rewrites[indices[1]] = variable_endpoint(second_operand)
            break

    for index, triple in enumerate(valid_triples):
        row = list(triple)
        if index in branch_rewrites:
            row[2] = branch_rewrites[index]
        repaired.append(row)
    return repaired


def shared_object_comparison_subject_operands(
    *,
    query_triples: Sequence[Sequence[Any]],
    required_relation_markers: Set[str],
) -> List[Any]:
    """Recover operands from malformed rows like A died earlier than ?x; B died earlier than ?x."""

    object_to_subjects: Dict[str, List[Any]] = {}
    for triple in query_triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        relation = relation_signature(triple[1])
        tokens = relation_tokens(relation)
        if not (tokens & required_relation_markers):
            continue
        if not is_comparison_relation(relation):
            continue
        object_key = endpoint_signature(triple[2])
        if not object_key:
            continue
        object_to_subjects.setdefault(object_key, []).append(triple[0])
    for subjects in object_to_subjects.values():
        clean_subjects = unique_strings(subjects)
        if len(clean_subjects) >= 2:
            return clean_subjects[:2]
    return []


def shared_object_comparison_keys(
    *,
    query_triples: Sequence[Sequence[Any]],
    required_relation_markers: Set[str],
) -> Set[str]:
    object_to_subjects: Dict[str, List[Any]] = {}
    for triple in query_triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        relation = relation_signature(triple[1])
        tokens = relation_tokens(relation)
        if not (tokens & required_relation_markers):
            continue
        if not is_comparison_relation(relation):
            continue
        object_key = endpoint_signature(triple[2])
        if object_key:
            object_to_subjects.setdefault(object_key, []).append(triple[0])
    return {
        object_key
        for object_key, subjects in object_to_subjects.items()
        if len(unique_strings(subjects)) >= 2
    }


def lower_temporal_comparison_operands(
    *,
    lowered: List[List[str]],
    seen: Set[tuple],
    query_triples: Sequence[Sequence[Any]],
    relation_text: str,
    relation_markers: Set[str],
    operands: Sequence[Any],
) -> None:
    if len(operands) != 2:
        return
    if any(is_temporal_variable_endpoint(operand) for operand in operands):
        return
    date_variables: List[str] = []
    for index, operand in enumerate(operands, start=1):
        date_variable = temporal_operand_variable_for_subject(
            triples=lowered,
            subject=operand,
            relation_markers=relation_markers,
        )
        if date_variable:
            date_variables.append(date_variable)
            continue
        date_variable = comparison_operand_date_variable(index)
        date_variables.append(date_variable)
        add_unique_lowered_triple(
            lowered=lowered,
            seen=seen,
            row=[operand, relation_text, date_variable],
        )
    add_unique_lowered_triple(
        lowered=lowered,
        seen=seen,
        row=[date_variables[0], "earlier than", date_variables[1]],
    )


def lower_lifespan_comparison_operands(
    *,
    lowered: List[List[str]],
    seen: Set[tuple],
    query_triples: Sequence[Sequence[Any]],
    operands: Sequence[Any],
) -> None:
    """Declare birth/death evidence needed by a lifespan comparison."""

    if len(operands) != 2:
        return
    for index, operand in enumerate(operands, start=1):
        if not temporal_operand_variable_for_subject(
            triples=lowered,
            subject=operand,
            relation_markers={"birth", "born"},
        ):
            add_unique_lowered_triple(
                lowered=lowered,
                seen=seen,
                row=[operand, "born on", comparison_operand_date_variable((index - 1) * 2 + 1)],
            )
        if not temporal_operand_variable_for_subject(
            triples=lowered,
            subject=operand,
            relation_markers={"died", "die", "dead", "death"},
        ):
            add_unique_lowered_triple(
                lowered=lowered,
                seen=seen,
                row=[operand, "died on", comparison_operand_date_variable((index - 1) * 2 + 2)],
            )


def lower_query_triples_for_typed_program(
    *,
    query: str,
    query_triples: Sequence[Sequence[Any]],
) -> List[List[str]]:
    """Lower operator triples into retrieval-critical operand evidence.

    This is not a rescue rule. It turns a typed comparison operator into the
    evidence operands that the operator needs. The current lowering is
    deliberately narrow: age/older/younger comparisons over entity variables
    require birth-date evidence for each operand.
    """

    program_triples = repair_comparison_branch_aliases(query_triples)
    lowered: List[List[str]] = []
    seen = set()
    for triple in program_triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        add_unique_lowered_triple(
            lowered=lowered,
            seen=seen,
            row=normalize_temporal_event_relation(triple[:3]),
        )

    query_tokens = set(endpoint_signature(query).split())
    birth_context = bool(query_tokens & ({"age"} | BIRTH_EVENT_CUES)) or bool(
        query_tokens & PERSON_COMPARISON_CUES
    )
    birth_shared_object_keys = (
        shared_object_comparison_keys(
            query_triples=program_triples,
            required_relation_markers=AGE_COMPARISON_CUES | BIRTH_EVENT_CUES,
        )
        if birth_context
        else set()
    )
    death_context = bool(query_tokens & DEATH_EVENT_CUES)
    death_shared_object_keys = (
        shared_object_comparison_keys(
            query_triples=program_triples,
            required_relation_markers=DEATH_EVENT_CUES,
        )
        if death_context
        else set()
    )
    lifespan_context = bool(query_tokens & LIFESPAN_EVENT_CUES) and bool(query_tokens & LIFESPAN_COMPARISON_CUES)
    lifespan_shared_object_keys = (
        shared_object_comparison_keys(
            query_triples=program_triples,
            required_relation_markers=LIFESPAN_EVENT_CUES | LIFESPAN_COMPARISON_CUES,
        )
        if lifespan_context
        else set()
    )
    release_context = bool(query_tokens & WORK_RELEASE_CUES) and bool(query_tokens & RELEASE_EVENT_CUES)
    creation_relation = ""
    for cue, relation_text in CREATION_EVENT_TO_RELATION.items():
        if cue in query_tokens:
            creation_relation = relation_text
            break

    for triple in program_triples:
        operands = comparison_operator_operands(triple)
        if not operands:
            continue
        tokens = relation_tokens(relation_signature(triple[1]))
        if birth_context and (
            tokens & AGE_COMPARISON_CUES
            or (tokens & BIRTH_EVENT_CUES and tokens & TEMPORAL_COMPARISON_RELATION_MARKERS)
        ):
            if endpoint_signature(triple[2]) in birth_shared_object_keys:
                continue
            lower_temporal_comparison_operands(
                lowered=lowered,
                seen=seen,
                query_triples=program_triples,
                relation_text="born on",
                relation_markers={"birth", "born"},
                operands=operands,
            )
        elif death_context and tokens & DEATH_EVENT_CUES:
            if endpoint_signature(triple[2]) in death_shared_object_keys:
                continue
            lower_temporal_comparison_operands(
                lowered=lowered,
                seen=seen,
                query_triples=program_triples,
                relation_text="died on",
                relation_markers={"died", "die", "dead", "death"},
                operands=operands,
            )
        elif lifespan_context and tokens & LIFESPAN_COMPARISON_CUES:
            if endpoint_signature(triple[2]) in lifespan_shared_object_keys:
                continue
            lower_lifespan_comparison_operands(
                lowered=lowered,
                seen=seen,
                query_triples=program_triples,
                operands=operands,
            )
        elif release_context:
            lower_temporal_comparison_operands(
                lowered=lowered,
                seen=seen,
                query_triples=program_triples,
                relation_text="released in",
                relation_markers={"came", "out", "releas", "release"},
                operands=operands,
            )
        elif creation_relation:
            lower_temporal_comparison_operands(
                lowered=lowered,
                seen=seen,
                query_triples=program_triples,
                relation_text=creation_relation,
                relation_markers=set(relation_signature(creation_relation).split("_")),
                operands=operands,
            )
    if birth_context:
        operands = shared_object_comparison_subject_operands(
            query_triples=program_triples,
            required_relation_markers=AGE_COMPARISON_CUES | BIRTH_EVENT_CUES,
        )
        if operands:
            lower_temporal_comparison_operands(
                lowered=lowered,
                seen=seen,
                query_triples=program_triples,
                relation_text="born on",
                relation_markers={"birth", "born"},
                operands=operands,
            )
    if death_context:
        operands = shared_object_comparison_subject_operands(
            query_triples=program_triples,
            required_relation_markers=DEATH_EVENT_CUES,
        )
        if operands:
            lower_temporal_comparison_operands(
                lowered=lowered,
                seen=seen,
                query_triples=program_triples,
                relation_text="died on",
                relation_markers={"died", "die", "dead", "death"},
                operands=operands,
            )
    if lifespan_context:
        operands = shared_object_comparison_subject_operands(
            query_triples=program_triples,
            required_relation_markers=LIFESPAN_EVENT_CUES | LIFESPAN_COMPARISON_CUES,
        )
        if operands:
            lower_lifespan_comparison_operands(
                lowered=lowered,
                seen=seen,
                query_triples=program_triples,
                operands=operands,
            )
    return lowered


def has_concrete_endpoint(subject: Mapping[str, Any], obj: Mapping[str, Any]) -> bool:
    return not bool(subject["is_variable"]) or not bool(obj["is_variable"])


def has_only_variables(subject: Mapping[str, Any], obj: Mapping[str, Any]) -> bool:
    return bool(subject["is_variable"]) and bool(obj["is_variable"])


def relation_has_marker(relation: str, markers: Set[str]) -> bool:
    tokens = relation_tokens(relation)
    return bool(tokens & markers) or str(relation or "") in markers


def is_comparison_relation(relation: str) -> bool:
    tokens = relation_tokens(relation)
    if relation_has_marker(relation, COMPARISON_RELATION_MARKERS):
        return True
    return "than" in tokens and bool(tokens & COMPARISON_THAN_RELATION_MARKERS)


def is_temporal_comparison_relation(relation: str) -> bool:
    tokens = relation_tokens(relation)
    return bool(tokens & TEMPORAL_COMPARISON_RELATION_MARKERS)


def is_temporal_answer_obligation(*, query_shape: str, variables: Sequence[str], relation: str) -> bool:
    if query_shape not in {"date", "month"}:
        return False
    if not set(variables) & AGGREGATION_VARIABLES:
        return False
    return True


def has_temporal_relation(relation: str) -> bool:
    temporal_relation_tokens = {
        "abolish",
        "began",
        "begin",
        "born",
        "construct",
        "creat",
        "died",
        "end",
        "founded",
        "occur",
        "reach",
        "sign",
        "start",
    }
    return bool(relation_tokens(relation) & temporal_relation_tokens)


def is_type_descriptor_constraint(*, raw_relation: Any, subject: Mapping[str, Any], obj: Mapping[str, Any]) -> bool:
    if not has_only_variables(subject, obj):
        return False
    normalized = endpoint_signature(raw_relation)
    descriptor_prefixes = (
        "are a ",
        "are an ",
        "is a ",
        "is an ",
        "was a ",
        "was an ",
        "were a ",
        "were an ",
    )
    return normalized.startswith(descriptor_prefixes)


def is_derived_attribute_operand(*, query: str, relation: str, subject: Mapping[str, Any], obj: Mapping[str, Any]) -> bool:
    if not has_only_variables(subject, obj):
        return False
    if not relation_has_marker(relation, DERIVED_ATTRIBUTE_RELATION_MARKERS):
        return False
    query_tokens = set(endpoint_signature(query).split())
    comparison_cues = AGE_COMPARISON_CUES | LIFESPAN_COMPARISON_CUES | COMPARISON_RELATION_MARKERS
    return bool(query_tokens & comparison_cues)


def is_descriptor_relation_constraint(*, relation: str, subject: Mapping[str, Any], obj: Mapping[str, Any]) -> bool:
    """Return True for schema descriptors that should not relation-scan OpenIE.

    These rows describe a variable's type or derived attribute.  When both
    endpoints are variables, or when the subject is still an unbound variable,
    forcing exact grounding turns retrieval into a broad relation scan and
    blocks otherwise grounded programs.
    """

    if not bool(subject["is_variable"]):
        return False
    if not relation_has_marker(relation, DESCRIPTOR_RELATION_MARKERS):
        return False
    return bool(obj["is_variable"]) or bool(endpoint_signature(obj["raw"]))


def classify_query_triple(*, query: str, triple: Sequence[Any]) -> Dict[str, Any]:
    """Classify one raw query triple as a typed query-program obligation."""

    if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
        return {
            "obligation_type": "unknown",
            "retrieval_critical": False,
            "reason": "invalid_triple",
        }
    raw_subject, raw_relation, raw_object = triple
    subject = raw_endpoint_role(raw_subject)
    obj = raw_endpoint_role(raw_object)
    relation = relation_signature(raw_relation)
    variables = variable_names_for_raw_triple(triple)
    query_shape = query_answer_shape(query)
    materialized = build_query_obligation_units(query=query, query_triples=[triple])
    filtered_by_builder = not bool(materialized)

    if set(variables) & ANSWER_VARIABLES:
        return {
            "obligation_type": "answer_target",
            "retrieval_critical": False,
            "reason": "answer_variable",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if is_comparison_relation(relation):
        return {
            "obligation_type": "comparison",
            "retrieval_critical": False,
            "reason": "comparison_relation",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if is_temporal_answer_obligation(query_shape=query_shape, variables=variables, relation=relation):
        return {
            "obligation_type": "aggregation",
            "retrieval_critical": False,
            "reason": "answer_shape_temporal_extraction",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if query_shape == "count" and set(variables) & AGGREGATION_VARIABLES:
        return {
            "obligation_type": "aggregation",
            "retrieval_critical": False,
            "reason": "answer_shape_count",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if relation_has_marker(relation, CONSTRAINT_RELATION_MARKERS):
        return {
            "obligation_type": "constraint",
            "retrieval_critical": False,
            "reason": "constraint_relation",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if is_type_descriptor_constraint(raw_relation=raw_relation, subject=subject, obj=obj):
        return {
            "obligation_type": "constraint",
            "retrieval_critical": False,
            "reason": "type_descriptor_constraint",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if is_derived_attribute_operand(query=query, relation=relation, subject=subject, obj=obj):
        return {
            "obligation_type": "comparison",
            "retrieval_critical": False,
            "reason": "derived_attribute_operand",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if is_descriptor_relation_constraint(relation=relation, subject=subject, obj=obj):
        return {
            "obligation_type": "constraint",
            "retrieval_critical": False,
            "reason": "descriptor_relation_constraint",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if has_only_variables(subject, obj):
        return {
            "obligation_type": "bridge",
            "retrieval_critical": True,
            "reason": "variable_bridge",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    if has_concrete_endpoint(subject, obj):
        if variables:
            return {
                "obligation_type": "bridge",
                "retrieval_critical": True,
                "reason": "concrete_to_variable_binding",
                "query_answer_shape": query_shape,
                "filtered_by_obligation_builder": bool(filtered_by_builder),
            }
        return {
            "obligation_type": "retrieval_evidence",
            "retrieval_critical": True,
            "reason": "fully_concrete_evidence",
            "query_answer_shape": query_shape,
            "filtered_by_obligation_builder": bool(filtered_by_builder),
        }

    return {
        "obligation_type": "unknown",
        "retrieval_critical": False,
        "reason": "unclassified",
        "query_answer_shape": query_shape,
        "filtered_by_obligation_builder": bool(filtered_by_builder),
    }


def typed_query_program(*, query: str, query_triples: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {key: 0 for key in sorted(OBLIGATION_TYPES)}
    retrieval_critical_count = 0
    for index, triple in enumerate(query_triples):
        typing = classify_query_triple(query=query, triple=triple)
        obligation_type = str(typing.get("obligation_type") or "unknown")
        type_counts[obligation_type] = int(type_counts.get(obligation_type, 0)) + 1
        retrieval_critical = bool(typing.get("retrieval_critical", False))
        retrieval_critical_count += 1 if retrieval_critical else 0
        raw_subject, raw_relation, raw_object = list(triple)[:3]
        rows.append(
            {
                "triple_index": index,
                "raw_triple": [
                    "" if raw_subject is None else str(raw_subject),
                    "" if raw_relation is None else str(raw_relation),
                    "" if raw_object is None else str(raw_object),
                ],
                "normalized_triple": [
                    endpoint_signature(raw_subject),
                    relation_signature(raw_relation),
                    endpoint_signature(raw_object),
                ],
                "variables": variable_names_for_raw_triple(triple),
                **typing,
            }
        )
    variable_type_constraints: Dict[str, str] = {}
    variable_type_constraint_reasons: Dict[str, str] = {}
    for row in rows:
        if str(row.get("obligation_type") or "") != "comparison":
            continue
        relation = str((row.get("normalized_triple") or ["", "", ""])[1] or "")
        variables = [str(variable) for variable in row.get("variables", []) or []]
        if not (is_temporal_comparison_relation(relation) and any(is_temporal_variable_name(variable) for variable in variables)):
            continue
        for variable in variables:
            if is_temporal_variable_name(variable):
                variable_type_constraints[variable] = "temporal"
                variable_type_constraint_reasons[variable] = "temporal_comparison_operand"
    return {
        "query": str(query),
        "query_answer_shape": query_answer_shape(query),
        "triple_count": len(rows),
        "retrieval_critical_count": retrieval_critical_count,
        "type_counts": {key: value for key, value in sorted(type_counts.items()) if value},
        "variable_type_constraints": dict(sorted(variable_type_constraints.items())),
        "variable_type_constraint_reasons": dict(sorted(variable_type_constraint_reasons.items())),
        "typed_obligations": rows,
    }
