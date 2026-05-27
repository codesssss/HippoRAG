#!/usr/bin/env python3
"""Support-grounding utilities for query obligations.

This module is deliberately about grounding, not ranking. It identifies source
facts/spans that can support an otherwise ungrounded query obligation when the
obligation already has at least one endpoint anchored by exact fact grounding.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from build_query_obligation_units import (
    bound_endpoint_matches_signatures,
    compatible_binding_intersection,
    endpoint_signature,
    endpoint_signature_tokens,
    fact_role_signatures,
    obligation_fact_variable_bindings,
    obligation_variable_names,
    relation_signature,
)
from query_obligation_typing import is_temporal_variable_name, looks_like_temporal_value


GENERIC_VARIABLE_NAMES = {
    "answer",
    "date",
    "entity",
    "month",
    "one",
    "person",
    "place",
    "state",
    "thing",
    "time",
    "x",
    "year",
}

WEAK_RELATION_TOKENS = {
    "a",
    "an",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def variable_values_from_matches(
    *,
    obligations: Sequence[Mapping[str, Any]],
    matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Set[str]]:
    values: Dict[str, Set[str]] = {}
    for obligation in obligations:
        obligation_id = str(obligation.get("obligation_id") or "")
        for match in matches.get(obligation_id, []) or []:
            for variable, binding_values in (match.get("variable_bindings", {}) or {}).items():
                clean_values = {
                    endpoint_signature(value)
                    for value in binding_values or []
                    if endpoint_signature(value)
                }
                if clean_values:
                    values.setdefault(str(variable), set()).update(clean_values)
    return values


def filter_matches_by_variable_values(
    *,
    matches: Mapping[str, Sequence[Mapping[str, Any]]],
    variable_values: Mapping[str, Set[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for obligation_id, obligation_matches in matches.items():
        rows: List[Dict[str, Any]] = []
        for match in obligation_matches or []:
            bindings = match.get("variable_bindings", {}) or {}
            consistent = True
            for variable, values in bindings.items():
                allowed = variable_values.get(str(variable), set())
                if not allowed:
                    continue
                clean_values = {
                    endpoint_signature(value)
                    for value in values or []
                    if endpoint_signature(value)
                }
                if clean_values and not compatible_binding_intersection(clean_values, allowed):
                    consistent = False
                    break
            if consistent:
                rows.append(dict(match))
        filtered[str(obligation_id)] = rows
    return filtered


def obligation_anchor_values(
    *,
    obligation: Mapping[str, Any],
    variable_values: Mapping[str, Set[str]],
) -> Dict[str, Set[str]]:
    anchors = {"subject": set(), "object": set(), "any": set()}
    if bool(obligation.get("subject_is_variable", False)):
        anchors["subject"].update(variable_values.get(str(obligation.get("subject_variable") or ""), set()))
    else:
        subject = endpoint_signature(obligation.get("subject"))
        if subject:
            anchors["subject"].add(subject)

    if bool(obligation.get("object_is_variable", False)):
        anchors["object"].update(variable_values.get(str(obligation.get("object_variable") or ""), set()))
    else:
        obj = endpoint_signature(obligation.get("object"))
        if obj:
            anchors["object"].add(obj)

    anchors["any"].update(anchors["subject"])
    anchors["any"].update(anchors["object"])
    return anchors


def compatible_any(left: Iterable[str], right: Iterable[str]) -> bool:
    return bool(compatible_binding_intersection(set(left), set(right)))


def normalized_token_set(*values: Any) -> Set[str]:
    tokens: Set[str] = set()
    for value in values:
        tokens.update(endpoint_signature_tokens(value))
    return {token for token in tokens if token}


def source_tokens_for_unit(unit: Mapping[str, Any]) -> Set[str]:
    return normalized_token_set(
        unit.get("title"),
        unit.get("source_text"),
        unit.get("span_text"),
        unit.get("source_passage_head"),
        " ".join(str(item) for item in (unit.get("fact", []) or [])),
    )


def relation_tokens_for_obligation(obligation: Mapping[str, Any]) -> Set[str]:
    relation = str(relation_signature(obligation.get("relation")) or obligation.get("relation") or "")
    return {token for token in relation.replace("_", " ").split() if token}


def content_relation_tokens_for_obligation(obligation: Mapping[str, Any]) -> Set[str]:
    return {
        token
        for token in relation_tokens_for_obligation(obligation)
        if token not in WEAK_RELATION_TOKENS
    }


def content_relation_tokens_for_text(value: Any) -> Set[str]:
    return {
        token
        for token in str(relation_signature(value) or "").replace("_", " ").split()
        if token and token not in WEAK_RELATION_TOKENS
    }


def literal_attribute_value_grounding_allowed(
    *,
    obligation: Mapping[str, Any],
    object_sentence_hits: Sequence[str],
) -> bool:
    """Allow source-span grounding for concrete literal attributes.

    OpenIE often stores lead-sentence attributes such as ``2009 film`` under an
    empty/copular relation.  A fully concrete ``year`` obligation is supported
    when the requested literal value appears in the anchored source span.  This
    does not apply to variable date obligations such as release-date searches.
    """

    if bool(obligation.get("object_is_variable", False)):
        return False
    if str(obligation.get("relation") or "") != "year":
        return False
    if not looks_like_temporal_value(obligation.get("object")):
        return False
    return bool(object_sentence_hits)


def simple_source_sentences(text: str) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"[\n.!?;]+", str(text or ""))
        if sentence.strip()
    ]


def source_span_grounding_candidates_for_obligation(
    *,
    obligation: Mapping[str, Any],
    candidate_units: Sequence[Mapping[str, Any]],
    variable_values: Mapping[str, Set[str]],
    max_examples: int,
) -> List[Dict[str, Any]]:
    """Find role-aware source spans that support an ungrounded obligation.

    This is stricter than display-only near misses. It requires contentful
    relation evidence in a local sentence and explicit provenance for every
    already-grounded role anchor. Tuple-aligned spans can bind unanchored
    operand variables, but they cannot overwrite text-only anchored roles.
    """

    anchors = obligation_anchor_values(obligation=obligation, variable_values=variable_values)
    if not anchors["any"]:
        return []
    relation_tokens = content_relation_tokens_for_obligation(obligation)
    if not relation_tokens:
        return []
    temporal_object_variable = bool(obligation.get("object_is_variable", False)) and is_temporal_variable_name(
        obligation.get("object_variable")
    )

    rows: List[Dict[str, Any]] = []
    seen_units: Set[str] = set()
    for unit in candidate_units:
        if str(unit.get("unit_type") or "") != "openie_fact":
            continue
        unit_id = str(unit.get("unit_id") or "")
        if unit_id in seen_units:
            continue
        fact = fact_role_signatures(unit)
        source_text = str(unit.get("source_text") or unit.get("span_text") or unit.get("source_passage_head") or "")
        title = str(unit.get("title") or "")
        title_tokens = normalized_token_set(title)
        subject_role_hit = compatible_any(anchors["subject"], fact.get("subject_signatures", set()))
        object_role_hit = compatible_any(anchors["object"], fact.get("object_signatures", set()))
        title_subject_hits = sorted(
            anchor
            for anchor in anchors["subject"]
            if set(endpoint_signature_tokens(anchor)) <= title_tokens
        )
        unit_relation_hits = sorted(
            relation_tokens & content_relation_tokens_for_text(unit.get("relation"))
        )
        sentence_rows: List[Dict[str, Any]] = []
        for sentence in simple_source_sentences(source_text):
            sentence_tokens = normalized_token_set(sentence)
            relation_hits = sorted(relation_tokens & content_relation_tokens_for_text(sentence))
            if temporal_object_variable and not looks_like_temporal_value(unit.get("object")):
                continue
            subject_sentence_hits = sorted(
                anchor
                for anchor in anchors["subject"]
                if set(endpoint_signature_tokens(anchor)) <= sentence_tokens
            )
            object_sentence_hits = sorted(
                anchor
                for anchor in anchors["object"]
                if set(endpoint_signature_tokens(anchor)) <= sentence_tokens
            )
            subject_satisfied = not anchors["subject"] or bool(subject_sentence_hits) or bool(title_subject_hits)
            object_satisfied = not anchors["object"] or bool(object_sentence_hits)
            if not (subject_satisfied and object_satisfied):
                continue
            if not (subject_sentence_hits or object_sentence_hits or title_subject_hits):
                continue
            literal_attribute_value_hit = literal_attribute_value_grounding_allowed(
                obligation=obligation,
                object_sentence_hits=object_sentence_hits,
            )
            if not relation_hits and not literal_attribute_value_hit:
                continue
            sentence_rows.append(
                {
                    "sentence": sentence[:360],
                    "relation_token_hits": relation_hits,
                    "literal_attribute_value_hit": literal_attribute_value_hit,
                    "subject_sentence_anchor_hits": subject_sentence_hits,
                    "object_sentence_anchor_hits": object_sentence_hits,
                    "title_subject_anchor_hits": title_subject_hits,
                }
            )
        if not sentence_rows:
            continue
        seen_units.add(unit_id)
        first = sentence_rows[0]
        reasons = ["source_span_grounding", "content_relation_token_hit"]
        if first["subject_sentence_anchor_hits"]:
            reasons.append("subject_sentence_anchor_hit")
        if first["object_sentence_anchor_hits"]:
            reasons.append("object_sentence_anchor_hit")
        if first["title_subject_anchor_hits"]:
            reasons.append("title_subject_anchor_hit")
        if unit_relation_hits:
            reasons.append("tuple_relation_aligned")
        if first.get("literal_attribute_value_hit", False):
            reasons.append("literal_attribute_value_hit")
        transport_endpoint_keys = sorted(
            {
                endpoint_signature(anchor)
                for anchor in (
                    list(first["subject_sentence_anchor_hits"])
                    + list(first["object_sentence_anchor_hits"])
                    + list(first["title_subject_anchor_hits"])
                )
                if endpoint_signature(anchor)
            }
        )
        rows.append(
            {
                "unit_id": unit_id,
                "fact_id": unit.get("fact_id"),
                "doc_index": int(unit.get("doc_index", -1) or -1),
                "title": title,
                "fact": list(unit.get("fact", []) or []),
                "subject": unit.get("subject"),
                "relation": unit.get("relation"),
                "object": unit.get("object"),
                "variable_bindings": (
                    grounding_variable_bindings(
                        obligation=obligation,
                        subject=unit.get("subject"),
                        obj=unit.get("object"),
                        bind_subject=not anchors["subject"] or subject_role_hit,
                        bind_object=not anchors["object"] or object_role_hit,
                    )
                    if bool(unit_relation_hits)
                    else {}
                ),
                "source_text": source_text[:360],
                "source_span_sentence": first["sentence"],
                "source_span_grounding_reasons": reasons,
                "relation_token_hits": first["relation_token_hits"],
                "source_span_tuple_relation_aligned": bool(unit_relation_hits),
                "source_span_tuple_relation_token_hits": unit_relation_hits,
                "source_span_transport_endpoint_keys": transport_endpoint_keys,
                "subject_sentence_anchor_hits": first["subject_sentence_anchor_hits"],
                "object_sentence_anchor_hits": first["object_sentence_anchor_hits"],
                "title_subject_anchor_hits": first["title_subject_anchor_hits"],
            }
        )

    rows.sort(
        key=lambda row: (
            not bool(row.get("source_span_tuple_relation_aligned", False)),
            "subject_sentence_anchor_hit" not in row["source_span_grounding_reasons"],
            "object_sentence_anchor_hit" not in row["source_span_grounding_reasons"],
            "title_subject_anchor_hit" not in row["source_span_grounding_reasons"],
            int(row["doc_index"]),
            str(row["title"]),
        )
    )
    return rows[:max_examples]


def variable_cue_tokens(obligation: Mapping[str, Any], variable_values: Mapping[str, Set[str]]) -> Set[str]:
    cues: Set[str] = set()
    for variable in obligation_variable_names(obligation):
        if variable in variable_values:
            continue
        normalized = endpoint_signature(variable)
        base_normalized = re.sub(r"\d+$", "", normalized)
        if (
            not normalized
            or normalized in GENERIC_VARIABLE_NAMES
            or base_normalized in GENERIC_VARIABLE_NAMES
            or normalized.startswith("x")
        ):
            continue
        cues.update(endpoint_signature_tokens(normalized))
    return cues


def concrete_endpoint_cue_tokens(obligation: Mapping[str, Any]) -> Set[str]:
    cues: Set[str] = set()
    if not bool(obligation.get("subject_is_variable", False)):
        cues.update(endpoint_signature_tokens(obligation.get("subject")))
    if not bool(obligation.get("object_is_variable", False)):
        cues.update(endpoint_signature_tokens(obligation.get("object")))
    return cues


def grounding_variable_bindings(
    *,
    obligation: Mapping[str, Any],
    subject: Any,
    obj: Any,
    bind_subject: bool = True,
    bind_object: bool = True,
) -> Dict[str, List[str]]:
    bindings: Dict[str, List[str]] = {}
    if bind_subject and bool(obligation.get("subject_is_variable", False)):
        variable = str(obligation.get("subject_variable") or "")
        value = endpoint_signature(subject)
        if variable and value:
            bindings[variable] = [value]
    if bind_object and bool(obligation.get("object_is_variable", False)):
        variable = str(obligation.get("object_variable") or "")
        value = endpoint_signature(obj)
        if variable and value:
            bindings[variable] = [value]
    return bindings


def support_grounding_candidates_for_obligation(
    *,
    obligation: Mapping[str, Any],
    candidate_units: Sequence[Mapping[str, Any]],
    variable_values: Mapping[str, Set[str]],
    max_examples: int,
    require_relation_exact: bool = False,
) -> List[Dict[str, Any]]:
    """Find source facts/spans that may support an ungrounded obligation."""

    anchors = obligation_anchor_values(obligation=obligation, variable_values=variable_values)
    if not anchors["any"]:
        return []
    relation_tokens = relation_tokens_for_obligation(obligation)
    obligation_cue_tokens = variable_cue_tokens(obligation, variable_values) | concrete_endpoint_cue_tokens(obligation)
    rows: List[Dict[str, Any]] = []
    target_relation = str(obligation.get("relation") or "")

    for unit in candidate_units:
        if str(unit.get("unit_type") or "") != "openie_fact":
            continue
        fact = fact_role_signatures(unit)
        source_tokens = source_tokens_for_unit(unit)
        fact_relation = str(fact.get("relation") or "")
        fact_relation_tokens = set(fact_relation.replace("_", " ").split())
        subject_role_hit = compatible_any(anchors["subject"], fact.get("subject_signatures", set()))
        object_role_hit = compatible_any(anchors["object"], fact.get("object_signatures", set()))
        text_anchor_hits = sorted(anchor for anchor in anchors["any"] if set(endpoint_signature_tokens(anchor)) <= source_tokens)
        subject_text_anchor_hits = sorted(
            anchor for anchor in anchors["subject"] if set(endpoint_signature_tokens(anchor)) <= source_tokens
        )
        object_text_anchor_hits = sorted(
            anchor for anchor in anchors["object"] if set(endpoint_signature_tokens(anchor)) <= source_tokens
        )
        subject_anchor_satisfied = not anchors["subject"] or subject_role_hit or bool(subject_text_anchor_hits)
        object_anchor_satisfied = not anchors["object"] or object_role_hit or bool(object_text_anchor_hits)
        if not (subject_anchor_satisfied and object_anchor_satisfied):
            continue

        relation_exact = fact_relation == target_relation
        relation_token_hits = sorted((relation_tokens & fact_relation_tokens) | (relation_tokens & source_tokens))
        relation_hit = relation_exact or bool(relation_token_hits)
        if require_relation_exact and not relation_exact:
            continue
        if not relation_hit:
            continue

        cue_token_hits = sorted(obligation_cue_tokens & source_tokens)
        if obligation_cue_tokens and not cue_token_hits:
            continue

        reasons = []
        if subject_role_hit:
            reasons.append("subject_role_anchor_hit")
        if object_role_hit:
            reasons.append("object_role_anchor_hit")
        if text_anchor_hits:
            reasons.append("source_text_anchor_hit")
        if subject_text_anchor_hits:
            reasons.append("subject_text_anchor_hit")
        if object_text_anchor_hits:
            reasons.append("object_text_anchor_hit")
        if relation_exact:
            reasons.append("relation_exact")
        elif relation_token_hits:
            reasons.append("relation_token_hit")
        if cue_token_hits:
            reasons.append("obligation_cue_token_hit")

        rows.append(
            {
                "unit_id": str(unit.get("unit_id") or ""),
                "fact_id": unit.get("fact_id"),
                "doc_index": int(unit.get("doc_index", -1) or -1),
                "title": str(unit.get("title") or ""),
                "fact": list(unit.get("fact", []) or []),
                "subject": unit.get("subject"),
                "relation": fact_relation,
                "object": unit.get("object"),
                "variable_bindings": grounding_variable_bindings(
                    obligation=obligation,
                    subject=unit.get("subject"),
                    obj=unit.get("object"),
                    bind_subject=not anchors["subject"] or subject_role_hit,
                    bind_object=not anchors["object"] or object_role_hit,
                ),
                "source_text": str(unit.get("source_text") or "")[:360],
                "support_grounding_reasons": reasons,
                "text_anchor_hits": text_anchor_hits,
                "subject_text_anchor_hits": subject_text_anchor_hits,
                "object_text_anchor_hits": object_text_anchor_hits,
                "relation_token_hits": relation_token_hits,
                "obligation_cue_token_hits": cue_token_hits,
            }
        )

    rows.sort(
        key=lambda row: (
            "relation_exact" not in row["support_grounding_reasons"],
            "source_text_anchor_hit" not in row["support_grounding_reasons"],
            -len(row["support_grounding_reasons"]),
            int(row["doc_index"]),
            str(row["title"]),
        )
    )
    return rows[:max_examples]


def support_grounded_obligation_matches(
    *,
    obligations: Sequence[Mapping[str, Any]],
    candidate_units: Sequence[Mapping[str, Any]],
    exact_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    max_matches_per_obligation: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return support-grounded pseudo-matches for exact-ungrounded obligations.

    The rule is intentionally stricter than the diagnostic display: relation
    signatures must match exactly, and at least one endpoint must already be
    grounded by exact fact matches. These pseudo-matches are eligible as graph
    seeds only under an explicit ablation flag.
    """

    variable_values = variable_values_from_matches(obligations=obligations, matches=exact_matches)
    support_matches: Dict[str, List[Dict[str, Any]]] = {}
    for obligation in obligations:
        obligation_id = str(obligation.get("obligation_id") or "")
        if not obligation_id or exact_matches.get(obligation_id):
            continue
        candidates = support_grounding_candidates_for_obligation(
            obligation=obligation,
            candidate_units=candidate_units,
            variable_values=variable_values,
            max_examples=max_matches_per_obligation,
            require_relation_exact=True,
        )
        if not candidates:
            continue
        support_matches[obligation_id] = [
            {
                "obligation_id": obligation_id,
                "unit_id": candidate["unit_id"],
                "fact_id": candidate.get("fact_id"),
                "doc_index": int(candidate.get("doc_index", -1) or -1),
                "subject": candidate.get("subject"),
                "relation": candidate.get("relation"),
                "object": candidate.get("object"),
                "variable_bindings": dict(candidate.get("variable_bindings", {}) or {}),
                "support_grounding": True,
                "support_grounding_reasons": candidate.get("support_grounding_reasons", []),
                "text_anchor_hits": candidate.get("text_anchor_hits", []),
                "subject_text_anchor_hits": candidate.get("subject_text_anchor_hits", []),
                "object_text_anchor_hits": candidate.get("object_text_anchor_hits", []),
                "obligation_cue_token_hits": candidate.get("obligation_cue_token_hits", []),
            }
            for candidate in candidates
            if str(candidate.get("unit_id") or "")
        ]
    return support_matches


def source_span_grounded_obligation_matches(
    *,
    obligations: Sequence[Mapping[str, Any]],
    candidate_units: Sequence[Mapping[str, Any]],
    exact_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    max_matches_per_obligation: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return role-aware source-span pseudo-matches.

    These matches are an explicit grounding object: they require existing role
    anchors and contentful relation evidence in source text, and carry source
    span provenance. Tuple-aligned source spans can bind the variable role that
    the supporting tuple realizes.
    """

    variable_values = variable_values_from_matches(obligations=obligations, matches=exact_matches)
    span_matches: Dict[str, List[Dict[str, Any]]] = {}
    for obligation in obligations:
        obligation_id = str(obligation.get("obligation_id") or "")
        if not obligation_id or exact_matches.get(obligation_id):
            continue
        candidates = source_span_grounding_candidates_for_obligation(
            obligation=obligation,
            candidate_units=candidate_units,
            variable_values=variable_values,
            max_examples=max_matches_per_obligation,
        )
        if not candidates:
            continue
        span_matches[obligation_id] = [
            {
                "obligation_id": obligation_id,
                "unit_id": candidate["unit_id"],
                "fact_id": candidate.get("fact_id"),
                "doc_index": int(candidate.get("doc_index", -1) or -1),
                "subject": candidate.get("subject"),
                "relation": candidate.get("relation"),
                "object": candidate.get("object"),
                "variable_bindings": (
                    dict(candidate.get("variable_bindings", {}) or {})
                    if bool(candidate.get("source_span_tuple_relation_aligned", False))
                    else {}
                ),
                "source_span_grounding": True,
                "source_span_grounding_reasons": candidate.get("source_span_grounding_reasons", []),
                "source_span_sentence": candidate.get("source_span_sentence", ""),
                "relation_token_hits": candidate.get("relation_token_hits", []),
                "source_span_tuple_relation_aligned": bool(
                    candidate.get("source_span_tuple_relation_aligned", False)
                ),
                "source_span_tuple_relation_token_hits": candidate.get(
                    "source_span_tuple_relation_token_hits", []
                ),
                "source_span_transport_endpoint_keys": candidate.get(
                    "source_span_transport_endpoint_keys", []
                ),
                "subject_sentence_anchor_hits": candidate.get("subject_sentence_anchor_hits", []),
                "object_sentence_anchor_hits": candidate.get("object_sentence_anchor_hits", []),
                "title_subject_anchor_hits": candidate.get("title_subject_anchor_hits", []),
            }
            for candidate in candidates
            if str(candidate.get("unit_id") or "")
        ]
    return span_matches


def merge_exact_and_support_matches(
    *,
    exact_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    support_matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, List[Dict[str, Any]]] = {
        str(obligation_id): [dict(match) for match in matches]
        for obligation_id, matches in exact_matches.items()
    }
    for obligation_id, matches in support_matches.items():
        merged.setdefault(str(obligation_id), []).extend(dict(match) for match in matches)
    return merged
