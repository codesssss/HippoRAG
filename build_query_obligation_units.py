#!/usr/bin/env python3
"""Build and bind query obligation units.

The closed-STO selector diagnostics showed that lexical query seeds are the
wrong reset distribution for local PPR. This module introduces the missing
upstream object: a query-side obligation unit with explicit subject, relation,
object, and variable boundary fields.

The module does not call an LLM. It consumes query triples when they are
available and binds them to OpenIE/STO fact units by role-aligned normalized
signatures. This keeps the interface clean: query parsing can improve later
without changing the graph/PPR layer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from build_obligation_closed_sto_units import fact_units
from build_source_title_openie_substrate import normalize_text, stable_id


RELATION_PREFIX_STOPWORDS = {
    "a",
    "an",
    "also",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "of",
    "the",
    "to",
    "was",
    "were",
}

VARIABLE_MARKERS = {
    "",
    "?",
    "?x",
    "answer",
    "date",
    "entity",
    "location",
    "one",
    "person",
    "place",
    "someone",
    "somebody",
    "something",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "x",
    "year",
}

TEMPORAL_VARIABLE_MARKERS = {
    "date",
    "day",
    "month",
    "time",
    "year",
}

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

BIRTH_EVENT_RELATION_TOKENS = {"birth", "born"}
DEATH_EVENT_RELATION_TOKENS = {"dead", "death", "die", "died"}
RELEASE_EVENT_RELATION_TOKENS = {"releas", "release", "released"}

# Relation frames are schema-level aliases, not ranking weights. They map
# query-side role nouns and corpus-side OpenIE predicates onto the same
# role-aware contract when the subject/object orientation is unchanged.
RELATION_FRAME_ALIASES = {
    "author": "written",
    "birthplace": "born_in",
    "designer": "design",
    "director": "direct",
    "owner": "own",
    "wrote": "written",
    "writer": "written",
    "winner": "won",
}

INVERTIBLE_RELATION_KEYS = {
    "written",
}


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


def soften_relation_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def relation_signature(value: Any) -> str:
    normalized = normalize_text(value)
    if normalized == "by":
        return "written"
    tokens = [
        soften_relation_token(token)
        for token in normalized.split()
        if token and token not in RELATION_PREFIX_STOPWORDS
    ]
    signature = "_".join(tokens)
    return RELATION_FRAME_ALIASES.get(signature, signature)


def relation_signature_tokens(value: Any) -> Set[str]:
    return {token for token in str(value or "").replace("_", " ").split() if token}


def endpoint_signature(value: Any) -> str:
    return normalize_text(value)


def endpoint_signature_tokens(value: Any) -> List[str]:
    return [token for token in endpoint_signature(value).split() if token]


TITLE_QUALIFIER_PREFIX_TOKENS = {
    "album",
    "book",
    "episode",
    "film",
    "movie",
    "novel",
    "play",
    "series",
    "song",
}


def endpoint_alias_signatures(value: Any) -> Set[str]:
    """Return conservative aliases for bound query endpoints.

    This is intentionally narrower than substring matching.  It covers two
    recurring query/corpus mismatches without turning every endpoint into a
    fuzzy token search: query-side type prefixes (``film Blood Street``) and
    parenthetical title qualifiers (``Dark River (2017 film)``).
    """

    aliases: Set[str] = set()
    raw = "" if value is None else str(value)
    base = endpoint_signature(raw)
    if base:
        aliases.add(base)

    without_parenthetical = endpoint_signature(re.sub(r"\([^)]*\)", " ", raw))
    if without_parenthetical:
        aliases.add(without_parenthetical)

    for signature in list(aliases):
        tokens = endpoint_signature_tokens(signature)
        if len(tokens) > 1 and tokens[0] in TITLE_QUALIFIER_PREFIX_TOKENS:
            stripped = " ".join(tokens[1:])
            if stripped:
                aliases.add(stripped)
    return aliases


def contains_token_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(list(haystack[start : start + width]) == list(needle) for start in range(0, len(haystack) - width + 1))


def merge_compatible_binding_value(left: str, right: str) -> str:
    """Return the canonical shared endpoint for compatible binding strings.

    OpenIE often emits a composite endpoint such as "108 miles southeast of
    Phoenix" in one fact and the embedded endpoint "Phoenix" in another. Query
    variable binding should converge to the embedded endpoint when it appears
    as complete normalized tokens, rather than treating the two strings as
    unrelated variables.
    """

    left_sig = endpoint_signature(left)
    right_sig = endpoint_signature(right)
    if not left_sig or not right_sig:
        return ""
    if left_sig == right_sig:
        return left_sig
    left_tokens = endpoint_signature_tokens(left_sig)
    right_tokens = endpoint_signature_tokens(right_sig)
    if contains_token_sequence(left_tokens, right_tokens):
        return right_sig
    if contains_token_sequence(right_tokens, left_tokens):
        return left_sig
    return ""


def compatible_binding_intersection(left_values: Iterable[str], right_values: Iterable[str]) -> Set[str]:
    merged: Set[str] = set()
    for left in left_values:
        for right in right_values:
            value = merge_compatible_binding_value(str(left), str(right))
            if value:
                merged.add(value)
    return merged


def is_variable_endpoint(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    if raw.startswith("?"):
        return True
    normalized = endpoint_signature(value)
    return normalized in VARIABLE_MARKERS or normalized.startswith("?")


def variable_name(value: Any) -> str:
    if not is_variable_endpoint(value):
        return ""
    raw = str(value or "").strip().lower()
    if raw.startswith("?"):
        raw = raw[1:]
    normalized = endpoint_signature(raw)
    return normalized or "answer"


def base_variable_name(value: Any) -> str:
    return re.sub(r"\d+$", "", variable_name(value))


def looks_like_temporal_endpoint(value: Any) -> bool:
    signature = endpoint_signature(value)
    if not signature:
        return False
    tokens = set(endpoint_signature_tokens(signature))
    if tokens & MONTH_TOKENS:
        return True
    if re.search(r"\b\d{3,4}\b", signature):
        return True
    if re.search(r"\b\d{1,2}\s+(?:%s)\b" % "|".join(sorted(MONTH_TOKENS)), signature):
        return True
    return False


def is_temporal_endpoint(value: Any) -> bool:
    if is_variable_endpoint(value):
        return base_variable_name(value) in TEMPORAL_VARIABLE_MARKERS
    return looks_like_temporal_endpoint(value)


def temporal_event_relation_signature(relation: Any, object_endpoint: Any) -> str:
    """Canonicalize typed event relations when the object role supports it.

    ``born in London`` and ``died in Paris`` remain place relations. ``born in
    1950`` and ``born on 1 January 1950`` both become the same typed event
    relation, which lets query obligations and OpenIE facts meet at the
    semantic role boundary rather than at surface prepositions.
    """

    signature = relation_signature(relation)
    if not signature:
        return signature
    if not is_temporal_endpoint(object_endpoint):
        return signature
    tokens = relation_signature_tokens(signature)
    if tokens & BIRTH_EVENT_RELATION_TOKENS:
        return "born_on"
    if tokens & DEATH_EVENT_RELATION_TOKENS:
        return "died_on"
    if signature == "release_date" or tokens & RELEASE_EVENT_RELATION_TOKENS:
        return "released_on"
    return signature


def is_open_answer_assertion(
    *,
    subject_is_variable: bool,
    object_is_variable: bool,
    subject_variable: str,
    object_variable: str,
) -> bool:
    """Return True for answer-target assertions that cannot seed retrieval.

    A triple such as ``?answer --won--> ?race`` states what the reader must
    solve after evidence is retrieved. With both endpoints unbound, forcing it
    to ground during retrieval either blocks valid queries or relation-scans the
    graph. A triple like ``?answer --directed--> Inception`` is not filtered
    because the concrete endpoint can ground the answer-bearing fact directly.
    """

    if not subject_is_variable or not object_is_variable:
        return False
    return subject_variable == "answer" or object_variable == "answer"


def build_query_obligation_units(
    *,
    query: str,
    query_triples: Sequence[Sequence[Any]],
) -> List[Dict[str, Any]]:
    """Convert query triples into role-aligned obligation units."""

    obligations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, bool, bool]] = set()
    for triple in query_triples:
        if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
            continue
        raw_subject, raw_relation, raw_object = triple
        relation = temporal_event_relation_signature(raw_relation, raw_object)
        if not relation:
            continue
        subject = endpoint_signature(raw_subject)
        obj = endpoint_signature(raw_object)
        subject_is_variable = is_variable_endpoint(raw_subject)
        object_is_variable = is_variable_endpoint(raw_object)
        subject_variable = variable_name(raw_subject) if subject_is_variable else ""
        object_variable = variable_name(raw_object) if object_is_variable else ""
        if is_open_answer_assertion(
            subject_is_variable=subject_is_variable,
            object_is_variable=object_is_variable,
            subject_variable=subject_variable,
            object_variable=object_variable,
        ):
            continue
        key = (subject, relation, obj, subject_is_variable, object_is_variable)
        if key in seen:
            continue
        seen.add(key)
        obligation_id = stable_id(
            "query-obligation",
            [
                normalize_text(query),
                subject,
                relation,
                obj,
                "subjvar" if subject_is_variable else "subjbound",
                "objvar" if object_is_variable else "objbound",
            ],
        )
        obligations.append(
            {
                "obligation_id": obligation_id,
                "unit_type": "query_obligation_unit",
                "query": str(query),
                "raw_subject": "" if raw_subject is None else str(raw_subject),
                "raw_relation": "" if raw_relation is None else str(raw_relation),
                "raw_object": "" if raw_object is None else str(raw_object),
                "subject": subject,
                "relation": relation,
                "object": obj,
                "subject_is_variable": subject_is_variable,
                "object_is_variable": object_is_variable,
                "subject_variable": subject_variable,
                "object_variable": object_variable,
            }
        )
    return obligations


def fact_role_signatures(unit: Mapping[str, Any]) -> Dict[str, Any]:
    subject_values = unique_strings(
        [
            unit.get("grounded_subject"),
            unit.get("subject"),
        ]
    )
    object_values = unique_strings(
        [
            unit.get("grounded_object"),
            unit.get("object"),
        ]
    )
    object_for_relation = next((value for value in object_values if is_temporal_endpoint(value)), object_values[0] if object_values else "")
    return {
        "unit_id": str(unit.get("unit_id") or ""),
        "fact_id": unit.get("fact_id"),
        "doc_index": int(unit.get("doc_index", -1) or -1),
        "subject_signatures": {endpoint_signature(value) for value in subject_values if endpoint_signature(value)},
        "relation": temporal_event_relation_signature(unit.get("relation"), object_for_relation),
        "object_signatures": {endpoint_signature(value) for value in object_values if endpoint_signature(value)},
    }


def bound_endpoint_matches_signatures(
    query_endpoint: Any,
    fact_signatures: Iterable[str],
) -> bool:
    """Return True when a bound query endpoint aligns with one of the fact's role signatures.

    Queries often add qualifying prefixes or parenthetical aliases that the
    corpus does not carry.  We handle those as explicit endpoint aliases rather
    than accepting arbitrary token containment.
    """

    query_aliases = endpoint_alias_signatures(query_endpoint)
    if not query_aliases:
        return False
    for fact_signature in fact_signatures:
        if endpoint_signature(fact_signature) in query_aliases:
            return True
    return False


def obligation_matches_fact(
    *,
    obligation: Mapping[str, Any],
    fact_unit: Mapping[str, Any],
) -> bool:
    """Return True when a query obligation binds to a fact with aligned roles."""

    return bool(obligation_fact_role_alignment(obligation=obligation, fact_unit=fact_unit))


def obligation_endpoint_roles_match(
    *,
    obligation: Mapping[str, Any],
    fact: Mapping[str, Any],
    alignment: str,
) -> bool:
    subject_is_variable = bool(obligation.get("subject_is_variable", False))
    object_is_variable = bool(obligation.get("object_is_variable", False))
    fact_subject_signatures = fact["subject_signatures"]
    fact_object_signatures = fact["object_signatures"]
    if alignment == "inverse":
        fact_subject_signatures, fact_object_signatures = fact_object_signatures, fact_subject_signatures
    if not subject_is_variable:
        subject_endpoint = obligation.get("raw_subject") or obligation.get("subject")
        if not bound_endpoint_matches_signatures(subject_endpoint, fact_subject_signatures):
            return False
    if not object_is_variable:
        object_endpoint = obligation.get("raw_object") or obligation.get("object")
        if not bound_endpoint_matches_signatures(object_endpoint, fact_object_signatures):
            return False
    return True


def obligation_fact_role_alignment(
    *,
    obligation: Mapping[str, Any],
    fact_unit: Mapping[str, Any],
) -> str:
    """Return direct/inverse role alignment for a matched obligation-fact pair."""

    fact = fact_role_signatures(fact_unit)
    if not str(obligation.get("relation") or ""):
        return ""
    if str(obligation.get("relation") or "") != str(fact.get("relation") or ""):
        return ""
    if obligation_endpoint_roles_match(obligation=obligation, fact=fact, alignment="direct"):
        return "direct"
    if str(obligation.get("relation") or "") in INVERTIBLE_RELATION_KEYS and obligation_endpoint_roles_match(
        obligation=obligation,
        fact=fact,
        alignment="inverse",
    ):
        return "inverse"
    return ""


def obligation_fact_variable_bindings(
    *,
    obligation: Mapping[str, Any],
    fact_unit: Mapping[str, Any],
    role_alignment: str = "direct",
) -> Dict[str, Set[str]]:
    fact = fact_role_signatures(fact_unit)
    subject_signatures = fact["subject_signatures"]
    object_signatures = fact["object_signatures"]
    if role_alignment == "inverse":
        subject_signatures, object_signatures = object_signatures, subject_signatures
    bindings: Dict[str, Set[str]] = {}
    if bool(obligation.get("subject_is_variable", False)):
        name = str(obligation.get("subject_variable") or "")
        if name:
            bindings[name] = set(subject_signatures)
    if bool(obligation.get("object_is_variable", False)):
        name = str(obligation.get("object_variable") or "")
        if name:
            bindings[name] = set(object_signatures)
    return bindings


def obligation_variable_names(obligation: Mapping[str, Any]) -> List[str]:
    return unique_strings(
        [
            obligation.get("subject_variable") if bool(obligation.get("subject_is_variable", False)) else "",
            obligation.get("object_variable") if bool(obligation.get("object_is_variable", False)) else "",
        ]
    )


def obligation_has_bound_endpoint(obligation: Mapping[str, Any]) -> bool:
    return not bool(obligation.get("subject_is_variable", False)) or not bool(obligation.get("object_is_variable", False))


def match_consistent_with_allowed(
    *,
    match: Mapping[str, Any],
    variables: Sequence[str],
    allowed_bindings: Mapping[str, Set[str]],
    require_all_variables_bound: bool,
) -> bool:
    variable_bindings = match.get("variable_bindings", {}) or {}
    for variable in variables:
        values = set(variable_bindings.get(variable, []) or [])
        allowed = allowed_bindings.get(variable)
        if allowed is None:
            if require_all_variables_bound:
                return False
            continue
        if values and not compatible_binding_intersection(values, allowed):
            return False
    return True


def union_variable_support(
    matches: Sequence[Mapping[str, Any]],
    variable: str,
) -> Set[str]:
    return {
        value
        for match in matches
        for value in (match.get("variable_bindings", {}) or {}).get(variable, []) or []
    }


def match_query_obligations_to_sto_facts(
    *,
    obligations: Sequence[Mapping[str, Any]],
    candidate_units: Sequence[Mapping[str, Any]],
    seed_matches: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Bind query obligations to STO/OpenIE fact units without lexical bagging."""

    facts = fact_units(candidate_units)
    raw_matches: Dict[str, List[Dict[str, Any]]] = {}
    obligations_by_id: Dict[str, Mapping[str, Any]] = {}
    for obligation in obligations:
        obligation_id = str(obligation.get("obligation_id") or "")
        if not obligation_id:
            continue
        obligations_by_id[obligation_id] = obligation
        bound: List[Dict[str, Any]] = []
        for unit in facts:
            role_alignment = obligation_fact_role_alignment(obligation=obligation, fact_unit=unit)
            if not role_alignment:
                continue
            variable_bindings = obligation_fact_variable_bindings(
                obligation=obligation,
                fact_unit=unit,
                role_alignment=role_alignment,
            )
            bound.append(
                {
                    "obligation_id": obligation_id,
                    "unit_id": str(unit.get("unit_id") or ""),
                    "fact_id": unit.get("fact_id"),
                    "doc_index": int(unit.get("doc_index", -1) or -1),
                    "subject": unit.get("subject"),
                    "relation": unit.get("relation"),
                    "object": unit.get("object"),
                    "role_alignment": role_alignment,
                    "variable_bindings": {
                        name: sorted(values) for name, values in variable_bindings.items()
                    },
                }
            )
        raw_matches[obligation_id] = bound

    allowed_bindings: Dict[str, Set[str]] = {}
    variable_supports: Dict[str, List[Set[str]]] = {}

    # Seed variable bindings only from obligations that contain a concrete
    # endpoint. Fully variable obligations are useful for propagation after a
    # variable is grounded, but by themselves they are just relation-name scans.
    for matches in (seed_matches or {}).values():
        seed_support_by_variable: Dict[str, Set[str]] = {}
        for match in matches or []:
            for variable, values in (match.get("variable_bindings", {}) or {}).items():
                support = {
                    endpoint_signature(value)
                    for value in values or []
                    if endpoint_signature(value)
                }
                if support:
                    seed_support_by_variable.setdefault(str(variable), set()).update(support)
        for variable, support in seed_support_by_variable.items():
            if support:
                variable_supports.setdefault(variable, []).append(support)

    for obligation_id, bound in raw_matches.items():
        obligation = obligations_by_id.get(obligation_id, {})
        if not obligation_has_bound_endpoint(obligation):
            continue
        for variable in obligation_variable_names(obligation):
            support = union_variable_support(bound, variable)
            if support:
                variable_supports.setdefault(variable, []).append(support)

    for variable, supports in variable_supports.items():
        allowed = set(supports[0])
        for support in supports[1:]:
            allowed &= set(support)
        if allowed:
            allowed_bindings[variable] = allowed

    changed = True
    while changed:
        changed = False
        for obligation_id, bound in raw_matches.items():
            obligation = obligations_by_id.get(obligation_id, {})
            variables = obligation_variable_names(obligation)
            if not variables:
                continue
            if not obligation_has_bound_endpoint(obligation) and not any(variable in allowed_bindings for variable in variables):
                continue
            consistent = [
                match
                for match in bound
                if match_consistent_with_allowed(
                    match=match,
                    variables=variables,
                    allowed_bindings=allowed_bindings,
                    require_all_variables_bound=False,
                )
            ]
            if not consistent:
                continue
            for variable in variables:
                support = union_variable_support(consistent, variable)
                if not support:
                    continue
                if variable in allowed_bindings:
                    next_allowed = compatible_binding_intersection(allowed_bindings[variable], support)
                else:
                    next_allowed = support
                if next_allowed and next_allowed != allowed_bindings.get(variable):
                    allowed_bindings[variable] = next_allowed
                    changed = True

    matches: Dict[str, List[Dict[str, Any]]] = {}
    for obligation_id, bound in raw_matches.items():
        obligation = obligations_by_id.get(obligation_id, {})
        variables = obligation_variable_names(obligation)
        filtered: List[Dict[str, Any]] = []
        for match in bound:
            if match_consistent_with_allowed(
                match=match,
                variables=variables,
                allowed_bindings=allowed_bindings,
                require_all_variables_bound=bool(variables),
            ):
                filtered.append(match)
        matches[obligation_id] = filtered
    return matches


def obligation_seed_unit_ids(matches: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    return unique_strings(
        match.get("unit_id")
        for obligation_matches in matches.values()
        for match in obligation_matches
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build query obligation units from query triples.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-triples-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    triples = load_json(Path(args.query_triples_json))
    obligations = build_query_obligation_units(query=str(args.query), query_triples=triples)
    output = Path(args.output_json).resolve()
    output.write_text(json.dumps({"query": args.query, "obligations": obligations}, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"obligation_count": len(obligations)}, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
