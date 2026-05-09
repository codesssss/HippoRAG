"""Source-text certificate graph for vocab-strict retrieval.

This module extracts the clean graph-construction primitive behind the
high-score source-authorized path.  It does not rank with weighted score
fusion.  It only decides whether a source passage certifies an evidence edge
to a target passage, using deterministic corpus-observable evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .normalize import (
    is_specific_title_alias,
    normalize_text,
    phrase_occurs,
    title_aliases,
    tokens,
    unique_ints,
)


Triple = Tuple[str, str, str]

TITLE_ALIAS_CERTIFICATE = "title_alias"
ENDPOINT_TITLE_CERTIFICATE = "endpoint_title"
SOURCE_TITLE_ENDPOINT_CERTIFICATE = "source_title_endpoint"
RELATION_ROLE_CERTIFICATE = "relation_role"
SURFACE_ROLE_CERTIFICATE = "surface_role"
DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE = "directional_child_to_parent"

CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY = "canonical_source_text"
CANONICAL_TRANSITION_CERTIFICATE_POLICY = "canonical_transition"
CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY = "canonical_query_transition"
CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY = "canonical_fact_origin_transition"
LEGACY_ROLE_HINT_CERTIFICATE_POLICY = "legacy_role_hints"

SOURCE_TEXT_CERTIFICATE_GRAPH_CONTRACT: Mapping[str, bool | str] = {
    "paper_facing_graph_name": "source_text_certificate_graph",
    "edge_semantics": "sentence_grounded_evidence_transition",
    "certificate_policy": CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    "uses_sentence_grounded_evidence_transitions": True,
    "strong_certificates_require_source_sentence": True,
    "strong_certificates_require_source_target_binding": True,
    "uses_role_vocabulary": False,
    "uses_relation_role_certificates": False,
    "uses_surface_role_certificates": False,
    "uses_directional_child_parent_hints": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_proposition_support_evidence": False,
    "uses_learned_reranker": False,
}


def normalize_certificate_policy(policy: object) -> str:
    """Return the supported certificate policy name."""

    normalized = str(policy or "").strip().lower()
    if normalized in {
        CANONICAL_TRANSITION_CERTIFICATE_POLICY,
        "source_text_transition",
        "sentence_grounded_transition",
        "canonical_source_text_transition",
    }:
        return CANONICAL_TRANSITION_CERTIFICATE_POLICY
    if normalized in {
        CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY,
        "query_transition",
        "query_aligned_transition",
        "canonical_query_aligned_transition",
    }:
        return CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY
    if normalized in {
        CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY,
        "fact_origin_transition",
        "canonical_query_fact_origin_transition",
        "canonical_fact_transition",
    }:
        return CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY
    if normalized in {
        LEGACY_ROLE_HINT_CERTIFICATE_POLICY,
        "legacy",
        "role_hints",
        "source_text_with_role_hints",
    }:
        return LEGACY_ROLE_HINT_CERTIFICATE_POLICY
    return CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY


def certificate_policy_contract(policy: object) -> Mapping[str, bool | str]:
    """Describe whether a graph policy uses hand-written role/hint machinery."""

    normalized = normalize_certificate_policy(policy)
    uses_role_hints = normalized == LEGACY_ROLE_HINT_CERTIFICATE_POLICY
    return {
        "certificate_policy": normalized,
        "uses_source_bound_admission_guard": bool(
            normalized == CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY
        ),
        "uses_sentence_grounded_transition_admission": bool(
            normalized == CANONICAL_TRANSITION_CERTIFICATE_POLICY
        ),
        "uses_query_aligned_transition_admission": bool(
            normalized
            in {
                CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY,
                CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY,
            }
        ),
        "uses_fact_origin_fanout_suppression": bool(
            normalized == CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY
        ),
        "uses_role_vocabulary": bool(uses_role_hints),
        "uses_relation_role_certificates": bool(uses_role_hints),
        "uses_surface_role_certificates": bool(uses_role_hints),
        "uses_directional_child_parent_hints": bool(uses_role_hints),
    }

_ROLE_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
}

_CHILD_TO_PARENT_RELATION_HINTS = (
    "son of",
    "daughter of",
    "child of",
    "children of",
    "born to",
)

_PARENT_ROLE_TERMS = {"father", "mother", "parent"}

_SOURCE_HEAD_BINDING_STOP_TOKENS = {
    "entity",
    "page",
    "passage",
}

_ROLE_EQUIVALENTS = {
    "author": {"writer", "written", "wrote"},
    "band": {"group", "performer"},
    "composed": {"composer"},
    "composer": {"composed"},
    "group": {"band"},
    "husband": {"spouse"},
    "married": {"spouse"},
    "performed": {"performer"},
    "performer": {"band", "performed"},
    "spouse": {"husband", "married", "wife"},
    "wife": {"spouse"},
    "writer": {"author", "written", "wrote"},
    "written": {"author", "writer", "wrote"},
    "wrote": {"author", "writer", "written"},
}


@dataclass(frozen=True)
class EvidenceNode:
    """A candidate evidence passage with OpenIE facts."""

    doc_index: int
    text: str
    title: str = ""
    triples: Tuple[Triple, ...] = ()

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        first_line = str(self.text).splitlines()[0].strip() if self.text else ""
        return first_line


@dataclass(frozen=True)
class EvidenceCertificate:
    """A binary certificate authorizing source -> target evidence reachability."""

    source_doc_index: int
    target_doc_index: int
    certificate_type: str
    endpoint: str
    source_title: str
    target_title: str
    roles: Tuple[str, ...] = ()
    direction: str = ""
    triple: Optional[Triple] = None
    surface_evidence: str = ""
    source_sentence: str = ""
    source_binding: str = ""
    target_binding: str = ""
    evidence_unit_id: str = ""
    fact_origin: Tuple[int, int] = ()


@dataclass(frozen=True)
class SourceTextCertificateGraph:
    """A query-local graph whose edges are source-text certificates."""

    candidate_doc_indices: Tuple[int, ...]
    nodes: Tuple[EvidenceNode, ...]
    certificates: Tuple[EvidenceCertificate, ...]
    metadata: Mapping[str, object] = field(
        default_factory=lambda: dict(SOURCE_TEXT_CERTIFICATE_GRAPH_CONTRACT)
    )

    def outgoing(self, source_doc_index: int) -> Tuple[EvidenceCertificate, ...]:
        return tuple(
            certificate
            for certificate in self.certificates
            if int(certificate.source_doc_index) == int(source_doc_index)
        )

    def between(
        self,
        source_doc_index: int,
        target_doc_index: int,
    ) -> Tuple[EvidenceCertificate, ...]:
        return tuple(
            certificate
            for certificate in self.certificates
            if int(certificate.source_doc_index) == int(source_doc_index)
            and int(certificate.target_doc_index) == int(target_doc_index)
        )

    def certificate_type_counts(self) -> Mapping[str, int]:
        counts: Dict[str, int] = {}
        for certificate in self.certificates:
            counts[str(certificate.certificate_type)] = (
                counts.get(str(certificate.certificate_type), 0) + 1
            )
        return counts


def build_source_text_certificate_graph(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Optional[Sequence[int]] = None,
    source_doc_indices: Optional[Sequence[int]] = None,
    extra_role_terms: Optional[Iterable[str]] = None,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
) -> SourceTextCertificateGraph:
    """Build source-text certified evidence edges for a query-local pool.

    Dense retrieval or a previous retriever may define the local candidate
    universe, but it does not authorize final evidence edges.  An edge exists
    only if the source text or source OpenIE triples certify the target.
    """

    certificate_policy = normalize_certificate_policy(certificate_policy)
    node_lookup = {int(node.doc_index): node for node in nodes}
    if candidate_doc_indices is None:
        candidates = unique_ints(node_lookup)
    else:
        candidates = unique_ints(
            index for index in candidate_doc_indices if int(index) in node_lookup
        )
    if source_doc_indices is None:
        sources = tuple(candidates)
    else:
        candidate_set = {int(index) for index in candidates}
        sources = tuple(
            unique_ints(
                index
                for index in source_doc_indices
                if int(index) in candidate_set and int(index) in node_lookup
            )
        )
    candidate_nodes = tuple(node_lookup[index] for index in candidates)

    aliases_by_doc = {
        index: _source_aliases(node_lookup[index]) for index in sources
    }
    target_aliases_by_doc = {
        index: title_aliases(node_lookup[index].display_title) for index in candidates
    }
    target_alias_index = _build_target_alias_index(
        candidates=candidates,
        target_aliases_by_doc=target_aliases_by_doc,
    )
    target_alias_token_index = _build_target_alias_token_index(
        candidates=candidates,
        target_aliases_by_doc=target_aliases_by_doc,
    )
    source_text_tokens_by_doc = {
        index: tokens(node_lookup[index].text) for index in sources
    }
    if certificate_policy == LEGACY_ROLE_HINT_CERTIFICATE_POLICY:
        query_anchor_title_tokens = _query_mentioned_title_tokens(
            query=query,
            candidates=candidates,
            target_aliases_by_doc=target_aliases_by_doc,
        )
        demanded_roles = _role_terms(query) - query_anchor_title_tokens
        demanded_roles.update(_role_terms(" ".join(str(term) for term in extra_role_terms or ())))
    else:
        demanded_roles = set()

    certificates: List[EvidenceCertificate] = []
    seen = set()
    target_match_cache: Dict[str, Tuple[Tuple[int, str], ...]] = {}
    for source_index in sources:
        source = node_lookup[source_index]
        source_aliases = aliases_by_doc[source_index]
        for target_index in candidates:
            if int(source_index) == int(target_index):
                continue
            target = node_lookup[target_index]
            for target_alias in target_aliases_by_doc[target_index]:
                if not is_specific_title_alias(target_alias):
                    continue
                if not _phrase_occurs_in_tokens(
                    target_alias,
                    source_text_tokens_by_doc[source_index],
                ):
                    continue
                _append_certificate(
                    certificates,
                    seen,
                    _certificate(
                        source=source,
                        target=target,
                        certificate_type=TITLE_ALIAS_CERTIFICATE,
                        endpoint=target_alias,
                    ),
                )
        for certificate in _authorize_source_triples(
            source=source,
            nodes=node_lookup,
            candidates=candidates,
            source_aliases=source_aliases,
            target_aliases_by_doc=target_aliases_by_doc,
            target_alias_index=target_alias_index,
            target_alias_token_index=target_alias_token_index,
            demanded_roles=demanded_roles,
            target_match_cache=target_match_cache,
        ):
            _append_certificate(certificates, seen, certificate)

    return SourceTextCertificateGraph(
        candidate_doc_indices=candidates,
        nodes=candidate_nodes,
        certificates=tuple(certificates),
        metadata={
            **dict(SOURCE_TEXT_CERTIFICATE_GRAPH_CONTRACT),
            **certificate_policy_contract(certificate_policy),
            "certificate_source_doc_count": len(sources),
        },
    )


def _authorize_source_triples(
    *,
    source: EvidenceNode,
    nodes: Mapping[int, EvidenceNode],
    candidates: Sequence[int],
    source_aliases: Sequence[str],
    target_aliases_by_doc: Mapping[int, Sequence[str]],
    target_alias_index: Mapping[str, Sequence[Tuple[int, str]]],
    target_alias_token_index: Mapping[str, Sequence[Tuple[int, str]]],
    demanded_roles: set[str],
    target_match_cache: Dict[str, Tuple[Tuple[int, str], ...]],
) -> Tuple[EvidenceCertificate, ...]:
    certificates: List[EvidenceCertificate] = []
    source_text = str(source.text)
    for triple_index, triple in enumerate(source.triples):
        if len(triple) != 3:
            continue
        fact_origin = (int(source.doc_index), int(triple_index))
        subject, relation, obj = (str(triple[0]), str(triple[1]), str(triple[2]))
        subject_is_source = _matches_any_alias(subject, source_aliases)
        object_is_source = _matches_any_alias(obj, source_aliases)
        endpoint_matches = [
            ("subject", subject),
            ("object", obj),
        ]
        for endpoint_side, endpoint_text in endpoint_matches:
            for target_index, matched_target_alias in _matched_targets(
                endpoint_text=endpoint_text,
                source_doc_index=int(source.doc_index),
                nodes=nodes,
                candidates=candidates,
                target_aliases_by_doc=target_aliases_by_doc,
                target_alias_index=target_alias_index,
                target_alias_token_index=target_alias_token_index,
                target_match_cache=target_match_cache,
            ):
                target = nodes[int(target_index)]
                endpoint_sentence = _source_sentence_for_bindings(
                    source_text=source_text,
                    source_binding_texts=(),
                    target_binding_texts=(endpoint_text, matched_target_alias),
                )
                certificates.append(
                    _certificate(
                        source=source,
                        target=target,
                        certificate_type=ENDPOINT_TITLE_CERTIFICATE,
                        endpoint=matched_target_alias,
                        direction=endpoint_side,
                        triple=(subject, relation, obj),
                        fact_origin=fact_origin,
                        source_sentence=endpoint_sentence,
                        target_binding=matched_target_alias,
                    )
                )

                source_endpoint_bound = (
                    subject_is_source
                    if endpoint_side == "object"
                    else object_is_source
                )
                source_binding_texts = (
                    _source_binding_texts(
                        source_aliases=source_aliases,
                        source_endpoint_text=subject if endpoint_side == "object" else obj,
                    )
                    if source_endpoint_bound
                    else ()
                )
                transition_sentence = _source_sentence_for_bindings(
                    source_text=source_text,
                    source_binding_texts=source_binding_texts,
                    target_binding_texts=(endpoint_text, matched_target_alias),
                    relation_text=relation,
                    allow_document_scope_source_binding=source_endpoint_bound,
                )
                if source_endpoint_bound and transition_sentence:
                    source_binding = _first_occurring_binding(
                        transition_sentence,
                        source_binding_texts,
                    )
                    certificates.append(
                        _certificate(
                            source=source,
                            target=target,
                            certificate_type=SOURCE_TITLE_ENDPOINT_CERTIFICATE,
                            endpoint=matched_target_alias,
                            direction=(
                                "source_subject_to_object"
                                if endpoint_side == "object"
                                else "source_object_to_subject"
                            ),
                            triple=(subject, relation, obj),
                            fact_origin=fact_origin,
                            source_sentence=transition_sentence,
                            source_binding=source_binding,
                            target_binding=matched_target_alias,
                        )
                    )

                relation_roles = _matched_role_terms(relation, demanded_roles)
                if source_endpoint_bound and relation_roles and transition_sentence:
                    certificates.append(
                        _certificate(
                            source=source,
                            target=target,
                            certificate_type=RELATION_ROLE_CERTIFICATE,
                            endpoint=matched_target_alias,
                            roles=relation_roles,
                            direction=endpoint_side,
                            triple=(subject, relation, obj),
                            fact_origin=fact_origin,
                            source_sentence=transition_sentence,
                            source_binding=_first_occurring_binding(
                                transition_sentence,
                                source_binding_texts,
                            ),
                            target_binding=matched_target_alias,
                        )
                    )

                if (
                    endpoint_side == "object"
                    and subject_is_source
                    and _is_child_to_parent_relation(relation)
                    and transition_sentence
                ):
                    parent_roles = tuple(sorted(demanded_roles & _PARENT_ROLE_TERMS))
                    if parent_roles:
                        certificates.append(
                            _certificate(
                                source=source,
                                target=target,
                                certificate_type=DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE,
                                endpoint=matched_target_alias,
                                roles=parent_roles,
                                direction="child_subject_to_parent_object",
                                triple=(subject, relation, obj),
                                fact_origin=fact_origin,
                                source_sentence=transition_sentence,
                                source_binding=_first_occurring_binding(
                                    transition_sentence,
                                    source_binding_texts,
                                ),
                                target_binding=matched_target_alias,
                            )
                        )

                surface_endpoint_specific = (
                    len(tokens(endpoint_text)) >= 2
                    or normalize_text(endpoint_text) == normalize_text(matched_target_alias)
                )
                if source_endpoint_bound and surface_endpoint_specific:
                    surface_roles, surface_sentence = _surface_role_certificate(
                        source_text=source_text,
                        source_aliases=source_aliases,
                        target_text=endpoint_text,
                        demanded_roles=demanded_roles,
                    )
                    if surface_roles:
                        evidence_sentence = surface_sentence or transition_sentence
                        certificates.append(
                            _certificate(
                                source=source,
                                target=target,
                                certificate_type=SURFACE_ROLE_CERTIFICATE,
                                endpoint=matched_target_alias,
                                roles=surface_roles,
                                direction=endpoint_side,
                                triple=(subject, relation, obj),
                                fact_origin=fact_origin,
                                surface_evidence=evidence_sentence,
                                source_sentence=evidence_sentence,
                                source_binding=_first_occurring_binding(
                                    evidence_sentence,
                                    source_binding_texts,
                                ),
                                target_binding=matched_target_alias,
                            )
                        )
    return tuple(certificates)


def _matched_targets(
    *,
    endpoint_text: object,
    source_doc_index: int,
    nodes: Mapping[int, EvidenceNode],
    candidates: Sequence[int],
    target_aliases_by_doc: Mapping[int, Sequence[str]],
    target_alias_index: Mapping[str, Sequence[Tuple[int, str]]],
    target_alias_token_index: Mapping[str, Sequence[Tuple[int, str]]],
    target_match_cache: Dict[str, Tuple[Tuple[int, str], ...]],
) -> Tuple[Tuple[int, str], ...]:
    normalized_endpoint = normalize_text(endpoint_text)
    if not normalized_endpoint:
        return ()
    cached = target_match_cache.get(normalized_endpoint)
    if cached is None:
        matches: List[Tuple[int, str]] = []
        candidate_alias_entries = _rare_token_alias_entries(
            normalized_endpoint,
            target_alias_token_index,
        )
        exact_matches = target_alias_index.get(normalized_endpoint, ())
        if exact_matches:
            matches.extend((int(index), str(alias)) for index, alias in exact_matches)
        else:
            for target_index, alias in candidate_alias_entries:
                matched_alias = _matched_alias(normalized_endpoint, (alias,))
                if not matched_alias:
                    continue
                matches.append((int(target_index), matched_alias))
        matched_indices = {int(target_index) for target_index, _alias in matches}
        for target_index, _alias in candidate_alias_entries:
            if int(target_index) in matched_indices:
                continue
            target = nodes.get(int(target_index))
            if target is None:
                continue
            if not _endpoint_text_matches_target(
                normalized_endpoint=normalized_endpoint,
                target=target,
                target_aliases=target_aliases_by_doc.get(int(target_index), ()),
            ):
                continue
            matches.append((int(target_index), normalized_endpoint))
            matched_indices.add(int(target_index))
        cached = tuple(matches)
        target_match_cache[normalized_endpoint] = cached
    return tuple(
        (int(target_index), alias)
        for target_index, alias in cached
        if int(target_index) != int(source_doc_index) and int(target_index) in nodes
    )


def _build_target_alias_index(
    *,
    candidates: Sequence[int],
    target_aliases_by_doc: Mapping[int, Sequence[str]],
) -> Mapping[str, Tuple[Tuple[int, str], ...]]:
    index: Dict[str, List[Tuple[int, str]]] = {}
    for target_index in candidates:
        for alias in target_aliases_by_doc[int(target_index)]:
            if not alias:
                continue
            index.setdefault(str(alias), []).append((int(target_index), str(alias)))
    return {alias: tuple(values) for alias, values in index.items()}


def _build_target_alias_token_index(
    *,
    candidates: Sequence[int],
    target_aliases_by_doc: Mapping[int, Sequence[str]],
) -> Mapping[str, Tuple[Tuple[int, str], ...]]:
    index: Dict[str, List[Tuple[int, str]]] = {}
    for target_index in candidates:
        for alias in target_aliases_by_doc[int(target_index)]:
            for token in _rare_tokens(alias):
                index.setdefault(str(token), []).append((int(target_index), str(alias)))
    return {token: tuple(values) for token, values in index.items()}


def _rare_token_alias_entries(
    normalized_endpoint: str,
    target_alias_token_index: Mapping[str, Sequence[Tuple[int, str]]],
) -> Tuple[Tuple[int, str], ...]:
    candidate_lists = [
        (len(target_alias_token_index.get(token, ())), token, target_alias_token_index.get(token, ()))
        for token in _rare_tokens(normalized_endpoint)
        if target_alias_token_index.get(token)
    ]
    if not candidate_lists:
        return ()
    candidate_lists.sort(key=lambda item: (int(item[0]), str(item[1])))
    return tuple(candidate_lists[0][2])


def _rare_tokens(text: object) -> Tuple[str, ...]:
    return tuple(token for token in tokens(text) if len(token) >= 4)


def _endpoint_text_matches_target(
    *,
    normalized_endpoint: str,
    target: EvidenceNode,
    target_aliases: Sequence[str],
) -> bool:
    endpoint_tokens = tokens(normalized_endpoint)
    if len(endpoint_tokens) < 2:
        return False
    endpoint_rare_tokens = set(_rare_tokens(normalized_endpoint))
    title_rare_tokens = {
        token
        for alias in target_aliases
        for token in _rare_tokens(alias)
    }
    if not endpoint_rare_tokens & title_rare_tokens:
        return False
    return phrase_occurs(normalized_endpoint, target.text)


def _append_certificate(
    output: List[EvidenceCertificate],
    seen: set[Tuple[object, ...]],
    certificate: EvidenceCertificate,
) -> None:
    key = _certificate_key(certificate)
    if key in seen:
        return
    seen.add(key)
    output.append(certificate)


def _phrase_occurs_in_tokens(phrase: object, text_tokens: Sequence[str]) -> bool:
    phrase_tokens = tokens(phrase)
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    width = len(phrase_tokens)
    for start in range(0, len(text_tokens) - width + 1):
        if tuple(text_tokens[start : start + width]) == tuple(phrase_tokens):
            return True
    return False


def _certificate(
    *,
    source: EvidenceNode,
    target: EvidenceNode,
    certificate_type: str,
    endpoint: str,
    roles: Sequence[str] = (),
    direction: str = "",
    triple: Optional[Triple] = None,
    surface_evidence: str = "",
    source_sentence: str = "",
    source_binding: str = "",
    target_binding: str = "",
    fact_origin: Sequence[int] = (),
) -> EvidenceCertificate:
    normalized_triple = tuple(str(part) for part in (triple or ()))
    evidence_unit_id = ""
    if len(normalized_triple) == 3:
        evidence_unit_id = "::".join(normalize_text(part) for part in normalized_triple)
    return EvidenceCertificate(
        source_doc_index=int(source.doc_index),
        target_doc_index=int(target.doc_index),
        certificate_type=str(certificate_type),
        endpoint=normalize_text(endpoint),
        source_title=source.display_title,
        target_title=target.display_title,
        roles=tuple(sorted({str(role) for role in roles if str(role)})),
        direction=str(direction),
        triple=triple,
        surface_evidence=str(surface_evidence),
        source_sentence=str(source_sentence),
        source_binding=normalize_text(source_binding),
        target_binding=normalize_text(target_binding or endpoint),
        evidence_unit_id=evidence_unit_id,
        fact_origin=tuple(int(part) for part in tuple(fact_origin)[:2])
        if len(tuple(fact_origin)) >= 2
        else (),
    )


def _certificate_key(certificate: EvidenceCertificate) -> Tuple[object, ...]:
    return (
        int(certificate.source_doc_index),
        int(certificate.target_doc_index),
        str(certificate.certificate_type),
        str(certificate.endpoint),
        tuple(certificate.roles),
        str(certificate.direction),
        tuple(certificate.triple or ()),
        str(certificate.source_sentence),
        str(certificate.source_binding),
        str(certificate.target_binding),
        tuple(certificate.fact_origin),
    )


def _source_aliases(node: EvidenceNode) -> Tuple[str, ...]:
    aliases: List[str] = []
    for alias in title_aliases(node.display_title):
        if alias and alias not in aliases:
            aliases.append(alias)
    for triple in node.triples:
        if len(triple) != 3:
            continue
        relation = normalize_text(triple[1])
        if not any(hint in relation for hint in ("also known as", "known as", "titled")):
            continue
        for endpoint in (triple[0], triple[2]):
            normalized = normalize_text(endpoint)
            if normalized and normalized not in aliases:
                aliases.append(normalized)
    return tuple(aliases)


def _matches_any_alias(text: object, aliases: Sequence[str]) -> bool:
    return bool(_matched_alias(text, aliases))


def _source_binding_texts(
    *,
    source_aliases: Sequence[str],
    source_endpoint_text: object,
) -> Tuple[str, ...]:
    candidates: List[str] = []
    for value in (source_endpoint_text, *source_aliases):
        normalized = normalize_text(value)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
        alias_tokens = tokens(normalized)
        if len(alias_tokens) >= 2:
            head = alias_tokens[-1]
            if (
                len(head) >= 5
                and head not in _ROLE_STOP_TOKENS
                and head not in _SOURCE_HEAD_BINDING_STOP_TOKENS
                and head not in candidates
            ):
                candidates.append(head)
    return tuple(candidates)


def _source_sentence_for_bindings(
    *,
    source_text: str,
    source_binding_texts: Sequence[object],
    target_binding_texts: Sequence[object],
    relation_text: object = "",
    allow_document_scope_source_binding: bool = False,
) -> str:
    target_bindings = tuple(normalize_text(text) for text in target_binding_texts)
    target_bindings = tuple(binding for binding in target_bindings if binding)
    source_bindings = tuple(normalize_text(text) for text in source_binding_texts)
    source_bindings = tuple(binding for binding in source_bindings if binding)
    if not target_bindings:
        return ""

    passage_text = str(source_text).split("\n", 1)[1] if "\n" in str(source_text) else str(source_text)
    for sentence in _sentences(passage_text):
        if not any(phrase_occurs(binding, sentence) for binding in target_bindings):
            continue
        if source_bindings and any(
            phrase_occurs(binding, sentence) for binding in source_bindings
        ):
            return sentence.strip()
        if source_bindings and not allow_document_scope_source_binding:
            continue
        if source_bindings and not _relation_supported_in_sentence(
            relation_text,
            sentence,
        ):
            continue
        return sentence.strip()
    return ""


def _relation_supported_in_sentence(relation_text: object, sentence: object) -> bool:
    relation_tokens = [
        token
        for token in tokens(relation_text)
        if len(token) >= 3 and token not in _ROLE_STOP_TOKENS
    ]
    if not relation_tokens:
        return True
    sentence_tokens = set(tokens(sentence))
    return any(token in sentence_tokens for token in relation_tokens)


def _first_occurring_binding(sentence: object, bindings: Sequence[object]) -> str:
    for binding in bindings:
        normalized = normalize_text(binding)
        if normalized and phrase_occurs(normalized, sentence):
            return normalized
    return normalize_text(bindings[0]) if bindings else ""


def _matched_alias(text: object, aliases: Sequence[str]) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    for alias in aliases:
        if not alias:
            continue
        if normalized == alias:
            return alias
        if is_specific_title_alias(alias) and phrase_occurs(alias, normalized):
            return alias
        if (
            len(tokens(normalized)) >= 2
            and is_specific_title_alias(normalized)
            and phrase_occurs(normalized, alias)
        ):
            return alias
    return ""


def _role_terms(text: object) -> set[str]:
    role_terms: set[str] = set()
    for token in tokens(text):
        if token in _ROLE_STOP_TOKENS or len(token) < 3:
            continue
        role_terms.add(token)
        role_terms.update(_morphological_role_variants(token))
    return role_terms


def _query_mentioned_title_tokens(
    *,
    query: str,
    candidates: Sequence[int],
    target_aliases_by_doc: Mapping[int, Sequence[str]],
) -> set[str]:
    query_tokens = tokens(query)
    mentions: List[Tuple[int, int, int]] = []
    for doc_index in candidates:
        for alias in target_aliases_by_doc.get(int(doc_index), ()):
            if not is_specific_title_alias(alias):
                continue
            alias_tokens = tokens(alias)
            if not alias_tokens or len(alias_tokens) > len(query_tokens):
                continue
            width = len(alias_tokens)
            for start in range(0, len(query_tokens) - width + 1):
                end = start + width
                if tuple(query_tokens[start:end]) == alias_tokens:
                    mentions.append((start, end, width))

    output: set[str] = set()
    for start, end, width in mentions:
        if any(
            other_width > width
            and other_start <= start
            and end <= other_end
            for other_start, other_end, other_width in mentions
        ):
            continue
        output.update(query_tokens[start:end])
    return output


def _morphological_role_variants(token: str) -> set[str]:
    variants = {str(token)}
    variants.update(_ROLE_EQUIVALENTS.get(str(token), set()))
    if token.endswith("er") and len(token) > 4:
        stem = token[:-2]
        variants.update({stem, f"{stem}ed", f"{stem}ing"})
    if token.endswith("or") and len(token) > 4:
        stem = token[:-2]
        variants.update({stem, f"{stem}ed", f"{stem}ing"})
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        variants.update({stem, f"{stem}er", f"{stem}or", f"{stem}ing"})
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        variants.update({stem, f"{stem}ed", f"{stem}er", f"{stem}or"})
    return {variant for variant in variants if len(variant) >= 3}


def _matched_role_terms(text: object, demanded_roles: set[str]) -> Tuple[str, ...]:
    if not demanded_roles:
        return ()
    relation_terms = _role_terms(text)
    return tuple(sorted(relation_terms & demanded_roles))


def _is_child_to_parent_relation(relation: object) -> bool:
    normalized = normalize_text(relation)
    return any(hint in normalized for hint in _CHILD_TO_PARENT_RELATION_HINTS)


def _surface_role_certificate(
    *,
    source_text: str,
    source_aliases: Sequence[str],
    target_text: object,
    demanded_roles: set[str],
) -> Tuple[Tuple[str, ...], str]:
    if not demanded_roles:
        return (), ""
    target = normalize_text(target_text)
    if not target:
        return (), ""
    passage_text = str(source_text).split("\n", 1)[1] if "\n" in str(source_text) else str(source_text)
    for sentence in _sentences(passage_text):
        normalized_sentence = normalize_text(sentence)
        if not normalized_sentence:
            continue
        if not phrase_occurs(target, normalized_sentence):
            continue
        sentence_tokens = tokens(normalized_sentence)
        target_tokens = tokens(target)
        if not target_tokens:
            continue
        width = len(target_tokens)
        for start in range(0, len(sentence_tokens) - width + 1):
            end = start + width
            if tuple(sentence_tokens[start:end]) != target_tokens:
                continue
            window_start = max(0, start - 5)
            window_end = min(len(sentence_tokens), end + 5)
            local_context = " ".join(sentence_tokens[window_start:window_end])
            matched_roles = _matched_role_terms(local_context, demanded_roles)
            if matched_roles:
                return matched_roles, sentence.strip()
    return (), ""


def _sentences(text: str) -> Tuple[str, ...]:
    parts = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", str(text)) if sentence.strip()]
    merged: List[str] = []
    for part in parts:
        if merged and (_ends_with_initial(merged[-1]) or _is_initial_fragment(part)):
            merged[-1] = f"{merged[-1]} {part}".strip()
            continue
        merged.append(part)
    return tuple(merged)


def _ends_with_initial(text: object) -> bool:
    return bool(re.search(r"(?:^|\s)[A-Z]\.$", str(text).strip()))


def _is_initial_fragment(text: object) -> bool:
    return bool(re.fullmatch(r"[A-Z]\.", str(text).strip()))
