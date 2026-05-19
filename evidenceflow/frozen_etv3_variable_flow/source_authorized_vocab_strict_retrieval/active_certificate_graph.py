"""Query-conditioned active certificate graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .certificate_graph import (
    ENDPOINT_TITLE_CERTIFICATE,
    SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    TITLE_ALIAS_CERTIFICATE,
    EvidenceCertificate,
    EvidenceNode,
    SourceTextCertificateGraph,
)
from .normalize import normalize_text, tokens, unique_ints
from .query_conditioning import (
    QuerySignatures,
    extract_query_signatures,
    query_relation_overlap,
    token_stems_for_text,
)


ACTIVE_CERTIFICATE_GRAPH_CONTRACT: Mapping[str, bool | str] = {
    "active_graph": "query_conditioned_source_text_certificate_graph",
    "edge_activation": "query_surface_overlap_before_repair",
    "redundant_edge_suppression": "source_target_title_namespace",
    "uses_role_vocabulary": False,
    "uses_proposition_support_evidence": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
}

ACTIVE_CERTIFICATE_TYPES = frozenset(
    {
        TITLE_ALIAS_CERTIFICATE,
        ENDPOINT_TITLE_CERTIFICATE,
        SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    }
)


@dataclass(frozen=True)
class ActiveCertificateEdge:
    certificate: EvidenceCertificate
    activation_reasons: Tuple[str, ...]
    slot_key: str
    query_overlap_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class ActiveCertificateGraph:
    candidate_doc_indices: Tuple[int, ...]
    query_signatures: QuerySignatures
    active_edges: Tuple[ActiveCertificateEdge, ...]
    suppressed_edges: Tuple[ActiveCertificateEdge, ...]
    trace: Mapping[str, object]

    def outgoing(self, source_doc_index: int) -> Tuple[ActiveCertificateEdge, ...]:
        return tuple(
            edge
            for edge in self.active_edges
            if int(edge.certificate.source_doc_index) == int(source_doc_index)
        )


def activate_certificate_edges(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    certificate_graph: SourceTextCertificateGraph,
    candidate_doc_indices: Sequence[int],
    entry_doc_indices: Sequence[int],
    query_mentioned_doc_indices: Sequence[int] = (),
    require_source_entry: bool = True,
) -> ActiveCertificateGraph:
    """Filter source-text certificate edges into a query-active graph."""

    candidates = tuple(unique_ints(candidate_doc_indices))
    node_lookup = {int(node.doc_index): node for node in nodes}
    signatures = extract_query_signatures(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=candidates,
        query_mentioned_doc_indices=query_mentioned_doc_indices,
    )
    entry_set = {int(doc_index) for doc_index in unique_ints(entry_doc_indices)}
    active_candidates: List[ActiveCertificateEdge] = []
    for certificate in certificate_graph.certificates:
        if str(certificate.certificate_type) not in ACTIVE_CERTIFICATE_TYPES:
            continue
        source = int(certificate.source_doc_index)
        target = int(certificate.target_doc_index)
        if source == target:
            continue
        source_node = node_lookup.get(source)
        target_node = node_lookup.get(target)
        if source_node is None or target_node is None:
            continue
        reasons, overlap_tokens = _activation_reasons(
            certificate=certificate,
            source_node=source_node,
            target_node=target_node,
            signatures=signatures,
            source_is_entry=source in entry_set,
            require_source_entry=bool(require_source_entry),
        )
        if not reasons:
            continue
        active_candidates.append(
            ActiveCertificateEdge(
                certificate=certificate,
                activation_reasons=reasons,
                slot_key=_slot_key(certificate=certificate, overlap_tokens=overlap_tokens),
                query_overlap_tokens=overlap_tokens,
            )
        )

    active_edges, suppressed_edges = _suppress_siblings(
        active_candidates,
        candidate_doc_indices=candidates,
    )
    trace = {
        **dict(ACTIVE_CERTIFICATE_GRAPH_CONTRACT),
        "query_signatures": dict(signatures.trace),
        "candidate_doc_count": len(candidates),
        "raw_certificate_count": len(certificate_graph.certificates),
        "active_edge_count_before_suppression": len(active_candidates),
        "active_edge_count": len(active_edges),
        "require_source_entry": bool(require_source_entry),
        "suppressed_edge_count": len(suppressed_edges),
        "unique_active_target_count": len(
            {int(edge.certificate.target_doc_index) for edge in active_edges}
        ),
        "active_edge_preview": [_edge_trace(edge) for edge in active_edges[:12]],
        "suppressed_edge_preview": [_edge_trace(edge) for edge in suppressed_edges[:12]],
    }
    return ActiveCertificateGraph(
        candidate_doc_indices=candidates,
        query_signatures=signatures,
        active_edges=active_edges,
        suppressed_edges=suppressed_edges,
        trace=trace,
    )


def _activation_reasons(
    *,
    certificate: EvidenceCertificate,
    source_node: EvidenceNode,
    target_node: EvidenceNode,
    signatures: QuerySignatures,
    source_is_entry: bool,
    require_source_entry: bool,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if bool(require_source_entry) and not source_is_entry:
        return (), ()
    reasons: List[str] = []
    overlaps: List[str] = []
    triple = certificate.triple or ()
    relation_text = str(triple[1]) if len(triple) == 3 else ""
    endpoint_text = str(certificate.endpoint)
    relation_overlap = query_relation_overlap(signatures, relation_text)
    endpoint_overlap = query_relation_overlap(signatures, endpoint_text)
    target_title_overlap = query_relation_overlap(signatures, target_node.display_title)
    target_text_overlap = query_relation_overlap(signatures, target_node.text)
    target_fact_overlap = tuple(
        token
        for triple_item in target_node.triples
        for token in query_relation_overlap(signatures, " ".join(str(part) for part in triple_item))
    )
    source_title_stems = set(token_stems_for_text(source_node.display_title))
    query_relation_stems = set(signatures.relation_stems)

    if source_is_entry and (relation_overlap or endpoint_overlap or target_title_overlap):
        reasons.append("anchor_relation_or_endpoint_overlap")
    if target_text_overlap or target_fact_overlap:
        reasons.append("target_exposes_query_surface")
    if source_title_stems & query_relation_stems:
        reasons.append("source_title_exposes_query_surface")
    if str(certificate.certificate_type) == SOURCE_TITLE_ENDPOINT_CERTIFICATE and (
        relation_overlap or target_text_overlap or target_fact_overlap
    ):
        reasons.append("source_title_endpoint_query_active")

    overlaps.extend(relation_overlap)
    overlaps.extend(endpoint_overlap)
    overlaps.extend(target_title_overlap)
    overlaps.extend(target_text_overlap)
    overlaps.extend(target_fact_overlap)
    return tuple(_unique_strings(reasons)), tuple(_unique_strings(overlaps))


def _suppress_siblings(
    active_edges: Sequence[ActiveCertificateEdge],
    *,
    candidate_doc_indices: Sequence[int],
) -> Tuple[Tuple[ActiveCertificateEdge, ...], Tuple[ActiveCertificateEdge, ...]]:
    kept_by_slot: Dict[Tuple[int, str], ActiveCertificateEdge] = {}
    suppressed: List[ActiveCertificateEdge] = []
    for edge in sorted(
        active_edges,
        key=lambda item: (
            int(item.certificate.source_doc_index),
            str(item.slot_key),
            _edge_preference_key(item, candidate_doc_indices),
        ),
    ):
        key = (int(edge.certificate.source_doc_index), str(edge.slot_key))
        previous = kept_by_slot.get(key)
        if previous is None:
            kept_by_slot[key] = edge
            continue
        if _edge_preference_key(edge, candidate_doc_indices) < _edge_preference_key(
            previous,
            candidate_doc_indices,
        ):
            suppressed.append(previous)
            kept_by_slot[key] = edge
        else:
            suppressed.append(edge)

    kept, redundant = _deduplicate_source_target_namespaces(
        kept_by_slot.values(),
        candidate_doc_indices=candidate_doc_indices,
    )
    suppressed.extend(redundant)
    return kept, tuple(suppressed)


def _deduplicate_source_target_namespaces(
    active_edges: Sequence[ActiveCertificateEdge],
    *,
    candidate_doc_indices: Sequence[int],
) -> Tuple[Tuple[ActiveCertificateEdge, ...], Tuple[ActiveCertificateEdge, ...]]:
    kept_by_namespace: Dict[Tuple[int, str], ActiveCertificateEdge] = {}
    redundant: List[ActiveCertificateEdge] = []
    for edge in sorted(
        active_edges,
        key=lambda item: (
            int(item.certificate.source_doc_index),
            _target_namespace(item),
            _edge_preference_key(item, candidate_doc_indices),
            str(item.slot_key),
        ),
    ):
        key = (int(edge.certificate.source_doc_index), _target_namespace(edge))
        previous = kept_by_namespace.get(key)
        if previous is None:
            kept_by_namespace[key] = edge
            continue
        if _edge_preference_key(edge, candidate_doc_indices) < _edge_preference_key(
            previous,
            candidate_doc_indices,
        ):
            redundant.append(previous)
            kept_by_namespace[key] = edge
        else:
            redundant.append(edge)

    kept = tuple(
        sorted(
            kept_by_namespace.values(),
            key=lambda item: (
                _candidate_position(candidate_doc_indices, int(item.certificate.source_doc_index)),
                _candidate_position(candidate_doc_indices, int(item.certificate.target_doc_index)),
                _certificate_type_rank(str(item.certificate.certificate_type)),
                int(item.certificate.source_doc_index),
                int(item.certificate.target_doc_index),
            ),
        )
    )
    return kept, tuple(redundant)


def _edge_preference_key(
    edge: ActiveCertificateEdge,
    candidate_doc_indices: Sequence[int],
) -> Tuple[int, int, int]:
    return (
        _certificate_type_rank(str(edge.certificate.certificate_type)),
        _candidate_position(candidate_doc_indices, int(edge.certificate.target_doc_index)),
        int(edge.certificate.target_doc_index),
    )


def _certificate_type_rank(certificate_type: str) -> int:
    if certificate_type == SOURCE_TITLE_ENDPOINT_CERTIFICATE:
        return 0
    if certificate_type == ENDPOINT_TITLE_CERTIFICATE:
        return 1
    if certificate_type == TITLE_ALIAS_CERTIFICATE:
        return 2
    return 9


def _target_namespace(edge: ActiveCertificateEdge) -> str:
    certificate = edge.certificate
    title = normalize_text(certificate.target_title)
    if title:
        return title
    endpoint = normalize_text(certificate.endpoint)
    if endpoint:
        return endpoint
    return str(int(certificate.target_doc_index))


def _slot_key(*, certificate: EvidenceCertificate, overlap_tokens: Sequence[str]) -> str:
    if overlap_tokens:
        return "query:" + ",".join(overlap_tokens)
    triple = certificate.triple or ()
    if len(triple) == 3:
        return "triple:" + "|".join(tokens(" ".join(str(part) for part in triple)))
    return "endpoint:" + str(certificate.endpoint)


def _candidate_position(candidate_doc_indices: Sequence[int], doc_index: int) -> int:
    for position, candidate_doc_index in enumerate(candidate_doc_indices):
        if int(candidate_doc_index) == int(doc_index):
            return int(position)
    return 10**9


def _edge_trace(edge: ActiveCertificateEdge) -> Mapping[str, object]:
    certificate = edge.certificate
    return {
        "source_doc_index": int(certificate.source_doc_index),
        "target_doc_index": int(certificate.target_doc_index),
        "certificate_type": str(certificate.certificate_type),
        "endpoint": str(certificate.endpoint),
        "slot_key": str(edge.slot_key),
        "activation_reasons": tuple(edge.activation_reasons),
        "query_overlap_tokens": tuple(edge.query_overlap_tokens),
        "triple": list(certificate.triple or ()),
    }


def _unique_strings(values) -> Tuple[str, ...]:
    output = []
    seen = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return tuple(output)
