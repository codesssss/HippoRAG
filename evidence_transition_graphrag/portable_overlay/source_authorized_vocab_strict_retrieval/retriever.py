"""Standalone retriever over the source-text certificate graph."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .certificate_graph import (
    CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY,
    CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY,
    CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    CANONICAL_TRANSITION_CERTIFICATE_POLICY,
    DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE,
    ENDPOINT_TITLE_CERTIFICATE,
    LEGACY_ROLE_HINT_CERTIFICATE_POLICY,
    RELATION_ROLE_CERTIFICATE,
    SOURCE_TEXT_CERTIFICATE_GRAPH_CONTRACT,
    SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    SURFACE_ROLE_CERTIFICATE,
    EvidenceCertificate,
    EvidenceNode,
    SourceTextCertificateGraph,
    build_source_text_certificate_graph,
    certificate_policy_contract,
    normalize_certificate_policy,
)
from .normalize import unique_ints
from .query_mentions import query_mentioned_doc_indices


VOCAB_STRICT_RETRIEVER_CONTRACT: Mapping[str, bool | str] = {
    **dict(SOURCE_TEXT_CERTIFICATE_GRAPH_CONTRACT),
    "paper_facing_method_name": "source_authorized_vocab_strict_retrieval",
    "candidate_entrance": "local_candidate_pool",
    "graph_construction": "source_text_certificate_graph",
    "selection_policy": "source_certified_reader_context_assembly",
    "ranked_object": "reader_facing_context",
    "uses_evidence_set_objective": False,
    "uses_source_certified_reader_context_assembly": True,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_proposition_support_evidence": False,
    "uses_broad_location_blocklist": False,
}

SOURCE_PRIOR_TAIL_INSERTION_POLICY = "source_prior_preserving_tail_insertion"
SOURCE_PRIOR_CERTIFIED_CONTEXT_POLICY = "source_prior_preserving_certified_context"
SOURCE_PRIOR_TRANSITION_CLOSURE_POLICY = "source_prior_preserving_transition_closure"
SOURCE_CERTIFIED_PAIR_CONTEXT_POLICY = "source_certified_pair_context"

REPLACEMENT_CERTIFICATE_TYPES = frozenset(
    {
        RELATION_ROLE_CERTIFICATE,
        SURFACE_ROLE_CERTIFICATE,
        DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE,
    }
)

READER_ORDER_CERTIFICATE_TYPES = frozenset(
    {
        *REPLACEMENT_CERTIFICATE_TYPES,
        SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    }
)

BROAD_LOCATION_ROLE_TERMS = frozenset(
    {
        "border",
        "bordered",
        "bordering",
        "city",
        "country",
        "east",
        "eastern",
        "located",
        "location",
        "north",
        "northern",
        "region",
        "south",
        "southern",
        "west",
        "western",
    }
)


@dataclass(frozen=True)
class VocabStrictRetrievalResult:
    doc_indices: Tuple[int, ...]
    entry_doc_indices: Tuple[int, ...]
    certified_doc_indices: Tuple[int, ...]
    graph: SourceTextCertificateGraph
    trace: Mapping[str, object]


def source_authorized_vocab_strict_retrieve(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    candidate_pool_k: int = 200,
    max_hops: Optional[int] = None,
    extra_role_terms: Optional[Sequence[str]] = None,
    entry_doc_indices: Optional[Sequence[int]] = None,
    protected_doc_indices: Optional[Sequence[int]] = None,
    source_frontier_doc_indices: Optional[Sequence[int]] = None,
    prebuilt_graph: Optional[SourceTextCertificateGraph] = None,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    replacement_doc_indices: Optional[Sequence[int]] = None,
    allow_frontier_pair_insertion: bool = False,
    assembly_policy: str = SOURCE_PRIOR_TAIL_INSERTION_POLICY,
) -> VocabStrictRetrievalResult:
    """Assemble reader context with source-text certificate reachability.

    The candidate pool defines the local universe.  Ranking inside that
    universe is not a weighted fusion: the initial reader context is preserved
    unless a source-text certificate authorizes a conservative tail insertion.
    """

    top_k = max(int(top_k), 1)
    certificate_policy = normalize_certificate_policy(certificate_policy)
    candidate_pool_k = max(int(candidate_pool_k), top_k)
    pool = unique_ints(candidate_doc_indices)[:candidate_pool_k]
    replacement_doc_set = {int(index) for index in unique_ints(replacement_doc_indices or ())}
    canonical_policy = certificate_policy in {
        CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY,
        CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY,
        CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
        CANONICAL_TRANSITION_CERTIFICATE_POLICY,
    }
    assembly_policy = _normalize_assembly_policy(assembly_policy)
    transition_closure_admission = bool(
        canonical_policy
        and assembly_policy == SOURCE_PRIOR_TRANSITION_CLOSURE_POLICY
    )
    graph = (
        prebuilt_graph
        if prebuilt_graph is not None
        else build_source_text_certificate_graph(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=pool,
            extra_role_terms=extra_role_terms,
            certificate_policy=certificate_policy,
        )
    )
    node_lookup = {int(node.doc_index): node for node in nodes}
    entries = (
        unique_ints(entry_doc_indices)
        if entry_doc_indices is not None
        else _entry_doc_indices(query=str(query), nodes=node_lookup, candidate_doc_indices=pool)
    )
    source_frontier = set(unique_ints(source_frontier_doc_indices or entries))
    protected_docs = unique_ints(protected_doc_indices if protected_doc_indices is not None else entries)
    distance, parent_certificate = _certificate_reachable(
        graph=graph,
        entries=entries,
        max_hops=max_hops,
    )
    pool_rank = {int(doc_index): rank for rank, doc_index in enumerate(pool)}

    source_prior_docs = tuple(pool[:top_k])
    selected: List[int] = list(source_prior_docs)
    entry_set = {int(doc_index) for doc_index in entries}
    inserted_doc_indices = set()
    preclosed_replacement_source_role_keys = _selected_replacement_source_role_keys(
        selected=selected,
        graph=graph,
        certificate_policy=certificate_policy,
        replacement_doc_indices=replacement_doc_set,
    )
    used_replacement_source_doc_indices = set()
    if assembly_policy == SOURCE_CERTIFIED_PAIR_CONTEXT_POLICY:
        selected, inserted_doc_indices, used_replacement_source_doc_indices = (
            _source_certified_pair_context_selected(
                source_prior_docs=source_prior_docs,
                pool=pool,
                graph=graph,
                certificate_policy=certificate_policy,
                replacement_doc_indices=replacement_doc_set,
                top_k=top_k,
            )
        )
    else:
        reachable_non_entries = [
            int(doc_index)
            for doc_index in distance
            if int(doc_index) not in set(entries)
        ]
        reachable_non_entries.sort(
            key=lambda doc_index: (
                int(distance.get(int(doc_index), 10**9)),
                int(pool_rank.get(int(doc_index), 10**9)),
                int(doc_index),
            )
        )
        for doc_index in reachable_non_entries:
            if (
                canonical_policy
                and assembly_policy == SOURCE_PRIOR_TAIL_INSERTION_POLICY
                and inserted_doc_indices
            ):
                break
            if int(doc_index) in selected:
                continue
            parent = parent_certificate.get(int(doc_index))
            if parent is None or int(parent.source_doc_index) not in set(selected):
                if not (
                    bool(allow_frontier_pair_insertion)
                    and not canonical_policy
                    and parent is not None
                    and int(parent.source_doc_index) in source_frontier
                    and int(parent.source_doc_index) in entry_set
                    and _is_replacement_certificate(
                        parent,
                        certificate_policy=certificate_policy,
                        replacement_doc_indices=replacement_doc_set,
                    )
                    and int(parent.source_doc_index) not in used_replacement_source_doc_indices
                ):
                    continue
                victim_positions = _replacement_victim_positions(
                    selected=selected,
                    protected_doc_indices=protected_docs,
                    graph=graph,
                    count=2,
                    tail_only=False,
                )
                if len(victim_positions) < 2:
                    continue
                ordered_positions = tuple(sorted(victim_positions))
                selected[ordered_positions[0]] = int(parent.source_doc_index)
                selected[ordered_positions[1]] = int(doc_index)
                inserted_doc_indices.add(int(parent.source_doc_index))
                inserted_doc_indices.add(int(doc_index))
                used_replacement_source_doc_indices.add(int(parent.source_doc_index))
                continue
            if not _is_replacement_certificate(
                parent,
                certificate_policy=certificate_policy,
                replacement_doc_indices=replacement_doc_set,
                transition_closure_admission=transition_closure_admission,
            ):
                continue
            if int(parent.source_doc_index) in used_replacement_source_doc_indices:
                continue
            if int(parent.source_doc_index) in inserted_doc_indices and int(parent.source_doc_index) not in entry_set:
                continue
            if _certificate_has_preclosed_source_role(
                certificate=parent,
                preclosed_source_role_keys=preclosed_replacement_source_role_keys,
            ):
                continue
            victim_positions = _replacement_victim_positions(
                selected=selected,
                protected_doc_indices=[*protected_docs, int(parent.source_doc_index)],
                graph=graph,
                count=1,
                tail_only=bool(
                    canonical_policy and assembly_policy == SOURCE_PRIOR_TAIL_INSERTION_POLICY
                ),
            )
            if not victim_positions:
                continue
            selected[victim_positions[0]] = int(doc_index)
            inserted_doc_indices.add(int(doc_index))
            used_replacement_source_doc_indices.add(int(parent.source_doc_index))

    selected_tuple = _reader_order_selected(
        selected=selected[:top_k],
        graph=graph,
    )
    source_prior_set = {int(doc_index) for doc_index in source_prior_docs}
    source_prior_preserved_doc_indices = tuple(
        int(doc_index) for doc_index in selected_tuple if int(doc_index) in source_prior_set
    )
    selected_set = set(selected_tuple)
    closed_certificates = [
        certificate
        for certificate in graph.certificates
        if int(certificate.source_doc_index) in selected_set
        and int(certificate.target_doc_index) in selected_set
    ]
    certified_docs = tuple(
        doc_index for doc_index in selected_tuple if int(doc_index) in distance and int(doc_index) not in set(entries)
    )
    trace = {
        **dict(VOCAB_STRICT_RETRIEVER_CONTRACT),
        **certificate_policy_contract(certificate_policy),
        "uses_broad_location_blocklist": bool(
            certificate_policy == LEGACY_ROLE_HINT_CERTIFICATE_POLICY
        ),
        "top_k": top_k,
        "candidate_pool_k": candidate_pool_k,
        "max_hops": None if max_hops is None else max(int(max_hops), 0),
        "entry_doc_indices": entries,
        "protected_doc_indices": protected_docs,
        "source_frontier_doc_indices": tuple(sorted(source_frontier)),
        "source_prior_doc_indices": source_prior_docs,
        "source_prior_preserved_doc_indices": source_prior_preserved_doc_indices,
        "source_prior_preserved_doc_count": len(source_prior_preserved_doc_indices),
        "source_certified_insertion_target_doc_indices": tuple(sorted(replacement_doc_set)),
        "source_certified_inserted_doc_indices": tuple(sorted(inserted_doc_indices)),
        "source_certified_insertion_source_doc_indices": tuple(
            sorted(used_replacement_source_doc_indices)
        ),
        "source_certified_insertion_policy": assembly_policy,
        "uses_evidence_set_objective": False,
        "uses_source_certified_reader_context_assembly": True,
        "legacy_internal_operation": "replacement",
        "replacement_doc_indices": tuple(sorted(replacement_doc_set)),
        "canonical_replacement_tail_only": bool(
            canonical_policy and assembly_policy == SOURCE_PRIOR_TAIL_INSERTION_POLICY
        ),
        "canonical_replacement_max_inserted_docs": (
            1
            if canonical_policy and assembly_policy == SOURCE_PRIOR_TAIL_INSERTION_POLICY
            else None
        ),
        "transition_closure_admission": bool(transition_closure_admission),
        "allow_frontier_pair_insertion": bool(allow_frontier_pair_insertion),
        "inserted_doc_indices": tuple(sorted(inserted_doc_indices)),
        "preclosed_replacement_source_role_keys": tuple(
            sorted(f"{source_doc_index}:{role}" for source_doc_index, role in preclosed_replacement_source_role_keys)
        ),
        "replacement_source_doc_indices": tuple(sorted(used_replacement_source_doc_indices)),
        "selected_doc_indices": selected_tuple,
        "reader_order_policy": "source_before_selected_certificate_target",
        "certified_doc_indices": certified_docs,
        "candidate_doc_count": len(pool),
        "certificate_count": len(graph.certificates),
        "certificate_type_counts": dict(graph.certificate_type_counts()),
        "closed_certificate_count_topk": len(closed_certificates),
        "dense_fill_count": sum(1 for doc_index in selected_tuple if int(doc_index) not in distance),
        "parent_certificates": {
            int(doc_index): _certificate_trace(certificate)
            for doc_index, certificate in sorted(parent_certificate.items())
        },
    }
    return VocabStrictRetrievalResult(
        doc_indices=selected_tuple,
        entry_doc_indices=entries,
        certified_doc_indices=certified_docs,
        graph=graph,
        trace=trace,
    )


def _entry_doc_indices(
    *,
    query: str,
    nodes: Mapping[int, EvidenceNode],
    candidate_doc_indices: Sequence[int],
) -> Tuple[int, ...]:
    entries = list(
        query_mentioned_doc_indices(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=candidate_doc_indices,
        )
    )
    if entries:
        return unique_ints(entries)
    return unique_ints(candidate_doc_indices)[:1]


def _normalize_assembly_policy(policy: object) -> str:
    normalized = str(policy or "").strip().lower()
    if normalized in {
        SOURCE_CERTIFIED_PAIR_CONTEXT_POLICY,
        "source_certified_pair",
        "certified_pair_context",
        "certificate_pair_context",
    }:
        return SOURCE_CERTIFIED_PAIR_CONTEXT_POLICY
    if normalized in {
        SOURCE_PRIOR_TRANSITION_CLOSURE_POLICY,
        "source_prior_transition_closure",
        "transition_closure",
        "certified_transition_closure",
    }:
        return SOURCE_PRIOR_TRANSITION_CLOSURE_POLICY
    if normalized in {
        SOURCE_PRIOR_CERTIFIED_CONTEXT_POLICY,
        "source_prior_certified_context",
        "certified_context_insertion",
        "multi_certified_insertion",
    }:
        return SOURCE_PRIOR_CERTIFIED_CONTEXT_POLICY
    if normalized in {
        SOURCE_PRIOR_TAIL_INSERTION_POLICY,
        "source_prior_tail_insertion",
        "tail_insertion",
    }:
        return SOURCE_PRIOR_TAIL_INSERTION_POLICY
    return SOURCE_PRIOR_TAIL_INSERTION_POLICY


def _source_certified_pair_context_selected(
    *,
    source_prior_docs: Sequence[int],
    pool: Sequence[int],
    graph: SourceTextCertificateGraph,
    certificate_policy: str,
    replacement_doc_indices: set[int],
    top_k: int,
) -> Tuple[List[int], set[int], set[int]]:
    selected: List[int] = []
    inserted_doc_indices: set[int] = set()
    used_source_doc_indices: set[int] = set()
    for source_doc_index in unique_ints(source_prior_docs):
        _append_unique(selected, int(source_doc_index), limit=top_k)
        if len(selected) >= top_k:
            break
        target = _first_source_certified_target(
            source_doc_index=int(source_doc_index),
            graph=graph,
            certificate_policy=certificate_policy,
            replacement_doc_indices=replacement_doc_indices,
        )
        if target is None:
            continue
        before = set(selected)
        _append_unique(selected, int(target), limit=top_k)
        if int(target) in selected and int(target) not in before:
            inserted_doc_indices.add(int(target))
            used_source_doc_indices.add(int(source_doc_index))
        if len(selected) >= top_k:
            break
    for doc_index in unique_ints(source_prior_docs):
        _append_unique(selected, int(doc_index), limit=top_k)
    for doc_index in unique_ints(pool):
        _append_unique(selected, int(doc_index), limit=top_k)
    return selected[:top_k], inserted_doc_indices, used_source_doc_indices


def _first_source_certified_target(
    *,
    source_doc_index: int,
    graph: SourceTextCertificateGraph,
    certificate_policy: str,
    replacement_doc_indices: set[int],
) -> Optional[int]:
    outgoing = [
        certificate
        for certificate in graph.outgoing(int(source_doc_index))
        if _is_replacement_certificate(
            certificate,
            certificate_policy=certificate_policy,
            replacement_doc_indices=replacement_doc_indices,
        )
        and int(certificate.target_doc_index) != int(source_doc_index)
    ]
    if not outgoing:
        return None
    outgoing.sort(
        key=lambda certificate: (
            _candidate_position(graph.candidate_doc_indices, int(certificate.target_doc_index)),
            _certificate_replacement_rank(certificate),
            int(certificate.target_doc_index),
            str(certificate.certificate_type),
        )
    )
    return int(outgoing[0].target_doc_index)


def _certificate_reachable(
    *,
    graph: SourceTextCertificateGraph,
    entries: Sequence[int],
    max_hops: Optional[int],
) -> Tuple[Dict[int, int], Dict[int, EvidenceCertificate]]:
    hop_limit = None if max_hops is None else max(int(max_hops), 0)
    distance: Dict[int, int] = {int(entry): 0 for entry in entries}
    parent_certificate: Dict[int, EvidenceCertificate] = {}
    queue = deque(int(entry) for entry in entries)
    while queue:
        source = int(queue.popleft())
        source_distance = int(distance[source])
        if hop_limit is not None and source_distance >= hop_limit:
            continue
        outgoing = sorted(
            graph.outgoing(source),
            key=lambda certificate: (
                _candidate_position(graph.candidate_doc_indices, int(certificate.target_doc_index)),
                _certificate_replacement_rank(certificate),
                int(certificate.target_doc_index),
                str(certificate.certificate_type),
            ),
        )
        for certificate in outgoing:
            target = int(certificate.target_doc_index)
            if target in distance:
                continue
            distance[target] = source_distance + 1
            parent_certificate[target] = certificate
            queue.append(target)
    return distance, parent_certificate


def _candidate_position(candidate_doc_indices: Sequence[int], doc_index: int) -> int:
    for position, candidate_doc_index in enumerate(candidate_doc_indices):
        if int(candidate_doc_index) == int(doc_index):
            return int(position)
    return 10**9


def _certificate_replacement_rank(certificate: EvidenceCertificate) -> int:
    certificate_type = str(certificate.certificate_type)
    if certificate_type in REPLACEMENT_CERTIFICATE_TYPES:
        return 0
    if certificate_type == SOURCE_TITLE_ENDPOINT_CERTIFICATE:
        return 1
    return 2


def _is_replacement_certificate(
    certificate: EvidenceCertificate,
    *,
    certificate_policy: str,
    replacement_doc_indices: set[int],
    transition_closure_admission: bool = False,
) -> bool:
    certificate_policy = normalize_certificate_policy(certificate_policy)
    certificate_type = str(certificate.certificate_type)
    if certificate_policy in {
        CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY,
        CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY,
        CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
        CANONICAL_TRANSITION_CERTIFICATE_POLICY,
    }:
        if bool(transition_closure_admission):
            return (
                certificate_type == SOURCE_TITLE_ENDPOINT_CERTIFICATE
                or (
                    certificate_type == ENDPOINT_TITLE_CERTIFICATE
                    and int(certificate.target_doc_index) in replacement_doc_indices
                )
            )
        return (
            certificate_type
            in {
                ENDPOINT_TITLE_CERTIFICATE,
                SOURCE_TITLE_ENDPOINT_CERTIFICATE,
            }
            and int(certificate.target_doc_index) in replacement_doc_indices
        )
    if certificate_type not in REPLACEMENT_CERTIFICATE_TYPES:
        return False
    roles = {str(role) for role in certificate.roles}
    if (
        certificate_type in {RELATION_ROLE_CERTIFICATE, SURFACE_ROLE_CERTIFICATE}
        and roles
        and roles.issubset(BROAD_LOCATION_ROLE_TERMS)
    ):
        return False
    return True


def _append_unique(output: List[int], doc_index: int, *, limit: int) -> None:
    if len(output) >= int(limit):
        return
    if int(doc_index) in output:
        return
    output.append(int(doc_index))


def _replacement_victim_positions(
    *,
    selected: Sequence[int],
    protected_doc_indices: Sequence[int],
    graph: SourceTextCertificateGraph,
    count: int,
    tail_only: bool = False,
) -> Tuple[int, ...]:
    protected_set = {int(index) for index in protected_doc_indices}
    selected_set = {int(doc_index) for doc_index in selected}
    certified_participants = {
        int(certificate.source_doc_index)
        for certificate in graph.certificates
        if int(certificate.source_doc_index) in selected_set
        and int(certificate.target_doc_index) in selected_set
    }
    certified_participants.update(
        int(certificate.target_doc_index)
        for certificate in graph.certificates
        if int(certificate.source_doc_index) in selected_set
        and int(certificate.target_doc_index) in selected_set
    )
    positions: List[int] = []
    positions_to_consider = range(len(selected) - 1, -1, -1)
    if tail_only:
        positions_to_consider = range(len(selected) - 1, len(selected) - 2, -1)
    for position in positions_to_consider:
        doc_index = int(selected[position])
        if doc_index in protected_set:
            continue
        if doc_index in certified_participants:
            continue
        positions.append(int(position))
        if len(positions) >= max(int(count), 0):
            break
    return tuple(positions)


def _selected_replacement_source_role_keys(
    *,
    selected: Sequence[int],
    graph: SourceTextCertificateGraph,
    certificate_policy: str,
    replacement_doc_indices: set[int],
) -> Tuple[Tuple[int, str], ...]:
    if normalize_certificate_policy(certificate_policy) != LEGACY_ROLE_HINT_CERTIFICATE_POLICY:
        return ()
    selected_set = {int(doc_index) for doc_index in selected}
    return tuple(
        sorted(
            {
                (int(certificate.source_doc_index), str(role))
                for certificate in graph.certificates
                if int(certificate.source_doc_index) in selected_set
                and int(certificate.target_doc_index) in selected_set
                and _is_replacement_certificate(
                    certificate,
                    certificate_policy=certificate_policy,
                    replacement_doc_indices=replacement_doc_indices,
                )
                and str(certificate.certificate_type)
                in {RELATION_ROLE_CERTIFICATE, DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE}
                for role in certificate.roles
            }
        )
    )


def _certificate_has_preclosed_source_role(
    *,
    certificate: EvidenceCertificate,
    preclosed_source_role_keys: Sequence[Tuple[int, str]],
) -> bool:
    preclosed_keys = set(preclosed_source_role_keys)
    return any(
        (int(certificate.source_doc_index), str(role)) in preclosed_keys
        for role in certificate.roles
    )


def _reader_order_selected(
    *,
    selected: Sequence[int],
    graph: SourceTextCertificateGraph,
) -> Tuple[int, ...]:
    """Place selected certificate targets immediately after their source.

    This changes only reader-facing order, not the selected context.  Weak
    `source_title_endpoint` edges are allowed here because they are search/order
    certificates, not admission certificates.
    """

    selected_tuple = unique_ints(selected)
    selected_set = {int(doc_index) for doc_index in selected_tuple}
    children_by_source: Dict[int, List[int]] = {}
    parent_by_target: Dict[int, int] = {}
    for certificate in sorted(
        graph.certificates,
        key=lambda item: (
            _candidate_position(graph.candidate_doc_indices, int(item.source_doc_index)),
            _candidate_position(graph.candidate_doc_indices, int(item.target_doc_index)),
            _certificate_reader_order_rank(item),
            int(item.source_doc_index),
            int(item.target_doc_index),
        ),
    ):
        source = int(certificate.source_doc_index)
        target = int(certificate.target_doc_index)
        if source not in selected_set or target not in selected_set:
            continue
        if str(certificate.certificate_type) not in READER_ORDER_CERTIFICATE_TYPES:
            continue
        parent_by_target.setdefault(target, source)
        children_by_source.setdefault(source, [])
        if target not in children_by_source[source]:
            children_by_source[source].append(target)

    output: List[int] = []
    for doc_index in selected_tuple:
        doc_index = int(doc_index)
        parent = parent_by_target.get(doc_index)
        if parent in selected_set and parent not in output:
            continue
        _append_unique(output, doc_index, limit=len(selected_tuple))
        for child in children_by_source.get(doc_index, []):
            _append_unique(output, child, limit=len(selected_tuple))

    for doc_index in selected_tuple:
        _append_unique(output, int(doc_index), limit=len(selected_tuple))
    return tuple(output[: len(selected_tuple)])


def _certificate_reader_order_rank(certificate: EvidenceCertificate) -> int:
    if str(certificate.certificate_type) in REPLACEMENT_CERTIFICATE_TYPES:
        return 0
    if str(certificate.certificate_type) == SOURCE_TITLE_ENDPOINT_CERTIFICATE:
        return 1
    return 2


def _certificate_trace(certificate: EvidenceCertificate) -> Mapping[str, object]:
    return {
        "source_doc_index": int(certificate.source_doc_index),
        "target_doc_index": int(certificate.target_doc_index),
        "certificate_type": str(certificate.certificate_type),
        "endpoint": str(certificate.endpoint),
        "roles": list(certificate.roles),
        "direction": str(certificate.direction),
        "triple": list(certificate.triple or ()),
    }
