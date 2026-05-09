"""Graph-native evidence-set selection for source-text GraphRAG."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .certificate_graph import (
    CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY,
    CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY,
    CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    CANONICAL_TRANSITION_CERTIFICATE_POLICY,
    DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE,
    LEGACY_ROLE_HINT_CERTIFICATE_POLICY,
    RELATION_ROLE_CERTIFICATE,
    SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    SURFACE_ROLE_CERTIFICATE,
    EvidenceCertificate,
    SourceTextCertificateGraph,
    normalize_certificate_policy,
)
from .normalize import unique_ints
from .query_demand import (
    QueryDemandGraph,
    build_query_demand_graph,
    demand_coverage_for_docs,
)


GRAPH_NATIVE_EVIDENCE_SET_CONTRACT: Mapping[str, bool | str] = {
    "selection_policy": "graph_native_source_authorized_evidence_set_closure",
    "ranked_object": "closed_reader_facing_evidence_set",
    "dense_topk_is_final_answer_default": False,
    "uses_replacement_policy": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_learned_reranker": False,
    "uses_query_relation_surface_alignment": True,
}


LEGACY_ADMISSION_CERTIFICATE_TYPES = frozenset(
    {
        RELATION_ROLE_CERTIFICATE,
        SURFACE_ROLE_CERTIFICATE,
        DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE,
    }
)

READER_ORDER_CERTIFICATE_TYPES = frozenset(
    {
        SOURCE_TITLE_ENDPOINT_CERTIFICATE,
        *LEGACY_ADMISSION_CERTIFICATE_TYPES,
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
class EvidenceSetCandidate:
    doc_indices: Tuple[int, ...]
    root_doc_index: int
    graph_doc_indices: Tuple[int, ...]
    dense_fill_doc_indices: Tuple[int, ...]
    closed_certificate_count: int
    admissible_closed_certificate_count: int
    certified_doc_count: int
    source_prior_doc_count: int
    demand_coverage_count: int
    demand_covering_doc_count: int
    query_aligned_certificate_count: int
    key: Tuple[int, ...]
    trace: Mapping[str, object]


@dataclass(frozen=True)
class EvidenceSetSelectionResult:
    doc_indices: Tuple[int, ...]
    entry_doc_indices: Tuple[int, ...]
    graph_doc_indices: Tuple[int, ...]
    dense_fill_doc_indices: Tuple[int, ...]
    certified_doc_indices: Tuple[int, ...]
    selected_candidate: EvidenceSetCandidate
    candidate_sets: Tuple[EvidenceSetCandidate, ...]
    trace: Mapping[str, object]


def select_source_authorized_evidence_set(
    *,
    query: str = "",
    graph: SourceTextCertificateGraph,
    candidate_doc_indices: Sequence[int],
    entry_doc_indices: Sequence[int],
    root_candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    source_bound_doc_indices: Sequence[int] = (),
    source_frontier_doc_indices: Sequence[int] = (),
) -> EvidenceSetSelectionResult:
    """Select final top-k evidence as a graph-closed set, not by replacement."""

    certificate_policy = normalize_certificate_policy(certificate_policy)
    top_k = max(int(top_k), 1)
    pool = tuple(unique_ints(candidate_doc_indices))
    pool_rank = {int(doc_index): rank for rank, doc_index in enumerate(pool)}
    entries = tuple(unique_ints(entry_doc_indices or pool[:1]))
    roots = tuple(unique_ints([*root_candidate_doc_indices, *entries]) or list(pool[:1]))
    source_frontier = tuple(unique_ints([*source_frontier_doc_indices, *entries]))
    source_bound_set = {int(doc_index) for doc_index in unique_ints(source_bound_doc_indices)}
    query_demand = build_query_demand_graph(str(query))
    query_relation_tokens = _relation_alignment_tokens(str(query))

    candidates: List[EvidenceSetCandidate] = [
        _candidate_from_docs(
            doc_indices=pool[:top_k],
            root_doc_index=int(pool[0]) if pool else -1,
            graph_doc_indices=(),
            graph=graph,
            pool=pool,
            entries=entries,
            source_frontier_doc_indices=source_frontier,
            certificate_policy=certificate_policy,
            source_bound_doc_indices=source_bound_set,
            query_demand=query_demand,
            query_relation_tokens=query_relation_tokens,
            candidate_name="dense_fill_set",
        )
    ]
    for root_doc_index in roots:
        if int(root_doc_index) not in pool_rank:
            continue
        candidates.append(
            _build_component_candidate(
                root_doc_index=int(root_doc_index),
                graph=graph,
                pool=pool,
                entries=entries,
                source_frontier_doc_indices=source_frontier,
                top_k=top_k,
                certificate_policy=certificate_policy,
                source_bound_doc_indices=source_bound_set,
                query_demand=query_demand,
                query_relation_tokens=query_relation_tokens,
            )
        )
    max_frontier_width = min(len(roots), top_k)
    for frontier_width in range(2, max_frontier_width + 1):
        frontier_roots = tuple(root for root in roots[:frontier_width] if int(root) in pool_rank)
        if len(frontier_roots) < 2:
            continue
        candidates.append(
            _build_frontier_candidate(
                root_doc_indices=frontier_roots,
                graph=graph,
                pool=pool,
                entries=entries,
                source_frontier_doc_indices=source_frontier,
                top_k=top_k,
                certificate_policy=certificate_policy,
                source_bound_doc_indices=source_bound_set,
                query_demand=query_demand,
                query_relation_tokens=query_relation_tokens,
            )
        )

    selected_candidate = max(candidates, key=lambda candidate: candidate.key)
    ordered_docs = _reader_order_selected(
        selected_candidate.doc_indices,
        graph=graph,
    )
    selected_set = set(ordered_docs)
    certified_docs = tuple(
        doc_index
        for doc_index in ordered_docs
        if any(
            int(certificate.target_doc_index) == int(doc_index)
            and int(certificate.source_doc_index) in selected_set
            and _is_admissible_certificate(
                certificate,
                certificate_policy=certificate_policy,
                source_bound_doc_indices=source_bound_set,
                query_relation_tokens=query_relation_tokens,
            )
            for certificate in graph.certificates
        )
    )
    trace = {
        **dict(GRAPH_NATIVE_EVIDENCE_SET_CONTRACT),
        "top_k": top_k,
        "candidate_doc_count": len(pool),
        "entry_doc_indices": entries,
        "source_frontier_doc_indices": source_frontier,
        "query_demand": dict(query_demand.trace),
        "root_candidate_doc_indices": roots,
        "selected_root_doc_index": int(selected_candidate.root_doc_index),
        "selected_doc_indices": ordered_docs,
        "graph_doc_indices": tuple(
            doc_index for doc_index in ordered_docs if doc_index in set(selected_candidate.graph_doc_indices)
        ),
        "dense_fill_doc_indices": tuple(
            doc_index for doc_index in ordered_docs if doc_index in set(selected_candidate.dense_fill_doc_indices)
        ),
        "closed_certificate_count_topk": _closed_certificate_count(graph, ordered_docs),
        "admissible_closed_certificate_count_topk": _admissible_closed_certificate_count(
            graph,
            ordered_docs,
            certificate_policy=certificate_policy,
            source_bound_doc_indices=source_bound_set,
            query_relation_tokens=query_relation_tokens,
        ),
        "certified_doc_indices": certified_docs,
        "candidate_set_count": len(candidates),
        "candidate_keys": [candidate.key for candidate in candidates[:8]],
        "candidate_summaries": [dict(candidate.trace) for candidate in candidates[:8]],
    }
    return EvidenceSetSelectionResult(
        doc_indices=ordered_docs,
        entry_doc_indices=entries,
        graph_doc_indices=tuple(
            doc_index for doc_index in ordered_docs if doc_index in set(selected_candidate.graph_doc_indices)
        ),
        dense_fill_doc_indices=tuple(
            doc_index for doc_index in ordered_docs if doc_index in set(selected_candidate.dense_fill_doc_indices)
        ),
        certified_doc_indices=certified_docs,
        selected_candidate=selected_candidate,
        candidate_sets=tuple(candidates),
        trace=trace,
    )


def _build_component_candidate(
    *,
    root_doc_index: int,
    graph: SourceTextCertificateGraph,
    pool: Sequence[int],
    entries: Sequence[int],
    source_frontier_doc_indices: Sequence[int],
    top_k: int,
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_demand: QueryDemandGraph,
    query_relation_tokens: set[str],
) -> EvidenceSetCandidate:
    return _build_closure_candidate(
        seed_doc_indices=unique_ints([*entries, int(root_doc_index)]),
        root_doc_index=root_doc_index,
        graph=graph,
        pool=pool,
        entries=entries,
        source_frontier_doc_indices=source_frontier_doc_indices,
        top_k=top_k,
        certificate_policy=certificate_policy,
        source_bound_doc_indices=source_bound_doc_indices,
        query_demand=query_demand,
        query_relation_tokens=query_relation_tokens,
        candidate_name="source_authorized_component",
    )


def _build_frontier_candidate(
    *,
    root_doc_indices: Sequence[int],
    graph: SourceTextCertificateGraph,
    pool: Sequence[int],
    entries: Sequence[int],
    source_frontier_doc_indices: Sequence[int],
    top_k: int,
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_demand: QueryDemandGraph,
    query_relation_tokens: set[str],
) -> EvidenceSetCandidate:
    frontier_roots = tuple(unique_ints(root_doc_indices))
    return _build_closure_candidate(
        seed_doc_indices=unique_ints([*entries, *frontier_roots]),
        root_doc_index=int(frontier_roots[0]) if frontier_roots else -1,
        graph=graph,
        pool=pool,
        entries=entries,
        source_frontier_doc_indices=source_frontier_doc_indices,
        top_k=top_k,
        certificate_policy=certificate_policy,
        source_bound_doc_indices=source_bound_doc_indices,
        query_demand=query_demand,
        query_relation_tokens=query_relation_tokens,
        candidate_name="source_authorized_frontier_set",
    )


def _build_closure_candidate(
    *,
    seed_doc_indices: Sequence[int],
    root_doc_index: int,
    graph: SourceTextCertificateGraph,
    pool: Sequence[int],
    entries: Sequence[int],
    source_frontier_doc_indices: Sequence[int],
    top_k: int,
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_demand: QueryDemandGraph,
    query_relation_tokens: set[str],
    candidate_name: str,
) -> EvidenceSetCandidate:
    selected: List[int] = list(unique_ints(seed_doc_indices))[:top_k]
    graph_docs: set[int] = set()
    used_source_roles: set[Tuple[int, str]] = set()
    used_fact_origins: set[Tuple[int, int]] = set()
    queue = deque(selected)
    while queue:
        source_doc_index = int(queue.popleft())
        outgoing = sorted(
            graph.outgoing(source_doc_index),
            key=lambda certificate: (
                _certificate_rank(
                    certificate,
                    certificate_policy=certificate_policy,
                    source_bound_doc_indices=source_bound_doc_indices,
                    query_relation_tokens=query_relation_tokens,
                ),
                _certificate_query_alignment_rank(
                    certificate,
                    query_relation_tokens=query_relation_tokens,
                ),
                _candidate_position(pool, int(certificate.target_doc_index)),
                int(certificate.target_doc_index),
                str(certificate.certificate_type),
            ),
        )
        for certificate in outgoing:
            if not _is_admissible_certificate(
                certificate,
                certificate_policy=certificate_policy,
                source_bound_doc_indices=source_bound_doc_indices,
                query_relation_tokens=query_relation_tokens,
            ):
                continue
            if _forks_existing_source_role(certificate, used_source_roles):
                continue
            if _forks_existing_fact_origin(
                certificate,
                used_fact_origins=used_fact_origins,
                certificate_policy=certificate_policy,
            ):
                continue
            target_doc_index = int(certificate.target_doc_index)
            if target_doc_index in selected:
                graph_docs.add(source_doc_index)
                graph_docs.add(target_doc_index)
                _mark_source_roles(certificate, used_source_roles)
                _mark_fact_origin(certificate, used_fact_origins)
                continue
            if len(selected) >= top_k:
                continue
            if _candidate_position(pool, target_doc_index) >= 10**9:
                continue
            selected.append(target_doc_index)
            graph_docs.add(source_doc_index)
            graph_docs.add(target_doc_index)
            _mark_source_roles(certificate, used_source_roles)
            _mark_fact_origin(certificate, used_fact_origins)
            queue.append(target_doc_index)

    selected = _fill_with_dense(selected, pool=pool, top_k=top_k)
    return _candidate_from_docs(
        doc_indices=selected,
        root_doc_index=root_doc_index,
        graph_doc_indices=tuple(doc_index for doc_index in selected if doc_index in graph_docs),
        graph=graph,
        pool=pool,
        entries=entries,
        source_frontier_doc_indices=source_frontier_doc_indices,
        certificate_policy=certificate_policy,
        source_bound_doc_indices=source_bound_doc_indices,
        query_demand=query_demand,
        query_relation_tokens=query_relation_tokens,
        candidate_name=candidate_name,
    )


def _candidate_from_docs(
    *,
    doc_indices: Sequence[int],
    root_doc_index: int,
    graph_doc_indices: Sequence[int],
    graph: SourceTextCertificateGraph,
    pool: Sequence[int],
    entries: Sequence[int],
    source_frontier_doc_indices: Sequence[int],
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_demand: QueryDemandGraph,
    query_relation_tokens: set[str],
    candidate_name: str,
) -> EvidenceSetCandidate:
    docs = tuple(unique_ints(doc_indices))
    graph_docs = tuple(
        unique_ints([*graph_doc_indices, *_closed_admissible_graph_doc_indices(
            graph,
            docs,
            certificate_policy=certificate_policy,
            source_bound_doc_indices=source_bound_doc_indices,
            query_relation_tokens=query_relation_tokens,
        )])
    )
    dense_fill_docs = tuple(doc_index for doc_index in docs if int(doc_index) not in set(graph_docs))
    closed_count = _closed_certificate_count(graph, docs)
    admissible_closed_count = _admissible_closed_certificate_count(
        graph,
        docs,
        certificate_policy=certificate_policy,
        source_bound_doc_indices=source_bound_doc_indices,
        query_relation_tokens=query_relation_tokens,
    )
    certified_count = _certified_doc_count(
        graph,
        docs,
        certificate_policy=certificate_policy,
        source_bound_doc_indices=source_bound_doc_indices,
        query_relation_tokens=query_relation_tokens,
    )
    query_aligned_count = _query_aligned_closed_certificate_count(
        graph,
        docs,
        query_relation_tokens=query_relation_tokens,
        certificate_policy=certificate_policy,
        source_bound_doc_indices=source_bound_doc_indices,
    )
    entry_count = len(set(docs) & {int(doc_index) for doc_index in entries})
    source_frontier_count = len(
        set(docs) & {int(doc_index) for doc_index in source_frontier_doc_indices}
    )
    node_lookup = {int(node.doc_index): node for node in graph.nodes}
    demand_coverage = demand_coverage_for_docs(
        demand=query_demand,
        nodes=node_lookup,
        doc_indices=docs,
    )
    source_prior_count = len(set(docs) & set(pool[: len(docs)]))
    dense_rank_sum = sum(_candidate_position(pool, doc_index) for doc_index in docs)
    graph_doc_count = len(graph_docs)
    dense_fill_count = len(dense_fill_docs)
    source_frontier_connected = bool(
        set(graph_docs) & {int(doc_index) for doc_index in source_frontier_doc_indices}
    )
    has_graph_evidence = 1 if certified_count > 0 and source_frontier_connected else 0
    effective_certified_count = certified_count if has_graph_evidence else 0
    effective_graph_doc_count = graph_doc_count if has_graph_evidence else 0
    effective_admissible_count = admissible_closed_count if has_graph_evidence else 0
    effective_query_aligned_count = query_aligned_count if has_graph_evidence else 0
    effective_demand_covering_doc_count = (
        demand_coverage.covering_doc_count if has_graph_evidence else 0
    )
    effective_demand_coverage_count = demand_coverage.coverage_count if has_graph_evidence else 0
    key = (
        int(has_graph_evidence),
        int(effective_query_aligned_count),
        int(effective_certified_count),
        int(effective_graph_doc_count),
        int(source_prior_count),
        int(effective_demand_covering_doc_count),
        int(effective_demand_coverage_count),
        -int(dense_rank_sum),
        int(effective_admissible_count),
        int(entry_count),
        int(source_frontier_count),
        -int(dense_fill_count),
        -int(root_doc_index),
    )
    trace = {
        "candidate_name": str(candidate_name),
        "root_doc_index": int(root_doc_index),
        "doc_indices": docs,
        "graph_doc_indices": graph_docs,
        "dense_fill_doc_indices": dense_fill_docs,
        "closed_certificate_count": int(closed_count),
        "admissible_closed_certificate_count": int(admissible_closed_count),
        "certified_doc_count": int(certified_count),
        "query_aligned_certificate_count": int(query_aligned_count),
        "effective_query_aligned_certificate_count": int(effective_query_aligned_count),
        "demand_covered_tokens": demand_coverage.covered_tokens,
        "demand_coverage_count": int(demand_coverage.coverage_count),
        "demand_covering_doc_indices": demand_coverage.covering_doc_indices,
        "demand_covering_doc_count": int(demand_coverage.covering_doc_count),
        "effective_demand_coverage_count": int(effective_demand_coverage_count),
        "effective_demand_covering_doc_count": int(effective_demand_covering_doc_count),
        "source_frontier_connected_graph_evidence": bool(has_graph_evidence),
        "effective_certified_doc_count": int(effective_certified_count),
        "effective_graph_doc_count": int(effective_graph_doc_count),
        "effective_admissible_closed_certificate_count": int(effective_admissible_count),
        "entry_count": int(entry_count),
        "source_frontier_count": int(source_frontier_count),
        "source_prior_doc_count": int(source_prior_count),
        "dense_rank_sum": int(dense_rank_sum),
        "key": key,
    }
    return EvidenceSetCandidate(
        doc_indices=docs,
        root_doc_index=int(root_doc_index),
        graph_doc_indices=graph_docs,
        dense_fill_doc_indices=dense_fill_docs,
        closed_certificate_count=int(closed_count),
        admissible_closed_certificate_count=int(admissible_closed_count),
        certified_doc_count=int(certified_count),
        source_prior_doc_count=int(source_prior_count),
        demand_coverage_count=int(demand_coverage.coverage_count),
        demand_covering_doc_count=int(demand_coverage.covering_doc_count),
        query_aligned_certificate_count=int(query_aligned_count),
        key=key,
        trace=trace,
    )


def _fill_with_dense(selected: Sequence[int], *, pool: Sequence[int], top_k: int) -> List[int]:
    output = list(unique_ints(selected))
    for doc_index in pool:
        if len(output) >= top_k:
            break
        if int(doc_index) in output:
            continue
        output.append(int(doc_index))
    return output[:top_k]


def _closed_certificate_count(graph: SourceTextCertificateGraph, doc_indices: Sequence[int]) -> int:
    selected_set = {int(doc_index) for doc_index in doc_indices}
    return sum(
        1
        for certificate in graph.certificates
        if int(certificate.source_doc_index) in selected_set
        and int(certificate.target_doc_index) in selected_set
    )


def _query_aligned_closed_certificate_count(
    graph: SourceTextCertificateGraph,
    doc_indices: Sequence[int],
    *,
    query_relation_tokens: set[str],
    certificate_policy: str,
    source_bound_doc_indices: set[int],
) -> int:
    if not query_relation_tokens:
        return 0
    selected_set = {int(doc_index) for doc_index in doc_indices}
    return sum(
        1
        for certificate in graph.certificates
        if int(certificate.source_doc_index) in selected_set
        and int(certificate.target_doc_index) in selected_set
        and _is_admissible_certificate(
            certificate,
            certificate_policy=certificate_policy,
            source_bound_doc_indices=source_bound_doc_indices,
            query_relation_tokens=query_relation_tokens,
        )
        and _certificate_relation_aligned_with_query(
            certificate,
            query_relation_tokens=query_relation_tokens,
        )
    )


def _certificate_relation_aligned_with_query(
    certificate: EvidenceCertificate,
    *,
    query_relation_tokens: set[str],
) -> bool:
    triple = certificate.triple or ()
    if len(triple) != 3:
        return False
    relation_tokens = _relation_alignment_tokens(str(triple[1]))
    if not relation_tokens:
        return False
    effective_query_tokens = set(query_relation_tokens)
    effective_query_tokens -= _relation_alignment_tokens(str(certificate.source_title))
    effective_query_tokens -= _relation_alignment_tokens(str(certificate.target_title))
    return _relation_token_sets_overlap(effective_query_tokens, relation_tokens)


def _certificate_query_alignment_rank(
    certificate: EvidenceCertificate,
    *,
    query_relation_tokens: set[str],
) -> int:
    return 0 if _certificate_relation_aligned_with_query(
        certificate,
        query_relation_tokens=query_relation_tokens,
    ) else 1


RELATION_ALIGNMENT_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "and",
        "are",
        "before",
        "both",
        "did",
        "does",
        "film",
        "films",
        "for",
        "from",
        "had",
        "has",
        "have",
        "into",
        "later",
        "movie",
        "movies",
        "older",
        "other",
        "same",
        "song",
        "songs",
        "than",
        "that",
        "the",
        "then",
        "this",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whose",
        "with",
        "younger",
    }
)


def _relation_alignment_tokens(text: object) -> set[str]:
    output: set[str] = set()
    for raw_token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        token = _relation_alignment_token(raw_token)
        if token:
            output.add(token)
    return output


def _relation_alignment_token(token: object) -> str:
    normalized = str(token or "").strip().lower()
    if not normalized or normalized in RELATION_ALIGNMENT_STOPWORDS:
        return ""
    if len(normalized) > 5 and normalized.endswith("ing"):
        normalized = normalized[:-3]
    elif len(normalized) > 4 and normalized.endswith("ied"):
        normalized = normalized[:-3] + "y"
    elif len(normalized) > 4 and normalized.endswith("ed"):
        normalized = normalized[:-2]
    elif len(normalized) > 4 and normalized.endswith("es"):
        normalized = normalized[:-2]
    elif len(normalized) > 3 and normalized.endswith("s"):
        normalized = normalized[:-1]
    if len(normalized) < 3 or normalized in RELATION_ALIGNMENT_STOPWORDS:
        return ""
    return normalized


def _relation_token_sets_overlap(left_tokens: set[str], right_tokens: set[str]) -> bool:
    for left in left_tokens:
        for right in right_tokens:
            if left == right:
                return True
            prefix_len = min(len(left), len(right))
            if prefix_len >= 5 and left[:prefix_len] == right[:prefix_len]:
                return True
            if len(left) >= 5 and right.startswith(left):
                return True
            if len(right) >= 5 and left.startswith(right):
                return True
    return False


def _admissible_closed_certificate_count(
    graph: SourceTextCertificateGraph,
    doc_indices: Sequence[int],
    *,
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_relation_tokens: set[str],
) -> int:
    selected_set = {int(doc_index) for doc_index in doc_indices}
    return sum(
        1
        for certificate in graph.certificates
        if int(certificate.source_doc_index) in selected_set
        and int(certificate.target_doc_index) in selected_set
        and _is_admissible_certificate(
            certificate,
            certificate_policy=certificate_policy,
            source_bound_doc_indices=source_bound_doc_indices,
            query_relation_tokens=query_relation_tokens,
        )
    )


def _certified_doc_count(
    graph: SourceTextCertificateGraph,
    doc_indices: Sequence[int],
    *,
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_relation_tokens: set[str],
) -> int:
    selected_set = {int(doc_index) for doc_index in doc_indices}
    certified = {
        int(certificate.target_doc_index)
        for certificate in graph.certificates
        if int(certificate.source_doc_index) in selected_set
        and int(certificate.target_doc_index) in selected_set
        and _is_admissible_certificate(
            certificate,
            certificate_policy=certificate_policy,
            source_bound_doc_indices=source_bound_doc_indices,
            query_relation_tokens=query_relation_tokens,
        )
    }
    return len(certified)


def _closed_admissible_graph_doc_indices(
    graph: SourceTextCertificateGraph,
    doc_indices: Sequence[int],
    *,
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_relation_tokens: set[str],
) -> Tuple[int, ...]:
    selected_set = {int(doc_index) for doc_index in doc_indices}
    graph_docs: List[int] = []
    for certificate in graph.certificates:
        source = int(certificate.source_doc_index)
        target = int(certificate.target_doc_index)
        if source not in selected_set or target not in selected_set:
            continue
        if not _is_admissible_certificate(
            certificate,
            certificate_policy=certificate_policy,
            source_bound_doc_indices=source_bound_doc_indices,
            query_relation_tokens=query_relation_tokens,
        ):
            continue
        graph_docs.append(source)
        graph_docs.append(target)
    return tuple(unique_ints(graph_docs))


def _is_admissible_certificate(
    certificate: EvidenceCertificate,
    *,
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_relation_tokens: set[str] | None = None,
) -> bool:
    certificate_type = str(certificate.certificate_type)
    policy = normalize_certificate_policy(certificate_policy)
    if policy == CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY:
        return (
            certificate_type == SOURCE_TITLE_ENDPOINT_CERTIFICATE
            and int(certificate.target_doc_index) in source_bound_doc_indices
        )
    if policy == CANONICAL_TRANSITION_CERTIFICATE_POLICY:
        return (
            certificate_type == SOURCE_TITLE_ENDPOINT_CERTIFICATE
            and bool(str(certificate.source_sentence).strip())
            and bool(str(certificate.source_binding).strip())
            and bool(str(certificate.target_binding).strip())
        )
    if policy in {
        CANONICAL_QUERY_TRANSITION_CERTIFICATE_POLICY,
        CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY,
    }:
        if (
            certificate_type != SOURCE_TITLE_ENDPOINT_CERTIFICATE
            or not bool(str(certificate.source_sentence).strip())
            or not bool(str(certificate.source_binding).strip())
            or not bool(str(certificate.target_binding).strip())
        ):
            return False
        if int(certificate.target_doc_index) in source_bound_doc_indices:
            return True
        return bool(
            query_relation_tokens
            and _certificate_relation_aligned_with_query(
                certificate,
                query_relation_tokens=set(query_relation_tokens),
            )
        )
    if policy == LEGACY_ROLE_HINT_CERTIFICATE_POLICY:
        if certificate_type not in LEGACY_ADMISSION_CERTIFICATE_TYPES:
            return False
        roles = {str(role) for role in certificate.roles}
        if certificate_type in {RELATION_ROLE_CERTIFICATE, SURFACE_ROLE_CERTIFICATE}:
            if roles and roles.issubset(BROAD_LOCATION_ROLE_TERMS):
                return False
        return True
    return False


def _uses_fact_origin_suppression(certificate_policy: str) -> bool:
    return (
        normalize_certificate_policy(certificate_policy)
        == CANONICAL_FACT_ORIGIN_TRANSITION_CERTIFICATE_POLICY
    )


def _certificate_fact_origin(certificate: EvidenceCertificate) -> Tuple[int, int]:
    origin = tuple(certificate.fact_origin or ())
    if len(origin) < 2:
        return ()
    return int(origin[0]), int(origin[1])


def _certificate_rank(
    certificate: EvidenceCertificate,
    *,
    certificate_policy: str,
    source_bound_doc_indices: set[int],
    query_relation_tokens: set[str] | None = None,
) -> int:
    if not _is_admissible_certificate(
        certificate,
        certificate_policy=certificate_policy,
        source_bound_doc_indices=source_bound_doc_indices,
        query_relation_tokens=query_relation_tokens,
    ):
        return 9
    certificate_type = str(certificate.certificate_type)
    if certificate_type == DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE:
        return 0
    if certificate_type == RELATION_ROLE_CERTIFICATE:
        return 1
    if certificate_type == SOURCE_TITLE_ENDPOINT_CERTIFICATE:
        return 2
    if certificate_type == SURFACE_ROLE_CERTIFICATE:
        return 3
    return 4


def _candidate_position(candidate_doc_indices: Sequence[int], doc_index: int) -> int:
    for position, candidate_doc_index in enumerate(candidate_doc_indices):
        if int(candidate_doc_index) == int(doc_index):
            return int(position)
    return 10**9


def _forks_existing_source_role(
    certificate: EvidenceCertificate,
    used_source_roles: set[Tuple[int, str]],
) -> bool:
    roles = tuple(str(role) for role in certificate.roles)
    if not roles:
        return False
    source_doc_index = int(certificate.source_doc_index)
    return any((source_doc_index, role) in used_source_roles for role in roles)


def _forks_existing_fact_origin(
    certificate: EvidenceCertificate,
    *,
    used_fact_origins: set[Tuple[int, int]],
    certificate_policy: str,
) -> bool:
    if not _uses_fact_origin_suppression(certificate_policy):
        return False
    fact_origin = _certificate_fact_origin(certificate)
    if not fact_origin:
        return False
    return fact_origin in used_fact_origins


def _mark_source_roles(
    certificate: EvidenceCertificate,
    used_source_roles: set[Tuple[int, str]],
) -> None:
    for role in certificate.roles:
        used_source_roles.add((int(certificate.source_doc_index), str(role)))


def _mark_fact_origin(
    certificate: EvidenceCertificate,
    used_fact_origins: set[Tuple[int, int]],
) -> None:
    fact_origin = _certificate_fact_origin(certificate)
    if fact_origin:
        used_fact_origins.add(fact_origin)


def _reader_order_selected(
    selected: Sequence[int],
    *,
    graph: SourceTextCertificateGraph,
) -> Tuple[int, ...]:
    selected_tuple = tuple(unique_ints(selected))
    selected_set = {int(doc_index) for doc_index in selected_tuple}
    children_by_source: Dict[int, List[int]] = {}
    parent_by_target: Dict[int, int] = {}
    for certificate in sorted(
        graph.certificates,
        key=lambda item: (
            _candidate_position(graph.candidate_doc_indices, int(item.source_doc_index)),
            _candidate_position(graph.candidate_doc_indices, int(item.target_doc_index)),
            int(item.source_doc_index),
            int(item.target_doc_index),
            str(item.certificate_type),
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


def _append_unique(output: List[int], doc_index: int, *, limit: int) -> None:
    if len(output) >= int(limit):
        return
    if int(doc_index) in output:
        return
    output.append(int(doc_index))
