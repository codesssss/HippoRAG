"""Source-frontier construction for source-authorized vocab-strict retrieval."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

from .certificate_graph import (
    DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE,
    RELATION_ROLE_CERTIFICATE,
    SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    SURFACE_ROLE_CERTIFICATE,
    EvidenceNode,
    SourceTextCertificateGraph,
)
from .normalize import unique_ints
from .query_mentions import query_mentioned_doc_indices


FRONTIER_CERTIFICATE_TYPES = frozenset(
    {
        RELATION_ROLE_CERTIFICATE,
        SOURCE_TITLE_ENDPOINT_CERTIFICATE,
        SURFACE_ROLE_CERTIFICATE,
        DIRECTIONAL_CHILD_TO_PARENT_CERTIFICATE,
    }
)


SOURCE_FRONTIER_CONTRACT: Mapping[str, bool | str] = {
    "frontier_name": "source_text_frontier",
    "frontier_policy": "initial_head_plus_query_mentions_plus_source_authorized_certificate_closure",
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_proposition_support_evidence": False,
}


@dataclass(frozen=True)
class SourceTextFrontier:
    initial_doc_indices: Tuple[int, ...]
    query_mentioned_doc_indices: Tuple[int, ...]
    certificate_source_doc_indices: Tuple[int, ...]
    frontier_doc_indices: Tuple[int, ...]
    trace: Mapping[str, object]


def build_source_text_frontier(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    graph: SourceTextCertificateGraph,
    top_k: int = 5,
) -> SourceTextFrontier:
    """Build the source frontier used by the standalone end-to-end runner."""

    top_k = max(int(top_k), 1)
    candidates = unique_ints(candidate_doc_indices)
    node_lookup = {int(node.doc_index): node for node in nodes}
    initial = candidates[:top_k]

    query_mentioned = list(
        query_mentioned_doc_indices(
            query=str(query),
            nodes=node_lookup,
            candidate_doc_indices=candidates,
        )
    )

    seed_frontier = unique_ints([*initial, *query_mentioned])
    frontier: List[int] = list(seed_frontier)
    frontier_set = {int(doc_index) for doc_index in frontier}
    certificate_sources: List[int] = []
    queue = deque(frontier)

    while queue:
        source_doc_index = int(queue.popleft())
        outgoing = sorted(
            (
                certificate
                for certificate in graph.outgoing(source_doc_index)
                if str(certificate.certificate_type) in FRONTIER_CERTIFICATE_TYPES
            ),
            key=lambda certificate: (
                _candidate_position(candidates, int(certificate.target_doc_index)),
                int(certificate.target_doc_index),
                str(certificate.certificate_type),
            ),
        )
        if outgoing:
            certificate_sources.append(source_doc_index)
        for certificate in outgoing:
            target_doc_index = int(certificate.target_doc_index)
            if target_doc_index in frontier_set:
                continue
            frontier_set.add(target_doc_index)
            frontier.append(target_doc_index)
            queue.append(target_doc_index)

    all_certificate_sources = unique_ints(
        int(certificate.source_doc_index)
        for certificate in graph.certificates
        if str(certificate.certificate_type) in FRONTIER_CERTIFICATE_TYPES
    )
    trace = {
        **dict(SOURCE_FRONTIER_CONTRACT),
        "initial_doc_indices": initial,
        "query_mentioned_doc_indices": unique_ints(query_mentioned),
        "certificate_source_doc_indices": unique_ints(certificate_sources),
        "all_certificate_source_doc_count": len(all_certificate_sources),
        "unauthorized_certificate_source_doc_count": len(
            set(all_certificate_sources) - set(unique_ints(certificate_sources))
        ),
        "frontier_doc_indices": frontier,
        "initial_doc_count": len(initial),
        "query_mentioned_doc_count": len(unique_ints(query_mentioned)),
        "certificate_source_doc_count": len(unique_ints(certificate_sources)),
        "frontier_doc_count": len(frontier),
    }
    return SourceTextFrontier(
        initial_doc_indices=initial,
        query_mentioned_doc_indices=unique_ints(query_mentioned),
        certificate_source_doc_indices=unique_ints(certificate_sources),
        frontier_doc_indices=frontier,
        trace=trace,
    )


def _candidate_position(candidate_doc_indices: Sequence[int], doc_index: int) -> int:
    for position, candidate_doc_index in enumerate(candidate_doc_indices):
        if int(candidate_doc_index) == int(doc_index):
            return int(position)
    return 10**9
