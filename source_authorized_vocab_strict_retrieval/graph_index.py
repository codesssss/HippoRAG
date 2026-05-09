"""Corpus-side graph index boundary for source-text GraphRAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from .candidate_expansion import (
    SourceTextCandidateExpansion,
    SourceTextCandidateExpansionIndex,
    build_source_text_candidate_expansion_index,
    expand_source_text_candidates,
)
from .certificate_graph import (
    CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    EvidenceNode,
    SourceTextCertificateGraph,
    build_source_text_certificate_graph,
    certificate_policy_contract,
    normalize_certificate_policy,
)
from .entry_grounding import QueryEntryGrounding, ground_query_entries
from .normalize import unique_ints


SOURCE_TEXT_GRAPH_INDEX_CONTRACT: Mapping[str, bool | str] = {
    "graph_index": "corpus_nodes_plus_title_endpoint_indices",
    "query_time_graph": "candidate_induced_source_text_certificate_subgraph",
    "dense_topk_is_final_answer_default": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
}


@dataclass(frozen=True)
class SourceTextGraphWorkspace:
    candidate_doc_indices: Tuple[int, ...]
    candidate_expansion: SourceTextCandidateExpansion
    entry_grounding: QueryEntryGrounding
    graph: SourceTextCertificateGraph
    trace: Mapping[str, object]


@dataclass(frozen=True)
class SourceTextGraphIndex:
    nodes: Tuple[EvidenceNode, ...]
    candidate_expansion_index: SourceTextCandidateExpansionIndex
    trace: Mapping[str, object]

    @classmethod
    def build(
        cls,
        nodes: Sequence[EvidenceNode],
        *,
        max_specific_token_doc_count: int = 32,
    ) -> "SourceTextGraphIndex":
        node_tuple = tuple(nodes)
        expansion_index = build_source_text_candidate_expansion_index(
            node_tuple,
            max_specific_token_doc_count=max_specific_token_doc_count,
        )
        return cls(
            nodes=node_tuple,
            candidate_expansion_index=expansion_index,
            trace={
                **dict(SOURCE_TEXT_GRAPH_INDEX_CONTRACT),
                "node_count": len(node_tuple),
                "max_specific_token_doc_count": max(int(max_specific_token_doc_count), 1),
            },
        )

    def build_workspace(
        self,
        *,
        query: str,
        candidate_doc_indices: Sequence[int],
        top_k: int = 5,
        candidate_pool_k: int = 200,
        certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    ) -> SourceTextGraphWorkspace:
        """Extract the query-local graph workspace from the corpus index."""

        certificate_policy = normalize_certificate_policy(certificate_policy)
        top_k = max(int(top_k), 1)
        initial_pool = tuple(unique_ints(candidate_doc_indices)[: max(int(candidate_pool_k), top_k)])
        candidate_expansion = expand_source_text_candidates(
            query=str(query),
            nodes=self.nodes,
            candidate_doc_indices=initial_pool,
            top_k=top_k,
            expansion_index=self.candidate_expansion_index,
        )
        pool = candidate_expansion.candidate_doc_indices
        entry_grounding = ground_query_entries(
            query=str(query),
            nodes=self.nodes,
            candidate_doc_indices=pool,
            dense_entry_count=1,
            root_candidate_count=top_k,
        )
        graph = build_source_text_certificate_graph(
            query=str(query),
            nodes=self.nodes,
            candidate_doc_indices=pool,
            certificate_policy=certificate_policy,
        )
        trace = {
            **dict(SOURCE_TEXT_GRAPH_INDEX_CONTRACT),
            **certificate_policy_contract(certificate_policy),
            "initial_candidate_doc_count": len(initial_pool),
            "candidate_doc_count": len(pool),
            "candidate_expansion": dict(candidate_expansion.trace),
            "entry_grounding": dict(entry_grounding.trace),
        }
        return SourceTextGraphWorkspace(
            candidate_doc_indices=pool,
            candidate_expansion=candidate_expansion,
            entry_grounding=entry_grounding,
            graph=graph,
            trace=trace,
        )
