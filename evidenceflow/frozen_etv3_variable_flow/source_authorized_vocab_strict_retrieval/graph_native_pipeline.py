"""Graph-native source-text GraphRAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from .certificate_graph import (
    CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    EvidenceNode,
    certificate_policy_contract,
    normalize_certificate_policy,
)
from .contract import CANONICAL_CLEAN_METHOD_NAME
from .evidence_set_selector import (
    EvidenceSetSelectionResult,
    select_source_authorized_evidence_set,
)
from .graph_index import SourceTextGraphIndex, SourceTextGraphWorkspace


GRAPH_NATIVE_PIPELINE_CONTRACT: Mapping[str, bool | str] = {
    "paper_facing_method_name": f"{CANONICAL_CLEAN_METHOD_NAME}_graph_native",
    "pipeline": "dense_universe_to_graph_index_to_closed_evidence_set",
    "mainline_status": "ablation_not_c_mainline",
    "uses_evidence_set_objective": True,
    "dense_topk_is_final_answer_default": False,
    "dense_topk_is_source_frontier": False,
    "uses_replacement_policy": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_proposition_support_evidence": False,
}


@dataclass(frozen=True)
class GraphNativePipelineResult:
    doc_indices: Tuple[int, ...]
    workspace: SourceTextGraphWorkspace
    selection: EvidenceSetSelectionResult
    trace: Mapping[str, object]

    @property
    def certified_doc_indices(self) -> Tuple[int, ...]:
        return self.selection.certified_doc_indices


def source_authorized_graph_native_pipeline_retrieve(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    candidate_pool_k: int = 200,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    graph_index: SourceTextGraphIndex | None = None,
) -> GraphNativePipelineResult:
    """Run graph-native retrieval that selects top-k as an evidence set."""

    certificate_policy = normalize_certificate_policy(certificate_policy)
    if graph_index is None:
        graph_index = SourceTextGraphIndex.build(nodes)
    workspace = graph_index.build_workspace(
        query=str(query),
        candidate_doc_indices=candidate_doc_indices,
        top_k=top_k,
        candidate_pool_k=candidate_pool_k,
        certificate_policy=certificate_policy,
    )
    selection = select_source_authorized_evidence_set(
        query=str(query),
        graph=workspace.graph,
        candidate_doc_indices=workspace.candidate_doc_indices,
        entry_doc_indices=workspace.entry_grounding.entry_doc_indices,
        root_candidate_doc_indices=workspace.entry_grounding.root_candidate_doc_indices,
        top_k=top_k,
        certificate_policy=certificate_policy,
        source_bound_doc_indices=workspace.candidate_expansion.source_bound_doc_indices,
    )
    trace = {
        **dict(GRAPH_NATIVE_PIPELINE_CONTRACT),
        **certificate_policy_contract(certificate_policy),
        "graph_index": dict(graph_index.trace),
        "workspace": dict(workspace.trace),
        "selection": dict(selection.trace),
    }
    return GraphNativePipelineResult(
        doc_indices=selection.doc_indices,
        workspace=workspace,
        selection=selection,
        trace=trace,
    )
