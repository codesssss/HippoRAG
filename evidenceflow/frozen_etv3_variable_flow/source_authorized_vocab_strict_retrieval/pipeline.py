"""End-to-end source-authorized vocab-strict GraphRAG retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .candidate_expansion import (
    SourceTextCandidateExpansion,
    SourceTextCandidateExpansionIndex,
    expand_source_text_candidates,
)
from .active_certificate_graph import ActiveCertificateGraph, activate_certificate_edges
from .certificate_graph import (
    CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    ENDPOINT_TITLE_CERTIFICATE,
    EvidenceNode,
    SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    SourceTextCertificateGraph,
    build_source_text_certificate_graph,
    certificate_policy_contract,
    normalize_certificate_policy,
)
from .contract import CANONICAL_CLEAN_METHOD_NAME
from .evidence_completion import (
    SOURCE_CERTIFIED_EVIDENCE_COMPLETION_CONTRACT,
    SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME,
    EvidenceCompletionResult,
    complete_source_certified_evidence_set,
)
from .frontier import SourceTextFrontier, build_source_text_frontier
from .retriever import (
    VocabStrictRetrievalResult,
    source_authorized_vocab_strict_retrieve,
)
from .trust_region_repair import TrustRegionRepairResult, repair_with_active_certificates
from .normalize import unique_ints


END_TO_END_PIPELINE_CONTRACT: Mapping[str, bool | str] = {
    "paper_facing_method_name": CANONICAL_CLEAN_METHOD_NAME,
    "pipeline": "source_frontier_to_certificate_graph_to_reader_context_assembly",
    "certificate_policy": CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    "uses_evidence_set_objective": False,
    "uses_source_certified_reader_context_assembly": True,
    "uses_role_vocabulary": False,
    "uses_relation_role_certificates": False,
    "uses_surface_role_certificates": False,
    "uses_directional_child_parent_hints": False,
    "uses_in_pool_source_text_repair_targets": True,
    "in_pool_source_text_repair_window": "next_top_k_dense_tail",
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_proposition_support_evidence": False,
}


ACTIVE_TRUST_REGION_METHOD_NAME = "query_conditioned_source_certified_trust_region_repair"
ACTIVE_TRANSITION_CLOSURE_METHOD_NAME = "query_conditioned_source_certified_transition_closure"

ACTIVE_TRUST_REGION_PIPELINE_CONTRACT: Mapping[str, bool | str] = {
    "paper_facing_method_name": ACTIVE_TRUST_REGION_METHOD_NAME,
    "pipeline": "source_text_certificate_graph_to_query_active_graph_to_trust_region_repair",
    "certificate_policy": CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    "uses_query_conditioned_active_certificate_graph": True,
    "uses_trust_region_repair": True,
    "uses_role_vocabulary": False,
    "uses_relation_role_certificates": False,
    "uses_surface_role_certificates": False,
    "uses_directional_child_parent_hints": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_proposition_support_evidence": False,
}

ACTIVE_TRANSITION_CLOSURE_PIPELINE_CONTRACT: Mapping[str, bool | str] = {
    **dict(ACTIVE_TRUST_REGION_PIPELINE_CONTRACT),
    "paper_facing_method_name": ACTIVE_TRANSITION_CLOSURE_METHOD_NAME,
    "pipeline": "source_text_certificate_graph_to_query_active_graph_to_capacity_bounded_transition_closure",
    "uses_capacity_bounded_transition_closure": True,
    "exposes_replacement_count_hyperparameter": False,
}


@dataclass(frozen=True)
class VocabStrictPipelineResult:
    doc_indices: Tuple[int, ...]
    candidate_expansion: SourceTextCandidateExpansion
    frontier: SourceTextFrontier
    retrieval: VocabStrictRetrievalResult
    trace: Mapping[str, object]


@dataclass(frozen=True)
class SourceCertifiedEvidenceCompletionPipelineResult:
    doc_indices: Tuple[int, ...]
    candidate_expansion: SourceTextCandidateExpansion
    frontier: SourceTextFrontier
    completion: EvidenceCompletionResult
    trace: Mapping[str, object]


@dataclass(frozen=True)
class ActiveTrustRegionPipelineResult:
    doc_indices: Tuple[int, ...]
    candidate_expansion: SourceTextCandidateExpansion
    frontier: SourceTextFrontier
    active_graph: ActiveCertificateGraph
    repair: TrustRegionRepairResult
    trace: Mapping[str, object]


def source_authorized_vocab_strict_pipeline_retrieve(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    candidate_pool_k: int = 200,
    max_hops: Optional[int] = None,
    candidate_expansion_index: Optional[SourceTextCandidateExpansionIndex] = None,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    enable_candidate_expansion: bool = True,
    enable_replacement: bool = True,
    allow_frontier_pair_insertion: bool = True,
    assembly_policy: str = "source_prior_preserving_tail_insertion",
    source_prior_policy: str = "dense_head",
    universe_admissible_doc_indices: Sequence[int] = (),
) -> VocabStrictPipelineResult:
    """Run the full clean GraphRAG retrieval chain in one call."""

    top_k = max(int(top_k), 1)
    certificate_policy = normalize_certificate_policy(certificate_policy)
    initial_pool = unique_ints(candidate_doc_indices)[: max(int(candidate_pool_k), top_k)]
    if bool(enable_candidate_expansion):
        candidate_expansion = expand_source_text_candidates(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=initial_pool,
            top_k=top_k,
            expansion_index=candidate_expansion_index,
        )
    else:
        candidate_expansion = SourceTextCandidateExpansion(
            initial_candidate_doc_indices=tuple(initial_pool),
            seed_doc_indices=tuple(initial_pool[:top_k]),
            source_bound_doc_indices=(),
            expanded_doc_indices=(),
            candidate_doc_indices=tuple(initial_pool),
            trace={
                "candidate_expansion": "disabled_for_ablation",
                "initial_candidate_doc_count": len(initial_pool),
                "seed_doc_indices": tuple(initial_pool[:top_k]),
                "seed_doc_count": len(initial_pool[:top_k]),
                "source_bound_doc_indices": (),
                "source_bound_doc_count": 0,
                "expanded_doc_indices": (),
                "expanded_doc_count": 0,
                "candidate_doc_count": len(initial_pool),
                "uses_weighted_score_fusion": False,
                "uses_dataset_routing": False,
                "uses_external_baseline_frontier": False,
                "uses_proposition_support_evidence": False,
            },
        )
    pool = candidate_expansion.candidate_doc_indices
    graph = build_source_text_certificate_graph(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        certificate_policy=certificate_policy,
    )
    frontier = build_source_text_frontier(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        graph=graph,
        top_k=top_k,
    )
    graph_entry_doc_indices = unique_ints(
        frontier.query_mentioned_doc_indices or frontier.initial_doc_indices[:1]
    )
    in_pool_repair_doc_indices = source_text_in_pool_repair_doc_indices(
        graph=graph,
        source_doc_indices=graph_entry_doc_indices,
        candidate_doc_indices=pool,
        top_k=top_k,
    )
    replacement_doc_indices = (
        unique_ints(
            [
                *candidate_expansion.source_bound_doc_indices,
                *in_pool_repair_doc_indices,
                *unique_ints(universe_admissible_doc_indices),
            ]
        )
        if bool(enable_replacement)
        else []
    )
    retrieval_pool = _reader_context_candidate_order(
        candidate_doc_indices=pool,
        graph=graph,
        source_prior_policy=str(source_prior_policy),
        top_k=top_k,
    )
    retrieval = source_authorized_vocab_strict_retrieve(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=retrieval_pool,
        top_k=top_k,
        candidate_pool_k=len(retrieval_pool),
        max_hops=max_hops,
        entry_doc_indices=graph_entry_doc_indices,
        protected_doc_indices=graph_entry_doc_indices,
        source_frontier_doc_indices=frontier.frontier_doc_indices,
        prebuilt_graph=graph,
        certificate_policy=certificate_policy,
        replacement_doc_indices=replacement_doc_indices,
        allow_frontier_pair_insertion=bool(allow_frontier_pair_insertion),
        assembly_policy=str(assembly_policy),
    )
    trace = {
        **dict(END_TO_END_PIPELINE_CONTRACT),
        **certificate_policy_contract(certificate_policy),
        "enable_candidate_expansion": bool(enable_candidate_expansion),
        "enable_replacement": bool(enable_replacement),
        "enable_source_certified_insertion": bool(enable_replacement),
        "source_certified_insertion_target_doc_indices": tuple(replacement_doc_indices),
        "source_certified_insertion_target_doc_count": len(replacement_doc_indices),
        "universe_admissible_doc_indices": tuple(unique_ints(universe_admissible_doc_indices)),
        "universe_admissible_doc_count": len(unique_ints(universe_admissible_doc_indices)),
        "graph_entry_doc_indices": tuple(graph_entry_doc_indices),
        "graph_entry_policy": "query_mentions_else_dense_top1",
        "allow_frontier_pair_insertion": bool(allow_frontier_pair_insertion),
        "assembly_policy": str(assembly_policy),
        "source_prior_policy": str(source_prior_policy),
        "retrieval_candidate_doc_indices": tuple(retrieval_pool),
        "candidate_expansion": dict(candidate_expansion.trace),
        "in_pool_repair_doc_indices": in_pool_repair_doc_indices,
        "in_pool_repair_doc_count": len(in_pool_repair_doc_indices),
        "frontier": dict(frontier.trace),
        "retrieval": dict(retrieval.trace),
    }
    return VocabStrictPipelineResult(
        doc_indices=retrieval.doc_indices,
        candidate_expansion=candidate_expansion,
        frontier=frontier,
        retrieval=retrieval,
        trace=trace,
    )


def source_certified_evidence_completion_pipeline_retrieve(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    candidate_pool_k: int = 200,
    candidate_expansion_index: Optional[SourceTextCandidateExpansionIndex] = None,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
) -> SourceCertifiedEvidenceCompletionPipelineResult:
    """Run demand-aware source-certified evidence-set completion.

    This is the paper-facing completion path.  The older tail-only repair
    pipeline remains available as a safe-mode ablation.
    """

    top_k = max(int(top_k), 1)
    certificate_policy = normalize_certificate_policy(certificate_policy)
    initial_pool = unique_ints(candidate_doc_indices)[: max(int(candidate_pool_k), top_k)]
    candidate_expansion = expand_source_text_candidates(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=initial_pool,
        top_k=top_k,
        expansion_index=candidate_expansion_index,
    )
    pool = candidate_expansion.candidate_doc_indices
    graph = build_source_text_certificate_graph(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        certificate_policy=certificate_policy,
    )
    frontier = build_source_text_frontier(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        graph=graph,
        top_k=top_k,
    )
    protected_doc_indices = unique_ints(
        [
            *(pool[:1]),
            *frontier.query_mentioned_doc_indices,
        ]
    )
    in_pool_repair_doc_indices = source_text_in_pool_repair_doc_indices(
        graph=graph,
        source_doc_indices=frontier.initial_doc_indices,
        candidate_doc_indices=pool,
        top_k=top_k,
    )
    completion = complete_source_certified_evidence_set(
        query=str(query),
        nodes=nodes,
        graph=graph,
        candidate_doc_indices=pool,
        top_k=top_k,
        candidate_pool_k=len(pool),
        protected_doc_indices=protected_doc_indices,
        admissible_target_doc_indices=(),
    )
    trace = {
        **dict(SOURCE_CERTIFIED_EVIDENCE_COMPLETION_CONTRACT),
        **certificate_policy_contract(certificate_policy),
        "paper_facing_method_name": SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME,
        "pipeline": "source_text_certificate_graph_to_demand_aware_evidence_completion",
        "candidate_expansion": dict(candidate_expansion.trace),
        "in_pool_repair_doc_indices": in_pool_repair_doc_indices,
        "in_pool_repair_doc_count": len(in_pool_repair_doc_indices),
        "frontier": dict(frontier.trace),
        "completion": dict(completion.trace),
    }
    return SourceCertifiedEvidenceCompletionPipelineResult(
        doc_indices=completion.doc_indices,
        candidate_expansion=candidate_expansion,
        frontier=frontier,
        completion=completion,
        trace=trace,
    )


def source_active_trust_region_pipeline_retrieve(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    candidate_pool_k: int = 200,
    candidate_expansion_index: Optional[SourceTextCandidateExpansionIndex] = None,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    max_replacements: int = 1,
    tail_only: bool = True,
) -> ActiveTrustRegionPipelineResult:
    """Run query-conditioned active-graph trust-region repair."""

    top_k = max(int(top_k), 1)
    certificate_policy = normalize_certificate_policy(certificate_policy)
    initial_pool = unique_ints(candidate_doc_indices)[: max(int(candidate_pool_k), top_k)]
    candidate_expansion = expand_source_text_candidates(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=initial_pool,
        top_k=top_k,
        expansion_index=candidate_expansion_index,
    )
    pool = candidate_expansion.candidate_doc_indices
    graph = build_source_text_certificate_graph(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        certificate_policy=certificate_policy,
    )
    frontier = build_source_text_frontier(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        graph=graph,
        top_k=top_k,
    )
    entry_doc_indices = unique_ints(frontier.query_mentioned_doc_indices or pool[:1])
    active_graph = activate_certificate_edges(
        query=str(query),
        nodes=nodes,
        certificate_graph=graph,
        candidate_doc_indices=pool,
        entry_doc_indices=entry_doc_indices,
        query_mentioned_doc_indices=frontier.query_mentioned_doc_indices,
    )
    protected_doc_indices = unique_ints(
        [
            *(pool[:1]),
            *frontier.query_mentioned_doc_indices,
        ]
    )
    repair = repair_with_active_certificates(
        initial_doc_indices=pool[:top_k],
        active_graph=active_graph,
        candidate_doc_indices=pool,
        protected_doc_indices=protected_doc_indices,
        top_k=top_k,
        max_replacements=max_replacements,
        tail_only=tail_only,
    )
    trace = {
        **dict(ACTIVE_TRUST_REGION_PIPELINE_CONTRACT),
        **certificate_policy_contract(certificate_policy),
        "candidate_expansion": dict(candidate_expansion.trace),
        "frontier": dict(frontier.trace),
        "active_entry_source_policy": "anchor_or_top1",
        "active_entry_doc_indices": tuple(entry_doc_indices),
        "active_graph": dict(active_graph.trace),
        "repair": dict(repair.trace),
    }
    return ActiveTrustRegionPipelineResult(
        doc_indices=repair.doc_indices,
        candidate_expansion=candidate_expansion,
        frontier=frontier,
        active_graph=active_graph,
        repair=repair,
        trace=trace,
    )


def source_active_transition_closure_pipeline_retrieve(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    candidate_pool_k: int = 200,
    candidate_expansion_index: Optional[SourceTextCandidateExpansionIndex] = None,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
) -> ActiveTrustRegionPipelineResult:
    """Close query-active source-certified transitions up to reader capacity.

    This is the non-parameterized version of active trust-region repair: the
    only budget is the reader context size itself.  It uses the same graph and
    active-edge contract, but it does not expose an extra replacement-count
    knob.
    """

    top_k = max(int(top_k), 1)
    certificate_policy = normalize_certificate_policy(certificate_policy)
    initial_pool = unique_ints(candidate_doc_indices)[: max(int(candidate_pool_k), top_k)]
    candidate_expansion = expand_source_text_candidates(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=initial_pool,
        top_k=top_k,
        expansion_index=candidate_expansion_index,
    )
    pool = candidate_expansion.candidate_doc_indices
    graph = build_source_text_certificate_graph(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        certificate_policy=certificate_policy,
    )
    frontier = build_source_text_frontier(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        graph=graph,
        top_k=top_k,
    )
    entry_doc_indices = unique_ints(frontier.query_mentioned_doc_indices or pool[:1])
    active_graph = activate_certificate_edges(
        query=str(query),
        nodes=nodes,
        certificate_graph=graph,
        candidate_doc_indices=pool,
        entry_doc_indices=entry_doc_indices,
        query_mentioned_doc_indices=frontier.query_mentioned_doc_indices,
        require_source_entry=False,
    )
    protected_doc_indices = unique_ints(
        [
            *(pool[:1]),
            *frontier.query_mentioned_doc_indices,
        ]
    )
    repair = repair_with_active_certificates(
        initial_doc_indices=pool[:top_k],
        active_graph=active_graph,
        candidate_doc_indices=pool,
        protected_doc_indices=protected_doc_indices,
        top_k=top_k,
        max_replacements=top_k,
        tail_only=False,
        allow_pair_insertion=True,
    )
    trace = {
        **dict(ACTIVE_TRANSITION_CLOSURE_PIPELINE_CONTRACT),
        **certificate_policy_contract(certificate_policy),
        "candidate_expansion": dict(candidate_expansion.trace),
        "frontier": dict(frontier.trace),
        "active_entry_source_policy": "anchor_or_top1",
        "active_entry_doc_indices": tuple(entry_doc_indices),
        "active_graph": dict(active_graph.trace),
        "repair": {
            **dict(repair.trace),
            "max_replacements": top_k,
            "tail_only": False,
            "allow_pair_insertion": True,
            "replacement_budget_policy": "reader_capacity",
        },
    }
    return ActiveTrustRegionPipelineResult(
        doc_indices=repair.doc_indices,
        candidate_expansion=candidate_expansion,
        frontier=frontier,
        active_graph=active_graph,
        repair=repair,
        trace=trace,
    )


def source_text_in_pool_repair_doc_indices(
    *,
    graph: SourceTextCertificateGraph,
    source_doc_indices: Sequence[int],
    candidate_doc_indices: Sequence[int],
    top_k: int,
) -> Tuple[int, ...]:
    """Return dense-tail candidates certified by already selected source text.

    This is a local repair list, not corpus expansion.  It only considers
    candidates already present in the dense/local pool after the reader head.
    """

    candidates = unique_ints(candidate_doc_indices)
    head_size = max(int(top_k), 1)
    tail_set = {int(doc_index) for doc_index in candidates[head_size : 2 * head_size]}
    source_set = {int(doc_index) for doc_index in source_doc_indices}
    allowed_types = {ENDPOINT_TITLE_CERTIFICATE, SOURCE_TITLE_ENDPOINT_CERTIFICATE}
    eligible = {
        int(certificate.target_doc_index)
        for source_doc_index in source_set
        for certificate in graph.outgoing(int(source_doc_index))
        if str(certificate.certificate_type) in allowed_types
        and int(certificate.target_doc_index) in tail_set
    }
    return tuple(doc_index for doc_index in candidates if int(doc_index) in eligible)


def _reader_context_candidate_order(
    *,
    candidate_doc_indices: Sequence[int],
    graph: SourceTextCertificateGraph,
    source_prior_policy: str,
    top_k: int = 5,
) -> Tuple[int, ...]:
    candidates = unique_ints(candidate_doc_indices)
    source_prior_policy = str(source_prior_policy)
    if source_prior_policy == "head_certificate_neighbors_first":
        head_set = set(candidates[: max(int(top_k), 1)])
        neighbor_set = {
            int(certificate.source_doc_index)
            for certificate in graph.certificates
            if str(certificate.certificate_type)
            in {ENDPOINT_TITLE_CERTIFICATE, SOURCE_TITLE_ENDPOINT_CERTIFICATE}
            and (
                int(certificate.source_doc_index) in head_set
                or int(certificate.target_doc_index) in head_set
            )
        }
        neighbor_set.update(
            int(certificate.target_doc_index)
            for certificate in graph.certificates
            if str(certificate.certificate_type)
            in {ENDPOINT_TITLE_CERTIFICATE, SOURCE_TITLE_ENDPOINT_CERTIFICATE}
            and (
                int(certificate.source_doc_index) in head_set
                or int(certificate.target_doc_index) in head_set
            )
        )
        neighbor_docs = [int(doc_index) for doc_index in candidates if int(doc_index) in neighbor_set]
        return unique_ints([*neighbor_docs, *candidates])
    if source_prior_policy not in {
        "certificate_sources_first",
        "source_certificate_frontier",
    }:
        return tuple(candidates)
    source_set = {
        int(certificate.source_doc_index)
        for certificate in graph.certificates
        if str(certificate.certificate_type)
        in {ENDPOINT_TITLE_CERTIFICATE, SOURCE_TITLE_ENDPOINT_CERTIFICATE}
    }
    source_docs = [int(doc_index) for doc_index in candidates if int(doc_index) in source_set]
    return unique_ints([*source_docs, *candidates])
