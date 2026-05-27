"""End-to-end STO GraphRAG evaluation pipeline.

This module owns the full retrieval boundary:

query -> candidate universe -> STO graph retrieval -> final top-k evidence.

Reader QA stays in the existing fixed-top5 QA runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from evidence_transition_graphrag.contract import (
    METHOD_TRACE_NAME as EVIDENCE_TRANSITION_TRACE_NAME,
    METHOD_V2_TRACE_NAME as EVIDENCE_TRANSITION_V2_TRACE_NAME,
)
from .candidate_expansion import build_source_text_candidate_expansion_index
from .candidate_generator import CandidateUniverseGenerator
from .certificate_graph import (
    CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    EvidenceCertificate,
    EvidenceNode,
    SourceTextCertificateGraph,
    build_source_text_certificate_graph,
)
from .contract import CANONICAL_CLEAN_METHOD_NAME
from .evidence_completion import SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME
from .evaluate_report import all_gold_at_k, recall_at_k
from .graph_index import SourceTextGraphIndex
from .graph_native_pipeline import source_authorized_graph_native_pipeline_retrieve
from .normalize import unique_ints
from .pipeline import (
    ACTIVE_TRUST_REGION_METHOD_NAME,
    ACTIVE_TRANSITION_CLOSURE_METHOD_NAME,
    source_active_trust_region_pipeline_retrieve,
    source_active_transition_closure_pipeline_retrieve,
    source_authorized_vocab_strict_pipeline_retrieve,
    source_certified_evidence_completion_pipeline_retrieve,
)


E2E_PIPELINE_CONTRACT: Mapping[str, bool | str] = {
    "pipeline": "candidate_universe_to_sto_graph_to_reader_evidence",
    "candidate_boundary": "method_owned_candidate_universe",
    "dense_topk_is_final_answer_default": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_llm_query_compiler": False,
}

AGSTO_GRAPH_NATIVE_METHOD_NAME = EVIDENCE_TRANSITION_TRACE_NAME
AGSTO_GRAPH_NATIVE_V2_METHOD_NAME = EVIDENCE_TRANSITION_V2_TRACE_NAME
QUERY_GROUNDED_STO_RUNNERS = frozenset(
    {
        "agsto_graph_native",
        "query_grounded_sto_graph_native",
    }
)
QUERY_GROUNDED_STO_V2_RUNNERS = frozenset(
    {
        "agsto_graph_native_v2",
        "query_grounded_sto_graph_native_v2",
        "evidence_transition_v2",
    }
)


@dataclass(frozen=True)
class E2ERetrievalResult:
    doc_indices: Tuple[int, ...]
    certified_doc_indices: Tuple[int, ...]
    trace: Mapping[str, object]


@dataclass(frozen=True)
class AGSTOGraphNativeResult:
    doc_indices: Tuple[int, ...]
    certified_doc_indices: Tuple[int, ...]
    trace: Mapping[str, object]


def retrieve_one_e2e(
    *,
    query: str,
    query_index: int,
    row: Mapping[str, Any],
    nodes: Sequence[EvidenceNode],
    candidate_generator: CandidateUniverseGenerator,
    runner: str = "pipeline",
    top_k: int = 5,
    candidate_pool_k: int = 200,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    graph_index: SourceTextGraphIndex | None = None,
    candidate_expansion_index: object | None = None,
    enable_pipeline_candidate_expansion: bool = True,
    enable_pipeline_replacement: bool = True,
    allow_frontier_pair_insertion: bool = True,
    assembly_policy: str = "source_prior_preserving_tail_insertion",
    source_prior_policy: str = "dense_head",
) -> E2ERetrievalResult:
    """Run one query through candidate generation and STO retrieval."""

    candidate_universe = candidate_generator.generate(
        query=str(query),
        query_index=int(query_index),
        row=row,
        top_n=max(int(candidate_pool_k), int(top_k)),
    )
    candidates = candidate_universe.doc_indices
    runner = str(runner)
    if runner == "graph_native":
        result = source_authorized_graph_native_pipeline_retrieve(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=candidates,
            top_k=top_k,
            candidate_pool_k=candidate_pool_k,
            certificate_policy=certificate_policy,
            graph_index=graph_index,
        )
        certified = result.certified_doc_indices
        retrieval_trace = dict(result.trace)
    elif runner == "completion":
        result = source_certified_evidence_completion_pipeline_retrieve(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=candidates,
            top_k=top_k,
            candidate_pool_k=candidate_pool_k,
            candidate_expansion_index=candidate_expansion_index,
            certificate_policy=certificate_policy,
        )
        certified = result.completion.inserted_doc_indices
        retrieval_trace = dict(result.trace)
    elif runner == "active_closure":
        result = source_active_transition_closure_pipeline_retrieve(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=candidates,
            top_k=top_k,
            candidate_pool_k=candidate_pool_k,
            candidate_expansion_index=candidate_expansion_index,
            certificate_policy=certificate_policy,
        )
        certified = result.repair.inserted_doc_indices
        retrieval_trace = dict(result.trace)
    elif runner == "active_trust":
        result = source_active_trust_region_pipeline_retrieve(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=candidates,
            top_k=top_k,
            candidate_pool_k=candidate_pool_k,
            candidate_expansion_index=candidate_expansion_index,
            certificate_policy=certificate_policy,
        )
        certified = result.repair.inserted_doc_indices
        retrieval_trace = dict(result.trace)
    elif runner in QUERY_GROUNDED_STO_RUNNERS:
        result = _agsto_graph_native_retrieve(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=candidates,
            graph_payload=candidate_universe.graph_payload,
            top_k=top_k,
            certificate_policy=certificate_policy,
        )
        certified = result.certified_doc_indices
        retrieval_trace = dict(result.trace)
    elif runner in QUERY_GROUNDED_STO_V2_RUNNERS:
        result = _agsto_graph_native_v2_retrieve(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=candidates,
            graph_payload=candidate_universe.graph_payload,
            top_k=top_k,
            certificate_policy=certificate_policy,
        )
        certified = result.certified_doc_indices
        retrieval_trace = dict(result.trace)
    else:
        result = source_authorized_vocab_strict_pipeline_retrieve(
            query=str(query),
            nodes=nodes,
            candidate_doc_indices=candidates,
            top_k=top_k,
            candidate_pool_k=candidate_pool_k,
            candidate_expansion_index=candidate_expansion_index,
            certificate_policy=certificate_policy,
            enable_candidate_expansion=enable_pipeline_candidate_expansion,
            enable_replacement=enable_pipeline_replacement,
            allow_frontier_pair_insertion=allow_frontier_pair_insertion,
            assembly_policy=str(assembly_policy),
            source_prior_policy=str(source_prior_policy),
            universe_admissible_doc_indices=candidate_universe.admissible_doc_indices,
        )
        certified = result.retrieval.certified_doc_indices
        retrieval_trace = dict(result.trace)

    return E2ERetrievalResult(
        doc_indices=tuple(unique_ints(result.doc_indices)),
        certified_doc_indices=tuple(unique_ints(certified)),
        trace={
            **dict(E2E_PIPELINE_CONTRACT),
            "runner": runner,
            "candidate_universe": {
                **dict(candidate_universe.trace),
                "candidate_doc_indices": tuple(unique_ints(candidates)),
                "admissible_doc_indices": tuple(
                    unique_ints(candidate_universe.admissible_doc_indices)
                ),
            },
            "retrieval": retrieval_trace,
        },
    )


def evaluate_e2e_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    nodes: Sequence[EvidenceNode],
    candidate_generator: CandidateUniverseGenerator,
    runner: str = "pipeline",
    top_k: int = 5,
    candidate_pool_k: int = 200,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    enable_pipeline_candidate_expansion: bool = True,
    enable_pipeline_replacement: bool = True,
    allow_frontier_pair_insertion: bool = True,
    assembly_policy: str = "source_prior_preserving_tail_insertion",
    source_prior_policy: str = "dense_head",
) -> Dict[str, Any]:
    """Evaluate STO GraphRAG retrieval on already-loaded query rows."""

    output_rows: List[Dict[str, Any]] = []
    r5_values: List[float] = []
    all_gold_values: List[float] = []
    certified_counts: List[int] = []
    graph_index = SourceTextGraphIndex.build(nodes) if str(runner) == "graph_native" else None
    candidate_expansion_index = (
        build_source_text_candidate_expansion_index(nodes)
        if str(runner) in {"pipeline", "completion", "active_trust", "active_closure"}
        else None
    )
    prepare_queries = getattr(candidate_generator, "prepare_queries", None)
    if callable(prepare_queries):
        prepare_queries(
            [
                str(row.get("question") or row.get("query") or "")
                for row in rows
            ]
        )

    for fallback_idx, row in enumerate(rows):
        query = str(row.get("question") or row.get("query") or "")
        query_index = int(row.get("query_index", fallback_idx))
        gold = unique_ints(row.get("gold_doc_indices", []) or [])
        result = retrieve_one_e2e(
            query=query,
            query_index=query_index,
            row=row,
            nodes=nodes,
            candidate_generator=candidate_generator,
            runner=runner,
            top_k=top_k,
            candidate_pool_k=candidate_pool_k,
            certificate_policy=certificate_policy,
            graph_index=graph_index,
            candidate_expansion_index=candidate_expansion_index,
            enable_pipeline_candidate_expansion=enable_pipeline_candidate_expansion,
            enable_pipeline_replacement=enable_pipeline_replacement,
            allow_frontier_pair_insertion=allow_frontier_pair_insertion,
            assembly_policy=str(assembly_policy),
            source_prior_policy=str(source_prior_policy),
        )
        retrieved = list(result.doc_indices)
        r5 = recall_at_k(gold, retrieved, top_k)
        all_gold5 = all_gold_at_k(gold, retrieved, top_k)
        r5_values.append(float(r5))
        all_gold_values.append(1.0 if all_gold5 else 0.0)
        certified_counts.append(len(result.certified_doc_indices))
        output_rows.append(
            {
                "query_index": query_index,
                "question": query,
                "gold_doc_indices": gold,
                "retrieved_doc_indices_top5": retrieved[:top_k],
                "query_grounded_sto_recall_at5": round(float(r5), 6),
                "query_grounded_sto_all_gold_at5": bool(all_gold5),
                "source_authorized_vocab_strict_recall_at5": round(float(r5), 6),
                "source_authorized_vocab_strict_all_gold_at5": bool(all_gold5),
                "route_trace": dict(result.trace),
            }
        )

    row_count = len(output_rows)
    denom = float(max(row_count, 1))
    return {
        "row_count": row_count,
        "metrics": {
            "r5": round(sum(r5_values) / denom, 6),
            "all_gold_at5": round(sum(all_gold_values) / denom, 6),
            "mean_certified_doc_count_top5": round(sum(certified_counts) / denom, 6),
        },
        "rows": output_rows,
    }


def method_name_for_runner(runner: str) -> str:
    if str(runner) == "completion":
        return SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME
    if str(runner) == "active_trust":
        return ACTIVE_TRUST_REGION_METHOD_NAME
    if str(runner) == "active_closure":
        return ACTIVE_TRANSITION_CLOSURE_METHOD_NAME
    if str(runner) in QUERY_GROUNDED_STO_RUNNERS:
        return AGSTO_GRAPH_NATIVE_METHOD_NAME
    if str(runner) in QUERY_GROUNDED_STO_V2_RUNNERS:
        return AGSTO_GRAPH_NATIVE_V2_METHOD_NAME
    if str(runner) == "graph_native":
        return f"{CANONICAL_CLEAN_METHOD_NAME}_graph_native"
    return CANONICAL_CLEAN_METHOD_NAME


def _agsto_graph_native_retrieve(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    graph_payload: Mapping[str, object],
    top_k: int,
    certificate_policy: str,
) -> AGSTOGraphNativeResult:
    """Select reader top-k directly from a query-local AG-STO graph."""

    corpus_index = graph_payload.get("agsto_corpus_index")
    local_graph = graph_payload.get("agsto_local_graph")
    if not isinstance(corpus_index, Mapping) or not isinstance(local_graph, Mapping):
        raise ValueError(
            "runner=query_grounded_sto_graph_native requires a candidate generator that "
            "provides agsto_corpus_index and agsto_local_graph in graph_payload"
        )

    from src.agsto.local_graph import order_local_sto_admission_preserving_source_prior

    rooted_local_graph = _agsto_query_rooted_local_graph(local_graph)
    raw_graph_order = list(
        unique_ints(
            order_local_sto_admission_preserving_source_prior(
                local_graph=rooted_local_graph,
                max_docs=None,
            )
        )
    )
    graph_order, delayed_symbolic_entries = _agsto_reader_evidence_order(
        graph_order=raw_graph_order,
        local_graph=rooted_local_graph,
    )
    certificate_graph = build_source_text_certificate_graph(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=candidate_doc_indices,
        source_doc_indices=graph_order[: max(int(top_k), 1)],
        certificate_policy=certificate_policy,
    )
    closure = _agsto_source_text_certificate_closure(
        graph_order=graph_order,
        candidate_doc_indices=candidate_doc_indices,
        certificate_graph=certificate_graph,
        top_k=top_k,
    )
    selected = list(closure["doc_indices"])
    selection = {
        "selection_policy": "query_anchor_first_admission_preserving_sto_frontier",
        "selected_doc_indices": tuple(selected),
        "admitted_doc_count": len(
            unique_ints(rooted_local_graph.get("admitted_doc_indices", []) or [])
        ),
        "local_edge_count": len(list(rooted_local_graph.get("local_edges", []) or [])),
        "graph_order_doc_indices": tuple(graph_order[: max(int(top_k), 1)]),
        "raw_graph_order_doc_indices": tuple(raw_graph_order[: max(int(top_k), 1)]),
        "delayed_symbolic_entry_doc_indices": tuple(delayed_symbolic_entries),
        "query_endpoints": tuple(
            str(endpoint)
            for endpoint in rooted_local_graph.get("symbolic_anchor_endpoints", []) or []
        ),
        "seed_order_policy": rooted_local_graph.get("seed_order_policy", ""),
    }
    for doc_index in unique_ints(graph_order):
        if len(selected) >= max(int(top_k), 1):
            break
        if int(doc_index) not in selected:
            selected.append(int(doc_index))
    for doc_index in unique_ints(candidate_doc_indices):
        if len(selected) >= max(int(top_k), 1):
            break
        if int(doc_index) not in selected:
            selected.append(int(doc_index))
    doc_indices = tuple(selected[: max(int(top_k), 1)])
    certified_doc_indices = tuple(
        unique_ints(
            [
                *_agsto_certified_doc_indices(
                    doc_indices=doc_indices,
                    local_graph=rooted_local_graph,
                ),
                *closure["certified_doc_indices"],
            ]
        )
    )
    trace = {
        "paper_facing_method_name": AGSTO_GRAPH_NATIVE_METHOD_NAME,
        "pipeline": "query_grounded_sto_graph_to_native_reader_evidence",
        "candidate_entrance": "agsto_graph_payload",
        "ranked_object": "query_local_graph_selected_reader_context",
        "uses_agsto_query_local_graph": True,
        "uses_query_grounded_sto_graph": True,
        "uses_graph_native_selection": True,
        "uses_evidence_transition_audit": True,
        "evidence_transition_audit_only": True,
        "uses_source_text_evidence_validation": True,
        "source_text_evidence_validation_only": True,
        "uses_replacement_policy": False,
        "uses_source_prior_repair": False,
        "uses_agsto_legacy_selector": False,
        "uses_agsto_weighted_evidence_set_scoring": False,
        "uses_agsto_proposal_fusion": False,
        "uses_llm_query_schema": False,
        "dense_topk_is_final_answer_default": False,
        "uses_weighted_score_fusion": False,
        "uses_dataset_routing": False,
        "uses_external_baseline_frontier": False,
        "top_k": max(int(top_k), 1),
        "selected_doc_indices": doc_indices,
        "certified_doc_indices": certified_doc_indices,
        "local_graph": _agsto_local_graph_trace(rooted_local_graph),
        "selection": _agsto_selection_trace(selection),
        "evidence_transition_audit": dict(closure["trace"]),
        "source_text_evidence_validation": dict(closure["trace"]),
    }
    return AGSTOGraphNativeResult(
        doc_indices=doc_indices,
        certified_doc_indices=certified_doc_indices,
        trace=trace,
    )


def _agsto_graph_native_v2_retrieve(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    graph_payload: Mapping[str, object],
    top_k: int,
    certificate_policy: str,
) -> AGSTOGraphNativeResult:
    """Select reader top-k from channel-balanced query-local AG-STO evidence."""

    corpus_index = graph_payload.get("agsto_corpus_index")
    local_graph = graph_payload.get("agsto_local_graph")
    if not isinstance(corpus_index, Mapping) or not isinstance(local_graph, Mapping):
        raise ValueError(
            "runner=query_grounded_sto_graph_native_v2 requires a candidate generator that "
            "provides agsto_corpus_index and agsto_local_graph in graph_payload"
        )

    from src.agsto.local_graph import (
        local_sto_evidence_channels,
        order_local_sto_channel_balanced_source_prior,
    )

    rooted_local_graph = _agsto_query_rooted_local_graph(local_graph)
    channel_map = local_sto_evidence_channels(
        local_graph=rooted_local_graph,
        max_docs=None,
    )
    raw_graph_order = list(
        unique_ints(
            order_local_sto_channel_balanced_source_prior(
                local_graph=rooted_local_graph,
                max_docs=None,
            )
        )
    )
    graph_order, delayed_symbolic_entries = _agsto_reader_evidence_order(
        graph_order=raw_graph_order,
        local_graph=rooted_local_graph,
    )
    certificate_graph = build_source_text_certificate_graph(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=candidate_doc_indices,
        source_doc_indices=graph_order[: max(int(top_k), 1)],
        certificate_policy=certificate_policy,
    )
    closure = _agsto_source_text_certificate_closure(
        graph_order=graph_order,
        candidate_doc_indices=candidate_doc_indices,
        certificate_graph=certificate_graph,
        top_k=top_k,
    )
    selected = list(closure["doc_indices"])
    trace_preview_k = max(int(top_k), 50)
    selection = {
        "selection_policy": "etv2_channel_balanced_sto_evidence_exposure",
        "selected_doc_indices": tuple(selected),
        "admitted_doc_count": len(
            unique_ints(rooted_local_graph.get("admitted_doc_indices", []) or [])
        ),
        "local_edge_count": len(list(rooted_local_graph.get("local_edges", []) or [])),
        "graph_order_doc_indices": tuple(graph_order[:trace_preview_k]),
        "raw_graph_order_doc_indices": tuple(raw_graph_order[:trace_preview_k]),
        "trace_preview_k": trace_preview_k,
        "delayed_symbolic_entry_doc_indices": tuple(delayed_symbolic_entries),
        "query_endpoints": tuple(
            str(endpoint)
            for endpoint in rooted_local_graph.get("symbolic_anchor_endpoints", []) or []
        ),
        "seed_order_policy": rooted_local_graph.get("seed_order_policy", ""),
        "evidence_channels": _agsto_evidence_channel_trace(channel_map),
    }
    for doc_index in unique_ints(graph_order):
        if len(selected) >= max(int(top_k), 1):
            break
        if int(doc_index) not in selected:
            selected.append(int(doc_index))
    for doc_index in unique_ints(candidate_doc_indices):
        if len(selected) >= max(int(top_k), 1):
            break
        if int(doc_index) not in selected:
            selected.append(int(doc_index))
    doc_indices = tuple(selected[: max(int(top_k), 1)])
    certified_doc_indices = tuple(
        unique_ints(
            [
                *_agsto_certified_doc_indices(
                    doc_indices=doc_indices,
                    local_graph=rooted_local_graph,
                ),
                *closure["certified_doc_indices"],
            ]
        )
    )
    trace = {
        "paper_facing_method_name": AGSTO_GRAPH_NATIVE_V2_METHOD_NAME,
        "pipeline": "query_grounded_sto_graph_to_channel_balanced_reader_evidence",
        "candidate_entrance": "agsto_graph_payload",
        "ranked_object": "query_local_graph_channel_exposed_reader_context",
        "uses_agsto_query_local_graph": True,
        "uses_query_grounded_sto_graph": True,
        "uses_graph_native_selection": True,
        "uses_evidence_channels": True,
        "uses_channel_balanced_exposure": True,
        "uses_llm_membership_decision": False,
        "etv2_diagnostic_readout": True,
        "v1_frozen_baseline": AGSTO_GRAPH_NATIVE_METHOD_NAME,
        "uses_evidence_transition_audit": True,
        "evidence_transition_audit_only": True,
        "uses_source_text_evidence_validation": True,
        "source_text_evidence_validation_only": True,
        "uses_replacement_policy": False,
        "uses_source_prior_repair": False,
        "uses_agsto_legacy_selector": False,
        "uses_agsto_weighted_evidence_set_scoring": False,
        "uses_agsto_proposal_fusion": False,
        "uses_llm_query_schema": False,
        "dense_topk_is_final_answer_default": False,
        "uses_weighted_score_fusion": False,
        "uses_dataset_routing": False,
        "uses_external_baseline_frontier": False,
        "top_k": max(int(top_k), 1),
        "selected_doc_indices": doc_indices,
        "certified_doc_indices": certified_doc_indices,
        "local_graph": _agsto_local_graph_trace(rooted_local_graph),
        "selection": _agsto_selection_trace(selection),
        "evidence_transition_audit": dict(closure["trace"]),
        "source_text_evidence_validation": dict(closure["trace"]),
    }
    return AGSTOGraphNativeResult(
        doc_indices=doc_indices,
        certified_doc_indices=certified_doc_indices,
        trace=trace,
    )


def _agsto_reader_evidence_order(
    *,
    graph_order: Sequence[int],
    local_graph: Mapping[str, object],
) -> Tuple[List[int], Tuple[int, ...]]:
    """Delay graph-entry-only symbolic anchors behind evidence candidates.

    A symbolic endpoint match is a valid graph entry.  It is not automatically
    a reader-facing evidence passage before any textual/dense evidence has
    appeared.  Once the traversal has reached a textual entry, later graph
    neighbors remain evidence-visible.
    """

    textual_entries = {
        int(doc_index)
        for doc_index in unique_ints(local_graph.get("textual_seed_doc_indices", []) or [])
    }
    if not textual_entries:
        return list(unique_ints(graph_order)), tuple()
    symbolic_entries = {
        int(doc_index)
        for doc_index in unique_ints(local_graph.get("symbolic_seed_doc_indices", []) or [])
    }
    delayed_prefix: List[int] = []
    for doc_index in unique_ints(graph_order):
        doc = int(doc_index)
        if doc in textual_entries:
            break
        if doc in symbolic_entries:
            delayed_prefix.append(doc)
    delayed = tuple(delayed_prefix)
    if not delayed:
        return list(unique_ints(graph_order)), tuple()
    delayed_set = set(delayed)
    evidence_first = [
        int(doc_index)
        for doc_index in unique_ints(graph_order)
        if int(doc_index) not in delayed_set
    ]
    return [*evidence_first, *delayed], delayed


def _agsto_source_text_certificate_closure(
    *,
    graph_order: Sequence[int],
    candidate_doc_indices: Sequence[int],
    certificate_graph: SourceTextCertificateGraph,
    top_k: int,
) -> Mapping[str, object]:
    """Validate AG-STO frontier docs with deterministic source-text evidence.

    The AG-STO role graph chooses the source frontier.  Source-text certificates
    only verify whether selected frontier docs contain source-to-target evidence
    transitions. This is not a score fusion, insertion policy, or dense-top-k
    replacement.
    """

    clean_top_k = max(int(top_k), 1)
    ordered_sources = tuple(unique_ints(graph_order))
    candidate_rank = {
        int(doc_index): rank
        for rank, doc_index in enumerate(unique_ints(candidate_doc_indices))
    }
    source_frontier = tuple(ordered_sources[:clean_top_k])
    outgoing_by_source = {
        int(source_doc_index): _agsto_sorted_source_text_certificates(
            certificate_graph=certificate_graph,
            source_doc_index=int(source_doc_index),
            candidate_rank=candidate_rank,
        )
        for source_doc_index in source_frontier
    }

    selected: List[int] = []
    seen = set()

    def add(doc_index: int) -> bool:
        doc = int(doc_index)
        if doc in seen or len(selected) >= clean_top_k:
            return False
        if doc not in candidate_rank:
            return False
        seen.add(doc)
        selected.append(doc)
        return True

    for source_doc_index in ordered_sources:
        if len(selected) >= clean_top_k:
            break
        add(int(source_doc_index))
    for doc_index in unique_ints(candidate_doc_indices):
        if len(selected) >= clean_top_k:
            break
        add(int(doc_index))

    selected_set = {int(doc_index) for doc_index in selected[:clean_top_k]}
    used_certificates: List[EvidenceCertificate] = []
    for source_doc_index in source_frontier:
        for certificate in outgoing_by_source.get(int(source_doc_index), ()) or ():
            if int(certificate.target_doc_index) not in selected_set:
                continue
            if int(certificate.source_doc_index) not in selected_set:
                continue
            used_certificates.append(certificate)
            break

    certified_doc_indices = tuple(
        unique_ints(certificate.target_doc_index for certificate in used_certificates)
    )
    return {
        "doc_indices": tuple(selected[:clean_top_k]),
        "certified_doc_indices": certified_doc_indices,
        "trace": {
            "closure_policy": "validate_agsto_frontier_with_strong_source_text_certificates",
            "validation_policy": "validate_selected_frontier_with_sentence_grounded_evidence_transitions",
            "validation_only": True,
            "source_frontier_doc_indices": source_frontier,
            "selected_doc_indices": tuple(selected[:clean_top_k]),
            "certified_doc_indices": certified_doc_indices,
            "certificate_count": len(certificate_graph.certificates),
            "certificate_type_counts": dict(certificate_graph.certificate_type_counts()),
            "used_certificate_count": len(used_certificates),
            "used_certificates": tuple(
                _agsto_certificate_trace(certificate)
                for certificate in used_certificates[:10]
            ),
        },
    }


def _agsto_sorted_source_text_certificates(
    *,
    certificate_graph: SourceTextCertificateGraph,
    source_doc_index: int,
    candidate_rank: Mapping[int, int],
) -> Tuple[EvidenceCertificate, ...]:
    certificates = [
        certificate
        for certificate in certificate_graph.outgoing(int(source_doc_index))
        if _agsto_is_strong_source_text_certificate(certificate)
        and int(certificate.target_doc_index) in candidate_rank
        and int(certificate.target_doc_index) != int(source_doc_index)
    ]
    certificates.sort(
        key=lambda certificate: (
            int(candidate_rank.get(int(certificate.target_doc_index), 10**9)),
            int(certificate.target_doc_index),
            str(certificate.endpoint),
        )
    )
    return tuple(certificates)


def _agsto_is_strong_source_text_certificate(certificate: EvidenceCertificate) -> bool:
    return (
        str(certificate.certificate_type) == SOURCE_TITLE_ENDPOINT_CERTIFICATE
        and bool(str(certificate.source_sentence).strip())
        and bool(str(certificate.source_binding).strip())
        and bool(str(certificate.target_binding).strip())
    )


def _agsto_certificate_trace(certificate: EvidenceCertificate) -> Mapping[str, object]:
    return {
        "source_doc_index": int(certificate.source_doc_index),
        "target_doc_index": int(certificate.target_doc_index),
        "certificate_type": str(certificate.certificate_type),
        "endpoint": str(certificate.endpoint),
        "source_title": str(certificate.source_title),
        "target_title": str(certificate.target_title),
        "direction": str(certificate.direction),
        "source_binding": str(certificate.source_binding),
        "target_binding": str(certificate.target_binding),
    }


def _agsto_query_rooted_local_graph(
    local_graph: Mapping[str, object],
) -> Mapping[str, object]:
    """Return a local graph view whose seed order starts from query anchors."""

    admitted_set = {
        int(doc_index)
        for doc_index in unique_ints(local_graph.get("admitted_doc_indices", []) or [])
    }
    symbolic_seed_doc_indices = [
        int(doc_index)
        for doc_index in unique_ints(local_graph.get("symbolic_seed_doc_indices", []) or [])
        if int(doc_index) in admitted_set
    ]
    textual_seed_doc_indices = [
        int(doc_index)
        for doc_index in unique_ints(local_graph.get("textual_seed_doc_indices", []) or [])
        if int(doc_index) in admitted_set
    ]
    seed_doc_indices = unique_ints([*symbolic_seed_doc_indices, *textual_seed_doc_indices])
    if not seed_doc_indices:
        seed_doc_indices = unique_ints(local_graph.get("seed_doc_indices", []) or [])
    rooted = dict(local_graph)
    rooted["seed_doc_indices"] = tuple(seed_doc_indices)
    rooted["seed_order_policy"] = "query_symbolic_anchors_before_textual_entry"
    return rooted


def _agsto_certified_doc_indices(
    *,
    doc_indices: Sequence[int],
    local_graph: Mapping[str, object],
) -> Tuple[int, ...]:
    admission_trace = local_graph.get("doc_admission_trace", {}) or {}
    selected_set = {int(doc_index) for doc_index in unique_ints(doc_indices)}
    graph_participants = {
        int(edge.get("left_doc", -1))
        for edge in list(local_graph.get("local_edges", []) or [])
        if int(edge.get("left_doc", -1)) in selected_set
        and int(edge.get("right_doc", -1)) in selected_set
    }
    graph_participants.update(
        int(edge.get("right_doc", -1))
        for edge in list(local_graph.get("local_edges", []) or [])
        if int(edge.get("left_doc", -1)) in selected_set
        and int(edge.get("right_doc", -1)) in selected_set
    )
    first_doc = unique_ints(doc_indices)[:1]
    first_doc_set = {int(first_doc[0])} if first_doc else set()
    certified: List[int] = []
    for doc_index in unique_ints(doc_indices):
        row = (
            admission_trace.get(str(int(doc_index)), {})
            if isinstance(admission_trace, Mapping)
            else {}
        )
        if not isinstance(row, Mapping):
            continue
        try:
            distance = int(row.get("distance", 0) or 0)
        except (TypeError, ValueError):
            distance = 0
        if distance > 0 or (
            int(doc_index) in graph_participants and int(doc_index) not in first_doc_set
        ):
            certified.append(int(doc_index))
    return tuple(certified)


def _agsto_local_graph_trace(local_graph: Mapping[str, object]) -> Mapping[str, object]:
    stats = dict(local_graph.get("stats", {}) or {})
    local_edges = list(local_graph.get("local_edges", []) or [])
    return {
        "method": str(local_graph.get("method", "query_local_sto_graph_admission")),
        "seed_doc_indices": tuple(unique_ints(local_graph.get("seed_doc_indices", []) or [])),
        "textual_seed_doc_indices": tuple(
            unique_ints(local_graph.get("textual_seed_doc_indices", []) or [])
        ),
        "symbolic_seed_doc_indices": tuple(
            unique_ints(local_graph.get("symbolic_seed_doc_indices", []) or [])
        ),
        "symbolic_anchor_endpoints": tuple(
            str(endpoint)
            for endpoint in local_graph.get("symbolic_anchor_endpoints", []) or []
        ),
        "seed_order_policy": str(local_graph.get("seed_order_policy", "")),
        "admitted_doc_count": len(
            unique_ints(local_graph.get("admitted_doc_indices", []) or [])
        ),
        "local_edge_count": len(local_edges),
        "local_edge_preview": tuple(dict(edge) for edge in local_edges[:10]),
        "stats": stats,
    }


def _agsto_evidence_channel_trace(
    channels: Mapping[str, Sequence[int]],
    *,
    preview_k: int = 20,
) -> Mapping[str, object]:
    return {
        str(name): {
            "doc_indices": tuple(unique_ints(values)[: max(int(preview_k), 1)]),
            "doc_count": len(unique_ints(values)),
        }
        for name, values in sorted(channels.items(), key=lambda item: str(item[0]))
    }


def _agsto_selection_trace(selection: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "selection_policy": str(selection.get("selection_policy", "")),
        "selected_doc_indices": tuple(
            unique_ints(selection.get("selected_doc_indices", []) or [])
        ),
        "graph_order_doc_indices": tuple(
            unique_ints(selection.get("graph_order_doc_indices", []) or [])
        ),
        "raw_graph_order_doc_indices": tuple(
            unique_ints(selection.get("raw_graph_order_doc_indices", []) or [])
        ),
        "delayed_symbolic_entry_doc_indices": tuple(
            unique_ints(selection.get("delayed_symbolic_entry_doc_indices", []) or [])
        ),
        "admitted_doc_count": int(selection.get("admitted_doc_count", 0) or 0),
        "local_edge_count": int(selection.get("local_edge_count", 0) or 0),
        "query_endpoints": tuple(str(endpoint) for endpoint in selection.get("query_endpoints", []) or []),
        "selection_rows": tuple(
            dict(row)
            for row in list(selection.get("selection_rows", []) or [])[:10]
            if isinstance(row, Mapping)
        ),
        "set_objective": dict(selection.get("set_objective", {}) or {}),
        "evidence_channels": dict(selection.get("evidence_channels", {}) or {}),
        "trace_preview_k": int(selection.get("trace_preview_k", 0) or 0),
    }
