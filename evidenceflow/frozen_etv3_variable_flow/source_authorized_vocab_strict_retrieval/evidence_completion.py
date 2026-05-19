"""Demand-aware source-certified evidence-set completion.

This module is intentionally separate from the older graph-native selector
experiments.  It uses the current source-text certificate graph as an admission
constraint, then completes the reader evidence set only when a selected source
exposes a missing certified target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .certificate_graph import (
    ENDPOINT_TITLE_CERTIFICATE,
    SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    EvidenceCertificate,
    EvidenceNode,
    SourceTextCertificateGraph,
)
from .normalize import unique_ints


SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME = (
    "source_certified_evidence_set_completion"
)

SOURCE_CERTIFIED_EVIDENCE_COMPLETION_CONTRACT: Mapping[str, bool | str] = {
    "paper_facing_method_name": SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME,
    "method_object": "reader_budgeted_source_certified_evidence_set",
    "composition": "retrieval_grounded_evidence_demands_to_source_certified_set_completion",
    "ranked_object": "reader_facing_evidence_context",
    "source_text_certificate_is_admission_requirement": True,
    "uses_retrieval_grounded_evidence_demands": True,
    "uses_budgeted_set_completion": True,
    "uses_tail_only_safe_mode": False,
    "uses_role_vocabulary": False,
    "uses_relation_role_certificates": False,
    "uses_surface_role_certificates": False,
    "uses_directional_child_parent_hints": False,
    "uses_broad_location_blocklist": False,
    "uses_external_baseline_frontier": False,
    "uses_proposition_support_evidence": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_pre_ranked_sto_qstate_lists": False,
    "uses_learned_reranker": False,
}

COMPLETION_ADMISSION_CERTIFICATE_TYPES = frozenset(
    {
        ENDPOINT_TITLE_CERTIFICATE,
        SOURCE_TITLE_ENDPOINT_CERTIFICATE,
    }
)


@dataclass(frozen=True)
class EvidenceDemand:
    """A missing target exposed by a selected source certificate."""

    source_doc_index: int
    target_doc_index: int
    certificate_type: str
    endpoint: str
    covered_before: bool
    triple: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceCompletionStep:
    inserted_doc_index: int
    removed_doc_index: int
    source_doc_indices: Tuple[int, ...]
    covered_demand_count: int
    closed_certificate_count_after: int


@dataclass(frozen=True)
class EvidenceCompletionResult:
    doc_indices: Tuple[int, ...]
    initial_doc_indices: Tuple[int, ...]
    inserted_doc_indices: Tuple[int, ...]
    removed_doc_indices: Tuple[int, ...]
    certified_doc_indices: Tuple[int, ...]
    demands: Tuple[EvidenceDemand, ...]
    steps: Tuple[EvidenceCompletionStep, ...]
    trace: Mapping[str, object]


def complete_source_certified_evidence_set(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    graph: SourceTextCertificateGraph,
    candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    candidate_pool_k: int | None = None,
    protected_doc_indices: Sequence[int] = (),
    admissible_target_doc_indices: Sequence[int] = (),
) -> EvidenceCompletionResult:
    """Complete reader evidence under source-text certificate admission.

    The starting point is the dense/local top-k.  A new document can enter only
    if a currently selected source has an admissible source-text certificate to
    it and that target is not already selected.  There is no weighted score:
    completion candidates are considered in original candidate order, and a
    no-op is returned when no missing demand can be covered.
    """

    del query, nodes
    top_k = max(int(top_k), 1)
    pool_limit = len(candidate_doc_indices) if candidate_pool_k is None else int(candidate_pool_k)
    pool = unique_ints(candidate_doc_indices)[: max(pool_limit, top_k)]
    initial = tuple(pool[:top_k])
    selected: List[int] = list(initial)
    protected = {int(doc_index) for doc_index in unique_ints(protected_doc_indices)}
    if initial:
        protected.add(int(initial[0]))
    admissible_targets = {int(doc_index) for doc_index in unique_ints(admissible_target_doc_indices)}
    demands = _build_retrieval_grounded_demands(
        graph=graph,
        candidate_doc_indices=pool,
        initial_doc_indices=initial,
        admissible_target_doc_indices=admissible_targets,
    )
    steps: List[EvidenceCompletionStep] = []
    inserted: List[int] = []
    removed: List[int] = []
    blocked_targets: set[int] = set()
    active_source_doc_indices = set(selected)

    while len(selected) == top_k and len(steps) < top_k:
        proposal = _next_completion_proposal(
            graph=graph,
            demands=demands,
            selected=selected,
            active_source_doc_indices=active_source_doc_indices,
            protected_doc_indices=protected,
            candidate_doc_indices=pool,
            blocked_target_doc_indices=blocked_targets,
        )
        if proposal is None:
            break
        target_doc_index, victim_position, covered_demands = proposal
        removed_doc_index = int(selected[int(victim_position)])
        selected[int(victim_position)] = int(target_doc_index)
        inserted.append(int(target_doc_index))
        removed.append(int(removed_doc_index))
        blocked_targets.add(int(removed_doc_index))
        active_source_doc_indices = {int(target_doc_index)}
        selected_tuple = tuple(unique_ints(selected))
        steps.append(
            EvidenceCompletionStep(
                inserted_doc_index=int(target_doc_index),
                removed_doc_index=int(removed_doc_index),
                source_doc_indices=tuple(
                    unique_ints(demand.source_doc_index for demand in covered_demands)
                ),
                covered_demand_count=len(covered_demands),
                closed_certificate_count_after=_closed_admissible_certificate_count(
                    graph,
                    selected_tuple,
                ),
            )
        )

    ordered = _reader_order_selected(selected, graph=graph)
    final_set = set(ordered)
    certified_docs = tuple(
        doc_index
        for doc_index in ordered
        if any(
            int(certificate.target_doc_index) == int(doc_index)
            and int(certificate.source_doc_index) in final_set
            and _is_completion_certificate(certificate)
            for certificate in graph.certificates
        )
    )
    covered_before = _covered_demand_count(demands, set(initial))
    covered_after = _covered_demand_count(demands, final_set)
    selected_policy = (
        "source_certified_evidence_completion" if inserted else "stable_reader_context"
    )
    trace = {
        **dict(SOURCE_CERTIFIED_EVIDENCE_COMPLETION_CONTRACT),
        "top_k": top_k,
        "candidate_doc_count": len(pool),
        "certificate_count": len(graph.certificates),
        "certificate_type_counts": dict(graph.certificate_type_counts()),
        "initial_doc_indices": initial,
        "selected_doc_indices": ordered,
        "protected_doc_indices": tuple(sorted(protected)),
        "admissible_target_doc_indices": tuple(sorted(admissible_targets)),
        "admissible_target_doc_count": len(admissible_targets),
        "selected_policy": selected_policy,
        "changed": bool(inserted or removed),
        "inserted_doc_indices": tuple(unique_ints(inserted)),
        "removed_doc_indices": tuple(unique_ints(removed)),
        "blocked_reinsertion_doc_indices": tuple(sorted(blocked_targets)),
        "completion_policy": "source_certified_path_completion_not_source_fanout",
        "demand_count": len(demands),
        "covered_demands_before": int(covered_before),
        "covered_demands_after": int(covered_after),
        "missing_demands_before": max(len(demands) - int(covered_before), 0),
        "missing_demands_after": max(len(demands) - int(covered_after), 0),
        "closed_certificate_count_before": _closed_admissible_certificate_count(graph, initial),
        "closed_certificate_count_after": _closed_admissible_certificate_count(graph, ordered),
        "certified_doc_indices": certified_docs,
        "completion_steps": [_completion_step_trace(step) for step in steps],
        "demands_preview": [_demand_trace(demand) for demand in demands[:12]],
        "no_op_reason": "" if inserted else _no_op_reason(demands, set(initial)),
    }
    return EvidenceCompletionResult(
        doc_indices=ordered,
        initial_doc_indices=initial,
        inserted_doc_indices=tuple(unique_ints(inserted)),
        removed_doc_indices=tuple(unique_ints(removed)),
        certified_doc_indices=certified_docs,
        demands=demands,
        steps=tuple(steps),
        trace=trace,
    )


def _build_retrieval_grounded_demands(
    *,
    graph: SourceTextCertificateGraph,
    candidate_doc_indices: Sequence[int],
    initial_doc_indices: Sequence[int],
    admissible_target_doc_indices: set[int],
) -> Tuple[EvidenceDemand, ...]:
    pool_set = {int(doc_index) for doc_index in candidate_doc_indices}
    initial_set = {int(doc_index) for doc_index in initial_doc_indices}
    demands: List[EvidenceDemand] = []
    seen = set()
    for certificate in sorted(
        graph.certificates,
        key=lambda item: (
            _candidate_position(candidate_doc_indices, int(item.source_doc_index)),
            _candidate_position(candidate_doc_indices, int(item.target_doc_index)),
            int(item.source_doc_index),
            int(item.target_doc_index),
            str(item.certificate_type),
            str(item.endpoint),
        ),
    ):
        if not _is_completion_certificate(certificate):
            continue
        source = int(certificate.source_doc_index)
        target = int(certificate.target_doc_index)
        if source not in pool_set or target not in pool_set or source == target:
            continue
        if admissible_target_doc_indices and target not in admissible_target_doc_indices:
            continue
        key = (source, target, str(certificate.certificate_type), str(certificate.endpoint))
        if key in seen:
            continue
        seen.add(key)
        demands.append(
            EvidenceDemand(
                source_doc_index=source,
                target_doc_index=target,
                certificate_type=str(certificate.certificate_type),
                endpoint=str(certificate.endpoint),
                covered_before=target in initial_set,
                triple=tuple(str(part) for part in (certificate.triple or ())),
            )
        )
    return tuple(demands)


def _next_completion_proposal(
    *,
    graph: SourceTextCertificateGraph,
    demands: Sequence[EvidenceDemand],
    selected: Sequence[int],
    active_source_doc_indices: set[int],
    protected_doc_indices: set[int],
    candidate_doc_indices: Sequence[int],
    blocked_target_doc_indices: set[int],
) -> Tuple[int, int, Tuple[EvidenceDemand, ...]] | None:
    selected_set = {int(doc_index) for doc_index in selected}
    active_missing = [
        demand
        for demand in demands
        if int(demand.source_doc_index) in selected_set
        and int(demand.source_doc_index) in active_source_doc_indices
        and int(demand.target_doc_index) not in selected_set
        and int(demand.target_doc_index) not in blocked_target_doc_indices
        and not _source_endpoint_slot_covered(
            demand=demand,
            demands=demands,
            selected_doc_indices=selected_set,
        )
    ]
    if not active_missing:
        return None

    demand_by_target: Dict[int, List[EvidenceDemand]] = {}
    for demand in active_missing:
        demand_by_target.setdefault(int(demand.target_doc_index), []).append(demand)

    for target_doc_index in sorted(
        demand_by_target,
        key=lambda doc_index: (_candidate_position(candidate_doc_indices, doc_index), doc_index),
    ):
        covered_demands = tuple(demand_by_target[int(target_doc_index)])
        protected_for_target = set(protected_doc_indices)
        protected_for_target.update(int(demand.source_doc_index) for demand in covered_demands)
        victim_position = _victim_position(
            selected=selected,
            protected_doc_indices=protected_for_target,
        )
        if victim_position is None:
            continue
        return int(target_doc_index), int(victim_position), covered_demands
    return None


def _victim_position(
    *,
    selected: Sequence[int],
    protected_doc_indices: set[int],
) -> int | None:
    for position in range(len(selected) - 1, -1, -1):
        doc_index = int(selected[position])
        if doc_index in protected_doc_indices:
            continue
        return int(position)
    return None


def _is_completion_certificate(certificate: EvidenceCertificate) -> bool:
    return str(certificate.certificate_type) in COMPLETION_ADMISSION_CERTIFICATE_TYPES


def _covered_demand_count(
    demands: Sequence[EvidenceDemand],
    selected_doc_indices: set[int],
) -> int:
    return sum(
        1
        for demand in demands
        if int(demand.source_doc_index) in selected_doc_indices
        and int(demand.target_doc_index) in selected_doc_indices
    )


def _source_endpoint_slot_covered(
    *,
    demand: EvidenceDemand,
    demands: Sequence[EvidenceDemand],
    selected_doc_indices: set[int],
) -> bool:
    source = int(demand.source_doc_index)
    endpoint = str(demand.endpoint)
    return any(
        int(other.source_doc_index) == source
        and str(other.endpoint) == endpoint
        and int(other.target_doc_index) in selected_doc_indices
        for other in demands
    )


def _closed_admissible_certificate_count(
    graph: SourceTextCertificateGraph,
    doc_indices: Sequence[int],
) -> int:
    selected_set = {int(doc_index) for doc_index in doc_indices}
    return sum(
        1
        for certificate in graph.certificates
        if int(certificate.source_doc_index) in selected_set
        and int(certificate.target_doc_index) in selected_set
        and _is_completion_certificate(certificate)
    )


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
        if not _is_completion_certificate(certificate):
            continue
        source = int(certificate.source_doc_index)
        target = int(certificate.target_doc_index)
        if source not in selected_set or target not in selected_set:
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


def _candidate_position(candidate_doc_indices: Sequence[int], doc_index: int) -> int:
    for position, candidate_doc_index in enumerate(candidate_doc_indices):
        if int(candidate_doc_index) == int(doc_index):
            return int(position)
    return 10**9


def _demand_trace(demand: EvidenceDemand) -> Mapping[str, object]:
    return {
        "source_doc_index": int(demand.source_doc_index),
        "target_doc_index": int(demand.target_doc_index),
        "certificate_type": str(demand.certificate_type),
        "endpoint": str(demand.endpoint),
        "covered_before": bool(demand.covered_before),
        "triple": list(demand.triple),
    }


def _completion_step_trace(step: EvidenceCompletionStep) -> Mapping[str, object]:
    return {
        "inserted_doc_index": int(step.inserted_doc_index),
        "removed_doc_index": int(step.removed_doc_index),
        "source_doc_indices": tuple(step.source_doc_indices),
        "covered_demand_count": int(step.covered_demand_count),
        "closed_certificate_count_after": int(step.closed_certificate_count_after),
    }


def _no_op_reason(
    demands: Sequence[EvidenceDemand],
    initial_doc_indices: set[int],
) -> str:
    if not demands:
        return "no_retrieval_grounded_source_certified_demands"
    if all(
        int(demand.source_doc_index) in initial_doc_indices
        and int(demand.target_doc_index) in initial_doc_indices
        for demand in demands
        if int(demand.source_doc_index) in initial_doc_indices
    ):
        return "initial_reader_context_already_covers_active_demands"
    return "no_missing_demand_with_replaceable_reader_slot"
