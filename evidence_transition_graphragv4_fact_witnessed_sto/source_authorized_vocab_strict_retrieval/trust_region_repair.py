"""Trust-region repair over query-active source-text certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .active_certificate_graph import ActiveCertificateEdge, ActiveCertificateGraph
from .normalize import unique_ints


TRUST_REGION_REPAIR_CONTRACT: Mapping[str, bool | str] = {
    "repair_policy": "trust_region_active_certificate_repair",
    "active_edges_only": True,
    "protect_query_anchors": True,
    "uses_role_vocabulary": False,
    "uses_proposition_support_evidence": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
}


@dataclass(frozen=True)
class TrustRegionRepairStep:
    source_doc_index: int
    target_doc_index: int
    removed_doc_index: int
    slot_key: str
    certificate_type: str
    activation_reasons: Tuple[str, ...]
    source_removed_doc_index: int = -1
    insertion_mode: str = "target"


@dataclass(frozen=True)
class TrustRegionRepairResult:
    doc_indices: Tuple[int, ...]
    initial_doc_indices: Tuple[int, ...]
    inserted_doc_indices: Tuple[int, ...]
    removed_doc_indices: Tuple[int, ...]
    steps: Tuple[TrustRegionRepairStep, ...]
    trace: Mapping[str, object]


def repair_with_active_certificates(
    *,
    initial_doc_indices: Sequence[int],
    active_graph: ActiveCertificateGraph,
    candidate_doc_indices: Sequence[int],
    protected_doc_indices: Sequence[int] = (),
    top_k: int = 5,
    max_replacements: int = 1,
    tail_only: bool = True,
    allow_pair_insertion: bool = False,
) -> TrustRegionRepairResult:
    """Repair top-k using only query-active source-certified edges."""

    top_k = max(int(top_k), 1)
    max_replacements = max(int(max_replacements), 0)
    pool = tuple(unique_ints(candidate_doc_indices))
    selected: List[int] = list(unique_ints(initial_doc_indices))[:top_k]
    for doc_index in pool:
        if len(selected) >= top_k:
            break
        if int(doc_index) not in selected:
            selected.append(int(doc_index))
    protected = {int(doc_index) for doc_index in unique_ints(protected_doc_indices)}
    if selected:
        protected.add(int(selected[0]))

    steps: List[TrustRegionRepairStep] = []
    used_slots: set[Tuple[int, str]] = set()
    blocked_targets: set[int] = set()
    while len(steps) < max_replacements:
        edge = _next_active_edge(
            active_graph=active_graph,
            selected=selected,
            candidate_doc_indices=pool,
            used_slots=used_slots,
            blocked_targets=blocked_targets,
            allow_pair_insertion=bool(allow_pair_insertion),
        )
        if edge is None:
            break
        source = int(edge.certificate.source_doc_index)
        target = int(edge.certificate.target_doc_index)
        source_selected = source in {int(doc_index) for doc_index in selected}
        if source_selected:
            victim_positions = _victim_positions(
                selected=selected,
                protected_doc_indices={*protected, source},
                count=1,
                tail_only=tail_only,
            )
            if len(victim_positions) < 1:
                blocked_targets.add(target)
                continue
            source_removed_doc_index = -1
            removed_doc_index = int(selected[int(victim_positions[0])])
            selected[int(victim_positions[0])] = target
            insertion_mode = "target"
        else:
            victim_positions = _victim_positions(
                selected=selected,
                protected_doc_indices=protected,
                count=2,
                tail_only=False,
            )
            if len(victim_positions) < 2:
                blocked_targets.add(target)
                continue
            ordered_positions = tuple(sorted(victim_positions))
            source_removed_doc_index = int(selected[int(ordered_positions[0])])
            removed_doc_index = int(selected[int(ordered_positions[1])])
            selected[int(ordered_positions[0])] = source
            selected[int(ordered_positions[1])] = target
            insertion_mode = "source_target_pair"
        blocked_targets.add(removed_doc_index)
        if source_removed_doc_index >= 0:
            blocked_targets.add(source_removed_doc_index)
        used_slots.add((source, str(edge.slot_key)))
        steps.append(
            TrustRegionRepairStep(
                source_doc_index=source,
                target_doc_index=target,
                removed_doc_index=removed_doc_index,
                slot_key=str(edge.slot_key),
                certificate_type=str(edge.certificate.certificate_type),
                activation_reasons=tuple(edge.activation_reasons),
                source_removed_doc_index=source_removed_doc_index,
                insertion_mode=insertion_mode,
            )
        )

    ordered = _reader_order_selected(selected, active_graph=active_graph)
    trace = {
        **dict(TRUST_REGION_REPAIR_CONTRACT),
        "top_k": top_k,
        "max_replacements": max_replacements,
        "tail_only": bool(tail_only),
        "allow_pair_insertion": bool(allow_pair_insertion),
        "initial_doc_indices": tuple(unique_ints(initial_doc_indices))[:top_k],
        "selected_doc_indices": ordered,
        "protected_doc_indices": tuple(sorted(protected)),
        "changed": bool(steps),
        "inserted_doc_indices": _inserted_doc_indices(steps),
        "removed_doc_indices": _removed_doc_indices(steps),
        "repair_step_count": len(steps),
        "repair_steps": [_step_trace(step) for step in steps],
        "active_edge_count": len(active_graph.active_edges),
        "suppressed_edge_count": len(active_graph.suppressed_edges),
        "no_op_reason": "" if steps else _no_op_reason(active_graph=active_graph, selected=selected),
    }
    return TrustRegionRepairResult(
        doc_indices=ordered,
        initial_doc_indices=tuple(unique_ints(initial_doc_indices))[:top_k],
        inserted_doc_indices=_inserted_doc_indices(steps),
        removed_doc_indices=_removed_doc_indices(steps),
        steps=tuple(steps),
        trace=trace,
    )


def _next_active_edge(
    *,
    active_graph: ActiveCertificateGraph,
    selected: Sequence[int],
    candidate_doc_indices: Sequence[int],
    used_slots: set[Tuple[int, str]],
    blocked_targets: set[int],
    allow_pair_insertion: bool,
) -> ActiveCertificateEdge | None:
    selected_set = {int(doc_index) for doc_index in selected}
    candidates = [
        edge
        for edge in active_graph.active_edges
        if int(edge.certificate.target_doc_index) not in selected_set
        and int(edge.certificate.target_doc_index) not in blocked_targets
        and (int(edge.certificate.source_doc_index), str(edge.slot_key)) not in used_slots
        and (
            int(edge.certificate.source_doc_index) in selected_set
            or bool(allow_pair_insertion)
        )
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda edge: (
            0 if int(edge.certificate.source_doc_index) in selected_set else 1,
            _selected_position(selected, int(edge.certificate.source_doc_index)),
            _candidate_position(candidate_doc_indices, int(edge.certificate.source_doc_index)),
            _candidate_position(candidate_doc_indices, int(edge.certificate.target_doc_index)),
            int(edge.certificate.source_doc_index),
            int(edge.certificate.target_doc_index),
            str(edge.certificate.certificate_type),
        )
    )
    return candidates[0]


def _victim_positions(
    *,
    selected: Sequence[int],
    protected_doc_indices: set[int],
    count: int,
    tail_only: bool,
) -> Tuple[int, ...]:
    if tail_only:
        positions = range(len(selected) - 1, len(selected) - 2, -1)
    else:
        positions = range(len(selected) - 1, -1, -1)
    output: List[int] = []
    for position in positions:
        if position < 0 or position >= len(selected):
            continue
        if int(selected[position]) in protected_doc_indices:
            continue
        output.append(int(position))
        if len(output) >= max(int(count), 0):
            break
    return tuple(output)


def _reader_order_selected(
    selected: Sequence[int],
    *,
    active_graph: ActiveCertificateGraph,
) -> Tuple[int, ...]:
    selected_tuple = tuple(unique_ints(selected))
    selected_set = {int(doc_index) for doc_index in selected_tuple}
    children_by_source: Dict[int, List[int]] = {}
    parent_by_target: Dict[int, int] = {}
    for edge in sorted(
        active_graph.active_edges,
        key=lambda item: (
            _candidate_position(active_graph.candidate_doc_indices, int(item.certificate.source_doc_index)),
            _candidate_position(active_graph.candidate_doc_indices, int(item.certificate.target_doc_index)),
            int(item.certificate.source_doc_index),
            int(item.certificate.target_doc_index),
        ),
    ):
        source = int(edge.certificate.source_doc_index)
        target = int(edge.certificate.target_doc_index)
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


def _selected_position(selected: Sequence[int], doc_index: int) -> int:
    for position, selected_doc_index in enumerate(selected):
        if int(selected_doc_index) == int(doc_index):
            return int(position)
    return 10**9


def _step_trace(step: TrustRegionRepairStep) -> Mapping[str, object]:
    return {
        "source_doc_index": int(step.source_doc_index),
        "target_doc_index": int(step.target_doc_index),
        "removed_doc_index": int(step.removed_doc_index),
        "source_removed_doc_index": int(step.source_removed_doc_index),
        "insertion_mode": str(step.insertion_mode),
        "slot_key": str(step.slot_key),
        "certificate_type": str(step.certificate_type),
        "activation_reasons": tuple(step.activation_reasons),
    }


def _removed_doc_indices(steps: Sequence[TrustRegionRepairStep]) -> Tuple[int, ...]:
    return tuple(
        unique_ints(
            doc_index
            for step in steps
            for doc_index in (int(step.source_removed_doc_index), int(step.removed_doc_index))
            if int(doc_index) >= 0
        )
    )


def _inserted_doc_indices(steps: Sequence[TrustRegionRepairStep]) -> Tuple[int, ...]:
    return tuple(
        unique_ints(
            doc_index
            for step in steps
            for doc_index in (
                int(step.source_doc_index) if str(step.insertion_mode) == "source_target_pair" else -1,
                int(step.target_doc_index),
            )
            if int(doc_index) >= 0
        )
    )


def _no_op_reason(
    *,
    active_graph: ActiveCertificateGraph,
    selected: Sequence[int],
) -> str:
    if not active_graph.active_edges:
        return "no_query_active_certificate_edges"
    selected_set = {int(doc_index) for doc_index in selected}
    if not any(int(edge.certificate.source_doc_index) in selected_set for edge in active_graph.active_edges):
        return "no_active_edge_from_selected_source"
    return "no_active_edge_with_replaceable_target"
