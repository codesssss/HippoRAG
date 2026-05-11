"""Native PrefixResidualReadout for PCEC."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from evidence_transition_graphragv4_composition.contract import (
    DEFAULT_ADMISSION_OBJECTIVE,
    DEFAULT_ORDERING_POLICY,
    DEFAULT_RETENTION_PROXY,
    METHOD_NAME,
    PAPER_FACING_METHOD_NAME,
    residual_budget,
    validate_budgets,
)
from evidence_transition_graphragv4_composition.pcec_types import (
    PCECQueryState,
    PCECReadoutResult,
    ordered_unique_positions,
)


def finalized_positions(selected_positions: Sequence[int], *, pool_size: int, reader_budget_k: int) -> list[int]:
    front = ordered_unique_positions(selected_positions, pool_size=pool_size, limit=reader_budget_k)
    seen = set(front)
    for pos in range(int(pool_size)):
        if len(front) >= int(reader_budget_k):
            break
        if pos not in seen:
            seen.add(pos)
            front.append(pos)
    return front[: int(reader_budget_k)]


def safe_swap_steps(selector_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    safe_trace = selector_trace.get("safe_projection_trace")
    if isinstance(safe_trace, Mapping) and isinstance(safe_trace.get("safe_swap_steps"), list):
        return [dict(row) for row in safe_trace.get("safe_swap_steps") or [] if isinstance(row, Mapping)]
    steps = selector_trace.get("selection_steps")
    return [dict(row) for row in steps or [] if isinstance(row, Mapping)]


def compose_pcec_readout(
    state: PCECQueryState,
    *,
    requirements: Sequence[Any],
    utility_provider: Any,
) -> PCECReadoutResult:
    validate_budgets(reader_budget_k=state.reader_budget_k, prefix_budget_m=state.prefix_budget_m)
    baseline_positions = list(range(min(int(state.reader_budget_k), len(state.pool_docs))))
    retained_prefix_positions = list(range(min(int(state.prefix_budget_m), len(baseline_positions))))
    selected_positions, selector_trace = utility_provider.select_positions(state, requirements)
    final_positions = finalized_positions(
        selected_positions,
        pool_size=len(state.pool_docs),
        reader_budget_k=state.reader_budget_k,
    )
    admitted_positions = [pos for pos in final_positions if pos not in baseline_positions]
    steps = safe_swap_steps(selector_trace)
    first_step = steps[0] if steps else {}
    safe_trace = selector_trace.get("safe_projection_trace") if isinstance(selector_trace, Mapping) else {}
    if not isinstance(safe_trace, Mapping):
        safe_trace = {}
    decision = "admit" if admitted_positions or final_positions != baseline_positions else "keep_baseline"
    prefix_preserved = final_positions[: len(retained_prefix_positions)] == retained_prefix_positions
    final_titles = [state.pool_titles[pos] for pos in final_positions if pos < len(state.pool_titles)]
    final_docs = [state.pool_docs[pos] for pos in final_positions if pos < len(state.pool_docs)]
    trace = {
        "method": METHOD_NAME,
        "paper_facing_method": PAPER_FACING_METHOD_NAME,
        "readout": "prefix_residual_admission",
        "objective": DEFAULT_ADMISSION_OBJECTIVE,
        "retention_proxy": DEFAULT_RETENTION_PROXY,
        "reader_budget_k": int(state.reader_budget_k),
        "prefix_budget_m": int(state.prefix_budget_m),
        "residual_budget": int(residual_budget(state.reader_budget_k, state.prefix_budget_m)),
        "baseline_positions": list(baseline_positions),
        "baseline_titles": [state.pool_titles[pos] for pos in baseline_positions if pos < len(state.pool_titles)],
        "retained_prefix_positions": list(retained_prefix_positions),
        "retained_prefix_titles": [
            state.pool_titles[pos] for pos in retained_prefix_positions if pos < len(state.pool_titles)
        ],
        "baseline_objective": selector_trace.get("baseline_objective", safe_trace.get("baseline_objective")),
        "best_candidate_position": first_step.get("in_position") if first_step else None,
        "best_candidate_title": first_step.get("in_title") if first_step else None,
        "best_candidate_objective": first_step.get("objective") if first_step else None,
        "best_candidate_gain": first_step.get("objective_gain") if first_step else None,
        "decision": decision,
        "prefix_preserved": bool(prefix_preserved),
        "final_positions": list(final_positions),
        "final_titles": list(final_titles),
        "admitted_positions": list(admitted_positions),
        "ordering_policy": DEFAULT_ORDERING_POLICY,
        "binding_protocol": "dbec_best_binding_frozen",
        "selected_binding_id": selector_trace.get("selected_binding_id"),
        "safe_decision": safe_trace.get("safe_decision"),
        "safe_swap_steps": list(steps),
    }
    return PCECReadoutResult(
        final_positions=tuple(final_positions),
        final_docs=tuple(final_docs),
        final_titles=tuple(final_titles),
        baseline_positions=tuple(baseline_positions),
        retained_prefix_positions=tuple(retained_prefix_positions),
        admitted_positions=tuple(admitted_positions),
        trace=trace,
        legacy_selector_trace=dict(selector_trace),
    )
