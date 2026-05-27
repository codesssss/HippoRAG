"""Binding prior utilities for DAEC-DAPG.

The main invariant is that binding priors are explicit selection-layer scores.
They are never folded into the support tensor ``phi``.
"""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

from src.dpathrag.daec_dapg.schemas import BindingAssignment


def normalize_slot_priors(scores: Mapping[str, float], *, epsilon: float = 1e-6) -> dict[str, float]:
    """Normalize nonnegative slot grounding scores into a per-slot prior."""

    if not scores:
        return {}
    adjusted = {str(key): max(0.0, float(value)) + float(epsilon) for key, value in scores.items()}
    denom = sum(adjusted.values())
    if denom <= 0.0:
        uniform = 1.0 / max(1, len(adjusted))
        return {key: uniform for key in adjusted}
    return {key: value / denom for key, value in adjusted.items()}


def enumerate_binding_priors(
    slot_scores: Mapping[str, Mapping[str, float]],
    *,
    bmax: int = 16,
    epsilon: float = 1e-6,
) -> list[BindingAssignment]:
    """Enumerate top-``bmax`` binding assignments and renormalize retained priors.

    Args:
        slot_scores: ``slot -> candidate -> grounding score``.
        bmax: maximum retained cartesian assignments.
        epsilon: smoothing for per-slot normalization.

    Returns:
        Retained assignments sorted by decreasing renormalized prior.
    """

    if not slot_scores:
        return [BindingAssignment(slots={}, prior=1.0, raw_score=1.0)]

    normalized_slots: list[tuple[str, dict[str, float]]] = []
    for slot, scores in sorted(slot_scores.items()):
        priors = normalize_slot_priors(scores, epsilon=float(epsilon))
        if not priors:
            continue
        normalized_slots.append((str(slot), priors))
    if not normalized_slots:
        return [BindingAssignment(slots={}, prior=1.0, raw_score=1.0)]

    candidates: list[BindingAssignment] = []
    for values in product(*(list(priors.items()) for _, priors in normalized_slots)):
        slots: dict[str, str] = {}
        raw_score = 1.0
        for (slot, _), (candidate, prior) in zip(normalized_slots, values):
            slots[slot] = candidate
            raw_score *= float(prior)
        candidates.append(BindingAssignment(slots=slots, prior=raw_score, raw_score=raw_score))

    retained = sorted(candidates, key=lambda item: (-item.raw_score, item.key))[: max(1, int(bmax))]
    denom = sum(item.raw_score for item in retained)
    if denom <= 0.0:
        uniform = 1.0 / len(retained)
        return [BindingAssignment(slots=item.slots, prior=uniform, raw_score=item.raw_score) for item in retained]
    return [
        BindingAssignment(slots=item.slots, prior=item.raw_score / denom, raw_score=item.raw_score)
        for item in retained
    ]


def binding_priors_by_key(bindings: Sequence[BindingAssignment]) -> dict[str, float]:
    """Return ``binding.key -> prior`` for retained assignments."""

    return {binding.key: float(binding.prior) for binding in bindings}
