"""Sparse transition operators for BSGS."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


EPS = 1e-12


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def build_slot_marginal(
    slot_likelihoods: dict[str, dict[str, float]],
    slot_weights: dict[str, float],
) -> dict[str, float]:
    """Compute M_t(j) = sum_phi pi_t(phi) L_phi(j)."""
    marginal: dict[str, float] = defaultdict(float)
    for slot_id, likelihood in slot_likelihoods.items():
        weight = max(0.0, float(slot_weights.get(slot_id, 0.0)))
        if weight <= 0:
            continue
        for prop_id, value in likelihood.items():
            marginal[prop_id] += weight * clip01(value)
    return dict(marginal)


def sparse_transition(
    edges: dict[tuple[str, str], float],
    node_likelihood: dict[str, float],
    nodes: Iterable[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Build row-normalized T_t(i,j) over sparse graph edges.

    Edge mass is E(i,j) * M_t(j).  Rows with no positive outgoing mass receive a
    self-loop, which is safer for Week-1 diagnostics than dropping belief mass.
    """
    node_set = set(nodes or [])
    for src, dst in edges:
        node_set.add(src)
        node_set.add(dst)

    weighted_rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (src, dst), edge_weight in edges.items():
        if edge_weight <= 0:
            continue
        weight = float(edge_weight) * clip01(node_likelihood.get(dst, 0.0))
        if weight > 0:
            weighted_rows[src].append((dst, weight))

    transition: dict[tuple[str, str], float] = {}
    for src in node_set:
        row = weighted_rows.get(src, [])
        denom = sum(weight for _, weight in row)
        if denom <= EPS:
            transition[(src, src)] = 1.0
            continue
        for dst, weight in row:
            transition[(src, dst)] = weight / denom
    return transition


def semi_absorbing_transition(
    base_transition: dict[tuple[str, str], float],
    support_prob: dict[str, float],
    nodes: Iterable[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Apply T_tilde(i,j) = c_i 1[i=j] + (1-c_i) T_t(i,j)."""
    node_set = set(nodes or [])
    for src, dst in base_transition:
        node_set.add(src)
        node_set.add(dst)

    rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (src, dst), weight in base_transition.items():
        if weight > 0:
            rows[src].append((dst, float(weight)))

    result: dict[tuple[str, str], float] = {}
    for src in node_set:
        c_i = clip01(support_prob.get(src, 0.0))
        row = rows.get(src, [(src, 1.0)])
        denom = sum(weight for _, weight in row)
        if denom <= EPS:
            row = [(src, 1.0)]
            denom = 1.0
        for dst, weight in row:
            result[(src, dst)] = result.get((src, dst), 0.0) + (1.0 - c_i) * weight / denom
        result[(src, src)] = result.get((src, src), 0.0) + c_i
    return result
