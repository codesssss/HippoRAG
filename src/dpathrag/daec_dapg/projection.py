"""Noisy-OR reader-budget projection for DAEC-DAPG."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def noisy_or_coverage(phi_for_binding: np.ndarray, selected_indices: Sequence[int], weights: np.ndarray | None = None) -> float:
    """Compute ``sum_i pi_i (1 - prod_d(1 - phi[i,d]))`` for fixed binding."""

    phi = np.clip(np.asarray(phi_for_binding, dtype=float), 0.0, 1.0)
    if phi.ndim != 2:
        raise ValueError("phi_for_binding must have shape (num_demands, num_docs)")
    if weights is None:
        weights = np.ones(phi.shape[0], dtype=float) / max(1, phi.shape[0])
    weights = np.asarray(weights, dtype=float)
    selected = [int(idx) for idx in selected_indices if 0 <= int(idx) < phi.shape[1]]
    if not selected:
        return 0.0
    miss = np.prod(1.0 - phi[:, selected], axis=1)
    return float(np.sum(weights * (1.0 - miss)))


def greedy_noisy_or_select(
    phi_for_binding: np.ndarray,
    *,
    budget: int,
    weights: np.ndarray | None = None,
    candidate_indices: Sequence[int] | None = None,
    min_gain: float = 1e-12,
) -> tuple[list[int], float]:
    """Greedily maximize fixed-binding noisy-OR coverage under cardinality budget."""

    phi = np.clip(np.asarray(phi_for_binding, dtype=float), 0.0, 1.0)
    if phi.ndim != 2:
        raise ValueError("phi_for_binding must have shape (num_demands, num_docs)")
    candidates = list(candidate_indices) if candidate_indices is not None else list(range(phi.shape[1]))
    selected: list[int] = []
    current = 0.0
    for _ in range(max(0, int(budget))):
        best_idx = None
        best_value = current
        for idx in candidates:
            idx = int(idx)
            if idx in selected or idx < 0 or idx >= phi.shape[1]:
                continue
            value = noisy_or_coverage(phi, selected + [idx], weights)
            if value > best_value + float(min_gain):
                best_idx = idx
                best_value = value
        if best_idx is None:
            break
        selected.append(best_idx)
        current = best_value
    return selected, current


def select_best_binding(
    phi: np.ndarray,
    binding_priors: Mapping[str, float],
    binding_keys: Sequence[str],
    *,
    budget: int,
    weights: np.ndarray | None = None,
) -> dict[str, object]:
    """Select per-binding greedy sets, then choose by ``J_b = p(b|q) F_b``."""

    values = np.clip(np.asarray(phi, dtype=float), 0.0, 1.0)
    if values.ndim != 3:
        raise ValueError("phi must have shape (num_demands, num_bindings, num_docs)")
    if values.shape[1] != len(binding_keys):
        raise ValueError("binding_keys length must match phi binding dimension")

    candidates: list[dict[str, object]] = []
    for b_idx, key in enumerate(binding_keys):
        selected, objective = greedy_noisy_or_select(values[:, b_idx, :], budget=int(budget), weights=weights)
        prior = float(binding_priors.get(key, 0.0))
        candidates.append(
            {
                "binding_index": b_idx,
                "binding_key": key,
                "prior": prior,
                "selected_indices": selected,
                "objective": objective,
                "score": prior * objective,
            }
        )
    best = max(candidates, key=lambda item: (float(item["score"]), float(item["objective"]), -int(item["binding_index"])))
    return {"best": best, "candidates": candidates}


def marginalized_noisy_or_coverage(phi: np.ndarray, binding_priors: Sequence[float], selected_indices: Sequence[int], weights: np.ndarray | None = None) -> float:
    """Ablation coverage that marginalizes bindings inside the projection."""

    values = np.clip(np.asarray(phi, dtype=float), 0.0, 1.0)
    priors = np.asarray(binding_priors, dtype=float)
    if values.ndim != 3:
        raise ValueError("phi must have shape (num_demands, num_bindings, num_docs)")
    if priors.shape[0] != values.shape[1]:
        raise ValueError("binding_priors length must match phi binding dimension")
    if weights is None:
        weights = np.ones(values.shape[0], dtype=float) / max(1, values.shape[0])
    weights = np.asarray(weights, dtype=float)
    selected = [int(idx) for idx in selected_indices if 0 <= int(idx) < values.shape[2]]
    if not selected:
        return 0.0
    adjusted = np.clip(values[:, :, selected] * priors.reshape(1, -1, 1), 0.0, 1.0)
    miss = np.prod(1.0 - adjusted, axis=(1, 2))
    return float(np.sum(weights * (1.0 - miss)))
