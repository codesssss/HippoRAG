"""Evidence closure objectives for AREC-RAG."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def as_support_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"support matrix must be 2-D, got shape={matrix.shape}")
    return np.clip(matrix, 0.0, 1.0)


def obligation_closure(values: Sequence[Sequence[float]] | np.ndarray, selected: Sequence[int]) -> np.ndarray:
    matrix = as_support_matrix(values)
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if not selected:
        return np.zeros(matrix.shape[0], dtype=float)
    cols = [int(idx) for idx in selected if 0 <= int(idx) < matrix.shape[1]]
    if not cols:
        return np.zeros(matrix.shape[0], dtype=float)
    return 1.0 - np.prod(1.0 - matrix[:, cols], axis=1)


def closure_score(values: Sequence[Sequence[float]] | np.ndarray, selected: Sequence[int]) -> float:
    return float(np.sum(obligation_closure(values, selected)))


def residual_masses(values: Sequence[Sequence[float]] | np.ndarray, selected: Sequence[int]) -> np.ndarray:
    return 1.0 - obligation_closure(values, selected)


def marginal_gains(values: Sequence[Sequence[float]] | np.ndarray, selected: Sequence[int]) -> np.ndarray:
    matrix = as_support_matrix(values)
    if matrix.shape[1] == 0:
        return np.zeros(0, dtype=float)
    residual = residual_masses(matrix, selected)
    gains = residual @ matrix
    for idx in selected:
        if 0 <= int(idx) < len(gains):
            gains[int(idx)] = -np.inf
    return gains


def greedy_closure_select(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    budget: int,
    candidate_indices: Sequence[int] | None = None,
    initial_indices: Sequence[int] | None = None,
) -> tuple[list[int], float]:
    """Greedily maximize sum of noisy-OR obligation closure."""

    matrix = as_support_matrix(values)
    if int(budget) <= 0 or matrix.shape[1] == 0:
        return [], 0.0
    allowed = set(int(idx) for idx in (candidate_indices if candidate_indices is not None else range(matrix.shape[1])))
    selected: list[int] = []
    for idx in initial_indices or []:
        idx = int(idx)
        if idx in allowed and 0 <= idx < matrix.shape[1] and idx not in selected and len(selected) < int(budget):
            selected.append(idx)
    while len(selected) < int(budget):
        gains = marginal_gains(matrix, selected)
        best_idx = None
        best_gain = -np.inf
        for idx in sorted(allowed):
            if idx in selected or idx < 0 or idx >= matrix.shape[1]:
                continue
            gain = float(gains[idx])
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None or not np.isfinite(best_gain) or best_gain <= 0.0:
            break
        selected.append(best_idx)
    return selected, closure_score(matrix, selected)

