"""Metrics for DAEC-DAPG evidence assembly reports."""

from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

import numpy as np


def noise_rate(selected_gold_count: float, selected_doc_count: int) -> float:
    return 1.0 - float(selected_gold_count) / max(1, int(selected_doc_count))


def phi_auprc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute average precision without requiring sklearn."""

    pairs = sorted(zip(scores, labels), key=lambda item: float(item[0]), reverse=True)
    positives = sum(1 for _, label in pairs if int(label) == 1)
    if positives <= 0:
        return 0.0
    hit = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if int(label) != 1:
            continue
        hit += 1
        precision_sum += hit / rank
    return float(precision_sum / positives)


def summarize_numeric_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(row.get(key) or 0.0) for row in rows]
        summary[key] = round(float(mean(values)), 4)
    return summary


def paired_bootstrap_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    num_samples: int = 1000,
    seed: int = 13,
) -> dict[str, float]:
    """Return a paired bootstrap CI for ``right - left``."""

    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        raise ValueError("left and right must have the same shape")
    if left_arr.size == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(int(seed))
    deltas = []
    for _ in range(max(1, int(num_samples))):
        indices = rng.integers(0, left_arr.size, size=left_arr.size)
        deltas.append(float(np.mean(right_arr[indices] - left_arr[indices])))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": round(float(np.mean(right_arr - left_arr)), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
    }
