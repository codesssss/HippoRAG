"""Verifier calibration utilities for BSGS.

The Week-0 verifier target is a DeBERTa-MNLI model.  This module keeps the
calibration math independent of the model runtime so tests and report scripts
can run even when the model service is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from time import perf_counter
from typing import Callable, Sequence

import numpy as np


EPS = 1e-12


@dataclass
class CalibrationResult:
    temperature: float
    ece: float
    brier: float
    nll: float
    num_examples: int


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature), EPS)
    scaled = logits / temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[int],
    num_bins: int = 10,
) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper >= 1.0:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += float(np.mean(mask)) * abs(conf - acc)
    return ece


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def binary_nll(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 0.0
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def calibrate_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    support_class_index: int = 2,
    grid: Sequence[float] | None = None,
) -> CalibrationResult:
    """Fit temperature by grid-searching minimum binary NLL."""
    logit_arr = np.asarray(logits, dtype=float)
    label_arr = np.asarray(labels, dtype=int)
    if logit_arr.ndim != 2:
        raise ValueError("logits must be a 2D array")
    if logit_arr.shape[0] != label_arr.size:
        raise ValueError("logits and labels must have the same length")
    if label_arr.size == 0:
        return CalibrationResult(temperature=1.0, ece=0.0, brier=0.0, nll=0.0, num_examples=0)

    temps = list(grid or np.linspace(0.5, 5.0, 46))
    best_temp = 1.0
    best_nll = float("inf")
    best_probs: np.ndarray | None = None
    binary_labels = (label_arr == 1).astype(int)
    for temp in temps:
        probs = softmax(logit_arr, temperature=float(temp))[:, support_class_index]
        nll = binary_nll(probs, binary_labels)
        if nll < best_nll:
            best_temp = float(temp)
            best_nll = nll
            best_probs = probs
    assert best_probs is not None
    return CalibrationResult(
        temperature=best_temp,
        ece=expected_calibration_error(best_probs, binary_labels),
        brier=brier_score(best_probs, binary_labels),
        nll=best_nll,
        num_examples=int(label_arr.size),
    )


def measure_throughput(fn: Callable[[Sequence[object]], object], items: Sequence[object]) -> dict[str, float]:
    start = perf_counter()
    fn(items)
    elapsed = max(perf_counter() - start, EPS)
    return {
        "num_examples": float(len(items)),
        "elapsed_seconds": elapsed,
        "samples_per_second": float(len(items) / elapsed),
        "latency_per_1000_seconds": float(1000.0 * elapsed / max(1, len(items))),
    }
