"""Propagation and support tensor helpers for DAEC-DAPG."""

from __future__ import annotations

import numpy as np


def _normalize_source(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    denom = float(values.sum())
    if denom <= 0.0:
        return values
    return values / denom


def finite_horizon_absorption(
    q: np.ndarray,
    r: np.ndarray,
    source_u: np.ndarray,
    source_d: np.ndarray | None = None,
    *,
    alpha: float = 0.15,
    horizon: int = 2,
) -> np.ndarray:
    """Compute finite-horizon document absorption for one channel.

    Uses row-vector dynamics over transient nodes. The result is clipped to
    ``[0, 1]`` for noisy-OR compatibility.
    """

    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    source_u = _normalize_source(np.asarray(source_u, dtype=float))
    if source_d is None:
        source_d = np.zeros(r.shape[1], dtype=float)
    source_d = _normalize_source(np.asarray(source_d, dtype=float))
    if q.shape[0] != q.shape[1]:
        raise ValueError("q must be square")
    if q.shape[0] != r.shape[0]:
        raise ValueError("q and r must have the same transient row count")
    if source_u.shape[0] != q.shape[0]:
        raise ValueError("source_u length must match transient nodes")
    if source_d.shape[0] != r.shape[1]:
        raise ValueError("source_d length must match document nodes")

    if q.shape[0] == 0:
        return np.clip(source_d, 0.0, 1.0)

    one = np.ones((q.shape[0], 1), dtype=float)
    q_alpha = (1.0 - float(alpha)) * q + float(alpha) * (one @ source_u.reshape(1, -1))
    r_alpha = (1.0 - float(alpha)) * r + float(alpha) * (one @ source_d.reshape(1, -1))
    current = source_u.copy()
    absorbed = source_d.copy()
    for _ in range(max(0, int(horizon))):
        absorbed = absorbed + current @ r_alpha
        current = current @ q_alpha
    return np.clip(absorbed, 0.0, 1.0)


def cosine_gate(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    """Return ``max(0, cosine(query, doc))`` for each document vector."""

    qv = np.asarray(query_vector, dtype=float)
    docs = np.asarray(doc_vectors, dtype=float)
    q_norm = float(np.linalg.norm(qv))
    d_norm = np.linalg.norm(docs, axis=1)
    denom = np.maximum(q_norm * d_norm, 1e-12)
    sims = docs @ qv / denom
    return np.clip(sims, 0.0, 1.0)


def support_tensor(kappa: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """Return pure in-channel support ``phi = kappa * hit``."""

    return np.clip(np.asarray(kappa, dtype=float) * np.asarray(hit, dtype=float), 0.0, 1.0)
