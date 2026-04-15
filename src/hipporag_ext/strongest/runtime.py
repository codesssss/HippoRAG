from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import sparse as sp

from .candidates import select_union_candidate_indices
from .graph import (
    add_anchor_self_loops,
    apply_topology_suppression,
    build_true_head_local_reset,
    build_true_head_passage_adjacency,
)
from .types import (
    StrongestAnalysisModule,
    StrongestBaselineState,
    StrongestConfig,
    StrongestResult,
)


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    max_value = float(np.max(values))
    min_value = float(np.min(values))
    if max_value - min_value <= 1e-8:
        if max_value <= 0:
            return np.zeros_like(values, dtype=np.float32)
        return np.ones_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        denom = np.linalg.norm(values)
        if denom <= 0:
            return values.astype(np.float32, copy=False)
        return (values / denom).astype(np.float32)
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return (values / denom).astype(np.float32)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    scaled = values / max(float(temperature), 1e-6)
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values)
    if denom <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return (exp_values / denom).astype(np.float32)


def run_local_ppr(
    reset_scores: np.ndarray,
    local_adjacency: sp.csr_matrix,
    gamma: float = 0.15,
    max_iter: int = 64,
    tol: float = 1e-6,
) -> np.ndarray:
    reset_scores = np.asarray(reset_scores, dtype=np.float32).reshape(-1)
    if reset_scores.size == 0:
        return reset_scores

    clipped_reset = np.clip(reset_scores, 0.0, None)
    if not np.any(clipped_reset > 0):
        return clipped_reset

    row_sums = np.asarray(local_adjacency.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_row_sums = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    transition = sp.diags(inv_row_sums) @ local_adjacency
    transition = transition.transpose().tocsr()

    reset = clipped_reset / np.maximum(np.sum(clipped_reset), 1e-8)
    state = np.array(reset, copy=True)
    alpha = float(np.clip(gamma, 0.0, 1.0))
    for _ in range(max(int(max_iter), 1)):
        next_state = alpha * reset + (1.0 - alpha) * np.asarray(transition @ state).reshape(-1)
        if np.max(np.abs(next_state - state)) <= float(tol):
            state = next_state.astype(np.float32, copy=False)
            break
        state = next_state.astype(np.float32, copy=False)
    return state.astype(np.float32, copy=False)


def fallback_to_hippo_head(
    docs: List[str],
    hippo_rank: List[int],
    hippo_score_map: Dict[int, float],
    final_k: int,
) -> tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    final_indices = np.asarray(list(hippo_rank)[: max(int(final_k), 0)], dtype=np.int64)
    final_scores = np.asarray(
        [float(hippo_score_map.get(int(doc_idx), 0.0)) for doc_idx in final_indices],
        dtype=np.float32,
    )
    return final_indices, final_scores, {
        int(doc_idx): float(score) for doc_idx, score in zip(final_indices.tolist(), final_scores.tolist())
    }


def rank_hippo_head_query_conditioned(
    analysis_module: StrongestAnalysisModule,
    state: StrongestBaselineState,
    config: StrongestConfig,
) -> StrongestResult:
    docs = list(state.docs)
    num_docs = len(docs)
    final_k = min(max(int(config.final_k), 0), num_docs)
    candidate_k = min(max(int(config.candidate_k), 0), num_docs)
    hippo_head_k = min(max(int(config.hippo_head_k), 0), num_docs)
    smoothed_union_k = min(max(int(config.smoothed_union_k), 0), num_docs)

    if num_docs == 0:
        empty = np.zeros(0, dtype=np.float32)
        return StrongestResult(
            final_doc_indices=np.zeros(0, dtype=np.int64),
            final_scores=empty,
            candidate_indices=np.zeros(0, dtype=np.int64),
            local_reset_scores=empty,
            anchor_prior=empty,
            smoothed_rank=[],
            dense_rank=[],
            pool_order_indices=np.zeros(0, dtype=np.int64),
            score_map={},
            trace={"status": "empty_docs"},
        )

    passage_query_embedding = np.asarray(state.passage_query_embedding, dtype=np.float32).reshape(-1)
    passage_embeddings = np.asarray(state.passage_embeddings, dtype=np.float32)
    smoothed_embeddings = np.asarray(state.smoothed_embeddings, dtype=np.float32)
    dense_scores = (
        np.asarray(state.dense_scores, dtype=np.float32).reshape(-1)
        if state.dense_scores is not None
        else (passage_embeddings @ passage_query_embedding.T).astype(np.float32)
    )
    smoothed_scores = (smoothed_embeddings @ passage_query_embedding.T).astype(np.float32)
    dense_rank = np.argsort(dense_scores)[::-1].tolist()
    smoothed_rank = np.argsort(smoothed_scores)[::-1].tolist()

    hippo_rank = [int(idx) for idx in state.hippo_ranked_indices]
    hippo_score_map = {int(idx): float(score) for idx, score in state.hippo_score_map.items()}
    trace_state = state.trace_state

    if not hippo_rank:
        final_indices = np.asarray(dense_rank[:final_k], dtype=np.int64)
        final_scores = dense_scores[final_indices].astype(np.float32)
        return StrongestResult(
            final_doc_indices=final_indices,
            final_scores=final_scores,
            candidate_indices=final_indices,
            local_reset_scores=final_scores,
            anchor_prior=min_max_normalize(final_scores),
            smoothed_rank=smoothed_rank,
            dense_rank=dense_rank,
            pool_order_indices=np.asarray(list(final_indices) + [idx for idx in dense_rank if idx not in set(final_indices.tolist())], dtype=np.int64),
            score_map={int(idx): float(score) for idx, score in zip(final_indices.tolist(), final_scores.tolist())},
            trace={"status": "dense_fallback_no_hippo_rank"},
        )

    if trace_state.retrieval_mode != "graph":
        final_indices, final_scores, score_map = fallback_to_hippo_head(docs, hippo_rank, hippo_score_map, final_k)
        pool_order = list(final_indices.tolist()) + [idx for idx in hippo_rank if idx not in set(final_indices.tolist())]
        return StrongestResult(
            final_doc_indices=final_indices,
            final_scores=final_scores,
            candidate_indices=np.asarray(hippo_rank[:candidate_k], dtype=np.int64),
            local_reset_scores=np.asarray([hippo_score_map.get(idx, 0.0) for idx in hippo_rank[:candidate_k]], dtype=np.float32),
            anchor_prior=min_max_normalize(final_scores),
            smoothed_rank=smoothed_rank,
            dense_rank=dense_rank,
            pool_order_indices=np.asarray(pool_order, dtype=np.int64),
            score_map=score_map,
            trace={"status": "hippo_fallback_non_graph"},
        )

    if config.union_mode != "standard":
        raise ValueError(f"Unsupported strongest union_mode: {config.union_mode}")

    candidate_indices = select_union_candidate_indices(
        primary_indices=hippo_rank[:hippo_head_k],
        secondary_indices=smoothed_rank[:smoothed_union_k],
        fallback_indices=hippo_rank + smoothed_rank + dense_rank,
        candidate_k=candidate_k,
    )
    if candidate_indices.size == 0:
        final_indices, final_scores, score_map = fallback_to_hippo_head(docs, hippo_rank, hippo_score_map, final_k)
        return StrongestResult(
            final_doc_indices=final_indices,
            final_scores=final_scores,
            candidate_indices=np.zeros(0, dtype=np.int64),
            local_reset_scores=np.zeros(0, dtype=np.float32),
            anchor_prior=np.zeros(0, dtype=np.float32),
            smoothed_rank=smoothed_rank,
            dense_rank=dense_rank,
            pool_order_indices=final_indices,
            score_map=score_map,
            trace={"status": "hippo_fallback_empty_candidate_union"},
        )

    local_reset_scores, local_passage_prior = build_true_head_local_reset(
        incidence=state.incidence,
        candidate_indices=candidate_indices,
        passage_prior=trace_state.passage_prior,
        entity_prior=trace_state.entity_prior,
    )
    if not np.any(local_reset_scores > 0):
        final_indices, final_scores, score_map = fallback_to_hippo_head(docs, hippo_rank, hippo_score_map, final_k)
        return StrongestResult(
            final_doc_indices=final_indices,
            final_scores=final_scores,
            candidate_indices=candidate_indices,
            local_reset_scores=local_reset_scores,
            anchor_prior=np.zeros_like(local_reset_scores),
            smoothed_rank=smoothed_rank,
            dense_rank=dense_rank,
            pool_order_indices=np.asarray(list(final_indices.tolist()) + [idx for idx in hippo_rank if idx not in set(final_indices.tolist())], dtype=np.int64),
            score_map=score_map,
            trace={"status": "hippo_fallback_empty_local_reset"},
        )

    local_adj = build_true_head_passage_adjacency(
        incidence=state.incidence,
        candidate_indices=candidate_indices,
        entity_prior=trace_state.entity_prior,
    )
    if config.suppression_variant == "topology_actual":
        local_adj = apply_topology_suppression(
            local_adjacency=local_adj,
            candidate_indices=candidate_indices,
            redundancy_topology_features=state.topology_redundancy_features,
        )
    elif config.suppression_variant != "topology":
        raise ValueError(f"Unsupported strongest suppression_variant: {config.suppression_variant}")

    anchor_prior = (
        min_max_normalize(local_passage_prior)
        if np.any(local_passage_prior > 0)
        else np.zeros_like(local_passage_prior, dtype=np.float32)
    )
    local_adj = add_anchor_self_loops(local_adj, anchor_prior)
    ppr_scores = run_local_ppr(
        local_reset_scores,
        local_adj,
        gamma=analysis_module.gamma,
        max_iter=analysis_module.max_iter,
        tol=analysis_module.tol,
    )
    rerank_order = np.argsort(ppr_scores)[::-1]
    ranked_candidate_indices = candidate_indices[rerank_order]
    final_indices = ranked_candidate_indices[:final_k]
    final_scores = ppr_scores[rerank_order][:final_k].astype(np.float32)
    ranked_candidate_scores = ppr_scores[rerank_order].astype(np.float32)

    seen = set(ranked_candidate_indices.tolist())
    pool_order = list(ranked_candidate_indices.tolist()) + [idx for idx in hippo_rank + dense_rank if idx not in seen]
    score_map = {int(idx): float(score) for idx, score in zip(ranked_candidate_indices.tolist(), ranked_candidate_scores.tolist())}
    return StrongestResult(
        final_doc_indices=np.asarray(final_indices, dtype=np.int64),
        final_scores=np.asarray(final_scores, dtype=np.float32),
        candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
        local_reset_scores=np.asarray(local_reset_scores, dtype=np.float32),
        anchor_prior=np.asarray(anchor_prior, dtype=np.float32),
        smoothed_rank=smoothed_rank,
        dense_rank=dense_rank,
        pool_order_indices=np.asarray(pool_order, dtype=np.int64),
        score_map=score_map,
        trace={
            "status": "ok",
            "candidate_indices": candidate_indices.tolist(),
            "reranked_candidate_indices": ranked_candidate_indices.tolist(),
            "final_doc_indices": np.asarray(final_indices, dtype=np.int64).tolist(),
            "dense_rank_head": dense_rank[: min(len(dense_rank), 10)],
            "smoothed_rank_head": smoothed_rank[: min(len(smoothed_rank), 10)],
            "local_reset_mass": float(np.sum(local_reset_scores)),
            "anchor_prior_mass": float(np.sum(anchor_prior)),
            "suppression_variant": str(config.suppression_variant),
            "union_mode": str(config.union_mode),
        },
    )


def run_strongest_sidecar(
    state: StrongestBaselineState,
    config: StrongestConfig,
    analysis_module: StrongestAnalysisModule | None = None,
) -> StrongestResult:
    if analysis_module is None:
        analysis_module = StrongestAnalysisModule(gamma=float(config.gamma))
    return rank_hippo_head_query_conditioned(
        analysis_module=analysis_module,
        state=state,
        config=config,
    )
