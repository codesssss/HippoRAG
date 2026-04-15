from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse as sp


def build_true_head_passage_adjacency(
    incidence: sp.csr_matrix,
    candidate_indices: np.ndarray,
    entity_prior: np.ndarray,
) -> sp.csr_matrix:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    num_candidates = int(candidate_indices.size)
    if num_candidates == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)
    if num_candidates == 1:
        return sp.identity(1, dtype=np.float32, format="csr")

    entity_weights = np.clip(np.asarray(entity_prior, dtype=np.float32), 0.0, None)
    if not np.any(entity_weights > 0):
        return sp.identity(num_candidates, dtype=np.float32, format="csr")

    local_incidence = incidence[:, candidate_indices].tocsr()
    entity_degree = np.asarray(incidence.sum(axis=1)).reshape(-1).astype(np.float32)
    weighted_entity = entity_weights / np.maximum(entity_degree, 1.0)
    local_raw = (local_incidence.T @ sp.diags(weighted_entity) @ local_incidence).astype(np.float32).tocsr()
    local_raw.setdiag(0.0)
    local_raw.eliminate_zeros()
    if local_raw.nnz == 0:
        return sp.identity(num_candidates, dtype=np.float32, format="csr")

    passage_degree = np.asarray(local_raw.sum(axis=1)).reshape(-1).astype(np.float32)
    passage_scale = 1.0 / np.sqrt(np.maximum(passage_degree, 1.0))
    return (sp.diags(passage_scale) @ local_raw @ sp.diags(passage_scale)).tocsr()


def build_true_head_local_reset(
    incidence: sp.csr_matrix,
    candidate_indices: np.ndarray,
    passage_prior: np.ndarray,
    entity_prior: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    local_passage_prior = np.clip(np.asarray(passage_prior[candidate_indices], dtype=np.float32), 0.0, None)
    entity_weights = np.clip(np.asarray(entity_prior, dtype=np.float32), 0.0, None)
    if np.any(entity_weights > 0):
        entity_degree = np.asarray(incidence.sum(axis=1)).reshape(-1).astype(np.float32)
        projected_entity_mass = incidence[:, candidate_indices].T @ (
            entity_weights / np.maximum(entity_degree, 1.0)
        )
        projected_entity_mass = np.asarray(projected_entity_mass).reshape(-1).astype(np.float32)
    else:
        projected_entity_mass = np.zeros(len(candidate_indices), dtype=np.float32)
    return (local_passage_prior + projected_entity_mass).astype(np.float32), local_passage_prior


def add_anchor_self_loops(local_adjacency: sp.csr_matrix, anchor_prior: np.ndarray) -> sp.csr_matrix:
    if local_adjacency.shape[0] == 0:
        return local_adjacency
    local_dense = local_adjacency.toarray().astype(np.float32, copy=True)
    diagonal = np.diag(local_dense).astype(np.float32, copy=False)
    np.fill_diagonal(
        local_dense,
        diagonal + np.clip(np.asarray(anchor_prior, dtype=np.float32), 0.0, None),
    )
    return sp.csr_matrix(local_dense)


def apply_topology_suppression(
    local_adjacency: sp.csr_matrix,
    candidate_indices: np.ndarray,
    redundancy_topology_features: sp.csr_matrix,
) -> sp.csr_matrix:
    if local_adjacency.shape[0] <= 1:
        return local_adjacency

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_topology = redundancy_topology_features[candidate_indices]
    topology_similarity = np.clip((candidate_topology @ candidate_topology.T).toarray(), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(topology_similarity, 0.0)

    local_dense = local_adjacency.toarray().astype(np.float32, copy=True)
    local_dense *= (1.0 - topology_similarity)
    return sp.csr_matrix(local_dense)
