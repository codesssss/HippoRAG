from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy import sparse as sp


SOURCE_REPO_PATH = "/mnt/nvme/code/XiaoRAG"
SOURCE_COMMIT_SHA = "1dc9179373d9df83c6611edf2a71faf68fd20997"


@dataclass
class StrongestConfig:
    candidate_k: int
    final_k: int
    hippo_head_k: int
    smoothed_union_k: int
    gamma: float = 0.15
    union_mode: str = "standard"
    suppression_variant: str = "topology"
    source_repo_path: str = SOURCE_REPO_PATH
    source_commit_sha: str = SOURCE_COMMIT_SHA


@dataclass
class StrongestTraceState:
    query: str
    retrieval_mode: str
    passage_prior: np.ndarray
    entity_prior: np.ndarray
    raw_trace: Any = None


@dataclass
class StrongestBaselineState:
    query: str
    docs: Sequence[str]
    passage_query_embedding: np.ndarray
    passage_embeddings: np.ndarray
    smoothed_embeddings: np.ndarray
    incidence: sp.csr_matrix
    topology_redundancy_features: sp.csr_matrix
    hippo_ranked_indices: Sequence[int]
    hippo_score_map: Dict[int, float]
    trace_state: StrongestTraceState
    dense_scores: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrongestResult:
    final_doc_indices: np.ndarray
    final_scores: np.ndarray
    candidate_indices: np.ndarray
    local_reset_scores: np.ndarray
    anchor_prior: np.ndarray
    smoothed_rank: List[int]
    dense_rank: List[int]
    pool_order_indices: np.ndarray
    score_map: Dict[int, float]
    trace: Dict[str, Any]


@dataclass
class StrongestAnalysisModule:
    """Small, deterministic helper surface used by strongest sidecar."""

    gamma: float = 0.15
    max_iter: int = 64
    tol: float = 1e-6

