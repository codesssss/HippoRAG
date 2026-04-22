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
    rerank_mode: str = "standard"
    gbc_protected_anchor_k: int = 2
    gbc_head_coverage_k: int = 5
    gbc_top_passage_pool_k: int = 24
    gbc_frontier_bonus_k: int = 6
    gbc_bonus_weight: float = 1.0
    ras_enabled: bool = False
    ras_prefix_guard_k: int = 3
    ras_requirement_max_units: int = 4
    ras_enable_conflict_veto: bool = True
    ras_core_support_min_eligible: bool = True
    ras_extractor_mode: str = "rule"
    ras_support_mode: str = "lexical"
    ras_embedding_probe_threshold: float = 0.35
    ras_trace_enabled: bool = True
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


@dataclass
class RequirementUnit:
    unit_id: str
    tier: str
    anchor_entities: List[str]
    slot_family: str
    expected_answer_type: str
    bridge_targets: List[str] = field(default_factory=list)
    lexical_cues: List[str] = field(default_factory=list)
    is_single_valued: bool = True
    comparator: str | None = None
    raw_slot_family: str = ""
    raw_anchor: str = ""
    anchor_status: str = "grounded"
    bridge_ref_label: str | None = None


@dataclass
class RequirementCoverageTrace:
    unit_id: str
    tier: str
    slot_family: str
    head_coverage: float
    final_coverage: float
    unmet_mass: float


@dataclass
class RequirementConflictTrace:
    local_idx: int
    unit_id: str
    slot_family: str
    head_filler: str
    candidate_filler: str
    conflict_kind: str = "single_value_conflict"


@dataclass
class RASReadoutTrace:
    extractor_trace: Dict[str, Any]
    requirement_units: List[RequirementUnit]
    coverage: List[RequirementCoverageTrace]
    conflicts: List[RequirementConflictTrace]
    g_core_by_local_idx: Dict[int, float]
    g_support_by_local_idx: Dict[int, float]
    anchor_support_by_local_idx: Dict[int, float]
    base_score_by_local_idx: Dict[int, float]
    prefix_disruption_by_local_idx: Dict[int, float]
    prefix_guard_triggered: bool
    protected_prefix_before: List[int]
    protected_prefix_after: List[int]
    query_level_repair: bool
