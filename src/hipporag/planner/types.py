from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np


PlannerActionType = Literal["dense_seed", "fact_seed", "expand_entity", "inspect_passage", "stop"]


@dataclass(frozen=True)
class PlannerAction:
    action_type: PlannerActionType
    key: str
    label: str
    estimated_cost: float
    metadata: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class PlannerHistoryEntry:
    step_id: int
    action_key: str
    action_type: PlannerActionType
    score: float
    added_doc_ids: List[int]
    belief: Dict[str, float]


@dataclass
class PlannerState:
    query: str
    belief: Dict[str, float]
    selected_doc_ids: Set[int] = field(default_factory=set)
    doc_scores: Dict[int, float] = field(default_factory=dict)
    executed_actions: Set[str] = field(default_factory=set)
    step_id: int = 0
    history: List[PlannerHistoryEntry] = field(default_factory=list)


@dataclass
class PlannerContext:
    query: str
    num_to_retrieve: int
    dense_doc_ids: np.ndarray
    dense_doc_scores: np.ndarray
    fact_doc_ids: np.ndarray
    fact_doc_scores: np.ndarray
    fact_indices: List[int]
    facts: List[Tuple[str, str, str]]
    query_fact_scores: np.ndarray
    entity_candidates: List[Tuple[str, float]]
    entity_to_doc_ids: Dict[str, List[int]]
    dense_score_by_doc_id: Dict[int, float]
    fact_score_by_doc_id: Dict[int, float]


@dataclass
class PlannerDecision:
    action: PlannerAction
    score: float
    info_gain: float
    relevance: float
    novelty: float
    cost_penalty: float
    posterior_belief: Dict[str, float]
    candidate_doc_ids: List[int]


@dataclass
class PlannerResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    state: PlannerState
