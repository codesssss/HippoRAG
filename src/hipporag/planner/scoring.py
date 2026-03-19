from typing import Dict, List

import numpy as np

from .hypothesis import action_hypothesis_compatibility, entropy, update_belief
from .types import PlannerAction, PlannerContext, PlannerDecision, PlannerState


def candidate_doc_ids_for_action(action: PlannerAction, context: PlannerContext) -> List[int]:
    if action.action_type == "dense_seed":
        budget = int(action.metadata.get("doc_budget", 0))
        return [int(doc_id) for doc_id in context.dense_doc_ids[:budget]]
    if action.action_type == "fact_seed":
        budget = int(action.metadata.get("doc_budget", 0))
        return [int(doc_id) for doc_id in context.fact_doc_ids[:budget]]
    if action.action_type == "expand_entity":
        entity_key = str(action.metadata.get("entity_key"))
        budget = int(action.metadata.get("doc_budget", 0))
        return context.entity_to_doc_ids.get(entity_key, [])[:budget]
    if action.action_type == "inspect_passage":
        doc_id = action.metadata.get("doc_id")
        return [int(doc_id)] if doc_id is not None else []
    return []


def _relevance_score(action: PlannerAction, candidate_doc_ids: List[int], context: PlannerContext) -> float:
    if action.action_type == "dense_seed":
        if len(context.dense_doc_scores) == 0:
            return 0.0
        top_scores = context.dense_doc_scores[: max(1, len(candidate_doc_ids))]
        return float(np.mean(top_scores))

    if action.action_type == "fact_seed":
        if len(context.fact_doc_scores) == 0:
            return 0.0
        top_scores = context.fact_doc_scores[: max(1, len(candidate_doc_ids))]
        return float(np.mean(top_scores))

    if action.action_type == "expand_entity":
        entity_score = float(action.metadata.get("entity_score", 0.0))
        if not candidate_doc_ids:
            return entity_score
        doc_scores = [context.dense_score_by_doc_id.get(doc_id, 0.0) for doc_id in candidate_doc_ids]
        return float(0.6 * entity_score + 0.4 * np.mean(doc_scores))

    if action.action_type == "inspect_passage":
        doc_id = int(action.metadata.get("doc_id", -1))
        return float(max(context.dense_score_by_doc_id.get(doc_id, 0.0), context.fact_score_by_doc_id.get(doc_id, 0.0)))

    return 0.0


def score_action(action: PlannerAction, state: PlannerState, context: PlannerContext, config) -> PlannerDecision:
    candidate_doc_ids = candidate_doc_ids_for_action(action, context)
    unseen_doc_ids = [doc_id for doc_id in candidate_doc_ids if doc_id not in state.selected_doc_ids]
    novelty = float(len(unseen_doc_ids) / max(len(candidate_doc_ids), 1))
    relevance = _relevance_score(action, candidate_doc_ids, context)

    posterior_belief = update_belief(state.belief, action_hypothesis_compatibility(action.action_type))
    info_gain = entropy(state.belief) - entropy(posterior_belief)
    cost_penalty = action.estimated_cost * config.planner_cost_weight

    score = (
        info_gain * config.planner_info_gain_weight
        + relevance * config.planner_relevance_weight
        + novelty * config.planner_novelty_weight
        - cost_penalty
    )

    if state.step_id == 0:
        if action.action_type == "fact_seed":
            score += 0.25
        elif action.action_type == "dense_seed":
            score += 0.15
        elif action.action_type == "expand_entity":
            score += 0.10

    if action.action_type in {"dense_seed", "fact_seed", "expand_entity"}:
        coverage_bonus = min(len(unseen_doc_ids), 5) / 5.0
        score += 0.15 * coverage_bonus

    if action.action_type == "inspect_passage":
        score -= 0.15

    if action.action_type == "stop":
        top_belief = max(state.belief.values()) if state.belief else 0.0
        enough_docs = len(state.selected_doc_ids) >= context.num_to_retrieve
        score = top_belief - (0.15 if not enough_docs else 0.0)
        info_gain = 0.0
        relevance = 0.0
        novelty = 0.0
        posterior_belief = dict(state.belief)
        candidate_doc_ids = []
        cost_penalty = 0.0

    return PlannerDecision(
        action=action,
        score=float(score),
        info_gain=float(info_gain),
        relevance=float(relevance),
        novelty=float(novelty),
        cost_penalty=float(cost_penalty),
        posterior_belief=posterior_belief,
        candidate_doc_ids=candidate_doc_ids,
    )
