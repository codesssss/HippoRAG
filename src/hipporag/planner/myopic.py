import logging
from typing import Dict, List, Tuple

import numpy as np

from ..utils.misc_utils import compute_mdhash_id, min_max_normalize
from .actions import build_actions
from .hypothesis import build_initial_belief
from .scoring import score_action
from .stop import should_stop
from .types import PlannerContext, PlannerHistoryEntry, PlannerResult, PlannerState

logger = logging.getLogger(__name__)


class HippoRAGMyopicPlanner:
    def __init__(self, hipporag):
        self.hipporag = hipporag

    def retrieve(self,
                 query: str,
                 num_to_retrieve: int,
                 query_fact_scores: np.ndarray,
                 top_k_fact_indices: List[int],
                 top_k_facts: List[Tuple[str, str, str]]) -> PlannerResult:
        dense_doc_ids, dense_doc_scores = self.hipporag.dense_passage_retrieval(query)

        if top_k_facts:
            fact_doc_ids, fact_doc_scores = self.hipporag.graph_search_with_fact_entities(
                query=query,
                link_top_k=self.hipporag.global_config.linking_top_k,
                query_fact_scores=query_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=self.hipporag.global_config.passage_node_weight,
            )
        else:
            fact_doc_ids = np.array([], dtype=int)
            fact_doc_scores = np.array([], dtype=float)

        context = self._build_context(
            query=query,
            num_to_retrieve=num_to_retrieve,
            dense_doc_ids=dense_doc_ids,
            dense_doc_scores=dense_doc_scores,
            fact_doc_ids=fact_doc_ids,
            fact_doc_scores=fact_doc_scores,
            query_fact_scores=query_fact_scores,
            top_k_fact_indices=top_k_fact_indices,
            top_k_facts=top_k_facts,
        )

        state = PlannerState(query=query, belief=build_initial_belief())

        while state.step_id < self.hipporag.global_config.planner_max_steps:
            actions = build_actions(state, context, self.hipporag.global_config)
            decisions = [score_action(action, state, context, self.hipporag.global_config) for action in actions]
            best_decision = max(decisions, key=lambda decision: decision.score)

            logger.info(
                "Planner step=%s action=%s score=%.4f gain=%.4f relevance=%.4f novelty=%.4f",
                state.step_id,
                best_decision.action.key,
                best_decision.score,
                best_decision.info_gain,
                best_decision.relevance,
                best_decision.novelty,
            )

            if should_stop(best_decision, state, context, self.hipporag.global_config):
                break

            added_doc_ids = self._execute_action(best_decision, state, context)
            state.belief = best_decision.posterior_belief
            state.executed_actions.add(best_decision.action.key)
            state.history.append(
                PlannerHistoryEntry(
                    step_id=state.step_id,
                    action_key=best_decision.action.key,
                    action_type=best_decision.action.action_type,
                    score=best_decision.score,
                    added_doc_ids=added_doc_ids,
                    belief=dict(state.belief),
                )
            )
            state.step_id += 1

            if len(state.selected_doc_ids) >= num_to_retrieve and max(state.belief.values()) >= self.hipporag.global_config.planner_belief_stop_threshold:
                break

        sorted_doc_ids, sorted_doc_scores = self._finalize_docs(state, context, num_to_retrieve)
        return PlannerResult(sorted_doc_ids=sorted_doc_ids, sorted_doc_scores=sorted_doc_scores, state=state)

    def _build_context(self,
                       query: str,
                       num_to_retrieve: int,
                       dense_doc_ids: np.ndarray,
                       dense_doc_scores: np.ndarray,
                       fact_doc_ids: np.ndarray,
                       fact_doc_scores: np.ndarray,
                       query_fact_scores: np.ndarray,
                       top_k_fact_indices: List[int],
                       top_k_facts: List[Tuple[str, str, str]]) -> PlannerContext:
        dense_score_by_doc_id = {int(doc_id): float(score) for doc_id, score in zip(dense_doc_ids.tolist(), dense_doc_scores.tolist())}
        fact_score_by_doc_id = {int(doc_id): float(score) for doc_id, score in zip(fact_doc_ids.tolist(), fact_doc_scores.tolist())}

        entity_scores: Dict[str, List[float]] = {}
        for rank, fact in enumerate(top_k_facts):
            if len(fact) != 3:
                continue
            fact_index = top_k_fact_indices[rank]
            fact_score = float(query_fact_scores[fact_index]) if np.size(query_fact_scores) > 0 else 0.0
            for entity_text in (str(fact[0]).lower(), str(fact[2]).lower()):
                entity_key = compute_mdhash_id(content=entity_text, prefix="entity-")
                entity_scores.setdefault(entity_key, []).append(fact_score)

        entity_candidates = sorted(
            ((entity_key, float(np.mean(scores))) for entity_key, scores in entity_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )

        entity_to_doc_ids: Dict[str, List[int]] = {}
        for entity_key, _ in entity_candidates[: self.hipporag.global_config.planner_max_entity_actions * 2]:
            chunk_ids = list(self.hipporag.ent_node_to_chunk_ids.get(entity_key, set()))
            doc_ids = []
            for chunk_id in chunk_ids:
                doc_idx = self.hipporag.passage_node_key_to_doc_idx.get(chunk_id)
                if doc_idx is not None:
                    doc_ids.append(int(doc_idx))
            ranked_doc_ids = sorted(
                set(doc_ids),
                key=lambda doc_id: max(dense_score_by_doc_id.get(doc_id, 0.0), fact_score_by_doc_id.get(doc_id, 0.0)),
                reverse=True,
            )
            if ranked_doc_ids:
                entity_to_doc_ids[entity_key] = ranked_doc_ids

        return PlannerContext(
            query=query,
            num_to_retrieve=num_to_retrieve,
            dense_doc_ids=dense_doc_ids,
            dense_doc_scores=dense_doc_scores,
            fact_doc_ids=fact_doc_ids,
            fact_doc_scores=fact_doc_scores,
            fact_indices=top_k_fact_indices,
            facts=top_k_facts,
            query_fact_scores=query_fact_scores,
            entity_candidates=entity_candidates,
            entity_to_doc_ids=entity_to_doc_ids,
            dense_score_by_doc_id=dense_score_by_doc_id,
            fact_score_by_doc_id=fact_score_by_doc_id,
        )

    def _execute_action(self, decision, state: PlannerState, context: PlannerContext) -> List[int]:
        added_doc_ids: List[int] = []
        action = decision.action
        if action.action_type in {"dense_seed", "fact_seed", "expand_entity"}:
            for rank, doc_id in enumerate(decision.candidate_doc_ids):
                doc_id = int(doc_id)
                source_score = self._candidate_score(action.action_type, doc_id, context)
                blended_score = source_score * (1.0 - 0.02 * min(rank, 10))
                previous_score = state.doc_scores.get(doc_id, 0.0)
                state.doc_scores[doc_id] = max(previous_score, blended_score)
                if doc_id not in state.selected_doc_ids:
                    state.selected_doc_ids.add(doc_id)
                    added_doc_ids.append(doc_id)
        elif action.action_type == "inspect_passage":
            doc_id = int(action.metadata["doc_id"])
            score = self._candidate_score("inspect_passage", doc_id, context)
            previous_score = state.doc_scores.get(doc_id, 0.0)
            state.doc_scores[doc_id] = max(previous_score, score)
            if doc_id not in state.selected_doc_ids:
                state.selected_doc_ids.add(doc_id)
                added_doc_ids.append(doc_id)

        return added_doc_ids

    def _candidate_score(self, action_type: str, doc_id: int, context: PlannerContext) -> float:
        dense_score = context.dense_score_by_doc_id.get(doc_id, 0.0)
        fact_score = context.fact_score_by_doc_id.get(doc_id, 0.0)
        if action_type == "dense_seed":
            return dense_score
        if action_type == "fact_seed":
            return max(fact_score, 0.5 * dense_score)
        if action_type == "expand_entity":
            return 0.6 * dense_score + 0.4 * fact_score
        return max(dense_score, fact_score)

    def _finalize_docs(self, state: PlannerState, context: PlannerContext, num_to_retrieve: int):
        merged_scores = dict(state.doc_scores)
        dense_fallback_limit = max(num_to_retrieve * 3, self.hipporag.global_config.planner_seed_doc_budget)

        for rank, doc_id in enumerate(context.dense_doc_ids[:dense_fallback_limit]):
            doc_id = int(doc_id)
            dense_score = context.dense_score_by_doc_id.get(doc_id, 0.0)
            previous_score = merged_scores.get(doc_id, 0.0)
            merged_scores[doc_id] = max(previous_score, dense_score * self.hipporag.global_config.planner_dense_fallback_weight)

        for rank, doc_id in enumerate(context.fact_doc_ids[:dense_fallback_limit]):
            doc_id = int(doc_id)
            fact_score = context.fact_score_by_doc_id.get(doc_id, 0.0)
            previous_score = merged_scores.get(doc_id, 0.0)
            merged_scores[doc_id] = max(previous_score, fact_score)

        if not merged_scores:
            return context.dense_doc_ids[:num_to_retrieve], context.dense_doc_scores[:num_to_retrieve]

        sorted_items = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
        top_items = sorted_items[:num_to_retrieve]
        sorted_doc_ids = np.array([doc_id for doc_id, _ in top_items], dtype=int)
        sorted_doc_scores = np.array([score for _, score in top_items], dtype=float)

        if len(sorted_doc_scores) > 1:
            sorted_doc_scores = min_max_normalize(sorted_doc_scores)

        return sorted_doc_ids, sorted_doc_scores
