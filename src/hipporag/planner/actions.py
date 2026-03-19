from typing import List

from .types import PlannerAction, PlannerContext, PlannerState


def build_actions(state: PlannerState, context: PlannerContext, config) -> List[PlannerAction]:
    actions: List[PlannerAction] = []

    if "dense_seed" not in state.executed_actions:
        actions.append(
            PlannerAction(
                action_type="dense_seed",
                key="dense_seed",
                label="Seed from dense retrieval",
                estimated_cost=1.0,
                metadata={"doc_budget": int(config.planner_seed_doc_budget)},
            )
        )

    if context.facts and "fact_seed" not in state.executed_actions:
        actions.append(
            PlannerAction(
                action_type="fact_seed",
                key="fact_seed",
                label="Seed from fact-guided graph search",
                estimated_cost=1.2,
                metadata={"doc_budget": int(config.planner_seed_doc_budget)},
            )
        )

    for entity_key, entity_score in context.entity_candidates[: config.planner_max_entity_actions]:
        action_key = f"expand_entity::{entity_key}"
        if action_key in state.executed_actions:
            continue
        doc_ids = context.entity_to_doc_ids.get(entity_key, [])
        if not doc_ids:
            continue
        actions.append(
            PlannerAction(
                action_type="expand_entity",
                key=action_key,
                label=f"Expand entity {entity_key}",
                estimated_cost=0.75,
                metadata={
                    "entity_key": entity_key,
                    "entity_score": float(entity_score),
                    "doc_budget": int(config.planner_entity_doc_budget),
                },
            )
        )

    if state.selected_doc_ids:
        inspect_window = config.planner_seed_doc_budget + config.planner_max_inspect_passages
        dense_candidates = context.dense_doc_ids[:inspect_window]
        for doc_id in dense_candidates:
            doc_id = int(doc_id)
            action_key = f"inspect_passage::{doc_id}"
            if action_key in state.executed_actions or doc_id in state.selected_doc_ids:
                continue
            actions.append(
                PlannerAction(
                    action_type="inspect_passage",
                    key=action_key,
                    label=f"Inspect passage {doc_id}",
                    estimated_cost=0.35,
                    metadata={"doc_id": doc_id},
                )
            )

    actions.append(
        PlannerAction(
            action_type="stop",
            key="stop",
            label="Stop planning",
            estimated_cost=0.0,
            metadata={},
        )
    )

    return actions
