from .types import PlannerDecision, PlannerState


def should_stop(best_decision: PlannerDecision, state: PlannerState, context, config) -> bool:
    if state.step_id >= config.planner_max_steps:
        return True

    if best_decision.action.action_type == "stop":
        top_belief = max(state.belief.values()) if state.belief else 0.0
        return top_belief >= config.planner_belief_stop_threshold or len(state.selected_doc_ids) >= context.num_to_retrieve

    if best_decision.score < config.planner_min_action_score:
        enough_docs = len(state.selected_doc_ids) >= context.num_to_retrieve
        return enough_docs

    return False
