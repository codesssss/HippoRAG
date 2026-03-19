import math
from typing import Dict


EPS = 1e-8


def build_initial_belief() -> Dict[str, float]:
    belief = {
        "hybrid_path": 0.4,
        "fact_graph_path": 0.25,
        "dense_path": 0.2,
        "entity_expansion_path": 0.1,
        "insufficient_evidence": 0.05,
    }
    total = sum(belief.values())
    return {key: value / total for key, value in belief.items()}


def action_hypothesis_compatibility(action_type: str) -> Dict[str, float]:
    if action_type == "dense_seed":
        return {
            "dense_path": 1.35,
            "hybrid_path": 0.95,
            "entity_expansion_path": 0.2,
            "fact_graph_path": -0.15,
            "insufficient_evidence": -0.35,
        }
    if action_type == "fact_seed":
        return {
            "fact_graph_path": 1.45,
            "hybrid_path": 1.0,
            "entity_expansion_path": 0.35,
            "dense_path": -0.1,
            "insufficient_evidence": -0.35,
        }
    if action_type == "expand_entity":
        return {
            "entity_expansion_path": 1.4,
            "hybrid_path": 0.9,
            "fact_graph_path": 0.45,
            "dense_path": 0.1,
            "insufficient_evidence": -0.2,
        }
    if action_type == "inspect_passage":
        return {
            "hybrid_path": 0.95,
            "dense_path": 0.5,
            "fact_graph_path": 0.5,
            "entity_expansion_path": 0.3,
            "insufficient_evidence": -0.15,
        }
    return {
        "insufficient_evidence": 0.5,
        "hybrid_path": -0.2,
        "dense_path": -0.2,
        "fact_graph_path": -0.2,
        "entity_expansion_path": -0.2,
    }


def update_belief(prior: Dict[str, float], compatibility: Dict[str, float]) -> Dict[str, float]:
    logits = {}
    for hypothesis, prior_prob in prior.items():
        logits[hypothesis] = math.log(max(prior_prob, EPS)) + compatibility.get(hypothesis, 0.0)

    max_logit = max(logits.values())
    exp_values = {key: math.exp(value - max_logit) for key, value in logits.items()}
    total = sum(exp_values.values())
    return {key: value / total for key, value in exp_values.items()}


def entropy(probabilities: Dict[str, float]) -> float:
    return -sum(prob * math.log(max(prob, EPS)) for prob in probabilities.values())
