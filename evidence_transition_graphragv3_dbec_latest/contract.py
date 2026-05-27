"""Contract for the frozen ETv3 pool + baseline-stable DBEC/DAEC selector line."""

from __future__ import annotations

from typing import Mapping


METHOD_FAMILY = "evidence_transition_graphrag"
METHOD_VERSION = "v3_dbec_latest"
METHOD_NAME = "evidence_transition_graphragv3_dbec_latest"
BASE_METHOD_NAME = "evidence_transition_graphragv3_variable_flow"
SELECTOR_NAME = "daec_noisyor_safe_llm"
POOL_SOURCE_NAME = "etv3_pool100"
DEFAULT_SAFE_MIN_OBJECTIVE_GAIN = 0.0
DEFAULT_SAFE_MIN_SWAP_GAIN = 0.000001
DEFAULT_SAFE_MAX_SWAPS = 2
DEFAULT_SAFE_PRESERVE_TOP_M = 1

METHOD_CONTRACT: Mapping[str, object] = {
    "method_family": METHOD_FAMILY,
    "method_version": METHOD_VERSION,
    "paper_facing_method_name": METHOD_NAME,
    "base_retriever": BASE_METHOD_NAME,
    "method_object": "baseline_stable_dbec_daec_local_edit_over_frozen_etv3_candidate_pool",
    "candidate_pool": "frozen_etv3_top5_prefix_then_etv3_candidate_universe_pool100",
    "selector": SELECTOR_NAME,
    "selector_objective": "llm_decomposition_and_binding_plus_frozen_binding_noisy_or_coverage",
    "projection": "start_from_etv3_top5_then_apply_strict_positive_dbec_objective_swaps",
    "safe_min_objective_gain": DEFAULT_SAFE_MIN_OBJECTIVE_GAIN,
    "safe_min_swap_gain": DEFAULT_SAFE_MIN_SWAP_GAIN,
    "safe_max_swaps": DEFAULT_SAFE_MAX_SWAPS,
    "safe_preserve_top_m": DEFAULT_SAFE_PRESERVE_TOP_M,
    "qa_context": "selector_top5_passages",
    "changes_candidate_generation": False,
    "changes_top5_selection": True,
    "changes_reader_budget": False,
    "uses_llm_decomposition": True,
    "uses_llm_binding": True,
    "uses_baseline_stable_projection": True,
    "uses_identifiability_gate": False,
    "uses_graph_prior": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_prompt_objective_patch": False,
    "uses_hand_built_lexical_rules": False,
    "diagnostic_not_etv3_retrieval_mainline": True,
}
