"""Contract for the canonical EvidenceFlow ETv4 + PCEC implementation.

EvidenceFlow is the engineering package for the current evidence-transition
retrieval stack.  It keeps the PCEC paper-facing readout name for historical
result traceability while exposing ``evidenceflow`` as the canonical import and
artifact method name.
"""

from __future__ import annotations

from typing import Mapping


METHOD_FAMILY = "evidence_transition_graphrag"
METHOD_VERSION = "v4_composition"
METHOD_NAME = "evidenceflow"
PAPER_FACING_METHOD_NAME = "Preservation-Constrained Evidence Composition"
BASE_EXPANDER_NAME = "evidence_transition_graphragv3_variable_flow"
FROZEN_BASE_EXPANDER_PACKAGE = "evidenceflow.frozen_etv3_variable_flow"
FROZEN_BASE_EXPANDER_FREEZE_DATE = "2026-05-10"
READOUT_NAME = "prefix_residual_admission"
SELECTOR_NAME = "daec_noisyor_safe_llm"

DEFAULT_READER_BUDGET_K = 5
DEFAULT_PREFIX_BUDGET_M = 4
DEFAULT_POOL_K = 100
DEFAULT_SAFE_MIN_OBJECTIVE_GAIN = 0.0
DEFAULT_SAFE_MIN_SWAP_GAIN = 0.000001
DEFAULT_DTC_BINDING_MAX_CANDIDATES = 5
DEFAULT_ORDERING_POLICY = "preserve_et_prefix_order_and_fill_residual_slot"
DEFAULT_RETENTION_PROXY = "et_transition_rank_prefix"
DEFAULT_ADMISSION_OBJECTIVE = "dbec_frozen_binding_noisy_or_marginal_coverage"


METHOD_CONTRACT: Mapping[str, object] = {
    "method_family": METHOD_FAMILY,
    "method_version": METHOD_VERSION,
    "method_name": METHOD_NAME,
    "paper_facing_method_name": PAPER_FACING_METHOD_NAME,
    "base_expander": BASE_EXPANDER_NAME,
    "frozen_base_expander_package": FROZEN_BASE_EXPANDER_PACKAGE,
    "frozen_base_expander_freeze_date": FROZEN_BASE_EXPANDER_FREEZE_DATE,
    "readout_object": READOUT_NAME,
    "method_object": "preservation_constrained_fixed_budget_evidence_composition",
    "optimization_form": "argmax_U_bstar_S_subject_to_prefix_m_S0_subset_S_and_cardinality_K",
    "binding_protocol": "dbec_best_binding_frozen_before_constrained_readout",
    "retention_proxy": DEFAULT_RETENTION_PROXY,
    "admission_objective": DEFAULT_ADMISSION_OBJECTIVE,
    "selector": SELECTOR_NAME,
    "default_reader_budget_k": DEFAULT_READER_BUDGET_K,
    "default_prefix_budget_m": DEFAULT_PREFIX_BUDGET_M,
    "default_residual_budget": DEFAULT_READER_BUDGET_K - DEFAULT_PREFIX_BUDGET_M,
    "default_pool_k": DEFAULT_POOL_K,
    "default_safe_min_objective_gain": DEFAULT_SAFE_MIN_OBJECTIVE_GAIN,
    "default_safe_min_swap_gain": DEFAULT_SAFE_MIN_SWAP_GAIN,
    "ordering_policy": DEFAULT_ORDERING_POLICY,
    "qa_context": "fixed_top5_passages",
    "train_free": True,
    "changes_candidate_generation": False,
    "changes_reader_budget": False,
    "changes_readout_objective": True,
    "uses_et_transition_rank_prefix": True,
    "uses_dbec_frozen_binding_noisy_or": True,
    "uses_hard_preservation_constraint": True,
    "uses_lambda_regularization": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_query_adaptive_gate": False,
    "uses_cross_signal_agreement_retention": False,
    "uses_learned_reranker": False,
    "uses_prompt_objective_patch": False,
}


def residual_budget(reader_budget_k: int, prefix_budget_m: int) -> int:
    """Return the residual admission budget after preserving the ET prefix."""
    validate_budgets(reader_budget_k=reader_budget_k, prefix_budget_m=prefix_budget_m)
    return int(reader_budget_k) - int(prefix_budget_m)


def validate_budgets(*, reader_budget_k: int, prefix_budget_m: int) -> None:
    """Validate PCEC cardinality and preservation budgets."""
    if int(reader_budget_k) <= 0:
        raise ValueError(f"reader_budget_k must be positive, got {reader_budget_k}")
    if int(prefix_budget_m) < 0:
        raise ValueError(f"prefix_budget_m must be non-negative, got {prefix_budget_m}")
    if int(prefix_budget_m) > int(reader_budget_k):
        raise ValueError(
            f"prefix_budget_m cannot exceed reader_budget_k: {prefix_budget_m} > {reader_budget_k}"
        )
