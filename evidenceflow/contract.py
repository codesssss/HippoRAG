"""Contract for EvidenceFlow and its legacy adapters.

The reported EvidenceFlow protocol is:

    ETv4 fact-witnessed STO pool -> PCEC native-pool readout -> top-5 reader input

The frozen ETv3 package bundled under :mod:`evidenceflow.frozen_etv3_variable_flow`
is a legacy/fresh-parity adapter.  It is not the upstream retriever for the
main Table-1 EvidenceFlow row.
"""

from __future__ import annotations

from typing import Mapping


METHOD_FAMILY = "evidence_transition_graphrag"
METHOD_VERSION = "v4_composition"
METHOD_NAME = "evidenceflow"
PAPER_FACING_METHOD_NAME = "EvidenceFlow"
READOUT_COMPONENT_NAME = "Preservation-Constrained Evidence Composition"
READOUT_COMPONENT_ABBREVIATION = "PCEC"
READOUT_NAME = "pcec_prefix_residual_admission"
SELECTOR_NAME = "daec_noisyor_safe_llm"
MAIN_PROTOCOL_NAME = "etv4_fact_witnessed_sto_pool_to_pcec_native_readout"
MAIN_ENTRYPOINT = "evidenceflow/run_native_pool.py"
MAIN_UPSTREAM_RETRIEVER_NAME = "evidence_transition_graphragv4_fact_witnessed_sto"
MAIN_POOL_PROVENANCE_KEY = "retrieval.input_method"
LEGACY_FRESH_PROTOCOL_NAME = "legacy_etv3_fresh_pcec_parity"
LEGACY_FRESH_ENTRYPOINT = "evidenceflow/run_fresh_e2e.py"
LEGACY_FRESH_EXPANDER_NAME = "evidence_transition_graphragv3_variable_flow"
LEGACY_FRESH_EXPANDER_PACKAGE = "evidenceflow.frozen_etv3_variable_flow"
LEGACY_FRESH_EXPANDER_FREEZE_DATE = "2026-05-10"

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
    "main_protocol": MAIN_PROTOCOL_NAME,
    "main_entrypoint": MAIN_ENTRYPOINT,
    "main_upstream_retriever": MAIN_UPSTREAM_RETRIEVER_NAME,
    "main_pool_provenance_key": MAIN_POOL_PROVENANCE_KEY,
    "main_pool_contract": (
        "Table-1 EvidenceFlow requires an external pool whose "
        "retrieval.input_method is evidence_transition_graphragv4_fact_witnessed_sto."
    ),
    "legacy_fresh_protocol": LEGACY_FRESH_PROTOCOL_NAME,
    "legacy_fresh_entrypoint": LEGACY_FRESH_ENTRYPOINT,
    "legacy_fresh_expander": LEGACY_FRESH_EXPANDER_NAME,
    "legacy_fresh_expander_package": LEGACY_FRESH_EXPANDER_PACKAGE,
    "legacy_fresh_expander_freeze_date": LEGACY_FRESH_EXPANDER_FREEZE_DATE,
    "legacy_fresh_expander_is_main_table_retriever": False,
    "uses_etv4_fact_witnessed_sto_pool_for_main_table": True,
    "uses_frozen_etv3_for_main_table": False,
    "readout_component": READOUT_COMPONENT_NAME,
    "readout_component_abbreviation": READOUT_COMPONENT_ABBREVIATION,
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


def pool_upstream_retriever(pool_payload: Mapping[str, object]) -> str:
    """Return the upstream retriever recorded by an exported pool payload."""
    retrieval = pool_payload.get("retrieval")
    if not isinstance(retrieval, Mapping):
        return ""
    return str(retrieval.get("input_method") or "")


def is_main_evidenceflow_pool(pool_payload: Mapping[str, object]) -> bool:
    """Return whether a pool is the main ETv4 input expected by EvidenceFlow."""
    return pool_upstream_retriever(pool_payload) == MAIN_UPSTREAM_RETRIEVER_NAME


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
