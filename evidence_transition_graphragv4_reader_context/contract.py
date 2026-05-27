"""Contract for the Evidence Transition GraphRAG v4 reader-context diagnostic."""

from __future__ import annotations

from typing import Mapping


METHOD_FAMILY = "evidence_transition_graphrag"
METHOD_VERSION = "v4_reader_context"
METHOD_NAME = "evidence_transition_graphragv4_reader_context"
METHOD_TRACE_NAME = "evidence_transition_reader_context_v4"
BASE_METHOD_NAME = "evidence_transition_graphragv3_variable_flow"
QA_DOC_KEY = "evidence_transition_graphragv4_reader_context_doc_indices_top10"
DEFAULT_READER_CONTEXT_K = 10

METHOD_CONTRACT: Mapping[str, object] = {
    "method_family": METHOD_FAMILY,
    "method_version": METHOD_VERSION,
    "paper_facing_method_name": METHOD_NAME,
    "base_retriever": BASE_METHOD_NAME,
    "method_object": "reader_context_budget_diagnostic_over_frozen_etv3_retrieval",
    "retrieval": "frozen_etv3_top5_unchanged",
    "reader_context": "current_top5_then_candidate_prefix_unique10",
    "changes_candidate_generation": False,
    "changes_top5_selection": False,
    "changes_reader_budget": True,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_llm_query_compiler": False,
    "uses_prompt_objective_patch": False,
    "diagnostic_not_graph_readout_mainline": True,
}
