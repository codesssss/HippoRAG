"""Contract for the Evidence Transition GraphRAG v3 variable-flow line."""

from __future__ import annotations

from typing import Mapping


METHOD_FAMILY = "evidence_transition_graphrag"
METHOD_VERSION = "v3_variable_flow"
METHOD_NAME = "evidence_transition_graphragv3_variable_flow"
METHOD_TRACE_NAME = "evidence_transition_graph_native_retrieval_v3_variable_flow"
QA_DOC_KEY = "evidence_transition_graphragv3_variable_flow_doc_indices_top5"

METHOD_CONTRACT: Mapping[str, object] = {
    "method_family": METHOD_FAMILY,
    "method_version": METHOD_VERSION,
    "paper_facing_method_name": METHOD_NAME,
    "method_object": "query_rooted_variable_flow_sto_context_selection",
    "indexing": "fresh_openie_and_passage_embeddings_over_raw_passages",
    "corpus_graph": "global_sto_role_transition_graph",
    "query_graph": "dense_or_textual_entry_to_query_local_variable_flow_sto_graph",
    "retrieval": "source_authorized_sto_context_selection",
    "qa_context": "fixed_top5_passages",
    "pure_graph_only": False,
    "non_graph_signal_policy": "allowed_for_candidate_entry_and_source_prior_only",
    "uses_dense_entry": True,
    "uses_variable_flow_traversal": True,
    "uses_query_supported_same_object_handoff": False,
    "uses_source_prior_guard": True,
    "uses_source_prior_tail_insertion": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_llm_query_compiler": False,
    "uses_sfb_or_support_fusion": False,
    "uses_v13b_source_report": False,
    "uses_legacy_compare_report": False,
}
