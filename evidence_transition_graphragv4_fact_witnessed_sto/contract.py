"""Contract for the Evidence Transition GraphRAG v4 fact-witnessed STO line."""

from __future__ import annotations

from typing import Mapping


METHOD_FAMILY = "evidence_transition_graphrag"
METHOD_VERSION = "v4_fact_witnessed_sto"
METHOD_NAME = "evidence_transition_graphragv4_fact_witnessed_sto"
METHOD_TRACE_NAME = "evidence_transition_fact_witnessed_sto_retrieval_v4"
QA_DOC_KEY = "evidence_transition_graphragv4_fact_witnessed_sto_doc_indices_top5"

METHOD_CONTRACT: Mapping[str, object] = {
    "method_family": METHOD_FAMILY,
    "method_version": METHOD_VERSION,
    "paper_facing_method_name": METHOD_NAME,
    "method_object": "fact_witnessed_branch_sto_document_graph_retrieval",
    "indexing": "fresh_openie_and_passage_embeddings_over_raw_passages",
    "corpus_graph": "global_sto_document_graph_with_openie_fact_witnesses",
    "query_graph": "dense_or_textual_entry_to_query_local_fact_witnessed_sto_graph",
    "retrieval": "fact_witnessed_branch_document_context_selection",
    "qa_context": "fixed_top5_passages",
    "pure_graph_only": False,
    "non_graph_signal_policy": "allowed_for_query_local_graph_entry_only",
    "uses_dense_entry": True,
    "uses_variable_flow_traversal": True,
    "uses_openie_fact_edge_witnesses": True,
    "uses_llm_generated_propositions": False,
    "retrieval_unit": "document",
    "fact_unit_policy": "openie_facts_define_document_edges_not_ranked_as_reader_items",
    "uses_query_supported_same_object_handoff": False,
    "uses_source_prior_guard": False,
    "uses_source_prior_tail_insertion": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_llm_query_compiler": False,
    "uses_sfb_or_support_fusion": False,
    "uses_v13b_source_report": False,
    "uses_legacy_compare_report": False,
}
