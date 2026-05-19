"""Source-authorized vocab-strict retrieval contract.

This module names the high-score branch by its source contract, not by the
overloaded report variant label.  The historical high-score reports expose the
method as `graph_native_source_authorized_evidence_set_search`, but the actual
implementation branch is source-text evidence-set MCT v18.
"""

from __future__ import annotations

from typing import Mapping


PUBLIC_REPORT_ALIAS = "graph_native_source_authorized_evidence_set_search"
HIGH_SCORE_LEGACY_VARIANT = "graph_native_source_text_evidence_set_mct_v18"
CANONICAL_CLEAN_METHOD_NAME = "source_text_certificate_graphrag"
SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME = (
    "source_certified_evidence_set_completion"
)


CANONICAL_CLEAN_CONTRACT: Mapping[str, object] = {
    "paper_facing_method_name": CANONICAL_CLEAN_METHOD_NAME,
    "method_object": "source_text_certificate_reader_context",
    "candidate_entrance": "local_candidate_pool_plus_source_bound_endpoint_expansion",
    "composition": "source_text_certificate_frontier_to_conservative_reader_context",
    "ranked_object": "reader_facing_evidence_context",
    "uses_evidence_set_objective": False,
    "uses_source_certified_reader_context_assembly": True,
    "certificate_policy": "canonical_source_text",
    "uses_source_text_certificate_graph": True,
    "uses_source_bound_endpoint_expansion": True,
    "source_text_certificate_is_admission_requirement": True,
    "uses_in_pool_source_text_repair_targets": True,
    "in_pool_source_text_repair_window": "next_top_k_dense_tail",
    "canonical_replacement_tail_only": True,
    "canonical_replacement_max_inserted_docs": 1,
    "uses_role_vocabulary": False,
    "uses_relation_role_certificates": False,
    "uses_surface_role_certificates": False,
    "uses_directional_child_parent_hints": False,
    "uses_broad_location_blocklist": False,
    "uses_external_baseline_frontier": False,
    "uses_external_baseline_frontier_for_admission": False,
    "uses_baseline_reader_context_when_no_source_bound_path": False,
    "uses_proposition_support_evidence": False,
    "proposition_support_alone_can_select_reader_context": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_two_retriever_ensemble": False,
    "uses_pre_ranked_sto_qstate_lists": False,
    "uses_role_or_focal_override": False,
    "uses_compact_reader_dual_readout": False,
}


SOURCE_CERTIFIED_EVIDENCE_COMPLETION_CONTRACT: Mapping[str, object] = {
    "paper_facing_method_name": SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME,
    "method_object": "reader_budgeted_source_certified_evidence_set",
    "candidate_entrance": "local_candidate_pool_plus_source_bound_endpoint_expansion",
    "composition": "retrieval_grounded_evidence_demands_to_source_certified_set_completion",
    "ranked_object": "reader_facing_evidence_context",
    "certificate_policy": "canonical_source_text",
    "uses_source_text_certificate_graph": True,
    "uses_source_bound_endpoint_expansion": True,
    "uses_retrieval_grounded_evidence_demands": True,
    "uses_budgeted_set_completion": True,
    "uses_tail_only_safe_mode": False,
    "source_text_certificate_is_admission_requirement": True,
    "uses_role_vocabulary": False,
    "uses_relation_role_certificates": False,
    "uses_surface_role_certificates": False,
    "uses_directional_child_parent_hints": False,
    "uses_broad_location_blocklist": False,
    "uses_external_baseline_frontier": False,
    "uses_external_baseline_frontier_for_admission": False,
    "uses_baseline_reader_context_when_no_source_bound_path": False,
    "uses_proposition_support_evidence": False,
    "proposition_support_alone_can_select_reader_context": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_two_retriever_ensemble": False,
    "uses_pre_ranked_sto_qstate_lists": False,
    "uses_role_or_focal_override": False,
    "uses_compact_reader_dual_readout": False,
    "uses_learned_reranker": False,
}


HIGH_SCORE_CONTRACT: Mapping[str, object] = {
    "paper_facing_method_name": "source_authorized_evidence_set_search",
    "public_report_alias": PUBLIC_REPORT_ALIAS,
    "legacy_variant": HIGH_SCORE_LEGACY_VARIANT,
    "method_object": "source_text_authorized_evidence_set",
    "candidate_entrance": "method_internal_evidence_frontier",
    "composition": "source_text_query_chain_certified_evidence_set_mct_to_reader_facing_context",
    "ranked_object": "reader_facing_evidence_context",
    "reported_retrieval_context_is_reader_input": True,
    "uses_source_text_evidence_set_mct": True,
    "uses_source_text_frontier_seed_context_for_mct": True,
    "mct_initial_context_is_complete_reader_context": True,
    "source_text_frontier_is_search_exposure_not_reader_head_seed": True,
    "uses_method_internal_reader_title_namespace_saturation": True,
    "method_internal_reader_title_namespace_repeat_limit": 3,
    "requires_specific_source_text_rare_token_binding": True,
    "uses_expanded_source_text_location_hints": True,
    "uses_expanded_source_text_bridge_relation_hints": True,
    "uses_expanded_source_text_communication_relation_hints": True,
    "uses_query_mentioned_shallow_frontier_source_text_sources": True,
    "uses_morphological_query_mention_matching": True,
    "uses_directional_child_to_parent_source_text_authorization": True,
    "uses_source_text_surface_role_authorization": True,
    "broad_location_path_replacement_requires_query_chain_document_certificate": True,
    "prevents_generic_location_only_evidence_set_mct_reader_head_replacement": True,
    "protects_answer_type_title_incumbents_from_generic_location_evidence_set_mct": True,
    "prefers_initial_source_role_instance_coverage_for_mct": True,
    "protects_initial_reader_source_anchors_for_mct": True,
    "initial_reader_source_anchor_protection_count_for_mct": 1,
    "path_reader_head_replacement_max_inserted_nodes": 1,
    "allows_query_chain_document_certified_multi_insert_path_replacement": True,
    "mct_reader_head_replacement_max_inserted_nodes": 1,
    "uses_source_bound_evidence_paths": True,
    "uses_source_bound_evidence_edges": True,
    "source_text_certificate_is_admission_requirement": True,
    "source_text_certificate_alone_is_value_signal": False,
    "uses_external_baseline_frontier": False,
    "uses_external_baseline_frontier_for_admission": False,
    "uses_baseline_reader_context_when_no_source_bound_path": False,
    "uses_proposition_support_evidence": False,
    "proposition_support_alone_can_select_reader_context": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_two_retriever_ensemble": False,
    "uses_pre_ranked_sto_qstate_lists": False,
    "uses_role_or_focal_override": False,
    "uses_compact_reader_dual_readout": False,
}
