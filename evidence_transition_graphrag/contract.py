"""Method identity and clean-contract metadata for Evidence Transition GraphRAG.

The implementation still reuses the existing STO graph modules. This package is
the public boundary for the current end-to-end clean GraphRAG line.
"""

from __future__ import annotations

from typing import Mapping, Tuple


METHOD_NAME = "evidence_transition_graphrag"
METHOD_TRACE_NAME = "evidence_transition_graph_native_retrieval"
QA_DOC_KEY = "evidence_transition_doc_indices_top5"

LEGACY_METHOD_NAMES: Tuple[str, ...] = (
    "query_grounded_sto_graphrag",
)
LEGACY_QA_DOC_KEYS: Tuple[str, ...] = (
    "query_grounded_sto_doc_indices_top5",
    "source_authorized_vocab_strict_doc_indices_top5",
)

METHOD_CONTRACT: Mapping[str, object] = {
    "paper_facing_method_name": METHOD_NAME,
    "method_object": "query_local_evidence_transition_graph",
    "indexing": "fresh_passage_embeddings_plus_fresh_openie",
    "graph_construction": "global_sto_evidence_graph",
    "query_graph": "query_local_sto_graph_induction",
    "retrieval": "graph_native_reader_context_selection",
    "qa_context": "fixed_top5_passages",
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_llm_query_compiler": False,
    "uses_sfb_or_support_fusion": False,
    "uses_v13b_source_report": False,
    "uses_legacy_compare_report": False,
}
