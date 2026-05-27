from __future__ import annotations

from evidence_transition_graphrag.contract import METHOD_TRACE_NAME, METHOD_V2_TRACE_NAME
from run_query_grounded_sto_fresh_e2e import (
    qa_output_stem,
    retrieval_output_stem,
    summary_output_stem,
)
from source_authorized_vocab_strict_retrieval.e2e_pipeline import method_name_for_runner
from src.agsto.local_graph import (
    ROLE_BRIDGE,
    SAME_SUBJECT,
    SENTENCE_GROUNDED_TRANSITION,
    SOURCE_ENDPOINT_INCIDENCE,
    TITLE_ROLE_GROUNDING,
    local_sto_evidence_channels,
    order_local_sto_channel_balanced_source_prior,
)


def test_etv2_method_identity_is_separate_from_frozen_et_v1() -> None:
    assert method_name_for_runner("query_grounded_sto_graph_native") == METHOD_TRACE_NAME
    assert method_name_for_runner("agsto_graph_native") == METHOD_TRACE_NAME
    assert method_name_for_runner("query_grounded_sto_graph_native_v2") == METHOD_V2_TRACE_NAME
    assert method_name_for_runner("agsto_graph_native_v2") == METHOD_V2_TRACE_NAME
    assert method_name_for_runner("evidence_transition_v2") == METHOD_V2_TRACE_NAME


def test_etv2_outputs_do_not_reuse_frozen_et_v1_filenames() -> None:
    assert (
        retrieval_output_stem(dataset="musique", runner="query_grounded_sto_graph_native")
        == "musique_fresh_query_grounded_sto_retrieval"
    )
    assert (
        retrieval_output_stem(dataset="musique", runner="evidence_transition_v2")
        == "musique_fresh_query_grounded_sto_v2_retrieval"
    )
    assert qa_output_stem(runner="query_grounded_sto_graph_native") == "fresh_query_grounded_sto_qa"
    assert qa_output_stem(runner="evidence_transition_v2") == "fresh_query_grounded_sto_v2_qa"
    assert summary_output_stem(runner="query_grounded_sto_graph_native") == "fresh_query_grounded_sto_summary"
    assert summary_output_stem(runner="evidence_transition_v2") == "fresh_query_grounded_sto_v2_summary"


def test_channel_balanced_readout_exposes_graph_semantic_channels() -> None:
    local_graph = {
        "admitted_doc_indices": [0, 1, 2, 3, 4, 5, 6],
        "seed_doc_indices": [0, 1],
        "stats": {"closure_hops": 2},
        "local_edges": [
            {"left_doc": 0, "right_doc": 2, "kinds": [SENTENCE_GROUNDED_TRANSITION]},
            {"left_doc": 1, "right_doc": 3, "kinds": [ROLE_BRIDGE]},
            {"left_doc": 0, "right_doc": 4, "kinds": [TITLE_ROLE_GROUNDING]},
            {"left_doc": 1, "right_doc": 5, "kinds": [SAME_SUBJECT]},
            {"left_doc": 0, "right_doc": 6, "kinds": [SOURCE_ENDPOINT_INCIDENCE]},
        ],
    }

    channels = local_sto_evidence_channels(local_graph=local_graph)

    assert channels["entry"] == [0, 1]
    assert channels[SENTENCE_GROUNDED_TRANSITION][0] == 2
    assert channels[TITLE_ROLE_GROUNDING][0] == 4
    assert channels[ROLE_BRIDGE][0] == 3
    assert channels[SAME_SUBJECT][0] == 5
    assert channels[SOURCE_ENDPOINT_INCIDENCE][0] == 6

    ordered = order_local_sto_channel_balanced_source_prior(
        local_graph=local_graph,
        max_docs=5,
    )

    assert ordered == [0, 2, 4, 3, 5]


def test_channel_balanced_readout_falls_back_to_admission_order() -> None:
    local_graph = {
        "admitted_doc_indices": [7, 8, 9],
        "seed_doc_indices": [],
        "stats": {"closure_hops": 2},
        "local_edges": [],
    }

    assert order_local_sto_channel_balanced_source_prior(local_graph=local_graph) == [7, 8, 9]
