from run_transition_top5_qa import (
    base_channel_doc_indices,
    channel_fusion_doc_indices,
    compress_passage_for_query,
    method_doc_indices,
    native_anchor_neighborhood_base_doc_indices,
    native_consensus_base_doc_indices,
    prepare_reader_doc,
    qa_deltas,
    summarize_method_result,
    top5_failure_bucket,
)


def test_method_doc_indices_prefers_sfb_reader_topk():
    sfb_row = {
        "reader_doc_indices_topk": [5, 4, 4, 3, 2, 1],
        "retrieved_doc_indices_top5": [9, 8, 7, 6, 5],
    }
    transition_row = {"semantic_pair_doc_indices_top5": [1, 2, 3]}

    assert method_doc_indices(
        method="sfb_context",
        transition_row=transition_row,
        sfb_row=sfb_row,
        qa_top_k=5,
    ) == [5, 4, 3, 2, 1]
    assert method_doc_indices(
        method="semantic_pair",
        transition_row=transition_row,
        sfb_row=sfb_row,
        qa_top_k=5,
    ) == [1, 2, 3]


def test_channel_fusion_uses_both_rank_channels_without_gold():
    sfb_row = {"retrieved_doc_indices_top20": [10, 20, 30, 40, 50]}
    transition_row = {
        "specificity_pairwise_transition_emission": {
            "retrieved_doc_indices": [30, 60, 20, 70, 80]
        }
    }

    fused = channel_fusion_doc_indices(
        transition_row=transition_row,
        sfb_row=sfb_row,
        qa_top_k=5,
        fusion_sfb_weight=2.0,
        fusion_specificity_weight=3.0,
        fusion_sfb_depth=5,
        fusion_specificity_depth=5,
        fusion_stable_prefix_k=0,
        fusion_support_weight=0.0,
        fusion_support_overlap_bonus=0.0,
        fusion_support_depth=0,
    )

    assert fused[:3] == [30, 20, 10]
    assert 60 in fused


def test_channel_fusion_can_preserve_stable_sfb_prefix():
    sfb_row = {"retrieved_doc_indices_top20": [10, 20, 30, 40, 50]}
    transition_row = {
        "specificity_pairwise_transition_emission": {
            "retrieved_doc_indices": [30, 60, 20, 70, 80]
        }
    }

    fused = channel_fusion_doc_indices(
        transition_row=transition_row,
        sfb_row=sfb_row,
        qa_top_k=5,
        fusion_sfb_weight=2.0,
        fusion_specificity_weight=3.0,
        fusion_sfb_depth=5,
        fusion_specificity_depth=5,
        fusion_stable_prefix_k=3,
        fusion_support_weight=0.0,
        fusion_support_overlap_bonus=0.0,
        fusion_support_depth=0,
    )

    assert fused[:3] == [10, 20, 30]
    assert 60 in fused


def test_channel_fusion_can_use_support_set_structure_as_soft_regularizer():
    sfb_row = {"retrieved_doc_indices_top20": [10, 20, 30, 40, 50]}
    transition_row = {
        "specificity_pairwise_transition_emission": {
            "retrieved_doc_indices": [60, 70, 80, 90, 100]
        },
        "support_set_search": {
            "top_support_sets": [
                {"doc_indices": [10, 95], "score": 250.0},
            ]
        },
    }

    fused = channel_fusion_doc_indices(
        transition_row=transition_row,
        sfb_row=sfb_row,
        qa_top_k=5,
        fusion_sfb_weight=2.0,
        fusion_specificity_weight=3.0,
        fusion_sfb_depth=5,
        fusion_specificity_depth=5,
        fusion_stable_prefix_k=1,
        fusion_support_weight=1.0,
        fusion_support_overlap_bonus=1.0,
        fusion_support_depth=5,
    )

    assert fused[0] == 10
    assert 95 in fused


def test_native_support_fusion_uses_native_base_prefix_not_sfb():
    sfb_row = {"retrieved_doc_indices_top20": [10, 20, 30, 40, 50]}
    transition_row = {
        "native_dense_doc_indices_top10": [100, 200, 300, 400, 500],
        "specificity_pairwise_transition_emission": {
            "retrieved_doc_indices": [300, 600, 200, 700, 800]
        },
    }

    fused = method_doc_indices(
        method="native_support_fusion",
        transition_row=transition_row,
        sfb_row=sfb_row,
        qa_top_k=5,
        fusion_stable_prefix_k=2,
        fusion_support_weight=0.0,
    )

    assert fused[:2] == [100, 200]
    assert 10 not in fused


def test_base_channel_doc_indices_supports_non_sfb_channels():
    transition_row = {
        "context_anchor_doc_indices": [4, 4, 5, 6],
        "bm25_doc_indices_top10": [7, 8, 8, 9],
        "hybrid_residual_pair_doc_indices_top10": [10, 11, 12],
    }
    sfb_row = {"retrieved_doc_indices_top20": [1, 2, 3]}

    assert base_channel_doc_indices(
        base_channel="context_anchor",
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=5,
        qa_top_k=5,
    ) == [4, 5, 6]
    assert base_channel_doc_indices(
        base_channel="bm25",
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=5,
        qa_top_k=5,
    ) == [7, 8, 9]
    assert base_channel_doc_indices(
        base_channel="hybrid_residual",
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=5,
        qa_top_k=5,
    ) == [10, 11, 12]


def test_native_consensus_base_keeps_native_prefix_and_promotes_agreement_tail():
    transition_row = {
        "native_dense_doc_indices_top10": [1, 2, 9, 8, 7, 6],
        "specificity_pair_doc_indices_top10": [7, 5, 4, 3],
        "support_set_search": {
            "top_support_sets": [
                {"doc_indices": [7, 4], "score": 250.0},
            ]
        },
    }
    sfb_row = {"retrieved_doc_indices_top20": [99, 98, 97]}

    docs = native_consensus_base_doc_indices(
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=6,
        qa_top_k=5,
        stable_prefix_k=2,
    )

    assert docs[:2] == [1, 2]
    assert docs[2] == 7
    assert 99 not in docs


def test_native_spec_consensus_base_does_not_promote_support_only_tail():
    transition_row = {
        "native_dense_doc_indices_top10": [1, 2, 9, 8, 7, 6],
        "specificity_pair_doc_indices_top10": [5, 4, 3],
        "support_set_search": {
            "top_support_sets": [
                {"doc_indices": [7], "score": 250.0},
            ]
        },
    }
    sfb_row = {"retrieved_doc_indices_top20": [99, 98, 97]}

    docs = native_consensus_base_doc_indices(
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=6,
        qa_top_k=5,
        stable_prefix_k=2,
        include_support_channel=False,
    )

    assert docs[:2] == [1, 2]
    assert 5 in docs
    assert 7 not in docs


def test_native_anchor_neighborhood_keeps_native_top2_and_uses_anchor_completion():
    transition_row = {
        "native_dense_doc_indices_top10": [1, 2, 9, 8, 7, 6],
        "query_conditioned_neighborhood": {
            "retrieved_doc_indices": [1, 3, 4, 5],
            "selected_neighborhood_doc_indices": [1, 4, 5],
            "boundary_readout_pairs": [
                {
                    "anchor_doc": 1,
                    "completion_doc": 5,
                    "score": 10.0,
                    "direct_transition_score": 6.0,
                    "direct_endpoint_count": 2,
                    "direct_skipped_endpoint_count": 0,
                    "selected_by_neighborhood": True,
                }
            ],
            "top_edges": [
                {"left_doc": 2, "right_doc": 4, "weight": 3.0, "endpoints": ["shared"]},
            ],
        },
    }
    sfb_row = {"retrieved_doc_indices_top20": [99, 98, 97]}

    docs = native_anchor_neighborhood_base_doc_indices(
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=6,
        qa_top_k=5,
        stable_prefix_k=2,
    )

    assert docs[:2] == [1, 2]
    assert docs[2] == 5
    assert 99 not in docs


def test_native_anchor_neighborhood_does_not_double_count_support_sets():
    transition_row = {
        "native_dense_doc_indices_top10": [1, 2, 9, 8, 7, 6],
        "support_set_search": {
            "top_support_sets": [
                {"doc_indices": [30, 31], "score": 500.0, "shared_endpoint_count": 4},
                {"doc_indices": [2, 5], "score": 250.0, "shared_endpoint_count": 1},
            ]
        },
    }
    sfb_row = {"retrieved_doc_indices_top20": [99, 98, 97]}

    docs = native_anchor_neighborhood_base_doc_indices(
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=6,
        qa_top_k=5,
        stable_prefix_k=2,
    )

    assert docs[:2] == [1, 2]
    assert 5 not in docs
    assert 30 not in docs
    assert 31 not in docs


def test_native_anchor_neighborhood_support_fusion_does_not_use_sfb_base():
    transition_row = {
        "native_dense_doc_indices_top10": [1, 2, 9, 8, 7, 6],
        "query_conditioned_neighborhood": {
            "boundary_readout_pairs": [
                {"anchor_doc": 1, "completion_doc": 5, "score": 5.0, "direct_transition_score": 5.0}
            ],
        },
        "specificity_pairwise_transition_emission": {
            "retrieved_doc_indices": [5, 4, 3]
        },
    }
    sfb_row = {"retrieved_doc_indices_top20": [99, 98, 97]}

    docs = method_doc_indices(
        method="native_anchor_neighborhood_support_fusion",
        transition_row=transition_row,
        sfb_row=sfb_row,
        qa_top_k=5,
        fusion_stable_prefix_k=2,
        fusion_support_weight=0.0,
    )

    assert docs[:2] == [1, 2]
    assert 5 in docs
    assert 99 not in docs


def test_anchor_guided_evidence_method_reads_selected_set_without_sfb():
    transition_row = {
        "anchor_guided_evidence": {
            "selected_evidence_set": {
                "doc_indices": [100, 200, 300, 400, 500],
            },
            "retrieved_doc_indices": [100, 200, 300, 400, 500, 600],
        },
    }
    sfb_row = {"retrieved_doc_indices_top20": [99, 98, 97]}

    docs = method_doc_indices(
        method="anchor_guided_evidence",
        transition_row=transition_row,
        sfb_row=sfb_row,
        qa_top_k=5,
    )

    assert docs == [100, 200, 300, 400, 500]
    assert 99 not in docs


def test_compress_passage_for_query_preserves_title_and_relevant_sentence():
    passage = (
        "Palau de la Generalitat\n"
        "This palace is in Barcelona. "
        "The Palau de la Generalitat was built in the 15th century. "
        "A long unrelated sentence describes another building."
    )

    compressed = compress_passage_for_query(
        passage=passage,
        question="When was the Palau de la Generalitat built?",
        max_sentences=1,
        max_chars=200,
    )

    assert compressed.startswith("Palau de la Generalitat\n")
    assert "15th century" in compressed
    assert "unrelated" not in compressed


def test_prepare_reader_doc_full_mode_leaves_passage_unchanged():
    passage = "Title\nBody sentence."

    assert prepare_reader_doc(
        passage=passage,
        question="question",
        context_mode="full",
        context_max_sentences=1,
        context_max_chars=10,
    ) == passage


def test_top5_failure_bucket_separates_reader_and_retrieval_failures():
    all_gold = {"all_gold_at5": True, "any_gold_at5": True}
    partial = {"all_gold_at5": False, "any_gold_at5": True}
    no_gold = {"all_gold_at5": False, "any_gold_at5": False}

    assert top5_failure_bucket(all_gold, 1.0) == "qa_exact_all_gold_top5"
    assert top5_failure_bucket(partial, 1.0) == "qa_exact_partial_gold_top5"
    assert top5_failure_bucket(no_gold, 1.0) == "qa_exact_no_gold_top5"
    assert top5_failure_bucket(all_gold, 0.0) == "all_gold_top5_qa_wrong"
    assert top5_failure_bucket(partial, 0.0) == "partial_gold_top5_qa_wrong"
    assert top5_failure_bucket(no_gold, 0.0) == "no_gold_top5_qa_wrong"
    assert top5_failure_bucket(no_gold, None) == "qa_not_run"


def test_summarize_method_result_and_deltas():
    evidence_rows = [
        {
            "recall_at5": 1.0,
            "all_gold_at5": True,
            "any_gold_at5": True,
            "answer_string_hit_at5": True,
            "gold_count_at5": 2,
        },
        {
            "recall_at5": 0.5,
            "all_gold_at5": False,
            "any_gold_at5": True,
            "answer_string_hit_at5": False,
            "gold_count_at5": 1,
        },
    ]
    per_query_base = [
        {"query_index": 0, "all_gold_at5": True, "any_gold_at5": True},
        {"query_index": 1, "all_gold_at5": False, "any_gold_at5": True},
    ]
    result = summarize_method_result(
        method="candidate",
        evidence_rows=evidence_rows,
        per_query_base=per_query_base,
        qa_result={
            "qa_metrics": {"ExactMatch": 0.5, "F1": 0.75},
            "predicted_answers": ["a", "b"],
            "example_em": [1.0, 0.0],
            "example_f1": [1.0, 0.5],
        },
    )

    assert result["metrics"]["r5"] == 0.75
    assert result["metrics"]["ExactMatch"] == 0.5
    assert result["qa_buckets"] == {
        "qa_exact_all_gold_top5": 1,
        "partial_gold_top5_qa_wrong": 1,
    }

    deltas = qa_deltas(
        {
            "sfb_context": {"metrics": {"r5": 0.5, "all_gold_at5": 0.0, "answer_string_hit_at5": 0.25, "ExactMatch": 0.4, "F1": 0.6}},
            "candidate": result,
        }
    )
    assert deltas["candidate"]["delta_r5"] == 0.25
    assert deltas["candidate"]["delta_exact_match"] == 0.1
    assert deltas["candidate"]["delta_f1"] == 0.15
