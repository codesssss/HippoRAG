from analyze_v13b_evidence_frame_contract import (
    candidate_coverage_status,
    interpret_contract_counts,
    obligation_endpoint_shape,
)


def test_candidate_coverage_status_separates_missing_gold_docs():
    assert candidate_coverage_status(candidate_doc_indices=[1, 2, 3], gold_doc_indices=[2, 3]) == "gold_docs_covered"
    assert candidate_coverage_status(candidate_doc_indices=[1, 2], gold_doc_indices=[2, 3]) == "candidate_missing"
    assert candidate_coverage_status(candidate_doc_indices=[1, 2], gold_doc_indices=[]) == "gold_docs_missing"


def test_obligation_endpoint_shape_detects_descriptive_query_endpoint():
    obligation = {
        "raw_subject": "recently abdicated queen",
        "subject": "recently abdicated queen",
        "subject_is_variable": False,
        "raw_relation": "imprisoned by",
        "relation": "imprison_by",
        "raw_object": "?x1",
        "object": "x1",
        "object_is_variable": True,
    }

    assert obligation_endpoint_shape(obligation) == "descriptive_bound_endpoint"


def test_interpret_contract_counts_prefers_schema_bottleneck():
    report = interpret_contract_counts(
        total_obligations=10,
        frame_status_counts={
            "gold_endpoint_relation_mismatch": 4,
            "gold_endpoint_missing": 2,
            "gold_exact_openie_match_available": 1,
        },
    )

    assert report["main_bottleneck"] == "gold_openie_schema_contract"
    assert report["gold_openie_exact_frame_rate"] == 0.1
    assert report["endpoint_relation_or_endpoint_missing_rate"] == 0.6
