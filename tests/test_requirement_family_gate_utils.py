from src.hipporag.utils.requirement_family_gate_utils import (
    POOL_COVERAGE_GAP_FAMILY,
    STAGED_CLOSURE_GATE_FAMILY,
    UTILITY_REJECTION_FAMILY,
    build_staged_closure_gate_decision,
    classify_requirement_failure_families,
    compute_title_recall_at_k,
)


def test_classify_requirement_failure_families_marks_q6_style_staged_closure_candidate():
    exposure_rows = [
        {
            "title": "Anchor Gold",
            "stage": "selected",
            "best_support_completeness_gain": None,
            "best_utility_margin_gain": None,
        },
        {
            "title": "Bridge Gold 1",
            "stage": "pool_only",
            "best_support_completeness_gain": 0.0,
            "best_utility_margin_gain": 0.0,
        },
        {
            "title": "Bridge Gold 2",
            "stage": "pool_only",
            "best_support_completeness_gain": 0.0,
            "best_utility_margin_gain": 0.0,
        },
    ]

    families = classify_requirement_failure_families(
        exposure_rows=exposure_rows,
        selector_metrics={"ExactMatch": 0.0, "F1": 0.2},
        gold_count=3,
    )

    assert STAGED_CLOSURE_GATE_FAMILY in families
    assert UTILITY_REJECTION_FAMILY not in families
    decision = build_staged_closure_gate_decision(
        question="q6-like",
        exposure_rows=exposure_rows,
        selector_metrics={"ExactMatch": 0.0, "F1": 0.2},
        gold_count=3,
    )
    assert decision["gate_hit"] is True
    assert "has_pool_only_gold" in decision["gate_reasons"]


def test_classify_requirement_failure_families_excludes_utility_rejection_cases():
    exposure_rows = [
        {
            "title": "Selected Gold",
            "stage": "selected",
            "best_support_completeness_gain": None,
            "best_utility_margin_gain": None,
        },
        {
            "title": "Rejected Source Gold",
            "stage": "source",
            "best_support_completeness_gain": 0.1156,
            "best_utility_margin_gain": -0.0096,
        },
        {
            "title": "Deep Gold",
            "stage": "pool_only",
            "best_support_completeness_gain": 0.0,
            "best_utility_margin_gain": 0.0,
        },
    ]

    families = classify_requirement_failure_families(
        exposure_rows=exposure_rows,
        selector_metrics={"ExactMatch": 0.0, "F1": 0.0},
        gold_count=3,
    )

    assert UTILITY_REJECTION_FAMILY in families
    assert STAGED_CLOSURE_GATE_FAMILY not in families
    decision = build_staged_closure_gate_decision(
        question="utility-like",
        exposure_rows=exposure_rows,
        selector_metrics={"ExactMatch": 0.0, "F1": 0.0},
        gold_count=3,
    )
    assert decision["gate_hit"] is False
    assert "utility_rejection_signal" in decision["gate_blockers"]


def test_classify_requirement_failure_families_excludes_pool_coverage_gap_cases():
    exposure_rows = [
        {
            "title": "Selected Gold",
            "stage": "selected",
            "best_support_completeness_gain": None,
            "best_utility_margin_gain": None,
        },
        {
            "title": "Missing Gold",
            "stage": "not_in_pool",
            "best_support_completeness_gain": None,
            "best_utility_margin_gain": None,
        },
        {
            "title": "Pool Gold",
            "stage": "pool_only",
            "best_support_completeness_gain": 0.0,
            "best_utility_margin_gain": 0.0,
        },
    ]

    families = classify_requirement_failure_families(
        exposure_rows=exposure_rows,
        selector_metrics={"ExactMatch": 0.0, "F1": 0.0},
        gold_count=3,
    )

    assert POOL_COVERAGE_GAP_FAMILY in families
    assert STAGED_CLOSURE_GATE_FAMILY not in families


def test_compute_title_recall_at_k_normalizes_titles():
    recall = compute_title_recall_at_k(
        pool_titles=["Minneapolis", "Mississippi River", "Distractor"],
        gold_titles=["Minneapolis, ", "Mississippi River", "Minneapolis"],
        k=2,
    )

    assert recall == 1.0
