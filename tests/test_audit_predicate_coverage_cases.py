import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_predicate_coverage_cases import (
    audit_case,
    derive_best_offrank_title,
    propose_predicate_family,
    select_coverage_gap_cases,
)


def test_derive_best_offrank_title_prefers_control_only_gold():
    case = {
        "control_titles": ["A", "B", "Gold X", "C"],
        "candidate_titles": ["A", "B", "C", "D"],
        "gold_titles": ["Gold X", "Other Gold"],
    }

    assert derive_best_offrank_title(case) == "Gold X"


def test_select_coverage_gap_cases_filters_to_beneficial_withheld_zero_structure():
    control_report = {
        "setwise_selector_query_traces": [
            {
                "question": "Q1",
                "query_type": "bridge",
                "gold_titles": ["Gold A", "Gold B"],
                "selector_top_titles": ["Gold A", "Gold B", "D1", "D2", "D3"],
                "selector_metrics": {"F1": 0.4, "ExactMatch": 0.0},
                "selector_trace": {"gate_decision": {"use_selector": True, "reason": "disabled"}},
            }
        ]
    }
    candidate_report = {
        "setwise_selector_query_traces": [
            {
                "question": "Q1",
                "query_type": "bridge",
                "gold_titles": ["Gold A", "Gold B"],
                "selector_top_titles": ["Gold A", "D1", "D2", "D3", "D4"],
                "selector_metrics": {"F1": 0.2, "ExactMatch": 0.0},
                "selector_trace": {
                    "gate_decision": {
                        "use_selector": False,
                        "reason": "offrank_structure_below_threshold",
                        "best_offrank_structure_score": 0.0,
                        "avg_local_structure": 0.0,
                        "suffix_base_mean": 0.0,
                        "best_offrank_combined_score": 0.2,
                        "best_offrank_novelty_score": 0.8,
                        "best_offrank_frontier_gain_score": 0.0,
                        "best_offrank_path_coherence_score": 0.0,
                        "best_offrank_closure_score": 0.0,
                        "weakest_baseline_suffix_structure_score": 0.0,
                        "weakest_baseline_suffix_combined_score": 0.1,
                        "weakest_baseline_suffix_novelty_score": 0.2,
                        "weakest_baseline_suffix_frontier_gain_score": 0.0,
                        "weakest_baseline_suffix_path_coherence_score": 0.0,
                        "weakest_baseline_suffix_closure_score": 0.0,
                    }
                },
            }
        ]
    }

    cases = select_coverage_gap_cases(control_report, candidate_report)

    assert len(cases) == 1
    assert cases[0]["query_id"] == 1
    assert cases[0]["best_offrank_structure_score"] == 0.0
    assert cases[0]["best_offrank_title"] == "Gold B"


def test_propose_predicate_family_covers_part_of_and_family_relation():
    assert propose_predicate_family("is the entry for") == "factual_part_of_or_contains"
    assert propose_predicate_family("is daughter of") == "factual_family_relation"


def test_audit_case_marks_rejected_patchable_family_and_entity_link():
    case = {
        "query_id": 7,
        "question": "Q",
        "raw_case_bucket": "beneficial_withheld",
        "best_offrank_title": "Gold A",
        "gold_titles": ["Gold A", "Gold B"],
    }
    title_to_docs = {
        "Gold A": [
            {
                "idx": "doc-a",
                "passage": "Gold A\nText",
                "extracted_triples": [
                    ["Gold A", "is part of", "Shared Entity"],
                    ["Gold A", "stars", "Actor X"],
                ],
            }
        ],
        "Gold B": [
            {
                "idx": "doc-b",
                "passage": "Gold B\nText",
                "extracted_triples": [
                    ["Shared Entity", "is located in", "Place Y"],
                ],
            }
        ],
    }

    audit = audit_case(case, title_to_docs)

    assert audit["raw_triple_count"] == 3
    assert audit["candidate_rejected_triple_count"] >= 2
    assert audit["entity_link_possible"] is True
    assert audit["edge_reachable_after_patch"] is True
    assert "factual_part_of_or_contains" in audit["proposed_predicate_family"]
