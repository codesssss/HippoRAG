import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_gate_decision_boundary import build_analysis, build_case_record


def test_build_case_record_classifies_beneficial_withheld_and_computes_deltas():
    control_trace = {
        "question": "Q1",
        "query_type": "bridge",
        "gold_titles": ["Gold A", "Gold B"],
        "selector_top_titles": ["Gold A", "Distractor", "Gold B", "X", "Y"],
        "selector_metrics": {"F1": 0.4, "ExactMatch": 0.0},
        "selector_trace": {
            "gate_decision": {"use_selector": True, "reason": "disabled"},
            "selected_pool_positions": [0, 3, 5, 8, 9],
            "final_front_pool_positions": [0, 3, 5, 8, 9],
        },
    }
    candidate_trace = {
        "question": "Q1",
        "query_type": "bridge",
        "gold_titles": ["Gold A", "Gold B"],
        "selector_top_titles": ["Gold A", "Distractor", "X", "Y", "Z"],
        "selector_metrics": {"F1": 0.2, "ExactMatch": 0.0},
        "selector_trace": {
            "gate_decision": {
                "use_selector": False,
                "reason": "offrank_structure_below_threshold",
                "avg_local_structure": 0.0,
                "suffix_base_mean": 0.0,
                "best_offrank_structure_score": 0.0,
                "best_offrank_combined_score": 0.23,
                "best_offrank_novelty_score": 0.93,
                "best_offrank_frontier_gain_score": 0.0,
                "best_offrank_path_coherence_score": 0.0,
                "best_offrank_closure_score": 0.0,
                "weakest_baseline_suffix_structure_score": 0.0,
                "weakest_baseline_suffix_combined_score": 0.15,
                "weakest_baseline_suffix_novelty_score": 0.37,
                "weakest_baseline_suffix_frontier_gain_score": 0.0,
                "weakest_baseline_suffix_path_coherence_score": 0.0,
                "weakest_baseline_suffix_closure_score": 0.0,
            },
            "selected_pool_positions": [],
            "final_front_pool_positions": [0, 1, 2, 3, 4],
        },
    }

    record = build_case_record(control_trace, candidate_trace)

    assert record["bucket"] == "beneficial_withheld"
    assert record["control_gold_hit_count"] == 2
    assert record["candidate_gold_hit_count"] == 1
    assert record["delta_combined_score"] == 0.08
    assert record["delta_novelty_score"] == 0.56


def test_build_analysis_separates_blocked_and_allowed_buckets():
    control_report = {
        "setwise_selector_query_traces": [
            {
                "question": "Q1",
                "query_type": "bridge",
                "gold_titles": ["Gold A", "Gold B"],
                "selector_top_titles": ["Gold A", "Gold B", "D1", "D2", "D3"],
                "selector_metrics": {"F1": 0.4, "ExactMatch": 0.0},
                "selector_trace": {"gate_decision": {"use_selector": True, "reason": "disabled"}},
            },
            {
                "question": "Q2",
                "query_type": "bridge",
                "gold_titles": ["Gold A", "Gold B"],
                "selector_top_titles": ["Gold A", "D1", "D2", "D3", "D4"],
                "selector_metrics": {"F1": 0.2, "ExactMatch": 0.0},
                "selector_trace": {"gate_decision": {"use_selector": True, "reason": "disabled"}},
            },
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
                        "avg_local_structure": 0.0,
                        "suffix_base_mean": 0.0,
                        "best_offrank_structure_score": 0.0,
                        "best_offrank_combined_score": 0.23,
                        "best_offrank_novelty_score": 0.93,
                        "best_offrank_frontier_gain_score": 0.0,
                        "best_offrank_path_coherence_score": 0.0,
                        "best_offrank_closure_score": 0.0,
                        "weakest_baseline_suffix_structure_score": 0.0,
                        "weakest_baseline_suffix_combined_score": 0.15,
                        "weakest_baseline_suffix_novelty_score": 0.37,
                        "weakest_baseline_suffix_frontier_gain_score": 0.0,
                        "weakest_baseline_suffix_path_coherence_score": 0.0,
                        "weakest_baseline_suffix_closure_score": 0.0,
                    }
                },
            },
            {
                "question": "Q2",
                "query_type": "bridge",
                "gold_titles": ["Gold A", "Gold B"],
                "selector_top_titles": ["Gold A", "Gold B", "D1", "D2", "D3"],
                "selector_metrics": {"F1": 0.4, "ExactMatch": 0.0},
                "selector_trace": {
                    "gate_decision": {
                        "use_selector": True,
                        "reason": "offrank_bridge_signal_detected",
                        "avg_local_structure": 0.6,
                        "suffix_base_mean": 0.2,
                        "best_offrank_structure_score": 0.8,
                        "best_offrank_combined_score": 0.5,
                        "best_offrank_novelty_score": 0.7,
                        "best_offrank_frontier_gain_score": 0.4,
                        "best_offrank_path_coherence_score": 0.6,
                        "best_offrank_closure_score": 0.5,
                        "weakest_baseline_suffix_structure_score": 0.1,
                        "weakest_baseline_suffix_combined_score": 0.2,
                        "weakest_baseline_suffix_novelty_score": 0.3,
                        "weakest_baseline_suffix_frontier_gain_score": 0.1,
                        "weakest_baseline_suffix_path_coherence_score": 0.1,
                        "weakest_baseline_suffix_closure_score": 0.1,
                    }
                },
            },
        ]
    }

    analysis = build_analysis(control_report, candidate_report, case_limit=4)

    assert analysis["bucket_summaries"]["beneficial_withheld"]["count"] == 1
    assert analysis["bucket_summaries"]["beneficial_allowed"]["count"] == 1
    assert analysis["bucket_summaries"]["harmful_blocked"]["count"] == 0
    assert analysis["bucket_summaries"]["harmful_allowed"]["count"] == 0
    assert analysis["bucket_cases"]["beneficial_withheld"][0]["candidate_gate_reason"] == "offrank_structure_below_threshold"
    assert analysis["bucket_cases"]["beneficial_allowed"][0]["candidate_gate_reason"] == "offrank_bridge_signal_detected"
