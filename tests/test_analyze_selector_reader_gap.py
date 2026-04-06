import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_selector_reader_gap import build_query_comparison, compute_support_metrics


def test_compute_support_metrics_handles_overlap():
    metrics = compute_support_metrics(
        predicted_titles=["A", "B", "C"],
        gold_titles=["B", "C", "D"],
    )
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert metrics["f1"] == 2 / 3
    assert metrics["full_support"] == 0.0


def test_build_query_comparison_captures_selector_reader_gap_pattern():
    control_trace = {
        "question": "Q",
        "query_type": "bridge",
        "gold_titles": ["Gold A", "Gold B"],
        "selector_answer": "wrong",
        "selector_top_titles": ["Gold A", "Distractor 1", "Distractor 2", "Distractor 3", "Distractor 4"],
        "selector_metrics": {"ExactMatch": 0.0, "F1": 0.0},
    }
    candidate_trace = {
        "question": "Q",
        "query_type": "bridge",
        "gold_titles": ["Gold A", "Gold B"],
        "selector_answer": "wrong",
        "selector_top_titles": ["Gold A", "Distractor 1", "Distractor 2", "Gold B", "Bridge X"],
        "selector_metrics": {"ExactMatch": 0.0, "F1": 0.0},
        "selector_trace": {
            "gate_decision": {"reason": "offrank_bridge_signal_detected", "use_selector": True},
            "saturation_guard_triggered": False,
        },
    }

    record = build_query_comparison(control_trace, candidate_trace)

    assert record["quadrant"] == "selector_improved__qa_unchanged"
    assert record["support_f1_delta"] > 0
    assert record["qa_f1_delta"] == 0.0
    assert record["all_added_after_top3"] is True
    assert record["any_gold_position_improved"] is True
    assert record["candidate_gate_reason"] == "offrank_bridge_signal_detected"
