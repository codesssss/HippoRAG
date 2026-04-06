import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_reader_order_probe import build_query_record


def test_build_query_record_detects_clean_ordering_probe_improvement():
    control_trace = {
        "question": "Q",
        "gold_titles": ["Bridge", "Answer"],
        "selector_answer": "wrong",
        "selector_metrics": {"ExactMatch": 0.0, "F1": 0.0},
        "selector_trace": {
            "selected_titles": ["Anchor", "Bridge", "Answer"],
            "final_front_titles": ["Anchor", "Reserve", "Reserve 2", "Bridge", "Answer"],
        },
    }
    probe_trace = {
        "question": "Q",
        "gold_titles": ["Bridge", "Answer"],
        "selector_answer": "right",
        "selector_metrics": {"ExactMatch": 1.0, "F1": 1.0},
        "selector_trace": {
            "selected_titles": ["Anchor", "Bridge", "Answer"],
            "final_front_titles": ["Anchor", "Reserve", "Bridge", "Reserve 2", "Answer"],
            "reader_order_probe": {
                "applied": True,
                "mode": "promote_best_bridge_to_slot3",
                "promoted_title": "Bridge",
                "promoted_from_rank": 4,
            },
        },
    }

    record = build_query_record(control_trace, probe_trace)

    assert record["selected_set_unchanged"] is True
    assert record["probe_applied"] is True
    assert record["bucket"] == "probe_applied__qa_improved"
    assert record["gold_top3_coverage_delta"] > 0
    assert record["qa_f1_delta"] == 1.0


def test_build_query_record_flags_selected_set_changes_as_invalid():
    control_trace = {
        "question": "Q",
        "gold_titles": ["Bridge"],
        "selector_answer": "same",
        "selector_metrics": {"ExactMatch": 0.0, "F1": 0.0},
        "selector_trace": {
            "selected_titles": ["Anchor", "Bridge"],
            "final_front_titles": ["Anchor", "Reserve", "Bridge"],
        },
    }
    probe_trace = {
        "question": "Q",
        "gold_titles": ["Bridge"],
        "selector_answer": "same",
        "selector_metrics": {"ExactMatch": 0.0, "F1": 0.0},
        "selector_trace": {
            "selected_titles": ["Anchor", "Noise"],
            "final_front_titles": ["Anchor", "Noise", "Bridge"],
            "reader_order_probe": {
                "applied": True,
                "mode": "promote_best_bridge_to_slot3",
                "promoted_title": "Bridge",
                "promoted_from_rank": 3,
            },
        },
    }

    record = build_query_record(control_trace, probe_trace)

    assert record["selected_set_unchanged"] is False
    assert record["bucket"] == "invalid_selected_set_changed"
