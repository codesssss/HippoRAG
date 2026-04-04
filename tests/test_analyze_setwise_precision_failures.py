from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_setwise_precision_failures import build_analysis, build_query_record


def make_trace(question: str,
               baseline_f1: float,
               selector_f1: float,
               gate_reason: str,
               state_metrics: dict,
               beam_steps: list[dict]) -> dict:
    return {
        "question": question,
        "baseline_answer": "baseline",
        "selector_answer": "selector",
        "baseline_top_titles": ["Base A", "Base B"],
        "selector_top_titles": ["Sel A", "Sel B"],
        "baseline_metrics": {"ExactMatch": 0.0, "F1": baseline_f1},
        "selector_metrics": {"ExactMatch": 0.0, "F1": selector_f1},
        "selector_trace": {
            "gate_decision": {
                "reason": gate_reason,
                "use_selector": True,
            },
            "beam_best_state_score": state_metrics["state_score"],
            "beam_best_state_path_connectivity": state_metrics["path_connectivity"],
            "beam_best_state_reachable_doc_ratio": state_metrics["reachable_doc_ratio"],
            "beam_best_state_query_reachability": state_metrics["query_reachability"],
            "beam_best_state_query_coverage": state_metrics["query_coverage"],
            "beam_best_state_support_mean": state_metrics["support_mean"],
            "beam_best_state_suffix_base_mean": state_metrics["suffix_base_mean"],
            "selection_steps": beam_steps,
        },
    }


def test_build_query_record_marks_saturated_state():
    record = build_query_record(
        make_trace(
            question="q",
            baseline_f1=1.0,
            selector_f1=0.0,
            gate_reason="offrank_bridge_signal_detected",
            state_metrics={
                "state_score": 0.9,
                "path_connectivity": 1.0,
                "reachable_doc_ratio": 1.0,
                "query_reachability": 1.0,
                "query_coverage": 1.0,
                "support_mean": 0.87,
                "suffix_base_mean": 0.12,
            },
            beam_steps=[
                {
                    "step": 4,
                    "mode": "beam",
                    "pool_position": 10,
                    "structure_score": 1.0,
                    "novelty_score": 0.9,
                    "closure_score": 0.7,
                    "selection_score": 1.0,
                    "state_score": 0.9,
                }
            ],
        )
    )
    assert record["delta_f1"] == -1.0
    assert record["saturated_state"] is True
    assert record["max_local_structure"] == 1.0
    assert record["max_local_closure"] == 0.7


def test_build_analysis_reports_overlap_and_counts():
    report = {
        "setwise_selector_query_traces": [
            make_trace(
                question="positive",
                baseline_f1=0.0,
                selector_f1=1.0,
                gate_reason="offrank_bridge_signal_detected",
                state_metrics={
                    "state_score": 0.92,
                    "path_connectivity": 1.0,
                    "reachable_doc_ratio": 1.0,
                    "query_reachability": 1.0,
                    "query_coverage": 1.0,
                    "support_mean": 0.9,
                    "suffix_base_mean": 0.4,
                },
                beam_steps=[
                    {
                        "step": 4,
                        "mode": "beam",
                        "pool_position": 12,
                        "structure_score": 1.0,
                        "novelty_score": 0.97,
                        "closure_score": 0.85,
                        "selection_score": 1.0,
                        "state_score": 0.92,
                    }
                ],
            ),
            make_trace(
                question="negative",
                baseline_f1=1.0,
                selector_f1=0.0,
                gate_reason="offrank_bridge_signal_detected",
                state_metrics={
                    "state_score": 0.89,
                    "path_connectivity": 1.0,
                    "reachable_doc_ratio": 1.0,
                    "query_reachability": 1.0,
                    "query_coverage": 1.0,
                    "support_mean": 0.87,
                    "suffix_base_mean": 0.1,
                },
                    beam_steps=[
                        {
                            "step": 4,
                            "mode": "beam",
                            "pool_position": 26,
                            "structure_score": 1.0,
                            "novelty_score": 0.95,
                            "closure_score": 0.7,
                            "selection_score": 1.0,
                            "state_score": 0.89,
                        }
                    ],
                ),
        ]
    }

    analysis = build_analysis(report)
    assert analysis["positive_summary"]["count"] == 1
    assert analysis["negative_summary"]["count"] == 1
    assert analysis["feature_overlap"]["max_local_structure"]["ranges_overlap"] is True
    assert any("absolute scores do not cleanly separate" in finding for finding in analysis["findings"])
