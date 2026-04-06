import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_gate_delivery_ablation import compute_support_metrics, summarize_report
from run_gate_delivery_ablation import build_eval_command, normalize_variant_name


def test_normalize_variant_name_and_build_eval_command_apply_overrides():
    base_report = {
        "dataset": "musique",
        "limit": 100,
        "llm_name": "qwen3-8b",
        "llm_request_name": "qwen3-8b-train",
        "llm_base_url": "http://localhost:8043/v1",
        "embedding_name": "embed",
        "embedding_base_url": "http://localhost:8018/v1/embeddings",
        "config": {
            "causal_enabled": False,
            "causal_engine_version": "v2",
            "causal_v2_graph_mode": "causal",
            "causal_v2_base_retrieval_mode": "legacy_fact_graph",
            "setwise_selector": "bridge_beam",
            "setwise_score_mode": "bridge",
            "setwise_pool_k": 100,
            "setwise_anchor_count": 2,
            "setwise_structure_max_hops": 2,
            "setwise_base_weight": 0.25,
            "setwise_structure_weight": 0.60,
            "setwise_novelty_weight": 0.15,
        },
        "setwise_selector_qa": {
            "selector": "bridge_beam",
            "score_mode": "bridge",
            "pool_k": 100,
            "anchor_count": 2,
            "reserve_top_m": 3,
            "max_bridge_slots": 2,
            "non_anchor_title_dedup": True,
            "query_entity_source": "seed",
            "gate_mode": "suffix_bridge_saturation_guard",
            "gate_min_structure_score": 0.15,
        },
    }

    assert normalize_variant_name("UNGATED") == "ungated"
    command = build_eval_command(
        base_report=base_report,
        output_json=Path("out.json"),
        save_dir_root="outputs_step0_general",
        limit=100,
        overrides={
            "gate_mode": "none",
            "gate_min_structure_score": 0.0,
        },
    )

    assert "--setwise_gate_mode" in command
    mode_index = command.index("--setwise_gate_mode")
    assert command[mode_index + 1] == "none"
    structure_index = command.index("--setwise_gate_min_structure_score")
    assert command[structure_index + 1] == "0.0"


def test_compute_support_metrics_and_summarize_report_capture_delivery_loss():
    metrics = compute_support_metrics(["A", "B", "A"], ["B", "C"])
    assert round(metrics["precision"], 4) == 0.5
    assert round(metrics["recall"], 4) == 0.5
    assert round(metrics["f1"], 4) == 0.5

    report = {
        "dataset": "musique",
        "overall_recomputed": {"ExactMatch": 0.2, "F1": 0.3},
        "setwise_selector_qa": {
            "selector_EM": 0.25,
            "selector_F1": 0.35,
            "selector_retrieval_metrics": {"Recall@5": 0.6, "Recall@10": 0.7},
            "selector_summary": {
                "gate_apply_count": 1,
                "gate_skip_count": 1,
                "gate_reason_counts": {
                    "offrank_bridge_signal_detected": 1,
                    "structure_saturated_weak_suffix": 1,
                },
                "saturation_guard_apply_count": 1,
                "saturation_guard_skip_count": 1,
            },
        },
        "setwise_selector_query_traces": [
            {
                "question": "Q1",
                "gold_titles": ["Bridge", "Answer"],
                "selector_metrics": {"ExactMatch": 0.0, "F1": 0.0},
                "selector_trace": {
                    "selected_titles": ["Anchor", "Bridge", "Answer"],
                    "final_front_titles": ["Anchor", "Noise", "Noise 2"],
                    "gate_decision": {"reason": "structure_saturated_weak_suffix"},
                },
            },
            {
                "question": "Q2",
                "gold_titles": ["Anchor"],
                "selector_metrics": {"ExactMatch": 1.0, "F1": 1.0},
                "selector_trace": {
                    "selected_titles": ["Anchor"],
                    "final_front_titles": ["Anchor"],
                    "gate_decision": {"reason": "offrank_bridge_signal_detected"},
                },
            },
        ],
    }

    summary = summarize_report(report)

    assert summary["delivery_loss_query_count"] == 1
    assert summary["delivery_loss_reason_counts"] == {"structure_saturated_weak_suffix": 1}
    assert summary["selected_support_f1"] > summary["front_support_f1"]
