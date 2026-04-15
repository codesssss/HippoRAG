import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_action_swap_oracle_ceiling import (
    build_oracle_action_jobs,
    extract_policy_actions,
    summarize_oracle_results,
)


def _build_corpus():
    return [
        {"title": "Doc A", "text": "Alpha support."},
        {"title": "Doc B", "text": "Beta support."},
        {"title": "Doc C", "text": "Gamma support."},
        {"title": "Doc D", "text": "Delta support."},
        {"title": "Doc E", "text": "Epsilon support."},
        {"title": "Doc F", "text": "Bridge support."},
    ]


def _build_candidate_report():
    return {
        "expand_assemble_query_traces": [
            {
                "question": "Q1?",
                "query_type": "bridge",
                "gold_answers": ["A"],
                "expand_assemble_trace": {
                    "baseline_prefix_positions": [0, 1, 2, 3, 4],
                    "appended_positions": [8],
                    "final_front_doc_ids": [0, 1, 2, 3, 4],
                    "assemble_trace": {
                        "ranking_rows": [
                            {"rank": 1, "pool_position": 0, "doc_id": 0, "title": "Doc A", "source": "baseline_prefix", "assemble_score": 1.0},
                            {"rank": 2, "pool_position": 1, "doc_id": 1, "title": "Doc B", "source": "baseline_prefix", "assemble_score": 0.9},
                            {"rank": 3, "pool_position": 2, "doc_id": 2, "title": "Doc C", "source": "baseline_prefix", "assemble_score": 0.8},
                            {"rank": 4, "pool_position": 3, "doc_id": 3, "title": "Doc D", "source": "baseline_prefix", "assemble_score": 0.7},
                            {"rank": 5, "pool_position": 4, "doc_id": 4, "title": "Doc E", "source": "baseline_prefix", "assemble_score": 0.6},
                            {"rank": 6, "pool_position": 8, "doc_id": 5, "title": "Doc F", "source": "append_bridge", "assemble_score": 0.65, "base_score": 0.1},
                        ]
                    },
                },
            }
        ],
        "llm_name": "qwen3-8b",
        "llm_request_name": "qwen3-8b-train",
        "llm_base_url": "http://localhost:8043/v1",
    }


def test_build_oracle_action_jobs_current_and_relaxed():
    corpus = _build_corpus()
    candidate_report = _build_candidate_report()

    current_jobs, current_counters = build_oracle_action_jobs(
        dataset="musique",
        candidate_report=candidate_report,
        corpus=corpus,
        qa_top_k=5,
        replace_bottom_n=2,
        legality_mode="current",
    )
    relaxed_jobs, relaxed_counters = build_oracle_action_jobs(
        dataset="musique",
        candidate_report=candidate_report,
        corpus=corpus,
        qa_top_k=5,
        replace_bottom_n=2,
        legality_mode="relaxed",
    )

    assert current_counters["processed_queries"] == 1
    assert current_counters["total_keep_jobs"] == 1
    assert current_counters["total_swap_jobs"] == 1
    assert [job["action_type"] for job in current_jobs] == ["keep", "swap"]
    assert current_jobs[1]["replace_pool_position"] == 4
    assert current_jobs[1]["swapped_positions"] == [0, 1, 2, 3, 8]

    assert relaxed_counters["total_swap_jobs"] == 2
    assert [job["action_type"] for job in relaxed_jobs] == ["keep", "swap", "swap"]
    assert sorted(job["replace_pool_position"] for job in relaxed_jobs if job["action_type"] == "swap") == [3, 4]


def test_extract_policy_actions_and_summary_alignment():
    policy_report = {
        "expand_assemble_query_traces": [
            {
                "question": "Q1?",
                "expand_assemble_trace": {
                    "action_mode": "action_swap_v0_dryrun",
                    "action_executed": True,
                    "action_type": "swap",
                    "action_candidate_pool_position": 8,
                    "action_replace_pool_position": 4,
                },
            }
        ]
    }
    judge_report = {
        "expand_assemble_query_traces": [
            {
                "question": "Q1?",
                "expand_assemble_trace": {
                    "action_mode": "action_swap_v0_judge",
                    "action_executed": False,
                    "action_type": "keep",
                },
            }
        ]
    }
    action_results = [
        {
            "question": "Q1?",
            "query_type": "bridge",
            "action_type": "keep",
            "swap_answer": "wrong",
            "swap_metrics": {"ExactMatch": 0.0, "F1": 0.2},
            "candidate_pool_position": None,
            "replace_pool_position": None,
            "candidate_title": None,
            "replace_incumbent_title": None,
            "candidate_assemble_score": None,
            "score_delta": None,
        },
        {
            "question": "Q1?",
            "query_type": "bridge",
            "action_type": "swap",
            "swap_answer": "right",
            "swap_metrics": {"ExactMatch": 1.0, "F1": 1.0},
            "candidate_pool_position": 8,
            "replace_pool_position": 4,
            "candidate_title": "Doc F",
            "replace_incumbent_title": "Doc E",
            "candidate_assemble_score": 0.65,
            "score_delta": 0.05,
        },
        {
            "question": "Q1?",
            "query_type": "bridge",
            "action_type": "swap",
            "swap_answer": "wrong",
            "swap_metrics": {"ExactMatch": 0.0, "F1": 0.1},
            "candidate_pool_position": 8,
            "replace_pool_position": 3,
            "candidate_title": "Doc F",
            "replace_incumbent_title": "Doc D",
            "candidate_assemble_score": 0.65,
            "score_delta": -0.05,
        },
    ]

    summary = summarize_oracle_results(
        dataset="musique",
        candidate_report_path="candidate.json",
        output_json="oracle.json",
        legality_mode="relaxed",
        qa_top_k=5,
        replace_bottom_n=2,
        llm_name="qwen3-8b",
        llm_request_name="qwen3-8b-train",
        llm_base_url="http://localhost:8043/v1",
        generation_counters={
            "total_queries": 1,
            "processed_queries": 1,
            "queries_with_legal_swaps": 1,
            "total_keep_jobs": 1,
            "total_swap_jobs": 2,
        },
        action_results=action_results,
        policy_action_maps={
            "dryrun": extract_policy_actions(policy_report),
            "judge": extract_policy_actions(judge_report),
        },
    )

    assert summary["oracle"]["oracle_positive_queries"] == 1
    assert summary["oracle"]["oracle_delta_em"] == 1.0
    assert summary["oracle"]["oracle_delta_f1"] == 0.8
    assert summary["queries"][0]["oracle_best_action"]["action_type"] == "swap"
    assert summary["queries"][0]["oracle_best_action"]["candidate_pool_position"] == 8
    assert summary["queries"][0]["oracle_best_action"]["replace_pool_position"] == 4

    dryrun_alignment = summary["policy_alignment"]["dryrun"]
    judge_alignment = summary["policy_alignment"]["judge"]
    assert dryrun_alignment["executed_swap_count"] == 1
    assert dryrun_alignment["oracle_hit_rate"] == 100.0
    assert dryrun_alignment["oracle_positive_precision"] == 100.0
    assert dryrun_alignment["oracle_positive_recall"] == 100.0
    assert judge_alignment["keep_count"] == 1
    assert judge_alignment["oracle_hit_rate"] == 0.0
    assert judge_alignment["oracle_positive_recall"] == 0.0
