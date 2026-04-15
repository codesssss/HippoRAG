import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_swap_value_smoke import build_swap_jobs, summarize_results


def test_build_swap_jobs_and_summary():
    corpus = [
        {"title": "Doc A", "text": "text a"},
        {"title": "Doc B", "text": "text b"},
        {"title": "Doc C", "text": "text c"},
        {"title": "Doc D", "text": "text d"},
        {"title": "Doc E", "text": "text e"},
        {"title": "Doc F", "text": "text f"},
    ]
    baseline_report = {
        "expand_assemble_query_traces": [
            {
                "question": "Q1?",
                "query_type": "bridge",
                "gold_answers": ["A"],
                "method_answer": "wrong",
                "method_metrics": {"ExactMatch": 0.0, "F1": 0.2},
                "method_top_titles": ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"],
                "expand_assemble_trace": {
                    "final_front_doc_ids": [0, 1, 2, 3, 4],
                },
            }
        ]
    }
    candidate_report = {
        "expand_assemble_query_traces": [
            {
                "question": "Q1?",
                "query_type": "bridge",
                "gold_answers": ["A"],
                "method_answer": "wrong",
                "method_metrics": {"ExactMatch": 0.0, "F1": 0.2},
                "method_top_titles": ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"],
                "expand_assemble_trace": {
                    "appended_positions": [8],
                    "final_front_doc_ids": [0, 1, 2, 3, 4],
                    "assemble_trace": {
                        "ranking_rows": [
                            {
                                "rank": 6,
                                "pool_position": 8,
                                "doc_id": 5,
                                "title": "Doc F",
                                "source": "append_bridge",
                                "base_score": 0.1,
                                "assemble_score": -0.3,
                            }
                        ]
                    },
                },
            }
        ],
        "llm_name": "qwen3-8b",
        "llm_request_name": "qwen3-8b-train",
        "llm_base_url": "http://localhost:8043/v1",
    }

    jobs, counters = build_swap_jobs(
        dataset="musique",
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        corpus=corpus,
        qa_top_k=5,
    )

    assert counters["aligned_queries"] == 1
    assert counters["queries_with_swap_jobs"] == 1
    assert counters["total_swap_jobs"] == 1
    assert len(jobs) == 1
    assert jobs[0]["weakest_incumbent_doc_id"] == 4
    assert jobs[0]["candidate_doc_id"] == 5
    assert jobs[0]["swapped_top_titles"][-1] == "Doc F"

    swap_results = [
        {
            **jobs[0],
            "swap_answer": "A",
            "swap_metrics": {"ExactMatch": 1.0, "F1": 1.0},
            "delta_metrics": {"ExactMatch": 1.0, "F1": 0.8},
            "reader_trace": {"reader_status": "ok", "reader_cache_hit": False},
        }
    ]
    summary = summarize_results(
        dataset="musique",
        baseline_report_path="baseline.json",
        candidate_report_path="candidate.json",
        output_json="out.json",
        qa_top_k=5,
        llm_name="qwen3-8b",
        llm_request_name="qwen3-8b-train",
        llm_base_url="http://localhost:8043/v1",
        generation_counters=counters,
        swap_results=swap_results,
    )

    assert summary["all_swaps"]["positive_swap_count_em"] == 1
    assert summary["best_swap_oracle"]["queries_with_positive_em"] == 1
    assert summary["best_swap_oracle"]["oracle_delta_em"] == 1.0
