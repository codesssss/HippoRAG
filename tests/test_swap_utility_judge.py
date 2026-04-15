import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_swap_utility_judge import (
    build_claim_support_messages,
    build_swap_utility_messages,
    parse_claim_support_verifier_response,
    parse_swap_utility_judge_response,
    summarize_swap_utility_results,
)
from run_swap_value_smoke import build_swap_jobs


def _build_reports():
    corpus = [
        {"title": "Doc A", "text": "Doc A sentence one. Doc A sentence two. Doc A sentence three."},
        {"title": "Doc B", "text": "Doc B sentence one. Doc B sentence two."},
        {"title": "Doc C", "text": "Doc C sentence one. Doc C sentence two."},
        {"title": "Doc D", "text": "Doc D sentence one. Doc D sentence two."},
        {"title": "Doc E", "text": "Doc E sentence one. Doc E sentence two."},
        {"title": "Doc F", "text": "Doc F sentence one. Doc F sentence two."},
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
    }
    return corpus, baseline_report, candidate_report


def test_parse_swap_utility_judge_response():
    parsed = parse_swap_utility_judge_response(
        "\n".join([
            "VERDICT: helpful",
            "CONFIDENCE: 86",
            "REASON: The candidate adds the missing bridge relation without removing key answer support.",
            "MISSING_INFO_TYPE: bridge_relation",
            "EVIDENCE_SENTENCES: candidate=[1,2] incumbent=[2]",
        ])
    )

    assert parsed["parse_succeeded"] is True
    assert parsed["verdict"] == "helpful"
    assert parsed["confidence"] == 86.0
    assert parsed["missing_info_type"] == "bridge_relation"
    assert parsed["candidate_evidence_sentences"] == [1, 2]
    assert parsed["incumbent_evidence_sentences"] == [2]


def test_parse_swap_utility_judge_response_accepts_sentence_snippets():
    parsed = parse_swap_utility_judge_response(
        "\n".join([
            "VERDICT: harmful",
            "CONFIDENCE: 96",
            "REASON: The candidate drifts away from the query.",
            "MISSING_INFO_TYPE: semantic_drift",
            'EVIDENCE_SENTENCES: candidate=["Sentence A"] incumbent=["Sentence B"]',
        ])
    )

    assert parsed["parse_succeeded"] is True
    assert parsed["candidate_evidence_sentences"] == []
    assert parsed["incumbent_evidence_sentences"] == []
    assert parsed["candidate_evidence_snippets"] == ["Sentence A"]
    assert parsed["incumbent_evidence_snippets"] == ["Sentence B"]


def test_parse_claim_support_verifier_response():
    parsed = parse_claim_support_verifier_response(
        "\n".join([
            "VERDICT: supported",
            "REASON: The snippet directly states the needed relation.",
        ])
    )

    assert parsed["parse_succeeded"] is True
    assert parsed["verdict"] == "supported"
    assert parsed["supported"] is True
    assert "needed relation" in parsed["reason"]


def test_build_claim_support_messages_mentions_binary_format():
    messages = build_claim_support_messages(
        question="Where was Alice born?",
        claim_text="Find the birthplace of Alice.",
        doc_title="Alice",
        witness_text="Alice was born in Paris.",
        witness_unit_type="title_plus_1sent",
        max_doc_chars=200,
    )

    assert messages[0]["role"] == "system"
    assert "directly supports the claim" in messages[1]["content"]
    assert "VERDICT: supported / not_supported" in messages[1]["content"]
    assert messages[1]["content"].startswith("/no_think\n")
    assert "Do not think aloud." in messages[1]["content"]


def test_build_swap_jobs_supports_bottom2_replacements():
    corpus, baseline_report, candidate_report = _build_reports()
    jobs, counters = build_swap_jobs(
        dataset="musique",
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        corpus=corpus,
        qa_top_k=5,
        replace_bottom_n=2,
    )

    assert counters["aligned_queries"] == 1
    assert counters["queries_with_swap_jobs"] == 1
    assert counters["total_swap_jobs"] == 2
    assert len(jobs) == 2
    assert sorted(job["replace_incumbent_rank"] for job in jobs) == [4, 5]
    assert all(job["weakest_incumbent_doc_id"] == 4 for job in jobs)
    assert {job["replace_incumbent_doc_id"] for job in jobs} == {3, 4}


def test_build_swap_utility_messages_mentions_swap_vs_keep():
    corpus, baseline_report, candidate_report = _build_reports()
    jobs, _ = build_swap_jobs(
        dataset="musique",
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        corpus=corpus,
        qa_top_k=5,
        replace_bottom_n=2,
    )

    messages = build_swap_utility_messages(jobs[0], qa_top_k=5, max_doc_chars=200)
    assert messages[0]["role"] == "system"
    assert "swap utility" in messages[0]["content"].lower()
    assert "Compared with keeping the current evidence set unchanged" in messages[1]["content"]
    assert "VERDICT: helpful / neutral / harmful" in messages[1]["content"]


def test_summarize_swap_utility_results_with_oracle_alignment():
    judge_results = [
        {
            "question": "Q1?",
            "candidate_title": "Doc F",
            "replace_incumbent_title": "Doc E",
            "replace_incumbent_rank": 5,
            "candidate_ce_score": 0.3,
            "replace_incumbent_ce_score": 0.2,
            "judge_parsed": {
                "parse_succeeded": True,
                "verdict": "helpful",
                "confidence": 90.0,
                "reason": "Adds the missing bridge.",
                "missing_info_type": "bridge_relation",
            },
        },
        {
            "question": "Q2?",
            "candidate_title": "Doc Y",
            "replace_incumbent_title": "Doc X",
            "replace_incumbent_rank": 5,
            "candidate_ce_score": 0.9,
            "replace_incumbent_ce_score": 0.2,
            "judge_parsed": {
                "parse_succeeded": True,
                "verdict": "harmful",
                "confidence": 80.0,
                "reason": "Semantic drift.",
                "missing_info_type": "semantic_drift",
            },
        },
    ]
    oracle_payload = {
        "metadata": {"output_json": "oracle.json"},
        "swap_results": [
            {
                "question": "Q1?",
                "candidate_title": "Doc F",
                "replace_incumbent_title": "Doc E",
                "replace_incumbent_rank": 5,
                "delta_metrics": {"ExactMatch": 1.0, "F1": 0.6},
            },
            {
                "question": "Q2?",
                "candidate_title": "Doc Y",
                "replace_incumbent_title": "Doc X",
                "replace_incumbent_rank": 5,
                "delta_metrics": {"ExactMatch": 0.0, "F1": 0.0},
            },
        ],
    }
    judge_bundle = type(
        "Bundle",
        (),
        {
            "backend": "responses",
            "model_name": "gpt-5.4",
            "base_url": "https://example.com/v1",
            "reasoning_effort": "medium",
        },
    )()

    summary = summarize_swap_utility_results(
        dataset="musique",
        baseline_report_path="baseline.json",
        candidate_report_path="candidate.json",
        output_json="out.json",
        judge_bundle=judge_bundle,
        generation_counters={"aligned_queries": 2, "queries_with_swap_jobs": 2, "total_swap_jobs": 2},
        judge_results=judge_results,
        replace_bottom_n=2,
        oracle_payload=oracle_payload,
        confidence_threshold=70.0,
    )

    assert summary["judge"]["verdict_counts"]["helpful"] == 1
    assert summary["query_gate"]["queries_with_helpful"] == 1
    assert summary["oracle_alignment"]["matched_jobs"] == 2
    assert summary["oracle_alignment"]["helpful_precision_em"] == 100.0
    assert summary["oracle_alignment"]["judge_blocks_high_ce_nonpositive"] == 1
