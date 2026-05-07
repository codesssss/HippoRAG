from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pilot_dbir_retrieval_expansion import (  # noqa: E402
    LexicalCorpusRetriever,
    SLICE_A,
    SLICE_B,
    classify_decision,
    content_tokens,
    rank_of_gold,
    run_dbir_det,
    run_independent_demand,
    summarize_rows,
)


def test_content_tokens_drops_generic_question_words() -> None:
    assert content_tokens("What is the capital punishment policy in that country?") == [
        "capital",
        "punishment",
        "policy",
    ]


def test_lexical_retriever_ranks_title_and_body_matches() -> None:
    retriever = LexicalCorpusRetriever(
        [
            {"title": "New Zealand", "text": "A country in Oceania."},
            {"title": "Capital punishment in New Zealand", "text": "Capital punishment law and policy."},
            {"title": "Noise", "text": "Unrelated."},
        ]
    )

    results = retriever.retrieve("capital punishment New Zealand", top_k=3)

    assert results[0]["title"] == "Capital punishment in New Zealand"


def test_run_dbir_det_unlocks_downstream_after_root_resolution() -> None:
    retriever = LexicalCorpusRetriever(
        [
            {
                "title": "Markus Zusak",
                "text": "Markus Zusak is a New Zealand-born Australian writer and author of The Book Thief.",
            },
            {
                "title": "Capital punishment in New Zealand",
                "text": "Capital punishment in New Zealand was abolished for murder.",
            },
            {"title": "Capital punishment in Canada", "text": "Capital punishment in Canada was abolished."},
        ]
    )
    selector_trace = {
        "requirements": [
            {
                "unit_id": "s1",
                "subquery": "Who is The Book Thief author?",
                "depends_on": [],
                "anchor_mentions": ["The Book Thief"],
            },
            {
                "unit_id": "s2",
                "subquery": "What is the capital punishment policy in that country?",
                "depends_on": ["s1"],
                "anchor_mentions": [],
            },
        ]
    }

    order, trace = run_dbir_det(
        selector_trace=selector_trace,
        retriever=retriever,
        top_m=2,
        max_iterations=2,
        expanded_k=10,
    )

    assert trace["resolved_slot_count"] == 2
    assert rank_of_gold("Capital punishment in New Zealand", order, retriever) is not None
    assert trace["slot_traces"][1]["slot_id"] == "s2"
    assert "Markus Zusak" in trace["slot_traces"][1]["query"]


def test_independent_demand_does_not_need_upstream_resolution() -> None:
    retriever = LexicalCorpusRetriever(
        [
            {"title": "Root", "text": "Root text."},
            {"title": "Dependent", "text": "Dependent text."},
        ]
    )
    selector_trace = {
        "requirements": [
            {"unit_id": "s1", "subquery": "Root", "depends_on": []},
            {"unit_id": "s2", "subquery": "Dependent", "depends_on": ["s1"]},
        ]
    }

    order, trace = run_independent_demand(
        selector_trace=selector_trace,
        retriever=retriever,
        top_m=1,
        expanded_k=5,
    )

    assert len(order) == 2
    assert trace["retrieval_calls"] == 2
    assert trace["resolved_slot_count"] == 2


def test_summarize_rows_and_decision_require_beating_baselines() -> None:
    rows = []
    for policy, exp50, final5, pool_absent in [
        ("context_iterative_lite", 0, 0, 0),
        ("independent_demand", 0, 0, 0),
        ("dbir_det", 1, 1, 0),
    ]:
        rows.append(
            {
                "pilot_slice": SLICE_A,
                "policy": policy,
                "query_index": 1,
                "expanded_hit_at_5": final5,
                "expanded_hit_at_10": final5,
                "expanded_hit_at_20": final5,
                "expanded_hit_at_50": exp50,
                "expanded_hit_at_100": exp50,
                "final_new_at_5": final5,
                "pool_absent_recovered_at_100": 0,
                "retrieval_calls": 1,
                "slot_count": 1,
                "resolved_slot_count": 1,
                "slot_coverage": 1.0,
                "rank_improvement_vs_source": 10,
            }
        )
    rows.append(
        {
            "pilot_slice": SLICE_B,
            "policy": "dbir_det",
            "query_index": 2,
            "expanded_hit_at_5": 0,
            "expanded_hit_at_10": 0,
            "expanded_hit_at_20": 0,
            "expanded_hit_at_50": 1,
            "expanded_hit_at_100": 1,
            "final_new_at_5": 0,
            "pool_absent_recovered_at_100": 1,
            "retrieval_calls": 1,
            "slot_count": 1,
            "resolved_slot_count": 1,
            "slot_coverage": 1.0,
            "rank_improvement_vs_source": "",
        }
    )

    summary = summarize_rows(rows)

    assert classify_decision(summary) == "strong_go"
