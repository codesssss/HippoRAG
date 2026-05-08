from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from probe_fixed_pool_candidate_generation import (  # noqa: E402
    build_policy_order,
    build_prompt,
    classify_decision,
    content_tokens,
    extract_doc_ids_from_raw,
    extract_texts_from_raw,
    prompt_has_no_think,
    rank_pool_by_texts,
)


def test_content_tokens_keeps_specific_terms() -> None:
    assert content_tokens("What is the capital punishment policy in that country?") == [
        "capital",
        "punishment",
    ]


def test_all_prompt_builders_start_with_no_think() -> None:
    selector_trace = {
        "requirements": [
            {"unit_id": "s1", "subquery": "Who wrote The Book Thief?", "depends_on": []},
            {"unit_id": "s2", "subquery": "What is the capital punishment policy in that country?", "depends_on": ["s1"]},
        ]
    }
    source_record = {
        "pool_titles": ["Markus Zusak"],
        "pool_docs": ["Markus Zusak is a New Zealand-born Australian writer."],
    }

    for prompt_type in ("query_reform", "demand_reform", "demand_hyde", "listwise_select"):
        messages = build_prompt(
            prompt_type=prompt_type,
            question="What is the death penalty in the country where The Book Thief author has citizenship?",
            selector_trace=selector_trace,
            source_record=source_record,
            max_doc_chars=80,
        )
        assert prompt_has_no_think(messages)
        assert "/no_think" in messages[1]["content"]


def test_extract_texts_from_raw_handles_query_json() -> None:
    raw = '{"queries": [{"text": "capital punishment in New Zealand"}, {"query": "New Zealand death penalty"}]}'

    assert extract_texts_from_raw(raw, prompt_type="query_reform") == [
        "capital punishment in New Zealand",
        "New Zealand death penalty",
    ]


def test_extract_doc_ids_from_raw_handles_objects_and_bounds() -> None:
    raw = '{"doc_ids": [3, {"doc_id": 1}, 105, 3, "2"]}'

    assert extract_doc_ids_from_raw(raw, pool_size=5) == [3, 1, 2]


def test_rank_pool_by_texts_promotes_specific_topical_page() -> None:
    titles = [
        "New Zealand",
        "Geography of New Zealand",
        "Capital punishment in New Zealand",
    ]
    docs = [f"{title}\ntext" for title in titles]

    order, best = rank_pool_by_texts(
        pool_titles=titles,
        pool_docs=docs,
        texts=["capital punishment in New Zealand"],
    )

    assert order[0] == 2
    assert best[2]["score"] > best[0]["score"]


def test_listwise_policy_fills_remaining_source_order() -> None:
    source_record = {
        "pool_titles": ["A", "B", "C", "D"],
        "pool_docs": ["A text", "B text", "C text", "D text"],
    }

    order, best, texts = build_policy_order(
        policy="llm_listwise_select",
        source_record=source_record,
        selector_trace={"requirements": []},
        generated={"listwise_ids": [2, 0]},
    )

    assert order == [2, 0, 1, 3]
    assert best[2]["reason"] == "llm_listwise_selected"
    assert texts == ["2", "0"]


def test_classify_decision_uses_llm_new5_and_new10_thresholds() -> None:
    assert (
        classify_decision(
            [
                {"policy": "llm_query_reform", "new_recall_at_5": 0.25, "new_recall_at_10": 0.35},
                {"policy": "demand_lexical", "new_recall_at_5": 0.5, "new_recall_at_10": 0.5},
            ]
        )
        == "strong_fixed_pool_signal"
    )
    assert classify_decision([{"policy": "llm_query_reform", "new_recall_at_5": 0.1, "new_recall_at_10": 0.4}]) == "weak_fixed_pool_signal"
