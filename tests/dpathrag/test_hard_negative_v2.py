from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dpathrag_analyze_hard_negatives_v2.py"
_SPEC = importlib.util.spec_from_file_location("dpathrag_analyze_hard_negatives_v2", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

analyze_v2 = _MODULE.analyze_v2
classify_hard_negative = _MODULE.classify_hard_negative
oracle_edit_sequence = _MODULE.oracle_edit_sequence
rank_bucket = _MODULE.rank_bucket


def _candidate(index: int, title: str, *, gold: int = 0, text: str = "body") -> dict:
    return {
        "rank": index + 1,
        "title": title,
        "text": f"{title}\n{text}",
        "retriever_score": 1.0 / (index + 1),
        "gold_support": gold,
    }


def test_rank_bucket_boundaries() -> None:
    assert rank_bucket(6) == "6-10"
    assert rank_bucket(10) == "6-10"
    assert rank_bucket(11) == "11-20"
    assert rank_bucket(20) == "11-20"
    assert rank_bucket(21) == "21-50"
    assert rank_bucket(50) == "21-50"
    assert rank_bucket(51) == "51-100"
    assert rank_bucket(100) == "51-100"
    assert rank_bucket(5) == "outside"


def test_hard_negative_taxonomy_priority_order() -> None:
    assert classify_hard_negative({"answer_in_doc": 1.0, "bridge_entity_in_doc": 1.0}, semantic_threshold=0.5) == "answer_string_distractor"
    assert classify_hard_negative({"answer_in_doc": 0.0, "bridge_entity_in_doc": 1.0}, semantic_threshold=0.5) == "bridge_entity_distractor"
    assert classify_hard_negative({"question_token_coverage": 0.35, "q_doc_cosine": 0.9}, semantic_threshold=0.5) == "lexical_hard_negative"
    assert classify_hard_negative({"question_token_coverage": 0.1, "q_doc_cosine": 0.6}, semantic_threshold=0.5) == "semantic_hard_negative"
    assert classify_hard_negative({"question_token_coverage": 0.1, "q_doc_cosine": 0.1}, semantic_threshold=0.5) == "low_signal_deep_negative"


def test_oracle_edit_improves_missing_gold_and_stops_when_optimal() -> None:
    record = {
        "qid": "q1",
        "question": "Find Alpha and Beta",
        "answer": "Beta",
        "gold_titles": ["Alpha", "Beta"],
        "candidates": [
            _candidate(0, "Alpha", gold=1),
            _candidate(1, "Distractor 1"),
            _candidate(2, "Distractor 2"),
            _candidate(3, "Distractor 3"),
            _candidate(4, "Distractor 4"),
            _candidate(5, "Beta", gold=1),
        ],
    }
    result = oracle_edit_sequence(record, top_k=5, candidate_pool_size=6, max_candidates=6, steps=1)
    assert result["metrics"]["support_complete"] == 1.0
    assert result["added_gold"] == 1
    assert result["removed_non_gold"] == 1
    assert result["edits"]

    optimal = dict(record)
    optimal["candidates"] = [
        _candidate(0, "Alpha", gold=1),
        _candidate(1, "Beta", gold=1),
        _candidate(2, "Distractor 2"),
        _candidate(3, "Distractor 3"),
        _candidate(4, "Distractor 4"),
        _candidate(5, "Distractor 5"),
    ]
    stopped = oracle_edit_sequence(optimal, top_k=5, candidate_pool_size=6, max_candidates=6, steps=1)
    assert stopped["stopped"] is True
    assert stopped["edits"] == []


def test_analyze_v2_query_buckets_and_oracle_summary() -> None:
    cache_rows = [
        {
            "qid": "q1",
            "query_idx": 1,
            "type": "bridge",
            "question": "Find Alpha and Beta",
            "answer": "Beta",
            "gold_titles": ["Alpha", "Beta"],
            "evidences": [["Alpha", "rel", "Bridge"], ["Bridge", "rel", "Beta"]],
            "candidates": [
                _candidate(0, "Alpha", gold=1, text="Alpha Bridge"),
                _candidate(1, "Distractor 1", text="Find Alpha and Beta topical"),
                _candidate(2, "Distractor 2"),
                _candidate(3, "Distractor 3"),
                _candidate(4, "Distractor 4"),
                _candidate(5, "Beta", gold=1, text="Beta Bridge"),
            ],
        },
        {
            "qid": "q2",
            "query_idx": 2,
            "type": "comparison",
            "question": "Compare Gamma and Delta",
            "answer": "Gamma",
            "gold_titles": ["Gamma", "Delta"],
            "candidates": [
                _candidate(0, "Gamma", gold=1),
                _candidate(1, "Delta", gold=1),
                _candidate(2, "Distractor 2"),
                _candidate(3, "Distractor 3"),
                _candidate(4, "Distractor 4"),
                _candidate(5, "Deep NonGold", text="Compare Gamma and Delta"),
            ],
        },
    ]
    predictions = [
        {"qid": "q1", "selected_indices": [0, 5, 1, 2, 3]},
        {"qid": "q2", "selected_indices": [0, 1, 5, 2, 3]},
    ]
    output = analyze_v2(
        cache_rows,
        predictions,
        top_k=5,
        max_candidates=6,
        oracle_pool_size=6,
    )
    assert output["query_level_failure"]["buckets"]["rank_incomplete_and_selector_added_gold"]["queries"] == 1
    assert output["query_level_failure"]["buckets"]["rank_complete_and_selector_changed"]["queries"] == 1
    assert output["oracle_edit_opportunity"]["oracle_edit1"]["support_complete"] == 1.0
    assert output["oracle_edit_opportunity"]["oracle_edit1"]["queries_with_edit"] == 1
