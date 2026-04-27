from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "caps_day1_gates.py"
_SPEC = importlib.util.spec_from_file_location("caps_day1_gates", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _candidate(index: int, title: str, *, gold: int = 0, text: str = "body") -> dict:
    return {
        "rank": index + 1,
        "title": title,
        "text": f"{title}\n{text}",
        "retriever_score": 1.0 / (index + 1),
        "gold_support": gold,
    }


def _record() -> dict:
    return {
        "qid": "q1",
        "question": "When did Alpha's mother die?",
        "answer": "20 March 851",
        "gold_titles": ["Alpha", "Beta"],
        "evidences": [["Alpha", "mother", "Beta"], ["Beta", "date of death", "20 March 851"]],
        "candidates": [
            _candidate(0, "Alpha", gold=1, text="Alpha's mother was Beta."),
            _candidate(1, "Beta", gold=1, text="Beta died on 20 March 851."),
            _candidate(2, "Gamma", text="Gamma died in 1999."),
        ],
    }


def test_cluster_answer_candidates_merges_normalized_duplicates() -> None:
    rows = _MODULE.cluster_answer_candidates(
        [
            {"text": "The Alpha!", "source": "a", "score": 1.0},
            {"text": "alpha", "source": "b", "score": 2.0},
            {"text": "Beta", "source": "c", "score": 1.0},
        ],
        cap=5,
    )
    assert rows[0]["normalized"] == "alpha"
    assert rows[0]["score"] == 4.0
    assert rows[0]["source_count"] == 2


def test_extract_heuristic_answers_finds_dates_titles_and_yes_no() -> None:
    record = _record()
    record["question"] = "Are Alpha and Beta related?"
    answers = _MODULE.extract_heuristic_answers(record, record["candidates"], max_per_source=30)
    normalized = {_MODULE.normalize_answer(row["text"]) for row in answers}
    assert "yes" in normalized
    assert "no" in normalized
    assert "alpha" in normalized
    assert "20 march 851" in normalized


def test_candidate_obligations_substitute_final_answer_slot() -> None:
    obligations, substituted = _MODULE.candidate_obligations(_record(), "Wrong Date")
    assert substituted is True
    assert "The date of death of Beta is Wrong Date." in obligations
    assert "The mother of Alpha is Beta." in obligations


def test_answer_recall_at_uses_normalized_gold() -> None:
    candidates = [
        {"normalized": "alpha", "text": "Alpha"},
        {"normalized": "20 march 851", "text": "20 March 851"},
    ]
    assert _MODULE.answer_recall_at(candidates, ["20 March 851"], 1) == 0.0
    assert _MODULE.answer_recall_at(candidates, ["20 March 851"], 2) == 1.0


def test_proof_score_and_greedy_selection() -> None:
    matrix = [
        [0.9, 0.1, 0.2],
        [0.1, 0.8, 0.2],
    ]
    assert abs(_MODULE.proof_score_from_matrix(matrix) - 0.72) < 1e-6
    selected = _MODULE.greedy_proof_indices(matrix, [0, 1, 2], threshold=0.7, max_docs=2)
    assert selected == [0, 1]


def test_dedup_union_records_uses_title_normalization() -> None:
    left = {"candidates": [_candidate(0, "Alpha"), _candidate(1, "Beta")]}
    right = {"candidates": [_candidate(0, "alpha"), _candidate(1, "Gamma")]}
    docs = _MODULE.dedup_union_records([left, right], pool_k=2, cap=10)
    assert [doc["title"] for doc in docs] == ["Alpha", "Beta", "Gamma"]
