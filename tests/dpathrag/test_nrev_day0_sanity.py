from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_nrev_day0_sanity.py"
_SPEC = importlib.util.spec_from_file_location("run_nrev_day0_sanity", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_nrev_scores_use_strongest_null() -> None:
    scores = _MODULE.nrev_scores({"l_plus": -1.0, "l_minus": -2.0, "l0": -0.5, "l_alt": -3.0})

    assert scores["l_plus"] == -1.0
    assert scores["rev"] == 1.0
    assert scores["nrev_no_alt"] == -0.5
    assert scores["nrev_full"] == -0.5


def test_type_compatible_alt_excludes_incompatible_answers() -> None:
    assert _MODULE.answer_surface_type("20 March 851", "When did X die?") == "date"
    assert _MODULE.answer_surface_type("Christopher Nolan", "When did X die?") == "entity"
    assert _MODULE.heuristic_question_type("What is the place of birth of the performer of song Changed It?") == "place"
    assert _MODULE.answer_surface_type("Port of Spain", "What is the place of birth of the performer of song Changed It?") == "place"


def test_t_minus_creates_delete_and_replace_contexts() -> None:
    docs = [
        {"title": "A", "text": "A alpha", "doc_id": "a", "rank": 1},
        {"title": "B", "text": "B beta", "doc_id": "b", "rank": 2},
    ]
    pool = docs + [{"title": "A2", "text": "alpha other", "doc_id": "c", "rank": 3}]

    contexts = _MODULE.t_minus_contexts(docs, pool, max_cells=2)
    names = [name for name, _ctx in contexts]

    assert "delete_0" in names
    assert "replace_0" in names
    assert all(len(ctx) >= 1 for _name, ctx in contexts)


def test_matched_replacement_excludes_same_title_duplicates() -> None:
    docs = [
        {"title": "A", "text": "A alpha", "doc_id": "a", "rank": 1},
        {"title": "B", "text": "B beta", "doc_id": "b", "rank": 2},
    ]
    pool = docs + [
        {"title": "A", "text": "A alpha duplicate", "doc_id": "dup", "rank": 3},
        {"title": "C", "text": "alpha other", "doc_id": "c", "rank": 4},
    ]

    replacement = _MODULE.matched_replacement(docs[0], docs, pool)

    assert replacement is not None
    assert replacement["title"] == "C"


def test_extract_final_answer_prefers_final_answer_line() -> None:
    text = "Brief reasoning.\nAnswer: Port of Spain"

    assert _MODULE.extract_final_answer(text) == "Port of Spain"


def test_auc_and_paired_win_rate() -> None:
    rows = [
        {"gold_scores": {"nrev_full": 0.9}, "wrong_scores": {"nrev_full": 0.1}},
        {"gold_scores": {"nrev_full": 0.2}, "wrong_scores": {"nrev_full": 0.8}},
    ]

    assert _MODULE.auc_score([0.9, 0.2], [0.1, 0.8]) == 0.75
    assert _MODULE.paired_win_rate(rows, "nrev_full") == 0.5
