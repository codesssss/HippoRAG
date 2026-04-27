from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "qbindcert_phase0_sanity.py"
_SPEC = importlib.util.spec_from_file_location("qbindcert_phase0_sanity", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _sample_row() -> dict:
    return {
        "qid": "q1",
        "query_idx": 0,
        "type": "compositional",
        "question": "When did Lothair II's mother die?",
        "answer": "20 March 851",
        "gold_titles": ["Lothair II", "Ermengarde of Tours"],
        "evidences": [
            ["Lothair II", "mother", "Ermengarde of Tours"],
            ["Ermengarde of Tours", "date of death", "20 March 851"],
        ],
        "candidates": [
            {
                "doc_id": 1,
                "rank": 1,
                "title": "Lothair II",
                "text": "Lothair II was the second son of Emperor Lothair I and Ermengarde of Tours.",
                "gold_support": 1,
            },
            {
                "doc_id": 2,
                "rank": 2,
                "title": "Ermengarde of Tours",
                "text": "Ermengarde of Tours (d. 20 March 851) was the daughter of Hugh of Tours.",
                "gold_support": 1,
            },
            {
                "doc_id": 3,
                "rank": 3,
                "title": "Teutberga",
                "text": "Teutberga died 11 November 875.",
                "gold_support": 0,
            },
        ],
    }


def test_oracle_program_contains_answer_terminal_variable() -> None:
    program = _MODULE.oracle_program(_sample_row())

    assert program["answer_variable"] == "y"
    assert any("y" in demand["args"] for demand in program["demands"])
    assert [demand["predicate_bucket"] for demand in program["demands"]] == ["family_relation", "date_of_death"]


def test_certificate_search_accepts_gold_and_rejects_wrong() -> None:
    row = _sample_row()
    program = _MODULE.oracle_program(row)
    props = _MODULE.oracle_propositions(row)

    gold = _MODULE.certificate_for_answer(program, props, "20 March 851")
    wrong = _MODULE.certificate_for_answer(program, props, "11 November 875")

    assert gold["complete"] is True
    assert tuple(gold["key"]) > tuple(wrong["key"])
    assert "answer_terminal" in wrong["reject_reasons"] or "demand_incomplete" in wrong["reject_reasons"]


def test_program_lattice_match_requires_answer_variable_and_connectivity() -> None:
    row = _sample_row()
    oracle = _MODULE.oracle_program(row)
    good = {
        "answer_variable": "y",
        "demands": [
            {"id": "u1", "predicate_bucket": "family_relation", "args": ["x1", "x2"]},
            {"id": "u2", "predicate_bucket": "date_of_death", "args": ["x2", "y"]},
        ],
        "dependencies": [["u1", "u2", "shared:x2"]],
    }
    bad = {
        "answer_variable": "x2",
        "demands": [
            {"id": "u1", "predicate_bucket": "family_relation", "args": ["x1", "x2"]},
            {"id": "u2", "predicate_bucket": "date_of_death", "args": ["x3", "x4"]},
        ],
        "dependencies": [],
    }

    assert _MODULE.program_matches_oracle(good, oracle)
    assert not _MODULE.program_matches_oracle(bad, oracle)


def test_entity_hygiene_detects_same_title_but_avoids_distinct_merge() -> None:
    assert _MODULE.same_title_duplicate("Pirates of the Sky", "Pirates of the Sky")
    assert not _MODULE.normalizer_merge("Pirates of the Sky", "Pirates of the Caribbean")


def test_bucket_aliases_canonicalize_common_llm_outputs() -> None:
    assert _MODULE.canonical_bucket("child_of") == "family_relation"
    assert _MODULE.canonical_bucket("performer") == "performed_by"
    assert _MODULE.bucketize_relation("country of citizenship") == "nationality"


def test_parse_json_payload_strips_think_and_code_fence() -> None:
    raw = "<think>hidden</think>\n```json\n{\"programs\": []}\n```"

    assert _MODULE.parse_json_payload(raw) == {"programs": []}
