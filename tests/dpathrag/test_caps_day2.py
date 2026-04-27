from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "caps_day2_proof_separability.py"
_SPEC = importlib.util.spec_from_file_location("caps_day2_proof_separability", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _record(answer: str = "Beta") -> dict:
    return {
        "qid": "q1",
        "type": "compositional",
        "question": "Who is Alpha's mother?",
        "answer": answer,
        "evidences": [["Alpha", "mother", "Beta"], ["Beta", "date of death", "2001"]],
    }


def test_oracle_candidate_obligations_replace_object_answer() -> None:
    obligations, substituted = _MODULE.oracle_candidate_obligations(_record("Beta"), "Gamma")
    assert substituted is True
    assert "The mother of Alpha is Gamma." in obligations
    assert "The date of death of Gamma is 2001." in obligations


def test_oracle_candidate_obligations_replace_subject_answer() -> None:
    record = {
        "qid": "q2",
        "answer": "Film A",
        "evidences": [["Film A", "publication date", "1900"], ["Film B", "publication date", "2000"]],
    }
    obligations, substituted = _MODULE.oracle_candidate_obligations(record, "Film B")
    assert substituted is True
    assert "The publication date of Film B is 1900." in obligations


def test_decision_from_summary_uses_conditional_gates() -> None:
    assert (
        _MODULE.decision_from_summary({"top1_accuracy_all": 0.0, "top1_accuracy_cond_gold_present": 0.55, "top3_accuracy_cond_gold_present": 0.0})
        == "PROCEED_DAY3_NON_ORACLE_OBLIGATIONS"
    )
    assert (
        _MODULE.decision_from_summary({"top1_accuracy_all": 0.0, "top1_accuracy_cond_gold_present": 0.0, "top3_accuracy_cond_gold_present": 0.70})
        == "PROCEED_DAY3_NON_ORACLE_OBLIGATIONS"
    )
    assert (
        _MODULE.decision_from_summary({"top1_accuracy_all": 0.1, "top1_accuracy_cond_gold_present": 0.2, "top3_accuracy_cond_gold_present": 0.3})
        == "STOP_CAPS_PROOF_RANKER_FAIL"
    )


def test_summarize_rows_splits_all_and_conditional_accuracy() -> None:
    rows = [
        {"gold_present": True, "gold_rank": 1, "answer_conditioned_available": True, "gold_substituted": True},
        {"gold_present": True, "gold_rank": 4, "answer_conditioned_available": True, "gold_substituted": True},
        {"gold_present": False, "gold_rank": None, "answer_conditioned_available": False, "gold_substituted": False},
    ]
    summary = _MODULE.summarize_rows(rows)
    assert summary["candidate_recall_at_20"] == 0.666667
    assert summary["top1_accuracy_all"] == 0.333333
    assert summary["top1_accuracy_cond_gold_present"] == 0.5
    assert summary["top3_accuracy_cond_gold_present"] == 0.5
