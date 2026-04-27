from __future__ import annotations

import importlib.util
from pathlib import Path
import random


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "caps_day0_nli_sanity.py"
_SPEC = importlib.util.spec_from_file_location("caps_day0_nli_sanity", _SCRIPT_PATH)
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
        "answer": "2001",
        "gold_titles": ["Alpha", "Beta"],
        "evidences": [["Alpha", "mother", "Beta"], ["Beta", "date of death", "2001"]],
        "candidates": [
            _candidate(0, "Alpha", gold=1, text="Alpha's mother was Beta."),
            _candidate(1, "Beta", gold=1, text="Beta died in 2001."),
            _candidate(2, "Gamma", text="Gamma is unrelated."),
            _candidate(3, "Delta", text="Delta is unrelated."),
        ],
    }


def test_verbalize_evidence_triple_uses_template() -> None:
    assert _MODULE.verbalize_evidence_triple(["Alpha", "date_of_death", "2001"]) == "The date of death of Alpha is 2001."


def test_find_subject_doc_prefers_gold_title_match() -> None:
    found = _MODULE.find_subject_doc(_record(), "Beta", max_candidates=4)
    assert found is not None
    index, doc = found
    assert index == 1
    assert doc["title"] == "Beta"


def test_build_obligation_pairs_creates_positive_negative_pairs() -> None:
    pairs, stats = _MODULE.build_obligation_pairs([_record()], max_pairs=2, max_candidates=4, seed=17)
    assert len(pairs) == 4
    assert stats["positive_pairs"] == 2
    assert stats["negative_pairs"] == 2
    assert [row["label"] for row in pairs] == [1, 0, 1, 0]
    assert pairs[0]["hypothesis"] == "The mother of Alpha is Beta."
    assert pairs[0]["doc_title"] == "Alpha"
    assert pairs[1]["doc_title"] in {"Gamma", "Delta"}


def test_choose_negative_doc_excludes_gold_and_positive() -> None:
    record = _record()
    negative = _MODULE.choose_negative_doc(
        record,
        positive_index=0,
        hypothesis="The mother of Alpha is Beta.",
        rng=random.Random(17),
        max_candidates=4,
    )
    assert negative is not None
    index, doc = negative
    assert index in {2, 3}
    assert int(doc.get("gold_support") or 0) == 0


def test_auc_and_gate_decisions() -> None:
    assert _MODULE.auc_score([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert _MODULE.decide_from_auc(0.81) == _MODULE.DECISION_PROCEED
    assert _MODULE.decide_from_auc(0.75) == _MODULE.DECISION_REVIEW
    assert _MODULE.decide_from_auc(0.69) == _MODULE.DECISION_STOP_FAIL


def test_summarize_scores_uses_label_pairs() -> None:
    pairs, _ = _MODULE.build_obligation_pairs([_record()], max_pairs=2, max_candidates=4, seed=17)
    scores = [0.9, 0.1, 0.8, 0.2]
    summary = _MODULE.summarize_scores(pairs, scores, seed=17, resamples=20)
    assert summary["auc"] == 1.0
    assert summary["paired_win_rate"] == 1.0
    assert summary["num_positive"] == 2
    assert summary["num_negative"] == 2
