from __future__ import annotations

import importlib.util
from pathlib import Path
import random


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dpathrag_cee_pairwise_common.py"
_SPEC = importlib.util.spec_from_file_location("dpathrag_cee_pairwise_common", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_EVAL_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dpathrag_cee_pairwise_eval.py"
_EVAL_SPEC = importlib.util.spec_from_file_location("dpathrag_cee_pairwise_eval", _EVAL_PATH)
assert _EVAL_SPEC is not None and _EVAL_SPEC.loader is not None
_EVAL = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(_EVAL)


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
        "query_idx": 1,
        "type": "bridge",
        "question": "Find Alpha and Beta clues",
        "answer": "FinalAnswer",
        "gold_titles": ["Alpha", "Beta"],
        "evidences": [["Alpha", "rel", "Bridge"], ["Bridge", "rel", "Beta"]],
        "candidates": [
            _candidate(0, "Alpha", gold=1, text="Alpha Bridge"),
            _candidate(1, "Distractor 1"),
            _candidate(2, "Distractor 2"),
            _candidate(3, "Distractor 3"),
            _candidate(4, "Distractor 4"),
            _candidate(5, "Beta", gold=1, text="Beta Bridge"),
            _candidate(6, "Find Alpha Beta Fake", text="Find Alpha and Beta clues unrelated"),
            _candidate(7, "Low Signal", text="unrelated"),
        ],
    }


def test_edit_summary_marks_oracle_beneficial_edit() -> None:
    record = _record()
    example = _MODULE.make_example(record, max_candidates=8, top_k=5, embedding_features={})
    edit = _MODULE.edit_summary(record, 1, 5, top_k=5, max_candidates=8, example=example)
    assert edit["beneficial"] is True
    assert edit["added_gold"] == 1
    assert tuple(edit["after_objective"]) > tuple(edit["before_objective"])


def test_lexical_hard_negative_rule_for_same_query_candidate() -> None:
    record = _record()
    example = _MODULE.make_example(record, max_candidates=8, top_k=5, embedding_features={})
    edit = _MODULE.edit_summary(record, 1, 6, top_k=5, max_candidates=8, example=example)
    assert edit["beneficial"] is False
    assert edit["hard_negative_type"] == "lexical_hard_negative"


def test_pair_construction_finds_same_remove_positive_vs_lexical_negative() -> None:
    pairs = _MODULE.select_pair_rows(
        _record(),
        top_k=5,
        max_candidates=8,
        candidate_pool_size=8,
        embedding_by_qid={},
        embedding_feature_names=[],
        rng=random.Random(3),
        max_pairs_per_query=10,
    )
    lexical_pairs = [row for row in pairs if row["pair_type"] == "lexical"]
    assert lexical_pairs
    assert any(row["same_remove"] for row in lexical_pairs)
    assert lexical_pairs[0]["positive_edit"]["beneficial"] is True
    assert lexical_pairs[0]["negative_edit"]["hard_negative_type"] == "lexical_hard_negative"


def test_pairwise_linear_scores_toy_positive_above_negative() -> None:
    rows = [
        {"positive_features": [1.0, 2.0], "negative_features": [1.0, -1.0]},
        {"positive_features": [1.0, 3.0], "negative_features": [1.0, 0.0]},
    ]
    model = _MODULE.LinearPairwiseScorer()
    model.fit(rows)
    assert model.score([1.0, 4.0]) > model.score([1.0, -2.0])


def test_platt_probability_and_fixed_cutoff_induce_stop() -> None:
    a, b = _MODULE.platt_fit([3.0, -3.0], [1, 0], steps=50, lr=0.1)
    assert _MODULE.platt_prob(3.0, a, b) > 0.5
    assert _MODULE.platt_prob(-3.0, a, b) < 0.5


def test_pairwise_inference_applies_at_most_one_edit_and_preserves_size() -> None:
    class DummyModel:
        def score(self, features):  # noqa: ANN001
            return float(features[-3])

    preds = _EVAL.evaluate_fold(
        DummyModel(),
        {"a": 1.0, "b": -1.0},
        [_record()],
        embedding_features={},
        embedding_feature_names=[],
        top_k=5,
        max_candidates=8,
        candidate_pool_size=8,
        fold=0,
        variant="toy",
    )
    assert len(preds[0]["selected_indices"]) == 5
    assert len(preds[0]["edits"]) <= 1


def test_gold_labels_are_not_feature_names() -> None:
    pairs = _MODULE.select_pair_rows(
        _record(),
        top_k=5,
        max_candidates=8,
        candidate_pool_size=8,
        embedding_by_qid={},
        embedding_feature_names=[],
        rng=random.Random(5),
        max_pairs_per_query=1,
    )
    assert pairs
    lowered = " ".join(pairs[0]["feature_names"]).lower()
    assert "gold_support" not in lowered
    assert "beneficial" not in lowered
