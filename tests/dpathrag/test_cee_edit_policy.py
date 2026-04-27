from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dpathrag_cee_edit_policy.py"
_SPEC = importlib.util.spec_from_file_location("dpathrag_cee_edit_policy", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

apply_action = _MODULE.apply_action
best_support_edit_label = _MODULE.best_support_edit_label
build_training_examples = _MODULE.build_training_examples
evaluate_policy = _MODULE.evaluate_policy


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
        "question": "Find Alpha and Beta",
        "answer": "Beta",
        "gold_titles": ["Alpha", "Beta"],
        "evidences": [["Alpha", "rel", "Bridge"], ["Bridge", "rel", "Beta"]],
        "candidates": [
            _candidate(0, "Alpha", gold=1, text="Alpha Bridge"),
            _candidate(1, "Distractor 1"),
            _candidate(2, "Distractor 2"),
            _candidate(3, "Distractor 3"),
            _candidate(4, "Distractor 4"),
            _candidate(5, "Beta", gold=1, text="Beta Bridge"),
            _candidate(6, "Hard Negative", text="Find Alpha and Beta"),
        ],
    }


def test_edit_action_replaces_one_index() -> None:
    assert apply_action([0, 1, 2, 3, 4], {"type": "edit", "remove_index": 2, "add_index": 5}) == [0, 1, 5, 3, 4]
    assert apply_action([0, 1, 2], {"type": "stop"}) == [0, 1, 2]


def test_best_support_edit_label_requires_gold_add_and_non_gold_remove() -> None:
    record = _record()
    assert best_support_edit_label(record, {"type": "edit", "remove_index": 1, "add_index": 5}, top_k=5) == 1
    assert best_support_edit_label(record, {"type": "edit", "remove_index": 0, "add_index": 5}, top_k=5) == 0
    assert best_support_edit_label(record, {"type": "edit", "remove_index": 1, "add_index": 6}, top_k=5) == 0


def test_build_training_examples_includes_positive_edit_and_negative_stop() -> None:
    features, labels, summary = build_training_examples(
        [_record()],
        max_candidates=7,
        candidate_pool_size=7,
        top_k=5,
        embedding_features={},
        negatives_per_query=3,
        seed=13,
    )
    assert len(features) == len(labels)
    assert summary["positive_edits"] >= 1
    assert summary["negative_stops"] == 1
    assert 1 in labels
    assert 0 in labels


def test_evaluate_policy_applies_gold_edit_when_policy_prefers_it() -> None:
    class DummyPolicy:
        def score(self, features):  # noqa: ANN001
            scores = [-10.0 for _ in features]
            # Edit action order excludes STOP: remove 0 add 5/6, remove 1 add 5/6, ...
            scores[2] = 5.0
            return scores

    summary, predictions = evaluate_policy(
        DummyPolicy(),
        [_record()],
        max_candidates=7,
        candidate_pool_size=7,
        top_k=5,
        steps=1,
        embedding_features={},
        min_edit_margin=0.0,
    )
    assert summary["support_complete"] == 1.0
    assert summary["added_gold"] == 1
    assert summary["added_non_gold"] == 0
    assert predictions[0]["edits"]
