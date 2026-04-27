from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRAIN = _load_script("dpathrag_train_selector_warmstart.py")
_HARD_NEG = _load_script("dpathrag_analyze_hard_negatives.py")


def test_split_rows_preserves_legacy_contiguous_split() -> None:
    rows = [{"qid": f"q{i}"} for i in range(10)]
    train_rows, eval_rows, metadata = _TRAIN.split_rows(
        rows,
        train_size=6,
        eval_size=2,
        eval_start=-1,
        train_from_complement=False,
    )
    assert [row["qid"] for row in train_rows] == ["q0", "q1", "q2", "q3", "q4", "q5"]
    assert [row["qid"] for row in eval_rows] == ["q6", "q7"]
    assert metadata["train_eval_qid_overlap"] == 0


def test_split_rows_uses_eval_fold_complement_without_overlap() -> None:
    rows = [{"qid": f"q{i}"} for i in range(10)]
    train_rows, eval_rows, metadata = _TRAIN.split_rows(
        rows,
        train_size=0,
        eval_size=2,
        eval_start=4,
        train_from_complement=True,
    )
    assert [row["qid"] for row in eval_rows] == ["q4", "q5"]
    assert "q4" not in {row["qid"] for row in train_rows}
    assert "q5" not in {row["qid"] for row in train_rows}
    assert len(train_rows) == 8
    assert metadata["train_eval_qid_overlap"] == 0


def test_hard_negative_analysis_categorizes_selector_swaps() -> None:
    cache_rows = [
        {
            "qid": "q1",
            "question": "Which bridge connects Alpha to Answer?",
            "answer": "Answer",
            "evidences": [["Alpha", "rel", "Bridge"], ["Bridge", "rel", "Answer"]],
            "candidates": [
                {"rank": 1, "title": "Retained Distractor", "text": "Retained Distractor\nAlpha unrelated", "gold_support": 0},
                {"rank": 2, "title": "Removed Distractor", "text": "Removed Distractor\nnoise", "gold_support": 0},
                {"rank": 3, "title": "Removed Gold", "text": "Removed Gold\nAnswer appears", "gold_support": 1},
                {"rank": 4, "title": "Added Hard Negative", "text": "Added Hard Negative\nBridge but wrong answer", "gold_support": 0},
                {"rank": 5, "title": "Added Gold", "text": "Added Gold\nAnswer and Bridge", "gold_support": 1},
            ],
        }
    ]
    prediction_rows = [{"qid": "q1", "selected_indices": [0, 3, 4]}]
    output = _HARD_NEG.analyze_hard_negatives(cache_rows, prediction_rows, top_k=3, max_candidates=5)

    assert output["categories"]["rank_retained_non_gold"]["count"] == 1
    assert output["categories"]["rank_removed_non_gold"]["count"] == 1
    assert output["categories"]["selector_added_non_gold"]["count"] == 1
    assert output["categories"]["selector_added_gold"]["count"] == 1
    assert output["categories"]["selector_added_gold"]["features"]["answer_in_doc"]["mean"] == 1.0
    assert output["query_counts"]["selector_added_non_gold"] == 1
