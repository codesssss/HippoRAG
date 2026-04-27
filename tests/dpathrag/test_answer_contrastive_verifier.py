from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_answer_contrastive_verifier.py"
_SPEC = importlib.util.spec_from_file_location("run_answer_contrastive_verifier", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(qid: str, idx: int, label: int, score: float, qtype: str = "bridge") -> dict:
    return {
        "row_id": f"{qid}::{idx}",
        "qid": qid,
        "type": qtype,
        "question": "When did Alpha die?",
        "candidate_index": idx,
        "candidate_text": "1900" if label else "Gamma",
        "label": label,
        "proof_score": score,
        "model_score": score,
        "candidate_score": 10.0,
        "candidate_max_score": 10.0,
        "source_count": 1,
        "sources": ["date"] if label else ["title"],
        "substituted": False,
        "obligation_count": 2,
        "doc_count": 3,
        "answer_conditioned_available": True,
    }


def test_summarize_ranking_splits_all_and_conditional_metrics() -> None:
    rows = [
        _row("q1", 0, 1, 0.9),
        _row("q1", 1, 0, 0.1),
        _row("q2", 0, 0, 0.8),
        _row("q2", 1, 1, 0.7),
        _row("q3", 0, 0, 0.5),
        _row("q3", 1, 0, 0.4),
    ]
    summary = _MODULE.summarize_ranking(rows, "model_score")
    assert summary["queries"] == 3
    assert summary["candidate_recall_at_20"] == 0.666667
    assert summary["top1_accuracy_all"] == 0.333333
    assert summary["top1_accuracy_cond_gold_present"] == 0.5
    assert summary["top3_accuracy_cond_gold_present"] == 1.0
    assert summary["mrr_cond_gold_present"] == 0.75


def test_decision_from_metrics_uses_green_yellow_red_gates() -> None:
    assert (
        _MODULE.decision_from_metrics(
            {"gold_vs_best_wrong_auc": 0.76, "top1_accuracy_cond_gold_present": 0.56, "top3_accuracy_cond_gold_present": 0.6},
            {"ci_low": 0.71},
        )
        == "SUCCESS_ANSWER_CONTRASTIVE_VERIFIER"
    )
    assert (
        _MODULE.decision_from_metrics(
            {"gold_vs_best_wrong_auc": 0.69, "top1_accuracy_cond_gold_present": 0.4, "top3_accuracy_cond_gold_present": 0.7},
            {"ci_low": 0.5},
        )
        == "PARTIAL_TOPK_RECOVERY_NOT_MAINLINE"
    )
    assert (
        _MODULE.decision_from_metrics(
            {"gold_vs_best_wrong_auc": 0.69, "top1_accuracy_cond_gold_present": 0.4, "top3_accuracy_cond_gold_present": 0.69},
            {"ci_low": 0.5},
        )
        == "FAIL_ANSWER_CONTRASTIVE_VERIFIER"
    )


def test_candidate_feature_vector_matches_feature_names() -> None:
    row = _row("q1", 0, 1, 0.8)
    features = _MODULE.candidate_feature_vector(row)
    assert len(features) == len(_MODULE.feature_names())
    assert features[_MODULE.feature_names().index("candidate_matches_date_kind")] == 1.0
    assert features[_MODULE.feature_names().index("source_date")] == 1.0
