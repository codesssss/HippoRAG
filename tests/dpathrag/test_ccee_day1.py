from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "cee_day1_diagnostic.py"
_SPEC = importlib.util.spec_from_file_location("cee_day1_diagnostic", _SCRIPT_PATH)
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


def test_length_normalized_loglik_excludes_pad_tokens() -> None:
    import torch

    logits = torch.tensor(
        [
            [
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [4.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1, 2, -100]])
    value = _MODULE.masked_length_normalized_loglik(logits, labels)[0]
    expected = float(torch.nn.functional.log_softmax(logits, dim=-1)[0, [0, 1], [1, 2]].mean())
    assert abs(value - expected) < 1e-6


def test_identity_stop_delta_and_prior_beat_weak_edit() -> None:
    answers = [{"canonical_text": "alpha", "normalized": "alpha", "prior": 1.0}]
    decision = _MODULE.posterior_decision(
        [{"edit": {"remove_index": 1, "add_index": 5}, "deltas": {"alpha": 1.0}}],
        answers,
        pi_stop=0.9,
    )
    assert decision["is_stop"] is True


def test_posterior_selects_edit_when_delta_beats_stop_prior() -> None:
    answers = [{"canonical_text": "alpha", "normalized": "alpha", "prior": 1.0}]
    decision = _MODULE.posterior_decision(
        [{"edit": {"remove_index": 1, "add_index": 5}, "deltas": {"alpha": 10.0}}],
        answers,
        pi_stop=0.9,
    )
    assert decision["is_stop"] is False
    assert decision["edit"]["add_index"] == 5


def test_answer_clustering_merges_normalized_equivalents_and_drops_empty() -> None:
    clusters = _MODULE.cluster_answers(
        [
            {"source": "a", "text": "The Alpha!", "loglik": -1.0},
            {"source": "b", "text": "alpha", "loglik": -2.0},
            {"source": "c", "text": "!!!", "loglik": 0.0},
        ]
    )
    assert len(clusters) == 1
    assert clusters[0]["normalized"] == "alpha"
    assert clusters[0]["vote_count"] == 2
    assert abs(sum(cluster["prior"] for cluster in clusters) - 1.0) < 1e-6


def test_auc_and_bootstrap_are_deterministic_for_separated_scores() -> None:
    positives = [3.0, 4.0, 5.0]
    negatives = [0.0, 1.0, 2.0]
    assert _MODULE.auc_score(positives, negatives) == 1.0
    first = _MODULE.bootstrap_auc_ci(positives, negatives, seed=17, resamples=20)
    second = _MODULE.bootstrap_auc_ci(positives, negatives, seed=17, resamples=20)
    assert first == second
    assert first["auc"] == 1.0


def test_edit_category_labels_oracle_and_lexical_hn_on_toy_record() -> None:
    record = _record()
    example = _MODULE.make_example(record, max_candidates=8, top_k=5, embedding_features={})
    edits = _MODULE.enumerate_edit_summaries(
        record,
        top_k=5,
        max_candidates=8,
        candidate_pool_size=8,
        example=example,
    )
    beneficial = [edit for edit in edits if edit["remove_index"] == 1 and edit["add_index"] == 5][0]
    lexical = [edit for edit in edits if edit["remove_index"] == 1 and edit["add_index"] == 6][0]
    assert _MODULE.edit_category(beneficial, preferred_remove=4) == "oracle_beneficial"
    assert _MODULE.edit_category(lexical, preferred_remove=1) == "lexical_HN"


def test_dev_rows_slices_fold0_only() -> None:
    rows = [{"qid": str(idx)} for idx in range(1000)]
    dev = _MODULE.dev_rows(rows, 0, 200)
    assert len(dev) == 200
    assert dev[0]["qid"] == "0"
    assert dev[-1]["qid"] == "199"


def test_semantic_capacity_sample_uses_rank_complete_queries_only() -> None:
    rows = [_record()]
    sample = _MODULE.semantic_capacity_rows(rows, top_k=5, max_candidates=8, limit=10, seed=17)
    # The toy top-5 lacks Beta, so it must not enter rank-complete semantic capacity.
    assert sample == []
