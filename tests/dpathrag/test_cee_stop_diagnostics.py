from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUCKET = _load_script("dpathrag_cee_bucket_diagnostic.py")
_STOP = _load_script("dpathrag_cee_oracle_stop_eval.py")
_LEARNED_STOP = _load_script("dpathrag_cee_learned_stop_eval.py")


def _candidate(index: int, title: str, *, gold: int = 0, text: str = "body") -> dict:
    return {
        "rank": index + 1,
        "title": title,
        "text": f"{title}\n{text}",
        "retriever_score": 1.0 / (index + 1),
        "gold_support": gold,
    }


def _record(*, complete: bool = False) -> dict:
    candidates = [
        _candidate(0, "Alpha", gold=1, text="Alpha Bridge"),
        _candidate(1, "Beta" if complete else "Distractor 1", gold=1 if complete else 0, text="Beta Bridge" if complete else "noise"),
        _candidate(2, "Distractor 2"),
        _candidate(3, "Distractor 3"),
        _candidate(4, "Distractor 4"),
        _candidate(5, "Beta", gold=0 if complete else 1, text="Beta Bridge"),
        _candidate(6, "Hard Negative", text="Find Alpha and Beta"),
    ]
    return {
        "qid": "q_complete" if complete else "q_missing",
        "query_idx": 1 if complete else 2,
        "type": "bridge",
        "question": "Find Alpha and Beta",
        "answer": "Beta",
        "gold_titles": ["Alpha", "Beta"],
        "evidences": [["Alpha", "rel", "Bridge"], ["Bridge", "rel", "Beta"]],
        "candidates": candidates,
    }


def test_bucket_diagnostic_splits_complete_and_incomplete_edits() -> None:
    cache_rows = [_record(complete=True), _record(complete=False)]
    predictions = [
        {"qid": "q_complete", "variant": "learned_edit1", "selected_indices": [0, 1, 2, 3, 6], "edits": [{"remove_index": 4, "add_index": 6}]},
        {"qid": "q_missing", "variant": "learned_edit1", "selected_indices": [0, 5, 2, 3, 4], "edits": [{"remove_index": 1, "add_index": 5}]},
    ]
    output = _BUCKET.analyze_config(
        cache_rows,
        predictions,
        config_name="toy",
        variant="learned_edit1",
        top_k=5,
        max_candidates=7,
        oracle_pool_size=7,
        embedding_by_qid={},
        embedding_feature_names=[],
    )
    assert output["buckets"]["rank_complete_and_edited"]["queries"] == 1
    assert output["buckets"]["rank_incomplete_and_edited"]["queries"] == 1
    assert output["buckets"]["oracle_beneficial_and_edited"]["queries"] == 1
    assert output["buckets"]["oracle_not_beneficial_and_edited"]["queries"] == 1


def test_oracle_stop_rules_force_expected_stops() -> None:
    complete = _record(complete=True)
    missing = _record(complete=False)
    complete_example = _STOP.make_example(complete, max_candidates=7, top_k=5, embedding_features={})
    missing_example = _STOP.make_example(missing, max_candidates=7, top_k=5, embedding_features={})
    assert _STOP.should_force_stop(
        complete,
        stop_rule="rank_complete",
        top_k=5,
        max_candidates=7,
        candidate_pool_size=7,
        example=complete_example,
    )
    assert not _STOP.should_force_stop(
        missing,
        stop_rule="rank_complete",
        top_k=5,
        max_candidates=7,
        candidate_pool_size=7,
        example=missing_example,
    )
    assert _STOP.should_force_stop(
        complete,
        stop_rule="objective_oracle",
        top_k=5,
        max_candidates=7,
        candidate_pool_size=7,
        example=complete_example,
    )
    assert not _STOP.should_force_stop(
        missing,
        stop_rule="objective_oracle",
        top_k=5,
        max_candidates=7,
        candidate_pool_size=7,
        example=missing_example,
    )


def test_learned_stop_features_and_labels_are_query_level() -> None:
    complete = _record(complete=True)
    missing = _record(complete=False)
    features, labels, qids = _LEARNED_STOP.build_stop_examples(
        [complete, missing],
        top_k=5,
        max_candidates=7,
        candidate_pool_size=7,
        embedding_features={},
    )
    assert len(features) == 2
    assert labels == [0, 1]
    assert qids == ["q_complete", "q_missing"]

    policy = _LEARNED_STOP.NeedEditPolicy()
    report = policy.fit(features, labels)
    assert report["positive_rate"] == 0.5
    scores = policy.score(features)
    assert len(scores) == 2
    metrics = _LEARNED_STOP.stop_classification_metrics(labels, scores, threshold=sum(scores) / len(scores))
    assert set(metrics) >= {"auc", "accuracy", "precision", "recall"}
