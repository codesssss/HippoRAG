from __future__ import annotations

import json
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_daec_consistency_probe.py"
_SPEC = importlib.util.spec_from_file_location("run_daec_consistency_probe", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_build_base_records_from_daec_trace_and_pool(tmp_path: Path) -> None:
    report = {
        "setwise_selector_query_traces": [
            {
                "question": "Who directed Film A?",
                "query_type": "bridge",
                "gold_titles": ["Film A", "Director A"],
                "gold_answers": ["Director A"],
                "selector_metrics": {"F1": 1.0},
                "selector_trace": {"selected_pool_positions": [2, 0, 1, 3, 4]},
            }
        ]
    }
    pool = {
        "records": [
            {
                "query_idx": 7,
                "question": "Who directed Film A?",
                "gold_answers": ["Director A"],
                "gold_titles": ["Film A", "Director A"],
                "pool_docs": [
                    "Film A\nFilm A was directed by Director A.",
                    "Distractor\nWrong text.",
                    "Director A\nDirector A is a person.",
                    "Other\nOther text.",
                    "Tail\nTail text.",
                ],
                "pool_titles": ["Film A", "Distractor", "Director A", "Other", "Tail"],
                "pool_doc_ids": ["d0", "d1", "d2", "d3", "d4"],
                "pool_doc_scores": [1.0, 0.9, 0.8, 0.7, 0.6],
            }
        ]
    }
    report_path = tmp_path / "report.json"
    pool_path = tmp_path / "pool.json"
    write_json(report_path, report)
    write_json(pool_path, pool)

    records = _MODULE.build_base_records(
        report_json=report_path,
        pool_json=pool_path,
        limit=1,
        offset=0,
        top_k=5,
        source="test",
    )

    assert len(records) == 1
    assert records[0]["query_idx"] == 7
    assert [doc["title"] for doc in records[0]["selected_docs"]][:3] == ["Director A", "Film A", "Distractor"]
    assert records[0]["support_complete"] == 1.0
    assert records[0]["selected_docs"][0]["gold_support"] == 1


def test_make_variant_records_perturbs_doc_order() -> None:
    base = [
        {
            "qid": "q1",
            "source": "x",
            "selected_docs": [
                {"title": "A", "text": "A\nA"},
                {"title": "B", "text": "B\nB"},
                {"title": "C", "text": "C\nC"},
            ],
        }
    ]

    rows = _MODULE.make_variant_records(base, ["original", "swap01", "drop_last"])
    by_variant = {row["variant"]: row for row in rows}

    assert [doc["title"] for doc in by_variant["original"]["selected_docs"]] == ["A", "B", "C"]
    assert [doc["title"] for doc in by_variant["swap01"]["selected_docs"]] == ["B", "A", "C"]
    assert [doc["title"] for doc in by_variant["drop_last"]["selected_docs"]] == ["A", "B"]


def test_consistency_features_and_auc() -> None:
    features = _MODULE.consistency_features(["Paris", "Paris.", "London", "Paris"])

    assert features["majority_fraction"] == 0.75
    assert features["distinct_answer_count"] == 2.0
    assert 0.0 <= features["normalized_entropy"] <= 1.0
    assert _MODULE.rank_auc([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.7]) == 1.0


def test_aggregate_probe_uses_original_correctness_label() -> None:
    base_records = [
        {
            "qid": "q1",
            "query_idx": 0,
            "question": "Q1",
            "answer": ["Paris"],
            "selected_docs": [{"title": "A", "text": "A"}],
            "support_recall": 1.0,
            "support_complete": 1.0,
        },
        {
            "qid": "q2",
            "query_idx": 1,
            "question": "Q2",
            "answer": ["Berlin"],
            "selected_docs": [{"title": "B", "text": "B"}],
            "support_recall": 0.5,
            "support_complete": 0.0,
        },
    ]
    variant_records = _MODULE.make_variant_records(base_records, ["original", "reverse"])
    predictions = ["Paris", "Paris", "Munich", "Hamburg"]

    summary, per_query, prediction_rows = _MODULE.aggregate_probe(
        base_records,
        variant_records,
        predictions,
        ["original", "reverse"],
        bootstrap_rounds=10,
    )

    assert summary["rows"] == 2
    assert len(per_query) == 2
    assert len(prediction_rows) == 4
    labels = {row["qid"]: row["label_f1_ge_0_5"] for row in per_query}
    assert labels == {"q1": 1, "q2": 0}
