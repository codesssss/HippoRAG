from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.daec_dapg_phase0_diagnostics import build_rows, summarize
from scripts.daec_dapg_bridge_baseline_audit import build_reader_rows
from scripts.daec_dapg_phase25_dense_grounding import ApiEmbeddingClient, calibrate_phi, parse_float_list, parse_variant_list, run


def test_phase0_build_rows_categorizes_reader_bottleneck() -> None:
    baseline = [
        {
            "qid": "q1",
            "question": "Q?",
            "answer": "A",
            "gold_titles": ["Gold"],
            "selected_titles": ["Gold"],
            "em": 0.0,
            "f1": 0.0,
            "support_complete": 1.0,
            "support_recall": 1.0,
        }
    ]
    daec = [
        {
            "qid": "q1",
            "question": "Q?",
            "answer": "A",
            "gold_titles": ["Gold"],
            "selected_titles": ["Gold"],
            "em": 0.0,
            "f1": 0.0,
            "support_complete": 1.0,
            "support_recall": 1.0,
            "binding_count": 1,
        }
    ]
    rows = build_rows(baseline, daec, dataset="toy")
    assert rows[0]["failure_category"] == "reader_bottleneck"
    assert rows[0]["reader_wrong_despite_support_complete"] == 1
    assert summarize(rows)["failure_categories"]["reader_bottleneck"] == 1


def test_bridge_fallback_builds_reader_rows_from_pool() -> None:
    records = [
        {
            "qid": "q1",
            "question": "Where was Alice born?",
            "answer": "Paris",
            "gold_titles": ["Alice"],
            "pool_titles": ["Alice", "Paris", "Noise"],
            "pool_docs": [
                "Alice\nAlice was born in Paris.",
                "Paris\nParis is a city.",
                "Noise\nUnrelated text.",
            ],
        }
    ]
    rows = build_reader_rows(records, top_k=2, bridge_seed_count=1, source="bridge_conditioned_fallback")
    assert rows[0]["qid"] == "q1"
    assert len(rows[0]["selected_docs"]) == 2
    assert rows[0]["support_complete"] == 1.0


def test_api_embedding_client_parses_response_and_caches(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return (
                b'{"data": ['
                b'{"index": 0, "embedding": [3.0, 4.0]},'
                b'{"index": 1, "embedding": [0.0, 2.0]}'
                b"]}"
            )

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr("scripts.daec_dapg_phase25_dense_grounding.urllib.request.urlopen", fake_urlopen)
    client = ApiEmbeddingClient(base_url="http://example.test/v1/embeddings", model="toy", batch_size=8, timeout=7)

    vectors = client.embed(["alpha", "beta", "alpha"])

    assert len(calls) == 1
    assert vectors.shape == (3, 2)
    assert vectors[0].round(4).tolist() == [0.6, 0.8]
    assert vectors[1].round(4).tolist() == [0.0, 1.0]
    assert vectors[2].round(4).tolist() == [0.6, 0.8]


def test_phi_calibration_parses_and_scales() -> None:
    phi = np.array([[0.01, 0.5]], dtype=float)

    assert parse_float_list("1, 10") == [1.0, 10.0]
    assert calibrate_phi(phi, scale=10.0, transform="linear_clip").tolist() == [[0.1, 1.0]]

    exp_phi = calibrate_phi(phi, scale=100.0, transform="exp")
    assert 0.63 < float(exp_phi[0, 0]) < 0.64
    assert float(exp_phi[0, 1]) > 0.999


def test_phase25_run_includes_cross_talk_variants() -> None:
    records = [
        {
            "qid": "q1",
            "question": "Where was Alice born and what city is it?",
            "requirements": ["Find where Alice was born", "Find the city"],
            "gold_titles": ["Alice", "Paris"],
            "pool_titles": ["Alice", "Paris", "Noise"],
            "pool_docs": [
                "Alice\nAlice was born in Paris.",
                "Paris\nParis is a city.",
                "Noise\nUnrelated.",
            ],
        }
    ]

    rows = run(
        records,
        dataset="toy",
        top_k=2,
        max_docs=3,
        alpha=0.15,
        horizon=2,
        embedding_backend="tfidf",
        api_client=None,
        phi_scales=[1.0],
        phi_transform="linear_clip",
    )

    variants = {row["variant"] for row in rows}
    assert {
        "dense_query_cosine",
        "dense_demand_union_cosine",
        "channel_cross_talk",
        "demand_source_union",
        "multi_channel_sum",
        "multi_channel_noisy_or",
    } <= variants


def test_phase25_variant_filter_runs_only_requested_variant() -> None:
    records = [
        {
            "qid": "q1",
            "question": "Where was Alice born?",
            "requirements": ["Find where Alice was born"],
            "gold_titles": ["Alice"],
            "pool_titles": ["Alice", "Noise"],
            "pool_docs": ["Alice\nAlice was born in Paris.", "Noise\nUnrelated."],
        }
    ]

    rows = run(
        records,
        dataset="toy",
        top_k=1,
        max_docs=2,
        alpha=0.15,
        horizon=2,
        embedding_backend="tfidf",
        api_client=None,
        phi_scales=[1.0],
        phi_transform="linear_clip",
        variants=parse_variant_list("dense_query_cosine"),
    )

    assert [row["variant"] for row in rows] == ["dense_query_cosine"]
