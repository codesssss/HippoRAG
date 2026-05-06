import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dtc_embed_utils import _build_agsto_graph_prior


def test_build_agsto_graph_prior_aligns_metadata_to_pool_doc_ids():
    prior, trace = _build_agsto_graph_prior(
        agsto_metadata={
            "selected_doc_indices": [42],
            "native_dense_doc_indices": [99, 42],
            "retrieved_doc_indices": [7, 42, 99],
        },
        pool_doc_ids=[42, 7, 99, None],
        pool_limit=4,
        w_selected=1.0,
        w_anchor=0.3,
        w_rank=0.2,
    )

    expected = np.asarray([
        (1.0 + 0.3 + 0.1) / 1.5,
        0.2 / 1.5,
        0.3 / 1.5,
        0.0,
    ])
    assert np.allclose(prior, expected)
    assert trace["selected_positions"] == [0]
    assert trace["dense_anchor_positions"] == [0, 2]
    assert trace["retrieved_positions"] == [0, 1]
    assert trace["stats"]["nonzero_count"] == 3


def test_build_agsto_graph_prior_falls_back_to_selected_evidence_set():
    prior, trace = _build_agsto_graph_prior(
        agsto_metadata={
            "selected_evidence_set": {"doc_indices": [11]},
            "native_dense_doc_indices": [],
            "retrieved_doc_indices": [],
        },
        pool_doc_ids=[10, 11],
        pool_limit=2,
    )

    assert np.allclose(prior, np.asarray([0.0, 1.0 / 1.5]))
    assert trace["selected_positions"] == [1]


def test_build_agsto_graph_prior_missing_metadata_returns_zeros():
    prior, trace = _build_agsto_graph_prior(
        agsto_metadata=None,
        pool_doc_ids=[1, 2, 3],
        pool_limit=3,
    )

    assert np.allclose(prior, np.zeros(3))
    assert trace["metadata_present"] is False
    assert trace["stats"]["nonzero_count"] == 0
