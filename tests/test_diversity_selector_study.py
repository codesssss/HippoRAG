from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diversity_selector_study import (  # noqa: E402
    build_psd_similarity_matrix,
    build_rank_prior_scores,
    finalize_selected_positions,
    order_selected_positions,
    select_dpp_positions,
    select_mmr_positions,
)


def test_build_rank_prior_scores_descends_from_prefix():
    scores = build_rank_prior_scores(5)
    assert scores.shape == (5,)
    assert scores[0] == 1.0
    assert np.all(scores[:-1] >= scores[1:])
    assert scores[-1] == 0.0


def test_build_psd_similarity_matrix_keeps_diagonal_one():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    similarity = build_psd_similarity_matrix(embeddings)
    assert similarity.shape == (3, 3)
    assert np.allclose(np.diag(similarity), 1.0)
    assert similarity[0, 1] > similarity[0, 2]


def test_select_mmr_positions_prefers_diverse_second_pick():
    relevance = np.asarray([1.0, 0.95, 0.70], dtype=np.float32)
    similarity = np.asarray(
        [
            [1.0, 0.99, 0.10],
            [0.99, 1.0, 0.20],
            [0.10, 0.20, 1.0],
        ],
        dtype=np.float32,
    )
    selected = select_mmr_positions(relevance, similarity, top_k=2, lambda_weight=0.5)
    assert selected == [0, 2]


def test_select_dpp_positions_prefers_diverse_second_pick():
    relevance = np.asarray([1.0, 0.95, 0.70], dtype=np.float32)
    similarity = np.asarray(
        [
            [1.0, 0.99, 0.05],
            [0.99, 1.0, 0.10],
            [0.05, 0.10, 1.0],
        ],
        dtype=np.float32,
    )
    selected = select_dpp_positions(relevance, similarity, top_k=2, quality_power=1.0)
    assert selected == [0, 2]


def test_finalize_selected_positions_backfills_prefix_without_duplicates():
    finalized = finalize_selected_positions([3, 3, 1], pool_size=5, top_k=4)
    assert finalized == [3, 1, 0, 2]


def test_order_selected_positions_can_preserve_original_rank():
    assert order_selected_positions([4, 1, 3], order_mode="original_rank") == [1, 3, 4]
    assert order_selected_positions([4, 1, 3], order_mode="selection_order") == [4, 1, 3]
