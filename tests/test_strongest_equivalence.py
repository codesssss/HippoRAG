from pathlib import Path
import sys

import numpy as np
from scipy import sparse as sp

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag_ext.strongest.candidates import select_union_candidate_indices
from src.hipporag_ext.strongest.runtime import run_strongest_sidecar
from src.hipporag_ext.strongest.types import (
    StrongestBaselineState,
    StrongestConfig,
    StrongestTraceState,
)


def test_select_union_candidate_indices_prefers_primary_then_secondary_then_fallback():
    selected = select_union_candidate_indices(
        primary_indices=[4, 3],
        secondary_indices=[3, 2, 1],
        fallback_indices=[1, 0],
        candidate_k=4,
    )
    assert selected.tolist() == [4, 3, 2, 1]


def test_run_strongest_sidecar_returns_deterministic_topk():
    docs = ["Doc A", "Doc B", "Doc C", "Doc D"]
    passage_query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    passage_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    incidence = sp.csr_matrix(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    state = StrongestBaselineState(
        query="where is alpha",
        docs=docs,
        passage_query_embedding=passage_query_embedding,
        passage_embeddings=passage_embeddings,
        smoothed_embeddings=np.array(passage_embeddings, copy=True),
        incidence=incidence,
        topology_redundancy_features=incidence.transpose().tocsr(),
        hippo_ranked_indices=[0, 1, 2, 3],
        hippo_score_map={0: 0.9, 1: 0.8, 2: 0.3, 3: 0.1},
        trace_state=StrongestTraceState(
            query="where is alpha",
            retrieval_mode="graph",
            passage_prior=np.asarray([0.9, 0.8, 0.3, 0.1], dtype=np.float32),
            entity_prior=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        ),
        dense_scores=np.asarray([0.9, 0.8, 0.3, 0.1], dtype=np.float32),
    )
    result = run_strongest_sidecar(
        state=state,
        config=StrongestConfig(
            candidate_k=4,
            final_k=2,
            hippo_head_k=2,
            smoothed_union_k=2,
            gamma=0.2,
        ),
    )

    assert result.trace["status"] == "ok"
    assert result.candidate_indices.tolist() == [0, 1, 2, 3]
    assert result.final_doc_indices.tolist()[:2] == [1, 0]
    assert len(result.final_scores) == 2
    assert result.final_scores[0] >= result.final_scores[1]
