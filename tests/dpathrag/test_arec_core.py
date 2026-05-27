from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dpathrag.arec.closure import closure_score, greedy_closure_select, marginal_gains, obligation_closure
from src.dpathrag.arec.obligations import normalize_obligations, parse_obligations
from src.dpathrag.arec.pool import PoolDoc
from src.dpathrag.arec.retrieval import missing_support_hit_rate, top_indices_for_queries
from src.dpathrag.arec.verifier import LexicalSmokeVerifier, support_matrix


def test_obligation_parser_filters_duplicates_and_answer_restatements() -> None:
    raw = {
        "obligations": [
            {"id": "o1", "claim": "Film X was directed by Alice.", "retrieval_active": True},
            {"id": "o2", "claim": "Film X was directed by Alice.", "retrieval_active": True},
            {"id": "o3", "claim": "Therefore the answer is Italy.", "retrieval_active": True},
            {"id": "o4", "claim": "Compare Alice and Bob.", "type": "comparison_control", "retrieval_active": True},
        ]
    }

    obligations = normalize_obligations(parse_obligations(raw, answer="Italy"), answer="Italy")

    assert [item.id for item in obligations] == ["o1", "o4"]
    assert obligations[1].retrieval_active is False


def test_closure_and_marginal_gain_match_noisy_or_formula() -> None:
    matrix = np.array(
        [
            [0.5, 0.2, 0.0],
            [0.0, 0.4, 0.6],
        ]
    )

    assert np.allclose(obligation_closure(matrix, [0]), [0.5, 0.0])
    gains = marginal_gains(matrix, [0])
    assert np.isneginf(gains[0])
    assert np.allclose(gains[1:], [0.5, 0.6])
    selected, objective = greedy_closure_select(matrix, budget=2)
    assert selected == [1, 0]
    assert closure_score(matrix, selected) == objective


def test_retrieval_and_missing_hit_rate() -> None:
    docs = [
        PoolDoc(index=0, title="Alice", text="Alice was born in Paris.", full_text="Alice\nAlice was born in Paris."),
        PoolDoc(index=1, title="Paris", text="Paris is a city in France.", full_text="Paris\nParis is a city in France."),
        PoolDoc(index=2, title="Noise", text="Unrelated.", full_text="Noise\nUnrelated."),
    ]

    indices = top_indices_for_queries(["Paris city France"], docs, per_query_k=1, exclude_titles=["Alice"])

    assert indices == [1]
    assert missing_support_hit_rate(["Alice", "Paris"], ["Alice"], ["Paris"]) == 1.0


def test_lexical_verifier_support_matrix_bounds() -> None:
    verifier = LexicalSmokeVerifier(threshold=0.0)
    matrix = support_matrix(["Alice was born in Paris"], ["Alice was born in Paris.", "Unrelated."], verifier)

    assert len(matrix) == 1
    assert matrix[0][0] > matrix[0][1]
    assert all(0.0 <= score <= 1.0 for row in matrix for score in row)
