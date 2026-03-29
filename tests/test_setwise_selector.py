from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import (
    LEARNED_SETWISE_FEATURE_NAMES,
    collect_lexical_query_seed_entities,
    compute_candidate_feature_rows,
    select_bridge_greedy_positions,
    select_learned_greedy_positions,
)


class DummyReachabilityModel:
    def predict_proba(self, feature_matrix):
        structure_seed_idx = LEARNED_SETWISE_FEATURE_NAMES.index("structure_score_seed")
        structure_covered_idx = LEARNED_SETWISE_FEATURE_NAMES.index("structure_score_covered")
        base_score_idx = LEARNED_SETWISE_FEATURE_NAMES.index("base_score")

        positive_score = (
            0.25 * feature_matrix[:, base_score_idx]
            + 0.30 * feature_matrix[:, structure_seed_idx]
            + 0.45 * feature_matrix[:, structure_covered_idx]
        )
        positive_score = np.clip(positive_score, 0.0, 1.0)
        return np.stack([1.0 - positive_score, positive_score], axis=1)


def test_select_bridge_greedy_positions_prefers_bridge_docs_over_high_rank_distractor():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"film y", "person b"},
            2: {"person a", "birth a"},
            3: {"person b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("person a", "birth a", 1.0, "related_to")],
            3: [("person b", "birth b", 1.0, "related_to")],
            4: [("noise", "other noise", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("birth a", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities={"person a", "person b"},
        anchor_count=2,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["anchor_positions"] == [0, 1]
    assert trace["selection_steps"][2]["doc_id"] == 2
    assert trace["selection_steps"][3]["doc_id"] == 3


def test_select_bridge_greedy_positions_falls_back_to_rank_order_without_structure_signal():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[10, 11, 12, 13],
        pool_doc_scores=np.array([0.80, 0.60, 0.40, 0.20], dtype=float),
        doc_idx_to_entities={
            10: {"a"},
            11: {"b"},
            12: {"c"},
            13: {"d"},
        },
        doc_idx_to_edges={
            10: [],
            11: [],
            12: [],
            13: [],
        },
        adjacency={},
        qa_top_k=3,
        initial_seed_entities=set(),
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 2]
    assert trace["selection_steps"][0]["mode"] == "anchor"
    assert trace["selection_steps"][1]["pool_position"] == 1


def test_select_bridge_greedy_positions_can_bootstrap_from_anchor_entities():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"film y", "person b"},
            2: {"person a", "birth a"},
            3: {"person b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("person a", "birth a", 1.0, "related_to")],
            3: [("person b", "birth b", 1.0, "related_to")],
            4: [("noise", "other noise", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("birth a", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities=set(),
        anchor_count=2,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["selection_steps"][2]["doc_id"] == 2


def test_collect_lexical_query_seed_entities_matches_query_entity_strings():
    seeds = collect_lexical_query_seed_entities(
        query="When did Lothair II's mother die?",
        pool_doc_ids=[0, 1, 2],
        doc_idx_to_entities={
            0: {"donna summer"},
            1: {"lothair ii", "lotharingia"},
            2: {"ermengarde of tours"},
        },
    )

    assert "lothair ii" in seeds


def test_compute_candidate_feature_rows_exposes_bridge_structure_signal():
    feature_rows = compute_candidate_feature_rows(
        query="Who was born first, the director of film x or the director of film y?",
        pool_docs=[
            "Film X\nDirector A made film x.",
            "Film Y\nDirector B made film y.",
            "Director A\nDirector A was born in 1970.",
            "Director B\nDirector B was born in 1980.",
            "Noise\nThis is unrelated noise.",
        ],
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        doc_idx_to_entities={
            0: {"film x", "director a"},
            1: {"film y", "director b"},
            2: {"director a", "birth a"},
            3: {"director b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("director a", "birth a", 1.0, "related_to")],
            3: [("director b", "birth b", 1.0, "related_to")],
            4: [],
        },
        adjacency={
            "director a": [("birth a", 1.0, "related_to")],
            "director b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=5,
        selected_positions=[0, 1],
        seed_entities={"director a", "director b"},
        structure_max_hops=2,
        candidate_positions=[2, 3, 4],
    )

    rows_by_doc_id = {int(row["doc_id"]): row for row in feature_rows if row["doc_id"] is not None}
    assert rows_by_doc_id[2]["structure_score_seed"] > 0.0
    assert rows_by_doc_id[3]["structure_score_seed"] > 0.0
    assert rows_by_doc_id[4]["structure_score_seed"] == 0.0


def test_select_learned_greedy_positions_uses_model_scores_to_pick_bridge_docs():
    model_bundle = {"model": DummyReachabilityModel(), "feature_names": LEARNED_SETWISE_FEATURE_NAMES}
    selected_positions, trace = select_learned_greedy_positions(
        query="Who was born first, the director of film x or the director of film y?",
        pool_docs=[
            "Film X\nDirector A made film x.",
            "Film Y\nDirector B made film y.",
            "Director A\nDirector A was born in 1970.",
            "Director B\nDirector B was born in 1980.",
            "Noise\nThis is unrelated noise.",
        ],
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        doc_idx_to_entities={
            0: {"film x", "director a"},
            1: {"film y", "director b"},
            2: {"director a", "birth a"},
            3: {"director b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("director a", "birth a", 1.0, "related_to")],
            3: [("director b", "birth b", 1.0, "related_to")],
            4: [],
        },
        adjacency={
            "director a": [("birth a", 1.0, "related_to")],
            "director b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        learned_model_bundle=model_bundle,
        initial_seed_entities={"director a", "director b"},
        anchor_count=2,
        structure_max_hops=2,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["selection_steps"][2]["mode"] == "learned_greedy"
    assert trace["selection_steps"][2]["doc_id"] == 2
    assert trace["selection_steps"][3]["doc_id"] == 3
