from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import (
    LEARNED_SETWISE_FEATURE_NAMES,
    build_setwise_late_rerank_candidates,
    collect_lexical_query_seed_entities,
    compute_bridge_gate_decision,
    compute_candidate_feature_rows,
    materialize_reader_top_positions,
    parse_setwise_late_rerank_response,
    rerank_completed_evidence_sets_with_llm,
    score_evidence_state,
    select_bridge_beam_positions,
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


class DummyLateRerankModel:
    def __init__(self, response_text):
        if isinstance(response_text, list):
            self.responses = list(response_text)
        else:
            self.responses = [response_text]
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        response_text = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return response_text, {"finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}


def test_materialize_reader_top_positions_preserves_selected_prefix_and_fills_tail():
    top_positions = materialize_reader_top_positions(
        selected_positions=[3, 1, 3],
        pool_limit=6,
        qa_top_k=5,
    )

    assert top_positions == [3, 1, 0, 2, 4]


def test_parse_setwise_late_rerank_response_accepts_best_id_schema():
    parsed = parse_setwise_late_rerank_response(
        response_text='<JSON>{"best_id": 2, "confidence": 0.8}</JSON>',
        num_candidates=4,
    )

    assert parsed["parse_succeeded"] is True
    assert parsed["best_id"] == 2
    assert parsed["confidence"] == 0.8


def test_rerank_completed_evidence_sets_with_llm_selects_valid_candidate():
    llm_model = DummyLateRerankModel('<JSON>{"best_id": 1, "confidence": 0.67}</JSON>')
    candidate_sets = [
        {
            "source": "heuristic_best",
            "reader_top_positions": [0, 1],
        },
        {
            "source": "baseline_topk",
            "reader_top_positions": [2, 3],
        },
    ]
    best_id, trace = rerank_completed_evidence_sets_with_llm(
        query="Which city is person A from?",
        pool_docs=[
            "Doc A\nPerson A was born in City X.",
            "Doc B\nPerson A won an award.",
            "Doc C\nCity Y is in Country Z.",
            "Doc D\nCity X is in Country Z.",
        ],
        candidate_sets=candidate_sets,
        llm_infer_fn=llm_model.infer,
        model_name="dummy-model",
        max_doc_chars=80,
    )

    assert best_id == 1
    assert trace["parse_succeeded"] is True
    assert trace["selected_candidate_id"] == 1
    assert len(llm_model.calls) == 1


def test_rerank_completed_evidence_sets_with_llm_repairs_invalid_first_response():
    llm_model = DummyLateRerankModel([
        "best_id: 1",
        '<JSON>{"best_id": 0, "confidence": 0.55}</JSON>',
    ])
    best_id, trace = rerank_completed_evidence_sets_with_llm(
        query="Which city is person A from?",
        pool_docs=[
            "Doc A\nPerson A was born in City X.",
            "Doc B\nPerson A won an award.",
            "Doc C\nCity Y is in Country Z.",
            "Doc D\nCity X is in Country Z.",
        ],
        candidate_sets=[
            {"source": "heuristic_best", "reader_top_positions": [0, 1]},
            {"source": "baseline_topk", "reader_top_positions": [2, 3]},
        ],
        llm_infer_fn=llm_model.infer,
        model_name="dummy-model",
        max_doc_chars=80,
    )

    assert best_id == 0
    assert trace["repair_applied"] is True
    assert trace["repair_parse_succeeded"] is True
    assert len(llm_model.calls) == 2


def test_build_setwise_late_rerank_candidates_dedups_equivalent_reader_topk():
    candidates = build_setwise_late_rerank_candidates(
        selected_positions=[0, 2],
        selector_trace={
            "beam_finalists": [
                {"selected_positions": [0, 2]},
                {"selected_positions": [0, 1]},
            ],
        },
        pool_limit=5,
        qa_top_k=3,
        include_baseline=True,
        max_candidates=5,
    )

    assert [candidate["source"] for candidate in candidates] == [
        "heuristic_best",
        "beam_finalist",
    ]
    assert candidates[0]["reader_top_positions"] == [0, 2, 1]
    assert candidates[1]["reader_top_positions"] == [0, 1, 2]


def test_select_bridge_greedy_positions_prefers_bridge_docs_over_high_rank_distractor():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        pool_doc_titles=["Film X", "Film Y", "Director A", "Director B", "Noise"],
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
        pool_doc_titles=["A", "B", "C", "D"],
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
        pool_doc_titles=["Film X", "Film Y", "Director A", "Director B", "Noise"],
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


def test_select_bridge_beam_positions_matches_bridge_completion_case():
    selected_positions, trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        pool_doc_titles=["Film X", "Film Y", "Director A", "Director B", "Noise"],
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
        beam_width=4,
        beam_expand_per_state=4,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["beam_width"] == 4
    assert trace["beam_expand_per_state"] == 4
    assert trace["selection_steps"][2]["doc_id"] == 2
    assert trace["selection_steps"][3]["doc_id"] == 3


def test_select_bridge_beam_positions_recovers_two_step_chain_when_greedy_takes_distractor():
    common_kwargs = dict(
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_scores=np.array([1.0, 0.95, 0.30, 0.05], dtype=float),
        pool_doc_titles=["Film X", "Helper H", "Bridge AB", "Birth B"],
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"person a", "helper h"},
            2: {"person a", "helper h", "person b"},
            3: {"person b", "birth b"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h", 1.0, "related_to")],
            2: [("person a", "helper h", 0.40, "related_to")],
            3: [("person b", "birth b", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=3,
        initial_seed_entities=set(),
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    greedy_positions, greedy_trace = select_bridge_greedy_positions(**common_kwargs)
    beam_positions, beam_trace = select_bridge_beam_positions(
        **common_kwargs,
        beam_width=2,
        beam_expand_per_state=2,
    )

    assert greedy_positions == [0, 1, 3]
    assert beam_positions == [0, 2, 3]
    assert greedy_trace["selection_steps"][1]["doc_id"] == 1
    assert beam_trace["selection_steps"][1]["doc_id"] == 2
    assert beam_trace["selection_steps"][2]["doc_id"] == 3
    assert beam_trace["beam_best_cumulative_score"] > 1.0


def test_select_bridge_greedy_positions_closure_proxy_prefers_query_aligned_bridge():
    common_kwargs = dict(
        pool_doc_ids=[0, 1, 2],
        pool_doc_scores=np.array([1.0, 0.85, 0.84], dtype=float),
        pool_doc_titles=["Anchor", "Award", "Birth"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "award a"},
            2: {"person a", "birth a"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "award a", 1.0, "related_to")],
            2: [("person a", "birth a", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("award a", 1.0, "related_to"), ("birth a", 1.0, "related_to")],
        },
        qa_top_k=2,
        initial_seed_entities={"person a"},
        query_entities={"birth a"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    bridge_positions, _ = select_bridge_greedy_positions(
        **common_kwargs,
        score_mode="bridge",
    )
    closure_positions, closure_trace = select_bridge_greedy_positions(
        **common_kwargs,
        score_mode="closure_proxy",
    )

    assert bridge_positions == [0, 1]
    assert closure_positions == [0, 2]
    assert closure_trace["score_mode"] == "closure_proxy"
    assert closure_trace["selection_steps"][1]["doc_id"] == 2
    assert closure_trace["selection_steps"][1]["closure_score"] > 0.0


def test_select_bridge_greedy_positions_reserve_top_m_keeps_prefix_before_bridge_fill():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.85, 0.30, 0.25], dtype=float),
        pool_doc_titles=["Film X", "Film Y", "Distractor", "Director A", "Director B"],
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"film y", "person b"},
            2: {"distractor"},
            3: {"person a", "birth a"},
            4: {"person b", "birth b"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [("person a", "birth a", 1.0, "related_to")],
            4: [("person b", "birth b", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("birth a", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities={"person a", "person b"},
        anchor_count=2,
        reserve_top_m=3,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["anchor_positions"] == [0, 1]
    assert trace["reserved_positions"] == [0, 1, 2]
    assert trace["selection_steps"][2]["mode"] == "reserve"
    assert trace["selection_steps"][3]["mode"] == "greedy"


def test_select_bridge_beam_positions_non_anchor_title_dedup_skips_duplicate_titles():
    selected_positions, trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_scores=np.array([1.0, 0.9, 0.8, 0.7], dtype=float),
        pool_doc_titles=["Alpha", "Beta", "Beta", "Gamma"],
        doc_idx_to_entities={
            0: {"alpha"},
            1: {"beta"},
            2: {"beta-dup"},
            3: {"gamma"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
        },
        adjacency={},
        qa_top_k=3,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=2,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        beam_width=2,
        beam_expand_per_state=2,
        non_anchor_title_dedup=True,
    )

    assert selected_positions == [0, 1, 3]
    assert trace["non_anchor_title_dedup"] is True
    assert trace["selection_steps"][2]["mode"] == "beam"
    assert trace["selection_steps"][2]["pool_position"] == 3


def test_select_bridge_beam_positions_max_bridge_slots_only_fills_suffix_budget():
    selected_positions, trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.84, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Noise", "Bridge"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"noise"},
            5: {"entity d", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [("entity d", "target", 1.0, "related_to")],
        },
        adjacency={
            "entity d": [("target", 1.0, "related_to")],
        },
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        beam_width=2,
        beam_expand_per_state=2,
    )

    assert selected_positions == [0, 1, 2, 3, 5]
    assert trace["selection_target_k"] == 5
    assert trace["max_bridge_slots"] == 1
    assert trace["selection_steps"][-1]["mode"] == "beam"
    assert trace["selection_steps"][-1]["pool_position"] == 5


def test_score_evidence_state_prefers_connected_chain_over_redundant_helpers():
    pool_doc_ids = [0, 1, 2, 3, 4]
    normalized_base_scores = np.array([1.0, 0.9444, 0.9333, 0.5556, 0.0], dtype=float)
    score_redundant = score_evidence_state(
        pool_doc_ids=pool_doc_ids,
        normalized_base_scores=normalized_base_scores,
        pool_doc_titles=["Anchor", "Helper1", "Helper2", "Bridge", "Answer"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h1"},
            2: {"person a", "helper h2"},
            3: {"person a", "person b"},
            4: {"person b", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h1", 1.0, "related_to")],
            2: [("person a", "helper h2", 1.0, "related_to")],
            3: [("person a", "person b", 0.9, "related_to")],
            4: [("person b", "target", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h1", 1.0, "related_to"), ("helper h2", 1.0, "related_to"), ("person b", 0.9, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        selected_positions=[0, 1, 2],
        fixed_prefix_positions=[0],
        seed_entities={"person a"},
        query_entities={"target"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )
    score_connected_chain = score_evidence_state(
        pool_doc_ids=pool_doc_ids,
        normalized_base_scores=normalized_base_scores,
        pool_doc_titles=["Anchor", "Helper1", "Helper2", "Bridge", "Answer"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h1"},
            2: {"person a", "helper h2"},
            3: {"person a", "person b"},
            4: {"person b", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h1", 1.0, "related_to")],
            2: [("person a", "helper h2", 1.0, "related_to")],
            3: [("person a", "person b", 0.9, "related_to")],
            4: [("person b", "target", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h1", 1.0, "related_to"), ("helper h2", 1.0, "related_to"), ("person b", 0.9, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        selected_positions=[0, 3, 4],
        fixed_prefix_positions=[0],
        seed_entities={"person a"},
        query_entities={"target"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert score_connected_chain["path_connectivity"] == 1.0
    assert score_connected_chain["reachable_doc_ratio"] == 1.0
    assert score_connected_chain["query_reachability"] == 1.0
    assert score_connected_chain["state_score"] > score_redundant["state_score"]


def test_score_evidence_state_distinguishes_query_coverage_from_query_reachability():
    score = score_evidence_state(
        pool_doc_ids=[0, 1, 2],
        normalized_base_scores=np.array([1.0, 0.9, 0.1], dtype=float),
        pool_doc_titles=["Anchor", "Helper", "Answer"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h"},
            2: {"person b", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h", 1.0, "related_to")],
            2: [("person b", "target", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h", 1.0, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        selected_positions=[0, 1, 2],
        fixed_prefix_positions=[0],
        seed_entities={"person a"},
        query_entities={"target"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert score["query_coverage"] == 1.0
    assert score["query_reachability"] == 0.0
    assert score["path_connectivity"] == 0.5
    assert score["reachable_doc_ratio"] == 0.5


def test_score_evidence_state_uses_suffix_base_mean_not_prefix_base_mean():
    score = score_evidence_state(
        pool_doc_ids=[0, 1, 2],
        normalized_base_scores=np.array([1.0, 0.90, 0.20], dtype=float),
        pool_doc_titles=["Anchor", "Prefix", "Suffix"],
        doc_idx_to_entities={
            0: {"anchor"},
            1: {"prefix"},
            2: {"suffix"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
        },
        adjacency={},
        selected_positions=[0, 1, 2],
        fixed_prefix_positions=[0, 1],
        seed_entities={"anchor"},
        query_entities={"suffix"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert score["suffix_base_mean"] == 0.20
    assert score["query_coverage"] == 1.0


def test_score_evidence_state_respects_custom_state_weights():
    score = score_evidence_state(
        pool_doc_ids=[0, 1],
        normalized_base_scores=np.array([1.0, 0.2], dtype=float),
        pool_doc_titles=["Anchor", "Suffix"],
        doc_idx_to_entities={
            0: {"anchor"},
            1: {"target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
        },
        adjacency={},
        selected_positions=[0, 1],
        fixed_prefix_positions=[0],
        seed_entities={"anchor"},
        query_entities={"target"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        state_weight_config={
            "path_connectivity": 0.0,
            "reachable_doc_ratio": 0.0,
            "query_reachability": 0.0,
            "support_mean": 0.0,
            "support_min": 0.0,
            "closure_mean": 0.0,
            "suffix_base_mean": 0.5,
            "query_coverage": 0.5,
            "frontier_ratio": 0.0,
            "redundancy_penalty": 0.0,
        },
    )

    assert score["state_score"] == 0.6
    assert score["state_weight_config"]["suffix_base_mean"] == 0.5


def test_select_bridge_beam_positions_set_closure_reranks_by_state_score():
    common_kwargs = dict(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([1.0, 0.95, 0.94, 0.60, 0.10], dtype=float),
        pool_doc_titles=["Anchor", "Helper1", "Helper2", "Bridge", "Answer"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h1"},
            2: {"person a", "helper h2"},
            3: {"person a", "person b"},
            4: {"person b", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h1", 1.0, "related_to")],
            2: [("person a", "helper h2", 1.0, "related_to")],
            3: [("person a", "person b", 0.9, "related_to")],
            4: [("person b", "target", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h1", 1.0, "related_to"), ("helper h2", 1.0, "related_to"), ("person b", 0.9, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        qa_top_k=3,
        initial_seed_entities={"person a"},
        query_entities={"target"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        beam_width=3,
        beam_expand_per_state=3,
    )

    closure_positions, closure_trace = select_bridge_beam_positions(
        **common_kwargs,
        score_mode="closure_proxy",
    )
    set_positions, set_trace = select_bridge_beam_positions(
        **common_kwargs,
        score_mode="set_closure",
    )

    assert closure_positions == [0, 1, 2]
    assert set_positions == [0, 1, 4]
    assert closure_trace["beam_rank_metric"] == "cumulative_score"
    assert set_trace["beam_rank_metric"] == "state_score"
    assert set_trace["beam_best_state_path_connectivity"] < 1.0
    assert set_trace["beam_best_state_reachable_doc_ratio"] == 1.0
    assert set_trace["beam_best_state_query_reachability"] == 1.0
    assert set_trace["beam_best_state_suffix_base_mean"] > 0.1
    assert set_trace["beam_best_state_score"] > closure_trace["beam_best_state_score"]
    assert set_trace["beam_finalists"][0]["selected_positions"] == [0, 1, 4]
    assert any(finalist["selected_positions"] == [0, 3, 4] for finalist in set_trace["beam_finalists"])


def test_select_bridge_beam_positions_set_closure_stops_before_zero_signal_filler():
    selected_positions, trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([1.0, 0.35, 0.30, 0.95, 0.90], dtype=float),
        pool_doc_titles=["Anchor", "Bridge", "Answer", "Filler1", "Filler2"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "person b"},
            2: {"person b", "target"},
            3: {"noise one"},
            4: {"noise two"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "person b", 1.0, "related_to")],
            2: [("person b", "target", 1.0, "related_to")],
            3: [],
            4: [],
        },
        adjacency={
            "person a": [("person b", 1.0, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities={"person a"},
        query_entities={"target"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        score_mode="set_closure",
        beam_width=4,
        beam_expand_per_state=4,
    )

    assert selected_positions == [0, 1, 2]
    assert trace["selection_target_k"] == 4
    assert len(selected_positions) < trace["selection_target_k"]
    assert trace["beam_set_closure_guard_skip_count"] >= 1
    assert [step["doc_id"] for step in trace["selection_steps"]] == [0, 1, 2]


def test_compute_bridge_gate_decision_skips_without_strong_offrank_bridge_signal():
    decision = compute_bridge_gate_decision(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.84, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Suffix", "Noise"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"suffix"},
            5: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
        },
        adjacency={},
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        gate_mode="suffix_bridge",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.0,
    )

    assert decision["gate_enabled"] is True
    assert decision["use_selector"] is False
    assert decision["reason"] == "offrank_structure_below_threshold"
    assert decision["baseline_suffix_positions"] == [4]
    assert decision["best_offrank_pool_position"] == 5


def test_compute_bridge_gate_decision_applies_for_stronger_offrank_bridge_signal():
    decision = compute_bridge_gate_decision(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.84, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Suffix", "Bridge"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"suffix"},
            5: {"entity d", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [("entity d", "target", 1.0, "related_to")],
        },
        adjacency={
            "entity d": [("target", 1.0, "related_to")],
        },
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        gate_mode="suffix_bridge",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.0,
    )

    assert decision["gate_enabled"] is True
    assert decision["use_selector"] is True
    assert decision["reason"] == "offrank_bridge_signal_detected"
    assert decision["baseline_suffix_positions"] == [4]
    assert decision["best_offrank_pool_position"] == 5
    assert decision["weakest_baseline_suffix_pool_position"] == 4


def test_compute_bridge_gate_decision_skips_when_combined_margin_is_too_small():
    decision = compute_bridge_gate_decision(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.80, 0.79], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Suffix", "Bridge"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"suffix"},
            5: {"entity d", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [("entity d", "target", 1.0, "related_to")],
        },
        adjacency={
            "entity d": [("target", 1.0, "related_to")],
        },
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.9,
        structure_weight=0.05,
        novelty_weight=0.05,
        gate_mode="suffix_bridge",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.05,
    )

    assert decision["gate_enabled"] is True
    assert decision["use_selector"] is False
    assert decision["reason"] == "offrank_combined_margin_too_small"
    assert decision["best_offrank_pool_position"] == 5
    assert decision["weakest_baseline_suffix_pool_position"] == 4


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
