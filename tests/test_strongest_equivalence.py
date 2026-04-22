from pathlib import Path
import sys

import numpy as np
from scipy import sparse as sp

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag_ext.strongest.candidates import select_union_candidate_indices
from src.hipporag_ext.strongest.runtime import run_strongest_sidecar
from src.hipporag_ext.strongest.requirements import (
    _ground_anchor_to_entities,
    _parsed_to_requirement_units,
    _extract_json_array,
    _strip_think_tags,
)
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


def test_run_strongest_sidecar_gbc_promotes_boundary_completion_without_dropping_anchor():
    docs = ["Doc A", "Doc B", "Doc C", "Doc D"]
    passage_query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    passage_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    incidence = sp.csr_matrix(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
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
        hippo_score_map={0: 0.9, 1: 0.7, 2: 0.4, 3: 0.05},
        trace_state=StrongestTraceState(
            query="where is alpha",
            retrieval_mode="graph",
            passage_prior=np.asarray([0.9, 0.7, 0.4, 0.05], dtype=np.float32),
            entity_prior=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        dense_scores=np.asarray([0.9, 0.7, 0.4, 0.05], dtype=np.float32),
        metadata={
            "passage_chunk_ids": ["chunk-a", "chunk-b", "chunk-c", "chunk-d"],
            "chunk_triples_map": {
                "chunk-a": [("alpha", "linked to", "mid")],
                "chunk-b": [("alpha", "linked to", "other")],
                "chunk-c": [("alpha", "linked to", "target")],
                "chunk-d": [("noise", "linked to", "zzz")],
            },
            "seed_entities": ["alpha"],
            "query_entities": ["alpha"],
            "entity_vocab": ["alpha", "mid", "target", "other"],
        },
    )

    standard_result = run_strongest_sidecar(
        state=state,
        config=StrongestConfig(
            candidate_k=4,
            final_k=2,
            hippo_head_k=2,
            smoothed_union_k=2,
            gamma=1.0,
        ),
    )
    gbc_result = run_strongest_sidecar(
        state=state,
        config=StrongestConfig(
            candidate_k=4,
            final_k=2,
            hippo_head_k=2,
            smoothed_union_k=2,
            gamma=1.0,
            rerank_mode="gbc",
            gbc_protected_anchor_k=1,
            gbc_head_coverage_k=1,
            gbc_top_passage_pool_k=4,
            gbc_frontier_bonus_k=2,
            gbc_bonus_weight=1.0,
        ),
    )

    assert standard_result.final_doc_indices.tolist() == [0, 1]
    assert gbc_result.final_doc_indices.tolist() == [0, 2]
    assert gbc_result.trace["gbc"]["gbc_status"] == "applied"
    assert gbc_result.trace["gbc"]["protected_local_indices"] == [0]
    assert 2 in gbc_result.trace["gbc"]["boundary_local_indices"]


def test_run_strongest_sidecar_ras_fallback_uses_clean_gbc_readout():
    docs = ["Doc A", "Doc B", "Doc C", "Doc D"]
    passage_query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    passage_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    incidence = sp.csr_matrix(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
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
        hippo_score_map={0: 0.9, 1: 0.7, 2: 0.4, 3: 0.05},
        trace_state=StrongestTraceState(
            query="where is alpha",
            retrieval_mode="graph",
            passage_prior=np.asarray([0.9, 0.7, 0.4, 0.05], dtype=np.float32),
            entity_prior=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        dense_scores=np.asarray([0.9, 0.7, 0.4, 0.05], dtype=np.float32),
        metadata={
            "passage_chunk_ids": ["chunk-a", "chunk-b", "chunk-c", "chunk-d"],
            "chunk_triples_map": {
                "chunk-a": [("alpha", "linked to", "mid")],
                "chunk-b": [("alpha", "linked to", "other")],
                "chunk-c": [("alpha", "linked to", "target")],
                "chunk-d": [("noise", "linked to", "zzz")],
            },
            "seed_entities": ["alpha"],
            "query_entities": ["alpha"],
            "entity_vocab": ["alpha", "mid", "target", "other"],
        },
    )

    gbc_result = run_strongest_sidecar(
        state=state,
        config=StrongestConfig(
            candidate_k=4,
            final_k=2,
            hippo_head_k=2,
            smoothed_union_k=2,
            gamma=1.0,
            rerank_mode="gbc",
            gbc_protected_anchor_k=1,
            gbc_head_coverage_k=1,
            gbc_top_passage_pool_k=4,
            gbc_frontier_bonus_k=2,
            gbc_bonus_weight=1.0,
        ),
    )
    ras_fallback_result = run_strongest_sidecar(
        state=state,
        config=StrongestConfig(
            candidate_k=4,
            final_k=2,
            hippo_head_k=2,
            smoothed_union_k=2,
            gamma=1.0,
            rerank_mode="gbc",
            gbc_protected_anchor_k=1,
            gbc_head_coverage_k=1,
            gbc_top_passage_pool_k=4,
            gbc_frontier_bonus_k=2,
            gbc_bonus_weight=1.0,
            ras_enabled=True,
            ras_prefix_guard_k=1,
            ras_requirement_max_units=4,
            ras_enable_conflict_veto=True,
            ras_core_support_min_eligible=True,
        ),
    )

    assert gbc_result.final_doc_indices.tolist() == [0, 2]
    assert ras_fallback_result.final_doc_indices.tolist() == gbc_result.final_doc_indices.tolist()
    assert ras_fallback_result.trace["reranked_candidate_indices"] == gbc_result.trace["reranked_candidate_indices"]
    assert ras_fallback_result.trace["gbc"]["gbc_status"] == "applied"
    assert ras_fallback_result.trace["gbc"]["ras"]["ras_status"] == "fallback_no_requirements"
    assert ras_fallback_result.trace["gbc"]["query_fact_count"] == gbc_result.trace["gbc"]["query_fact_count"]
    assert ras_fallback_result.trace["gbc"]["completion_local_indices"] == gbc_result.trace["gbc"]["completion_local_indices"]


def test_run_strongest_sidecar_ras_promotes_unmet_requirement_completion():
    docs = ["Doc A", "Doc B", "Doc C", "Doc D"]
    passage_query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    passage_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    incidence = sp.csr_matrix(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    state = StrongestBaselineState(
        query="What is the birthplace of alpha?",
        docs=docs,
        passage_query_embedding=passage_query_embedding,
        passage_embeddings=passage_embeddings,
        smoothed_embeddings=np.array(passage_embeddings, copy=True),
        incidence=incidence,
        topology_redundancy_features=incidence.transpose().tocsr(),
        hippo_ranked_indices=[0, 1, 2, 3],
        hippo_score_map={0: 0.9, 1: 0.7, 2: 0.4, 3: 0.05},
        trace_state=StrongestTraceState(
            query="What is the birthplace of alpha?",
            retrieval_mode="graph",
            passage_prior=np.asarray([0.9, 0.7, 0.4, 0.05], dtype=np.float32),
            entity_prior=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        dense_scores=np.asarray([0.9, 0.7, 0.4, 0.05], dtype=np.float32),
        metadata={
            "passage_titles": ["Alpha", "Context", "Birthplace", "Noise"],
            "passage_texts": [
                "Alpha\nalpha overview",
                "Context\nalpha relation witness",
                "Birthplace\nalpha was born in paris",
                "Noise\nirrelevant",
            ],
            "passage_structure_entities": [
                ["alpha"],
                ["alpha"],
                ["alpha", "paris"],
                ["noise"],
            ],
            "passage_chunk_ids": ["chunk-a", "chunk-b", "chunk-c", "chunk-d"],
            "chunk_triples_map": {
                "chunk-a": [("alpha", "related to", "mid")],
                "chunk-b": [("alpha", "linked to", "other")],
                "chunk-c": [("alpha", "born in", "paris")],
                "chunk-d": [("noise", "linked to", "zzz")],
            },
            "seed_entities": ["alpha"],
            "query_entities": ["alpha"],
            "entity_vocab": ["alpha", "mid", "paris", "other"],
        },
    )

    standard_result = run_strongest_sidecar(
        state=state,
        config=StrongestConfig(
            candidate_k=4,
            final_k=2,
            hippo_head_k=2,
            smoothed_union_k=2,
            gamma=1.0,
        ),
    )
    ras_result = run_strongest_sidecar(
        state=state,
        config=StrongestConfig(
            candidate_k=4,
            final_k=2,
            hippo_head_k=2,
            smoothed_union_k=2,
            gamma=1.0,
            rerank_mode="gbc",
            gbc_protected_anchor_k=1,
            gbc_head_coverage_k=1,
            gbc_top_passage_pool_k=4,
            gbc_frontier_bonus_k=2,
            gbc_bonus_weight=1.0,
            ras_enabled=True,
            ras_prefix_guard_k=1,
            ras_requirement_max_units=4,
            ras_enable_conflict_veto=True,
            ras_core_support_min_eligible=True,
        ),
    )

    assert standard_result.final_doc_indices.tolist() == [0, 1]
    assert ras_result.final_doc_indices.tolist() == [2, 0]
    assert ras_result.trace["gbc"]["gbc_status"] == "applied"
    assert ras_result.trace["gbc"]["ras"]["ras_status"] == "applied"
    assert ras_result.trace["gbc"]["ras"]["extractor_trace"]["core_unit_count"] == 1
    assert ras_result.trace["gbc"]["ras"]["protected_prefix_before"] == [0]
    assert ras_result.trace["gbc"]["ras"]["protected_prefix_after"] == [2]


def test_ground_anchor_returns_none_for_unresolved():
    result = _ground_anchor_to_entities(
        "Bertha",
        seed_entities=["Lothair II"],
        query_entities=["Lothair II"],
        baseline_titles=["Lothair II", "Waldrada of Lotharingia", "Teutberga"],
    )
    assert result is None


def test_ground_anchor_matches_exact_seed():
    result = _ground_anchor_to_entities(
        "Lothair II",
        seed_entities=["Lothair II"],
        query_entities=[],
        baseline_titles=[],
    )
    assert result == "lothair ii"


def test_ground_anchor_rejects_short_substring_title():
    # "Waldrada" is only 35% of "Waldrada of Lotharingia" — too loose, should reject
    result = _ground_anchor_to_entities(
        "Waldrada",
        seed_entities=[],
        query_entities=[],
        baseline_titles=["Waldrada of Lotharingia"],
    )
    assert result is None


def test_ground_anchor_matches_quality_substring_title():
    # "Lothair II" (10 chars) vs "Lothair II of Italy" (19 chars) — 53%, should match
    result = _ground_anchor_to_entities(
        "Lothair II",
        seed_entities=[],
        query_entities=[],
        baseline_titles=["Lothair II of Italy"],
    )
    assert result == "lothair ii of italy"


def test_ground_anchor_rejects_bridge_ref():
    result = _ground_anchor_to_entities(
        "<director>",
        seed_entities=["film X"],
        query_entities=[],
        baseline_titles=[],
    )
    assert result is None


def test_parsed_to_requirement_units_filters_ungrounded():
    parsed = [
        {"tier": "support", "anchor": "Inception", "slot_family": "director",
         "expected_answer_type": "person", "is_single_valued": True},
        {"tier": "core", "anchor": "<director>", "slot_family": "birthplace",
         "expected_answer_type": "location", "is_single_valued": True},
    ]
    units, stats = _parsed_to_requirement_units(
        parsed,
        seed_entities=["inception"],
        query_entities=[],
        baseline_titles=["Inception (film)"],
        max_units=4,
    )
    assert len(units) == 2
    assert units[0].tier == "support"
    assert units[0].anchor_entities == ["inception"]
    assert units[1].tier == "core"
    assert units[1].anchor_entities == []
    assert units[1].bridge_targets == [units[0].unit_id]
    assert stats["grounded_count"] == 1
    assert stats["bridge_ref_count"] == 1


def test_parsed_to_requirement_units_drops_unknown_anchor():
    parsed = [
        {"tier": "core", "anchor": "Unknown Person XYZ", "slot_family": "birthplace",
         "expected_answer_type": "location", "is_single_valued": True},
    ]
    units, stats = _parsed_to_requirement_units(
        parsed,
        seed_entities=["Alpha"],
        query_entities=["Alpha"],
        baseline_titles=["Alpha", "Beta"],
        max_units=4,
    )
    assert len(units) == 0
    assert stats["ungrounded_count"] == 1


def test_strip_think_tags():
    assert _strip_think_tags("hello") == "hello"
    assert _strip_think_tags("<think>reasoning</think>[{\"tier\": \"core\"}]") == '[{"tier": "core"}]'
    assert _strip_think_tags("<think>no end tag") == "no end tag"
    assert _strip_think_tags('<think>reasoning [{"tier": "core"}]') == '[{"tier": "core"}]'


def test_extract_json_array():
    assert _extract_json_array('[{"a": 1}]') == [{"a": 1}]
    assert _extract_json_array('some text [{"a": 1}] more') == [{"a": 1}]
    assert _extract_json_array('<think>stuff</think>[{"a": 1}]') == [{"a": 1}]
    assert _extract_json_array('<think>stuff [{"a": 1}]') == [{"a": 1}]
    assert _extract_json_array("no json here") is None
    assert _extract_json_array("") is None
