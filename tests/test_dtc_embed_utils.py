from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dtc_embed_utils import (  # noqa: E402
    DTCRequirement,
    parse_dtc_decomposition_response,
    select_daec_noisyor_positions,
    select_dtc_embed_positions,
)


def test_parse_dtc_decomposition_response_accepts_dependency_schema():
    raw = """
    <think>ignore this</think>
    [
      {
        "id": "s1",
        "subquery": "Who directed Film X?",
        "depends_on": [],
        "expected_answer_type": "person",
        "anchor_mentions": ["Film X"],
        "role": "bridge",
        "satisfiable_by": "document"
      },
      {
        "id": "s2",
        "subquery": "Where was that director born?",
        "depends_on": ["s1"],
        "expected_answer_type": "location",
        "anchor_mentions": [],
        "role": "answer",
        "satisfiable_by": "document"
      }
    ]
    """
    requirements, trace = parse_dtc_decomposition_response(raw, max_steps=4)

    assert trace["parse_succeeded"] is True
    assert [req.unit_id for req in requirements] == ["s1", "s2"]
    assert requirements[1].depends_on == ("s1",)
    assert requirements[0].anchor_mentions == ("Film X",)
    assert requirements[0].satisfiable_by == "document"
    assert requirements[1].satisfiable_by == "document"


def test_parse_dtc_decomposition_response_keeps_satisfiable_by_optional():
    raw = """
    [
      {
        "id": "s1",
        "subquery": "Who directed Film X?",
        "depends_on": [],
        "expected_answer_type": "person",
        "anchor_mentions": ["Film X"],
        "role": "bridge"
      }
    ]
    """
    requirements, trace = parse_dtc_decomposition_response(raw, max_steps=4)

    assert trace["parse_succeeded"] is True
    assert len(requirements) == 1
    assert requirements[0].satisfiable_by == "unknown"


def test_select_dtc_embed_positions_covers_dependent_requirement_after_support():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Where was that director born?",
            depends_on=("s1",),
            expected_answer_type="location",
            role="answer",
            satisfiable_by="inference",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0, 0.0]),
        "s2": np.asarray([0.0, 1.0, 0.0]),
    }
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Distractor\nThis page is unrelated.",
        "Alice\nAlice was born in Paris.",
        "Other\nAnother page.",
    ]
    pool_doc_ids = [0, 1, 2, 3]
    passage_embeddings = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.1, 0.8],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Where was the director of Film X born?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.9, 0.2, 0.1]),
        pool_doc_titles=["Film X", "Distractor", "Alice", "Other"],
        doc_idx_to_entities={
            0: {"film x", "alice"},
            1: {"distractor"},
            2: {"alice", "paris"},
            3: {"other"},
        },
        passage_embeddings=passage_embeddings,
        qa_top_k=3,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.1,
        dependency_bonus_weight=0.1,
    )

    assert selected[:2] == [0, 2]
    assert trace["covered_requirement_count"] == 2
    assert trace["cover_position_by_requirement"]["s1"] == 0
    assert trace["cover_position_by_requirement"]["s2"] == 2


def test_select_dtc_embed_positions_uses_title_binding_for_dependent_requirement():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            expected_answer_type="person",
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Where was that director born?",
            depends_on=("s1",),
            expected_answer_type="location",
            role="answer",
            satisfiable_by="inference",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "s2": np.asarray([0.0, 1.0, 0.0, 0.0]),
    }
    pool_docs = [
        "Film X\nFilm X was directed by Alice and released in 1999.",
        "Place of birth\nA generic page about the concept of places of birth.",
        "Alice\nAlice was born in Paris.",
        "Other\nAnother page.",
    ]
    pool_doc_ids = [0, 1, 2, 3]
    passage_embeddings = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    def embed_bound_texts(texts):
        return {
            text: np.asarray([0.0, 0.0, 1.0, 0.0])
            for text in texts
            if "Alice" in text
        }

    selected, trace = select_dtc_embed_positions(
        query="Where was the director of Film X born?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.95, 0.2, 0.1]),
        pool_doc_titles=["Film X", "Place of birth", "Alice", "Other"],
        doc_idx_to_entities={
            0: {"film x", "alice"},
            1: {"place of birth"},
            2: {"alice", "paris"},
            3: {"other"},
        },
        passage_embeddings=passage_embeddings,
        qa_top_k=3,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.1,
        dependency_bonus_weight=0.1,
        enable_dependency_binding=True,
        binding_max_candidates=4,
        binding_entity_hit_required=True,
        repairable_filter_enabled=True,
        embed_texts_fn=embed_bound_texts,
    )

    assert selected[:2] == [0, 2]
    assert trace["cover_position_by_requirement"]["s2"] == 2
    assert trace["binding_candidates_by_requirement"]["s2"][0]["title"] == "Alice"
    assert trace["repairable_by_requirement"]["s2"]["repairable"] is True
    assert trace["repairable_by_requirement"]["s2"]["reason"] == "resolved_dependency_binding"
    assert trace["repairable_by_requirement"]["s2"]["raw_inference_veto"] is True
    assert trace["repairable_by_requirement"]["s2"]["inference_veto"] is False
    assert trace["repairable_by_requirement"]["s2"]["satisfiable_by_policy"] == "binding_override"


def test_strict_satisfiable_by_policy_vetoes_bound_inference_requirement():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            expected_answer_type="person",
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Where was that director born?",
            depends_on=("s1",),
            expected_answer_type="location",
            role="answer",
            satisfiable_by="inference",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "s2": np.asarray([0.0, 1.0, 0.0, 0.0]),
    }
    pool_docs = [
        "Film X\nFilm X was directed by Alice and released in 1999.",
        "Place of birth\nA generic page about the concept of places of birth.",
        "Alice\nAlice was born in Paris.",
    ]
    passage_embeddings = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    def embed_bound_texts(texts):
        return {
            text: np.asarray([0.0, 0.0, 1.0, 0.0])
            for text in texts
            if "Alice" in text
        }

    selected, trace = select_dtc_embed_positions(
        query="Where was the director of Film X born?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=[0, 1, 2],
        pool_doc_scores=np.asarray([1.0, 0.95, 0.2]),
        pool_doc_titles=["Film X", "Place of birth", "Alice"],
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"place of birth"}, 2: {"alice", "paris"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.1,
        dependency_bonus_weight=0.1,
        enable_dependency_binding=True,
        binding_entity_hit_required=True,
        repairable_filter_enabled=True,
        satisfiable_by_policy="strict",
        embed_texts_fn=embed_bound_texts,
    )

    assert selected == [0, 1]
    assert trace["repairable_by_requirement"]["s2"]["repairable"] is False
    assert trace["repairable_by_requirement"]["s2"]["raw_inference_veto"] is True
    assert trace["repairable_by_requirement"]["s2"]["inference_veto"] is True
    assert trace["repairable_by_requirement"]["s2"]["satisfiable_by_policy"] == "strict"


def test_repairable_filter_skips_inference_only_answer_requirement():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Which country is Film X from?",
            anchor_mentions=("Film X",),
            expected_answer_type="country",
            role="bridge",
            satisfiable_by="document",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Are the countries the same?",
            expected_answer_type="boolean",
            role="answer",
            satisfiable_by="inference",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0, 0.0]),
        "s2": np.asarray([0.0, 1.0, 0.0]),
    }
    pool_docs = [
        "Film X\nFilm X is from France.",
        "Baseline\nA high-ranked baseline page.",
        "Same country\nThis page talks about whether countries are the same.",
    ]
    pool_doc_ids = [0, 1, 2]
    passage_embeddings = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Are the countries from Film X and another film the same?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.9, 0.1]),
        pool_doc_titles=["Film X", "Baseline", "Same country"],
        doc_idx_to_entities={0: {"film x", "france"}, 1: {"baseline"}, 2: {"same country"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        repairable_filter_enabled=True,
    )

    assert selected == [0, 1]
    assert trace["repairable_by_requirement"]["s2"]["repairable"] is False
    assert trace["repairable_by_requirement"]["s2"]["reason"] == "inference_only_veto"
    assert trace["repairable_by_requirement"]["s2"]["satisfiable_by"] == "inference"


def test_repairable_filter_keeps_regex_fallback_without_satisfiable_by():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Which country is Film X from?",
            anchor_mentions=("Film X",),
            expected_answer_type="country",
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Are the countries the same?",
            expected_answer_type="boolean",
            role="answer",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0, 0.0]),
        "s2": np.asarray([0.0, 1.0, 0.0]),
    }
    pool_docs = [
        "Film X\nFilm X is from France.",
        "Baseline\nA high-ranked baseline page.",
        "Same country\nThis page talks about whether countries are the same.",
    ]
    pool_doc_ids = [0, 1, 2]
    passage_embeddings = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Are the countries from Film X and another film the same?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.9, 0.1]),
        pool_doc_titles=["Film X", "Baseline", "Same country"],
        doc_idx_to_entities={0: {"film x", "france"}, 1: {"baseline"}, 2: {"same country"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        repairable_filter_enabled=True,
    )

    assert selected == [0, 1]
    assert trace["repairable_by_requirement"]["s2"]["repairable"] is False
    assert trace["repairable_by_requirement"]["s2"]["reason"] == "inference_only_veto"
    assert trace["repairable_by_requirement"]["s2"]["satisfiable_by"] == "unknown"


def test_select_dtc_embed_positions_can_disable_dependency_ordering():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Where was that director born?",
            depends_on=("s1",),
            expected_answer_type="location",
            role="answer",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([0.0, 0.0, 1.0]),
        "s2": np.asarray([1.0, 0.0, 0.0]),
    }
    pool_docs = [
        "Film X\nA film page without useful director evidence.",
        "Alice\nAlice was born in Paris.",
        "Other\nAnother page.",
    ]
    pool_doc_ids = [0, 1, 2]
    passage_embeddings = np.asarray([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Where was the director of Film X born?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.1, 0.0]),
        pool_doc_titles=["Film X", "Alice", "Other"],
        doc_idx_to_entities={
            0: {"film x"},
            1: {"alice", "paris"},
            2: {"other"},
        },
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=0,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        enforce_dependencies=False,
    )

    assert selected[0] == 1
    assert trace["dependency_enforced"] is False
    assert trace["requirements"][1]["depends_on"] == []
    assert trace["cover_position_by_requirement"]["s2"] == 1


def test_select_dtc_embed_positions_can_require_new_threshold_crossing():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
    ]
    requirement_embeddings = {"s1": np.asarray([1.0, 0.0])}
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Baseline Fill\nA high-ranked page that should be preserved.",
        "Soft Gain\nThis page is slightly more similar but covers no new requirement.",
    ]
    pool_doc_ids = [0, 1, 2]
    pool_doc_scores = np.asarray([1.0, 0.9, 0.1])
    pool_doc_titles = ["Film X", "Baseline Fill", "Soft Gain"]
    passage_embeddings = np.asarray([
        [0.8, 0.2],
        [0.0, 1.0],
        [0.9, 0.1],
    ])

    soft_selected, soft_trace = select_dtc_embed_positions(
        query="Who directed Film X?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=pool_doc_scores,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"soft"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        require_new_crossing=False,
    )
    hard_selected, hard_trace = select_dtc_embed_positions(
        query="Who directed Film X?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=pool_doc_scores,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"soft"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        require_new_crossing=True,
    )

    assert soft_selected == [0, 2]
    assert soft_trace["selection_steps"][-1]["mode"] == "dtc_cover"
    assert soft_trace["selection_steps"][-1]["new_requirement_count"] == 0
    assert hard_selected == [0, 1]
    assert hard_trace["require_new_crossing"] is True
    assert hard_trace["selection_steps"][-1]["mode"] == "baseline_fill"
    assert hard_trace["selection_steps"][-1]["reason"] == "no_new_requirement_crossing"


def test_rank_weight_keeps_deep_soft_gain_from_replacing_baseline_fill():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        )
    ]
    requirement_embeddings = {"s1": np.asarray([1.0, 0.0])}
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Baseline Fill\nA high-ranked page that should be preserved.",
        "Deep Soft Gain\nThis page is slightly more similar but appears deeper.",
    ]
    pool_doc_ids = [0, 1, 2]
    pool_doc_scores = np.asarray([1.0, 0.9, 0.1])
    pool_doc_titles = ["Film X", "Baseline Fill", "Deep Soft Gain"]
    passage_embeddings = np.asarray([
        [0.8, 0.2],
        [0.0, 1.0],
        [0.9, 0.1],
    ])

    rank_selected, rank_trace = select_dtc_embed_positions(
        query="Who directed Film X?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=pool_doc_scores,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"soft"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        rank_weight=1.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        require_new_crossing=False,
    )

    assert rank_selected == [0, 1]
    assert rank_trace["rank_weight"] == 1.0
    assert rank_trace["selection_steps"][-1]["mode"] == "baseline_fill"
    assert rank_trace["selection_steps"][-1]["reason"] == "no_positive_requirement_gain"


def test_demand_gate_preserves_baseline_when_requirements_already_satisfied():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
    ]
    requirement_embeddings = {"s1": np.asarray([1.0, 0.0])}
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Baseline Fill\nA high-ranked page that should be preserved.",
        "Soft Gain\nThis page is slightly more similar but covers no new requirement.",
    ]
    pool_doc_ids = [0, 1, 2]
    pool_doc_scores = np.asarray([1.0, 0.9, 0.1])
    pool_doc_titles = ["Film X", "Baseline Fill", "Soft Gain"]
    passage_embeddings = np.asarray([
        [0.8, 0.2],
        [0.0, 1.0],
        [0.9, 0.1],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Who directed Film X?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=pool_doc_scores,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"soft"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        demand_gate_enabled=True,
        demand_gate_alpha=1.0,
    )

    assert selected == [0, 1]
    assert trace["status"] == "demand_gate_preserve"
    assert trace["demand_gate"]["preserve"] is True
    assert trace["baseline_demand_assessment"]["covered_requirement_rate"] == 1.0


def test_ser_repairs_missing_residual_requirement_with_one_swap():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Where was that director born?",
            expected_answer_type="location",
            role="answer",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Baseline Fill\nA high-ranked page that does not satisfy the missing demand.",
        "Alice\nAlice was born in Paris.",
    ]
    pool_doc_ids = [0, 1, 2]
    pool_doc_titles = ["Film X", "Baseline Fill", "Alice"]
    passage_embeddings = np.asarray([
        [1.0, 0.0],
        [0.1, 0.0],
        [0.0, 1.0],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Where was the director of Film X born?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.9, 0.1]),
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"alice", "paris"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        ser_enabled=True,
        ser_lambda0=0.2,
    )

    assert selected == [0, 2]
    assert trace["status"] == "ser_repair_applied"
    assert trace["ser"]["swap_count"] == 1
    assert trace["selection_steps"][0]["mode"] == "ser_swap"
    assert trace["selection_steps"][0]["add_position"] == 2


def test_ser_preserves_sufficient_baseline_when_repair_gain_is_not_enough():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
    ]
    requirement_embeddings = {"s1": np.asarray([1.0, 0.0])}
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Baseline Fill\nA high-ranked page that should be preserved.",
        "Soft Gain\nThis page is slightly more similar but covers no residual demand.",
    ]
    pool_doc_ids = [0, 1, 2]
    pool_doc_titles = ["Film X", "Baseline Fill", "Soft Gain"]
    passage_embeddings = np.asarray([
        [0.8, 0.2],
        [0.0, 1.0],
        [0.9, 0.1],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Who directed Film X?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.9, 0.1]),
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"soft"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        ser_enabled=True,
        ser_lambda0=2.0,
    )

    assert selected == [0, 1]
    assert trace["status"] == "ser_repair_preserve"
    assert trace["ser"]["swap_count"] == 0
    assert trace["ser"]["lambda_q"] > 0.0


def test_ser_anchor_binding_blocks_wrong_entity_distractor():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
    ]
    requirement_embeddings = {"s1": np.asarray([1.0, 0.0])}
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Baseline Fill\nA high-ranked page that should be preserved.",
        "Other Film\nOther Film was directed by Bob.",
    ]
    pool_doc_ids = [0, 1, 2]
    pool_doc_titles = ["Film X", "Baseline Fill", "Other Film"]
    passage_embeddings = np.asarray([
        [0.8, 0.2],
        [0.0, 1.0],
        [0.95, 0.05],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Who directed Film X?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.9, 0.1]),
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"other film", "bob"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        ser_enabled=True,
        ser_lambda0=0.1,
        ser_anchor_binding_enabled=True,
    )

    assert selected == [0, 1]
    assert trace["status"] == "ser_repair_preserve"
    assert trace["ser"]["anchor_binding_enabled"] is True


def test_repairable_residual_ignores_unbound_dependent_answer():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            expected_answer_type="person",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Where was that director born?",
            depends_on=("s1",),
            expected_answer_type="location",
            role="answer",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Baseline Fill\nA high-ranked page that should be preserved.",
        "Place of birth\nThis page is semantically close but has no resolved entity binding.",
    ]
    pool_doc_ids = [0, 1, 2]
    pool_doc_titles = ["Film X", "Baseline Fill", "Place of birth"]
    passage_embeddings = np.asarray([
        [1.0, 0.0],
        [0.1, 0.0],
        [0.0, 1.0],
    ])

    selected, trace = select_dtc_embed_positions(
        query="Where was the director of Film X born?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.9, 0.1]),
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"place of birth"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        ser_enabled=True,
        ser_lambda0=0.1,
        ser_repairable_residual_enabled=True,
    )

    assert selected == [0, 1]
    assert trace["status"] == "ser_repair_preserve"
    assert trace["ser"]["repairable_residual_enabled"] is True
    assert trace["ser"]["repairable_by_requirement"]["s2"]["repairable"] is False
    assert trace["ser"]["residual_by_requirement"]["s2"] == 0.0


def test_repairable_residual_uses_resolved_dependency_binding():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            expected_answer_type="person",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Where was that director born?",
            depends_on=("s1",),
            expected_answer_type="location",
            role="answer",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Baseline Fill\nA high-ranked page that should be preserved.",
        "Alice\nAlice was born in Paris.",
    ]
    pool_doc_ids = [0, 1, 2]
    pool_doc_titles = ["Film X", "Baseline Fill", "Alice"]
    passage_embeddings = np.asarray([
        [1.0, 0.0],
        [0.1, 0.0],
        [0.0, 1.0],
    ])

    def embed_bound_texts(texts):
        return {text: np.asarray([0.0, 1.0]) for text in texts}

    selected, trace = select_dtc_embed_positions(
        query="Where was the director of Film X born?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=np.asarray([1.0, 0.9, 0.1]),
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"baseline"}, 2: {"alice", "paris"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        reserve_top_m=1,
        match_threshold=0.35,
        redundancy_weight=0.0,
        base_weight=0.0,
        anchor_bonus_weight=0.0,
        dependency_bonus_weight=0.0,
        ser_enabled=True,
        ser_lambda0=0.1,
        ser_repairable_residual_enabled=True,
        enable_dependency_binding=True,
        binding_entity_hit_required=True,
        embed_texts_fn=embed_bound_texts,
    )

    assert selected == [0, 2]
    assert trace["status"] == "ser_repair_applied"
    assert trace["ser"]["repairable_by_requirement"]["s2"]["repairable"] is True
    assert trace["ser"]["repairable_by_requirement"]["s2"]["reason"] == "resolved_dependency_binding"
    assert trace["selection_steps"][0]["add_position"] == 2


def test_select_daec_noisyor_uses_frozen_binding_and_noisy_or_coverage():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            expected_answer_type="person",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Where was that director born?",
            depends_on=("s1",),
            expected_answer_type="location",
            role="bridge",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Birth A\nAlice was born in Paris.",
        "Birth B\nAlice grew up in Lyon.",
        "Alice\nEntity page with no useful birth evidence.",
    ]
    pool_doc_ids = [0, 1, 2, 3]
    pool_doc_titles = ["Film X", "Birth A", "Birth B", "Alice"]
    passage_embeddings = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    def embed_bound_texts(texts):
        return {text: np.asarray([0.0, 1.0]) for text in texts}

    selected, trace = select_daec_noisyor_positions(
        query="Where was the director of Film X born?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={
            0: {"film x", "alice"},
            1: {"paris"},
            2: {"lyon"},
            3: {"alice"},
        },
        passage_embeddings=passage_embeddings,
        qa_top_k=3,
        binding_top_m=2,
        embed_texts_fn=embed_bound_texts,
    )

    assert selected == [0, 1, 2]
    assert trace["selector"] == "daec_noisyor"
    assert trace["binding_selection_protocol"] == "best_binding_final_pool"
    assert "weighted_objective" not in trace
    assert trace["selected_binding"]["assignments"] == {"s2": "Alice"}
    assert trace["phi_shape"] == [2, 1, 4]
    assert trace["coverage_by_requirement"]["s1"] == 1.0
    assert trace["coverage_by_requirement"]["s2"] == 0.75
    assert trace["selection_steps"][2]["coverage_by_requirement"]["s2"] == 0.75
    assert all("weighted_objective" not in row for row in trace["binding_objectives"])


def test_select_daec_noisyor_excludes_operator_requirements_without_regex_policy():
    requirements = [
        DTCRequirement(
            unit_id="s1",
            subquery="Who directed Film X?",
            anchor_mentions=("Film X",),
            role="bridge",
        ),
        DTCRequirement(
            unit_id="s2",
            subquery="Which happened earlier?",
            role="comparison",
            satisfiable_by="inference",
        ),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "Film X\nFilm X was directed by Alice.",
        "Comparison\nThis document only matches comparison wording.",
    ]
    passage_embeddings = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    selected, trace = select_daec_noisyor_positions(
        query="Which happened earlier for the director of Film X?",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=[0, 1],
        pool_doc_titles=["Film X", "Comparison"],
        doc_idx_to_entities={0: {"film x", "alice"}, 1: {"comparison"}},
        passage_embeddings=passage_embeddings,
        qa_top_k=1,
        binding_top_m=2,
    )

    assert selected == [0]
    assert trace["requirement_count"] == 1
    assert trace["operator_requirement_count"] == 1
    assert [req["unit_id"] for req in trace["requirements"]] == ["s1"]


def test_select_daec_noisyor_caps_binding_product_without_new_hyperparameter():
    requirements = [
        DTCRequirement(unit_id="s1", subquery="Find root A."),
        DTCRequirement(unit_id="s2", subquery="Find root B."),
        DTCRequirement(unit_id="s3", subquery="Find root C."),
        DTCRequirement(unit_id="s4", subquery="Use that A entity.", depends_on=("s1",)),
        DTCRequirement(unit_id="s5", subquery="Use that B entity.", depends_on=("s2",)),
        DTCRequirement(unit_id="s6", subquery="Use that C entity.", depends_on=("s3",)),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "s2": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        "s3": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "s4": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        "s5": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        "s6": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    }
    a_titles = [f"Alpha{i}" for i in range(5)]
    b_titles = [f"Beta{i}" for i in range(5)]
    c_titles = [f"Gamma{i}" for i in range(5)]
    pool_doc_titles = ["Root A", "Root B", "Root C"] + a_titles + b_titles + c_titles
    pool_docs = [
        "Root A\n" + " ".join(a_titles),
        "Root B\n" + " ".join(b_titles),
        "Root C\n" + " ".join(c_titles),
    ] + [f"{title}\nEvidence for {title}." for title in a_titles + b_titles + c_titles]
    passage_embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
        + [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0] for _ in a_titles]
        + [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0] for _ in b_titles]
        + [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in c_titles],
        dtype=float,
    )

    _, trace = select_daec_noisyor_positions(
        query="Compose A, B, and C evidence.",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=list(range(len(pool_docs))),
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities={idx: {title.lower()} for idx, title in enumerate(pool_doc_titles)},
        passage_embeddings=passage_embeddings,
        qa_top_k=5,
        binding_top_m=5,
    )

    assert trace["binding_count_unpruned"] == 125
    assert trace["binding_count"] == 25
    assert trace["binding_max_bindings"] == 25
    assert trace["binding_pruned"] is True
    assert trace["selected_binding"]["prior"] == 0.04
    assert all(binding["prior"] == 0.04 for binding in trace["bindings"])


def test_select_daec_noisyor_binding_candidate_tiebreak_ignores_dep_score():
    requirements = [
        DTCRequirement(unit_id="s1", subquery="Find the bridge entity."),
        DTCRequirement(unit_id="s2", subquery="Use that entity.", depends_on=("s1",)),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "High support\npadding padding ZuluName",
        "Lower support\nAlphaName appears first",
        "ZuluName\nCandidate page.",
        "AlphaName\nCandidate page.",
    ]
    passage_embeddings = np.asarray([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    _, trace = select_daec_noisyor_positions(
        query="Use the bridge entity.",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_titles=["High support", "Lower support", "ZuluName", "AlphaName"],
        doc_idx_to_entities={idx: set() for idx in range(4)},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        binding_top_m=2,
    )

    candidates = trace["binding_candidates_by_requirement"]["s2"]
    assert candidates[0]["title"] == "AlphaName"
    assert candidates[0]["dep_score"] < candidates[1]["dep_score"]


def test_select_daec_noisyor_safe_keeps_baseline_when_gain_is_small():
    requirements = [
        DTCRequirement(unit_id="s1", subquery="Find first evidence."),
        DTCRequirement(unit_id="s2", subquery="Find second evidence."),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "Baseline A\nNearly complete first evidence.",
        "Baseline B\nNearly complete second evidence.",
        "Perfect A\nComplete first evidence.",
        "Perfect B\nComplete second evidence.",
    ]
    passage_embeddings = np.asarray([
        [0.95, 0.05],
        [0.05, 0.95],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    selected, trace = select_daec_noisyor_positions(
        query="Find both pieces of evidence.",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_titles=["Baseline A", "Baseline B", "Perfect A", "Perfect B"],
        pool_doc_scores=np.asarray([1.0, 0.9, 0.2, 0.1]),
        doc_idx_to_entities={idx: set() for idx in range(4)},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        safe_projection=True,
        safe_min_objective_gain=0.2,
        safe_min_swap_gain=0.01,
        safe_max_swaps=2,
    )

    assert selected == [0, 1]
    assert trace["selector"] == "daec_noisyor_safe"
    assert trace["safe_projection_trace"]["safe_decision"] == "fallback_low_rebuild_gain"
    assert trace["safe_projection_trace"]["rebuild_gain_over_baseline"] < 0.2


def test_select_daec_noisyor_safe_applies_minimal_swap():
    requirements = [
        DTCRequirement(unit_id="s1", subquery="Find first evidence."),
        DTCRequirement(unit_id="s2", subquery="Find second evidence."),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "Anchor\nStrong first evidence.",
        "Weak baseline\nWeak second evidence.",
        "Second support\nStrong second evidence.",
    ]
    passage_embeddings = np.asarray([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])

    selected, trace = select_daec_noisyor_positions(
        query="Find both pieces of evidence.",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=[0, 1, 2],
        pool_doc_titles=["Anchor", "Weak baseline", "Second support"],
        pool_doc_scores=np.asarray([1.0, 0.9, 0.2]),
        doc_idx_to_entities={idx: set() for idx in range(3)},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        safe_projection=True,
        safe_min_objective_gain=0.05,
        safe_min_swap_gain=0.01,
        safe_max_swaps=1,
        safe_preserve_top_m=1,
    )

    assert selected == [0, 2]
    assert trace["safe_projection_trace"]["safe_decision"] == "minimal_edit_applied"
    assert trace["selection_steps"][0]["out_position"] == 1
    assert trace["selection_steps"][0]["in_position"] == 2


def test_select_daec_noisyor_safe_retriever_margin_fallback():
    requirements = [
        DTCRequirement(unit_id="s1", subquery="Find first evidence."),
        DTCRequirement(unit_id="s2", subquery="Find second evidence."),
    ]
    requirement_embeddings = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.0, 1.0]),
    }
    pool_docs = [
        "Confident top\nStrong first evidence.",
        "Weak baseline\nWeak second evidence.",
        "Second support\nStrong second evidence.",
    ]
    passage_embeddings = np.asarray([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])

    selected, trace = select_daec_noisyor_positions(
        query="Find both pieces of evidence.",
        requirements=requirements,
        requirement_embeddings=requirement_embeddings,
        pool_docs=pool_docs,
        pool_doc_ids=[0, 1, 2],
        pool_doc_titles=["Confident top", "Weak baseline", "Second support"],
        pool_doc_scores=np.asarray([10.0, 1.0, 0.0]),
        doc_idx_to_entities={idx: set() for idx in range(3)},
        passage_embeddings=passage_embeddings,
        qa_top_k=2,
        safe_projection=True,
        safe_min_objective_gain=0.05,
        safe_min_swap_gain=0.01,
        safe_max_swaps=1,
        safe_retriever_margin_threshold=0.5,
    )

    assert selected == [0, 1]
    assert trace["safe_projection_trace"]["safe_decision"] == "fallback_retriever_margin"
