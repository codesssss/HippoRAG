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
