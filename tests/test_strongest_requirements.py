from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag_ext.strongest.gbc import (
    _build_local_requirement_support_tensor,
    apply_prefix_guard,
    compute_head_requirement_coverage,
    compute_unmet_requirement_gain,
    compute_requirement_support,
    detect_requirement_conflicts,
)
from src.hipporag_ext.strongest.requirements import (
    _parsed_to_requirement_units,
    extract_requirement_units,
    extract_requirement_units_llm_closed_grounded,
    extract_requirement_units_llm_grounded,
    fillers_equivalent,
)
from src.hipporag_ext.strongest.types import RequirementUnit


def test_extract_requirement_units_builds_support_and_core_for_nested_query():
    requirement_units, extractor_trace = extract_requirement_units(
        query="What is the birthplace of the director of Film X?",
        seed_entities=["Film X"],
        query_entities=["Film X"],
        baseline_titles=["Film X"],
        max_units=4,
    )

    assert extractor_trace["extractor_fallback"] is False
    assert extractor_trace["core_unit_count"] == 1
    assert extractor_trace["support_unit_count"] == 1
    assert [unit.slot_family for unit in requirement_units] == ["director", "birthplace"]
    assert [unit.tier for unit in requirement_units] == ["support", "core"]
    assert requirement_units[1].bridge_targets == [requirement_units[0].unit_id]


def test_extract_requirement_units_falls_back_when_no_slot_family_detected():
    requirement_units, extractor_trace = extract_requirement_units(
        query="Tell me about Film X",
        seed_entities=["Film X"],
        query_entities=["Film X"],
        baseline_titles=["Film X"],
        max_units=4,
    )

    assert requirement_units == []
    assert extractor_trace["extractor_fallback"] is True
    assert extractor_trace["extractor_fallback_reason"] == "no_slot_family"
    assert extractor_trace["empty_requirement_query"] is True


def test_fillers_equivalent_handles_normalized_and_substring_matches():
    assert fillers_equivalent("John Doe", "john doe") is True
    assert fillers_equivalent("University of Oxford", "Oxford") is True
    assert fillers_equivalent("Paris", "London") is False


def test_compute_requirement_support_requires_anchor_slot_and_role_filler():
    requirement_units = [
        RequirementUnit(
            unit_id="core-director-alpha",
            tier="core",
            anchor_entities=["alpha"],
            slot_family="director",
            expected_answer_type="person",
            is_single_valued=True,
        )
    ]
    support_tensor = np.asarray(
        [
            [[1.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    support_scores = compute_requirement_support(
        support_tensor=support_tensor,
        requirement_units=requirement_units,
        strict_eligibility=True,
    )

    assert support_scores[:, 0].tolist() == [0.0, 0.0, 1.0]


def test_compute_requirement_support_treats_extended_role_slots_like_role_relations():
    requirement_units = [
        RequirementUnit(
            unit_id="support-performer-song-x",
            tier="support",
            anchor_entities=["song x"],
            slot_family="performer",
            expected_answer_type="person",
            is_single_valued=True,
        )
    ]
    support_tensor = np.asarray(
        [
            [[1.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    support_scores = compute_requirement_support(
        support_tensor=support_tensor,
        requirement_units=requirement_units,
        strict_eligibility=True,
    )

    assert support_scores[:, 0].tolist() == [0.0, 1.0]


def test_embedding_probe_can_semantically_activate_support_unit(monkeypatch):
    requirement_units = [
        RequirementUnit(
            unit_id="support-performer-changed-it",
            tier="support",
            anchor_entities=["changed it"],
            slot_family="performer",
            expected_answer_type="person",
            is_single_valued=True,
        )
    ]
    metadata = {
        "query_text": "What is the place of birth of the performer of Changed It?",
        "ras_support_mode": "embedding_probe",
        "ras_embedding_probe_threshold": 0.35,
        "passage_titles": ["Changed It"],
        "passage_texts": ["Changed It\nThis song was recorded by Sam Cooke."],
        "passage_structure_entities": [["changed it", "sam cooke"]],
        "passage_chunk_ids": [None],
        "chunk_triples_map": {},
        "_ras_embedding_model": object(),
        "_ras_passage_embeddings": np.asarray([[1.0, 0.0]], dtype=np.float32),
    }

    def fake_encode(_embedding_model, texts):
        if texts and str(texts[0]).startswith("Question:"):
            return np.asarray([[0.9, 0.1]], dtype=np.float32)
        return np.asarray([[0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        "src.hipporag_ext.strongest.gbc._encode_probe_text_embeddings",
        fake_encode,
    )

    support_tensor, filler_values = _build_local_requirement_support_tensor(
        candidate_indices=np.asarray([0], dtype=np.int64),
        selected_local_indices=np.asarray([0], dtype=np.int64),
        requirement_units=requirement_units,
        metadata=metadata,
    )
    support_scores = compute_requirement_support(
        support_tensor=support_tensor,
        requirement_units=requirement_units,
        strict_eligibility=True,
    )

    assert float(support_tensor[0, 0, 0]) == 1.0
    assert float(support_tensor[0, 0, 1]) >= 0.35
    assert float(support_tensor[0, 0, 2]) == 1.0
    assert filler_values[0][0] == "sam cooke"
    assert float(support_scores[0, 0]) >= 0.35


def test_embedding_probe_uses_title_as_bridge_filler_when_semantic_slot_matches(monkeypatch):
    requirement_units = [
        RequirementUnit(
            unit_id="core-birthplace-performer",
            tier="core",
            anchor_entities=[],
            slot_family="birthplace",
            expected_answer_type="location",
            bridge_targets=["support-performer-changed-it"],
            is_single_valued=True,
        )
    ]
    metadata = {
        "query_text": "What is the place of birth of the performer of Changed It?",
        "ras_support_mode": "embedding_probe",
        "ras_embedding_probe_threshold": 0.35,
        "passage_titles": ["Sam Cooke"],
        "passage_texts": ["Sam Cooke\nA singer born in Mississippi."],
        "passage_structure_entities": [["sam cooke"]],
        "passage_chunk_ids": [None],
        "chunk_triples_map": {},
        "_ras_embedding_model": object(),
        "_ras_passage_embeddings": np.asarray([[1.0, 0.0]], dtype=np.float32),
    }

    def fake_encode(_embedding_model, texts):
        if texts and str(texts[0]).startswith("Question:"):
            return np.asarray([[0.8, 0.2]], dtype=np.float32)
        return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(
        "src.hipporag_ext.strongest.gbc._encode_probe_text_embeddings",
        fake_encode,
    )

    support_tensor, filler_values = _build_local_requirement_support_tensor(
        candidate_indices=np.asarray([0], dtype=np.int64),
        selected_local_indices=np.asarray([0], dtype=np.int64),
        requirement_units=requirement_units,
        metadata=metadata,
    )
    support_scores = compute_requirement_support(
        support_tensor=support_tensor,
        requirement_units=requirement_units,
        strict_eligibility=True,
    )

    assert float(support_tensor[0, 0, 0]) == 1.0
    assert float(support_tensor[0, 0, 1]) >= 0.35
    assert float(support_tensor[0, 0, 2]) == 1.0
    assert filler_values[0][0] == "sam cooke"
    assert float(support_scores[0, 0]) >= 0.35


def test_parsed_units_skip_unresolved_lexical_anchor():
    units, stats = _parsed_to_requirement_units(
        parsed=[
            {
                "tier": "support",
                "anchor": "Unknown Film Alias",
                "slot_family": "director",
                "expected_answer_type": "person",
                "is_single_valued": True,
            }
        ],
        seed_entities=["film x"],
        query_entities=["film x"],
        baseline_titles=["film x"],
        max_units=4,
    )

    assert units == []
    assert stats["inactive_unit_count"] == 1
    assert stats["resolved_anchor_count"] == 0
    assert stats["inactive_units"][0]["reason"] == "unresolved_anchor"


def test_parsed_units_resolve_bridge_ref_against_raw_support_slot():
    units, stats = _parsed_to_requirement_units(
        parsed=[
            {
                "tier": "support",
                "anchor": "Lothair II",
                "slot_family": "mother",
                "expected_answer_type": "person",
                "is_single_valued": True,
            },
            {
                "tier": "core",
                "anchor": "<mother>",
                "slot_family": "death_date",
                "expected_answer_type": "date",
                "is_single_valued": True,
            },
        ],
        seed_entities=["lothair ii"],
        query_entities=["lothair ii"],
        baseline_titles=["lothair ii"],
        max_units=4,
    )

    assert len(units) == 2
    assert units[0].slot_family == "parent"
    assert units[0].raw_slot_family == "mother"
    assert units[1].anchor_status == "bridge_ref"
    assert units[1].bridge_ref_label == "mother"
    assert units[1].bridge_targets == [units[0].unit_id]
    assert stats["bridge_ref_resolved_count"] == 1


def test_parsed_units_do_not_activate_unresolved_bridge_ref():
    units, stats = _parsed_to_requirement_units(
        parsed=[
            {
                "tier": "core",
                "anchor": "<director>",
                "slot_family": "birthplace",
                "expected_answer_type": "location",
                "is_single_valued": True,
            }
        ],
        seed_entities=["film x"],
        query_entities=["film x"],
        baseline_titles=["film x"],
        max_units=4,
    )

    assert units == []
    assert stats["bridge_ref_candidate_count"] == 1
    assert stats["bridge_ref_resolved_count"] == 0
    assert stats["inactive_units"][0]["reason"] == "unresolved_bridge_target"


def test_parsed_units_record_unsupported_closed_ontology_slot():
    units, stats = _parsed_to_requirement_units(
        parsed=[
            {
                "tier": "support",
                "anchor": "The Distribution of Industry act",
                "slot_family": "unsupported",
                "raw_slot_text": "passed_by",
                "expected_answer_type": "person",
                "is_single_valued": True,
            }
        ],
        seed_entities=["the distribution of industry act"],
        query_entities=["the distribution of industry act"],
        baseline_titles=["the distribution of industry act"],
        max_units=4,
    )

    assert units == []
    assert stats["unsupported_slot_count"] == 1
    assert stats["unsupported_raw_slots"] == ["passed by"]
    assert stats["inactive_units"][0]["reason"] == "unsupported_slot"


def test_llm_grounded_returns_empty_without_rule_fallback_when_grounding_fails(monkeypatch):
    monkeypatch.setattr(
        "src.hipporag_ext.strongest.requirements._request_requirement_units_from_llm",
        lambda _query, system_prompt=None: (
            [
                {
                    "tier": "support",
                    "anchor": "Unknown Film Alias",
                    "slot_family": "director",
                    "expected_answer_type": "person",
                    "is_single_valued": True,
                }
            ],
            '[{"tier":"support","anchor":"Unknown Film Alias","slot_family":"director"}]',
            None,
        ),
    )

    units, trace = extract_requirement_units_llm_grounded(
        query="Who directed Film X?",
        seed_entities=["film x"],
        query_entities=["film x"],
        baseline_titles=["film x"],
        max_units=4,
    )

    assert units == []
    assert trace["extractor_mode"] == "llm_grounded"
    assert trace["extractor_status"] == "empty_after_grounding"
    assert trace["extractor_fallback"] is False
    assert trace["resolved_anchor_count"] == 0
    assert trace["inactive_unit_count"] == 1


def test_llm_closed_grounded_surfaces_unsupported_schema_in_trace(monkeypatch):
    monkeypatch.setattr(
        "src.hipporag_ext.strongest.requirements._request_requirement_units_from_llm",
        lambda _query, system_prompt=None: (
            [
                {
                    "tier": "support",
                    "anchor": "The Distribution of Industry act",
                    "slot_family": "unsupported",
                    "raw_slot_text": "passed_by",
                    "expected_answer_type": "person",
                    "is_single_valued": True,
                }
            ],
            '[{"tier":"support","anchor":"The Distribution of Industry act","slot_family":"unsupported","raw_slot_text":"passed_by"}]',
            None,
        ),
    )

    units, trace = extract_requirement_units_llm_closed_grounded(
        query="The Distribution of Industry act was passed by a man who was prime minister when?",
        seed_entities=["the distribution of industry act"],
        query_entities=["the distribution of industry act"],
        baseline_titles=["the distribution of industry act"],
        max_units=4,
    )

    assert units == []
    assert trace["extractor_mode"] == "llm_closed_grounded"
    assert trace["unsupported_slot_count"] == 1
    assert trace["unsupported_raw_slots"] == ["passed by"]
    assert "unsupported" in trace["closed_ontology_slot_families"]


def test_compute_unmet_requirement_gain_requires_bridge_binding_to_support_filler():
    requirement_units = [
        RequirementUnit(
            unit_id="support-director-film-x",
            tier="support",
            anchor_entities=["film x"],
            slot_family="director",
            expected_answer_type="person",
            is_single_valued=True,
        ),
        RequirementUnit(
            unit_id="core-birthplace-director",
            tier="core",
            anchor_entities=[],
            slot_family="birthplace",
            expected_answer_type="location",
            bridge_targets=["support-director-film-x"],
            is_single_valued=True,
        ),
    ]
    support_scores = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    filler_values = [
        ["sam cooke", ""],
        ["", "sam cooke"],
        ["", "juan carlos salazar"],
    ]

    g_core, g_support, unmet_mass = compute_unmet_requirement_gain(
        requirement_units=requirement_units,
        support_scores=support_scores,
        head_coverage=np.asarray([1.0, 0.0], dtype=np.float32),
        frontier_local_indices=np.asarray([1, 2], dtype=np.int64),
        filler_values=filler_values,
        head_local_indices=np.asarray([0], dtype=np.int64),
    )

    assert g_core.tolist() == [0.0, 1.0, 0.0]
    assert g_support.tolist() == [0.0, 0.0, 0.0]
    assert unmet_mass.tolist() == [0.0, 1.0]


def test_compute_head_requirement_coverage_requires_bridge_binding_for_core_units():
    requirement_units = [
        RequirementUnit(
            unit_id="support-mother-lothair",
            tier="support",
            anchor_entities=["lothair ii"],
            slot_family="parent",
            expected_answer_type="person",
            is_single_valued=True,
        ),
        RequirementUnit(
            unit_id="core-death-date-mother",
            tier="core",
            anchor_entities=[],
            slot_family="death_date",
            expected_answer_type="date",
            bridge_targets=["support-mother-lothair"],
            is_single_valued=True,
        ),
    ]
    support_scores = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    filler_values = [
        ["", "theobald of arles"],
        ["ermengarde of tours", ""],
        ["", "ermengarde of tours"],
    ]

    head_cov, final_cov, traces = compute_head_requirement_coverage(
        requirement_units=requirement_units,
        support_scores=support_scores,
        core_local_indices=np.asarray([0, 1], dtype=np.int64),
        final_local_indices=np.asarray([1, 2], dtype=np.int64),
        filler_values=filler_values,
    )

    assert head_cov.tolist() == [1.0, 0.0]
    assert final_cov.tolist() == [1.0, 1.0]
    assert traces[1].head_coverage == 0.0
    assert traces[1].final_coverage == 1.0


def test_detect_requirement_conflicts_flags_single_value_filler_mismatch():
    requirement_units = [
        RequirementUnit(
            unit_id="core-spouse-alpha",
            tier="core",
            anchor_entities=["alpha"],
            slot_family="spouse",
            expected_answer_type="person",
            is_single_valued=True,
        )
    ]
    support_scores = np.asarray(
        [
            [1.0],
            [1.0],
            [1.0],
        ],
        dtype=np.float32,
    )
    filler_values = [
        ["beta"],
        ["beta"],
        ["gamma"],
    ]

    conflict_map = detect_requirement_conflicts(
        requirement_units=requirement_units,
        support_scores=support_scores,
        filler_values=filler_values,
        core_local_indices=np.asarray([0, 1], dtype=np.int64),
        frontier_local_indices=np.asarray([2], dtype=np.int64),
    )

    assert list(conflict_map) == [2]
    assert conflict_map[2][0].head_filler == "beta"
    assert conflict_map[2][0].candidate_filler == "gamma"


def test_apply_prefix_guard_preserves_prefix_without_positive_core_gain():
    g_core_scores = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    guarded_order, triggered, prefix_before, prefix_after, prefix_disruption = apply_prefix_guard(
        ranked_local_indices=[3, 0, 1, 2],
        baseline_prefix_local_indices=[0, 1],
        g_core_scores=g_core_scores,
        prefix_guard_k=2,
    )

    assert triggered is True
    assert guarded_order[:2] == [0, 1]
    assert prefix_before == [0, 1]
    assert prefix_after == [0, 1]
    assert prefix_disruption[3] == 1.0


def test_apply_prefix_guard_allows_positive_core_gain_intruder_into_prefix():
    g_core_scores = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    guarded_order, triggered, prefix_before, prefix_after, _prefix_disruption = apply_prefix_guard(
        ranked_local_indices=[2, 0, 1, 3],
        baseline_prefix_local_indices=[0, 1],
        g_core_scores=g_core_scores,
        prefix_guard_k=2,
    )

    assert triggered is False
    assert guarded_order[:3] == [2, 0, 1]
    assert prefix_before == [0, 1]
    assert prefix_after == [2, 0]
