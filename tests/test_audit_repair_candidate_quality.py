from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from audit_repair_candidate_quality import (  # noqa: E402
    binding_candidate_entries,
    classify_missing_gold,
    near_title_substitution,
    title_positions,
)


def test_classify_missing_gold_distinguishes_candidate_generation_and_scoring() -> None:
    assert classify_missing_gold(
        source_has=False,
        rank_has=False,
        binding_has=False,
        greedy_has=False,
        dbec_has=False,
    ) == "gold_absent_from_source_pool"
    assert classify_missing_gold(
        source_has=True,
        rank_has=True,
        binding_has=False,
        greedy_has=False,
        dbec_has=False,
    ) == "gold_in_rank_fill_not_dbec"
    assert classify_missing_gold(
        source_has=True,
        rank_has=False,
        binding_has=True,
        greedy_has=False,
        dbec_has=False,
    ) == "gold_in_binding_candidates_not_selected"
    assert classify_missing_gold(
        source_has=True,
        rank_has=True,
        binding_has=True,
        greedy_has=True,
        dbec_has=True,
    ) == "gold_recovered_by_dbec_final"


def test_near_title_substitution_catches_branch_like_title_overlap() -> None:
    assert near_title_substitution(
        "Capital punishment in the United States",
        "Capital punishment in New Zealand",
    )
    assert not near_title_substitution("Warsaw Pact", "Szlachta")


def test_binding_candidate_entries_flatten_requirements_and_positions() -> None:
    trace = {
        "binding_candidates_by_requirement": {
            "s2": [
                {
                    "title": "United States Navy",
                    "title_pool_position": 7,
                    "dep": "s1",
                    "dep_position": 1,
                    "dep_score": 0.9,
                    "llm_extracted_entity": "US Navy",
                    "entity_match_type": "alias",
                }
            ],
            "s3": [],
        }
    }

    assert binding_candidate_entries(trace) == [
        {
            "requirement_id": "s2",
            "title": "United States Navy",
            "title_pool_position": 7,
            "dep": "s1",
            "dep_position": 1,
            "dep_score": 0.9,
            "llm_extracted_entity": "US Navy",
            "entity_match_type": "alias",
        }
    ]


def test_title_positions_tracks_duplicate_titles_after_normalization() -> None:
    positions = title_positions(["Warsaw Pact", "Warsaw Pact (album)", "Szlachta"])

    assert positions["warsaw pact"] == [0, 1]
    assert positions["szlachta"] == [2]
