from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_chain_walking_anchor_variants import (  # noqa: E402
    add_current_delta_counts,
    content_tokens,
    demand_entity_title_score,
    entity_rows_by_query_from_outputs,
    rank_pool_by_policy,
    title_only_score,
    title_strict_score,
)


def test_title_strict_does_not_use_body_matches() -> None:
    score, reason = title_strict_score("Guadalajara", "Paula Santiago")

    assert score == 0.0
    assert reason == ""


def test_title_only_allows_title_token_overlap() -> None:
    score, reason = title_only_score("Warner Bros. Record", "Warner Records")

    assert score > 0
    assert reason == "title_token_overlap"


def test_demand_entity_title_promotes_specific_topical_page_over_entity_page() -> None:
    entity_row = {
        "entity": "New Zealand",
        "requirement_subquery": "What is the capital punishment policy in that country?",
    }
    broad_score, broad_reason = demand_entity_title_score(entity_row, "New Zealand")
    specific_score, specific_reason = demand_entity_title_score(entity_row, "Capital punishment in New Zealand")

    assert specific_score > broad_score
    assert specific_reason == "demand_entity_title_phrase"
    assert broad_reason == "entity_title_phrase"


def test_rank_pool_by_policy_demand_entity_title_reorders_broad_entity_case() -> None:
    titles = [
        "New Zealand",
        "Geography of New Zealand",
        "Capital punishment in New Zealand",
    ]
    docs = [f"{title}\ntext" for title in titles]
    entity_rows = [
        {
            "entity": "New Zealand",
            "requirement_subquery": "What is the capital punishment policy in that country?",
        }
    ]

    strict_order, _strict_best = rank_pool_by_policy(
        policy="title_strict",
        pool_titles=titles,
        pool_docs=docs,
        entity_rows=entity_rows,
    )
    demand_order, demand_best = rank_pool_by_policy(
        policy="demand_entity_title",
        pool_titles=titles,
        pool_docs=docs,
        entity_rows=entity_rows,
    )

    assert strict_order[0] == 0
    assert demand_order[0] == 2
    assert demand_best[2]["reason"] == "demand_entity_title_phrase"


def test_entity_rows_by_query_from_outputs_parses_and_deduplicates() -> None:
    rows = [
        {
            "query_index": 7,
            "requirement_id": "s2",
            "requirement_subquery": "Where was the author born?",
            "upstream_position": 3,
            "upstream_title": "Author",
            "entities_json": json.dumps(
                [
                    {"entity": "New Zealand", "support_span": "born in New Zealand"},
                    {"entity": "New Zealand", "support_span": "duplicate"},
                ]
            ),
        }
    ]

    by_query = entity_rows_by_query_from_outputs(rows)

    assert list(by_query) == [7]
    assert len(by_query[7]) == 1
    assert by_query[7][0]["entity"] == "New Zealand"
    assert by_query[7][0]["requirement_id"] == "s2"


def test_content_tokens_keeps_topical_words_and_drops_generic_words() -> None:
    assert content_tokens("What is the capital punishment policy in that country?") == {
        "capital",
        "punishment",
        "policy",
    }


def test_add_current_delta_counts_tracks_unique_gain_and_loss() -> None:
    comparison = [{"policy": "current_all"}, {"policy": "title_only"}]
    rows = [
        {"query_index": 1, "gold_title": "A", "policy": "current_all", "new_hit_at_5": 1},
        {"query_index": 1, "gold_title": "A", "policy": "title_only", "new_hit_at_5": 0},
        {"query_index": 2, "gold_title": "B", "policy": "current_all", "new_hit_at_5": 0},
        {"query_index": 2, "gold_title": "B", "policy": "title_only", "new_hit_at_5": 1},
    ]

    output = add_current_delta_counts(comparison, rows)

    assert output[0]["unique_new5_vs_current"] == 0
    assert output[0]["lost_new5_vs_current"] == 0
    assert output[1]["unique_new5_vs_current"] == 1
    assert output[1]["lost_new5_vs_current"] == 1
