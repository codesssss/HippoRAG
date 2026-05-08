from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_rankgpt_fixed_pool_baseline import (  # noqa: E402
    apply_window_permutation,
    build_messages,
    build_sliding_window_messages,
    extract_positions_from_raw,
    fill_source_order,
    sliding_window_ranges,
    support_metrics_for_order,
)
from probe_fixed_pool_candidate_generation import prompt_has_no_think  # noqa: E402


def test_rankgpt_prompts_use_no_think() -> None:
    record = {
        "pool_titles": ["A", "B"],
        "pool_docs": ["A text", "B text"],
    }

    for variant in ("rank5_no_think", "select5_no_think"):
        messages = build_messages(
            variant=variant,
            question="Which document answers the question?",
            record=record,
            max_doc_chars=20,
        )
        assert prompt_has_no_think(messages)
        assert "/no_think" in messages[1]["content"]


def test_sliding_window_prompt_ranks_full_window_with_no_think() -> None:
    messages = build_sliding_window_messages(
        question="Which passages matter?",
        window_titles=["A", "B", "C"],
        window_docs=["A text", "B text", "C text"],
        max_doc_chars=20,
    )

    assert prompt_has_no_think(messages)
    assert "Rank all passages in this window" in messages[1]["content"]
    assert "1 to 3" in messages[1]["content"]


def test_extract_positions_from_rank_json_dedups_and_bounds() -> None:
    raw = '{"ranking": [3, 1, 3, 99, {"doc_id": 2}]}'

    positions, source = extract_positions_from_raw(raw, pool_size=5, top_k=5, one_based_ids=False)

    assert positions == [3, 1, 2]
    assert source == "json"


def test_extract_positions_defaults_to_one_based_rankgpt_ids() -> None:
    raw = '{"ranking": [3, 1, 3, 99, {"doc_id": 2}]}'

    positions, source = extract_positions_from_raw(raw, pool_size=5, top_k=5)

    assert positions == [2, 0, 1]
    assert source == "json"


def test_fill_source_order_keeps_prefix_then_rank_order() -> None:
    assert fill_source_order([2, 0, 2], pool_size=5) == [2, 0, 1, 3, 4]


def test_sliding_window_ranges_match_rankgpt_back_to_front_order() -> None:
    assert sliding_window_ranges(
        pool_size=100,
        window_size=20,
        step=10,
        rank_start=0,
        rank_end=100,
    ) == [
        (80, 100),
        (70, 90),
        (60, 80),
        (50, 70),
        (40, 60),
        (30, 50),
        (20, 40),
        (10, 30),
        (0, 20),
    ]


def test_apply_window_permutation_updates_only_window() -> None:
    current = list(range(10))
    updated = apply_window_permutation(current, start=3, end=7, local_positions=[2, 0])

    assert updated == [0, 1, 2, 5, 3, 4, 6, 7, 8, 9]


def test_support_metrics_handles_duplicate_gold_titles() -> None:
    record = {
        "gold_titles": ["A", "A", "B"],
        "pool_titles": ["A", "B", "A"],
    }

    metrics = support_metrics_for_order([0, 1], record, top_k=2)

    assert metrics["hit_count"] == 2
    assert metrics["missing_count"] == 1
    assert metrics["complete"] == 0
