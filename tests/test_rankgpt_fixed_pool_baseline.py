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
    build_messages,
    extract_positions_from_raw,
    fill_source_order,
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


def test_support_metrics_handles_duplicate_gold_titles() -> None:
    record = {
        "gold_titles": ["A", "A", "B"],
        "pool_titles": ["A", "B", "A"],
    }

    metrics = support_metrics_for_order([0, 1], record, top_k=2)

    assert metrics["hit_count"] == 2
    assert metrics["missing_count"] == 1
    assert metrics["complete"] == 0
