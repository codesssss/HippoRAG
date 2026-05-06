from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_llm_direct_pool_selector import (  # noqa: E402
    complete_selection,
    parse_direct_selection,
    reorder_record,
)


def test_parse_direct_selection_prefers_json_schema_and_deduplicates():
    positions, method = parse_direct_selection('{"selected_ids": [3, 1, 3, 5]}', max_position=10)

    assert method == "json"
    assert positions == [2, 0, 4]


def test_parse_direct_selection_accepts_final_selection_brackets():
    positions, method = parse_direct_selection("### Final Selection: [4] [2] [99]", max_position=5)

    assert method == "bracket"
    assert positions == [3, 1]


def test_parse_direct_selection_accepts_zero_based_json_when_zero_is_present():
    positions, method = parse_direct_selection('{"selected_ids": [0, 2, 4]}', max_position=5)

    assert method == "json"
    assert positions == [0, 2, 4]


def test_complete_selection_fills_tail_from_rank_order():
    final_positions, fallback_positions = complete_selection(
        [4, 1],
        pool_size=6,
        selection_count=5,
    )

    assert final_positions == [4, 1, 0, 2, 3]
    assert fallback_positions == [0, 2, 3]


def test_reorder_record_places_llm_prefix_before_original_tail():
    record = {
        "query_idx": 0,
        "question": "q",
        "pool_docs": [f"T{i}\nBody {i}" for i in range(6)],
        "pool_titles": [f"T{i}" for i in range(6)],
        "pool_doc_scores": [float(10 - i) for i in range(6)],
        "pool_doc_ids": list(range(100, 106)),
    }
    selection_row = {
        "selected_positions": [3, 1],
        "selected_1based": [4, 2],
        "parse_success": True,
        "parse_method": "json",
        "raw_output": '{"selected_ids":[4,2]}',
        "model": "qwen3-8b-train",
    }

    output = reorder_record(
        record,
        selection_row,
        pool_k=6,
        selection_count=5,
        context_mode="title",
        snippet_chars=0,
    )

    assert output["pool_titles"][:6] == ["T3", "T1", "T0", "T2", "T4", "T5"]
    assert output["pool_doc_ids"][:5] == [103, 101, 100, 102, 104]
    trace = output["llm_direct_select"]
    assert trace["final_prefix_positions"] == [3, 1, 0, 2, 4]
    assert trace["fallback_positions"] == [0, 2, 4]
