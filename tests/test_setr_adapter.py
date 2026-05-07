from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_pool_support_depth import analyze_record, coverage_by_query
from apply_setr_selection_to_pool import apply_selection_to_record, rank_order_fill_to_k_positions, reorder_positions
from export_setr_inputs import build_setr_record
from run_setr_style_selector import call_selector, parse_setr_selection
from run_setr_windowed_selector import run_windowed_record, stage1_windows


def test_parse_setr_selection_prefers_final_selection_and_filters_invalid_ids():
    text = "Step 1: requirement [9]\n### Final Selection: [3] [1] [3] [999] [2]"

    assert parse_setr_selection(text, max_position=4) == [2, 0, 1]


def test_call_selector_records_usage_latency_and_prompt_size():
    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "fake-model"
            assert kwargs["temperature"] == 0.0
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="### Final Selection: [2] [1]"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=123, completion_tokens=7, total_tokens=130),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    record = {
        "query_idx": 4,
        "question": "Where was Alice born?",
        "pool_k": 2,
        "contexts": [
            {"position": 1, "title": "Alice", "text": "Alice was born in Paris."},
            {"position": 2, "title": "Paris", "text": "Paris is a city."},
        ],
    }

    row = call_selector(
        client,
        record,
        model="fake-model",
        max_tokens=64,
        temperature=0.0,
        append_no_think=True,
    )

    assert row["selected_positions"] == [1, 0]
    assert row["usage"] == {
        "prompt_tokens": 123,
        "completion_tokens": 7,
        "total_tokens": 130,
        "finish_reason": "stop",
    }
    assert row["latency_s"] >= 0.0
    assert row["prompt_chars"] > 0
    assert row["completion_chars"] == len("### Final Selection: [2] [1]")


def test_reorder_positions_frontloads_selection_then_rank_order_fallback():
    assert reorder_positions([2, 0, 2], pool_size=5) == [2, 0, 1, 3, 4]


def test_rank_order_fill_to_k_positions_truncates_and_fills_to_budget():
    order, fallback_count = rank_order_fill_to_k_positions([2, 0, 2], pool_size=5, fill_to_k=4)

    assert order == [2, 0, 1, 3]
    assert fallback_count == 2

    order, fallback_count = rank_order_fill_to_k_positions([4, 3, 2, 1, 0], pool_size=5, fill_to_k=3)

    assert order == [4, 3, 2]
    assert fallback_count == 0


def test_apply_selection_to_record_reorders_all_pool_fields_consistently():
    record = {
        "query_idx": 7,
        "pool_docs": ["doc0", "doc1", "doc2"],
        "pool_titles": ["t0", "t1", "t2"],
        "pool_doc_scores": [0.9, 0.8, 0.7],
        "pool_doc_ids": ["d0", "d1", "d2"],
    }
    selected, trace = apply_selection_to_record(record, {"selected_positions": [2, 0]})

    assert selected["pool_docs"] == ["doc2", "doc0", "doc1"]
    assert selected["pool_titles"] == ["t2", "t0", "t1"]
    assert selected["pool_doc_scores"] == [0.7, 0.9, 0.8]
    assert selected["pool_doc_ids"] == ["d2", "d0", "d1"]
    assert trace["selected_1based"] == [3, 1]
    assert trace["fallback_mode"] == "rank_order_fill"


def test_apply_selection_to_record_rank_order_fill_to_k_truncates_pool():
    record = {
        "query_idx": 7,
        "pool_docs": ["doc0", "doc1", "doc2", "doc3", "doc4"],
        "pool_titles": ["t0", "t1", "t2", "t3", "t4"],
        "pool_doc_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        "pool_doc_ids": ["d0", "d1", "d2", "d3", "d4"],
    }
    selected, trace = apply_selection_to_record(
        record,
        {"selected_positions": [2, 0, 2]},
        fallback_mode="rank_order_fill_to_k",
        fill_to_k=4,
    )

    assert selected["pool_docs"] == ["doc2", "doc0", "doc1", "doc3"]
    assert selected["pool_titles"] == ["t2", "t0", "t1", "t3"]
    assert selected["pool_doc_scores"] == [0.7, 0.9, 0.8, 0.6]
    assert selected["pool_doc_ids"] == ["d2", "d0", "d1", "d3"]
    assert trace["fallback_mode"] == "rank_order_fill_to_k"
    assert trace["fill_to_k"] == 4
    assert trace["fallback_count"] == 2
    assert trace["reader_pool_size"] == 4


def test_apply_selection_to_record_selected_only_does_not_rank_fill():
    record = {
        "query_idx": 7,
        "pool_docs": ["doc0", "doc1", "doc2"],
        "pool_titles": ["t0", "t1", "t2"],
        "pool_doc_scores": [0.9, 0.8, 0.7],
        "pool_doc_ids": ["d0", "d1", "d2"],
    }
    selected, trace = apply_selection_to_record(
        record,
        {"selected_positions": [2, 0]},
        fallback_mode="selected_only",
    )

    assert selected["pool_docs"] == ["doc2", "doc0"]
    assert selected["pool_titles"] == ["t2", "t0"]
    assert selected["pool_doc_scores"] == [0.7, 0.9]
    assert selected["pool_doc_ids"] == ["d2", "d0"]
    assert trace["fallback_mode"] == "selected_only"
    assert trace["fallback_count"] == 0
    assert trace["reader_pool_size"] == 2


def test_apply_selection_to_record_selected_only_empty_fallback_top1():
    record = {
        "query_idx": 7,
        "pool_docs": ["doc0", "doc1", "doc2"],
        "pool_titles": ["t0", "t1", "t2"],
        "pool_doc_scores": [0.9, 0.8, 0.7],
        "pool_doc_ids": ["d0", "d1", "d2"],
    }
    selected, trace = apply_selection_to_record(
        record,
        {"selected_positions": []},
        fallback_mode="selected_only",
        empty_fallback_top_n=1,
    )

    assert selected["pool_docs"] == ["doc0"]
    assert selected["pool_titles"] == ["t0"]
    assert trace["parse_success"] is False
    assert trace["empty_fallback_used"] is True
    assert trace["fallback_count"] == 1
    assert trace["reader_pool_size"] == 1


def test_build_setr_record_truncates_contexts_and_preserves_positions():
    record = {
        "query_idx": 1,
        "question": "Where was Alice born?",
        "gold_answers": ["Paris"],
        "gold_titles": ["Alice"],
        "pool_docs": ["Alice\nAlice was born in Paris.", "Other\nOther text."],
        "pool_titles": ["Alice", "Other"],
        "pool_doc_scores": [1.0, 0.5],
        "pool_doc_ids": ["a", "b"],
    }

    out = build_setr_record(record, pool_k=1, doc_max_chars=10, source_pool_path="pool.json")

    assert out["pool_k"] == 1
    assert out["contexts"][0]["position"] == 1
    assert out["contexts"][0]["title"] == "Alice"
    assert out["contexts"][0]["has_gold_title"] is True
    assert len(out["contexts"][0]["text"]) <= 10


def test_analyze_support_depth_uses_exact_doc_then_title_fallback():
    record = {
        "query_idx": 0,
        "question": "q",
        "gold_docs": ["Gold\nFull gold text.", "Missing\nMissing text."],
        "gold_titles": ["Gold", "Alias Title"],
        "pool_docs": ["Distractor\nx", "Gold\nFull gold text.", "Alias Title\ny"],
        "pool_titles": ["Distractor", "Gold", "Alias Title"],
    }

    row = analyze_record(record)

    assert row["support_ranks"] == [2, 3]
    stats = coverage_by_query([row], 2)
    assert stats["all_support_within_k"] == 0.0
    assert stats["any_support_beyond_k_or_missing"] == 1.0


def test_stage1_windows_cover_pool_prefix():
    assert stage1_windows(pool_k=45, window_size=20) == [
        (0, 20, list(range(0, 20))),
        (20, 40, list(range(20, 40))),
        (40, 45, list(range(40, 45))),
    ]


def test_windowed_selector_maps_local_stage_outputs_to_global_positions():
    record = {
        "query_idx": 3,
        "question": "Where was Alice born?",
        "gold_answers": ["Paris"],
        "gold_titles": ["Alice"],
        "pool_docs": [f"Doc {idx}\nText {idx}" for idx in range(12)],
        "pool_titles": [f"Doc {idx}" for idx in range(12)],
        "pool_doc_scores": list(range(12, 0, -1)),
        "pool_doc_ids": [str(idx) for idx in range(12)],
    }

    row = run_windowed_record(
        record,
        client=None,
        pool_k=12,
        window_size=5,
        stage1_max_per_window=2,
        stage2_max_candidates=6,
        final_max_count=3,
        doc_max_chars=128,
        model="mock",
        max_tokens=32,
        temperature=0.0,
        append_no_think=True,
        candidate_order="global_rank",
        mock_rank_order=True,
    )

    assert row["candidate_global_positions"] == [0, 1, 5, 6, 10, 11]
    assert row["selected_positions"] == [0, 1, 5]
    assert row["selected_1based"] == [1, 2, 6]
    assert row["stage1_parse_success_rate"] == 1.0
    assert row["selector_call_count"] == 4
    assert row["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert row["latency_s"] == 0.0
    assert row["completion_chars"] > 0
