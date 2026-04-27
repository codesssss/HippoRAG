from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_pool_support_depth import analyze_record, coverage_by_query
from apply_setr_selection_to_pool import apply_selection_to_record, reorder_positions
from export_setr_inputs import build_setr_record
from run_setr_style_selector import parse_setr_selection
from run_setr_windowed_selector import run_windowed_record, stage1_windows


def test_parse_setr_selection_prefers_final_selection_and_filters_invalid_ids():
    text = "Step 1: requirement [9]\n### Final Selection: [3] [1] [3] [999] [2]"

    assert parse_setr_selection(text, max_position=4) == [2, 0, 1]


def test_reorder_positions_frontloads_selection_then_rank_order_fallback():
    assert reorder_positions([2, 0, 2], pool_size=5) == [2, 0, 1, 3, 4]


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
