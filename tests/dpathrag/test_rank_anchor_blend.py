import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dpathrag_build_rank_anchor_blend.py"
_SPEC = importlib.util.spec_from_file_location("dpathrag_build_rank_anchor_blend", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

blend_diagnostics = _MODULE.blend_diagnostics
blend_indices = _MODULE.blend_indices


def test_rank_anchor_blend_preserves_anchor_then_selector_fills() -> None:
    selected = blend_indices(
        rank_anchor_m=2,
        selector_indices=[4, 1, 5, 0, 3],
        candidate_count=10,
        top_k=5,
        max_candidates=10,
    )
    assert selected == [0, 1, 4, 5, 3]


def test_rank_anchor_blend_m5_matches_rank_topk() -> None:
    selected = blend_indices(
        rank_anchor_m=5,
        selector_indices=[8, 7, 6, 5, 4],
        candidate_count=10,
        top_k=5,
        max_candidates=10,
    )
    assert selected == [0, 1, 2, 3, 4]


def test_rank_anchor_blend_diagnostics_track_net_gold_gain() -> None:
    candidates = [
        {"gold_support": 0},
        {"gold_support": 1},
        {"gold_support": 0},
        {"gold_support": 1},
        {"gold_support": 0},
        {"gold_support": 1},
    ]
    diagnostics = blend_diagnostics(candidates=candidates, selected_indices=[0, 1, 5, 4, 3], rank_anchor_m=2, top_k=5)
    assert diagnostics["avg_rank_anchor_docs"] == 2.0
    assert diagnostics["avg_selector_added_docs"] == 3.0
    assert diagnostics["avg_selector_added_gold"] == 2.0
    assert diagnostics["avg_rank_tail_gold_removed"] == 0.0
    assert diagnostics["avg_net_gold_gain"] == 2.0
