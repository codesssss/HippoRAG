import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dpathrag_build_context_ablation.py"
_SPEC = importlib.util.spec_from_file_location("dpathrag_build_context_ablation", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

select_indices = _MODULE.select_indices


def test_context_ablation_selects_expected_orders() -> None:
    candidates = [
        {"gold_support": 0},
        {"gold_support": 1},
        {"gold_support": 0},
        {"gold_support": 1},
        {"gold_support": 0},
    ]
    prediction = {"selected_indices": [2, 3, 0, 1, 4]}
    assert select_indices("rank_top5", candidates, prediction, top_k=5, max_candidates=5) == [0, 1, 2, 3, 4]
    assert select_indices("selector_ar_order", candidates, prediction, top_k=5, max_candidates=5) == [2, 3, 0, 1, 4]
    assert select_indices("selector_rank_order", candidates, prediction, top_k=5, max_candidates=5) == [0, 1, 2, 3, 4]
    assert select_indices("selector_gold_first", candidates, prediction, top_k=5, max_candidates=5) == [3, 1, 2, 0, 4]
    assert select_indices("gold_support_only", candidates, prediction, top_k=5, max_candidates=5) == [1, 3]
    assert select_indices("gold_plus_selector_distractors", candidates, prediction, top_k=5, max_candidates=5) == [1, 3, 2, 0, 4]
