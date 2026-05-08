from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_daec_setr_cross_pool_ci import (  # noqa: E402
    fill_reference_gold_and_recompute_r5,
    setr_path,
)


def test_setr_path_uses_cross_pool_k20_doc768_name() -> None:
    assert setr_path("hotpotqa", "hipporag") == Path(
        "reports/setr_full1000_20260503/hotpotqa_hipporag_setr_k20_doc768.eval.json"
    )


def test_fill_reference_gold_and_recompute_r5_for_setr_rows() -> None:
    methods = {
        "DAEC-selective": [
            {
                "gold_titles": ["A Film (1999)", "B"],
                "top_titles": ["A Film", "B"],
                "R5_TITLE": 0.0,
            }
        ],
        "SetR-style k20": [
            {
                "gold_titles": [],
                "top_titles": ["A Film", "C"],
                "R5_TITLE": 0.0,
            }
        ],
    }

    fill_reference_gold_and_recompute_r5(methods)

    assert methods["DAEC-selective"][0]["R5_TITLE"] == 1.0
    assert methods["SetR-style k20"][0]["gold_titles"] == ["A Film (1999)", "B"]
    assert methods["SetR-style k20"][0]["R5_TITLE"] == 0.5
