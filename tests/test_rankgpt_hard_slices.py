from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_rankgpt_hard_slices import (  # noqa: E402
    build_takeaway,
    gold_doc_count,
    subset_rows,
)


def test_gold_doc_count_handles_missing_and_present_titles() -> None:
    assert gold_doc_count({"gold_titles": ["A", "B"]}) == 2
    assert gold_doc_count({"gold_titles": []}) == 0
    assert gold_doc_count({}) == 0


def test_subset_rows_preserves_requested_order() -> None:
    rows = [{"id": 0}, {"id": 1}, {"id": 2}]

    assert subset_rows(rows, [2, 0]) == [{"id": 2}, {"id": 0}]


def test_takeaway_uses_row_values_not_static_numbers() -> None:
    rows = [
        {
            "dataset": "2Wiki",
            "slice": "gold_doc_count>=3",
            "comparison": "DAEC-selective - RankGPT-style sliding",
            "metric": "F1",
            "delta_mean": 0.1111,
            "ci_low": 0.01,
            "ci_high": 0.2,
            "ci_excludes_zero": True,
        },
        {
            "dataset": "2Wiki",
            "slice": "gold_doc_count>=3",
            "comparison": "DAEC-selective - SetR-faithful",
            "metric": "F1",
            "delta_mean": 0.2222,
            "ci_low": 0.03,
            "ci_high": 0.4,
            "ci_excludes_zero": True,
        },
        {
            "dataset": "2Wiki",
            "slice": "gold_doc_count>=3",
            "comparison": "SetR-faithful - RankGPT-style sliding",
            "metric": "F1",
            "delta_mean": -0.3333,
            "ci_low": -0.5,
            "ci_high": -0.1,
            "ci_excludes_zero": True,
        },
    ]

    takeaway = "\n".join(build_takeaway(rows))

    assert "`+0.1111` F1" in takeaway
    assert "`[+0.0100, +0.2000]`" in takeaway
    assert "`+0.2222` F1" in takeaway
    assert "SetR-faithful - RankGPT-style sliding = -0.3333" in takeaway
    assert "+0.0582" not in takeaway
