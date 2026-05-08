from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_daec_selective_cross_pool import metric_rows, top_titles_equal  # noqa: E402


def test_top_titles_equal_normalizes_titles() -> None:
    assert top_titles_equal(["A Film (1999)", "B-Title"], ["a film", "b title"])
    assert not top_titles_equal(["A", "B"], ["B", "A"])


def test_metric_rows_recomputes_title_recall() -> None:
    data = {
        "setwise_selector_query_traces": [
            {
                "question": "q",
                "gold_titles": ["A Film (1999)", "B"],
                "selector_top_titles": ["A Film", "C"],
                "selector_metrics": {"ExactMatch": 1, "F1": 0.75},
            }
        ]
    }

    rows = metric_rows(data, metric_field="selector_metrics", title_field="selector_top_titles")

    assert rows == [
        {
            "question": "q",
            "EM": 1.0,
            "F1": 0.75,
            "gold_titles": ["A Film (1999)", "B"],
            "top_titles": ["A Film", "C"],
            "R5_TITLE": 0.5,
        }
    ]
