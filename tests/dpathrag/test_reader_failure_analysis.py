import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dpathrag_analyze_reader_failures.py"
_SPEC = importlib.util.spec_from_file_location("dpathrag_analyze_reader_failures", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

analyze = _MODULE.analyze
answer_in_context = _MODULE.answer_in_context


def test_answer_in_context_normalizes_gold_answer() -> None:
    record = {
        "answer": "The Kearney",
        "selected_docs": [
            {"title": "Doc", "text": "Jon Bokenkamp was born in Kearney, Nebraska."},
        ],
    }
    assert answer_in_context(record)


def test_analyze_reader_failures_counts_context_failures() -> None:
    reader_rows = [
        {
            "qid": "q1",
            "answer": "Kearney",
            "support_complete": 1.0,
            "support_recall": 1.0,
            "selected_docs": [{"title": "Doc", "text": "The answer is Kearney."}],
        },
        {
            "qid": "q2",
            "answer": "Paris",
            "support_complete": 0.0,
            "support_recall": 0.5,
            "selected_docs": [{"title": "Doc", "text": "No answer here."}],
        },
    ]
    prediction_rows = [
        {"qid": "q1", "prediction": "Frederick", "gold_answers": ["Kearney"], "em": 0.0, "f1": 0.0},
        {"qid": "q2", "prediction": "Paris", "gold_answers": ["Paris"], "em": 1.0, "f1": 1.0},
    ]
    report = analyze(reader_rows, prediction_rows)
    assert report["summary"]["answer_in_context_rate"] == 0.5
    assert report["summary"]["answer_in_context_but_fail_rate"] == 0.5
    assert report["summary"]["support_complete_but_fail_rate"] == 0.5
