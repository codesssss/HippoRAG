from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "caps_day1_5_candidate_v2.py"
_SPEC = importlib.util.spec_from_file_location("caps_day1_5_candidate_v2", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_parse_llm_candidates_prefers_json_answers() -> None:
    text = '<think>hidden reasoning</think>\n{"answers": ["Andy Summers", "31 December 1942"]}'
    assert _MODULE.parse_llm_candidates(text) == ["Andy Summers", "31 December 1942"]


def test_parse_llm_candidates_recovers_short_entity_variants() -> None:
    text = "\n".join(
        [
            "1. Andy Summers was born on 31 December 1942",
            "2. Aivar Kuusmaa was born on 12 June 1967",
        ]
    )
    parsed = _MODULE.parse_llm_candidates(text)
    assert "Andy Summers was born on 31 December 1942" in parsed
    assert "Andy Summers" in parsed
    assert "Aivar Kuusmaa" in parsed


def test_summarize_recall_decisions() -> None:
    rows = [{"recall_at_5": 1.0, "recall_at_10": 1.0, "recall_at_20": 1.0, "candidate_count": 10}]
    assert _MODULE.summarize_recall(rows, recall5_gate=0.85, recall10_gate=0.9)["decision"] == "PROCEED_GATE2_WITH_TOP5"
    rows = [{"recall_at_5": 0.0, "recall_at_10": 1.0, "recall_at_20": 1.0, "candidate_count": 10}]
    assert _MODULE.summarize_recall(rows, recall5_gate=0.85, recall10_gate=0.9)["decision"] == "REVIEW_PROCEED_GATE2_WITH_TOP10"
    rows = [{"recall_at_5": 0.0, "recall_at_10": 0.0, "recall_at_20": 1.0, "candidate_count": 10}]
    assert _MODULE.summarize_recall(rows, recall5_gate=0.85, recall10_gate=0.9)["decision"] == "STOP_CANDIDATE_V2_RECALL_FAIL"
