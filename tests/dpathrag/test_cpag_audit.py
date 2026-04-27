from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_cpag_implementation.py"
_SPEC = importlib.util.spec_from_file_location("audit_cpag_implementation", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_manual_rrf_by_doc_key_matches_expected_sum() -> None:
    prop = {"candidates": [{"title": "Alpha", "doc_id": 1}, {"title": "Beta", "doc_id": 2}]}
    dense = {"candidates": [{"title": "Alpha", "doc_id": 1}, {"title": "Gamma", "doc_id": 3}]}
    scores = _MODULE.manual_rrf_by_doc_key(prop, dense, pool_k=2, k_const=60)
    key_a = "title::alpha"
    assert abs(scores[key_a] - (1 / 61 + 1 / 61)) < 1e-9


def test_support_complete_manual_uses_titles() -> None:
    record = {"gold_titles": ["A", "B"]}
    assert _MODULE.support_complete_manual(record, [{"title": "A"}, {"title": "B"}]) == 1.0
    assert _MODULE.support_complete_manual(record, [{"title": "A"}, {"title": "C"}]) == 0.0
