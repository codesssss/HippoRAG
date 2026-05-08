from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_daec_depth_resolvability import (  # noqa: E402
    align_rows_by_reference,
    gate_decision,
    slice_specs,
    support_depth,
    trace_metadata_rows,
)


def test_support_depth_prefers_gold_doc_count_with_len_fallback() -> None:
    assert support_depth({"gold_doc_count": "4", "gold_titles": ["A", "B"]}) == 4
    assert support_depth({"gold_doc_count": "bad", "gold_titles": ["A", "B", "C"]}) == 3
    assert support_depth({"gold_titles": ["A"]}) == 1


def test_gate_decision_normalizes_known_values_and_unknowns() -> None:
    assert gate_decision({"selector_trace": {"selective_binding_decision": "Bind"}}) == "bind"
    assert gate_decision({"selector_trace": {"selective_binding_decision": "abstain"}}) == "abstain"
    assert gate_decision({"selector_trace": {"selective_binding_decision": "skip"}}) == "unknown"
    assert gate_decision({}) == "unknown"


def test_trace_metadata_rows_extracts_gate_and_binding_fields() -> None:
    data = {
        "setwise_selector_query_traces": [
            {
                "question": "q",
                "gold_titles": ["A", "B"],
                "selector_trace": {
                    "selective_binding_decision": "bind",
                    "selective_binding_title_unique_rate": 0.9,
                    "binding_count": 2,
                    "binding_count_unpruned": 3,
                    "requirement_count": 4,
                },
            }
        ]
    }

    assert trace_metadata_rows(data) == [
        {
            "question": "q",
            "gold_doc_count": 2,
            "gate_decision": "bind",
            "title_unique_rate": 0.9,
            "binding_count": 2,
            "binding_count_unpruned": 3,
            "requirement_count": 4,
        }
    ]


def test_align_rows_by_reference_recomputes_r5_from_reference_gold() -> None:
    methods = {
        "DBEC-IG": [
            {
                "question": "q",
                "EM": 1.0,
                "F1": 1.0,
                "gold_titles": ["A Film (1999)", "B"],
                "top_titles": ["A Film", "B"],
            }
        ],
        "Other": [
            {
                "question": "q",
                "EM": 0.0,
                "F1": 0.0,
                "gold_titles": [],
                "top_titles": ["A Film"],
            }
        ],
    }

    aligned = align_rows_by_reference(dataset="Toy", reference_method="DBEC-IG", methods=methods)

    assert aligned["DBEC-IG"][0]["R5_TITLE"] == 1.0
    assert aligned["Other"][0]["gold_titles"] == ["A Film (1999)", "B"]
    assert aligned["Other"][0]["R5_TITLE"] == 0.5


def test_slice_specs_include_depth_gate_joint_predicates() -> None:
    specs = dict(slice_specs())

    assert specs["support_depth>=3|gate=bind"]({"gold_doc_count": 4, "gate_decision": "bind"})
    assert not specs["support_depth>=3|gate=bind"]({"gold_doc_count": 2, "gate_decision": "bind"})
    assert specs["support_depth=2|gate=abstain"]({"gold_doc_count": 2, "gate_decision": "abstain"})
