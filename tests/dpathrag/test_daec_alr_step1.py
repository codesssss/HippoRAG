from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_daec_alr_step1.py"
_SPEC = importlib.util.spec_from_file_location("run_daec_alr_step1", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def toy_files(tmp_path: Path) -> tuple[Path, Path]:
    report = {
        "setwise_selector_query_traces": [
            {
                "question": "Who directed Film A?",
                "query_type": "bridge",
                "gold_titles": ["Film A", "Director A"],
                "gold_answers": ["Director A"],
                "selector_metrics": {"F1": 0.0},
                "selector_trace": {
                    "selected_pool_positions": [0, 1, 2],
                    "final_front_pool_positions": [0, 1, 2],
                    "match_threshold": 0.35,
                    "requirements": [
                        {
                            "unit_id": "s1",
                            "subquery": "Which film is Film A?",
                            "depends_on": [],
                            "expected_answer_type": "entity",
                            "anchor_mentions": ["Film A"],
                        },
                        {
                            "unit_id": "s2",
                            "subquery": "Who directed that film?",
                            "depends_on": ["s1"],
                            "expected_answer_type": "person",
                            "anchor_mentions": [],
                        },
                    ],
                    "coverage_by_requirement": {"s1": 1.0},
                    "cover_position_by_requirement": {"s1": 0},
                    "selection_steps": [
                        {
                            "pool_position": 0,
                            "covered_requirement_id": "s1",
                            "coverage_score": 1.0,
                            "mode": "reserve",
                        },
                        {"pool_position": 1, "reason": "baseline_fill", "mode": "baseline_fill"},
                        {"pool_position": 2, "reason": "baseline_fill", "mode": "baseline_fill"},
                    ],
                    "binding_candidates_by_requirement": {
                        "s2": [{"title": "Director A", "dep": "s1", "first_pos": 3}]
                    },
                    "grounded_question_entities_preview": ["Film A"],
                },
            }
        ]
    }
    pool = {
        "records": [
            {
                "query_idx": 0,
                "question": "Who directed Film A?",
                "gold_answers": ["Director A"],
                "gold_titles": ["Film A", "Director A"],
                "pool_docs": [
                    "Film A\nFilm A is a film.",
                    "Distractor\nWrong text.",
                    "Other\nOther text.",
                    "Director A\nDirector A directed Film A.",
                    "Film A\nDuplicate title.",
                    "Unrelated\nNo shared terms.",
                ],
                "pool_titles": ["Film A", "Distractor", "Other", "Director A", "Film A", "Unrelated"],
                "pool_doc_ids": ["d0", "d1", "d2", "d3", "d4", "d5"],
                "pool_doc_scores": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            }
        ]
    }
    report_path = tmp_path / "report.json"
    pool_path = tmp_path / "pool.json"
    write_json(report_path, report)
    write_json(pool_path, pool)
    return report_path, pool_path


def test_build_step1_records_preserves_pool_and_protected_positions(tmp_path: Path) -> None:
    report_path, pool_path = toy_files(tmp_path)

    records = _MODULE.build_step1_records(
        report_json=report_path,
        pool_json=pool_path,
        limit=1,
        offset=0,
        top_k=3,
        source="test",
    )

    assert len(records) == 1
    record = records[0]
    assert [doc["title"] for doc in record["selected_docs"]] == ["Film A", "Distractor", "Other"]
    assert len(record["pool_docs"]) == 6
    assert record["protected_positions"] == [0]
    assert record["selected_doc_utilities"][0] == 1.0


def test_candidate_prefilter_rejects_duplicate_and_keeps_binding_candidate(tmp_path: Path) -> None:
    report_path, pool_path = toy_files(tmp_path)
    record = _MODULE.build_step1_records(
        report_json=report_path,
        pool_json=pool_path,
        limit=1,
        offset=0,
        top_k=3,
        source="test",
    )[0]

    plans, meta = _MODULE.build_candidate_plans(
        record,
        variant_name="double_gate_skip",
        binding_required=True,
        allow_protected_replacement=False,
        k_edit=5,
    )

    assert meta["duplicate_rejected"] == 1
    assert plans
    assert plans[0]["candidate"]["title"] == "Director A"
    assert plans[0]["binding_pass"] is True
    assert plans[0]["target"]["pool_position"] in {1, 2}


def test_perturb_docs_validation_variants_drop_expected_slots() -> None:
    docs = [{"title": "A"}, {"title": "B"}, {"title": "C"}]

    assert [doc["title"] for doc in _MODULE.perturb_docs(docs, "drop_first")] == ["B", "C"]
    assert [doc["title"] for doc in _MODULE.perturb_docs(docs, "drop_weakest_selected", weakest_index=1)] == ["A", "C"]
    assert [doc["title"] for doc in _MODULE.perturb_docs(docs, "drop_candidate", candidate_index=2)] == ["A", "B"]
    assert [doc["title"] for doc in _MODULE.perturb_docs(docs, "swap01_after_drop_last")] == ["B", "A"]


def test_main_gate_decision_red_when_added_gold_missing() -> None:
    baseline = {"answer_f1": 0.5, "support_complete": 0.8}
    summary = {
        "answer_f1": 0.51,
        "support_complete": 0.8,
        "f1_delta_ci95": [0.001, 0.02],
        "accepted_edit_rate": 0.2,
        "edited_subset_pre_f1": 0.2,
        "edited_subset_post_f1": 0.4,
        "wrong_to_correct_flips": 3,
        "correct_to_wrong_flips": 1,
        "added_gold": 0,
        "added_non_gold_per_added_gold": None,
    }

    assert _MODULE.main_gate_decision(summary, baseline) == "RED"


def test_cache_key_includes_generation_limits() -> None:
    record = {
        "qid": "q1",
        "selected_docs": [{"doc_id": "d1"}],
        "family": "round0",
        "variant": "original",
        "max_new_tokens": 32,
        "max_docs": 0,
        "max_doc_chars": 0,
    }
    key_32 = _MODULE.cache_key_for_record(record, dataset="d", prompt_id="p", model="m")
    record["max_new_tokens"] = 64
    key_64 = _MODULE.cache_key_for_record(record, dataset="d", prompt_id="p", model="m")

    assert key_32 != key_64
