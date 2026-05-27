from __future__ import annotations

import json
from importlib import util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "v13b_make_source_report.py"
_SPEC = util.spec_from_file_location("v13b_make_source_report", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_report = _MODULE.build_report


def test_build_report_maps_pool_doc_ids_and_gold_titles(tmp_path: Path):
    pool_path = tmp_path / "toy_pool.json"
    openie_path = tmp_path / "openie.json"
    output_path = tmp_path / "report.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "query_idx": 7,
                        "question": "When did Alpha's mother die?",
                        "gold_answers": ["1900"],
                        "gold_titles": ["Alpha", "Beta"],
                        "gold_docs": ["Alpha\nAlpha text.", "Beta\nBeta text."],
                        "pool_doc_ids": [2, 1, 2, 0],
                        "pool_doc_scores": [0.9, 0.8, 0.7, 0.6],
                        "pool_titles": ["Gamma", "Beta", "Gamma", "Alpha"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    openie_path.write_text(
        json.dumps(
            {
                "docs": [
                    {"passage": "Alpha\nAlpha text."},
                    {"passage": "Beta\nBeta text."},
                    {"passage": "Gamma\nGamma text."},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        datasets=["toy"],
        pool_template=str(pool_path),
        openie_template=str(openie_path),
        output_report_path=output_path,
        max_queries=0,
        candidate_limit=3,
    )

    row = report["datasets"][0]["rows"][0]
    assert row["query_index"] == 7
    assert row["candidate_doc_indices"] == [2, 1, 0]
    assert row["retrieved_doc_indices_top5"] == [2, 1, 0]
    assert row["gold_doc_indices"] == [0, 1]
    variant_rows = report["variants"]["hippohead_qgate_lexbeam_topkfact_stabilityfallback_guarded_local_ppr_gamma_0.3"]
    assert variant_rows[0]["dataset"] == "toy"


def test_build_report_prefers_passage_match_over_duplicate_titles(tmp_path: Path):
    pool_path = tmp_path / "toy_pool.json"
    openie_path = tmp_path / "openie.json"
    output_path = tmp_path / "report.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "query_idx": 0,
                        "question": "Which Alpha?",
                        "gold_titles": ["Alpha"],
                        "gold_docs": ["Alpha\nGold passage."],
                        "pool_doc_ids": [0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    openie_path.write_text(
        json.dumps(
            {
                "docs": [
                    {"passage": "Alpha\nDistractor passage."},
                    {"passage": "Alpha\nGold passage."},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        datasets=["toy"],
        pool_template=str(pool_path),
        openie_template=str(openie_path),
        output_report_path=output_path,
        max_queries=0,
        candidate_limit=100,
    )

    row = report["datasets"][0]["rows"][0]
    assert row["gold_doc_indices"] == [1]
