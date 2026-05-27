from __future__ import annotations

import json
from importlib import util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_v13b_structural_source_report.py"
_SPEC = util.spec_from_file_location("build_v13b_structural_source_report", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_report = _MODULE.build_report


def test_structural_report_does_not_forge_anchor_guided_fields(tmp_path: Path):
    pool_path = tmp_path / "toy_pool.json"
    openie_path = tmp_path / "openie.json"
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "report.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "query_idx": 0,
                        "question": "Who signed Messi?",
                        "gold_titles": ["Barcelona"],
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
                    {
                        "idx": "messi",
                        "passage": "Messi\nMessi was signed by Barcelona.",
                        "extracted_triples": [["Messi", "signed by", "Barcelona"]],
                    },
                    {
                        "idx": "barcelona",
                        "passage": "Barcelona\nBarcelona signed Messi.",
                        "extracted_triples": [["Barcelona", "signed", "Messi"]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset": "toy",
                        "rows": [
                            {
                                "query_index": 0,
                                "question": "Who signed Messi?",
                                "query_triples": [["Messi", "signed by", "?answer"]],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        datasets=["toy"],
        pool_template=str(pool_path),
        openie_template=str(openie_path),
        query_obligation_cache_template=str(cache_path),
        output_report_path=output_path,
        max_queries=0,
        dense_limit=2,
    )

    row = report["datasets"][0]["rows"][0]
    assert "anchor_guided_evidence_doc_indices_top10" not in row
    assert row["native_dense_doc_indices_top10"] == [0, 1]
    assert 0 in row["support_set_doc_indices_top10"]
    assert "dense_seed" in row["candidate_doc_provenance"]["0"]
    assert "support_grounding" in row["candidate_doc_provenance"]["0"]


def test_structural_report_skips_hub_endpoint_expansion(tmp_path: Path):
    pool_path = tmp_path / "toy_pool.json"
    openie_path = tmp_path / "openie.json"
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "report.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "query_idx": 0,
                        "question": "What does Alpha mention?",
                        "pool_doc_ids": [0],
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
                    {
                        "idx": "alpha",
                        "passage": "Alpha\nAlpha mentions Hub.",
                        "extracted_triples": [["Alpha", "mentions", "Hub"]],
                    },
                    {
                        "idx": "beta",
                        "passage": "Beta\nBeta mentions Hub.",
                        "extracted_triples": [["Beta", "mentions", "Hub"]],
                    },
                    {
                        "idx": "gamma",
                        "passage": "Gamma\nGamma mentions Hub.",
                        "extracted_triples": [["Gamma", "mentions", "Hub"]],
                    },
                    {
                        "idx": "delta",
                        "passage": "Delta\nDelta mentions Hub.",
                        "extracted_triples": [["Delta", "mentions", "Hub"]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset": "toy",
                        "rows": [
                            {
                                "query_index": 0,
                                "question": "What does Alpha mention?",
                                "query_triples": [["Alpha", "mentions", "?x1"]],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        datasets=["toy"],
        pool_template=str(pool_path),
        openie_template=str(openie_path),
        query_obligation_cache_template=str(cache_path),
        output_report_path=output_path,
        max_queries=0,
        dense_limit=1,
        max_endpoint_doc_degree=2,
    )

    row = report["datasets"][0]["rows"][0]
    assert row["candidate_doc_indices"] == [0]
    assert row["structural_expansion_summary"]["endpoint_hub_skipped_count"] == 1
    assert row["structural_expansion_summary"]["endpoint_hub_skipped"] == {"hub": 4}


def test_structural_report_keeps_low_degree_endpoint_expansion(tmp_path: Path):
    pool_path = tmp_path / "toy_pool.json"
    openie_path = tmp_path / "openie.json"
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "report.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "query_idx": 0,
                        "question": "What does Alpha mention?",
                        "pool_doc_ids": [0],
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
                    {
                        "idx": "alpha",
                        "passage": "Alpha\nAlpha mentions Hub.",
                        "extracted_triples": [["Alpha", "mentions", "Hub"]],
                    },
                    {
                        "idx": "beta",
                        "passage": "Beta\nBeta mentions Hub.",
                        "extracted_triples": [["Beta", "mentions", "Hub"]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset": "toy",
                        "rows": [
                            {
                                "query_index": 0,
                                "question": "What does Alpha mention?",
                                "query_triples": [["Alpha", "mentions", "?x1"]],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        datasets=["toy"],
        pool_template=str(pool_path),
        openie_template=str(openie_path),
        query_obligation_cache_template=str(cache_path),
        output_report_path=output_path,
        max_queries=0,
        dense_limit=1,
        max_endpoint_doc_degree=2,
    )

    row = report["datasets"][0]["rows"][0]
    assert row["candidate_doc_indices"] == [0, 1]
    assert row["structural_expansion_summary"]["endpoint_hub_skipped_count"] == 0
    assert "endpoint_transition" in row["candidate_doc_provenance"]["1"]
