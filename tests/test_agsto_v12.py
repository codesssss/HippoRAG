from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.agsto_v12 import (
    AGSTOConfig,
    AGSTORetriever,
    build_corpus_unit_index,
    rank_docs_bm25,
    rank_sto_proposal_consensus_docs,
    unique_ranked,
)
from src.agsto_v12.build_openie import build_openie_from_corpus, corpus_rows_to_passages, heuristic_openie
from src.agsto_v12.evaluate_from_openie import (
    evaluate_agsto_v12_from_openie,
    gold_doc_indices,
    load_query_rows,
)


def toy_openie_docs() -> list[dict[str, object]]:
    return [
        {
            "idx": "0",
            "passage": "Alice\nAlice was born in Paris. Alice wrote a book.",
            "extracted_triples": [
                ("Alice", "was born in", "Paris"),
                ("Alice", "wrote", "Book"),
            ],
        },
        {
            "idx": "1",
            "passage": "Paris\nParis is in France. France is in Europe.",
            "extracted_triples": [
                ("Paris", "is in", "France"),
                ("France", "is in", "Europe"),
            ],
        },
        {
            "idx": "2",
            "passage": "Bob\nBob was born in Rome. Rome is in Italy.",
            "extracted_triples": [
                ("Bob", "was born in", "Rome"),
                ("Rome", "is in", "Italy"),
            ],
        },
    ]


def toy_config(**overrides: object) -> AGSTOConfig:
    values = {
        "evidence_set_size": 2,
        "retrieval_top_k": 3,
        "stable_anchor_k": 1,
        "proposal_candidate_depth": 4,
        "support_proposal_depth": 3,
        "candidate_limit": 10,
        "beam_size": 4,
        "max_endpoint_degree": 5,
    }
    values.update(overrides)
    return AGSTOConfig(**values)


def test_config_validation_and_policy_mapping() -> None:
    assert AGSTOConfig(policy="graph").completion_policy == "graph_obligated"
    assert AGSTOConfig(policy="base").completion_policy == "none"

    with pytest.raises(ValueError, match="Unsupported AG-STO policy"):
        AGSTOConfig(policy="diagnostic")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="retrieval_top_k"):
        AGSTOConfig(retrieval_top_k=0)


def test_index_bm25_and_ranking_helpers_are_deterministic() -> None:
    corpus_index = build_corpus_unit_index(toy_openie_docs())

    assert len(corpus_index["units"]) >= 6
    assert "paris" in corpus_index["endpoint_to_docs"]
    assert unique_ranked([2, 1, 2, -1, 0]) == [2, 1, 0]

    ranked = rank_docs_bm25(query="Alice born Paris", corpus_index=corpus_index, top_k=3)
    assert ranked[:2] == [0, 1]

    consensus = rank_sto_proposal_consensus_docs(
        proposal_doc_indices=[[2, 0], [0, 1], [1, 0]],
        proposal_weights=[1.0, 2.0, 1.0],
        top_k=3,
    )
    assert consensus == [0, 1, 2]


def test_retrieve_native_selects_connected_agsto_evidence_set() -> None:
    retriever = AGSTORetriever.from_openie_docs(toy_openie_docs(), config=toy_config())

    result = retriever.retrieve_native(
        query="Where was Alice born and what country is Paris in?",
        anchor_doc_indices=[0],
    )

    assert result["method"] == "AG-STO"
    assert result["agsto_policy"] == "graph"
    assert result["clean_api"] is True
    assert result["selected_evidence_set"]["doc_indices"] == [0, 1]
    assert result["retrieved_doc_indices"][:2] == [0, 1]
    assert result["selected_evidence_set"]["missing_tree_edges"] == 0
    assert result["graph_obligated_completion"]["policy"] == "graph_obligated"
    assert "0" in result["proposal_role_summary_by_doc"]


def test_load_query_rows_and_gold_inference_accept_supported_shapes(tmp_path: Path) -> None:
    rows_path = tmp_path / "queries.json"
    rows_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset": "toy",
                        "rows": [
                            {
                                "question": "Where is Alice from?",
                                "paragraphs": [
                                    {"idx": "0", "is_supporting": True},
                                    {"idx": "1", "is_supporting": False},
                                    {"idx": 2, "is_supporting": True},
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = load_query_rows(rows_path, dataset="toy")

    assert len(rows) == 1
    assert gold_doc_indices(rows[0]) == [0, 2]
    with pytest.raises(ValueError, match="No rows found"):
        load_query_rows(rows_path, dataset="missing")


def test_build_openie_from_corpus_heuristic_writes_portable_cache(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    output_path = tmp_path / "openie.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"idx": 7, "title": "Alice", "text": "Alice visited Paris in 1999."},
                {"idx": 8, "passage": "Paris\nParis is in France."},
            ]
        ),
        encoding="utf-8",
    )

    passages = corpus_rows_to_passages(json.loads(corpus_path.read_text(encoding="utf-8")))
    assert passages[0]["passage"].startswith("Alice\n")
    assert heuristic_openie(str(passages[0]["passage"]))["extracted_triples"]

    payload = build_openie_from_corpus(
        argparse.Namespace(
            corpus_json=str(corpus_path),
            output_openie_json=str(output_path),
            mode="heuristic",
            limit_docs=0,
            workers=1,
            continue_on_error=False,
            heuristic_max_entities=8,
            heuristic_max_triples=8,
            llm_base_url="http://unused.example/v1",
            llm_model="unused",
            api_key=None,
            api_key_file=None,
            allow_empty_api_key=False,
            timeout=1.0,
            max_tokens=64,
            retries=1,
        )
    )

    assert output_path.exists()
    assert payload["metadata"]["mode"] == "heuristic"
    assert [doc["idx"] for doc in payload["docs"]] == ["chunk-7", "chunk-8"]


def test_evaluate_from_openie_smoke_writes_summary_files(tmp_path: Path) -> None:
    openie_path = tmp_path / "openie.json"
    queries_path = tmp_path / "queries.json"
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    openie_path.write_text(json.dumps({"docs": toy_openie_docs()}), encoding="utf-8")
    queries_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "question": "Where was Alice born and what country is Paris in?",
                        "gold_doc_indices": [0, 1],
                        "context_anchor_doc_indices": [0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_agsto_v12_from_openie(
        argparse.Namespace(
            openie_results=str(openie_path),
            queries_json=str(queries_path),
            dataset="toy",
            policy="graph",
            max_queries=1,
            retrieval_top_k=3,
            evidence_set_size=2,
            stable_anchor_k=1,
            proposal_candidate_depth=4,
            support_proposal_depth=3,
            candidate_limit=10,
            beam_size=4,
            set_search_policy="beam",
            max_endpoint_degree=5,
            semantic_residual_weight=0.0,
            support_realization="direct",
            support_mct_iterations=4,
            support_mct_depth=2,
            support_mct_exploration=0.0,
            output_json=str(output_json),
            output_md=str(output_md),
        )
    )

    assert output_json.exists()
    assert output_md.exists()
    assert summary["clean_package"] == "agsto_v12"
    dataset = summary["datasets"][0]
    assert dataset["metrics"]["agsto_v12_r5"] == 1.0
    assert dataset["rows"][0]["agsto_v12_doc_indices_top5"][:2] == [0, 1]
