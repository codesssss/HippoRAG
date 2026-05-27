from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from export_agsto_cached_pool import build_cached_agsto_pool_records
from src.agsto_v12 import AGSTOConfig


def test_build_cached_agsto_pool_records_uses_cached_proposals(tmp_path: Path) -> None:
    openie_path = tmp_path / "openie.json"
    openie_path.write_text(
        json.dumps(
            {
                "docs": [
                    {
                        "idx": "0",
                        "passage": "Alice\nAlice was born in Paris.",
                        "extracted_triples": [["Alice", "was born in", "Paris"]],
                    },
                    {
                        "idx": "1",
                        "passage": "Paris\nParis is in France.",
                        "extracted_triples": [["Paris", "is in", "France"]],
                    },
                    {
                        "idx": "2",
                        "passage": "Rome\nRome is in Italy.",
                        "extracted_triples": [["Rome", "is in", "Italy"]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    samples = [
        {
            "question": "Where was Alice born and what country is Paris in?",
            "answer": "France",
            "paragraphs": [
                {"title": "Alice", "paragraph_text": "Alice was born in Paris.", "is_supporting": True},
                {"title": "Paris", "paragraph_text": "Paris is in France.", "is_supporting": True},
            ],
        }
    ]
    dataset_payload = {
        "dataset": "toy",
        "openie_path": str(openie_path),
        "rows": [
            {
                "query_index": 0,
                "question": samples[0]["question"],
                "gold_doc_indices": [0, 1],
                "context_anchor_doc_indices": [0],
                "native_dense_doc_indices_top10": [0, 1, 2],
                "specificity_pair_doc_indices_top10": [1, 0],
                "endpoint_transition_doc_indices_top10": [1],
                "hybrid_residual_pair_doc_indices_top10": [0, 1],
                "query_conditioned_neighborhood": {
                    "selected_neighborhood_doc_indices": [0, 1],
                    "retrieved_doc_indices": [0, 1, 2],
                },
                "support_set_search": {
                    "top_support_sets": [{"doc_indices": [0, 1], "score": 2.0}],
                },
            }
        ],
    }

    records, recall, stats = build_cached_agsto_pool_records(
        dataset="toy",
        samples=samples,
        dataset_payload=dataset_payload,
        transition_report_dir=tmp_path,
        config=AGSTOConfig(
            retrieval_top_k=3,
            evidence_set_size=2,
            stable_anchor_k=1,
            proposal_candidate_depth=3,
            support_proposal_depth=2,
            candidate_limit=5,
            beam_size=3,
            max_endpoint_degree=5,
        ),
        max_queries=0,
        pool_k=100,
        include_candidate_fill=True,
    )

    assert recall["Recall@5"] == 1.0
    assert stats["mean_candidate_doc_count"] >= 2.0
    assert len(records) == 1
    record = records[0]
    assert record["query_idx"] == 0
    assert record["question"] == samples[0]["question"]
    assert record["pool_titles"][:2] == ["Alice", "Paris"]
    assert record["agsto"]["proposal_source"] == "cached"
    assert record["agsto"]["evidence_completion_policy"] == "graph_obligated"
