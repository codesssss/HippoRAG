from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from export_agsto_pool import build_agsto_pool_records
from src.agsto_v12 import AGSTOConfig


def test_build_agsto_pool_records_exports_external_pool_schema() -> None:
    openie_docs = [
        {
            "idx": "0",
            "passage": "Alice\nAlice was born in Paris.",
            "extracted_triples": [("Alice", "was born in", "Paris")],
        },
        {
            "idx": "1",
            "passage": "Paris\nParis is in France.",
            "extracted_triples": [("Paris", "is in", "France")],
        },
        {
            "idx": "2",
            "passage": "Bob\nBob was born in Rome.",
            "extracted_triples": [("Bob", "was born in", "Rome")],
        },
    ]
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

    records, recall = build_agsto_pool_records(
        dataset="toy",
        samples=samples,
        openie_docs=openie_docs,
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
        pool_k=3,
        include_candidate_fill=True,
    )

    assert recall["Recall@5"] == 1.0
    assert len(records) == 1
    record = records[0]
    assert record["query_idx"] == 0
    assert record["question"] == samples[0]["question"]
    assert record["pool_doc_ids"][:2] == [0, 1]
    assert record["pool_titles"][:2] == ["Alice", "Paris"]
    assert len(record["pool_docs"]) == len(record["pool_doc_scores"]) == len(record["pool_doc_ids"])
    assert record["agsto"]["selected_doc_indices"] == [0, 1]
