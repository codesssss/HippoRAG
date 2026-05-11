from __future__ import annotations

from dataclasses import dataclass

from evidence_transition_graphragv4_composition.expander import build_pool_record_from_retrieval


@dataclass(frozen=True)
class DummyNode:
    text: str


@dataclass(frozen=True)
class DummyResult:
    doc_indices: tuple[int, ...]
    trace: dict


def test_build_pool_record_keeps_top5_prefix_then_candidate_tail() -> None:
    nodes = [DummyNode(f"Doc {idx}\nBody {idx}") for idx in range(8)]
    result = DummyResult(
        doc_indices=(2, 1, 0, 4, 3),
        trace={
            "candidate_universe": {
                "candidate_doc_indices": [5, 3, 6, 2],
                "candidate_count": 4,
                "candidate_source": "toy",
            }
        },
    )

    record = build_pool_record_from_retrieval(
        dataset="toy",
        query_idx=0,
        sample={"question": "Q?"},
        retrieval_result=result,
        nodes=nodes,
        gold_docs=[["Doc 2\nBody 2"]],
        gold_answers=[["A"]],
        pool_k=6,
        top_k=5,
    )

    assert record["pool_doc_ids"] == [2, 1, 0, 4, 3, 5]
    assert record["pool_titles"] == ["Doc 2", "Doc 1", "Doc 0", "Doc 4", "Doc 3", "Doc 5"]
    assert record["pool_doc_scores"][:5] == [12.0, 11.0, 10.0, 9.0, 8.0]
    assert record["agsto"]["selected_doc_indices"] == [2, 1, 0, 4, 3]
    assert record["agsto"]["external_pool_doc_ids"] == [2, 1, 0, 4, 3, 5]
