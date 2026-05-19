from __future__ import annotations

from dataclasses import dataclass

from evidenceflow.expander import build_pool_record_from_retrieval
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.candidate_generator import (
    apply_role_graph_edge_policy,
)


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


def test_phrase_source_policy_keeps_only_phrase_grounding_edges() -> None:
    role_graph = {
        "edges": {
            (1, 2): {
                "left_doc": 1,
                "right_doc": 2,
                "kinds": ["sentence_grounded_transition", "source_endpoint_incidence"],
                "endpoints": ["alpha"],
                "samples": [
                    {"kind": "sentence_grounded_transition"},
                    {"kind": "source_endpoint_incidence"},
                ],
            },
            (2, 3): {
                "left_doc": 2,
                "right_doc": 3,
                "kinds": ["role_bridge"],
                "endpoints": ["beta"],
                "samples": [{"kind": "role_bridge"}],
            },
            (3, 4): {
                "left_doc": 3,
                "right_doc": 4,
                "kinds": ["title_role_grounding"],
                "endpoints": ["gamma"],
                "samples": [{"kind": "title_role_grounding"}],
            },
        },
        "stats": {},
    }

    filtered = apply_role_graph_edge_policy(role_graph, policy="phrase_source_only")

    assert sorted(filtered["edges"]) == [(1, 2), (3, 4)]
    assert filtered["edges"][(1, 2)]["kinds"] == ["source_endpoint_incidence"]
    assert filtered["edges"][(3, 4)]["kinds"] == ["title_role_grounding"]
    assert filtered["stats"]["edge_count"] == 2
