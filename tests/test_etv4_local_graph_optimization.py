from collections import Counter

from evidence_transition_graphragv4_fact_witnessed_sto.agsto.local_graph import (
    _doc_query_token_coverage_map,
    _induced_local_edges,
)
from evidence_transition_graphragv4_fact_witnessed_sto.agsto.role_transition import (
    ROLE_BRIDGE,
    SAME_OBJECT,
    SENTENCE_GROUNDED_TRANSITION,
    UNKNOWN_EDGE_TIER,
)


def _legacy_induced_local_edges(role_graph, admitted_docs):
    admitted = {int(doc_idx) for doc_idx in admitted_docs if int(doc_idx) >= 0}
    rows = []
    for edge in (role_graph.get("edges", {}) or {}).values():
        left = int(edge.get("left_doc", -1))
        right = int(edge.get("right_doc", -1))
        allowed_kinds = [
            str(kind)
            for kind in edge.get("kinds", []) or []
            if str(kind) in {SENTENCE_GROUNDED_TRANSITION, ROLE_BRIDGE, SAME_OBJECT}
        ]
        if left not in admitted or right not in admitted or not allowed_kinds:
            continue
        rows.append(
            {
                "left_doc": left,
                "right_doc": right,
                "kinds": allowed_kinds,
                "evidence_tiers": list(edge.get("evidence_tiers", []) or []),
                "best_evidence_tier": str(edge.get("best_evidence_tier") or UNKNOWN_EDGE_TIER),
                "endpoints": list(edge.get("endpoints", []) or []),
                "samples": list(edge.get("samples", []) or [])[:5],
            }
        )
    return sorted(rows, key=lambda row: (int(row["left_doc"]), int(row["right_doc"]), tuple(row["kinds"])))


def test_induced_local_edges_adjacency_path_matches_legacy_global_scan():
    edge_a = {
        "left_doc": 1,
        "right_doc": 2,
        "kinds": [SENTENCE_GROUNDED_TRANSITION, "not_allowed"],
        "evidence_tiers": ["evidence_transition"],
        "best_evidence_tier": "evidence_transition",
        "endpoints": ["alpha"],
        "samples": [{"sample": idx} for idx in range(8)],
    }
    edge_b = {
        "left_doc": 2,
        "right_doc": 3,
        "kinds": [ROLE_BRIDGE],
        "evidence_tiers": ["evidence_transition"],
        "best_evidence_tier": "evidence_transition",
        "endpoints": ["beta"],
        "samples": [{"sample": "bridge"}],
    }
    edge_not_admitted = {
        "left_doc": 3,
        "right_doc": 9,
        "kinds": [SAME_OBJECT],
        "endpoints": ["gamma"],
    }
    edge_not_allowed = {
        "left_doc": 1,
        "right_doc": 3,
        "kinds": ["not_allowed"],
        "endpoints": ["delta"],
    }
    role_graph = {
        "edges": {
            (1, 2): edge_a,
            (2, 3): edge_b,
            (3, 9): edge_not_admitted,
            (1, 3): edge_not_allowed,
        },
        "adjacency": {
            1: [edge_a, edge_not_allowed],
            2: [edge_a, edge_b],
            3: [edge_b, edge_not_admitted, edge_not_allowed],
            9: [edge_not_admitted],
        },
    }

    optimized = _induced_local_edges(role_graph=role_graph, admitted_docs=[1, 2, 3])
    legacy = _legacy_induced_local_edges(role_graph, [1, 2, 3])

    assert optimized == legacy
    assert optimized[0]["samples"] == [{"sample": idx} for idx in range(5)]


def test_induced_local_edges_falls_back_to_global_scan_without_adjacency():
    edge = {
        "left_doc": 4,
        "right_doc": 5,
        "kinds": [SAME_OBJECT],
        "endpoints": ["shared"],
    }
    role_graph = {"edges": {(4, 5): edge}}

    assert _induced_local_edges(role_graph=role_graph, admitted_docs=[4, 5]) == _legacy_induced_local_edges(
        role_graph,
        [4, 5],
    )


def test_doc_query_token_coverage_uses_inverted_index_equivalently():
    corpus_index = {
        "doc_token_counts": {
            1: Counter({"alpha": 2, "beta": 1}),
            2: Counter({"beta": 3, "gamma": 1}),
            3: Counter({"other": 1}),
        },
        "token_to_docs": {
            "alpha": [1],
            "beta": [1, 2],
            "gamma": [2],
            "other": [3],
        },
    }

    assert _doc_query_token_coverage_map(corpus_index=corpus_index, query_tokens={"alpha", "gamma"}) == {
        1: ("alpha",),
        2: ("gamma",),
    }


def test_doc_query_token_coverage_falls_back_without_inverted_index():
    corpus_index = {
        "doc_token_counts": {
            1: Counter({"alpha": 2, "beta": 1}),
            2: Counter({"beta": 3, "gamma": 1}),
        }
    }

    assert _doc_query_token_coverage_map(corpus_index=corpus_index, query_tokens={"beta"}) == {
        1: ("beta",),
        2: ("beta",),
    }
