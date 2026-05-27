from select_minimal_connected_evidence_cover import (
    binding_consistent,
    matches_connected,
    select_minimal_connected_cover,
)


def test_binding_consistent_requires_variable_overlap():
    left = {"variable_bindings": {"x1": ["alice", "bob"]}}
    right = {"variable_bindings": {"x1": ["bob", "carol"]}}
    ok, bindings = binding_consistent([left, right])

    assert ok
    assert bindings == {"x1": ["bob"]}

    bad, _ = binding_consistent([left, {"variable_bindings": {"x1": ["dora"]}}])
    assert not bad


def test_matches_connected_by_shared_binding_or_doc():
    assert matches_connected(
        [
            {"doc_index": 1, "variable_bindings": {"x1": ["alice"]}},
            {"doc_index": 2, "mention_surfaces": ["Alice"]},
        ]
    )
    assert matches_connected([{"doc_index": 1}, {"doc_index": 1}])
    assert not matches_connected([{"doc_index": 1, "mention_surfaces": ["Alice"]}, {"doc_index": 2, "mention_surfaces": ["Bob"]}])


def test_select_minimal_connected_cover_prefers_full_connected_cover():
    obligations = [{"obligation_id": "o1"}, {"obligation_id": "o2"}]
    candidates = {
        "o1": [
            {
                "obligation_id": "o1",
                "doc_index": 0,
                "mention_surfaces": ["Film A", "Alice"],
                "variable_bindings": {"x1": ["alice"]},
                "source_rank": 0,
            }
        ],
        "o2": [
            {
                "obligation_id": "o2",
                "doc_index": 1,
                "mention_surfaces": ["Alice", "Paris"],
                "variable_bindings": {"x1": ["alice"]},
                "source_rank": 1,
            }
        ],
    }

    cover = select_minimal_connected_cover(
        obligations=obligations,
        candidates_by_obligation=candidates,
        candidate_doc_indices=[9, 0, 1],
        evidence_set_size=5,
    )

    assert cover["full_cover_feasible"]
    assert cover["binding_consistent"]
    assert cover["connected"]
    assert cover["cover_doc_indices"] == [0, 1]
    assert cover["selected_doc_indices"][:3] == [0, 1, 9]


def test_select_minimal_connected_cover_rejects_inconsistent_full_cover():
    obligations = [{"obligation_id": "o1"}, {"obligation_id": "o2"}]
    candidates = {
        "o1": [{"obligation_id": "o1", "doc_index": 0, "variable_bindings": {"x1": ["alice"]}}],
        "o2": [{"obligation_id": "o2", "doc_index": 1, "variable_bindings": {"x1": ["bob"]}}],
    }

    cover = select_minimal_connected_cover(
        obligations=obligations,
        candidates_by_obligation=candidates,
        candidate_doc_indices=[0, 1],
        evidence_set_size=5,
    )

    assert not cover["full_cover_feasible"]
    assert cover["covered_obligation_count"] < 2


def test_select_minimal_connected_cover_counts_obligations_when_no_candidates():
    cover = select_minimal_connected_cover(
        obligations=[{"obligation_id": "o1"}, {"obligation_id": "o2"}],
        candidates_by_obligation={},
        candidate_doc_indices=[4, 5],
        evidence_set_size=5,
    )

    assert cover["covered_obligation_count"] == 0
    assert cover["total_obligation_count"] == 2
    assert cover["selected_doc_indices"] == [4, 5]
