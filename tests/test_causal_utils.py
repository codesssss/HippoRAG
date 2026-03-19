import numpy as np

from src.hipporag.utils.causal_utils import (
    derive_composed_structure_edges,
    derive_directed_structure_edge,
    route_query_type,
    run_personalized_pagerank,
    sanitize_causal_relations,
    score_candidate_docs_by_structure,
)
from src.hipporag.utils.misc_utils import compute_fact_id


def test_compute_fact_id_is_stable_for_normalized_triples():
    triple_a = ["Heavy Rain", "Caused", "Flooding"]
    triple_b = ["heavy rain", "caused", "flooding"]
    assert compute_fact_id(triple_a) == compute_fact_id(triple_b)


def test_sanitize_causal_relations_filters_invalid_and_dedups():
    valid_fact_ids = {"fact-a", "fact-b"}
    raw_relations = [
        {"source_fact_id": "fact-a", "target_fact_id": "fact-b", "relation_type": "caused", "confidence": 0.4},
        {"source_fact_id": "fact-a", "target_fact_id": "fact-b", "relation_type": "caused", "confidence": 0.9},
        {"source_fact_id": "fact-a", "target_fact_id": "fact-z", "relation_type": "prevents", "confidence": 0.8},
    ]

    relations = sanitize_causal_relations(raw_relations, valid_fact_ids, confidence_threshold=0.5)

    assert len(relations) == 1
    assert relations[0].relation_type == "causes"
    assert relations[0].confidence == 0.9


def test_route_query_type_distinguishes_causal_queries():
    assert route_query_type("Why did the bridge collapse?") == "cause"
    assert route_query_type("What happens if the pressure rises?") == "effect"
    assert route_query_type("What was caused by the storm?") == "effect"
    assert route_query_type("What leads to flooding downtown?") == "cause"
    assert route_query_type("What does smoking cause?") == "effect"
    assert route_query_type("How can vaccines prevent infection?") == "prevention"
    assert route_query_type("Who directed the film?") == "non_causal"


def test_run_personalized_pagerank_prefers_reverse_chain_when_seeded_downstream():
    adjacency = {
        1: [(0, 1.0)],
        2: [(1, 1.0)],
    }
    reset = np.array([0.0, 0.0, 1.0], dtype=float)

    scores = run_personalized_pagerank(num_nodes=3, adjacency=adjacency, reset_prob=reset, damping=0.7)

    assert scores[1] > scores[0]
    assert scores[1] > 0


def test_derive_directed_structure_edge_handles_forward_and_reverse_predicates():
    assert derive_directed_structure_edge("Heavy Rain", "caused", "Flooding") == (
        "heavy rain",
        "flooding",
        "causes",
        1.0,
    )
    assert derive_directed_structure_edge("Flooding", "caused by", "Heavy Rain") == (
        "heavy rain",
        "flooding",
        "causes",
        1.0,
    )
    assert derive_directed_structure_edge("Infection", "prevented by", "Vaccine") == (
        "vaccine",
        "infection",
        "prevents",
        0.95,
    )
    assert derive_directed_structure_edge("Saint Petersburg", "renamed from", "Leningrad") == (
        "leningrad",
        "saint petersburg",
        "state_transition",
        0.8,
    )


def test_derive_composed_structure_edges_builds_bridge_edge_from_fact_chain():
    composed_edges = derive_composed_structure_edges(
        source_triple=("Smoking", "causes", "Cancer"),
        target_triple=("Cancer", "causes", "Death"),
        relation_type="causes",
        confidence=0.9,
    )

    assert ("smoking", "death", "causes", 0.9) in composed_edges


def test_score_candidate_docs_by_structure_prefers_bridge_doc():
    candidate_doc_ids = [0, 1, 2]
    doc_idx_to_entities = {
        0: {"heavy rain", "flooding"},
        1: {"flooding", "road closure"},
        2: {"sunshine", "picnic"},
    }
    doc_idx_to_edges = {
        0: [("heavy rain", "flooding", 1.0, "causes")],
        1: [("flooding", "road closure", 0.9, "causes")],
        2: [],
    }
    adjacency = {
        "heavy rain": [("flooding", 1.0, "causes")],
        "flooding": [("road closure", 0.9, "causes")],
    }

    scores = score_candidate_docs_by_structure(
        candidate_doc_ids=candidate_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        seed_entities={"heavy rain"},
        adjacency=adjacency,
        max_hops=2,
    )

    assert 1 in scores and scores[1] > 0.0
    assert 0 in scores and scores[0] > 0.0
    assert 2 not in scores or scores[2] == 0.0


def test_score_candidate_docs_by_structure_requires_explicit_bridge_edge():
    candidate_doc_ids = [0]
    doc_idx_to_entities = {
        0: {"heavy rain", "flooding"},
    }
    doc_idx_to_edges = {
        0: [("heavy rain", "shelter demand", 0.9, "causes")],
    }
    adjacency = {
        "heavy rain": [("flooding", 1.0, "causes")],
    }

    scores = score_candidate_docs_by_structure(
        candidate_doc_ids=candidate_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        seed_entities={"heavy rain"},
        adjacency=adjacency,
        max_hops=2,
    )

    assert scores == {}


if __name__ == "__main__":
    test_compute_fact_id_is_stable_for_normalized_triples()
    test_sanitize_causal_relations_filters_invalid_and_dedups()
    test_route_query_type_distinguishes_causal_queries()
    test_run_personalized_pagerank_prefers_reverse_chain_when_seeded_downstream()
    test_derive_directed_structure_edge_handles_forward_and_reverse_predicates()
    test_derive_composed_structure_edges_builds_bridge_edge_from_fact_chain()
    test_score_candidate_docs_by_structure_prefers_bridge_doc()
    test_score_candidate_docs_by_structure_requires_explicit_bridge_edge()
