from types import MethodType, SimpleNamespace

import numpy as np

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.causal_utils import (
    derive_composed_structure_edges,
    derive_directed_structure_edge,
    route_query_type,
    run_personalized_pagerank,
    sanitize_causal_relations,
    score_candidate_docs_by_structure,
    score_query_causal_intent,
)
from src.hipporag.utils.misc_utils import compute_fact_id


def make_dummy_hipporag(**config_overrides):
    config = {
        "causal_blend_dense_weight": 0.35,
        "causal_blend_fact_weight": 0.15,
        "causal_blend_graph_weight": 0.50,
        "causal_gate_mode": "soft",
        "causal_margin_gate_enabled": False,
        "causal_margin_threshold": 0.02,
        "causal_blend_top_k": 0,
    }
    config.update(config_overrides)
    dummy = SimpleNamespace(
        global_config=SimpleNamespace(**config),
        passage_node_key_to_doc_idx={f"doc-{idx}": idx for idx in range(20)},
    )
    dummy._rank_doc_score_map = MethodType(HippoRAG._rank_doc_score_map, dummy)
    dummy._compute_causal_blend_weight = MethodType(HippoRAG._compute_causal_blend_weight, dummy)
    return dummy


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
    assert route_query_type("In what month was the flight born with a stop in New York?") == "non_causal"
    assert route_query_type("Who directed the film?") == "non_causal"


def test_score_query_causal_intent_uses_strong_weak_and_non_causal_buckets():
    assert score_query_causal_intent("Why did the bridge collapse?") == 1.0
    assert score_query_causal_intent("What happened after the CEO resigned?") == 0.6
    assert score_query_causal_intent("Who is the director of the film?") == 0.0
    assert score_query_causal_intent("Name the film starring actor X.") == 0.1


def test_blend_causal_retrieval_scores_applies_soft_gate_scaling():
    dummy = make_dummy_hipporag()
    _, _, trace = HippoRAG.blend_causal_retrieval_scores(
        dummy,
        baseline_doc_scores={0: 1.0},
        causal_doc_scores={1: 1.0},
        route_causal_intent_score=0.6,
    )

    assert trace["causal_weight_before_attenuation"] == 0.3
    assert trace["causal_weight_after_attenuation"] == 0.06


def test_blend_causal_retrieval_scores_respects_margin_gate():
    dummy = make_dummy_hipporag(causal_margin_gate_enabled=True, causal_margin_threshold=0.2)
    sorted_doc_ids, _, trace = HippoRAG.blend_causal_retrieval_scores(
        dummy,
        baseline_doc_scores={0: 1.0, 1: 0.2},
        causal_doc_scores={2: 1.0},
        route_causal_intent_score=1.0,
    )

    assert trace["margin_gate_applied"] is True
    assert trace["use_causal_for_blend"] is False
    assert trace["causal_docs_used_for_blend_count"] == 0
    assert trace["causal_weight_after_attenuation"] == 0.0
    assert sorted_doc_ids.tolist()[:2] == [0, 1]


def test_blend_causal_retrieval_scores_limits_causal_docs_to_top_k():
    dummy = make_dummy_hipporag(causal_blend_top_k=2)
    sorted_doc_ids, _, trace = HippoRAG.blend_causal_retrieval_scores(
        dummy,
        baseline_doc_scores={0: 1.0},
        causal_doc_scores={10: 0.9, 11: 0.8, 12: 0.7},
        route_causal_intent_score=1.0,
    )

    assert trace["causal_docs_used_for_blend_count"] == 2
    assert trace["causal_docs_used_for_blend"] == [10, 11]
    assert 12 not in sorted_doc_ids.tolist()


def test_run_personalized_pagerank_prefers_reverse_chain_when_seeded_downstream():
    adjacency = {
        1: [(0, 1.0)],
        2: [(1, 1.0)],
    }
    reset = np.array([0.0, 0.0, 1.0], dtype=float)

    scores = run_personalized_pagerank(num_nodes=3, adjacency=adjacency, reset_prob=reset, damping=0.7)

    assert scores[1] > scores[0]
    assert scores[1] > 0


def test_graph_search_with_causal_facts_soft_non_causal_uses_weak_bidirectional_path():
    dummy = SimpleNamespace(
        global_config=SimpleNamespace(
            causal_seed_top_k=2,
            causal_damping=0.7,
            causal_gate_mode="soft",
        ),
        fact_node_keys=["fact-a", "fact-b"],
        passage_node_keys=["doc-a", "doc-b"],
        causal_graph_out={0: [(1, 1.0, "causes")]},
        causal_graph_in={1: [(0, 1.0, "causes")]},
        fact_id_to_doc_idxs={"fact-a": [0], "fact-b": [1]},
    )

    sorted_doc_ids, sorted_doc_scores, trace = HippoRAG.graph_search_with_causal_facts(
        dummy,
        query_fact_scores=np.array([1.0, 0.0], dtype=float),
        query_type="non_causal",
        preferred_fact_indices=[0],
        return_trace=True,
    )

    assert trace["mode"] == "soft_non_causal_weak"
    assert trace["status"] == "ok"
    assert trace["seeds_with_route_edge_count"] == 2
    assert sorted_doc_ids.tolist()[0] == 0
    assert 1 in sorted_doc_ids.tolist()
    assert sorted_doc_scores[0] > 0


def test_graph_search_with_causal_facts_hard_non_causal_still_returns_empty():
    dummy = SimpleNamespace(
        global_config=SimpleNamespace(
            causal_seed_top_k=2,
            causal_damping=0.7,
            causal_gate_mode="hard",
        ),
        fact_node_keys=["fact-a", "fact-b"],
        passage_node_keys=["doc-a", "doc-b"],
        causal_graph_out={0: [(1, 1.0, "causes")]},
        causal_graph_in={1: [(0, 1.0, "causes")]},
        fact_id_to_doc_idxs={"fact-a": [0], "fact-b": [1]},
    )

    sorted_doc_ids, sorted_doc_scores, trace = HippoRAG.graph_search_with_causal_facts(
        dummy,
        query_fact_scores=np.array([1.0, 0.0], dtype=float),
        query_type="non_causal",
        preferred_fact_indices=[0],
        return_trace=True,
    )

    assert trace["mode"] is None
    assert trace["status"] == "non_causal_query_type"
    assert sorted_doc_ids.size == 0
    assert sorted_doc_scores.size == 0


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


def test_derive_directed_structure_edge_relation_probe_is_flag_gated():
    assert derive_directed_structure_edge(
        "Southeast Library",
        "designed by",
        "Ralph Rapson",
    ) is None

    assert derive_directed_structure_edge(
        "Southeast Library",
        "designed by",
        "Ralph Rapson",
        relation_probe_mode="q6_factual",
    ) == (
        "ralph rapson",
        "southeast library",
        "factual_attribution",
        0.85,
    )
    assert derive_directed_structure_edge(
        "Riverside Plaza",
        "opened in",
        "Minneapolis",
        relation_probe_mode="q6_factual",
    ) == (
        "riverside plaza",
        "minneapolis",
        "factual_located_in",
        0.8,
    )
    assert derive_directed_structure_edge(
        "Mississippi River",
        "drains into",
        "Gulf of Mexico",
        relation_probe_mode="q6_factual",
    ) == (
        "mississippi river",
        "gulf of mexico",
        "factual_flows_to",
        0.85,
    )
    assert derive_directed_structure_edge(
        "Mississippi River",
        "drains into",
        "Gulf of Mexico",
        relation_probe_mode="general_factual",
    ) == (
        "mississippi river",
        "gulf of mexico",
        "factual_flows_to",
        0.85,
    )


def test_prepare_structure_retrieval_objects_relation_probe_adds_q6_edges():
    docs = [
        {
            "idx": "doc-0",
            "extracted_triples": [
                ["Southeast Library", "designed by", "Ralph Rapson"],
                ["Riverside Plaza", "opened in", "Minneapolis"],
            ],
            "extracted_causal_relations": [],
        },
        {
            "idx": "doc-1",
            "extracted_triples": [
                ["Mississippi River", "drains into", "Gulf of Mexico"],
            ],
            "extracted_causal_relations": [],
        },
    ]

    without_probe = SimpleNamespace(
        global_config=SimpleNamespace(
            structure_rerank_enabled=True,
            causal_confidence_threshold=0.5,
            structure_relation_probe_mode="off",
        ),
        passage_node_key_to_doc_idx={"doc-0": 0, "doc-1": 1},
    )
    HippoRAG._prepare_structure_retrieval_objects(without_probe, docs)
    assert without_probe.doc_idx_to_structure_edges[0] == []
    assert without_probe.doc_idx_to_structure_edges[1] == []
    assert dict(without_probe.structure_graph_out) == {}

    with_probe = SimpleNamespace(
        global_config=SimpleNamespace(
            structure_rerank_enabled=True,
            causal_confidence_threshold=0.5,
            structure_relation_probe_mode="q6_factual",
        ),
        passage_node_key_to_doc_idx={"doc-0": 0, "doc-1": 1},
    )
    HippoRAG._prepare_structure_retrieval_objects(with_probe, docs)

    assert ("ralph rapson", "southeast library", 0.85, "factual_attribution") in with_probe.doc_idx_to_structure_edges[0]
    assert ("riverside plaza", "minneapolis", 0.8, "factual_located_in") in with_probe.doc_idx_to_structure_edges[0]
    assert ("mississippi river", "gulf of mexico", 0.85, "factual_flows_to") in with_probe.doc_idx_to_structure_edges[1]
    assert ("southeast library", 0.85, "factual_attribution") in with_probe.structure_graph_out["ralph rapson"]
    assert ("minneapolis", 0.8, "factual_located_in") in with_probe.structure_graph_out["riverside plaza"]
    assert ("gulf of mexico", 0.85, "factual_flows_to") in with_probe.structure_graph_out["mississippi river"]

    with_general_probe = SimpleNamespace(
        global_config=SimpleNamespace(
            structure_rerank_enabled=True,
            causal_confidence_threshold=0.5,
            structure_relation_probe_mode="general_factual",
        ),
        passage_node_key_to_doc_idx={"doc-0": 0, "doc-1": 1},
    )
    HippoRAG._prepare_structure_retrieval_objects(with_general_probe, docs)

    assert with_general_probe.doc_idx_to_structure_edges[0] == with_probe.doc_idx_to_structure_edges[0]
    assert with_general_probe.doc_idx_to_structure_edges[1] == with_probe.doc_idx_to_structure_edges[1]


def test_prepare_structure_retrieval_objects_continuity_probe_adds_city_state_aliases():
    docs = [
        {
            "idx": "doc-0",
            "extracted_triples": [
                ["Southeast Library", "designed by", "Ralph Rapson"],
            ],
            "extracted_causal_relations": [],
        },
        {
            "idx": "doc-1",
            "extracted_triples": [
                ["Riverside Plaza", "designed by", "Ralph Rapson"],
                ["Riverside Plaza", "opened in", "Minneapolis, Minnesota"],
            ],
            "extracted_causal_relations": [],
        },
        {
            "idx": "doc-2",
            "extracted_triples": [
                ["Minneapolis", "lies on", "Mississippi River"],
            ],
            "extracted_causal_relations": [],
        },
    ]

    without_continuity = SimpleNamespace(
        global_config=SimpleNamespace(
            structure_rerank_enabled=True,
            causal_confidence_threshold=0.5,
            structure_relation_probe_mode="q6_factual",
            structure_continuity_probe_mode="off",
        ),
        passage_node_key_to_doc_idx={"doc-0": 0, "doc-1": 1, "doc-2": 2},
    )
    HippoRAG._prepare_structure_retrieval_objects(without_continuity, docs)
    assert "minneapolis" not in without_continuity.doc_idx_to_structure_entities[1]
    assert ("riverside plaza", "minneapolis", 0.8, "factual_located_in") not in without_continuity.doc_idx_to_structure_edges[1]

    with_continuity = SimpleNamespace(
        global_config=SimpleNamespace(
            structure_rerank_enabled=True,
            causal_confidence_threshold=0.5,
            structure_relation_probe_mode="q6_factual",
            structure_continuity_probe_mode="city_state_alias",
        ),
        passage_node_key_to_doc_idx={"doc-0": 0, "doc-1": 1, "doc-2": 2},
    )
    HippoRAG._prepare_structure_retrieval_objects(with_continuity, docs)

    assert "minneapolis" in with_continuity.doc_idx_to_structure_entities[1]
    assert ("minneapolis", 0.8, "factual_located_in") in with_continuity.structure_graph_out["riverside plaza"]
    assert ("minneapolis minnesota", 1.0, "alias_city_state") in with_continuity.structure_graph_out["minneapolis"]
    assert ("minneapolis", 1.0, "alias_city_state") in with_continuity.structure_graph_out["minneapolis minnesota"]

    with_location_alias = SimpleNamespace(
        global_config=SimpleNamespace(
            structure_rerank_enabled=True,
            causal_confidence_threshold=0.5,
            structure_relation_probe_mode="q6_factual",
            structure_continuity_probe_mode="location_alias",
        ),
        passage_node_key_to_doc_idx={"doc-0": 0, "doc-1": 1, "doc-2": 2},
    )
    HippoRAG._prepare_structure_retrieval_objects(with_location_alias, docs)

    assert with_location_alias.doc_idx_to_structure_entities[1] == with_continuity.doc_idx_to_structure_entities[1]
    assert with_location_alias.structure_graph_out["riverside plaza"] == with_continuity.structure_graph_out["riverside plaza"]


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


def test_score_candidate_docs_by_structure_seed_target_bridge_mode_accepts_seed_target():
    candidate_doc_ids = [0]
    doc_idx_to_entities = {
        0: {"minneapolis", "mississippi river"},
    }
    doc_idx_to_edges = {
        0: [("minneapolis", "mississippi river", 1.0, "lies on")],
    }
    adjacency = {
        "minneapolis minnesota": [("minneapolis", 1.0, "alias")],
    }
    seed_entities = {"minneapolis minnesota", "mississippi river"}

    legacy_scores = score_candidate_docs_by_structure(
        candidate_doc_ids=candidate_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        seed_entities=seed_entities,
        adjacency=adjacency,
        max_hops=2,
        seed_target_bridge_mode="off",
    )
    bridge_scores = score_candidate_docs_by_structure(
        candidate_doc_ids=candidate_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        seed_entities=seed_entities,
        adjacency=adjacency,
        max_hops=2,
        seed_target_bridge_mode="allow_seed_target",
    )

    assert legacy_scores == {}
    assert 0 in bridge_scores
    assert bridge_scores[0] > 0.0


def _run_all_tests() -> None:
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()


if __name__ == "__main__":
    _run_all_tests()
