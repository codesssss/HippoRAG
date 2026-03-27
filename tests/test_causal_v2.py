import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from src.hipporag.causal_v2 import CausalV2Engine


class DummyEmbeddingModel:
    def batch_encode(self, texts, instruction=None, norm=True):
        del instruction
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            lowered = str(text).lower()
            vector = np.zeros(6, dtype=float)
            if any(token in lowered for token in ["why", "because", "cause", "caused"]):
                vector[0] = 1.0
            if any(token in lowered for token in ["what happens", "happens", "effect", "result", "after"]):
                vector[1] = 1.0
            if any(token in lowered for token in ["prevent", "prevention", "avoid", "stop"]):
                vector[2] = 1.0
            if any(token in lowered for token in ["film", "director", "capital", "where is"]):
                vector[3] = 1.0
            if any(token in lowered for token in ["rate", "rates", "interest", "borrowing"]):
                vector[4] = 1.0
            if any(token in lowered for token in ["rain", "flood", "closure"]):
                vector[5] = 1.0
            if not vector.any():
                vector[3] = 1.0
            if norm:
                norm_value = np.linalg.norm(vector)
                if norm_value > 0:
                    vector = vector / norm_value
            vectors.append(vector)
        return np.asarray(vectors, dtype=float)


class DummyLLM:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def infer(self, messages, **kwargs):
        self.calls.append({
            "messages": messages,
            "kwargs": kwargs,
        })
        return self.response_text, {"finish_reason": "stop"}, False


class DummyChunkStore:
    def __init__(self, rows=None):
        self.rows = rows or {}

    def get_row(self, hash_id):
        return self.rows[str(hash_id)]


def make_dummy_config(**overrides):
    values = {
        "embedding_batch_size": 2,
        "force_index_from_scratch": False,
        "causal_v2_probe_mode": "router",
        "causal_v2_graph_mode": "causal",
        "general_graph_related_to_weight": 0.3,
        "general_graph_seed_top_k": 10,
        "causal_v2_extraction_max_tokens": 128,
        "causal_v2_extraction_retry_attempts": 1,
        "causal_v2_extraction_workers": 1,
        "causal_event_top_k": 2,
        "causal_v2_max_hops": 2,
        "causal_chain_top_k": 4,
        "causal_context_max_items": 0,
        "causal_er_similarity_threshold": 0.8,
        "causal_er_text_threshold": 0.4,
        "causal_v2_min_edge_confidence": 0.7,
        "passage_node_weight": 0.05,
        "damping": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_dummy_engine(tmp_dir: str, llm_model=None, **config_overrides):
    config = make_dummy_config(**config_overrides)
    hipporag = SimpleNamespace(
        global_config=config,
        working_dir=tmp_dir,
        embedding_model=DummyEmbeddingModel(),
        llm_model=llm_model or DummyLLM('{"events": [], "causal_edges": []}'),
        chunk_embedding_store=DummyChunkStore(),
    )
    return CausalV2Engine(hipporag)


def test_route_query_uses_rule_router_labels():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(tmp_dir)
        effect_info = engine.route_query(
            query="What happens if heavy rain continues?",
            query_embedding=np.asarray([1.0, 0.0, 0.0], dtype=float),
        )
        standard_info = engine.route_query(
            query="Who directed the film?",
            query_embedding=np.asarray([0.0, 1.0, 0.0], dtype=float),
        )

    assert effect_info["label"] == "effect"
    assert effect_info["is_causal"] is True
    assert effect_info["route_source"] == "rule_router"
    assert standard_info["label"] == "standard"
    assert standard_info["is_causal"] is False
    assert standard_info["route_source"] == "rule_router"


def test_route_query_general_mode_stays_on_standard_trace():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(tmp_dir, causal_v2_graph_mode="general")
        route_info = engine.route_query(
            query="Why did the bridge collapse?",
            query_embedding=np.asarray([1.0, 0.0, 0.0], dtype=float),
        )

    assert route_info["label"] == "standard"
    assert route_info["is_causal"] is False
    assert route_info["best_causal_label"] == "general"
    assert route_info["route_source"] == "general_graph_default"


def test_validate_extraction_payload_filters_invalid_events_and_edges():
    payload = {
        "events": [
            {"event_id": "e1", "text": "rates were cut", "quote": "the central bank cut rates"},
            {"event_id": "e1", "text": "duplicate", "quote": "duplicate quote"},
            {"event_id": "e2", "text": "borrowing increased", "quote": "borrowing increased"},
            {"event_id": "", "text": "missing id", "quote": "bad"},
        ],
        "causal_edges": [
            {
                "source_event_id": "e1",
                "target_event_id": "e2",
                "relation_type": "causes",
                "confidence": 0.9,
                "quote": "borrowing increased after the cut",
            },
            {
                "source_event_id": "e1",
                "target_event_id": "e9",
                "relation_type": "cause",
                "confidence": 0.8,
                "quote": "unknown target",
            },
            {
                "source_event_id": "e2",
                "target_event_id": "e2",
                "relation_type": "prevent",
                "confidence": 0.7,
                "quote": "self loop",
            },
            {
                "source_event_id": "e1",
                "target_event_id": "e2",
                "relation_type": "correlates",
                "confidence": 0.7,
                "quote": "invalid relation",
            },
        ],
    }

    events, edges = CausalV2Engine._validate_extraction_payload(payload)

    assert [event["event_id"] for event in events] == ["e1", "e2"]
    assert len(edges) == 1
    assert edges[0]["relation_type"] == "cause"
    assert edges[0]["confidence"] == 0.9


def test_validate_extraction_payload_general_mode_canonicalizes_relations():
    payload = {
        "entities": [
            {"id": "n1", "text": "John Wallop", "type": "person", "quote": "John Wallop"},
            {"id": "n2", "text": "Coulson Wallop", "type": "person", "quote": "Coulson Wallop"},
            {"id": "n3", "text": "20 November 1851", "type": "date", "quote": "20 November 1851"},
        ],
        "relations": [
            {
                "source": "n1",
                "target": "n2",
                "type": "father",
                "confidence": 0.8,
                "quote": "Coulson Wallop was the eldest son of John Wallop",
            },
            {
                "source": "n2",
                "target": "n3",
                "type": "date of birth",
                "confidence": 0.9,
                "quote": "Coulson Wallop (20 November 1851)",
            },
        ],
    }

    events, edges = CausalV2Engine._validate_extraction_payload(payload, graph_mode="general")

    assert [event["event_id"] for event in events] == ["n1", "n2", "n3"]
    assert [event["entity_type"] for event in events] == ["person", "person", "date"]
    assert [edge["relation_type"] for edge in edges] == ["parent_of", "born_on"]


def test_extract_single_chunk_uses_max_completion_tokens_and_parses_output():
    response_text = json.dumps({
        "events": [
            {"event_id": "e1", "text": "rates were cut", "quote": "the central bank cut interest rates"},
            {"event_id": "e2", "text": "borrowing increased", "quote": "borrowing increased after the cut"},
        ],
        "causal_edges": [
            {
                "source_event_id": "e1",
                "target_event_id": "e2",
                "relation_type": "cause",
                "confidence": 0.9,
                "quote": "borrowing increased after the cut",
            }
        ],
    })
    llm_model = DummyLLM(response_text)
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(tmp_dir, llm_model=llm_model)
        extracted = engine._extract_single_chunk(
            chunk_id="chunk-1",
            content="Economy\nThe central bank cut interest rates, and borrowing increased after the cut.",
        )

    assert extracted.parse_error is None
    assert len(extracted.events) == 2
    assert len(extracted.edges) == 1
    assert "max_completion_tokens" in llm_model.calls[0]["kwargs"]
    assert "max_tokens" not in llm_model.calls[0]["kwargs"]


def test_resolve_events_merges_similar_surface_forms():
    surface_events = [
        {
            "temp_event_id": "chunk-a:e1",
            "text": "interest rate cut",
            "quote": "interest rates were cut",
            "chunk_id": "chunk-a",
            "doc_title": "Doc A",
        },
        {
            "temp_event_id": "chunk-b:e1",
            "text": "cut in interest rates",
            "quote": "after the cut in interest rates",
            "chunk_id": "chunk-b",
            "doc_title": "Doc B",
        },
    ]
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(tmp_dir)
        temp_to_event_id, event_nodes = engine._resolve_events(surface_events)

    merged_event_ids = set(temp_to_event_id.values())
    assert len(merged_event_ids) == 1
    merged_event = next(iter(event_nodes.values()))
    assert set(merged_event["chunk_ids"]) == {"chunk-a", "chunk-b"}
    assert "interest rate cut" in merged_event["surface_forms"]
    assert "cut in interest rates" in merged_event["surface_forms"]


def test_retrieve_subgraph_serializes_directed_causal_chain():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(tmp_dir)
        event_a = "event-a"
        event_b = "event-b"
        event_c = "event-c"
        edge_ab = {
            "edge_id": "edge-ab",
            "source_event_id": event_a,
            "target_event_id": event_b,
            "relation_type": "cause",
            "confidence": 0.9,
            "quotes": ["heavy rain caused flooding"],
            "chunk_ids": ["chunk-1"],
            "doc_titles": ["Weather"],
        }
        edge_bc = {
            "edge_id": "edge-bc",
            "source_event_id": event_b,
            "target_event_id": event_c,
            "relation_type": "enable",
            "confidence": 0.8,
            "quotes": ["flooding led to road closures"],
            "chunk_ids": ["chunk-2"],
            "doc_titles": ["Traffic"],
        }
        engine.loaded = True
        engine.event_nodes = {
            event_a: {"canonical_text": "heavy rain"},
            event_b: {"canonical_text": "flooding"},
            event_c: {"canonical_text": "road closures"},
        }
        engine.event_ids = [event_a, event_b, event_c]
        engine.event_embeddings = np.asarray([
            [1.0, 0.0, 0.0],
            [0.7, 0.3, 0.0],
            [0.2, 0.8, 0.0],
        ], dtype=float)
        engine.adjacency_out = {
            event_a: [edge_ab],
            event_b: [edge_bc],
        }
        engine.adjacency_in = {
            event_b: [edge_ab],
            event_c: [edge_bc],
        }

        result = engine.retrieve_subgraph(
            query="What happens after heavy rain?",
            query_embedding=np.asarray([1.0, 0.0, 0.0], dtype=float),
            route_info={"is_causal": True, "label": "effect"},
        )

    assert result["seed_event_ids"][0] == event_a
    assert result["serialized_contexts"]
    assert "heavy rain" in result["serialized_contexts"][0]
    assert "causes" in result["serialized_contexts"][0]
    assert result["causal_context_doc_ids"] == ["chunk-1", "chunk-2"]
    assert result["probe_attempted"] is True
    assert result["probe_forced"] is False
    assert result["probe_route_label"] == "effect"


def test_retrieve_subgraph_always_probe_uses_best_causal_label_for_standard_query():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(tmp_dir, causal_v2_probe_mode="always")
        event_a = "event-a"
        event_b = "event-b"
        edge_ab = {
            "edge_id": "edge-ab",
            "source_event_id": event_a,
            "target_event_id": event_b,
            "relation_type": "cause",
            "confidence": 0.9,
            "quotes": ["heavy rain caused flooding"],
            "chunk_ids": ["chunk-1"],
            "doc_titles": ["Weather"],
        }
        engine.loaded = True
        engine.event_nodes = {
            event_a: {"canonical_text": "heavy rain"},
            event_b: {"canonical_text": "flooding"},
        }
        engine.event_ids = [event_a, event_b]
        engine.event_embeddings = np.asarray([
            [1.0, 0.0, 0.0],
            [0.7, 0.3, 0.0],
        ], dtype=float)
        engine.adjacency_out = {
            event_a: [edge_ab],
        }
        engine.adjacency_in = {
            event_b: [edge_ab],
        }

        result = engine.retrieve_subgraph(
            query="Who directed the film?",
            query_embedding=np.asarray([1.0, 0.0, 0.0], dtype=float),
            route_info={"is_causal": False, "label": "standard", "best_causal_label": "effect"},
        )

    assert result["probe_attempted"] is True
    assert result["probe_forced"] is True
    assert result["probe_route_label"] == "effect"
    assert result["serialized_contexts"]


def test_extract_query_entities_keeps_title_case_mentions():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(tmp_dir)
        query_entities = engine._extract_query_entities(
            "Where did Coulson Wallop's father study before moving to Oxford?"
        )

    assert "Coulson Wallop" in query_entities
    assert "Wallop" in query_entities
    assert "Oxford" in query_entities


def test_build_retrieval_igraph_general_mode_builds_weighted_entity_and_passage_edges():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(tmp_dir, causal_v2_graph_mode="general", general_graph_related_to_weight=0.3)
        with open(engine.manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"graph_mode": "general"}, handle)

        engine.loaded = True
        engine.event_nodes = {
            "e1": {"canonical_text": "Entity One", "chunk_ids": ["chunk-1", "chunk-2"]},
            "e2": {"canonical_text": "Entity Two", "chunk_ids": ["chunk-1"]},
            "e3": {"canonical_text": "Entity Three", "chunk_ids": ["chunk-2"]},
        }
        engine.edges = {
            "edge-12": {
                "edge_id": "edge-12",
                "source_event_id": "e1",
                "target_event_id": "e2",
                "relation_type": "parent_of",
                "confidence": 0.8,
            },
            "edge-23": {
                "edge_id": "edge-23",
                "source_event_id": "e2",
                "target_event_id": "e3",
                "relation_type": "related_to",
                "confidence": 0.6,
            },
            "edge-13": {
                "edge_id": "edge-13",
                "source_event_id": "e1",
                "target_event_id": "e3",
                "relation_type": "born_in",
                "confidence": 0.5,
            },
        }

        graph, entity_map, passage_map, passage_idxs = engine.build_retrieval_igraph(["chunk-1", "chunk-2"])

    assert graph.vcount() == 5
    assert graph.ecount() == 7
    assert entity_map == {"e1": 0, "e2": 1, "e3": 2}
    assert passage_map == {"chunk-1": 3, "chunk-2": 4}
    assert passage_idxs == [3, 4]

    typed_weight = graph.es[graph.get_eid(entity_map["e1"], entity_map["e2"])]["weight"]
    related_weight = graph.es[graph.get_eid(entity_map["e2"], entity_map["e3"])]["weight"]
    anti_hub_weight = graph.es[graph.get_eid(entity_map["e1"], passage_map["chunk-1"])]["weight"]
    singleton_weight = graph.es[graph.get_eid(entity_map["e2"], passage_map["chunk-1"])]["weight"]

    # Patch B: entity degrees = edge_count + chunk_count
    # e1: 2 edges + 2 chunks = 4, e2: 2 edges + 1 chunk = 3, e3: 2 edges + 1 chunk = 3
    # hub_decay(s,t) = 1/sqrt(log2(deg_s+1)*log2(deg_t+1))
    decay_e1_e2 = 1.0 / np.sqrt(np.log2(5) * np.log2(4))  # deg 4, 3
    decay_e2_e3 = 1.0 / np.sqrt(np.log2(4) * np.log2(4))  # deg 3, 3
    assert np.isclose(typed_weight, 0.8 * decay_e1_e2)      # parent_of, confidence=0.8
    assert np.isclose(related_weight, 0.6 * 0.3 * decay_e2_e3)  # related_to, confidence=0.6
    assert np.isclose(anti_hub_weight, 1.0 / np.log2(3))    # entity-passage unchanged
    assert np.isclose(singleton_weight, 1.0)                 # single-chunk entity


def _run_all_tests() -> None:
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()


if __name__ == "__main__":
    _run_all_tests()
