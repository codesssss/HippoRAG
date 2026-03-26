import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from src.hipporag.causal_v2 import CausalV2Engine, SemanticIntentRouter


class DummyEmbeddingModel:
    def batch_encode(self, texts, instruction=None, norm=True):
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
        "causal_router_anchor_path": None,
        "causal_router_causal_threshold": 0.45,
        "causal_router_standard_threshold": 0.45,
        "causal_router_margin_threshold": 0.01,
        "causal_v2_probe_mode": "router",
        "causal_v2_graph_mode": "causal",
        "causal_context_injection_mode": "all",
        "causal_context_min_chain_score": 0.1,
        "causal_v2_candidate_injection_enabled": False,
        "causal_v2_candidate_injection_top_n": 20,
        "causal_v2_candidate_injection_max_docs": 5,
        "causal_v2_candidate_injection_preserve_top_k": 2,
        "causal_v2_candidate_injection_hops": 1,
        "causal_v2_candidate_injection_blend_weight": 0.15,
        "causal_v2_candidate_injection_require_query_entity_overlap": True,
        "causal_v2_extraction_max_tokens": 128,
        "causal_v2_extraction_retry_attempts": 1,
        "causal_v2_extraction_workers": 1,
        "causal_event_top_k": 2,
        "causal_v2_max_hops": 2,
        "causal_chain_top_k": 4,
        "causal_context_max_items": 4,
        "causal_er_similarity_threshold": 0.8,
        "causal_er_text_threshold": 0.4,
        "causal_v2_min_edge_confidence": 0.7,
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


def test_semantic_intent_router_routes_all_supported_labels():
    with TemporaryDirectory() as tmp_dir:
        anchor_path = Path(tmp_dir) / "anchors.json"
        anchor_path.write_text(json.dumps({
            "cause": ["why did the bridge collapse"],
            "effect": ["what happens after the rate cut"],
            "prevention": ["how can vaccines prevent infection"],
            "standard": ["who directed the film"],
        }))
        config = make_dummy_config(causal_router_anchor_path=str(anchor_path))
        model = DummyEmbeddingModel()
        router = SemanticIntentRouter(model, config)

        cause_info = router.route(model.batch_encode(["Why did the bridge collapse?"])[0])
        effect_info = router.route(model.batch_encode(["What happens after the rate cut?"])[0])
        prevention_info = router.route(model.batch_encode(["How can vaccines prevent infection?"])[0])
        standard_info = router.route(model.batch_encode(["Who directed the film?"])[0])

        assert cause_info["label"] == "cause" and cause_info["is_causal"] is True
        assert effect_info["label"] == "effect" and effect_info["is_causal"] is True
        assert prevention_info["label"] == "prevention" and prevention_info["is_causal"] is True
        assert standard_info["label"] == "standard" and standard_info["is_causal"] is False


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


def test_selective_injection_keeps_only_top_overlap_chain():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(
            tmp_dir,
            causal_context_injection_mode="selective",
            causal_context_min_chain_score=0.1,
        )
        chains = [
            {
                "score": 0.6,
                "serialized": '[Other Movie] "x" shows that `other movie event` causes `other effect`.',
                "chunk_ids": ["chunk-1"],
            },
            {
                "score": 0.4,
                "serialized": '[Wallop Family] "x" shows that `John Wallop was born` causes `John Wallop inherited electoral interests`.',
                "chunk_ids": ["chunk-2"],
            },
            {
                "score": 0.2,
                "serialized": '[Wallop Family] "x" shows that `Wallop attended Oxford` enables `received degree`.',
                "chunk_ids": ["chunk-3"],
            },
        ]

        selected_chains, query_entities, trace = engine._select_chains_for_injection(
            query="Where did Coulson Wallop's father study?",
            chains=chains,
        )

    assert "Wallop" in query_entities
    assert len(selected_chains) == 1
    assert selected_chains[0]["chunk_ids"] == ["chunk-2"]
    assert trace["entity_overlap_chain_count"] == 2
    assert trace["selected_chain_scores"] == [0.4]


def test_selective_injection_drops_low_score_overlap_chain():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(
            tmp_dir,
            causal_context_injection_mode="selective",
            causal_context_min_chain_score=0.5,
        )
        chains = [
            {
                "score": 0.3,
                "serialized": '[Wallop Family] "x" shows that `John Wallop was born` causes `John Wallop inherited electoral interests`.',
                "chunk_ids": ["chunk-2"],
            },
        ]

        selected_chains, query_entities, trace = engine._select_chains_for_injection(
            query="Where did Coulson Wallop's father study?",
            chains=chains,
        )

    assert "Wallop" in query_entities
    assert selected_chains == []
    assert trace["entity_overlap_chain_count"] == 1
    assert trace["selected_chain_scores"] == []


def test_propose_candidate_doc_injections_general_mode_discovers_bridge_doc():
    with TemporaryDirectory() as tmp_dir:
        engine = make_dummy_engine(
            tmp_dir,
            causal_v2_graph_mode="general",
            causal_v2_candidate_injection_top_n=2,
            causal_v2_candidate_injection_max_docs=2,
            causal_v2_candidate_injection_hops=1,
        )
        event_a = "event-a"
        event_b = "event-b"
        event_c = "event-c"
        edge_ac = {
            "edge_id": "edge-ac",
            "source_event_id": event_a,
            "target_event_id": event_c,
            "relation_type": "parent_of",
            "confidence": 0.95,
            "quotes": ["Coulson Wallop was the eldest son of John Wallop"],
            "chunk_ids": ["chunk-c"],
            "doc_titles": ["Wallop Family"],
        }

        engine.loaded = True
        engine.event_nodes = {
            event_a: {"canonical_text": "Coulson Wallop", "chunk_ids": ["chunk-a"]},
            event_b: {"canonical_text": "Wizards of the Lost Kingdom", "chunk_ids": ["chunk-b"]},
            event_c: {"canonical_text": "John Wallop", "chunk_ids": ["chunk-c"]},
        }
        engine.event_ids = [event_a, event_b, event_c]
        engine.chunk_to_event_ids = {
            "chunk-a": [event_a],
            "chunk-b": [event_b],
            "chunk-c": [event_c],
        }
        engine.adjacency_out = {
            event_a: [edge_ac],
        }
        engine.adjacency_in = {
            event_c: [edge_ac],
        }
        engine.hipporag.passage_node_keys = ["chunk-a", "chunk-b", "chunk-c", "chunk-d"]
        engine.hipporag.passage_node_key_to_doc_idx = {
            "chunk-a": 0,
            "chunk-b": 1,
            "chunk-c": 2,
            "chunk-d": 3,
        }
        engine.hipporag.chunk_embedding_store.rows = {
            "chunk-a": {"content": "Coulson Wallop was the eldest son of John Wallop."},
            "chunk-b": {"content": "Wizards of the Lost Kingdom is a fantasy film."},
            "chunk-c": {"content": "John Wallop was educated at Oxford."},
            "chunk-d": {"content": "Final Exam is an American slasher film."},
        }

        trace = engine.propose_candidate_doc_injections(
            dense_sorted_doc_ids=np.asarray([0, 1, 3, 2], dtype=int),
            dense_sorted_doc_scores=np.asarray([1.0, 0.9, 0.8, 0.7], dtype=float),
            query_entities=["Coulson Wallop"],
        )

    assert trace["noop_reason"] is None
    assert trace["seed_top_n"] == 2
    assert trace["candidate_seed_chunk_count"] == 2
    assert trace["query_entity_seed_chunk_count"] == 1
    assert trace["seed_node_count"] == 1
    assert trace["proposed_doc_count"] >= 1
    assert trace["injected_doc_ids"] == [2]
    assert trace["injected_chunk_ids"] == ["chunk-c"]


if __name__ == "__main__":
    test_semantic_intent_router_routes_all_supported_labels()
    test_validate_extraction_payload_filters_invalid_events_and_edges()
    test_validate_extraction_payload_general_mode_canonicalizes_relations()
    test_extract_single_chunk_uses_max_completion_tokens_and_parses_output()
    test_resolve_events_merges_similar_surface_forms()
    test_retrieve_subgraph_serializes_directed_causal_chain()
    test_retrieve_subgraph_always_probe_uses_best_causal_label_for_standard_query()
    test_selective_injection_keeps_only_top_overlap_chain()
    test_selective_injection_drops_low_score_overlap_chain()
    test_propose_candidate_doc_injections_general_mode_discovers_bridge_doc()
