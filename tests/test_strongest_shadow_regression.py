from pathlib import Path
import sys
import types

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.completion = lambda **kwargs: None
    sys.modules["litellm"] = litellm_stub
if "gritlm" not in sys.modules:
    gritlm_stub = types.ModuleType("gritlm")
    gritlm_stub.GritLM = object
    sys.modules["gritlm"] = gritlm_stub
if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers_stub
if "igraph" not in sys.modules:
    igraph_stub = types.ModuleType("igraph")
    igraph_stub.Graph = object
    sys.modules["igraph"] = igraph_stub
if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = object
    vllm_stub.LLM = object
    sys.modules["vllm"] = vllm_stub
if "outlines" not in sys.modules:
    outlines_stub = types.ModuleType("outlines")
    outlines_generate_stub = types.ModuleType("outlines.generate")
    outlines_generate_stub.json = lambda *args, **kwargs: (lambda prompts, **inner_kwargs: [])
    outlines_models_stub = types.ModuleType("outlines.models")
    outlines_models_stub.Transformers = object
    sys.modules["outlines"] = outlines_stub
    sys.modules["outlines.generate"] = outlines_generate_stub
    sys.modules["outlines.models"] = outlines_models_stub

from eval_causal_qwen3 import apply_setwise_selector
from src.hipporag.utils.misc_utils import QuerySolution


class DummyChunkStore:
    def __init__(self, mapping):
        self.text_to_hash_id = mapping


class DummyHippoRAG:
    def __init__(self):
        self.doc_idx_to_structure_entities = {
            0: {"alpha", "beta"},
            1: {"alpha"},
            2: {"gamma"},
        }
        self.doc_idx_to_structure_edges = {0: [], 1: [], 2: []}
        self.structure_graph_out = {}
        self.passage_embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.7, 0.3],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.query_to_embedding = {
            "passage": {
                "where is alpha": np.asarray([1.0, 0.0], dtype=np.float32),
            }
        }
        self.passage_node_key_to_doc_idx = {"chunk-a": 0, "chunk-b": 1, "chunk-c": 2}
        self.chunk_embedding_store = DummyChunkStore(
            {
                "Doc A\nalpha beta": "chunk-a",
                "Doc B\nalpha witness": "chunk-b",
                "Doc C\ngamma": "chunk-c",
            }
        )

    def _get_passage_query_embeddings(self, queries):
        if isinstance(queries, list):
            for query in queries:
                self.query_to_embedding["passage"].setdefault(
                    str(query), np.asarray([1.0, 0.0], dtype=np.float32)
                )
        else:
            self.query_to_embedding["passage"].setdefault(
                str(queries), np.asarray([1.0, 0.0], dtype=np.float32)
            )


def test_bridge_append_strongest_shadow_does_not_change_baseline_by_default():
    hipporag = DummyHippoRAG()
    query_solution = QuerySolution(
        question="where is alpha",
        docs=["Doc A\nalpha beta", "Doc B\nalpha witness", "Doc C\ngamma"],
        doc_scores=np.asarray([0.9, 0.7, 0.2], dtype=float),
    )

    selected_solutions, _summary = apply_setwise_selector(
        hipporag=hipporag,
        query_solutions=[query_solution],
        doc_text_to_chunk_id={
            "Doc A\nalpha beta": "chunk-a",
            "Doc B\nalpha witness": "chunk-b",
            "Doc C\ngamma": "chunk-c",
        },
        pool_k=3,
        qa_top_k=2,
        selector_name="bridge_append",
        score_mode="bridge",
        anchor_count=0,
        reserve_top_m=0,
        max_bridge_slots=0,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        expand_base_k=2,
        expand_min_structure_score=0.35,
        assemble_mode="none",
        append_max_docs=0,
        append_policy="bridge",
        strongest_shadow_enabled=True,
        strongest_shadow_apply_to_pool=False,
        strongest_candidate_k=3,
        strongest_final_k=2,
        strongest_hippo_head_k=2,
        strongest_smoothed_union_k=2,
        strongest_gamma=0.2,
    )

    selected = selected_solutions[0]
    assert selected.docs[:2] == ["Doc A\nalpha beta", "Doc B\nalpha witness"]
    trace = selected.retrieval_trace["expand_assemble_trace"]
    assert trace["strongest_shadow"]["enabled"] is True
    assert trace["strongest_shadow"]["applied_to_pool"] is False
    assert trace["strongest_shadow"]["shadow_status"] == "ok"
    assert trace["strongest_shadow"]["final_doc_indices"] == [0, 1]


def test_bridge_append_strongest_gbc_shadow_falls_back_cleanly_without_openie():
    hipporag = DummyHippoRAG()
    query_solution = QuerySolution(
        question="where is alpha",
        docs=["Doc A\nalpha beta", "Doc B\nalpha witness", "Doc C\ngamma"],
        doc_scores=np.asarray([0.9, 0.7, 0.2], dtype=float),
    )

    selected_solutions, _summary = apply_setwise_selector(
        hipporag=hipporag,
        query_solutions=[query_solution],
        doc_text_to_chunk_id={
            "Doc A\nalpha beta": "chunk-a",
            "Doc B\nalpha witness": "chunk-b",
            "Doc C\ngamma": "chunk-c",
        },
        pool_k=3,
        qa_top_k=2,
        selector_name="bridge_append",
        score_mode="bridge",
        anchor_count=0,
        reserve_top_m=0,
        max_bridge_slots=0,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        expand_base_k=2,
        expand_min_structure_score=0.35,
        assemble_mode="none",
        append_max_docs=0,
        append_policy="bridge",
        strongest_shadow_enabled=True,
        strongest_shadow_apply_to_pool=False,
        strongest_candidate_k=3,
        strongest_final_k=2,
        strongest_hippo_head_k=2,
        strongest_smoothed_union_k=2,
        strongest_gamma=0.2,
        strongest_rerank_mode="gbc",
        strongest_gbc_protected_anchor_k=1,
        strongest_gbc_head_coverage_k=1,
        strongest_gbc_top_passage_pool_k=3,
        strongest_gbc_frontier_bonus_k=2,
        strongest_gbc_bonus_weight=1.0,
    )

    selected = selected_solutions[0]
    assert selected.docs[:2] == ["Doc A\nalpha beta", "Doc B\nalpha witness"]
    trace = selected.retrieval_trace["expand_assemble_trace"]
    assert trace["strongest_shadow"]["enabled"] is True
    assert trace["strongest_shadow"]["shadow_status"] == "ok"
    assert trace["strongest_shadow"]["trace"]["rerank_mode"] == "gbc"
    assert trace["strongest_shadow"]["trace"]["gbc"]["gbc_status"] == "fallback_no_query_facts"


def test_bridge_append_strongest_ras_shadow_falls_back_cleanly_without_requirements():
    hipporag = DummyHippoRAG()
    query_solution = QuerySolution(
        question="where is alpha",
        docs=["Doc A\nalpha beta", "Doc B\nalpha witness", "Doc C\ngamma"],
        doc_scores=np.asarray([0.9, 0.7, 0.2], dtype=float),
    )

    selected_solutions, _summary = apply_setwise_selector(
        hipporag=hipporag,
        query_solutions=[query_solution],
        doc_text_to_chunk_id={
            "Doc A\nalpha beta": "chunk-a",
            "Doc B\nalpha witness": "chunk-b",
            "Doc C\ngamma": "chunk-c",
        },
        pool_k=3,
        qa_top_k=2,
        selector_name="bridge_append",
        score_mode="bridge",
        anchor_count=0,
        reserve_top_m=0,
        max_bridge_slots=0,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        expand_base_k=2,
        expand_min_structure_score=0.35,
        assemble_mode="none",
        append_max_docs=0,
        append_policy="bridge",
        strongest_shadow_enabled=True,
        strongest_shadow_apply_to_pool=False,
        strongest_candidate_k=3,
        strongest_final_k=2,
        strongest_hippo_head_k=2,
        strongest_smoothed_union_k=2,
        strongest_gamma=0.2,
        strongest_rerank_mode="gbc",
        strongest_gbc_protected_anchor_k=1,
        strongest_gbc_head_coverage_k=1,
        strongest_gbc_top_passage_pool_k=3,
        strongest_gbc_frontier_bonus_k=2,
        strongest_gbc_bonus_weight=1.0,
        strongest_ras_enabled=True,
        strongest_ras_prefix_guard_k=2,
        strongest_ras_requirement_max_units=4,
        strongest_ras_enable_conflict_veto=True,
        strongest_ras_core_support_min_eligible=True,
        strongest_ras_trace_enabled=True,
    )

    selected = selected_solutions[0]
    assert selected.docs[:2] == ["Doc A\nalpha beta", "Doc B\nalpha witness"]
    trace = selected.retrieval_trace["expand_assemble_trace"]
    assert trace["strongest_shadow"]["enabled"] is True
    assert trace["strongest_shadow"]["shadow_status"] == "ok"
    assert trace["strongest_shadow"]["trace"]["rerank_mode"] == "gbc"
    assert trace["strongest_shadow"]["trace"]["gbc"]["gbc_status"] == "fallback_no_query_facts"
    assert trace["strongest_shadow"]["trace"]["gbc"]["ras"]["ras_status"] == "fallback_no_requirements"
    assert trace["strongest_shadow"]["trace"]["gbc"]["ras"]["ras_status"] == "fallback_no_requirements"
