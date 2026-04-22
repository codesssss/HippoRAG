#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import (
    build_config,
    build_doc_text_to_chunk_id,
    collect_lexical_query_seed_entities,
    collect_query_seed_entities,
    collect_question_query_entities,
    extract_doc_title,
    get_gold_answers,
    get_gold_docs,
)
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag_ext.strongest.shadow_entry import run_strongest_shadow_for_pool
from src.hipporag_ext.strongest.types import StrongestConfig


LOCAL_LLM_CANDIDATES = [
    ("http://127.0.0.1:8039/v1", "qwen3-8b"),
    ("http://127.0.0.1:8041/v1", "qwen3-8b-train"),
    ("http://127.0.0.1:8042/v1", "qwen3-8b-train"),
    ("http://127.0.0.1:8043/v1", "qwen3-8b-train"),
]


def _probe_openai_models(base_url: str, timeout_s: float = 2.0) -> list[str]:
    models_url = str(base_url).rstrip("/") + "/models"
    opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
    req = urllib_request.Request(models_url, headers={"Accept": "application/json"})
    try:
        with opener.open(req, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError):
        return []
    model_entries = payload.get("data", []) if isinstance(payload, dict) else []
    return [
        str(entry.get("id"))
        for entry in model_entries
        if isinstance(entry, dict) and entry.get("id")
    ]


def _resolve_local_llm_runtime(args) -> tuple[str, str | None]:
    requested_base_url = str(getattr(args, "llm_base_url", "") or "").strip()
    requested_model = str(getattr(args, "llm_request_name", "") or "").strip()
    fallback_model = str(getattr(args, "llm_name", "") or "").strip()

    if requested_base_url:
        live_models = _probe_openai_models(requested_base_url)
        if live_models:
            if requested_model:
                return requested_base_url, requested_model
            if fallback_model and fallback_model in live_models:
                return requested_base_url, fallback_model
            return requested_base_url, live_models[0]

    for candidate_base_url, candidate_model in LOCAL_LLM_CANDIDATES:
        live_models = _probe_openai_models(candidate_base_url)
        if not live_models:
            continue
        if requested_model and requested_model in live_models:
            return candidate_base_url, requested_model
        if fallback_model and fallback_model in live_models:
            return candidate_base_url, fallback_model
        if candidate_model in live_models:
            return candidate_base_url, candidate_model
        return candidate_base_url, live_models[0]

    return requested_base_url, requested_model or None


def parse_args():
    parser = argparse.ArgumentParser(description="Run strongest sidecar in shadow mode on baseline HippoRAG retrieval.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--llm_request_name", type=str, default="",
                        help="Optional API-side model name for the chat/rerank service. When omitted, the smoke script will auto-detect a live local Qwen3 endpoint.")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    parser.add_argument("--causal_engine_version", choices=["legacy", "v2"], default="v2")
    parser.add_argument("--causal_v2_base_retrieval_mode", choices=["dense", "legacy_fact_graph", "general_relation_graph"], default="legacy_fact_graph")
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=100)
    parser.add_argument("--candidate_k", type=int, default=20)
    parser.add_argument("--final_k", type=int, default=10)
    parser.add_argument("--hippo_head_k", type=int, default=10)
    parser.add_argument("--smoothed_union_k", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--rerank_mode", choices=["standard", "gbc"], default="standard")
    parser.add_argument("--gbc_protected_anchor_k", type=int, default=2)
    parser.add_argument("--gbc_head_coverage_k", type=int, default=5)
    parser.add_argument("--gbc_top_passage_pool_k", type=int, default=24)
    parser.add_argument("--gbc_frontier_bonus_k", type=int, default=6)
    parser.add_argument("--gbc_bonus_weight", type=float, default=1.0)
    args = parser.parse_args()
    default_attrs = {
        "force_index_from_scratch": "false",
        "force_openie_from_scratch": "false",
        "linking_top_k": 10,
        "max_qa_steps": 1,
        "embedding_batch_size": 4,
        "max_retry_attempts": 1,
        "planner_enabled": "false",
        "planner_mode": "none",
        "planner_max_steps": 0,
        "causal_enabled": "false",
        "causal_query_only": "false",
        "causal_gate_mode": "none",
        "causal_seed_top_k": 8,
        "causal_confidence_threshold": 0.7,
        "causal_damping": 0.15,
        "causal_blend_dense_weight": 1.0,
        "causal_blend_fact_weight": 0.0,
        "causal_blend_graph_weight": 0.0,
        "causal_margin_gate_enabled": "false",
        "causal_margin_threshold": 0.0,
        "causal_blend_top_k": 10,
        "structure_rerank_enabled": "false",
        "structure_rerank_top_n": 0,
        "structure_rerank_bonus_weight": 0.0,
        "structure_rerank_min_edge_support": 0,
        "structure_rerank_max_top5_swaps": 0,
        "structure_rerank_seed_top_k": 0,
        "structure_rerank_max_hops": 0,
        "structure_rerank_margin_threshold": 0.0,
        "rerank_require_non_empty": "true",
    }
    for attr_name, attr_value in default_attrs.items():
        if not hasattr(args, attr_name):
            setattr(args, attr_name, attr_value)
    return args


def main():
    args = parse_args()
    os.environ.setdefault("HIPPORAG_RERANK_FORCE_NO_THINK", "1")
    resolved_llm_base_url, resolved_llm_request_name = _resolve_local_llm_runtime(args)
    if resolved_llm_base_url:
        args.llm_base_url = resolved_llm_base_url
    args.llm_request_name = resolved_llm_request_name
    print(
        json.dumps(
            {
                "llm_base_url": args.llm_base_url,
                "llm_name": args.llm_name,
                "llm_request_name": args.llm_request_name,
                "rerank_force_no_think": os.environ.get("HIPPORAG_RERANK_FORCE_NO_THINK"),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    dataset_name = args.dataset
    requested_save_dir = Path(args.save_dir)
    if requested_save_dir.name == dataset_name or requested_save_dir.name.endswith(f"_{dataset_name}"):
        resolved_save_dir = str(requested_save_dir)
    else:
        resolved_save_dir = str(requested_save_dir / dataset_name)
    corpus_path = Path(f"reproduce/dataset/{dataset_name}_corpus.json")
    sample_path = Path(f"reproduce/dataset/{dataset_name}.json")
    corpus = json.load(corpus_path.open())
    samples = json.load(sample_path.open())[: args.limit]
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    _ = get_gold_answers(samples)
    _ = get_gold_docs(samples, dataset_name, corpus=corpus)
    queries = [sample["question"] for sample in samples]

    args.save_dir = resolved_save_dir
    config = build_config(args, corpus_len=len(corpus))
    hipporag = HippoRAG(
        global_config=config,
        save_dir=resolved_save_dir,
        llm_model_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
    )
    hipporag.index(docs=docs)
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=args.retrieval_top_k)
    strongest_config = StrongestConfig(
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        hippo_head_k=args.hippo_head_k,
        smoothed_union_k=args.smoothed_union_k,
        gamma=args.gamma,
        rerank_mode=str(args.rerank_mode),
        gbc_protected_anchor_k=int(args.gbc_protected_anchor_k),
        gbc_head_coverage_k=int(args.gbc_head_coverage_k),
        gbc_top_passage_pool_k=int(args.gbc_top_passage_pool_k),
        gbc_frontier_bonus_k=int(args.gbc_frontier_bonus_k),
        gbc_bonus_weight=float(args.gbc_bonus_weight),
    )

    for result in retrieval_results:
        pool_docs = list(result.docs[: max(args.final_k, args.candidate_k, args.qa_top_k)])
        pool_doc_scores = np.asarray(result.doc_scores[: len(pool_docs)], dtype=np.float32)
        pool_doc_ids = [
            hipporag.passage_node_key_to_doc_idx.get(doc_text_to_chunk_id.get(doc))
            for doc in pool_docs
        ]
        seed_entities = collect_query_seed_entities(hipporag, result.question)
        if not seed_entities:
            seed_entities = collect_lexical_query_seed_entities(
                query=result.question,
                pool_doc_ids=pool_doc_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )
        query_entities = collect_question_query_entities(
            hipporag=hipporag,
            query=result.question,
            pool_doc_ids=pool_doc_ids,
            doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
        )
        shadow_result = run_strongest_shadow_for_pool(
            hipporag=hipporag,
            query=result.question,
            pool_docs=pool_docs,
            pool_doc_ids=pool_doc_ids,
            pool_doc_scores=pool_doc_scores,
            seed_entities=seed_entities,
            query_entities=query_entities,
            config=strongest_config,
        )
        baseline_titles = [extract_doc_title(doc) for doc in pool_docs[: args.qa_top_k]]
        strongest_titles = (
            [extract_doc_title(pool_docs[idx]) for idx in shadow_result.final_doc_indices.tolist()]
            if shadow_result is not None
            else []
        )
        print(json.dumps(
            {
                "query": result.question,
                "baseline_top_k": baseline_titles,
                "strongest_top_k": strongest_titles,
                "diff_summary": {
                    "overlap": len(set(baseline_titles) & set(strongest_titles)),
                    "baseline_only": [title for title in baseline_titles if title not in strongest_titles],
                    "strongest_only": [title for title in strongest_titles if title not in baseline_titles],
                },
                "strongest_trace": None if shadow_result is None else shadow_result.trace,
                "seed_entities": sorted(seed_entities),
                "query_entities": sorted(query_entities),
            },
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
