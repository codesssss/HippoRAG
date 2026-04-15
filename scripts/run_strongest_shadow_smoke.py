#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

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
    extract_doc_title,
    get_gold_answers,
    get_gold_docs,
)
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag_ext.strongest.shadow_entry import run_strongest_shadow_for_pool
from src.hipporag_ext.strongest.types import StrongestConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run strongest sidecar in shadow mode on baseline HippoRAG retrieval.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
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
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset
    corpus_path = Path(f"reproduce/dataset/{dataset_name}_corpus.json")
    sample_path = Path(f"reproduce/dataset/{dataset_name}.json")
    corpus = json.load(corpus_path.open())
    samples = json.load(sample_path.open())[: args.limit]
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    _ = get_gold_answers(samples)
    _ = get_gold_docs(samples, dataset_name, corpus=corpus)
    queries = [sample["question"] for sample in samples]

    config = build_config(args, corpus_len=len(corpus))
    hipporag = HippoRAG(
        global_config=config,
        save_dir=os.path.join(args.save_dir, dataset_name),
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
    )

    for result in retrieval_results:
        pool_docs = list(result.docs[: max(args.final_k, args.candidate_k, args.qa_top_k)])
        pool_doc_scores = np.asarray(result.doc_scores[: len(pool_docs)], dtype=np.float32)
        pool_doc_ids = [
            hipporag.passage_node_key_to_doc_idx.get(doc_text_to_chunk_id.get(doc))
            for doc in pool_docs
        ]
        seed_entities = set()
        query_entities = set()
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
            },
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
