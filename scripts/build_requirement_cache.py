import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

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
    get_gold_docs,
)
from requirement_beam_utils import (
    DEFAULT_REQUIREMENT_ANNOTATION_POOL_K,
    REQUIREMENT_CACHE_VERSION,
    build_requirement_cache_entry,
    save_requirement_cache,
)
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.dataset_utils import resolve_dataset_paths


def resolve_save_dir(save_dir: str, dataset: str) -> str:
    if save_dir == "outputs":
        return str(Path(save_dir) / dataset)
    return f"{save_dir}_{dataset}"


def load_dataset(dataset: str, limit: int) -> tuple[list[dict], list[dict]]:
    corpus_path, sample_path = resolve_dataset_paths(dataset)
    corpus = json.loads(corpus_path.read_text())
    samples = json.loads(sample_path.read_text())
    if limit and limit > 0:
        samples = samples[:limit]
    return corpus, samples


def build_canonical_args(cli_args: argparse.Namespace, save_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        save_dir=save_dir,
        llm_base_url=cli_args.llm_base_url,
        llm_name=cli_args.llm_name,
        embedding_base_url=cli_args.embedding_base_url,
        dataset=cli_args.dataset,
        embedding_name=cli_args.embedding_name,
        force_index_from_scratch=cli_args.force_index_from_scratch,
        force_openie_from_scratch=cli_args.force_openie_from_scratch,
        max_retry_attempts=cli_args.max_retry_attempts,
        openie_mode=cli_args.openie_mode,
        planner_enabled="false",
        planner_mode="none",
        planner_max_steps=3,
        retrieval_top_k=cli_args.retrieval_top_k,
        linking_top_k=cli_args.linking_top_k,
        qa_top_k=cli_args.qa_top_k,
        max_qa_steps=cli_args.max_qa_steps,
        embedding_batch_size=cli_args.embedding_batch_size,
        causal_enabled="false",
        causal_query_only="true",
        causal_gate_mode="hard",
        causal_seed_top_k=20,
        causal_confidence_threshold=0.5,
        causal_damping=0.7,
        causal_blend_dense_weight=0.35,
        causal_blend_fact_weight=0.15,
        causal_blend_graph_weight=0.50,
        causal_margin_gate_enabled="false",
        causal_margin_threshold=0.02,
        causal_blend_top_k=0,
        causal_engine_version="v2",
        causal_v2_probe_mode="router",
        causal_v2_graph_mode="causal",
        causal_v2_base_retrieval_mode="legacy_fact_graph",
        general_graph_related_to_weight=0.3,
        general_graph_seed_top_k=10,
        causal_v2_extraction_max_tokens=768,
        causal_v2_extraction_retry_attempts=2,
        causal_v2_extraction_workers=4,
        causal_event_top_k=8,
        causal_v2_max_hops=2,
        causal_chain_top_k=6,
        causal_context_max_items=0,
        causal_er_similarity_threshold=0.92,
        causal_er_text_threshold=0.55,
        causal_v2_min_edge_confidence=0.7,
        structure_rerank_enabled="true",
        structure_rerank_top_n=40,
        structure_rerank_bonus_weight=0.08,
        structure_rerank_min_edge_support=2,
        structure_rerank_max_top5_swaps=2,
        structure_rerank_seed_top_k=4,
        structure_rerank_max_hops=2,
        structure_rerank_margin_threshold=0.02,
        rerank_require_non_empty="true",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline requirement caches for PCRS-RAG V1.")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--setwise_pool_k", type=int, default=100)
    parser.add_argument("--annotation_pool_k", type=int, default=DEFAULT_REQUIREMENT_ANNOTATION_POOL_K)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    resolved_save_dir = resolve_save_dir(args.save_dir, args.dataset)
    corpus, samples = load_dataset(args.dataset, args.limit)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_docs = get_gold_docs(samples, args.dataset, corpus=corpus)
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)

    config_args = build_canonical_args(args, resolved_save_dir)
    config = build_config(config_args, corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)

    logger.info("Retrieving %d queries to build requirement cache on %s", len(queries), args.dataset)
    query_solutions, retrieval_metrics = hipporag.retrieve(
        queries=queries,
        gold_docs=gold_docs,
    )

    chunk_text_to_hash = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}
    cache_entries = []
    mapped_pool_doc_counts = []
    for q_idx, qs in enumerate(query_solutions):
        pool_limit = min(
            len(qs.docs),
            max(int(args.setwise_pool_k), int(args.annotation_pool_k), int(args.qa_top_k)),
        )
        pool_docs = list(qs.docs[:pool_limit])
        pool_doc_ids = []
        for doc_text in pool_docs:
            chunk_id = doc_text_to_chunk_id.get(doc_text) or chunk_text_to_hash.get(doc_text)
            mapped_doc_id = hipporag.passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
            pool_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)

        seed_entities = collect_query_seed_entities(hipporag, qs.question)
        if not seed_entities:
            seed_entities = collect_lexical_query_seed_entities(
                query=qs.question,
                pool_doc_ids=pool_doc_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )
        question_entities = collect_question_query_entities(
            hipporag=hipporag,
            query=qs.question,
            pool_doc_ids=pool_doc_ids,
            doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
        )
        pool_doc_entities = [
            hipporag.doc_idx_to_structure_entities.get(int(doc_id), set())
            if doc_id is not None else set()
            for doc_id in pool_doc_ids
        ]
        cache_entry = build_requirement_cache_entry(
            query_index=q_idx,
            question=qs.question,
            pool_docs=pool_docs,
            pool_doc_entities=pool_doc_entities,
            seed_entities=seed_entities,
            question_entities=question_entities,
            annotation_pool_k=int(args.annotation_pool_k),
        )
        cache_entries.append(cache_entry)
        mapped_pool_doc_counts.append(sum(doc_id is not None for doc_id in pool_doc_ids))

    output_path = Path(args.output_path) if args.output_path else Path(
        "research_memory/emnlp_expand_then_compose/models"
    ) / f"{args.dataset}_requirement_cache_pool{args.setwise_pool_k}_ann{args.annotation_pool_k}_limit{len(samples)}.json"
    payload = {
        "version": REQUIREMENT_CACHE_VERSION,
        "dataset": args.dataset,
        "limit": len(samples),
        "save_dir": resolved_save_dir,
        "setwise_pool_k": int(args.setwise_pool_k),
        "annotation_pool_k": int(args.annotation_pool_k),
        "qa_top_k": int(args.qa_top_k),
        "retrieval_metrics": retrieval_metrics,
        "avg_mapped_pool_doc_count": round(sum(mapped_pool_doc_counts) / max(1, len(mapped_pool_doc_counts)), 4),
        "queries": cache_entries,
    }
    save_requirement_cache(output_path, payload)
    logger.info(
        "Requirement cache saved to %s (queries=%d, avg mapped pool docs=%.2f)",
        output_path,
        len(cache_entries),
        float(payload["avg_mapped_pool_doc_count"]),
    )


if __name__ == "__main__":
    main()
