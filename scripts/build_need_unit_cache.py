import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_requirement_cache import build_canonical_args, load_dataset, resolve_save_dir
from eval_causal_qwen3 import (
    build_config,
    build_doc_text_to_chunk_id,
    collect_lexical_query_seed_entities,
    collect_query_seed_entities,
    collect_question_query_entities,
    get_gold_docs,
)
from requirement_beam_utils import (
    DEFAULT_NEED_UNIT_MAX_RELATION_HOPS,
    DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS,
    DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE,
    DEFAULT_QDMR_POOL_TITLE_HEAD,
    DEFAULT_REQUIREMENT_ANNOTATION_POOL_K,
    ATOMIC_ANNOTATION_SCORE_MODES,
    NEED_UNIT_CACHE_VERSION,
    build_need_unit_cache_entry,
    guess_answer_type_label,
    infer_qdmr_step_plan,
    load_need_unit_atomic_model_bundle,
    save_requirement_cache,
)
from src.hipporag.HippoRAG import HippoRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline need-unit caches for PCRS-RAG V2.")
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
    parser.add_argument("--max_counterfactual_sets", type=int, default=DEFAULT_NEED_UNIT_MAX_COUNTERFACTUALS)
    parser.add_argument("--max_relation_hops", type=int, default=DEFAULT_NEED_UNIT_MAX_RELATION_HOPS)
    parser.add_argument("--relation_hop_cap_mode", choices=["fixed", "conditional"], default=DEFAULT_NEED_UNIT_RELATION_HOP_CAP_MODE)
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    parser.add_argument("--annotation_score_mode", choices=sorted(ATOMIC_ANNOTATION_SCORE_MODES), default="heuristic")
    parser.add_argument("--atomic_model_path", type=str, default="")
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

    annotation_score_mode = str(args.annotation_score_mode or "heuristic").strip().lower()
    atomic_model_path = str(args.atomic_model_path or "").strip()
    atomic_scorer_bundle = None
    if annotation_score_mode == "hybrid":
        if not atomic_model_path:
            raise ValueError("--annotation_score_mode=hybrid requires --atomic_model_path")
        atomic_scorer_bundle = load_need_unit_atomic_model_bundle(atomic_model_path)

    logger.info("Retrieving %d queries to build need-unit cache on %s", len(queries), args.dataset)
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
        predicted_answer_type = guess_answer_type_label(qs.question)
        parser_trace = infer_qdmr_step_plan(
            question=qs.question,
            question_entities=question_entities,
            seed_entities=seed_entities,
            pool_titles=[
                str(doc_text).split("\n", 1)[0].strip()
                for doc_text in pool_docs[:DEFAULT_QDMR_POOL_TITLE_HEAD]
            ],
            predicted_answer_type=predicted_answer_type,
            llm_infer_fn=hipporag.llm_model.infer,
            model_name=hipporag.global_config.llm_name,
        )
        cache_entry = build_need_unit_cache_entry(
            query_index=q_idx,
            question=qs.question,
            pool_docs=pool_docs,
            pool_doc_entities=pool_doc_entities,
            seed_entities=seed_entities,
            question_entities=question_entities,
            annotation_pool_k=int(args.annotation_pool_k),
            max_counterfactual_sets=int(args.max_counterfactual_sets),
            max_relation_hops=int(args.max_relation_hops),
            relation_hop_cap_mode=str(args.relation_hop_cap_mode),
            atomic_scorer_bundle=atomic_scorer_bundle,
            score_mode=annotation_score_mode,
            predicted_answer_type=predicted_answer_type,
            step_plan_payload=parser_trace.get("payload"),
            parser_trace=parser_trace,
        )
        cache_entries.append(cache_entry)
        mapped_pool_doc_counts.append(sum(doc_id is not None for doc_id in pool_doc_ids))

    output_path = Path(args.output_path) if args.output_path else Path(
        "research_memory/emnlp_expand_then_compose/models"
    ) / f"{args.dataset}_need_unit_cache_pool{args.setwise_pool_k}_ann{args.annotation_pool_k}_limit{len(samples)}.json"
    payload = {
        "version": NEED_UNIT_CACHE_VERSION,
        "dataset": args.dataset,
        "limit": len(samples),
        "save_dir": resolved_save_dir,
        "setwise_pool_k": int(args.setwise_pool_k),
        "annotation_pool_k": int(args.annotation_pool_k),
        "qa_top_k": int(args.qa_top_k),
        "max_relation_hops": int(args.max_relation_hops),
        "relation_hop_cap_mode": str(args.relation_hop_cap_mode),
        "annotation_score_mode": annotation_score_mode,
        "atomic_model_path": atomic_model_path,
        "retrieval_metrics": retrieval_metrics,
        "avg_mapped_pool_doc_count": round(sum(mapped_pool_doc_counts) / max(1, len(mapped_pool_doc_counts)), 4),
        "queries": cache_entries,
    }
    save_requirement_cache(output_path, payload)
    logger.info(
        "Need-unit cache saved to %s (queries=%d, avg mapped pool docs=%.2f)",
        output_path,
        len(cache_entries),
        float(payload["avg_mapped_pool_doc_count"]),
    )


if __name__ == "__main__":
    main()
