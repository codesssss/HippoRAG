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
from eval_causal_qwen3 import build_config, build_doc_text_to_chunk_id, get_gold_docs
from requirement_beam_utils import (
    align_requirement_cache_entry_to_pool,
    build_cache_doc_annotation,
    load_need_unit_atomic_model_bundle,
    is_need_unit_cache_version,
    load_requirement_cache,
    save_requirement_cache,
)
from src.hipporag.HippoRAG import HippoRAG


def resolve_atomic_annotation_bundle(score_mode: str,
                                     atomic_model_path: str) -> dict | None:
    normalized_score_mode = str(score_mode or "heuristic").strip().lower()
    if normalized_score_mode not in {"heuristic", "hybrid"}:
        raise ValueError(f"Unsupported --score_mode: {score_mode}")
    if normalized_score_mode == "heuristic":
        return None
    if not str(atomic_model_path or "").strip():
        raise ValueError("--score_mode=hybrid requires --atomic_model_path")
    return load_need_unit_atomic_model_bundle(atomic_model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild need-unit doc annotations against the current retrieval pool.")
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general")
    parser.add_argument("--output_path", type=str, default="")
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
    parser.add_argument("--score_mode", choices=["heuristic", "hybrid"], default="heuristic")
    parser.add_argument("--atomic_model_path", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cache_payload = load_requirement_cache(args.cache_path)
    if not is_need_unit_cache_version(cache_payload):
        raise ValueError(f"annotate_need_unit_support requires a V2 cache, got {cache_payload.get('version')}")
    atomic_scorer_bundle = resolve_atomic_annotation_bundle(
        score_mode=args.score_mode,
        atomic_model_path=args.atomic_model_path,
    )

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
    query_solutions, _ = hipporag.retrieve(
        queries=queries,
        gold_docs=gold_docs,
    )
    chunk_text_to_hash = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}
    passage_node_key_to_doc_idx = getattr(hipporag, "passage_node_key_to_doc_idx", {}) or {}

    rebuilt_entries = []
    rebuilt_count = 0
    for q_idx, qs in enumerate(query_solutions):
        entry = cache_payload["queries"][q_idx]
        pool_limit = min(len(qs.docs), int(entry.get("annotation_pool_k", 0) or 0))
        pool_docs = list(qs.docs[:pool_limit])
        pool_titles = [str(doc_text).split("\n", 1)[0].strip() for doc_text in pool_docs]
        pool_doc_ids = []
        for doc_text in pool_docs:
            chunk_id = doc_text_to_chunk_id.get(doc_text) or chunk_text_to_hash.get(doc_text)
            mapped_doc_id = passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
            pool_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)
        pool_doc_entities = [
            hipporag.doc_idx_to_structure_entities.get(int(doc_id), set())
            if doc_id is not None else set()
            for doc_id in pool_doc_ids
        ]

        aligned_entry = align_requirement_cache_entry_to_pool(
            cache_entry=entry,
            pool_titles=pool_titles,
            pool_docs=pool_docs,
            pool_doc_entities=pool_doc_entities,
        )
        doc_annotations = []
        for pool_position in range(pool_limit):
            doc_annotations.append(build_cache_doc_annotation(
                pool_position=pool_position,
                doc_text=pool_docs[pool_position],
                doc_entities=pool_doc_entities[pool_position],
                cache_entry=aligned_entry,
                atomic_scorer_bundle=atomic_scorer_bundle,
                score_mode=str(args.score_mode),
            ))
        aligned_entry["doc_annotations"] = doc_annotations
        aligned_entry["diagnostics"] = dict(aligned_entry.get("diagnostics", {}) or {})
        aligned_entry["diagnostics"]["annotation_rebuilt"] = True
        aligned_entry["diagnostics"]["annotation_score_mode"] = str(args.score_mode)
        if atomic_scorer_bundle:
            aligned_entry["diagnostics"]["atomic_scorer_version"] = str(atomic_scorer_bundle.get("scorer_version", ""))
        rebuilt_entries.append(aligned_entry)
        rebuilt_count += 1

    output_path = Path(args.output_path) if args.output_path else Path(args.cache_path)
    updated_payload = dict(cache_payload)
    updated_payload["queries"] = rebuilt_entries
    save_requirement_cache(output_path, updated_payload)
    logger.info("Rebuilt need-unit annotations for %d queries -> %s", rebuilt_count, output_path)


if __name__ == "__main__":
    main()
