#!/usr/bin/env python3
"""Requirement reachability audit for strongest RAS.

Separates three failure sources:
1. parser/schema never produced usable requirement units
2. pool already contains a witness but the current matcher misses it
3. pool does not contain a usable witness, so fixed-pool rerank has no action space
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_ras_llm import make_args
from eval_causal_qwen3 import (
    build_config,
    build_doc_text_to_chunk_id,
    collect_lexical_query_seed_entities,
    collect_query_seed_entities,
    collect_question_query_entities,
    extract_doc_title,
)
from run_strongest_shadow_smoke import _resolve_local_llm_runtime
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag_ext.strongest.gbc import (
    _build_local_requirement_support_tensor,
    build_requirement_units,
    compute_requirement_set_coverage,
    compute_requirement_support,
)
from src.hipporag_ext.strongest.runtime import l2_normalize_rows
from src.hipporag_ext.strongest.shadow_entry import (
    _build_passage_chunk_ids_for_pool,
    _get_chunk_triples_map,
    run_strongest_shadow_for_pool,
)
from src.hipporag_ext.strongest.traces import normalize_entity_text
from src.hipporag_ext.strongest.types import StrongestConfig

CLASSIFICATION_KEYS = (
    "present_in_head",
    "present_in_pool_but_matcher_missed",
    "present_in_pool_and_matched",
    "absent_from_pool",
)


def _build_pool_metadata(
    *,
    hipporag,
    query: str,
    pool_docs: list[str],
    pool_doc_ids: list[int | None],
    seed_entities: list[str],
    query_entities: list[str],
    extractor_mode: str,
    support_mode: str,
    embedding_probe_threshold: float,
) -> dict[str, object]:
    passage_embeddings_raw = getattr(hipporag, "passage_embeddings", None)
    if passage_embeddings_raw is None:
        passage_embeddings = np.zeros((0, 0), dtype=np.float32)
    else:
        passage_embeddings = np.asarray(passage_embeddings_raw, dtype=np.float32)
    embedding_dim = int(passage_embeddings.shape[1]) if passage_embeddings.ndim == 2 and passage_embeddings.shape[0] > 0 else 0
    local_passage_embeddings = np.zeros((len(pool_docs), embedding_dim), dtype=np.float32)
    if embedding_dim > 0:
        for local_idx, doc_id in enumerate(pool_doc_ids):
            if doc_id is None or int(doc_id) < 0 or int(doc_id) >= len(passage_embeddings):
                continue
            local_passage_embeddings[local_idx] = passage_embeddings[int(doc_id)]
        local_passage_embeddings = l2_normalize_rows(local_passage_embeddings)

    passage_chunk_ids = _build_passage_chunk_ids_for_pool(hipporag=hipporag, pool_doc_ids=pool_doc_ids)
    passage_titles = [extract_doc_title(doc_text) for doc_text in pool_docs]
    passage_texts = [str(doc_text) for doc_text in pool_docs]
    doc_idx_to_entities = getattr(hipporag, "doc_idx_to_structure_entities", {}) or {}
    passage_structure_entities = [
        sorted(
            {
                normalize_entity_text(entity)
                for entity in doc_idx_to_entities.get(int(doc_id), set())
                if doc_id is not None and normalize_entity_text(entity)
            }
        )
        if doc_id is not None
        else []
        for doc_id in pool_doc_ids
    ]
    chunk_triples_cache = _get_chunk_triples_map(hipporag)
    chunk_triples_map = {
        str(chunk_id): list(chunk_triples_cache.get(str(chunk_id), []))
        for chunk_id in passage_chunk_ids
        if chunk_id is not None
    }
    return {
        "pool_doc_ids": [None if doc_id is None else int(doc_id) for doc_id in pool_doc_ids],
        "passage_chunk_ids": list(passage_chunk_ids),
        "passage_titles": list(passage_titles),
        "passage_texts": list(passage_texts),
        "passage_structure_entities": list(passage_structure_entities),
        "chunk_triples_map": dict(chunk_triples_map),
        "query_text": str(query),
        "seed_entities": sorted(
            {
                normalize_entity_text(entity)
                for entity in seed_entities
                if normalize_entity_text(entity)
            }
        ),
        "query_entities": sorted(
            {
                normalize_entity_text(entity)
                for entity in query_entities
                if normalize_entity_text(entity)
            }
        ),
        "ras_extractor_mode": str(extractor_mode),
        "ras_support_mode": str(support_mode),
        "ras_embedding_probe_threshold": float(embedding_probe_threshold),
        "_ras_embedding_model": getattr(hipporag, "embedding_model", None),
        "_ras_passage_embeddings": np.asarray(local_passage_embeddings, dtype=np.float32),
    }


def _positive_titles(
    titles: list[str],
    support_scores: np.ndarray,
    unit_idx: int,
    limit: int = 3,
) -> list[str]:
    if support_scores.ndim != 2 or unit_idx < 0 or unit_idx >= support_scores.shape[1]:
        return []
    indices = [idx for idx in range(support_scores.shape[0]) if float(support_scores[idx, unit_idx]) > 0.0]
    return [titles[idx] for idx in indices[:limit] if 0 <= idx < len(titles)]


def _increment_bucket(summary: dict[str, int], key: str) -> None:
    summary[key] = int(summary.get(key, 0)) + 1


def _classify_units(
    *,
    requirement_units,
    titles: list[str],
    head_local_indices: np.ndarray,
    current_scores: np.ndarray,
    current_fillers,
    oracle_scores: np.ndarray,
    oracle_fillers,
):
    local_indices = np.arange(current_scores.shape[0], dtype=np.int64)
    current_head_cov = compute_requirement_set_coverage(
        requirement_units=requirement_units,
        support_scores=current_scores,
        local_indices=head_local_indices,
        filler_values=current_fillers,
    )
    current_pool_cov = compute_requirement_set_coverage(
        requirement_units=requirement_units,
        support_scores=current_scores,
        local_indices=local_indices,
        filler_values=current_fillers,
    )
    oracle_head_cov = compute_requirement_set_coverage(
        requirement_units=requirement_units,
        support_scores=oracle_scores,
        local_indices=head_local_indices,
        filler_values=oracle_fillers,
    )
    oracle_pool_cov = compute_requirement_set_coverage(
        requirement_units=requirement_units,
        support_scores=oracle_scores,
        local_indices=local_indices,
        filler_values=oracle_fillers,
    )

    per_unit = []
    counts = {key: 0 for key in CLASSIFICATION_KEYS}
    tier_counts = {
        "support": {key: 0 for key in CLASSIFICATION_KEYS},
        "core": {key: 0 for key in CLASSIFICATION_KEYS},
    }
    for unit_idx, unit in enumerate(requirement_units):
        if float(oracle_head_cov[unit_idx]) > 0.0:
            classification = "present_in_head"
        elif float(oracle_pool_cov[unit_idx]) > 0.0 and float(current_pool_cov[unit_idx]) <= 0.0:
            classification = "present_in_pool_but_matcher_missed"
        elif float(oracle_pool_cov[unit_idx]) > 0.0:
            classification = "present_in_pool_and_matched"
        else:
            classification = "absent_from_pool"

        _increment_bucket(counts, classification)
        if unit.tier in tier_counts:
            _increment_bucket(tier_counts[unit.tier], classification)

        per_unit.append(
            {
                "unit_id": unit.unit_id,
                "tier": unit.tier,
                "slot_family": unit.slot_family,
                "raw_slot_family": unit.raw_slot_family,
                "anchor_entities": list(unit.anchor_entities),
                "bridge_targets": list(unit.bridge_targets),
                "classification": classification,
                "current_head_coverage": float(current_head_cov[unit_idx]),
                "current_pool_coverage": float(current_pool_cov[unit_idx]),
                "oracle_head_coverage": float(oracle_head_cov[unit_idx]),
                "oracle_pool_coverage": float(oracle_pool_cov[unit_idx]),
                "current_positive_titles": _positive_titles(titles, current_scores, unit_idx),
                "oracle_positive_titles": _positive_titles(titles, oracle_scores, unit_idx),
            }
        )
    return per_unit, counts, tier_counts


def run_reachability_audit(
    dataset_name: str,
    limit: int,
    *,
    extractor_mode: str,
    current_support_mode: str,
    oracle_support_mode: str,
    current_threshold: float,
    oracle_threshold: float,
):
    os.environ.setdefault("HIPPORAG_RERANK_FORCE_NO_THINK", "1")
    save_dir = f"outputs_step0_general_{dataset_name}"
    args = make_args(dataset_name, limit, save_dir)
    resolved_url, resolved_model = _resolve_local_llm_runtime(args)
    if resolved_url:
        args.llm_base_url = resolved_url
    args.llm_request_name = resolved_model
    print(f"LLM: {args.llm_base_url} / {args.llm_request_name}", file=sys.stderr)

    corpus = json.load(open(f"reproduce/dataset/{dataset_name}_corpus.json"))
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json"))[:limit]
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    queries = [s["question"] for s in samples]

    args.save_dir = save_dir
    config = build_config(args, corpus_len=len(corpus))
    hipporag = HippoRAG(
        global_config=config, save_dir=save_dir,
        llm_model_name=args.llm_name, llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name, embedding_base_url=args.embedding_base_url,
    )
    hipporag.index(docs=docs)
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=args.retrieval_top_k)

    base_cfg = StrongestConfig(
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        hippo_head_k=args.hippo_head_k,
        smoothed_union_k=args.smoothed_union_k,
        gamma=args.gamma,
        rerank_mode="gbc",
        gbc_protected_anchor_k=args.gbc_protected_anchor_k,
        gbc_head_coverage_k=args.gbc_head_coverage_k,
        gbc_top_passage_pool_k=args.gbc_top_passage_pool_k,
        gbc_frontier_bonus_k=args.gbc_frontier_bonus_k,
        gbc_bonus_weight=args.gbc_bonus_weight,
    )

    agg = {
        "query_total": 0,
        "queries_with_active_units": 0,
        "queries_with_unsupported_slots": 0,
        "unsupported_slot_count_sum": 0,
        "skipped_unknown_slot_sum": 0,
        "unit_total": 0,
        "support_unit_total": 0,
        "core_unit_total": 0,
        "classification_counts": {key: 0 for key in CLASSIFICATION_KEYS},
        "support_classification_counts": {key: 0 for key in CLASSIFICATION_KEYS},
        "core_classification_counts": {key: 0 for key in CLASSIFICATION_KEYS},
    }
    per_query = []

    for result in retrieval_results:
        pool_docs = list(result.docs[:max(args.final_k, args.candidate_k, args.qa_top_k)])
        pool_scores = np.asarray(result.doc_scores[: len(pool_docs)], dtype=np.float32)
        pool_ids = [
            hipporag.passage_node_key_to_doc_idx.get(doc_text_to_chunk_id.get(doc))
            for doc in pool_docs
        ]
        seed_ents = collect_query_seed_entities(hipporag, result.question)
        if not seed_ents:
            seed_ents = collect_lexical_query_seed_entities(
                query=result.question,
                pool_doc_ids=pool_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )
        query_ents = collect_question_query_entities(
            hipporag=hipporag,
            query=result.question,
            pool_doc_ids=pool_ids,
            doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
        )

        base_result = run_strongest_shadow_for_pool(
            hipporag=hipporag,
            query=result.question,
            pool_docs=pool_docs,
            pool_doc_ids=pool_ids,
            pool_doc_scores=pool_scores,
            seed_entities=seed_ents,
            query_entities=query_ents,
            config=base_cfg,
        )

        current_metadata = _build_pool_metadata(
            hipporag=hipporag,
            query=result.question,
            pool_docs=pool_docs,
            pool_doc_ids=pool_ids,
            seed_entities=seed_ents,
            query_entities=query_ents,
            extractor_mode=extractor_mode,
            support_mode=current_support_mode,
            embedding_probe_threshold=current_threshold,
        )
        requirement_units, extractor_trace = build_requirement_units(
            query=result.question,
            metadata=current_metadata,
            max_units=4,
        )

        agg["query_total"] += 1
        unsupported_slot_count = int(extractor_trace.get("unsupported_slot_count", 0) or 0)
        skipped_unknown_slot = int(extractor_trace.get("skipped_unknown_slot", 0) or 0)
        agg["queries_with_unsupported_slots"] += int(unsupported_slot_count > 0)
        agg["unsupported_slot_count_sum"] += unsupported_slot_count
        agg["skipped_unknown_slot_sum"] += skipped_unknown_slot

        query_record = {
            "query": result.question,
            "extractor_mode": extractor_trace.get("extractor_mode", extractor_mode),
            "extractor_status": extractor_trace.get("extractor_status", ""),
            "active_unit_count": len(requirement_units),
            "unsupported_slot_count": unsupported_slot_count,
            "unsupported_raw_slots": list(extractor_trace.get("unsupported_raw_slots", [])),
            "skipped_unknown_slot": skipped_unknown_slot,
            "units": [],
        }

        if not requirement_units:
            per_query.append(query_record)
            continue

        agg["queries_with_active_units"] += 1

        oracle_metadata = dict(current_metadata)
        oracle_metadata["ras_support_mode"] = str(oracle_support_mode)
        oracle_metadata["ras_embedding_probe_threshold"] = float(oracle_threshold)

        local_indices = np.arange(len(pool_docs), dtype=np.int64)
        current_tensor, current_fillers = _build_local_requirement_support_tensor(
            candidate_indices=local_indices,
            selected_local_indices=local_indices,
            requirement_units=requirement_units,
            metadata=current_metadata,
        )
        current_scores = compute_requirement_support(
            support_tensor=current_tensor,
            requirement_units=requirement_units,
            strict_eligibility=True,
        )
        oracle_tensor, oracle_fillers = _build_local_requirement_support_tensor(
            candidate_indices=local_indices,
            selected_local_indices=local_indices,
            requirement_units=requirement_units,
            metadata=oracle_metadata,
        )
        oracle_scores = compute_requirement_support(
            support_tensor=oracle_tensor,
            requirement_units=requirement_units,
            strict_eligibility=True,
        )

        if base_result is not None:
            head_local_indices = np.asarray(
                base_result.final_doc_indices.tolist()[: int(args.gbc_head_coverage_k)],
                dtype=np.int64,
            )
        else:
            head_local_indices = np.arange(min(int(args.gbc_head_coverage_k), len(pool_docs)), dtype=np.int64)

        unit_records, counts, tier_counts = _classify_units(
            requirement_units=requirement_units,
            titles=[extract_doc_title(doc) for doc in pool_docs],
            head_local_indices=head_local_indices,
            current_scores=current_scores,
            current_fillers=current_fillers,
            oracle_scores=oracle_scores,
            oracle_fillers=oracle_fillers,
        )
        query_record["units"] = unit_records
        per_query.append(query_record)

        for unit in requirement_units:
            agg["unit_total"] += 1
            if unit.tier == "support":
                agg["support_unit_total"] += 1
            elif unit.tier == "core":
                agg["core_unit_total"] += 1
        for key, value in counts.items():
            agg["classification_counts"][key] += int(value)
        for key, value in tier_counts["support"].items():
            agg["support_classification_counts"][key] += int(value)
        for key, value in tier_counts["core"].items():
            agg["core_classification_counts"][key] += int(value)

    return agg, per_query


def print_summary(name: str, agg: dict[str, object]) -> None:
    print(f"\n{'=' * 60}")
    print(f"{name} — Requirement Reachability Summary")
    print(f"{'=' * 60}")
    print(f"  {'queries_with_active_units':34s}: {agg['queries_with_active_units']}/{agg['query_total']}")
    print(f"  {'queries_with_unsupported_slots':34s}: {agg['queries_with_unsupported_slots']}/{agg['query_total']}")
    print(f"  {'unsupported_slot_count_sum':34s}: {agg['unsupported_slot_count_sum']}")
    print(f"  {'skipped_unknown_slot_sum':34s}: {agg['skipped_unknown_slot_sum']}")
    print(f"  {'unit_total':34s}: {agg['unit_total']}")
    print(f"  {'support_unit_total':34s}: {agg['support_unit_total']}")
    print(f"  {'core_unit_total':34s}: {agg['core_unit_total']}")
    for key in CLASSIFICATION_KEYS:
        print(f"  {key:34s}: {agg['classification_counts'][key]}")
    print("  support_tier:")
    for key in CLASSIFICATION_KEYS:
        print(f"    {key:32s}: {agg['support_classification_counts'][key]}")
    print("  core_tier:")
    for key in CLASSIFICATION_KEYS:
        print(f"    {key:32s}: {agg['core_classification_counts'][key]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_limit", type=int, default=20)
    parser.add_argument("--hotpot_limit", type=int, default=10)
    parser.add_argument("--extractor_mode", choices=["llm_closed_grounded", "llm_grounded", "llm"], default="llm_closed_grounded")
    parser.add_argument("--current_support_mode", choices=["lexical", "embedding_probe"], default="lexical")
    parser.add_argument("--oracle_support_mode", choices=["lexical", "embedding_probe"], default="embedding_probe")
    parser.add_argument("--current_threshold", type=float, default=0.35)
    parser.add_argument("--oracle_threshold", type=float, default=0.30)
    cli_args = parser.parse_args()

    print("Running 2Wiki reachability audit...")
    agg_2w, pq_2w = run_reachability_audit(
        "2wikimultihopqa",
        cli_args.wiki_limit,
        extractor_mode=cli_args.extractor_mode,
        current_support_mode=cli_args.current_support_mode,
        oracle_support_mode=cli_args.oracle_support_mode,
        current_threshold=float(cli_args.current_threshold),
        oracle_threshold=float(cli_args.oracle_threshold),
    )
    print_summary(f"2Wiki-{cli_args.wiki_limit}", agg_2w)

    print("\nRunning Hotpot reachability audit...")
    agg_hp, pq_hp = run_reachability_audit(
        "hotpotqa",
        cli_args.hotpot_limit,
        extractor_mode=cli_args.extractor_mode,
        current_support_mode=cli_args.current_support_mode,
        oracle_support_mode=cli_args.oracle_support_mode,
        current_threshold=float(cli_args.current_threshold),
        oracle_threshold=float(cli_args.oracle_threshold),
    )
    print_summary(f"Hotpot-{cli_args.hotpot_limit}", agg_hp)

    out_path = "outputs_smoke/ras_reachability_audit.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(
        {
            "extractor_mode": cli_args.extractor_mode,
            "current_support_mode": cli_args.current_support_mode,
            "oracle_support_mode": cli_args.oracle_support_mode,
            "current_threshold": float(cli_args.current_threshold),
            "oracle_threshold": float(cli_args.oracle_threshold),
            "2wiki": {"agg": agg_2w, "per_query": pq_2w},
            "hotpot": {"agg": agg_hp, "per_query": pq_hp},
        },
        open(out_path, "w"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"\nFull trace saved to {out_path}")
