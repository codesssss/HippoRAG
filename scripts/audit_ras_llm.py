#!/usr/bin/env python3
"""RAS-LLM activation audit: runs clean GBC vs GBC+RAS(LLM) side by side."""
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
from src.hipporag_ext.strongest.shadow_entry import run_strongest_shadow_for_pool
from src.hipporag_ext.strongest.types import StrongestConfig


def make_args(dataset, limit, save_dir):
    class Args:
        pass
    a = Args()
    a.dataset = dataset
    a.limit = limit
    a.save_dir = save_dir
    a.llm_base_url = "http://localhost:8039/v1"
    a.llm_name = "qwen3-8b"
    a.llm_request_name = ""
    a.embedding_name = "VLLM//mnt/nvme/Qwen3-Embedding-8B"
    a.embedding_base_url = "http://localhost:8018/v1/embeddings"
    a.openie_mode = "online"
    a.causal_engine_version = "v2"
    a.causal_v2_base_retrieval_mode = "legacy_fact_graph"
    a.qa_top_k = 5
    a.retrieval_top_k = 100
    a.candidate_k = 20
    a.final_k = 10
    a.hippo_head_k = 10
    a.smoothed_union_k = 10
    a.gamma = 0.15
    a.rerank_mode = "gbc"
    a.gbc_protected_anchor_k = 2
    a.gbc_head_coverage_k = 5
    a.gbc_top_passage_pool_k = 24
    a.gbc_frontier_bonus_k = 6
    a.gbc_bonus_weight = 1.0
    for attr, val in {
        "force_index_from_scratch": "false",
        "force_openie_from_scratch": "false",
        "linking_top_k": 10, "max_qa_steps": 1,
        "embedding_batch_size": 4, "max_retry_attempts": 1,
        "planner_enabled": "false", "planner_mode": "none", "planner_max_steps": 0,
        "causal_enabled": "false", "causal_query_only": "false",
        "causal_gate_mode": "none", "causal_seed_top_k": 8,
        "causal_confidence_threshold": 0.7, "causal_damping": 0.15,
        "causal_blend_dense_weight": 1.0, "causal_blend_fact_weight": 0.0,
        "causal_blend_graph_weight": 0.0, "causal_margin_gate_enabled": "false",
        "causal_margin_threshold": 0.0, "causal_blend_top_k": 10,
        "structure_rerank_enabled": "false", "structure_rerank_top_n": 0,
        "structure_rerank_bonus_weight": 0.0, "structure_rerank_min_edge_support": 0,
        "structure_rerank_max_top5_swaps": 0, "structure_rerank_seed_top_k": 0,
        "structure_rerank_max_hops": 0, "structure_rerank_margin_threshold": 0.0,
        "rerank_require_non_empty": "true",
    }.items():
        setattr(a, attr, val)
    return a


def run_comparison(
    dataset_name,
    limit,
    support_mode="lexical",
    embedding_probe_threshold=0.35,
    extractor_mode="llm_grounded",
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
        candidate_k=args.candidate_k, final_k=args.final_k,
        hippo_head_k=args.hippo_head_k, smoothed_union_k=args.smoothed_union_k,
        gamma=args.gamma, rerank_mode="gbc",
        gbc_protected_anchor_k=args.gbc_protected_anchor_k,
        gbc_head_coverage_k=args.gbc_head_coverage_k,
        gbc_top_passage_pool_k=args.gbc_top_passage_pool_k,
        gbc_frontier_bonus_k=args.gbc_frontier_bonus_k,
        gbc_bonus_weight=args.gbc_bonus_weight,
    )
    ras_cfg = StrongestConfig(
        candidate_k=args.candidate_k, final_k=args.final_k,
        hippo_head_k=args.hippo_head_k, smoothed_union_k=args.smoothed_union_k,
        gamma=args.gamma, rerank_mode="gbc",
        gbc_protected_anchor_k=args.gbc_protected_anchor_k,
        gbc_head_coverage_k=args.gbc_head_coverage_k,
        gbc_top_passage_pool_k=args.gbc_top_passage_pool_k,
        gbc_frontier_bonus_k=args.gbc_frontier_bonus_k,
        gbc_bonus_weight=args.gbc_bonus_weight,
        ras_enabled=True,
        ras_prefix_guard_k=3,
        ras_requirement_max_units=4,
        ras_enable_conflict_veto=True,
        ras_core_support_min_eligible=True,
        ras_extractor_mode=str(extractor_mode),
        ras_support_mode=str(support_mode),
        ras_embedding_probe_threshold=float(embedding_probe_threshold),
    )

    agg = {
        "total": 0, "ras_applied": 0, "ras_fallback": 0,
        "rule_fallback": 0, "empty_after_grounding": 0,
        "has_nonzero_g_support": 0, "has_nonzero_g_core": 0,
        "conflict_veto_count": 0, "prefix_guard_triggered": 0,
        "query_level_repair": 0,
        "top5_changed": 0, "top3_changed": 0, "top1_changed": 0,
        "any_changed": 0,
        "queries_with_support_units": 0,
        "queries_with_core_units": 0,
        "queries_with_parsed_units": 0,
        "queries_with_resolved_anchor": 0,
        "queries_with_resolved_bridge_ref": 0,
        "queries_with_inactive_units": 0,
        "parsed_unit_count_sum": 0,
        "active_unit_count_sum": 0,
        "inactive_unit_count_sum": 0,
        "lexical_anchor_candidate_count_sum": 0,
        "resolved_anchor_count_sum": 0,
        "bridge_ref_candidate_count_sum": 0,
        "bridge_ref_resolved_count_sum": 0,
        "queries_with_unsupported_slots": 0,
        "unsupported_slot_count_sum": 0,
        "skipped_unknown_slot_sum": 0,
    }
    per_query = []

    for result in retrieval_results:
        pool_docs = list(result.docs[:max(args.final_k, args.candidate_k, args.qa_top_k)])
        pool_scores = np.asarray(result.doc_scores[:len(pool_docs)], dtype=np.float32)
        pool_ids = [
            hipporag.passage_node_key_to_doc_idx.get(doc_text_to_chunk_id.get(doc))
            for doc in pool_docs
        ]
        seed_ents = collect_query_seed_entities(hipporag, result.question)
        if not seed_ents:
            seed_ents = collect_lexical_query_seed_entities(
                query=result.question, pool_doc_ids=pool_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )
        query_ents = collect_question_query_entities(
            hipporag=hipporag, query=result.question,
            pool_doc_ids=pool_ids,
            doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
        )

        base_result = run_strongest_shadow_for_pool(
            hipporag=hipporag, query=result.question,
            pool_docs=pool_docs, pool_doc_ids=pool_ids,
            pool_doc_scores=pool_scores,
            seed_entities=seed_ents, query_entities=query_ents,
            config=base_cfg,
        )
        ras_result = run_strongest_shadow_for_pool(
            hipporag=hipporag, query=result.question,
            pool_docs=pool_docs, pool_doc_ids=pool_ids,
            pool_doc_scores=pool_scores,
            seed_entities=seed_ents, query_entities=query_ents,
            config=ras_cfg,
        )

        base_top = base_result.final_doc_indices.tolist() if base_result else []
        ras_top = ras_result.final_doc_indices.tolist() if ras_result else []
        base_titles = [extract_doc_title(pool_docs[i]) for i in base_top[:10]]
        ras_titles = [extract_doc_title(pool_docs[i]) for i in ras_top[:10]]

        ras_trace = (ras_result.trace.get("gbc", {}).get("ras", {}) if ras_result else {})
        ras_status = ras_trace.get("ras_status", "none")
        ext_trace = ras_trace.get("extractor_trace", {})
        grounding_stats = ext_trace.get("grounding_stats", {}) if isinstance(ext_trace, dict) else {}
        extractor_status = ext_trace.get("extractor_status", "")
        extractor_mode_used = ext_trace.get("extractor_mode", "?")
        parsed_unit_count = int(ext_trace.get("parsed_unit_count", grounding_stats.get("parsed_unit_count", 0)) or 0)
        active_unit_count = int(ext_trace.get("active_requirement_unit_count", ext_trace.get("requirement_unit_count", 0)) or 0)
        inactive_unit_count = int(ext_trace.get("inactive_unit_count", grounding_stats.get("inactive_unit_count", 0)) or 0)
        support_count = int(ext_trace.get("support_unit_count", 0) or 0)
        core_count = int(ext_trace.get("core_unit_count", 0) or 0)
        resolved_anchor_count = int(ext_trace.get("resolved_anchor_count", grounding_stats.get("resolved_anchor_count", 0)) or 0)
        lexical_anchor_candidate_count = int(ext_trace.get("lexical_anchor_candidate_count", grounding_stats.get("lexical_anchor_candidate_count", 0)) or 0)
        bridge_ref_candidate_count = int(ext_trace.get("bridge_ref_candidate_count", grounding_stats.get("bridge_ref_candidate_count", 0)) or 0)
        bridge_ref_resolved_count = int(ext_trace.get("bridge_ref_resolved_count", grounding_stats.get("bridge_ref_resolved_count", 0)) or 0)
        unsupported_slot_count = int(ext_trace.get("unsupported_slot_count", grounding_stats.get("unsupported_slot_count", 0)) or 0)
        skipped_unknown_slot = int(ext_trace.get("skipped_unknown_slot", grounding_stats.get("skipped_unknown_slot", 0)) or 0)

        agg["total"] += 1
        agg["ras_applied"] += int(ras_status == "applied")
        agg["ras_fallback"] += int(ras_status != "applied")
        agg["rule_fallback"] += int(str(extractor_mode_used).startswith("rule_fallback"))
        agg["empty_after_grounding"] += int(extractor_status == "empty_after_grounding")
        agg["has_nonzero_g_support"] += int(bool(ras_trace.get("g_support_by_local_idx")))
        agg["has_nonzero_g_core"] += int(bool(ras_trace.get("g_core_by_local_idx")))
        agg["conflict_veto_count"] += int(bool(ras_trace.get("conflict_pairs")))
        agg["prefix_guard_triggered"] += int(bool(ras_trace.get("prefix_guard_triggered")))
        agg["query_level_repair"] += int(bool(ras_trace.get("query_level_repair")))
        agg["top1_changed"] += int(base_top[:1] != ras_top[:1])
        agg["top3_changed"] += int(base_top[:3] != ras_top[:3])
        agg["top5_changed"] += int(base_top[:5] != ras_top[:5])
        agg["any_changed"] += int(base_top != ras_top)
        agg["queries_with_support_units"] += int(support_count > 0)
        agg["queries_with_core_units"] += int(core_count > 0)
        agg["queries_with_parsed_units"] += int(parsed_unit_count > 0)
        agg["queries_with_resolved_anchor"] += int(resolved_anchor_count > 0)
        agg["queries_with_resolved_bridge_ref"] += int(bridge_ref_resolved_count > 0)
        agg["queries_with_inactive_units"] += int(inactive_unit_count > 0)
        agg["parsed_unit_count_sum"] += parsed_unit_count
        agg["active_unit_count_sum"] += active_unit_count
        agg["inactive_unit_count_sum"] += inactive_unit_count
        agg["lexical_anchor_candidate_count_sum"] += lexical_anchor_candidate_count
        agg["resolved_anchor_count_sum"] += resolved_anchor_count
        agg["bridge_ref_candidate_count_sum"] += bridge_ref_candidate_count
        agg["bridge_ref_resolved_count_sum"] += bridge_ref_resolved_count
        agg["queries_with_unsupported_slots"] += int(unsupported_slot_count > 0)
        agg["unsupported_slot_count_sum"] += unsupported_slot_count
        agg["skipped_unknown_slot_sum"] += skipped_unknown_slot

        pq = {
            "query": result.question,
            "ras_status": ras_status,
            "support_mode": ras_trace.get("support_mode", support_mode),
            "extractor_mode": extractor_mode_used,
            "extractor_status": extractor_status,
            "extractor_fallback_reason": ext_trace.get("extractor_fallback_reason"),
            "llm_fallback_reason": ext_trace.get("llm_fallback_reason"),
            "parsed_unit_count": parsed_unit_count,
            "active_unit_count": active_unit_count,
            "inactive_unit_count": inactive_unit_count,
            "support_count": support_count,
            "core_count": core_count,
            "resolved_anchor_count": resolved_anchor_count,
            "lexical_anchor_candidate_count": lexical_anchor_candidate_count,
            "resolved_anchor_rate": float(ext_trace.get("resolved_anchor_rate", grounding_stats.get("resolved_anchor_rate", 0.0)) or 0.0),
            "bridge_ref_candidate_count": bridge_ref_candidate_count,
            "bridge_ref_resolved_count": bridge_ref_resolved_count,
            "bridge_ref_resolved_rate": float(ext_trace.get("bridge_ref_resolved_rate", grounding_stats.get("bridge_ref_resolved_rate", 0.0)) or 0.0),
            "unsupported_slot_count": unsupported_slot_count,
            "unsupported_raw_slots": list(ext_trace.get("unsupported_raw_slots", grounding_stats.get("unsupported_raw_slots", []))),
            "skipped_unknown_slot": skipped_unknown_slot,
            "inactive_reason_distribution": dict(ext_trace.get("inactive_reason_distribution", grounding_stats.get("inactive_reason_distribution", {}))),
            "has_g_support": bool(ras_trace.get("g_support_by_local_idx")),
            "has_g_core": bool(ras_trace.get("g_core_by_local_idx")),
            "top5_changed": base_top[:5] != ras_top[:5],
            "base_top5": base_titles[:5],
            "ras_top5": ras_titles[:5],
        }
        per_query.append(pq)

        if base_top[:5] != ras_top[:5]:
            print(f"  [CHANGED] {result.question[:70]}")
            print(f"    base: {base_titles[:5]}")
            print(f"    ras:  {ras_titles[:5]}")

    return agg, per_query


def print_summary(name, agg):
    n = agg["total"]
    print(f"\n{'='*60}")
    print(f"{name} — RAS(LLM) Activation Summary")
    print(f"{'='*60}")
    for key in ["ras_applied", "ras_fallback", "rule_fallback", "empty_after_grounding",
                 "queries_with_parsed_units", "queries_with_support_units", "queries_with_core_units",
                 "queries_with_resolved_anchor", "queries_with_resolved_bridge_ref", "queries_with_inactive_units",
                 "queries_with_unsupported_slots",
                 "has_nonzero_g_support", "has_nonzero_g_core",
                 "conflict_veto_count", "prefix_guard_triggered", "query_level_repair",
                 "top1_changed", "top3_changed", "top5_changed", "any_changed"]:
        print(f"  {key:30s}: {agg[key]:3d}/{n} = {agg[key]/n:.1%}")
    lexical_total = max(int(agg["lexical_anchor_candidate_count_sum"]), 1)
    bridge_total = max(int(agg["bridge_ref_candidate_count_sum"]), 1)
    print(f"  {'resolved_anchor_rate':30s}: {agg['resolved_anchor_count_sum']:3d}/{lexical_total} = {agg['resolved_anchor_count_sum']/lexical_total:.1%}")
    print(f"  {'bridge_ref_resolved_rate':30s}: {agg['bridge_ref_resolved_count_sum']:3d}/{bridge_total} = {agg['bridge_ref_resolved_count_sum']/bridge_total:.1%}")
    print(f"  {'unsupported_slot_count_sum':30s}: {agg['unsupported_slot_count_sum']}")
    print(f"  {'skipped_unknown_slot_sum':30s}: {agg['skipped_unknown_slot_sum']}")
    print(f"  {'avg_active_units_per_query':30s}: {agg['active_unit_count_sum']/max(n, 1):.2f}")
    print(f"  {'avg_inactive_units_per_query':30s}: {agg['inactive_unit_count_sum']/max(n, 1):.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_limit", type=int, default=40)
    parser.add_argument("--hotpot_limit", type=int, default=20)
    parser.add_argument("--extractor_mode", choices=["rule", "llm", "llm_grounded", "llm_closed_grounded"], default="llm_grounded")
    parser.add_argument("--support_mode", choices=["lexical", "embedding_probe"], default="lexical")
    parser.add_argument("--embedding_probe_threshold", type=float, default=0.35)
    cli_args = parser.parse_args()

    print("Running 2Wiki comparison...")
    agg_2w, pq_2w = run_comparison(
        "2wikimultihopqa",
        cli_args.wiki_limit,
        extractor_mode=cli_args.extractor_mode,
        support_mode=cli_args.support_mode,
        embedding_probe_threshold=cli_args.embedding_probe_threshold,
    )
    print_summary(f"2Wiki-{cli_args.wiki_limit}", agg_2w)

    print("\nRunning Hotpot comparison...")
    agg_hp, pq_hp = run_comparison(
        "hotpotqa",
        cli_args.hotpot_limit,
        extractor_mode=cli_args.extractor_mode,
        support_mode=cli_args.support_mode,
        embedding_probe_threshold=cli_args.embedding_probe_threshold,
    )
    print_summary(f"Hotpot-{cli_args.hotpot_limit}", agg_hp)

    print(f"\n{'='*60}")
    print(f"COMPARISON: Rule-based RAS (before) vs {cli_args.extractor_mode} RAS (now)")
    print(f"{'='*60}")
    print(f"{'':30s}  2Wiki     Hotpot")
    print(f"{'RAS applied':30s}: {agg_2w['ras_applied']}/{agg_2w['total']}      {agg_hp['ras_applied']}/{agg_hp['total']}")
    print(f"{'nonzero g_support':30s}: {agg_2w['has_nonzero_g_support']}/{agg_2w['total']}      {agg_hp['has_nonzero_g_support']}/{agg_hp['total']}")
    print(f"{'nonzero g_core':30s}: {agg_2w['has_nonzero_g_core']}/{agg_2w['total']}      {agg_hp['has_nonzero_g_core']}/{agg_hp['total']}")
    print(f"{'top5 changed':30s}: {agg_2w['top5_changed']}/{agg_2w['total']}      {agg_hp['top5_changed']}/{agg_hp['total']}")

    out_path = "outputs_smoke/ras_llm_audit.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({
        "extractor_mode": cli_args.extractor_mode,
        "support_mode": cli_args.support_mode,
        "embedding_probe_threshold": float(cli_args.embedding_probe_threshold),
        "2wiki": {"agg": agg_2w, "per_query": pq_2w},
        "hotpot": {"agg": agg_hp, "per_query": pq_hp},
    },
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nFull trace saved to {out_path}")
