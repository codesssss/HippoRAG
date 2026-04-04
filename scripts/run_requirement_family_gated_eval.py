#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_requirement_beam_report import build_runtime_args, build_runtime_context  # noqa: E402
from build_requirement_cache import load_dataset, resolve_save_dir  # noqa: E402
from eval_causal_qwen3 import (  # noqa: E402
    build_config,
    build_requirement_title_exposure_summary,
    collect_lexical_query_seed_entities,
    collect_query_seed_entities,
)
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402
from src.hipporag.utils.causal_utils import score_candidate_docs_by_structure  # noqa: E402
from src.hipporag.utils.requirement_family_gate_utils import (  # noqa: E402
    POOL_COVERAGE_GAP_FAMILY,
    STAGED_CLOSURE_GATE_FAMILY,
    UTILITY_REJECTION_FAMILY,
    build_staged_closure_gate_decision,
    classify_requirement_failure_families,
    compute_title_recall_at_k,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _stringify_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    return "true" if text in {"1", "true", "yes"} else "false"


def _question_key(question: str) -> str:
    return str(question or "").strip()


def build_family_rows(report: Dict[str, Any],
                      dataset: str,
                      limit: int,
                      save_dir_root: str) -> Dict[str, Any]:
    runtime_context = build_runtime_context(
        report=report,
        dataset=dataset,
        limit=limit,
        save_dir_root=save_dir_root,
    )
    selector_pool_k = int(
        ((report.get("setwise_selector_qa") or {}).get("pool_k", 0))
        or ((report.get("config") or {}).get("setwise_pool_k", 100))
        or 100
    )
    runtime_by_question = runtime_context["runtime_by_question"]
    query_rows: List[Dict[str, Any]] = []
    for query_trace in report.get("setwise_selector_query_traces", []) or []:
        question = _question_key(query_trace.get("question", ""))
        runtime_row = runtime_by_question.get(question)
        if runtime_row is None:
            continue
        gold_titles = [str(title).strip() for title in query_trace.get("gold_titles", []) if str(title).strip()]
        exposure_rows = build_requirement_title_exposure_summary(
            pool_titles=list((runtime_row.get("pool_titles", []) or [])[:selector_pool_k]),
            selector_trace=query_trace.get("selector_trace"),
            target_titles=gold_titles,
        )
        selector_metrics = dict(query_trace.get("selector_metrics", {}) or {})
        baseline_metrics = dict(query_trace.get("baseline_metrics", {}) or {})
        families = classify_requirement_failure_families(
            exposure_rows=exposure_rows,
            selector_metrics=selector_metrics,
            gold_count=len(gold_titles),
        )
        gate_decision = build_staged_closure_gate_decision(
            question=question,
            exposure_rows=exposure_rows,
            selector_metrics=selector_metrics,
            gold_count=len(gold_titles),
        )
        query_rows.append({
            "question": question,
            "gold_titles": gold_titles,
            "gold_count": len(gold_titles),
            "baseline_metrics": baseline_metrics,
            "selector_metrics": selector_metrics,
            "baseline_answer": query_trace.get("baseline_answer"),
            "selector_answer": query_trace.get("selector_answer"),
            "baseline_top_titles": list(query_trace.get("baseline_top_titles", []) or []),
            "selector_top_titles": list(query_trace.get("selector_top_titles", []) or []),
            "families": families,
            "gold_exposure_rows": exposure_rows,
            "gate_decision": gate_decision,
        })
    return {
        "dataset": dataset,
        "limit": int(limit),
        "runtime_context": runtime_context,
        "query_rows": query_rows,
    }


def materialize_subset_dataset(dataset: str,
                               subset_dataset: str,
                               selected_questions: Sequence[str]) -> Dict[str, Any]:
    corpus, samples = load_dataset(dataset, limit=0)
    selected_question_keys = {_question_key(question) for question in selected_questions}
    subset_samples = [
        sample
        for sample in samples
        if _question_key(sample.get("question", "")) in selected_question_keys
    ]
    subset_path = ROOT_DIR / "reproduce" / "dataset" / f"{subset_dataset}.json"
    subset_corpus_path = ROOT_DIR / "reproduce" / "dataset" / f"{subset_dataset}_corpus.json"
    subset_path.write_text(json.dumps(subset_samples, indent=2, ensure_ascii=False), encoding="utf-8")
    if subset_corpus_path.exists() or subset_corpus_path.is_symlink():
        subset_corpus_path.unlink()
    base_corpus_path = ROOT_DIR / "reproduce" / "dataset" / f"{dataset}_corpus.json"
    subset_corpus_path.symlink_to(base_corpus_path.resolve())
    return {
        "subset_path": subset_path,
        "subset_corpus_path": subset_corpus_path,
        "subset_samples": subset_samples,
    }


def ensure_save_dir_symlink(dataset: str,
                            subset_dataset: str,
                            save_dir_root: str) -> Path:
    base_save_dir = ROOT_DIR / resolve_save_dir(save_dir_root, dataset)
    subset_save_dir = ROOT_DIR / resolve_save_dir(save_dir_root, subset_dataset)
    if subset_save_dir.exists() or subset_save_dir.is_symlink():
        return subset_save_dir
    subset_save_dir.symlink_to(base_save_dir.resolve())
    return subset_save_dir


def build_subset_eval_command(baseline_report: Dict[str, Any],
                              subset_dataset: str,
                              subset_limit: int,
                              output_json: Path,
                              save_dir_root: str,
                              structure_relation_probe_mode: str,
                              structure_continuity_probe_mode: str,
                              structure_seed_target_bridge_mode: str) -> List[str]:
    cfg = dict(baseline_report.get("config", {}) or {})
    qa_block = dict(baseline_report.get("setwise_selector_qa", {}) or {})
    selector_summary = dict(qa_block.get("selector_summary", {}) or {})
    return [
        sys.executable,
        str(ROOT_DIR / "scripts" / "eval_causal_qwen3.py"),
        "--dataset", subset_dataset,
        "--limit", str(int(subset_limit)),
        "--save_dir", str(save_dir_root),
        "--llm_name", str(baseline_report.get("llm_name", "qwen3-8b")),
        "--llm_base_url", str(baseline_report.get("llm_base_url", "http://localhost:8039/v1")),
        "--embedding_name", str(baseline_report.get("embedding_name", "VLLM//mnt/nvme/Qwen3-Embedding-8B")),
        "--embedding_base_url", str(baseline_report.get("embedding_base_url", "http://localhost:8018/v1/embeddings")),
        "--openie_mode", "online",
        "--causal_enabled", _stringify_bool(cfg.get("causal_enabled", False)),
        "--causal_engine_version", str(cfg.get("causal_engine_version", "v2")),
        "--causal_v2_graph_mode", str(cfg.get("causal_v2_graph_mode", "causal")),
        "--causal_v2_base_retrieval_mode", str(cfg.get("causal_v2_base_retrieval_mode", "legacy_fact_graph")),
        "--setwise_selector", str(qa_block.get("selector", cfg.get("setwise_selector", "requirement_beam"))),
        "--setwise_score_mode", str(qa_block.get("score_mode", cfg.get("setwise_score_mode", "bridge"))),
        "--setwise_pool_k", str(int(qa_block.get("pool_k", cfg.get("setwise_pool_k", 100)) or 100)),
        "--qa_top_k", str(int(cfg.get("qa_top_k", 5) or 5)),
        "--setwise_anchor_count", str(int(qa_block.get("anchor_count", cfg.get("setwise_anchor_count", 2)) or 2)),
        "--setwise_reserve_top_m", str(int(qa_block.get("reserve_top_m", cfg.get("setwise_reserve_top_m", 3)) or 3)),
        "--setwise_non_anchor_title_dedup", _stringify_bool(qa_block.get("non_anchor_title_dedup", cfg.get("setwise_non_anchor_title_dedup", True))),
        "--setwise_beam_width", str(int(qa_block.get("beam_width", cfg.get("setwise_beam_width", 4)) or 4)),
        "--setwise_beam_expand_per_state", str(int(qa_block.get("beam_expand_per_state", cfg.get("setwise_beam_expand_per_state", 4)) or 4)),
        "--setwise_beam_projected_shortlist_factor", str(int(qa_block.get("beam_projected_shortlist_factor", cfg.get("setwise_beam_projected_shortlist_factor", 1)) or 1)),
        "--setwise_requirement_cache_path", str(qa_block.get("setwise_requirement_cache_path", selector_summary.get("requirement_cache_path", cfg.get("setwise_requirement_cache_path", "")))),
        "--setwise_requirement_mode", str(qa_block.get("setwise_requirement_mode", selector_summary.get("requirement_mode", cfg.get("setwise_requirement_mode", "oracle")))),
        "--setwise_requirement_annotation_pool_k", str(int(qa_block.get("setwise_requirement_annotation_pool_k", cfg.get("setwise_requirement_annotation_pool_k", 20)) or 20)),
        "--setwise_requirement_live_annotation_pool_k", str(int(qa_block.get("setwise_requirement_live_annotation_pool_k", selector_summary.get("requirement_live_annotation_pool_k", cfg.get("setwise_requirement_live_annotation_pool_k", 100))) or 100)),
        "--setwise_requirement_live_annotation_score_mode", str(qa_block.get("setwise_requirement_live_annotation_score_mode", selector_summary.get("requirement_live_annotation_score_mode", cfg.get("setwise_requirement_live_annotation_score_mode", "hybrid")))),
        "--setwise_requirement_live_atomic_model_path", str(qa_block.get("setwise_requirement_live_atomic_model_path", selector_summary.get("requirement_live_atomic_model_path", cfg.get("setwise_requirement_live_atomic_model_path", "")))),
        "--setwise_requirement_live_source_expand_factor", str(int(qa_block.get("setwise_requirement_live_source_expand_factor", selector_summary.get("requirement_live_source_expand_factor", cfg.get("setwise_requirement_live_source_expand_factor", 3))) or 3)),
        "--setwise_requirement_reserve_policy", str(qa_block.get("setwise_requirement_reserve_policy", selector_summary.get("requirement_reserve_policy", cfg.get("setwise_requirement_reserve_policy", "fixed")))),
        "--structure_relation_probe_mode", str(structure_relation_probe_mode),
        "--structure_continuity_probe_mode", str(structure_continuity_probe_mode),
        "--structure_seed_target_bridge_mode", str(structure_seed_target_bridge_mode),
        "--output_json", str(output_json),
    ]


def build_probe_hipporag(report: Dict[str, Any],
                         dataset: str,
                         save_dir_root: str,
                         corpus: Sequence[Dict[str, Any]],
                         structure_relation_probe_mode: str,
                         structure_continuity_probe_mode: str) -> HippoRAG:
    runtime_args = build_runtime_args(report=report, dataset=dataset, save_dir_root=save_dir_root)
    cfg = dict(report.get("config", {}) or {})
    default_config_values = {
        "force_index_from_scratch": "false",
        "force_openie_from_scratch": "false",
        "max_retry_attempts": 5,
        "openie_mode": "online",
        "retrieval_top_k": 200,
        "linking_top_k": 5,
        "qa_top_k": int(cfg.get("qa_top_k", 5) or 5),
        "max_qa_steps": 3,
        "embedding_batch_size": 8,
        "planner_enabled": "false",
        "planner_mode": "none",
        "planner_max_steps": 3,
        "causal_enabled": "false",
        "causal_query_only": "true",
        "causal_gate_mode": "hard",
        "causal_seed_top_k": 20,
        "causal_confidence_threshold": 0.5,
        "causal_damping": 0.7,
        "causal_blend_dense_weight": 0.35,
        "causal_blend_fact_weight": 0.15,
        "causal_blend_graph_weight": 0.5,
        "causal_margin_gate_enabled": "false",
        "causal_margin_threshold": 0.02,
        "causal_blend_top_k": 0,
        "causal_engine_version": "v2",
        "causal_v2_probe_mode": "router",
        "causal_v2_graph_mode": "causal",
        "causal_v2_base_retrieval_mode": "legacy_fact_graph",
        "general_graph_related_to_weight": 0.3,
        "general_graph_seed_top_k": 10,
        "causal_v2_extraction_max_tokens": 768,
        "causal_v2_extraction_retry_attempts": 2,
        "causal_v2_extraction_workers": 4,
        "causal_event_top_k": 8,
        "causal_v2_max_hops": 2,
        "causal_chain_top_k": 6,
        "causal_context_max_items": 0,
        "causal_er_similarity_threshold": 0.92,
        "causal_er_text_threshold": 0.55,
        "causal_v2_min_edge_confidence": 0.7,
        "structure_rerank_enabled": "true",
        "structure_rerank_top_n": 40,
        "structure_rerank_bonus_weight": 0.08,
        "structure_rerank_min_edge_support": 2,
        "structure_rerank_max_top5_swaps": 2,
        "structure_rerank_seed_top_k": 4,
        "structure_rerank_max_hops": 2,
        "structure_rerank_margin_threshold": 0.02,
        "rerank_require_non_empty": "true",
    }
    for attr, default in default_config_values.items():
        setattr(runtime_args, attr, cfg.get(attr, default))
    runtime_args.structure_relation_probe_mode = structure_relation_probe_mode
    runtime_args.structure_continuity_probe_mode = structure_continuity_probe_mode
    runtime_args.structure_seed_target_bridge_mode = "off"
    config = build_config(runtime_args, corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    hipporag.index(docs)
    return hipporag


def compute_probe_trigger_summary(query_questions: Sequence[str],
                                  runtime_context: Dict[str, Any],
                                  probe_hipporag: HippoRAG,
                                  structure_max_hops: int,
                                  selector_pool_k: int) -> Dict[str, Dict[str, Any]]:
    runtime_by_question = runtime_context["runtime_by_question"]
    query_payload: Dict[str, Dict[str, Any]] = {}
    for question in query_questions:
        runtime_row = runtime_by_question.get(question)
        if runtime_row is None:
            continue
        pool_doc_ids = [
            int(doc_id)
            for doc_id in (runtime_row.get("pool_doc_ids", []) or [])[:selector_pool_k]
            if doc_id is not None
        ]
        relation_triggered = any(
            any(str(relation_type).startswith("factual_") for _, _, _, relation_type in probe_hipporag.doc_idx_to_structure_edges.get(doc_id, []))
            for doc_id in pool_doc_ids
        )
        continuity_triggered = any(
            any(str(relation_type) == "alias_city_state" for _, _, _, relation_type in probe_hipporag.doc_idx_to_structure_edges.get(doc_id, []))
            for doc_id in pool_doc_ids
        )
        seed_entities = collect_query_seed_entities(probe_hipporag, question)
        if not seed_entities:
            seed_entities = collect_lexical_query_seed_entities(
                query=question,
                pool_doc_ids=list((runtime_row.get("pool_doc_ids", []) or [])[:selector_pool_k]),
                doc_idx_to_entities=probe_hipporag.doc_idx_to_structure_entities,
            )
        if pool_doc_ids and seed_entities:
            structure_scores_off = score_candidate_docs_by_structure(
                candidate_doc_ids=pool_doc_ids,
                doc_idx_to_entities=probe_hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=probe_hipporag.doc_idx_to_structure_edges,
                seed_entities=set(seed_entities),
                adjacency=probe_hipporag.structure_graph_out,
                max_hops=structure_max_hops,
                seed_target_bridge_mode="off",
            )
            structure_scores_on = score_candidate_docs_by_structure(
                candidate_doc_ids=pool_doc_ids,
                doc_idx_to_entities=probe_hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=probe_hipporag.doc_idx_to_structure_edges,
                seed_entities=set(seed_entities),
                adjacency=probe_hipporag.structure_graph_out,
                max_hops=structure_max_hops,
                seed_target_bridge_mode="allow_seed_target",
            )
            seed_target_triggered = any(
                _safe_float(structure_scores_on.get(doc_id)) > _safe_float(structure_scores_off.get(doc_id)) + 1e-9
                for doc_id in pool_doc_ids
            )
        else:
            seed_target_triggered = False
        query_payload[question] = {
            "relation_probe_triggered": bool(relation_triggered),
            "continuity_probe_triggered": bool(continuity_triggered),
            "seed_target_bridge_triggered": bool(seed_target_triggered),
        }
    return query_payload


def summarize_query_metrics(query_rows: Sequence[Dict[str, Any]],
                            trace_by_question: Dict[str, Dict[str, Any]],
                            runtime_by_question: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    questions = [_question_key(row.get("question", "")) for row in query_rows]
    traces = [trace_by_question[question] for question in questions if question in trace_by_question]
    em = _mean([_safe_float(trace.get("selector_metrics", {}).get("ExactMatch")) for trace in traces])
    f1 = _mean([_safe_float(trace.get("selector_metrics", {}).get("F1")) for trace in traces])
    baseline_em = _mean([_safe_float(trace.get("baseline_metrics", {}).get("ExactMatch")) for trace in traces])
    baseline_f1 = _mean([_safe_float(trace.get("baseline_metrics", {}).get("F1")) for trace in traces])
    recall5 = _mean([
        compute_title_recall_at_k(
            pool_titles=list((runtime_by_question.get(_question_key(row.get("question", "")), {}) or {}).get("pool_titles", []) or []),
            gold_titles=list(row.get("gold_titles", []) or []),
            k=5,
        )
        for row in query_rows
    ])
    recall10 = _mean([
        compute_title_recall_at_k(
            pool_titles=list((runtime_by_question.get(_question_key(row.get("question", "")), {}) or {}).get("pool_titles", []) or []),
            gold_titles=list(row.get("gold_titles", []) or []),
            k=10,
        )
        for row in query_rows
    ])
    return {
        "count": int(len(query_rows)),
        "baseline_EM": round(baseline_em, 4),
        "baseline_F1": round(baseline_f1, 4),
        "gated_EM": round(em, 4),
        "gated_F1": round(f1, 4),
        "delta_EM": round(em - baseline_em, 4),
        "delta_F1": round(f1 - baseline_f1, 4),
        "Recall@5": round(recall5, 4),
        "Recall@10": round(recall10, 4),
    }


def build_stop_loss_decision(overall_summary: Dict[str, float],
                             staged_summary: Dict[str, float],
                             utility_summary: Dict[str, float],
                             non_staged_summary: Dict[str, float]) -> Dict[str, Any]:
    overall_gap = round(overall_summary["baseline_F1"] - overall_summary["gated_F1"], 4)
    staged_delta = float(staged_summary.get("delta_F1", 0.0) or 0.0)
    utility_delta = float(utility_summary.get("delta_F1", 0.0) or 0.0)
    non_staged_delta = float(non_staged_summary.get("delta_F1", 0.0) or 0.0)

    if overall_gap <= 0.01 and staged_delta >= 0.05 and utility_delta >= -0.02 and non_staged_delta >= -0.01:
        verdict = "值得继续"
        rationale = "smoke40 F1 已接近 baseline，staged-closure 子集明确正向，且未观察到明显全局副作用。"
    elif overall_gap <= 0.03 and staged_delta >= 0.05 and utility_delta >= -0.03 and non_staged_delta >= -0.03:
        verdict = "可再试一天"
        rationale = "总体仍略低于 baseline，但 gap 已收窄到可接受窗口，且 staged-closure 子集收益明确。"
    else:
        verdict = "应切线"
        rationale = "总体 F1 相对 baseline 的缺口仍过大，或 staged-closure 收益不足以抵消非 closure 子集上的副作用。"

    return {
        "verdict": verdict,
        "overall_f1_gap_to_baseline": overall_gap,
        "staged_closure_delta_f1": round(staged_delta, 4),
        "utility_rejection_delta_f1": round(utility_delta, 4),
        "non_staged_delta_f1": round(non_staged_delta, 4),
        "rationale": rationale,
    }


def build_markdown_report(payload: Dict[str, Any]) -> str:
    overall = payload["overall_summary"]
    staged = payload["subset_summaries"]["staged_closure"]
    utility = payload["subset_summaries"]["utility_rejection"]
    pool_limited = payload["subset_summaries"]["pool_limited"]
    decision = payload["stop_loss_decision"]
    lines = ["# Requirement Family-Aware Gated Smoke40", ""]
    lines.append("## Gate")
    lines.append("")
    lines.append(f"- mode: `{payload['gate_mode']}`")
    lines.append(f"- gate_hits: `{payload['gate_hit_count']}` / `{payload['num_queries']}`")
    lines.append(f"- subset_eval_report: `{payload['subset_eval_report']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Slice | Baseline EM | Gated EM | ΔEM | Baseline F1 | Gated F1 | ΔF1 | R@5 | R@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| overall | {overall['baseline_EM']:.4f} | {overall['gated_EM']:.4f} | {overall['delta_EM']:+.4f} | "
        f"{overall['baseline_F1']:.4f} | {overall['gated_F1']:.4f} | {overall['delta_F1']:+.4f} | "
        f"{overall['Recall@5']:.4f} | {overall['Recall@10']:.4f} |"
    )
    lines.append(
        f"| staged_closure | {staged['baseline_EM']:.4f} | {staged['gated_EM']:.4f} | {staged['delta_EM']:+.4f} | "
        f"{staged['baseline_F1']:.4f} | {staged['gated_F1']:.4f} | {staged['delta_F1']:+.4f} | "
        f"{staged['Recall@5']:.4f} | {staged['Recall@10']:.4f} |"
    )
    lines.append(
        f"| utility_rejection | {utility['baseline_EM']:.4f} | {utility['gated_EM']:.4f} | {utility['delta_EM']:+.4f} | "
        f"{utility['baseline_F1']:.4f} | {utility['gated_F1']:.4f} | {utility['delta_F1']:+.4f} | "
        f"{utility['Recall@5']:.4f} | {utility['Recall@10']:.4f} |"
    )
    lines.append(
        f"| pool_limited | {pool_limited['baseline_EM']:.4f} | {pool_limited['gated_EM']:.4f} | {pool_limited['delta_EM']:+.4f} | "
        f"{pool_limited['baseline_F1']:.4f} | {pool_limited['gated_F1']:.4f} | {pool_limited['delta_F1']:+.4f} | "
        f"{pool_limited['Recall@5']:.4f} | {pool_limited['Recall@10']:.4f} |"
    )
    lines.append("")
    lines.append("## Trigger Counts")
    lines.append("")
    trigger_counts = payload["actual_trigger_counts"]
    lines.append(f"- `q6_factual` triggered on `{trigger_counts['relation_probe_triggered']}` gated queries")
    lines.append(f"- `city_state_alias` triggered on `{trigger_counts['continuity_probe_triggered']}` gated queries")
    lines.append(f"- `allow_seed_target` triggered on `{trigger_counts['seed_target_bridge_triggered']}` gated queries")
    lines.append("")
    lines.append("## Gate-Hit Queries")
    lines.append("")
    gate_queries = payload["gate_queries"]
    if not gate_queries:
        lines.append("- none")
    else:
        for row in gate_queries:
            lines.append(f"- question: {row['question']}")
            lines.append(f"  reasons: {row['gate_reasons']}")
            lines.append(f"  families: {row['families']}")
            lines.append(f"  baseline: {row['baseline_metrics']} / gated: {row['gated_metrics']}")
            lines.append(f"  triggers: {row['actual_triggers']}")
            lines.append(f"  baseline_top: {row['baseline_top_titles']}")
            lines.append(f"  gated_top: {row['gated_top_titles']}")
    lines.append("")
    lines.append("## Stop-Loss")
    lines.append("")
    lines.append(f"- verdict: `{decision['verdict']}`")
    lines.append(f"- rationale: {decision['rationale']}")
    lines.append("")
    lines.append("## Subset Eval Command")
    lines.append("")
    lines.append("```bash")
    lines.append(" ".join(payload["subset_eval_command"]))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run family-aware gated requirement_beam smoke eval by rerunning only conservative staged-closure candidates.")
    parser.add_argument("--baseline_report", required=True, type=str)
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--save_dir_root", type=str, default="outputs_step0_general")
    parser.add_argument("--gate_mode", choices=["off", "conservative_staged_closure"], default="conservative_staged_closure")
    parser.add_argument("--structure_relation_probe_mode", choices=["off", "q6_factual"], default="q6_factual")
    parser.add_argument("--structure_continuity_probe_mode", choices=["off", "city_state_alias"], default="city_state_alias")
    parser.add_argument("--structure_seed_target_bridge_mode", choices=["off", "allow_seed_target"], default="allow_seed_target")
    parser.add_argument("--subset_dataset", type=str, default="")
    parser.add_argument("--subset_output_json", type=str, default="")
    parser.add_argument("--force_rerun_subset", action="store_true")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--output_md", type=str, default="")
    args = parser.parse_args()

    baseline_report_path = Path(args.baseline_report)
    baseline_report = json.loads(baseline_report_path.read_text())

    family_payload = build_family_rows(
        report=baseline_report,
        dataset=str(args.dataset),
        limit=int(args.limit),
        save_dir_root=str(args.save_dir_root),
    )
    runtime_context = family_payload["runtime_context"]
    query_rows = family_payload["query_rows"]
    runtime_by_question = runtime_context["runtime_by_question"]
    gate_rows = [
        row for row in query_rows
        if args.gate_mode != "off" and bool((row.get("gate_decision") or {}).get("gate_hit", False))
    ]
    gate_questions = [_question_key(row["question"]) for row in gate_rows]
    base_eval_dir = ROOT_DIR / resolve_save_dir(str(args.save_dir_root), str(args.dataset)) / "eval_reports"
    base_eval_dir.mkdir(parents=True, exist_ok=True)

    subset_dataset = str(args.subset_dataset or f"{args.dataset}_smoke{int(args.limit)}_familygate_20260404")
    subset_artifacts = materialize_subset_dataset(
        dataset=str(args.dataset),
        subset_dataset=subset_dataset,
        selected_questions=gate_questions,
    )
    ensure_save_dir_symlink(
        dataset=str(args.dataset),
        subset_dataset=subset_dataset,
        save_dir_root=str(args.save_dir_root),
    )

    subset_output_json = Path(args.subset_output_json) if args.subset_output_json else (
        base_eval_dir
        / f"requirement_beam_needunit_oracle_smoke{int(args.limit)}_familygate_subset_e_20260404.json"
    )
    subset_output_json.parent.mkdir(parents=True, exist_ok=True)
    subset_eval_command = build_subset_eval_command(
        baseline_report=baseline_report,
        subset_dataset=subset_dataset,
        subset_limit=len(subset_artifacts["subset_samples"]),
        output_json=subset_output_json,
        save_dir_root=str(args.save_dir_root),
        structure_relation_probe_mode=str(args.structure_relation_probe_mode),
        structure_continuity_probe_mode=str(args.structure_continuity_probe_mode),
        structure_seed_target_bridge_mode=str(args.structure_seed_target_bridge_mode),
    )
    if gate_questions and subset_output_json.exists() and not args.force_rerun_subset:
        subset_report = json.loads(subset_output_json.read_text())
        gated_trace_by_question = {
            _question_key(trace.get("question", "")): trace
            for trace in subset_report.get("setwise_selector_query_traces", []) or []
        }
    elif gate_questions:
        subprocess.run(subset_eval_command, cwd=str(ROOT_DIR), check=True)
        subset_report = json.loads(subset_output_json.read_text())
        gated_trace_by_question = {
            _question_key(trace.get("question", "")): trace
            for trace in subset_report.get("setwise_selector_query_traces", []) or []
        }
    else:
        subset_report = {"setwise_selector_query_traces": []}
        gated_trace_by_question = {}

    baseline_trace_by_question = {
        _question_key(trace.get("question", "")): trace
        for trace in baseline_report.get("setwise_selector_query_traces", []) or []
    }
    composite_trace_by_question: Dict[str, Dict[str, Any]] = {}
    for question, baseline_trace in baseline_trace_by_question.items():
        composite_trace_by_question[question] = gated_trace_by_question.get(question, baseline_trace)

    probe_hipporag = build_probe_hipporag(
        report=baseline_report,
        dataset=str(args.dataset),
        save_dir_root=str(args.save_dir_root),
        corpus=runtime_context["corpus"],
        structure_relation_probe_mode=str(args.structure_relation_probe_mode),
        structure_continuity_probe_mode=str(args.structure_continuity_probe_mode),
    )
    structure_max_hops = int((baseline_report.get("config") or {}).get("setwise_structure_max_hops", 2) or 2)
    selector_pool_k = int(
        ((baseline_report.get("setwise_selector_qa") or {}).get("pool_k", 0))
        or ((baseline_report.get("config") or {}).get("setwise_pool_k", 100))
        or 100
    )
    trigger_by_question = compute_probe_trigger_summary(
        query_questions=gate_questions,
        runtime_context=runtime_context,
        probe_hipporag=probe_hipporag,
        structure_max_hops=structure_max_hops,
        selector_pool_k=selector_pool_k,
    )

    staged_rows = [row for row in query_rows if STAGED_CLOSURE_GATE_FAMILY in row.get("families", [])]
    utility_rows = [row for row in query_rows if UTILITY_REJECTION_FAMILY in row.get("families", [])]
    pool_limited_rows = [row for row in query_rows if POOL_COVERAGE_GAP_FAMILY in row.get("families", [])]
    non_staged_rows = [row for row in query_rows if STAGED_CLOSURE_GATE_FAMILY not in row.get("families", [])]

    overall_summary = summarize_query_metrics(
        query_rows=query_rows,
        trace_by_question=composite_trace_by_question,
        runtime_by_question=runtime_by_question,
    )
    staged_summary = summarize_query_metrics(
        query_rows=staged_rows,
        trace_by_question=composite_trace_by_question,
        runtime_by_question=runtime_by_question,
    )
    utility_summary = summarize_query_metrics(
        query_rows=utility_rows,
        trace_by_question=composite_trace_by_question,
        runtime_by_question=runtime_by_question,
    )
    pool_limited_summary = summarize_query_metrics(
        query_rows=pool_limited_rows,
        trace_by_question=composite_trace_by_question,
        runtime_by_question=runtime_by_question,
    )
    non_staged_summary = summarize_query_metrics(
        query_rows=non_staged_rows,
        trace_by_question=composite_trace_by_question,
        runtime_by_question=runtime_by_question,
    )

    gate_queries_payload: List[Dict[str, Any]] = []
    for row in gate_rows:
        question = _question_key(row["question"])
        baseline_trace = baseline_trace_by_question.get(question, {})
        gated_trace = composite_trace_by_question.get(question, {})
        gate_queries_payload.append({
            "question": question,
            "gate_reasons": list((row.get("gate_decision") or {}).get("gate_reasons", []) or []),
            "gate_blockers": list((row.get("gate_decision") or {}).get("gate_blockers", []) or []),
            "families": list(row.get("families", []) or []),
            "baseline_metrics": dict(baseline_trace.get("selector_metrics", {}) or {}),
            "gated_metrics": dict(gated_trace.get("selector_metrics", {}) or {}),
            "baseline_top_titles": list(baseline_trace.get("selector_top_titles", []) or []),
            "gated_top_titles": list(gated_trace.get("selector_top_titles", []) or []),
            "actual_triggers": trigger_by_question.get(question, {
                "relation_probe_triggered": False,
                "continuity_probe_triggered": False,
                "seed_target_bridge_triggered": False,
            }),
        })

    actual_trigger_counts = {
        "relation_probe_triggered": int(sum(1 for payload in gate_queries_payload if payload["actual_triggers"]["relation_probe_triggered"])),
        "continuity_probe_triggered": int(sum(1 for payload in gate_queries_payload if payload["actual_triggers"]["continuity_probe_triggered"])),
        "seed_target_bridge_triggered": int(sum(1 for payload in gate_queries_payload if payload["actual_triggers"]["seed_target_bridge_triggered"])),
    }

    stop_loss_decision = build_stop_loss_decision(
        overall_summary=overall_summary,
        staged_summary=staged_summary,
        utility_summary=utility_summary,
        non_staged_summary=non_staged_summary,
    )

    output_json = Path(args.output_json) if args.output_json else (
        base_eval_dir
        / f"requirement_beam_needunit_oracle_smoke{int(args.limit)}_familygate_conservative_20260404.json"
    )
    output_md = Path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "baseline_report": str(baseline_report_path),
        "subset_eval_report": str(subset_output_json),
        "dataset": str(args.dataset),
        "limit": int(args.limit),
        "gate_mode": str(args.gate_mode),
        "gate_hit_count": int(len(gate_rows)),
        "num_queries": int(len(query_rows)),
        "subset_dataset": subset_dataset,
        "subset_dataset_path": str(subset_artifacts["subset_path"]),
        "subset_corpus_path": str(subset_artifacts["subset_corpus_path"]),
        "subset_eval_command": subset_eval_command,
        "enabled_repairs": {
            "structure_relation_probe_mode": str(args.structure_relation_probe_mode),
            "structure_continuity_probe_mode": str(args.structure_continuity_probe_mode),
            "structure_seed_target_bridge_mode": str(args.structure_seed_target_bridge_mode),
        },
        "actual_trigger_counts": actual_trigger_counts,
        "overall_summary": overall_summary,
        "subset_summaries": {
            "staged_closure": staged_summary,
            "utility_rejection": utility_summary,
            "pool_limited": pool_limited_summary,
            "non_staged_closure": non_staged_summary,
        },
        "gate_queries": gate_queries_payload,
        "query_rows": query_rows,
        "stop_loss_decision": stop_loss_decision,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(build_markdown_report(payload), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(output_md),
        "gate_hit_count": len(gate_rows),
        "overall_summary": overall_summary,
        "stop_loss_decision": stop_loss_decision,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
