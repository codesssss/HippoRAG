#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# This audit is fully local and should not inherit workstation proxy settings.
# Some optional deps imported by the main eval module initialize HTTP clients
# at import time, which can fail under SOCKS proxy envs without socks extras.
for _proxy_env in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_proxy_env, None)

from eval_causal_qwen3 import score_action_swap_tiered_witness_jobs


DEFAULT_CE_MODEL = "/mnt/nvme/bge-reranker-v2-m3"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_question(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe_pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(100.0 * float(numerator) / float(denominator), 2)


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 4)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 4)


def _extract_candidate_doc_text(action_row: Mapping[str, Any]) -> str:
    baseline_docs = list(action_row.get("baseline_docs") or [])
    swapped_docs = list(action_row.get("docs") or [])
    replace_index = action_row.get("replace_incumbent_index")
    if replace_index is not None:
        idx = int(replace_index)
        if 0 <= idx < len(swapped_docs):
            return str(swapped_docs[idx])
    for base_doc, swapped_doc in zip(baseline_docs, swapped_docs):
        if str(base_doc) != str(swapped_doc):
            return str(swapped_doc)
    return ""


def _query_context_map(report_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in list(report_payload.get("expand_assemble_query_traces") or []):
        question_key = _normalize_question(row.get("question", ""))
        if not question_key:
            continue
        expand_trace = dict(row.get("expand_assemble_trace") or {})
        mapping[question_key] = {
            "query_type": row.get("query_type"),
            "grounded_question_entities_preview": list(expand_trace.get("grounded_question_entities_preview") or []),
            "question_entities_preview": list(expand_trace.get("question_entities_preview") or []),
            "query_entities_preview": list(expand_trace.get("query_entities_preview") or []),
            "proposal_query_entities_preview": list(expand_trace.get("proposal_query_entities_preview") or []),
        }
    return mapping


def _extract_policy_actions(report_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    action_map: Dict[str, Dict[str, Any]] = {}
    for row in list(report_payload.get("expand_assemble_query_traces") or []):
        question_key = _normalize_question(row.get("question", ""))
        if not question_key:
            continue
        expand_trace = dict(row.get("expand_assemble_trace") or {})
        if bool(expand_trace.get("action_executed")) and expand_trace.get("action_candidate_pool_position") is not None:
            action_map[question_key] = {
                "action_type": "swap",
                "candidate_pool_position": int(expand_trace.get("action_candidate_pool_position")),
                "replace_pool_position": int(expand_trace.get("action_replace_pool_position")),
            }
        else:
            action_map[question_key] = {
                "action_type": "keep",
                "candidate_pool_position": None,
                "replace_pool_position": None,
            }
    return action_map


def _action_key(row: Mapping[str, Any]) -> Tuple[str, int | None, int | None]:
    action_type = str(row.get("action_type") or "keep").strip().lower()
    if action_type != "swap":
        return ("keep", None, None)
    candidate = row.get("candidate_pool_position")
    replace = row.get("replace_pool_position")
    return (
        "swap",
        int(candidate) if candidate is not None else None,
        int(replace) if replace is not None else None,
    )


def _policy_selected(question_key: str,
                     policy_actions: Mapping[str, Mapping[str, Any]],
                     action_row: Mapping[str, Any]) -> bool:
    policy_row = dict(policy_actions.get(question_key) or {})
    return (
        str(policy_row.get("action_type") or "").strip().lower() == "swap"
        and int(policy_row.get("candidate_pool_position")) == int(action_row.get("candidate_pool_position"))
        and int(policy_row.get("replace_pool_position")) == int(action_row.get("replace_pool_position"))
    )


def _load_ce_reranker(model_name: str, device: str):
    from FlagEmbedding import FlagReranker

    return FlagReranker(model_name, use_fp16=True, devices=[str(device)])


def _select_action_rows(oracle_payload: Mapping[str, Any],
                        *,
                        dryrun_actions: Mapping[str, Mapping[str, Any]],
                        judge_actions: Mapping[str, Mapping[str, Any]],
                        negative_sample_size: int,
                        seed: int) -> List[Dict[str, Any]]:
    legal_swap_rows: List[Dict[str, Any]] = []
    for row in list(oracle_payload.get("action_results") or []):
        if str(row.get("action_type") or "").strip().lower() != "swap" or not bool(row.get("is_legal")):
            continue
        materialized = dict(row)
        delta_metrics = dict(materialized.get("delta_metrics") or {})
        materialized["oracle_positive"] = bool(
            float(delta_metrics.get("F1", 0.0) or 0.0) > 0.0
            or float(delta_metrics.get("ExactMatch", 0.0) or 0.0) > 0.0
        )
        question_key = _normalize_question(materialized.get("question", ""))
        materialized["selected_by_dryrun"] = _policy_selected(question_key, dryrun_actions, materialized)
        materialized["selected_by_judge"] = _policy_selected(question_key, judge_actions, materialized)
        legal_swap_rows.append(materialized)

    positive_rows = [row for row in legal_swap_rows if bool(row.get("oracle_positive"))]
    policy_rows = [row for row in legal_swap_rows if row.get("selected_by_dryrun") or row.get("selected_by_judge")]
    selected_keys = {
        (_normalize_question(row.get("question", "")), _action_key(row))
        for row in positive_rows + policy_rows
    }
    negative_candidates = [
        row for row in legal_swap_rows
        if not bool(row.get("oracle_positive"))
        and (_normalize_question(row.get("question", "")), _action_key(row)) not in selected_keys
    ]
    random.Random(int(seed)).shuffle(negative_candidates)
    sampled_negative_rows = negative_candidates[:max(int(negative_sample_size), 0)]

    combined_rows: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, Tuple[str, int | None, int | None]]] = set()
    for row in positive_rows + policy_rows + sampled_negative_rows:
        key = (_normalize_question(row.get("question", "")), _action_key(row))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        combined_rows.append(dict(row))
    return combined_rows


def _compute_query_audit_rows(action_rows: Sequence[Mapping[str, Any]],
                              *,
                              query_context: Mapping[str, Any],
                              ce_reranker: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not action_rows:
        return [], {}
    question = str(action_rows[0].get("question") or "")
    baseline_docs = list(action_rows[0].get("baseline_docs") or [])
    query_entities = (
        list(query_context.get("grounded_question_entities_preview") or [])
        or list(query_context.get("question_entities_preview") or [])
        or list(query_context.get("query_entities_preview") or [])
        or list(query_context.get("proposal_query_entities_preview") or [])
    )

    local_docs = [str(doc) for doc in baseline_docs]
    candidate_local_pos: Dict[str, int] = {}
    for row in action_rows:
        candidate_doc = _extract_candidate_doc_text(row)
        candidate_key = str(row.get("candidate_pool_position"))
        if candidate_doc and candidate_key not in candidate_local_pos:
            candidate_local_pos[candidate_key] = len(local_docs)
            local_docs.append(str(candidate_doc))

    scaffold_positions = list(range(len(baseline_docs)))
    local_action_jobs: List[Dict[str, Any]] = []
    for row in action_rows:
        candidate_key = str(row.get("candidate_pool_position"))
        if candidate_key not in candidate_local_pos:
            continue
        replace_index = int(row.get("replace_incumbent_index"))
        swapped_positions = list(scaffold_positions)
        swapped_positions[replace_index] = int(candidate_local_pos[candidate_key])
        local_action_jobs.append({
            "question": question,
            "action_type": "swap",
            "candidate_pool_position": int(candidate_local_pos[candidate_key]),
            "candidate_doc_id": row.get("candidate_doc_id"),
            "replace_pool_position": int(replace_index),
            "replace_doc_id": row.get("replace_doc_id"),
            "candidate_assemble_score": float(row.get("candidate_assemble_score", row.get("candidate_ce_score", 0.0)) or 0.0),
            "score_delta": float(row.get("score_delta", 0.0) or 0.0),
            "swapped_positions": list(swapped_positions),
            "is_duplicate_with_scaffold": False,
            "oracle_candidate_pool_position": int(row.get("candidate_pool_position")),
            "oracle_replace_pool_position": int(row.get("replace_pool_position")),
            "oracle_candidate_title": row.get("candidate_title"),
            "oracle_replace_title": row.get("replace_incumbent_title"),
            "oracle_delta_em": float((row.get("delta_metrics") or {}).get("ExactMatch", 0.0) or 0.0),
            "oracle_delta_f1": float((row.get("delta_metrics") or {}).get("F1", 0.0) or 0.0),
            "oracle_positive": bool(row.get("oracle_positive")),
            "selected_by_dryrun": bool(row.get("selected_by_dryrun")),
            "selected_by_judge": bool(row.get("selected_by_judge")),
            "query_type": row.get("query_type"),
        })

    scored_bundle = score_action_swap_tiered_witness_jobs(
        local_action_jobs,
        query=question,
        scaffold_positions=scaffold_positions,
        pool_docs=local_docs,
        query_entities=query_entities,
        ce_reranker=ce_reranker,
    )
    audit_rows: List[Dict[str, Any]] = []
    for row in list(scored_bundle.get("scored_jobs") or []):
        audit_rows.append({
            "dataset": None,
            "question": question,
            "action_type": "swap",
            "query_type": row.get("query_type"),
            "oracle_label": "positive" if bool(row.get("oracle_positive")) else "negative",
            "selected_by_dryrun": bool(row.get("selected_by_dryrun")),
            "selected_by_judge": bool(row.get("selected_by_judge")),
            "query_tier_mode": row.get("query_tier_mode"),
            "query_tiers": row.get("query_tiers"),
            "bottleneck_tier_index": row.get("bottleneck_tier_index"),
            "target_facet_id": row.get("target_facet_id"),
            "target_facet_text": row.get("target_facet_text"),
            "candidate_pool_position": row.get("oracle_candidate_pool_position"),
            "replace_pool_position": row.get("oracle_replace_pool_position"),
            "candidate_title": row.get("oracle_candidate_title"),
            "replace_incumbent_title": row.get("oracle_replace_title"),
            "score_delta": _round(row.get("score_delta")),
            "oracle_delta_em": _round(row.get("oracle_delta_em")),
            "oracle_delta_f1": _round(row.get("oracle_delta_f1")),
            "target_candidate_gain": _round(row.get("target_candidate_gain")),
            "replacee_loss_earlier_tiers": _round(row.get("replacee_loss_earlier_tiers")),
            "replacee_loss_same_tier": _round(row.get("replacee_loss_same_tier")),
            "replacee_loss_target_facet": _round(row.get("replacee_loss_target_facet")),
            "swap_gain_vs_loss": _round(row.get("swap_gain_vs_loss")),
            "action_should_swap": bool(row.get("action_should_swap")),
            "candidate_best_witness": row.get("candidate_best_witness"),
            "facet_supports_before": row.get("facet_supports_before"),
            "facet_supports_after": row.get("facet_supports_after"),
        })

    ranking = sorted(
        [
            ((_action_key(row)), float(row.get("swap_gain_vs_loss", 0.0) or 0.0), row.get("oracle_label") == "positive")
            for row in audit_rows
        ],
        key=lambda item: (float(item[1]), item[2]),
        reverse=True,
    )
    query_summary = {
        "question": question,
        "query_tier_mode": str(scored_bundle.get("query_tier_mode") or "flat_fallback"),
        "ranking_tiered_witness": ranking,
    }
    return audit_rows, query_summary


def _topk_positive_recall(query_summaries: Sequence[Mapping[str, Any]], ranking_key: str, *, k: int) -> Tuple[int, int]:
    hit_count = 0
    total_count = 0
    for summary in query_summaries:
        ranking = list(summary.get(ranking_key) or [])
        if not ranking:
            continue
        total_count += 1
        if any(bool(item[2]) for item in ranking[:max(int(k), 0)]):
            hit_count += 1
    return hit_count, total_count


def _build_summary(dataset: str,
                   audit_rows: Sequence[Mapping[str, Any]],
                   query_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positive_rows = [row for row in audit_rows if row.get("oracle_label") == "positive"]
    negative_rows = [row for row in audit_rows if row.get("oracle_label") == "negative"]
    dryrun_rows = [row for row in audit_rows if bool(row.get("selected_by_dryrun"))]
    judge_rows = [row for row in audit_rows if bool(row.get("selected_by_judge"))]

    def _delta_stats(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
        values = [float(row.get(field)) for row in rows if row.get(field) is not None]
        return {
            "count": len(values),
            "mean": _mean(values),
            "median": _median(values),
            "nonpositive_count": int(sum(value <= 0.0 for value in values)),
            "nonpositive_rate": _safe_pct(sum(value <= 0.0 for value in values), len(values)),
        }

    heuristic_query_count = sum(str(row.get("query_tier_mode") or "") == "heuristic_tiers" for row in query_summaries)
    flat_query_count = sum(str(row.get("query_tier_mode") or "") == "flat_fallback" for row in query_summaries)
    hit_top1, total = _topk_positive_recall(query_summaries, "ranking_tiered_witness", k=1)
    hit_top3, _ = _topk_positive_recall(query_summaries, "ranking_tiered_witness", k=3)

    return {
        "dataset": dataset,
        "selected_action_counts": {
            "total": int(len(audit_rows)),
            "oracle_positive": int(len(positive_rows)),
            "oracle_negative": int(len(negative_rows)),
            "dryrun_selected": int(len(dryrun_rows)),
            "judge_selected": int(len(judge_rows)),
        },
        "oracle_positive_gain_stats": _delta_stats(positive_rows, "swap_gain_vs_loss"),
        "oracle_negative_gain_stats": _delta_stats(negative_rows, "swap_gain_vs_loss"),
        "policy_selected_gain_stats": {
            "dryrun": _delta_stats(dryrun_rows, "swap_gain_vs_loss"),
            "judge": _delta_stats(judge_rows, "swap_gain_vs_loss"),
        },
        "ranking_recall": {
            "top1_hit_count": hit_top1,
            "top1_hit_rate": _safe_pct(hit_top1, total),
            "top3_hit_count": hit_top3,
            "top3_hit_rate": _safe_pct(hit_top3, total),
            "query_count": total,
        },
        "tier_mode_counts": {
            "heuristic_tiers": heuristic_query_count,
            "flat_fallback": flat_query_count,
        },
    }


def _build_markdown(summary: Mapping[str, Any], audit_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"# Action-Level Audit: {summary.get('dataset')}",
        "",
        f"- selected actions: `{(summary.get('selected_action_counts') or {}).get('total')}`",
        f"- oracle-positive selected swaps: `{(summary.get('selected_action_counts') or {}).get('oracle_positive')}`",
        f"- oracle-negative sampled swaps: `{(summary.get('selected_action_counts') or {}).get('oracle_negative')}`",
        f"- dryrun executed swaps in audit: `{(summary.get('selected_action_counts') or {}).get('dryrun_selected')}`",
        f"- judge executed swaps in audit: `{(summary.get('selected_action_counts') or {}).get('judge_selected')}`",
        f"- heuristic tiers queries: `{(summary.get('tier_mode_counts') or {}).get('heuristic_tiers')}`",
        f"- flat fallback queries: `{(summary.get('tier_mode_counts') or {}).get('flat_fallback')}`",
        "",
        "## Oracle-Positive Gain Audit",
        "",
        "| Group | Count | Median gain | Nonpositive |",
        "|---|---:|---:|---:|",
    ]
    for label, stats in (
        ("oracle_positive", dict(summary.get("oracle_positive_gain_stats") or {})),
        ("oracle_negative", dict(summary.get("oracle_negative_gain_stats") or {})),
    ):
        lines.append(
            f"| {label} | {stats.get('count', 0)} | {stats.get('median', '—')} | "
            f"{stats.get('nonpositive_count', 0)} ({stats.get('nonpositive_rate', '—')}) |"
        )

    ranking = dict(summary.get("ranking_recall") or {})
    lines.extend([
        "",
        "## Ranking Recall",
        "",
        f"- top-1 oracle-positive hit rate: `{ranking.get('top1_hit_rate')}` over `{ranking.get('query_count')}` queries",
        f"- top-3 oracle-positive hit rate: `{ranking.get('top3_hit_rate')}` over `{ranking.get('query_count')}` queries",
        "",
        "## Sample Failure Cases",
        "",
    ])
    failure_rows = [
        row for row in audit_rows
        if row.get("oracle_label") == "positive" and float(row.get("swap_gain_vs_loss", 0.0) or 0.0) <= 0.0
    ][:10]
    if not failure_rows:
        lines.append("- none")
    else:
        for row in failure_rows:
            lines.extend([
                f"- question: `{row.get('question', '')}`",
                f"  - action: swap in `{row.get('candidate_title')}` for `{row.get('replace_incumbent_title')}`",
                f"  - oracle ΔEM / ΔF1: `{row.get('oracle_delta_em')}` / `{row.get('oracle_delta_f1')}`",
                f"  - bottleneck tier / facet: `{row.get('bottleneck_tier_index')}` / `{row.get('target_facet_id')}`",
                f"  - gain vs loss: `{row.get('target_candidate_gain')}` vs `{row.get('replacee_loss_earlier_tiers')}` + `{row.get('replacee_loss_target_facet')}`",
                f"  - net score: `{row.get('swap_gain_vs_loss')}`",
            ])
    return "\n".join(lines) + "\n"


def run_audit(dataset: str,
              *,
              oracle_relaxed_report: str,
              query_report: str,
              dryrun_report: str,
              judge_report: str,
              output_json: str,
              output_md: str,
              output_csv: str,
              ce_model: str,
              ce_device: str,
              negative_sample_size: int,
              sample_seed: int) -> Dict[str, Any]:
    oracle_payload = _load_json(oracle_relaxed_report)
    query_payload = _load_json(query_report)
    dryrun_payload = _load_json(dryrun_report)
    judge_payload = _load_json(judge_report)

    dryrun_actions = _extract_policy_actions(dryrun_payload)
    judge_actions = _extract_policy_actions(judge_payload)
    query_contexts = _query_context_map(query_payload)
    selected_rows = _select_action_rows(
        oracle_payload,
        dryrun_actions=dryrun_actions,
        judge_actions=judge_actions,
        negative_sample_size=negative_sample_size,
        seed=sample_seed,
    )
    selected_keys = {
        (_normalize_question(row.get("question", "")), _action_key(row))
        for row in selected_rows
    }

    all_legal_rows: List[Dict[str, Any]] = []
    for row in list(oracle_payload.get("action_results") or []):
        if str(row.get("action_type") or "").strip().lower() != "swap" or not bool(row.get("is_legal")):
            continue
        materialized = dict(row)
        delta_metrics = dict(materialized.get("delta_metrics") or {})
        materialized["oracle_positive"] = bool(
            float(delta_metrics.get("F1", 0.0) or 0.0) > 0.0
            or float(delta_metrics.get("ExactMatch", 0.0) or 0.0) > 0.0
        )
        question_key = _normalize_question(materialized.get("question", ""))
        materialized["selected_by_dryrun"] = _policy_selected(question_key, dryrun_actions, materialized)
        materialized["selected_by_judge"] = _policy_selected(question_key, judge_actions, materialized)
        all_legal_rows.append(materialized)

    by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_legal_rows:
        by_question[_normalize_question(row.get("question", ""))].append(dict(row))

    ce_reranker = _load_ce_reranker(ce_model, ce_device)
    audit_rows: List[Dict[str, Any]] = []
    query_summaries: List[Dict[str, Any]] = []
    for question_key, rows in by_question.items():
        query_context = dict(query_contexts.get(question_key) or {})
        query_audit_rows, query_summary = _compute_query_audit_rows(
            rows,
            query_context=query_context,
            ce_reranker=ce_reranker,
        )
        for row in query_audit_rows:
            row["dataset"] = dataset
        audit_rows.extend([
            row
            for row in query_audit_rows
            if (question_key, _action_key(row)) in selected_keys
        ])
        if query_summary:
            query_summaries.append(query_summary)

    summary = _build_summary(dataset, audit_rows, query_summaries)
    payload = {
        "summary": summary,
        "audit_rows": audit_rows,
        "query_summaries": query_summaries,
        "inputs": {
            "dataset": dataset,
            "oracle_relaxed_report": str(oracle_relaxed_report),
            "query_report": str(query_report),
            "dryrun_report": str(dryrun_report),
            "judge_report": str(judge_report),
            "ce_model": str(ce_model),
            "ce_device": str(ce_device),
            "negative_sample_size": int(negative_sample_size),
        },
    }
    Path(output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(output_md).write_text(_build_markdown(summary, audit_rows), encoding="utf-8")
    with Path(output_csv).open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "question",
            "query_type",
            "oracle_label",
            "selected_by_dryrun",
            "selected_by_judge",
            "query_tier_mode",
            "bottleneck_tier_index",
            "target_facet_id",
            "candidate_pool_position",
            "replace_pool_position",
            "candidate_title",
            "replace_incumbent_title",
            "score_delta",
            "oracle_delta_em",
            "oracle_delta_f1",
            "target_candidate_gain",
            "replacee_loss_earlier_tiers",
            "replacee_loss_same_tier",
            "replacee_loss_target_facet",
            "swap_gain_vs_loss",
            "action_should_swap",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local action-level audit for the tiered witness action controller.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--oracle_relaxed_report", required=True)
    parser.add_argument("--query_report", required=True)
    parser.add_argument("--dryrun_report", required=True)
    parser.add_argument("--judge_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--ce_model", default=DEFAULT_CE_MODEL)
    parser.add_argument("--ce_device", default="cuda:0")
    parser.add_argument("--negative_sample_size", type=int, default=40)
    parser.add_argument("--sample_seed", type=int, default=7)
    args = parser.parse_args()

    run_audit(
        dataset=str(args.dataset),
        oracle_relaxed_report=str(args.oracle_relaxed_report),
        query_report=str(args.query_report),
        dryrun_report=str(args.dryrun_report),
        judge_report=str(args.judge_report),
        output_json=str(args.output_json),
        output_md=str(args.output_md),
        output_csv=str(args.output_csv),
        ce_model=str(args.ce_model),
        ce_device=str(args.ce_device),
        negative_sample_size=int(args.negative_sample_size),
        sample_seed=int(args.sample_seed),
    )


if __name__ == "__main__":
    main()
