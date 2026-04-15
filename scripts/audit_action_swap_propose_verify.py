#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

for _proxy_env in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_proxy_env, None)

from eval_causal_qwen3 import (  # noqa: E402
    build_setwise_late_rerank_judge_bundle,
    score_action_swap_propose_verify_jobs,
)


DEFAULT_CE_MODEL = "/mnt/nvme/bge-reranker-v2-m3"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_question(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(100.0 * float(numerator) / float(denominator), 2)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 4)


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
                              ce_reranker: Any,
                              verifier_bundle: Any,
                              max_doc_chars: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
        local_action_jobs.append({
            "question": question,
            "action_type": "swap",
            "candidate_pool_position": int(candidate_local_pos[candidate_key]),
            "candidate_doc_id": row.get("candidate_doc_id"),
            "replace_pool_position": int(replace_index),
            "replace_doc_id": row.get("replace_doc_id"),
            "candidate_assemble_score": float(row.get("candidate_assemble_score", row.get("candidate_ce_score", 0.0)) or 0.0),
            "score_delta": float(row.get("score_delta", 0.0) or 0.0),
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

    scored_bundle = score_action_swap_propose_verify_jobs(
        local_action_jobs,
        query=question,
        scaffold_positions=scaffold_positions,
        pool_docs=local_docs,
        query_entities=query_entities,
        ce_reranker=ce_reranker,
        verifier_bundle=verifier_bundle,
        max_doc_chars=max_doc_chars,
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
            "claim_mode": row.get("claim_mode"),
            "local_candidate_pool_position": row.get("candidate_pool_position"),
            "local_replace_pool_position": row.get("replace_pool_position"),
            "candidate_pool_position": row.get("oracle_candidate_pool_position"),
            "replace_pool_position": row.get("oracle_replace_pool_position"),
            "candidate_title": row.get("oracle_candidate_title"),
            "replace_incumbent_title": row.get("oracle_replace_title"),
            "score_delta": _round(row.get("score_delta")),
            "oracle_delta_em": _round(row.get("oracle_delta_em")),
            "oracle_delta_f1": _round(row.get("oracle_delta_f1")),
            "earliest_unsupported_claim_id": row.get("earliest_unsupported_claim_id"),
            "earliest_unsupported_claim_text": row.get("earliest_unsupported_claim_text"),
            "gain_verifier_verdict": row.get("gain_verifier_verdict"),
            "gain_verifier_reason": row.get("gain_verifier_reason"),
            "preservation_unique_support_claim_ids": list(row.get("preservation_unique_support_claim_ids") or []),
            "action_should_swap": bool(row.get("action_should_swap")),
            "action_skip_reason": row.get("action_skip_reason"),
            "candidate_best_witness": row.get("candidate_best_witness"),
            "claim_supports_before": row.get("claim_supports_before"),
        })

    proposal_key = scored_bundle.get("proposal_key")
    proposal_row = None
    if proposal_key is not None:
        proposal_row = next(
            (
                row for row in audit_rows
                if int(row.get("local_candidate_pool_position")) == int(proposal_key[0])
                and int(row.get("local_replace_pool_position")) == int(proposal_key[1])
            ),
            None,
        )
    query_summary = {
        "question": question,
        "claim_mode": str(scored_bundle.get("claim_mode") or "flat_fallback"),
        "proposal_key": list(proposal_key) if proposal_key is not None else None,
        "proposal_passed": bool(proposal_row.get("action_should_swap")) if proposal_row else False,
        "proposal_skip_reason": proposal_row.get("action_skip_reason") if proposal_row else "no_proposal",
    }
    return audit_rows, query_summary


def _build_summary(dataset: str,
                   audit_rows: Sequence[Mapping[str, Any]],
                   query_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positive_rows = [row for row in audit_rows if row.get("oracle_label") == "positive"]
    negative_rows = [row for row in audit_rows if row.get("oracle_label") == "negative"]
    dryrun_rows = [row for row in audit_rows if bool(row.get("selected_by_dryrun"))]
    judge_rows = [row for row in audit_rows if bool(row.get("selected_by_judge"))]

    def _pass_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        passed = sum(bool(row.get("action_should_swap")) for row in rows)
        return {
            "count": total,
            "passed": int(passed),
            "pass_rate": _safe_pct(passed, total),
        }

    reject_counter = Counter(
        str(row.get("action_skip_reason") or "passed")
        for row in audit_rows
    )
    proposal_passed = sum(bool(row.get("proposal_passed")) for row in query_summaries)
    proposal_total = len(query_summaries)

    positive_f1 = [float(row.get("oracle_delta_f1", 0.0) or 0.0) for row in positive_rows]
    negative_f1 = [float(row.get("oracle_delta_f1", 0.0) or 0.0) for row in negative_rows]
    return {
        "dataset": dataset,
        "selected_action_counts": {
            "total": int(len(audit_rows)),
            "oracle_positive": int(len(positive_rows)),
            "oracle_negative": int(len(negative_rows)),
            "dryrun_selected": int(len(dryrun_rows)),
            "judge_selected": int(len(judge_rows)),
        },
        "pass_stats": {
            "oracle_positive": _pass_stats(positive_rows),
            "oracle_negative": _pass_stats(negative_rows),
            "dryrun_selected": _pass_stats(dryrun_rows),
            "judge_selected": _pass_stats(judge_rows),
        },
        "oracle_delta_f1": {
            "oracle_positive_median": _median(positive_f1),
            "oracle_negative_median": _median(negative_f1),
        },
        "proposal_pass": {
            "query_count": proposal_total,
            "proposal_passed": int(proposal_passed),
            "proposal_pass_rate": _safe_pct(proposal_passed, proposal_total),
        },
        "reject_reason_counts": dict(sorted(reject_counter.items())),
    }


def _build_markdown(summary: Mapping[str, Any], audit_rows: Sequence[Mapping[str, Any]]) -> str:
    pass_stats = dict(summary.get("pass_stats") or {})
    lines = [
        f"# Action-Level Audit: {summary.get('dataset')}",
        "",
        f"- selected actions: `{(summary.get('selected_action_counts') or {}).get('total')}`",
        f"- oracle-positive selected swaps: `{(summary.get('selected_action_counts') or {}).get('oracle_positive')}`",
        f"- oracle-negative sampled swaps: `{(summary.get('selected_action_counts') or {}).get('oracle_negative')}`",
        f"- dryrun executed swaps in audit: `{(summary.get('selected_action_counts') or {}).get('dryrun_selected')}`",
        f"- judge executed swaps in audit: `{(summary.get('selected_action_counts') or {}).get('judge_selected')}`",
        f"- proposal pass rate: `{(summary.get('proposal_pass') or {}).get('proposal_pass_rate')}`",
        "",
        "## Pass Rates",
        "",
        "| Group | Count | Passed | Pass rate |",
        "|---|---:|---:|---:|",
    ]
    for label in ("oracle_positive", "oracle_negative", "dryrun_selected", "judge_selected"):
        stats = dict(pass_stats.get(label) or {})
        lines.append(
            f"| {label} | {stats.get('count', 0)} | {stats.get('passed', 0)} | {stats.get('pass_rate', '—')} |"
        )

    lines.extend([
        "",
        "## Reject Reasons",
        "",
    ])
    for key, value in dict(summary.get("reject_reason_counts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend([
        "",
        "## Sample Oracle-Positive Rejections",
        "",
    ])
    rejected_positive_rows = [
        row for row in audit_rows
        if row.get("oracle_label") == "positive" and not bool(row.get("action_should_swap"))
    ][:10]
    if not rejected_positive_rows:
        lines.append("- none")
    else:
        for row in rejected_positive_rows:
            lines.extend([
                f"- question: `{row.get('question', '')}`",
                f"  - action: swap in `{row.get('candidate_title')}` for `{row.get('replace_incumbent_title')}`",
                f"  - oracle ΔEM / ΔF1: `{row.get('oracle_delta_em')}` / `{row.get('oracle_delta_f1')}`",
                f"  - claim: `{row.get('earliest_unsupported_claim_id')}` `{row.get('earliest_unsupported_claim_text')}`",
                f"  - gain verdict: `{row.get('gain_verifier_verdict')}`",
                f"  - unique-support veto: `{','.join(row.get('preservation_unique_support_claim_ids') or [])}`",
                f"  - skip reason: `{row.get('action_skip_reason')}`",
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
              sample_seed: int,
              verifier_bundle: Any,
              max_doc_chars: int) -> Dict[str, Any]:
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
            verifier_bundle=verifier_bundle,
            max_doc_chars=max_doc_chars,
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
            "max_doc_chars": int(max_doc_chars),
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
            "claim_mode",
            "candidate_pool_position",
            "replace_pool_position",
            "candidate_title",
            "replace_incumbent_title",
            "score_delta",
            "oracle_delta_em",
            "oracle_delta_f1",
            "earliest_unsupported_claim_id",
            "earliest_unsupported_claim_text",
            "gain_verifier_verdict",
            "action_skip_reason",
            "action_should_swap",
            "preservation_unique_support_claim_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({
                **{key: row.get(key) for key in fieldnames},
                "preservation_unique_support_claim_ids": json.dumps(row.get("preservation_unique_support_claim_ids") or [], ensure_ascii=False),
            })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local action-level audit for the propose-verify action controller.")
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
    parser.add_argument("--max_doc_chars", type=int, default=320)

    parser.add_argument("--setwise_late_rerank_judge_backend", default="inherit")
    parser.add_argument("--setwise_late_rerank_judge_model", default="")
    parser.add_argument("--setwise_late_rerank_judge_base_url", default="")
    parser.add_argument("--setwise_late_rerank_judge_api_key", default="")
    parser.add_argument("--setwise_late_rerank_judge_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--setwise_late_rerank_judge_reasoning_effort", default="")
    parser.add_argument("--setwise_late_rerank_judge_timeout_s", type=float, default=120.0)
    args = parser.parse_args()

    verifier_bundle = build_setwise_late_rerank_judge_bundle(
        args=args,
        fallback_model_name=str(args.setwise_late_rerank_judge_model or "gpt-5.4-mini"),
        fallback_base_url=str(args.setwise_late_rerank_judge_base_url or "") or None,
    )
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
        verifier_bundle=verifier_bundle,
        max_doc_chars=int(args.max_doc_chars),
    )


if __name__ == "__main__":
    main()
