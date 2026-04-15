#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


logger = logging.getLogger(__name__)


from run_swap_value_smoke import (
    _build_chunk_id_to_doc_text,
    _load_json,
    _mean,
    _median,
    _normalize_question,
    _resolve_doc_text,
    _round,
    _safe_pct,
    infer_llm_config,
    run_reader,
)


def extract_baseline_scaffold_positions_from_ranking_rows(
    ranking_rows: Sequence[Mapping[str, object]],
    baseline_prefix_positions: Sequence[int],
    qa_top_k: int,
) -> list[int]:
    baseline_position_set = {int(pos) for pos in baseline_prefix_positions}
    scaffold_positions: list[int] = []
    seen_positions: set[int] = set()
    for row in ranking_rows:
        raw_pool_position = row.get("pool_position", -1)
        pool_position = int(raw_pool_position) if raw_pool_position is not None else -1
        if pool_position < 0 or pool_position in seen_positions:
            continue
        if baseline_position_set and pool_position not in baseline_position_set:
            continue
        scaffold_positions.append(pool_position)
        seen_positions.add(pool_position)
        if len(scaffold_positions) >= max(int(qa_top_k), 0):
            break
    if scaffold_positions or not ranking_rows:
        return scaffold_positions

    for row in ranking_rows:
        raw_pool_position = row.get("pool_position", -1)
        pool_position = int(raw_pool_position) if raw_pool_position is not None else -1
        if pool_position < 0 or pool_position in seen_positions:
            continue
        scaffold_positions.append(pool_position)
        seen_positions.add(pool_position)
        if len(scaffold_positions) >= max(int(qa_top_k), 0):
            break
    return scaffold_positions


def _action_key(row: Mapping[str, Any]) -> tuple[str, int | None, int | None]:
    action_type = str(row.get("action_type", "keep") or "keep").strip().lower()
    if action_type != "swap":
        return ("keep", None, None)
    candidate_position = row.get("candidate_pool_position")
    replace_position = row.get("replace_pool_position")
    return (
        "swap",
        int(candidate_position) if candidate_position is not None else None,
        int(replace_position) if replace_position is not None else None,
    )


def _metric_pair(row: Mapping[str, Any]) -> tuple[float, float]:
    metrics = dict(row.get("swap_metrics") or {})
    return (
        float(metrics.get("ExactMatch", 0.0) or 0.0),
        float(metrics.get("F1", 0.0) or 0.0),
    )


def _is_swap_improving(row: Mapping[str, Any], keep_row: Mapping[str, Any]) -> bool:
    row_em, row_f1 = _metric_pair(row)
    keep_em, keep_f1 = _metric_pair(keep_row)
    return (row_em > keep_em) or (row_em == keep_em and row_f1 > keep_f1)


def _oracle_sort_key(row: Mapping[str, Any]) -> tuple[float, float, int, float, int]:
    em, f1 = _metric_pair(row)
    action_type = str(row.get("action_type", "keep") or "keep").strip().lower()
    prefer_keep = 1 if action_type != "swap" else 0
    candidate_score = (
        float(row.get("candidate_assemble_score"))
        if row.get("candidate_assemble_score") is not None else
        float("-inf")
    )
    candidate_pool_position = (
        int(row.get("candidate_pool_position"))
        if row.get("candidate_pool_position") is not None else
        10**9
    )
    return (em, f1, prefer_keep, candidate_score, -candidate_pool_position)


def _build_row_by_pool_position(ranking_rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int(row.get("pool_position")): dict(row)
        for row in ranking_rows
        if row.get("pool_position") is not None and int(row.get("pool_position")) >= 0
    }


def _resolve_identity(*,
                      row: Mapping[str, Any],
                      corpus: Sequence[Mapping[str, Any]],
                      chunk_id_to_doc_text: Mapping[str, str]) -> dict[str, Any]:
    raw_doc_id = row.get("doc_id")
    doc_text = _resolve_doc_text(doc_id=raw_doc_id, corpus=corpus, chunk_id_to_doc_text=chunk_id_to_doc_text)
    doc_id: int | None = None
    chunk_id: str | None = None
    if isinstance(raw_doc_id, int):
        doc_id = int(raw_doc_id)
    elif isinstance(raw_doc_id, str):
        normalized = str(raw_doc_id).strip()
        chunk_id = normalized or None
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "doc_text": doc_text,
    }


def _is_duplicate_identity(candidate_identity: Mapping[str, Any],
                           scaffold_identities: Sequence[Mapping[str, Any]]) -> bool:
    candidate_doc_id = candidate_identity.get("doc_id")
    if candidate_doc_id is not None and any(identity.get("doc_id") is not None and int(identity["doc_id"]) == int(candidate_doc_id) for identity in scaffold_identities):
        return True

    candidate_chunk_id = candidate_identity.get("chunk_id")
    if candidate_chunk_id and any(identity.get("chunk_id") and str(identity["chunk_id"]) == str(candidate_chunk_id) for identity in scaffold_identities):
        return True

    candidate_text = candidate_identity.get("doc_text")
    return bool(candidate_text and any(identity.get("doc_text") == candidate_text for identity in scaffold_identities))


def _build_keep_job(*,
                    dataset: str,
                    trace: Mapping[str, Any],
                    keep_docs: Sequence[str],
                    scaffold_positions: Sequence[int],
                    scaffold_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "question": str(trace.get("question", "")),
        "query_type": str(trace.get("query_type", "")),
        "gold_answers": list(trace.get("gold_answers") or []),
        "baseline_answer": "",
        "baseline_metrics": {},
        "baseline_top_titles": [str(doc.split("\n", 1)[0]).strip() for doc in keep_docs],
        "baseline_docs": list(keep_docs),
        "baseline_doc_titles": [str(doc.split("\n", 1)[0]).strip() for doc in keep_docs],
        "baseline_scaffold_positions": [int(pos) for pos in scaffold_positions],
        "baseline_final_doc_ids": [row.get("doc_id") for row in scaffold_rows],
        "replace_bottom_n": 0,
        "replace_incumbent_index": None,
        "replace_incumbent_rank": None,
        "replace_incumbent_doc_id": None,
        "replace_incumbent_title": None,
        "replace_incumbent_ce_rank": None,
        "replace_incumbent_ce_score": None,
        "candidate_doc_id": None,
        "candidate_title": None,
        "candidate_source": "keep",
        "candidate_pool_position": None,
        "candidate_ce_rank": None,
        "candidate_ce_score": None,
        "candidate_assemble_score": None,
        "candidate_base_score": None,
        "candidate_entered_method_topk": None,
        "docs": list(keep_docs),
        "swapped_top_titles": [str(doc.split("\n", 1)[0]).strip() for doc in keep_docs],
        "action_type": "keep",
        "replace_pool_position": None,
        "swapped_positions": [int(pos) for pos in scaffold_positions],
        "score_delta": None,
        "is_duplicate_with_scaffold": False,
        "is_legal": True,
    }


def build_oracle_action_jobs(
    *,
    dataset: str,
    candidate_report: Mapping[str, Any],
    corpus: Sequence[Mapping[str, Any]],
    qa_top_k: int,
    replace_bottom_n: int = 2,
    legality_mode: str = "current",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    chunk_id_to_doc_text = _build_chunk_id_to_doc_text(list(corpus))
    counters: Counter[str] = Counter()
    jobs: list[dict[str, Any]] = []

    legality_mode = str(legality_mode or "current").strip().lower()
    if legality_mode not in {"current", "relaxed"}:
        raise ValueError(f"Unsupported legality_mode: {legality_mode}")

    query_traces = list(candidate_report.get("expand_assemble_query_traces") or [])
    counters["total_queries"] = len(query_traces)
    replace_count = max(1, int(replace_bottom_n))

    for trace in query_traces:
        expand_trace = dict(trace.get("expand_assemble_trace") or {})
        ranking_rows = list((expand_trace.get("assemble_trace") or {}).get("ranking_rows") or [])
        if not ranking_rows:
            counters["skipped_missing_ranking_rows"] += 1
            continue

        scaffold_positions = extract_baseline_scaffold_positions_from_ranking_rows(
            ranking_rows=ranking_rows,
            baseline_prefix_positions=expand_trace.get("baseline_prefix_positions") or [],
            qa_top_k=int(qa_top_k),
        )
        if len(scaffold_positions) < int(qa_top_k):
            counters["skipped_short_scaffold"] += 1
            continue

        row_by_pool_position = _build_row_by_pool_position(ranking_rows)
        scaffold_rows: list[dict[str, Any]] = []
        scaffold_docs: list[str] = []
        scaffold_identities: list[dict[str, Any]] = []
        scaffold_missing = False
        for scaffold_position in scaffold_positions[: int(qa_top_k)]:
            row = row_by_pool_position.get(int(scaffold_position))
            if row is None:
                counters["skipped_missing_scaffold_row"] += 1
                scaffold_missing = True
                break
            identity = _resolve_identity(row=row, corpus=corpus, chunk_id_to_doc_text=chunk_id_to_doc_text)
            doc_text = identity.get("doc_text")
            if not doc_text:
                counters["skipped_missing_scaffold_doc_text"] += 1
                scaffold_missing = True
                break
            scaffold_rows.append(dict(row))
            scaffold_docs.append(str(doc_text))
            scaffold_identities.append(dict(identity))
        if scaffold_missing:
            continue

        counters["processed_queries"] += 1
        jobs.append(
            _build_keep_job(
                dataset=dataset,
                trace=trace,
                keep_docs=scaffold_docs,
                scaffold_positions=scaffold_positions[: int(qa_top_k)],
                scaffold_rows=scaffold_rows,
            )
        )
        counters["total_keep_jobs"] += 1

        appended_positions = [int(pos) for pos in (expand_trace.get("appended_positions") or [])]
        if appended_positions:
            counters["queries_with_appended_candidates"] += 1
        else:
            counters["queries_without_appended_candidates"] += 1
            continue

        replace_positions = list(scaffold_positions[: int(qa_top_k)])[-replace_count:]
        query_legal_job_count = 0
        for candidate_position in appended_positions:
            candidate_row = row_by_pool_position.get(int(candidate_position))
            if candidate_row is None:
                counters["skipped_missing_candidate_row"] += 1
                continue

            candidate_identity = _resolve_identity(
                row=candidate_row,
                corpus=corpus,
                chunk_id_to_doc_text=chunk_id_to_doc_text,
            )
            candidate_doc_text = candidate_identity.get("doc_text")
            if not candidate_doc_text:
                counters["skipped_missing_candidate_doc_text"] += 1
                continue

            if _is_duplicate_identity(candidate_identity, scaffold_identities):
                counters["skipped_duplicate_candidate_doc"] += 1
                continue

            candidate_score = (
                float(candidate_row.get("assemble_score"))
                if candidate_row.get("assemble_score") is not None else
                float("-inf")
            )

            for replace_position in replace_positions:
                replace_index = next(
                    (idx for idx, pos in enumerate(scaffold_positions[: int(qa_top_k)]) if int(pos) == int(replace_position)),
                    -1,
                )
                if replace_index < 0:
                    continue
                replace_row = row_by_pool_position.get(int(replace_position))
                if replace_row is None:
                    counters["skipped_missing_replace_row"] += 1
                    continue
                replace_score = (
                    float(replace_row.get("assemble_score"))
                    if replace_row.get("assemble_score") is not None else
                    float("-inf")
                )
                score_delta = float(candidate_score - replace_score)
                if legality_mode == "current" and score_delta <= 0.0:
                    counters["filtered_nonpositive_score_delta"] += 1
                    continue

                swapped_docs = list(scaffold_docs)
                swapped_docs[replace_index] = str(candidate_doc_text)
                query_legal_job_count += 1
                jobs.append({
                    "dataset": dataset,
                    "question": str(trace.get("question", "")),
                    "query_type": str(trace.get("query_type", "")),
                    "gold_answers": list(trace.get("gold_answers") or []),
                    "baseline_answer": "",
                    "baseline_metrics": {},
                    "baseline_top_titles": [str(doc.split("\n", 1)[0]).strip() for doc in scaffold_docs],
                    "baseline_final_doc_ids": [row.get("doc_id") for row in scaffold_rows],
                    "baseline_docs": list(scaffold_docs),
                    "baseline_doc_titles": [str(doc.split("\n", 1)[0]).strip() for doc in scaffold_docs],
                    "baseline_scaffold_positions": [int(pos) for pos in scaffold_positions[: int(qa_top_k)]],
                    "replace_bottom_n": int(replace_count),
                    "replace_incumbent_index": int(replace_index),
                    "replace_incumbent_rank": int(replace_index + 1),
                    "replace_incumbent_doc_id": replace_row.get("doc_id"),
                    "replace_incumbent_title": str(replace_row.get("title", "")),
                    "replace_incumbent_ce_rank": (
                        int(replace_row.get("rank")) if replace_row.get("rank") is not None else None
                    ),
                    "replace_incumbent_ce_score": _round(replace_score),
                    "candidate_doc_id": candidate_row.get("doc_id"),
                    "candidate_title": str(candidate_row.get("title", "")),
                    "candidate_source": str(candidate_row.get("source", "")),
                    "candidate_pool_position": int(candidate_position),
                    "candidate_ce_rank": (
                        int(candidate_row.get("rank")) if candidate_row.get("rank") is not None else None
                    ),
                    "candidate_ce_score": _round(candidate_score),
                    "candidate_assemble_score": _round(candidate_score),
                    "candidate_base_score": (
                        _round(float(candidate_row.get("base_score"))) if candidate_row.get("base_score") is not None else None
                    ),
                    "candidate_entered_method_topk": bool(
                        candidate_row.get("doc_id") in set(expand_trace.get("final_front_doc_ids") or [])
                    ),
                    "docs": swapped_docs,
                    "swapped_top_titles": [str(doc.split("\n", 1)[0]).strip() for doc in swapped_docs],
                    "action_type": "swap",
                    "replace_pool_position": int(replace_position),
                    "swapped_positions": [
                        int(candidate_position) if idx == replace_index else int(pos)
                        for idx, pos in enumerate(scaffold_positions[: int(qa_top_k)])
                    ],
                    "score_delta": _round(score_delta),
                    "is_duplicate_with_scaffold": False,
                    "is_legal": True,
                })

        if query_legal_job_count > 0:
            counters["queries_with_legal_swaps"] += 1
        counters["total_swap_jobs"] += query_legal_job_count

    return jobs, dict(counters)


def extract_policy_actions(report_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    policy_actions: dict[str, dict[str, Any]] = {}
    for trace in list(report_payload.get("expand_assemble_query_traces") or []):
        question_key = _normalize_question(trace.get("question", ""))
        if not question_key:
            continue
        expand_trace = dict(trace.get("expand_assemble_trace") or {})
        action_executed = bool(expand_trace.get("action_executed"))
        if action_executed and expand_trace.get("action_candidate_pool_position") is not None and expand_trace.get("action_replace_pool_position") is not None:
            policy_actions[question_key] = {
                "action_type": "swap",
                "action_executed": True,
                "candidate_pool_position": int(expand_trace.get("action_candidate_pool_position")),
                "replace_pool_position": int(expand_trace.get("action_replace_pool_position")),
                "action_mode": expand_trace.get("action_mode"),
                "action_key": (
                    "swap",
                    int(expand_trace.get("action_candidate_pool_position")),
                    int(expand_trace.get("action_replace_pool_position")),
                ),
            }
        else:
            policy_actions[question_key] = {
                "action_type": "keep",
                "action_executed": False,
                "candidate_pool_position": None,
                "replace_pool_position": None,
                "action_mode": expand_trace.get("action_mode"),
                "action_key": ("keep", None, None),
            }
    return policy_actions


def summarize_oracle_results(
    *,
    dataset: str,
    candidate_report_path: str,
    output_json: str,
    legality_mode: str,
    qa_top_k: int,
    replace_bottom_n: int,
    llm_name: str,
    llm_request_name: str | None,
    llm_base_url: str,
    generation_counters: Mapping[str, int],
    action_results: Sequence[Mapping[str, Any]],
    policy_action_maps: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    swap_delta_em_all: list[float] = []
    swap_delta_f1_all: list[float] = []
    positive_swap_count_em = 0
    positive_swap_count_f1 = 0
    swap_action_count = 0

    for row in action_results:
        grouped.setdefault(_normalize_question(row.get("question", "")), []).append(dict(row))

    policy_action_maps = policy_action_maps or {}
    baseline_em_by_query: list[float] = []
    baseline_f1_by_query: list[float] = []
    oracle_em_by_query: list[float] = []
    oracle_f1_by_query: list[float] = []
    query_summaries: list[dict[str, Any]] = []
    oracle_positive_query_count = 0

    improving_action_keys_by_question: dict[str, set[tuple[str, int | None, int | None]]] = {}
    oracle_best_action_key_by_question: dict[str, tuple[str, int | None, int | None]] = {}

    for question_key, rows in grouped.items():
        keep_row = next((dict(row) for row in rows if str(row.get("action_type", "keep")) != "swap"), None)
        if keep_row is None:
            continue

        baseline_em, baseline_f1 = _metric_pair(keep_row)
        baseline_em_by_query.append(baseline_em)
        baseline_f1_by_query.append(baseline_f1)

        swap_rows = [dict(row) for row in rows if str(row.get("action_type", "keep")) == "swap"]
        for row in swap_rows:
            swap_action_count += 1
            if _is_swap_improving(row, keep_row):
                swap_delta_em_all.append(_metric_pair(row)[0] - baseline_em)
                swap_delta_f1_all.append(_metric_pair(row)[1] - baseline_f1)
                if _metric_pair(row)[0] > baseline_em:
                    positive_swap_count_em += 1
                if _metric_pair(row)[1] > baseline_f1 or _metric_pair(row)[0] > baseline_em:
                    positive_swap_count_f1 += 1

        all_rows = [keep_row] + swap_rows
        best_row = max(all_rows, key=_oracle_sort_key)
        oracle_em, oracle_f1 = _metric_pair(best_row)
        oracle_em_by_query.append(oracle_em)
        oracle_f1_by_query.append(oracle_f1)

        improving_keys = {
            _action_key(row)
            for row in swap_rows
            if _is_swap_improving(row, keep_row)
        }
        improving_action_keys_by_question[question_key] = set(improving_keys)

        best_key = _action_key(best_row)
        oracle_best_action_key_by_question[question_key] = best_key

        oracle_positive = best_key != ("keep", None, None) and _is_swap_improving(best_row, keep_row)
        if oracle_positive:
            oracle_positive_query_count += 1

        query_summaries.append({
            "question": keep_row.get("question"),
            "query_type": keep_row.get("query_type"),
            "num_legal_swaps": len(swap_rows),
            "keep_metrics": {
                "ExactMatch": _round(baseline_em),
                "F1": _round(baseline_f1),
            },
            "oracle_best_action": {
                "action_type": best_key[0],
                "candidate_pool_position": best_key[1],
                "replace_pool_position": best_key[2],
                "candidate_title": best_row.get("candidate_title"),
                "replace_incumbent_title": best_row.get("replace_incumbent_title"),
                "candidate_assemble_score": best_row.get("candidate_assemble_score"),
                "score_delta": best_row.get("score_delta"),
                "answer": best_row.get("swap_answer"),
                "metrics": {
                    "ExactMatch": _round(oracle_em),
                    "F1": _round(oracle_f1),
                },
            },
            "oracle_delta_metrics": {
                "ExactMatch": _round(oracle_em - baseline_em),
                "F1": _round(oracle_f1 - baseline_f1),
            },
            "oracle_positive_query": bool(oracle_positive),
        })

    query_summaries.sort(
        key=lambda item: (
            float((item.get("oracle_delta_metrics") or {}).get("ExactMatch", 0.0) or 0.0),
            float((item.get("oracle_delta_metrics") or {}).get("F1", 0.0) or 0.0),
        ),
        reverse=True,
    )

    policy_alignment: dict[str, dict[str, Any]] = {}
    for label, action_map in policy_action_maps.items():
        covered_queries = 0
        missing_query_count = 0
        keep_count = 0
        executed_swap_count = 0
        oracle_hit_count = 0
        policy_positive_count = 0
        improving_action_match_count = 0
        oracle_positive_recall_count = 0

        for summary in query_summaries:
            question_key = _normalize_question(summary.get("question", ""))
            policy_action = dict(action_map.get(question_key) or {})
            if not policy_action:
                missing_query_count += 1
                continue
            covered_queries += 1
            action_key = tuple(policy_action.get("action_key") or ("keep", None, None))
            executed = bool(policy_action.get("action_executed"))
            if executed:
                executed_swap_count += 1
            else:
                keep_count += 1

            if action_key == oracle_best_action_key_by_question.get(question_key):
                oracle_hit_count += 1

            improving_keys = improving_action_keys_by_question.get(question_key, set())
            if executed:
                policy_positive_count += 1
                if action_key in improving_keys:
                    improving_action_match_count += 1

            if summary.get("oracle_positive_query") and action_key in improving_keys:
                oracle_positive_recall_count += 1

        oracle_positive_total = int(sum(1 for row in query_summaries if row.get("oracle_positive_query")))
        policy_alignment[str(label)] = {
            "covered_queries": int(covered_queries),
            "missing_query_count": int(missing_query_count),
            "executed_swap_count": int(executed_swap_count),
            "keep_count": int(keep_count),
            "oracle_hit_count": int(oracle_hit_count),
            "oracle_hit_rate": _safe_pct(oracle_hit_count, covered_queries),
            "exact_action_match_rate": _safe_pct(oracle_hit_count, covered_queries),
            "oracle_positive_precision": _safe_pct(improving_action_match_count, policy_positive_count),
            "oracle_positive_recall": _safe_pct(oracle_positive_recall_count, oracle_positive_total),
        }

    payload = {
        "metadata": {
            "dataset": str(dataset),
            "candidate_report": str(candidate_report_path),
            "output_json": str(output_json),
            "legality_mode": str(legality_mode),
            "qa_top_k": int(qa_top_k),
            "replace_bottom_n": int(replace_bottom_n),
            "llm_name": str(llm_name),
            "llm_request_name": str(llm_request_name) if llm_request_name else None,
            "llm_base_url": str(llm_base_url),
        },
        "generation": dict(generation_counters),
        "all_actions": {
            "count": int(len(action_results)),
            "keep_count": int(sum(1 for row in action_results if str(row.get("action_type", "keep")) != "swap")),
            "swap_count": int(swap_action_count),
            "positive_swap_count_em": int(positive_swap_count_em),
            "positive_swap_rate_em": _safe_pct(positive_swap_count_em, swap_action_count),
            "positive_swap_count_f1": int(positive_swap_count_f1),
            "positive_swap_rate_f1": _safe_pct(positive_swap_count_f1, swap_action_count),
            "mean_delta_em": _mean(swap_delta_em_all),
            "median_delta_em": _median(swap_delta_em_all),
            "mean_delta_f1": _mean(swap_delta_f1_all),
            "median_delta_f1": _median(swap_delta_f1_all),
        },
        "oracle": {
            "total_queries": int(len(query_summaries)),
            "queries_with_legal_swaps": int(generation_counters.get("queries_with_legal_swaps", 0)),
            "oracle_positive_queries": int(oracle_positive_query_count),
            "oracle_positive_query_rate": _safe_pct(oracle_positive_query_count, len(query_summaries)),
            "baseline_em": _mean(baseline_em_by_query),
            "baseline_f1": _mean(baseline_f1_by_query),
            "oracle_em": _mean(oracle_em_by_query),
            "oracle_f1": _mean(oracle_f1_by_query),
            "oracle_delta_em": (
                _round(float(_mean(oracle_em_by_query) or 0.0) - float(_mean(baseline_em_by_query) or 0.0))
                if query_summaries else None
            ),
            "oracle_delta_f1": (
                _round(float(_mean(oracle_f1_by_query) or 0.0) - float(_mean(baseline_f1_by_query) or 0.0))
                if query_summaries else None
            ),
        },
        "policy_alignment": policy_alignment,
        "queries": query_summaries,
        "action_results": list(action_results),
    }
    return payload


def render_markdown(summary: Mapping[str, Any]) -> str:
    metadata = dict(summary.get("metadata") or {})
    generation = dict(summary.get("generation") or {})
    all_actions = dict(summary.get("all_actions") or {})
    oracle = dict(summary.get("oracle") or {})
    policy_alignment = dict(summary.get("policy_alignment") or {})
    queries = list(summary.get("queries") or [])

    lines = [
        f"# Action-Swap Oracle Ceiling ({metadata.get('dataset', 'unknown')})",
        "",
        f"- candidate report: `{metadata.get('candidate_report', '')}`",
        f"- legality_mode: `{metadata.get('legality_mode', '')}`",
        f"- reader: `{metadata.get('llm_request_name') or metadata.get('llm_name')}` @ `{metadata.get('llm_base_url', '')}`",
        f"- qa_top_k: `{metadata.get('qa_top_k', '')}`",
        f"- replace_bottom_n: `{metadata.get('replace_bottom_n', '')}`",
        "",
        "## Job Generation",
        "",
    ]
    for key in [
        "total_queries",
        "processed_queries",
        "queries_with_appended_candidates",
        "queries_without_appended_candidates",
        "queries_with_legal_swaps",
        "total_keep_jobs",
        "total_swap_jobs",
    ]:
        if key in generation:
            lines.append(f"- {key}: `{generation.get(key, 0)}`")
    for key in sorted(generation):
        if key in {
            "total_queries",
            "processed_queries",
            "queries_with_appended_candidates",
            "queries_without_appended_candidates",
            "queries_with_legal_swaps",
            "total_keep_jobs",
            "total_swap_jobs",
        }:
            continue
        lines.append(f"- {key}: `{generation.get(key, 0)}`")

    lines.extend([
        "",
        "## Summary",
        "",
        "| View | Count | +EM (%) | +F1 (%) | Mean ΔEM | Mean ΔF1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| legal swaps | {int(all_actions.get('swap_count', 0) or 0)} | "
            f"{all_actions.get('positive_swap_rate_em', '—')} | {all_actions.get('positive_swap_rate_f1', '—')} | "
            f"{all_actions.get('mean_delta_em', '—')} | {all_actions.get('mean_delta_f1', '—')} |"
        ),
        (
            f"| oracle best | {int(oracle.get('total_queries', 0) or 0)} | "
            f"{oracle.get('oracle_positive_query_rate', '—')} | {oracle.get('oracle_positive_query_rate', '—')} | "
            f"{oracle.get('oracle_delta_em', '—')} | {oracle.get('oracle_delta_f1', '—')} |"
        ),
        "",
        "## Oracle Aggregate",
        "",
        f"- baseline EM / F1: `{oracle.get('baseline_em', '—')}` / `{oracle.get('baseline_f1', '—')}`",
        f"- oracle EM / F1: `{oracle.get('oracle_em', '—')}` / `{oracle.get('oracle_f1', '—')}`",
        f"- oracle delta EM / F1: `{oracle.get('oracle_delta_em', '—')}` / `{oracle.get('oracle_delta_f1', '—')}`",
    ])

    if policy_alignment:
        lines.extend([
            "",
            "## Policy Alignment",
            "",
            "| Policy | Covered | Swaps | Keep | Oracle Hit (%) | Oracle+ Precision (%) | Oracle+ Recall (%) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, row in policy_alignment.items():
            lines.append(
                f"| {label} | {int(row.get('covered_queries', 0) or 0)} | "
                f"{int(row.get('executed_swap_count', 0) or 0)} | {int(row.get('keep_count', 0) or 0)} | "
                f"{row.get('oracle_hit_rate', '—')} | {row.get('oracle_positive_precision', '—')} | "
                f"{row.get('oracle_positive_recall', '—')} |"
            )

    lines.extend([
        "",
        "## Top Oracle-Positive Queries",
        "",
    ])
    positive_rows = [
        row for row in queries
        if bool(row.get("oracle_positive_query"))
    ][:10]
    if not positive_rows:
        lines.append("- none")
    else:
        for row in positive_rows:
            best = dict(row.get("oracle_best_action") or {})
            delta = dict(row.get("oracle_delta_metrics") or {})
            lines.extend([
                f"- question: `{row.get('question', '')}`",
                f"  keep EM / F1: `{(row.get('keep_metrics') or {}).get('ExactMatch', '—')}` / `{(row.get('keep_metrics') or {}).get('F1', '—')}`",
                (
                    "  oracle action: "
                    f"`{best.get('action_type', 'keep')}` "
                    f"`{best.get('candidate_title') or 'keep'}` "
                    f"(replace `{best.get('replace_incumbent_title') or '—'}`)"
                ),
                f"  oracle delta EM / F1: `{delta.get('ExactMatch', '—')}` / `{delta.get('F1', '—')}`",
            ])

    return "\n".join(lines) + "\n"


def _parse_policy_report_specs(raw_specs: Sequence[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for raw_spec in raw_specs:
        label, separator, path = str(raw_spec).partition("=")
        if separator != "=" or not label.strip() or not path.strip():
            raise ValueError(f"Invalid --policy_report spec: {raw_spec}")
        specs.append((label.strip(), path.strip()))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run oracle ceiling analysis over action_swap_v0-style keep/swap actions.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--save_dir", default="outputs_step0_general")
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--replace_bottom_n", type=int, default=2)
    parser.add_argument("--legality_mode", choices=["current", "relaxed"], default="current")
    parser.add_argument("--policy_report", action="append", default=[])
    parser.add_argument("--llm_name", default="")
    parser.add_argument("--llm_request_name", default="")
    parser.add_argument("--llm_base_url", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dataset = str(args.dataset)

    from src.hipporag.utils.dataset_utils import resolve_dataset_paths

    corpus_path, _ = resolve_dataset_paths(dataset)
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    candidate_report = _load_json(args.candidate_report)
    llm_name, llm_request_name, llm_base_url = infer_llm_config(candidate_report, args)

    run_save_dir = Path(str(args.save_dir).rstrip("/"))
    if not run_save_dir.is_absolute():
        run_save_dir = ROOT_DIR / f"{str(args.save_dir).rstrip('/')}_{dataset}" / "action_swap_oracle_cache"
    run_save_dir.mkdir(parents=True, exist_ok=True)

    jobs, generation_counters = build_oracle_action_jobs(
        dataset=dataset,
        candidate_report=candidate_report,
        corpus=corpus,
        qa_top_k=int(args.qa_top_k),
        replace_bottom_n=int(args.replace_bottom_n),
        legality_mode=str(args.legality_mode),
    )
    logger.info(
        "Prepared oracle action jobs for %s: processed_queries=%d total_keep_jobs=%d total_swap_jobs=%d legality=%s",
        dataset,
        int(generation_counters.get("processed_queries", 0)),
        int(generation_counters.get("total_keep_jobs", 0)),
        int(generation_counters.get("total_swap_jobs", 0)),
        str(args.legality_mode),
    )

    action_results = run_reader(
        jobs,
        dataset=dataset,
        llm_name=llm_name,
        llm_request_name=llm_request_name,
        llm_base_url=llm_base_url,
        save_dir=str(run_save_dir),
        qa_top_k=int(args.qa_top_k),
    )

    policy_action_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in _parse_policy_report_specs(args.policy_report):
        policy_action_maps[label] = extract_policy_actions(_load_json(path))

    summary = summarize_oracle_results(
        dataset=dataset,
        candidate_report_path=args.candidate_report,
        output_json=args.output_json,
        legality_mode=str(args.legality_mode),
        qa_top_k=int(args.qa_top_k),
        replace_bottom_n=int(args.replace_bottom_n),
        llm_name=llm_name,
        llm_request_name=llm_request_name,
        llm_base_url=llm_base_url,
        generation_counters=generation_counters,
        action_results=action_results,
        policy_action_maps=policy_action_maps,
    )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(output_md),
        "processed_queries": int(generation_counters.get("processed_queries", 0)),
        "total_keep_jobs": int(generation_counters.get("total_keep_jobs", 0)),
        "total_swap_jobs": int(generation_counters.get("total_swap_jobs", 0)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
