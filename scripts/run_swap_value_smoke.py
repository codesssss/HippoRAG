#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


logger = logging.getLogger(__name__)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_question(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(float(statistics.median(values)))


def _safe_pct(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return _round(100.0 * float(count) / float(total))


def _build_chunk_id_to_doc_text(corpus: list[dict[str, Any]]) -> dict[str, str]:
    from src.hipporag.utils.misc_utils import compute_mdhash_id

    chunk_id_to_doc_text: dict[str, str] = {}
    for row in corpus:
        doc_text = f"{row['title']}\n{row['text']}"
        chunk_id = compute_mdhash_id(doc_text, prefix="chunk-")
        chunk_id_to_doc_text[chunk_id] = doc_text
    return chunk_id_to_doc_text


def _resolve_doc_text(
    *,
    doc_id: Any,
    corpus: list[dict[str, Any]],
    chunk_id_to_doc_text: dict[str, str],
) -> str | None:
    if isinstance(doc_id, int):
        if 0 <= int(doc_id) < len(corpus):
            row = corpus[int(doc_id)]
            return f"{row['title']}\n{row['text']}"
        return None
    if isinstance(doc_id, str):
        return chunk_id_to_doc_text.get(doc_id)
    return None


def _index_query_traces_by_question(report_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    query_map: dict[str, dict[str, Any]] = {}
    for trace in list(report_payload.get("expand_assemble_query_traces") or []):
        question_key = _normalize_question(trace.get("question", ""))
        if not question_key:
            continue
        query_map[question_key] = dict(trace)
    return query_map


def build_swap_jobs(
    *,
    dataset: str,
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    corpus: list[dict[str, Any]],
    qa_top_k: int,
    replace_bottom_n: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    baseline_by_question = _index_query_traces_by_question(baseline_report)
    candidate_by_question = _index_query_traces_by_question(candidate_report)
    chunk_id_to_doc_text = _build_chunk_id_to_doc_text(corpus)

    counters: Counter[str] = Counter()
    jobs: list[dict[str, Any]] = []
    common_questions = [
        question for question in baseline_by_question
        if question in candidate_by_question
    ]
    counters["aligned_queries"] = len(common_questions)

    for question_key in common_questions:
        baseline_trace = dict(baseline_by_question[question_key] or {})
        candidate_trace = dict(candidate_by_question[question_key] or {})
        baseline_expand = dict(baseline_trace.get("expand_assemble_trace") or {})
        candidate_expand = dict(candidate_trace.get("expand_assemble_trace") or {})

        baseline_front_doc_ids = list(baseline_expand.get("final_front_doc_ids") or [])
        if len(baseline_front_doc_ids) < int(qa_top_k):
            counters["skipped_short_baseline_front"] += 1
            continue

        baseline_docs: list[str] = []
        baseline_titles: list[str] = []
        baseline_doc_ids: list[Any] = []
        baseline_missing_doc = False
        for raw_doc_id in baseline_front_doc_ids[:qa_top_k]:
            doc_text = _resolve_doc_text(doc_id=raw_doc_id, corpus=corpus, chunk_id_to_doc_text=chunk_id_to_doc_text)
            if doc_text is None:
                baseline_missing_doc = True
                break
            baseline_docs.append(doc_text)
            baseline_doc_ids.append(raw_doc_id)
            baseline_titles.append(str(doc_text.split("\n", 1)[0]).strip())
        if baseline_missing_doc:
            counters["skipped_missing_baseline_doc_text"] += 1
            continue

        weakest_index = int(min(max(int(qa_top_k), 1), len(baseline_docs)) - 1)
        weakest_doc_id = baseline_doc_ids[weakest_index]
        weakest_title = baseline_titles[weakest_index]
        replace_count = max(1, int(replace_bottom_n))
        replace_indices = list(range(max(0, len(baseline_docs) - replace_count), len(baseline_docs)))

        ranking_rows = list((candidate_expand.get("assemble_trace") or {}).get("ranking_rows") or [])
        rank_by_pool_position = {
            int(row.get("pool_position")): dict(row)
            for row in ranking_rows
            if row.get("pool_position") is not None
        }
        rank_by_doc_id = {}
        for row in ranking_rows:
            doc_id = row.get("doc_id")
            if doc_id is None or doc_id in rank_by_doc_id:
                continue
            rank_by_doc_id[doc_id] = dict(row)
        appended_positions = [int(pos) for pos in (candidate_expand.get("appended_positions") or [])]
        if not appended_positions:
            counters["queries_without_appended_candidates"] += 1
            continue

        query_jobs = 0
        for appended_pool_position in appended_positions:
            candidate_row = rank_by_pool_position.get(int(appended_pool_position))
            if candidate_row is None:
                counters["skipped_missing_candidate_row"] += 1
                continue
            candidate_doc_id = candidate_row.get("doc_id")
            candidate_doc_text = _resolve_doc_text(
                doc_id=candidate_doc_id,
                corpus=corpus,
                chunk_id_to_doc_text=chunk_id_to_doc_text,
            )
            if candidate_doc_text is None:
                counters["skipped_missing_candidate_doc_text"] += 1
                continue
            if candidate_doc_text in baseline_docs:
                counters["skipped_duplicate_candidate_doc"] += 1
                continue

            for replace_index in replace_indices:
                replace_doc_id = baseline_doc_ids[replace_index]
                replace_title = baseline_titles[replace_index]
                replace_rank_row = rank_by_doc_id.get(replace_doc_id, {})
                swapped_docs = list(baseline_docs)
                swapped_docs[replace_index] = candidate_doc_text
                query_jobs += 1
                jobs.append({
                    "dataset": dataset,
                    "question": str(baseline_trace.get("question", "")),
                    "query_type": str(baseline_trace.get("query_type", "")),
                    "gold_answers": list(baseline_trace.get("gold_answers") or []),
                    "baseline_answer": str(baseline_trace.get("method_answer", "")),
                    "baseline_metrics": dict(baseline_trace.get("method_metrics") or {}),
                    "baseline_top_titles": list(baseline_trace.get("method_top_titles") or []),
                    "baseline_final_doc_ids": list(baseline_doc_ids),
                    "baseline_docs": list(baseline_docs),
                    "baseline_doc_titles": list(baseline_titles),
                    "replace_bottom_n": int(replace_count),
                    "replace_incumbent_index": int(replace_index),
                    "replace_incumbent_rank": int(replace_index + 1),
                    "replace_incumbent_doc_id": replace_doc_id,
                    "replace_incumbent_title": replace_title,
                    "replace_incumbent_ce_rank": (
                        int(replace_rank_row.get("rank", 0) or 0) if replace_rank_row else None
                    ),
                    "replace_incumbent_ce_score": (
                        _round(float(replace_rank_row.get("assemble_score", 0.0) or 0.0)) if replace_rank_row else None
                    ),
                    "weakest_incumbent_index": int(weakest_index),
                    "weakest_incumbent_doc_id": weakest_doc_id,
                    "weakest_incumbent_title": weakest_title,
                    "candidate_doc_id": candidate_doc_id,
                    "candidate_title": str(candidate_row.get("title", "")),
                    "candidate_source": str(candidate_row.get("source", "")),
                    "candidate_pool_position": int(appended_pool_position),
                    "candidate_ce_rank": int(candidate_row.get("rank", 0) or 0),
                    "candidate_ce_score": _round(float(candidate_row.get("assemble_score", 0.0) or 0.0)),
                    "candidate_base_score": _round(float(candidate_row.get("base_score", 0.0) or 0.0)),
                    "candidate_entered_method_topk": bool(
                        candidate_doc_id in set(candidate_expand.get("final_front_doc_ids") or [])
                    ),
                    "docs": swapped_docs,
                    "swapped_top_titles": [str(doc.split("\n", 1)[0]).strip() for doc in swapped_docs],
                })

        if query_jobs > 0:
            counters["queries_with_swap_jobs"] += 1
            counters["total_swap_jobs"] += query_jobs

    return jobs, dict(counters)


def run_reader(jobs: list[dict[str, Any]], *, dataset: str, llm_name: str, llm_request_name: str | None,
               llm_base_url: str, save_dir: str, qa_top_k: int) -> list[dict[str, Any]]:
    from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
    from src.hipporag.llm import _get_llm_class
    from src.hipporag.prompts.prompt_template_manager import PromptTemplateManager
    from src.hipporag.utils.config_utils import BaseConfig
    from src.hipporag.utils.misc_utils import extract_answer_from_response

    config = BaseConfig(
        dataset=dataset,
        llm_name=str(llm_name),
        llm_request_name=str(llm_request_name) if llm_request_name else None,
        llm_base_url=str(llm_base_url),
        save_dir=str(save_dir),
        qa_top_k=int(qa_top_k),
        max_retry_attempts=5,
        temperature=0.0,
        openie_mode="online",
    )
    llm_model = _get_llm_class(config)
    prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
    qa_em = QAExactMatch(global_config=config)
    qa_f1 = QAF1Score(global_config=config)

    results: list[dict[str, Any]] = []
    total_jobs = len(jobs)
    prompt_dataset_name = dataset if prompt_template_manager.is_template_name_valid(name=f"rag_qa_{dataset}") else "musique"

    for idx, job in enumerate(jobs, start=1):
        prompt_user = ""
        for passage in list(job.get("docs") or [])[:qa_top_k]:
            prompt_user += f"Wikipedia Title: {passage}\n\n"
        prompt_user += f"Question: {job['question']}\nThought: "
        qa_messages = prompt_template_manager.render(name=f"rag_qa_{prompt_dataset_name}", prompt_user=prompt_user)

        logger.info(
            "Reader swap %d/%d | dataset=%s | ce_rank=%s | candidate=%s",
            idx,
            total_jobs,
            dataset,
            str(job.get("candidate_ce_rank")),
            str(job.get("candidate_title", ""))[:120],
        )
        try:
            response_message, metadata, cache_hit = llm_model.infer(qa_messages)
        except Exception as exc:  # pragma: no cover - runtime safety
            response_message = ""
            metadata = {
                "reader_status": "llm_exception",
                "reader_error": str(exc),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "exception",
            }
            cache_hit = False

        pred_answer, parse_info = extract_answer_from_response(response_message)
        gold_answers = [list(job.get("gold_answers") or [])]
        _, per_query_em = qa_em.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=[pred_answer],
            aggregation_fn=max,
        )
        _, per_query_f1 = qa_f1.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=[pred_answer],
            aggregation_fn=max,
        )
        swap_em = float(per_query_em[0].get("ExactMatch", 0.0))
        swap_f1 = float(per_query_f1[0].get("F1", 0.0))
        baseline_em = float((job.get("baseline_metrics") or {}).get("ExactMatch", 0.0) or 0.0)
        baseline_f1 = float((job.get("baseline_metrics") or {}).get("F1", 0.0) or 0.0)

        result = dict(job)
        result.update({
            "swap_answer": pred_answer,
            "swap_metrics": {
                "ExactMatch": _round(swap_em),
                "F1": _round(swap_f1),
            },
            "delta_metrics": {
                "ExactMatch": _round(swap_em - baseline_em),
                "F1": _round(swap_f1 - baseline_f1),
            },
            "reader_trace": {
                "reader_status": str(metadata.get("reader_status", "ok")),
                "reader_error": metadata.get("reader_error"),
                "reader_cache_hit": bool(cache_hit),
                "answer_parser_used_fallback": bool(parse_info.get("used_fallback")),
                "answer_parser_error_type": parse_info.get("error_type"),
                "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
                "finish_reason": str(metadata.get("finish_reason", "")),
            },
        })
        results.append(result)

    return results


def summarize_results(
    *,
    dataset: str,
    baseline_report_path: str,
    candidate_report_path: str,
    output_json: str,
    qa_top_k: int,
    llm_name: str,
    llm_request_name: str | None,
    llm_base_url: str,
    generation_counters: dict[str, int],
    swap_results: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    all_delta_em: list[float] = []
    all_delta_f1: list[float] = []
    positive_swap_count_em = 0
    positive_swap_count_f1 = 0
    ce_ranks_all: list[float] = []
    ce_ranks_positive: list[float] = []
    queries_with_positive_em = 0
    queries_with_positive_f1 = 0
    baseline_em_by_query: list[float] = []
    baseline_f1_by_query: list[float] = []
    oracle_em_by_query: list[float] = []
    oracle_f1_by_query: list[float] = []
    query_summaries: list[dict[str, Any]] = []

    for row in swap_results:
        grouped.setdefault(_normalize_question(row.get("question", "")), []).append(row)
        delta_em = float((row.get("delta_metrics") or {}).get("ExactMatch", 0.0) or 0.0)
        delta_f1 = float((row.get("delta_metrics") or {}).get("F1", 0.0) or 0.0)
        all_delta_em.append(delta_em)
        all_delta_f1.append(delta_f1)
        ce_rank = float(row.get("candidate_ce_rank", 0) or 0)
        ce_ranks_all.append(ce_rank)
        if delta_em > 0:
            positive_swap_count_em += 1
            ce_ranks_positive.append(ce_rank)
        if delta_f1 > 0:
            positive_swap_count_f1 += 1

    for question_key, rows in grouped.items():
        rows_sorted = sorted(
            rows,
            key=lambda item: (
                float((item.get("swap_metrics") or {}).get("ExactMatch", 0.0) or 0.0),
                float((item.get("swap_metrics") or {}).get("F1", 0.0) or 0.0),
                -float(item.get("candidate_ce_rank", 10_000) or 10_000),
            ),
            reverse=True,
        )
        best_row = rows_sorted[0]
        baseline_metrics = dict(best_row.get("baseline_metrics") or {})
        baseline_em = float(baseline_metrics.get("ExactMatch", 0.0) or 0.0)
        baseline_f1 = float(baseline_metrics.get("F1", 0.0) or 0.0)
        oracle_em = float((best_row.get("swap_metrics") or {}).get("ExactMatch", 0.0) or 0.0)
        oracle_f1 = float((best_row.get("swap_metrics") or {}).get("F1", 0.0) or 0.0)
        baseline_em_by_query.append(baseline_em)
        baseline_f1_by_query.append(baseline_f1)
        oracle_em_by_query.append(oracle_em)
        oracle_f1_by_query.append(oracle_f1)
        if oracle_em > baseline_em:
            queries_with_positive_em += 1
        if oracle_f1 > baseline_f1:
            queries_with_positive_f1 += 1
        query_summaries.append({
            "question": best_row.get("question"),
            "query_type": best_row.get("query_type"),
            "baseline_answer": best_row.get("baseline_answer"),
            "baseline_metrics": {
                "ExactMatch": _round(baseline_em),
                "F1": _round(baseline_f1),
            },
            "best_swap": {
                "candidate_title": best_row.get("candidate_title"),
                "candidate_ce_rank": int(best_row.get("candidate_ce_rank", 0) or 0),
                "candidate_ce_score": best_row.get("candidate_ce_score"),
                "swap_answer": best_row.get("swap_answer"),
                "swap_metrics": dict(best_row.get("swap_metrics") or {}),
                "delta_metrics": dict(best_row.get("delta_metrics") or {}),
                "weakest_incumbent_title": best_row.get("weakest_incumbent_title"),
                "candidate_entered_method_topk": bool(best_row.get("candidate_entered_method_topk")),
            },
            "num_swap_candidates": len(rows),
            "has_positive_swap_em": oracle_em > baseline_em,
            "has_positive_swap_f1": oracle_f1 > baseline_f1,
        })

    query_summaries.sort(
        key=lambda item: (
            float(((item.get("best_swap") or {}).get("delta_metrics") or {}).get("ExactMatch", 0.0) or 0.0),
            float(((item.get("best_swap") or {}).get("delta_metrics") or {}).get("F1", 0.0) or 0.0),
        ),
        reverse=True,
    )

    payload = {
        "metadata": {
            "dataset": dataset,
            "baseline_report": str(baseline_report_path),
            "candidate_report": str(candidate_report_path),
            "output_json": str(output_json),
            "qa_top_k": int(qa_top_k),
            "llm_name": str(llm_name),
            "llm_request_name": str(llm_request_name) if llm_request_name else None,
            "llm_base_url": str(llm_base_url),
        },
        "generation": dict(generation_counters),
        "all_swaps": {
            "count": len(swap_results),
            "positive_swap_count_em": int(positive_swap_count_em),
            "positive_swap_rate_em": _safe_pct(positive_swap_count_em, len(swap_results)),
            "positive_swap_count_f1": int(positive_swap_count_f1),
            "positive_swap_rate_f1": _safe_pct(positive_swap_count_f1, len(swap_results)),
            "mean_delta_em": _mean(all_delta_em),
            "median_delta_em": _median(all_delta_em),
            "mean_delta_f1": _mean(all_delta_f1),
            "median_delta_f1": _median(all_delta_f1),
            "mean_candidate_ce_rank": _mean(ce_ranks_all),
            "mean_positive_candidate_ce_rank": _mean(ce_ranks_positive),
        },
        "best_swap_oracle": {
            "num_queries_with_candidates": len(grouped),
            "queries_with_positive_em": int(queries_with_positive_em),
            "queries_with_positive_em_rate": _safe_pct(queries_with_positive_em, len(grouped)),
            "queries_with_positive_f1": int(queries_with_positive_f1),
            "queries_with_positive_f1_rate": _safe_pct(queries_with_positive_f1, len(grouped)),
            "baseline_em": _mean(baseline_em_by_query),
            "oracle_em": _mean(oracle_em_by_query),
            "oracle_delta_em": (
                _round(float(_mean(oracle_em_by_query) or 0.0) - float(_mean(baseline_em_by_query) or 0.0))
                if grouped else None
            ),
            "baseline_f1": _mean(baseline_f1_by_query),
            "oracle_f1": _mean(oracle_f1_by_query),
            "oracle_delta_f1": (
                _round(float(_mean(oracle_f1_by_query) or 0.0) - float(_mean(baseline_f1_by_query) or 0.0))
                if grouped else None
            ),
        },
        "queries": query_summaries,
        "swap_results": swap_results,
    }
    return payload


def render_markdown(summary: dict[str, Any]) -> str:
    metadata = dict(summary.get("metadata") or {})
    generation = dict(summary.get("generation") or {})
    all_swaps = dict(summary.get("all_swaps") or {})
    oracle = dict(summary.get("best_swap_oracle") or {})
    queries = list(summary.get("queries") or [])

    lines = [
        f"# Swap-Value Smoke Test ({metadata.get('dataset', 'unknown')})",
        "",
        f"- baseline report: `{metadata.get('baseline_report', '')}`",
        f"- candidate report: `{metadata.get('candidate_report', '')}`",
        f"- reader: `{metadata.get('llm_request_name') or metadata.get('llm_name')}` @ `{metadata.get('llm_base_url', '')}`",
        f"- qa_top_k: `{metadata.get('qa_top_k', '')}`",
        "",
        "## Job Generation",
        "",
        f"- aligned queries: `{generation.get('aligned_queries', 0)}`",
        f"- queries with swap jobs: `{generation.get('queries_with_swap_jobs', 0)}`",
        f"- total swap jobs: `{generation.get('total_swap_jobs', 0)}`",
    ]
    for key in sorted(generation):
        if key in {"aligned_queries", "queries_with_swap_jobs", "total_swap_jobs"}:
            continue
        lines.append(f"- {key}: `{generation.get(key, 0)}`")

    lines.extend([
        "",
        "## Summary",
        "",
        "| View | Count | +EM (%) | +F1 (%) | Mean ΔEM | Mean ΔF1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| all swaps | {int(all_swaps.get('count', 0) or 0)} | "
            f"{all_swaps.get('positive_swap_rate_em', '—')} | {all_swaps.get('positive_swap_rate_f1', '—')} | "
            f"{all_swaps.get('mean_delta_em', '—')} | {all_swaps.get('mean_delta_f1', '—')} |"
        ),
        (
            f"| best-swap oracle | {int(oracle.get('num_queries_with_candidates', 0) or 0)} | "
            f"{oracle.get('queries_with_positive_em_rate', '—')} | {oracle.get('queries_with_positive_f1_rate', '—')} | "
            f"{oracle.get('oracle_delta_em', '—')} | {oracle.get('oracle_delta_f1', '—')} |"
        ),
        "",
        "## Oracle Aggregate",
        "",
        f"- baseline EM / F1: `{oracle.get('baseline_em', '—')}` / `{oracle.get('baseline_f1', '—')}`",
        f"- oracle EM / F1: `{oracle.get('oracle_em', '—')}` / `{oracle.get('oracle_f1', '—')}`",
        f"- oracle delta EM / F1: `{oracle.get('oracle_delta_em', '—')}` / `{oracle.get('oracle_delta_f1', '—')}`",
        "",
        "## Top Positive Best-Swap Queries",
        "",
    ])

    positive_rows = [
        row for row in queries
        if float(((row.get("best_swap") or {}).get("delta_metrics") or {}).get("ExactMatch", 0.0) or 0.0) > 0
        or float(((row.get("best_swap") or {}).get("delta_metrics") or {}).get("F1", 0.0) or 0.0) > 0
    ][:10]
    if not positive_rows:
        lines.append("- none")
    else:
        for row in positive_rows:
            best_swap = dict(row.get("best_swap") or {})
            delta = dict(best_swap.get("delta_metrics") or {})
            lines.extend([
                f"- question: `{row.get('question', '')}`",
                f"  baseline -> swap: `{row.get('baseline_answer', '')}` -> `{best_swap.get('swap_answer', '')}`",
                (
                    "  candidate: "
                    f"`{best_swap.get('candidate_title', '')}` "
                    f"(CE rank `{best_swap.get('candidate_ce_rank', '—')}`, "
                    f"replace `{best_swap.get('weakest_incumbent_title', '')}`)"
                ),
                f"  delta EM / F1: `{delta.get('ExactMatch', '—')}` / `{delta.get('F1', '—')}`",
            ])

    return "\n".join(lines) + "\n"


def infer_llm_config(report_payload: dict[str, Any], args: argparse.Namespace) -> tuple[str, str | None, str]:
    llm_name = str(args.llm_name or report_payload.get("llm_name") or "qwen3-8b")
    llm_request_name = str(args.llm_request_name or report_payload.get("llm_request_name") or "").strip() or None
    llm_base_url = str(args.llm_base_url or report_payload.get("llm_base_url") or "http://localhost:8043/v1")
    return llm_name, llm_request_name, llm_base_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a swap-value smoke test from existing baseline/candidate reports.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline_report", required=True)
    parser.add_argument("--candidate_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--save_dir", default="outputs_step0_general")
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--max_jobs", type=int, default=0)
    parser.add_argument("--llm_name", default="")
    parser.add_argument("--llm_request_name", default="")
    parser.add_argument("--llm_base_url", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dataset = str(args.dataset)

    from src.hipporag.utils.dataset_utils import resolve_dataset_paths

    corpus_path, _ = resolve_dataset_paths(dataset)
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    logger.info("Loaded corpus for %s from %s", dataset, corpus_path)

    baseline_report = _load_json(args.baseline_report)
    candidate_report = _load_json(args.candidate_report)
    llm_name, llm_request_name, llm_base_url = infer_llm_config(candidate_report, args)
    run_save_dir = Path(str(args.save_dir).rstrip("/"))
    if not run_save_dir.is_absolute():
        run_save_dir = ROOT_DIR / f"{str(args.save_dir).rstrip('/')}_{dataset}" / "swap_value_smoke_cache"
    run_save_dir.mkdir(parents=True, exist_ok=True)

    jobs, generation_counters = build_swap_jobs(
        dataset=dataset,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        corpus=corpus,
        qa_top_k=int(args.qa_top_k),
    )
    logger.info(
        "Prepared swap jobs for %s: queries_with_jobs=%d total_jobs=%d",
        dataset,
        int(generation_counters.get("queries_with_swap_jobs", 0)),
        int(generation_counters.get("total_swap_jobs", 0)),
    )
    if int(args.max_jobs or 0) > 0:
        jobs = jobs[: int(args.max_jobs)]
        generation_counters["total_swap_jobs"] = len(jobs)
        logger.info("Truncated swap jobs for dry run: dataset=%s max_jobs=%d", dataset, len(jobs))
    swap_results = run_reader(
        jobs,
        dataset=dataset,
        llm_name=llm_name,
        llm_request_name=llm_request_name,
        llm_base_url=llm_base_url,
        save_dir=str(run_save_dir),
        qa_top_k=int(args.qa_top_k),
    )
    summary = summarize_results(
        dataset=dataset,
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
        output_json=args.output_json,
        qa_top_k=int(args.qa_top_k),
        llm_name=llm_name,
        llm_request_name=llm_request_name,
        llm_base_url=llm_base_url,
        generation_counters=generation_counters,
        swap_results=swap_results,
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
        "total_swap_jobs": int(generation_counters.get("total_swap_jobs", 0)),
        "queries_with_jobs": int(generation_counters.get("queries_with_swap_jobs", 0)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
