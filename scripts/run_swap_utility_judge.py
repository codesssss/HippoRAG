#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import SETWISE_LLM_NO_THINK_PREFIX, build_setwise_late_rerank_judge_bundle
from run_swap_value_smoke import _load_json, _normalize_question, _round, _safe_pct, build_swap_jobs
from src.hipporag.utils.dataset_utils import resolve_dataset_paths


logger = logging.getLogger(__name__)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(?P<value>helpful|neutral|harmful)\s*$", re.IGNORECASE | re.MULTILINE)
CONFIDENCE_RE = re.compile(r"^\s*CONFIDENCE\s*:\s*(?P<value>-?\d+(?:\.\d+)?)\s*$", re.IGNORECASE | re.MULTILINE)
REASON_RE = re.compile(r"^\s*REASON\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
CLAIM_SUPPORT_VERDICT_RE = re.compile(
    r"^\s*VERDICT\s*:\s*(?P<value>supported|not_supported)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
MISSING_INFO_TYPE_RE = re.compile(
    r"^\s*MISSING_INFO_TYPE\s*:\s*"
    r"(?P<value>bridge_relation|entity_grounding|redundancy_repair|broader_context|semantic_drift|none)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
EVIDENCE_SENTENCES_RE = re.compile(
    r"^\s*EVIDENCE_SENTENCES\s*:\s*candidate=(?P<candidate>\[[^\]]*\])\s+incumbent=(?P<incumbent>\[[^\]]*\])\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ALLOWED_MISSING_INFO_TYPES = {
    "bridge_relation",
    "entity_grounding",
    "redundancy_repair",
    "broader_context",
    "semantic_drift",
    "none",
}


def _truncate_text(text: str, *, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= int(max_chars):
        return normalized
    return normalized[: max(int(max_chars) - 3, 0)].rstrip() + "..."


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(float(statistics.median(values)))


def _split_doc_text(doc_text: str) -> tuple[str, str]:
    return (str(doc_text).split("\n", 1) + [""])[:2]


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(str(text or ""))]
    sentences = [part for part in parts if part]
    return sentences or ([str(text or "").strip()] if str(text or "").strip() else [])


def build_doc_preview(doc_text: str, *, max_sentences: int = 2, max_chars: int = 320) -> str:
    title, body = _split_doc_text(doc_text)
    sentences = split_sentences(body)
    preview = " ".join(sentences[: max(int(max_sentences), 1)]).strip()
    if not preview:
        preview = title.strip()
    if len(preview) > int(max_chars):
        preview = preview[: max(int(max_chars) - 3, 0)].rstrip() + "..."
    return preview


def _parse_evidence_list(value: str) -> tuple[list[int], list[str]]:
    raw = str(value or "").strip()
    if not raw.startswith("[") or not raw.endswith("]"):
        raise ValueError(f"Expected bracketed list, got: {value}")
    inner = raw[1:-1].strip()
    if not inner:
        return [], []
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        if all(isinstance(item, int) for item in parsed):
            return [int(item) for item in parsed], []
        if all(isinstance(item, str) for item in parsed):
            return [], [str(item).strip() for item in parsed if str(item).strip()]

    parts = [part.strip() for part in inner.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"Malformed evidence list: {value}")
    try:
        return [int(part) for part in parts], []
    except ValueError:
        cleaned = [part.strip("\"' ") for part in parts if part.strip("\"' ")]
        if cleaned:
            return [], cleaned
        raise ValueError(f"Unsupported evidence list payload: {value}")


def parse_swap_utility_judge_response(response_text: str) -> dict[str, Any]:
    text = str(response_text or "").strip()
    parsed: dict[str, Any] = {
        "raw_response": text,
        "parse_succeeded": False,
        "parse_errors": [],
        "verdict": None,
        "confidence": None,
        "reason": None,
        "missing_info_type": None,
        "candidate_evidence_sentences": [],
        "incumbent_evidence_sentences": [],
        "candidate_evidence_snippets": [],
        "incumbent_evidence_snippets": [],
    }
    if not text:
        parsed["parse_errors"] = ["empty_response"]
        return parsed

    verdict_match = VERDICT_RE.search(text)
    confidence_match = CONFIDENCE_RE.search(text)
    reason_match = REASON_RE.search(text)
    missing_info_match = MISSING_INFO_TYPE_RE.search(text)
    evidence_match = EVIDENCE_SENTENCES_RE.search(text)

    if verdict_match is None:
        parsed["parse_errors"].append("missing_verdict")
    else:
        parsed["verdict"] = str(verdict_match.group("value")).strip().lower()

    if confidence_match is None:
        parsed["parse_errors"].append("missing_confidence")
    else:
        confidence = float(confidence_match.group("value"))
        if confidence < 0.0 or confidence > 100.0:
            parsed["parse_errors"].append("confidence_out_of_range")
        else:
            parsed["confidence"] = _round(confidence, 2)

    if reason_match is None:
        parsed["parse_errors"].append("missing_reason")
    else:
        parsed["reason"] = str(reason_match.group("value")).strip()

    if missing_info_match is None:
        parsed["parse_errors"].append("missing_info_type")
    else:
        missing_info_type = str(missing_info_match.group("value")).strip().lower()
        if missing_info_type not in ALLOWED_MISSING_INFO_TYPES:
            parsed["parse_errors"].append("invalid_missing_info_type")
        else:
            parsed["missing_info_type"] = missing_info_type

    if evidence_match is None:
        parsed["parse_errors"].append("missing_evidence_sentences")
    else:
        try:
            candidate_sentences, candidate_snippets = _parse_evidence_list(evidence_match.group("candidate"))
            incumbent_sentences, incumbent_snippets = _parse_evidence_list(evidence_match.group("incumbent"))
            parsed["candidate_evidence_sentences"] = candidate_sentences
            parsed["incumbent_evidence_sentences"] = incumbent_sentences
            parsed["candidate_evidence_snippets"] = candidate_snippets
            parsed["incumbent_evidence_snippets"] = incumbent_snippets
        except ValueError as exc:
            parsed["parse_errors"].append(f"invalid_evidence_sentences:{exc}")

    parsed["parse_succeeded"] = not parsed["parse_errors"]
    return parsed


def parse_claim_support_verifier_response(response_text: str) -> dict[str, Any]:
    text = str(response_text or "").strip()
    parsed: dict[str, Any] = {
        "raw_response": text,
        "parse_succeeded": False,
        "parse_errors": [],
        "verdict": None,
        "reason": None,
        "supported": False,
    }
    if not text:
        parsed["parse_errors"] = ["empty_response"]
        return parsed

    verdict_match = CLAIM_SUPPORT_VERDICT_RE.search(text)
    reason_match = REASON_RE.search(text)

    if verdict_match is None:
        parsed["parse_errors"].append("missing_verdict")
    else:
        verdict = str(verdict_match.group("value")).strip().lower()
        parsed["verdict"] = verdict
        parsed["supported"] = verdict == "supported"

    if reason_match is None:
        parsed["parse_errors"].append("missing_reason")
    else:
        parsed["reason"] = str(reason_match.group("value")).strip()

    parsed["parse_succeeded"] = not parsed["parse_errors"]
    return parsed


def build_swap_utility_messages(job: dict[str, Any], *, qa_top_k: int, max_doc_chars: int) -> list[dict[str, str]]:
    scaffold_docs = list(job.get("baseline_docs") or [])
    scaffold_lines: list[str] = []
    for idx, doc_text in enumerate(scaffold_docs[:qa_top_k], start=1):
        title = str(doc_text.split("\n", 1)[0]).strip()
        preview = build_doc_preview(doc_text, max_chars=max_doc_chars)
        scaffold_lines.append(f"{idx}. [{title}] {preview}")

    candidate_doc_text = ""
    for baseline_doc_text, swapped_doc_text in zip(scaffold_docs, list(job.get("docs") or [])):
        if baseline_doc_text != swapped_doc_text:
            candidate_doc_text = str(swapped_doc_text)
            break
    if not candidate_doc_text and list(job.get("docs") or []):
        candidate_doc_text = str((job.get("docs") or [])[-1])

    incumbent_title = str(job.get("replace_incumbent_title") or "")
    incumbent_preview = ""
    replace_index = int(job.get("replace_incumbent_index", max(int(qa_top_k) - 1, 0)) or 0)
    if 0 <= replace_index < len(scaffold_docs):
        incumbent_preview = build_doc_preview(scaffold_docs[replace_index], max_chars=max_doc_chars)

    candidate_title = str(job.get("candidate_title") or "")
    candidate_preview = build_doc_preview(candidate_doc_text, max_chars=max_doc_chars)

    prompt = "\n".join([
        "You are evaluating whether a document swap would improve a fixed-budget evidence set for multi-hop question answering.",
        "",
        f"Question: {job.get('question', '')}",
        "",
        "Current evidence set (top-5):",
        *scaffold_lines,
        "",
        "Proposed swap:",
        f"Replace document #{int(job.get('replace_incumbent_rank', replace_index + 1) or (replace_index + 1))}:",
        f"[{incumbent_title}] {incumbent_preview}",
        "",
        "with candidate:",
        f"[{candidate_title}] {candidate_preview}",
        "",
        "Task:",
        "Compared with keeping the current evidence set unchanged, judge whether replacing the incumbent with the candidate would make the evidence set strictly better, about the same, or worse for answering the question.",
        "",
        "Use these definitions:",
        "- helpful: the candidate adds missing information that is likely needed for answering, without removing more important support",
        "- neutral: the candidate is somewhat related but mostly adds broad context, repetition, or nearby-but-not-needed information",
        "- harmful: the candidate causes semantic drift or removes more query-local / answer-facing support",
        "",
        "Respond in this exact format:",
        "VERDICT: helpful / neutral / harmful",
        "CONFIDENCE: 0-100",
        "REASON: [one sentence]",
        "MISSING_INFO_TYPE: bridge_relation / entity_grounding / redundancy_repair / broader_context / semantic_drift / none",
        "EVIDENCE_SENTENCES: candidate=[...] incumbent=[...]",
    ])
    return [
        {
            "role": "system",
            "content": "You are a careful multi-hop QA evidence judge. Focus on swap utility, not candidate quality in isolation.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def build_claim_support_messages(*,
                                 question: str,
                                 claim_text: str,
                                 doc_title: str,
                                 witness_text: str,
                                 witness_unit_type: str,
                                 max_doc_chars: int) -> list[dict[str, str]]:
    witness_preview = _truncate_text(witness_text, max_chars=max_doc_chars)
    prompt = "\n".join([
        SETWISE_LLM_NO_THINK_PREFIX,
        "You are verifying whether a candidate evidence snippet directly supports a single multi-hop QA claim.",
        "",
        f"Question: {question}",
        f"Claim: {claim_text}",
        "",
        f"Document title: {doc_title}",
        f"Witness unit type: {witness_unit_type}",
        "Witness snippet:",
        witness_preview,
        "",
        "Task:",
        "Return supported only if the snippet directly supports the claim on its own or with obvious coreference resolution inside the snippet.",
        "Return not_supported if the snippet is merely related, broad context, entity-nearby, or requires unsupported extra inference.",
        "Do not think aloud. Do not output analysis. Do not use <think> tags.",
        "",
        "Respond with exactly two lines in this exact format and nothing else:",
        "VERDICT: supported / not_supported",
        "REASON: [one sentence]",
    ])
    return [
        {
            "role": "system",
            "content": (
                "You are a careful evidence verifier. "
                "Decide only whether the provided snippet directly supports the claim. "
                "Return only the final verdict format. Never emit reasoning or <think> content."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def run_swap_utility_judge(
    jobs: list[dict[str, Any]],
    *,
    judge_bundle: Any,
    qa_top_k: int,
    max_doc_chars: int,
    max_completion_tokens: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total_jobs = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        messages = build_swap_utility_messages(job, qa_top_k=qa_top_k, max_doc_chars=max_doc_chars)
        logger.info(
            "Judge swap %d/%d | question=%s | candidate=%s | replace_rank=%s",
            idx,
            total_jobs,
            str(job.get("question", ""))[:100],
            str(job.get("candidate_title", ""))[:120],
            str(job.get("replace_incumbent_rank", "")),
        )
        response_text = ""
        metadata: dict[str, Any] = {}
        judge_error = None
        try:
            response_text, metadata = judge_bundle.infer_fn(
                messages=messages,
                model=judge_bundle.model_name,
                response_format=judge_bundle.response_format,
                max_completion_tokens=int(max_completion_tokens),
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            judge_error = str(exc)
            metadata = {
                "judge_status": "judge_exception",
                "judge_error": judge_error,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "exception",
            }

        parsed = parse_swap_utility_judge_response(response_text)
        result = dict(job)
        result.update({
            "judge_response": response_text,
            "judge_trace": {
                "judge_status": str(metadata.get("judge_status", "ok")),
                "judge_error": metadata.get("judge_error") or judge_error,
                "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
                "finish_reason": str(metadata.get("finish_reason", "")),
                "backend": metadata.get("backend"),
                "model": metadata.get("model"),
                "response_text_source": metadata.get("response_text_source"),
            },
            "judge_parsed": parsed,
        })
        results.append(result)
    return results


def _build_oracle_match_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        _normalize_question(str(row.get("question", ""))),
        str(row.get("candidate_title", "")),
        str(row.get("replace_incumbent_title") or row.get("weakest_incumbent_title") or ""),
        int(row.get("replace_incumbent_rank", row.get("weakest_incumbent_index", 0) + 1) or 0),
    )


def _index_oracle_rows(oracle_payload: dict[str, Any]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in list(oracle_payload.get("swap_results") or []):
        indexed[_build_oracle_match_key(row)] = dict(row)
    return indexed


def summarize_swap_utility_results(
    *,
    dataset: str,
    baseline_report_path: str,
    candidate_report_path: str,
    output_json: str,
    judge_bundle: Any,
    generation_counters: dict[str, int],
    judge_results: list[dict[str, Any]],
    replace_bottom_n: int,
    oracle_payload: dict[str, Any] | None,
    confidence_threshold: float,
) -> dict[str, Any]:
    verdict_counter: Counter[str] = Counter()
    missing_info_counter: Counter[str] = Counter()
    confidence_by_verdict: dict[str, list[float]] = {"helpful": [], "neutral": [], "harmful": []}
    parse_failure_count = 0
    grouped: dict[str, list[dict[str, Any]]] = {}

    oracle_rows_by_key = _index_oracle_rows(oracle_payload) if oracle_payload else {}
    oracle_matched_jobs = 0
    oracle_positive_em = 0
    oracle_positive_f1 = 0
    helpful_on_positive_em = 0
    helpful_on_positive_f1 = 0
    helpful_total_with_oracle = 0
    high_ce_nonpositive = 0
    judge_blocks_high_ce_nonpositive = 0
    helpful_oracle_delta_em: list[float] = []
    helpful_oracle_delta_f1: list[float] = []

    for row in judge_results:
        grouped.setdefault(_normalize_question(row.get("question", "")), []).append(row)
        parsed = dict(row.get("judge_parsed") or {})
        if not bool(parsed.get("parse_succeeded")):
            parse_failure_count += 1
            continue
        verdict = str(parsed.get("verdict", "")).lower()
        verdict_counter[verdict] += 1
        missing_info_type = str(parsed.get("missing_info_type", "")).lower()
        if missing_info_type:
            missing_info_counter[missing_info_type] += 1
        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)) and verdict in confidence_by_verdict:
            confidence_by_verdict[verdict].append(float(confidence))

        if oracle_rows_by_key:
            oracle_row = oracle_rows_by_key.get(_build_oracle_match_key(row))
            if oracle_row is not None:
                oracle_matched_jobs += 1
                oracle_delta_em = float((oracle_row.get("delta_metrics") or {}).get("ExactMatch", 0.0) or 0.0)
                oracle_delta_f1 = float((oracle_row.get("delta_metrics") or {}).get("F1", 0.0) or 0.0)
                is_positive_em = oracle_delta_em > 0.0
                is_positive_f1 = oracle_delta_f1 > 0.0
                if is_positive_em:
                    oracle_positive_em += 1
                if is_positive_f1:
                    oracle_positive_f1 += 1
                if verdict == "helpful":
                    helpful_total_with_oracle += 1
                    helpful_oracle_delta_em.append(oracle_delta_em)
                    helpful_oracle_delta_f1.append(oracle_delta_f1)
                    if is_positive_em:
                        helpful_on_positive_em += 1
                    if is_positive_f1:
                        helpful_on_positive_f1 += 1
                candidate_ce_score = float(row.get("candidate_ce_score", 0.0) or 0.0)
                replace_ce_score = float(row.get("replace_incumbent_ce_score", 0.0) or 0.0)
                if candidate_ce_score > replace_ce_score and not is_positive_em and not is_positive_f1:
                    high_ce_nonpositive += 1
                    if verdict in {"neutral", "harmful"}:
                        judge_blocks_high_ce_nonpositive += 1

    per_query_best_helpful: list[dict[str, Any]] = []
    queries_with_helpful = 0
    queries_with_high_conf_helpful = 0
    for question_key, rows in grouped.items():
        parsed_rows = [row for row in rows if bool((row.get("judge_parsed") or {}).get("parse_succeeded"))]
        if not parsed_rows:
            continue
        helpful_rows = [
            row for row in parsed_rows
            if str((row.get("judge_parsed") or {}).get("verdict", "")).lower() == "helpful"
        ]
        if helpful_rows:
            queries_with_helpful += 1
            helpful_rows.sort(
                key=lambda item: (
                    float((item.get("judge_parsed") or {}).get("confidence", 0.0) or 0.0),
                    -float(item.get("candidate_ce_rank", 10_000) or 10_000),
                ),
                reverse=True,
            )
            best_row = helpful_rows[0]
            confidence = float((best_row.get("judge_parsed") or {}).get("confidence", 0.0) or 0.0)
            if confidence >= float(confidence_threshold):
                queries_with_high_conf_helpful += 1
            per_query_best_helpful.append({
                "question": question_key,
                "candidate_title": best_row.get("candidate_title"),
                "replace_incumbent_title": best_row.get("replace_incumbent_title"),
                "replace_incumbent_rank": best_row.get("replace_incumbent_rank"),
                "candidate_ce_rank": best_row.get("candidate_ce_rank"),
                "confidence": _round(confidence, 2),
                "missing_info_type": (best_row.get("judge_parsed") or {}).get("missing_info_type"),
                "reason": (best_row.get("judge_parsed") or {}).get("reason"),
            })

    per_query_best_helpful.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0) or 0.0),
            -float(item.get("candidate_ce_rank", 10_000) or 10_000),
        ),
        reverse=True,
    )

    summary = {
        "metadata": {
            "dataset": dataset,
            "baseline_report": str(baseline_report_path),
            "candidate_report": str(candidate_report_path),
            "output_json": str(output_json),
            "replace_bottom_n": int(replace_bottom_n),
            "judge_backend": judge_bundle.backend,
            "judge_model": judge_bundle.model_name,
            "judge_base_url": judge_bundle.base_url,
            "judge_reasoning_effort": judge_bundle.reasoning_effort,
            "oracle_swap_json": None if oracle_payload is None else oracle_payload.get("metadata", {}).get("output_json"),
            "confidence_threshold": float(confidence_threshold),
        },
        "generation": dict(generation_counters),
        "judge": {
            "count": len(judge_results),
            "parse_failure_count": int(parse_failure_count),
            "parse_failure_rate": _safe_pct(parse_failure_count, len(judge_results)),
            "verdict_counts": dict(verdict_counter),
            "verdict_rates": {
                verdict: _safe_pct(int(verdict_counter.get(verdict, 0)), len(judge_results))
                for verdict in ("helpful", "neutral", "harmful")
            },
            "confidence_mean_by_verdict": {
                verdict: _mean(values)
                for verdict, values in confidence_by_verdict.items()
            },
            "confidence_median_by_verdict": {
                verdict: _median(values)
                for verdict, values in confidence_by_verdict.items()
            },
            "missing_info_type_counts": dict(missing_info_counter),
        },
        "query_gate": {
            "num_queries": len(grouped),
            "queries_with_helpful": int(queries_with_helpful),
            "queries_with_helpful_rate": _safe_pct(queries_with_helpful, len(grouped)),
            "queries_with_high_conf_helpful": int(queries_with_high_conf_helpful),
            "queries_with_high_conf_helpful_rate": _safe_pct(queries_with_high_conf_helpful, len(grouped)),
            "top_helpful_queries": per_query_best_helpful[:10],
        },
        "oracle_alignment": {
            "matched_jobs": int(oracle_matched_jobs),
            "oracle_positive_em_jobs": int(oracle_positive_em),
            "oracle_positive_f1_jobs": int(oracle_positive_f1),
            "helpful_total_with_oracle": int(helpful_total_with_oracle),
            "helpful_precision_em": _safe_pct(helpful_on_positive_em, helpful_total_with_oracle),
            "helpful_precision_f1": _safe_pct(helpful_on_positive_f1, helpful_total_with_oracle),
            "helpful_recall_em": _safe_pct(helpful_on_positive_em, oracle_positive_em),
            "helpful_recall_f1": _safe_pct(helpful_on_positive_f1, oracle_positive_f1),
            "helpful_mean_oracle_delta_em": _mean(helpful_oracle_delta_em),
            "helpful_mean_oracle_delta_f1": _mean(helpful_oracle_delta_f1),
            "high_ce_nonpositive_jobs": int(high_ce_nonpositive),
            "judge_blocks_high_ce_nonpositive": int(judge_blocks_high_ce_nonpositive),
            "judge_blocks_high_ce_nonpositive_rate": _safe_pct(judge_blocks_high_ce_nonpositive, high_ce_nonpositive),
        } if oracle_payload else {},
        "judge_results": judge_results,
    }
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    metadata = dict(summary.get("metadata") or {})
    generation = dict(summary.get("generation") or {})
    judge = dict(summary.get("judge") or {})
    query_gate = dict(summary.get("query_gate") or {})
    oracle_alignment = dict(summary.get("oracle_alignment") or {})
    top_helpful_queries = list(query_gate.get("top_helpful_queries") or [])

    lines = [
        f"# Swap-Utility Judge ({metadata.get('dataset', 'unknown')})",
        "",
        f"- baseline report: `{metadata.get('baseline_report', '')}`",
        f"- candidate report: `{metadata.get('candidate_report', '')}`",
        f"- judge: `{metadata.get('judge_model', '')}` via `{metadata.get('judge_backend', '')}`",
        f"- judge base_url: `{metadata.get('judge_base_url', '')}`",
        f"- judge reasoning_effort: `{metadata.get('judge_reasoning_effort', '')}`",
        f"- replace_bottom_n: `{metadata.get('replace_bottom_n', '')}`",
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
        "## Judge Summary",
        "",
        f"- parse failure count / rate: `{judge.get('parse_failure_count', 0)}` / `{judge.get('parse_failure_rate', '—')}`",
        f"- helpful / neutral / harmful: `{judge.get('verdict_counts', {}).get('helpful', 0)}` / `{judge.get('verdict_counts', {}).get('neutral', 0)}` / `{judge.get('verdict_counts', {}).get('harmful', 0)}`",
        f"- helpful / neutral / harmful rates: `{judge.get('verdict_rates', {}).get('helpful', '—')}` / `{judge.get('verdict_rates', {}).get('neutral', '—')}` / `{judge.get('verdict_rates', {}).get('harmful', '—')}`",
        "",
        "## Query Gate",
        "",
        f"- queries with helpful swap: `{query_gate.get('queries_with_helpful', 0)}` / `{query_gate.get('num_queries', 0)}`",
        f"- queries with high-confidence helpful swap: `{query_gate.get('queries_with_high_conf_helpful', 0)}` / `{query_gate.get('num_queries', 0)}`",
    ])

    if oracle_alignment:
        lines.extend([
            "",
            "## Oracle Alignment",
            "",
            f"- matched jobs: `{oracle_alignment.get('matched_jobs', 0)}`",
            f"- helpful precision EM / F1: `{oracle_alignment.get('helpful_precision_em', '—')}` / `{oracle_alignment.get('helpful_precision_f1', '—')}`",
            f"- helpful recall EM / F1: `{oracle_alignment.get('helpful_recall_em', '—')}` / `{oracle_alignment.get('helpful_recall_f1', '—')}`",
            f"- helpful mean oracle delta EM / F1: `{oracle_alignment.get('helpful_mean_oracle_delta_em', '—')}` / `{oracle_alignment.get('helpful_mean_oracle_delta_f1', '—')}`",
            f"- judge blocks high-CE nonpositive swaps: `{oracle_alignment.get('judge_blocks_high_ce_nonpositive', 0)}` / `{oracle_alignment.get('high_ce_nonpositive_jobs', 0)}`",
        ])

    lines.extend([
        "",
        "## Top Helpful Swaps",
        "",
    ])
    if not top_helpful_queries:
        lines.append("- none")
    else:
        for row in top_helpful_queries:
            lines.extend([
                f"- question: `{row.get('question', '')}`",
                f"  candidate: `{row.get('candidate_title', '')}` replace `{row.get('replace_incumbent_title', '')}` (slot `{row.get('replace_incumbent_rank', '')}`, CE rank `{row.get('candidate_ce_rank', '')}`)",
                f"  confidence / type: `{row.get('confidence', '—')}` / `{row.get('missing_info_type', '—')}`",
                f"  reason: `{row.get('reason', '')}`",
            ])
    return "\n".join(lines) + "\n"


def filter_jobs_by_max_queries(jobs: list[dict[str, Any]], *, max_queries: int) -> list[dict[str, Any]]:
    if int(max_queries or 0) <= 0:
        return list(jobs)
    kept_jobs: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for job in jobs:
        question_key = _normalize_question(str(job.get("question", "")))
        if question_key not in seen_questions:
            if len(seen_questions) >= int(max_queries):
                continue
            seen_questions.add(question_key)
        kept_jobs.append(job)
    return kept_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline swap-utility judge for candidate x bottom-n incumbent swaps.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline_report", required=True)
    parser.add_argument("--candidate_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--oracle_swap_json", default="")
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--replace_bottom_n", type=int, default=2)
    parser.add_argument("--max_queries", type=int, default=100)
    parser.add_argument("--max_jobs", type=int, default=0)
    parser.add_argument("--max_doc_chars", type=int, default=320)
    parser.add_argument("--max_completion_tokens", type=int, default=192)
    parser.add_argument("--confidence_threshold", type=float, default=70.0)
    parser.add_argument("--setwise_late_rerank_judge_backend", choices=["inherit", "responses", "chat_completions"], default="responses")
    parser.add_argument("--setwise_late_rerank_judge_model", type=str, default="gpt-5.4")
    parser.add_argument("--setwise_late_rerank_judge_base_url", type=str, default="")
    parser.add_argument("--setwise_late_rerank_judge_api_key", type=str, default="")
    parser.add_argument("--setwise_late_rerank_judge_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--setwise_late_rerank_judge_reasoning_effort", type=str, default="medium")
    parser.add_argument("--setwise_late_rerank_judge_timeout_s", type=float, default=120.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dataset = str(args.dataset)
    corpus_path, _ = resolve_dataset_paths(dataset)
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    baseline_report = _load_json(args.baseline_report)
    candidate_report = _load_json(args.candidate_report)
    oracle_payload = _load_json(args.oracle_swap_json) if str(args.oracle_swap_json).strip() else None

    jobs, generation_counters = build_swap_jobs(
        dataset=dataset,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        corpus=corpus,
        qa_top_k=int(args.qa_top_k),
        replace_bottom_n=int(args.replace_bottom_n),
    )
    jobs = filter_jobs_by_max_queries(jobs, max_queries=int(args.max_queries))
    if int(args.max_jobs or 0) > 0:
        jobs = jobs[: int(args.max_jobs)]
    generation_counters["total_swap_jobs"] = len(jobs)
    generation_counters["replace_bottom_n"] = int(args.replace_bottom_n)
    generation_counters["max_queries"] = int(args.max_queries)

    if not jobs:
        raise ValueError("No swap jobs generated. Check report alignment and appended candidates.")

    fallback_base_url = str(args.setwise_late_rerank_judge_base_url or "").strip() or None
    judge_bundle = build_setwise_late_rerank_judge_bundle(
        args=args,
        fallback_model_name=str(args.setwise_late_rerank_judge_model or "gpt-5.4"),
        fallback_base_url=fallback_base_url,
    )
    if judge_bundle.infer_fn is None:
        raise ValueError("Swap-utility judge requires an explicit judge backend; got inherit with no infer function.")

    judge_results = run_swap_utility_judge(
        jobs,
        judge_bundle=judge_bundle,
        qa_top_k=int(args.qa_top_k),
        max_doc_chars=int(args.max_doc_chars),
        max_completion_tokens=int(args.max_completion_tokens),
    )
    summary = summarize_swap_utility_results(
        dataset=dataset,
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
        output_json=args.output_json,
        judge_bundle=judge_bundle,
        generation_counters=generation_counters,
        judge_results=judge_results,
        replace_bottom_n=int(args.replace_bottom_n),
        oracle_payload=oracle_payload,
        confidence_threshold=float(args.confidence_threshold),
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
        "total_swap_jobs": len(jobs),
        "judge_model": judge_bundle.model_name,
        "judge_backend": judge_bundle.backend,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
