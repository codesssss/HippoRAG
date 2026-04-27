#!/usr/bin/env python3
"""DAEC-ALR Step-1: consistency-gated single-edit probe.

This script implements the frozen pre-flight plan in
research_memory/emnlp_expand_then_compose/30_daec_alr_step1_plan.md.
It only edits within the existing PropRAG top100 pool and accepts at most one
replacement per query.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from run_daec_consistency_probe import (  # noqa: E402
    call_openai_compatible_reader,
    consistency_features,
    format_llm_reader_messages,
    gold_answers,
    load_json,
    mock_prediction,
    norm_text,
    score_original,
    selected_pool_positions,
    split_title_text,
    support_metrics,
)
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import exact_match, token_f1  # noqa: E402


ROUND0_VARIANTS = ("original", "swap01", "reverse", "rotate_left", "drop_last")
ADMISSION_VARIANTS = ("original", "swap01", "reverse", "rotate_left")
VALIDATION_VARIANTS = (
    "original",
    "drop_last",
    "drop_first",
    "drop_weakest_selected",
    "drop_candidate",
    "swap01_after_drop_last",
)
STEP1_VARIANTS = (
    "no_edit_daec_only",
    "binding_gate_only",
    "consistency_gate_only",
    "double_gate_no_skip",
    "double_gate_skip",
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
DATE_WORD_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)


def tokenize(value: Any) -> set[str]:
    return set(TOKEN_RE.findall(norm_text(str(value or ""))))


def normalized_phrase(value: Any) -> str:
    return norm_text(str(value or ""))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def stable_key(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class ReaderCache:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.rows: dict[str, str] = {}
        if self.path and self.path.exists():
            for row in read_jsonl(self.path):
                key = str(row.get("key") or "")
                if key:
                    self.rows[key] = str(row.get("prediction") or "")

    def get(self, key: str) -> str | None:
        return self.rows.get(key)

    def set_many(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        for row in rows:
            self.rows[str(row["key"])] = str(row.get("prediction") or "")
        if self.path:
            append_jsonl(rows, self.path)


def doc_identity(doc: dict[str, Any]) -> Any:
    return doc.get("doc_id") if doc.get("doc_id") is not None else doc.get("pool_position")


def doc_title(doc: dict[str, Any]) -> str:
    return str(doc.get("title") or "")


def doc_text(doc: dict[str, Any]) -> str:
    return str(doc.get("text") or "")


def make_pool_doc(
    *,
    pool_record: dict[str, Any],
    position: int,
    source: str,
    gold_title_set: set[str],
) -> dict[str, Any]:
    pool_docs = list(pool_record.get("pool_docs") or [])
    pool_titles = list(pool_record.get("pool_titles") or [])
    pool_ids = list(pool_record.get("pool_doc_ids") or [])
    pool_scores = list(pool_record.get("pool_doc_scores") or [])
    title, _body = split_title_text(pool_docs[position], pool_titles[position] if position < len(pool_titles) else "")
    if position < len(pool_titles) and pool_titles[position]:
        title = str(pool_titles[position])
    return {
        "doc_id": pool_ids[position] if position < len(pool_ids) else position,
        "pool_position": int(position),
        "rank": int(position) + 1,
        "title": title,
        "text": pool_docs[position],
        "source": source,
        "retriever_score": pool_scores[position] if position < len(pool_scores) else 0.0,
        "gold_support": int(norm_text(title) in gold_title_set),
    }


def compute_selected_utilities(trace: dict[str, Any], selected_positions: Sequence[int]) -> dict[int, float]:
    selector_trace = trace.get("selector_trace") or {}
    utilities = {int(pos): 0.0 for pos in selected_positions}
    for step in selector_trace.get("selection_steps") or []:
        pos = step.get("pool_position")
        if pos is None:
            continue
        pos = int(pos)
        if pos not in utilities:
            continue
        utilities[pos] = max(utilities[pos], safe_float(step.get("coverage_score")))
    coverage = selector_trace.get("coverage_by_requirement") or {}
    cover_positions = selector_trace.get("cover_position_by_requirement") or {}
    for req_id, pos in cover_positions.items():
        try:
            pos_int = int(pos)
        except (TypeError, ValueError):
            continue
        if pos_int in utilities:
            utilities[pos_int] = max(utilities[pos_int], safe_float(coverage.get(req_id)))
    return {pos: round(float(value), 6) for pos, value in utilities.items()}


def protected_positions(trace: dict[str, Any], selected_positions: Sequence[int]) -> set[int]:
    selector_trace = trace.get("selector_trace") or {}
    selected = {int(pos) for pos in selected_positions}
    match_threshold = safe_float(selector_trace.get("match_threshold"), 0.35)
    coverage = selector_trace.get("coverage_by_requirement") or {}
    protected: set[int] = set()
    for req_id, pos in (selector_trace.get("cover_position_by_requirement") or {}).items():
        try:
            pos_int = int(pos)
        except (TypeError, ValueError):
            continue
        if pos_int in selected and safe_float(coverage.get(req_id)) >= match_threshold:
            protected.add(pos_int)
    return protected


def build_step1_records(
    *,
    report_json: str | Path,
    pool_json: str | Path,
    limit: int,
    offset: int,
    top_k: int,
    source: str,
) -> list[dict[str, Any]]:
    report = load_json(report_json)
    pool = load_json(pool_json)
    traces = list(report.get("setwise_selector_query_traces") or [])
    pool_records = list(pool.get("records") or [])
    if not traces:
        raise ValueError(f"No setwise_selector_query_traces found in {report_json}")
    if not pool_records:
        raise ValueError(f"No records found in {pool_json}")

    end = len(traces) if int(limit) <= 0 else min(len(traces), int(offset) + int(limit))
    records: list[dict[str, Any]] = []
    for row_idx in range(int(offset), end):
        trace = traces[row_idx]
        pool_record = pool_records[row_idx]
        trace_question = str(trace.get("question") or "").strip()
        pool_question = str(pool_record.get("question") or "").strip()
        if trace_question != pool_question:
            raise ValueError(
                f"Question mismatch at row {row_idx}: trace={trace_question!r} pool={pool_question!r}"
            )
        gold_titles = [str(title) for title in pool_record.get("gold_titles") or trace.get("gold_titles") or []]
        gold_title_set = {norm_text(title) for title in gold_titles if norm_text(title)}
        pool_docs = list(pool_record.get("pool_docs") or [])
        selected_positions = selected_pool_positions(trace, top_k=int(top_k))
        selected_docs = [
            make_pool_doc(
                pool_record=pool_record,
                position=int(pos),
                source=source,
                gold_title_set=gold_title_set,
            )
            for pos in selected_positions
            if 0 <= int(pos) < len(pool_docs)
        ]
        pool_doc_rows = [
            make_pool_doc(
                pool_record=pool_record,
                position=pos,
                source=source,
                gold_title_set=gold_title_set,
            )
            for pos in range(len(pool_docs))
        ]
        selected_titles = [doc_title(doc) for doc in selected_docs]
        answers = pool_record.get("gold_answers") or trace.get("gold_answers") or []
        metrics = support_metrics(selected_titles, gold_titles)
        selected_utilities = compute_selected_utilities(trace, selected_positions)
        records.append(
            {
                "qid": str(pool_record.get("qid") or pool_record.get("query_idx") or row_idx),
                "query_idx": int(pool_record.get("query_idx", row_idx)),
                "source": source,
                "question": pool_question,
                "answer": [str(answer) for answer in answers],
                "type": str(trace.get("query_type") or pool_record.get("type") or ""),
                "gold_titles": gold_titles,
                "selected_docs": selected_docs,
                "pool_docs": pool_doc_rows,
                "support_recall": metrics["support_recall"],
                "support_complete": metrics["support_complete"],
                "selected_pool_positions": [int(pos) for pos in selected_positions],
                "selected_doc_utilities": selected_utilities,
                "protected_positions": sorted(protected_positions(trace, selected_positions)),
                "selector_trace": trace.get("selector_trace") or {},
                "selector_answer": str(trace.get("selector_answer") or ""),
                "selector_f1_from_report": safe_float((trace.get("selector_metrics") or {}).get("F1")),
            }
        )
    return records


def perturb_docs(
    docs: Sequence[dict[str, Any]],
    variant: str,
    *,
    weakest_index: int | None = None,
    candidate_index: int | None = None,
) -> list[dict[str, Any]]:
    values = [dict(doc) for doc in docs]
    if variant == "original":
        return values
    if variant == "swap01":
        if len(values) >= 2:
            values[0], values[1] = values[1], values[0]
        return values
    if variant == "reverse":
        return list(reversed(values))
    if variant == "rotate_left":
        return values[1:] + values[:1] if values else values
    if variant == "drop_last":
        return values[:-1] if len(values) > 1 else values
    if variant == "drop_first":
        return values[1:] if len(values) > 1 else values
    if variant == "drop_weakest_selected":
        idx = int(weakest_index) if weakest_index is not None else max(0, len(values) - 1)
        return [doc for pos, doc in enumerate(values) if pos != idx] if len(values) > 1 else values
    if variant == "drop_candidate":
        idx = int(candidate_index) if candidate_index is not None else max(0, len(values) - 1)
        return [doc for pos, doc in enumerate(values) if pos != idx] if len(values) > 1 else values
    if variant == "swap01_after_drop_last":
        dropped = values[:-1] if len(values) > 1 else values
        if len(dropped) >= 2:
            dropped[0], dropped[1] = dropped[1], dropped[0]
        return dropped
    raise ValueError(f"Unknown perturbation variant: {variant}")


def make_reader_record(
    base: dict[str, Any],
    *,
    selected_docs: Sequence[dict[str, Any]],
    family: str,
    variant: str,
    step1_variant: str,
    context_id: str,
    weakest_index: int | None = None,
    candidate_index: int | None = None,
) -> dict[str, Any]:
    perturbed_docs = perturb_docs(
        selected_docs,
        variant,
        weakest_index=weakest_index,
        candidate_index=candidate_index,
    )
    item = {
        "qid": base.get("qid"),
        "query_idx": base.get("query_idx"),
        "question": base.get("question"),
        "answer": list(base.get("answer") or []),
        "type": base.get("type"),
        "gold_titles": list(base.get("gold_titles") or []),
        "selected_docs": perturbed_docs,
        "family": family,
        "variant": variant,
        "step1_variant": step1_variant,
        "context_id": context_id,
    }
    item.update(support_metrics([doc_title(doc) for doc in selected_docs], base.get("gold_titles") or []))
    return item


def cache_key_for_record(record: dict[str, Any], *, dataset: str, prompt_id: str, model: str) -> str:
    doc_ids = [doc_identity(doc) for doc in record.get("selected_docs") or []]
    return stable_key(
        {
            "dataset": dataset,
            "qid": record.get("qid"),
            "selected_doc_ids": doc_ids,
            "family": record.get("family"),
            "variant": record.get("variant"),
            "prompt_id": prompt_id,
            "model": model,
            "max_new_tokens": record.get("max_new_tokens"),
            "max_docs": record.get("max_docs"),
            "max_doc_chars": record.get("max_doc_chars"),
        }
    )


def predict_records(
    records: Sequence[dict[str, Any]],
    *,
    dataset: str,
    prompt_id: str,
    model: str,
    cache: ReaderCache,
    mock_mode: str,
    llm_base_url: str,
    llm_timeout: int,
    llm_retries: int,
    max_new_tokens: int,
    max_docs: int,
    max_doc_chars: int,
    concurrency: int,
) -> list[str]:
    predictions: list[str | None] = [None] * len(records)
    pending: list[tuple[int, str, dict[str, Any]]] = []
    for idx, record in enumerate(records):
        cache_record = dict(record)
        cache_record["max_new_tokens"] = int(max_new_tokens)
        cache_record["max_docs"] = int(max_docs)
        cache_record["max_doc_chars"] = int(max_doc_chars)
        key = cache_key_for_record(cache_record, dataset=dataset, prompt_id=prompt_id, model=model)
        cached = cache.get(key)
        if cached is not None:
            predictions[idx] = cached
        else:
            pending.append((idx, key, dict(record)))

    new_rows: list[dict[str, Any]] = []
    if mock_mode != "none":
        for idx, key, record in pending:
            prediction = mock_prediction(record, mock_mode)
            predictions[idx] = prediction
            new_rows.append({"key": key, "prediction": prediction})
    elif pending:
        workers = max(1, int(concurrency))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    call_openai_compatible_reader,
                    record,
                    base_url=llm_base_url,
                    model=model,
                    max_new_tokens=max_new_tokens,
                    max_docs=max_docs,
                    max_doc_chars=max_doc_chars,
                    timeout=llm_timeout,
                    retries=llm_retries,
                ): (idx, key)
                for idx, key, record in pending
            }
            for future in as_completed(futures):
                idx, key = futures[future]
                prediction = future.result()
                predictions[idx] = prediction
                new_rows.append({"key": key, "prediction": prediction})
    cache.set_many(new_rows)
    return [str(prediction or "") for prediction in predictions]


def majority_answer(predictions: Sequence[str]) -> str:
    if not predictions:
        return ""
    normalized = [norm_text(prediction) for prediction in predictions]
    counts = Counter(normalized)
    if not counts:
        return ""
    best_norm, _count = counts.most_common(1)[0]
    for raw, normed in zip(predictions, normalized):
        if normed == best_norm:
            return str(raw or "").strip()
    return best_norm


def phrase_in_text(phrase: str, text: str) -> bool:
    norm_phrase = normalized_phrase(phrase)
    if not norm_phrase:
        return False
    return norm_phrase in normalized_phrase(text)


def overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(max(1, len(left | right)))


def expected_type_hit(expected_type: str, text: str) -> float:
    lowered = str(expected_type or "").lower()
    raw = str(text or "")
    if "date" in lowered or "time" in lowered or "year" in lowered:
        return float(bool(YEAR_RE.search(raw) or DATE_WORD_RE.search(raw)))
    if "number" in lowered or "count" in lowered or "quantity" in lowered:
        return float(bool(re.search(r"\d", raw)))
    if "person" in lowered:
        return float(bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", raw)))
    if "place" in lowered or "location" in lowered:
        return float(bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", raw)))
    return 0.0


def question_anchor_tokens(record: dict[str, Any]) -> set[str]:
    selector_trace = record.get("selector_trace") or {}
    values: list[str] = [str(record.get("question") or "")]
    for key in (
        "grounded_question_entities_preview",
        "question_entities_preview",
        "query_entities_preview",
        "seed_entities_preview",
    ):
        values.extend(str(item) for item in selector_trace.get(key) or [])
    for req in selector_trace.get("requirements") or []:
        values.extend(str(item) for item in req.get("anchor_mentions") or [])
    return set().union(*(tokenize(value) for value in values))


def selected_context_tokens(record: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for doc in record.get("selected_docs") or []:
        tokens.update(tokenize(doc_title(doc)))
        tokens.update(tokenize(doc_text(doc)) if len(tokens) < 2000 else set())
    return tokens


def candidate_overlap_ok(record: dict[str, Any], candidate: dict[str, Any]) -> bool:
    candidate_tokens = tokenize(doc_title(candidate)) | tokenize(doc_text(candidate))
    if not candidate_tokens:
        return False
    anchors = question_anchor_tokens(record)
    if candidate_tokens & anchors:
        return True
    context_tokens = selected_context_tokens(record)
    return bool(candidate_tokens & context_tokens)


def requirement_score_for_candidate(
    record: dict[str, Any],
    candidate: dict[str, Any],
    requirement: dict[str, Any],
) -> dict[str, Any]:
    title = doc_title(candidate)
    text = doc_text(candidate)
    joined = f"{title}\n{text}"
    subquery_tokens = tokenize(requirement.get("subquery"))
    doc_tokens = tokenize(joined)
    anchor_mentions = [str(item) for item in requirement.get("anchor_mentions") or []]
    anchor_hit = float(any(phrase_in_text(anchor, joined) for anchor in anchor_mentions))
    anchor_overlap = max((overlap_ratio(tokenize(anchor), doc_tokens) for anchor in anchor_mentions), default=0.0)
    binding_rows = (record.get("selector_trace") or {}).get("binding_candidates_by_requirement") or {}
    binding_candidates = list(binding_rows.get(str(requirement.get("unit_id"))) or [])
    binding_hit = float(any(phrase_in_text(str(row.get("title") or ""), joined) for row in binding_candidates))
    subquery_overlap = overlap_ratio(subquery_tokens, doc_tokens)
    type_hit = expected_type_hit(str(requirement.get("expected_answer_type") or ""), joined)
    raw_score = max(
        1.0 if binding_hit > 0.0 else 0.0,
        0.85 if anchor_hit > 0.0 else 0.0,
        min(0.70, 1.20 * subquery_overlap),
        min(0.45, 0.25 * anchor_overlap + 0.20 * type_hit),
    )
    return {
        "unit_id": str(requirement.get("unit_id") or ""),
        "score": round(float(raw_score), 6),
        "anchor_hit": round(float(anchor_hit), 6),
        "anchor_overlap": round(float(anchor_overlap), 6),
        "binding_hit": round(float(binding_hit), 6),
        "subquery_overlap": round(float(subquery_overlap), 6),
        "expected_type_hit": round(float(type_hit), 6),
    }


def candidate_binding_proxy(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    requirements = list((record.get("selector_trace") or {}).get("requirements") or [])
    requirement_scores = [
        requirement_score_for_candidate(record, candidate, requirement)
        for requirement in requirements
    ]
    best_score = max((safe_float(row.get("score")) for row in requirement_scores), default=0.0)
    rank_prior = 1.0 / float(max(1, int(candidate.get("rank") or 1)))
    utility = min(1.0, best_score + 0.05 * rank_prior)
    return {
        "candidate_pool_position": int(candidate.get("pool_position", -1)),
        "candidate_title": doc_title(candidate),
        "utility": round(float(utility), 6),
        "best_requirement_score": round(float(best_score), 6),
        "rank_prior": round(float(rank_prior), 6),
        "requirement_scores": requirement_scores,
    }


def choose_replacement_target(record: dict[str, Any], *, protect_demands: bool) -> dict[str, Any] | None:
    docs = list(record.get("selected_docs") or [])
    utilities = {int(k): safe_float(v) for k, v in (record.get("selected_doc_utilities") or {}).items()}
    protected = {int(pos) for pos in record.get("protected_positions") or []} if protect_demands else set()
    candidates: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        pos = int(doc.get("pool_position", idx))
        if pos in protected:
            continue
        candidates.append(
            {
                "selected_index": int(idx),
                "pool_position": int(pos),
                "title": doc_title(doc),
                "utility": round(float(utilities.get(pos, 0.0)), 6),
                "protected": bool(pos in protected),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda row: (safe_float(row.get("utility")), int(row.get("selected_index"))))
    return candidates[0]


def build_candidate_plans(
    record: dict[str, Any],
    *,
    variant_name: str,
    binding_required: bool,
    allow_protected_replacement: bool,
    k_edit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_positions = {int(pos) for pos in record.get("selected_pool_positions") or []}
    selected_title_keys = {norm_text(doc_title(doc)) for doc in record.get("selected_docs") or []}
    target = choose_replacement_target(record, protect_demands=not allow_protected_replacement)
    if target is None:
        return [], {"candidate_count": 0, "reason": "no_non_destructive_target"}

    raw_candidates: list[dict[str, Any]] = []
    duplicate_rejected = 0
    overlap_rejected = 0
    for candidate in record.get("pool_docs") or []:
        pos = int(candidate.get("pool_position", -1))
        if pos in selected_positions:
            continue
        title_key = norm_text(doc_title(candidate))
        if title_key and title_key in selected_title_keys:
            duplicate_rejected += 1
            continue
        if not candidate_overlap_ok(record, candidate):
            overlap_rejected += 1
            continue
        proxy = candidate_binding_proxy(record, candidate)
        target_utility = safe_float(target.get("utility"))
        improvement = safe_float(proxy.get("utility")) - target_utility
        binding_pass = improvement > 1e-9
        if binding_required and not binding_pass:
            continue
        edited_docs = [dict(doc) for doc in record.get("selected_docs") or []]
        replacement_index = int(target["selected_index"])
        edited_docs[replacement_index] = dict(candidate)
        raw_candidates.append(
            {
                "candidate_key": f"{record.get('qid')}::{variant_name}::{pos}",
                "qid": record.get("qid"),
                "variant": variant_name,
                "candidate": candidate,
                "target": target,
                "replacement_index": replacement_index,
                "candidate_index": replacement_index,
                "weakest_index": int(target["selected_index"]),
                "edited_docs": edited_docs,
                "binding_proxy": proxy,
                "binding_improvement": round(float(improvement), 6),
                "binding_pass": bool(binding_pass),
            }
        )
    raw_candidates.sort(
        key=lambda row: (
            -safe_float(row.get("binding_improvement")),
            -safe_float((row.get("binding_proxy") or {}).get("utility")),
            int((row.get("candidate") or {}).get("pool_position", 999999)),
        )
    )
    kept = raw_candidates[: max(0, int(k_edit))]
    return kept, {
        "candidate_count": len(kept),
        "raw_candidate_count": len(raw_candidates),
        "duplicate_rejected": duplicate_rejected,
        "overlap_rejected": overlap_rejected,
        "target": target,
        "reason": "ok" if kept else "no_candidate_after_prefilter",
    }


def group_predictions_by_context(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    variants: Sequence[str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for record, prediction in zip(records, predictions):
        grouped[str(record.get("context_id"))][str(record.get("variant"))] = str(prediction or "")
    out: dict[str, dict[str, Any]] = {}
    for context_id, by_variant in grouped.items():
        ordered = [by_variant.get(variant, "") for variant in variants]
        features = consistency_features(ordered)
        out[context_id] = {
            "predictions": dict(by_variant),
            "ordered_predictions": ordered,
            "features": features,
            "majority_answer": majority_answer(ordered),
        }
    return out


def bootstrap_delta_ci(deltas: Sequence[float], *, rounds: int, seed: int = 13) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(seed)
    values: list[float] = []
    n = len(deltas)
    for _ in range(max(1, int(rounds))):
        sample = [deltas[rng.randrange(n)] for _idx in range(n)]
        values.append(sum(sample) / float(n))
    values.sort()
    return [
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    ]


def score_support_for_docs(docs: Sequence[dict[str, Any]], gold_titles: Sequence[str]) -> dict[str, float]:
    return support_metrics([doc_title(doc) for doc in docs], gold_titles)


def added_removed_stats(
    base_docs: Sequence[dict[str, Any]],
    final_docs: Sequence[dict[str, Any]],
) -> dict[str, int]:
    base = {doc_identity(doc): doc for doc in base_docs}
    final = {doc_identity(doc): doc for doc in final_docs}
    added = [doc for key, doc in final.items() if key not in base]
    removed = [doc for key, doc in base.items() if key not in final]
    return {
        "added_gold": sum(int(doc.get("gold_support") or 0) for doc in added),
        "added_non_gold": sum(1 - int(doc.get("gold_support") or 0) for doc in added),
        "removed_gold": sum(int(doc.get("gold_support") or 0) for doc in removed),
        "removed_non_gold": sum(1 - int(doc.get("gold_support") or 0) for doc in removed),
    }


def config_for_variant(name: str) -> dict[str, bool]:
    if name == "no_edit_daec_only":
        return {
            "edit_enabled": False,
            "binding_required": False,
            "consistency_required": False,
            "skip_stable": False,
            "allow_protected_replacement": True,
        }
    if name == "binding_gate_only":
        return {
            "edit_enabled": True,
            "binding_required": True,
            "consistency_required": False,
            "skip_stable": False,
            "allow_protected_replacement": False,
        }
    if name == "consistency_gate_only":
        return {
            "edit_enabled": True,
            "binding_required": False,
            "consistency_required": True,
            "skip_stable": False,
            "allow_protected_replacement": True,
        }
    if name == "double_gate_no_skip":
        return {
            "edit_enabled": True,
            "binding_required": True,
            "consistency_required": True,
            "skip_stable": False,
            "allow_protected_replacement": False,
        }
    if name == "double_gate_skip":
        return {
            "edit_enabled": True,
            "binding_required": True,
            "consistency_required": True,
            "skip_stable": True,
            "allow_protected_replacement": False,
        }
    raise ValueError(f"Unknown Step-1 variant: {name}")


def evaluate_variant(
    *,
    variant_name: str,
    records: Sequence[dict[str, Any]],
    round0_by_qid: dict[str, dict[str, Any]],
    baseline_by_qid: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    cache: ReaderCache,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = config_for_variant(variant_name)
    query_traces: dict[str, dict[str, Any]] = {}
    candidate_plans: list[dict[str, Any]] = []

    for record in records:
        qid = str(record.get("qid"))
        base_consistency = round0_by_qid[qid]["features"]
        stable_skip = bool(config["skip_stable"] and safe_float(base_consistency.get("majority_fraction")) >= 1.0)
        trace = {
            "qid": qid,
            "query_idx": record.get("query_idx"),
            "variant": variant_name,
            "question": record.get("question"),
            "eligible_for_edit": bool(config["edit_enabled"] and not stable_skip),
            "skip_reason": "round0_majority_fraction_1.0" if stable_skip else "",
            "round0_consistency": base_consistency,
            "base_prediction": baseline_by_qid[qid]["prediction"],
            "base_f1": baseline_by_qid[qid]["f1"],
            "base_em": baseline_by_qid[qid]["em"],
            "base_support_recall": record.get("support_recall"),
            "base_support_complete": record.get("support_complete"),
            "selected_pool_positions": record.get("selected_pool_positions"),
            "protected_positions": record.get("protected_positions"),
            "candidate_prefilter": {},
            "accepted": False,
            "accepted_plan": None,
            "admitted_to_validation": False,
            "final_docs": list(record.get("selected_docs") or []),
        }
        query_traces[qid] = trace
        if not bool(config["edit_enabled"]) or stable_skip:
            continue
        plans, prefilter = build_candidate_plans(
            record,
            variant_name=variant_name,
            binding_required=bool(config["binding_required"]),
            allow_protected_replacement=bool(config["allow_protected_replacement"]),
            k_edit=int(args.k_edit),
        )
        trace["candidate_prefilter"] = prefilter
        trace["candidate_plan_count"] = len(plans)
        candidate_plans.extend(plans)

    if bool(config["consistency_required"]) and candidate_plans:
        admission_records: list[dict[str, Any]] = []
        for plan in candidate_plans:
            record = next(item for item in records if str(item.get("qid")) == str(plan.get("qid")))
            for perturb in ADMISSION_VARIANTS:
                admission_records.append(
                    make_reader_record(
                        record,
                        selected_docs=plan["edited_docs"],
                        family="admission_order",
                        variant=perturb,
                        step1_variant=variant_name,
                        context_id=str(plan["candidate_key"]),
                        weakest_index=int(plan["weakest_index"]),
                        candidate_index=int(plan["candidate_index"]),
                    )
                )
        admission_predictions = predict_records(
            admission_records,
            dataset=str(args.dataset),
            prompt_id=str(args.prompt_id),
            model=str(args.llm_model or args.mock_mode),
            cache=cache,
            mock_mode=str(args.mock_mode),
            llm_base_url=str(args.llm_base_url),
            llm_timeout=int(args.llm_timeout),
            llm_retries=int(args.llm_retries),
            max_new_tokens=int(args.max_new_tokens),
            max_docs=int(args.max_docs),
            max_doc_chars=int(args.max_doc_chars),
            concurrency=int(args.llm_concurrency),
        )
        admission_by_context = group_predictions_by_context(
            admission_records,
            admission_predictions,
            ADMISSION_VARIANTS,
        )
        by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for plan in candidate_plans:
            qid = str(plan["qid"])
            admission = admission_by_context.get(str(plan["candidate_key"]), {})
            features = admission.get("features") or {}
            base_inverse = safe_float(round0_by_qid[qid]["features"].get("inverse_entropy"))
            delta_inverse = safe_float(features.get("inverse_entropy")) - base_inverse
            plan["admission"] = {
                "features": features,
                "predictions": admission.get("predictions") or {},
                "majority_answer": admission.get("majority_answer") or "",
                "delta_inverse_entropy": round(float(delta_inverse), 6),
                "pass": bool(delta_inverse >= float(args.admission_inverse_entropy_margin)),
            }
            if bool(plan["admission"]["pass"]):
                by_qid[qid].append(plan)

        validation_plans: list[dict[str, Any]] = []
        for qid, plans in by_qid.items():
            plans.sort(
                key=lambda row: (
                    -safe_float((row.get("admission") or {}).get("delta_inverse_entropy")),
                    -safe_float(row.get("binding_improvement")),
                    -safe_float(((row.get("admission") or {}).get("features") or {}).get("majority_fraction")),
                )
            )
            best = plans[0]
            query_traces[qid]["admitted_to_validation"] = True
            query_traces[qid]["best_admission_plan"] = summarize_plan(best)
            validation_plans.append(best)

        if validation_plans:
            validation_records: list[dict[str, Any]] = []
            for plan in validation_plans:
                record = next(item for item in records if str(item.get("qid")) == str(plan.get("qid")))
                for perturb in VALIDATION_VARIANTS:
                    validation_records.append(
                        make_reader_record(
                            record,
                            selected_docs=plan["edited_docs"],
                            family="validation_subset_drop",
                            variant=perturb,
                            step1_variant=variant_name,
                            context_id=str(plan["candidate_key"]),
                            weakest_index=int(plan["weakest_index"]),
                            candidate_index=int(plan["candidate_index"]),
                        )
                    )
            validation_predictions = predict_records(
                validation_records,
                dataset=str(args.dataset),
                prompt_id=str(args.prompt_id),
                model=str(args.llm_model or args.mock_mode),
                cache=cache,
                mock_mode=str(args.mock_mode),
                llm_base_url=str(args.llm_base_url),
                llm_timeout=int(args.llm_timeout),
                llm_retries=int(args.llm_retries),
                max_new_tokens=int(args.max_new_tokens),
                max_docs=int(args.max_docs),
                max_doc_chars=int(args.max_doc_chars),
                concurrency=int(args.llm_concurrency),
            )
            validation_by_context = group_predictions_by_context(
                validation_records,
                validation_predictions,
                VALIDATION_VARIANTS,
            )
            for plan in validation_plans:
                qid = str(plan["qid"])
                validation = validation_by_context.get(str(plan["candidate_key"]), {})
                features = validation.get("features") or {}
                base_inverse = safe_float(round0_by_qid[qid]["features"].get("inverse_entropy"))
                delta_inverse = safe_float(features.get("inverse_entropy")) - base_inverse
                majority = str(validation.get("majority_answer") or "").strip()
                plan["validation"] = {
                    "features": features,
                    "predictions": validation.get("predictions") or {},
                    "majority_answer": majority,
                    "delta_inverse_entropy": round(float(delta_inverse), 6),
                    "pass": bool(delta_inverse >= 0.0 and majority),
                }
                if bool(plan["validation"]["pass"]):
                    query_traces[qid]["accepted"] = True
                    query_traces[qid]["accepted_plan"] = summarize_plan(plan)
                    query_traces[qid]["final_docs"] = plan["edited_docs"]
    elif not bool(config["consistency_required"]):
        by_qid = defaultdict(list)
        for plan in candidate_plans:
            by_qid[str(plan["qid"])].append(plan)
        for qid, plans in by_qid.items():
            plans.sort(
                key=lambda row: (
                    -safe_float(row.get("binding_improvement")),
                    -safe_float((row.get("binding_proxy") or {}).get("utility")),
                )
            )
            best = plans[0]
            query_traces[qid]["accepted"] = True
            query_traces[qid]["accepted_plan"] = summarize_plan(best)
            query_traces[qid]["final_docs"] = best["edited_docs"]

    final_records: list[dict[str, Any]] = []
    for record in records:
        qid = str(record.get("qid"))
        trace = query_traces[qid]
        final_records.append(
            make_reader_record(
                record,
                selected_docs=trace["final_docs"],
                family="final_original",
                variant="original",
                step1_variant=variant_name,
                context_id=f"{qid}::{variant_name}::final",
            )
        )
    final_predictions = predict_records(
        final_records,
        dataset=str(args.dataset),
        prompt_id=str(args.prompt_id),
        model=str(args.llm_model or args.mock_mode),
        cache=cache,
        mock_mode=str(args.mock_mode),
        llm_base_url=str(args.llm_base_url),
        llm_timeout=int(args.llm_timeout),
        llm_retries=int(args.llm_retries),
        max_new_tokens=int(args.max_new_tokens),
        max_docs=int(args.max_docs),
        max_doc_chars=int(args.max_doc_chars),
        concurrency=int(args.llm_concurrency),
    )

    rows: list[dict[str, Any]] = []
    for record, final_record, prediction in zip(records, final_records, final_predictions):
        qid = str(record.get("qid"))
        trace = query_traces[qid]
        support = score_support_for_docs(trace["final_docs"], record.get("gold_titles") or [])
        em = exact_match(gold_answers(record), prediction)
        f1 = token_f1(gold_answers(record), prediction)
        base = baseline_by_qid[qid]
        edit_stats = added_removed_stats(record.get("selected_docs") or [], trace["final_docs"])
        pre_correct = safe_float(base["f1"]) >= 0.5
        post_correct = safe_float(f1) >= 0.5
        row = {
            "qid": qid,
            "query_idx": record.get("query_idx"),
            "variant": variant_name,
            "question": record.get("question"),
            "prediction": prediction,
            "gold_answers": gold_answers(record),
            "em": em,
            "f1": f1,
            "base_em": base["em"],
            "base_f1": base["f1"],
            "f1_delta": round(float(f1) - safe_float(base["f1"]), 8),
            "support_recall": support["support_recall"],
            "support_complete": support["support_complete"],
            "base_support_recall": record.get("support_recall"),
            "base_support_complete": record.get("support_complete"),
            "eligible_for_edit": bool(trace.get("eligible_for_edit")),
            "candidate_count": int((trace.get("candidate_prefilter") or {}).get("candidate_count") or 0),
            "admitted_to_validation": bool(trace.get("admitted_to_validation")),
            "accepted": bool(trace.get("accepted")),
            "wrong_to_correct": int((not pre_correct) and post_correct),
            "correct_to_wrong": int(pre_correct and (not post_correct)),
            "final_titles": [doc_title(doc) for doc in trace["final_docs"]],
            "base_titles": [doc_title(doc) for doc in record.get("selected_docs") or []],
        }
        row.update(edit_stats)
        trace.update(
            {
                "final_prediction": prediction,
                "final_em": em,
                "final_f1": f1,
                "final_support_recall": support["support_recall"],
                "final_support_complete": support["support_complete"],
                "edit_stats": edit_stats,
            }
        )
        rows.append(row)
    return rows, list(query_traces.values())


def summarize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_key": plan.get("candidate_key"),
        "candidate_pool_position": (plan.get("candidate") or {}).get("pool_position"),
        "candidate_title": doc_title(plan.get("candidate") or {}),
        "target": plan.get("target"),
        "binding_improvement": plan.get("binding_improvement"),
        "binding_pass": plan.get("binding_pass"),
        "binding_proxy": plan.get("binding_proxy"),
        "admission": plan.get("admission"),
        "validation": plan.get("validation"),
    }


def summarize_variant(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_rows: Sequence[dict[str, Any]],
    bootstrap_rounds: int,
) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    n = float(len(rows))
    edited = [row for row in rows if bool(row.get("accepted"))]
    eligible = [row for row in rows if bool(row.get("eligible_for_edit"))]
    baseline_by_qid = {str(row.get("qid")): row for row in baseline_rows}
    deltas = [
        safe_float(row.get("f1")) - safe_float(baseline_by_qid.get(str(row.get("qid")), {}).get("f1"))
        for row in rows
    ]
    ci = bootstrap_delta_ci(deltas, rounds=int(bootstrap_rounds))
    added_gold = sum(int(row.get("added_gold") or 0) for row in rows)
    added_non_gold = sum(int(row.get("added_non_gold") or 0) for row in rows)
    ratio = None if added_gold <= 0 else added_non_gold / float(added_gold)
    return {
        "rows": len(rows),
        "answer_em": round(sum(safe_float(row.get("em")) for row in rows) / n, 6),
        "answer_f1": round(sum(safe_float(row.get("f1")) for row in rows) / n, 6),
        "support_recall": round(sum(safe_float(row.get("support_recall")) for row in rows) / n, 6),
        "support_complete": round(sum(safe_float(row.get("support_complete")) for row in rows) / n, 6),
        "mean_f1_delta_vs_no_edit": round(sum(deltas) / n, 8),
        "f1_delta_ci95": [round(float(ci[0]), 8), round(float(ci[1]), 8)],
        "eligible_query_rate": round(len(eligible) / n, 6),
        "candidate_trigger_rate": round(
            sum(1 for row in rows if int(row.get("candidate_count") or 0) > 0) / n,
            6,
        ),
        "admitted_to_validation_rate": round(
            sum(1 for row in rows if bool(row.get("admitted_to_validation"))) / n,
            6,
        ),
        "accepted_edit_rate": round(len(edited) / n, 6),
        "accepted_edit_rate_eligible": round(len(edited) / float(max(1, len(eligible))), 6),
        "edited_subset_pre_f1": round(
            sum(safe_float(row.get("base_f1")) for row in edited) / float(max(1, len(edited))),
            6,
        ),
        "edited_subset_post_f1": round(
            sum(safe_float(row.get("f1")) for row in edited) / float(max(1, len(edited))),
            6,
        ),
        "wrong_to_correct_flips": sum(int(row.get("wrong_to_correct") or 0) for row in rows),
        "correct_to_wrong_flips": sum(int(row.get("correct_to_wrong") or 0) for row in rows),
        "added_gold": added_gold,
        "added_non_gold": added_non_gold,
        "removed_gold": sum(int(row.get("removed_gold") or 0) for row in rows),
        "removed_non_gold": sum(int(row.get("removed_non_gold") or 0) for row in rows),
        "added_non_gold_per_added_gold": None if ratio is None else round(float(ratio), 6),
    }


def main_gate_decision(summary: dict[str, Any], baseline: dict[str, Any]) -> str:
    f1_delta = safe_float(summary.get("answer_f1")) - safe_float(baseline.get("answer_f1"))
    ci_low = safe_float((summary.get("f1_delta_ci95") or [0.0, 0.0])[0])
    accepted_rate = safe_float(summary.get("accepted_edit_rate"))
    edited_gain = safe_float(summary.get("edited_subset_post_f1")) > safe_float(summary.get("edited_subset_pre_f1"))
    flip_ok = int(summary.get("wrong_to_correct_flips") or 0) > int(summary.get("correct_to_wrong_flips") or 0)
    support_ok = safe_float(summary.get("support_complete")) >= safe_float(baseline.get("support_complete"))
    added_gold = int(summary.get("added_gold") or 0)
    ratio = summary.get("added_non_gold_per_added_gold")
    ratio_ok = added_gold > 0 and ratio is not None and safe_float(ratio) <= 8.0
    primary = (
        f1_delta >= 0.004
        and ci_low > 0.0
        and 0.10 <= accepted_rate <= 0.35
        and edited_gain
        and flip_ok
        and support_ok
        and ratio_ok
    )
    if primary:
        return "GREEN"
    destructive = (
        f1_delta < 0.0
        or not support_ok
        or not (0.10 <= accepted_rate <= 0.35)
        or not edited_gain
        or not flip_ok
        or added_gold <= 0
        or (ratio is not None and safe_float(ratio) > 8.0)
    )
    if destructive:
        return "RED"
    return "YELLOW"


def write_markdown_report(output: dict[str, Any], path: str | Path) -> None:
    summary = output["summary"]
    lines = [
        "# DAEC-ALR Step-1 Single-Edit Gate",
        "",
        "## Purpose",
        "",
        "Test the frozen DAEC-ALR Step-1 plan: single-edit repair inside the existing PropRAG top100 pool.",
        "",
        "## Configuration",
        "",
        f"- dataset: `{output['inputs']['dataset']}`",
        f"- DAEC report: `{output['inputs']['report_json']}`",
        f"- pool JSON: `{output['inputs']['pool_json']}`",
        f"- rows: `{output['inputs']['limit']}`",
        f"- variants: `{', '.join(output['inputs']['variants'])}`",
        f"- reader mode: `{output['runtime']['mode']}`",
        f"- model/cache key model: `{output['runtime']['model']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | F1 | ΔF1 | CI95 ΔF1 | Support Complete | Accepted | Edit Subset F1 Pre/Post | W→C | C→W | Added G/NG | NG/G |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in output["inputs"]["variants"]:
        row = summary["variants"][variant]
        ci = row.get("f1_delta_ci95") or [0.0, 0.0]
        ratio = row.get("added_non_gold_per_added_gold")
        ratio_text = "n/a" if ratio is None else f"{safe_float(ratio):.2f}"
        lines.append(
            f"| `{variant}` | {safe_float(row.get('answer_f1')):.4f} | "
            f"{safe_float(row.get('mean_f1_delta_vs_no_edit')):.4f} | "
            f"[{safe_float(ci[0]):.4f}, {safe_float(ci[1]):.4f}] | "
            f"{safe_float(row.get('support_complete')):.4f} | "
            f"{safe_float(row.get('accepted_edit_rate')):.3f} | "
            f"{safe_float(row.get('edited_subset_pre_f1')):.4f}/{safe_float(row.get('edited_subset_post_f1')):.4f} | "
            f"{int(row.get('wrong_to_correct_flips') or 0)} | "
            f"{int(row.get('correct_to_wrong_flips') or 0)} | "
            f"{int(row.get('added_gold') or 0)}/{int(row.get('added_non_gold') or 0)} | "
            f"{ratio_text} |"
        )
    main = summary["variants"].get("double_gate_skip", {})
    lines.extend(
        [
            "",
            "## Main Decision",
            "",
            f"- main variant: `double_gate_skip`",
            f"- decision: `{summary.get('main_decision')}`",
            f"- F1 delta: `{safe_float(main.get('mean_f1_delta_vs_no_edit')):.6f}`",
            f"- accepted edit rate: `{safe_float(main.get('accepted_edit_rate')):.6f}`",
            f"- support_complete: `{safe_float(main.get('support_complete')):.6f}`",
            "",
            "## Notes",
            "",
            "- The binding gate uses a fixed trace-local demand/binding proxy because the exported DAEC report does not contain enough embedding state to recompute exact DAEC candidate scores.",
            "- The proxy protects covered requirement positions, replaces low-utility selected docs, and logs component scores for every candidate.",
            "- No retrieval outside the cached pool is performed.",
            "",
        ]
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--output_dir", default="reports/daec_alr")
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--source", default="daec_alr_step1")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--variants", default=",".join(STEP1_VARIANTS))
    parser.add_argument("--k_edit", type=int, default=5)
    parser.add_argument("--admission_inverse_entropy_margin", type=float, default=0.10)
    parser.add_argument("--reader_cache_jsonl", default="")
    parser.add_argument("--prompt_id", default="hipporag_rag_qa_musique_one_shot")
    parser.add_argument("--llm_base_url", default="")
    parser.add_argument("--llm_model", default="")
    parser.add_argument("--llm_timeout", type=int, default=120)
    parser.add_argument("--llm_retries", type=int, default=1)
    parser.add_argument("--llm_concurrency", type=int, default=8)
    parser.add_argument("--mock_mode", choices=["none", "oracle", "empty", "title"], default="none")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--bootstrap_rounds", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in STEP1_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if args.mock_mode == "none" and not (args.llm_base_url and args.llm_model):
        raise ValueError("Provide --llm_base_url/--llm_model or use --mock_mode for tests/smoke runs.")

    output_dir = Path(args.output_dir)
    cache_path = args.reader_cache_jsonl or str(output_dir / "step1_single_edit_reader_cache.jsonl")
    cache = ReaderCache(cache_path)
    records = build_step1_records(
        report_json=args.report_json,
        pool_json=args.pool_json,
        limit=int(args.limit),
        offset=int(args.offset),
        top_k=int(args.top_k),
        source=str(args.source),
    )

    round0_records: list[dict[str, Any]] = []
    for record in records:
        for perturb in ROUND0_VARIANTS:
            round0_records.append(
                make_reader_record(
                    record,
                    selected_docs=record.get("selected_docs") or [],
                    family="round0_order",
                    variant=perturb,
                    step1_variant="round0",
                    context_id=str(record.get("qid")),
                )
            )
    round0_predictions = predict_records(
        round0_records,
        dataset=str(args.dataset),
        prompt_id=str(args.prompt_id),
        model=str(args.llm_model or args.mock_mode),
        cache=cache,
        mock_mode=str(args.mock_mode),
        llm_base_url=str(args.llm_base_url),
        llm_timeout=int(args.llm_timeout),
        llm_retries=int(args.llm_retries),
        max_new_tokens=int(args.max_new_tokens),
        max_docs=int(args.max_docs),
        max_doc_chars=int(args.max_doc_chars),
        concurrency=int(args.llm_concurrency),
    )
    round0_by_qid = group_predictions_by_context(round0_records, round0_predictions, ROUND0_VARIANTS)
    baseline_by_qid: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record.get("qid"))
        original_prediction = str((round0_by_qid[qid]["predictions"] or {}).get("original") or "")
        score = score_original(record, original_prediction)
        baseline_by_qid[qid] = {
            "prediction": original_prediction,
            "em": safe_float(score.get("original_em")),
            "f1": safe_float(score.get("original_f1")),
        }

    all_rows: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_name in variants:
        rows, traces = evaluate_variant(
            variant_name=variant_name,
            records=records,
            round0_by_qid=round0_by_qid,
            baseline_by_qid=baseline_by_qid,
            args=args,
            cache=cache,
        )
        all_rows.extend(rows)
        all_traces.extend(traces)
        rows_by_variant[variant_name] = rows

    baseline_rows = rows_by_variant.get("no_edit_daec_only") or [
        row for row in all_rows if row.get("variant") == "no_edit_daec_only"
    ]
    summaries = {
        variant: summarize_variant(
            rows_by_variant.get(variant, []),
            baseline_rows=baseline_rows,
            bootstrap_rounds=int(args.bootstrap_rounds),
        )
        for variant in variants
    }
    main_decision = (
        main_gate_decision(summaries["double_gate_skip"], summaries["no_edit_daec_only"])
        if "double_gate_skip" in summaries and "no_edit_daec_only" in summaries
        else "NOT_EVALUATED"
    )

    output = {
        "inputs": {
            "dataset": str(args.dataset),
            "report_json": str(args.report_json),
            "pool_json": str(args.pool_json),
            "limit": int(args.limit),
            "offset": int(args.offset),
            "top_k": int(args.top_k),
            "variants": variants,
            "k_edit": int(args.k_edit),
            "admission_inverse_entropy_margin": float(args.admission_inverse_entropy_margin),
        },
        "runtime": {
            "mode": "mock" if args.mock_mode != "none" else "openai_compatible_chat",
            "model": str(args.llm_model or args.mock_mode),
            "base_url": str(args.llm_base_url),
            "concurrency": int(args.llm_concurrency),
            "reader_cache_jsonl": cache_path,
        },
        "summary": {
            "variants": summaries,
            "main_decision": main_decision,
        },
    }
    write_json(output, output_dir / "step1_single_edit_gate.json")
    write_jsonl(all_rows, output_dir / "step1_single_edit_rows.jsonl")
    write_jsonl(all_traces, output_dir / "step1_single_edit_traces.jsonl")
    write_markdown_report(output, output_dir / "step1_single_edit_gate.md")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
