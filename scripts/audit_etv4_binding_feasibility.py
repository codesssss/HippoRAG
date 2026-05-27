#!/usr/bin/env python3
"""Audit whether ETv3/DBEC residual MuSiQue 4-hop failures are binding-feasible.

This script is deliberately diagnostic-only.  It does not run a reader, does
not call an LLM, and does not modify any ETv3/DBEC method code.  It asks whether
the remaining 4-doc MuSiQue limit100 failures after baseline-stable DBEC look
like:

* state/binding-feasible residual composition failures,
* binding extraction / branch consistency failures,
* pool/candidate-universe misses, or
* reader/context-budget failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ETV3_RETRIEVAL = Path(
    "run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_limit100_20260509/"
    "musique/reports/musique_evidence_transition_graphragv3_variable_flow_retrieval.json"
)
DEFAULT_ETV3_QA = Path(
    "run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_limit100_20260509/"
    "reports/evidence_transition_graphragv3_variable_flow_reader_only_qa.json"
)
DEFAULT_ETV4_READER10_QA = Path(
    "run_logs/evidence_transition_graphragv4_reader_context_qwen8b_nv2_limit100_20260509/"
    "reports/evidence_transition_graphragv4_reader_context_qa.json"
)
DEFAULT_STABLE_DBEC = Path(
    "run_logs/etv3_dbec_latest_limit100_20260510/evals/"
    "musique_etv3_pool100_dbec_latest_minimal_edit_strict_limit100.json"
)
DEFAULT_CONSERVATIVE_DBEC = Path(
    "run_logs/etv3_dbec_latest_limit100_20260510/evals/"
    "musique_etv3_pool100_dbec_latest_safe_llm_limit100.json"
)
DEFAULT_ETV3_POOL = Path(
    "run_logs/etv3_dbec_latest_limit100_20260510/pools/musique_etv3_pool100_limit100.json"
)
DEFAULT_STABLE_DBEC_READER10_INPUT = Path(
    "run_logs/evidence_transition_graphragv4_reader_context_qwen8b_nv2_limit100_20260509/"
    "reports/musique_etv3_dbec_stable_reader_context_k10_input.json"
)
DEFAULT_STABLE_DBEC_READER10_QA = Path(
    "run_logs/evidence_transition_graphragv4_reader_context_qwen8b_nv2_limit100_20260509/"
    "reports/evidence_transition_graphragv4_stable_dbec_reader_context_qa.json"
)
DEFAULT_REPORT_DIR = Path("reports/etv4_binding_feasibility_20260510")

TOP_K = 5
EPS = 1e-9

STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "list",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        output = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(output) or math.isinf(output):
        return default
    return output


def unique_ints(values: Iterable[Any], *, limit: int | None = None) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if limit is not None and len(output) >= int(limit):
            break
    return output


def unique_texts(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_title(value).split()
        if len(token) > 2 and token not in STOP_TOKENS
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(len(a | b))


def first_position(doc_id: int, docs: Sequence[int]) -> int | None:
    for index, item in enumerate(docs):
        if int(item) == int(doc_id):
            return index
    return None


def rank_bucket(position: int | None) -> str:
    if position is None:
        return "missing"
    rank = int(position) + 1
    if rank <= 5:
        return "top5"
    if rank <= 10:
        return "6-10"
    if rank <= 100:
        return "11-100"
    if rank <= 200:
        return "101-200"
    return ">200"


def method_payload(qa_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    datasets = qa_payload.get("datasets", []) if isinstance(qa_payload, Mapping) else []
    if not datasets:
        return {}
    methods = datasets[0].get("methods", {}) if isinstance(datasets[0], Mapping) else {}
    if not isinstance(methods, Mapping) or not methods:
        return {}
    first_key = next(iter(methods))
    method = methods.get(first_key, {})
    return method if isinstance(method, Mapping) else {}


def qa_rows_by_query(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.exists():
        return {}
    method = method_payload(read_json(path))
    return {
        safe_int(row.get("query_index"), default=-1): row
        for row in method.get("per_query", []) or []
        if isinstance(row, Mapping) and safe_int(row.get("query_index"), default=-1) >= 0
    }


def doc_title_from_pool(record: Mapping[str, Any], doc_id: int) -> str:
    docs = unique_ints(record.get("pool_doc_ids", []) or [])
    titles = list(record.get("pool_titles", []) or [])
    for pos, item in enumerate(docs):
        if int(item) == int(doc_id) and pos < len(titles):
            return str(titles[pos] or "")
    return ""


def doc_title_from_openie(openie_docs: Sequence[Mapping[str, Any]], doc_id: int) -> str:
    if int(doc_id) < 0 or int(doc_id) >= len(openie_docs):
        return ""
    passage = str(openie_docs[int(doc_id)].get("passage") or "")
    return passage.split("\n", 1)[0].strip()


def doc_title(record: Mapping[str, Any], openie_docs: Sequence[Mapping[str, Any]], doc_id: int) -> str:
    return doc_title_from_pool(record, doc_id) or doc_title_from_openie(openie_docs, doc_id)


def selected_doc_indices(trace: Mapping[str, Any], pool_docs: Sequence[int]) -> list[int]:
    selector_trace = trace.get("selector_trace", {}) or {}
    for key in ("final_front_doc_ids", "selected_doc_ids"):
        docs = unique_ints(selector_trace.get(key, []) or [], limit=TOP_K)
        if docs:
            return docs
    for key in ("final_front_pool_positions", "selected_pool_positions", "selected_positions"):
        positions = unique_ints(selector_trace.get(key, []) or [], limit=TOP_K)
        docs = [int(pool_docs[pos]) for pos in positions if 0 <= int(pos) < len(pool_docs)]
        if docs:
            return unique_ints(docs, limit=TOP_K)
    return []


def selected_titles(trace: Mapping[str, Any], pool_record: Mapping[str, Any]) -> list[str]:
    selector_trace = trace.get("selector_trace", {}) or {}
    titles = unique_texts(selector_trace.get("final_front_titles", []) or selector_trace.get("selected_titles", []) or [])
    if titles:
        return titles[:TOP_K]
    return [doc_title_from_pool(pool_record, doc_id) for doc_id in selected_doc_indices(trace, pool_record.get("pool_doc_ids", []) or [])]


def binding_candidate_items(selector_trace: Mapping[str, Any], pool_docs: Sequence[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    raw = selector_trace.get("binding_candidates_by_requirement", {}) or {}
    if not isinstance(raw, Mapping):
        return output
    for requirement_id, items in raw.items():
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            pos = safe_int(item.get("title_pool_position"), default=-1)
            doc_id = int(pool_docs[pos]) if 0 <= pos < len(pool_docs) else None
            output.append(
                {
                    "requirement_id": str(requirement_id),
                    "title": str(item.get("title") or ""),
                    "position": pos if pos >= 0 else None,
                    "doc_id": doc_id,
                    "dep": str(item.get("dep") or ""),
                    "dep_position": safe_int(item.get("dep_position"), default=-1),
                    "dep_score": safe_float(item.get("dep_score")),
                    "llm_extracted_entity": str(item.get("llm_extracted_entity") or ""),
                    "entity_match_type": str(item.get("entity_match_type") or ""),
                }
            )
    return output


def selected_binding_titles(selector_trace: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    selected = selector_trace.get("selected_binding", {}) or {}
    if isinstance(selected, Mapping):
        assignments = selected.get("assignments", {}) or {}
        if isinstance(assignments, Mapping):
            values.extend(str(value) for value in assignments.values())
    for binding in selector_trace.get("bindings", []) or []:
        if not isinstance(binding, Mapping):
            continue
        assignments = binding.get("assignments", {}) or {}
        if isinstance(assignments, Mapping):
            values.extend(str(value) for value in assignments.values())
    return unique_texts(values)


def safe_projection_positions(selector_trace: Mapping[str, Any], pool_size: int) -> dict[str, set[int]]:
    safe_trace = selector_trace.get("safe_projection_trace", {}) or {}
    if not isinstance(safe_trace, Mapping):
        safe_trace = {}
    rebuild_positions = {
        pos
        for pos in unique_ints(safe_trace.get("rebuild_positions", []) or [])
        if 0 <= int(pos) < pool_size
    }
    swap_in_positions = set()
    for step in safe_trace.get("safe_swap_steps", []) or []:
        if not isinstance(step, Mapping):
            continue
        pos = safe_int(step.get("in_position"), default=-1)
        if 0 <= pos < pool_size:
            swap_in_positions.add(pos)
    return {
        "rebuild_positions": rebuild_positions,
        "swap_in_positions": swap_in_positions,
    }


def near_title_substitution(missing_titles: Sequence[str], selected_non_gold_titles: Sequence[str]) -> bool:
    missing_norms = {normalize_title(title) for title in missing_titles}
    missing_token_sets = [title_tokens(title) for title in missing_titles]
    for selected in selected_non_gold_titles:
        norm = normalize_title(selected)
        if not norm:
            continue
        if norm in missing_norms:
            return True
        selected_tokens = title_tokens(selected)
        for missing_tokens in missing_token_sets:
            if jaccard(selected_tokens, missing_tokens) >= 0.5:
                return True
    return False


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    etv3_retrieval = read_json(Path(args.etv3_retrieval))
    openie_path = Path(str(etv3_retrieval.get("openie_path") or ""))
    if openie_path and not openie_path.is_absolute():
        openie_path = (Path(args.etv3_retrieval).parent / openie_path).resolve()
    openie_docs = []
    if openie_path.exists():
        openie_docs = list((read_json(openie_path).get("docs", []) or []))
    stable_dbec = read_json(Path(args.stable_dbec))
    pool_payload = read_json(Path(args.etv3_pool))
    conservative_dbec = read_json(Path(args.conservative_dbec)) if Path(args.conservative_dbec).exists() else {}

    etv3_rows = {
        safe_int(row.get("query_index"), default=-1): row
        for row in etv3_retrieval.get("rows", []) or []
        if isinstance(row, Mapping)
    }
    etv3_by_question = {str(row.get("question") or ""): row for row in etv3_rows.values()}
    pool_by_question = {
        str(record.get("question") or ""): record
        for record in pool_payload.get("records", []) or []
        if isinstance(record, Mapping)
    }
    pool_by_qidx = {
        safe_int(record.get("query_idx"), default=-1): record
        for record in pool_payload.get("records", []) or []
        if isinstance(record, Mapping)
    }
    stable_traces = [
        row
        for row in stable_dbec.get("setwise_selector_query_traces", []) or []
        if isinstance(row, Mapping)
    ]
    conservative_by_question = {
        str(row.get("question") or ""): row
        for row in conservative_dbec.get("setwise_selector_query_traces", []) or []
        if isinstance(row, Mapping)
    }
    etv3_qa = qa_rows_by_query(Path(args.etv3_qa))
    etv4_reader10_qa = qa_rows_by_query(Path(args.etv4_reader10_qa))
    stable_reader10_qa = qa_rows_by_query(Path(args.stable_dbec_reader10_qa))
    stable_reader10_input = {}
    if Path(args.stable_dbec_reader10_input).exists():
        stable_reader10_payload = read_json(Path(args.stable_dbec_reader10_input))
        stable_reader10_input = {
            safe_int(row.get("query_index"), default=-1): row
            for row in stable_reader10_payload.get("rows", []) or []
            if isinstance(row, Mapping)
        }

    query_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for trace in stable_traces:
        if safe_int(trace.get("gold_doc_count"), default=0) != 4:
            continue
        question = str(trace.get("question") or "")
        pool_record = pool_by_question.get(question, {})
        qidx = safe_int(pool_record.get("query_idx"), default=-1)
        etv3_row = etv3_rows.get(qidx) or etv3_by_question.get(question, {})
        if qidx < 0:
            qidx = safe_int(etv3_row.get("query_index"), default=-1)
            pool_record = pool_by_qidx.get(qidx, pool_record)

        pool_docs = unique_ints(pool_record.get("pool_doc_ids", []) or [])
        pool_titles = list(pool_record.get("pool_titles", []) or [])
        candidate200 = unique_ints(
            (((etv3_row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}).get("candidate_doc_indices", []) or [])
        )
        etv3_top5 = unique_ints(etv3_row.get("retrieved_doc_indices_top5", []) or [], limit=TOP_K)
        gold_docs = unique_ints(etv3_row.get("gold_doc_indices", []) or [])
        stable_selected = selected_doc_indices(trace, pool_docs)
        stable_selected_titles = selected_titles(trace, pool_record)
        stable_selected_set = set(stable_selected)
        gold_set = set(gold_docs)
        missing_docs = [doc_id for doc_id in gold_docs if doc_id not in stable_selected_set]
        missing_titles = [
            doc_title(pool_record, openie_docs, doc_id)
            or next(
                (
                    str(title)
                    for title in trace.get("gold_titles", []) or []
                    if normalize_title(title) == normalize_title(doc_title_from_pool(pool_record, doc_id))
                ),
                "",
            )
            for doc_id in missing_docs
        ]
        selected_non_gold_titles = [
            title
            for doc_id, title in zip(stable_selected, stable_selected_titles)
            if int(doc_id) not in gold_set
        ]

        selector_trace = trace.get("selector_trace", {}) or {}
        binding_candidates = binding_candidate_items(selector_trace, pool_docs)
        binding_candidate_doc_ids = {int(item["doc_id"]) for item in binding_candidates if item.get("doc_id") is not None}
        binding_candidate_norms = {normalize_title(item.get("title")) for item in binding_candidates}
        binding_titles = selected_binding_titles(selector_trace)
        binding_title_norms = {normalize_title(title) for title in binding_titles}
        projection_positions = safe_projection_positions(selector_trace, len(pool_docs))
        rebuild_doc_ids = {
            int(pool_docs[pos])
            for pos in projection_positions["rebuild_positions"]
            if 0 <= pos < len(pool_docs)
        }
        swap_in_doc_ids = {
            int(pool_docs[pos])
            for pos in projection_positions["swap_in_positions"]
            if 0 <= pos < len(pool_docs)
        }
        non_gold_binding_titles = [
            title
            for title in binding_titles
            if normalize_title(title) not in {normalize_title(item) for item in trace.get("gold_titles", []) or []}
        ]
        binding_wrong_branch_proxy = bool(non_gold_binding_titles and missing_docs)
        near_substitution = near_title_substitution(missing_titles, selected_non_gold_titles)

        missing_statuses: list[dict[str, Any]] = []
        for doc_id, title in zip(missing_docs, missing_titles):
            pos100 = first_position(doc_id, pool_docs)
            pos200 = first_position(doc_id, candidate200)
            title_norm = normalize_title(title)
            state_bindable_proxy = (
                int(doc_id) in binding_candidate_doc_ids
                or title_norm in binding_candidate_norms
                or title_norm in binding_title_norms
            )
            positive_rebuild_signal = int(doc_id) in rebuild_doc_ids
            positive_swap_signal = int(doc_id) in swap_in_doc_ids
            positive_dbec_signal = positive_rebuild_signal or positive_swap_signal
            status = {
                "query_index": qidx,
                "question": question,
                "missing_doc_id": int(doc_id),
                "missing_title": title,
                "rank_pool100": None if pos100 is None else pos100 + 1,
                "rank_candidate200": None if pos200 is None else pos200 + 1,
                "rank_bucket": rank_bucket(pos100 if pos100 is not None else pos200),
                "in_pool100": pos100 is not None,
                "in_candidate200": pos200 is not None,
                "state_bindable_proxy": state_bindable_proxy,
                "positive_rebuild_signal": positive_rebuild_signal,
                "positive_swap_signal": positive_swap_signal,
                "positive_dbec_signal": positive_dbec_signal,
                "binding_candidate_doc": int(doc_id) in binding_candidate_doc_ids,
                "binding_candidate_title": title_norm in binding_candidate_norms,
                "selected_binding_title": title_norm in binding_title_norms,
            }
            missing_statuses.append(status)
            missing_rows.append(
                {
                    **status,
                    "stable_f1": safe_float((trace.get("selector_metrics", {}) or {}).get("F1")),
                    "stable_em": safe_float((trace.get("selector_metrics", {}) or {}).get("ExactMatch")),
                    "etv3_f1": safe_float((trace.get("baseline_metrics", {}) or {}).get("F1")),
                    "etv3_em": safe_float((trace.get("baseline_metrics", {}) or {}).get("ExactMatch")),
                    "selected_non_gold_titles": " || ".join(selected_non_gold_titles),
                    "selected_binding_titles": " || ".join(binding_titles),
                    "binding_wrong_branch_proxy": binding_wrong_branch_proxy,
                    "near_title_substitution_proxy": near_substitution,
                }
            )

        support_incomplete = bool(missing_docs)
        stable_f1 = safe_float((trace.get("selector_metrics", {}) or {}).get("F1"))
        stable_em = safe_float((trace.get("selector_metrics", {}) or {}).get("ExactMatch"))
        all_missing_in_pool100 = bool(missing_statuses) and all(row["in_pool100"] for row in missing_statuses)
        all_missing_in_candidate200 = bool(missing_statuses) and all(row["in_candidate200"] for row in missing_statuses)
        all_missing_state_bindable = bool(missing_statuses) and all(row["state_bindable_proxy"] for row in missing_statuses)
        any_missing_state_bindable = any(row["state_bindable_proxy"] for row in missing_statuses)
        all_missing_positive = bool(missing_statuses) and all(row["positive_dbec_signal"] for row in missing_statuses)
        any_missing_positive = any(row["positive_dbec_signal"] for row in missing_statuses)
        stable_reader10_docs = unique_ints(
            (stable_reader10_input.get(qidx, {}) or {}).get("retrieved_doc_indices_top5", []) or []
        )
        all_missing_in_stable_reader10 = bool(missing_docs) and set(missing_docs).issubset(set(stable_reader10_docs))
        stable_reader10_row = stable_reader10_qa.get(qidx, {})
        etv4_reader10_row = etv4_reader10_qa.get(qidx, {})
        etv3_qa_row = etv3_qa.get(qidx, {})
        conservative_row = conservative_by_question.get(question, {})
        conservative_f1 = safe_float((conservative_row.get("selector_metrics", {}) or {}).get("F1"), default=-1.0)

        if stable_f1 >= 1.0 - EPS:
            primary_bucket = "already_answered_by_stable_dbec"
        elif not support_incomplete:
            primary_bucket = "D_reader_failure_all_gold_selected"
        elif not all_missing_in_candidate200:
            primary_bucket = "C_missing_from_etv3_candidate200"
        elif not all_missing_in_pool100:
            primary_bucket = "C_outside_dbec_pool100_inside_candidate200"
        elif all_missing_state_bindable and all_missing_positive:
            primary_bucket = "A_strict_state_binding_feasible"
        elif any_missing_state_bindable or any_missing_positive:
            primary_bucket = "A_partial_binding_or_objective_signal"
        elif binding_wrong_branch_proxy:
            primary_bucket = "B_wrong_binding_or_state_extraction"
        elif all_missing_in_stable_reader10:
            primary_bucket = "D_budget_exposure_candidate10"
        else:
            primary_bucket = "B_calibration_or_unmodeled_state"

        query_rows.append(
            {
                "query_index": qidx,
                "question": question,
                "gold_titles": " || ".join(str(item) for item in trace.get("gold_titles", []) or []),
                "gold_doc_indices": json.dumps(gold_docs),
                "etv3_top5_titles": " || ".join(
                    doc_title(pool_record, openie_docs, doc_id) for doc_id in etv3_top5
                ),
                "stable_top5_titles": " || ".join(stable_selected_titles),
                "missing_gold_count_after_stable": len(missing_docs),
                "missing_gold_titles_after_stable": " || ".join(missing_titles),
                "stable_f1": stable_f1,
                "stable_em": stable_em,
                "etv3_f1": safe_float((trace.get("baseline_metrics", {}) or {}).get("F1")),
                "etv3_em": safe_float((trace.get("baseline_metrics", {}) or {}).get("ExactMatch")),
                "etv3_reader10_f1": safe_float(etv4_reader10_row.get("F1"), default=-1.0),
                "stable_dbec_reader10_f1": safe_float(stable_reader10_row.get("F1"), default=-1.0),
                "conservative_dbec_f1": conservative_f1,
                "etv3_qa_f1": safe_float(etv3_qa_row.get("F1"), default=-1.0),
                "all_missing_in_pool100": all_missing_in_pool100,
                "all_missing_in_candidate200": all_missing_in_candidate200,
                "all_missing_in_stable_reader10": all_missing_in_stable_reader10,
                "all_missing_state_bindable_proxy": all_missing_state_bindable,
                "any_missing_state_bindable_proxy": any_missing_state_bindable,
                "all_missing_positive_dbec_signal": all_missing_positive,
                "any_missing_positive_dbec_signal": any_missing_positive,
                "binding_candidate_titles": " || ".join(item["title"] for item in binding_candidates),
                "selected_binding_titles": " || ".join(binding_titles),
                "binding_wrong_branch_proxy": binding_wrong_branch_proxy,
                "near_title_substitution_proxy": near_substitution,
                "selected_non_gold_titles": " || ".join(selected_non_gold_titles),
                "safe_decision": str((selector_trace.get("safe_projection_trace", {}) or {}).get("safe_decision") or ""),
                "safe_rebuild_titles": " || ".join(
                    str(item)
                    for item in ((selector_trace.get("safe_projection_trace", {}) or {}).get("rebuild_titles", []) or [])
                ),
                "safe_swap_count": len((selector_trace.get("safe_projection_trace", {}) or {}).get("safe_swap_steps", []) or []),
                "primary_bucket": primary_bucket,
            }
        )

    residual_rows = [row for row in query_rows if safe_float(row["stable_f1"]) < 1.0 - EPS]
    support_incomplete_rows = [row for row in residual_rows if safe_int(row["missing_gold_count_after_stable"], 0) > 0]
    bucket_counts = Counter(row["primary_bucket"] for row in residual_rows)
    all_rows = query_rows
    bucket_a_rows = [
        row
        for row in residual_rows
        if row["primary_bucket"] in {
            "A_strict_state_binding_feasible",
            "A_partial_binding_or_objective_signal",
        }
    ]
    strict_a_rows = [row for row in residual_rows if row["primary_bucket"] == "A_strict_state_binding_feasible"]

    current_f1_sum = sum(safe_float(row["stable_f1"]) for row in all_rows)
    strict_a_ceiling_f1 = (current_f1_sum + sum(1.0 - safe_float(row["stable_f1"]) for row in strict_a_rows)) / max(len(all_rows), 1)
    relaxed_a_ceiling_f1 = (current_f1_sum + sum(1.0 - safe_float(row["stable_f1"]) for row in bucket_a_rows)) / max(len(all_rows), 1)
    support_all_fixed_ceiling_f1 = (
        current_f1_sum
        + sum(1.0 - safe_float(row["stable_f1"]) for row in support_incomplete_rows)
    ) / max(len(all_rows), 1)

    summary = {
        "dataset": "musique",
        "slice": "gold_doc_count == 4, limit100",
        "counts": {
            "queries_4doc": len(query_rows),
            "stable_dbec_residual_answer_failures": len(residual_rows),
            "stable_dbec_support_incomplete_residual_failures": len(support_incomplete_rows),
            "bucket_counts": dict(bucket_counts),
        },
        "rates": {
            "strict_A_rate_among_residual": len(strict_a_rows) / max(len(residual_rows), 1),
            "relaxed_A_rate_among_residual": len(bucket_a_rows) / max(len(residual_rows), 1),
            "support_incomplete_rate_among_residual": len(support_incomplete_rows) / max(len(residual_rows), 1),
            "wrong_binding_proxy_rate_among_residual": sum(
                1 for row in residual_rows if bool(row["binding_wrong_branch_proxy"])
            )
            / max(len(residual_rows), 1),
            "budget_exposure_rate_among_residual": sum(
                1 for row in residual_rows if bool(row["all_missing_in_stable_reader10"])
            )
            / max(len(residual_rows), 1),
        },
        "f1": {
            "stable_dbec_4doc_f1": mean([safe_float(row["stable_f1"]) for row in query_rows]) if query_rows else 0.0,
            "etv3_4doc_f1": mean([safe_float(row["etv3_f1"]) for row in query_rows]) if query_rows else 0.0,
            "conservative_dbec_4doc_f1": mean(
                [safe_float(row["conservative_dbec_f1"]) for row in query_rows if safe_float(row["conservative_dbec_f1"], -1.0) >= 0.0]
            )
            if query_rows
            else 0.0,
            "strict_A_fixed_ceiling_f1": strict_a_ceiling_f1,
            "relaxed_A_fixed_ceiling_f1": relaxed_a_ceiling_f1,
            "all_support_incomplete_fixed_ceiling_f1": support_all_fixed_ceiling_f1,
        },
        "proceed_gate": {
            "strict_A_rate_threshold": 0.5,
            "relaxed_A_rate_threshold": 0.6,
            "ceiling_f1_threshold": 0.28,
            "strict_A_pass": len(strict_a_rows) / max(len(residual_rows), 1) >= 0.5
            and strict_a_ceiling_f1 >= 0.28,
            "relaxed_A_pass": len(bucket_a_rows) / max(len(residual_rows), 1) >= 0.6
            and relaxed_a_ceiling_f1 >= 0.28,
        },
        "notes": {
            "state_bindable_proxy": (
                "A missing gold document is counted as state-bindable only when it "
                "appears as a DBEC LLM binding candidate or selected binding title/doc."
            ),
            "positive_dbec_signal": (
                "A missing gold document has positive DBEC signal when it appears in "
                "the full DBEC rebuild set or in a safe swap-in step."
            ),
            "wrong_binding_proxy": (
                "Selected binding titles that are not gold titles while gold remains "
                "missing. This is a proxy requiring manual review, not ground truth."
            ),
        },
    }
    return {
        "metadata": {
            "etv3_retrieval": str(Path(args.etv3_retrieval)),
            "openie_path": str(openie_path),
            "etv3_qa": str(Path(args.etv3_qa)),
            "etv4_reader10_qa": str(Path(args.etv4_reader10_qa)),
            "stable_dbec": str(Path(args.stable_dbec)),
            "conservative_dbec": str(Path(args.conservative_dbec)),
            "etv3_pool": str(Path(args.etv3_pool)),
            "stable_dbec_reader10_input": str(Path(args.stable_dbec_reader10_input)),
            "stable_dbec_reader10_qa": str(Path(args.stable_dbec_reader10_qa)),
        },
        "summary": summary,
        "query_rows": query_rows,
        "missing_gold_rows": missing_rows,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def write_markdown(payload: Mapping[str, Any], output_md: Path) -> None:
    summary = payload["summary"]
    counts = summary["counts"]
    rates = summary["rates"]
    f1 = summary["f1"]
    gate = summary["proceed_gate"]
    bucket_counts = counts["bucket_counts"]
    query_rows = payload["query_rows"]
    residual_rows = [row for row in query_rows if safe_float(row["stable_f1"]) < 1.0 - EPS]

    lines = [
        "# ETv4 Binding Feasibility Audit",
        "",
        "Scope: MuSiQue limit100, 4-doc slice, frozen ETv3 retrieval plus baseline-stable DBEC.",
        "No new LLM calls or reader runs were made.",
        "",
        "## Headline",
        "",
        (
            f"- 4-doc queries: `{counts['queries_4doc']}`; stable DBEC residual answer failures: "
            f"`{counts['stable_dbec_residual_answer_failures']}`."
        ),
        (
            f"- Strict state-binding-feasible residual rate: "
            f"`{pct(rates['strict_A_rate_among_residual'])}`; relaxed binding/objective-signal rate: "
            f"`{pct(rates['relaxed_A_rate_among_residual'])}`."
        ),
        (
            f"- Stable DBEC 4-doc F1: `{fmt(f1['stable_dbec_4doc_f1'])}`; strict-A fixed ceiling: "
            f"`{fmt(f1['strict_A_fixed_ceiling_f1'])}`; relaxed-A fixed ceiling: "
            f"`{fmt(f1['relaxed_A_fixed_ceiling_f1'])}`."
        ),
        (
            f"- Proceed gate: strict-A pass = `{gate['strict_A_pass']}`, relaxed-A pass = "
            f"`{gate['relaxed_A_pass']}`."
        ),
        "",
        "## Bucket Counts",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## F1 Ceilings",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| ETv3 4-doc F1 | {fmt(f1['etv3_4doc_f1'])} |",
            f"| Stable DBEC 4-doc F1 | {fmt(f1['stable_dbec_4doc_f1'])} |",
            f"| Conservative diagnostic DBEC 4-doc F1 | {fmt(f1['conservative_dbec_4doc_f1'])} |",
            f"| If strict-A residuals are all fixed | {fmt(f1['strict_A_fixed_ceiling_f1'])} |",
            f"| If relaxed-A residuals are all fixed | {fmt(f1['relaxed_A_fixed_ceiling_f1'])} |",
            f"| If all support-incomplete residuals are fixed | {fmt(f1['all_support_incomplete_fixed_ceiling_f1'])} |",
            "",
            "## Residual Failure Table",
            "",
            "| qid | bucket | stable F1 | missing gold | state-bindable | positive DBEC signal | wrong-binding proxy | stable reader@10 F1 | selected non-gold |",
            "|---:|---|---:|---:|---|---|---|---:|---|",
        ]
    )
    for row in residual_rows:
        lines.append(
            "| {qid} | {bucket} | {f1} | {missing} | {bind} | {pos} | {wrong} | {reader10} | {nongold} |".format(
                qid=row["query_index"],
                bucket=row["primary_bucket"],
                f1=fmt(row["stable_f1"]),
                missing=row["missing_gold_count_after_stable"],
                bind="yes" if row["any_missing_state_bindable_proxy"] else "no",
                pos="yes" if row["any_missing_positive_dbec_signal"] else "no",
                wrong="yes" if row["binding_wrong_branch_proxy"] else "no",
                reader10=fmt(row["stable_dbec_reader10_f1"]),
                nongold=str(row["selected_non_gold_titles"])[:120].replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- `state_bindable_proxy` is conservative: the missing gold must appear in DBEC's LLM binding candidates or selected binding titles/docs.",
            "- `positive_dbec_signal` means the missing gold appears in the DBEC full rebuild set or a safe swap-in step.",
            "- `wrong_binding_proxy` is not ground truth; it flags non-gold selected bindings while gold evidence remains missing.",
            "- These numbers decide whether ETv4-binding is worth implementing; they are not evidence that ETv4-binding already works.",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etv3-retrieval", type=Path, default=DEFAULT_ETV3_RETRIEVAL)
    parser.add_argument("--etv3-qa", type=Path, default=DEFAULT_ETV3_QA)
    parser.add_argument("--etv4-reader10-qa", type=Path, default=DEFAULT_ETV4_READER10_QA)
    parser.add_argument("--stable-dbec", type=Path, default=DEFAULT_STABLE_DBEC)
    parser.add_argument("--conservative-dbec", type=Path, default=DEFAULT_CONSERVATIVE_DBEC)
    parser.add_argument("--etv3-pool", type=Path, default=DEFAULT_ETV3_POOL)
    parser.add_argument("--stable-dbec-reader10-input", type=Path, default=DEFAULT_STABLE_DBEC_READER10_INPUT)
    parser.add_argument("--stable-dbec-reader10-qa", type=Path, default=DEFAULT_STABLE_DBEC_READER10_QA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = build_audit(args)
    report_dir = Path(args.report_dir)
    write_json(payload, report_dir / "etv4_binding_feasibility_audit.json")
    write_csv(payload["query_rows"], report_dir / "etv4_binding_feasibility_queries.csv")
    write_csv(payload["missing_gold_rows"], report_dir / "etv4_binding_feasibility_missing_gold.csv")
    write_markdown(payload, report_dir / "etv4_binding_feasibility_audit.md")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
