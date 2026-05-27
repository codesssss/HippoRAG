"""Shared AREC-RAG smoke-script helpers."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from src.dpathrag.arec.closure import closure_score, greedy_closure_select, obligation_closure
from src.dpathrag.arec.obligations import (
    Obligation,
    build_obligation_query,
    normalize_obligations,
    obligations_to_dicts,
    parse_obligations,
)
from src.dpathrag.arec.pool import PoolDoc, missing_gold_titles, pool_docs, selected_titles
from src.dpathrag.arec.retrieval import missing_support_hit_rate, top_indices_for_queries
from src.dpathrag.arec.verifier import Verifier, support_matrix
from src.dpathrag.data import normalize_text
from src.dpathrag.io import read_json
from src.dpathrag.metrics import recall_at_k, support_complete_at_k
from src.dpathrag.reader import exact_match, token_f1


def load_pool_records(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        data = list(data.get("records") or data.get("rows") or data.get("data") or [])
    rows = list(data)
    return rows[: int(limit)] if int(limit) > 0 else rows


def load_jsonl_map(path: str | Path, *, key_field: str = "qid") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path or not Path(path).exists():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            key = str(payload.get(key_field) or payload.get("id") or payload.get("query_idx") or "")
            if key:
                rows[key] = payload
    return rows


def qid_for(record: dict[str, Any], index: int) -> str:
    return str(record.get("qid") or record.get("id") or record.get("_id") or record.get("query_idx") or index)


def hop_bucket(record: dict[str, Any]) -> str:
    if record.get("hop_bucket") is not None:
        return str(record["hop_bucket"])
    gold_titles = list(record.get("gold_titles") or [])
    return str(len(gold_titles)) if gold_titles else "unknown"


def gold_answers(record: dict[str, Any]) -> list[str]:
    values = record.get("gold_answers")
    if values is None:
        values = record.get("answer")
    if isinstance(values, list):
        return [str(item) for item in values]
    return [str(values or "")]


def initial_answer_for(
    record: dict[str, Any],
    *,
    mock_reader_from_gold: bool = False,
    annotation: dict[str, Any] | None = None,
) -> str:
    annotation = annotation or {}
    for key in ("initial_answer", "answer", "prediction", "reader_answer"):
        if annotation.get(key):
            return str(annotation[key])
    for key in ("initial_answer", "prediction", "answer_prediction", "reader_answer"):
        if record.get(key):
            return str(record[key])
    if mock_reader_from_gold:
        return gold_answers(record)[0]
    return ""


def obligations_for(
    record: dict[str, Any],
    *,
    index: int,
    obligation_map: dict[str, dict[str, Any]],
    answer: str,
    source: str,
    require_active: bool = True,
) -> list[Obligation]:
    qid = qid_for(record, index)
    raw = None
    if qid in obligation_map:
        raw = obligation_map[qid].get("obligations") or obligation_map[qid]
    elif record.get("obligations"):
        raw = record.get("obligations")
    if raw is None:
        return []
    return normalize_obligations(parse_obligations(raw, answer=answer, source=source), answer=answer, require_active=require_active)


def make_oracle_template(records: Sequence[dict[str, Any]], *, limit: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(list(records)[: int(limit)]):
        rows.append(
            {
                "qid": qid_for(record, idx),
                "question": record.get("question"),
                "gold_titles": list(record.get("gold_titles") or []),
                "gold_answers": gold_answers(record),
                "obligations": [
                    {
                        "id": "o1",
                        "claim": "TODO: write a verifier-checkable support claim from the gold chain.",
                        "type": "factual",
                        "retrieval_active": True,
                    }
                ],
            }
        )
    return rows


def score_current_closure(
    obligations: Sequence[Obligation],
    docs: Sequence[PoolDoc],
    selected_indices: Sequence[int],
    verifier: Verifier,
) -> tuple[list[list[float]], float, list[float]]:
    claims = [obligation.claim for obligation in obligations if obligation.retrieval_active]
    matrix = support_matrix(claims, [doc.full_text for doc in docs], verifier) if claims else []
    selected = [int(idx) for idx in selected_indices if 0 <= int(idx) < len(docs)]
    score = closure_score(matrix, selected) if matrix else 0.0
    per_obligation = obligation_closure(matrix, selected).tolist() if matrix else []
    return matrix, float(score), [float(value) for value in per_obligation]


def retrieve_for_obligations(
    record: dict[str, Any],
    obligations: Sequence[Obligation],
    docs: Sequence[PoolDoc],
    *,
    answer: str,
    per_query_k: int,
    exclude_titles: Sequence[str],
) -> list[int]:
    queries = [
        obligation.query or build_obligation_query(str(record.get("question") or ""), answer, obligation.claim)
        for obligation in obligations
        if obligation.retrieval_active
    ]
    return top_indices_for_queries(queries, docs, per_query_k=int(per_query_k), exclude_titles=exclude_titles)


def transition_category(initial_em: float, final_em: float) -> str:
    if initial_em >= 1.0 and final_em >= 1.0:
        return "correct_to_correct"
    if initial_em >= 1.0 and final_em < 1.0:
        return "correct_to_wrong"
    if initial_em < 1.0 and final_em >= 1.0:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def closure_row(
    record: dict[str, Any],
    *,
    index: int,
    dataset: str,
    top_k: int,
    max_docs: int,
    obligation_map: dict[str, dict[str, Any]],
    oracle_map: dict[str, dict[str, Any]] | None,
    verifier: Verifier,
    mock_reader_from_gold: bool = False,
) -> dict[str, Any]:
    start = perf_counter()
    docs = pool_docs(record, max_docs=max_docs)
    selected = list(range(min(int(top_k), len(docs))))
    selected_names = selected_titles(docs, selected)
    qid = qid_for(record, index)
    answer = initial_answer_for(record, mock_reader_from_gold=mock_reader_from_gold, annotation=obligation_map.get(qid))
    obligations = obligations_for(record, index=index, obligation_map=obligation_map, answer=answer, source="generated")
    matrix, score, per_obligation = score_current_closure(obligations, docs, selected, verifier)
    oracle_obligations: list[Obligation] = []
    oracle_score = 0.0
    if oracle_map is not None:
        oracle_obligations = obligations_for(record, index=index, obligation_map=oracle_map, answer=answer, source="oracle")
        _, oracle_score, _ = score_current_closure(oracle_obligations, docs, selected, verifier)
    answers = gold_answers(record)
    initial_em = exact_match(answers, answer) if answer else 0.0
    initial_f1 = token_f1(answers, answer) if answer else 0.0
    gold_titles = list(record.get("gold_titles") or [])
    elapsed = perf_counter() - start
    return {
        "qid": qid,
        "dataset": dataset,
        "hop_bucket": hop_bucket(record),
        "question": record.get("question"),
        "gold_answers": answers,
        "gold_titles": gold_titles,
        "initial_selected_titles": selected_names,
        "initial_answer": answer,
        "initial_em": initial_em,
        "initial_f1": initial_f1,
        "initial_support_recall": recall_at_k(gold_titles, selected_names, int(top_k)),
        "initial_support_complete": support_complete_at_k(gold_titles, selected_names, int(top_k)),
        "obligations": obligations_to_dicts(obligations),
        "oracle_obligations": obligations_to_dicts(oracle_obligations),
        "active_obligation_count": len([ob for ob in obligations if ob.retrieval_active]),
        "verifier_scores_current": matrix,
        "closure_score_initial": score,
        "closure_per_obligation_initial": per_obligation,
        "oracle_closure_score_initial": oracle_score,
        "open_obligations": [ob.as_dict() for ob, value in zip(obligations, per_obligation) if value < 0.95],
        "reader_calls": 0 if not answer else 1,
        "verifier_calls": len(obligations) * len(selected),
        "retriever_calls": 0,
        "token_count": 0,
        "latency_seconds": elapsed,
    }


def residual_row(
    record: dict[str, Any],
    *,
    index: int,
    dataset: str,
    top_k: int,
    max_docs: int,
    per_query_k: int,
    obligation_map: dict[str, dict[str, Any]],
    cot_query_map: dict[str, dict[str, Any]],
    verifier: Verifier,
    mock_reader_from_gold: bool = False,
    obligation_source: str = "generated",
) -> dict[str, Any]:
    start = perf_counter()
    docs = pool_docs(record, max_docs=max_docs)
    initial = list(range(min(int(top_k), len(docs))))
    initial_names = selected_titles(docs, initial)
    qid = qid_for(record, index)
    answer = initial_answer_for(record, mock_reader_from_gold=mock_reader_from_gold, annotation=obligation_map.get(qid))
    obligations = obligations_for(record, index=index, obligation_map=obligation_map, answer=answer, source=obligation_source)
    matrix, score, per_obligation = score_current_closure(obligations, docs, initial, verifier)
    open_obligations = [ob for ob, value in zip(obligations, per_obligation) if value < 0.95]
    raw_indices = top_indices_for_queries([str(record.get("question") or "")], docs, per_query_k=int(per_query_k), exclude_titles=initial_names)
    cot_query = str((cot_query_map.get(qid) or {}).get("query") or (cot_query_map.get(qid) or {}).get("cot_step_query") or record.get("cot_step_query") or "")
    cot_indices = top_indices_for_queries([cot_query], docs, per_query_k=int(per_query_k), exclude_titles=initial_names) if cot_query else []
    residual_indices = retrieve_for_obligations(
        record,
        open_obligations,
        docs,
        answer=answer,
        per_query_k=int(per_query_k),
        exclude_titles=initial_names,
    )

    workspace = list(dict.fromkeys(initial + residual_indices))
    workspace_matrix = support_matrix(
        [ob.claim for ob in obligations if ob.retrieval_active],
        [docs[idx].full_text for idx in workspace],
        verifier,
    ) if obligations and workspace else []
    selected_local, objective = greedy_closure_select(workspace_matrix, budget=int(top_k)) if workspace_matrix else (initial[: int(top_k)], 0.0)
    projected_indices = [workspace[idx] for idx in selected_local if 0 <= int(idx) < len(workspace)]
    projected_names = selected_titles(docs, projected_indices)
    raw_titles = selected_titles(docs, raw_indices[: int(per_query_k)])
    cot_titles = selected_titles(docs, cot_indices[: int(per_query_k)])
    residual_titles = selected_titles(docs, residual_indices[: int(per_query_k)])
    gold_titles = list(record.get("gold_titles") or [])
    answers = gold_answers(record)
    final_answer = answer
    initial_em = exact_match(answers, answer) if answer else 0.0
    initial_f1 = token_f1(answers, answer) if answer else 0.0
    final_em = initial_em
    final_f1 = initial_f1
    elapsed = perf_counter() - start
    missing = missing_gold_titles(gold_titles, initial_names)
    return {
        "qid": qid,
        "dataset": dataset,
        "hop_bucket": hop_bucket(record),
        "question": record.get("question"),
        "gold_answers": answers,
        "gold_titles": gold_titles,
        "initial_selected_titles": initial_names,
        "initial_answer": answer,
        "initial_em": initial_em,
        "initial_f1": initial_f1,
        "initial_support_recall": recall_at_k(gold_titles, initial_names, int(top_k)),
        "initial_support_complete": support_complete_at_k(gold_titles, initial_names, int(top_k)),
        "obligations": obligations_to_dicts(obligations),
        "active_obligation_count": len([ob for ob in obligations if ob.retrieval_active]),
        "verifier_scores_current": matrix,
        "closure_score_initial": score,
        "open_obligations": obligations_to_dicts(open_obligations),
        "raw_question_second_titles": raw_titles,
        "cot_step_query": cot_query,
        "cot_second_titles": cot_titles,
        "arec_residual_titles": residual_titles,
        "raw_question_missing_hits": len(missing & {normalize_text(title) for title in raw_titles}),
        "cot_missing_hits": len(missing & {normalize_text(title) for title in cot_titles}),
        "arec_missing_hits": len(missing & {normalize_text(title) for title in residual_titles}),
        "raw_question_missing_hit_rate": missing_support_hit_rate(gold_titles, initial_names, raw_titles),
        "cot_missing_hit_rate": missing_support_hit_rate(gold_titles, initial_names, cot_titles),
        "arec_missing_hit_rate": missing_support_hit_rate(gold_titles, initial_names, residual_titles),
        "new_gold_outside_initial_topk": sorted(missing & {normalize_text(title) for title in residual_titles}),
        "new_gold_outside_initial_top100": [],
        "projected_titles": projected_names,
        "projection_objective": float(objective),
        "final_answer": final_answer,
        "final_em": final_em,
        "final_f1": final_f1,
        "final_support_recall": recall_at_k(gold_titles, projected_names, int(top_k)),
        "final_support_complete": support_complete_at_k(gold_titles, projected_names, int(top_k)),
        "transition_category": transition_category(initial_em, final_em),
        "reader_calls": 0 if not answer else 1,
        "verifier_calls": len(obligations) * (len(initial) + len(workspace)),
        "retriever_calls": 2 + len(open_obligations),
        "token_count": 0,
        "latency_seconds": elapsed,
        "K": 1,
        "M": int(per_query_k),
        "candidate_count_before_verification": len(workspace),
        "candidate_count_after_pruning": len(workspace),
    }


def summarize_numeric(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    output: dict[str, float] = {}
    for key in keys:
        output[key] = round(float(mean(float(row.get(key) or 0.0) for row in rows)), 4)
    return output


def summarize_by_hop(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("hop_bucket") or "unknown")].append(dict(row))
    return {
        hop: summarize_numeric(items, keys) | {"rows": len(items)}
        for hop, items in sorted(groups.items())
    }
