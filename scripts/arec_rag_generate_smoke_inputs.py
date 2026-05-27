#!/usr/bin/env python3
"""Generate train-free AREC smoke inputs with a frozen OpenAI-compatible LLM.

This script prepares input artifacts for the smoke harness.  It does not
produce paper claims and it labels gold-doc-derived obligations as
``silver_oracle`` rather than manual oracle annotations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import urllib.request

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.arec.obligations import normalize_obligations, obligations_to_dicts, parse_obligations
from src.dpathrag.arec.pool import pool_docs
from src.dpathrag.arec.smoke import gold_answers, load_pool_records, qid_for
from src.dpathrag.io import write_json, write_jsonl


JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def call_chat(base_url: str, model: str, prompt: str, *, timeout: int, max_tokens: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"])


def parse_json_obj(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = JSON_RE.search(raw or "")
        if not match:
            raise
        return json.loads(match.group(0))


def evidence_block(docs: list[Any], *, max_docs: int, max_chars: int) -> str:
    parts: list[str] = []
    for idx, doc in enumerate(docs[: int(max_docs)], start=1):
        text = doc.full_text if hasattr(doc, "full_text") else str(doc)
        parts.append(f"[{idx}] {text[: int(max_chars)]}")
    return "\n\n".join(parts)


def answer_prompt(question: str, evidence: str) -> str:
    return (
        "/no_think\n"
        "Answer the question using only the provided evidence. Return JSON only.\n"
        "Schema: {\"answer\": \"short answer\"}\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence}\n"
    )


def obligations_prompt(question: str, answer: str) -> str:
    return (
        "/no_think\n"
        "Create 2 to 5 factual proof obligations needed to support the candidate answer.\n"
        "Each obligation must be atomic, verifier-checkable, and retrieval-active only if it needs external evidence.\n"
        "Do not include vague claims or restatements like 'therefore the answer is ...'.\n"
        "Return JSON only: {\"obligations\": [{\"id\": \"o1\", \"claim\": \"...\", \"type\": \"factual\", \"retrieval_active\": true}]}\n\n"
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
    )


def silver_oracle_prompt(question: str, answers: list[str], gold_docs: list[str]) -> str:
    return (
        "/no_think\n"
        "You are given the gold support documents for a multi-hop QA example.\n"
        "Write 2 to 5 factual proof obligations that the gold documents establish for the answer.\n"
        "Each obligation must be atomic and verifier-checkable. Do not copy the final answer as a standalone obligation.\n"
        "Return JSON only: {\"obligations\": [{\"id\": \"o1\", \"claim\": \"...\", \"type\": \"factual\", \"retrieval_active\": true}]}\n\n"
        f"Question: {question}\n"
        f"Gold answers: {answers}\n\n"
        f"Gold support documents:\n{evidence_block(gold_docs, max_docs=8, max_chars=1200)}\n"
    )


def cot_prompt(question: str, evidence: str) -> str:
    return (
        "/no_think\n"
        "You are running IRCoT-style retrieval for multi-hop QA.\n"
        "Given the original question and retrieved evidence, write one short follow-up retrieval query for the next missing hop.\n"
        "Do not answer the original question. Do not include reasoning.\n"
        "Return JSON only: {\"query\": \"...\"}\n\n"
        f"Question: {question}\n\n"
        f"Retrieved evidence:\n{evidence}\n"
    )


def parse_answer(raw: str) -> str:
    try:
        payload = parse_json_obj(raw)
        return str(payload.get("answer") or "").strip()
    except Exception:
        return " ".join(raw.split())[:120]


def parse_query(raw: str, fallback: str) -> str:
    try:
        payload = parse_json_obj(raw)
        query = str(payload.get("query") or "").strip()
        return query or fallback
    except Exception:
        return " ".join(raw.split())[:240] or fallback


def generate_one(record: dict[str, Any], idx: int, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    qid = qid_for(record, idx)
    question = str(record.get("question") or "")
    docs = pool_docs(record, max_docs=int(args.max_docs))
    evidence = evidence_block(docs, max_docs=int(args.top_k), max_chars=int(args.doc_max_chars))
    answer_raw = call_chat(args.llm_base_url, args.llm_model, answer_prompt(question, evidence), timeout=args.timeout, max_tokens=128)
    answer = parse_answer(answer_raw)
    obligations_raw = call_chat(args.llm_base_url, args.llm_model, obligations_prompt(question, answer), timeout=args.timeout, max_tokens=768)
    obligations = normalize_obligations(parse_obligations(obligations_raw, answer=answer, source="generated"), answer=answer, require_active=False)
    cot_raw = call_chat(args.llm_base_url, args.llm_model, cot_prompt(question, evidence), timeout=args.timeout, max_tokens=256)
    cot_query = parse_query(cot_raw, fallback=question)

    gold_docs = list(record.get("gold_docs") or [])
    silver_raw = call_chat(args.llm_base_url, args.llm_model, silver_oracle_prompt(question, gold_answers(record), gold_docs), timeout=args.timeout, max_tokens=768)
    silver = normalize_obligations(parse_obligations(silver_raw, answer=answer, source="silver_oracle"), answer=answer, require_active=False)

    generated_row = {
        "qid": qid,
        "query_idx": record.get("query_idx", idx),
        "question": question,
        "initial_answer": answer,
        "raw_initial_answer_response": answer_raw,
        "obligations": obligations_to_dicts(obligations),
        "raw_obligation_response": obligations_raw,
        "source": "frozen_llm_generated",
    }
    cot_row = {
        "qid": qid,
        "query_idx": record.get("query_idx", idx),
        "question": question,
        "query": cot_query,
        "raw_cot_response": cot_raw,
        "prompt_source": str(args.ircot_prompt_path or "local_ircot_style_frozen_prompt"),
    }
    silver_row = {
        "qid": qid,
        "query_idx": record.get("query_idx", idx),
        "question": question,
        "gold_answers": gold_answers(record),
        "gold_titles": list(record.get("gold_titles") or []),
        "initial_answer": answer,
        "obligations": obligations_to_dicts(silver),
        "raw_silver_oracle_response": silver_raw,
        "source": "silver_oracle_gold_docs_frozen_llm",
    }
    return generated_row, cot_row, silver_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_docs", type=int, default=100)
    parser.add_argument("--doc_max_chars", type=int, default=1000)
    parser.add_argument("--llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--llm_model", default="qwen3-8b-train")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ircot_prompt_path", default="")
    parser.add_argument("--generated_obligations_jsonl", default="data/dpathrag/arec/generated_obligations_musique_limit100_20260428.jsonl")
    parser.add_argument("--cot_queries_jsonl", default="data/dpathrag/arec/cot_queries_musique_limit100_20260428.jsonl")
    parser.add_argument("--silver_oracle_jsonl", default="data/dpathrag/arec/silver_oracle_obligations_musique_limit100_20260428.jsonl")
    parser.add_argument("--report_json", default="reports/dpathrag/arec_rag_generate_inputs_musique_limit100_20260428.json")
    args = parser.parse_args()

    records = load_pool_records(args.pool_json, int(args.limit))
    generated: list[dict[str, Any]] = []
    cot: list[dict[str, Any]] = []
    silver: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(generate_one, record, idx, args): idx for idx, record in enumerate(records)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                gen_row, cot_row, silver_row = future.result()
                generated.append(gen_row)
                cot.append(cot_row)
                silver.append(silver_row)
            except Exception as exc:
                errors.append(f"{idx}: {type(exc).__name__}: {exc}")
    generated.sort(key=lambda row: int(row.get("query_idx", 0)))
    cot.sort(key=lambda row: int(row.get("query_idx", 0)))
    silver.sort(key=lambda row: int(row.get("query_idx", 0)))
    write_jsonl(generated, args.generated_obligations_jsonl)
    write_jsonl(cot, args.cot_queries_jsonl)
    write_jsonl(silver, args.silver_oracle_jsonl)
    report = {
        "status": "completed" if not errors else "completed_with_errors",
        "dataset": str(args.dataset),
        "pool_json": str(args.pool_json),
        "limit": len(records),
        "generated_rows": len(generated),
        "cot_rows": len(cot),
        "silver_oracle_rows": len(silver),
        "llm_base_url": str(args.llm_base_url),
        "llm_model": str(args.llm_model),
        "ircot_prompt_path": str(args.ircot_prompt_path or "local_ircot_style_frozen_prompt"),
        "errors": errors[:50],
        "outputs": {
            "generated_obligations_jsonl": str(args.generated_obligations_jsonl),
            "cot_queries_jsonl": str(args.cot_queries_jsonl),
            "silver_oracle_jsonl": str(args.silver_oracle_jsonl),
        },
    }
    write_json(report, args.report_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

