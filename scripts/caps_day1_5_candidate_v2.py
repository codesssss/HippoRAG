#!/usr/bin/env python3
"""CAPS Day-1.5 candidate-answer generator v2 diagnostic.

This script tests whether an OpenAI-compatible local LLM can repair the CAPS
candidate-answer recall bottleneck. It is diagnostic-only: it does not run proof
scoring or held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence
from urllib import request

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from caps_day1_gates import (  # noqa: E402
    answer_recall_at,
    cluster_answer_candidates,
    dedup_union_records,
    extract_heuristic_answers,
    generate_answer_candidates,
    write_markdown,
)
from cee_day1_diagnostic import FrozenReaderLoglikProbe  # noqa: E402
from dpathrag_analyze_hard_negatives import load_jsonl, row_qid  # noqa: E402
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import gold_answers, normalize_answer  # noqa: E402


class LlmCandidateCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.rows[str(row["qid"])] = row

    def get(self, qid: str) -> dict[str, Any] | None:
        return self.rows.get(str(qid))

    def set(self, qid: str, row: dict[str, Any]) -> None:
        self.rows[str(qid)] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def qwen_health(endpoint: str) -> dict[str, Any]:
    try:
        req = request.Request(str(endpoint).rstrip("/") + "/models", method="GET")
        with request.urlopen(req, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"available": True, "status": int(response.status), "models": [row.get("id") for row in payload.get("data", [])]}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def build_llm_prompt(question: str, docs: Sequence[dict[str, Any]], *, max_doc_chars: int) -> str:
    lines = [
        "/no_think",
        "",
        f"Question: {question}",
        "",
        "Candidate documents:",
    ]
    for idx, doc in enumerate(docs, start=1):
        title = str(doc.get("title") or "")
        text = str(doc.get("text") or "")
        lines.append(f"[{idx}] {title}")
        lines.append(text[: int(max_doc_chars)])
    lines.extend(
        [
            "",
            "List up to 10 plausible candidate answers.",
            "Include entities, dates, numbers, yes/no answers, or short phrases.",
            "Be inclusive: list candidates even if uncertain.",
            "Do not explain. Do not show reasoning.",
            "Return valid JSON only, exactly in this format:",
            '{"answers": ["answer 1", "answer 2"]}',
            "",
            "Candidate answers:",
        ]
    )
    return "\n".join(lines)


def call_openai_compatible(endpoint: str, model: str, prompt: str, *, timeout: int, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
    }
    req = request.Request(
        str(endpoint).rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=int(timeout)) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


def parse_llm_candidates(text: str) -> list[str]:
    raw_text = str(text or "").strip()
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()
    json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            answers = payload.get("answers") if isinstance(payload, dict) else None
            if isinstance(answers, list):
                return dedup_candidates([str(item) for item in answers])
        except Exception:
            pass
    outputs = []
    for raw in raw_text.splitlines():
        line = raw.strip()
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        line = line.strip("`\"' ")
        if not line or line.lower() in {"<think>", "</think>"}:
            continue
        if len(line.split()) > 12:
            continue
        if not line:
            continue
        if ":" in line and len(line.split(":", 1)[0].split()) <= 3:
            line = line.split(":", 1)[1].strip()
        if normalize_answer(line):
            outputs.append(line)
    return dedup_candidates(outputs)


def dedup_candidates(outputs: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for item in outputs:
        expanded.append(str(item))
        expanded.extend(expand_candidate_variants(str(item)))
    dedup = []
    seen = set()
    for item in expanded:
        norm = normalize_answer(item)
        if norm and norm not in seen:
            seen.add(norm)
            dedup.append(item)
    return dedup[:20]


def expand_candidate_variants(text: str) -> list[str]:
    """Recover short answer entities from common explanatory LLM lines."""
    value = str(text or "").strip()
    variants: list[str] = []
    patterns = [
        r"^(.+?)\s+was born\b",
        r"^(.+?)\s+were born\b",
        r"^(.+?)\s+born\s+\d",
        r"^(.+?)\s+is older\b",
        r"^(.+?)\s+is younger\b",
        r"^(.+?)\s+was released\b",
        r"^(.+?)\s+released\s+\d",
        r"^(.+?)\s+was first\b",
        r"^(.+?)\s+is first\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" .,:;")
            if 0 < len(candidate.split()) <= 8:
                variants.append(candidate)
    return variants


def load_reader_probe(args: argparse.Namespace) -> FrozenReaderLoglikProbe | None:
    if bool(args.no_reader_v1):
        return None
    return FrozenReaderLoglikProbe(
        model_name_or_path=str(args.reader_model),
        cache_path=Path(args.cache_dir) / "caps_day1_5_loglik_cache.jsonl",
        max_input_tokens=int(args.max_input_tokens),
        max_target_tokens=int(args.max_target_tokens),
        device=str(args.device),
    )


def candidate_rows_for_record(
    record: dict[str, Any],
    dense_record: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
    llm_cache: LlmCandidateCache,
    probe: FrozenReaderLoglikProbe | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qid = row_qid(record)
    docs = dedup_union_records([row for row in [record, dense_record] if row is not None], pool_k=int(args.pool_k), cap=int(args.llm_docs))
    cached = llm_cache.get(qid)
    if cached is None:
        prompt = build_llm_prompt(str(record.get("question") or ""), docs, max_doc_chars=int(args.doc_chars))
        start = time.perf_counter()
        raw = call_openai_compatible(
            str(args.llm_endpoint),
            str(args.llm_model),
            prompt,
            timeout=int(args.timeout),
            max_tokens=int(args.max_tokens),
        )
        elapsed = time.perf_counter() - start
        cached = {
            "qid": qid,
            "raw": raw,
            "parsed": parse_llm_candidates(raw),
            "elapsed_seconds": round(elapsed, 4),
        }
        llm_cache.set(qid, cached)
    else:
        cached = {**cached, "parsed": parse_llm_candidates(str(cached.get("raw") or ""))}
    candidates = [{"text": text, "source": "llm_candidate_list", "score": 80.0 - idx} for idx, text in enumerate(cached.get("parsed") or [])]
    if bool(args.include_v1):
        candidates.extend(
            generate_answer_candidates(
                record,
                dense_record,
                args=args,
                probe=probe,
                answer_cache=None if probe is None else __import__("caps_day1_gates").ReaderAnswerCache(Path(args.cache_dir) / "caps_day1_answer_cache.jsonl"),
            )
        )
    if bool(args.include_string_extract):
        union_docs = dedup_union_records([row for row in [record, dense_record] if row is not None], pool_k=int(args.pool_k), cap=int(args.union_cap))
        candidates.extend(extract_heuristic_answers(record, union_docs, max_per_source=int(args.max_heuristic_answers)))
    clustered = cluster_answer_candidates(candidates, cap=int(args.answer_cap))
    return clustered, cached


def summarize_recall(rows: Sequence[dict[str, Any]], *, recall5_gate: float, recall10_gate: float) -> dict[str, Any]:
    denom = max(1, len(rows))
    summary = {
        "rows": len(rows),
        "recall_at_5": round(sum(float(row["recall_at_5"]) for row in rows) / denom, 6),
        "recall_at_10": round(sum(float(row["recall_at_10"]) for row in rows) / denom, 6),
        "recall_at_20": round(sum(float(row["recall_at_20"]) for row in rows) / denom, 6),
        "avg_candidate_count": round(sum(float(row["candidate_count"]) for row in rows) / denom, 4),
    }
    if summary["recall_at_5"] >= float(recall5_gate):
        decision = "PROCEED_GATE2_WITH_TOP5"
    elif summary["recall_at_10"] >= float(recall10_gate):
        decision = "REVIEW_PROCEED_GATE2_WITH_TOP10"
    else:
        decision = "STOP_CANDIDATE_V2_RECALL_FAIL"
    summary["decision"] = decision
    summary["passed_top5"] = bool(summary["recall_at_5"] >= float(recall5_gate))
    summary["passed_top10"] = bool(summary["recall_at_10"] >= float(recall10_gate))
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if not hasattr(args, "no_reader_answers"):
        setattr(args, "no_reader_answers", bool(args.no_reader_v1))
    health = qwen_health(str(args.llm_endpoint))
    if not health.get("available"):
        payload = {"status": "llm_unavailable", "health": health, "summary": {"decision": "STOP_LLM_UNAVAILABLE"}}
        write_outputs(payload, [], args)
        return payload

    prop_rows = load_jsonl(args.proprag_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))[int(args.dev_start) : int(args.dev_end)]
    dense_by_qid = {row_qid(row): row for row in load_jsonl(args.dense_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))}
    cache = LlmCandidateCache(Path(args.cache_dir) / "caps_day1_5_llm_candidates.jsonl")
    probe = load_reader_probe(args)
    rows = []
    for record in prop_rows:
        qid = row_qid(record)
        candidates, cached = candidate_rows_for_record(record, dense_by_qid.get(qid), args=args, llm_cache=cache, probe=probe)
        golds = gold_answers(record)
        rows.append(
            {
                "qid": qid,
                "gold_answers": golds,
                "candidate_count": len(candidates),
                "recall_at_5": answer_recall_at(candidates, golds, 5),
                "recall_at_10": answer_recall_at(candidates, golds, 10),
                "recall_at_20": answer_recall_at(candidates, golds, 20),
                "llm_candidates": cached.get("parsed") or [],
                "top_candidates": candidates[:20],
            }
        )
    summary = summarize_recall(rows, recall5_gate=float(args.recall5_gate), recall10_gate=float(args.recall10_gate))
    payload = {
        "status": "completed",
        "health": health,
        "config": {
            "llm_endpoint": str(args.llm_endpoint),
            "llm_model": str(args.llm_model),
            "include_v1": bool(args.include_v1),
            "include_string_extract": bool(args.include_string_extract),
            "answer_cap": int(args.answer_cap),
        },
        "summary": summary,
    }
    write_outputs(payload, rows, args)
    return payload


def write_outputs(payload: dict[str, Any], rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    write_json({"summary": payload.get("summary", {}), "payload": payload, "rows": list(rows)}, out_dir / "caps_day1_5_candidate_v2.json")
    write_jsonl(rows, out_dir / "caps_day1_5_candidate_v2.rows.jsonl")
    summary = payload.get("summary", {})
    lines = [
        "# CAPS Day-1.5 Candidate Generator v2",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Decision: `{summary.get('decision')}`",
        f"- LLM endpoint: `{args.llm_endpoint}`",
        f"- LLM model: `{args.llm_model}`",
        f"- Include v1 candidates: `{args.include_v1}`",
        f"- Include string extraction: `{args.include_string_extract}`",
        f"- Rows: `{summary.get('rows', 0)}`",
        f"- Recall@5: `{summary.get('recall_at_5', 0.0)}`",
        f"- Recall@10: `{summary.get('recall_at_10', 0.0)}`",
        f"- Recall@20: `{summary.get('recall_at_20', 0.0)}`",
        f"- Avg candidate count: `{summary.get('avg_candidate_count', 0.0)}`",
        "",
        "## Gates",
        "",
        f"- Top-5 success: `Recall@5 >= {args.recall5_gate}`",
        f"- Top-10 review: `Recall@10 >= {args.recall10_gate}`",
    ]
    if payload.get("health"):
        lines.extend(["", f"Health: `{payload['health']}`"])
    write_markdown(lines, out_dir / "caps_day1_5_candidate_v2.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proprag_cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--dense_cache_jsonl", default="data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl")
    parser.add_argument("--llm_endpoint", default="http://localhost:8043/v1")
    parser.add_argument("--llm_model", default="qwen3-8b-train")
    parser.add_argument("--reader_model", default="data/dpathrag/models/flan_t5_base_gold_k5_5k")
    parser.add_argument("--dev_start", type=int, default=0)
    parser.add_argument("--dev_end", type=int, default=200)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--union_cap", type=int, default=60)
    parser.add_argument("--llm_docs", type=int, default=30)
    parser.add_argument("--doc_chars", type=int, default=500)
    parser.add_argument("--answer_cap", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--recall5_gate", type=float, default=0.85)
    parser.add_argument("--recall10_gate", type=float, default=0.90)
    parser.add_argument("--include_v1", action="store_true")
    parser.add_argument("--include_string_extract", action="store_true")
    parser.add_argument("--no_reader_v1", action="store_true")
    parser.add_argument("--max_heuristic_answers", type=int, default=240)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--single_reader_docs", type=int, default=12)
    parser.add_argument("--pair_reader_pairs", type=int, default=5)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_target_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="reports/caps")
    parser.add_argument("--cache_dir", default="data/dpathrag/cache/caps")
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"CAPS Day1.5 decision: {payload.get('summary', {}).get('decision')}")


if __name__ == "__main__":
    main()
