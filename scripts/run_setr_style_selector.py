#!/usr/bin/env python3
"""Run a SetR-style IRI selector over exported pool records."""

from __future__ import annotations

import argparse
from concurrent import futures
import json
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any
from urllib.parse import urlparse

SETR_SELECTION_SYS_PROMPT = (
    "You are RankLLM, an intelligent assistant that can rank and select passages based on their relevancy to the query."
)

SETR_SELECTION_IRI_PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Please follow the steps below:
Step 1. Please list up the information requirements to answer the query.
Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.
Step 3. Choose the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [4] [2]."""


FINAL_SELECTION_RE = re.compile(r"final\s+selection\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
BRACKET_RE = re.compile(r"\[(\d+)\]")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_setr_selection(text: str, max_position: int) -> list[int]:
    """Parse SetR 1-based bracket ids into zero-based unique pool positions."""
    value = str(text or "")
    match = FINAL_SELECTION_RE.search(value)
    parse_region = match.group(1) if match else value
    positions: list[int] = []
    seen: set[int] = set()
    for raw in BRACKET_RE.findall(parse_region):
        pos = int(raw) - 1
        if 0 <= pos < max_position and pos not in seen:
            positions.append(pos)
            seen.add(pos)
    return positions


def usage_value(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


def build_context_text(record: dict[str, Any]) -> str:
    chunks = []
    for ctx in record.get("contexts") or []:
        pos = int(ctx.get("position") or 0)
        title = str(ctx.get("title") or "").strip()
        text = str(ctx.get("text") or "").strip()
        if title:
            chunks.append(re.sub(r"\n+", " ", f"[{pos}] title: {title}\t{text}"))
        else:
            chunks.append(re.sub(r"\n+", " ", f"[{pos}] {text}"))
    return "\n\n\n".join(chunks)


def build_prompt(record: dict[str, Any], *, append_no_think: bool = True) -> str:
    contexts = list(record.get("contexts") or [])
    question = str(record.get("question") or "")
    prompt = SETR_SELECTION_IRI_PROMPT.format(
        question=question,
        context=build_context_text(record),
        num=len(contexts),
    )
    if append_no_think:
        prompt = "/no_think\n" + prompt + "\nOnly output the final selection after any brief analysis. Do not output hidden reasoning tags such as <think>."
    return prompt


def record_key(record: dict[str, Any]) -> str:
    query_idx = record.get("query_idx")
    if query_idx is not None:
        return str(query_idx)
    return str(record.get("question") or "")


def load_existing_keys(path: str | Path) -> set[str]:
    out_path = Path(path)
    if not out_path.exists():
        return set()
    keys = set()
    with out_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                keys.add(record_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return keys


def call_selector(
    client: Any,
    record: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    append_no_think: bool,
) -> dict[str, Any]:
    user_prompt = build_prompt(record, append_no_think=append_no_think)
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SETR_SELECTION_SYS_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    latency_s = time.monotonic() - start
    output_text = response.choices[0].message.content or ""
    parsed = parse_setr_selection(output_text, max_position=len(record.get("contexts") or []))
    usage = getattr(response, "usage", None)
    finish_reason = None
    if getattr(response, "choices", None):
        finish_reason = getattr(response.choices[0], "finish_reason", None)
    prompt_tokens = usage_value(usage, "prompt_tokens")
    completion_tokens = usage_value(usage, "completion_tokens")
    total_tokens = usage_value(usage, "total_tokens") or (prompt_tokens + completion_tokens)
    return {
        "query_idx": record.get("query_idx"),
        "question": record.get("question"),
        "pool_k": record.get("pool_k"),
        "raw_output": output_text,
        "selected_positions": parsed,
        "selected_1based": [pos + 1 for pos in parsed],
        "parse_success": bool(parsed),
        "prompt_mode": "selection_IRI",
        "append_no_think": append_no_think,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": str(finish_reason or ""),
        },
        "latency_s": round(float(latency_s), 4),
        "prompt_chars": len(user_prompt),
        "completion_chars": len(output_text),
    }


def build_error_row(record: dict[str, Any], *, model: str, error: BaseException, append_no_think: bool) -> dict[str, Any]:
    return {
        "query_idx": record.get("query_idx"),
        "question": record.get("question"),
        "pool_k": record.get("pool_k"),
        "raw_output": "",
        "selected_positions": [],
        "selected_1based": [],
        "parse_success": False,
        "prompt_mode": "selection_IRI",
        "append_no_think": append_no_think,
        "model": model,
        "usage": {},
        "latency_s": 0.0,
        "prompt_chars": 0,
        "completion_chars": 0,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(limit=5),
    }


def is_local_base_url(base_url: str) -> bool:
    host = urlparse(str(base_url or "")).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}


def build_openai_client(base_url: str, api_key: str) -> Any:
    from openai import OpenAI
    import httpx

    http_client = httpx.Client(trust_env=not is_local_base_url(base_url), timeout=httpx.Timeout(5 * 60, read=5 * 60))
    return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY") or "sk-")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_append_no_think", action="store_true", help="Do not add the Qwen /no_think compatibility prefix/suffix.")
    parser.add_argument("--mock_rank_order", action="store_true", help="Write deterministic rank-order selections without calling an LLM.")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    end_idx = len(rows) if args.limit <= 0 else min(len(rows), args.start_idx + args.limit)
    rows = rows[args.start_idx:end_idx]
    if args.resume:
        existing = load_existing_keys(args.output_jsonl)
        rows = [row for row in rows if record_key(row) not in existing]

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        if args.mock_rank_order:
            positions = list(range(min(5, len(row.get("contexts") or []))))
            raw_output = "### Final Selection: " + " ".join(f"[{pos + 1}]" for pos in positions)
            return {
                "query_idx": row.get("query_idx"),
                "question": row.get("question"),
                "pool_k": row.get("pool_k"),
                "raw_output": raw_output,
                "selected_positions": positions,
                "selected_1based": [pos + 1 for pos in positions],
                "parse_success": True,
                "prompt_mode": "selection_IRI_mock_rank_order",
                "append_no_think": not args.no_append_no_think,
                "model": args.model,
                "usage": {},
                "latency_s": 0.0,
                "prompt_chars": 0,
                "completion_chars": len(raw_output),
            }
        client = build_openai_client(args.llm_base_url, args.api_key)
        try:
            return call_selector(
                client,
                row,
                model=str(args.model),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                append_no_think=not bool(args.no_append_no_think),
            )
        except Exception as exc:
            return build_error_row(
                row,
                model=str(args.model),
                error=exc,
                append_no_think=not bool(args.no_append_no_think),
            )

    mode = "a" if args.resume else "w"
    with out.open(mode, encoding="utf-8") as handle:
        if int(args.num_workers) <= 1:
            for row in rows:
                handle.write(json.dumps(run_one(row), ensure_ascii=False) + "\n")
                handle.flush()
        else:
            with futures.ThreadPoolExecutor(max_workers=int(args.num_workers)) as executor:
                for output in executor.map(run_one, rows):
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    handle.flush()
    print(f"Wrote {len(rows)} SetR-style selector rows to {out}")


if __name__ == "__main__":
    main()
