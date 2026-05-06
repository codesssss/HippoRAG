#!/usr/bin/env python3
"""Run an LLM-direct top-k selector over exported pool JSON records.

The script produces a new external-pool JSON whose first ``selection_count``
documents are the LLM-selected passages. Existing evaluation code can then read
that pool with ``--setwise_selector none`` and use the normal reader path.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib.parse import urlparse


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
FINAL_SELECTION_RE = re.compile(r"final\s+selection\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
BRACKET_RE = re.compile(r"\[(\d+)\]")
INTEGER_RE = re.compile(r"(?<![\w.-])(\d+)(?![\w.-])")
SELECT_KEYS = (
    "selected_ids",
    "selected_doc_ids",
    "selected_passage_ids",
    "selected_passages",
    "selection",
    "ids",
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def title_from_doc(doc_text: str) -> str:
    first = str(doc_text or "").split("\n", 1)[0].strip()
    return first


def body_from_doc(doc_text: str) -> str:
    value = str(doc_text or "")
    if "\n" not in value:
        return value.strip()
    return value.split("\n", 1)[1].strip()


def compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate_text(text: str, max_chars: int) -> str:
    value = compact_ws(text)
    if int(max_chars) <= 0 or len(value) <= int(max_chars):
        return value
    return value[: max(0, int(max_chars) - 1)].rstrip() + "..."


def record_cache_key(record: dict[str, Any],
                     *,
                     model: str,
                     context_mode: str,
                     snippet_chars: int,
                     selection_count: int,
                     pool_k: int) -> str:
    query_idx = record.get("query_idx")
    question = str(record.get("question") or "")
    question_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return "|".join(
        [
            str(model),
            str(context_mode),
            str(snippet_chars),
            str(selection_count),
            str(pool_k),
            str(query_idx if query_idx is not None else "na"),
            question_hash,
        ]
    )


def candidate_context_lines(record: dict[str, Any],
                            *,
                            context_mode: str,
                            pool_k: int,
                            snippet_chars: int) -> list[str]:
    docs = list(record.get("pool_docs") or [])[: int(pool_k)]
    titles = list(record.get("pool_titles") or [])
    lines: list[str] = []
    for idx, doc in enumerate(docs):
        raw_doc = str(doc or "")
        title = str(titles[idx] if idx < len(titles) else title_from_doc(raw_doc)).strip()
        if context_mode == "title":
            lines.append(f"[{idx + 1}] Title: {title}")
        elif context_mode == "snippet":
            snippet = truncate_text(body_from_doc(raw_doc), int(snippet_chars))
            lines.append(f"[{idx + 1}] Title: {title}\nSnippet: {snippet}")
        else:
            raise ValueError(f"Unsupported context_mode: {context_mode}")
    return lines


def build_direct_select_messages(record: dict[str, Any],
                                 *,
                                 context_mode: str,
                                 pool_k: int,
                                 snippet_chars: int,
                                 selection_count: int,
                                 append_no_think: bool) -> list[dict[str, str]]:
    question = str(record.get("question") or "").strip()
    lines = "\n\n".join(
        candidate_context_lines(
            record,
            context_mode=context_mode,
            pool_k=pool_k,
            snippet_chars=snippet_chars,
        )
    )
    user = f"""Question:
{question}

Candidate passages are numbered from 1 to {int(pool_k)}. Select exactly {int(selection_count)} passages that together provide the strongest evidence for answering the question. Prefer passages that directly support required facts over passages that are merely topically related. Keep the original passage ids.

Candidates:
{lines}

Return JSON only, with this schema:
{{"selected_ids": [1, 2, 3, 4, 5]}}
"""
    if append_no_think:
        user = "/no_think\n" + user
    return [
        {
            "role": "system",
            "content": (
                "You are an evidence selector for retrieval-augmented question answering. "
                "You must choose passage ids, not answer the question."
            ),
        },
        {"role": "user", "content": user},
    ]


def _flatten_json_numbers(value: Any) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        return [int(raw) for raw in INTEGER_RE.findall(value)]
    if isinstance(value, list):
        output: list[int] = []
        for item in value:
            output.extend(_flatten_json_numbers(item))
        return output
    if isinstance(value, dict):
        output: list[int] = []
        for key in SELECT_KEYS:
            if key in value:
                output.extend(_flatten_json_numbers(value[key]))
        return output
    return []


def _json_candidates(text: str) -> list[str]:
    value = str(text or "")
    candidates = [match.group(1).strip() for match in JSON_BLOCK_RE.finditer(value)]
    object_match = JSON_OBJECT_RE.search(value)
    if object_match:
        candidates.append(object_match.group(0).strip())
    return candidates


def normalize_selection_ids(raw_ids: list[int], max_position: int) -> list[int]:
    if not raw_ids:
        return []
    zero_based = any(raw == 0 for raw in raw_ids)
    positions: list[int] = []
    seen: set[int] = set()
    for raw in raw_ids:
        pos = int(raw) if zero_based else int(raw) - 1
        if 0 <= pos < int(max_position) and pos not in seen:
            positions.append(pos)
            seen.add(pos)
    return positions


def parse_direct_selection(text: str, max_position: int) -> tuple[list[int], str]:
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        positions = normalize_selection_ids(_flatten_json_numbers(payload), max_position=max_position)
        if positions:
            return positions, "json"

    value = str(text or "")
    final_match = FINAL_SELECTION_RE.search(value)
    parse_region = final_match.group(1) if final_match else value
    bracket_ids = [int(raw) for raw in BRACKET_RE.findall(parse_region)]
    positions = normalize_selection_ids(bracket_ids, max_position=max_position)
    if positions:
        return positions, "bracket"

    integer_ids = [int(raw) for raw in INTEGER_RE.findall(parse_region)]
    positions = normalize_selection_ids(integer_ids, max_position=max_position)
    if positions:
        return positions, "integer"

    return [], "none"


def complete_selection(selected_positions: list[int],
                       *,
                       pool_size: int,
                       selection_count: int) -> tuple[list[int], list[int]]:
    final_positions: list[int] = []
    seen: set[int] = set()
    for pos in selected_positions:
        if 0 <= int(pos) < int(pool_size) and int(pos) not in seen:
            final_positions.append(int(pos))
            seen.add(int(pos))
        if len(final_positions) >= int(selection_count):
            return final_positions, []

    fallback_positions: list[int] = []
    for pos in range(int(pool_size)):
        if pos not in seen:
            final_positions.append(pos)
            fallback_positions.append(pos)
            seen.add(pos)
        if len(final_positions) >= int(selection_count):
            break
    return final_positions, fallback_positions


def reorder_list(values: list[Any], order: list[int]) -> list[Any]:
    return [values[pos] for pos in order if 0 <= int(pos) < len(values)]


def reorder_record(record: dict[str, Any],
                   selection_row: dict[str, Any],
                   *,
                   pool_k: int,
                   selection_count: int,
                   context_mode: str,
                   snippet_chars: int) -> dict[str, Any]:
    output = copy.deepcopy(record)
    docs = list(record.get("pool_docs") or [])
    pool_size = min(len(docs), int(pool_k))
    selected = [int(pos) for pos in selection_row.get("selected_positions") or []]
    final_prefix, fallback_positions = complete_selection(
        selected,
        pool_size=pool_size,
        selection_count=selection_count,
    )
    tail = [pos for pos in range(pool_size) if pos not in set(final_prefix)]
    if len(docs) > pool_size:
        tail.extend(range(pool_size, len(docs)))
    order = list(final_prefix) + tail

    for key in ("pool_docs", "pool_titles", "pool_doc_scores", "pool_doc_ids"):
        values = list(record.get(key) or [])
        if values:
            output[key] = reorder_list(values, order)

    output["llm_direct_select"] = {
        "context_mode": str(context_mode),
        "snippet_chars": int(snippet_chars),
        "selection_count": int(selection_count),
        "parse_success": bool(selection_row.get("parse_success", False)),
        "parse_method": str(selection_row.get("parse_method", "none")),
        "raw_selected_positions": selected,
        "raw_selected_1based": [pos + 1 for pos in selected],
        "final_prefix_positions": list(final_prefix),
        "final_prefix_1based": [pos + 1 for pos in final_prefix],
        "fallback_positions": list(fallback_positions),
        "fallback_1based": [pos + 1 for pos in fallback_positions],
        "raw_output": str(selection_row.get("raw_output", "")),
        "error": selection_row.get("error"),
        "usage": dict(selection_row.get("usage") or {}),
        "latency_s": float(selection_row.get("latency_s", 0.0) or 0.0),
        "model": str(selection_row.get("model", "")),
    }
    return output


def is_local_base_url(base_url: str) -> bool:
    host = urlparse(str(base_url or "")).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}


def build_openai_client(base_url: str, api_key: str) -> Any:
    from openai import OpenAI
    import httpx

    http_client = httpx.Client(
        trust_env=not is_local_base_url(base_url),
        timeout=httpx.Timeout(5 * 60, read=5 * 60),
    )
    return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=2)


def call_llm_selector(client: Any,
                      record: dict[str, Any],
                      *,
                      model: str,
                      context_mode: str,
                      pool_k: int,
                      snippet_chars: int,
                      selection_count: int,
                      max_tokens: int,
                      temperature: float,
                      append_no_think: bool) -> dict[str, Any]:
    messages = build_direct_select_messages(
        record,
        context_mode=context_mode,
        pool_k=pool_k,
        snippet_chars=snippet_chars,
        selection_count=selection_count,
        append_no_think=append_no_think,
    )
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        stream=False,
    )
    latency_s = time.monotonic() - start
    output_text = response.choices[0].message.content or ""
    selected_positions, parse_method = parse_direct_selection(
        output_text,
        max_position=min(len(record.get("pool_docs") or []), int(pool_k)),
    )
    usage = getattr(response, "usage", None)
    finish_reason = None
    if getattr(response, "choices", None):
        finish_reason = getattr(response.choices[0], "finish_reason", None)
    return {
        "query_idx": record.get("query_idx"),
        "question": record.get("question"),
        "model": model,
        "raw_output": output_text,
        "selected_positions": selected_positions,
        "selected_1based": [pos + 1 for pos in selected_positions],
        "parse_success": bool(selected_positions),
        "parse_method": parse_method,
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "finish_reason": str(finish_reason or ""),
        },
        "latency_s": round(float(latency_s), 4),
    }


def build_error_row(record: dict[str, Any], *, model: str, error: BaseException) -> dict[str, Any]:
    return {
        "query_idx": record.get("query_idx"),
        "question": record.get("question"),
        "model": model,
        "raw_output": "",
        "selected_positions": [],
        "selected_1based": [],
        "parse_success": False,
        "parse_method": "error",
        "usage": {},
        "latency_s": 0.0,
        "error": f"{type(error).__name__}: {error}",
    }


def load_cache_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("cache_key") or "")
            if key:
                rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pool_json", required=True)
    parser.add_argument("--output_pool_json", required=True)
    parser.add_argument("--cache_jsonl", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--selection_count", type=int, default=5)
    parser.add_argument("--context_mode", choices=["title", "snippet"], default="title")
    parser.add_argument("--snippet_chars", type=int, default=320)
    parser.add_argument("--llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--model", default="qwen3-8b-train")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY") or "sk-")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_append_no_think", action="store_true")
    parser.add_argument("--mock_rank_order", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.input_pool_json)
    records = list(payload.get("records") or [])
    if int(args.limit) > 0:
        records = records[: int(args.limit)]

    cache_path = Path(args.cache_jsonl) if args.cache_jsonl else Path(str(args.output_pool_json) + ".cache.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached_rows = load_cache_rows(cache_path) if bool(args.resume) else {}
    cache_lock = threading.Lock()

    def run_one(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        key = record_cache_key(
            record,
            model=str(args.model),
            context_mode=str(args.context_mode),
            snippet_chars=int(args.snippet_chars),
            selection_count=int(args.selection_count),
            pool_k=int(args.pool_k),
        )
        if key in cached_rows:
            return key, dict(cached_rows[key])
        if bool(args.mock_rank_order):
            positions = list(range(min(int(args.selection_count), len(record.get("pool_docs") or []), int(args.pool_k))))
            row = {
                "query_idx": record.get("query_idx"),
                "question": record.get("question"),
                "model": str(args.model),
                "raw_output": json.dumps({"selected_ids": [pos + 1 for pos in positions]}),
                "selected_positions": positions,
                "selected_1based": [pos + 1 for pos in positions],
                "parse_success": True,
                "parse_method": "mock_rank_order",
                "usage": {},
                "latency_s": 0.0,
            }
        else:
            client = build_openai_client(str(args.llm_base_url), str(args.api_key))
            try:
                row = call_llm_selector(
                    client,
                    record,
                    model=str(args.model),
                    context_mode=str(args.context_mode),
                    pool_k=int(args.pool_k),
                    snippet_chars=int(args.snippet_chars),
                    selection_count=int(args.selection_count),
                    max_tokens=int(args.max_tokens),
                    temperature=float(args.temperature),
                    append_no_think=not bool(args.no_append_no_think),
                )
            except Exception as exc:
                row = build_error_row(record, model=str(args.model), error=exc)
        row["cache_key"] = key
        row["context_mode"] = str(args.context_mode)
        row["snippet_chars"] = int(args.snippet_chars)
        row["selection_count"] = int(args.selection_count)
        with cache_lock:
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return key, row

    selection_by_key: dict[str, dict[str, Any]] = dict(cached_rows)
    if int(args.num_workers) <= 1:
        for idx, record in enumerate(records, start=1):
            key, row = run_one(record)
            selection_by_key[key] = row
            print(
                f"[{idx}/{len(records)}] q={record.get('query_idx')} "
                f"parse={row.get('parse_success')} method={row.get('parse_method')} "
                f"selected={row.get('selected_1based')}",
                flush=True,
            )
    else:
        with futures.ThreadPoolExecutor(max_workers=int(args.num_workers)) as executor:
            future_to_record = {executor.submit(run_one, record): record for record in records}
            for idx, future in enumerate(futures.as_completed(future_to_record), start=1):
                record = future_to_record[future]
                key, row = future.result()
                selection_by_key[key] = row
                print(
                    f"[{idx}/{len(records)}] q={record.get('query_idx')} "
                    f"parse={row.get('parse_success')} method={row.get('parse_method')} "
                    f"selected={row.get('selected_1based')}",
                    flush=True,
                )

    output_records: list[dict[str, Any]] = []
    parse_success_count = 0
    fallback_total = 0
    for record in records:
        key = record_cache_key(
            record,
            model=str(args.model),
            context_mode=str(args.context_mode),
            snippet_chars=int(args.snippet_chars),
            selection_count=int(args.selection_count),
            pool_k=int(args.pool_k),
        )
        selection_row = selection_by_key.get(key) or {}
        parse_success_count += int(bool(selection_row.get("parse_success", False)))
        reordered = reorder_record(
            record,
            selection_row,
            pool_k=int(args.pool_k),
            selection_count=int(args.selection_count),
            context_mode=str(args.context_mode),
            snippet_chars=int(args.snippet_chars),
        )
        fallback_total += len((reordered.get("llm_direct_select") or {}).get("fallback_positions") or [])
        output_records.append(reordered)

    output_payload = copy.deepcopy(payload)
    output_payload["records"] = output_records
    output_payload["limit"] = len(output_records)
    output_payload["source"] = "llm_direct_select"
    output_payload["llm_direct_select"] = {
        "input_pool_json": str(args.input_pool_json),
        "cache_jsonl": str(cache_path),
        "context_mode": str(args.context_mode),
        "snippet_chars": int(args.snippet_chars),
        "selection_count": int(args.selection_count),
        "pool_k": int(args.pool_k),
        "model": str(args.model),
        "llm_base_url": str(args.llm_base_url),
        "parse_success_count": int(parse_success_count),
        "record_count": int(len(output_records)),
        "fallback_total": int(fallback_total),
    }
    write_json(args.output_pool_json, output_payload)
    print(
        f"Wrote {len(output_records)} selected pool records to {args.output_pool_json}; "
        f"parse_success={parse_success_count}/{len(output_records)} fallback_total={fallback_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
