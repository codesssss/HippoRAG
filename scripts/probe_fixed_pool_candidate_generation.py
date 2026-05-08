#!/usr/bin/env python3
"""Probe fixed-pool candidate generation on MuSiQue missing-support cases.

This is a strict fixed-pool diagnostic.  It never retrieves new documents from
the corpus.  Every policy only reranks or selects from the original PropRAG
pool100 for each query.

The target slice is the MuSiQue ``gold_source_only_not_rank_or_dbec`` bucket
from the repair-candidate audit.  These gold titles are already present in the
source pool, but current rank-fill and DBEC candidate generation do not surface
them into the final evidence.  The probe asks whether stronger query
reformulation, demand-level pseudo evidence, or listwise LLM pool selection can
move these source-visible gold titles into top-k without expanding the pool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import (  # noqa: E402
    DBEC_SELECTIVE_DIR,
    SOURCE_POOL_DIR,
    normalize_title,
    read_json,
    safe_float,
    safe_int,
)
from probe_chain_walking_binding import (  # noqa: E402
    CANDIDATE_AUDIT_DIR,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    JsonlCache,
    MUSIQUE_DATASET,
    MUSIQUE_LABEL,
    TARGET_BUCKET,
    chat_completion_raw,
    grouped_missing_gold,
    parse_json_payload,
    probe_llm,
    selector_trace_from_query_trace,
    strip_think_blocks,
    target_missing_gold_rows,
    write_csv,
    write_json,
    write_jsonl,
)


REPORT_DIR = Path("reports/fixed_pool_candidate_generation_probe_primary_20260508")
SOURCE_POOL_PATH = SOURCE_POOL_DIR / "musique_pool100.json"
DBEC_TRACE_PATH = DBEC_SELECTIVE_DIR / "musique_proprag_wiki_title_daec_selective_titleuniq_full1000.json"
PROMPT_VERSION = "fixed_pool_candidate_generation_v1_no_think"

POLICIES = (
    "source_order",
    "question_lexical",
    "demand_lexical",
    "llm_query_reform",
    "llm_demand_reform",
    "llm_demand_hyde",
    "llm_combined",
    "llm_listwise_select",
)

LLM_PROMPT_TYPES = (
    "query_reform",
    "demand_reform",
    "demand_hyde",
    "listwise_select",
)

STOP_TOKENS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}

GENERIC_TOKENS = {
    "answer",
    "area",
    "article",
    "city",
    "continent",
    "country",
    "demand",
    "document",
    "entity",
    "evidence",
    "find",
    "located",
    "location",
    "name",
    "named",
    "nation",
    "page",
    "person",
    "place",
    "policy",
    "query",
    "region",
    "state",
    "support",
    "thing",
    "title",
}


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value: Any) -> list[str]:
    return [
        token
        for token in normalize_text(value).split()
        if len(token) >= 3 and token not in STOP_TOKENS and token not in GENERIC_TOKENS
    ]


def unique_ordered(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def clean_generated_text(value: Any, *, max_chars: int = 600) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[\s\-*•\"']+", "", text)
    text = re.sub(r"[\s\"']+$", "", text)
    text = re.sub(r"\s+", " ", text)
    if not text or text.upper() in {"NONE", "N/A", "NULL"}:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def source_records() -> list[dict[str, Any]]:
    return list(read_json(SOURCE_POOL_PATH).get("records") or [])


def dbec_traces() -> list[dict[str, Any]]:
    return list(read_json(DBEC_TRACE_PATH).get("setwise_selector_query_traces") or [])


def all_requirements(selector_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(req) for req in selector_trace.get("requirements") or [] if isinstance(req, Mapping)]


def requirement_text(req: Mapping[str, Any]) -> str:
    anchors = " ".join(str(value) for value in req.get("anchor_mentions") or [])
    return f"{req.get('unit_id', '')}: {req.get('subquery', '')} {anchors}".strip()


def requirement_lines(selector_trace: Mapping[str, Any]) -> str:
    lines = []
    for req in all_requirements(selector_trace):
        deps = ",".join(str(dep) for dep in req.get("depends_on") or []) or "ROOT"
        role = str(req.get("role") or "")
        lines.append(f"- {req.get('unit_id')}: {req.get('subquery')} | depends_on={deps} | role={role}")
    return "\n".join(lines) if lines else "- none"


def deterministic_demand_texts(selector_trace: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for req in all_requirements(selector_trace):
        text = requirement_text(req)
        if text:
            texts.append(text)
    return unique_ordered(texts)


def pool_doc_list(source_record: Mapping[str, Any], *, max_doc_chars: int) -> str:
    titles = list(source_record.get("pool_titles") or [])
    docs = list(source_record.get("pool_docs") or [])
    lines = []
    for pos, title in enumerate(titles):
        doc = str(docs[pos]) if pos < len(docs) else str(title)
        snippet = re.sub(r"\s+", " ", doc).strip()[: int(max_doc_chars)]
        lines.append(f"[{pos}] Title: {title}\nSnippet: {snippet}")
    return "\n\n".join(lines)


def build_prompt(
    *,
    prompt_type: str,
    question: str,
    selector_trace: Mapping[str, Any],
    source_record: Mapping[str, Any],
    max_doc_chars: int,
) -> list[dict[str, str]]:
    demands = requirement_lines(selector_trace)
    if prompt_type == "query_reform":
        user = (
            "/no_think\n"
            "Generate search queries for finding the support documents of a multi-hop question inside an already-fixed candidate pool.\n"
            "Do not answer the question. Produce specific Wikipedia-style search queries that could match support page titles or support passages.\n"
            "Include topical page queries when a broad entity must be combined with the demand, e.g. 'capital punishment in New Zealand'.\n"
            "Return strict JSON only: {\"queries\": [{\"text\": string, \"why\": string}]}.\n"
            "Use at most 8 queries.\n\n"
            f"Question: {question}\n\n"
            f"Demand graph:\n{demands}\n"
        )
    elif prompt_type == "demand_reform":
        user = (
            "/no_think\n"
            "For each demand in this multi-hop question, write 1-2 search queries that would retrieve the evidence document for that demand from a fixed candidate pool.\n"
            "Use concrete entity/topic terms when possible. Do not solve the final QA task. Return strict JSON only:\n"
            "{\"demand_queries\": [{\"demand_id\": string, \"query\": string, \"why\": string}]}.\n\n"
            f"Question: {question}\n\n"
            f"Demand graph:\n{demands}\n"
        )
    elif prompt_type == "demand_hyde":
        user = (
            "/no_think\n"
            "For each demand in this multi-hop question, write a short hypothetical evidence passage that would appear in the Wikipedia page needed for that demand.\n"
            "These snippets will be used only to match documents inside a fixed pool. Keep snippets factual in style, title/entity rich, and concise.\n"
            "Do not include chain-of-thought. Return strict JSON only:\n"
            "{\"evidence_snippets\": [{\"demand_id\": string, \"text\": string}]}.\n\n"
            f"Question: {question}\n\n"
            f"Demand graph:\n{demands}\n"
        )
    elif prompt_type == "listwise_select":
        user = (
            "/no_think\n"
            "Select likely support documents from the fixed candidate pool below. You may only choose document ids that appear in the pool; do not retrieve or invent documents.\n"
            "Prefer documents that jointly cover the demand graph, including intermediate bridge evidence and downstream evidence.\n"
            "Return strict JSON only: {\"doc_ids\": [integer], \"why\": [{\"doc_id\": integer, \"reason\": string}]}.\n"
            "Choose at most 15 doc ids, ordered by usefulness.\n\n"
            f"Question: {question}\n\n"
            f"Demand graph:\n{demands}\n\n"
            f"Fixed pool100:\n{pool_doc_list(source_record, max_doc_chars=max_doc_chars)}\n"
        )
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    return [
        {"role": "system", "content": "You are a retrieval assistant. Output JSON only."},
        {"role": "user", "content": user},
    ]


def prompt_has_no_think(messages: Sequence[Mapping[str, str]]) -> bool:
    return any(str(message.get("content") or "").lstrip().startswith("/no_think") for message in messages)


def extract_texts_from_raw(raw: Any, *, prompt_type: str) -> list[str]:
    parsed = parse_json_payload(raw)
    texts: list[str] = []

    def add(value: Any, *, max_chars: int = 600) -> None:
        text = clean_generated_text(value, max_chars=max_chars)
        if text:
            texts.append(text)

    if isinstance(parsed, Mapping):
        if prompt_type == "query_reform":
            items = parsed.get("queries") or parsed.get("search_queries") or []
            for item in items:
                add(item.get("text") or item.get("query") if isinstance(item, Mapping) else item, max_chars=240)
        elif prompt_type == "demand_reform":
            items = parsed.get("demand_queries") or parsed.get("queries") or []
            for item in items:
                add(item.get("query") or item.get("text") if isinstance(item, Mapping) else item, max_chars=260)
        elif prompt_type == "demand_hyde":
            items = parsed.get("evidence_snippets") or parsed.get("snippets") or parsed.get("passages") or []
            for item in items:
                add(item.get("text") or item.get("passage") if isinstance(item, Mapping) else item, max_chars=600)
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, Mapping):
                add(item.get("query") or item.get("text") or item.get("passage"), max_chars=600)
            else:
                add(item, max_chars=600)

    if not texts:
        for line in strip_think_blocks(raw).splitlines():
            add(line, max_chars=300)
            if len(texts) >= 8:
                break

    return unique_ordered(texts)[:12]


def extract_doc_ids_from_raw(raw: Any, *, pool_size: int) -> list[int]:
    parsed = parse_json_payload(raw)
    raw_ids: list[Any] = []
    if isinstance(parsed, Mapping):
        raw_ids = list(parsed.get("doc_ids") or parsed.get("documents") or parsed.get("ids") or [])
    elif isinstance(parsed, list):
        raw_ids = list(parsed)
    else:
        raw_ids = re.findall(r"\b\d{1,3}\b", strip_think_blocks(raw))

    output: list[int] = []
    seen: set[int] = set()
    for item in raw_ids:
        if isinstance(item, Mapping):
            item = item.get("doc_id") or item.get("id") or item.get("position")
        pos = safe_int(item, default=-1)
        if 0 <= pos < int(pool_size) and pos not in seen:
            seen.add(pos)
            output.append(pos)
        if len(output) >= 20:
            break
    return output


class PoolTextScorer:
    def __init__(self, pool_titles: Sequence[Any], pool_docs: Sequence[Any]) -> None:
        self.pool_titles = [str(title or "") for title in pool_titles]
        self.pool_docs = [str(doc or "") for doc in pool_docs]
        self.title_tokens = [set(content_tokens(title)) for title in self.pool_titles]
        self.doc_tokens = [Counter(content_tokens(f"{title} {doc}")) for title, doc in zip(self.pool_titles, self.pool_docs)]
        df: Counter[str] = Counter()
        for counter in self.doc_tokens:
            for token in counter:
                df[token] += 1
        total = max(1, len(self.pool_titles))
        self.idf = {token: math.log((total + 1) / (count + 1)) + 1.0 for token, count in df.items()}

    def score_text_for_doc(self, text: Any, pos: int) -> tuple[float, str]:
        qtokens = content_tokens(text)
        if not qtokens or not (0 <= pos < len(self.pool_titles)):
            return 0.0, ""
        qset = set(qtokens)
        title_set = self.title_tokens[pos]
        doc_counter = self.doc_tokens[pos]
        doc_set = set(doc_counter)
        overlap_title = qset & title_set
        overlap_doc = qset & doc_set

        score = 0.0
        for token in qset:
            idf = self.idf.get(token, 1.0)
            if token in title_set:
                score += 3.5 * idf
            if token in doc_counter:
                score += (1.0 + math.log(1.0 + doc_counter[token])) * idf

        title_recall = len(overlap_title) / len(title_set) if title_set else 0.0
        query_title_recall = len(overlap_title) / len(qset)
        query_doc_recall = len(overlap_doc) / len(qset)
        score += 40.0 * title_recall + 30.0 * query_title_recall + 12.0 * query_doc_recall

        title_norm = normalize_text(self.pool_titles[pos])
        text_norm = normalize_text(text)
        if title_norm and text_norm:
            if title_norm == text_norm:
                score += 200.0
            elif len(text_norm) >= 5 and text_norm in title_norm:
                score += 90.0
            elif len(title_norm) >= 5 and title_norm in text_norm:
                score += 55.0

        reason = ""
        if title_norm == text_norm:
            reason = "title_exact"
        elif overlap_title:
            reason = "title_token_overlap"
        elif overlap_doc:
            reason = "body_token_overlap"
        return score, reason

    def score_texts_for_doc(self, texts: Sequence[Any], pos: int) -> dict[str, Any]:
        scores: list[tuple[float, str, str]] = []
        for text in texts:
            score, reason = self.score_text_for_doc(text, pos)
            if score > 0:
                scores.append((score, str(text), reason))
        if not scores:
            return {"score": 0.0, "best_text": "", "reason": ""}
        scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_text, reason = scores[0]
        combined = best_score + 0.12 * sum(score for score, _text, _reason in scores[1:])
        return {"score": combined, "best_text": best_text, "reason": reason}


def rank_pool_by_texts(
    *,
    pool_titles: Sequence[Any],
    pool_docs: Sequence[Any],
    texts: Sequence[Any],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    scorer = PoolTextScorer(pool_titles, pool_docs)
    best_by_position = {pos: scorer.score_texts_for_doc(texts, pos) for pos in range(len(pool_titles))}
    order = sorted(range(len(pool_titles)), key=lambda pos: (-safe_float(best_by_position[pos].get("score")), pos))
    return order, best_by_position


def fill_source_order(prefix: Sequence[int], *, pool_size: int) -> list[int]:
    seen = {int(pos) for pos in prefix}
    return [int(pos) for pos in prefix] + [pos for pos in range(pool_size) if pos not in seen]


def rank_of_title(title: Any, order: Sequence[int], pool_titles: Sequence[Any]) -> int | None:
    norm = normalize_title(title)
    if not norm:
        return None
    for rank, pos in enumerate(order):
        if 0 <= int(pos) < len(pool_titles) and normalize_title(pool_titles[int(pos)]) == norm:
            return rank
    return None


def mean_value(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [safe_float(row.get(field)) for row in rows if str(row.get(field, "")) != ""]
    return sum(values) / len(values) if values else 0.0


def safe_rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def split_base_urls(values: Sequence[str]) -> list[str]:
    urls: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                urls.append(part.rstrip("/"))
    return unique_ordered(urls) or [DEFAULT_LLM_BASE_URL]


def build_prompt_tasks(
    *,
    query_indices: Sequence[int],
    source_records_by_index: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    max_doc_chars: int,
    llm_model: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for query_index in query_indices:
        source_record = source_records_by_index[query_index]
        selector_trace = selector_trace_from_query_trace(traces[query_index])
        question = str(source_record.get("question") or selector_trace.get("query") or "")
        for prompt_type in LLM_PROMPT_TYPES:
            messages = build_prompt(
                prompt_type=prompt_type,
                question=question,
                selector_trace=selector_trace,
                source_record=source_record,
                max_doc_chars=max_doc_chars,
            )
            key = stable_hash(
                {
                    "prompt_version": PROMPT_VERSION,
                    "query_index": query_index,
                    "prompt_type": prompt_type,
                    "question": question,
                    "demands": requirement_lines(selector_trace),
                    "pool_title_hash": stable_hash(source_record.get("pool_titles") or []),
                    "model": llm_model,
                    "max_doc_chars": max_doc_chars,
                }
            )
            tasks.append(
                {
                    "key": key,
                    "query_index": query_index,
                    "prompt_type": prompt_type,
                    "messages": messages,
                }
            )
    return tasks


def run_llm_tasks(
    *,
    tasks: Sequence[Mapping[str, Any]],
    cache: JsonlCache,
    run_llm: bool,
    base_urls: Sequence[str],
    model: str,
    max_tokens: int,
    timeout: int,
    workers: int,
) -> list[dict[str, Any]]:
    lock = threading.Lock()
    rows: list[dict[str, Any]] = []
    base_urls = list(base_urls)

    def one(task_index: int, task: Mapping[str, Any]) -> dict[str, Any]:
        key = str(task["key"])
        with lock:
            raw = cache.get(key)
        cache_hit = raw is not None
        endpoint = ""
        error = ""
        if raw is None and run_llm:
            for attempt in range(max(1, len(base_urls))):
                endpoint = base_urls[(task_index + attempt) % len(base_urls)]
                try:
                    raw = chat_completion_raw(
                        base_url=endpoint,
                        model=model,
                        messages=list(task["messages"]),
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    with lock:
                        cache.set(
                            key,
                            raw,
                            {
                                "query_index": task.get("query_index"),
                                "prompt_type": task.get("prompt_type"),
                                "model": model,
                                "endpoint": endpoint,
                            },
                        )
                    break
                except Exception as exc:  # pragma: no cover - exercised by integration runs
                    error = str(exc)
                    raw = None
        if raw is None:
            raw = ""
        parsed = parse_json_payload(raw)
        return {
            "key": key,
            "query_index": safe_int(task.get("query_index")),
            "prompt_type": str(task.get("prompt_type") or ""),
            "endpoint": endpoint,
            "cache_hit": int(cache_hit),
            "run_llm": int(run_llm),
            "prompt_has_no_think": int(prompt_has_no_think(task["messages"])),
            "parse_ok": int(parsed is not None),
            "raw": raw,
            "error": error,
        }

    if not tasks:
        return rows
    if not run_llm:
        return [one(idx, task) for idx, task in enumerate(tasks)]

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(one, idx, task) for idx, task in enumerate(tasks)]
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: (safe_int(row.get("query_index")), str(row.get("prompt_type"))))


def generated_payloads(prompt_rows: Sequence[Mapping[str, Any]], *, pool_size_by_query: Mapping[int, int]) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    for row in prompt_rows:
        query_index = safe_int(row.get("query_index"))
        prompt_type = str(row.get("prompt_type") or "")
        raw = row.get("raw") or ""
        if prompt_type == "listwise_select":
            payloads[query_index]["listwise_ids"] = extract_doc_ids_from_raw(
                raw,
                pool_size=pool_size_by_query.get(query_index, 100),
            )
        else:
            payloads[query_index][prompt_type] = extract_texts_from_raw(raw, prompt_type=prompt_type)
    return {query_index: dict(value) for query_index, value in payloads.items()}


def build_policy_order(
    *,
    policy: str,
    source_record: Mapping[str, Any],
    selector_trace: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> tuple[list[int], dict[int, dict[str, Any]], list[str]]:
    pool_titles = list(source_record.get("pool_titles") or [])
    pool_docs = list(source_record.get("pool_docs") or [])
    pool_size = len(pool_titles)
    empty_best = {pos: {"score": 0.0, "best_text": "", "reason": ""} for pos in range(pool_size)}

    if policy == "source_order":
        return list(range(pool_size)), empty_best, []
    if policy == "question_lexical":
        texts = [str(source_record.get("question") or selector_trace.get("query") or "")]
    elif policy == "demand_lexical":
        texts = deterministic_demand_texts(selector_trace)
    elif policy == "llm_query_reform":
        texts = list(generated.get("query_reform") or [])
    elif policy == "llm_demand_reform":
        texts = list(generated.get("demand_reform") or [])
    elif policy == "llm_demand_hyde":
        texts = list(generated.get("demand_hyde") or [])
    elif policy == "llm_combined":
        texts = unique_ordered(
            list(generated.get("query_reform") or [])
            + list(generated.get("demand_reform") or [])
            + list(generated.get("demand_hyde") or [])
            + deterministic_demand_texts(selector_trace)
        )
    elif policy == "llm_listwise_select":
        selected = [safe_int(pos, default=-1) for pos in generated.get("listwise_ids") or []]
        selected = [pos for pos in selected if 0 <= pos < pool_size]
        order = fill_source_order(unique_ordered(selected), pool_size=pool_size)
        best = dict(empty_best)
        for rank, pos in enumerate(selected):
            best[pos] = {
                "score": float(1000 - rank),
                "best_text": f"listwise_doc_id={pos}",
                "reason": "llm_listwise_selected",
            }
        return order, best, [str(pos) for pos in selected]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    texts = [text for text in texts if content_tokens(text)]
    if not texts:
        return list(range(pool_size)), empty_best, []
    order, best = rank_pool_by_texts(pool_titles=pool_titles, pool_docs=pool_docs, texts=texts)
    return order, best, list(texts)


def build_probe_rows(
    *,
    missing_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
    prompt_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = source_records()
    traces = dbec_traces()
    grouped = grouped_missing_gold(missing_rows)
    pool_size_by_query = {
        query_index: len(records[query_index].get("pool_titles") or [])
        for query_index in grouped
    }
    generated = generated_payloads(prompt_rows, pool_size_by_query=pool_size_by_query)

    probe_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for query_index in sorted(grouped):
        source_record = records[query_index]
        selector_trace = selector_trace_from_query_trace(traces[query_index])
        pool_titles = list(source_record.get("pool_titles") or [])
        for policy in policies:
            order, best_by_position, texts = build_policy_order(
                policy=policy,
                source_record=source_record,
                selector_trace=selector_trace,
                generated=generated.get(query_index, {}),
            )
            trace_rows.append(
                {
                    "query_index": query_index,
                    "policy": policy,
                    "question": str(source_record.get("question") or selector_trace.get("query") or ""),
                    "generated_text_count": len(texts),
                    "generated_texts_json": json.dumps(texts, ensure_ascii=False),
                    "top10_positions_json": json.dumps(order[:10], ensure_ascii=False),
                    "top10_titles_json": json.dumps([pool_titles[pos] for pos in order[:10]], ensure_ascii=False),
                }
            )
            for missing_row in grouped[query_index]:
                gold_title = str(missing_row.get("gold_title") or "")
                source_rank = safe_int(missing_row.get("source_best_gold_rank"), default=999)
                policy_rank = rank_of_title(gold_title, order, pool_titles)
                if policy_rank is None:
                    policy_rank = 999
                gold_positions = [
                    pos
                    for pos, title in enumerate(pool_titles)
                    if normalize_title(title) == normalize_title(gold_title)
                ]
                gold_pos = gold_positions[0] if gold_positions else -1
                best = best_by_position.get(gold_pos, {"score": 0.0, "best_text": "", "reason": ""})
                probe_rows.append(
                    {
                        "dataset": MUSIQUE_LABEL,
                        "base_dataset": MUSIQUE_DATASET,
                        "query_index": query_index,
                        "question": str(missing_row.get("question") or source_record.get("question") or ""),
                        "gold_title": gold_title,
                        "policy": policy,
                        "source_best_gold_rank": source_rank,
                        "policy_best_gold_rank": policy_rank,
                        "rank_improvement": source_rank - policy_rank,
                        "baseline_hit_at_5": int(source_rank < 5),
                        "baseline_hit_at_10": int(source_rank < 10),
                        "baseline_hit_at_20": int(source_rank < 20),
                        "policy_hit_at_5": int(policy_rank < 5),
                        "policy_hit_at_10": int(policy_rank < 10),
                        "policy_hit_at_20": int(policy_rank < 20),
                        "new_hit_at_5": int(source_rank >= 5 and policy_rank < 5),
                        "new_hit_at_10": int(source_rank >= 10 and policy_rank < 10),
                        "new_hit_at_20": int(source_rank >= 20 and policy_rank < 20),
                        "best_score": safe_float(best.get("score")),
                        "best_generated_text": str(best.get("best_text") or ""),
                        "best_reason": str(best.get("reason") or ""),
                        "policy_top5_titles_json": json.dumps([pool_titles[pos] for pos in order[:5]], ensure_ascii=False),
                    }
                )
    return probe_rows, trace_rows


def summarize_probe_rows(rows: Sequence[Mapping[str, Any]], prompt_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy in POLICIES:
        policy_rows = [row for row in rows if str(row.get("policy")) == policy]
        if not policy_rows:
            continue
        total = len(policy_rows)
        output.append(
            {
                "policy": policy,
                "missing_gold_titles": total,
                "queries": len({safe_int(row.get("query_index")) for row in policy_rows}),
                "baseline_recall_at_5": mean_value(policy_rows, "baseline_hit_at_5"),
                "baseline_recall_at_10": mean_value(policy_rows, "baseline_hit_at_10"),
                "baseline_recall_at_20": mean_value(policy_rows, "baseline_hit_at_20"),
                "policy_recall_at_5": mean_value(policy_rows, "policy_hit_at_5"),
                "policy_recall_at_10": mean_value(policy_rows, "policy_hit_at_10"),
                "policy_recall_at_20": mean_value(policy_rows, "policy_hit_at_20"),
                "new_recall_at_5": mean_value(policy_rows, "new_hit_at_5"),
                "new_recall_at_10": mean_value(policy_rows, "new_hit_at_10"),
                "new_recall_at_20": mean_value(policy_rows, "new_hit_at_20"),
                "mean_source_rank": mean_value(policy_rows, "source_best_gold_rank"),
                "mean_policy_rank": mean_value(policy_rows, "policy_best_gold_rank"),
                "mean_rank_improvement": mean_value(policy_rows, "rank_improvement"),
                "positive_score_rate": safe_rate(sum(1 for row in policy_rows if safe_float(row.get("best_score")) > 0), total),
            }
        )

    prompt_total_by_type = Counter(str(row.get("prompt_type") or "") for row in prompt_rows)
    prompt_parse_by_type = Counter(str(row.get("prompt_type") or "") for row in prompt_rows if safe_int(row.get("parse_ok")))
    for row in output:
        policy = str(row["policy"])
        if policy.startswith("llm_"):
            if policy == "llm_query_reform":
                prompt_type = "query_reform"
            elif policy == "llm_demand_reform":
                prompt_type = "demand_reform"
            elif policy == "llm_demand_hyde":
                prompt_type = "demand_hyde"
            elif policy == "llm_listwise_select":
                prompt_type = "listwise_select"
            else:
                prompt_type = ""
            row["prompt_parse_ok_rate"] = safe_rate(prompt_parse_by_type[prompt_type], prompt_total_by_type[prompt_type]) if prompt_type else ""
        else:
            row["prompt_parse_ok_rate"] = ""
    return output


def classify_decision(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = [row for row in summary_rows if str(row.get("policy")).startswith("llm_")]
    if not rows:
        return "deterministic_only"
    best_new5 = max(safe_float(row.get("new_recall_at_5")) for row in rows)
    best_new10 = max(safe_float(row.get("new_recall_at_10")) for row in rows)
    if best_new5 >= 0.25 and best_new10 >= 0.35:
        return "strong_fixed_pool_signal"
    if best_new5 >= 0.15:
        return "mixed_fixed_pool_signal"
    return "weak_fixed_pool_signal"


def pct(value: Any) -> str:
    return f"{100.0 * safe_float(value):.1f}%"


def build_markdown(
    *,
    metadata: Mapping[str, Any],
    policy_summary: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    report_dir: Path,
) -> str:
    lines = [
        "# Fixed-Pool Candidate Generation Probe",
        "",
        "This diagnostic reranks only the original PropRAG pool100. It does not retrieve new documents from the corpus and does not call the reader.",
        "",
        "## Decision",
        "",
        f"- Decision: `{metadata.get('decision')}`",
        f"- Query-primary slice: `{metadata.get('query_primary_only')}`",
        f"- Target queries: `{metadata.get('queries')}`",
        f"- Target missing gold titles: `{metadata.get('missing_gold_titles')}`",
        f"- Run LLM: `{metadata.get('run_llm')}`",
        f"- LLM base URLs: `{metadata.get('llm_base_urls')}`",
        f"- Workers: `{metadata.get('workers')}`",
        f"- All prompts use `/no_think`: `{metadata.get('all_prompts_no_think')}`",
        "",
        "## Policy Summary",
        "",
        "| Policy | New@5 | New@10 | New@20 | Policy R@5 | Policy R@10 | mean rank improvement | positive-score | parse ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_summary:
        parse_rate = row.get("prompt_parse_ok_rate")
        lines.append(
            "| {policy} | {new5} | {new10} | {new20} | {r5} | {r10} | {imp:.1f} | {pos} | {parse_ok} |".format(
                policy=row.get("policy"),
                new5=pct(row.get("new_recall_at_5")),
                new10=pct(row.get("new_recall_at_10")),
                new20=pct(row.get("new_recall_at_20")),
                r5=pct(row.get("policy_recall_at_5")),
                r10=pct(row.get("policy_recall_at_10")),
                imp=safe_float(row.get("mean_rank_improvement")),
                pos=pct(row.get("positive_score_rate")),
                parse_ok=pct(parse_rate) if parse_rate != "" else "",
            )
        )

    best_cases = sorted(probe_rows, key=lambda row: safe_float(row.get("rank_improvement")), reverse=True)[:16]
    lines.extend(["", "## Top Rank Improvements", ""])
    for row in best_cases:
        lines.append(
            "- `{policy}` q{qid} `{gold}` rank {src} -> {rank} via `{text}` ({reason})".format(
                policy=row.get("policy"),
                qid=row.get("query_index"),
                gold=row.get("gold_title"),
                src=row.get("source_best_gold_rank"),
                rank=row.get("policy_best_gold_rank"),
                text=str(row.get("best_generated_text") or "")[:120],
                reason=row.get("best_reason"),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Strong fixed-pool signal requires an LLM policy to reach New@5 >= 25% and New@10 >= 35%.",
            "- Mixed signal means New@5 >= 15%, enough to inspect but not enough to start a new method line.",
            "- Weak signal means fixed-pool candidate generation remains difficult even under LLM reformulation or listwise pool selection.",
            "",
            "## Files",
            "",
            f"- Probe rows: `{report_dir / 'probe_rows.csv'}`",
            f"- Policy summary: `{report_dir / 'policy_summary.csv'}`",
            f"- Prompt outputs: `{report_dir / 'prompt_outputs.jsonl'}`",
            f"- Query traces: `{report_dir / 'query_traces.jsonl'}`",
            f"- Full summary: `{report_dir / 'summary.json'}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    base_urls = split_base_urls(args.llm_base_urls)

    missing_rows = target_missing_gold_rows(
        audit_dir=args.audit_dir,
        dataset_label=MUSIQUE_LABEL,
        bucket=TARGET_BUCKET,
        query_primary_only=bool(args.query_primary_only),
    )
    grouped = grouped_missing_gold(missing_rows)
    query_indices = sorted(grouped)
    if int(args.limit_queries) > 0:
        query_indices = query_indices[: int(args.limit_queries)]
        missing_rows = [row for row in missing_rows if safe_int(row.get("query_index")) in set(query_indices)]

    records = source_records()
    traces = dbec_traces()
    llm_probes = [probe_llm(base_url, str(args.llm_model), min(int(args.timeout), 15)) for base_url in base_urls]
    if bool(args.run_llm) and not any(probe.get("available") for probe in llm_probes):
        raise RuntimeError(f"No available LLM endpoints: {dict(zip(base_urls, llm_probes))}")
    active_urls = [url for url, probe in zip(base_urls, llm_probes) if probe.get("available")]
    if not active_urls:
        active_urls = base_urls

    tasks = build_prompt_tasks(
        query_indices=query_indices,
        source_records_by_index=records,
        traces=traces,
        max_doc_chars=int(args.max_doc_chars),
        llm_model=str(args.llm_model),
    )
    cache_path = Path(args.cache_path) if args.cache_path else report_dir / "llm_cache.jsonl"
    cache = JsonlCache(cache_path)
    prompt_rows = run_llm_tasks(
        tasks=tasks,
        cache=cache,
        run_llm=bool(args.run_llm),
        base_urls=active_urls,
        model=str(args.llm_model),
        max_tokens=int(args.max_tokens),
        timeout=int(args.timeout),
        workers=int(args.workers),
    )
    policies = list(POLICIES)
    if not bool(args.run_llm) and not any(row.get("raw") for row in prompt_rows):
        policies = [policy for policy in policies if not policy.startswith("llm_")]
    probe_rows, trace_rows = build_probe_rows(
        missing_rows=missing_rows,
        policies=policies,
        prompt_rows=prompt_rows,
    )
    policy_summary = summarize_probe_rows(probe_rows, prompt_rows)
    decision = classify_decision(policy_summary)
    metadata = {
        "dataset": MUSIQUE_LABEL,
        "base_dataset": MUSIQUE_DATASET,
        "target_bucket": TARGET_BUCKET,
        "query_primary_only": bool(args.query_primary_only),
        "queries": len(query_indices),
        "missing_gold_titles": len(missing_rows),
        "strict_fixed_pool": True,
        "source_pool": str(SOURCE_POOL_PATH),
        "dbec_trace": str(DBEC_TRACE_PATH),
        "audit_dir": str(args.audit_dir),
        "report_dir": str(report_dir),
        "cache_path": str(cache_path),
        "run_llm": bool(args.run_llm),
        "llm_base_urls": active_urls,
        "llm_model": str(args.llm_model),
        "llm_probes": dict(zip(base_urls, llm_probes)),
        "workers": int(args.workers),
        "max_doc_chars": int(args.max_doc_chars),
        "all_prompts_no_think": all(safe_int(row.get("prompt_has_no_think")) for row in prompt_rows) if prompt_rows else True,
        "decision": decision,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {
        "metadata": metadata,
        "policy_summary": policy_summary,
        "probe_rows": probe_rows,
    }
    write_csv(probe_rows, report_dir / "probe_rows.csv")
    write_csv(policy_summary, report_dir / "policy_summary.csv")
    write_jsonl(prompt_rows, report_dir / "prompt_outputs.jsonl")
    write_jsonl(trace_rows, report_dir / "query_traces.jsonl")
    write_json(payload, report_dir / "summary.json")
    (report_dir / "summary.md").write_text(
        build_markdown(
            metadata=metadata,
            policy_summary=policy_summary,
            probe_rows=probe_rows,
            report_dir=report_dir,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--audit_dir", type=Path, default=CANDIDATE_AUDIT_DIR)
    parser.add_argument("--query_primary_only", action="store_true")
    parser.add_argument("--limit_queries", type=int, default=0)
    parser.add_argument("--run_llm", action="store_true")
    parser.add_argument("--llm_base_urls", nargs="+", default=[DEFAULT_LLM_BASE_URL])
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--cache_path", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_doc_chars", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    run_probe(parse_args())


if __name__ == "__main__":
    main()
