#!/usr/bin/env python3
"""Run PathRAG under the local QA protocol.

This adapter keeps the upstream PathRAG implementation in /mnt/nvme/code/PathRAG
unchanged. It builds a PathRAG index over our local benchmark corpus, retrieves
reader contexts with ``only_need_context=True``, maps PathRAG source chunks back
to corpus passage ids, and writes the saved-docs reader-input format consumed by
``evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py``.

Generative LLM usage is recorded as JSONL so the offline/online token cost can be
summarized without relying on PathRAG's cache files.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from openai import AsyncOpenAI, BadRequestError
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHRAG_ROOT = Path("/mnt/nvme/code/PathRAG")
if str(PATHRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(PATHRAG_ROOT))

from PathRAG import PathRAG, QueryParam  # noqa: E402
from PathRAG.utils import wrap_embedding_func_with_attrs  # noqa: E402


DATASET_FILE_ALIASES = {
    "nq": "nq_rear",
    "natural_questions": "nq_rear",
}

DISPLAY_NAMES = {
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultiHopQA",
    "musique": "MuSiQue",
    "nq_rear": "NQ",
    "popqa": "PopQA",
}

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
LLM_CONTEXT_TOKENS = 8192
LLM_PROMPT_SAFETY_TOKENS = 512
APPROX_CHARS_PER_TOKEN = 3
TRUNCATION_MARKER = "\n\n[...truncated to fit the model context window...]\n\n"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def strip_think_blocks(value: Any) -> str:
    return THINK_BLOCK_RE.sub("", str(value or "")).strip()


def trim_text_middle(value: str, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_MARKER) + 32:
        return text[:max_chars]
    keep = max_chars - len(TRUNCATION_MARKER)
    head = max(16, keep // 2)
    tail = max(16, keep - head)
    return text[:head].rstrip() + TRUNCATION_MARKER + text[-tail:].lstrip()


def trim_messages_to_char_budget(
    messages: Sequence[Mapping[str, str]],
    *,
    max_chars: int,
) -> tuple[list[dict[str, str]], bool, int, int]:
    """Trim oversized chat messages while preserving prompt head and instructions tail."""

    trimmed = [{"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")} for m in messages]
    original_chars = sum(len(m["content"]) for m in trimmed)
    if original_chars <= max_chars:
        return trimmed, False, original_chars, original_chars

    while sum(len(m["content"]) for m in trimmed) > max_chars:
        overflow = sum(len(m["content"]) for m in trimmed) - max_chars
        candidate_indices = [i for i, m in enumerate(trimmed) if m["role"] != "system" and m["content"]]
        if not candidate_indices:
            candidate_indices = [i for i, m in enumerate(trimmed) if m["content"]]
        if not candidate_indices:
            break
        idx = max(candidate_indices, key=lambda i: len(trimmed[i]["content"]))
        current = trimmed[idx]["content"]
        target_len = max(0, len(current) - overflow)
        trimmed[idx]["content"] = trim_text_middle(current, target_len)
        if target_len == 0 and overflow > 0:
            break

    final_chars = sum(len(m["content"]) for m in trimmed)
    return trimmed, True, original_chars, final_chars


def canonical_dataset_name(dataset: str) -> str:
    raw = str(dataset).strip().lower()
    return DATASET_FILE_ALIASES.get(raw, raw)


def passage_from_corpus_row(row: Mapping[str, Any]) -> str:
    return f"{row.get('title', '')}\n{row.get('text', '')}".strip()


def load_corpus(data_root: Path, dataset: str, *, max_index_docs: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    corpus = list(read_json(data_root / f"{dataset}_corpus.json"))
    if int(max_index_docs) > 0:
        corpus = corpus[: int(max_index_docs)]
    docs = [passage_from_corpus_row(row) for row in corpus]
    return corpus, docs


def title_to_indices(corpus: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for idx, row in enumerate(corpus):
        title = str(row.get("title") or "").strip()
        if title:
            result.setdefault(title, []).append(int(idx))
    return result


def passage_to_index(corpus: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for idx, row in enumerate(corpus):
        passage = passage_from_corpus_row(row)
        if passage:
            result.setdefault(normalize_text(passage), int(idx))
    return result


def unique_ints(values: Iterable[Any]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def context_passage(item: Sequence[Any]) -> str:
    title = str(item[0]) if item else ""
    sentences = item[1] if len(item) > 1 else []
    if isinstance(sentences, list):
        body = " ".join(str(s) for s in sentences)
    else:
        body = str(sentences or "")
    return f"{title}\n{body}".strip()


def gold_from_supporting_facts(
    row: Mapping[str, Any],
    *,
    title_map: Mapping[str, Sequence[int]],
    passage_map: Mapping[str, int],
) -> list[int]:
    context = list(row.get("context", []) or [])
    local_by_title: dict[str, list[int]] = {}
    for local_idx, item in enumerate(context):
        if isinstance(item, Sequence) and item:
            local_by_title.setdefault(str(item[0]), []).append(int(local_idx))
    values: list[int] = []
    for fact in row.get("supporting_facts", []) or []:
        if not isinstance(fact, Sequence) or not fact:
            continue
        title = str(fact[0])
        for local_idx in local_by_title.get(title, []):
            passage = context_passage(context[local_idx])
            doc_idx = passage_map.get(normalize_text(passage))
            if doc_idx is not None:
                values.append(int(doc_idx))
        if not local_by_title.get(title):
            matches = title_map.get(title, [])
            if len(matches) == 1:
                values.append(int(matches[0]))
    return unique_ints(values)


def gold_from_musique(
    row: Mapping[str, Any],
    *,
    title_map: Mapping[str, Sequence[int]],
    passage_map: Mapping[str, int],
) -> list[int]:
    values: list[int] = []
    for paragraph in row.get("paragraphs", []) or []:
        if not isinstance(paragraph, Mapping) or not paragraph.get("is_supporting"):
            continue
        passage = f"{paragraph.get('title', '')}\n{paragraph.get('paragraph_text', '')}".strip()
        doc_idx = passage_map.get(normalize_text(passage))
        if doc_idx is not None:
            values.append(int(doc_idx))
            continue
        matches = title_map.get(str(paragraph.get("title") or ""), [])
        if len(matches) == 1:
            values.append(int(matches[0]))
    return unique_ints(values)


def parse_possible_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [text]


def gold_from_open_domain(
    row: Mapping[str, Any],
    *,
    key: str,
    title_map: Mapping[str, Sequence[int]],
    passage_map: Mapping[str, int],
) -> list[int]:
    values: list[int] = []
    for item in row.get(key, []) or []:
        if not isinstance(item, Mapping) or not item.get("is_supporting"):
            continue
        passage = f"{item.get('title', '')}\n{item.get('text', '')}".strip()
        doc_idx = passage_map.get(normalize_text(passage))
        if doc_idx is not None:
            values.append(int(doc_idx))
            continue
        matches = title_map.get(str(item.get("title") or ""), [])
        if len(matches) == 1:
            values.append(int(matches[0]))
    return unique_ints(values)


def build_query_rows(data_root: Path, dataset: str, corpus: Sequence[Mapping[str, Any]], *, limit: int = 0) -> list[dict[str, Any]]:
    rows = list(read_json(data_root / f"{dataset}.json"))
    if int(limit) > 0:
        rows = rows[: int(limit)]
    title_map = title_to_indices(corpus)
    passage_map = passage_to_index(corpus)
    out: list[dict[str, Any]] = []
    for query_index, row in enumerate(rows):
        if dataset in {"hotpotqa", "2wikimultihopqa"}:
            answers = [str(row.get("answer") or "")]
            gold = gold_from_supporting_facts(row, title_map=title_map, passage_map=passage_map)
        elif dataset == "musique":
            answers = [str(row.get("answer") or "")]
            answers.extend(str(alias) for alias in row.get("answer_aliases", []) or [])
            gold = gold_from_musique(row, title_map=title_map, passage_map=passage_map)
        elif dataset == "nq_rear":
            answers = [str(item) for item in row.get("reference", []) or []]
            gold = gold_from_open_domain(row, key="contexts", title_map=title_map, passage_map=passage_map)
        elif dataset == "popqa":
            answers = parse_possible_answers(row.get("possible_answers"))
            if row.get("obj") is not None:
                answers.insert(0, str(row.get("obj")))
            gold = gold_from_open_domain(row, key="paragraphs", title_map=title_map, passage_map=passage_map)
        else:
            raise ValueError(f"unsupported dataset: {dataset}")
        out.append(
            {
                "query_index": int(query_index),
                "question": str(row.get("question") or ""),
                "gold_answers": [x for x in dict.fromkeys(answers) if x],
                "gold_doc_indices": gold,
            }
        )
    return out


@dataclass
class UsageLogger:
    path: Path
    phase: str = "unknown"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def use_phase(self, phase: str):
        old = self.phase
        self.phase = str(phase)
        try:
            yield
        finally:
            self.phase = old

    def record(self, payload: Mapping[str, Any]) -> None:
        row = {"time": time.time(), "phase": self.phase, **dict(payload)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_logged_llm(*, base_url: str, model: str, api_key: str, usage_logger: UsageLogger):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def logged_complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[Mapping[str, Any]] | None = None,
        keyword_extraction: bool | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        for message in history_messages or []:
            if isinstance(message, Mapping):
                messages.append(
                    {
                        "role": str(message.get("role") or "user"),
                        "content": str(message.get("content") or ""),
                    }
                )
        user_prompt = str(prompt or "")
        if "/no_think" not in user_prompt:
            user_prompt = "/no_think\n" + user_prompt
        messages.append({"role": "user", "content": user_prompt})
        allowed_keys = {"max_tokens", "temperature", "top_p", "stop", "stream", "response_format"}
        call_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys and v is not None}
        call_kwargs.setdefault("temperature", 0)
        max_tokens = int(call_kwargs.get("max_tokens") or 512)
        prompt_budget_tokens = max(1024, LLM_CONTEXT_TOKENS - max_tokens - LLM_PROMPT_SAFETY_TOKENS)
        prompt_budget_chars = prompt_budget_tokens * APPROX_CHARS_PER_TOKEN
        response = None
        started = time.time()
        messages_for_call: list[dict[str, str]] = []
        truncated = False
        original_prompt_chars = sum(len(m.get("content", "")) for m in messages)
        final_prompt_chars = original_prompt_chars
        retry_count = 0
        for attempt, budget_scale in enumerate((1.0, 0.75, 0.55, 0.4)):
            retry_count = attempt
            budget = max(4096, int(prompt_budget_chars * budget_scale))
            messages_for_call, truncated, original_prompt_chars, final_prompt_chars = trim_messages_to_char_budget(
                messages,
                max_chars=budget,
            )
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages_for_call,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    **call_kwargs,
                )
                break
            except BadRequestError as exc:
                message = str(exc)
                if "maximum context length" not in message and "context length" not in message:
                    raise
                usage_logger.record(
                    {
                        "kind": "chat_completion_error",
                        "model": model,
                        "keyword_extraction": bool(keyword_extraction),
                        "error_type": "context_length",
                        "attempt": attempt + 1,
                        "prompt_chars": final_prompt_chars,
                        "original_prompt_chars": original_prompt_chars,
                        "truncated": bool(truncated),
                        "elapsed_s": time.time() - started,
                    }
                )
                if attempt == 3:
                    raise
        if response is None:
            raise RuntimeError("LLM completion failed without a response")
        elapsed = time.time() - started
        content = strip_think_blocks(response.choices[0].message.content or "")
        usage = getattr(response, "usage", None)
        usage_logger.record(
            {
                "kind": "chat_completion",
                "model": model,
                "keyword_extraction": bool(keyword_extraction),
                "prompt_chars": final_prompt_chars,
                "original_prompt_chars": original_prompt_chars,
                "truncated": bool(truncated),
                "context_retry_count": int(retry_count),
                "completion_chars": len(content),
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                "elapsed_s": elapsed,
            }
        )
        return content

    return logged_complete


def make_logged_embedding(*, base_url: str, model: str, api_key: str, usage_logger: UsageLogger):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @wrap_embedding_func_with_attrs(embedding_dim=4096, max_token_size=2048)
    async def logged_embedding(texts: list[str]) -> np.ndarray:
        started = time.time()
        response = await client.embeddings.create(model=model, input=texts, encoding_format="float")
        elapsed = time.time() - started
        usage = getattr(response, "usage", None)
        usage_logger.record(
            {
                "kind": "embedding",
                "model": model,
                "batch_size": len(texts),
                "input_chars": sum(len(str(text)) for text in texts),
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": 0,
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                "elapsed_s": elapsed,
            }
        )
        return np.asarray([item.embedding for item in response.data], dtype=np.float32)

    return logged_embedding


def parse_sources_from_context(context: str) -> list[str]:
    marker = "-----Sources-----"
    if marker not in context:
        return []
    tail = context.split(marker, 1)[1]
    match = re.search(r"```csv\s*(.*?)```", tail, flags=re.DOTALL)
    if not match:
        return []
    csv_text = match.group(1).strip()
    if not csv_text:
        return []
    out: list[str] = []
    current: list[str] | None = None
    for line in csv_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current is not None:
                current.append("")
            continue
        if stripped.lower().startswith("id,"):
            continue
        entry = re.match(r"^\s*\d+\s*,\s*(.*)$", line)
        if entry:
            if current:
                text = "\n".join(current).strip()
                if text:
                    out.append(text)
            current = [entry.group(1).strip()]
        elif current is not None:
            current.append(stripped)
    if current:
        text = "\n".join(current).strip()
        if text:
            out.append(text)
    return out


def build_chunk_doc_lookup(working_dir: Path, docs: Sequence[str]) -> dict[str, int]:
    doc_norm_to_idx = {normalize_text(doc): idx for idx, doc in enumerate(docs)}
    full_docs = read_json(working_dir / "kv_store_full_docs.json") if (working_dir / "kv_store_full_docs.json").exists() else {}
    text_chunks = read_json(working_dir / "kv_store_text_chunks.json") if (working_dir / "kv_store_text_chunks.json").exists() else {}
    full_id_to_idx: dict[str, int] = {}
    for doc_id, row in (full_docs or {}).items():
        idx = doc_norm_to_idx.get(normalize_text(row.get("content") if isinstance(row, Mapping) else row))
        if idx is not None:
            full_id_to_idx[str(doc_id)] = int(idx)
    chunk_norm_to_idx: dict[str, int] = {}
    for _chunk_id, row in (text_chunks or {}).items():
        if not isinstance(row, Mapping):
            continue
        full_doc_id = str(row.get("full_doc_id") or "")
        idx = full_id_to_idx.get(full_doc_id)
        if idx is None:
            idx = doc_norm_to_idx.get(normalize_text(row.get("content") or ""))
        if idx is not None:
            chunk_norm_to_idx[normalize_text(row.get("content") or "")] = int(idx)
    return chunk_norm_to_idx


def docs_for_indices(docs: Sequence[str], indices: Sequence[int]) -> list[str]:
    out: list[str] = []
    for idx in indices:
        if 0 <= int(idx) < len(docs):
            out.append(docs[int(idx)])
    return out


def recall_at_k(gold: Sequence[int], retrieved: Sequence[int], k: int) -> float:
    gold_set = {int(x) for x in gold}
    if not gold_set:
        return 0.0
    return len(gold_set & set(int(x) for x in retrieved[:k])) / float(len(gold_set))


def summarize_retrieval(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denom = max(len(examples), 1)
    metrics: dict[str, Any] = {"count": len(examples)}
    for k in (5, 20, 100, 200):
        metrics[f"Recall@{k}"] = sum(
            recall_at_k(ex.get("gold_doc_indices", []) or [], ex.get("retrieved_doc_ids", []) or [], k)
            for ex in examples
        ) / denom
    metrics["All@5"] = sum(
        1.0
        for ex in examples
        if set(int(x) for x in ex.get("gold_doc_indices", []) or [])
        and set(int(x) for x in ex.get("gold_doc_indices", []) or []).issubset(
            set(int(x) for x in (ex.get("retrieved_doc_ids", []) or [])[:5])
        )
    ) / denom
    metrics["mean_docs"] = sum(len(ex.get("docs", []) or []) for ex in examples) / denom
    return metrics


def summarize_usage(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary: dict[str, Any] = {"path": str(path), "rows": len(rows), "by_phase": {}}
    for row in rows:
        phase = str(row.get("phase") or "unknown")
        kind = str(row.get("kind") or "unknown")
        bucket = summary["by_phase"].setdefault(
            phase,
            {
                "calls": 0,
                "chat_calls": 0,
                "embedding_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_s": 0.0,
                "by_kind": {},
            },
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(row.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(row.get("total_tokens") or 0)
        bucket["elapsed_s"] += float(row.get("elapsed_s") or 0.0)
        if kind == "embedding":
            bucket["embedding_calls"] += 1
        elif kind == "chat_completion":
            bucket["chat_calls"] += 1
        kb = bucket["by_kind"].setdefault(kind, {"calls": 0, "total_tokens": 0})
        kb["calls"] += 1
        kb["total_tokens"] += int(row.get("total_tokens") or 0)
    return summary


async def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = canonical_dataset_name(args.dataset)
    run_root = Path(args.output_root).expanduser().resolve()
    working_dir = run_root / "work" / dataset
    usage_path = run_root / "usage" / f"{dataset}_usage.jsonl"
    usage_logger = UsageLogger(usage_path)

    corpus, docs = load_corpus(Path(args.data_root), dataset, max_index_docs=int(args.max_index_docs))
    query_rows = build_query_rows(Path(args.data_root), dataset, corpus, limit=int(args.limit))

    llm_func = make_logged_llm(
        base_url=str(args.llm_base_url),
        model=str(args.llm_name),
        api_key=str(args.api_key),
        usage_logger=usage_logger,
    )
    embedding_func = make_logged_embedding(
        base_url=str(args.embedding_base_url),
        model=str(args.embedding_name),
        api_key=str(args.api_key),
        usage_logger=usage_logger,
    )

    working_dir.mkdir(parents=True, exist_ok=True)
    rag = PathRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        llm_model_name=str(args.llm_name),
        llm_model_max_async=int(args.llm_max_async),
        llm_model_kwargs={},
        embedding_func=embedding_func,
        embedding_batch_num=int(args.embedding_batch_num),
        embedding_func_max_async=int(args.embedding_max_async),
        chunk_token_size=int(args.chunk_token_size),
        chunk_overlap_token_size=int(args.chunk_overlap_token_size),
        entity_extract_max_gleaning=int(args.entity_extract_max_gleaning),
        enable_llm_cache=not bool(args.disable_llm_cache),
    )

    if not args.skip_index:
        with usage_logger.use_phase("offline_index"):
            await rag.ainsert(docs)

    chunk_lookup = build_chunk_doc_lookup(working_dir, docs)
    examples: list[dict[str, Any]] = []
    contexts_dir = run_root / "contexts" / dataset
    contexts_dir.mkdir(parents=True, exist_ok=True)
    param = QueryParam(
        mode="hybrid",
        only_need_context=True,
        top_k=int(args.pathrag_top_k),
        max_token_for_text_unit=int(args.max_token_for_text_unit),
        max_token_for_global_context=int(args.max_token_for_global_context),
        max_token_for_local_context=int(args.max_token_for_local_context),
    )
    with usage_logger.use_phase("online_retrieval"):
        for row in tqdm(query_rows, desc=f"PathRAG retrieve {dataset}", unit="q"):
            context = await rag.aquery(str(row["question"]), param=param)
            sources = parse_sources_from_context(str(context or ""))
            retrieved_ids = unique_ints(chunk_lookup.get(normalize_text(source), -1) for source in sources)
            top_ids = retrieved_ids[: int(args.qa_top_k)]
            contexts_dir.joinpath(f"{int(row['query_index']):04d}.txt").write_text(str(context or ""), encoding="utf-8")
            examples.append(
                {
                    "query_index": int(row["query_index"]),
                    "question": row["question"],
                    "gold_answers": row["gold_answers"],
                    "gold_doc_indices": row["gold_doc_indices"],
                    "retrieved_doc_ids": top_ids,
                    "docs": docs_for_indices(docs, top_ids),
                    "pathrag": {
                        "source_count": len(sources),
                        "mapped_doc_count": len(retrieved_ids),
                    },
                }
            )

    retrieval_metrics = summarize_retrieval(examples)
    payload = {
        "format": "pathrag_saved_docs_reader_input_v1",
        "dataset": dataset,
        "display_dataset": DISPLAY_NAMES.get(dataset, dataset),
        "method": "PathRAG",
        "limit": int(args.limit),
        "max_index_docs": int(args.max_index_docs),
        "config": {
            "llm_name": args.llm_name,
            "llm_base_url": args.llm_base_url,
            "llm_no_think": True,
            "embedding_name": args.embedding_name,
            "embedding_base_url": args.embedding_base_url,
            "pathrag_top_k": int(args.pathrag_top_k),
            "chunk_token_size": int(args.chunk_token_size),
            "chunk_overlap_token_size": int(args.chunk_overlap_token_size),
            "entity_extract_max_gleaning": int(args.entity_extract_max_gleaning),
        },
        "retrieval": {
            "recomputed_title_recall": retrieval_metrics,
            "mapping": "PathRAG source chunks normalized to local corpus passages",
        },
        "usage": summarize_usage(usage_path),
        "examples": examples,
    }
    output_json = run_root / "reader_inputs" / f"{dataset}_pathrag_qwen32b_no_think_top{args.qa_top_k}_limit{args.limit}_reader_input.json"
    write_json(output_json, payload)
    summary_json = run_root / "summaries" / f"{dataset}_pathrag_summary.json"
    write_json(summary_json, {k: payload[k] for k in ("dataset", "method", "limit", "config", "retrieval", "usage")})
    print(json.dumps({"dataset": dataset, "output_json": str(output_json), "summary_json": str(summary_json), "retrieval": retrieval_metrics}, sort_keys=True))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "reproduce/dataset"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "run_logs/pathrag_qwen32b_no_think_gpt4omini_full1000_20260520"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-index-docs", type=int, default=0)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--llm-base-url", default="http://localhost:8045/v1")
    parser.add_argument("--llm-name", default="qwen3-32b-judge")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1")
    parser.add_argument("--embedding-name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--llm-max-async", type=int, default=16)
    parser.add_argument("--embedding-max-async", type=int, default=1)
    parser.add_argument("--embedding-batch-num", type=int, default=4)
    parser.add_argument("--chunk-token-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=100)
    parser.add_argument("--entity-extract-max-gleaning", type=int, default=1)
    parser.add_argument("--pathrag-top-k", type=int, default=40)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--max-token-for-text-unit", type=int, default=4000)
    parser.add_argument("--max-token-for-global-context", type=int, default=3000)
    parser.add_argument("--max-token-for-local-context", type=int, default=5000)
    parser.add_argument("--disable-llm-cache", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
