#!/usr/bin/env python3
"""Build AG-STO-compatible OpenIE JSON from a raw corpus.

This is a small standalone adapter for portability. It supports two extraction
modes:

- ``llm``: OpenAI-compatible chat-completions NER + triple extraction.
- ``heuristic``: deterministic lightweight fallback for smoke tests.

The retrieval method still starts from OpenIE documents. This module only
creates the OpenIE cache when one is not already available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


NER_SYSTEM = """Your task is to extract named entities from the given paragraph.
Respond only as JSON with key "named_entities" and a list of strings.
"""

TRIPLE_SYSTEM = """Your task is to construct an RDF-style OpenIE graph from the given paragraph and named entity list.
Respond only as JSON with key "triples"; each triple must be a list of exactly three strings.

Requirements:
- Each triple should contain at least one named entity when possible.
- Prefer factual relations stated in the paragraph.
- Resolve pronouns to specific names when the paragraph makes the referent clear.
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def corpus_rows_to_passages(corpus_payload: Any) -> List[Dict[str, Any]]:
    rows = corpus_payload.get("docs", []) if isinstance(corpus_payload, Mapping) else corpus_payload
    if not isinstance(rows, list):
        raise ValueError("Corpus JSON must be a list or a mapping with a 'docs' list.")

    docs: List[Dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        idx = row.get("idx", position)
        if row.get("passage"):
            passage = str(row.get("passage") or "")
        else:
            title = normalize_space(row.get("title"))
            text = str(row.get("text") or row.get("body") or "")
            passage = f"{title}\n{text}".strip() if title else text.strip()
        if not passage:
            continue
        docs.append({"idx": idx, "passage": passage})
    return docs


def extract_json_object(text: Any) -> Dict[str, Any]:
    if isinstance(text, Mapping):
        return dict(text)
    raw = str(text or "").strip()
    if not raw:
        return {}
    for candidate in (raw, raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw and "}" in raw else ""):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def sanitize_entities(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    output: List[str] = []
    seen = set()
    for value in values:
        text = normalize_space(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def sanitize_triples(values: Any) -> List[List[str]]:
    if not isinstance(values, list):
        return []
    output: List[List[str]] = []
    seen = set()
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        triple = [normalize_space(item) for item in value]
        if not all(triple):
            continue
        key = tuple(triple)
        if key in seen:
            continue
        seen.add(key)
        output.append(triple)
    return output


def split_title_and_body(passage: str) -> Tuple[str, str]:
    lines = str(passage or "").splitlines()
    if not lines:
        return "", ""
    return lines[0].strip(), "\n".join(lines[1:]).strip()


def heuristic_openie(passage: str, *, max_entities: int = 24, max_triples: int = 32) -> Dict[str, Any]:
    title, body = split_title_and_body(passage)
    text = body or passage
    candidates = []
    if title:
        candidates.append(title)
    candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5}", text))
    candidates.extend(re.findall(r"\b\d{3,4}(?:\s*[-/]\s*\d{1,4})?\b", text))
    entities = sanitize_entities(candidates)[: max(int(max_entities), 1)]

    subject = title or (entities[0] if entities else "")
    triples: List[List[str]] = []
    if subject:
        for entity in entities:
            if entity == subject:
                continue
            triples.append([subject, "mentions", entity])
            if len(triples) >= max(int(max_triples), 1):
                break
    return {"extracted_entities": entities, "extracted_triples": triples}


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    timeout: float,
    max_tokens: int,
    disable_thinking: bool = False,
) -> str:
    endpoint = str(base_url).rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    patched_messages = [dict(message) for message in messages]
    if disable_thinking:
        for message in patched_messages:
            content = message.get("content")
            if (
                str(message.get("role", "")).lower() == "user"
                and isinstance(content, str)
                and "/no_think" not in content[:128]
            ):
                message["content"] = f"/no_think\n{content}"
        patched_messages = patched_messages

    payload = {
        "model": model,
        "messages": patched_messages,
        "temperature": 0,
        "max_tokens": max(int(max_tokens), 1),
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(float(timeout), 1.0)) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    raw = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    if disable_thinking:
        raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    return raw


def llm_openie_doc(
    *,
    passage: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    max_tokens: int,
    retries: int,
    disable_thinking: bool,
) -> Dict[str, Any]:
    ner_messages = [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user", "content": passage},
    ]
    last_error: Exception | None = None
    for attempt in range(max(int(retries), 1)):
        try:
            ner_response = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=ner_messages,
                timeout=timeout,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            entities = sanitize_entities(extract_json_object(ner_response).get("named_entities", []))
            triple_messages = [
                {"role": "system", "content": TRIPLE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Paragraph:\n"
                        f"```\n{passage}\n```\n\n"
                        + json.dumps({"named_entities": entities}, ensure_ascii=False)
                    ),
                },
            ]
            triple_response = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=triple_messages,
                timeout=timeout,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            triples = sanitize_triples(extract_json_object(triple_response).get("triples", []))
            return {"extracted_entities": entities, "extracted_triples": triples}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < max(int(retries), 1):
                time.sleep(min(2.0 * float(attempt + 1), 10.0))
    raise RuntimeError(f"OpenIE LLM extraction failed: {last_error}")


def read_api_key(*, api_key: str | None, api_key_file: str | None) -> str:
    if api_key:
        return str(api_key).strip()
    if api_key_file:
        path = Path(api_key_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return str(os.environ.get("OPENAI_API_KEY") or "").strip()


def build_openie_from_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    corpus_path = Path(args.corpus_json).resolve()
    output_path = Path(args.output_openie_json).resolve()
    corpus_docs = corpus_rows_to_passages(load_json(corpus_path))
    if int(args.limit_docs) > 0:
        corpus_docs = corpus_docs[: int(args.limit_docs)]

    mode = str(args.mode)
    api_key = read_api_key(api_key=getattr(args, "api_key", None), api_key_file=getattr(args, "api_key_file", None))
    if mode == "llm" and not api_key and not bool(args.allow_empty_api_key):
        raise ValueError("LLM mode requires --api-key-file, --api-key, OPENAI_API_KEY wrapper, or --allow-empty-api-key.")

    docs_by_position: Dict[int, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []

    if bool(getattr(args, "resume", False)) and output_path.exists():
        existing = load_json(output_path)
        existing_docs = existing.get("docs", []) if isinstance(existing, Mapping) else []
        if isinstance(existing_docs, list):
            for doc in existing_docs:
                if not isinstance(doc, Mapping):
                    continue
                position_value = doc.get("position")
                if position_value is None:
                    idx_text = str(doc.get("idx", ""))
                    if idx_text.startswith("chunk-") and idx_text[len("chunk-") :].isdigit():
                        position_value = int(idx_text[len("chunk-") :])
                try:
                    position = int(position_value)
                except (TypeError, ValueError):
                    continue
                if 0 <= position < len(corpus_docs):
                    docs_by_position[position] = dict(doc)
        existing_metadata = existing.get("metadata", {}) if isinstance(existing, Mapping) else {}
        existing_errors = existing_metadata.get("errors", []) if isinstance(existing_metadata, Mapping) else []
        if isinstance(existing_errors, list):
            errors = [dict(item) for item in existing_errors if isinstance(item, Mapping)]

    def write_payload(*, partial: bool) -> Dict[str, Any]:
        docs = [docs_by_position[position] for position in sorted(docs_by_position)]
        payload = {
            "docs": docs,
            "metadata": {
                "source_corpus_json": str(corpus_path),
                "mode": mode,
                "llm_base_url": str(args.llm_base_url) if mode == "llm" else None,
                "llm_model": str(args.llm_model) if mode == "llm" else None,
                "qwen_disable_thinking": bool(getattr(args, "qwen_disable_thinking", False)),
                "partial": bool(partial),
                "expected_doc_count": len(corpus_docs),
                "doc_count": len(docs),
                "error_count": len(errors),
                "errors": errors,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(output_path.name + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(output_path)
        return payload

    def convert_doc(position_doc: Tuple[int, Mapping[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        position, doc = position_doc
        passage = str(doc.get("passage") or "")
        if mode == "heuristic":
            extracted = heuristic_openie(
                passage,
                max_entities=max(int(args.heuristic_max_entities), 1),
                max_triples=max(int(args.heuristic_max_triples), 1),
            )
        elif mode == "llm":
            extracted = llm_openie_doc(
                passage=passage,
                base_url=str(args.llm_base_url),
                api_key=api_key,
                model=str(args.llm_model),
                timeout=max(float(args.timeout), 1.0),
                max_tokens=max(int(args.max_tokens), 1),
                retries=max(int(args.retries), 1),
                disable_thinking=bool(getattr(args, "qwen_disable_thinking", False)),
            )
        else:
            raise ValueError("mode must be 'heuristic' or 'llm'")
        return position, {
            "idx": f"chunk-{doc.get('idx', position)}",
            "position": int(position),
            "passage": passage,
            "extracted_entities": extracted["extracted_entities"],
            "extracted_triples": extracted["extracted_triples"],
        }

    pending_items = [
        (position, doc)
        for position, doc in enumerate(corpus_docs)
        if int(position) not in docs_by_position
    ]
    checkpoint_every = max(int(getattr(args, "checkpoint_every", 0) or 0), 0)
    completed_since_start = 0

    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        future_to_doc = {
            executor.submit(convert_doc, (position, doc)): (position, doc)
            for position, doc in pending_items
        }
        for future in as_completed(future_to_doc):
            position, doc = future_to_doc[future]
            try:
                result_position, result_doc = future.result()
                docs_by_position[int(result_position)] = result_doc
            except Exception as exc:
                if not bool(args.continue_on_error):
                    raise
                errors.append({"position": int(position), "idx": doc.get("idx", position), "error": str(exc)})
                docs_by_position[int(position)] = {
                    "idx": f"chunk-{doc.get('idx', position)}",
                    "position": int(position),
                    "passage": str(doc.get("passage") or ""),
                    "extracted_entities": [],
                    "extracted_triples": [],
                }
            completed_since_start += 1
            if checkpoint_every and completed_since_start % checkpoint_every == 0:
                write_payload(partial=True)

    return write_payload(partial=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build AG-STO-compatible OpenIE JSON from a raw corpus.")
    parser.add_argument("--corpus-json", required=True)
    parser.add_argument("--output-openie-json", required=True)
    parser.add_argument("--mode", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--heuristic-max-entities", type=int, default=24)
    parser.add_argument("--heuristic-max-triples", type=int, default=32)
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument(
        "--qwen-disable-thinking",
        action="store_true",
        help="Add /no_think and chat_template_kwargs.enable_thinking=false for Qwen-family local servers.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--allow-empty-api-key", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = build_openie_from_corpus(args)
    compact = {
        "doc_count": len(payload.get("docs", []) or []),
        "mode": payload.get("metadata", {}).get("mode"),
        "error_count": payload.get("metadata", {}).get("error_count"),
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {Path(args.output_openie_json).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
