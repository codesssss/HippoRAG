"""Lightweight proposition extraction helpers.

Week 1 may use Qwen extraction, but tests and smoke scripts need a deterministic
fallback.  The fallback below keeps context-rich sentence propositions instead
of reducing passages to ambiguous bare triples.
"""

from __future__ import annotations

import re
from typing import Any

from .state import PropositionNode


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MENTION_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*\b")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(str(text or "").strip()) if s.strip()]
    return sentences or ([str(text).strip()] if str(text or "").strip() else [])


def extract_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    for match in _MENTION_RE.finditer(str(text or "")):
        value = match.group(0).strip()
        if value and value not in mentions:
            mentions.append(value)
    return mentions


def heuristic_extract_propositions(
    passage: str,
    source_doc_id: str,
    max_props: int = 5,
    metadata: dict[str, Any] | None = None,
) -> list[PropositionNode]:
    props: list[PropositionNode] = []
    for idx, sentence in enumerate(split_sentences(passage)[:max_props], start=1):
        prop_id = f"{source_doc_id}::p{idx}"
        props.append(
            PropositionNode(
                prop_id=prop_id,
                text=sentence,
                source_doc_id=source_doc_id,
                source_span=sentence,
                mentions=extract_mentions(sentence),
                metadata=dict(metadata or {}),
            )
        )
    return props


def parse_extraction_response(
    response_items: list[dict[str, Any]],
    source_doc_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[PropositionNode]:
    props: list[PropositionNode] = []
    for idx, item in enumerate(response_items, start=1):
        text = str(item.get("proposition") or "").strip()
        if not text:
            continue
        span = str(item.get("source_span") or text).strip()
        mentions = [str(v) for v in (item.get("mentions") or []) if str(v).strip()]
        props.append(
            PropositionNode(
                prop_id=f"{source_doc_id}::llm{idx}",
                text=text,
                source_doc_id=source_doc_id,
                source_span=span,
                mentions=mentions,
                metadata=dict(metadata or {}),
            )
        )
    return props
