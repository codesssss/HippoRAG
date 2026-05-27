#!/usr/bin/env python3
"""Build schema-light evidence units from source text and OpenIE output.

This module is intentionally only an interface adapter.  It does not define a
fixed relation inventory, does not rank documents, and does not call an LLM.
The goal is to represent source-grounded evidence even when OpenIE relation
labels are missing, generic, or slightly wrong.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from build_source_title_openie_substrate import (
    extracted_entities,
    extracted_triples,
    normalize_text,
    split_title_and_body,
    stable_id,
    title_aliases,
)


DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{3,4}\b"
    r"|\b\d{3,4}\b",
    flags=re.IGNORECASE,
)


def unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_text(text)
        if not text or not key or key in seen:
            continue
        result.append(text)
        seen.add(key)
    return result


def source_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text) if part.strip()]


def normalized_contains(text: Any, mention: Any) -> bool:
    text_tokens = normalize_text(text).split()
    mention_tokens = normalize_text(mention).split()
    if not text_tokens or not mention_tokens or len(mention_tokens) > len(text_tokens):
        return False
    width = len(mention_tokens)
    return any(text_tokens[start : start + width] == mention_tokens for start in range(0, len(text_tokens) - width + 1))


def date_mentions(text: str) -> List[str]:
    return unique_strings(match.group(0) for match in DATE_PATTERN.finditer(str(text or "")))


def explicit_mentions_for_span(
    *,
    title: str,
    span_text: str,
    entity_surfaces: Sequence[Any],
    triple_endpoints: Sequence[Any],
) -> List[str]:
    """Return explicit mention surfaces visible in the unit provenance.

    Mentions come from source-side artifacts only: document title, extracted
    entities, OpenIE endpoints, and literal dates visible in the span.  This is
    deliberately not a fixed ontology or benchmark-specific schema.
    """

    candidates: List[Any] = []
    if title:
        candidates.append(title)
    candidates.extend(entity_surfaces)
    candidates.extend(triple_endpoints)
    candidates.extend(date_mentions(span_text))

    visible_text = f"{title}\n{span_text}"
    visible: List[str] = []
    for candidate in unique_strings(candidates):
        if normalize_text(candidate) == normalize_text(title) or normalized_contains(visible_text, candidate):
            visible.append(candidate)
    return unique_strings(visible)


def build_sentence_unit(
    *,
    doc_index: int,
    chunk_id: str,
    title: str,
    sentence_index: int,
    sentence: str,
    entity_surfaces: Sequence[Any],
    triple_endpoints: Sequence[Any],
) -> Dict[str, Any]:
    mentions = explicit_mentions_for_span(
        title=title,
        span_text=sentence,
        entity_surfaces=entity_surfaces,
        triple_endpoints=triple_endpoints,
    )
    return {
        "unit_id": stable_id("minimal-evidence", [chunk_id, doc_index, "sentence", sentence_index, sentence]),
        "unit_type": "minimal_evidence_unit",
        "doc_index": int(doc_index),
        "title": title,
        "span_text": sentence,
        "mention_surfaces": mentions,
        "predicate_text": "",
        "argument_texts": [],
        "source": "sentence",
        "sentence_index": int(sentence_index),
        "openie_fact_index": None,
    }


def build_openie_unit(
    *,
    doc_index: int,
    chunk_id: str,
    title: str,
    fact_index: int,
    triple: Tuple[str, str, str],
    source_text: str,
    entity_surfaces: Sequence[Any],
) -> Dict[str, Any]:
    subject, relation, obj = triple
    mentions = explicit_mentions_for_span(
        title=title,
        span_text=source_text,
        entity_surfaces=entity_surfaces,
        triple_endpoints=[subject, obj],
    )
    return {
        "unit_id": stable_id("minimal-evidence", [chunk_id, doc_index, "openie", fact_index, subject, relation, obj]),
        "unit_type": "minimal_evidence_unit",
        "doc_index": int(doc_index),
        "title": title,
        "span_text": source_text,
        "mention_surfaces": mentions,
        "predicate_text": relation,
        "argument_texts": [subject, obj],
        "source": "openie",
        "sentence_index": None,
        "openie_fact_index": int(fact_index),
        "fact": [subject, relation, obj],
    }


def best_sentence_for_triple(sentences: Sequence[str], triple: Tuple[str, str, str]) -> str:
    if not sentences:
        return ""
    triple_tokens = set(normalize_text(" ".join(triple)).split())
    return max(
        sentences,
        key=lambda sentence: (
            len(set(normalize_text(sentence).split()) & triple_tokens),
            -len(sentence),
        ),
    )


def build_minimal_evidence_units_for_doc(
    doc: Mapping[str, Any],
    *,
    doc_index: int,
    include_sentence_units: bool = True,
    include_openie_units: bool = True,
) -> List[Dict[str, Any]]:
    chunk_id = str(doc.get("idx") or f"doc-{doc_index}")
    passage = str(doc.get("passage") or "")
    title, body = split_title_and_body(passage)
    if not body:
        body = passage
    sentences = source_sentences(body)
    if not sentences and body:
        sentences = [body]

    triples = extracted_triples(doc)
    triple_endpoints = [value for triple in triples for value in (triple[0], triple[2])]
    entities = extracted_entities(doc)
    units: List[Dict[str, Any]] = []
    if include_sentence_units:
        for sentence_index, sentence in enumerate(sentences):
            units.append(
                build_sentence_unit(
                    doc_index=doc_index,
                    chunk_id=chunk_id,
                    title=title,
                    sentence_index=sentence_index,
                    sentence=sentence,
                    entity_surfaces=entities,
                    triple_endpoints=triple_endpoints,
                )
            )
    if include_openie_units:
        for fact_index, triple in enumerate(triples):
            units.append(
                build_openie_unit(
                    doc_index=doc_index,
                    chunk_id=chunk_id,
                    title=title,
                    fact_index=fact_index,
                    triple=triple,
                    source_text=best_sentence_for_triple(sentences, triple) or body or passage,
                    entity_surfaces=entities,
                )
            )
    return units


def build_minimal_evidence_units_for_docs(
    *,
    doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
    include_sentence_units: bool = True,
    include_openie_units: bool = True,
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    seen_docs: Set[int] = set()
    for value in doc_indices:
        try:
            doc_index = int(value)
        except (TypeError, ValueError):
            continue
        if doc_index in seen_docs or doc_index < 0 or doc_index >= len(openie_docs):
            continue
        seen_docs.add(doc_index)
        units.extend(
            build_minimal_evidence_units_for_doc(
                openie_docs[doc_index],
                doc_index=doc_index,
                include_sentence_units=include_sentence_units,
                include_openie_units=include_openie_units,
            )
        )
    return units


def openie_docs_from_payload(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        docs = payload.get("docs", []) or []
    else:
        docs = payload
    return [doc for doc in docs if isinstance(doc, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build schema-light minimal evidence units.")
    parser.add_argument("--openie-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--no-sentence-units", action="store_true")
    parser.add_argument("--no-openie-units", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = json.loads(Path(args.openie_json).read_text(encoding="utf-8"))
    docs = openie_docs_from_payload(payload)
    if int(args.limit_docs) > 0:
        docs = docs[: int(args.limit_docs)]

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for doc_index, doc in enumerate(docs):
            for unit in build_minimal_evidence_units_for_doc(
                doc,
                doc_index=doc_index,
                include_sentence_units=not bool(args.no_sentence_units),
                include_openie_units=not bool(args.no_openie_units),
            ):
                counts[str(unit.get("source") or "")] += 1
                handle.write(json.dumps(unit, ensure_ascii=True, sort_keys=True) + "\n")

    summary = {"doc_count": len(docs), "unit_count": sum(counts.values()), "source_counts": dict(counts)}
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
