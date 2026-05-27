#!/usr/bin/env python3
"""Build a source/title/OpenIE evidence substrate.

This is an index-time construction utility. It does not run retrieval, QA,
reranking, PPR, gates, or reader calls. It compiles raw source spans, page
titles, and existing OpenIE triples into explicit evidence units that later
retrieval code can audit or consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


WEAK_ENDPOINTS = {
    "he",
    "she",
    "it",
    "they",
    "him",
    "her",
    "them",
    "the river",
    "the city",
    "the town",
    "the village",
    "the state",
    "the country",
    "the region",
    "the province",
    "the district",
    "the county",
    "the company",
    "the club",
    "the team",
    "the band",
    "the film",
    "the album",
    "the song",
    "the book",
    "the novel",
    "the play",
    "the church",
    "the cathedral",
    "the school",
    "the university",
    "the station",
    "the airport",
    "the building",
    "the series",
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def stable_id(prefix: str, values: Iterable[Any]) -> str:
    payload = json.dumps([str(value) for value in values], ensure_ascii=True, sort_keys=True)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_text(value).split()
        if len(token) >= 3 and token not in STOPWORDS
    }


def split_title_and_body(passage: str) -> Tuple[str, str]:
    lines = str(passage or "").splitlines()
    if not lines:
        return "", ""
    title = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    return title, body


def title_aliases(title: str) -> List[str]:
    aliases: List[str] = []
    normalized = normalize_text(title)
    if normalized:
        aliases.append(normalized)
    no_parenthetical = normalize_text(re.sub(r"\([^)]*\)", " ", str(title or "")))
    if no_parenthetical and no_parenthetical not in aliases:
        aliases.append(no_parenthetical)
    comma_head = normalize_text(str(title or "").split(",", 1)[0])
    if comma_head and comma_head not in aliases:
        aliases.append(comma_head)
    return aliases


def split_source_spans(body: str, *, max_chars: int = 420) -> List[Dict[str, Any]]:
    text = re.sub(r"\s+", " ", str(body or "")).strip()
    if not text:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    spans: List[Dict[str, Any]] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            spans.append({"span_index": len(spans), "span_text": current})
            current = sentence
        elif current:
            current = f"{current} {sentence}"
        else:
            current = sentence
    if current:
        spans.append({"span_index": len(spans), "span_text": current})
    return spans


def extracted_triples(doc: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    triples = doc.get("extracted_triples") or []
    clean: List[Tuple[str, str, str]] = []
    for triple in triples:
        if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes)) and len(triple) >= 3:
            clean.append((str(triple[0]), str(triple[1]), str(triple[2])))
    return clean


def extracted_entities(doc: Mapping[str, Any]) -> List[str]:
    return [str(value) for value in doc.get("extracted_entities", []) or [] if str(value).strip()]


def is_weak_endpoint(value: str) -> bool:
    normalized = normalize_text(value)
    if normalized in WEAK_ENDPOINTS:
        return True
    if normalized.startswith("the ") and len(normalized.split()) <= 3:
        return True
    return False


def text_matches_alias(value: str, aliases: Sequence[str]) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return False
    for alias in aliases:
        if normalized == alias or normalized in alias or alias in normalized:
            return True
    return False


def span_score_for_triple(span_text: str, triple: Tuple[str, str, str]) -> int:
    span_tokens = content_tokens(span_text)
    triple_tokens = content_tokens(" ".join(triple))
    return len(span_tokens & triple_tokens)


def best_span_for_triple(spans: Sequence[Mapping[str, Any]], triple: Tuple[str, str, str]) -> Mapping[str, Any]:
    if not spans:
        return {"span_index": 0, "span_text": ""}
    return max(spans, key=lambda span: (span_score_for_triple(str(span.get("span_text", "")), triple), -int(span.get("span_index", 0))))


def grounded_endpoint(
    *,
    endpoint: str,
    endpoint_role: str,
    title: str,
    aliases: Sequence[str],
    span_text: str,
) -> Tuple[str, List[str]]:
    bindings: List[str] = []
    grounded = str(endpoint)
    if text_matches_alias(endpoint, aliases):
        bindings.append(f"title_exact_{endpoint_role}")
        grounded = title
    elif is_weak_endpoint(endpoint):
        bindings.append(f"weak_{endpoint_role}_to_title")
        grounded = title
    elif any(alias and alias in normalize_text(span_text) for alias in aliases):
        bindings.append(f"title_visible_in_span_for_{endpoint_role}")
    return grounded, bindings


def build_source_span_unit(
    *,
    doc_index: int,
    chunk_id: str,
    title: str,
    aliases: Sequence[str],
    span: Mapping[str, Any],
    passage: str,
) -> Dict[str, Any]:
    span_index = int(span.get("span_index", 0))
    span_text = str(span.get("span_text", ""))
    unit_id = stable_id("sto-span", [chunk_id, doc_index, span_index, span_text])
    return {
        "unit_id": unit_id,
        "unit_type": "source_span",
        "doc_index": doc_index,
        "chunk_id": chunk_id,
        "passage_id": chunk_id,
        "title": title,
        "title_aliases": list(aliases),
        "span_id": stable_id("sto-source-span", [chunk_id, span_index, span_text]),
        "span_index": span_index,
        "span_text": span_text,
        "source_text": span_text,
        "source_passage_head": passage[:600],
        "fact_id": None,
        "fact": None,
        "subject": None,
        "relation": None,
        "object": None,
        "grounded_subject": None,
        "grounded_object": None,
        "endpoint_entities": [title] if title else [],
        "relation_cues": [],
        "title_anchor_bindings": ["source_title_anchor"] if title else [],
        "binding_status": "source_span_title_bound" if title else "source_span_without_title",
    }


def build_openie_fact_unit(
    *,
    doc_index: int,
    triple_index: int,
    chunk_id: str,
    title: str,
    aliases: Sequence[str],
    spans: Sequence[Mapping[str, Any]],
    triple: Tuple[str, str, str],
    passage: str,
) -> Dict[str, Any]:
    subject, relation, obj = triple
    span = best_span_for_triple(spans, triple)
    span_text = str(span.get("span_text", ""))
    grounded_subject, subject_bindings = grounded_endpoint(
        endpoint=subject,
        endpoint_role="subject",
        title=title,
        aliases=aliases,
        span_text=span_text,
    )
    grounded_object, object_bindings = grounded_endpoint(
        endpoint=obj,
        endpoint_role="object",
        title=title,
        aliases=aliases,
        span_text=span_text,
    )
    bindings = subject_bindings + object_bindings
    if not bindings and title:
        bindings = ["passage_title_provenance_anchor"]
    endpoint_entities = []
    for value in (grounded_subject, grounded_object, subject, obj, title):
        normalized = normalize_text(value)
        if normalized and normalized not in {normalize_text(item) for item in endpoint_entities}:
            endpoint_entities.append(str(value))
    has_endpoint_title_binding = any(
        binding.startswith("title_exact_")
        or binding.startswith("weak_")
        or binding.startswith("title_visible_in_span_")
        for binding in bindings
    )
    if has_endpoint_title_binding:
        binding_status = "title_endpoint_bound"
    elif "passage_title_provenance_anchor" in bindings:
        binding_status = "passage_title_provenance_only"
    else:
        binding_status = "openie_fact_unbound"
    fact_id = stable_id("sto-fact", [chunk_id, triple_index, subject, relation, obj])
    return {
        "unit_id": stable_id("sto-unit", [fact_id, span.get("span_index", 0), title]),
        "unit_type": "openie_fact",
        "doc_index": doc_index,
        "chunk_id": chunk_id,
        "passage_id": chunk_id,
        "title": title,
        "title_aliases": list(aliases),
        "span_id": stable_id("sto-source-span", [chunk_id, span.get("span_index", 0), span_text]),
        "span_index": int(span.get("span_index", 0)),
        "span_text": span_text,
        "source_text": span_text or passage[:600],
        "source_passage_head": passage[:600],
        "fact_id": fact_id,
        "fact": [subject, relation, obj],
        "subject": subject,
        "relation": relation,
        "object": obj,
        "grounded_subject": grounded_subject,
        "grounded_object": grounded_object,
        "endpoint_entities": endpoint_entities,
        "relation_cues": [relation] if relation else [],
        "title_anchor_bindings": bindings,
        "binding_status": binding_status,
    }


def build_units_for_doc(doc: Mapping[str, Any], *, doc_index: int, include_source_spans: bool = True) -> List[Dict[str, Any]]:
    chunk_id = str(doc.get("idx") or f"doc-{doc_index}")
    passage = str(doc.get("passage") or "")
    title, body = split_title_and_body(passage)
    if not body:
        body = passage
    aliases = title_aliases(title)
    spans = split_source_spans(body)
    if not spans and body:
        spans = [{"span_index": 0, "span_text": body}]

    units: List[Dict[str, Any]] = []
    if include_source_spans:
        units.extend(
            build_source_span_unit(
                doc_index=doc_index,
                chunk_id=chunk_id,
                title=title,
                aliases=aliases,
                span=span,
                passage=passage,
            )
            for span in spans
        )
    for triple_index, triple in enumerate(extracted_triples(doc)):
        units.append(
            build_openie_fact_unit(
                doc_index=doc_index,
                triple_index=triple_index,
                chunk_id=chunk_id,
                title=title,
                aliases=aliases,
                spans=spans,
                triple=triple,
                passage=passage,
            )
        )
    return units


def build_source_title_openie_substrate(
    *,
    openie_results_path: Path,
    output_jsonl_path: Path,
    output_summary_json_path: Path,
    limit_docs: int | None = None,
    include_source_spans: bool = True,
) -> Dict[str, Any]:
    data = json.loads(openie_results_path.read_text(encoding="utf-8"))
    docs = list(data.get("docs", []) or [])
    if limit_docs is not None and int(limit_docs) >= 0:
        docs = docs[: int(limit_docs)]

    unit_counts: Counter[str] = Counter()
    binding_counts: Counter[str] = Counter()
    doc_count_with_openie = 0
    total_units = 0
    with output_jsonl_path.open("w", encoding="utf-8") as handle:
        for doc_index, doc in enumerate(docs):
            if extracted_triples(doc):
                doc_count_with_openie += 1
            units = build_units_for_doc(doc, doc_index=doc_index, include_source_spans=include_source_spans)
            for unit in units:
                unit_counts[str(unit.get("unit_type"))] += 1
                binding_counts[str(unit.get("binding_status"))] += 1
                total_units += 1
                handle.write(json.dumps(unit, ensure_ascii=True, sort_keys=True) + "\n")

    summary = {
        "openie_results_path": str(openie_results_path),
        "output_jsonl_path": str(output_jsonl_path),
        "limit_docs": limit_docs,
        "doc_count": len(docs),
        "doc_count_with_openie": doc_count_with_openie,
        "unit_count": total_units,
        "unit_type_counts": dict(unit_counts),
        "binding_status_counts": dict(binding_counts),
    }
    output_summary_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build source/title/OpenIE evidence substrate units.")
    parser.add_argument("--openie-results", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--no-source-spans", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = build_source_title_openie_substrate(
        openie_results_path=Path(args.openie_results).resolve(),
        output_jsonl_path=Path(args.output_jsonl).resolve(),
        output_summary_json_path=Path(args.output_summary_json).resolve(),
        limit_docs=args.limit_docs,
        include_source_spans=not bool(args.no_source_spans),
    )
    print(json.dumps(summary, sort_keys=True))
    print(f"Wrote {Path(args.output_jsonl).resolve()}")
    print(f"Wrote {Path(args.output_summary_json).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
