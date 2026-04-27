#!/usr/bin/env python3
"""CPAG: query-local consensus proposition agreement evidence assembly.

This script evaluates a training-free, LLM-assisted evidence assembly operator.
It uses cached Qwen OpenIE outputs as local proposition/entity extraction, builds
a query-local agreement graph over PropRAG/Dense candidate pools, and selects a
5-document evidence set with deterministic greedy agreement coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import normalize_answer  # noqa: E402
from src.dpathrag.reader_data import selected_doc_payload, summarize_reader_records  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
CAPITALIZED_SPAN_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9'’.-]+(?:\s+|$)){1,6}")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{1,2},?)?\s+\d{3,4}\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")

ENTITY_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "who",
    "what",
    "which",
    "where",
    "when",
    "was",
    "were",
    "is",
    "are",
    "did",
    "does",
    "do",
    "this",
    "that",
    "these",
    "those",
}

ANCHOR_STOP_PHRASES = {
    "place of birth",
    "date of birth",
    "date of death",
    "place of death",
    "country",
    "film",
    "song",
    "director",
    "performer",
    "mother",
    "father",
    "spouse",
    "wife",
    "husband",
}


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def row_qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("query_idx") or "")


def norm_text(value: Any) -> str:
    return normalize_answer(str(value or ""))


def token_set(value: Any) -> set[str]:
    return set(TOKEN_RE.findall(norm_text(value)))


def candidate_body(candidate: dict[str, Any]) -> str:
    text = str(candidate.get("text") or "")
    return text.split("\n", 1)[1] if "\n" in text else text


def normalize_title_key(title: Any) -> str:
    return norm_text(title)


def normalize_entity(value: Any) -> str:
    norm = norm_text(value)
    if not norm or norm in ENTITY_STOP:
        return ""
    if norm in ANCHOR_STOP_PHRASES:
        return ""
    if len(norm) <= 1:
        return ""
    return norm


def canonical_doc_key(candidate: dict[str, Any]) -> str:
    title_key = normalize_title_key(candidate.get("title"))
    if title_key:
        return f"title::{title_key}"
    return f"text::{norm_text(candidate.get('text'))[:120]}"


def extract_question_anchors(question: str, pool_titles: Sequence[str]) -> set[str]:
    anchors: set[str] = set()
    for match in CAPITALIZED_SPAN_RE.findall(question):
        entity = normalize_entity(" ".join(str(match).split()))
        if entity:
            anchors.add(entity)
    q_norm = norm_text(question)
    q_tokens = token_set(question)
    for title in pool_titles:
        title_norm = normalize_entity(title)
        if not title_norm:
            continue
        title_tokens = token_set(title)
        if title_norm in q_norm or (title_tokens and len(title_tokens & q_tokens) >= min(2, len(title_tokens))):
            anchors.add(title_norm)
    return anchors


def expected_answer_kind(question: str) -> str:
    q = norm_text(question)
    if q.startswith("when") or "what year" in q or "which year" in q or "date" in q:
        return "date"
    if "how many" in q or "number" in q or "population" in q:
        return "number"
    if q.startswith("where") or "place" in q or "birthplace" in q or "located" in q:
        return "place"
    if q.startswith("who") or "person" in q:
        return "person"
    return "entity"


def looks_answer_type(entity: str, *, question: str) -> bool:
    kind = expected_answer_kind(question)
    raw = str(entity or "")
    if kind == "date":
        return bool(DATE_RE.search(raw) or YEAR_RE.search(raw))
    if kind == "number":
        return bool(re.search(r"\d", raw))
    return bool(normalize_entity(raw))


def load_openie_index(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    docs = payload.get("docs") if isinstance(payload, dict) else payload
    index: dict[str, dict[str, Any]] = {}
    for doc in docs or []:
        passage = str(doc.get("passage") or "")
        title = passage.split("\n", 1)[0].strip() if passage else ""
        if title:
            index.setdefault(normalize_title_key(title), doc)
        idx = str(doc.get("idx") or "")
        if idx:
            index.setdefault(f"id::{idx}", doc)
    return index


def heuristic_entities(text: str, *, title: str = "") -> list[str]:
    entities: list[str] = []
    if title:
        entities.append(str(title))
    for match in DATE_RE.findall(text):
        entities.append(str(match))
    for match in YEAR_RE.findall(text):
        entities.append(str(match))
    for match in CAPITALIZED_SPAN_RE.findall(text):
        value = " ".join(str(match).split())
        if normalize_entity(value):
            entities.append(value)
    seen: set[str] = set()
    output: list[str] = []
    for entity in entities:
        norm = normalize_entity(entity)
        if norm and norm not in seen:
            seen.add(norm)
            output.append(str(entity))
    return output[:20]


def split_sentences(text: str, *, max_sentences: int) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return [part.strip() for part in parts if part.strip()][: int(max_sentences)]


def propositions_for_doc(
    candidate: dict[str, Any],
    *,
    openie_index: dict[str, dict[str, Any]],
    max_props: int,
) -> list[dict[str, Any]]:
    title = str(candidate.get("title") or "")
    body = candidate_body(candidate)
    openie = openie_index.get(normalize_title_key(title)) or openie_index.get(f"id::{candidate.get('doc_id')}")
    props: list[dict[str, Any]] = []
    if openie:
        extracted_entities = [str(item) for item in openie.get("extracted_entities") or []]
        for idx, triple in enumerate(list(openie.get("extracted_triples") or [])[: int(max_props)], start=1):
            if not isinstance(triple, list) or len(triple) < 3:
                continue
            subj, rel, obj = str(triple[0]), str(triple[1]), str(triple[2])
            entities = [subj, obj]
            text = f"{subj} {rel} {obj}."
            props.append(
                {
                    "text": text,
                    "entities": [entity for entity in entities if normalize_entity(entity)],
                    "source": "openie_triple",
                    "triple": [subj, rel, obj],
                }
            )
        if len(props) < int(max_props):
            for entity in extracted_entities:
                norm = normalize_entity(entity)
                if not norm:
                    continue
                props.append({"text": str(entity), "entities": [str(entity)], "source": "openie_entity"})
                if len(props) >= int(max_props):
                    break
    if not props:
        for sentence in split_sentences(body, max_sentences=int(max_props)):
            ents = heuristic_entities(sentence, title="")
            props.append({"text": sentence, "entities": ents, "source": "heuristic_sentence"})
    if not props and title:
        props.append({"text": title, "entities": [title], "source": "title_fallback"})
    return props[: int(max_props)]


def dedup_union_docs(
    rows: Sequence[tuple[str, dict[str, Any] | None]],
    *,
    pool_k: int,
    cap: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source, record in rows:
        if record is None:
            continue
        for idx, candidate in enumerate(list(record.get("candidates") or [])[: int(pool_k)], start=1):
            key = canonical_doc_key(candidate)
            if key not in merged:
                row = dict(candidate)
                row["pool_sources"] = [source]
                row["pool_ranks"] = {source: idx}
                row["best_rank"] = idx
                row["doc_key"] = key
                merged[key] = row
                order.append(key)
            else:
                row = merged[key]
                if source not in row["pool_sources"]:
                    row["pool_sources"].append(source)
                row["pool_ranks"][source] = min(int(row["pool_ranks"].get(source, idx)), idx)
                row["best_rank"] = min(int(row.get("best_rank") or idx), idx)
    docs = [merged[key] for key in order]
    docs.sort(key=lambda item: (int(item.get("best_rank") or 10**9), -len(item.get("pool_sources") or []), str(item.get("title") or "")))
    return docs[: int(cap)]


def build_doc_signals(
    docs: Sequence[dict[str, Any]],
    *,
    question: str,
    openie_index: dict[str, dict[str, Any]],
    max_props: int,
) -> list[dict[str, Any]]:
    pool_titles = [str(doc.get("title") or "") for doc in docs]
    anchors = extract_question_anchors(question, pool_titles)
    entity_doc_counts: Counter[str] = Counter()
    entity_pool_sets: dict[str, set[str]] = defaultdict(set)
    doc_rows: list[dict[str, Any]] = []
    for doc_idx, doc in enumerate(docs):
        props = propositions_for_doc(doc, openie_index=openie_index, max_props=int(max_props))
        entity_norms: set[str] = set()
        for prop in props:
            for entity in prop.get("entities") or []:
                norm = normalize_entity(entity)
                if norm:
                    entity_norms.add(norm)
        title_norm = normalize_entity(doc.get("title"))
        if title_norm:
            entity_norms.add(title_norm)
        sources = {str(source) for source in doc.get("pool_sources") or []}
        for entity in entity_norms:
            entity_doc_counts[entity] += 1
            entity_pool_sets[entity].update(sources)
        doc_rows.append({**doc, "doc_index": doc_idx, "propositions": props, "entity_norms": sorted(entity_norms)})

    agreed_entities = {
        entity
        for entity, count in entity_doc_counts.items()
        if int(count) >= 2 or len(entity_pool_sets.get(entity) or set()) >= 2
    }
    answer_entities = {entity for row in doc_rows for entity in row["entity_norms"] if looks_answer_type(entity, question=question)}
    for row in doc_rows:
        entities = set(row["entity_norms"])
        row["anchor_entities"] = sorted(entities & anchors)
        row["agreed_entities"] = sorted(entities & agreed_entities)
        row["bridge_entities"] = sorted((entities & agreed_entities) - anchors)
        row["answer_type_entities"] = sorted((entities & answer_entities) - anchors)
        row["pool_count"] = len(row.get("pool_sources") or [])
        row["is_singleton_lexical"] = bool(not row["agreed_entities"] and not row["anchor_entities"] and int(row["pool_count"]) <= 1)
    return doc_rows


def greedy_agreement_select(rows: Sequence[dict[str, Any]], *, top_k: int) -> list[int]:
    selected: list[int] = []
    covered: set[str] = set()
    remaining = set(range(len(rows)))

    def doc_elements(row: dict[str, Any]) -> set[str]:
        elements: set[str] = set()
        for entity in row.get("bridge_entities") or []:
            elements.add(f"bridge::{entity}")
        for entity in row.get("answer_type_entities") or []:
            elements.add(f"answer::{entity}")
        for entity in row.get("anchor_entities") or []:
            elements.add(f"anchor::{entity}")
        if int(row.get("pool_count") or 0) >= 2:
            elements.add(f"cross_pool::{row.get('doc_key')}")
        return elements

    while remaining and len(selected) < int(top_k):
        best_idx: int | None = None
        best_key: tuple[float, float, float, float, str] | None = None
        for idx in sorted(remaining):
            row = rows[idx]
            if bool(row.get("is_singleton_lexical")) and len(selected) < max(1, int(top_k) - 1):
                singleton_penalty = -1.0
            else:
                singleton_penalty = 0.0
            new_elements = doc_elements(row) - covered
            key = (
                float(len(new_elements)) + singleton_penalty,
                float(len(row.get("bridge_entities") or [])),
                float(row.get("pool_count") or 0),
                -float(row.get("best_rank") or 10**9),
                str(row.get("title") or ""),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        covered.update(doc_elements(rows[best_idx]))
        remaining.remove(best_idx)
    return selected


def anchor_first_agreement_select(rows: Sequence[dict[str, Any]], *, top_k: int) -> list[int]:
    """Select anchor-bearing docs first, then close agreement coverage.

    The pure agreement cover tends to chase high-degree family/location hubs.  CPAG
    v0 is intentionally conservative: it first anchors the set on query-linked
    documents, then uses proposition/entity agreement only for the remaining
    slots.  This keeps the method from becoming an unconstrained global hub
    selector while preserving the agreement-graph operator.
    """
    selected: list[int] = []
    anchor_candidates = [
        idx
        for idx, row in enumerate(rows)
        if row.get("anchor_entities") or (int(row.get("pool_count") or 0) >= 2 and norm_text(row.get("title")) in {norm_text(entity) for entity in row.get("anchor_entities") or []})
    ]
    anchor_candidates.sort(
        key=lambda idx: (
            -len(rows[idx].get("anchor_entities") or []),
            -int(rows[idx].get("pool_count") or 0),
            int(rows[idx].get("best_rank") or 10**9),
            str(rows[idx].get("title") or ""),
        )
    )
    for idx in anchor_candidates:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= min(int(top_k), 2):
            break
    if len(selected) >= int(top_k):
        return selected[: int(top_k)]

    tail_rows = list(rows)
    greedy = greedy_agreement_select(tail_rows, top_k=int(top_k))
    for idx in greedy:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= int(top_k):
            break
    if len(selected) < int(top_k):
        ranked = sorted(
            range(len(rows)),
            key=lambda idx: (
                -int(rows[idx].get("pool_count") or 0),
                int(rows[idx].get("best_rank") or 10**9),
                str(rows[idx].get("title") or ""),
            ),
        )
        for idx in ranked:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= int(top_k):
                break
    return selected[: int(top_k)]


def rank_fusion_select(docs: Sequence[dict[str, Any]], *, top_k: int, k_const: int = 60) -> list[int]:
    scored = []
    for idx, doc in enumerate(docs):
        score = 0.0
        for rank in (doc.get("pool_ranks") or {}).values():
            score += 1.0 / (int(k_const) + int(rank))
        scored.append((score, -int(doc.get("pool_count") or 0), int(doc.get("best_rank") or 10**9), idx))
    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [idx for _, _, _, idx in scored[: int(top_k)]]


def support_metrics(gold_titles: Sequence[str], selected_docs: Sequence[dict[str, Any]]) -> dict[str, float]:
    gold = {norm_text(title) for title in gold_titles if norm_text(title)}
    observed = {norm_text(doc.get("title")) for doc in selected_docs if norm_text(doc.get("title"))}
    if not gold:
        recall = 0.0
        complete = 0.0
    else:
        recall = len(gold & observed) / len(gold)
        complete = 1.0 if gold.issubset(observed) else 0.0
    selected_gold_count = sum(1 for doc in selected_docs if norm_text(doc.get("title")) in gold)
    return {
        "support_recall": float(recall),
        "support_complete": float(complete),
        "selected_gold_count": float(selected_gold_count),
    }


def selected_titles(docs: Sequence[dict[str, Any]], indices: Sequence[int]) -> list[str]:
    return [str(docs[int(idx)].get("title") or "") for idx in indices if 0 <= int(idx) < len(docs)]


def build_reader_row(
    record: dict[str, Any],
    selected_docs: Sequence[dict[str, Any]],
    *,
    source: str,
    max_doc_chars: int,
) -> dict[str, Any]:
    gold_titles = list(record.get("gold_titles") or [])
    docs = []
    for rank, doc in enumerate(selected_docs, start=1):
        docs.append(
            selected_doc_payload(
                title=str(doc.get("title") or ""),
                text=candidate_body(doc),
                rank=rank,
                source=source,
                gold_titles=gold_titles,
                doc_id=doc.get("doc_id"),
                score=doc.get("retriever_score"),
                max_chars=int(max_doc_chars),
            )
        )
    metrics = support_metrics(gold_titles, selected_docs)
    return {
        "qid": row_qid(record),
        "query_idx": record.get("query_idx"),
        "split": "cpag_eval",
        "source": source,
        "top_k": len(docs),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "type": record.get("type"),
        "gold_titles": gold_titles,
        "selected_docs": docs,
        "support_recall": metrics["support_recall"],
        "support_complete": metrics["support_complete"],
    }


def summarize_selection_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    denom = float(len(rows))
    return {
        "rows": len(rows),
        "support_recall": round(sum(float(row["support_recall"]) for row in rows) / denom, 4),
        "support_complete": round(sum(float(row["support_complete"]) for row in rows) / denom, 4),
        "selected_gold_count": round(sum(float(row["selected_gold_count"]) for row in rows) / denom, 4),
        "avg_pool_size": round(sum(float(row.get("pool_size") or 0.0) for row in rows) / denom, 4),
        "avg_cross_pool_docs": round(sum(float(row.get("cross_pool_docs") or 0.0) for row in rows) / denom, 4),
        "avg_agreed_entities": round(sum(float(row.get("agreed_entity_count") or 0.0) for row in rows) / denom, 4),
    }


def summarize_by_type(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("type") or "unknown")].append(row)
    return {qtype: summarize_selection_rows(items) for qtype, items in sorted(grouped.items())}


def pool_signal_auc(details: Sequence[dict[str, Any]]) -> float:
    positives = []
    negatives = []
    for row in details:
        for doc in row.get("doc_diagnostics") or []:
            value = float(doc.get("pool_count") or 0.0)
            if int(doc.get("gold_support") or 0) == 1:
                positives.append(value)
            else:
                negatives.append(value)
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif math.isclose(pos, neg):
                wins += 0.5
    return round(wins / max(1, total), 6)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prop_rows = load_jsonl(args.proprag_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))[int(args.dev_start) : int(args.dev_end)]
    dense_rows = load_jsonl(args.dense_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))
    dense_by_qid = {row_qid(row): row for row in dense_rows}
    openie_index = load_openie_index(args.openie_json)

    variant_details: dict[str, list[dict[str, Any]]] = {
        "proprag_rank": [],
        "dense_rank": [],
        "rrf": [],
        "cpag_pure": [],
        "cpag": [],
    }
    reader_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in variant_details}

    for prop_record in prop_rows:
        qid = row_qid(prop_record)
        dense_record = dense_by_qid.get(qid)
        union_docs = dedup_union_docs(
            [("proprag", prop_record), ("dense", dense_record)],
            pool_k=int(args.pool_k),
            cap=int(args.pool_cap),
        )
        signal_rows = build_doc_signals(
            union_docs,
            question=str(prop_record.get("question") or ""),
            openie_index=openie_index,
            max_props=int(args.max_props),
        )
        variant_indices = {
            "proprag_rank": [idx for idx, doc in enumerate(union_docs) if "proprag" in set(doc.get("pool_sources") or [])][: int(args.top_k)],
            "dense_rank": [idx for idx, doc in enumerate(union_docs) if "dense" in set(doc.get("pool_sources") or [])][: int(args.top_k)],
            "rrf": rank_fusion_select(union_docs, top_k=int(args.top_k)),
            "cpag_pure": greedy_agreement_select(signal_rows, top_k=int(args.top_k)),
            "cpag": anchor_first_agreement_select(signal_rows, top_k=int(args.top_k)),
        }
        doc_diagnostics = [
            {
                "title": str(row.get("title") or ""),
                "pool_count": int(row.get("pool_count") or 0),
                "pool_sources": list(row.get("pool_sources") or []),
                "best_rank": int(row.get("best_rank") or 0),
                "gold_support": int(norm_text(row.get("title")) in {norm_text(title) for title in prop_record.get("gold_titles") or []}),
                "agreed_entity_count": len(row.get("agreed_entities") or []),
                "bridge_entity_count": len(row.get("bridge_entities") or []),
                "anchor_entity_count": len(row.get("anchor_entities") or []),
            }
            for row in signal_rows
        ]
        for variant, indices in variant_indices.items():
            chosen_docs = [signal_rows[int(idx)] for idx in indices if 0 <= int(idx) < len(signal_rows)]
            metrics = support_metrics(prop_record.get("gold_titles") or [], chosen_docs)
            rank_docs = [signal_rows[int(idx)] for idx in variant_indices["proprag_rank"] if 0 <= int(idx) < len(signal_rows)]
            rank_gold = int(support_metrics(prop_record.get("gold_titles") or [], rank_docs)["selected_gold_count"])
            selected_gold = int(metrics["selected_gold_count"])
            detail = {
                "qid": qid,
                "query_idx": prop_record.get("query_idx"),
                "type": prop_record.get("type"),
                "question": prop_record.get("question"),
                "gold_titles": list(prop_record.get("gold_titles") or []),
                "selected_titles": selected_titles(signal_rows, indices),
                "selected_indices": [int(idx) for idx in indices],
                "support_recall": metrics["support_recall"],
                "support_complete": metrics["support_complete"],
                "selected_gold_count": metrics["selected_gold_count"],
                "added_gold_vs_proprag": max(0, selected_gold - rank_gold),
                "added_non_gold_vs_proprag": max(0, int(args.top_k) - selected_gold - (int(args.top_k) - rank_gold)),
                "pool_size": len(signal_rows),
                "cross_pool_docs": sum(1 for row in signal_rows if int(row.get("pool_count") or 0) >= 2),
                "agreed_entity_count": len({entity for row in signal_rows for entity in row.get("agreed_entities") or []}),
                "doc_diagnostics": doc_diagnostics if variant == "cpag" else [],
            }
            variant_details[variant].append(detail)
            reader_rows[variant].append(
                build_reader_row(
                    prop_record,
                    chosen_docs,
                    source=f"cpag_{variant}",
                    max_doc_chars=int(args.max_doc_chars),
                )
            )

    summaries = {variant: summarize_selection_rows(rows) for variant, rows in variant_details.items()}
    by_type = {variant: summarize_by_type(rows) for variant, rows in variant_details.items()}
    rank_sc = float(summaries["proprag_rank"].get("support_complete") or 0.0)
    cpag_sc = float(summaries["cpag"].get("support_complete") or 0.0)
    added_gold = sum(float(row.get("added_gold_vs_proprag") or 0.0) for row in variant_details["cpag"])
    added_non_gold = sum(float(row.get("added_non_gold_vs_proprag") or 0.0) for row in variant_details["cpag"])
    non_gold_per_gold = round(added_non_gold / added_gold, 6) if added_gold > 0 else None
    signal = {
        "cross_pool_gold_vs_nongold_auc": pool_signal_auc(variant_details["cpag"]),
        "cpag_support_complete_delta_vs_proprag": round(cpag_sc - rank_sc, 6),
        "cpag_added_gold_vs_proprag": int(added_gold),
        "cpag_added_non_gold_vs_proprag": int(added_non_gold),
        "cpag_non_gold_per_gold_vs_proprag": non_gold_per_gold,
    }
    payload = {
        "config": {
            "proprag_cache_jsonl": str(args.proprag_cache_jsonl),
            "dense_cache_jsonl": str(args.dense_cache_jsonl),
            "openie_json": str(args.openie_json),
            "dev_start": int(args.dev_start),
            "dev_end": int(args.dev_end),
            "pool_k": int(args.pool_k),
            "pool_cap": int(args.pool_cap),
            "top_k": int(args.top_k),
            "max_props": int(args.max_props),
        },
        "summaries": summaries,
        "by_type": by_type,
        "signal": signal,
        "reader_manifests": {variant: summarize_reader_records(rows) for variant, rows in reader_rows.items()},
    }
    write_json(payload, output_dir / "cpag_day1_agreement_gate.json")
    for variant, rows in variant_details.items():
        write_jsonl(rows, output_dir / f"cpag_day1_{variant}.rows.jsonl")
        write_jsonl(reader_rows[variant], output_dir / f"cpag_day1_{variant}.reader.jsonl")
    write_markdown_report(payload, output_dir / "cpag_day1_agreement_gate.md")
    return payload


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# CPAG Day-1 Agreement Gate",
        "",
        "## Summary",
        "",
        "| Variant | Rows | Support Recall | Support Complete | Selected Gold | Avg Pool | Avg Cross-Pool Docs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in payload["summaries"].items():
        lines.append(
            f"| {variant} | {row.get('rows')} | {row.get('support_recall')} | {row.get('support_complete')} | "
            f"{row.get('selected_gold_count')} | {row.get('avg_pool_size')} | {row.get('avg_cross_pool_docs')} |"
        )
    signal = payload["signal"]
    lines.extend(
        [
            "",
            "## CPAG Signal",
            "",
            f"- Cross-pool gold-vs-non-gold AUC: `{signal['cross_pool_gold_vs_nongold_auc']}`",
            f"- CPAG support-complete delta vs PropRAG rank: `{signal['cpag_support_complete_delta_vs_proprag']}`",
            f"- CPAG added gold vs PropRAG: `{signal['cpag_added_gold_vs_proprag']}`",
            f"- CPAG added non-gold vs PropRAG: `{signal['cpag_added_non_gold_vs_proprag']}`",
            f"- CPAG non-gold/gold vs PropRAG: `{signal['cpag_non_gold_per_gold_vs_proprag']}`",
            "",
            "## Per-Type Support Complete",
            "",
            "| Variant | Type | Rows | Support Complete | Support Recall |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant, grouped in payload["by_type"].items():
        for qtype, row in grouped.items():
            lines.append(f"| {variant} | {qtype} | {row.get('rows')} | {row.get('support_complete')} | {row.get('support_recall')} |")
    lines.extend(
        [
            "",
            "## Reader Rows",
            "",
            "Reader JSONL files are written per variant in this directory and should be evaluated with `scripts/dpathrag_eval_reader_baseline.py`.",
        ]
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proprag_cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--dense_cache_jsonl", default="data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl")
    parser.add_argument("--openie_json", default="outputs/2wikimultihopqa/openie_results_ner_qwen3-8b.json")
    parser.add_argument("--dev_start", type=int, default=0)
    parser.add_argument("--dev_end", type=int, default=200)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--pool_cap", type=int, default=60)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_props", type=int, default=8)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--output_dir", default="reports/cpag")
    return parser.parse_args()


def main() -> None:
    payload = evaluate(parse_args())
    print(json.dumps({"summaries": payload["summaries"], "signal": payload["signal"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
