"""Reader-baseline JSONL builders for D-PathRAG.

These records intentionally separate reader-input documents from selector
training labels.  Gold support can define a `gold` reader baseline, but it must
not be inserted into selector input features.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

from src.dpathrag.data import context_documents, extract_title, normalize_text, support_titles
from src.dpathrag.metrics import recall_at_k, support_complete_at_k


def _dedupe_docs(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for doc in docs:
        key = (normalize_text(doc.get("title")), normalize_text(doc.get("text") or doc.get("doc")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(doc))
    return deduped


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    return text[:max_chars]


def selected_doc_payload(
    *,
    title: str,
    text: str,
    rank: int,
    source: str,
    gold_titles: Sequence[str],
    doc_id: Any = None,
    score: float | None = None,
    max_chars: int = 0,
) -> dict[str, Any]:
    full_text = f"{title}\n{text}".strip()
    return {
        "doc_id": doc_id,
        "rank": int(rank),
        "title": str(title),
        "text": _truncate_text(full_text, int(max_chars)),
        "source": source,
        "retriever_score": score,
        "gold_support": int(normalize_text(title) in {normalize_text(item) for item in gold_titles}),
    }


def build_gold_reader_record(
    sample: dict[str, Any],
    *,
    split: str,
    top_k: int = 0,
    max_doc_chars: int = 0,
) -> dict[str, Any]:
    gold_titles = support_titles(sample)
    wanted = {normalize_text(title) for title in gold_titles}
    docs: list[dict[str, Any]] = []
    for doc in context_documents(sample):
        if normalize_text(doc["title"]) not in wanted:
            continue
        docs.append(
            selected_doc_payload(
                title=str(doc["title"]),
                text=str(doc["text"]),
                rank=len(docs) + 1,
                source="gold",
                gold_titles=gold_titles,
                doc_id=doc.get("idx"),
                score=None,
                max_chars=max_doc_chars,
            )
        )
    selected_docs = _dedupe_docs(docs)
    if top_k > 0:
        selected_docs = selected_docs[: int(top_k)]
    selected_titles = [doc["title"] for doc in selected_docs]
    return {
        "qid": sample.get("_id") or sample.get("id"),
        "split": split,
        "source": "gold",
        "top_k": len(selected_docs),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "type": sample.get("type"),
        "gold_titles": gold_titles,
        "selected_docs": selected_docs,
        "support_recall": recall_at_k(gold_titles, selected_titles, len(selected_titles)),
        "support_complete": support_complete_at_k(gold_titles, selected_titles, len(selected_titles)),
    }


def build_pool_reader_record(
    sample: dict[str, Any],
    pool_record: dict[str, Any],
    *,
    split: str,
    source: str,
    top_k: int,
    max_doc_chars: int = 0,
    strict_question: bool = True,
) -> dict[str, Any]:
    if strict_question and normalize_text(sample.get("question")) != normalize_text(pool_record.get("question")):
        raise ValueError(f"Question mismatch for qid={sample.get('_id') or sample.get('id')}")
    gold_titles = support_titles(sample) or list(pool_record.get("gold_titles") or [])
    pool_docs = list(pool_record.get("pool_docs") or [])
    pool_titles = list(pool_record.get("pool_titles") or [extract_title(doc) for doc in pool_docs])
    pool_scores = list(pool_record.get("pool_doc_scores") or [])
    pool_ids = list(pool_record.get("pool_doc_ids") or [])
    selected_docs: list[dict[str, Any]] = []
    for idx, doc_text in enumerate(pool_docs[: int(top_k)]):
        title = pool_titles[idx] if idx < len(pool_titles) else extract_title(doc_text)
        body = str(doc_text).split("\n", 1)[1] if "\n" in str(doc_text) else str(doc_text)
        score = float(pool_scores[idx]) if idx < len(pool_scores) else None
        doc_id = pool_ids[idx] if idx < len(pool_ids) else None
        selected_docs.append(
            selected_doc_payload(
                title=title,
                text=body,
                rank=idx + 1,
                source=source,
                gold_titles=gold_titles,
                doc_id=doc_id,
                score=score,
                max_chars=max_doc_chars,
            )
        )
    selected_titles = [doc["title"] for doc in selected_docs]
    return {
        "qid": sample.get("_id") or sample.get("id") or pool_record.get("query_idx"),
        "query_idx": pool_record.get("query_idx"),
        "split": split,
        "source": source,
        "top_k": int(top_k),
        "question": sample.get("question") or pool_record.get("question"),
        "answer": sample.get("answer"),
        "type": sample.get("type"),
        "gold_titles": list(gold_titles),
        "selected_docs": selected_docs,
        "support_recall": recall_at_k(gold_titles, selected_titles, int(top_k)),
        "support_complete": support_complete_at_k(gold_titles, selected_titles, int(top_k)),
    }


def summarize_reader_records(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "avg_selected_docs": 0.0,
            "avg_support_recall": 0.0,
            "support_complete_rate": 0.0,
        }
    return {
        "rows": len(rows),
        "avg_selected_docs": round(float(mean(len(row.get("selected_docs") or []) for row in rows)), 4),
        "avg_support_recall": round(float(mean(float(row.get("support_recall") or 0.0) for row in rows)), 4),
        "support_complete_rate": round(float(mean(float(row.get("support_complete") or 0.0) for row in rows)), 4),
    }
