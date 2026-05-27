"""Pool-record helpers for AREC-RAG scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.dpathrag.data import normalize_text


@dataclass(frozen=True)
class PoolDoc:
    index: int
    title: str
    text: str
    full_text: str
    doc_id: Any = None
    score: float = 0.0


def title_from_doc(text: Any) -> str:
    return str(text or "").split("\n", 1)[0].strip()


def body_from_doc(text: Any) -> str:
    value = str(text or "")
    return value.split("\n", 1)[1] if "\n" in value else value


def pool_docs(record: dict[str, Any], *, max_docs: int = 0) -> list[PoolDoc]:
    docs = list(record.get("pool_docs") or record.get("docs") or [])
    titles = list(record.get("pool_titles") or [])
    scores = list(record.get("pool_doc_scores") or [])
    ids = list(record.get("pool_doc_ids") or [])
    limit = int(max_docs)
    if limit > 0:
        docs = docs[:limit]
    output: list[PoolDoc] = []
    for idx, raw in enumerate(docs):
        title = str(titles[idx]) if idx < len(titles) else title_from_doc(raw)
        body = body_from_doc(raw)
        full_text = f"{title}\n{body}".strip()
        score = float(scores[idx]) if idx < len(scores) and scores[idx] is not None else 0.0
        doc_id = ids[idx] if idx < len(ids) else idx
        output.append(PoolDoc(index=idx, title=title, text=body, full_text=full_text, doc_id=doc_id, score=score))
    return output


def selected_titles(docs: Sequence[PoolDoc], indices: Sequence[int]) -> list[str]:
    return [docs[int(idx)].title for idx in indices if 0 <= int(idx) < len(docs)]


def normalized_titles(values: Sequence[Any]) -> set[str]:
    return {normalize_text(value) for value in values if normalize_text(value)}


def missing_gold_titles(gold_titles: Sequence[str], current_titles: Sequence[str]) -> set[str]:
    return normalized_titles(gold_titles) - normalized_titles(current_titles)


def support_counts(gold_titles: Sequence[str], titles: Sequence[str]) -> int:
    return len(normalized_titles(gold_titles) & normalized_titles(titles))

