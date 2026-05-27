"""Lightweight retrieval utilities for AREC smoke tests."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Sequence

from src.dpathrag.data import normalize_text
from src.dpathrag.arec.pool import PoolDoc, normalized_titles


def tokens(text: Any) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", normalize_text(text)) if len(tok) > 1]


def token_cosine(left: Any, right: Any) -> float:
    a = set(tokens(left))
    b = set(tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def rank_docs_by_query(query: str, docs: Sequence[PoolDoc], *, exclude_titles: Sequence[str] = ()) -> list[tuple[int, float]]:
    excluded = normalized_titles(exclude_titles)
    scored: list[tuple[int, float]] = []
    for doc in docs:
        if normalize_text(doc.title) in excluded:
            continue
        scored.append((doc.index, token_cosine(query, doc.full_text)))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def top_indices_for_queries(
    queries: Sequence[str],
    docs: Sequence[PoolDoc],
    *,
    per_query_k: int,
    exclude_titles: Sequence[str] = (),
) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for query in queries:
        for idx, _score in rank_docs_by_query(query, docs, exclude_titles=exclude_titles)[: int(per_query_k)]:
            if idx in seen:
                continue
            seen.add(idx)
            output.append(idx)
    return output


def missing_support_hit_count(gold_titles: Sequence[str], initial_titles: Sequence[str], retrieved_titles: Sequence[str]) -> int:
    missing = normalized_titles(gold_titles) - normalized_titles(initial_titles)
    return len(missing & normalized_titles(retrieved_titles))


def missing_support_hit_rate(gold_titles: Sequence[str], initial_titles: Sequence[str], retrieved_titles: Sequence[str]) -> float:
    if not retrieved_titles:
        return 0.0
    return missing_support_hit_count(gold_titles, initial_titles, retrieved_titles) / max(1, len(retrieved_titles))


def tfidf_scores(query: str, docs: Sequence[PoolDoc]) -> list[float]:
    """Small deterministic scorer kept for future extension and tests."""

    doc_tokens = [tokens(doc.full_text) for doc in docs]
    query_counts = Counter(tokens(query))
    if not query_counts:
        return [0.0 for _ in docs]
    df = Counter(tok for row in doc_tokens for tok in set(row))
    n = max(1, len(docs))
    q_norm = 0.0
    q_weights: dict[str, float] = {}
    for tok, count in query_counts.items():
        idf = math.log((n + 1.0) / (df.get(tok, 0) + 1.0)) + 1.0
        weight = float(count) * idf
        q_weights[tok] = weight
        q_norm += weight * weight
    q_norm = math.sqrt(q_norm) or 1.0
    scores: list[float] = []
    for row in doc_tokens:
        counts = Counter(row)
        d_norm = 0.0
        dot = 0.0
        for tok, count in counts.items():
            idf = math.log((n + 1.0) / (df[tok] + 1.0)) + 1.0
            weight = float(count) * idf
            d_norm += weight * weight
            dot += q_weights.get(tok, 0.0) * weight
        scores.append(dot / (q_norm * (math.sqrt(d_norm) or 1.0)))
    return scores

