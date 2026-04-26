"""Small metric helpers for D-PathRAG audits and cache validation."""

from __future__ import annotations

from typing import Sequence

from src.dpathrag.data import normalize_text


def recall_at_k(gold_titles: Sequence[str], pool_titles: Sequence[str], k: int) -> float:
    gold = {normalize_text(title) for title in gold_titles if normalize_text(title)}
    if not gold:
        return 0.0
    retrieved = {normalize_text(title) for title in list(pool_titles)[: int(k)]}
    return len(gold & retrieved) / len(gold)


def support_complete_at_k(gold_titles: Sequence[str], pool_titles: Sequence[str], k: int) -> float:
    gold = {normalize_text(title) for title in gold_titles if normalize_text(title)}
    if not gold:
        return 0.0
    retrieved = {normalize_text(title) for title in list(pool_titles)[: int(k)]}
    return 1.0 if gold.issubset(retrieved) else 0.0

