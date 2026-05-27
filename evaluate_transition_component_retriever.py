#!/usr/bin/env python3
"""Minimal transition-component helpers needed by the v13b STO selector.

The original development tree contains a much larger diagnostic retriever under
this module name.  The v13b port only depends on these ranking/recall helpers,
so keep the compatibility surface deliberately small.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def unique_ranked(values: Iterable[Any]) -> list[int]:
    """Return non-negative integer ids in first-seen order without duplicates."""

    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        try:
            doc_idx = int(value)
        except (TypeError, ValueError):
            continue
        if doc_idx < 0 or doc_idx in seen:
            continue
        seen.add(doc_idx)
        result.append(doc_idx)
    return result


def recall_at_k(gold_indices: Sequence[Any], retrieved_indices: Sequence[Any], k: int) -> float:
    gold = set(unique_ranked(gold_indices))
    if not gold:
        return 0.0
    retrieved = set(unique_ranked(retrieved_indices)[: max(int(k), 0)])
    return len(gold & retrieved) / float(len(gold))


def all_gold_at_k(gold_indices: Sequence[Any], retrieved_indices: Sequence[Any], k: int) -> bool:
    gold = set(unique_ranked(gold_indices))
    if not gold:
        return False
    retrieved = set(unique_ranked(retrieved_indices)[: max(int(k), 0)])
    return gold.issubset(retrieved)
