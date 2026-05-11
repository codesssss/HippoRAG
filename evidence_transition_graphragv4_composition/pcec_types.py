"""Typed data boundaries for native PCEC readout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PCECQueryState:
    dataset: str
    query_idx: int
    question: str
    gold_answers: tuple[str, ...] = ()
    gold_docs: tuple[str, ...] = ()
    gold_titles: tuple[str, ...] = ()
    pool_docs: tuple[str, ...] = ()
    pool_titles: tuple[str, ...] = ()
    pool_doc_ids: tuple[int | None, ...] = ()
    pool_doc_scores: tuple[float, ...] = ()
    reader_budget_k: int = 5
    prefix_budget_m: int = 4
    et_trace: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pool_record(
        cls,
        record: Mapping[str, Any],
        *,
        dataset: str,
        reader_budget_k: int,
        prefix_budget_m: int,
    ) -> "PCECQueryState":
        pool_docs = tuple(str(item) for item in list(record.get("pool_docs") or []))
        pool_titles = tuple(
            str(item)
            for item in (
                list(record.get("pool_titles") or [])
                or [str(doc).split("\n", 1)[0].strip() for doc in pool_docs]
            )
        )
        raw_scores = list(record.get("pool_doc_scores") or [])
        if len(raw_scores) != len(pool_docs):
            raw_scores = [float(len(pool_docs) - idx) for idx in range(len(pool_docs))]
        raw_doc_ids = list(record.get("pool_doc_ids") or [])
        if len(raw_doc_ids) < len(pool_docs):
            raw_doc_ids.extend([None] * (len(pool_docs) - len(raw_doc_ids)))
        doc_ids: list[int | None] = []
        for value in raw_doc_ids[: len(pool_docs)]:
            try:
                doc_ids.append(int(value) if value is not None else None)
            except (TypeError, ValueError):
                doc_ids.append(None)
        return cls(
            dataset=str(dataset),
            query_idx=int(record.get("query_idx", record.get("query_index", 0)) or 0),
            question=str(record.get("question") or ""),
            gold_answers=tuple(str(item) for item in list(record.get("gold_answers") or [])),
            gold_docs=tuple(str(item) for item in list(record.get("gold_docs") or [])),
            gold_titles=tuple(str(item) for item in list(record.get("gold_titles") or [])),
            pool_docs=pool_docs,
            pool_titles=pool_titles,
            pool_doc_ids=tuple(doc_ids),
            pool_doc_scores=tuple(float(value) for value in raw_scores[: len(pool_docs)]),
            reader_budget_k=int(reader_budget_k),
            prefix_budget_m=int(prefix_budget_m),
            et_trace=dict(record.get("agsto") or record.get("et_trace") or {}),
        )


@dataclass(frozen=True)
class DBECSetScore:
    objective: float
    coverage_by_requirement: Mapping[str, float] = field(default_factory=dict)
    selected_binding_id: str | None = None
    binding_trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PCECReadoutResult:
    final_positions: tuple[int, ...]
    final_docs: tuple[str, ...]
    final_titles: tuple[str, ...]
    baseline_positions: tuple[int, ...]
    retained_prefix_positions: tuple[int, ...]
    admitted_positions: tuple[int, ...]
    trace: Mapping[str, Any]
    legacy_selector_trace: Mapping[str, Any] = field(default_factory=dict)


def ordered_unique_positions(values: Sequence[Any], *, pool_size: int, limit: int) -> list[int]:
    positions: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            pos = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= pos < int(pool_size) and pos not in seen:
            seen.add(pos)
            positions.append(pos)
        if len(positions) >= int(limit):
            break
    return positions
