"""Feature and metric helpers for D-PathRAG selector warm-start.

Gold support labels are used only as warm-start targets and evaluation labels.
They are deliberately excluded from selector input features.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

from src.dpathrag.data import evidence_path_entities, normalize_text


FEATURE_NAMES = [
    "bias",
    "rank_reciprocal",
    "rank_fraction",
    "retriever_score",
    "retriever_score_minmax",
    "title_question_jaccard",
    "body_question_jaccard",
    "title_in_question",
    "question_token_coverage",
    "log_text_chars",
    "type_bridge",
    "type_comparison",
]

IGNORE_TARGET = -100


@dataclass(frozen=True)
class SelectorExample:
    qid: str
    question: str
    candidate_features: list[list[float]]
    query_features: list[float]
    candidate_mask: list[bool]
    target_indices: list[int]
    gold_titles: list[str]
    bridge_entities: list[str]
    candidate_titles: list[str]
    candidate_texts: list[str]
    candidate_gold_support: list[int]


def _tokens(value: Any) -> set[str]:
    text = normalize_text(value)
    if not text:
        return set()
    return {token for token in re.findall(r"[a-z0-9]+", text) if token}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _coverage(needed: set[str], observed: set[str]) -> float:
    if not needed:
        return 0.0
    return len(needed & observed) / len(needed)


def _split_title_body(text: str, fallback_title: str) -> tuple[str, str]:
    value = str(text or "")
    if "\n" not in value:
        return str(fallback_title or ""), value
    title, body = value.split("\n", 1)
    return title or str(fallback_title or ""), body


def _minmax_scores(candidates: Sequence[dict[str, Any]]) -> list[float]:
    scores = [float(candidate.get("retriever_score") or 0.0) for candidate in candidates]
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def featurize_selector_record(
    record: dict[str, Any],
    *,
    max_candidates: int = 100,
    path_len: int = 5,
    candidate_extra_features: Sequence[Sequence[float]] | None = None,
    query_extra_features: Sequence[float] | None = None,
) -> SelectorExample:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    question = str(record.get("question") or "")
    q_tokens = _tokens(question)
    q_type = normalize_text(record.get("type"))
    score_minmax = _minmax_scores(candidates)
    features: list[list[float]] = []
    titles: list[str] = []
    texts: list[str] = []
    support_labels: list[int] = []
    extra_dim = len(candidate_extra_features[0]) if candidate_extra_features is not None else 0
    for idx, candidate in enumerate(candidates):
        title = str(candidate.get("title") or "")
        raw_text = str(candidate.get("text") or "")
        title_from_text, body = _split_title_body(raw_text, title)
        if not title:
            title = title_from_text
        title_tokens = _tokens(title)
        body_tokens = _tokens(body)
        rank = float(candidate.get("rank") or (idx + 1))
        score = float(candidate.get("retriever_score") or 0.0)
        base_features = [
                1.0,
                1.0 / max(1.0, rank),
                rank / max(1.0, float(max_candidates)),
                score,
                score_minmax[idx] if idx < len(score_minmax) else 0.0,
                _jaccard(title_tokens, q_tokens),
                _jaccard(body_tokens, q_tokens),
                1.0 if normalize_text(title) and normalize_text(title) in normalize_text(question) else 0.0,
                _coverage(q_tokens, title_tokens | body_tokens),
                math.log1p(len(raw_text)) / 10.0,
                1.0 if q_type == "bridge" else 0.0,
                1.0 if q_type == "comparison" else 0.0,
            ]
        extras = (
            list(candidate_extra_features[idx])
            if candidate_extra_features is not None and idx < len(candidate_extra_features)
            else []
        )
        if extra_dim and len(extras) != extra_dim:
            raise ValueError("All candidate_extra_features rows must have the same dimension")
        features.append(base_features + [float(value) for value in extras])
        titles.append(title)
        texts.append(raw_text)
        support_labels.append(int(candidate.get("gold_support") or 0))

    while len(features) < int(max_candidates):
        features.append([0.0 for _ in FEATURE_NAMES] + [0.0 for _ in range(extra_dim)])
        titles.append("")
        texts.append("")
        support_labels.append(0)

    target_indices = [idx for idx, label in enumerate(support_labels[: len(candidates)]) if int(label) == 1]
    target_indices = target_indices[: int(path_len)]
    while len(target_indices) < int(path_len):
        target_indices.append(IGNORE_TARGET)

    q_type_bridge = 1.0 if q_type == "bridge" else 0.0
    q_type_comparison = 1.0 if q_type == "comparison" else 0.0
    query_base_features = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        min(1.0, len(q_tokens) / 20.0),
        math.log1p(len(question)) / 10.0,
        q_type_bridge,
        q_type_comparison,
    ]
    query_extras = list(query_extra_features) if query_extra_features is not None else []
    if extra_dim and len(query_extras) != extra_dim:
        raise ValueError("query_extra_features dimension must match candidate_extra_features")
    query_features = query_base_features + [float(value) for value in query_extras]
    bridge_entities = evidence_path_entities(record)
    if len(bridge_entities) > 2:
        bridge_entities = bridge_entities[1:-1]
    else:
        bridge_entities = []

    return SelectorExample(
        qid=str(record.get("qid") or record.get("query_idx") or ""),
        question=question,
        candidate_features=features,
        query_features=query_features,
        candidate_mask=[idx < len(candidates) for idx in range(int(max_candidates))],
        target_indices=target_indices,
        gold_titles=[str(title) for title in record.get("gold_titles") or []],
        bridge_entities=[str(entity) for entity in bridge_entities],
        candidate_titles=titles,
        candidate_texts=texts,
        candidate_gold_support=support_labels,
    )


def support_metrics_for_indices(example: SelectorExample, selected_indices: Sequence[int]) -> dict[str, float]:
    selected = [int(index) for index in selected_indices if 0 <= int(index) < len(example.candidate_titles)]
    selected_titles = [example.candidate_titles[index] for index in selected]
    selected_texts = [example.candidate_texts[index] for index in selected]
    gold = {normalize_text(title) for title in example.gold_titles if normalize_text(title)}
    observed = {normalize_text(title) for title in selected_titles if normalize_text(title)}
    if not gold:
        recall = 0.0
        complete = 0.0
    else:
        recall = len(gold & observed) / len(gold)
        complete = 1.0 if gold.issubset(observed) else 0.0
    title_counts = Counter(normalize_text(title) for title in selected_titles if normalize_text(title))
    duplicate_title = 1.0 if any(count > 1 for count in title_counts.values()) else 0.0
    unique_title_rate = len(title_counts) / max(1, len(selected_titles))
    selected_gold_count = sum(example.candidate_gold_support[index] for index in selected)
    bridge_entities = {normalize_text(entity) for entity in example.bridge_entities if normalize_text(entity)}
    selected_blob = normalize_text(" ".join(selected_titles + selected_texts))
    if bridge_entities:
        bridge_entity_recall = sum(1 for entity in bridge_entities if entity in selected_blob) / len(bridge_entities)
    else:
        bridge_entity_recall = 0.0
    return {
        "support_recall": recall,
        "support_complete": complete,
        "duplicate_title": duplicate_title,
        "unique_title_rate": unique_title_rate,
        "selected_gold_count": float(selected_gold_count),
        "bridge_entity_recall": bridge_entity_recall,
    }


def selection_overlap(left_indices: Sequence[int], right_indices: Sequence[int]) -> float:
    left = {int(index) for index in left_indices}
    right = {int(index) for index in right_indices}
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def summarize_selector_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "rows": 0,
            "support_recall": 0.0,
            "support_complete": 0.0,
            "duplicate_title_rate": 0.0,
            "unique_title_rate": 0.0,
            "selected_gold_count": 0.0,
            "bridge_entity_recall": 0.0,
            "selection_overlap": 0.0,
        }
    denom = float(len(rows))
    return {
        "rows": len(rows),
        "support_recall": round(sum(float(row.get("support_recall") or 0.0) for row in rows) / denom, 4),
        "support_complete": round(sum(float(row.get("support_complete") or 0.0) for row in rows) / denom, 4),
        "duplicate_title_rate": round(sum(float(row.get("duplicate_title") or 0.0) for row in rows) / denom, 4),
        "unique_title_rate": round(sum(float(row.get("unique_title_rate") or 0.0) for row in rows) / denom, 4),
        "selected_gold_count": round(sum(float(row.get("selected_gold_count") or 0.0) for row in rows) / denom, 4),
        "bridge_entity_recall": round(sum(float(row.get("bridge_entity_recall") or 0.0) for row in rows) / denom, 4),
        "selection_overlap": round(sum(float(row.get("selection_overlap") or 0.0) for row in rows) / denom, 4),
    }
