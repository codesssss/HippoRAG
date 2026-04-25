"""Metrics for BSGS Week-0/Week-1 diagnostics."""

from __future__ import annotations

from collections import Counter
from math import log
from typing import Iterable, Sequence

try:
    from src.hipporag.utils.eval_utils import normalize_answer
except Exception:  # pragma: no cover - fallback for standalone script execution
    import re
    import string

    def normalize_answer(text: str) -> str:
        value = str(text or "").lower()
        value = "".join(ch for ch in value if ch not in set(string.punctuation))
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        return " ".join(value.split())


EPS = 1e-12


def exact_match(gold_answers: Sequence[str], prediction: str) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(gold) == pred for gold in gold_answers))


def token_f1(gold: str, prediction: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(prediction).split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = Counter(gold_tokens) & Counter(pred_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answer_f1(gold_answers: Sequence[str], prediction: str) -> float:
    return max((token_f1(gold, prediction) for gold in gold_answers), default=0.0)


def recall_at_gold(selected: Iterable[str], gold: Iterable[str]) -> float:
    selected_set = {normalize_answer(item) for item in selected if normalize_answer(item)}
    gold_set = {normalize_answer(item) for item in gold if normalize_answer(item)}
    if not gold_set:
        return 0.0
    return len(selected_set & gold_set) / len(gold_set)


def all_gold_covered(selected: Iterable[str], gold: Iterable[str]) -> bool:
    selected_set = {normalize_answer(item) for item in selected if normalize_answer(item)}
    gold_set = {normalize_answer(item) for item in gold if normalize_answer(item)}
    return bool(gold_set) and gold_set.issubset(selected_set)


def belief_entropy(belief: dict[str, float]) -> float:
    return -sum(float(p) * log(float(p)) for p in belief.values() if p > EPS)


def answer_in_context_but_fail(context_text: str, gold_answers: Sequence[str], prediction: str) -> bool:
    context_norm = normalize_answer(context_text)
    has_answer = any(normalize_answer(gold) and normalize_answer(gold) in context_norm for gold in gold_answers)
    return has_answer and exact_match(gold_answers, prediction) < 1.0


def slot_alignment_scores(
    predicted_slots: Sequence[str],
    gold_slots: Sequence[str],
    threshold: float = 0.35,
) -> dict[str, float]:
    """Greedy text-F1 slot alignment for Week-0 slot quality diagnostics."""
    matched_gold: set[int] = set()
    matched_pred = 0
    for pred in predicted_slots:
        best_idx = -1
        best_score = 0.0
        for idx, gold in enumerate(gold_slots):
            if idx in matched_gold:
                continue
            score = token_f1(gold, pred)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0 and best_score >= threshold:
            matched_gold.add(best_idx)
            matched_pred += 1
    precision = matched_pred / len(predicted_slots) if predicted_slots else 0.0
    recall = len(matched_gold) / len(gold_slots) if gold_slots else 0.0
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    return {"slot_precision": precision, "slot_recall": recall, "slot_f1": f1}
