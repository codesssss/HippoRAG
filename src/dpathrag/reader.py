"""Reader-baseline formatting and evaluation helpers."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from src.hipporag.utils.eval_utils import normalize_answer
except Exception:  # pragma: no cover
    import re
    import string

    def normalize_answer(answer: str) -> str:
        value = str(answer or "").lower()
        value = "".join(ch for ch in value if ch not in set(string.punctuation))
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        return " ".join(value.split())


def load_reader_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def format_reader_input(
    record: dict[str, Any],
    *,
    max_docs: int = 0,
    max_doc_chars: int = 0,
    instruction: str = "Answer the question using only the provided documents.",
) -> str:
    docs = list(record.get("selected_docs") or [])
    if max_docs > 0:
        docs = docs[: int(max_docs)]
    lines = [instruction.strip(), "", f"Question: {record.get('question')}", "", "Documents:"]
    for idx, doc in enumerate(docs, start=1):
        text = str(doc.get("text") or "")
        if max_doc_chars > 0:
            text = text[: int(max_doc_chars)]
        title = str(doc.get("title") or "")
        lines.append(f"[{idx}] {title}")
        lines.append(text)
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def exact_match(gold_answers: Sequence[str], prediction: str) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(gold) == pred for gold in gold_answers))


def token_f1_single(gold: str, prediction: str) -> float:
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


def token_f1(gold_answers: Sequence[str], prediction: str) -> float:
    return max((token_f1_single(gold, prediction) for gold in gold_answers), default=0.0)


def gold_answers(record: dict[str, Any]) -> list[str]:
    answer = record.get("answer")
    if isinstance(answer, list):
        return [str(item) for item in answer]
    return [str(answer or "")]


def score_prediction(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    answers = gold_answers(record)
    return {
        "qid": record.get("qid"),
        "source": record.get("source"),
        "prediction": prediction,
        "gold_answers": answers,
        "em": exact_match(answers, prediction),
        "f1": token_f1(answers, prediction),
        "support_recall": float(record.get("support_recall") or 0.0),
        "support_complete": float(record.get("support_complete") or 0.0),
        "selected_doc_count": len(record.get("selected_docs") or []),
    }


def summarize_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "answer_em": 0.0,
            "answer_f1": 0.0,
            "support_recall": 0.0,
            "support_complete": 0.0,
            "avg_selected_doc_count": 0.0,
        }
    denom = float(len(rows))
    return {
        "rows": len(rows),
        "answer_em": round(sum(float(row.get("em") or 0.0) for row in rows) / denom, 4),
        "answer_f1": round(sum(float(row.get("f1") or 0.0) for row in rows) / denom, 4),
        "support_recall": round(sum(float(row.get("support_recall") or 0.0) for row in rows) / denom, 4),
        "support_complete": round(sum(float(row.get("support_complete") or 0.0) for row in rows) / denom, 4),
        "avg_selected_doc_count": round(sum(float(row.get("selected_doc_count") or 0.0) for row in rows) / denom, 4),
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

