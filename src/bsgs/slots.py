"""Slot parsing, oracle slot construction, and slot-quality metrics."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .metrics import slot_alignment_scores


@dataclass
class Slot:
    slot_id: str
    slot_text: str
    input_variables: list[str]
    output_variable: str
    expected_answer_type: str | None = None
    gold_subquestion: str | None = None
    gold_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REF_RE = re.compile(r"#(\d+)")


def parse_json_list(text: str) -> list[dict[str, Any]]:
    """Parse a JSON list, tolerating fenced code blocks around model output."""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list of slots")
    return [item for item in payload if isinstance(item, dict)]


def references_to_variables(text: str) -> list[str]:
    variables: list[str] = []
    for ref in _REF_RE.findall(str(text or "")):
        name = f"x{ref}"
        if name not in variables:
            variables.append(name)
    return variables


def infer_expected_answer_type(answer: str) -> str:
    value = str(answer or "").strip()
    if not value:
        return "other"
    if re.fullmatch(r"\d+(?:[.,]\d+)?", value):
        return "number"
    if re.search(r"\b(19|20)\d{2}\b", value):
        return "date"
    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", value.lower()):
        return "date"
    return "other"


def build_oracle_slots_for_sample(sample: dict[str, Any]) -> list[Slot]:
    decomposition = sample.get("question_decomposition") or []
    slots: list[Slot] = []
    for idx, item in enumerate(decomposition, start=1):
        subquestion = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        output_variable = "answer" if idx == len(decomposition) else f"x{idx}"
        slots.append(
            Slot(
                slot_id=f"s{idx}",
                slot_text=subquestion,
                input_variables=references_to_variables(subquestion),
                output_variable=output_variable,
                expected_answer_type=infer_expected_answer_type(answer),
                gold_subquestion=subquestion,
                gold_answer=answer,
            )
        )
    return slots


def extract_supporting_paragraphs(sample: dict[str, Any]) -> list[dict[str, Any]]:
    paragraphs = sample.get("paragraphs") or []
    supporting: list[dict[str, Any]] = []
    support_indices = {
        item.get("paragraph_support_idx")
        for item in (sample.get("question_decomposition") or [])
        if item.get("paragraph_support_idx") is not None
    }
    for para in paragraphs:
        idx = para.get("idx")
        if bool(para.get("is_supporting")) or idx in support_indices:
            supporting.append(
                {
                    "idx": idx,
                    "title": para.get("title"),
                    "paragraph_text": para.get("paragraph_text"),
                }
            )
    return supporting


def sample_to_oracle_slot_record(sample: dict[str, Any]) -> dict[str, Any]:
    qid = sample.get("id") or sample.get("_id") or sample.get("qid")
    return {
        "qid": qid,
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "slots": [slot.to_dict() for slot in build_oracle_slots_for_sample(sample)],
        "supporting_paragraphs": extract_supporting_paragraphs(sample),
    }


def parse_predicted_slots(payload: Any) -> list[Slot]:
    if isinstance(payload, str):
        items = parse_json_list(payload)
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("predicted slots must be a JSON string or list")
    slots: list[Slot] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        slots.append(
            Slot(
                slot_id=str(item.get("slot_id") or f"p{idx}"),
                slot_text=str(item.get("slot_text") or item.get("text") or "").strip(),
                input_variables=[str(v) for v in (item.get("input_variables") or [])],
                output_variable=str(item.get("output_variable") or f"x{idx}"),
                expected_answer_type=item.get("expected_answer_type"),
            )
        )
    return [slot for slot in slots if slot.slot_text]


def evaluate_slot_quality(
    predicted: Sequence[Slot],
    gold: Sequence[Slot],
    threshold: float = 0.35,
) -> dict[str, float]:
    pred_texts = [slot.slot_text for slot in predicted]
    gold_texts = [slot.slot_text for slot in gold]
    scores = slot_alignment_scores(pred_texts, gold_texts, threshold=threshold)
    gold_vars = {slot.output_variable for slot in gold}
    pred_vars = {slot.output_variable for slot in predicted}
    scores["variable_grounding_accuracy"] = len(gold_vars & pred_vars) / len(gold_vars) if gold_vars else 0.0
    return scores
