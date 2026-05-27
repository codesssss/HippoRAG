"""Proof-obligation parsing and validation for AREC-RAG smokes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Sequence

from src.dpathrag.data import normalize_text


GENERIC_CLAIMS = {
    "the answer is supported by the documents",
    "the answer is supported by the evidence",
    "the documents support the answer",
    "the evidence supports the answer",
}
CONTROL_TYPES = {"comparison_control", "aggregation_control", "reasoning_control"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "therefore",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
}


@dataclass(frozen=True)
class Obligation:
    """A verifier-checkable support claim for a candidate answer."""

    id: str
    claim: str
    type: str = "factual"
    retrieval_active: bool = True
    query: str = ""
    source: str = "generated"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "type": self.type,
            "retrieval_active": self.retrieval_active,
            "query": self.query,
            "source": self.source,
        }


def _json_payload(raw: Any) -> Any:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))
    return raw


def parse_obligations(raw: Any, *, answer: str = "", source: str = "generated") -> list[Obligation]:
    """Parse obligations from JSON-like data.

    Accepted shapes:
    - ``{"obligations": [...]}``
    - ``[...]``
    - ``{"claim": "..."}``
    - ``["claim one", "claim two"]``
    """

    payload = _json_payload(raw)
    if isinstance(payload, dict) and "obligations" in payload:
        items = payload.get("obligations") or []
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        items = []

    obligations: list[Obligation] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            claim = item
            item_dict: dict[str, Any] = {}
        elif isinstance(item, dict):
            item_dict = dict(item)
            claim = str(item_dict.get("claim") or item_dict.get("text") or item_dict.get("obligation") or "")
        else:
            continue
        claim = " ".join(claim.split())
        if not claim:
            continue
        kind = str(item_dict.get("type") or "factual")
        active = bool(item_dict.get("retrieval_active", kind not in CONTROL_TYPES))
        query = str(item_dict.get("query") or build_obligation_query("", answer, claim)).strip()
        obligations.append(
            Obligation(
                id=str(item_dict.get("id") or f"o{idx}"),
                claim=claim,
                type=kind,
                retrieval_active=active,
                query=query,
                source=str(item_dict.get("source") or source),
            )
        )
    return obligations


def claim_key(claim: str) -> tuple[str, ...]:
    toks = [
        tok
        for tok in re.findall(r"[a-z0-9]+", normalize_text(claim))
        if len(tok) > 1 and tok not in STOPWORDS
    ]
    return tuple(sorted(set(toks)))


def is_answer_restatement(claim: str, answer: str) -> bool:
    text = normalize_text(claim)
    ans = normalize_text(answer)
    if not ans:
        return False
    if text in GENERIC_CLAIMS:
        return True
    if ans in text and any(marker in text for marker in ("therefore", "answer", "final answer")):
        return True
    return False


def is_vague_claim(claim: str) -> bool:
    text = normalize_text(claim)
    if text in GENERIC_CLAIMS:
        return True
    content = [tok for tok in re.findall(r"[a-z0-9]+", text) if tok not in STOPWORDS]
    return len(content) < 3


def normalize_obligations(
    obligations: Sequence[Obligation],
    *,
    answer: str = "",
    require_active: bool = False,
) -> list[Obligation]:
    """Drop duplicate and invalid obligations while preserving order."""

    seen: set[tuple[str, ...]] = set()
    kept: list[Obligation] = []
    for obligation in obligations:
        active = bool(obligation.retrieval_active)
        if obligation.type in CONTROL_TYPES:
            active = False
        if active and (is_answer_restatement(obligation.claim, answer) or is_vague_claim(obligation.claim)):
            continue
        if require_active and not active:
            continue
        key = claim_key(obligation.claim)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(
            Obligation(
                id=obligation.id,
                claim=obligation.claim,
                type=obligation.type,
                retrieval_active=active,
                query=obligation.query,
                source=obligation.source,
            )
        )
    return kept


def build_obligation_query(question: str, answer: str, claim: str) -> str:
    parts = []
    if question:
        parts.append(f"Question: {question}")
    if answer:
        parts.append(f"Candidate answer: {answer}")
    parts.append(f"Evidence needed: {claim}")
    return "\n".join(parts)


def obligations_to_dicts(obligations: Iterable[Obligation]) -> list[dict[str, Any]]:
    return [obligation.as_dict() for obligation in obligations]

