"""Frozen verifier adapters for AREC-RAG smokes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.dpathrag.arec.retrieval import token_cosine


class Verifier(Protocol):
    name: str

    def score(self, obligation: str, document: str) -> float:
        ...


@dataclass
class LexicalSmokeVerifier:
    """Deterministic verifier for tests and smoke plumbing.

    This is not a paper verifier. Reports must label it as lexical smoke.
    """

    threshold: float = 0.05
    name: str = "lexical_smoke"

    def score(self, obligation: str, document: str) -> float:
        score = token_cosine(obligation, document)
        return float(score if score >= self.threshold else 0.0)

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [self.score(obligation, document) for obligation, document in pairs]


class NliVerifier:
    """Lazy HuggingFace NLI verifier.

    The implementation intentionally imports transformers lazily so unit tests
    and lexical smoke runs do not require model dependencies.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base-mnli", *, batch_size: int = 16) -> None:
        self.name = model_name
        self.batch_size = int(batch_size)
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency/environment dependent
            raise RuntimeError(f"transformers is required for NLI verifier '{model_name}': {exc}") from exc
        self._pipeline = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)

    def score(self, obligation: str, document: str) -> float:
        return self.score_many([(obligation, document)])[0]

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        inputs = [{"text": str(document), "text_pair": str(obligation)} for obligation, document in pairs]
        result = self._pipeline(inputs, batch_size=self.batch_size)
        return [self._support_score(rows) for rows in result]

    @staticmethod
    def _support_score(rows: Any) -> float:
        if rows and isinstance(rows[0], list):
            rows = rows[0]
        best_support = 0.0
        for row in rows:
            label = str(row.get("label") or "").lower()
            if "entail" in label or "support" in label:
                best_support = max(best_support, float(row.get("score") or 0.0))
        return float(best_support)


def build_verifier(backend: str, *, model_name: str = "microsoft/deberta-v3-base-mnli", batch_size: int = 16) -> Verifier:
    if backend == "lexical_smoke":
        return LexicalSmokeVerifier()
    if backend == "nli":
        return NliVerifier(model_name=model_name, batch_size=batch_size)
    raise ValueError("verifier backend must be 'nli' or 'lexical_smoke'")


def support_matrix(obligations: Sequence[str], documents: Sequence[str], verifier: Verifier) -> list[list[float]]:
    pairs = [(obligation, document) for obligation in obligations for document in documents]
    if hasattr(verifier, "score_many"):
        flat = [float(value) for value in verifier.score_many(pairs)]  # type: ignore[attr-defined]
    else:
        flat = [float(verifier.score(obligation, document)) for obligation, document in pairs]
    rows: list[list[float]] = []
    width = len(documents)
    for start in range(0, len(flat), width):
        rows.append(flat[start : start + width])
    return rows
