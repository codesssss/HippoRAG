"""Shared schemas for DAEC-DAPG experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BindingAssignment:
    """A retained latent binding assignment and its explicit prior."""

    slots: dict[str, str]
    prior: float
    raw_score: float

    @property
    def key(self) -> str:
        if not self.slots:
            return "empty"
        return "|".join(f"{slot}={value}" for slot, value in sorted(self.slots.items()))


@dataclass(frozen=True)
class Demand:
    """Minimal retrieval-active demand representation."""

    demand_id: str
    text: str
    role: str = "lookup"
    depends_on: tuple[str, ...] = ()
    weight: float = 1.0


@dataclass(frozen=True)
class EvidenceDocument:
    """Document payload used by local DAPG smoke scripts."""

    doc_id: str
    title: str
    text: str
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}".strip()
