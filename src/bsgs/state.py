"""Belief-state graph data structures for BSGS.

BSGS does not maintain a full posterior over evidence sets.  It tracks the
node-level marginal

    b_t(p) = sum_{S_t contains p} P(S_t | q, o_{1:t})

which is a tractable approximation to DAEC-style set posterior composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Any


EPS = 1e-12


@dataclass
class PropositionNode:
    prop_id: str
    text: str
    source_doc_id: str
    source_span: str
    mentions: list[str] = field(default_factory=list)
    support_prob: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefStateGraph:
    propositions: dict[str, PropositionNode]
    belief: dict[str, float]
    edges: dict[tuple[str, str], float]

    @classmethod
    def uniform(
        cls,
        propositions: dict[str, PropositionNode],
        edges: dict[tuple[str, str], float] | None = None,
    ) -> "BeliefStateGraph":
        belief = {pid: 1.0 for pid in propositions}
        graph = cls(propositions=dict(propositions), belief=belief, edges=dict(edges or {}))
        graph.normalize()
        return graph

    def normalize(self) -> None:
        """Normalize belief mass over known proposition ids.

        If all current mass is non-positive, fall back to a uniform distribution.
        This keeps every update step well-defined during diagnostic runs.
        """
        for prop_id in self.propositions:
            self.belief.setdefault(prop_id, 0.0)
        for prop_id in list(self.belief):
            if prop_id not in self.propositions:
                del self.belief[prop_id]

        total = sum(max(0.0, float(v)) for v in self.belief.values())
        if not self.propositions:
            self.belief.clear()
            return
        if total <= EPS:
            mass = 1.0 / len(self.propositions)
            self.belief = {prop_id: mass for prop_id in self.propositions}
            return
        self.belief = {
            prop_id: max(0.0, float(self.belief.get(prop_id, 0.0))) / total
            for prop_id in self.propositions
        }

    def add_propositions(self, new_props: list[PropositionNode]) -> None:
        """Add propositions with zero initial mass, preserving existing belief."""
        for prop in new_props:
            self.propositions[prop.prop_id] = prop
            self.belief.setdefault(prop.prop_id, 0.0)
        self.normalize()

    def predict(self, transition: dict[tuple[str, str], float]) -> dict[str, float]:
        """Return predicted belief under a row-stochastic sparse transition."""
        predicted = {prop_id: 0.0 for prop_id in self.propositions}
        outgoing: dict[str, list[tuple[str, float]]] = {}
        for (src, dst), weight in transition.items():
            if src in self.propositions and dst in self.propositions and weight > 0:
                outgoing.setdefault(src, []).append((dst, float(weight)))

        for src, mass in self.belief.items():
            if mass <= 0:
                continue
            row = outgoing.get(src)
            if not row:
                predicted[src] = predicted.get(src, 0.0) + mass
                continue
            row_sum = sum(weight for _, weight in row)
            if row_sum <= EPS:
                predicted[src] = predicted.get(src, 0.0) + mass
                continue
            for dst, weight in row:
                predicted[dst] = predicted.get(dst, 0.0) + mass * weight / row_sum

        total = sum(predicted.values())
        if total <= EPS and predicted:
            uniform = 1.0 / len(predicted)
            return {prop_id: uniform for prop_id in predicted}
        if total > EPS:
            predicted = {prop_id: value / total for prop_id, value in predicted.items()}
        return predicted

    def observe(self, likelihood: dict[str, float]) -> None:
        """Apply an observation likelihood and renormalize in place."""
        self.belief = {
            prop_id: self.belief.get(prop_id, 0.0) * max(0.0, float(likelihood.get(prop_id, 0.0)))
            for prop_id in self.propositions
        }
        self.normalize()

    def entropy(self) -> float:
        return -sum(p * log(p) for p in self.belief.values() if p > EPS)

    def topk(self, k: int, support_weighted: bool = False) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        for prop_id, mass in self.belief.items():
            score = mass
            if support_weighted:
                score *= max(0.0, float(self.propositions[prop_id].support_prob))
            scores.append((prop_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[: max(0, k)]
