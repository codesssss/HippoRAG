"""Configuration for the standalone AG-STO v12 retriever package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AGSTOPolicy = Literal["base", "graph"]
AGSTOSetSearchPolicy = Literal["beam", "mct"]


@dataclass(frozen=True)
class AGSTOConfig:
    """Frozen AG-STO v12 configuration.

    ``policy`` is the only method-level switch:

    - ``graph`` is the final AG-STO v12 graph-obligated completion policy.
    - ``base`` is retained only for the clean no-completion ablation.

    The remaining fields are execution bounds, not new method variants.
    """

    policy: AGSTOPolicy = "graph"
    retrieval_top_k: int = 20
    evidence_set_size: int = 5
    stable_anchor_k: int = 2
    proposal_candidate_depth: int = 10
    support_proposal_depth: int = 6
    candidate_limit: int = 120
    beam_size: int = 12
    set_search_policy: AGSTOSetSearchPolicy = "beam"
    max_endpoint_degree: int = 30

    def __post_init__(self) -> None:
        if self.policy not in {"base", "graph"}:
            raise ValueError(
                f"Unsupported AG-STO policy={self.policy!r}. "
                "Use 'base' or 'graph'. Diagnostic policies are not part of the clean API."
            )
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be >= 1")
        if self.evidence_set_size < 1:
            raise ValueError("evidence_set_size must be >= 1")
        if self.stable_anchor_k < 0:
            raise ValueError("stable_anchor_k must be >= 0")
        if self.proposal_candidate_depth < 1:
            raise ValueError("proposal_candidate_depth must be >= 1")
        if self.support_proposal_depth < 1:
            raise ValueError("support_proposal_depth must be >= 1")
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be >= 1")
        if self.beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if self.set_search_policy not in {"beam", "mct"}:
            raise ValueError("set_search_policy must be 'beam' or 'mct'")
        if self.max_endpoint_degree < 2:
            raise ValueError("max_endpoint_degree must be >= 2")

    @property
    def completion_policy(self) -> str:
        """Map the public AG-STO policy to the internal completion policy."""

        return "graph_obligated" if self.policy == "graph" else "none"
