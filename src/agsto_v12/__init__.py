"""Standalone public API for AG-STO v12 retrieval.

AG-STO is treated as an independent retrieval method. This package intentionally
exposes only the clean v12 mainline and its base ablation:

- ``graph``: stable anchor entry + STO proposals + graph-constrained evidence-set
  selection + conservative graph-obligated completion.
- ``base``: the same retriever without graph-obligated completion, retained for
  ablation only.

Diagnostic probes such as answer-obligation, answer-role-gap, role-path,
reader-ordering, and native/support weight variants remain in analysis scripts
and are not part of this API.

Native proposal generation is available from ``agsto_v12.proposals`` for
implementation diagnostics, but is intentionally not exported at the package
top level.
"""

from .config import AGSTOConfig
from .completion import apply_graph_obligated_completion
from .index import build_corpus_unit_index
from .lexical import rank_docs_bm25, score_docs_bm25
from .ranking import rank_sto_proposal_consensus_docs, unique_ranked
from .retriever import AGSTORetriever
from .scoring import score_anchor_guided_evidence_set

__all__ = [
    "AGSTOConfig",
    "AGSTORetriever",
    "apply_graph_obligated_completion",
    "build_corpus_unit_index",
    "rank_docs_bm25",
    "rank_sto_proposal_consensus_docs",
    "score_anchor_guided_evidence_set",
    "score_docs_bm25",
    "unique_ranked",
]
