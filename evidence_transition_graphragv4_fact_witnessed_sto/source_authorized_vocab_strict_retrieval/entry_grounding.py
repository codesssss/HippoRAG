"""Query-to-graph entry grounding for source-text GraphRAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from .certificate_graph import EvidenceNode
from .normalize import unique_ints
from .query_demand import build_query_demand_graph, demand_root_doc_indices
from .query_mentions import query_mentioned_doc_indices


ENTRY_GROUNDING_CONTRACT: Mapping[str, bool | str] = {
    "entry_grounding": "dense_head_plus_query_title_mention_entries",
    "dense_topk_is_final_answer_default": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
}


@dataclass(frozen=True)
class QueryEntryGrounding:
    query_mentioned_doc_indices: Tuple[int, ...]
    dense_entry_doc_indices: Tuple[int, ...]
    entry_doc_indices: Tuple[int, ...]
    root_candidate_doc_indices: Tuple[int, ...]
    trace: Mapping[str, object]


def ground_query_entries(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    dense_entry_count: int = 1,
    root_candidate_count: int = 5,
) -> QueryEntryGrounding:
    """Ground a query to graph entry nodes without making dense top5 final.

    Dense rank is allowed to expose possible graph roots.  The final evidence
    set is selected later by graph closure, not initialized as dense top5.
    """

    candidates = unique_ints(candidate_doc_indices)
    node_lookup = {int(node.doc_index): node for node in nodes}
    query_mentions = query_mentioned_doc_indices(
        query=str(query),
        nodes=node_lookup,
        candidate_doc_indices=candidates,
    )
    query_demand = build_query_demand_graph(str(query))
    demand_roots = demand_root_doc_indices(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=candidates,
        limit=max(int(root_candidate_count), 1),
    )
    dense_entries = tuple(candidates[: max(int(dense_entry_count), 0)])
    root_window = set(candidates[: max(int(root_candidate_count), 1)])
    query_mentioned_entries = tuple(
        doc_index for doc_index in query_mentions if int(doc_index) in root_window
    )
    entries = tuple(unique_ints([*dense_entries, *query_mentioned_entries]))
    root_candidates = tuple(unique_ints([*query_mentions, *candidates[: max(int(root_candidate_count), 1)]]))
    trace = {
        **dict(ENTRY_GROUNDING_CONTRACT),
        "query_mentioned_doc_indices": query_mentions,
        "query_mentioned_entry_doc_indices": query_mentioned_entries,
        "query_demand": dict(query_demand.trace),
        "demand_root_doc_indices": demand_roots,
        "dense_entry_doc_indices": dense_entries,
        "entry_doc_indices": entries,
        "root_candidate_doc_indices": root_candidates,
        "dense_entry_count": max(int(dense_entry_count), 0),
        "root_candidate_count": max(int(root_candidate_count), 1),
    }
    return QueryEntryGrounding(
        query_mentioned_doc_indices=query_mentions,
        dense_entry_doc_indices=dense_entries,
        entry_doc_indices=entries,
        root_candidate_doc_indices=root_candidates,
        trace=trace,
    )
