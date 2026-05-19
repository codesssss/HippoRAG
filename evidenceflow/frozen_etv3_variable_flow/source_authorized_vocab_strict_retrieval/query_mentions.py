"""Query-visible title mention resolution for source frontier seeds."""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

from .certificate_graph import EvidenceNode
from .normalize import is_specific_title_alias, title_aliases, tokens, unique_ints


def query_mentioned_doc_indices(
    *,
    query: str,
    nodes: Mapping[int, EvidenceNode],
    candidate_doc_indices: Sequence[int],
) -> Tuple[int, ...]:
    """Return candidate docs whose titles are maximal mentions in the query.

    A short title mention is ignored when it is strictly contained in a longer
    candidate title mention over the same query span.  This prevents a document
    titled "Warsaw" from becoming a query source only because the query mentions
    "Warsaw Pact".
    """

    query_tokens = tokens(query)
    mentions: List[Tuple[int, int, int, int]] = []
    for doc_index in candidate_doc_indices:
        node = nodes.get(int(doc_index))
        if node is None:
            continue
        for alias in title_aliases(node.display_title):
            if not is_specific_title_alias(alias):
                continue
            alias_tokens = tokens(alias)
            if not alias_tokens or len(alias_tokens) > len(query_tokens):
                continue
            width = len(alias_tokens)
            for start in range(0, len(query_tokens) - width + 1):
                end = start + width
                if tuple(query_tokens[start:end]) == alias_tokens:
                    mentions.append((int(doc_index), start, end, width))

    maximal_doc_indices: List[int] = []
    for doc_index, start, end, width in mentions:
        if any(
            other_width > width
            and other_start <= start
            and end <= other_end
            for _other_doc_index, other_start, other_end, other_width in mentions
        ):
            continue
        maximal_doc_indices.append(int(doc_index))
    return unique_ints(maximal_doc_indices)
