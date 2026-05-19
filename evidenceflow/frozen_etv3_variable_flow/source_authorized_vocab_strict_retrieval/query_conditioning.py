"""Query-surface conditioning for source-text certificate edges.

This module does not decompose a query into roles or propositions.  It only
keeps normalized query surface tokens after removing candidate-title anchors and
question function words, so certificate edges can be filtered before repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from .certificate_graph import EvidenceNode
from .normalize import title_aliases, tokens, unique_ints


QUERY_CONDITIONING_CONTRACT: Mapping[str, bool | str] = {
    "query_conditioning": "surface_token_conditioned_certificate_activation",
    "uses_llm_query_decomposition": False,
    "uses_role_vocabulary": False,
    "uses_proposition_support_evidence": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
}

QUERY_FUNCTION_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "with",
    }
)

COMPARISON_SURFACE_TOKENS = frozenset(
    {
        "after",
        "before",
        "earlier",
        "later",
        "larger",
        "less",
        "more",
        "older",
        "same",
        "than",
        "younger",
    }
)


@dataclass(frozen=True)
class QuerySignatures:
    query_tokens: Tuple[str, ...]
    relation_tokens: Tuple[str, ...]
    relation_stems: Tuple[str, ...]
    anchor_doc_indices: Tuple[int, ...]
    anchor_tokens: Tuple[str, ...]
    comparison_markers: Tuple[str, ...]
    wh_type: str
    title_mention_doc_indices: Tuple[int, ...]
    trace: Mapping[str, object]


def extract_query_signatures(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    query_mentioned_doc_indices: Sequence[int] = (),
) -> QuerySignatures:
    """Extract deterministic query signatures for edge activation."""

    query_tokens = tuple(tokens(query))
    node_lookup = {int(node.doc_index): node for node in nodes}
    candidates = tuple(unique_ints(candidate_doc_indices))
    explicit_mentions = set(unique_ints(query_mentioned_doc_indices))
    title_mentions = _title_mentions(
        query_tokens=query_tokens,
        nodes=node_lookup,
        candidate_doc_indices=candidates,
    )
    anchor_doc_indices = tuple(unique_ints([*explicit_mentions, *title_mentions]))
    anchor_tokens = _anchor_tokens(
        nodes=node_lookup,
        doc_indices=anchor_doc_indices,
        query_tokens=query_tokens,
    )
    anchor_token_set = set(anchor_tokens)
    relation_tokens = tuple(
        token
        for token in query_tokens
        if len(token) >= 3
        and token not in QUERY_FUNCTION_TOKENS
        and token not in anchor_token_set
    )
    relation_stems = tuple(_unique_strings(_token_stem(token) for token in relation_tokens))
    comparison_markers = tuple(token for token in query_tokens if token in COMPARISON_SURFACE_TOKENS)
    wh_type = _wh_type(query_tokens)
    trace = {
        **dict(QUERY_CONDITIONING_CONTRACT),
        "query_tokens": query_tokens,
        "relation_tokens": relation_tokens,
        "relation_stems": relation_stems,
        "anchor_doc_indices": anchor_doc_indices,
        "anchor_tokens": anchor_tokens,
        "comparison_markers": comparison_markers,
        "wh_type": wh_type,
        "title_mention_doc_indices": title_mentions,
    }
    return QuerySignatures(
        query_tokens=query_tokens,
        relation_tokens=relation_tokens,
        relation_stems=relation_stems,
        anchor_doc_indices=anchor_doc_indices,
        anchor_tokens=anchor_tokens,
        comparison_markers=comparison_markers,
        wh_type=wh_type,
        title_mention_doc_indices=title_mentions,
        trace=trace,
    )


def token_stems_for_text(text: object) -> Tuple[str, ...]:
    """Return deterministic stems for text tokens."""

    return tuple(_unique_strings(_token_stem(token) for token in tokens(text) if len(token) >= 3))


def query_relation_overlap(signatures: QuerySignatures, text: object) -> Tuple[str, ...]:
    """Return query relation stems present in text."""

    text_stems = set(token_stems_for_text(text))
    return tuple(stem for stem in signatures.relation_stems if stem in text_stems)


def _title_mentions(
    *,
    query_tokens: Sequence[str],
    nodes: Mapping[int, EvidenceNode],
    candidate_doc_indices: Sequence[int],
) -> Tuple[int, ...]:
    mentions = []
    for doc_index in candidate_doc_indices:
        node = nodes.get(int(doc_index))
        if node is None:
            continue
        for alias in title_aliases(node.display_title):
            alias_tokens = tokens(alias)
            if not alias_tokens or len(alias_tokens) > len(query_tokens):
                continue
            width = len(alias_tokens)
            for start in range(0, len(query_tokens) - width + 1):
                end = start + width
                if tuple(query_tokens[start:end]) == alias_tokens:
                    mentions.append((int(doc_index), start, end, width))

    output = []
    for doc_index, start, end, width in mentions:
        if any(
            other_width > width
            and other_start <= start
            and end <= other_end
            for _other_doc_index, other_start, other_end, other_width in mentions
        ):
            continue
        output.append(int(doc_index))
    return tuple(unique_ints(output))


def _anchor_tokens(
    *,
    nodes: Mapping[int, EvidenceNode],
    doc_indices: Sequence[int],
    query_tokens: Sequence[str],
) -> Tuple[str, ...]:
    query_token_set = set(query_tokens)
    output = []
    for doc_index in doc_indices:
        node = nodes.get(int(doc_index))
        if node is None:
            continue
        for alias in title_aliases(node.display_title):
            output.extend(token for token in tokens(alias) if token in query_token_set)
    return tuple(_unique_strings(output))


def _wh_type(query_tokens: Sequence[str]) -> str:
    for token in ("who", "where", "when", "which", "what", "whose", "whom", "how"):
        if token in query_tokens:
            return token
    return ""


def _token_stem(token: object) -> str:
    item = str(token or "")
    for suffix in ("ing", "ers", "ors", "ed", "er", "or", "es", "s"):
        if item.endswith(suffix) and len(item) > len(suffix) + 3:
            return item[: -len(suffix)]
    return item


def _unique_strings(values) -> Tuple[str, ...]:
    output = []
    seen = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return tuple(output)
