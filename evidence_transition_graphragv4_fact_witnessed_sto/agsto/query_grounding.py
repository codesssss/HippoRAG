"""Query-to-STO endpoint grounding utilities.

The STO graph is entity-centric: documents are entered through title/entity
endpoints, while relation and answer-type words should guide evidence selection
rather than become independent graph anchors. These helpers keep that separation
closed-form and corpus-derived; they do not use prompt schemas, learned models,
gold labels, or dataset-specific rules.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from .index import content_tokens, normalize_text


QUERY_FUNCTION_TOKENS = {
    "answer",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "many",
    "much",
    "name",
    "find",
}


def _endpoint_degree(endpoint_to_docs: Mapping[str, Sequence[int]], endpoint: str) -> int:
    return len(endpoint_to_docs.get(str(endpoint), []) or [])


def _candidate_endpoints_for_tokens(
    *,
    tokens: Set[str],
    endpoint_to_docs: Mapping[str, Sequence[int]],
    endpoint_token_to_endpoints: Mapping[str, Sequence[str]] | None,
) -> List[str]:
    if not endpoint_token_to_endpoints:
        return [str(endpoint) for endpoint in endpoint_to_docs.keys()]
    candidates: Set[str] = set()
    for token in tokens:
        candidates.update(str(endpoint) for endpoint in endpoint_token_to_endpoints.get(str(token), []) or [])
    return sorted(endpoint for endpoint in candidates if endpoint in endpoint_to_docs)


def capitalized_content_tokens(query: str) -> Set[str]:
    """Return normalized query content tokens that are capitalized in surface form."""

    tokens: Set[str] = set()
    for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", str(query or "")):
        raw = match.group(0)
        first_alpha = next((char for char in raw if char.isalpha()), "")
        if not first_alpha or not first_alpha.isupper():
            continue
        tokens.update(content_tokens(raw))
    return tokens - QUERY_FUNCTION_TOKENS


def lexical_query_endpoints(
    *,
    query: str,
    endpoint_to_docs: Mapping[str, Sequence[int]],
    max_endpoint_degree: int,
    endpoint_token_to_endpoints: Mapping[str, Sequence[str]] | None = None,
) -> List[str]:
    """Ground query tokens to maximal non-hub STO endpoints."""

    query_tokens = content_tokens(query) - QUERY_FUNCTION_TOKENS
    matched: List[Tuple[str, Set[str]]] = []
    for endpoint in _candidate_endpoints_for_tokens(
        tokens=query_tokens,
        endpoint_to_docs=endpoint_to_docs,
        endpoint_token_to_endpoints=endpoint_token_to_endpoints,
    ):
        endpoint = str(endpoint)
        doc_indices = endpoint_to_docs.get(endpoint, []) or []
        endpoint_tokens = content_tokens(endpoint)
        if not endpoint_tokens or not endpoint_tokens.issubset(query_tokens):
            continue
        if len(doc_indices or []) > max_endpoint_degree:
            continue
        matched.append((endpoint, endpoint_tokens))

    maximal: List[str] = []
    for endpoint, endpoint_tokens in matched:
        if any(endpoint != other and endpoint_tokens < other_tokens for other, other_tokens in matched):
            continue
        maximal.append(endpoint)
    return sorted(set(maximal))


def named_anchor_query_endpoints(
    *,
    query: str,
    endpoint_to_docs: Mapping[str, Sequence[int]],
    max_endpoint_degree: int,
    title_endpoints: Set[str] | None = None,
    endpoint_token_to_endpoints: Mapping[str, Sequence[str]] | None = None,
) -> List[str]:
    """Keep endpoint anchors that are named mentions in the query surface."""

    lexical = lexical_query_endpoints(
        query=query,
        endpoint_to_docs=endpoint_to_docs,
        max_endpoint_degree=max_endpoint_degree,
        endpoint_token_to_endpoints=endpoint_token_to_endpoints,
    )
    capitalized_tokens = capitalized_content_tokens(query)
    if not capitalized_tokens:
        return []
    exact_anchors = exact_mentioned_query_endpoints(
        query=query,
        endpoint_to_docs=endpoint_to_docs,
        max_endpoint_degree=max_endpoint_degree,
        endpoint_token_to_endpoints=endpoint_token_to_endpoints,
    )
    if exact_anchors:
        return exact_anchors

    title_endpoint_set = {str(endpoint) for endpoint in title_endpoints or set() if str(endpoint).strip()}
    anchors = [
        endpoint
        for endpoint in lexical
        if content_tokens(endpoint) & capitalized_tokens
        and (not title_endpoint_set or endpoint in title_endpoint_set)
    ]
    return sorted(set(anchors), key=lambda endpoint: (_endpoint_degree(endpoint_to_docs, endpoint), endpoint))


def exact_mentioned_query_endpoints(
    *,
    query: str,
    endpoint_to_docs: Mapping[str, Sequence[int]],
    max_endpoint_degree: int,
    endpoint_token_to_endpoints: Mapping[str, Sequence[str]] | None = None,
) -> List[str]:
    """Ground endpoints whose normalized phrase appears in the query."""

    query_norm = f" {normalize_text(query)} "
    capitalized_tokens = capitalized_content_tokens(query)
    matched: List[Tuple[str, str]] = []
    for endpoint in _candidate_endpoints_for_tokens(
        tokens=capitalized_tokens,
        endpoint_to_docs=endpoint_to_docs,
        endpoint_token_to_endpoints=endpoint_token_to_endpoints,
    ):
        endpoint = str(endpoint)
        doc_indices = endpoint_to_docs.get(endpoint, []) or []
        if len(doc_indices or []) > max_endpoint_degree:
            continue
        endpoint_norm = normalize_text(endpoint)
        if not endpoint_norm or f" {endpoint_norm} " not in query_norm:
            continue
        if not (content_tokens(endpoint) & capitalized_tokens):
            continue
        matched.append((endpoint, endpoint_norm))

    maximal: List[str] = []
    for endpoint, endpoint_norm in matched:
        if any(
            endpoint != other and f" {endpoint_norm} " in f" {other_norm} "
            for other, other_norm in matched
        ):
            continue
        maximal.append(endpoint)
    return sorted(set(maximal), key=lambda endpoint: (_endpoint_degree(endpoint_to_docs, endpoint), endpoint))


def ground_query_endpoints(
    *,
    query: str,
    endpoint_to_docs: Mapping[str, Sequence[int]],
    max_endpoint_degree: int,
    prefer_named_anchors: bool = True,
    title_endpoints: Set[str] | None = None,
    endpoint_token_to_endpoints: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, Any]:
    """Ground entity-like query endpoints for symbolic STO graph entry.

    Lexical endpoint matches are exposed for audit, but they are not promoted
    to graph roots on their own.  Generic relation/answer words such as
    "branch", "house", or "military" can be useful for textual retrieval, but
    treating them as symbolic STO roots lets weak corpus endpoints displace
    actual evidence passages from the reader budget.
    """

    lexical = lexical_query_endpoints(
        query=query,
        endpoint_to_docs=endpoint_to_docs,
        max_endpoint_degree=max_endpoint_degree,
        endpoint_token_to_endpoints=endpoint_token_to_endpoints,
    )
    named = (
        named_anchor_query_endpoints(
            query=query,
            endpoint_to_docs=endpoint_to_docs,
            max_endpoint_degree=max_endpoint_degree,
            title_endpoints=title_endpoints,
            endpoint_token_to_endpoints=endpoint_token_to_endpoints,
        )
        if prefer_named_anchors
        else []
    )
    if named:
        selected = named
        mode = "named_anchor"
    else:
        selected = []
        mode = "lexical_observed_no_symbolic_anchor" if lexical else "none"
    return {
        "query_endpoints": selected,
        "lexical_query_endpoints": lexical,
        "named_anchor_query_endpoints": named,
        "query_grounding_mode": mode,
    }
