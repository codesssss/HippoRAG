"""Deterministic query-demand graph helpers for source-text GraphRAG.

This module keeps demand extraction deliberately small: it derives answer-slot
tokens from the question surface and checks whether candidate documents expose
those tokens in their title or OpenIE facts.  It does not call an LLM, use a
typed schema, learn weights, or route by dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .certificate_graph import EvidenceNode
from .normalize import tokens, unique_ints


QUERY_DEMAND_CONTRACT: Mapping[str, bool | str] = {
    "query_demand": "deterministic_answer_slot_surface_demand",
    "uses_llm_query_compiler": False,
    "uses_typed_schema": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
}


DEMAND_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
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

PHRASE_BOUNDARY_TOKENS = frozenset(
    {
        "by",
        "for",
        "from",
        "in",
        "near",
        "of",
        "on",
        "that",
        "where",
        "which",
        "who",
        "whose",
        "with",
    }
)

ANSWER_ROLE_TOKENS = frozenset(
    {
        "actor",
        "artist",
        "author",
        "bridge",
        "city",
        "composer",
        "country",
        "county",
        "date",
        "designer",
        "director",
        "district",
        "father",
        "figure",
        "founder",
        "husband",
        "location",
        "mother",
        "name",
        "person",
        "place",
        "producer",
        "region",
        "state",
        "term",
        "terms",
        "wife",
        "writer",
        "year",
    }
)


@dataclass(frozen=True)
class QueryDemandGraph:
    answer_tokens: Tuple[str, ...]
    role_tokens: Tuple[str, ...]
    trace: Mapping[str, object]

    @property
    def active_tokens(self) -> Tuple[str, ...]:
        return tuple(unique_ordered_strings([*self.answer_tokens, *self.role_tokens]))


@dataclass(frozen=True)
class DemandCoverage:
    covered_tokens: Tuple[str, ...]
    covering_doc_indices: Tuple[int, ...]
    coverage_count: int
    covering_doc_count: int


def build_query_demand_graph(query: str) -> QueryDemandGraph:
    """Build a deterministic answer-demand graph from question text."""

    raw_query_tokens = tokens(query)
    query_tokens = _content_tokens(query)
    answer_tokens: List[str] = []

    answer_tokens.extend(_name_of_phrase_tokens(raw_query_tokens))
    if not answer_tokens:
        answer_tokens.extend(token for token in query_tokens if token in ANSWER_ROLE_TOKENS)

    if not answer_tokens:
        answer_tokens.extend(_wh_answer_tokens(raw_query_tokens))

    answer_tokens = list(unique_ordered_strings(answer_tokens))
    role_tokens = [
        token
        for token in query_tokens
        if token not in set(answer_tokens) and token in ANSWER_ROLE_TOKENS
    ]
    trace = {
        **dict(QUERY_DEMAND_CONTRACT),
        "answer_tokens": tuple(answer_tokens),
        "role_tokens": tuple(unique_ordered_strings(role_tokens)),
    }
    return QueryDemandGraph(
        answer_tokens=tuple(answer_tokens),
        role_tokens=tuple(unique_ordered_strings(role_tokens)),
        trace=trace,
    )


def demand_root_doc_indices(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    limit: int,
) -> Tuple[int, ...]:
    """Return candidate roots whose title/facts expose the answer demand."""

    demand = build_query_demand_graph(query)
    if not demand.answer_tokens:
        return ()
    node_lookup = {int(node.doc_index): node for node in nodes}
    ranked: List[Tuple[int, int, int]] = []
    for rank, doc_index in enumerate(unique_ints(candidate_doc_indices)):
        node = node_lookup.get(int(doc_index))
        if node is None:
            continue
        coverage = demand_coverage_for_docs(
            demand=demand,
            nodes=node_lookup,
            doc_indices=(int(doc_index),),
        )
        if coverage.coverage_count <= 0:
            continue
        ranked.append((-int(coverage.coverage_count), int(rank), int(doc_index)))
    ranked.sort()
    return tuple(doc_index for _negative_coverage, _rank, doc_index in ranked[: max(int(limit), 0)])


def demand_coverage_for_docs(
    *,
    demand: QueryDemandGraph,
    nodes: Mapping[int, EvidenceNode],
    doc_indices: Sequence[int],
) -> DemandCoverage:
    demand_tokens = set(demand.answer_tokens)
    if not demand_tokens:
        return DemandCoverage((), (), 0, 0)

    covered: Dict[str, int] = {}
    covering_docs: List[int] = []
    for doc_index in unique_ints(doc_indices):
        node = nodes.get(int(doc_index))
        if node is None:
            continue
        doc_tokens = _node_demand_tokens(node)
        hits = sorted(demand_tokens & doc_tokens)
        if not hits:
            continue
        covering_docs.append(int(doc_index))
        for token in hits:
            covered.setdefault(str(token), int(doc_index))
    return DemandCoverage(
        covered_tokens=tuple(sorted(covered)),
        covering_doc_indices=tuple(unique_ints(covering_docs)),
        coverage_count=len(covered),
        covering_doc_count=len(unique_ints(covering_docs)),
    )


def unique_ordered_strings(values: Sequence[str]) -> Tuple[str, ...]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return tuple(output)


def _content_tokens(text: object) -> Tuple[str, ...]:
    return tuple(
        token
        for token in tokens(text)
        if len(token) >= 3 and token not in DEMAND_STOP_TOKENS
    )


def _name_of_phrase_tokens(query_tokens: Sequence[str]) -> Tuple[str, ...]:
    output: List[str] = []
    for index, token in enumerate(query_tokens):
        if token != "name":
            continue
        cursor = index + 1
        while cursor < len(query_tokens) and query_tokens[cursor] in {"of", "the", "a", "an"}:
            cursor += 1
        while cursor < len(query_tokens):
            candidate = query_tokens[cursor]
            if candidate in PHRASE_BOUNDARY_TOKENS:
                break
            if candidate not in DEMAND_STOP_TOKENS:
                output.append(candidate)
            cursor += 1
        if output:
            break
    return tuple(output)


def _wh_answer_tokens(query_tokens: Sequence[str]) -> Tuple[str, ...]:
    query_set = set(query_tokens)
    if "when" in query_set:
        return tuple(token for token in ("date", "year", "established", "formed", "born", "died") if token in query_set or token in {"date", "year"})
    if "where" in query_set:
        return tuple(token for token in ("location", "place", "city", "country", "county", "district", "state", "region") if token in query_set or token in {"location", "place"})
    if query_set & {"who", "whom", "whose"}:
        return tuple(token for token in ("person", "father", "mother", "husband", "wife", "composer", "director", "writer", "producer", "figure") if token in query_set or token == "person")
    if "long" in query_set:
        return tuple(token for token in ("duration", "term", "terms", "year", "years") if token in query_set or token in {"duration", "year"})
    return ()


def _node_demand_tokens(node: EvidenceNode) -> set[str]:
    text_parts: List[str] = [node.display_title]
    for triple in node.triples:
        if len(triple) != 3:
            continue
        text_parts.extend([str(triple[0]), str(triple[1]), str(triple[2])])
    return set(_content_tokens(" ".join(text_parts)))
