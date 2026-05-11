"""Source-authorized candidate expansion for the standalone GraphRAG runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .certificate_graph import EvidenceNode
from .normalize import normalize_text, phrase_occurs, title_aliases, tokens, unique_ints
from .query_mentions import query_mentioned_doc_indices


SOURCE_TEXT_CANDIDATE_EXPANSION_CONTRACT: Mapping[str, bool | str] = {
    "candidate_expansion": "source_bound_endpoint_text_induction",
    "specific_endpoint_policy": "rarest_title_token_must_have_bounded_document_support",
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
    "uses_proposition_support_evidence": False,
}


@dataclass(frozen=True)
class SourceTextCandidateExpansion:
    initial_candidate_doc_indices: Tuple[int, ...]
    seed_doc_indices: Tuple[int, ...]
    source_bound_doc_indices: Tuple[int, ...]
    expanded_doc_indices: Tuple[int, ...]
    candidate_doc_indices: Tuple[int, ...]
    trace: Mapping[str, object]


@dataclass(frozen=True)
class SourceTextCandidateExpansionIndex:
    title_rare_token_index: Mapping[str, Tuple[int, ...]]
    max_specific_token_doc_count: int = 32


def build_source_text_candidate_expansion_index(
    nodes: Sequence[EvidenceNode],
    max_specific_token_doc_count: int = 32,
) -> SourceTextCandidateExpansionIndex:
    return SourceTextCandidateExpansionIndex(
        title_rare_token_index=_build_title_rare_token_index(nodes),
        max_specific_token_doc_count=max(int(max_specific_token_doc_count), 1),
    )


def expand_source_text_candidates(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    top_k: int = 5,
    expansion_index: SourceTextCandidateExpansionIndex | None = None,
) -> SourceTextCandidateExpansion:
    """Expand dense candidates with docs certified by seed source endpoints."""

    initial_candidates = unique_ints(candidate_doc_indices)
    node_lookup = {int(node.doc_index): node for node in nodes}
    seed_doc_indices = unique_ints(
        [
            *initial_candidates[: max(int(top_k), 1)],
            *query_mentioned_doc_indices(
                query=str(query),
                nodes=node_lookup,
                candidate_doc_indices=initial_candidates,
            ),
        ]
    )
    if expansion_index is None:
        expansion_index = build_source_text_candidate_expansion_index(nodes)
    title_rare_token_index = expansion_index.title_rare_token_index
    source_bound: List[int] = []
    for source_index in seed_doc_indices:
        source = node_lookup.get(int(source_index))
        if source is None:
            continue
        source_aliases = title_aliases(source.display_title)
        for endpoint in _source_bound_endpoints(source, source_aliases):
            endpoint_tokens = tokens(endpoint)
            if len(endpoint_tokens) < 2:
                continue
            candidate_targets = _candidate_targets_for_endpoint(
                endpoint=endpoint,
                title_rare_token_index=title_rare_token_index,
                max_specific_token_doc_count=expansion_index.max_specific_token_doc_count,
            )
            for target_index in candidate_targets:
                if int(target_index) == int(source_index):
                    continue
                target = node_lookup.get(int(target_index))
                if target is None:
                    continue
                if not phrase_occurs(endpoint, target.text):
                    continue
                source_bound.append(int(target_index))

    source_bound_doc_indices = tuple(
        doc_index
        for doc_index in unique_ints(source_bound)
        if int(doc_index) not in set(seed_doc_indices)
    )
    expanded_doc_indices = tuple(
        doc_index
        for doc_index in source_bound_doc_indices
        if int(doc_index) not in set(initial_candidates)
    )
    final_candidates = unique_ints([*initial_candidates, *expanded_doc_indices])
    trace = {
        **dict(SOURCE_TEXT_CANDIDATE_EXPANSION_CONTRACT),
        "initial_candidate_doc_count": len(initial_candidates),
        "seed_doc_indices": seed_doc_indices,
        "seed_doc_count": len(seed_doc_indices),
        "source_bound_doc_indices": source_bound_doc_indices,
        "source_bound_doc_count": len(source_bound_doc_indices),
        "expanded_doc_indices": expanded_doc_indices,
        "expanded_doc_count": len(expanded_doc_indices),
        "candidate_doc_count": len(final_candidates),
        "max_specific_token_doc_count": expansion_index.max_specific_token_doc_count,
    }
    return SourceTextCandidateExpansion(
        initial_candidate_doc_indices=initial_candidates,
        seed_doc_indices=seed_doc_indices,
        source_bound_doc_indices=source_bound_doc_indices,
        expanded_doc_indices=expanded_doc_indices,
        candidate_doc_indices=final_candidates,
        trace=trace,
    )


def _source_bound_endpoints(
    source: EvidenceNode,
    source_aliases: Sequence[str],
) -> Tuple[str, ...]:
    endpoints: List[str] = []
    for triple in source.triples:
        if len(triple) != 3:
            continue
        subject, _relation, obj = (str(triple[0]), str(triple[1]), str(triple[2]))
        if _matches_any_alias(subject, source_aliases):
            endpoints.append(normalize_text(obj))
        if _matches_any_alias(obj, source_aliases):
            endpoints.append(normalize_text(subject))
    return tuple(endpoint for endpoint in dict.fromkeys(endpoints) if endpoint)


def _matches_any_alias(text: object, aliases: Sequence[str]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    for alias in aliases:
        if not alias:
            continue
        if normalized == alias:
            return True
        if phrase_occurs(alias, normalized) or phrase_occurs(normalized, alias):
            return True
    return False


def _build_title_rare_token_index(nodes: Sequence[EvidenceNode]) -> Mapping[str, Tuple[int, ...]]:
    index: Dict[str, List[int]] = {}
    for node in nodes:
        for alias in title_aliases(node.display_title):
            for token in _rare_tokens(alias):
                index.setdefault(token, []).append(int(node.doc_index))
    return {token: tuple(unique_ints(indices)) for token, indices in index.items()}


def _candidate_targets_for_endpoint(
    *,
    endpoint: str,
    title_rare_token_index: Mapping[str, Sequence[int]],
    max_specific_token_doc_count: int,
) -> Tuple[int, ...]:
    candidate_lists = [
        (len(title_rare_token_index.get(token, ())), token, title_rare_token_index.get(token, ()))
        for token in _rare_tokens(endpoint)
        if title_rare_token_index.get(token)
    ]
    if not candidate_lists:
        return ()
    candidate_lists.sort(key=lambda item: (int(item[0]), str(item[1])))
    if int(candidate_lists[0][0]) > max(int(max_specific_token_doc_count), 1):
        return ()
    return tuple(unique_ints(candidate_lists[0][2]))


def _rare_tokens(text: object) -> Tuple[str, ...]:
    return tuple(token for token in tokens(text) if len(token) >= 4)
