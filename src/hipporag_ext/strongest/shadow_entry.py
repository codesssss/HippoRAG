from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
from scipy import sparse as sp

from src.hipporag.utils.misc_utils import min_max_normalize

from .runtime import l2_normalize_rows, run_strongest_sidecar
from .traces import normalize_entity_text
from .types import (
    StrongestBaselineState,
    StrongestConfig,
    StrongestResult,
    StrongestTraceState,
)


def _build_entity_vocab_for_pool(
    pool_doc_ids: Sequence[int | None],
    doc_idx_to_entities: Dict[int, Set[str]],
    seed_entities: Iterable[str],
    query_entities: Iterable[str],
) -> list[str]:
    entity_vocab = {
        normalize_entity_text(entity)
        for entity in list(seed_entities) + list(query_entities)
        if normalize_entity_text(entity)
    }
    for doc_id in pool_doc_ids:
        if doc_id is None:
            continue
        for entity in doc_idx_to_entities.get(int(doc_id), set()):
            normalized = normalize_entity_text(entity)
            if normalized:
                entity_vocab.add(normalized)
    return sorted(entity_vocab)


def _build_pool_incidence(
    pool_doc_ids: Sequence[int | None],
    entity_vocab: Sequence[str],
    doc_idx_to_entities: Dict[int, Set[str]],
) -> sp.csr_matrix:
    entity_to_idx = {entity: idx for idx, entity in enumerate(entity_vocab)}
    rows = []
    cols = []
    data = []
    for local_idx, doc_id in enumerate(pool_doc_ids):
        if doc_id is None:
            continue
        for entity in doc_idx_to_entities.get(int(doc_id), set()):
            normalized = normalize_entity_text(entity)
            entity_idx = entity_to_idx.get(normalized)
            if entity_idx is None:
                continue
            rows.append(int(entity_idx))
            cols.append(int(local_idx))
            data.append(1.0)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(entity_vocab), len(pool_doc_ids)), dtype=np.float32)


def _build_pool_trace_state(
    query: str,
    pool_doc_scores: np.ndarray,
    entity_vocab: Sequence[str],
    seed_entities: Iterable[str],
    query_entities: Iterable[str],
) -> StrongestTraceState:
    seed_set = {
        normalize_entity_text(entity)
        for entity in list(seed_entities) + list(query_entities)
        if normalize_entity_text(entity)
    }
    passage_prior = min_max_normalize(np.asarray(pool_doc_scores, dtype=np.float32))
    entity_prior = np.zeros(len(entity_vocab), dtype=np.float32)
    for idx, entity in enumerate(entity_vocab):
        if entity in seed_set:
            entity_prior[idx] = 1.0
    retrieval_mode = "graph" if np.any(entity_prior > 0) else "dense_fallback"
    return StrongestTraceState(
        query=query,
        retrieval_mode=retrieval_mode,
        passage_prior=passage_prior.astype(np.float32),
        entity_prior=entity_prior,
        raw_trace=None,
    )


def _normalize_triples(raw_triples: Sequence[object]) -> List[tuple[str, str, str]]:
    triples: List[tuple[str, str, str]] = []
    for triple in raw_triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject = normalize_entity_text(str(triple[0]))
        relation = " ".join(str(triple[1] or "").strip().lower().split())
        obj = normalize_entity_text(str(triple[2]))
        if not subject or not relation or not obj:
            continue
        triples.append((subject, relation, obj))
    return triples


def _get_chunk_triples_map(hipporag) -> Dict[str, List[tuple[str, str, str]]]:
    cached = getattr(hipporag, "_strongest_chunk_triples_map", None)
    if isinstance(cached, dict):
        return cached

    if not hasattr(hipporag, "load_existing_openie"):
        return {}

    try:
        all_openie_info, _chunk_keys_to_process = hipporag.load_existing_openie([])
    except Exception:
        return {}

    chunk_triples_map: Dict[str, List[tuple[str, str, str]]] = {}
    for doc in all_openie_info or []:
        chunk_id = doc.get("idx")
        if chunk_id is None:
            continue
        chunk_triples_map[str(chunk_id)] = _normalize_triples(doc.get("extracted_triples", []) or [])
    setattr(hipporag, "_strongest_chunk_triples_map", chunk_triples_map)
    return chunk_triples_map


def _build_passage_chunk_ids_for_pool(
    hipporag,
    pool_doc_ids: Sequence[int | None],
) -> List[str | None]:
    passage_node_keys = list(getattr(hipporag, "passage_node_keys", []) or [])
    if not passage_node_keys and hasattr(hipporag, "passage_node_key_to_doc_idx"):
        inverse = {
            int(doc_idx): str(chunk_id)
            for chunk_id, doc_idx in (getattr(hipporag, "passage_node_key_to_doc_idx", {}) or {}).items()
            if doc_idx is not None
        }
        return [inverse.get(int(doc_id)) if doc_id is not None else None for doc_id in pool_doc_ids]

    chunk_ids: List[str | None] = []
    for doc_id in pool_doc_ids:
        if doc_id is None:
            chunk_ids.append(None)
            continue
        if int(doc_id) < 0 or int(doc_id) >= len(passage_node_keys):
            chunk_ids.append(None)
            continue
        chunk_ids.append(str(passage_node_keys[int(doc_id)]))
    return chunk_ids


def _extract_doc_title(doc_text: str) -> str:
    lines = str(doc_text or "").splitlines()
    return lines[0].strip() if lines else str(doc_text or "").strip()


def run_strongest_shadow_for_pool(
    hipporag,
    query: str,
    pool_docs: Sequence[str],
    pool_doc_ids: Sequence[int | None],
    pool_doc_scores: np.ndarray,
    seed_entities: Iterable[str],
    query_entities: Iterable[str],
    config: StrongestConfig,
) -> StrongestResult | None:
    pool_docs = list(pool_docs)
    pool_doc_scores = np.asarray(pool_doc_scores, dtype=np.float32)
    if not pool_docs or pool_doc_scores.size == 0:
        return None

    passage_query_embedding = None
    if hasattr(hipporag, "query_to_embedding"):
        passage_query_embedding = (((getattr(hipporag, "query_to_embedding", {}) or {}).get("passage", {}) or {}).get(query))
    if passage_query_embedding is None and hasattr(hipporag, "_get_passage_query_embeddings"):
        hipporag._get_passage_query_embeddings([query])
        passage_query_embedding = (((getattr(hipporag, "query_to_embedding", {}) or {}).get("passage", {}) or {}).get(query))
    if passage_query_embedding is None:
        return None

    passage_embeddings = getattr(hipporag, "passage_embeddings", None)
    if passage_embeddings is None:
        return None
    passage_embeddings = np.asarray(passage_embeddings, dtype=np.float32)
    if passage_embeddings.ndim != 2 or passage_embeddings.shape[0] == 0:
        return None

    embedding_dim = int(passage_embeddings.shape[1])
    local_passage_embeddings = np.zeros((len(pool_docs), embedding_dim), dtype=np.float32)
    for local_idx, doc_id in enumerate(pool_doc_ids):
        if doc_id is None or int(doc_id) < 0 or int(doc_id) >= len(passage_embeddings):
            continue
        local_passage_embeddings[local_idx] = passage_embeddings[int(doc_id)]
    local_passage_embeddings = l2_normalize_rows(local_passage_embeddings)

    entity_vocab = _build_entity_vocab_for_pool(
        pool_doc_ids=pool_doc_ids,
        doc_idx_to_entities=getattr(hipporag, "doc_idx_to_structure_entities", {}) or {},
        seed_entities=seed_entities,
        query_entities=query_entities,
    )
    incidence = _build_pool_incidence(
        pool_doc_ids=pool_doc_ids,
        entity_vocab=entity_vocab,
        doc_idx_to_entities=getattr(hipporag, "doc_idx_to_structure_entities", {}) or {},
    )
    trace_state = _build_pool_trace_state(
        query=query,
        pool_doc_scores=pool_doc_scores,
        entity_vocab=entity_vocab,
        seed_entities=seed_entities,
        query_entities=query_entities,
    )

    pool_limit = len(pool_docs)
    pool_order = np.argsort(np.asarray(pool_doc_scores, dtype=np.float32))[::-1].tolist()
    passage_chunk_ids = _build_passage_chunk_ids_for_pool(hipporag=hipporag, pool_doc_ids=pool_doc_ids)
    passage_titles = [_extract_doc_title(doc_text) for doc_text in pool_docs]
    passage_texts = [str(doc_text) for doc_text in pool_docs]
    doc_idx_to_entities = getattr(hipporag, "doc_idx_to_structure_entities", {}) or {}
    passage_structure_entities = [
        sorted(
            {
                normalize_entity_text(entity)
                for entity in doc_idx_to_entities.get(int(doc_id), set())
                if doc_id is not None and normalize_entity_text(entity)
            }
        )
        if doc_id is not None
        else []
        for doc_id in pool_doc_ids
    ]
    baseline_prefix_local_indices = pool_order[: min(max(int(config.ras_prefix_guard_k), 0), pool_limit)]
    chunk_triples_cache = _get_chunk_triples_map(hipporag)
    chunk_triples_map = {
        str(chunk_id): list(chunk_triples_cache.get(str(chunk_id), []))
        for chunk_id in passage_chunk_ids
        if chunk_id is not None
    }
    state = StrongestBaselineState(
        query=query,
        docs=pool_docs,
        passage_query_embedding=np.asarray(passage_query_embedding, dtype=np.float32),
        passage_embeddings=local_passage_embeddings,
        smoothed_embeddings=np.array(local_passage_embeddings, copy=True),
        incidence=incidence,
        topology_redundancy_features=incidence.transpose().tocsr(),
        hippo_ranked_indices=pool_order,
        hippo_score_map={int(idx): float(pool_doc_scores[int(idx)]) for idx in range(pool_limit)},
        trace_state=trace_state,
        dense_scores=np.asarray(pool_doc_scores, dtype=np.float32),
        metadata={
            "entity_vocab_size": len(entity_vocab),
            "entity_vocab": list(entity_vocab),
            "pool_doc_ids": [None if doc_id is None else int(doc_id) for doc_id in pool_doc_ids],
            "passage_chunk_ids": list(passage_chunk_ids),
            "passage_titles": list(passage_titles),
            "passage_texts": list(passage_texts),
            "passage_structure_entities": list(passage_structure_entities),
            "chunk_triples_map": dict(chunk_triples_map),
            "baseline_prefix_local_indices": [int(idx) for idx in baseline_prefix_local_indices],
            "baseline_prefix_titles": [
                passage_titles[int(idx)]
                for idx in baseline_prefix_local_indices
                if 0 <= int(idx) < len(passage_titles)
            ],
            "baseline_prefix_doc_ids": [
                None if pool_doc_ids[int(idx)] is None else int(pool_doc_ids[int(idx)])
                for idx in baseline_prefix_local_indices
                if 0 <= int(idx) < len(pool_doc_ids)
            ],
            "query_text": str(query),
            "seed_entities": sorted(
                {
                    normalize_entity_text(entity)
                    for entity in seed_entities
                    if normalize_entity_text(entity)
                }
            ),
            "query_entities": sorted(
                {
                    normalize_entity_text(entity)
                    for entity in query_entities
                    if normalize_entity_text(entity)
                }
            ),
            "ras_extractor_mode": str(config.ras_extractor_mode),
            "ras_support_mode": str(config.ras_support_mode),
            "ras_embedding_probe_threshold": float(config.ras_embedding_probe_threshold),
            "_ras_embedding_model": getattr(hipporag, "embedding_model", None),
            "_ras_passage_embeddings": np.asarray(local_passage_embeddings, dtype=np.float32),
        },
    )
    return run_strongest_sidecar(state=state, config=config)
