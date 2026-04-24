"""Query-conditioned Backward Flow retrieval operators.

This module is intentionally standalone: it reads prepared HippoRAG fact/doc
objects and produces retrieval-only rankings without changing the existing
HippoRAG retrieval path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse

from .prompts.linking import get_query_instruction
from .utils.misc_utils import text_processing


@dataclass(frozen=True)
class TerminalSchema:
    label: str
    phrase: str
    matched_rule: str


@dataclass
class QBFResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    trace: Dict[str, Any]


_SCHEMA_RULES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("country", "country nationality citizenship", ("nationality", "citizenship", "country", "countries")),
    ("birthplace", "place of birth birthplace born in", ("birthplace", "place of birth", "born in", "born at")),
    ("date", "date year time when", ("when", "date", "year", "day", "month")),
    ("spouse", "spouse wife husband married", ("spouse", "wife", "husband", "married")),
    ("author", "author writer wrote written by", ("author", "writer", "wrote", "written by")),
    ("director", "director directed by filmmaker", ("director", "directed", "filmmaker")),
    ("publisher", "publisher published by publication", ("publisher", "published by", "publication")),
    ("capital", "capital city", ("capital",)),
    ("award", "award prize winner won", ("award", "prize", "winner", "won")),
    ("location", "location place located in", ("where", "location", "located", "place")),
    ("occupation", "occupation profession job", ("occupation", "profession", "job")),
    ("parent", "parent father mother child", ("father", "mother", "parent", "child")),
)


def infer_terminal_schema(query: str) -> TerminalSchema:
    """Map a question to a terminal evidence schema phrase.

    The rule list is deliberately small and relation-type oriented. It predicts
    the last-hop evidence type, not the final entity/title.
    """

    normalized = _normalize_text(query)
    for label, phrase, cues in _SCHEMA_RULES:
        for cue in cues:
            if cue in normalized:
                return TerminalSchema(label=label, phrase=phrase, matched_rule=cue)

    tail = normalized
    if "?" in tail:
        tail = tail.rsplit("?", 1)[0]
    tokens = tail.split()
    phrase = " ".join(tokens[-8:]) if tokens else normalized
    return TerminalSchema(label="fallback", phrase=phrase or str(query), matched_rule="fallback_tail")


def min_max(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    arr = np.where(finite, arr, 0.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def stable_topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(int(k), arr.size)
    order = np.lexsort((np.arange(arr.size), -arr))
    return order[:k].astype(int)


def rerank_doc_ids_by_scores(base_doc_ids: Sequence[int],
                             rerank_scores: np.ndarray,
                             top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rerank a baseline list by a secondary score with baseline-rank tie break."""

    base = [int(doc_id) for doc_id in base_doc_ids]
    if not base or top_k <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    items = [
        (int(doc_id), float(rerank_scores[int(doc_id)]) if int(doc_id) < len(rerank_scores) else 0.0, rank)
        for rank, doc_id in enumerate(base)
    ]
    items.sort(key=lambda item: (-item[1], item[2]))
    selected = items[:top_k]
    return (
        np.asarray([doc_id for doc_id, _, _ in selected], dtype=int),
        np.asarray([score for _, score, _ in selected], dtype=np.float64),
    )


class QBFFactEntityIndex:
    """Sparse fact/entity(/doc) index used by QBF operators."""

    def __init__(
        self,
        *,
        fact_ids: Sequence[str],
        fact_triples: Sequence[Tuple[str, str, str]],
        fact_embeddings: np.ndarray,
        source_fact_indices: Sequence[int],
        fact_doc_indices: Sequence[Sequence[int]],
        num_docs: int,
    ) -> None:
        self.fact_ids = [str(fact_id) for fact_id in fact_ids]
        self.fact_triples = [tuple(str(part) for part in triple) for triple in fact_triples]
        self.fact_embeddings = _as_2d_float(fact_embeddings)
        self.source_fact_indices = np.asarray(source_fact_indices, dtype=int)
        self.fact_doc_indices = [tuple(int(idx) for idx in doc_idxs) for doc_idxs in fact_doc_indices]
        self.num_docs = int(num_docs)

        if len(self.fact_ids) != len(self.fact_triples):
            raise ValueError("fact_ids and fact_triples length mismatch")
        if len(self.fact_ids) != self.fact_embeddings.shape[0]:
            raise ValueError("fact_ids and fact_embeddings length mismatch")
        if len(self.fact_ids) != len(self.source_fact_indices):
            raise ValueError("fact_ids and source_fact_indices length mismatch")
        if len(self.fact_ids) != len(self.fact_doc_indices):
            raise ValueError("fact_ids and fact_doc_indices length mismatch")

        self.entity_to_idx: Dict[str, int] = {}
        fact_rows: List[int] = []
        entity_cols: List[int] = []
        doc_rows: List[int] = []
        doc_cols: List[int] = []
        for fact_idx, triple in enumerate(self.fact_triples):
            for entity in (triple[0], triple[2]):
                norm_entity = _normalize_text(entity)
                if not norm_entity:
                    continue
                entity_idx = self.entity_to_idx.setdefault(norm_entity, len(self.entity_to_idx))
                fact_rows.append(fact_idx)
                entity_cols.append(entity_idx)
            for doc_idx in self.fact_doc_indices[fact_idx]:
                if 0 <= int(doc_idx) < self.num_docs:
                    doc_rows.append(fact_idx)
                    doc_cols.append(int(doc_idx))

        self.num_facts = len(self.fact_ids)
        self.num_entities = len(self.entity_to_idx)
        self.fact_entity = sparse.csr_matrix(
            (np.ones(len(fact_rows), dtype=np.float64), (fact_rows, entity_cols)),
            shape=(self.num_facts, self.num_entities),
            dtype=np.float64,
        )
        self.entity_fact = self.fact_entity.T.tocsr()
        self.fact_doc = sparse.csr_matrix(
            (np.ones(len(doc_rows), dtype=np.float64), (doc_rows, doc_cols)),
            shape=(self.num_facts, self.num_docs),
            dtype=np.float64,
        )
        self.doc_fact = self.fact_doc.T.tocsr()
        self.fact_entity_degree = np.maximum(np.asarray(self.fact_entity.sum(axis=1)).ravel(), 1.0)
        self.fact_doc_degree = np.maximum(np.asarray(self.fact_doc.sum(axis=1)).ravel(), 1.0)

    @classmethod
    def from_hipporag(cls, hipporag: Any) -> "QBFFactEntityIndex":
        """Build a QBF index from a prepared HippoRAG instance."""

        fact_ids: List[str] = []
        fact_triples: List[Tuple[str, str, str]] = []
        fact_embeddings: List[np.ndarray] = []
        source_fact_indices: List[int] = []
        fact_doc_indices: List[Sequence[int]] = []

        fact_store = getattr(hipporag, "v2_base_fact_embedding_store", None) or hipporag.fact_embedding_store
        fact_id_to_triple = dict(getattr(hipporag, "fact_id_to_triple", {}) or {})
        fact_id_to_doc_idxs = getattr(hipporag, "fact_id_to_doc_idxs", {}) or {}
        all_fact_ids = list(getattr(hipporag, "fact_node_keys", []) or [])
        all_fact_embeddings = _as_2d_float(getattr(hipporag, "fact_embeddings", np.zeros((0, 0))))

        missing_triple_ids = [
            str(fact_id)
            for fact_id in all_fact_ids
            if str(fact_id) not in fact_id_to_triple
        ]
        if missing_triple_ids:
            fact_id_to_triple.update(_load_fact_triples_from_store(fact_store, missing_triple_ids))

        for source_idx, fact_id in enumerate(all_fact_ids):
            fact_id = str(fact_id)
            triple = fact_id_to_triple.get(fact_id)
            docs = sorted(int(doc_idx) for doc_idx in fact_id_to_doc_idxs.get(fact_id, set()))
            if triple is None or len(triple) != 3 or not docs:
                continue
            if source_idx >= all_fact_embeddings.shape[0]:
                continue
            normalized_triple = tuple(_normalize_text(part) for part in triple)
            if not all(normalized_triple):
                continue
            fact_ids.append(fact_id)
            fact_triples.append(normalized_triple)
            fact_embeddings.append(np.asarray(all_fact_embeddings[source_idx], dtype=np.float64))
            source_fact_indices.append(source_idx)
            fact_doc_indices.append(tuple(docs))

        if fact_embeddings:
            fact_embedding_matrix = np.vstack(fact_embeddings)
        else:
            fact_embedding_matrix = np.zeros((0, 0), dtype=np.float64)

        return cls(
            fact_ids=fact_ids,
            fact_triples=fact_triples,
            fact_embeddings=fact_embedding_matrix,
            source_fact_indices=source_fact_indices,
            fact_doc_indices=fact_doc_indices,
            num_docs=len(getattr(hipporag, "passage_node_keys", []) or []),
        )

    def fact_query_scores_from_full(self, full_scores: np.ndarray) -> np.ndarray:
        full = np.asarray(full_scores, dtype=np.float64).reshape(-1)
        scores = np.zeros(self.num_facts, dtype=np.float64)
        valid = (self.source_fact_indices >= 0) & (self.source_fact_indices < full.size)
        scores[valid] = full[self.source_fact_indices[valid]]
        return min_max(scores)

    def score_terminal_schema(self,
                              embedding_model: Any,
                              schema_phrase: str,
                              *,
                              random_seed_key: str | None = None) -> np.ndarray:
        if self.num_facts == 0:
            return np.array([], dtype=np.float64)
        if random_seed_key is not None:
            return _deterministic_random_scores(self.num_facts, random_seed_key)

        schema_embedding = embedding_model.batch_encode(
            [str(schema_phrase)],
            instruction=get_query_instruction("query_to_fact"),
            norm=True,
        )[0]
        schema_embedding = np.asarray(schema_embedding, dtype=np.float64).reshape(-1)
        scores = self.fact_embeddings @ schema_embedding.T
        return min_max(scores)

    def doc_max_fact_scores(self, fact_scores: np.ndarray) -> np.ndarray:
        fact_scores = np.asarray(fact_scores, dtype=np.float64).reshape(-1)
        doc_scores = np.zeros(self.num_docs, dtype=np.float64)
        if fact_scores.size == 0:
            return doc_scores
        for fact_idx, score in enumerate(fact_scores):
            if score <= 0:
                continue
            for doc_idx in self.fact_doc_indices[fact_idx]:
                if 0 <= doc_idx < self.num_docs and score > doc_scores[doc_idx]:
                    doc_scores[doc_idx] = float(score)
        return doc_scores

    def qbf_retrieve(self,
                     *,
                     query_fact_scores: np.ndarray,
                     fact_chi: np.ndarray,
                     top_k: int,
                     alpha: float = 0.5,
                     max_iter: int = 50,
                     tol: float = 1e-6,
                     psi_floor: float = 0.05,
                     include_doc_nodes: bool = False) -> QBFResult:
        doc_scores, trace = self.qbf_doc_scores(
            query_fact_scores=query_fact_scores,
            fact_chi=fact_chi,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            psi_floor=psi_floor,
            include_doc_nodes=include_doc_nodes,
        )
        sorted_doc_ids = stable_topk_indices(doc_scores, top_k)
        sorted_doc_scores = doc_scores[sorted_doc_ids] if sorted_doc_ids.size else np.array([], dtype=np.float64)
        trace.update(
            {
                "operator": "qbf_schema_all_edge" if include_doc_nodes else "qbf_schema_relational",
                "positive_doc_count": int(np.count_nonzero(doc_scores > 0)),
            }
        )
        return QBFResult(sorted_doc_ids=sorted_doc_ids, sorted_doc_scores=sorted_doc_scores, trace=trace)

    def qbf_doc_scores(self,
                       *,
                       query_fact_scores: np.ndarray,
                       fact_chi: np.ndarray,
                       alpha: float = 0.5,
                       max_iter: int = 50,
                       tol: float = 1e-6,
                       psi_floor: float = 0.05,
                       include_doc_nodes: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        fact_query = self.fact_query_scores_from_full(query_fact_scores)
        fact_chi = min_max(np.asarray(fact_chi, dtype=np.float64))
        beta_fact, trace = self.backward_potential(
            fact_chi=fact_chi,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            psi_floor=psi_floor,
            include_doc_nodes=include_doc_nodes,
        )
        fact_scores = min_max(fact_query) * min_max(beta_fact)
        doc_scores = self.doc_max_fact_scores(fact_scores)
        trace.update(
            {
                "top_fact_score": float(np.max(fact_scores)) if fact_scores.size else 0.0,
            }
        )
        return doc_scores, trace

    def backward_potential(self,
                           *,
                           fact_chi: np.ndarray,
                           alpha: float = 0.5,
                           max_iter: int = 50,
                           tol: float = 1e-6,
                           psi_floor: float = 0.05,
                           include_doc_nodes: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        fact_chi = min_max(np.asarray(fact_chi, dtype=np.float64).reshape(-1))
        if fact_chi.size != self.num_facts:
            raise ValueError(f"fact_chi has {fact_chi.size} entries, expected {self.num_facts}")

        alpha = float(np.clip(alpha, 1e-6, 1.0))
        psi_floor = float(np.clip(psi_floor, 0.0, 1.0))
        fact_psi = psi_floor + (1.0 - psi_floor) * fact_chi

        entity_chi = _safe_divide(self.entity_fact @ fact_chi, np.maximum(np.asarray(self.entity_fact.sum(axis=1)).ravel(), 1.0))
        fact_beta = fact_chi.copy()
        entity_beta = entity_chi.copy()

        if include_doc_nodes:
            doc_chi = _safe_divide(self.doc_fact @ fact_chi, np.maximum(np.asarray(self.doc_fact.sum(axis=1)).ravel(), 1.0))
            doc_beta = doc_chi.copy()
            fact_total_degree = np.maximum(self.fact_entity_degree + self.fact_doc_degree, 1.0)
        else:
            doc_chi = np.zeros(self.num_docs, dtype=np.float64)
            doc_beta = np.zeros(self.num_docs, dtype=np.float64)
            fact_total_degree = self.fact_entity_degree

        converged = False
        delta = 0.0
        iterations = 0
        for iterations in range(1, int(max_iter) + 1):
            fact_neighbor = self.fact_entity @ entity_beta
            if include_doc_nodes:
                fact_neighbor = fact_neighbor + (self.fact_doc @ doc_beta)
            fact_neighbor = _safe_divide(fact_neighbor, fact_total_degree)

            entity_num = self.entity_fact @ (fact_beta * fact_psi)
            entity_den = self.entity_fact @ fact_psi
            entity_neighbor = _safe_divide(entity_num, entity_den)

            new_fact_beta = alpha * fact_chi + (1.0 - alpha) * fact_neighbor
            new_entity_beta = alpha * entity_chi + (1.0 - alpha) * entity_neighbor

            if include_doc_nodes:
                doc_num = self.doc_fact @ (fact_beta * fact_psi)
                doc_den = self.doc_fact @ fact_psi
                doc_neighbor = _safe_divide(doc_num, doc_den)
                new_doc_beta = alpha * doc_chi + (1.0 - alpha) * doc_neighbor
                delta = max(
                    float(np.max(np.abs(new_fact_beta - fact_beta))) if fact_beta.size else 0.0,
                    float(np.max(np.abs(new_entity_beta - entity_beta))) if entity_beta.size else 0.0,
                    float(np.max(np.abs(new_doc_beta - doc_beta))) if doc_beta.size else 0.0,
                )
                doc_beta = np.clip(new_doc_beta, 0.0, 1.0)
            else:
                delta = max(
                    float(np.max(np.abs(new_fact_beta - fact_beta))) if fact_beta.size else 0.0,
                    float(np.max(np.abs(new_entity_beta - entity_beta))) if entity_beta.size else 0.0,
                )

            fact_beta = np.clip(new_fact_beta, 0.0, 1.0)
            entity_beta = np.clip(new_entity_beta, 0.0, 1.0)
            if delta <= tol:
                converged = True
                break

        return fact_beta, {
            "qbf_iterations": int(iterations),
            "qbf_converged": bool(converged),
            "qbf_delta": float(delta),
            "qbf_alpha": float(alpha),
            "qbf_psi_floor": float(psi_floor),
            "qbf_include_doc_nodes": bool(include_doc_nodes),
            "qbf_fact_count": int(self.num_facts),
            "qbf_entity_count": int(self.num_entities),
        }


def _normalize_text(value: Any) -> str:
    try:
        normalized = text_processing(str(value))
    except Exception:
        normalized = str(value).lower()
    return re.sub(r"\s+", " ", str(normalized)).strip()


def _as_2d_float(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 0:
        return arr.reshape(0, 0)
    return arr


def _load_fact_triples_from_store(fact_store: Any, fact_ids: Sequence[str]) -> Dict[str, Tuple[str, str, str]]:
    triples: Dict[str, Tuple[str, str, str]] = {}
    if fact_store is None or not fact_ids:
        return triples
    try:
        rows = fact_store.get_rows(list(fact_ids))
    except Exception:
        return triples
    for fact_id, row in rows.items():
        content = row.get("content", "") if isinstance(row, Mapping) else ""
        parsed = _parse_fact_content(content)
        if parsed is not None:
            triples[str(fact_id)] = tuple(_normalize_text(part) for part in parsed)
    return triples


def _parse_fact_content(content: Any) -> Tuple[str, str, str] | None:
    if isinstance(content, (list, tuple)) and len(content) == 3:
        return tuple(str(item) for item in content)
    text = str(content or "").strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            return tuple(str(item) for item in parsed)
    return None


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray | float) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)
    return out


def _deterministic_random_scores(size: int, seed_key: str) -> np.ndarray:
    values = np.zeros(int(size), dtype=np.float64)
    for idx in range(int(size)):
        digest = hashlib.md5(f"{seed_key}:{idx}".encode("utf-8")).hexdigest()
        values[idx] = int(digest[:12], 16) / float(16 ** 12 - 1)
    return values
