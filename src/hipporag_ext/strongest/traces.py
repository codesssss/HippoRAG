from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.hipporag.prompts.linking import get_query_instruction
from src.hipporag.utils.misc_utils import QuerySolution, compute_mdhash_id, min_max_normalize, text_processing

from .types import StrongestTraceState


@dataclass
class HippoHeadTrace:
    query: str
    retrieval_mode: str
    passage_prior_by_doc: Dict[str, float]
    entity_prior_by_text: Dict[str, float]
    query_fact_scores: np.ndarray | None = None
    top_k_fact_indices: List[int] | None = None
    top_k_facts: List[Tuple] | None = None


def normalize_entity_text(entity: str) -> str:
    processed = text_processing(entity)
    if not isinstance(processed, str):
        return ""
    return " ".join(processed.split())


def prefix_queries_with_instruction(queries: Sequence[str], instruction: str) -> List[str]:
    normalized_instruction = str(instruction or "").strip()
    if not normalized_instruction:
        return [str(query) for query in queries]
    return [f"Instruct: {normalized_instruction}\nQuery: {str(query)}" for query in queries]


def attach_hippo_head_trace_hooks(system) -> None:
    if getattr(system, "_strongest_head_trace_hooks_attached", False):
        return

    system._strongest_head_trace_hooks_attached = True
    system._strongest_head_traces = {}
    system._strongest_last_query_fact_scores = None
    system._strongest_last_top_k_fact_indices = None
    system._strongest_last_top_k_facts = None
    system._strongest_last_graph_dense = None
    system._strongest_inside_graph_search = False

    original_get_fact_scores = system.get_fact_scores
    original_rerank_facts = system.rerank_facts
    original_dense_passage_retrieval = system.dense_passage_retrieval
    original_graph_search = system.graph_search_with_fact_entities

    def traced_get_fact_scores(self, query: str):
        scores = original_get_fact_scores(query)
        self._strongest_last_query_fact_scores = np.asarray(scores, dtype=np.float32)
        return scores

    def traced_rerank_facts(self, query: str, query_fact_scores: np.ndarray):
        top_k_fact_indices, top_k_facts, rerank_log = original_rerank_facts(query, query_fact_scores)
        self._strongest_last_top_k_fact_indices = list(top_k_fact_indices)
        self._strongest_last_top_k_facts = list(top_k_facts)
        return top_k_fact_indices, top_k_facts, rerank_log

    def traced_dense_passage_retrieval(self, query: str):
        sorted_doc_ids, sorted_doc_scores = original_dense_passage_retrieval(query)
        sorted_doc_ids = np.asarray(sorted_doc_ids, dtype=np.int64)
        sorted_doc_scores = np.asarray(sorted_doc_scores, dtype=np.float32)

        dense_prior_by_doc: Dict[str, float] = {}
        for doc_id, score in zip(sorted_doc_ids.tolist(), sorted_doc_scores.tolist()):
            doc_text = self.chunk_embedding_store.get_row(self.passage_node_keys[int(doc_id)])["content"]
            dense_prior_by_doc[doc_text] = float(score)

        if getattr(self, "_strongest_inside_graph_search", False):
            self._strongest_last_graph_dense = dense_prior_by_doc
        else:
            self._strongest_head_traces[query] = HippoHeadTrace(
                query=query,
                retrieval_mode="dense_fallback",
                passage_prior_by_doc=dense_prior_by_doc,
                entity_prior_by_text={},
                query_fact_scores=(
                    None
                    if self._strongest_last_query_fact_scores is None
                    else np.array(self._strongest_last_query_fact_scores, copy=True)
                ),
                top_k_fact_indices=None,
                top_k_facts=None,
            )

        return sorted_doc_ids, sorted_doc_scores

    def traced_graph_search_with_fact_entities(
        self,
        query: str,
        link_top_k: int,
        query_fact_scores: np.ndarray,
        top_k_facts: List[Tuple],
        top_k_fact_indices: List[str],
        passage_node_weight: float = 0.05,
    ):
        self._strongest_inside_graph_search = True
        self._strongest_last_graph_dense = None
        try:
            sorted_doc_ids, sorted_doc_scores = original_graph_search(
                query=query,
                link_top_k=link_top_k,
                query_fact_scores=query_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=passage_node_weight,
            )
        finally:
            self._strongest_inside_graph_search = False

        phrase_scores: Dict[str, List[float]] = {}
        phrase_weights = np.zeros(len(self.graph.vs["name"]), dtype=np.float32)
        number_of_occurs = np.zeros(len(self.graph.vs["name"]), dtype=np.float32)
        phrases_and_ids = set()

        query_fact_scores_np = np.asarray(query_fact_scores, dtype=np.float32)
        for rank, fact in enumerate(top_k_facts):
            fact_score = (
                float(query_fact_scores_np[top_k_fact_indices[rank]])
                if query_fact_scores_np.ndim > 0
                else float(query_fact_scores_np)
            )
            for phrase in [fact[0].lower(), fact[2].lower()]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key)
                if phrase_id is not None:
                    weighted_fact_score = fact_score
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])
                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1.0
                phrases_and_ids.add((phrase, phrase_id))

        valid_phrase_mask = number_of_occurs > 0
        phrase_weights[valid_phrase_mask] /= number_of_occurs[valid_phrase_mask]

        for phrase, phrase_id in phrases_and_ids:
            if phrase_id is None:
                continue
            phrase_scores.setdefault(phrase, []).append(float(phrase_weights[phrase_id]))

        linking_score_map = {phrase: float(np.mean(scores)) for phrase, scores in phrase_scores.items()}
        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k, phrase_weights, linking_score_map)

        entity_prior_by_text: Dict[str, float] = {}
        for phrase, score in linking_score_map.items():
            normalized_phrase = normalize_entity_text(phrase)
            if not normalized_phrase:
                continue
            entity_prior_by_text[normalized_phrase] = max(
                entity_prior_by_text.get(normalized_phrase, 0.0),
                float(score),
            )

        dense_prior_by_doc = {}
        if self._strongest_last_graph_dense is not None:
            dense_prior_by_doc = {
                doc: float(score) * float(passage_node_weight)
                for doc, score in self._strongest_last_graph_dense.items()
            }

        self._strongest_head_traces[query] = HippoHeadTrace(
            query=query,
            retrieval_mode="graph",
            passage_prior_by_doc=dense_prior_by_doc,
            entity_prior_by_text=entity_prior_by_text,
            query_fact_scores=np.array(query_fact_scores_np, copy=True),
            top_k_fact_indices=list(top_k_fact_indices),
            top_k_facts=list(top_k_facts),
        )

        return sorted_doc_ids, sorted_doc_scores

    system.get_fact_scores = MethodType(traced_get_fact_scores, system)
    system.rerank_facts = MethodType(traced_rerank_facts, system)
    system.dense_passage_retrieval = MethodType(traced_dense_passage_retrieval, system)
    system.graph_search_with_fact_entities = MethodType(traced_graph_search_with_fact_entities, system)


def load_instruction_aware_query_embeddings(
    embedding_model,
    queries: Sequence[str],
    instruction: str | None = None,
) -> np.ndarray:
    prefixed_queries = prefix_queries_with_instruction(
        list(queries),
        instruction or get_query_instruction("query_to_passage"),
    )
    encoded = embedding_model.batch_encode(prefixed_queries, norm=True)
    return np.asarray(encoded, dtype=np.float32)


def build_solution_rank_cache(
    query_solutions: Sequence[QuerySolution],
    doc_to_idx: Dict[str, int],
) -> Tuple[List[List[int]], List[Dict[int, float]]]:
    ranked_doc_indices: List[List[int]] = []
    score_maps: List[Dict[int, float]] = []

    for solution in query_solutions:
        indices: List[int] = []
        score_map: Dict[int, float] = {}
        doc_scores = (
            np.asarray(solution.doc_scores, dtype=np.float32)
            if solution.doc_scores is not None
            else np.zeros(len(solution.docs), dtype=np.float32)
        )
        for doc, score in zip(solution.docs, doc_scores):
            doc_idx = doc_to_idx.get(doc)
            if doc_idx is None:
                continue
            indices.append(doc_idx)
            score_map[doc_idx] = float(score)
        ranked_doc_indices.append(indices)
        score_maps.append(score_map)
    return ranked_doc_indices, score_maps


def build_true_head_trace_cache(
    queries: Sequence[str],
    baseline_system,
    doc_to_idx: Dict[str, int],
    entity_vocab: Sequence[str],
    raw_traces: Dict[str, HippoHeadTrace] | None = None,
) -> List[StrongestTraceState]:
    entity_to_idx = {normalize_entity_text(entity): idx for idx, entity in enumerate(entity_vocab)}
    num_docs = len(doc_to_idx)
    if raw_traces is None:
        raw_traces = getattr(baseline_system, "_strongest_head_traces", {})

    aligned_traces: List[StrongestTraceState] = []
    for query in queries:
        raw_trace = raw_traces.get(query)
        passage_prior = np.zeros(num_docs, dtype=np.float32)
        entity_prior = np.zeros(len(entity_vocab), dtype=np.float32)
        retrieval_mode = "missing"

        if raw_trace is not None:
            retrieval_mode = raw_trace.retrieval_mode
            for doc, score in raw_trace.passage_prior_by_doc.items():
                doc_idx = doc_to_idx.get(doc)
                if doc_idx is None:
                    continue
                passage_prior[doc_idx] = max(passage_prior[doc_idx], float(score))

            for entity_text, score in raw_trace.entity_prior_by_text.items():
                entity_idx = entity_to_idx.get(normalize_entity_text(entity_text))
                if entity_idx is None:
                    continue
                entity_prior[entity_idx] = max(entity_prior[entity_idx], float(score))

        aligned_traces.append(
            StrongestTraceState(
                query=query,
                retrieval_mode=retrieval_mode,
                passage_prior=passage_prior,
                entity_prior=entity_prior,
                raw_trace=raw_trace,
            )
        )

    return aligned_traces
