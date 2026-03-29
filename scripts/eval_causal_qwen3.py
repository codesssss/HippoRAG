import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import joblib
import numpy as np

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.utils.causal_utils import (
    normalize_structure_text,
    route_query_type,
    score_candidate_docs_by_structure,
)
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import QuerySolution, compute_mdhash_id, string_to_bool

LEARNED_SETWISE_FEATURE_NAMES = [
    "base_score",
    "rank_fraction",
    "reciprocal_rank",
    "selected_count_fraction",
    "seed_entity_overlap_ratio",
    "covered_entity_overlap_ratio",
    "selected_entity_overlap_ratio",
    "query_token_overlap_ratio",
    "novelty_score",
    "structure_score_seed",
    "structure_score_covered",
    "reachable_from_seed",
    "reachable_from_covered",
    "entity_count_norm",
    "edge_count_norm",
]


def get_gold_docs(samples: List, dataset_name: str = None, corpus: List | None = None) -> List:
    gold_docs = []
    corpus = corpus or []
    for sample in samples:
        if 'supporting_facts' in sample:
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        elif 'document' in sample and isinstance(sample['document'], dict) and corpus:
            document_id = str(sample['document'].get('id', '')).strip()
            gold_doc = [
                item['title'] + '\n' + item['text']
                for item in corpus
                if str(item.get('idx', '')).startswith(f"{document_id}_")
            ]
        else:
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample in samples:
        gold_ans = None
        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = list(set([sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']]))
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])
        gold_answers.append(list(gold_ans))
    return gold_answers


def build_doc_text_to_chunk_id(corpus: List[dict]) -> Dict[str, str]:
    doc_text_to_chunk_id: Dict[str, str] = {}
    for row in corpus:
        doc_text = f"{row['title']}\n{row['text']}"
        doc_text_to_chunk_id[doc_text] = compute_mdhash_id(doc_text, prefix="chunk-")
    return doc_text_to_chunk_id


def serialize_retrieved_doc_ids(retrieved_docs: List[str], doc_text_to_chunk_id: Dict[str, str]) -> List[str | None]:
    return [doc_text_to_chunk_id.get(doc_text) for doc_text in retrieved_docs]


def subset_by_indices(values: List, indices: List[int]) -> List:
    return [values[idx] for idx in indices]


def min_max_normalize_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=float)

    values = np.asarray(values, dtype=float)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        return np.ones_like(values, dtype=float) if max_value > 0 else np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def normalize_entity_set(entities: Sequence[str] | Set[str] | None) -> Set[str]:
    normalized_entities: Set[str] = set()
    for entity in entities or []:
        normalized = normalize_structure_text(str(entity))
        if normalized:
            normalized_entities.add(normalized)
    return normalized_entities


def collect_query_seed_entities(hipporag: HippoRAG, query: str) -> Set[str]:
    try:
        query_fact_scores = hipporag.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, _ = hipporag.rerank_facts(query, query_fact_scores)
        if hasattr(hipporag, "_collect_structure_seed_entities"):
            return normalize_entity_set(
                hipporag._collect_structure_seed_entities(
                    query_fact_scores=query_fact_scores,
                    top_k_fact_indices=top_k_fact_indices,
                    top_k_facts=top_k_facts,
                )
            )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to derive structure seed entities for query %r: %s",
            query[:160],
            exc,
        )
    return set()


def collect_lexical_query_seed_entities(query: str,
                                        pool_doc_ids: Sequence[int | None],
                                        doc_idx_to_entities: Dict[int, Set[str]],
                                        max_seed_entities: int = 8) -> Set[str]:
    normalized_query = normalize_structure_text(query)
    if not normalized_query:
        return set()

    query_tokens = {token for token in normalized_query.split() if len(token) > 1}
    scored_entities: List[Tuple[float, int, str]] = []
    seen_entities: Set[str] = set()
    for doc_id in pool_doc_ids:
        if doc_id is None:
            continue
        for entity in doc_idx_to_entities.get(int(doc_id), set()):
            normalized_entity = normalize_structure_text(entity)
            if not normalized_entity or normalized_entity in seen_entities:
                continue
            if len(normalized_entity) < 3:
                continue
            entity_tokens = {token for token in normalized_entity.split() if len(token) > 1}
            if not entity_tokens:
                continue
            overlap = len(entity_tokens & query_tokens)
            substring_match = (
                (len(normalized_entity) >= 4 and normalized_entity in normalized_query)
                or normalized_query in normalized_entity
            )
            if overlap <= 0 and not substring_match:
                continue
            score = 0.0
            if substring_match:
                score += 2.0
            score += overlap / max(1, len(entity_tokens))
            score += overlap / max(1, len(query_tokens))
            scored_entities.append((score, len(normalized_entity), normalized_entity))
            seen_entities.add(normalized_entity)

    scored_entities.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return {entity for _, _, entity in scored_entities[:max_seed_entities]}


def normalized_token_set(text: str | None) -> Set[str]:
    normalized_text = normalize_structure_text(text or "")
    if not normalized_text:
        return set()
    return {token for token in normalized_text.split() if len(token) > 1}


def compute_candidate_feature_rows(query: str,
                                   pool_docs: Sequence[str],
                                   pool_doc_ids: Sequence[int | None],
                                   pool_doc_scores: np.ndarray,
                                   doc_idx_to_entities: Dict[int, Set[str]],
                                   doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                   adjacency: Dict[str, List[Tuple[str, float, str]]],
                                   qa_top_k: int,
                                   selected_positions: Sequence[int] | None = None,
                                   seed_entities: Sequence[str] | Set[str] | None = None,
                                   structure_max_hops: int = 2,
                                   candidate_positions: Sequence[int] | None = None) -> List[Dict[str, float | int | str | None]]:
    pool_size = len(pool_doc_ids)
    normalized_base_scores = min_max_normalize_array(np.asarray(pool_doc_scores, dtype=float))
    selected_positions = list(selected_positions or [])
    candidate_positions = list(candidate_positions or range(pool_size))
    normalized_seed_entities = normalize_entity_set(seed_entities)
    query_tokens = normalized_token_set(query)

    selected_entity_union: Set[str] = set()
    for pos in selected_positions:
        if pos >= pool_size:
            continue
        doc_id = pool_doc_ids[pos]
        if doc_id is None:
            continue
        selected_entity_union.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))
    covered_entities = normalized_seed_entities | selected_entity_union

    candidate_doc_ids = [
        int(pool_doc_ids[pos])
        for pos in candidate_positions
        if pos < pool_size and pool_doc_ids[pos] is not None
    ]
    structure_scores_seed = (
        score_candidate_docs_by_structure(
            candidate_doc_ids=candidate_doc_ids,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            seed_entities=normalized_seed_entities,
            adjacency=adjacency,
            max_hops=structure_max_hops,
        )
        if candidate_doc_ids and normalized_seed_entities
        else {}
    )
    structure_scores_covered = (
        score_candidate_docs_by_structure(
            candidate_doc_ids=candidate_doc_ids,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            seed_entities=covered_entities,
            adjacency=adjacency,
            max_hops=structure_max_hops,
        )
        if candidate_doc_ids and covered_entities
        else {}
    )

    rows: List[Dict[str, float | int | str | None]] = []
    for pos in candidate_positions:
        if pos >= pool_size:
            continue
        doc_id = pool_doc_ids[pos]
        doc_entities = (
            normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
            if doc_id is not None
            else set()
        )
        doc_edges = doc_idx_to_edges.get(int(doc_id), []) if doc_id is not None else []
        title = pool_docs[pos].split("\n", 1)[0] if pos < len(pool_docs) else ""
        title_tokens = normalized_token_set(title)
        entity_count = len(doc_entities)
        edge_count = len(doc_edges)

        seed_overlap = len(doc_entities & normalized_seed_entities)
        covered_overlap = len(doc_entities & covered_entities)
        selected_overlap = len(doc_entities & selected_entity_union)
        novelty_score = len(doc_entities - covered_entities) / max(1, entity_count) if doc_entities else 0.0

        row = {
            "pool_position": int(pos),
            "doc_id": int(doc_id) if doc_id is not None else None,
            "title": title,
            "base_score": float(normalized_base_scores[pos]) if pos < len(normalized_base_scores) else 0.0,
            "rank_fraction": 1.0 - (float(pos) / max(1, pool_size - 1)) if pool_size > 1 else 1.0,
            "reciprocal_rank": 1.0 / float(pos + 1),
            "selected_count_fraction": len(selected_positions) / max(1, qa_top_k),
            "seed_entity_overlap_ratio": seed_overlap / max(1, len(normalized_seed_entities)) if normalized_seed_entities else 0.0,
            "covered_entity_overlap_ratio": covered_overlap / max(1, len(covered_entities)) if covered_entities else 0.0,
            "selected_entity_overlap_ratio": selected_overlap / max(1, len(selected_entity_union)) if selected_entity_union else 0.0,
            "query_token_overlap_ratio": len(title_tokens & query_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0,
            "novelty_score": novelty_score,
            "structure_score_seed": float(structure_scores_seed.get(int(doc_id), 0.0)) if doc_id is not None else 0.0,
            "structure_score_covered": float(structure_scores_covered.get(int(doc_id), 0.0)) if doc_id is not None else 0.0,
            "reachable_from_seed": float(structure_scores_seed.get(int(doc_id), 0.0) > 0.0) if doc_id is not None else 0.0,
            "reachable_from_covered": float(structure_scores_covered.get(int(doc_id), 0.0) > 0.0) if doc_id is not None else 0.0,
            "entity_count_norm": min(entity_count, 20) / 20.0,
            "edge_count_norm": min(edge_count, 20) / 20.0,
        }
        rows.append(row)

    return rows


def feature_rows_to_matrix(feature_rows: Sequence[Dict[str, float | int | str | None]]) -> np.ndarray:
    if not feature_rows:
        return np.zeros((0, len(LEARNED_SETWISE_FEATURE_NAMES)), dtype=float)
    matrix = np.asarray([
        [float(row.get(feature_name, 0.0) or 0.0) for feature_name in LEARNED_SETWISE_FEATURE_NAMES]
        for row in feature_rows
    ], dtype=float)
    return matrix


def predict_binary_scores(model_bundle: Dict[str, object], feature_matrix: np.ndarray) -> np.ndarray:
    if feature_matrix.size == 0:
        return np.zeros(0, dtype=float)

    model = model_bundle["model"]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_matrix)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return np.asarray(probabilities[:, 1], dtype=float)
        return np.asarray(probabilities, dtype=float).reshape(-1)

    decision_scores = np.asarray(model.decision_function(feature_matrix), dtype=float).reshape(-1)
    return 1.0 / (1.0 + np.exp(-decision_scores))


def load_learned_model_bundle(model_path: str) -> Dict[str, object]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid learned setwise model bundle at {model_path}")

    feature_names = bundle.get("feature_names")
    if feature_names and list(feature_names) != LEARNED_SETWISE_FEATURE_NAMES:
        raise ValueError(
            "Learned setwise model feature mismatch: "
            f"expected {LEARNED_SETWISE_FEATURE_NAMES}, got {feature_names}"
        )
    bundle.setdefault("feature_names", list(LEARNED_SETWISE_FEATURE_NAMES))
    bundle.setdefault("model_path", model_path)
    return bundle


def select_learned_greedy_positions(query: str,
                                    pool_docs: Sequence[str],
                                    pool_doc_ids: Sequence[int | None],
                                    pool_doc_scores: np.ndarray,
                                    doc_idx_to_entities: Dict[int, Set[str]],
                                    doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                    adjacency: Dict[str, List[Tuple[str, float, str]]],
                                    qa_top_k: int,
                                    learned_model_bundle: Dict[str, object],
                                    initial_seed_entities: Sequence[str] | Set[str] | None = None,
                                    anchor_count: int = 0,
                                    structure_max_hops: int = 2) -> Tuple[List[int], Dict[str, object]]:
    candidate_count = len(pool_doc_ids)
    if candidate_count == 0 or qa_top_k <= 0:
        return [], {
            "selection_steps": [],
            "anchor_positions": [],
            "seed_entity_count_initial": 0,
            "covered_entity_count_final": 0,
            "candidate_count": candidate_count,
        }

    target_k = min(candidate_count, qa_top_k)
    selected_positions: List[int] = []
    remaining_positions = list(range(candidate_count))
    seed_entities = normalize_entity_set(initial_seed_entities)
    selection_steps: List[Dict[str, object]] = []

    actual_anchor_count = min(max(anchor_count, 0), target_k)
    anchor_positions = remaining_positions[:actual_anchor_count]
    for anchor_pos in anchor_positions:
        selected_positions.append(anchor_pos)
        remaining_positions.remove(anchor_pos)
        selection_steps.append({
            "step": len(selection_steps) + 1,
            "mode": "anchor",
            "pool_position": int(anchor_pos),
            "doc_id": int(pool_doc_ids[anchor_pos]) if pool_doc_ids[anchor_pos] is not None else None,
            "predicted_score": 1.0,
        })

    while len(selected_positions) < target_k and remaining_positions:
        feature_rows = compute_candidate_feature_rows(
            query=query,
            pool_docs=pool_docs,
            pool_doc_ids=pool_doc_ids,
            pool_doc_scores=pool_doc_scores,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            adjacency=adjacency,
            qa_top_k=qa_top_k,
            selected_positions=selected_positions,
            seed_entities=seed_entities,
            structure_max_hops=structure_max_hops,
            candidate_positions=remaining_positions,
        )
        feature_matrix = feature_rows_to_matrix(feature_rows)
        predicted_scores = predict_binary_scores(learned_model_bundle, feature_matrix)

        best_idx = 0
        best_score = float("-inf")
        for idx, (row, predicted_score) in enumerate(zip(feature_rows, predicted_scores)):
            pool_position = int(row["pool_position"])
            if predicted_score > best_score + 1e-9 or (
                abs(predicted_score - best_score) <= 1e-9 and pool_position < int(feature_rows[best_idx]["pool_position"])
            ):
                best_idx = idx
                best_score = float(predicted_score)

        chosen_row = feature_rows[best_idx]
        chosen_pos = int(chosen_row["pool_position"])
        selected_positions.append(chosen_pos)
        remaining_positions.remove(chosen_pos)
        selection_steps.append({
            "step": len(selection_steps) + 1,
            "mode": "learned_greedy",
            "pool_position": chosen_pos,
            "doc_id": int(chosen_row["doc_id"]) if chosen_row["doc_id"] is not None else None,
            "predicted_score": round(float(best_score), 4),
            "base_score": round(float(chosen_row["base_score"]), 4),
            "structure_score_seed": round(float(chosen_row["structure_score_seed"]), 4),
            "structure_score_covered": round(float(chosen_row["structure_score_covered"]), 4),
            "novelty_score": round(float(chosen_row["novelty_score"]), 4),
        })

    covered_entities = seed_entities.copy()
    for pos in selected_positions:
        doc_id = pool_doc_ids[pos]
        if doc_id is None:
            continue
        covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))

    return selected_positions, {
        "selection_steps": selection_steps,
        "anchor_positions": [int(pos) for pos in anchor_positions],
        "seed_entity_count_initial": len(seed_entities),
        "covered_entity_count_final": len(covered_entities),
        "candidate_count": candidate_count,
    }


def select_bridge_greedy_positions(pool_doc_ids: Sequence[int | None],
                                   pool_doc_scores: np.ndarray,
                                   doc_idx_to_entities: Dict[int, Set[str]],
                                   doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                   adjacency: Dict[str, List[Tuple[str, float, str]]],
                                   qa_top_k: int,
                                   initial_seed_entities: Sequence[str] | Set[str] | None = None,
                                   anchor_count: int = 2,
                                   structure_max_hops: int = 2,
                                   base_weight: float = 0.25,
                                   structure_weight: float = 0.60,
                                   novelty_weight: float = 0.15) -> Tuple[List[int], Dict[str, object]]:
    candidate_count = len(pool_doc_ids)
    if candidate_count == 0 or qa_top_k <= 0:
        return [], {
            "selection_steps": [],
            "anchor_positions": [],
            "seed_entity_count_initial": 0,
            "covered_entity_count_final": 0,
            "candidate_count": candidate_count,
        }

    normalized_base_scores = min_max_normalize_array(np.asarray(pool_doc_scores, dtype=float))
    target_k = min(candidate_count, qa_top_k)
    selected_positions: List[int] = []
    remaining_positions = list(range(candidate_count))
    covered_entities = normalize_entity_set(initial_seed_entities)
    selection_steps: List[Dict[str, object]] = []

    actual_anchor_count = min(max(anchor_count, 0), target_k)
    anchor_positions = remaining_positions[:actual_anchor_count]
    for anchor_pos in anchor_positions:
        selected_positions.append(anchor_pos)
        doc_id = pool_doc_ids[anchor_pos]
        if doc_id is not None:
            covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))
        selection_steps.append({
            "step": len(selection_steps) + 1,
            "mode": "anchor",
            "pool_position": int(anchor_pos),
            "doc_id": int(doc_id) if doc_id is not None else None,
            "base_score": round(float(normalized_base_scores[anchor_pos]), 4),
            "structure_score": 0.0,
            "novelty_score": 0.0,
            "combined_score": round(float(normalized_base_scores[anchor_pos]), 4),
        })

    remaining_positions = [pos for pos in remaining_positions if pos not in set(selected_positions)]

    while len(selected_positions) < target_k and remaining_positions:
        candidate_doc_ids = [
            int(pool_doc_ids[pos])
            for pos in remaining_positions
            if pool_doc_ids[pos] is not None
        ]
        structure_scores = (
            score_candidate_docs_by_structure(
                candidate_doc_ids=candidate_doc_ids,
                doc_idx_to_entities=doc_idx_to_entities,
                doc_idx_to_edges=doc_idx_to_edges,
                seed_entities=covered_entities,
                adjacency=adjacency,
                max_hops=structure_max_hops,
            )
            if candidate_doc_ids and covered_entities
            else {}
        )

        best_pos = remaining_positions[0]
        best_combined = float("-inf")
        best_detail: Dict[str, object] = {}
        for pos in remaining_positions:
            doc_id = pool_doc_ids[pos]
            doc_entities = (
                normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
                if doc_id is not None
                else set()
            )
            novelty_score = (
                len(doc_entities - covered_entities) / max(1, len(doc_entities))
                if doc_entities
                else 0.0
            )
            structure_score = (
                float(structure_scores.get(int(doc_id), 0.0))
                if doc_id is not None
                else 0.0
            )
            base_score = float(normalized_base_scores[pos]) if pos < len(normalized_base_scores) else 0.0
            combined_score = (
                base_weight * base_score
                + structure_weight * structure_score
                + novelty_weight * novelty_score
            )
            if combined_score > best_combined + 1e-9 or (
                abs(combined_score - best_combined) <= 1e-9 and pos < best_pos
            ):
                best_pos = pos
                best_combined = combined_score
                best_detail = {
                    "base_score": round(base_score, 4),
                    "structure_score": round(structure_score, 4),
                    "novelty_score": round(float(novelty_score), 4),
                    "combined_score": round(float(combined_score), 4),
                }

        selected_positions.append(best_pos)
        remaining_positions.remove(best_pos)
        chosen_doc_id = pool_doc_ids[best_pos]
        if chosen_doc_id is not None:
            covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(chosen_doc_id), set())))
        selection_steps.append({
            "step": len(selection_steps) + 1,
            "mode": "greedy",
            "pool_position": int(best_pos),
            "doc_id": int(chosen_doc_id) if chosen_doc_id is not None else None,
            **best_detail,
        })

    return selected_positions, {
        "selection_steps": selection_steps,
        "anchor_positions": [int(pos) for pos in anchor_positions],
        "seed_entity_count_initial": len(normalize_entity_set(initial_seed_entities)),
        "covered_entity_count_final": len(covered_entities),
        "candidate_count": candidate_count,
    }


def apply_setwise_selector(hipporag: HippoRAG,
                           query_solutions: List[QuerySolution],
                           doc_text_to_chunk_id: Dict[str, str],
                           pool_k: int,
                           qa_top_k: int,
                           selector_name: str,
                           anchor_count: int,
                           structure_max_hops: int,
                           base_weight: float,
                           structure_weight: float,
                           novelty_weight: float,
                           learned_model_bundle: Dict[str, object] | None = None) -> Tuple[List[QuerySolution], Dict[str, object]]:
    logger = logging.getLogger(__name__)
    selector_name = str(selector_name).strip().lower()
    if selector_name not in {"bridge_greedy", "learned_greedy"}:
        raise ValueError(f"Unsupported setwise selector: {selector_name}")

    selected_solutions: List[QuerySolution] = []
    mapped_pool_doc_counts: List[int] = []
    seed_entity_counts: List[int] = []
    selected_structured_doc_counts: List[int] = []
    selector_examples: List[Dict[str, object]] = []

    chunk_text_to_hash = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}

    for qs in query_solutions:
        pool_limit = min(len(qs.docs), max(pool_k, qa_top_k))
        pool_docs = list(qs.docs[:pool_limit])
        if qs.doc_scores is not None and len(qs.doc_scores) >= pool_limit:
            pool_scores = np.asarray(qs.doc_scores[:pool_limit], dtype=float)
            tail_scores = np.asarray(qs.doc_scores[pool_limit:], dtype=float) if len(qs.doc_scores) > pool_limit else np.array([], dtype=float)
        else:
            pool_scores = np.linspace(pool_limit, 1, pool_limit, dtype=float)
            tail_scores = np.array([], dtype=float)

        pool_doc_ids: List[int | None] = []
        for doc_text in pool_docs:
            chunk_id = doc_text_to_chunk_id.get(doc_text) or chunk_text_to_hash.get(doc_text)
            mapped_doc_id = hipporag.passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
            pool_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)

        seed_entities = collect_query_seed_entities(hipporag, qs.question)
        if not seed_entities:
            seed_entities = collect_lexical_query_seed_entities(
                query=qs.question,
                pool_doc_ids=pool_doc_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )
        if selector_name == "bridge_greedy":
            selected_positions, selector_trace = select_bridge_greedy_positions(
                pool_doc_ids=pool_doc_ids,
                pool_doc_scores=pool_scores,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                adjacency=hipporag.structure_graph_out,
                qa_top_k=qa_top_k,
                initial_seed_entities=seed_entities,
                anchor_count=anchor_count,
                structure_max_hops=structure_max_hops,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
            )
        else:
            if learned_model_bundle is None:
                raise ValueError("learned_greedy selector requires a loaded model bundle")
            selected_positions, selector_trace = select_learned_greedy_positions(
                query=qs.question,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                pool_doc_scores=pool_scores,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                adjacency=hipporag.structure_graph_out,
                qa_top_k=qa_top_k,
                learned_model_bundle=learned_model_bundle,
                initial_seed_entities=seed_entities,
                anchor_count=anchor_count,
                structure_max_hops=structure_max_hops,
            )

        selected_position_set = set(selected_positions)
        reordered_pool_positions = selected_positions + [
            pos for pos in range(pool_limit)
            if pos not in selected_position_set
        ]
        reordered_docs = [pool_docs[pos] for pos in reordered_pool_positions] + list(qs.docs[pool_limit:])
        reordered_scores = np.concatenate(
            [
                np.asarray([pool_scores[pos] for pos in reordered_pool_positions], dtype=float),
                tail_scores,
            ]
        )

        retrieval_trace = dict(qs.retrieval_trace or {})
        retrieval_trace["setwise_selector"] = selector_name
        retrieval_trace["setwise_selector_trace"] = {
            "pool_k": int(pool_limit),
            "anchor_count": int(min(max(anchor_count, 0), min(pool_limit, qa_top_k))),
            "seed_entities_preview": sorted(seed_entities)[:12],
            "selected_pool_positions": [int(pos) for pos in selected_positions],
            "selected_doc_ids": [
                int(pool_doc_ids[pos]) if pool_doc_ids[pos] is not None else None
                for pos in selected_positions
            ],
            "selected_titles": [pool_docs[pos].split("\n", 1)[0] for pos in selected_positions],
            **selector_trace,
        }

        selected_qs = QuerySolution(
            question=qs.question,
            docs=reordered_docs,
            doc_scores=reordered_scores,
            gold_answers=qs.gold_answers,
            gold_docs=qs.gold_docs,
            retrieval_trace=retrieval_trace,
            qa_trace=qs.qa_trace,
        )
        selected_solutions.append(selected_qs)

        mapped_pool_doc_counts.append(sum(doc_id is not None for doc_id in pool_doc_ids))
        seed_entity_counts.append(len(seed_entities))
        selected_structured_doc_counts.append(sum(
            1
            for pos in selected_positions
            if pool_doc_ids[pos] is not None
            and bool(hipporag.doc_idx_to_structure_entities.get(int(pool_doc_ids[pos]), set()))
        ))
        if len(selector_examples) < 5:
            selector_examples.append({
                "question": qs.question,
                "selected_titles": [pool_docs[pos].split("\n", 1)[0] for pos in selected_positions],
                "seed_entities_preview": sorted(seed_entities)[:8],
                "selection_steps": selector_trace["selection_steps"],
            })

    summary = {
        "selector": selector_name,
        "avg_mapped_pool_doc_count": round(float(np.mean(mapped_pool_doc_counts)) if mapped_pool_doc_counts else 0.0, 4),
        "avg_seed_entity_count": round(float(np.mean(seed_entity_counts)) if seed_entity_counts else 0.0, 4),
        "avg_selected_structured_doc_count": round(
            float(np.mean(selected_structured_doc_counts)) if selected_structured_doc_counts else 0.0,
            4,
        ),
        "examples_preview": selector_examples,
    }
    logger.info(
        "Applied %s selector over pool@%d for %d queries (avg mapped pool docs=%.2f, avg seed entities=%.2f)",
        selector_name,
        pool_k,
        len(query_solutions),
        float(np.mean(mapped_pool_doc_counts)) if mapped_pool_doc_counts else 0.0,
        float(np.mean(seed_entity_counts)) if seed_entity_counts else 0.0,
    )
    return selected_solutions, summary


def is_causal_query_solution(config: BaseConfig, query_solution: QuerySolution) -> bool:
    if getattr(config, "causal_engine_version", "legacy") == "v2":
        trace = query_solution.retrieval_trace or {}
        return trace.get("router_label") in {"cause", "effect", "prevention", "causal"}
    return route_query_type(query_solution.question) != "non_causal"


def has_nonempty_v2_subgraph(query_solution: QuerySolution) -> bool:
    trace = query_solution.retrieval_trace or {}
    return bool(trace.get("subgraph_nonempty", False))


def compute_slice_metrics(config: BaseConfig,
                          query_solutions: List[QuerySolution],
                          gold_docs: List[List[str]],
                          gold_answers: List[List[str]] | None) -> Dict[str, Dict[str, float]]:
    retrieved_docs = [query_solution.docs for query_solution in query_solutions]

    retrieval = RetrievalRecall(global_config=config)

    retrieval_metrics, _ = retrieval.calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=retrieved_docs,
        k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
    )
    em_metrics: Dict[str, float] = {}
    f1_metrics: Dict[str, float] = {}
    predicted_answers = []
    if gold_answers is not None:
        predicted_answers = [query_solution.answer for query_solution in query_solutions]
        qa_em = QAExactMatch(global_config=config)
        qa_f1 = QAF1Score(global_config=config)
        em_metrics, _ = qa_em.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            aggregation_fn=np.max,
        )
        f1_metrics, _ = qa_f1.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            aggregation_fn=np.max,
        )

    causal_indices = [
        idx for idx, query_solution in enumerate(query_solutions)
        if is_causal_query_solution(config, query_solution)
    ]
    if causal_indices:
        causal_retrieval_metrics, _ = retrieval.calculate_metric_scores(
            gold_docs=subset_by_indices(gold_docs, causal_indices),
            retrieved_docs=subset_by_indices(retrieved_docs, causal_indices),
            k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
        )
        if gold_answers is not None:
            qa_em = QAExactMatch(global_config=config)
            qa_f1 = QAF1Score(global_config=config)
            causal_em_metrics, _ = qa_em.calculate_metric_scores(
                gold_answers=subset_by_indices(gold_answers, causal_indices),
                predicted_answers=subset_by_indices(predicted_answers, causal_indices),
                aggregation_fn=np.max,
            )
            causal_f1_metrics, _ = qa_f1.calculate_metric_scores(
                gold_answers=subset_by_indices(gold_answers, causal_indices),
                predicted_answers=subset_by_indices(predicted_answers, causal_indices),
                aggregation_fn=np.max,
            )
        else:
            causal_em_metrics, causal_f1_metrics = {}, {}
    else:
        causal_retrieval_metrics, causal_em_metrics, causal_f1_metrics = {}, {}, {}

    nonempty_subgraph_indices = [
        idx for idx, query_solution in enumerate(query_solutions)
        if getattr(config, "causal_engine_version", "legacy") == "v2"
        and has_nonempty_v2_subgraph(query_solution)
    ]
    if nonempty_subgraph_indices:
        subgraph_retrieval_metrics, _ = retrieval.calculate_metric_scores(
            gold_docs=subset_by_indices(gold_docs, nonempty_subgraph_indices),
            retrieved_docs=subset_by_indices(retrieved_docs, nonempty_subgraph_indices),
            k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
        )
        if gold_answers is not None:
            qa_em = QAExactMatch(global_config=config)
            qa_f1 = QAF1Score(global_config=config)
            subgraph_em_metrics, _ = qa_em.calculate_metric_scores(
                gold_answers=subset_by_indices(gold_answers, nonempty_subgraph_indices),
                predicted_answers=subset_by_indices(predicted_answers, nonempty_subgraph_indices),
                aggregation_fn=np.max,
            )
            subgraph_f1_metrics, _ = qa_f1.calculate_metric_scores(
                gold_answers=subset_by_indices(gold_answers, nonempty_subgraph_indices),
                predicted_answers=subset_by_indices(predicted_answers, nonempty_subgraph_indices),
                aggregation_fn=np.max,
            )
        else:
            subgraph_em_metrics, subgraph_f1_metrics = {}, {}
    else:
        subgraph_retrieval_metrics, subgraph_em_metrics, subgraph_f1_metrics = {}, {}, {}

    return {
        "overall": {
            **retrieval_metrics,
            **em_metrics,
            **f1_metrics,
            "num_queries": len(query_solutions),
        },
        "causal_slice": {
            **causal_retrieval_metrics,
            **causal_em_metrics,
            **causal_f1_metrics,
            "num_queries": len(causal_indices),
        },
        "nonempty_subgraph_slice": {
            **subgraph_retrieval_metrics,
            **subgraph_em_metrics,
            **subgraph_f1_metrics,
            "num_queries": len(nonempty_subgraph_indices),
        },
    }


def summarize_v2_metrics(config: BaseConfig,
                         hipporag: HippoRAG,
                         query_solutions: List[QuerySolution]) -> Dict[str, object]:
    if getattr(config, "causal_engine_version", "legacy") != "v2":
        return {}

    traces = [query_solution.retrieval_trace or {} for query_solution in query_solutions]
    router_label_counts = Counter(str(trace.get("router_label", "unknown")) for trace in traces)
    chain_counts = [int(trace.get("subgraph_chain_count", 0)) for trace in traces]
    selected_chain_counts = [int(trace.get("selected_subgraph_chain_count", 0)) for trace in traces]
    seed_counts = [len(trace.get("event_seed_ids", [])) for trace in traces]
    query_entity_counts = [len(trace.get("query_entities", [])) for trace in traces]
    serialized_counts = [len(trace.get("serialized_causal_context", [])) for trace in traces]
    causal_doc_counts = [int(trace.get("causal_doc_count", 0)) for trace in traces]
    base_retrieval_mode_counts = Counter(str(trace.get("v2_base_retrieval_mode", "dense")) for trace in traces)
    base_retrieval_status_counts = Counter(str(trace.get("v2_base_retrieval_status", "unknown")) for trace in traces)
    base_retrieval_route_counts = Counter(str(trace.get("baseline_route_name", "unknown")) for trace in traces)
    base_retrieval_fact_counts = [int(trace.get("v2_base_retrieval_fact_count", 0)) for trace in traces]
    base_retrieval_dense_fallback_count = sum(1 for trace in traces if trace.get("v2_base_retrieval_used_dense_fallback"))
    use_causal_path_count = sum(1 for trace in traces if trace.get("use_causal_path"))
    probe_attempted_count = sum(1 for trace in traces if trace.get("causal_probe_attempted"))
    forced_probe_count = sum(1 for trace in traces if trace.get("causal_probe_forced"))
    subgraph_nonempty_count = sum(1 for trace in traces if trace.get("subgraph_nonempty"))
    causal_v2_used_count = sum(1 for trace in traces if trace.get("causal_v2_used"))
    generator_used_count = sum(1 for trace in traces if trace.get("generator_used_causal_context"))

    manifest_stats = {}
    engine = getattr(hipporag, "causal_v2_engine", None)
    if engine is not None:
        manifest_stats = engine.read_manifest()

    num_queries = max(1, len(traces))
    causal_examples = []
    for query_solution in query_solutions:
        trace = query_solution.retrieval_trace or {}
        if trace.get("causal_v2_used") or trace.get("causal_probe_attempted"):
            causal_examples.append({
                "question": query_solution.question,
                "router_label": trace.get("router_label"),
                "probe_route_label": trace.get("probe_route_label"),
                "router_score": trace.get("router_score"),
                "router_margin": trace.get("router_margin"),
                "causal_probe_forced": trace.get("causal_probe_forced"),
                "subgraph_nonempty": trace.get("subgraph_nonempty"),
                "subgraph_chain_count": trace.get("subgraph_chain_count"),
                "selected_subgraph_chain_count": trace.get("selected_subgraph_chain_count"),
                "query_entities": trace.get("query_entities", [])[:6],
                "selected_chain_scores": trace.get("selected_chain_scores", []),
                "serialized_causal_context_preview": trace.get("serialized_causal_context_preview", []),
            })
        if len(causal_examples) >= 5:
            break

    return {
        "index_manifest": manifest_stats,
        "router_label_counts": dict(sorted(router_label_counts.items())),
        "base_retrieval_mode_counts": dict(sorted(base_retrieval_mode_counts.items())),
        "base_retrieval_status_counts": dict(sorted(base_retrieval_status_counts.items())),
        "base_retrieval_route_counts": dict(sorted(base_retrieval_route_counts.items())),
        "base_retrieval_dense_fallback_count": int(base_retrieval_dense_fallback_count),
        "base_retrieval_dense_fallback_rate": round(base_retrieval_dense_fallback_count / num_queries, 4),
        "avg_base_retrieval_fact_count": round(float(np.mean(base_retrieval_fact_counts)) if base_retrieval_fact_counts else 0.0, 4),
        "use_causal_path_count": int(use_causal_path_count),
        "use_causal_path_rate": round(use_causal_path_count / num_queries, 4),
        "probe_attempted_count": int(probe_attempted_count),
        "probe_attempted_rate": round(probe_attempted_count / num_queries, 4),
        "forced_probe_count": int(forced_probe_count),
        "forced_probe_rate": round(forced_probe_count / num_queries, 4),
        "subgraph_nonempty_count": int(subgraph_nonempty_count),
        "subgraph_nonempty_rate": round(subgraph_nonempty_count / num_queries, 4),
        "causal_v2_used_count": int(causal_v2_used_count),
        "causal_v2_used_rate": round(causal_v2_used_count / num_queries, 4),
        "generator_used_causal_context_count": int(generator_used_count),
        "generator_used_causal_context_rate": round(generator_used_count / num_queries, 4),
        "avg_subgraph_chain_count": round(float(np.mean(chain_counts)) if chain_counts else 0.0, 4),
        "avg_selected_chain_count": round(float(np.mean(selected_chain_counts)) if selected_chain_counts else 0.0, 4),
        "max_subgraph_chain_count": int(max(chain_counts) if chain_counts else 0),
        "avg_seed_event_count": round(float(np.mean(seed_counts)) if seed_counts else 0.0, 4),
        "avg_query_entity_count": round(float(np.mean(query_entity_counts)) if query_entity_counts else 0.0, 4),
        "avg_serialized_context_count": round(float(np.mean(serialized_counts)) if serialized_counts else 0.0, 4),
        "avg_causal_doc_count": round(float(np.mean(causal_doc_counts)) if causal_doc_counts else 0.0, 4),
        "causal_examples_preview": causal_examples,
    }


def build_config(args, corpus_len: int) -> BaseConfig:
    return BaseConfig(
        save_dir=args.save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_base_url=args.embedding_base_url,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=args.qa_top_k,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=args.embedding_batch_size,
        max_new_tokens=None,
        max_retry_attempts=args.max_retry_attempts,
        corpus_len=corpus_len,
        openie_mode=args.openie_mode,
        planner_enabled=string_to_bool(args.planner_enabled),
        planner_mode=args.planner_mode,
        planner_max_steps=args.planner_max_steps,
        causal_enabled=string_to_bool(args.causal_enabled),
        causal_query_only=string_to_bool(args.causal_query_only),
        causal_gate_mode=args.causal_gate_mode,
        causal_seed_top_k=args.causal_seed_top_k,
        causal_confidence_threshold=args.causal_confidence_threshold,
        causal_damping=args.causal_damping,
        causal_blend_dense_weight=args.causal_blend_dense_weight,
        causal_blend_fact_weight=args.causal_blend_fact_weight,
        causal_blend_graph_weight=args.causal_blend_graph_weight,
        causal_margin_gate_enabled=string_to_bool(args.causal_margin_gate_enabled),
        causal_margin_threshold=args.causal_margin_threshold,
        causal_blend_top_k=args.causal_blend_top_k,
        causal_engine_version=getattr(args, "causal_engine_version", "legacy"),
        causal_v2_probe_mode=getattr(args, "causal_v2_probe_mode", "router"),
        causal_v2_graph_mode=getattr(args, "causal_v2_graph_mode", "causal"),
        causal_v2_base_retrieval_mode=getattr(args, "causal_v2_base_retrieval_mode", "dense"),
        general_graph_related_to_weight=getattr(args, "general_graph_related_to_weight", 0.3),
        general_graph_seed_top_k=getattr(args, "general_graph_seed_top_k", 10),
        causal_v2_extraction_max_tokens=getattr(args, "causal_v2_extraction_max_tokens", 768),
        causal_v2_extraction_retry_attempts=getattr(args, "causal_v2_extraction_retry_attempts", 2),
        causal_v2_extraction_workers=getattr(args, "causal_v2_extraction_workers", 4),
        causal_event_top_k=getattr(args, "causal_event_top_k", 8),
        causal_v2_max_hops=getattr(args, "causal_v2_max_hops", 2),
        causal_chain_top_k=getattr(args, "causal_chain_top_k", 6),
        causal_context_max_items=getattr(args, "causal_context_max_items", 0),
        causal_er_similarity_threshold=getattr(args, "causal_er_similarity_threshold", 0.92),
        causal_er_text_threshold=getattr(args, "causal_er_text_threshold", 0.55),
        causal_v2_min_edge_confidence=getattr(args, "causal_v2_min_edge_confidence", 0.7),
        structure_rerank_enabled=string_to_bool(args.structure_rerank_enabled),
        structure_rerank_top_n=args.structure_rerank_top_n,
        structure_rerank_bonus_weight=args.structure_rerank_bonus_weight,
        structure_rerank_min_edge_support=args.structure_rerank_min_edge_support,
        structure_rerank_max_top5_swaps=args.structure_rerank_max_top5_swaps,
        structure_rerank_seed_top_k=args.structure_rerank_seed_top_k,
        structure_rerank_max_hops=args.structure_rerank_max_hops,
        structure_rerank_margin_threshold=args.structure_rerank_margin_threshold,
        rerank_require_non_empty=string_to_bool(args.rerank_require_non_empty),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate HippoRAG with local Qwen3 causal retrieval.")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    parser.add_argument("--planner_enabled", type=str, default="false")
    parser.add_argument("--planner_mode", choices=["none", "myopic"], default="none")
    parser.add_argument("--planner_max_steps", type=int, default=3)
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--causal_enabled", type=str, default="true")
    parser.add_argument("--causal_query_only", type=str, default="true")
    parser.add_argument("--causal_gate_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--causal_seed_top_k", type=int, default=20)
    parser.add_argument("--causal_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--causal_damping", type=float, default=0.7)
    parser.add_argument("--causal_blend_dense_weight", type=float, default=0.35)
    parser.add_argument("--causal_blend_fact_weight", type=float, default=0.15)
    parser.add_argument("--causal_blend_graph_weight", type=float, default=0.50)
    parser.add_argument("--causal_margin_gate_enabled", type=str, default="false")
    parser.add_argument("--causal_margin_threshold", type=float, default=0.02)
    parser.add_argument("--causal_blend_top_k", type=int, default=0)
    parser.add_argument("--causal_engine_version", choices=["legacy", "v2"], default="legacy")
    parser.add_argument("--causal_v2_probe_mode", choices=["router", "always"], default="router")
    parser.add_argument("--causal_v2_graph_mode", choices=["causal", "general"], default="causal")
    parser.add_argument("--causal_v2_base_retrieval_mode", choices=["dense", "legacy_fact_graph", "general_relation_graph"], default="dense")
    parser.add_argument("--general_graph_related_to_weight", type=float, default=0.3)
    parser.add_argument("--general_graph_seed_top_k", type=int, default=10)
    parser.add_argument("--causal_v2_extraction_max_tokens", type=int, default=768)
    parser.add_argument("--causal_v2_extraction_retry_attempts", type=int, default=2)
    parser.add_argument("--causal_v2_extraction_workers", type=int, default=4)
    parser.add_argument("--causal_event_top_k", type=int, default=8)
    parser.add_argument("--causal_v2_max_hops", type=int, default=2)
    parser.add_argument("--causal_chain_top_k", type=int, default=6)
    parser.add_argument("--causal_context_max_items", type=int, default=0)
    parser.add_argument("--causal_er_similarity_threshold", type=float, default=0.92)
    parser.add_argument("--causal_er_text_threshold", type=float, default=0.55)
    parser.add_argument("--causal_v2_min_edge_confidence", type=float, default=0.7)
    parser.add_argument("--structure_rerank_enabled", type=str, default="true")
    parser.add_argument("--structure_rerank_top_n", type=int, default=40)
    parser.add_argument("--structure_rerank_bonus_weight", type=float, default=0.08)
    parser.add_argument("--structure_rerank_min_edge_support", type=int, default=2)
    parser.add_argument("--structure_rerank_max_top5_swaps", type=int, default=2)
    parser.add_argument("--structure_rerank_seed_top_k", type=int, default=4)
    parser.add_argument("--structure_rerank_max_hops", type=int, default=2)
    parser.add_argument("--structure_rerank_margin_threshold", type=float, default=0.02)
    parser.add_argument("--rerank_require_non_empty", type=str, default="true")
    parser.add_argument("--retrieval_only", type=str, default="false")
    parser.add_argument("--gold_doc_reader", type=str, default="false",
                        help="Skip retrieval, feed gold docs directly to reader. Tests reader ceiling.")
    parser.add_argument("--oracle_reorder_k", type=int, default=0,
                        help="Move gold docs found within top-K to front. Tests reranker ceiling. 0=disabled.")
    parser.add_argument("--oracle_select_k", type=str, default="0",
                        help="Oracle select: comma-separated K values (e.g. '20,30,50,100'). From top-K pool, prioritize gold docs in reader's top-5. 0=disabled.")
    parser.add_argument("--cross_encoder_rerank", type=str, default="false",
                        help="Apply cross-encoder rerank on baseline top-K docs. Eval-time only.")
    parser.add_argument("--ce_model", type=str, default="/mnt/nvme/bge-reranker-v2-m3",
                        help="Cross-encoder model path or HF name for FlagEmbedding.")
    parser.add_argument("--ce_alpha", type=float, default=0.7,
                        help="Hybrid weight: alpha * ppr_norm + (1-alpha) * ce_norm. 1.0 = pure PPR.")
    parser.add_argument("--ce_window", type=int, default=20,
                        help="Number of top docs to rerank with cross-encoder.")
    parser.add_argument("--ce_device", type=str, default="cuda:1",
                        help="Device for cross-encoder model.")
    parser.add_argument("--setwise_selector", choices=["none", "bridge_greedy", "learned_greedy"], default="none",
                        help="Apply a non-oracle setwise selector over a larger pool before reader top-k truncation.")
    parser.add_argument("--setwise_pool_k", type=int, default=20,
                        help="Candidate pool size used by the setwise selector.")
    parser.add_argument("--setwise_anchor_count", type=int, default=2,
                        help="Number of top-ranked anchor docs preserved before greedy bridge completion.")
    parser.add_argument("--setwise_structure_max_hops", type=int, default=2,
                        help="Directed structure expansion depth used by the setwise selector.")
    parser.add_argument("--setwise_base_weight", type=float, default=0.25,
                        help="Weight assigned to baseline retrieval score inside setwise selection.")
    parser.add_argument("--setwise_structure_weight", type=float, default=0.60,
                        help="Weight assigned to structure bridge score inside setwise selection.")
    parser.add_argument("--setwise_novelty_weight", type=float, default=0.15,
                        help="Weight assigned to entity novelty inside setwise selection.")
    parser.add_argument("--setwise_model_path", type=str, default="",
                        help="Joblib bundle path used by --setwise_selector learned_greedy.")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    if save_dir == "outputs":
        save_dir = os.path.join(save_dir, dataset_name)
    else:
        save_dir = f"{save_dir}_{dataset_name}"
    args.save_dir = save_dir

    corpus_path = Path(f"reproduce/dataset/{dataset_name}_corpus.json")
    sample_path = Path(f"reproduce/dataset/{dataset_name}.json")
    corpus = json.load(corpus_path.open())
    samples = json.load(sample_path.open())

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    queries = [sample["question"] for sample in samples]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset_name, corpus=corpus)
    retrieval_only = string_to_bool(args.retrieval_only)
    gold_doc_reader = string_to_bool(args.gold_doc_reader)
    oracle_reorder_k = int(args.oracle_reorder_k)
    oracle_select_ks = [int(x) for x in args.oracle_select_k.split(",") if int(x) > 0]

    config = build_config(args, corpus_len=len(corpus))
    logging.basicConfig(level=logging.INFO)

    oracle_reorder_qa_results = None
    setwise_selector_results = None
    learned_model_bundle = None
    setwise_selector = args.setwise_selector.lower() if not gold_doc_reader else "none"
    if setwise_selector == "learned_greedy":
        if not args.setwise_model_path:
            raise ValueError("--setwise_model_path is required when --setwise_selector learned_greedy")
        learned_model_bundle = load_learned_model_bundle(args.setwise_model_path)

    if gold_doc_reader:
        # Exp2: Gold-doc reader — skip retrieval, feed gold docs to reader
        hipporag = HippoRAG(global_config=config)
        hipporag.index(docs)
        gold_query_solutions = [
            QuerySolution(
                question=query,
                docs=gold_docs[q_idx],
                doc_scores=np.ones(len(gold_docs[q_idx])),
            )
            for q_idx, query in enumerate(queries)
        ]
        # rag_qa accepts QuerySolution list directly — skips retrieve()
        query_solutions, responses, metadata, _, overall_qa_results = hipporag.rag_qa(
            queries=gold_query_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )
        overall_retrieval_result = {"note": "gold_doc_reader mode — retrieval metrics are N/A (oracle-by-construction)"}
        effective_gold_answers = gold_answers
    else:
        hipporag = HippoRAG(global_config=config)
        hipporag.index(docs)
        if retrieval_only:
            query_solutions, overall_retrieval_result = hipporag.retrieve(
                queries=queries,
                gold_docs=gold_docs,
            )
            responses = []
            metadata = []
            overall_qa_results = {}
            effective_gold_answers = None
        else:
            query_solutions, responses, metadata, overall_retrieval_result, overall_qa_results = hipporag.rag_qa(
                queries=queries,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
            effective_gold_answers = gold_answers

        # Exp3: Oracle reorder within top-K
        if oracle_reorder_k > 0 and not retrieval_only:
            qa_em = QAExactMatch(global_config=config)
            qa_f1 = QAF1Score(global_config=config)
            reordered_solutions = []
            full_support_in_topk_count = 0
            for q_idx, qs in enumerate(query_solutions):
                gold_set = set(gold_docs[q_idx])
                top_k_docs = qs.docs[:oracle_reorder_k]
                gold_in_topk = [d for d in top_k_docs if d in gold_set]
                non_gold_in_topk = [d for d in top_k_docs if d not in gold_set]
                rest = qs.docs[oracle_reorder_k:]
                reordered_docs = gold_in_topk + non_gold_in_topk + rest
                if gold_set.issubset(set(top_k_docs)):
                    full_support_in_topk_count += 1
                reordered_qs = QuerySolution(
                    question=qs.question,
                    docs=reordered_docs,
                    doc_scores=qs.doc_scores,
                    gold_docs=gold_docs[q_idx],
                )
                reordered_solutions.append(reordered_qs)
            # Run QA on reordered docs via rag_qa (skips retrieve since input is QuerySolution)
            reordered_solutions, _, _, _, reorder_qa_results = hipporag.rag_qa(
                queries=reordered_solutions,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
            reordered_answers = [qs.answer for qs in reordered_solutions]
            reorder_em = reorder_qa_results.get("ExactMatch", 0.0)
            reorder_f1 = reorder_qa_results.get("F1", 0.0)
            baseline_em = overall_qa_results.get("ExactMatch", 0.0) if overall_qa_results else 0.0
            oracle_reorder_qa_results = {
                "oracle_reorder_k": oracle_reorder_k,
                "oracle_reorder_EM": round(float(reorder_em), 4),
                "oracle_reorder_F1": round(float(reorder_f1), 4),
                "baseline_EM": round(float(baseline_em), 4),
                "EM_delta": round(float(reorder_em) - float(baseline_em), 4),
                "full_support_in_top_k_rate": round(full_support_in_topk_count / max(1, len(queries)), 4),
                "full_support_in_top_k_count": full_support_in_topk_count,
            }

    # Exp4: Oracle select sweep — ceiling curve across multiple K values
    oracle_select_qa_results = None
    if oracle_select_ks and not retrieval_only and not gold_doc_reader:
        qa_top = config.qa_top_k  # typically 5
        logger = logging.getLogger(__name__)
        baseline_em = overall_qa_results.get("ExactMatch", 0.0) if overall_qa_results else 0.0
        baseline_f1 = overall_qa_results.get("F1", 0.0) if overall_qa_results else 0.0

        # --- Minimal full-support depth per query ---
        # Smallest K such that all gold docs are in docs[:K]
        per_query_support_depth = []
        for q_idx, qs in enumerate(query_solutions):
            gold_set = set(gold_docs[q_idx])
            found = set()
            depth = None
            for rank, d in enumerate(qs.docs, 1):
                if d in gold_set:
                    found.add(d)
                if found == gold_set:
                    depth = rank
                    break
            per_query_support_depth.append(depth)  # None = never fully supported

        # Bucket support depth stats
        depth_by_bucket: dict[int, list] = {}
        for q_idx in range(len(query_solutions)):
            n_gold = len(set(gold_docs[q_idx]))
            if n_gold not in depth_by_bucket:
                depth_by_bucket[n_gold] = []
            depth_by_bucket[n_gold].append(per_query_support_depth[q_idx])

        support_depth_summary = {}
        for n_gold in sorted(depth_by_bucket.keys()):
            depths = depth_by_bucket[n_gold]
            finite = [d for d in depths if d is not None]
            support_depth_summary[f"{n_gold}-doc"] = {
                "count": len(depths),
                "fully_supported": len(finite),
                "never_supported": len(depths) - len(finite),
                "median_depth": round(float(np.median(finite)), 1) if finite else None,
                "mean_depth": round(float(np.mean(finite)), 1) if finite else None,
                "p90_depth": round(float(np.percentile(finite, 90)), 1) if finite else None,
                "max_depth": int(max(finite)) if finite else None,
            }
        logger.info("Minimal full-support depth:")
        for bk, bv in support_depth_summary.items():
            logger.info(f"  {bk}: supported={bv['fully_supported']}/{bv['count']}, "
                         f"median={bv['median_depth']}, mean={bv['mean_depth']}, p90={bv['p90_depth']}")

        # --- Oracle select sweep over K values ---
        sweep_results = {}
        for sel_k in sorted(oracle_select_ks):
            logger.info(f"Oracle select: pool={sel_k}, reader sees top-{qa_top}")
            selected_solutions = []
            bucket_stats: dict[int, dict] = {}

            for q_idx, qs in enumerate(query_solutions):
                pool = qs.docs[:sel_k]
                gold_set = set(gold_docs[q_idx])
                n_gold = len(gold_set)

                gold_in_pool = [d for d in pool if d in gold_set]
                non_gold_in_pool = [d for d in pool if d not in gold_set]
                selected_docs = (gold_in_pool + non_gold_in_pool)[:max(sel_k, qa_top)]
                rest = qs.docs[sel_k:]
                all_docs = selected_docs + rest

                selected_qs = QuerySolution(
                    question=qs.question,
                    docs=all_docs,
                    doc_scores=qs.doc_scores,
                    gold_docs=gold_docs[q_idx],
                )
                selected_solutions.append(selected_qs)

                if n_gold not in bucket_stats:
                    bucket_stats[n_gold] = {"count": 0, "gold_found_sum": 0, "fs_sum": 0}
                bucket_stats[n_gold]["count"] += 1
                bucket_stats[n_gold]["gold_found_sum"] += len(gold_in_pool)
                if gold_set.issubset(set(pool)):
                    bucket_stats[n_gold]["fs_sum"] += 1

            # Run QA on oracle-selected docs
            selected_solutions, _, _, _, select_qa_results = hipporag.rag_qa(
                queries=selected_solutions,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
            select_em = select_qa_results.get("ExactMatch", 0.0)
            select_f1 = select_qa_results.get("F1", 0.0)

            # Per-bucket EM/F1 breakdown
            qa_em_metric = QAExactMatch(global_config=config)
            qa_f1_metric = QAF1Score(global_config=config)
            sel_answers = [qs.answer or "" for qs in selected_solutions]
            _, per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, sel_answers)
            _, per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, sel_answers)
            bucket_em: dict[int, list] = {}
            bucket_f1: dict[int, list] = {}
            for q_idx in range(len(selected_solutions)):
                n_gold = len(set(gold_docs[q_idx]))
                if n_gold not in bucket_em:
                    bucket_em[n_gold] = []
                    bucket_f1[n_gold] = []
                bucket_em[n_gold].append(per_query_em[q_idx]["ExactMatch"])
                bucket_f1[n_gold].append(per_query_f1[q_idx]["F1"])

            bucket_breakdown = {}
            for n_gold in sorted(bucket_stats.keys()):
                bs = bucket_stats[n_gold]
                bucket_breakdown[f"{n_gold}-doc"] = {
                    "count": bs["count"],
                    "avg_gold_found_in_pool": round(bs["gold_found_sum"] / max(1, bs["count"]), 3),
                    "full_support_rate": round(bs["fs_sum"] / max(1, bs["count"]), 4),
                    "EM": round(float(np.mean(bucket_em.get(n_gold, [0]))), 4),
                    "F1": round(float(np.mean(bucket_f1.get(n_gold, [0]))), 4),
                }

            total_fs = sum(bs["fs_sum"] for bs in bucket_stats.values())
            sweep_results[f"K={sel_k}"] = {
                "pool_k": sel_k,
                "oracle_select_EM": round(float(select_em), 4),
                "oracle_select_F1": round(float(select_f1), 4),
                "EM_delta": round(float(select_em) - float(baseline_em), 4),
                "F1_delta": round(float(select_f1) - float(baseline_f1), 4),
                "full_support_in_pool_rate": round(total_fs / max(1, len(queries)), 4),
                "bucket_breakdown": bucket_breakdown,
            }
            logger.info(f"Oracle select@{sel_k}: EM={select_em:.4f} (delta={float(select_em)-float(baseline_em):+.4f}), "
                         f"F1={select_f1:.4f}, FS_in_pool={total_fs}/{len(queries)}")
            for bk, bv in bucket_breakdown.items():
                logger.info(f"  {bk}: count={bv['count']}, FS={bv['full_support_rate']}, EM={bv['EM']}, F1={bv['F1']}")

        oracle_select_qa_results = {
            "baseline_EM": round(float(baseline_em), 4),
            "baseline_F1": round(float(baseline_f1), 4),
            "support_depth": support_depth_summary,
            "sweep": sweep_results,
        }

    if setwise_selector != "none" and not retrieval_only and query_solutions:
        logger = logging.getLogger(__name__)
        selected_solutions, selector_summary = apply_setwise_selector(
            hipporag=hipporag,
            query_solutions=query_solutions,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
            pool_k=int(args.setwise_pool_k),
            qa_top_k=int(config.qa_top_k),
            selector_name=setwise_selector,
            anchor_count=int(args.setwise_anchor_count),
            structure_max_hops=int(args.setwise_structure_max_hops),
            base_weight=float(args.setwise_base_weight),
            structure_weight=float(args.setwise_structure_weight),
            novelty_weight=float(args.setwise_novelty_weight),
            learned_model_bundle=learned_model_bundle,
        )
        selected_solutions, _, _, _, selector_qa_results = hipporag.rag_qa(
            queries=selected_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )

        qa_em_metric = QAExactMatch(global_config=config)
        qa_f1_metric = QAF1Score(global_config=config)
        bl_answers = [qs.answer or "" for qs in query_solutions]
        selector_answers = [qs.answer or "" for qs in selected_solutions]
        _, bl_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, bl_answers)
        _, bl_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, bl_answers)
        _, selector_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, selector_answers)
        _, selector_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, selector_answers)

        bucket_results = {}
        for q_idx in range(len(queries)):
            n_gold = len(set(gold_docs[q_idx]))
            if n_gold not in bucket_results:
                bucket_results[n_gold] = {
                    "baseline_em": [],
                    "selector_em": [],
                    "baseline_f1": [],
                    "selector_f1": [],
                    "count": 0,
                }
            bucket_results[n_gold]["count"] += 1
            bucket_results[n_gold]["baseline_em"].append(bl_per_query_em[q_idx]["ExactMatch"])
            bucket_results[n_gold]["selector_em"].append(selector_per_query_em[q_idx]["ExactMatch"])
            bucket_results[n_gold]["baseline_f1"].append(bl_per_query_f1[q_idx]["F1"])
            bucket_results[n_gold]["selector_f1"].append(selector_per_query_f1[q_idx]["F1"])

        bucket_summary = {}
        for n_gold, data in sorted(bucket_results.items()):
            bucket_summary[f"{n_gold}_doc"] = {
                "count": data["count"],
                "baseline_EM": round(float(np.mean(data["baseline_em"])), 4),
                "selector_EM": round(float(np.mean(data["selector_em"])), 4),
                "EM_delta": round(float(np.mean(data["selector_em"])) - float(np.mean(data["baseline_em"])), 4),
                "baseline_F1": round(float(np.mean(data["baseline_f1"])), 4),
                "selector_F1": round(float(np.mean(data["selector_f1"])), 4),
                "F1_delta": round(float(np.mean(data["selector_f1"])) - float(np.mean(data["baseline_f1"])), 4),
            }

        selector_retrieval_metrics = {}
        for k in [1, 2, 5, 10, 20]:
            recalls = []
            for q_idx, qs in enumerate(selected_solutions):
                gold_set = set(gold_docs[q_idx])
                top_k_set = set(qs.docs[:k])
                recalls.append(len(gold_set & top_k_set) / max(1, len(gold_set)))
            selector_retrieval_metrics[f"Recall@{k}"] = round(float(np.mean(recalls)), 4)

        selector_em = selector_qa_results.get("ExactMatch", 0.0)
        selector_f1 = selector_qa_results.get("F1", 0.0)
        baseline_em = overall_qa_results.get("ExactMatch", 0.0) if overall_qa_results else 0.0
        baseline_f1 = overall_qa_results.get("F1", 0.0) if overall_qa_results else 0.0
        setwise_selector_results = {
            "selector": setwise_selector,
            "pool_k": int(args.setwise_pool_k),
            "anchor_count": int(args.setwise_anchor_count),
            "structure_max_hops": int(args.setwise_structure_max_hops),
            "base_weight": float(args.setwise_base_weight),
            "structure_weight": float(args.setwise_structure_weight),
            "novelty_weight": float(args.setwise_novelty_weight),
            "setwise_model_path": args.setwise_model_path or None,
            "selector_EM": round(float(selector_em), 4),
            "selector_F1": round(float(selector_f1), 4),
            "baseline_EM": round(float(baseline_em), 4),
            "baseline_F1": round(float(baseline_f1), 4),
            "EM_delta": round(float(selector_em) - float(baseline_em), 4),
            "F1_delta": round(float(selector_f1) - float(baseline_f1), 4),
            "selector_retrieval_metrics": selector_retrieval_metrics,
            "per_bucket": bucket_summary,
            "selector_summary": selector_summary,
        }
        logger.info(
            "Setwise selector %s@%d: EM=%.4f (delta=%+.4f), F1=%.4f",
            setwise_selector,
            int(args.setwise_pool_k),
            float(selector_em),
            float(selector_em) - float(baseline_em),
            float(selector_f1),
        )

    # Cross-encoder rerank on baseline final top-K
    cross_encoder_rerank_results = None
    cross_encoder_rerank = string_to_bool(args.cross_encoder_rerank) if not gold_doc_reader else False
    if cross_encoder_rerank and not retrieval_only and query_solutions:
        from FlagEmbedding import FlagReranker

        ce_window = int(args.ce_window)
        ce_alpha = float(args.ce_alpha)
        ce_model_name = args.ce_model
        ce_device = args.ce_device

        logger = logging.getLogger(__name__)
        logger.info(f"Loading cross-encoder model: {ce_model_name} on {ce_device}")
        ce_reranker = FlagReranker(ce_model_name, use_fp16=True, device=ce_device)

        reranked_solutions = []
        for q_idx, qs in enumerate(query_solutions):
            window = min(ce_window, len(qs.docs))
            window_docs = qs.docs[:window]
            window_scores = qs.doc_scores[:window] if qs.doc_scores is not None and len(qs.doc_scores) >= window else np.ones(window)
            rest_docs = qs.docs[window:]
            rest_scores = qs.doc_scores[window:] if qs.doc_scores is not None and len(qs.doc_scores) > window else np.array([])

            # Cross-encoder scoring
            pairs = [[qs.question, doc] for doc in window_docs]
            ce_scores = ce_reranker.compute_score(pairs)
            if isinstance(ce_scores, (int, float)):
                ce_scores = [ce_scores]
            ce_scores = np.array(ce_scores, dtype=float)

            # Min-max normalize both score arrays within window
            ppr_arr = np.array(window_scores, dtype=float)
            ppr_range = ppr_arr.max() - ppr_arr.min()
            ppr_norm = (ppr_arr - ppr_arr.min()) / (ppr_range + 1e-9) if ppr_range > 0 else np.ones_like(ppr_arr)
            ce_range = ce_scores.max() - ce_scores.min()
            ce_norm = (ce_scores - ce_scores.min()) / (ce_range + 1e-9) if ce_range > 0 else np.ones_like(ce_scores)

            combined = ce_alpha * ppr_norm + (1 - ce_alpha) * ce_norm
            reorder_idx = np.argsort(-combined)

            reranked_docs = [window_docs[i] for i in reorder_idx] + list(rest_docs)
            reranked_scores = np.concatenate([combined[reorder_idx], rest_scores]) if len(rest_scores) > 0 else combined[reorder_idx]

            reranked_qs = QuerySolution(
                question=qs.question,
                docs=reranked_docs,
                doc_scores=reranked_scores,
                gold_docs=gold_docs[q_idx],
            )
            reranked_solutions.append(reranked_qs)

        # Run QA on reranked docs
        logger.info("Running QA on cross-encoder reranked docs...")
        reranked_solutions, _, _, _, ce_qa_results = hipporag.rag_qa(
            queries=reranked_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )

        # Compute retrieval metrics on reranked order
        retrieval_recall = RetrievalRecall(global_config=config)
        ce_retrieval_metrics = {}
        for k in [1, 2, 5, 10, 20]:
            recalls = []
            for q_idx, qs in enumerate(reranked_solutions):
                gold_set = set(gold_docs[q_idx])
                top_k_set = set(qs.docs[:k])
                recalls.append(len(gold_set & top_k_set) / max(1, len(gold_set)))
            ce_retrieval_metrics[f"Recall@{k}"] = round(float(np.mean(recalls)), 4)

        ce_em = ce_qa_results.get("ExactMatch", 0.0)
        ce_f1 = ce_qa_results.get("F1", 0.0)
        baseline_em = overall_qa_results.get("ExactMatch", 0.0) if overall_qa_results else 0.0
        baseline_f1 = overall_qa_results.get("F1", 0.0) if overall_qa_results else 0.0

        # Per-bucket breakdown (2-doc vs 4-doc)
        # Compute per-query EM/F1 via the list API
        qa_em_metric = QAExactMatch(global_config=config)
        qa_f1_metric = QAF1Score(global_config=config)
        bl_answers = [qs.answer or "" for qs in query_solutions]
        ce_answers = [qs.answer or "" for qs in reranked_solutions]
        _, bl_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, bl_answers)
        _, bl_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, bl_answers)
        _, ce_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, ce_answers)
        _, ce_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, ce_answers)

        bucket_results = {}
        for q_idx in range(len(queries)):
            n_gold = len(set(gold_docs[q_idx]))
            if n_gold not in bucket_results:
                bucket_results[n_gold] = {"baseline_em": [], "ce_em": [], "baseline_f1": [], "ce_f1": [], "count": 0}
            bucket_results[n_gold]["count"] += 1
            bucket_results[n_gold]["baseline_em"].append(bl_per_query_em[q_idx]["ExactMatch"])
            bucket_results[n_gold]["ce_em"].append(ce_per_query_em[q_idx]["ExactMatch"])
            bucket_results[n_gold]["baseline_f1"].append(bl_per_query_f1[q_idx]["F1"])
            bucket_results[n_gold]["ce_f1"].append(ce_per_query_f1[q_idx]["F1"])

        bucket_summary = {}
        for n_gold, data in sorted(bucket_results.items()):
            bucket_summary[f"{n_gold}_doc"] = {
                "count": data["count"],
                "baseline_EM": round(float(np.mean(data["baseline_em"])), 4),
                "ce_rerank_EM": round(float(np.mean(data["ce_em"])), 4),
                "EM_delta": round(float(np.mean(data["ce_em"])) - float(np.mean(data["baseline_em"])), 4),
                "baseline_F1": round(float(np.mean(data["baseline_f1"])), 4),
                "ce_rerank_F1": round(float(np.mean(data["ce_f1"])), 4),
            }

        cross_encoder_rerank_results = {
            "ce_model": ce_model_name,
            "ce_alpha": ce_alpha,
            "ce_window": ce_window,
            "ce_rerank_EM": round(float(ce_em), 4),
            "ce_rerank_F1": round(float(ce_f1), 4),
            "baseline_EM": round(float(baseline_em), 4),
            "baseline_F1": round(float(baseline_f1), 4),
            "EM_delta": round(float(ce_em) - float(baseline_em), 4),
            "F1_delta": round(float(ce_f1) - float(baseline_f1), 4),
            "ce_retrieval_metrics": ce_retrieval_metrics,
            "per_bucket": bucket_summary,
        }

    slice_metrics = compute_slice_metrics(
        config=config,
        query_solutions=query_solutions,
        gold_docs=gold_docs,
        gold_answers=effective_gold_answers,
    )
    v2_metrics = summarize_v2_metrics(
        config=config,
        hipporag=hipporag,
        query_solutions=query_solutions,
    )
    result = {
        "dataset": dataset_name,
        "limit": len(samples),
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "embedding_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "config": {
            "causal_enabled": config.causal_enabled,
            "causal_query_only": config.causal_query_only,
            "causal_gate_mode": config.causal_gate_mode,
            "causal_seed_top_k": config.causal_seed_top_k,
            "causal_confidence_threshold": config.causal_confidence_threshold,
            "causal_damping": config.causal_damping,
            "causal_blend_dense_weight": config.causal_blend_dense_weight,
            "causal_blend_fact_weight": config.causal_blend_fact_weight,
            "causal_blend_graph_weight": config.causal_blend_graph_weight,
            "causal_margin_gate_enabled": config.causal_margin_gate_enabled,
            "causal_margin_threshold": config.causal_margin_threshold,
            "causal_blend_top_k": config.causal_blend_top_k,
            "causal_engine_version": config.causal_engine_version,
            "causal_v2_probe_mode": config.causal_v2_probe_mode,
            "causal_v2_graph_mode": config.causal_v2_graph_mode,
            "causal_v2_base_retrieval_mode": config.causal_v2_base_retrieval_mode,
            "general_graph_related_to_weight": config.general_graph_related_to_weight,
            "general_graph_seed_top_k": config.general_graph_seed_top_k,
            "retrieval_only": retrieval_only,
            "gold_doc_reader": gold_doc_reader,
            "oracle_reorder_k": oracle_reorder_k,
            "oracle_select_ks": oracle_select_ks,
            "cross_encoder_rerank": cross_encoder_rerank,
            "setwise_selector": setwise_selector,
            "setwise_pool_k": int(args.setwise_pool_k),
            "setwise_anchor_count": int(args.setwise_anchor_count),
            "setwise_structure_max_hops": int(args.setwise_structure_max_hops),
            "setwise_base_weight": float(args.setwise_base_weight),
            "setwise_structure_weight": float(args.setwise_structure_weight),
            "setwise_novelty_weight": float(args.setwise_novelty_weight),
            "setwise_model_path": args.setwise_model_path or None,
            "causal_v2_extraction_max_tokens": config.causal_v2_extraction_max_tokens,
            "causal_v2_extraction_retry_attempts": config.causal_v2_extraction_retry_attempts,
            "causal_v2_extraction_workers": config.causal_v2_extraction_workers,
            "causal_event_top_k": config.causal_event_top_k,
            "causal_v2_max_hops": config.causal_v2_max_hops,
            "causal_chain_top_k": config.causal_chain_top_k,
            "causal_context_max_items": config.causal_context_max_items,
            "causal_er_similarity_threshold": config.causal_er_similarity_threshold,
            "causal_er_text_threshold": config.causal_er_text_threshold,
            "causal_v2_min_edge_confidence": config.causal_v2_min_edge_confidence,
            "structure_rerank_enabled": config.structure_rerank_enabled,
            "structure_rerank_top_n": config.structure_rerank_top_n,
            "structure_rerank_bonus_weight": config.structure_rerank_bonus_weight,
            "structure_rerank_min_edge_support": config.structure_rerank_min_edge_support,
            "structure_rerank_max_top5_swaps": config.structure_rerank_max_top5_swaps,
            "structure_rerank_seed_top_k": config.structure_rerank_seed_top_k,
            "structure_rerank_max_hops": config.structure_rerank_max_hops,
            "structure_rerank_margin_threshold": config.structure_rerank_margin_threshold,
        },
        "overall_from_pipeline": {
            **(overall_retrieval_result or {}),
            **(overall_qa_results or {}),
        },
        "overall_recomputed": slice_metrics["overall"],
        "causal_slice": slice_metrics["causal_slice"],
        "nonempty_subgraph_slice": slice_metrics["nonempty_subgraph_slice"],
        "v2_metrics": v2_metrics,
        **({"oracle_reorder_qa": oracle_reorder_qa_results} if oracle_reorder_qa_results else {}),
        **({"oracle_select_qa": oracle_select_qa_results} if oracle_select_qa_results else {}),
        **({"setwise_selector_qa": setwise_selector_results} if setwise_selector_results else {}),
        **({"cross_encoder_rerank_qa": cross_encoder_rerank_results} if cross_encoder_rerank_results else {}),
        "examples": [
            {
                "question": query_solution.question,
                "query_type": (
                    (query_solution.retrieval_trace or {}).get("router_label")
                    if getattr(config, "causal_engine_version", "legacy") == "v2"
                    else route_query_type(query_solution.question)
                ),
                "answer": query_solution.answer if not retrieval_only else None,
                "gold_answers": query_solution.gold_answers if not retrieval_only else None,
                "docs": query_solution.docs[:3],
                "retrieved_doc_ids": serialize_retrieved_doc_ids(query_solution.docs, doc_text_to_chunk_id),
                "retrieval_trace": query_solution.retrieval_trace or {},
            }
            for query_solution in query_solutions
        ],
    }

    output_json = args.output_json
    if output_json is None:
        report_dir = Path(save_dir) / "eval_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_json = report_dir / f"causal_eval_{dataset_name}_{len(samples)}_{args.llm_name.replace('/', '_')}.json"
    else:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print_result = {
        "output_json": str(output_json),
        "overall_recomputed": result["overall_recomputed"],
        "causal_slice": result["causal_slice"],
        "nonempty_subgraph_slice": result["nonempty_subgraph_slice"],
        "v2_metrics": result["v2_metrics"],
    }
    if oracle_reorder_qa_results:
        print_result["oracle_reorder_qa"] = oracle_reorder_qa_results
    if setwise_selector_results:
        print_result["setwise_selector_qa"] = setwise_selector_results
    if cross_encoder_rerank_results:
        print_result["cross_encoder_rerank_qa"] = cross_encoder_rerank_results
    if gold_doc_reader:
        print_result["mode"] = "gold_doc_reader"
    print(json.dumps(print_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
