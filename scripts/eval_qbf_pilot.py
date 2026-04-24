#!/usr/bin/env python3
"""Run retrieval-only QBF operator pilots.

The script compares HippoRAG-style PPR, PPR+terminal-schema rerank, and QBF
variants on the first N examples of a dataset. It saves parseable JSON only; it
does not call the QA reader.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.qbf import (
    QBFFactEntityIndex,
    QBFResult,
    infer_terminal_schema,
    min_max,
    rerank_doc_ids_by_scores,
    stable_topk_indices,
)
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import compute_mdhash_id


DEFAULT_VARIANTS = (
    "ppr",
    "ppr_chi_rerank",
    "ppr_qbf_rerank",
    "qbf_schema_relational",
    "qbf_schema_all_edge",
    "qbf_random_chi",
    "qbf_oracle_relation",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only QBF pilot evaluator.")
    parser.add_argument("--dataset", type=str, required=True, choices=["2wikimultihopqa", "hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general_nvembed")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--llm_request_name", type=str, default=None)
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8041/v1")
    parser.add_argument("--embedding_name", type=str, default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--retrieval_top_k", type=int, default=100)
    parser.add_argument("--candidate_rerank_k", type=int, default=100)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--qbf_alpha", type=float, default=0.5)
    parser.add_argument("--qbf_max_iter", type=int, default=50)
    parser.add_argument("--qbf_tol", type=float, default=1e-6)
    parser.add_argument("--qbf_psi_floor", type=float, default=0.05)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dataset = args.dataset
    run_save_dir = args.save_dir if args.save_dir == "outputs" else f"{args.save_dir}_{dataset}"

    corpus_path = ROOT_DIR / f"reproduce/dataset/{dataset}_corpus.json"
    sample_path = ROOT_DIR / f"reproduce/dataset/{dataset}.json"
    corpus = json.load(corpus_path.open())
    samples = json.load(sample_path.open())
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [str(sample["question"]) for sample in samples]
    gold_docs = get_gold_docs(samples, dataset, corpus=corpus)

    config = BaseConfig(
        save_dir=run_save_dir,
        dataset=dataset,
        llm_name=args.llm_name,
        llm_request_name=args.llm_request_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
        causal_enabled=False,
        causal_engine_version="v2",
        causal_v2_base_retrieval_mode="legacy_fact_graph",
        causal_v2_legacy_preferred_embedding_name=args.embedding_name,
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        corpus_len=len(corpus),
        openie_mode="online",
        structure_rerank_enabled=True,
    )

    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs=docs)
    if not hipporag.ready_to_retrieve_v2:
        hipporag._prepare_retrieval_objects_v2()  # pylint: disable=protected-access
    hipporag._get_passage_query_embeddings(queries)  # pylint: disable=protected-access

    qbf_index = QBFFactEntityIndex.from_hipporag(hipporag)
    doc_texts = [
        hipporag.chunk_embedding_store.get_row(chunk_id)["content"]
        for chunk_id in hipporag.passage_node_keys
    ]
    doc_text_to_idx, title_to_idx = build_doc_lookup(doc_texts)
    gold_doc_idx_sets = [
        resolve_gold_doc_indices(gold, doc_text_to_idx=doc_text_to_idx, title_to_idx=title_to_idx)
        for gold in gold_docs
    ]

    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    unsupported = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unsupported:
        raise ValueError(f"Unsupported QBF variants: {unsupported}")

    variant_docs: Dict[str, List[List[str]]] = {variant: [] for variant in variants}
    variant_doc_ids: Dict[str, List[List[int]]] = {variant: [] for variant in variants}
    records: List[Dict[str, Any]] = []

    start_time = time.time()
    for query_idx, query in enumerate(tqdm(queries, desc=f"QBF pilot {dataset}")):
        query_start = time.time()
        full_fact_scores = hipporag.get_fact_scores(query)
        ppr_doc_ids, ppr_doc_scores, ppr_trace = run_ppr_no_llm(
            hipporag=hipporag,
            query=query,
            query_fact_scores=full_fact_scores,
            top_k=max(args.retrieval_top_k, args.candidate_rerank_k),
        )
        ppr_doc_ids = fill_doc_ids(ppr_doc_ids, [], hipporag_docs=len(doc_texts), top_k=max(args.retrieval_top_k, args.candidate_rerank_k))
        schema = infer_terminal_schema(query)
        active_embedding_model = hipporag._active_embedding_model_for_fact_retrieval()  # pylint: disable=protected-access
        fact_chi = qbf_index.score_terminal_schema(active_embedding_model, schema.phrase)
        doc_chi = qbf_index.doc_max_fact_scores(fact_chi)
        oracle_phrase = build_oracle_relation_phrase(qbf_index, gold_doc_idx_sets[query_idx], fallback=schema.phrase)
        oracle_fact_chi = qbf_index.score_terminal_schema(active_embedding_model, oracle_phrase)

        per_variant_trace: Dict[str, Any] = {}
        for variant in variants:
            if variant == "ppr":
                doc_ids = fill_doc_ids(ppr_doc_ids, [], hipporag_docs=len(doc_texts), top_k=args.retrieval_top_k)
                scores = np.asarray(ppr_doc_scores[: len(doc_ids)], dtype=float)
                trace = dict(ppr_trace)
            elif variant == "ppr_chi_rerank":
                rerank_ids, scores = rerank_doc_ids_by_scores(
                    ppr_doc_ids[: args.candidate_rerank_k],
                    doc_chi,
                    top_k=args.retrieval_top_k,
                )
                doc_ids = fill_doc_ids(rerank_ids, ppr_doc_ids, hipporag_docs=len(doc_texts), top_k=args.retrieval_top_k)
                trace = {"operator": variant, "candidate_rerank_k": int(args.candidate_rerank_k)}
            elif variant == "ppr_qbf_rerank":
                qbf_doc_scores, qbf_trace = qbf_index.qbf_doc_scores(
                    query_fact_scores=full_fact_scores,
                    fact_chi=fact_chi,
                    alpha=args.qbf_alpha,
                    max_iter=args.qbf_max_iter,
                    tol=args.qbf_tol,
                    psi_floor=args.qbf_psi_floor,
                    include_doc_nodes=False,
                )
                rerank_ids, scores = rerank_doc_ids_by_scores(
                    ppr_doc_ids[: args.candidate_rerank_k],
                    qbf_doc_scores,
                    top_k=args.retrieval_top_k,
                )
                doc_ids = fill_doc_ids(rerank_ids, ppr_doc_ids, hipporag_docs=len(doc_texts), top_k=args.retrieval_top_k)
                trace = {**qbf_trace, "operator": variant, "candidate_rerank_k": int(args.candidate_rerank_k)}
            elif variant == "qbf_schema_relational":
                result = qbf_index.qbf_retrieve(
                    query_fact_scores=full_fact_scores,
                    fact_chi=fact_chi,
                    top_k=args.retrieval_top_k,
                    alpha=args.qbf_alpha,
                    max_iter=args.qbf_max_iter,
                    tol=args.qbf_tol,
                    psi_floor=args.qbf_psi_floor,
                    include_doc_nodes=False,
                )
                doc_ids = fill_doc_ids(result.sorted_doc_ids, ppr_doc_ids, hipporag_docs=len(doc_texts), top_k=args.retrieval_top_k)
                scores = result.sorted_doc_scores
                trace = result.trace
            elif variant == "qbf_schema_all_edge":
                result = qbf_index.qbf_retrieve(
                    query_fact_scores=full_fact_scores,
                    fact_chi=fact_chi,
                    top_k=args.retrieval_top_k,
                    alpha=args.qbf_alpha,
                    max_iter=args.qbf_max_iter,
                    tol=args.qbf_tol,
                    psi_floor=args.qbf_psi_floor,
                    include_doc_nodes=True,
                )
                doc_ids = fill_doc_ids(result.sorted_doc_ids, ppr_doc_ids, hipporag_docs=len(doc_texts), top_k=args.retrieval_top_k)
                scores = result.sorted_doc_scores
                trace = result.trace
            elif variant == "qbf_random_chi":
                random_chi = qbf_index.score_terminal_schema(
                    active_embedding_model,
                    schema.phrase,
                    random_seed_key=f"{dataset}:{query_idx}:{query}",
                )
                result = qbf_index.qbf_retrieve(
                    query_fact_scores=full_fact_scores,
                    fact_chi=random_chi,
                    top_k=args.retrieval_top_k,
                    alpha=args.qbf_alpha,
                    max_iter=args.qbf_max_iter,
                    tol=args.qbf_tol,
                    psi_floor=args.qbf_psi_floor,
                    include_doc_nodes=False,
                )
                doc_ids = fill_doc_ids(result.sorted_doc_ids, ppr_doc_ids, hipporag_docs=len(doc_texts), top_k=args.retrieval_top_k)
                scores = result.sorted_doc_scores
                trace = {**result.trace, "operator": variant}
            elif variant == "qbf_oracle_relation":
                result = qbf_index.qbf_retrieve(
                    query_fact_scores=full_fact_scores,
                    fact_chi=oracle_fact_chi,
                    top_k=args.retrieval_top_k,
                    alpha=args.qbf_alpha,
                    max_iter=args.qbf_max_iter,
                    tol=args.qbf_tol,
                    psi_floor=args.qbf_psi_floor,
                    include_doc_nodes=False,
                )
                doc_ids = fill_doc_ids(result.sorted_doc_ids, ppr_doc_ids, hipporag_docs=len(doc_texts), top_k=args.retrieval_top_k)
                scores = result.sorted_doc_scores
                trace = {**result.trace, "operator": variant, "oracle_relation_phrase": oracle_phrase}
            else:
                raise AssertionError(f"Unhandled variant: {variant}")

            doc_list = [doc_texts[int(doc_id)] for doc_id in doc_ids]
            variant_doc_ids[variant].append([int(doc_id) for doc_id in doc_ids])
            variant_docs[variant].append(doc_list)
            per_variant_trace[variant] = {
                **trace,
                "top5_doc_ids": [int(doc_id) for doc_id in doc_ids[:5]],
                "top5_titles": [extract_title(doc_texts[int(doc_id)]) for doc_id in doc_ids[:5]],
                "score_preview": [float(score) for score in np.asarray(scores[:5], dtype=float).reshape(-1).tolist()],
            }

        records.append(
            {
                "query_idx": int(query_idx),
                "question": query,
                "gold_titles": [extract_title(doc) for doc in gold_docs[query_idx]],
                "terminal_schema": {
                    "label": schema.label,
                    "phrase": schema.phrase,
                    "matched_rule": schema.matched_rule,
                    "oracle_relation_phrase": oracle_phrase,
                },
                "latency_sec": round(float(time.time() - query_start), 4),
                "variants": per_variant_trace,
            }
        )

    metrics = {
        variant: compute_retrieval_metrics(gold_docs, variant_docs[variant], k_list=[5, 20, 100])
        for variant in variants
    }
    elapsed = time.time() - start_time
    payload = {
        "dataset": dataset,
        "limit": len(samples),
        "save_dir": run_save_dir,
        "config": {
            "retrieval_top_k": int(args.retrieval_top_k),
            "candidate_rerank_k": int(args.candidate_rerank_k),
            "linking_top_k": int(args.linking_top_k),
            "qbf_alpha": float(args.qbf_alpha),
            "qbf_max_iter": int(args.qbf_max_iter),
            "qbf_tol": float(args.qbf_tol),
            "qbf_psi_floor": float(args.qbf_psi_floor),
            "variants": variants,
            "fact_count": int(qbf_index.num_facts),
            "entity_count": int(qbf_index.num_entities),
            "doc_count": int(qbf_index.num_docs),
            "v2_base_retrieval_status": hipporag.v2_base_retrieval_status,
            "v2_base_retrieval_asset_dir": hipporag.v2_base_retrieval_asset_dir,
        },
        "metrics": metrics,
        "elapsed_sec": round(float(elapsed), 4),
        "avg_latency_sec": round(float(elapsed / max(1, len(samples))), 4),
        "records": records,
    }

    output_json = args.output_json
    if output_json is None:
        output_dir = ROOT_DIR / "run_logs" / "qbf_pilot_20260424"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = str(output_dir / f"{dataset}_limit{len(samples)}.json")
    else:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output_json": output_json, "metrics": metrics}, ensure_ascii=False, indent=2))


def run_ppr_no_llm(hipporag: HippoRAG,
                   query: str,
                   query_fact_scores: np.ndarray,
                   top_k: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    top_fact_indices = stable_topk_indices(query_fact_scores, hipporag.global_config.linking_top_k)
    top_facts: List[Tuple[str, str, str]] = []
    top_indices: List[int] = []
    for fact_idx in top_fact_indices.tolist():
        fact_id = str(hipporag.fact_node_keys[int(fact_idx)])
        triple = (getattr(hipporag, "fact_id_to_triple", {}) or {}).get(fact_id)
        if triple is None:
            continue
        top_facts.append(tuple(str(part) for part in triple))
        top_indices.append(int(fact_idx))

    if not top_facts:
        sorted_doc_ids, sorted_doc_scores = hipporag.dense_passage_retrieval(query)
        return sorted_doc_ids[:top_k], sorted_doc_scores[:top_k], {
            "operator": "ppr",
            "route": "dense_fallback_no_top_facts",
            "top_fact_count": 0,
        }

    try:
        sorted_doc_ids, sorted_doc_scores = hipporag.graph_search_with_fact_entities(
            query=query,
            link_top_k=hipporag.global_config.linking_top_k,
            query_fact_scores=query_fact_scores,
            top_k_facts=top_facts,
            top_k_fact_indices=top_indices,
            passage_node_weight=hipporag.global_config.passage_node_weight,
        )
        return sorted_doc_ids[:top_k], sorted_doc_scores[:top_k], {
            "operator": "ppr",
            "route": "fact_graph_no_llm_rerank",
            "top_fact_count": len(top_facts),
        }
    except Exception as exc:  # pylint: disable=broad-except
        sorted_doc_ids, sorted_doc_scores = hipporag.dense_passage_retrieval(query)
        return sorted_doc_ids[:top_k], sorted_doc_scores[:top_k], {
            "operator": "ppr",
            "route": "dense_fallback_after_ppr_exception",
            "top_fact_count": len(top_facts),
            "error": str(exc),
        }


def compute_retrieval_metrics(gold_docs: Sequence[Sequence[str]],
                              retrieved_docs: Sequence[Sequence[str]],
                              k_list: Sequence[int]) -> Dict[str, Any]:
    recall_metric = RetrievalRecall()
    recall, per_query = recall_metric.calculate_metric_scores(
        gold_docs=[list(gold) for gold in gold_docs],
        retrieved_docs=[list(docs) for docs in retrieved_docs],
        k_list=list(k_list),
    )
    support_complete: Dict[str, float] = {}
    for k in k_list:
        hits = []
        for gold, docs in zip(gold_docs, retrieved_docs):
            gold_set = set(gold)
            hits.append(1.0 if gold_set and gold_set.issubset(set(docs[:k])) else 0.0)
        support_complete[f"SupportComplete@{k}"] = round(float(np.mean(hits)) if hits else 0.0, 4)

    depths = []
    for gold, docs in zip(gold_docs, retrieved_docs):
        gold_set = set(gold)
        if not gold_set:
            depths.append(None)
            continue
        found_depth = None
        seen: Set[str] = set()
        for rank, doc in enumerate(docs, start=1):
            seen.add(doc)
            if gold_set.issubset(seen):
                found_depth = rank
                break
        depths.append(found_depth)
    finite_depths = [depth for depth in depths if depth is not None]
    return {
        **recall,
        **support_complete,
        "MeanFullSupportDepth": round(float(np.mean(finite_depths)) if finite_depths else 0.0, 4),
        "FullSupportWithin100": round(float(len(finite_depths) / max(1, len(depths))), 4),
        "per_query": per_query,
    }


def fill_doc_ids(primary_doc_ids: Sequence[int],
                 fallback_doc_ids: Sequence[int],
                 *,
                 hipporag_docs: int,
                 top_k: int) -> np.ndarray:
    selected: List[int] = []
    seen: Set[int] = set()
    for raw_doc_id in list(primary_doc_ids) + list(fallback_doc_ids) + list(range(hipporag_docs)):
        doc_id = int(raw_doc_id)
        if doc_id < 0 or doc_id >= hipporag_docs or doc_id in seen:
            continue
        selected.append(doc_id)
        seen.add(doc_id)
        if len(selected) >= top_k:
            break
    return np.asarray(selected, dtype=int)


def build_oracle_relation_phrase(qbf_index: QBFFactEntityIndex,
                                 gold_doc_indices: Set[int],
                                 fallback: str) -> str:
    predicates: List[str] = []
    for triple, doc_indices in zip(qbf_index.fact_triples, qbf_index.fact_doc_indices):
        if gold_doc_indices.intersection(doc_indices):
            predicates.append(str(triple[1]))
    if not predicates:
        return fallback
    counts = Counter(predicates)
    return " ".join(predicate for predicate, _ in counts.most_common(6))


def build_doc_lookup(doc_texts: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    doc_text_to_idx: Dict[str, int] = {}
    title_to_indices: Dict[str, List[int]] = {}
    for idx, doc_text in enumerate(doc_texts):
        doc_text_to_idx[str(doc_text)] = int(idx)
        title_to_indices.setdefault(normalize_title(extract_title(doc_text)), []).append(int(idx))
    title_to_idx = {
        title: indices[0]
        for title, indices in title_to_indices.items()
        if title and len(indices) == 1
    }
    return doc_text_to_idx, title_to_idx


def resolve_gold_doc_indices(gold_docs: Sequence[str],
                             *,
                             doc_text_to_idx: Dict[str, int],
                             title_to_idx: Dict[str, int]) -> Set[int]:
    resolved: Set[int] = set()
    for doc in gold_docs:
        if doc in doc_text_to_idx:
            resolved.add(int(doc_text_to_idx[doc]))
            continue
        title = normalize_title(extract_title(doc))
        if title in title_to_idx:
            resolved.add(int(title_to_idx[title]))
    return resolved


def get_gold_docs(samples: List[dict], dataset_name: str, corpus: List[dict]) -> List[List[str]]:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_title = {item[0] for item in sample["supporting_facts"]}
            gold_title_and_content_list = [item for item in sample["context"] if item[0] in gold_title]
            if dataset_name.startswith("hotpotqa"):
                gold_doc = [item[0] + "\n" + "".join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content_list]
        elif "contexts" in sample:
            gold_doc = [item["title"] + "\n" + item["text"] for item in sample["contexts"] if item["is_supporting"]]
        elif "document" in sample and isinstance(sample["document"], dict):
            document_id = str(sample["document"].get("id", "")).strip()
            gold_doc = [
                item["title"] + "\n" + item["text"]
                for item in corpus
                if str(item.get("idx", "")).startswith(f"{document_id}_")
            ]
        else:
            gold_paragraphs = []
            for item in sample["paragraphs"]:
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [
                item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"])
                for item in gold_paragraphs
            ]
        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def extract_title(doc_text: str) -> str:
    return str(doc_text or "").split("\n", 1)[0].strip()


def normalize_title(title: str) -> str:
    return " ".join(str(title or "").lower().split())


if __name__ == "__main__":
    main()
