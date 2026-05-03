#!/usr/bin/env python3
"""Export HippoRAG (legacy_fact_graph) retrieval pools in external-pool JSON format.

Runs HippoRAG PPR-based retrieval for each query and writes a fixed pool
consumable by ``eval_causal_qwen3.py --external_pool_json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

DEFAULT_DATA_ROOT = Path("reproduce/dataset")


def extract_title(doc: str) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def get_gold_docs(samples: Sequence[Dict[str, Any]], dataset_name: str) -> List[List[str]]:
    gold_docs: List[List[str]] = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            gold_title_and_content = [item for item in sample["context"] if item[0] in gold_titles]
            if dataset_name.startswith("hotpotqa"):
                docs = [item[0] + "\n" + "".join(item[1]) for item in gold_title_and_content]
            else:
                docs = [item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content]
        elif "contexts" in sample:
            docs = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item.get("is_supporting")
            ]
        else:
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting") is not False]
            docs = [
                item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"])
                for item in paragraphs
            ]
        gold_docs.append(sorted(set(docs)))
    return gold_docs


def get_gold_answers(samples: Sequence[Dict[str, Any]]) -> List[List[str]]:
    gold_answers: List[List[str]] = []
    for sample in samples:
        if "answer" in sample or "gold_ans" in sample:
            answer = sample["answer"] if "answer" in sample else sample["gold_ans"]
        elif "reference" in sample:
            answer = sample["reference"]
        elif "obj" in sample:
            answer = [sample["obj"], sample["possible_answers"], sample["o_wiki_title"], *sample["o_aliases"]]
        else:
            raise ValueError("Sample has no recognized answer field.")
        answers = {answer} if isinstance(answer, str) else set(answer)
        if "answer_aliases" in sample:
            answers.update(sample["answer_aliases"])
        gold_answers.append(sorted(str(item) for item in answers))
    return gold_answers


def compute_title_recall(
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
    k_values: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in k_values:
        scores: List[float] = []
        for gold, retrieved in zip(gold_docs, retrieved_docs):
            gold_titles = {extract_title(doc) for doc in gold}
            retrieved_titles = {extract_title(doc) for doc in list(retrieved)[:int(k)]}
            scores.append(0.0 if not gold_titles else len(gold_titles & retrieved_titles) / len(gold_titles))
        metrics[f"Recall@{int(k)}"] = round(float(np.mean(scores)) if scores else 0.0, 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save_dir", default="outputs_step0_general_nvembed")
    parser.add_argument("--llm_name", default="qwen3-8b")
    parser.add_argument("--llm_request_name", default="qwen3-8b-train")
    parser.add_argument("--llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--embedding_name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--retrieval_top_k", type=int, default=100)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    samples_path = args.data_root / f"{args.dataset}.json"
    corpus_path = args.data_root / f"{args.dataset}_corpus.json"
    samples = json.loads(samples_path.read_text())
    corpus = json.loads(corpus_path.read_text())
    if int(args.limit) > 0:
        samples = samples[:int(args.limit)]

    queries = [str(sample["question"]) for sample in samples]
    gold_docs = get_gold_docs(samples, args.dataset)
    gold_answers = get_gold_answers(samples)

    docs = []
    for item in corpus:
        if "title" in item and "text" in item:
            docs.append(item["title"] + "\n" + item["text"])
        elif "title" in item and "body_text" in item:
            docs.append(item["title"] + "\n" + item["body_text"])
        else:
            docs.append(str(item.get("text", item.get("body_text", ""))))

    save_dir = str(args.save_dir)
    if save_dir == "outputs":
        save_dir = f"{save_dir}/{args.dataset}"
    else:
        save_dir = f"{save_dir}_{args.dataset}"

    config = BaseConfig(
        dataset=args.dataset,
        llm_name=args.llm_name,
        llm_request_name=args.llm_request_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
        save_dir=save_dir,
        retrieval_top_k=args.retrieval_top_k,
        corpus_len=len(docs),
        graph_type="facts_and_sim_passage_node_unidirectional",
        causal_v2_base_retrieval_mode="legacy_fact_graph",
        causal_engine_version="v2",
        causal_enabled=False,
        causal_context_max_items=0,
    )

    print(f"Initializing HippoRAG for {args.dataset} with legacy_fact_graph...")
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)

    print(f"Running retrieval for {len(queries)} queries...")
    query_solutions, overall_retrieval_result = hipporag.retrieve(
        queries=queries,
        gold_docs=gold_docs,
    )

    pool_k = int(args.pool_k)
    records: List[Dict[str, Any]] = []
    retrieved_doc_lists: List[List[str]] = []

    for qi, qs in enumerate(query_solutions):
        retrieved = qs.docs or []
        scores = qs.doc_scores.tolist() if qs.doc_scores is not None else [0.0] * len(retrieved)
        pool_docs = retrieved[:pool_k]
        pool_scores = [float(s) for s in scores[:pool_k]]
        retrieved_doc_lists.append(pool_docs)
        records.append({
            "query_idx": qi,
            "question": queries[qi],
            "gold_answers": list(gold_answers[qi]),
            "gold_docs": list(gold_docs[qi]),
            "gold_titles": [extract_title(doc) for doc in gold_docs[qi]],
            "pool_k": len(pool_docs),
            "pool_docs": pool_docs,
            "pool_titles": [extract_title(doc) for doc in pool_docs],
            "pool_doc_scores": pool_scores,
            "pool_doc_ids": list(range(len(pool_docs))),
        })

    recall = compute_title_recall(gold_docs, retrieved_doc_lists, [5, 20, 100])
    output = {
        "dataset": str(args.dataset),
        "limit": len(samples),
        "pool_k": pool_k,
        "source": "hipporag_legacy_fact_graph_export",
        "save_dir": str(args.save_dir),
        "config": {
            "base_retrieval_mode": "legacy_fact_graph",
            "retrieval_top_k": args.retrieval_top_k,
            "embedding_name": str(args.embedding_name),
        },
        "retrieval": {
            "recomputed_title_recall": recall,
            "hipporag_metrics": recall,
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False))
    print(json.dumps({
        "output_json": str(args.output_json),
        "dataset": args.dataset,
        "limit": len(samples),
        "pool_k": pool_k,
        "retrieval": recall,
    }, indent=2))


if __name__ == "__main__":
    main()
