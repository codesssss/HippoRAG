#!/usr/bin/env python3
"""Export dense-only top-N retrieval pools in the external-pool JSON format.

This is the substrate-control counterpart to ``export_proprag_pool.py``.  It
does not invoke HippoRAG graph retrieval.  It loads cached passage embeddings,
embeds queries with the same VLLM/NV endpoint, ranks passages by cosine
similarity, and writes a fixed pool that can be consumed by
``eval_causal_qwen3.py --external_pool_json``.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DEFAULT_SAVE_DIR = Path("outputs_step0_general_nvembed")
DATASET_FILE_ALIASES = {
    "nq": "nq_rear",
    "natural_questions": "nq_rear",
}


def resolve_dataset_file_stem(dataset_name: str | None) -> str:
    normalized = str(dataset_name or "").strip().lower()
    return DATASET_FILE_ALIASES.get(normalized, str(dataset_name or "").strip())


def extract_title(doc: str) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def string_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_answer_alias_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                return [cleaned]
            return parse_answer_alias_values(parsed)
        return [cleaned]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        aliases: List[str] = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]


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
            paragraphs = []
            for item in sample["paragraphs"]:
                if item.get("is_supporting") is False:
                    continue
                paragraphs.append(item)
            docs = [
                item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"])
                for item in paragraphs
            ]
        gold_docs.append(sorted(set(docs)))
    return gold_docs


def get_gold_answers(samples: Sequence[Dict[str, Any]]) -> List[List[str]]:
    gold_answers: List[List[str]] = []
    for sample in samples:
        answers: List[str]
        if "answer" in sample or "gold_ans" in sample:
            answer = sample["answer"] if "answer" in sample else sample["gold_ans"]
            answers = parse_answer_alias_values(answer)
        elif "reference" in sample:
            answers = parse_answer_alias_values(sample["reference"])
        elif "obj" in sample:
            answers = []
            answers.extend(parse_answer_alias_values(sample.get("obj")))
            answers.extend(parse_answer_alias_values(sample.get("possible_answers")))
            answers.extend(parse_answer_alias_values(sample.get("o_wiki_title")))
            answers.extend(parse_answer_alias_values(sample.get("o_aliases")))
        else:
            raise ValueError("Sample has no recognized answer field.")
        if "answer_aliases" in sample:
            answers.extend(parse_answer_alias_values(sample["answer_aliases"]))
        gold_answers.append(sorted({str(item).strip() for item in answers if str(item).strip()}))
    return gold_answers


def compute_title_recall(
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
    k_values: Iterable[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in k_values:
        scores: List[float] = []
        for gold, retrieved in zip(gold_docs, retrieved_docs):
            gold_titles = {extract_title(doc) for doc in gold}
            retrieved_titles = {extract_title(doc) for doc in list(retrieved)[: int(k)]}
            scores.append(0.0 if not gold_titles else len(gold_titles & retrieved_titles) / len(gold_titles))
        metrics[f"Recall@{int(k)}"] = round(float(np.mean(scores)) if scores else 0.0, 4)
    return metrics


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return matrix / norms


def embed_queries(
    queries: Sequence[str],
    *,
    model_name: str,
    base_url: str,
    batch_size: int,
) -> np.ndarray:
    model_id = model_name[len("VLLM/") :] if model_name.startswith("VLLM/") else model_name
    outputs: List[np.ndarray] = []
    headers = {"Content-Type": "application/json"}
    for start in tqdm(range(0, len(queries), batch_size), desc="Embedding queries"):
        batch = list(queries[start : start + batch_size])
        response = requests.post(
            base_url,
            headers=headers,
            json={"model": model_id, "input": batch},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        outputs.append(np.asarray([row["embedding"] for row in payload["data"]], dtype=np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def resolve_chunk_embedding_path(save_dir: Path, dataset: str, llm_name: str, embedding_name: str) -> Path:
    suffix = f"{llm_name}_{embedding_name.replace('/', '_')}"
    dataset_file_stem = resolve_dataset_file_stem(dataset)
    candidates = [
        save_dir / f"{dataset}_{suffix}" / "chunk_embeddings" / "vdb_chunk.parquet",
        save_dir / f"{dataset_file_stem}_{suffix}" / "chunk_embeddings" / "vdb_chunk.parquet",
        save_dir / f"{dataset}" / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
        save_dir / f"{dataset_file_stem}" / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
        Path(f"outputs_step0_general_nvembed_{dataset}") / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
        Path(f"outputs_step0_general_nvembed_{dataset_file_stem}") / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find chunk embedding parquet. Tried:\n" + "\n".join(str(p) for p in candidates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--llm_name", default="qwen3-8b")
    parser.add_argument("--embedding_name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--normalize", type=string_to_bool, default=True)
    args = parser.parse_args()

    dataset_file_stem = resolve_dataset_file_stem(args.dataset)
    samples_path = args.data_root / f"{dataset_file_stem}.json"
    samples = json.loads(samples_path.read_text())
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]

    queries = [str(sample["question"]) for sample in samples]
    gold_docs = get_gold_docs(samples, args.dataset)
    gold_answers = get_gold_answers(samples)

    chunk_path = resolve_chunk_embedding_path(
        save_dir=args.save_dir,
        dataset=args.dataset,
        llm_name=str(args.llm_name),
        embedding_name=str(args.embedding_name),
    )
    chunk_df = pd.read_parquet(chunk_path)
    docs = [str(item) for item in chunk_df["content"].tolist()]
    doc_embeddings = np.stack(chunk_df["embedding"].to_numpy()).astype(np.float32)
    query_embeddings = embed_queries(
        queries,
        model_name=str(args.embedding_name),
        base_url=str(args.embedding_base_url),
        batch_size=int(args.embedding_batch_size),
    )
    if bool(args.normalize):
        doc_embeddings = normalize_rows(doc_embeddings)
        query_embeddings = normalize_rows(query_embeddings)

    pool_k = int(args.pool_k)
    records: List[Dict[str, Any]] = []
    retrieved_doc_lists: List[List[str]] = []
    for query_idx, query_vec in enumerate(tqdm(query_embeddings, desc="Ranking dense pools")):
        scores = doc_embeddings @ query_vec
        if pool_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -pool_k)[-pool_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        pool_docs = [docs[int(idx)] for idx in top_indices[:pool_k]]
        pool_scores = [float(scores[int(idx)]) for idx in top_indices[:pool_k]]
        retrieved_doc_lists.append(pool_docs)
        records.append(
            {
                "query_idx": int(query_idx),
                "question": queries[query_idx],
                "gold_answers": list(gold_answers[query_idx]),
                "gold_docs": list(gold_docs[query_idx]),
                "gold_titles": [extract_title(doc) for doc in gold_docs[query_idx]],
                "pool_k": pool_k,
                "pool_docs": pool_docs,
                "pool_titles": [extract_title(doc) for doc in pool_docs],
                "pool_doc_scores": pool_scores,
                "pool_doc_ids": [int(idx) for idx in top_indices[:pool_k]],
            }
        )

    recall = compute_title_recall(gold_docs, retrieved_doc_lists, [5, 20, 100])
    output = {
        "dataset": str(args.dataset),
        "limit": int(len(samples)),
        "pool_k": pool_k,
        "source": "dense_nvembed_retrieval_export",
        "chunk_embedding_path": str(chunk_path),
        "embedding_name": str(args.embedding_name),
        "embedding_base_url": str(args.embedding_base_url),
        "normalize": bool(args.normalize),
        "retrieval": {
            "recomputed_title_recall": recall,
            "dense_metrics": recall,
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
    print(json.dumps({"output_json": str(args.output_json), "dataset": args.dataset, "limit": len(samples), "pool_k": pool_k, "retrieval": recall}, indent=2))


if __name__ == "__main__":
    main()
