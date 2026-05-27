#!/usr/bin/env python3
"""Export AG-STO v12 retrieval pools in external-pool JSON format.

The exported JSON is consumable by ``scripts/eval_causal_qwen3.py
--external_pool_json``.  This script only builds the AG-STO pool; downstream
reader/selector experiments can reuse the existing DAEC selectors unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agsto_v12 import AGSTOConfig, AGSTORetriever  # noqa: E402
from src.agsto_v12.ranking import unique_ranked  # noqa: E402


DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DEFAULT_OPENIE_TEMPLATE = "outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json"
DEFAULT_CHUNK_EMBEDDING_TEMPLATE = (
    "outputs_step0_general_nvembed_{dataset}/qwen3-8b_VLLM_nvidia_NV-Embed-v2/"
    "chunk_embeddings/vdb_chunk.parquet"
)


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


def get_gold_docs(samples: Sequence[Mapping[str, Any]], dataset_name: str) -> List[List[str]]:
    gold_docs: List[List[str]] = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            gold_title_and_content = [item for item in sample["context"] if item[0] in gold_titles]
            if str(dataset_name).startswith("hotpotqa"):
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


def get_gold_answers(samples: Sequence[Mapping[str, Any]]) -> List[List[str]]:
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
    *,
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


def load_openie_docs(path: Path) -> List[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload.get("docs", []) if isinstance(payload, Mapping) else payload
    if not isinstance(docs, list):
        raise ValueError(f"OpenIE input must be a list or mapping with docs: {path}")
    return [doc for doc in docs if isinstance(doc, Mapping)]


def load_chunk_embedding_matrix(openie_docs: Sequence[Mapping[str, Any]], path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    if len(frame) != len(openie_docs):
        raise ValueError(f"Chunk embedding row count mismatch: {len(frame)} embeddings vs {len(openie_docs)} OpenIE docs")
    contents = frame["content"].astype(str).tolist()
    if openie_docs:
        expected_first = str(openie_docs[0].get("passage") or "")
        expected_last = str(openie_docs[-1].get("passage") or "")
        if contents[0] != expected_first or contents[-1] != expected_last:
            raise ValueError(
                f"Chunk embedding order/content mismatch for {path}; "
                "native dense anchors require embeddings to align with OpenIE doc indices."
            )
    matrix = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def maybe_prefix_dense_queries(
    *,
    queries: Sequence[str],
    instruction: str,
    instruction_mode: str,
) -> List[str]:
    if str(instruction_mode) == "prefixed":
        return [f"{instruction}\n\n{query}" for query in queries]
    return [str(query) for query in queries]


def embed_dense_queries(
    *,
    queries: Sequence[str],
    embedding_base_url: str,
    embedding_model: str,
    batch_size: int,
    timeout: float,
    instruction: str,
    instruction_mode: str,
) -> np.ndarray:
    texts = maybe_prefix_dense_queries(
        queries=queries,
        instruction=instruction,
        instruction_mode=instruction_mode,
    )
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    embeddings: List[np.ndarray] = []
    headers = {"Content-Type": "application/json"}
    for start in range(0, len(texts), max(int(batch_size), 1)):
        batch = texts[start : start + max(int(batch_size), 1)]
        payload = {"model": embedding_model, "input": batch}
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = requests.post(embedding_base_url, headers=headers, json=payload, timeout=float(timeout))
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: int(item.get("index", 0)))
                embeddings.extend(np.asarray(item["embedding"], dtype=np.float32) for item in data)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= 5:
                    raise
        if last_error is not None:
            raise last_error
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def dense_anchor_rows(query_embeddings: np.ndarray, chunk_embedding_matrix: np.ndarray, top_k: int) -> List[List[int]]:
    if query_embeddings.size == 0 or chunk_embedding_matrix.size == 0:
        return []
    keep_k = min(max(int(top_k), 1), int(chunk_embedding_matrix.shape[0]))
    scores = query_embeddings @ chunk_embedding_matrix.T
    rows: List[List[int]] = []
    for row_scores in scores:
        if keep_k >= len(row_scores):
            ranked = np.argsort(-row_scores)
        else:
            candidates = np.argpartition(-row_scores, keep_k - 1)[:keep_k]
            ranked = candidates[np.argsort(-row_scores[candidates])]
        rows.append([int(doc_idx) for doc_idx in ranked[:keep_k]])
    return rows


def passage_for_doc(openie_docs: Sequence[Mapping[str, Any]], doc_idx: int) -> str:
    if 0 <= int(doc_idx) < len(openie_docs):
        return str(openie_docs[int(doc_idx)].get("passage") or "")
    return ""


def score_pool_docs(doc_indices: Sequence[int], *, selected_doc_indices: Sequence[int]) -> List[float]:
    selected = set(unique_ranked(selected_doc_indices))
    pool_size = max(len(doc_indices), 1)
    scores: List[float] = []
    for rank, doc_idx in enumerate(doc_indices, start=1):
        base = float(pool_size - rank + 1)
        if int(doc_idx) in selected:
            base += float(pool_size)
        scores.append(base)
    return scores


def build_agsto_pool_records(
    *,
    dataset: str,
    samples: Sequence[Mapping[str, Any]],
    openie_docs: Sequence[Mapping[str, Any]],
    config: AGSTOConfig,
    pool_k: int,
    include_candidate_fill: bool,
    native_dense_anchor_rows: Sequence[Sequence[int]] | None = None,
    semantic_query_embeddings: np.ndarray | None = None,
    chunk_embedding_matrix: np.ndarray | None = None,
    semantic_residual_weight: float = 8.0,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    retriever = AGSTORetriever.from_openie_docs(openie_docs, config=config)
    gold_docs = get_gold_docs(samples, dataset)
    gold_answers = get_gold_answers(samples)
    records: List[Dict[str, Any]] = []
    retrieved_doc_lists: List[List[str]] = []

    for query_idx, sample in enumerate(tqdm(samples, desc=f"AG-STO pool export ({dataset})")):
        question = str(sample.get("question") or "")
        native_dense_docs = (
            list(native_dense_anchor_rows[query_idx])
            if native_dense_anchor_rows is not None and query_idx < len(native_dense_anchor_rows)
            else None
        )
        semantic_query_embedding = (
            semantic_query_embeddings[query_idx]
            if semantic_query_embeddings is not None and query_idx < int(semantic_query_embeddings.shape[0])
            else None
        )
        result = retriever.retrieve_native(
            query=question,
            native_dense_doc_indices=native_dense_docs,
            semantic_query_embedding=semantic_query_embedding,
            chunk_embedding_matrix=chunk_embedding_matrix,
            semantic_residual_weight=semantic_residual_weight,
        )
        selected_doc_indices = unique_ranked((result.get("selected_evidence_set", {}) or {}).get("doc_indices", []) or [])
        pool_doc_indices = unique_ranked(
            list(selected_doc_indices)
            + list(result.get("retrieved_doc_indices", []) or [])
            + (list(result.get("candidate_doc_indices", []) or []) if include_candidate_fill else [])
        )
        pool_doc_indices = [idx for idx in pool_doc_indices if passage_for_doc(openie_docs, int(idx))][: int(pool_k)]
        pool_docs = [passage_for_doc(openie_docs, int(doc_idx)) for doc_idx in pool_doc_indices]
        pool_scores = score_pool_docs(pool_doc_indices, selected_doc_indices=selected_doc_indices)
        retrieved_doc_lists.append(pool_docs)
        records.append(
            {
                "query_idx": int(query_idx),
                "question": question,
                "gold_answers": list(gold_answers[query_idx]),
                "gold_docs": list(gold_docs[query_idx]),
                "gold_titles": [extract_title(doc) for doc in gold_docs[query_idx]],
                "pool_k": int(pool_k),
                "pool_docs": pool_docs,
                "pool_titles": [extract_title(doc) for doc in pool_docs],
                "pool_doc_scores": pool_scores,
                "pool_doc_ids": [int(doc_idx) for doc_idx in pool_doc_indices],
                "agsto": {
                    "native_dense_anchor_enabled": native_dense_anchor_rows is not None,
                    "native_dense_doc_indices": list(native_dense_docs or [])[: int(pool_k)],
                    "semantic_residual_enabled": semantic_query_embedding is not None and chunk_embedding_matrix is not None,
                    "selected_doc_indices": selected_doc_indices,
                    "retrieved_doc_indices": list(result.get("retrieved_doc_indices", []) or [])[: int(pool_k)],
                    "candidate_doc_count": int(result.get("candidate_doc_count", 0) or 0),
                    "selection_policy": result.get("selection_policy"),
                    "evidence_completion_policy": result.get("evidence_completion_policy"),
                    "graph_obligated_completion": result.get("graph_obligated_completion", {}),
                    "selected_evidence_set": result.get("selected_evidence_set", {}),
                },
            }
        )

    return records, compute_title_recall(gold_docs=gold_docs, retrieved_docs=retrieved_doc_lists, k_values=[5, 20, 100])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--openie_results", type=Path, default=None)
    parser.add_argument("--openie_template", type=str, default=DEFAULT_OPENIE_TEMPLATE)
    parser.add_argument("--policy", choices=["base", "graph"], default="graph")
    parser.add_argument("--retrieval_top_k", type=int, default=0)
    parser.add_argument("--evidence_set_size", type=int, default=5)
    parser.add_argument("--stable_anchor_k", type=int, default=2)
    parser.add_argument("--proposal_candidate_depth", type=int, default=20)
    parser.add_argument("--support_proposal_depth", type=int, default=10)
    parser.add_argument("--candidate_limit", type=int, default=160)
    parser.add_argument("--beam_size", type=int, default=12)
    parser.add_argument("--set_search_policy", choices=["beam", "mct"], default="beam")
    parser.add_argument("--max_endpoint_degree", type=int, default=30)
    parser.add_argument("--include_candidate_fill", type=string_to_bool, default=True)
    parser.add_argument("--native_dense_anchor", type=string_to_bool, default=False)
    parser.add_argument("--native_dense_anchor_top_k", type=int, default=20)
    parser.add_argument("--chunk_embedding_path", type=Path, default=None)
    parser.add_argument("--chunk_embedding_template", type=str, default=DEFAULT_CHUNK_EMBEDDING_TEMPLATE)
    parser.add_argument("--embedding_base_url", type=str, default="")
    parser.add_argument("--embedding_model", type=str, default="")
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_timeout", type=float, default=120.0)
    parser.add_argument("--dense_query_instruction_mode", choices=["raw", "prefixed"], default="raw")
    parser.add_argument("--dense_query_instruction", default="Given a question, retrieve relevant documents that best answer the question.")
    parser.add_argument("--semantic_residual_weight", type=float, default=8.0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    samples_path = Path(args.data_root) / f"{args.dataset}.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]
    openie_path = (
        Path(args.openie_results)
        if args.openie_results is not None
        else Path(str(args.openie_template).format(dataset=args.dataset))
    )
    openie_docs = load_openie_docs(openie_path)
    chunk_embedding_matrix = None
    query_embeddings = None
    native_dense_rows = None
    chunk_embedding_path = None
    if bool(args.native_dense_anchor):
        if not args.embedding_base_url:
            raise ValueError("--native_dense_anchor requires --embedding_base_url")
        if not args.embedding_model:
            raise ValueError("--native_dense_anchor requires --embedding_model")
        chunk_embedding_path = (
            Path(args.chunk_embedding_path)
            if args.chunk_embedding_path is not None
            else Path(str(args.chunk_embedding_template).format(dataset=args.dataset))
        )
        chunk_embedding_matrix = load_chunk_embedding_matrix(openie_docs, chunk_embedding_path)
        query_embeddings = embed_dense_queries(
            queries=[str(sample.get("question") or "") for sample in samples],
            embedding_base_url=str(args.embedding_base_url),
            embedding_model=str(args.embedding_model),
            batch_size=int(args.embedding_batch_size),
            timeout=float(args.embedding_timeout),
            instruction=str(args.dense_query_instruction),
            instruction_mode=str(args.dense_query_instruction_mode),
        )
        native_dense_rows = dense_anchor_rows(
            query_embeddings,
            chunk_embedding_matrix,
            top_k=max(int(args.native_dense_anchor_top_k), int(args.evidence_set_size), 1),
        )
    config = AGSTOConfig(
        policy=str(args.policy),
        retrieval_top_k=max(
            int(args.retrieval_top_k) if int(args.retrieval_top_k) > 0 else int(args.pool_k),
            int(args.evidence_set_size),
            1,
        ),
        evidence_set_size=max(int(args.evidence_set_size), 1),
        stable_anchor_k=max(int(args.stable_anchor_k), 0),
        proposal_candidate_depth=max(int(args.proposal_candidate_depth), int(args.evidence_set_size), 1),
        support_proposal_depth=max(int(args.support_proposal_depth), int(args.evidence_set_size), 1),
        candidate_limit=max(int(args.candidate_limit), int(args.pool_k), 1),
        beam_size=max(int(args.beam_size), 1),
        set_search_policy=str(args.set_search_policy),
        max_endpoint_degree=max(int(args.max_endpoint_degree), 2),
    )
    records, recall = build_agsto_pool_records(
        dataset=str(args.dataset),
        samples=samples,
        openie_docs=openie_docs,
        config=config,
        pool_k=int(args.pool_k),
        include_candidate_fill=bool(args.include_candidate_fill),
        native_dense_anchor_rows=native_dense_rows,
        semantic_query_embeddings=query_embeddings,
        chunk_embedding_matrix=chunk_embedding_matrix,
        semantic_residual_weight=max(float(args.semantic_residual_weight), 0.0),
    )
    output = {
        "dataset": str(args.dataset),
        "limit": int(len(samples)),
        "pool_k": int(args.pool_k),
        "source": "agsto_v12_pool_export",
        "openie_path": str(openie_path),
        "config": {
            "policy": config.policy,
            "completion_policy": config.completion_policy,
            "retrieval_top_k": int(config.retrieval_top_k),
            "evidence_set_size": int(config.evidence_set_size),
            "stable_anchor_k": int(config.stable_anchor_k),
            "proposal_candidate_depth": int(config.proposal_candidate_depth),
            "support_proposal_depth": int(config.support_proposal_depth),
            "candidate_limit": int(config.candidate_limit),
            "beam_size": int(config.beam_size),
            "set_search_policy": str(config.set_search_policy),
            "max_endpoint_degree": int(config.max_endpoint_degree),
            "include_candidate_fill": bool(args.include_candidate_fill),
            "native_dense_anchor": bool(args.native_dense_anchor),
            "native_dense_anchor_top_k": int(args.native_dense_anchor_top_k),
            "chunk_embedding_path": None if chunk_embedding_path is None else str(chunk_embedding_path),
            "embedding_base_url": str(args.embedding_base_url),
            "embedding_model": str(args.embedding_model),
            "dense_query_instruction_mode": str(args.dense_query_instruction_mode),
            "semantic_residual_weight": max(float(args.semantic_residual_weight), 0.0),
        },
        "retrieval": {
            "recomputed_title_recall": recall,
            "agsto_metrics": recall,
        },
        "records": records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "dataset": str(args.dataset),
                "limit": int(len(samples)),
                "pool_k": int(args.pool_k),
                "recomputed_title_recall": recall,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
