#!/usr/bin/env python3
"""Run LinearRAG retrieval under the local saved-docs reader protocol.

This adapter keeps /mnt/nvme/code/LinearRAG unchanged. It builds a LinearRAG
index over the local benchmark corpus, runs retrieval only, maps returned
passages back to corpus ids, and writes the saved-docs reader-input format
consumed by evidence_transition_graphragv4_fact_witnessed_sto/run_docs_reader_qa.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
LINEARRAG_ROOT = Path("/mnt/nvme/code/LinearRAG")
if str(LINEARRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LINEARRAG_ROOT))


DISPLAY_NAMES = {
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultiHopQA",
    "musique": "MuSiQue",
}

PASSAGE_PREFIX_RE = re.compile(r"^\s*(\d+):")


class OpenAIEmbeddingModel:
    """Small sync adapter matching SentenceTransformer.encode for LinearRAG."""

    def __init__(self, *, base_url: str, model: str, api_key: str, default_batch_size: int = 32):
        from openai import OpenAI

        normalized_base_url = str(base_url).rstrip("/")
        if normalized_base_url.endswith("/embeddings"):
            normalized_base_url = normalized_base_url[: -len("/embeddings")]
        self.client = OpenAI(base_url=normalized_base_url, api_key=api_key)
        self.model = str(model)
        self.default_batch_size = int(default_batch_size)

    def encode(
        self,
        sentences: Any,
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int | None = None,
        **_: Any,
    ) -> np.ndarray:
        is_single = isinstance(sentences, str)
        texts = [str(sentences)] if is_single else [str(item) for item in list(sentences or [])]
        if not texts:
            return np.empty((0,), dtype=np.float32)

        batch = max(int(batch_size or self.default_batch_size), 1)
        vectors: list[list[float]] = []
        iterator = range(0, len(texts), batch)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="NV-Embed", unit="batch")
        for start in iterator:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts[start : start + batch],
                encoding_format="float",
            )
            vectors.extend(item.embedding for item in response.data)

        arr = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings and arr.size:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
        return arr[0] if is_single else arr


def load_embedding_model(args: argparse.Namespace) -> Any:
    if args.embedding_backend == "openai":
        return OpenAIEmbeddingModel(
            base_url=str(args.embedding_base_url),
            model=str(args.embedding_model),
            api_key=str(args.embedding_api_key),
            default_batch_size=int(args.embedding_batch_size),
        )
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(args.embedding_model), device=str(args.device))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def passage_from_corpus_row(row: Mapping[str, Any]) -> str:
    return f"{row.get('title', '')}\n{row.get('text', '')}".strip()


def indexed_passage(idx: int, row: Mapping[str, Any]) -> str:
    return f"{idx}:{passage_from_corpus_row(row)}"


def load_corpus(data_root: Path, dataset: str, *, max_index_docs: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    corpus = list(read_json(data_root / f"{dataset}_corpus.json"))
    if int(max_index_docs) > 0:
        corpus = corpus[: int(max_index_docs)]
    return corpus, [indexed_passage(idx, row) for idx, row in enumerate(corpus)]


def title_to_indices(corpus: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for idx, row in enumerate(corpus):
        title = str(row.get("title") or "").strip()
        if title:
            result.setdefault(title, []).append(int(idx))
    return result


def passage_to_index(corpus: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for idx, row in enumerate(corpus):
        passage = passage_from_corpus_row(row)
        if passage:
            result.setdefault(normalize_text(passage), int(idx))
            result.setdefault(normalize_text(f"{idx}:{passage}"), int(idx))
    return result


def unique_ints(values: Iterable[Any]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item < 0 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def context_passage(item: Sequence[Any]) -> str:
    title = str(item[0]) if item else ""
    sentences = item[1] if len(item) > 1 else []
    if isinstance(sentences, list):
        body = " ".join(str(s) for s in sentences)
    else:
        body = str(sentences or "")
    return f"{title}\n{body}".strip()


def gold_from_supporting_facts(
    row: Mapping[str, Any],
    *,
    title_map: Mapping[str, Sequence[int]],
    passage_map: Mapping[str, int],
) -> list[int]:
    context = list(row.get("context", []) or [])
    local_by_title: dict[str, list[int]] = {}
    for local_idx, item in enumerate(context):
        if isinstance(item, Sequence) and item:
            local_by_title.setdefault(str(item[0]), []).append(int(local_idx))
    values: list[int] = []
    for fact in row.get("supporting_facts", []) or []:
        if not isinstance(fact, Sequence) or not fact:
            continue
        title = str(fact[0])
        for local_idx in local_by_title.get(title, []):
            doc_idx = passage_map.get(normalize_text(context_passage(context[local_idx])))
            if doc_idx is not None:
                values.append(int(doc_idx))
        if not local_by_title.get(title):
            matches = title_map.get(title, [])
            if len(matches) == 1:
                values.append(int(matches[0]))
    return unique_ints(values)


def gold_from_musique(
    row: Mapping[str, Any],
    *,
    title_map: Mapping[str, Sequence[int]],
    passage_map: Mapping[str, int],
) -> list[int]:
    values: list[int] = []
    for paragraph in row.get("paragraphs", []) or []:
        if not isinstance(paragraph, Mapping) or not paragraph.get("is_supporting"):
            continue
        passage = f"{paragraph.get('title', '')}\n{paragraph.get('paragraph_text', '')}".strip()
        doc_idx = passage_map.get(normalize_text(passage))
        if doc_idx is not None:
            values.append(int(doc_idx))
            continue
        matches = title_map.get(str(paragraph.get("title") or ""), [])
        if len(matches) == 1:
            values.append(int(matches[0]))
    return unique_ints(values)


def build_query_rows(data_root: Path, dataset: str, corpus: Sequence[Mapping[str, Any]], *, limit: int = 0) -> list[dict[str, Any]]:
    rows = list(read_json(data_root / f"{dataset}.json"))
    if int(limit) > 0:
        rows = rows[: int(limit)]
    title_map = title_to_indices(corpus)
    passage_map = passage_to_index(corpus)
    out: list[dict[str, Any]] = []
    for query_index, row in enumerate(rows):
        answers = [str(row.get("answer") or "")]
        if dataset == "musique":
            answers.extend(str(alias) for alias in row.get("answer_aliases", []) or [])
            gold = gold_from_musique(row, title_map=title_map, passage_map=passage_map)
        elif dataset in {"hotpotqa", "2wikimultihopqa"}:
            gold = gold_from_supporting_facts(row, title_map=title_map, passage_map=passage_map)
        else:
            raise ValueError(f"unsupported dataset: {dataset}")
        out.append(
            {
                "query_index": int(query_index),
                "question": str(row.get("question") or ""),
                "answer": answers[0] if answers else "",
                "gold_answers": [x for x in dict.fromkeys(answers) if x],
                "gold_doc_indices": gold,
            }
        )
    return out


def doc_id_from_passage(text: Any, passage_map: Mapping[str, int]) -> int | None:
    raw = str(text or "")
    match = PASSAGE_PREFIX_RE.match(raw)
    if match:
        return int(match.group(1))
    return passage_map.get(normalize_text(raw))


def docs_for_indices(corpus: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> list[str]:
    docs: list[str] = []
    for idx in indices:
        if 0 <= int(idx) < len(corpus):
            docs.append(passage_from_corpus_row(corpus[int(idx)]))
    return docs


def recall_at_k(gold: Sequence[int], retrieved: Sequence[int], k: int) -> float:
    gold_set = {int(x) for x in gold}
    if not gold_set:
        return 0.0
    return len(gold_set & set(int(x) for x in retrieved[:k])) / float(len(gold_set))


def summarize_retrieval(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denom = max(len(examples), 1)
    metrics: dict[str, Any] = {"count": len(examples)}
    for k in (5, 20, 100, 200):
        metrics[f"Recall@{k}"] = sum(
            recall_at_k(ex.get("gold_doc_indices", []) or [], ex.get("retrieved_doc_ids", []) or [], k)
            for ex in examples
        ) / denom
    metrics["All@5"] = sum(
        1.0
        for ex in examples
        if set(int(x) for x in ex.get("gold_doc_indices", []) or [])
        and set(int(x) for x in ex.get("gold_doc_indices", []) or []).issubset(
            set(int(x) for x in (ex.get("retrieved_doc_ids", []) or [])[:5])
        )
    ) / denom
    metrics["mean_docs"] = sum(len(ex.get("docs", []) or []) for ex in examples) / denom
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.config import LinearRAGConfig
    from src.LinearRAG import LinearRAG

    dataset = str(args.dataset).lower()
    data_root = Path(args.data_root).expanduser()
    run_root = Path(args.output_root).expanduser().resolve()
    working_dir = run_root / "work"

    corpus, passages = load_corpus(data_root, dataset, max_index_docs=int(args.max_index_docs))
    query_rows = build_query_rows(data_root, dataset, corpus, limit=int(args.limit))
    passage_map = passage_to_index(corpus)

    started = time.time()
    embedding_model = load_embedding_model(args)
    config = LinearRAGConfig(
        dataset_name=dataset,
        embedding_model=embedding_model,
        spacy_model=str(args.spacy_model),
        working_dir=str(working_dir),
        max_workers=int(args.max_workers),
        llm_model=None,
        max_iterations=int(args.max_iterations),
        iteration_threshold=float(args.iteration_threshold),
        passage_ratio=float(args.passage_ratio),
        top_k_sentence=int(args.top_k_sentence),
        use_vectorized_retrieval=bool(args.use_vectorized_retrieval),
    )
    rag = LinearRAG(global_config=config)
    if not args.skip_index:
        rag.index(passages)

    retrieve_started = time.time()
    retrieval_results = rag.retrieve(query_rows)
    examples: list[dict[str, Any]] = []
    for row, result in tqdm(list(zip(query_rows, retrieval_results)), desc=f"LinearRAG map {dataset}", unit="q"):
        retrieved_ids = unique_ints(doc_id_from_passage(p, passage_map) for p in result.get("sorted_passage", []) or [])
        examples.append(
            {
                "query_index": int(row["query_index"]),
                "question": row["question"],
                "gold_answers": row["gold_answers"],
                "gold_doc_indices": row["gold_doc_indices"],
                "retrieved_doc_ids": retrieved_ids[: int(args.qa_top_k)],
                "retrieved_scores": list(result.get("sorted_passage_scores", []) or [])[: int(args.qa_top_k)],
                "docs": docs_for_indices(corpus, retrieved_ids[: int(args.qa_top_k)]),
            }
        )

    retrieval_metrics = summarize_retrieval(examples)
    payload = {
        "format": "linearrag_saved_docs_reader_input_v1",
        "dataset": dataset,
        "display_dataset": DISPLAY_NAMES.get(dataset, dataset),
        "method": "LinearRAG",
        "limit": int(args.limit),
        "max_index_docs": int(args.max_index_docs),
        "config": {
            "linear_rag_root": str(LINEARRAG_ROOT),
            "embedding_backend": str(args.embedding_backend),
            "embedding_model": str(args.embedding_model),
            "embedding_base_url": str(args.embedding_base_url) if args.embedding_backend == "openai" else None,
            "spacy_model": str(args.spacy_model),
            "qa_top_k": int(args.qa_top_k),
            "max_iterations": int(args.max_iterations),
            "iteration_threshold": float(args.iteration_threshold),
            "passage_ratio": float(args.passage_ratio),
            "top_k_sentence": int(args.top_k_sentence),
            "use_vectorized_retrieval": bool(args.use_vectorized_retrieval),
        },
        "retrieval": {
            "recomputed_title_recall": retrieval_metrics,
            "mapping": "LinearRAG indexed-passage prefix mapped back to local corpus ids",
        },
        "usage": {
            "offline_generative_tokens": 0,
            "online_auxiliary_generative_tokens": 0,
            "shared_reader_tokens": "excluded; measured by run_docs_reader_qa",
            "elapsed_s_total": round(time.time() - started, 4),
            "elapsed_s_retrieve": round(time.time() - retrieve_started, 4),
        },
        "examples": examples,
    }
    output_json = run_root / "reader_inputs" / f"{dataset}_linearrag_top{args.qa_top_k}_limit{args.limit}_reader_input.json"
    summary_json = run_root / "summaries" / f"{dataset}_linearrag_summary.json"
    write_json(output_json, payload)
    write_json(summary_json, {k: payload[k] for k in ("dataset", "method", "limit", "config", "retrieval", "usage")})
    print(json.dumps({"dataset": dataset, "output_json": str(output_json), "summary_json": str(summary_json), "retrieval": retrieval_metrics}, sort_keys=True))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DISPLAY_NAMES))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "reproduce/dataset"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "run_logs/linearrag_aligned"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-index-docs", type=int, default=0)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--embedding-backend", choices=["openai", "sentence_transformers"], default="openai")
    parser.add_argument("--embedding-model", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1")
    parser.add_argument("--embedding-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--iteration-threshold", type=float, default=0.4)
    parser.add_argument("--passage-ratio", type=float, default=0.05)
    parser.add_argument("--top-k-sentence", type=int, default=1)
    parser.add_argument("--use-vectorized-retrieval", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
