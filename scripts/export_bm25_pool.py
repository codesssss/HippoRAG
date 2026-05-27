#!/usr/bin/env python3
"""Export BM25 top-N retrieval pools in the external-pool JSON format."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DATASET_FILE_ALIASES = {
    "nq": "nq_rear",
    "natural_questions": "nq_rear",
}
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def resolve_dataset_file_stem(dataset_name: str | None) -> str:
    normalized = str(dataset_name or "").strip().lower()
    return DATASET_FILE_ALIASES.get(normalized, str(dataset_name or "").strip())


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def extract_title(doc: str) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def parse_answer_alias_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                return parse_answer_alias_values(ast.literal_eval(cleaned))
            except (ValueError, SyntaxError):
                return [cleaned]
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
            paragraphs = [
                item for item in sample.get("paragraphs", []) if item.get("is_supporting") is not False
            ]
            docs = [
                item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"])
                for item in paragraphs
            ]
        gold_docs.append(sorted(set(docs)))
    return gold_docs


def get_gold_answers(samples: Sequence[Mapping[str, Any]]) -> List[List[str]]:
    gold_answers: List[List[str]] = []
    for sample in samples:
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


def load_corpus_docs(corpus_path: Path) -> list[str]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    docs: list[str] = []
    for row in corpus:
        title = str(row.get("title") or "")
        text = str(row.get("text") or row.get("paragraph_text") or "")
        docs.append(f"{title}\n{text}".strip())
    return docs


def build_bm25_index(docs: Sequence[str]) -> tuple[dict[str, list[tuple[int, int]]], dict[str, float], list[int], float]:
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_lens: list[int] = []
    for doc_idx, doc in enumerate(tqdm(docs, desc="Index BM25 docs")):
        counts = Counter(tokenize(doc))
        doc_lens.append(sum(counts.values()))
        for token, tf in counts.items():
            postings[token].append((int(doc_idx), int(tf)))
    n_docs = max(len(docs), 1)
    avgdl = float(sum(doc_lens)) / float(n_docs)
    idf = {
        token: math.log(1.0 + (n_docs - len(rows) + 0.5) / (len(rows) + 0.5))
        for token, rows in postings.items()
    }
    return dict(postings), idf, doc_lens, avgdl


def rank_bm25(
    query: str,
    *,
    postings: Mapping[str, Sequence[tuple[int, int]]],
    idf: Mapping[str, float],
    doc_lens: Sequence[int],
    avgdl: float,
    pool_k: int,
    k1: float,
    b: float,
) -> tuple[list[int], list[float]]:
    scores: dict[int, float] = defaultdict(float)
    q_counts = Counter(tokenize(query))
    avgdl = avgdl if avgdl > 0 else 1.0
    for token, qtf in q_counts.items():
        token_idf = float(idf.get(token, 0.0))
        if token_idf <= 0:
            continue
        for doc_idx, tf in postings.get(token, ()):
            dl = float(doc_lens[int(doc_idx)] if int(doc_idx) < len(doc_lens) else avgdl)
            denom = float(tf) + k1 * (1.0 - b + b * dl / avgdl)
            scores[int(doc_idx)] += float(qtf) * token_idf * (float(tf) * (k1 + 1.0)) / max(denom, 1e-12)
    ranked = [doc_idx for doc_idx, _ in sorted(scores.items(), key=lambda item: (-float(item[1]), int(item[0])))]
    if len(ranked) < pool_k:
        seen = set(ranked)
        ranked.extend(doc_idx for doc_idx in range(len(doc_lens)) if doc_idx not in seen)
    top = ranked[: max(int(pool_k), 0)]
    return top, [float(scores.get(doc_idx, 0.0)) for doc_idx in top]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pool_k", type=int, default=200)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    dataset_file_stem = resolve_dataset_file_stem(args.dataset)
    data_root = Path(args.data_root)
    samples = json.loads((data_root / f"{dataset_file_stem}.json").read_text(encoding="utf-8"))
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]
    corpus_docs = load_corpus_docs(data_root / f"{dataset_file_stem}_corpus.json")
    queries = [str(sample.get("question") or "") for sample in samples]
    gold_docs = get_gold_docs(samples, dataset_file_stem)
    gold_answers = get_gold_answers(samples)

    postings, idf, doc_lens, avgdl = build_bm25_index(corpus_docs)
    records: list[dict[str, Any]] = []
    retrieved_doc_lists: list[list[str]] = []
    for query_idx, query in enumerate(tqdm(queries, desc="Rank BM25 queries")):
        doc_ids, scores = rank_bm25(
            query,
            postings=postings,
            idf=idf,
            doc_lens=doc_lens,
            avgdl=avgdl,
            pool_k=int(args.pool_k),
            k1=float(args.k1),
            b=float(args.b),
        )
        pool_docs = [corpus_docs[int(doc_id)] for doc_id in doc_ids]
        retrieved_doc_lists.append(pool_docs)
        records.append(
            {
                "query_idx": int(query_idx),
                "question": query,
                "gold_answers": list(gold_answers[query_idx]),
                "gold_docs": list(gold_docs[query_idx]),
                "gold_titles": [extract_title(doc) for doc in gold_docs[query_idx]],
                "pool_k": int(args.pool_k),
                "pool_docs": pool_docs,
                "pool_titles": [extract_title(doc) for doc in pool_docs],
                "pool_doc_scores": [float(score) for score in scores],
                "pool_doc_ids": [int(doc_id) for doc_id in doc_ids],
            }
        )

    recall = compute_title_recall(gold_docs, retrieved_doc_lists, [5, 20, 100, 200])
    output = {
        "dataset": str(dataset_file_stem),
        "limit": int(len(samples)),
        "pool_k": int(args.pool_k),
        "source": "bm25_retrieval_export",
        "data_root": str(data_root),
        "k1": float(args.k1),
        "b": float(args.b),
        "retrieval": {
            "recomputed_title_recall": recall,
            "bm25_metrics": recall,
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
        errors="replace",
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "dataset": str(dataset_file_stem),
                "limit": len(samples),
                "pool_k": int(args.pool_k),
                "retrieval": recall,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
