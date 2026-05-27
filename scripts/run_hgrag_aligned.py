#!/usr/bin/env python3
"""Run HGRAG under the local limit-N comparison protocol.

This wrapper preserves HGRAG's entity-hypergraph retrieval shape, but adapts the
two heavy model calls to local OpenAI-compatible endpoints:

- Qwen3 chat API for query/corpus NER and QA, with /no_think injected.
- NV-Embed API for dense entity/entity and query/document retrieval.

The output JSON follows the compact metric layout used by the PropRAG /
NEOCORRAG comparison reports.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx
import numpy as np
import pandas as pd
import requests
import torch
from openai import OpenAI
from tqdm import tqdm


NO_THINK = "/no_think"

DATASET_PATHS = {
    "2wikimultihopqa": ("2wiki/raw/2wikimultihopqa.json", "2wiki/raw/2wikimultihopqa_corpus.json"),
    "hotpotqa": ("hotpot/raw/hotpotqa.json", "hotpot/raw/hotpotqa_corpus.json"),
    "musique": ("musique/raw/musique.json", "musique/raw/musique_corpus.json"),
}

ENTITY_TEXT_KEYS = ("name", "entity", "text", "title", "mention", "value", "label")


def _add_hgrag_to_path(root: Path) -> None:
    sys.path.insert(0, str(root))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _strip_think(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", str(text or ""))
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json(text: str) -> Any:
    cleaned = _strip_markdown_fence(_strip_think(text))
    for candidate in (cleaned,):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    decoder = json.JSONDecoder()
    starts = [idx for idx, char in enumerate(cleaned) if char in "[{"]
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj
        except Exception:
            continue
    return None


def _coerce_entity_text(entity: Any) -> list[str]:
    if isinstance(entity, str):
        text = entity.strip()
        return [text] if text else []
    if isinstance(entity, dict):
        for key in ENTITY_TEXT_KEYS:
            value = entity.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        for value in entity.values():
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            if isinstance(value, (list, tuple)):
                nested = _normalize_entities(value)
                if nested:
                    return [nested[0]]
        return []
    if isinstance(entity, (list, tuple)):
        return _normalize_entities(entity)
    return []


def _normalize_entities(entities: Any) -> list[str]:
    if entities is None:
        return []
    if isinstance(entities, dict):
        if "entities" in entities:
            entities = entities["entities"]
        elif "named_entities" in entities:
            entities = entities["named_entities"]
        else:
            entities = [entities]
    if isinstance(entities, str):
        entities = [entities]
    normalized: list[str] = []
    seen = set()
    for entity in entities:
        for text in _coerce_entity_text(entity):
            if text not in seen:
                normalized.append(text)
                seen.add(text)
    return normalized


def parse_entities(raw_text: str, field: str) -> list[str]:
    obj = _extract_first_json(raw_text)
    if isinstance(obj, dict):
        if field in obj:
            return _normalize_entities(obj[field])
        alternate = "named_entities" if field == "entities" else "entities"
        if alternate in obj:
            return _normalize_entities(obj[alternate])
    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) for item in obj):
            merged = []
            for item in obj:
                if field in item:
                    merged.extend(_normalize_entities(item[field]))
            if merged:
                return _normalize_entities(merged)
        return _normalize_entities(obj)
    return []


def parse_answer(raw_text: str) -> str:
    cleaned = _strip_think(raw_text)
    match = re.search(r"Answer:\s*(.+)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    answer = match.group(1).strip() if match else cleaned.strip()
    answer = answer.splitlines()[0].strip() if answer else ""
    if answer.endswith("."):
        answer = answer[:-1].strip()
    return answer


def normalize_answer(answer: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        import string

        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(str(answer).lower())))


def parse_answer_alias_values(value: Any) -> list[str]:
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
        aliases: list[str] = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]


def answer_em(gold_answers: list[list[str]], predicted_answers: list[str]) -> tuple[float, list[float]]:
    scores = []
    for gold_list, predicted in zip(gold_answers, predicted_answers):
        pred = normalize_answer(predicted)
        scores.append(float(any(normalize_answer(gold) == pred for gold in gold_list)))
    return float(np.mean(scores) if scores else 0.0), scores


def answer_f1(gold_answers: list[list[str]], predicted_answers: list[str]) -> tuple[float, list[float]]:
    from collections import Counter

    def f1_one(gold: str, predicted: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        pred_tokens = normalize_answer(predicted).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            return 0.0
        precision = same / max(len(pred_tokens), 1)
        recall = same / max(len(gold_tokens), 1)
        return 2 * precision * recall / (precision + recall)

    scores = []
    for gold_list, predicted in zip(gold_answers, predicted_answers):
        scores.append(float(max((f1_one(gold, predicted) for gold in gold_list), default=0.0)))
    return float(np.mean(scores) if scores else 0.0), scores


def retrieval_recall(gold_docs: list[list[int]], retrieved_docs: list[list[int]], k_values: list[int]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pooled = {f"Recall@{k}": 0.0 for k in k_values}
    records = []
    for qid, (gold, docs) in enumerate(zip(gold_docs, retrieved_docs)):
        gold_set = set(int(x) for x in gold)
        row = {"qid": qid}
        for k in k_values:
            if gold_set:
                value = len(set(int(x) for x in docs[:k]) & gold_set) / len(gold_set)
            else:
                value = 0.0
            row[f"Recall@{k}"] = value
            pooled[f"Recall@{k}"] += value
        records.append(row)
    denom = max(len(records), 1)
    pooled = {key: round(value / denom, 4) for key, value in pooled.items()}
    return pooled, records


def resolve_dataset_paths(dataset: str, hgrag_root: Path, hipporag_root: Path) -> tuple[Path, Path]:
    hippo_data = hipporag_root / "reproduce" / "dataset" / f"{dataset}.json"
    hippo_corpus = hipporag_root / "reproduce" / "dataset" / f"{dataset}_corpus.json"
    if hippo_data.exists() and hippo_corpus.exists():
        return hippo_data, hippo_corpus
    if dataset not in DATASET_PATHS:
        raise KeyError(f"Unknown HGRAG dataset mapping: {dataset}")
    data_rel, corpus_rel = DATASET_PATHS[dataset]
    return hgrag_root / "data" / data_rel, hgrag_root / "data" / corpus_rel


def build_processed_data(dataset: str, raw_samples: list[dict[str, Any]], raw_corpus: list[dict[str, Any]], limit: int | None, corpus_limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[int]], list[list[str]]]:
    samples = [dict(item) for item in raw_samples[:limit]]
    corpus = [dict(item) for item in raw_corpus[:corpus_limit]]
    for did, doc in enumerate(corpus):
        doc["did"] = did
    title_to_did = {doc.get("title"): int(doc["did"]) for doc in corpus}
    text_to_did = {doc.get("text"): int(doc["did"]) for doc in corpus}

    gold_doc_ids: list[list[int]] = []
    gold_answers: list[list[str]] = []

    for qid, sample in enumerate(samples):
        sample["qid"] = qid
        if "supporting_facts" in sample:
            ids = []
            new_supporting = []
            for fact in sample.get("supporting_facts", []):
                fact_list = list(fact)
                did = title_to_did.get(fact_list[0])
                if did is not None:
                    ids.append(did)
                    if len(fact_list) < 3:
                        fact_list.append(did)
                new_supporting.append(fact_list)
            sample["supporting_facts"] = new_supporting
            gold_doc_ids.append(sorted(set(ids)))
        elif "contexts" in sample:
            ids = []
            for context in sample.get("contexts", []):
                if context.get("is_supporting"):
                    did = title_to_did.get(context.get("title"))
                    if did is not None:
                        ids.append(did)
            gold_doc_ids.append(sorted(set(ids)))
        else:
            ids = []
            for paragraph in sample.get("paragraphs", []):
                text = paragraph.get("text", paragraph.get("paragraph_text"))
                did = text_to_did.get(text)
                paragraph["did"] = did
                if paragraph.get("is_supporting") is not False and did is not None:
                    ids.append(did)
            gold_doc_ids.append(sorted(set(ids)))

        answers: list[str] = []
        if "answer" in sample or "gold_ans" in sample:
            answers.extend(parse_answer_alias_values(sample.get("answer", sample.get("gold_ans"))))
        elif "reference" in sample:
            answers.extend(parse_answer_alias_values(sample["reference"]))
        elif "obj" in sample:
            answers.extend(parse_answer_alias_values(sample.get("obj")))
            answers.extend(parse_answer_alias_values(sample.get("possible_answers")))
            answers.extend(parse_answer_alias_values(sample.get("o_wiki_title")))
            answers.extend(parse_answer_alias_values(sample.get("o_aliases")))
        if "answer_aliases" in sample:
            answers.extend(parse_answer_alias_values(sample["answer_aliases"]))
        gold_answers.append(sorted({str(item).strip() for item in answers if str(item).strip()}))

    return samples, corpus, gold_doc_ids, gold_answers


class LocalEndpoints:
    def __init__(self, *, llm_name: str, llm_base_url: str, embedding_name: str, embedding_url: str, embedding_batch_size: int, timeout: float) -> None:
        self.llm_name = llm_name
        self.embedding_name = embedding_name
        self.embedding_url = embedding_url
        self.embedding_batch_size = embedding_batch_size
        self.session = requests.Session()
        self.session.trust_env = False
        self.timeout = timeout
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _strip_think(response.choices[0].message.content or "")

    def embed(self, texts: list[str], *, instruction: str | None = None) -> np.ndarray:
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
        outputs = []
        for start in tqdm(range(0, len(texts), self.embedding_batch_size), desc="Embedding API"):
            batch = [text if text else " " for text in texts[start:start + self.embedding_batch_size]]
            payload = {"model": self.embedding_name, "input": batch}
            response = self.session.post(self.embedding_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()["data"]
            outputs.append(np.asarray([item["embedding"] for item in data], dtype=np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)


def qner_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a very effective keywords extraction system. Extract all named entities "
                'important for solving the question. Return JSON only: {"entities": [..]}. '
                "Do not output hidden reasoning or <think> tags."
            ),
        },
        {"role": "user", "content": "What city is the Eiffel Tower located in?"},
        {"role": "assistant", "content": '{"entities": ["Eiffel Tower"]}'},
        {"role": "user", "content": f"{NO_THINK}\n{question}"},
    ]


def cner_messages(title: str, text: str) -> list[dict[str, str]]:
    passage = f"{title}\n{text}"
    return [
        {
            "role": "system",
            "content": (
                "Your task is to extract named entities from the given paragraph. "
                'Return JSON only: {"named_entities": [..]}. Do not output hidden reasoning or <think> tags.'
            ),
        },
        {
            "role": "user",
            "content": (
                "Radio City\nRadio City is India's first private FM radio station and was started on 3 July 2001."
            ),
        },
        {
            "role": "assistant",
            "content": '{"named_entities": ["Radio City", "India", "3 July 2001"]}',
        },
        {"role": "user", "content": f"{NO_THINK}\n{passage}"},
    ]


def qa_messages(question: str, docs: list[str]) -> list[dict[str, str]]:
    context = ""
    for doc in docs:
        context += f"Wikipedia Title: {doc}\n"
    user = (
        f"{NO_THINK}\n"
        f"{context}\nQuestion: {question}\nThought: "
        'Keep reasoning brief and end with exactly "Answer: <short answer>".'
    )
    return [
        {
            "role": "system",
            "content": (
                "As an advanced reading comprehension assistant, answer using only the passages. "
                "Do not output hidden reasoning or <think> tags. Conclude with Answer: followed by a concise answer."
            ),
        },
        {"role": "user", "content": user},
    ]


def run_parallel(items: list[Any], worker: Any, max_workers: int, desc: str) -> list[Any]:
    results = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, idx, item): idx for idx, item in enumerate(items)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = exc
    return results


def valid_jsonl(path: Path, expected_rows: int) -> bool:
    try:
        return path.exists() and sum(1 for line in path.open("r", encoding="utf-8") if line.strip()) == expected_rows
    except Exception:
        return False


def run_ner(samples: list[dict[str, Any]], corpus: list[dict[str, Any]], endpoints: LocalEndpoints, out_dir: Path, *, max_workers: int, max_new_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q_path = out_dir / "ner" / "q_ner_resp.jsonl"
    c_path = out_dir / "ner" / "c_ner_resp.jsonl"

    if valid_jsonl(q_path, len(samples)):
        q_rows = _read_jsonl(q_path)
    else:
        def q_worker(_: int, sample: dict[str, Any]) -> dict[str, Any]:
            raw = endpoints.chat(qner_messages(sample["question"]), max_tokens=max_new_tokens)
            return {
                "idd": int(sample["qid"]),
                "extracted_data": [{"entities": parse_entities(raw, "entities")}],
                "rawout": raw,
            }

        q_rows = run_parallel(samples, q_worker, max_workers, "HGRAG QNER")
        q_rows = [row if isinstance(row, dict) else {"idd": samples[idx]["qid"], "extracted_data": [{"entities": []}], "error": repr(row)} for idx, row in enumerate(q_rows)]
        _write_jsonl(q_path, q_rows)

    if valid_jsonl(c_path, len(corpus)):
        c_rows = _read_jsonl(c_path)
    else:
        def c_worker(_: int, doc: dict[str, Any]) -> dict[str, Any]:
            raw = endpoints.chat(cner_messages(doc.get("title", ""), doc.get("text", "")), max_tokens=max_new_tokens)
            return {
                "idd": int(doc["did"]),
                "extracted_data": [{"named_entities": parse_entities(raw, "named_entities")}],
                "rawout": raw,
            }

        c_rows = run_parallel(corpus, c_worker, max_workers, "HGRAG CNER")
        c_rows = [row if isinstance(row, dict) else {"idd": corpus[idx]["did"], "extracted_data": [{"named_entities": []}], "error": repr(row)} for idx, row in enumerate(c_rows)]
        _write_jsonl(c_path, c_rows)

    return q_rows, c_rows


def write_entity_files(q_rows: list[dict[str, Any]], c_rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path, Path, Path, list[dict[str, Any]]]:
    ret_dir = out_dir / "ret"
    hg_dir = out_dir / "hg"
    ret_dir.mkdir(parents=True, exist_ok=True)
    hg_dir.mkdir(parents=True, exist_ok=True)

    query_entities_by_qid = []
    query_entity_set = set()
    for row in q_rows:
        entities = []
        for payload in row.get("extracted_data", []):
            entities.extend(_normalize_entities(payload.get("entities", [])))
        entities = _normalize_entities(entities)
        query_entities_by_qid.append({"qid": int(row["idd"]), "entity_texts": entities})
        query_entity_set.update(entities)
    query_entities = sorted(query_entity_set)
    q_ent_to_id = {entity: idx for idx, entity in enumerate(query_entities)}

    qe_rows = []
    for row in query_entities_by_qid:
        qe_rows.append({"qid": row["qid"], "entities": [q_ent_to_id[e] for e in row["entity_texts"] if e in q_ent_to_id]})

    q_ent_id_path = ret_dir / "q_ent_id.tsv"
    with q_ent_id_path.open("w", encoding="utf-8") as handle:
        for idx, entity in enumerate(query_entities):
            handle.write(f'{idx}\t"{entity.replace(chr(34), chr(34) + chr(34))}"\n')

    qe_path = ret_dir / "qe.json"
    _write_json(qe_path, qe_rows)

    ent_did_rows = []
    for row in c_rows:
        did = int(row["idd"])
        entities = []
        for payload in row.get("extracted_data", []):
            entities.extend(_normalize_entities(payload.get("named_entities", [])))
        for entity in _normalize_entities(entities):
            ent_did_rows.append({"ent": entity, "did": did})

    ent_did_path = hg_dir / "c_ent_did.csv"
    ent_did_path.parent.mkdir(parents=True, exist_ok=True)
    with ent_did_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ent", "did"])
        writer.writeheader()
        writer.writerows(ent_did_rows)

    df = pd.DataFrame(ent_did_rows, columns=["ent", "did"])
    if df.empty:
        corpus_entities = []
    else:
        _, unique_entities = pd.factorize(df["ent"])
        corpus_entities = list(unique_entities)
    c_ent_id_path = ret_dir / "c_ent_id.tsv"
    with c_ent_id_path.open("w", encoding="utf-8") as handle:
        for idx, entity in enumerate(corpus_entities):
            handle.write(f'{idx}\t"{str(entity).replace(chr(34), chr(34) + chr(34))}"\n')

    return q_ent_id_path, c_ent_id_path, qe_path, ent_did_path, qe_rows


def write_query_doc_tsv(samples: list[dict[str, Any]], corpus: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    ret_dir = out_dir / "ret"
    query_path = ret_dir / "query.tsv"
    doc_path = ret_dir / "doc.tsv"
    ret_dir.mkdir(parents=True, exist_ok=True)
    with query_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            text = sample["question"].replace('"', '""')
            handle.write(f'{int(sample["qid"])}\t"{text}"\n')
    with doc_path.open("w", encoding="utf-8") as handle:
        for doc in corpus:
            text = f'{doc.get("title", "")}\n{doc.get("text", "")}'.replace('"', '""')
            handle.write(f'{int(doc["did"])}\t"{text}"\n')
    return query_path, doc_path


def read_tsv_texts(path: Path) -> tuple[list[int], list[str]]:
    ids = []
    texts = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                ids.append(int(row[0]))
                texts.append(row[1])
    return ids, texts


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def load_or_embed(path: Path, texts: list[str], endpoints: LocalEndpoints, *, instruction: str | None) -> np.ndarray:
    if path.exists():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return np.asarray(payload["vecs"], dtype=np.float32)
    vecs = endpoints.embed(texts, instruction=instruction)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"vecs": vecs}, handle)
    return vecs


def dense_retrieval(query_vecs: np.ndarray, corpus_vecs: np.ndarray, query_ids: list[int], corpus_ids: list[int], *, top_k: int, chunk_size: int = 64) -> dict[str, list[list[Any]]]:
    if len(query_vecs) == 0 or len(corpus_vecs) == 0:
        return {str(qid): [[], []] for qid in query_ids}
    qv = l2_normalize(query_vecs.astype(np.float32))
    cv = l2_normalize(corpus_vecs.astype(np.float32))
    top_k = min(top_k, len(corpus_ids))
    results = {}
    corpus_ids_np = np.asarray(corpus_ids)
    for start in tqdm(range(0, len(qv), chunk_size), desc="Dense retrieval"):
        sims = qv[start:start + chunk_size] @ cv.T
        if top_k >= sims.shape[1]:
            idx = np.argsort(-sims, axis=1)
        else:
            part = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
            row_scores = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-row_scores, axis=1)
            idx = np.take_along_axis(part, order, axis=1)
        scores = np.take_along_axis(sims, idx, axis=1)
        for offset, (doc_idx, doc_scores) in enumerate(zip(idx, scores)):
            qid = query_ids[start + offset]
            results[str(qid)] = [corpus_ids_np[doc_idx].astype(int).tolist(), doc_scores.astype(float).tolist()]
    return results


def run_dense_retrievals(out_dir: Path, endpoints: LocalEndpoints, q_ent_path: Path, c_ent_path: Path, query_path: Path, doc_path: Path, *, e2e_top_k: int) -> tuple[Path, Path]:
    ret_dir = out_dir / "ret"
    vec_dir = out_dir / "vecs"
    e2e_path = ret_dir / "e2e_ret.json"
    q2d_path = ret_dir / "q2d_ret.json"

    if not e2e_path.exists():
        q_ent_ids, q_ent_texts = read_tsv_texts(q_ent_path)
        c_ent_ids, c_ent_texts = read_tsv_texts(c_ent_path)
        q_ent_vecs = load_or_embed(vec_dir / "q_ent_vecs.pkl", q_ent_texts, endpoints, instruction=None)
        c_ent_vecs = load_or_embed(vec_dir / "c_ent_vecs.pkl", c_ent_texts, endpoints, instruction=None)
        e2e = dense_retrieval(q_ent_vecs, c_ent_vecs, q_ent_ids, c_ent_ids, top_k=e2e_top_k)
        _write_json(e2e_path, e2e)

    if not q2d_path.exists():
        q_ids, q_texts = read_tsv_texts(query_path)
        d_ids, d_texts = read_tsv_texts(doc_path)
        q_vecs = load_or_embed(
            vec_dir / "query_vecs.pkl",
            q_texts,
            endpoints,
            instruction="Given a question, retrieve passages that answer the question",
        )
        d_vecs = load_or_embed(vec_dir / "doc_vecs.pkl", d_texts, endpoints, instruction=None)
        q2d = dense_retrieval(q_vecs, d_vecs, q_ids, d_ids, top_k=len(d_ids))
        _write_json(q2d_path, q2d)

    return e2e_path, q2d_path


def run_hgrag_retrieval(hgrag_root: Path, out_dir: Path, data_path: Path, corpus_path: Path, ent_did_path: Path, qe_path: Path, e2e_path: Path, q2d_path: Path, *, device: str, beta: float, step: int, ent_topk: int, qa_top_k: int, recall_top_k: int) -> tuple[Path, Path]:
    _add_hgrag_to_path(hgrag_root)
    from src.modules.hgraph import HG  # type: ignore

    grag_ret_path = out_dir / "ret" / "hgrag_ret.json"
    grag_docs_path = out_dir / "ret" / "hgrag_docs.json"
    if grag_ret_path.exists() and grag_docs_path.exists():
        return grag_ret_path, grag_docs_path

    hg = HG(str(ent_did_path), device=device)
    e2e = _load_json(e2e_path)
    q2d = _load_json(q2d_path)
    qe_rows = _load_json(qe_path)
    samples = _load_json(data_path)
    corpus = _load_json(corpus_path)
    id2doc = {int(doc["did"]): f'{doc.get("title", "")}\n{doc.get("text", "")}' for doc in corpus}
    id2query = {int(sample["qid"]): sample["question"] for sample in samples}

    ret_res: dict[str, list[list[Any]]] = {}
    no_entity_fallbacks = 0
    for row in tqdm(qe_rows, desc="HGRAG diffusion"):
        qid = int(row["qid"])
        doc_ids, doc_scores = q2d[str(qid)]
        qd_sims = {int(did): float(score) for did, score in zip(doc_ids, doc_scores)}
        entity_ids = []
        entity_scores = []
        for q_ent_id in row.get("entities", []):
            ent_docs, ent_scores = e2e.get(str(q_ent_id), [[], []])
            entity_ids.extend([int(item) for item in ent_docs[:ent_topk]])
            entity_scores.extend([float(item) for item in ent_scores[:ent_topk]])
        if not entity_ids:
            no_entity_fallbacks += 1
            sorted_docs = [int(item) for item in doc_ids[:recall_top_k]]
            sorted_scores = [float(item) for item in doc_scores[:recall_top_k]]
        else:
            sorted_docs, sorted_scores = hg.hg_diffusion(
                entity_ids,
                entity_scores,
                qd_sims,
                beta=beta,
                step=step,
                multihot=True,
            )
            sorted_docs = [int(item) for item in sorted_docs[:recall_top_k]]
            sorted_scores = [float(item) for item in sorted_scores[:recall_top_k]]
        ret_res[str(qid)] = [sorted_docs, sorted_scores]

    if qa_top_k < recall_top_k:
        ret_res = hg.struct_enhance(ret_res, qa_top_k, recall_top_k)

    docs_payload = {}
    for qid, (dids, _) in ret_res.items():
        docs_payload[str(qid)] = [id2query[int(qid)], [id2doc[int(did)] for did in dids if int(did) in id2doc]]

    _write_json(grag_ret_path, ret_res)
    _write_json(grag_docs_path, docs_payload)
    _write_json(out_dir / "ret" / "hgrag_retrieval_diagnostics.json", {"no_entity_fallbacks": no_entity_fallbacks})
    return grag_ret_path, grag_docs_path


def run_qa(grag_docs_path: Path, endpoints: LocalEndpoints, out_dir: Path, *, qa_top_k: int, max_workers: int, max_new_tokens: int) -> tuple[list[str], list[dict[str, Any]], Path]:
    pred_path = out_dir / "qa" / "qa_pred_ans.json"
    resp_path = out_dir / "qa" / "qa_resp.jsonl"
    docs_payload = _load_json(grag_docs_path)
    ordered_items = sorted(docs_payload.items(), key=lambda item: int(item[0]))
    if pred_path.exists() and valid_jsonl(resp_path, len(ordered_items)):
        return _load_json(pred_path), _read_jsonl(resp_path), pred_path

    def qa_worker(_: int, item: tuple[str, Any]) -> dict[str, Any]:
        qid, (question, docs) = item
        raw = endpoints.chat(qa_messages(question, docs[:qa_top_k]), max_tokens=max_new_tokens)
        answer = parse_answer(raw)
        return {"qid": int(qid), "extracted_data": {"Answer": answer}, "rawout": raw}

    rows = run_parallel(ordered_items, qa_worker, max_workers, "HGRAG QA")
    cleaned_rows = []
    predictions = []
    for idx, row in enumerate(rows):
        qid = int(ordered_items[idx][0])
        if isinstance(row, dict):
            cleaned_rows.append(row)
            predictions.append(row.get("extracted_data", {}).get("Answer", ""))
        else:
            cleaned_rows.append({"qid": qid, "extracted_data": {"Answer": ""}, "error": repr(row)})
            predictions.append("")
    _write_jsonl(resp_path, cleaned_rows)
    _write_json(pred_path, predictions)
    return predictions, cleaned_rows, pred_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--corpus_limit", type=int, default=None)
    parser.add_argument("--hgrag_root", type=Path, default=Path("/mnt/nvme/code/HGRAG"))
    parser.add_argument("--hipporag_root", type=Path, default=Path("/mnt/nvme/code/HippoRAG"))
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_method_name", default=None)
    parser.add_argument("--llm_name", default="qwen3-8b-train")
    parser.add_argument("--llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--embedding_name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding_batch_size", type=int, default=4)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--recall_top_k", type=int, default=20)
    parser.add_argument("--e2e_top_k", type=int, default=20)
    parser.add_argument("--ent_topk", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--hgraph_device", default="cpu")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--ner_max_new_tokens", type=int, default=512)
    parser.add_argument("--qa_max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--skip_qa",
        action="store_true",
        help="Only run HGRAG retrieval and skip the built-in QA reader.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    raw_data_path, raw_corpus_path = resolve_dataset_paths(args.dataset, args.hgrag_root, args.hipporag_root)
    raw_samples = _load_json(raw_data_path)
    raw_corpus = _load_json(raw_corpus_path)
    samples, corpus, gold_doc_ids, gold_answers = build_processed_data(
        args.dataset,
        raw_samples,
        raw_corpus,
        args.limit,
        args.corpus_limit,
    )

    tag = args.output_method_name or f"aligned_limit{args.limit}_{args.llm_name}_nothink"
    if args.corpus_limit is not None:
        tag += f"_corpus{args.corpus_limit}"
    out_dir = args.hgrag_root / "output" / tag / args.dataset
    data_dir = out_dir / "data"
    data_path = data_dir / f"{args.dataset}_id.json"
    corpus_path = data_dir / f"{args.dataset}_corpus_id.json"
    gold_docs_path = data_dir / f"{args.dataset}_gold_docs_id.json"
    gold_answers_path = data_dir / f"{args.dataset}_gold_answers.json"
    _write_json(data_path, samples)
    _write_json(corpus_path, corpus)
    _write_json(gold_docs_path, gold_doc_ids)
    _write_json(gold_answers_path, gold_answers)

    endpoints = LocalEndpoints(
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_name=args.embedding_name,
        embedding_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        timeout=args.timeout,
    )

    q_rows, c_rows = run_ner(
        samples,
        corpus,
        endpoints,
        out_dir,
        max_workers=args.max_workers,
        max_new_tokens=args.ner_max_new_tokens,
    )
    q_ent_path, c_ent_path, qe_path, ent_did_path, qe_rows = write_entity_files(q_rows, c_rows, out_dir)
    query_path, doc_path = write_query_doc_tsv(samples, corpus, out_dir)
    e2e_path, q2d_path = run_dense_retrievals(
        out_dir,
        endpoints,
        q_ent_path,
        c_ent_path,
        query_path,
        doc_path,
        e2e_top_k=args.e2e_top_k,
    )
    grag_ret_path, grag_docs_path = run_hgrag_retrieval(
        args.hgrag_root,
        out_dir,
        data_path,
        corpus_path,
        ent_did_path,
        qe_path,
        e2e_path,
        q2d_path,
        device=args.hgraph_device,
        beta=args.beta,
        step=args.step,
        ent_topk=args.ent_topk,
        qa_top_k=args.qa_top_k,
        recall_top_k=args.recall_top_k,
    )
    if args.skip_qa:
        predictions = [""] * len(samples)
        qa_rows = []
        qa_pred_path = out_dir / "qa" / "qa_pred_ans.json"
        em = None
        f1 = None
        em_records = [None] * len(samples)
        f1_records = [None] * len(samples)
    else:
        predictions, qa_rows, qa_pred_path = run_qa(
            grag_docs_path,
            endpoints,
            out_dir,
            qa_top_k=args.qa_top_k,
            max_workers=args.max_workers,
            max_new_tokens=args.qa_max_new_tokens,
        )

    ret_res = _load_json(grag_ret_path)
    retrieved_doc_ids = [[int(doc_id) for doc_id in ret_res[str(qid)][0]] for qid in range(len(samples))]
    recall_overall, recall_records = retrieval_recall(gold_doc_ids, retrieved_doc_ids, [5, 20])
    if not args.skip_qa:
        em, em_records = answer_em(gold_answers, predictions)
        f1, f1_records = answer_f1(gold_answers, predictions)

    docs_payload = _load_json(grag_docs_path)
    records = []
    for qid, sample in enumerate(samples):
        records.append(
            {
                "query_idx": qid,
                "question": sample["question"],
                "gold_doc_ids": gold_doc_ids[qid],
                "gold_answers": gold_answers[qid],
                "retrieved_doc_ids": retrieved_doc_ids[qid],
                "retrieved_docs": docs_payload.get(str(qid), ["", []])[1],
                "answer": predictions[qid] if qid < len(predictions) else "",
                "Recall@5": recall_records[qid]["Recall@5"],
                "Recall@20": recall_records[qid]["Recall@20"],
                "ExactMatch": em_records[qid],
                "F1": f1_records[qid],
                "query_entities": qe_rows[qid]["entities"] if qid < len(qe_rows) else [],
            }
        )

    output = {
        "method": "hgrag",
        "dataset": args.dataset,
        "limit": int(args.limit) if args.limit is not None else None,
        "corpus_limit": args.corpus_limit,
        "num_queries": len(samples),
        "num_corpus_docs": len(corpus),
        "raw_data_path": str(raw_data_path),
        "raw_corpus_path": str(raw_corpus_path),
        "hgrag_root": str(args.hgrag_root),
        "work_dir": str(out_dir),
        "config": {
            "llm_name": args.llm_name,
            "llm_base_url": args.llm_base_url,
            "embedding_name": args.embedding_name,
            "embedding_base_url": args.embedding_base_url,
            "qa_top_k": int(args.qa_top_k),
            "recall_top_k": int(args.recall_top_k),
            "e2e_top_k": int(args.e2e_top_k),
            "ent_topk": int(args.ent_topk),
            "beta": float(args.beta),
            "step": int(args.step),
            "hgraph_device": args.hgraph_device,
            "no_think": True,
            "built_in_qa_skipped": bool(args.skip_qa),
        },
        "overall_retrieval_result": recall_overall,
        "overall_qa_results": (
            {"skipped": True}
            if args.skip_qa
            else {
                "ExactMatch": round(float(em), 4),
                "F1": round(float(f1), 4),
            }
        ),
        "paths": {
            "data_path": str(data_path),
            "corpus_path": str(corpus_path),
            "q_ner": str(out_dir / "ner" / "q_ner_resp.jsonl"),
            "c_ner": str(out_dir / "ner" / "c_ner_resp.jsonl"),
            "e2e_ret": str(e2e_path),
            "q2d_ret": str(q2d_path),
            "hgrag_ret": str(grag_ret_path),
            "hgrag_docs": str(grag_docs_path),
            "qa_pred": str(qa_pred_path),
        },
        "diagnostics": {
            "qner_empty": sum(1 for row in q_rows if not row.get("extracted_data", [{}])[0].get("entities")),
            "cner_empty": sum(1 for row in c_rows if not row.get("extracted_data", [{}])[0].get("named_entities")),
            "qa_empty": sum(1 for pred in predictions if not pred),
        },
        "records": records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset": args.dataset,
        "limit": args.limit,
        "overall_retrieval_result": output["overall_retrieval_result"],
        "overall_qa_results": output["overall_qa_results"],
        "output_json": str(args.output_json),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
