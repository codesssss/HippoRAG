#!/usr/bin/env python3
"""Export HippoRAG (legacy_fact_graph) retrieval pools in external-pool JSON format.

Runs HippoRAG PPR-based retrieval for each query and writes a fixed pool
consumable by ``eval_causal_qwen3.py --external_pool_json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DATASET_FILE_ALIASES = {
    "nq": "nq_rear",
    "natural_questions": "nq_rear",
}
NO_THINK_PREFIX = "/no_think"


def _strip_qwen_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", str(text or ""), flags=re.DOTALL).strip()


def _with_qwen_no_think_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    patched: List[Dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        content = msg.get("content")
        if (
            str(msg.get("role", "")).lower() == "user"
            and isinstance(content, str)
            and NO_THINK_PREFIX not in content[:128]
        ):
            msg["content"] = f"{NO_THINK_PREFIX}\n{content}"
        patched.append(msg)
    return patched


def install_qwen_disable_thinking(llm_model: Any, enabled: bool) -> None:
    if not enabled or llm_model is None:
        return
    llm_config = getattr(llm_model, "global_config", None)
    model_name = str(
        getattr(llm_config, "llm_request_name", None)
        or getattr(llm_config, "llm_name", None)
        or getattr(llm_model, "llm_name", "")
        or ""
    ).lower()
    if "qwen" not in model_name:
        return
    if getattr(llm_model, "_codex_qwen_disable_thinking_installed", False):
        return

    cache_file_name = getattr(llm_model, "cache_file_name", None)
    if isinstance(cache_file_name, str) and cache_file_name:
        if "_no_think_cache" not in os.path.basename(cache_file_name):
            if cache_file_name.endswith("_cache.sqlite"):
                llm_model.cache_file_name = cache_file_name[: -len("_cache.sqlite")] + "_no_think_cache.sqlite"
            else:
                root, ext = os.path.splitext(cache_file_name)
                llm_model.cache_file_name = f"{root}_no_think{ext}"

    original_infer = llm_model.infer

    def no_think_infer(messages, *infer_args, **infer_kwargs):
        if isinstance(messages, list):
            messages = _with_qwen_no_think_messages(messages)
        extra_body = dict(infer_kwargs.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        infer_kwargs["extra_body"] = extra_body
        result = original_infer(messages, *infer_args, **infer_kwargs)
        if isinstance(result, tuple) and result and isinstance(result[0], str):
            return (_strip_qwen_thinking(result[0]), *result[1:])
        return result

    llm_model.infer = no_think_infer
    llm_model._codex_qwen_disable_thinking_installed = True


def resolve_dataset_file_stem(dataset_name: str | None) -> str:
    normalized = str(dataset_name or "").strip().lower()
    return DATASET_FILE_ALIASES.get(normalized, str(dataset_name or "").strip())


def extract_title(doc: str) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def normalize_doc_text(doc: str) -> str:
    return re.sub(r"\s+", " ", str(doc or "")).strip()


def parse_answer_alias_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                return [cleaned]
            return parse_answer_alias_values(parsed)
        return [cleaned]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        aliases: List[str] = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]


def string_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_doc_id_index(docs: Sequence[str]) -> Dict[str, int]:
    """Map retrieved document text back to corpus/OpenIE document indices."""

    index: Dict[str, int] = {}
    for doc_id, doc in enumerate(docs):
        key = normalize_doc_text(doc)
        if key and key not in index:
            index[key] = int(doc_id)
    return index


def pool_doc_ids_for_docs(pool_docs: Sequence[str], doc_id_index: Mapping[str, int]) -> List[int]:
    doc_ids: List[int] = []
    unresolved: List[str] = []
    for doc in pool_docs:
        key = normalize_doc_text(doc)
        doc_id = doc_id_index.get(key)
        if doc_id is None:
            unresolved.append(extract_title(doc) or key[:80])
            continue
        doc_ids.append(int(doc_id))
    if unresolved:
        preview = "; ".join(unresolved[:5])
        raise ValueError(f"Could not resolve {len(unresolved)} retrieved docs to corpus indices: {preview}")
    return doc_ids


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
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--force_index_from_scratch", type=string_to_bool, default=False)
    parser.add_argument("--force_openie_from_scratch", type=string_to_bool, default=False)
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    parser.add_argument("--qwen_disable_thinking", action="store_true")
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    dataset_file_stem = resolve_dataset_file_stem(args.dataset)
    samples_path = args.data_root / f"{dataset_file_stem}.json"
    corpus_path = args.data_root / f"{dataset_file_stem}_corpus.json"
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
    doc_id_index = build_doc_id_index(docs)

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
        embedding_batch_size=int(args.embedding_batch_size),
        max_new_tokens=int(args.max_new_tokens),
        force_index_from_scratch=bool(args.force_index_from_scratch),
        force_openie_from_scratch=bool(args.force_openie_from_scratch),
        openie_mode=str(args.openie_mode),
        corpus_len=len(docs),
        graph_type="facts_and_sim_passage_node_unidirectional",
        # Use HippoRAG's legacy graph index/retrieval path. The V2 wrapper with
        # causal disabled only builds dense chunk embeddings and can silently
        # fall back to dense ranking for legacy_fact_graph retrieval.
        causal_v2_base_retrieval_mode="dense",
        causal_engine_version="legacy",
        causal_enabled=False,
        causal_context_max_items=0,
    )

    print(f"Initializing HippoRAG for {args.dataset} with legacy graph retrieval...")
    hipporag = HippoRAG(global_config=config)
    install_qwen_disable_thinking(
        getattr(hipporag, "llm_model", None),
        enabled=bool(args.qwen_disable_thinking),
    )
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
        pool_doc_ids = pool_doc_ids_for_docs(pool_docs, doc_id_index)
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
            "pool_doc_ids": pool_doc_ids,
        })

    recall = compute_title_recall(gold_docs, retrieved_doc_lists, [5, 20, 100, 200])
    output = {
        "dataset": str(args.dataset),
        "limit": len(samples),
        "pool_k": pool_k,
        "source": "hipporag_legacy_graph_export",
        "save_dir": str(args.save_dir),
        "config": {
            "causal_engine_version": "legacy",
            "base_retrieval_mode": "legacy_graph",
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
