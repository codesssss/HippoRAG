#!/usr/bin/env python3
"""Export PropRAG retrieval pools without modifying the PropRAG repository.

This script is intentionally HippoRAG-side glue. It imports PropRAG read-only,
loads an existing clean index directory, runs retrieval only, and writes
per-query top-N pools for downstream oracle/composition experiments.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_PROPRAG_ROOT = Path("/mnt/nvme/code/PropRAG")
DEFAULT_SAVE_DIR = Path(
    "/mnt/nvme/code/PropRAG/outputs_aligned_clean_nothink_top100_nvembed_rebuild_20260423"
)
DATASET_FILE_ALIASES = {
    "nq": "nq_rear",
    "natural_questions": "nq_rear",
}


def resolve_dataset_file_stem(dataset_name: str | None) -> str:
    normalized = str(dataset_name or "").strip().lower()
    return DATASET_FILE_ALIASES.get(normalized, str(dataset_name or "").strip())


def string_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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
        gold_docs.append(list(set(docs)))
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


def build_doc_id_lookup(corpus_docs: Sequence[str]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, doc in enumerate(corpus_docs):
        lookup.setdefault(str(doc), int(idx))
    return lookup


def compute_title_recall(
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
    k_values: Iterable[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in k_values:
        hits = []
        for gold, retrieved in zip(gold_docs, retrieved_docs):
            gold_titles = {extract_title(doc) for doc in gold}
            retrieved_titles = {extract_title(doc) for doc in list(retrieved)[: int(k)]}
            if not gold_titles:
                hits.append(0.0)
                continue
            hits.append(len(gold_titles & retrieved_titles) / len(gold_titles))
        metrics[f"Recall@{int(k)}"] = round(float(sum(hits) / max(len(hits), 1)), 4)
    return metrics


def sanitize_config(config: Any) -> Dict[str, Any]:
    payload = asdict(config)
    payload["api_key"] = "<redacted>"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PropRAG top-N retrieval pools.")
    parser.add_argument("--proprag_root", type=Path, default=DEFAULT_PROPRAG_ROOT)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--llm_name", default="qwen3-8b")
    parser.add_argument("--llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--embedding_name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--retrieval_top_k", type=int, default=100)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--reuse_preextracted_openie", type=string_to_bool, default=True)
    parser.add_argument("--openie_llm_name", default="qwen3-8b-train")
    parser.add_argument("--force_index_from_scratch", type=string_to_bool, default=False)
    parser.add_argument("--force_openie_from_scratch", type=string_to_bool, default=False)
    parser.add_argument("--openie_mode", choices=["online", "offline"], default="online")
    parser.add_argument("--use_propositions", type=string_to_bool, default=True)
    parser.add_argument("--use_beam_search", type=string_to_bool, default=True)
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--max_path_length", type=int, default=3)
    parser.add_argument("--second_stage_filter_k", type=int, default=40)
    parser.add_argument("--sim_threshold", type=float, default=0.75)
    args = parser.parse_args()

    proprag_root = args.proprag_root.resolve()
    output_json = args.output_json.resolve()
    save_dir = args.save_dir.resolve()
    if not proprag_root.exists():
        raise FileNotFoundError(f"PropRAG root not found: {proprag_root}")

    sys.path.insert(0, str(proprag_root))
    os.chdir(proprag_root)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.proprag.PropRAG import PropRAG  # noqa: WPS433
    from src.proprag.utils.config_utils import BaseConfig  # noqa: WPS433

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_file_stem = resolve_dataset_file_stem(args.dataset)
    corpus_path = proprag_root / "reproduce" / "dataset" / f"{dataset_file_stem}_corpus.json"
    samples_path = proprag_root / "reproduce" / "dataset" / f"{dataset_file_stem}.json"
    corpus = json.loads(corpus_path.read_text())
    samples = json.loads(samples_path.read_text())
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]

    corpus_docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    doc_id_lookup = build_doc_id_lookup(corpus_docs)
    queries = [sample["question"] for sample in samples]
    gold_docs = get_gold_docs(samples, args.dataset)
    gold_answers = get_gold_answers(samples)

    dataset_save_dir = save_dir / args.dataset
    config = BaseConfig(
        save_dir=str(dataset_save_dir),
        llm_base_url=str(args.llm_base_url),
        llm_name=str(args.llm_name),
        dataset=str(args.dataset),
        embedding_model_name=str(args.embedding_name),
        embedding_base_url=str(args.embedding_base_url),
        force_index_from_scratch=bool(args.force_index_from_scratch),
        force_openie_from_scratch=bool(args.force_openie_from_scratch),
        retrieval_top_k=int(args.retrieval_top_k),
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=int(args.qa_top_k),
        max_new_tokens=int(args.max_new_tokens),
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=int(args.embedding_batch_size),
        beam_width=int(args.beam_width),
        max_path_length=int(args.max_path_length),
        second_stage_filter_k=int(args.second_stage_filter_k),
        corpus_len=len(corpus),
        openie_mode=str(args.openie_mode),
        use_propositions=bool(args.use_propositions),
        qa_max_workers=1,
    )

    if bool(args.reuse_preextracted_openie):
        source_name = str(args.openie_llm_name).replace("/", "_")
        target_name = config.llm_name.replace("/", "_")
        source_openie = proprag_root / "outputs" / args.dataset / f"openie_results_ner_{source_name}.json"
        target_openie = Path(config.save_dir) / f"openie_results_ner_{target_name}.json"
        if source_openie.exists() and not target_openie.exists():
            target_openie.parent.mkdir(parents=True, exist_ok=True)
            target_openie.write_bytes(source_openie.read_bytes())

    proprag = PropRAG(global_config=config)
    proprag.index(corpus_docs)
    retrieval = proprag.retrieve(
        queries=queries,
        num_to_retrieve=int(args.pool_k),
        gold_docs=gold_docs,
        use_beam_search=bool(args.use_beam_search),
        beam_width=int(args.beam_width),
        second_stage_filter_k=int(args.second_stage_filter_k),
        sim_threshold=float(args.sim_threshold),
    )
    if isinstance(retrieval, tuple):
        query_solutions, proprag_retrieval_metrics = retrieval
    else:
        query_solutions = retrieval
        proprag_retrieval_metrics = {}

    records: List[Dict[str, Any]] = []
    for idx, solution in enumerate(query_solutions):
        pool_docs = list(solution.docs[: int(args.pool_k)])
        scores = list(solution.doc_scores[: int(args.pool_k)] if solution.doc_scores is not None else [])
        records.append(
            {
                "query_idx": int(idx),
                "question": str(solution.question),
                "gold_answers": list(gold_answers[idx]),
                "gold_docs": list(gold_docs[idx]),
                "gold_titles": [extract_title(doc) for doc in gold_docs[idx]],
                "pool_k": int(args.pool_k),
                "pool_docs": pool_docs,
                "pool_titles": [extract_title(doc) for doc in pool_docs],
                "pool_doc_scores": [float(score) for score in scores],
                "pool_doc_ids": [doc_id_lookup.get(doc) for doc in pool_docs],
            }
        )

    recall_metrics = compute_title_recall(
        gold_docs=gold_docs,
        retrieved_docs=[record["pool_docs"] for record in records],
        k_values=[5, 20, 100],
    )
    output = {
        "dataset": str(args.dataset),
        "limit": int(len(samples)),
        "pool_k": int(args.pool_k),
        "source": "proprag_clean_nothink_retrieval_export",
        "proprag_root": str(proprag_root),
        "save_dir": str(dataset_save_dir),
        "config": sanitize_config(config),
        "retrieval": {
            "recomputed_title_recall": recall_metrics,
            "proprag_metrics": proprag_retrieval_metrics,
        },
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "dataset": args.dataset,
                "limit": len(samples),
                "pool_k": int(args.pool_k),
                "recomputed_title_recall": recall_metrics,
                "proprag_metrics": proprag_retrieval_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
