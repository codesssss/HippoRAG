#!/usr/bin/env python3
"""Convert ReLink retrieval-only outputs into saved-docs reader inputs.

ReLink's ``results_expand_COG.json`` stores graph paths rather than a plain
ranked document list.  This adapter preserves the path order, groups the
source-grounded chunks/evidence by title, and exports the first ``qa_top_k``
unique source titles as ``examples[*].docs`` for the shared GPT reader.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping


DATASET_ALIASES = {
    "hotpotqa1000": "hotpotqa",
    "wikimqa1000": "2wikimultihopqa",
    "popqa1000": "popqa",
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def parse_maybe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except Exception:
                    continue
                if isinstance(parsed, list):
                    return parsed
        return [text]
    return [value]


def normalize_answers(value: Any) -> list[str]:
    out: list[str] = []
    for item in parse_maybe_list(value):
        if isinstance(item, (list, tuple, set)):
            out.extend(normalize_answers(list(item)))
        elif item is not None:
            text = str(item).strip()
            if text:
                out.append(text)
    return sorted(set(out))


def sample_gold_answers(sample: Mapping[str, Any]) -> list[str]:
    answers = normalize_answers(sample.get("answer"))
    answers.extend(normalize_answers(sample.get("possible_answers")))
    answers.extend(normalize_answers(sample.get("obj")))
    answers.extend(normalize_answers(sample.get("o_aliases")))
    return sorted({answer for answer in answers if answer})


def sample_gold_titles(sample: Mapping[str, Any]) -> list[str]:
    titles: list[str] = []
    for fact in sample.get("supporting_facts") or []:
        if isinstance(fact, (list, tuple)) and fact:
            titles.append(str(fact[0]))
        elif isinstance(fact, Mapping):
            for key in ("title", "source", "doc_title"):
                if fact.get(key):
                    titles.append(str(fact[key]))
                    break
    if not titles:
        for paragraph in sample.get("paragraphs") or []:
            if isinstance(paragraph, Mapping) and paragraph.get("is_supporting") and paragraph.get("title"):
                titles.append(str(paragraph["title"]))
    return list(dict.fromkeys(title.strip() for title in titles if title and title.strip()))


def add_piece(store: OrderedDict[str, list[str]], title: Any, piece: Any) -> None:
    title_text = str(title or "").strip()
    piece_text = str(piece or "").strip()
    if not title_text or not piece_text:
        return
    bucket = store.setdefault(title_text, [])
    if piece_text not in bucket:
        bucket.append(piece_text)


def add_doc_from_meta(store: OrderedDict[str, list[str]], meta: Mapping[str, Any]) -> None:
    title = meta.get("title")
    chunks = meta.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            add_piece(store, title, chunk)
    else:
        add_piece(store, title, chunks)


def add_node(store: OrderedDict[str, list[str]], node: Mapping[str, Any]) -> None:
    meta = node.get("meta")
    if isinstance(meta, Mapping):
        add_doc_from_meta(store, meta)

    source = node.get("source")
    for key in ("description", "mention", "name"):
        if node.get(key):
            add_piece(store, source, node[key])

    source_list = node.get("source_list")
    description_list = node.get("description_list")
    if isinstance(source_list, list) and isinstance(description_list, list):
        for src, desc in zip(source_list, description_list):
            add_piece(store, src, desc)


def add_relation(store: OrderedDict[str, list[str]], relation: Mapping[str, Any]) -> None:
    title = relation.get("title")
    evidence = relation.get("evidence")
    rel = relation.get("r")
    if evidence and rel:
        add_piece(store, title, f"{evidence} ({rel})")
    else:
        add_piece(store, title, evidence)
    for side in ("begin", "end"):
        endpoint = relation.get(side)
        if isinstance(endpoint, Mapping):
            source = endpoint.get("source")
            for key in ("description", "mention", "name"):
                if endpoint.get(key):
                    add_piece(store, source, endpoint[key])


def docs_from_relink_context(context: Any, qa_top_k: int, max_pieces_per_doc: int) -> tuple[list[str], list[str]]:
    store: OrderedDict[str, list[str]] = OrderedDict()
    for path in context or []:
        if not isinstance(path, Mapping):
            continue
        for node in path.get("nodes") or []:
            if isinstance(node, Mapping):
                add_node(store, node)
        for relation in path.get("relations") or []:
            if isinstance(relation, Mapping):
                add_relation(store, relation)
        if len(store) >= qa_top_k:
            break

    titles = list(store.keys())[:qa_top_k]
    docs: list[str] = []
    for title in titles:
        pieces = store.get(title, [])[:max_pieces_per_doc]
        doc = "\n".join([title, *pieces]).strip()
        docs.append(doc)
    return titles, docs


def compute_recall(examples: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    total_gold = 0
    hit_gold = 0
    all_hit = 0
    any_hit = 0
    count = 0
    for example in examples:
        count += 1
        gold = {normalize_text(title) for title in example.get("gold_titles", []) if normalize_text(title)}
        pred = {normalize_text(title) for title in example.get("retrieved_doc_ids", [])[:5] if normalize_text(title)}
        if not gold:
            continue
        hit = len(gold & pred)
        total_gold += len(gold)
        hit_gold += hit
        all_hit += int(hit == len(gold))
        any_hit += int(hit > 0)
    return {
        "Recall@5": hit_gold / total_gold if total_gold else 0.0,
        "All@5": all_hit / count if count else 0.0,
        "Any@5": any_hit / count if count else 0.0,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    relink_root = Path(args.relink_root).expanduser()
    dataset = args.dataset
    result_path = Path(args.results_json).expanduser() if args.results_json else (
        relink_root / "results" / dataset / "results_expand_COG.json"
    )
    samples_path = Path(args.samples_json).expanduser() if args.samples_json else (
        relink_root / "data" / dataset / "dataset" / "samples.json"
    )
    output_path = Path(args.output_json).expanduser()

    results = load_json(result_path)
    samples = load_json(samples_path)
    samples_by_question = {
        str(sample.get("question") or ""): sample
        for sample in samples
        if isinstance(sample, Mapping)
    }

    examples: list[dict[str, Any]] = []
    for offset, item in enumerate(results):
        if not isinstance(item, Mapping):
            continue
        query_index = item.get("query_index")
        sample = None
        if isinstance(query_index, int) and 0 <= query_index < len(samples):
            maybe_sample = samples[query_index]
            if isinstance(maybe_sample, Mapping):
                sample = maybe_sample
        if not isinstance(sample, Mapping) or sample.get("question") != item.get("question"):
            sample = samples_by_question.get(str(item.get("question") or ""))
        if not isinstance(sample, Mapping):
            continue

        titles, docs = docs_from_relink_context(
            item.get("context"),
            qa_top_k=int(args.qa_top_k),
            max_pieces_per_doc=int(args.max_pieces_per_doc),
        )
        examples.append(
            {
                "query_index": int(query_index if isinstance(query_index, int) else offset),
                "question": str(item.get("question") or sample.get("question") or ""),
                "gold_answers": sample_gold_answers(sample),
                "gold_titles": sample_gold_titles(sample),
                "docs": docs,
                "retrieved_doc_ids": titles,
                "retrieval_trace": {
                    "source_output_json": str(result_path.resolve()),
                    "source_method": "ReLink",
                    "source_format": "results_expand_COG.json",
                    "does_not_modify_retrieval": True,
                    "doc_adapter": "path_context_to_source_title_docs",
                },
            }
        )

    metrics = compute_recall(examples)
    output_payload = {
        "format": "relink_saved_docs_reader_input_v1",
        "dataset": args.output_dataset or DATASET_ALIASES.get(dataset.lower(), dataset),
        "display_dataset": dataset,
        "method": args.method,
        "source": str(result_path.resolve()),
        "samples": str(samples_path.resolve()),
        "limit": len(examples),
        "num_queries": len(examples),
        "retrieval": {
            "recomputed_title_recall": metrics,
            "mapping": "ReLink path/context grouped by source title; first qa_top_k titles become reader docs",
        },
        "overall_recomputed": metrics,
        "config": {
            "qa_top_k": int(args.qa_top_k),
            "max_pieces_per_doc": int(args.max_pieces_per_doc),
            "reader_replay_protocol": "relink_path_context_gpt4omini",
        },
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output_json": str(output_path),
        "count": len(examples),
        "retrieval": metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--relink-root", default="/mnt/nvme/code/Relink")
    parser.add_argument("--results-json", default="")
    parser.add_argument("--samples-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dataset", default="")
    parser.add_argument("--method", default="ReLink")
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--max-pieces-per-doc", type=int, default=16)
    return parser


def main() -> int:
    payload = convert_dataset(build_arg_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
