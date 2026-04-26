#!/usr/bin/env python3
"""Download and normalize full 2WikiMultiHopQA splits via HuggingFace mirror.

This script deliberately writes to ``data/dpathrag/full_2wiki`` instead of
``reproduce/dataset``.  The latter keeps the historical 1000-example protocol
used by existing DAEC/SetR/BSGS comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "framolfese/2WikiMultihopQA"
DEFAULT_OUTPUT_DIR = Path("data/dpathrag/full_2wiki")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def normalize_context(context: Any) -> list[list[Any]]:
    if isinstance(context, dict):
        titles = list(context.get("title") or [])
        sentences = list(context.get("sentences") or [])
        return [[str(title), list(sents or [])] for title, sents in zip(titles, sentences)]
    if isinstance(context, list):
        return context
    return []


def normalize_supporting_facts(facts: Any) -> list[list[Any]]:
    if isinstance(facts, dict):
        titles = list(facts.get("title") or [])
        sent_ids = list(facts.get("sent_id") or [])
        return [[str(title), int(sent_id)] for title, sent_id in zip(titles, sent_ids)]
    if isinstance(facts, list):
        return facts
    return []


def normalize_sample(row: dict[str, Any]) -> dict[str, Any]:
    qid = row.get("id") or row.get("_id")
    return {
        "_id": qid,
        "id": qid,
        "type": row.get("type"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "context": normalize_context(row.get("context")),
        "supporting_facts": normalize_supporting_facts(row.get("supporting_facts")),
        "evidences": row.get("evidences") or [],
    }


def context_to_corpus_docs(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, str], int] = {}
    title_to_texts: dict[str, set[str]] = {}
    title_conflicts: Counter[str] = Counter()
    for row in rows:
        for title, sentences in normalize_context(row.get("context")):
            text = " ".join(str(sentence) for sentence in sentences)
            if title in title_to_texts and text not in title_to_texts[title]:
                title_conflicts[title] += 1
            title_to_texts.setdefault(title, set()).add(text)
            by_key.setdefault((title, text), len(by_key))
    corpus = [
        {"idx": idx, "title": title, "text": text}
        for (title, text), idx in sorted(by_key.items(), key=lambda item: item[1])
    ]
    return corpus, {
        "unique_docs": len(corpus),
        "unique_titles": len(title_to_texts),
        "title_conflict_count": int(sum(title_conflicts.values())),
        "title_conflict_examples": dict(title_conflicts.most_common(10)),
        "dedupe_key": "title+text",
    }


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--hf_home", default="data/hf_cache")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split; 0 means full split.")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", str(args.hf_endpoint))
    os.environ.setdefault("HF_HOME", str(args.hf_home))

    from datasets import load_dataset  # noqa: WPS433

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset_name": str(args.dataset_name),
        "hf_endpoint": str(args.hf_endpoint),
        "hf_home": str(args.hf_home),
        "output_dir": str(output_dir),
        "splits": {},
        "separation_policy": "Full train/dev/test data live under data/dpathrag/full_2wiki; reproduce/dataset keeps the 1000-example protocol.",
    }
    all_rows_for_corpus: list[dict[str, Any]] = []
    for split in args.splits:
        hf_split = split
        ds = load_dataset(str(args.dataset_name), split=hf_split)
        if int(args.limit) > 0:
            ds = ds.select(range(min(int(args.limit), len(ds))))
        rows = [normalize_sample(dict(row)) for row in ds]
        all_rows_for_corpus.extend(rows)
        split_path = output_dir / f"2wikimultihopqa_{split}.json"
        write_json(rows, split_path)
        manifest["splits"][split] = {
            "rows": len(rows),
            "path": str(split_path),
            "fields": list(rows[0].keys()) if rows else [],
        }

    corpus, corpus_stats = context_to_corpus_docs(all_rows_for_corpus)
    corpus_path = output_dir / "2wikimultihopqa_full_corpus.json"
    write_json(corpus, corpus_path)
    manifest["corpus"] = {"path": str(corpus_path), **corpus_stats}

    manifest_path = output_dir / "manifest.json"
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
