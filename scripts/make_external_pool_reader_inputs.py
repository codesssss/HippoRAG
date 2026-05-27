#!/usr/bin/env python3
"""Wrap exported retrieval pools for reader-only QA replay.

The pool exporters write ``records`` with full ``pool_docs``.  The reader replay
expects ``examples`` plus a retrieval trace pointing back to the immutable pool.
This adapter preserves the retrieval metrics and avoids copying all passages
into the reader input wrapper.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Mapping


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    retrieval = payload.get("retrieval")
    if isinstance(retrieval, Mapping):
        metrics = retrieval.get("recomputed_title_recall")
        if isinstance(metrics, Mapping):
            return {str(key): float(value) for key, value in metrics.items()}
    return {}


def parse_answer_alias_values(value: Any) -> list[str]:
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
    if isinstance(value, (list, tuple, set)):
        aliases: list[str] = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]


def normalize_gold_answers(value: Any) -> list[str]:
    return sorted(
        {str(item).strip() for item in parse_answer_alias_values(value) if str(item).strip()}
    )


def make_examples(pool_path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for offset, record in enumerate(payload.get("records", []) or []):
        if not isinstance(record, Mapping):
            continue
        query_idx = int(record.get("query_idx", offset))
        pool_doc_ids = list(record.get("pool_doc_ids", []) or [])
        examples.append(
            {
                "query_index": query_idx,
                "question": str(record.get("question") or ""),
                "gold_answers": normalize_gold_answers(record.get("gold_answers", [])),
                "gold_titles": list(record.get("gold_titles", []) or []),
                "retrieved_doc_ids": pool_doc_ids[:5],
                "retrieval_trace": {
                    "external_pool_path": str(pool_path.resolve()),
                    "external_query_idx": query_idx,
                    "external_pool_source": str(payload.get("source") or ""),
                },
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-json", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    pool_path = Path(args.pool_json).expanduser()
    payload = load_json(pool_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected object payload in {pool_path}")

    metrics = retrieval_metrics(payload)
    output = {
        "format": "external_pool_reader_input_v1",
        "dataset": str(payload.get("dataset") or ""),
        "method": str(args.method_name),
        "source": str(payload.get("source") or ""),
        "pool_k": int(payload.get("pool_k") or 0),
        "limit": int(payload.get("limit") or 0),
        "overall_recomputed": metrics,
        "retrieval": {"recomputed_title_recall": metrics},
        "config": {
            "external_pool_source_name": str(args.method_name),
            "external_pool_path": str(pool_path.resolve()),
        },
        "examples": make_examples(pool_path, payload),
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(output_path), "count": len(output["examples"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
