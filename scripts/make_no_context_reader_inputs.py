#!/usr/bin/env python3
"""Create reader-only QA inputs with no retrieved context."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DATASET_FILE_ALIASES = {
    "nq": "nq_rear",
    "natural_questions": "nq_rear",
}


def resolve_dataset_file_stem(dataset_name: str | None) -> str:
    normalized = str(dataset_name or "").strip().lower()
    return DATASET_FILE_ALIASES.get(normalized, str(dataset_name or "").strip())


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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        aliases: list[str] = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]


def get_gold_answers(sample: Mapping[str, Any]) -> list[str]:
    if "answer" in sample or "gold_ans" in sample:
        answers = parse_answer_alias_values(sample.get("answer", sample.get("gold_ans")))
    elif "reference" in sample:
        answers = parse_answer_alias_values(sample.get("reference"))
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
    return sorted({str(item).strip() for item in answers if str(item).strip()})


def get_gold_titles(sample: Mapping[str, Any]) -> list[str]:
    if "supporting_facts" in sample:
        return sorted({str(item[0]).strip() for item in sample["supporting_facts"] if item})
    if "contexts" in sample:
        return sorted(
            {
                str(item.get("title") or "").strip()
                for item in sample.get("contexts", [])
                if item.get("is_supporting") and str(item.get("title") or "").strip()
            }
        )
    return sorted(
        {
            str(item.get("title") or "").strip()
            for item in sample.get("paragraphs", [])
            if item.get("is_supporting") is not False and str(item.get("title") or "").strip()
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--method-name", default="gpt4omini_no_context")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    dataset_file_stem = resolve_dataset_file_stem(args.dataset)
    samples = json.loads(
        (Path(args.data_root) / f"{dataset_file_stem}.json").read_text(encoding="utf-8")
    )
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]

    examples = []
    for query_idx, sample in enumerate(samples):
        examples.append(
            {
                "query_index": int(query_idx),
                "question": str(sample.get("question") or ""),
                "gold_answers": get_gold_answers(sample),
                "gold_titles": get_gold_titles(sample),
                "docs": [],
                "retrieved_doc_ids": [],
            }
        )

    output = {
        "format": "no_context_reader_input_v1",
        "dataset": str(dataset_file_stem),
        "method": str(args.method_name),
        "source": "no_retrieval_no_context",
        "pool_k": 0,
        "limit": int(len(examples)),
        "overall_recomputed": {},
        "retrieval": {"recomputed_title_recall": {}},
        "config": {"external_pool_source_name": str(args.method_name)},
        "examples": examples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        errors="replace",
    )
    print(json.dumps({"output_json": str(args.output_json), "count": len(examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
