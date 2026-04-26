#!/usr/bin/env python3
"""Evaluate a HuggingFace seq2seq reader on D-PathRAG reader JSONL records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import write_json
from src.dpathrag.reader import (
    format_reader_input,
    gold_answers,
    load_reader_jsonl,
    score_prediction,
    summarize_scores,
    write_jsonl,
)


def mock_prediction(record: dict[str, Any], mode: str) -> str:
    if mode == "oracle":
        return gold_answers(record)[0]
    if mode == "empty":
        return ""
    raise ValueError(f"Unsupported mock mode: {mode}")


def generate_predictions_hf(
    records: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    max_docs: int,
    max_doc_chars: int,
    device: str,
) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    resolved_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()
    predictions: list[str] = []
    for start in range(0, len(records), int(batch_size)):
        batch = records[start : start + int(batch_size)]
        inputs = [
            format_reader_input(record, max_docs=int(max_docs), max_doc_chars=int(max_doc_chars))
            for record in batch
        ]
        encoded = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=int(max_input_tokens),
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(**encoded, max_new_tokens=int(max_new_tokens))
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--predictions_jsonl", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--mock_mode", choices=["none", "oracle", "empty"], default="none")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    records = load_reader_jsonl(args.input_jsonl, limit=int(args.limit))
    if args.mock_mode != "none":
        predictions = [mock_prediction(record, str(args.mock_mode)) for record in records]
        runtime = {"mode": f"mock_{args.mock_mode}"}
    elif args.model_name_or_path:
        predictions = generate_predictions_hf(
            records,
            model_name_or_path=str(args.model_name_or_path),
            batch_size=int(args.batch_size),
            max_input_tokens=int(args.max_input_tokens),
            max_new_tokens=int(args.max_new_tokens),
            max_docs=int(args.max_docs),
            max_doc_chars=int(args.max_doc_chars),
            device=str(args.device),
        )
        runtime = {"mode": "hf_seq2seq", "model_name_or_path": str(args.model_name_or_path)}
    else:
        raise ValueError("Provide --model_name_or_path or set --mock_mode oracle/empty for smoke testing.")

    scored = [score_prediction(record, prediction) for record, prediction in zip(records, predictions)]
    summary = summarize_scores(scored)
    output = {
        "input_jsonl": str(args.input_jsonl),
        "limit": int(args.limit),
        "runtime": runtime,
        "generation": {
            "batch_size": int(args.batch_size),
            "max_input_tokens": int(args.max_input_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "max_docs": int(args.max_docs),
            "max_doc_chars": int(args.max_doc_chars),
        },
        "summary": summary,
    }
    write_json(output, args.output_json)
    predictions_path = Path(args.predictions_jsonl) if args.predictions_jsonl else Path(args.output_json).with_suffix(".predictions.jsonl")
    write_jsonl(scored, predictions_path)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

