#!/usr/bin/env python3
"""Evaluate all D-PathRAG context ablation reader inputs with one HF model load."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import read_json, write_json
from src.dpathrag.reader import format_reader_input, load_reader_jsonl, score_prediction, summarize_scores, write_jsonl


def generate_predictions(
    records: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    max_docs: int,
    max_doc_chars: int,
) -> list[str]:
    import torch
    from tqdm import tqdm

    predictions: list[str] = []
    for start in tqdm(range(0, len(records), int(batch_size)), desc="Evaluating reader"):
        batch = list(records[start : start + int(batch_size)])
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
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(**encoded, max_new_tokens=int(max_new_tokens))
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return predictions


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    has_diagnostics = any(bool(payload.get("diagnostics")) for payload in summary["configs"].values())
    lines = [
        "# D-PathRAG PropRAG Context Ablation",
        "",
        f"- Reader: `{summary['model_name_or_path']}`",
        f"- Rows per config: {summary['rows_per_config']}",
        "",
    ]
    if has_diagnostics:
        lines.extend(
            [
                "| Config | Support Recall | Support Complete | Answer EM | Answer F1 | Anchor Docs | Selector Added | Added Gold | Rank Gold Removed | Net Gold Gain |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| Config | Support Recall | Support Complete | Answer EM | Answer F1 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
    for config, payload in summary["configs"].items():
        reader_summary = payload["reader_summary"]
        input_summary = payload["input_summary"]
        cells = [
            config,
            f"{input_summary['avg_support_recall']:.4f}",
            f"{input_summary['support_complete_rate']:.4f}",
            f"{reader_summary['answer_em']:.4f}",
            f"{reader_summary['answer_f1']:.4f}",
        ]
        if has_diagnostics:
            diagnostics = payload.get("diagnostics") or {}
            cells.extend(
                [
                    f"{float(diagnostics.get('avg_rank_anchor_docs') or 0.0):.4f}",
                    f"{float(diagnostics.get('avg_selector_added_docs') or 0.0):.4f}",
                    f"{float(diagnostics.get('avg_selector_added_gold') or 0.0):.4f}",
                    f"{float(diagnostics.get('avg_rank_tail_gold_removed') or 0.0):.4f}",
                    f"{float(diagnostics.get('avg_net_gold_gain') or 0.0):.4f}",
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_json", default="reports/dpathrag/context_ablation_proprag_eval200_manifest.json")
    parser.add_argument("--model_name_or_path", default="data/dpathrag/models/flan_t5_base_gold_k5_5k")
    parser.add_argument("--output_json", default="reports/dpathrag/context_ablation_proprag_eval200_results.json")
    parser.add_argument("--output_md", default="reports/dpathrag/context_ablation_proprag_eval200_results.md")
    parser.add_argument("--predictions_dir", default="reports/dpathrag/context_ablation_predictions")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    manifest = read_json(args.manifest_json)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_name_or_path))
    device = torch.device(str(args.device) if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    predictions_dir = Path(args.predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "manifest_json": str(args.manifest_json),
        "model_name_or_path": str(args.model_name_or_path),
        "rows_per_config": int(manifest.get("prediction_rows") or 0),
        "generation": {
            "batch_size": int(args.batch_size),
            "max_input_tokens": int(args.max_input_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "max_docs": int(args.max_docs),
            "max_doc_chars": int(args.max_doc_chars),
        },
        "configs": {},
    }
    for config, payload in manifest["configs"].items():
        records = load_reader_jsonl(payload["output_jsonl"])
        predictions = generate_predictions(
            records,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=int(args.batch_size),
            max_input_tokens=int(args.max_input_tokens),
            max_new_tokens=int(args.max_new_tokens),
            max_docs=int(args.max_docs),
            max_doc_chars=int(args.max_doc_chars),
        )
        scored = [score_prediction(record, prediction) for record, prediction in zip(records, predictions)]
        predictions_path = predictions_dir / f"{config}.predictions.jsonl"
        write_jsonl(scored, predictions_path)
        summary["configs"][config] = {
            "input_jsonl": payload["output_jsonl"],
            "predictions_jsonl": str(predictions_path),
            "input_summary": payload["summary"],
            "diagnostics": payload.get("diagnostics", {}),
            "reader_summary": summarize_scores(scored),
        }
        write_json(summary, args.output_json)
        write_markdown(summary, args.output_md)
        print(json.dumps({config: summary["configs"][config]["reader_summary"]}, ensure_ascii=False, indent=2))

    write_json(summary, args.output_json)
    write_markdown(summary, args.output_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
