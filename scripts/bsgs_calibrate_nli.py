#!/usr/bin/env python3
"""Calibrate a DeBERTa-MNLI verifier for BSGS Week 0."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.io import load_dataset, read_json, write_json
from src.bsgs.verifier import calibrate_temperature, measure_throughput


def build_hotpot_weak_pairs(samples: list[dict[str, Any]], max_pairs: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for sample in samples:
        question = str(sample.get("question") or "")
        support = {(title, sent_idx) for title, sent_idx in sample.get("supporting_facts") or []}
        for title, sentences in sample.get("context") or []:
            for sent_idx, sentence in enumerate(sentences):
                label = 1 if (title, sent_idx) in support else 0
                pairs.append({"premise": sentence, "hypothesis": question, "label": label})
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def run_transformers_model(model_name: str, pairs: list[dict[str, Any]], batch_size: int) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"transformers runtime unavailable: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [p["premise"] for p in batch],
                [p["hypothesis"] for p in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            out = model(**encoded)
            logits.append(out.logits.detach().cpu().numpy())
    return np.concatenate(logits, axis=0) if logits else np.zeros((0, 3), dtype=float)


def lexical_smoke_logits(pairs: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for pair in pairs:
        premise_tokens = {tok for tok in pair["premise"].lower().split() if len(tok) > 3}
        hyp_tokens = {tok for tok in pair["hypothesis"].lower().split() if len(tok) > 3}
        overlap = len(premise_tokens & hyp_tokens) / max(1, len(hyp_tokens))
        rows.append([1.0 - overlap, 0.5, overlap])
    return np.asarray(rows, dtype=float)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# NLI Calibration",
        "",
        f"- Status: `{report['status']}`",
        f"- Model: `{report.get('model_name', '')}`",
        f"- Calibration source: `{report.get('calibration_source', '')}`",
        f"- Examples: `{report.get('num_examples', 0)}`",
        f"- ECE: `{report.get('ece', 0.0):.4f}`",
        f"- Brier: `{report.get('brier', 0.0):.4f}`",
        f"- NLL: `{report.get('nll', 0.0):.4f}`",
        f"- Temperature: `{report.get('temperature', 1.0):.3f}`",
        f"- Absorbing status: `{report.get('absorbing_status', 'unknown')}`",
    ]
    if report.get("error"):
        lines.extend(["", f"Error: `{report['error']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base-mnli")
    parser.add_argument("--input_logits_json", default="")
    parser.add_argument("--max_pairs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--allow_lexical_smoke", action="store_true")
    parser.add_argument("--support_class_index", type=int, default=2)
    parser.add_argument("--report_json", default="reports/week0/nli_calibration.json")
    parser.add_argument("--throughput_json", default="reports/week0/nli_throughput.json")
    parser.add_argument("--report_md", default="reports/week0/nli_calibration.md")
    args = parser.parse_args()

    report: dict[str, Any]
    try:
        if args.input_logits_json:
            payload = read_json(args.input_logits_json)
            logits = np.asarray(payload["logits"], dtype=float)
            labels = [int(v) for v in payload["labels"]]
            source = args.input_logits_json
            throughput = {"status": "not_measured"}
        else:
            _, samples = load_dataset(args.dataset, dataset_dir=args.dataset_dir)
            pairs = build_hotpot_weak_pairs(samples, max_pairs=args.max_pairs)
            labels = [int(pair["label"]) for pair in pairs]
            source = f"{args.dataset} weak sentence labels"
            if args.allow_lexical_smoke:
                logits = lexical_smoke_logits(pairs)
                throughput = measure_throughput(lambda items: lexical_smoke_logits(list(items)), pairs)
                source += " (lexical smoke, not DeBERTa)"
            else:
                start = perf_counter()
                logits = run_transformers_model(args.model_name, pairs, batch_size=args.batch_size)
                elapsed = max(perf_counter() - start, 1e-12)
                throughput = {
                    "status": "completed",
                    "num_examples": float(len(pairs)),
                    "elapsed_seconds": elapsed,
                    "samples_per_second": float(len(pairs) / elapsed),
                    "latency_per_1000_seconds": float(1000.0 * elapsed / max(1, len(pairs))),
                }
        result = calibrate_temperature(logits, labels, support_class_index=args.support_class_index)
        absorbing_status = "main" if result.ece <= 0.15 and not args.allow_lexical_smoke else "ablation"
        report = {
            "status": "completed",
            "model_name": args.model_name,
            "calibration_source": source,
            "num_examples": result.num_examples,
            "temperature": result.temperature,
            "ece": result.ece,
            "brier": result.brier,
            "nll": result.nll,
            "absorbing_status": absorbing_status,
            "lexical_smoke": bool(args.allow_lexical_smoke),
        }
        write_json(throughput, args.throughput_json)
    except Exception as exc:  # external model failures should not corrupt Week-0 state
        report = {
            "status": "model_unavailable",
            "model_name": args.model_name,
            "num_examples": 0,
            "ece": 1.0,
            "brier": 1.0,
            "nll": 0.0,
            "temperature": 1.0,
            "absorbing_status": "ablation",
            "error": str(exc),
        }
        write_json({"status": "not_measured", "error": str(exc)}, args.throughput_json)
    write_json(report, args.report_json)
    write_markdown(report, Path(args.report_md))
    print(f"Wrote {args.report_json}, {args.throughput_json}, and {args.report_md}")


if __name__ == "__main__":
    main()
