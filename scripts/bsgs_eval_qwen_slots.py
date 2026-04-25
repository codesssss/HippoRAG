#!/usr/bin/env python3
"""Evaluate Qwen-generated slot sets against MuSiQue oracle slots."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.io import load_dataset, read_jsonl, write_json, write_jsonl
from src.bsgs.prompts import QWEN_SLOT_GENERATION_PROMPT
from src.bsgs.slots import build_oracle_slots_for_sample, evaluate_slot_quality, parse_predicted_slots


def call_openai_compatible(base_url: str, model: str, prompt: str, timeout: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def load_predictions(path: str | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    preds: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid") or row.get("id") or "")
        if qid:
            preds[qid] = row.get("predicted_slots") or []
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--predicted_jsonl", default="")
    parser.add_argument("--llm_base_url", default="")
    parser.add_argument("--llm_name", default="qwen3-8b")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--allow_oracle_smoke", action="store_true")
    parser.add_argument("--predictions_out", default="data/processed/musique_qwen_slots.jsonl")
    parser.add_argument("--report_json", default="reports/week0/qwen_slot_quality.json")
    parser.add_argument("--report_md", default="reports/week0/qwen_slot_quality.md")
    args = parser.parse_args()

    _, samples = load_dataset(args.dataset, dataset_dir=args.dataset_dir)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    predictions = load_predictions(args.predicted_jsonl)
    generated_rows: list[dict[str, Any]] = []
    scores: list[dict[str, float]] = []
    errors: list[str] = []

    if not predictions and not args.llm_base_url and not args.allow_oracle_smoke:
        report = {
            "status": "not_run",
            "reason": "No predicted_jsonl or llm_base_url provided. Use --llm_base_url for real Qwen evaluation.",
            "num_examples": 0,
        }
        write_json(report, args.report_json)
        Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_md).write_text(
            "# Qwen Slot Quality\n\n"
            "- Status: `not_run`\n"
            "- Reason: no predictions or LLM endpoint provided.\n",
            encoding="utf-8",
        )
        print(f"Wrote not_run report to {args.report_json}")
        return

    for sample in samples:
        qid = str(sample.get("id") or sample.get("_id"))
        gold = build_oracle_slots_for_sample(sample)
        try:
            if qid in predictions:
                predicted = parse_predicted_slots(predictions[qid])
            elif args.allow_oracle_smoke:
                predicted = gold
            else:
                prompt = QWEN_SLOT_GENERATION_PROMPT.format(question=sample.get("question"))
                response = call_openai_compatible(args.llm_base_url, args.llm_name, prompt, args.timeout)
                predicted = parse_predicted_slots(response)
            generated_rows.append(
                {
                    "qid": qid,
                    "question": sample.get("question"),
                    "predicted_slots": [slot.to_dict() for slot in predicted],
                    "oracle_smoke": bool(args.allow_oracle_smoke and qid not in predictions and not args.llm_base_url),
                }
            )
            scores.append(evaluate_slot_quality(predicted, gold))
        except (ValueError, KeyError, urllib.error.URLError, TimeoutError) as exc:
            errors.append(f"{qid}: {exc}")

    if generated_rows:
        write_jsonl(generated_rows, args.predictions_out)
    avg = {
        key: sum(score.get(key, 0.0) for score in scores) / len(scores)
        for key in ["slot_precision", "slot_recall", "slot_f1", "variable_grounding_accuracy"]
    } if scores else {}
    report = {
        "status": "completed" if scores else "failed",
        "oracle_smoke": bool(args.allow_oracle_smoke),
        "num_examples": len(scores),
        "metrics": avg,
        "errors": errors[:20],
        "predictions_out": args.predictions_out if generated_rows else "",
    }
    write_json(report, args.report_json)
    lines = [
        "# Qwen Slot Quality",
        "",
        f"- Status: `{report['status']}`",
        f"- Oracle smoke: `{report['oracle_smoke']}`",
        f"- Examples: `{report['num_examples']}`",
    ]
    for key, value in avg.items():
        lines.append(f"- {key}: `{value:.4f}`")
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {err}" for err in errors[:10])
    out = Path(args.report_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report_json} and {args.report_md}")


if __name__ == "__main__":
    main()
