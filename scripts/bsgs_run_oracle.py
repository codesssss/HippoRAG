#!/usr/bin/env python3
"""Run a minimal oracle-slot BSGS operator diagnostic."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.io import load_dataset, read_json, read_jsonl, write_json
from src.bsgs.runner_oracle import run_oracle_bsgs_sample
from src.bsgs.slots import Slot


def slot_records_by_qid(path: str) -> dict[str, list[Slot]]:
    records = read_jsonl(path)
    by_qid: dict[str, list[Slot]] = {}
    for row in records:
        qid = str(row.get("qid"))
        by_qid[qid] = [Slot(**slot) for slot in row.get("slots") or []]
    return by_qid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--oracle_slots", default="data/processed/musique_oracle_slots.jsonl")
    parser.add_argument("--nli_calibration", default="reports/week0/nli_calibration.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--force_absorbing", action="store_true")
    parser.add_argument("--json_out", default="reports/week1/operator_validation.json")
    args = parser.parse_args()

    _, samples = load_dataset(args.dataset, dataset_dir=args.dataset_dir)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    slots_by_qid = slot_records_by_qid(args.oracle_slots)
    calibration = read_json(args.nli_calibration) if Path(args.nli_calibration).exists() else {}
    absorbing_enabled = bool(args.force_absorbing or calibration.get("absorbing_status") == "main")

    rows = []
    for sample in samples:
        qid = str(sample.get("id") or sample.get("_id"))
        slots = slots_by_qid.get(qid)
        if not slots:
            continue
        rows.append(run_oracle_bsgs_sample(sample, slots, absorbing_enabled=absorbing_enabled, top_k=args.top_k))

    avg_recall = sum(row["supporting_paragraph_recall"] for row in rows) / len(rows) if rows else 0.0
    report = {
        "status": "completed",
        "dataset": args.dataset,
        "n": len(rows),
        "slot_mode": "oracle-slot",
        "binding": "hard normalized string / alias match",
        "verifier": "calibrated DeBERTa-NLI if available; lexical likelihood fallback in this minimal runner",
        "absorbing_enabled": absorbing_enabled,
        "evidence_selector": "posterior top-k",
        "metrics": {
            "supporting_paragraph_recall": avg_recall,
            "avg_belief_entropy": sum(row["belief_entropy"] for row in rows) / len(rows) if rows else 0.0,
        },
        "rows": rows,
    }
    write_json(report, args.json_out)
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
