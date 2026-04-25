#!/usr/bin/env python3
"""Build oracle slots from local MuSiQue gold decomposition."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.io import load_dataset, write_json, write_jsonl
from src.bsgs.slots import sample_to_oracle_slot_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jsonl_out", default="data/processed/musique_oracle_slots.jsonl")
    parser.add_argument("--report_md", default="reports/week0/oracle_slot_pipeline.md")
    parser.add_argument("--report_json", default="reports/week0/oracle_slot_pipeline.json")
    args = parser.parse_args()

    _, samples = load_dataset(args.dataset, dataset_dir=args.dataset_dir)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    rows = [sample_to_oracle_slot_record(sample) for sample in samples]
    rows = [row for row in rows if row["slots"]]
    write_jsonl(rows, args.jsonl_out)

    slot_counts = [len(row["slots"]) for row in rows]
    report = {
        "status": "completed",
        "dataset": args.dataset,
        "num_examples": len(rows),
        "output": args.jsonl_out,
        "avg_slots": sum(slot_counts) / len(slot_counts) if slot_counts else 0.0,
        "min_slots": min(slot_counts) if slot_counts else 0,
        "max_slots": max(slot_counts) if slot_counts else 0,
        "contamination_warning": "Oracle slots are only for BSGS-oracle-slot diagnostics; do not tune latent-slot prompts on the same split.",
    }
    write_json(report, args.report_json)

    lines = [
        "# Oracle Slot Pipeline",
        "",
        f"- Status: `{report['status']}`",
        f"- Dataset: `{args.dataset}`",
        f"- Built examples: `{report['num_examples']}`",
        f"- Output: `{args.jsonl_out}`",
        f"- Avg slots: `{report['avg_slots']:.3f}`",
        f"- Min / max slots: `{report['min_slots']}` / `{report['max_slots']}`",
        "",
        "Oracle slots are upper-bound diagnostics only and must not be presented as latent-slot main results.",
    ]
    out = Path(args.report_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.jsonl_out}, {args.report_json}, and {args.report_md}")


if __name__ == "__main__":
    main()
