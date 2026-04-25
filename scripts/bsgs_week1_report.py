#!/usr/bin/env python3
"""Create Week-1 operator validation report from BSGS oracle output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.io import read_json, write_json


def safe_read(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "missing", "path": path}
    return read_json(p)


def evaluate_gates(bsgs: dict[str, Any], daec: dict[str, Any], ircot: dict[str, Any]) -> dict[str, Any]:
    metrics = bsgs.get("metrics") or {}
    bsgs_recall = float(metrics.get("supporting_paragraph_recall") or 0.0)
    daec_recall = float((daec.get("metrics") or {}).get("supporting_paragraph_recall") or 0.0)
    ircot_recall = float((ircot.get("metrics") or {}).get("supporting_paragraph_recall") or 0.0)
    best_base = max(daec_recall, ircot_recall)
    mechanism_delta = bsgs_recall - best_base
    mechanism_passed = mechanism_delta >= 0.05
    return {
        "mechanism_passed": mechanism_passed,
        "mechanism_metric": "supporting_paragraph_recall",
        "mechanism_delta": mechanism_delta,
        "answer_grade": "Yellow" if mechanism_passed else "Red",
        "decision": "continue_bsgs_diagnostics" if mechanism_passed else "fallback_to_daec",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsgs_json", default="reports/week1/operator_validation.json")
    parser.add_argument("--daec_json", default="")
    parser.add_argument("--ircot_json", default="reports/week0/ircot_musique200.json")
    parser.add_argument("--json_out", default="reports/week1/operator_validation_summary.json")
    parser.add_argument("--md_out", default="reports/week1/operator_validation.md")
    args = parser.parse_args()

    bsgs = safe_read(args.bsgs_json)
    daec = safe_read(args.daec_json) if args.daec_json else {"status": "missing", "metrics": {}}
    ircot = safe_read(args.ircot_json)
    gates = evaluate_gates(bsgs, daec, ircot)
    payload = {"bsgs": bsgs, "daec": daec, "ircot": ircot, "gates": gates}
    write_json(payload, args.json_out)

    metrics = bsgs.get("metrics") or {}
    lines = [
        "# Week 1 Oracle-Slot BSGS Operator Validation",
        "",
        "## Config",
        f"- dataset: `{bsgs.get('dataset', '')}`",
        f"- split: `local MuSiQue diagnostic`",
        f"- n: `{bsgs.get('n', '')}`",
        f"- verifier: `{bsgs.get('verifier', '')}`",
        f"- absorbing enabled: `{bsgs.get('absorbing_enabled', '')}`",
        f"- binding: `{bsgs.get('binding', '')}`",
        f"- evidence selector: `{bsgs.get('evidence_selector', '')}`",
        "",
        "## Main comparison",
        "| Method | EM | F1 | Bridge Recall | Evidence Path Recall | AICBF | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| DAEC |  |  |  |  |  |  |",
        "| IRCoT |  |  |  |  |  |  |",
        f"| BSGS-oracle-slot |  |  |  | `{metrics.get('supporting_paragraph_recall', 0.0):.4f}` |  |  |",
        "",
        "## Mechanism gate",
        f"- Passed: `{gates['mechanism_passed']}`",
        f"- Metric: `{gates['mechanism_metric']}`",
        f"- Delta: `{gates['mechanism_delta']:.4f}`",
        "",
        "## Answer gate",
        f"- Grade: `{gates['answer_grade']}`",
        "",
        "## Failure taxonomy",
        "- bridge missing: pending manual analysis",
        "- proposition extraction failure: pending manual analysis",
        "- verifier false support: pending manual analysis",
        "- reader failure: pending manual analysis",
        "- entity binding error: pending manual analysis",
        "",
        "## Decision",
        f"- `{gates['decision']}`",
    ]
    out = Path(args.md_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
