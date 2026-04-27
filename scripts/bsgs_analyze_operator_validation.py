#!/usr/bin/env python3
"""Analyze BSGS oracle-slot operator rows without running new retrieval."""

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


def recall_bin(value: float) -> str:
    if value >= 0.999:
        return "full_support_covered"
    if value <= 0.0:
        return "no_support_covered"
    return "partial_support_covered"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/week1/operator_validation.json")
    parser.add_argument("--json_out", default="reports/week1/operator_mechanism_analysis.json")
    parser.add_argument("--md_out", default="reports/week1/operator_mechanism_analysis.md")
    args = parser.parse_args()

    report = read_json(args.input)
    rows = list(report.get("rows") or [])
    bins: dict[str, list[dict[str, Any]]] = {
        "full_support_covered": [],
        "partial_support_covered": [],
        "no_support_covered": [],
    }
    duplicate_title_rates: list[float] = []
    entropies: list[float] = []
    recalls: list[float] = []
    for row in rows:
        recall = float(row.get("supporting_paragraph_recall") or 0.0)
        bins[recall_bin(recall)].append(row)
        recalls.append(recall)
        entropy = float(row.get("belief_entropy") or 0.0)
        entropies.append(entropy)
        titles = [title for title in (row.get("selected_titles") or []) if title]
        duplicate_title_rates.append(1.0 - (len(set(titles)) / len(titles)) if titles else 0.0)

    bin_summary: dict[str, dict[str, float]] = {}
    for name, items in bins.items():
        bin_summary[name] = {
            "count": float(len(items)),
            "fraction": len(items) / len(rows) if rows else 0.0,
            "avg_entropy": mean([float(item.get("belief_entropy") or 0.0) for item in items]),
            "avg_recall": mean([float(item.get("supporting_paragraph_recall") or 0.0) for item in items]),
        }

    failure_samples: list[dict[str, Any]] = []
    for name in ["no_support_covered", "partial_support_covered"]:
        for row in bins[name][:8]:
            failure_samples.append(
                {
                    "category": name,
                    "qid": row.get("qid"),
                    "question": row.get("question"),
                    "selected_titles": row.get("selected_titles"),
                    "gold_titles": row.get("gold_titles"),
                    "supporting_paragraph_recall": row.get("supporting_paragraph_recall"),
                    "belief_entropy": row.get("belief_entropy"),
                    "last_step_top_props": ((row.get("trace") or [{}])[-1].get("top_props") if row.get("trace") else []),
                }
            )

    payload = {
        "status": "completed" if rows else "failed",
        "n": len(rows),
        "overall": {
            "supporting_paragraph_recall": mean(recalls),
            "avg_belief_entropy": mean(entropies),
            "avg_duplicate_title_rate": mean(duplicate_title_rates),
        },
        "support_coverage_bins": bin_summary,
        "failure_samples": failure_samples,
    }
    write_json(payload, args.json_out)

    out = Path(args.md_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BSGS Oracle-Slot Mechanism Analysis",
        "",
        f"- Status: `{payload['status']}`",
        f"- Examples: `{payload['n']}`",
        "",
        "## Overall",
    ]
    for key, value in payload["overall"].items():
        lines.append(f"- {key}: `{value:.4f}`")
    lines.extend(["", "## Support Coverage Bins", "", "| Bin | Count | Fraction | Avg Recall | Avg Entropy |", "|---|---:|---:|---:|---:|"])
    for name, stats in bin_summary.items():
        lines.append(
            f"| {name} | `{int(stats['count'])}` | `{stats['fraction']:.4f}` | `{stats['avg_recall']:.4f}` | `{stats['avg_entropy']:.4f}` |"
        )
    lines.extend(["", "## Failure Samples", ""])
    for sample in failure_samples[:10]:
        lines.append(
            f"- `{sample['category']}` `{sample['qid']}` recall=`{sample['supporting_paragraph_recall']}` "
            f"selected={sample['selected_titles']} gold={sample['gold_titles']}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
