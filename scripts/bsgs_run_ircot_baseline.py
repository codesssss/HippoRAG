#!/usr/bin/env python3
"""IRCoT-style baseline scaffold for BSGS Week 0.

This script intentionally refuses to fabricate IRCoT numbers.  If no runnable
LLM/retriever setup or replay input is provided, it writes a `not_run` report
instead of a heuristic result.
"""

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


def replay_existing_report(path: str) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "status": "replayed",
        "source_report": path,
        "dataset": payload.get("dataset"),
        "answer_em": payload.get("qa", {}).get("ExactMatch") or payload.get("ExactMatch"),
        "answer_f1": payload.get("qa", {}).get("F1") or payload.get("F1"),
        "retrieval": payload.get("retrieval") or {},
        "note": "This is a replayed report, not a newly executed IRCoT run.",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# IRCoT Baseline",
        "",
        f"- Status: `{report['status']}`",
        f"- Dataset: `{report.get('dataset', 'musique')}`",
        f"- Limit: `{report.get('limit', '')}`",
        f"- Max iter: `{report.get('max_iter', '')}`",
        f"- Top-k per iter: `{report.get('top_k_per_iter', '')}`",
        f"- Answer EM: `{report.get('answer_em', '')}`",
        f"- Answer F1: `{report.get('answer_f1', '')}`",
        f"- LLM calls/query: `{report.get('llm_calls_per_query', '')}`",
        f"- Latency/query: `{report.get('latency_per_query', '')}`",
    ]
    if report.get("reason"):
        lines.extend(["", f"Reason: {report['reason']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--top_k_per_iter", type=int, default=5)
    parser.add_argument("--llm_base_url", default="")
    parser.add_argument("--llm_name", default="qwen3-8b")
    parser.add_argument("--replay_report", default="")
    parser.add_argument("--report_json", default="reports/week0/ircot_musique50.json")
    parser.add_argument("--report_md", default="reports/week0/ircot_baseline.md")
    args = parser.parse_args()

    if args.replay_report:
        report = replay_existing_report(args.replay_report)
    elif not args.llm_base_url:
        report = {
            "status": "not_run",
            "dataset": args.dataset,
            "limit": args.limit,
            "max_iter": args.max_iter,
            "top_k_per_iter": args.top_k_per_iter,
            "llm_name": args.llm_name,
            "reason": "No --llm_base_url or --replay_report provided. This script will not fabricate IRCoT results.",
        }
    else:
        report = {
            "status": "not_implemented_runtime",
            "dataset": args.dataset,
            "limit": args.limit,
            "max_iter": args.max_iter,
            "top_k_per_iter": args.top_k_per_iter,
            "llm_name": args.llm_name,
            "llm_base_url": args.llm_base_url,
            "reason": "IRCoT runtime adapter is not wired yet; use this config to launch the actual iterative run next.",
        }
    write_json(report, args.report_json)
    write_markdown(report, Path(args.report_md))
    print(f"Wrote {args.report_json} and {args.report_md}")


if __name__ == "__main__":
    main()
