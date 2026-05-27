#!/usr/bin/env python3
"""Summarize aligned PathRAG retrieval, reader, and token usage outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "nq_rear", "popqa"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def usage_tokens(summary: Mapping[str, Any], phase: str, kind: str = "chat_completion") -> int:
    usage = summary.get("usage", {}) if isinstance(summary, Mapping) else {}
    phase_row = (usage.get("by_phase", {}) or {}).get(phase, {}) if isinstance(usage, Mapping) else {}
    kind_row = (phase_row.get("by_kind", {}) or {}).get(kind, {}) if isinstance(phase_row, Mapping) else {}
    return int(kind_row.get("total_tokens", 0) or 0)


def phase_calls(summary: Mapping[str, Any], phase: str, key: str) -> int:
    usage = summary.get("usage", {}) if isinstance(summary, Mapping) else {}
    phase_row = (usage.get("by_phase", {}) or {}).get(phase, {}) if isinstance(usage, Mapping) else {}
    return int(phase_row.get(key, 0) or 0)


def reader_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    datasets = payload.get("datasets", []) or []
    if not datasets:
        return None
    methods = datasets[0].get("methods", {}) or {}
    method = methods.get("pathrag") or next(iter(methods.values()), None)
    if not isinstance(method, Mapping):
        return None
    metrics = method.get("metrics", {}) or {}
    return {
        "em": float(metrics.get("ExactMatch", 0.0) or 0.0),
        "f1": float(metrics.get("F1", 0.0) or 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        summary_path = run_root / "summaries" / f"{dataset}_pathrag_summary.json"
        reader_path = run_root / "reader_outputs" / f"{dataset}_pathrag_gpt4omini_top5_limit1000.json"
        if not summary_path.exists():
            rows.append({"dataset": dataset, "status": "missing"})
            continue
        summary = load_json(summary_path)
        retrieval = ((summary.get("retrieval", {}) or {}).get("recomputed_title_recall", {}) or {})
        reader = reader_metrics(reader_path)
        row = {
            "dataset": dataset,
            "status": "done" if reader is not None else "retrieval_done",
            "r5": float(retrieval.get("Recall@5", 0.0) or 0.0),
            "all5": float(retrieval.get("All@5", 0.0) or 0.0),
            "em": None if reader is None else reader["em"],
            "f1": None if reader is None else reader["f1"],
            "offline_chat_tokens": usage_tokens(summary, "offline_index"),
            "online_chat_tokens": usage_tokens(summary, "online_retrieval"),
            "offline_chat_calls": phase_calls(summary, "offline_index", "chat_calls"),
            "online_chat_calls": phase_calls(summary, "online_retrieval", "chat_calls"),
        }
        rows.append(row)

    output_json = run_root / "summaries" / "pathrag_aligned_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_md = run_root / "summaries" / "pathrag_aligned_summary.md"
    lines = [
        "| Dataset | Status | R@5 | All@5 | EM | F1 | Offline Chat Tok. | Online Chat Tok. | Offline Calls | Online Calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        em_text = "" if row.get("em") is None else f"{float(row['em']):.4f}"
        f1_text = "" if row.get("f1") is None else f"{float(row['f1']):.4f}"
        lines.append(
            f"| {row['dataset']} | {row['status']} | "
            f"{row.get('r5', 0.0):.4f} | {row.get('all5', 0.0):.4f} | "
            f"{em_text} | "
            f"{f1_text} | "
            f"{int(row.get('offline_chat_tokens', 0) or 0)} | "
            f"{int(row.get('online_chat_tokens', 0) or 0)} | "
            f"{int(row.get('offline_chat_calls', 0) or 0)} | "
            f"{int(row.get('online_chat_calls', 0) or 0)} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
