#!/usr/bin/env python3
"""Diagnose where DAEC-L1 fixed-pool QA gains come from."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.data import normalize_text
from src.dpathrag.daec_dapg.metrics import noise_rate, summarize_numeric_rows
from src.dpathrag.io import read_json, write_json, write_jsonl
from src.dpathrag.metrics import recall_at_k, support_complete_at_k


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalized_set(values: list[Any]) -> set[str]:
    return {normalize_text(value) for value in values if normalize_text(value)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def overlap_stats(before: list[str], after: list[str], gold: list[str]) -> dict[str, Any]:
    before_set = normalized_set(before)
    after_set = normalized_set(after)
    gold_set = normalized_set(gold)
    added = after_set - before_set
    removed = before_set - after_set
    added_gold = added & gold_set
    removed_gold = removed & gold_set
    added_non_gold = added - gold_set
    removed_non_gold = removed - gold_set
    return {
        "title_jaccard": jaccard(before_set, after_set),
        "same_title_set": float(before_set == after_set),
        "added_title_count": len(added),
        "removed_title_count": len(removed),
        "added_gold_count": len(added_gold),
        "removed_gold_count": len(removed_gold),
        "added_non_gold_count": len(added_non_gold),
        "removed_non_gold_count": len(removed_non_gold),
        "noise_delta": noise_rate(len(after_set & gold_set), len(after_set)) - noise_rate(len(before_set & gold_set), len(before_set)),
        "support_recall_delta": recall_at_k(gold, after, len(after)) - recall_at_k(gold, before, len(before)),
        "support_complete_delta": support_complete_at_k(gold, after, len(after)) - support_complete_at_k(gold, before, len(before)),
    }


def answer_transition(base_em: float, selector_em: float) -> str:
    base_correct = base_em >= 0.5
    selector_correct = selector_em >= 0.5
    if not base_correct and selector_correct:
        return "wrong_to_correct"
    if base_correct and not selector_correct:
        return "correct_to_wrong"
    if base_correct and selector_correct:
        return "both_correct"
    return "both_wrong"


def f1_bucket(delta: float) -> str:
    if delta > 1e-9:
        return "f1_improved"
    if delta < -1e-9:
        return "f1_regressed"
    return "f1_unchanged"


def build_rows(report: dict[str, Any], *, dataset: str, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, trace in enumerate(report.get("setwise_selector_query_traces") or []):
        gold_titles = list(trace.get("gold_titles") or [])
        baseline_titles = list(trace.get("baseline_top_titles") or [])
        selector_titles = list(trace.get("selector_top_titles") or [])
        baseline_docs = [str(value) for value in trace.get("baseline_top_doc_ids") or []]
        selector_docs = [str(value) for value in trace.get("selector_top_doc_ids") or []]
        baseline_metrics = trace.get("baseline_metrics") or {}
        selector_metrics = trace.get("selector_metrics") or {}
        baseline_em = as_float(baseline_metrics.get("ExactMatch"))
        selector_em = as_float(selector_metrics.get("ExactMatch"))
        baseline_f1 = as_float(baseline_metrics.get("F1"))
        selector_f1 = as_float(selector_metrics.get("F1"))
        title_stats = overlap_stats(baseline_titles, selector_titles, gold_titles)
        doc_jaccard = jaccard(set(baseline_docs), set(selector_docs))
        row = {
            "qid": str(trace.get("qid") or trace.get("question") or idx),
            "query_idx": idx,
            "dataset": dataset,
            "source_name": source_name,
            "question": trace.get("question"),
            "gold_doc_count": int(trace.get("gold_doc_count") or len(gold_titles)),
            "changed_from_baseline": float(bool(trace.get("changed_from_baseline"))),
            "same_doc_set": float(set(baseline_docs) == set(selector_docs)),
            "doc_jaccard": doc_jaccard,
            "baseline_em": baseline_em,
            "selector_em": selector_em,
            "delta_em": selector_em - baseline_em,
            "baseline_f1": baseline_f1,
            "selector_f1": selector_f1,
            "delta_f1": selector_f1 - baseline_f1,
            "answer_transition": answer_transition(baseline_em, selector_em),
            "f1_bucket": f1_bucket(selector_f1 - baseline_f1),
            "baseline_support_recall": recall_at_k(gold_titles, baseline_titles, len(baseline_titles)),
            "selector_support_recall": recall_at_k(gold_titles, selector_titles, len(selector_titles)),
            "baseline_support_complete": support_complete_at_k(gold_titles, baseline_titles, len(baseline_titles)),
            "selector_support_complete": support_complete_at_k(gold_titles, selector_titles, len(selector_titles)),
            "baseline_noise_rate": noise_rate(len(normalized_set(baseline_titles) & normalized_set(gold_titles)), len(normalized_set(baseline_titles))),
            "selector_noise_rate": noise_rate(len(normalized_set(selector_titles) & normalized_set(gold_titles)), len(normalized_set(selector_titles))),
            "baseline_duplicate_count": int(trace.get("baseline_title_duplicate_count") or 0),
            "selector_duplicate_count": int(trace.get("selector_title_duplicate_count") or 0),
            "duplicate_delta": int(trace.get("selector_title_duplicate_count") or 0) - int(trace.get("baseline_title_duplicate_count") or 0),
            "baseline_answer": trace.get("baseline_answer"),
            "selector_answer": trace.get("selector_answer"),
            "baseline_top_titles": baseline_titles,
            "selector_top_titles": selector_titles,
        }
        row.update(title_stats)
        rows.append(row)
    return rows


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    metrics = [
        "baseline_em",
        "selector_em",
        "delta_em",
        "baseline_f1",
        "selector_f1",
        "delta_f1",
        "changed_from_baseline",
        "title_jaccard",
        "doc_jaccard",
        "same_title_set",
        "same_doc_set",
        "support_recall_delta",
        "support_complete_delta",
        "noise_delta",
        "added_gold_count",
        "removed_gold_count",
        "added_non_gold_count",
        "removed_non_gold_count",
        "duplicate_delta",
    ]
    return {name: summarize_numeric_rows(items, metrics) | {"rows": len(items)} for name, items in sorted(grouped.items())}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "baseline_em",
        "selector_em",
        "delta_em",
        "baseline_f1",
        "selector_f1",
        "delta_f1",
        "changed_from_baseline",
        "title_jaccard",
        "doc_jaccard",
        "same_title_set",
        "same_doc_set",
        "support_recall_delta",
        "support_complete_delta",
        "noise_delta",
        "duplicate_delta",
    ]
    f1_improved = [row for row in rows if row["delta_f1"] > 1e-9]
    f1_regressed = [row for row in rows if row["delta_f1"] < -1e-9]
    unchanged_docs = [row for row in rows if row["same_doc_set"] >= 0.5]
    changed_docs = [row for row in rows if row["same_doc_set"] < 0.5]
    return {
        "rows": len(rows),
        "overall": summarize_numeric_rows(rows, metrics) | {"rows": len(rows)},
        "answer_transitions": dict(Counter(row["answer_transition"] for row in rows)),
        "f1_buckets": dict(Counter(row["f1_bucket"] for row in rows)),
        "by_answer_transition": summarize_by(rows, "answer_transition"),
        "by_f1_bucket": summarize_by(rows, "f1_bucket"),
        "changed_doc_set": summarize_numeric_rows(changed_docs, metrics) | {"rows": len(changed_docs)},
        "unchanged_doc_set": summarize_numeric_rows(unchanged_docs, metrics) | {"rows": len(unchanged_docs)},
        "improved_f1": summarize_numeric_rows(f1_improved, metrics) | {"rows": len(f1_improved)},
        "regressed_f1": summarize_numeric_rows(f1_regressed, metrics) | {"rows": len(f1_regressed)},
    }


def example_rows(rows: list[dict[str, Any]], *, limit: int) -> dict[str, list[dict[str, Any]]]:
    fields = [
        "query_idx",
        "question",
        "baseline_answer",
        "selector_answer",
        "delta_f1",
        "delta_em",
        "support_recall_delta",
        "noise_delta",
        "added_gold_count",
        "removed_gold_count",
        "added_non_gold_count",
        "removed_non_gold_count",
        "baseline_top_titles",
        "selector_top_titles",
    ]

    def trim(row: dict[str, Any]) -> dict[str, Any]:
        return {field: row.get(field) for field in fields}

    return {
        "largest_f1_gains": [trim(row) for row in sorted(rows, key=lambda row: (-row["delta_f1"], row["query_idx"]))[:limit]],
        "largest_f1_losses": [trim(row) for row in sorted(rows, key=lambda row: (row["delta_f1"], row["query_idx"]))[:limit]],
        "wrong_to_correct": [trim(row) for row in rows if row["answer_transition"] == "wrong_to_correct"][:limit],
        "correct_to_wrong": [trim(row) for row in rows if row["answer_transition"] == "correct_to_wrong"][:limit],
    }


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    summary = payload["summary"]
    overall = summary["overall"]
    lines = [
        "# DAEC-L1 Gain Diagnostics",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Source: `{payload['source_name']}`",
        f"- Rows: `{summary['rows']}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline EM | {overall['baseline_em']:.4f} |",
        f"| Selector EM | {overall['selector_em']:.4f} |",
        f"| Delta EM | {overall['delta_em']:+.4f} |",
        f"| Baseline F1 | {overall['baseline_f1']:.4f} |",
        f"| Selector F1 | {overall['selector_f1']:.4f} |",
        f"| Delta F1 | {overall['delta_f1']:+.4f} |",
        f"| Changed-from-baseline rate | {overall['changed_from_baseline']:.4f} |",
        f"| Same title set rate | {overall['same_title_set']:.4f} |",
        f"| Same doc set rate | {overall['same_doc_set']:.4f} |",
        f"| Avg title Jaccard | {overall['title_jaccard']:.4f} |",
        f"| Avg doc Jaccard | {overall['doc_jaccard']:.4f} |",
        f"| Support recall delta | {overall['support_recall_delta']:+.4f} |",
        f"| Support complete delta | {overall['support_complete_delta']:+.4f} |",
        f"| Noise delta | {overall['noise_delta']:+.4f} |",
        f"| Duplicate delta | {overall['duplicate_delta']:+.4f} |",
        "",
        "## Answer Transitions",
        "",
        "| Transition | Rows | Delta F1 | Support Recall Δ | Noise Δ | Title Jaccard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary["by_answer_transition"].items():
        lines.append(
            f"| {name} | {value['rows']} | {value['delta_f1']:+.4f} | {value['support_recall_delta']:+.4f} | "
            f"{value['noise_delta']:+.4f} | {value['title_jaccard']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## F1 Buckets",
            "",
            "| Bucket | Rows | Delta F1 | Support Recall Δ | Noise Δ | Added Gold | Removed Gold | Removed Non-Gold |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in summary["by_f1_bucket"].items():
        lines.append(
            f"| {name} | {value['rows']} | {value['delta_f1']:+.4f} | {value['support_recall_delta']:+.4f} | "
            f"{value['noise_delta']:+.4f} | {value['added_gold_count']:.4f} | {value['removed_gold_count']:.4f} | "
            f"{value['removed_non_gold_count']:.4f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--source_name", default="")
    parser.add_argument("--output_json", default="reports/dpathrag/daec_l1_gain_diagnostics.json")
    parser.add_argument("--rows_jsonl", default="reports/dpathrag/daec_l1_gain_diagnostics_rows.jsonl")
    parser.add_argument("--output_md", default="reports/dpathrag/daec_l1_gain_diagnostics.md")
    parser.add_argument("--example_limit", type=int, default=10)
    args = parser.parse_args()

    report = read_json(args.report_json)
    dataset = str(args.dataset or report.get("dataset") or "unknown")
    source_name = str(args.source_name or ((report.get("external_pool") or {}).get("source_name")) or args.report_json)
    rows = build_rows(report, dataset=dataset, source_name=source_name)
    payload = {
        "report_json": str(args.report_json),
        "dataset": dataset,
        "source_name": source_name,
        "summary": summarize(rows),
        "examples": example_rows(rows, limit=int(args.example_limit)),
    }
    write_jsonl(rows, args.rows_jsonl)
    write_json(payload, args.output_json)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
