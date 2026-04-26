#!/usr/bin/env python3
"""Analyze reader failures against selected D-PathRAG evidence contexts."""

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

from src.dpathrag.io import write_json
from src.dpathrag.reader import gold_answers, normalize_answer


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= int(limit):
                    break
    return rows


def selected_context(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for doc in record.get("selected_docs") or []:
        parts.append(str(doc.get("title") or ""))
        parts.append(str(doc.get("text") or ""))
    return normalize_answer(" ".join(parts))


def answer_in_context(record: dict[str, Any]) -> bool:
    context = selected_context(record)
    return any(normalize_answer(answer) and normalize_answer(answer) in context for answer in gold_answers(record))


def summarize_flags(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "answer_in_context_rate": 0.0,
            "answer_failure_rate": 0.0,
            "answer_in_context_but_fail_rate": 0.0,
            "support_complete_but_fail_rate": 0.0,
            "incomplete_but_answer_in_context_rate": 0.0,
            "avg_f1_answer_in_context": 0.0,
            "avg_f1_answer_absent": 0.0,
        }
    denom = float(len(rows))
    answer_in_ctx = [row for row in rows if bool(row["answer_in_context"])]
    answer_absent = [row for row in rows if not bool(row["answer_in_context"])]
    failures = [row for row in rows if float(row["f1"]) < 1.0]
    answer_in_ctx_fail = [row for row in rows if bool(row["answer_in_context"]) and float(row["f1"]) < 1.0]
    complete_fail = [row for row in rows if float(row["support_complete"]) >= 1.0 and float(row["f1"]) < 1.0]
    incomplete_answer_in_ctx = [
        row for row in rows if float(row["support_complete"]) < 1.0 and bool(row["answer_in_context"])
    ]
    return {
        "rows": len(rows),
        "answer_in_context_rate": round(len(answer_in_ctx) / denom, 4),
        "answer_failure_rate": round(len(failures) / denom, 4),
        "answer_in_context_but_fail_rate": round(len(answer_in_ctx_fail) / denom, 4),
        "support_complete_but_fail_rate": round(len(complete_fail) / denom, 4),
        "incomplete_but_answer_in_context_rate": round(len(incomplete_answer_in_ctx) / denom, 4),
        "avg_f1_answer_in_context": round(
            sum(float(row["f1"]) for row in answer_in_ctx) / max(1, len(answer_in_ctx)),
            4,
        ),
        "avg_f1_answer_absent": round(
            sum(float(row["f1"]) for row in answer_absent) / max(1, len(answer_absent)),
            4,
        ),
    }


def analyze(reader_rows: Sequence[dict[str, Any]], prediction_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    predictions_by_qid = {str(row.get("qid")): row for row in prediction_rows}
    rows: list[dict[str, Any]] = []
    for record in reader_rows:
        qid = str(record.get("qid"))
        if qid not in predictions_by_qid:
            raise KeyError(f"Missing prediction for qid={qid}")
        prediction = predictions_by_qid[qid]
        rows.append(
            {
                "qid": qid,
                "source": record.get("source"),
                "answer_in_context": answer_in_context(record),
                "support_recall": float(record.get("support_recall") or 0.0),
                "support_complete": float(record.get("support_complete") or 0.0),
                "em": float(prediction.get("em") or 0.0),
                "f1": float(prediction.get("f1") or 0.0),
                "prediction": prediction.get("prediction"),
                "gold_answers": prediction.get("gold_answers") or gold_answers(record),
            }
        )
    return {
        "summary": summarize_flags(rows),
        "rows": rows,
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    summary = report["summary"]
    lines = [
        "# D-PathRAG Reader Failure Analysis",
        "",
        f"- Input: `{report['input_jsonl']}`",
        f"- Predictions: `{report['predictions_jsonl']}`",
        f"- Rows: {summary['rows']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "answer_in_context_rate",
        "answer_failure_rate",
        "answer_in_context_but_fail_rate",
        "support_complete_but_fail_rate",
        "incomplete_but_answer_in_context_rate",
        "avg_f1_answer_in_context",
        "avg_f1_answer_absent",
    ]:
        lines.append(f"| {key} | {summary[key]:.4f} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include_rows", action="store_true")
    args = parser.parse_args()

    reader_rows = load_jsonl(args.input_jsonl, limit=int(args.limit))
    prediction_rows = load_jsonl(args.predictions_jsonl, limit=int(args.limit))
    result = analyze(reader_rows, prediction_rows)
    result["input_jsonl"] = str(args.input_jsonl)
    result["predictions_jsonl"] = str(args.predictions_jsonl)
    result["limit"] = int(args.limit)
    if not bool(args.include_rows):
        result.pop("rows", None)
    write_json(result, args.output_json)
    if args.output_md:
        write_markdown(result, args.output_md)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
