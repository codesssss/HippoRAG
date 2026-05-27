#!/usr/bin/env python3
"""Build DAEC-DAPG Phase 0 diagnostics from baseline and DAEC-L1 predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.data import normalize_text
from src.dpathrag.daec_dapg.metrics import noise_rate, summarize_numeric_rows
from src.dpathrag.io import write_json, write_jsonl
from src.dpathrag.metrics import recall_at_k, support_complete_at_k


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def index_by_qid(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        qid = str(row.get("qid") or row.get("id") or row.get("query_idx") or idx)
        indexed[qid] = row
    return indexed


def extract_titles(row: dict[str, Any], *, explicit_keys: tuple[str, ...]) -> list[str]:
    for key in explicit_keys:
        value = row.get(key)
        if isinstance(value, list):
            return [str(item.get("title") if isinstance(item, dict) else item) for item in value]
    docs = row.get("selected_docs") or row.get("docs") or []
    titles: list[str] = []
    for doc in docs:
        if isinstance(doc, dict):
            titles.append(str(doc.get("title") or ""))
        else:
            titles.append(str(doc).split("\n", 1)[0])
    return titles


def extract_gold_titles(row: dict[str, Any]) -> list[str]:
    for key in ("gold_titles", "support_titles", "gold_support_titles"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def selected_gold_count(gold_titles: list[str], selected_titles: list[str]) -> int:
    gold = {normalize_text(title) for title in gold_titles if normalize_text(title)}
    selected = {normalize_text(title) for title in selected_titles if normalize_text(title)}
    return len(gold & selected)


def hop_bucket(row: dict[str, Any], gold_titles: list[str]) -> str:
    for key in ("hop_bucket", "hop", "num_hops", "support_count"):
        value = row.get(key)
        if value is not None:
            return str(value)
    count = len(gold_titles)
    return str(count) if count else "unknown"


def is_correct(row: dict[str, Any]) -> bool:
    return float(row.get("em") if row.get("em") is not None else row.get("answer_em") or 0.0) > 0.0


def support_complete(row: dict[str, Any], gold_titles: list[str], selected_titles: list[str]) -> float:
    for key in ("support_complete", "support_complete_at_k"):
        if row.get(key) is not None:
            return float(row.get(key) or 0.0)
    return support_complete_at_k(gold_titles, selected_titles, len(selected_titles))


def support_recall(row: dict[str, Any], gold_titles: list[str], selected_titles: list[str]) -> float:
    for key in ("support_recall", "support_recall_at_k"):
        if row.get(key) is not None:
            return float(row.get(key) or 0.0)
    return recall_at_k(gold_titles, selected_titles, len(selected_titles))


def failure_category(
    *,
    baseline_correct: bool,
    daec_correct: bool,
    baseline_complete: float,
    daec_complete: float,
) -> str:
    any_complete = max(float(baseline_complete), float(daec_complete)) >= 1.0
    if (not baseline_correct) and (not daec_correct) and not any_complete:
        return "retrieval_or_composition_bottleneck"
    if baseline_correct and not daec_correct and float(daec_complete) < float(baseline_complete):
        return "destructive_projection"
    if (not baseline_correct) and (not daec_correct) and any_complete:
        return "reader_bottleneck"
    if baseline_correct and daec_correct:
        return "no_headroom"
    if (not baseline_correct) and daec_correct:
        return "already_solved_by_daec_l1"
    return "mixed_or_unclear"


def build_rows(
    baseline_rows: list[dict[str, Any]],
    daec_rows: list[dict[str, Any]],
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    baseline_by_qid = index_by_qid(baseline_rows)
    daec_by_qid = index_by_qid(daec_rows)
    rows: list[dict[str, Any]] = []
    for qid, baseline in baseline_by_qid.items():
        daec = daec_by_qid.get(qid)
        if daec is None:
            continue
        gold_titles = extract_gold_titles(daec) or extract_gold_titles(baseline)
        baseline_titles = extract_titles(baseline, explicit_keys=("baseline_selected_titles", "selected_titles"))
        daec_titles = extract_titles(daec, explicit_keys=("daec_l1_selected_titles", "selected_titles"))
        baseline_sc = support_complete(baseline, gold_titles, baseline_titles)
        daec_sc = support_complete(daec, gold_titles, daec_titles)
        baseline_sr = support_recall(baseline, gold_titles, baseline_titles)
        daec_sr = support_recall(daec, gold_titles, daec_titles)
        baseline_ok = is_correct(baseline)
        daec_ok = is_correct(daec)
        daec_gold_count = selected_gold_count(gold_titles, daec_titles)
        category = failure_category(
            baseline_correct=baseline_ok,
            daec_correct=daec_ok,
            baseline_complete=baseline_sc,
            daec_complete=daec_sc,
        )
        rows.append(
            {
                "qid": qid,
                "dataset": dataset,
                "hop_bucket": hop_bucket(daec, gold_titles),
                "question": daec.get("question") or baseline.get("question"),
                "answer": daec.get("answer") or baseline.get("answer") or daec.get("gold_answers") or baseline.get("gold_answers"),
                "gold_titles": gold_titles,
                "baseline_selected_titles": baseline_titles,
                "daec_l1_selected_titles": daec_titles,
                "baseline_em": float(baseline.get("em") if baseline.get("em") is not None else baseline.get("answer_em") or 0.0),
                "baseline_f1": float(baseline.get("f1") if baseline.get("f1") is not None else baseline.get("answer_f1") or 0.0),
                "daec_l1_em": float(daec.get("em") if daec.get("em") is not None else daec.get("answer_em") or 0.0),
                "daec_l1_f1": float(daec.get("f1") if daec.get("f1") is not None else daec.get("answer_f1") or 0.0),
                "baseline_support_recall": baseline_sr,
                "baseline_support_complete": baseline_sc,
                "daec_l1_support_recall": daec_sr,
                "daec_l1_support_complete": daec_sc,
                "selected_gold_count": daec_gold_count,
                "selected_doc_count": len(daec_titles),
                "noise_rate": noise_rate(daec_gold_count, len(daec_titles)),
                "reader_wrong_despite_support_complete": int((not daec_ok) and daec_sc >= 1.0),
                "binding_count": int(daec.get("binding_count") or daec.get("bindings") or 0),
                "non_empty_binding": int(bool(daec.get("selected_binding") or daec.get("sel_binding"))),
                "selected_binding": daec.get("selected_binding") or daec.get("sel_binding") or {},
                "selected_coverage": float(daec.get("selected_coverage") or daec.get("coverage") or 0.0),
                "failure_category": category,
            }
        )
    return rows


def rows_from_daec_report(report: dict[str, Any], *, dataset: str) -> list[dict[str, Any]]:
    """Build Phase 0 rows directly from an eval_causal_qwen3 DAEC report JSON."""

    rows: list[dict[str, Any]] = []
    traces = list(report.get("setwise_selector_query_traces") or [])
    for idx, trace in enumerate(traces):
        gold_titles = [str(title) for title in trace.get("gold_titles") or []]
        baseline_titles = [str(title) for title in trace.get("baseline_top_titles") or []]
        daec_titles = [str(title) for title in trace.get("selector_top_titles") or []]
        baseline_metrics = trace.get("baseline_metrics") or {}
        selector_metrics = trace.get("selector_metrics") or {}
        selector_trace = trace.get("selector_trace") or {}
        baseline_em = float(baseline_metrics.get("ExactMatch") or baseline_metrics.get("em") or 0.0)
        baseline_f1 = float(baseline_metrics.get("F1") or baseline_metrics.get("f1") or 0.0)
        daec_em = float(selector_metrics.get("ExactMatch") or selector_metrics.get("em") or 0.0)
        daec_f1 = float(selector_metrics.get("F1") or selector_metrics.get("f1") or 0.0)
        baseline_sc = support_complete_at_k(gold_titles, baseline_titles, len(baseline_titles))
        daec_sc = support_complete_at_k(gold_titles, daec_titles, len(daec_titles))
        baseline_sr = recall_at_k(gold_titles, baseline_titles, len(baseline_titles))
        daec_sr = recall_at_k(gold_titles, daec_titles, len(daec_titles))
        daec_gold_count = selected_gold_count(gold_titles, daec_titles)
        category = failure_category(
            baseline_correct=baseline_em > 0.0,
            daec_correct=daec_em > 0.0,
            baseline_complete=baseline_sc,
            daec_complete=daec_sc,
        )
        rows.append(
            {
                "qid": str(trace.get("qid") or trace.get("query_idx") or idx),
                "dataset": dataset,
                "hop_bucket": str(trace.get("gold_doc_count") or len(gold_titles) or "unknown"),
                "question": trace.get("question"),
                "answer": trace.get("answer") or trace.get("gold_answers"),
                "gold_titles": gold_titles,
                "baseline_selected_titles": baseline_titles,
                "daec_l1_selected_titles": daec_titles,
                "baseline_em": baseline_em,
                "baseline_f1": baseline_f1,
                "daec_l1_em": daec_em,
                "daec_l1_f1": daec_f1,
                "baseline_support_recall": baseline_sr,
                "baseline_support_complete": baseline_sc,
                "daec_l1_support_recall": daec_sr,
                "daec_l1_support_complete": daec_sc,
                "selected_gold_count": daec_gold_count,
                "selected_doc_count": len(daec_titles),
                "noise_rate": noise_rate(daec_gold_count, len(daec_titles)),
                "reader_wrong_despite_support_complete": int(daec_em <= 0.0 and daec_sc >= 1.0),
                "binding_count": int(selector_trace.get("binding_count") or 0),
                "non_empty_binding": int(bool(selector_trace.get("selected_binding"))),
                "selected_binding": selector_trace.get("selected_binding") or {},
                "selected_coverage": float(selector_trace.get("objective") or selector_trace.get("covered_requirement_rate") or 0.0),
                "failure_category": category,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = Counter(str(row.get("failure_category")) for row in rows)
    by_hop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_hop[str(row.get("hop_bucket"))].append(row)
    metric_keys = [
        "baseline_em",
        "baseline_f1",
        "daec_l1_em",
        "daec_l1_f1",
        "baseline_support_recall",
        "baseline_support_complete",
        "daec_l1_support_recall",
        "daec_l1_support_complete",
        "noise_rate",
        "reader_wrong_despite_support_complete",
    ]
    return {
        "rows": len(rows),
        "overall": summarize_numeric_rows(rows, metric_keys),
        "failure_categories": dict(sorted(by_category.items())),
        "by_hop": {bucket: summarize_numeric_rows(bucket_rows, metric_keys) | {"rows": len(bucket_rows)} for bucket, bucket_rows in sorted(by_hop.items())},
    }


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# DAEC-DAPG Phase 0 Diagnostics",
        "",
        f"- Rows: `{summary['rows']}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary["overall"].items():
        lines.append(f"| {key} | {float(value):.4f} |")
    lines.extend(["", "## Failure Categories", "", "| Category | Count |", "|---|---:|"])
    for key, value in summary["failure_categories"].items():
        lines.append(f"| {key} | {int(value)} |")
    lines.extend(["", "## Hop Buckets", "", "| Hop | Rows | DAEC EM | DAEC F1 | DAEC Support Complete | Reader Wrong Despite Complete |", "|---|---:|---:|---:|---:|---:|"])
    for bucket, payload in summary["by_hop"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(bucket),
                    str(payload["rows"]),
                    f"{float(payload['daec_l1_em']):.4f}",
                    f"{float(payload['daec_l1_f1']):.4f}",
                    f"{float(payload['daec_l1_support_complete']):.4f}",
                    f"{float(payload['reader_wrong_despite_support_complete']):.4f}",
                ]
            )
            + " |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_predictions_jsonl", default="")
    parser.add_argument("--daec_predictions_jsonl", default="")
    parser.add_argument("--daec_report_json", default="")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_md", default="reports/dpathrag/daec_dapg_phase0_diagnostics_20260428.md")
    parser.add_argument("--output_json", default="reports/dpathrag/daec_dapg_phase0_diagnostics_20260428.json")
    parser.add_argument("--rows_jsonl", default="reports/dpathrag/daec_dapg_phase0_rows_20260428.jsonl")
    args = parser.parse_args()

    if args.daec_report_json:
        report = json.loads(Path(args.daec_report_json).read_text(encoding="utf-8"))
        rows = rows_from_daec_report(report, dataset=str(args.dataset))
        if int(args.limit) > 0:
            rows = rows[: int(args.limit)]
    else:
        if not args.baseline_predictions_jsonl or not args.daec_predictions_jsonl:
            raise ValueError("Provide either --daec_report_json or both prediction JSONL inputs.")
        baseline_rows = load_jsonl(args.baseline_predictions_jsonl, limit=int(args.limit))
        daec_rows = load_jsonl(args.daec_predictions_jsonl, limit=int(args.limit))
        rows = build_rows(baseline_rows, daec_rows, dataset=str(args.dataset))
    summary = summarize(rows)
    output = {
        "baseline_predictions_jsonl": str(args.baseline_predictions_jsonl),
        "daec_predictions_jsonl": str(args.daec_predictions_jsonl),
        "daec_report_json": str(args.daec_report_json),
        "dataset": str(args.dataset),
        "limit": int(args.limit),
        **summary,
    }
    write_jsonl(rows, args.rows_jsonl)
    write_json(output, args.output_json)
    write_markdown(output, args.output_md)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
