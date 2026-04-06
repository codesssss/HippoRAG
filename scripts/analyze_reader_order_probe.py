#!/usr/bin/env python3
"""Analyze paired reader-order probe reports.

This compares a frozen selector control report against a probe report that only
changes final reader context order. The key invariant is that the selected
evidence set must stay identical; if it changes, the comparison is not a clean
ordering probe.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


EPS = 1e-9


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(value: str) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def compute_recall_at_k(predicted_titles: Iterable[str], gold_titles: Iterable[str], k: int) -> float:
    predicted = set(unique_preserve_order(list(predicted_titles)[: max(int(k), 0)]))
    gold = set(unique_preserve_order(gold_titles))
    if not gold:
        return 0.0
    return len(predicted & gold) / len(gold)


def classify_delta(delta: float) -> str:
    if delta > EPS:
        return "improved"
    if delta < -EPS:
        return "worsened"
    return "unchanged"


def build_pair_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(trace.get("question", "")): trace
        for trace in list(report.get("setwise_selector_query_traces", []) or [])
    }


def _selector_trace_titles(trace: Dict[str, Any], key: str) -> List[str]:
    selector_trace = dict(trace.get("selector_trace", {}) or {})
    return list(selector_trace.get(key, []) or [])


def build_query_record(control_trace: Dict[str, Any], probe_trace: Dict[str, Any]) -> Dict[str, Any]:
    question = str(control_trace.get("question", ""))
    gold_titles = list(control_trace.get("gold_titles", []) or probe_trace.get("gold_titles", []) or [])
    control_selector_trace = dict(control_trace.get("selector_trace", {}) or {})
    probe_selector_trace = dict(probe_trace.get("selector_trace", {}) or {})
    probe_metadata = dict(probe_selector_trace.get("reader_order_probe", {}) or {})

    control_selected_titles = _selector_trace_titles(control_trace, "selected_titles")
    probe_selected_titles = _selector_trace_titles(probe_trace, "selected_titles")
    control_front_titles = _selector_trace_titles(control_trace, "final_front_titles") or list(control_trace.get("selector_top_titles", []) or [])
    probe_front_titles = _selector_trace_titles(probe_trace, "final_front_titles") or list(probe_trace.get("selector_top_titles", []) or [])

    selected_set_unchanged = unique_preserve_order(control_selected_titles) == unique_preserve_order(probe_selected_titles)
    qa_f1_control = float(dict(control_trace.get("selector_metrics", {}) or {}).get("F1", 0.0) or 0.0)
    qa_f1_probe = float(dict(probe_trace.get("selector_metrics", {}) or {}).get("F1", 0.0) or 0.0)
    qa_em_control = float(dict(control_trace.get("selector_metrics", {}) or {}).get("ExactMatch", 0.0) or 0.0)
    qa_em_probe = float(dict(probe_trace.get("selector_metrics", {}) or {}).get("ExactMatch", 0.0) or 0.0)

    top3_recall_control = compute_recall_at_k(control_front_titles, gold_titles, k=3)
    top3_recall_probe = compute_recall_at_k(probe_front_titles, gold_titles, k=3)
    qa_f1_delta = qa_f1_probe - qa_f1_control
    qa_em_delta = qa_em_probe - qa_em_control
    qa_bucket = classify_delta(qa_f1_delta)
    probe_applied = bool(probe_metadata.get("applied", False))
    if not selected_set_unchanged:
        bucket = "invalid_selected_set_changed"
    elif not probe_applied:
        bucket = "probe_not_applied"
    else:
        bucket = f"probe_applied__qa_{qa_bucket}"

    return {
        "question": question,
        "query_type": str(control_trace.get("query_type", probe_trace.get("query_type", "unknown"))),
        "gold_titles": gold_titles,
        "control_answer": str(control_trace.get("selector_answer", "")),
        "probe_answer": str(probe_trace.get("selector_answer", "")),
        "control_selected_titles": control_selected_titles,
        "probe_selected_titles": probe_selected_titles,
        "control_front_titles": control_front_titles,
        "probe_front_titles": probe_front_titles,
        "selected_set_unchanged": bool(selected_set_unchanged),
        "probe_applied": probe_applied,
        "probe_mode": str(probe_metadata.get("mode", "")),
        "promoted_title": str(probe_metadata.get("promoted_title", "")),
        "promoted_from_rank": probe_metadata.get("promoted_from_rank"),
        "qa_f1_control": qa_f1_control,
        "qa_f1_probe": qa_f1_probe,
        "qa_f1_delta": qa_f1_delta,
        "qa_em_control": qa_em_control,
        "qa_em_probe": qa_em_probe,
        "qa_em_delta": qa_em_delta,
        "gold_top3_coverage_control": top3_recall_control,
        "gold_top3_coverage_probe": top3_recall_probe,
        "gold_top3_coverage_delta": top3_recall_probe - top3_recall_control,
        "bucket": bucket,
    }


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[str(record.get("bucket", "unknown"))] += 1
    return dict(sorted(counts.items()))


def summarize_applied(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    applied = [record for record in records if record.get("bucket", "").startswith("probe_applied__")]
    promoted_hist: Counter[str] = Counter()
    for record in applied:
        promoted_hist[str(record.get("promoted_from_rank"))] += 1
    return {
        "applied_count": len(applied),
        "applied_rate": (len(applied) / len(records)) if records else 0.0,
        "mean_gold_top3_coverage_delta": mean([float(r["gold_top3_coverage_delta"]) for r in applied]) if applied else 0.0,
        "mean_qa_f1_delta": mean([float(r["qa_f1_delta"]) for r in applied]) if applied else 0.0,
        "promoted_from_rank_histogram": dict(sorted(promoted_hist.items())),
    }


def select_case_audit(records: List[Dict[str, Any]], limit: int) -> Dict[str, List[Dict[str, Any]]]:
    def _pick(bucket: str, reverse: bool = False) -> List[Dict[str, Any]]:
        rows = [record for record in records if record["bucket"] == bucket]
        rows.sort(
            key=lambda row: (
                float(row["qa_f1_delta"]),
                float(row["gold_top3_coverage_delta"]),
                row["question"],
            ),
            reverse=reverse,
        )
        return rows[:limit]

    return {
        "probe_applied__qa_improved": _pick("probe_applied__qa_improved", reverse=True),
        "probe_applied__qa_unchanged": _pick("probe_applied__qa_unchanged", reverse=True),
        "probe_applied__qa_worsened": _pick("probe_applied__qa_worsened", reverse=False),
        "invalid_selected_set_changed": _pick("invalid_selected_set_changed", reverse=False),
    }


def build_analysis(control_report: Dict[str, Any], probe_report: Dict[str, Any], case_limit: int) -> Dict[str, Any]:
    control_index = build_pair_index(control_report)
    probe_index = build_pair_index(probe_report)
    shared_questions = [question for question in control_index if question in probe_index]
    records = [
        build_query_record(control_index[question], probe_index[question])
        for question in shared_questions
    ]
    return {
        "dataset": str(control_report.get("dataset", probe_report.get("dataset", "unknown"))),
        "num_queries": len(records),
        "control_report": str(control_report.get("report_path", "")),
        "probe_report": str(probe_report.get("report_path", "")),
        "bucket_counts": summarize_records(records),
        "applied_summary": summarize_applied(records),
        "case_audit": select_case_audit(records, limit=case_limit),
        "records": records,
    }


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_case_block(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["- none"]
    lines: List[str] = []
    for row in rows:
        lines.append(f"- Q: {row['question']}")
        lines.append(
            f"  QA ΔF1={row['qa_f1_delta']:+.4f}, QA ΔEM={row['qa_em_delta']:+.4f}, "
            f"gold top3 Δrecall={row['gold_top3_coverage_delta']:+.4f}"
        )
        lines.append(
            f"  promoted={row['promoted_title'] or '<none>'} from rank {row['promoted_from_rank']}, "
            f"selected_set_unchanged={row['selected_set_unchanged']}, probe_applied={row['probe_applied']}"
        )
        lines.append(f"  control front: {row['control_front_titles']}")
        lines.append(f"  probe front: {row['probe_front_titles']}")
        lines.append(f"  gold: {row['gold_titles']}")
    return lines


def build_markdown(analysis: Dict[str, Any]) -> str:
    lines = [f"# Reader Order Probe Analysis: {analysis['dataset']}", ""]
    lines.append("## Buckets")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|---|---:|")
    for name, count in analysis["bucket_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.append("")

    applied_summary = dict(analysis.get("applied_summary", {}) or {})
    lines.append("## Applied Summary")
    lines.append("")
    lines.append(f"- Applied: `{int(applied_summary.get('applied_count', 0))}` / `{int(analysis.get('num_queries', 0))}` ({_format_pct(applied_summary.get('applied_rate', 0.0))})")
    lines.append(f"- Mean gold top3 Δrecall: `{float(applied_summary.get('mean_gold_top3_coverage_delta', 0.0)):+.4f}`")
    lines.append(f"- Mean QA ΔF1: `{float(applied_summary.get('mean_qa_f1_delta', 0.0)):+.4f}`")
    promoted_hist = dict(applied_summary.get("promoted_from_rank_histogram", {}) or {})
    if promoted_hist:
        lines.append("- Promoted-from-rank histogram:")
        for rank, count in promoted_hist.items():
            lines.append(f"  - rank `{rank}`: {count}")
    else:
        lines.append("- Promoted-from-rank histogram: none")
    lines.append("")

    for title, key in [
        ("Case Audit: Probe Applied, QA Improved", "probe_applied__qa_improved"),
        ("Case Audit: Probe Applied, QA Unchanged", "probe_applied__qa_unchanged"),
        ("Case Audit: Probe Applied, QA Worsened", "probe_applied__qa_worsened"),
        ("Case Audit: Invalid Selected Set Changed", "invalid_selected_set_changed"),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(_format_case_block(list((analysis.get("case_audit", {}) or {}).get(key, []) or [])))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired reader-order probe eval reports.")
    parser.add_argument("--control_report", required=True, help="Frozen-candidate eval report JSON.")
    parser.add_argument("--probe_report", required=True, help="Reader-order-probe eval report JSON.")
    parser.add_argument("--output_json", required=True, help="Machine-readable output path.")
    parser.add_argument("--output_md", required=True, help="Markdown summary output path.")
    parser.add_argument("--case_limit", type=int, default=10, help="Max cases per bucket.")
    args = parser.parse_args()

    control_path = Path(args.control_report)
    probe_path = Path(args.probe_report)
    control_report = load_report(control_path)
    probe_report = load_report(probe_path)
    control_report["report_path"] = str(control_path)
    probe_report["report_path"] = str(probe_path)
    analysis = build_analysis(control_report, probe_report, case_limit=int(args.case_limit))

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(analysis), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
