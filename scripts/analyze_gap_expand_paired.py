#!/usr/bin/env python3
"""Paired audit for gap_expand vs bridge_append reports.

This script is intentionally narrow. It compares two eval reports produced by
scripts/eval_causal_qwen3.py on the same question set and answers:

1. How often does gap_expand win / tie / lose against bridge_append+CE?
2. Are regressions concentrated in specific gold-doc buckets?
3. Do losses correlate with gap candidates entering the final front?
4. What gap traces are most common in the negative cases?
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping


EPS = 1e-9


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _round(value: float | int | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def build_trace_index(report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    traces = list(report.get("expand_assemble_query_traces", []) or [])
    return {str(trace.get("question", "")): dict(trace) for trace in traces}


def classify_delta(delta: float) -> str:
    if delta > EPS:
        return "win"
    if delta < -EPS:
        return "lose"
    return "tie"


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        key = normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(str(value))
    return ordered


def select_titles(titles: Iterable[str], keep_set: set[str]) -> List[str]:
    return [str(title) for title in titles if normalize_text(title) in keep_set]


def build_query_row(control_trace: Mapping[str, Any], candidate_trace: Mapping[str, Any]) -> Dict[str, Any]:
    question = str(candidate_trace.get("question", control_trace.get("question", "")))
    control_metrics = dict(control_trace.get("method_metrics", {}) or {})
    candidate_metrics = dict(candidate_trace.get("method_metrics", {}) or {})
    control_f1 = float(control_metrics.get("F1", 0.0) or 0.0)
    candidate_f1 = float(candidate_metrics.get("F1", 0.0) or 0.0)
    control_em = float(control_metrics.get("ExactMatch", 0.0) or 0.0)
    candidate_em = float(candidate_metrics.get("ExactMatch", 0.0) or 0.0)

    control_titles = list(control_trace.get("method_top_titles", []) or [])
    candidate_titles = list(candidate_trace.get("method_top_titles", []) or [])
    candidate_expand = dict(candidate_trace.get("expand_assemble_trace", {}) or {})
    gap_titles = unique_preserve_order(candidate_expand.get("gap_candidate_titles", []) or [])
    gap_title_set = {normalize_text(title) for title in gap_titles}
    selected_gap_titles = select_titles(candidate_titles, gap_title_set)

    gold_doc_count = int(candidate_trace.get("gold_doc_count", control_trace.get("gold_doc_count", 0)) or 0)
    bucket_label = f"{gold_doc_count}_doc" if gold_doc_count > 0 else "unknown"
    delta_f1 = candidate_f1 - control_f1
    delta_em = candidate_em - control_em

    control_title_set = {normalize_text(title) for title in control_titles}
    candidate_title_set = {normalize_text(title) for title in candidate_titles}
    added_titles = [title for title in candidate_titles if normalize_text(title) not in control_title_set]
    removed_titles = [title for title in control_titles if normalize_text(title) not in candidate_title_set]

    return {
        "question": question,
        "query_type": str(candidate_trace.get("query_type", control_trace.get("query_type", "unknown"))),
        "gold_doc_count": gold_doc_count,
        "bucket": bucket_label,
        "control_f1": _round(control_f1),
        "candidate_f1": _round(candidate_f1),
        "delta_f1": _round(delta_f1),
        "control_em": _round(control_em),
        "candidate_em": _round(candidate_em),
        "delta_em": _round(delta_em),
        "f1_outcome": classify_delta(delta_f1),
        "em_outcome": classify_delta(delta_em),
        "control_answer": str(control_trace.get("method_answer", "")),
        "candidate_answer": str(candidate_trace.get("method_answer", "")),
        "control_titles": control_titles,
        "candidate_titles": candidate_titles,
        "added_titles": added_titles,
        "removed_titles": removed_titles,
        "gap_type": str(candidate_expand.get("gap_type", "")),
        "gap_slot": str(candidate_expand.get("gap_slot", "")),
        "gap_fallback_used": bool(candidate_expand.get("gap_fallback_used", False)),
        "append_stop_reason": str(candidate_expand.get("append_stop_reason", "")),
        "gap_micro_queries": list(candidate_expand.get("gap_micro_queries", []) or []),
        "gap_candidate_titles": gap_titles,
        "gap_candidate_positions": list(candidate_expand.get("gap_candidate_positions", []) or []),
        "gap_candidate_selected_count": int(candidate_expand.get("gap_candidate_selected_count", 0) or 0),
        "gap_candidate_selected_into_final": bool(candidate_expand.get("gap_candidate_selected_into_final", False)),
        "gap_candidate_final_front_rate": _round(candidate_expand.get("gap_candidate_final_front_rate")),
        "selected_gap_titles": selected_gap_titles,
        "changed_evidence": control_titles != candidate_titles,
    }


def summarize_outcomes(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    counter = Counter(str(row.get(field, "tie")) for row in rows)
    values = [float(row.get("delta_f1", 0.0) or 0.0) for row in rows] if field == "f1_outcome" else [
        float(row.get("delta_em", 0.0) or 0.0) for row in rows
    ]
    return {
        "win": int(counter.get("win", 0)),
        "tie": int(counter.get("tie", 0)),
        "lose": int(counter.get("lose", 0)),
        "mean_delta": _round(mean(values) if values else 0.0),
    }


def summarize_subset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "f1": {"win": 0, "tie": 0, "lose": 0, "mean_delta": 0.0},
            "em": {"win": 0, "tie": 0, "lose": 0, "mean_delta": 0.0},
            "changed_evidence_rate": 0.0,
            "selected_gap_rate": 0.0,
        }
    return {
        "count": len(rows),
        "f1": summarize_outcomes(rows, "f1_outcome"),
        "em": summarize_outcomes(rows, "em_outcome"),
        "changed_evidence_rate": _round(mean(1.0 if row["changed_evidence"] else 0.0 for row in rows)),
        "selected_gap_rate": _round(mean(1.0 if row["gap_candidate_selected_into_final"] else 0.0 for row in rows)),
    }


def summarize_by_bucket(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["bucket"])].append(row)
    return {bucket: summarize_subset(bucket_rows) for bucket, bucket_rows in sorted(buckets.items())}


def summarize_by_stop_reason(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("append_stop_reason", ""))].append(row)
    payload: Dict[str, Any] = {}
    for reason, group_rows in sorted(groups.items()):
        payload[reason] = {
            "count": len(group_rows),
            "mean_delta_f1": _round(mean(float(row["delta_f1"]) for row in group_rows)),
            "win": sum(1 for row in group_rows if row["f1_outcome"] == "win"),
            "lose": sum(1 for row in group_rows if row["f1_outcome"] == "lose"),
        }
    return payload


def summarize_by_field(rows: List[Dict[str, Any]], field_name: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        field_value = str(row.get(field_name, "") or "")
        groups[field_value if field_value else "<empty>"].append(row)
    return {field_value: summarize_subset(group_rows) for field_value, group_rows in sorted(groups.items())}


def top_title_counter(rows: List[Dict[str, Any]], field: str, limit: int) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for title in list(row.get(field, []) or []):
            normalized = normalize_text(title)
            if normalized:
                counter[str(title)] += 1
    return [{"title": title, "count": int(count)} for title, count in counter.most_common(limit)]


def select_cases(rows: List[Dict[str, Any]], limit: int) -> Dict[str, List[Dict[str, Any]]]:
    improvements = sorted(
        [row for row in rows if row["f1_outcome"] == "win"],
        key=lambda row: (-float(row["delta_f1"]), row["question"]),
    )[:limit]
    regressions = sorted(
        [row for row in rows if row["f1_outcome"] == "lose"],
        key=lambda row: (float(row["delta_f1"]), row["question"]),
    )[:limit]
    regression_with_selected_gap = [
        row for row in regressions if row["gap_candidate_selected_into_final"]
    ][:limit]
    return {
        "improvements": improvements,
        "regressions": regressions,
        "regressions_with_selected_gap": regression_with_selected_gap,
    }


def build_analysis(control_report: Dict[str, Any], candidate_report: Dict[str, Any], case_limit: int) -> Dict[str, Any]:
    control_index = build_trace_index(control_report)
    candidate_index = build_trace_index(candidate_report)
    questions = [question for question in candidate_index if question in control_index]
    rows = [build_query_row(control_index[question], candidate_index[question]) for question in questions]

    selected_gap_rows = [row for row in rows if row["gap_candidate_selected_into_final"]]
    non_selected_gap_rows = [row for row in rows if not row["gap_candidate_selected_into_final"]]
    changed_evidence_rows = [row for row in rows if row["changed_evidence"]]
    unchanged_evidence_rows = [row for row in rows if not row["changed_evidence"]]
    regressions = [row for row in rows if row["f1_outcome"] == "lose"]
    improvements = [row for row in rows if row["f1_outcome"] == "win"]

    analysis = {
        "shared_query_count": len(rows),
        "control_report": str(control_report.get("config", {}).get("output_json", "")),
        "candidate_report": str(candidate_report.get("config", {}).get("output_json", "")),
        "overall": {
            "f1": summarize_outcomes(rows, "f1_outcome"),
            "em": summarize_outcomes(rows, "em_outcome"),
        },
        "subsets": {
            "all": summarize_subset(rows),
            "selected_gap": summarize_subset(selected_gap_rows),
            "non_selected_gap": summarize_subset(non_selected_gap_rows),
            "changed_evidence": summarize_subset(changed_evidence_rows),
            "unchanged_evidence": summarize_subset(unchanged_evidence_rows),
            "improvements": summarize_subset(improvements),
            "regressions": summarize_subset(regressions),
        },
        "by_bucket": summarize_by_bucket(rows),
        "by_gap_type": summarize_by_field(rows, "gap_type"),
        "by_gap_slot": summarize_by_field(rows, "gap_slot"),
        "by_stop_reason": summarize_by_stop_reason(rows),
        "gap_summary": {
            "gap_type_distribution": dict(Counter(str(row.get("gap_type", "")) for row in rows)),
            "gap_slot_distribution": dict(Counter(str(row.get("gap_slot", "")) for row in rows if str(row.get("gap_slot", "")))),
            "fallback_used_count": sum(1 for row in rows if row["gap_fallback_used"]),
            "selected_gap_count": len(selected_gap_rows),
            "selected_gap_rate": _round(len(selected_gap_rows) / len(rows) if rows else 0.0),
            "top_selected_gap_titles_in_regressions": top_title_counter(
                [row for row in regressions if row["gap_candidate_selected_into_final"]],
                field="selected_gap_titles",
                limit=10,
            ),
            "top_selected_gap_titles_in_improvements": top_title_counter(
                [row for row in improvements if row["gap_candidate_selected_into_final"]],
                field="selected_gap_titles",
                limit=10,
            ),
        },
        "cases": select_cases(rows, limit=case_limit),
        "query_rows": rows,
    }
    return analysis


def format_case_rows(rows: List[Mapping[str, Any]]) -> List[str]:
    if not rows:
        return ["None."]
    lines: List[str] = []
    for row in rows:
        lines.append(f"- Question: {row['question']}")
        lines.append(
            "  "
            f"F1 {row['control_f1']} -> {row['candidate_f1']} "
            f"(delta {row['delta_f1']}); "
            f"EM {row['control_em']} -> {row['candidate_em']} "
            f"(delta {row['delta_em']})"
        )
        lines.append(
            "  "
            f"bucket={row['bucket']}, stop_reason={row['append_stop_reason']}, "
            f"selected_gap={row['gap_candidate_selected_into_final']}, "
            f"selected_gap_titles={row['selected_gap_titles']}"
        )
        lines.append(f"  gap_micro_queries={row['gap_micro_queries']}")
        lines.append(f"  added_titles={row['added_titles']}")
        lines.append(f"  removed_titles={row['removed_titles']}")
    return lines


def build_markdown(analysis: Mapping[str, Any]) -> str:
    overall = analysis["overall"]
    subsets = analysis["subsets"]
    lines = [
        "# Gap Expand Paired Audit",
        "",
        f"Shared queries: {analysis['shared_query_count']}",
        "",
        "## Overall",
        "",
        f"- F1 win/tie/lose: {overall['f1']['win']} / {overall['f1']['tie']} / {overall['f1']['lose']} "
        f"(mean delta {overall['f1']['mean_delta']})",
        f"- EM win/tie/lose: {overall['em']['win']} / {overall['em']['tie']} / {overall['em']['lose']} "
        f"(mean delta {overall['em']['mean_delta']})",
        f"- Selected-gap subset: {subsets['selected_gap']['count']} queries, "
        f"F1 win/tie/lose {subsets['selected_gap']['f1']['win']} / "
        f"{subsets['selected_gap']['f1']['tie']} / {subsets['selected_gap']['f1']['lose']}",
        f"- Changed-evidence subset: {subsets['changed_evidence']['count']} queries, "
        f"F1 win/tie/lose {subsets['changed_evidence']['f1']['win']} / "
        f"{subsets['changed_evidence']['f1']['tie']} / {subsets['changed_evidence']['f1']['lose']}",
        f"- Unchanged-evidence subset: {subsets['unchanged_evidence']['count']} queries, "
        f"F1 win/tie/lose {subsets['unchanged_evidence']['f1']['win']} / "
        f"{subsets['unchanged_evidence']['f1']['tie']} / {subsets['unchanged_evidence']['f1']['lose']}",
        "",
        "## By Bucket",
        "",
    ]
    for bucket, payload in analysis["by_bucket"].items():
        lines.append(
            f"- {bucket}: count={payload['count']}, F1 win/tie/lose="
            f"{payload['f1']['win']} / {payload['f1']['tie']} / {payload['f1']['lose']}, "
            f"mean delta={payload['f1']['mean_delta']}, "
            f"selected_gap_rate={payload['selected_gap_rate']}"
        )

    lines.extend(["", "## Gap Trace", ""])
    gap_summary = analysis["gap_summary"]
    lines.append(f"- gap_type_distribution: {gap_summary['gap_type_distribution']}")
    lines.append(f"- gap_slot_distribution: {gap_summary['gap_slot_distribution']}")
    lines.append(f"- fallback_used_count: {gap_summary['fallback_used_count']}")
    lines.append(f"- selected_gap_count: {gap_summary['selected_gap_count']} (rate {gap_summary['selected_gap_rate']})")
    lines.append(f"- by_stop_reason: {analysis['by_stop_reason']}")
    lines.append(
        f"- top selected-gap titles in regressions: {gap_summary['top_selected_gap_titles_in_regressions']}"
    )
    lines.append(
        f"- top selected-gap titles in improvements: {gap_summary['top_selected_gap_titles_in_improvements']}"
    )

    lines.extend(["", "## By Gap Type", ""])
    for gap_type, payload in analysis["by_gap_type"].items():
        lines.append(
            f"- {gap_type}: count={payload['count']}, F1 win/tie/lose="
            f"{payload['f1']['win']} / {payload['f1']['tie']} / {payload['f1']['lose']}, "
            f"mean delta={payload['f1']['mean_delta']}, "
            f"selected_gap_rate={payload['selected_gap_rate']}"
        )

    lines.extend(["", "## By Gap Slot", ""])
    for gap_slot, payload in analysis["by_gap_slot"].items():
        lines.append(
            f"- {gap_slot}: count={payload['count']}, F1 win/tie/lose="
            f"{payload['f1']['win']} / {payload['f1']['tie']} / {payload['f1']['lose']}, "
            f"mean delta={payload['f1']['mean_delta']}, "
            f"selected_gap_rate={payload['selected_gap_rate']}"
        )

    lines.extend(["", "## Top Improvements", ""])
    lines.extend(format_case_rows(analysis["cases"]["improvements"]))
    lines.extend(["", "## Top Regressions", ""])
    lines.extend(format_case_rows(analysis["cases"]["regressions"]))
    lines.extend(["", "## Regressions With Selected Gap Candidate", ""])
    lines.extend(format_case_rows(analysis["cases"]["regressions_with_selected_gap"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired audit for gap_expand smoke vs bridge_append control.")
    parser.add_argument("--control_report", type=str, required=True)
    parser.add_argument("--candidate_report", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--case_limit", type=int, default=8)
    args = parser.parse_args()

    control_report = load_report(Path(args.control_report))
    candidate_report = load_report(Path(args.candidate_report))
    analysis = build_analysis(control_report, candidate_report, case_limit=int(args.case_limit))

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(build_markdown(analysis), encoding="utf-8")
    print(json.dumps({
        "shared_query_count": analysis["shared_query_count"],
        "overall": analysis["overall"],
        "gap_summary": analysis["gap_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
