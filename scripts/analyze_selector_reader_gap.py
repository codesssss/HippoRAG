#!/usr/bin/env python3
"""Analyze selector-reader gap from paired fresh eval reports.

This script compares a control and candidate report produced by
scripts/eval_causal_qwen3.py and answers a narrow question:

Did the candidate improve evidence-set quality without translating that gain
into answer quality, and if so, what position-sensitive patterns explain the
gap?
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


EPS = 1e-9


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_title(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        normalized = normalize_title(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def compute_support_metrics(predicted_titles: Iterable[str], gold_titles: Iterable[str]) -> Dict[str, float]:
    predicted = unique_preserve_order(predicted_titles)
    gold = unique_preserve_order(gold_titles)
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit_count": 0.0, "full_support": 0.0}

    hit_count = len(predicted_set & gold_set)
    precision = hit_count / len(predicted_set) if predicted_set else 0.0
    recall = hit_count / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if precision + recall <= 0 else (2 * precision * recall) / (precision + recall)
    full_support = 1.0 if gold_set.issubset(predicted_set) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_count": float(hit_count),
        "full_support": full_support,
    }


def title_positions(titles: Iterable[str]) -> Dict[str, int]:
    positions: Dict[str, int] = {}
    for index, title in enumerate(titles, start=1):
        normalized = normalize_title(title)
        if normalized and normalized not in positions:
            positions[normalized] = index
    return positions


def classify_delta(delta: float) -> str:
    if delta > EPS:
        return "improved"
    if delta < -EPS:
        return "worsened"
    return "unchanged"


def classify_quadrant(selector_delta: float, qa_delta: float) -> str:
    selector_label = classify_delta(selector_delta)
    qa_label = classify_delta(qa_delta)
    return f"selector_{selector_label}__qa_{qa_label}"


def build_query_comparison(control_trace: Dict[str, Any], candidate_trace: Dict[str, Any]) -> Dict[str, Any]:
    question = str(control_trace.get("question", ""))
    gold_titles = list(control_trace.get("gold_titles", []) or candidate_trace.get("gold_titles", []) or [])

    control_titles = list(control_trace.get("selector_top_titles", []) or [])
    candidate_titles = list(candidate_trace.get("selector_top_titles", []) or [])

    control_support = compute_support_metrics(control_titles, gold_titles)
    candidate_support = compute_support_metrics(candidate_titles, gold_titles)
    control_support_top3 = compute_support_metrics(control_titles[:3], gold_titles)
    candidate_support_top3 = compute_support_metrics(candidate_titles[:3], gold_titles)

    control_positions = title_positions(control_titles)
    candidate_positions = title_positions(candidate_titles)
    gold_positions_control = {title: control_positions.get(title) for title in unique_preserve_order(gold_titles)}
    gold_positions_candidate = {title: candidate_positions.get(title) for title in unique_preserve_order(gold_titles)}

    added_titles = [title for title in candidate_titles if normalize_title(title) not in control_positions]
    removed_titles = [title for title in control_titles if normalize_title(title) not in candidate_positions]
    added_positions = [candidate_positions[normalize_title(title)] for title in added_titles if normalize_title(title) in candidate_positions]

    control_qa_metrics = dict(control_trace.get("selector_metrics", {}) or {})
    candidate_qa_metrics = dict(candidate_trace.get("selector_metrics", {}) or {})
    control_qa_f1 = float(control_qa_metrics.get("F1", 0.0) or 0.0)
    candidate_qa_f1 = float(candidate_qa_metrics.get("F1", 0.0) or 0.0)
    control_qa_em = float(control_qa_metrics.get("ExactMatch", 0.0) or 0.0)
    candidate_qa_em = float(candidate_qa_metrics.get("ExactMatch", 0.0) or 0.0)

    support_f1_delta = float(candidate_support["f1"]) - float(control_support["f1"])
    qa_f1_delta = candidate_qa_f1 - control_qa_f1
    qa_em_delta = candidate_qa_em - control_qa_em

    control_top3_norm = unique_preserve_order(control_titles[:3])
    candidate_top3_norm = unique_preserve_order(candidate_titles[:3])
    control_top2_norm = unique_preserve_order(control_titles[:2])
    candidate_top2_norm = unique_preserve_order(candidate_titles[:2])

    gold_top3_coverage_delta = float(candidate_support_top3["recall"]) - float(control_support_top3["recall"])
    any_gold_position_improved = False
    for title, control_pos in gold_positions_control.items():
        candidate_pos = gold_positions_candidate.get(title)
        if control_pos is None and candidate_pos is not None:
            any_gold_position_improved = True
            break
        if control_pos is not None and candidate_pos is not None and candidate_pos < control_pos:
            any_gold_position_improved = True
            break

    return {
        "question": question,
        "query_type": str(control_trace.get("query_type", candidate_trace.get("query_type", "unknown"))),
        "gold_titles": gold_titles,
        "control_answer": str(control_trace.get("selector_answer", "")),
        "candidate_answer": str(candidate_trace.get("selector_answer", "")),
        "control_titles": control_titles,
        "candidate_titles": candidate_titles,
        "added_titles": added_titles,
        "removed_titles": removed_titles,
        "added_positions": added_positions,
        "control_support": control_support,
        "candidate_support": candidate_support,
        "control_support_top3": control_support_top3,
        "candidate_support_top3": candidate_support_top3,
        "support_f1_delta": support_f1_delta,
        "qa_f1_control": control_qa_f1,
        "qa_f1_candidate": candidate_qa_f1,
        "qa_f1_delta": qa_f1_delta,
        "qa_em_control": control_qa_em,
        "qa_em_candidate": candidate_qa_em,
        "qa_em_delta": qa_em_delta,
        "quadrant": classify_quadrant(support_f1_delta, qa_f1_delta),
        "top2_unchanged": control_top2_norm == candidate_top2_norm,
        "top3_unchanged": control_top3_norm == candidate_top3_norm,
        "all_added_after_top3": bool(added_positions) and min(added_positions) >= 4,
        "any_added_in_top3": any(position <= 3 for position in added_positions),
        "gold_top3_coverage_delta": gold_top3_coverage_delta,
        "any_gold_position_improved": any_gold_position_improved,
        "control_gold_positions": gold_positions_control,
        "candidate_gold_positions": gold_positions_candidate,
        "candidate_gate_reason": str(((candidate_trace.get("selector_trace", {}) or {}).get("gate_decision", {}) or {}).get("reason", "")),
        "candidate_gate_used_selector": bool(((candidate_trace.get("selector_trace", {}) or {}).get("gate_decision", {}) or {}).get("use_selector", False)),
        "candidate_saturation_triggered": bool((candidate_trace.get("selector_trace", {}) or {}).get("saturation_guard_triggered", False)),
    }


def build_pair_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    traces = list(report.get("setwise_selector_query_traces", []) or [])
    return {str(trace.get("question", "")): trace for trace in traces}


def summarize_quadrants(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[str(record.get("quadrant", "unknown"))] += 1
    return dict(sorted(counts.items()))


def summarize_position_patterns(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "top2_unchanged_rate": 0.0,
            "top3_unchanged_rate": 0.0,
            "all_added_after_top3_rate": 0.0,
            "any_added_in_top3_rate": 0.0,
            "any_gold_position_improved_rate": 0.0,
            "gold_top3_coverage_delta_mean": 0.0,
            "avg_added_title_count": 0.0,
        }
    return {
        "count": len(records),
        "top2_unchanged_rate": mean(1.0 if record["top2_unchanged"] else 0.0 for record in records),
        "top3_unchanged_rate": mean(1.0 if record["top3_unchanged"] else 0.0 for record in records),
        "all_added_after_top3_rate": mean(1.0 if record["all_added_after_top3"] else 0.0 for record in records),
        "any_added_in_top3_rate": mean(1.0 if record["any_added_in_top3"] else 0.0 for record in records),
        "any_gold_position_improved_rate": mean(1.0 if record["any_gold_position_improved"] else 0.0 for record in records),
        "gold_top3_coverage_delta_mean": mean(float(record["gold_top3_coverage_delta"]) for record in records),
        "avg_added_title_count": mean(float(len(record["added_titles"])) for record in records),
    }


def select_case_audit(records: List[Dict[str, Any]], limit: int) -> Dict[str, List[Dict[str, Any]]]:
    selector_up_qa_same = sorted(
        [record for record in records if record["quadrant"] == "selector_improved__qa_unchanged"],
        key=lambda row: (-float(row["support_f1_delta"]), row["question"]),
    )[:limit]
    selector_up_qa_down = sorted(
        [record for record in records if record["quadrant"] == "selector_improved__qa_worsened"],
        key=lambda row: (float(row["qa_f1_delta"]), -float(row["support_f1_delta"]), row["question"]),
    )[:limit]
    selector_up_qa_up = sorted(
        [record for record in records if record["quadrant"] == "selector_improved__qa_improved"],
        key=lambda row: (-float(row["qa_f1_delta"]), -float(row["support_f1_delta"]), row["question"]),
    )[:limit]
    return {
        "selector_improved__qa_unchanged": selector_up_qa_same,
        "selector_improved__qa_worsened": selector_up_qa_down,
        "selector_improved__qa_improved": selector_up_qa_up,
    }


def build_analysis(control_report: Dict[str, Any],
                   candidate_report: Dict[str, Any],
                   case_limit: int) -> Dict[str, Any]:
    control_index = build_pair_index(control_report)
    candidate_index = build_pair_index(candidate_report)
    questions = [question for question in control_index if question in candidate_index]

    records = [
        build_query_comparison(control_index[question], candidate_index[question])
        for question in questions
    ]

    summary_records = {
        name: [record for record in records if record["quadrant"] == name]
        for name in sorted({record["quadrant"] for record in records})
    }

    pattern_focus = {
        "selector_improved__qa_unchanged": summarize_position_patterns(
            summary_records.get("selector_improved__qa_unchanged", [])
        ),
        "selector_improved__qa_worsened": summarize_position_patterns(
            summary_records.get("selector_improved__qa_worsened", [])
        ),
        "selector_improved__qa_improved": summarize_position_patterns(
            summary_records.get("selector_improved__qa_improved", [])
        ),
    }

    gate_reason_counts: Counter[str] = Counter()
    for record in records:
        reason = str(record.get("candidate_gate_reason", "")).strip()
        if reason:
            gate_reason_counts[reason] += 1

    dataset = str(control_report.get("dataset", candidate_report.get("dataset", "unknown")))
    return {
        "dataset": dataset,
        "num_queries": len(records),
        "control_report": str(control_report.get("report_path", "")),
        "candidate_report": str(candidate_report.get("report_path", "")),
        "quadrants": summarize_quadrants(records),
        "position_patterns": pattern_focus,
        "candidate_gate_reason_counts": dict(sorted(gate_reason_counts.items())),
        "case_audit": select_case_audit(records, limit=case_limit),
        "records": records,
    }


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_case_block(rows: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if not rows:
        lines.append("- none")
        return lines
    for row in rows:
        lines.append(f"- Q: {row['question']}")
        lines.append(
            f"  support ΔF1={row['support_f1_delta']:+.4f}, QA ΔF1={row['qa_f1_delta']:+.4f}, "
            f"QA ΔEM={row['qa_em_delta']:+.4f}"
        )
        lines.append(
            f"  top2_unchanged={row['top2_unchanged']}, top3_unchanged={row['top3_unchanged']}, "
            f"all_added_after_top3={row['all_added_after_top3']}, any_gold_position_improved={row['any_gold_position_improved']}"
        )
        lines.append(f"  gold: {row['gold_titles']}")
        lines.append(f"  control answer: {row['control_answer']}")
        lines.append(f"  candidate answer: {row['candidate_answer']}")
        lines.append(f"  control titles: {row['control_titles']}")
        lines.append(f"  candidate titles: {row['candidate_titles']}")
        lines.append(f"  added titles: {row['added_titles']}")
    return lines


def build_markdown(analysis: Dict[str, Any]) -> str:
    lines = [f"# Selector Reader Gap Analysis: {analysis['dataset']}", ""]
    lines.append("## Quadrants")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|---|---:|")
    for name, count in analysis["quadrants"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.append("")

    lines.append("## Position Patterns")
    lines.append("")
    lines.append("| Bucket | Count | Top2 unchanged | Top3 unchanged | Added only @4-5 | Added in top3 | Gold pos improved | Gold top3 Δrecall | Avg added docs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bucket, stats in analysis["position_patterns"].items():
        lines.append(
            f"| `{bucket}` | {stats['count']} | {_format_pct(stats['top2_unchanged_rate'])} | "
            f"{_format_pct(stats['top3_unchanged_rate'])} | {_format_pct(stats['all_added_after_top3_rate'])} | "
            f"{_format_pct(stats['any_added_in_top3_rate'])} | {_format_pct(stats['any_gold_position_improved_rate'])} | "
            f"{stats['gold_top3_coverage_delta_mean']:+.4f} | {stats['avg_added_title_count']:.2f} |"
        )
    lines.append("")

    lines.append("## Candidate Gate Reasons")
    lines.append("")
    if analysis["candidate_gate_reason_counts"]:
        for reason, count in analysis["candidate_gate_reason_counts"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Case Audit: Selector Up, QA Unchanged")
    lines.append("")
    lines.extend(_format_case_block(analysis["case_audit"]["selector_improved__qa_unchanged"]))
    lines.append("")

    lines.append("## Case Audit: Selector Up, QA Worsened")
    lines.append("")
    lines.extend(_format_case_block(analysis["case_audit"]["selector_improved__qa_worsened"]))
    lines.append("")

    lines.append("## Case Audit: Selector Up, QA Improved")
    lines.append("")
    lines.extend(_format_case_block(analysis["case_audit"]["selector_improved__qa_improved"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze selector-reader gap from paired fresh eval reports.")
    parser.add_argument("--control_report", required=True, help="Control eval report JSON.")
    parser.add_argument("--candidate_report", required=True, help="Candidate eval report JSON.")
    parser.add_argument("--output_json", required=True, help="Machine-readable output path.")
    parser.add_argument("--output_md", required=True, help="Markdown summary output path.")
    parser.add_argument("--case_limit", type=int, default=10, help="Max cases per audit bucket.")
    args = parser.parse_args()

    control_path = Path(args.control_report)
    candidate_path = Path(args.candidate_report)
    control_report = load_report(control_path)
    candidate_report = load_report(candidate_path)
    control_report["report_path"] = str(control_path)
    candidate_report["report_path"] = str(candidate_path)

    analysis = build_analysis(control_report, candidate_report, case_limit=int(args.case_limit))
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(analysis), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
