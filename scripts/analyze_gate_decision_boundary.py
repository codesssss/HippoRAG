#!/usr/bin/env python3
"""Audit gate decision quality from paired selector reports.

This script compares an ungated/control report and a gated/candidate report
produced by scripts/eval_causal_qwen3.py. It answers a narrow question:

Can the current gate features separate beneficial off-rank substitutions from
harmful ones, or is the gate blocking good replacements and bad replacements
together?
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
FEATURE_NAMES = (
    "structure_score",
    "combined_score",
    "novelty_score",
    "frontier_gain_score",
    "path_coherence_score",
    "closure_score",
)


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


def title_hit_count(predicted_titles: Iterable[str], gold_titles: Iterable[str]) -> int:
    predicted_set = set(unique_preserve_order(predicted_titles))
    gold_set = set(unique_preserve_order(gold_titles))
    return len(predicted_set & gold_set)


def title_positions(titles: Iterable[str]) -> Dict[str, int]:
    positions: Dict[str, int] = {}
    for index, title in enumerate(titles, start=1):
        normalized = normalize_title(title)
        if normalized and normalized not in positions:
            positions[normalized] = index
    return positions


def round_or_none(value: Any, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def mean_or_none(values: Iterable[float | None], ndigits: int = 4) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(mean(numeric), ndigits)


def metric_range(values: Iterable[float | None]) -> Dict[str, float | None]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": round(min(numeric), 4),
        "max": round(max(numeric), 4),
        "mean": round(mean(numeric), 4),
    }


def ranges_overlap(left: Dict[str, float | None], right: Dict[str, float | None]) -> bool:
    if left["min"] is None or left["max"] is None or right["min"] is None or right["max"] is None:
        return False
    return not (float(left["max"]) < float(right["min"]) or float(right["max"]) < float(left["min"]))


def build_pair_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    traces = list(report.get("setwise_selector_query_traces", []) or [])
    return {str(trace.get("question", "")): trace for trace in traces}


def classify_bucket(candidate_gate_used: bool,
                    control_gold_hit: int,
                    candidate_gold_hit: int) -> str:
    if candidate_gate_used:
        if candidate_gold_hit > control_gold_hit:
            return "beneficial_allowed"
        if candidate_gold_hit < control_gold_hit:
            return "harmful_allowed"
        return "neutral_allowed"

    if control_gold_hit > candidate_gold_hit:
        return "beneficial_withheld"
    if control_gold_hit < candidate_gold_hit:
        return "harmful_blocked"
    return "neutral_blocked"


def extract_gate_feature_deltas(gate_decision: Dict[str, Any]) -> Dict[str, float | None]:
    payload: Dict[str, float | None] = {
        "avg_local_structure": round_or_none(gate_decision.get("avg_local_structure")),
        "suffix_base_mean": round_or_none(gate_decision.get("suffix_base_mean")),
    }
    for metric_name in FEATURE_NAMES:
        offrank_key = f"best_offrank_{metric_name}"
        suffix_key = f"weakest_baseline_suffix_{metric_name}"
        offrank_value = gate_decision.get(offrank_key)
        suffix_value = gate_decision.get(suffix_key)
        payload[offrank_key] = round_or_none(offrank_value)
        payload[suffix_key] = round_or_none(suffix_value)
        if offrank_value is None or suffix_value is None:
            payload[f"delta_{metric_name}"] = None
        else:
            payload[f"delta_{metric_name}"] = round(float(offrank_value) - float(suffix_value), 4)
    return payload


def build_case_record(control_trace: Dict[str, Any], candidate_trace: Dict[str, Any]) -> Dict[str, Any]:
    gold_titles = list(control_trace.get("gold_titles", []) or candidate_trace.get("gold_titles", []) or [])
    control_titles = list(control_trace.get("selector_top_titles", []) or [])
    candidate_titles = list(candidate_trace.get("selector_top_titles", []) or [])

    control_hit_count = title_hit_count(control_titles, gold_titles)
    candidate_hit_count = title_hit_count(candidate_titles, gold_titles)

    control_metrics = dict(control_trace.get("selector_metrics", {}) or {})
    candidate_metrics = dict(candidate_trace.get("selector_metrics", {}) or {})

    control_gate = dict((control_trace.get("selector_trace", {}) or {}).get("gate_decision", {}) or {})
    candidate_gate = dict((candidate_trace.get("selector_trace", {}) or {}).get("gate_decision", {}) or {})
    candidate_used = bool(candidate_gate.get("use_selector", False))
    bucket = classify_bucket(candidate_used, control_hit_count, candidate_hit_count)

    control_positions = title_positions(control_titles)
    candidate_positions = title_positions(candidate_titles)
    gold_positions_control = {title: control_positions.get(title) for title in unique_preserve_order(gold_titles)}
    gold_positions_candidate = {title: candidate_positions.get(title) for title in unique_preserve_order(gold_titles)}

    record: Dict[str, Any] = {
        "question": str(control_trace.get("question", "")),
        "query_type": str(control_trace.get("query_type", candidate_trace.get("query_type", "unknown"))),
        "gold_titles": gold_titles,
        "control_titles": control_titles,
        "candidate_titles": candidate_titles,
        "control_gold_hit_count": int(control_hit_count),
        "candidate_gold_hit_count": int(candidate_hit_count),
        "gold_hit_delta_candidate_minus_control": int(candidate_hit_count - control_hit_count),
        "control_selector_f1": round_or_none(control_metrics.get("F1")),
        "candidate_selector_f1": round_or_none(candidate_metrics.get("F1")),
        "selector_f1_delta_candidate_minus_control": round(
            float(candidate_metrics.get("F1", 0.0) or 0.0) - float(control_metrics.get("F1", 0.0) or 0.0),
            4,
        ),
        "control_selector_em": round_or_none(control_metrics.get("ExactMatch")),
        "candidate_selector_em": round_or_none(candidate_metrics.get("ExactMatch")),
        "selector_em_delta_candidate_minus_control": round(
            float(candidate_metrics.get("ExactMatch", 0.0) or 0.0) - float(control_metrics.get("ExactMatch", 0.0) or 0.0),
            4,
        ),
        "candidate_gate_used_selector": candidate_used,
        "candidate_gate_reason": str(candidate_gate.get("reason", "")),
        "control_gate_used_selector": bool(control_gate.get("use_selector", False)),
        "control_gate_reason": str(control_gate.get("reason", "")),
        "candidate_selected_pool_positions": list((candidate_trace.get("selector_trace", {}) or {}).get("selected_pool_positions", []) or []),
        "control_selected_pool_positions": list((control_trace.get("selector_trace", {}) or {}).get("selected_pool_positions", []) or []),
        "candidate_final_front_pool_positions": list((candidate_trace.get("selector_trace", {}) or {}).get("final_front_pool_positions", []) or []),
        "control_final_front_pool_positions": list((control_trace.get("selector_trace", {}) or {}).get("final_front_pool_positions", []) or []),
        "bucket": bucket,
        "control_gold_positions": gold_positions_control,
        "candidate_gold_positions": gold_positions_candidate,
    }
    record.update(extract_gate_feature_deltas(candidate_gate))
    return record


def summarize_bucket(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(records),
        "candidate_gate_reasons": dict(sorted(Counter(str(record.get("candidate_gate_reason", "")) for record in records).items())),
        "gold_hit_delta": metric_range(record.get("gold_hit_delta_candidate_minus_control") for record in records),
        "selector_f1_delta": metric_range(record.get("selector_f1_delta_candidate_minus_control") for record in records),
        "selector_em_delta": metric_range(record.get("selector_em_delta_candidate_minus_control") for record in records),
        "avg_local_structure": metric_range(record.get("avg_local_structure") for record in records),
        "suffix_base_mean": metric_range(record.get("suffix_base_mean") for record in records),
        **{
            metric_name: metric_range(record.get(metric_name) for record in records)
            for metric_name in (
                *[f"best_offrank_{name}" for name in FEATURE_NAMES],
                *[f"weakest_baseline_suffix_{name}" for name in FEATURE_NAMES],
                *[f"delta_{name}" for name in FEATURE_NAMES],
            )
        },
    }


def summarize_overlap(analysis: Dict[str, Any],
                      left_bucket: str,
                      right_bucket: str,
                      metric_names: Iterable[str]) -> Dict[str, Any]:
    left_summary = analysis["bucket_summaries"].get(left_bucket, {})
    right_summary = analysis["bucket_summaries"].get(right_bucket, {})
    overlap: Dict[str, Any] = {}
    for metric_name in metric_names:
        overlap[metric_name] = {
            "left": dict(left_summary.get(metric_name, {})),
            "right": dict(right_summary.get(metric_name, {})),
            "ranges_overlap": ranges_overlap(left_summary.get(metric_name, {}), right_summary.get(metric_name, {})),
        }
    return overlap


def select_cases(records: List[Dict[str, Any]], bucket: str, limit: int) -> List[Dict[str, Any]]:
    filtered = [record for record in records if record.get("bucket") == bucket]
    if bucket == "beneficial_withheld":
        filtered.sort(
            key=lambda row: (
                row.get("candidate_gate_reason", ""),
                -float(row.get("delta_combined_score") or -1e9),
                -float(row.get("delta_structure_score") or -1e9),
                row.get("question", ""),
            )
        )
    elif bucket == "harmful_blocked":
        filtered.sort(
            key=lambda row: (
                row.get("candidate_gate_reason", ""),
                float(row.get("delta_combined_score") or 1e9),
                row.get("question", ""),
            )
        )
    elif bucket == "beneficial_allowed":
        filtered.sort(
            key=lambda row: (
                -int(row.get("gold_hit_delta_candidate_minus_control") or 0),
                -float(row.get("delta_combined_score") or -1e9),
                row.get("question", ""),
            )
        )
    else:
        filtered.sort(
            key=lambda row: (
                int(row.get("gold_hit_delta_candidate_minus_control") or 0),
                float(row.get("delta_combined_score") or 1e9),
                row.get("question", ""),
            )
        )
    return filtered[:limit]


def infer_findings(records: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[str]:
    findings: List[str] = []
    bucket_summaries = analysis["bucket_summaries"]
    beneficial_withheld = bucket_summaries.get("beneficial_withheld", {}).get("count", 0)
    harmful_blocked = bucket_summaries.get("harmful_blocked", {}).get("count", 0)
    beneficial_allowed = bucket_summaries.get("beneficial_allowed", {}).get("count", 0)
    harmful_allowed = bucket_summaries.get("harmful_allowed", {}).get("count", 0)
    findings.append(
        f"Changed-gold paired cases: {beneficial_withheld + harmful_blocked + beneficial_allowed + harmful_allowed}. "
        f"Buckets = beneficial_withheld {beneficial_withheld}, harmful_blocked {harmful_blocked}, "
        f"beneficial_allowed {beneficial_allowed}, harmful_allowed {harmful_allowed}."
    )

    if beneficial_withheld:
        reasons = bucket_summaries["beneficial_withheld"].get("candidate_gate_reasons", {})
        findings.append(
            f"Beneficial-withheld cases are dominated by gate rejections rather than applied selector swaps: {reasons}."
        )

    if beneficial_allowed + harmful_allowed > 0:
        findings.append(
            "Allowed cases provide the counterfactual reference for whether current gate features can actually separate good and bad off-rank substitutions."
        )

    overlap = analysis["feature_overlap"]["beneficial_withheld_vs_harmful_blocked"]
    overlapping_metrics = [
        metric_name
        for metric_name, payload in overlap.items()
        if payload.get("ranges_overlap")
    ]
    if overlapping_metrics:
        findings.append(
            "Current absolute gate features overlap between beneficial_withheld and harmful_blocked for: "
            + ", ".join(overlapping_metrics[:8])
            + "."
        )

    delta_overlap = analysis["feature_overlap"]["beneficial_withheld_vs_harmful_blocked"].get("delta_combined_score", {})
    if delta_overlap.get("ranges_overlap"):
        findings.append(
            "Even incumbent-relative combined-score margin overlaps across blocked good and blocked bad cases, so threshold-only calibration may be insufficient."
        )
    else:
        findings.append(
            "Incumbent-relative combined-score margin shows some separation between blocked good and blocked bad cases, so small gate recalibration may still be viable."
        )

    if beneficial_withheld and harmful_blocked == 0:
        findings.append(
            "All blocked changed-gold cases are beneficial_withheld in this paired setting, which points to false-negative gating rather than successful precision control."
        )
    return findings


def build_analysis(control_report: Dict[str, Any],
                   candidate_report: Dict[str, Any],
                   case_limit: int) -> Dict[str, Any]:
    control_index = build_pair_index(control_report)
    candidate_index = build_pair_index(candidate_report)
    questions = [question for question in control_index if question in candidate_index]
    records = [build_case_record(control_index[question], candidate_index[question]) for question in questions]

    bucket_summaries = {
        bucket: summarize_bucket([record for record in records if record["bucket"] == bucket])
        for bucket in (
            "beneficial_withheld",
            "harmful_blocked",
            "beneficial_allowed",
            "harmful_allowed",
            "neutral_blocked",
            "neutral_allowed",
        )
    }

    metric_names = (
        "avg_local_structure",
        "suffix_base_mean",
        *[f"best_offrank_{name}" for name in FEATURE_NAMES],
        *[f"weakest_baseline_suffix_{name}" for name in FEATURE_NAMES],
        *[f"delta_{name}" for name in FEATURE_NAMES],
    )
    analysis: Dict[str, Any] = {
        "query_count": len(records),
        "bucket_summaries": bucket_summaries,
        "bucket_cases": {
            bucket: select_cases(records, bucket, case_limit)
            for bucket in (
                "beneficial_withheld",
                "harmful_blocked",
                "beneficial_allowed",
                "harmful_allowed",
            )
        },
    }
    analysis["feature_overlap"] = {
        "beneficial_withheld_vs_harmful_blocked": summarize_overlap(
            analysis, "beneficial_withheld", "harmful_blocked", metric_names
        ),
        "beneficial_allowed_vs_harmful_allowed": summarize_overlap(
            analysis, "beneficial_allowed", "harmful_allowed", metric_names
        ),
    }
    analysis["findings"] = infer_findings(records, analysis)
    return analysis


def render_case_block(case: Dict[str, Any]) -> List[str]:
    lines = [
        f"- Question: {case['question']}",
        f"  Bucket: `{case['bucket']}` | gate reason `{case['candidate_gate_reason']}` | candidate used selector `{case['candidate_gate_used_selector']}`",
        f"  Gold hit control -> candidate: {case['control_gold_hit_count']} -> {case['candidate_gold_hit_count']}",
        f"  Selector F1 control -> candidate: {case['control_selector_f1']} -> {case['candidate_selector_f1']}",
        f"  Delta combined / structure / novelty / frontier / path / closure: "
        f"{case.get('delta_combined_score')} / {case.get('delta_structure_score')} / {case.get('delta_novelty_score')} / "
        f"{case.get('delta_frontier_gain_score')} / {case.get('delta_path_coherence_score')} / {case.get('delta_closure_score')}",
        f"  avg_local_structure `{case.get('avg_local_structure')}` | suffix_base_mean `{case.get('suffix_base_mean')}`",
        f"  Control titles: {case['control_titles']}",
        f"  Candidate titles: {case['candidate_titles']}",
    ]
    return lines


def render_markdown(analysis: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Gate Decision Boundary Audit")
    lines.append("")
    lines.append(f"- Query count: {analysis['query_count']}")
    for bucket in ("beneficial_withheld", "harmful_blocked", "beneficial_allowed", "harmful_allowed"):
        lines.append(f"- {bucket}: {analysis['bucket_summaries'][bucket]['count']}")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    for finding in analysis.get("findings", []):
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Bucket Summary")
    lines.append("")
    lines.append("| Bucket | Count | Gate Reasons | Delta Combined Mean | Delta Structure Mean | Delta Novelty Mean | Delta Closure Mean |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for bucket in ("beneficial_withheld", "harmful_blocked", "beneficial_allowed", "harmful_allowed"):
        summary = analysis["bucket_summaries"][bucket]
        lines.append(
            f"| {bucket} | {summary['count']} | {summary['candidate_gate_reasons']} | "
            f"{summary['delta_combined_score']['mean']} | {summary['delta_structure_score']['mean']} | "
            f"{summary['delta_novelty_score']['mean']} | {summary['delta_closure_score']['mean']} |"
        )

    def append_overlap(section_title: str, payload: Dict[str, Any]) -> None:
        lines.append("")
        lines.append(f"## {section_title}")
        lines.append("")
        lines.append("| Metric | Left Range | Right Range | Overlap |")
        lines.append("|---|---:|---:|---|")
        for metric_name, value in payload.items():
            left = value.get("left", {})
            right = value.get("right", {})
            lines.append(
                f"| {metric_name} | {left.get('min')}..{left.get('max')} | "
                f"{right.get('min')}..{right.get('max')} | {value.get('ranges_overlap')} |"
            )

    append_overlap(
        "Blocked Feature Overlap",
        analysis["feature_overlap"]["beneficial_withheld_vs_harmful_blocked"],
    )
    append_overlap(
        "Allowed Feature Overlap",
        analysis["feature_overlap"]["beneficial_allowed_vs_harmful_allowed"],
    )

    for bucket in ("beneficial_withheld", "harmful_blocked", "beneficial_allowed", "harmful_allowed"):
        lines.append("")
        lines.append(f"## {bucket}")
        lines.append("")
        cases = analysis["bucket_cases"].get(bucket, [])
        if not cases:
            lines.append("- None")
            continue
        for case in cases:
            lines.extend(render_case_block(case))
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit gate decision quality from paired selector reports.")
    parser.add_argument("--control_report", type=Path, required=True)
    parser.add_argument("--candidate_report", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--case_limit", type=int, default=8)
    args = parser.parse_args()

    analysis = build_analysis(
        control_report=load_report(args.control_report),
        candidate_report=load_report(args.candidate_report),
        case_limit=max(1, int(args.case_limit)),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(analysis), encoding="utf-8")


if __name__ == "__main__":
    main()
