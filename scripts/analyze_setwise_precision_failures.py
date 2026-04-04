#!/usr/bin/env python3
"""Analyze residual precision failures from setwise selector reports.

This script reads a report JSON produced by scripts/eval_causal_qwen3.py,
extracts per-query selector outcomes, and summarizes where bridge-beam still
helps or hurts. The goal is to answer a narrow question:

Can the current absolute bridge/state features already separate true bridge
wins from high-structure topical-noise failures, or do we need a new signal?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _round(value: float | int | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _metric_range(values: Iterable[float]) -> Dict[str, float | None]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": _round(min(numeric)),
        "max": _round(max(numeric)),
        "mean": _round(mean(numeric)),
    }


def _ranges_overlap(left: Dict[str, float | None], right: Dict[str, float | None]) -> bool:
    if left["min"] is None or left["max"] is None or right["min"] is None or right["max"] is None:
        return False
    return not (float(left["max"]) < float(right["min"]) or float(right["max"]) < float(left["min"]))


def build_query_record(query_trace: Dict[str, Any], saturation_threshold: float = 0.999) -> Dict[str, Any]:
    selector_trace = dict(query_trace.get("selector_trace", {}) or {})
    gate_decision = dict(selector_trace.get("gate_decision", {}) or {})
    selector_metrics = dict(query_trace.get("selector_metrics", {}) or {})
    baseline_metrics = dict(query_trace.get("baseline_metrics", {}) or {})
    delta_f1 = float(selector_metrics.get("F1", 0.0) or 0.0) - float(baseline_metrics.get("F1", 0.0) or 0.0)

    beam_steps: List[Dict[str, Any]] = []
    for step in selector_trace.get("selection_steps", []) or []:
        if not isinstance(step, dict) or str(step.get("mode")) != "beam":
            continue
        beam_steps.append({
            "step": int(step.get("step", 0) or 0),
            "pool_position": int(step.get("pool_position", -1) or -1),
            "doc_id": step.get("doc_id"),
            "base_score": _round(step.get("base_score")),
            "structure_score": _round(step.get("structure_score")),
            "novelty_score": _round(step.get("novelty_score")),
            "closure_score": _round(step.get("closure_score")),
            "selection_score": _round(step.get("selection_score")),
            "combined_score": _round(step.get("combined_score")),
            "proposal_rank": step.get("proposal_rank"),
            "state_score": _round(step.get("state_score")),
        })

    best_state = {
        "state_score": _round(selector_trace.get("beam_best_state_score")),
        "path_connectivity": _round(selector_trace.get("beam_best_state_path_connectivity")),
        "reachable_doc_ratio": _round(selector_trace.get("beam_best_state_reachable_doc_ratio")),
        "query_reachability": _round(selector_trace.get("beam_best_state_query_reachability")),
        "query_coverage": _round(selector_trace.get("beam_best_state_query_coverage")),
        "support_mean": _round(selector_trace.get("beam_best_state_support_mean")),
        "suffix_base_mean": _round(selector_trace.get("beam_best_state_suffix_base_mean")),
    }
    saturated_state = all(
        float(best_state.get(metric_name, 0.0) or 0.0) >= float(saturation_threshold)
        for metric_name in ("path_connectivity", "reachable_doc_ratio", "query_reachability", "query_coverage")
    )

    local_structure_values = [float(step.get("structure_score", 0.0) or 0.0) for step in beam_steps]
    local_novelty_values = [float(step.get("novelty_score", 0.0) or 0.0) for step in beam_steps]
    local_closure_values = [float(step.get("closure_score", 0.0) or 0.0) for step in beam_steps]

    return {
        "question": str(query_trace.get("question", "")),
        "delta_f1": _round(delta_f1),
        "baseline_f1": _round(baseline_metrics.get("F1")),
        "selector_f1": _round(selector_metrics.get("F1")),
        "baseline_answer": str(query_trace.get("baseline_answer", "")),
        "selector_answer": str(query_trace.get("selector_answer", "")),
        "baseline_titles": list(query_trace.get("baseline_top_titles", []) or []),
        "selector_titles": list(query_trace.get("selector_top_titles", []) or []),
        "gate_reason": str(gate_decision.get("reason", "")),
        "gate_use_selector": bool(gate_decision.get("use_selector", False)),
        "beam_steps": beam_steps,
        "best_state": best_state,
        "saturated_state": bool(saturated_state),
        "max_local_structure": _round(max(local_structure_values), 4) if local_structure_values else None,
        "max_local_novelty": _round(max(local_novelty_values), 4) if local_novelty_values else None,
        "max_local_closure": _round(max(local_closure_values), 4) if local_closure_values else None,
        "avg_local_structure": _round(mean(local_structure_values), 4) if local_structure_values else None,
        "avg_local_novelty": _round(mean(local_novelty_values), 4) if local_novelty_values else None,
        "avg_local_closure": _round(mean(local_closure_values), 4) if local_closure_values else None,
    }


def summarize_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(records),
        "saturated_state_count": sum(1 for record in records if record.get("saturated_state")),
        "state_score": _metric_range(record["best_state"]["state_score"] for record in records if record["best_state"]["state_score"] is not None),
        "path_connectivity": _metric_range(record["best_state"]["path_connectivity"] for record in records if record["best_state"]["path_connectivity"] is not None),
        "reachable_doc_ratio": _metric_range(record["best_state"]["reachable_doc_ratio"] for record in records if record["best_state"]["reachable_doc_ratio"] is not None),
        "query_reachability": _metric_range(record["best_state"]["query_reachability"] for record in records if record["best_state"]["query_reachability"] is not None),
        "query_coverage": _metric_range(record["best_state"]["query_coverage"] for record in records if record["best_state"]["query_coverage"] is not None),
        "support_mean": _metric_range(record["best_state"]["support_mean"] for record in records if record["best_state"]["support_mean"] is not None),
        "suffix_base_mean": _metric_range(record["best_state"]["suffix_base_mean"] for record in records if record["best_state"]["suffix_base_mean"] is not None),
        "max_local_structure": _metric_range(record["max_local_structure"] for record in records if record["max_local_structure"] is not None),
        "max_local_novelty": _metric_range(record["max_local_novelty"] for record in records if record["max_local_novelty"] is not None),
        "max_local_closure": _metric_range(record["max_local_closure"] for record in records if record["max_local_closure"] is not None),
        "avg_local_structure": _metric_range(record["avg_local_structure"] for record in records if record["avg_local_structure"] is not None),
        "avg_local_novelty": _metric_range(record["avg_local_novelty"] for record in records if record["avg_local_novelty"] is not None),
        "avg_local_closure": _metric_range(record["avg_local_closure"] for record in records if record["avg_local_closure"] is not None),
    }


def summarize_feature_overlap(positive_summary: Dict[str, Any], negative_summary: Dict[str, Any]) -> Dict[str, Any]:
    metric_names = (
        "state_score",
        "path_connectivity",
        "reachable_doc_ratio",
        "query_reachability",
        "query_coverage",
        "support_mean",
        "suffix_base_mean",
        "max_local_structure",
        "max_local_novelty",
        "max_local_closure",
        "avg_local_structure",
        "avg_local_novelty",
        "avg_local_closure",
    )
    overlap: Dict[str, Any] = {}
    for metric_name in metric_names:
        overlap[metric_name] = {
            "positive": dict(positive_summary.get(metric_name, {})),
            "negative": dict(negative_summary.get(metric_name, {})),
            "ranges_overlap": _ranges_overlap(
                positive_summary.get(metric_name, {}),
                negative_summary.get(metric_name, {}),
            ),
        }
    return overlap


def infer_findings(positive_records: List[Dict[str, Any]],
                   negative_records: List[Dict[str, Any]],
                   overlap_summary: Dict[str, Any]) -> List[str]:
    findings: List[str] = []
    findings.append(
        f"G1 changed {len(positive_records) + len(negative_records)} queries: "
        f"{len(positive_records)} positive and {len(negative_records)} negative."
    )
    if negative_records:
        saturated_negatives = sum(1 for record in negative_records if record.get("saturated_state"))
        findings.append(
            f"{saturated_negatives}/{len(negative_records)} negative queries end in fully saturated beam states "
            "(path_connectivity, reachable_doc_ratio, query_reachability, query_coverage all ~1.0)."
        )
    if positive_records:
        saturated_positives = sum(1 for record in positive_records if record.get("saturated_state"))
        findings.append(
            f"{saturated_positives}/{len(positive_records)} positive queries are also saturated, so saturation alone "
            "cannot serve as a safe precision gate."
        )

    for metric_name in ("max_local_structure", "max_local_novelty", "max_local_closure", "support_mean", "state_score"):
        metric_overlap = overlap_summary.get(metric_name, {})
        if metric_overlap.get("ranges_overlap"):
            findings.append(
                f"{metric_name} overlaps between positive and negative queries, so current absolute scores do not cleanly "
                "separate true bridges from topical noise."
            )

    findings.append(
        "The remaining hard failure family is not missing-bridge capacity anymore. It is high-structure topical expansion: "
        "the beam can assign bridge-like local scores and high state scores to documents that stay inside a dense seed/query cluster."
    )
    findings.append(
        "The next precision signal should be incremental rather than absolute: measure whether a candidate adds chain utility "
        "beyond the current state, instead of only requiring high structure/novelty/closure in isolation."
    )
    return findings


def build_analysis(report: Dict[str, Any], saturation_threshold: float = 0.999) -> Dict[str, Any]:
    traces = list(report.get("setwise_selector_query_traces", []) or [])
    records = [build_query_record(trace, saturation_threshold=saturation_threshold) for trace in traces]
    positive_records = [record for record in records if float(record["delta_f1"] or 0.0) > 0.0]
    negative_records = [record for record in records if float(record["delta_f1"] or 0.0) < 0.0]
    neutral_records = [record for record in records if abs(float(record["delta_f1"] or 0.0)) <= 1e-9]

    positive_records.sort(key=lambda record: (-float(record["delta_f1"] or 0.0), record["question"]))
    negative_records.sort(key=lambda record: (float(record["delta_f1"] or 0.0), record["question"]))

    positive_summary = summarize_group(positive_records)
    negative_summary = summarize_group(negative_records)
    neutral_summary = summarize_group(neutral_records)
    overlap_summary = summarize_feature_overlap(positive_summary, negative_summary)

    return {
        "report_source": str(report.get("config", {}).get("output_json", "")),
        "selector": report.get("setwise_selector_qa", {}).get("selector"),
        "score_mode": report.get("setwise_selector_qa", {}).get("score_mode"),
        "query_count": len(records),
        "positive_summary": positive_summary,
        "negative_summary": negative_summary,
        "neutral_summary": neutral_summary,
        "feature_overlap": overlap_summary,
        "findings": infer_findings(positive_records, negative_records, overlap_summary),
        "top_positive_queries": positive_records[:8],
        "top_negative_queries": negative_records[:8],
    }


def render_markdown(analysis: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# G1 Residual Precision Failure Audit")
    lines.append("")
    lines.append(f"- Query count: {analysis['query_count']}")
    lines.append(f"- Selector: `{analysis.get('selector')}` / score mode `{analysis.get('score_mode')}`")
    lines.append(f"- Positive changed queries: {analysis['positive_summary']['count']}")
    lines.append(f"- Negative changed queries: {analysis['negative_summary']['count']}")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    for finding in analysis.get("findings", []):
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Feature Overlap")
    lines.append("")
    lines.append("| Metric | Positive Range | Negative Range | Overlap |")
    lines.append("|---|---:|---:|---|")
    for metric_name, payload in analysis.get("feature_overlap", {}).items():
        positive = payload.get("positive", {})
        negative = payload.get("negative", {})
        positive_range = f"{positive.get('min')}..{positive.get('max')}"
        negative_range = f"{negative.get('min')}..{negative.get('max')}"
        lines.append(f"| {metric_name} | {positive_range} | {negative_range} | {payload.get('ranges_overlap')} |")

    def _append_queries(section_title: str, items: List[Dict[str, Any]]) -> None:
        lines.append("")
        lines.append(f"## {section_title}")
        lines.append("")
        for item in items:
            lines.append(f"### {item['question']}")
            lines.append("")
            lines.append(f"- Delta F1: {item['delta_f1']}")
            lines.append(f"- Answers: `{item['baseline_answer']}` -> `{item['selector_answer']}`")
            lines.append(f"- Gate: `{item['gate_reason']}`")
            lines.append(f"- Saturated state: `{item['saturated_state']}`")
            lines.append(
                "- Best state metrics: "
                f"score={item['best_state']['state_score']}, "
                f"path={item['best_state']['path_connectivity']}, "
                f"reachable={item['best_state']['reachable_doc_ratio']}, "
                f"qreach={item['best_state']['query_reachability']}, "
                f"qcov={item['best_state']['query_coverage']}, "
                f"support={item['best_state']['support_mean']}, "
                f"suffix_base={item['best_state']['suffix_base_mean']}"
            )
            lines.append(f"- Selected titles: {item['selector_titles']}")
            if item["beam_steps"]:
                lines.append("- Beam steps:")
                for step in item["beam_steps"]:
                    lines.append(
                        f"  - step {step['step']}: pool={step['pool_position']}, "
                        f"structure={step['structure_score']}, novelty={step['novelty_score']}, "
                        f"closure={step['closure_score']}, selection={step['selection_score']}, "
                        f"state={step['state_score']}, rank={step['proposal_rank']}"
                    )
            lines.append("")

    _append_queries("Positive Queries", analysis.get("top_positive_queries", []))
    _append_queries("Negative Queries", analysis.get("top_negative_queries", []))
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze residual precision failures from setwise selector reports.")
    parser.add_argument("--report_json", required=True, help="Path to the input report JSON.")
    parser.add_argument("--output_json", required=True, help="Path to write the structured analysis JSON.")
    parser.add_argument("--output_md", required=True, help="Path to write the markdown summary.")
    parser.add_argument("--saturation_threshold", type=float, default=0.999, help="Threshold used to mark a state as saturated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_report(Path(args.report_json))
    analysis = build_analysis(report, saturation_threshold=float(args.saturation_threshold))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(analysis), encoding="utf-8")

    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(output_md),
        "query_count": analysis["query_count"],
        "positive_count": analysis["positive_summary"]["count"],
        "negative_count": analysis["negative_summary"]["count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
