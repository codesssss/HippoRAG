#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping


EPS = 1e-9


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def compute_support_metrics(predicted_titles: Iterable[str], gold_titles: Iterable[str]) -> Dict[str, float]:
    predicted = unique_preserve_order(predicted_titles)
    gold = unique_preserve_order(gold_titles)
    predicted_set = set(predicted)
    gold_set = set(gold)
    overlap = len(predicted_set & gold_set)
    precision = (overlap / len(predicted_set)) if predicted_set else 0.0
    recall = (overlap / len(gold_set)) if gold_set else 0.0
    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)
    exact_match = 1.0 if predicted_set == gold_set else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "em": exact_match,
    }


def _mean_metric(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(mean(float(row[key]) for row in rows))


def summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    qa_block = dict(report.get("setwise_selector_qa", {}) or {})
    selector_summary = dict(qa_block.get("selector_summary", {}) or {})
    traces = list(report.get("setwise_selector_query_traces", []) or [])
    per_query: List[Dict[str, Any]] = []
    delivery_loss_reason_counts: Counter[str] = Counter()
    delivery_loss_questions: List[str] = []

    for trace in traces:
        selector_trace = dict(trace.get("selector_trace", {}) or {})
        gate_decision = dict(selector_trace.get("gate_decision", {}) or {})
        selected_metrics = compute_support_metrics(
            selector_trace.get("selected_titles", []) or [],
            trace.get("gold_titles", []) or [],
        )
        front_metrics = compute_support_metrics(
            selector_trace.get("final_front_titles", []) or trace.get("selector_top_titles", []) or [],
            trace.get("gold_titles", []) or [],
        )
        row = {
            "question": str(trace.get("question", "")),
            "selected_support_f1": float(selected_metrics["f1"]),
            "selected_support_em": float(selected_metrics["em"]),
            "selected_support_recall": float(selected_metrics["recall"]),
            "front_support_f1": float(front_metrics["f1"]),
            "front_support_em": float(front_metrics["em"]),
            "front_support_recall": float(front_metrics["recall"]),
            "selector_qa_f1": float(dict(trace.get("selector_metrics", {}) or {}).get("F1", 0.0) or 0.0),
            "selector_qa_em": float(dict(trace.get("selector_metrics", {}) or {}).get("ExactMatch", 0.0) or 0.0),
            "gate_reason": str(gate_decision.get("reason", "")),
        }
        if row["selected_support_f1"] > row["front_support_f1"] + EPS:
            delivery_loss_reason_counts[row["gate_reason"] or "unknown"] += 1
            delivery_loss_questions.append(row["question"])
        per_query.append(row)

    overall = dict(report.get("overall_recomputed", {}) or {})
    selector_retrieval = dict(qa_block.get("selector_retrieval_metrics", {}) or {})
    return {
        "dataset": str(report.get("dataset", "unknown")),
        "num_queries": len(per_query),
        "qa_em": float(overall.get("ExactMatch", 0.0) or 0.0),
        "qa_f1": float(overall.get("F1", 0.0) or 0.0),
        "selector_em": float(qa_block.get("selector_EM", 0.0) or 0.0),
        "selector_f1": float(qa_block.get("selector_F1", 0.0) or 0.0),
        "selector_recall_at_5": float(selector_retrieval.get("Recall@5", 0.0) or 0.0),
        "selector_recall_at_10": float(selector_retrieval.get("Recall@10", 0.0) or 0.0),
        "selected_support_em": _mean_metric(per_query, "selected_support_em"),
        "selected_support_f1": _mean_metric(per_query, "selected_support_f1"),
        "selected_support_recall": _mean_metric(per_query, "selected_support_recall"),
        "front_support_em": _mean_metric(per_query, "front_support_em"),
        "front_support_f1": _mean_metric(per_query, "front_support_f1"),
        "front_support_recall": _mean_metric(per_query, "front_support_recall"),
        "delivery_gap_f1": _mean_metric(per_query, "selected_support_f1") - _mean_metric(per_query, "front_support_f1"),
        "delivery_loss_query_count": len(delivery_loss_questions),
        "delivery_loss_reason_counts": dict(sorted(delivery_loss_reason_counts.items())),
        "gate_apply_count": int(selector_summary.get("gate_apply_count", 0) or 0),
        "gate_skip_count": int(selector_summary.get("gate_skip_count", 0) or 0),
        "gate_reason_counts": dict(selector_summary.get("gate_reason_counts", {}) or {}),
        "saturation_guard_apply_count": int(selector_summary.get("saturation_guard_apply_count", 0) or 0),
        "saturation_guard_skip_count": int(selector_summary.get("saturation_guard_skip_count", 0) or 0),
        "per_query": per_query,
    }


def compute_deltas(summary: Mapping[str, Dict[str, Any]], reference_label: str) -> Dict[str, Dict[str, float]]:
    reference = dict(summary.get(reference_label, {}) or {})
    metrics = [
        "qa_em",
        "qa_f1",
        "selector_em",
        "selector_f1",
        "selector_recall_at_5",
        "selector_recall_at_10",
        "selected_support_em",
        "selected_support_f1",
        "selected_support_recall",
        "front_support_em",
        "front_support_f1",
        "front_support_recall",
        "delivery_gap_f1",
    ]
    deltas: Dict[str, Dict[str, float]] = {}
    for label, row in summary.items():
        if label == reference_label:
            continue
        deltas[label] = {
            metric: float(row.get(metric, 0.0) or 0.0) - float(reference.get(metric, 0.0) or 0.0)
            for metric in metrics
        }
        deltas[label]["delivery_loss_query_count"] = float(row.get("delivery_loss_query_count", 0) or 0) - float(reference.get("delivery_loss_query_count", 0) or 0)
    return deltas


def select_case_examples(summary: Mapping[str, Dict[str, Any]], reference_label: str, compare_label: str, limit: int) -> Dict[str, List[Dict[str, Any]]]:
    reference_index = {
        str(row.get("question", "")): row
        for row in list((summary.get(reference_label, {}) or {}).get("per_query", []) or [])
    }
    compare_rows = list((summary.get(compare_label, {}) or {}).get("per_query", []) or [])
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in compare_rows:
        question = str(row.get("question", ""))
        base_row = reference_index.get(question)
        if not base_row:
            continue
        selected_delta = float(row.get("selected_support_f1", 0.0) or 0.0) - float(base_row.get("selected_support_f1", 0.0) or 0.0)
        front_delta = float(row.get("front_support_f1", 0.0) or 0.0) - float(base_row.get("front_support_f1", 0.0) or 0.0)
        qa_delta = float(row.get("selector_qa_f1", 0.0) or 0.0) - float(base_row.get("selector_qa_f1", 0.0) or 0.0)
        record = {
            "question": question,
            "gate_reason": str(row.get("gate_reason", "")),
            "selected_support_f1_delta": selected_delta,
            "front_support_f1_delta": front_delta,
            "selector_qa_f1_delta": qa_delta,
        }
        if front_delta > EPS:
            buckets["front_improved"].append(record)
        if qa_delta > EPS:
            buckets["selector_qa_improved"].append(record)
        if selected_delta > EPS and front_delta <= EPS:
            buckets["internal_improved_but_not_delivered"].append(record)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda row: (
                float(row.get("front_support_f1_delta", 0.0)),
                float(row.get("selector_qa_f1_delta", 0.0)),
                row.get("question", ""),
            ),
            reverse=True,
        )
    return {key: value[:limit] for key, value in sorted(buckets.items())}


def build_markdown(analysis: Dict[str, Any], reference_label: str) -> str:
    lines = [f"# Gate Delivery Ablation: {analysis['dataset']}", ""]
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Label | QA EM | QA F1 | Selector EM | Selector F1 | Sel R@5 | Pre-gate support F1 | Post-gate/front support F1 | Delivery gap | Gate apply | Gate skip |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, row in analysis["summaries"].items():
        lines.append(
            f"| `{label}` | {row['qa_em']:.4f} | {row['qa_f1']:.4f} | {row['selector_em']:.4f} | {row['selector_f1']:.4f} | "
            f"{row['selector_recall_at_5']:.4f} | {row['selected_support_f1']:.4f} | {row['front_support_f1']:.4f} | "
            f"{row['delivery_gap_f1']:+.4f} | {int(row['gate_apply_count'])} | {int(row['gate_skip_count'])} |"
        )
    lines.append("")

    lines.append(f"## Deltas vs `{reference_label}`")
    lines.append("")
    lines.append("| Label | QA F1 Δ | Selector F1 Δ | Sel R@5 Δ | Pre-gate support F1 Δ | Post-gate/front support F1 Δ | Delivery gap Δ | Delivery-loss queries Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, row in analysis["deltas_vs_reference"].items():
        lines.append(
            f"| `{label}` | {row['qa_f1']:+.4f} | {row['selector_f1']:+.4f} | {row['selector_recall_at_5']:+.4f} | "
            f"{row['selected_support_f1']:+.4f} | {row['front_support_f1']:+.4f} | {row['delivery_gap_f1']:+.4f} | {row['delivery_loss_query_count']:+.0f} |"
        )
    lines.append("")

    lines.append("## Gate Reasons")
    lines.append("")
    for label, row in analysis["summaries"].items():
        lines.append(f"### `{label}`")
        lines.append("")
        lines.append(f"- Gate reasons: `{json.dumps(row.get('gate_reason_counts', {}), ensure_ascii=False, sort_keys=True)}`")
        lines.append(f"- Delivery-loss reasons: `{json.dumps(row.get('delivery_loss_reason_counts', {}), ensure_ascii=False, sort_keys=True)}`")
        lines.append("")

    lines.append("## Case Examples")
    lines.append("")
    for label, examples in analysis["case_examples"].items():
        lines.append(f"### `{label}` vs `{reference_label}`")
        lines.append("")
        if not examples:
            lines.append("- none")
            lines.append("")
            continue
        for bucket_name, rows in examples.items():
            lines.append(f"- {bucket_name}:")
            for row in rows:
                lines.append(
                    f"  - {row['question']} | gate={row['gate_reason'] or '<none>'} | "
                    f"selected ΔF1={row['selected_support_f1_delta']:+.4f} | "
                    f"front ΔF1={row['front_support_f1_delta']:+.4f} | "
                    f"selector QA ΔF1={row['selector_qa_f1_delta']:+.4f}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze gate-delivery ablation reports.")
    parser.add_argument("--summary_json", required=True, help="Summary JSON emitted by run_gate_delivery_ablation.py")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--reference_label", default="satguard")
    parser.add_argument("--case_limit", type=int, default=5)
    args = parser.parse_args()

    run_summary = load_json(Path(args.summary_json))
    summaries: Dict[str, Dict[str, Any]] = {}

    baseline_report_path = Path(str(run_summary.get("baseline_report", "")).strip()) if str(run_summary.get("baseline_report", "")).strip() else None
    if baseline_report_path:
        summaries["baseline"] = summarize_report(load_json(baseline_report_path))

    for label, meta in dict(run_summary.get("variants", {}) or {}).items():
        report_path = Path(str(meta.get("report", "")))
        summaries[str(label)] = summarize_report(load_json(report_path))

    reference_label = str(args.reference_label)
    if reference_label not in summaries:
        raise ValueError(f"Reference label `{reference_label}` not found in summaries: {sorted(summaries)}")

    analysis = {
        "dataset": str(run_summary.get("dataset", "unknown")),
        "limit": int(run_summary.get("limit", 0) or 0),
        "reference_label": reference_label,
        "summaries": summaries,
        "deltas_vs_reference": compute_deltas(summaries, reference_label=reference_label),
        "case_examples": {
            label: select_case_examples(summaries, reference_label=reference_label, compare_label=label, limit=int(args.case_limit))
            for label in summaries
            if label != reference_label
        },
    }
    Path(args.output_json).write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(build_markdown(analysis, reference_label=reference_label), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
