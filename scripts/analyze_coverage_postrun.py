#!/usr/bin/env python3
"""Post-run diagnostics for coverage assembly experiments.

This script is intentionally narrow. It answers three questions:

1. On the same append=0 candidate pool, how often does coverage beat / tie / lose
   against cross-encoder assembly?
2. How different are the selected subsets (coverage-vs-CE overlap and
   coverage-vs-baseline overlap), and what are representative cases?
3. When moving from append=0 to append=3 under coverage assembly, how often do
   appended docs get selected, and how often does that hurt relative to append=0?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
QA_TOP_KS = [5, 7, 10]


def _round(value: Any, ndigits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    numeric = [float(v) for v in values]
    if not numeric:
        return None
    return round(mean(numeric), 4)


def _slug(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def report_path(dataset: str, filename: str) -> Path:
    return ROOT_DIR / f"outputs_step0_general_{dataset}" / "eval_reports" / filename


def load_report_bundle(
    dataset: str,
    qa_top_k: int,
    baseline_date_tag: str,
    new_date_tag: str,
) -> Dict[str, Optional[Dict[str, Any]]]:
    return {
        "baseline": load_json(
            report_path(
                dataset,
                f"causal_eval_{dataset}_100_qwen3-8b_qatopk{qa_top_k}_baseline_{baseline_date_tag}.json",
            )
        ),
        "append0_ce": load_json(
            report_path(
                dataset,
                f"bridge_append_cross_encoder_pool100_base10_append0_qatopk{qa_top_k}_{new_date_tag}.json",
            )
        ),
        "append0_cov": load_json(
            report_path(
                dataset,
                f"bridge_append_coverage_pool100_base10_append0_qatopk{qa_top_k}_{new_date_tag}.json",
            )
        ),
        "append3_cov": load_json(
            report_path(
                dataset,
                f"bridge_append_coverage_pool100_base10_append3_qatopk{qa_top_k}_{new_date_tag}.json",
            )
        ),
    }


def get_query_traces(report: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if report is None:
        return []
    return list(report.get("expand_assemble_query_traces", []) or [])


def get_overall_metrics(report: Optional[Dict[str, Any]], kind: str) -> Tuple[Optional[float], Optional[float]]:
    if report is None:
        return None, None
    if kind == "baseline":
        overall = dict(report.get("overall_recomputed", {}) or {})
        return _round(overall.get("ExactMatch")), _round(overall.get("F1"))
    summary = dict(report.get("expand_assemble_qa", {}) or {})
    return _round(summary.get("method_EM")), _round(summary.get("method_F1"))


def compare_traces(
    left_traces: Sequence[Dict[str, Any]],
    right_traces: Sequence[Dict[str, Any]],
    left_name: str,
    right_name: str,
) -> Dict[str, Any]:
    if not left_traces or not right_traces:
        return {"available": False}
    if len(left_traces) != len(right_traces):
        raise ValueError(f"{left_name} and {right_name} traces have different lengths.")

    em_gain = 0
    em_tie = 0
    em_loss = 0
    f1_gain = 0
    f1_tie = 0
    f1_loss = 0
    delta_em_values: List[float] = []
    delta_f1_values: List[float] = []
    winners: List[Dict[str, Any]] = []
    losers: List[Dict[str, Any]] = []

    for left_trace, right_trace in zip(left_traces, right_traces):
        question = str(left_trace.get("question", ""))
        left_metrics = dict(left_trace.get("method_metrics", {}) or {})
        right_metrics = dict(right_trace.get("method_metrics", {}) or {})
        delta_em = float(left_metrics.get("ExactMatch", 0.0) or 0.0) - float(right_metrics.get("ExactMatch", 0.0) or 0.0)
        delta_f1 = float(left_metrics.get("F1", 0.0) or 0.0) - float(right_metrics.get("F1", 0.0) or 0.0)
        delta_em_values.append(delta_em)
        delta_f1_values.append(delta_f1)

        if delta_em > 1e-9:
            em_gain += 1
        elif delta_em < -1e-9:
            em_loss += 1
        else:
            em_tie += 1

        if delta_f1 > 1e-9:
            f1_gain += 1
        elif delta_f1 < -1e-9:
            f1_loss += 1
        else:
            f1_tie += 1

        record = {
            "question": question,
            "delta_em": round(delta_em, 4),
            "delta_f1": round(delta_f1, 4),
            "left_titles": list(left_trace.get("method_top_titles", []) or []),
            "right_titles": list(right_trace.get("method_top_titles", []) or []),
        }
        if delta_f1 > 1e-9 or delta_em > 1e-9:
            winners.append(record)
        elif delta_f1 < -1e-9 or delta_em < -1e-9:
            losers.append(record)

    winners.sort(key=lambda row: (-row["delta_f1"], -row["delta_em"], row["question"]))
    losers.sort(key=lambda row: (row["delta_f1"], row["delta_em"], row["question"]))

    return {
        "available": True,
        "pair": f"{left_name}_vs_{right_name}",
        "query_count": len(left_traces),
        "em": {"gain": em_gain, "tie": em_tie, "loss": em_loss, "avg_delta": _safe_mean(delta_em_values)},
        "f1": {"gain": f1_gain, "tie": f1_tie, "loss": f1_loss, "avg_delta": _safe_mean(delta_f1_values)},
        "top_winners": winners[:10],
        "top_losers": losers[:10],
    }


def summarize_coverage_trace(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    traces = get_query_traces(report)
    if not traces:
        return {"available": False}

    overlap_ce: List[float] = []
    overlap_baseline: List[float] = []
    num_appended_selected: List[float] = []
    num_baseline_selected: List[float] = []
    changed_from_baseline = 0
    same_as_ce = 0
    same_as_baseline = 0
    examples_far_from_ce: List[Dict[str, Any]] = []

    for trace in traces:
        if bool(trace.get("changed_from_baseline", False)):
            changed_from_baseline += 1
        expand_trace = dict(trace.get("expand_assemble_trace", {}) or {})
        assemble_trace = dict(expand_trace.get("assemble_trace", {}) or {})
        overlap_ce_value = float(assemble_trace.get("coverage_vs_ce_overlap", 0) or 0)
        overlap_baseline_value = float(assemble_trace.get("coverage_vs_baseline_overlap", 0) or 0)
        appended_selected_value = float(assemble_trace.get("num_appended_selected", 0) or 0)
        baseline_selected_value = float(assemble_trace.get("num_baseline_selected", 0) or 0)
        overlap_ce.append(overlap_ce_value)
        overlap_baseline.append(overlap_baseline_value)
        num_appended_selected.append(appended_selected_value)
        num_baseline_selected.append(baseline_selected_value)
        if overlap_ce_value >= 4.999:
            same_as_ce += 1
        if overlap_baseline_value >= 4.999:
            same_as_baseline += 1

        examples_far_from_ce.append(
            {
                "question": str(trace.get("question", "")),
                "coverage_vs_ce_overlap": int(overlap_ce_value),
                "coverage_vs_baseline_overlap": int(overlap_baseline_value),
                "num_appended_selected": int(appended_selected_value),
                "method_f1": _round((trace.get("method_metrics", {}) or {}).get("F1")),
                "method_titles": list(trace.get("method_top_titles", []) or []),
                "best_score": dict(assemble_trace.get("best_score", {}) or {}),
            }
        )

    examples_far_from_ce.sort(
        key=lambda row: (
            int(row["coverage_vs_ce_overlap"]),
            -float(row["method_f1"] or 0.0),
            row["question"],
        )
    )

    return {
        "available": True,
        "query_count": len(traces),
        "changed_from_baseline_count": int(changed_from_baseline),
        "same_as_ce_count": int(same_as_ce),
        "same_as_baseline_count": int(same_as_baseline),
        "avg_overlap_ce": _safe_mean(overlap_ce),
        "avg_overlap_baseline": _safe_mean(overlap_baseline),
        "avg_num_appended_selected": _safe_mean(num_appended_selected),
        "avg_num_baseline_selected": _safe_mean(num_baseline_selected),
        "low_overlap_examples": examples_far_from_ce[:10],
    }


def analyze_append_effect(
    append0_cov_report: Optional[Dict[str, Any]],
    append3_cov_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    append0_traces = get_query_traces(append0_cov_report)
    append3_traces = get_query_traces(append3_cov_report)
    if not append0_traces or not append3_traces:
        return {"available": False}
    if len(append0_traces) != len(append3_traces):
        raise ValueError("append0 coverage and append3 coverage traces have different lengths.")

    paired = compare_traces(append3_traces, append0_traces, "append3_coverage", "append0_coverage")

    appended_selected_when_worse = 0
    appended_selected_total = 0
    hurt_cases: List[Dict[str, Any]] = []
    for append0_trace, append3_trace in zip(append0_traces, append3_traces):
        append3_expand = dict(append3_trace.get("expand_assemble_trace", {}) or {})
        assemble_trace = dict(append3_expand.get("assemble_trace", {}) or {})
        num_appended_selected = int(assemble_trace.get("num_appended_selected", 0) or 0)
        if num_appended_selected > 0:
            appended_selected_total += 1

        append3_f1 = float((append3_trace.get("method_metrics", {}) or {}).get("F1", 0.0) or 0.0)
        append0_f1 = float((append0_trace.get("method_metrics", {}) or {}).get("F1", 0.0) or 0.0)
        delta_f1 = append3_f1 - append0_f1
        if num_appended_selected > 0 and delta_f1 < -1e-9:
            appended_selected_when_worse += 1

        if delta_f1 < -1e-9:
            hurt_cases.append(
                {
                    "question": str(append3_trace.get("question", "")),
                    "delta_f1": round(delta_f1, 4),
                    "append3_em": _round((append3_trace.get("method_metrics", {}) or {}).get("ExactMatch")),
                    "append0_em": _round((append0_trace.get("method_metrics", {}) or {}).get("ExactMatch")),
                    "num_appended_selected": num_appended_selected,
                    "coverage_vs_baseline_overlap": int(assemble_trace.get("coverage_vs_baseline_overlap", 0) or 0),
                    "append3_titles": list(append3_trace.get("method_top_titles", []) or []),
                    "append0_titles": list(append0_trace.get("method_top_titles", []) or []),
                    "appended_titles": list(append3_expand.get("appended_titles", []) or []),
                }
            )
    hurt_cases.sort(key=lambda row: (row["delta_f1"], row["question"]))

    return {
        "available": True,
        "paired_comparison": paired,
        "queries_with_appended_docs_selected": int(appended_selected_total),
        "queries_where_appended_docs_selected_and_hurt_vs_append0": int(appended_selected_when_worse),
        "top_hurt_cases": hurt_cases[:10],
    }


def build_result_for_setting(
    dataset: str,
    qa_top_k: int,
    baseline_date_tag: str,
    new_date_tag: str,
) -> Dict[str, Any]:
    reports = load_report_bundle(
        dataset=dataset,
        qa_top_k=qa_top_k,
        baseline_date_tag=baseline_date_tag,
        new_date_tag=new_date_tag,
    )
    baseline_em, baseline_f1 = get_overall_metrics(reports["baseline"], "baseline")
    append0_ce_em, append0_ce_f1 = get_overall_metrics(reports["append0_ce"], "method")
    append0_cov_em, append0_cov_f1 = get_overall_metrics(reports["append0_cov"], "method")
    append3_cov_em, append3_cov_f1 = get_overall_metrics(reports["append3_cov"], "method")

    return {
        "dataset": dataset,
        "qa_top_k": qa_top_k,
        "overall": {
            "baseline": {"em": baseline_em, "f1": baseline_f1},
            "append0_cross_encoder": {"em": append0_ce_em, "f1": append0_ce_f1},
            "append0_coverage": {"em": append0_cov_em, "f1": append0_cov_f1},
            "append3_coverage": {"em": append3_cov_em, "f1": append3_cov_f1},
        },
        "coverage_vs_ce_append0": compare_traces(
            get_query_traces(reports["append0_cov"]),
            get_query_traces(reports["append0_ce"]),
            "append0_coverage",
            "append0_cross_encoder",
        ),
        "coverage_trace_append0": summarize_coverage_trace(reports["append0_cov"]),
        "coverage_trace_append3": summarize_coverage_trace(reports["append3_cov"]),
        "append_effect_under_coverage": analyze_append_effect(
            reports["append0_cov"],
            reports["append3_cov"],
        ),
        "report_paths": {
            key: (
                None
                if report is None
                else str(
                    report_path(
                        dataset,
                        (
                            f"causal_eval_{dataset}_100_qwen3-8b_qatopk{qa_top_k}_baseline_{baseline_date_tag}.json"
                            if key == "baseline"
                            else (
                                f"bridge_append_cross_encoder_pool100_base10_append0_qatopk{qa_top_k}_{new_date_tag}.json"
                                if key == "append0_ce"
                                else (
                                    f"bridge_append_coverage_pool100_base10_append0_qatopk{qa_top_k}_{new_date_tag}.json"
                                    if key == "append0_cov"
                                    else f"bridge_append_coverage_pool100_base10_append3_qatopk{qa_top_k}_{new_date_tag}.json"
                                )
                            )
                        ),
                    )
                )
            )
            for key, report in reports.items()
        },
    }


def build_markdown(results: Dict[str, Any]) -> str:
    def fmt_metric_pair(payload: Dict[str, Any]) -> str:
        em = payload.get("em")
        f1 = payload.get("f1")
        if em is None or f1 is None:
            return "—"
        return f"{float(em):.2f}/{float(f1):.4f}"

    def fmt_float(value: Any, ndigits: int = 4) -> str:
        if value is None:
            return "—"
        return f"{float(value):.{ndigits}f}"

    lines: List[str] = []
    lines.append("# Coverage Post-Run Diagnostics")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("- Focus 1: append=0 same-pool coverage vs cross-encoder.")
    lines.append("- Focus 2: coverage-vs-CE / coverage-vs-baseline overlap behavior.")
    lines.append("- Focus 3: append=3 under coverage versus append=0 under coverage.")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append("| Dataset | K | Baseline EM/F1 | append0+CE EM/F1 | append0+Cov EM/F1 | append3+Cov EM/F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in results["settings"]:
        overall = row["overall"]
        lines.append(
            f"| {row['dataset']} | {row['qa_top_k']} | "
            f"{fmt_metric_pair(overall['baseline'])} | "
            f"{fmt_metric_pair(overall['append0_cross_encoder'])} | "
            f"{fmt_metric_pair(overall['append0_coverage'])} | "
            f"{fmt_metric_pair(overall['append3_coverage'])} |"
        )
    lines.append("")

    lines.append("## Paired Gain/Tie/Loss")
    lines.append("")
    lines.append("| Dataset | K | Cov vs CE EM G/T/L | Cov vs CE F1 G/T/L | Avg ΔEM | Avg ΔF1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in results["settings"]:
        paired = row["coverage_vs_ce_append0"]
        if not paired.get("available"):
            continue
        lines.append(
            f"| {row['dataset']} | {row['qa_top_k']} | "
            f"{paired['em']['gain']}/{paired['em']['tie']}/{paired['em']['loss']} | "
            f"{paired['f1']['gain']}/{paired['f1']['tie']}/{paired['f1']['loss']} | "
            f"{fmt_float(paired['em']['avg_delta'], 4)} | {fmt_float(paired['f1']['avg_delta'], 4)} |"
        )
    lines.append("")

    lines.append("## Overlap")
    lines.append("")
    lines.append("| Dataset | K | append0 avg overlap CE | append0 avg overlap baseline | append0 changed | append3 avg overlap baseline | append3 avg appended selected |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in results["settings"]:
        trace0 = row["coverage_trace_append0"]
        trace3 = row["coverage_trace_append3"]
        if not trace0.get("available"):
            continue
        lines.append(
            f"| {row['dataset']} | {row['qa_top_k']} | "
            f"{float(trace0['avg_overlap_ce'] or 0.0):.2f} | "
            f"{float(trace0['avg_overlap_baseline'] or 0.0):.2f} | "
            f"{int(trace0['changed_from_baseline_count'])}/{int(trace0['query_count'])} | "
            f"{float(trace3.get('avg_overlap_baseline') or 0.0):.2f} | "
            f"{float(trace3.get('avg_num_appended_selected') or 0.0):.2f} |"
        )
    lines.append("")

    lines.append("## Append Effect")
    lines.append("")
    lines.append("| Dataset | K | append3 vs append0 EM G/T/L | append3 vs append0 F1 G/T/L | queries with appended selected | appended selected and hurt |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in results["settings"]:
        effect = row["append_effect_under_coverage"]
        paired = effect.get("paired_comparison", {})
        if not effect.get("available") or not paired.get("available"):
            continue
        lines.append(
            f"| {row['dataset']} | {row['qa_top_k']} | "
            f"{paired['em']['gain']}/{paired['em']['tie']}/{paired['em']['loss']} | "
            f"{paired['f1']['gain']}/{paired['f1']['tie']}/{paired['f1']['loss']} | "
            f"{effect['queries_with_appended_docs_selected']} | "
            f"{effect['queries_where_appended_docs_selected_and_hurt_vs_append0']} |"
        )
    lines.append("")

    lines.append("## Representative Failures")
    lines.append("")
    for row in results["settings"]:
        effect = row["append_effect_under_coverage"]
        hurt_cases = list(effect.get("top_hurt_cases", []) or [])
        if not hurt_cases:
            continue
        lines.append(f"### {row['dataset']} K={row['qa_top_k']}")
        lines.append("")
        lines.append("| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |")
        lines.append("|---:|---:|---:|---|")
        for case in hurt_cases[:5]:
            lines.append(
                f"| {case['delta_f1']:+.4f} | {case['num_appended_selected']} | "
                f"{case['coverage_vs_baseline_overlap']} | {case['question']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc diagnostics for coverage assembly experiments.")
    parser.add_argument("--baseline_date_tag", type=str, default="20260406")
    parser.add_argument("--new_date_tag", type=str, default="20260407")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    settings = [
        build_result_for_setting(
            dataset=dataset,
            qa_top_k=qa_top_k,
            baseline_date_tag=str(args.baseline_date_tag),
            new_date_tag=str(args.new_date_tag),
        )
        for dataset in DATASETS
        for qa_top_k in QA_TOP_KS
    ]
    result = {
        "baseline_date_tag": str(args.baseline_date_tag),
        "new_date_tag": str(args.new_date_tag),
        "settings": settings,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(build_markdown(result), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
