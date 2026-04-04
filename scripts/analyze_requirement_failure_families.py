#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent

from analyze_requirement_beam_report import build_runtime_context  # noqa: E402
from eval_causal_qwen3 import build_requirement_title_exposure_summary  # noqa: E402


def classify_query_family(exposure_rows: List[Dict[str, Any]], selector_metrics: Dict[str, Any]) -> List[str]:
    stage_counts = Counter(str(row.get("stage", "")) for row in exposure_rows)
    families: List[str] = []

    if stage_counts.get("not_in_pool", 0) > 0:
        families.append("pool_coverage_gap")
    if stage_counts.get("pool_only", 0) > 0:
        families.append("pool_to_source_gap")
    if stage_counts.get("source", 0) > 0:
        families.append("source_to_shortlist_gap")
    if stage_counts.get("shortlist", 0) > 0:
        families.append("shortlist_to_selected_gap")
    if (
        stage_counts.get("selected", 0) + stage_counts.get("final_only", 0) > 0
        and float(selector_metrics.get("ExactMatch", 0.0) or 0.0) < 1.0
    ):
        families.append("final_selected_but_answer_incomplete")
    if (
        stage_counts.get("not_in_pool", 0) == 0
        and stage_counts.get("selected", 0) + stage_counts.get("final_only", 0) > 0
        and stage_counts.get("pool_only", 0) + stage_counts.get("source", 0) + stage_counts.get("shortlist", 0) > 0
    ):
        families.append("partial_chain_closure_candidate")
    return families


def summarize_query_trace(query_trace: Dict[str, Any], pool_titles: List[str]) -> Dict[str, Any]:
    gold_titles = [str(title).strip() for title in query_trace.get("gold_titles", []) if str(title).strip()]
    exposure_rows = build_requirement_title_exposure_summary(
        pool_titles=pool_titles,
        selector_trace=query_trace.get("selector_trace"),
        target_titles=gold_titles,
    )
    selector_metrics = query_trace.get("selector_metrics", {}) or {}
    families = classify_query_family(exposure_rows, selector_metrics)
    stage_counts = Counter(str(row.get("stage", "")) for row in exposure_rows)
    return {
        "question": query_trace.get("question"),
        "gold_titles": gold_titles,
        "gold_count": len(gold_titles),
        "selector_answer": query_trace.get("selector_answer"),
        "selector_metrics": selector_metrics,
        "selected_titles": list((query_trace.get("selector_trace") or {}).get("selected_titles", []) or []),
        "stage_counts": dict(stage_counts),
        "families": families,
        "gold_exposure_rows": exposure_rows,
    }


def build_markdown(summary: Dict[str, Any]) -> str:
    lines = ["# Requirement Failure-Family Scan", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- report: `{summary['report_path']}`")
    lines.append(f"- dataset: `{summary['dataset']}`")
    lines.append(f"- num_queries: `{summary['num_queries']}`")
    lines.append("")
    lines.append("## Family Counts")
    lines.append("")
    for family, count in summary["family_counts"]:
        lines.append(f"- `{family}`: {count}")
    lines.append("")
    lines.append("## Partial-Closure Candidates")
    lines.append("")
    candidates = [row for row in summary["query_rows"] if "partial_chain_closure_candidate" in row["families"]]
    if not candidates:
        lines.append("- none")
    else:
        for row in candidates[:10]:
            lines.append(f"- question: {row['question']}")
            lines.append(f"  gold: {row['gold_titles']}")
            lines.append(f"  stages: {row['stage_counts']}")
            lines.append(f"  answer: {row['selector_answer']} / metrics={row['selector_metrics']}")
    lines.append("")
    lines.append("## Pool->Source Gap Candidates")
    lines.append("")
    gap_rows = [row for row in summary["query_rows"] if "pool_to_source_gap" in row["families"]]
    if not gap_rows:
        lines.append("- none")
    else:
        for row in gap_rows[:10]:
            pool_only_titles = [
                exposure["title"]
                for exposure in row["gold_exposure_rows"]
                if exposure.get("stage") == "pool_only"
            ]
            lines.append(f"- question: {row['question']}")
            lines.append(f"  pool_only_gold: {pool_only_titles}")
            lines.append(f"  selected: {row['selected_titles']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan requirement-beam reports for recurring failure families.")
    parser.add_argument("--report", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--save_dir_root", type=str, default="outputs_step0_general")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--output_md", type=str, default="")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.load(report_path.open())
    runtime_context = build_runtime_context(
        report=report,
        dataset=str(args.dataset),
        limit=int(args.limit),
        save_dir_root=str(args.save_dir_root),
    )

    query_rows: List[Dict[str, Any]] = []
    runtime_by_question = runtime_context["runtime_by_question"]
    for query_trace in report.get("setwise_selector_query_traces", []) or []:
        question = str(query_trace.get("question", "")).strip()
        runtime_row = runtime_by_question.get(question)
        if runtime_row is None:
            continue
        query_rows.append(
            summarize_query_trace(
                query_trace=query_trace,
                pool_titles=list(runtime_row.get("pool_titles", []) or []),
            )
        )

    family_counts = Counter()
    for row in query_rows:
        for family in row["families"]:
            family_counts[family] += 1

    summary = {
        "report_path": str(report_path),
        "dataset": str(args.dataset),
        "num_queries": len(query_rows),
        "family_counts": family_counts.most_common(),
        "query_rows": query_rows,
    }

    output_json = Path(args.output_json) if args.output_json else report_path.with_suffix(".failure_families.json")
    output_md = Path(args.output_md) if args.output_md else report_path.with_suffix(".failure_families.md")
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    output_md.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "family_counts": summary["family_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
