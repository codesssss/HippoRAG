#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def fmt(value):
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_primary_metrics(payload: dict) -> tuple[dict, dict]:
    report_metrics = dict(payload.get("report_metrics") or {})
    retrieval = dict(report_metrics.get("primary_retrieval_metrics") or {})
    qa = dict(report_metrics.get("primary_qa_metrics") or {})
    if retrieval or qa:
        return retrieval, qa

    method = dict(payload.get("expand_assemble_qa") or {})
    if method:
        return (
            dict(method.get("method_retrieval_metrics") or {}),
            {
                "ExactMatch": method.get("method_EM"),
                "F1": method.get("method_F1"),
            },
        )

    overall = dict(payload.get("overall_recomputed") or {})
    retrieval = {k: v for k, v in overall.items() if str(k).startswith("Recall@")}
    qa = {k: v for k, v in overall.items() if str(k) in {"ExactMatch", "F1"}}
    return retrieval, qa


def compute_trace_stats(payload: dict) -> dict:
    traces = list(payload.get("expand_assemble_query_traces") or [])
    if not traces:
        return {}

    append_counts = [int((trace.get("expand_assemble_trace") or {}).get("append_count", 0) or 0) for trace in traces]
    stop_reasons = Counter(
        str((trace.get("expand_assemble_trace") or {}).get("append_stop_reason", "missing"))
        for trace in traces
    )
    query_nonempty = 0
    proposal_nonempty = 0
    final_appended = 0
    changed_vs_baseline = 0
    covered_positive = 0
    for trace in traces:
        expand_trace = dict(trace.get("expand_assemble_trace") or {})
        if list(expand_trace.get("query_entities_preview") or []):
            query_nonempty += 1
        if list(expand_trace.get("proposal_query_entities_preview") or []):
            proposal_nonempty += 1
        if float(expand_trace.get("covered_entity_count_after_expand", 0) or 0) > 0:
            covered_positive += 1
        baseline_positions = set(int(pos) for pos in (expand_trace.get("baseline_prefix_positions") or []))
        final_positions = set(int(pos) for pos in (expand_trace.get("final_front_pool_positions") or []))
        if final_positions - baseline_positions:
            final_appended += 1
        if bool(trace.get("changed_from_baseline", False)):
            changed_vs_baseline += 1

    total = max(1, len(traces))
    return {
        "num_queries": len(traces),
        "append_query_rate": round(sum(1 for count in append_counts if count > 0) / total, 4),
        "avg_append_count": round(sum(append_counts) / total, 4),
        "final_appended_rate": round(final_appended / total, 4),
        "changed_vs_baseline_rate": round(changed_vs_baseline / total, 4),
        "query_entities_nonempty_rate": round(query_nonempty / total, 4),
        "proposal_query_entities_nonempty_rate": round(proposal_nonempty / total, 4),
        "covered_entity_positive_rate": round(covered_positive / total, 4),
        "append_stop_reason_top": ", ".join(
            f"{reason}:{count}" for reason, count in stop_reasons.most_common(3)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Build summary for NQ/PopQA bridge diagnosis runs.")
    parser.add_argument("--diag-date-tag", required=True)
    parser.add_argument("--control-date-tag", default="20260410nqpopqa100dual")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = []
    for dataset in ("nq", "popqa"):
        base_dir = Path(f"outputs_step0_general_{dataset}/eval_reports")
        control_paths = [
            ("baseline_top10_plus_ce", base_dir / f"width_match_baseline_top10_plus_ce_qatopk5_{args.control_date_tag}.json"),
            ("bridge_append_plus_ce_default", base_dir / f"width_match_bridge_append_plus_ce_qatopk5_{args.control_date_tag}.json"),
        ]
        for run_name, path in control_paths:
            if not path.exists():
                continue
            payload = load_json(path)
            retrieval, qa = extract_primary_metrics(payload)
            trace_stats = compute_trace_stats(payload)
            rows.append({
                "dataset": dataset,
                "run": run_name,
                "query_source": str((payload.get("config") or {}).get("setwise_query_entity_source", "—")),
                "threshold": (payload.get("config") or {}).get("expand_min_structure_score"),
                "em": qa.get("ExactMatch"),
                "f1": qa.get("F1"),
                "r5": retrieval.get("Recall@5"),
                "r20": retrieval.get("Recall@20"),
                "r100": retrieval.get("Recall@100"),
                **trace_stats,
            })

        for path in sorted(base_dir.glob(f"bridge_diag_*_{args.diag_date_tag}.json")):
            payload = load_json(path)
            retrieval, qa = extract_primary_metrics(payload)
            trace_stats = compute_trace_stats(payload)
            rows.append({
                "dataset": dataset,
                "run": path.stem.replace(f"_{args.diag_date_tag}", ""),
                "query_source": str((payload.get("config") or {}).get("setwise_query_entity_source", "—")),
                "threshold": (payload.get("config") or {}).get("expand_min_structure_score"),
                "em": qa.get("ExactMatch"),
                "f1": qa.get("F1"),
                "r5": retrieval.get("Recall@5"),
                "r20": retrieval.get("Recall@20"),
                "r100": retrieval.get("Recall@100"),
                **trace_stats,
            })

    output_path = Path(f"run_logs/bridge_failure_diagnosis_nq_popqa_100_{args.diag_date_tag}.summary.md")
    lines = [
        "# Bridge Failure Diagnosis: NQ + PopQA (limit=100)",
        "",
        f"- control date tag: `{args.control_date_tag}`",
        f"- diagnosis date tag: `{args.diag_date_tag}`",
        "- `baseline_top10_plus_ce` is the no-append width-matched CE control.",
        "- `bridge_append_plus_ce_default` is the original bridge setting from the previous 100-query run.",
        "- `append_query_rate` = share of queries with at least one appended bridge doc.",
        "- `final_appended_rate` = share of queries whose final top-5 contains at least one appended doc.",
        "",
        "| Dataset | Run | q_source | thr | EM | F1 | R@5 | R@20 | R@100 | append_q_rate | avg_append | final_app_q_rate | changed_vs_base | query_ent_rate | proposal_ent_rate | covered_pos_rate | top_stop_reasons |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['run']} | {row.get('query_source', '—')} | {fmt(row.get('threshold'))} | "
            f"{fmt(row.get('em'))} | {fmt(row.get('f1'))} | {fmt(row.get('r5'))} | {fmt(row.get('r20'))} | {fmt(row.get('r100'))} | "
            f"{fmt(row.get('append_query_rate'))} | {fmt(row.get('avg_append_count'))} | {fmt(row.get('final_appended_rate'))} | "
            f"{fmt(row.get('changed_vs_baseline_rate'))} | {fmt(row.get('query_entities_nonempty_rate'))} | "
            f"{fmt(row.get('proposal_query_entities_nonempty_rate'))} | {fmt(row.get('covered_entity_positive_rate'))} | "
            f"{row.get('append_stop_reason_top', '—')} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
