import argparse
import json
from pathlib import Path


def _truncate(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _format_bool(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def _build_mapping_rows(paired_examples: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for example in paired_examples:
        structure = example.get("structure") or {}
        trace = structure.get("retrieval_trace") or {}
        if trace.get("rerank_facts_empty_stage") != "mapping_failure_drop":
            continue
        rows.append({
            "question": example.get("question", ""),
            "delta_f1": float((example.get("delta") or {}).get("F1", 0.0)),
            "n_candidates": int(trace.get("rerank_n_candidates_initial", 0) or 0),
            "after_rerank_raw": int(trace.get("rerank_n_facts_after_rerank_raw", 0) or 0),
            "after_mapping": int(trace.get("rerank_n_facts_after_mapping", 0) or 0),
            "final_facts": int(trace.get("rerank_n_facts_final", 0) or 0),
            "issue_reasons": dict(trace.get("rerank_mapping_issue_reason_counts") or {}),
            "unmapped": int(trace.get("rerank_mapping_unmapped_count", 0) or 0),
            "fallback_reason": trace.get("rerank_final_non_empty_fallback_reason"),
            "snapshot_hash": trace.get("rerank_candidate_snapshot_hash"),
            "applied": bool(trace.get("applied")),
            "order_changed": bool(trace.get("top5_order_changed")),
        })
    return rows


def _build_prompt_changed_rows(paired_examples: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for example in paired_examples:
        baseline = example.get("baseline") or {}
        structure = example.get("structure") or {}
        baseline_qa = baseline.get("qa_trace") or {}
        structure_qa = structure.get("qa_trace") or {}
        baseline_prompt = baseline_qa.get("reader_prompt_hash")
        structure_prompt = structure_qa.get("reader_prompt_hash")
        if baseline_prompt is None or structure_prompt is None or baseline_prompt == structure_prompt:
            continue
        trace = structure.get("retrieval_trace") or {}
        rows.append({
            "question": example.get("question", ""),
            "delta_em": float((example.get("delta") or {}).get("ExactMatch", 0.0)),
            "delta_f1": float((example.get("delta") or {}).get("F1", 0.0)),
            "applied": bool(trace.get("applied")),
            "effective_applied": bool(trace.get("applied")) and int(trace.get("num_swaps_top5", 0) or 0) > 0,
            "noop_reason": trace.get("noop_reason"),
            "order_changed": bool(trace.get("top5_order_changed")),
            "swaps": int(trace.get("num_swaps_top5", 0) or 0),
            "scored_docs": int(trace.get("scored_doc_count", 0) or 0),
            "bridge_edges": int(trace.get("total_bridge_edge_count", 0) or 0),
            "prompt_hash_changed": True,
            "response_hash_changed": baseline_qa.get("reader_response_hash") != structure_qa.get("reader_response_hash"),
            "cache_pair": f"{baseline_qa.get('reader_cache_hit')}->{structure_qa.get('reader_cache_hit')}",
            "call_pair": f"{baseline_qa.get('reader_call_made')}->{structure_qa.get('reader_call_made')}",
            "baseline_answer": baseline.get("answer", ""),
            "structure_answer": structure.get("answer", ""),
        })
    rows.sort(key=lambda row: (row["delta_f1"], row["delta_em"]))
    return rows


def _markdown_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _build_summary(report: dict,
                   paired_examples: list[dict],
                   mapping_rows: list[dict],
                   prompt_rows: list[dict]) -> str:
    paired_summary = report.get("paired_summary") or {}
    structure_analysis = ((report.get("structure") or {}).get("structure_analysis")) or {}

    proposed_applied = int(structure_analysis.get("structure_enabled_count", 0) or 0)
    effective_applied = int(structure_analysis.get("effective_applied_count", 0) or 0)
    if not effective_applied:
        effective_applied = sum(
            1
            for example in paired_examples
            if bool(((example.get("structure") or {}).get("retrieval_trace") or {}).get("applied"))
            and int((((example.get("structure") or {}).get("retrieval_trace") or {}).get("num_swaps_top5", 0) or 0)) > 0
        )
    prompt_changed = int(paired_summary.get("reader_prompt_changed_count", 0) or len(prompt_rows))
    prompt_improve = sum(1 for row in prompt_rows if row["delta_f1"] > 0 or row["delta_em"] > 0)
    prompt_hurt = sum(1 for row in prompt_rows if row["delta_f1"] < 0 or row["delta_em"] < 0)

    effective_rate = (prompt_changed / proposed_applied) if proposed_applied else 0.0
    benefit_rate = (prompt_improve / prompt_changed) if prompt_changed else 0.0

    lines = [
        f"# Structure Case Export: {Path(report.get('dataset', 'report')).name}",
        "",
        f"- report: `{report.get('dataset')}` / `{report.get('limit')}`",
        f"- delta_em: {float((report.get('delta') or {}).get('ExactMatch', 0.0)):+.6f}",
        f"- delta_f1: {float((report.get('delta') or {}).get('F1', 0.0)):+.6f}",
        f"- proposed_applied_count: {proposed_applied}",
        f"- effective_applied_count: {effective_applied}",
        f"- reader_prompt_changed_count: {prompt_changed}",
        f"- effective_applied_rate: {effective_rate:.4f}",
        f"- benefit_rate: {benefit_rate:.4f}",
        f"- prompt_changed_hurt_count: {prompt_hurt}",
        f"- mapping_failure_count: {len(mapping_rows)}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export mapping-failure and prompt-changed case tables from a struct_compare report."
    )
    parser.add_argument("--report", required=True, help="Path to struct_compare JSON report.")
    parser.add_argument("--output_md", required=True, help="Path to write markdown summary.")
    args = parser.parse_args()

    report_path = Path(args.report)
    output_path = Path(args.output_md)
    report = json.loads(report_path.read_text())
    paired_examples = report.get("paired_examples") or []

    mapping_rows = _build_mapping_rows(paired_examples)
    prompt_rows = _build_prompt_changed_rows(paired_examples)

    sections = [_build_summary(report, paired_examples, mapping_rows, prompt_rows)]

    mapping_table_rows = [
        [
            str(index + 1),
            f"{row['delta_f1']:+.4f}",
            str(row["n_candidates"]),
            str(row["after_rerank_raw"]),
            str(row["after_mapping"]),
            str(row["final_facts"]),
            str(row["unmapped"]),
            json.dumps(row["issue_reasons"], ensure_ascii=False),
            str(row["fallback_reason"] or ""),
            _truncate(row["question"]),
        ]
        for index, row in enumerate(mapping_rows)
    ]
    sections.append(
        _markdown_table(
            "Mapping Failures",
            [
                "#",
                "ΔF1",
                "N Cand",
                "After Raw",
                "After Map",
                "Final",
                "Unmapped",
                "Issue Reasons",
                "Fallback",
                "Question",
            ],
            mapping_table_rows,
        )
    )

    prompt_table_rows = [
        [
            str(index + 1),
            f"{row['delta_em']:+.1f}",
            f"{row['delta_f1']:+.4f}",
            _format_bool(row["applied"]),
            _format_bool(row["effective_applied"]),
            _format_bool(row["order_changed"]),
            str(row["swaps"]),
            row["cache_pair"],
            row["call_pair"],
            _truncate(row["baseline_answer"]),
            _truncate(row["structure_answer"]),
            _truncate(row["question"]),
        ]
        for index, row in enumerate(prompt_rows)
    ]
    sections.append(
        _markdown_table(
            "Prompt Changed Cases",
            [
                "#",
                "ΔEM",
                "ΔF1",
                "Applied",
                "Effective",
                "Order Changed",
                "Swaps",
                "Cache Pair",
                "Call Pair",
                "Baseline Answer",
                "Structure Answer",
                "Question",
            ],
            prompt_table_rows,
        )
    )

    output_path.write_text("\n".join(sections))
    print(json.dumps({
        "report": str(report_path),
        "output_md": str(output_path),
        "mapping_failure_count": len(mapping_rows),
        "prompt_changed_count": len(prompt_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
