#!/usr/bin/env python3
"""Verify integrated PCEC output against a reference top4/max1 artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evidence_transition_graphragv4_composition.readout import (  # noqa: E402
    baseline_titles_from_trace,
    final_titles_from_trace,
    normalize_title,
    read_json,
    selected_query_traces,
    selector_trace_from_query_trace,
    swap_steps,
    title_all_covered,
    title_counter,
    title_recall,
    write_json,
)


def records_as_traces(payload: Mapping[str, Any], *, qa_top_k: int) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        return []
    traces: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        titles = [str(item) for item in list(record.get("pool_titles") or [])[:qa_top_k]]
        traces.append(
            {
                "question": str(record.get("question") or ""),
                "baseline_top_titles": titles,
                "selector_top_titles": titles,
                "gold_titles": list(record.get("gold_titles") or []),
                "selector_trace": {},
            }
        )
    return traces


def extract_rows(payload: Mapping[str, Any], *, qa_top_k: int) -> list[dict[str, Any]]:
    traces = selected_query_traces(payload)
    if not traces:
        traces = records_as_traces(payload, qa_top_k=qa_top_k)
    rows: list[dict[str, Any]] = []
    for idx, trace in enumerate(traces):
        final_titles = final_titles_from_trace(trace)[:qa_top_k]
        baseline_titles = baseline_titles_from_trace(trace)[:qa_top_k]
        if not baseline_titles:
            baseline_titles = final_titles
        selector_trace = selector_trace_from_query_trace(trace)
        rows.append(
            {
                "idx": idx,
                "question": str(trace.get("question") or ""),
                "gold_titles": list(trace.get("gold_titles") or []),
                "baseline_titles": baseline_titles,
                "final_titles": final_titles,
                "swap_steps": swap_steps(selector_trace),
                "selector_trace": selector_trace,
            }
        )
    return rows


def count_gold_swaps(row: Mapping[str, Any]) -> tuple[int, int]:
    gold_norm = title_counter(list(row.get("gold_titles") or []))
    swapped_in = 0
    swapped_out = 0
    for step in list(row.get("swap_steps") or []):
        if gold_norm.get(normalize_title(step.get("in_title")), 0) > 0:
            swapped_in += 1
        if gold_norm.get(normalize_title(step.get("out_title")), 0) > 0:
            swapped_out += 1
    return swapped_in, swapped_out


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, qa_top_k: int) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": int(len(rows)),
        "changed_count": int(sum(list(row.get("baseline_titles") or []) != list(row.get("final_titles") or []) for row in rows)),
        "total_swaps": int(sum(len(list(row.get("swap_steps") or [])) for row in rows)),
        "queries_swapped_in_gold": int(sum(count_gold_swaps(row)[0] > 0 for row in rows)),
        "queries_swapped_out_gold": int(sum(count_gold_swaps(row)[1] > 0 for row in rows)),
        "baseline_title_recall_top5": round(
            mean(title_recall(list(row.get("gold_titles") or []), list(row.get("baseline_titles") or []), k=qa_top_k) for row in rows),
            4,
        ),
        "final_title_recall_top5": round(
            mean(title_recall(list(row.get("gold_titles") or []), list(row.get("final_titles") or []), k=qa_top_k) for row in rows),
            4,
        ),
        "baseline_title_all_gold_top5": round(
            sum(title_all_covered(list(row.get("gold_titles") or []), list(row.get("baseline_titles") or []), k=qa_top_k) for row in rows)
            / len(rows),
            4,
        ),
        "final_title_all_gold_top5": round(
            sum(title_all_covered(list(row.get("gold_titles") or []), list(row.get("final_titles") or []), k=qa_top_k) for row in rows)
            / len(rows),
            4,
        ),
    }


def compare_rows(
    pcec_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    qa_top_k: int,
    max_examples: int,
) -> dict[str, Any]:
    count = min(len(pcec_rows), len(reference_rows))
    mismatches: list[dict[str, Any]] = []
    question_mismatch_count = 0
    for idx in range(count):
        pcec = pcec_rows[idx]
        reference = reference_rows[idx]
        if str(pcec.get("question") or "") != str(reference.get("question") or ""):
            question_mismatch_count += 1
        pcec_titles = list(pcec.get("final_titles") or [])[:qa_top_k]
        reference_titles = list(reference.get("final_titles") or [])[:qa_top_k]
        if pcec_titles != reference_titles and len(mismatches) < max_examples:
            mismatches.append(
                {
                    "idx": int(idx),
                    "question": pcec.get("question") or reference.get("question") or "",
                    "pcec_titles": pcec_titles,
                    "reference_titles": reference_titles,
                }
            )
    exact_match_count = sum(
        list(pcec_rows[idx].get("final_titles") or [])[:qa_top_k]
        == list(reference_rows[idx].get("final_titles") or [])[:qa_top_k]
        for idx in range(count)
    )
    return {
        "count_compared": int(count),
        "pcec_count": int(len(pcec_rows)),
        "reference_count": int(len(reference_rows)),
        "question_mismatch_count": int(question_mismatch_count),
        "exact_topk_match_count": int(exact_match_count),
        "exact_topk_match_rate": round(float(exact_match_count) / float(max(count, 1)), 6),
        "mismatch_count": int(count - exact_match_count),
        "mismatch_examples": mismatches,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    comparison = dict(payload.get("comparison") or {})
    pcec_summary = dict(payload.get("pcec_summary") or {})
    reference_summary = dict(payload.get("reference_summary") or {})
    lines = [
        "# PCEC Parity Verification",
        "",
        f"Dataset: `{payload.get('dataset') or ''}`",
        "",
        "## Exact Top-K Parity",
        "",
        "| Compared | Exact matches | Mismatches | Match rate | Question mismatches |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| {count} | {match} | {mismatch} | {rate:.6f} | {qmis} |".format(
            count=int(comparison.get("count_compared", 0)),
            match=int(comparison.get("exact_topk_match_count", 0)),
            mismatch=int(comparison.get("mismatch_count", 0)),
            rate=float(comparison.get("exact_topk_match_rate", 0.0)),
            qmis=int(comparison.get("question_mismatch_count", 0)),
        ),
        "",
        "## Metric Summary",
        "",
        "| Artifact | Changed | Swaps | Gold-out queries | Title-all@5 | Title recall@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| PCEC | {changed} | {swaps} | {gold_out} | {all_gold:.4f} | {recall:.4f} |".format(
            changed=int(pcec_summary.get("changed_count", 0)),
            swaps=int(pcec_summary.get("total_swaps", 0)),
            gold_out=int(pcec_summary.get("queries_swapped_out_gold", 0)),
            all_gold=float(pcec_summary.get("final_title_all_gold_top5", 0.0)),
            recall=float(pcec_summary.get("final_title_recall_top5", 0.0)),
        ),
        "| Reference | {changed} | {swaps} | {gold_out} | {all_gold:.4f} | {recall:.4f} |".format(
            changed=int(reference_summary.get("changed_count", 0)),
            swaps=int(reference_summary.get("total_swaps", 0)),
            gold_out=int(reference_summary.get("queries_swapped_out_gold", 0)),
            all_gold=float(reference_summary.get("final_title_all_gold_top5", 0.0)),
            recall=float(reference_summary.get("final_title_recall_top5", 0.0)),
        ),
    ]
    mismatches = list(comparison.get("mismatch_examples") or [])
    if mismatches:
        lines.extend(["", "## Mismatch Examples", ""])
        for item in mismatches:
            lines.append(f"- `{item.get('idx')}` {item.get('question')}")
            lines.append(f"  - PCEC: {item.get('pcec_titles')}")
            lines.append(f"  - Reference: {item.get('reference_titles')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcec-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    pcec_payload = read_json(args.pcec_json)
    reference_payload = read_json(args.reference_json)
    pcec_rows = extract_rows(pcec_payload, qa_top_k=int(args.qa_top_k))
    reference_rows = extract_rows(reference_payload, qa_top_k=int(args.qa_top_k))
    output = {
        "dataset": str(args.dataset or pcec_payload.get("dataset") or reference_payload.get("dataset") or ""),
        "pcec_json": str(args.pcec_json),
        "reference_json": str(args.reference_json),
        "qa_top_k": int(args.qa_top_k),
        "comparison": compare_rows(
            pcec_rows,
            reference_rows,
            qa_top_k=int(args.qa_top_k),
            max_examples=int(args.max_examples),
        ),
        "pcec_summary": summarize_rows(pcec_rows, qa_top_k=int(args.qa_top_k)),
        "reference_summary": summarize_rows(reference_rows, qa_top_k=int(args.qa_top_k)),
    }
    write_json(output, args.output_json)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(output), encoding="utf-8")
    print(
        "PCEC parity {match}/{count} exact top-{k} matches".format(
            match=output["comparison"]["exact_topk_match_count"],
            count=output["comparison"]["count_compared"],
            k=int(args.qa_top_k),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
