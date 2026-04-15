#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_question(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _safe_pct(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return _round(100.0 * float(count) / float(total))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(float(statistics.median(values)))


def _infer_dataset(*paths: str | Path, payload: dict[str, Any] | None = None) -> str:
    dataset = str((payload or {}).get("dataset") or "").strip().lower()
    if dataset:
        return dataset
    joined = " ".join(str(path).lower() for path in paths)
    for candidate in ("musique", "hotpotqa", "2wikimultihopqa", "2wiki", "nq", "popqa"):
        if candidate in joined:
            return "2wikimultihopqa" if candidate == "2wiki" else candidate
    return "unknown"


def _infer_qa_top_k(path: str | Path, payload: dict[str, Any]) -> int | None:
    direct = payload.get("qa_top_k")
    if direct is not None:
        return int(direct)
    expand = dict(payload.get("expand_assemble_qa") or {})
    if expand.get("qa_top_k") is not None:
        return int(expand["qa_top_k"])
    match = re.search(r"qatopk(\d+)", str(path))
    if match:
        return int(match.group(1))
    return None


def _index_query_traces(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in list(payload.get("expand_assemble_query_traces") or []):
        key = _normalize_question(row.get("question", ""))
        if key:
            mapping[key] = dict(row)
    return mapping


def _index_swap_queries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in list(payload.get("queries") or []):
        key = _normalize_question(row.get("question", ""))
        if key:
            mapping[key] = dict(row)
    return mapping


def analyze_reports(baseline_payload: dict[str, Any],
                    repair_payload: dict[str, Any],
                    swap_payload: dict[str, Any] | None = None,
                    baseline_path: str | Path | None = None,
                    repair_path: str | Path | None = None,
                    swap_path: str | Path | None = None) -> dict[str, Any]:
    baseline_by_question = _index_query_traces(baseline_payload)
    repair_by_question = _index_query_traces(repair_payload)
    swap_by_question = _index_swap_queries(swap_payload or {})

    common_questions = [
        question for question in baseline_by_question
        if question in repair_by_question
    ]

    connector_hist: Counter[str] = Counter()
    anchor_hist: Counter[str] = Counter()
    missing_anchor_counts: list[float] = []
    scaffold_component_counts: list[float] = []
    applied_ce_drops: list[float] = []
    applied_delta_em: list[float] = []
    applied_delta_f1: list[float] = []

    queries_with_positive_connector = 0
    queries_with_positive_anchor = 0
    queries_with_positive_both = 0
    queries_with_positive_req_swap = 0
    anchor_only_queries = 0
    connector_only_queries = 0
    repair_applied_queries = 0
    applied_positive_em = 0
    applied_positive_f1 = 0
    proxy_oracle_overlap = 0
    proxy_positive_but_oracle_negative = 0
    oracle_positive_but_proxy_negative = 0
    oracle_positive_total = 0

    top_cases: list[dict[str, Any]] = []

    for question in common_questions:
        baseline_trace = dict(baseline_by_question[question] or {})
        repair_trace = dict(repair_by_question[question] or {})
        repair_expand = dict(repair_trace.get("expand_assemble_trace") or {})
        assemble_trace = dict(repair_expand.get("assemble_trace") or {})
        repair_rows = list(assemble_trace.get("repair_candidate_rows") or [])
        best_swap = dict(assemble_trace.get("best_repair_swap") or {})
        repair_applied = bool(assemble_trace.get("repair_applied", False))

        missing_anchor_counts.append(float(assemble_trace.get("missing_query_anchor_count", 0) or 0))
        scaffold_component_counts.append(float(assemble_trace.get("scaffold_component_count", 0) or 0))

        has_positive_connector = any(int(row.get("connector_gain_count", 0) or 0) > 0 for row in repair_rows)
        has_positive_anchor = any(int(row.get("anchor_gain_count", 0) or 0) > 0 for row in repair_rows)
        has_positive_both = any(
            int(row.get("connector_gain_count", 0) or 0) > 0 and int(row.get("anchor_gain_count", 0) or 0) > 0
            for row in repair_rows
        )
        has_positive_req_swap = bool(has_positive_connector or has_positive_anchor)

        queries_with_positive_connector += int(has_positive_connector)
        queries_with_positive_anchor += int(has_positive_anchor)
        queries_with_positive_both += int(has_positive_both)
        queries_with_positive_req_swap += int(has_positive_req_swap)
        anchor_only_queries += int(has_positive_anchor and not has_positive_connector)
        connector_only_queries += int(has_positive_connector and not has_positive_anchor)

        for row in repair_rows:
            connector_hist[str(int(row.get("connector_gain_count", 0) or 0))] += 1
            anchor_hist[str(int(row.get("anchor_gain_count", 0) or 0))] += 1

        baseline_metrics = dict(baseline_trace.get("method_metrics") or {})
        repair_metrics = dict(repair_trace.get("method_metrics") or {})
        delta_em = float(repair_metrics.get("ExactMatch", 0.0) or 0.0) - float(baseline_metrics.get("ExactMatch", 0.0) or 0.0)
        delta_f1 = float(repair_metrics.get("F1", 0.0) or 0.0) - float(baseline_metrics.get("F1", 0.0) or 0.0)

        if repair_applied:
            repair_applied_queries += 1
            applied_positive_em += int(delta_em > 0)
            applied_positive_f1 += int(delta_f1 > 0)
            applied_delta_em.append(delta_em)
            applied_delta_f1.append(delta_f1)
            applied_ce_drops.append(float(best_swap.get("ce_drop_vs_scaffold", 0.0) or 0.0))

        swap_row = dict(swap_by_question.get(question) or {})
        oracle_positive = bool(
            swap_row.get("has_positive_swap_em", False) or swap_row.get("has_positive_swap_f1", False)
        )
        oracle_positive_total += int(oracle_positive)
        proxy_oracle_overlap += int(has_positive_req_swap and oracle_positive)
        proxy_positive_but_oracle_negative += int(has_positive_req_swap and not oracle_positive)
        oracle_positive_but_proxy_negative += int(oracle_positive and not has_positive_req_swap)

        if repair_applied:
            top_cases.append({
                "question": question,
                "delta_em": _round(delta_em),
                "delta_f1": _round(delta_f1),
                "ce_drop_vs_scaffold": _round(float(best_swap.get("ce_drop_vs_scaffold", 0.0) or 0.0)),
                "connector_gain_count": int(best_swap.get("connector_gain_count", 0) or 0),
                "anchor_gain_count": int(best_swap.get("anchor_gain_count", 0) or 0),
                "candidate_title": str(best_swap.get("candidate_title", "")),
                "replace_title": str(best_swap.get("replace_title", "")),
            })

    top_cases.sort(
        key=lambda row: (
            -float(row.get("delta_f1", 0.0) or 0.0),
            -float(row.get("delta_em", 0.0) or 0.0),
            float(row.get("ce_drop_vs_scaffold", 0.0) or 0.0),
        )
    )

    return {
        "metadata": {
            "dataset": _infer_dataset(baseline_path or "", repair_path or "", payload=repair_payload),
            "baseline_report": str(baseline_path or ""),
            "repair_report": str(repair_path or ""),
            "swap_report": str(swap_path or "") if swap_payload else None,
            "qa_top_k": _infer_qa_top_k(repair_path or "", repair_payload),
            "aligned_queries": int(len(common_questions)),
        },
        "proxy_sanity": {
            "queries_with_positive_req_swap": int(queries_with_positive_req_swap),
            "positive_req_swap_rate": _safe_pct(queries_with_positive_req_swap, len(common_questions)),
            "mean_missing_anchor_count": _mean(missing_anchor_counts),
            "median_missing_anchor_count": _median(missing_anchor_counts),
            "mean_scaffold_component_count": _mean(scaffold_component_counts),
            "median_scaffold_component_count": _median(scaffold_component_counts),
            "queries_with_positive_connector_gain": int(queries_with_positive_connector),
            "queries_with_positive_anchor_gain": int(queries_with_positive_anchor),
            "queries_with_positive_both": int(queries_with_positive_both),
            "anchor_only_rate": _safe_pct(anchor_only_queries, len(common_questions)),
            "connector_only_rate": _safe_pct(connector_only_queries, len(common_questions)),
            "connector_gain_histogram": dict(sorted(connector_hist.items(), key=lambda item: int(item[0]))),
            "anchor_gain_histogram": dict(sorted(anchor_hist.items(), key=lambda item: int(item[0]))),
        },
        "repair_effect": {
            "repair_applied_queries": int(repair_applied_queries),
            "repair_applied_query_rate": _safe_pct(repair_applied_queries, len(common_questions)),
            "applied_swap_positive_em_rate": _safe_pct(applied_positive_em, repair_applied_queries),
            "applied_swap_positive_f1_rate": _safe_pct(applied_positive_f1, repair_applied_queries),
            "applied_swap_avg_ce_drop_vs_scaffold": _mean(applied_ce_drops),
            "applied_swap_median_ce_drop_vs_scaffold": _median(applied_ce_drops),
            "applied_swap_mean_delta_em": _mean(applied_delta_em),
            "applied_swap_mean_delta_f1": _mean(applied_delta_f1),
        },
        "swap_overlap": {
            "oracle_positive_queries": int(oracle_positive_total),
            "proxy_positive_and_oracle_positive": int(proxy_oracle_overlap),
            "proxy_positive_and_oracle_positive_rate": _safe_pct(proxy_oracle_overlap, len(common_questions)),
            "proxy_positive_but_oracle_negative": int(proxy_positive_but_oracle_negative),
            "oracle_positive_but_proxy_negative": int(oracle_positive_but_proxy_negative),
        },
        "top_repair_cases": top_cases[:10],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    meta = dict(summary.get("metadata") or {})
    proxy = dict(summary.get("proxy_sanity") or {})
    repair = dict(summary.get("repair_effect") or {})
    overlap = dict(summary.get("swap_overlap") or {})
    lines = [
        f"# CE Local Repair Proxy Analysis ({meta.get('dataset', 'unknown')})",
        "",
        f"- baseline report: `{meta.get('baseline_report', '')}`",
        f"- repair report: `{meta.get('repair_report', '')}`",
        f"- swap report: `{meta.get('swap_report', '')}`",
        f"- qa_top_k: `{meta.get('qa_top_k', 'unknown')}`",
        f"- aligned queries: `{meta.get('aligned_queries', 0)}`",
        "",
        "## Proxy Sanity",
        "",
        f"- queries with positive req swap: `{proxy.get('queries_with_positive_req_swap', 0)}`",
        f"- positive req swap rate: `{proxy.get('positive_req_swap_rate', '—')}`",
        f"- mean missing anchor count: `{proxy.get('mean_missing_anchor_count', '—')}`",
        f"- mean scaffold component count: `{proxy.get('mean_scaffold_component_count', '—')}`",
        f"- queries with positive connector gain: `{proxy.get('queries_with_positive_connector_gain', 0)}`",
        f"- queries with positive anchor gain: `{proxy.get('queries_with_positive_anchor_gain', 0)}`",
        f"- queries with positive both: `{proxy.get('queries_with_positive_both', 0)}`",
        f"- anchor-only rate: `{proxy.get('anchor_only_rate', '—')}`",
        f"- connector-only rate: `{proxy.get('connector_only_rate', '—')}`",
        "",
        "## Repair Effect",
        "",
        f"- repair applied query rate: `{repair.get('repair_applied_query_rate', '—')}`",
        f"- applied swap positive EM rate: `{repair.get('applied_swap_positive_em_rate', '—')}`",
        f"- applied swap positive F1 rate: `{repair.get('applied_swap_positive_f1_rate', '—')}`",
        f"- applied swap avg CE drop vs scaffold: `{repair.get('applied_swap_avg_ce_drop_vs_scaffold', '—')}`",
        f"- applied swap mean ΔEM / ΔF1: `{repair.get('applied_swap_mean_delta_em', '—')}` / `{repair.get('applied_swap_mean_delta_f1', '—')}`",
        "",
        "## Swap Overlap",
        "",
        f"- oracle positive queries: `{overlap.get('oracle_positive_queries', 0)}`",
        f"- proxy positive and oracle positive: `{overlap.get('proxy_positive_and_oracle_positive', 0)}`",
        f"- overlap rate: `{overlap.get('proxy_positive_and_oracle_positive_rate', '—')}`",
        "",
        "## Top Repair Cases",
        "",
    ]
    cases = list(summary.get("top_repair_cases") or [])
    if not cases:
        lines.append("- none")
    else:
        for row in cases:
            lines.extend([
                f"- question: `{row.get('question', '')}`",
                (
                    "  repair: "
                    f"`{row.get('candidate_title', '')}` -> replace `{row.get('replace_title', '')}` "
                    f"(connector `{row.get('connector_gain_count', 0)}`, "
                    f"anchor `{row.get('anchor_gain_count', 0)}`, "
                    f"CE drop `{row.get('ce_drop_vs_scaffold', '—')}`)"
                ),
                f"  delta EM/F1: `{row.get('delta_em', '—')}` / `{row.get('delta_f1', '—')}`",
            ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ce_local_repair report traces against a baseline CE report.")
    parser.add_argument("--baseline_report", required=True)
    parser.add_argument("--repair_report", required=True)
    parser.add_argument("--swap_report", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args()

    baseline_payload = _load_json(args.baseline_report)
    repair_payload = _load_json(args.repair_report)
    swap_payload = _load_json(args.swap_report) if str(args.swap_report).strip() else None
    summary = analyze_reports(
        baseline_payload=baseline_payload,
        repair_payload=repair_payload,
        swap_payload=swap_payload,
        baseline_path=args.baseline_report,
        repair_path=args.repair_report,
        swap_path=args.swap_report,
    )

    if str(args.output_json).strip():
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if str(args.output_md).strip():
        Path(args.output_md).write_text(render_markdown(summary), encoding="utf-8")
    if not str(args.output_json).strip() and not str(args.output_md).strip():
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
