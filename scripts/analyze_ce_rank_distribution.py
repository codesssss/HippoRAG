#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent


def _warn(message: str) -> None:
    print(f"[warn] {message}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(float(statistics.median(values)))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return _round(values[0])
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return _round(ordered[lower])
    weight = position - lower
    return _round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _safe_pct(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return _round(100.0 * float(count) / float(total))


def infer_dataset(path: Path) -> str:
    parts = path.parts
    for part in parts:
        if part.startswith("outputs_step0_general_"):
            return part.replace("outputs_step0_general_", "", 1)
    path_str = str(path).lower()
    for dataset in ("musique", "hotpotqa", "2wikimultihopqa"):
        if dataset in path_str:
            return dataset
    return "unknown"


def infer_run_label(path: Path) -> str:
    name = path.name.lower()
    if "random3_deep_plus_ce" in name:
        return "random3_deep_plus_ce"
    if "bridge_append_plus_ce" in name:
        return "bridge_append_plus_ce"
    return path.stem


def _score_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _mean(values),
        "median": _median(values),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
    }


def analyze_report(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    traces = list(payload.get("expand_assemble_query_traces") or [])

    num_queries = len(traces)
    queries_with_append = 0
    append_counts_positive: list[float] = []
    total_appended_docs = 0

    appended_ranks: list[float] = []
    appended_scores: list[float] = []
    baseline_scores: list[float] = []
    appended_in_final_top5_docs = 0
    appended_in_final_top5_queries = 0
    histogram: Counter[int] = Counter()
    missing_rank_matches = 0
    max_rank_seen = 0

    for trace in traces:
        expand_trace = dict(trace.get("expand_assemble_trace") or {})
        append_count = int(expand_trace.get("append_count", 0) or 0)
        appended_positions = [int(pos) for pos in (expand_trace.get("appended_positions") or [])]
        final_front_positions = {int(pos) for pos in (expand_trace.get("final_front_pool_positions") or [])}
        assemble_trace = dict(expand_trace.get("assemble_trace") or {})
        ranking_rows = list(assemble_trace.get("ranking_rows") or [])

        if append_count > 0:
            queries_with_append += 1
            append_counts_positive.append(float(append_count))
        total_appended_docs += append_count

        rank_by_pool_position: dict[int, dict[str, Any]] = {}
        for row in ranking_rows:
            pool_position = row.get("pool_position")
            if pool_position is None:
                continue
            pool_position = int(pool_position)
            rank_by_pool_position[pool_position] = row
            rank_value = int(row.get("rank", 0) or 0)
            max_rank_seen = max(max_rank_seen, rank_value)

            source = str(row.get("source", "") or "")
            assemble_score = row.get("assemble_score")
            if assemble_score is None:
                continue
            score = float(assemble_score)
            if source == "baseline_prefix":
                baseline_scores.append(score)
            elif source.startswith("append_"):
                appended_scores.append(score)

        query_has_appended_in_final = False
        for pool_position in appended_positions:
            row = rank_by_pool_position.get(int(pool_position))
            if row is None:
                missing_rank_matches += 1
                continue
            rank_value = int(row.get("rank", 0) or 0)
            appended_ranks.append(float(rank_value))
            histogram[rank_value] += 1
            if int(pool_position) in final_front_positions:
                appended_in_final_top5_docs += 1
                query_has_appended_in_final = True
        if query_has_appended_in_final:
            appended_in_final_top5_queries += 1

    in_top5 = sum(1 for rank in appended_ranks if rank <= 5)
    rank_6_to_10 = sum(1 for rank in appended_ranks if 6 <= rank <= 10)
    rank_11_plus = sum(1 for rank in appended_ranks if rank >= 11)

    ce_rank = {
        "mean": _mean(appended_ranks),
        "median": _median(appended_ranks),
        "in_top5": int(in_top5),
        "in_top5_pct": _safe_pct(in_top5, len(appended_ranks)),
        "rank_6_to_10": int(rank_6_to_10),
        "rank_6_to_10_pct": _safe_pct(rank_6_to_10, len(appended_ranks)),
        "rank_11_plus": int(rank_11_plus),
        "rank_11_plus_pct": _safe_pct(rank_11_plus, len(appended_ranks)),
        "histogram": {
            str(rank): int(histogram.get(rank, 0))
            for rank in range(1, max_rank_seen + 1)
        },
    }

    appended_stats = _score_stats(appended_scores)
    baseline_stats = _score_stats(baseline_scores)
    baseline_mean = baseline_stats.get("mean")
    appended_mean = appended_stats.get("mean")
    ce_score_gap = None
    if baseline_mean is not None and appended_mean is not None:
        ce_score_gap = _round(float(baseline_mean) - float(appended_mean))

    result = {
        "path": str(path),
        "dataset": infer_dataset(path),
        "run_label": infer_run_label(path),
        "num_queries": int(num_queries),
        "queries_with_append": int(queries_with_append),
        "total_appended_docs": int(total_appended_docs),
        "avg_append_count_when_positive": _mean(append_counts_positive),
        "ce_rank": ce_rank,
        "final_penetration": {
            "docs_in_final": int(appended_in_final_top5_docs),
            "queries_with_appended_in_final": int(appended_in_final_top5_queries),
            "penetration_rate": _round(
                float(appended_in_final_top5_docs) / float(total_appended_docs)
            ) if total_appended_docs > 0 else None,
        },
        "ce_score_stats": {
            "appended": appended_stats,
            "baseline_prefix": baseline_stats,
            "gap": ce_score_gap,
        },
        "missing_rank_matches": int(missing_rank_matches),
    }
    return result


def build_cross_report_comparison(reports: list[dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for report in reports:
        dataset = str(report.get("dataset", "unknown"))
        run_label = str(report.get("run_label", "unknown"))
        dataset_entry = comparison.setdefault(dataset, {})
        if run_label == "bridge_append_plus_ce":
            key = "bridge"
        elif run_label == "random3_deep_plus_ce":
            key = "random"
        else:
            key = run_label
        dataset_entry[key] = {
            "mean_ce_rank": ((report.get("ce_rank") or {}).get("mean")),
            "top5_pct": ((report.get("ce_rank") or {}).get("in_top5_pct")),
            "penetration_rate": ((report.get("final_penetration") or {}).get("penetration_rate")),
            "ce_score_gap": ((report.get("ce_score_stats") or {}).get("gap")),
            "queries_with_append": report.get("queries_with_append"),
            "total_appended_docs": report.get("total_appended_docs"),
        }
    return comparison


def build_markdown(payload: dict[str, Any]) -> str:
    reports = list(payload.get("reports") or [])
    lines = [
        "# CE-Rank Distribution of Appended Docs (Full-Scale K=5)",
        "",
        "| Dataset | Append Policy | Queries w/ Append | Total Appended | Mean CE Rank | In Top-5 (%) | Rank 6-10 (%) | Rank 11+ (%) | Final Penetration (%) | CE Score Gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        ce_rank = dict(report.get("ce_rank") or {})
        final_penetration = dict(report.get("final_penetration") or {})
        ce_score_stats = dict(report.get("ce_score_stats") or {})
        lines.append(
            f"| {report.get('dataset')} | {report.get('run_label')} | {report.get('queries_with_append')} | "
            f"{report.get('total_appended_docs')} | {_fmt(ce_rank.get('mean'))} | {_fmt(ce_rank.get('in_top5_pct'))} | "
            f"{_fmt(ce_rank.get('rank_6_to_10_pct'))} | {_fmt(ce_rank.get('rank_11_plus_pct'))} | "
            f"{_fmt(None if final_penetration.get('penetration_rate') is None else 100.0 * float(final_penetration.get('penetration_rate')))} | "
            f"{_fmt(ce_score_stats.get('gap'))} |"
        )

    for report in reports:
        ce_rank = dict(report.get("ce_rank") or {})
        histogram = dict(ce_rank.get("histogram") or {})
        if not histogram:
            continue
        lines.extend(
            [
                "",
                f"## {report.get('dataset')} / {report.get('run_label')}",
                "",
                f"- report: `{report.get('path')}`",
                f"- appended docs analyzed: `{report.get('total_appended_docs')}`",
                f"- mean / median CE rank: `{_fmt(ce_rank.get('mean'))} / {_fmt(ce_rank.get('median'))}`",
                f"- final penetration: `{_fmt(None if (report.get('final_penetration') or {}).get('penetration_rate') is None else 100.0 * float((report.get('final_penetration') or {}).get('penetration_rate')))}%`",
                "",
                "| Rank | Count |",
                "|---:|---:|",
            ]
        )
        for rank_str, count in histogram.items():
            lines.append(f"| {rank_str} | {count} |")

    skipped = list(payload.get("skipped_reports") or [])
    if skipped:
        lines.extend(["", "## Skipped Reports", ""])
        for item in skipped:
            lines.append(f"- `{item}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CE-rank distribution of appended docs from width-matched control reports.")
    parser.add_argument("--reports", nargs="+", required=True, help="One or more eval report JSON files.")
    parser.add_argument("--output_json", required=True, help="Path to write the machine-readable summary JSON.")
    parser.add_argument("--output_md", required=True, help="Path to write the markdown summary.")
    args = parser.parse_args()

    report_paths = [Path(report) for report in args.reports]
    analyzed_reports: list[dict[str, Any]] = []
    skipped_reports: list[str] = []

    for report_path in report_paths:
        if not report_path.exists():
            _warn(f"Missing report, skipping: {report_path}")
            skipped_reports.append(str(report_path))
            continue
        analyzed_reports.append(analyze_report(report_path))

    payload = {
        "reports": analyzed_reports,
        "cross_report_comparison": build_cross_report_comparison(analyzed_reports),
        "skipped_reports": skipped_reports,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
