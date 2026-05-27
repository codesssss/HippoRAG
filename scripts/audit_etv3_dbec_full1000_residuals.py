#!/usr/bin/env python3
"""Audit residual failures for ETv3-pool + stable DBEC full1000.

This is an offline diagnostic.  It reads the completed setwise selector JSONs
and summarizes where the local DBEC edits help or hurt, with special focus on
MuSiQue depth buckets and swap behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_RUN_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510")
DEFAULT_REPORT_DIR = Path("reports/etv3_dbec_latest_full1000_residual_audit_20260510")
DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
RANK_BUCKET_ORDER = ("top5", "rank6_10", "rank11_20", "rank21_50", "rank51_100", "missing_100")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        output = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(output) or math.isinf(output):
        return default
    return output


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_counter(titles: Sequence[Any], *, k: int | None = None) -> Counter[str]:
    selected = titles[:k] if k is not None else titles
    return Counter(title for title in (normalize_title(item) for item in selected) if title)


def title_all_covered(gold_titles: Sequence[Any], candidate_titles: Sequence[Any], *, k: int) -> bool:
    gold = title_counter(gold_titles)
    if not gold:
        return False
    have = title_counter(candidate_titles, k=k)
    return all(have[title] >= count for title, count in gold.items())


def title_recall(gold_titles: Sequence[Any], candidate_titles: Sequence[Any], *, k: int) -> float:
    gold = title_counter(gold_titles)
    if not gold:
        return 0.0
    have = title_counter(candidate_titles, k=k)
    hit = sum(min(count, have[title]) for title, count in gold.items())
    return hit / sum(gold.values())


def first_unconsumed_ranks(gold_titles: Sequence[Any], pool_titles: Sequence[Any]) -> list[int | None]:
    used: set[int] = set()
    ranks: list[int | None] = []
    normalized_pool = [normalize_title(title) for title in pool_titles]
    for gold_title in gold_titles:
        target = normalize_title(gold_title)
        rank: int | None = None
        for idx, title in enumerate(normalized_pool):
            if idx in used:
                continue
            if title == target:
                rank = idx + 1
                used.add(idx)
                break
        ranks.append(rank)
    return ranks


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing_100"
    if rank <= 5:
        return "top5"
    if rank <= 10:
        return "rank6_10"
    if rank <= 20:
        return "rank11_20"
    if rank <= 50:
        return "rank21_50"
    if rank <= 100:
        return "rank51_100"
    return "missing_100"


def max_rank_bucket(ranks: Sequence[int | None]) -> str:
    valid = [rank for rank in ranks if rank is not None]
    if not valid:
        return "missing_100"
    return rank_bucket(max(valid))


def result_path(run_root: Path, dataset: str) -> Path:
    return run_root / "evals" / f"{dataset}_etv3_pool100_dbec_stable_limit1000.json"


def swap_rows(query_idx: int, dataset: str, trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    selector_trace = trace.get("selector_trace", {}) if isinstance(trace, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for step in selector_trace.get("selection_steps", []) or []:
        if not isinstance(step, Mapping):
            continue
        rows.append(
            {
                "dataset": dataset,
                "query_idx": query_idx,
                "step": step.get("step"),
                "out_position": step.get("out_position"),
                "out_title": step.get("out_title", ""),
                "in_position": step.get("in_position"),
                "in_title": step.get("in_title", ""),
                "objective_gain": safe_float(step.get("objective_gain")),
                "retriever_rank_loss": safe_float(step.get("retriever_rank_loss")),
                "adjusted_gain": safe_float(step.get("adjusted_gain")),
            }
        )
    return rows


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    count = len(rows)
    changed = [row for row in rows if row["changed"]]
    improved = [row for row in rows if row["f1_delta"] > 1e-9]
    worsened = [row for row in rows if row["f1_delta"] < -1e-9]
    same = count - len(improved) - len(worsened)
    baseline_title_complete = [row for row in rows if row["baseline_title_all_gold_top5"]]
    selector_title_complete = [row for row in rows if row["selector_title_all_gold_top5"]]
    pool_title_complete = [row for row in rows if row["pool_title_all_gold_top100"]]
    return {
        "count": count,
        "changed_count": len(changed),
        "changed_rate": len(changed) / count,
        "baseline_em": mean(row["baseline_em"] for row in rows),
        "selector_em": mean(row["selector_em"] for row in rows),
        "em_delta": mean(row["em_delta"] for row in rows),
        "baseline_f1": mean(row["baseline_f1"] for row in rows),
        "selector_f1": mean(row["selector_f1"] for row in rows),
        "f1_delta": mean(row["f1_delta"] for row in rows),
        "improved_count": len(improved),
        "same_count": same,
        "worsened_count": len(worsened),
        "baseline_title_all_gold_top5": len(baseline_title_complete) / count,
        "selector_title_all_gold_top5": len(selector_title_complete) / count,
        "pool_title_all_gold_top100": len(pool_title_complete) / count,
        "title_complete_rescue_count": sum(
            (not row["baseline_title_all_gold_top5"]) and row["selector_title_all_gold_top5"]
            for row in rows
        ),
        "title_complete_regression_count": sum(
            row["baseline_title_all_gold_top5"] and (not row["selector_title_all_gold_top5"])
            for row in rows
        ),
        "avg_swap_count": mean(row["swap_count"] for row in rows),
        "avg_baseline_title_recall_top5": mean(row["baseline_title_recall_top5"] for row in rows),
        "avg_selector_title_recall_top5": mean(row["selector_title_recall_top5"] for row in rows),
    }


def non_gold_swapout_harm_categories(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    output = Counter()
    for row in rows:
        if not (row["f1_delta"] < -1e-9 and row["swapped_out_gold_title_count"] == 0):
            continue
        if row["baseline_title_all_gold_top5"] and row["selector_title_all_gold_top5"]:
            output["gold_complete_preserved_order_or_context_sensitivity"] += 1
        elif (
            not row["baseline_title_all_gold_top5"]
            and not row["selector_title_all_gold_top5"]
            and abs(row["selector_title_recall_top5"] - row["baseline_title_recall_top5"]) < 1e-9
        ):
            output["gold_recall_same_non_gold_reorder_or_context"] += 1
        elif row["selector_title_recall_top5"] < row["baseline_title_recall_top5"]:
            output["title_recall_decreased_without_exact_gold_swapout"] += 1
        elif row["selector_title_recall_top5"] > row["baseline_title_recall_top5"]:
            output["title_recall_increased_but_reader_worse"] += 1
        else:
            output["other"] += 1
    return dict(output)


def audit_dataset(dataset: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = read_json(path)
    traces = payload.get("setwise_selector_query_traces", []) or []
    examples = payload.get("examples", []) or []
    query_rows: list[dict[str, Any]] = []
    all_swap_rows: list[dict[str, Any]] = []

    for idx, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            continue
        example = examples[idx] if idx < len(examples) and isinstance(examples[idx], Mapping) else {}
        retrieval_trace = example.get("retrieval_trace", {}) if isinstance(example, Mapping) else {}
        pool_titles = list(retrieval_trace.get("external_pool_titles", []) or [])
        gold_titles = list(trace.get("gold_titles", []) or [])
        baseline_top = list(trace.get("baseline_top_titles", []) or [])
        selector_top = list(trace.get("selector_top_titles", []) or [])
        baseline_metrics = trace.get("baseline_metrics", {}) or {}
        selector_metrics = trace.get("selector_metrics", {}) or {}
        baseline_f1 = safe_float(baseline_metrics.get("F1"))
        selector_f1 = safe_float(selector_metrics.get("F1"))
        baseline_em = safe_float(baseline_metrics.get("ExactMatch"))
        selector_em = safe_float(selector_metrics.get("ExactMatch"))
        ranks = first_unconsumed_ranks(gold_titles, pool_titles)
        swaps = swap_rows(idx, dataset, trace)
        all_swap_rows.extend(swaps)

        gold_norm = title_counter(gold_titles)
        swapped_in_gold = 0
        swapped_out_gold = 0
        for swap in swaps:
            in_title = normalize_title(swap.get("in_title"))
            out_title = normalize_title(swap.get("out_title"))
            if gold_norm.get(in_title, 0) > 0:
                swapped_in_gold += 1
            if gold_norm.get(out_title, 0) > 0:
                swapped_out_gold += 1

        row = {
            "dataset": dataset,
            "query_idx": idx,
            "question": trace.get("question", ""),
            "gold_doc_count": int(trace.get("gold_doc_count") or 0),
            "gold_titles": " || ".join(str(title) for title in gold_titles),
            "baseline_top_titles": " || ".join(str(title) for title in baseline_top),
            "selector_top_titles": " || ".join(str(title) for title in selector_top),
            "changed": bool(trace.get("changed_from_baseline")),
            "baseline_em": baseline_em,
            "selector_em": selector_em,
            "em_delta": selector_em - baseline_em,
            "baseline_f1": baseline_f1,
            "selector_f1": selector_f1,
            "f1_delta": selector_f1 - baseline_f1,
            "baseline_title_recall_top5": title_recall(gold_titles, baseline_top, k=5),
            "selector_title_recall_top5": title_recall(gold_titles, selector_top, k=5),
            "baseline_title_all_gold_top5": title_all_covered(gold_titles, baseline_top, k=5),
            "selector_title_all_gold_top5": title_all_covered(gold_titles, selector_top, k=5),
            "pool_title_all_gold_top100": title_all_covered(gold_titles, pool_titles, k=100),
            "max_gold_title_rank_bucket_pool100": max_rank_bucket(ranks),
            "gold_title_ranks_pool100": ",".join("NA" if rank is None else str(rank) for rank in ranks),
            "swap_count": len(swaps),
            "swapped_in_gold_title_count": swapped_in_gold,
            "swapped_out_gold_title_count": swapped_out_gold,
            "baseline_title_duplicate_count": int(trace.get("baseline_title_duplicate_count") or 0),
            "selector_title_duplicate_count": int(trace.get("selector_title_duplicate_count") or 0),
        }
        query_rows.append(row)

    by_depth: dict[str, Any] = {}
    depth_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in query_rows:
        depth_groups[int(row["gold_doc_count"])].append(row)
    for depth in sorted(depth_groups):
        by_depth[str(depth)] = summarize_group(depth_groups[depth])

    changed_rows = [row for row in query_rows if row["changed"]]
    worsened_rows = [row for row in query_rows if row["f1_delta"] < -1e-9]
    changed_worsened_rows = [row for row in query_rows if row["changed"] and row["f1_delta"] < -1e-9]
    improved_rows = [row for row in query_rows if row["f1_delta"] > 1e-9]
    rank_bucket_counts = Counter(row["max_gold_title_rank_bucket_pool100"] for row in query_rows)
    residual_pool_complete = [
        row
        for row in query_rows
        if row["pool_title_all_gold_top100"] and not row["selector_title_all_gold_top5"]
    ]
    residual_bucket_counts = Counter(row["max_gold_title_rank_bucket_pool100"] for row in residual_pool_complete)
    swap_counter = Counter()
    for row in query_rows:
        if row["swap_count"]:
            swap_counter["queries_with_swap"] += 1
        if row["swapped_in_gold_title_count"]:
            swap_counter["queries_swapped_in_gold"] += 1
        if row["swapped_out_gold_title_count"]:
            swap_counter["queries_swapped_out_gold"] += 1
    objective_gains = [row["objective_gain"] for row in all_swap_rows]

    summary = {
        "dataset": dataset,
        "result_path": str(path),
        "overall": summarize_group(query_rows),
        "by_depth": by_depth,
        "changed": summarize_group(changed_rows),
        "worsened": summarize_group(worsened_rows),
        "changed_worsened": summarize_group(changed_worsened_rows),
        "improved": summarize_group(improved_rows),
        "rank_bucket_counts": {key: rank_bucket_counts.get(key, 0) for key in RANK_BUCKET_ORDER},
        "residual_pool_complete_not_selector_complete_rank_buckets": {
            key: residual_bucket_counts.get(key, 0) for key in RANK_BUCKET_ORDER
        },
        "harm_summary": {
            "worsened_count": len(worsened_rows),
            "changed_worsened_count": len(changed_worsened_rows),
            "worsened_with_gold_title_swapped_out": sum(
                row["swapped_out_gold_title_count"] > 0 for row in worsened_rows
            ),
            "worsened_with_gold_title_swapped_in": sum(
                row["swapped_in_gold_title_count"] > 0 for row in worsened_rows
            ),
            "worsened_without_gold_title_swapped_out": sum(
                row["swapped_out_gold_title_count"] == 0 for row in worsened_rows
            ),
            "worsened_without_gold_title_swapped_out_categories": non_gold_swapout_harm_categories(query_rows),
            "pool_title_complete_selector_incomplete": len(residual_pool_complete),
        },
        "swap_summary": {
            "total_swaps": len(all_swap_rows),
            "queries_with_swap": swap_counter["queries_with_swap"],
            "queries_swapped_in_gold": swap_counter["queries_swapped_in_gold"],
            "queries_swapped_out_gold": swap_counter["queries_swapped_out_gold"],
            "avg_objective_gain": mean(objective_gains) if objective_gains else 0.0,
            "median_objective_gain": median(objective_gains) if objective_gains else 0.0,
        },
    }
    return summary, query_rows, all_swap_rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(audit: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# ETv3 Pool + Stable DBEC Residual Audit",
        "",
        "Date: 2026-05-10",
        "",
        "This is an offline diagnostic over the completed full1000 JSON outputs.",
        "It does not call any model and does not change ETv3 or DBEC code.",
        "",
        "## Overall By Dataset",
        "",
        "| Dataset | Count | Changed | Base F1 | DBEC F1 | Delta F1 | Base title-all@5 | DBEC title-all@5 | Pool title-all@100 | Swaps | Swap-in gold | Swap-out gold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, summary in audit["datasets"].items():
        overall = summary["overall"]
        swaps = summary["swap_summary"]
        lines.append(
            "| {dataset} | {count} | {changed} | {base_f1} | {sel_f1} | {delta} | {base_all} | {sel_all} | {pool_all} | {swaps_total} | {swap_in} | {swap_out} |".format(
                dataset=dataset,
                count=overall["count"],
                changed=overall["changed_count"],
                base_f1=fmt(overall["baseline_f1"]),
                sel_f1=fmt(overall["selector_f1"]),
                delta=fmt(overall["f1_delta"]),
                base_all=fmt(overall["baseline_title_all_gold_top5"]),
                sel_all=fmt(overall["selector_title_all_gold_top5"]),
                pool_all=fmt(overall["pool_title_all_gold_top100"]),
                swaps_total=swaps["total_swaps"],
                swap_in=swaps["queries_swapped_in_gold"],
                swap_out=swaps["queries_swapped_out_gold"],
            )
        )

    musique = audit["datasets"].get("musique", {})
    lines.extend(
        [
            "",
            "## MuSiQue Depth Breakdown",
            "",
            "| Gold docs | Count | Changed | Base F1 | DBEC F1 | Delta F1 | Base title-all@5 | DBEC title-all@5 | Pool title-all@100 | Rescue | Regression | Avg swaps |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for depth, row in musique.get("by_depth", {}).items():
        lines.append(
            "| {depth} | {count} | {changed} | {base_f1} | {sel_f1} | {delta} | {base_all} | {sel_all} | {pool_all} | {rescue} | {regression} | {swaps} |".format(
                depth=depth,
                count=row["count"],
                changed=row["changed_count"],
                base_f1=fmt(row["baseline_f1"]),
                sel_f1=fmt(row["selector_f1"]),
                delta=fmt(row["f1_delta"]),
                base_all=fmt(row["baseline_title_all_gold_top5"]),
                sel_all=fmt(row["selector_title_all_gold_top5"]),
                pool_all=fmt(row["pool_title_all_gold_top100"]),
                rescue=row["title_complete_rescue_count"],
                regression=row["title_complete_regression_count"],
                swaps=fmt(row["avg_swap_count"]),
            )
        )

    wiki = audit["datasets"].get("2wikimultihopqa", {})
    lines.extend(
        [
            "",
            "## 2Wiki Gain Source",
            "",
            "| Gold docs | Count | Changed | Base F1 | DBEC F1 | Delta F1 | Base title-all@5 | DBEC title-all@5 | Rescue | Regression |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    total_delta = wiki.get("overall", {}).get("f1_delta", 0.0) or 0.0
    for depth, row in wiki.get("by_depth", {}).items():
        lines.append(
            "| {depth} | {count} | {changed} | {base_f1} | {sel_f1} | {delta} | {base_all} | {sel_all} | {rescue} | {regression} |".format(
                depth=depth,
                count=row["count"],
                changed=row["changed_count"],
                base_f1=fmt(row["baseline_f1"]),
                sel_f1=fmt(row["selector_f1"]),
                delta=fmt(row["f1_delta"]),
                base_all=fmt(row["baseline_title_all_gold_top5"]),
                sel_all=fmt(row["selector_title_all_gold_top5"]),
                rescue=row["title_complete_rescue_count"],
                regression=row["title_complete_regression_count"],
            )
        )
    if total_delta:
        total_count = wiki.get("overall", {}).get("count", 0) or 0
        lines.extend(["", "2Wiki contribution to the overall F1 delta:"])
        for depth, row in wiki.get("by_depth", {}).items():
            contribution = float(row["f1_delta"]) * int(row["count"]) / int(total_count)
            share = contribution / float(total_delta)
            lines.append(f"- `{depth}`-doc: absolute contribution `{contribution:.4f}`, share `{share:.1%}`.")

    lines.extend(
        [
            "",
            "## MuSiQue Residual Rank Buckets",
            "",
            "Bucket is the maximum title-rank needed to cover all gold titles in the ETv3 pool100.",
            "",
            "| Bucket | All queries | Pool title-complete but DBEC top5 title-incomplete |",
            "|---|---:|---:|",
        ]
    )
    all_buckets = musique.get("rank_bucket_counts", {})
    residual_buckets = musique.get("residual_pool_complete_not_selector_complete_rank_buckets", {})
    for bucket in RANK_BUCKET_ORDER:
        lines.append(f"| `{bucket}` | {all_buckets.get(bucket, 0)} | {residual_buckets.get(bucket, 0)} |")

    harm = musique.get("harm_summary", {})
    lines.extend(
        [
            "",
            "## MuSiQue Harm Summary",
            "",
            "| Metric | Count |",
            "|---|---:|",
            f"| Worsened queries | {harm.get('worsened_count', 0)} |",
            f"| Worsened and changed queries | {harm.get('changed_worsened_count', 0)} |",
            f"| Worsened with a gold title swapped out | {harm.get('worsened_with_gold_title_swapped_out', 0)} |",
            f"| Worsened with a gold title swapped in | {harm.get('worsened_with_gold_title_swapped_in', 0)} |",
            f"| Worsened without a gold title swapped out | {harm.get('worsened_without_gold_title_swapped_out', 0)} |",
            f"| Pool title-complete but DBEC top5 title-incomplete | {harm.get('pool_title_complete_selector_incomplete', 0)} |",
            "",
            "Breakdown of worsened queries without a gold-title swap-out:",
            "",
            "| Category | Count |",
            "|---|---:|",
        ]
    )
    for category, count in sorted(
        (harm.get("worsened_without_gold_title_swapped_out_categories") or {}).items()
    ):
        lines.append(f"| `{category}` | {count} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The MuSiQue regression is not a missing-binding-call artifact: many swaps are made, but title-level set completeness does not improve enough and F1 drops.",
            "- The 2-doc harm is especially important: the selector changes many shallow cases while the pre-registered regression gate required preserving them.",
            "- The local-edit objective is positive on 2Wiki but miscalibrated on MuSiQue; this supports a residual-audit path rather than immediately naming ETv4 as state-binding.",
            "- A clean next diagnostic is query-level manual inspection of the MuSiQue changed-and-worsened rows, especially cases with `swapped_out_gold_title_count > 0` or pool title-complete but selector title-incomplete.",
            "",
            "## Files",
            "",
            f"- JSON: `{audit['json_path']}`",
            f"- Query CSV: `{audit['query_csv_path']}`",
            f"- Swap CSV: `{audit['swap_csv_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    args.report_dir.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {
        "run_root": str(args.run_root),
        "datasets": {},
    }
    all_query_rows: list[dict[str, Any]] = []
    all_swap_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        summary, query_rows, swap_rows_for_dataset = audit_dataset(dataset, result_path(args.run_root, dataset))
        audit["datasets"][dataset] = summary
        all_query_rows.extend(query_rows)
        all_swap_rows.extend(swap_rows_for_dataset)

    json_path = args.report_dir / "etv3_dbec_full1000_residual_audit.json"
    query_csv_path = args.report_dir / "etv3_dbec_full1000_residual_queries.csv"
    swap_csv_path = args.report_dir / "etv3_dbec_full1000_residual_swaps.csv"
    md_path = args.report_dir / "etv3_dbec_full1000_residual_audit.md"
    audit["json_path"] = str(json_path)
    audit["query_csv_path"] = str(query_csv_path)
    audit["swap_csv_path"] = str(swap_csv_path)
    audit["markdown_path"] = str(md_path)

    write_json(audit, json_path)
    write_csv(all_query_rows, query_csv_path)
    write_csv(all_swap_rows, swap_csv_path)
    md_path.write_text(markdown_report(audit), encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
