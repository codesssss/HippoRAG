#!/usr/bin/env python3
"""Summarize ETv3-pool DBEC preservation/admission selector frontier.

The frontier mixes completed full-QA runs and selector-only runs.  This script
normalizes both formats into title-level set-composition metrics only; it does
not compare reader F1 for selector-only points.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
VARIANTS = ("top5_max0", "top4_max1", "top3_max2", "top2_max3", "top1_max2")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*?\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_counter(titles: Sequence[Any], *, k: int | None = None) -> Counter[str]:
    selected = list(titles[:k] if k is not None else titles)
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


def swap_rows(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selector_trace = dict(trace.get("selector_trace", {}) or {})
    return [row for row in list(selector_trace.get("selection_steps", []) or []) if isinstance(row, Mapping)]


def rows_from_traces(
    *,
    dataset: str,
    traces: Sequence[Mapping[str, Any]],
    pool_records: Sequence[Mapping[str, Any]],
    qa_top_k: int,
    force_noop: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            continue
        pool_record = pool_records[idx] if idx < len(pool_records) and isinstance(pool_records[idx], Mapping) else {}
        gold_titles = list(trace.get("gold_titles") or pool_record.get("gold_titles") or [])
        baseline_titles = list(trace.get("baseline_top_titles") or pool_record.get("pool_titles", [])[:qa_top_k])
        selector_titles = baseline_titles if force_noop else list(trace.get("selector_top_titles") or baseline_titles)
        pool_titles = list(pool_record.get("pool_titles") or [])
        selected_trace = {"selector_trace": {}} if force_noop else trace
        swaps = [] if force_noop else swap_rows(selected_trace)
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
        rows.append(
            {
                "dataset": dataset,
                "query_idx": int(idx),
                "gold_doc_count": int(trace.get("gold_doc_count") or len(gold_titles)),
                "gold_titles": gold_titles,
                "baseline_top_titles": baseline_titles,
                "selector_top_titles": selector_titles,
                "changed": bool(baseline_titles != selector_titles),
                "swap_count": int(len(swaps)),
                "swapped_in_gold_title_count": int(swapped_in_gold),
                "swapped_out_gold_title_count": int(swapped_out_gold),
                "baseline_title_recall_top5": title_recall(gold_titles, baseline_titles, k=qa_top_k),
                "selector_title_recall_top5": title_recall(gold_titles, selector_titles, k=qa_top_k),
                "baseline_title_all_gold_top5": title_all_covered(gold_titles, baseline_titles, k=qa_top_k),
                "selector_title_all_gold_top5": title_all_covered(gold_titles, selector_titles, k=qa_top_k),
                "pool_title_all_gold_top100": title_all_covered(gold_titles, pool_titles, k=100),
            }
        )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    count = len(rows)
    changed = [row for row in rows if bool(row["changed"])]
    rescue = [
        row
        for row in rows
        if (not bool(row["baseline_title_all_gold_top5"])) and bool(row["selector_title_all_gold_top5"])
    ]
    regression = [
        row
        for row in rows
        if bool(row["baseline_title_all_gold_top5"]) and (not bool(row["selector_title_all_gold_top5"]))
    ]
    recall_improved = [row for row in rows if row["selector_title_recall_top5"] > row["baseline_title_recall_top5"] + 1e-9]
    recall_worsened = [row for row in rows if row["selector_title_recall_top5"] < row["baseline_title_recall_top5"] - 1e-9]
    gold_out = [row for row in rows if int(row["swapped_out_gold_title_count"]) > 0]
    return {
        "count": int(count),
        "changed_count": int(len(changed)),
        "changed_rate": round(len(changed) / count, 4),
        "total_swaps": int(sum(int(row["swap_count"]) for row in rows)),
        "avg_swaps": round(mean(float(row["swap_count"]) for row in rows), 4),
        "baseline_title_recall_top5": round(mean(float(row["baseline_title_recall_top5"]) for row in rows), 4),
        "selector_title_recall_top5": round(mean(float(row["selector_title_recall_top5"]) for row in rows), 4),
        "title_recall_delta_top5": round(
            mean(float(row["selector_title_recall_top5"]) - float(row["baseline_title_recall_top5"]) for row in rows),
            4,
        ),
        "baseline_title_all_gold_top5": round(sum(bool(row["baseline_title_all_gold_top5"]) for row in rows) / count, 4),
        "selector_title_all_gold_top5": round(sum(bool(row["selector_title_all_gold_top5"]) for row in rows) / count, 4),
        "title_all_gold_delta_top5": round(
            (
                sum(bool(row["selector_title_all_gold_top5"]) for row in rows)
                - sum(bool(row["baseline_title_all_gold_top5"]) for row in rows)
            )
            / count,
            4,
        ),
        "pool_title_all_gold_top100": round(sum(bool(row["pool_title_all_gold_top100"]) for row in rows) / count, 4),
        "title_complete_rescue_count": int(len(rescue)),
        "title_complete_regression_count": int(len(regression)),
        "title_recall_improved_count": int(len(recall_improved)),
        "title_recall_worsened_count": int(len(recall_worsened)),
        "queries_swapped_in_gold": int(sum(int(row["swapped_in_gold_title_count"]) > 0 for row in rows)),
        "queries_swapped_out_gold": int(len(gold_out)),
    }


def summarize_by_depth(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{int(row['gold_doc_count'])}_doc", []).append(row)
    return {key: summarize_rows(grouped[key]) for key in sorted(grouped)}


def pool_records_for(dataset: str, pool_root: Path) -> list[Mapping[str, Any]]:
    payload = read_json(pool_root / f"{dataset}_etv3_pool100_limit1000.json")
    return list(payload.get("records", []) or [])


def full_run_path(dataset: str, variant: str, args: argparse.Namespace) -> Path:
    if variant == "top4_max1":
        if dataset == "musique":
            return Path(args.additive_root) / "evals" / "musique_etv3_pool100_dbec_additive_only_limit1000.json"
        return Path(args.grid_root) / "evals" / f"{dataset}_etv3_pool100_dbec_top4_max1_limit1000.json"
    if variant == "top3_max2" and dataset == "musique":
        return Path(args.grid_root) / "evals" / "musique_etv3_pool100_dbec_top3_max2_limit1000.json"
    if variant == "top1_max2":
        return Path(args.stable_root) / "evals" / f"{dataset}_etv3_pool100_dbec_stable_limit1000.json"
    raise KeyError((dataset, variant))


def selector_run_path(dataset: str, variant: str, args: argparse.Namespace) -> Path:
    return Path(args.selector_root) / "evals" / f"{dataset}_etv3_pool100_dbec_{variant}_selector_limit1000.json"


def traces_for(dataset: str, variant: str, args: argparse.Namespace) -> tuple[list[Mapping[str, Any]], str, Path]:
    if variant == "top5_max0":
        path = full_run_path(dataset, "top1_max2", args)
        payload = read_json(path)
        return list(payload.get("setwise_selector_query_traces", []) or []), "derived_noop_from_stable_baseline", path
    if variant in {"top4_max1", "top1_max2"} or (variant == "top3_max2" and dataset == "musique"):
        path = full_run_path(dataset, variant, args)
        payload = read_json(path)
        return list(payload.get("setwise_selector_query_traces", []) or []), "full_qa_trace", path
    path = selector_run_path(dataset, variant, args)
    payload = read_json(path)
    return list(payload.get("setwise_selector_query_traces", []) or []), "selector_only", path


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    pool_root = Path(args.stable_root) / "pools"
    output: dict[str, Any] = {
        "mode": "etv3_dbec_selector_frontier_summary",
        "qa_top_k": 5,
        "datasets": {},
    }
    for dataset in DATASETS:
        pool_records = pool_records_for(dataset, pool_root)
        dataset_payload: dict[str, Any] = {}
        for variant in VARIANTS:
            traces, source_kind, source_path = traces_for(dataset, variant, args)
            rows = rows_from_traces(
                dataset=dataset,
                traces=traces,
                pool_records=pool_records,
                qa_top_k=5,
                force_noop=variant == "top5_max0",
            )
            dataset_payload[variant] = {
                "source_kind": source_kind,
                "source_path": str(source_path),
                "overall": summarize_rows(rows),
                "by_depth": summarize_by_depth(rows),
            }
        output["datasets"][dataset] = dataset_payload
    return output


def fmt_delta(value: Any) -> str:
    number = float(value)
    return f"{number:+.4f}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ETv3 + DBEC Selector Frontier P1")
    lines.append("")
    lines.append("Scope: selector-level title metrics only. Reader QA is not mixed into this frontier table.")
    lines.append("")
    lines.append("## Overall Frontier")
    lines.append("")
    lines.append("| Dataset | Variant | Preserve / Admit | Title-all@5 | Δ all@5 | Title recall@5 | Δ recall | Changed | Swaps | Rescue | Regression | Gold-out queries |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, dataset_payload in summary["datasets"].items():
        for variant in VARIANTS:
            overall = dataset_payload[variant]["overall"]
            preserve = int(variant.split("_")[0].replace("top", ""))
            max_swaps = int(variant.split("_")[1].replace("max", ""))
            lines.append(
                "| {dataset} | {variant} | {preserve}/{max_swaps} | {all5:.4f} | {delta_all} | "
                "{recall:.4f} | {delta_recall} | {changed} | {swaps} | {rescue} | {regress} | {gold_out} |".format(
                    dataset=dataset,
                    variant=variant,
                    preserve=preserve,
                    max_swaps=max_swaps,
                    all5=float(overall["selector_title_all_gold_top5"]),
                    delta_all=fmt_delta(overall["title_all_gold_delta_top5"]),
                    recall=float(overall["selector_title_recall_top5"]),
                    delta_recall=fmt_delta(overall["title_recall_delta_top5"]),
                    changed=int(overall["changed_count"]),
                    swaps=int(overall["total_swaps"]),
                    rescue=int(overall["title_complete_rescue_count"]),
                    regress=int(overall["title_complete_regression_count"]),
                    gold_out=int(overall["queries_swapped_out_gold"]),
                )
            )
    lines.append("")
    lines.append("## MuSiQue Depth Frontier")
    lines.append("")
    lines.append("| Variant | Slice | Title-all@5 | Δ all@5 | Title recall@5 | Δ recall | Changed | Swaps | Rescue | Regression | Gold-out queries |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    musique = summary["datasets"]["musique"]
    for variant in VARIANTS:
        for depth, data in musique[variant]["by_depth"].items():
            lines.append(
                "| {variant} | {depth} | {all5:.4f} | {delta_all} | {recall:.4f} | {delta_recall} | "
                "{changed} | {swaps} | {rescue} | {regress} | {gold_out} |".format(
                    variant=variant,
                    depth=depth,
                    all5=float(data["selector_title_all_gold_top5"]),
                    delta_all=fmt_delta(data["title_all_gold_delta_top5"]),
                    recall=float(data["selector_title_recall_top5"]),
                    delta_recall=fmt_delta(data["title_recall_delta_top5"]),
                    changed=int(data["changed_count"]),
                    swaps=int(data["total_swaps"]),
                    rescue=int(data["title_complete_rescue_count"]),
                    regress=int(data["title_complete_regression_count"]),
                    gold_out=int(data["queries_swapped_out_gold"]),
                )
            )
    lines.append("")
    lines.append("## Immediate Read")
    lines.append("")
    lines.append("- `top3/max2` gives the highest 2Wiki title-all@5, so the frontier is not a simple top4-only story.")
    lines.append("- `top4/max1` is the most robust preservation point: it is best on HotpotQA and MuSiQue overall and keeps gold-out regressions much lower than looser variants.")
    lines.append("- Loosening preservation beyond top4 (`top2/max3`, `top1/max2`) increases edit volume and gold-out risk faster than it improves set completeness.")
    lines.append("- The frontier is non-monotonic: admission capacity cannot compensate for weaker preservation.")
    lines.append("- This supports preservation-constrained residual admission as the ETv4-composition prior, but it does not prove a state-binding mechanism.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-root", type=Path, default=Path("run_logs/etv3_dbec_latest_full1000_20260510"))
    parser.add_argument("--grid-root", type=Path, default=Path("run_logs/etv3_dbec_preservation_grid_full1000_20260510"))
    parser.add_argument("--additive-root", type=Path, default=Path("run_logs/etv3_dbec_additive_only_full1000_20260510"))
    parser.add_argument("--selector-root", type=Path, default=Path("run_logs/etv3_dbec_selector_frontier_full1000_20260510"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/etv3_dbec_selector_frontier_full1000_20260510/summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/etv3_dbec_selector_frontier_full1000_20260510.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    write_json(summary, Path(args.output_json))
    write_markdown(render_markdown(summary), Path(args.output_md))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
