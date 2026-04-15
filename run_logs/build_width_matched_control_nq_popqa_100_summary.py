#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/nvme/code/HippoRAG")
RUN_ROOT = ROOT / "run_logs"
OUTPUT_ROOT = ROOT
DATASETS = ("nq", "popqa")
RUN_TYPES = (
    "baseline_top10_plus_ce",
    "random3_deep_plus_ce",
    "bridge_append_plus_ce",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _safe_delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return round(float(value) - float(baseline), 4)


def _row_to_md(row: dict[str, Any]) -> str:
    delta_baseline = row.get("delta_vs_baseline_em")
    delta_top10 = row.get("delta_vs_baseline_top10_plus_ce_em")
    delta_baseline_str = "—" if delta_baseline is None else f"{float(delta_baseline):+.4f}"
    delta_top10_str = "—" if delta_top10 is None else f"{float(delta_top10):+.4f}"
    return (
        f"| {row['dataset']} | {row['run_type']} | {_fmt(row.get('em'))} | {_fmt(row.get('f1'))} | "
        f"{_fmt(row.get('recall_at_5'))} | {_fmt(row.get('recall_at_20'))} | {_fmt(row.get('recall_at_100'))} | "
        f"{row.get('num_queries', '—')} | {delta_baseline_str} | {delta_top10_str} |"
    )


def collect_dataset_rows(dataset: str, date_tag: str, limit: int) -> list[dict[str, Any]]:
    eval_dir = OUTPUT_ROOT / f"outputs_step0_general_{dataset}" / "eval_reports"
    rows: list[dict[str, Any]] = []
    baseline_path = eval_dir / f"causal_eval_{dataset}_{limit}_qwen3-8b_qatopk5_baseline_{date_tag}.json"
    baseline_top10_row: dict[str, Any] | None = None

    baseline = _read_json(baseline_path)
    if baseline is not None:
        overall = baseline.get("overall_recomputed", {}) or {}
        rows.append(
            {
                "dataset": dataset,
                "run_type": "baseline_top5",
                "qa_top_k": 5,
                "em": overall.get("ExactMatch"),
                "f1": overall.get("F1"),
                "recall_at_5": overall.get("Recall@5"),
                "recall_at_20": overall.get("Recall@20"),
                "recall_at_100": overall.get("Recall@100"),
                "num_queries": overall.get("num_queries"),
                "report": str(baseline_path.relative_to(ROOT)),
            }
        )

    dataset_rows: list[dict[str, Any]] = []
    for run_type in RUN_TYPES:
        report_path = eval_dir / f"width_match_{run_type}_qatopk5_{date_tag}.json"
        report = _read_json(report_path)
        if report is None:
            continue
        method = report.get("expand_assemble_qa", {}) or {}
        retrieval = method.get("method_retrieval_metrics", {}) or {}
        row = {
            "dataset": dataset,
            "run_type": run_type,
            "qa_top_k": 5,
            "em": method.get("method_EM"),
            "f1": method.get("method_F1"),
            "recall_at_5": retrieval.get("Recall@5"),
            "recall_at_20": retrieval.get("Recall@20"),
            "recall_at_100": retrieval.get("Recall@100"),
            "delta_vs_baseline_em": method.get("EM_delta"),
            "delta_vs_baseline_f1": method.get("F1_delta"),
            "num_queries": report.get("limit"),
            "report": str(report_path.relative_to(ROOT)),
        }
        dataset_rows.append(row)
        if run_type == "baseline_top10_plus_ce":
            baseline_top10_row = row

    if baseline_top10_row is not None:
        for row in dataset_rows:
            row["delta_vs_baseline_top10_plus_ce_em"] = _safe_delta(row.get("em"), baseline_top10_row.get("em"))
            row["delta_vs_baseline_top10_plus_ce_f1"] = _safe_delta(row.get("f1"), baseline_top10_row.get("f1"))

    rows.extend(dataset_rows)
    return rows


def write_dataset_summary(dataset: str, rows: list[dict[str, Any]], date_tag: str) -> None:
    json_path = RUN_ROOT / f"width_matched_control_{dataset}_100_{date_tag}.summary.json"
    md_path = RUN_ROOT / f"width_matched_control_{dataset}_100_{date_tag}.summary.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    title = f"# Width-Matched Control: {dataset} top-5 (limit=100)"
    lines = [
        title,
        "",
        "Naming note:",
        "- `nq` resolves to the packaged `nq_rear` files already present in `reproduce/dataset/`.",
        "- `nq_rear` and `popqa` here are the repo/HippoRAG-bundle `1000`-query evaluation subsets.",
        "- `baseline_top10_plus_ce` = width-matched no-append CE control over the top-10 baseline prefix.",
        "- `bridge_append_plus_ce` = same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`.",
        "- `random3_deep_plus_ce` = matched random-append CE control.",
        "",
        "| Dataset | Run | EM | F1 | R@5 | R@20 | R@100 | num_queries | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_row_to_md(row) for row in rows)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_merged_summary(rows: list[dict[str, Any]], date_tag: str) -> None:
    json_path = RUN_ROOT / f"width_matched_control_nq_popqa_100_{date_tag}.summary.json"
    md_path = RUN_ROOT / f"width_matched_control_nq_popqa_100_{date_tag}.summary.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Width-Matched Control: NQ + PopQA top-5 (limit=100)",
        "",
        "Naming note:",
        "- `nq` resolves to the packaged `nq_rear` files already present in `reproduce/dataset/`.",
        "- `nq_rear` and `popqa` here are the repo/HippoRAG-bundle `1000`-query evaluation subsets.",
        "- `baseline_top10_plus_ce` = width-matched no-append CE control over the top-10 baseline prefix.",
        "- `bridge_append_plus_ce` = same-pool pure CE rerank over `top-10 baseline prefix + 3 bridge-appended docs`.",
        "- `random3_deep_plus_ce` = matched random-append CE control.",
        "",
        "| Dataset | Run | EM | F1 | R@5 | R@20 | R@100 | num_queries | ΔEM vs baseline top-5 | ΔEM vs baseline top-10+CE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_row_to_md(row) for row in rows)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    merged_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        rows = collect_dataset_rows(dataset=dataset, date_tag=args.date_tag, limit=args.limit)
        write_dataset_summary(dataset=dataset, rows=rows, date_tag=args.date_tag)
        merged_rows.extend(rows)
    write_merged_summary(rows=merged_rows, date_tag=args.date_tag)


if __name__ == "__main__":
    main()
