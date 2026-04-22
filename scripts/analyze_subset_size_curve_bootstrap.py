#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def percentile_interval(values: List[float], alpha: float) -> Dict[str, float]:
    lower = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
    }


def bootstrap_mean_ci(samples: np.ndarray, num_bootstrap: int, alpha: float, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if samples.size == 0:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
        }
    boot = []
    n = int(samples.size)
    for _ in range(int(num_bootstrap)):
        resample = samples[rng.integers(0, n, size=n)]
        boot.append(float(np.mean(resample)))
    interval = percentile_interval(boot, alpha)
    return {
        "mean": round(float(np.mean(samples)), 4),
        "lower": interval["lower"],
        "upper": interval["upper"],
    }


def extract_per_query(report: Dict, condition: str, metric: str) -> Dict[int, np.ndarray]:
    by_size: Dict[int, List[float]] = {}
    for row in list(report.get("query_records", []) or []):
        if not row.get("eligible"):
            continue
        curve = ((row.get("curve_metrics") or {}).get(condition) or {})
        for size_str, metrics in curve.items():
            size = int(size_str)
            by_size.setdefault(size, []).append(float(metrics.get(metric, 0.0)))
    return {
        size: np.asarray(values, dtype=float)
        for size, values in sorted(by_size.items())
    }


def build_markdown(summary: Dict) -> str:
    lines = [
        f"# Bootstrap CI: {summary['dataset']} / {summary['condition']}",
        "",
        f"- Metric: `{summary['metric']}`",
        f"- Bootstrap samples: `{summary['num_bootstrap']}`",
        f"- Eligible queries: `{summary['eligible_count']}`",
        "",
        "## Mean CI by Size",
        "",
        "| |S| | mean | 95% CI |",
        "|---:|---:|---:|---:|",
    ]
    for size in sorted(summary["size_stats"], key=int):
        stat = summary["size_stats"][size]
        lines.append(
            f"| {size} | {stat['mean']:.4f} | [{stat['lower']:.4f}, {stat['upper']:.4f}] |"
        )

    lines.extend([
        "",
        "## Pairwise Delta CI",
        "",
        "| Δ | mean | 95% CI |",
        "|---|---:|---:|",
    ])
    for name, stat in summary["pairwise_delta_stats"].items():
        lines.append(
            f"| {name} | {stat['mean']:+.4f} | [{stat['lower']:+.4f}, {stat['upper']:+.4f}] |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for subset-size curve reports.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--condition", default="oracle_chain_padded")
    parser.add_argument("--metric", default="F1", choices=["F1", "ExactMatch"])
    parser.add_argument("--num_bootstrap", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args()

    report_path = Path(args.input_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_query = extract_per_query(report, condition=str(args.condition), metric=str(args.metric))
    if not per_query:
        raise ValueError(f"No per-query metrics found for condition={args.condition}")

    size_stats = {
        str(size): bootstrap_mean_ci(
            samples=values,
            num_bootstrap=int(args.num_bootstrap),
            alpha=float(args.alpha),
            seed=int(args.seed) + int(size),
        )
        for size, values in per_query.items()
    }

    rng = np.random.default_rng(int(args.seed))
    pairwise_delta_stats = {}
    sizes = sorted(per_query.keys())
    for left, right in zip(sizes[:-1], sizes[1:]):
        left_values = per_query[left]
        right_values = per_query[right]
        n = int(min(left_values.size, right_values.size))
        deltas = right_values[:n] - left_values[:n]
        boot = []
        for _ in range(int(args.num_bootstrap)):
            resample = deltas[rng.integers(0, n, size=n)]
            boot.append(float(np.mean(resample)))
        interval = percentile_interval(boot, float(args.alpha))
        pairwise_delta_stats[f"{left}->{right}"] = {
            "mean": round(float(np.mean(deltas)), 4),
            "lower": interval["lower"],
            "upper": interval["upper"],
        }

    summary = {
        "dataset": report.get("dataset"),
        "condition": str(args.condition),
        "metric": str(args.metric),
        "eligible_count": int(report.get("eligible_count", 0)),
        "num_bootstrap": int(args.num_bootstrap),
        "size_stats": size_stats,
        "pairwise_delta_stats": pairwise_delta_stats,
        "source_report": str(report_path),
    }

    output_json = Path(args.output_json) if args.output_json else report_path.with_name(report_path.stem + "_bootstrap.json")
    output_md = Path(args.output_md) if args.output_md else report_path.with_name(report_path.stem + "_bootstrap.md")
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
