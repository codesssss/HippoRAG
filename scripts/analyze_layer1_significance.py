#!/usr/bin/env python3
"""Paired bootstrap/sign-flip significance analysis for Layer-1 DAEC reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def iter_report_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        else:
            paths.append(path)
    return [path for path in paths if path.exists()]


def percentile_interval(values: Iterable[float], alpha: float) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "lower": round(float(np.percentile(arr, 100.0 * alpha / 2.0)), 6),
        "upper": round(float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0))), 6),
    }


def extract_deltas(report: Dict[str, Any], metric: str) -> np.ndarray:
    deltas: List[float] = []
    for trace in report.get("setwise_selector_query_traces") or []:
        baseline = trace.get("baseline_metrics") or {}
        selector = trace.get("selector_metrics") or {}
        deltas.append(as_float(selector.get(metric)) - as_float(baseline.get(metric)))
    return np.asarray(deltas, dtype=float)


def stable_offset(*parts: str) -> int:
    digest = hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100000


def paired_sign_flip_pvalue(deltas: np.ndarray, num_samples: int, seed: int) -> float:
    if deltas.size == 0:
        return 1.0
    observed = abs(float(np.mean(deltas)))
    rng = np.random.default_rng(seed)
    count = 0
    batch = 1000
    done = 0
    while done < num_samples:
        take = min(batch, num_samples - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(take, deltas.size))
        means = np.mean(signs * deltas, axis=1)
        count += int(np.sum(np.abs(means) >= observed))
        done += take
    return float((count + 1) / (num_samples + 1))


def bootstrap_ci(deltas: np.ndarray, num_bootstrap: int, alpha: float, seed: int) -> Dict[str, float]:
    if deltas.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    n = int(deltas.size)
    boot = []
    batch = 1000
    done = 0
    while done < num_bootstrap:
        take = min(batch, num_bootstrap - done)
        indices = rng.integers(0, n, size=(take, n))
        boot.extend(np.mean(deltas[indices], axis=1).tolist())
        done += take
    interval = percentile_interval(boot, alpha)
    return {
        "mean": round(float(np.mean(deltas)), 6),
        "lower": interval["lower"],
        "upper": interval["upper"],
    }


def summarize_report(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metric_summaries: Dict[str, Any] = {}
    for metric in args.metrics:
        deltas = extract_deltas(report, metric)
        ci = bootstrap_ci(
            deltas=deltas,
            num_bootstrap=int(args.num_bootstrap),
            alpha=float(args.alpha),
            seed=int(args.seed) + stable_offset(str(path), metric, "bootstrap"),
        )
        pvalue = paired_sign_flip_pvalue(
            deltas=deltas,
            num_samples=int(args.num_signflip),
            seed=int(args.seed) + 17 + stable_offset(str(path), metric, "signflip"),
        )
        metric_summaries[metric] = {
            **ci,
            "p_value_signflip": round(float(pvalue), 6),
            "positive_count": int(np.sum(deltas > 1e-9)),
            "negative_count": int(np.sum(deltas < -1e-9)),
            "zero_count": int(np.sum(np.abs(deltas) <= 1e-9)),
        }

    external_pool = report.get("external_pool") or {}
    return {
        "path": str(path),
        "dataset": report.get("dataset"),
        "pool_source": external_pool.get("source") or external_pool.get("source_name") or external_pool.get("external_pool_source_name") or "",
        "num_traces": int(len(report.get("setwise_selector_query_traces") or [])),
        "metrics": metric_summaries,
    }


def write_markdown(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    headers = [
        "Dataset",
        "Pool",
        "N",
        "dF1 mean",
        "dF1 95% CI",
        "F1 p",
        "F1 +/-/0",
        "dEM mean",
        "dEM 95% CI",
        "EM p",
        "EM +/-/0",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        f1 = row["metrics"].get("F1") or {}
        em = row["metrics"].get("ExactMatch") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset")),
                    str(row.get("pool_source")),
                    str(row.get("num_traces")),
                    f"{f1.get('mean', 0.0):+.6f}",
                    f"[{f1.get('lower', 0.0):+.6f}, {f1.get('upper', 0.0):+.6f}]",
                    f"{f1.get('p_value_signflip', 1.0):.6f}",
                    f"{f1.get('positive_count', 0)}/{f1.get('negative_count', 0)}/{f1.get('zero_count', 0)}",
                    f"{em.get('mean', 0.0):+.6f}",
                    f"[{em.get('lower', 0.0):+.6f}, {em.get('upper', 0.0):+.6f}]",
                    f"{em.get('p_value_signflip', 1.0):.6f}",
                    f"{em.get('positive_count', 0)}/{em.get('negative_count', 0)}/{em.get('zero_count', 0)}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Evaluation report JSON files or directories.")
    parser.add_argument("--output_dir", type=Path, default=Path("run_logs/layer1_significance_20260424"))
    parser.add_argument("--metrics", nargs="+", default=["F1", "ExactMatch"])
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--num_signflip", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_report(path, args) for path in iter_report_paths(args.reports)]
    (args.output_dir / "layer1_significance.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(rows, args.output_dir / "layer1_significance.md")
    print((args.output_dir / "layer1_significance.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
