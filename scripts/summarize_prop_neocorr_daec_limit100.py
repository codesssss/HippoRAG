#!/usr/bin/env python3
"""Summarize PropRAG/DAEC/NeocorRAG limit-100 comparison JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DATASETS = ["2wikimultihopqa", "hotpotqa", "musique"]
METHODS = [
    ("proprag", "PropRAG"),
    ("daec", "PropRAG + DAEC"),
    ("daec_l1", "PropRAG + DAEC-L1"),
    ("neocorrag", "NEOCORRAG"),
]


def _load(path: Path) -> Dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _round(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except Exception:
        return None


def summarize_file(path: Path, dataset: str, variant: str, label: str) -> Dict[str, Any]:
    data = _load(path)
    row: Dict[str, Any] = {
        "dataset": dataset,
        "method": label,
        "variant": variant,
        "path": str(path),
        "status": "missing" if data is None else "done",
        "Recall@5": None,
        "Recall@20": None,
        "EM": None,
        "F1": None,
    }
    if data is None:
        return row

    if variant == "proprag":
        metrics = data.get("overall_recomputed") or {}
        row.update(
            {
                "Recall@5": _round(metrics.get("Recall@5")),
                "Recall@20": _round(metrics.get("Recall@20")),
                "EM": _round(metrics.get("ExactMatch")),
                "F1": _round(metrics.get("F1")),
            }
        )
    elif variant in {"daec", "daec_l1"}:
        selector = data.get("setwise_selector_qa") or {}
        retrieval = selector.get("selector_retrieval_metrics") or {}
        row.update(
            {
                "Recall@5": _round(retrieval.get("Recall@5")),
                "Recall@20": _round(retrieval.get("Recall@20")),
                "EM": _round(selector.get("selector_EM")),
                "F1": _round(selector.get("selector_F1")),
                "baseline_EM": _round(selector.get("baseline_EM")),
                "baseline_F1": _round(selector.get("baseline_F1")),
                "EM_delta_vs_prop": _round(selector.get("EM_delta")),
                "F1_delta_vs_prop": _round(selector.get("F1_delta")),
            }
        )
    elif variant == "neocorrag":
        retrieval = data.get("overall_retrieval_result") or {}
        qa = data.get("overall_qa_results") or {}
        row.update(
            {
                "Recall@5": _round(retrieval.get("Recall@5")),
                "Recall@20": _round(retrieval.get("Recall@20")),
                "EM": _round(qa.get("ExactMatch")),
                "F1": _round(qa.get("F1")),
            }
        )
    return row


def build_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for variant, label in METHODS:
            rows.append(summarize_file(results_dir / f"{dataset}_{variant}.json", dataset, variant, label))
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown(rows: List[Dict[str, Any]], results_dir: Path) -> str:
    lines = [
        "# PropRAG / DAEC / NEOCORRAG Limit-100 Aligned Smoke",
        "",
        f"Results dir: `{results_dir}`",
        "",
        "| Dataset | Method | Status | Recall@5 | Recall@20 | EM | F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {status} | {r5} | {r20} | {em} | {f1} |".format(
                dataset=row["dataset"],
                method=row["method"],
                status=row["status"],
                r5=_fmt(row["Recall@5"]),
                r20=_fmt(row["Recall@20"]),
                em=_fmt(row["EM"]),
                f1=_fmt(row["F1"]),
            )
        )
    lines.extend(
        [
            "",
            "Protocol notes:",
            "",
            "- PropRAG rows use the fixed PropRAG top-100 pool exported under `run_logs/proprag_pool_exports_full1000_20260424/`.",
            "- `PropRAG + DAEC` uses `eval_causal_qwen3.py --setwise_selector dtc_embed` with the prior original-DAEC PropRAG-pool flags.",
            "- `PropRAG + DAEC-L1` uses `--setwise_selector daec_noisyor` on the same PropRAG pool.",
            "- NEOCORRAG is run natively through `scripts/run_neocorrag_aligned.py`; retrieval Recall@5/20 is from NeocorRAG retrieval, EM/F1 from its QA output.",
            "- All rows are `limit=100`, `qa_top_k=5`, NV-Embed endpoint `localhost:8019`, and Qwen3-8B API reader where applicable.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    rows = build_rows(args.results_dir)
    payload = {"results_dir": str(args.results_dir), "rows": rows}
    md = build_markdown(rows, args.results_dir)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
