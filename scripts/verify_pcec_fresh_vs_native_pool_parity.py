#!/usr/bin/env python3
"""Verify PCEC fresh E2E output against native-pool PCEC output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evidence_transition_graphragv4_composition.readout import write_json  # noqa: E402


DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DEFAULT_FRESH_ROOT = Path("run_logs/pcec_fresh_e2e/evals")
DEFAULT_NATIVE_ROOT = Path("run_logs/evidence_transition_graphragv4_composition_native_pool/evals")
DEFAULT_OUTPUT_JSON = Path("run_logs/pcec_fresh_e2e/fresh_vs_native_pool_parity.json")
DEFAULT_OUTPUT_MD = Path("run_logs/pcec_fresh_e2e/fresh_vs_native_pool_parity.md")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_csv(value: str | Sequence[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    return [str(item).strip() for item in raw if str(item).strip()]


def fresh_path(root: Path, dataset: str, pool_k: int, prefix_m: int, reader_k: int, limit: int) -> Path:
    residual = int(reader_k) - int(prefix_m)
    return Path(root) / f"{dataset}_pcec_fresh_e2e_prefix{prefix_m}_residual{residual}_pool{pool_k}_limit{limit}.json"


def native_path(root: Path, dataset: str, pool_k: int, prefix_m: int, reader_k: int, limit: int) -> Path:
    residual = int(reader_k) - int(prefix_m)
    return Path(root) / f"{dataset}_pcec_native_pool_prefix{prefix_m}_residual{residual}_pool{pool_k}_limit{limit}.json"


def row_map(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows: dict[int, Mapping[str, Any]] = {}
    for row in list(payload.get("rows") or []):
        if not isinstance(row, Mapping):
            continue
        rows[int(row.get("query_index", row.get("query_idx", len(rows))) or 0)] = row
    return rows


def gain(row: Mapping[str, Any]) -> float | None:
    value = ((row.get("pcec_readout") or {}).get("best_candidate_gain"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_rows(fresh_row: Mapping[str, Any], native_row: Mapping[str, Any]) -> dict[str, Any]:
    fresh_readout = dict(fresh_row.get("pcec_readout") or {})
    native_readout = dict(native_row.get("pcec_readout") or {})
    fresh_gain = gain(fresh_row)
    native_gain = gain(native_row)
    gain_delta = 0.0 if fresh_gain is None or native_gain is None else abs(float(fresh_gain) - float(native_gain))
    return {
        "question_match": str(fresh_row.get("question") or "") == str(native_row.get("question") or ""),
        "final_top5_title_match": list(fresh_row.get("retrieved_titles_top5") or []) == list(native_row.get("retrieved_titles_top5") or []),
        "final_top5_doc_id_match": list(fresh_row.get("retrieved_doc_indices_top5") or []) == list(native_row.get("retrieved_doc_indices_top5") or []),
        "decision_match": fresh_readout.get("decision") == native_readout.get("decision"),
        "admitted_position_match": list(fresh_readout.get("admitted_positions") or []) == list(native_readout.get("admitted_positions") or []),
        "prefix_preserved_match": bool(fresh_readout.get("prefix_preserved")) == bool(native_readout.get("prefix_preserved")),
        "best_candidate_gain_abs_delta": round(float(gain_delta), 12),
    }


def summarize(comparisons: Sequence[Mapping[str, Any]], *, gain_eps: float) -> dict[str, Any]:
    fields = [
        "question_match",
        "final_top5_title_match",
        "final_top5_doc_id_match",
        "decision_match",
        "admitted_position_match",
        "prefix_preserved_match",
    ]
    count = len(comparisons)
    summary: dict[str, Any] = {"count": int(count)}
    for field in fields:
        matches = sum(bool(row.get(field)) for row in comparisons)
        summary[f"{field}_count"] = int(matches)
        summary[f"{field}_rate"] = round(float(matches) / float(max(count, 1)), 6)
    gain_deltas = [float(row.get("best_candidate_gain_abs_delta", 0.0) or 0.0) for row in comparisons]
    summary["best_candidate_gain_max_abs_delta"] = max(gain_deltas) if gain_deltas else 0.0
    summary["best_candidate_gain_mean_abs_delta"] = round(mean(gain_deltas), 12) if gain_deltas else 0.0
    summary["best_candidate_gain_within_eps_count"] = int(sum(delta <= float(gain_eps) for delta in gain_deltas))
    summary["all_exact_count"] = int(
        sum(
            all(bool(row.get(field)) for field in fields)
            and float(row.get("best_candidate_gain_abs_delta", 0.0) or 0.0) <= float(gain_eps)
            for row in comparisons
        )
    )
    summary["all_exact_rate"] = round(float(summary["all_exact_count"]) / float(max(count, 1)), 6)
    return summary


def run_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    fresh_json = Path(args.fresh_json) if args.fresh_json and len(parse_csv(args.datasets)) == 1 else fresh_path(
        Path(args.fresh_root),
        dataset,
        int(args.pool_k),
        int(args.prefix_budget_m),
        int(args.reader_budget_k),
        int(args.limit),
    )
    native_json = Path(args.native_json) if args.native_json and len(parse_csv(args.datasets)) == 1 else native_path(
        Path(args.native_root),
        dataset,
        int(args.pool_k),
        int(args.prefix_budget_m),
        int(args.reader_budget_k),
        int(args.limit),
    )
    fresh_payload = read_json(fresh_json)
    native_payload = read_json(native_json)
    fresh_rows = row_map(fresh_payload)
    native_rows = row_map(native_payload)
    common = sorted(set(fresh_rows) & set(native_rows))
    comparisons = [compare_rows(fresh_rows[idx], native_rows[idx]) for idx in common]
    mismatches = []
    for idx, comparison in zip(common, comparisons):
        if (
            not bool(comparison.get("final_top5_title_match"))
            or not bool(comparison.get("decision_match"))
            or float(comparison.get("best_candidate_gain_abs_delta", 0.0)) > float(args.gain_eps)
        ):
            mismatches.append(
                {
                    "query_idx": int(idx),
                    "fresh_titles": list(fresh_rows[idx].get("retrieved_titles_top5") or []),
                    "native_titles": list(native_rows[idx].get("retrieved_titles_top5") or []),
                    "fresh_decision": (fresh_rows[idx].get("pcec_readout") or {}).get("decision"),
                    "native_decision": (native_rows[idx].get("pcec_readout") or {}).get("decision"),
                    "gain_delta": comparison.get("best_candidate_gain_abs_delta"),
                }
            )
            if len(mismatches) >= int(args.max_examples):
                break
    return {
        "dataset": dataset,
        "fresh_json": str(fresh_json),
        "native_json": str(native_json),
        "summary": summarize(comparisons, gain_eps=float(args.gain_eps)),
        "mismatch_examples": mismatches,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# PCEC Fresh E2E vs Native-Pool Parity",
        "",
        "| Dataset | Count | All exact | Top5 titles | Decisions | Gain max delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload.get("datasets", []) or []:
        summary = result.get("summary", {}) or {}
        lines.append(
            "| {dataset} | {count} | {all_exact} | {top5} | {decision} | {gain} |".format(
                dataset=result.get("dataset", ""),
                count=int(summary.get("count", 0)),
                all_exact=int(summary.get("all_exact_count", 0)),
                top5=int(summary.get("final_top5_title_match_count", 0)),
                decision=int(summary.get("decision_match_count", 0)),
                gain=summary.get("best_candidate_gain_max_abs_delta", 0.0),
            )
        )
    for result in payload.get("datasets", []) or []:
        examples = list(result.get("mismatch_examples") or [])
        if not examples:
            continue
        lines.extend(["", f"## {result.get('dataset')} Mismatches", ""])
        for item in examples:
            lines.append(f"- `{item.get('query_idx')}` gain_delta={item.get('gain_delta')}")
            lines.append(f"  - Fresh: {item.get('fresh_titles')} decision={item.get('fresh_decision')}")
            lines.append(f"  - Native: {item.get('native_titles')} decision={item.get('native_decision')}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--reader-budget-k", type=int, default=5)
    parser.add_argument("--prefix-budget-m", type=int, default=4)
    parser.add_argument("--fresh-root", type=Path, default=DEFAULT_FRESH_ROOT)
    parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE_ROOT)
    parser.add_argument("--fresh-json", type=Path, default=None)
    parser.add_argument("--native-json", type=Path, default=None)
    parser.add_argument("--gain-eps", type=float, default=1e-9)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-examples", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    results = [run_dataset(args, dataset) for dataset in parse_csv(args.datasets)]
    output = {
        "datasets": results,
        "config": {
            "limit": int(args.limit),
            "pool_k": int(args.pool_k),
            "reader_budget_k": int(args.reader_budget_k),
            "prefix_budget_m": int(args.prefix_budget_m),
            "gain_eps": float(args.gain_eps),
        },
    }
    write_json(output, args.output_json)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(output), encoding="utf-8")
    print(
        json.dumps(
            {
                result["dataset"]: {
                    "all_exact": result["summary"]["all_exact_count"],
                    "count": result["summary"]["count"],
                    "top5": result["summary"]["final_top5_title_match_count"],
                }
                for result in results
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
