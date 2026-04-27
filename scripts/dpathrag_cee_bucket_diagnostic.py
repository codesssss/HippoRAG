#!/usr/bin/env python3
"""Bucket-level diagnostics for learned Conservative Evidence Editing outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dpathrag_analyze_hard_negatives import doc_features, is_gold, load_embedding_payload, load_jsonl, row_qid  # noqa: E402
from dpathrag_analyze_hard_negatives_v2 import classify_hard_negative, oracle_edit_sequence, percentile  # noqa: E402
from src.dpathrag.io import write_json  # noqa: E402
from src.dpathrag.selector_data import featurize_selector_record, summarize_selector_metrics, support_metrics_for_indices  # noqa: E402


BUCKETS = (
    "rank_complete_and_edited",
    "rank_complete_and_stopped",
    "rank_incomplete_and_edited",
    "rank_incomplete_and_stopped",
    "oracle_beneficial_and_edited",
    "oracle_beneficial_and_stopped",
    "oracle_not_beneficial_and_edited",
    "oracle_not_beneficial_and_stopped",
)


def unique_indices(values: Sequence[Any], *, candidate_count: int, top_k: int) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        index = int(value)
        if 0 <= index < int(candidate_count) and index not in seen:
            output.append(index)
            seen.add(index)
        if len(output) >= int(top_k):
            break
    return output


def edit_counts(record: dict[str, Any], edits: Sequence[dict[str, Any]], *, max_candidates: int) -> dict[str, int]:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    counts = {"added_gold": 0, "added_non_gold": 0, "removed_gold": 0, "removed_non_gold": 0}
    for edit in edits:
        add_index = int(edit["add_index"])
        remove_index = int(edit["remove_index"])
        if not (0 <= add_index < len(candidates) and 0 <= remove_index < len(candidates)):
            continue
        counts["added_gold"] += int(is_gold(candidates[add_index]))
        counts["added_non_gold"] += int(not is_gold(candidates[add_index]))
        counts["removed_gold"] += int(is_gold(candidates[remove_index]))
        counts["removed_non_gold"] += int(not is_gold(candidates[remove_index]))
    return counts


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "queries": 0,
            "support_complete": 0.0,
            "support_recall": 0.0,
            "avg_support_complete_delta": 0.0,
            "avg_support_recall_delta": 0.0,
            "added_gold": 0,
            "added_non_gold": 0,
            "removed_gold": 0,
            "removed_non_gold": 0,
            "non_gold_per_gold": 0.0,
            "hard_negative_types": {},
        }
    metrics = summarize_selector_metrics([row["metrics"] for row in rows])
    added_gold = sum(int(row["added_gold"]) for row in rows)
    added_non_gold = sum(int(row["added_non_gold"]) for row in rows)
    hard_types: dict[str, int] = {}
    for row in rows:
        for name, count in row.get("hard_negative_types", {}).items():
            hard_types[name] = hard_types.get(name, 0) + int(count)
    return {
        **metrics,
        "queries": len(rows),
        "avg_support_complete_delta": round(sum(float(row["support_complete_delta"]) for row in rows) / len(rows), 6),
        "avg_support_recall_delta": round(sum(float(row["support_recall_delta"]) for row in rows) / len(rows), 6),
        "added_gold": added_gold,
        "added_non_gold": added_non_gold,
        "removed_gold": sum(int(row["removed_gold"]) for row in rows),
        "removed_non_gold": sum(int(row["removed_non_gold"]) for row in rows),
        "non_gold_per_gold": round(float(added_non_gold) / max(1.0, float(added_gold)), 4),
        "hard_negative_types": dict(sorted(hard_types.items())),
    }


def analyze_config(
    cache_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    *,
    config_name: str,
    variant: str,
    top_k: int,
    max_candidates: int,
    oracle_pool_size: int,
    embedding_by_qid: dict[str, Any],
    embedding_feature_names: Sequence[str],
) -> dict[str, Any]:
    cache_by_qid = {row_qid(row): row for row in cache_rows}
    filtered = [row for row in prediction_rows if str(row.get("variant") or variant) == str(variant)]
    case_rows: list[dict[str, Any]] = []
    added_non_gold_features: list[dict[str, Any]] = []
    for prediction in filtered:
        qid = row_qid(prediction)
        record = cache_by_qid[qid]
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        example = featurize_selector_record(record, max_candidates=max_candidates, path_len=top_k)
        rank_indices = list(range(min(int(top_k), len(candidates))))
        selected_indices = unique_indices(prediction.get("selected_indices") or [], candidate_count=len(candidates), top_k=top_k)
        rank_metrics = support_metrics_for_indices(example, rank_indices)
        metrics = support_metrics_for_indices(example, selected_indices)
        edits = list(prediction.get("edits") or [])
        counts = edit_counts(record, edits, max_candidates=max_candidates)
        hard_features = []
        for edit in edits:
            add_index = int(edit["add_index"])
            if 0 <= add_index < len(candidates) and not is_gold(candidates[add_index]):
                features = doc_features(
                    record,
                    candidates[add_index],
                    candidate_index=add_index,
                    embedding_by_qid=embedding_by_qid,
                    embedding_feature_names=embedding_feature_names,
                )
                hard_features.append(features)
                added_non_gold_features.append(features)
        oracle = oracle_edit_sequence(
            record,
            top_k=top_k,
            candidate_pool_size=oracle_pool_size,
            max_candidates=max_candidates,
            steps=1,
            example=example,
        )
        case_rows.append(
            {
                "config": config_name,
                "qid": qid,
                "rank_complete": float(rank_metrics.get("support_complete") or 0.0) >= 1.0,
                "edited": bool(edits),
                "oracle_beneficial": bool(oracle.get("edits")),
                "metrics": metrics,
                "support_complete_delta": float(metrics.get("support_complete") or 0.0)
                - float(rank_metrics.get("support_complete") or 0.0),
                "support_recall_delta": float(metrics.get("support_recall") or 0.0) - float(rank_metrics.get("support_recall") or 0.0),
                **counts,
                "_hard_features": hard_features,
            }
        )
    semantic_threshold = percentile([float(row.get("q_doc_cosine") or 0.0) for row in added_non_gold_features], 0.75)
    for row in case_rows:
        type_counts: dict[str, int] = {}
        for features in row.pop("_hard_features"):
            name = classify_hard_negative(features, semantic_threshold=semantic_threshold)
            type_counts[name] = type_counts.get(name, 0) + 1
        row["hard_negative_types"] = type_counts
    buckets = {name: [] for name in BUCKETS}
    for row in case_rows:
        buckets["rank_complete_and_edited" if row["rank_complete"] and row["edited"] else "rank_complete_and_stopped" if row["rank_complete"] else "rank_incomplete_and_edited" if row["edited"] else "rank_incomplete_and_stopped"].append(row)
        buckets[
            "oracle_beneficial_and_edited"
            if row["oracle_beneficial"] and row["edited"]
            else "oracle_beneficial_and_stopped"
            if row["oracle_beneficial"]
            else "oracle_not_beneficial_and_edited"
            if row["edited"]
            else "oracle_not_beneficial_and_stopped"
        ].append(row)
    return {
        "config": config_name,
        "rows": len(case_rows),
        "variant": variant,
        "semantic_threshold_q_doc_cosine_p75": round(float(semantic_threshold), 6),
        "overall": summarize_rows(case_rows),
        "buckets": {name: summarize_rows(rows) for name, rows in buckets.items()},
        "case_rows": case_rows,
    }


def write_case_csv(configs: Sequence[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "qid",
        "rank_complete",
        "oracle_beneficial",
        "edited",
        "support_complete_delta",
        "support_recall_delta",
        "added_gold",
        "added_non_gold",
        "removed_gold",
        "removed_non_gold",
        "hard_negative_types",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            for row in config.get("case_rows", []):
                payload = {key: row.get(key, "") for key in fieldnames}
                payload["hard_negative_types"] = json.dumps(payload["hard_negative_types"], sort_keys=True)
                writer.writerow(payload)


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# CEE Bucket Diagnostic",
        "",
        f"- Rows per config: `{payload['rows']}`",
        f"- Variant: `{payload['variant']}`",
        "",
    ]
    for config in payload["configs"]:
        lines.extend(
            [
                f"## {config['config']}",
                "",
                "| Bucket | Queries | Support Complete | Avg ΔComplete | Avg ΔRecall | Added Gold | Added Non-Gold | Non-Gold/Gold | Hard Negative Types |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for name in BUCKETS:
            item = config["buckets"][name]
            hard = ", ".join(f"{key}:{value}" for key, value in item.get("hard_negative_types", {}).items())
            lines.append(
                f"| {name} | {item['queries']} | {item.get('support_complete', 0.0):.4f} | "
                f"{item['avg_support_complete_delta']:+.4f} | {item['avg_support_recall_delta']:+.4f} | "
                f"{item['added_gold']} | {item['added_non_gold']} | {item['non_gold_per_gold']:.4f} | {hard} |"
            )
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--prediction_jsonl", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--embedding_npz", default="")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--variant", default="learned_edit1")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--oracle_pool_size", type=int, default=20)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_cases_csv", default="")
    args = parser.parse_args()

    cache_rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    embedding_by_qid, embedding_feature_names = load_embedding_payload(args.embedding_npz, limit=int(args.limit))
    configs = []
    for spec in args.prediction_jsonl:
        if "=" not in spec:
            raise ValueError(f"--prediction_jsonl must be NAME=PATH, got {spec}")
        name, path = spec.split("=", 1)
        configs.append(
            analyze_config(
                cache_rows,
                load_jsonl(path),
                config_name=name,
                variant=args.variant,
                top_k=int(args.top_k),
                max_candidates=int(args.max_candidates),
                oracle_pool_size=int(args.oracle_pool_size),
                embedding_by_qid=embedding_by_qid,
                embedding_feature_names=embedding_feature_names,
            )
        )
    output = {
        "rows": len(cache_rows),
        "variant": str(args.variant),
        "cache_jsonl": str(args.cache_jsonl),
        "configs": [{key: value for key, value in config.items() if key != "case_rows"} for config in configs],
    }
    write_json(output, args.output_json)
    write_markdown(output, args.output_md)
    if args.output_cases_csv:
        write_case_csv(configs, args.output_cases_csv)
    print(json.dumps({"configs": [config["config"] for config in configs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
