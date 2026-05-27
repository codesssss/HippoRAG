#!/usr/bin/env python3
"""Compute gold-support bridge recall from existing EvLink artifacts.

This is a lightweight companion to ``compute_evidencelink_fcrg.py``.  It uses
the same fixed OpenIE/certificate graph and the same full1000 PCEC rows, but
reports link-coverage diagnostics directly instead of rank gain.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from compute_evidencelink_fcrg import (
    DATASET_LABELS,
    DATASETS,
    build_fact_edges,
    gold_rows_for_dataset,
)


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def unique_ints(values: Sequence[Any]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def bridge_metrics(
    *,
    fact_edges: Mapping[int, Sequence[tuple[int, int]]],
    gold_docs: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    queries = 0
    queries_with_gold_pair = 0
    queries_with_connected_pair = 0
    gold_docs_total = 0
    gold_docs_covered = 0
    unordered_pairs_total = 0
    unordered_pairs_connected = 0
    directed_pairs_total = 0
    directed_pairs_connected = 0

    for qid, raw_gold in gold_docs.items():
        gold = unique_ints(raw_gold)
        gold_set = set(gold)
        queries += 1
        gold_docs_total += len(gold)
        edges = {
            (int(source), int(target))
            for source, target in fact_edges.get(int(qid), [])
            if int(source) in gold_set
            and int(target) in gold_set
            and int(source) != int(target)
        }
        covered_targets = {target for _source, target in edges}
        gold_docs_covered += len(covered_targets)

        unordered_pairs = {tuple(sorted(pair)) for pair in itertools.combinations(gold, 2)}
        connected_unordered = {
            pair
            for pair in unordered_pairs
            if (pair[0], pair[1]) in edges or (pair[1], pair[0]) in edges
        }
        unordered_pairs_total += len(unordered_pairs)
        unordered_pairs_connected += len(connected_unordered)
        if unordered_pairs:
            queries_with_gold_pair += 1
        if connected_unordered:
            queries_with_connected_pair += 1

        directed_pairs = {
            (source, target)
            for source in gold
            for target in gold
            if int(source) != int(target)
        }
        directed_pairs_total += len(directed_pairs)
        directed_pairs_connected += len(edges & directed_pairs)

    return {
        "queries": queries,
        "queries_with_gold_pair": queries_with_gold_pair,
        "queries_with_connected_pair": queries_with_connected_pair,
        "query_connected_pair_rate": (
            0.0
            if queries_with_gold_pair == 0
            else queries_with_connected_pair / float(queries_with_gold_pair)
        ),
        "gold_docs_total": gold_docs_total,
        "gold_docs_covered": gold_docs_covered,
        "gold_support_coverage": (
            0.0 if gold_docs_total == 0 else gold_docs_covered / float(gold_docs_total)
        ),
        "gold_pairs_total": unordered_pairs_total,
        "gold_pairs_connected": unordered_pairs_connected,
        "gold_bridge_pair_recall": (
            0.0
            if unordered_pairs_total == 0
            else unordered_pairs_connected / float(unordered_pairs_total)
        ),
        "directed_gold_pairs_total": directed_pairs_total,
        "directed_gold_pairs_connected": directed_pairs_connected,
        "directed_gold_bridge_recall": (
            0.0
            if directed_pairs_total == 0
            else directed_pairs_connected / float(directed_pairs_total)
        ),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "dataset",
        "gold_bridge_pair_recall_pct",
        "gold_support_coverage_pct",
        "query_connected_pair_rate_pct",
        "gold_pairs_connected",
        "gold_pairs_total",
        "gold_docs_covered",
        "gold_docs_total",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("run_logs/evidencelink_gold_bridge_recall_20260526"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--certificate-policy", default="canonical_source_text")
    parser.add_argument(
        "--include-non-triple-certificates",
        action="store_true",
        help="Include title/source certificates without OpenIE triples.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_root = args.output_root.expanduser()
    require_triple = not bool(args.include_non_triple_certificates)

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    aggregate_counts = {
        "queries": 0,
        "queries_with_gold_pair": 0,
        "queries_with_connected_pair": 0,
        "gold_docs_total": 0,
        "gold_docs_covered": 0,
        "gold_pairs_total": 0,
        "gold_pairs_connected": 0,
        "directed_gold_pairs_total": 0,
        "directed_gold_pairs_connected": 0,
    }

    for dataset in args.datasets:
        gold_rows, openie_path = gold_rows_for_dataset(root, dataset)
        fact_edges, gold_docs, edge_metadata = build_fact_edges(
            rows=gold_rows,
            openie_path=openie_path,
            certificate_policy=str(args.certificate_policy),
            require_triple=require_triple,
        )
        metrics = bridge_metrics(fact_edges=fact_edges, gold_docs=gold_docs)
        metadata[dataset] = edge_metadata
        for key in aggregate_counts:
            aggregate_counts[key] += int(metrics[key])
        rows.append(
            {
                "dataset_key": dataset,
                "dataset": DATASET_LABELS.get(dataset, dataset),
                **metrics,
                "gold_bridge_pair_recall_pct": pct(metrics["gold_bridge_pair_recall"]),
                "gold_support_coverage_pct": pct(metrics["gold_support_coverage"]),
                "query_connected_pair_rate_pct": pct(metrics["query_connected_pair_rate"]),
                "directed_gold_bridge_recall_pct": pct(metrics["directed_gold_bridge_recall"]),
            }
        )

    aggregate_metrics = {
        **aggregate_counts,
        "query_connected_pair_rate": (
            0.0
            if aggregate_counts["queries_with_gold_pair"] == 0
            else aggregate_counts["queries_with_connected_pair"]
            / float(aggregate_counts["queries_with_gold_pair"])
        ),
        "gold_support_coverage": (
            0.0
            if aggregate_counts["gold_docs_total"] == 0
            else aggregate_counts["gold_docs_covered"]
            / float(aggregate_counts["gold_docs_total"])
        ),
        "gold_bridge_pair_recall": (
            0.0
            if aggregate_counts["gold_pairs_total"] == 0
            else aggregate_counts["gold_pairs_connected"]
            / float(aggregate_counts["gold_pairs_total"])
        ),
        "directed_gold_bridge_recall": (
            0.0
            if aggregate_counts["directed_gold_pairs_total"] == 0
            else aggregate_counts["directed_gold_pairs_connected"]
            / float(aggregate_counts["directed_gold_pairs_total"])
        ),
    }
    aggregate_row = {
        "dataset_key": "avg",
        "dataset": "Aggregate",
        **aggregate_metrics,
        "gold_bridge_pair_recall_pct": pct(aggregate_metrics["gold_bridge_pair_recall"]),
        "gold_support_coverage_pct": pct(aggregate_metrics["gold_support_coverage"]),
        "query_connected_pair_rate_pct": pct(aggregate_metrics["query_connected_pair_rate"]),
        "directed_gold_bridge_recall_pct": pct(aggregate_metrics["directed_gold_bridge_recall"]),
    }
    display_rows = [*rows, aggregate_row]

    payload = {
        "protocol": {
            "gold_bridge_pair_recall": (
                "unordered gold support pairs with at least one source-grounded "
                "EvLink certificate in either direction divided by all unordered "
                "gold support pairs"
            ),
            "gold_support_coverage": (
                "gold support documents that are the target of a source-grounded "
                "certificate from another gold support document divided by all "
                "gold support documents"
            ),
            "certificate_policy": str(args.certificate_policy),
            "require_openie_triple_certificate": require_triple,
        },
        "edge_metadata": metadata,
        "rows": rows,
        "aggregate": aggregate_row,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "gold_bridge_recall.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "gold_bridge_recall.csv", display_rows)
    (output_root / "gold_bridge_recall.md").write_text(
        "\n\n".join(
            [
                "# Gold Bridge Recall",
                "Computed from existing EvLink full1000 PCEC rows and the fixed OpenIE certificate graph.",
                markdown_table(display_rows),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "rows": display_rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
