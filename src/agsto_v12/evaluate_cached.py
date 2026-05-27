#!/usr/bin/env python3
"""Evaluate AG-STO v12 from cached STO transition-report proposals.

This is the standalone package runner for the clean paper-facing path. It
reuses cached proposal lanes, runs ``agsto_v12.AGSTORetriever`` with the graph
policy by default, and writes retrieval metrics. It does not expose native
proposal diagnostics, local PPR variants, support_fusion, or QA calls.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .config import AGSTOConfig
from .ranking import unique_ranked
from .retriever import AGSTORetriever


DEFAULT_TRANSITION_REPORT = (
    "outputs_full_sfb_supportfusion_gpt4omini_qwen_cleanbaseline_rebuild_20260425/"
    "reports/transition_component_retriever_full_native_denseanchor20_context10_support400.json"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_path(raw_path: Any, *, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def recall_at_k(gold_doc_indices: Sequence[int], retrieved_doc_indices: Sequence[int], k: int) -> float:
    gold = {int(value) for value in gold_doc_indices or []}
    if not gold:
        return 0.0
    retrieved = {int(value) for value in list(retrieved_doc_indices or [])[: max(int(k), 0)]}
    return len(gold & retrieved) / float(len(gold))


def all_gold_at_k(gold_doc_indices: Sequence[int], retrieved_doc_indices: Sequence[int], k: int) -> bool:
    gold = {int(value) for value in gold_doc_indices or []}
    if not gold:
        return False
    retrieved = {int(value) for value in list(retrieved_doc_indices or [])[: max(int(k), 0)]}
    return gold.issubset(retrieved)


def row_doc_indices(row: Mapping[str, Any], *keys: str, limit: int = 20) -> List[int]:
    for key in keys:
        values = row.get(key)
        if values:
            return unique_ranked(values)[:limit]
    return []


def specificity_doc_indices(row: Mapping[str, Any], *, limit: int) -> List[int]:
    values = row_doc_indices(
        row,
        "specificity_pair_doc_indices_top10",
        "specificity_pair_doc_indices_top5",
        limit=limit,
    )
    if values:
        return values
    emission = row.get("specificity_pairwise_transition_emission")
    if isinstance(emission, Mapping):
        return unique_ranked(emission.get("retrieved_doc_indices", []) or [])[:limit]
    return []


def retrieve_row(
    *,
    retriever: AGSTORetriever,
    row: Mapping[str, Any],
    row_limit: int,
) -> Dict[str, Any]:
    query = str(row.get("question") or row.get("query") or "")
    neighborhood = row.get("query_conditioned_neighborhood", {})
    support_search = row.get("support_set_search", {})
    return retriever.retrieve(
        query=query,
        anchor_doc_indices=row_doc_indices(row, "context_anchor_doc_indices", limit=row_limit),
        native_dense_doc_indices=row_doc_indices(
            row,
            "native_dense_doc_indices_top10",
            "native_dense_doc_indices_top5",
            limit=row_limit,
        ),
        bm25_doc_indices=row_doc_indices(row, "bm25_doc_indices_top10", "bm25_doc_indices_top5", limit=row_limit),
        specificity_doc_indices=specificity_doc_indices(row, limit=row_limit),
        endpoint_transition_doc_indices=row_doc_indices(
            row,
            "endpoint_transition_doc_indices_top10",
            "endpoint_transition_doc_indices_top5",
            limit=row_limit,
        ),
        hybrid_residual_doc_indices=row_doc_indices(
            row,
            "hybrid_residual_pair_doc_indices_top10",
            "hybrid_residual_pair_doc_indices_top5",
            limit=row_limit,
        ),
        query_conditioned_neighborhood=neighborhood if isinstance(neighborhood, Mapping) else {},
        support_set_search=support_search if isinstance(support_search, Mapping) else {},
    )


def evaluate_dataset(
    *,
    dataset_payload: Mapping[str, Any],
    transition_report_dir: Path,
    config: AGSTOConfig,
    max_queries: int,
) -> Dict[str, Any]:
    openie_path = resolve_path(dataset_payload["openie_path"], base_dir=transition_report_dir)
    openie_docs = list(load_json(openie_path).get("docs", []) or [])
    retriever = AGSTORetriever.from_openie_docs(openie_docs, config=config)
    row_limit = max(int(config.retrieval_top_k), int(config.proposal_candidate_depth), 10)

    source_rows = list(dataset_payload.get("rows", []) or [])
    if max_queries > 0:
        source_rows = source_rows[:max_queries]

    rows: List[Dict[str, Any]] = []
    sums: Counter[str] = Counter()
    candidate_counts: List[int] = []
    selected_scores: List[float] = []
    for row in source_rows:
        gold_doc_indices = [int(value) for value in row.get("gold_doc_indices", []) or []]
        result = retrieve_row(retriever=retriever, row=row, row_limit=row_limit)
        result["proposal_source"] = "cached"
        retrieved = [int(doc_idx) for doc_idx in result.get("retrieved_doc_indices", []) or []]
        r5 = recall_at_k(gold_doc_indices, retrieved, 5)
        r10 = recall_at_k(gold_doc_indices, retrieved, 10)
        sums["r5"] += r5
        sums["r10"] += r10
        sums["all_gold_at5"] += 1.0 if all_gold_at_k(gold_doc_indices, retrieved, 5) else 0.0
        sums["all_gold_at10"] += 1.0 if all_gold_at_k(gold_doc_indices, retrieved, 10) else 0.0
        candidate_counts.append(int(result.get("candidate_doc_count", 0) or 0))
        selected_scores.append(float((result.get("selected_evidence_set", {}) or {}).get("score", 0.0) or 0.0))

        augmented_row = dict(row)
        augmented_row["agsto_v12_doc_indices_top5"] = retrieved[:5]
        augmented_row["agsto_v12_doc_indices_top10"] = retrieved[:10]
        augmented_row["agsto_v12_recall_at5"] = round(float(r5), 6)
        augmented_row["agsto_v12_recall_at10"] = round(float(r10), 6)
        augmented_row["agsto_v12_all_gold_at5"] = bool(all_gold_at_k(gold_doc_indices, retrieved, 5))
        augmented_row["agsto_v12_all_gold_at10"] = bool(all_gold_at_k(gold_doc_indices, retrieved, 10))
        augmented_row["agsto_v12"] = result
        rows.append(augmented_row)

    denom = max(len(rows), 1)
    metrics = {
        "agsto_v12_r5": round(float(sums["r5"]) / float(denom), 6),
        "agsto_v12_r10": round(float(sums["r10"]) / float(denom), 6),
        "agsto_v12_all_gold_at5": round(float(sums["all_gold_at5"]) / float(denom), 6),
        "agsto_v12_all_gold_at10": round(float(sums["all_gold_at10"]) / float(denom), 6),
        "mean_candidate_doc_count": round(sum(candidate_counts) / float(max(len(candidate_counts), 1)), 6),
        "mean_selected_evidence_score": round(sum(selected_scores) / float(max(len(selected_scores), 1)), 6),
    }
    return {
        "dataset": dataset_payload.get("dataset"),
        "openie_path": str(openie_path),
        "metrics": {**dict(dataset_payload.get("metrics", {}) or {}), **metrics},
        "rows": rows,
        "corpus_stats": {
            "doc_count": len(openie_docs),
            "unit_count": len(retriever.corpus_index["units"]),
            "endpoint_count": len(retriever.corpus_index["endpoint_to_units"]),
        },
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], aligns: Sequence[str]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(headers[idx])), *(len(row[idx]) for row in string_rows)) if string_rows else len(str(headers[idx]))
        for idx in range(len(headers))
    ]

    def fmt_row(row: Sequence[Any]) -> str:
        cells = []
        for idx, cell in enumerate(row):
            text = str(cell)
            cells.append(text.rjust(widths[idx]) if aligns[idx] == "right" else text.ljust(widths[idx]))
        return "| " + " | ".join(cells) + " |"

    separators = []
    for idx, align in enumerate(aligns):
        dash_count = max(widths[idx], 3)
        if align == "right":
            separators.append("-" * (dash_count - 1) + ":")
        else:
            separators.append("-" * dash_count)
    return "\n".join([fmt_row(headers), "| " + " | ".join(separators) + " |", *[fmt_row(row) for row in rows]])


def write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# AG-STO v12 Cached Evaluation",
        "",
        "Paper-facing cached-proposal reproduction using the standalone `agsto_v12` package.",
        "",
        "## Config",
        "",
    ]
    config_rows = [[f"`{key}`", value] for key, value in (summary.get("config", {}) or {}).items()]
    lines.append(markdown_table(["parameter", "value"], config_rows, ["left", "right"]))
    lines.extend(["", "## Metrics", ""])
    metric_rows = []
    for dataset in summary.get("datasets", []) or []:
        metrics = dataset.get("metrics", {}) or {}
        metric_rows.append(
            [
                dataset.get("dataset"),
                len(dataset.get("rows", []) or []),
                f"{float(metrics.get('agsto_v12_r5', 0.0)):.4f}",
                f"{float(metrics.get('agsto_v12_r10', 0.0)):.4f}",
                f"{float(metrics.get('agsto_v12_all_gold_at5', 0.0)):.4f}",
                f"{float(metrics.get('agsto_v12_all_gold_at10', 0.0)):.4f}",
                f"{float(metrics.get('mean_candidate_doc_count', 0.0)):.1f}",
            ]
        )
    lines.append(
        markdown_table(
            ["dataset", "rows", "R@5", "R@10", "all-gold@5", "all-gold@10", "mean candidates"],
            metric_rows,
            ["left", "right", "right", "right", "right", "right", "right"],
        )
    )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def evaluate_agsto_v12_cached(args: argparse.Namespace) -> Dict[str, Any]:
    source_report_path = Path(args.transition_report).resolve()
    source_report = load_json(source_report_path)
    selected_datasets = set(parse_csv(args.datasets))
    config = AGSTOConfig(
        policy=str(args.policy),
        retrieval_top_k=max(int(args.retrieval_top_k), 5),
        evidence_set_size=max(int(args.evidence_set_size), 1),
        stable_anchor_k=max(int(args.stable_anchor_k), 0),
        proposal_candidate_depth=max(int(args.proposal_candidate_depth), 1),
        support_proposal_depth=max(int(args.support_proposal_depth), 1),
        candidate_limit=max(int(args.candidate_limit), 5),
        beam_size=max(int(args.beam_size), 1),
        set_search_policy=str(args.set_search_policy),
        max_endpoint_degree=max(int(args.max_endpoint_degree), 2),
    )
    datasets = []
    for dataset_payload in source_report.get("datasets", []) or []:
        dataset_name = str(dataset_payload.get("dataset", ""))
        if selected_datasets and dataset_name not in selected_datasets:
            continue
        datasets.append(
            evaluate_dataset(
                dataset_payload=dataset_payload,
                transition_report_dir=source_report_path.parent,
                config=config,
                max_queries=max(int(args.max_queries), 0),
            )
        )
    summary = {
        "method": "AG-STO-v12",
        "method_full_name": "Anchor-Guided Source-Title-OpenIE Evidence Set Retrieval",
        "source_transition_report_path": str(source_report_path),
        "proposal_source": "cached",
        "clean_package": "agsto_v12",
        "config": {
            "policy": config.policy,
            "completion_policy": config.completion_policy,
            "proposal_source": "cached",
            "max_queries": int(args.max_queries),
            "retrieval_top_k": int(config.retrieval_top_k),
            "evidence_set_size": int(config.evidence_set_size),
            "stable_anchor_k": int(config.stable_anchor_k),
            "proposal_candidate_depth": int(config.proposal_candidate_depth),
            "support_proposal_depth": int(config.support_proposal_depth),
            "candidate_limit": int(config.candidate_limit),
            "beam_size": int(config.beam_size),
            "set_search_policy": str(config.set_search_policy),
            "max_endpoint_degree": int(config.max_endpoint_degree),
        },
        "datasets": datasets,
    }
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, output_md)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AG-STO v12 from cached STO proposal reports.")
    parser.add_argument("--transition-report", default=DEFAULT_TRANSITION_REPORT)
    parser.add_argument("--datasets", default="2wikimultihopqa,musique,hotpotqa")
    parser.add_argument("--policy", default="graph", choices=["base", "graph"])
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--evidence-set-size", type=int, default=5)
    parser.add_argument("--stable-anchor-k", type=int, default=2)
    parser.add_argument("--proposal-candidate-depth", type=int, default=10)
    parser.add_argument("--support-proposal-depth", type=int, default=6)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--beam-size", type=int, default=12)
    parser.add_argument("--set-search-policy", default="beam", choices=["beam", "mct"])
    parser.add_argument("--max-endpoint-degree", type=int, default=30)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    summary = evaluate_agsto_v12_cached(args)
    compact = {
        dataset["dataset"]: {
            key: value
            for key, value in dataset.get("metrics", {}).items()
            if key.startswith("agsto_v12_")
        }
        for dataset in summary.get("datasets", []) or []
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
