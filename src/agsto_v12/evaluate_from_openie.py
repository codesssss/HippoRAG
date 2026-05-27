#!/usr/bin/env python3
"""Evaluate AG-STO v12 directly from OpenIE documents.

This runner is the standalone package path for:

    OpenIE docs -> STO graph -> native STO proposals -> AG-STO v12 retrieval

It does not depend on historical transition-component scripts, support_fusion,
local PPR variants, or QA calls. Cached-proposal parity reproduction remains in
``agsto_v12.evaluate_cached``; this module is for package-native proposal
generation from an OpenIE corpus.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .config import AGSTOConfig
from .proposals import build_native_sto_proposals
from .ranking import unique_ranked
from .retriever import AGSTORetriever


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_openie_docs(path: Path) -> List[Mapping[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, Mapping):
        docs = payload.get("docs", [])
    else:
        docs = payload
    if not isinstance(docs, list):
        raise ValueError(f"OpenIE input must be a list or a mapping with a 'docs' list: {path}")
    return [doc for doc in docs if isinstance(doc, Mapping)]


def load_query_rows(path: Path, *, dataset: str | None = None) -> List[Mapping[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Query input must be a list or mapping: {path}")

    if isinstance(payload.get("rows"), list):
        return [row for row in payload.get("rows", []) if isinstance(row, Mapping)]

    datasets = payload.get("datasets")
    if isinstance(datasets, list):
        for dataset_payload in datasets:
            if not isinstance(dataset_payload, Mapping):
                continue
            if dataset and str(dataset_payload.get("dataset", "")) != str(dataset):
                continue
            rows = dataset_payload.get("rows")
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, Mapping)]
        raise ValueError(f"No rows found for dataset={dataset!r} in {path}")

    raise ValueError(f"Query input must contain rows, datasets[].rows, or be a row list: {path}")


def row_doc_indices(row: Mapping[str, Any], *keys: str, limit: int = 20) -> List[int]:
    for key in keys:
        values = row.get(key)
        if values:
            doc_indices: List[int] = []
            seen = set()
            for value in values:
                try:
                    doc_idx = int(value)
                except (TypeError, ValueError):
                    continue
                if doc_idx < 0 or doc_idx in seen:
                    continue
                seen.add(doc_idx)
                doc_indices.append(doc_idx)
                if len(doc_indices) >= int(limit):
                    break
            return doc_indices
    return []


def query_text(row: Mapping[str, Any]) -> str:
    return str(row.get("question") or row.get("query") or "")


def gold_doc_indices(row: Mapping[str, Any]) -> List[int]:
    direct = row_doc_indices(
        row,
        "gold_doc_indices",
        "gold_docs",
        "support_doc_indices",
        "supporting_doc_indices",
        limit=100,
    )
    if direct:
        return direct

    paragraphs = row.get("paragraphs")
    if isinstance(paragraphs, list):
        gold: List[int] = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping):
                continue
            if not bool(paragraph.get("is_supporting", False)):
                continue
            if paragraph.get("idx") is None:
                continue
            try:
                gold.append(int(paragraph["idx"]))
            except (TypeError, ValueError):
                continue
        return unique_ranked(gold)
    return []


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
        separators.append("-" * (dash_count - 1) + ":" if align == "right" else "-" * dash_count)
    return "\n".join([fmt_row(headers), "| " + " | ".join(separators) + " |", *[fmt_row(row) for row in rows]])


def retrieve_row_native(
    *,
    retriever: AGSTORetriever,
    row: Mapping[str, Any],
    config: AGSTOConfig,
    support_realization: str,
    support_mct_iterations: int,
    support_mct_depth: int,
    support_mct_exploration: float,
    semantic_residual_weight: float,
) -> Dict[str, Any]:
    query = query_text(row)
    proposal_depth = max(int(config.proposal_candidate_depth), int(config.retrieval_top_k), 10)
    anchor_docs = row_doc_indices(row, "context_anchor_doc_indices", limit=proposal_depth)
    native_dense_docs = row_doc_indices(
        row,
        "native_dense_doc_indices_top10",
        "native_dense_doc_indices_top5",
        "reader_doc_indices_topk",
        "retrieved_doc_indices_top5",
        limit=proposal_depth,
    )
    proposals = build_native_sto_proposals(
        query=query,
        corpus_index=retriever.corpus_index,
        config=config,
        anchor_doc_indices=anchor_docs,
        native_dense_doc_indices=native_dense_docs,
        semantic_query_embedding=None,
        chunk_embedding_matrix=None,
        semantic_residual_weight=semantic_residual_weight,
        support_realization=support_realization,
        support_mct_iterations=support_mct_iterations,
        support_mct_depth=support_mct_depth,
        support_mct_exploration=support_mct_exploration,
    )
    result = retriever.retrieve(query=query, **proposals)
    result["proposal_source"] = "native_openie"
    result["support_realization"] = str(support_realization)
    return result


def evaluate_agsto_v12_from_openie(args: argparse.Namespace) -> Dict[str, Any]:
    openie_path = Path(args.openie_results).resolve()
    query_path = Path(args.queries_json).resolve()
    dataset_name = str(args.dataset or query_path.stem)
    openie_docs = load_openie_docs(openie_path)
    source_rows = load_query_rows(query_path, dataset=dataset_name)
    if int(args.max_queries) > 0:
        source_rows = source_rows[: int(args.max_queries)]

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
    support_realization = str(args.support_realization)
    if support_realization not in {"direct", "mct"}:
        raise ValueError("support_realization must be 'direct' or 'mct'")

    retriever = AGSTORetriever.from_openie_docs(openie_docs, config=config)
    rows: List[Dict[str, Any]] = []
    sums: Counter[str] = Counter()
    candidate_counts: List[int] = []
    selected_scores: List[float] = []
    gold_row_count = 0
    for row in source_rows:
        gold = gold_doc_indices(row)
        if gold:
            gold_row_count += 1
        result = retrieve_row_native(
            retriever=retriever,
            row=row,
            config=config,
            support_realization=support_realization,
            support_mct_iterations=max(int(args.support_mct_iterations), 1),
            support_mct_depth=max(int(args.support_mct_depth), 1),
            support_mct_exploration=max(float(args.support_mct_exploration), 0.0),
            semantic_residual_weight=max(float(args.semantic_residual_weight), 0.0),
        )
        retrieved = [int(doc_idx) for doc_idx in result.get("retrieved_doc_indices", []) or []]
        r5 = recall_at_k(gold, retrieved, 5)
        r10 = recall_at_k(gold, retrieved, 10)
        sums["r5"] += r5
        sums["r10"] += r10
        sums["all_gold_at5"] += 1.0 if all_gold_at_k(gold, retrieved, 5) else 0.0
        sums["all_gold_at10"] += 1.0 if all_gold_at_k(gold, retrieved, 10) else 0.0
        candidate_counts.append(int(result.get("candidate_doc_count", 0) or 0))
        selected_scores.append(float((result.get("selected_evidence_set", {}) or {}).get("score", 0.0) or 0.0))

        augmented_row = dict(row)
        augmented_row["gold_doc_indices"] = gold
        augmented_row["agsto_v12_doc_indices_top5"] = retrieved[:5]
        augmented_row["agsto_v12_doc_indices_top10"] = retrieved[:10]
        augmented_row["agsto_v12_recall_at5"] = round(float(r5), 6)
        augmented_row["agsto_v12_recall_at10"] = round(float(r10), 6)
        augmented_row["agsto_v12_all_gold_at5"] = bool(all_gold_at_k(gold, retrieved, 5))
        augmented_row["agsto_v12_all_gold_at10"] = bool(all_gold_at_k(gold, retrieved, 10))
        augmented_row["agsto_v12"] = result
        rows.append(augmented_row)

    denom = max(len(rows), 1)
    metrics = {
        "agsto_v12_r5": round(float(sums["r5"]) / float(denom), 6),
        "agsto_v12_r10": round(float(sums["r10"]) / float(denom), 6),
        "agsto_v12_all_gold_at5": round(float(sums["all_gold_at5"]) / float(denom), 6),
        "agsto_v12_all_gold_at10": round(float(sums["all_gold_at10"]) / float(denom), 6),
        "gold_row_count": int(gold_row_count),
        "mean_candidate_doc_count": round(sum(candidate_counts) / float(max(len(candidate_counts), 1)), 6),
        "mean_selected_evidence_score": round(sum(selected_scores) / float(max(len(selected_scores), 1)), 6),
    }
    dataset_payload = {
        "dataset": dataset_name,
        "openie_path": str(openie_path),
        "queries_path": str(query_path),
        "metrics": metrics,
        "rows": rows,
        "corpus_stats": {
            "doc_count": len(openie_docs),
            "unit_count": len(retriever.corpus_index["units"]),
            "endpoint_count": len(retriever.corpus_index["endpoint_to_units"]),
        },
    }
    summary = {
        "method": "AG-STO-v12",
        "method_full_name": "Anchor-Guided Source-Title-OpenIE Evidence Set Retrieval",
        "proposal_source": "native_openie",
        "clean_package": "agsto_v12",
        "config": {
            "policy": config.policy,
            "completion_policy": config.completion_policy,
            "proposal_source": "native_openie",
            "support_realization": support_realization,
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
            "semantic_residual_weight": float(args.semantic_residual_weight),
            "support_mct_iterations": int(args.support_mct_iterations),
            "support_mct_depth": int(args.support_mct_depth),
            "support_mct_exploration": float(args.support_mct_exploration),
        },
        "datasets": [dataset_payload],
    }
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, output_md)
    return summary


def write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# AG-STO v12 From OpenIE Evaluation",
        "",
        "Standalone native-proposal run: OpenIE docs -> STO graph -> AG-STO v12 retrieval.",
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
                int(metrics.get("gold_row_count", 0) or 0),
                f"{float(metrics.get('agsto_v12_r5', 0.0)):.4f}",
                f"{float(metrics.get('agsto_v12_r10', 0.0)):.4f}",
                f"{float(metrics.get('agsto_v12_all_gold_at5', 0.0)):.4f}",
                f"{float(metrics.get('agsto_v12_all_gold_at10', 0.0)):.4f}",
                f"{float(metrics.get('mean_candidate_doc_count', 0.0)):.1f}",
            ]
        )
    lines.append(
        markdown_table(
            ["dataset", "rows", "gold rows", "R@5", "R@10", "all-gold@5", "all-gold@10", "mean candidates"],
            metric_rows,
            ["left", "right", "right", "right", "right", "right", "right", "right"],
        )
    )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AG-STO v12 directly from OpenIE documents.")
    parser.add_argument("--openie-results", required=True)
    parser.add_argument("--queries-json", required=True)
    parser.add_argument("--dataset", default=None)
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
    parser.add_argument("--semantic-residual-weight", type=float, default=8.0)
    parser.add_argument("--support-realization", default="direct", choices=["direct", "mct"])
    parser.add_argument("--support-mct-iterations", type=int, default=64)
    parser.add_argument("--support-mct-depth", type=int, default=2)
    parser.add_argument("--support-mct-exploration", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    summary = evaluate_agsto_v12_from_openie(args)
    compact = {
        dataset["dataset"]: {
            key: value
            for key, value in dataset.get("metrics", {}).items()
            if key.startswith("agsto_v12_") or key == "gold_row_count"
        }
        for dataset in summary.get("datasets", []) or []
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
