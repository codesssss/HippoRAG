#!/usr/bin/env python3
"""Export AG-STO v12 cached-transition pools for external-pool QA evals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_agsto_pool import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    compute_title_recall,
    extract_title,
    get_gold_answers,
    get_gold_docs,
    passage_for_doc,
    score_pool_docs,
)
from src.agsto_v12 import AGSTOConfig, AGSTORetriever  # noqa: E402
from src.agsto_v12.evaluate_cached import (  # noqa: E402
    DEFAULT_TRANSITION_REPORT,
    load_json,
    resolve_path,
    retrieve_row,
)
from src.agsto_v12.ranking import unique_ranked  # noqa: E402


def sample_question(sample: Mapping[str, Any]) -> str:
    return str(sample.get("question") or "").strip()


def find_dataset_payload(report: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    for payload in report.get("datasets", []) or []:
        if str(payload.get("dataset")) == str(dataset):
            return payload
    raise ValueError(f"Dataset {dataset!r} not found in transition report.")


def cached_pool_doc_indices(
    result: Mapping[str, Any],
    *,
    pool_k: int,
    include_candidate_fill: bool,
) -> List[int]:
    selected_doc_indices = unique_ranked(
        ((result.get("selected_evidence_set", {}) or {}).get("doc_indices", []) or [])
    )
    doc_indices = unique_ranked(
        list(result.get("retrieved_doc_indices", []) or [])
        + selected_doc_indices
        + (list(result.get("candidate_doc_indices", []) or []) if include_candidate_fill else [])
    )
    return [int(doc_idx) for doc_idx in doc_indices[: max(int(pool_k), 1)]]


def build_cached_agsto_pool_records(
    *,
    dataset: str,
    samples: Sequence[Mapping[str, Any]],
    dataset_payload: Mapping[str, Any],
    transition_report_dir: Path,
    config: AGSTOConfig,
    max_queries: int,
    pool_k: int,
    include_candidate_fill: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
    openie_path = resolve_path(dataset_payload["openie_path"], base_dir=transition_report_dir)
    openie_docs = list(load_json(openie_path).get("docs", []) or [])
    retriever = AGSTORetriever.from_openie_docs(openie_docs, config=config)
    row_limit = max(int(config.retrieval_top_k), int(config.proposal_candidate_depth), 10)

    rows = list(dataset_payload.get("rows", []) or [])
    if int(max_queries) > 0:
        rows = rows[: int(max_queries)]
    if len(samples) < len(rows):
        raise ValueError(f"Dataset file has {len(samples)} samples, fewer than cached rows ({len(rows)}).")

    gold_docs = get_gold_docs(samples[: len(rows)], dataset)
    gold_answers = get_gold_answers(samples[: len(rows)])
    records: List[Dict[str, Any]] = []
    retrieved_doc_lists: List[List[str]] = []
    candidate_counts: List[int] = []

    for q_idx, row in enumerate(tqdm(rows, desc=f"AG-STO cached pool export ({dataset})")):
        sample = samples[q_idx]
        question = str(row.get("question") or row.get("query") or "").strip()
        if question != sample_question(sample):
            raise ValueError(
                f"Cached report question mismatch at index {q_idx}: "
                f"report={question!r}, sample={sample_question(sample)!r}"
            )

        result = retrieve_row(retriever=retriever, row=row, row_limit=row_limit)
        result["proposal_source"] = "cached"
        selected_doc_indices = unique_ranked(
            ((result.get("selected_evidence_set", {}) or {}).get("doc_indices", []) or [])
        )
        pool_doc_indices = [
            doc_idx
            for doc_idx in cached_pool_doc_indices(
                result,
                pool_k=int(pool_k),
                include_candidate_fill=bool(include_candidate_fill),
            )
            if passage_for_doc(openie_docs, int(doc_idx))
        ]
        pool_docs = [passage_for_doc(openie_docs, int(doc_idx)) for doc_idx in pool_doc_indices]
        pool_scores = score_pool_docs(pool_doc_indices, selected_doc_indices=selected_doc_indices)
        candidate_counts.append(int(result.get("candidate_doc_count", 0) or 0))
        retrieved_doc_lists.append(pool_docs)
        records.append(
            {
                "query_idx": int(q_idx),
                "question": question,
                "gold_answers": list(gold_answers[q_idx]),
                "gold_docs": list(gold_docs[q_idx]),
                "gold_titles": [extract_title(doc) for doc in gold_docs[q_idx]],
                "pool_k": int(pool_k),
                "pool_docs": pool_docs,
                "pool_titles": [extract_title(doc) for doc in pool_docs],
                "pool_doc_scores": pool_scores,
                "pool_doc_ids": [int(doc_idx) for doc_idx in pool_doc_indices],
                "agsto": {
                    "proposal_source": "cached",
                    "selected_doc_indices": selected_doc_indices,
                    "retrieved_doc_indices": list(result.get("retrieved_doc_indices", []) or [])[: int(pool_k)],
                    "candidate_doc_indices": list(result.get("candidate_doc_indices", []) or [])[: int(pool_k)],
                    "candidate_doc_count": int(result.get("candidate_doc_count", 0) or 0),
                    "selection_policy": result.get("selection_policy"),
                    "evidence_completion_policy": result.get("evidence_completion_policy"),
                    "graph_obligated_completion": result.get("graph_obligated_completion", {}),
                    "selected_evidence_set": result.get("selected_evidence_set", {}),
                },
            }
        )

    stats = {
        "openie_path": str(openie_path),
        "row_count": int(len(rows)),
        "mean_candidate_doc_count": round(
            float(sum(candidate_counts)) / float(max(len(candidate_counts), 1)),
            4,
        ),
        "min_candidate_doc_count": int(min(candidate_counts)) if candidate_counts else 0,
        "max_candidate_doc_count": int(max(candidate_counts)) if candidate_counts else 0,
    }
    recall = compute_title_recall(gold_docs=gold_docs, retrieved_docs=retrieved_doc_lists, k_values=[5, 20, 100])
    return records, recall, stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-report", type=Path, default=Path(DEFAULT_TRANSITION_REPORT))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--policy", choices=["base", "graph"], default="graph")
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--evidence_set_size", type=int, default=5)
    parser.add_argument("--stable_anchor_k", type=int, default=2)
    parser.add_argument("--proposal_candidate_depth", type=int, default=10)
    parser.add_argument("--support_proposal_depth", type=int, default=6)
    parser.add_argument("--candidate_limit", type=int, default=120)
    parser.add_argument("--beam_size", type=int, default=12)
    parser.add_argument("--set_search_policy", choices=["beam", "mct"], default="beam")
    parser.add_argument("--max_endpoint_degree", type=int, default=30)
    parser.add_argument("--include_candidate_fill", type=lambda value: str(value).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    transition_report_path = Path(args.transition_report).resolve()
    report = load_json(transition_report_path)
    dataset_payload = find_dataset_payload(report, str(args.dataset))
    samples_path = Path(args.data_root) / f"{args.dataset}.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    config = AGSTOConfig(
        policy=str(args.policy),
        retrieval_top_k=max(int(args.retrieval_top_k), int(args.evidence_set_size), 1),
        evidence_set_size=max(int(args.evidence_set_size), 1),
        stable_anchor_k=max(int(args.stable_anchor_k), 0),
        proposal_candidate_depth=max(int(args.proposal_candidate_depth), int(args.evidence_set_size), 1),
        support_proposal_depth=max(int(args.support_proposal_depth), int(args.evidence_set_size), 1),
        candidate_limit=max(int(args.candidate_limit), int(args.evidence_set_size), 1),
        beam_size=max(int(args.beam_size), 1),
        set_search_policy=str(args.set_search_policy),
        max_endpoint_degree=max(int(args.max_endpoint_degree), 2),
    )
    records, recall, stats = build_cached_agsto_pool_records(
        dataset=str(args.dataset),
        samples=samples,
        dataset_payload=dataset_payload,
        transition_report_dir=transition_report_path.parent,
        config=config,
        max_queries=max(int(args.max_queries), 0),
        pool_k=int(args.pool_k),
        include_candidate_fill=bool(args.include_candidate_fill),
    )
    output = {
        "dataset": str(args.dataset),
        "limit": int(len(records)),
        "pool_k": int(args.pool_k),
        "source": "agsto_v12_cached_pool_export",
        "proposal_source": "cached",
        "transition_report_path": str(transition_report_path),
        "openie_path": stats["openie_path"],
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
            "include_candidate_fill": bool(args.include_candidate_fill),
        },
        "retrieval": {
            "recomputed_title_recall": recall,
            "agsto_metrics": recall,
            "mean_candidate_doc_count": stats["mean_candidate_doc_count"],
            "min_candidate_doc_count": stats["min_candidate_doc_count"],
            "max_candidate_doc_count": stats["max_candidate_doc_count"],
        },
        "records": records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "dataset": str(args.dataset),
                "limit": int(len(records)),
                "pool_k": int(args.pool_k),
                "proposal_source": "cached",
                "recomputed_title_recall": recall,
                "mean_candidate_doc_count": stats["mean_candidate_doc_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
