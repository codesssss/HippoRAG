#!/usr/bin/env python3
"""Lightweight causal router probe across all datasets.

Only runs the SemanticIntentRouter (embedding similarity to anchor bank) on
every query in each dataset.  No indexing, no retrieval, no QA.

Outputs per dataset:
  - router_label_counts
  - best_causal_score percentiles (p50/p75/p90/p95/max)
  - score bucket ratios (>0.3, >0.5, >0.7)
  - top-20 highest causal-score queries for manual inspection

Usage:
    python scripts/probe_causal_router.py \
        --embedding_name "VLLM//mnt/nvme/Qwen3-Embedding-8B" \
        --embedding_base_url http://localhost:8018/v1/embeddings \
        --output_dir outputs/router_probe
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.hipporag.embedding_model import _get_embedding_model_class, EmbeddingConfig
from src.hipporag.causal_v2 import SemanticIntentRouter
from src.hipporag.prompts.linking import get_query_instruction


# ---------------------------------------------------------------------------
# Minimal config stub – only fields the router reads
# ---------------------------------------------------------------------------
@dataclass
class _RouterConfig:
    causal_router_anchor_path: str = None
    causal_router_causal_threshold: float = 0.62
    causal_router_standard_threshold: float = 0.62
    causal_router_margin_threshold: float = 0.02


# ---------------------------------------------------------------------------
# Dataset loading – extract question strings from each format
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "reproduce", "dataset")


def _load_questions(dataset_name: str) -> list[str]:
    path = os.path.join(DATASET_DIR, f"{dataset_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else list(data.values())
    questions: list[str] = []
    for item in items:
        q = item.get("question") or item.get("input") or ""
        if isinstance(q, str) and q.strip():
            questions.append(q.strip())
    return questions


def _discover_datasets() -> list[str]:
    names = []
    for fn in sorted(os.listdir(DATASET_DIR)):
        if not fn.endswith(".json"):
            continue
        if fn.endswith("_corpus.json") or fn.startswith("sample"):
            continue
        name = fn[: -len(".json")]
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------
def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Probe causal router scores across datasets.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names. Default: auto-discover all.")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--anchor_path", type=str, default=None,
                        help="Custom anchor bank JSON path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max queries per dataset (0 = all).")
    parser.add_argument("--top_k_display", type=int, default=20,
                        help="Number of top causal-score queries to include per dataset.")
    parser.add_argument("--output_dir", type=str, default="outputs/router_probe")
    args = parser.parse_args()

    # Discover datasets
    if args.datasets:
        dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        dataset_names = _discover_datasets()
    print(f"Datasets to probe: {dataset_names}")

    # Init embedding model
    emb_config = EmbeddingConfig()
    emb_config.embedding_model_name = args.embedding_name
    emb_config.embedding_base_url = args.embedding_base_url
    EmbClass = _get_embedding_model_class(args.embedding_name)
    embedding_model = EmbClass(
        global_config=emb_config,
        embedding_model_name=args.embedding_name,
    )
    print(f"Embedding model: {args.embedding_name}")

    # Init router
    router_config = _RouterConfig(causal_router_anchor_path=args.anchor_path)
    router = SemanticIntentRouter(embedding_model, router_config)
    print(f"Anchor bank labels: {list(router.anchor_bank.keys())}, "
          f"total anchors: {len(router.anchor_texts)}")

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows: list[dict] = []

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            questions = _load_questions(ds_name)
        except Exception as exc:
            print(f"  SKIP: {exc}")
            continue

        if not questions:
            print(f"  SKIP: no questions found")
            continue

        if args.limit > 0:
            questions = questions[: args.limit]
        print(f"  Loaded {len(questions)} queries")

        # Batch-embed all queries
        query_instruction = get_query_instruction("query_to_passage")
        all_embeddings = []
        for i in tqdm(range(0, len(questions), args.batch_size), desc=f"  Embedding {ds_name}"):
            batch = questions[i : i + args.batch_size]
            embs = embedding_model.batch_encode(batch, instruction=query_instruction, norm=True)
            all_embeddings.extend(embs)
        all_embeddings = np.array(all_embeddings, dtype=float)

        # Route each query
        results: list[dict] = []
        for idx in range(len(questions)):
            route_info = router.route(all_embeddings[idx])
            results.append({
                "question": questions[idx],
                "label": route_info["label"],
                "is_causal": route_info["is_causal"],
                "best_causal_label": route_info["best_causal_label"],
                "best_causal_score": route_info["best_causal_score"],
                "standard_score": route_info["standard_score"],
                "margin": route_info["margin"],
                "scores_by_label": route_info["scores_by_label"],
            })

        # Aggregate
        label_counts = dict(Counter(r["label"] for r in results))
        causal_scores = [r["best_causal_score"] for r in results]
        standard_scores = [r["standard_score"] for r in results]
        margins = [r["margin"] for r in results]

        n = len(results)
        score_buckets = {
            "gt_0.3": sum(1 for s in causal_scores if s > 0.3) / n,
            "gt_0.4": sum(1 for s in causal_scores if s > 0.4) / n,
            "gt_0.5": sum(1 for s in causal_scores if s > 0.5) / n,
            "gt_0.6": sum(1 for s in causal_scores if s > 0.6) / n,
            "gt_0.7": sum(1 for s in causal_scores if s > 0.7) / n,
        }

        margin_buckets = {
            "margin_gt_0.0": sum(1 for m in margins if m > 0.0) / n,
            "margin_gt_0.02": sum(1 for m in margins if m > 0.02) / n,
            "margin_gt_0.05": sum(1 for m in margins if m > 0.05) / n,
        }

        # Top queries by causal score
        sorted_results = sorted(results, key=lambda r: r["best_causal_score"], reverse=True)
        top_queries = [
            {
                "question": r["question"],
                "best_causal_score": round(r["best_causal_score"], 4),
                "best_causal_label": r["best_causal_label"],
                "standard_score": round(r["standard_score"], 4),
                "margin": round(r["margin"], 4),
            }
            for r in sorted_results[: args.top_k_display]
        ]

        ds_report = {
            "dataset": ds_name,
            "num_queries": n,
            "router_label_counts": label_counts,
            "best_causal_score_percentiles": _percentiles(causal_scores),
            "standard_score_percentiles": _percentiles(standard_scores),
            "margin_percentiles": _percentiles(margins),
            "causal_score_buckets": score_buckets,
            "margin_buckets": margin_buckets,
            "top_causal_queries": top_queries,
        }

        # Print summary
        pct = ds_report["best_causal_score_percentiles"]
        print(f"  Labels: {label_counts}")
        print(f"  Causal score: p50={pct['p50']:.3f}  p75={pct['p75']:.3f}  "
              f"p90={pct['p90']:.3f}  p95={pct['p95']:.3f}  max={pct['max']:.3f}")
        print(f"  Buckets: {', '.join(f'{k}={v:.1%}' for k, v in score_buckets.items())}")
        print(f"  Margin:  {', '.join(f'{k}={v:.1%}' for k, v in margin_buckets.items())}")
        print(f"  Top-3 causal queries:")
        for tq in top_queries[:3]:
            print(f"    [{tq['best_causal_score']:.3f} {tq['best_causal_label']}] {tq['question'][:100]}")

        # Save per-dataset report
        ds_path = os.path.join(args.output_dir, f"probe_{ds_name}.json")
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump(ds_report, f, indent=2, ensure_ascii=False)

        summary_rows.append({
            "dataset": ds_name,
            "num_queries": n,
            "router_label_counts": label_counts,
            "causal_p50": round(pct["p50"], 4),
            "causal_p75": round(pct["p75"], 4),
            "causal_p90": round(pct["p90"], 4),
            "causal_p95": round(pct["p95"], 4),
            "causal_max": round(pct["max"], 4),
            "gt_0.5_rate": round(score_buckets["gt_0.5"], 4),
            "gt_0.6_rate": round(score_buckets["gt_0.6"], 4),
            "margin_gt_0.02_rate": round(margin_buckets["margin_gt_0.02"], 4),
        })

    # Save overall summary
    summary_path = os.path.join(args.output_dir, "probe_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"\n\nSummary written to {summary_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Dataset':<30} {'N':>5} {'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6} "
          f"{'max':>6} {'>0.5':>6} {'>0.6':>6} {'m>0.02':>7}")
    print(f"{'-'*100}")
    for row in summary_rows:
        print(f"{row['dataset']:<30} {row['num_queries']:>5} "
              f"{row['causal_p50']:>6.3f} {row['causal_p75']:>6.3f} "
              f"{row['causal_p90']:>6.3f} {row['causal_p95']:>6.3f} "
              f"{row['causal_max']:>6.3f} {row['gt_0.5_rate']:>6.1%} "
              f"{row['gt_0.6_rate']:>6.1%} {row['margin_gt_0.02_rate']:>7.1%}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
