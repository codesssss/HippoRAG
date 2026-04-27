#!/usr/bin/env python3
"""Build fold-safe pairwise CEE-v2 training data."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import random
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dpathrag_analyze_hard_negatives import load_jsonl, row_qid  # noqa: E402
from dpathrag_cee_edit_policy import kfold_ranges, load_embedding_features  # noqa: E402
from dpathrag_cee_pairwise_common import select_pair_rows, split_train_calib, write_jsonl  # noqa: E402
from src.dpathrag.io import write_json  # noqa: E402


def summarize_pairs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pair_types = Counter(str(row.get("pair_type") or "") for row in rows)
    same_remove = sum(1 for row in rows if row.get("same_remove"))
    neg_types = Counter(str(row.get("negative_edit", {}).get("hard_negative_type") or "graded") for row in rows)
    qids = {str(row.get("qid")) for row in rows}
    return {
        "pairs": len(rows),
        "queries": len(qids),
        "pair_types": dict(sorted(pair_types.items())),
        "negative_types": dict(sorted(neg_types.items())),
        "same_remove_pairs": same_remove,
        "same_remove_rate": round(float(same_remove) / max(1.0, float(len(rows))), 6),
    }


def build_folds(
    rows: Sequence[dict[str, Any]],
    *,
    output_dir: str | Path,
    embedding_features: dict[str, Any],
    embedding_feature_names: Sequence[str],
    folds: int,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    max_pairs_per_query: int,
    seed: int,
    calib_fraction: float,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = kfold_ranges(len(rows), folds)
    fold_reports = []
    for fold, (start, end) in enumerate(ranges):
        eval_rows = list(rows[start:end])
        eval_qids = {row_qid(row) for row in eval_rows}
        train_source = [row for row in rows if row_qid(row) not in eval_qids]
        rng = random.Random(int(seed) + fold)
        pairs: list[dict[str, Any]] = []
        for record in train_source:
            for pair in select_pair_rows(
                record,
                top_k=top_k,
                max_candidates=max_candidates,
                candidate_pool_size=candidate_pool_size,
                embedding_by_qid=embedding_features,
                embedding_feature_names=embedding_feature_names,
                rng=rng,
                max_pairs_per_query=max_pairs_per_query,
            ):
                pair["fold"] = fold
                pairs.append(pair)
        train_pairs, calib_pairs = split_train_calib(pairs, seed=int(seed) + 1000 + fold, calib_fraction=calib_fraction)
        eval_meta = [{"qid": row_qid(row), "row_index": idx} for idx, row in enumerate(eval_rows, start=start)]
        write_jsonl(train_pairs, output_dir / f"fold_{fold}.train.jsonl")
        write_jsonl(calib_pairs, output_dir / f"fold_{fold}.calib.jsonl")
        write_jsonl(eval_meta, output_dir / f"fold_{fold}.eval_meta.jsonl")
        fold_reports.append(
            {
                "fold": fold,
                "train_source_rows": len(train_source),
                "eval_rows": len(eval_rows),
                "train_pairs": summarize_pairs(train_pairs),
                "calib_pairs": summarize_pairs(calib_pairs),
                "all_pairs": summarize_pairs(pairs),
            }
        )
    all_train = []
    all_calib = []
    for fold in range(int(folds)):
        from dpathrag_cee_pairwise_common import read_jsonl  # local import keeps script standalone for tests

        all_train.extend(read_jsonl(output_dir / f"fold_{fold}.train.jsonl"))
        all_calib.extend(read_jsonl(output_dir / f"fold_{fold}.calib.jsonl"))
    return {
        "rows": len(rows),
        "folds": folds,
        "top_k": top_k,
        "candidate_pool_size": candidate_pool_size,
        "output_dir": str(output_dir),
        "train_pairs_total": summarize_pairs(all_train),
        "calib_pairs_total": summarize_pairs(all_calib),
        "fold_reports": fold_reports,
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    total = report["train_pairs_total"]
    calib = report["calib_pairs_total"]
    lines = [
        "# CEE Pairwise Dataset Stats",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Folds: `{report['folds']}`",
        f"- Candidate pool: `top{report['candidate_pool_size']}`",
        f"- Train pairs: `{total['pairs']}` across `{total['queries']}` queries",
        f"- Calibration pairs: `{calib['pairs']}` across `{calib['queries']}` queries",
        f"- Train same-remove rate: `{total['same_remove_rate']}`",
        "",
        "## Pair Types",
        "",
        "| Split | Lexical | Mixed | Graded |",
        "|---|---:|---:|---:|",
        f"| Train | {total['pair_types'].get('lexical', 0)} | {total['pair_types'].get('mixed', 0)} | {total['pair_types'].get('graded', 0)} |",
        f"| Calib | {calib['pair_types'].get('lexical', 0)} | {calib['pair_types'].get('mixed', 0)} | {calib['pair_types'].get('graded', 0)} |",
        "",
        "## Fold Summary",
        "",
        "| Fold | Train Pairs | Calib Pairs | Eval Rows | Same-Remove Rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold in report["fold_reports"]:
        lines.append(
            f"| {fold['fold']} | {fold['train_pairs']['pairs']} | {fold['calib_pairs']['pairs']} | {fold['eval_rows']} | {fold['all_pairs']['same_remove_rate']} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--embedding_npz", default="data/dpathrag/cache/selector_embedding_proprag_local1000_rp64.npz")
    parser.add_argument("--output_dir", default="data/dpathrag/cee_pairwise_train_folds")
    parser.add_argument("--output_json", default="reports/dpathrag/cee_pairwise_dataset_stats.json")
    parser.add_argument("--output_md", default="reports/dpathrag/cee_pairwise_dataset_stats.md")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--max_pairs_per_query", type=int, default=6)
    parser.add_argument("--calib_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    embedding_features, embedding_feature_names = load_embedding_features(args.embedding_npz, limit=int(args.limit))
    report = build_folds(
        rows,
        output_dir=args.output_dir,
        embedding_features=embedding_features,
        embedding_feature_names=embedding_feature_names,
        folds=int(args.folds),
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        candidate_pool_size=int(args.candidate_pool_size),
        max_pairs_per_query=int(args.max_pairs_per_query),
        seed=int(args.seed),
        calib_fraction=float(args.calib_fraction),
    )
    write_json(report, args.output_json)
    write_markdown(report, args.output_md)


if __name__ == "__main__":
    main()
