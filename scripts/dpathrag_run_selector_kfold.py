#!/usr/bin/env python3
"""Run non-leaky k-fold D-PathRAG selector warm-start experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import write_json, write_jsonl
from src.dpathrag.selector_data import summarize_selector_metrics


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= int(limit):
                    break
    return rows


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("query_idx") or "")


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_rows = [dict(row.get("model") or {}) for row in rows]
    rank_rows = [dict(row.get("rank_topk") or {}) for row in rows]
    return {
        "model": summarize_selector_metrics(model_rows),
        "rank_topk": summarize_selector_metrics(rank_rows),
    }


def write_markdown_summary(summary: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# D-PathRAG Selector K-Fold Summary",
        "",
        f"- Pool: `{summary['pool_name']}`",
        f"- Cache: `{summary['cache_jsonl']}`",
        f"- Embeddings: `{summary.get('embedding_npz') or ''}`",
        f"- Rows: `{summary['merged']['rows']}`",
        f"- Fold size: `{summary['config']['fold_size']}`",
        f"- Num folds: `{summary['config']['num_folds']}`",
        f"- Train/eval qid overlap: `{summary['merged']['train_eval_qid_overlap']}`",
        "",
        "## Aggregate Selector Metrics",
        "",
        "| Variant | Support Recall | Support Complete | Selected Gold | Bridge Entity Recall | Selection Overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("rank_topk", "model"):
        metrics = summary["merged"]["metrics"][key]
        lines.append(
            "| "
            + key
            + " | "
            + f"{metrics.get('support_recall', 0.0):.4f} | "
            + f"{metrics.get('support_complete', 0.0):.4f} | "
            + f"{metrics.get('selected_gold_count', 0.0):.4f} | "
            + f"{metrics.get('bridge_entity_recall', 0.0):.4f} | "
            + f"{metrics.get('selection_overlap', 0.0):.4f} |"
        )
    delta = summary["merged"]["deltas"]
    lines.extend(
        [
            "",
            "## Delta",
            "",
            f"- Support-complete delta: `{delta['support_complete']:+.4f}`",
            f"- Support-recall delta: `{delta['support_recall']:+.4f}`",
            f"- Bridge-entity-recall delta: `{delta['bridge_entity_recall']:+.4f}`",
            "",
            "## Folds",
            "",
            "| Fold | Eval Start | Train Rows | Eval Rows | Best Epoch | Rank Complete | Model Complete | Delta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in summary["folds"]:
        after = fold["eval_after_train"]
        rank_complete = float(after["rank_topk"].get("support_complete", 0.0))
        model_complete = float(after["model"].get("support_complete", 0.0))
        lines.append(
            f"| {fold['fold']} | {fold['eval_start']} | {fold['train_rows']} | {fold['eval_rows']} | "
            f"{fold['best_epoch']} | {rank_complete:.4f} | {model_complete:.4f} | {model_complete - rank_complete:+.4f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_train_command(args: argparse.Namespace, *, fold: int, eval_start: int, report_json: Path, predictions_jsonl: Path) -> list[str]:
    script = Path(__file__).resolve().with_name("dpathrag_train_selector_warmstart.py")
    command = [
        sys.executable,
        str(script),
        "--cache_jsonl",
        str(args.cache_jsonl),
        "--output_dir",
        str(Path(args.model_dir) / f"fold_{fold}"),
        "--report_json",
        str(report_json),
        "--predictions_jsonl",
        str(predictions_jsonl),
        "--limit",
        str(args.limit),
        "--train_size",
        str(args.train_size),
        "--eval_size",
        str(args.fold_size),
        "--eval_start",
        str(eval_start),
        "--train_from_complement",
        "--max_candidates",
        str(args.max_candidates),
        "--path_len",
        str(args.path_len),
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_layers",
        str(args.num_layers),
        "--num_heads",
        str(args.num_heads),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--seed",
        str(args.seed + fold),
        "--device",
        str(args.device),
        "--eval_each_epoch",
        "--best_metric",
        str(args.best_metric),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--load_best_at_end",
    ]
    if str(args.embedding_npz):
        command.extend(["--embedding_npz", str(args.embedding_npz)])
    if bool(args.shuffle_targets):
        command.append("--shuffle_targets")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--embedding_npz", default="")
    parser.add_argument("--pool_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--fold_size", type=int, default=200)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=0, help="0 means use the full complement for each fold.")
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--path_len", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--shuffle_targets", action="store_true")
    parser.add_argument("--best_metric", choices=["support_complete", "support_recall", "bridge_entity_recall"], default="support_complete")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--load_existing", action="store_true", help="Reuse existing fold reports/predictions instead of rerunning them.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    expected_rows = int(args.fold_size) * int(args.num_folds)
    if len(rows) < expected_rows:
        raise ValueError(f"Need at least {expected_rows} rows for k-fold, found {len(rows)}")

    fold_summaries: list[dict[str, Any]] = []
    merged_predictions: list[dict[str, Any]] = []
    for fold in range(int(args.num_folds)):
        eval_start = fold * int(args.fold_size)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        report_json = fold_dir / "selector_report.json"
        predictions_jsonl = fold_dir / "selector_predictions.jsonl"
        command = build_train_command(args, fold=fold, eval_start=eval_start, report_json=report_json, predictions_jsonl=predictions_jsonl)
        log_path = fold_dir / "train.log"
        if not (bool(args.load_existing) and report_json.exists() and predictions_jsonl.exists()):
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write(" ".join(command) + "\n\n")
                log_handle.flush()
                subprocess.run(command, check=True, stdout=log_handle, stderr=subprocess.STDOUT, cwd=_PROJECT_ROOT)
        report = read_json(report_json)
        predictions = load_jsonl(predictions_jsonl)
        if len(predictions) != int(args.fold_size):
            raise ValueError(f"Fold {fold} predictions mismatch: expected {args.fold_size}, got {len(predictions)}")
        split = dict(report.get("split") or {})
        fold_summary = {
            "fold": fold,
            "eval_start": eval_start,
            "report_json": str(report_json),
            "predictions_jsonl": str(predictions_jsonl),
            "train_rows": int(report.get("train_rows") or 0),
            "eval_rows": int(report.get("eval_rows") or 0),
            "train_eval_qid_overlap": int(split.get("train_eval_qid_overlap") or 0),
            "best_epoch": int((report.get("training") or {}).get("best_epoch") or -1),
            "epochs_run": int((report.get("training") or {}).get("epochs_run") or 0),
            "eval_after_train": report.get("eval_after_train") or {},
        }
        fold_summaries.append(fold_summary)
        merged_predictions.extend(predictions)

    cache_order = {row_qid(row): idx for idx, row in enumerate(rows[:expected_rows])}
    seen: set[str] = set()
    for prediction in merged_predictions:
        qid = row_qid(prediction)
        if qid in seen:
            raise ValueError(f"Duplicate qid in merged predictions: {qid}")
        seen.add(qid)
    missing = [row_qid(row) for row in rows[:expected_rows] if row_qid(row) not in seen]
    if missing:
        raise ValueError(f"Missing {len(missing)} qids from merged predictions; first={missing[0]}")
    merged_predictions = sorted(merged_predictions, key=lambda row: cache_order[row_qid(row)])
    merged_path = output_dir / f"selector_kfold_{args.pool_name}_embed_local1000.predictions.jsonl"
    write_jsonl(merged_predictions, merged_path)

    aggregate = summarize_prediction_rows(merged_predictions)
    deltas = {
        key: round(float(aggregate["model"].get(key, 0.0)) - float(aggregate["rank_topk"].get(key, 0.0)), 4)
        for key in ("support_recall", "support_complete", "bridge_entity_recall", "selected_gold_count")
    }
    summary = {
        "cache_jsonl": str(args.cache_jsonl),
        "embedding_npz": str(args.embedding_npz),
        "pool_name": str(args.pool_name),
        "merged_predictions_jsonl": str(merged_path),
        "config": {
            "limit": int(args.limit),
            "fold_size": int(args.fold_size),
            "num_folds": int(args.num_folds),
            "train_size": int(args.train_size),
            "max_candidates": int(args.max_candidates),
            "path_len": int(args.path_len),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "best_metric": str(args.best_metric),
            "early_stop_patience": int(args.early_stop_patience),
            "seed": int(args.seed),
            "device": str(args.device),
        },
        "folds": fold_summaries,
        "merged": {
            "rows": len(merged_predictions),
            "unique_qids": len(seen),
            "train_eval_qid_overlap": sum(int(fold["train_eval_qid_overlap"]) for fold in fold_summaries),
            "metrics": aggregate,
            "deltas": deltas,
        },
    }
    summary_json = output_dir / f"selector_kfold_{args.pool_name}_embed_local1000_summary.json"
    summary_md = output_dir / f"selector_kfold_{args.pool_name}_embed_local1000_summary.md"
    write_json(summary, summary_json)
    write_markdown_summary(summary, summary_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
