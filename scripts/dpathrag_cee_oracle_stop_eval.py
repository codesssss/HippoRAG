#!/usr/bin/env python3
"""Evaluate oracle STOP rules with the current shallow CEE edit scorer."""

from __future__ import annotations

import argparse
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

from dpathrag_analyze_hard_negatives import is_gold, load_jsonl, row_qid  # noqa: E402
from dpathrag_analyze_hard_negatives_v2 import oracle_edit_sequence  # noqa: E402
from dpathrag_cee_edit_policy import (  # noqa: E402
    LinearEditPolicy,
    apply_action,
    build_training_examples,
    cee_pilot_gates,
    choose_action,
    kfold_ranges,
    load_embedding_features,
    make_example,
    summarize_rank_and_oracle,
    summarize_selector_predictions,
)
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.selector_data import selection_overlap, summarize_selector_metrics, support_metrics_for_indices  # noqa: E402


STOP_RULES = ("none", "rank_complete", "objective_oracle")


def should_force_stop(
    record: dict[str, Any],
    *,
    stop_rule: str,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    example: Any,
) -> bool:
    if stop_rule == "none":
        return False
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    rank_indices = list(range(min(int(top_k), len(candidates))))
    if stop_rule == "rank_complete":
        rank_metrics = support_metrics_for_indices(example, rank_indices)
        return float(rank_metrics.get("support_complete") or 0.0) >= 1.0
    if stop_rule == "objective_oracle":
        oracle = oracle_edit_sequence(
            record,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            max_candidates=max_candidates,
            steps=1,
            example=example,
        )
        return not bool(oracle.get("edits"))
    raise ValueError(f"Unknown stop_rule={stop_rule}")


def evaluate_policy_with_stop_rule(
    policy: LinearEditPolicy,
    records: Sequence[dict[str, Any]],
    *,
    stop_rule: str,
    max_candidates: int,
    candidate_pool_size: int,
    top_k: int,
    embedding_features: dict[str, tuple[Any, Any]],
    min_edit_margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, float]] = []
    predictions: list[dict[str, Any]] = []
    totals = {"added_gold": 0, "added_non_gold": 0, "removed_gold": 0, "removed_non_gold": 0, "queries_with_edit": 0}
    for record in records:
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features=embedding_features)
        rank_indices = list(range(min(int(top_k), len(candidates))))
        indices = list(rank_indices)
        edits: list[dict[str, Any]] = []
        if not should_force_stop(
            record,
            stop_rule=stop_rule,
            top_k=top_k,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            example=example,
        ):
            action, score, stop_score = choose_action(
                policy,
                example,
                indices,
                candidate_count=len(candidates),
                candidate_pool_size=candidate_pool_size,
                min_edit_margin=min_edit_margin,
            )
            if action["type"] != "stop":
                remove_index = int(action["remove_index"])
                add_index = int(action["add_index"])
                edits.append({"remove_index": remove_index, "add_index": add_index, "score": score, "stop_score": stop_score})
                totals["added_gold"] += int(is_gold(candidates[add_index]))
                totals["added_non_gold"] += int(not is_gold(candidates[add_index]))
                totals["removed_gold"] += int(is_gold(candidates[remove_index]))
                totals["removed_non_gold"] += int(not is_gold(candidates[remove_index]))
                indices = apply_action(indices, action)
        if edits:
            totals["queries_with_edit"] += 1
        metrics = support_metrics_for_indices(example, indices)
        metrics["selection_overlap"] = selection_overlap(indices, rank_indices)
        rows.append(metrics)
        predictions.append(
            {
                "qid": row_qid(record),
                "selected_indices": indices,
                "selected_titles": [example.candidate_titles[index] for index in indices],
                "rank_topk_titles": [example.candidate_titles[index] for index in rank_indices],
                "edits": edits,
                "metrics": metrics,
            }
        )
    summary = summarize_selector_metrics(rows)
    denom = float(max(1, len(records)))
    return {
        **summary,
        **totals,
        "stop_rate": round(1.0 - totals["queries_with_edit"] / denom, 4),
        "non_gold_per_gold": round(float(totals["added_non_gold"]) / max(1.0, float(totals["added_gold"])), 4),
    }, predictions


def aggregate_predictions(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], *, max_candidates: int) -> dict[str, Any]:
    by_qid = {row_qid(row): row for row in records}
    metrics = summarize_selector_metrics([row["metrics"] for row in predictions])
    totals = {"added_gold": 0, "added_non_gold": 0, "removed_gold": 0, "removed_non_gold": 0, "queries_with_edit": 0}
    for row in predictions:
        record = by_qid[str(row["qid"])]
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        if row.get("edits"):
            totals["queries_with_edit"] += 1
        for edit in row.get("edits") or []:
            add_index = int(edit["add_index"])
            remove_index = int(edit["remove_index"])
            totals["added_gold"] += int(is_gold(candidates[add_index]))
            totals["added_non_gold"] += int(not is_gold(candidates[add_index]))
            totals["removed_gold"] += int(is_gold(candidates[remove_index]))
            totals["removed_non_gold"] += int(not is_gold(candidates[remove_index]))
    return {
        **metrics,
        **totals,
        "stop_rate": round(1.0 - totals["queries_with_edit"] / max(1.0, float(len(predictions))), 4),
        "non_gold_per_gold": round(float(totals["added_non_gold"]) / max(1.0, float(totals["added_gold"])), 4),
    }


def run_eval(
    rows: Sequence[dict[str, Any]],
    *,
    selector_predictions: Sequence[dict[str, Any]],
    embedding_features: dict[str, tuple[Any, Any]],
    margins: Sequence[float],
    folds: int,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    negatives_per_query: int,
    seed: int,
    verbose: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_predictions: list[dict[str, Any]] = []
    variants: dict[str, list[dict[str, Any]]] = {}
    ranges = kfold_ranges(len(rows), folds)
    fold_reports: list[dict[str, Any]] = []
    for fold, (start, end) in enumerate(ranges):
        if verbose:
            print(f"[oracle-stop] fold {fold}: train scorer", file=sys.stderr, flush=True)
        eval_rows = list(rows[start:end])
        eval_qids = {row_qid(row) for row in eval_rows}
        train_rows = [row for row in rows if row_qid(row) not in eval_qids]
        train_features, train_labels, train_summary = build_training_examples(
            train_rows,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            top_k=top_k,
            embedding_features=embedding_features,
            negatives_per_query=negatives_per_query,
            seed=seed + fold,
        )
        policy = LinearEditPolicy(epochs=1, learning_rate=0.0, batch_size=256, seed=seed + fold)
        fit_summary = policy.fit(train_features, train_labels)
        fold_report = {"fold": fold, "train_examples": train_summary, "fit": fit_summary, "variants": {}}
        for margin in margins:
            for stop_rule in STOP_RULES:
                name = f"{stop_rule}_margin{int(margin) if float(margin).is_integer() else margin}"
                summary, preds = evaluate_policy_with_stop_rule(
                    policy,
                    eval_rows,
                    stop_rule=stop_rule,
                    max_candidates=max_candidates,
                    candidate_pool_size=candidate_pool_size,
                    top_k=top_k,
                    embedding_features=embedding_features,
                    min_edit_margin=float(margin),
                )
                fold_report["variants"][name] = summary
                for row in preds:
                    row["variant"] = name
                    row["fold"] = fold
                variants.setdefault(name, []).extend(preds)
                all_predictions.extend(preds)
        fold_reports.append(fold_report)
    oracle_and_rank = summarize_rank_and_oracle(
        rows,
        top_k=top_k,
        max_candidates=max_candidates,
        candidate_pool_size=candidate_pool_size,
    )
    report: dict[str, Any] = {
        "rows": len(rows),
        "folds": int(folds),
        "top_k": int(top_k),
        "max_candidates": int(max_candidates),
        "candidate_pool_size": int(candidate_pool_size),
        "rank_topk": oracle_and_rank["rank_topk"],
        "selector_v1": summarize_selector_predictions(rows, selector_predictions, top_k=top_k, max_candidates=max_candidates),
        "oracle_edit1": oracle_and_rank["oracle_edit1"],
        "variants": {},
        "fold_reports": fold_reports,
    }
    for name, preds in sorted(variants.items()):
        report["variants"][name] = aggregate_predictions(rows, preds, max_candidates=max_candidates)
    for name, summary in report["variants"].items():
        gate_payload = {"rank_topk": report["rank_topk"], "selector_v1": report["selector_v1"], "learned_edit1": summary, "learned_edit2": summary}
        summary["gates"] = cee_pilot_gates(gate_payload)["learned_edit1"]
    return report, all_predictions


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# CEE Oracle STOP Evaluation",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Folds: `{report['folds']}`",
        f"- Candidate pool size: `{report['candidate_pool_size']}`",
        "",
        "| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Edits | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base_variants = {"rank_topk": report["rank_topk"], "selector_v1": report["selector_v1"], "oracle_edit1": report["oracle_edit1"]}
    for name, item in {**base_variants, **report["variants"]}.items():
        gates = item.get("gates", {})
        gate_text = f"{gates.get('passed_count', '')}/3" if gates else ""
        lines.append(
            f"| {name} | {item.get('support_recall', 0.0):.4f} | {item.get('support_complete', 0.0):.4f} | "
            f"{item.get('selected_gold_count', 0.0):.4f} | {item.get('bridge_entity_recall', 0.0):.4f} | "
            f"{item.get('selection_overlap', 0.0):.4f} | {item.get('queries_with_edit', '')} | "
            f"{item.get('stop_rate', '')} | {item.get('added_gold', '')} | {item.get('added_non_gold', '')} | "
            f"{item.get('non_gold_per_gold', '')} | {gate_text} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--selector_predictions_jsonl", required=True)
    parser.add_argument("--embedding_npz", default="")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--negatives_per_query", type=int, default=16)
    parser.add_argument("--margin", action="append", type=float, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--predictions_jsonl", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    selector_predictions = load_jsonl(args.selector_predictions_jsonl)
    embedding_features, _ = load_embedding_features(args.embedding_npz, limit=int(args.limit))
    report, predictions = run_eval(
        rows,
        selector_predictions=selector_predictions,
        embedding_features=embedding_features,
        margins=list(args.margin),
        folds=int(args.folds),
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        candidate_pool_size=int(args.candidate_pool_size),
        negatives_per_query=int(args.negatives_per_query),
        seed=int(args.seed),
        verbose=bool(args.verbose),
    )
    report["cache_jsonl"] = str(args.cache_jsonl)
    report["selector_predictions_jsonl"] = str(args.selector_predictions_jsonl)
    report["embedding_npz"] = str(args.embedding_npz)
    write_json(report, args.output_json)
    write_markdown(report, args.output_md)
    if args.predictions_jsonl:
        write_jsonl(predictions, args.predictions_jsonl)
    print(json.dumps({name: item["gates"] for name, item in report["variants"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
