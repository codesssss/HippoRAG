#!/usr/bin/env python3
"""Train and evaluate CEE-v2 pairwise edit admission."""

from __future__ import annotations

import argparse
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
from dpathrag_cee_edit_policy import kfold_ranges, load_embedding_features, make_example, summarize_rank_and_oracle, summarize_selector_predictions  # noqa: E402
from dpathrag_cee_pairwise_common import (  # noqa: E402
    LinearPairwiseScorer,
    MLPPairwiseScorer,
    edit_feature_vector,
    enumerate_edit_summaries,
    platt_fit,
    platt_prob,
    rank_indices_for,
    read_jsonl,
    replace_index,
    write_jsonl,
)
from src.dpathrag.io import write_json  # noqa: E402
from src.dpathrag.selector_data import selection_overlap, summarize_selector_metrics, support_metrics_for_indices  # noqa: E402


def summarize_prediction_rows(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], *, top_k: int, max_candidates: int) -> dict[str, Any]:
    by_qid = {row_qid(row): row for row in records}
    predictions = [row for row in predictions if str(row.get("qid")) in by_qid]
    metrics = summarize_selector_metrics([row["metrics"] for row in predictions])
    totals = {"added_gold": 0, "added_non_gold": 0, "removed_gold": 0, "removed_non_gold": 0, "queries_with_edit": 0}
    rank_complete_over_edit = 0
    oracle_beneficial_queries = 0
    oracle_beneficial_admitted = 0
    oracle_not_beneficial_false_positive = 0
    lexical_hard_negative_admissions = 0
    for row in predictions:
        record = by_qid[str(row["qid"])]
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features={})
        rank_indices = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
        rank_metrics = support_metrics_for_indices(example, rank_indices)
        beneficial_keys = {tuple(item) for item in row.get("oracle_beneficial_keys", [])}
        if beneficial_keys:
            oracle_beneficial_queries += 1
        edits = list(row.get("edits") or [])
        if edits:
            totals["queries_with_edit"] += 1
            if float(rank_metrics.get("support_complete") or 0.0) >= 1.0:
                rank_complete_over_edit += 1
            edit = edits[0]
            key = (int(edit["remove_index"]), int(edit["add_index"]))
            if beneficial_keys and key in beneficial_keys:
                oracle_beneficial_admitted += 1
            if not beneficial_keys:
                oracle_not_beneficial_false_positive += 1
            if str(edit.get("hard_negative_type") or "") == "lexical_hard_negative":
                lexical_hard_negative_admissions += 1
            candidates = list(record.get("candidates") or [])[: int(max_candidates)]
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
        "rank_complete_over_edit_rate": round(float(rank_complete_over_edit) / max(1.0, float(len(predictions))), 6),
        "oracle_beneficial_edit_recall": round(float(oracle_beneficial_admitted) / max(1.0, float(oracle_beneficial_queries)), 6),
        "oracle_not_beneficial_false_positive_rate": round(float(oracle_not_beneficial_false_positive) / max(1.0, float(len(predictions) - oracle_beneficial_queries)), 6),
        "lexical_hard_negative_admission_rate": round(float(lexical_hard_negative_admissions) / max(1.0, float(totals["queries_with_edit"])), 6),
    }


def fit_calibration(model: Any, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scores = []
    labels = []
    for row in rows:
        scores.append(model.score(row["positive_features"]))
        labels.append(1)
        scores.append(model.score(row["negative_features"]))
        labels.append(0)
    a, b = platt_fit(scores, labels)
    return {
        "a": a,
        "b": b,
        "calib_rows": len(rows),
        "calib_scores": len(scores),
        "positive_prob_mean": round(sum(platt_prob(score, a, b) for score, label in zip(scores, labels) if label) / max(1, sum(labels)), 6),
        "negative_prob_mean": round(
            sum(platt_prob(score, a, b) for score, label in zip(scores, labels) if not label) / max(1, len(labels) - sum(labels)),
            6,
        ),
    }


def fit_candidate_calibration(
    model: Any,
    calib_pairs: Sequence[dict[str, Any]],
    row_by_qid: dict[str, dict[str, Any]],
    *,
    embedding_features: dict[str, Any],
    embedding_feature_names: Sequence[str],
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
) -> dict[str, Any]:
    qids = sorted({str(row.get("qid")) for row in calib_pairs if str(row.get("qid")) in row_by_qid})
    scores = []
    labels = []
    for qid in qids:
        record = row_by_qid[qid]
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features=embedding_features)
        base = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
        edits = enumerate_edit_summaries(
            record,
            top_k=top_k,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            example=example,
            embedding_by_qid=embedding_features,
            embedding_feature_names=embedding_feature_names,
        )
        for edit in edits:
            scores.append(model.score(edit_feature_vector(example, base, edit)))
            labels.append(1 if edit["beneficial"] else 0)
    if not scores or not any(labels):
        return fit_calibration(model, calib_pairs)
    a, b = platt_fit(scores, labels, steps=800, lr=0.05)
    positives = sum(labels)
    return {
        "a": a,
        "b": b,
        "calib_rows": len(calib_pairs),
        "calib_queries": len(qids),
        "calib_scores": len(scores),
        "positive_rate": round(float(positives) / max(1.0, float(len(labels))), 6),
        "positive_prob_mean": round(sum(platt_prob(score, a, b) for score, label in zip(scores, labels) if label) / max(1, positives), 6),
        "negative_prob_mean": round(
            sum(platt_prob(score, a, b) for score, label in zip(scores, labels) if not label) / max(1, len(labels) - positives),
            6,
        ),
        "calibration": "candidate_level_all_edits",
    }


def evaluate_fold(
    model: Any,
    calibration: dict[str, Any],
    eval_rows: Sequence[dict[str, Any]],
    *,
    embedding_features: dict[str, Any],
    embedding_feature_names: Sequence[str],
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    fold: int,
    variant: str,
) -> list[dict[str, Any]]:
    predictions = []
    for record in eval_rows:
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features=embedding_features)
        base = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
        edits = enumerate_edit_summaries(
            record,
            top_k=top_k,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            example=example,
            embedding_by_qid=embedding_features,
            embedding_feature_names=embedding_feature_names,
        )
        scored = []
        for edit in edits:
            score = model.score(edit_feature_vector(example, base, edit))
            prob = platt_prob(score, float(calibration["a"]), float(calibration["b"]))
            scored.append((prob, score, edit))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = list(base)
        admitted = []
        if scored and float(scored[0][0]) >= 0.5:
            prob, score, edit = scored[0]
            selected = replace_index(base, int(edit["remove_index"]), int(edit["add_index"]))
            admitted.append(
                {
                    "remove_index": int(edit["remove_index"]),
                    "add_index": int(edit["add_index"]),
                    "prob_beneficial": round(float(prob), 6),
                    "score": round(float(score), 6),
                    "beneficial": bool(edit["beneficial"]),
                    "hard_negative_type": edit.get("hard_negative_type", ""),
                }
            )
        metrics = support_metrics_for_indices(example, selected)
        metrics["selection_overlap"] = selection_overlap(selected, base)
        predictions.append(
            {
                "qid": row_qid(record),
                "variant": variant,
                "fold": fold,
                "selected_indices": selected,
                "selected_titles": [example.candidate_titles[index] for index in selected],
                "rank_topk_titles": [example.candidate_titles[index] for index in base],
                "edits": admitted,
                "metrics": metrics,
                "oracle_beneficial_keys": [[int(edit["remove_index"]), int(edit["add_index"])] for edit in edits if edit["beneficial"]],
                "best_prob": round(float(scored[0][0]), 6) if scored else 0.0,
            }
        )
    return predictions


def make_model(name: str, *, seed: int, epochs: int, hidden_dim: int) -> Any:
    if name == "linear":
        return LinearPairwiseScorer()
    if name == "mlp":
        return MLPPairwiseScorer(hidden_dim=hidden_dim, epochs=epochs, seed=seed)
    raise ValueError(f"Unknown model: {name}")


def run_eval(
    rows: Sequence[dict[str, Any]],
    *,
    dataset_dir: str | Path,
    selector_predictions: Sequence[dict[str, Any]],
    shallow_predictions: Sequence[dict[str, Any]],
    embedding_features: dict[str, Any],
    embedding_feature_names: Sequence[str],
    folds: int,
    models: Sequence[str],
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    seed: int,
    mlp_epochs: int,
    hidden_dim: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_dir = Path(dataset_dir)
    ranges = kfold_ranges(len(rows), folds)
    row_by_qid = {row_qid(row): row for row in rows}
    all_predictions = []
    fold_reports = []
    variant_predictions: dict[str, list[dict[str, Any]]] = {f"cee_pairwise_{name}_v0": [] for name in models}
    for fold, (start, end) in enumerate(ranges):
        eval_rows = list(rows[start:end])
        train_pairs = read_jsonl(dataset_dir / f"fold_{fold}.train.jsonl")
        calib_pairs = read_jsonl(dataset_dir / f"fold_{fold}.calib.jsonl")
        fold_report: dict[str, Any] = {"fold": fold, "train_pairs": len(train_pairs), "calib_pairs": len(calib_pairs), "models": {}}
        for model_name in models:
            variant = f"cee_pairwise_{model_name}_v0"
            model = make_model(model_name, seed=int(seed) + fold, epochs=mlp_epochs, hidden_dim=hidden_dim)
            fit = model.fit(train_pairs)
            calibration = fit_candidate_calibration(
                model,
                calib_pairs,
                row_by_qid,
                embedding_features=embedding_features,
                embedding_feature_names=embedding_feature_names,
                top_k=top_k,
                max_candidates=max_candidates,
                candidate_pool_size=candidate_pool_size,
            )
            preds = evaluate_fold(
                model,
                calibration,
                eval_rows,
                embedding_features=embedding_features,
                embedding_feature_names=embedding_feature_names,
                top_k=top_k,
                max_candidates=max_candidates,
                candidate_pool_size=candidate_pool_size,
                fold=fold,
                variant=variant,
            )
            variant_predictions[variant].extend(preds)
            all_predictions.extend(preds)
            fold_report["models"][variant] = {"fit": fit, "calibration": calibration}
        fold_reports.append(fold_report)

    rank_oracle = summarize_rank_and_oracle(
        rows,
        top_k=top_k,
        max_candidates=max_candidates,
        candidate_pool_size=candidate_pool_size,
    )
    baselines = {
        "rank_top5": rank_oracle["rank_topk"],
        "oracle_edit1_top20": rank_oracle["oracle_edit1"],
    }
    if selector_predictions:
        baselines["selector_v1"] = summarize_selector_predictions(rows, selector_predictions, top_k=top_k, max_candidates=max_candidates)
    if shallow_predictions:
        baselines["shallow_cee_margin20"] = summarize_prediction_rows(rows, shallow_predictions, top_k=top_k, max_candidates=max_candidates)
    summaries = {
        variant: summarize_prediction_rows(rows, preds, top_k=top_k, max_candidates=max_candidates)
        for variant, preds in variant_predictions.items()
    }
    gates = {}
    rank_complete = float(baselines["rank_top5"].get("support_complete") or 0.0)
    selector_added_non_gold = int(baselines.get("selector_v1", {}).get("added_non_gold") or 1205)
    for variant, summary in summaries.items():
        checks = {
            "support_complete_ge_rank_plus_1pp": float(summary.get("support_complete") or 0.0) >= rank_complete + 0.01,
            "non_gold_per_gold_le_5": float(summary.get("non_gold_per_gold") or 999.0) <= 5.0,
            "added_non_gold_le_half_selector_v1": int(summary.get("added_non_gold") or 0) <= int(0.5 * selector_added_non_gold),
        }
        gates[variant] = {"checks": checks, "passed": sum(1 for value in checks.values() if value), "total": len(checks)}
    return {
        "rows": len(rows),
        "folds": folds,
        "top_k": top_k,
        "candidate_pool_size": candidate_pool_size,
        "baselines": baselines,
        "summaries": summaries,
        "gates": gates,
        "fold_reports": fold_reports,
    }, all_predictions


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# CEE Pairwise v0 Eval",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Folds: `{report['folds']}`",
        f"- Candidate pool: `top{report['candidate_pool_size']}`",
        "",
        "## Main Selection Metrics",
        "",
        "| Variant | Support Complete | Support Recall | Added Gold | Added Non-Gold | Non-Gold/Gold | Stop Rate | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    variants = {**report["baselines"], **report["summaries"]}
    for name, summary in variants.items():
        gate = report["gates"].get(name, {})
        gate_text = f"{gate.get('passed', '-')}/{gate.get('total', '-')}" if gate else "-"
        lines.append(
            f"| `{name}` | {summary.get('support_complete', 0):.4f} | {summary.get('support_recall', 0):.4f} | "
            f"{summary.get('added_gold', '-')} | {summary.get('added_non_gold', '-')} | {summary.get('non_gold_per_gold', '-')} | "
            f"{summary.get('stop_rate', '-')} | {gate_text} |"
        )
    lines.extend(
        [
            "",
            "## Admission Diagnostics",
            "",
            "| Variant | Rank-Complete Over-Edit | Oracle Beneficial Recall | Oracle Non-Beneficial FP | Lexical HN Admission |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, summary in report["summaries"].items():
        lines.append(
            f"| `{name}` | {summary.get('rank_complete_over_edit_rate', 0):.4f} | {summary.get('oracle_beneficial_edit_recall', 0):.4f} | "
            f"{summary.get('oracle_not_beneficial_false_positive_rate', 0):.4f} | {summary.get('lexical_hard_negative_admission_rate', 0):.4f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--embedding_npz", default="data/dpathrag/cache/selector_embedding_proprag_local1000_rp64.npz")
    parser.add_argument("--pairwise_dir", default="data/dpathrag/cee_pairwise_train_folds")
    parser.add_argument("--selector_predictions_jsonl", default="reports/dpathrag/kfold_proprag/selector_kfold_proprag_embed_local1000.predictions.jsonl")
    parser.add_argument("--shallow_predictions_jsonl", default="reports/dpathrag/cee_edit_policy_proprag_kfold1000_margin20.predictions.jsonl")
    parser.add_argument("--output_json", default="reports/dpathrag/cee_pairwise_v0_eval.json")
    parser.add_argument("--output_md", default="reports/dpathrag/cee_pairwise_v0_eval.md")
    parser.add_argument("--output_predictions", default="reports/dpathrag/cee_pairwise_v0.predictions.jsonl")
    parser.add_argument("--output_ablation_md", default="reports/dpathrag/cee_pairwise_ablation.md")
    parser.add_argument("--models", default="linear,mlp")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mlp_epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    selector_predictions = load_jsonl(args.selector_predictions_jsonl) if args.selector_predictions_jsonl else []
    shallow_predictions = load_jsonl(args.shallow_predictions_jsonl) if args.shallow_predictions_jsonl else []
    embedding_features, embedding_feature_names = load_embedding_features(args.embedding_npz, limit=int(args.limit))
    models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    report, predictions = run_eval(
        rows,
        dataset_dir=args.pairwise_dir,
        selector_predictions=selector_predictions,
        shallow_predictions=shallow_predictions,
        embedding_features=embedding_features,
        embedding_feature_names=embedding_feature_names,
        folds=int(args.folds),
        models=models,
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        candidate_pool_size=int(args.candidate_pool_size),
        seed=int(args.seed),
        mlp_epochs=int(args.mlp_epochs),
        hidden_dim=int(args.hidden_dim),
    )
    write_json(report, args.output_json)
    write_jsonl(predictions, args.output_predictions)
    write_markdown(report, args.output_md)
    write_markdown(report, args.output_ablation_md)


if __name__ == "__main__":
    main()
