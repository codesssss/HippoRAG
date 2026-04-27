#!/usr/bin/env python3
"""Train and evaluate a separate CEE need-edit STOP head."""

from __future__ import annotations

import argparse
import json
import math
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
from dpathrag_analyze_hard_negatives_v2 import oracle_edit_sequence  # noqa: E402
from dpathrag_cee_edit_policy import (  # noqa: E402
    LinearEditPolicy,
    build_training_examples,
    kfold_ranges,
    load_embedding_features,
    make_example,
    summarize_rank_and_oracle,
    summarize_selector_predictions,
)
from dpathrag_cee_oracle_stop_eval import aggregate_predictions, evaluate_policy_with_stop_rule  # noqa: E402
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.selector_data import support_metrics_for_indices  # noqa: E402


def mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def stdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((float(value) - m) ** 2 for value in values) / len(values))


def vector_stats(rows: Sequence[Sequence[float]], dim: int) -> list[float]:
    if not rows:
        return [0.0 for _ in range(dim * 4)]
    columns = [[float(row[index]) for row in rows] for index in range(dim)]
    output: list[float] = []
    for column in columns:
        output.extend([mean(column), max(column), min(column), stdev(column)])
    return output


def stop_feature(example: Any, *, top_k: int, candidate_pool_size: int) -> list[float]:
    candidate_count = sum(1 for item in example.candidate_mask if item)
    usable_pool = min(int(candidate_count), int(candidate_pool_size))
    dim = len(example.candidate_features[0])
    top_indices = list(range(min(int(top_k), candidate_count)))
    outside_indices = [index for index in range(usable_pool) if index not in set(top_indices)]
    top_rows = [example.candidate_features[index] for index in top_indices]
    outside_rows = [example.candidate_features[index] for index in outside_indices]
    top_mean = [mean([float(row[index]) for row in top_rows]) if top_rows else 0.0 for index in range(dim)]
    outside_max = [max([float(row[index]) for row in outside_rows]) if outside_rows else 0.0 for index in range(dim)]
    return (
        [1.0, float(candidate_count), float(usable_pool), float(len(outside_indices))]
        + [float(value) for value in example.query_features]
        + vector_stats(top_rows, dim)
        + vector_stats(outside_rows, dim)
        + [outside_max[index] - top_mean[index] for index in range(dim)]
    )


def beneficial_edit_exists(record: dict[str, Any], *, top_k: int, max_candidates: int, candidate_pool_size: int, example: Any) -> int:
    # Fast equivalent for the support-improving oracle used in this pilot:
    # an edit is useful when rank top-k contains a removable non-gold and the
    # candidate pool contains a missing gold document. The full oracle remains
    # the reference in the oracle-stop report; this label keeps learned STOP
    # cross-fit feasible.
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    rank_indices = list(range(min(int(top_k), len(candidates))))
    pool_indices = list(range(min(int(candidate_pool_size), len(candidates))))
    removable_non_gold = any(not int(candidates[index].get("gold_support") or 0) for index in rank_indices)
    missing_gold = any(index not in rank_indices and int(candidates[index].get("gold_support") or 0) == 1 for index in pool_indices)
    return int(removable_non_gold and missing_gold)


class NeedEditPolicy:
    def __init__(self) -> None:
        self.mean: Any = None
        self.std: Any = None
        self.weight: Any = None
        self.bias: float = 0.0

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> dict[str, Any]:
        import numpy as np

        array = np.asarray(features, dtype="float32")
        target = np.asarray(labels, dtype="float32")
        self.mean = array.mean(axis=0, keepdims=True)
        self.std = array.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        standardized = (array - self.mean) / self.std
        pos = standardized[target >= 0.5]
        neg = standardized[target < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("Need-edit policy requires positive and negative labels")
        pos_mean = pos.mean(axis=0)
        neg_mean = neg.mean(axis=0)
        self.weight = (pos_mean - neg_mean).astype("float32")
        prior = math.log((len(pos) + 1.0) / (len(neg) + 1.0))
        self.bias = float(-0.5 * (float(pos_mean @ pos_mean) - float(neg_mean @ neg_mean)) + prior)
        return {"rows": len(target), "positive_rate": round(float(target.mean()), 6), "fit_method": "standardized_nearest_centroid"}

    def score(self, features: Sequence[Sequence[float]]) -> list[float]:
        import numpy as np

        if self.weight is None or self.mean is None or self.std is None:
            raise RuntimeError("Need-edit policy must be fit before scoring")
        array = np.asarray(features, dtype="float32")
        standardized = (array - self.mean) / self.std
        logits = standardized @ self.weight + self.bias
        return [float(value) for value in logits.tolist()]


def build_stop_examples(
    records: Sequence[dict[str, Any]],
    *,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    embedding_features: dict[str, tuple[Any, Any]],
) -> tuple[list[list[float]], list[int], list[str]]:
    features: list[list[float]] = []
    labels: list[int] = []
    qids: list[str] = []
    for record in records:
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features=embedding_features)
        features.append(stop_feature(example, top_k=top_k, candidate_pool_size=candidate_pool_size))
        labels.append(
            beneficial_edit_exists(
                record,
                top_k=top_k,
                max_candidates=max_candidates,
                candidate_pool_size=candidate_pool_size,
                example=example,
            )
        )
        qids.append(row_qid(record))
    return features, labels, qids


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    pos = sum(int(label) for _, label in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.0
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if int(label) == 1:
            rank_sum += rank
    return round((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg), 6)


def stop_classification_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    preds = [1 if float(score) > float(threshold) else 0 for score in scores]
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and int(label) == 1)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and int(label) == 0)
    tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and int(label) == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and int(label) == 1)
    return {
        "threshold": round(float(threshold), 6),
        "auc": auc_score(labels, scores),
        "accuracy": round((tp + tn) / max(1, len(labels)), 6),
        "precision": round(tp / max(1, tp + fp), 6),
        "recall": round(tp / max(1, tp + fn), 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_candidates(scores: Sequence[float]) -> list[float]:
    if not scores:
        return [0.0]
    ordered = sorted(float(score) for score in scores)
    indices = sorted({0, len(ordered) // 10, len(ordered) // 4, len(ordered) // 2, 3 * len(ordered) // 4, 9 * len(ordered) // 10, len(ordered) - 1})
    values = [ordered[index] for index in indices]
    values.extend([min(ordered) - 1e-3, max(ordered) + 1e-3, 0.0])
    return sorted(set(round(value, 6) for value in values))


def evaluate_learned_stop(
    edit_policy: LinearEditPolicy,
    stop_policy: NeedEditPolicy,
    records: Sequence[dict[str, Any]],
    *,
    threshold: float,
    margin: float,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    embedding_features: dict[str, tuple[Any, Any]],
    stop_scores_by_qid: dict[str, float] | None = None,
    stop_labels_by_qid: dict[str, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int], list[float]]:
    stop_scores_by_qid = stop_scores_by_qid or {}
    stop_labels_by_qid = stop_labels_by_qid or {}
    labels: list[int] = []
    scores: list[float] = []
    force_stop_qids: set[str] = set()
    for record in records:
        qid = row_qid(record)
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features=embedding_features)
        if qid in stop_scores_by_qid:
            score = float(stop_scores_by_qid[qid])
        else:
            feature = stop_feature(example, top_k=top_k, candidate_pool_size=candidate_pool_size)
            score = stop_policy.score([feature])[0]
        if qid in stop_labels_by_qid:
            label = int(stop_labels_by_qid[qid])
        else:
            label = beneficial_edit_exists(record, top_k=top_k, max_candidates=max_candidates, candidate_pool_size=candidate_pool_size, example=example)
        labels.append(label)
        scores.append(score)
        if score <= float(threshold):
            force_stop_qids.add(qid)

    class LearnedStopWrapper:
        pass

    # Reuse the oracle-stop evaluator by providing a tiny custom loop here.
    from dpathrag_cee_oracle_stop_eval import evaluate_policy_with_stop_rule as _unused  # noqa: F401
    from dpathrag_cee_oracle_stop_eval import should_force_stop as _unused2  # noqa: F401
    from dpathrag_cee_edit_policy import choose_action, apply_action
    from dpathrag_analyze_hard_negatives import is_gold
    from src.dpathrag.selector_data import selection_overlap, summarize_selector_metrics, support_metrics_for_indices

    metric_rows: list[dict[str, float]] = []
    predictions: list[dict[str, Any]] = []
    totals = {"added_gold": 0, "added_non_gold": 0, "removed_gold": 0, "removed_non_gold": 0, "queries_with_edit": 0}
    for record in records:
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features=embedding_features)
        rank_indices = list(range(min(int(top_k), len(candidates))))
        indices = list(rank_indices)
        edits: list[dict[str, Any]] = []
        if row_qid(record) not in force_stop_qids:
            action, score, stop_score = choose_action(
                edit_policy,
                example,
                indices,
                candidate_count=len(candidates),
                candidate_pool_size=candidate_pool_size,
                min_edit_margin=float(margin),
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
        metric_rows.append(metrics)
        predictions.append({"qid": row_qid(record), "selected_indices": indices, "edits": edits, "metrics": metrics})
    summary = summarize_selector_metrics(metric_rows)
    summary.update(totals)
    summary["stop_rate"] = round(1.0 - totals["queries_with_edit"] / max(1.0, float(len(records))), 4)
    summary["non_gold_per_gold"] = round(float(totals["added_non_gold"]) / max(1.0, float(totals["added_gold"])), 4)
    return summary, predictions, labels, scores


def choose_threshold(
    edit_policy: LinearEditPolicy,
    stop_policy: NeedEditPolicy,
    dev_rows: Sequence[dict[str, Any]],
    *,
    margin: float,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    embedding_features: dict[str, tuple[Any, Any]],
    rank_support_complete: float,
    selector_added_non_gold: float,
) -> tuple[float, dict[str, Any]]:
    features, labels, _ = build_stop_examples(
        dev_rows,
        top_k=top_k,
        max_candidates=max_candidates,
        candidate_pool_size=candidate_pool_size,
        embedding_features=embedding_features,
    )
    scores = stop_policy.score(features)
    dev_qids = [row_qid(row) for row in dev_rows]
    scores_by_qid = {qid: float(score) for qid, score in zip(dev_qids, scores)}
    labels_by_qid = {qid: int(label) for qid, label in zip(dev_qids, labels)}
    best_threshold = 0.0
    best_report: dict[str, Any] = {}
    best_key = (-1, -1.0, -999.0, -999999.0)
    for threshold in threshold_candidates(scores):
        summary, _, _, _ = evaluate_learned_stop(
            edit_policy,
            stop_policy,
            dev_rows,
            threshold=threshold,
            margin=margin,
            top_k=top_k,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            embedding_features=embedding_features,
            stop_scores_by_qid=scores_by_qid,
            stop_labels_by_qid=labels_by_qid,
        )
        gates = {
            "support_complete_gain_ge_1pp": float(summary.get("support_complete") or 0.0) >= float(rank_support_complete) + 0.01,
            "non_gold_per_gold_le_5": float(summary.get("non_gold_per_gold") or 999.0) <= 5.0,
            "added_non_gold_50pct_below_v1": float(summary.get("added_non_gold") or 0.0) <= 0.5 * float(selector_added_non_gold),
        }
        passed = sum(1 for value in gates.values() if value)
        key = (
            passed,
            float(summary.get("support_complete") or 0.0),
            -float(summary.get("non_gold_per_gold") or 999.0),
            -float(summary.get("added_non_gold") or 0.0),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_report = {"summary": summary, "gates": {**gates, "passed_count": passed}, "classification": stop_classification_metrics(labels, scores, threshold)}
    return best_threshold, best_report


def split_train_dev(rows: Sequence[dict[str, Any]], *, seed: int, dev_fraction: float = 0.2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    shuffled = list(rows)
    rng.shuffle(shuffled)
    dev_size = max(1, int(round(len(shuffled) * float(dev_fraction))))
    return shuffled[dev_size:], shuffled[:dev_size]


def run_crossfit(
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
    ranges = kfold_ranges(len(rows), folds)
    oracle_and_rank = summarize_rank_and_oracle(rows, top_k=top_k, max_candidates=max_candidates, candidate_pool_size=candidate_pool_size)
    selector_summary = summarize_selector_predictions(rows, selector_predictions, top_k=top_k, max_candidates=max_candidates)
    all_predictions: list[dict[str, Any]] = []
    variants: dict[str, list[dict[str, Any]]] = {f"learned_stop_margin{int(m)}": [] for m in margins}
    fold_reports: list[dict[str, Any]] = []
    for fold, (start, end) in enumerate(ranges):
        if verbose:
            print(f"[learned-stop] fold {fold}", file=sys.stderr, flush=True)
        eval_rows = list(rows[start:end])
        eval_qids = {row_qid(row) for row in eval_rows}
        train_rows = [row for row in rows if row_qid(row) not in eval_qids]
        stop_train_rows, stop_dev_rows = split_train_dev(train_rows, seed=seed + fold)
        edit_features, edit_labels, edit_train_summary = build_training_examples(
            train_rows,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            top_k=top_k,
            embedding_features=embedding_features,
            negatives_per_query=negatives_per_query,
            seed=seed + fold,
        )
        edit_policy = LinearEditPolicy(epochs=1, learning_rate=0.0, batch_size=256, seed=seed + fold)
        edit_fit = edit_policy.fit(edit_features, edit_labels)
        stop_features, stop_labels, _ = build_stop_examples(
            stop_train_rows,
            top_k=top_k,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            embedding_features=embedding_features,
        )
        stop_policy = NeedEditPolicy()
        stop_fit = stop_policy.fit(stop_features, stop_labels)
        eval_stop_features, eval_stop_labels, eval_stop_qids = build_stop_examples(
            eval_rows,
            top_k=top_k,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            embedding_features=embedding_features,
        )
        eval_stop_scores = stop_policy.score(eval_stop_features)
        eval_scores_by_qid = {qid: float(score) for qid, score in zip(eval_stop_qids, eval_stop_scores)}
        eval_labels_by_qid = {qid: int(label) for qid, label in zip(eval_stop_qids, eval_stop_labels)}
        fold_report: dict[str, Any] = {"fold": fold, "edit_train": edit_train_summary, "edit_fit": edit_fit, "stop_fit": stop_fit, "margins": {}}
        for margin in margins:
            threshold, dev_report = choose_threshold(
                edit_policy,
                stop_policy,
                stop_dev_rows,
                margin=margin,
                top_k=top_k,
                max_candidates=max_candidates,
                candidate_pool_size=candidate_pool_size,
                embedding_features=embedding_features,
                rank_support_complete=float(oracle_and_rank["rank_topk"].get("support_complete") or 0.0),
                selector_added_non_gold=float(selector_summary.get("added_non_gold") or 0.0),
            )
            summary, preds, eval_labels, eval_scores = evaluate_learned_stop(
                edit_policy,
                stop_policy,
                eval_rows,
                threshold=threshold,
                margin=margin,
                top_k=top_k,
                max_candidates=max_candidates,
                candidate_pool_size=candidate_pool_size,
                embedding_features=embedding_features,
                stop_scores_by_qid=eval_scores_by_qid,
                stop_labels_by_qid=eval_labels_by_qid,
            )
            variant = f"learned_stop_margin{int(margin) if float(margin).is_integer() else margin}"
            for row in preds:
                row["variant"] = variant
                row["fold"] = fold
            variants[variant].extend(preds)
            all_predictions.extend(preds)
            fold_report["margins"][variant] = {
                "threshold": threshold,
                "dev": dev_report,
                "eval_summary": summary,
                "eval_classification": stop_classification_metrics(eval_labels, eval_scores, threshold),
            }
        fold_reports.append(fold_report)
    report: dict[str, Any] = {
        "rows": len(rows),
        "folds": int(folds),
        "candidate_pool_size": int(candidate_pool_size),
        "rank_topk": oracle_and_rank["rank_topk"],
        "selector_v1": selector_summary,
        "oracle_edit1": oracle_and_rank["oracle_edit1"],
        "variants": {},
        "fold_reports": fold_reports,
    }
    for name, preds in sorted(variants.items()):
        summary = aggregate_predictions(rows, preds, max_candidates=max_candidates)
        gates_payload = {"rank_topk": report["rank_topk"], "selector_v1": report["selector_v1"], "learned_edit1": summary, "learned_edit2": summary}
        from dpathrag_cee_edit_policy import cee_pilot_gates

        summary["gates"] = cee_pilot_gates(gates_payload)["learned_edit1"]
        report["variants"][name] = summary
    return report, all_predictions


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# CEE Learned STOP Evaluation",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Folds: `{report['folds']}`",
        f"- Candidate pool size: `{report['candidate_pool_size']}`",
        "",
        "| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Edits | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = {"rank_topk": report["rank_topk"], "selector_v1": report["selector_v1"], "oracle_edit1": report["oracle_edit1"]}
    for name, item in {**base, **report["variants"]}.items():
        gates = item.get("gates", {})
        gate_text = f"{gates.get('passed_count', '')}/3" if gates else ""
        lines.append(
            f"| {name} | {item.get('support_recall', 0.0):.4f} | {item.get('support_complete', 0.0):.4f} | "
            f"{item.get('selected_gold_count', 0.0):.4f} | {item.get('bridge_entity_recall', 0.0):.4f} | "
            f"{item.get('selection_overlap', 0.0):.4f} | {item.get('queries_with_edit', '')} | "
            f"{item.get('stop_rate', '')} | {item.get('added_gold', '')} | {item.get('added_non_gold', '')} | "
            f"{item.get('non_gold_per_gold', '')} | {gate_text} |"
        )
    lines.extend(["", "## Fold Thresholds", "", "| Fold | Variant | Threshold | Dev Gates | Eval AUC | Eval Precision | Eval Recall |", "|---|---|---:|---:|---:|---:|---:|"])
    for fold in report["fold_reports"]:
        for variant, item in fold["margins"].items():
            dev_gates = item["dev"]["gates"]["passed_count"]
            cls = item["eval_classification"]
            lines.append(
                f"| {fold['fold']} | {variant} | {item['threshold']:.4f} | {dev_gates}/3 | "
                f"{cls['auc']:.4f} | {cls['precision']:.4f} | {cls['recall']:.4f} |"
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
    report, predictions = run_crossfit(
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
