#!/usr/bin/env python3
"""Cross-fit Conservative Evidence Editing pilot for D-PathRAG.

The policy starts from rank top-k and predicts whether to STOP or apply one
local replacement edit. Gold labels are used only to construct supervised edit
targets and report oracle/evaluation metrics; inference features are derived
from rank, lexical, retriever, and optional embedding features.
"""

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

from dpathrag_analyze_hard_negatives import is_gold, load_jsonl, row_qid  # noqa: E402
from dpathrag_analyze_hard_negatives_v2 import oracle_edit_sequence, replace_index  # noqa: E402
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.selector_data import (  # noqa: E402
    FEATURE_NAMES,
    featurize_selector_record,
    selection_overlap,
    summarize_selector_metrics,
    support_metrics_for_indices,
)


Action = dict[str, Any]


def finite_floats(values: Sequence[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        item = float(value)
        if not math.isfinite(item):
            item = 0.0
        output.append(item)
    return output


def load_embedding_features(path: str, *, limit: int = 0) -> tuple[dict[str, tuple[Any, Any]], list[str]]:
    if not path:
        return {}, []
    import numpy as np

    payload = np.load(path, allow_pickle=True)
    qids = [str(qid) for qid in payload["qids"].tolist()]
    candidate_features = payload["candidate_features"]
    query_features = payload["query_features"]
    if limit > 0:
        qids = qids[: int(limit)]
        candidate_features = candidate_features[: int(limit)]
        query_features = query_features[: int(limit)]
    feature_names = [str(item) for item in payload["feature_names"].tolist()] if "feature_names" in payload.files else []
    return {
        qid: (candidate_features[idx].astype("float32"), query_features[idx].astype("float32"))
        for idx, qid in enumerate(qids)
    }, list(feature_names)


def make_example(record: dict[str, Any], *, max_candidates: int, top_k: int, embedding_features: dict[str, tuple[Any, Any]]) -> Any:
    qid = row_qid(record)
    candidate_extra, query_extra = embedding_features.get(qid, (None, None))
    return featurize_selector_record(
        record,
        max_candidates=max_candidates,
        path_len=top_k,
        candidate_extra_features=candidate_extra,
        query_extra_features=query_extra,
    )


def action_feature(example: Any, current_indices: Sequence[int], action: Action) -> list[float]:
    feature_dim = len(example.candidate_features[0])
    selected = [int(index) for index in current_indices]
    set_mean = [
        sum(float(example.candidate_features[index][dim]) for index in selected) / max(1, len(selected))
        for dim in range(feature_dim)
    ]
    if action["type"] == "stop":
        add_features = [0.0 for _ in range(feature_dim)]
        remove_features = [0.0 for _ in range(feature_dim)]
    else:
        add_features = finite_floats(example.candidate_features[int(action["add_index"])])
        remove_features = finite_floats(example.candidate_features[int(action["remove_index"])])
    diff_features = [add - remove for add, remove in zip(add_features, remove_features)]
    return (
        [1.0 if action["type"] == "stop" else 0.0]
        + finite_floats(example.query_features)
        + set_mean
        + add_features
        + remove_features
        + diff_features
    )


def enumerate_actions(current_indices: Sequence[int], *, candidate_count: int, candidate_pool_size: int) -> list[Action]:
    base = list(int(index) for index in current_indices)
    base_set = set(base)
    usable_pool = min(int(candidate_count), int(candidate_pool_size))
    actions: list[Action] = [{"type": "stop"}]
    for remove_index in base:
        for add_index in range(usable_pool):
            if add_index in base_set:
                continue
            actions.append({"type": "edit", "remove_index": int(remove_index), "add_index": int(add_index)})
    return actions


def apply_action(current_indices: Sequence[int], action: Action) -> list[int]:
    if action["type"] == "stop":
        return [int(index) for index in current_indices]
    return replace_index(current_indices, int(action["remove_index"]), int(action["add_index"]))


def best_support_edit_label(record: dict[str, Any], action: Action, *, top_k: int) -> int:
    if action["type"] == "stop":
        return 0
    candidates = list(record.get("candidates") or [])
    add_index = int(action["add_index"])
    remove_index = int(action["remove_index"])
    if add_index >= len(candidates) or remove_index >= len(candidates):
        return 0
    return int(is_gold(candidates[add_index]) and not is_gold(candidates[remove_index]))


def oracle_stop_label(record: dict[str, Any], *, top_k: int, candidate_pool_size: int, max_candidates: int) -> int:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    rank_indices = list(range(min(int(top_k), len(candidates))))
    pool_indices = list(range(min(int(candidate_pool_size), len(candidates))))
    removable_non_gold = any(not is_gold(candidates[index]) for index in rank_indices)
    missing_gold = any(index not in rank_indices and is_gold(candidates[index]) for index in pool_indices)
    return int(not (removable_non_gold and missing_gold))


def build_training_examples(
    records: Sequence[dict[str, Any]],
    *,
    max_candidates: int,
    candidate_pool_size: int,
    top_k: int,
    embedding_features: dict[str, tuple[Any, Any]],
    negatives_per_query: int,
    seed: int,
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    rng = random.Random(int(seed))
    features: list[list[float]] = []
    labels: list[int] = []
    summary = {"queries": len(records), "positive_edits": 0, "positive_stops": 0, "negative_edits": 0, "negative_stops": 0}
    for record in records:
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        if len(candidates) < int(top_k):
            continue
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features=embedding_features)
        current = list(range(int(top_k)))
        actions = enumerate_actions(current, candidate_count=len(candidates), candidate_pool_size=candidate_pool_size)
        stop_is_positive = oracle_stop_label(
            record,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            max_candidates=max_candidates,
        )
        if stop_is_positive:
            summary["positive_stops"] += 1
        else:
            summary["negative_stops"] += 1

        positive_actions: list[Action] = []
        negative_actions: list[Action] = []
        for action in actions[1:]:
            label = best_support_edit_label(record, action, top_k=top_k)
            if label:
                positive_actions.append(action)
            else:
                negative_actions.append(action)
        for action in positive_actions:
            features.append(action_feature(example, current, action))
            labels.append(1)
            summary["positive_edits"] += 1
        rng.shuffle(negative_actions)
        for action in negative_actions[: int(negatives_per_query)]:
            features.append(action_feature(example, current, action))
            labels.append(0)
            summary["negative_edits"] += 1
    return features, labels, summary


class LinearEditPolicy:
    def __init__(self, *, epochs: int, learning_rate: float, batch_size: int, seed: int) -> None:
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mean: Any = None
        self.std: Any = None
        self.model: Any = None
        self.weight: Any = None
        self.bias: float = 0.0

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> dict[str, Any]:
        import numpy as np

        array = np.asarray(features, dtype="float32")
        target = np.asarray(labels, dtype="float32")
        self.mean = array.mean(axis=0, keepdims=True)
        self.std = array.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        array = (array - self.mean) / self.std
        pos = array[target >= 0.5]
        neg = array[target < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("Edit policy needs both positive and negative training examples")
        pos_mean = pos.mean(axis=0)
        neg_mean = neg.mean(axis=0)
        self.weight = (pos_mean - neg_mean).astype("float32")
        prior = math.log((len(pos) + 1.0) / (len(neg) + 1.0))
        self.bias = float(-0.5 * (float(pos_mean @ pos_mean) - float(neg_mean @ neg_mean)) + prior)
        return {
            "rows": len(target),
            "positive_rate": round(float(target.mean()) if len(target) else 0.0, 6),
            "fit_method": "standardized_nearest_centroid_log_prior",
        }

    def score(self, features: Sequence[Sequence[float]]) -> list[float]:
        import numpy as np

        if self.weight is None or self.mean is None or self.std is None:
            raise RuntimeError("Policy must be fit before scoring")
        array = np.asarray(features, dtype="float32")
        array = (array - self.mean) / self.std
        logits = array @ self.weight + self.bias
        return [float(value) for value in logits.tolist()]


def choose_action(
    policy: LinearEditPolicy,
    example: Any,
    current_indices: Sequence[int],
    *,
    candidate_count: int,
    candidate_pool_size: int,
    min_edit_margin: float,
) -> tuple[Action, float, float]:
    actions = enumerate_actions(current_indices, candidate_count=candidate_count, candidate_pool_size=candidate_pool_size)
    if len(actions) <= 1:
        return actions[0], float(min_edit_margin), float(min_edit_margin)
    features = [action_feature(example, current_indices, action) for action in actions[1:]]
    scores = policy.score(features)
    stop_score = float(min_edit_margin)
    best_offset = max(range(len(scores)), key=lambda idx: scores[idx])
    if scores[best_offset] <= stop_score:
        return actions[0], scores[best_offset], stop_score
    return actions[best_offset + 1], scores[best_offset], stop_score


def evaluate_policy(
    policy: LinearEditPolicy,
    records: Sequence[dict[str, Any]],
    *,
    max_candidates: int,
    candidate_pool_size: int,
    top_k: int,
    steps: int,
    embedding_features: dict[str, tuple[Any, Any]],
    min_edit_margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, float]] = []
    prediction_rows: list[dict[str, Any]] = []
    totals = {"added_gold": 0, "added_non_gold": 0, "removed_gold": 0, "removed_non_gold": 0, "queries_with_edit": 0}
    for record in records:
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features=embedding_features)
        indices = list(range(min(int(top_k), len(candidates))))
        edits: list[dict[str, Any]] = []
        for _ in range(int(steps)):
            action, score, stop_score = choose_action(
                policy,
                example,
                indices,
                candidate_count=len(candidates),
                candidate_pool_size=candidate_pool_size,
                min_edit_margin=min_edit_margin,
            )
            if action["type"] == "stop":
                break
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
        rank_indices = list(range(min(int(top_k), len(candidates))))
        metrics["selection_overlap"] = selection_overlap(indices, rank_indices)
        rows.append(metrics)
        prediction_rows.append(
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
    }, prediction_rows


def summarize_rank_and_oracle(records: Sequence[dict[str, Any]], *, top_k: int, max_candidates: int, candidate_pool_size: int) -> dict[str, Any]:
    rank_rows: list[dict[str, float]] = []
    oracle1_rows: list[dict[str, Any]] = []
    oracle2_rows: list[dict[str, Any]] = []
    for record in records:
        example = featurize_selector_record(record, max_candidates=max_candidates, path_len=top_k)
        rank_indices = list(range(min(int(top_k), len(record.get("candidates") or []))))
        rank_rows.append(support_metrics_for_indices(example, rank_indices))
        oracle1_rows.append(
            oracle_edit_sequence(
                record,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                max_candidates=max_candidates,
                steps=1,
                example=example,
            )
        )
        oracle2_rows.append(
            oracle_edit_sequence(
                record,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                max_candidates=max_candidates,
                steps=2,
                example=example,
            )
        )

    def summarize_oracle(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        metrics = summarize_selector_metrics([row["metrics"] for row in rows])
        added_gold = sum(int(row.get("added_gold") or 0) for row in rows)
        added_non_gold = sum(int(row.get("added_non_gold") or 0) for row in rows)
        queries_with_edit = sum(1 for row in rows if row["edits"])
        return {
            **metrics,
            "queries_with_edit": queries_with_edit,
            "stop_rate": round(1.0 - queries_with_edit / max(1.0, float(len(rows))), 4),
            "added_gold": added_gold,
            "added_non_gold": added_non_gold,
            "non_gold_per_gold": round(float(added_non_gold) / max(1.0, float(added_gold)), 4),
        }

    return {
        "rank_topk": summarize_selector_metrics(rank_rows),
        "oracle_edit1": summarize_oracle(oracle1_rows),
        "oracle_edit2": summarize_oracle(oracle2_rows),
    }


def summarize_selector_predictions(
    records: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    max_candidates: int,
) -> dict[str, Any]:
    by_qid = {row_qid(row): row for row in prediction_rows}
    rows: list[dict[str, float]] = []
    added_gold = 0
    added_non_gold = 0
    for record in records:
        qid = row_qid(record)
        if qid not in by_qid:
            continue
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        example = featurize_selector_record(record, max_candidates=max_candidates, path_len=top_k)
        rank_set = set(range(min(int(top_k), len(candidates))))
        selected = []
        seen = set()
        for item in by_qid[qid].get("selected_indices") or []:
            index = int(item)
            if 0 <= index < len(candidates) and index not in seen:
                selected.append(index)
                seen.add(index)
            if len(selected) >= int(top_k):
                break
        metrics = support_metrics_for_indices(example, selected)
        metrics["selection_overlap"] = selection_overlap(selected, list(rank_set))
        rows.append(metrics)
        for index in set(selected) - rank_set:
            added_gold += int(is_gold(candidates[index]))
            added_non_gold += int(not is_gold(candidates[index]))
    summary = summarize_selector_metrics(rows)
    return {
        **summary,
        "added_gold": added_gold,
        "added_non_gold": added_non_gold,
        "non_gold_per_gold": round(float(added_non_gold) / max(1.0, float(added_gold)), 4),
    }


def kfold_ranges(row_count: int, folds: int) -> list[tuple[int, int]]:
    fold_size = row_count // int(folds)
    ranges = []
    for fold in range(int(folds)):
        start = fold * fold_size
        end = row_count if fold == int(folds) - 1 else start + fold_size
        ranges.append((start, end))
    return ranges


def run_crossfit(
    rows: Sequence[dict[str, Any]],
    *,
    selector_predictions: Sequence[dict[str, Any]],
    embedding_features: dict[str, tuple[Any, Any]],
    folds: int,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    negatives_per_query: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    min_edit_margin: float,
    verbose: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_predictions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    ranges = kfold_ranges(len(rows), folds)
    for fold, (start, end) in enumerate(ranges):
        if verbose:
            print(f"[cee] fold {fold}: build train/eval", file=sys.stderr, flush=True)
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
        if verbose:
            print(f"[cee] fold {fold}: fit {len(train_labels)} action rows", file=sys.stderr, flush=True)
        policy = LinearEditPolicy(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed + fold)
        fit_summary = policy.fit(train_features, train_labels)
        if verbose:
            print(f"[cee] fold {fold}: eval edit1/edit2", file=sys.stderr, flush=True)
        edit1_summary, edit1_predictions = evaluate_policy(
            policy,
            eval_rows,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            top_k=top_k,
            steps=1,
            embedding_features=embedding_features,
            min_edit_margin=min_edit_margin,
        )
        edit2_summary, edit2_predictions = evaluate_policy(
            policy,
            eval_rows,
            max_candidates=max_candidates,
            candidate_pool_size=candidate_pool_size,
            top_k=top_k,
            steps=2,
            embedding_features=embedding_features,
            min_edit_margin=min_edit_margin,
        )
        for row in edit1_predictions:
            row["variant"] = "learned_edit1"
            row["fold"] = fold
        for row in edit2_predictions:
            row["variant"] = "learned_edit2"
            row["fold"] = fold
        all_predictions.extend(edit1_predictions)
        all_predictions.extend(edit2_predictions)
        if verbose:
            print(f"[cee] fold {fold}: done", file=sys.stderr, flush=True)
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_examples": train_summary,
                "fit": fit_summary,
                "learned_edit1": edit1_summary,
                "learned_edit2": edit2_summary,
            }
        )

    learned_edit1 = [row for row in all_predictions if row["variant"] == "learned_edit1"]
    learned_edit2 = [row for row in all_predictions if row["variant"] == "learned_edit2"]

    def aggregate_prediction_rows(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
        metrics = summarize_selector_metrics([row["metrics"] for row in predictions])
        added_gold = 0
        added_non_gold = 0
        removed_gold = 0
        removed_non_gold = 0
        for row in predictions:
            record = next(item for item in rows if row_qid(item) == str(row["qid"]))
            candidates = list(record.get("candidates") or [])[: int(max_candidates)]
            for edit in row["edits"]:
                add_index = int(edit["add_index"])
                remove_index = int(edit["remove_index"])
                added_gold += int(is_gold(candidates[add_index]))
                added_non_gold += int(not is_gold(candidates[add_index]))
                removed_gold += int(is_gold(candidates[remove_index]))
                removed_non_gold += int(not is_gold(candidates[remove_index]))
        queries_with_edit = sum(1 for row in predictions if row["edits"])
        return {
            **metrics,
            "queries_with_edit": queries_with_edit,
            "stop_rate": round(1.0 - queries_with_edit / max(1.0, float(len(predictions))), 4),
            "added_gold": added_gold,
            "added_non_gold": added_non_gold,
            "removed_gold": removed_gold,
            "removed_non_gold": removed_non_gold,
            "non_gold_per_gold": round(float(added_non_gold) / max(1.0, float(added_gold)), 4),
        }

    if verbose:
        print("[cee] summarize rank/oracle", file=sys.stderr, flush=True)
    oracle_and_rank = summarize_rank_and_oracle(
        rows,
        top_k=top_k,
        max_candidates=max_candidates,
        candidate_pool_size=candidate_pool_size,
    )
    report = {
        "rows": len(rows),
        "folds": int(folds),
        "top_k": int(top_k),
        "max_candidates": int(max_candidates),
        "candidate_pool_size": int(candidate_pool_size),
        "rank_topk": oracle_and_rank["rank_topk"],
        "oracle_edit1": oracle_and_rank["oracle_edit1"],
        "oracle_edit2": oracle_and_rank["oracle_edit2"],
        "selector_v1": summarize_selector_predictions(rows, selector_predictions, top_k=top_k, max_candidates=max_candidates)
        if selector_predictions
        else {},
        "learned_edit1": aggregate_prediction_rows(learned_edit1),
        "learned_edit2": aggregate_prediction_rows(learned_edit2),
        "fold_reports": fold_reports,
    }
    report["gates"] = cee_pilot_gates(report)
    return report, all_predictions


def cee_pilot_gates(report: dict[str, Any]) -> dict[str, Any]:
    rank_support = float(report["rank_topk"].get("support_complete") or 0.0)
    selector_added_non_gold = float(report.get("selector_v1", {}).get("added_non_gold") or 0.0)
    gates: dict[str, Any] = {}
    for variant in ("learned_edit1", "learned_edit2"):
        item = report[variant]
        gates[variant] = {
            "support_complete_gain_ge_1pp": float(item.get("support_complete") or 0.0) >= rank_support + 0.01,
            "non_gold_per_gold_le_5": float(item.get("non_gold_per_gold") or 999.0) <= 5.0,
            "added_non_gold_50pct_below_v1": float(item.get("added_non_gold") or 0.0) <= 0.5 * selector_added_non_gold
            if selector_added_non_gold > 0
            else True,
        }
        gates[variant]["passed_count"] = sum(1 for value in gates[variant].values() if value is True)
    return gates


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    variants = ("rank_topk", "selector_v1", "oracle_edit1", "oracle_edit2", "learned_edit1", "learned_edit2")
    lines = [
        "# D-PathRAG Conservative Evidence Editing Pilot",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Folds: `{report['folds']}`",
        f"- Top-k: `{report['top_k']}`",
        f"- Candidate pool size: `{report['candidate_pool_size']}`",
        "",
        "## Summary",
        "",
        "| Variant | Support Recall | Support Complete | Selected Gold | Bridge Recall | Selection Overlap | Queries With Edit | Stop Rate | Added Gold | Added Non-Gold | Non-Gold/Gold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in variants:
        item = report.get(name) or {}
        lines.append(
            f"| {name} | {item.get('support_recall', 0.0):.4f} | {item.get('support_complete', 0.0):.4f} | "
            f"{item.get('selected_gold_count', 0.0):.4f} | {item.get('bridge_entity_recall', 0.0):.4f} | "
            f"{item.get('selection_overlap', 0.0):.4f} | {item.get('queries_with_edit', '')} | "
            f"{item.get('stop_rate', '')} | {item.get('added_gold', '')} | {item.get('added_non_gold', '')} | "
            f"{item.get('non_gold_per_gold', '')} |"
        )
    lines.extend(["", "## Pilot Gates", "", "| Variant | support +1pp | non-gold/gold <= 5 | added non-gold <= 50% v1 | Passed |", "|---|---:|---:|---:|---:|"])
    for variant, gates in report["gates"].items():
        lines.append(
            f"| {variant} | {gates['support_complete_gain_ge_1pp']} | {gates['non_gold_per_gold_le_5']} | "
            f"{gates['added_non_gold_50pct_below_v1']} | {gates['passed_count']}/3 |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--selector_predictions_jsonl", default="")
    parser.add_argument("--embedding_npz", default="")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--negatives_per_query", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min_edit_margin", type=float, default=0.0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--predictions_jsonl", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    selector_predictions = load_jsonl(args.selector_predictions_jsonl) if args.selector_predictions_jsonl else []
    embedding_features, embedding_feature_names = load_embedding_features(args.embedding_npz, limit=int(args.limit))
    report, predictions = run_crossfit(
        rows,
        selector_predictions=selector_predictions,
        embedding_features=embedding_features,
        folds=int(args.folds),
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        candidate_pool_size=int(args.candidate_pool_size),
        negatives_per_query=int(args.negatives_per_query),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        min_edit_margin=float(args.min_edit_margin),
        verbose=bool(args.verbose),
    )
    report["cache_jsonl"] = str(args.cache_jsonl)
    report["selector_predictions_jsonl"] = str(args.selector_predictions_jsonl)
    report["embedding_npz"] = str(args.embedding_npz)
    report["embedding_feature_names"] = [*FEATURE_NAMES, *embedding_feature_names]
    write_json(report, args.output_json)
    write_markdown(report, args.output_md)
    if args.predictions_jsonl:
        write_jsonl(predictions, args.predictions_jsonl)
    print(json.dumps(report["gates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
