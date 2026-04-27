#!/usr/bin/env python3
"""Shared helpers for CEE-v2 pairwise edit admission experiments."""

from __future__ import annotations

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

from dpathrag_analyze_hard_negatives import doc_features, is_gold, row_qid  # noqa: E402
from dpathrag_analyze_hard_negatives_v2 import classify_hard_negative, edit_objective, percentile  # noqa: E402
from dpathrag_cee_edit_policy import apply_action, enumerate_actions, load_embedding_features, make_example  # noqa: E402
from src.dpathrag.data import normalize_text  # noqa: E402
from src.dpathrag.selector_data import FEATURE_NAMES, selection_overlap, summarize_selector_metrics, support_metrics_for_indices  # noqa: E402


DEFAULT_OBJECTIVE = ("support_complete", "support_recall", "selected_gold_count", "bridge_entity_recall", "-added_non_gold")
PAIR_TYPES = ("lexical", "mixed", "graded")
HARD_NEGATIVE_TYPES = (
    "answer_string_distractor",
    "bridge_entity_distractor",
    "lexical_hard_negative",
    "semantic_hard_negative",
    "low_signal_deep_negative",
)


def finite_floats(values: Sequence[Any]) -> list[float]:
    output = []
    for value in values:
        item = float(value)
        output.append(item if math.isfinite(item) else 0.0)
    return output


def mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def stdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((float(value) - value_mean) ** 2 for value in values) / len(values))


def row_lookup(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row_qid(row): row for row in rows}


def doc_embedding_view(embedding_by_qid: dict[str, Any] | None) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for qid, value in (embedding_by_qid or {}).items():
        if isinstance(value, tuple) and value:
            output[str(qid)] = value[0]
        else:
            output[str(qid)] = value
    return output


def rank_indices_for(record: dict[str, Any], *, top_k: int, max_candidates: int) -> list[int]:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    return list(range(min(int(top_k), len(candidates))))


def replace_index(indices: Sequence[int], remove_index: int, add_index: int) -> list[int]:
    return [int(add_index) if int(index) == int(remove_index) else int(index) for index in indices]


def action_indices(base_indices: Sequence[int], edit: dict[str, Any]) -> list[int]:
    return replace_index(base_indices, int(edit["remove_index"]), int(edit["add_index"]))


def metrics_for(record: dict[str, Any], indices: Sequence[int], *, example: Any, added_non_gold: int = 0) -> dict[str, Any]:
    metrics = support_metrics_for_indices(example, indices)
    objective = edit_objective(metrics, added_non_gold=int(added_non_gold))
    return {"metrics": metrics, "objective": objective}


def edit_summary(
    record: dict[str, Any],
    remove_index: int,
    add_index: int,
    *,
    top_k: int,
    max_candidates: int,
    example: Any,
    embedding_by_qid: dict[str, Any] | None = None,
    embedding_feature_names: Sequence[str] | None = None,
    semantic_threshold: float = 0.0,
) -> dict[str, Any]:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    base = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
    edited = replace_index(base, int(remove_index), int(add_index))
    added_non_gold = int(not is_gold(candidates[int(add_index)]))
    added_gold = int(is_gold(candidates[int(add_index)]))
    removed_non_gold = int(not is_gold(candidates[int(remove_index)]))
    removed_gold = int(is_gold(candidates[int(remove_index)]))
    before = metrics_for(record, base, example=example, added_non_gold=0)
    after = metrics_for(record, edited, example=example, added_non_gold=added_non_gold)
    features = doc_features(
        record,
        candidates[int(add_index)],
        candidate_index=int(add_index),
        embedding_by_qid=doc_embedding_view(embedding_by_qid),
        embedding_feature_names=embedding_feature_names or [],
    )
    hard_type = "" if added_gold else classify_hard_negative(features, semantic_threshold=semantic_threshold)
    return {
        "remove_index": int(remove_index),
        "add_index": int(add_index),
        "remove_title": str(candidates[int(remove_index)].get("title") or ""),
        "add_title": str(candidates[int(add_index)].get("title") or ""),
        "remove_rank": int(remove_index) + 1,
        "add_rank": int(add_index) + 1,
        "added_gold": added_gold,
        "added_non_gold": added_non_gold,
        "removed_gold": removed_gold,
        "removed_non_gold": removed_non_gold,
        "before_metrics": before["metrics"],
        "after_metrics": after["metrics"],
        "before_objective": list(before["objective"]),
        "after_objective": list(after["objective"]),
        "beneficial": tuple(after["objective"]) > tuple(before["objective"]),
        "hard_negative_type": hard_type,
        "add_doc_features": features,
    }


def enumerate_edit_summaries(
    record: dict[str, Any],
    *,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    example: Any,
    embedding_by_qid: dict[str, Any] | None = None,
    embedding_feature_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    base = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
    usable_pool = min(len(candidates), int(candidate_pool_size))
    add_features = []
    for add_index in range(usable_pool):
        if add_index in set(base):
            continue
        if not is_gold(candidates[add_index]):
            add_features.append(
                doc_features(
                    record,
                    candidates[add_index],
                    candidate_index=add_index,
                    embedding_by_qid=doc_embedding_view(embedding_by_qid),
                    embedding_feature_names=embedding_feature_names or [],
                )
            )
    semantic_threshold = percentile([float(row.get("q_doc_cosine") or 0.0) for row in add_features], 0.75)
    edits = []
    for remove_index in base:
        for add_index in range(usable_pool):
            if add_index in set(base):
                continue
            edits.append(
                edit_summary(
                    record,
                    remove_index,
                    add_index,
                    top_k=top_k,
                    max_candidates=max_candidates,
                    example=example,
                    embedding_by_qid=embedding_by_qid,
                    embedding_feature_names=embedding_feature_names,
                    semantic_threshold=semantic_threshold,
                )
            )
    return edits


def normalize_title(value: Any) -> str:
    return normalize_text(value)


def edit_feature_vector(example: Any, base_indices: Sequence[int], edit: dict[str, Any]) -> list[float]:
    feature_dim = len(example.candidate_features[0])
    base = [int(index) for index in base_indices]
    add_index = int(edit["add_index"])
    remove_index = int(edit["remove_index"])
    add_features = finite_floats(example.candidate_features[add_index])
    remove_features = finite_floats(example.candidate_features[remove_index])
    base_mean = [
        sum(float(example.candidate_features[index][dim]) for index in base) / max(1, len(base))
        for dim in range(feature_dim)
    ]
    edited = replace_index(base, remove_index, add_index)
    edited_titles = [normalize_title(example.candidate_titles[index]) for index in edited if normalize_title(example.candidate_titles[index])]
    unique_titles = len(set(edited_titles))
    duplicate_title = 1.0 if unique_titles < len(edited_titles) else 0.0
    unique_title_rate = unique_titles / max(1, len(edited_titles))
    return (
        [1.0]
        + finite_floats(example.query_features)
        + base_mean
        + add_features
        + remove_features
        + [add - remove for add, remove in zip(add_features, remove_features)]
        + [
            float(add_index + 1),
            float(remove_index + 1),
            float((add_index + 1) - (remove_index + 1)),
            duplicate_title,
            unique_title_rate,
        ]
    )


def feature_names(base_feature_names: Sequence[str], query_dim: int, candidate_dim: int) -> list[str]:
    query_names = [f"query_{idx}" for idx in range(query_dim)]
    if len(base_feature_names) == candidate_dim:
        cand_names = list(base_feature_names)
    else:
        cand_names = [f"candidate_{idx}" for idx in range(candidate_dim)]
    return (
        ["bias"]
        + query_names
        + [f"s0_mean_{name}" for name in cand_names]
        + [f"add_{name}" for name in cand_names]
        + [f"remove_{name}" for name in cand_names]
        + [f"delta_{name}" for name in cand_names]
        + ["add_rank", "remove_rank", "rank_delta", "duplicate_title_after_edit", "unique_title_rate_after_edit"]
    )


def select_pair_rows(
    record: dict[str, Any],
    *,
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    embedding_by_qid: dict[str, Any] | None,
    embedding_feature_names: Sequence[str] | None,
    rng: random.Random,
    max_pairs_per_query: int = 6,
) -> list[dict[str, Any]]:
    example = make_example(
        record,
        max_candidates=max_candidates,
        top_k=top_k,
        embedding_features=embedding_by_qid or {},
    )
    edits = enumerate_edit_summaries(
        record,
        top_k=top_k,
        max_candidates=max_candidates,
        candidate_pool_size=candidate_pool_size,
        example=example,
        embedding_by_qid=embedding_by_qid,
        embedding_feature_names=embedding_feature_names,
    )
    positives = [edit for edit in edits if bool(edit["beneficial"])]
    if not positives:
        return []
    positives = sorted(positives, key=lambda item: tuple(item["after_objective"]), reverse=True)
    lexical = [edit for edit in edits if not edit["beneficial"] and edit["hard_negative_type"] == "lexical_hard_negative"]
    mixed = [edit for edit in edits if not edit["beneficial"] and edit["hard_negative_type"] in {"semantic_hard_negative", "answer_string_distractor", "bridge_entity_distractor", "low_signal_deep_negative"}]
    graded = sorted([edit for edit in edits if not edit["beneficial"]], key=lambda item: tuple(item["after_objective"]), reverse=True)

    base = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
    feature_example = example

    def make_pair(pos: dict[str, Any], neg: dict[str, Any], pair_type: str, same_remove_preferred: bool) -> dict[str, Any]:
        return {
            "qid": row_qid(record),
            "pair_type": pair_type,
            "same_remove": int(pos["remove_index"]) == int(neg["remove_index"]),
            "same_remove_preferred": bool(same_remove_preferred),
            "positive_edit": {key: value for key, value in pos.items() if key != "add_doc_features"},
            "negative_edit": {key: value for key, value in neg.items() if key != "add_doc_features"},
            "positive_features": edit_feature_vector(feature_example, base, pos),
            "negative_features": edit_feature_vector(feature_example, base, neg),
            "feature_names": feature_names(
                list(FEATURE_NAMES) + list(embedding_feature_names or []),
                query_dim=len(feature_example.query_features),
                candidate_dim=len(feature_example.candidate_features[0]),
            ),
        }

    pairs: list[dict[str, Any]] = []
    plan = [("lexical", lexical, 3), ("mixed", mixed, 2), ("graded", graded, 1)]
    for pair_type, negatives, quota in plan:
        if not negatives:
            continue
        type_pairs: list[dict[str, Any]] = []
        for pos in positives:
            same_remove = [neg for neg in negatives if int(neg["remove_index"]) == int(pos["remove_index"])]
            pool = same_remove or list(negatives)
            rng.shuffle(pool)
            for neg in pool:
                type_pairs.append(make_pair(pos, neg, pair_type, same_remove_preferred=bool(same_remove)))
        rng.shuffle(type_pairs)
        pairs.extend(type_pairs[: int(quota)])
        if len(pairs) >= int(max_pairs_per_query):
            return pairs[: int(max_pairs_per_query)]
    return pairs


def split_train_calib(rows: Sequence[dict[str, Any]], *, seed: int, calib_fraction: float = 0.2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    shuffled = list(rows)
    rng.shuffle(shuffled)
    calib_size = max(1, int(round(len(shuffled) * float(calib_fraction))))
    return shuffled[calib_size:], shuffled[:calib_size]


def write_jsonl(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_prediction_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = summarize_selector_metrics([row["metrics"] for row in rows])
    added_gold = sum(int(row.get("added_gold") or 0) for row in rows)
    added_non_gold = sum(int(row.get("added_non_gold") or 0) for row in rows)
    queries_with_edit = sum(1 for row in rows if row.get("edits"))
    return {
        **metrics,
        "queries_with_edit": queries_with_edit,
        "stop_rate": round(1.0 - queries_with_edit / max(1.0, float(len(rows))), 4),
        "added_gold": added_gold,
        "added_non_gold": added_non_gold,
        "non_gold_per_gold": round(float(added_non_gold) / max(1.0, float(added_gold)), 4),
    }


def platt_fit(scores: Sequence[float], labels: Sequence[int], *, steps: int = 400, lr: float = 0.05) -> tuple[float, float]:
    a = 1.0
    b = 0.0
    if not scores:
        return a, b
    for _ in range(int(steps)):
        grad_a = 0.0
        grad_b = 0.0
        for score, label in zip(scores, labels):
            z = max(-50.0, min(50.0, a * float(score) + b))
            pred = 1.0 / (1.0 + math.exp(-z))
            diff = pred - float(label)
            grad_a += diff * float(score)
            grad_b += diff
        scale = 1.0 / max(1, len(scores))
        a -= float(lr) * grad_a * scale
        b -= float(lr) * grad_b * scale
    return a, b


def platt_prob(score: float, a: float, b: float) -> float:
    z = max(-50.0, min(50.0, float(a) * float(score) + float(b)))
    return 1.0 / (1.0 + math.exp(-z))


class LinearPairwiseScorer:
    def __init__(self) -> None:
        self.mean: Any = None
        self.std: Any = None
        self.weight: Any = None

    def fit(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import numpy as np

        pos = np.asarray([row["positive_features"] for row in rows], dtype="float32")
        neg = np.asarray([row["negative_features"] for row in rows], dtype="float32")
        all_features = np.concatenate([pos, neg], axis=0)
        self.mean = all_features.mean(axis=0, keepdims=True)
        self.std = all_features.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        pos = (pos - self.mean) / self.std
        neg = (neg - self.mean) / self.std
        self.weight = (pos.mean(axis=0) - neg.mean(axis=0)).astype("float32")
        return {"model": "linear", "pairs": len(rows)}

    def score(self, features: Sequence[float]) -> float:
        import numpy as np

        if self.weight is None or self.mean is None or self.std is None:
            raise RuntimeError("LinearPairwiseScorer must be fit before scoring")
        array = np.asarray(features, dtype="float32")
        standardized = (array - self.mean.reshape(-1)) / self.std.reshape(-1)
        return float(standardized @ self.weight)


class MLPPairwiseScorer:
    def __init__(self, *, hidden_dim: int = 128, epochs: int = 20, seed: int = 17) -> None:
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.mean: Any = None
        self.std: Any = None
        self.model: Any = None

    def fit(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        pos = np.asarray([row["positive_features"] for row in rows], dtype="float32")
        neg = np.asarray([row["negative_features"] for row in rows], dtype="float32")
        all_features = np.concatenate([pos, neg], axis=0)
        self.mean = all_features.mean(axis=0, keepdims=True)
        self.std = all_features.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        pos = (pos - self.mean) / self.std
        neg = (neg - self.mean) / self.std
        pos_t = torch.tensor(pos, dtype=torch.float32)
        neg_t = torch.tensor(neg, dtype=torch.float32)
        dataset = TensorDataset(pos_t, neg_t)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(pos_t.shape[1], self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
        losses = []
        self.model.train()
        for _ in range(int(self.epochs)):
            total = 0.0
            count = 0
            for batch_pos, batch_neg in loader:
                pos_score = self.model(batch_pos).squeeze(-1)
                neg_score = self.model(batch_neg).squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach())
                count += 1
            losses.append(total / max(1, count))
        return {"model": "mlp", "pairs": len(rows), "final_loss": round(losses[-1], 6) if losses else 0.0}

    def score(self, features: Sequence[float]) -> float:
        import numpy as np
        import torch

        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("MLPPairwiseScorer must be fit before scoring")
        array = np.asarray(features, dtype="float32")
        standardized = (array - self.mean.reshape(-1)) / self.std.reshape(-1)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(torch.tensor(standardized, dtype=torch.float32).unsqueeze(0)).squeeze())
