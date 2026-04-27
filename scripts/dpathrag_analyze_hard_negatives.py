#!/usr/bin/env python3
"""Characterize selector-added hard negatives in D-PathRAG predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.data import evidence_path_entities, normalize_text
from src.dpathrag.io import write_json


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= int(limit):
                    break
    return rows


def row_qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("query_idx") or "")


def tokens(value: Any) -> set[str]:
    text = normalize_text(value)
    return {token for token in re.findall(r"[a-z0-9]+", text) if token}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def coverage(needed: set[str], observed: set[str]) -> float:
    if not needed:
        return 0.0
    return len(needed & observed) / len(needed)


def split_title_body(candidate: dict[str, Any]) -> tuple[str, str]:
    title = str(candidate.get("title") or "")
    text = str(candidate.get("text") or "")
    if "\n" in text:
        text_title, body = text.split("\n", 1)
        return title or text_title, body
    return title, text


def is_gold(candidate: dict[str, Any]) -> bool:
    return int(candidate.get("gold_support") or 0) == 1


def bridge_entities(row: dict[str, Any]) -> list[str]:
    entities = evidence_path_entities(row)
    if len(entities) > 2:
        return [str(entity) for entity in entities[1:-1]]
    return []


def load_embedding_payload(path: str | Path, *, limit: int = 0) -> tuple[dict[str, Any], list[str]]:
    if not path:
        return {}, []
    import numpy as np

    payload = np.load(path, allow_pickle=True)
    qids = [str(qid) for qid in payload["qids"].tolist()]
    features = payload["candidate_features"]
    feature_names = [str(name) for name in payload["feature_names"].tolist()] if "feature_names" in payload.files else []
    if limit > 0:
        qids = qids[: int(limit)]
        features = features[: int(limit)]
    return {qid: features[idx] for idx, qid in enumerate(qids)}, feature_names


def embedding_feature_value(
    embedding_by_qid: dict[str, Any],
    feature_names: Sequence[str],
    *,
    qid: str,
    candidate_index: int,
) -> float | None:
    if not embedding_by_qid or not feature_names:
        return None
    cosine_indices = [idx for idx, name in enumerate(feature_names) if "cosine" in str(name).lower()]
    if not cosine_indices:
        return None
    rows = embedding_by_qid.get(str(qid))
    if rows is None or int(candidate_index) >= len(rows):
        return None
    return float(rows[int(candidate_index)][cosine_indices[0]])


def doc_features(
    row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_index: int,
    embedding_by_qid: dict[str, Any] | None = None,
    embedding_feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    embedding_by_qid = embedding_by_qid or {}
    embedding_feature_names = list(embedding_feature_names or [])
    question_tokens = tokens(row.get("question"))
    title, body = split_title_body(candidate)
    title_tokens = tokens(title)
    body_tokens = tokens(body)
    blob_norm = normalize_text(f"{title} {body}")
    answer_norm = normalize_text(row.get("answer"))
    bridges = {normalize_text(entity) for entity in bridge_entities(row) if normalize_text(entity)}
    q_doc_cosine = embedding_feature_value(
        embedding_by_qid,
        embedding_feature_names,
        qid=row_qid(row),
        candidate_index=int(candidate_index),
    )
    result = {
        "rank": float(candidate.get("rank") or (int(candidate_index) + 1)),
        "retriever_score": float(candidate.get("retriever_score") or 0.0),
        "title_question_jaccard": jaccard(title_tokens, question_tokens),
        "body_question_jaccard": jaccard(body_tokens, question_tokens),
        "question_token_coverage": coverage(question_tokens, title_tokens | body_tokens),
        "answer_in_doc": 1.0 if answer_norm and answer_norm in blob_norm else 0.0,
        "bridge_entity_in_doc": 1.0 if bridges and any(entity in blob_norm for entity in bridges) else 0.0,
        "doc_chars": float(len(str(candidate.get("text") or ""))),
        "log_doc_chars": math.log1p(len(str(candidate.get("text") or ""))),
    }
    if q_doc_cosine is not None:
        result["q_doc_cosine"] = float(q_doc_cosine)
    return result


def categorize_candidate(index: int, *, rank_set: set[int], selector_set: set[int], gold: bool) -> str | None:
    in_rank = int(index) in rank_set
    in_selector = int(index) in selector_set
    if in_selector and not in_rank and gold:
        return "selector_added_gold"
    if in_selector and not in_rank and not gold:
        return "selector_added_non_gold"
    if in_selector and in_rank and not gold:
        return "rank_retained_non_gold"
    if in_rank and not in_selector and not gold:
        return "rank_removed_non_gold"
    return None


def summarize_category(rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "features": {}}
    keys = sorted({key for row in rows for key in row.keys()})
    features: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        features[key] = {
            "mean": round(sum(values) / len(values), 6),
            "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        }
    return {"count": len(rows), "features": features}


def compare_categories(summary: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    left_features = summary.get(left, {}).get("features", {})
    right_features = summary.get(right, {}).get("features", {})
    shared = sorted(set(left_features) & set(right_features))
    return {
        feature: round(float(left_features[feature]["mean"]) - float(right_features[feature]["mean"]), 6)
        for feature in shared
    }


def analyze_hard_negatives(
    cache_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    max_candidates: int,
    embedding_by_qid: dict[str, Any] | None = None,
    embedding_feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    cache_by_qid = {row_qid(row): row for row in cache_rows}
    buckets: dict[str, list[dict[str, float]]] = {
        "selector_added_non_gold": [],
        "rank_retained_non_gold": [],
        "rank_removed_non_gold": [],
        "selector_added_gold": [],
    }
    query_counts = {key: 0 for key in buckets}
    for prediction in prediction_rows:
        qid = row_qid(prediction)
        if qid not in cache_by_qid:
            raise KeyError(f"Missing cache row for qid={qid}")
        row = cache_by_qid[qid]
        candidates = list(row.get("candidates") or [])[: int(max_candidates)]
        rank_indices = list(range(min(int(top_k), len(candidates))))
        selector_indices = [
            int(index)
            for index in list(prediction.get("selected_indices") or [])[: int(top_k)]
            if 0 <= int(index) < len(candidates)
        ]
        rank_set = set(rank_indices)
        selector_set = set(selector_indices)
        query_hit = {key: False for key in buckets}
        for index in sorted(rank_set | selector_set):
            category = categorize_candidate(index, rank_set=rank_set, selector_set=selector_set, gold=is_gold(candidates[index]))
            if category is None:
                continue
            buckets[category].append(
                doc_features(
                    row,
                    candidates[index],
                    candidate_index=index,
                    embedding_by_qid=embedding_by_qid,
                    embedding_feature_names=embedding_feature_names,
                )
            )
            query_hit[category] = True
        for key, hit in query_hit.items():
            if hit:
                query_counts[key] += 1
    summary = {key: summarize_category(rows) for key, rows in buckets.items()}
    comparisons = {
        "selector_added_non_gold_minus_rank_retained_non_gold": compare_categories(
            summary, "selector_added_non_gold", "rank_retained_non_gold"
        ),
        "selector_added_non_gold_minus_rank_removed_non_gold": compare_categories(
            summary, "selector_added_non_gold", "rank_removed_non_gold"
        ),
        "selector_added_gold_minus_selector_added_non_gold": compare_categories(
            summary, "selector_added_gold", "selector_added_non_gold"
        ),
    }
    return {
        "rows": len(prediction_rows),
        "top_k": int(top_k),
        "max_candidates": int(max_candidates),
        "query_counts": query_counts,
        "categories": summary,
        "comparisons": comparisons,
    }


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    feature_order = [
        "rank",
        "retriever_score",
        "title_question_jaccard",
        "body_question_jaccard",
        "question_token_coverage",
        "q_doc_cosine",
        "answer_in_doc",
        "bridge_entity_in_doc",
        "log_doc_chars",
    ]
    lines = [
        "# D-PathRAG Hard-Negative Characterization",
        "",
        f"- Rows: `{payload['rows']}`",
        f"- Top-k: `{payload['top_k']}`",
        f"- Max candidates: `{payload['max_candidates']}`",
        "",
        "## Category Means",
        "",
        "| Category | Docs | Queries | "
        + " | ".join(feature_order)
        + " |",
        "|---|---:|---:|" + "---:|" * len(feature_order),
    ]
    for category, summary in payload["categories"].items():
        values = []
        for feature in feature_order:
            item = summary.get("features", {}).get(feature)
            values.append(f"{float(item['mean']):.4f}" if item else "")
        lines.append(
            f"| {category} | {summary['count']} | {payload['query_counts'].get(category, 0)} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(["", "## Mean Differences", ""])
    for name, comparison in payload["comparisons"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Feature | Mean Delta |")
        lines.append("|---|---:|")
        for feature in feature_order:
            if feature in comparison:
                lines.append(f"| {feature} | {float(comparison[feature]):+.4f} |")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--embedding_npz", default="")
    parser.add_argument("--cache_limit", type=int, default=1000)
    parser.add_argument("--prediction_limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    cache_rows = load_jsonl(args.cache_jsonl, limit=int(args.cache_limit))
    prediction_rows = load_jsonl(args.predictions_jsonl, limit=int(args.prediction_limit))
    embedding_by_qid, embedding_feature_names = load_embedding_payload(args.embedding_npz, limit=int(args.cache_limit))
    output = analyze_hard_negatives(
        cache_rows,
        prediction_rows,
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        embedding_by_qid=embedding_by_qid,
        embedding_feature_names=embedding_feature_names,
    )
    output["cache_jsonl"] = str(args.cache_jsonl)
    output["predictions_jsonl"] = str(args.predictions_jsonl)
    output["embedding_npz"] = str(args.embedding_npz)
    output["embedding_feature_names"] = list(embedding_feature_names)
    write_json(output, args.output_json)
    write_markdown(output, args.output_md)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
