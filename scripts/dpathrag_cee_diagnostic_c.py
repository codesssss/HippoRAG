#!/usr/bin/env python3
"""Diagnostic C for CEE-v2: same-remove hard-negative admission analysis."""

from __future__ import annotations

import argparse
import csv
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

from dpathrag_analyze_hard_negatives import load_jsonl, row_qid  # noqa: E402
from dpathrag_cee_edit_policy import load_embedding_features, make_example  # noqa: E402
from dpathrag_cee_pairwise_common import (  # noqa: E402
    enumerate_edit_summaries,
    mean,
    rank_indices_for,
    stdev,
)
from src.dpathrag.io import write_json  # noqa: E402


FEATURE_KEYS = (
    "rank",
    "retriever_score",
    "title_question_jaccard",
    "body_question_jaccard",
    "question_token_coverage",
    "q_doc_cosine",
    "answer_in_doc",
    "bridge_entity_in_doc",
    "log_doc_chars",
)


def summarize_feature_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(rows)}
    for key in FEATURE_KEYS:
        values = [float(row.get(key) or 0.0) for row in rows if key in row]
        if values:
            summary[key] = {"mean": round(mean(values), 6), "std": round(stdev(values), 6)}
    return summary


def prediction_lookup(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {row_qid(row): row for row in load_jsonl(path)}


def edit_key(edit: dict[str, Any]) -> tuple[int, int]:
    return int(edit["remove_index"]), int(edit["add_index"])


def analyze(
    rows: Sequence[dict[str, Any]],
    *,
    shallow_predictions: Sequence[dict[str, Any]],
    embedding_features: dict[str, Any],
    embedding_feature_names: Sequence[str],
    top_k: int,
    max_candidates: int,
    candidate_pool_size: int,
    max_pairs_csv: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_by_qid = {row_qid(row): row for row in shallow_predictions}
    pairs_csv: list[dict[str, Any]] = []
    same_remove_pairs = []
    positive_features = []
    lexical_features = []
    shallow_fp = []
    shallow_fn = []
    beneficial_total = 0
    lexical_negative_total = 0
    for record in rows:
        qid = row_qid(record)
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
        beneficial = [edit for edit in edits if edit["beneficial"]]
        lexical = [edit for edit in edits if not edit["beneficial"] and edit["hard_negative_type"] == "lexical_hard_negative"]
        beneficial_total += len(beneficial)
        lexical_negative_total += len(lexical)
        positive_features.extend(edit.get("add_doc_features", {}) for edit in beneficial)
        lexical_features.extend(edit.get("add_doc_features", {}) for edit in lexical)
        for pos in beneficial:
            same = [neg for neg in lexical if int(neg["remove_index"]) == int(pos["remove_index"])]
            if not same:
                continue
            same_remove_pairs.append((qid, pos, same[0]))
            if len(pairs_csv) < int(max_pairs_csv):
                neg = same[0]
                pairs_csv.append(
                    {
                        "qid": qid,
                        "remove_index": pos["remove_index"],
                        "remove_rank": pos["remove_rank"],
                        "positive_add_index": pos["add_index"],
                        "positive_add_rank": pos["add_rank"],
                        "positive_add_title": pos["add_title"],
                        "negative_add_index": neg["add_index"],
                        "negative_add_rank": neg["add_rank"],
                        "negative_add_title": neg["add_title"],
                        "negative_type": neg["hard_negative_type"],
                        "positive_after_objective": json.dumps(pos["after_objective"]),
                        "negative_after_objective": json.dumps(neg["after_objective"]),
                    }
                )

        pred = pred_by_qid.get(qid)
        if pred is None:
            continue
        pred_edits = [edit for edit in pred.get("edits") or []]
        beneficial_keys = {edit_key(edit) for edit in beneficial}
        pred_keys = {edit_key(edit) for edit in pred_edits}
        for edit in pred_edits:
            if edit_key(edit) not in beneficial_keys:
                shallow_fp.append({"qid": qid, **edit})
        if beneficial and not (pred_keys & beneficial_keys):
            shallow_fn.append({"qid": qid, "beneficial_edits": len(beneficial), "rank_complete": float(edits[0]["before_metrics"].get("support_complete") or 0.0) >= 1.0})

    payload = {
        "rows": len(rows),
        "top_k": top_k,
        "candidate_pool_size": candidate_pool_size,
        "beneficial_edits": beneficial_total,
        "lexical_negative_edits": lexical_negative_total,
        "same_remove_beneficial_vs_lexical_pairs": len(same_remove_pairs),
        "queries_with_same_remove_pairs": len({qid for qid, _, _ in same_remove_pairs}),
        "shallow_false_positive_edits": len(shallow_fp),
        "shallow_false_positive_queries": len({row["qid"] for row in shallow_fp}),
        "shallow_false_negative_queries": len(shallow_fn),
        "positive_add_features": summarize_feature_rows(positive_features),
        "lexical_negative_add_features": summarize_feature_rows(lexical_features),
        "feature_mean_delta_positive_minus_lexical": {},
        "sample_shallow_false_positives": shallow_fp[:20],
        "sample_shallow_false_negatives": shallow_fn[:20],
    }
    for key in FEATURE_KEYS:
        left = payload["positive_add_features"].get(key, {}).get("mean")
        right = payload["lexical_negative_add_features"].get(key, {}).get("mean")
        if left is not None and right is not None:
            payload["feature_mean_delta_positive_minus_lexical"][key] = round(float(left) - float(right), 6)
    return payload, pairs_csv


def audit_daec(root: str | Path) -> dict[str, Any]:
    terms = ("daec", "demand", "supply", "binding", "compatibility", "residual", "type")
    suffixes = {".py", ".json", ".jsonl", ".md", ".txt", ".sh"}
    matches = []
    root = Path(root)
    for current, dirs, files in os.walk(root):
        if any(part in {".git", ".venv-hipporag", "__pycache__", "outputs", "node_modules"} for part in Path(current).parts):
            dirs[:] = []
            continue
        for name in files:
            path = Path(current) / name
            rel = str(path.relative_to(root))
            lower = rel.lower()
            if path.suffix.lower() not in suffixes:
                continue
            score = sum(1 for term in terms if term in lower)
            snippet = ""
            if score == 0 and path.stat().st_size < 2_000_000:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    text = ""
                score = sum(1 for term in terms if term in text)
                if score:
                    pos = min((text.find(term) for term in terms if text.find(term) >= 0), default=0)
                    snippet = text[max(0, pos - 80) : pos + 160].replace("\n", " ")
            if score:
                matches.append({"path": rel, "score": score, "snippet": snippet[:240]})
    matches = sorted(matches, key=lambda item: (-int(item["score"]), item["path"]))[:80]
    joinable = [row for row in matches if row["path"].endswith((".json", ".jsonl")) and any(term in row["path"].lower() for term in ("daec", "demand", "binding", "compat"))]
    return {
        "root": str(root),
        "matches": matches,
        "joinable_cache_candidates": joinable,
        "status": "joinable_candidates_found" if joinable else "no_direct_joinable_cache_found",
        "decision": "Use CEE-pairwise v0 first; add DAEC features only after a fold-safe per-query/per-doc cache is confirmed."
        if not joinable
        else "Inspect joinable candidates before wiring DAEC features into CEE-pairwise v1.",
    }


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qid",
        "remove_index",
        "remove_rank",
        "positive_add_index",
        "positive_add_rank",
        "positive_add_title",
        "negative_add_index",
        "negative_add_rank",
        "negative_add_title",
        "negative_type",
        "positive_after_objective",
        "negative_after_objective",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(payload: dict[str, Any], audit: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# CEE Diagnostic C",
        "",
        "## Same-Remove Admission Signal",
        "",
        f"- Rows: `{payload['rows']}`",
        f"- Beneficial edits: `{payload['beneficial_edits']}`",
        f"- Lexical negative edits: `{payload['lexical_negative_edits']}`",
        f"- Same-remove beneficial-vs-lexical pairs: `{payload['same_remove_beneficial_vs_lexical_pairs']}`",
        f"- Queries with same-remove pairs: `{payload['queries_with_same_remove_pairs']}`",
        "",
        "## Current Shallow CEE Error Surface",
        "",
        f"- False-positive edited queries/edits: `{payload['shallow_false_positive_queries']}` / `{payload['shallow_false_positive_edits']}`",
        f"- False-negative queries with beneficial edit missed: `{payload['shallow_false_negative_queries']}`",
        "",
        "## Positive Add vs Lexical Negative Add Feature Means",
        "",
        "| Feature | Positive Mean | Lexical Negative Mean | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in FEATURE_KEYS:
        pos = payload["positive_add_features"].get(key, {}).get("mean", "")
        neg = payload["lexical_negative_add_features"].get(key, {}).get("mean", "")
        delta = payload["feature_mean_delta_positive_minus_lexical"].get(key, "")
        lines.append(f"| `{key}` | {pos} | {neg} | {delta} |")
    lines.extend(
        [
            "",
            "## DAEC Feature Audit",
            "",
            f"- Status: `{audit['status']}`",
            f"- Decision: {audit['decision']}",
            "",
            "| Candidate Artifact | Score |",
            "|---|---:|",
        ]
    )
    for row in audit["matches"][:20]:
        lines.append(f"| `{row['path']}` | {row['score']} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--embedding_npz", default="data/dpathrag/cache/selector_embedding_proprag_local1000_rp64.npz")
    parser.add_argument("--shallow_predictions_jsonl", default="reports/dpathrag/cee_edit_policy_proprag_kfold1000_margin20.predictions.jsonl")
    parser.add_argument("--output_json", default="reports/dpathrag/cee_diagnostic_c.json")
    parser.add_argument("--output_md", default="reports/dpathrag/cee_diagnostic_c.md")
    parser.add_argument("--output_pairs_csv", default="reports/dpathrag/cee_diagnostic_c_pairs.csv")
    parser.add_argument("--daec_audit_md", default="reports/dpathrag/daec_feature_audit.md")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--max_pairs_csv", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    shallow_predictions = load_jsonl(args.shallow_predictions_jsonl) if args.shallow_predictions_jsonl else []
    embedding_features, embedding_feature_names = load_embedding_features(args.embedding_npz, limit=int(args.limit))
    payload, pairs = analyze(
        rows,
        shallow_predictions=shallow_predictions,
        embedding_features=embedding_features,
        embedding_feature_names=embedding_feature_names,
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        candidate_pool_size=int(args.candidate_pool_size),
        max_pairs_csv=int(args.max_pairs_csv),
    )
    audit = audit_daec(_PROJECT_ROOT)
    write_json({"diagnostic_c": payload, "daec_audit": audit}, args.output_json)
    write_csv(pairs, args.output_pairs_csv)
    write_markdown(payload, audit, args.output_md)
    write_markdown(payload, audit, args.daec_audit_md)


if __name__ == "__main__":
    main()
