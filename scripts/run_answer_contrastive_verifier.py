#!/usr/bin/env python3
"""Answer-contrastive verifier over cached CAPS Day-2 candidates.

This diagnostic asks whether the CAPS Day-2 residual failure is recoverable by
a lightweight supervised answer-level ranker over existing candidate/proof
features. It deliberately does not call an LLM and does not fine-tune a neural
encoder in v0.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from caps_day0_nli_sanity import auc_score  # noqa: E402
from caps_day1_gates import write_markdown  # noqa: E402
from caps_day2_proof_separability import (  # noqa: E402
    NliScorer,
    build_nli_work,
    evidence_answer_positions,
    reciprocal_rank,
    score_tasks,
)
from dpathrag_analyze_hard_negatives import load_jsonl, row_qid  # noqa: E402
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import gold_answers, normalize_answer  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
DATE_WORD_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)

SOURCE_FEATURES = [
    "llm_candidate_list",
    "reader_prop_top5",
    "reader_union_top5",
    "reader_single",
    "reader_pair",
    "title",
    "date",
    "number",
    "capitalized_span",
    "connected_entity_span",
    "yes_no_prior",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def logit(value: float) -> float:
    prob = clamp(value, 1e-6, 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def token_set(value: Any) -> set[str]:
    return set(TOKEN_RE.findall(normalize_answer(value)))


def expected_answer_kind(question: str) -> str:
    q = normalize_answer(question)
    if q.startswith("when") or "what year" in q or "which year" in q or "date" in q:
        return "date"
    if "how many" in q or "number" in q or "population" in q:
        return "number"
    if q.startswith("are ") or q.startswith("do ") or q.startswith("does ") or q.startswith("did ") or q.startswith("is "):
        return "yes_no"
    if q.startswith("where") or "place of birth" in q or "birthplace" in q:
        return "place"
    if q.startswith("who"):
        return "person"
    return "entity"


def source_flags(sources: Sequence[str]) -> dict[str, float]:
    lowered = [str(source or "").lower() for source in sources]
    joined = " ".join(lowered)
    return {
        "llm_candidate_list": float("llm_candidate_list" in lowered),
        "reader_prop_top5": float("reader_prop_top5" in lowered),
        "reader_union_top5": float("reader_union_top5" in lowered),
        "reader_single": float(any(source.startswith("reader_single") for source in lowered)),
        "reader_pair": float(any(source.startswith("reader_pair") for source in lowered)),
        "title": float("title" in lowered),
        "date": float("date" in lowered),
        "number": float("number" in lowered),
        "capitalized_span": float("capitalized_span" in lowered),
        "connected_entity_span": float("connected_entity_span" in lowered),
        "yes_no_prior": float("yes_no_prior" in lowered),
        "empty_source": float(any(source == "" for source in lowered)),
        "reader_any": float("reader" in joined),
    }


def feature_names() -> list[str]:
    names = [
        "proof_score",
        "proof_logit",
        "candidate_rank",
        "candidate_inv_rank",
        "candidate_score",
        "candidate_max_score",
        "source_count",
        "obligation_count",
        "doc_count",
        "candidate_char_len",
        "candidate_token_len",
        "candidate_has_digit",
        "candidate_has_year",
        "candidate_has_date_word",
        "candidate_is_yes",
        "candidate_is_no",
        "answer_kind_date",
        "answer_kind_number",
        "answer_kind_yes_no",
        "answer_kind_person",
        "answer_kind_place",
        "candidate_matches_date_kind",
        "candidate_matches_number_kind",
        "candidate_matches_yes_no_kind",
        "question_candidate_token_overlap",
        "candidate_in_question",
        "substituted",
        "answer_conditioned_available",
    ]
    names.extend(f"source_{name}" for name in SOURCE_FEATURES)
    names.extend(["source_empty", "source_reader_any"])
    return names


def candidate_feature_vector(row: dict[str, Any]) -> list[float]:
    candidate_text = str(row.get("candidate_text") or "")
    question = str(row.get("question") or "")
    candidate_tokens = token_set(candidate_text)
    question_tokens = token_set(question)
    q_norm = normalize_answer(question)
    c_norm = normalize_answer(candidate_text)
    kind = expected_answer_kind(question)
    sources = [str(source) for source in row.get("sources") or []]
    flags = source_flags(sources)
    has_digit = bool(re.search(r"\d", candidate_text))
    has_year = bool(YEAR_RE.search(candidate_text))
    has_date_word = bool(DATE_WORD_RE.search(candidate_text))
    is_yes = c_norm == "yes"
    is_no = c_norm == "no"
    overlap = len(candidate_tokens & question_tokens) / max(1.0, float(len(candidate_tokens | question_tokens)))
    base = [
        safe_float(row.get("proof_score")),
        logit(safe_float(row.get("proof_score"))),
        safe_float(row.get("candidate_index")),
        1.0 / (1.0 + safe_float(row.get("candidate_index"))),
        safe_float(row.get("candidate_score")) / 100.0,
        safe_float(row.get("candidate_max_score")) / 100.0,
        safe_float(row.get("source_count")),
        safe_float(row.get("obligation_count")),
        safe_float(row.get("doc_count")),
        float(len(candidate_text)),
        float(len(candidate_tokens)),
        float(has_digit),
        float(has_year),
        float(has_date_word),
        float(is_yes),
        float(is_no),
        float(kind == "date"),
        float(kind == "number"),
        float(kind == "yes_no"),
        float(kind == "person"),
        float(kind == "place"),
        float(kind == "date" and (has_year or has_date_word)),
        float(kind == "number" and has_digit),
        float(kind == "yes_no" and (is_yes or is_no)),
        overlap,
        float(bool(c_norm and c_norm in q_norm)),
        float(bool(row.get("substituted"))),
        float(bool(row.get("answer_conditioned_available"))),
    ]
    base.extend(flags[name] for name in SOURCE_FEATURES)
    base.extend([flags["empty_source"], flags["reader_any"]])
    return base


def candidate_row_id(row: dict[str, Any]) -> str:
    return f"{row['qid']}::{int(row['candidate_index'])}"


def load_day1_5_candidates(path: str | Path, *, answer_cap: int) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        qid = str(row.get("qid") or "")
        candidates = list(row.get("top_candidates") or [])[: int(answer_cap)]
        if qid:
            by_qid[qid] = candidates
    return by_qid


def load_dev_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    limit = max(int(args.dev_end), int(args.max_rows))
    prop_rows = load_jsonl(args.proprag_cache_jsonl, limit=limit)[int(args.dev_start) : int(args.dev_end)]
    dense_by_qid = {row_qid(row): row for row in load_jsonl(args.dense_cache_jsonl, limit=limit)}
    candidates_by_qid = load_day1_5_candidates(args.day1_5_json, answer_cap=int(args.answer_cap))
    missing = [row_qid(row) for row in prop_rows if row_qid(row) not in candidates_by_qid]
    if missing:
        raise ValueError(f"Missing Day-1.5 cached candidates for {len(missing)} qids; first missing qid={missing[0]}")
    return prop_rows, dense_by_qid, candidates_by_qid


def make_candidate_rows(prop_rows: Sequence[dict[str, Any]], scored: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records_by_qid = {row_qid(record): record for record in prop_rows}
    rows: list[dict[str, Any]] = []
    for scored_row in scored:
        qid = str(scored_row["qid"])
        record = records_by_qid[qid]
        candidate = dict(scored_row.get("candidate") or {})
        text = str(candidate.get("text") or "")
        norm = normalize_answer(candidate.get("normalized") or text)
        gold_norms = {normalize_answer(answer) for answer in gold_answers(record) if normalize_answer(answer)}
        sources = [str(source) for source in candidate.get("sources") or []]
        rows.append(
            {
                "row_id": f"{qid}::{int(scored_row['candidate_index'])}",
                "qid": qid,
                "type": str(record.get("type") or "unknown"),
                "question": str(record.get("question") or ""),
                "gold_answers": gold_answers(record),
                "candidate_index": int(scored_row["candidate_index"]),
                "candidate_text": text,
                "candidate_norm": norm,
                "label": int(norm in gold_norms),
                "proof_score": round(safe_float(scored_row.get("proof_score")), 8),
                "candidate_score": round(safe_float(candidate.get("score")), 8),
                "candidate_max_score": round(safe_float(candidate.get("max_score")), 8),
                "source_count": int(candidate.get("source_count") or len(sources)),
                "sources": sources,
                "substituted": bool(scored_row.get("substituted")),
                "obligation_count": len(scored_row.get("obligations") or []),
                "doc_count": int(scored_row.get("doc_count") or 0),
                "answer_conditioned_available": bool(evidence_answer_positions(record)),
            }
        )
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qid[str(row["qid"])].append(row)
    for group in by_qid.values():
        gold_present = any(bool(row["label"]) for row in group)
        for row in group:
            row["gold_present_query"] = bool(gold_present)
    return sorted(rows, key=lambda row: (str(row["qid"]), int(row["candidate_index"])))


def build_candidate_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prop_rows, dense_by_qid, candidates_by_qid = load_dev_inputs(args)
    pairs, _, tasks = build_nli_work(prop_rows, dense_by_qid, candidates_by_qid, args=args)
    scorer = NliScorer(
        str(args.nli_model),
        batch_size=int(args.nli_batch_size),
        max_length=int(args.nli_max_length),
        entailment_class_index=int(args.entailment_class_index),
        local_files_only=not bool(args.allow_download),
    )
    pair_scores = scorer.score_pairs(pairs)
    scored = score_tasks(tasks, pair_scores)
    rows = make_candidate_rows(prop_rows, scored)
    metadata = {
        "candidate_rows": len(rows),
        "queries": len(prop_rows),
        "nli_pairs": len(pairs),
        "tasks": len(tasks),
        "source": str(args.day1_5_json),
    }
    return rows, metadata


def load_or_build_candidate_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(args.candidate_rows_jsonl)
    if path.exists() and not bool(args.force_rebuild_candidate_rows):
        rows = load_jsonl(path)
        return rows, {"candidate_rows": len(rows), "loaded_from": str(path), "reused": True}
    rows, metadata = build_candidate_rows(args)
    write_jsonl(rows, path)
    return rows, {**metadata, "written_to": str(path), "reused": False}


def by_qid(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["qid"])].append(row)
    return dict(groups)


def summarize_by_type(rows: Sequence[dict[str, Any]], score_key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("type") or "unknown")].append(row)
    return {key: summarize_ranking(value, score_key) for key, value in sorted(buckets.items())}


def summarize_ranking(rows: Sequence[dict[str, Any]], score_key: str) -> dict[str, Any]:
    groups = by_qid(rows)
    qid_count = len(groups)
    present_groups = [group for group in groups.values() if any(int(row.get("label") or 0) for row in group)]
    present_count = len(present_groups)
    top1_all = 0.0
    top3_all = 0.0
    top1_cond = 0.0
    top3_cond = 0.0
    mrr_cond = 0.0
    gold_scores: list[float] = []
    best_wrong_scores: list[float] = []
    paired_gold_wins = 0.0
    candidate_scores: list[float] = []
    candidate_labels: list[int] = []
    for group in groups.values():
        ranked = sorted(group, key=lambda row: (-safe_float(row.get(score_key)), int(row.get("candidate_index") or 0)))
        is_top1_gold = bool(ranked and int(ranked[0].get("label") or 0))
        is_top3_gold = any(int(row.get("label") or 0) for row in ranked[:3])
        top1_all += float(is_top1_gold)
        top3_all += float(is_top3_gold)
        gold_rows = [row for row in ranked if int(row.get("label") or 0)]
        wrong_rows = [row for row in ranked if not int(row.get("label") or 0)]
        if gold_rows:
            gold_ids = {candidate_row_id(row) for row in gold_rows}
            gold_rank = None
            for idx, row in enumerate(ranked, start=1):
                if candidate_row_id(row) in gold_ids:
                    gold_rank = idx
                    break
            top1_cond += float(gold_rank == 1)
            top3_cond += float(gold_rank is not None and gold_rank <= 3)
            mrr_cond += reciprocal_rank(gold_rank)
            gold_score = max(safe_float(row.get(score_key)) for row in gold_rows)
            gold_scores.append(gold_score)
            if wrong_rows:
                wrong_score = max(safe_float(row.get(score_key)) for row in wrong_rows)
                best_wrong_scores.append(wrong_score)
                paired_gold_wins += float(gold_score > wrong_score)
        for row in ranked:
            candidate_scores.append(safe_float(row.get(score_key)))
            candidate_labels.append(int(row.get("label") or 0))
    summary = {
        "queries": qid_count,
        "candidate_rows": len(rows),
        "gold_present_queries": present_count,
        "candidate_recall_at_20": round(float(present_count) / max(1.0, float(qid_count)), 6),
        "top1_accuracy_all": round(top1_all / max(1.0, float(qid_count)), 6),
        "top3_accuracy_all": round(top3_all / max(1.0, float(qid_count)), 6),
        "top1_accuracy_cond_gold_present": round(top1_cond / max(1.0, float(present_count)), 6),
        "top3_accuracy_cond_gold_present": round(top3_cond / max(1.0, float(present_count)), 6),
        "mrr_cond_gold_present": round(mrr_cond / max(1.0, float(present_count)), 6),
        "gold_vs_best_wrong_auc": round(auc_score(gold_scores, best_wrong_scores), 6),
        "paired_gold_win_rate": round(paired_gold_wins / max(1.0, float(len(best_wrong_scores))), 6),
        "mean_gold_score": round(sum(gold_scores) / max(1.0, float(len(gold_scores))), 6),
        "mean_best_wrong_score": round(sum(best_wrong_scores) / max(1.0, float(len(best_wrong_scores))), 6),
    }
    if len(set(candidate_labels)) == 2:
        from sklearn.metrics import roc_auc_score

        summary["candidate_level_auc"] = round(float(roc_auc_score(candidate_labels, candidate_scores)), 6)
    else:
        summary["candidate_level_auc"] = 0.0
    return summary


def bootstrap_gold_vs_best_wrong_ci(
    rows: Sequence[dict[str, Any]],
    score_key: str,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    pairs: list[tuple[float, float]] = []
    for group in by_qid(rows).values():
        gold_rows = [row for row in group if int(row.get("label") or 0)]
        wrong_rows = [row for row in group if not int(row.get("label") or 0)]
        if gold_rows and wrong_rows:
            pairs.append((max(safe_float(row.get(score_key)) for row in gold_rows), max(safe_float(row.get(score_key)) for row in wrong_rows)))
    if not pairs:
        return {"auc": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(int(seed))
    values = []
    for _ in range(int(resamples)):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        positives = [item[0] for item in sampled]
        negatives = [item[1] for item in sampled]
        values.append(auc_score(positives, negatives))
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return {"auc": round(auc_score([p for p, _ in pairs], [n for _, n in pairs]), 6), "ci_low": round(low, 6), "ci_high": round(high, 6)}


def make_model(name: str, *, seed: int) -> Any:
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=int(seed), solver="liblinear"),
        )
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            class_weight="balanced",
            l2_regularization=0.1,
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            random_state=int(seed),
        )
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=int(seed))
    raise ValueError(f"Unknown model: {name}")


def predict_proba(model: Any, features: Sequence[Sequence[float]]) -> list[float]:
    if not features:
        return []
    probs = model.predict_proba(list(features))
    return [float(value) for value in probs[:, 1].tolist()]


def cross_validated_predictions(
    rows: Sequence[dict[str, Any]],
    *,
    model_name: str,
    n_splits: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.model_selection import GroupKFold

    qids = [str(row["qid"]) for row in rows]
    unique_qids = sorted(set(qids))
    splits = min(int(n_splits), len(unique_qids))
    if splits < 2:
        raise ValueError("Need at least two qids for GroupKFold")
    features = [candidate_feature_vector(row) for row in rows]
    labels = [int(row.get("label") or 0) for row in rows]
    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=splits).split(features, labels, groups=qids), start=1):
        y_train = [labels[idx] for idx in train_idx]
        if len(set(y_train)) < 2:
            raise ValueError(f"Fold {fold} training split has a single class")
        model = make_model(model_name, seed=int(seed) + fold)
        model.fit([features[idx] for idx in train_idx], y_train)
        scores = predict_proba(model, [features[idx] for idx in test_idx])
        train_qids = {qids[idx] for idx in train_idx}
        test_qids = {qids[idx] for idx in test_idx}
        fold_rows.append(
            {
                "fold": fold,
                "train_queries": len(train_qids),
                "test_queries": len(test_qids),
                "train_candidate_rows": len(train_idx),
                "test_candidate_rows": len(test_idx),
                "qid_overlap": len(train_qids & test_qids),
            }
        )
        for idx, score in zip(test_idx, scores):
            row = dict(rows[idx])
            row["model"] = model_name
            row["fold"] = fold
            row["model_score"] = round(float(score), 8)
            predictions.append(row)
    return predictions, fold_rows


def proof_baseline_predictions(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    for row in rows:
        out = dict(row)
        out["model"] = "proof_score"
        out["fold"] = 0
        out["model_score"] = safe_float(row.get("proof_score"))
        outputs.append(out)
    return outputs


def decision_from_metrics(best: dict[str, Any], ci: dict[str, float]) -> str:
    auc = safe_float(best.get("gold_vs_best_wrong_auc"))
    top1 = safe_float(best.get("top1_accuracy_cond_gold_present"))
    top3 = safe_float(best.get("top3_accuracy_cond_gold_present"))
    ci_low = safe_float(ci.get("ci_low"))
    if auc >= 0.75 and ci_low >= 0.70 and top1 >= 0.55:
        return "SUCCESS_ANSWER_CONTRASTIVE_VERIFIER"
    if auc >= 0.70 or top3 >= 0.70:
        return "PARTIAL_TOPK_RECOVERY_NOT_MAINLINE"
    return "FAIL_ANSWER_CONTRASTIVE_VERIFIER"


def prediction_rows_for_write(predictions_by_model: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for model_name, rows in predictions_by_model.items():
        for row in rows:
            outputs.append(
                {
                    "model": model_name,
                    "fold": int(row.get("fold") or 0),
                    "qid": str(row.get("qid") or ""),
                    "type": str(row.get("type") or "unknown"),
                    "candidate_index": int(row.get("candidate_index") or 0),
                    "candidate_text": str(row.get("candidate_text") or ""),
                    "label": int(row.get("label") or 0),
                    "proof_score": round(safe_float(row.get("proof_score")), 8),
                    "model_score": round(safe_float(row.get("model_score")), 8),
                    "sources": list(row.get("sources") or []),
                }
            )
    return outputs


def write_report(payload: dict[str, Any], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    summaries = payload["summaries"]
    best_model = payload["best_supervised_model"]
    lines = [
        "# Answer-Contrastive Verifier Day-1 Gate",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Candidate rows: `{payload['candidate_rows']}`",
        f"- Queries: `{payload['queries']}`",
        f"- Gold-present queries: `{payload['gold_present_queries']}`",
        f"- Feature mode: `cached candidate/proof features only`",
        "- No new LLM calls: `true`",
        f"- No DeBERTa/Qwen fine-tuning: `true`",
        f"- Best supervised model: `{best_model}`",
        "",
        "## Model Comparison",
        "",
        "| Model | Cand AUC | Gold-vs-best-wrong AUC | 95% CI | Top1 Cond | Top3 Cond | MRR Cond | Top1 All |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        ci = payload["bootstrap_ci"].get(name, {"ci_low": 0.0, "ci_high": 0.0})
        lines.append(
            f"| {name} | {summary['candidate_level_auc']} | {summary['gold_vs_best_wrong_auc']} | "
            f"[{ci['ci_low']}, {ci['ci_high']}] | {summary['top1_accuracy_cond_gold_present']} | "
            f"{summary['top3_accuracy_cond_gold_present']} | {summary['mrr_cond_gold_present']} | {summary['top1_accuracy_all']} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "- Green if best supervised gold-vs-best-wrong AUC >= 0.75, CI lower >= 0.70, and conditional top1 >= 0.55.",
            "- Yellow if AUC >= 0.70 or conditional top3 >= 0.70; treat as partial top-k recovery, not a mainline method.",
            "- Red otherwise; write as the sixth negative diagnostic and stop method exploration.",
            "",
            "## Best Supervised Per-Type Breakdown",
            "",
            "| Type | Queries | Recall@20 | Top1 Cond | Top3 Cond | MRR Cond | Gold-vs-Best-Wrong AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for qtype, summary in payload["by_type"].get(best_model, {}).items():
        lines.append(
            f"| {qtype} | {summary['queries']} | {summary['candidate_recall_at_20']} | "
            f"{summary['top1_accuracy_cond_gold_present']} | {summary['top3_accuracy_cond_gold_present']} | "
            f"{summary['mrr_cond_gold_present']} | {summary['gold_vs_best_wrong_auc']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This v0 tests whether cached answer/proof features contain enough residual signal for answer-level ranking.",
            "It is not an end-to-end neural verifier because the model only consumes scalar candidate/proof/source features under GroupKFold by qid.",
        ]
    )
    write_markdown(lines, out_dir / "day1_gate.md")


def run(args: argparse.Namespace) -> dict[str, Any]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rows, candidate_metadata = load_or_build_candidate_rows(args)
    if not rows:
        raise ValueError("No candidate rows available")
    predictions_by_model: dict[str, list[dict[str, Any]]] = {"proof_score": proof_baseline_predictions(rows)}
    fold_summaries: dict[str, Any] = {}
    for model_name in [name.strip() for name in str(args.models).split(",") if name.strip()]:
        predictions, folds = cross_validated_predictions(rows, model_name=model_name, n_splits=int(args.n_splits), seed=int(args.seed))
        predictions_by_model[model_name] = predictions
        fold_summaries[model_name] = folds
    summaries = {name: summarize_ranking(predictions, "model_score") for name, predictions in predictions_by_model.items()}
    bootstrap_ci = {
        name: bootstrap_gold_vs_best_wrong_ci(predictions, "model_score", seed=int(args.seed), resamples=int(args.bootstrap_resamples))
        for name, predictions in predictions_by_model.items()
    }
    supervised_names = [name for name in predictions_by_model if name != "proof_score"]
    best_supervised_model = max(
        supervised_names,
        key=lambda name: (
            safe_float(summaries[name].get("gold_vs_best_wrong_auc")),
            safe_float(summaries[name].get("top1_accuracy_cond_gold_present")),
            safe_float(summaries[name].get("top3_accuracy_cond_gold_present")),
        ),
    )
    decision = decision_from_metrics(summaries[best_supervised_model], bootstrap_ci[best_supervised_model])
    by_type_payload = {name: summarize_by_type(predictions, "model_score") for name, predictions in predictions_by_model.items()}
    payload = {
        "decision": decision,
        "best_supervised_model": best_supervised_model,
        "candidate_rows": len(rows),
        "queries": len(by_qid(rows)),
        "gold_present_queries": sum(any(int(row.get("label") or 0) for row in group) for group in by_qid(rows).values()),
        "feature_names": feature_names(),
        "candidate_metadata": candidate_metadata,
        "config": {
            "day1_5_json": str(args.day1_5_json),
            "candidate_rows_jsonl": str(args.candidate_rows_jsonl),
            "n_splits": int(args.n_splits),
            "models": [name for name in predictions_by_model if name != "proof_score"],
            "seed": int(args.seed),
        },
        "summaries": summaries,
        "bootstrap_ci": bootstrap_ci,
        "folds": fold_summaries,
        "by_type": by_type_payload,
    }
    write_json(payload, Path(args.output_dir) / "day1_gate.json")
    write_jsonl(prediction_rows_for_write(predictions_by_model), Path(args.output_dir) / "predictions.jsonl")
    write_report(payload, args)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proprag_cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--dense_cache_jsonl", default="data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl")
    parser.add_argument("--day1_5_json", default="reports/caps/day1_5_llm_v1_string/caps_day1_5_candidate_v2.json")
    parser.add_argument("--candidate_rows_jsonl", default="reports/contrastive_verifier/candidate_rows.jsonl")
    parser.add_argument("--force_rebuild_candidate_rows", action="store_true")
    parser.add_argument("--dev_start", type=int, default=0)
    parser.add_argument("--dev_end", type=int, default=200)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--answer_cap", type=int, default=20)
    parser.add_argument("--proof_pool_k", type=int, default=30)
    parser.add_argument("--proof_doc_chars", type=int, default=1200)
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--nli_batch_size", type=int, default=64)
    parser.add_argument("--nli_max_length", type=int, default=384)
    parser.add_argument("--entailment_class_index", type=int, default=1)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--models", default="logistic,hist_gradient_boosting,gradient_boosting")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--output_dir", default="reports/contrastive_verifier")
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    best = payload["summaries"][payload["best_supervised_model"]]
    print(
        "Answer-Contrastive decision: "
        f"{payload['decision']} | best={payload['best_supervised_model']} "
        f"auc={best['gold_vs_best_wrong_auc']} top1_cond={best['top1_accuracy_cond_gold_present']}"
    )


if __name__ == "__main__":
    main()
