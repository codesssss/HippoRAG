#!/usr/bin/env python3
"""CAPS Day-0 NLI verifier sanity check.

This is a blocking diagnostic for Candidate-Answer Proof Search (CAPS). It
tests whether an off-the-shelf NLI verifier can separate template-generated
gold proof obligations from same-query random non-gold document pairs.

The script deliberately does not implement CAPS Day-1 candidate generation or
proof search. If this verifier check fails, downstream proof scoring would be
noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from time import perf_counter
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dpathrag_analyze_hard_negatives import is_gold, load_jsonl, row_qid, split_title_body  # noqa: E402
from src.dpathrag.data import normalize_text  # noqa: E402
from src.dpathrag.io import write_json  # noqa: E402


DECISION_PROCEED = "PROCEED_DAY1"
DECISION_REVIEW = "REVIEW_VERIFIER_OR_REPLACE"
DECISION_STOP_FAIL = "STOP_NLI_VERIFIER_FAIL"
DECISION_STOP_UNAVAILABLE = "STOP_NLI_MODEL_UNAVAILABLE"
DECISION_STOP_NO_PAIRS = "STOP_INSUFFICIENT_OBLIGATION_PAIRS"


def write_markdown(lines: Sequence[str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalized_title(value: Any) -> str:
    return normalize_text(value)


def clean_relation(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def verbalize_evidence_triple(triple: Sequence[Any]) -> str:
    """Convert a 2Wiki evidence triple into a simple atomic hypothesis."""
    if len(triple) < 3:
        raise ValueError(f"Expected evidence triple with 3 fields, got {triple!r}")
    subject = str(triple[0] or "").strip()
    relation = clean_relation(triple[1])
    obj = str(triple[2] or "").strip()
    if not subject or not relation or not obj:
        raise ValueError(f"Cannot verbalize incomplete evidence triple: {triple!r}")
    return f"The {relation} of {subject} is {obj}."


def candidate_body(candidate: dict[str, Any]) -> str:
    _, body = split_title_body(candidate)
    title = str(candidate.get("title") or "")
    return f"{title}\n{body}".strip()


def find_subject_doc(record: dict[str, Any], subject: str, *, max_candidates: int) -> tuple[int, dict[str, Any]] | None:
    subject_norm = normalized_title(subject)
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    for idx, candidate in enumerate(candidates):
        if normalized_title(candidate.get("title")) == subject_norm and is_gold(candidate):
            return idx, candidate
    for idx, candidate in enumerate(candidates):
        if normalized_title(candidate.get("title")) == subject_norm:
            return idx, candidate
    return None


def choose_negative_doc(
    record: dict[str, Any],
    *,
    positive_index: int,
    hypothesis: str,
    rng: random.Random,
    max_candidates: int,
) -> tuple[int, dict[str, Any]] | None:
    candidates = list(record.get("candidates") or [])[: int(max_candidates)]
    hyp_norm = normalize_text(hypothesis)
    pool = []
    for idx, candidate in enumerate(candidates):
        if idx == int(positive_index) or is_gold(candidate):
            continue
        body_norm = normalize_text(candidate.get("text"))
        title_norm = normalize_text(candidate.get("title"))
        # Prefer negatives that do not trivially contain the full hypothesis.
        if hyp_norm and hyp_norm in f"{title_norm} {body_norm}":
            continue
        pool.append((idx, candidate))
    if not pool:
        for idx, candidate in enumerate(candidates):
            if idx != int(positive_index) and not is_gold(candidate):
                pool.append((idx, candidate))
    return rng.choice(pool) if pool else None


def build_obligation_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    max_pairs: int,
    max_candidates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build positive and negative premise-hypothesis pairs from evidence triples."""
    rng = random.Random(int(seed))
    pairs: list[dict[str, Any]] = []
    stats = {
        "rows_seen": 0,
        "triples_seen": 0,
        "triples_missing_subject_doc": 0,
        "triples_missing_negative_doc": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
    }
    for record in rows:
        stats["rows_seen"] += 1
        qid = row_qid(record)
        for triple in list(record.get("evidences") or []):
            if len(pairs) >= int(max_pairs) * 2:
                break
            stats["triples_seen"] += 1
            try:
                hypothesis = verbalize_evidence_triple(triple)
            except ValueError:
                continue
            found = find_subject_doc(record, str(triple[0]), max_candidates=max_candidates)
            if found is None:
                stats["triples_missing_subject_doc"] += 1
                continue
            positive_index, positive_doc = found
            negative = choose_negative_doc(
                record,
                positive_index=positive_index,
                hypothesis=hypothesis,
                rng=rng,
                max_candidates=max_candidates,
            )
            if negative is None:
                stats["triples_missing_negative_doc"] += 1
                continue
            negative_index, negative_doc = negative
            pairs.append(
                {
                    "qid": qid,
                    "label": 1,
                    "premise": candidate_body(positive_doc),
                    "hypothesis": hypothesis,
                    "triple": list(triple),
                    "doc_index": positive_index,
                    "doc_title": str(positive_doc.get("title") or ""),
                }
            )
            pairs.append(
                {
                    "qid": qid,
                    "label": 0,
                    "premise": candidate_body(negative_doc),
                    "hypothesis": hypothesis,
                    "triple": list(triple),
                    "doc_index": negative_index,
                    "doc_title": str(negative_doc.get("title") or ""),
                }
            )
            stats["positive_pairs"] += 1
            stats["negative_pairs"] += 1
        if len(pairs) >= int(max_pairs) * 2:
            break
    return pairs, stats


def auc_score(positives: Sequence[float], negatives: Sequence[float]) -> float:
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if float(pos) > float(neg):
                wins += 1.0
            elif math.isclose(float(pos), float(neg)):
                wins += 0.5
    return wins / max(1.0, total)


def bootstrap_auc_ci(
    positives: Sequence[float],
    negatives: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    if not positives or not negatives:
        return {"auc": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(int(seed))
    pos_values = [float(value) for value in positives]
    neg_values = [float(value) for value in negatives]
    values = []
    for _ in range(int(resamples)):
        pos = [rng.choice(pos_values) for _ in pos_values]
        neg = [rng.choice(neg_values) for _ in neg_values]
        values.append(auc_score(pos, neg))
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return {"auc": round(auc_score(pos_values, neg_values), 6), "ci_low": round(low, 6), "ci_high": round(high, 6)}


def decide_from_auc(auc: float, *, proceed_threshold: float = 0.80, stop_threshold: float = 0.70) -> str:
    value = float(auc)
    if value >= float(proceed_threshold):
        return DECISION_PROCEED
    if value < float(stop_threshold):
        return DECISION_STOP_FAIL
    return DECISION_REVIEW


def entailment_index_from_config(config: Any, fallback: int) -> int:
    id2label = getattr(config, "id2label", {}) or {}
    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)
    return int(fallback)


def run_transformers_nli(
    model_name_or_path: str,
    pairs: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    entailment_class_index: int,
    local_files_only: bool,
) -> tuple[list[float], dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=bool(local_files_only))
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=bool(local_files_only))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    entail_idx = entailment_index_from_config(model.config, entailment_class_index)
    scores: list[float] = []
    start_time = perf_counter()
    with torch.no_grad():
        for start in range(0, len(pairs), int(batch_size)):
            batch = list(pairs[start : start + int(batch_size)])
            encoded = tokenizer(
                [str(row["premise"]) for row in batch],
                [str(row["hypothesis"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            scores.extend(float(value) for value in probs[:, entail_idx].detach().cpu().tolist())
    elapsed = max(perf_counter() - start_time, 1e-12)
    meta = {
        "device": str(device),
        "entailment_class_index": entail_idx,
        "id2label": {str(key): str(value) for key, value in (getattr(model.config, "id2label", {}) or {}).items()},
        "elapsed_seconds": round(elapsed, 4),
        "pairs_per_second": round(len(pairs) / elapsed, 4),
    }
    return scores, meta


def lexical_overlap_scores(pairs: Sequence[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    scores = []
    for pair in pairs:
        premise_tokens = {tok for tok in normalize_text(pair["premise"]).split() if len(tok) > 2}
        hyp_tokens = {tok for tok in normalize_text(pair["hypothesis"]).split() if len(tok) > 2}
        scores.append(len(premise_tokens & hyp_tokens) / max(1, len(hyp_tokens)))
    return scores, {"device": "none", "entailment_class_index": -1, "lexical_smoke": True}


def summarize_scores(pairs: Sequence[dict[str, Any]], scores: Sequence[float], *, seed: int, resamples: int) -> dict[str, Any]:
    positives = [float(score) for row, score in zip(pairs, scores) if int(row["label"]) == 1]
    negatives = [float(score) for row, score in zip(pairs, scores) if int(row["label"]) == 0]
    ci = bootstrap_auc_ci(positives, negatives, seed=seed, resamples=resamples)
    paired_wins = []
    for idx in range(0, min(len(pairs), len(scores)), 2):
        if idx + 1 >= len(pairs):
            continue
        if int(pairs[idx]["label"]) == 1 and int(pairs[idx + 1]["label"]) == 0:
            paired_wins.append(1.0 if float(scores[idx]) > float(scores[idx + 1]) else 0.0)
    return {
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "auc": ci["auc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "mean_positive_score": round(sum(positives) / max(1, len(positives)), 6),
        "mean_negative_score": round(sum(negatives) / max(1, len(negatives)), 6),
        "paired_win_rate": round(sum(paired_wins) / max(1, len(paired_wins)), 6),
    }


def write_report(payload: dict[str, Any], args: argparse.Namespace) -> None:
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    write_json(payload, report_json)
    examples = payload.get("examples", [])[:3]
    lines = [
        "# CAPS Day-0 NLI Sanity",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Status: `{payload['status']}`",
        f"- Model: `{payload.get('model_name', '')}`",
        f"- Lexical smoke: `{payload.get('lexical_smoke', False)}`",
        f"- Pairs: `{payload.get('num_pairs', 0)}`",
        f"- Positive/negative: `{payload.get('metrics', {}).get('num_positive', 0)}` / `{payload.get('metrics', {}).get('num_negative', 0)}`",
        f"- AUC: `{payload.get('metrics', {}).get('auc', 0.0)}`",
        f"- 95% CI: `[{payload.get('metrics', {}).get('ci_low', 0.0)}, {payload.get('metrics', {}).get('ci_high', 0.0)}]`",
        f"- Paired win rate: `{payload.get('metrics', {}).get('paired_win_rate', 0.0)}`",
        f"- Mean positive score: `{payload.get('metrics', {}).get('mean_positive_score', 0.0)}`",
        f"- Mean negative score: `{payload.get('metrics', {}).get('mean_negative_score', 0.0)}`",
        f"- Throughput: `{payload.get('model_meta', {}).get('pairs_per_second', 0.0)}` pairs/s",
        "",
        "## Gate",
        "",
        "- `AUC >= 0.80`: proceed to CAPS Day 1.",
        "- `0.70 <= AUC < 0.80`: verifier is marginal; replace/calibrate verifier before Day 1.",
        "- `AUC < 0.70` or model unavailable: stop CAPS with this verifier.",
    ]
    if payload.get("error"):
        lines.extend(["", "## Error", "", f"`{payload['error']}`"])
    if examples:
        lines.extend(["", "## Example Pairs"])
        for example in examples:
            lines.extend(
                [
                    "",
                    f"- qid: `{example.get('qid')}`, label: `{example.get('label')}`, score: `{example.get('score')}`",
                    f"- doc: `{example.get('doc_title')}`",
                    f"- hypothesis: {example.get('hypothesis')}",
                ]
            )
    write_markdown(lines, report_md)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))
    dev = list(rows[int(args.dev_start) : int(args.dev_end)])
    pairs, pair_stats = build_obligation_pairs(
        dev,
        max_pairs=int(args.max_pairs),
        max_candidates=int(args.max_candidates),
        seed=int(args.seed),
    )
    if len(pairs) < 2:
        payload = {
            "status": "insufficient_pairs",
            "decision": DECISION_STOP_NO_PAIRS,
            "model_name": args.model_name,
            "num_pairs": len(pairs),
            "pair_stats": pair_stats,
            "metrics": {},
            "examples": [],
        }
        write_report(payload, args)
        return payload

    try:
        if args.allow_lexical_smoke:
            scores, model_meta = lexical_overlap_scores(pairs)
        else:
            scores, model_meta = run_transformers_nli(
                args.model_name,
                pairs,
                batch_size=int(args.batch_size),
                max_length=int(args.max_length),
                entailment_class_index=int(args.entailment_class_index),
                local_files_only=not bool(args.allow_download),
            )
    except Exception as exc:
        payload = {
            "status": "model_unavailable",
            "decision": DECISION_STOP_UNAVAILABLE,
            "model_name": args.model_name,
            "num_pairs": len(pairs),
            "pair_stats": pair_stats,
            "metrics": {},
            "model_meta": {},
            "examples": [],
            "error": str(exc),
        }
        write_report(payload, args)
        return payload

    metrics = summarize_scores(pairs, scores, seed=int(args.seed), resamples=int(args.bootstrap_resamples))
    decision = decide_from_auc(
        float(metrics["auc"]),
        proceed_threshold=float(args.proceed_auc),
        stop_threshold=float(args.stop_auc),
    )
    payload = {
        "status": "completed",
        "decision": decision,
        "model_name": args.model_name,
        "lexical_smoke": bool(args.allow_lexical_smoke),
        "cache_jsonl": args.cache_jsonl,
        "dev_start": int(args.dev_start),
        "dev_end": int(args.dev_end),
        "num_pairs": len(pairs),
        "pair_stats": pair_stats,
        "metrics": metrics,
        "model_meta": model_meta,
        "examples": [
            {
                **{key: row[key] for key in ("qid", "label", "hypothesis", "doc_title", "doc_index") if key in row},
                "score": round(float(score), 6),
            }
            for row, score in zip(pairs[: min(12, len(pairs))], scores[: min(12, len(scores))])
        ],
    }
    write_report(payload, args)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base-mnli")
    parser.add_argument("--dev_start", type=int, default=0)
    parser.add_argument("--dev_end", type=int, default=200)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--max_pairs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--entailment_class_index", type=int, default=2)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--proceed_auc", type=float, default=0.80)
    parser.add_argument("--stop_auc", type=float, default=0.70)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--allow_lexical_smoke", action="store_true")
    parser.add_argument("--report_json", default="reports/caps/nli_sanity_day0.json")
    parser.add_argument("--report_md", default="reports/caps/nli_sanity_day0.md")
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"CAPS Day0 decision: {payload['decision']} ({payload['status']})")


if __name__ == "__main__":
    main()
