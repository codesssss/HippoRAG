#!/usr/bin/env python3
"""CAPS Day-1 diagnostic gates.

CAPS Day 1 is still diagnostic-only. It checks:

1. Candidate answer recall on the dev fold.
2. Oracle-answer proof separability.
3. A small non-oracle proof-selection pilot.

It does not run held-out eval800/eval1000 and does not modify any D-PathRAG,
CEE, or DAEC code paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from caps_day0_nli_sanity import (  # noqa: E402
    auc_score,
    bootstrap_auc_ci,
    run_transformers_nli,
    verbalize_evidence_triple,
)
from cee_day1_diagnostic import FrozenReaderLoglikProbe, selected_docs  # noqa: E402
from dpathrag_analyze_hard_negatives import is_gold, load_jsonl, row_qid, split_title_body  # noqa: E402
from dpathrag_cee_pairwise_common import rank_indices_for  # noqa: E402
from dpathrag_cee_edit_policy import make_example  # noqa: E402
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import exact_match, gold_answers, normalize_answer, token_f1  # noqa: E402
from src.dpathrag.selector_data import support_metrics_for_indices  # noqa: E402


MONTH_PATTERN = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_PATTERNS = [
    re.compile(rf"\b\d{{1,2}}\s+{MONTH_PATTERN}\s+\d{{3,4}}\b", re.IGNORECASE),
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{1,2}},?\s+\d{{3,4}}\b", re.IGNORECASE),
    re.compile(r"\b\d{3,4}\b"),
]
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
CAPITALIZED_SPAN = re.compile(r"\b(?:[A-Z][A-Za-z0-9'’.-]+(?:\s+|$)){1,5}")
CONNECTED_ENTITY_SPAN = re.compile(
    r"\b[A-Z][A-Za-z0-9'’.-]+(?:\s+(?:of|de|da|del|di|du|van|von|the|and|&)\s+[A-Z][A-Za-z0-9'’.-]+)+(?:\s+[A-Z][A-Za-z0-9'’.-]+){0,3}\b"
)
YES_NO_START = re.compile(r"^\s*(?:are|is|was|were|do|does|did|can|could|would|should|has|have|had)\b", re.IGNORECASE)


def write_markdown(lines: Sequence[str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha1_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def record_docs(record: dict[str, Any], indices: Sequence[int]) -> list[dict[str, Any]]:
    return selected_docs(record, indices)


def top_indices(record: dict[str, Any], top_k: int, max_candidates: int) -> list[int]:
    return rank_indices_for(record, top_k=int(top_k), max_candidates=int(max_candidates))


def load_rows_by_qid(path: str | Path, *, limit: int = 0) -> dict[str, dict[str, Any]]:
    return {row_qid(row): row for row in load_jsonl(path, limit=limit)}


def dedup_union_records(records: Sequence[dict[str, Any]], *, pool_k: int, cap: int) -> list[dict[str, Any]]:
    seen = set()
    docs: list[dict[str, Any]] = []
    for record in records:
        for candidate in list(record.get("candidates") or [])[: int(pool_k)]:
            key = normalize_answer(candidate.get("title")) or normalize_answer(candidate.get("doc_id"))
            if not key or key in seen:
                continue
            seen.add(key)
            docs.append(candidate)
            if len(docs) >= int(cap):
                return docs
    return docs


def cluster_answer_candidates(candidates: Sequence[dict[str, Any]], *, cap: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        text = str(cand.get("text") or "").strip()
        norm = normalize_answer(text)
        if not norm:
            continue
        score = float(cand.get("score") or 0.0)
        row = grouped.setdefault(
            norm,
            {
                "text": text,
                "normalized": norm,
                "score": 0.0,
                "max_score": 0.0,
                "sources": [],
                "source_count": 0,
            },
        )
        row["max_score"] = max(float(row.get("max_score") or 0.0), score)
        row["source_count"] = int(row["source_count"]) + 1
        row["sources"].append(str(cand.get("source") or ""))
        if len(text) < len(str(row.get("text") or "")):
            row["text"] = text
        row["score"] = float(row["max_score"]) + 2.0 * min(5, max(0, int(row["source_count"]) - 1))
    ranked = sorted(grouped.values(), key=lambda item: (float(item["score"]), int(item["source_count"]), -len(str(item["text"]))), reverse=True)
    return ranked[: int(cap)]


class ReaderAnswerCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.values: dict[str, str] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.values[str(row["key"])] = str(row.get("answer") or "")

    def key(self, qid: str, question: str, source: str, docs: Sequence[dict[str, Any]]) -> str:
        return sha1_json(
            {
                "qid": qid,
                "question": question,
                "source": source,
                "doc_titles": [str(doc.get("title") or "") for doc in docs],
                "doc_hashes": [hashlib.sha1(str(doc.get("text") or "").encode("utf-8")).hexdigest() for doc in docs],
            }
        )

    def get(self, key: str) -> str | None:
        return self.values.get(str(key))

    def set(self, key: str, answer: str, metadata: dict[str, Any]) -> None:
        self.values[str(key)] = str(answer)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "answer": answer, "metadata": metadata}, ensure_ascii=False) + "\n")


def extract_heuristic_answers(record: dict[str, Any], docs: Sequence[dict[str, Any]], *, max_per_source: int) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    question = str(record.get("question") or "")
    if YES_NO_START.search(question):
        answers.extend(
            [
                {"text": "yes", "source": "yes_no_prior", "score": 24.0},
                {"text": "no", "source": "yes_no_prior", "score": 24.0},
            ]
        )
    for rank, doc in enumerate(docs, start=1):
        title = str(doc.get("title") or "").strip()
        _, body = split_title_body(doc)
        if title:
            answers.append({"text": title, "source": "title", "score": max(1.0, 18.0 - 0.5 * rank)})
        text = str(body or doc.get("text") or "")
        for pattern in DATE_PATTERNS:
            for match in pattern.findall(text):
                value = match if isinstance(match, str) else " ".join(item for item in match if item)
                answers.append({"text": value, "source": "date", "score": max(1.0, 16.0 - 0.4 * rank)})
        for match in NUMBER_PATTERN.findall(text)[:8]:
            if re.fullmatch(r"\d{3,4}", str(match)):
                continue
            answers.append({"text": match, "source": "number", "score": max(0.5, 8.0 - 0.3 * rank)})
        for match in CONNECTED_ENTITY_SPAN.findall(text)[:8]:
            value = " ".join(str(match).split())
            answers.append({"text": value, "source": "connected_entity_span", "score": max(0.5, 12.0 - 0.3 * rank)})
        for match in CAPITALIZED_SPAN.findall(text)[:8]:
            value = " ".join(str(match).split())
            if len(value) > 2 and normalize_answer(value) not in {"the", "a", "an"}:
                answers.append({"text": value, "source": "capitalized_span", "score": max(0.5, 7.0 - 0.25 * rank)})
    return answers[: int(max_per_source)]


def build_reader_probe(args: argparse.Namespace) -> FrozenReaderLoglikProbe:
    return FrozenReaderLoglikProbe(
        model_name_or_path=str(args.reader_model),
        cache_path=Path(args.cache_dir) / "caps_day1_loglik_cache.jsonl",
        max_input_tokens=int(args.max_input_tokens),
        max_target_tokens=int(args.max_target_tokens),
        device=str(args.device),
    )


def reader_contexts(
    prop_record: dict[str, Any],
    dense_record: dict[str, Any] | None,
    *,
    top_k: int,
    max_candidates: int,
    union_pool_k: int,
    single_reader_docs: int,
    pair_reader_pairs: int,
) -> dict[str, list[dict[str, Any]]]:
    prop_docs = record_docs(prop_record, top_indices(prop_record, top_k, max_candidates))
    contexts = {"reader_prop_top5": prop_docs}
    union_docs_raw = dedup_union_records([row for row in [prop_record, dense_record] if row is not None], pool_k=union_pool_k, cap=max(top_k, single_reader_docs, 8))
    union_docs = [{"title": str(doc.get("title") or ""), "text": str(doc.get("text") or "")} for doc in union_docs_raw]
    if dense_record is not None:
        dense_docs = record_docs(dense_record, top_indices(dense_record, top_k, max_candidates))
        contexts["reader_dense_top5"] = dense_docs
        contexts["reader_union_top5"] = union_docs[:top_k]
    for idx, doc in enumerate(union_docs[: int(single_reader_docs)]):
        contexts[f"reader_single_{idx + 1:02d}"] = [doc]
    pair_candidates = []
    for left in range(min(4, len(union_docs))):
        for right in range(left + 1, min(8, len(union_docs))):
            pair_candidates.append((left, right))
    for pair_idx, (left, right) in enumerate(pair_candidates[: int(pair_reader_pairs)], start=1):
        contexts[f"reader_pair_{pair_idx:02d}_{left + 1}_{right + 1}"] = [union_docs[left], union_docs[right]]
    return contexts


def generate_answer_candidates(
    prop_record: dict[str, Any],
    dense_record: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
    probe: FrozenReaderLoglikProbe | None,
    answer_cache: ReaderAnswerCache | None,
) -> list[dict[str, Any]]:
    qid = row_qid(prop_record)
    question = str(prop_record.get("question") or "")
    union_docs = dedup_union_records(
        [row for row in [prop_record, dense_record] if row is not None],
        pool_k=int(args.pool_k),
        cap=int(args.union_cap),
    )
    answers = extract_heuristic_answers(prop_record, union_docs, max_per_source=int(args.max_heuristic_answers))
    if probe is not None and not bool(args.no_reader_answers):
        assert answer_cache is not None
        for source, docs in reader_contexts(
            prop_record,
            dense_record,
            top_k=int(args.top_k),
            max_candidates=int(args.max_candidates),
            union_pool_k=int(args.pool_k),
            single_reader_docs=int(args.single_reader_docs),
            pair_reader_pairs=int(args.pair_reader_pairs),
        ).items():
            key = answer_cache.key(qid, question, source, docs)
            cached = answer_cache.get(key)
            if cached is None:
                cached = probe.generate_answer(qid, question, docs, max_new_tokens=int(args.max_new_tokens))
                answer_cache.set(key, cached, {"qid": qid, "source": source})
            if source.startswith("reader_single"):
                score = 42.0
            elif source.startswith("reader_pair"):
                score = 44.0
            else:
                score = 48.0
            answers.append({"text": cached, "source": source, "score": score})
    return cluster_answer_candidates(answers, cap=int(args.answer_cap))


def answer_recall_at(candidates: Sequence[dict[str, Any]], golds: Sequence[str], k: int) -> float:
    observed = {str(row["normalized"]) for row in list(candidates)[: int(k)]}
    return float(any(normalize_answer(gold) in observed for gold in golds))


def candidate_recall_gate(
    prop_rows: Sequence[dict[str, Any]],
    dense_by_qid: dict[str, dict[str, Any]],
    *,
    args: argparse.Namespace,
    probe: FrozenReaderLoglikProbe | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    answer_cache = ReaderAnswerCache(Path(args.cache_dir) / "caps_day1_answer_cache.jsonl") if probe is not None else None
    rows = []
    candidates_by_qid: dict[str, list[dict[str, Any]]] = {}
    for record in prop_rows:
        qid = row_qid(record)
        candidates = generate_answer_candidates(
            record,
            dense_by_qid.get(qid),
            args=args,
            probe=probe,
            answer_cache=answer_cache,
        )
        candidates_by_qid[qid] = candidates
        golds = gold_answers(record)
        rows.append(
            {
                "qid": qid,
                "gold_answers": golds,
                "candidate_count": len(candidates),
                "recall_at_5": answer_recall_at(candidates, golds, 5),
                "recall_at_10": answer_recall_at(candidates, golds, 10),
                "recall_at_20": answer_recall_at(candidates, golds, 20),
                "top_candidates": candidates[:10],
            }
        )
    denom = max(1, len(rows))
    summary = {
        "rows": len(rows),
        "answer_cap": int(args.answer_cap),
        "recall_at_5": round(sum(row["recall_at_5"] for row in rows) / denom, 6),
        "recall_at_10": round(sum(row["recall_at_10"] for row in rows) / denom, 6),
        "recall_at_20": round(sum(row["recall_at_20"] for row in rows) / denom, 6),
        "avg_candidate_count": round(sum(row["candidate_count"] for row in rows) / denom, 4),
        "passed": False,
    }
    summary["passed"] = bool(summary["recall_at_5"] >= float(args.recall5_gate))
    write_json({"summary": summary, "rows": rows}, Path(args.output_dir) / "candidate_answer_recall.json")
    write_jsonl(rows, Path(args.output_dir) / "candidate_answer_recall.rows.jsonl")
    write_markdown(
        [
            "# CAPS Candidate Answer Recall",
            "",
            f"- Passed: `{summary['passed']}`",
            f"- Rows: `{summary['rows']}`",
            f"- Recall@5: `{summary['recall_at_5']}`",
            f"- Recall@10: `{summary['recall_at_10']}`",
            f"- Recall@20: `{summary['recall_at_20']}`",
            f"- Avg candidate count: `{summary['avg_candidate_count']}`",
            f"- Gate: `Recall@5 >= {args.recall5_gate}`",
        ],
        Path(args.output_dir) / "candidate_answer_recall.md",
    )
    return summary, candidates_by_qid


def candidate_obligations(record: dict[str, Any], candidate_answer: str) -> tuple[list[str], bool]:
    """Build answer-conditioned template obligations.

    Returns (obligations, substituted). If substituted is false, the candidate
    cannot be tied to a final-answer slot, so proof scores are less diagnostic.
    """
    gold_norms = {normalize_answer(answer) for answer in gold_answers(record) if normalize_answer(answer)}
    candidate = str(candidate_answer or "").strip()
    candidate_norm = normalize_answer(candidate)
    obligations = []
    substituted = False
    for triple in list(record.get("evidences") or []):
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        new_triple = list(triple[:3])
        if normalize_answer(new_triple[2]) in gold_norms and candidate_norm and candidate_norm not in gold_norms:
            new_triple[2] = candidate
            substituted = True
        elif normalize_answer(new_triple[2]) in gold_norms and candidate_norm in gold_norms:
            substituted = True
        try:
            obligations.append(verbalize_evidence_triple(new_triple))
        except ValueError:
            continue
    return obligations, substituted


def noisy_or(values: Sequence[float]) -> float:
    product = 1.0
    for value in values:
        product *= 1.0 - max(0.0, min(1.0, float(value)))
    return 1.0 - product


def proof_score_from_matrix(matrix: Sequence[Sequence[float]]) -> float:
    if not matrix:
        return 0.0
    coverages = [max(row) if row else 0.0 for row in matrix]
    score = 1.0
    for coverage in coverages:
        score *= max(0.0, min(1.0, float(coverage)))
    return float(score)


def greedy_proof_indices(matrix: Sequence[Sequence[float]], doc_indices: Sequence[int], *, threshold: float, max_docs: int) -> list[int]:
    selected: list[int] = []
    remaining = list(range(len(doc_indices)))
    current = [0.0 for _ in matrix]
    while remaining and len(selected) < int(max_docs):
        best_pos = None
        best_gain = -1.0
        for pos in remaining:
            trial = [max(current[row_idx], float(matrix[row_idx][pos])) for row_idx in range(len(matrix))]
            gain = proof_score_from_matrix([[value] for value in trial]) - proof_score_from_matrix([[value] for value in current])
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
        if best_pos is None or best_gain <= 0.0:
            break
        selected.append(int(doc_indices[best_pos]))
        remaining.remove(best_pos)
        current = [max(current[row_idx], float(matrix[row_idx][best_pos])) for row_idx in range(len(matrix))]
        if proof_score_from_matrix([[value] for value in current]) >= float(threshold):
            break
    return selected


def score_candidate_proofs(
    records_and_candidates: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Score candidate proof obligations with NLI in one batched pass."""
    nli_pairs = []
    tasks = []
    for record, answer_rows in records_and_candidates:
        doc_indices = list(range(min(int(args.proof_pool_k), len(record.get("candidates") or []))))
        docs = list(record.get("candidates") or [])
        for answer in answer_rows:
            obligations, substituted = candidate_obligations(record, str(answer.get("text") or ""))
            start = len(nli_pairs)
            for obligation in obligations:
                for doc_idx in doc_indices:
                    nli_pairs.append(
                        {
                            "premise": str(docs[doc_idx].get("text") or ""),
                            "hypothesis": obligation,
                            "label": 0,
                        }
                    )
            tasks.append(
                {
                    "qid": row_qid(record),
                    "answer": answer,
                    "obligations": obligations,
                    "substituted": substituted,
                    "doc_indices": doc_indices,
                    "start": start,
                    "count": len(obligations) * len(doc_indices),
                }
            )
    if not nli_pairs:
        return []
    scores, _ = run_transformers_nli(
        str(args.nli_model),
        nli_pairs,
        batch_size=int(args.nli_batch_size),
        max_length=int(args.nli_max_length),
        entailment_class_index=int(args.entailment_class_index),
        local_files_only=not bool(args.allow_download),
    )
    outputs = []
    for task in tasks:
        doc_count = len(task["doc_indices"])
        flat = scores[int(task["start"]) : int(task["start"]) + int(task["count"])]
        matrix = [flat[start : start + doc_count] for start in range(0, len(flat), doc_count)]
        proof_score = proof_score_from_matrix(matrix)
        selected = greedy_proof_indices(
            matrix,
            task["doc_indices"],
            threshold=float(args.proof_threshold),
            max_docs=int(args.top_k),
        )
        outputs.append({**task, "matrix": matrix, "proof_score": proof_score, "selected_indices": selected})
    return outputs


def wrong_candidates_for_record(record: dict[str, Any], candidates: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    gold_norms = {normalize_answer(answer) for answer in gold_answers(record)}
    wrong = [row for row in candidates if normalize_answer(row.get("text")) not in gold_norms]
    return wrong[: int(limit)]


def oracle_proof_gate(
    prop_rows: Sequence[dict[str, Any]],
    candidates_by_qid: dict[str, list[dict[str, Any]]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    jobs = []
    for record in prop_rows:
        gold_text = gold_answers(record)[0]
        rows = [{"text": gold_text, "normalized": normalize_answer(gold_text), "score": 1.0, "sources": ["gold_oracle"]}]
        rows.extend(wrong_candidates_for_record(record, candidates_by_qid.get(row_qid(record), []), limit=int(args.wrong_candidates_per_query)))
        jobs.append((record, rows))
    scored = score_candidate_proofs(jobs, args=args)
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_qid.setdefault(str(row["qid"]), []).append(row)
    gold_scores = []
    wrong_scores = []
    top1_hits = []
    substituted_rows = 0
    details = []
    for record in prop_rows:
        qid = row_qid(record)
        rows = by_qid.get(qid, [])
        if not rows:
            continue
        gold_norm = normalize_answer(gold_answers(record)[0])
        gold = [row for row in rows if normalize_answer(row["answer"].get("text")) == gold_norm]
        wrong = [row for row in rows if normalize_answer(row["answer"].get("text")) != gold_norm]
        if not gold or not wrong:
            continue
        if any(bool(row.get("substituted")) for row in rows):
            substituted_rows += 1
        gold_score = max(float(row["proof_score"]) for row in gold)
        best_wrong = max(float(row["proof_score"]) for row in wrong)
        gold_scores.append(gold_score)
        wrong_scores.append(best_wrong)
        top1_hits.append(1.0 if gold_score > best_wrong else 0.0)
        details.append(
            {
                "qid": qid,
                "gold_answer": gold_answers(record)[0],
                "gold_score": round(gold_score, 6),
                "best_wrong_score": round(best_wrong, 6),
                "top1": gold_score > best_wrong,
                "substituted": any(bool(row.get("substituted")) for row in rows),
            }
        )
    ci = bootstrap_auc_ci(gold_scores, wrong_scores, seed=int(args.seed), resamples=int(args.bootstrap_resamples))
    summary = {
        "rows": len(gold_scores),
        "substituted_rows": substituted_rows,
        "auc": ci["auc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "gold_top1_rate": round(sum(top1_hits) / max(1, len(top1_hits)), 6),
        "mean_gold_score": round(sum(gold_scores) / max(1, len(gold_scores)), 6),
        "mean_best_wrong_score": round(sum(wrong_scores) / max(1, len(wrong_scores)), 6),
        "passed": bool(ci["auc"] >= float(args.oracle_auc_gate)),
    }
    write_json({"summary": summary, "rows": details}, Path(args.output_dir) / "oracle_answer_proof_separability.json")
    write_jsonl(details, Path(args.output_dir) / "oracle_answer_proof_separability.rows.jsonl")
    write_markdown(
        [
            "# CAPS Oracle-Answer Proof Separability",
            "",
            f"- Passed: `{summary['passed']}`",
            f"- Rows: `{summary['rows']}`",
            f"- Substituted rows: `{summary['substituted_rows']}`",
            f"- AUC gold vs best wrong: `{summary['auc']}`",
            f"- 95% CI: `[{summary['ci_low']}, {summary['ci_high']}]`",
            f"- Gold top-1 rate: `{summary['gold_top1_rate']}`",
            f"- Mean gold score: `{summary['mean_gold_score']}`",
            f"- Mean best wrong score: `{summary['mean_best_wrong_score']}`",
            f"- Gate: `AUC >= {args.oracle_auc_gate}`",
        ],
        Path(args.output_dir) / "oracle_answer_proof_separability.md",
    )
    return summary


def non_oracle_pilot(
    prop_rows: Sequence[dict[str, Any]],
    candidates_by_qid: dict[str, list[dict[str, Any]]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    jobs = [(record, candidates_by_qid.get(row_qid(record), [])[: int(args.answer_cap)]) for record in prop_rows]
    scored = score_candidate_proofs(jobs, args=args)
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_qid.setdefault(str(row["qid"]), []).append(row)
    rows = []
    for record in prop_rows:
        qid = row_qid(record)
        scored_rows = by_qid.get(qid, [])
        if not scored_rows:
            continue
        best = max(scored_rows, key=lambda row: float(row["proof_score"]))
        answer = str(best["answer"].get("text") or "")
        example = make_example(record, max_candidates=int(args.max_candidates), top_k=int(args.top_k), embedding_features={})
        selected = list(best.get("selected_indices") or [])[: int(args.top_k)]
        metrics = support_metrics_for_indices(example, selected)
        rows.append(
            {
                "qid": qid,
                "prediction": answer,
                "gold_answers": gold_answers(record),
                "em": exact_match(gold_answers(record), answer),
                "f1": token_f1(gold_answers(record), answer),
                "proof_score": round(float(best["proof_score"]), 6),
                "selected_indices": selected,
                "support_complete": float(metrics.get("support_complete") or 0.0),
                "support_recall": float(metrics.get("support_recall") or 0.0),
            }
        )
    denom = max(1, len(rows))
    summary = {
        "rows": len(rows),
        "answer_em": round(sum(row["em"] for row in rows) / denom, 6),
        "answer_f1": round(sum(row["f1"] for row in rows) / denom, 6),
        "support_complete": round(sum(row["support_complete"] for row in rows) / denom, 6),
        "support_recall": round(sum(row["support_recall"] for row in rows) / denom, 6),
        "baseline_best_single_pool_f1": float(args.best_single_pool_f1),
        "baseline_rank_support_complete": float(args.rank_support_complete),
        "passed": False,
    }
    summary["passed"] = bool(
        summary["answer_f1"] >= float(args.best_single_pool_f1) - 0.005
        and summary["support_complete"] >= float(args.rank_support_complete) + 0.01
    )
    write_json({"summary": summary, "rows": rows}, Path(args.output_dir) / "non_oracle_proof_selection_dev.json")
    write_jsonl(rows, Path(args.output_dir) / "non_oracle_proof_selection_dev.rows.jsonl")
    write_markdown(
        [
            "# CAPS Non-Oracle Proof Selection Dev",
            "",
            f"- Passed: `{summary['passed']}`",
            f"- Rows: `{summary['rows']}`",
            f"- Answer EM: `{summary['answer_em']}`",
            f"- Answer F1: `{summary['answer_f1']}`",
            f"- Support complete: `{summary['support_complete']}`",
            f"- Support recall: `{summary['support_recall']}`",
            f"- Gate F1: `>= {float(args.best_single_pool_f1) - 0.005}`",
            f"- Gate support_complete: `>= {float(args.rank_support_complete) + 0.01}`",
        ],
        Path(args.output_dir) / "non_oracle_proof_selection_dev.md",
    )
    return summary


def write_day1_summary(report: dict[str, Any], args: argparse.Namespace) -> None:
    reports = report["reports"]
    if not reports.get("candidate_answer_recall", {}).get("passed"):
        decision = "STOP_CANDIDATE_RECALL_FAIL"
    elif not reports.get("oracle_answer_proof_separability", {}).get("passed"):
        decision = "STOP_ORACLE_PROOF_SEPARABILITY_FAIL"
    elif not reports.get("non_oracle_proof_selection", {}).get("passed"):
        decision = "STOP_NON_ORACLE_PROOF_SELECTION_FAIL"
    else:
        decision = "PROCEED_CAPS_MINIMAL_EVAL"
    report["decision"] = decision
    write_json(report, Path(args.output_dir) / "day1_summary.json")
    lines = ["# CAPS Day-1 Summary", "", f"- Decision: `{decision}`", ""]
    for name, payload in reports.items():
        lines.append(f"- {name}: passed=`{payload.get('passed')}`")
    write_markdown(lines, Path(args.output_dir) / "day1_summary.md")


def run(args: argparse.Namespace) -> dict[str, Any]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    prop_rows_all = load_jsonl(args.proprag_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))
    prop_rows = list(prop_rows_all[int(args.dev_start) : int(args.dev_end)])
    dense_by_qid = load_rows_by_qid(args.dense_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows))) if Path(args.dense_cache_jsonl).exists() else {}
    probe = None if bool(args.no_reader_answers) else build_reader_probe(args)
    candidate_summary, candidates_by_qid = candidate_recall_gate(prop_rows, dense_by_qid, args=args, probe=probe)
    report = {"reports": {"candidate_answer_recall": candidate_summary}}
    if bool(args.stop_after_candidate_recall) or not candidate_summary.get("passed"):
        write_day1_summary(report, args)
        return report
    oracle_summary = oracle_proof_gate(prop_rows, candidates_by_qid, args=args)
    report["reports"]["oracle_answer_proof_separability"] = oracle_summary
    if bool(args.stop_after_oracle_proof) or not oracle_summary.get("passed"):
        write_day1_summary(report, args)
        return report
    non_oracle_summary = non_oracle_pilot(prop_rows, candidates_by_qid, args=args)
    report["reports"]["non_oracle_proof_selection"] = non_oracle_summary
    write_day1_summary(report, args)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proprag_cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--dense_cache_jsonl", default="data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl")
    parser.add_argument("--reader_model", default="data/dpathrag/models/flan_t5_base_gold_k5_5k")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--dev_start", type=int, default=0)
    parser.add_argument("--dev_end", type=int, default=200)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--union_cap", type=int, default=60)
    parser.add_argument("--answer_cap", type=int, default=20)
    parser.add_argument("--max_heuristic_answers", type=int, default=160)
    parser.add_argument("--single_reader_docs", type=int, default=12)
    parser.add_argument("--pair_reader_pairs", type=int, default=5)
    parser.add_argument("--wrong_candidates_per_query", type=int, default=3)
    parser.add_argument("--proof_pool_k", type=int, default=20)
    parser.add_argument("--proof_threshold", type=float, default=0.70)
    parser.add_argument("--reader_batch_size", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_target_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nli_batch_size", type=int, default=32)
    parser.add_argument("--nli_max_length", type=int, default=384)
    parser.add_argument("--entailment_class_index", type=int, default=1)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--recall5_gate", type=float, default=0.85)
    parser.add_argument("--oracle_auc_gate", type=float, default=0.70)
    parser.add_argument("--best_single_pool_f1", type=float, default=0.5274)
    parser.add_argument("--rank_support_complete", type=float, default=0.72)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--no_reader_answers", action="store_true")
    parser.add_argument("--stop_after_candidate_recall", action="store_true")
    parser.add_argument("--stop_after_oracle_proof", action="store_true")
    parser.add_argument("--output_dir", default="reports/caps")
    parser.add_argument("--cache_dir", default="data/dpathrag/cache/caps")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(f"CAPS Day1 decision: {report.get('decision')}")


if __name__ == "__main__":
    main()
