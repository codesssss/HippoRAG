#!/usr/bin/env python3
"""CAPS Day-2 oracle-obligation proof-ranker diagnostic.

This diagnostic asks whether NLI proof scoring can rerank a broad top-20
candidate answer set. It uses oracle/template obligations derived from 2Wiki
evidence triples, so it is a mechanism upper-bound diagnostic, not a non-oracle
CAPS result.
"""

from __future__ import annotations

import argparse
import hashlib
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

from caps_day0_nli_sanity import auc_score, verbalize_evidence_triple  # noqa: E402
from caps_day1_5_candidate_v2 import LlmCandidateCache, candidate_rows_for_record, load_reader_probe  # noqa: E402
from caps_day1_gates import (  # noqa: E402
    answer_recall_at,
    dedup_union_records,
    proof_score_from_matrix,
    write_markdown,
)
from dpathrag_analyze_hard_negatives import load_jsonl, row_qid  # noqa: E402
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import gold_answers, normalize_answer  # noqa: E402


def sha1_text(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()


class NliScorer:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int,
        max_length: int,
        entailment_class_index: int,
        local_files_only: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=bool(local_files_only))
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=bool(local_files_only))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.entailment_index = self._entailment_index(entailment_class_index)

    def _entailment_index(self, fallback: int) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                return int(idx)
        return int(fallback)

    def score_pairs(self, pairs: Sequence[dict[str, str]]) -> list[float]:
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = list(pairs[start : start + self.batch_size])
                encoded = self.tokenizer(
                    [row["premise"] for row in batch],
                    [row["hypothesis"] for row in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                probs = self.torch.nn.functional.softmax(logits, dim=-1)
                scores.extend(float(value) for value in probs[:, self.entailment_index].detach().cpu().tolist())
        return scores


def evidence_answer_positions(record: dict[str, Any]) -> set[int]:
    gold_norms = {normalize_answer(answer) for answer in gold_answers(record) if normalize_answer(answer)}
    positions = set()
    for triple in list(record.get("evidences") or []):
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        if normalize_answer(triple[0]) in gold_norms:
            positions.add(0)
        if normalize_answer(triple[2]) in gold_norms:
            positions.add(2)
    return positions


def oracle_candidate_obligations(record: dict[str, Any], candidate_answer: str) -> tuple[list[str], bool]:
    """Build oracle/template obligations conditioned on a candidate answer.

    If the gold answer appears as a subject or object in a gold evidence triple,
    replace that field with the candidate answer. This is oracle because the
    evidence triples are gold annotations; it is intended only as a mechanism
    upper bound.
    """
    gold_norms = {normalize_answer(answer) for answer in gold_answers(record) if normalize_answer(answer)}
    candidate = str(candidate_answer or "").strip()
    candidate_norm = normalize_answer(candidate)
    obligations: list[str] = []
    substituted = False
    for triple in list(record.get("evidences") or []):
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        new_triple = [triple[0], triple[1], triple[2]]
        for pos in (0, 2):
            if normalize_answer(new_triple[pos]) in gold_norms and candidate_norm:
                if candidate_norm not in gold_norms:
                    new_triple[pos] = candidate
                substituted = True
        try:
            obligations.append(verbalize_evidence_triple(new_triple))
        except ValueError:
            continue
    return obligations, substituted


def proof_docs_for_record(record: dict[str, Any], dense_record: dict[str, Any] | None, *, pool_k: int, proof_pool_k: int, doc_chars: int) -> list[dict[str, str]]:
    docs = dedup_union_records([row for row in [record, dense_record] if row is not None], pool_k=int(pool_k), cap=int(proof_pool_k))
    output = []
    for doc in docs:
        title = str(doc.get("title") or "")
        text = str(doc.get("text") or "")
        output.append({"title": title, "text": f"{title}\n{text[: int(doc_chars)]}".strip()})
    return output


def build_candidates_for_dev(args: argparse.Namespace, prop_rows: Sequence[dict[str, Any]], dense_by_qid: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not hasattr(args, "no_reader_answers"):
        setattr(args, "no_reader_answers", bool(args.no_reader_v1))
    cache = LlmCandidateCache(Path(args.cache_dir) / "caps_day1_5_llm_candidates.jsonl")
    probe = load_reader_probe(args)
    by_qid = {}
    for record in prop_rows:
        qid = row_qid(record)
        candidates, _ = candidate_rows_for_record(record, dense_by_qid.get(qid), args=args, llm_cache=cache, probe=probe)
        by_qid[qid] = candidates[: int(args.answer_cap)]
    return by_qid


def build_nli_work(
    prop_rows: Sequence[dict[str, Any]],
    dense_by_qid: dict[str, dict[str, Any]],
    candidates_by_qid: dict[str, list[dict[str, Any]]],
    *,
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, int], list[dict[str, Any]]]:
    pair_index: dict[str, int] = {}
    pairs: list[dict[str, str]] = []
    tasks: list[dict[str, Any]] = []
    for record in prop_rows:
        qid = row_qid(record)
        docs = proof_docs_for_record(
            record,
            dense_by_qid.get(qid),
            pool_k=int(args.pool_k),
            proof_pool_k=int(args.proof_pool_k),
            doc_chars=int(args.proof_doc_chars),
        )
        for cand_idx, candidate in enumerate(candidates_by_qid.get(qid, [])[: int(args.answer_cap)]):
            obligations, substituted = oracle_candidate_obligations(record, str(candidate.get("text") or ""))
            keys = []
            for obligation in obligations:
                row_keys = []
                for doc_idx, doc in enumerate(docs):
                    key = json.dumps(
                        {
                            "qid": qid,
                            "obligation": obligation,
                            "doc_title": doc["title"],
                            "doc_hash": sha1_text(doc["text"]),
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    if key not in pair_index:
                        pair_index[key] = len(pairs)
                        pairs.append({"premise": doc["text"], "hypothesis": obligation})
                    row_keys.append(pair_index[key])
                keys.append(row_keys)
            tasks.append(
                {
                    "qid": qid,
                    "candidate_index": cand_idx,
                    "candidate": candidate,
                    "obligations": obligations,
                    "substituted": substituted,
                    "pair_indices": keys,
                    "doc_count": len(docs),
                }
            )
    return pairs, pair_index, tasks


def score_tasks(tasks: Sequence[dict[str, Any]], pair_scores: Sequence[float]) -> list[dict[str, Any]]:
    outputs = []
    for task in tasks:
        matrix = []
        for row in task["pair_indices"]:
            matrix.append([float(pair_scores[idx]) for idx in row])
        proof_score = proof_score_from_matrix(matrix)
        outputs.append({**task, "proof_score": proof_score})
    return outputs


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or int(rank) <= 0:
        return 0.0
    return 1.0 / float(rank)


def summarize_by_type(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("type") or "unknown"), []).append(row)
    return {key: summarize_rows(value) for key, value in sorted(buckets.items())}


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    denom = max(1, len(rows))
    present = [row for row in rows if row["gold_present"]]
    present_denom = max(1, len(present))
    return {
        "rows": len(rows),
        "candidate_recall_at_20": round(sum(float(row["gold_present"]) for row in rows) / denom, 6),
        "top1_accuracy_all": round(sum(float(row["gold_rank"] == 1) for row in rows) / denom, 6),
        "top3_accuracy_all": round(sum(float(row["gold_rank"] is not None and row["gold_rank"] <= 3) for row in rows) / denom, 6),
        "top1_accuracy_cond_gold_present": round(sum(float(row["gold_rank"] == 1) for row in present) / present_denom, 6),
        "top3_accuracy_cond_gold_present": round(sum(float(row["gold_rank"] is not None and row["gold_rank"] <= 3) for row in present) / present_denom, 6),
        "mrr_cond_gold_present": round(sum(reciprocal_rank(row["gold_rank"]) for row in present) / present_denom, 6),
        "answer_conditioned_rate": round(sum(float(row["answer_conditioned_available"]) for row in rows) / denom, 6),
        "substituted_rate": round(sum(float(row["gold_substituted"]) for row in rows) / denom, 6),
    }


def evaluate_rankings(prop_rows: Sequence[dict[str, Any]], scored: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_qid.setdefault(str(row["qid"]), []).append(row)
    details = []
    gold_scores = []
    best_wrong_scores = []
    for record in prop_rows:
        qid = row_qid(record)
        gold_norms = {normalize_answer(answer) for answer in gold_answers(record) if normalize_answer(answer)}
        rows = sorted(by_qid.get(qid, []), key=lambda item: (-float(item["proof_score"]), int(item["candidate_index"])))
        gold_rows = [row for row in rows if normalize_answer(row["candidate"].get("text")) in gold_norms]
        wrong_rows = [row for row in rows if normalize_answer(row["candidate"].get("text")) not in gold_norms]
        gold_rank = None
        if gold_rows:
            gold_ids = {id(row) for row in gold_rows}
            for rank, row in enumerate(rows, start=1):
                if id(row) in gold_ids:
                    gold_rank = rank
                    break
            gold_scores.append(max(float(row["proof_score"]) for row in gold_rows))
            if wrong_rows:
                best_wrong_scores.append(max(float(row["proof_score"]) for row in wrong_rows))
        details.append(
            {
                "qid": qid,
                "type": str(record.get("type") or ""),
                "question": str(record.get("question") or ""),
                "gold_answers": gold_answers(record),
                "gold_present": bool(gold_rows),
                "gold_rank": gold_rank,
                "answer_conditioned_available": bool(evidence_answer_positions(record)),
                "gold_substituted": any(bool(row.get("substituted")) for row in gold_rows),
                "best_answer": str(rows[0]["candidate"].get("text") or "") if rows else "",
                "best_score": round(float(rows[0]["proof_score"]), 6) if rows else 0.0,
                "gold_best_score": round(max(float(row["proof_score"]) for row in gold_rows), 6) if gold_rows else 0.0,
                "best_wrong_score": round(max(float(row["proof_score"]) for row in wrong_rows), 6) if wrong_rows else 0.0,
                "top5": [
                    {
                        "answer": str(row["candidate"].get("text") or ""),
                        "score": round(float(row["proof_score"]), 6),
                        "is_gold": normalize_answer(row["candidate"].get("text")) in gold_norms,
                        "substituted": bool(row.get("substituted")),
                    }
                    for row in rows[:5]
                ],
            }
        )
    summary = summarize_rows(details)
    if gold_scores and best_wrong_scores:
        summary["gold_vs_best_wrong_auc"] = round(auc_score(gold_scores, best_wrong_scores), 6)
        summary["mean_gold_score"] = round(sum(gold_scores) / len(gold_scores), 6)
        summary["mean_best_wrong_score"] = round(sum(best_wrong_scores) / len(best_wrong_scores), 6)
        summary["paired_gold_win_rate"] = round(sum(float(g > w) for g, w in zip(gold_scores, best_wrong_scores)) / max(1, len(best_wrong_scores)), 6)
    else:
        summary.update({"gold_vs_best_wrong_auc": 0.0, "mean_gold_score": 0.0, "mean_best_wrong_score": 0.0, "paired_gold_win_rate": 0.0})
    summary["by_type"] = summarize_by_type(details)
    return summary, details


def decision_from_summary(summary: dict[str, Any]) -> str:
    cond_top1 = float(summary.get("top1_accuracy_cond_gold_present") or 0.0)
    cond_top3 = float(summary.get("top3_accuracy_cond_gold_present") or 0.0)
    all_top1 = float(summary.get("top1_accuracy_all") or 0.0)
    if all_top1 >= 0.55 or cond_top1 >= 0.55 or cond_top3 >= 0.70:
        return "PROCEED_DAY3_NON_ORACLE_OBLIGATIONS"
    return "STOP_CAPS_PROOF_RANKER_FAIL"


def write_report(payload: dict[str, Any], details: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    write_json({"summary": payload["summary"], "payload": payload, "rows": list(details)}, out_dir / "caps_day2_proof_separability.json")
    write_jsonl(details, out_dir / "caps_day2_proof_separability.rows.jsonl")
    summary = payload["summary"]
    lines = [
        "# CAPS Day-2 Oracle-Obligation Proof Separability",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Candidate recall@20: `{summary['candidate_recall_at_20']}`",
        f"- Top1 accuracy all: `{summary['top1_accuracy_all']}`",
        f"- Top3 accuracy all: `{summary['top3_accuracy_all']}`",
        f"- Top1 accuracy conditional gold present: `{summary['top1_accuracy_cond_gold_present']}`",
        f"- Top3 accuracy conditional gold present: `{summary['top3_accuracy_cond_gold_present']}`",
        f"- MRR conditional gold present: `{summary['mrr_cond_gold_present']}`",
        f"- Gold-vs-best-wrong AUC: `{summary['gold_vs_best_wrong_auc']}`",
        f"- Paired gold win rate: `{summary['paired_gold_win_rate']}`",
        f"- Mean gold proof score: `{summary['mean_gold_score']}`",
        f"- Mean best wrong proof score: `{summary['mean_best_wrong_score']}`",
        f"- Answer-conditioned available rate: `{summary['answer_conditioned_rate']}`",
        f"- Gold substituted rate: `{summary['substituted_rate']}`",
        f"- Unique NLI pairs: `{payload['nli_pairs']}`",
        "",
        "## Gate",
        "",
        "- Proceed if all-query top1 >= 0.55, conditional top1 >= 0.55, or conditional top3 >= 0.70.",
        "- Otherwise stop CAPS proof search.",
        "",
        "## Per-Type Breakdown",
        "",
        "| Type | Rows | Recall@20 | Top1 All | Top3 All | Top1 Cond | Top3 Cond | MRR Cond | Answer-Cond Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for qtype, row in summary.get("by_type", {}).items():
        lines.append(
            f"| {qtype} | {row['rows']} | {row['candidate_recall_at_20']} | {row['top1_accuracy_all']} | "
            f"{row['top3_accuracy_all']} | {row['top1_accuracy_cond_gold_present']} | {row['top3_accuracy_cond_gold_present']} | "
            f"{row['mrr_cond_gold_present']} | {row['answer_conditioned_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This is an oracle-obligation mechanism diagnostic because obligations are derived from gold 2Wiki evidence triples.",
            "A pass would justify testing non-oracle obligation generation; it is not itself a deployable CAPS result.",
        ]
    )
    write_markdown(lines, out_dir / "caps_day2_proof_separability.md")


def run(args: argparse.Namespace) -> dict[str, Any]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prop_rows = load_jsonl(args.proprag_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))[int(args.dev_start) : int(args.dev_end)]
    dense_by_qid = {row_qid(row): row for row in load_jsonl(args.dense_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))}
    candidates_by_qid = build_candidates_for_dev(args, prop_rows, dense_by_qid)
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
    summary, details = evaluate_rankings(prop_rows, scored)
    decision = decision_from_summary(summary)
    payload = {
        "decision": decision,
        "summary": summary,
        "nli_pairs": len(pairs),
        "tasks": len(tasks),
        "config": {
            "answer_cap": int(args.answer_cap),
            "proof_pool_k": int(args.proof_pool_k),
            "pool_k": int(args.pool_k),
            "nli_model": str(args.nli_model),
        },
    }
    write_report(payload, details, args)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proprag_cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--dense_cache_jsonl", default="data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl")
    parser.add_argument("--reader_model", default="data/dpathrag/models/flan_t5_base_gold_k5_5k")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--llm_endpoint", default="http://localhost:8043/v1")
    parser.add_argument("--llm_model", default="qwen3-8b-train")
    parser.add_argument("--dev_start", type=int, default=0)
    parser.add_argument("--dev_end", type=int, default=200)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--union_cap", type=int, default=60)
    parser.add_argument("--llm_docs", type=int, default=30)
    parser.add_argument("--doc_chars", type=int, default=500)
    parser.add_argument("--proof_doc_chars", type=int, default=1200)
    parser.add_argument("--answer_cap", type=int, default=20)
    parser.add_argument("--proof_pool_k", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--include_v1", action="store_true", default=True)
    parser.add_argument("--include_string_extract", action="store_true", default=True)
    parser.add_argument("--no_reader_v1", action="store_true")
    parser.add_argument("--max_heuristic_answers", type=int, default=240)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--single_reader_docs", type=int, default=12)
    parser.add_argument("--pair_reader_pairs", type=int, default=5)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_target_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nli_batch_size", type=int, default=64)
    parser.add_argument("--nli_max_length", type=int, default=384)
    parser.add_argument("--entailment_class_index", type=int, default=1)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--output_dir", default="reports/caps")
    parser.add_argument("--cache_dir", default="data/dpathrag/cache/caps")
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(f"CAPS Day2 decision: {payload['decision']}")


if __name__ == "__main__":
    main()
