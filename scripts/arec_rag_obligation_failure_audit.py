#!/usr/bin/env python3
"""Prepare a manual failure-mode audit packet for generated AREC obligations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.arec.obligations import claim_key, is_vague_claim
from src.dpathrag.arec.pool import pool_docs
from src.dpathrag.arec.smoke import gold_answers, load_jsonl_map, load_pool_records, qid_for
from src.dpathrag.data import normalize_text
from src.dpathrag.io import write_json, write_jsonl


LABEL_OPTIONS = [
    "correct_useful",
    "duplicate_or_paraphrase",
    "answer_leak",
    "vacuous",
    "wrong_entity_or_wrong_relation",
    "other",
]


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fixed_pool_map(path: str | Path, *, source: str, pool_n: int, mode: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not path or not Path(path).exists():
        return output
    for row in load_jsonl_rows(path):
        if row.get("source") == source and int(row.get("pool_n") or 0) == int(pool_n) and row.get("mode") == mode:
            output[str(row.get("qid"))] = row
    return output


def token_set(text: str) -> set[str]:
    return set(claim_key(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def anchor_count(text: str) -> int:
    """Approximate named-entity/date anchoring without adding NLP dependencies."""

    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9'’-]{2,}\b", text)
    numbers = re.findall(r"\b\d{2,4}\b", text)
    return len(set(capitalized + numbers))


def obligation_flags(claim: str, *, answer: str, other_claims: list[str]) -> list[str]:
    flags: list[str] = []
    claim_norm = normalize_text(claim)
    answer_norm = normalize_text(answer)
    if answer_norm and answer_norm in claim_norm:
        flags.append("answer_string_present")
    if is_vague_claim(claim) or anchor_count(claim) == 0:
        flags.append("vacuous_or_underanchored")
    current = token_set(claim)
    if any(jaccard(current, token_set(other)) >= 0.65 for other in other_claims if other != claim):
        flags.append("duplicate_or_paraphrase_candidate")
    return flags


def row_priority(row: dict[str, Any]) -> tuple[int, float]:
    delta_complete = float(row.get("projected_support_complete", 0.0)) - float(row.get("initial_support_complete", 0.0))
    delta_recall = float(row.get("projected_support_recall", 0.0)) - float(row.get("initial_support_recall", 0.0))
    if delta_complete < 0 or delta_recall < 0:
        bucket = 0
    elif delta_complete == 0 and delta_recall == 0:
        bucket = 1
    else:
        bucket = 2
    return bucket, delta_complete + delta_recall


def build_audit_rows(
    records: list[dict[str, Any]],
    generated_map: dict[str, dict[str, Any]],
    projection_map: dict[str, dict[str, Any]],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        qid = qid_for(record, idx)
        generated = generated_map.get(qid) or {}
        obligations = list(generated.get("obligations") or [])
        answer = str(generated.get("initial_answer") or "")
        docs = pool_docs(record, max_docs=5)
        projection = projection_map.get(qid) or {}
        claims = [str(item.get("claim") or "") for item in obligations]
        audited_obligations = []
        flag_counter: Counter[str] = Counter()
        for item in obligations:
            claim = str(item.get("claim") or "")
            flags = obligation_flags(claim, answer=answer, other_claims=claims)
            flag_counter.update(flags)
            audited_obligations.append(
                {
                    "id": item.get("id"),
                    "claim": claim,
                    "retrieval_active": bool(item.get("retrieval_active", True)),
                    "heuristic_flags": flags,
                    "manual_label": "",
                    "manual_notes": "",
                }
            )
        initial_titles = [doc.title for doc in docs]
        projected_titles = list(projection.get("selected_titles") or [])
        initial_complete = float(projection.get("initial_support_complete", 0.0))
        projected_complete = float(projection.get("support_complete", initial_complete))
        initial_recall = float(projection.get("initial_support_recall", 0.0))
        projected_recall = float(projection.get("support_recall", initial_recall))
        rows.append(
            {
                "qid": qid,
                "query_idx": record.get("query_idx", idx),
                "hop_bucket": str(len(record.get("gold_titles") or []) or record.get("hop_bucket") or "unknown"),
                "question": record.get("question"),
                "gold_answers": gold_answers(record),
                "gold_titles": list(record.get("gold_titles") or []),
                "initial_answer": answer,
                "initial_top5_titles": initial_titles,
                "generated_top20_projected_titles": projected_titles,
                "initial_support_recall": initial_recall,
                "projected_support_recall": projected_recall,
                "delta_support_recall": projected_recall - initial_recall,
                "initial_support_complete": initial_complete,
                "projected_support_complete": projected_complete,
                "delta_support_complete": projected_complete - initial_complete,
                "obligations": audited_obligations,
                "heuristic_flag_counts": dict(flag_counter),
                "manual_query_label": "",
                "manual_query_notes": "",
                "label_options": LABEL_OPTIONS,
            }
        )
    rows.sort(key=row_priority)
    return rows[: int(sample_size)]


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# AREC Generated Obligation Failure Audit",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Rows sampled: `{summary['sample_size']}`",
        f"- Sampling: generated top20 fixed-pool harms first, then unchanged, then improved.",
        f"- Label options: `{', '.join(LABEL_OPTIONS)}`",
        "",
        "## Aggregate Heuristic Flags",
        "",
        "| Flag | Count |",
        "|---|---:|",
    ]
    for flag, count in sorted((summary.get("heuristic_flag_counts") or {}).items()):
        lines.append(f"| {flag} | {count} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"### {idx}. qid={row['qid']} hop={row['hop_bucket']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold answers: `{row['gold_answers']}`",
                "",
                f"Gold titles: `{row['gold_titles']}`",
                "",
                f"Initial answer: `{row['initial_answer']}`",
                "",
                f"Initial top5: `{row['initial_top5_titles']}`",
                "",
                f"Generated top20 projection: `{row['generated_top20_projected_titles']}`",
                "",
                f"Support delta: recall {row['initial_support_recall']:.4f} -> {row['projected_support_recall']:.4f}; SC {row['initial_support_complete']:.4f} -> {row['projected_support_complete']:.4f}",
                "",
                "Manual query label: `TODO`",
                "",
                "| Obligation | Heuristic flags | Manual label |",
                "|---|---|---|",
            ]
        )
        for obligation in row["obligations"]:
            claim = str(obligation["claim"]).replace("|", "\\|")
            flags = ", ".join(obligation["heuristic_flags"]) or "none"
            lines.append(f"| {claim} | {flags} | TODO |")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--generated_obligations_jsonl", required=True)
    parser.add_argument("--fixed_pool_rows_jsonl", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--source", default="generated")
    parser.add_argument("--pool_n", type=int, default=20)
    parser.add_argument("--mode", default="greedy_only")
    parser.add_argument("--output_json", default="reports/dpathrag/arec_rag_obligation_failure_audit_musique_limit100_20260428.json")
    parser.add_argument("--output_jsonl", default="reports/dpathrag/arec_rag_obligation_failure_audit_musique_limit100_rows_20260428.jsonl")
    parser.add_argument("--output_md", default="reports/dpathrag/arec_rag_obligation_failure_audit_musique_limit100_20260428.md")
    args = parser.parse_args()

    records = load_pool_records(args.pool_json, int(args.limit))
    generated_map = load_jsonl_map(args.generated_obligations_jsonl)
    projection_map = fixed_pool_map(args.fixed_pool_rows_jsonl, source=args.source, pool_n=int(args.pool_n), mode=args.mode)
    rows = build_audit_rows(records, generated_map, projection_map, sample_size=int(args.sample_size))
    flag_counts: Counter[str] = Counter()
    for row in rows:
        flag_counts.update(row.get("heuristic_flag_counts") or {})
    summary = {
        "status": "completed",
        "dataset": args.dataset,
        "limit": int(args.limit),
        "sample_size": len(rows),
        "pool_json": args.pool_json,
        "generated_obligations_jsonl": args.generated_obligations_jsonl,
        "fixed_pool_rows_jsonl": args.fixed_pool_rows_jsonl,
        "projection_control": {"source": args.source, "pool_n": int(args.pool_n), "mode": args.mode},
        "label_options": LABEL_OPTIONS,
        "heuristic_flag_counts": dict(flag_counts),
        "note": "This file is an audit packet, not manual labels. Fill manual_label/manual_query_label before claiming failure-mode rates.",
    }
    write_json({"summary": summary, "rows": rows}, args.output_json)
    write_jsonl(rows, args.output_jsonl)
    write_markdown(rows, summary, args.output_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
