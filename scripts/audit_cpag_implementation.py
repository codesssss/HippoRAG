#!/usr/bin/env python3
"""Audit CPAG implementation sanity before accepting CPAG failure conclusions."""

from __future__ import annotations

import argparse
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

from run_cpag_agreement import (  # noqa: E402
    build_doc_signals,
    canonical_doc_key,
    dedup_union_docs,
    load_jsonl,
    load_openie_index,
    norm_text,
    rank_fusion_select,
    row_qid,
    selected_titles,
    support_metrics,
)
from src.dpathrag.io import write_json  # noqa: E402


def doc_id_key(candidate: dict[str, Any]) -> str:
    value = candidate.get("doc_id")
    if value is None:
        return ""
    return str(value)


def title_key(candidate: dict[str, Any]) -> str:
    return norm_text(candidate.get("title"))


def manual_rrf_by_doc_key(
    prop_record: dict[str, Any],
    dense_record: dict[str, Any] | None,
    *,
    pool_k: int,
    k_const: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for record in [prop_record, dense_record]:
        if record is None:
            continue
        for rank, candidate in enumerate(list(record.get("candidates") or [])[: int(pool_k)], start=1):
            key = canonical_doc_key(candidate)
            scores[key] = scores.get(key, 0.0) + 1.0 / (int(k_const) + rank)
    return scores


def manual_rrf_indices_like_impl(union_docs: Sequence[dict[str, Any]], *, top_k: int, k_const: int = 60) -> list[int]:
    scored = []
    for idx, doc in enumerate(union_docs):
        score = 0.0
        for rank in (doc.get("pool_ranks") or {}).values():
            score += 1.0 / (int(k_const) + int(rank))
        scored.append((score, -int(doc.get("pool_count") or 0), int(doc.get("best_rank") or 10**9), idx))
    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [idx for _, _, _, idx in scored[: int(top_k)]]


def support_complete_manual(record: dict[str, Any], selected: Sequence[dict[str, Any]]) -> float:
    gold = {norm_text(title) for title in record.get("gold_titles") or [] if norm_text(title)}
    observed = {norm_text(doc.get("title")) for doc in selected if norm_text(doc.get("title"))}
    return float(bool(gold) and gold.issubset(observed))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    prop_rows = load_jsonl(args.proprag_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))[int(args.dev_start) : int(args.dev_end)]
    dense_rows = load_jsonl(args.dense_cache_jsonl, limit=max(int(args.dev_end), int(args.max_rows)))
    dense_by_qid = {row_qid(row): row for row in dense_rows}
    openie_index = load_openie_index(args.openie_json)
    sample_rows = prop_rows[: int(args.sample_queries)]
    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []
    overlap_doc_id_values = []
    overlap_title_values = []
    prop_top1_rrf_ranks = []

    for prop_record in sample_rows:
        qid = row_qid(prop_record)
        dense_record = dense_by_qid.get(qid)
        prop_candidates = list(prop_record.get("candidates") or [])[: int(args.pool_k)]
        dense_candidates = list((dense_record or {}).get("candidates") or [])[: int(args.pool_k)]
        prop_doc_ids = {doc_id_key(doc) for doc in prop_candidates if doc_id_key(doc)}
        dense_doc_ids = {doc_id_key(doc) for doc in dense_candidates if doc_id_key(doc)}
        prop_titles = {title_key(doc) for doc in prop_candidates if title_key(doc)}
        dense_titles = {title_key(doc) for doc in dense_candidates if title_key(doc)}
        doc_id_overlap = len(prop_doc_ids & dense_doc_ids)
        title_overlap = len(prop_titles & dense_titles)
        overlap_doc_id_values.append(doc_id_overlap)
        overlap_title_values.append(title_overlap)

        union_docs = dedup_union_docs(
            [("proprag", prop_record), ("dense", dense_record)],
            pool_k=int(args.pool_k),
            cap=int(args.pool_cap),
        )
        rrf_indices = rank_fusion_select(union_docs, top_k=int(args.top_k))
        manual_scores = manual_rrf_by_doc_key(prop_record, dense_record, pool_k=int(args.pool_k))
        impl_rrf_titles = selected_titles(union_docs, rrf_indices)
        manual_top_keys = sorted(manual_scores, key=lambda key: (-manual_scores[key], key))[: int(args.top_k)]
        manual_top_titles = []
        key_to_title = {canonical_doc_key(doc): str(doc.get("title") or "") for doc in union_docs}
        for key in manual_top_keys:
            manual_top_titles.append(key_to_title.get(key, key))

        prop_top1_key = canonical_doc_key(prop_candidates[0]) if prop_candidates else ""
        manual_rank = None
        if prop_top1_key:
            sorted_keys = sorted(manual_scores, key=lambda key: (-manual_scores[key], key))
            if prop_top1_key in sorted_keys:
                manual_rank = sorted_keys.index(prop_top1_key) + 1
                prop_top1_rrf_ranks.append(manual_rank)
                if manual_rank > 3:
                    warnings.append(f"qid={qid}: PropRAG top1 manual RRF rank is {manual_rank}; this can happen when cross-pool shared docs outrank PropRAG-only docs")

        prop_rank_docs = union_docs[:0]
        prop_rank_indices = [idx for idx, doc in enumerate(union_docs) if "proprag" in set(doc.get("pool_sources") or [])][: int(args.top_k)]
        prop_rank_docs = [union_docs[idx] for idx in prop_rank_indices]
        manual_complete = support_complete_manual(prop_record, prop_rank_docs)
        reported_complete = support_metrics(prop_record.get("gold_titles") or [], prop_rank_docs)["support_complete"]
        if manual_complete != reported_complete:
            failures.append(f"qid={qid}: support_complete mismatch manual={manual_complete} reported={reported_complete}")

        shared_doc = None
        for doc in union_docs:
            if len(doc.get("pool_sources") or []) >= 2:
                shared_doc = doc
                break
        cross_pool_count = None
        if shared_doc is not None:
            signal_rows = build_doc_signals(
                union_docs,
                question=str(prop_record.get("question") or ""),
                openie_index=openie_index,
                max_props=int(args.max_props),
            )
            matching = [row for row in signal_rows if row.get("doc_key") == shared_doc.get("doc_key")]
            cross_pool_count = int(matching[0].get("pool_count") or 0) if matching else None
            if cross_pool_count != 2:
                failures.append(f"qid={qid}: shared doc cross_pool_count={cross_pool_count}, expected 2")

        samples.append(
            {
                "qid": qid,
                "question": prop_record.get("question"),
                "prop_doc_id_count": len(prop_doc_ids),
                "dense_doc_id_count": len(dense_doc_ids),
                "doc_id_overlap_top20": doc_id_overlap,
                "title_overlap_top20": title_overlap,
                "prop_doc_id_sample": list(sorted(prop_doc_ids))[:5],
                "dense_doc_id_sample": list(sorted(dense_doc_ids))[:5],
                "prop_title_sample": [str(doc.get("title") or "") for doc in prop_candidates[:5]],
                "dense_title_sample": [str(doc.get("title") or "") for doc in dense_candidates[:5]],
                "union_size": len(union_docs),
                "cross_pool_docs": sum(1 for doc in union_docs if len(doc.get("pool_sources") or []) >= 2),
                "prop_top1_title": str(prop_candidates[0].get("title") or "") if prop_candidates else "",
                "prop_top1_manual_rrf_rank": manual_rank,
                "manual_rrf_top5_titles": manual_top_titles,
                "impl_rrf_top5_titles": impl_rrf_titles,
                "rrf_top5_match_manual": manual_top_titles == impl_rrf_titles,
                "prop_rank_top5_titles_from_union": [str(doc.get("title") or "") for doc in prop_rank_docs],
                "manual_support_complete_proprag": manual_complete,
                "reported_support_complete_proprag": reported_complete,
                "shared_doc_title": str(shared_doc.get("title") or "") if shared_doc else None,
                "shared_doc_pool_sources": list(shared_doc.get("pool_sources") or []) if shared_doc else [],
                "shared_doc_cross_pool_count_in_graph": cross_pool_count,
            }
        )
        if manual_top_titles != impl_rrf_titles:
            impl_like_indices = manual_rrf_indices_like_impl(union_docs, top_k=int(args.top_k))
            impl_like_titles = selected_titles(union_docs, impl_like_indices)
            if impl_like_titles != impl_rrf_titles:
                failures.append(f"qid={qid}: RRF formula/tie-break mismatch")
            else:
                warnings.append(f"qid={qid}: score-only manual RRF top5 differs from implementation due to tie-break ordering")

    summary = {
        "sample_queries": len(samples),
        "avg_doc_id_overlap_top20": round(sum(overlap_doc_id_values) / max(1, len(overlap_doc_id_values)), 4),
        "avg_title_overlap_top20": round(sum(overlap_title_values) / max(1, len(overlap_title_values)), 4),
        "prop_top1_rrf_rank_values": prop_top1_rrf_ranks,
        "all_checks_pass": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "warning_count": len(warnings),
        "warnings": warnings,
        "prop_top1_rrf_rank_gt3_count": sum(1 for rank in prop_top1_rrf_ranks if int(rank) > 3),
    }
    payload = {"summary": summary, "samples": samples}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(payload, out_dir / "cpag_implementation_audit.json")
    write_markdown(payload, out_dir / "cpag_implementation_audit.md")
    return payload


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    summary = payload["summary"]
    lines = [
        "# CPAG Implementation Audit",
        "",
        "## Summary",
        "",
        f"- Sample queries: `{summary['sample_queries']}`",
        f"- Avg doc_id overlap@20: `{summary['avg_doc_id_overlap_top20']}`",
        f"- Avg title overlap@20: `{summary['avg_title_overlap_top20']}`",
        f"- PropRAG top1 manual RRF ranks: `{summary['prop_top1_rrf_rank_values']}`",
        f"- All checks pass: `{summary['all_checks_pass']}`",
        f"- Failure count: `{summary['failure_count']}`",
        f"- Warning count: `{summary['warning_count']}`",
        f"- PropRAG top1 RRF rank > 3 count: `{summary['prop_top1_rrf_rank_gt3_count']}`",
        "",
        "## Failures",
        "",
    ]
    if summary["failures"]:
        for failure in summary["failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if summary["warnings"]:
        for warning in summary["warnings"][:50]:
            lines.append(f"- {warning}")
        if len(summary["warnings"]) > 50:
            lines.append(f"- ... truncated {len(summary['warnings']) - 50} additional warnings")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Sample Checks",
            "",
        ]
    )
    for sample in payload["samples"]:
        lines.extend(
            [
                f"### {sample['qid']}",
                "",
                f"- Question: {sample['question']}",
                f"- doc_id overlap@20: `{sample['doc_id_overlap_top20']}`",
                f"- title overlap@20: `{sample['title_overlap_top20']}`",
                f"- union size: `{sample['union_size']}`",
                f"- cross-pool docs: `{sample['cross_pool_docs']}`",
                f"- PropRAG top1: `{sample['prop_top1_title']}`",
                f"- PropRAG top1 manual RRF rank: `{sample['prop_top1_manual_rrf_rank']}`",
                f"- Manual RRF top5: `{sample['manual_rrf_top5_titles']}`",
                f"- Impl RRF top5: `{sample['impl_rrf_top5_titles']}`",
                f"- RRF top5 match manual: `{sample['rrf_top5_match_manual']}`",
                f"- PropRAG rank top5 from union: `{sample['prop_rank_top5_titles_from_union']}`",
                f"- support_complete manual/reported: `{sample['manual_support_complete_proprag']}` / `{sample['reported_support_complete_proprag']}`",
                f"- Shared doc: `{sample['shared_doc_title']}` sources `{sample['shared_doc_pool_sources']}` graph count `{sample['shared_doc_cross_pool_count_in_graph']}`",
                "",
            ]
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proprag_cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--dense_cache_jsonl", default="data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl")
    parser.add_argument("--openie_json", default="outputs/2wikimultihopqa/openie_results_ner_qwen3-8b.json")
    parser.add_argument("--dev_start", type=int, default=0)
    parser.add_argument("--dev_end", type=int, default=200)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--sample_queries", type=int, default=5)
    parser.add_argument("--pool_k", type=int, default=20)
    parser.add_argument("--pool_cap", type=int, default=60)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_props", type=int, default=8)
    parser.add_argument("--output_dir", default="reports/cpag")
    return parser.parse_args()


def main() -> None:
    payload = audit(parse_args())
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
