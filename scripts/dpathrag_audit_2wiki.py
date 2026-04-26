#!/usr/bin/env python3
"""Audit local 2Wiki data and fixed-pool compatibility for D-PathRAG."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from statistics import mean
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.data import extract_title, infer_split_status, load_2wiki_bundle, normalize_text, summarize_2wiki_samples, support_titles
from src.dpathrag.io import read_json, write_json
from src.dpathrag.metrics import recall_at_k, support_complete_at_k


def check_pool_compatibility(samples: list[dict[str, Any]], pool_path: Path, label: str) -> dict[str, Any]:
    if not pool_path.exists():
        return {"exists": False, "path": str(pool_path), "source": label, "notes": ["Pool file not found."]}

    payload = read_json(pool_path)
    records = list(payload.get("records") or [])
    compare_n = min(len(samples), len(records))
    question_mismatches = 0
    query_idx_mismatches = 0
    recall_5: list[float] = []
    recall_20: list[float] = []
    recall_100: list[float] = []
    complete_5: list[float] = []
    complete_20: list[float] = []
    complete_100: list[float] = []
    gold_title_mismatch = 0
    examples: list[dict[str, Any]] = []

    for idx in range(compare_n):
        sample = samples[idx]
        record = records[idx]
        sample_question = str(sample.get("question") or "")
        record_question = str(record.get("question") or "")
        if normalize_text(sample_question) != normalize_text(record_question):
            question_mismatches += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "idx": idx,
                        "sample_question": sample_question,
                        "pool_question": record_question,
                    }
                )
        if int(record.get("query_idx", idx)) != idx:
            query_idx_mismatches += 1
        sample_gold = support_titles(sample)
        record_gold = [str(item) for item in record.get("gold_titles") or []]
        if {normalize_text(item) for item in sample_gold} != {normalize_text(item) for item in record_gold}:
            gold_title_mismatch += 1
        pool_titles = list(record.get("pool_titles") or [extract_title(doc) for doc in record.get("pool_docs") or []])
        recall_5.append(recall_at_k(sample_gold, pool_titles, 5))
        recall_20.append(recall_at_k(sample_gold, pool_titles, 20))
        recall_100.append(recall_at_k(sample_gold, pool_titles, 100))
        complete_5.append(support_complete_at_k(sample_gold, pool_titles, 5))
        complete_20.append(support_complete_at_k(sample_gold, pool_titles, 20))
        complete_100.append(support_complete_at_k(sample_gold, pool_titles, 100))

    return {
        "exists": True,
        "path": str(pool_path),
        "source": label,
        "dataset": payload.get("dataset"),
        "limit": payload.get("limit"),
        "pool_k": payload.get("pool_k"),
        "records": len(records),
        "compared_records": compare_n,
        "question_mismatch_count": question_mismatches,
        "query_idx_mismatch_count": query_idx_mismatches,
        "gold_title_mismatch_count": gold_title_mismatch,
        "support_recall_at_5": round(float(mean(recall_5)) if recall_5 else 0.0, 4),
        "support_recall_at_20": round(float(mean(recall_20)) if recall_20 else 0.0, 4),
        "support_recall_at_100": round(float(mean(recall_100)) if recall_100 else 0.0, 4),
        "support_complete_at_5": round(float(mean(complete_5)) if complete_5 else 0.0, 4),
        "support_complete_at_20": round(float(mean(complete_20)) if complete_20 else 0.0, 4),
        "support_complete_at_100": round(float(mean(complete_100)) if complete_100 else 0.0, 4),
        "question_mismatch_examples": examples,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_2wiki_bundle(args.data_root, args.dataset)
    samples = bundle.samples[: int(args.limit)] if int(args.limit) > 0 else bundle.samples
    split = infer_split_status(args.data_root, args.dataset, len(bundle.samples))
    sample_summary = summarize_2wiki_samples(samples)
    pool_compatibility = {
        "dense": check_pool_compatibility(samples, Path(args.dense_pool_json), "dense"),
        "proprag": check_pool_compatibility(samples, Path(args.proprag_pool_json), "proprag"),
    }
    can_build_smoke_cache = any(item.get("exists") and item.get("question_mismatch_count") == 0 for item in pool_compatibility.values())
    recommendation = (
        "Build a D-PathRAG smoke cache from the local 1000-example subset, but acquire/prepare official 2Wiki train/dev before E2E training."
        if split["status"] == "local_eval_subset_only"
        else "Proceed to cache construction after verifying the detected train/dev split provenance."
    )
    return {
        "dataset": args.dataset,
        "local_samples_path": str(bundle.samples_path),
        "local_corpus_path": str(bundle.corpus_path),
        "sample_count": len(bundle.samples),
        "audited_sample_count": len(samples),
        "corpus_count": len(bundle.corpus),
        "split_status": split["status"],
        "has_train_dev_test": split["has_train_dev_test"],
        "split_files": split["split_files"],
        "field_coverage": sample_summary["field_coverage"],
        "type_distribution": sample_summary["type_distribution"],
        "support_stats": {
            "avg_supporting_facts": sample_summary["avg_supporting_facts"],
            "avg_support_titles": sample_summary["avg_support_titles"],
            "avg_evidence_path_len": sample_summary["avg_evidence_path_len"],
            "avg_context_docs": sample_summary["avg_context_docs"],
        },
        "examples": sample_summary["examples"],
        "pool_compatibility": pool_compatibility,
        "can_build_smoke_cache": bool(can_build_smoke_cache),
        "recommendation": recommendation,
        "notes": split["notes"]
        + [
            "D-PathRAG Phase 1 must use HuggingFace differentiable reader training; vLLM/Qwen is only for later hard-selection evaluation.",
            "The local audit does not validate full training feasibility unless an official train split is present.",
        ],
    }


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# D-PathRAG 2Wiki Split And Pool Audit",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{audit['dataset']}`",
        f"- Local samples: `{audit['sample_count']}` from `{audit['local_samples_path']}`",
        f"- Audited samples: `{audit['audited_sample_count']}`",
        f"- Local corpus: `{audit['corpus_count']}` from `{audit['local_corpus_path']}`",
        f"- Split status: `{audit['split_status']}`",
        f"- Has train/dev/test locally: `{audit['has_train_dev_test']}`",
        "",
        "## Schema",
        "",
    ]
    for field, coverage in audit["field_coverage"].items():
        lines.append(f"- `{field}` coverage: `{coverage}`")
    lines.extend(
        [
            "",
            "## Support / Evidence Stats",
            "",
            f"- Avg supporting facts: `{audit['support_stats']['avg_supporting_facts']}`",
            f"- Avg support titles: `{audit['support_stats']['avg_support_titles']}`",
            f"- Avg evidence path length: `{audit['support_stats']['avg_evidence_path_len']}`",
            f"- Avg context docs: `{audit['support_stats']['avg_context_docs']}`",
            "",
            "## Pool Compatibility",
            "",
        ]
    )
    for name, result in audit["pool_compatibility"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Exists: `{result.get('exists')}`",
                f"- Path: `{result.get('path')}`",
                f"- Records: `{result.get('records', 0)}`",
                f"- Question mismatches: `{result.get('question_mismatch_count', 0)}`",
                f"- Gold-title mismatches: `{result.get('gold_title_mismatch_count', 0)}`",
                f"- Support recall@5 / @20 / @100: `{result.get('support_recall_at_5', 0.0)}` / `{result.get('support_recall_at_20', 0.0)}` / `{result.get('support_recall_at_100', 0.0)}`",
                f"- Support complete@5 / @20 / @100: `{result.get('support_complete_at_5', 0.0)}` / `{result.get('support_complete_at_20', 0.0)}` / `{result.get('support_complete_at_100', 0.0)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            "",
            f"- Can build smoke cache: `{audit['can_build_smoke_cache']}`",
            f"- {audit['recommendation']}",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in audit.get("notes") or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--data_root", default="reproduce/dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dense_pool_json", default="run_logs/dense_pool_exports_full1000_20260424/2wikimultihopqa_dense_pool100.json")
    parser.add_argument("--proprag_pool_json", default="run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json")
    parser.add_argument("--json_out", default="reports/dpathrag/2wiki_split_audit.json")
    parser.add_argument("--md_out", default="reports/dpathrag/2wiki_split_audit.md")
    args = parser.parse_args()

    audit = build_audit(args)
    write_json(audit, args.json_out)
    write_markdown(audit, Path(args.md_out))
    print(f"Wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()

