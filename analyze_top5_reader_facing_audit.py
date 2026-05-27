#!/usr/bin/env python3
"""Audit reader-facing top-5 evidence for STO transition variants.

This script intentionally avoids R@10 as an objective. It compares top-5
evidence sets for SFB-context, semantic residual pair, and
specificity-regularized pair using retrieval-side evidence proxies that can be
computed without running an LLM reader:

- partial R@5
- all-gold@5
- answer-string presence in the top-5 passages
- win/loss buckets against SFB and semantic residual pair

The output is a decision aid for whether to run expensive QA EM/F1 next.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from analyze_upstream_substrate_failures import SFB_VARIANT, load_json
from src.hipporag.utils.eval_utils import normalize_answer


DEFAULT_TRANSITION_REPORT = (
    "outputs_anchor_source_audit_limit100_20260423_v2/reports/"
    "transition_component_retriever_v28_all_native_denseanchor10_specificitypair_w12_limit100.json"
)

METHOD_DOC_KEYS = {
    "sfb_context": "sfb_doc_indices_top5",
    "semantic_pair": "semantic_pair_doc_indices_top5",
    "specificity_pair": "specificity_pair_doc_indices_top5",
}


def unique_ints(values: Iterable[Any], *, limit: int = 5) -> List[int]:
    seen: set[int] = set()
    result: List[int] = []
    for value in values:
        doc_idx = int(value)
        if doc_idx < 0 or doc_idx in seen:
            continue
        seen.add(doc_idx)
        result.append(doc_idx)
        if len(result) >= limit:
            break
    return result


def title_for_doc(openie_docs: Sequence[Mapping[str, Any]], doc_idx: int) -> str:
    if int(doc_idx) < 0 or int(doc_idx) >= len(openie_docs):
        return ""
    passage = str(openie_docs[int(doc_idx)].get("passage") or "")
    return passage.split("\n", 1)[0].strip()


def passage_for_doc(openie_docs: Sequence[Mapping[str, Any]], doc_idx: int) -> str:
    if int(doc_idx) < 0 or int(doc_idx) >= len(openie_docs):
        return ""
    return str(openie_docs[int(doc_idx)].get("passage") or "")


def answer_string_hit(gold_answers: Sequence[str], passages: Sequence[str]) -> bool:
    normalized_context = normalize_answer("\n".join(str(passage) for passage in passages))
    if not normalized_context:
        return False
    for answer in gold_answers:
        normalized_answer = normalize_answer(str(answer))
        if normalized_answer and normalized_answer in normalized_context:
            return True
    return False


def recall_at_5(gold_doc_indices: Sequence[int], top5_doc_indices: Sequence[int]) -> float:
    gold = {int(value) for value in gold_doc_indices}
    if not gold:
        return 0.0
    return float(len(gold & set(int(value) for value in top5_doc_indices[:5]))) / float(len(gold))


def evaluate_top5_row(
    *,
    row: Mapping[str, Any],
    method: str,
    doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    top5 = unique_ints(doc_indices, limit=5)
    gold = {int(value) for value in row.get("gold_doc_indices", []) or []}
    hit_gold = sorted(gold & set(top5))
    passages = [passage_for_doc(openie_docs, doc_idx) for doc_idx in top5]
    return {
        "method": method,
        "doc_indices_top5": top5,
        "titles_top5": [title_for_doc(openie_docs, doc_idx) for doc_idx in top5],
        "gold_count_at5": len(hit_gold),
        "recall_at5": recall_at_5(list(gold), top5),
        "any_gold_at5": bool(hit_gold),
        "all_gold_at5": bool(gold) and gold.issubset(set(top5)),
        "answer_string_hit_at5": answer_string_hit(row.get("gold_answers", []) or [], passages),
    }


def bucket_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    """Return left-vs-right top5 evidence bucket."""

    left_r5 = float(left["recall_at5"])
    right_r5 = float(right["recall_at5"])
    if left_r5 > right_r5:
        return "r5_improved"
    if left_r5 < right_r5:
        return "r5_degraded"
    if bool(left["all_gold_at5"]) and not bool(right["all_gold_at5"]):
        return "same_r5_gained_all_gold"
    if bool(right["all_gold_at5"]) and not bool(left["all_gold_at5"]):
        return "same_r5_lost_all_gold"
    if bool(left["answer_string_hit_at5"]) and not bool(right["answer_string_hit_at5"]):
        return "same_r5_gained_answer_string"
    if bool(right["answer_string_hit_at5"]) and not bool(left["answer_string_hit_at5"]):
        return "same_r5_lost_answer_string"
    return "unchanged_or_tie"


def aggregate_method_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    denom = float(max(len(rows), 1))
    return {
        "count": len(rows),
        "r5": round(sum(float(row["recall_at5"]) for row in rows) / denom, 6),
        "all_gold_at5": round(sum(1.0 for row in rows if row["all_gold_at5"]) / denom, 6),
        "any_gold_at5": round(sum(1.0 for row in rows if row["any_gold_at5"]) / denom, 6),
        "answer_string_hit_at5": round(sum(1.0 for row in rows if row["answer_string_hit_at5"]) / denom, 6),
        "mean_gold_count_at5": round(mean([float(row["gold_count_at5"]) for row in rows]) if rows else 0.0, 6),
    }


def load_sfb_rows(source_report_path: Path) -> Dict[int, Mapping[str, Any]]:
    source_report = load_json(source_report_path)
    sfb_rows = source_report.get("variants", {}).get(SFB_VARIANT)
    if sfb_rows is None:
        raise KeyError(f"SFB variant {SFB_VARIANT!r} not found in {source_report_path}")
    return {int(row["query_index"]): row for row in sfb_rows}


def audit_dataset(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    source_report_path = Path(str(dataset["report_path"]))
    openie_path = Path(str(dataset["openie_path"]))
    openie_docs = list(load_json(openie_path).get("docs", []) or [])
    sfb_rows = load_sfb_rows(source_report_path)

    method_rows: Dict[str, List[Dict[str, Any]]] = {method: [] for method in METHOD_DOC_KEYS}
    per_query: List[Dict[str, Any]] = []
    compare_to_sfb: Counter[str] = Counter()
    compare_to_semantic: Counter[str] = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = {
        "specificity_beats_sfb": [],
        "specificity_loses_to_sfb": [],
        "specificity_beats_semantic": [],
        "specificity_loses_to_semantic": [],
    }

    for row in dataset.get("rows", []) or []:
        query_index = int(row["query_index"])
        sfb_row = sfb_rows[query_index]
        merged_row = dict(row)
        merged_row["gold_answers"] = list(sfb_row.get("gold_answers", []) or [])

        doc_sets = {
            "sfb_context": unique_ints(
                sfb_row.get("reader_doc_indices_topk")
                or sfb_row.get("retrieved_doc_indices_top5")
                or [],
                limit=5,
            ),
            "semantic_pair": unique_ints(row.get("semantic_pair_doc_indices_top5", []) or [], limit=5),
            "specificity_pair": unique_ints(row.get("specificity_pair_doc_indices_top5", []) or [], limit=5),
        }
        evaluated = {
            method: evaluate_top5_row(
                row=merged_row,
                method=method,
                doc_indices=doc_indices,
                openie_docs=openie_docs,
            )
            for method, doc_indices in doc_sets.items()
        }
        for method, evaluated_row in evaluated.items():
            method_rows[method].append(evaluated_row)

        spec_vs_sfb = bucket_delta(evaluated["specificity_pair"], evaluated["sfb_context"])
        spec_vs_sem = bucket_delta(evaluated["specificity_pair"], evaluated["semantic_pair"])
        compare_to_sfb[spec_vs_sfb] += 1
        compare_to_semantic[spec_vs_sem] += 1

        query_record = {
            "query_index": query_index,
            "question": row.get("question"),
            "gold_answers": merged_row["gold_answers"],
            "gold_doc_indices": list(row.get("gold_doc_indices", []) or []),
            "methods": evaluated,
            "specificity_vs_sfb": spec_vs_sfb,
            "specificity_vs_semantic": spec_vs_sem,
        }
        per_query.append(query_record)

        if spec_vs_sfb == "r5_improved" and len(examples["specificity_beats_sfb"]) < 8:
            examples["specificity_beats_sfb"].append(query_record)
        if spec_vs_sfb == "r5_degraded" and len(examples["specificity_loses_to_sfb"]) < 8:
            examples["specificity_loses_to_sfb"].append(query_record)
        if spec_vs_sem == "r5_improved" and len(examples["specificity_beats_semantic"]) < 8:
            examples["specificity_beats_semantic"].append(query_record)
        if spec_vs_sem == "r5_degraded" and len(examples["specificity_loses_to_semantic"]) < 8:
            examples["specificity_loses_to_semantic"].append(query_record)

    return {
        "dataset": dataset["dataset"],
        "source_report_path": str(source_report_path),
        "openie_path": str(openie_path),
        "metrics": {method: aggregate_method_rows(rows) for method, rows in method_rows.items()},
        "specificity_vs_sfb_buckets": dict(compare_to_sfb),
        "specificity_vs_semantic_buckets": dict(compare_to_semantic),
        "examples": examples,
        "rows": per_query,
    }


def write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Top5 Reader-Facing Evidence Audit",
        "",
        "This audit treats R@5, all-gold@5, and answer-string presence in the reader-facing top5 as the primary retrieval-side proxies. R@10 is intentionally not used.",
        "",
        "## Metrics",
        "",
        "| dataset | method | R@5 | all-gold@5 | any-gold@5 | answer-string@5 | mean gold count@5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in summary.get("datasets", []) or []:
        for method, metrics in dataset.get("metrics", {}).items():
            lines.append(
                f"| {dataset['dataset']} | {method} | "
                f"{metrics.get('r5', 0.0):.4f} | "
                f"{metrics.get('all_gold_at5', 0.0):.4f} | "
                f"{metrics.get('any_gold_at5', 0.0):.4f} | "
                f"{metrics.get('answer_string_hit_at5', 0.0):.4f} | "
                f"{metrics.get('mean_gold_count_at5', 0.0):.4f} |"
            )

    lines.extend(["", "## Specificity-Pair Buckets", ""])
    for dataset in summary.get("datasets", []) or []:
        lines.append(f"### {dataset['dataset']}")
        lines.append("")
        lines.append("| comparison | bucket | count |")
        lines.append("| --- | --- | ---: |")
        for bucket, count in sorted(dataset.get("specificity_vs_sfb_buckets", {}).items()):
            lines.append(f"| vs SFB | {bucket} | {count} |")
        for bucket, count in sorted(dataset.get("specificity_vs_semantic_buckets", {}).items()):
            lines.append(f"| vs semantic | {bucket} | {count} |")
        lines.append("")

    lines.extend(["", "## Example Deltas", ""])
    for dataset in summary.get("datasets", []) or []:
        lines.append(f"### {dataset['dataset']}")
        lines.append("")
        for bucket_name, rows in dataset.get("examples", {}).items():
            lines.append(f"#### {bucket_name}")
            lines.append("")
            if not rows:
                lines.append("- none")
                lines.append("")
                continue
            for row in rows[:3]:
                spec = row["methods"]["specificity_pair"]
                sfb = row["methods"]["sfb_context"]
                sem = row["methods"]["semantic_pair"]
                lines.append(
                    f"- q{row['query_index']}: {row['question']} | "
                    f"gold={row['gold_doc_indices']} | "
                    f"SFB={sfb['doc_indices_top5']} | "
                    f"semantic={sem['doc_indices_top5']} | "
                    f"specificity={spec['doc_indices_top5']}"
                )
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_top5_reader_facing(
    *,
    transition_report_path: Path,
    output_json_path: Path,
    output_md_path: Path,
) -> Dict[str, Any]:
    transition_report = load_json(transition_report_path)
    datasets = [audit_dataset(dataset) for dataset in transition_report.get("datasets", []) or []]
    summary = {
        "transition_report_path": str(transition_report_path),
        "methods": list(METHOD_DOC_KEYS),
        "policy": "top5_only; R@10 is not an optimization objective",
        "datasets": datasets,
    }
    output_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, output_md_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit reader-facing top5 evidence for STO transition variants.")
    parser.add_argument("--transition-report", default=DEFAULT_TRANSITION_REPORT)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    summary = analyze_top5_reader_facing(
        transition_report_path=Path(args.transition_report).resolve(),
        output_json_path=Path(args.output_json).resolve(),
        output_md_path=Path(args.output_md).resolve(),
    )
    compact = {
        dataset["dataset"]: {
            "metrics": dataset["metrics"],
            "specificity_vs_sfb_buckets": dataset["specificity_vs_sfb_buckets"],
            "specificity_vs_semantic_buckets": dataset["specificity_vs_semantic_buckets"],
        }
        for dataset in summary["datasets"]
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
