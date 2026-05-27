#!/usr/bin/env python3
"""Audit ETv3 native full1000 retrieval/readout failures.

This is a diagnostic-only script. It reads the completed ETv3 variable-flow
full1000 retrieval and QA reports, then separates candidate-boundary failures
from top-k composition/readout failures and reader failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_RUN_ROOT = Path("run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510")
DEFAULT_REPORT_DIR = Path("reports/etv3_native_full1000_audit_20260510")
DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
K_VALUES = (5, 10, 20, 50, 100, 200)
RANK_BUCKET_ORDER = ("rank6_10", "rank11_20", "rank21_50", "rank51_100", "rank101_200", "missing_200")


STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        output = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(output) or math.isinf(output):
        return default
    return output


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def unique_ints(values: Iterable[Any], *, limit: int | None = None) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if limit is not None and len(output) >= limit:
            break
    return output


def method_payload(qa_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    datasets = qa_payload.get("datasets", []) if isinstance(qa_payload, Mapping) else []
    if not datasets:
        return {}
    methods = datasets[0].get("methods", {}) if isinstance(datasets[0], Mapping) else {}
    if not isinstance(methods, Mapping) or not methods:
        return {}
    first_key = next(iter(methods))
    method = methods.get(first_key, {})
    return method if isinstance(method, Mapping) else {}


def qa_rows_by_query(path: Path) -> dict[int, Mapping[str, Any]]:
    method = method_payload(read_json(path))
    output: dict[int, Mapping[str, Any]] = {}
    for row in method.get("per_query", []) or []:
        if not isinstance(row, Mapping):
            continue
        qidx = safe_int(row.get("query_index"), default=-1)
        if qidx >= 0:
            output[qidx] = row
    return output


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_title(value).split()
        if len(token) > 2 and token not in STOP_TOKENS
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def doc_title(openie_docs: Sequence[Mapping[str, Any]], doc_id: int) -> str:
    if doc_id < 0 or doc_id >= len(openie_docs):
        return ""
    passage = str(openie_docs[doc_id].get("passage") or "")
    return passage.split("\n", 1)[0].strip()


def load_openie_docs(retrieval_payload: Mapping[str, Any], retrieval_path: Path) -> list[Mapping[str, Any]]:
    openie_path = Path(str(retrieval_payload.get("openie_path") or ""))
    if openie_path and not openie_path.is_absolute():
        openie_path = (retrieval_path.parent / openie_path).resolve()
    if not openie_path.exists():
        return []
    docs = read_json(openie_path).get("docs", []) or []
    return list(docs) if isinstance(docs, Sequence) else []


def candidate_docs(row: Mapping[str, Any]) -> list[int]:
    route_trace = row.get("route_trace", {}) or {}
    candidate_universe = route_trace.get("candidate_universe", {}) or {}
    top5 = unique_ints(row.get("retrieved_doc_indices_top5", []) or [], limit=5)
    candidates = unique_ints(candidate_universe.get("candidate_doc_indices", []) or [])
    if not candidates:
        candidates = unique_ints(candidate_universe.get("admissible_doc_indices", []) or [])
    return unique_ints([*top5, *candidates])


def recall_at(gold_docs: Sequence[int], docs: Sequence[int], k: int) -> float:
    if not gold_docs:
        return 0.0
    return len(set(gold_docs) & set(docs[:k])) / len(set(gold_docs))


def all_gold_at(gold_docs: Sequence[int], docs: Sequence[int], k: int) -> bool:
    if not gold_docs:
        return False
    return set(gold_docs).issubset(set(docs[:k]))


def any_gold_at(gold_docs: Sequence[int], docs: Sequence[int], k: int) -> bool:
    return bool(set(gold_docs) & set(docs[:k]))


def first_rank(doc_id: int, docs: Sequence[int]) -> int | None:
    for idx, item in enumerate(docs):
        if int(item) == int(doc_id):
            return idx + 1
    return None


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing_200"
    if rank <= 5:
        return "top5"
    if rank <= 10:
        return "rank6_10"
    if rank <= 20:
        return "rank11_20"
    if rank <= 50:
        return "rank21_50"
    if rank <= 100:
        return "rank51_100"
    return "rank101_200"


def max_rank_bucket(ranks: Sequence[int | None]) -> str:
    valid = [int(rank) for rank in ranks if rank is not None]
    if not valid:
        return "missing_200"
    return rank_bucket(max(valid))


def near_title_substitution(missing_titles: Sequence[str], selected_non_gold_titles: Sequence[str]) -> bool:
    missing_token_sets = [title_tokens(title) for title in missing_titles]
    for selected in selected_non_gold_titles:
        selected_tokens = title_tokens(selected)
        for missing_tokens in missing_token_sets:
            if jaccard(selected_tokens, missing_tokens) >= 0.5:
                return True
    return False


def dataset_paths(run_root: Path, dataset: str) -> tuple[Path, Path]:
    retrieval = (
        run_root
        / "runs"
        / dataset
        / dataset
        / "reports"
        / f"{dataset}_evidence_transition_graphragv3_variable_flow_retrieval.json"
    )
    qa = run_root / "reports" / f"{dataset}_evidence_transition_graphragv3_variable_flow_qa.json"
    return retrieval, qa


def bucket_name(*, gold_docs: Sequence[int], candidates: Sequence[int], qa_row: Mapping[str, Any]) -> str:
    if all_gold_at(gold_docs, candidates, 5):
        if safe_int(qa_row.get("ExactMatch"), default=0) == 1:
            return "top5_complete_answer_exact"
        return "top5_complete_reader_not_exact"
    if all_gold_at(gold_docs, candidates, 200):
        return "candidate200_complete_top5_incomplete"
    if any_gold_at(gold_docs, candidates, 5):
        return "candidate200_incomplete_top5_partial"
    if any_gold_at(gold_docs, candidates, 200):
        return "candidate200_incomplete_not_top5"
    return "candidate200_no_gold"


def summarize_rows(rows: Sequence[Mapping[str, Any]], k_values: Sequence[int] = K_VALUES) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    summary: dict[str, Any] = {
        "count": len(rows),
        "EM": mean(safe_float(row.get("ExactMatch")) for row in rows),
        "F1": mean(safe_float(row.get("F1")) for row in rows),
        "answer_hit_at5": mean(1.0 if row.get("answer_string_hit_at5") else 0.0 for row in rows),
        "mean_gold_count": mean(safe_float(row.get("gold_doc_count")) for row in rows),
    }
    for k in k_values:
        summary[f"all_gold_at{k}"] = mean(1.0 if row.get(f"all_gold_at{k}") else 0.0 for row in rows)
        summary[f"any_gold_at{k}"] = mean(1.0 if row.get(f"any_gold_at{k}") else 0.0 for row in rows)
        summary[f"recall_at{k}"] = mean(safe_float(row.get(f"recall_at{k}")) for row in rows)
    bucket_counts = Counter(str(row.get("failure_bucket")) for row in rows)
    summary["failure_bucket_counts"] = dict(sorted(bucket_counts.items()))
    gap_rows = [
        row
        for row in rows
        if str(row.get("failure_bucket")) == "candidate200_complete_top5_incomplete"
    ]
    max_rank_counts = Counter(str(row.get("max_missing_gold_rank_bucket")) for row in gap_rows)
    missing_doc_rank_counts: Counter[str] = Counter()
    for row in gap_rows:
        missing_doc_rank_counts.update(str(item) for item in row.get("missing_gold_rank_buckets", []) or [])
    summary["candidate200_complete_top5_incomplete"] = {
        "query_count": len(gap_rows),
        "max_missing_rank_bucket_counts": {
            bucket: max_rank_counts.get(bucket, 0) for bucket in RANK_BUCKET_ORDER
        },
        "missing_doc_rank_bucket_counts": {
            bucket: missing_doc_rank_counts.get(bucket, 0) for bucket in RANK_BUCKET_ORDER
        },
        "all_missing_within10": sum(1 for row in gap_rows if safe_int(row.get("max_missing_gold_rank_candidate200"), default=9999) <= 10),
        "all_missing_within20": sum(1 for row in gap_rows if safe_int(row.get("max_missing_gold_rank_candidate200"), default=9999) <= 20),
        "all_missing_within50": sum(1 for row in gap_rows if safe_int(row.get("max_missing_gold_rank_candidate200"), default=9999) <= 50),
    }
    return summary


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    all_query_rows: list[dict[str, Any]] = []
    missing_gold_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}

    for dataset in datasets:
        retrieval_path, qa_path = dataset_paths(run_root, dataset)
        retrieval_payload = read_json(retrieval_path)
        qa_by_query = qa_rows_by_query(qa_path)
        openie_docs = load_openie_docs(retrieval_payload, retrieval_path)
        query_rows: list[dict[str, Any]] = []

        for row in retrieval_payload.get("rows", []) or []:
            if not isinstance(row, Mapping):
                continue
            qidx = safe_int(row.get("query_index"), default=-1)
            qa_row = qa_by_query.get(qidx, {})
            gold_docs = unique_ints(row.get("gold_doc_indices", []) or [])
            selected_top5 = unique_ints(row.get("retrieved_doc_indices_top5", []) or [], limit=5)
            candidates = candidate_docs(row)
            selected_non_gold_titles = [
                doc_title(openie_docs, doc_id)
                for doc_id in selected_top5
                if doc_id not in set(gold_docs)
            ]
            missing_docs = [doc_id for doc_id in gold_docs if doc_id not in set(selected_top5)]
            missing_titles = [doc_title(openie_docs, doc_id) for doc_id in missing_docs]
            missing_ranks = [first_rank(doc_id, candidates) for doc_id in missing_docs]
            missing_rank_buckets = [rank_bucket(rank) for rank in missing_ranks]
            valid_missing_ranks = [int(rank) for rank in missing_ranks if rank is not None]
            query_record: dict[str, Any] = {
                "dataset": dataset,
                "query_index": qidx,
                "question": str(row.get("question") or qa_row.get("question") or ""),
                "gold_doc_count": len(gold_docs),
                "gold_doc_indices": gold_docs,
                "selected_top5": selected_top5,
                "candidate_count": len(candidates),
                "ExactMatch": safe_float(qa_row.get("ExactMatch")),
                "F1": safe_float(qa_row.get("F1")),
                "answer_string_hit_at5": bool(qa_row.get("answer_string_hit_at5")),
                "gold_count_at5": safe_float(qa_row.get("gold_count_at5")),
                "failure_bucket": bucket_name(gold_docs=gold_docs, candidates=candidates, qa_row=qa_row),
                "missing_gold_count_top5": len(missing_docs),
                "missing_gold_doc_indices_top5": missing_docs,
                "missing_gold_titles_top5": missing_titles,
                "missing_gold_ranks_candidate200": missing_ranks,
                "missing_gold_rank_buckets": missing_rank_buckets,
                "max_missing_gold_rank_candidate200": max(valid_missing_ranks) if valid_missing_ranks else None,
                "max_missing_gold_rank_bucket": max_rank_bucket(missing_ranks),
                "near_title_substitution_proxy": near_title_substitution(missing_titles, selected_non_gold_titles),
                "selected_non_gold_titles": selected_non_gold_titles,
            }
            for k in K_VALUES:
                query_record[f"all_gold_at{k}"] = all_gold_at(gold_docs, candidates, k)
                query_record[f"any_gold_at{k}"] = any_gold_at(gold_docs, candidates, k)
                query_record[f"recall_at{k}"] = recall_at(gold_docs, candidates, k)
            query_rows.append(query_record)

            for doc_id, title, rank, bucket in zip(missing_docs, missing_titles, missing_ranks, missing_rank_buckets):
                missing_gold_rows.append(
                    {
                        "dataset": dataset,
                        "query_index": qidx,
                        "gold_doc_count": len(gold_docs),
                        "missing_doc_id": doc_id,
                        "missing_title": title,
                        "candidate_rank": rank,
                        "rank_bucket": bucket,
                        "query_f1": query_record["F1"],
                        "query_em": query_record["ExactMatch"],
                        "failure_bucket": query_record["failure_bucket"],
                        "question": query_record["question"],
                        "selected_non_gold_titles": " || ".join(selected_non_gold_titles),
                    }
                )

        by_depth: dict[str, Any] = {}
        for depth in sorted({safe_int(row["gold_doc_count"]) for row in query_rows}):
            depth_rows = [row for row in query_rows if safe_int(row["gold_doc_count"]) == depth]
            by_depth[str(depth)] = summarize_rows(depth_rows)

        long_chain_rows = [row for row in query_rows if safe_int(row["gold_doc_count"]) >= 4]
        dataset_summaries[dataset] = {
            "paths": {
                "retrieval": str(retrieval_path),
                "qa": str(qa_path),
            },
            "overall": summarize_rows(query_rows),
            "by_gold_doc_count": by_depth,
            "long_chain_ge4": summarize_rows(long_chain_rows),
        }
        all_query_rows.extend(query_rows)

    return {
        "metadata": {
            "run_root": str(run_root),
            "datasets": datasets,
            "k_values": list(K_VALUES),
            "audit_type": "native_etv3_full1000_failure_audit",
            "notes": [
                "No LLM calls.",
                "No method-code changes.",
                "candidate200 means the method-owned ETv3 candidate universe, not PropRAG pool.",
            ],
        },
        "datasets": dataset_summaries,
        "query_rows": all_query_rows,
        "missing_gold_rows": missing_gold_rows,
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# ETv3 Native Full1000 Failure Audit",
        "",
        "Date: 2026-05-10",
        "",
        "This diagnostic audits the completed ETv3 variable-flow full1000 run as a",
        "native method-owned retrieval/readout line.  It does not use PropRAG pools,",
        "does not call an LLM, and does not modify ETv3/ETv4 method code.",
        "",
        "## Core Question",
        "",
        "For ETv3 failures, is the evidence missing from the method-owned candidate",
        "universe, present in candidate200 but not composed into top5, or already in",
        "top5 but not answered exactly by the reader?",
        "",
        "## Overall",
        "",
        "| Dataset | Count | EM | F1 | all_gold@5 | all_gold@20 | all_gold@100 | all_gold@200 | R@5 | R@20 | R@100 | R@200 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    datasets = payload.get("datasets", {}) or {}
    for dataset, info in datasets.items():
        overall = info["overall"]
        lines.append(
            "| {dataset} | {count} | {em} | {f1} | {ag5} | {ag20} | {ag100} | {ag200} | {r5} | {r20} | {r100} | {r200} |".format(
                dataset=dataset,
                count=overall["count"],
                em=fmt(overall["EM"]),
                f1=fmt(overall["F1"]),
                ag5=fmt(overall["all_gold_at5"]),
                ag20=fmt(overall["all_gold_at20"]),
                ag100=fmt(overall["all_gold_at100"]),
                ag200=fmt(overall["all_gold_at200"]),
                r5=fmt(overall["recall_at5"]),
                r20=fmt(overall["recall_at20"]),
                r100=fmt(overall["recall_at100"]),
                r200=fmt(overall["recall_at200"]),
            )
        )

    lines.extend(
        [
            "",
            "## R@k vs Set Completeness",
            "",
            "`R@k` measures average per-document support recall.  Multi-hop QA, however,",
            "needs the evidence set to be complete.  `all_gold@k` is therefore the",
            "stricter composition metric: all supporting documents must be present in",
            "the reader context.  The large `R@5 - all_gold@5` gap shows why pointwise",
            "recall can overstate long-chain retrieval quality.",
            "",
            "| Dataset | R@5 | all_gold@5 | R@5 - all_gold@5 | R@200 | all_gold@200 | R@200 - all_gold@200 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, info in datasets.items():
        overall = info["overall"]
        lines.append(
            "| {dataset} | {r5} | {ag5} | {gap5} | {r200} | {ag200} | {gap200} |".format(
                dataset=dataset,
                r5=fmt(overall["recall_at5"]),
                ag5=fmt(overall["all_gold_at5"]),
                gap5=fmt(overall["recall_at5"] - overall["all_gold_at5"]),
                r200=fmt(overall["recall_at200"]),
                ag200=fmt(overall["all_gold_at200"]),
                gap200=fmt(overall["recall_at200"] - overall["all_gold_at200"]),
            )
        )

    lines.extend(
        [
            "",
            "## Depth Breakdown",
            "",
            "| Dataset | Gold docs | Count | EM | F1 | all_gold@5 | all_gold@20 | all_gold@100 | all_gold@200 | R@5 | R@20 | R@100 | R@200 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, info in datasets.items():
        for depth, summary in sorted(info["by_gold_doc_count"].items(), key=lambda item: int(item[0])):
            lines.append(
                "| {dataset} | {depth} | {count} | {em} | {f1} | {ag5} | {ag20} | {ag100} | {ag200} | {r5} | {r20} | {r100} | {r200} |".format(
                    dataset=dataset,
                    depth=depth,
                    count=summary["count"],
                    em=fmt(summary["EM"]),
                    f1=fmt(summary["F1"]),
                    ag5=fmt(summary["all_gold_at5"]),
                    ag20=fmt(summary["all_gold_at20"]),
                    ag100=fmt(summary["all_gold_at100"]),
                    ag200=fmt(summary["all_gold_at200"]),
                    r5=fmt(summary["recall_at5"]),
                    r20=fmt(summary["recall_at20"]),
                    r100=fmt(summary["recall_at100"]),
                    r200=fmt(summary["recall_at200"]),
                )
            )

    lines.extend(
        [
            "",
            "## Selection Gap by Depth",
            "",
            "The selection gap is not unique to 4-doc examples; it grows with evidence",
            "depth.  This is the full1000 version of the Expand-then-Compose failure",
            "pattern: the candidate universe often contains the set, but the top5",
            "readout does not compose the full support set.",
            "",
            "| Dataset | Gold docs | Count | all_gold@5 | all_gold@200 | all_gold@200 - all_gold@5 | R@5 - all_gold@5 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, info in datasets.items():
        for depth, summary in sorted(info["by_gold_doc_count"].items(), key=lambda item: int(item[0])):
            lines.append(
                "| {dataset} | {depth} | {count} | {ag5} | {ag200} | {setgap} | {rsetgap} |".format(
                    dataset=dataset,
                    depth=depth,
                    count=summary["count"],
                    ag5=fmt(summary["all_gold_at5"]),
                    ag200=fmt(summary["all_gold_at200"]),
                    setgap=fmt(summary["all_gold_at200"] - summary["all_gold_at5"]),
                    rsetgap=fmt(summary["recall_at5"] - summary["all_gold_at5"]),
                )
            )

    lines.extend(
        [
            "",
            "## Failure Buckets",
            "",
            "Bucket definitions:",
            "",
            "- `top5_complete_answer_exact`: all gold docs are in top5 and reader exact match is correct.",
            "- `top5_complete_reader_not_exact`: all gold docs are in top5, but reader exact match is not correct.",
            "- `candidate200_complete_top5_incomplete`: all gold docs are in ETv3 candidate200, but not all are selected into top5.",
            "- `candidate200_incomplete_top5_partial`: at least one gold doc is in top5, but candidate200 still misses at least one gold doc.",
            "- `candidate200_incomplete_not_top5`: candidate200 has at least one gold doc, but top5 has none and candidate200 is incomplete.",
            "- `candidate200_no_gold`: candidate200 contains no gold support.",
            "",
            "| Dataset | Gold docs | Bucket | Count |",
            "|---|---:|---|---:|",
        ]
    )
    for dataset, info in datasets.items():
        for depth, summary in sorted(info["by_gold_doc_count"].items(), key=lambda item: int(item[0])):
            for bucket, count in sorted((summary.get("failure_bucket_counts") or {}).items()):
                lines.append(f"| {dataset} | {depth} | `{bucket}` | {count} |")

    lines.extend(
        [
            "",
            "## Rank Distribution Within Candidate-Complete Top5 Failures",
            "",
            "For `candidate200_complete_top5_incomplete` cases, the table buckets each",
            "query by the worst-ranked missing gold document in ETv3 candidate200.  If",
            "most rows fall in `rank6_10` or `rank11_20`, budget extension alone is a",
            "plausible fix.  If many rows require `rank51_100` or `rank101_200`, the",
            "problem is a deeper composition/selection objective, not just top-k budget.",
            "",
            "| Dataset | Gold docs | Queries | max 6-10 | max 11-20 | max 21-50 | max 51-100 | max 101-200 | all missing <=20 | all missing <=50 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, info in datasets.items():
        for depth, summary in sorted(info["by_gold_doc_count"].items(), key=lambda item: int(item[0])):
            gap = summary.get("candidate200_complete_top5_incomplete") or {}
            query_count = int(gap.get("query_count") or 0)
            if not query_count:
                continue
            max_counts = gap.get("max_missing_rank_bucket_counts") or {}
            lines.append(
                "| {dataset} | {depth} | {queries} | {r6} | {r11} | {r21} | {r51} | {r101} | {within20} | {within50} |".format(
                    dataset=dataset,
                    depth=depth,
                    queries=query_count,
                    r6=max_counts.get("rank6_10", 0),
                    r11=max_counts.get("rank11_20", 0),
                    r21=max_counts.get("rank21_50", 0),
                    r51=max_counts.get("rank51_100", 0),
                    r101=max_counts.get("rank101_200", 0),
                    within20=gap.get("all_missing_within20", 0),
                    within50=gap.get("all_missing_within50", 0),
                )
            )

    lines.extend(
        [
            "",
            "## MuSiQue Long-Chain Reading",
            "",
        ]
    )
    musique = datasets.get("musique", {})
    if musique:
        for depth in ("3", "4"):
            if depth in musique["by_gold_doc_count"]:
                summary = musique["by_gold_doc_count"][depth]
                lines.extend(
                    [
                        f"MuSiQue {depth}-doc:",
                        "",
                        f"- count: `{summary['count']}`",
                        f"- top5 all-gold: `{fmt(summary['all_gold_at5'])}`",
                        f"- candidate200 all-gold: `{fmt(summary['all_gold_at200'])}`",
                        f"- R@5 / R@200: `{fmt(summary['recall_at5'])}` / `{fmt(summary['recall_at200'])}`",
                        f"- EM/F1: `{fmt(summary['EM'])}` / `{fmt(summary['F1'])}`",
                        "",
                    ]
                )
        lines.extend(
            [
                "Interpretation:",
                "",
                "- If `candidate200_complete_top5_incomplete` dominates, ETv3 has a composition/readout problem.",
                "- If candidate200 all-gold is low, the candidate universe itself is a hard ceiling; a top5-only selector cannot fix those cases.",
                "- If `top5_complete_reader_not_exact` is large, the next bottleneck is reader/context rather than retrieval.",
                "",
                "This audit is not a DBEC residual audit.  A DBEC residual audit needs",
                "ETv3-pool + stable DBEC full1000 outputs first.",
            ]
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = build_audit(args)
    report_dir = Path(args.report_dir)
    write_json(payload, report_dir / "etv3_native_full1000_failure_audit.json")
    write_csv(payload["query_rows"], report_dir / "etv3_native_full1000_failure_queries.csv")
    write_csv(payload["missing_gold_rows"], report_dir / "etv3_native_full1000_missing_gold.csv")
    write_markdown(payload, report_dir / "etv3_native_full1000_failure_audit.md")
    print(json.dumps({dataset: info["overall"] for dataset, info in payload["datasets"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
