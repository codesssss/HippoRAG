#!/usr/bin/env python3
"""Audit DBEC repair candidate quality on under-selected fixed-pool slices.

This is a reader-free and LLM-free diagnostic.  It asks where SetR-missing
gold support titles are visible:

* in the source PropRAG pool,
* in rank-fill top-5,
* in DBEC binding candidates,
* on the DBEC greedy selection path,
* in the final DBEC-selected evidence.

The goal is to distinguish candidate-generation failures from scoring /
selection failures before attempting a deeper chain-walking binding redesign.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import (  # noqa: E402
    DATASETS,
    DBEC_SELECTIVE_DIR,
    REPORT_DIR as REPAIR_GATED_REPORT_DIR,
    SETR_FAITHFUL_DIR,
    SOURCE_POOL_DIR,
    SUPPORT_ROWS,
    TOP_K,
    extract_dbec_candidates,
    extract_dbec_final_order,
    extract_setr_seed_positions,
    list_from_json,
    normalize_title,
    rank_fill_order,
    read_json,
    safe_float,
    safe_int,
)


REPORT_DIR = Path("reports/repair_candidate_quality_audit_20260507")

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
    "list",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_support_rows() -> list[dict[str, Any]]:
    with SUPPORT_ROWS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_pool_path(dataset: str) -> Path:
    return SOURCE_POOL_DIR / f"{dataset}_pool100.json"


def setr_pool_path(dataset: str) -> Path:
    return SETR_FAITHFUL_DIR / f"{dataset}_proprag_setr_k20_doc768_faithful.selected_pool.json"


def dbec_report_path(dataset: str) -> Path:
    return DBEC_SELECTIVE_DIR / f"{dataset}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def repair_gated_audit_path() -> Path:
    return REPAIR_GATED_REPORT_DIR / "replacement_audit.csv"


def read_repair_gated_best_rows() -> dict[tuple[str, int], Mapping[str, Any]]:
    path = repair_gated_audit_path()
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    # Use the most conservative MuSiQue-selected variant if available.  This is
    # only for auxiliary admission diagnostics, not for candidate visibility.
    preferred = "rq1_mr1_g010_branchstrict"
    for row in rows:
        if str(row.get("variant")) != preferred:
            continue
        by_key[(str(row.get("dataset")), safe_int(row.get("query_index")))] = row
    return by_key


def title_positions(pool_titles: Sequence[Any]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for index, title in enumerate(pool_titles):
        norm = normalize_title(title)
        if norm:
            output.setdefault(norm, []).append(index)
    return output


def titles_at_positions(pool_titles: Sequence[Any], positions: Sequence[int]) -> list[str]:
    return [
        str(pool_titles[int(pos)])
        for pos in positions
        if 0 <= int(pos) < len(pool_titles)
    ]


def norm_set(titles: Sequence[Any]) -> set[str]:
    return {normalize_title(title) for title in titles if normalize_title(title)}


def positions_for_title(title: Any, positions_by_title: Mapping[str, list[int]]) -> list[int]:
    return [int(pos) for pos in positions_by_title.get(normalize_title(title), [])]


def token_set(title: Any) -> set[str]:
    tokens = set(normalize_title(title).split())
    return {token for token in tokens if token and token not in STOP_TOKENS}


def title_overlap(left: Any, right: Any) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def near_title_substitution(candidate_title: Any, gold_title: Any) -> bool:
    left = token_set(candidate_title)
    right = token_set(gold_title)
    if not left or not right:
        return False
    overlap = left & right
    if len(overlap) >= 2 and len(overlap) / max(1, min(len(left), len(right))) >= 0.5:
        return True
    return title_overlap(candidate_title, gold_title) >= 0.30 and len(overlap) >= 2


def duplicate_title_count(titles: Sequence[Any]) -> int:
    counts = Counter(normalize_title(title) for title in titles if normalize_title(title))
    return sum(count - 1 for count in counts.values() if count > 1)


def binding_candidate_entries(selector_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_requirement = selector_trace.get("binding_candidates_by_requirement")
    by_requirement = by_requirement if isinstance(by_requirement, Mapping) else {}
    for req_id, candidates in by_requirement.items():
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, Mapping):
                continue
            output.append({
                "requirement_id": str(req_id),
                "title": str(candidate.get("title") or ""),
                "title_pool_position": safe_int(candidate.get("title_pool_position"), default=-1),
                "dep": str(candidate.get("dep") or ""),
                "dep_position": safe_int(candidate.get("dep_position"), default=-1),
                "dep_score": safe_float(candidate.get("dep_score")),
                "llm_extracted_entity": str(candidate.get("llm_extracted_entity") or ""),
                "entity_match_type": str(candidate.get("entity_match_type") or ""),
            })
    return output


def binding_candidate_titles(selector_trace: Mapping[str, Any]) -> list[str]:
    return [entry["title"] for entry in binding_candidate_entries(selector_trace) if entry.get("title")]


def binding_candidate_positions(selector_trace: Mapping[str, Any]) -> list[int]:
    positions: list[int] = []
    seen: set[int] = set()
    for entry in binding_candidate_entries(selector_trace):
        pos = safe_int(entry.get("title_pool_position"), default=-1)
        if pos >= 0 and pos not in seen:
            seen.add(pos)
            positions.append(pos)
    return positions


def classify_missing_gold(
    *,
    source_has: bool,
    rank_has: bool,
    binding_has: bool,
    greedy_has: bool,
    dbec_has: bool,
) -> str:
    if not source_has:
        return "gold_absent_from_source_pool"
    if dbec_has:
        return "gold_recovered_by_dbec_final"
    if greedy_has:
        return "gold_on_dbec_greedy_path_not_final"
    if binding_has:
        return "gold_in_binding_candidates_not_selected"
    if rank_has:
        return "gold_in_rank_fill_not_dbec"
    return "gold_source_only_not_rank_or_dbec"


def load_artifacts(dataset: str) -> dict[str, Any]:
    return {
        "source_records": list(read_json(source_pool_path(dataset)).get("records") or []),
        "setr_records": list(read_json(setr_pool_path(dataset)).get("records") or []),
        "dbec_traces": list(read_json(dbec_report_path(dataset)).get("setwise_selector_query_traces") or []),
    }


def build_missing_gold_rows(
    *,
    dataset_label: str,
    dataset: str,
    support_row: Mapping[str, Any],
    source_record: Mapping[str, Any],
    setr_record: Mapping[str, Any],
    dbec_query_trace: Mapping[str, Any],
    repair_gated_row: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_titles = list(source_record.get("pool_titles") or [])
    pool_size = len(pool_titles)
    positions_by_title = title_positions(pool_titles)
    selector_trace = dbec_query_trace.get("selector_trace")
    selector_trace = selector_trace if isinstance(selector_trace, Mapping) else {}

    setr_positions = extract_setr_seed_positions(setr_record, pool_size)
    rank_positions = rank_fill_order(setr_positions, pool_size=pool_size, top_k=TOP_K)
    dbec_positions = extract_dbec_final_order(selector_trace, pool_size)
    dbec_candidates = extract_dbec_candidates(selector_trace, pool_titles)
    greedy_positions = [candidate.position for candidate in dbec_candidates]
    binding_positions = binding_candidate_positions(selector_trace)
    binding_titles = binding_candidate_titles(selector_trace)

    setr_titles = titles_at_positions(pool_titles, setr_positions)
    rank_titles = titles_at_positions(pool_titles, rank_positions)
    dbec_titles = titles_at_positions(pool_titles, dbec_positions)
    greedy_titles = titles_at_positions(pool_titles, greedy_positions)
    binding_pool_titles = titles_at_positions(pool_titles, binding_positions)

    missing_gold_titles = list_from_json(support_row.get("setr_missing_titles_json"))
    gold_titles = list_from_json(support_row.get("gold_titles_json"))
    source_norm = norm_set(pool_titles)
    rank_norm = norm_set(rank_titles)
    dbec_norm = norm_set(dbec_titles)
    greedy_norm = norm_set(greedy_titles)
    binding_norm = norm_set(binding_titles + binding_pool_titles)
    rank_added_titles = [
        title for pos, title in zip(rank_positions, rank_titles)
        if pos not in set(setr_positions)
    ]
    dbec_added_titles = [
        title for pos, title in zip(dbec_positions, dbec_titles)
        if pos not in set(setr_positions)
    ]
    gold_norm = norm_set(gold_titles)
    dbec_added_non_gold = [
        title for title in dbec_added_titles
        if normalize_title(title) not in gold_norm
    ]
    rank_added_gold = [
        title for title in rank_added_titles
        if normalize_title(title) in gold_norm
    ]
    dbec_added_gold = [
        title for title in dbec_added_titles
        if normalize_title(title) in gold_norm
    ]
    near_substitution_count = 0
    near_substitution_pairs: list[dict[str, str]] = []
    for candidate_title in dbec_added_non_gold:
        for gold_title in missing_gold_titles:
            if near_title_substitution(candidate_title, gold_title):
                near_substitution_count += 1
                near_substitution_pairs.append({
                    "dbec_title": str(candidate_title),
                    "missing_gold_title": str(gold_title),
                })

    missing_rows: list[dict[str, Any]] = []
    for gold_title in missing_gold_titles:
        norm = normalize_title(gold_title)
        source_positions = positions_for_title(gold_title, positions_by_title)
        source_has = norm in source_norm
        rank_has = norm in rank_norm
        dbec_has = norm in dbec_norm
        greedy_has = norm in greedy_norm
        binding_has = norm in binding_norm
        bucket = classify_missing_gold(
            source_has=source_has,
            rank_has=rank_has,
            binding_has=binding_has,
            greedy_has=greedy_has,
            dbec_has=dbec_has,
        )
        missing_rows.append({
            "dataset": dataset_label,
            "base_dataset": dataset,
            "query_index": safe_int(support_row.get("query_index")),
            "question": str(support_row.get("question") or source_record.get("question") or ""),
            "gold_title": str(gold_title),
            "bucket": bucket,
            "source_has_gold": int(source_has),
            "source_gold_positions_json": json.dumps(source_positions),
            "source_best_gold_rank": min(source_positions) if source_positions else "",
            "rank_fill5_has_gold": int(rank_has),
            "binding_candidates_have_gold": int(binding_has),
            "dbec_greedy_path_has_gold": int(greedy_has),
            "dbec_final_has_gold": int(dbec_has),
            "setr_selected_count": safe_int(support_row.get("setr_selected_count")),
            "gold_doc_count": safe_int(support_row.get("gold_doc_count")),
            "requirement_count": safe_int(support_row.get("requirement_count")),
            "dependent_req_count": safe_int(support_row.get("dependent_req_count")),
            "setr_f1": safe_float(support_row.get("setr_f1")),
            "dbec_f1": safe_float(support_row.get("dbec_f1")),
            "delta_f1_dbec_minus_setr": safe_float(support_row.get("delta_f1_dbec_minus_setr")),
            "rank_fill_positions_json": json.dumps(rank_positions),
            "rank_fill_titles_json": json.dumps(rank_titles, ensure_ascii=False),
            "dbec_final_positions_json": json.dumps(dbec_positions),
            "dbec_final_titles_json": json.dumps(dbec_titles, ensure_ascii=False),
            "dbec_greedy_path_positions_json": json.dumps(greedy_positions),
            "dbec_greedy_path_titles_json": json.dumps(greedy_titles, ensure_ascii=False),
            "binding_candidate_positions_json": json.dumps(binding_positions),
            "binding_candidate_titles_json": json.dumps(binding_titles, ensure_ascii=False),
        })

    query_bucket_counts = Counter(row["bucket"] for row in missing_rows)
    unrecovered_rows = [row for row in missing_rows if not safe_int(row.get("dbec_final_has_gold"))]
    primary_bucket = "all_missing_gold_recovered_by_dbec"
    if unrecovered_rows:
        priority = [
            "gold_absent_from_source_pool",
            "gold_in_binding_candidates_not_selected",
            "gold_on_dbec_greedy_path_not_final",
            "gold_in_rank_fill_not_dbec",
            "gold_source_only_not_rank_or_dbec",
        ]
        counts = Counter(row["bucket"] for row in unrecovered_rows)
        primary_bucket = max(
            counts,
            key=lambda bucket: (counts[bucket], -priority.index(bucket) if bucket in priority else -999),
        )

    query_row = {
        "dataset": dataset_label,
        "base_dataset": dataset,
        "query_index": safe_int(support_row.get("query_index")),
        "question": str(support_row.get("question") or source_record.get("question") or ""),
        "primary_bucket": primary_bucket,
        "missing_gold_count": len(missing_gold_titles),
        "source_has_all_missing_gold": int(all(safe_int(row["source_has_gold"]) for row in missing_rows)) if missing_rows else 0,
        "rank_fill5_recovered_missing_count": sum(safe_int(row["rank_fill5_has_gold"]) for row in missing_rows),
        "binding_candidate_recovered_missing_count": sum(safe_int(row["binding_candidates_have_gold"]) for row in missing_rows),
        "dbec_greedy_path_recovered_missing_count": sum(safe_int(row["dbec_greedy_path_has_gold"]) for row in missing_rows),
        "dbec_final_recovered_missing_count": sum(safe_int(row["dbec_final_has_gold"]) for row in missing_rows),
        "rank_added_gold_count": len(rank_added_gold),
        "dbec_added_gold_count": len(dbec_added_gold),
        "dbec_added_non_gold_count": len(dbec_added_non_gold),
        "dbec_replaced_rank_gold_count": max(0, len(rank_added_gold) - len([
            title for title in rank_added_gold if normalize_title(title) in dbec_norm
        ])),
        "dbec_final_duplicate_title_count": duplicate_title_count(dbec_titles),
        "near_title_substitution_count": near_substitution_count,
        "near_title_substitution_pairs_json": json.dumps(near_substitution_pairs, ensure_ascii=False),
        "bucket_counts_json": json.dumps(dict(query_bucket_counts), ensure_ascii=False, sort_keys=True),
        "gold_titles_json": json.dumps(gold_titles, ensure_ascii=False),
        "setr_missing_titles_json": json.dumps(missing_gold_titles, ensure_ascii=False),
        "setr_titles_json": json.dumps(setr_titles, ensure_ascii=False),
        "rank_fill_titles_json": json.dumps(rank_titles, ensure_ascii=False),
        "dbec_titles_json": json.dumps(dbec_titles, ensure_ascii=False),
        "dbec_greedy_path_titles_json": json.dumps(greedy_titles, ensure_ascii=False),
        "binding_candidate_titles_json": json.dumps(binding_titles, ensure_ascii=False),
        "setr_f1": safe_float(support_row.get("setr_f1")),
        "dbec_f1": safe_float(support_row.get("dbec_f1")),
        "delta_f1_dbec_minus_setr": safe_float(support_row.get("delta_f1_dbec_minus_setr")),
        "repair_gated_net_gold_delta_vs_rank": safe_int((repair_gated_row or {}).get("net_gold_delta_vs_rank")),
        "repair_gated_harmful_replacement": safe_int((repair_gated_row or {}).get("harmful_replacement")),
    }
    return missing_rows, query_row


def build_audit_rows(*, limit_per_dataset: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support_rows = read_support_rows()
    repair_gated_by_key = read_repair_gated_best_rows()
    missing_gold_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for config in DATASETS:
        dataset_label = str(config["label"])
        dataset = str(config["dataset"])
        selected_rows = [row for row in support_rows if config["predicate"](row)]
        selected_rows = sorted(selected_rows, key=lambda row: safe_int(row.get("query_index")))
        if limit_per_dataset > 0:
            selected_rows = selected_rows[: int(limit_per_dataset)]
        artifacts = load_artifacts(dataset)
        source_records = artifacts["source_records"]
        setr_records = artifacts["setr_records"]
        dbec_traces = artifacts["dbec_traces"]
        for support_row in selected_rows:
            query_index = safe_int(support_row.get("query_index"))
            missing_rows, query_row = build_missing_gold_rows(
                dataset_label=dataset_label,
                dataset=dataset,
                support_row=support_row,
                source_record=source_records[query_index],
                setr_record=setr_records[query_index],
                dbec_query_trace=dbec_traces[query_index],
                repair_gated_row=repair_gated_by_key.get((dataset_label, query_index)),
            )
            missing_gold_rows.extend(missing_rows)
            query_rows.append(query_row)
    return missing_gold_rows, query_rows


def mean_float(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(mean([safe_float(row.get(field)) for row in rows])) if rows else 0.0


def rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return mean_float(rows, field)


def build_bucket_summary(missing_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in missing_rows}):
        dataset_rows = [row for row in missing_rows if str(row.get("dataset")) == dataset]
        total = len(dataset_rows)
        for bucket, count in Counter(str(row.get("bucket")) for row in dataset_rows).most_common():
            bucket_rows = [row for row in dataset_rows if str(row.get("bucket")) == bucket]
            output.append({
                "dataset": dataset,
                "bucket": bucket,
                "missing_gold_titles": count,
                "missing_gold_title_rate": count / total if total else 0.0,
                "mean_setr_f1": mean_float(bucket_rows, "setr_f1"),
                "mean_dbec_f1": mean_float(bucket_rows, "dbec_f1"),
                "mean_delta_f1_dbec_minus_setr": mean_float(bucket_rows, "delta_f1_dbec_minus_setr"),
            })
    return output


def build_query_bucket_summary(query_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in query_rows}):
        dataset_rows = [row for row in query_rows if str(row.get("dataset")) == dataset]
        total = len(dataset_rows)
        for bucket, count in Counter(str(row.get("primary_bucket")) for row in dataset_rows).most_common():
            bucket_rows = [row for row in dataset_rows if str(row.get("primary_bucket")) == bucket]
            output.append({
                "dataset": dataset,
                "primary_bucket": bucket,
                "queries": count,
                "query_rate": count / total if total else 0.0,
                "mean_missing_gold_count": mean_float(bucket_rows, "missing_gold_count"),
                "mean_rank_fill5_recovered_missing_count": mean_float(bucket_rows, "rank_fill5_recovered_missing_count"),
                "mean_binding_candidate_recovered_missing_count": mean_float(bucket_rows, "binding_candidate_recovered_missing_count"),
                "mean_dbec_final_recovered_missing_count": mean_float(bucket_rows, "dbec_final_recovered_missing_count"),
                "mean_dbec_replaced_rank_gold_count": mean_float(bucket_rows, "dbec_replaced_rank_gold_count"),
                "near_title_substitution_rate": rate(bucket_rows, "near_title_substitution_count"),
                "mean_setr_f1": mean_float(bucket_rows, "setr_f1"),
                "mean_dbec_f1": mean_float(bucket_rows, "dbec_f1"),
                "mean_delta_f1_dbec_minus_setr": mean_float(bucket_rows, "delta_f1_dbec_minus_setr"),
            })
    return output


def percentile(values: Sequence[int], q: float) -> int | str:
    if not values:
        return ""
    sorted_values = sorted(int(value) for value in values)
    index = round((len(sorted_values) - 1) * float(q))
    return sorted_values[int(index)]


def build_rank_depth_summary(missing_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in missing_rows}):
        dataset_rows = [row for row in missing_rows if str(row.get("dataset")) == dataset]
        for bucket in sorted({str(row.get("bucket")) for row in dataset_rows}):
            bucket_rows = [row for row in dataset_rows if str(row.get("bucket")) == bucket]
            ranks = [
                safe_int(row.get("source_best_gold_rank"))
                for row in bucket_rows
                if row.get("source_best_gold_rank") is not None
                and str(row.get("source_best_gold_rank")) != ""
            ]
            output.append({
                "dataset": dataset,
                "bucket": bucket,
                "source_visible_missing_titles": len(ranks),
                "mean_source_best_gold_rank": float(mean(ranks)) if ranks else "",
                "p50_source_best_gold_rank": percentile(ranks, 0.50),
                "p75_source_best_gold_rank": percentile(ranks, 0.75),
                "p90_source_best_gold_rank": percentile(ranks, 0.90),
                "min_source_best_gold_rank": min(ranks) if ranks else "",
                "max_source_best_gold_rank": max(ranks) if ranks else "",
            })
    return output


def build_stage_summary(query_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in query_rows}):
        rows = [row for row in query_rows if str(row.get("dataset")) == dataset]
        total_missing = sum(safe_int(row.get("missing_gold_count")) for row in rows)
        output.append({
            "dataset": dataset,
            "queries": len(rows),
            "missing_gold_titles": total_missing,
            "source_has_all_missing_gold_rate": rate(rows, "source_has_all_missing_gold"),
            "rank_fill5_missing_gold_recovery_rate": (
                sum(safe_int(row.get("rank_fill5_recovered_missing_count")) for row in rows) / total_missing
                if total_missing else 0.0
            ),
            "binding_candidate_missing_gold_recovery_rate": (
                sum(safe_int(row.get("binding_candidate_recovered_missing_count")) for row in rows) / total_missing
                if total_missing else 0.0
            ),
            "dbec_greedy_path_missing_gold_recovery_rate": (
                sum(safe_int(row.get("dbec_greedy_path_recovered_missing_count")) for row in rows) / total_missing
                if total_missing else 0.0
            ),
            "dbec_final_missing_gold_recovery_rate": (
                sum(safe_int(row.get("dbec_final_recovered_missing_count")) for row in rows) / total_missing
                if total_missing else 0.0
            ),
            "mean_dbec_added_non_gold_count": mean_float(rows, "dbec_added_non_gold_count"),
            "mean_dbec_replaced_rank_gold_count": mean_float(rows, "dbec_replaced_rank_gold_count"),
            "near_title_substitution_query_rate": sum(
                1 for row in rows if safe_int(row.get("near_title_substitution_count")) > 0
            ) / len(rows) if rows else 0.0,
            "dbec_duplicate_title_query_rate": sum(
                1 for row in rows if safe_int(row.get("dbec_final_duplicate_title_count")) > 0
            ) / len(rows) if rows else 0.0,
            "mean_setr_f1": mean_float(rows, "setr_f1"),
            "mean_dbec_f1": mean_float(rows, "dbec_f1"),
            "mean_delta_f1_dbec_minus_setr": mean_float(rows, "delta_f1_dbec_minus_setr"),
        })
    return output


def build_case_rows(query_rows: Sequence[Mapping[str, Any]], *, limit_per_bucket: int = 8) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in query_rows}):
        dataset_rows = [row for row in query_rows if str(row.get("dataset")) == dataset]
        for bucket in sorted({str(row.get("primary_bucket")) for row in dataset_rows}):
            rows = [row for row in dataset_rows if str(row.get("primary_bucket")) == bucket]
            rows = sorted(
                rows,
                key=lambda row: (
                    safe_int(row.get("dbec_replaced_rank_gold_count")),
                    safe_int(row.get("dbec_added_non_gold_count")),
                    -safe_float(row.get("delta_f1_dbec_minus_setr")),
                ),
                reverse=True,
            )
            for row in rows[:limit_per_bucket]:
                output.append(dict(row))
    return output


def fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):.{digits}f}"


def pct(value: Any) -> str:
    return f"{100.0 * safe_float(value):.1f}%"


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def build_markdown(
    *,
    stage_summary: Sequence[Mapping[str, Any]],
    bucket_summary: Sequence[Mapping[str, Any]],
    query_bucket_summary: Sequence[Mapping[str, Any]],
    rank_depth_summary: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Repair Candidate Quality Audit",
        "",
        "This is an offline diagnostic. It uses gold support titles only to locate where SetR-missing support becomes visible in the fixed PropRAG pool and DBEC trace artifacts. It makes no reader or LLM calls.",
        "",
        "## Stage Visibility",
        "",
        "| Dataset | Q | Missing gold | Source all | rank_fill5 recover | binding-candidate recover | DBEC greedy recover | DBEC final recover | DBEC non-gold adds | rank gold replaced | near-title query | duplicate query | dF1 DBEC-SetR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_summary:
        lines.append(
            "| {dataset} | {q} | {missing} | {source} | {rank} | {binding} | {greedy} | {final} | {nongold} | {replace} | {near} | {dup} | {df1} |".format(
                dataset=row["dataset"],
                q=row["queries"],
                missing=row["missing_gold_titles"],
                source=pct(row["source_has_all_missing_gold_rate"]),
                rank=pct(row["rank_fill5_missing_gold_recovery_rate"]),
                binding=pct(row["binding_candidate_missing_gold_recovery_rate"]),
                greedy=pct(row["dbec_greedy_path_missing_gold_recovery_rate"]),
                final=pct(row["dbec_final_missing_gold_recovery_rate"]),
                nongold=fmt(row["mean_dbec_added_non_gold_count"], 2),
                replace=fmt(row["mean_dbec_replaced_rank_gold_count"], 2),
                near=pct(row["near_title_substitution_query_rate"]),
                dup=pct(row["dbec_duplicate_title_query_rate"]),
                df1=fmt(row["mean_delta_f1_dbec_minus_setr"], 4),
            )
        )

    lines.extend([
        "",
        "## Missing-Gold Buckets",
        "",
        "| Dataset | Bucket | Missing titles | Rate | dF1 DBEC-SetR |",
        "|---|---|---:|---:|---:|",
    ])
    for row in bucket_summary:
        lines.append(
            "| {dataset} | {bucket} | {n} | {rate} | {df1} |".format(
                dataset=row["dataset"],
                bucket=row["bucket"],
                n=row["missing_gold_titles"],
                rate=pct(row["missing_gold_title_rate"]),
                df1=fmt(row["mean_delta_f1_dbec_minus_setr"]),
            )
        )

    lines.extend([
        "",
        "## Query Primary Buckets",
        "",
        "| Dataset | Primary bucket | Q | Rate | missing/query | rank recover | binding recover | DBEC recover | replaced rank gold | dF1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in query_bucket_summary:
        lines.append(
            "| {dataset} | {bucket} | {q} | {rate} | {missing} | {rank} | {binding} | {dbec} | {replace} | {df1} |".format(
                dataset=row["dataset"],
                bucket=row["primary_bucket"],
                q=row["queries"],
                rate=pct(row["query_rate"]),
                missing=fmt(row["mean_missing_gold_count"], 2),
                rank=fmt(row["mean_rank_fill5_recovered_missing_count"], 2),
                binding=fmt(row["mean_binding_candidate_recovered_missing_count"], 2),
                dbec=fmt(row["mean_dbec_final_recovered_missing_count"], 2),
                replace=fmt(row["mean_dbec_replaced_rank_gold_count"], 2),
                df1=fmt(row["mean_delta_f1_dbec_minus_setr"]),
            )
        )

    lines.extend([
        "",
        "## Source Rank Depth By Missing-Gold Bucket",
        "",
        "| Dataset | Bucket | source-visible titles | mean rank | p50 | p75 | p90 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rank_depth_summary:
        mean_rank = row.get("mean_source_best_gold_rank")
        mean_text = fmt(mean_rank, 1) if str(mean_rank or "") != "" else ""
        lines.append(
            "| {dataset} | {bucket} | {n} | {mean_rank} | {p50} | {p75} | {p90} | {max_rank} |".format(
                dataset=row["dataset"],
                bucket=row["bucket"],
                n=row["source_visible_missing_titles"],
                mean_rank=mean_text,
                p50=row["p50_source_best_gold_rank"],
                p75=row["p75_source_best_gold_rank"],
                p90=row["p90_source_best_gold_rank"],
                max_rank=row["max_source_best_gold_rank"],
            )
        )

    lines.extend([
        "",
        "## MuSiQue Examples",
        "",
    ])
    musique_cases = [row for row in case_rows if str(row.get("dataset")) == "MuSiQue"]
    for row in musique_cases[:12]:
        lines.extend([
            f"- q{row['query_index']} `{row['primary_bucket']}` dF1={fmt(row['delta_f1_dbec_minus_setr'])}",
            f"  - Q: {row['question']}",
            f"  - missing: `{row['setr_missing_titles_json']}`",
            f"  - rank_fill5: `{row['rank_fill_titles_json']}`",
            f"  - DBEC: `{row['dbec_titles_json']}`",
            f"  - binding candidates: `{row['binding_candidate_titles_json']}`",
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If `gold_in_binding_candidates_not_selected` dominates, the next method change should target scoring/aggregation rather than chain-walking candidate generation.",
        "- If `gold_in_rank_fill_not_dbec` or `gold_source_only_not_rank_or_dbec` dominates, DBEC's current binding-derived candidate path is missing reachable support; chain-aware binding or candidate expansion is justified.",
        "- If `gold_absent_from_source_pool` is large, the fixed-pool setup is the ceiling and iterative retrieval would be a different method line.",
        "",
        "## Files",
        "",
        f"- Missing-gold rows: `{REPORT_DIR / 'missing_gold_audit.csv'}`",
        f"- Query rows: `{REPORT_DIR / 'query_audit.csv'}`",
        f"- Stage summary: `{REPORT_DIR / 'stage_summary.csv'}`",
        f"- Bucket summary: `{REPORT_DIR / 'bucket_summary.csv'}`",
        f"- Query bucket summary: `{REPORT_DIR / 'query_bucket_summary.csv'}`",
        f"- Source-rank depth summary: `{REPORT_DIR / 'rank_depth_summary.csv'}`",
        f"- Examples: `{REPORT_DIR / 'case_examples.csv'}`",
        f"- Full JSON: `{REPORT_DIR / 'summary.json'}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    global REPORT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--limit_per_dataset", type=int, default=0)
    args = parser.parse_args()

    REPORT_DIR = Path(args.report_dir)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    missing_gold_rows, query_rows = build_audit_rows(limit_per_dataset=int(args.limit_per_dataset))
    stage_summary = build_stage_summary(query_rows)
    bucket_summary = build_bucket_summary(missing_gold_rows)
    query_bucket_summary = build_query_bucket_summary(query_rows)
    rank_depth_summary = build_rank_depth_summary(missing_gold_rows)
    case_rows = build_case_rows(query_rows)
    payload = {
        "metadata": {
            "support_rows": str(SUPPORT_ROWS),
            "source_pool_dir": str(SOURCE_POOL_DIR),
            "setr_faithful_dir": str(SETR_FAITHFUL_DIR),
            "dbec_selective_dir": str(DBEC_SELECTIVE_DIR),
            "repair_gated_audit": str(repair_gated_audit_path()),
            "limit_per_dataset": int(args.limit_per_dataset),
        },
        "stage_summary": stage_summary,
        "bucket_summary": bucket_summary,
        "query_bucket_summary": query_bucket_summary,
        "rank_depth_summary": rank_depth_summary,
        "missing_gold_rows": missing_gold_rows,
        "query_rows": query_rows,
        "case_rows": case_rows,
    }
    write_csv(missing_gold_rows, REPORT_DIR / "missing_gold_audit.csv")
    write_csv(query_rows, REPORT_DIR / "query_audit.csv")
    write_csv(stage_summary, REPORT_DIR / "stage_summary.csv")
    write_csv(bucket_summary, REPORT_DIR / "bucket_summary.csv")
    write_csv(query_bucket_summary, REPORT_DIR / "query_bucket_summary.csv")
    write_csv(rank_depth_summary, REPORT_DIR / "rank_depth_summary.csv")
    write_csv(case_rows, REPORT_DIR / "case_examples.csv")
    write_json(payload, REPORT_DIR / "summary.json")
    (REPORT_DIR / "summary.md").write_text(
        build_markdown(
            stage_summary=stage_summary,
            bucket_summary=bucket_summary,
            query_bucket_summary=query_bucket_summary,
            rank_depth_summary=rank_depth_summary,
            case_rows=case_rows,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    print(f"Wrote candidate-quality audit to {REPORT_DIR}")


if __name__ == "__main__":
    main()
