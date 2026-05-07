#!/usr/bin/env python3
"""Offline support diagnostics for repair-gated DBEC arbitration.

This script treats DBEC as a repair proposer rather than as the final
selector.  It uses existing SetR-faithful pools, DBEC selective traces, and
PropRAG full pools to simulate rank-gated final-5 pools without calling the
reader.

Important scope: all metrics here are support/replacement diagnostics only.
They are not answer EM/F1 for the newly simulated pools.  Reader F1 requires
materializing pools and running ``eval_causal_qwen3.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


REPORT_DIR = Path("reports/repair_gated_arbitration_offline_20260507")
SUPPORT_ROWS = Path("reports/support_repair_mechanism_20260507/support_repair_rows.csv")
SOURCE_POOL_DIR = Path("run_logs/proprag_pool_exports_full1000_20260424")
SETR_FAITHFUL_DIR = Path("run_logs/setr_faithful_proprag_full1000_20260507")
DBEC_SELECTIVE_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")

TOP_K = 5
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260507
UNKNOWN_BRANCH_EXTRA_GAIN = 0.05

DATASETS: tuple[dict[str, Any], ...] = (
    {
        "label": "2Wiki",
        "dataset": "2wikimultihopqa",
        "slice": "gold_doc_count>=4 and SetR count-underselected",
        "predicate": lambda row: (
            row["dataset"] == "2Wiki"
            and safe_int(row["gold_doc_count"]) >= 4
            and safe_int(row["count_under_selected"]) == 1
        ),
    },
    {
        "label": "MuSiQue",
        "dataset": "musique",
        "slice": "gold_doc_count>=3 and SetR count-underselected",
        "predicate": lambda row: (
            row["dataset"] == "MuSiQue"
            and safe_int(row["gold_doc_count"]) >= 3
            and safe_int(row["count_under_selected"]) == 1
        ),
    },
    {
        "label": "HotpotQA",
        "dataset": "hotpotqa",
        "slice": "SetR count-underselected negative/control slice",
        "predicate": lambda row: (
            row["dataset"] == "HotpotQA"
            and safe_int(row["count_under_selected"]) == 1
        ),
    },
)


@dataclass(frozen=True)
class RepairCandidate:
    position: int
    title: str
    gain: float
    step: int
    mode: str
    coverage_by_requirement: dict[str, float]


@dataclass(frozen=True)
class VariantConfig:
    rank_quota: int
    max_repairs: int
    min_gain: float
    branch_policy: str = "off"

    @property
    def name(self) -> str:
        gain_tag = f"{int(round(self.min_gain * 100)):03d}"
        return (
            f"rq{self.rank_quota}_mr{self.max_repairs}_"
            f"g{gain_tag}_branch{self.branch_policy}"
        )


@dataclass(frozen=True)
class BranchContext:
    positive_anchors: tuple[str, ...]
    sibling_anchors: tuple[str, ...]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def list_from_json(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else []


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


ENTITY_ALIASES: dict[str, str] = {
    "u s": "united states",
    "us": "united states",
    "u s a": "united states",
    "usa": "united states",
    "america": "united states",
    "uk": "united kingdom",
    "u k": "united kingdom",
    "ussr": "soviet union",
}


def canonical_anchor(value: Any) -> str:
    text = normalize_title(value)
    if not text:
        return ""
    tokens = [ENTITY_ALIASES.get(token, token) for token in text.split()]
    expanded = " ".join(tokens)
    for alias, canonical in ENTITY_ALIASES.items():
        expanded = re.sub(rf"\b{re.escape(alias)}\b", canonical, expanded)
    return re.sub(r"\s+", " ", expanded).strip()


def compatible_anchor(left: Any, right: Any) -> bool:
    left_text = canonical_anchor(left)
    right_text = canonical_anchor(right)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    left_tokens = left_text.split()
    right_tokens = right_text.split()
    short_len = min(len(left_tokens), len(right_tokens))
    if short_len < 2:
        return False
    return left_text in right_text or right_text in left_text


def support_match(gold_titles: Sequence[Any], selected_titles: Sequence[Any]) -> dict[str, Any]:
    gold_pairs = [
        (str(title), normalize_title(title))
        for title in gold_titles
        if normalize_title(title)
    ]
    selected_counts = Counter(
        normalize_title(title)
        for title in selected_titles
        if normalize_title(title)
    )
    missing: list[str] = []
    hit_count = 0
    for original_title, normalized in gold_pairs:
        if selected_counts[normalized] > 0:
            selected_counts[normalized] -= 1
            hit_count += 1
        else:
            missing.append(original_title)
    total = len(gold_pairs)
    return {
        "hit_count": hit_count,
        "missing_count": total - hit_count,
        "recall": float(hit_count / total) if total else 0.0,
        "complete": int(total > 0 and hit_count == total),
        "missing_titles": missing,
    }


def gold_added_removed(
    gold_titles: Sequence[Any],
    baseline_titles: Sequence[Any],
    final_titles: Sequence[Any],
) -> dict[str, Any]:
    baseline_norm = {normalize_title(title) for title in baseline_titles if normalize_title(title)}
    final_norm = {normalize_title(title) for title in final_titles if normalize_title(title)}
    added: list[str] = []
    removed: list[str] = []
    for title in gold_titles:
        norm = normalize_title(title)
        if not norm:
            continue
        if norm in final_norm and norm not in baseline_norm:
            added.append(str(title))
        if norm in baseline_norm and norm not in final_norm:
            removed.append(str(title))
    return {
        "added_gold_count": len(added),
        "removed_gold_count": len(removed),
        "added_gold_titles": added,
        "removed_gold_titles": removed,
    }


def unique_positions(values: Sequence[Any], *, pool_size: int, limit: int | None = None) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for raw in values:
        pos = safe_int(raw, default=-1)
        if pos < 0 or pos >= pool_size or pos in seen:
            continue
        output.append(pos)
        seen.add(pos)
        if limit is not None and len(output) >= int(limit):
            break
    return output


def rank_fill_order(seed_positions: Sequence[int], pool_size: int, *, top_k: int = TOP_K) -> list[int]:
    seed = unique_positions(seed_positions, pool_size=pool_size, limit=top_k)
    seen = set(seed)
    order = list(seed)
    for pos in range(pool_size):
        if len(order) >= top_k:
            break
        if pos not in seen:
            order.append(pos)
            seen.add(pos)
    return order


def next_rank_position(rank_positions: Sequence[int], start_index: int, used: set[int]) -> tuple[int | None, int]:
    index = int(start_index)
    while index < len(rank_positions):
        pos = int(rank_positions[index])
        index += 1
        if pos not in used:
            return pos, index
    return None, index


def branch_status(title: str, context: BranchContext) -> str:
    if any(compatible_anchor(title, anchor) for anchor in context.positive_anchors):
        return "support"
    if any(compatible_anchor(title, anchor) for anchor in context.sibling_anchors):
        return "conflict"
    return "unknown"


def repair_gated_order(
    *,
    seed_positions: Sequence[int],
    pool_titles: Sequence[Any],
    dbec_candidates: Sequence[RepairCandidate],
    config: VariantConfig,
    branch_context: BranchContext | None = None,
    top_k: int = TOP_K,
) -> tuple[list[int], dict[str, Any]]:
    pool_size = len(pool_titles)
    final = unique_positions(seed_positions, pool_size=pool_size, limit=top_k)
    used = set(final)
    final_title_norm = {
        normalize_title(pool_titles[pos])
        for pos in final
        if 0 <= pos < pool_size and normalize_title(pool_titles[pos])
    }
    rank_positions = [pos for pos in range(pool_size) if pos not in used]
    fill_budget = max(0, min(top_k, pool_size) - len(final))
    required_rank_slots = min(max(int(config.rank_quota), 0), fill_budget)
    rank_used = 0
    accepted_repairs: list[dict[str, Any]] = []
    rejected_repairs: list[dict[str, Any]] = []
    branch_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    candidate_index = 0
    rank_index = 0

    def reject(candidate: RepairCandidate, reason: str, status: str) -> None:
        rejection_counts[reason] += 1
        rejected_repairs.append({
            "position": candidate.position,
            "title": candidate.title,
            "gain": candidate.gain,
            "step": candidate.step,
            "branch_status": status,
            "reason": reason,
        })

    while len(final) < min(top_k, pool_size):
        remaining_slots = min(top_k, pool_size) - len(final)
        remaining_rank_required = max(0, required_rank_slots - rank_used)
        force_rank = remaining_slots <= remaining_rank_required
        admitted: RepairCandidate | None = None
        admitted_status = "unknown"

        if not force_rank and len(accepted_repairs) < int(config.max_repairs):
            while candidate_index < len(dbec_candidates):
                candidate = dbec_candidates[candidate_index]
                candidate_index += 1
                status = (
                    branch_status(candidate.title, branch_context or BranchContext((), ()))
                    if config.branch_policy != "off"
                    else "unknown"
                )
                branch_counts[status] += 1
                title_norm = normalize_title(candidate.title)
                if candidate.position in used:
                    reject(candidate, "already_selected_position", status)
                    continue
                if title_norm and title_norm in final_title_norm:
                    reject(candidate, "duplicate_title", status)
                    continue
                if config.branch_policy != "off" and status == "conflict":
                    reject(candidate, "branch_conflict", status)
                    continue
                required_gain = float(config.min_gain)
                if config.branch_policy != "off" and status == "unknown":
                    required_gain += UNKNOWN_BRANCH_EXTRA_GAIN
                if candidate.gain < required_gain:
                    reject(candidate, "low_gain", status)
                    continue
                admitted = candidate
                admitted_status = status
                break

        if admitted is not None:
            final.append(admitted.position)
            used.add(admitted.position)
            norm = normalize_title(admitted.title)
            if norm:
                final_title_norm.add(norm)
            accepted_repairs.append({
                "position": admitted.position,
                "title": admitted.title,
                "gain": admitted.gain,
                "step": admitted.step,
                "branch_status": admitted_status,
            })
            continue

        rank_pos, rank_index = next_rank_position(rank_positions, rank_index, used)
        if rank_pos is None:
            break
        final.append(rank_pos)
        used.add(rank_pos)
        rank_used += 1

    trace = {
        "variant": config.name,
        "rank_quota": int(config.rank_quota),
        "max_repairs": int(config.max_repairs),
        "min_gain": float(config.min_gain),
        "branch_policy": config.branch_policy,
        "seed_positions": list(unique_positions(seed_positions, pool_size=pool_size, limit=top_k)),
        "final_order_positions": final,
        "final_titles": [str(pool_titles[pos]) for pos in final],
        "accepted_repairs": accepted_repairs,
        "rejected_repairs": rejected_repairs,
        "rank_fill_positions": [pos for pos in final if pos not in set(seed_positions) and all(pos != r["position"] for r in accepted_repairs)],
        "accepted_repair_count": len(accepted_repairs),
        "rank_fill_count": rank_used,
        "branch_status_counts": dict(branch_counts),
        "rejection_counts": dict(rejection_counts),
        "branch_conflict_reject_count": int(rejection_counts.get("branch_conflict", 0)),
        "branch_unknown_count": int(branch_counts.get("unknown", 0)),
    }
    return final, trace


def source_pool_path(dataset: str) -> Path:
    return SOURCE_POOL_DIR / f"{dataset}_pool100.json"


def setr_pool_path(dataset: str) -> Path:
    return SETR_FAITHFUL_DIR / f"{dataset}_proprag_setr_k20_doc768_faithful.selected_pool.json"


def dbec_report_path(dataset: str) -> Path:
    return DBEC_SELECTIVE_DIR / f"{dataset}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def read_support_rows() -> list[dict[str, Any]]:
    with SUPPORT_ROWS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dataset_artifacts(dataset: str) -> dict[str, Any]:
    source_payload = read_json(source_pool_path(dataset))
    setr_payload = read_json(setr_pool_path(dataset))
    dbec_payload = read_json(dbec_report_path(dataset))
    return {
        "source_records": list(source_payload.get("records") or []),
        "setr_records": list(setr_payload.get("records") or []),
        "dbec_traces": list(dbec_payload.get("setwise_selector_query_traces") or []),
    }


def extract_setr_seed_positions(setr_record: Mapping[str, Any], pool_size: int) -> list[int]:
    trace = setr_record.get("setr_selection_trace")
    trace = trace if isinstance(trace, Mapping) else {}
    return unique_positions(trace.get("selected_positions") or [], pool_size=pool_size, limit=TOP_K)


def extract_dbec_candidates(selector_trace: Mapping[str, Any], pool_titles: Sequence[Any]) -> list[RepairCandidate]:
    pool_size = len(pool_titles)
    output: list[RepairCandidate] = []
    seen: set[int] = set()
    for raw_step in selector_trace.get("selection_steps") or []:
        if not isinstance(raw_step, Mapping):
            continue
        pos = safe_int(raw_step.get("pool_position"), default=-1)
        if pos < 0 or pos >= pool_size or pos in seen:
            continue
        seen.add(pos)
        title = str(raw_step.get("title") or pool_titles[pos])
        gain = safe_float(
            raw_step.get("effective_gain"),
            default=safe_float(
                raw_step.get("objective_gain"),
                default=safe_float(raw_step.get("coverage_gain")),
            ),
        )
        coverage = raw_step.get("coverage_by_requirement")
        coverage_by_requirement = {
            str(key): safe_float(value)
            for key, value in (coverage.items() if isinstance(coverage, Mapping) else [])
        }
        output.append(
            RepairCandidate(
                position=pos,
                title=title,
                gain=gain,
                step=safe_int(raw_step.get("step"), default=len(output) + 1),
                mode=str(raw_step.get("mode") or ""),
                coverage_by_requirement=coverage_by_requirement,
            )
        )
    if output:
        return output

    fallback_positions = selector_trace.get("selected_pool_positions") or selector_trace.get("selected_positions") or []
    for step, pos in enumerate(unique_positions(fallback_positions, pool_size=pool_size), start=1):
        output.append(
            RepairCandidate(
                position=pos,
                title=str(pool_titles[pos]),
                gain=1.0,
                step=step,
                mode="selected_position_fallback",
                coverage_by_requirement={},
            )
        )
    return output


def extract_dbec_final_order(selector_trace: Mapping[str, Any], pool_size: int) -> list[int]:
    positions = (
        selector_trace.get("final_front_pool_positions")
        or selector_trace.get("selected_pool_positions")
        or selector_trace.get("selected_positions")
        or []
    )
    return rank_fill_order(unique_positions(positions, pool_size=pool_size), pool_size=pool_size, top_k=TOP_K)


def add_anchor(output: list[str], value: Any) -> None:
    if isinstance(value, str):
        text = canonical_anchor(value)
        if text and text not in output:
            output.append(text)


def add_anchor_list(output: list[str], values: Any) -> None:
    if isinstance(values, list):
        for value in values:
            add_anchor(output, value)


def build_branch_context(selector_trace: Mapping[str, Any], seed_titles: Sequence[Any]) -> BranchContext:
    positive: list[str] = []
    sibling: list[str] = []
    for key in (
        "question_entities_preview",
        "grounded_question_entities_preview",
        "query_entities_preview",
        "proposal_query_entities_preview",
    ):
        add_anchor_list(positive, selector_trace.get(key))
    for title in seed_titles:
        add_anchor(positive, title)

    selected_binding = selector_trace.get("selected_binding")
    assignments = selected_binding.get("assignments") if isinstance(selected_binding, Mapping) else {}
    assignments = assignments if isinstance(assignments, Mapping) else {}
    for value in assignments.values():
        add_anchor(positive, value)

    candidates_by_requirement = selector_trace.get("binding_candidates_by_requirement")
    candidates_by_requirement = candidates_by_requirement if isinstance(candidates_by_requirement, Mapping) else {}
    for requirement_id, candidates in candidates_by_requirement.items():
        selected_value = assignments.get(requirement_id)
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, Mapping):
                continue
            title = str(candidate.get("title") or "")
            if not title:
                continue
            if selected_value and compatible_anchor(title, selected_value):
                add_anchor(positive, title)
            else:
                add_anchor(sibling, title)
    return BranchContext(tuple(positive), tuple(sibling))


def variant_grid(include_branch_variants: bool = True) -> list[VariantConfig]:
    variants: list[VariantConfig] = []
    branch_policies = ["off", "strict"] if include_branch_variants else ["off"]
    for branch_policy in branch_policies:
        for rank_quota in (1, 2):
            for max_repairs in (1, 2, 3):
                for min_gain in (0.00, 0.03, 0.05, 0.10):
                    variants.append(
                        VariantConfig(
                            rank_quota=rank_quota,
                            max_repairs=max_repairs,
                            min_gain=min_gain,
                            branch_policy=branch_policy,
                        )
                    )
    return variants


def support_fields(prefix: str, support: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_support_hit_count": safe_int(support.get("hit_count")),
        f"{prefix}_support_missing_count": safe_int(support.get("missing_count")),
        f"{prefix}_support_recall": safe_float(support.get("recall")),
        f"{prefix}_support_complete": safe_int(support.get("complete")),
        f"{prefix}_support_missing_titles_json": json.dumps(
            list(support.get("missing_titles") or []),
            ensure_ascii=False,
        ),
    }


def build_row_for_variant(
    *,
    dataset_label: str,
    dataset: str,
    support_row: Mapping[str, Any],
    source_record: Mapping[str, Any],
    setr_record: Mapping[str, Any],
    dbec_query_trace: Mapping[str, Any],
    config: VariantConfig,
) -> dict[str, Any]:
    pool_titles = list(source_record.get("pool_titles") or [])
    pool_size = len(pool_titles)
    selector_trace = dbec_query_trace.get("selector_trace")
    selector_trace = selector_trace if isinstance(selector_trace, Mapping) else {}
    seed_positions = extract_setr_seed_positions(setr_record, pool_size)
    seed_titles = [pool_titles[pos] for pos in seed_positions]
    dbec_candidates = extract_dbec_candidates(selector_trace, pool_titles)
    branch_context = build_branch_context(selector_trace, seed_titles)
    final_order, trace = repair_gated_order(
        seed_positions=seed_positions,
        pool_titles=pool_titles,
        dbec_candidates=dbec_candidates,
        config=config,
        branch_context=branch_context,
        top_k=TOP_K,
    )
    rank_order = rank_fill_order(seed_positions, pool_size=pool_size, top_k=TOP_K)
    dbec_order = extract_dbec_final_order(selector_trace, pool_size)
    gold_titles = list_from_json(support_row.get("gold_titles_json"))

    final_titles = [pool_titles[pos] for pos in final_order]
    rank_titles = [pool_titles[pos] for pos in rank_order]
    dbec_titles = [pool_titles[pos] for pos in dbec_order]
    final_support = support_match(gold_titles, final_titles)
    rank_support = support_match(gold_titles, rank_titles)
    dbec_support = support_match(gold_titles, dbec_titles)
    gold_delta = gold_added_removed(gold_titles, rank_titles, final_titles)
    net_gold_delta = safe_int(final_support.get("hit_count")) - safe_int(rank_support.get("hit_count"))
    accepted = list(trace.get("accepted_repairs") or [])
    rejection_counts = dict(trace.get("rejection_counts") or {})
    branch_counts = dict(trace.get("branch_status_counts") or {})
    row = {
        "dataset": dataset_label,
        "base_dataset": dataset,
        "query_index": safe_int(support_row.get("query_index")),
        "question": str(support_row.get("question") or source_record.get("question") or ""),
        "variant": config.name,
        "rank_quota": int(config.rank_quota),
        "max_repairs": int(config.max_repairs),
        "min_gain": float(config.min_gain),
        "branch_policy": config.branch_policy,
        "gold_doc_count": safe_int(support_row.get("gold_doc_count")),
        "requirement_count": safe_int(support_row.get("requirement_count")),
        "dependent_req_count": safe_int(support_row.get("dependent_req_count")),
        "selective_binding_decision": str(support_row.get("selective_binding_decision") or ""),
        "seed_count": len(seed_positions),
        "seed_positions_json": json.dumps(seed_positions),
        "seed_titles_json": json.dumps([str(title) for title in seed_titles], ensure_ascii=False),
        "rank_fill_positions_json": json.dumps(rank_order),
        "rank_fill_titles_json": json.dumps([str(title) for title in rank_titles], ensure_ascii=False),
        "dbec_selective_positions_json": json.dumps(dbec_order),
        "dbec_selective_titles_json": json.dumps([str(title) for title in dbec_titles], ensure_ascii=False),
        "final_positions_json": json.dumps(final_order),
        "final_titles_json": json.dumps([str(title) for title in final_titles], ensure_ascii=False),
        "accepted_repair_count": safe_int(trace.get("accepted_repair_count")),
        "accepted_repair_positions_json": json.dumps([safe_int(item.get("position")) for item in accepted]),
        "accepted_repair_titles_json": json.dumps([str(item.get("title") or "") for item in accepted], ensure_ascii=False),
        "accepted_repair_gains_json": json.dumps([safe_float(item.get("gain")) for item in accepted]),
        "accepted_repair_branch_status_json": json.dumps([str(item.get("branch_status") or "") for item in accepted]),
        "rank_fill_count": safe_int(trace.get("rank_fill_count")),
        "branch_conflict_reject_count": safe_int(trace.get("branch_conflict_reject_count")),
        "branch_unknown_count": safe_int(trace.get("branch_unknown_count")),
        "rejection_counts_json": json.dumps(rejection_counts, sort_keys=True),
        "branch_status_counts_json": json.dumps(branch_counts, sort_keys=True),
        "added_gold_count": safe_int(gold_delta.get("added_gold_count")),
        "removed_gold_count": safe_int(gold_delta.get("removed_gold_count")),
        "net_gold_delta_vs_rank": net_gold_delta,
        "harmful_replacement": int(net_gold_delta < 0),
        "added_gold_titles_json": json.dumps(gold_delta.get("added_gold_titles") or [], ensure_ascii=False),
        "removed_gold_titles_json": json.dumps(gold_delta.get("removed_gold_titles") or [], ensure_ascii=False),
        "setr_f1": safe_float(support_row.get("setr_f1")),
        "dbec_f1": safe_float(support_row.get("dbec_f1")),
        "delta_f1_dbec_minus_setr": safe_float(support_row.get("delta_f1_dbec_minus_setr")),
    }
    row.update(support_fields("final", final_support))
    row.update(support_fields("rank", rank_support))
    row.update(support_fields("dbec", dbec_support))
    row["delta_support_recall_vs_rank"] = safe_float(row["final_support_recall"]) - safe_float(row["rank_support_recall"])
    row["delta_support_complete_vs_rank"] = safe_int(row["final_support_complete"]) - safe_int(row["rank_support_complete"])
    row["delta_support_recall_vs_dbec"] = safe_float(row["final_support_recall"]) - safe_float(row["dbec_support_recall"])
    row["delta_support_complete_vs_dbec"] = safe_int(row["final_support_complete"]) - safe_int(row["dbec_support_complete"])
    return row


def build_offline_rows(
    *,
    limit_per_dataset: int = 0,
    include_branch_variants: bool = True,
) -> list[dict[str, Any]]:
    support_rows = read_support_rows()
    variants = variant_grid(include_branch_variants=include_branch_variants)
    output: list[dict[str, Any]] = []
    for config in DATASETS:
        dataset_label = str(config["label"])
        dataset = str(config["dataset"])
        selected_rows = [row for row in support_rows if config["predicate"](row)]
        selected_rows = sorted(selected_rows, key=lambda row: safe_int(row.get("query_index")))
        if limit_per_dataset > 0:
            selected_rows = selected_rows[: int(limit_per_dataset)]
        artifacts = load_dataset_artifacts(dataset)
        source_records = artifacts["source_records"]
        setr_records = artifacts["setr_records"]
        dbec_traces = artifacts["dbec_traces"]
        for support_row in selected_rows:
            query_index = safe_int(support_row.get("query_index"))
            source_record = source_records[query_index]
            setr_record = setr_records[query_index]
            dbec_query_trace = dbec_traces[query_index]
            for variant in variants:
                output.append(
                    build_row_for_variant(
                        dataset_label=dataset_label,
                        dataset=dataset,
                        support_row=support_row,
                        source_record=source_record,
                        setr_record=setr_record,
                        dbec_query_trace=dbec_query_trace,
                        config=variant,
                    )
                )
    return output


def mean_float(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(mean([safe_float(row.get(field)) for row in rows])) if rows else 0.0


def mean_int(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(mean([safe_int(row.get(field)) for row in rows])) if rows else 0.0


def build_variant_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in rows}):
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for variant in sorted({str(row.get("variant")) for row in dataset_rows}):
            variant_rows = [row for row in dataset_rows if str(row.get("variant")) == variant]
            if not variant_rows:
                continue
            first = variant_rows[0]
            output.append({
                "dataset": dataset,
                "variant": variant,
                "n": len(variant_rows),
                "rank_quota": safe_int(first.get("rank_quota")),
                "max_repairs": safe_int(first.get("max_repairs")),
                "min_gain": safe_float(first.get("min_gain")),
                "branch_policy": str(first.get("branch_policy") or ""),
                "final_support_recall": mean_float(variant_rows, "final_support_recall"),
                "final_support_complete": mean_int(variant_rows, "final_support_complete"),
                "rank_support_recall": mean_float(variant_rows, "rank_support_recall"),
                "rank_support_complete": mean_int(variant_rows, "rank_support_complete"),
                "dbec_support_recall": mean_float(variant_rows, "dbec_support_recall"),
                "dbec_support_complete": mean_int(variant_rows, "dbec_support_complete"),
                "delta_support_recall_vs_rank": mean_float(variant_rows, "delta_support_recall_vs_rank"),
                "delta_support_complete_vs_rank": mean_float(variant_rows, "delta_support_complete_vs_rank"),
                "delta_support_recall_vs_dbec": mean_float(variant_rows, "delta_support_recall_vs_dbec"),
                "delta_support_complete_vs_dbec": mean_float(variant_rows, "delta_support_complete_vs_dbec"),
                "accepted_repair_count": mean_float(variant_rows, "accepted_repair_count"),
                "rank_fill_count": mean_float(variant_rows, "rank_fill_count"),
                "added_gold_count": mean_float(variant_rows, "added_gold_count"),
                "removed_gold_count": mean_float(variant_rows, "removed_gold_count"),
                "net_gold_delta_vs_rank": mean_float(variant_rows, "net_gold_delta_vs_rank"),
                "harmful_replacement_rate": mean_float(variant_rows, "harmful_replacement"),
                "branch_conflict_reject_count": mean_float(variant_rows, "branch_conflict_reject_count"),
                "branch_unknown_count": mean_float(variant_rows, "branch_unknown_count"),
            })
    return output


def paired_bootstrap_delta(
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if len(left_values) != len(right_values):
        raise ValueError("paired bootstrap requires equal-length inputs")
    if not left_values:
        return {
            "n": 0,
            "delta_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_delta_gt_0": 0.0,
            "ci_excludes_zero": False,
        }
    deltas = np.asarray(left_values, dtype=float) - np.asarray(right_values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = deltas[rng.integers(0, deltas.size, size=deltas.size)]
        boot[index] = float(np.mean(sampled))
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return {
        "n": int(deltas.size),
        "delta_mean": float(np.mean(deltas)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_delta_gt_0": float(np.mean(boot > 0.0)),
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def build_paired_support_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = (
        ("support_recall", "final_support_recall", "rank_support_recall", "dbec_support_recall"),
        ("support_complete", "final_support_complete", "rank_support_complete", "dbec_support_complete"),
        ("support_hit_count", "final_support_hit_count", "rank_support_hit_count", "dbec_support_hit_count"),
    )
    for dataset in sorted({str(row.get("dataset")) for row in rows}):
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for variant in sorted({str(row.get("variant")) for row in dataset_rows}):
            variant_rows = [row for row in dataset_rows if str(row.get("variant")) == variant]
            for metric, final_field, rank_field, dbec_field in metrics:
                for baseline, baseline_field in (("rank_fill5", rank_field), ("dbec_selective", dbec_field)):
                    stats = paired_bootstrap_delta(
                        [safe_float(row.get(final_field)) for row in variant_rows],
                        [safe_float(row.get(baseline_field)) for row in variant_rows],
                        seed=BOOTSTRAP_SEED + len(output) * 31,
                        samples=bootstrap_samples,
                    )
                    first = variant_rows[0] if variant_rows else {}
                    output.append({
                        "dataset": dataset,
                        "variant": variant,
                        "baseline": baseline,
                        "metric": metric,
                        "rank_quota": safe_int(first.get("rank_quota")),
                        "max_repairs": safe_int(first.get("max_repairs")),
                        "min_gain": safe_float(first.get("min_gain")),
                        "branch_policy": str(first.get("branch_policy") or ""),
                        **stats,
                    })
    return output


def top_rows_for_dataset(
    summary_rows: Sequence[Mapping[str, Any]],
    dataset: str,
    *,
    limit: int = 5,
) -> list[Mapping[str, Any]]:
    rows = [row for row in summary_rows if str(row.get("dataset")) == dataset]
    return sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("delta_support_complete_vs_rank")),
            safe_float(row.get("delta_support_recall_vs_rank")),
            -safe_float(row.get("harmful_replacement_rate")),
            -safe_float(row.get("accepted_repair_count")),
            safe_float(row.get("min_gain")),
            -safe_int(row.get("rank_quota")),
        ),
        reverse=True,
    )[:limit]


def build_markdown(
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    paired_ci: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Repair-Gated DBEC Offline Arbitration",
        "",
        "This is an offline support/replacement diagnostic. It does not report reader EM/F1 for simulated final-5 pools.",
        "",
        "## Setup",
        "",
        f"- Top-k budget: `{TOP_K}`",
        "- Seed: SetR-faithful selected positions.",
        "- Repair proposer: DBEC selective `selection_steps` candidates.",
        "- Rank fallback: original PropRAG pool order excluding already selected positions.",
        "- Admission rule: accept DBEC candidates only when the rank quota, max repair count, min gain, duplicate-title filter, and optional branch filter allow it.",
        "",
        "## Slices",
        "",
    ]
    for config in DATASETS:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == str(config["label"])]
        n = len({safe_int(row.get("query_index")) for row in dataset_rows})
        lines.append(f"- {config['label']}: N={n}, `{config['slice']}`")

    lines.extend(["", "## Best Variants By Dataset", ""])
    for dataset in sorted({str(row.get("dataset")) for row in rows}):
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| variant | dComplete vs rank | dRecall vs rank | harmful | repairs | min_gain | branch |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in top_rows_for_dataset(summary_rows, dataset, limit=8):
            lines.append(
                "| {variant} | {dc:+.4f} | {dr:+.4f} | {harm:.4f} | {repairs:.3f} | {gain:.2f} | {branch} |".format(
                    variant=row["variant"],
                    dc=safe_float(row.get("delta_support_complete_vs_rank")),
                    dr=safe_float(row.get("delta_support_recall_vs_rank")),
                    harm=safe_float(row.get("harmful_replacement_rate")),
                    repairs=safe_float(row.get("accepted_repair_count")),
                    gain=safe_float(row.get("min_gain")),
                    branch=row.get("branch_policy"),
                )
            )
        lines.append("")

    lines.extend(["## CI Highlights", ""])
    for dataset in sorted({str(row.get("dataset")) for row in rows}):
        best = top_rows_for_dataset(summary_rows, dataset, limit=1)
        if not best:
            continue
        variant = str(best[0]["variant"])
        ci_rows = [
            row for row in paired_ci
            if str(row.get("dataset")) == dataset
            and str(row.get("variant")) == variant
            and str(row.get("baseline")) == "rank_fill5"
            and str(row.get("metric")) in {"support_recall", "support_complete"}
        ]
        lines.append(f"- {dataset} best `{variant}` vs rank_fill5:")
        for row in ci_rows:
            lines.append(
                "  - {metric}: delta={delta:+.4f}, 95% CI=[{lo:+.4f}, {hi:+.4f}], p(delta>0)={p:.3f}".format(
                    metric=row.get("metric"),
                    delta=safe_float(row.get("delta_mean")),
                    lo=safe_float(row.get("ci_low")),
                    hi=safe_float(row.get("ci_high")),
                    p=safe_float(row.get("p_delta_gt_0")),
                )
            )

    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Per-query audit: `{REPORT_DIR / 'replacement_audit.csv'}`",
        f"- Variant summary: `{REPORT_DIR / 'variant_support_summary.csv'}`",
        f"- Paired support CI: `{REPORT_DIR / 'paired_support_ci.csv'}`",
        f"- Full JSON: `{REPORT_DIR / 'summary.json'}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    global REPORT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--limit_per_dataset", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--no_branch_variants", action="store_true")
    args = parser.parse_args()

    REPORT_DIR = Path(args.report_dir)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_offline_rows(
        limit_per_dataset=int(args.limit_per_dataset),
        include_branch_variants=not bool(args.no_branch_variants),
    )
    summary_rows = build_variant_summary(rows)
    paired_ci = build_paired_support_ci(rows, bootstrap_samples=int(args.bootstrap_samples))
    payload = {
        "metadata": {
            "top_k": TOP_K,
            "support_rows": str(SUPPORT_ROWS),
            "source_pool_dir": str(SOURCE_POOL_DIR),
            "setr_faithful_dir": str(SETR_FAITHFUL_DIR),
            "dbec_selective_dir": str(DBEC_SELECTIVE_DIR),
            "limit_per_dataset": int(args.limit_per_dataset),
            "branch_variants": not bool(args.no_branch_variants),
        },
        "replacement_audit": rows,
        "variant_support_summary": summary_rows,
        "paired_support_ci": paired_ci,
    }

    write_csv(rows, REPORT_DIR / "replacement_audit.csv")
    write_csv(summary_rows, REPORT_DIR / "variant_support_summary.csv")
    write_csv(paired_ci, REPORT_DIR / "paired_support_ci.csv")
    write_json(payload, REPORT_DIR / "summary.json")
    (REPORT_DIR / "summary.md").write_text(
        build_markdown(rows, summary_rows, paired_ci) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    print(f"Wrote {len(rows)} audit rows to {REPORT_DIR}")


if __name__ == "__main__":
    main()
