from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

from src.hipporag.utils.causal_utils import normalize_structure_text


PARTIAL_CHAIN_CLOSURE_FAMILY = "partial_chain_closure_candidate"
POOL_COVERAGE_GAP_FAMILY = "pool_coverage_gap"
POOL_TO_SOURCE_GAP_FAMILY = "pool_to_source_gap"
SOURCE_TO_SHORTLIST_GAP_FAMILY = "source_to_shortlist_gap"
SHORTLIST_TO_SELECTED_GAP_FAMILY = "shortlist_to_selected_gap"
FINAL_SELECTED_INCOMPLETE_FAMILY = "final_selected_but_answer_incomplete"
UTILITY_REJECTION_FAMILY = "utility_rejection_candidate"
STAGED_CLOSURE_GATE_FAMILY = "staged_closure_gate_candidate"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def count_exposure_stages(exposure_rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("stage", "")).strip() for row in exposure_rows))


def detect_utility_rejection_rows(exposure_rows: Sequence[Dict[str, Any]],
                                  min_support_gain: float = 0.05,
                                  max_margin_gain: float = 0.0) -> List[Dict[str, Any]]:
    rejected_rows: List[Dict[str, Any]] = []
    for row in exposure_rows:
        stage = str(row.get("stage", "")).strip()
        support_gain = _safe_float(row.get("best_support_completeness_gain"))
        margin_gain = _safe_float(row.get("best_utility_margin_gain"))
        if stage not in {"source", "shortlist"}:
            continue
        if support_gain < float(min_support_gain):
            continue
        if margin_gain > float(max_margin_gain):
            continue
        rejected_rows.append({
            "title": str(row.get("title", "")).strip(),
            "stage": stage,
            "best_support_completeness_gain": round(support_gain, 4),
            "best_utility_margin_gain": round(margin_gain, 4),
        })
    return rejected_rows


def classify_requirement_failure_families(exposure_rows: Sequence[Dict[str, Any]],
                                          selector_metrics: Dict[str, Any] | None,
                                          gold_count: int | None = None,
                                          min_support_gain: float = 0.05,
                                          max_margin_gain: float = 0.0,
                                          staged_gate_min_gold_count: int = 3) -> List[str]:
    stage_counts = Counter(str(row.get("stage", "")).strip() for row in exposure_rows)
    metrics = selector_metrics or {}
    families: List[str] = []

    if stage_counts.get("not_in_pool", 0) > 0:
        families.append(POOL_COVERAGE_GAP_FAMILY)
    if stage_counts.get("pool_only", 0) > 0:
        families.append(POOL_TO_SOURCE_GAP_FAMILY)
    if stage_counts.get("source", 0) > 0:
        families.append(SOURCE_TO_SHORTLIST_GAP_FAMILY)
    if stage_counts.get("shortlist", 0) > 0:
        families.append(SHORTLIST_TO_SELECTED_GAP_FAMILY)
    if (
        stage_counts.get("selected", 0) + stage_counts.get("final_only", 0) > 0
        and _safe_float(metrics.get("ExactMatch")) < 1.0
    ):
        families.append(FINAL_SELECTED_INCOMPLETE_FAMILY)
    if (
        stage_counts.get("not_in_pool", 0) == 0
        and stage_counts.get("selected", 0) + stage_counts.get("final_only", 0) > 0
        and stage_counts.get("pool_only", 0) + stage_counts.get("source", 0) + stage_counts.get("shortlist", 0) > 0
    ):
        families.append(PARTIAL_CHAIN_CLOSURE_FAMILY)

    utility_rejection_rows = detect_utility_rejection_rows(
        exposure_rows=exposure_rows,
        min_support_gain=min_support_gain,
        max_margin_gain=max_margin_gain,
    )
    if utility_rejection_rows:
        families.append(UTILITY_REJECTION_FAMILY)

    effective_gold_count = int(gold_count if gold_count is not None else len(exposure_rows))
    if (
        PARTIAL_CHAIN_CLOSURE_FAMILY in families
        and POOL_COVERAGE_GAP_FAMILY not in families
        and UTILITY_REJECTION_FAMILY not in families
        and stage_counts.get("selected", 0) + stage_counts.get("final_only", 0) > 0
        and stage_counts.get("pool_only", 0) > 0
        and effective_gold_count >= int(staged_gate_min_gold_count)
    ):
        families.append(STAGED_CLOSURE_GATE_FAMILY)

    return families


def build_staged_closure_gate_decision(question: str,
                                       exposure_rows: Sequence[Dict[str, Any]],
                                       selector_metrics: Dict[str, Any] | None,
                                       gold_count: int | None = None,
                                       min_support_gain: float = 0.05,
                                       max_margin_gain: float = 0.0,
                                       staged_gate_min_gold_count: int = 3) -> Dict[str, Any]:
    stage_counts = Counter(str(row.get("stage", "")).strip() for row in exposure_rows)
    effective_gold_count = int(gold_count if gold_count is not None else len(exposure_rows))
    utility_rejection_rows = detect_utility_rejection_rows(
        exposure_rows=exposure_rows,
        min_support_gain=min_support_gain,
        max_margin_gain=max_margin_gain,
    )
    families = classify_requirement_failure_families(
        exposure_rows=exposure_rows,
        selector_metrics=selector_metrics,
        gold_count=effective_gold_count,
        min_support_gain=min_support_gain,
        max_margin_gain=max_margin_gain,
        staged_gate_min_gold_count=staged_gate_min_gold_count,
    )
    gate_hit = STAGED_CLOSURE_GATE_FAMILY in families

    positive_reasons: List[str] = []
    blockers: List[str] = []
    if PARTIAL_CHAIN_CLOSURE_FAMILY in families:
        positive_reasons.append("partial_chain_closure_candidate")
    else:
        blockers.append("partial_chain_closure_absent")
    if POOL_COVERAGE_GAP_FAMILY not in families:
        positive_reasons.append("no_pool_coverage_gap")
    else:
        blockers.append("pool_coverage_gap")
    if UTILITY_REJECTION_FAMILY not in families:
        positive_reasons.append("no_utility_rejection_signal")
    else:
        blockers.append("utility_rejection_signal")
    if stage_counts.get("selected", 0) + stage_counts.get("final_only", 0) > 0:
        positive_reasons.append("has_selected_or_final_gold")
    else:
        blockers.append("missing_selected_or_final_gold")
    if stage_counts.get("pool_only", 0) > 0:
        positive_reasons.append("has_pool_only_gold")
    else:
        blockers.append("missing_pool_only_gold")
    if effective_gold_count >= int(staged_gate_min_gold_count):
        positive_reasons.append(f"gold_count_ge_{int(staged_gate_min_gold_count)}")
    else:
        blockers.append(f"gold_count_lt_{int(staged_gate_min_gold_count)}")

    return {
        "question": str(question or "").strip(),
        "gate_mode": "conservative_staged_closure",
        "gate_hit": bool(gate_hit),
        "families": list(families),
        "stage_counts": dict(stage_counts),
        "gate_reasons": positive_reasons if gate_hit else [],
        "gate_blockers": [] if gate_hit else blockers,
        "utility_rejection_rows": utility_rejection_rows,
        "gold_count": effective_gold_count,
    }


def compute_title_recall_at_k(pool_titles: Sequence[str],
                              gold_titles: Sequence[str],
                              k: int) -> float:
    normalized_gold = []
    seen_gold = set()
    for title in gold_titles:
        normalized = normalize_structure_text(title)
        if not normalized or normalized in seen_gold:
            continue
        seen_gold.add(normalized)
        normalized_gold.append(normalized)

    if not normalized_gold:
        return 0.0

    normalized_pool = set()
    for raw_title in list(pool_titles[:max(int(k), 0)]):
        normalized = normalize_structure_text(raw_title)
        if normalized:
            normalized_pool.add(normalized)

    hit_count = sum(1 for normalized in normalized_gold if normalized in normalized_pool)
    return float(hit_count / max(1, len(normalized_gold)))
