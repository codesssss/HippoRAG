"""Trace helpers for Preservation-Constrained Evidence Composition."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from evidence_transition_graphragv4_composition.contract import (
    DEFAULT_ADMISSION_OBJECTIVE,
    DEFAULT_ORDERING_POLICY,
    DEFAULT_RETENTION_PROXY,
    METHOD_CONTRACT,
    METHOD_NAME,
    PAPER_FACING_METHOD_NAME,
    residual_budget,
    validate_budgets,
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_counter(titles: Sequence[Any], *, k: int | None = None) -> Counter[str]:
    selected = list(titles[:k] if k is not None else titles)
    return Counter(title for title in (normalize_title(item) for item in selected) if title)


def title_recall(gold_titles: Sequence[Any], candidate_titles: Sequence[Any], *, k: int) -> float:
    gold = title_counter(gold_titles)
    if not gold:
        return 0.0
    have = title_counter(candidate_titles, k=k)
    hit = sum(min(count, have[title]) for title, count in gold.items())
    return hit / sum(gold.values())


def title_all_covered(gold_titles: Sequence[Any], candidate_titles: Sequence[Any], *, k: int) -> bool:
    gold = title_counter(gold_titles)
    if not gold:
        return False
    have = title_counter(candidate_titles, k=k)
    return all(have[title] >= count for title, count in gold.items())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(output) or math.isinf(output):
        return default
    return output


def selected_query_traces(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    traces = payload.get("setwise_selector_query_traces")
    if not isinstance(traces, list):
        traces = payload.get("pcec_query_traces")
    return [dict(row) for row in traces or [] if isinstance(row, Mapping)]


def final_titles_from_trace(trace: Mapping[str, Any]) -> list[str]:
    pcec = trace.get("pcec_readout")
    if isinstance(pcec, Mapping) and isinstance(pcec.get("final_top5_titles"), list):
        return [str(item) for item in pcec.get("final_top5_titles") or []]
    for key in ("selector_top_titles", "method_top_titles", "final_top_titles"):
        value = trace.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    selector_trace = trace.get("selector_trace")
    if isinstance(selector_trace, Mapping) and isinstance(selector_trace.get("final_front_titles"), list):
        return [str(item) for item in selector_trace.get("final_front_titles") or []]
    return []


def baseline_titles_from_trace(trace: Mapping[str, Any]) -> list[str]:
    value = trace.get("baseline_top_titles")
    if isinstance(value, list):
        return [str(item) for item in value]
    pcec = trace.get("pcec_readout")
    if isinstance(pcec, Mapping) and isinstance(pcec.get("baseline_top5_titles"), list):
        return [str(item) for item in pcec.get("baseline_top5_titles") or []]
    selector_trace = trace.get("selector_trace")
    if isinstance(selector_trace, Mapping):
        safe_trace = selector_trace.get("safe_projection_trace")
        if isinstance(safe_trace, Mapping) and isinstance(safe_trace.get("baseline_titles"), list):
            return [str(item) for item in safe_trace.get("baseline_titles") or []]
    return []


def selector_trace_from_query_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    selector_trace = trace.get("selector_trace")
    return dict(selector_trace) if isinstance(selector_trace, Mapping) else {}


def safe_projection_trace(selector_trace: Mapping[str, Any]) -> dict[str, Any]:
    safe_trace = selector_trace.get("safe_projection_trace")
    return dict(safe_trace) if isinstance(safe_trace, Mapping) else {}


def swap_steps(selector_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    safe_trace = safe_projection_trace(selector_trace)
    raw_steps = safe_trace.get("safe_swap_steps")
    if not isinstance(raw_steps, list):
        raw_steps = selector_trace.get("selection_steps")
    return [dict(row) for row in raw_steps or [] if isinstance(row, Mapping)]


def pcec_readout_trace(
    trace: Mapping[str, Any],
    *,
    reader_budget_k: int,
    prefix_budget_m: int,
    pool_k: int,
    ordering_policy: str = DEFAULT_ORDERING_POLICY,
) -> dict[str, Any]:
    validate_budgets(reader_budget_k=reader_budget_k, prefix_budget_m=prefix_budget_m)
    residual = residual_budget(reader_budget_k, prefix_budget_m)
    selector_trace = selector_trace_from_query_trace(trace)
    safe_trace = safe_projection_trace(selector_trace)
    baseline_titles = baseline_titles_from_trace(trace)[:reader_budget_k]
    final_titles = final_titles_from_trace(trace)[:reader_budget_k]
    steps = swap_steps(selector_trace)
    first_step = steps[0] if steps else {}
    decision = "admit" if steps and baseline_titles != final_titles else "keep_baseline"
    residual_incumbent = baseline_titles[prefix_budget_m] if prefix_budget_m < len(baseline_titles) else None
    best_candidate_title = first_step.get("in_title") or first_step.get("title")
    best_candidate_rank = first_step.get("in_position")
    return {
        "method": METHOD_NAME,
        "paper_facing_method": PAPER_FACING_METHOD_NAME,
        "reader_budget_k": int(reader_budget_k),
        "prefix_budget_m": int(prefix_budget_m),
        "residual_budget": int(residual),
        "pool_k": int(pool_k),
        "retention_proxy": DEFAULT_RETENTION_PROXY,
        "admission_objective": DEFAULT_ADMISSION_OBJECTIVE,
        "baseline_top5_titles": baseline_titles,
        "retained_prefix_titles": baseline_titles[:prefix_budget_m],
        "residual_incumbent_title": residual_incumbent,
        "baseline_objective": safe_float(
            selector_trace.get("baseline_objective", safe_trace.get("baseline_objective")),
            default=0.0,
        ),
        "best_candidate_title": str(best_candidate_title) if best_candidate_title is not None else None,
        "best_candidate_pool_rank": int(best_candidate_rank) + 1 if best_candidate_rank is not None else None,
        "best_candidate_objective": safe_float(first_step.get("objective"), default=0.0),
        "best_candidate_gain": safe_float(first_step.get("objective_gain"), default=0.0),
        "decision": decision,
        "final_top5_titles": final_titles,
        "ordering_policy": ordering_policy,
        "binding_protocol": "dbec_best_binding_frozen",
        "selected_binding_id": selector_trace.get("selected_binding_id"),
        "safe_decision": safe_trace.get("safe_decision"),
        "legacy_selector": selector_trace.get("selector"),
        "legacy_safe_projection_trace": safe_trace,
    }


def enrich_query_traces(
    traces: Sequence[Mapping[str, Any]],
    *,
    reader_budget_k: int,
    prefix_budget_m: int,
    pool_k: int,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for trace in traces:
        row = dict(trace)
        row["pcec_readout"] = pcec_readout_trace(
            row,
            reader_budget_k=reader_budget_k,
            prefix_budget_m=prefix_budget_m,
            pool_k=pool_k,
        )
        enriched.append(row)
    return enriched


def summarize_pcec_traces(
    traces: Sequence[Mapping[str, Any]],
    *,
    reader_budget_k: int,
) -> dict[str, Any]:
    if not traces:
        return {"count": 0}
    rows = []
    for trace in traces:
        baseline_titles = baseline_titles_from_trace(trace)[:reader_budget_k]
        final_titles = final_titles_from_trace(trace)[:reader_budget_k]
        gold_titles = list(trace.get("gold_titles") or [])
        selector_trace = selector_trace_from_query_trace(trace)
        steps = swap_steps(selector_trace)
        gold_norm = title_counter(gold_titles)
        swapped_out_gold = 0
        swapped_in_gold = 0
        for step in steps:
            out_title = normalize_title(step.get("out_title"))
            in_title = normalize_title(step.get("in_title"))
            if gold_norm.get(out_title, 0) > 0:
                swapped_out_gold += 1
            if gold_norm.get(in_title, 0) > 0:
                swapped_in_gold += 1
        rows.append(
            {
                "changed": baseline_titles != final_titles,
                "swap_count": len(steps),
                "swapped_out_gold": swapped_out_gold,
                "swapped_in_gold": swapped_in_gold,
                "baseline_title_recall": title_recall(gold_titles, baseline_titles, k=reader_budget_k),
                "final_title_recall": title_recall(gold_titles, final_titles, k=reader_budget_k),
                "baseline_title_all": title_all_covered(gold_titles, baseline_titles, k=reader_budget_k),
                "final_title_all": title_all_covered(gold_titles, final_titles, k=reader_budget_k),
            }
        )
    count = len(rows)
    return {
        "count": int(count),
        "changed_count": int(sum(bool(row["changed"]) for row in rows)),
        "total_swaps": int(sum(int(row["swap_count"]) for row in rows)),
        "queries_swapped_out_gold": int(sum(int(row["swapped_out_gold"]) > 0 for row in rows)),
        "queries_swapped_in_gold": int(sum(int(row["swapped_in_gold"]) > 0 for row in rows)),
        "baseline_title_recall_top5": round(mean(float(row["baseline_title_recall"]) for row in rows), 4),
        "pcec_title_recall_top5": round(mean(float(row["final_title_recall"]) for row in rows), 4),
        "baseline_title_all_gold_top5": round(sum(bool(row["baseline_title_all"]) for row in rows) / count, 4),
        "pcec_title_all_gold_top5": round(sum(bool(row["final_title_all"]) for row in rows) / count, 4),
    }


def enrich_payload(
    payload: Mapping[str, Any],
    *,
    reader_budget_k: int,
    prefix_budget_m: int,
    pool_k: int,
) -> dict[str, Any]:
    validate_budgets(reader_budget_k=reader_budget_k, prefix_budget_m=prefix_budget_m)
    output = dict(payload)
    traces = enrich_query_traces(
        selected_query_traces(output),
        reader_budget_k=reader_budget_k,
        prefix_budget_m=prefix_budget_m,
        pool_k=pool_k,
    )
    output["method"] = METHOD_NAME
    output["paper_facing_method"] = PAPER_FACING_METHOD_NAME
    output["mode"] = "pcec_integrated_eval"
    output["pcec_contract"] = dict(METHOD_CONTRACT)
    output["pcec_config"] = {
        "reader_budget_k": int(reader_budget_k),
        "prefix_budget_m": int(prefix_budget_m),
        "residual_budget": int(residual_budget(reader_budget_k, prefix_budget_m)),
        "pool_k": int(pool_k),
        "ordering_policy": DEFAULT_ORDERING_POLICY,
    }
    output["setwise_selector_query_traces"] = traces
    output["pcec_summary"] = summarize_pcec_traces(traces, reader_budget_k=reader_budget_k)
    return output
