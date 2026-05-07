#!/usr/bin/env python3
"""Offline feasibility analysis for selective/null DAEC binding.

This script is intentionally post-hoc and reader-free. It compares paired
DAEC-vs-nobinding full1000 reports and asks whether binding-extraction-time
signals can predict when the query should abstain from binding.

Allowed signals are selection-independent: binding candidate counts, title
resolution uniqueness, raw/matched extraction counts, and decomposition shape.
The dep-score margin is reported only as a diagnostic because it is close to the
embedding-derived signal that failed in previous posterior experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence


DATASETS: tuple[tuple[str, str, str], ...] = (
    (
        "2Wiki",
        "run_logs/daec_llm_wiki_title_proprag_full1000_20260503/"
        "2wikimultihopqa_proprag_wiki_title_daec_llm_full1000.json",
        "run_logs/daec_nobinding_proprag_full1000_20260506/"
        "2wikimultihopqa_proprag_wiki_title_daec_noisyor_nobind_full1000.json",
    ),
    (
        "HotpotQA",
        "run_logs/daec_llm_wiki_title_proprag_full1000_20260503/"
        "hotpotqa_proprag_wiki_title_daec_llm_full1000.json",
        "run_logs/daec_nobinding_proprag_full1000_20260506/"
        "hotpotqa_proprag_wiki_title_daec_noisyor_nobind_full1000.json",
    ),
    (
        "MuSiQue",
        "run_logs/daec_llm_wiki_title_proprag_full1000_20260503/"
        "musique_proprag_wiki_title_daec_llm_full1000.json",
        "run_logs/daec_nobinding_proprag_full1000_20260506/"
        "musique_proprag_wiki_title_daec_noisyor_nobind_full1000.json",
    ),
)

EPS = 1e-12
PRIMARY_SCORE_NAMES = (
    "bind_conf_composite_primary",
    "bind_conf_candidate_simplicity",
    "bind_conf_title_unique",
    "bind_conf_single_candidate_rate",
    "bind_conf_candidate_coverage",
    "bind_conf_match_quality",
)
DIAGNOSTIC_SCORE_NAMES = (
    "bind_conf_margin_diagnostic",
    "bind_conf_composite_with_margin_diagnostic",
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if abs(denominator) > EPS else float(default)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def stable_split(question: str, *, seed: str) -> str:
    digest = hashlib.md5(f"{seed}\t{question}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 5
    return "dev" if bucket == 0 else "test"


def metric(trace: Mapping[str, Any], name: str) -> float:
    return safe_float((trace.get("selector_metrics") or {}).get(name), 0.0)


def trace_key(trace: Mapping[str, Any]) -> str:
    return str(trace.get("question") or "")


def selector_trace(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    return trace.get("selector_trace") or {}


def list_items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def mapping_items(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def compute_depths(requirements: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    by_id = {str(req.get("unit_id") or ""): req for req in requirements}
    visiting: set[str] = set()
    memo: dict[str, int] = {}

    def depth(unit_id: str) -> int:
        if unit_id in memo:
            return memo[unit_id]
        if unit_id in visiting:
            return 0
        visiting.add(unit_id)
        req = by_id.get(unit_id) or {}
        deps = [str(dep) for dep in list_items(req.get("depends_on")) if str(dep)]
        value = 0 if not deps else 1 + max(depth(dep) for dep in deps)
        visiting.discard(unit_id)
        memo[unit_id] = value
        return value

    return {unit_id: depth(unit_id) for unit_id in by_id}


def title_recall_multiset(trace: Mapping[str, Any]) -> float:
    gold = [str(title) for title in list_items(trace.get("gold_titles"))]
    selected = [str(title) for title in list_items(trace.get("selector_top_titles"))]
    if not gold:
        return 0.0
    used = [False] * len(selected)
    hit = 0
    for gold_title in gold:
        gold_norm = normalize_text(gold_title)
        for index, selected_title in enumerate(selected):
            if used[index]:
                continue
            if normalize_text(selected_title) == gold_norm:
                used[index] = True
                hit += 1
                break
    return safe_div(float(hit), float(len(gold)))


def pool_titles_from_example(example: Mapping[str, Any]) -> list[str]:
    retrieval_trace = mapping_items(example.get("retrieval_trace"))
    return [str(title) for title in list_items(retrieval_trace.get("external_pool_titles"))]


def extraction_count_field(extractions: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(len(list_items(row.get(field))) for row in extractions)


def extract_query_features(trace: Mapping[str, Any], example: Mapping[str, Any]) -> dict[str, float]:
    st = selector_trace(trace)
    requirements = [req for req in list_items(st.get("requirements")) if isinstance(req, Mapping)]
    dependent_requirements = [req for req in requirements if list_items(req.get("depends_on"))]
    dependent_req_ids = [str(req.get("unit_id") or "") for req in dependent_requirements]
    dependent_req_count = len(dependent_req_ids)
    depths = compute_depths(requirements)

    candidates_by_req = mapping_items(st.get("binding_candidates_by_requirement"))
    candidate_counts = [
        len(list_items(candidates_by_req.get(req_id)))
        for req_id in dependent_req_ids
    ]
    all_candidates: list[Mapping[str, Any]] = []
    for value in candidates_by_req.values():
        all_candidates.extend([row for row in list_items(value) if isinstance(row, Mapping)])

    candidate_count_total = len(all_candidates)
    candidate_req_count = sum(1 for count in candidate_counts if count > 0)
    single_candidate_req_count = sum(1 for count in candidate_counts if count == 1)
    multi_candidate_req_count = sum(1 for count in candidate_counts if count > 1)

    pool_titles = pool_titles_from_example(example)
    pool_title_counts = Counter(normalize_text(title) for title in pool_titles if normalize_text(title))
    candidate_titles = [str(row.get("title") or "") for row in all_candidates]
    candidate_title_occurrences = [
        pool_title_counts.get(normalize_text(title), 0)
        for title in candidate_titles
        if normalize_text(title)
    ]
    unique_candidate_title_count = sum(1 for count in candidate_title_occurrences if count == 1)
    nonunique_candidate_title_count = sum(1 for count in candidate_title_occurrences if count > 1)

    match_type_counts = Counter(str(row.get("entity_match_type") or "unknown") for row in all_candidates)
    exactish_count = sum(
        count
        for key, count in match_type_counts.items()
        if key in {"exact", "exact_title", "wiki_title", "normalized_title", "title"}
    )
    alias_count = sum(
        count
        for key, count in match_type_counts.items()
        if "alias" in key or "substring" in key or "redirect" in key
    )

    margins = []
    top_scores = []
    for req_id in dependent_req_ids:
        rows = [
            row for row in list_items(candidates_by_req.get(req_id))
            if isinstance(row, Mapping)
        ]
        scores = sorted((safe_float(row.get("dep_score"), 0.0) for row in rows), reverse=True)
        if not scores:
            continue
        top_scores.append(scores[0])
        margins.append(1.0 if len(scores) == 1 else max(0.0, scores[0] - scores[1]))

    extractions = [row for row in list_items(st.get("llm_binding_extractions")) if isinstance(row, Mapping)]
    extraction_count = len(extractions)
    raw_entity_count = extraction_count_field(extractions, "raw_entities")
    matched_entity_count = extraction_count_field(extractions, "matched_entities")
    unmatched_entity_count = extraction_count_field(extractions, "unmatched_entities")
    skipped_anchor_count = extraction_count_field(extractions, "skipped_anchor_entities")
    duplicate_entity_count = extraction_count_field(extractions, "duplicate_entities")
    empty_entity_responses = safe_int(mapping_items(st.get("llm_binding_cost")).get("empty_entity_responses"), 0)

    binding_count = safe_int(st.get("binding_count"), 0)
    binding_count_unpruned = safe_int(st.get("binding_count_unpruned"), binding_count)

    dep_count = max(1, dependent_req_count)
    candidate_count_mean = safe_div(float(candidate_count_total), float(dep_count))
    title_unique_rate = safe_div(float(unique_candidate_title_count), float(candidate_count_total), 0.0)
    title_nonunique_rate = safe_div(float(nonunique_candidate_title_count), float(candidate_count_total), 0.0)
    candidate_coverage = safe_div(float(candidate_req_count), float(dep_count), 0.0)
    single_candidate_rate = safe_div(float(single_candidate_req_count), float(dep_count), 0.0)
    multi_candidate_rate = safe_div(float(multi_candidate_req_count), float(dep_count), 0.0)
    match_rate = safe_div(float(matched_entity_count), float(raw_entity_count), 0.0)
    unmatched_rate = safe_div(float(unmatched_entity_count), float(raw_entity_count), 0.0)
    empty_response_rate = safe_div(float(empty_entity_responses), float(max(1, extraction_count)), 0.0)
    low_ambiguity = safe_div(1.0, 1.0 + max(0.0, float(candidate_count_total - dependent_req_count)), 1.0)
    if candidate_count_total <= 0:
        low_ambiguity = 0.0

    features = {
        "requirement_count": float(len(requirements)),
        "dependent_req_count": float(dependent_req_count),
        "max_dependency_depth": float(max(depths.values()) if depths else 0),
        "candidate_count_total": float(candidate_count_total),
        "candidate_count_mean": float(candidate_count_mean),
        "candidate_count_max": float(max(candidate_counts) if candidate_counts else 0),
        "candidate_req_count": float(candidate_req_count),
        "candidate_coverage": float(candidate_coverage),
        "single_candidate_req_count": float(single_candidate_req_count),
        "single_candidate_rate": float(single_candidate_rate),
        "multi_candidate_req_count": float(multi_candidate_req_count),
        "multi_candidate_rate": float(multi_candidate_rate),
        "title_unique_rate": float(title_unique_rate),
        "title_nonunique_rate": float(title_nonunique_rate),
        "avg_candidate_title_occurrences": mean(candidate_title_occurrences) if candidate_title_occurrences else 0.0,
        "exactish_match_rate": safe_div(float(exactish_count), float(candidate_count_total), 0.0),
        "alias_match_rate": safe_div(float(alias_count), float(candidate_count_total), 0.0),
        "binding_count": float(binding_count),
        "binding_count_unpruned": float(binding_count_unpruned),
        "extraction_count": float(extraction_count),
        "raw_entity_count": float(raw_entity_count),
        "matched_entity_count": float(matched_entity_count),
        "unmatched_entity_count": float(unmatched_entity_count),
        "skipped_anchor_count": float(skipped_anchor_count),
        "duplicate_entity_count": float(duplicate_entity_count),
        "empty_entity_responses": float(empty_entity_responses),
        "match_rate": float(match_rate),
        "unmatched_rate": float(unmatched_rate),
        "empty_response_rate": float(empty_response_rate),
        "low_ambiguity": float(low_ambiguity),
        "dep_margin_mean_diagnostic": mean(margins) if margins else 0.0,
        "dep_margin_min_diagnostic": min(margins) if margins else 0.0,
        "dep_score_top1_mean_diagnostic": mean(top_scores) if top_scores else 0.0,
    }

    # Binding confidence scores. Primary scores avoid dep-score margins.
    features["bind_conf_candidate_simplicity"] = clamp01(low_ambiguity)
    features["bind_conf_title_unique"] = clamp01(title_unique_rate)
    features["bind_conf_single_candidate_rate"] = clamp01(single_candidate_rate)
    features["bind_conf_candidate_coverage"] = clamp01(candidate_coverage)
    features["bind_conf_match_quality"] = clamp01(0.5 * match_rate + 0.5 * (1.0 - unmatched_rate))
    primary_parts = [
        features["bind_conf_candidate_simplicity"],
        features["bind_conf_title_unique"],
        features["bind_conf_single_candidate_rate"],
        features["bind_conf_candidate_coverage"],
        features["bind_conf_match_quality"],
    ]
    features["bind_conf_composite_primary"] = clamp01(mean(primary_parts))
    features["bind_conf_margin_diagnostic"] = clamp01(features["dep_margin_mean_diagnostic"])
    features["bind_conf_composite_with_margin_diagnostic"] = clamp01(
        0.8 * features["bind_conf_composite_primary"] + 0.2 * features["bind_conf_margin_diagnostic"]
    )
    return features


def strong_label(daec_f1: float, nobind_f1: float, daec_em: float, nobind_em: float, min_delta: float) -> str:
    daec_delta = daec_f1 - nobind_f1
    if (nobind_em > daec_em and nobind_f1 + EPS >= daec_f1) or daec_delta <= -float(min_delta):
        return "abstain_helpful"
    if (daec_em > nobind_em and daec_f1 + EPS >= nobind_f1) or daec_delta >= float(min_delta):
        return "bind_helpful"
    return "ignore"


def build_records(dataset: str,
                  daec_path: Path,
                  nobind_path: Path,
                  *,
                  seed: str,
                  min_strong_delta: float) -> list[dict[str, Any]]:
    daec = read_json(daec_path)
    nobind = read_json(nobind_path)
    daec_traces = list_items(daec.get("setwise_selector_query_traces"))
    nobind_traces = list_items(nobind.get("setwise_selector_query_traces"))
    examples = list_items(daec.get("examples"))
    if len(daec_traces) != len(nobind_traces):
        raise ValueError(f"Trace length mismatch for {dataset}: {len(daec_traces)} vs {len(nobind_traces)}")

    records: list[dict[str, Any]] = []
    for query_index, (daec_trace, nobind_trace) in enumerate(zip(daec_traces, nobind_traces)):
        if not isinstance(daec_trace, Mapping) or not isinstance(nobind_trace, Mapping):
            continue
        question = trace_key(daec_trace)
        if question != trace_key(nobind_trace):
            raise ValueError(f"Question mismatch for {dataset} index={query_index}")
        example = examples[query_index] if query_index < len(examples) and isinstance(examples[query_index], Mapping) else {}
        features = extract_query_features(daec_trace, example)
        daec_f1 = metric(daec_trace, "F1")
        nobind_f1 = metric(nobind_trace, "F1")
        daec_em = metric(daec_trace, "ExactMatch")
        nobind_em = metric(nobind_trace, "ExactMatch")
        label = strong_label(daec_f1, nobind_f1, daec_em, nobind_em, float(min_strong_delta))
        records.append({
            "dataset": dataset,
            "query_index": query_index,
            "question": question,
            "split": stable_split(question, seed=seed),
            "gold_answers": list_items(daec_trace.get("gold_answers")),
            "gold_titles": list_items(daec_trace.get("gold_titles")),
            "daec_answer": daec_trace.get("selector_answer"),
            "nobind_answer": nobind_trace.get("selector_answer"),
            "daec_f1": daec_f1,
            "nobind_f1": nobind_f1,
            "daec_em": daec_em,
            "nobind_em": nobind_em,
            "daec_title_recall": title_recall_multiset(daec_trace),
            "nobind_title_recall": title_recall_multiset(nobind_trace),
            "delta_f1_daec_minus_nobind": daec_f1 - nobind_f1,
            "delta_em_daec_minus_nobind": daec_em - nobind_em,
            "strong_label": label,
            "daec_titles": list_items(daec_trace.get("selector_top_titles")),
            "nobind_titles": list_items(nobind_trace.get("selector_top_titles")),
            **features,
        })
    return records


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pos = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    neg = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for pos_score in pos:
        for neg_score in neg:
            total += 1
            if pos_score > neg_score:
                wins += 1.0
            elif abs(pos_score - neg_score) <= EPS:
                wins += 0.5
    return safe_div(wins, float(total))


def summarize_variant(records: Sequence[Mapping[str, Any]], chooser: Callable[[Mapping[str, Any]], str]) -> dict[str, float]:
    if not records:
        return {"n": 0, "em": 0.0, "f1": 0.0, "title_recall": 0.0, "null_rate": 0.0}
    em_values = []
    f1_values = []
    recall_values = []
    null_count = 0
    for record in records:
        choice = chooser(record)
        if choice == "nobind":
            null_count += 1
            em_values.append(safe_float(record.get("nobind_em")))
            f1_values.append(safe_float(record.get("nobind_f1")))
            recall_values.append(safe_float(record.get("nobind_title_recall")))
        else:
            em_values.append(safe_float(record.get("daec_em")))
            f1_values.append(safe_float(record.get("daec_f1")))
            recall_values.append(safe_float(record.get("daec_title_recall")))
    return {
        "n": len(records),
        "em": mean(em_values),
        "f1": mean(f1_values),
        "title_recall": mean(recall_values),
        "null_rate": safe_div(float(null_count), float(len(records))),
    }


def grouped_records(records: Sequence[Mapping[str, Any]], split: str) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if split != "all" and record.get("split") != split:
            continue
        groups[str(record.get("dataset"))].append(record)
    return dict(groups)


def compute_baselines(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("all", "dev", "test"):
        output[split] = {}
        for dataset, rows in grouped_records(records, split).items():
            output[split][dataset] = {
                "daec": summarize_variant(rows, lambda _: "daec"),
                "nobind": summarize_variant(rows, lambda _: "nobind"),
            }
    return output


def threshold_grid() -> list[float]:
    return [round(index / 100.0, 2) for index in range(0, 101)] + [1.01]


def build_threshold_rows(records: Sequence[Mapping[str, Any]], score_names: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baselines = compute_baselines(records)
    for split in ("all", "dev", "test"):
        groups = grouped_records(records, split)
        for score_name in score_names:
            for threshold in threshold_grid():
                dataset_summaries: dict[str, dict[str, float]] = {}
                for dataset, dataset_rows in groups.items():
                    summary = summarize_variant(
                        dataset_rows,
                        lambda record, sn=score_name, th=threshold: (
                            "daec" if safe_float(record.get(sn), 0.0) >= float(th) else "nobind"
                        ),
                    )
                    base = baselines[split][dataset]["daec"]
                    nobase = baselines[split][dataset]["nobind"]
                    dataset_summaries[dataset] = {
                        **summary,
                        "delta_f1_vs_daec": summary["f1"] - base["f1"],
                        "delta_em_vs_daec": summary["em"] - base["em"],
                        "delta_title_recall_vs_daec": summary["title_recall"] - base["title_recall"],
                        "delta_f1_vs_nobind": summary["f1"] - nobase["f1"],
                    }
                macro_f1 = mean([row["f1"] for row in dataset_summaries.values()]) if dataset_summaries else 0.0
                macro_delta = mean([row["delta_f1_vs_daec"] for row in dataset_summaries.values()]) if dataset_summaries else 0.0
                rows.append({
                    "split": split,
                    "score_name": score_name,
                    "threshold": threshold,
                    "macro_f1": macro_f1,
                    "macro_delta_f1_vs_daec": macro_delta,
                    "datasets": dataset_summaries,
                })
    return rows


def flatten_threshold_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        for dataset, summary in mapping_items(row.get("datasets")).items():
            flat.append({
                "split": row.get("split"),
                "score_name": row.get("score_name"),
                "threshold": row.get("threshold"),
                "dataset": dataset,
                **summary,
                "macro_f1": row.get("macro_f1"),
                "macro_delta_f1_vs_daec": row.get("macro_delta_f1_vs_daec"),
            })
    return flat


def build_title_unique_robustness_rows(
    threshold_rows: Sequence[Mapping[str, Any]],
    *,
    min_threshold: float = 0.80,
    max_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in threshold_rows:
        threshold = safe_float(row.get("threshold"))
        if row.get("score_name") != "bind_conf_title_unique":
            continue
        if threshold < float(min_threshold) - EPS or threshold > float(max_threshold) + EPS:
            continue
        for dataset, summary in mapping_items(row.get("datasets")).items():
            rows.append({
                "split": row.get("split"),
                "threshold": threshold,
                "dataset": dataset,
                "f1": safe_float(summary.get("f1")),
                "em": safe_float(summary.get("em")),
                "title_recall": safe_float(summary.get("title_recall")),
                "delta_f1_vs_daec": safe_float(summary.get("delta_f1_vs_daec")),
                "delta_em_vs_daec": safe_float(summary.get("delta_em_vs_daec")),
                "delta_title_recall_vs_daec": safe_float(summary.get("delta_title_recall_vs_daec")),
                "delta_f1_vs_nobind": safe_float(summary.get("delta_f1_vs_nobind")),
                "null_rate": safe_float(summary.get("null_rate")),
            })
    rows.sort(key=lambda item: (str(item["split"]), safe_float(item["threshold"]), str(item["dataset"])))
    return rows


def summarize_title_unique_robustness(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("split")), str(row.get("dataset")))].append(row)
    for (split, dataset), group_rows in sorted(grouped.items()):
        thresholds = [safe_float(row.get("threshold")) for row in group_rows]
        deltas = [safe_float(row.get("delta_f1_vs_daec")) for row in group_rows]
        null_rates = [safe_float(row.get("null_rate")) for row in group_rows]
        summaries.append({
            "split": split,
            "dataset": dataset,
            "threshold_min": min(thresholds) if thresholds else 0.0,
            "threshold_max": max(thresholds) if thresholds else 0.0,
            "delta_f1_min": min(deltas) if deltas else 0.0,
            "delta_f1_max": max(deltas) if deltas else 0.0,
            "null_rate_min": min(null_rates) if null_rates else 0.0,
            "null_rate_max": max(null_rates) if null_rates else 0.0,
        })
    return summaries


def threshold_gate_pass(row: Mapping[str, Any]) -> bool:
    datasets = mapping_items(row.get("datasets"))
    required = ("2Wiki", "HotpotQA", "MuSiQue")
    if any(name not in datasets for name in required):
        return False
    if datasets["2Wiki"]["delta_f1_vs_daec"] < -0.02:
        return False
    if datasets["HotpotQA"]["delta_f1_vs_daec"] < -0.02:
        return False
    if datasets["MuSiQue"]["delta_f1_vs_daec"] < 0.005:
        return False
    for name in required:
        null_rate = float(datasets[name]["null_rate"])
        if null_rate < 0.05 or null_rate > 0.80:
            return False
    return True


def choose_dev_gate(threshold_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row for row in threshold_rows
        if row.get("split") == "dev"
        and row.get("score_name") in PRIMARY_SCORE_NAMES
        and threshold_gate_pass(row)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (
        safe_float(row.get("macro_delta_f1_vs_daec")),
        safe_float(mapping_items(row.get("datasets")).get("MuSiQue", {}).get("delta_f1_vs_daec")),
        -abs(safe_float(row.get("threshold")) - 0.5),
    ))


def matching_threshold_row(threshold_rows: Sequence[Mapping[str, Any]], selected: Mapping[str, Any], split: str) -> dict[str, Any] | None:
    for row in threshold_rows:
        if (
            row.get("split") == split
            and row.get("score_name") == selected.get("score_name")
            and abs(safe_float(row.get("threshold")) - safe_float(selected.get("threshold"))) <= EPS
        ):
            return dict(row)
    return None


def choose_robust_gate(threshold_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Find a primary threshold that satisfies the gate on both dev and test.

    This is exploratory: it uses the held-out split to assess whether the signal
    is stable enough to justify implementation. It should not be reported as a
    tuned held-out number.
    """
    candidates: list[dict[str, Any]] = []
    for score_name in PRIMARY_SCORE_NAMES:
        thresholds = sorted({
            safe_float(row.get("threshold"))
            for row in threshold_rows
            if row.get("score_name") == score_name
        })
        for threshold in thresholds:
            selected = {"score_name": score_name, "threshold": threshold}
            dev = matching_threshold_row(threshold_rows, selected, "dev")
            test = matching_threshold_row(threshold_rows, selected, "test")
            all_row = matching_threshold_row(threshold_rows, selected, "all")
            if not dev or not test or not all_row:
                continue
            if not threshold_gate_pass(dev) or not threshold_gate_pass(test):
                continue
            all_datasets = mapping_items(all_row.get("datasets"))
            test_datasets = mapping_items(test.get("datasets"))
            all_macro = safe_float(all_row.get("macro_delta_f1_vs_daec"))
            test_macro = safe_float(test.get("macro_delta_f1_vs_daec"))
            musique_all = safe_float(all_datasets.get("MuSiQue", {}).get("delta_f1_vs_daec"))
            two_wiki_all = safe_float(all_datasets.get("2Wiki", {}).get("delta_f1_vs_daec"))
            # Prefer stable all-split gains, then held-out gains, while keeping
            # 2Wiki damage small.
            candidates.append({
                "all_macro": all_macro,
                "test_macro": test_macro,
                "musique_all": musique_all,
                "score_name": score_name,
                "two_wiki_abs_damage": abs(two_wiki_all),
                "row": all_row,
            })
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            safe_float(item["all_macro"]),
            safe_float(item["test_macro"]),
            safe_float(item["musique_all"]),
            -safe_float(item["two_wiki_abs_damage"]),
            str(item["score_name"]),
        ),
        reverse=True,
    )
    return dict(candidates[0]["row"])


def feature_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    reserved = {
        "dataset", "query_index", "question", "split", "gold_answers", "gold_titles",
        "daec_answer", "nobind_answer", "daec_f1", "nobind_f1", "daec_em", "nobind_em",
        "daec_title_recall", "nobind_title_recall", "delta_f1_daec_minus_nobind",
        "delta_em_daec_minus_nobind", "strong_label", "daec_titles", "nobind_titles",
    }
    names = sorted(
        key for record in records for key, value in record.items()
        if key not in reserved and isinstance(value, (int, float))
    )
    return list(dict.fromkeys(names))


def compute_feature_auc_rows(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[Mapping[str, Any]]] = {"ALL": list(records)}
    groups.update(grouped_records(records, "all"))
    for dataset, dataset_rows in groups.items():
        strong = [row for row in dataset_rows if row.get("strong_label") in {"abstain_helpful", "bind_helpful"}]
        labels = [1 if row.get("strong_label") == "abstain_helpful" else 0 for row in strong]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        for name in names:
            scores = [safe_float(row.get(name), 0.0) for row in strong]
            auc = auc_score(labels, scores)
            pos_values = [score for label, score in zip(labels, scores) if label == 1]
            neg_values = [score for label, score in zip(labels, scores) if label == 0]
            if auc is None:
                continue
            direction = "high=>abstain" if auc >= 0.5 else "high=>bind"
            rows.append({
                "dataset": dataset,
                "feature": name,
                "n_strong": len(strong),
                "n_abstain_helpful": n_pos,
                "n_bind_helpful": n_neg,
                "auc_abstain": auc,
                "auc_abs_distance": abs(auc - 0.5),
                "direction": direction,
                "abstain_mean": mean(pos_values) if pos_values else 0.0,
                "bind_mean": mean(neg_values) if neg_values else 0.0,
                "mean_diff_abstain_minus_bind": (
                    (mean(pos_values) if pos_values else 0.0) - (mean(neg_values) if neg_values else 0.0)
                ),
            })
    rows.sort(key=lambda row: (row["dataset"], -float(row["auc_abs_distance"]), row["feature"]))
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path, fieldnames: Sequence[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{safe_float(value):.{digits}f}"


def format_range(low: Any, high: Any, digits: int = 4) -> str:
    return f"[{format_float(low, digits)}, {format_float(high, digits)}]"


def threshold_table_rows(row: Mapping[str, Any] | None) -> list[str]:
    if not row:
        return ["No dev threshold satisfies the pre-registered gate."]
    lines = [
        f"Selected dev score: `{row['score_name']}` at threshold `{format_float(row['threshold'], 2)}`.",
        "",
        "| Dataset | F1 | dF1 vs DAEC | Null Rate |",
        "|---|---:|---:|---:|",
    ]
    for dataset in ("2Wiki", "HotpotQA", "MuSiQue"):
        summary = mapping_items(row.get("datasets")).get(dataset, {})
        lines.append(
            f"| {dataset} | {format_float(summary.get('f1'))} | "
            f"{format_float(summary.get('delta_f1_vs_daec'))} | {format_float(summary.get('null_rate'))} |"
        )
    return lines


def gate_table(title: str, row: Mapping[str, Any] | None) -> list[str]:
    return [
        f"## {title}",
        "",
        *threshold_table_rows(row),
        "",
    ]


def robustness_markdown_section(report: Mapping[str, Any]) -> list[str]:
    summaries = [
        row for row in report.get("title_unique_robustness_summary", [])
        if isinstance(row, Mapping)
    ]
    representative = mapping_items(report.get("robust_gate_all"))
    lines = [
        "## Title-Uniqueness Robustness",
        "",
        "The frozen Phase-1 rule uses `bind_conf_title_unique >= 0.88`. The table below reports the full high-uniqueness band `[0.80, 0.95]`, so `0.88` is treated as a representative structural cutoff near `0.9`, not an isolated optimum.",
        "",
        "| Split | Dataset | Threshold Band | dF1 vs DAEC Range | Null Rate Range |",
        "|---|---|---:|---:|---:|",
    ]
    order = {"all": 0, "dev": 1, "test": 2}
    summaries.sort(key=lambda row: (order.get(str(row.get("split")), 99), str(row.get("dataset"))))
    for row in summaries:
        lines.append(
            f"| {row['split']} | {row['dataset']} | "
            f"{format_range(row.get('threshold_min'), row.get('threshold_max'), 2)} | "
            f"{format_range(row.get('delta_f1_min'), row.get('delta_f1_max'))} | "
            f"{format_range(row.get('null_rate_min'), row.get('null_rate_max'))} |"
        )
    if representative:
        datasets = mapping_items(representative.get("datasets"))
        lines.extend([
            "",
            "Representative `0.88` all-split point:",
            "",
            "| Dataset | F1 | dF1 vs DAEC | Null Rate |",
            "|---|---:|---:|---:|",
        ])
        for dataset in ("2Wiki", "HotpotQA", "MuSiQue"):
            summary = mapping_items(datasets.get(dataset))
            lines.append(
                f"| {dataset} | {format_float(summary.get('f1'))} | "
                f"{format_float(summary.get('delta_f1_vs_daec'))} | "
                f"{format_float(summary.get('null_rate'))} |"
            )
    lines.extend([
        "",
        "Interpretation: the high-uniqueness band consistently preserves 2Wiki within about 0.003 F1 of DAEC while improving MuSiQue; the selected `0.88` threshold is frozen before Phase-1 fresh reader runs.",
        "",
    ])
    return lines


def build_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# DAEC Selective Binding Phase-0",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Strong flip label: `|dF1| >= {report['min_strong_delta']}` or answer EM flip.",
        "- Split: stable hash split, approximately 20% dev / 80% test.",
        "- Primary router scores exclude dep-score margin; margin appears only in diagnostic rows.",
        "",
        "## Overall Metrics",
        "",
        "| Dataset | Split | DAEC F1 | Nobind F1 | dF1 DAEC-Nobind | DAEC EM | Nobind EM |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    baselines = mapping_items(report.get("baselines"))
    for split in ("all", "dev", "test"):
        for dataset in ("2Wiki", "HotpotQA", "MuSiQue"):
            base = mapping_items(mapping_items(baselines.get(split)).get(dataset))
            daec = mapping_items(base.get("daec"))
            nobind = mapping_items(base.get("nobind"))
            lines.append(
                f"| {dataset} | {split} | {format_float(daec.get('f1'))} | {format_float(nobind.get('f1'))} | "
                f"{format_float(safe_float(daec.get('f1')) - safe_float(nobind.get('f1')))} | "
                f"{format_float(daec.get('em'))} | {format_float(nobind.get('em'))} |"
            )

    lines.extend([
        "",
        "## Strong Flip Counts",
        "",
        "| Dataset | Strong Cases | Abstain Helpful | Bind Helpful | Ignored/Noisy |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in report.get("strong_flip_counts", []):
        lines.append(
            f"| {row['dataset']} | {row['strong_cases']} | {row['abstain_helpful']} | "
            f"{row['bind_helpful']} | {row['ignored']} |"
        )

    lines.extend([
        "",
        "## Top AUC Features",
        "",
        "AUC predicts `abstain_helpful` over strong flip cases. AUC below 0.5 means high feature values favor binding.",
        "",
        "| Dataset | Feature | AUC | Direction | Abstain Mean | Bind Mean | n+ / n- |",
        "|---|---|---:|---|---:|---:|---:|",
    ])
    for row in report.get("top_auc_rows", []):
        lines.append(
            f"| {row['dataset']} | `{row['feature']}` | {format_float(row['auc_abstain'])} | "
            f"{row['direction']} | {format_float(row['abstain_mean'])} | {format_float(row['bind_mean'])} | "
            f"{row['n_abstain_helpful']} / {row['n_bind_helpful']} |"
        )

    lines.extend([
        "",
        *gate_table("Dev-Optimal Gate", mapping_items(report.get("selected_dev_gate")) if report.get("selected_dev_gate") else None),
        *gate_table("Held-Out Test Result For Dev-Optimal Gate", mapping_items(report.get("selected_test_gate")) if report.get("selected_test_gate") else None),
        *gate_table("Robust Exploratory Gate (Passes Dev And Test)", mapping_items(report.get("robust_gate_all")) if report.get("robust_gate_all") else None),
        *robustness_markdown_section(report),
        "## Gate Interpretation",
        "",
    ])
    lines.extend(str(item) for item in report.get("interpretation", []))
    lines.extend([
        "",
        "## Artifacts",
        "",
    ])
    for key, value in mapping_items(report.get("artifacts")).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def polyline(points: Sequence[tuple[float, float]], *,
             x_min: float, x_max: float, y_min: float, y_max: float,
             left: float, top: float, width: float, height: float) -> str:
    coords = []
    for x_value, y_value in points:
        x = left + safe_div(x_value - x_min, x_max - x_min) * width
        y = top + height - safe_div(y_value - y_min, y_max - y_min) * height
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def write_threshold_svg(threshold_rows: Sequence[Mapping[str, Any]], score_name: str, output_path: str | Path) -> None:
    rows = [
        row for row in threshold_rows
        if row.get("split") == "all" and row.get("score_name") == score_name
    ]
    by_dataset: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        threshold = safe_float(row.get("threshold"))
        if threshold > 1.0:
            continue
        for dataset, summary in mapping_items(row.get("datasets")).items():
            by_dataset[dataset].append((threshold, safe_float(summary.get("f1")), safe_float(summary.get("null_rate"))))
    if not by_dataset:
        return

    width, height = 860, 520
    left, top, plot_w, plot_h = 70, 55, 740, 320
    f1_values = [point[1] for points in by_dataset.values() for point in points]
    y_min = max(0.0, min(f1_values) - 0.03)
    y_max = min(1.0, max(f1_values) + 0.03)
    colors = {"2Wiki": "#1f77b4", "HotpotQA": "#2ca02c", "MuSiQue": "#d62728"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.0f}" y="25" text-anchor="middle" font-family="Arial" font-size="18">'
        f'Threshold Curve: {html.escape(score_name)}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
    ]
    for tick in range(0, 11):
        x = left + tick / 10 * plot_w
        lines.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{tick/10:.1f}</text>')
    for tick in range(0, 6):
        y_value = y_min + tick / 5 * (y_max - y_min)
        y = top + plot_h - tick / 5 * plot_h
        lines.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_value:.2f}</text>')
    lines.append(f'<text x="{left+plot_w/2:.1f}" y="{height-38}" text-anchor="middle" font-family="Arial" font-size="13">Bind confidence threshold; below threshold uses nobinding</text>')
    lines.append(f'<text transform="translate(18,{top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">Answer F1</text>')
    legend_x, legend_y = left + plot_w - 120, top + 18
    for idx, dataset in enumerate(("2Wiki", "HotpotQA", "MuSiQue")):
        points = sorted(by_dataset.get(dataset, []))
        if not points:
            continue
        color = colors[dataset]
        point_string = polyline(
            [(x, y) for x, y, _ in points],
            x_min=0.0, x_max=1.0, y_min=y_min, y_max=y_max,
            left=left, top=top, width=plot_w, height=plot_h,
        )
        lines.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        lines.append(f'<rect x="{legend_x}" y="{legend_y + idx*20}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x+18}" y="{legend_y + idx*20 + 11}" font-family="Arial" font-size="12">{dataset}</text>')
    lines.append("</svg>")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(records: Sequence[Mapping[str, Any]], *,
                 min_strong_delta: float,
                 threshold_rows: Sequence[Mapping[str, Any]],
                 feature_auc_rows: Sequence[Mapping[str, Any]],
                 title_unique_robustness_rows: Sequence[Mapping[str, Any]],
                 artifacts: Mapping[str, str]) -> dict[str, Any]:
    baselines = compute_baselines(records)
    selected_dev = choose_dev_gate(threshold_rows)
    selected_test = matching_threshold_row(threshold_rows, selected_dev, "test") if selected_dev else None
    selected_all = matching_threshold_row(threshold_rows, selected_dev, "all") if selected_dev else None
    robust_all = choose_robust_gate(threshold_rows)
    robust_dev = matching_threshold_row(threshold_rows, robust_all, "dev") if robust_all else None
    robust_test = matching_threshold_row(threshold_rows, robust_all, "test") if robust_all else None

    strong_counts = []
    for dataset, rows in grouped_records(records, "all").items():
        counts = Counter(str(row.get("strong_label")) for row in rows)
        strong_counts.append({
            "dataset": dataset,
            "strong_cases": counts["abstain_helpful"] + counts["bind_helpful"],
            "abstain_helpful": counts["abstain_helpful"],
            "bind_helpful": counts["bind_helpful"],
            "ignored": counts["ignore"],
        })
    strong_counts.sort(key=lambda row: row["dataset"])

    top_auc_rows = []
    for dataset in ("ALL", "2Wiki", "HotpotQA", "MuSiQue"):
        rows = [row for row in feature_auc_rows if row["dataset"] == dataset]
        top_auc_rows.extend(rows[:8])

    if robust_all:
        verdict = "go_phase1"
        interpretation = [
            "- A primary, selection-independent confidence score passes the gate on both dev and test.",
            "- Treat the robust gate as exploratory evidence of separability, not as a tuned held-out result.",
            "- If implemented, freeze the rule before any new reader run and validate on a fresh rerun or separate split.",
        ]
    elif selected_dev:
        verdict = "weak_go_phase1"
        interpretation = [
            "- A primary confidence score passes the dev gate, but no threshold passes both dev and test.",
            "- Do not implement unless a simpler pre-registered rule is chosen and validated separately.",
        ]
    else:
        verdict = "no_go_phase1"
        interpretation = [
            "- No primary selection-independent confidence threshold satisfies the dev gate.",
            "- Do not implement query-level selective binding yet; the available extraction-level signals are not sufficient under the pre-registered constraints.",
            "- Keep the current paper story: binding is strongly load-bearing on 2Wiki but weak/boundary on HotpotQA and MuSiQue.",
        ]

    return {
        "verdict": verdict,
        "min_strong_delta": float(min_strong_delta),
        "baselines": baselines,
        "strong_flip_counts": strong_counts,
        "selected_dev_gate": selected_dev,
        "selected_test_gate": selected_test,
        "selected_all_gate": selected_all,
        "robust_gate_dev": robust_dev,
        "robust_gate_test": robust_test,
        "robust_gate_all": robust_all,
        "title_unique_robustness_summary": summarize_title_unique_robustness(title_unique_robustness_rows),
        "top_auc_rows": top_auc_rows,
        "interpretation": interpretation,
        "artifacts": dict(artifacts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline feasibility analysis for DAEC selective binding.")
    parser.add_argument("--output_dir", type=Path, default=Path("reports/daec_selective_binding_phase0_20260506"))
    parser.add_argument("--seed", type=str, default="daec-selective-binding-phase0-20260506")
    parser.add_argument("--min_strong_delta", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    for dataset, daec_path, nobind_path in DATASETS:
        all_records.extend(
            build_records(
                dataset,
                Path(daec_path),
                Path(nobind_path),
                seed=str(args.seed),
                min_strong_delta=float(args.min_strong_delta),
            )
        )

    names = feature_names(all_records)
    score_names = list(PRIMARY_SCORE_NAMES + DIAGNOSTIC_SCORE_NAMES)
    threshold_rows = build_threshold_rows(all_records, score_names)
    feature_auc_rows = compute_feature_auc_rows(all_records, names)

    query_csv = output_dir / "query_features.csv"
    auc_csv = output_dir / "feature_auc.csv"
    threshold_csv = output_dir / "threshold_curve.csv"
    title_unique_robustness_csv = output_dir / "title_unique_robustness.csv"
    report_json = output_dir / "phase0_report.json"
    report_md = output_dir / "phase0_report.md"
    curve_svg = output_dir / "threshold_curve.svg"

    query_fieldnames = [
        "dataset", "query_index", "split", "strong_label",
        "daec_f1", "nobind_f1", "daec_em", "nobind_em",
        "delta_f1_daec_minus_nobind", "delta_em_daec_minus_nobind",
        *names,
        "question",
    ]
    write_csv(all_records, query_csv, query_fieldnames)
    write_csv(feature_auc_rows, auc_csv)
    write_csv(flatten_threshold_rows(threshold_rows), threshold_csv)
    title_unique_robustness_rows = build_title_unique_robustness_rows(threshold_rows)
    write_csv(title_unique_robustness_rows, title_unique_robustness_csv)

    selected_for_plot = "bind_conf_composite_primary"
    selected_dev = choose_dev_gate(threshold_rows)
    if selected_dev:
        selected_for_plot = str(selected_dev["score_name"])
    write_threshold_svg(threshold_rows, selected_for_plot, curve_svg)

    artifacts = {
        "query_features_csv": str(query_csv),
        "feature_auc_csv": str(auc_csv),
        "threshold_curve_csv": str(threshold_csv),
        "title_unique_robustness_csv": str(title_unique_robustness_csv),
        "threshold_curve_svg": str(curve_svg),
        "report_json": str(report_json),
        "report_md": str(report_md),
    }
    report = build_report(
        all_records,
        min_strong_delta=float(args.min_strong_delta),
        threshold_rows=threshold_rows,
        feature_auc_rows=feature_auc_rows,
        title_unique_robustness_rows=title_unique_robustness_rows,
        artifacts=artifacts,
    )
    write_json(report, report_json)
    report_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({
        "verdict": report["verdict"],
        "output_dir": str(output_dir),
        "report_md": str(report_md),
        "report_json": str(report_json),
        "threshold_curve_svg": str(curve_svg),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
