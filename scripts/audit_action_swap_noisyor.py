#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import build_query_dependency_graph, build_title_prefixed_windows


DEFAULT_CE_MODEL = "/mnt/nvme/bge-reranker-v2-m3"
DEFAULT_MARGIN = 0.05


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_question(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(100.0 * float(numerator) / float(denominator), 2)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 4)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 4)


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(float(value), -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _variant_support(doc_support_by_position: Mapping[int, float],
                     positions: Sequence[int],
                     *,
                     variant: str) -> float:
    values = sorted(
        [
            float(np.clip(float(doc_support_by_position.get(int(pos), 0.0) or 0.0), 0.0, 1.0))
            for pos in positions
        ],
        reverse=True,
    )
    if not values:
        return 0.0
    if variant == "max":
        return float(values[0])
    if variant == "full":
        residual = 1.0
        for value in values:
            residual *= 1.0 - value
        return float(1.0 - residual)
    if variant == "top2":
        residual = 1.0
        for value in values[:2]:
            residual *= 1.0 - value
        return float(1.0 - residual)
    raise ValueError(f"Unsupported variant: {variant}")


def _collect_ancestor_map(query_graph: Mapping[str, Any]) -> Dict[str, List[str]]:
    parent_map = {
        str(node.get("id") or ""): [
            str(parent)
            for parent in (node.get("parents") or [])
            if str(parent)
        ]
        for node in list(query_graph.get("nodes") or [])
        if str(node.get("id") or "")
    }
    ancestor_map: Dict[str, List[str]] = {}

    def _dfs(node_id: str, seen: set[str]) -> List[str]:
        if node_id in ancestor_map:
            return list(ancestor_map[node_id])
        ordered: List[str] = []
        for parent_id in parent_map.get(node_id, []):
            if parent_id in seen:
                continue
            if parent_id not in ordered:
                ordered.append(parent_id)
            for ancestor_id in _dfs(parent_id, seen | {parent_id}):
                if ancestor_id not in ordered:
                    ordered.append(ancestor_id)
        ancestor_map[node_id] = list(ordered)
        return list(ordered)

    for node_id in list(parent_map):
        _dfs(node_id, {node_id})
    return ancestor_map


def _compute_utility(query_graph: Mapping[str, Any],
                     doc_support_matrix: Mapping[str, Mapping[int, float]],
                     positions: Sequence[int],
                     *,
                     dep_enabled: bool,
                     variant: str,
                     doc_titles: Mapping[int, str]) -> Tuple[float, List[Dict[str, Any]]]:
    ancestor_map = _collect_ancestor_map(query_graph) if dep_enabled else {}
    facet_rows: List[Dict[str, Any]] = []
    total_utility = 0.0
    for node in list(query_graph.get("nodes") or []):
        facet_id = str(node.get("id") or "")
        base_support = _variant_support(doc_support_matrix.get(facet_id, {}), positions, variant=variant)
        effective_support = float(base_support)
        ancestors = ancestor_map.get(facet_id, []) if dep_enabled else []
        for ancestor_id in ancestors:
            effective_support *= _variant_support(doc_support_matrix.get(ancestor_id, {}), positions, variant=variant)
        ranked_positions = sorted(
            [int(pos) for pos in positions],
            key=lambda pos: (
                -float(doc_support_matrix.get(facet_id, {}).get(int(pos), 0.0) or 0.0),
                int(pos),
            ),
        )
        top_positions = ranked_positions[:2]
        facet_rows.append({
            "facet_id": facet_id,
            "facet": str(node.get("facet") or ""),
            "support": _round(base_support),
            "effective_support": _round(effective_support),
            "parents": list(node.get("parents") or []),
            "ancestors": list(ancestors),
            "top_support_positions": list(top_positions),
            "top_support_titles": [doc_titles.get(int(pos), f"pos:{int(pos)}") for pos in top_positions],
        })
        total_utility += float(effective_support)
    return float(total_utility), facet_rows


def _extract_candidate_doc_text(action_row: Mapping[str, Any]) -> str:
    if str(action_row.get("action_type") or "").strip().lower() != "swap":
        return ""
    baseline_docs = list(action_row.get("baseline_docs") or [])
    swapped_docs = list(action_row.get("docs") or [])
    replace_index = action_row.get("replace_incumbent_index")
    if replace_index is not None:
        idx = int(replace_index)
        if 0 <= idx < len(swapped_docs):
            return str(swapped_docs[idx])
    for base_doc, swapped_doc in zip(baseline_docs, swapped_docs):
        if str(base_doc) != str(swapped_doc):
            return str(swapped_doc)
    return ""


def _query_context_map(report_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in list(report_payload.get("expand_assemble_query_traces") or []):
        question_key = _normalize_question(row.get("question", ""))
        if not question_key:
            continue
        expand_trace = dict(row.get("expand_assemble_trace") or {})
        mapping[question_key] = {
            "query_type": row.get("query_type"),
            "grounded_question_entities_preview": list(expand_trace.get("grounded_question_entities_preview") or []),
            "question_entities_preview": list(expand_trace.get("question_entities_preview") or []),
            "query_entities_preview": list(expand_trace.get("query_entities_preview") or []),
            "proposal_query_entities_preview": list(expand_trace.get("proposal_query_entities_preview") or []),
            "query_dependency_mode": expand_trace.get("query_dependency_mode"),
        }
    return mapping


def _extract_policy_actions(report_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    action_map: Dict[str, Dict[str, Any]] = {}
    for row in list(report_payload.get("expand_assemble_query_traces") or []):
        question_key = _normalize_question(row.get("question", ""))
        if not question_key:
            continue
        expand_trace = dict(row.get("expand_assemble_trace") or {})
        if bool(expand_trace.get("action_executed")) and expand_trace.get("action_candidate_pool_position") is not None:
            action_map[question_key] = {
                "action_executed": True,
                "candidate_pool_position": int(expand_trace.get("action_candidate_pool_position")),
                "replace_pool_position": int(expand_trace.get("action_replace_pool_position")),
                "action_type": "swap",
            }
        else:
            action_map[question_key] = {
                "action_executed": False,
                "action_type": "keep",
            }
    return action_map


def _action_key(row: Mapping[str, Any]) -> Tuple[str, int | None, int | None]:
    action_type = str(row.get("action_type") or "keep").strip().lower()
    candidate = row.get("candidate_pool_position")
    replace = row.get("replace_pool_position")
    return (
        action_type,
        int(candidate) if candidate is not None else None,
        int(replace) if replace is not None else None,
    )


def _policy_selected(question_key: str,
                     policy_actions: Mapping[str, Mapping[str, Any]],
                     action_row: Mapping[str, Any]) -> bool:
    policy_row = dict(policy_actions.get(question_key) or {})
    if not policy_row:
        return False
    if not bool(policy_row.get("action_executed")):
        return False
    return (
        int(policy_row.get("candidate_pool_position")) == int(action_row.get("candidate_pool_position"))
        and int(policy_row.get("replace_pool_position")) == int(action_row.get("replace_pool_position"))
    )


def _select_action_rows(oracle_payload: Mapping[str, Any],
                        *,
                        dryrun_actions: Mapping[str, Mapping[str, Any]],
                        judge_actions: Mapping[str, Mapping[str, Any]],
                        negative_sample_size: int,
                        seed: int) -> List[Dict[str, Any]]:
    legal_swap_rows = [
        dict(row)
        for row in list(oracle_payload.get("action_results") or [])
        if str(row.get("action_type") or "").strip().lower() == "swap"
        and bool(row.get("is_legal"))
    ]
    for row in legal_swap_rows:
        delta_metrics = dict(row.get("delta_metrics") or {})
        row["oracle_positive"] = bool(
            float(delta_metrics.get("F1", 0.0) or 0.0) > 0.0
            or float(delta_metrics.get("ExactMatch", 0.0) or 0.0) > 0.0
        )
        question_key = _normalize_question(row.get("question", ""))
        row["selected_by_dryrun"] = _policy_selected(question_key, dryrun_actions, row)
        row["selected_by_judge"] = _policy_selected(question_key, judge_actions, row)

    positive_rows = [row for row in legal_swap_rows if bool(row.get("oracle_positive"))]
    policy_rows = [row for row in legal_swap_rows if row.get("selected_by_dryrun") or row.get("selected_by_judge")]
    selected_keys = {
        (
            _normalize_question(row.get("question", "")),
            _action_key(row),
        )
        for row in positive_rows + policy_rows
    }
    negative_candidates = [
        row for row in legal_swap_rows
        if not bool(row.get("oracle_positive"))
        and (
            _normalize_question(row.get("question", "")),
            _action_key(row),
        ) not in selected_keys
    ]
    random.Random(int(seed)).shuffle(negative_candidates)
    sampled_negative_rows = negative_candidates[:max(int(negative_sample_size), 0)]
    combined_rows: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, Tuple[str, int | None, int | None]]] = set()
    for row in positive_rows + policy_rows + sampled_negative_rows:
        key = (_normalize_question(row.get("question", "")), _action_key(row))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        combined_rows.append(dict(row))
    return combined_rows


def _load_ce_reranker(model_name: str, device: str):
    from FlagEmbedding import FlagReranker

    return FlagReranker(model_name, use_fp16=True, devices=[str(device)])


def _compute_support_audit(question: str,
                           baseline_docs: Sequence[str],
                           candidate_doc: str,
                           *,
                           query_entities: Sequence[str],
                           ce_reranker: Any) -> Dict[str, Any]:
    local_docs = [str(doc) for doc in baseline_docs] + [str(candidate_doc)]
    doc_titles = {
        idx: str(doc.split("\n", 1)[0]).strip()
        for idx, doc in enumerate(local_docs)
    }
    query_graphs = {
        "flat": build_query_dependency_graph(
            question,
            query_entities=query_entities,
            action_mode="action_swap_noisyor_flat",
        ),
        "dep": build_query_dependency_graph(
            question,
            query_entities=query_entities,
            action_mode="action_swap_noisyor_dep",
        ),
    }

    windows_by_doc: Dict[int, List[str]] = {
        idx: build_title_prefixed_windows(doc_text)
        for idx, doc_text in enumerate(local_docs)
    }

    support_by_graph: Dict[str, Dict[str, Dict[int, float]]] = {}
    top_window_by_graph: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for graph_name, query_graph in query_graphs.items():
        pairs: List[List[str]] = []
        pair_index: List[Tuple[str, int, str]] = []
        graph_support: Dict[str, Dict[int, float]] = {}
        graph_top_windows: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for node in list(query_graph.get("nodes") or []):
            facet_id = str(node.get("id") or "")
            facet_text = str(node.get("facet") or "")
            graph_support[facet_id] = {}
            graph_top_windows[facet_id] = {}
            for doc_pos, windows in windows_by_doc.items():
                graph_support[facet_id][int(doc_pos)] = 0.0
                graph_top_windows[facet_id][int(doc_pos)] = {
                    "score": 0.0,
                    "window": "",
                }
                for window_text in windows:
                    pairs.append([facet_text, window_text])
                    pair_index.append((facet_id, int(doc_pos), window_text))
        raw_scores = ce_reranker.compute_score(pairs) if pairs else []
        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]
        for (facet_id, doc_pos, window_text), raw_score in zip(pair_index, list(raw_scores)):
            score = _sigmoid(float(raw_score))
            if score > float(graph_support[facet_id].get(int(doc_pos), 0.0)):
                graph_support[facet_id][int(doc_pos)] = float(score)
                graph_top_windows[facet_id][int(doc_pos)] = {
                    "score": float(score),
                    "window": str(window_text),
                }
        support_by_graph[graph_name] = graph_support
        top_window_by_graph[graph_name] = graph_top_windows

    return {
        "query_graphs": query_graphs,
        "doc_titles": doc_titles,
        "support_by_graph": support_by_graph,
        "top_window_by_graph": top_window_by_graph,
    }


def _best_doc_window(top_window_by_facet: Mapping[str, Mapping[int, Mapping[str, Any]]],
                     doc_pos: int) -> Dict[str, Any]:
    best: Dict[str, Any] = {"score": 0.0, "window": "", "facet_id": None}
    for facet_id, by_doc in top_window_by_facet.items():
        row = dict(by_doc.get(int(doc_pos)) or {})
        score = float(row.get("score", 0.0) or 0.0)
        if score > float(best.get("score", 0.0) or 0.0):
            best = {
                "score": float(score),
                "window": str(row.get("window") or ""),
                "facet_id": str(facet_id),
            }
    return best


def _compute_query_audit_rows(action_rows: Sequence[Mapping[str, Any]],
                              *,
                              query_context: Mapping[str, Any],
                              ce_reranker: Any,
                              margin: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not action_rows:
        return [], {}
    question = str(action_rows[0].get("question") or "")
    baseline_docs = list(action_rows[0].get("baseline_docs") or [])
    query_entities = (
        list(query_context.get("grounded_question_entities_preview") or [])
        or list(query_context.get("question_entities_preview") or [])
        or list(query_context.get("query_entities_preview") or [])
        or list(query_context.get("proposal_query_entities_preview") or [])
    )
    support_audit = _compute_support_audit(
        question,
        baseline_docs=baseline_docs,
        candidate_doc="",
        query_entities=query_entities,
        ce_reranker=ce_reranker,
    )

    unique_candidate_docs: Dict[str, str] = {}
    for row in action_rows:
        candidate_doc = _extract_candidate_doc_text(row)
        if candidate_doc:
            unique_candidate_docs[str(row.get("candidate_pool_position"))] = candidate_doc
    local_docs = [str(doc) for doc in baseline_docs]
    candidate_doc_pos: Dict[str, int] = {}
    for candidate_key, candidate_doc in unique_candidate_docs.items():
        candidate_doc_pos[candidate_key] = len(local_docs)
        local_docs.append(str(candidate_doc))

    doc_titles = {
        idx: str(doc.split("\n", 1)[0]).strip()
        for idx, doc in enumerate(local_docs)
    }
    query_graphs = {
        "flat": build_query_dependency_graph(question, query_entities=query_entities, action_mode="action_swap_noisyor_flat"),
        "dep": build_query_dependency_graph(question, query_entities=query_entities, action_mode="action_swap_noisyor_dep"),
    }

    windows_by_doc: Dict[int, List[str]] = {
        idx: build_title_prefixed_windows(doc_text)
        for idx, doc_text in enumerate(local_docs)
    }
    support_by_graph: Dict[str, Dict[str, Dict[int, float]]] = {}
    top_window_by_graph: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for graph_name, query_graph in query_graphs.items():
        pairs: List[List[str]] = []
        pair_index: List[Tuple[str, int, str]] = []
        graph_support: Dict[str, Dict[int, float]] = {}
        graph_top_windows: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for node in list(query_graph.get("nodes") or []):
            facet_id = str(node.get("id") or "")
            facet_text = str(node.get("facet") or "")
            graph_support[facet_id] = {}
            graph_top_windows[facet_id] = {}
            for doc_pos, windows in windows_by_doc.items():
                graph_support[facet_id][int(doc_pos)] = 0.0
                graph_top_windows[facet_id][int(doc_pos)] = {"score": 0.0, "window": ""}
                for window_text in windows:
                    pairs.append([facet_text, window_text])
                    pair_index.append((facet_id, int(doc_pos), window_text))
        raw_scores = ce_reranker.compute_score(pairs) if pairs else []
        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]
        for (facet_id, doc_pos, window_text), raw_score in zip(pair_index, list(raw_scores)):
            score = _sigmoid(float(raw_score))
            if score > float(graph_support[facet_id].get(int(doc_pos), 0.0)):
                graph_support[facet_id][int(doc_pos)] = float(score)
                graph_top_windows[facet_id][int(doc_pos)] = {
                    "score": float(score),
                    "window": str(window_text),
                }
        support_by_graph[graph_name] = graph_support
        top_window_by_graph[graph_name] = graph_top_windows

    before_positions = list(range(len(baseline_docs)))
    audit_rows: List[Dict[str, Any]] = []
    rankings: Dict[str, Dict[str, List[Tuple[Tuple[str, int | None, int | None], float, bool]]]] = {
        "flat": defaultdict(list),
        "dep": defaultdict(list),
    }

    for row in action_rows:
        replace_index = int(row.get("replace_incumbent_index"))
        candidate_position = candidate_doc_pos[str(row.get("candidate_pool_position"))]
        after_positions = list(before_positions)
        after_positions[replace_index] = int(candidate_position)

        candidate_best_flat = _best_doc_window(top_window_by_graph["flat"], int(candidate_position))
        incumbent_best_flat = _best_doc_window(top_window_by_graph["flat"], int(replace_index))
        candidate_best_dep = _best_doc_window(top_window_by_graph["dep"], int(candidate_position))
        incumbent_best_dep = _best_doc_window(top_window_by_graph["dep"], int(replace_index))

        utility_bundle: Dict[str, Dict[str, Any]] = {}
        for graph_name, dep_enabled in (("flat", False), ("dep", True)):
            utility_bundle[graph_name] = {}
            for variant in ("max", "full", "top2"):
                before_utility, before_facets = _compute_utility(
                    query_graphs[graph_name],
                    support_by_graph[graph_name],
                    before_positions,
                    dep_enabled=dep_enabled,
                    variant=variant,
                    doc_titles=doc_titles,
                )
                after_utility, after_facets = _compute_utility(
                    query_graphs[graph_name],
                    support_by_graph[graph_name],
                    after_positions,
                    dep_enabled=dep_enabled,
                    variant=variant,
                    doc_titles=doc_titles,
                )
                delta = float(after_utility - before_utility)
                utility_bundle[graph_name][variant] = {
                    "before": float(before_utility),
                    "after": float(after_utility),
                    "delta": float(delta),
                    "before_facets": before_facets,
                    "after_facets": after_facets,
                }
                rankings[graph_name][variant].append((_action_key(row), float(delta), bool(row.get("oracle_positive"))))

        audit_rows.append({
            "dataset": row.get("dataset"),
            "question": question,
            "action_type": "swap",
            "query_type": row.get("query_type"),
            "oracle_label": "positive" if bool(row.get("oracle_positive")) else "negative",
            "selected_by_dryrun": bool(row.get("selected_by_dryrun")),
            "selected_by_judge": bool(row.get("selected_by_judge")),
            "dep_triggered": bool(query_graphs["dep"].get("mode") == "dependency"),
            "query_dependency_mode": str(query_graphs["dep"].get("mode")),
            "candidate_pool_position": row.get("candidate_pool_position"),
            "replace_pool_position": row.get("replace_pool_position"),
            "candidate_title": row.get("candidate_title"),
            "replace_incumbent_title": row.get("replace_incumbent_title"),
            "score_delta": _round(_safe_float(row.get("score_delta"))),
            "oracle_delta_em": _round(_safe_float((row.get("delta_metrics") or {}).get("ExactMatch"))),
            "oracle_delta_f1": _round(_safe_float((row.get("delta_metrics") or {}).get("F1"))),
            "candidate_top_psi_flat": _round(candidate_best_flat.get("score")),
            "candidate_top_psi_dep": _round(candidate_best_dep.get("score")),
            "incumbent_top_psi_flat": _round(incumbent_best_flat.get("score")),
            "incumbent_top_psi_dep": _round(incumbent_best_dep.get("score")),
            "candidate_best_window_flat": candidate_best_flat.get("window"),
            "candidate_best_window_dep": candidate_best_dep.get("window"),
            "incumbent_best_window_flat": incumbent_best_flat.get("window"),
            "incumbent_best_window_dep": incumbent_best_dep.get("window"),
            "G_flat_before_top2": _round(utility_bundle["flat"]["top2"]["before"]),
            "G_flat_after_top2": _round(utility_bundle["flat"]["top2"]["after"]),
            "delta_flat_top2": _round(utility_bundle["flat"]["top2"]["delta"]),
            "G_dep_before_top2": _round(utility_bundle["dep"]["top2"]["before"]),
            "G_dep_after_top2": _round(utility_bundle["dep"]["top2"]["after"]),
            "delta_dep_top2": _round(utility_bundle["dep"]["top2"]["delta"]),
            "delta_flat_max": _round(utility_bundle["flat"]["max"]["delta"]),
            "delta_flat_full": _round(utility_bundle["flat"]["full"]["delta"]),
            "delta_dep_max": _round(utility_bundle["dep"]["max"]["delta"]),
            "delta_dep_full": _round(utility_bundle["dep"]["full"]["delta"]),
            "facet_supports_flat_before": utility_bundle["flat"]["top2"]["before_facets"],
            "facet_supports_flat_after": utility_bundle["flat"]["top2"]["after_facets"],
            "facet_supports_dep_before": utility_bundle["dep"]["top2"]["before_facets"],
            "facet_supports_dep_after": utility_bundle["dep"]["top2"]["after_facets"],
            "margin": float(margin),
        })
    query_summary = {
        "question": question,
        "dep_triggered": bool(query_graphs["dep"].get("mode") == "dependency"),
        "ranking_flat_top2": sorted(rankings["flat"]["top2"], key=lambda item: item[1], reverse=True),
        "ranking_dep_top2": sorted(rankings["dep"]["top2"], key=lambda item: item[1], reverse=True),
        "ranking_flat_max": sorted(rankings["flat"]["max"], key=lambda item: item[1], reverse=True),
        "ranking_flat_full": sorted(rankings["flat"]["full"], key=lambda item: item[1], reverse=True),
        "ranking_dep_max": sorted(rankings["dep"]["max"], key=lambda item: item[1], reverse=True),
        "ranking_dep_full": sorted(rankings["dep"]["full"], key=lambda item: item[1], reverse=True),
    }
    return audit_rows, query_summary


def _topk_positive_recall(query_summaries: Sequence[Mapping[str, Any]],
                          ranking_key: str,
                          *,
                          k: int) -> Tuple[int, int]:
    hit_count = 0
    total_count = 0
    for summary in query_summaries:
        ranking = list(summary.get(ranking_key) or [])
        if not ranking:
            continue
        total_count += 1
        top_rows = ranking[:max(int(k), 0)]
        if any(bool(item[2]) for item in top_rows):
            hit_count += 1
    return hit_count, total_count


def _build_summary(dataset: str,
                   audit_rows: Sequence[Mapping[str, Any]],
                   query_summaries: Sequence[Mapping[str, Any]],
                   *,
                   margin: float) -> Dict[str, Any]:
    oracle_positive_rows = [row for row in audit_rows if row.get("oracle_label") == "positive"]
    oracle_negative_rows = [row for row in audit_rows if row.get("oracle_label") == "negative"]
    dryrun_rows = [row for row in audit_rows if bool(row.get("selected_by_dryrun"))]
    judge_rows = [row for row in audit_rows if bool(row.get("selected_by_judge"))]

    def _delta_stats(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return {
            "count": len(values),
            "mean": _mean(values),
            "median": _median(values),
            "nonpositive_count": int(sum(value <= 0.0 for value in values)),
            "nonpositive_rate": _safe_pct(sum(value <= 0.0 for value in values), len(values)),
            "below_margin_count": int(sum(value <= float(margin) for value in values)),
            "below_margin_rate": _safe_pct(sum(value <= float(margin) for value in values), len(values)),
        }

    dep_queries = [row for row in query_summaries if bool(row.get("dep_triggered"))]
    ranking_changed = 0
    top1_changed = 0
    for row in dep_queries:
        flat_keys = [item[0] for item in list(row.get("ranking_flat_top2") or [])]
        dep_keys = [item[0] for item in list(row.get("ranking_dep_top2") or [])]
        if flat_keys != dep_keys:
            ranking_changed += 1
        if flat_keys[:1] != dep_keys[:1]:
            top1_changed += 1

    recall_rows = {}
    for ranking_key in ("ranking_flat_max", "ranking_flat_full", "ranking_flat_top2", "ranking_dep_top2"):
        hit_top1, total = _topk_positive_recall(query_summaries, ranking_key, k=1)
        hit_top3, _ = _topk_positive_recall(query_summaries, ranking_key, k=3)
        recall_rows[ranking_key] = {
            "top1_hit_count": hit_top1,
            "top1_hit_rate": _safe_pct(hit_top1, total),
            "top3_hit_count": hit_top3,
            "top3_hit_rate": _safe_pct(hit_top3, total),
            "query_count": total,
        }

    return {
        "dataset": dataset,
        "margin": float(margin),
        "selected_action_counts": {
            "total": int(len(audit_rows)),
            "oracle_positive": int(len(oracle_positive_rows)),
            "oracle_negative": int(len(oracle_negative_rows)),
            "dryrun_selected": int(len(dryrun_rows)),
            "judge_selected": int(len(judge_rows)),
        },
        "oracle_positive_delta_stats": {
            "flat_top2": _delta_stats(oracle_positive_rows, "delta_flat_top2"),
            "dep_top2": _delta_stats(oracle_positive_rows, "delta_dep_top2"),
            "flat_max": _delta_stats(oracle_positive_rows, "delta_flat_max"),
            "flat_full": _delta_stats(oracle_positive_rows, "delta_flat_full"),
        },
        "oracle_negative_delta_stats": {
            "flat_top2": _delta_stats(oracle_negative_rows, "delta_flat_top2"),
            "dep_top2": _delta_stats(oracle_negative_rows, "delta_dep_top2"),
        },
        "policy_selected_delta_stats": {
            "dryrun_flat_top2": _delta_stats(dryrun_rows, "delta_flat_top2"),
            "judge_flat_top2": _delta_stats(judge_rows, "delta_flat_top2"),
        },
        "ranking_recall": recall_rows,
        "dep_vs_flat_ranking": {
            "dep_triggered_queries": int(len(dep_queries)),
            "ranking_changed_count": int(ranking_changed),
            "ranking_changed_rate": _safe_pct(ranking_changed, len(dep_queries)),
            "top1_changed_count": int(top1_changed),
            "top1_changed_rate": _safe_pct(top1_changed, len(dep_queries)),
        },
    }


def _build_markdown(summary: Mapping[str, Any],
                    audit_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"# Action-Level Audit: {summary.get('dataset')}",
        "",
        f"- action margin: `{summary.get('margin')}`",
        f"- selected actions: `{(summary.get('selected_action_counts') or {}).get('total')}`",
        f"- oracle-positive selected swaps: `{(summary.get('selected_action_counts') or {}).get('oracle_positive')}`",
        f"- oracle-negative sampled swaps: `{(summary.get('selected_action_counts') or {}).get('oracle_negative')}`",
        f"- dryrun executed swaps in audit: `{(summary.get('selected_action_counts') or {}).get('dryrun_selected')}`",
        f"- judge executed swaps in audit: `{(summary.get('selected_action_counts') or {}).get('judge_selected')}`",
        "",
        "## Oracle-Positive Delta Audit",
        "",
        "| Variant | Count | Median Δ | Nonpositive | <= margin |",
        "|---|---:|---:|---:|---:|",
    ]
    positive_stats = dict(summary.get("oracle_positive_delta_stats") or {})
    for label in ("flat_top2", "dep_top2", "flat_max", "flat_full"):
        stats = dict(positive_stats.get(label) or {})
        lines.append(
            f"| {label} | {stats.get('count', 0)} | {stats.get('median', '—')} | "
            f"{stats.get('nonpositive_count', 0)} ({stats.get('nonpositive_rate', '—')}) | "
            f"{stats.get('below_margin_count', 0)} ({stats.get('below_margin_rate', '—')}) |"
        )

    lines.extend([
        "",
        "## Ranking Recall",
        "",
        "| Ranking | Top-1 positive hit rate | Top-3 positive hit rate | Queries |",
        "|---|---:|---:|---:|",
    ])
    for label, stats in dict(summary.get("ranking_recall") or {}).items():
        lines.append(
            f"| {label} | {stats.get('top1_hit_rate', '—')} | {stats.get('top3_hit_rate', '—')} | {stats.get('query_count', 0)} |"
        )

    dep_stats = dict(summary.get("dep_vs_flat_ranking") or {})
    lines.extend([
        "",
        "## Dep vs Flat",
        "",
        f"- dep-triggered queries: `{dep_stats.get('dep_triggered_queries', 0)}`",
        f"- flat/dep ranking changed: `{dep_stats.get('ranking_changed_count', 0)}` (`{dep_stats.get('ranking_changed_rate', '—')}`)",
        f"- flat/dep top-1 action changed: `{dep_stats.get('top1_changed_count', 0)}` (`{dep_stats.get('top1_changed_rate', '—')}`)",
        "",
        "## Sample Failure Cases",
        "",
    ])
    failure_rows = [
        row for row in audit_rows
        if row.get("oracle_label") == "positive"
        and (
            (row.get("delta_flat_top2") is not None and float(row.get("delta_flat_top2")) <= 0.0)
            or (row.get("delta_dep_top2") is not None and float(row.get("delta_dep_top2")) <= 0.0)
        )
    ][:10]
    if not failure_rows:
        lines.append("- none")
    else:
        for row in failure_rows:
            lines.extend([
                f"- question: `{row.get('question', '')}`",
                f"  - action: swap in `{row.get('candidate_title')}` for `{row.get('replace_incumbent_title')}`",
                f"  - oracle ΔEM / ΔF1: `{row.get('oracle_delta_em')}` / `{row.get('oracle_delta_f1')}`",
                f"  - flat Δ / dep Δ: `{row.get('delta_flat_top2')}` / `{row.get('delta_dep_top2')}`",
                f"  - candidate ψ(flat/dep): `{row.get('candidate_top_psi_flat')}` / `{row.get('candidate_top_psi_dep')}`",
                f"  - incumbent ψ(flat/dep): `{row.get('incumbent_top_psi_flat')}` / `{row.get('incumbent_top_psi_dep')}`",
            ])
    return "\n".join(lines) + "\n"


def run_audit(dataset: str,
              *,
              oracle_relaxed_report: str,
              query_report: str,
              dryrun_report: str,
              judge_report: str,
              output_json: str,
              output_md: str,
              output_csv: str,
              ce_model: str,
              ce_device: str,
              negative_sample_size: int,
              margin: float,
              sample_seed: int) -> Dict[str, Any]:
    oracle_payload = _load_json(oracle_relaxed_report)
    query_payload = _load_json(query_report)
    dryrun_payload = _load_json(dryrun_report)
    judge_payload = _load_json(judge_report)

    dryrun_actions = _extract_policy_actions(dryrun_payload)
    judge_actions = _extract_policy_actions(judge_payload)
    query_contexts = _query_context_map(query_payload)

    selected_rows = _select_action_rows(
        oracle_payload,
        dryrun_actions=dryrun_actions,
        judge_actions=judge_actions,
        negative_sample_size=negative_sample_size,
        seed=sample_seed,
    )
    selected_keys = {
        (_normalize_question(row.get("question", "")), _action_key(row))
        for row in selected_rows
    }
    all_legal_rows: List[Dict[str, Any]] = []
    for row in list(oracle_payload.get("action_results") or []):
        if str(row.get("action_type") or "").strip().lower() != "swap" or not bool(row.get("is_legal")):
            continue
        materialized = dict(row)
        delta_metrics = dict(materialized.get("delta_metrics") or {})
        materialized["oracle_positive"] = bool(
            float(delta_metrics.get("F1", 0.0) or 0.0) > 0.0
            or float(delta_metrics.get("ExactMatch", 0.0) or 0.0) > 0.0
        )
        question_key = _normalize_question(materialized.get("question", ""))
        materialized["selected_by_dryrun"] = _policy_selected(question_key, dryrun_actions, materialized)
        materialized["selected_by_judge"] = _policy_selected(question_key, judge_actions, materialized)
        all_legal_rows.append(materialized)

    by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_legal_rows:
        by_question[_normalize_question(row.get("question", ""))].append(dict(row))

    ce_reranker = _load_ce_reranker(ce_model, ce_device)

    audit_rows: List[Dict[str, Any]] = []
    query_summaries: List[Dict[str, Any]] = []
    for question_key, rows in by_question.items():
        query_context = dict(query_contexts.get(question_key) or {})
        query_audit_rows, query_summary = _compute_query_audit_rows(
            rows,
            query_context=query_context,
            ce_reranker=ce_reranker,
            margin=margin,
        )
        audit_rows.extend([
            row
            for row in query_audit_rows
            if (question_key, _action_key(row)) in selected_keys
        ])
        if query_summary:
            query_summaries.append(query_summary)

    summary = _build_summary(dataset, audit_rows, query_summaries, margin=margin)
    payload = {
        "summary": summary,
        "audit_rows": audit_rows,
        "query_summaries": query_summaries,
        "inputs": {
            "dataset": dataset,
            "oracle_relaxed_report": str(oracle_relaxed_report),
            "query_report": str(query_report),
            "dryrun_report": str(dryrun_report),
            "judge_report": str(judge_report),
            "ce_model": str(ce_model),
            "ce_device": str(ce_device),
            "negative_sample_size": int(negative_sample_size),
            "margin": float(margin),
        },
    }
    Path(output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(output_md).write_text(_build_markdown(summary, audit_rows), encoding="utf-8")
    with Path(output_csv).open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "question",
            "query_type",
            "oracle_label",
            "selected_by_dryrun",
            "selected_by_judge",
            "dep_triggered",
            "query_dependency_mode",
            "candidate_pool_position",
            "replace_pool_position",
            "candidate_title",
            "replace_incumbent_title",
            "score_delta",
            "oracle_delta_em",
            "oracle_delta_f1",
            "candidate_top_psi_flat",
            "candidate_top_psi_dep",
            "incumbent_top_psi_flat",
            "incumbent_top_psi_dep",
            "G_flat_before_top2",
            "G_flat_after_top2",
            "delta_flat_top2",
            "G_dep_before_top2",
            "G_dep_after_top2",
            "delta_dep_top2",
            "delta_flat_max",
            "delta_flat_full",
            "delta_dep_max",
            "delta_dep_full",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local action-level audit for the noisy-or action controller.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--oracle_relaxed_report", required=True)
    parser.add_argument("--query_report", required=True)
    parser.add_argument("--dryrun_report", required=True)
    parser.add_argument("--judge_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--ce_model", default=DEFAULT_CE_MODEL)
    parser.add_argument("--ce_device", default="cuda:0")
    parser.add_argument("--negative_sample_size", type=int, default=40)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--sample_seed", type=int, default=7)
    args = parser.parse_args()

    run_audit(
        dataset=str(args.dataset),
        oracle_relaxed_report=str(args.oracle_relaxed_report),
        query_report=str(args.query_report),
        dryrun_report=str(args.dryrun_report),
        judge_report=str(args.judge_report),
        output_json=str(args.output_json),
        output_md=str(args.output_md),
        output_csv=str(args.output_csv),
        ce_model=str(args.ce_model),
        ce_device=str(args.ce_device),
        negative_sample_size=int(args.negative_sample_size),
        margin=float(args.margin),
        sample_seed=int(args.sample_seed),
    )


if __name__ == "__main__":
    main()
