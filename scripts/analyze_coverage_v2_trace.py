#!/usr/bin/env python3
"""Analyze frozen-atoms v2 coverage traces against v1 and append0 baselines.

This script is intentionally narrow. It answers:

1. Did v2 improve because it still selects appended docs with real frozen-edge gain?
2. Or did it improve mostly by suppressing appended docs?
3. Does the current MuSiQue evidence support continuing toward a v3 atom-source
   relaxation, or should the next fix move to the Expand/proposal interface?
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_coverage_atoms import (  # noqa: E402
    build_chunk_id_to_doc_text,
    build_pool_mapping_from_report,
    build_runtime_args,
    load_corpus,
    normalize_candidate_positions,
    resolve_runtime_save_dir,
)
from eval_causal_qwen3 import (  # noqa: E402
    build_config,
    collect_lexical_query_seed_entities,
    collect_query_seed_entities,
    extract_doc_title,
    normalize_entity_set,
    normalize_structure_text,
)
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402


def _round(value: Any, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _safe_mean(values: Iterable[float]) -> float:
    numeric = [float(v) for v in values]
    if not numeric:
        return 0.0
    return round(float(sum(numeric) / len(numeric)), 4)


def _safe_median(values: Iterable[float]) -> float:
    numeric = [float(v) for v in values]
    if not numeric:
        return 0.0
    return round(float(statistics.median(numeric)), 4)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_edge(edge: Tuple[str, str, float, str] | Tuple[str, str]) -> Tuple[str, str] | None:
    src = normalize_structure_text(edge[0])
    tgt = normalize_structure_text(edge[1])
    if not src or not tgt:
        return None
    return src, tgt


def build_doc_cover_maps_for_atom_source(
    candidate_positions: Sequence[int],
    atom_source_positions: Sequence[int],
    pool_doc_ids: Sequence[int | None],
    doc_idx_to_entities: Mapping[int, Set[str]],
    doc_idx_to_edges: Mapping[int, List[Tuple[str, str, float, str]]],
    seed_entities: Set[str],
    atom_source_mode: str = "baseline_prefix",
) -> Tuple[Set[str], Set[Tuple[str, str]], Set[str], Dict[int, Set[str]], Dict[int, Set[Tuple[str, str]]], Dict[int, Set[str]]]:
    normalized_seed_entities = normalize_entity_set(seed_entities)
    a_q: Set[str] = set(normalized_seed_entities)
    a_e: Set[Tuple[str, str]] = set()
    a_b: Set[str] = set()
    base_entity_set: Set[str] = set()
    doc_entities_by_position: Dict[int, Set[str]] = {}
    doc_edges_by_position: Dict[int, Set[Tuple[str, str]]] = {}

    doc_covers_q: Dict[int, Set[str]] = {}
    doc_covers_e: Dict[int, Set[Tuple[str, str]]] = {}
    doc_covers_b: Dict[int, Set[str]] = {}
    for pos in candidate_positions:
        doc_id = pool_doc_ids[pos]
        if doc_id is None:
            doc_entities_by_position[pos] = set()
            doc_edges_by_position[pos] = set()
            doc_covers_q[pos] = set()
            doc_covers_e[pos] = set()
            doc_covers_b[pos] = set()
            continue
        normalized_entities = normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
        normalized_edges = {
            normalized_edge
            for normalized_edge in (
                normalize_edge(edge)
                for edge in doc_idx_to_edges.get(int(doc_id), [])
            )
            if normalized_edge is not None
        }
        doc_entities_by_position[pos] = normalized_entities
        doc_edges_by_position[pos] = normalized_edges

    for pos in atom_source_positions:
        base_entity_set.update(doc_entities_by_position.get(int(pos), set()))

    if str(atom_source_mode).strip().lower() == "baseline_anchored":
        for pos in candidate_positions:
            anchored_edges = {
                edge for edge in doc_edges_by_position.get(int(pos), set())
                if edge[0] in base_entity_set or edge[1] in base_entity_set
            }
            a_e.update(anchored_edges)
            for src, tgt in anchored_edges:
                a_b.add(src)
                a_b.add(tgt)
    else:
        for pos in atom_source_positions:
            a_b.update(doc_entities_by_position.get(int(pos), set()))
            a_e.update(doc_edges_by_position.get(int(pos), set()))
    a_b -= a_q

    for pos in candidate_positions:
        normalized_entities = doc_entities_by_position.get(int(pos), set())
        normalized_edges = doc_edges_by_position.get(int(pos), set())
        doc_covers_q[pos] = normalized_entities & a_q
        doc_covers_e[pos] = normalized_edges & a_e
        if str(atom_source_mode).strip().lower() == "baseline_anchored":
            anchored_entities = {
                entity
                for src, tgt in doc_covers_e[pos]
                for entity in (src, tgt)
            }
            doc_covers_b[pos] = (anchored_entities - a_q) & a_b
        else:
            doc_covers_b[pos] = (normalized_entities - a_q) & a_b

    return a_q, a_e, a_b, doc_covers_q, doc_covers_e, doc_covers_b


def compute_edge_support_counts(
    candidate_positions: Sequence[int],
    doc_covers_e: Mapping[int, Set[Tuple[str, str]]],
) -> Counter[Tuple[str, str]]:
    support_counts: Counter[Tuple[str, str]] = Counter()
    for pos in candidate_positions:
        for edge in doc_covers_e.get(int(pos), set()):
            support_counts[edge] += 1
    return support_counts


def compute_selected_appended_doc_rows(
    selected_positions: Sequence[int],
    appended_positions: Sequence[int],
    pool_docs: Sequence[str],
    doc_covers_e: Mapping[int, Set[Tuple[str, str]]],
    edge_support_counts: Mapping[Tuple[str, str], int],
) -> List[Dict[str, Any]]:
    selected_set = {int(pos) for pos in selected_positions}
    appended_selected = [int(pos) for pos in selected_positions if int(pos) in set(int(p) for p in appended_positions)]
    rows: List[Dict[str, Any]] = []
    for pos in appended_selected:
        other_edges: Set[Tuple[str, str]] = set()
        for other_pos in selected_set:
            if int(other_pos) == pos:
                continue
            other_edges.update(doc_covers_e.get(int(other_pos), set()))
        covered_edges = set(doc_covers_e.get(int(pos), set()))
        unique_edges = covered_edges - other_edges
        covered_supports = sorted(int(edge_support_counts.get(edge, 0)) for edge in covered_edges)
        unique_supports = sorted(int(edge_support_counts.get(edge, 0)) for edge in unique_edges)
        rows.append(
            {
                "pool_position": int(pos),
                "title": extract_doc_title(pool_docs[pos]),
                "covered_edges": sorted([list(edge) for edge in covered_edges]),
                "unique_edges": sorted([list(edge) for edge in unique_edges]),
                "covered_edge_count": int(len(covered_edges)),
                "unique_covE_gain": int(len(unique_edges)),
                "has_nonzero_unique_covE_gain": bool(unique_edges),
                "covered_edge_support_counts": covered_supports,
                "unique_edge_support_counts": unique_supports,
                "avg_covered_edge_support": _safe_mean(covered_supports),
                "avg_unique_edge_support": _safe_mean(unique_supports),
                "covered_frozen_edges": sorted([list(edge) for edge in covered_edges]),
                "unique_frozen_edges": sorted([list(edge) for edge in unique_edges]),
                "covered_frozen_edge_count": int(len(covered_edges)),
                "unique_frozen_covE_gain": int(len(unique_edges)),
                "has_nonzero_unique_frozen_covE_gain": bool(unique_edges),
                "covered_frozen_edge_support_counts": covered_supports,
                "unique_frozen_edge_support_counts": unique_supports,
                "avg_covered_frozen_edge_support": _safe_mean(covered_supports),
                "avg_unique_frozen_edge_support": _safe_mean(unique_supports),
            }
        )
    return rows


def compare_trace_metrics(
    left_traces: Sequence[Mapping[str, Any]],
    right_traces: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if len(left_traces) != len(right_traces):
        raise ValueError("Trace lengths do not match.")

    em_deltas: List[float] = []
    f1_deltas: List[float] = []
    em_gain = em_tie = em_loss = 0
    f1_gain = f1_tie = f1_loss = 0
    winners: List[Dict[str, Any]] = []
    losers: List[Dict[str, Any]] = []

    for left_trace, right_trace in zip(left_traces, right_traces):
        question = str(left_trace.get("question", ""))
        left_metrics = dict(left_trace.get("method_metrics", {}) or {})
        right_metrics = dict(right_trace.get("method_metrics", {}) or {})
        delta_em = float(left_metrics.get("ExactMatch", 0.0) or 0.0) - float(right_metrics.get("ExactMatch", 0.0) or 0.0)
        delta_f1 = float(left_metrics.get("F1", 0.0) or 0.0) - float(right_metrics.get("F1", 0.0) or 0.0)
        em_deltas.append(delta_em)
        f1_deltas.append(delta_f1)

        if delta_em > 1e-9:
            em_gain += 1
        elif delta_em < -1e-9:
            em_loss += 1
        else:
            em_tie += 1

        if delta_f1 > 1e-9:
            f1_gain += 1
        elif delta_f1 < -1e-9:
            f1_loss += 1
        else:
            f1_tie += 1

        row = {
            "question": question,
            "delta_em": round(delta_em, 4),
            "delta_f1": round(delta_f1, 4),
            "left_titles": list(left_trace.get("method_top_titles", []) or []),
            "right_titles": list(right_trace.get("method_top_titles", []) or []),
        }
        if delta_em > 1e-9 or delta_f1 > 1e-9:
            winners.append(row)
        elif delta_em < -1e-9 or delta_f1 < -1e-9:
            losers.append(row)

    winners.sort(key=lambda row: (-row["delta_f1"], -row["delta_em"], row["question"]))
    losers.sort(key=lambda row: (row["delta_f1"], row["delta_em"], row["question"]))

    return {
        "query_count": len(left_traces),
        "em": {"gain": em_gain, "tie": em_tie, "loss": em_loss, "avg_delta": _safe_mean(em_deltas)},
        "f1": {"gain": f1_gain, "tie": f1_tie, "loss": f1_loss, "avg_delta": _safe_mean(f1_deltas)},
        "top_winners": winners[:10],
        "top_losers": losers[:10],
    }


def classify_route(selected_appended_query_rate: float, nonzero_gain_doc_rate: float) -> Dict[str, Any]:
    query_rate_threshold = 0.20
    gain_rate_threshold = 0.50
    if selected_appended_query_rate >= query_rate_threshold and nonzero_gain_doc_rate >= gain_rate_threshold:
        return {
            "route": "A",
            "decision": "coverage line is worth continuing; try a narrow v3 atom-source relaxation next",
            "thresholds": {
                "selected_appended_query_rate_min": query_rate_threshold,
                "nonzero_gain_doc_rate_min": gain_rate_threshold,
            },
        }
    return {
        "route": "B",
        "decision": "v2 mainly looks like suppression or weak appended utility; prioritize Expand/interface diagnostics before v3",
        "thresholds": {
            "selected_appended_query_rate_min": query_rate_threshold,
            "nonzero_gain_doc_rate_min": gain_rate_threshold,
        },
    }


def summarize_decision_layer_distribution(query_rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in query_rows:
        label = str(row.get("effective_decision_layer", "") or "")
        if not label:
            continue
        counts[label] += 1
    return {key: int(counts[key]) for key in sorted(counts)}


def summarize_admissibility_metrics(query_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    query_count = len(query_rows)
    admissible_query_count = sum(1 for row in query_rows if int(row.get("admissible_appended_count", 0) or 0) > 0)
    admissible_doc_count = sum(int(row.get("admissible_appended_count", 0) or 0) for row in query_rows)
    selected_from_admissible_doc_count = sum(
        int(row.get("selected_appended_from_admissible_count", 0) or 0)
        for row in query_rows
    )
    selected_appended_doc_count = sum(
        int(row.get("num_appended_selected_v2", 0) or 0)
        for row in query_rows
    )

    kept_rows = [
        kept_row
        for row in query_rows
        for kept_row in list(row.get("admissibility_rows", []) or [])
        if bool(kept_row.get("kept", False))
    ]
    marginal_gap_edge_counts = [
        int(kept_row.get("marginal_gap_edge_count", 0) or 0)
        for kept_row in kept_rows
    ]
    marginal_gap_edge_support_means = [
        float(kept_row.get("marginal_gap_edge_support_mean", 0.0) or 0.0)
        for kept_row in kept_rows
    ]
    marginal_gap_edge_support_maxes = [
        float(kept_row.get("marginal_gap_edge_support_max", 0.0) or 0.0)
        for kept_row in kept_rows
    ]
    reference_gap_edge_counts = [
        int(row.get("admissibility_reference_gap_edge_count", 0) or 0)
        for row in query_rows
        if row.get("coverage_admissibility_mode", "off") != "off"
    ]

    return {
        "admissible_appended_query_rate_target": round(admissible_query_count / query_count, 4) if query_count else 0.0,
        "avg_num_admissible_appended_target": _safe_mean(
            row.get("admissible_appended_count", 0) for row in query_rows
        ),
        "selected_from_admissible_doc_rate_target": round(
            selected_from_admissible_doc_count / selected_appended_doc_count, 4
        ) if selected_appended_doc_count else 0.0,
        "avg_marginal_gap_edge_count_kept": _safe_mean(marginal_gap_edge_counts),
        "avg_marginal_gap_edge_support_mean_kept": _safe_mean(marginal_gap_edge_support_means),
        "avg_marginal_gap_edge_support_max_kept": _safe_mean(marginal_gap_edge_support_maxes),
        "avg_reference_gap_edge_count_target": _safe_mean(reference_gap_edge_counts),
    }


def align_question_rows(report: Mapping[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    traces = list(report.get("expand_assemble_query_traces", []) or [])
    examples = list(report.get("examples", []) or [])
    if len(traces) != len(examples):
        raise ValueError("Report trace/example lengths do not match.")
    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for trace_row, example_row in zip(traces, examples):
        question = str(trace_row.get("question", ""))
        if str(example_row.get("question", "")) != question:
            raise ValueError("Report examples and traces are misaligned by question.")
        aligned.append((dict(trace_row), dict(example_row)))
    return aligned


def build_report_lookup(report: Mapping[str, Any]) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
    return {
        str(trace_row.get("question", "")): (trace_row, example_row)
        for trace_row, example_row in align_question_rows(report)
    }


def summarize_v2_triplet(
    question: str,
    v1_trace: Mapping[str, Any],
    v2_trace: Mapping[str, Any],
    append0_trace: Mapping[str, Any],
    pool_docs: Sequence[str],
    pool_doc_ids: Sequence[int | None],
    seed_entities: Set[str],
    doc_idx_to_entities: Mapping[int, Set[str]],
    doc_idx_to_edges: Mapping[int, List[Tuple[str, str, float, str]]],
) -> Dict[str, Any]:
    v1_expand = dict(v1_trace.get("expand_assemble_trace", {}) or {})
    v2_expand = dict(v2_trace.get("expand_assemble_trace", {}) or {})
    append0_expand = dict(append0_trace.get("expand_assemble_trace", {}) or {})
    v1_assemble = dict(v1_expand.get("assemble_trace", {}) or {})
    v2_assemble = dict(v2_expand.get("assemble_trace", {}) or {})
    atom_source_mode = str(v2_assemble.get("atom_source_mode", v2_expand.get("coverage_atom_source", "candidate_pool")) or "candidate_pool")

    candidate_positions = normalize_candidate_positions(
        v2_expand.get("candidate_set_positions", []) or [],
        len(pool_docs),
    )
    atom_source_positions = normalize_candidate_positions(
        v2_assemble.get("atom_source_positions", []) or v2_expand.get("baseline_prefix_positions", []) or [],
        len(pool_docs),
    )
    appended_positions = normalize_candidate_positions(
        v2_expand.get("appended_positions", []) or [],
        len(pool_docs),
    )
    selected_positions_v1 = normalize_candidate_positions(
        v1_assemble_selected_positions(v1_trace),
        len(pool_docs),
    )
    selected_positions_v2 = normalize_candidate_positions(
        v2_assemble.get("selected_positions", []) or v2_expand.get("final_front_pool_positions", []) or [],
        len(pool_docs),
    )
    selected_positions_append0 = normalize_candidate_positions(
        append0_selected_positions(append0_trace),
        len(pool_docs),
    )

    _, frozen_edges, _, _, doc_covers_e, _ = build_doc_cover_maps_for_atom_source(
        candidate_positions=candidate_positions,
        atom_source_positions=atom_source_positions,
        pool_doc_ids=pool_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        seed_entities=seed_entities,
        atom_source_mode=atom_source_mode,
    )
    edge_support_counts = compute_edge_support_counts(candidate_positions, doc_covers_e)
    selected_appended_rows_v2 = compute_selected_appended_doc_rows(
        selected_positions=selected_positions_v2,
        appended_positions=appended_positions,
        pool_docs=pool_docs,
        doc_covers_e=doc_covers_e,
        edge_support_counts=edge_support_counts,
    )
    unique_supports_flat: List[int] = []
    covered_supports_flat: List[int] = []
    for row in selected_appended_rows_v2:
        unique_supports_flat.extend(int(v) for v in row["unique_frozen_edge_support_counts"])
        covered_supports_flat.extend(int(v) for v in row["covered_frozen_edge_support_counts"])

    v1_metrics = dict(v1_trace.get("method_metrics", {}) or {})
    v2_metrics = dict(v2_trace.get("method_metrics", {}) or {})
    append0_metrics = dict(append0_trace.get("method_metrics", {}) or {})
    v1_title_keys = {
        normalize_structure_text(title)
        for title in list(v1_trace.get("method_top_titles", []) or [])
        if normalize_structure_text(title)
    }
    v2_title_keys = {
        normalize_structure_text(title)
        for title in list(v2_trace.get("method_top_titles", []) or [])
        if normalize_structure_text(title)
    }
    append0_title_keys = {
        normalize_structure_text(title)
        for title in list(append0_trace.get("method_top_titles", []) or [])
        if normalize_structure_text(title)
    }
    num_appended_selected_v1 = int(v1_assemble.get("num_appended_selected", 0) or 0)
    admissibility_trace = dict(v2_assemble.get("admissibility_trace", {}) or {})
    admissibility_rows = [dict(row) for row in list(admissibility_trace.get("rows", []) or [])]
    admissible_appended_positions = normalize_candidate_positions(
        admissibility_trace.get("appended_positions_after_filter", []) or [],
        len(pool_docs),
    )
    selected_appended_positions_v2 = normalize_candidate_positions(
        [row.get("pool_position") for row in selected_appended_rows_v2],
        len(pool_docs),
    )
    admissible_appended_position_set = {int(pos) for pos in admissible_appended_positions}
    selected_from_admissible_count = sum(
        1 for pos in selected_appended_positions_v2
        if int(pos) in admissible_appended_position_set
    )

    return {
        "question": question,
        "target_atom_source_mode": atom_source_mode,
        "coverage_admissibility_mode": str(
            v2_assemble.get("coverage_admissibility_mode", v2_expand.get("coverage_admissibility_mode", "off")) or "off"
        ),
        "candidate_count": int(len(candidate_positions)),
        "frozen_atom_edge_count": int(len(frozen_edges)),
        "selected_positions_v1": selected_positions_v1,
        "selected_positions_v2": selected_positions_v2,
        "selected_positions_append0": selected_positions_append0,
        "selected_titles_v2": [extract_doc_title(pool_docs[pos]) for pos in selected_positions_v2],
        "selected_appended_titles_v2": [row["title"] for row in selected_appended_rows_v2],
        "selected_titles_v1": list(v1_trace.get("method_top_titles", []) or []),
        "selected_titles_append0": list(append0_trace.get("method_top_titles", []) or []),
        "num_appended_selected_v1": num_appended_selected_v1,
        "num_appended_selected_v2": int(len(selected_appended_rows_v2)),
        "selected_appended_rows_v2": selected_appended_rows_v2,
        "selected_appended_rows_target": selected_appended_rows_v2,
        "admissibility_rows": admissibility_rows,
        "admissible_appended_count": int(len(admissible_appended_positions)),
        "admissible_appended_positions": list(admissible_appended_positions),
        "admissibility_reference_positions": normalize_candidate_positions(
            admissibility_trace.get("reference_positions", []) or [],
            len(pool_docs),
        ),
        "admissibility_reference_gap_edge_count": int(admissibility_trace.get("reference_gap_edge_count", 0) or 0),
        "selected_appended_from_admissible_count": int(selected_from_admissible_count),
        "effective_decision_layer": str(v2_assemble.get("effective_decision_layer", "")),
        "covQ_margin": _round(v2_assemble.get("covQ_margin")),
        "covE_margin_within_max_covQ": _round(v2_assemble.get("covE_margin_within_max_covQ")),
        "ce_margin_within_max_covQ_covE": _round(v2_assemble.get("ce_margin_within_max_covQ_covE")),
        "selected_appended_with_nonzero_unique_gain": int(
            sum(1 for row in selected_appended_rows_v2 if row["has_nonzero_unique_covE_gain"])
        ),
        "selected_appended_with_zero_unique_gain": int(
            sum(1 for row in selected_appended_rows_v2 if not row["has_nonzero_unique_covE_gain"])
        ),
        "selected_appended_with_nonzero_frozen_covE_gain": int(
            sum(1 for row in selected_appended_rows_v2 if row["has_nonzero_unique_frozen_covE_gain"])
        ),
        "selected_appended_with_zero_frozen_covE_gain": int(
            sum(1 for row in selected_appended_rows_v2 if not row["has_nonzero_unique_frozen_covE_gain"])
        ),
        "coverage_vs_append0_overlap": int(len(v2_title_keys & append0_title_keys)),
        "coverage_vs_v1_overlap": int(len(v2_title_keys & v1_title_keys)),
        "replaced_baseline_incumbents_count": int(len(append0_title_keys - v2_title_keys)),
        "new_edge_support_count": {
            "covered_edge_count": int(len(covered_supports_flat)),
            "unique_edge_count": int(len(unique_supports_flat)),
            "avg_covered_edge_support": _safe_mean(covered_supports_flat),
            "median_covered_edge_support": _safe_median(covered_supports_flat),
            "avg_unique_edge_support": _safe_mean(unique_supports_flat),
            "median_unique_edge_support": _safe_median(unique_supports_flat),
        },
        "v2_minus_v1": {
            "em": round(float(v2_metrics.get("ExactMatch", 0.0) or 0.0) - float(v1_metrics.get("ExactMatch", 0.0) or 0.0), 4),
            "f1": round(float(v2_metrics.get("F1", 0.0) or 0.0) - float(v1_metrics.get("F1", 0.0) or 0.0), 4),
        },
        "v2_minus_append0": {
            "em": round(float(v2_metrics.get("ExactMatch", 0.0) or 0.0) - float(append0_metrics.get("ExactMatch", 0.0) or 0.0), 4),
            "f1": round(float(v2_metrics.get("F1", 0.0) or 0.0) - float(append0_metrics.get("F1", 0.0) or 0.0), 4),
        },
    }


def v1_assemble_selected_positions(trace: Mapping[str, Any]) -> Sequence[int]:
    expand_trace = dict(trace.get("expand_assemble_trace", {}) or {})
    assemble_trace = dict(expand_trace.get("assemble_trace", {}) or {})
    return assemble_trace.get("selected_positions", []) or expand_trace.get("final_front_pool_positions", []) or []


def append0_selected_positions(trace: Mapping[str, Any]) -> Sequence[int]:
    return v1_assemble_selected_positions(trace)


def build_markdown_summary(payload: Mapping[str, Any]) -> str:
    aggregate = dict(payload.get("aggregate", {}) or {})
    route = dict(payload.get("route_decision", {}) or {})
    target_label = str(payload.get("target_label", "target"))
    paired = dict(payload.get("paired", {}) or {})
    target_vs_v1 = dict(paired.get("target_vs_v1", {}) or {})
    target_vs_append0 = dict(paired.get("target_vs_append0", {}) or {})
    lines = [
        "# Coverage v2 Trace Diagnostic",
        "",
        "## Aggregate",
        "",
        f"- Query count: {aggregate.get('query_count', 0)}",
        f"- v1 selected appended query rate: {aggregate.get('selected_appended_query_rate_v1', 0.0)}",
        f"- {target_label} selected appended query rate: {aggregate.get('selected_appended_query_rate_target', 0.0)}",
        f"- {target_label} avg appended selected: {aggregate.get('avg_num_appended_selected_target', 0.0)}",
        f"- {target_label} nonzero unique CovE gain doc rate: {aggregate.get('selected_appended_nonzero_gain_doc_rate_target', 0.0)}",
        f"- {target_label} admissible appended query rate: {aggregate.get('admissible_appended_query_rate_target', 0.0)}",
        f"- {target_label} avg admissible appended: {aggregate.get('avg_num_admissible_appended_target', 0.0)}",
        f"- {target_label} selected-from-admissible doc rate: {aggregate.get('selected_from_admissible_doc_rate_target', 0.0)}",
        f"- {target_label} avg kept marginal gap edges: {aggregate.get('avg_marginal_gap_edge_count_kept', 0.0)}",
        f"- {target_label} effective decision layer distribution: {aggregate.get('effective_decision_layer_distribution_target', {})}",
        f"- Avg overlap with append0: {aggregate.get('avg_coverage_vs_append0_overlap', 0.0)}",
        f"- Avg replaced append0 incumbents: {aggregate.get('avg_replaced_baseline_incumbents_count', 0.0)}",
        f"- Route: {route.get('route', '')} ({route.get('decision', '')})",
        "",
        "## Paired Comparisons",
        "",
        f"- {target_label} vs v1 EM gain/tie/loss: {target_vs_v1.get('em', {}).get('gain', 0)}/{target_vs_v1.get('em', {}).get('tie', 0)}/{target_vs_v1.get('em', {}).get('loss', 0)}",
        f"- {target_label} vs v1 F1 gain/tie/loss: {target_vs_v1.get('f1', {}).get('gain', 0)}/{target_vs_v1.get('f1', {}).get('tie', 0)}/{target_vs_v1.get('f1', {}).get('loss', 0)}",
        f"- {target_label} vs append0 EM gain/tie/loss: {target_vs_append0.get('em', {}).get('gain', 0)}/{target_vs_append0.get('em', {}).get('tie', 0)}/{target_vs_append0.get('em', {}).get('loss', 0)}",
        f"- {target_label} vs append0 F1 gain/tie/loss: {target_vs_append0.get('f1', {}).get('gain', 0)}/{target_vs_append0.get('f1', {}).get('tie', 0)}/{target_vs_append0.get('f1', {}).get('loss', 0)}",
        "",
        "## Top Route-Informative Cases",
        "",
    ]
    for row in list(payload.get("query_rows", []) or [])[:10]:
        lines.append(
            f"- {row['question']} | appended_target={row['num_appended_selected_v2']} | "
            f"nonzero_gain={row['selected_appended_with_nonzero_unique_gain']} | "
            f"decision={row['effective_decision_layer'] or 'NA'} | "
            f"target-v1 EM/F1={row['v2_minus_v1']['em']}/{row['v2_minus_v1']['f1']} | "
            f"target-append0 EM/F1={row['v2_minus_append0']['em']}/{row['v2_minus_append0']['f1']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze coverage frozen-atoms v2 traces.")
    parser.add_argument("--dataset", choices=["musique", "hotpotqa", "2wikimultihopqa"], required=True)
    parser.add_argument("--qa_top_k", type=int, required=True)
    parser.add_argument("--report_v1", type=str, required=True, help="append3 + coverage + candidate_pool report")
    parser.add_argument("--report_target", "--report_v2", dest="report_target", type=str, required=True,
                        help="append3 + coverage target report (baseline_prefix for v2, baseline_anchored for v3, etc.)")
    parser.add_argument("--report_append0", type=str, required=True, help="append0 + coverage + baseline_prefix report")
    parser.add_argument("--target_label", type=str, default="v2", help="Short label for the target report in outputs.")
    parser.add_argument("--save_dir", type=str, default="", help="Optional runtime save_dir override")
    parser.add_argument("--output_json", type=str, default="", help="Optional output JSON path")
    parser.add_argument("--output_md", type=str, default="", help="Optional output markdown path")
    args = parser.parse_args()

    report_v1_path = Path(args.report_v1).resolve()
    report_target_path = Path(args.report_target).resolve()
    report_append0_path = Path(args.report_append0).resolve()
    report_v1 = load_json(report_v1_path)
    report_target = load_json(report_target_path)
    report_append0 = load_json(report_append0_path)

    v1_lookup = build_report_lookup(report_v1)
    v2_lookup = build_report_lookup(report_target)
    append0_lookup = build_report_lookup(report_append0)
    questions = list(v2_lookup.keys())
    if list(v1_lookup.keys()) != questions or list(append0_lookup.keys()) != questions:
        raise ValueError("Report question order does not match across inputs.")

    resolved_save_dir = resolve_runtime_save_dir(
        report_path=report_target_path,
        dataset=args.dataset,
        save_dir_arg=args.save_dir,
    )
    corpus = load_corpus(args.dataset)
    docs = [f"{row['title']}\n{row['text']}" for row in corpus]
    chunk_id_to_doc_text = build_chunk_id_to_doc_text(corpus)

    runtime_args = build_runtime_args(
        report_payload=report_target,
        dataset=args.dataset,
        qa_top_k=args.qa_top_k,
        save_dir=resolved_save_dir,
    )
    config = build_config(runtime_args, corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    hipporag.prepare_retrieval_objects()

    query_rows: List[Dict[str, Any]] = []
    v1_traces: List[Mapping[str, Any]] = []
    v2_traces: List[Mapping[str, Any]] = []
    append0_traces: List[Mapping[str, Any]] = []
    v1_pool_divergence_count = 0
    append0_pool_divergence_count = 0

    for question in questions:
        v1_trace, v1_example = v1_lookup[question]
        v2_trace, v2_example = v2_lookup[question]
        append0_trace, append0_example = append0_lookup[question]
        v1_traces.append(v1_trace)
        v2_traces.append(v2_trace)
        append0_traces.append(append0_trace)

        retrieved_doc_ids_v2 = list(v2_example.get("retrieved_doc_ids", []) or [])
        if not retrieved_doc_ids_v2:
            raise RuntimeError(f"Missing retrieved_doc_ids for question: {question}")
        if list(v1_example.get("retrieved_doc_ids", []) or []) != retrieved_doc_ids_v2:
            v1_pool_divergence_count += 1
        if list(append0_example.get("retrieved_doc_ids", []) or []) != retrieved_doc_ids_v2:
            append0_pool_divergence_count += 1

        requested_positions: List[int] = []
        for trace_row in (v1_trace, v2_trace, append0_trace):
            expand_trace = dict(trace_row.get("expand_assemble_trace", {}) or {})
            requested_positions.extend(expand_trace.get("candidate_set_positions", []) or [])
            requested_positions.extend(expand_trace.get("baseline_prefix_positions", []) or [])
            requested_positions.extend(expand_trace.get("appended_positions", []) or [])
            requested_positions.extend(expand_trace.get("final_front_pool_positions", []) or [])
        requested_positions = normalize_candidate_positions(requested_positions, len(retrieved_doc_ids_v2))

        pool_limit = min(
            len(retrieved_doc_ids_v2),
            max(
                (max(requested_positions) + 1) if requested_positions else 0,
                int((v2_trace.get("expand_assemble_trace", {}) or {}).get("pool_k", 0) or 0),
                int(args.qa_top_k),
            ),
        )
        pool_docs, _pool_scores, pool_doc_ids = build_pool_mapping_from_report(
            retrieved_doc_ids=retrieved_doc_ids_v2,
            pool_limit=pool_limit,
            chunk_id_to_doc_text=chunk_id_to_doc_text,
            hipporag=hipporag,
        )

        expand_trace_v2 = dict(v2_trace.get("expand_assemble_trace", {}) or {})
        trace_seed_entities = normalize_entity_set(expand_trace_v2.get("seed_entities_preview", []) or [])
        seed_entities = set(trace_seed_entities)
        if not seed_entities:
            seed_entities = collect_query_seed_entities(hipporag, question)
        if not seed_entities:
            seed_entities = collect_lexical_query_seed_entities(
                query=question,
                pool_doc_ids=pool_doc_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )

        query_rows.append(
            summarize_v2_triplet(
                question=question,
                v1_trace=v1_trace,
                v2_trace=v2_trace,
                append0_trace=append0_trace,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                seed_entities=seed_entities,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
            )
        )

    query_count = len(query_rows)
    v1_selected_appended_queries = sum(1 for row in query_rows if row["num_appended_selected_v1"] > 0)
    v2_selected_appended_queries = sum(1 for row in query_rows if row["num_appended_selected_v2"] > 0)
    selected_appended_doc_count_v2 = sum(int(row["num_appended_selected_v2"]) for row in query_rows)
    nonzero_gain_doc_count_v2 = sum(int(row["selected_appended_with_nonzero_unique_gain"]) for row in query_rows)
    zero_gain_doc_count_v2 = sum(int(row["selected_appended_with_zero_unique_gain"]) for row in query_rows)
    route_decision = classify_route(
        selected_appended_query_rate=(v2_selected_appended_queries / query_count) if query_count else 0.0,
        nonzero_gain_doc_rate=(nonzero_gain_doc_count_v2 / selected_appended_doc_count_v2) if selected_appended_doc_count_v2 else 0.0,
    )

    aggregate = {
        "query_count": int(query_count),
        "selected_appended_query_rate_v1": round(v1_selected_appended_queries / query_count, 4) if query_count else 0.0,
        "selected_appended_query_rate_target": round(v2_selected_appended_queries / query_count, 4) if query_count else 0.0,
        "avg_num_appended_selected_v1": _safe_mean(row["num_appended_selected_v1"] for row in query_rows),
        "avg_num_appended_selected_target": _safe_mean(row["num_appended_selected_v2"] for row in query_rows),
        "selected_appended_doc_count_target": int(selected_appended_doc_count_v2),
        "selected_appended_with_nonzero_unique_gain_target": int(nonzero_gain_doc_count_v2),
        "selected_appended_with_zero_unique_gain_target": int(zero_gain_doc_count_v2),
        "selected_appended_nonzero_gain_doc_rate_target": round(nonzero_gain_doc_count_v2 / selected_appended_doc_count_v2, 4) if selected_appended_doc_count_v2 else 0.0,
        "selected_appended_zero_gain_doc_rate_target": round(zero_gain_doc_count_v2 / selected_appended_doc_count_v2, 4) if selected_appended_doc_count_v2 else 0.0,
        "queries_with_any_nonzero_unique_gain_target": int(sum(1 for row in query_rows if row["selected_appended_with_nonzero_unique_gain"] > 0)),
        "effective_decision_layer_distribution_target": summarize_decision_layer_distribution(query_rows),
        "avg_coverage_vs_append0_overlap": _safe_mean(row["coverage_vs_append0_overlap"] for row in query_rows),
        "avg_coverage_vs_v1_overlap": _safe_mean(row["coverage_vs_v1_overlap"] for row in query_rows),
        "avg_replaced_baseline_incumbents_count": _safe_mean(row["replaced_baseline_incumbents_count"] for row in query_rows),
        "new_edge_support_count": {
            "avg_covered_edge_support": _safe_mean(
                row["new_edge_support_count"]["avg_covered_edge_support"] for row in query_rows if row["new_edge_support_count"]["covered_edge_count"] > 0
            ),
            "avg_unique_edge_support": _safe_mean(
                row["new_edge_support_count"]["avg_unique_edge_support"] for row in query_rows if row["new_edge_support_count"]["unique_edge_count"] > 0
            ),
        },
        "retrieved_pool_divergence_count_v1": int(v1_pool_divergence_count),
        "retrieved_pool_divergence_count_append0": int(append0_pool_divergence_count),
    }
    aggregate.update(summarize_admissibility_metrics(query_rows))

    payload = {
        "dataset": args.dataset,
        "qa_top_k": int(args.qa_top_k),
        "target_label": str(args.target_label),
        "report_v1": str(report_v1_path),
        "report_target": str(report_target_path),
        "report_append0": str(report_append0_path),
        "aggregate": aggregate,
        "paired": {
            "target_vs_v1": compare_trace_metrics(v2_traces, v1_traces),
            "target_vs_append0": compare_trace_metrics(v2_traces, append0_traces),
        },
        "route_decision": route_decision,
        "query_rows": query_rows,
    }

    output_json = Path(args.output_json).resolve() if str(args.output_json).strip() else report_target_path.parent / (
        report_target_path.stem + "_v2_trace_diagnostic.json"
    )
    output_md = Path(args.output_md).resolve() if str(args.output_md).strip() else report_target_path.parent / (
        report_target_path.stem + "_v2_trace_diagnostic.md"
    )
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown_summary(payload), encoding="utf-8")

    print(json.dumps({"aggregate": aggregate, "route_decision": route_decision}, ensure_ascii=False, indent=2))
    print(f"Saved JSON to {output_json}")
    print(f"Saved markdown to {output_md}")


if __name__ == "__main__":
    main()
