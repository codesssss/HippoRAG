#!/usr/bin/env python3
"""Audit why transition-valid closure does not improve V4 retrieval.

The audit is diagnostic only.  It uses gold document ids to classify failures,
but no output from this script is consumed by the retriever.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from evidence_transition_graphragv4_fact_witnessed_sto.agsto.index import (  # noqa: E402
    build_corpus_unit_index,
)
from evidence_transition_graphragv4_fact_witnessed_sto.agsto.local_graph import (  # noqa: E402
    _edge_has_openie_fact_witness,
    _edge_has_role_consistent_transition,
    _edge_neighbor,
    _local_edge_adjacency,
    _sort_transition_valid_edges,
    _unit_by_id,
    build_query_local_sto_graph,
)
from evidence_transition_graphragv4_fact_witnessed_sto.agsto.role_transition import (  # noqa: E402
    build_role_transition_graph,
)
from evidence_transition_graphragv4_fact_witnessed_sto.source_authorized_vocab_strict_retrieval.evaluate_report import (  # noqa: E402
    load_nodes,
)
from evidence_transition_graphragv4_fact_witnessed_sto.source_authorized_vocab_strict_retrieval.normalize import (  # noqa: E402
    unique_ints,
)
from evidence_transition_graphragv4_fact_witnessed_sto.source_authorized_vocab_strict_retrieval.certificate_graph import (  # noqa: E402
    build_source_text_certificate_graph,
)
from evidence_transition_graphragv4_fact_witnessed_sto.source_authorized_vocab_strict_retrieval.active_certificate_graph import (  # noqa: E402
    activate_certificate_edges,
)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _node_openie_docs(nodes: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "idx": str(node.doc_index),
            "passage": str(node.text),
            "title": str(node.display_title),
            "extracted_triples": [list(triple) for triple in node.triples],
        }
        for node in nodes
    ]


def _doc_endpoint_coverage(local_graph: Mapping[str, Any]) -> Dict[int, Set[str]]:
    coverage: Dict[int, Set[str]] = defaultdict(set)
    for endpoint, raw_docs in (local_graph.get("symbolic_endpoint_seed_doc_indices", {}) or {}).items():
        endpoint_text = str(endpoint)
        if not endpoint_text.strip():
            continue
        for doc_idx in unique_ints(raw_docs or []):
            coverage[int(doc_idx)].add(endpoint_text)
    return dict(coverage)


def _edge_kind_counter(edges: Iterable[Mapping[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for edge in edges:
        for kind in edge.get("kinds", []) or []:
            counts[str(kind)] += 1
    return counts


def _edges_between(
    *,
    adjacency: Mapping[int, Sequence[Mapping[str, Any]]],
    source_docs: Sequence[int],
    target_doc: int,
) -> List[Mapping[str, Any]]:
    edges: List[Mapping[str, Any]] = []
    target = int(target_doc)
    seen: Set[Tuple[int, int, Tuple[str, ...]]] = set()
    for source_doc in unique_ints(source_docs):
        for edge in adjacency.get(int(source_doc), []) or []:
            if _edge_neighbor(edge, int(source_doc)) != target:
                continue
            key = (
                int(edge.get("left_doc", -1)),
                int(edge.get("right_doc", -1)),
                tuple(str(kind) for kind in edge.get("kinds", []) or []),
            )
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)
    return edges


def _classify_missing_gold(
    *,
    gold_doc: int,
    selected_docs: Sequence[int],
    candidate_docs: Sequence[int],
    local_graph: Mapping[str, Any],
    corpus_unit_by_id: Mapping[int, Mapping[str, Any]],
    selected_to_gold_certificate_counts: Mapping[str, int] | None = None,
    gold_to_selected_certificate_counts: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    selected = set(unique_ints(selected_docs))
    candidates = set(unique_ints(candidate_docs))
    admitted = set(unique_ints(local_graph.get("admitted_doc_indices", []) or []))
    doc_endpoint_coverage = _doc_endpoint_coverage(local_graph)
    selected_endpoint_coverage: Set[str] = set()
    for doc_idx in selected:
        selected_endpoint_coverage.update(doc_endpoint_coverage.get(int(doc_idx), set()))

    doc = int(gold_doc)
    if doc in selected:
        return {
            "status": "already_selected",
            "gold_doc": doc,
            "new_query_endpoints": [],
            "edge_kind_counts": {},
            "selected_to_gold_certificate_counts": {},
            "gold_to_selected_certificate_counts": {},
        }
    selected_to_gold = dict(selected_to_gold_certificate_counts or {})
    gold_to_selected = dict(gold_to_selected_certificate_counts or {})
    certificate_payload = {
        "selected_to_gold_certificate_counts": selected_to_gold,
        "gold_to_selected_certificate_counts": gold_to_selected,
        "has_selected_to_gold_certificate": bool(selected_to_gold),
        "has_gold_to_selected_certificate": bool(gold_to_selected),
    }
    if doc not in candidates:
        return {
            "status": "candidate_missing",
            "gold_doc": doc,
            "new_query_endpoints": sorted(doc_endpoint_coverage.get(doc, set()) - selected_endpoint_coverage),
            "edge_kind_counts": {},
            **certificate_payload,
        }
    if doc not in admitted:
        return {
            "status": "candidate_dense_only_not_admitted",
            "gold_doc": doc,
            "new_query_endpoints": sorted(doc_endpoint_coverage.get(doc, set()) - selected_endpoint_coverage),
            "edge_kind_counts": {},
            **certificate_payload,
        }

    adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
    context_edges = _edges_between(
        adjacency=adjacency,
        source_docs=selected_docs,
        target_doc=doc,
    )
    edge_kind_counts = dict(_edge_kind_counter(context_edges))
    new_endpoints = doc_endpoint_coverage.get(doc, set()) - selected_endpoint_coverage
    if not context_edges:
        return {
            "status": "admitted_no_edge_to_selected_context",
            "gold_doc": doc,
            "new_query_endpoints": sorted(new_endpoints),
            "edge_kind_counts": edge_kind_counts,
            **certificate_payload,
        }

    has_fact_witness = any(_edge_has_openie_fact_witness(edge) for edge in context_edges)
    transition_from_context = any(
        _edge_has_role_consistent_transition(
            edge,
            from_doc=int(source_doc),
            unit_by_id=corpus_unit_by_id,
        )
        for source_doc in unique_ints(selected_docs)
        for edge in adjacency.get(int(source_doc), []) or []
        if _edge_neighbor(edge, int(source_doc)) == doc
    )
    transition_to_context = any(
        _edge_has_role_consistent_transition(
            edge,
            from_doc=doc,
            unit_by_id=corpus_unit_by_id,
        )
        for edge in adjacency.get(doc, []) or []
        if _edge_neighbor(edge, doc) in selected
    )
    if transition_from_context and new_endpoints:
        status = "transition_valid_and_endpoint_available"
    elif transition_from_context:
        status = "transition_valid_but_no_new_query_endpoint"
    elif transition_to_context:
        status = "transition_valid_reverse_only"
    elif has_fact_witness:
        status = "fact_witnessed_but_not_transition_valid"
    else:
        status = "weak_or_title_edge_only"
    return {
        "status": status,
        "gold_doc": doc,
        "new_query_endpoints": sorted(new_endpoints),
        "edge_kind_counts": edge_kind_counts,
        "has_fact_witness": bool(has_fact_witness),
        "transition_from_context": bool(transition_from_context),
        "transition_to_context": bool(transition_to_context),
        **certificate_payload,
    }


def _build_local_graph_for_row(
    *,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    corpus_index: Mapping[str, Any],
    role_graph: Mapping[str, Any],
) -> Mapping[str, Any]:
    trace_local_graph = (
        ((row.get("route_trace", {}) or {}).get("retrieval", {}) or {}).get("local_graph", {}) or {}
    )
    textual_seeds = unique_ints(trace_local_graph.get("textual_seed_doc_indices", []) or [])
    return build_query_local_sto_graph(
        query=str(row.get("question") or row.get("query") or ""),
        corpus_index=corpus_index,
        role_graph=role_graph,
        textual_seed_doc_indices=textual_seeds,
        textual_seed_top_k=max(len(textual_seeds), int(config.get("agsto_textual_seed_top_k", 20) or 20)),
        max_endpoint_degree=max(int(config.get("agsto_max_endpoint_degree", 30) or 30), 1),
        closure_hops=max(int(config.get("agsto_closure_hops", 2) or 2), 0),
        candidate_limit=max(int(config.get("candidate_pool_k", 200) or 200), 1),
        enable_query_supported_same_object_handoff=bool(
            config.get("ablation_query_supported_object_handoff", False)
        ),
        enable_variable_flow_traversal=bool(config.get("enable_variable_flow_traversal", True)),
    )


def run_audit(
    *,
    retrieval_report: Path,
    comparison_report: Path | None,
    max_queries: int,
) -> Mapping[str, Any]:
    payload = _load_json(retrieval_report)
    config = dict(payload.get("config", {}) or {})
    openie_path = Path(str(payload.get("openie_path") or "")).expanduser()
    nodes = load_nodes(openie_path)
    corpus_index = build_corpus_unit_index(_node_openie_docs(nodes))
    role_graph = build_role_transition_graph(
        corpus_index=corpus_index,
        max_endpoint_degree=max(int(config.get("agsto_max_endpoint_degree", 30) or 30), 1),
        include_title_role_grounding=True,
    )
    corpus_unit_by_id = _unit_by_id(corpus_index)
    rows = list(payload.get("rows", []) or [])
    if max_queries > 0:
        rows = rows[:max_queries]

    comparison_rows_by_query: Dict[int, Mapping[str, Any]] = {}
    comparison_metrics: Mapping[str, Any] = {}
    if comparison_report is not None:
        comparison_payload = _load_json(comparison_report)
        comparison_metrics = dict(comparison_payload.get("metrics", {}) or {})
        for row in comparison_payload.get("rows", []) or []:
            comparison_rows_by_query[int(row.get("query_index", len(comparison_rows_by_query)))] = row

    status_counts: Counter = Counter()
    missing_status_counts: Counter = Counter()
    query_primary_status_counts: Counter = Counter()
    edge_kind_counts_by_status: Dict[str, Counter] = defaultdict(Counter)
    selected_to_gold_certificate_counts_by_status: Dict[str, Counter] = defaultdict(Counter)
    gold_to_selected_certificate_counts_by_status: Dict[str, Counter] = defaultdict(Counter)
    selected_to_gold_certificate_status_counts: Counter = Counter()
    gold_to_selected_certificate_status_counts: Counter = Counter()
    examples: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    query_rows: List[Mapping[str, Any]] = []
    selected_gold_count = 0
    total_gold_count = 0
    selected_to_candidate_certificate_target_count = 0
    selected_to_gold_certificate_target_count = 0
    selected_to_missing_gold_certificate_target_count = 0
    active_selected_to_candidate_certificate_target_count = 0
    active_selected_to_gold_certificate_target_count = 0
    active_selected_to_missing_gold_certificate_target_count = 0
    transition_valid_frontier_target_count = 0
    transition_valid_frontier_gold_target_count = 0
    transition_valid_frontier_missing_gold_target_count = 0
    active_certified_transition_target_count = 0
    active_certified_transition_gold_target_count = 0
    active_certified_transition_missing_gold_target_count = 0
    certificate_target_counts_by_type: Dict[str, Counter] = defaultdict(Counter)
    active_certificate_target_counts_by_type: Dict[str, Counter] = defaultdict(Counter)

    for fallback_idx, row in enumerate(rows):
        query_index = int(row.get("query_index", fallback_idx))
        local_graph = _build_local_graph_for_row(
            row=row,
            config=config,
            corpus_index=corpus_index,
            role_graph=role_graph,
        )
        selected = unique_ints(row.get("retrieved_doc_indices_top5", []) or [])
        candidate_docs = unique_ints(
            (((row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}).get("candidate_doc_indices", []) or [])
        )
        gold_docs = unique_ints(row.get("gold_doc_indices", []) or [])
        admitted_set = set(unique_ints(local_graph.get("admitted_doc_indices", []) or []))
        local_adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
        transition_valid_targets: Set[int] = set()
        for source_doc in unique_ints(selected):
            for edge in _sort_transition_valid_edges(
                local_adjacency.get(int(source_doc), []) or [],
                from_doc=int(source_doc),
                unit_by_id=corpus_unit_by_id,
            ):
                neighbor = _edge_neighbor(edge, int(source_doc))
                if neighbor >= 0 and neighbor in admitted_set and neighbor not in set(unique_ints(selected)):
                    transition_valid_targets.add(int(neighbor))
        selected_to_gold_certificates_by_doc: Dict[int, Counter] = defaultdict(Counter)
        gold_to_selected_certificates_by_doc: Dict[int, Counter] = defaultdict(Counter)
        if candidate_docs and selected:
            certificate_graph = build_source_text_certificate_graph(
                query=str(row.get("question") or row.get("query") or ""),
                nodes=nodes,
                candidate_doc_indices=candidate_docs,
                source_doc_indices=selected,
            )
            selected_set = {int(doc_idx) for doc_idx in unique_ints(selected)}
            gold_set = {int(doc_idx) for doc_idx in unique_ints(gold_docs)}
            selected_to_candidate_targets: Set[int] = set()
            selected_targets_by_type: Dict[str, Set[int]] = defaultdict(set)
            for certificate in certificate_graph.certificates:
                source = int(getattr(certificate, "source_doc_index"))
                target = int(getattr(certificate, "target_doc_index"))
                certificate_type = str(getattr(certificate, "certificate_type", ""))
                if source in selected_set:
                    selected_to_candidate_targets.add(target)
                    selected_targets_by_type[certificate_type].add(target)
                    selected_to_gold_certificates_by_doc[target][certificate_type] += 1
            selected_to_candidate_targets -= selected_set
            for certificate_type, raw_targets in selected_targets_by_type.items():
                targets = set(raw_targets) - selected_set
                certificate_target_counts_by_type[certificate_type]["targets"] += len(targets)
                certificate_target_counts_by_type[certificate_type]["gold_targets"] += len(targets & gold_set)
                certificate_target_counts_by_type[certificate_type]["missing_gold_targets"] += len(
                    targets & (gold_set - selected_set)
                )
            selected_to_candidate_certificate_target_count += len(selected_to_candidate_targets)
            selected_to_gold_certificate_target_count += len(selected_to_candidate_targets & gold_set)
            selected_to_missing_gold_certificate_target_count += len(
                selected_to_candidate_targets & (gold_set - selected_set)
            )
            active_graph = activate_certificate_edges(
                query=str(row.get("question") or row.get("query") or ""),
                nodes=nodes,
                certificate_graph=certificate_graph,
                candidate_doc_indices=candidate_docs,
                entry_doc_indices=selected,
                query_mentioned_doc_indices=(),
                require_source_entry=True,
            )
            active_targets = {
                int(edge.certificate.target_doc_index)
                for edge in active_graph.active_edges
                if int(edge.certificate.source_doc_index) in selected_set
            }
            active_targets_by_type: Dict[str, Set[int]] = defaultdict(set)
            for edge in active_graph.active_edges:
                certificate = edge.certificate
                if int(certificate.source_doc_index) not in selected_set:
                    continue
                active_targets_by_type[str(certificate.certificate_type)].add(
                    int(certificate.target_doc_index)
                )
            active_targets -= selected_set
            for certificate_type, raw_targets in active_targets_by_type.items():
                targets = set(raw_targets) - selected_set
                active_certificate_target_counts_by_type[certificate_type]["targets"] += len(targets)
                active_certificate_target_counts_by_type[certificate_type]["gold_targets"] += len(
                    targets & gold_set
                )
                active_certificate_target_counts_by_type[certificate_type][
                    "missing_gold_targets"
                ] += len(targets & (gold_set - selected_set))
            active_selected_to_candidate_certificate_target_count += len(active_targets)
            active_selected_to_gold_certificate_target_count += len(active_targets & gold_set)
            active_selected_to_missing_gold_certificate_target_count += len(
                active_targets & (gold_set - selected_set)
            )
            transition_valid_frontier_target_count += len(transition_valid_targets)
            transition_valid_frontier_gold_target_count += len(transition_valid_targets & gold_set)
            transition_valid_frontier_missing_gold_target_count += len(
                transition_valid_targets & (gold_set - selected_set)
            )
            active_certified_transition_targets = active_targets & transition_valid_targets
            active_certified_transition_target_count += len(active_certified_transition_targets)
            active_certified_transition_gold_target_count += len(
                active_certified_transition_targets & gold_set
            )
            active_certified_transition_missing_gold_target_count += len(
                active_certified_transition_targets & (gold_set - selected_set)
            )
            gold_as_source_graph = build_source_text_certificate_graph(
                query=str(row.get("question") or row.get("query") or ""),
                nodes=nodes,
                candidate_doc_indices=candidate_docs,
                source_doc_indices=row.get("gold_doc_indices", []) or [],
            )
            for certificate in gold_as_source_graph.certificates:
                source = int(getattr(certificate, "source_doc_index"))
                target = int(getattr(certificate, "target_doc_index"))
                certificate_type = str(getattr(certificate, "certificate_type", ""))
                if target in selected_set:
                    gold_to_selected_certificates_by_doc[source][certificate_type] += 1
        total_gold_count += len(gold_docs)
        selected_gold_count += len([doc for doc in gold_docs if int(doc) in set(selected)])
        gold_status_rows = [
            _classify_missing_gold(
                gold_doc=int(gold_doc),
                selected_docs=selected,
                candidate_docs=candidate_docs,
                local_graph=local_graph,
                corpus_unit_by_id=corpus_unit_by_id,
                selected_to_gold_certificate_counts=selected_to_gold_certificates_by_doc.get(
                    int(gold_doc), {}
                ),
                gold_to_selected_certificate_counts=gold_to_selected_certificates_by_doc.get(
                    int(gold_doc), {}
                ),
            )
            for gold_doc in gold_docs
        ]
        for status_row in gold_status_rows:
            status = str(status_row.get("status") or "")
            status_counts[status] += 1
            if status != "already_selected":
                missing_status_counts[status] += 1
            edge_kind_counts_by_status[status].update(status_row.get("edge_kind_counts", {}) or {})
            selected_to_gold_counts = status_row.get("selected_to_gold_certificate_counts", {}) or {}
            gold_to_selected_counts = status_row.get("gold_to_selected_certificate_counts", {}) or {}
            selected_to_gold_certificate_counts_by_status[status].update(selected_to_gold_counts)
            gold_to_selected_certificate_counts_by_status[status].update(gold_to_selected_counts)
            if selected_to_gold_counts:
                selected_to_gold_certificate_status_counts[status] += 1
            if gold_to_selected_counts:
                gold_to_selected_certificate_status_counts[status] += 1
            if len(examples[status]) < 5:
                examples[status].append(
                    {
                        "query_index": query_index,
                        "question": str(row.get("question") or ""),
                        "gold_doc": int(status_row.get("gold_doc", -1)),
                        "gold_doc_indices": gold_docs,
                        "selected_doc_indices": selected,
                        "status": status,
                        "new_query_endpoints": list(status_row.get("new_query_endpoints", []) or []),
                        "edge_kind_counts": dict(status_row.get("edge_kind_counts", {}) or {}),
                        "selected_to_gold_certificate_counts": dict(selected_to_gold_counts),
                        "gold_to_selected_certificate_counts": dict(gold_to_selected_counts),
                    }
                )
        missing_rows = [status for status in gold_status_rows if status.get("status") != "already_selected"]
        primary_status = "all_gold_selected"
        if missing_rows:
            primary_status = str(missing_rows[0].get("status") or "unknown")
        query_primary_status_counts[primary_status] += 1
        comparison_row = comparison_rows_by_query.get(query_index)
        query_rows.append(
            {
                "query_index": query_index,
                "question": str(row.get("question") or ""),
                "gold_doc_indices": gold_docs,
                "selected_doc_indices": selected,
                "recall_at5": float(row.get("query_grounded_sto_recall_at5", 0.0) or 0.0),
                "all_gold_at5": bool(row.get("query_grounded_sto_all_gold_at5", False)),
                "gold_status": gold_status_rows,
                "comparison_doc_indices": (
                    unique_ints(comparison_row.get("retrieved_doc_indices_top5", []) or [])
                    if comparison_row
                    else []
                ),
                "comparison_recall_at5": (
                    float(comparison_row.get("query_grounded_sto_recall_at5", 0.0) or 0.0)
                    if comparison_row
                    else None
                ),
            }
        )

    return {
        "audit": "transition_valid_edge_failure_audit",
        "source_report": str(retrieval_report),
        "comparison_report": str(comparison_report) if comparison_report else "",
        "dataset": str(payload.get("dataset") or ""),
        "row_count": len(rows),
        "source_metrics": dict(payload.get("metrics", {}) or {}),
        "comparison_metrics": dict(comparison_metrics),
        "gold_doc_count": int(total_gold_count),
        "selected_gold_doc_count": int(selected_gold_count),
        "missing_gold_doc_count": int(total_gold_count - selected_gold_count),
        "gold_status_counts": dict(status_counts),
        "missing_gold_status_counts": dict(missing_status_counts),
        "query_primary_status_counts": dict(query_primary_status_counts),
        "edge_kind_counts_by_status": {
            status: dict(counter)
            for status, counter in sorted(edge_kind_counts_by_status.items())
        },
        "selected_to_gold_certificate_counts_by_status": {
            status: dict(counter)
            for status, counter in sorted(selected_to_gold_certificate_counts_by_status.items())
        },
        "gold_to_selected_certificate_counts_by_status": {
            status: dict(counter)
            for status, counter in sorted(gold_to_selected_certificate_counts_by_status.items())
        },
        "selected_to_gold_certificate_status_counts": dict(selected_to_gold_certificate_status_counts),
        "gold_to_selected_certificate_status_counts": dict(gold_to_selected_certificate_status_counts),
        "selected_to_candidate_certificate_target_count": int(
            selected_to_candidate_certificate_target_count
        ),
        "selected_to_gold_certificate_target_count": int(selected_to_gold_certificate_target_count),
        "selected_to_missing_gold_certificate_target_count": int(
            selected_to_missing_gold_certificate_target_count
        ),
        "active_selected_to_candidate_certificate_target_count": int(
            active_selected_to_candidate_certificate_target_count
        ),
        "active_selected_to_gold_certificate_target_count": int(
            active_selected_to_gold_certificate_target_count
        ),
        "active_selected_to_missing_gold_certificate_target_count": int(
            active_selected_to_missing_gold_certificate_target_count
        ),
        "transition_valid_frontier_target_count": int(transition_valid_frontier_target_count),
        "transition_valid_frontier_gold_target_count": int(
            transition_valid_frontier_gold_target_count
        ),
        "transition_valid_frontier_missing_gold_target_count": int(
            transition_valid_frontier_missing_gold_target_count
        ),
        "active_certified_transition_target_count": int(active_certified_transition_target_count),
        "active_certified_transition_gold_target_count": int(
            active_certified_transition_gold_target_count
        ),
        "active_certified_transition_missing_gold_target_count": int(
            active_certified_transition_missing_gold_target_count
        ),
        "certificate_target_counts_by_type": {
            certificate_type: dict(counter)
            for certificate_type, counter in sorted(certificate_target_counts_by_type.items())
        },
        "active_certificate_target_counts_by_type": {
            certificate_type: dict(counter)
            for certificate_type, counter in sorted(active_certificate_target_counts_by_type.items())
        },
        "examples": {status: rows for status, rows in sorted(examples.items())},
        "rows": query_rows,
    }


def _format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def write_markdown(payload: Mapping[str, Any], output_md: Path) -> None:
    source_metrics = payload.get("source_metrics", {}) or {}
    comparison_metrics = payload.get("comparison_metrics", {}) or {}
    lines = [
        "# Transition-Valid Edge Failure Audit",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| rows | {payload.get('row_count', 0)} |",
        f"| source report | `{payload.get('source_report', '')}` |",
        f"| comparison report | `{payload.get('comparison_report', '')}` |",
        "",
        "| method | R@5 | all-gold@5 | mean certified docs@5 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| source | {_format_float(source_metrics.get('r5'))} | "
            f"{_format_float(source_metrics.get('all_gold_at5'))} | "
            f"{_format_float(source_metrics.get('mean_certified_doc_count_top5'))} |"
        ),
    ]
    if comparison_metrics:
        lines.append(
            f"| comparison | {_format_float(comparison_metrics.get('r5'))} | "
            f"{_format_float(comparison_metrics.get('all_gold_at5'))} | "
            f"{_format_float(comparison_metrics.get('mean_certified_doc_count_top5'))} |"
        )
    lines.extend(
        [
            "",
            "| gold status | count |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted((payload.get("gold_status_counts", {}) or {}).items()):
        lines.append(f"| {status} | {int(count)} |")
    lines.extend(
        [
            "",
            "| missing-gold status | count |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted((payload.get("missing_gold_status_counts", {}) or {}).items()):
        lines.append(f"| {status} | {int(count)} |")
    lines.extend(
        [
            "",
            "| query primary status | count |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted((payload.get("query_primary_status_counts", {}) or {}).items()):
        lines.append(f"| {status} | {int(count)} |")
    lines.extend(
        [
            "",
            "| missing-gold status | selected->gold certificate docs | gold->selected certificate docs |",
            "| --- | ---: | ---: |",
        ]
    )
    selected_to_gold_status_counts = payload.get("selected_to_gold_certificate_status_counts", {}) or {}
    gold_to_selected_status_counts = payload.get("gold_to_selected_certificate_status_counts", {}) or {}
    for status in sorted(set(selected_to_gold_status_counts) | set(gold_to_selected_status_counts)):
        lines.append(
            f"| {status} | {int(selected_to_gold_status_counts.get(status, 0))} | "
            f"{int(gold_to_selected_status_counts.get(status, 0))} |"
        )
    lines.extend(
        [
            "",
            "| certificate precision field | value |",
            "| --- | ---: |",
            (
                "| selected->candidate certificate targets | "
                f"{int(payload.get('selected_to_candidate_certificate_target_count', 0))} |"
            ),
            (
                "| selected->gold certificate targets | "
                f"{int(payload.get('selected_to_gold_certificate_target_count', 0))} |"
            ),
            (
                "| selected->missing-gold certificate targets | "
                f"{int(payload.get('selected_to_missing_gold_certificate_target_count', 0))} |"
            ),
            (
                "| active selected->candidate certificate targets | "
                f"{int(payload.get('active_selected_to_candidate_certificate_target_count', 0))} |"
            ),
            (
                "| active selected->gold certificate targets | "
                f"{int(payload.get('active_selected_to_gold_certificate_target_count', 0))} |"
            ),
            (
                "| active selected->missing-gold certificate targets | "
                f"{int(payload.get('active_selected_to_missing_gold_certificate_target_count', 0))} |"
            ),
            (
                "| transition-valid frontier targets | "
                f"{int(payload.get('transition_valid_frontier_target_count', 0))} |"
            ),
            (
                "| transition-valid frontier gold targets | "
                f"{int(payload.get('transition_valid_frontier_gold_target_count', 0))} |"
            ),
            (
                "| transition-valid frontier missing-gold targets | "
                f"{int(payload.get('transition_valid_frontier_missing_gold_target_count', 0))} |"
            ),
            (
                "| active-certified transition targets | "
                f"{int(payload.get('active_certified_transition_target_count', 0))} |"
            ),
            (
                "| active-certified transition gold targets | "
                f"{int(payload.get('active_certified_transition_gold_target_count', 0))} |"
            ),
            (
                "| active-certified transition missing-gold targets | "
                f"{int(payload.get('active_certified_transition_missing_gold_target_count', 0))} |"
            ),
            "",
            "| certificate type | targets | gold targets | missing-gold targets | precision |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for certificate_type, counts in sorted((payload.get("certificate_target_counts_by_type", {}) or {}).items()):
        targets = int((counts or {}).get("targets", 0))
        gold_targets = int((counts or {}).get("gold_targets", 0))
        precision = (float(gold_targets) / float(targets)) if targets else 0.0
        lines.append(
            f"| {certificate_type} | {targets} | {gold_targets} | "
            f"{int((counts or {}).get('missing_gold_targets', 0))} | {precision:.4f} |"
        )
    lines.extend(
        [
            "",
            "| active certificate type | targets | gold targets | missing-gold targets | precision |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for certificate_type, counts in sorted((payload.get("active_certificate_target_counts_by_type", {}) or {}).items()):
        targets = int((counts or {}).get("targets", 0))
        gold_targets = int((counts or {}).get("gold_targets", 0))
        precision = (float(gold_targets) / float(targets)) if targets else 0.0
        lines.append(
            f"| {certificate_type} | {targets} | {gold_targets} | "
            f"{int((counts or {}).get('missing_gold_targets', 0))} | {precision:.4f} |"
        )
    lines.extend(
        [
            "## Examples",
            "",
        ]
    )
    for status, rows in sorted((payload.get("examples", {}) or {}).items()):
        lines.extend(
            [
                f"### {status}",
                "",
                "| query | gold doc | selected top5 | new endpoints | edge kinds |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in rows[:5]:
            lines.append(
                f"| {int(row.get('query_index', -1))} | {int(row.get('gold_doc', -1))} | "
                f"`{row.get('selected_doc_indices', [])}` | "
                f"`{row.get('new_query_endpoints', [])}` | "
                f"`{row.get('edge_kind_counts', {})}; "
                f"s2g={row.get('selected_to_gold_certificate_counts', {})}; "
                f"g2s={row.get('gold_to_selected_certificate_counts', {})}` |"
            )
        lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--comparison-report", default="")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_audit(
        retrieval_report=Path(args.retrieval_report),
        comparison_report=Path(args.comparison_report) if str(args.comparison_report).strip() else None,
        max_queries=max(int(args.max_queries), 0),
    )
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, Path(args.output_md).expanduser())
    print(
        json.dumps(
            {
                "dataset": payload.get("dataset"),
                "rows": payload.get("row_count"),
                "missing_gold_status_counts": payload.get("missing_gold_status_counts"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
