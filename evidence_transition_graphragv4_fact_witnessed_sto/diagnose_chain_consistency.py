#!/usr/bin/env python3
"""Diagnose chain-consistency failure modes for ETv4 graph readout.

This script is diagnostic-only. It reconstructs each query-local STO graph from
the saved ETv4 retrieval report and compares ETv4 top-5 against the same-entry
dense prefix saved in the trace. Gold labels are used only for offline
classification of gain/loss buckets; no output from this script is consumed by
the retriever.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from evidence_transition_graphragv4_fact_witnessed_sto.audit_transition_valid_edges import (  # noqa: E402
    _build_local_graph_for_row,
    _load_json,
    _node_openie_docs,
)
from evidence_transition_graphragv4_fact_witnessed_sto.agsto.index import (  # noqa: E402
    build_corpus_unit_index,
)
from evidence_transition_graphragv4_fact_witnessed_sto.agsto.local_graph import (  # noqa: E402
    EVIDENCE_TRANSITION_TIER,
    _edge_has_openie_fact_witness,
    _edge_has_role_consistent_transition,
    _edge_neighbor,
    _local_edge_adjacency,
    _unit_by_id,
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


DEFAULT_RETRIEVAL_REPORT = Path(
    "run_logs/etv4_full1000_optimized_equivalence_20260511/"
    "musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
)
DEFAULT_OUTPUT_DIR = Path("run_logs/etv4_full1000_optimized_equivalence_20260511/reports")

STRONG_KIND_ALLOWLIST = {
    "sentence_grounded_transition",
    "source_endpoint_incidence",
    "title_role_grounding",
    "same_subject",
}


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hit_count(docs: Sequence[int], gold_docs: Sequence[int]) -> int:
    gold = {int(doc_idx) for doc_idx in unique_ints(gold_docs)}
    return len({int(doc_idx) for doc_idx in unique_ints(docs)} & gold)


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _round4(value: float) -> float:
    return round(float(value), 4)


def _edge_kind_set(edge: Mapping[str, Any]) -> set[str]:
    return {str(kind) for kind in edge.get("kinds", []) or [] if str(kind).strip()}


def _edge_tier_set(edge: Mapping[str, Any]) -> set[str]:
    return {
        str(tier)
        for tier in edge.get("evidence_tiers", []) or []
        if str(tier).strip()
    }


def _edge_has_evidence_transition_tier(edge: Mapping[str, Any]) -> bool:
    best = str(edge.get("best_evidence_tier", "") or "")
    return best == EVIDENCE_TRANSITION_TIER or EVIDENCE_TRANSITION_TIER in _edge_tier_set(edge)


def _edge_is_chain_strong(
    *,
    edge: Mapping[str, Any],
    from_doc: int,
    unit_by_id: Mapping[int, Mapping[str, Any]],
) -> bool:
    """Return whether the edge is strong enough to count as chain evidence.

    Fact-witnessed weak connectivity is reported separately. For this diagnostic
    we reserve "chain-strong" for evidence-tier or role-consistent transitions,
    because the suspected failure mode is weak topical branches replacing
    answer-tail evidence.
    """

    if _edge_has_evidence_transition_tier(edge):
        return True
    if _edge_kind_set(edge) & STRONG_KIND_ALLOWLIST:
        return True
    return _edge_has_role_consistent_transition(edge, from_doc=int(from_doc), unit_by_id=unit_by_id)


def _admission_distance(local_graph: Mapping[str, Any], doc_idx: int) -> int:
    trace = local_graph.get("doc_admission_trace", {}) or {}
    if not isinstance(trace, Mapping):
        return 0
    row = trace.get(str(int(doc_idx)), {}) or trace.get(int(doc_idx), {}) or {}
    if not isinstance(row, Mapping):
        return 0
    return _safe_int(row.get("distance", 0), default=0)


def _doc_query_coverage(local_graph: Mapping[str, Any], doc_idx: int) -> tuple[str, ...]:
    coverage = local_graph.get("doc_query_token_coverage", {}) or {}
    if not isinstance(coverage, Mapping):
        return tuple()
    values = coverage.get(str(int(doc_idx)), ()) or coverage.get(int(doc_idx), ()) or ()
    return tuple(sorted({str(value) for value in values if str(value).strip()}))


def _internal_edges_for_doc(
    *,
    doc_idx: int,
    selected_docs: Sequence[int],
    adjacency: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    selected_set = {int(doc) for doc in unique_ints(selected_docs)}
    out: list[Mapping[str, Any]] = []
    seen: set[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = set()
    for edge in adjacency.get(int(doc_idx), []) or []:
        neighbor = _edge_neighbor(edge, int(doc_idx))
        if neighbor < 0 or int(neighbor) not in selected_set:
            continue
        key = (
            min(int(doc_idx), int(neighbor)),
            max(int(doc_idx), int(neighbor)),
            tuple(sorted(_edge_kind_set(edge))),
            tuple(sorted(_edge_tier_set(edge))),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _classify_inserted_doc(
    *,
    doc_idx: int,
    selected_docs: Sequence[int],
    dense_top5: Sequence[int],
    local_graph: Mapping[str, Any],
    adjacency: Mapping[int, Sequence[Mapping[str, Any]]],
    unit_by_id: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    selected = list(unique_ints(selected_docs))
    selected_pos = {int(doc): pos for pos, doc in enumerate(selected)}
    source_prior_set = {int(doc) for doc in unique_ints(dense_top5)}
    source_prior_coverage: set[str] = set()
    for doc in source_prior_set:
        source_prior_coverage.update(_doc_query_coverage(local_graph, int(doc)))

    edges = _internal_edges_for_doc(
        doc_idx=int(doc_idx),
        selected_docs=selected,
        adjacency=adjacency,
    )
    doc_pos = selected_pos.get(int(doc_idx), 10**9)
    internal_neighbors: set[int] = set()
    chain_strong_neighbors: set[int] = set()
    fact_witness_neighbors: set[int] = set()
    evidence_tier_neighbors: set[int] = set()
    later_chain_strong_neighbors: set[int] = set()
    later_evidence_tier_neighbors: set[int] = set()
    edge_kind_counts: Counter[str] = Counter()
    edge_tier_counts: Counter[str] = Counter()
    examples: list[Mapping[str, Any]] = []

    for edge in edges:
        neighbor = _edge_neighbor(edge, int(doc_idx))
        if neighbor < 0:
            continue
        internal_neighbors.add(int(neighbor))
        kinds = _edge_kind_set(edge)
        tiers = _edge_tier_set(edge)
        edge_kind_counts.update(kinds)
        edge_tier_counts.update(tiers)
        has_fact = _edge_has_openie_fact_witness(edge)
        is_evidence = _edge_has_evidence_transition_tier(edge)
        is_chain_strong = _edge_is_chain_strong(
            edge=edge,
            from_doc=int(doc_idx),
            unit_by_id=unit_by_id,
        ) or _edge_is_chain_strong(
            edge=edge,
            from_doc=int(neighbor),
            unit_by_id=unit_by_id,
        )
        if has_fact:
            fact_witness_neighbors.add(int(neighbor))
        if is_evidence:
            evidence_tier_neighbors.add(int(neighbor))
            if selected_pos.get(int(neighbor), -1) > doc_pos:
                later_evidence_tier_neighbors.add(int(neighbor))
        if is_chain_strong:
            chain_strong_neighbors.add(int(neighbor))
            if selected_pos.get(int(neighbor), -1) > doc_pos:
                later_chain_strong_neighbors.add(int(neighbor))
        if len(examples) < 3:
            examples.append(
                {
                    "neighbor": int(neighbor),
                    "kinds": sorted(kinds),
                    "tiers": sorted(tiers),
                    "best_tier": str(edge.get("best_evidence_tier", "")),
                    "has_fact_witness": bool(has_fact),
                    "is_chain_strong": bool(is_chain_strong),
                    "endpoints": tuple(str(endpoint) for endpoint in edge.get("endpoints", []) or []),
                }
            )

    query_coverage = set(_doc_query_coverage(local_graph, int(doc_idx)))
    new_query_coverage = query_coverage - source_prior_coverage

    if later_chain_strong_neighbors or len(chain_strong_neighbors) >= 2:
        chain_class = "chain_consumed"
    elif chain_strong_neighbors:
        chain_class = "supported_leaf"
    elif fact_witness_neighbors or internal_neighbors:
        chain_class = "weak_or_topical_leaf"
    else:
        chain_class = "isolated_fill"

    if later_evidence_tier_neighbors or len(evidence_tier_neighbors) >= 2:
        evidence_class = "evidence_consumed"
    elif evidence_tier_neighbors:
        evidence_class = "evidence_leaf"
    elif fact_witness_neighbors or internal_neighbors:
        evidence_class = "weak_graph_connected"
    else:
        evidence_class = "isolated_fill"

    return {
        "doc_index": int(doc_idx),
        "class": chain_class,
        "evidence_class": evidence_class,
        "admission_distance": _admission_distance(local_graph, int(doc_idx)),
        "internal_degree": len(internal_neighbors),
        "chain_strong_degree": len(chain_strong_neighbors),
        "later_chain_strong_degree": len(later_chain_strong_neighbors),
        "fact_witness_degree": len(fact_witness_neighbors),
        "evidence_tier_degree": len(evidence_tier_neighbors),
        "later_evidence_tier_degree": len(later_evidence_tier_neighbors),
        "new_query_coverage": tuple(sorted(new_query_coverage)),
        "edge_kind_counts": dict(sorted(edge_kind_counts.items())),
        "edge_tier_counts": dict(sorted(edge_tier_counts.items())),
        "edge_examples": tuple(examples),
    }


def _row_bucket(delta_hits: int, top5_changed: bool) -> str:
    if delta_hits > 0:
        return "gain"
    if delta_hits < 0:
        return "loss"
    if top5_changed:
        return "changed_tie"
    return "unchanged_tie"


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    rendered = []
    widths: dict[str, int] = {}
    for key, label in columns:
        values = [label, *[str(row.get(key, "")) for row in rows]]
        widths[key] = max(len(value) for value in values)
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    sep = "  ".join("-" * widths[key] for key, _ in columns)
    rendered.append(header)
    rendered.append(sep)
    for row in rows:
        rendered.append("  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns))
    return "```text\n" + "\n".join(rendered) + "\n```"


def _pct(part: float, whole: float) -> float:
    return round((float(part) / float(whole) * 100.0), 2) if whole else 0.0


def _summarize_row_group(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    inserted_doc_count = sum(int(row.get("inserted_doc_count", 0) or 0) for row in rows)
    class_counts: Counter[str] = Counter()
    evidence_class_counts: Counter[str] = Counter()
    dropped_gold_ranks: list[int] = []
    for row in rows:
        class_counts.update(dict(row.get("inserted_class_counts", {}) or {}))
        evidence_class_counts.update(dict(row.get("inserted_evidence_class_counts", {}) or {}))
        dropped_gold_ranks.extend(int(rank) for rank in row.get("dropped_gold_dense_ranks", []) or [])
    chain_consumed = int(class_counts.get("chain_consumed", 0))
    supported_leaf = int(class_counts.get("supported_leaf", 0))
    weak_leaf = int(class_counts.get("weak_or_topical_leaf", 0))
    isolated = int(class_counts.get("isolated_fill", 0))
    non_consumed = supported_leaf + weak_leaf + isolated
    evidence_consumed = int(evidence_class_counts.get("evidence_consumed", 0))
    evidence_leaf = int(evidence_class_counts.get("evidence_leaf", 0))
    weak_graph_connected = int(evidence_class_counts.get("weak_graph_connected", 0))
    evidence_isolated = int(evidence_class_counts.get("isolated_fill", 0))
    return {
        "rows": len(rows),
        "inserted_docs": inserted_doc_count,
        "chain_consumed_docs": chain_consumed,
        "supported_leaf_docs": supported_leaf,
        "weak_topical_leaf_docs": weak_leaf,
        "isolated_fill_docs": isolated,
        "chain_consumed_pct": _pct(chain_consumed, inserted_doc_count),
        "non_consumed_pct": _pct(non_consumed, inserted_doc_count),
        "weak_or_isolated_pct": _pct(weak_leaf + isolated, inserted_doc_count),
        "evidence_consumed_docs": evidence_consumed,
        "evidence_leaf_docs": evidence_leaf,
        "weak_graph_connected_docs": weak_graph_connected,
        "evidence_isolated_docs": evidence_isolated,
        "evidence_consumed_pct": _pct(evidence_consumed, inserted_doc_count),
        "evidence_supported_pct": _pct(evidence_consumed + evidence_leaf, inserted_doc_count),
        "weak_graph_connected_pct": _pct(weak_graph_connected, inserted_doc_count),
        "mean_dropped_gold_dense_rank": _round4(_mean_or_zero(dropped_gold_ranks)),
        "tail_dropped_gold_pct": _pct(
            sum(1 for rank in dropped_gold_ranks if int(rank) >= 4),
            len(dropped_gold_ranks),
        ),
    }


def run_diagnostic(
    *,
    retrieval_report: Path,
    output_dir: Path,
    max_queries: int,
    include_ties: bool,
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

    query_rows: list[Mapping[str, Any]] = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    hop_bucket_counts: Counter[tuple[int, str]] = Counter()

    for fallback_idx, row in enumerate(rows):
        query_index = _safe_int(row.get("query_index", fallback_idx), fallback_idx)
        selected = list(unique_ints(row.get("retrieved_doc_indices_top5", []) or []))[:5]
        candidate_universe = (row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}
        dense_top5 = list(
            unique_ints(candidate_universe.get("source_prior_prefix_doc_indices", []) or [])
        )[:5]
        if not dense_top5:
            dense_top5 = list(unique_ints(candidate_universe.get("candidate_doc_indices", []) or []))[:5]
        gold_docs = list(unique_ints(row.get("gold_doc_indices", []) or []))
        hop = len(gold_docs)
        dense_hits = _hit_count(dense_top5, gold_docs)
        etv4_hits = _hit_count(selected, gold_docs)
        delta_hits = etv4_hits - dense_hits
        top5_changed = tuple(dense_top5) != tuple(selected)
        bucket = _row_bucket(delta_hits, top5_changed)
        hop_bucket_counts[(hop, bucket)] += 1
        if bucket == "unchanged_tie":
            continue
        if bucket == "changed_tie" and not include_ties:
            continue

        local_graph = _build_local_graph_for_row(
            row=row,
            config=config,
            corpus_index=corpus_index,
            role_graph=role_graph,
        )
        adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
        dense_set = {int(doc) for doc in dense_top5}
        selected_set = {int(doc) for doc in selected}
        inserted = [int(doc) for doc in selected if int(doc) not in dense_set]
        removed = [int(doc) for doc in dense_top5 if int(doc) not in selected_set]
        dropped_gold = [int(doc) for doc in removed if int(doc) in set(gold_docs)]
        dropped_gold_dense_ranks = [
            int(dense_top5.index(int(doc)) + 1)
            for doc in dropped_gold
            if int(doc) in dense_top5
        ]
        inserted_infos = [
            _classify_inserted_doc(
                doc_idx=int(doc),
                selected_docs=selected,
                dense_top5=dense_top5,
                local_graph=local_graph,
                adjacency=adjacency,
                unit_by_id=corpus_unit_by_id,
            )
            for doc in inserted
        ]
        inserted_class_counts = Counter(str(info.get("class", "")) for info in inserted_infos)
        inserted_evidence_class_counts = Counter(
            str(info.get("evidence_class", "")) for info in inserted_infos
        )
        row_payload = {
            "query_index": query_index,
            "question": str(row.get("question", "")),
            "hop": hop,
            "bucket": bucket,
            "dense_top5": tuple(dense_top5),
            "etv4_top5": tuple(selected),
            "gold_doc_indices": tuple(gold_docs),
            "dense_hits": dense_hits,
            "etv4_hits": etv4_hits,
            "delta_hits": delta_hits,
            "inserted_doc_indices": tuple(inserted),
            "removed_dense_doc_indices": tuple(removed),
            "dropped_gold_doc_indices": tuple(dropped_gold),
            "dropped_gold_dense_ranks": tuple(dropped_gold_dense_ranks),
            "inserted_doc_count": len(inserted),
            "inserted_class_counts": dict(sorted(inserted_class_counts.items())),
            "inserted_evidence_class_counts": dict(sorted(inserted_evidence_class_counts.items())),
            "inserted_docs": tuple(inserted_infos),
        }
        query_rows.append(row_payload)
        grouped[f"hop{hop}_{bucket}"].append(row_payload)

    summary_rows: list[Mapping[str, Any]] = []
    for hop in sorted({key[0] for key in hop_bucket_counts}):
        for bucket in ("gain", "loss", "changed_tie"):
            key = f"hop{hop}_{bucket}"
            items = grouped.get(key, [])
            if not items:
                continue
            summary = dict(_summarize_row_group(items))
            summary_rows.append({"hop": hop, "bucket": bucket, **summary})
    for bucket in ("gain", "loss", "changed_tie"):
        items = [row for row in query_rows if row.get("bucket") == bucket]
        if not items:
            continue
        summary = dict(_summarize_row_group(items))
        summary_rows.append({"hop": "all", "bucket": bucket, **summary})

    payload_out = {
        "metadata": {
            "retrieval_report": str(retrieval_report),
            "openie_path": str(openie_path),
            "row_count": len(rows),
            "max_queries": int(max_queries),
            "include_ties": bool(include_ties),
            "diagnostic_only": True,
            "gold_usage": "offline_bucket_classification_only",
            "chain_class_definition": {
                "chain_consumed": "inserted doc has a later chain-strong selected neighbor or >=2 chain-strong selected neighbors",
                "supported_leaf": "inserted doc has exactly one chain-strong selected neighbor and is not consumed later",
                "weak_or_topical_leaf": "inserted doc has selected-neighbor connectivity/fact witness but no chain-strong selected edge",
                "isolated_fill": "inserted doc has no internal edge to selected top5 in the reconstructed local graph",
            },
            "evidence_class_definition": {
                "evidence_consumed": "inserted doc has a later evidence-transition selected neighbor or >=2 evidence-transition selected neighbors",
                "evidence_leaf": "inserted doc has exactly one evidence-transition selected neighbor and is not consumed later",
                "weak_graph_connected": "inserted doc is graph-connected/fact-witnessed but lacks evidence-transition support inside selected top5",
                "isolated_fill": "inserted doc has no internal edge to selected top5 in the reconstructed local graph",
            },
        },
        "hop_bucket_counts": {
            f"hop{hop}_{bucket}": count
            for (hop, bucket), count in sorted(hop_bucket_counts.items())
        },
        "summary_rows": summary_rows,
        "query_rows": query_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "etv4_musique_chain_consistency_diagnostic.json"
    md_path = output_dir / "etv4_musique_chain_consistency_diagnostic.md"
    json_path.write_text(json.dumps(payload_out, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(payload_out), encoding="utf-8")
    return {"json_path": str(json_path), "md_path": str(md_path), **payload_out}


def _render_markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata", {}) or {}
    summary_rows = list(payload.get("summary_rows", []) or [])
    lines: list[str] = [
        "# ETv4 MuSiQue Chain-Consistency Diagnostic",
        "",
        f"Retrieval report: `{metadata.get('retrieval_report', '')}`",
        f"OpenIE: `{metadata.get('openie_path', '')}`",
        "",
        "This is diagnostic-only. Gold is used only to label gain/loss buckets.",
        "",
        "## Summary",
        "",
        _table(
            [
                {
                    "hop": row.get("hop"),
                    "bucket": row.get("bucket"),
                    "rows": row.get("rows"),
                    "inserted": row.get("inserted_docs"),
                    "consumed%": row.get("chain_consumed_pct"),
                    "evi_consumed%": row.get("evidence_consumed_pct"),
                    "evi_supported%": row.get("evidence_supported_pct"),
                    "weak_graph%": row.get("weak_graph_connected_pct"),
                    "nonconsumed%": row.get("non_consumed_pct"),
                    "weak/isolated%": row.get("weak_or_isolated_pct"),
                    "tail_drop%": row.get("tail_dropped_gold_pct"),
                    "drop_rank": row.get("mean_dropped_gold_dense_rank"),
                }
                for row in summary_rows
            ],
            (
                ("hop", "Hop"),
                ("bucket", "Bucket"),
                ("rows", "Rows"),
                ("inserted", "Inserted"),
                ("consumed%", "Consumed%"),
                ("evi_consumed%", "Evi-consumed%"),
                ("evi_supported%", "Evi-supported%"),
                ("weak_graph%", "Weak-graph%"),
                ("nonconsumed%", "Non-consumed%"),
                ("weak/isolated%", "Weak/isolated%"),
                ("tail_drop%", "Tail gold drop%"),
                ("drop_rank", "Mean drop rank"),
            ),
        ),
        "",
        "## Key Buckets",
        "",
    ]

    interesting_keys = [
        ("hop3_loss", "MuSiQue 3-hop losses"),
        ("hop3_gain", "MuSiQue 3-hop gains"),
        ("hop4_gain", "MuSiQue 4-hop gains"),
        ("hop2_gain", "MuSiQue 2-hop gains"),
    ]
    row_by_key = {
        f"hop{row.get('hop')}_{row.get('bucket')}": row
        for row in summary_rows
    }
    for key, title in interesting_keys:
        row = row_by_key.get(key)
        if not row:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append(
            _table(
                [
                    {
                        "rows": row.get("rows"),
                        "inserted": row.get("inserted_docs"),
                        "chain_consumed": row.get("chain_consumed_docs"),
                        "supported_leaf": row.get("supported_leaf_docs"),
                        "evidence_consumed": row.get("evidence_consumed_docs"),
                        "evidence_leaf": row.get("evidence_leaf_docs"),
                        "weak_graph": row.get("weak_graph_connected_docs"),
                        "weak_leaf": row.get("weak_topical_leaf_docs"),
                        "isolated": row.get("isolated_fill_docs"),
                        "tail_drop": row.get("tail_dropped_gold_pct"),
                    }
                ],
                (
                    ("rows", "Rows"),
                    ("inserted", "Inserted"),
                    ("chain_consumed", "Chain-consumed"),
                    ("supported_leaf", "Supported leaf"),
                    ("evidence_consumed", "Evi-consumed"),
                    ("evidence_leaf", "Evi-leaf"),
                    ("weak_graph", "Weak-graph"),
                    ("weak_leaf", "Weak/topical leaf"),
                    ("isolated", "Isolated"),
                    ("tail_drop", "Tail gold drop%"),
                ),
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Example Losses",
            "",
        ]
    )
    loss_examples = [
        row
        for row in payload.get("query_rows", []) or []
        if row.get("hop") == 3 and row.get("bucket") == "loss"
    ][:10]
    example_rows = []
    for row in loss_examples:
        class_counts = row.get("inserted_class_counts", {}) or {}
        evidence_class_counts = row.get("inserted_evidence_class_counts", {}) or {}
        example_rows.append(
            {
                "qid": row.get("query_index"),
                "hits": f"{row.get('dense_hits')}->{row.get('etv4_hits')}",
                "drop": ",".join(str(doc) for doc in row.get("dropped_gold_doc_indices", []) or []),
                "inserted": ",".join(str(doc) for doc in row.get("inserted_doc_indices", []) or []),
                "classes": ",".join(f"{key}:{value}" for key, value in sorted(class_counts.items())),
                "evi_classes": ",".join(
                    f"{key}:{value}" for key, value in sorted(evidence_class_counts.items())
                ),
                "drop_rank": ",".join(str(rank) for rank in row.get("dropped_gold_dense_ranks", []) or []),
            }
        )
    lines.append(
        _table(
            example_rows,
            (
                ("qid", "QID"),
                ("hits", "Hits"),
                ("drop", "Dropped gold"),
                ("inserted", "Inserted"),
                ("classes", "Inserted classes"),
                ("evi_classes", "Evidence classes"),
                ("drop_rank", "Dropped dense rank"),
            ),
        )
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, default=DEFAULT_RETRIEVAL_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all rows")
    parser.add_argument(
        "--include-ties",
        action="store_true",
        help="Also rebuild changed-tie rows; default only gain/loss rows.",
    )
    args = parser.parse_args(argv)
    result = run_diagnostic(
        retrieval_report=args.retrieval_report,
        output_dir=args.output_dir,
        max_queries=int(args.max_queries),
        include_ties=bool(args.include_ties),
    )
    print(f"Wrote {result['json_path']}")
    print(f"Wrote {result['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
