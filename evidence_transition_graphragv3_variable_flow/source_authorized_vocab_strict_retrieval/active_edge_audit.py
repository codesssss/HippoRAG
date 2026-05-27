"""Audit raw vs query-active source-text certificate edges.

The audit is method-facing: it measures whether query conditioning improves the
action space before any trust-region repair is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .active_certificate_graph import ACTIVE_CERTIFICATE_TYPES, activate_certificate_edges
from .candidate_expansion import (
    SourceTextCandidateExpansionIndex,
    build_source_text_candidate_expansion_index,
    expand_source_text_candidates,
)
from .certificate_graph import (
    CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    EvidenceCertificate,
    EvidenceNode,
    build_source_text_certificate_graph,
    normalize_certificate_policy,
)
from .evaluate_report import (
    DEFAULT_CANDIDATE_FIELDS,
    candidate_doc_indices_from_row,
    load_candidate_cache_doc_indices,
    load_json,
    load_nodes,
    rows_for_variant,
    unique_ints,
)
from .frontier import build_source_text_frontier


ACTIVE_EDGE_AUDIT_CONTRACT: Mapping[str, bool | str] = {
    "audit": "raw_certificate_vs_query_active_edge_gap",
    "uses_query_conditioned_active_certificate_graph": True,
    "uses_role_vocabulary": False,
    "uses_proposition_support_evidence": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
}


def audit_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    nodes: Sequence[EvidenceNode],
    candidate_fields: Sequence[str] = DEFAULT_CANDIDATE_FIELDS,
    candidate_field_mode: str = "first",
    top_k: int = 5,
    candidate_pool_k: int = 200,
    certificate_policy: str = CANONICAL_SOURCE_TEXT_CERTIFICATE_POLICY,
    candidate_expansion_index: SourceTextCandidateExpansionIndex | None = None,
    entry_source_policy: str = "anchor_or_top1",
) -> Dict[str, Any]:
    """Audit raw and active certificate target quality for rows."""

    top_k = max(int(top_k), 1)
    candidate_pool_k = max(int(candidate_pool_k), top_k)
    certificate_policy = normalize_certificate_policy(certificate_policy)
    if candidate_expansion_index is None:
        candidate_expansion_index = build_source_text_candidate_expansion_index(nodes)

    output_rows: List[Dict[str, Any]] = []
    totals: Dict[str, int] = {
        "queries": 0,
        "queries_with_missing_gold_in_pool": 0,
        "queries_with_raw_gold_target": 0,
        "queries_with_active_gold_target": 0,
        "missing_gold_in_pool": 0,
        "raw_gold_target_docs": 0,
        "active_gold_target_docs": 0,
        "necessary_but_not_raw_docs": 0,
        "raw_legal_target_docs": 0,
        "active_legal_target_docs": 0,
        "raw_legal_but_unnecessary_docs": 0,
        "active_legal_but_unnecessary_docs": 0,
        "raw_edge_count": 0,
        "active_edge_count": 0,
        "suppressed_edge_count": 0,
        "suppressed_gold_target_docs": 0,
        "suppressed_lost_gold_target_docs": 0,
        "queries_with_raw_tail_oracle_gain": 0,
        "queries_with_active_tail_oracle_gain": 0,
    }
    by_certificate_type: Dict[str, Dict[str, int]] = {}
    by_activation_reason: Dict[str, Dict[str, int]] = {}

    for fallback_idx, row in enumerate(rows):
        query = str(row.get("question") or row.get("query") or "")
        query_index = int(row.get("query_index", fallback_idx))
        gold = set(unique_ints(row.get("gold_doc_indices", []) or []))
        candidates = candidate_doc_indices_from_row(
            row,
            candidate_fields,
            field_mode=candidate_field_mode,
        )
        initial_pool = tuple(unique_ints(candidates))[:candidate_pool_k]
        candidate_expansion = expand_source_text_candidates(
            query=query,
            nodes=nodes,
            candidate_doc_indices=initial_pool,
            top_k=top_k,
            expansion_index=candidate_expansion_index,
        )
        pool = tuple(candidate_expansion.candidate_doc_indices)
        initial_topk = tuple(pool[:top_k])
        graph = build_source_text_certificate_graph(
            query=query,
            nodes=nodes,
            candidate_doc_indices=pool,
            certificate_policy=certificate_policy,
        )
        frontier = build_source_text_frontier(
            query=query,
            nodes=nodes,
            candidate_doc_indices=pool,
            graph=graph,
            top_k=top_k,
        )
        entry_doc_indices = _entry_doc_indices(
            policy=entry_source_policy,
            pool=pool,
            frontier=frontier,
        )
        active_graph = activate_certificate_edges(
            query=query,
            nodes=nodes,
            certificate_graph=graph,
            candidate_doc_indices=pool,
            entry_doc_indices=entry_doc_indices,
            query_mentioned_doc_indices=frontier.query_mentioned_doc_indices,
        )

        raw_edges = tuple(
            certificate
            for certificate in graph.certificates
            if int(certificate.source_doc_index) in set(entry_doc_indices)
            and str(certificate.certificate_type) in ACTIVE_CERTIFICATE_TYPES
        )
        active_edges = tuple(edge.certificate for edge in active_graph.active_edges)
        suppressed_edges = tuple(edge.certificate for edge in active_graph.suppressed_edges)

        raw_targets = _target_docs(raw_edges)
        active_targets = _target_docs(active_edges)
        suppressed_targets = _target_docs(suppressed_edges)
        missing_gold_in_pool = set(pool) & gold - set(initial_topk)
        raw_gold_targets = raw_targets & missing_gold_in_pool
        active_gold_targets = active_targets & missing_gold_in_pool
        necessary_but_not_raw = missing_gold_in_pool - raw_targets
        raw_unnecessary = raw_targets - gold
        active_unnecessary = active_targets - gold
        suppressed_gold = suppressed_targets & missing_gold_in_pool
        suppressed_lost_gold = suppressed_gold - active_targets
        protected_doc_indices = set(unique_ints([*(pool[:1]), *frontier.query_mentioned_doc_indices]))
        raw_tail_oracle_gain = _has_tail_repair_oracle_gain(
            edges=raw_edges,
            initial_topk=initial_topk,
            missing_gold_in_pool=missing_gold_in_pool,
            gold=gold,
            protected_doc_indices=protected_doc_indices,
        )
        active_tail_oracle_gain = _has_tail_repair_oracle_gain(
            edges=active_edges,
            initial_topk=initial_topk,
            missing_gold_in_pool=missing_gold_in_pool,
            gold=gold,
            protected_doc_indices=protected_doc_indices,
        )

        totals["queries"] += 1
        totals["queries_with_missing_gold_in_pool"] += int(bool(missing_gold_in_pool))
        totals["queries_with_raw_gold_target"] += int(bool(raw_gold_targets))
        totals["queries_with_active_gold_target"] += int(bool(active_gold_targets))
        totals["missing_gold_in_pool"] += len(missing_gold_in_pool)
        totals["raw_gold_target_docs"] += len(raw_gold_targets)
        totals["active_gold_target_docs"] += len(active_gold_targets)
        totals["necessary_but_not_raw_docs"] += len(necessary_but_not_raw)
        totals["raw_legal_target_docs"] += len(raw_targets)
        totals["active_legal_target_docs"] += len(active_targets)
        totals["raw_legal_but_unnecessary_docs"] += len(raw_unnecessary)
        totals["active_legal_but_unnecessary_docs"] += len(active_unnecessary)
        totals["raw_edge_count"] += len(raw_edges)
        totals["active_edge_count"] += len(active_edges)
        totals["suppressed_edge_count"] += len(suppressed_edges)
        totals["suppressed_gold_target_docs"] += len(suppressed_gold)
        totals["suppressed_lost_gold_target_docs"] += len(suppressed_lost_gold)
        totals["queries_with_raw_tail_oracle_gain"] += int(raw_tail_oracle_gain)
        totals["queries_with_active_tail_oracle_gain"] += int(active_tail_oracle_gain)
        _accumulate_by_type(by_certificate_type, raw_edges, active_edges, missing_gold_in_pool, gold)
        _accumulate_by_activation_reason(
            by_activation_reason,
            active_graph.active_edges,
            missing_gold_in_pool,
            gold,
        )

        output_rows.append(
            {
                "query_index": query_index,
                "question": query,
                "gold_doc_indices": sorted(gold),
                "initial_topk_doc_indices": initial_topk,
                "entry_doc_indices": entry_doc_indices,
                "candidate_doc_count": len(pool),
                "missing_gold_in_pool_doc_indices": tuple(sorted(missing_gold_in_pool)),
                "raw_gold_target_doc_indices": tuple(sorted(raw_gold_targets)),
                "active_gold_target_doc_indices": tuple(sorted(active_gold_targets)),
                "necessary_but_not_raw_doc_indices": tuple(sorted(necessary_but_not_raw)),
                "suppressed_gold_target_doc_indices": tuple(sorted(suppressed_gold)),
                "suppressed_lost_gold_target_doc_indices": tuple(sorted(suppressed_lost_gold)),
                "raw_legal_target_count": len(raw_targets),
                "active_legal_target_count": len(active_targets),
                "raw_legal_but_unnecessary_count": len(raw_unnecessary),
                "active_legal_but_unnecessary_count": len(active_unnecessary),
                "raw_tail_oracle_gain": bool(raw_tail_oracle_gain),
                "active_tail_oracle_gain": bool(active_tail_oracle_gain),
                "raw_edge_count": len(raw_edges),
                "active_edge_count": len(active_edges),
                "suppressed_edge_count": len(suppressed_edges),
                "active_edge_preview": _edge_preview(active_edges),
                "active_graph_trace": dict(active_graph.trace),
            }
        )

    metrics = _metrics_from_totals(totals)
    return {
        **dict(ACTIVE_EDGE_AUDIT_CONTRACT),
        "row_count": len(output_rows),
        "metrics": metrics,
        "totals": totals,
        "by_certificate_type": {
            certificate_type: _metrics_from_totals(counts)
            for certificate_type, counts in sorted(by_certificate_type.items())
        },
        "by_activation_reason": {
            reason: _activation_reason_metrics(counts)
            for reason, counts in sorted(by_activation_reason.items())
        },
        "entry_source_policy": str(entry_source_policy),
        "rows": output_rows,
    }


def _entry_doc_indices(
    *,
    policy: str,
    pool: Sequence[int],
    frontier,
) -> Tuple[int, ...]:
    normalized = str(policy or "anchor_or_top1").strip().lower()
    if normalized == "initial_topk":
        return tuple(
            unique_ints(
                [
                    *frontier.initial_doc_indices,
                    *frontier.query_mentioned_doc_indices,
                ]
            )
        )
    if normalized == "frontier":
        return tuple(frontier.frontier_doc_indices)
    return tuple(unique_ints(frontier.query_mentioned_doc_indices or pool[:1]))


def _target_docs(edges: Sequence[EvidenceCertificate]) -> set[int]:
    return {int(edge.target_doc_index) for edge in edges}


def _has_tail_repair_oracle_gain(
    *,
    edges: Sequence[EvidenceCertificate],
    initial_topk: Sequence[int],
    missing_gold_in_pool: set[int],
    gold: set[int],
    protected_doc_indices: set[int],
) -> bool:
    if not initial_topk or not missing_gold_in_pool:
        return False
    victim = int(initial_topk[-1])
    if victim in gold or victim in protected_doc_indices:
        return False
    for edge in edges:
        source = int(edge.source_doc_index)
        target = int(edge.target_doc_index)
        if target not in missing_gold_in_pool:
            continue
        if source not in {int(doc_index) for doc_index in initial_topk}:
            continue
        if victim == source:
            continue
        return True
    return False


def _accumulate_by_type(
    output: Dict[str, Dict[str, int]],
    raw_edges: Sequence[EvidenceCertificate],
    active_edges: Sequence[EvidenceCertificate],
    missing_gold_in_pool: set[int],
    gold: set[int],
) -> None:
    for certificate_type in sorted(
        {
            *(str(edge.certificate_type) for edge in raw_edges),
            *(str(edge.certificate_type) for edge in active_edges),
        }
    ):
        counts = output.setdefault(
            certificate_type,
            {
                "raw_legal_target_docs": 0,
                "active_legal_target_docs": 0,
                "raw_gold_target_docs": 0,
                "active_gold_target_docs": 0,
                "raw_legal_but_unnecessary_docs": 0,
                "active_legal_but_unnecessary_docs": 0,
            },
        )
        raw_targets = {
            int(edge.target_doc_index)
            for edge in raw_edges
            if str(edge.certificate_type) == certificate_type
        }
        active_targets = {
            int(edge.target_doc_index)
            for edge in active_edges
            if str(edge.certificate_type) == certificate_type
        }
        counts["raw_legal_target_docs"] += len(raw_targets)
        counts["active_legal_target_docs"] += len(active_targets)
        counts["raw_gold_target_docs"] += len(raw_targets & missing_gold_in_pool)
        counts["active_gold_target_docs"] += len(active_targets & missing_gold_in_pool)
        counts["raw_legal_but_unnecessary_docs"] += len(raw_targets - gold)
        counts["active_legal_but_unnecessary_docs"] += len(active_targets - gold)


def _accumulate_by_activation_reason(
    output: Dict[str, Dict[str, int]],
    active_edges,
    missing_gold_in_pool: set[int],
    gold: set[int],
) -> None:
    reasons = sorted(
        {
            str(reason)
            for edge in active_edges
            for reason in tuple(edge.activation_reasons)
        }
    )
    for reason in reasons:
        counts = output.setdefault(
            reason,
            {
                "active_edge_count": 0,
                "active_legal_target_docs": 0,
                "active_gold_target_docs": 0,
                "active_legal_but_unnecessary_docs": 0,
            },
        )
        reason_edges = [
            edge
            for edge in active_edges
            if reason in set(str(item) for item in tuple(edge.activation_reasons))
        ]
        reason_targets = {int(edge.certificate.target_doc_index) for edge in reason_edges}
        counts["active_edge_count"] += len(reason_edges)
        counts["active_legal_target_docs"] += len(reason_targets)
        counts["active_gold_target_docs"] += len(reason_targets & missing_gold_in_pool)
        counts["active_legal_but_unnecessary_docs"] += len(reason_targets - gold)


def _activation_reason_metrics(counts: Mapping[str, int]) -> Dict[str, float]:
    return {
        "active_edge_count": float(counts.get("active_edge_count", 0)),
        "active_legal_target_docs": float(counts.get("active_legal_target_docs", 0)),
        "active_gold_target_docs": float(counts.get("active_gold_target_docs", 0)),
        "active_legal_precision": _safe_div(
            counts.get("active_gold_target_docs", 0),
            counts.get("active_legal_target_docs", 0),
        ),
        "active_legal_but_unnecessary_rate": _safe_div(
            counts.get("active_legal_but_unnecessary_docs", 0),
            counts.get("active_legal_target_docs", 0),
        ),
    }


def _metrics_from_totals(totals: Mapping[str, int]) -> Dict[str, float]:
    return {
        "raw_legal_precision": _safe_div(
            totals.get("raw_gold_target_docs", 0),
            totals.get("raw_legal_target_docs", 0),
        ),
        "active_legal_precision": _safe_div(
            totals.get("active_gold_target_docs", 0),
            totals.get("active_legal_target_docs", 0),
        ),
        "active_gold_recall_over_raw_gold": _safe_div(
            totals.get("active_gold_target_docs", 0),
            totals.get("raw_gold_target_docs", 0),
        ),
        "raw_missing_gold_coverage": _safe_div(
            totals.get("raw_gold_target_docs", 0),
            totals.get("missing_gold_in_pool", 0),
        ),
        "active_missing_gold_coverage": _safe_div(
            totals.get("active_gold_target_docs", 0),
            totals.get("missing_gold_in_pool", 0),
        ),
        "necessary_but_not_raw_rate": _safe_div(
            totals.get("necessary_but_not_raw_docs", 0),
            totals.get("missing_gold_in_pool", 0),
        ),
        "raw_legal_but_unnecessary_rate": _safe_div(
            totals.get("raw_legal_but_unnecessary_docs", 0),
            totals.get("raw_legal_target_docs", 0),
        ),
        "active_legal_but_unnecessary_rate": _safe_div(
            totals.get("active_legal_but_unnecessary_docs", 0),
            totals.get("active_legal_target_docs", 0),
        ),
        "query_raw_gold_target_rate": _safe_div(
            totals.get("queries_with_raw_gold_target", 0),
            totals.get("queries_with_missing_gold_in_pool", 0),
        ),
        "query_active_gold_target_rate": _safe_div(
            totals.get("queries_with_active_gold_target", 0),
            totals.get("queries_with_missing_gold_in_pool", 0),
        ),
        "mean_raw_edge_count": _safe_div(
            totals.get("raw_edge_count", 0),
            totals.get("queries", 0),
        ),
        "mean_active_edge_count": _safe_div(
            totals.get("active_edge_count", 0),
            totals.get("queries", 0),
        ),
        "suppressed_gold_target_rate": _safe_div(
            totals.get("suppressed_gold_target_docs", 0),
            totals.get("raw_gold_target_docs", 0),
        ),
        "suppressed_lost_gold_target_rate": _safe_div(
            totals.get("suppressed_lost_gold_target_docs", 0),
            totals.get("raw_gold_target_docs", 0),
        ),
        "raw_tail_oracle_query_rate": _safe_div(
            totals.get("queries_with_raw_tail_oracle_gain", 0),
            totals.get("queries_with_missing_gold_in_pool", 0),
        ),
        "active_tail_oracle_query_rate": _safe_div(
            totals.get("queries_with_active_tail_oracle_gain", 0),
            totals.get("queries_with_missing_gold_in_pool", 0),
        ),
    }


def _safe_div(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _edge_preview(edges: Sequence[EvidenceCertificate], limit: int = 8) -> List[Mapping[str, object]]:
    return [
        {
            "source_doc_index": int(edge.source_doc_index),
            "target_doc_index": int(edge.target_doc_index),
            "certificate_type": str(edge.certificate_type),
            "endpoint": str(edge.endpoint),
            "triple": list(edge.triple or ()),
        }
        for edge in edges[: max(int(limit), 0)]
    ]


def audit_report(args: argparse.Namespace) -> Dict[str, Any]:
    report_path = Path(args.report).expanduser().resolve()
    report = load_json(report_path)
    openie_path = Path(args.openie_path or report.get("analysis_openie_path") or "").expanduser()
    if not openie_path:
        raise ValueError("--openie-path is required when report has no analysis_openie_path")
    if not openie_path.is_absolute():
        openie_path = (report_path.parent / openie_path).resolve()
    nodes = load_nodes(openie_path)
    rows = rows_for_variant(report, str(args.variant))
    candidate_cache_path_text = str(
        args.candidate_cache_path
        or (report.get("config", {}) or {}).get("baseline_retrieval_cache_path")
        or ""
    )
    candidate_cache_path = Path(candidate_cache_path_text).expanduser()
    candidate_cache_docs = (
        load_candidate_cache_doc_indices(candidate_cache_path)
        if candidate_cache_path_text
        else {}
    )
    if candidate_cache_docs:
        enriched_rows: List[Mapping[str, Any]] = []
        for fallback_idx, row in enumerate(rows):
            query_index = int(row.get("query_index", fallback_idx))
            copied = dict(row)
            copied["candidate_cache_doc_indices"] = candidate_cache_docs.get(query_index, [])
            enriched_rows.append(copied)
        rows = enriched_rows
    max_queries = max(int(args.max_queries), 0)
    if max_queries:
        rows = rows[:max_queries]
    candidate_fields = tuple(
        field.strip()
        for field in str(args.candidate_fields).split(",")
        if field.strip()
    )
    payload = audit_rows(
        rows=rows,
        nodes=nodes,
        candidate_fields=candidate_fields or DEFAULT_CANDIDATE_FIELDS,
        candidate_field_mode=str(args.candidate_field_mode),
        top_k=max(int(args.top_k), 1),
        candidate_pool_k=max(int(args.candidate_pool_k), 1),
        certificate_policy=str(args.certificate_policy),
        entry_source_policy=str(args.entry_source_policy),
    )
    return {
        "dataset": str(report.get("dataset") or args.dataset or ""),
        "input_report": str(report_path),
        "openie_path": str(openie_path),
        "source_variant": str(args.variant),
        "candidate_fields": list(candidate_fields or DEFAULT_CANDIDATE_FIELDS),
        "config": {
            "top_k": max(int(args.top_k), 1),
            "candidate_pool_k": max(int(args.candidate_pool_k), 1),
            "candidate_field_mode": str(args.candidate_field_mode),
            "candidate_cache_path": str(candidate_cache_path) if candidate_cache_path_text else "",
            "max_queries": max_queries,
            "certificate_policy": str(args.certificate_policy),
            "entry_source_policy": str(args.entry_source_policy),
        },
        **payload,
    }


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    metrics = payload.get("metrics", {}) or {}
    lines = [
        "# Active Certificate Edge Audit",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| rows | {payload.get('row_count', 0)} |",
        f"| source variant | {payload.get('source_variant', '')} |",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| raw legal precision | {float(metrics.get('raw_legal_precision', 0.0)):.4f} |",
        f"| active legal precision | {float(metrics.get('active_legal_precision', 0.0)):.4f} |",
        f"| active gold recall over raw gold | {float(metrics.get('active_gold_recall_over_raw_gold', 0.0)):.4f} |",
        f"| raw missing-gold coverage | {float(metrics.get('raw_missing_gold_coverage', 0.0)):.4f} |",
        f"| active missing-gold coverage | {float(metrics.get('active_missing_gold_coverage', 0.0)):.4f} |",
        f"| necessary-but-not-raw rate | {float(metrics.get('necessary_but_not_raw_rate', 0.0)):.4f} |",
        f"| raw legal-but-unnecessary rate | {float(metrics.get('raw_legal_but_unnecessary_rate', 0.0)):.4f} |",
        f"| active legal-but-unnecessary rate | {float(metrics.get('active_legal_but_unnecessary_rate', 0.0)):.4f} |",
        f"| mean raw edge count | {float(metrics.get('mean_raw_edge_count', 0.0)):.2f} |",
        f"| mean active edge count | {float(metrics.get('mean_active_edge_count', 0.0)):.2f} |",
        f"| suppressed lost gold target rate | {float(metrics.get('suppressed_lost_gold_target_rate', 0.0)):.4f} |",
        f"| raw tail oracle query rate | {float(metrics.get('raw_tail_oracle_query_rate', 0.0)):.4f} |",
        f"| active tail oracle query rate | {float(metrics.get('active_tail_oracle_query_rate', 0.0)):.4f} |",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--openie-path", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--variant", default="hipporag_v2")
    parser.add_argument("--candidate-fields", default=",".join(DEFAULT_CANDIDATE_FIELDS))
    parser.add_argument("--candidate-field-mode", choices=("first", "union"), default="first")
    parser.add_argument(
        "--certificate-policy",
        choices=("canonical_source_text", "legacy_role_hints"),
        default="canonical_source_text",
    )
    parser.add_argument("--candidate-cache-path", default="")
    parser.add_argument(
        "--entry-source-policy",
        choices=("anchor_or_top1", "initial_topk", "frontier"),
        default="anchor_or_top1",
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-k", type=int, default=200)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = audit_report(args)
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        write_markdown(payload, Path(args.output_md).expanduser())
    print(json.dumps({payload["dataset"]: payload["metrics"]}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
