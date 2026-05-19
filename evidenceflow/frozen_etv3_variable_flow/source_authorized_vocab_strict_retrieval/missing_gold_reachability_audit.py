"""Classify where baseline-top-k missing gold supports are reachable.

This is a read-only audit.  It does not define a retriever and it may use gold
labels only to explain whether a candidate generator can expose missing support
documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .evaluate_report import (
    load_candidate_cache_doc_indices,
    load_json,
    rows_for_variant,
    unique_ints,
)


MISSING_GOLD_REACHABILITY_AUDIT_CONTRACT: Mapping[str, bool | str] = {
    "audit": "missing_gold_reachability_by_candidate_layer",
    "uses_gold_support_labels_for_audit_only": True,
    "is_retriever": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_proposition_support_evidence": False,
}


def audit_report(
    *,
    report: Mapping[str, Any],
    candidate_cache_doc_indices: Mapping[int, Sequence[int]],
    variant: str = "graph_native_source_authorized_evidence_set_search",
    baseline_variant: str = "hipporag_v2",
    k_values: Sequence[int] = (5, 10, 20),
) -> Dict[str, Any]:
    """Audit missing gold reachability against cache, path tail, and extended tail."""

    variant_rows = rows_for_variant(report, str(variant))
    baseline_rows = rows_for_variant(report, str(baseline_variant))
    baseline_by_query = {
        int(row.get("query_index", fallback_idx)): row
        for fallback_idx, row in enumerate(baseline_rows)
    }

    per_k: Dict[str, Any] = {}
    row_outputs: List[Dict[str, Any]] = []
    normalized_k_values = tuple(sorted({max(int(k), 1) for k in k_values}))
    for k in normalized_k_values:
        summary = _empty_summary()
        rows_for_k: List[Dict[str, Any]] = []
        for fallback_idx, row in enumerate(variant_rows):
            query_index = int(row.get("query_index", fallback_idx))
            baseline_row = baseline_by_query.get(query_index, row)
            gold = set(unique_ints(row.get("gold_doc_indices", []) or []))
            baseline_topk = set(
                unique_ints(
                    baseline_row.get(f"retrieved_doc_indices_top{k}", [])
                    or baseline_row.get("retrieved_doc_indices_top20", [])[:k]
                    or []
                )
            )
            missing_gold = sorted(gold - baseline_topk)
            if not missing_gold:
                continue

            candidate_cache = tuple(
                unique_ints(candidate_cache_doc_indices.get(query_index, ()) or ())
            )
            candidate_cache_set = set(candidate_cache)
            candidate_cache_rank = {
                int(doc_index): rank + 1
                for rank, doc_index in enumerate(candidate_cache)
            }
            route_trace = row.get("route_trace", {}) or {}
            path_tail = set(
                unique_ints(route_trace.get("source_authorized_tail_path_indices", []) or [])
            )
            extended_tail = set(
                unique_ints(route_trace.get("extended_tail_candidate_indices", []) or [])
            )
            any_source_tail = path_tail | extended_tail

            missing_in_cache = sorted(doc for doc in missing_gold if doc in candidate_cache_set)
            missing_not_in_cache = sorted(
                doc for doc in missing_gold if doc not in candidate_cache_set
            )
            missing_in_path_tail = sorted(doc for doc in missing_gold if doc in path_tail)
            missing_in_extended_tail = sorted(doc for doc in missing_gold if doc in extended_tail)
            missing_in_any_tail = sorted(doc for doc in missing_gold if doc in any_source_tail)
            cache_ranks = [
                int(candidate_cache_rank[doc])
                for doc in missing_gold
                if doc in candidate_cache_rank
            ]

            summary["queries_with_missing_gold"] += 1
            summary["missing_gold_docs"] += len(missing_gold)
            summary["missing_gold_in_candidate_cache"] += len(missing_in_cache)
            summary["missing_gold_not_in_candidate_cache"] += len(missing_not_in_cache)
            summary["missing_gold_in_path_tail"] += len(missing_in_path_tail)
            summary["missing_gold_in_extended_tail"] += len(missing_in_extended_tail)
            summary["missing_gold_in_any_source_tail"] += len(missing_in_any_tail)
            summary["queries_with_any_source_tail_gold"] += int(bool(missing_in_any_tail))
            summary["cache_rank_observations"].extend(cache_ranks)
            rows_for_k.append(
                {
                    "query_index": query_index,
                    "question": str(row.get("question") or ""),
                    "gold_doc_indices": tuple(sorted(gold)),
                    "missing_gold_doc_indices": tuple(missing_gold),
                    "missing_gold_in_candidate_cache_doc_indices": tuple(missing_in_cache),
                    "missing_gold_not_in_candidate_cache_doc_indices": tuple(
                        missing_not_in_cache
                    ),
                    "missing_gold_in_path_tail_doc_indices": tuple(missing_in_path_tail),
                    "missing_gold_in_extended_tail_doc_indices": tuple(
                        missing_in_extended_tail
                    ),
                    "missing_gold_in_any_source_tail_doc_indices": tuple(missing_in_any_tail),
                    "candidate_cache_ranks_for_missing_gold": tuple(cache_ranks),
                }
            )

        per_k[str(k)] = _finalize_summary(summary)
        row_outputs.append({"k": int(k), "rows": rows_for_k})

    return {
        **dict(MISSING_GOLD_REACHABILITY_AUDIT_CONTRACT),
        "dataset": str(report.get("dataset") or ""),
        "variant": str(variant),
        "baseline_variant": str(baseline_variant),
        "k_values": tuple(int(k) for k in normalized_k_values),
        "per_k": per_k,
        "rows_by_k": row_outputs,
    }


def _empty_summary() -> Dict[str, Any]:
    return {
        "queries_with_missing_gold": 0,
        "missing_gold_docs": 0,
        "missing_gold_in_candidate_cache": 0,
        "missing_gold_not_in_candidate_cache": 0,
        "missing_gold_in_path_tail": 0,
        "missing_gold_in_extended_tail": 0,
        "missing_gold_in_any_source_tail": 0,
        "queries_with_any_source_tail_gold": 0,
        "cache_rank_observations": [],
    }


def _finalize_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    missing = int(summary.get("missing_gold_docs", 0))
    queries = int(summary.get("queries_with_missing_gold", 0))
    ranks = [int(rank) for rank in summary.get("cache_rank_observations", []) or []]
    return {
        "queries_with_missing_gold": queries,
        "missing_gold_docs": missing,
        "missing_gold_in_candidate_cache": int(
            summary.get("missing_gold_in_candidate_cache", 0)
        ),
        "missing_gold_not_in_candidate_cache": int(
            summary.get("missing_gold_not_in_candidate_cache", 0)
        ),
        "missing_gold_in_path_tail": int(summary.get("missing_gold_in_path_tail", 0)),
        "missing_gold_in_extended_tail": int(
            summary.get("missing_gold_in_extended_tail", 0)
        ),
        "missing_gold_in_any_source_tail": int(
            summary.get("missing_gold_in_any_source_tail", 0)
        ),
        "queries_with_any_source_tail_gold": int(
            summary.get("queries_with_any_source_tail_gold", 0)
        ),
        "candidate_cache_missing_gold_coverage": _safe_div(
            summary.get("missing_gold_in_candidate_cache", 0),
            missing,
        ),
        "source_tail_missing_gold_coverage": _safe_div(
            summary.get("missing_gold_in_any_source_tail", 0),
            missing,
        ),
        "path_tail_missing_gold_coverage": _safe_div(
            summary.get("missing_gold_in_path_tail", 0),
            missing,
        ),
        "extended_tail_missing_gold_coverage": _safe_div(
            summary.get("missing_gold_in_extended_tail", 0),
            missing,
        ),
        "source_tail_query_recovery_rate": _safe_div(
            summary.get("queries_with_any_source_tail_gold", 0),
            queries,
        ),
        "candidate_cache_rank_min": min(ranks) if ranks else None,
        "candidate_cache_rank_median": round(float(median(ranks)), 6) if ranks else None,
        "candidate_cache_rank_max": max(ranks) if ranks else None,
    }


def _safe_div(numerator: object, denominator: object) -> float:
    denominator_int = int(denominator or 0)
    if denominator_int <= 0:
        return 0.0
    return round(float(numerator or 0) / float(denominator_int), 6)


def parse_k_values(text: str) -> Tuple[int, ...]:
    values = []
    for part in str(text or "").split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(max(int(stripped), 1))
    return tuple(values or (5, 10, 20))


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Missing Gold Reachability Audit",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| variant | {payload.get('variant', '')} |",
        f"| baseline variant | {payload.get('baseline_variant', '')} |",
        "",
        "| k | missing docs | queries | in candidate cache | not in cache | path tail | extended tail | any source tail | source-tail query recovery | cache rank median |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, summary in sorted(
        (payload.get("per_k", {}) or {}).items(),
        key=lambda item: int(item[0]),
    ):
        lines.append(
            "| {k} | {missing} | {queries} | {cache} ({cache_rate:.4f}) | "
            "{not_cache} | {path} | {extended} | {tail} ({tail_rate:.4f}) | "
            "{query_rate:.4f} | {median_rank} |".format(
                k=int(key),
                missing=int(summary.get("missing_gold_docs", 0)),
                queries=int(summary.get("queries_with_missing_gold", 0)),
                cache=int(summary.get("missing_gold_in_candidate_cache", 0)),
                cache_rate=float(summary.get("candidate_cache_missing_gold_coverage", 0.0)),
                not_cache=int(summary.get("missing_gold_not_in_candidate_cache", 0)),
                path=int(summary.get("missing_gold_in_path_tail", 0)),
                extended=int(summary.get("missing_gold_in_extended_tail", 0)),
                tail=int(summary.get("missing_gold_in_any_source_tail", 0)),
                tail_rate=float(summary.get("source_tail_missing_gold_coverage", 0.0)),
                query_rate=float(summary.get("source_tail_query_recovery_rate", 0.0)),
                median_rank=summary.get("candidate_cache_rank_median"),
            )
        )
    output_path.expanduser().write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify baseline-top-k missing gold reachability by candidate layer."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate-cache-path", required=True)
    parser.add_argument(
        "--variant",
        default="graph_native_source_authorized_evidence_set_search",
    )
    parser.add_argument("--baseline-variant", default="hipporag_v2")
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args(argv)

    report_path = Path(args.report).expanduser()
    cache_path = Path(args.candidate_cache_path).expanduser()
    payload = audit_report(
        report=load_json(report_path),
        candidate_cache_doc_indices=load_candidate_cache_doc_indices(cache_path),
        variant=str(args.variant),
        baseline_variant=str(args.baseline_variant),
        k_values=parse_k_values(str(args.k_values)),
    )
    output_json = Path(args.output_json).expanduser()
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        write_markdown(payload, Path(args.output_md).expanduser())
    print(
        json.dumps(
            {
                str(k): payload["per_k"][str(k)]
                for k in payload.get("k_values", ())
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
