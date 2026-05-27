#!/usr/bin/env python3
"""Compute Fact-Conditioned Rank Gain for EvLink diagnostics.

FCRG asks a narrow mechanism question: when a gold support document is
licensed by a source-grounded OpenIE fact transition from another gold support
document, does a retrieval method move that document up relative to a
query-only dense retriever?

This script does not call any model.  It reuses saved pool/PCEC artifacts and
the fixed OpenIE index produced by the EvLink runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.certificate_graph import (
    EvidenceCertificate,
    build_source_text_certificate_graph,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.evaluate_report import (
    load_nodes,
)


DATASETS: tuple[str, ...] = ("hotpotqa", "2wikimultihopqa", "musique")
DATASET_LABELS: Mapping[str, str] = {
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultiHopQA",
    "musique": "MuSiQue",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    pool_path: Path
    pcec_path: Path | None = None
    rank_source: str = "pool"


def read_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_ints(values: Iterable[Any]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def query_id(row: Mapping[str, Any], fallback: int) -> int:
    return int(row.get("query_idx", row.get("query_index", fallback)))


def rank_utility(rank: int) -> float:
    return 1.0 / math.log2(1.0 + float(rank))


def doc_rank(rank_map: Mapping[int, Mapping[int, int]], qid: int, doc_id: int, audit_depth: int) -> int:
    rank = int(rank_map.get(int(qid), {}).get(int(doc_id), int(audit_depth) + 1))
    return min(rank, int(audit_depth) + 1)


def first_occurrence_ranks(doc_ids: Sequence[Any], audit_depth: int) -> dict[int, int]:
    ranks: dict[int, int] = {}
    for position, raw_doc_id in enumerate(doc_ids, start=1):
        if position > int(audit_depth):
            break
        try:
            doc_id = int(raw_doc_id)
        except (TypeError, ValueError):
            continue
        ranks.setdefault(doc_id, position)
    return ranks


def rows_by_query(payload: Mapping[str, Any], key: str) -> dict[int, Mapping[str, Any]]:
    rows = list(payload.get(key) or [])
    return {query_id(row, idx): row for idx, row in enumerate(rows) if isinstance(row, Mapping)}


def load_pool_ranks(pool_path: Path, audit_depth: int) -> dict[int, dict[int, int]]:
    payload = read_json(pool_path)
    output: dict[int, dict[int, int]] = {}
    for idx, row in enumerate(payload.get("records") or []):
        if not isinstance(row, Mapping):
            continue
        qid = query_id(row, idx)
        output[qid] = first_occurrence_ranks(row.get("pool_doc_ids") or [], audit_depth)
    return output


def load_delivered_ranks(pool_path: Path, pcec_path: Path, audit_depth: int) -> dict[int, dict[int, int]]:
    pool_payload = read_json(pool_path)
    pcec_payload = read_json(pcec_path)
    pool_rows = rows_by_query(pool_payload, "records")
    pcec_rows = rows_by_query(pcec_payload, "rows")
    output: dict[int, dict[int, int]] = {}
    for qid, pool_row in pool_rows.items():
        final_prefix = []
        if qid in pcec_rows:
            final_prefix = unique_ints(pcec_rows[qid].get("retrieved_doc_indices_top5") or [])
        pool_ids = unique_ints(pool_row.get("pool_doc_ids") or [])
        combined = list(final_prefix)
        seen = set(combined)
        for doc_id in pool_ids:
            if doc_id in seen:
                continue
            combined.append(doc_id)
            seen.add(doc_id)
        output[qid] = first_occurrence_ranks(combined, audit_depth)
    return output


def full_pcec_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
        / "pcec/etv4/evals"
        / f"{dataset}_pcec_native_pool_prefix4_residual1_pool100_limit1000.json"
    )


def full_pool_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "run_logs/etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514"
        / "pools"
        / f"{dataset}_etv4_pool100_limit1000.json"
    )


def dense_pool_path(root: Path, dataset: str) -> Path:
    return root / "run_logs/dense_pool_exports_full1000_20260424" / f"{dataset}_dense_pool100.json"


def phrase_source_pool_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "run_logs/evidencelink_phrase_source_only_unified_qwen32b_gpt4omini_full1000_20260519"
        / "pools"
        / f"{dataset}_phrase_source_only_pool100_limit1000.json"
    )


def phrase_source_pcec_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "run_logs/evidencelink_phrase_source_only_unified_qwen32b_gpt4omini_full1000_20260519"
        / "pcec/phrase_source/evals"
        / f"{dataset}_phrase_source_only_pcec_native_pool_prefix4_residual1_pool100_limit1000.json"
    )


def control_pool_path(root: Path, dataset: str, policy: str) -> Path:
    return (
        root
        / "run_logs/evidencelink_doc_transition_controls_retrieval_full1000_20260520"
        / "pools"
        / policy
        / f"{dataset}_{policy}_pool100_limit1000.json"
    )


def control_pcec_path(root: Path, dataset: str, policy: str) -> Path:
    return (
        root
        / "run_logs/evidencelink_doc_transition_controls_retrieval_full1000_20260520"
        / "pcec"
        / policy
        / "evals"
        / f"{dataset}_{policy}_pcec_native_pool_prefix4_residual1_pool100_limit1000.json"
    )


def strict_dense_pool_path(root: Path, dataset: str) -> Path:
    policy = "edge_count_matched_dense_doc_knn"
    return (
        root
        / "run_logs/evidencelink_edge_count_matched_dense_control_full1000_20260520"
        / "pools"
        / policy
        / f"{dataset}_{policy}_pool100_limit1000.json"
    )


def strict_dense_pcec_path(root: Path, dataset: str) -> Path:
    policy = "edge_count_matched_dense_doc_knn"
    return (
        root
        / "run_logs/evidencelink_edge_count_matched_dense_control_full1000_20260520"
        / "pcec"
        / policy
        / "evals"
        / f"{dataset}_{policy}_pcec_native_pool_prefix4_residual1_pool100_limit1000.json"
    )


def hipporag_pool_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "run_logs/hipporag_qwen32b_valid_graph_top200_full1000_20260513_r2"
        / "pools/hipporag"
        / f"{dataset}_hipporag_qwen32b_valid_graph_pool200.json"
    )


def proprag_pool_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512"
        / "pools/proprag"
        / f"{dataset}_proprag_qwen32b_nothink_pool200.json"
    )


def method_specs(root: Path, dataset: str) -> list[MethodSpec]:
    specs = [
        MethodSpec("Dense query-only", dense_pool_path(root, dataset), None, "pool"),
        MethodSpec("EvLink pool", full_pool_path(root, dataset), None, "pool"),
        MethodSpec("EvLink delivered", full_pool_path(root, dataset), full_pcec_path(root, dataset), "delivered"),
        MethodSpec(
            "w/o evidence-linked transitions pool",
            phrase_source_pool_path(root, dataset),
            None,
            "pool",
        ),
        MethodSpec(
            "w/o evidence-linked transitions delivered",
            phrase_source_pool_path(root, dataset),
            phrase_source_pcec_path(root, dataset),
            "delivered",
        ),
        MethodSpec("Dense-doc KNN transitions pool", control_pool_path(root, dataset, "dense_doc_knn"), None, "pool"),
        MethodSpec(
            "Dense-doc KNN transitions delivered",
            control_pool_path(root, dataset, "dense_doc_knn"),
            control_pcec_path(root, dataset, "dense_doc_knn"),
            "delivered",
        ),
        MethodSpec(
            "Degree-matched shuffled transitions pool",
            control_pool_path(root, dataset, "degree_matched_shuffle"),
            None,
            "pool",
        ),
        MethodSpec(
            "Degree-matched shuffled transitions delivered",
            control_pool_path(root, dataset, "degree_matched_shuffle"),
            control_pcec_path(root, dataset, "degree_matched_shuffle"),
            "delivered",
        ),
        MethodSpec("Edge-count matched dense transitions pool", strict_dense_pool_path(root, dataset), None, "pool"),
        MethodSpec(
            "Edge-count matched dense transitions delivered",
            strict_dense_pool_path(root, dataset),
            strict_dense_pcec_path(root, dataset),
            "delivered",
        ),
        MethodSpec("HippoRAG2 pool", hipporag_pool_path(root, dataset), None, "pool"),
        MethodSpec("PropRAG pool", proprag_pool_path(root, dataset), None, "pool"),
    ]
    return specs


def gold_rows_for_dataset(root: Path, dataset: str) -> tuple[list[Mapping[str, Any]], Path]:
    report_path = full_pcec_path(root, dataset)
    payload = read_json(report_path)
    openie_path = Path(str(payload.get("openie_path") or ""))
    if not openie_path.exists():
        pool_payload = read_json(full_pool_path(root, dataset))
        openie_path = Path(str(pool_payload.get("openie_path") or ""))
    if not openie_path.exists():
        raise FileNotFoundError(f"OpenIE path not found for {dataset}: {openie_path}")
    return [row for row in payload.get("rows") or [] if isinstance(row, Mapping)], openie_path


def is_fact_certificate(certificate: EvidenceCertificate) -> bool:
    triple = tuple(certificate.triple or ())
    return len(triple) == 3 and any(str(part).strip() for part in triple)


def build_fact_edges(
    *,
    rows: Sequence[Mapping[str, Any]],
    openie_path: Path,
    certificate_policy: str,
    require_triple: bool,
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[int]], dict[str, Any]]:
    nodes = load_nodes(openie_path)
    fact_edges: dict[int, list[tuple[int, int]]] = {}
    gold_docs: dict[int, list[int]] = {}
    type_counts: dict[str, int] = {}
    total_certificates = 0
    total_kept_certificates = 0

    for fallback_idx, row in enumerate(rows):
        qid = query_id(row, fallback_idx)
        gold = unique_ints(row.get("gold_doc_indices") or [])
        gold_docs[qid] = gold
        if len(gold) < 2:
            fact_edges[qid] = []
            continue
        graph = build_source_text_certificate_graph(
            query=str(row.get("question") or row.get("query") or ""),
            nodes=nodes,
            candidate_doc_indices=gold,
            source_doc_indices=gold,
            certificate_policy=certificate_policy,
        )
        total_certificates += len(graph.certificates)
        edges: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()
        gold_set = set(gold)
        for certificate in graph.certificates:
            type_counts[str(certificate.certificate_type)] = (
                type_counts.get(str(certificate.certificate_type), 0) + 1
            )
            if require_triple and not is_fact_certificate(certificate):
                continue
            source = int(certificate.source_doc_index)
            target = int(certificate.target_doc_index)
            if source == target or source not in gold_set or target not in gold_set:
                continue
            edge = (source, target)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            edges.append(edge)
        total_kept_certificates += len(edges)
        fact_edges[qid] = edges

    metadata = {
        "openie_path": str(openie_path),
        "certificate_policy": certificate_policy,
        "require_triple": bool(require_triple),
        "certificate_type_counts_before_filter": type_counts,
        "total_certificates_before_filter": total_certificates,
        "total_unique_fact_edges_after_filter": total_kept_certificates,
    }
    return fact_edges, gold_docs, metadata


def compute_fcrg(
    *,
    query_ids: Sequence[int],
    gold_docs: Mapping[int, Sequence[int]],
    fact_edges: Mapping[int, Sequence[tuple[int, int]]],
    rank0: Mapping[int, Mapping[int, int]],
    rank_m: Mapping[int, Mapping[int, int]],
    audit_depth: int,
    min_baseline_rank: int = 1,
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    avg_delta_numerator = 0.0
    total_gold_docs = 0
    total_fact_conditioned_gold_docs = 0
    queries_with_fact_conditioned_gold = 0
    promoted = 0
    demoted = 0
    unchanged = 0
    baseline_at5 = 0
    method_at5 = 0
    baseline_ranks: list[int] = []
    method_ranks: list[int] = []

    for qid in query_ids:
        gold_set = {int(doc_id) for doc_id in gold_docs.get(int(qid), [])}
        total_gold_docs += len(gold_set)
        raw_fact_conditioned = {
            int(target)
            for source, target in fact_edges.get(int(qid), [])
            if int(source) in gold_set and int(target) in gold_set and int(source) != int(target)
        }
        fact_conditioned = {
            doc_id
            for doc_id in raw_fact_conditioned
            if doc_rank(rank0, int(qid), int(doc_id), audit_depth) >= int(min_baseline_rank)
        }
        if fact_conditioned:
            queries_with_fact_conditioned_gold += 1
        total_fact_conditioned_gold_docs += len(fact_conditioned)

        for doc_id in sorted(fact_conditioned):
            r0 = doc_rank(rank0, int(qid), int(doc_id), audit_depth)
            r_m = doc_rank(rank_m, int(qid), int(doc_id), audit_depth)
            u0 = rank_utility(r0)
            u_m = rank_utility(r_m)
            gain = u_m - u0
            numerator += gain
            denominator += 1.0 - u0
            avg_delta_numerator += gain
            baseline_ranks.append(r0)
            method_ranks.append(r_m)
            if r0 <= 5:
                baseline_at5 += 1
            if r_m <= 5:
                method_at5 += 1
            if gain > 1e-12:
                promoted += 1
            elif gain < -1e-12:
                demoted += 1
            else:
                unchanged += 1

    fact_count = total_fact_conditioned_gold_docs
    return {
        "FCRG": None if denominator == 0.0 else numerator / denominator,
        "avg_utility_delta": None if fact_count == 0 else avg_delta_numerator / float(fact_count),
        "fact_linked_support_coverage": (
            0.0 if total_gold_docs == 0 else fact_count / float(total_gold_docs)
        ),
        "num_queries_total": len(query_ids),
        "num_queries_with_B_f": queries_with_fact_conditioned_gold,
        "num_gold_docs_total": total_gold_docs,
        "num_fact_conditioned_gold_docs": fact_count,
        "numerator": numerator,
        "denominator": denominator,
        "promoted_count": promoted,
        "demoted_count": demoted,
        "unchanged_count": unchanged,
        "promoted_pct": 0.0 if fact_count == 0 else promoted / float(fact_count),
        "demoted_pct": 0.0 if fact_count == 0 else demoted / float(fact_count),
        "baseline_at5_pct": 0.0 if fact_count == 0 else baseline_at5 / float(fact_count),
        "method_at5_pct": 0.0 if fact_count == 0 else method_at5 / float(fact_count),
        "mean_baseline_rank": None if not baseline_ranks else sum(baseline_ranks) / len(baseline_ranks),
        "mean_method_rank": None if not method_ranks else sum(method_ranks) / len(method_ranks),
        "min_baseline_rank": int(min_baseline_rank),
    }


def load_method_ranks(spec: MethodSpec, audit_depth: int) -> dict[int, dict[int, int]] | None:
    if not spec.pool_path.exists():
        print(f"[skip] missing pool: {spec.pool_path}", file=sys.stderr)
        return None
    if spec.pcec_path is None:
        return load_pool_ranks(spec.pool_path, audit_depth)
    if not spec.pcec_path.exists():
        print(f"[skip] missing PCEC: {spec.pcec_path}", file=sys.stderr)
        return None
    return load_delivered_ranks(spec.pool_path, spec.pcec_path, audit_depth)


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.2f}"


def decimal(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numerator = sum(float(row.get("numerator") or 0.0) for row in rows)
    denominator = sum(float(row.get("denominator") or 0.0) for row in rows)
    fact_count = sum(int(row.get("num_fact_conditioned_gold_docs") or 0) for row in rows)
    gold_count = sum(int(row.get("num_gold_docs_total") or 0) for row in rows)
    avg_delta_sum = sum(
        float(row.get("avg_utility_delta") or 0.0)
        * int(row.get("num_fact_conditioned_gold_docs") or 0)
        for row in rows
    )
    return {
        "FCRG": None if denominator == 0.0 else numerator / denominator,
        "avg_utility_delta": None if fact_count == 0 else avg_delta_sum / float(fact_count),
        "fact_linked_support_coverage": 0.0 if gold_count == 0 else fact_count / float(gold_count),
        "num_queries_total": sum(int(row.get("num_queries_total") or 0) for row in rows),
        "num_queries_with_B_f": sum(int(row.get("num_queries_with_B_f") or 0) for row in rows),
        "num_gold_docs_total": gold_count,
        "num_fact_conditioned_gold_docs": fact_count,
        "numerator": numerator,
        "denominator": denominator,
        "promoted_count": sum(int(row.get("promoted_count") or 0) for row in rows),
        "demoted_count": sum(int(row.get("demoted_count") or 0) for row in rows),
        "unchanged_count": sum(int(row.get("unchanged_count") or 0) for row in rows),
        "promoted_pct": (
            0.0
            if fact_count == 0
            else sum(int(row.get("promoted_count") or 0) for row in rows) / float(fact_count)
        ),
        "demoted_pct": (
            0.0
            if fact_count == 0
            else sum(int(row.get("demoted_count") or 0) for row in rows) / float(fact_count)
        ),
        "baseline_at5_pct": (
            0.0
            if fact_count == 0
            else sum(float(row.get("baseline_at5_pct") or 0.0) * int(row.get("num_fact_conditioned_gold_docs") or 0) for row in rows)
            / float(fact_count)
        ),
        "method_at5_pct": (
            0.0
            if fact_count == 0
            else sum(float(row.get("method_at5_pct") or 0.0) * int(row.get("num_fact_conditioned_gold_docs") or 0) for row in rows)
            / float(fact_count)
        ),
    }


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], aggregate: Sequence[Mapping[str, Any]]) -> None:
    aggregate_columns = [
        "method",
        "FCRG",
        "avg_delta",
        "rho_f",
        "method_at5",
        "promoted",
        "demoted",
        "fact_docs",
    ]
    dataset_columns = [
        "method",
        "dataset",
        "FCRG",
        "avg_delta",
        "rho_f",
        "method_at5",
        "baseline_at5",
        "promoted",
        "demoted",
        "fact_docs",
    ]
    text = "\n\n".join(
        [
            "# EvLink FCRG Mechanism Diagnostic",
            (
                "FCRG measures whether a method promotes gold supporting documents whose "
                "relevance is licensed by a source-grounded OpenIE fact transition from "
                "another gold supporting document. Dense query-only is the reference "
                "ranker, so its FCRG is expected to be 0."
            ),
            "## Aggregate",
            markdown_table(aggregate, aggregate_columns),
            "## By Dataset",
            markdown_table(rows, dataset_columns),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def display_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "dataset": row["dataset"],
        "FCRG": decimal(row.get("FCRG")),
        "avg_delta": decimal(row.get("avg_utility_delta")),
        "rho_f": pct(row.get("fact_linked_support_coverage")),
        "method_at5": pct(row.get("method_at5_pct")),
        "baseline_at5": pct(row.get("baseline_at5_pct")),
        "promoted": pct(row.get("promoted_pct")),
        "demoted": pct(row.get("demoted_pct")),
        "fact_docs": int(row.get("num_fact_conditioned_gold_docs") or 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("run_logs/evidencelink_fcrg_20260520"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--audit-depth", type=int, default=100)
    parser.add_argument(
        "--min-baseline-rank",
        type=int,
        default=1,
        help=(
            "Only evaluate fact-conditioned gold documents whose dense query-only "
            "rank is at least this value. The default 1 is the original FCRG; "
            "6 audits the hard subset missing from dense top-5."
        ),
    )
    parser.add_argument("--certificate-policy", default="canonical_source_text")
    parser.add_argument(
        "--include-non-triple-certificates",
        action="store_true",
        help="Include title/source certificates without OpenIE triples. Off by default for FCRG.",
    )
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    output_root = args.output_root.expanduser()
    require_triple = not bool(args.include_non_triple_certificates)
    all_rows: list[dict[str, Any]] = []
    fact_edge_metadata: dict[str, Any] = {}

    for dataset in args.datasets:
        rows, openie_path = gold_rows_for_dataset(root, dataset)
        fact_edges, gold_docs, metadata = build_fact_edges(
            rows=rows,
            openie_path=openie_path,
            certificate_policy=str(args.certificate_policy),
            require_triple=require_triple,
        )
        fact_edge_metadata[dataset] = metadata
        query_ids = sorted(gold_docs)
        rank0 = load_pool_ranks(dense_pool_path(root, dataset), int(args.audit_depth))

        for spec in method_specs(root, dataset):
            ranks = load_method_ranks(spec, int(args.audit_depth))
            if ranks is None:
                continue
            result = compute_fcrg(
                query_ids=query_ids,
                gold_docs=gold_docs,
                fact_edges=fact_edges,
                rank0=rank0,
                rank_m=ranks,
                audit_depth=int(args.audit_depth),
                min_baseline_rank=int(args.min_baseline_rank),
            )
            all_rows.append(
                {
                    "method": spec.name,
                    "dataset": DATASET_LABELS.get(dataset, dataset),
                    "dataset_key": dataset,
                    "rank_source": spec.rank_source,
                    "pool_path": str(spec.pool_path),
                    "pcec_path": str(spec.pcec_path) if spec.pcec_path else "",
                    **result,
                }
            )

    rows_for_report = [display_row(row) for row in all_rows]
    rows_by_method: dict[str, list[Mapping[str, Any]]] = {}
    for row in all_rows:
        rows_by_method.setdefault(str(row["method"]), []).append(row)
    aggregate_rows_raw = []
    for method, method_rows in rows_by_method.items():
        aggregate_rows_raw.append({"method": method, **aggregate_rows(method_rows)})
    aggregate_for_report = [
        {
            "method": row["method"],
            "FCRG": decimal(row.get("FCRG")),
            "avg_delta": decimal(row.get("avg_utility_delta")),
            "rho_f": pct(row.get("fact_linked_support_coverage")),
            "method_at5": pct(row.get("method_at5_pct")),
            "promoted": pct(row.get("promoted_pct")),
            "demoted": pct(row.get("demoted_pct")),
            "fact_docs": int(row.get("num_fact_conditioned_gold_docs") or 0),
        }
        for row in aggregate_rows_raw
    ]

    payload = {
        "protocol": {
            "metric": "Fact-Conditioned Rank Gain",
            "rank_reference": "dense query-only pool@100",
            "audit_depth": int(args.audit_depth),
            "fact_edges": "gold-doc-to-gold-doc source-grounded certificate edges from fixed OpenIE index",
            "require_openie_triple_certificate": require_triple,
            "formula": "sum(u(r_M)-u(r_0)) / sum(1-u(r_0)), u(r)=1/log2(1+r)",
            "min_baseline_rank": int(args.min_baseline_rank),
        },
        "fact_edge_metadata": fact_edge_metadata,
        "rows": all_rows,
        "aggregate": aggregate_rows_raw,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "fcrg.json", payload)
    write_csv_rows(output_root / "fcrg_rows.csv", all_rows)
    write_csv_rows(output_root / "fcrg_aggregate.csv", aggregate_rows_raw)
    write_markdown(output_root / "fcrg.md", rows_for_report, aggregate_for_report)
    print(json.dumps({"output_root": str(output_root), "rows": len(all_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
