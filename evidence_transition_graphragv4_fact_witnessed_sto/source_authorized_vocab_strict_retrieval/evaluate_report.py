"""Evaluate source-authorized vocab-strict retrieval from per-query reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .candidate_expansion import build_source_text_candidate_expansion_index
from .certificate_graph import EvidenceNode, Triple
from .contract import CANONICAL_CLEAN_METHOD_NAME
from .evidence_completion import SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME
from .graph_index import SourceTextGraphIndex
from .graph_native_pipeline import source_authorized_graph_native_pipeline_retrieve
from .pipeline import (
    ACTIVE_TRUST_REGION_METHOD_NAME,
    source_authorized_vocab_strict_pipeline_retrieve,
    source_active_trust_region_pipeline_retrieve,
    source_certified_evidence_completion_pipeline_retrieve,
)
from .retriever import source_authorized_vocab_strict_retrieve


DEFAULT_CANDIDATE_FIELDS = (
    "candidate_cache_doc_indices",
    "retrieved_doc_indices_top200",
    "retrieved_doc_indices_top100",
    "retrieved_doc_indices_top20",
    "retrieved_doc_indices_top10",
    "candidate_doc_indices",
)


def unique_ints(values: Iterable[object]) -> List[int]:
    seen = set()
    output: List[int] = []
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


def recall_at_k(gold: Sequence[int], retrieved: Sequence[int], k: int) -> float:
    gold_set = set(unique_ints(gold))
    if not gold_set:
        return 0.0
    retrieved_set = set(unique_ints(retrieved[: max(int(k), 0)]))
    return len(gold_set & retrieved_set) / float(len(gold_set))


def all_gold_at_k(gold: Sequence[int], retrieved: Sequence[int], k: int) -> bool:
    gold_set = set(unique_ints(gold))
    if not gold_set:
        return False
    retrieved_set = set(unique_ints(retrieved[: max(int(k), 0)]))
    return gold_set.issubset(retrieved_set)


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def passage_title(text: object) -> str:
    raw = str(text or "")
    return raw.splitlines()[0].strip() if raw else ""


def openie_doc_to_node(doc_index: int, row: Mapping[str, Any]) -> EvidenceNode:
    triples: List[Triple] = []
    for raw_triple in row.get("extracted_triples", []) or []:
        if not isinstance(raw_triple, (list, tuple)) or len(raw_triple) != 3:
            continue
        triples.append((str(raw_triple[0]), str(raw_triple[1]), str(raw_triple[2])))
    text = str(row.get("passage") or "")
    return EvidenceNode(
        doc_index=int(doc_index),
        text=text,
        title=passage_title(text),
        triples=tuple(triples),
    )


def load_nodes(openie_path: Path) -> List[EvidenceNode]:
    payload = load_json(openie_path)
    docs = list(payload.get("docs", []) or [])
    return [openie_doc_to_node(index, row) for index, row in enumerate(docs)]


def load_candidate_cache_doc_indices(cache_path: Path) -> Dict[int, List[int]]:
    if not cache_path:
        return {}
    payload = load_json(cache_path)
    query_solutions = payload.get("query_solutions", []) if isinstance(payload, Mapping) else []
    output: Dict[int, List[int]] = {}
    for index, row in enumerate(query_solutions or []):
        if not isinstance(row, Mapping):
            continue
        output[int(index)] = unique_ints(row.get("doc_indices", []) or [])
    return output


def rows_for_variant(report: Mapping[str, Any], variant: str) -> List[Mapping[str, Any]]:
    variants = report.get("variants", {}) or {}
    rows = variants.get(variant)
    if rows is None:
        raise KeyError(f"variant not found in report: {variant}")
    if isinstance(rows, Mapping):
        ordered_items = sorted(rows.items(), key=lambda item: int(item[0]))
        return [row for _key, row in ordered_items if isinstance(row, Mapping)]
    return [row for row in rows if isinstance(row, Mapping)]


def value_at_field(row: Mapping[str, Any], field: str) -> Any:
    current: Any = row
    for part in str(field).split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def candidate_doc_indices_from_row(
    row: Mapping[str, Any],
    candidate_fields: Sequence[str],
    *,
    field_mode: str = "first",
) -> List[int]:
    if str(field_mode) == "union":
        docs: List[int] = []
        for field in candidate_fields:
            docs.extend(unique_ints(value_at_field(row, field) or []))
        return unique_ints(docs)
    for field in candidate_fields:
        docs = unique_ints(value_at_field(row, field) or [])
        if docs:
            return docs
    return []


def evaluate_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    nodes: Sequence[EvidenceNode],
    candidate_fields: Sequence[str] = DEFAULT_CANDIDATE_FIELDS,
    candidate_field_mode: str = "first",
    runner: str = "retriever",
    top_k: int = 5,
    candidate_pool_k: int = 200,
    max_hops: int | None = None,
    certificate_policy: str = "canonical_source_text",
) -> Dict[str, Any]:
    output_rows: List[Dict[str, Any]] = []
    r5_values: List[float] = []
    all_gold_values: List[float] = []
    certificate_counts: List[int] = []
    certified_counts: List[int] = []
    candidate_expansion_index = (
        build_source_text_candidate_expansion_index(nodes)
        if str(runner) in {"pipeline", "completion", "active_trust"}
        else None
    )
    graph_index = SourceTextGraphIndex.build(nodes) if str(runner) == "graph_native" else None

    for fallback_idx, row in enumerate(rows):
        query = str(row.get("question") or row.get("query") or "")
        query_index = int(row.get("query_index", fallback_idx))
        gold = unique_ints(row.get("gold_doc_indices", []) or [])
        candidates = candidate_doc_indices_from_row(
            row,
            candidate_fields,
            field_mode=candidate_field_mode,
        )
        if str(runner) == "graph_native":
            result = source_authorized_graph_native_pipeline_retrieve(
                query=query,
                nodes=nodes,
                candidate_doc_indices=candidates,
                top_k=top_k,
                candidate_pool_k=candidate_pool_k,
                certificate_policy=certificate_policy,
                graph_index=graph_index,
            )
            retrieval_trace = {
                **dict(result.selection.trace),
                "certificate_count": len(result.workspace.graph.certificates),
                "certificate_type_counts": dict(result.workspace.graph.certificate_type_counts()),
            }
            certified_doc_indices = result.certified_doc_indices
        elif str(runner) == "pipeline":
            result = source_authorized_vocab_strict_pipeline_retrieve(
                query=query,
                nodes=nodes,
                candidate_doc_indices=candidates,
                top_k=top_k,
                candidate_pool_k=candidate_pool_k,
                max_hops=max_hops,
                candidate_expansion_index=candidate_expansion_index,
                certificate_policy=certificate_policy,
            )
            retrieval_trace = dict(result.retrieval.trace)
            certified_doc_indices = result.retrieval.certified_doc_indices
        elif str(runner) == "completion":
            result = source_certified_evidence_completion_pipeline_retrieve(
                query=query,
                nodes=nodes,
                candidate_doc_indices=candidates,
                top_k=top_k,
                candidate_pool_k=candidate_pool_k,
                candidate_expansion_index=candidate_expansion_index,
                certificate_policy=certificate_policy,
            )
            retrieval_trace = dict(result.completion.trace)
            certified_doc_indices = result.completion.certified_doc_indices
        elif str(runner) == "active_trust":
            result = source_active_trust_region_pipeline_retrieve(
                query=query,
                nodes=nodes,
                candidate_doc_indices=candidates,
                top_k=top_k,
                candidate_pool_k=candidate_pool_k,
                candidate_expansion_index=candidate_expansion_index,
                certificate_policy=certificate_policy,
            )
            retrieval_trace = {
                **dict(result.repair.trace),
                "certificate_count": int(result.active_graph.trace.get("active_edge_count", 0)),
                "raw_certificate_count": int(result.active_graph.trace.get("raw_certificate_count", 0)),
            }
            certified_doc_indices = result.repair.inserted_doc_indices
        else:
            result = source_authorized_vocab_strict_retrieve(
                query=query,
                nodes=nodes,
                candidate_doc_indices=candidates,
                top_k=top_k,
                candidate_pool_k=candidate_pool_k,
                max_hops=max_hops,
                certificate_policy=certificate_policy,
            )
            retrieval_trace = dict(result.trace)
            certified_doc_indices = result.certified_doc_indices
        retrieved = list(result.doc_indices)
        r5 = recall_at_k(gold, retrieved, 5)
        all_gold5 = all_gold_at_k(gold, retrieved, 5)
        r5_values.append(float(r5))
        all_gold_values.append(1.0 if all_gold5 else 0.0)
        certificate_counts.append(int(retrieval_trace.get("certificate_count", 0)))
        certified_counts.append(len(certified_doc_indices))
        output_rows.append(
            {
                "query_index": query_index,
                "question": query,
                "gold_doc_indices": gold,
                "candidate_doc_indices": candidates[:candidate_pool_k],
                "retrieved_doc_indices_top5": retrieved[:5],
                "source_authorized_vocab_strict_recall_at5": round(float(r5), 6),
                "source_authorized_vocab_strict_all_gold_at5": bool(all_gold5),
            "route_trace": dict(result.trace),
            }
        )

    row_count = len(output_rows)
    denom = float(max(row_count, 1))
    return {
        "row_count": row_count,
        "metrics": {
            "r5": round(sum(r5_values) / denom, 6),
            "all_gold_at5": round(sum(all_gold_values) / denom, 6),
            "mean_certificate_count": round(sum(certificate_counts) / denom, 6),
            "mean_certified_doc_count_top5": round(sum(certified_counts) / denom, 6),
        },
        "rows": output_rows,
    }


def evaluate_report(args: argparse.Namespace) -> Dict[str, Any]:
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
    runner = str(getattr(args, "runner", "retriever"))
    certificate_policy = str(getattr(args, "certificate_policy", "canonical_source_text"))
    max_hops = None if int(args.max_hops) < 0 else max(int(args.max_hops), 0)
    dataset_result = evaluate_rows(
        rows=rows,
        nodes=nodes,
        candidate_fields=candidate_fields or DEFAULT_CANDIDATE_FIELDS,
        candidate_field_mode=str(args.candidate_field_mode),
        runner=runner,
        top_k=max(int(args.top_k), 1),
        candidate_pool_k=max(int(args.candidate_pool_k), 1),
        max_hops=max_hops,
        certificate_policy=certificate_policy,
    )
    if runner == "graph_native":
        method = f"{CANONICAL_CLEAN_METHOD_NAME}_graph_native"
    elif runner == "active_trust":
        method = ACTIVE_TRUST_REGION_METHOD_NAME
    elif runner == "completion":
        method = SOURCE_CERTIFIED_EVIDENCE_COMPLETION_METHOD_NAME
    elif runner == "pipeline":
        method = CANONICAL_CLEAN_METHOD_NAME
    else:
        method = "source_authorized_vocab_strict_retrieval"
    return {
        "method": method,
        "runner": runner,
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
            "max_hops": max_hops,
            "max_queries": max_queries,
            "runner": runner,
            "certificate_policy": certificate_policy,
        },
        **dataset_result,
    }


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    metrics = payload.get("metrics", {}) or {}
    lines = [
        "# Source-Authorized Vocab-Strict Retrieval",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| method | {payload.get('method', '')} |",
        f"| runner | {payload.get('runner', '')} |",
        f"| rows | {payload.get('row_count', 0)} |",
        f"| source variant | {payload.get('source_variant', '')} |",
        f"| candidate fields | {', '.join(payload.get('candidate_fields', []) or [])} |",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| R@5 | {float(metrics.get('r5', 0.0)):.4f} |",
        f"| all-gold@5 | {float(metrics.get('all_gold_at5', 0.0)):.4f} |",
        f"| mean certificates | {float(metrics.get('mean_certificate_count', 0.0)):.2f} |",
        f"| mean certified docs@5 | {float(metrics.get('mean_certified_doc_count_top5', 0.0)):.2f} |",
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
        "--runner",
        choices=(
            "retriever",
            "pipeline",
            "completion",
            "active_trust",
            "graph_native",
        ),
        default="retriever",
    )
    parser.add_argument(
        "--certificate-policy",
        choices=(
            "canonical_source_text",
            "canonical_transition",
            "canonical_query_transition",
            "canonical_fact_origin_transition",
            "legacy_role_hints",
        ),
        default="canonical_source_text",
    )
    parser.add_argument("--candidate-cache-path", default="")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-k", type=int, default=200)
    parser.add_argument("--max-hops", type=int, default=-1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = evaluate_report(args)
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
