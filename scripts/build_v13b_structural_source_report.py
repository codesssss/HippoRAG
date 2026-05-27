#!/usr/bin/env python3
"""Build a structural v13b source report from a dense pool plus OpenIE/STO facts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from build_query_obligation_units import build_query_obligation_units, match_query_obligations_to_sto_facts
from build_source_title_openie_substrate import build_units_for_doc
from query_obligation_typing import lower_query_triples_for_typed_program
from v13b_graph_normalization import bound_endpoint_keys, fact_role_keys, safe_int


DEFAULT_DATASETS = "2wikimultihopqa,musique,hotpotqa"
DEFAULT_POOL_TEMPLATE = "run_logs/proprag_pool_exports_full1000_20260424/{dataset}_pool100.json"
DEFAULT_OPENIE_TEMPLATE = "outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json"
DEFAULT_CACHE_TEMPLATE = "run_logs/v13b_proprag_pool100_full1000_fulloblg_20260430/{dataset}_query_obligation_cache_full1000.json"
DEFAULT_MAX_ENDPOINT_DOC_DEGREE = 30


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_passage(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique_ints(values: Iterable[Any], *, limit: int = 0) -> List[int]:
    result: List[int] = []
    seen: Set[int] = set()
    for value in values:
        item = safe_int(value)
        if item < 0 or item in seen:
            continue
        result.append(item)
        seen.add(item)
        if limit > 0 and len(result) >= limit:
            break
    return result


def openie_docs_from_payload(payload: Any) -> List[Mapping[str, Any]]:
    docs = payload.get("docs", []) if isinstance(payload, Mapping) else payload
    return [doc for doc in docs if isinstance(doc, Mapping)]


def title_for_doc(doc: Mapping[str, Any]) -> str:
    passage = str(doc.get("passage") or "")
    return passage.split("\n", 1)[0].strip() if passage else ""


def build_title_index(openie_docs: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for doc_idx, doc in enumerate(openie_docs):
        title = normalize_title(title_for_doc(doc))
        if title:
            index.setdefault(title, []).append(int(doc_idx))
    return index


def build_passage_index(openie_docs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for doc_idx, doc in enumerate(openie_docs):
        passage = normalize_passage(doc.get("passage"))
        if passage and passage not in index:
            index[passage] = int(doc_idx)
    return index


def records_from_pool_payload(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("records", "rows", "queries"):
            rows = payload.get(key)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                return [row for row in rows if isinstance(row, Mapping)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return [row for row in payload if isinstance(row, Mapping)]
    return []


def gold_doc_indices_for_record(
    *,
    record: Mapping[str, Any],
    passage_index: Mapping[str, int],
    title_index: Mapping[str, Sequence[int]],
) -> List[int]:
    docs: List[int] = []
    for value in record.get("gold_doc_indices", []) or []:
        docs.append(safe_int(value))
    if docs:
        return unique_ints(docs)

    for passage in record.get("gold_docs", []) or record.get("gold_passages", []) or []:
        doc_idx = passage_index.get(normalize_passage(passage))
        if doc_idx is not None:
            docs.append(doc_idx)
    if docs:
        return unique_ints(docs)

    for passage in record.get("gold_docs", []) or record.get("gold_passages", []) or []:
        title = str(passage or "").split("\n", 1)[0].strip()
        docs.extend(title_index.get(normalize_title(title), []) or [])
    if docs:
        return unique_ints(docs)

    for title in record.get("gold_titles", []) or []:
        docs.extend(title_index.get(normalize_title(title), []) or [])
    return unique_ints(docs)


def pool_doc_indices(record: Mapping[str, Any], *, dense_limit: int) -> List[int]:
    for key in ("pool_doc_ids", "candidate_doc_indices", "doc_indices", "retrieved_doc_indices"):
        values = record.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            return unique_ints(values, limit=dense_limit)
    return []


def cache_rows_by_query(cache_payload: Any, dataset: str) -> Dict[int, Mapping[str, Any]]:
    rows: Sequence[Any] = []
    if isinstance(cache_payload, Mapping) and isinstance(cache_payload.get("datasets"), Sequence):
        for payload in cache_payload.get("datasets", []) or []:
            if isinstance(payload, Mapping) and str(payload.get("dataset") or "") == str(dataset):
                rows = payload.get("rows", []) or []
                break
    elif isinstance(cache_payload, Mapping) and isinstance(cache_payload.get("rows"), Sequence):
        rows = cache_payload.get("rows", []) or []
    return {
        safe_int(row.get("query_index"), ordinal): row
        for ordinal, row in enumerate(rows)
        if isinstance(row, Mapping)
    }


def build_doc_units(openie_docs: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    return {
        doc_index: build_units_for_doc(doc, doc_index=doc_index, include_source_spans=False)
        for doc_index, doc in enumerate(openie_docs)
    }


def build_endpoint_to_docs(doc_units: Mapping[int, Sequence[Mapping[str, Any]]]) -> Dict[str, List[int]]:
    endpoint_to_docs: Dict[str, Set[int]] = defaultdict(set)
    for doc_index, units in doc_units.items():
        for unit in units:
            if str(unit.get("unit_type") or "") != "openie_fact":
                continue
            row = fact_role_keys(unit)
            for key in set(row.get("subject_keys", set())) | set(row.get("object_keys", set())):
                if key:
                    endpoint_to_docs[str(key)].add(int(doc_index))
    return {key: sorted(values) for key, values in endpoint_to_docs.items()}


def add_provenance(provenance: Dict[int, Set[str]], docs: Iterable[int], label: str) -> None:
    for doc in unique_ints(docs):
        provenance.setdefault(doc, set()).add(label)


def low_degree_endpoint_docs(
    *,
    endpoint_to_docs: Mapping[str, Sequence[int]],
    endpoint_key: str,
    max_endpoint_doc_degree: int,
) -> List[int]:
    """Return docs for an endpoint only when it is not a corpus-level hub."""

    docs = unique_ints(endpoint_to_docs.get(str(endpoint_key), []) or [])
    if max_endpoint_doc_degree > 0 and len(docs) > max_endpoint_doc_degree:
        return []
    return docs


def expand_low_degree_endpoints(
    *,
    endpoint_to_docs: Mapping[str, Sequence[int]],
    endpoint_keys: Iterable[str],
    max_endpoint_doc_degree: int,
) -> tuple[List[int], Dict[str, int]]:
    docs: List[int] = []
    skipped: Dict[str, int] = {}
    for key in sorted({str(item) for item in endpoint_keys if str(item)}):
        endpoint_docs = unique_ints(endpoint_to_docs.get(key, []) or [])
        if max_endpoint_doc_degree > 0 and len(endpoint_docs) > max_endpoint_doc_degree:
            skipped[key] = len(endpoint_docs)
            continue
        docs.extend(endpoint_docs)
    return unique_ints(docs), skipped


def structural_row(
    *,
    dataset: str,
    record: Mapping[str, Any],
    dense_docs: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
    doc_units: Mapping[int, Sequence[Mapping[str, Any]]],
    endpoint_to_docs: Mapping[str, Sequence[int]],
    cache_rows: Mapping[int, Mapping[str, Any]],
    passage_index: Mapping[str, int],
    title_index: Mapping[str, Sequence[int]],
    max_endpoint_doc_degree: int,
) -> Dict[str, Any]:
    query_index = safe_int(record.get("query_index", record.get("query_idx", record.get("qid", 0))))
    question = str(record.get("question") or record.get("query") or "")
    cache_row = cache_rows.get(query_index, {})
    query_triples = cache_row.get("query_triples", []) or []
    program_triples = lower_query_triples_for_typed_program(query=question, query_triples=query_triples)
    obligations = build_query_obligation_units(query=question, query_triples=program_triples)
    dense_units = [unit for doc in dense_docs for unit in doc_units.get(int(doc), [])]
    units_by_id = {
        str(unit.get("unit_id") or ""): unit
        for unit in dense_units
        if str(unit.get("unit_id") or "")
    }
    matches = match_query_obligations_to_sto_facts(obligations=obligations, candidate_units=dense_units)
    support_docs = unique_ints(
        (units_by_id.get(str(match.get("unit_id") or ""), {}) or {}).get("doc_index", match.get("doc_index"))
        for obligation_matches in matches.values()
        for match in obligation_matches
    )
    anchor_keys: Set[str] = set()
    for obligation in obligations:
        anchor_keys.update(bound_endpoint_keys(obligation))
    for doc in support_docs:
        for unit in doc_units.get(doc, []) or []:
            row = fact_role_keys(unit)
            anchor_keys.update(str(key) for key in set(row.get("subject_keys", set())) | set(row.get("object_keys", set())) if key)

    endpoint_transition_docs, endpoint_hub_skipped = expand_low_degree_endpoints(
        endpoint_to_docs=endpoint_to_docs,
        endpoint_keys=anchor_keys,
        max_endpoint_doc_degree=max_endpoint_doc_degree,
    )
    bridge_keys = [
        str(value)
        for obligation_matches in matches.values()
        for match in obligation_matches
        for values in (match.get("variable_bindings", {}) or {}).values()
        for value in values or []
    ]
    bridge_docs, bridge_hub_skipped = expand_low_degree_endpoints(
        endpoint_to_docs=endpoint_to_docs,
        endpoint_keys=bridge_keys,
        max_endpoint_doc_degree=max_endpoint_doc_degree,
    )
    neighborhood_docs = unique_ints(endpoint_transition_docs + bridge_docs)
    provenance: Dict[int, Set[str]] = {}
    add_provenance(provenance, dense_docs, "dense_seed")
    add_provenance(provenance, support_docs, "support_grounding")
    add_provenance(provenance, endpoint_transition_docs, "endpoint_transition")
    add_provenance(provenance, bridge_docs, "bridge_expansion")
    add_provenance(provenance, neighborhood_docs, "neighborhood")
    candidate_docs = unique_ints(list(dense_docs) + support_docs + endpoint_transition_docs + bridge_docs + neighborhood_docs)
    return {
        "dataset": dataset,
        "query_index": query_index,
        "question": question,
        "gold_answers": list(record.get("gold_answers", []) or []),
        "gold_titles": list(record.get("gold_titles", []) or []),
        "gold_doc_indices": gold_doc_indices_for_record(
            record=record,
            passage_index=passage_index,
            title_index=title_index,
        ),
        "native_dense_doc_indices_top10": list(dense_docs[:10]),
        "dense_pool_doc_indices": list(dense_docs),
        "retrieved_doc_indices_top5": list(dense_docs[:5]),
        "retrieved_doc_indices_top10": list(dense_docs[:10]),
        "support_set_doc_indices_top10": support_docs[:10],
        "endpoint_transition_doc_indices_top10": endpoint_transition_docs[:10],
        "bridge_doc_indices_top10": bridge_docs[:10],
        "neighborhood_doc_indices_top10": neighborhood_docs[:10],
        "candidate_doc_indices": candidate_docs,
        "candidate_doc_provenance": {str(doc): sorted(labels) for doc, labels in sorted(provenance.items())},
        "structural_expansion_summary": {
            "max_endpoint_doc_degree": int(max_endpoint_doc_degree),
            "endpoint_hub_skipped_count": len(endpoint_hub_skipped),
            "bridge_hub_skipped_count": len(bridge_hub_skipped),
            "endpoint_hub_skipped": dict(sorted(endpoint_hub_skipped.items())[:20]),
            "bridge_hub_skipped": dict(sorted(bridge_hub_skipped.items())[:20]),
        },
    }


def build_dataset_payload(
    *,
    dataset: str,
    pool_path: Path,
    openie_path: Path,
    cache_path: Path,
    output_report_path: Path,
    max_queries: int,
    dense_limit: int,
    max_endpoint_doc_degree: int,
) -> Dict[str, Any]:
    pool_payload = load_json(pool_path)
    openie_docs = openie_docs_from_payload(load_json(openie_path))
    cache_rows = cache_rows_by_query(load_json(cache_path), dataset)
    passage_index = build_passage_index(openie_docs)
    title_index = build_title_index(openie_docs)
    doc_units = build_doc_units(openie_docs)
    endpoint_to_docs = build_endpoint_to_docs(doc_units)
    records = records_from_pool_payload(pool_payload)
    if max_queries > 0:
        records = records[:max_queries]
    rows = [
        structural_row(
            dataset=dataset,
            record=record,
            dense_docs=pool_doc_indices(record, dense_limit=dense_limit),
            openie_docs=openie_docs,
            doc_units=doc_units,
            endpoint_to_docs=endpoint_to_docs,
            cache_rows=cache_rows,
            passage_index=passage_index,
            title_index=title_index,
            max_endpoint_doc_degree=max_endpoint_doc_degree,
        )
        for record in records
    ]
    return {
        "dataset": dataset,
        "report_path": str(output_report_path.resolve()),
        "openie_path": str(openie_path.resolve()),
        "pool_path": str(pool_path.resolve()),
        "query_obligation_cache_path": str(cache_path.resolve()),
        "rows": rows,
    }


def build_report(
    *,
    datasets: Sequence[str],
    pool_template: str,
    openie_template: str,
    query_obligation_cache_template: str,
    output_report_path: Path,
    max_queries: int,
    dense_limit: int,
    max_endpoint_doc_degree: int = DEFAULT_MAX_ENDPOINT_DOC_DEGREE,
) -> Dict[str, Any]:
    dataset_payloads: List[Dict[str, Any]] = []
    variant_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        payload = build_dataset_payload(
            dataset=dataset,
            pool_path=Path(pool_template.format(dataset=dataset)).resolve(),
            openie_path=Path(openie_template.format(dataset=dataset)).resolve(),
            cache_path=Path(query_obligation_cache_template.format(dataset=dataset)).resolve(),
            output_report_path=output_report_path,
            max_queries=max_queries,
            dense_limit=dense_limit,
            max_endpoint_doc_degree=max_endpoint_doc_degree,
        )
        dataset_payloads.append(payload)
        variant_rows.extend(dict(row, dataset=dataset) for row in payload["rows"])
    return {
        "source": "structural_v13b_report_from_proprag_pool",
        "dense_pool_source": pool_template,
        "openie_template": openie_template,
        "query_obligation_cache_template": query_obligation_cache_template,
        "max_queries": int(max_queries),
        "dense_limit": int(dense_limit),
        "max_endpoint_doc_degree": int(max_endpoint_doc_degree),
        "datasets": dataset_payloads,
        "variants": {
            "hippohead_qgate_lexbeam_topkfact_stabilityfallback_guarded_local_ppr_gamma_0.3": variant_rows
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build structural v13b source report from dense pool.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--pool-template", default=DEFAULT_POOL_TEMPLATE)
    parser.add_argument("--openie-template", default=DEFAULT_OPENIE_TEMPLATE)
    parser.add_argument("--query-obligation-cache-template", default=DEFAULT_CACHE_TEMPLATE)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--dense-limit", type=int, default=100)
    parser.add_argument("--max-endpoint-doc-degree", type=int, default=DEFAULT_MAX_ENDPOINT_DOC_DEGREE)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(args.output_json)
    report = build_report(
        datasets=parse_csv(args.datasets),
        pool_template=str(args.pool_template),
        openie_template=str(args.openie_template),
        query_obligation_cache_template=str(args.query_obligation_cache_template),
        output_report_path=output,
        max_queries=max(int(args.max_queries), 0),
        dense_limit=max(int(args.dense_limit), 1),
        max_endpoint_doc_degree=max(int(args.max_endpoint_doc_degree), 0),
    )
    write_json(output, report)
    print(f"Wrote structural v13b source report: {output} ({sum(len(d['rows']) for d in report['datasets'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
