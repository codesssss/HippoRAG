#!/usr/bin/env python3
"""Diagnose why v13b query demands fail to ground in the current substrate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from build_query_obligation_units import (
    bound_endpoint_matches_signatures,
    build_query_obligation_units,
    match_query_obligations_to_sto_facts,
)
from build_source_title_openie_substrate import build_units_for_doc
from evaluate_obligation_closed_sto_local_ppr import query_triples_from_row
from evaluate_obligation_closed_sto_selector import candidate_doc_indices_from_row, row_doc_indices, unique_ints
from evaluate_transition_component_retriever import all_gold_at_k, recall_at_k
from query_obligation_typing import lower_query_triples_for_typed_program
from query_obligation_support_grounding import variable_values_from_matches
from v13b_graph_normalization import fact_role_keys, obligation_role_keys, safe_int


FAILURE_TYPES = (
    "candidate_missing",
    "endpoint_missing",
    "relation_mismatch",
    "direction_mismatch",
    "bridge_missing",
    "upstream_binding_missing",
    "overconstrained_query_demand",
    "non_retrieval_materialized",
    "openie_malformed",
    "unknown",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_rows_by_query(source_report: Mapping[str, Any], dataset: str) -> Dict[int, Dict[str, Any]]:
    for payload in source_report.get("datasets", []) or []:
        if str(payload.get("dataset") or "") != str(dataset):
            continue
        return {
            safe_int(row.get("query_index"), ordinal): dict(row)
            for ordinal, row in enumerate(payload.get("rows", []) or [])
            if isinstance(row, Mapping)
        }
    return {}


def selector_rows(selector_payload: Mapping[str, Any], dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for payload in selector_payload.get("datasets", []) or []:
        if str(payload.get("dataset") or "") != str(dataset):
            continue
        rows.extend(dict(row) for row in payload.get("rows", []) or [] if isinstance(row, Mapping))
    return rows


def cache_triples_by_query(cache_payload: Any, dataset: str) -> Dict[int, List[Sequence[Any]]]:
    rows: Sequence[Any] = []
    if isinstance(cache_payload, Mapping) and isinstance(cache_payload.get("datasets"), Sequence):
        for payload in cache_payload.get("datasets", []) or []:
            if isinstance(payload, Mapping) and str(payload.get("dataset") or "") == str(dataset):
                rows = payload.get("rows", []) or []
                break
    elif isinstance(cache_payload, Mapping) and isinstance(cache_payload.get("rows"), Sequence):
        rows = cache_payload.get("rows", []) or []
    result: Dict[int, List[Sequence[Any]]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        query_index = safe_int(row.get("query_index"), ordinal)
        triples = row.get("query_triples", []) or []
        if isinstance(triples, Sequence) and not isinstance(triples, (str, bytes)):
            result[query_index] = [
                triple
                for triple in triples
                if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes))
            ]
    return result


def openie_docs_from_payload(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        docs = payload.get("docs", []) or []
    else:
        docs = payload
    return [doc for doc in docs if isinstance(doc, Mapping)]


def build_units_for_docs(
    *,
    doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for doc_index in unique_ints(doc_indices):
        if 0 <= doc_index < len(openie_docs):
            units.extend(build_units_for_doc(openie_docs[doc_index], doc_index=doc_index, include_source_spans=False))
    return units


def relation_matched_docs(
    *,
    obligation: Mapping[str, Any],
    fact_rows: Sequence[Mapping[str, Any]],
) -> List[int]:
    ob = obligation_role_keys(obligation)
    return unique_ints(row.get("doc_index") for row in fact_rows if row.get("relation_key") == ob["relation_key"])


def role_endpoint_matches(
    *,
    endpoint_values: Iterable[str],
    fact_row: Mapping[str, Any],
    role: str,
) -> bool:
    """Use the selector's conservative endpoint alias contract for diagnostics."""

    if role == "subject":
        signatures = fact_row.get("subject_keys") or set()
    elif role == "object":
        signatures = fact_row.get("object_keys") or set()
    else:
        return False
    return any(bound_endpoint_matches_signatures(endpoint, signatures) for endpoint in endpoint_values)


def obligation_endpoint_values(
    *,
    obligation: Mapping[str, Any],
    role: str,
    grounded_variable_values: Mapping[str, Iterable[str]],
) -> List[str]:
    """Return concrete endpoint anchors available for one obligation role."""

    if role == "subject":
        if bool(obligation.get("subject_is_variable", False)):
            return list(grounded_variable_values.get(str(obligation.get("subject_variable") or ""), []) or [])
        value = obligation.get("raw_subject") or obligation.get("subject")
    elif role == "object":
        if bool(obligation.get("object_is_variable", False)):
            return list(grounded_variable_values.get(str(obligation.get("object_variable") or ""), []) or [])
        value = obligation.get("raw_object") or obligation.get("object")
    else:
        value = ""
    return [str(value)] if str(value or "").strip() else []


def obligation_missing_variable_names(
    *,
    obligation: Mapping[str, Any],
    grounded_variable_values: Mapping[str, Iterable[str]],
) -> List[str]:
    missing: List[str] = []
    if bool(obligation.get("subject_is_variable", False)):
        variable = str(obligation.get("subject_variable") or "")
        if variable and not list(grounded_variable_values.get(variable, []) or []):
            missing.append(variable)
    if bool(obligation.get("object_is_variable", False)):
        variable = str(obligation.get("object_variable") or "")
        if variable and not list(grounded_variable_values.get(variable, []) or []) and variable not in missing:
            missing.append(variable)
    return missing


def merge_match_maps(*maps: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for match_map in maps:
        for obligation_id, matches in (match_map or {}).items():
            merged.setdefault(str(obligation_id), []).extend(dict(match) for match in matches or [])
    return merged


def retrieval_critical_obligations_for_analysis(
    *,
    obligations: Sequence[Mapping[str, Any]],
    selector: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    """Return only obligations the typed selector actually required.

    The selector may materialize non-retrieval constraints for diagnostics even
    when it intentionally excludes them from execution.  Counting those rows as
    grounding failures would reintroduce the protocol pollution this analyzer
    is meant to expose.
    """

    if not bool(selector.get("use_typed_retrieval_critical_obligations", False)):
        return list(obligations)
    typed_program = selector.get("typed_query_program", {}) or {}
    if not isinstance(typed_program, Mapping):
        return list(obligations)
    required_ids = {
        str(obligation_id)
        for obligation_id in typed_program.get("retrieval_critical_obligation_ids", []) or []
        if str(obligation_id)
    }
    return [
        obligation
        for obligation in obligations
        if str(obligation.get("obligation_id") or "") in required_ids
    ]


def classify_obligation_failure(
    *,
    obligation: Mapping[str, Any],
    candidate_units: Sequence[Mapping[str, Any]],
    candidate_doc_indices: Sequence[int],
    gold_doc_indices: Sequence[int],
    grounded_variable_values: Mapping[str, Iterable[str]] | None = None,
) -> Dict[str, Any]:
    """Classify why one retrieval-critical obligation has no usable grounding."""

    ob = obligation_role_keys(obligation)
    fact_rows = [fact_role_keys(unit) for unit in candidate_units if str(unit.get("unit_type") or "") == "openie_fact"]
    malformed = [row for row in fact_rows if not row.get("subject_keys") or not row.get("relation_key") or not row.get("object_keys")]
    candidate_set = set(unique_ints(candidate_doc_indices))
    gold_set = set(unique_ints(gold_doc_indices))
    if gold_set and not gold_set <= candidate_set:
        return failure_row("candidate_missing", obligation, fact_rows, extra={"missing_gold_doc_indices": sorted(gold_set - candidate_set)})
    if not fact_rows and candidate_doc_indices:
        return failure_row("openie_malformed", obligation, fact_rows, extra={"malformed_fact_count": len(malformed)})

    variable_values = grounded_variable_values or {}
    subject_values = obligation_endpoint_values(
        obligation=obligation,
        role="subject",
        grounded_variable_values=variable_values,
    )
    object_values = obligation_endpoint_values(
        obligation=obligation,
        role="object",
        grounded_variable_values=variable_values,
    )
    if not subject_values and not object_values:
        missing_variables = obligation_missing_variable_names(
            obligation=obligation,
            grounded_variable_values=variable_values,
        )
        if missing_variables:
            return failure_row(
                "upstream_binding_missing",
                obligation,
                fact_rows,
                extra={"missing_variable_bindings": missing_variables},
            )
        return failure_row("overconstrained_query_demand", obligation, fact_rows)

    subject_role_hits = [
        row
        for row in fact_rows
        if not subject_values or role_endpoint_matches(endpoint_values=subject_values, fact_row=row, role="subject")
    ]
    object_role_hits = [
        row
        for row in fact_rows
        if not object_values or role_endpoint_matches(endpoint_values=object_values, fact_row=row, role="object")
    ]
    if subject_values and object_values:
        reversed_hits = [
            row
            for row in fact_rows
            if any(bound_endpoint_matches_signatures(value, row.get("object_keys") or set()) for value in subject_values)
            and any(bound_endpoint_matches_signatures(value, row.get("subject_keys") or set()) for value in object_values)
            and row.get("relation_key") == ob["relation_key"]
        ]
        if reversed_hits:
            return failure_row("direction_mismatch", obligation, reversed_hits)
    if not subject_role_hits or not object_role_hits:
        return failure_row(
            "endpoint_missing",
            obligation,
            fact_rows,
            extra={
                "subject_role_hit_count": len(subject_role_hits),
                "object_role_hit_count": len(object_role_hits),
            },
        )

    aligned_endpoint_hits = [
        row
        for row in fact_rows
        if (not subject_values or role_endpoint_matches(endpoint_values=subject_values, fact_row=row, role="subject"))
        and (not object_values or role_endpoint_matches(endpoint_values=object_values, fact_row=row, role="object"))
    ]
    if aligned_endpoint_hits and not any(row.get("relation_key") == ob["relation_key"] for row in aligned_endpoint_hits):
        return failure_row("relation_mismatch", obligation, aligned_endpoint_hits)

    if relation_matched_docs(obligation=obligation, fact_rows=fact_rows):
        return failure_row("bridge_missing", obligation, fact_rows)
    return failure_row("unknown", obligation, fact_rows)


def failure_row(
    failure_type: str,
    obligation: Mapping[str, Any],
    fact_rows: Sequence[Mapping[str, Any]],
    *,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    ob = obligation_role_keys(obligation)
    relation_key_counts = Counter(str(row.get("relation_key") or "") for row in fact_rows)
    examples = [
        {
            "doc_index": safe_int(row.get("doc_index")),
            "title": row.get("title", ""),
            "fact": row.get("fact", []),
            "relation_key": row.get("relation_key", ""),
        }
        for row in fact_rows[:5]
    ]
    return {
        "failure_type": failure_type if failure_type in FAILURE_TYPES else "unknown",
        "obligation_id": ob["obligation_id"],
        "raw_triple": ob["raw_triple"],
        "normalized_obligation": ob,
        "fact_relation_key_counts": dict(sorted(relation_key_counts.items())),
        "example_facts": examples,
        **dict(extra or {}),
    }


def source_vs_v13b_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = 0
    source_r5 = 0.0
    final_r5 = 0.0
    source_all = 0
    final_all = 0
    improved = 0
    regressed = 0
    for row in rows:
        gold = unique_ints(row.get("gold_doc_indices", []) or [])
        if not gold:
            continue
        source = row_doc_indices(row, "anchor_guided_evidence_doc_indices_top5", "retrieved_doc_indices_top5", limit=5)
        final = row_doc_indices(row, "obligation_closed_sto_local_ppr_doc_indices_top5", limit=5)
        count += 1
        source_r5 += recall_at_k(gold, source, 5)
        final_r5 += recall_at_k(gold, final, 5)
        source_all += int(all_gold_at_k(gold, source, 5))
        final_all += int(all_gold_at_k(gold, final, 5))
        old_count = len(set(gold) & set(source))
        new_count = len(set(gold) & set(final))
        improved += int(new_count > old_count)
        regressed += int(new_count < old_count)
    denom = float(count or 1)
    return {
        "count": count,
        "source_r5": round(source_r5 / denom, 6),
        "v13b_r5": round(final_r5 / denom, 6),
        "source_all_gold_at5": round(source_all / denom, 6),
        "v13b_all_gold_at5": round(final_all / denom, 6),
        "gold_count_improved_queries": improved,
        "gold_count_regressed_queries": regressed,
    }


def analyze_grounding_failures(
    *,
    dataset: str,
    source_report_path: Path,
    selector_json_path: Path,
    query_obligation_cache_path: Path,
    openie_json_path: Path,
    max_cases_per_type: int = 20,
    max_queries: int = 0,
) -> Dict[str, Any]:
    source_report = load_json(source_report_path)
    selector_payload = load_json(selector_json_path)
    cache_payload = load_json(query_obligation_cache_path)
    openie_docs = openie_docs_from_payload(load_json(openie_json_path))

    source_by_query = source_rows_by_query(source_report, dataset)
    cache_by_query = cache_triples_by_query(cache_payload, dataset)
    rows = selector_rows(selector_payload, dataset)
    if int(max_queries) > 0:
        rows = rows[: int(max_queries)]
    failures: List[Dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    relation_mismatch_pairs: Counter[str] = Counter()
    representative: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_query_failure_count: Dict[str, int] = {}

    for row in rows:
        query_index = safe_int(row.get("query_index"))
        selector = row.get("obligation_closed_sto_local_ppr_selector", {}) or {}
        source_row = source_by_query.get(query_index, row)
        candidate_docs = candidate_doc_indices_from_row(source_row)
        candidate_units = build_units_for_docs(doc_indices=candidate_docs, openie_docs=openie_docs)
        query_triples = query_triples_from_row(
            source_row,
            query_obligation_cache=cache_by_query,
        )
        program_triples = lower_query_triples_for_typed_program(
            query=str(source_row.get("question") or source_row.get("query") or ""),
            query_triples=query_triples,
        )
        obligations = selector.get("query_obligations") or build_query_obligation_units(
            query=str(source_row.get("question") or source_row.get("query") or ""),
            query_triples=program_triples,
        )
        obligations = retrieval_critical_obligations_for_analysis(
            obligations=obligations,
            selector=selector,
        )
        exact_matches = match_query_obligations_to_sto_facts(
            obligations=obligations,
            candidate_units=candidate_units,
        )
        support_matches = selector.get("support_grounded_obligation_matches", {}) or {}
        source_span_matches = selector.get("source_span_grounded_obligation_matches", {}) or {}
        selector_grounded_ids = {
            str(obligation_id)
            for match_map in (support_matches, source_span_matches)
            for obligation_id, matches in match_map.items()
            if matches
        }
        grounded_variable_values = variable_values_from_matches(
            obligations=obligations,
            matches=merge_match_maps(exact_matches, support_matches, source_span_matches),
        )
        row_failures: List[Dict[str, Any]] = []
        for obligation in obligations:
            obligation_id = str(obligation.get("obligation_id") or "")
            if exact_matches.get(obligation_id) or obligation_id in selector_grounded_ids:
                continue
            failure = classify_obligation_failure(
                obligation=obligation,
                candidate_units=candidate_units,
                candidate_doc_indices=candidate_docs,
                gold_doc_indices=source_row.get("gold_doc_indices", []) or [],
                grounded_variable_values=grounded_variable_values,
            )
            failure.update(
                {
                    "query_index": query_index,
                    "question": str(source_row.get("question") or source_row.get("query") or ""),
                    "abstention_reason": str(selector.get("abstention_reason") or ""),
                }
            )
            row_failures.append(failure)
            failures.append(failure)
            type_counts[failure["failure_type"]] += 1
            if failure["failure_type"] == "relation_mismatch":
                query_relation = str((failure.get("normalized_obligation", {}) or {}).get("relation_key") or "")
                for fact_relation, count in (failure.get("fact_relation_key_counts", {}) or {}).items():
                    relation_mismatch_pairs[f"{query_relation} -> {fact_relation or '<empty>'}"] += int(count or 0)
            if len(representative[failure["failure_type"]]) < max_cases_per_type:
                representative[failure["failure_type"]].append(failure)
        if row_failures:
            per_query_failure_count[str(query_index)] = len(row_failures)

    selector_metrics = (selector_payload.get("datasets", [{}])[0] or {}).get("metrics", {}) if selector_payload.get("datasets") else {}
    return {
        "dataset": dataset,
        "source_report_path": str(source_report_path),
        "selector_json_path": str(selector_json_path),
        "query_obligation_cache_path": str(query_obligation_cache_path),
        "openie_json_path": str(openie_json_path),
        "max_queries": int(max_queries),
        "failure_types": list(FAILURE_TYPES),
        "retrieval_critical_ungrounded_count": int(selector_metrics.get("abstention_reason_counts", {}).get("retrieval_critical_ungrounded", 0)),
        "classified_failure_count": len(failures),
        "failure_type_counts": dict(sorted(type_counts.items())),
        "relation_mismatch_pair_counts": dict(relation_mismatch_pairs.most_common(40)),
        "unknown_fraction": round(float(type_counts.get("unknown", 0)) / float(len(failures) or 1), 6),
        "per_query_failure_count": per_query_failure_count,
        "program_feasibility": {
            "attempted": selector_metrics.get("program_assembler_attempted_query_count", 0),
            "feasible": selector_metrics.get("program_assembler_feasible_query_count", 0),
            "changed_top5": selector_metrics.get("changed_top5_queries", 0),
            "gains": selector_metrics.get("gold_count_improved_queries", 0),
            "losses": selector_metrics.get("gold_count_regressed_queries", 0),
        },
        "source_prior_vs_v13b": source_vs_v13b_metrics(rows),
        "representative_cases": dict(representative),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"# V13B Grounding Failure Report: {report.get('dataset')}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| retrieval_critical_ungrounded | {report.get('retrieval_critical_ungrounded_count', 0)} |",
        f"| classified_failure_count | {report.get('classified_failure_count', 0)} |",
        f"| unknown_fraction | {report.get('unknown_fraction', 0.0)} |",
        "",
        "## Failure Types",
        "",
        "| failure_type | count |",
        "|---|---:|",
    ]
    for key, value in (report.get("failure_type_counts", {}) or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Relation Mismatch Pairs", "", "| query_relation -> fact_relation | count |", "|---|---:|"])
    for key, value in (report.get("relation_mismatch_pair_counts", {}) or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Source Prior vs V13B", "", "| metric | value |", "|---|---:|"])
    for key, value in (report.get("source_prior_vs_v13b", {}) or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Representative Cases", ""])
    for failure_type, cases in (report.get("representative_cases", {}) or {}).items():
        lines.append(f"### {failure_type}")
        for case in cases[:5]:
            lines.append(
                f"- q={case.get('query_index')} triple={case.get('raw_triple')} "
                f"reason={case.get('failure_type')}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze v13b grounding failures.")
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--selector-json", required=True)
    parser.add_argument("--query-obligation-cache", required=True)
    parser.add_argument("--openie-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-cases-per-type", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = analyze_grounding_failures(
        dataset=str(args.dataset),
        source_report_path=Path(args.source_report).resolve(),
        selector_json_path=Path(args.selector_json).resolve(),
        query_obligation_cache_path=Path(args.query_obligation_cache).resolve(),
        openie_json_path=Path(args.openie_json).resolve(),
        max_cases_per_type=max(int(args.max_cases_per_type), 1),
        max_queries=max(int(args.max_queries), 0),
    )
    write_json(Path(args.output_json), report)
    write_markdown(Path(args.output_md), report)
    print(json.dumps({k: report[k] for k in ("dataset", "classified_failure_count", "failure_type_counts", "unknown_fraction")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
