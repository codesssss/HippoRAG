#!/usr/bin/env python3
"""Summarize schema-level gaps from V13B gold OpenIE audits.

This is a diagnostic-only report generator.  It reads the outputs produced by
``analyze_v13b_gold_openie_audit.py`` and groups failures into coarse schema
families.  It does not change retrieval, matching, graph construction, PPR, or
fallback behavior.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


STATUS_EXPLANATIONS = {
    "candidate_missing": "gold docs are not fully present in the candidate universe",
    "gold_endpoint_missing": "gold docs exist, but OpenIE facts do not expose the required endpoint role",
    "gold_endpoint_relation_mismatch": "gold docs expose the endpoint, but not the required relation frame",
    "gold_exact_openie_match_available": "gold docs contain an exact role-aligned fact, but selector still did not ground it",
    "gold_relation_present_without_endpoint_binding": "gold docs expose the relation, but not bound to the required endpoint",
    "gold_openie_no_facts": "gold docs have no usable OpenIE fact units",
    "gold_docs_missing": "source report does not carry gold document indices",
    "unknown": "unclassified audit status",
}

TEMPORAL_MARKERS = {
    "birth",
    "born",
    "date",
    "dead",
    "death",
    "die",
    "died",
    "released",
    "release",
    "year",
}
PLACE_MARKERS = {
    "birthplace",
    "city",
    "country",
    "hometown",
    "language",
    "located",
    "location",
    "origin",
    "place",
    "spoken",
}
WORK_MARKERS = {
    "actor",
    "artist",
    "author",
    "cast",
    "character",
    "compose",
    "composer",
    "direct",
    "film",
    "music",
    "perform",
    "produce",
    "produc",
    "star",
    "voice",
    "written",
    "writer",
}
PERSON_ROLE_MARKERS = {
    "ancestor",
    "child",
    "daughter",
    "father",
    "mother",
    "parent",
    "predecessor",
    "relative",
    "sibling",
    "son",
    "spouse",
    "successor",
    "wife",
}
OFFICE_ROLE_MARKERS = {
    "appoint",
    "chief",
    "minister",
    "office",
    "president",
    "secretary",
    "serve",
    "serv",
    "treasury",
}
EVENT_MARKERS = {
    "award",
    "competition",
    "draft",
    "event",
    "game",
    "league",
    "match",
    "olympic",
    "season",
    "tournament",
    "won",
}
TYPE_PREFIXES = {
    "album",
    "book",
    "episode",
    "film",
    "movie",
    "novel",
    "play",
    "series",
    "song",
}
DESCRIPTIVE_ENDPOINT_MARKERS = {
    "a",
    "an",
    "current",
    "former",
    "last",
    "recently",
    "same",
    "the",
    "these",
    "this",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relation_tokens(relation_key: str) -> set[str]:
    return {token for token in re.split(r"[_\W]+", str(relation_key or "").lower()) if token and token != "empty"}


def relation_family(relation_key: str) -> str:
    key = str(relation_key or "")
    if key in {"", "<empty>"}:
        return "lead_or_untyped_fact"
    tokens = relation_tokens(key)
    if key in {"born_on", "died_on", "released_on"} or tokens & TEMPORAL_MARKERS:
        return "temporal_attribute"
    if tokens & PLACE_MARKERS or key.endswith("_in"):
        return "place_or_origin_attribute"
    if tokens & WORK_MARKERS:
        return "work_metadata_role"
    if tokens & PERSON_ROLE_MARKERS:
        return "person_relation_role"
    if tokens & OFFICE_ROLE_MARKERS:
        return "office_or_position_role"
    if tokens & EVENT_MARKERS:
        return "event_or_competition_role"
    return "other_relation"


def parse_relation_pair(pair: str) -> Tuple[str, str]:
    if " -> " not in pair:
        return str(pair or ""), ""
    left, right = pair.split(" -> ", 1)
    return left.strip(), right.strip()


def pair_gap_kind(query_relation: str, gold_relation: str) -> str:
    query_family = relation_family(query_relation)
    gold_family = relation_family(gold_relation)
    if gold_family == "lead_or_untyped_fact" and query_family in {
        "temporal_attribute",
        "place_or_origin_attribute",
        "work_metadata_role",
    }:
        return "lead_sentence_attribute_not_materialized"
    if query_family != gold_family:
        return "cross_family_relation_mismatch"
    return "same_family_relation_mismatch"


def bound_endpoint_surfaces(case: Mapping[str, Any]) -> List[str]:
    obligation = case.get("normalized_obligation", {}) or {}
    surfaces: List[str] = []
    if not obligation.get("subject_is_variable", False):
        subject = str(obligation.get("subject_surface") or "").strip()
        if subject:
            surfaces.append(subject)
    if not obligation.get("object_is_variable", False):
        obj = str(obligation.get("object_surface") or "").strip()
        if obj:
            surfaces.append(obj)
    return surfaces


def endpoint_shape(surface: str) -> str:
    raw = str(surface or "").strip()
    tokens = [token for token in re.split(r"\W+", raw.lower()) if token]
    if not raw:
        return "empty_endpoint"
    if re.search(r"\([^)]*\)", raw):
        return "title_qualified_endpoint"
    if tokens and tokens[0] in TYPE_PREFIXES and len(tokens) > 1:
        return "typed_title_endpoint"
    if tokens and (tokens[0] in DESCRIPTIVE_ENDPOINT_MARKERS or re.match(r"^\d+(st|nd|rd|th)$", tokens[0])):
        return "descriptive_bound_endpoint"
    if len(tokens) >= 5:
        return "long_bound_endpoint"
    return "named_endpoint"


def case_endpoint_shape(case: Mapping[str, Any]) -> str:
    surfaces = bound_endpoint_surfaces(case)
    if not surfaces:
        return "all_variable_obligation"
    shapes = [endpoint_shape(surface) for surface in surfaces]
    priority = [
        "descriptive_bound_endpoint",
        "typed_title_endpoint",
        "title_qualified_endpoint",
        "long_bound_endpoint",
        "named_endpoint",
        "empty_endpoint",
    ]
    for shape in priority:
        if shape in shapes:
            return shape
    return shapes[0]


def representative_endpoint_examples(report: Mapping[str, Any], *, limit_per_shape: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for status, cases in (report.get("representative_cases", {}) or {}).items():
        for case in cases or []:
            shape = case_endpoint_shape(case)
            if len(examples[shape]) >= limit_per_shape:
                continue
            examples[shape].append(
                {
                    "dataset": report.get("dataset", ""),
                    "status": status,
                    "query_index": case.get("query_index"),
                    "raw_triple": case.get("raw_triple"),
                    "bound_endpoints": bound_endpoint_surfaces(case),
                    "question": case.get("question", ""),
                }
            )
    return dict(sorted(examples.items()))


def summarize_reports(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    status_totals: Counter[str] = Counter()
    status_by_dataset: Dict[str, Dict[str, int]] = {}
    audited_by_dataset: Dict[str, int] = {}
    relation_pair_totals: Counter[str] = Counter()
    mismatch_by_query_family: Counter[str] = Counter()
    mismatch_family_matrix: Counter[str] = Counter()
    gap_kind_counts: Counter[str] = Counter()
    endpoint_shape_counts: Counter[str] = Counter()
    endpoint_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for report in reports:
        dataset = str(report.get("dataset") or "")
        status_counts = {str(key): int(value or 0) for key, value in (report.get("status_counts", {}) or {}).items()}
        status_by_dataset[dataset] = dict(sorted(status_counts.items()))
        audited_by_dataset[dataset] = int(report.get("audited_ungrounded_obligation_count") or 0)
        status_totals.update(status_counts)

        for pair, count in (report.get("relation_mismatch_pair_counts", {}) or {}).items():
            query_relation, gold_relation = parse_relation_pair(str(pair))
            count = int(count or 0)
            relation_pair_totals[str(pair)] += count
            query_family = relation_family(query_relation)
            gold_family = relation_family(gold_relation)
            mismatch_by_query_family[query_family] += count
            mismatch_family_matrix[f"{query_family} -> {gold_family}"] += count
            gap_kind_counts[pair_gap_kind(query_relation, gold_relation)] += count

        for shape, examples in representative_endpoint_examples(report).items():
            endpoint_shape_counts[shape] += len(examples)
            for example in examples:
                if len(endpoint_examples[shape]) < 8:
                    endpoint_examples[shape].append(example)

    return {
        "dataset_count": len(reports),
        "counting_notes": {
            "status_counts": "Counts ungrounded retrieval-critical obligations.",
            "relation_mismatch_counts": "Counts endpoint-hit fact relations from audit reports, so totals can exceed obligation counts.",
            "endpoint_shape_counts": "Counts representative cases retained by audit reports, not all failures.",
        },
        "audited_ungrounded_obligation_count_by_dataset": dict(sorted(audited_by_dataset.items())),
        "status_counts_by_dataset": dict(sorted(status_by_dataset.items())),
        "status_totals": dict(status_totals.most_common()),
        "status_explanations": STATUS_EXPLANATIONS,
        "relation_mismatch_pair_top": dict(relation_pair_totals.most_common(40)),
        "relation_mismatch_by_query_family": dict(mismatch_by_query_family.most_common()),
        "relation_mismatch_family_matrix": dict(mismatch_family_matrix.most_common()),
        "schema_gap_kind_counts": dict(gap_kind_counts.most_common()),
        "representative_endpoint_shape_counts": dict(endpoint_shape_counts.most_common()),
        "representative_endpoint_examples": dict(sorted(endpoint_examples.items())),
        "interpretation": build_interpretation(status_totals=status_totals, gap_kind_counts=gap_kind_counts),
    }


def build_interpretation(*, status_totals: Counter[str], gap_kind_counts: Counter[str]) -> Dict[str, Any]:
    audited = sum(status_totals.values())
    candidate_missing = status_totals.get("candidate_missing", 0)
    endpoint_relation = status_totals.get("gold_endpoint_relation_mismatch", 0)
    endpoint_missing = status_totals.get("gold_endpoint_missing", 0)
    exact_available = status_totals.get("gold_exact_openie_match_available", 0)
    return {
        "main_bottleneck": main_bottleneck_label(
            audited=audited,
            candidate_missing=candidate_missing,
            endpoint_relation=endpoint_relation,
            endpoint_missing=endpoint_missing,
        ),
        "candidate_missing_fraction": round(candidate_missing / max(1, audited), 6),
        "endpoint_relation_mismatch_fraction": round(endpoint_relation / max(1, audited), 6),
        "endpoint_missing_fraction": round(endpoint_missing / max(1, audited), 6),
        "exact_available_fraction": round(exact_available / max(1, audited), 6),
        "top_schema_gap_kind": next(iter(gap_kind_counts), ""),
        "recommended_next_step": recommended_next_step(
            audited=audited,
            candidate_missing=candidate_missing,
            endpoint_relation=endpoint_relation,
            endpoint_missing=endpoint_missing,
            exact_available=exact_available,
        ),
    }


def main_bottleneck_label(*, audited: int, candidate_missing: int, endpoint_relation: int, endpoint_missing: int) -> str:
    if audited <= 0:
        return "no_audited_failures"
    if candidate_missing / audited > 0.5:
        return "candidate_universe"
    if (endpoint_relation + endpoint_missing) / audited > 0.5:
        return "qwen_openie_schema_alignment"
    return "mixed"


def recommended_next_step(
    *,
    audited: int,
    candidate_missing: int,
    endpoint_relation: int,
    endpoint_missing: int,
    exact_available: int,
) -> str:
    if audited <= 0:
        return "No action: there are no audited ungrounded obligations."
    if candidate_missing / audited > 0.5:
        return "Inspect structural candidate construction before changing the selector."
    if exact_available / audited > 0.2:
        return "Inspect selector grounding plumbing because exact gold OpenIE matches are available but unused."
    if endpoint_relation >= endpoint_missing:
        return "Define a corpus/query evidence-frame contract and test Qwen OpenIE against it; do not tune selector parameters yet."
    return "Inspect endpoint canonicalization and query compiler endpoint surfaces before adding relation rules."


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" if index == 0 else "---:" for index, _ in enumerate(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    datasets = sorted((summary.get("status_counts_by_dataset", {}) or {}).keys())
    statuses = sorted((summary.get("status_totals", {}) or {}).keys())
    status_rows = []
    for dataset in datasets:
        counts = summary.get("status_counts_by_dataset", {}).get(dataset, {}) or {}
        status_rows.append([dataset, summary.get("audited_ungrounded_obligation_count_by_dataset", {}).get(dataset, 0)] + [counts.get(status, 0) for status in statuses])

    lines = [
        "# V13B Gold OpenIE Schema Gap Summary",
        "",
        "This report is diagnostic-only. It summarizes gold OpenIE audit outputs and does not change retrieval, graph construction, matching, or fallback behavior.",
        "",
        "Counting note: status counts are obligation counts; relation mismatch tables count endpoint-hit fact relations, so they can exceed the number of audited obligations. Endpoint shape tables use representative cases retained by the audit reports.",
        "",
        "## Status By Dataset",
        "",
    ]
    lines.extend(markdown_table(["dataset", "audited"] + statuses, status_rows))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "| item | value |",
            "|---|---|",
        ]
    )
    for key, value in (summary.get("interpretation", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Schema Gap Kinds", "", "| gap_kind | count |", "|---|---:|"])
    for key, value in (summary.get("schema_gap_kind_counts", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Relation Mismatch By Query Family", "", "| query_relation_family | count |", "|---|---:|"])
    for key, value in (summary.get("relation_mismatch_by_query_family", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Relation Family Matrix", "", "| query_family -> gold_family | count |", "|---|---:|"])
    for key, value in (summary.get("relation_mismatch_family_matrix", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Top Relation Mismatch Pairs", "", "| query_relation -> gold_relation | count |", "|---|---:|"])
    for key, value in (summary.get("relation_mismatch_pair_top", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Representative Endpoint Shapes", "", "| endpoint_shape | representative_case_count |", "|---|---:|"])
    for key, value in (summary.get("representative_endpoint_shape_counts", {}) or {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Endpoint Shape Examples", ""])
    for shape, examples in (summary.get("representative_endpoint_examples", {}) or {}).items():
        lines.append(f"### {shape}")
        for example in examples[:5]:
            lines.append(
                f"- dataset={example.get('dataset')} q={example.get('query_index')} "
                f"status={example.get('status')} endpoints={example.get('bound_endpoints')} "
                f"triple={example.get('raw_triple')}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize V13B gold OpenIE schema gaps from audit JSON files.")
    parser.add_argument("--audit-json", action="append", required=True, help="Path to one gold OpenIE audit JSON. Repeat for multiple datasets.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    reports = [load_json(Path(path)) for path in args.audit_json]
    summary = summarize_reports(reports)
    write_json(Path(args.output_json), summary)
    write_markdown(Path(args.output_md), summary)
    print(
        json.dumps(
            {
                "dataset_count": summary["dataset_count"],
                "status_totals": summary["status_totals"],
                "interpretation": summary["interpretation"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
