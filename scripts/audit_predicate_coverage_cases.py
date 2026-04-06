#!/usr/bin/env python3
"""Audit predicate coverage gaps on selected paired-report cases.

This script focuses on A-class coverage gaps: beneficial_withheld cases whose
best off-rank structure score is exactly zero. It answers:

1. Do raw OpenIE triples already contain potentially useful relations?
2. Are those relations being dropped by the current general_factual predicate
   classifier?
3. Do the rejected triples have entity endpoints that look connectable across
   the relevant document set?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_gate_decision_boundary import build_analysis, load_report


PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hipporag.utils.causal_utils import classify_directed_predicate, normalize_structure_text


CASE_BUCKET = "beneficial_withheld"
MAX_TRIPLE_EXAMPLES = 12


def load_openie_docs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("docs", []) or [])


def build_title_to_docs(openie_docs: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    title_to_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc in openie_docs:
        passage = str(doc.get("passage", "") or "")
        title = passage.split("\n", 1)[0].strip()
        if title:
            title_to_docs[title].append(doc)
    return title_to_docs


def derive_best_offrank_title(case: Dict[str, Any]) -> str | None:
    control_titles = list(case.get("control_titles", []) or [])
    candidate_titles = set(case.get("candidate_titles", []) or [])
    gold_titles = set(case.get("gold_titles", []) or [])
    control_only = [title for title in control_titles if title not in candidate_titles]
    control_only_gold = [title for title in control_only if title in gold_titles]
    selected = control_only_gold or control_only
    return str(selected[0]) if selected else None


def build_query_id_map(report: Dict[str, Any]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for index, trace in enumerate(list(report.get("setwise_selector_query_traces", []) or []), start=1):
        question = str(trace.get("question", "") or "")
        if question:
            mapping[question] = index
    return mapping


def select_coverage_gap_cases(control_report: Dict[str, Any],
                              candidate_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    analysis = build_analysis(control_report, candidate_report, case_limit=64)
    query_id_map = build_query_id_map(candidate_report)
    selected: List[Dict[str, Any]] = []
    for case in analysis["bucket_cases"][CASE_BUCKET]:
        if float(case.get("best_offrank_structure_score") or 0.0) != 0.0:
            continue
        question = str(case.get("question", ""))
        selected.append({
            "query_id": query_id_map.get(question),
            "question": question,
            "control_top_titles": list(case.get("control_titles", []) or []),
            "candidate_top_titles": list(case.get("candidate_titles", []) or []),
            "gold_titles": list(case.get("gold_titles", []) or []),
            "gate_reason": str(case.get("candidate_gate_reason", "")),
            "best_offrank_title": derive_best_offrank_title(case),
            "best_offrank_structure_score": float(case.get("best_offrank_structure_score") or 0.0),
            "raw_case_bucket": CASE_BUCKET,
        })
    selected.sort(key=lambda row: (int(row.get("query_id") or 10**9), row["question"]))
    return selected


def propose_predicate_family(predicate: str) -> str:
    normalized = normalize_structure_text(predicate)
    if not normalized:
        return "other"
    patterns = (
        ("factual_alias_or_name", r"\balso known as\b|\bofficially called\b|\bcalled\b|\breferred to as\b|\bname means\b|\bmeaning of the name\b"),
        ("factual_part_of_or_contains", r"\bpart of\b|\bis part of\b|\blocated in\b|\bis located in\b|\bneighborhood of\b|\bdistrict of\b|\bregion of\b|\bcontained in\b|\bbelongs to\b|\bentry for\b|\bcapital of\b"),
        ("factual_membership_or_affiliation", r"\bmember of\b|\baffiliated with\b|\bdivision of\b|\bnetwork\b|\bpublished by\b"),
        ("factual_event_context", r"\bduring\b|\bduring the reign of\b|\bunder\b|\bpresident under\b|\breign of\b|\bera\b|\baddressed to\b|\bgreeted in\b"),
        ("factual_media_or_work_relation", r"\bstars\b|\bwritten by\b|\bdirected by\b|\bset in\b|\bbased on\b|\badapted from\b|\bentry for\b"),
        ("factual_family_relation", r"\bdaughter of\b|\bson of\b|\bchild of\b"),
    )
    for family_name, pattern in patterns:
        if re.search(pattern, normalized):
            return family_name
    return "other"


def _entity_link_signal(normalized_subject: str,
                        normalized_object: str,
                        cross_doc_entities: set[str]) -> bool:
    return bool(normalized_subject in cross_doc_entities or normalized_object in cross_doc_entities)


def audit_case(case: Dict[str, Any], title_to_docs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    relevant_titles: List[str] = []
    for title in [case.get("best_offrank_title"), *(case.get("gold_titles") or [])]:
        normalized = str(title or "").strip()
        if normalized and normalized not in relevant_titles:
            relevant_titles.append(normalized)

    relevant_docs: List[Dict[str, Any]] = []
    for title in relevant_titles:
        relevant_docs.extend(title_to_docs.get(title, []))

    cross_doc_entity_counter: Counter[str] = Counter()
    for doc in relevant_docs:
        triples = list(doc.get("extracted_triples", []) or [])
        doc_entities = set()
        for triple in triples:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            normalized_subject = normalize_structure_text(triple[0])
            normalized_object = normalize_structure_text(triple[2])
            if normalized_subject:
                doc_entities.add(normalized_subject)
            if normalized_object:
                doc_entities.add(normalized_object)
        for entity in doc_entities:
            cross_doc_entity_counter[entity] += 1

    cross_doc_entities = {entity for entity, count in cross_doc_entity_counter.items() if count >= 2}

    raw_triples_found: List[Dict[str, Any]] = []
    rejected_triples: List[Dict[str, Any]] = []
    family_counter: Counter[str] = Counter()
    rejected_predicate_counter: Counter[str] = Counter()

    for doc in relevant_docs:
        passage = str(doc.get("passage", "") or "")
        title = passage.split("\n", 1)[0].strip()
        for triple in list(doc.get("extracted_triples", []) or []):
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            subject, predicate, object_ = [str(item) for item in triple]
            raw_triples_found.append({
                "title": title,
                "doc_idx": str(doc.get("idx", "")),
                "triple": [subject, predicate, object_],
            })
            if classify_directed_predicate(predicate, relation_probe_mode="general_factual") is not None:
                continue
            normalized_subject = normalize_structure_text(subject)
            normalized_object = normalize_structure_text(object_)
            family_name = propose_predicate_family(predicate)
            entity_link_possible = _entity_link_signal(normalized_subject, normalized_object, cross_doc_entities)
            rejected_predicate_counter[predicate] += 1
            family_counter[family_name] += 1
            rejected_triples.append({
                "title": title,
                "doc_idx": str(doc.get("idx", "")),
                "triple": [subject, predicate, object_],
                "normalized_subject": normalized_subject,
                "normalized_object": normalized_object,
                "proposed_predicate_family": family_name,
                "entity_link_possible": bool(entity_link_possible),
            })

    rejected_triples.sort(
        key=lambda row: (
            row["proposed_predicate_family"] == "other",
            not row["entity_link_possible"],
            row["title"],
            row["triple"][1],
        )
    )
    useful_rejected = [
        row for row in rejected_triples
        if row["proposed_predicate_family"] != "other" or row["entity_link_possible"]
    ]
    notes: List[str] = []
    if not raw_triples_found:
        notes.append("No OpenIE triples found in the relevant title set.")
    elif useful_rejected:
        notes.append("Raw triples exist and include rejected predicates that may support bridge closure.")
    else:
        notes.append("Raw triples exist, but the rejected predicates do not look immediately reusable for bridge closure.")

    return {
        "query_id": case.get("query_id"),
        "question": case.get("question"),
        "candidate_gap_type": "structure_zero",
        "raw_case_bucket": case.get("raw_case_bucket"),
        "best_offrank_title": case.get("best_offrank_title"),
        "relevant_titles": relevant_titles,
        "raw_triples_found": raw_triples_found[:MAX_TRIPLE_EXAMPLES],
        "raw_triple_count": len(raw_triples_found),
        "candidate_triples_rejected_by_classifier": useful_rejected[:MAX_TRIPLE_EXAMPLES],
        "candidate_rejected_triple_count": len(useful_rejected),
        "rejected_predicate_texts": [predicate for predicate, _ in rejected_predicate_counter.most_common()],
        "proposed_predicate_family": dict(sorted(family_counter.items())),
        "entity_link_possible": any(row["entity_link_possible"] for row in useful_rejected),
        "entity_nodes_present": bool(cross_doc_entity_counter),
        "head_tail_canonicalized": bool(any(row["normalized_subject"] and row["normalized_object"] for row in useful_rejected)),
        "edge_reachable_after_patch": bool(any(
            row["entity_link_possible"] and row["proposed_predicate_family"] != "other"
            for row in useful_rejected
        )),
        "structure_score_before": 0.0,
        "structure_score_after": None,
        "notes": notes,
    }


def build_summary(audits: List[Dict[str, Any]]) -> Dict[str, Any]:
    family_counter: Counter[str] = Counter()
    rejected_predicate_counter: Counter[str] = Counter()
    raw_triple_cases = 0
    usable_cases = 0
    for audit in audits:
        if int(audit.get("raw_triple_count", 0)) > 0:
            raw_triple_cases += 1
        if audit.get("edge_reachable_after_patch"):
            usable_cases += 1
        family_counter.update(dict(audit.get("proposed_predicate_family", {}) or {}))
        rejected_predicate_counter.update(list(audit.get("rejected_predicate_texts", []) or []))
    return {
        "case_count": len(audits),
        "cases_with_raw_triples": raw_triple_cases,
        "cases_with_patchable_rejected_family": usable_cases,
        "proposed_family_counts": dict(sorted(family_counter.items(), key=lambda item: (-item[1], item[0]))),
        "top_rejected_predicates": rejected_predicate_counter.most_common(20),
        "avg_rejected_triples_per_case": round(
            mean(float(audit.get("candidate_rejected_triple_count", 0)) for audit in audits), 4
        ) if audits else 0.0,
    }


def render_markdown(selected_cases: List[Dict[str, Any]],
                    audits: List[Dict[str, Any]],
                    summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# MuSiQue Predicate Coverage Audit")
    lines.append("")
    lines.append(f"- Coverage-gap cases: {len(selected_cases)}")
    lines.append(f"- Cases with raw triples: {summary['cases_with_raw_triples']}")
    lines.append(f"- Cases with patchable rejected family: {summary['cases_with_patchable_rejected_family']}")
    lines.append(f"- Proposed family counts: {summary['proposed_family_counts']}")
    lines.append("")
    lines.append("## Coverage Gap Cases")
    lines.append("")
    for case in selected_cases:
        lines.append(
            f"- Q{case['query_id']}: `{case['gate_reason']}` | best off-rank `{case['best_offrank_title']}` | "
            f"gold {case['gold_titles']}"
        )
    lines.append("")
    lines.append("## Case Audits")
    lines.append("")
    for audit in audits:
        lines.append(f"### Q{audit['query_id']} {audit['question']}")
        lines.append("")
        lines.append(f"- Best off-rank title: `{audit['best_offrank_title']}`")
        lines.append(f"- Relevant titles: {audit['relevant_titles']}")
        lines.append(f"- Raw triple count: {audit['raw_triple_count']}")
        lines.append(f"- Rejected useful triple count: {audit['candidate_rejected_triple_count']}")
        lines.append(f"- Proposed family counts: {audit['proposed_predicate_family']}")
        lines.append(f"- Entity link possible: {audit['entity_link_possible']}")
        for note in audit.get("notes", []):
            lines.append(f"- Note: {note}")
        lines.append("")
        lines.append("Rejected triple samples:")
        if not audit["candidate_triples_rejected_by_classifier"]:
            lines.append("- None")
        else:
            for row in audit["candidate_triples_rejected_by_classifier"][:6]:
                lines.append(
                    f"- `{row['title']}`: {row['triple']} -> family `{row['proposed_predicate_family']}`, "
                    f"entity_link_possible={row['entity_link_possible']}"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit predicate coverage gaps on selected MuSiQue cases.")
    parser.add_argument("--control_report", type=Path, required=True)
    parser.add_argument("--candidate_report", type=Path, required=True)
    parser.add_argument("--openie_path", type=Path, required=True)
    parser.add_argument("--coverage_cases_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    control_report = load_report(args.control_report)
    candidate_report = load_report(args.candidate_report)
    selected_cases = select_coverage_gap_cases(control_report, candidate_report)
    openie_docs = load_openie_docs(args.openie_path)
    title_to_docs = build_title_to_docs(openie_docs)
    audits = [audit_case(case, title_to_docs) for case in selected_cases]
    summary = build_summary(audits)

    coverage_payload = {
        "case_count": len(selected_cases),
        "cases": selected_cases,
    }
    audit_payload = {
        "summary": summary,
        "cases": audits,
    }

    args.coverage_cases_json.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_cases_json.write_text(json.dumps(coverage_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_json.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(selected_cases, audits, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
