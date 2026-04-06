#!/usr/bin/env python3
"""Audit alias/canonicalization gaps for unresolved MuSiQue coverage cases.

This script is intentionally offline-only. It does not modify retrieval,
selector, gate, or rerank behavior. It diagnoses whether remaining A-class
coverage failures are best explained by:

1. missing title -> bare-entity alias closure
2. missing descriptor-stripped canonicalization
3. alias already being available, with additional reachability / hop issues
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hipporag.utils.causal_utils import normalize_structure_text, resolve_structure_city_state_alias_pairs


DEFAULT_QUERY_IDS = (65, 69, 85)
DESCRIPTOR_PREFIXES = (
    "us president ",
    "u s president ",
    "president ",
    "president of the united states ",
    "city of ",
    "county of ",
    "district of ",
    "mukim of ",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_openie_docs(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    return list(payload.get("docs", []) or [])


def build_title_to_docs(openie_docs: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    title_to_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc in openie_docs:
        passage = str(doc.get("passage", "") or "")
        title = passage.split("\n", 1)[0].strip()
        if title:
            title_to_docs[title].append(doc)
    return title_to_docs


def normalize_alias_core(text: str) -> str:
    normalized = normalize_structure_text(text)
    if not normalized:
        return ""
    for prefix in DESCRIPTOR_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_bare_title_alias_candidates(title: str, normalized_entities: set[str]) -> List[Dict[str, str]]:
    stripped_title = str(title or "").strip()
    if not stripped_title:
        return []

    normalized_title = normalize_structure_text(stripped_title)
    candidates: List[Dict[str, str]] = []

    if "," in stripped_title:
        bare_title = stripped_title.split(",", 1)[0].strip()
        normalized_bare = normalize_structure_text(bare_title)
        if normalized_bare and normalized_bare in normalized_entities and normalized_bare != normalized_title:
            candidates.append({
                "full_form": normalized_title,
                "alias_form": normalized_bare,
                "source": "comma_suffix_strip",
            })

    return candidates


def cluster_descriptor_aliases(normalized_entities: Iterable[str]) -> List[Dict[str, Any]]:
    core_to_forms: Dict[str, set[str]] = defaultdict(set)
    for entity in normalized_entities:
        core = normalize_alias_core(entity)
        if core and core != entity:
            core_to_forms[core].add(entity)

    clusters: List[Dict[str, Any]] = []
    for core, forms in sorted(core_to_forms.items()):
        if len(forms) < 2:
            continue
        clusters.append({
            "core_form": core,
            "forms": sorted(forms),
        })
    return clusters


def collect_case_entities(case: Dict[str, Any], title_to_docs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    relevant_titles: List[str] = []
    for title in [case.get("best_offrank_title"), *(case.get("gold_titles") or [])]:
        normalized = str(title or "").strip()
        if normalized and normalized not in relevant_titles:
            relevant_titles.append(normalized)

    docs: List[Dict[str, Any]] = []
    for title in relevant_titles:
        docs.extend(title_to_docs.get(title, []))

    normalized_entities: set[str] = set()
    title_entity_mismatches: List[Dict[str, Any]] = []
    title_to_entity_overlap: Dict[str, List[str]] = {}
    for title in relevant_titles:
        normalized_title = normalize_structure_text(title)
        doc_entities: set[str] = set()
        for doc in title_to_docs.get(title, []):
            for triple in list(doc.get("extracted_triples", []) or []):
                if not isinstance(triple, Sequence) or len(triple) != 3:
                    continue
                for endpoint in (triple[0], triple[2]):
                    normalized = normalize_structure_text(str(endpoint))
                    if normalized:
                        normalized_entities.add(normalized)
                        doc_entities.add(normalized)

        overlapping_entities = sorted(
            entity for entity in doc_entities
            if entity == normalized_title
            or entity.startswith(normalized_title + " ")
            or normalized_title.startswith(entity + " ")
        )
        title_to_entity_overlap[title] = overlapping_entities
        if normalized_title and normalized_title not in doc_entities:
            title_entity_mismatches.append({
                "title": title,
                "normalized_title": normalized_title,
                "entity_overlap": overlapping_entities[:8],
            })

    return {
        "relevant_titles": relevant_titles,
        "normalized_entities": normalized_entities,
        "title_entity_mismatches": title_entity_mismatches,
        "title_to_entity_overlap": title_to_entity_overlap,
    }


def diagnose_case(case: Dict[str, Any], title_to_docs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    entity_info = collect_case_entities(case, title_to_docs)
    normalized_entities = entity_info["normalized_entities"]
    current_location_alias_pairs = resolve_structure_city_state_alias_pairs(
        normalized_entities,
        continuity_probe_mode="location_alias",
    )
    bare_title_alias_candidates: List[Dict[str, str]] = []
    for title in entity_info["relevant_titles"]:
        bare_title_alias_candidates.extend(
            extract_bare_title_alias_candidates(title, normalized_entities)
        )

    descriptor_clusters = cluster_descriptor_aliases(normalized_entities)
    best_offrank_normalized = normalize_structure_text(str(case.get("best_offrank_title") or ""))
    best_offrank_candidates = [
        candidate for candidate in bare_title_alias_candidates
        if candidate["full_form"] == best_offrank_normalized
    ]
    best_offrank_alias_already_available = any(
        current_location_alias_pairs.get(candidate["full_form"]) == candidate["alias_form"]
        for candidate in best_offrank_candidates
    )
    missing_bare_candidates = [
        candidate
        for candidate in bare_title_alias_candidates
        if current_location_alias_pairs.get(candidate["full_form"]) != candidate["alias_form"]
    ]
    missing_best_offrank_candidates = [
        candidate
        for candidate in best_offrank_candidates
        if current_location_alias_pairs.get(candidate["full_form"]) != candidate["alias_form"]
    ]

    if descriptor_clusters:
        primary_diagnosis = "alias_plus_relation_family"
    elif missing_best_offrank_candidates:
        primary_diagnosis = "title_alias_missing"
    elif best_offrank_alias_already_available:
        primary_diagnosis = "alias_present_but_additional_reachability_needed"
    elif missing_bare_candidates:
        primary_diagnosis = "title_alias_missing"
    elif current_location_alias_pairs:
        primary_diagnosis = "alias_present_but_additional_reachability_needed"
    else:
        primary_diagnosis = "alias_signal_weak"

    alias_only_plausible = primary_diagnosis == "title_alias_missing"
    notes: List[str] = []
    if missing_bare_candidates:
        notes.append("Relevant title form does not appear as a structure entity, but its bare form does.")
    if current_location_alias_pairs:
        notes.append("Current location_alias probe could already derive at least one city/state alias in this case.")
    if descriptor_clusters:
        notes.append("Entity forms collapse under descriptor stripping, suggesting canonicalization beyond city/state alias.")
    if primary_diagnosis == "alias_present_but_additional_reachability_needed":
        notes.append("Alias closure alone is unlikely to fix the case; additional hop/doc reachability is implicated.")

    return {
        "query_id": case.get("query_id"),
        "question": case.get("question"),
        "best_offrank_title": case.get("best_offrank_title"),
        "gold_titles": list(case.get("gold_titles") or []),
        "relevant_titles": entity_info["relevant_titles"],
        "title_entity_mismatches": entity_info["title_entity_mismatches"],
        "current_location_alias_pairs": dict(sorted(current_location_alias_pairs.items())),
        "bare_title_alias_candidates": bare_title_alias_candidates,
        "best_offrank_bare_title_alias_candidates": best_offrank_candidates,
        "best_offrank_alias_already_available": best_offrank_alias_already_available,
        "missing_bare_title_alias_candidates": missing_bare_candidates,
        "missing_best_offrank_bare_title_alias_candidates": missing_best_offrank_candidates,
        "descriptor_alias_clusters": descriptor_clusters,
        "primary_diagnosis": primary_diagnosis,
        "alias_only_plausible": alias_only_plausible,
        "notes": notes,
    }


def render_markdown(cases: List[Dict[str, Any]]) -> str:
    lines = ["# MuSiQue Alias Closure Audit", ""]
    lines.append(f"- Cases audited: {len(cases)}")
    lines.append("")
    for case in cases:
        lines.append(f"## Q{case['query_id']} {case['question']}")
        lines.append("")
        lines.append(f"- Best off-rank title: `{case['best_offrank_title']}`")
        lines.append(f"- Primary diagnosis: `{case['primary_diagnosis']}`")
        lines.append(f"- Alias-only plausible: `{case['alias_only_plausible']}`")
        lines.append(f"- Current location_alias pairs: {case['current_location_alias_pairs']}")
        lines.append(f"- Best off-rank bare-title alias candidates: {case['best_offrank_bare_title_alias_candidates']}")
        lines.append(f"- Best off-rank alias already available: `{case['best_offrank_alias_already_available']}`")
        lines.append(f"- Missing bare-title alias candidates: {case['missing_bare_title_alias_candidates']}")
        lines.append(f"- Missing best off-rank bare-title alias candidates: {case['missing_best_offrank_bare_title_alias_candidates']}")
        lines.append(f"- Descriptor alias clusters: {case['descriptor_alias_clusters']}")
        lines.append(f"- Title/entity mismatches: {case['title_entity_mismatches']}")
        for note in case.get("notes", []):
            lines.append(f"- Note: {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit alias/canonicalization gaps for unresolved MuSiQue cases.")
    parser.add_argument("--coverage_cases_json", type=Path, required=True)
    parser.add_argument("--openie_path", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--query_ids", type=int, nargs="*", default=list(DEFAULT_QUERY_IDS))
    args = parser.parse_args()

    coverage_payload = load_json(args.coverage_cases_json)
    selected_cases = [
        case for case in list(coverage_payload.get("cases", []) or [])
        if int(case.get("query_id") or -1) in set(args.query_ids)
    ]
    openie_docs = load_openie_docs(args.openie_path)
    title_to_docs = build_title_to_docs(openie_docs)
    audited_cases = [diagnose_case(case, title_to_docs) for case in selected_cases]

    output_payload = {
        "summary": {
            "case_count": len(audited_cases),
            "query_ids": [case["query_id"] for case in audited_cases],
            "diagnosis_counts": {
                diagnosis: sum(1 for case in audited_cases if case["primary_diagnosis"] == diagnosis)
                for diagnosis in sorted({case["primary_diagnosis"] for case in audited_cases})
            },
        },
        "cases": audited_cases,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(audited_cases), encoding="utf-8")


if __name__ == "__main__":
    main()
