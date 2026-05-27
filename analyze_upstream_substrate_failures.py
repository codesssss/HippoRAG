#!/usr/bin/env python3
"""Audit upstream substrate availability for residual retrieval failures.

This diagnostic reads existing source-clean per-query retrieval reports and
OpenIE caches. It does not run retrieval, QA, graph search, reranking, gates, or
PPR. The goal is to decide whether a missed gold passage is blocked upstream by
fact extraction/title grounding, or whether the needed substrate exists and the
failure belongs to activation/ranking/selection.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


DEFAULT_REPORTS = {
    "musique": Path("outputs_anchor_source_audit_limit100_20260423_v2/reports/musique_anchor_source_audit_limit100_per_query.json"),
    "2wikimultihopqa": Path("outputs_anchor_source_audit_limit100_20260423_v2/reports/2wiki_anchor_source_audit_limit100_per_query.json"),
    "hotpotqa": Path("outputs_anchor_source_audit_limit100_20260423_v2/reports/hotpotqa_anchor_source_audit_limit100_per_query.json"),
}


SFB_VARIANT = "hippohead_qgate_lexbeam_topkfact_stabilityfallback_guarded_local_ppr_gamma_0.3"
ANCHORINV_VARIANT = "hippohead_qgate_lexbeam_anchorinv_candexp_guarded_local_ppr_gamma_0.3"


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "with",
}


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value: Any) -> Set[str]:
    return {
        token
        for token in normalize_text(value).split()
        if len(token) >= 4 and token not in STOPWORDS
    }


def title_from_openie_doc(doc: Mapping[str, Any]) -> str:
    passage = str(doc.get("passage") or "")
    first_line = passage.splitlines()[0].strip() if passage else ""
    return first_line


def extracted_entities(doc: Mapping[str, Any]) -> List[str]:
    entities = doc.get("extracted_entities") or []
    return [str(entity) for entity in entities if str(entity).strip()]


def extracted_triples(doc: Mapping[str, Any]) -> List[Sequence[str]]:
    triples = doc.get("extracted_triples") or []
    clean = []
    for triple in triples:
        if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes)) and len(triple) >= 3:
            clean.append((str(triple[0]), str(triple[1]), str(triple[2])))
    return clean


def text_matches_any(text: str, candidates: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        if not candidate_norm:
            continue
        if normalized == candidate_norm or normalized in candidate_norm or candidate_norm in normalized:
            return True
    return False


def triple_endpoint_matches(text: str, triples: Iterable[Sequence[str]]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    for triple in triples:
        for endpoint in (triple[0], triple[2]):
            endpoint_norm = normalize_text(endpoint)
            if normalized == endpoint_norm or normalized in endpoint_norm or endpoint_norm in normalized:
                return True
    return False


def query_overlap_count(query: str, values: Iterable[str]) -> int:
    query_tokens = content_tokens(query)
    overlap: Set[str] = set()
    for value in values:
        overlap.update(query_tokens & content_tokens(value))
    return len(overlap)


def classify_gold_substrate(
    *,
    query: str,
    gold_title: str,
    openie_doc: Mapping[str, Any] | None,
    retrieved_top5: Sequence[int],
    gold_index: int,
) -> Dict[str, Any]:
    if openie_doc is None:
        return {
            "gold_in_top5": gold_index in set(retrieved_top5),
            "substrate_status": "openie_doc_missing",
            "openie_fact_count": 0,
            "openie_entity_count": 0,
            "title_entity_present": False,
            "title_triple_endpoint_present": False,
            "query_entity_overlap": 0,
            "query_triple_overlap": 0,
            "openie_title": "",
        }

    entities = extracted_entities(openie_doc)
    triples = extracted_triples(openie_doc)
    triple_texts = [" ".join(triple) for triple in triples]
    openie_title = title_from_openie_doc(openie_doc)
    title_candidates = [gold_title, openie_title]
    title_entity_present = any(text_matches_any(title, entities) for title in title_candidates if title)
    title_triple_endpoint_present = any(triple_endpoint_matches(title, triples) for title in title_candidates if title)
    query_entity_overlap = query_overlap_count(query, entities)
    query_triple_overlap = query_overlap_count(query, triple_texts)

    if not triples:
        substrate_status = "openie_triples_missing"
    elif not title_entity_present and not title_triple_endpoint_present:
        substrate_status = "title_anchor_not_materialized"
    elif query_entity_overlap == 0 and query_triple_overlap == 0:
        substrate_status = "facts_present_but_query_disconnected"
    elif title_triple_endpoint_present or title_entity_present:
        substrate_status = "substrate_available_selection_or_ranking_loss"
    else:
        substrate_status = "substrate_ambiguous"

    return {
        "gold_in_top5": gold_index in set(retrieved_top5),
        "substrate_status": substrate_status,
        "openie_fact_count": len(triples),
        "openie_entity_count": len(entities),
        "title_entity_present": title_entity_present,
        "title_triple_endpoint_present": title_triple_endpoint_present,
        "query_entity_overlap": query_entity_overlap,
        "query_triple_overlap": query_triple_overlap,
        "openie_title": openie_title,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_dataset(
    *,
    dataset_name: str,
    report_path: Path,
    variant_name: str,
) -> Dict[str, Any]:
    report = load_json(report_path)
    openie_path = Path(str(report.get("analysis_openie_path", "")))
    openie_docs = load_json(openie_path).get("docs", []) if openie_path.exists() else []
    rows = report.get("variants", {}).get(variant_name)
    if rows is None:
        raise KeyError(f"Variant {variant_name!r} not found in {report_path}")

    status_counts: Counter[str] = Counter()
    missed_status_counts: Counter[str] = Counter()
    gold_rows = []
    query_any_missed_counts: Counter[str] = Counter()
    query_all_gold_top5 = 0

    for row in rows:
        query = str(row.get("question") or "")
        retrieved_top5 = [int(value) for value in row.get("retrieved_doc_indices_top5", []) or []]
        gold_indices = [int(value) for value in row.get("gold_doc_indices", []) or []]
        gold_titles = [str(value) for value in row.get("gold_doc_titles", []) or []]
        per_query_missed_statuses: Set[str] = set()
        per_query_all_top5 = True
        for offset, gold_index in enumerate(gold_indices):
            gold_title = gold_titles[offset] if offset < len(gold_titles) else ""
            openie_doc = openie_docs[gold_index] if 0 <= gold_index < len(openie_docs) else None
            gold_audit = classify_gold_substrate(
                query=query,
                gold_title=gold_title,
                openie_doc=openie_doc,
                retrieved_top5=retrieved_top5,
                gold_index=gold_index,
            )
            status = str(gold_audit["substrate_status"])
            status_counts[status] += 1
            if not gold_audit["gold_in_top5"]:
                missed_status_counts[status] += 1
                per_query_missed_statuses.add(status)
                per_query_all_top5 = False
            gold_rows.append(
                {
                    "query_index": row.get("query_index"),
                    "query": query,
                    "gold_index": gold_index,
                    "gold_title": gold_title,
                    **gold_audit,
                }
            )
        if per_query_all_top5:
            query_all_gold_top5 += 1
        for status in per_query_missed_statuses:
            query_any_missed_counts[status] += 1

    return {
        "dataset": dataset_name,
        "report_path": str(report_path),
        "openie_path": str(openie_path),
        "variant_name": variant_name,
        "scope": {
            "queries": len(rows),
            "queries_all_gold_top5": query_all_gold_top5,
            "gold_rows": len(gold_rows),
            "missed_gold_rows": sum(missed_status_counts.values()),
        },
        "gold_substrate_status_counts": dict(status_counts),
        "missed_gold_substrate_status_counts": dict(missed_status_counts),
        "query_any_missed_status_counts": dict(query_any_missed_counts),
        "rows": gold_rows,
        "examples_head": gold_rows[:80],
    }


def write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Upstream Substrate Failure Audit",
        "",
        "This diagnostic reads existing retrieval reports and OpenIE caches. It does not run retrieval, QA, reranking, gates, or graph search.",
        "",
        f"- Variant: `{summary.get('variant_name')}`",
        "",
        "## Scope",
        "",
        "| dataset | queries | all-gold top5 queries | gold rows | missed gold rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in summary.get("datasets", []) or []:
        scope = dataset["scope"]
        lines.append(
            f"| {dataset['dataset']} | {scope['queries']} | {scope['queries_all_gold_top5']} | "
            f"{scope['gold_rows']} | {scope['missed_gold_rows']} |"
        )

    lines.extend(["", "## Missed Gold Rows By Upstream Status", "", "| dataset | status | count |", "| --- | --- | ---: |"])
    for dataset in summary.get("datasets", []) or []:
        for status, count in sorted(dataset["missed_gold_substrate_status_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {dataset['dataset']} | `{status}` | {count} |")

    lines.extend(["", "## Query-Level Missed Status", "", "| dataset | status | queries |", "| --- | --- | ---: |"])
    for dataset in summary.get("datasets", []) or []:
        for status, count in sorted(dataset["query_any_missed_status_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {dataset['dataset']} | `{status}` | {count} |")

    lines.extend(["", "## Examples", "", "| dataset | qid | gold | in top5 | status | facts | title entity | title endpoint | query overlap | question |", "| --- | ---: | --- | --- | --- | ---: | --- | --- | --- | --- |"])
    for dataset in summary.get("datasets", []) or []:
        examples = [row for row in dataset.get("examples_head", []) if not row.get("gold_in_top5")]
        for row in examples[:30]:
            question = " ".join(str(row.get("query", "")).split())
            if len(question) > 110:
                question = question[:107] + "..."
            overlap = f"ent={row.get('query_entity_overlap')},fact={row.get('query_triple_overlap')}"
            lines.append(
                f"| {dataset['dataset']} | {row.get('query_index')} | {row.get('gold_index')}:{row.get('gold_title')} | "
                f"{row.get('gold_in_top5')} | `{row.get('substrate_status')}` | {row.get('openie_fact_count')} | "
                f"{row.get('title_entity_present')} | {row.get('title_triple_endpoint_present')} | {overlap} | {question} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- `openie_triples_missing` and `title_anchor_not_materialized` point to construction-side fixes.",
            "- `facts_present_but_query_disconnected` points to query-to-fact activation/scoring.",
            "- `substrate_available_selection_or_ranking_loss` means the upstream object exists; downstream selection or reader-facing evidence composition is more likely the bottleneck.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_upstream_substrate_failures(
    *,
    report_paths: Mapping[str, Path],
    variant_name: str,
    output_json_path: Path,
    output_md_path: Path,
) -> Dict[str, Any]:
    datasets = [
        analyze_dataset(dataset_name=dataset, report_path=path, variant_name=variant_name)
        for dataset, path in report_paths.items()
    ]
    summary = {
        "variant_name": variant_name,
        "datasets": datasets,
    }
    output_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, output_md_path)
    return summary


def parse_report_arg(values: Sequence[str], root: Path) -> Dict[str, Path]:
    if not values:
        return {dataset: (root / path).resolve() for dataset, path in DEFAULT_REPORTS.items()}
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=PATH, got {value!r}")
        dataset, path_text = value.split("=", 1)
        result[dataset] = Path(path_text).resolve()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit upstream substrate status for missed gold passages.")
    parser.add_argument("--report", action="append", default=[], help="Dataset report path as DATASET=PATH. Defaults to the 20260423 anchor source audit reports.")
    parser.add_argument("--variant", default=SFB_VARIANT)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parent
    summary = analyze_upstream_substrate_failures(
        report_paths=parse_report_arg(args.report, root),
        variant_name=str(args.variant),
        output_json_path=Path(args.output_json).resolve(),
        output_md_path=Path(args.output_md).resolve(),
    )
    compact = {
        dataset["dataset"]: {
            "scope": dataset["scope"],
            "missed_gold_substrate_status_counts": dataset["missed_gold_substrate_status_counts"],
            "query_any_missed_status_counts": dataset["query_any_missed_status_counts"],
        }
        for dataset in summary["datasets"]
    }
    print(json.dumps(compact, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
