#!/usr/bin/env python3
"""Analyze 2Wiki retrieval ceiling from an existing eval report.

This script is intentionally 2Wiki-specific. It combines three cheap checks:

1. Oracle ceiling:
   - baseline support-doc recall/full-support at top-k
   - trivial upper bound if all missing gold docs were injected
   - realistic upper bound if one missing gold doc were injected
   - realistic upper bound if one *bridgeable* missing gold doc were injected

2. Bridge feasibility:
   - among queries with partial support coverage, how often does a missing gold
     support doc have a short bridge path from an already-found gold doc?

3. Relation inventory:
   - frequency of gold evidence relations
   - frequency of gold relation-pair patterns
   - frequency of bridge relation paths observed in bridgeable cases

Expected input report shape matches scripts/eval_causal_qwen3.py outputs where
examples[i].retrieval_trace contains dense_context_doc_ids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.hipporag.utils.eval_utils import normalize_answer
from src.hipporag.utils.misc_utils import compute_mdhash_id


def _normalize_node(text: str) -> str:
    return normalize_answer(str(text))


def _normalize_relation(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _build_chunk_id_to_title(corpus: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    chunk_id_to_title: dict[str, str] = {}
    chunk_id_to_doc: dict[str, str] = {}
    for row in corpus:
        doc_text = f"{row['title']}\n{row['text']}"
        chunk_id = compute_mdhash_id(doc_text, prefix="chunk-")
        chunk_id_to_title[chunk_id] = row["title"]
        chunk_id_to_doc[chunk_id] = doc_text
    return chunk_id_to_title, chunk_id_to_doc


def _extract_dense_titles(example: dict, chunk_id_to_title: dict[str, str]) -> list[str]:
    trace = example.get("retrieval_trace") or {}
    dense_doc_ids = trace.get("dense_context_doc_ids") or []
    titles: list[str] = []
    for doc_id in dense_doc_ids:
        if doc_id in chunk_id_to_title:
            titles.append(chunk_id_to_title[doc_id])
    return titles


def _build_evidence_graph(sample: dict) -> tuple[dict[str, list[tuple[str, str]]], dict[str, Counter], list[dict]]:
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    surface_forms: dict[str, Counter] = defaultdict(Counter)
    triples: list[dict] = []

    for triple in sample.get("evidences", []):
        if not isinstance(triple, list) or len(triple) != 3:
            continue
        subj, rel, obj = triple
        subj_norm = _normalize_node(subj)
        obj_norm = _normalize_node(obj)
        rel_norm = _normalize_relation(rel)
        adjacency[subj_norm].append((obj_norm, rel_norm))
        adjacency[obj_norm].append((subj_norm, rel_norm))
        surface_forms[subj_norm][str(subj)] += 1
        surface_forms[obj_norm][str(obj)] += 1
        triples.append(
            {
                "subject": str(subj),
                "relation": rel_norm,
                "object": str(obj),
                "subject_norm": subj_norm,
                "object_norm": obj_norm,
            }
        )

    return adjacency, surface_forms, triples


def _best_surface(surface_forms: dict[str, Counter], node_norm: str) -> str:
    counter = surface_forms.get(node_norm)
    if not counter:
        return node_norm
    return counter.most_common(1)[0][0]


def _find_bridge_path(
    found_titles: list[str],
    missing_title: str,
    adjacency: dict[str, list[tuple[str, str]]],
    surface_forms: dict[str, Counter],
) -> dict[str, Any] | None:
    missing_norm = _normalize_node(missing_title)
    for found_title in found_titles:
        found_norm = _normalize_node(found_title)
        if not found_norm or not missing_norm:
            continue

        for neighbor_norm, rel1 in adjacency.get(found_norm, []):
            if neighbor_norm == missing_norm:
                return {
                    "found_title": found_title,
                    "missing_title": missing_title,
                    "bridge_hops": 1,
                    "bridge_entity": None,
                    "relation_path": [rel1],
                }

        for bridge_norm, rel1 in adjacency.get(found_norm, []):
            for neighbor_norm, rel2 in adjacency.get(bridge_norm, []):
                if neighbor_norm != missing_norm:
                    continue
                return {
                    "found_title": found_title,
                    "missing_title": missing_title,
                    "bridge_hops": 2,
                    "bridge_entity": _best_surface(surface_forms, bridge_norm),
                    "relation_path": [rel1, rel2],
                }
    return None


def _support_recall(retrieved_titles: list[str], gold_titles: list[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    retrieved_set = set(retrieved_titles[:k])
    return len(retrieved_set & set(gold_titles)) / len(set(gold_titles))


def _full_support(retrieved_titles: list[str], gold_titles: list[str], k: int) -> float:
    return 1.0 if set(gold_titles).issubset(set(retrieved_titles[:k])) else 0.0


def _oracle_recall_inject_one(retrieved_titles: list[str], gold_titles: list[str], k: int) -> float:
    gold_set = set(gold_titles)
    found = len(set(retrieved_titles[:k]) & gold_set)
    if not gold_set:
        return 0.0
    if found >= len(gold_set):
        return 1.0
    missing = len(gold_set) - found
    inject_capacity = max(0, k - found)
    recovered = found + min(1, missing, inject_capacity)
    return recovered / len(gold_set)


def _oracle_full_support_inject_one(retrieved_titles: list[str], gold_titles: list[str], k: int) -> float:
    return 1.0 if _oracle_recall_inject_one(retrieved_titles, gold_titles, k) >= 1.0 else 0.0


def _oracle_recall_inject_all(retrieved_titles: list[str], gold_titles: list[str], k: int) -> float:
    gold_set = set(gold_titles)
    if not gold_set:
        return 0.0
    found = len(set(retrieved_titles[:k]) & gold_set)
    missing = len(gold_set) - found
    inject_capacity = max(0, k - found)
    recovered = found + min(missing, inject_capacity)
    return recovered / len(gold_set)


def _oracle_full_support_inject_all(retrieved_titles: list[str], gold_titles: list[str], k: int) -> float:
    return 1.0 if _oracle_recall_inject_all(retrieved_titles, gold_titles, k) >= 1.0 else 0.0


def _oracle_recall_inject_bridgeable(
    retrieved_titles: list[str],
    gold_titles: list[str],
    bridgeable_missing_titles: list[str],
    k: int,
) -> float:
    gold_set = set(gold_titles)
    found = len(set(retrieved_titles[:k]) & gold_set)
    if not gold_set:
        return 0.0
    if found >= len(gold_set):
        return 1.0
    if not bridgeable_missing_titles:
        return found / len(gold_set)
    missing = len(gold_set) - found
    inject_capacity = max(0, k - found)
    recovered = found + min(1, missing, inject_capacity)
    return recovered / len(gold_set)


def _oracle_full_support_inject_bridgeable(
    retrieved_titles: list[str],
    gold_titles: list[str],
    bridgeable_missing_titles: list[str],
    k: int,
) -> float:
    return 1.0 if _oracle_recall_inject_bridgeable(retrieved_titles, gold_titles, bridgeable_missing_titles, k) >= 1.0 else 0.0


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 2Wiki oracle support-doc ceiling from an eval report.")
    parser.add_argument(
        "--report",
        required=True,
        help="Path to an eval JSON report produced by scripts/eval_causal_qwen3.py.",
    )
    parser.add_argument(
        "--dataset_path",
        default="reproduce/dataset/2wikimultihopqa.json",
        help="Path to 2Wiki samples JSON.",
    )
    parser.add_argument(
        "--corpus_path",
        default="reproduce/dataset/2wikimultihopqa_corpus.json",
        help="Path to 2Wiki corpus JSON.",
    )
    parser.add_argument(
        "--top_ks",
        default="2,5",
        help="Comma-separated top-k values to analyze. Default: 2,5",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Path to write analysis JSON. Default: <report>.oracle_analysis.json",
    )
    parser.add_argument(
        "--output_md",
        default=None,
        help="Path to write markdown summary. Default: <report>.oracle_analysis.md",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20,
        help="Max per-query case rows to keep in the output summary buckets.",
    )
    args = parser.parse_args()

    report = _load_json(args.report)
    samples = _load_json(args.dataset_path)
    corpus = _load_json(args.corpus_path)
    examples = report.get("examples") or []
    num_queries = len(examples)
    samples = samples[:num_queries]

    if len(samples) != num_queries:
        raise ValueError(f"Sample/report length mismatch: {len(samples)} vs {num_queries}")

    chunk_id_to_title, _ = _build_chunk_id_to_title(corpus)
    top_ks = [int(item.strip()) for item in args.top_ks.split(",") if item.strip()]

    relation_counts: Counter[str] = Counter()
    relation_pair_counts: Counter[str] = Counter()
    bridge_relation_path_counts: Counter[str] = Counter()

    metrics: dict[str, dict[str, float]] = {
        "baseline_recall": {f"Recall@{k}": 0.0 for k in top_ks},
        "baseline_full_support": {f"FullSupport@{k}": 0.0 for k in top_ks},
        "oracle_all_missing_recall": {f"Recall@{k}": 0.0 for k in top_ks},
        "oracle_all_missing_full_support": {f"FullSupport@{k}": 0.0 for k in top_ks},
        "oracle_inject_one_recall": {f"Recall@{k}": 0.0 for k in top_ks},
        "oracle_inject_one_full_support": {f"FullSupport@{k}": 0.0 for k in top_ks},
        "oracle_bridgeable_recall": {f"Recall@{k}": 0.0 for k in top_ks},
        "oracle_bridgeable_full_support": {f"FullSupport@{k}": 0.0 for k in top_ks},
    }

    feasibility = {
        "queries_with_partial_support_top5": 0,
        "queries_with_any_missing_support_top5": 0,
        "queries_with_bridgeable_missing_support_top5": 0,
        "missing_support_docs_top5": 0,
        "bridgeable_missing_support_docs_top5": 0,
        "direct_bridge_count_top5": 0,
        "two_hop_bridge_count_top5": 0,
    }

    per_query_rows: list[dict[str, Any]] = []

    for idx, (sample, example) in enumerate(zip(samples, examples)):
        if sample.get("question") != example.get("question"):
            raise ValueError(
                f"Question alignment mismatch at idx={idx}: "
                f"{sample.get('question')} != {example.get('question')}"
            )

        gold_titles = sorted({item[0] for item in sample.get("supporting_facts", [])})
        retrieved_titles = _extract_dense_titles(example, chunk_id_to_title)

        adjacency, surface_forms, triples = _build_evidence_graph(sample)
        relation_counts.update(
            _normalize_relation(triple[1])
            for triple in sample.get("evidences", [])
            if isinstance(triple, list) and len(triple) == 3
        )
        if sample.get("evidences"):
            relation_pair_counts[" -> ".join(_normalize_relation(triple[1]) for triple in sample["evidences"] if len(triple) == 3)] += 1

        found_top5 = sorted(set(retrieved_titles[:5]) & set(gold_titles))
        missing_top5 = sorted(set(gold_titles) - set(retrieved_titles[:5]))
        bridgeable_rows: list[dict[str, Any]] = []
        for missing_title in missing_top5:
            bridge_row = _find_bridge_path(found_top5, missing_title, adjacency, surface_forms)
            if bridge_row is not None:
                bridgeable_rows.append(bridge_row)
                relation_path = " -> ".join(bridge_row["relation_path"])
                bridge_relation_path_counts[relation_path] += 1
                if bridge_row["bridge_hops"] == 1:
                    feasibility["direct_bridge_count_top5"] += 1
                elif bridge_row["bridge_hops"] == 2:
                    feasibility["two_hop_bridge_count_top5"] += 1

        if missing_top5:
            feasibility["queries_with_any_missing_support_top5"] += 1
            feasibility["missing_support_docs_top5"] += len(missing_top5)
        if found_top5 and missing_top5:
            feasibility["queries_with_partial_support_top5"] += 1
        if bridgeable_rows:
            feasibility["queries_with_bridgeable_missing_support_top5"] += 1
            feasibility["bridgeable_missing_support_docs_top5"] += len(bridgeable_rows)

        bridgeable_missing_titles = sorted({row["missing_title"] for row in bridgeable_rows})

        for k in top_ks:
            metrics["baseline_recall"][f"Recall@{k}"] += _support_recall(retrieved_titles, gold_titles, k)
            metrics["baseline_full_support"][f"FullSupport@{k}"] += _full_support(retrieved_titles, gold_titles, k)
            metrics["oracle_all_missing_recall"][f"Recall@{k}"] += _oracle_recall_inject_all(retrieved_titles, gold_titles, k)
            metrics["oracle_all_missing_full_support"][f"FullSupport@{k}"] += _oracle_full_support_inject_all(retrieved_titles, gold_titles, k)
            metrics["oracle_inject_one_recall"][f"Recall@{k}"] += _oracle_recall_inject_one(retrieved_titles, gold_titles, k)
            metrics["oracle_inject_one_full_support"][f"FullSupport@{k}"] += _oracle_full_support_inject_one(retrieved_titles, gold_titles, k)
            metrics["oracle_bridgeable_recall"][f"Recall@{k}"] += _oracle_recall_inject_bridgeable(
                retrieved_titles, gold_titles, bridgeable_missing_titles, k
            )
            metrics["oracle_bridgeable_full_support"][f"FullSupport@{k}"] += _oracle_full_support_inject_bridgeable(
                retrieved_titles, gold_titles, bridgeable_missing_titles, k
            )

        per_query_rows.append(
            {
                "idx": idx,
                "question": sample["question"],
                "gold_support_titles": gold_titles,
                "retrieved_top5_titles": retrieved_titles[:5],
                "found_support_top5": found_top5,
                "missing_support_top5": missing_top5,
                "bridgeable_missing_support_top5": bridgeable_missing_titles,
                "bridge_rows": bridgeable_rows,
                "gold_evidences": sample.get("evidences", []),
            }
        )

    for family in metrics.values():
        for key in list(family.keys()):
            family[key] = round(family[key] / max(1, num_queries), 4)

    summary = {
        "report": str(Path(args.report)),
        "dataset_path": str(Path(args.dataset_path)),
        "corpus_path": str(Path(args.corpus_path)),
        "num_queries": num_queries,
        "metrics": metrics,
        "feasibility": {
            **feasibility,
            "bridgeable_missing_doc_rate_top5": round(
                feasibility["bridgeable_missing_support_docs_top5"] / max(1, feasibility["missing_support_docs_top5"]),
                4,
            ),
            "partial_query_bridgeable_rate_top5": round(
                feasibility["queries_with_bridgeable_missing_support_top5"] / max(1, feasibility["queries_with_partial_support_top5"]),
                4,
            ),
        },
        "relation_inventory": {
            "relation_counts": dict(relation_counts.most_common()),
            "relation_pair_counts": dict(relation_pair_counts.most_common()),
            "bridge_relation_path_counts": dict(bridge_relation_path_counts.most_common()),
        },
        "examples": {
            "bridgeable_partial_queries_top5": [
                row
                for row in per_query_rows
                if row["found_support_top5"] and row["bridgeable_missing_support_top5"]
            ][: args.max_examples],
            "nonbridgeable_partial_queries_top5": [
                row
                for row in per_query_rows
                if row["found_support_top5"] and row["missing_support_top5"] and not row["bridgeable_missing_support_top5"]
            ][: args.max_examples],
        },
    }

    report_path = Path(args.report)
    output_json = Path(args.output_json) if args.output_json else report_path.with_suffix(report_path.suffix + ".oracle_analysis.json")
    output_md = Path(args.output_md) if args.output_md else report_path.with_suffix(report_path.suffix + ".oracle_analysis.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        "# 2Wiki Oracle Ceiling Analysis",
        "",
        f"- Report: `{report_path}`",
        f"- Queries: `{num_queries}`",
        "",
        "## Metrics",
        "",
    ]
    for metric_family, metric_values in metrics.items():
        lines.append(f"### {metric_family}")
        lines.append("")
        for key, value in metric_values.items():
            lines.append(f"- `{key}`: {value:.4f}")
        lines.append("")

    lines.extend(
        [
            "## Feasibility",
            "",
            f"- `queries_with_partial_support_top5`: {feasibility['queries_with_partial_support_top5']}",
            f"- `queries_with_bridgeable_missing_support_top5`: {feasibility['queries_with_bridgeable_missing_support_top5']}",
            f"- `missing_support_docs_top5`: {feasibility['missing_support_docs_top5']}",
            f"- `bridgeable_missing_support_docs_top5`: {feasibility['bridgeable_missing_support_docs_top5']}",
            f"- `bridgeable_missing_doc_rate_top5`: {_format_pct(summary['feasibility']['bridgeable_missing_doc_rate_top5'])}",
            f"- `partial_query_bridgeable_rate_top5`: {_format_pct(summary['feasibility']['partial_query_bridgeable_rate_top5'])}",
            f"- `direct_bridge_count_top5`: {feasibility['direct_bridge_count_top5']}",
            f"- `two_hop_bridge_count_top5`: {feasibility['two_hop_bridge_count_top5']}",
            "",
            "## Top Relation Counts",
            "",
        ]
    )

    for relation, count in relation_counts.most_common(20):
        lines.append(f"- `{relation}`: {count}")

    lines.extend(["", "## Top Bridge Relation Paths", ""])
    for relation_path, count in bridge_relation_path_counts.most_common(20):
        lines.append(f"- `{relation_path}`: {count}")

    output_md.write_text("\n".join(lines))

    print(json.dumps(
        {
            "output_json": str(output_json),
            "output_md": str(output_md),
            "metrics": metrics,
            "feasibility": summary["feasibility"],
            "top_relations": relation_counts.most_common(10),
            "top_bridge_relation_paths": bridge_relation_path_counts.most_common(10),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
