#!/usr/bin/env python3
"""Build obligation-closed STO evidence units.

This is a construction-time materializer. It consumes Source/Title/OpenIE
units and emits first-class connected evidence subgraphs. It deliberately does
not consume queries, retrieval outputs, QA labels, SFB traces, PPR scores, or
runtime completion decisions.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from build_source_title_openie_substrate import is_weak_endpoint, normalize_text, stable_id


CLOSED_STO_SCHEMA_VERSION = "obligation_closed_sto_unit.v0"

GENERIC_ENDPOINTS = {
    "album",
    "award",
    "band",
    "battle",
    "book",
    "city",
    "country",
    "film",
    "group",
    "person",
    "place",
    "play",
    "race",
    "river",
    "school",
    "series",
    "song",
    "state",
    "team",
    "town",
    "village",
}


def unique_ordered(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen: Set[str] = set()
    for value in values:
        key = str(value)
        if not key or key in seen:
            continue
        result.append(value)
        seen.add(key)
    return result


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def usable_endpoint(value: Any) -> bool:
    endpoint = normalize_text(value)
    if not endpoint:
        return False
    if endpoint in GENERIC_ENDPOINTS:
        return False
    if is_weak_endpoint(endpoint):
        return False
    if endpoint.isdigit():
        return False
    tokens = endpoint.split()
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return False
    return True


def endpoint_key(value: Any) -> str:
    return normalize_text(value)


def fact_units(units: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    clean_units: List[Dict[str, Any]] = []
    for unit in units:
        if str(unit.get("unit_type") or "") != "openie_fact":
            continue
        if not str(unit.get("unit_id") or "").strip():
            continue
        clean_units.append(dict(unit))
    return sorted(clean_units, key=lambda unit: (safe_int(unit.get("doc_index")), str(unit.get("unit_id") or "")))


def unit_endpoint_entries(unit: Mapping[str, Any]) -> List[Tuple[str, str]]:
    values: List[Any] = []
    values.extend(unit.get("endpoint_entities", []) or [])
    values.extend(
        [
            unit.get("grounded_subject"),
            unit.get("grounded_object"),
            unit.get("subject"),
            unit.get("object"),
        ]
    )
    entries: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for value in values:
        key = endpoint_key(value)
        if not key or key in seen or not usable_endpoint(value):
            continue
        entries.append((key, str(value)))
        seen.add(key)
    return entries


def build_endpoint_index(
    units: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, int]]:
    endpoint_to_unit_ids: Dict[str, List[str]] = defaultdict(list)
    endpoint_labels: Dict[str, str] = {}
    endpoint_docs: Dict[str, Set[int]] = defaultdict(set)

    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        doc_index = safe_int(unit.get("doc_index"))
        for key, label in unit_endpoint_entries(unit):
            endpoint_to_unit_ids[key].append(unit_id)
            endpoint_docs[key].add(doc_index)
            endpoint_labels.setdefault(key, label)

    endpoint_to_unit_ids = {
        key: sorted(unique_ordered(unit_ids))
        for key, unit_ids in endpoint_to_unit_ids.items()
    }
    endpoint_doc_degrees = {key: len(doc_indices) for key, doc_indices in endpoint_docs.items()}
    return endpoint_to_unit_ids, endpoint_labels, endpoint_doc_degrees


def build_endpoint_adjacency(
    units_by_id: Mapping[str, Mapping[str, Any]],
    endpoint_to_unit_ids: Mapping[str, Sequence[str]],
    endpoint_doc_degrees: Mapping[str, int],
    *,
    max_endpoint_degree: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Counter]:
    adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()
    edge_keys: Set[Tuple[str, str, str]] = set()

    for key, unit_ids in sorted(endpoint_to_unit_ids.items()):
        doc_degree = int(endpoint_doc_degrees.get(key, 0))
        if doc_degree < 2:
            stats["single_doc_endpoint_skipped"] += 1
            continue
        if doc_degree > max_endpoint_degree:
            stats["hub_endpoint_skipped"] += 1
            continue
        clean_ids = [unit_id for unit_id in unit_ids if unit_id in units_by_id]
        if len(clean_ids) < 2:
            continue
        stats["usable_endpoint_count"] += 1
        for left_index, left_id in enumerate(clean_ids):
            for right_id in clean_ids[left_index + 1 :]:
                left_doc = safe_int(units_by_id[left_id].get("doc_index"))
                right_doc = safe_int(units_by_id[right_id].get("doc_index"))
                if left_doc == right_doc:
                    stats["same_doc_endpoint_pair_skipped"] += 1
                    continue
                edge_key = (left_id, right_id, key)
                if edge_key in edge_keys:
                    continue
                edge_keys.add(edge_key)
                adjacency[left_id].append(
                    {"unit_id": right_id, "edge_kind": "endpoint_transfer", "endpoint_key": key}
                )
                adjacency[right_id].append(
                    {"unit_id": left_id, "edge_kind": "endpoint_transfer", "endpoint_key": key}
                )
                stats["endpoint_edge_count"] += 1

    transfer_bearing_unit_ids: Set[str] = set()
    for key, unit_ids in endpoint_to_unit_ids.items():
        doc_degree = int(endpoint_doc_degrees.get(key, 0))
        if doc_degree < 2 or doc_degree > max_endpoint_degree:
            continue
        transfer_bearing_unit_ids.update(str(unit_id) for unit_id in unit_ids if str(unit_id) in units_by_id)

    doc_to_unit_ids: Dict[int, List[str]] = defaultdict(list)
    for unit_id, unit in units_by_id.items():
        if unit_id not in transfer_bearing_unit_ids:
            stats["non_transfer_fact_excluded_from_source_edge_count"] += 1
            continue
        doc_to_unit_ids[safe_int(unit.get("doc_index"))].append(unit_id)
    for doc_index, unit_ids in sorted(doc_to_unit_ids.items()):
        clean_ids = sorted(unique_ordered(unit_ids))
        if len(clean_ids) < 2:
            continue
        stats["source_doc_with_internal_edges_count"] += 1
        for left_index, left_id in enumerate(clean_ids):
            for right_id in clean_ids[left_index + 1 :]:
                source_edge_key = (left_id, right_id, f"source:{doc_index}")
                if source_edge_key in edge_keys:
                    continue
                edge_keys.add(source_edge_key)
                adjacency[left_id].append(
                    {
                        "unit_id": right_id,
                        "edge_kind": "source_co_grounding",
                        "source_doc_index": doc_index,
                    }
                )
                adjacency[right_id].append(
                    {
                        "unit_id": left_id,
                        "edge_kind": "source_co_grounding",
                        "source_doc_index": doc_index,
                    }
                )
                stats["source_edge_count"] += 1

    for unit_id in list(adjacency):
        adjacency[unit_id] = sorted(
            adjacency[unit_id],
            key=lambda edge: (
                str(edge.get("edge_kind") or ""),
                str(edge.get("endpoint_key") or edge.get("source_doc_index") or ""),
                str(edge.get("unit_id") or ""),
            ),
        )
    return adjacency, stats


def shortest_endpoint_paths_from(
    source_unit_id: str,
    adjacency: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_path_edges: int,
) -> Dict[str, Dict[str, Any]]:
    paths: Dict[str, Dict[str, Any]] = {}
    best_cost: Dict[str, Tuple[int, int]] = {source_unit_id: (0, 0)}
    counter = itertools.count()
    heap: List[Tuple[int, int, int, str, List[str], List[str], List[Dict[str, Any]]]] = [
        (0, 0, next(counter), source_unit_id, [source_unit_id], [], [])
    ]

    while heap:
        endpoint_count, total_edge_count, _sequence, current_id, path_unit_ids, path_endpoint_keys, path_edges = heapq.heappop(heap)
        if best_cost.get(current_id) != (endpoint_count, total_edge_count):
            continue
        for edge in adjacency.get(current_id, []) or []:
            next_id = str(edge.get("unit_id") or "")
            if not next_id or next_id in path_unit_ids:
                continue
            edge_kind = str(edge.get("edge_kind") or "endpoint_transfer")
            is_endpoint_edge = edge_kind == "endpoint_transfer"
            next_endpoint_count = endpoint_count + (1 if is_endpoint_edge else 0)
            if next_endpoint_count > max_path_edges:
                continue
            next_total_edge_count = total_edge_count + 1
            next_cost = (next_endpoint_count, next_total_edge_count)
            if best_cost.get(next_id, (max_path_edges + 1, max_path_edges + len(adjacency) + 1)) <= next_cost:
                continue
            endpoint = str(edge.get("endpoint_key") or "") if is_endpoint_edge else ""
            next_path_unit_ids = path_unit_ids + [next_id]
            next_endpoint_keys = path_endpoint_keys + ([endpoint] if endpoint else [])
            next_edge = {
                "edge_kind": edge_kind,
                "from_unit_id": current_id,
                "to_unit_id": next_id,
            }
            if is_endpoint_edge:
                next_edge["endpoint_key"] = endpoint
            else:
                next_edge["source_doc_index"] = safe_int(edge.get("source_doc_index"))
            next_path_edges = path_edges + [next_edge]
            best_cost[next_id] = next_cost
            paths[next_id] = {
                "path_unit_ids": next_path_unit_ids,
                "path_endpoint_keys": next_endpoint_keys,
                "path_edges": next_path_edges,
            }
            heapq.heappush(
                heap,
                (
                    next_endpoint_count,
                    next_total_edge_count,
                    next(counter),
                    next_id,
                    next_path_unit_ids,
                    next_endpoint_keys,
                    next_path_edges,
                ),
            )
    return paths


def relation_role(unit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "unit_id": str(unit.get("unit_id") or ""),
        "fact_id": unit.get("fact_id"),
        "doc_index": safe_int(unit.get("doc_index")),
        "title": str(unit.get("title") or ""),
        "subject": unit.get("subject"),
        "relation": unit.get("relation"),
        "object": unit.get("object"),
    }


def build_closed_unit(
    *,
    source_unit: Mapping[str, Any],
    target_unit: Mapping[str, Any],
    path_unit_ids: Sequence[str],
    path_endpoint_keys: Sequence[str],
    path_edges: Sequence[Mapping[str, Any]],
    units_by_id: Mapping[str, Mapping[str, Any]],
    endpoint_labels: Mapping[str, str],
    max_path_edges: int,
    max_endpoint_degree: int,
) -> Dict[str, Any]:
    path_units = [units_by_id[unit_id] for unit_id in path_unit_ids]
    emitted_doc_indices = unique_ordered(safe_int(unit.get("doc_index")) for unit in path_units)
    emitted_doc_set = set(emitted_doc_indices)
    path_fact_ids = unique_ordered(unit.get("fact_id") for unit in path_units if unit.get("fact_id"))
    path_titles = unique_ordered(str(unit.get("title") or "") for unit in path_units if str(unit.get("title") or ""))
    path_endpoint_labels = [str(endpoint_labels.get(key) or key) for key in path_endpoint_keys]
    emitted_doc_unit_ids: Dict[str, List[str]] = {}
    emitted_doc_fact_ids: Dict[str, List[Any]] = {}
    for doc_index in emitted_doc_indices:
        doc_units = [
            unit
            for unit in units_by_id.values()
            if safe_int(unit.get("doc_index")) == safe_int(doc_index) and safe_int(doc_index) in emitted_doc_set
        ]
        emitted_doc_unit_ids[str(doc_index)] = unique_ordered(str(unit.get("unit_id") or "") for unit in doc_units)
        emitted_doc_fact_ids[str(doc_index)] = unique_ordered(unit.get("fact_id") for unit in doc_units if unit.get("fact_id"))
    source_edge_count = sum(1 for edge in path_edges if str(edge.get("edge_kind") or "") == "source_co_grounding")
    closed_obligation_type = (
        "direct_variable_transfer"
        if len(path_endpoint_keys) == 1 and source_edge_count == 0
        else "variable_transfer_path"
    )
    unit_id = stable_id(
        "sto-closed",
        [
            source_unit.get("unit_id"),
            target_unit.get("unit_id"),
            "|".join(path_unit_ids),
            "|".join(path_endpoint_keys),
        ],
    )
    return {
        "closed_evidence_unit_id": unit_id,
        "unit_type": "obligation_closed_sto_unit",
        "schema_version": CLOSED_STO_SCHEMA_VERSION,
        "construction_origin": "index_time_sto_minimal_endpoint_path",
        "closed_obligation_type": closed_obligation_type,
        "terminal_unit_id": str(source_unit.get("unit_id") or ""),
        "anchor_unit_id": str(target_unit.get("unit_id") or ""),
        "terminal_fact_id": source_unit.get("fact_id"),
        "anchor_fact_id": target_unit.get("fact_id"),
        "terminal_doc_index": safe_int(source_unit.get("doc_index")),
        "anchor_doc_index": safe_int(target_unit.get("doc_index")),
        "boundary_unit_ids": [str(source_unit.get("unit_id") or ""), str(target_unit.get("unit_id") or "")],
        "path_unit_ids": list(path_unit_ids),
        "path_edges": [dict(edge) for edge in path_edges],
        "path_fact_ids": path_fact_ids,
        "path_doc_indices": emitted_doc_indices,
        "path_titles": path_titles,
        "path_endpoint_keys": list(path_endpoint_keys),
        "path_endpoints": path_endpoint_labels,
        "relation_roles": [relation_role(unit) for unit in path_units],
        "emitted_doc_indices": emitted_doc_indices,
        "emitted_doc_unit_ids": emitted_doc_unit_ids,
        "emitted_doc_fact_ids": emitted_doc_fact_ids,
        "path_edge_count": len(path_endpoint_keys),
        "sto_edge_count": len(path_edges),
        "source_edge_count": source_edge_count,
        "max_path_edges": int(max_path_edges),
        "endpoint_degree_cap": int(max_endpoint_degree),
    }


def closed_unit_canonical_id(unit: Mapping[str, Any]) -> str:
    return stable_id(
        "sto-closed",
        [
            "directed-doc-endpoint-path",
            unit.get("terminal_doc_index"),
            unit.get("anchor_doc_index"),
            "|".join(str(value) for value in unit.get("path_doc_indices", []) or []),
            "|".join(str(value) for value in unit.get("path_endpoint_keys", []) or []),
        ],
    )


def closed_unit_preference_key(unit: Mapping[str, Any]) -> Tuple[int, int, int, str]:
    return (
        safe_int(unit.get("path_edge_count"), 0),
        safe_int(unit.get("source_edge_count"), 0),
        safe_int(unit.get("sto_edge_count"), 0),
        "|".join(str(value) for value in unit.get("path_unit_ids", []) or []),
    )


def build_obligation_closed_sto_unit_store(
    units: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str = "",
    max_path_edges: int = 3,
    max_endpoint_degree: int = 30,
) -> Dict[str, Any]:
    if max_path_edges < 1:
        raise ValueError("max_path_edges must be >= 1")
    if max_endpoint_degree < 2:
        raise ValueError("max_endpoint_degree must be >= 2")

    facts = fact_units(units)
    units_by_id = {str(unit.get("unit_id") or ""): unit for unit in facts}
    endpoint_to_unit_ids, endpoint_labels, endpoint_doc_degrees = build_endpoint_index(facts)
    adjacency, adjacency_stats = build_endpoint_adjacency(
        units_by_id,
        endpoint_to_unit_ids,
        endpoint_doc_degrees,
        max_endpoint_degree=max_endpoint_degree,
    )

    closed_units_by_id: Dict[str, Dict[str, Any]] = {}
    raw_closed_path_count = 0
    for source_id in sorted(units_by_id):
        source_unit = units_by_id[source_id]
        source_doc = safe_int(source_unit.get("doc_index"))
        paths = shortest_endpoint_paths_from(source_id, adjacency, max_path_edges=max_path_edges)
        for target_id, path in sorted(paths.items()):
            target_unit = units_by_id[target_id]
            target_doc = safe_int(target_unit.get("doc_index"))
            if source_doc == target_doc:
                continue
            closed_unit = build_closed_unit(
                source_unit=source_unit,
                target_unit=target_unit,
                path_unit_ids=path["path_unit_ids"],
                path_endpoint_keys=path["path_endpoint_keys"],
                path_edges=path["path_edges"],
                units_by_id=units_by_id,
                endpoint_labels=endpoint_labels,
                max_path_edges=max_path_edges,
                max_endpoint_degree=max_endpoint_degree,
            )
            raw_closed_path_count += 1
            canonical_id = closed_unit_canonical_id(closed_unit)
            closed_unit["closed_evidence_unit_id"] = canonical_id
            closed_unit["canonicalization"] = {
                "key": "directed_doc_path_plus_endpoint_transfers",
                "preference": "fewest_endpoint_transfers_then_fewest_source_edges_then_deterministic_path",
            }
            existing = closed_units_by_id.get(canonical_id)
            if existing is None or closed_unit_preference_key(closed_unit) < closed_unit_preference_key(existing):
                closed_units_by_id[canonical_id] = closed_unit

    closed_units = [closed_units_by_id[unit_id] for unit_id in sorted(closed_units_by_id)]
    type_counts = Counter(str(unit.get("closed_obligation_type") or "") for unit in closed_units)
    edge_count_by_depth = Counter(str(unit.get("path_edge_count") or 0) for unit in closed_units)
    phase_guard = {
        "uses_query_text": False,
        "uses_gold_or_target_labels": False,
        "uses_retrieval_or_qa_outcomes": False,
        "uses_sfb_outputs": False,
        "uses_ppr_scores": False,
        "uses_runtime_completion": False,
        "uses_weighted_late_fusion": False,
        "index_time_object_finalized": True,
    }
    return {
        "dataset": dataset_name,
        "schema": {
            "schema_version": CLOSED_STO_SCHEMA_VERSION,
            "object": "obligation_closed_sto_unit",
        },
        "inputs": {
            "max_path_edges": int(max_path_edges),
            "max_endpoint_degree": int(max_endpoint_degree),
        },
        "phase_guard": phase_guard,
        "summary": {
            "input_unit_count": len(units),
            "fact_unit_count": len(facts),
            "endpoint_count": len(endpoint_to_unit_ids),
            "usable_endpoint_count": int(adjacency_stats["usable_endpoint_count"]),
            "single_doc_endpoint_skipped_count": int(adjacency_stats["single_doc_endpoint_skipped"]),
            "hub_endpoint_skipped_count": int(adjacency_stats["hub_endpoint_skipped"]),
            "same_doc_endpoint_pair_skipped_count": int(adjacency_stats["same_doc_endpoint_pair_skipped"]),
            "endpoint_edge_count": int(adjacency_stats["endpoint_edge_count"]),
            "source_doc_with_internal_edges_count": int(adjacency_stats["source_doc_with_internal_edges_count"]),
            "source_edge_count": int(adjacency_stats["source_edge_count"]),
            "non_transfer_fact_excluded_from_source_edge_count": int(
                adjacency_stats["non_transfer_fact_excluded_from_source_edge_count"]
            ),
            "raw_closed_path_count": raw_closed_path_count,
            "deduplicated_closed_path_count": raw_closed_path_count - len(closed_units),
            "closed_unit_count": len(closed_units),
            "closed_unit_type_counts": dict(sorted(type_counts.items())),
            "closed_unit_edge_count_distribution": dict(sorted(edge_count_by_depth.items())),
        },
        "closed_units": closed_units,
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_closed_unit_store_markdown(store: Mapping[str, Any], path: Path) -> None:
    stats = store.get("summary", {}) or {}
    lines = [
        "# Obligation-Closed STO Unit Store",
        "",
        f"Dataset: `{store.get('dataset', '')}`",
        "",
        "This store is built at index time from Source/Title/OpenIE units only.",
        "It materializes connected evidence subgraphs before retrieval.",
        "",
        "## Guard",
        "",
        "- Uses query text: `False`",
        "- Uses gold/target labels: `False`",
        "- Uses retrieval or QA outcomes: `False`",
        "- Uses SFB outputs: `False`",
        "- Uses PPR scores: `False`",
        "- Uses runtime completion: `False`",
        "- Uses weighted late fusion: `False`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| input units | {stats.get('input_unit_count', 0)} |",
        f"| fact units | {stats.get('fact_unit_count', 0)} |",
        f"| endpoints | {stats.get('endpoint_count', 0)} |",
        f"| usable endpoints | {stats.get('usable_endpoint_count', 0)} |",
        f"| endpoint edges | {stats.get('endpoint_edge_count', 0)} |",
        f"| source edges | {stats.get('source_edge_count', 0)} |",
        f"| non-transfer facts excluded from source edges | {stats.get('non_transfer_fact_excluded_from_source_edge_count', 0)} |",
        f"| raw closed paths | {stats.get('raw_closed_path_count', 0)} |",
        f"| deduplicated closed paths | {stats.get('deduplicated_closed_path_count', 0)} |",
        f"| closed units | {stats.get('closed_unit_count', 0)} |",
        f"| hub endpoints skipped | {stats.get('hub_endpoint_skipped_count', 0)} |",
        "",
        "## Closed Unit Types",
        "",
        "| type | count |",
        "| --- | ---: |",
    ]
    for unit_type, count in (stats.get("closed_unit_type_counts", {}) or {}).items():
        lines.append(f"| `{unit_type}` | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build obligation-closed STO evidence units.")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--sto-units-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-path-edges", type=int, default=3)
    parser.add_argument("--max-endpoint-degree", type=int, default=30)
    args = parser.parse_args(list(argv) if argv is not None else None)

    store = build_obligation_closed_sto_unit_store(
        load_jsonl(Path(args.sto_units_jsonl)),
        dataset_name=str(args.dataset_name),
        max_path_edges=int(args.max_path_edges),
        max_endpoint_degree=int(args.max_endpoint_degree),
    )
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.write_text(json.dumps(store, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_closed_unit_store_markdown(store, output_md)
    print(json.dumps(store["summary"], ensure_ascii=True, sort_keys=True))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
