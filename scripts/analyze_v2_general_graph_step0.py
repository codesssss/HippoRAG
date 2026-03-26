#!/usr/bin/env python3
"""Probe V2 general graph quality before integrating with PPR."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from hashlib import md5
from itertools import combinations
from pathlib import Path
from typing import Any


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _build_chunk_maps(corpus: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    title_to_chunk: dict[str, str] = {}
    chunk_to_title: dict[str, str] = {}
    for row in corpus:
        doc_text = f"{row['title']}\n{row['text']}"
        chunk_id = compute_mdhash_id(doc_text, prefix="chunk-")
        chunk_to_title[chunk_id] = row["title"]
        title_to_chunk[row["title"]] = chunk_id
    return chunk_to_title, title_to_chunk


def _load_dataset(path: Path) -> list[dict]:
    return _load_json(path)


def _load_manifest(index_dir: Path) -> dict:
    manifest_path = index_dir / "manifest.json"
    return _load_json(manifest_path)


def _load_events(index_dir: Path) -> dict[str, dict[str, Any]]:
    raw = _load_json(index_dir / "event_nodes.json")
    return {
        event["event_id"]: event
        for event in raw.get("events", [])
        if isinstance(event, dict) and event.get("event_id")
    }


def _load_edges(index_dir: Path) -> dict[str, dict[str, Any]]:
    raw = _load_json(index_dir / "causal_edges.json")
    return {
        edge["edge_id"]: edge
        for edge in raw.get("edges", [])
        if isinstance(edge, dict) and edge.get("edge_id")
    }


def _load_chunk_map(index_dir: Path) -> dict[str, list[str]]:
    raw = _load_json(index_dir / "chunk_to_event_ids.json")
    return {chunk_id: list(event_ids) for chunk_id, event_ids in raw.items()}


def _build_adjacency(
    edges: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[frozenset[str], list[str]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_relations: dict[frozenset[str], list[str]] = defaultdict(list)
    for edge in edges.values():
        src = edge["source_event_id"]
        tgt = edge["target_event_id"]
        adjacency[src].add(tgt)
        adjacency[tgt].add(src)
        edge_relations[frozenset({src, tgt})].append(edge.get("relation_type"))
    return adjacency, edge_relations


def _bridgeability_summary(
    samples: list[dict],
    event_nodes: dict[str, dict[str, Any]],
    chunk_to_events: dict[str, list[str]],
    title_to_chunk: dict[str, str],
    adjacency: dict[str, set[str]],
    edge_relations: dict[frozenset[str], list[str]],
) -> tuple[dict, list[dict]]:
    summary = {
        "total_queries": len(samples),
        "queries_with_2plus_gold_docs": 0,
        "queries_with_all_support_mapped": 0,
        "queries_with_any_unmapped_support": 0,
        "queries_with_shared_entity": 0,
        "queries_with_relation_bridge": 0,
        "gold_doc_pairs": 0,
        "shared_pairs": 0,
        "relation_pairs": 0,
    }
    details: list[dict] = []

    for sample in samples:
        question = sample.get("question", "")
        supporting = sample.get("supporting_facts", [])
        gold_titles = []
        chunk_ids = []
        unmapped_titles = []
        for fact in supporting:
            title = fact[0]
            gold_titles.append(title)
            chunk_id = title_to_chunk.get(title)
            if chunk_id:
                chunk_ids.append(chunk_id)
            else:
                unmapped_titles.append(title)
        chunk_ids = list(dict.fromkeys(chunk_ids))
        if len(chunk_ids) < 2:
            continue

        summary["queries_with_2plus_gold_docs"] += 1
        if unmapped_titles:
            summary["queries_with_any_unmapped_support"] += 1
        else:
            summary["queries_with_all_support_mapped"] += 1

        has_shared = False
        has_relation = False
        pair_count = 0
        shared_pair_count = 0
        relation_pair_count = 0
        bridge_info: dict[str, Any] = {
            "question": question,
            "gold_titles": list(dict.fromkeys(gold_titles)),
            "unmapped_titles": unmapped_titles,
            "shared_entities": [],
            "relation_bridges": [],
        }

        for a, b in combinations(chunk_ids, 2):
            pair_count += 1
            events_a = set(chunk_to_events.get(a, []))
            events_b = set(chunk_to_events.get(b, []))
            shared = events_a & events_b
            if shared:
                has_shared = True
                shared_pair_count += 1
                bridge_info["shared_entities"].append(
                    {
                        "pair_chunk_ids": [a, b],
                        "pair_titles": [next((title for title, cid in title_to_chunk.items() if cid == a), a), next((title for title, cid in title_to_chunk.items() if cid == b), b)],
                        "entities": [
                            {
                                "event_id": event_id,
                                "text": event_nodes.get(event_id, {}).get("canonical_text"),
                            }
                            for event_id in sorted(shared)
                        ],
                    }
                )
            for evt_a in sorted(events_a):
                neighbors = events_b & adjacency.get(evt_a, set())
                if not neighbors:
                    continue
                evt_b = sorted(neighbors)[0]
                relations = sorted(edge_relations.get(frozenset({evt_a, evt_b}), []))
                has_relation = True
                relation_pair_count += 1
                bridge_info["relation_bridges"].append(
                    {
                        "pair_chunk_ids": [a, b],
                        "pair_titles": [next((title for title, cid in title_to_chunk.items() if cid == a), a), next((title for title, cid in title_to_chunk.items() if cid == b), b)],
                        "source_event": {
                            "event_id": evt_a,
                            "text": event_nodes.get(evt_a, {}).get("canonical_text"),
                        },
                        "target_event": {
                            "event_id": evt_b,
                            "text": event_nodes.get(evt_b, {}).get("canonical_text"),
                        },
                        "relations": relations,
                    }
                )
                break
        summary["gold_doc_pairs"] += pair_count
        if has_shared:
            summary["queries_with_shared_entity"] += 1
            summary["shared_pairs"] += shared_pair_count
        if has_relation:
            summary["queries_with_relation_bridge"] += 1
            summary["relation_pairs"] += relation_pair_count
        if has_shared or has_relation:
            details.append(bridge_info)

    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V2 general graph quality for Step 0.")
    parser.add_argument("--index_dir", type=Path, required=True, help="Path to the causal_v2 index directory.")
    parser.add_argument("--dataset", type=Path, required=True, help="2Wiki samples JSON (e.g., reproduce/dataset/2wikimultihopqa.json).")
    parser.add_argument("--corpus", type=Path, required=True, help="2Wiki corpus JSON (e.g., reproduce/dataset/2wikimultihopqa_corpus.json).")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the JSON report.")
    args = parser.parse_args()

    manifest = _load_manifest(args.index_dir)
    event_nodes = _load_events(args.index_dir)
    edges = _load_edges(args.index_dir)
    chunk_to_events = _load_chunk_map(args.index_dir)
    samples = _load_dataset(args.dataset)
    corpus = _load_dataset(args.corpus)

    chunk_to_title, title_to_chunk = _build_chunk_maps(corpus)
    adjacency, edge_relations = _build_adjacency(edges)

    total_chunk_events = sum(len(events) for events in chunk_to_events.values())
    total_chunks = len(corpus)
    eventful_chunks = len(chunk_to_events)
    total_edge_mentions = sum(len(edge["chunk_ids"]) for edge in edges.values())
    avg_entities_per_all_chunks = total_chunk_events / total_chunks if total_chunks else 0.0
    avg_entities_per_eventful_chunk = total_chunk_events / eventful_chunks if eventful_chunks else 0.0
    avg_edge_mentions_per_all_chunks = total_edge_mentions / total_chunks if total_chunks else 0.0
    avg_edge_mentions_per_eventful_chunk = total_edge_mentions / eventful_chunks if eventful_chunks else 0.0

    chunk_entity_counts = Counter()
    for chunk_id, events in chunk_to_events.items():
        for event_id in events:
            chunk_entity_counts[event_id] += 1

    event_degrees = {event_id: len(adjacency.get(event_id, [])) for event_id in event_nodes.keys()}

    hub_by_chunk = sorted(chunk_entity_counts.items(), key=lambda item: item[1], reverse=True)[:20]
    hub_by_degree = sorted(event_degrees.items(), key=lambda item: item[1], reverse=True)[:20]

    bridge_summary, bridge_details = _bridgeability_summary(
        samples,
        event_nodes,
        chunk_to_events,
        title_to_chunk,
        adjacency,
        edge_relations,
    )

    rel_counter = Counter(edge.get("relation_type") for edge in edges.values() if edge.get("relation_type"))
    support_titles = {
        fact[0]
        for sample in samples
        for fact in sample.get("supporting_facts", [])
        if isinstance(fact, list) and fact
    }
    missing_support_titles = sorted(title for title in support_titles if title not in title_to_chunk)
    num_entities = len(event_nodes)
    num_edges = len(edges)
    graph_mode = manifest.get("graph_mode")
    parse_failures = manifest.get("parse_failure_examples", [])
    related_to_count = rel_counter.get("related_to", 0)
    mapped_queries = max(1, bridge_summary["queries_with_2plus_gold_docs"])
    mapped_pairs = max(1, bridge_summary["gold_doc_pairs"])

    analysis = {
        "manifest": {
            "num_chunks": manifest.get("num_chunks"),
            "num_chunks_with_events": manifest.get("num_chunks_with_events"),
            "num_chunks_with_edges": manifest.get("num_chunks_with_edges"),
            "num_parse_failures": manifest.get("num_parse_failures"),
            "num_surface_events": manifest.get("num_surface_events"),
            "num_surface_edges": manifest.get("num_surface_edges"),
            "num_events": manifest.get("num_events"),
            "num_edges": manifest.get("num_edges"),
            "graph_mode": manifest.get("graph_mode"),
            "version": manifest.get("version"),
        },
        "graph_overview": {
            "num_entities": num_entities,
            "num_edges": num_edges,
            "chunks_in_chunk_map": eventful_chunks,
            "avg_entities_per_all_chunks": round(avg_entities_per_all_chunks, 4),
            "avg_entities_per_eventful_chunk": round(avg_entities_per_eventful_chunk, 4),
            "avg_edge_mentions_per_all_chunks": round(avg_edge_mentions_per_all_chunks, 4),
            "avg_edge_mentions_per_eventful_chunk": round(avg_edge_mentions_per_eventful_chunk, 4),
            "edge_relation_types": len(rel_counter),
            "avg_graph_degree": round(sum(event_degrees.values()) / max(1, num_entities), 4),
            "related_to_fraction": round(related_to_count / max(1, num_edges), 4),
            "chunks_with_events_rate": round(manifest.get("num_chunks_with_events", 0) / max(1, manifest.get("num_chunks", 0)), 4),
            "chunks_with_edges_rate": round(manifest.get("num_chunks_with_edges", 0) / max(1, manifest.get("num_chunks", 0)), 4),
            "parse_failure_rate": round(manifest.get("num_parse_failures", 0) / max(1, manifest.get("num_chunks", 0)), 4),
        },
        "relation_counts": dict(rel_counter),
        "support_title_mapping": {
            "num_unique_support_titles": len(support_titles),
            "num_missing_support_titles": len(missing_support_titles),
            "missing_support_titles_examples": missing_support_titles[:20],
        },
        "parse_failures": {
            "total": len(parse_failures),
            "examples": parse_failures[:6],
        },
        "hubs": {
            "by_chunk_count": [
                {
                    "event_id": event_id,
                    "text": event_nodes.get(event_id, {}).get("canonical_text"),
                    "chunk_count": count,
                }
                for event_id, count in hub_by_chunk
            ],
            "by_degree": [
                {
                    "event_id": event_id,
                    "text": event_nodes.get(event_id, {}).get("canonical_text"),
                    "degree": degree,
                }
                for event_id, degree in hub_by_degree
            ],
        },
        "bridge_summary": {
            **bridge_summary,
            "shared_entity_query_rate": round(bridge_summary["queries_with_shared_entity"] / mapped_queries, 4),
            "relation_bridge_query_rate": round(bridge_summary["queries_with_relation_bridge"] / mapped_queries, 4),
            "shared_entity_pair_rate": round(bridge_summary["shared_pairs"] / mapped_pairs, 4),
            "relation_bridge_pair_rate": round(bridge_summary["relation_pairs"] / mapped_pairs, 4),
        },
        "bridge_examples": bridge_details[:20],
        "verdict_hints": {
            "graph_mode": graph_mode,
            "go_signal_relation_not_collapsed": related_to_count < num_edges if num_edges else False,
            "go_signal_bridge_query_rate_ge_0_3": round(bridge_summary["queries_with_relation_bridge"] / mapped_queries, 4) >= 0.3,
            "go_signal_parse_failure_le_0_1": round(manifest.get("num_parse_failures", 0) / max(1, manifest.get("num_chunks", 0)), 4) <= 0.1,
        },
    }

    output_path = args.output or args.index_dir / "general_graph_step0_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    print(json.dumps({"output": str(output_path), "stats": analysis["graph_overview"]}, indent=2))


if __name__ == "__main__":
    main()
