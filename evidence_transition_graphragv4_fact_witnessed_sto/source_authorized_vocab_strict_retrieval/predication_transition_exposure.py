"""Exposure audit for Predication-Transition GraphRAG.

This module turns the existing source-warranted atom / transition-witness
diagnostics into a narrow candidate-exposure test.  It is not a reader reranker:
transition-visible documents are emitted in the original base-rank order, and
the base reader head is left untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .certificate_graph import EvidenceNode
from .evaluate_report import (
    all_gold_at_k,
    load_candidate_cache_doc_indices,
    load_json,
    load_nodes,
    recall_at_k,
    rows_for_variant,
    unique_ints,
)
from .query_conditioning import extract_query_signatures
from .transition_witness_audit import (
    _atoms_by_doc,
    _entity_to_docs,
    _navigation_targets,
    _relation_alias_index,
    _transition_witnesses,
)
from .witness_atoms import atom_entities_for_doc, title_entities_for_doc


PREDICATION_TRANSITION_EXPOSURE_CONTRACT: Mapping[str, bool | str] = {
    "audit": "predication_transition_tail_exposure",
    "method_object": "source_atom_connector_target_atom_transition",
    "is_retriever": False,
    "uses_gold_support_labels_for_audit_only": True,
    "uses_reader_head_replacement": False,
    "uses_base_rank_preserving_tail": True,
    "uses_weighted_score_fusion": False,
    "uses_learned_reranker": False,
    "uses_llm_generated_propositions": False,
    "uses_role_vocabulary": False,
    "uses_dataset_routing": False,
}


@dataclass(frozen=True)
class PredicationTransitionConfig:
    name: str
    mode: str
    link_policy: str = "target_atom_bound"
    relation_signal_policy: str = "surface"
    sibling_policy: str = "all"


DEFAULT_CONFIGS: Tuple[PredicationTransitionConfig, ...] = (
    PredicationTransitionConfig(
        name="connector_only_target_atom",
        mode="connector_only",
        link_policy="target_atom_bound",
    ),
    PredicationTransitionConfig(
        name="predication_transition_surface",
        mode="predication_transition",
        link_policy="target_atom_bound",
        relation_signal_policy="surface",
    ),
    PredicationTransitionConfig(
        name="predication_transition_pool_alias",
        mode="predication_transition",
        link_policy="target_atom_bound",
        relation_signal_policy="pool_alias",
    ),
    PredicationTransitionConfig(
        name="no_target_predication_connector_mention",
        mode="no_target_predication",
        link_policy="entity_overlap",
    ),
)

MINIMAL_SIBLING_CONFIGS: Tuple[PredicationTransitionConfig, ...] = (
    PredicationTransitionConfig(
        name="predication_transition_surface_minimal",
        mode="predication_transition",
        link_policy="target_atom_bound",
        relation_signal_policy="surface",
        sibling_policy="source_connector_best_rank",
    ),
    PredicationTransitionConfig(
        name="predication_transition_pool_alias_minimal",
        mode="predication_transition",
        link_policy="target_atom_bound",
        relation_signal_policy="pool_alias",
        sibling_policy="source_connector_best_rank",
    ),
)


@dataclass(frozen=True)
class PredicationTransitionTail:
    transition_doc_indices: Tuple[int, ...]
    assembled_doc_indices: Tuple[int, ...]
    trace: Mapping[str, object]


def build_predication_transition_tail(
    *,
    query: str,
    nodes: Sequence[EvidenceNode],
    candidate_doc_indices: Sequence[int],
    config: PredicationTransitionConfig,
    source_head_k: int = 5,
    preserve_head_k: int | None = None,
    candidate_pool_k: int = 200,
    final_k: int = 200,
) -> PredicationTransitionTail:
    """Return transition-visible docs, preserving the base candidate order."""

    source_head_k = max(int(source_head_k), 1)
    preserve_head_k = source_head_k if preserve_head_k is None else max(int(preserve_head_k), 1)
    candidate_pool_k = max(int(candidate_pool_k), source_head_k)
    candidate_pool_k = max(int(candidate_pool_k), preserve_head_k)
    final_k = max(int(final_k), source_head_k)
    final_k = max(int(final_k), preserve_head_k)
    pool = tuple(unique_ints(candidate_doc_indices))[:candidate_pool_k]
    if not pool:
        return PredicationTransitionTail(
            transition_doc_indices=(),
            assembled_doc_indices=(),
            trace={
                **dict(PREDICATION_TRANSITION_EXPOSURE_CONTRACT),
                "status": "empty_candidate_pool",
                "config": config.name,
            },
        )

    node_lookup = {int(node.doc_index): node for node in nodes}
    entry_doc_indices = tuple(pool[:source_head_k])
    signatures = extract_query_signatures(
        query=str(query),
        nodes=nodes,
        candidate_doc_indices=pool,
        query_mentioned_doc_indices=(),
    )
    atoms_by_doc = _atoms_by_doc(pool, node_lookup)
    candidate_rank = {int(doc_index): rank for rank, doc_index in enumerate(pool)}
    entities_by_doc = {
        int(doc_index): atom_entities_for_doc(
            node=node_lookup[int(doc_index)],
            atoms=atoms_by_doc.get(int(doc_index), ()),
        )
        for doc_index in pool
        if int(doc_index) in node_lookup
    }
    title_entities_by_doc = {
        int(doc_index): title_entities_for_doc(node_lookup[int(doc_index)])
        for doc_index in pool
        if int(doc_index) in node_lookup
    }
    entity_to_docs = _entity_to_docs(entities_by_doc)
    title_entity_to_docs = _entity_to_docs(title_entities_by_doc)
    relation_alias_index = _relation_alias_index(atoms_by_doc)

    mode = str(config.mode)
    transition_witness_count = 0
    raw_transition_witness_count = 0
    if mode == "connector_only":
        transition_doc_set = _navigation_targets(
            entry_doc_indices=entry_doc_indices,
            link_policy=str(config.link_policy),
            atoms_by_doc=atoms_by_doc,
            entities_by_doc=entities_by_doc,
            title_entities_by_doc=title_entities_by_doc,
            entity_to_docs=entity_to_docs,
            title_entity_to_docs=title_entity_to_docs,
        )
    elif mode == "no_target_predication":
        transition_doc_set = _connector_mention_targets(
            entry_doc_indices=entry_doc_indices,
            atoms_by_doc=atoms_by_doc,
            entity_to_docs=entity_to_docs,
        )
    else:
        witnesses = _transition_witnesses(
            entry_doc_indices=entry_doc_indices,
            link_policy=str(config.link_policy),
            atoms_by_doc=atoms_by_doc,
            entities_by_doc=entities_by_doc,
            title_entities_by_doc=title_entities_by_doc,
            entity_to_docs=entity_to_docs,
            title_entity_to_docs=title_entity_to_docs,
            signatures=signatures,
            relation_alias_index=relation_alias_index,
            relation_signal_policy=str(config.relation_signal_policy),
        )
        raw_transition_witness_count = len(witnesses)
        witnesses = _suppress_transition_siblings(
            witnesses=witnesses,
            candidate_rank=candidate_rank,
            sibling_policy=str(config.sibling_policy),
        )
        transition_witness_count = len(witnesses)
        transition_doc_set = {int(witness.target_doc_index) for witness in witnesses}

    head = tuple(pool[:preserve_head_k])
    head_set = set(head)
    transition_tail = tuple(
        int(doc_index)
        for doc_index in pool
        if int(doc_index) in transition_doc_set and int(doc_index) not in head_set
    )
    transition_tail_set = set(transition_tail)
    base_remainder = tuple(
        int(doc_index)
        for doc_index in pool[preserve_head_k:]
        if int(doc_index) not in transition_tail_set
    )
    assembled = tuple(unique_ints([*head, *transition_tail, *base_remainder]))[:final_k]
    return PredicationTransitionTail(
        transition_doc_indices=transition_tail,
        assembled_doc_indices=assembled,
        trace={
            **dict(PREDICATION_TRANSITION_EXPOSURE_CONTRACT),
            "status": "ok",
            "config": config.name,
            "mode": mode,
            "link_policy": str(config.link_policy),
            "relation_signal_policy": str(config.relation_signal_policy),
            "sibling_policy": str(config.sibling_policy),
            "source_head_k": int(source_head_k),
            "preserve_head_k": int(preserve_head_k),
            "candidate_pool_k": int(candidate_pool_k),
            "candidate_doc_count": len(pool),
            "entry_doc_indices": entry_doc_indices,
            "query_mentioned_doc_indices": tuple(signatures.title_mention_doc_indices),
            "transition_tail_doc_count": len(transition_tail),
            "transition_witness_count": int(transition_witness_count),
            "raw_transition_witness_count": int(raw_transition_witness_count),
            "query_relation_stems": tuple(signatures.relation_stems),
            "relation_alias_pair_count": int(
                relation_alias_index.trace.get("relation_alias_pair_count", 0)
            ),
        },
    )


def audit_report(
    *,
    report: Mapping[str, Any],
    nodes: Sequence[EvidenceNode],
    candidate_cache_doc_indices: Mapping[int, Sequence[int]],
    source_variant: str = "hipporag_v2",
    gap_reference_variant: str = "hipporag_v2",
    k_values: Sequence[int] = (5, 10, 20),
    source_head_k: int = 5,
    preserve_head_k: int | None = None,
    candidate_pool_k: int = 200,
    configs: Sequence[PredicationTransitionConfig] = DEFAULT_CONFIGS,
) -> Dict[str, Any]:
    rows = rows_for_variant(report, str(source_variant))
    gap_reference_rows = rows_for_variant(report, str(gap_reference_variant))
    gap_reference_by_query = {
        int(row.get("query_index", fallback_idx)): row
        for fallback_idx, row in enumerate(gap_reference_rows)
    }
    normalized_k_values = tuple(sorted({max(int(k), 1) for k in k_values}))
    normalized_preserve_head_k = (
        int(source_head_k) if preserve_head_k is None else max(int(preserve_head_k), 1)
    )

    config_summaries: Dict[str, Dict[str, Any]] = {}
    output_rows_by_config: Dict[str, List[Mapping[str, Any]]] = {}
    for config in configs:
        summaries = {str(k): _empty_summary() for k in normalized_k_values}
        row_outputs: List[Mapping[str, Any]] = []
        for fallback_idx, row in enumerate(rows):
            query_index = int(row.get("query_index", fallback_idx))
            query = str(row.get("question") or row.get("query") or "")
            candidates = tuple(
                unique_ints(candidate_cache_doc_indices.get(query_index, ()) or ())
            )
            tail = build_predication_transition_tail(
                query=query,
                nodes=nodes,
                candidate_doc_indices=candidates,
                config=config,
                source_head_k=source_head_k,
                preserve_head_k=normalized_preserve_head_k,
                candidate_pool_k=candidate_pool_k,
                final_k=candidate_pool_k,
            )
            transition_set = set(tail.transition_doc_indices)
            assembled = tuple(tail.assembled_doc_indices)
            candidate_rank = {
                int(doc_index): rank + 1
                for rank, doc_index in enumerate(candidates[:candidate_pool_k])
            }
            row_result = {
                "query_index": query_index,
                "question": query,
                "transition_tail_doc_indices": tail.transition_doc_indices,
                "transition_tail_doc_count": len(tail.transition_doc_indices),
                "trace": dict(tail.trace),
                "by_k": {},
            }
            for k in normalized_k_values:
                summary = summaries[str(k)]
                reference_row = gap_reference_by_query.get(query_index, row)
                gold = set(unique_ints(row.get("gold_doc_indices", []) or []))
                reference_topk = set(
                    unique_ints(
                        reference_row.get(f"retrieved_doc_indices_top{k}", [])
                        or reference_row.get("retrieved_doc_indices_top20", [])[:k]
                        or []
                    )
                )
                reference_retrieved = unique_ints(
                    reference_row.get(f"retrieved_doc_indices_top{k}", [])
                    or reference_row.get("retrieved_doc_indices_top20", [])[:k]
                    or []
                )
                missing_gold = sorted(gold - reference_topk)
                missing_in_candidate = [
                    doc for doc in missing_gold if int(doc) in candidate_rank
                ]
                exposed_gold = [
                    doc for doc in missing_gold if int(doc) in transition_set
                ]
                assembled_topk = set(assembled[:k])
                assembled_recovered = [
                    doc for doc in missing_gold if int(doc) in assembled_topk
                ]
                exposed_ranks = [
                    int(candidate_rank[doc])
                    for doc in exposed_gold
                    if int(doc) in candidate_rank
                ]
                if missing_gold:
                    summary["queries_with_missing_gold"] += 1
                    summary["missing_gold_docs"] += len(missing_gold)
                    summary["missing_gold_in_candidate_pool"] += len(missing_in_candidate)
                    summary["exposed_missing_gold_docs"] += len(exposed_gold)
                    summary["assembled_recovered_missing_gold_docs"] += len(
                        assembled_recovered
                    )
                    summary["queries_with_exposed_missing_gold"] += int(bool(exposed_gold))
                    summary["queries_with_assembled_recovered_missing_gold"] += int(
                        bool(assembled_recovered)
                    )
                    summary["exposed_candidate_ranks"].extend(exposed_ranks)
                summary["baseline_recall_sum"] += recall_at_k(gold, reference_retrieved, k)
                summary["assembled_recall_sum"] += recall_at_k(gold, assembled, k)
                summary["baseline_all_gold_count"] += int(
                    all_gold_at_k(gold, reference_retrieved, k)
                )
                summary["assembled_all_gold_count"] += int(all_gold_at_k(gold, assembled, k))
                baseline_recall = recall_at_k(gold, reference_retrieved, k)
                assembled_recall = recall_at_k(gold, assembled, k)
                summary["queries_with_assembled_recall_gain"] += int(
                    assembled_recall > baseline_recall
                )
                summary["queries_with_assembled_recall_loss"] += int(
                    assembled_recall < baseline_recall
                )
                summary["tail_doc_count_sum"] += len(tail.transition_doc_indices)
                summary["queries"] += 1
                row_result["by_k"][str(k)] = {
                    "missing_gold_doc_indices": tuple(missing_gold),
                    "missing_gold_in_candidate_pool_doc_indices": tuple(missing_in_candidate),
                    "exposed_missing_gold_doc_indices": tuple(exposed_gold),
                    "assembled_recovered_missing_gold_doc_indices": tuple(
                        assembled_recovered
                    ),
                    "exposed_candidate_ranks": tuple(exposed_ranks),
                }
            row_outputs.append(row_result)
        config_summaries[config.name] = {
            str(k): _finalize_summary(summary)
            for k, summary in summaries.items()
        }
        output_rows_by_config[config.name] = row_outputs

    return {
        **dict(PREDICATION_TRANSITION_EXPOSURE_CONTRACT),
        "dataset": str(report.get("dataset") or ""),
        "source_variant": str(source_variant),
        "gap_reference_variant": str(gap_reference_variant),
        "source_head_k": int(source_head_k),
        "preserve_head_k": int(normalized_preserve_head_k),
        "candidate_pool_k": int(candidate_pool_k),
        "k_values": tuple(int(k) for k in normalized_k_values),
        "configs": [config.__dict__ for config in configs],
        "per_config": config_summaries,
        "rows_by_config": output_rows_by_config,
    }


def _suppress_transition_siblings(
    *,
    witnesses: Sequence[object],
    candidate_rank: Mapping[int, int],
    sibling_policy: str,
) -> Tuple[object, ...]:
    """Keep one target per source connector when structural suppression is enabled."""

    normalized = str(sibling_policy or "all").strip().lower()
    if normalized == "all":
        return tuple(witnesses)
    if normalized != "source_connector_best_rank":
        raise ValueError(f"unknown sibling policy: {sibling_policy}")
    best_by_connector: Dict[Tuple[int, str], object] = {}
    for witness in witnesses:
        key = (
            int(getattr(witness, "source_doc_index")),
            str(getattr(witness, "connector_entity")),
        )
        current = best_by_connector.get(key)
        if current is None or _witness_rank_key(witness, candidate_rank) < _witness_rank_key(
            current,
            candidate_rank,
        ):
            best_by_connector[key] = witness
    return tuple(
        sorted(
            best_by_connector.values(),
            key=lambda item: (
                int(getattr(item, "source_doc_index")),
                int(getattr(item, "target_doc_index")),
                str(getattr(item, "connector_entity")),
                str(getattr(item, "target_atom_id")),
            ),
        )
    )


def _witness_rank_key(witness: object, candidate_rank: Mapping[int, int]) -> Tuple[int, int, str]:
    target = int(getattr(witness, "target_doc_index"))
    return (
        int(candidate_rank.get(target, 10**9)),
        target,
        str(getattr(witness, "target_atom_id")),
    )


def _connector_mention_targets(
    *,
    entry_doc_indices: Sequence[int],
    atoms_by_doc: Mapping[int, Sequence[object]],
    entity_to_docs: Mapping[str, Sequence[int]],
) -> set[int]:
    targets: set[int] = set()
    entry_set = set(unique_ints(entry_doc_indices))
    for source_doc_index in unique_ints(entry_doc_indices):
        for source_atom in atoms_by_doc.get(int(source_doc_index), ()):
            for connector in getattr(source_atom, "endpoint_entities", ()):
                connector_text = str(connector or "")
                if not connector_text:
                    continue
                for target_doc_index in entity_to_docs.get(connector_text, ()):
                    target = int(target_doc_index)
                    if target in entry_set:
                        continue
                    targets.add(target)
    return targets


def _empty_summary() -> Dict[str, Any]:
    return {
        "queries": 0,
        "queries_with_missing_gold": 0,
        "missing_gold_docs": 0,
        "missing_gold_in_candidate_pool": 0,
        "exposed_missing_gold_docs": 0,
        "assembled_recovered_missing_gold_docs": 0,
        "queries_with_exposed_missing_gold": 0,
        "queries_with_assembled_recovered_missing_gold": 0,
        "tail_doc_count_sum": 0,
        "exposed_candidate_ranks": [],
        "baseline_recall_sum": 0.0,
        "assembled_recall_sum": 0.0,
        "baseline_all_gold_count": 0,
        "assembled_all_gold_count": 0,
        "queries_with_assembled_recall_gain": 0,
        "queries_with_assembled_recall_loss": 0,
    }


def _finalize_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    missing = int(summary.get("missing_gold_docs", 0))
    queries_with_missing = int(summary.get("queries_with_missing_gold", 0))
    queries = int(summary.get("queries", 0))
    ranks = [int(rank) for rank in summary.get("exposed_candidate_ranks", []) or []]
    return {
        "queries_with_missing_gold": queries_with_missing,
        "missing_gold_docs": missing,
        "missing_gold_in_candidate_pool": int(
            summary.get("missing_gold_in_candidate_pool", 0)
        ),
        "exposed_missing_gold_docs": int(summary.get("exposed_missing_gold_docs", 0)),
        "assembled_recovered_missing_gold_docs": int(
            summary.get("assembled_recovered_missing_gold_docs", 0)
        ),
        "queries_with_exposed_missing_gold": int(
            summary.get("queries_with_exposed_missing_gold", 0)
        ),
        "queries_with_assembled_recovered_missing_gold": int(
            summary.get("queries_with_assembled_recovered_missing_gold", 0)
        ),
        "candidate_pool_missing_gold_coverage": _safe_div(
            summary.get("missing_gold_in_candidate_pool", 0),
            missing,
        ),
        "exposed_missing_gold_coverage": _safe_div(
            summary.get("exposed_missing_gold_docs", 0),
            missing,
        ),
        "assembled_recovered_missing_gold_coverage": _safe_div(
            summary.get("assembled_recovered_missing_gold_docs", 0),
            missing,
        ),
        "query_exposure_rate": _safe_div(
            summary.get("queries_with_exposed_missing_gold", 0),
            queries_with_missing,
        ),
        "query_assembled_recovery_rate": _safe_div(
            summary.get("queries_with_assembled_recovered_missing_gold", 0),
            queries_with_missing,
        ),
        "mean_transition_tail_doc_count": _safe_div(
            summary.get("tail_doc_count_sum", 0),
            queries,
        ),
        "baseline_recall": round(
            float(summary.get("baseline_recall_sum", 0.0)) / float(max(queries, 1)),
            6,
        ),
        "assembled_recall": round(
            float(summary.get("assembled_recall_sum", 0.0)) / float(max(queries, 1)),
            6,
        ),
        "assembled_recall_delta": round(
            (
                float(summary.get("assembled_recall_sum", 0.0))
                - float(summary.get("baseline_recall_sum", 0.0))
            )
            / float(max(queries, 1)),
            6,
        ),
        "baseline_all_gold_at_k": _safe_div(
            summary.get("baseline_all_gold_count", 0),
            queries,
        ),
        "assembled_all_gold_at_k": _safe_div(
            summary.get("assembled_all_gold_count", 0),
            queries,
        ),
        "assembled_all_gold_delta": _safe_div(
            int(summary.get("assembled_all_gold_count", 0))
            - int(summary.get("baseline_all_gold_count", 0)),
            queries,
        ),
        "queries_with_assembled_recall_gain": int(
            summary.get("queries_with_assembled_recall_gain", 0)
        ),
        "queries_with_assembled_recall_loss": int(
            summary.get("queries_with_assembled_recall_loss", 0)
        ),
        "exposed_candidate_rank_min": min(ranks) if ranks else None,
        "exposed_candidate_rank_median": round(float(median(ranks)), 6)
        if ranks
        else None,
        "exposed_candidate_rank_max": max(ranks) if ranks else None,
    }


def _safe_div(numerator: object, denominator: object) -> float:
    denominator_int = int(denominator or 0)
    if denominator_int <= 0:
        return 0.0
    return round(float(numerator or 0) / float(denominator_int), 6)


def _parse_k_values(text: str) -> Tuple[int, ...]:
    values = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            values.append(max(int(part), 1))
    return tuple(values or (5, 10, 20))


def _configs_from_args(args: argparse.Namespace) -> Tuple[PredicationTransitionConfig, ...]:
    config_text = str(getattr(args, "configs", "") or "default")
    if config_text == "default":
        return DEFAULT_CONFIGS
    configs = []
    for raw in config_text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name == "connector_only":
            configs.append(DEFAULT_CONFIGS[0])
        elif name == "predication_transition":
            configs.append(DEFAULT_CONFIGS[1])
        elif name == "predication_transition_alias":
            configs.append(DEFAULT_CONFIGS[2])
        elif name == "predication_transition_minimal":
            configs.append(MINIMAL_SIBLING_CONFIGS[0])
        elif name == "predication_transition_alias_minimal":
            configs.append(MINIMAL_SIBLING_CONFIGS[1])
        elif name == "no_target_predication":
            configs.append(DEFAULT_CONFIGS[3])
        else:
            raise ValueError(f"unknown config preset: {name}")
    return tuple(configs or DEFAULT_CONFIGS)


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Predication-Transition Exposure Audit",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| source variant | {payload.get('source_variant', '')} |",
        f"| gap reference variant | {payload.get('gap_reference_variant', '')} |",
        f"| source head k | {payload.get('source_head_k', '')} |",
        f"| preserve head k | {payload.get('preserve_head_k', '')} |",
        f"| candidate pool k | {payload.get('candidate_pool_k', '')} |",
        "",
        "| config | k | baseline R | assembled R | delta | missing docs | in pool | exposed | assembled top-k recovered | query exposure | gain/loss q | mean tail size | exposed rank median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    per_config = payload.get("per_config", {}) or {}
    for config_name, per_k in sorted(per_config.items()):
        for key, summary in sorted(per_k.items(), key=lambda item: int(item[0])):
            lines.append(
                "| {config} | {k} | {baseline_r:.4f} | {assembled_r:.4f} | {delta:+.4f} | "
                "{missing} | {in_pool} ({in_pool_rate:.4f}) | "
                "{exposed} ({exposed_rate:.4f}) | {assembled} ({assembled_rate:.4f}) | "
                "{query_rate:.4f} | {gain}/{loss} | {tail_size:.2f} | {rank_median} |".format(
                    config=config_name,
                    k=int(key),
                    baseline_r=float(summary.get("baseline_recall", 0.0)),
                    assembled_r=float(summary.get("assembled_recall", 0.0)),
                    delta=float(summary.get("assembled_recall_delta", 0.0)),
                    missing=int(summary.get("missing_gold_docs", 0)),
                    in_pool=int(summary.get("missing_gold_in_candidate_pool", 0)),
                    in_pool_rate=float(
                        summary.get("candidate_pool_missing_gold_coverage", 0.0)
                    ),
                    exposed=int(summary.get("exposed_missing_gold_docs", 0)),
                    exposed_rate=float(summary.get("exposed_missing_gold_coverage", 0.0)),
                    assembled=int(
                        summary.get("assembled_recovered_missing_gold_docs", 0)
                    ),
                    assembled_rate=float(
                        summary.get("assembled_recovered_missing_gold_coverage", 0.0)
                    ),
                    query_rate=float(summary.get("query_exposure_rate", 0.0)),
                    gain=int(summary.get("queries_with_assembled_recall_gain", 0)),
                    loss=int(summary.get("queries_with_assembled_recall_loss", 0)),
                    tail_size=float(summary.get("mean_transition_tail_doc_count", 0.0)),
                    rank_median=summary.get("exposed_candidate_rank_median"),
                )
            )
    output_path.expanduser().write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit predication-transition tail exposure from a per-query report."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--openie-path")
    parser.add_argument("--candidate-cache-path", required=True)
    parser.add_argument("--source-variant", default="hipporag_v2")
    parser.add_argument("--gap-reference-variant", default="hipporag_v2")
    parser.add_argument("--source-head-k", type=int, default=5)
    parser.add_argument(
        "--preserve-head-k",
        type=int,
        default=0,
        help="Base head size kept fixed before transition tail; 0 means source-head-k.",
    )
    parser.add_argument("--candidate-pool-k", type=int, default=200)
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument(
        "--configs",
        default="default",
        help=(
            "Comma-separated presets: connector_only,predication_transition,"
            "predication_transition_alias,predication_transition_minimal,"
            "predication_transition_alias_minimal,no_target_predication; default runs all."
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args(argv)

    report_path = Path(args.report).expanduser().resolve()
    report = load_json(report_path)
    openie_text = str(args.openie_path or report.get("analysis_openie_path") or "")
    if not openie_text:
        raise ValueError("--openie-path is required when report has no analysis_openie_path")
    openie_path = Path(openie_text).expanduser()
    if not openie_path.is_absolute():
        openie_path = (report_path.parent / openie_path).resolve()
    payload = audit_report(
        report=report,
        nodes=load_nodes(openie_path),
        candidate_cache_doc_indices=load_candidate_cache_doc_indices(
            Path(args.candidate_cache_path).expanduser()
        ),
        source_variant=str(args.source_variant),
        gap_reference_variant=str(args.gap_reference_variant),
        k_values=_parse_k_values(str(args.k_values)),
        source_head_k=max(int(args.source_head_k), 1),
        preserve_head_k=(
            None if int(args.preserve_head_k) <= 0 else max(int(args.preserve_head_k), 1)
        ),
        candidate_pool_k=max(int(args.candidate_pool_k), 1),
        configs=_configs_from_args(args),
    )
    output_json = Path(args.output_json).expanduser()
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        write_markdown(payload, Path(args.output_md).expanduser())
    print(
        json.dumps(
            {
                config: per_k
                for config, per_k in (payload.get("per_config", {}) or {}).items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
