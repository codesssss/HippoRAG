"""Minimal oracle-slot BSGS runner used by Week-1 diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from .extractor import heuristic_extract_propositions
from .metrics import belief_entropy, recall_at_gold
from .slots import Slot
from .state import BeliefStateGraph, PropositionNode
from .transition import semi_absorbing_transition, sparse_transition


def lexical_likelihood(query: str, text: str) -> float:
    q_tokens = {tok for tok in query.lower().split() if len(tok) > 2}
    t_tokens = {tok for tok in text.lower().split() if len(tok) > 2}
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens) / len(q_tokens)
    return min(1.0, max(0.0, overlap))


def build_paragraph_propositions(sample: dict[str, Any], max_props_per_paragraph: int = 4) -> dict[str, PropositionNode]:
    propositions: dict[str, PropositionNode] = {}
    for para in sample.get("paragraphs") or []:
        title = str(para.get("title") or f"paragraph-{para.get('idx')}")
        doc_id = f"{para.get('idx')}::{title}"
        metadata = {
            "title": title,
            "paragraph_idx": para.get("idx"),
            "is_supporting": bool(para.get("is_supporting")),
        }
        for prop in heuristic_extract_propositions(
            passage=str(para.get("paragraph_text") or ""),
            source_doc_id=doc_id,
            max_props=max_props_per_paragraph,
            metadata=metadata,
        ):
            propositions[prop.prop_id] = prop
    return propositions


def build_shared_doc_edges(propositions: dict[str, PropositionNode]) -> dict[tuple[str, str], float]:
    by_doc: dict[str, list[str]] = {}
    for prop in propositions.values():
        by_doc.setdefault(prop.source_doc_id, []).append(prop.prop_id)
    edges: dict[tuple[str, str], float] = {}
    for prop_ids in by_doc.values():
        for src in prop_ids:
            for dst in prop_ids:
                edges[(src, dst)] = 1.0
    for prop_id in propositions:
        edges.setdefault((prop_id, prop_id), 1.0)
    return edges


def run_oracle_bsgs_sample(
    sample: dict[str, Any],
    slots: Sequence[Slot],
    absorbing_enabled: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    propositions = build_paragraph_propositions(sample)
    edges = build_shared_doc_edges(propositions)
    graph = BeliefStateGraph.uniform(propositions=propositions, edges=edges)

    trace: list[dict[str, Any]] = []
    for slot in slots:
        likelihood = {
            prop_id: lexical_likelihood(slot.slot_text, prop.text)
            for prop_id, prop in graph.propositions.items()
        }
        transition = sparse_transition(graph.edges, likelihood, nodes=graph.propositions)
        if absorbing_enabled:
            support_prob = {
                prop_id: max(likelihood.get(prop_id, 0.0), prop.support_prob)
                for prop_id, prop in graph.propositions.items()
            }
            transition = semi_absorbing_transition(transition, support_prob, nodes=graph.propositions)
        graph.belief = graph.predict(transition)
        graph.observe(likelihood)
        trace.append(
            {
                "slot": asdict(slot),
                "entropy": belief_entropy(graph.belief),
                "top_props": graph.topk(3),
            }
        )

    selected = graph.topk(top_k, support_weighted=False)
    selected_props = [graph.propositions[prop_id] for prop_id, _ in selected]
    selected_titles = [str(prop.metadata.get("title") or prop.source_doc_id) for prop in selected_props]
    gold_titles = [
        str(para.get("title"))
        for para in (sample.get("paragraphs") or [])
        if bool(para.get("is_supporting"))
    ]
    return {
        "qid": sample.get("id") or sample.get("_id"),
        "question": sample.get("question"),
        "selected_prop_ids": [prop_id for prop_id, _ in selected],
        "selected_titles": selected_titles,
        "gold_titles": gold_titles,
        "supporting_paragraph_recall": recall_at_gold(selected_titles, gold_titles),
        "belief_entropy": graph.entropy(),
        "trace": trace,
    }
