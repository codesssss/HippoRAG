"""Local, deterministic DAEC-DAPG smoke implementation.

This module intentionally uses lexical grounding so Phase 1/2 smoke tests can
run without external embedding or extraction services. Production experiments
can replace the source and proposition extraction layers while preserving the
same graph / propagation / projection contracts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Sequence

import numpy as np

from src.dpathrag.data import normalize_text
from src.dpathrag.daec_dapg.graph import build_absorption_graph
from src.dpathrag.daec_dapg.projection import greedy_noisy_or_select, noisy_or_coverage
from src.dpathrag.daec_dapg.propagation import finite_horizon_absorption, support_tensor
from src.dpathrag.daec_dapg.schemas import EvidenceDocument


def tokens(text: Any) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", normalize_text(text)) if len(tok) > 1}


def token_cosine(left: Any, right: Any) -> float:
    a = tokens(left)
    b = tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def extract_demands(record: dict[str, Any]) -> list[str]:
    for key in ("active_requirements", "retrieval_active_demands", "demands", "requirements"):
        values = record.get(key)
        if isinstance(values, list) and values:
            out: list[str] = []
            for item in values:
                if isinstance(item, dict):
                    role = str(item.get("role") or item.get("type") or "lookup").lower()
                    if role in {"comparison", "aggregation", "operator", "inference"}:
                        continue
                    text = item.get("text") or item.get("subquery") or item.get("question")
                    if text:
                        out.append(str(text))
                elif item:
                    out.append(str(item))
            if out:
                return out
    return [str(record.get("question") or "")]


def documents_from_pool_record(record: dict[str, Any], *, max_docs: int = 100) -> list[EvidenceDocument]:
    pool_docs = list(record.get("pool_docs") or record.get("docs") or [])[: int(max_docs)]
    pool_titles = list(record.get("pool_titles") or [])
    pool_scores = list(record.get("pool_doc_scores") or [])
    pool_ids = list(record.get("pool_doc_ids") or [])
    docs: list[EvidenceDocument] = []
    for idx, raw in enumerate(pool_docs):
        raw_text = str(raw)
        title = str(pool_titles[idx]) if idx < len(pool_titles) else raw_text.split("\n", 1)[0]
        body = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
        docs.append(
            EvidenceDocument(
                doc_id=str(pool_ids[idx] if idx < len(pool_ids) else idx),
                title=title,
                text=body,
                score=float(pool_scores[idx]) if idx < len(pool_scores) and pool_scores[idx] is not None else 0.0,
            )
        )
    return docs


def _proposition_text(doc: EvidenceDocument) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", doc.text.strip())[0] if doc.text.strip() else doc.title
    return f"{doc.title} {first_sentence}".strip()


def _entity_terms(doc: EvidenceDocument) -> list[str]:
    values = list(tokens(doc.title))
    values.extend(list(tokens(doc.text))[:12])
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def build_local_absorption_inputs(docs: Sequence[EvidenceDocument]) -> tuple[list[str], list[str], dict[str, str], list[tuple[str, str, float]], dict[str, str]]:
    node_types: dict[str, str] = {}
    node_text: dict[str, str] = {}
    edges: list[tuple[str, str, float]] = []
    entity_df: Counter[str] = Counter()
    doc_entities: list[list[str]] = []
    for doc in docs:
        terms = _entity_terms(doc)
        doc_entities.append(terms)
        entity_df.update(set(terms))

    n_docs = max(1, len(docs))
    for idx, doc in enumerate(docs):
        d_node = f"D:{idx}"
        p_node = f"P:{idx}"
        node_types[d_node] = "D"
        node_types[p_node] = "P"
        node_text[d_node] = doc.full_text
        node_text[p_node] = _proposition_text(doc)
        edges.append((p_node, d_node, 1.0))
        for term in doc_entities[idx]:
            e_node = f"E:{term}"
            node_types[e_node] = "E"
            node_text[e_node] = term
            idf = math.log((n_docs + 1.0) / (entity_df[term] + 1.0)) + 1.0
            edges.append((p_node, e_node, idf))
            edges.append((e_node, p_node, idf))
    return [f"D:{idx}" for idx in range(len(docs))], [f"P:{idx}" for idx in range(len(docs))], node_types, edges, node_text


def lexical_phi(
    demand_texts: Sequence[str],
    docs: Sequence[EvidenceDocument],
    *,
    alpha: float = 0.15,
    horizon: int = 2,
    use_absorption: bool = True,
    use_kappa: bool = True,
) -> np.ndarray:
    """Return ``phi`` with shape ``(num_demands, num_docs)`` for one binding."""

    if not docs:
        return np.zeros((len(demand_texts), 0), dtype=float)
    _, _, node_types, edges, node_text = build_local_absorption_inputs(docs)
    graph = build_absorption_graph(node_types=node_types, edges=edges)
    doc_texts = [doc.full_text for doc in docs]
    rows: list[np.ndarray] = []
    for demand in demand_texts:
        source_u = np.array([token_cosine(demand, node_text[node]) for node in graph.transient_nodes], dtype=float)
        source_d = np.array([token_cosine(demand, doc_texts[idx]) for idx, _ in enumerate(graph.document_nodes)], dtype=float)
        if use_absorption:
            hit = finite_horizon_absorption(graph.q, graph.r, source_u, source_d * 0.0, alpha=float(alpha), horizon=int(horizon))
        else:
            hit = source_d / max(float(source_d.max()), 1e-12)
        kappa = np.array([token_cosine(demand, text) for text in doc_texts], dtype=float)
        rows.append(support_tensor(kappa if use_kappa else np.ones_like(kappa), hit))
    return np.vstack(rows) if rows else np.zeros((0, len(docs)), dtype=float)


def select_variant(
    record: dict[str, Any],
    *,
    variant: str,
    top_k: int = 5,
    max_docs: int = 100,
    alpha: float = 0.15,
    horizon: int = 2,
) -> dict[str, Any]:
    docs = documents_from_pool_record(record, max_docs=int(max_docs))
    demands = extract_demands(record)
    if variant == "daec_l1":
        phi = lexical_phi(demands, docs, alpha=alpha, horizon=horizon, use_absorption=False, use_kappa=True)
    elif variant == "local_occupancy":
        phi = lexical_phi([str(record.get("question") or "")], docs, alpha=alpha, horizon=horizon, use_absorption=False, use_kappa=False)
    elif variant == "local_absorption":
        phi = lexical_phi(demands, docs, alpha=alpha, horizon=horizon, use_absorption=True, use_kappa=False)
    elif variant in {"local_absorption_kappa", "local_absorption_prior"}:
        phi = lexical_phi(demands, docs, alpha=alpha, horizon=horizon, use_absorption=True, use_kappa=True)
    elif variant == "query_level_ppr":
        phi = lexical_phi([str(record.get("question") or "")], docs, alpha=alpha, horizon=horizon, use_absorption=True, use_kappa=False)
    elif variant == "mixed_source_absorption":
        phi = lexical_phi([" ".join(demands)], docs, alpha=alpha, horizon=horizon, use_absorption=True, use_kappa=False)
    elif variant == "multi_channel_sum":
        phi = lexical_phi(demands, docs, alpha=alpha, horizon=horizon, use_absorption=True, use_kappa=True)
        scores = phi.mean(axis=0) if phi.size else np.zeros(len(docs), dtype=float)
        selected = list(np.argsort(-scores)[: int(top_k)])
        return {
            "variant": variant,
            "selected_indices": selected,
            "selected_titles": [docs[idx].title for idx in selected],
            "objective": float(np.sum(scores[selected])) if selected else 0.0,
            "demand_count": len(demands),
        }
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    selected, objective = greedy_noisy_or_select(phi, budget=int(top_k))
    return {
        "variant": variant,
        "selected_indices": selected,
        "selected_titles": [docs[idx].title for idx in selected],
        "objective": objective if variant != "multi_channel_sum" else noisy_or_coverage(phi, selected),
        "demand_count": len(demands),
    }
