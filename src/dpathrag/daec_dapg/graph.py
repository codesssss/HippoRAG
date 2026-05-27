"""Absorption graph construction for DAEC-DAPG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class AbsorptionGraph:
    """Transition blocks for a document-absorbing Markov graph."""

    transient_nodes: list[str]
    document_nodes: list[str]
    q: np.ndarray
    r: np.ndarray


def _allowed_edge(src_type: str, dst_type: str, *, allow_entity_to_doc: bool) -> bool:
    if src_type == "P" and dst_type in {"E", "D"}:
        return True
    if src_type == "E" and dst_type == "P":
        return True
    if allow_entity_to_doc and src_type == "E" and dst_type == "D":
        return True
    return False


def build_absorption_graph(
    *,
    node_types: Mapping[str, str],
    edges: Iterable[tuple[str, str, float]],
    allow_entity_to_doc: bool = False,
) -> AbsorptionGraph:
    """Project an index graph into an absorption graph.

    Main-method edges are ``P -> E``, ``E -> P``, and ``P -> D``. Document
    nodes are sinks and therefore do not appear as transient rows.
    """

    transient_nodes = sorted(node for node, kind in node_types.items() if str(kind).upper() in {"P", "E"})
    document_nodes = sorted(node for node, kind in node_types.items() if str(kind).upper() == "D")
    t_index = {node: idx for idx, node in enumerate(transient_nodes)}
    d_index = {node: idx for idx, node in enumerate(document_nodes)}
    q = np.zeros((len(transient_nodes), len(transient_nodes)), dtype=float)
    r = np.zeros((len(transient_nodes), len(document_nodes)), dtype=float)

    for src, dst, weight in edges:
        if src not in node_types or dst not in node_types:
            continue
        src_type = str(node_types[src]).upper()
        dst_type = str(node_types[dst]).upper()
        if not _allowed_edge(src_type, dst_type, allow_entity_to_doc=bool(allow_entity_to_doc)):
            continue
        value = max(0.0, float(weight))
        if value <= 0.0 or src not in t_index:
            continue
        if dst in t_index:
            q[t_index[src], t_index[dst]] += value
        elif dst in d_index:
            r[t_index[src], d_index[dst]] += value

    row_sums = q.sum(axis=1) + r.sum(axis=1)
    for row, total in enumerate(row_sums):
        if total > 0.0:
            q[row, :] /= total
            r[row, :] /= total
    return AbsorptionGraph(transient_nodes=transient_nodes, document_nodes=document_nodes, q=q, r=r)
