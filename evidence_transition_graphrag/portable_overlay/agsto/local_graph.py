"""Compatibility import for query-time local STO graph admission."""

from src.agsto.local_graph import (
    LOCAL_ADMISSION_EDGE_KINDS,
    build_query_local_sto_graph,
    select_local_sto_evidence_docs,
)

__all__ = ["LOCAL_ADMISSION_EDGE_KINDS", "build_query_local_sto_graph", "select_local_sto_evidence_docs"]
