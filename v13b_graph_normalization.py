#!/usr/bin/env python3
"""Shared normalization helpers for v13b query-demand/corpus-fact alignment.

This module is intentionally small and deterministic.  It does not score,
rerank, call LLMs, or consult gold labels.  Its job is to make the query-side
obligation language and corpus-side OpenIE/STO language pass through the same
surface normalization before diagnostics or structural source-report building.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Set

from build_query_obligation_units import (
    endpoint_alias_signatures,
    endpoint_signature,
    temporal_event_relation_signature,
)


def entity_key(value: Any) -> str:
    """Return the canonical endpoint key used by query demands and STO facts."""

    return endpoint_signature(value)


def relation_key(value: Any, object_value: Any = "") -> str:
    """Return the canonical relation key used by query demands and STO facts."""

    return temporal_event_relation_signature(value, object_value)


def fact_role_keys(unit: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one OpenIE/STO fact unit into role-aware keys."""

    subject_values = [
        unit.get("grounded_subject"),
        unit.get("subject"),
    ]
    object_values = [
        unit.get("grounded_object"),
        unit.get("object"),
    ]
    subject_keys = {entity_key(value) for value in subject_values if entity_key(value)}
    object_keys = {entity_key(value) for value in object_values if entity_key(value)}
    object_for_relation = next(iter(object_keys), unit.get("object", ""))
    return {
        "unit_id": str(unit.get("unit_id") or ""),
        "doc_index": safe_int(unit.get("doc_index")),
        "title": str(unit.get("title") or ""),
        "subject_surface": str(unit.get("subject") or ""),
        "subject_key": entity_key(unit.get("subject")),
        "subject_keys": subject_keys,
        "relation_surface": str(unit.get("relation") or ""),
        "relation_key": relation_key(unit.get("relation"), object_for_relation),
        "object_surface": str(unit.get("object") or ""),
        "object_key": entity_key(unit.get("object")),
        "object_keys": object_keys,
        "source_sentence": str(unit.get("source_text") or unit.get("span_text") or ""),
        "fact": list(unit.get("fact", []) or []),
    }


def obligation_role_keys(obligation: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one query obligation into the same role-aware key space."""

    return {
        "obligation_id": str(obligation.get("obligation_id") or ""),
        "subject_surface": str(obligation.get("raw_subject") or obligation.get("subject") or ""),
        "subject_key": entity_key(obligation.get("subject")),
        "subject_is_variable": bool(obligation.get("subject_is_variable", False)),
        "subject_variable": str(obligation.get("subject_variable") or ""),
        "relation_surface": str(obligation.get("raw_relation") or obligation.get("relation") or ""),
        "relation_key": relation_key(
            obligation.get("raw_relation") or obligation.get("relation"),
            obligation.get("raw_object") or obligation.get("object"),
        ),
        "object_surface": str(obligation.get("raw_object") or obligation.get("object") or ""),
        "object_key": entity_key(obligation.get("object")),
        "object_is_variable": bool(obligation.get("object_is_variable", False)),
        "object_variable": str(obligation.get("object_variable") or ""),
        "raw_triple": [
            obligation.get("raw_subject", ""),
            obligation.get("raw_relation", ""),
            obligation.get("raw_object", ""),
        ],
    }


def bound_endpoint_keys(obligation: Mapping[str, Any]) -> Set[str]:
    """Return concrete endpoint keys required by an obligation.

    This mirrors the selector's conservative bound-endpoint alias contract so
    structural expansion and diagnostics do not silently use a stricter entity
    language than grounding itself.
    """

    normalized = obligation_role_keys(obligation)
    keys: Set[str] = set()
    if not normalized["subject_is_variable"] and normalized["subject_key"]:
        keys.update(endpoint_alias_signatures(obligation.get("raw_subject") or obligation.get("subject")))
    if not normalized["object_is_variable"] and normalized["object_key"]:
        keys.update(endpoint_alias_signatures(obligation.get("raw_object") or obligation.get("object")))
    return keys


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
