"""Compatibility import for the source-layout AG-STO index utilities."""

from src.agsto.index import (
    build_corpus_unit_index,
    light_unit_endpoint_set,
    light_unit_text,
    light_units_for_doc,
    unit_title_tokens,
)

__all__ = [
    "build_corpus_unit_index",
    "light_unit_endpoint_set",
    "light_unit_text",
    "light_units_for_doc",
    "unit_title_tokens",
]
