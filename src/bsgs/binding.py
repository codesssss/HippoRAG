"""Hard binding helpers for Week-1 BSGS."""

from __future__ import annotations

import re
import string
from typing import Iterable


_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def normalize_entity(text: str) -> str:
    value = str(text or "").lower().translate(_PUNCT_TABLE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def build_alias_map(aliases: dict[str, Iterable[str]] | None = None) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, values in (aliases or {}).items():
        norm_canonical = normalize_entity(canonical)
        if not norm_canonical:
            continue
        alias_map[norm_canonical] = norm_canonical
        for alias in values:
            norm_alias = normalize_entity(alias)
            if norm_alias:
                alias_map[norm_alias] = norm_canonical
    return alias_map


def canonicalize_entity(text: str, alias_map: dict[str, str] | None = None) -> str:
    normalized = normalize_entity(text)
    if not alias_map:
        return normalized
    return alias_map.get(normalized, normalized)


def hard_entity_match(left: str, right: str, alias_map: dict[str, str] | None = None) -> bool:
    return bool(canonicalize_entity(left, alias_map)) and (
        canonicalize_entity(left, alias_map) == canonicalize_entity(right, alias_map)
    )


def any_mention_matches(
    mentions: Iterable[str],
    targets: Iterable[str],
    alias_map: dict[str, str] | None = None,
) -> bool:
    target_set = {canonicalize_entity(target, alias_map) for target in targets if normalize_entity(target)}
    return any(canonicalize_entity(mention, alias_map) in target_set for mention in mentions)
