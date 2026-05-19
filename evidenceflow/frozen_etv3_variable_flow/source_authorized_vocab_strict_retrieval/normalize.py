"""Normalization helpers for source-authorized vocab-strict retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Tuple


_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_PAREN_RE = re.compile(r"\([^)]*\)")
_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
}


def normalize_text(text: object) -> str:
    raw = unicodedata.normalize("NFKD", str(text or ""))
    raw = raw.encode("ascii", "ignore").decode("ascii")
    lowered = raw.lower()
    normalized = _NON_WORD_RE.sub(" ", lowered)
    return " ".join(normalized.split())


def tokens(text: object) -> Tuple[str, ...]:
    normalized = normalize_text(text)
    return tuple(normalized.split()) if normalized else ()


def title_aliases(title: object) -> Tuple[str, ...]:
    raw = str(title or "").strip()
    aliases = []
    comma_prefix = raw.split(",", 1)[0].strip() if "," in raw else ""
    for candidate in (raw, _PAREN_RE.sub("", raw), comma_prefix):
        normalized = normalize_text(candidate)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return tuple(aliases)


def phrase_occurs(phrase: object, text: object) -> bool:
    phrase_tokens = tokens(phrase)
    text_tokens = tokens(text)
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    width = len(phrase_tokens)
    for start in range(0, len(text_tokens) - width + 1):
        if tuple(text_tokens[start : start + width]) == phrase_tokens:
            return True
    return False


def is_specific_title_alias(alias: object) -> bool:
    alias_tokens = [token for token in tokens(alias) if token not in _STOP_TOKENS]
    if len(alias_tokens) >= 2:
        return True
    return bool(alias_tokens and len(alias_tokens[0]) >= 5)


def unique_ints(values: Iterable[object]) -> Tuple[int, ...]:
    seen = set()
    ordered = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)
