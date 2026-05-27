#!/usr/bin/env python3
"""Strip row-level gold/answer fields from per-query graph-compare reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


STRIP_KEY_FRAGMENTS = (
    "gold",
    "answer",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def should_strip_row_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(fragment in lowered for fragment in STRIP_KEY_FRAGMENTS)


def strip_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    stripped: dict[str, Any] = {}
    removed: list[str] = []
    for key, value in row.items():
        if should_strip_row_key(str(key)):
            removed.append(str(key))
            continue
        stripped[str(key)] = value
    return stripped, removed


def strip_report(payload: Mapping[str, Any], *, limit_rows: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    output = dict(payload)
    variants = payload.get("variants", {})
    if not isinstance(variants, Mapping):
        raise ValueError("Expected report['variants'] to be a mapping.")

    stripped_variants: dict[str, Any] = {}
    removed_counts: dict[str, dict[str, int]] = {}
    row_counts: dict[str, int] = {}
    for variant_name, variant_payload in variants.items():
        if not isinstance(variant_payload, list):
            stripped_variants[str(variant_name)] = variant_payload
            continue
        rows = list(variant_payload)
        if int(limit_rows) > 0:
            rows = rows[: int(limit_rows)]
        stripped_rows: list[dict[str, Any]] = []
        per_variant_removed: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                stripped_rows.append({"value": row})
                continue
            stripped_row, removed_keys = strip_row(row)
            for key in removed_keys:
                per_variant_removed[key] = per_variant_removed.get(key, 0) + 1
            stripped_rows.append(stripped_row)
        stripped_variants[str(variant_name)] = stripped_rows
        removed_counts[str(variant_name)] = dict(sorted(per_variant_removed.items()))
        row_counts[str(variant_name)] = len(stripped_rows)

    output["variants"] = stripped_variants
    if int(limit_rows) > 0:
        output["limit"] = int(limit_rows)
        output["num_queries"] = int(limit_rows)
    output["gold_stripped"] = True
    output["gold_stripped_note"] = (
        "Row-level keys containing 'gold' or 'answer' were removed. "
        "Question text, retrieved doc indices, route traces, and non-gold metadata were preserved."
    )
    summary = {
        "limit_rows": int(limit_rows),
        "variant_row_counts": row_counts,
        "removed_counts": removed_counts,
    }
    output["gold_stripped_summary"] = summary
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-rows", type=int, default=0)
    args = parser.parse_args()

    output, summary = strip_report(load_json(args.input), limit_rows=max(int(args.limit_rows), 0))
    write_json(args.output, output)
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
