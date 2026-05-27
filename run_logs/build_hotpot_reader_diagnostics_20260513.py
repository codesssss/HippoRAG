#!/usr/bin/env python3
"""Build HotpotQA reader diagnostic reports.

These reports are oracle diagnostics only. They must not be mixed into method
tables because they use gold document ids to isolate reader/order effects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/mnt/nvme/code/HippoRAG")
SRC = (
    ROOT
    / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
    / "hotpotqa/reports/hotpotqa_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
)
OUT_DIR = ROOT / "run_logs/hotpotqa_reader_diagnostics_20260513/inputs"


def unique_ints(values: Iterable[Any], *, limit: int | None = None) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        try:
            doc = int(value)
        except (TypeError, ValueError):
            continue
        if doc in seen:
            continue
        seen.add(doc)
        out.append(doc)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def transform(payload: dict[str, Any], *, policy: str) -> dict[str, Any]:
    rows = []
    for row in payload.get("rows", []) or []:
        copied = dict(row)
        original_top5 = unique_ints(copied.get("retrieved_doc_indices_top5", []) or [])
        gold = unique_ints(copied.get("gold_doc_indices", []) or [])
        if policy == "gold_first":
            doc_indices = unique_ints([*gold, *original_top5], limit=5)
        elif policy == "gold_only":
            doc_indices = gold
        else:
            raise ValueError(f"unknown policy: {policy}")
        copied["retrieved_doc_indices_top5"] = doc_indices
        copied["reader_context_doc_indices"] = doc_indices
        copied["reader_context_policy"] = f"oracle_{policy}"
        copied["oracle_reader_diagnostic"] = True
        rows.append(copied)

    return {
        **payload,
        "method": f"{payload.get('method', 'etv4')}_hotpot_oracle_{policy}",
        "config": {
            **dict(payload.get("config", {}) or {}),
            "oracle_reader_diagnostic": True,
            "reader_diagnostic_policy": policy,
            "not_a_method_result": True,
        },
        "reader_diagnostic": {
            "policy": policy,
            "uses_gold_doc_indices": True,
            "not_a_method_result": True,
            "purpose": "isolate HotpotQA reader ordering/distractor sensitivity",
        },
        "rows": rows,
    }


def main() -> int:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for policy in ("gold_first", "gold_only"):
        out = OUT_DIR / f"hotpotqa_etv4_oracle_{policy}_reader_diagnostic.json"
        out.write_text(
            json.dumps(transform(payload, policy=policy), ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
