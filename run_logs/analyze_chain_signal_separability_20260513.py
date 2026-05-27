#!/usr/bin/env python3
"""Offline separability checks for ETv4 MuSiQue chain-tail substitutions.

This is diagnostic-only. Gold labels are used only to score counterfactual
rules, not by the retriever.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DIAGNOSTIC_JSON = Path(
    "/mnt/nvme/code/HippoRAG/run_logs/"
    "etv4_clean_mainline_qwen32b_nothink_full1000_20260512/"
    "reports_chain_diagnostic_gainloss/"
    "etv4_musique_chain_consistency_diagnostic.json"
)


def _unique_ints(values: Sequence[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _hits(docs: Sequence[int], gold_docs: Sequence[int]) -> int:
    gold = set(_unique_ints(gold_docs))
    return len(set(_unique_ints(docs[:5])) & gold)


def _all_gold(docs: Sequence[int], gold_docs: Sequence[int]) -> int:
    gold = set(_unique_ints(gold_docs))
    if not gold:
        return 0
    return int(gold <= set(_unique_ints(docs[:5])))


def _metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    if key == "dense_hits":
        return sum(_hits(row["dense_top5"], row["gold_doc_indices"]) / len(set(row["gold_doc_indices"])) for row in rows) / len(rows)
    if key == "etv4_hits":
        return sum(_hits(row["etv4_top5"], row["gold_doc_indices"]) / len(set(row["gold_doc_indices"])) for row in rows) / len(rows)
    if key == "cf_hits":
        return sum(_hits(row["cf_top5"], row["gold_doc_indices"]) / len(set(row["gold_doc_indices"])) for row in rows) / len(rows)
    raise KeyError(key)


Predicate = Callable[[Mapping[str, Any]], bool]


def _counterfactual_top5(row: Mapping[str, Any], predicate: Predicate) -> tuple[int, ...]:
    selected = _unique_ints(row["etv4_top5"])[:5]
    dense = _unique_ints(row["dense_top5"])[:5]
    removed_queue = [doc for doc in dense if doc not in set(selected)]
    inserted_by_doc = {int(info["doc_index"]): info for info in row.get("inserted_docs", [])}

    out: list[int] = []
    seen: set[int] = set()
    for doc in selected:
        info = inserted_by_doc.get(int(doc))
        if info is not None and predicate(info):
            while removed_queue and removed_queue[0] in seen:
                removed_queue.pop(0)
            if removed_queue:
                replacement = int(removed_queue.pop(0))
                if replacement not in seen:
                    out.append(replacement)
                    seen.add(replacement)
                    continue
        if int(doc) not in seen:
            out.append(int(doc))
            seen.add(int(doc))

    for doc in dense + selected:
        if len(out) >= 5:
            break
        if int(doc) not in seen:
            out.append(int(doc))
            seen.add(int(doc))
    return tuple(out[:5])


def _is_weak_graph(info: Mapping[str, Any]) -> bool:
    return str(info.get("evidence_class")) in {"weak_graph_connected", "isolated_fill"}


def _is_weak_no_new_coverage(info: Mapping[str, Any]) -> bool:
    return _is_weak_graph(info) and not info.get("new_query_coverage")


def _is_no_evidence_tier(info: Mapping[str, Any]) -> bool:
    return int(info.get("evidence_tier_degree", 0) or 0) == 0


def _is_weak_low_fact(info: Mapping[str, Any]) -> bool:
    return _is_weak_graph(info) and int(info.get("fact_witness_degree", 0) or 0) <= 1


def _is_supported_leaf_no_later(info: Mapping[str, Any]) -> bool:
    return (
        str(info.get("class")) == "supported_leaf"
        and int(info.get("later_chain_strong_degree", 0) or 0) == 0
    )


def _summarize_rule(name: str, rows: Sequence[Mapping[str, Any]], predicate: Predicate) -> Mapping[str, Any]:
    changed_rows: list[dict[str, Any]] = []
    for row in rows:
        cf = _counterfactual_top5(row, predicate)
        if tuple(_unique_ints(row["etv4_top5"])[:5]) == cf:
            continue
        before_hits = _hits(row["etv4_top5"], row["gold_doc_indices"])
        after_hits = _hits(cf, row["gold_doc_indices"])
        before_all = _all_gold(row["etv4_top5"], row["gold_doc_indices"])
        after_all = _all_gold(cf, row["gold_doc_indices"])
        changed = dict(row)
        changed["cf_top5"] = cf
        changed["cf_delta_hits"] = after_hits - before_hits
        changed["cf_delta_all"] = after_all - before_all
        changed_rows.append(changed)

    delta_counter = Counter(int(row["cf_delta_hits"]) for row in changed_rows)
    all_counter = Counter(int(row["cf_delta_all"]) for row in changed_rows)
    by_hop_bucket = Counter((row["hop"], row["bucket"]) for row in changed_rows)
    return {
        "rule": name,
        "changed_rows": len(changed_rows),
        "delta_hit_sum": sum(int(row["cf_delta_hits"]) for row in changed_rows),
        "delta_all_sum": sum(int(row["cf_delta_all"]) for row in changed_rows),
        "hit_delta_counts": dict(sorted(delta_counter.items())),
        "all_delta_counts": dict(sorted(all_counter.items())),
        "changed_by_hop_bucket": {f"hop{hop}_{bucket}": count for (hop, bucket), count in sorted(by_hop_bucket.items())},
        "sample_bad": [
            {
                "query_index": row["query_index"],
                "hop": row["hop"],
                "bucket": row["bucket"],
                "delta_hits": row["cf_delta_hits"],
                "dense_top5": row["dense_top5"],
                "etv4_top5": row["etv4_top5"],
                "cf_top5": row["cf_top5"],
                "gold": row["gold_doc_indices"],
            }
            for row in changed_rows
            if int(row["cf_delta_hits"]) < 0
        ][:5],
        "sample_good": [
            {
                "query_index": row["query_index"],
                "hop": row["hop"],
                "bucket": row["bucket"],
                "delta_hits": row["cf_delta_hits"],
                "dense_top5": row["dense_top5"],
                "etv4_top5": row["etv4_top5"],
                "cf_top5": row["cf_top5"],
                "gold": row["gold_doc_indices"],
            }
            for row in changed_rows
            if int(row["cf_delta_hits"]) > 0
        ][:5],
    }


def main() -> int:
    payload = json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
    rows = list(payload.get("query_rows", []) or [])
    rules: list[tuple[str, Predicate]] = [
        ("replace_weak_graph_or_isolated", _is_weak_graph),
        ("replace_weak_graph_no_new_query_coverage", _is_weak_no_new_coverage),
        ("replace_no_evidence_tier", _is_no_evidence_tier),
        ("replace_weak_low_fact_degree", _is_weak_low_fact),
        ("replace_supported_leaf_no_later", _is_supported_leaf_no_later),
    ]
    summary = {
        "source": str(DIAGNOSTIC_JSON),
        "scope": "gain_loss_rows_only",
        "row_count": len(rows),
        "baseline": {
            "dense_mean_recall": _metric(rows, "dense_hits"),
            "etv4_mean_recall": _metric(rows, "etv4_hits"),
        },
        "rules": [_summarize_rule(name, rows, pred) for name, pred in rules],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
