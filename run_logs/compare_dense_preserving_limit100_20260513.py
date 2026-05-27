#!/usr/bin/env python3
from __future__ import annotations

import json
from statistics import mean


CLEAN = "/mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
DP = "/mnt/nvme/code/HippoRAG/run_logs/etv4_dense_preserving_musique_limit100_20260513/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"


def hit(top: list[int], gold: list[int]) -> float:
    gold_set = set(gold)
    return len(set(top) & gold_set) / len(gold_set) if gold_set else 0.0


def allg(top: list[int], gold: list[int]) -> int:
    gold_set = set(gold)
    return int(bool(gold_set) and gold_set <= set(top))


def pool(row: dict) -> list[int]:
    return list((row.get("route_trace", {}) or {}).get("candidate_universe", {}).get("candidate_doc_indices", []) or [])[:200]


def main() -> int:
    clean_payload = json.load(open(CLEAN))
    dp_payload = json.load(open(DP))
    clean = clean_payload["rows"][:100]
    dp = dp_payload["rows"]
    print("dense_preserving metrics", dp_payload["metrics"])
    for name, rows in (("clean", clean), ("dense_preserve", dp)):
        print(
            name,
            "r5",
            round(mean(hit(r["retrieved_doc_indices_top5"], r["gold_doc_indices"]) for r in rows), 6),
            "all5",
            round(mean(allg(r["retrieved_doc_indices_top5"], r["gold_doc_indices"]) for r in rows), 6),
            "pool_all200",
            round(mean(allg(pool(r), r["gold_doc_indices"]) for r in rows), 6),
        )
    print(
        "top5_diffs",
        sum(tuple(a["retrieved_doc_indices_top5"]) != tuple(b["retrieved_doc_indices_top5"]) for a, b in zip(clean, dp)),
    )
    pool_gain = sum(
        (not allg(pool(a), a["gold_doc_indices"])) and allg(pool(b), b["gold_doc_indices"])
        for a, b in zip(clean, dp)
    )
    pool_loss = sum(
        allg(pool(a), a["gold_doc_indices"]) and (not allg(pool(b), b["gold_doc_indices"]))
        for a, b in zip(clean, dp)
    )
    print("pool_all_gain_loss", pool_gain, pool_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
