#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Iterable


ROOT = Path("/mnt/nvme/code/HippoRAG")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_transition_graphragv4_fact_witnessed_sto.agsto.index import build_corpus_unit_index
from evidence_transition_graphragv4_fact_witnessed_sto.agsto.lexical import score_docs_bm25


METHOD = "evidence_transition_graphragv4_fact_witnessed_sto"
OPENIE = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index/openie_results_ner_qwen3-32b-judge.json"
CLEAN = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
POOL300 = ROOT / "run_logs/etv4_pool300_musique_limit100_20260514_direct/clean_pool300/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
OUT_DIR = ROOT / "run_logs/etv4_clean_root_cause_20260513"
OUT_JSON = OUT_DIR / "musique_textual_coverage_readout_probe_20260514.json"
OUT_MD = OUT_DIR / "musique_textual_coverage_readout_probe_20260514.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ints(values: Iterable[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values or []:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def norm(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def title_set(doc_ids: Iterable[int], titles: dict[int, str]) -> set[str]:
    return {norm(titles.get(int(doc), "")) for doc in doc_ids if norm(titles.get(int(doc), ""))}


def exact_r(top: list[int], gold: list[int]) -> float:
    g = set(gold)
    return len(set(top) & g) / len(g) if g else 0.0


def exact_all(top: list[int], gold: list[int]) -> int:
    g = set(gold)
    return int(bool(g) and g <= set(top))


def title_r(top: list[int], gold: list[int], titles: dict[int, str]) -> float:
    gt = title_set(gold, titles)
    return len(title_set(top, titles) & gt) / len(gt) if gt else 0.0


def title_all(top: list[int], gold: list[int], titles: dict[int, str]) -> int:
    gt = title_set(gold, titles)
    return int(bool(gt) and gt <= title_set(top, titles))


def pool(row: dict[str, Any]) -> list[int]:
    return ints(((row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}).get("candidate_doc_indices", []) or [])


def source_prefix(row: dict[str, Any]) -> list[int]:
    return ints(((row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}).get("source_prior_prefix_doc_indices", []) or [])


def build_index_and_titles() -> tuple[dict[str, Any], dict[int, str]]:
    payload = load(OPENIE)
    docs = payload.get("docs", []) or []
    openie_docs = []
    titles: dict[int, str] = {}
    for i, doc in enumerate(docs):
        passage = str(doc.get("passage", "") or "")
        title = passage.splitlines()[0].strip() if passage else ""
        titles[int(i)] = title
        openie_docs.append(
            {
                "idx": str(i),
                "passage": passage,
                "title": title,
                "extracted_triples": doc.get("extracted_triples", []) or [],
            }
        )
    return build_corpus_unit_index(openie_docs), titles


def ranked_by_bm25(query: str, candidates: list[int], scores: dict[int, float]) -> list[int]:
    rank = {int(doc): pos for pos, doc in enumerate(candidates)}
    return [
        int(doc)
        for doc in sorted(
            candidates,
            key=lambda doc: (
                -float(scores.get(int(doc), 0.0)),
                int(rank.get(int(doc), 10**9)),
                int(doc),
            ),
        )
    ]


def unique_take(*groups: Iterable[int], k: int = 5) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for doc in group:
            item = int(doc)
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
            if len(out) >= k:
                return out
    return out


def gold_title_best_rank(order: list[int], gold: list[int], titles: dict[int, str]) -> int | None:
    gt = title_set(gold, titles)
    if not gt:
        return None
    best = None
    for rank, doc in enumerate(order, start=1):
        if norm(titles.get(int(doc), "")) in gt:
            best = rank if best is None else min(best, rank)
    return best


def evaluate_rows(rows: list[dict[str, Any]], corpus_index: dict[str, Any], titles: dict[int, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies: dict[str, list[list[int]]] = {
        "clean": [],
        "source_prefix": [],
        "bm25_pool": [],
        "clean1_bm25": [],
        "clean2_bm25": [],
        "clean3_bm25": [],
        "clean4_bm25": [],
        "clean4_source_tail": [],
        "source1_clean2_bm25": [],
        "source2_clean2_bm25": [],
    }
    rank_rows: list[dict[str, Any]] = []
    for row in rows[:100]:
        query = str(row.get("question") or "")
        gold = ints(row.get("gold_doc_indices", []))
        clean_top = ints(row.get("retrieved_doc_indices_top5", []))
        candidates = pool(row)
        prefix = source_prefix(row) or candidates[:5]
        scores = score_docs_bm25(query=query, corpus_index=corpus_index)
        bm25_order = ranked_by_bm25(query, candidates, scores)

        policies["clean"].append(unique_take(clean_top, candidates, k=5))
        policies["source_prefix"].append(unique_take(prefix, candidates, k=5))
        policies["bm25_pool"].append(unique_take(bm25_order, candidates, k=5))
        policies["clean1_bm25"].append(unique_take(clean_top[:1], bm25_order, clean_top, candidates, k=5))
        policies["clean2_bm25"].append(unique_take(clean_top[:2], bm25_order, clean_top, candidates, k=5))
        policies["clean3_bm25"].append(unique_take(clean_top[:3], bm25_order, clean_top, candidates, k=5))
        policies["clean4_bm25"].append(unique_take(clean_top[:4], bm25_order, clean_top, candidates, k=5))
        policies["clean4_source_tail"].append(unique_take(clean_top[:4], prefix, candidates, k=5))
        policies["source1_clean2_bm25"].append(unique_take(prefix[:1], clean_top[:2], bm25_order, clean_top, candidates, k=5))
        policies["source2_clean2_bm25"].append(unique_take(prefix[:2], clean_top[:2], bm25_order, clean_top, candidates, k=5))

        rank_rows.append(
            {
                "query_index": int(row["query_index"]),
                "question": query,
                "clean_title_all": bool(title_all(clean_top, gold, titles)),
                "pool_title_all": bool(title_all(candidates, gold, titles)),
                "best_gold_title_rank_in_bm25_pool": gold_title_best_rank(bm25_order, gold, titles),
                "best_gold_title_rank_in_candidate_order": gold_title_best_rank(candidates, gold, titles),
                "best_gold_title_rank_in_clean_order": gold_title_best_rank(clean_top, gold, titles),
            }
        )

    summaries: list[dict[str, Any]] = []
    for name, top_rows in policies.items():
        summaries.append(
            {
                "policy": name,
                "exact_r5": mean(exact_r(top, ints(row.get("gold_doc_indices", []))) for top, row in zip(top_rows, rows[:100])),
                "exact_all5": mean(exact_all(top, ints(row.get("gold_doc_indices", []))) for top, row in zip(top_rows, rows[:100])),
                "title_r5": mean(title_r(top, ints(row.get("gold_doc_indices", [])), titles) for top, row in zip(top_rows, rows[:100])),
                "title_all5": mean(title_all(top, ints(row.get("gold_doc_indices", [])), titles) for top, row in zip(top_rows, rows[:100])),
                "top5_diff_vs_clean": sum(tuple(top) != tuple(policies["clean"][idx]) for idx, top in enumerate(top_rows)),
                "title_all_gain_vs_clean": sum(
                    not title_all(policies["clean"][idx], ints(rows[idx].get("gold_doc_indices", [])), titles)
                    and title_all(top, ints(rows[idx].get("gold_doc_indices", [])), titles)
                    for idx, top in enumerate(top_rows)
                ),
                "title_all_loss_vs_clean": sum(
                    title_all(policies["clean"][idx], ints(rows[idx].get("gold_doc_indices", [])), titles)
                    and not title_all(top, ints(rows[idx].get("gold_doc_indices", [])), titles)
                    for idx, top in enumerate(top_rows)
                ),
            }
        )
    return summaries, rank_rows


def main() -> int:
    corpus_index, titles = build_index_and_titles()
    output = {}
    for name, path in [("pool200", CLEAN), ("pool300", POOL300)]:
        rows = load(path)["rows"][:100]
        summaries, rank_rows = evaluate_rows(rows, corpus_index, titles)
        output[name] = {
            "summaries": summaries,
            "gold_rank_rows": rank_rows,
            "pool_title_all": mean(title_all(pool(row), ints(row.get("gold_doc_indices", [])), titles) for row in rows),
            "clean_title_all": mean(title_all(ints(row.get("retrieved_doc_indices_top5", [])), ints(row.get("gold_doc_indices", [])), titles) for row in rows),
            "mean_best_gold_title_rank_in_bm25_when_pool_has_all": mean(
                row["best_gold_title_rank_in_bm25_pool"]
                for row in rank_rows
                if row["pool_title_all"] and row["best_gold_title_rank_in_bm25_pool"] is not None
            ),
        }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# MuSiQue Textual Coverage Readout Probe", ""]
    for block_name, block in output.items():
        lines.extend(
            [
                f"## {block_name}",
                "",
                f"- clean title all@5: `{block['clean_title_all']:.6f}`",
                f"- pool title all: `{block['pool_title_all']:.6f}`",
                f"- mean best gold-title BM25 rank when pool complete: `{block['mean_best_gold_title_rank_in_bm25_when_pool_has_all']:.2f}`",
                "",
                "| Policy | Exact R@5 | Exact All@5 | Title R@5 | Title All@5 | Diff | Gain/Loss |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in block["summaries"]:
            lines.append(
                f"| {row['policy']} | {row['exact_r5']:.6f} | {row['exact_all5']:.6f} | "
                f"{row['title_r5']:.6f} | {row['title_all5']:.6f} | "
                f"{row['top5_diff_vs_clean']} | {row['title_all_gain_vs_clean']}/{row['title_all_loss_vs_clean']} |"
            )
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(str(OUT_JSON))
    print(str(OUT_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
