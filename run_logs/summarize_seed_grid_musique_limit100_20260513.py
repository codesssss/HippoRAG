#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path("/mnt/nvme/code/HippoRAG")
METHOD = "evidence_transition_graphragv4_fact_witnessed_sto"
CLEAN = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
DENSE_PRESERVE = ROOT / "run_logs/etv4_dense_preserving_musique_limit100_20260513/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
GRID = ROOT / "run_logs/etv4_seed_grid_musique_limit100_20260513"
OPENIE = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index/openie_results_ner_qwen3-32b-judge.json"


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


def title_map() -> dict[int, str]:
    docs = load(OPENIE).get("docs", []) or []
    return {i: str(doc.get("passage", "")).splitlines()[0].strip() for i, doc in enumerate(docs)}


def norm(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def exact_r(top: list[int], gold: list[int]) -> float:
    g = set(gold)
    return len(set(top) & g) / len(g) if g else 0.0


def exact_all(top: list[int], gold: list[int]) -> int:
    g = set(gold)
    return int(bool(g) and g <= set(top))


def title_r(top: list[int], gold: list[int], titles: dict[int, str]) -> float:
    gt = {norm(titles.get(doc, "")) for doc in gold if norm(titles.get(doc, ""))}
    tt = {norm(titles.get(doc, "")) for doc in top if norm(titles.get(doc, ""))}
    return len(tt & gt) / len(gt) if gt else 0.0


def title_all(top: list[int], gold: list[int], titles: dict[int, str]) -> int:
    gt = {norm(titles.get(doc, "")) for doc in gold if norm(titles.get(doc, ""))}
    tt = {norm(titles.get(doc, "")) for doc in top if norm(titles.get(doc, ""))}
    return int(bool(gt) and gt <= tt)


def pool(row: dict[str, Any]) -> list[int]:
    return ints(((row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}).get("candidate_doc_indices", []) or [])[:200]


def summarize(name: str, path: Path, clean_rows: list[dict[str, Any]], titles: dict[int, str]) -> dict[str, Any]:
    rows = load(path)["rows"][:100]
    return {
        "name": name,
        "exact_r5": mean(exact_r(ints(r["retrieved_doc_indices_top5"]), ints(r["gold_doc_indices"])) for r in rows),
        "exact_all5": mean(exact_all(ints(r["retrieved_doc_indices_top5"]), ints(r["gold_doc_indices"])) for r in rows),
        "title_r5": mean(title_r(ints(r["retrieved_doc_indices_top5"]), ints(r["gold_doc_indices"]), titles) for r in rows),
        "title_all5": mean(title_all(ints(r["retrieved_doc_indices_top5"]), ints(r["gold_doc_indices"]), titles) for r in rows),
        "title_pool_all200": mean(title_all(pool(r), ints(r["gold_doc_indices"]), titles) for r in rows),
        "top5_diff_vs_clean": sum(
            tuple(ints(a["retrieved_doc_indices_top5"])) != tuple(ints(b["retrieved_doc_indices_top5"]))
            for a, b in zip(clean_rows, rows)
        ),
        "title_all_gain_vs_clean": sum(
            (not title_all(ints(a["retrieved_doc_indices_top5"]), ints(a["gold_doc_indices"]), titles))
            and title_all(ints(b["retrieved_doc_indices_top5"]), ints(b["gold_doc_indices"]), titles)
            for a, b in zip(clean_rows, rows)
        ),
        "title_all_loss_vs_clean": sum(
            title_all(ints(a["retrieved_doc_indices_top5"]), ints(a["gold_doc_indices"]), titles)
            and (not title_all(ints(b["retrieved_doc_indices_top5"]), ints(b["gold_doc_indices"]), titles))
            for a, b in zip(clean_rows, rows)
        ),
    }


def main() -> int:
    titles = title_map()
    clean_rows = load(CLEAN)["rows"][:100]
    specs = [
        ("clean", CLEAN),
        ("dense_preserve", DENSE_PRESERVE),
        ("roots50", GRID / f"dense_seeded_roots50/musique/reports/musique_{METHOD}_retrieval.json"),
        ("roots100", GRID / f"dense_seeded_roots100/musique/reports/musique_{METHOD}_retrieval.json"),
        ("roots200", GRID / f"dense_seeded_roots200/musique/reports/musique_{METHOD}_retrieval.json"),
    ]
    rows = [summarize(name, path, clean_rows, titles) for name, path in specs]
    print("Name            ExactR5  ExactAll5  TitleR5  TitleAll5  TitlePoolAll200  Diff  Gain/Loss")
    print("--------------  -------  ---------  -------  ---------  ---------------  ----  ---------")
    for r in rows:
        print(
            f"{r['name'].ljust(14)}  "
            f"{r['exact_r5']:.6f}  {r['exact_all5']:.6f}   "
            f"{r['title_r5']:.6f}  {r['title_all5']:.6f}   "
            f"{r['title_pool_all200']:.6f}         "
            f"{str(r['top5_diff_vs_clean']).rjust(4)}  "
            f"{r['title_all_gain_vs_clean']}/{r['title_all_loss_vs_clean']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
