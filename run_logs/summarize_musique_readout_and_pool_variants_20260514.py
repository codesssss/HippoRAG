#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path("/mnt/nvme/code/HippoRAG")
METHOD = "evidence_transition_graphragv4_fact_witnessed_sto"
OPENIE = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index/openie_results_ner_qwen3-32b-judge.json"
CLEAN = ROOT / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
VARIANTS = {
    "clean_pool200": CLEAN,
    "dense_preserve_pool200": ROOT / "run_logs/etv4_dense_preserving_musique_limit100_20260513/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json",
    "clean_pool300": ROOT / "run_logs/etv4_pool300_musique_limit100_20260514_direct/clean_pool300/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json",
    "dense_preserve_pool300": ROOT / "run_logs/etv4_pool300_musique_limit100_20260514_direct/dense_preserve_pool300/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json",
    "path_cover": ROOT / "run_logs/etv4_readout_grid_direct_musique_limit100_20260514/fact_witnessed_path_cover/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json",
    "transition_valid": ROOT / "run_logs/etv4_readout_grid_direct_musique_limit100_20260514/transition_valid_closure/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json",
    "source_aligned": ROOT / "run_logs/etv4_readout_grid_direct_musique_limit100_20260514/source_aligned/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json",
}
OUT_DIR = ROOT / "run_logs/etv4_clean_root_cause_20260513"
OUT_JSON = OUT_DIR / "musique_readout_pool_variant_summary_20260514.json"
OUT_MD = OUT_DIR / "musique_readout_pool_variant_summary_20260514.md"


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


def title_map() -> dict[int, str]:
    docs = load(OPENIE).get("docs", []) or []
    return {int(i): str(doc.get("passage", "")).splitlines()[0].strip() for i, doc in enumerate(docs)}


def exact_r(top: list[int], gold: list[int]) -> float:
    g = set(gold)
    return len(set(top) & g) / len(g) if g else 0.0


def exact_all(top: list[int], gold: list[int]) -> int:
    g = set(gold)
    return int(bool(g) and g <= set(top))


def title_set(doc_ids: Iterable[int], titles: dict[int, str]) -> set[str]:
    return {norm(titles.get(int(doc), "")) for doc in doc_ids if norm(titles.get(int(doc), ""))}


def title_r(top: list[int], gold: list[int], titles: dict[int, str]) -> float:
    gt = title_set(gold, titles)
    return len(title_set(top, titles) & gt) / len(gt) if gt else 0.0


def title_all(top: list[int], gold: list[int], titles: dict[int, str]) -> int:
    gt = title_set(gold, titles)
    return int(bool(gt) and gt <= title_set(top, titles))


def pool(row: dict[str, Any]) -> list[int]:
    return ints(((row.get("route_trace", {}) or {}).get("candidate_universe", {}) or {}).get("candidate_doc_indices", []) or [])


def summarize(name: str, path: Path, clean_rows: list[dict[str, Any]], titles: dict[int, str]) -> dict[str, Any]:
    if not path.exists():
        return {"name": name, "status": "missing", "path": str(path)}
    rows = load(path)["rows"][:100]
    return {
        "name": name,
        "status": "done",
        "path": str(path),
        "exact_r5": mean(exact_r(ints(r["retrieved_doc_indices_top5"]), ints(r["gold_doc_indices"])) for r in rows),
        "exact_all5": mean(exact_all(ints(r["retrieved_doc_indices_top5"]), ints(r["gold_doc_indices"])) for r in rows),
        "title_r5": mean(title_r(ints(r["retrieved_doc_indices_top5"]), ints(r["gold_doc_indices"]), titles) for r in rows),
        "title_all5": mean(title_all(ints(r["retrieved_doc_indices_top5"]), ints(r["gold_doc_indices"]), titles) for r in rows),
        "title_pool_all": mean(title_all(pool(r), ints(r["gold_doc_indices"]), titles) for r in rows),
        "mean_pool_size": mean(len(pool(r)) for r in rows),
        "top5_diff_vs_clean": sum(
            tuple(ints(a["retrieved_doc_indices_top5"])) != tuple(ints(b["retrieved_doc_indices_top5"]))
            for a, b in zip(clean_rows, rows)
        ),
        "title_all_gain_vs_clean": sum(
            not title_all(ints(a["retrieved_doc_indices_top5"]), ints(a["gold_doc_indices"]), titles)
            and title_all(ints(b["retrieved_doc_indices_top5"]), ints(b["gold_doc_indices"]), titles)
            for a, b in zip(clean_rows, rows)
        ),
        "title_all_loss_vs_clean": sum(
            title_all(ints(a["retrieved_doc_indices_top5"]), ints(a["gold_doc_indices"]), titles)
            and not title_all(ints(b["retrieved_doc_indices_top5"]), ints(b["gold_doc_indices"]), titles)
            for a, b in zip(clean_rows, rows)
        ),
    }


def main() -> int:
    titles = title_map()
    clean_rows = load(CLEAN)["rows"][:100]
    rows = [summarize(name, path, clean_rows, titles) for name, path in VARIANTS.items()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# MuSiQue Readout/Pool Variant Summary",
        "",
        "| Variant | Status | Exact R@5 | Exact All@5 | Title R@5 | Title All@5 | Title Pool All | Pool Size | Top5 Diff | Gain/Loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "done":
            lines.append(f"| {row['name']} | {row['status']} |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| {row['name']} | done | {row['exact_r5']:.6f} | {row['exact_all5']:.6f} | "
            f"{row['title_r5']:.6f} | {row['title_all5']:.6f} | {row['title_pool_all']:.6f} | "
            f"{row['mean_pool_size']:.1f} | {row['top5_diff_vs_clean']} | "
            f"{row['title_all_gain_vs_clean']}/{row['title_all_loss_vs_clean']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(str(OUT_JSON))
    print(str(OUT_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
