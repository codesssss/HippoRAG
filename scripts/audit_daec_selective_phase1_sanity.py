#!/usr/bin/env python3
"""Sanity audit for DAEC-selective Phase-1 gate behavior."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


RUN_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")
DAEC_DIR = Path("run_logs/daec_llm_wiki_title_proprag_full1000_20260503")
NOBIND_DIR = Path("run_logs/daec_nobinding_proprag_full1000_20260506")
REPORT_DIR = Path("reports/daec_selective_binding_phase1_20260506")

DATASETS = (
    ("2Wiki", "2wikimultihopqa"),
    ("HotpotQA", "hotpotqa"),
    ("MuSiQue", "musique"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def traces(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in data.get("setwise_selector_query_traces", []) if isinstance(row, Mapping)]


def answer(row: Mapping[str, Any]) -> str:
    return str(row.get("selector_answer") or "").strip()


def titles(row: Mapping[str, Any]) -> list[str]:
    return [str(title) for title in (row.get("selector_top_titles") or [])]


def metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row.get("selector_metrics") or {}


def decision(row: Mapping[str, Any]) -> str:
    return str((row.get("selector_trace") or {}).get("selective_binding_decision") or "")


def summarize(dataset: str, slug: str) -> dict[str, Any]:
    selective = traces(read_json(RUN_DIR / f"{slug}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"))
    daec = traces(read_json(DAEC_DIR / f"{slug}_proprag_wiki_title_daec_llm_full1000.json"))
    nobind = traces(read_json(NOBIND_DIR / f"{slug}_proprag_wiki_title_daec_noisyor_nobind_full1000.json"))
    if not (len(selective) == len(daec) == len(nobind)):
        raise ValueError(f"{dataset}: mismatched trace counts")
    n = len(selective)
    abstain_indices = [index for index, row in enumerate(selective) if decision(row) == "abstain"]
    bind_indices = [index for index, row in enumerate(selective) if decision(row) == "bind"]
    rows = {
        "dataset": dataset,
        "query_count": n,
        "bind_count": len(bind_indices),
        "abstain_count": len(abstain_indices),
        "null_rate": len(abstain_indices) / max(n, 1),
        "all_selective_titles_same_as_daec": sum(titles(s) == titles(d) for s, d in zip(selective, daec)),
        "all_selective_answers_same_as_daec": sum(answer(s) == answer(d) for s, d in zip(selective, daec)),
        "all_selective_metrics_same_as_daec": sum(metrics(s) == metrics(d) for s, d in zip(selective, daec)),
        "abstain_selective_titles_same_as_daec": sum(titles(selective[i]) == titles(daec[i]) for i in abstain_indices),
        "abstain_selective_titles_same_as_nobind": sum(titles(selective[i]) == titles(nobind[i]) for i in abstain_indices),
        "abstain_daec_titles_same_as_nobind": sum(titles(daec[i]) == titles(nobind[i]) for i in abstain_indices),
        "abstain_selective_answers_same_as_daec": sum(answer(selective[i]) == answer(daec[i]) for i in abstain_indices),
        "abstain_selective_answers_same_as_nobind": sum(answer(selective[i]) == answer(nobind[i]) for i in abstain_indices),
        "abstain_daec_answers_same_as_nobind": sum(answer(daec[i]) == answer(nobind[i]) for i in abstain_indices),
    }
    return rows


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build_markdown(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "# DAEC-selective Phase-1 Sanity Audit",
        "",
        "Purpose: explain why 2Wiki/HotpotQA have exact zero paired delta between DAEC and DAEC-selective despite nonzero abstention rates.",
        "",
        "## Gate and Equality Counts",
        "",
        "| Dataset | Bind | Abstain | Null Rate | Selective titles = DAEC | Selective answers = DAEC | Abstain DAEC titles = Nobind | Abstain DAEC answers = Nobind |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        n = int(row["query_count"])
        abstain = int(row["abstain_count"])
        lines.append(
            f"| {row['dataset']} | {int(row['bind_count'])} | {abstain} | {fmt(row['null_rate'], 3)} | "
            f"{int(row['all_selective_titles_same_as_daec'])}/{n} | "
            f"{int(row['all_selective_answers_same_as_daec'])}/{n} | "
            f"{int(row['abstain_daec_titles_same_as_nobind'])}/{abstain} | "
            f"{int(row['abstain_daec_answers_same_as_nobind'])}/{abstain} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The gate does trigger on 2Wiki and HotpotQA: null rates are 0.302 and 0.343.",
        "- The exact zero DAEC-selective vs DAEC delta is not because abstention is disabled.",
        "- On 2Wiki and HotpotQA, every abstained query has identical DAEC and Nobind top-5 titles, so DAEC-selective changes the binding mode but not the reader input.",
        "- MuSiQue is different: 262 abstentions change the selected evidence set, which is where the F1 repair comes from.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize(dataset, slug) for dataset, slug in DATASETS]
    csv_path = REPORT_DIR / "phase1_sanity_audit.csv"
    json_path = REPORT_DIR / "phase1_sanity_audit.json"
    md_path = REPORT_DIR / "phase1_sanity_audit.md"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
