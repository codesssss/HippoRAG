#!/usr/bin/env python3
"""Summarize EvLink mechanism audits from existing run artifacts.

This script intentionally does not call any model.  It reads saved pool and
PCEC/reader reports to quantify two diagnostics:

1. Pool-to-top5 conversion: how much gold evidence is present in the candidate
   pool versus what the method delivers to the top-5 prefix.
2. Transition-witness wins: among PCEC changes, whether the admitted document
   came from the EvLink graph tail and whether it repaired missing gold
   evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASETS: Sequence[str] = ("hotpotqa", "2wikimultihopqa", "musique")
DATASET_LABELS: Mapping[str, str] = {
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultiHopQA",
    "musique": "MuSiQue",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def title_set(values: Iterable[Any]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def recall_and_all(gold_titles: Sequence[Any], retrieved_titles: Sequence[Any]) -> tuple[float, bool]:
    gold = title_set(gold_titles)
    retrieved = title_set(retrieved_titles)
    if not gold:
        return 0.0, False
    overlap = len(gold & retrieved)
    return overlap / len(gold), overlap == len(gold)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> float:
    return round(100.0 * float(value), 2)


def summarize_pool_records(pool_path: Path, *, top_k: int, pool_k: int) -> dict[str, Any]:
    payload = read_json(pool_path)
    records = list(payload.get("records") or [])
    r_top: list[float] = []
    all_top: list[float] = []
    r_pool: list[float] = []
    all_pool: list[float] = []
    for row in records:
        gold = list(row.get("gold_titles") or [])
        titles = list(row.get("pool_titles") or [])
        top_recall, top_all = recall_and_all(gold, titles[:top_k])
        pool_recall, pool_all_gold = recall_and_all(gold, titles[:pool_k])
        r_top.append(top_recall)
        all_top.append(float(top_all))
        r_pool.append(pool_recall)
        all_pool.append(float(pool_all_gold))
    return {
        "count": len(records),
        "top5_r": mean(r_top),
        "top5_all": mean(all_top),
        "pool_r": mean(r_pool),
        "pool_all": mean(all_pool),
        "source": str(pool_path),
    }


def summarize_pcec_report(report_path: Path) -> dict[str, Any]:
    payload = read_json(report_path)
    rows = list(payload.get("rows") or [])
    r_top: list[float] = []
    all_top: list[float] = []
    baseline_r_top: list[float] = []
    baseline_all_top: list[float] = []
    for row in rows:
        gold = list(row.get("gold_titles") or [])
        final_titles = list(row.get("retrieved_titles_top5") or [])
        baseline_titles = list(row.get("baseline_titles_top5") or [])
        final_recall, final_all = recall_and_all(gold, final_titles)
        baseline_recall, baseline_all = recall_and_all(gold, baseline_titles)
        r_top.append(final_recall)
        all_top.append(float(final_all))
        baseline_r_top.append(baseline_recall)
        baseline_all_top.append(float(baseline_all))
    summary = dict(payload.get("summary") or {})
    return {
        "count": len(rows),
        "top5_r": float(summary.get("pcec_title_recall_top5", mean(r_top))),
        "top5_all": float(summary.get("pcec_title_all_gold_top5", mean(all_top))),
        "baseline_top5_r": mean(baseline_r_top),
        "baseline_top5_all": mean(baseline_all_top),
        "changed_count": int(summary.get("changed_count", 0) or 0),
        "admit_count": int(summary.get("admit_count", 0) or 0),
        "source": str(report_path),
    }


def evidence_link_paths(root: Path, dataset: str) -> tuple[Path, Path]:
    pool = (
        root
        / "run_logs/etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514"
        / "pools"
        / f"{dataset}_etv4_pool100_limit1000.json"
    )
    pcec = (
        root
        / "run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
        / "pcec/etv4/evals"
        / f"{dataset}_pcec_native_pool_prefix4_residual1_pool100_limit1000.json"
    )
    return pool, pcec


def phrase_source_paths(root: Path, dataset: str) -> tuple[Path, Path]:
    run_root = root / "run_logs/evidencelink_phrase_source_only_unified_qwen32b_gpt4omini_full1000_20260519"
    pool = run_root / "pools" / f"{dataset}_phrase_source_only_pool100_limit1000.json"
    pcec = (
        run_root
        / "pcec/phrase_source/evals"
        / f"{dataset}_phrase_source_only_pcec_native_pool_prefix4_residual1_pool100_limit1000.json"
    )
    return pool, pcec


def hipporag_pool_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "run_logs/hipporag_qwen32b_valid_graph_top200_full1000_20260513_r2"
        / "pools/hipporag"
        / f"{dataset}_hipporag_qwen32b_valid_graph_pool200.json"
    )


def proprag_pool_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512"
        / "pools/proprag"
        / f"{dataset}_proprag_qwen32b_nothink_pool200.json"
    )


def append_conversion_row(
    rows: list[dict[str, Any]],
    *,
    method: str,
    dataset: str,
    pool_path: Path,
    pool_k: int,
    pcec_path: Path | None = None,
) -> None:
    pool = summarize_pool_records(pool_path, top_k=5, pool_k=pool_k)
    if pcec_path is not None:
        top = summarize_pcec_report(pcec_path)
        delivered_r = top["top5_r"]
        delivered_all = top["top5_all"]
        native_top5_r = top["baseline_top5_r"]
        native_top5_all = top["baseline_top5_all"]
        changed_count = top["changed_count"]
        top_source = top["source"]
    else:
        delivered_r = pool["top5_r"]
        delivered_all = pool["top5_all"]
        native_top5_r = pool["top5_r"]
        native_top5_all = pool["top5_all"]
        changed_count = 0
        top_source = pool["source"]
    rows.append(
        {
            "method": method,
            "dataset": DATASET_LABELS.get(dataset, dataset),
            "count": pool["count"],
            "pool_k": pool_k,
            "native_top5_r_pct": pct(native_top5_r),
            "native_top5_all_pct": pct(native_top5_all),
            "delivered_top5_r_pct": pct(delivered_r),
            "delivered_top5_all_pct": pct(delivered_all),
            "pool_recall_pct": pct(pool["pool_r"]),
            "pool_all_pct": pct(pool["pool_all"]),
            "pool_to_delivered_r_gap_pct": pct(pool["pool_r"] - delivered_r),
            "pool_to_delivered_all_gap_pct": pct(pool["pool_all"] - delivered_all),
            "changed_count": changed_count,
            "pool_source": pool["source"],
            "top5_source": top_source,
        }
    )


def build_conversion_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        pool, pcec = evidence_link_paths(root, dataset)
        append_conversion_row(
            rows,
            method="EvLink",
            dataset=dataset,
            pool_path=pool,
            pool_k=100,
            pcec_path=pcec,
        )
        phrase_pool, phrase_pcec = phrase_source_paths(root, dataset)
        if phrase_pool.exists() and phrase_pcec.exists():
            append_conversion_row(
                rows,
                method="w/o evidence-linked transitions",
                dataset=dataset,
                pool_path=phrase_pool,
                pool_k=100,
                pcec_path=phrase_pcec,
            )
        append_conversion_row(
            rows,
            method="HippoRAG2",
            dataset=dataset,
            pool_path=hipporag_pool_path(root, dataset),
            pool_k=200,
        )
        append_conversion_row(
            rows,
            method="PropRAG",
            dataset=dataset,
            pool_path=proprag_pool_path(root, dataset),
            pool_k=200,
        )
    return rows


def build_witness_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        pool_path, pcec_path = evidence_link_paths(root, dataset)
        pool_payload = read_json(pool_path)
        pcec_payload = read_json(pcec_path)
        pool_by_query = {
            int(row.get("query_idx", row.get("query_index", idx))): row
            for idx, row in enumerate(pool_payload.get("records") or [])
        }
        pcec_rows = list(pcec_payload.get("rows") or [])
        admit_decisions = 0
        added_total = 0
        added_from_graph_tail = 0
        added_from_source_prefix = 0
        added_from_seed = 0
        added_gold = 0
        added_gold_from_graph_tail = 0
        all_gold_rescues = 0
        all_gold_rescues_from_graph_tail = 0
        all_gold_regressions = 0
        recall_improvements = 0
        recall_regressions = 0
        no_change_already_all = 0
        no_change_missing_all = 0

        for row in pcec_rows:
            query_idx = int(row.get("query_index", row.get("query_idx", 0)))
            pool_row = pool_by_query.get(query_idx, {})
            agsto = dict(pool_row.get("agsto") or {})
            pool_doc_ids = [int(x) for x in pool_row.get("pool_doc_ids", [])]
            baseline_ids = pool_doc_ids[:5]
            final_ids = [int(x) for x in row.get("retrieved_doc_indices_top5", [])]
            gold_ids = {int(x) for x in row.get("gold_doc_indices", [])}
            graph_tail = {int(x) for x in agsto.get("agsto_graph_tail_doc_indices", [])}
            source_prefix = {int(x) for x in agsto.get("source_prior_prefix_doc_indices", [])}
            seeds = {int(x) for x in agsto.get("agsto_seed_doc_indices", [])}
            baseline_titles = list(row.get("baseline_titles_top5") or [])
            final_titles = list(row.get("retrieved_titles_top5") or [])
            gold_titles = list(row.get("gold_titles") or [])
            baseline_r, baseline_all = recall_and_all(gold_titles, baseline_titles)
            final_r, final_all = recall_and_all(gold_titles, final_titles)

            decision = str((row.get("pcec_readout") or {}).get("decision") or "")
            if decision != "admit":
                if baseline_all:
                    no_change_already_all += 1
                else:
                    no_change_missing_all += 1
                continue

            admit_decisions += 1
            if final_r > baseline_r:
                recall_improvements += 1
            elif final_r < baseline_r:
                recall_regressions += 1
            if (not baseline_all) and final_all:
                all_gold_rescues += 1
            if baseline_all and not final_all:
                all_gold_regressions += 1

            added = [doc_id for doc_id in final_ids if doc_id not in set(baseline_ids)]
            added_total += len(added)
            added_has_graph_tail = False
            for doc_id in added:
                in_tail = doc_id in graph_tail
                if in_tail:
                    added_from_graph_tail += 1
                    added_has_graph_tail = True
                if doc_id in source_prefix:
                    added_from_source_prefix += 1
                if doc_id in seeds:
                    added_from_seed += 1
                if doc_id in gold_ids:
                    added_gold += 1
                    if in_tail:
                        added_gold_from_graph_tail += 1
            if (not baseline_all) and final_all and added_has_graph_tail:
                all_gold_rescues_from_graph_tail += 1

        count = len(pcec_rows)
        summary = dict(pcec_payload.get("summary") or {})
        rows.append(
            {
                "dataset": DATASET_LABELS.get(dataset, dataset),
                "count": count,
                "admit_decisions": admit_decisions,
                "admit_pct": pct(admit_decisions / count if count else 0.0),
                "summary_changed_count": int(summary.get("changed_count", 0) or 0),
                "summary_admit_count": int(summary.get("admit_count", 0) or 0),
                "added_docs": added_total,
                "added_from_graph_tail": added_from_graph_tail,
                "added_from_graph_tail_pct": pct(added_from_graph_tail / added_total if added_total else 0.0),
                "added_from_source_prefix": added_from_source_prefix,
                "added_from_seed": added_from_seed,
                "added_gold": added_gold,
                "added_gold_from_graph_tail": added_gold_from_graph_tail,
                "all_gold_rescues": all_gold_rescues,
                "all_gold_rescues_from_graph_tail": all_gold_rescues_from_graph_tail,
                "all_gold_regressions": all_gold_regressions,
                "recall_improvements": recall_improvements,
                "recall_regressions": recall_regressions,
                "no_change_already_all": no_change_already_all,
                "no_change_missing_all": no_change_missing_all,
                "pool_source": str(pool_path),
                "pcec_source": str(pcec_path),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, conversion_rows: Sequence[Mapping[str, Any]], witness_rows: Sequence[Mapping[str, Any]]) -> None:
    conversion_columns = [
        "method",
        "dataset",
        "delivered_top5_r_pct",
        "delivered_top5_all_pct",
        "pool_recall_pct",
        "pool_all_pct",
        "pool_to_delivered_r_gap_pct",
        "pool_to_delivered_all_gap_pct",
        "changed_count",
    ]
    witness_columns = [
        "dataset",
        "admit_decisions",
        "added_docs",
        "added_from_graph_tail",
        "added_from_graph_tail_pct",
        "added_gold",
        "added_gold_from_graph_tail",
        "all_gold_rescues",
        "all_gold_rescues_from_graph_tail",
        "all_gold_regressions",
    ]
    text = "\n\n".join(
        [
            "# EvLink Mechanism Audits",
            "## Pool-to-Top5 Conversion",
            markdown_table(conversion_rows, conversion_columns),
            "## Transition-Witness Win Audit",
            markdown_table(witness_rows, witness_columns),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("run_logs/evidencelink_mechanism_audits_20260520"),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_root = args.output_root
    conversion_rows = build_conversion_rows(root)
    witness_rows = build_witness_rows(root)
    payload = {
        "protocol": {
            "model_calls": "none",
            "pool_to_top5": "compare saved candidate pools against delivered top-5 prefixes",
            "witness_win": "compare PCEC final top-5 against pool-native top-5 and audit added document origin",
        },
        "conversion_rows": conversion_rows,
        "witness_rows": witness_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mechanism_audits.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_root / "pool_to_top5_conversion.csv", conversion_rows)
    write_csv(output_root / "transition_witness_win_audit.csv", witness_rows)
    write_markdown(output_root / "mechanism_audits.md", conversion_rows, witness_rows)
    print(json.dumps({"output_root": str(output_root), "conversion_rows": len(conversion_rows), "witness_rows": len(witness_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
