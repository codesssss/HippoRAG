#!/usr/bin/env python3
"""Summarize reader-only evaluation for repair-gated DBEC pools."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import (  # noqa: E402
    BOOTSTRAP_SEED,
    SUPPORT_ROWS,
    paired_bootstrap_delta,
    read_json,
    safe_float,
    safe_int,
)
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score  # noqa: E402


RUN_DIR = Path("run_logs/repair_gated_arbitration_20260507")
READER_REPORT_DIR = Path("reports/repair_gated_arbitration_reader_20260507")
COUNTERFACTUAL_ROWS = Path("reports/counterfactual_fill_mechanism_20260507/counterfactual_rows.csv")


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def eval_metric_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples = [row for row in list_items(data.get("examples")) if isinstance(row, Mapping)]
    gold_answers = [list_items(row.get("gold_answers")) for row in examples]
    predictions = [str(row.get("answer") or "") for row in examples]
    _, per_query_em = QAExactMatch(global_config=None).calculate_metric_scores(gold_answers, predictions)
    _, per_query_f1 = QAF1Score(global_config=None).calculate_metric_scores(gold_answers, predictions)
    rows: list[dict[str, Any]] = []
    for example, em_row, f1_row in zip(examples, per_query_em, per_query_f1):
        answer = str(example.get("answer") or "")
        rows.append({
            "question": str(example.get("question") or ""),
            "answer": answer,
            "em": safe_float(em_row.get("ExactMatch")),
            "f1": safe_float(f1_row.get("F1")),
            "empty_answer": int(not answer.strip()),
        })
    return rows


def support_rows_by_dataset_index() -> dict[tuple[str, int], Mapping[str, Any]]:
    rows = read_csv_rows(SUPPORT_ROWS)
    return {
        (str(row.get("dataset")), safe_int(row.get("query_index"))): row
        for row in rows
    }


def counterfactual_rows_by_dataset_index() -> dict[tuple[str, int], Mapping[str, Any]]:
    if not COUNTERFACTUAL_ROWS.exists():
        return {}
    rows = read_csv_rows(COUNTERFACTUAL_ROWS)
    return {
        (str(row.get("dataset")), safe_int(row.get("query_index"))): row
        for row in rows
    }


def build_reader_rows(manifest: Mapping[str, Any], report_dir: Path) -> list[dict[str, Any]]:
    support_by_key = support_rows_by_dataset_index()
    counterfactual_by_key = counterfactual_rows_by_dataset_index()
    output: list[dict[str, Any]] = []
    for dataset_entry in manifest.get("datasets", []) or []:
        dataset_label = str(dataset_entry.get("label") or "")
        subset_dataset = str(dataset_entry.get("subset_dataset") or "")
        selected_indices = [safe_int(index) for index in dataset_entry.get("selected_indices") or []]
        for variant_entry in dataset_entry.get("variants", []) or []:
            variant = str(variant_entry.get("variant") or "")
            eval_json = report_dir / f"{subset_dataset}_{variant}.eval.json"
            pool_json = Path(str(variant_entry.get("pool_json")))
            eval_rows = eval_metric_rows(read_json(eval_json))
            pool_records = list_items(read_json(pool_json).get("records"))
            if len(eval_rows) != len(selected_indices):
                raise ValueError(f"{eval_json}: eval rows {len(eval_rows)} != selected indices {len(selected_indices)}")
            if len(pool_records) != len(selected_indices):
                raise ValueError(f"{pool_json}: pool records {len(pool_records)} != selected indices {len(selected_indices)}")
            for subset_idx, source_idx in enumerate(selected_indices):
                support_row = support_by_key[(dataset_label, source_idx)]
                counterfactual_row = counterfactual_by_key.get((dataset_label, source_idx), {})
                has_rank_fill5 = bool(counterfactual_row)
                trace = pool_records[subset_idx].get("repair_gated_trace")
                trace = trace if isinstance(trace, Mapping) else {}
                offline_support = trace.get("offline_support")
                offline_support = offline_support if isinstance(offline_support, Mapping) else {}
                eval_row = eval_rows[subset_idx]
                row = {
                    "dataset": dataset_label,
                    "subset_dataset": subset_dataset,
                    "variant": variant,
                    "query_index": source_idx,
                    "question": str(support_row.get("question") or eval_row.get("question") or ""),
                    "repair_em": safe_float(eval_row.get("em")),
                    "repair_f1": safe_float(eval_row.get("f1")),
                    "repair_empty_answer": safe_int(eval_row.get("empty_answer")),
                    "setr_em": safe_float(support_row.get("setr_em")),
                    "setr_f1": safe_float(support_row.get("setr_f1")),
                    "dbec_em": safe_float(support_row.get("dbec_em")),
                    "dbec_f1": safe_float(support_row.get("dbec_f1")),
                    "rank_fill5_available": int(has_rank_fill5),
                    "rank_fill5_em": safe_float(counterfactual_row.get("rank_fill5_em")) if has_rank_fill5 else "",
                    "rank_fill5_f1": safe_float(counterfactual_row.get("rank_fill5_f1")) if has_rank_fill5 else "",
                    "delta_em_vs_setr": safe_float(eval_row.get("em")) - safe_float(support_row.get("setr_em")),
                    "delta_f1_vs_setr": safe_float(eval_row.get("f1")) - safe_float(support_row.get("setr_f1")),
                    "delta_em_vs_dbec": safe_float(eval_row.get("em")) - safe_float(support_row.get("dbec_em")),
                    "delta_f1_vs_dbec": safe_float(eval_row.get("f1")) - safe_float(support_row.get("dbec_f1")),
                    "delta_em_vs_rank_fill5": (
                        safe_float(eval_row.get("em")) - safe_float(counterfactual_row.get("rank_fill5_em"))
                        if has_rank_fill5
                        else ""
                    ),
                    "delta_f1_vs_rank_fill5": (
                        safe_float(eval_row.get("f1")) - safe_float(counterfactual_row.get("rank_fill5_f1"))
                        if has_rank_fill5
                        else ""
                    ),
                    "accepted_repair_count": len(trace.get("accepted_repair_positions") or []),
                    "accepted_repair_titles_json": json.dumps(trace.get("accepted_repair_titles") or [], ensure_ascii=False),
                    "offline_final_support_recall": safe_float(offline_support.get("final_support_recall")),
                    "offline_final_support_complete": safe_int(offline_support.get("final_support_complete")),
                    "offline_rank_support_recall": safe_float(offline_support.get("rank_support_recall")),
                    "offline_rank_support_complete": safe_int(offline_support.get("rank_support_complete")),
                    "offline_net_gold_delta_vs_rank": safe_int(offline_support.get("net_gold_delta_vs_rank")),
                    "offline_harmful_replacement": safe_int(offline_support.get("harmful_replacement")),
                }
                output.append(row)
    return output


def mean_float(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(mean([safe_float(row.get(field)) for row in rows])) if rows else 0.0


def mean_optional_float(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [
        safe_float(row.get(field))
        for row in rows
        if row.get(field) is not None and str(row.get(field)) != ""
    ]
    return float(mean(values)) if values else ""


def build_method_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in rows}):
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for variant in sorted({str(row.get("variant")) for row in dataset_rows}):
            variant_rows = [row for row in dataset_rows if str(row.get("variant")) == variant]
            output.append({
                "dataset": dataset,
                "variant": variant,
                "n": len(variant_rows),
                "repair_em": mean_float(variant_rows, "repair_em"),
                "repair_f1": mean_float(variant_rows, "repair_f1"),
                "setr_em": mean_float(variant_rows, "setr_em"),
                "setr_f1": mean_float(variant_rows, "setr_f1"),
                "dbec_em": mean_float(variant_rows, "dbec_em"),
                "dbec_f1": mean_float(variant_rows, "dbec_f1"),
                "rank_fill5_available": int(all(safe_int(row.get("rank_fill5_available")) for row in variant_rows)),
                "rank_fill5_em": mean_optional_float(variant_rows, "rank_fill5_em"),
                "rank_fill5_f1": mean_optional_float(variant_rows, "rank_fill5_f1"),
                "delta_em_vs_setr": mean_float(variant_rows, "delta_em_vs_setr"),
                "delta_f1_vs_setr": mean_float(variant_rows, "delta_f1_vs_setr"),
                "delta_em_vs_dbec": mean_float(variant_rows, "delta_em_vs_dbec"),
                "delta_f1_vs_dbec": mean_float(variant_rows, "delta_f1_vs_dbec"),
                "delta_em_vs_rank_fill5": mean_optional_float(variant_rows, "delta_em_vs_rank_fill5"),
                "delta_f1_vs_rank_fill5": mean_optional_float(variant_rows, "delta_f1_vs_rank_fill5"),
                "empty_answer_rate": mean_float(variant_rows, "repair_empty_answer"),
                "accepted_repair_count": mean_float(variant_rows, "accepted_repair_count"),
                "offline_final_support_complete": mean_float(variant_rows, "offline_final_support_complete"),
                "offline_rank_support_complete": mean_float(variant_rows, "offline_rank_support_complete"),
                "offline_net_gold_delta_vs_rank": mean_float(variant_rows, "offline_net_gold_delta_vs_rank"),
                "offline_harmful_replacement_rate": mean_float(variant_rows, "offline_harmful_replacement"),
            })
    return output


def build_paired_ci(rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    comparisons = (
        ("setr", "setr_em", "setr_f1"),
        ("dbec", "dbec_em", "dbec_f1"),
        ("rank_fill5", "rank_fill5_em", "rank_fill5_f1"),
    )
    for dataset in sorted({str(row.get("dataset")) for row in rows}):
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for variant in sorted({str(row.get("variant")) for row in dataset_rows}):
            variant_rows = [row for row in dataset_rows if str(row.get("variant")) == variant]
            for baseline, baseline_em, baseline_f1 in comparisons:
                if baseline == "rank_fill5" and not all(
                    safe_int(row.get("rank_fill5_available")) for row in variant_rows
                ):
                    continue
                for metric, repair_field, baseline_field in (
                    ("em", "repair_em", baseline_em),
                    ("f1", "repair_f1", baseline_f1),
                ):
                    stats = paired_bootstrap_delta(
                        [safe_float(row.get(repair_field)) for row in variant_rows],
                        [safe_float(row.get(baseline_field)) for row in variant_rows],
                        seed=BOOTSTRAP_SEED + 1000 + len(output) * 31,
                        samples=bootstrap_samples,
                    )
                    output.append({
                        "dataset": dataset,
                        "variant": variant,
                        "baseline": baseline,
                        "metric": metric,
                        **stats,
                    })
    return output


def build_markdown(summary_rows: Sequence[Mapping[str, Any]], paired_ci: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Repair-Gated DBEC Reader Evaluation",
        "",
        "Reader-only evaluation on repair-gated external pools. Baselines are the frozen SetR-faithful and DBEC-selective per-query results for the same slices.",
        "",
        "## Summary",
        "",
    ]
    for dataset in sorted({str(row.get("dataset")) for row in summary_rows}):
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| variant | F1 | dF1 vs SetR | dF1 vs DBEC | dF1 vs rank_fill5 | EM | accepted repairs | offline dGold |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        dataset_rows = [row for row in summary_rows if str(row.get("dataset")) == dataset]
        dataset_rows = sorted(dataset_rows, key=lambda row: safe_float(row.get("repair_f1")), reverse=True)
        for row in dataset_rows:
            rank_delta = row.get("delta_f1_vs_rank_fill5")
            rank_delta_text = (
                f"{safe_float(rank_delta):+.4f}"
                if rank_delta is not None and str(rank_delta) != ""
                else "n/a"
            )
            lines.append(
                "| {variant} | {f1:.4f} | {df1s:+.4f} | {df1d:+.4f} | {df1r} | {em:.4f} | {repairs:.3f} | {dgold:+.3f} |".format(
                    variant=row["variant"],
                    f1=safe_float(row.get("repair_f1")),
                    df1s=safe_float(row.get("delta_f1_vs_setr")),
                    df1d=safe_float(row.get("delta_f1_vs_dbec")),
                    df1r=rank_delta_text,
                    em=safe_float(row.get("repair_em")),
                    repairs=safe_float(row.get("accepted_repair_count")),
                    dgold=safe_float(row.get("offline_net_gold_delta_vs_rank")),
                )
            )
        lines.append("")

    lines.extend(["## Paired CI", ""])
    for row in paired_ci:
        if str(row.get("metric")) != "f1" or str(row.get("baseline")) != "setr":
            continue
        lines.append(
            "- {dataset} `{variant}` vs SetR F1: delta={delta:+.4f}, 95% CI=[{lo:+.4f}, {hi:+.4f}], p(delta>0)={p:.3f}".format(
                dataset=row.get("dataset"),
                variant=row.get("variant"),
                delta=safe_float(row.get("delta_mean")),
                lo=safe_float(row.get("ci_low")),
                hi=safe_float(row.get("ci_high")),
                p=safe_float(row.get("p_delta_gt_0")),
            )
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Per-query rows: `{READER_REPORT_DIR / 'reader_rows.csv'}`",
        f"- Method summary: `{READER_REPORT_DIR / 'method_summary.csv'}`",
        f"- Paired CI: `{READER_REPORT_DIR / 'paired_ci.csv'}`",
        f"- Full JSON: `{READER_REPORT_DIR / 'summary.json'}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    global RUN_DIR, READER_REPORT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--reader_report_dir", type=Path, default=READER_REPORT_DIR)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    args = parser.parse_args()

    RUN_DIR = Path(args.run_dir)
    READER_REPORT_DIR = Path(args.reader_report_dir)
    manifest = read_json(RUN_DIR / "manifest.json")
    rows = build_reader_rows(manifest, READER_REPORT_DIR)
    summary_rows = build_method_summary(rows)
    paired_ci = build_paired_ci(rows, bootstrap_samples=int(args.bootstrap_samples))
    payload = {
        "metadata": {
            "run_dir": str(RUN_DIR),
            "reader_report_dir": str(READER_REPORT_DIR),
            "support_rows": str(SUPPORT_ROWS),
            "counterfactual_rows": str(COUNTERFACTUAL_ROWS) if COUNTERFACTUAL_ROWS.exists() else "",
        },
        "reader_rows": rows,
        "method_summary": summary_rows,
        "paired_ci": paired_ci,
    }
    write_csv(rows, READER_REPORT_DIR / "reader_rows.csv")
    write_csv(summary_rows, READER_REPORT_DIR / "method_summary.csv")
    write_csv(paired_ci, READER_REPORT_DIR / "paired_ci.csv")
    write_json(payload, READER_REPORT_DIR / "summary.json")
    (READER_REPORT_DIR / "summary.md").write_text(
        build_markdown(summary_rows, paired_ci) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    print(f"Wrote reader summary to {READER_REPORT_DIR}")


if __name__ == "__main__":
    main()
