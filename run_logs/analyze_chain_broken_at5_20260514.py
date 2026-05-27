#!/usr/bin/env python3
"""Offline chain-broken@5 summary for existing retrieval reports.

This script intentionally uses only gold support titles and reader-facing top-5
titles. It does not call any LLM and does not inspect binding internals.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "chain_broken_at5_20260514"
DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")


def norm_title(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def title_from_doc(doc: Any) -> str:
    text = str(doc or "")
    return text.split("\n", 1)[0].strip()


def list_titles(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def final_titles_from_trace(trace: Mapping[str, Any]) -> list[str]:
    pcec = trace.get("pcec_readout")
    if isinstance(pcec, Mapping):
        titles = list_titles(pcec.get("final_top5_titles"))
        if titles:
            return titles
    for key in (
        "selector_top_titles",
        "retrieved_titles_top5",
        "method_top_titles",
        "final_top_titles",
        "titles_top5",
    ):
        titles = list_titles(trace.get(key))
        if titles:
            return titles
    selector_trace = trace.get("selector_trace")
    if isinstance(selector_trace, Mapping):
        titles = list_titles(selector_trace.get("final_front_titles"))
        if titles:
            return titles
    return []


def rows_from_retrieval_report(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)

    # PCEC-style reports.
    traces = payload.get("setwise_selector_query_traces")
    if isinstance(traces, list) and traces:
        rows = []
        for idx, trace in enumerate(traces):
            if not isinstance(trace, Mapping):
                continue
            rows.append(
                {
                    "query_index": int(trace.get("query_index", idx) or idx),
                    "question": str(trace.get("question") or ""),
                    "gold_titles": list_titles(trace.get("gold_titles")),
                    "top5_titles": final_titles_from_trace(trace)[:5],
                }
            )
        return rows

    raw_rows = payload.get("rows")
    if isinstance(raw_rows, list) and raw_rows:
        rows = []
        for idx, row in enumerate(raw_rows):
            if not isinstance(row, Mapping):
                continue
            rows.append(
                {
                    "query_index": int(row.get("query_index", idx) or idx),
                    "question": str(row.get("question") or ""),
                    "gold_titles": list_titles(row.get("gold_titles")),
                    "top5_titles": final_titles_from_trace(row)[:5],
                }
            )
        return rows

    # Pool/NeocorRAG-style records. Some reports store reader-facing documents
    # as docs, while exported graph baselines store the same top-K under
    # pool_titles/pool_docs.
    records = payload.get("records")
    if isinstance(records, list) and records:
        rows = []
        for idx, row in enumerate(records):
            if not isinstance(row, Mapping):
                continue
            top_titles = list_titles(row.get("pool_titles"))
            if not top_titles:
                top_titles = [title_from_doc(doc) for doc in (row.get("pool_docs", []) or [])]
            if not top_titles:
                top_titles = [title_from_doc(doc) for doc in (row.get("docs", []) or [])]
            gold_titles = list_titles(row.get("gold_titles"))
            if not gold_titles:
                gold_titles = [title_from_doc(doc) for doc in row.get("gold_docs", []) or []]
            rows.append(
                {
                    "query_index": int(row.get("query_idx", idx) or idx),
                    "question": str(row.get("question") or ""),
                    "gold_titles": gold_titles,
                    "top5_titles": top_titles[:5],
                }
            )
        return rows

    return []


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, k: int = 5) -> dict[str, Any]:
    total = 0
    broken = 0
    no_gold = 0
    any_gold = 0
    recall_sum = 0.0
    gold_counts = []
    top_counts = []
    per_query = []

    for row in rows:
        gold = [norm_title(title) for title in row.get("gold_titles", []) if norm_title(title)]
        top = [norm_title(title) for title in list(row.get("top5_titles", []) or [])[:k] if norm_title(title)]
        if not gold:
            no_gold += 1
            continue
        gold_set = set(gold)
        top_set = set(top)
        hits = gold_set & top_set
        is_broken = not gold_set.issubset(top_set)
        total += 1
        broken += int(is_broken)
        any_gold += int(bool(hits))
        recall_sum += len(hits) / max(len(gold_set), 1)
        gold_counts.append(len(gold_set))
        top_counts.append(len(top))
        per_query.append(
            {
                "query_index": int(row.get("query_index", len(per_query)) or len(per_query)),
                "chain_broken": int(is_broken),
                "title_recall_at5": len(hits) / max(len(gold_set), 1),
                "gold_count": len(gold_set),
                "top5_count": len(top),
            }
        )

    return {
        "count": total,
        "skipped_no_gold": no_gold,
        "chain_broken_at5": broken / total if total else None,
        "chain_intact_at5": 1.0 - broken / total if total else None,
        "all_gold_at5": 1.0 - broken / total if total else None,
        "any_gold_at5": any_gold / total if total else None,
        "title_recall_at5": recall_sum / total if total else None,
        "mean_gold_count": sum(gold_counts) / len(gold_counts) if gold_counts else None,
        "mean_top5_count": sum(top_counts) / len(top_counts) if top_counts else None,
        "per_query": per_query,
    }


def qa_from_reader_report(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if not path.exists():
        return {}, {}
    payload = read_json(path)
    if "datasets" not in payload:
        return {}, {}
    dataset = (payload.get("datasets") or [{}])[0]
    methods = dataset.get("methods") or {}
    if not methods:
        return {}, {}
    method = next(iter(methods.values()))
    metrics = dict(method.get("metrics") or {})
    per_query = {}
    for idx, row in enumerate(method.get("per_query") or []):
        if isinstance(row, Mapping):
            per_query[int(row.get("query_index", idx) or idx)] = {
                "ExactMatch": row.get("ExactMatch"),
                "F1": row.get("F1"),
            }
    return metrics, per_query


def qa_from_neocor_report(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    qa = payload.get("overall_qa_results")
    return dict(qa or {}) if isinstance(qa, Mapping) else {}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for _, y in pairs))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def method_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for dataset in DATASETS:
        specs.append(
            {
                "method": "ETv3+PCEC_all32",
                "dataset": dataset,
                "protocol": "Qwen32B graph/OpenIE + Qwen32B PCEC requirement/binding + GPT-4o-mini reader",
                "retrieval": ROOT
                / "run_logs"
                / "all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
                / "pcec"
                / "etv3"
                / "evals"
                / f"{dataset}_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json",
                "reader": ROOT
                / "run_logs"
                / "all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
                / "reader_qa"
                / "etv3"
                / "reports"
                / f"{dataset}_etv3_all32_pcec_gpt4omini_none_reader_qa_full1000.json",
            }
        )
        specs.append(
            {
                "method": "ETV4+PCEC_all32",
                "dataset": dataset,
                "protocol": "Qwen32B ETV4 graph/OpenIE + Qwen32B PCEC requirement/binding + GPT-4o-mini reader",
                "retrieval": ROOT
                / "run_logs"
                / "all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
                / "pcec"
                / "etv4"
                / "evals"
                / f"{dataset}_pcec_native_pool_prefix4_residual1_pool100_limit1000.json",
                "reader": ROOT
                / "run_logs"
                / "all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514"
                / "reader_qa"
                / "etv4"
                / "reports"
                / f"{dataset}_etv4_all32_pcec_gpt4omini_none_reader_qa_full1000.json",
            }
        )
        specs.append(
            {
                "method": "ETv3+PCEC_old8b_utility",
                "dataset": dataset,
                "protocol": "qwen32b_graph + frozen/old PCEC utility + GPT-4o-mini reader",
                "retrieval": ROOT
                / "run_logs"
                / "pcec_qwen32b_graph_e2e_full1000_20260511"
                / "retrieval_reports_reader_docids"
                / f"{dataset}_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json",
                "reader": ROOT
                / "run_logs"
                / "evidenceflow_pcec_qwen32b_gpt4omini_none_full1000_20260514"
                / "reports"
                / f"{dataset}_evidenceflow_qwen32b_gpt4omini_none_reader_qa_full1000.json",
            }
        )
        specs.append(
            {
                "method": "ETV4+PCEC_old8b_utility",
                "dataset": dataset,
                "protocol": "qwen32b_etv4_graph + frozen/old PCEC utility + GPT-4o-mini reader",
                "retrieval": ROOT
                / "run_logs"
                / "etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514"
                / "evals"
                / f"{dataset}_pcec_native_pool_prefix4_residual1_pool100_limit1000.json",
                "reader": ROOT
                / "run_logs"
                / "etv4_pcec_qwen32b_gpt4omini_none_full1000_20260514"
                / "reader_qa"
                / "reports"
                / f"{dataset}_etv4_pcec_gpt4omini_none_reader_qa_full1000.json",
            }
        )
        no_fact = (
            ROOT
            / "run_logs"
            / "pcec_no_fact_witness_qwen32b_gpt4omini_none_full1000_20260514"
            / "evals"
            / f"{dataset}_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json"
        )
        specs.append(
            {
                "method": "ETv3+PCEC_no_fact_witness_old8b_utility",
                "dataset": dataset,
                "protocol": "diagnostic old 8B PCEC utility + GPT-4o-mini reader",
                "retrieval": no_fact,
                "reader": ROOT
                / "run_logs"
                / "pcec_no_fact_witness_qwen32b_gpt4omini_none_full1000_20260514"
                / "reader_qa"
                / "reports"
                / f"{dataset}_pcec_no_fact_witness_gpt4omini_none_reader_qa_full1000.json",
            }
        )
        specs.append(
            {
                "method": "PropRAG_qwen32b_top200",
                "dataset": dataset,
                "protocol": "PropRAG qwen32b_nothink graph + NV-Embed-v2 + GPT-4o-mini reader top5",
                "retrieval": ROOT
                / "run_logs"
                / "baseline_qwen32b_graph_top200_gpt4omini_full_20260512"
                / "pools"
                / "proprag"
                / f"{dataset}_proprag_qwen32b_nothink_pool200.json",
                "reader": ROOT
                / "run_logs"
                / "baseline_qwen32b_graph_top200_gpt4omini_full_20260512"
                / "reader_qa"
                / "proprag"
                / f"{dataset}_proprag_qwen32b_nothink_top200_gpt4omini_reader_top5.json",
            }
        )
        specs.append(
            {
                "method": "HippoRAG_valid_qwen32b_top200",
                "dataset": dataset,
                "protocol": "HippoRAG valid qwen32b graph + NV-Embed-v2 + GPT-4o-mini reader top5",
                "retrieval": ROOT
                / "run_logs"
                / "hipporag_qwen32b_valid_graph_top200_full1000_20260513_r2"
                / "pools"
                / "hipporag"
                / f"{dataset}_hipporag_qwen32b_valid_graph_pool200.json",
                "reader": ROOT
                / "run_logs"
                / "hipporag_qwen32b_valid_graph_top200_full1000_20260513_r2"
                / "reader_qa"
                / "hipporag"
                / f"{dataset}_hipporag_qwen32b_valid_graph_top200_gpt4omini_reader_top5.json",
            }
        )

        for slot in range(1, 5):
            specs.append(
                {
                    "method": f"PCEC_position_slot{slot}_old8b",
                    "dataset": dataset,
                    "protocol": "position-sensitivity diagnostic, 8B reader/utility",
                    "retrieval": ROOT
                    / "reports"
                    / "pcec_m4_position_sensitivity_8b_20260511"
                    / "retrieval_reports"
                    / f"slot{slot}"
                    / f"{dataset}_pcec_m4_admitted_slot{slot}_pool100_limit1000.json",
                    "reader": ROOT
                    / "reports"
                    / "pcec_m4_position_sensitivity_8b_20260511"
                    / "reader_qa"
                    / "reports"
                    / f"{dataset}_pcec_m4_admitted_slot{slot}_reader_qa_8b_full1000.json",
                }
            )

        specs.append(
            {
                "method": "NeocorRAG_k3_old",
                "dataset": dataset,
                "protocol": "completed 2026-05-01 NeocorRAG k=3 report",
                "retrieval": ROOT
                / "run_logs"
                / "neocorrag_aligned_k3_full1000_20260430"
                / f"{dataset}_neocorrag_k3.json",
                "reader": None,
                "qa_in_retrieval": True,
            }
        )

    return specs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for spec in method_specs():
        retrieval_path = Path(spec["retrieval"])
        if not retrieval_path.exists():
            missing.append(
                {
                    "method": str(spec["method"]),
                    "dataset": str(spec["dataset"]),
                    "missing": str(retrieval_path),
                }
            )
            continue

        rows = rows_from_retrieval_report(retrieval_path)
        chain = summarize_rows(rows)

        qa_metrics: dict[str, Any] = {}
        qa_per_query: dict[int, dict[str, Any]] = {}
        if spec.get("qa_in_retrieval"):
            qa_metrics = qa_from_neocor_report(retrieval_path)
        elif spec.get("reader"):
            qa_metrics, qa_per_query = qa_from_reader_report(Path(spec["reader"]))

        broken_by_qid = {row["query_index"]: row for row in chain.pop("per_query")}
        joined_f1_broken: list[float] = []
        joined_f1_intact: list[float] = []
        joined_em_broken: list[float] = []
        joined_em_intact: list[float] = []
        corr_x: list[float] = []
        corr_f1: list[float] = []
        corr_em: list[float] = []
        for qid, chain_row in broken_by_qid.items():
            qa_row = qa_per_query.get(qid)
            if not qa_row:
                continue
            broken = int(chain_row["chain_broken"])
            f1 = maybe_float(qa_row.get("F1"))
            em = maybe_float(qa_row.get("ExactMatch"))
            if f1 is not None:
                (joined_f1_broken if broken else joined_f1_intact).append(f1)
                corr_x.append(float(broken))
                corr_f1.append(f1)
            if em is not None:
                (joined_em_broken if broken else joined_em_intact).append(em)
                corr_em.append(em)
            per_query_rows.append(
                {
                    "method": spec["method"],
                    "dataset": spec["dataset"],
                    "query_index": qid,
                    "chain_broken": broken,
                    "title_recall_at5": chain_row["title_recall_at5"],
                    "ExactMatch": em,
                    "F1": f1,
                }
            )

        summary_rows.append(
            {
                "method": spec["method"],
                "dataset": spec["dataset"],
                "protocol": spec["protocol"],
                "count": chain["count"],
                "chain_broken_at5": chain["chain_broken_at5"],
                "all_gold_at5": chain["all_gold_at5"],
                "any_gold_at5": chain["any_gold_at5"],
                "title_recall_at5": chain["title_recall_at5"],
                "mean_gold_count": chain["mean_gold_count"],
                "ExactMatch": qa_metrics.get("ExactMatch"),
                "F1": qa_metrics.get("F1"),
                "per_query_corr_chain_broken_vs_F1": pearson(corr_x, corr_f1),
                "per_query_corr_chain_broken_vs_EM": pearson(corr_x, corr_em),
                "mean_F1_when_chain_intact": mean(joined_f1_intact),
                "mean_F1_when_chain_broken": mean(joined_f1_broken),
                "mean_EM_when_chain_intact": mean(joined_em_intact),
                "mean_EM_when_chain_broken": mean(joined_em_broken),
                "retrieval_path": str(retrieval_path.relative_to(ROOT)),
                "reader_path": str(Path(spec["reader"]).relative_to(ROOT)) if spec.get("reader") else "",
            }
        )

    summary_rows.sort(key=lambda row: (str(row["dataset"]), str(row["method"])))

    summary_csv = OUT_DIR / "chain_broken_at5_summary.csv"
    if summary_rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    per_query_csv = OUT_DIR / "chain_broken_at5_per_query_joined_reader.csv"
    if per_query_rows:
        with per_query_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_query_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_query_rows)

    across = {
        "method_dataset_rows_with_qa": sum(1 for row in summary_rows if row.get("F1") is not None),
        "corr_chain_broken_at5_vs_F1_across_method_dataset": pearson(
            [maybe_float(row.get("chain_broken_at5")) for row in summary_rows if row.get("F1") is not None],
            [maybe_float(row.get("F1")) for row in summary_rows if row.get("F1") is not None],
        ),
        "corr_chain_broken_at5_vs_EM_across_method_dataset": pearson(
            [maybe_float(row.get("chain_broken_at5")) for row in summary_rows if row.get("ExactMatch") is not None],
            [maybe_float(row.get("ExactMatch")) for row in summary_rows if row.get("ExactMatch") is not None],
        ),
        "corr_all_gold_at5_vs_F1_across_method_dataset": pearson(
            [maybe_float(row.get("all_gold_at5")) for row in summary_rows if row.get("F1") is not None],
            [maybe_float(row.get("F1")) for row in summary_rows if row.get("F1") is not None],
        ),
    }

    payload = {"summary": summary_rows, "missing": missing, "across_method_dataset_correlation": across}
    (OUT_DIR / "chain_broken_at5_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Chain-Broken@5 Offline Summary",
        "",
        "Definition: `chain_broken@5 = mean_q[gold_titles(q) not subset of top5_titles(q)]`.",
        "This is computed from retrieval outputs only; no LLM is called.",
        "",
        "## Across Method-Dataset Correlation",
        "",
        f"- Rows with QA metrics: {across['method_dataset_rows_with_qa']}",
        f"- corr(chain_broken@5, F1): {across['corr_chain_broken_at5_vs_F1_across_method_dataset']}",
        f"- corr(chain_broken@5, EM): {across['corr_chain_broken_at5_vs_EM_across_method_dataset']}",
        f"- corr(all_gold@5, F1): {across['corr_all_gold_at5_vs_F1_across_method_dataset']}",
        "",
        "## Summary",
        "",
        "| Method | Dataset | chain-broken@5 | all-gold@5 | R@5 title | EM | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def fmt(value: Any) -> str:
            value = maybe_float(value)
            return "" if value is None else f"{value:.4f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["dataset"]),
                    fmt(row["chain_broken_at5"]),
                    fmt(row["all_gold_at5"]),
                    fmt(row["title_recall_at5"]),
                    fmt(row["ExactMatch"]),
                    fmt(row["F1"]),
                ]
            )
            + " |"
        )
    if missing:
        lines += ["", "## Missing Inputs", ""]
        for item in missing:
            lines.append(f"- {item['method']} / {item['dataset']}: `{item['missing']}`")
    (OUT_DIR / "chain_broken_at5_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_csv.relative_to(ROOT)}")
    print(f"Wrote {per_query_csv.relative_to(ROOT)}")
    print(f"Wrote {(OUT_DIR / 'chain_broken_at5_summary.md').relative_to(ROOT)}")
    print(json.dumps(across, indent=2))


if __name__ == "__main__":
    main()
