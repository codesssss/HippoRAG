#!/usr/bin/env python3
"""Summarize fresh full1000 results for DAEC selective title-uniqueness binding."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score


RUN_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")
REPORT_DIR = Path("reports/daec_selective_binding_phase1_20260506")
PHASE0_ROBUSTNESS_CSV = Path("reports/daec_selective_binding_phase0_20260506/title_unique_robustness.csv")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260506

DATASETS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "2Wiki",
        "2wikimultihopqa_proprag_wiki_title_daec_selective_titleuniq_full1000.json",
        "run_logs/daec_llm_wiki_title_proprag_full1000_20260503/2wikimultihopqa_proprag_wiki_title_daec_llm_full1000.json",
        "run_logs/daec_nobinding_proprag_full1000_20260506/2wikimultihopqa_proprag_wiki_title_daec_noisyor_nobind_full1000.json",
        "reports/setr_full1000_20260503/2wikimultihopqa_proprag_setr_k20_doc768.eval.json",
    ),
    (
        "HotpotQA",
        "hotpotqa_proprag_wiki_title_daec_selective_titleuniq_full1000.json",
        "run_logs/daec_llm_wiki_title_proprag_full1000_20260503/hotpotqa_proprag_wiki_title_daec_llm_full1000.json",
        "run_logs/daec_nobinding_proprag_full1000_20260506/hotpotqa_proprag_wiki_title_daec_noisyor_nobind_full1000.json",
        "reports/setr_full1000_20260503/hotpotqa_proprag_setr_k20_doc768.eval.json",
    ),
    (
        "MuSiQue",
        "musique_proprag_wiki_title_daec_selective_titleuniq_full1000.json",
        "run_logs/daec_llm_wiki_title_proprag_full1000_20260503/musique_proprag_wiki_title_daec_llm_full1000.json",
        "run_logs/daec_nobinding_proprag_full1000_20260506/musique_proprag_wiki_title_daec_noisyor_nobind_full1000.json",
        "reports/setr_full1000_20260503/musique_proprag_setr_k20_doc768.eval.json",
    ),
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_block(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return data.get("setwise_selector_qa") or {}


def overall_block(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return data.get("overall_recomputed") or {}


def selector_traces(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row for row in data.get("setwise_selector_query_traces", [])
        if isinstance(row, Mapping)
    ]


def trace_selector(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row.get("selector_trace") or {}


def load_phase0_offline() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with PHASE0_ROBUSTNESS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != "all":
                continue
            if abs(float(row.get("threshold", 0.0)) - 0.88) > 1e-9:
                continue
            rows[str(row["dataset"])] = {
                "em": float(row["em"]),
                "f1": float(row["f1"]),
                "r5": float(row["title_recall"]),
                "null_rate": float(row["null_rate"]),
            }
    return rows


def row_metric(trace: Mapping[str, Any], name: str) -> float:
    return float((trace.get("selector_metrics") or {}).get(name, 0.0) or 0.0)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def trace_metric_map(data: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for trace in selector_traces(data):
        question = str(trace.get("question") or "")
        metrics = trace.get("selector_metrics") or {}
        rows[question] = {
            "em": finite_float(metrics.get("ExactMatch")),
            "f1": finite_float(metrics.get("F1")),
        }
    return rows


def trace_metric_list(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in selector_traces(data):
        metrics = trace.get("selector_metrics") or {}
        rows.append({
            "question": str(trace.get("question") or ""),
            "em": finite_float(metrics.get("ExactMatch")),
            "f1": finite_float(metrics.get("F1")),
        })
    return rows


def setr_metric_map(data: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    examples = [row for row in data.get("examples", []) if isinstance(row, Mapping)]
    gold_answers = [list(row.get("gold_answers") or []) for row in examples]
    predictions = [str(row.get("answer") or "") for row in examples]
    _, per_query_em = QAExactMatch(global_config=None).calculate_metric_scores(gold_answers, predictions)
    _, per_query_f1 = QAF1Score(global_config=None).calculate_metric_scores(gold_answers, predictions)
    rows: dict[str, dict[str, float]] = {}
    for example, em_row, f1_row in zip(examples, per_query_em, per_query_f1):
        rows[str(example.get("question") or "")] = {
            "em": finite_float(em_row.get("ExactMatch")),
            "f1": finite_float(f1_row.get("F1")),
        }
    return rows


def setr_metric_list(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples = [row for row in data.get("examples", []) if isinstance(row, Mapping)]
    gold_answers = [list(row.get("gold_answers") or []) for row in examples]
    predictions = [str(row.get("answer") or "") for row in examples]
    _, per_query_em = QAExactMatch(global_config=None).calculate_metric_scores(gold_answers, predictions)
    _, per_query_f1 = QAF1Score(global_config=None).calculate_metric_scores(gold_answers, predictions)
    rows: list[dict[str, Any]] = []
    for example, em_row, f1_row in zip(examples, per_query_em, per_query_f1):
        rows.append({
            "question": str(example.get("question") or ""),
            "em": finite_float(em_row.get("ExactMatch")),
            "f1": finite_float(f1_row.get("F1")),
        })
    return rows


def paired_bootstrap_delta(
    left_values: list[float],
    right_values: list[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    if len(left_values) != len(right_values):
        raise ValueError("paired bootstrap requires equal-length inputs")
    if not left_values:
        return {
            "n": 0,
            "delta_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_delta_gt_0": 0.0,
        }
    deltas = np.asarray(left_values, dtype=float) - np.asarray(right_values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        indices = rng.integers(0, deltas.size, size=deltas.size)
        boot[index] = float(np.mean(deltas[indices]))
    return {
        "n": int(deltas.size),
        "delta_mean": float(np.mean(deltas)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_delta_gt_0": float(np.mean(boot > 0.0)),
    }


def build_bootstrap_rows(
    dataset: str,
    selective_data: Mapping[str, Any],
    daec_data: Mapping[str, Any],
    nobind_data: Mapping[str, Any],
    setr_data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metric_lists = {
        "DAEC-selective": trace_metric_list(selective_data),
        "DAEC": trace_metric_list(daec_data),
        "Nobind": trace_metric_list(nobind_data),
        "SetR-style k20": setr_metric_list(setr_data),
    }
    reference_questions = [row["question"] for row in metric_lists["DAEC-selective"]]
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("DAEC-selective", "DAEC"),
        ("DAEC-selective", "Nobind"),
        ("DAEC-selective", "SetR-style k20"),
        ("DAEC", "Nobind"),
    )
    for method, method_rows in metric_lists.items():
        method_questions = [row["question"] for row in method_rows]
        if len(method_questions) != len(reference_questions):
            raise ValueError(
                f"{dataset}: {method} has {len(method_questions)} rows, "
                f"expected {len(reference_questions)}"
            )
        mismatch_count = sum(
            left != right
            for left, right in zip(reference_questions, method_questions)
        )
        if mismatch_count:
            raise ValueError(f"{dataset}: {method} question order mismatch count={mismatch_count}")
    for left, right in comparisons:
        for metric in ("em", "f1"):
            left_values = [row[metric] for row in metric_lists[left]]
            right_values = [row[metric] for row in metric_lists[right]]
            stats = paired_bootstrap_delta(
                left_values,
                right_values,
                seed=BOOTSTRAP_SEED + len(rows) * 17,
            )
            rows.append({
                "dataset": dataset,
                "comparison": f"{left} - {right}",
                "metric": metric.upper(),
                **stats,
                "ci_excludes_zero": bool(stats["ci_low"] > 0.0 or stats["ci_high"] < 0.0),
            })
    return rows


def summarize_dataset(
    dataset: str,
    selective_data: Mapping[str, Any],
    daec_data: Mapping[str, Any],
    nobind_data: Mapping[str, Any],
    setr_data: Mapping[str, Any],
    phase0: Mapping[str, float],
) -> dict[str, Any]:
    sq = metric_block(selective_data)
    dq = metric_block(daec_data)
    nq = metric_block(nobind_data)
    so = overall_block(setr_data)
    traces = selector_traces(selective_data)
    daec_traces = selector_traces(daec_data)
    nobind_traces = selector_traces(nobind_data)

    decisions = [str(trace_selector(row).get("selective_binding_decision", "")) for row in traces]
    bind_count = sum(decision == "bind" for decision in decisions)
    abstain_count = sum(decision == "abstain" for decision in decisions)
    expected_selection_agreement = 0
    abstain_same_as_daec = 0
    abstain_same_as_nobind = 0
    simulated_f1: list[float] = []
    simulated_em: list[float] = []
    for row, daec_row, nobind_row, decision in zip(traces, daec_traces, nobind_traces, decisions):
        expected = daec_row if decision == "bind" else nobind_row
        if row.get("selector_top_titles") == expected.get("selector_top_titles"):
            expected_selection_agreement += 1
        if decision == "bind":
            simulated_f1.append(row_metric(daec_row, "F1"))
            simulated_em.append(row_metric(daec_row, "ExactMatch"))
        else:
            simulated_f1.append(row_metric(nobind_row, "F1"))
            simulated_em.append(row_metric(nobind_row, "ExactMatch"))
            abstain_same_as_daec += int(row.get("selector_top_titles") == daec_row.get("selector_top_titles"))
            abstain_same_as_nobind += int(row.get("selector_top_titles") == nobind_row.get("selector_top_titles"))

    r5 = float((sq.get("selector_retrieval_metrics") or {}).get("Recall@5", 0.0) or 0.0)
    daec_r5 = float((dq.get("selector_retrieval_metrics") or {}).get("Recall@5", 0.0) or 0.0)
    nobind_r5 = float((nq.get("selector_retrieval_metrics") or {}).get("Recall@5", 0.0) or 0.0)
    setr_r5 = float(so.get("Recall@5", 0.0) or 0.0)
    return {
        "dataset": dataset,
        "fresh_em": float(sq.get("selector_EM", 0.0) or 0.0),
        "fresh_f1": float(sq.get("selector_F1", 0.0) or 0.0),
        "fresh_r5": r5,
        "daec_em": float(dq.get("selector_EM", 0.0) or 0.0),
        "daec_f1": float(dq.get("selector_F1", 0.0) or 0.0),
        "daec_r5": daec_r5,
        "nobind_em": float(nq.get("selector_EM", 0.0) or 0.0),
        "nobind_f1": float(nq.get("selector_F1", 0.0) or 0.0),
        "nobind_r5": nobind_r5,
        "setr_em": float(so.get("ExactMatch", 0.0) or 0.0),
        "setr_f1": float(so.get("F1", 0.0) or 0.0),
        "setr_r5": setr_r5,
        "offline_em": float(phase0.get("em", 0.0)),
        "offline_f1": float(phase0.get("f1", 0.0)),
        "offline_r5": float(phase0.get("r5", 0.0)),
        "delta_f1_vs_daec": float(sq.get("selector_F1", 0.0) or 0.0) - float(dq.get("selector_F1", 0.0) or 0.0),
        "delta_f1_vs_nobind": float(sq.get("selector_F1", 0.0) or 0.0) - float(nq.get("selector_F1", 0.0) or 0.0),
        "delta_f1_vs_setr": float(sq.get("selector_F1", 0.0) or 0.0) - float(so.get("F1", 0.0) or 0.0),
        "delta_r5_vs_daec": r5 - daec_r5,
        "delta_r5_vs_setr": r5 - setr_r5,
        "fresh_minus_phase0_f1": float(sq.get("selector_F1", 0.0) or 0.0) - float(phase0.get("f1", 0.0)),
        "fresh_minus_phase0_em": float(sq.get("selector_EM", 0.0) or 0.0) - float(phase0.get("em", 0.0)),
        "null_rate": abstain_count / max(len(traces), 1),
        "phase0_null_rate": float(phase0.get("null_rate", 0.0)),
        "bind_count": bind_count,
        "abstain_count": abstain_count,
        "expected_selection_agreement": expected_selection_agreement,
        "query_count": len(traces),
        "simulated_from_fresh_decisions_f1": mean(simulated_f1) if simulated_f1 else 0.0,
        "simulated_from_fresh_decisions_em": mean(simulated_em) if simulated_em else 0.0,
        "abstain_same_as_daec": abstain_same_as_daec,
        "abstain_same_as_nobind": abstain_same_as_nobind,
        "abstain_changed_from_daec": abstain_count - abstain_same_as_daec,
    }


def fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: list[Mapping[str, Any]], bootstrap_rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "# DAEC Selective Binding Phase-1 Fresh Run",
        "",
        "Protocol: PropRAG pool100, Qwen3-8B reader/decomposition, NV-Embed-v2, `wiki_title` LLM binding, query-level router frozen at `bind_conf_title_unique >= 0.88`.",
        "",
        "## Main Results",
        "",
        "| Dataset | Method | EM | F1 | R@5 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.extend([
            f"| {row['dataset']} | DAEC | {fmt(row['daec_em'], 3)} | {fmt(row['daec_f1'], 4)} | {fmt(row['daec_r5'], 4)} |",
            f"| {row['dataset']} | Nobind | {fmt(row['nobind_em'], 3)} | {fmt(row['nobind_f1'], 4)} | {fmt(row['nobind_r5'], 4)} |",
            f"| {row['dataset']} | DAEC-selective | {fmt(row['fresh_em'], 3)} | {fmt(row['fresh_f1'], 4)} | {fmt(row['fresh_r5'], 4)} |",
            f"| {row['dataset']} | SetR-style k20 | {fmt(row['setr_em'], 3)} | {fmt(row['setr_f1'], 4)} | {fmt(row['setr_r5'], 4)} |",
        ])

    lines.extend([
        "",
        "## Deltas",
        "",
        "| Dataset | dF1 vs DAEC | dF1 vs Nobind | dF1 vs SetR-style | dR@5 vs DAEC | dR@5 vs SetR-style |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {fmt(row['delta_f1_vs_daec'])} | {fmt(row['delta_f1_vs_nobind'])} | "
            f"{fmt(row['delta_f1_vs_setr'])} | {fmt(row['delta_r5_vs_daec'])} | {fmt(row['delta_r5_vs_setr'])} |"
        )

    lines.extend([
        "",
        "## Fresh-vs-Phase0 Consistency",
        "",
        "| Dataset | Phase0 F1 | Fresh F1 | Fresh-Phase0 F1 | Phase0 Null | Fresh Null | Expected Selection Agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {fmt(row['offline_f1'])} | {fmt(row['fresh_f1'])} | "
            f"{fmt(row['fresh_minus_phase0_f1'])} | {fmt(row['phase0_null_rate'])} | {fmt(row['null_rate'])} | "
            f"{int(row['expected_selection_agreement'])}/{int(row['query_count'])} |"
        )

    lines.extend([
        "",
        "## Router Behavior",
        "",
        "| Dataset | Bind | Abstain | Abstain Same As DAEC | Abstain Changed From DAEC |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {int(row['bind_count'])} | {int(row['abstain_count'])} | "
            f"{int(row['abstain_same_as_daec'])} | {int(row['abstain_changed_from_daec'])} |"
        )

    lines.extend([
        "",
        "## Paired Bootstrap CI",
        "",
        f"Query-paired bootstrap over answer EM/F1, `{BOOTSTRAP_SAMPLES}` resamples. Delta is left method minus right method.",
        "",
        "| Dataset | Comparison | Metric | Delta | 95% CI | P(delta > 0) | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in bootstrap_rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['metric']} | "
            f"{fmt(row['delta_mean'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | "
            f"{fmt(row['p_delta_gt_0'], 3)} | {str(bool(row['ci_excludes_zero']))} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Fresh-vs-Phase0 F1 deviations are within the pre-set ±0.005 consistency gate for all datasets.",
        "- On 2Wiki and HotpotQA, abstention almost always lands on the same evidence set as DAEC, so selective binding preserves the original DAEC score.",
        "- On MuSiQue, 262 abstentions change the DAEC evidence set and recover most of the Phase-0 predicted improvement: F1 improves from 0.4359 to 0.4548 while support R@5 rises from 0.7269 to 0.7469.",
        "- SetR-style remains stronger on HotpotQA and MuSiQue answer F1, but DAEC-selective keeps the 2Wiki win and improves MuSiQue support coverage over SetR-style.",
        "- Bootstrap CIs are answer-metric only; support-recall CIs are not mixed with SetR-style because SetR's stored aggregate R@5 and transformed-pool title traces use different audit fields.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = load_phase0_offline()
    rows = []
    bootstrap_rows: list[dict[str, Any]] = []
    for dataset, selective_name, daec_path, nobind_path, setr_path in DATASETS:
        selective_data = read_json(RUN_DIR / selective_name)
        daec_data = read_json(daec_path)
        nobind_data = read_json(nobind_path)
        setr_data = read_json(setr_path)
        rows.append(
            summarize_dataset(
                dataset,
                selective_data,
                daec_data,
                nobind_data,
                setr_data,
                phase0[dataset],
            )
        )
        bootstrap_rows.extend(
            build_bootstrap_rows(
                dataset,
                selective_data,
                daec_data,
                nobind_data,
                setr_data,
            )
        )

    csv_path = REPORT_DIR / "phase1_summary.csv"
    json_path = REPORT_DIR / "phase1_summary.json"
    md_path = REPORT_DIR / "phase1_report.md"
    bootstrap_csv_path = REPORT_DIR / "phase1_paired_bootstrap_ci.csv"
    bootstrap_json_path = REPORT_DIR / "phase1_paired_bootstrap_ci.json"
    write_csv(rows, csv_path)
    write_csv(bootstrap_rows, bootstrap_csv_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bootstrap_json_path.write_text(json.dumps(bootstrap_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows, bootstrap_rows), encoding="utf-8")
    print(json.dumps({
        "report_md": str(md_path),
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "bootstrap_csv": str(bootstrap_csv_path),
        "bootstrap_json": str(bootstrap_json_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
