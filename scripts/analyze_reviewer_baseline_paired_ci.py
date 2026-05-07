#!/usr/bin/env python3
"""Paired bootstrap CIs for PropRAG full1000 reviewer baselines."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score


REPORT_DIR = Path("reports/reviewer_baseline_ci_20260506")
DAEC_DIR = Path("run_logs/daec_llm_wiki_title_proprag_full1000_20260503")
DAEC_SELECTIVE_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")
SETR_REPORT_DIR = Path("reports/setr_full1000_20260503")
IRCOT_DIR = Path("run_logs/ircot_style_full1000_20260506")
LLM_DIRECT_DIR = Path("run_logs/llm_direct_select_proprag_full1000_20260506")

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260506

DATASETS: tuple[tuple[str, str], ...] = (
    ("2Wiki", "2wikimultihopqa"),
    ("HotpotQA", "hotpotqa"),
    ("MuSiQue", "musique"),
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_multiset_recall(gold_titles: list[Any], selected_titles: list[Any]) -> float:
    gold = [normalize_title(title) for title in gold_titles if normalize_title(title)]
    selected = [normalize_title(title) for title in selected_titles if normalize_title(title)]
    if not gold:
        return 0.0
    used = [False] * len(selected)
    hits = 0
    for gold_title in gold:
        for index, selected_title in enumerate(selected):
            if used[index]:
                continue
            if selected_title == gold_title:
                used[index] = True
                hits += 1
                break
    return hits / len(gold)


def doc_title(doc: Any) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def daec_path(slug: str) -> Path:
    return DAEC_DIR / f"{slug}_proprag_wiki_title_daec_llm_full1000.json"


def selective_path(slug: str) -> Path:
    return DAEC_SELECTIVE_DIR / f"{slug}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def trace_metric_rows(data: Mapping[str, Any], *, field: str) -> list[dict[str, Any]]:
    rows = []
    for trace in data.get("setwise_selector_query_traces") or []:
        metrics = trace.get(field) or {}
        title_field = "baseline_top_titles" if field == "baseline_metrics" else "selector_top_titles"
        rows.append({
            "question": str(trace.get("question") or ""),
            "EM": safe_float(metrics.get("ExactMatch")),
            "F1": safe_float(metrics.get("F1")),
            "gold_titles": list(trace.get("gold_titles") or []),
            "top_titles": list(trace.get(title_field) or []),
        })
    return rows


def examples_metric_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples = [row for row in data.get("examples", []) if isinstance(row, Mapping)]
    gold_answers = [list(row.get("gold_answers") or []) for row in examples]
    predictions = [str(row.get("answer") or "") for row in examples]
    _, per_query_em = QAExactMatch(global_config=None).calculate_metric_scores(gold_answers, predictions)
    _, per_query_f1 = QAF1Score(global_config=None).calculate_metric_scores(gold_answers, predictions)
    rows = []
    for example, em_row, f1_row in zip(examples, per_query_em, per_query_f1):
        rows.append({
            "question": str(example.get("question") or ""),
            "EM": safe_float(em_row.get("ExactMatch")),
            "F1": safe_float(f1_row.get("F1")),
            "gold_titles": list(example.get("gold_titles") or []),
            "top_titles": list((example.get("retrieval_trace") or {}).get("external_pool_titles") or [])[:5],
        })
    return rows


def ircot_metric_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for trace in data.get("traces") or []:
        rows.append({
            "question": str(trace.get("question") or ""),
            "EM": safe_float(trace.get("em")),
            "F1": safe_float(trace.get("f1")),
            "gold_titles": [],
            "top_titles": [doc_title(doc) for doc in list(trace.get("docs") or [])[:5]],
        })
    return rows


def metric_mean(rows: list[Mapping[str, Any]], metric: str) -> float:
    return float(np.mean([safe_float(row.get(metric)) for row in rows])) if rows else 0.0


def paired_bootstrap(
    left_values: list[float],
    right_values: list[float],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if len(left_values) != len(right_values):
        raise ValueError("paired bootstrap requires equal-length inputs")
    deltas = np.asarray(left_values, dtype=float) - np.asarray(right_values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = deltas[rng.integers(0, deltas.size, size=deltas.size)]
        boot[index] = float(np.mean(sampled))
    return {
        "n": int(deltas.size),
        "delta_mean": float(np.mean(deltas)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_delta_gt_0": float(np.mean(boot > 0.0)),
        "ci_excludes_zero": bool(np.percentile(boot, 2.5) > 0.0 or np.percentile(boot, 97.5) < 0.0),
    }


def occurrence_keys(rows: list[Mapping[str, Any]]) -> list[tuple[str, int]]:
    counts: defaultdict[str, int] = defaultdict(int)
    keys: list[tuple[str, int]] = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        occurrence = counts[question]
        counts[question] += 1
        keys.append((question, occurrence))
    return keys


def align_methods(
    dataset: str,
    methods: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    reference_keys = occurrence_keys(methods["DAEC"])
    aligned: dict[str, list[dict[str, Any]]] = {}
    for name, rows in methods.items():
        keys = occurrence_keys(rows)
        if len(keys) != len(reference_keys):
            raise ValueError(f"{dataset}: {name} length={len(keys)}, expected={len(reference_keys)}")
        keyed = {key: row for key, row in zip(keys, rows)}
        missing = [key for key in reference_keys if key not in keyed]
        if missing:
            raise ValueError(f"{dataset}: {name} missing {len(missing)} occurrence-aligned questions")
        aligned[name] = [keyed[key] for key in reference_keys]
    reference_gold = [list(row.get("gold_titles") or []) for row in aligned["DAEC"]]
    for name, rows in aligned.items():
        for row, gold_titles in zip(rows, reference_gold):
            if not row.get("gold_titles"):
                row["gold_titles"] = list(gold_titles)
            row["R5_TITLE"] = title_multiset_recall(list(gold_titles), list(row.get("top_titles") or []))
    return aligned


def dataset_methods(slug: str) -> dict[str, list[dict[str, Any]]]:
    daec = read_json(daec_path(slug))
    selective = read_json(selective_path(slug))
    return {
        "Top5": trace_metric_rows(daec, field="baseline_metrics"),
        "DAEC": trace_metric_rows(daec, field="selector_metrics"),
        "DAEC-selective": trace_metric_rows(selective, field="selector_metrics"),
        "SetR-style k20": examples_metric_rows(read_json(SETR_REPORT_DIR / f"{slug}_proprag_setr_k20_doc768.eval.json")),
        "IRCoT-style local": ircot_metric_rows(read_json(IRCOT_DIR / f"{slug}_ircot_style.json")),
        "LLM-direct title": examples_metric_rows(read_json(LLM_DIRECT_DIR / f"{slug}_title_llm_direct.json")),
        "LLM-direct snippet128": examples_metric_rows(read_json(LLM_DIRECT_DIR / f"{slug}_snippet128_llm_direct.json")),
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("DAEC", "Top5"),
        ("DAEC-selective", "Top5"),
        ("DAEC-selective", "DAEC"),
        ("DAEC", "SetR-style k20"),
        ("DAEC-selective", "SetR-style k20"),
        ("DAEC", "IRCoT-style local"),
        ("DAEC-selective", "IRCoT-style local"),
        ("DAEC", "LLM-direct title"),
        ("DAEC", "LLM-direct snippet128"),
        ("DAEC-selective", "LLM-direct title"),
        ("DAEC-selective", "LLM-direct snippet128"),
    )
    for dataset, slug in DATASETS:
        methods = align_methods(dataset, dataset_methods(slug))
        for left, right in comparisons:
            for metric in ("EM", "F1", "R5_TITLE"):
                left_values = [safe_float(row.get(metric)) for row in methods[left]]
                right_values = [safe_float(row.get(metric)) for row in methods[right]]
                stats = paired_bootstrap(
                    left_values,
                    right_values,
                    seed=BOOTSTRAP_SEED + len(rows) * 31,
                )
                rows.append({
                    "dataset": dataset,
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "left_mean": metric_mean(methods[left], metric),
                    "right_mean": metric_mean(methods[right], metric),
                    **stats,
                })
    return rows


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):.{digits}f}"


def build_markdown(rows: list[Mapping[str, Any]]) -> str:
    main_rows = [
        row for row in rows
        if row["metric"] == "F1" and (
            row["comparison"].startswith("DAEC-selective -")
            or row["comparison"] in {"DAEC - Top5", "DAEC - SetR-style k20"}
        )
    ]
    lines = [
        "# Reviewer Baseline Paired CI - 2026-05-06",
        "",
        f"Query-paired percentile bootstrap over answer EM/F1 and unified title-multiset support R@5, `{BOOTSTRAP_SAMPLES}` resamples. Delta is left method minus right method.",
        "",
        "Support R@5 here is recomputed uniformly from each method's final top-5 titles against DAEC's reference gold_titles using a title-multiset match. These numbers may differ from pipeline-stored aggregate R@5 fields.",
        "",
        "## Main F1 Rows",
        "",
        "| Dataset | Comparison | Left F1 | Right F1 | dF1 | 95% CI | P(delta > 0) | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {fmt(row['left_mean'])} | "
            f"{fmt(row['right_mean'])} | {fmt(row['delta_mean'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | "
            f"{fmt(row['p_delta_gt_0'], 3)} | {str(bool(row['ci_excludes_zero']))} |"
        )

    lines.extend([
        "",
        "## Main Unified Support R@5 Rows",
        "",
        "| Dataset | Comparison | Left R@5 | Right R@5 | dR@5 | 95% CI | P(delta > 0) | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    support_rows = [
        row for row in rows
        if row["metric"] == "R5_TITLE" and (
            row["comparison"].startswith("DAEC-selective -")
            or row["comparison"] in {"DAEC - Top5", "DAEC - SetR-style k20"}
        )
    ]
    for row in support_rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {fmt(row['left_mean'])} | "
            f"{fmt(row['right_mean'])} | {fmt(row['delta_mean'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | "
            f"{fmt(row['p_delta_gt_0'], 3)} | {str(bool(row['ci_excludes_zero']))} |"
        )

    lines.extend([
        "",
        "## All EM/F1/R@5 Rows",
        "",
        "| Dataset | Comparison | Metric | Left | Right | Delta | 95% CI | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['metric']} | "
            f"{fmt(row['left_mean'])} | {fmt(row['right_mean'])} | {fmt(row['delta_mean'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {str(bool(row['ci_excludes_zero']))} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- DAEC/DAEC-selective significantly beat Top5 on 2Wiki and HotpotQA answer F1; on MuSiQue, base DAEC vs Top5 crosses zero but DAEC-selective is significant.",
        "- DAEC-selective significantly beats IRCoT-style and both LLM-direct variants on answer F1 across all three datasets.",
        "- DAEC-selective does not significantly beat SetR-style on answer F1 on any dataset. The SetR comparison must be framed as competitive/mixed, not as a clean win.",
        "- Unified title support shows why answer F1 and evidence coverage should both be reported: on MuSiQue, DAEC-selective improves support over base DAEC and remains close to SetR-style by support even though SetR-style is stronger on answer F1.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    csv_path = REPORT_DIR / "paired_ci.csv"
    json_path = REPORT_DIR / "paired_ci.json"
    md_path = REPORT_DIR / "paired_ci.md"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps({
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
