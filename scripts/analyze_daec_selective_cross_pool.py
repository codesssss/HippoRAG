#!/usr/bin/env python3
"""Cross-pool paired-CI audit for DAEC selective binding."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_reviewer_baseline_paired_ci import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    metric_mean,
    normalize_title,
    occurrence_keys,
    paired_bootstrap,
    safe_float,
    title_multiset_recall,
)


OUT_DIR = Path("reports/daec_selective_cross_pool_20260508")
DENSE_HIPPO_SELECTIVE_DIR = Path("run_logs/daec_selective_titleuniq_dense_hipporag_full1000_20260506")
DENSE_HIPPO_DAEC_DIR = Path("run_logs/daec_llm_wiki_title_dense_hipporag_full1000_20260503")
PROPRAG_SELECTIVE_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")
PROPRAG_DAEC_DIR = Path("run_logs/daec_llm_wiki_title_proprag_full1000_20260503")

POOLS: tuple[tuple[str, str, Path, Path], ...] = (
    ("Dense", "dense", DENSE_HIPPO_SELECTIVE_DIR, DENSE_HIPPO_DAEC_DIR),
    ("HippoRAG", "hipporag", DENSE_HIPPO_SELECTIVE_DIR, DENSE_HIPPO_DAEC_DIR),
    ("PropRAG", "proprag", PROPRAG_SELECTIVE_DIR, PROPRAG_DAEC_DIR),
)
METHODS = ("Top-5", "DAEC", "DAEC-selective")
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("DAEC-selective", "Top-5"),
    ("DAEC", "Top-5"),
    ("DAEC-selective", "DAEC"),
)
METRICS = ("EM", "F1", "R5_TITLE")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selective_path(run_dir: Path, slug: str, pool_slug: str) -> Path:
    return run_dir / f"{slug}_{pool_slug}_wiki_title_daec_selective_titleuniq_full1000.json"


def daec_path(run_dir: Path, slug: str, pool_slug: str) -> Path:
    return run_dir / f"{slug}_{pool_slug}_wiki_title_daec_llm_full1000.json"


def selector_traces(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row for row in data.get("setwise_selector_query_traces", [])
        if isinstance(row, Mapping)
    ]


def metric_rows(
    data: Mapping[str, Any],
    *,
    metric_field: str,
    title_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in selector_traces(data):
        metrics = trace.get(metric_field) or {}
        gold_titles = list(trace.get("gold_titles") or [])
        top_titles = list(trace.get(title_field) or [])[:5]
        rows.append(
            {
                "question": str(trace.get("question") or ""),
                "EM": safe_float(metrics.get("ExactMatch")),
                "F1": safe_float(metrics.get("F1")),
                "gold_titles": gold_titles,
                "top_titles": top_titles,
                "R5_TITLE": title_multiset_recall(gold_titles, top_titles),
            }
        )
    return rows


def align_methods(
    *,
    pool: str,
    dataset: str,
    methods: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    reference_keys = occurrence_keys(methods["DAEC-selective"])
    aligned: dict[str, list[dict[str, Any]]] = {}
    for name, rows in methods.items():
        keys = occurrence_keys(rows)
        if len(keys) != len(reference_keys):
            raise ValueError(f"{pool}/{dataset}: {name} length={len(keys)}, expected={len(reference_keys)}")
        keyed = {key: row for key, row in zip(keys, rows)}
        missing = [key for key in reference_keys if key not in keyed]
        if missing:
            raise ValueError(f"{pool}/{dataset}: {name} missing {len(missing)} occurrence-aligned questions")
        aligned[name] = [keyed[key] for key in reference_keys]
    return aligned


def normalized_titles(titles: list[Any]) -> list[str]:
    return [normalize_title(title) for title in titles if normalize_title(title)]


def top_titles_equal(left: list[Any], right: list[Any]) -> bool:
    return normalized_titles(left) == normalized_titles(right)


def build_method_summary(pool: str, dataset: str, methods: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows.append(
            {
                "pool": pool,
                "dataset": dataset,
                "method": method,
                "n": len(methods[method]),
                "EM": metric_mean(list(methods[method]), "EM"),
                "F1": metric_mean(list(methods[method]), "F1"),
                "R5_TITLE": metric_mean(list(methods[method]), "R5_TITLE"),
            }
        )
    return rows


def build_ci_rows(pool: str, dataset: str, methods: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in COMPARISONS:
        for metric in METRICS:
            stats = paired_bootstrap(
                [safe_float(row.get(metric)) for row in methods[left]],
                [safe_float(row.get(metric)) for row in methods[right]],
                seed=BOOTSTRAP_SEED + 20260508 + len(rows) * 47 + len(pool) * 1009 + len(dataset) * 101,
                samples=BOOTSTRAP_SAMPLES,
            )
            rows.append(
                {
                    "pool": pool,
                    "dataset": dataset,
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "left_mean": metric_mean(list(methods[left]), metric),
                    "right_mean": metric_mean(list(methods[right]), metric),
                    **stats,
                }
            )
    return rows


def build_router_row(
    pool: str,
    dataset: str,
    selective_data: Mapping[str, Any],
    methods: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    traces = selector_traces(selective_data)
    decisions = [str((trace.get("selector_trace") or {}).get("selective_binding_decision") or "") for trace in traces]
    bind_count = sum(decision == "bind" for decision in decisions)
    abstain_count = sum(decision == "abstain" for decision in decisions)
    same_as_daec = [
        top_titles_equal(
            list(selective_row.get("top_titles") or []),
            list(daec_row.get("top_titles") or []),
        )
        for selective_row, daec_row in zip(methods["DAEC-selective"], methods["DAEC"])
    ]
    same_as_top5 = [
        top_titles_equal(
            list(selective_row.get("top_titles") or []),
            list(top5_row.get("top_titles") or []),
        )
        for selective_row, top5_row in zip(methods["DAEC-selective"], methods["Top-5"])
    ]
    abstain_same_as_daec = sum(
        decision == "abstain" and same
        for decision, same in zip(decisions, same_as_daec)
    )
    abstain_changed_from_daec = sum(
        decision == "abstain" and not same
        for decision, same in zip(decisions, same_as_daec)
    )
    return {
        "pool": pool,
        "dataset": dataset,
        "n": len(traces),
        "bind_count": bind_count,
        "abstain_count": abstain_count,
        "bind_rate": bind_count / max(len(traces), 1),
        "abstain_rate": abstain_count / max(len(traces), 1),
        "same_as_daec_count": sum(same_as_daec),
        "same_as_daec_rate": sum(same_as_daec) / max(len(same_as_daec), 1),
        "same_as_top5_count": sum(same_as_top5),
        "same_as_top5_rate": sum(same_as_top5) / max(len(same_as_top5), 1),
        "abstain_same_as_daec": abstain_same_as_daec,
        "abstain_changed_from_daec": abstain_changed_from_daec,
    }


def load_pool_dataset(pool_name: str, pool_slug: str, selective_dir: Path, daec_dir: Path, dataset: str, slug: str) -> tuple[
    Mapping[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    selective_data = read_json(selective_path(selective_dir, slug, pool_slug))
    daec_data = read_json(daec_path(daec_dir, slug, pool_slug))
    methods = align_methods(
        pool=pool_name,
        dataset=dataset,
        methods={
            "Top-5": metric_rows(selective_data, metric_field="baseline_metrics", title_field="baseline_top_titles"),
            "DAEC": metric_rows(daec_data, metric_field="selector_metrics", title_field="selector_top_titles"),
            "DAEC-selective": metric_rows(selective_data, metric_field="selector_metrics", title_field="selector_top_titles"),
        },
    )
    return selective_data, methods


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    for pool_name, pool_slug, selective_dir, daec_dir in POOLS:
        for dataset, slug in DATASETS:
            selective_data, methods = load_pool_dataset(pool_name, pool_slug, selective_dir, daec_dir, dataset, slug)
            summary_rows.extend(build_method_summary(pool_name, dataset, methods))
            ci_rows.extend(build_ci_rows(pool_name, dataset, methods))
            router_rows.append(build_router_row(pool_name, dataset, selective_data, methods))
    return summary_rows, ci_rows, router_rows


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):.{digits}f}"


def sign_fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):+.{digits}f}"


def pct(value: Any) -> str:
    return f"{100.0 * safe_float(value):.1f}%"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def ci_text(row: Mapping[str, Any]) -> str:
    return f"[{sign_fmt(row['ci_low'])}, {sign_fmt(row['ci_high'])}]"


def find_summary(
    rows: list[Mapping[str, Any]],
    *,
    pool: str,
    dataset: str,
    method: str,
) -> Mapping[str, Any]:
    for row in rows:
        if row["pool"] == pool and row["dataset"] == dataset and row["method"] == method:
            return row
    raise KeyError((pool, dataset, method))


def find_ci(
    rows: list[Mapping[str, Any]],
    *,
    pool: str,
    dataset: str,
    comparison: str,
    metric: str,
) -> Mapping[str, Any]:
    for row in rows:
        if row["pool"] == pool and row["dataset"] == dataset and row["comparison"] == comparison and row["metric"] == metric:
            return row
    raise KeyError((pool, dataset, comparison, metric))


def build_markdown(
    summary_rows: list[Mapping[str, Any]],
    ci_rows: list[Mapping[str, Any]],
    router_rows: list[Mapping[str, Any]],
) -> str:
    lines = [
        "# DAEC Selective Binding Cross-Pool Paired-CI Audit",
        "",
        "Date: 2026-05-08",
        "",
        "Purpose: close the reviewer-facing cross-pool stability gap for the identifiability-gated DAEC-selective variant using existing full1000 outputs. No new LLM calls are made.",
        "",
        "Protocol: fixed pool100 per retriever, Qwen3-8B `/no_think`, `qa_top_k=5`, `qa_doc_max_chars=2048`, title-uniqueness threshold `0.88`. Delta is left method minus right method; CIs use query-paired percentile bootstrap with 10,000 resamples.",
        "",
        "## F1 Cross-Pool Summary",
        "",
        "| Pool | Dataset | Top-5 F1 | DAEC F1 | DAEC-selective F1 | Selective - Top-5 F1 | 95% CI | Selective - DAEC F1 | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pool, _, _, _ in POOLS:
        for dataset, _ in DATASETS:
            top5 = find_summary(summary_rows, pool=pool, dataset=dataset, method="Top-5")
            daec = find_summary(summary_rows, pool=pool, dataset=dataset, method="DAEC")
            selective = find_summary(summary_rows, pool=pool, dataset=dataset, method="DAEC-selective")
            vs_top5 = find_ci(ci_rows, pool=pool, dataset=dataset, comparison="DAEC-selective - Top-5", metric="F1")
            vs_daec = find_ci(ci_rows, pool=pool, dataset=dataset, comparison="DAEC-selective - DAEC", metric="F1")
            lines.append(
                f"| {pool} | {dataset} | {fmt(top5['F1'])} | {fmt(daec['F1'])} | {fmt(selective['F1'])} | "
                f"{sign_fmt(vs_top5['delta_mean'])} | {ci_text(vs_top5)} | "
                f"{sign_fmt(vs_daec['delta_mean'])} | {ci_text(vs_daec)} |"
            )

    lines.extend(
        [
            "",
            "## Router Behavior",
            "",
            "| Pool | Dataset | Bind | Abstain | Selective same as DAEC | Abstain same as DAEC | Abstain changed from DAEC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in router_rows:
        lines.append(
            f"| {row['pool']} | {row['dataset']} | {int(row['bind_count'])} ({pct(row['bind_rate'])}) | "
            f"{int(row['abstain_count'])} ({pct(row['abstain_rate'])}) | "
            f"{int(row['same_as_daec_count'])}/{int(row['n'])} ({pct(row['same_as_daec_rate'])}) | "
            f"{int(row['abstain_same_as_daec'])} | {int(row['abstain_changed_from_daec'])} |"
        )

    lines.extend(
        [
            "",
            "## Detailed F1 CI",
            "",
            "| Pool | Dataset | Comparison | Left | Right | Delta | 95% CI | P(delta > 0) | Excludes 0 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ci_rows:
        if row["metric"] != "F1":
            continue
        lines.append(
            f"| {row['pool']} | {row['dataset']} | {row['comparison']} | "
            f"{fmt(row['left_mean'])} | {fmt(row['right_mean'])} | {sign_fmt(row['delta_mean'])} | "
            f"{ci_text(row)} | {fmt(row['p_delta_gt_0'])} | {yes_no(row['ci_excludes_zero'])} |"
        )

    selective_vs_daec = [row for row in ci_rows if row["comparison"] == "DAEC-selective - DAEC" and row["metric"] == "F1"]
    selective_vs_top5 = [row for row in ci_rows if row["comparison"] == "DAEC-selective - Top-5" and row["metric"] == "F1"]
    nonnegative_vs_daec = sum(safe_float(row["delta_mean"]) >= -1e-12 for row in selective_vs_daec)
    significant_vs_daec = [row for row in selective_vs_daec if bool(row["ci_excludes_zero"]) and safe_float(row["delta_mean"]) > 0]
    positive_vs_top5 = sum(safe_float(row["delta_mean"]) > 0 for row in selective_vs_top5)
    significant_vs_top5 = [row for row in selective_vs_top5 if bool(row["ci_excludes_zero"]) and safe_float(row["delta_mean"]) > 0]

    lines.extend(
        [
            "",
            "## Paper-Facing Takeaway",
            "",
            f"- DAEC-selective is non-negative relative to ungated DAEC in `{nonnegative_vs_daec}/9` pool-dataset cells by mean F1; statistically significant positive gains occur in `{len(significant_vs_daec)}/9` cells.",
            f"- DAEC-selective is positive relative to the same-pool Top-5 retriever baseline in `{positive_vs_top5}/9` cells by mean F1; statistically significant positive gains occur in `{len(significant_vs_top5)}/9` cells.",
            "- Router behavior confirms the gate is often score-preserving rather than score-changing: many abstentions reproduce the ungated DAEC top-5 exactly, especially outside MuSiQue.",
            "- Paper-safe framing: the identifiability gate is a conservative cross-pool safety layer with concentrated positive effect, not a large universal improvement mechanism.",
            "",
            "## Files",
            "",
            f"- Summary CSV: `{OUT_DIR / 'method_summary.csv'}`",
            f"- Paired CI CSV: `{OUT_DIR / 'paired_ci.csv'}`",
            f"- Router CSV: `{OUT_DIR / 'router_behavior.csv'}`",
            f"- JSON: `{OUT_DIR / 'summary.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, ci_rows, router_rows = build_rows()
    write_csv(summary_rows, OUT_DIR / "method_summary.csv")
    write_csv(ci_rows, OUT_DIR / "paired_ci.csv")
    write_csv(router_rows, OUT_DIR / "router_behavior.csv")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "method_summary": summary_rows,
                "paired_ci": ci_rows,
                "router_behavior": router_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.md").write_text(
        build_markdown(summary_rows, ci_rows, router_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(OUT_DIR), "summary_rows": len(summary_rows), "ci_rows": len(ci_rows)}, indent=2))


if __name__ == "__main__":
    main()
