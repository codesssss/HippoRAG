#!/usr/bin/env python3
"""Conditional DAEC/DBEC analysis by support depth and gate resolvability."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_daec_selective_cross_pool import (  # noqa: E402
    PROPRAG_DAEC_DIR,
    PROPRAG_SELECTIVE_DIR,
    metric_rows,
    read_json,
    selective_path,
)
from analyze_daec_setr_cross_pool_ci import setr_path  # noqa: E402
from analyze_rankgpt_sliding_reader import (  # noqa: E402
    RANKGPT_METHOD,
    examples_metric_rows_with_titles,
    rankgpt_reader_path,
    setr_faithful_path,
)
from analyze_reviewer_baseline_paired_ci import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    examples_metric_rows,
    metric_mean,
    occurrence_keys,
    paired_bootstrap,
    safe_float,
    title_multiset_recall,
)


OUT_DIR = Path("reports/daec_depth_resolvability_20260508")
NOBINDING_DIR = Path("run_logs/daec_nobinding_proprag_full1000_20260506")
POOL = "PropRAG"
POOL_SLUG = "proprag"

METHODS = (
    "Top-5",
    "DBEC",
    "DBEC-IG",
    "DBEC-nobinding",
    "SetR-style k20",
    "SetR-faithful",
    RANKGPT_METHOD,
)
PRIMARY_METHODS = ("DBEC-IG", "DBEC-nobinding", "SetR-faithful", RANKGPT_METHOD)
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("DBEC-IG", "DBEC-nobinding"),
    ("DBEC-IG", "DBEC"),
    ("DBEC-IG", "SetR-faithful"),
    ("DBEC-IG", "SetR-style k20"),
    ("DBEC-IG", RANKGPT_METHOD),
    ("DBEC-nobinding", "SetR-faithful"),
)
METRICS = ("EM", "F1", "R5_TITLE")

MetaRow = Mapping[str, Any]
SlicePredicate = Callable[[MetaRow], bool]


def nobinding_path(slug: str) -> Path:
    return NOBINDING_DIR / f"{slug}_proprag_wiki_title_daec_noisyor_nobind_full1000.json"


def daec_path(slug: str) -> Path:
    return PROPRAG_DAEC_DIR / f"{slug}_proprag_wiki_title_daec_llm_full1000.json"


def support_depth(trace: Mapping[str, Any]) -> int:
    value = trace.get("gold_doc_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return len(list(trace.get("gold_titles") or []))


def gate_decision(trace: Mapping[str, Any]) -> str:
    selector_trace = trace.get("selector_trace") or {}
    decision = str(selector_trace.get("selective_binding_decision") or "").strip().lower()
    return decision if decision in {"bind", "abstain"} else "unknown"


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return safe_float(value)


def trace_metadata_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in data.get("setwise_selector_query_traces") or []:
        if not isinstance(trace, Mapping):
            continue
        selector_trace = trace.get("selector_trace") or {}
        rows.append(
            {
                "question": str(trace.get("question") or ""),
                "gold_doc_count": support_depth(trace),
                "gate_decision": gate_decision(trace),
                "title_unique_rate": optional_float(selector_trace.get("selective_binding_title_unique_rate")),
                "binding_count": int(selector_trace.get("binding_count") or 0),
                "binding_count_unpruned": int(selector_trace.get("binding_count_unpruned") or 0),
                "requirement_count": int(selector_trace.get("requirement_count") or 0),
            }
        )
    return rows


def align_rows_by_reference(
    *,
    dataset: str,
    reference_method: str,
    methods: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    reference_keys = occurrence_keys(methods[reference_method])
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

    reference_gold = [list(row.get("gold_titles") or []) for row in aligned[reference_method]]
    for rows in aligned.values():
        for row, gold_titles in zip(rows, reference_gold):
            if not row.get("gold_titles"):
                row["gold_titles"] = list(gold_titles)
            row["R5_TITLE"] = title_multiset_recall(list(gold_titles), list(row.get("top_titles") or []))
    return aligned


def load_dataset(dataset: str, slug: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    selective_data = read_json(selective_path(PROPRAG_SELECTIVE_DIR, slug, POOL_SLUG))
    metadata = trace_metadata_rows(selective_data)
    methods = align_rows_by_reference(
        dataset=dataset,
        reference_method="DBEC-IG",
        methods={
            "Top-5": metric_rows(selective_data, metric_field="baseline_metrics", title_field="baseline_top_titles"),
            "DBEC-IG": metric_rows(selective_data, metric_field="selector_metrics", title_field="selector_top_titles"),
            "DBEC": metric_rows(read_json(daec_path(slug)), metric_field="selector_metrics", title_field="selector_top_titles"),
            "DBEC-nobinding": metric_rows(read_json(nobinding_path(slug)), metric_field="selector_metrics", title_field="selector_top_titles"),
            "SetR-style k20": examples_metric_rows(read_json(setr_path(slug, POOL_SLUG))),
            "SetR-faithful": examples_metric_rows_with_titles(read_json(setr_faithful_path(slug))),
            RANKGPT_METHOD: examples_metric_rows_with_titles(read_json(rankgpt_reader_path(slug))),
        },
    )
    if len(metadata) != len(methods["DBEC-IG"]):
        raise ValueError(f"{dataset}: metadata length={len(metadata)}, expected={len(methods['DBEC-IG'])}")
    return metadata, methods


def slice_specs() -> tuple[tuple[str, SlicePredicate], ...]:
    return (
        ("all", lambda row: True),
        ("support_depth=2", lambda row: int(row["gold_doc_count"]) == 2),
        ("support_depth>=3", lambda row: int(row["gold_doc_count"]) >= 3),
        ("support_depth>=4", lambda row: int(row["gold_doc_count"]) >= 4),
        ("gate=bind", lambda row: row["gate_decision"] == "bind"),
        ("gate=abstain", lambda row: row["gate_decision"] == "abstain"),
        ("support_depth=2|gate=bind", lambda row: int(row["gold_doc_count"]) == 2 and row["gate_decision"] == "bind"),
        ("support_depth=2|gate=abstain", lambda row: int(row["gold_doc_count"]) == 2 and row["gate_decision"] == "abstain"),
        ("support_depth>=3|gate=bind", lambda row: int(row["gold_doc_count"]) >= 3 and row["gate_decision"] == "bind"),
        ("support_depth>=3|gate=abstain", lambda row: int(row["gold_doc_count"]) >= 3 and row["gate_decision"] == "abstain"),
        ("support_depth>=4|gate=bind", lambda row: int(row["gold_doc_count"]) >= 4 and row["gate_decision"] == "bind"),
        ("support_depth>=4|gate=abstain", lambda row: int(row["gold_doc_count"]) >= 4 and row["gate_decision"] == "abstain"),
    )


def subset_rows(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [rows[index] for index in indices]


def mean_optional(values: list[float | None]) -> float | str:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else ""


def build_slice_profile(dataset: str, slice_name: str, metadata: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    rows = subset_rows(metadata, indices)
    depths = [int(row["gold_doc_count"]) for row in rows]
    decisions = [str(row["gate_decision"]) for row in rows]
    return {
        "pool": POOL,
        "dataset": dataset,
        "slice": slice_name,
        "n": len(rows),
        "min_gold_doc_count": min(depths) if depths else "",
        "max_gold_doc_count": max(depths) if depths else "",
        "mean_gold_doc_count": float(np.mean(depths)) if depths else "",
        "bind_count": sum(decision == "bind" for decision in decisions),
        "abstain_count": sum(decision == "abstain" for decision in decisions),
        "unknown_count": sum(decision == "unknown" for decision in decisions),
        "bind_rate": sum(decision == "bind" for decision in decisions) / max(len(rows), 1),
        "abstain_rate": sum(decision == "abstain" for decision in decisions) / max(len(rows), 1),
        "title_unique_rate_mean": mean_optional([row.get("title_unique_rate") for row in rows]),
        "binding_count_mean": float(np.mean([int(row["binding_count"]) for row in rows])) if rows else "",
        "requirement_count_mean": float(np.mean([int(row["requirement_count"]) for row in rows])) if rows else "",
    }


def build_method_summary(
    dataset: str,
    slice_name: str,
    methods: Mapping[str, list[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = list(methods[method])
        rows.append(
            {
                "pool": POOL,
                "dataset": dataset,
                "slice": slice_name,
                "method": method,
                "n": len(method_rows),
                "EM": metric_mean(method_rows, "EM"),
                "F1": metric_mean(method_rows, "F1"),
                "R5_TITLE": metric_mean(method_rows, "R5_TITLE"),
            }
        )
    return rows


def build_ci_rows(
    dataset: str,
    slice_name: str,
    methods: Mapping[str, list[Mapping[str, Any]]],
    *,
    seed_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in COMPARISONS:
        for metric in METRICS:
            stats = paired_bootstrap(
                [safe_float(row.get(metric)) for row in methods[left]],
                [safe_float(row.get(metric)) for row in methods[right]],
                seed=BOOTSTRAP_SEED + 20260508 + seed_offset + len(rows) * 67,
                samples=BOOTSTRAP_SAMPLES,
            )
            rows.append(
                {
                    "pool": POOL,
                    "dataset": dataset,
                    "slice": slice_name,
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "left_mean": metric_mean(list(methods[left]), metric),
                    "right_mean": metric_mean(list(methods[right]), metric),
                    **stats,
                }
            )
    return rows


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    profile_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    seed_offset = 0
    for dataset, slug in DATASETS:
        metadata, methods = load_dataset(dataset, slug)
        for slice_name, predicate in slice_specs():
            indices = [index for index, row in enumerate(metadata) if predicate(row)]
            if not indices:
                continue
            slice_methods = {
                name: subset_rows(list(method_rows), indices)
                for name, method_rows in methods.items()
            }
            profile_rows.append(build_slice_profile(dataset, slice_name, metadata, indices))
            summary_rows.extend(build_method_summary(dataset, slice_name, slice_methods))
            ci_rows.extend(build_ci_rows(dataset, slice_name, slice_methods, seed_offset=seed_offset))
            seed_offset += 997
    return profile_rows, summary_rows, ci_rows


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
    if value == "":
        return "--"
    return f"{safe_float(value):.{digits}f}"


def sign_fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):+.{digits}f}"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def find_summary(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    slice_name: str,
    method: str,
) -> Mapping[str, Any] | None:
    for row in rows:
        if row["dataset"] == dataset and row["slice"] == slice_name and row["method"] == method:
            return row
    return None


def find_ci(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    slice_name: str,
    comparison: str,
    metric: str,
) -> Mapping[str, Any] | None:
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["slice"] == slice_name
            and row["comparison"] == comparison
            and row["metric"] == metric
        ):
            return row
    return None


def require_ci(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    slice_name: str,
    comparison: str,
    metric: str = "F1",
) -> Mapping[str, Any]:
    row = find_ci(rows, dataset=dataset, slice_name=slice_name, comparison=comparison, metric=metric)
    if row is None:
        raise KeyError((dataset, slice_name, comparison, metric))
    return row


def ci_text(row: Mapping[str, Any]) -> str:
    return f"[{sign_fmt(row['ci_low'])}, {sign_fmt(row['ci_high'])}]"


def append_conditional_table(
    lines: list[str],
    summary_rows: list[Mapping[str, Any]],
    ci_rows: list[Mapping[str, Any]],
    row_specs: tuple[tuple[str, str], ...],
) -> None:
    lines.extend(
        [
            "",
            "## Main Conditional F1 Table",
            "",
            "| Dataset | Slice | N | DBEC-IG | Nobinding | SetR-faithful | RankGPT-style | DBEC-IG - Nobind | 95% CI | DBEC-IG - SetR | 95% CI | DBEC-IG - RankGPT | 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, slice_name in row_specs:
        method_rows = {
            method: find_summary(summary_rows, dataset=dataset, slice_name=slice_name, method=method)
            for method in PRIMARY_METHODS
        }
        if any(row is None for row in method_rows.values()):
            continue
        n = int(method_rows["DBEC-IG"]["n"])  # type: ignore[index]
        vs_nobind = find_ci(ci_rows, dataset=dataset, slice_name=slice_name, comparison="DBEC-IG - DBEC-nobinding", metric="F1")
        vs_setr = find_ci(ci_rows, dataset=dataset, slice_name=slice_name, comparison="DBEC-IG - SetR-faithful", metric="F1")
        vs_rankgpt = find_ci(ci_rows, dataset=dataset, slice_name=slice_name, comparison=f"DBEC-IG - {RANKGPT_METHOD}", metric="F1")
        if vs_nobind is None or vs_setr is None or vs_rankgpt is None:
            continue
        lines.append(
            f"| {dataset} | `{slice_name}` | {n} | "
            f"{fmt(method_rows['DBEC-IG']['F1'])} | {fmt(method_rows['DBEC-nobinding']['F1'])} | "
            f"{fmt(method_rows['SetR-faithful']['F1'])} | {fmt(method_rows[RANKGPT_METHOD]['F1'])} | "
            f"{sign_fmt(vs_nobind['delta_mean'])} | {ci_text(vs_nobind)} | "
            f"{sign_fmt(vs_setr['delta_mean'])} | {ci_text(vs_setr)} | "
            f"{sign_fmt(vs_rankgpt['delta_mean'])} | {ci_text(vs_rankgpt)} |"
        )


def build_takeaways(ci_rows: list[Mapping[str, Any]]) -> list[str]:
    wiki_hard_bind_nobind = require_ci(
        ci_rows,
        dataset="2Wiki",
        slice_name="support_depth>=3|gate=bind",
        comparison="DBEC-IG - DBEC-nobinding",
    )
    wiki_hard_bind_setr = require_ci(
        ci_rows,
        dataset="2Wiki",
        slice_name="support_depth>=3|gate=bind",
        comparison="DBEC-IG - SetR-faithful",
    )
    wiki_hard_bind_rankgpt = require_ci(
        ci_rows,
        dataset="2Wiki",
        slice_name="support_depth>=3|gate=bind",
        comparison=f"DBEC-IG - {RANKGPT_METHOD}",
    )
    hotpot_bind_nobind = require_ci(
        ci_rows,
        dataset="HotpotQA",
        slice_name="gate=bind",
        comparison="DBEC-IG - DBEC-nobinding",
    )
    musique_deep_bind_nobind = require_ci(
        ci_rows,
        dataset="MuSiQue",
        slice_name="support_depth>=4|gate=bind",
        comparison="DBEC-IG - DBEC-nobinding",
    )
    musique_deep_bind_setr = require_ci(
        ci_rows,
        dataset="MuSiQue",
        slice_name="support_depth>=4|gate=bind",
        comparison="DBEC-IG - SetR-faithful",
    )
    return [
        "- On the 2Wiki high-support-depth, gate-bind slice, explicit binding is load-bearing: "
        f"DBEC-IG beats DBEC-nobinding by `{sign_fmt(wiki_hard_bind_nobind['delta_mean'])}` F1 "
        f"with 95% CI `{ci_text(wiki_hard_bind_nobind)}`.",
        "- The same 2Wiki slice also preserves the stronger-baseline story: "
        f"DBEC-IG beats SetR-faithful by `{sign_fmt(wiki_hard_bind_setr['delta_mean'])}` F1 "
        f"and RankGPT-style sliding by `{sign_fmt(wiki_hard_bind_rankgpt['delta_mean'])}` F1.",
        "- HotpotQA is not where the mechanism shows large marginal value: on gate-bind cases, "
        f"DBEC-IG - DBEC-nobinding is only `{sign_fmt(hotpot_bind_nobind['delta_mean'])}` F1.",
        "- MuSiQue is the boundary case, not a clean positive replication: on support-depth>=4 and gate-bind, "
        f"DBEC-IG trails DBEC-nobinding by `{sign_fmt(musique_deep_bind_nobind['delta_mean'])}` F1 "
        f"and SetR-faithful by `{sign_fmt(musique_deep_bind_setr['delta_mean'])}` F1.",
        "- Therefore the paper-safe claim is conditional: explicit binding provides strong value on 2Wiki deep, resolvable dependencies, while MuSiQue exposes the ambiguity/budget boundary that motivates the identifiability gate and limitations section.",
    ]


def append_gate_fallback_check(lines: list[str], ci_rows: list[Mapping[str, Any]]) -> None:
    lines.extend(
        [
            "",
            "## Gate Fallback Check",
            "",
            "| Dataset | Slice | DBEC-IG - Nobinding F1 | 95% CI | Exact fallback? |",
            "|---|---|---:|---:|---|",
        ]
    )
    for dataset, _ in DATASETS:
        row = find_ci(
            ci_rows,
            dataset=dataset,
            slice_name="gate=abstain",
            comparison="DBEC-IG - DBEC-nobinding",
            metric="F1",
        )
        if row is None:
            continue
        exact = abs(safe_float(row["delta_mean"])) < 1e-12 and abs(safe_float(row["ci_low"])) < 1e-12 and abs(safe_float(row["ci_high"])) < 1e-12
        lines.append(f"| {dataset} | `gate=abstain` | {sign_fmt(row['delta_mean'])} | {ci_text(row)} | {yes_no(exact)} |")


def build_markdown(
    profile_rows: list[Mapping[str, Any]],
    summary_rows: list[Mapping[str, Any]],
    ci_rows: list[Mapping[str, Any]],
) -> str:
    row_specs = (
        ("2Wiki", "all"),
        ("2Wiki", "gate=bind"),
        ("2Wiki", "gate=abstain"),
        ("2Wiki", "support_depth>=3|gate=bind"),
        ("HotpotQA", "all"),
        ("HotpotQA", "gate=bind"),
        ("HotpotQA", "gate=abstain"),
        ("MuSiQue", "all"),
        ("MuSiQue", "gate=bind"),
        ("MuSiQue", "gate=abstain"),
        ("MuSiQue", "support_depth>=3|gate=bind"),
        ("MuSiQue", "support_depth>=4|gate=bind"),
        ("MuSiQue", "support_depth>=4|gate=abstain"),
    )
    lines = [
        "# DAEC/DBEC Depth x Gate-Resolvability Conditional Analysis",
        "",
        "Date: 2026-05-08",
        "",
        "Purpose: test the paper narrative that explicit dependency binding has conditional value rather than universal dominance. This report slices existing PropRAG full1000 outputs by support depth and the pre-selection DBEC-IG title-uniqueness gate decision.",
        "",
        "Protocol: existing outputs only; no new LLM calls. `support_depth` is measured by gold support-document count, not manually annotated reasoning depth. `gate=bind` is a pre-selection binding-resolvability proxy from DBEC-IG's title-uniqueness gate; it is not an oracle identifiability label. CIs use query-paired percentile bootstrap with 10,000 resamples.",
        "",
        "## Slice Profile",
        "",
        "| Dataset | Slice | N | Mean Support Depth | Bind Rate | Abstain Rate | Mean Title-Unique Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, slice_name in row_specs:
        row = next((item for item in profile_rows if item["dataset"] == dataset and item["slice"] == slice_name), None)
        if row is None:
            continue
        lines.append(
            f"| {dataset} | `{slice_name}` | {int(row['n'])} | {fmt(row['mean_gold_doc_count'])} | "
            f"{fmt(row['bind_rate'])} | {fmt(row['abstain_rate'])} | {fmt(row['title_unique_rate_mean'])} |"
        )
    append_conditional_table(lines, summary_rows, ci_rows, row_specs)
    append_gate_fallback_check(lines, ci_rows)
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            *build_takeaways(ci_rows),
            "",
            "## Paper-Facing Interpretation",
            "",
            "Allowed:",
            "",
            "```text",
            "DBEC-IG's explicit binding is strongly load-bearing on the 2Wiki high-support-depth gate-bind slice, but this is a conditional mechanism result rather than a universal advantage over set selection.",
            "```",
            "",
            "Allowed:",
            "",
            "```text",
            "MuSiQue exposes the boundary of the current binding instantiation: deep support count alone is insufficient when intermediate referents are ambiguous or reader budget is too tight.",
            "```",
            "",
            "Not allowed:",
            "",
            "```text",
            "Explicit binding is universally better on all deep multi-hop queries.",
            "```",
            "",
            "Not allowed:",
            "",
            "```text",
            "The gate decision is a ground-truth dependency identifiability label.",
            "```",
            "",
            "## Files",
            "",
            f"- Slice profile CSV: `{OUT_DIR / 'slice_profile.csv'}`",
            f"- Method summary CSV: `{OUT_DIR / 'method_summary.csv'}`",
            f"- Paired CI CSV: `{OUT_DIR / 'paired_ci.csv'}`",
            f"- JSON: `{OUT_DIR / 'summary.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    profile_rows, summary_rows, ci_rows = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(profile_rows, OUT_DIR / "slice_profile.csv")
    write_csv(summary_rows, OUT_DIR / "method_summary.csv")
    write_csv(ci_rows, OUT_DIR / "paired_ci.csv")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "slice_profile": profile_rows,
                "method_summary": summary_rows,
                "paired_ci": ci_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.md").write_text(build_markdown(profile_rows, summary_rows, ci_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
