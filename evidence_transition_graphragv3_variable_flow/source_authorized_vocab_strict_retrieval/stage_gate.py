"""Performance gate for STO GraphRAG stage promotion.

This module does not retrieve, rerank, or alter reader contexts.  It only reads
already produced retrieval/QA reports and decides whether a cleaner stage can
replace the current performance anchor without metric regressions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_QA_METHOD = "source_authorized_vocab_strict_graphrag"
GATE_METRICS = ("R@5", "EM", "F1")


@dataclass(frozen=True)
class StageReportSpec:
    """Input files for one stage on one dataset."""

    stage: str
    dataset: str
    retrieval_json: Path
    qa_json: Path


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def parse_stage_spec(raw: str) -> StageReportSpec:
    """Parse ``stage:dataset:retrieval_json:qa_json``."""

    parts = str(raw).split(":", 3)
    if len(parts) != 4 or not all(parts[:3]) or not parts[3]:
        raise ValueError(
            "stage spec must be stage:dataset:retrieval_json:qa_json; "
            f"got {raw!r}"
        )
    return StageReportSpec(
        stage=parts[0],
        dataset=parts[1],
        retrieval_json=Path(parts[2]),
        qa_json=Path(parts[3]),
    )


def stage_row_from_reports(
    spec: StageReportSpec,
    *,
    qa_method: str = DEFAULT_QA_METHOD,
) -> Dict[str, Any]:
    """Extract a normalized metric row for one stage/dataset."""

    retrieval = load_json(spec.retrieval_json)
    qa_payload = load_json(spec.qa_json)
    qa_metrics = qa_metrics_for_dataset(
        qa_payload,
        dataset=spec.dataset,
        qa_method=qa_method,
    )
    retrieval_metrics = retrieval.get("metrics", {}) or {}
    config = retrieval.get("config", {}) or {}
    return {
        "dataset": spec.dataset,
        "stage": spec.stage,
        "candidate_source": str(config.get("candidate_source", "")),
        "runner": str(retrieval.get("runner", "")),
        "certificate_policy": str(config.get("certificate_policy", "")),
        "R@5": float(qa_metrics.get("r5", retrieval_metrics.get("r5", 0.0)) or 0.0),
        "EM": float(qa_metrics.get("ExactMatch", 0.0) or 0.0),
        "F1": float(qa_metrics.get("F1", 0.0) or 0.0),
        "all_gold@5": float(
            qa_metrics.get("all_gold_at5", retrieval_metrics.get("all_gold_at5", 0.0))
            or 0.0
        ),
        "retrieval_json": str(spec.retrieval_json),
        "qa_json": str(spec.qa_json),
    }


def qa_metrics_for_dataset(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    qa_method: str = DEFAULT_QA_METHOD,
) -> Mapping[str, Any]:
    for dataset_row in payload.get("datasets", []) or []:
        if str(dataset_row.get("dataset")) != str(dataset):
            continue
        methods = dataset_row.get("methods", {}) or {}
        method_row = methods.get(qa_method)
        if not isinstance(method_row, Mapping):
            raise KeyError(f"QA method {qa_method!r} missing for dataset {dataset!r}")
        metrics = method_row.get("metrics", {}) or {}
        if not isinstance(metrics, Mapping):
            raise TypeError(f"QA metrics are malformed for dataset {dataset!r}")
        return metrics
    raise KeyError(f"dataset {dataset!r} not found in QA payload")


def prop_baseline_rows(
    payload: Mapping[str, Any],
    *,
    method: str = "PropRAG-local",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in payload.get("table", []) or []:
        if str(row.get("method")) != str(method):
            continue
        rows.append(
            {
                "dataset": str(row.get("dataset", "")),
                "stage": method,
                "candidate_source": "PropRAG",
                "runner": "baseline",
                "certificate_policy": "",
                "R@5": float(row.get("R@5", 0.0) or 0.0),
                "EM": float(row.get("EM", 0.0) or 0.0),
                "F1": float(row.get("F1", 0.0) or 0.0),
                "all_gold@5": None,
                "retrieval_json": "",
                "qa_json": "",
            }
        )
    return rows


def build_stage_gate_payload(
    *,
    stage_rows: Sequence[Mapping[str, Any]],
    prop_rows: Sequence[Mapping[str, Any]] = (),
    anchor_stage: str,
    gate_metrics: Sequence[str] = GATE_METRICS,
) -> Dict[str, Any]:
    """Build comparison rows and stage promotion decisions."""

    all_rows = [dict(row) for row in [*prop_rows, *stage_rows]]
    anchor_by_dataset = {
        str(row.get("dataset")): row
        for row in stage_rows
        if str(row.get("stage")) == str(anchor_stage)
    }
    prop_by_dataset = {
        str(row.get("dataset")): row
        for row in prop_rows
    }
    comparison_rows: List[Dict[str, Any]] = []
    for row in stage_rows:
        dataset = str(row.get("dataset"))
        anchor = anchor_by_dataset.get(dataset, {})
        prop = prop_by_dataset.get(dataset, {})
        comparison = dict(row)
        for metric in gate_metrics:
            comparison[f"delta_vs_anchor_{metric}"] = round(
                float(row.get(metric, 0.0) or 0.0)
                - float(anchor.get(metric, 0.0) or 0.0),
                6,
            )
            comparison[f"delta_vs_prop_{metric}"] = round(
                float(row.get(metric, 0.0) or 0.0)
                - float(prop.get(metric, 0.0) or 0.0),
                6,
            ) if prop else None
        comparison["passes_anchor_gate"] = all(
            float(comparison[f"delta_vs_anchor_{metric}"] or 0.0) >= 0.0
            for metric in gate_metrics
        )
        comparison["passes_prop_gate"] = all(
            comparison.get(f"delta_vs_prop_{metric}") is not None
            and float(comparison[f"delta_vs_prop_{metric}"] or 0.0) >= 0.0
            for metric in gate_metrics
        ) if prop else None
        comparison_rows.append(comparison)

    stage_decisions = []
    for stage in sorted({str(row.get("stage")) for row in stage_rows}):
        rows = [row for row in comparison_rows if str(row.get("stage")) == stage]
        stage_decisions.append(
            {
                "stage": stage,
                "promote_over_anchor": bool(rows)
                and all(bool(row.get("passes_anchor_gate")) for row in rows),
                "beats_prop_all_datasets": bool(rows)
                and all(bool(row.get("passes_prop_gate")) for row in rows),
                "row_count": len(rows),
            }
        )

    return {
        "schema_version": 1,
        "anchor_stage": str(anchor_stage),
        "gate_metrics": list(gate_metrics),
        "rows": all_rows,
        "comparisons": comparison_rows,
        "stage_decisions": stage_decisions,
    }


def format_metric_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Dataset",
        "Stage",
        "Candidate",
        "Runner",
        "Policy",
        "R@5",
        "EM",
        "F1",
    ]
    body = [
        [
            str(row.get("dataset", "")),
            str(row.get("stage", "")),
            str(row.get("candidate_source", "")),
            str(row.get("runner", "")),
            str(row.get("certificate_policy", "")),
            _fmt_float(row.get("R@5")),
            _fmt_float(row.get("EM")),
            _fmt_float(row.get("F1")),
        ]
        for row in rows
    ]
    return _format_table(headers, body)


def format_gate_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Dataset",
        "Stage",
        "dR@5 vs A",
        "dEM vs A",
        "dF1 vs A",
        "Gate A",
        "Gate Prop",
    ]
    body = [
        [
            str(row.get("dataset", "")),
            str(row.get("stage", "")),
            _fmt_signed(row.get("delta_vs_anchor_R@5")),
            _fmt_signed(row.get("delta_vs_anchor_EM")),
            _fmt_signed(row.get("delta_vs_anchor_F1")),
            "pass" if row.get("passes_anchor_gate") else "fail",
            _format_optional_gate(row.get("passes_prop_gate")),
        ]
        for row in rows
    ]
    return _format_table(headers, body)


def write_stage_gate_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# STO GraphRAG Stage Gate",
        "",
        "| field | value |",
        "| ----- | ----- |",
        f"| anchor stage | {payload.get('anchor_stage', '')} |",
        f"| gate metrics | {', '.join(payload.get('gate_metrics', []) or [])} |",
        "",
        "## Metrics",
        "",
        format_metric_table(payload.get("rows", []) or []),
        "",
        "## Promotion Gate",
        "",
        format_gate_table(payload.get("comparisons", []) or []),
        "",
        "## Stage Decisions",
        "",
        _format_table(
            ["Stage", "Promote Over Anchor", "Beats Prop All Datasets", "Rows"],
            [
                [
                    str(row.get("stage", "")),
                    "yes" if row.get("promote_over_anchor") else "no",
                    "yes" if row.get("beats_prop_all_datasets") else "no",
                    str(row.get("row_count", 0)),
                ]
                for row in payload.get("stage_decisions", []) or []
            ],
        ),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _format_optional_gate(value: object) -> str:
    if value is None:
        return "n/a"
    return "pass" if value else "fail"


def _fmt_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _fmt_signed(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):+.4f}"


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(headers[col])), *(len(str(row[col])) for row in rows))
        for col in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(str(headers[col]).ljust(widths[col]) for col in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in range(len(headers))) + " |"
        )
    return "\n".join(lines)

