"""Compare two per-query retrieval reports for one variant pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


TRACE_SUMMARY_KEYS = (
    "selected_policy",
    "path_admission_status",
    "source_bound_evidence_edge_count",
    "source_document_role_target_count",
    "source_title_role_induced_candidate_count",
    "source_text_mct_source_count",
    "query_mentioned_shallow_frontier_source_count",
    "source_text_path_exposed_source_count",
    "source_text_evidence_set_mct_state_count",
    "source_text_evidence_set_mct_terminal_context_count",
    "method_internal_source_anchor_indices_preview",
    "path_new_node_indices",
    "displaced_incumbent_indices",
    "selected_reader_context_indices",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_variant_rows(path: Path, variant: str) -> Dict[int, Mapping[str, Any]]:
    payload = load_json(path)
    variants = payload.get("variants", {}) if isinstance(payload, Mapping) else {}
    rows = variants.get(variant)
    if not isinstance(rows, list):
        raise KeyError(f"Variant {variant!r} not found as a per-query row list in {path}")
    return {int(row["query_index"]): row for row in rows}


def recall_at_k(row: Mapping[str, Any], *, k: int) -> float | None:
    gold = {int(item) for item in row.get("gold_doc_indices", []) or [] if item is not None}
    if not gold:
        return None
    retrieved = {
        int(item)
        for item in (row.get(f"retrieved_doc_indices_top{k}", []) or [])[:k]
        if item is not None
    }
    return len(gold & retrieved) / len(gold)


def trace_summary_diff(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    left_trace = left.get("route_trace", {}) or {}
    right_trace = right.get("route_trace", {}) or {}
    output: Dict[str, Dict[str, Any]] = {}
    for key in TRACE_SUMMARY_KEYS:
        left_value = left_trace.get(key)
        right_value = right_trace.get(key)
        if left_value != right_value:
            output[key] = {
                "left": left_value,
                "right": right_value,
            }
    return output


def compare_per_query_reports(
    *,
    left_path: Path,
    left_variant: str,
    right_path: Path,
    right_variant: str,
    k: int = 5,
) -> Dict[str, Any]:
    left_rows = load_variant_rows(left_path, left_variant)
    right_rows = load_variant_rows(right_path, right_variant)
    common_indices = sorted(set(left_rows) & set(right_rows))
    rows: List[Dict[str, Any]] = []
    topk_docdiff = 0
    changed = 0
    right_losses = 0
    right_gains = 0
    for query_index in common_indices:
        left = left_rows[query_index]
        right = right_rows[query_index]
        left_topk = list(left.get(f"retrieved_doc_indices_top{k}", []) or [])[:k]
        right_topk = list(right.get(f"retrieved_doc_indices_top{k}", []) or [])[:k]
        if left_topk != right_topk:
            topk_docdiff += 1
        left_recall = recall_at_k(left, k=k)
        right_recall = recall_at_k(right, k=k)
        if left_recall == right_recall:
            continue
        changed += 1
        delta = None if left_recall is None or right_recall is None else right_recall - left_recall
        if delta is not None and delta < 0:
            right_losses += 1
        elif delta is not None and delta > 0:
            right_gains += 1
        rows.append(
            {
                "query_index": int(query_index),
                "question": str(left.get("question", "")),
                "left_recall": left_recall,
                "right_recall": right_recall,
                "delta": None if delta is None else round(float(delta), 4),
                "left_titles_topk": list(left.get(f"retrieved_titles_top{k}", []) or [])[:k],
                "right_titles_topk": list(right.get(f"retrieved_titles_top{k}", []) or [])[:k],
                "trace_diffs": trace_summary_diff(left, right),
            }
        )
    return {
        "schema_version": 1,
        "left_path": str(left_path),
        "left_variant": str(left_variant),
        "right_path": str(right_path),
        "right_variant": str(right_variant),
        "k": int(k),
        "common_query_count": len(common_indices),
        "topk_docdiff_count": int(topk_docdiff),
        "recall_changed_count": int(changed),
        "right_loss_count": int(right_losses),
        "right_gain_count": int(right_gains),
        "rows": rows,
    }


def format_summary_table(payload: Mapping[str, Any]) -> str:
    headers = ["Metric", "Value"]
    body = [
        ["common_query_count", str(payload.get("common_query_count", ""))],
        ["topk_docdiff_count", str(payload.get("topk_docdiff_count", ""))],
        ["recall_changed_count", str(payload.get("recall_changed_count", ""))],
        ["right_loss_count", str(payload.get("right_loss_count", ""))],
        ["right_gain_count", str(payload.get("right_gain_count", ""))],
    ]
    widths = [max(len(headers[col]), *(len(row[col]) for row in body)) for col in range(2)]
    lines = [
        "| " + " | ".join(headers[col].ljust(widths[col]) for col in range(2)) + " |",
        "| " + " | ".join("-" * widths[col] for col in range(2)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row[col].ljust(widths[col]) for col in range(2)) + " |")
    return "\n".join(lines)


def format_changed_query_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = ["Query", "Left R@5", "Right R@5", "Delta", "Question"]
    body: List[List[str]] = []
    for row in rows:
        body.append(
            [
                str(row.get("query_index", "")),
                f"{float(row.get('left_recall') or 0.0):.4f}",
                f"{float(row.get('right_recall') or 0.0):.4f}",
                f"{float(row.get('delta') or 0.0):+.4f}",
                str(row.get("question", ""))[:80],
            ]
        )
    if not body:
        body = [["-", "-", "-", "-", "No recall changes"]]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in body))
        for col in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[col].ljust(widths[col]) for col in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row[col].ljust(widths[col]) for col in range(len(headers))) + " |")
    return "\n".join(lines)


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Per-Query Retrieval Diff",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| left_variant | {payload.get('left_variant')} |",
        f"| right_variant | {payload.get('right_variant')} |",
        f"| k | {payload.get('k')} |",
        "",
        format_summary_table(payload),
        "",
        format_changed_query_table(payload.get("rows", []) or []),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-path", required=True)
    parser.add_argument("--left-variant", required=True)
    parser.add_argument("--right-path", required=True)
    parser.add_argument("--right-variant", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = compare_per_query_reports(
        left_path=Path(args.left_path),
        left_variant=str(args.left_variant),
        right_path=Path(args.right_path),
        right_variant=str(args.right_variant),
        k=int(args.k),
    )
    print(format_summary_table(payload))
    print()
    print(format_changed_query_table(payload.get("rows", []) or []))
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        write_markdown(payload, Path(args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
