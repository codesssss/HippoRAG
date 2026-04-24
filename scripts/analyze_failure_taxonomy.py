#!/usr/bin/env python3
"""Produce a conservative failure taxonomy from DAEC/DtC evaluation traces.

The automatic labels are intentionally modest.  They separate robust categories
that can be inferred from gold-title coverage and reader F1, and emit detailed
case records for manual wrong-entity/binding annotation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


EPS = 1e-9


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def norm_title(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def title_set(values: Iterable[str]) -> set[str]:
    return {norm_title(v) for v in values if norm_title(v)}


def covers_gold(selected_titles: Sequence[str], gold_titles: Sequence[str]) -> bool:
    gold = title_set(gold_titles)
    selected = title_set(selected_titles)
    return bool(gold) and gold.issubset(selected)


def get_step_requirement_ids(trace: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for step in trace.get("selection_steps") or []:
        if not isinstance(step, dict):
            continue
        req = step.get("covered_requirement_id")
        if req and req not in ids:
            ids.append(str(req))
    return ids


def has_dependency_binding_signal(selector_trace: Dict[str, Any]) -> bool:
    binding_candidates = selector_trace.get("binding_candidates_by_requirement") or {}
    if any(bool(v) for v in binding_candidates.values()):
        return True
    for step in selector_trace.get("selection_steps") or []:
        if isinstance(step, dict) and step.get("binding_title"):
            return True
    return False


def classify_case(row: Dict[str, Any]) -> str:
    base_f1 = as_float(row.get("baseline_f1"))
    sel_f1 = as_float(row.get("selector_f1"))
    if sel_f1 > base_f1 + EPS:
        return "win"
    if sel_f1 < base_f1 - EPS:
        base_complete = bool(row["baseline_gold_complete"])
        selector_complete = bool(row["selector_gold_complete"])
        if base_complete and not selector_complete:
            return "loss_gold_pushed_out"
        if selector_complete:
            return "loss_reader_interference"
        if bool(row.get("binding_signal")) and bool(row.get("new_non_gold_titles")):
            return "loss_binding_or_wrong_entity_candidate"
        return "loss_other_incomplete"
    return "tie"


def iter_report_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        else:
            paths.append(path)
    return [path for path in paths if path.exists()]


def analyze_report(path: Path) -> Dict[str, Any]:
    payload = json.load(path.open())
    traces = payload.get("setwise_selector_query_traces") or []
    rows: List[Dict[str, Any]] = []
    for idx, trace in enumerate(traces):
        selector_trace = trace.get("selector_trace") or {}
        gold_titles = list(trace.get("gold_titles") or [])
        baseline_titles = list(trace.get("baseline_top_titles") or [])
        selector_titles = list(trace.get("selector_top_titles") or [])

        baseline_set = title_set(baseline_titles)
        selector_set = title_set(selector_titles)
        gold_set = title_set(gold_titles)
        new_titles = [title for title in selector_titles if norm_title(title) not in baseline_set]
        removed_titles = [title for title in baseline_titles if norm_title(title) not in selector_set]
        new_non_gold = [title for title in new_titles if norm_title(title) not in gold_set]
        removed_gold = [title for title in removed_titles if norm_title(title) in gold_set]

        row = {
            "query_idx": idx,
            "question": trace.get("question"),
            "gold_answers": trace.get("gold_answers") or [],
            "gold_titles": gold_titles,
            "baseline_answer": trace.get("baseline_answer"),
            "selector_answer": trace.get("selector_answer"),
            "baseline_f1": as_float((trace.get("baseline_metrics") or {}).get("F1")),
            "selector_f1": as_float((trace.get("selector_metrics") or {}).get("F1")),
            "baseline_em": as_float((trace.get("baseline_metrics") or {}).get("ExactMatch")),
            "selector_em": as_float((trace.get("selector_metrics") or {}).get("ExactMatch")),
            "baseline_top_titles": baseline_titles,
            "selector_top_titles": selector_titles,
            "baseline_gold_complete": covers_gold(baseline_titles, gold_titles),
            "selector_gold_complete": covers_gold(selector_titles, gold_titles),
            "changed_from_baseline": bool(trace.get("changed_from_baseline")),
            "new_titles": new_titles,
            "removed_titles": removed_titles,
            "new_non_gold_titles": new_non_gold,
            "removed_gold_titles": removed_gold,
            "binding_signal": has_dependency_binding_signal(selector_trace),
            "step_requirement_ids": get_step_requirement_ids(selector_trace),
            "selected_pool_positions": selector_trace.get("selected_pool_positions") or [],
            "requirements": selector_trace.get("requirements") or [],
            "repairable_by_requirement": selector_trace.get("repairable_by_requirement") or {},
            "binding_candidates_by_requirement": selector_trace.get("binding_candidates_by_requirement") or {},
        }
        row["delta_f1"] = row["selector_f1"] - row["baseline_f1"]
        row["delta_em"] = row["selector_em"] - row["baseline_em"]
        row["category"] = classify_case(row)
        rows.append(row)

    category_counts = Counter(row["category"] for row in rows)
    changed_rows = [row for row in rows if row["changed_from_baseline"]]
    loss_rows = [row for row in rows if row["category"].startswith("loss")]
    win_rows = [row for row in rows if row["category"] == "win"]
    summary = {
        "report": str(path),
        "dataset": payload.get("dataset"),
        "num_traces": len(rows),
        "category_counts": dict(category_counts),
        "changed_count": len(changed_rows),
        "changed_rate": len(changed_rows) / max(len(rows), 1),
        "win_count": len(win_rows),
        "loss_count": len(loss_rows),
        "tie_count": category_counts.get("tie", 0),
        "mean_delta_f1": sum(row["delta_f1"] for row in rows) / max(len(rows), 1),
        "mean_delta_f1_changed": sum(row["delta_f1"] for row in changed_rows) / max(len(changed_rows), 1),
        "loss_binding_signal_count": sum(1 for row in loss_rows if row["binding_signal"]),
        "loss_new_non_gold_count": sum(1 for row in loss_rows if row["new_non_gold_titles"]),
    }
    return {"summary": summary, "cases": rows}


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Evaluation report JSON files or directories.")
    parser.add_argument("--output_dir", type=Path, default=Path("run_logs/failure_taxonomy_20260424"))
    parser.add_argument("--json", action="store_true", help="Print JSON summary instead of markdown.")
    args = parser.parse_args()

    outputs = []
    for path in iter_report_paths(args.reports):
        result = analyze_report(path)
        summary = result["summary"]
        cases = result["cases"]
        stem = path.stem
        write_jsonl(args.output_dir / f"{stem}.cases.jsonl", cases)
        write_jsonl(
            args.output_dir / f"{stem}.loss_cases.jsonl",
            [row for row in cases if str(row.get("category", "")).startswith("loss")],
        )
        (args.output_dir / f"{stem}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        outputs.append(summary)

    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return

    headers = [
        "Dataset",
        "N",
        "Changed",
        "Wins",
        "Losses",
        "Ties",
        "Mean dF1",
        "Loss: gold pushed",
        "Loss: reader noise",
        "Loss: binding cand",
        "Loss: other",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for summary in outputs:
        counts = summary.get("category_counts") or {}
        values = [
            str(summary.get("dataset")),
            str(summary.get("num_traces")),
            f"{summary.get('changed_count', 0)} ({summary.get('changed_rate', 0.0):.1%})",
            str(summary.get("win_count", 0)),
            str(summary.get("loss_count", 0)),
            str(summary.get("tie_count", 0)),
            f"{summary.get('mean_delta_f1', 0.0):+.4f}",
            str(counts.get("loss_gold_pushed_out", 0)),
            str(counts.get("loss_reader_interference", 0)),
            str(counts.get("loss_binding_or_wrong_entity_candidate", 0)),
            str(counts.get("loss_other_incomplete", 0)),
        ]
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
