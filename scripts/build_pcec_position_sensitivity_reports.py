#!/usr/bin/env python3
"""Build reader reports that move PCEC's admitted residual doc to each slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DEFAULT_SOURCE_ROOT = Path("run_logs/pcec_fresh_e2e_reader_qa/retrieval_reports_reader_docids")
DEFAULT_OUTPUT_ROOT = Path("reports/pcec_m4_position_sensitivity_8b_20260511/retrieval_reports")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_datasets(value: str | Sequence[str]) -> list[str]:
    items = value.split(",") if isinstance(value, str) else list(value)
    return [str(item).strip() for item in items if str(item).strip()]


def source_report_path(root: Path, dataset: str, *, limit: int) -> Path:
    return root / f"{dataset}_pcec_fresh_e2e_prefix4_residual1_pool100_limit{limit}.json"


def output_report_path(root: Path, dataset: str, *, slot: int, limit: int) -> Path:
    return root / f"slot{slot}" / f"{dataset}_pcec_m4_admitted_slot{slot}_pool100_limit{limit}.json"


def move_item(values: Sequence[Any], source_idx: int, target_idx: int) -> list[Any]:
    items = list(values)
    if source_idx < 0 or source_idx >= len(items):
        return items
    item = items.pop(source_idx)
    target = max(0, min(int(target_idx), len(items)))
    items.insert(target, item)
    return items


def admitted_final_index(row: Mapping[str, Any]) -> int | None:
    readout = row.get("pcec_readout") or {}
    admitted = list(readout.get("admitted_positions") or [])
    final = list(readout.get("final_positions") or [])
    if not admitted or not final:
        return None
    admitted_pos = int(admitted[0])
    for idx, pos in enumerate(final):
        if int(pos) == admitted_pos:
            return idx
    return None


def reorder_row(row: Mapping[str, Any], *, slot: int) -> tuple[dict[str, Any], bool]:
    output = dict(row)
    source_idx = admitted_final_index(row)
    if source_idx is None:
        output["pcec_position_probe"] = {
            "target_slot": int(slot),
            "applied": False,
            "reason": "no_admitted_residual",
        }
        return output, False
    target_idx = int(slot) - 1
    titles = list(row.get("retrieved_titles_top5") or [])
    doc_ids = list(row.get("retrieved_doc_indices_top5") or [])
    dbec_ids = list(row.get("pcec_dbec_doc_indices_top5") or [])
    output["retrieved_titles_top5"] = move_item(titles, source_idx, target_idx)
    output["retrieved_doc_indices_top5"] = move_item(doc_ids, source_idx, target_idx)
    if dbec_ids:
        output["pcec_dbec_doc_indices_top5"] = move_item(dbec_ids, source_idx, target_idx)
    output["pcec_position_probe"] = {
        "target_slot": int(slot),
        "applied": True,
        "source_slot": int(source_idx) + 1,
        "admitted_title": titles[source_idx] if source_idx < len(titles) else "",
    }
    return output, source_idx != target_idx


def build_dataset_reports(*, dataset: str, source_root: Path, output_root: Path, limit: int, slots: Sequence[int]) -> list[dict[str, Any]]:
    source_path = source_report_path(source_root, dataset, limit=limit)
    payload = read_json(source_path)
    rows = list(payload.get("rows") or [])[:limit]
    summaries: list[dict[str, Any]] = []
    for slot in slots:
        changed = 0
        applied = 0
        new_payload = dict(payload)
        new_rows = []
        for row in rows:
            reordered, row_changed = reorder_row(row, slot=int(slot))
            if bool((reordered.get("pcec_position_probe") or {}).get("applied")):
                applied += 1
            if row_changed:
                changed += 1
            new_rows.append(reordered)
        new_payload["rows"] = new_rows
        new_payload["mode"] = "pcec_m4_admitted_position_sensitivity"
        new_payload["position_sensitivity"] = {
            "source_report": str(source_path),
            "target_slot": int(slot),
            "changed_rows": int(changed),
            "admitted_rows": int(applied),
            "description": "Move the admitted residual document to the target reader slot while preserving the selected set.",
        }
        out_path = output_report_path(output_root, dataset, slot=int(slot), limit=limit)
        write_json(out_path, new_payload)
        summaries.append(
            {
                "dataset": dataset,
                "slot": int(slot),
                "output_json": str(out_path),
                "admitted_rows": int(applied),
                "changed_rows": int(changed),
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--slots", default="1,2,3,4,5")
    args = parser.parse_args()
    slots = [int(item) for item in parse_datasets(args.slots)]
    summaries = []
    for dataset in parse_datasets(args.datasets):
        summaries.extend(
            build_dataset_reports(
                dataset=dataset,
                source_root=Path(args.source_root),
                output_root=Path(args.output_root),
                limit=int(args.limit),
                slots=slots,
            )
        )
    print(json.dumps(summaries, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
