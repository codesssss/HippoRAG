#!/usr/bin/env python3
"""Repair PCEC fresh reports so reader QA uses external corpus doc ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DEFAULT_REPORT_ROOT = Path("run_logs/pcec_fresh_e2e/evals")
DEFAULT_FRESH_POOL_ROOT = Path("run_logs/pcec_fresh_pool_parity/fresh_pools")
DEFAULT_OUTPUT_ROOT = Path("run_logs/pcec_fresh_e2e_reader_qa/retrieval_reports_reader_docids")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def report_path(root: Path, dataset: str) -> Path:
    return root / f"{dataset}_pcec_fresh_e2e_prefix4_residual1_pool100_limit1000.json"


def pool_path(root: Path, dataset: str) -> Path:
    return root / f"{dataset}_fresh_frozen_etv3_pool100_limit1000.json"


def repair_report(*, dataset: str, report_root: Path, fresh_pool_root: Path, output_root: Path) -> dict[str, Any]:
    source_report = report_path(report_root, dataset)
    source_pool = pool_path(fresh_pool_root, dataset)
    payload = read_json(source_report)
    pool_payload = read_json(source_pool)
    records = {int(record["query_idx"]): record for record in list(pool_payload.get("records") or [])}
    changed = 0
    for row in list(payload.get("rows") or []):
        query_idx = int(row["query_index"])
        record = records[query_idx]
        external_doc_ids = list(record.get("pool_doc_ids") or [])
        positions = list((row.get("pcec_readout") or {}).get("final_positions") or [])
        repaired_ids = [
            int(external_doc_ids[int(position)])
            for position in positions
            if int(position) < len(external_doc_ids)
        ]
        if list(row.get("retrieved_doc_indices_top5") or []) != repaired_ids:
            changed += 1
        row["pcec_dbec_doc_indices_top5"] = list(row.get("retrieved_doc_indices_top5") or [])
        row["retrieved_doc_indices_top5"] = repaired_ids
        row["reader_doc_id_source"] = "fresh_frozen_etv3_external_pool_doc_ids"
    payload["reader_doc_id_source"] = "fresh_frozen_etv3_external_pool_doc_ids"
    payload["reader_doc_id_repair"] = {
        "changed_rows": int(changed),
        "fresh_pool_json": str(source_pool),
        "source_report": str(source_report),
    }
    output_path = report_path(output_root, dataset)
    write_json(output_path, payload)
    return {
        "dataset": dataset,
        "changed_rows": int(changed),
        "output_json": str(output_path),
    }


def parse_datasets(value: str | Sequence[str]) -> list[str]:
    items = value.split(",") if isinstance(value, str) else list(value)
    return [str(item).strip() for item in items if str(item).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--fresh-pool-root", type=Path, default=DEFAULT_FRESH_POOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    summaries = [
        repair_report(
            dataset=dataset,
            report_root=Path(args.report_root),
            fresh_pool_root=Path(args.fresh_pool_root),
            output_root=Path(args.output_root),
        )
        for dataset in parse_datasets(args.datasets)
    ]
    print(json.dumps(summaries, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
