from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_rankgpt_selection_to_pool import load_selector_rows, parse_positions, run  # noqa: E402


def test_parse_positions_deduplicates_and_bounds() -> None:
    row = {"selected_positions_json": "[2, 0, 2, 99, -1]"}

    assert parse_positions(row, pool_size=3) == [2, 0]


def test_run_reorders_pool_by_selector_rows(tmp_path: Path) -> None:
    pool_json = tmp_path / "pool.json"
    selector_csv = tmp_path / "selector.csv"
    output_pool = tmp_path / "out.json"
    pool_json.write_text(
        json.dumps(
            {
                "source": "test_pool",
                "records": [
                    {
                        "query_idx": 7,
                        "pool_docs": ["d0", "d1", "d2"],
                        "pool_titles": ["t0", "t1", "t2"],
                        "pool_doc_scores": [0.3, 0.2, 0.1],
                        "pool_doc_ids": ["0", "1", "2"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with selector_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_index", "method", "selected_positions_json"])
        writer.writeheader()
        writer.writerow(
            {
                "query_index": 0,
                "method": "rankgpt_sliding20_step10_no_think",
                "selected_positions_json": "[2, 0]",
            }
        )

    result = run(
        type(
            "Args",
            (),
            {
                "pool_json": str(pool_json),
                "selector_rows_csv": str(selector_csv),
                "method": "rankgpt_sliding20_step10_no_think",
                "output_pool_json": str(output_pool),
                "trace_json": "",
                "limit": 0,
                "dataset": "",
            },
        )()
    )

    output = json.loads(output_pool.read_text(encoding="utf-8"))
    record = output["records"][0]
    assert result["parse_success_count"] == 1
    assert record["pool_titles"] == ["t2", "t0", "t1"]
    assert record["rankgpt_selection_trace"]["selected_positions"] == [2, 0, 1]


def test_load_selector_rows_filters_dataset_before_query_index(tmp_path: Path) -> None:
    selector_csv = tmp_path / "selector.csv"
    with selector_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "base_dataset", "query_index", "method", "selected_positions_json"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "2Wiki",
                "base_dataset": "2wikimultihopqa",
                "query_index": 0,
                "method": "rankgpt_sliding20_step10_no_think",
                "selected_positions_json": "[2, 0]",
            }
        )
        writer.writerow(
            {
                "dataset": "MuSiQue",
                "base_dataset": "musique",
                "query_index": 0,
                "method": "rankgpt_sliding20_step10_no_think",
                "selected_positions_json": "[1]",
            }
        )

    rows = load_selector_rows(
        selector_csv,
        method="rankgpt_sliding20_step10_no_think",
        dataset="2wikimultihopqa",
    )

    assert parse_positions(rows[0], pool_size=3) == [2, 0]
