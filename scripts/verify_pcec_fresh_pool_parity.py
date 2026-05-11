#!/usr/bin/env python3
"""Verify frozen ETv3 fresh expansion against historical ETv3 pool JSONs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evidence_transition_graphragv4_composition.expander import (  # noqa: E402
    DEFAULT_FROZEN_ETV3_RUNS_ROOT,
    FrozenETv3Expander,
)


DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DEFAULT_HISTORICAL_POOL_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/pools")
DEFAULT_OUTPUT_JSON = Path("run_logs/pcec_fresh_pool_parity/fresh_pool_parity.json")
DEFAULT_OUTPUT_MD = Path("run_logs/pcec_fresh_pool_parity/fresh_pool_parity.md")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_csv(value: str | Sequence[str]) -> list[str]:
    items = value.split(",") if isinstance(value, str) else list(value)
    return [str(item).strip() for item in items if str(item).strip()]


def historical_pool_path(root: Path, dataset: str, pool_k: int, limit: int) -> Path:
    return Path(root) / f"{dataset}_etv3_pool{pool_k}_limit{limit}.json"


def float_lists_match(left: Sequence[Any], right: Sequence[Any], *, eps: float) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        try:
            if math.fabs(float(a) - float(b)) > float(eps):
                return False
        except (TypeError, ValueError):
            return False
    return True


def compare_record(
    fresh: Mapping[str, Any],
    historical: Mapping[str, Any],
    *,
    pool_k: int,
    score_eps: float,
) -> dict[str, Any]:
    fresh_docs = list(fresh.get("pool_docs") or [])[: int(pool_k)]
    hist_docs = list(historical.get("pool_docs") or [])[: int(pool_k)]
    fresh_titles = list(fresh.get("pool_titles") or [])[: int(pool_k)]
    hist_titles = list(historical.get("pool_titles") or [])[: int(pool_k)]
    fresh_ids = list(fresh.get("pool_doc_ids") or [])[: int(pool_k)]
    hist_ids = list(historical.get("pool_doc_ids") or [])[: int(pool_k)]
    fresh_scores = list(fresh.get("pool_doc_scores") or [])[: int(pool_k)]
    hist_scores = list(historical.get("pool_doc_scores") or [])[: int(pool_k)]
    return {
        "query_idx": int(fresh.get("query_idx", historical.get("query_idx", -1)) or 0),
        "question_match": str(fresh.get("question") or "") == str(historical.get("question") or ""),
        "doc_text_exact": fresh_docs == hist_docs,
        "title_exact": fresh_titles == hist_titles,
        "external_doc_id_exact": fresh_ids == hist_ids,
        "score_exact": float_lists_match(fresh_scores, hist_scores, eps=score_eps),
        "baseline_top5_title_exact": fresh_titles[:5] == hist_titles[:5],
        "baseline_top5_doc_text_exact": fresh_docs[:5] == hist_docs[:5],
    }


def summarize(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(comparisons)
    fields = [
        "question_match",
        "doc_text_exact",
        "title_exact",
        "external_doc_id_exact",
        "score_exact",
        "baseline_top5_title_exact",
        "baseline_top5_doc_text_exact",
    ]
    summary = {"count": int(count)}
    for field in fields:
        matches = sum(bool(row.get(field)) for row in comparisons)
        summary[f"{field}_count"] = int(matches)
        summary[f"{field}_rate"] = round(float(matches) / float(max(count, 1)), 6)
    summary["all_exact_count"] = int(
        sum(
            all(bool(row.get(field)) for field in fields)
            for row in comparisons
        )
    )
    summary["all_exact_rate"] = round(float(summary["all_exact_count"]) / float(max(count, 1)), 6)
    return summary


def run_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    started = time.monotonic()
    print(f"[START] dataset={dataset} fresh_pool_parity", flush=True)
    historical_path = historical_pool_path(
        Path(args.historical_pool_root),
        dataset,
        int(args.pool_k),
        int(args.limit),
    )
    historical_payload = read_json(historical_path)
    historical_records = list(historical_payload.get("records") or [])
    limit = len(historical_records)
    if int(args.max_queries) > 0:
        limit = min(limit, int(args.max_queries))
    expander = FrozenETv3Expander(
        dataset=dataset,
        candidate_pool_k=int(args.pool_k),
        reader_budget_k=int(args.reader_budget_k),
        et_candidate_pool_k=int(args.et_candidate_pool_k),
        data_root=Path(args.data_root),
        run_root=Path(args.frozen_runs_root) / dataset,
        llm_name=str(args.llm_name),
        embedding_name=str(args.embedding_name),
        embedding_base_url=str(args.embedding_base_url),
    )
    print(
        f"[READY] dataset={dataset} limit={limit} runner={expander.runner} "
        f"openie={expander.metadata().get('openie_path')}",
        flush=True,
    )
    expander.prepare_queries(max_queries=limit)
    fresh_records: list[dict[str, Any]] = []
    progress_every = max(int(args.progress_every), 0)
    for idx in range(limit):
        if progress_every and (idx == 0 or idx % progress_every == 0):
            elapsed = time.monotonic() - started
            print(f"[PROGRESS] dataset={dataset} query={idx}/{limit} elapsed_s={elapsed:.1f}", flush=True)
        fresh_records.append(expander.expand_query_record(idx))
    if progress_every:
        elapsed = time.monotonic() - started
        print(f"[PROGRESS] dataset={dataset} query={limit}/{limit} elapsed_s={elapsed:.1f}", flush=True)
    comparisons = [
        compare_record(
            fresh_records[idx],
            historical_records[idx],
            pool_k=int(args.pool_k),
            score_eps=float(args.score_eps),
        )
        for idx in range(limit)
    ]
    mismatches = [
        {
            "query_idx": int(row.get("query_idx", idx)),
            "question": fresh_records[idx].get("question") or historical_records[idx].get("question") or "",
            "fresh_top5": list(fresh_records[idx].get("pool_titles") or [])[:5],
            "historical_top5": list(historical_records[idx].get("pool_titles") or [])[:5],
            "mismatch_fields": [
                field
                for field in (
                    "question_match",
                    "doc_text_exact",
                    "title_exact",
                    "external_doc_id_exact",
                    "score_exact",
                    "baseline_top5_title_exact",
                    "baseline_top5_doc_text_exact",
                )
                if not bool(row.get(field))
            ],
        }
        for idx, row in enumerate(comparisons)
        if not all(
            bool(row.get(field))
            for field in (
                "question_match",
                "doc_text_exact",
                "title_exact",
                "external_doc_id_exact",
                "score_exact",
                "baseline_top5_title_exact",
                "baseline_top5_doc_text_exact",
            )
        )
    ][: int(args.max_examples)]
    if args.fresh_pool_output_root:
        fresh_pool_path = Path(args.fresh_pool_output_root) / f"{dataset}_fresh_frozen_etv3_pool{args.pool_k}_limit{limit}.json"
        write_json(
            {
                "dataset": dataset,
                "limit": int(limit),
                "pool_k": int(args.pool_k),
                "source": "pcec_fresh_frozen_etv3_expander",
                "historical_pool_json": str(historical_path),
                "expander": expander.metadata(),
                "records": fresh_records,
            },
            fresh_pool_path,
        )
    else:
        fresh_pool_path = None
    result = {
        "dataset": dataset,
        "historical_pool_json": str(historical_path),
        "fresh_pool_json": str(fresh_pool_path) if fresh_pool_path else "",
        "expander": expander.metadata(),
        "summary": summarize(comparisons),
        "mismatch_examples": mismatches,
    }
    summary = result["summary"]
    elapsed = time.monotonic() - started
    print(
        "[DONE] dataset={dataset} all_exact={all_exact}/{count} top5={top5}/{count} elapsed_s={elapsed:.1f}".format(
            dataset=dataset,
            all_exact=int(summary.get("all_exact_count", 0)),
            top5=int(summary.get("baseline_top5_title_exact_count", 0)),
            count=int(summary.get("count", 0)),
            elapsed=elapsed,
        ),
        flush=True,
    )
    return result


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# PCEC Fresh Frozen ETv3 Pool Parity",
        "",
        "| Dataset | Count | All exact | Doc text | Doc IDs | Scores | Top5 titles |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload.get("datasets", []) or []:
        summary = result.get("summary", {}) or {}
        lines.append(
            "| {dataset} | {count} | {all_exact} | {doc} | {ids} | {scores} | {top5} |".format(
                dataset=result.get("dataset", ""),
                count=int(summary.get("count", 0)),
                all_exact=int(summary.get("all_exact_count", 0)),
                doc=int(summary.get("doc_text_exact_count", 0)),
                ids=int(summary.get("external_doc_id_exact_count", 0)),
                scores=int(summary.get("score_exact_count", 0)),
                top5=int(summary.get("baseline_top5_title_exact_count", 0)),
            )
        )
    for result in payload.get("datasets", []) or []:
        examples = list(result.get("mismatch_examples") or [])
        if not examples:
            continue
        lines.extend(["", f"## {result.get('dataset')} Mismatches", ""])
        for item in examples:
            lines.append(f"- `{item.get('query_idx')}` fields={item.get('mismatch_fields')}")
            lines.append(f"  - Fresh: {item.get('fresh_top5')}")
            lines.append(f"  - Historical: {item.get('historical_top5')}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--reader-budget-k", type=int, default=5)
    parser.add_argument("--et-candidate-pool-k", type=int, default=200)
    parser.add_argument("--score-eps", type=float, default=1e-6)
    parser.add_argument("--historical-pool-root", type=Path, default=DEFAULT_HISTORICAL_POOL_ROOT)
    parser.add_argument("--frozen-runs-root", type=Path, default=DEFAULT_FROZEN_ETV3_RUNS_ROOT)
    parser.add_argument("--fresh-pool-output-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("reproduce/dataset"))
    parser.add_argument("--llm-name", default="qwen3-8b-train")
    parser.add_argument("--embedding-name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    datasets = parse_csv(args.datasets)
    if int(args.jobs) > 1 and len(datasets) > 1:
        results_by_dataset: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=min(int(args.jobs), len(datasets))) as executor:
            future_to_dataset = {
                executor.submit(run_dataset, args, dataset): dataset
                for dataset in datasets
            }
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                results_by_dataset[dataset] = future.result()
        results = [results_by_dataset[dataset] for dataset in datasets]
    else:
        results = [run_dataset(args, dataset) for dataset in datasets]
    output = {
        "datasets": results,
        "config": {
            "limit": int(args.limit),
            "max_queries": int(args.max_queries),
            "pool_k": int(args.pool_k),
            "reader_budget_k": int(args.reader_budget_k),
            "et_candidate_pool_k": int(args.et_candidate_pool_k),
            "score_eps": float(args.score_eps),
        },
    }
    write_json(output, args.output_json)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(output), encoding="utf-8")
    print(
        json.dumps(
            {
                result["dataset"]: {
                    "all_exact": result["summary"]["all_exact_count"],
                    "count": result["summary"]["count"],
                    "top5": result["summary"]["baseline_top5_title_exact_count"],
                }
                for result in results
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
