#!/usr/bin/env python3
"""Run RankGPT-style fixed-pool reranking baselines.

This script evaluates selector-level support metrics for RankGPT-style listwise
reranking over the existing PropRAG pool100.  It does not retrieve new
documents and does not run the answer reader.

Two no-thinking variants are supported:

* ``rank5_no_think``: ask the model to rank the top 5 passages by relevance.
* ``select5_no_think``: ask the model to directly select 5 support passages.

Both prompts include ``/no_think`` and require strict JSON output with pool
positions.  The output order is filled with the original source order after the
LLM-selected top positions so support@5 metrics can be compared to DBEC and
SetR on the same fixed pool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import (  # noqa: E402
    DBEC_SELECTIVE_DIR,
    SETR_FAITHFUL_DIR,
    SOURCE_POOL_DIR,
    TOP_K,
    extract_dbec_final_order,
    extract_setr_seed_positions,
    normalize_title,
    read_json,
    safe_float,
    safe_int,
    support_match,
    write_csv,
    write_json,
)
from probe_chain_walking_binding import (  # noqa: E402
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    JsonlCache,
    chat_completion_raw,
    parse_json_payload,
    probe_llm,
    strip_think_blocks,
    write_jsonl,
)
from probe_fixed_pool_candidate_generation import (  # noqa: E402
    prompt_has_no_think,
    split_base_urls,
    stable_hash,
)


REPORT_DIR = Path("reports/rankgpt_fixed_pool_baseline_20260508")
PROMPT_VERSION = "rankgpt_fixed_pool_v1_no_think"
DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DATASET_LABELS = {
    "2wikimultihopqa": "2Wiki",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
}
VARIANTS = ("rank5_no_think", "select5_no_think")


def write_csv_lf(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def source_pool_path(dataset: str) -> Path:
    return SOURCE_POOL_DIR / f"{dataset}_pool100.json"


def setr_pool_path(dataset: str) -> Path:
    return SETR_FAITHFUL_DIR / f"{dataset}_proprag_setr_k20_doc768_faithful.selected_pool.json"


def dbec_report_path(dataset: str) -> Path:
    return DBEC_SELECTIVE_DIR / f"{dataset}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def fill_source_order(prefix: Sequence[int], *, pool_size: int) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for raw_pos in prefix:
        pos = int(raw_pos)
        if 0 <= pos < pool_size and pos not in seen:
            output.append(pos)
            seen.add(pos)
    for pos in range(pool_size):
        if pos not in seen:
            output.append(pos)
            seen.add(pos)
    return output


def pool_doc_list(record: Mapping[str, Any], *, max_doc_chars: int, one_based_ids: bool = True) -> str:
    titles = list(record.get("pool_titles") or [])
    docs = list(record.get("pool_docs") or [])
    lines: list[str] = []
    for pos, title in enumerate(titles):
        display_id = pos + 1 if one_based_ids else pos
        doc = str(docs[pos]) if pos < len(docs) else str(title)
        snippet = re.sub(r"\s+", " ", doc).strip()[: int(max_doc_chars)]
        lines.append(f"[{display_id}] {title}\n{snippet}")
    return "\n\n".join(lines)


def build_messages(
    *,
    variant: str,
    question: str,
    record: Mapping[str, Any],
    max_doc_chars: int,
    one_based_ids: bool = True,
) -> list[dict[str, str]]:
    docs = pool_doc_list(record, max_doc_chars=max_doc_chars, one_based_ids=one_based_ids)
    id_note = (
        "Passage ids are 1-based: valid ids are integers from 1 to 100."
        if one_based_ids
        else "Passage ids are 0-based: valid ids are integers from 0 to 99."
    )
    if variant == "rank5_no_think":
        task = (
            "You are RankGPT, an assistant that ranks passages by relevance to a query.\n"
            "I will provide a query and 100 candidate passages from a fixed retrieval pool.\n"
            "Rank the five passages that should be most useful for answering the query.\n"
            f"{id_note} Use only passage ids that appear below. Do not invent ids. Do not explain.\n"
            "Return strict JSON only: {\"ranking\": [integer, integer, integer, integer, integer]}."
        )
    elif variant == "select5_no_think":
        task = (
            "Select exactly five support passages from the fixed candidate pool for answering the multi-hop query.\n"
            "Prefer a set that jointly covers bridge evidence and final-answer evidence, not just lexical relevance.\n"
            f"{id_note} Use only passage ids that appear below. Do not invent ids. Do not explain.\n"
            "Return strict JSON only: {\"doc_ids\": [integer, integer, integer, integer, integer]}."
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    user = (
        "/no_think\n"
        f"{task}\n\n"
        f"Query: {question}\n\n"
        f"Candidate passages:\n{docs}\n"
    )
    return [
        {"role": "system", "content": "You are a listwise passage reranker. Output JSON only."},
        {"role": "user", "content": user},
    ]


def extract_positions_from_raw(
    raw: Any,
    *,
    pool_size: int,
    top_k: int = TOP_K,
    one_based_ids: bool = True,
) -> tuple[list[int], str]:
    parsed = parse_json_payload(raw)
    raw_ids: list[Any] = []
    source = "fallback_regex"
    if isinstance(parsed, Mapping):
        raw_ids = list(parsed.get("ranking") or parsed.get("doc_ids") or parsed.get("ids") or [])
        source = "json"
    elif isinstance(parsed, list):
        raw_ids = list(parsed)
        source = "json_list"
    else:
        raw_ids = re.findall(r"\b\d{1,3}\b", strip_think_blocks(raw))

    output: list[int] = []
    seen: set[int] = set()
    for item in raw_ids:
        if isinstance(item, Mapping):
            item = item.get("doc_id") or item.get("id") or item.get("position")
        raw_pos = safe_int(item, default=-1)
        pos = raw_pos - 1 if one_based_ids else raw_pos
        if 0 <= pos < int(pool_size) and pos not in seen:
            output.append(pos)
            seen.add(pos)
        if len(output) >= int(top_k):
            break
    return output, source


def selected_titles(order: Sequence[int], record: Mapping[str, Any], *, top_k: int = TOP_K) -> list[str]:
    titles = list(record.get("pool_titles") or [])
    return [str(titles[pos]) for pos in list(order)[: int(top_k)] if 0 <= int(pos) < len(titles)]


def support_metrics_for_order(order: Sequence[int], record: Mapping[str, Any], *, top_k: int = TOP_K) -> dict[str, Any]:
    return support_match(record.get("gold_titles") or [], selected_titles(order, record, top_k=top_k))


def rank_of_title(title: Any, order: Sequence[int], pool_titles: Sequence[Any]) -> int | None:
    norm = normalize_title(title)
    if not norm:
        return None
    for rank, pos in enumerate(order):
        if 0 <= int(pos) < len(pool_titles) and normalize_title(pool_titles[int(pos)]) == norm:
            return rank
    return None


def baseline_orders(
    *,
    record: Mapping[str, Any],
    setr_record: Mapping[str, Any],
    dbec_trace: Mapping[str, Any],
) -> dict[str, list[int]]:
    pool_size = len(record.get("pool_titles") or [])
    selector_trace = dbec_trace.get("selector_trace")
    selector_trace = selector_trace if isinstance(selector_trace, Mapping) else {}
    source = list(range(pool_size))
    setr = fill_source_order(extract_setr_seed_positions(setr_record, pool_size), pool_size=pool_size)
    dbec = fill_source_order(extract_dbec_final_order(selector_trace, pool_size), pool_size=pool_size)
    return {
        "source_order": source,
        "setr_faithful": setr,
        "dbec_selective": dbec,
    }


def build_tasks(
    *,
    datasets: Sequence[str],
    limit: int,
    max_doc_chars: int,
    llm_model: str,
    variants: Sequence[str],
    one_based_ids: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    tasks: list[dict[str, Any]] = []
    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        records = list(read_json(source_pool_path(dataset)).get("records") or [])
        if int(limit) > 0:
            records = records[: int(limit)]
        records_by_dataset[dataset] = records
        for query_index, record in enumerate(records):
            question = str(record.get("question") or "")
            for variant in variants:
                messages = build_messages(
                    variant=variant,
                    question=question,
                    record=record,
                    max_doc_chars=max_doc_chars,
                    one_based_ids=one_based_ids,
                )
                key = stable_hash(
                    {
                        "prompt_version": PROMPT_VERSION,
                        "dataset": dataset,
                        "query_index": query_index,
                        "query_idx": record.get("query_idx"),
                        "question": question,
                        "variant": variant,
                        "pool_title_hash": stable_hash(record.get("pool_titles") or []),
                        "llm_model": llm_model,
                        "max_doc_chars": max_doc_chars,
                        "one_based_ids": one_based_ids,
                    }
                )
                tasks.append(
                    {
                        "key": key,
                        "dataset": dataset,
                        "query_index": query_index,
                        "variant": variant,
                        "messages": messages,
                    }
                )
    return tasks, records_by_dataset


def run_llm_tasks(
    *,
    tasks: Sequence[Mapping[str, Any]],
    cache: JsonlCache,
    run_llm: bool,
    base_urls: Sequence[str],
    model: str,
    max_tokens: int,
    timeout: int,
    workers: int,
) -> list[dict[str, Any]]:
    lock = threading.Lock()

    def one(task_index: int, task: Mapping[str, Any]) -> dict[str, Any]:
        key = str(task["key"])
        with lock:
            raw = cache.get(key)
        cache_hit = raw is not None
        endpoint = ""
        error = ""
        if raw is None and run_llm:
            for attempt in range(max(1, len(base_urls))):
                endpoint = base_urls[(task_index + attempt) % len(base_urls)]
                try:
                    raw = chat_completion_raw(
                        base_url=endpoint,
                        model=model,
                        messages=list(task["messages"]),
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    with lock:
                        cache.set(
                            key,
                            raw,
                            {
                                "dataset": task.get("dataset"),
                                "query_index": task.get("query_index"),
                                "variant": task.get("variant"),
                                "endpoint": endpoint,
                            },
                        )
                    break
                except Exception as exc:  # pragma: no cover - integration behavior
                    error = str(exc)
                    raw = None
        if raw is None:
            raw = ""
        parsed = parse_json_payload(raw)
        return {
            "key": key,
            "dataset": str(task.get("dataset") or ""),
            "dataset_label": DATASET_LABELS.get(str(task.get("dataset") or ""), str(task.get("dataset") or "")),
            "query_index": safe_int(task.get("query_index")),
            "variant": str(task.get("variant") or ""),
            "endpoint": endpoint,
            "cache_hit": int(cache_hit),
            "run_llm": int(run_llm),
            "prompt_has_no_think": int(prompt_has_no_think(task["messages"])),
            "parse_ok": int(parsed is not None),
            "raw": raw,
            "error": error,
        }

    if not run_llm:
        return [one(idx, task) for idx, task in enumerate(tasks)]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(one, idx, task) for idx, task in enumerate(tasks)]
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: (row["dataset"], safe_int(row["query_index"]), row["variant"]))


def prompt_rows_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    return {
        (str(row.get("dataset")), safe_int(row.get("query_index")), str(row.get("variant"))): row
        for row in rows
    }


def build_eval_rows(
    *,
    datasets: Sequence[str],
    variants: Sequence[str],
    prompt_rows: Sequence[Mapping[str, Any]],
    limit: int,
    one_based_ids: bool,
) -> list[dict[str, Any]]:
    prompt_by_key = prompt_rows_by_key(prompt_rows)
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        records = list(read_json(source_pool_path(dataset)).get("records") or [])
        setr_records = list(read_json(setr_pool_path(dataset)).get("records") or [])
        dbec_traces = list(read_json(dbec_report_path(dataset)).get("setwise_selector_query_traces") or [])
        if int(limit) > 0:
            records = records[: int(limit)]
        for query_index, record in enumerate(records):
            setr_record = setr_records[query_index]
            dbec_trace = dbec_traces[query_index]
            gold_titles = list(record.get("gold_titles") or [])
            pool_titles = list(record.get("pool_titles") or [])
            methods = baseline_orders(record=record, setr_record=setr_record, dbec_trace=dbec_trace)
            for variant in variants:
                prompt_row = prompt_by_key.get((dataset, query_index, variant), {})
                positions, parse_source = extract_positions_from_raw(
                    prompt_row.get("raw") or "",
                    pool_size=len(pool_titles),
                    top_k=TOP_K,
                    one_based_ids=one_based_ids,
                )
                methods[f"rankgpt_{variant}"] = fill_source_order(positions, pool_size=len(pool_titles))
                methods[f"rankgpt_{variant}__parse"] = positions  # type: ignore[assignment]
                methods[f"rankgpt_{variant}__parse_source"] = parse_source  # type: ignore[assignment]

            for method, order in methods.items():
                if method.endswith("__parse") or method.endswith("__parse_source"):
                    continue
                metric = support_metrics_for_order(order, record, top_k=TOP_K)
                parsed_positions = methods.get(f"{method.replace('rankgpt_', 'rankgpt_')}__parse")
                parse_source = methods.get(f"{method.replace('rankgpt_', 'rankgpt_')}__parse_source", "")
                prompt_key_variant = method.replace("rankgpt_", "") if method.startswith("rankgpt_") else ""
                prompt_row = prompt_by_key.get((dataset, query_index, prompt_key_variant), {})
                output.append(
                    {
                        "dataset": DATASET_LABELS.get(dataset, dataset),
                        "base_dataset": dataset,
                        "query_index": query_index,
                        "query_idx": record.get("query_idx"),
                        "question": str(record.get("question") or ""),
                        "method": method,
                        "gold_doc_count": len(gold_titles),
                        "support_hit_count": metric["hit_count"],
                        "support_recall": metric["recall"],
                        "support_complete": metric["complete"],
                        "missing_count": metric["missing_count"],
                        "selected_positions_json": json.dumps(list(order)[:TOP_K], ensure_ascii=False),
                        "selected_titles_json": json.dumps(selected_titles(order, record, top_k=TOP_K), ensure_ascii=False),
                        "missing_titles_json": json.dumps(metric["missing_titles"], ensure_ascii=False),
                        "rankgpt_parse_ok": safe_int(prompt_row.get("parse_ok")) if method.startswith("rankgpt_") else "",
                        "rankgpt_prompt_has_no_think": safe_int(prompt_row.get("prompt_has_no_think")) if method.startswith("rankgpt_") else "",
                        "rankgpt_selected_count": len(parsed_positions) if isinstance(parsed_positions, list) else "",
                        "rankgpt_parse_source": parse_source if method.startswith("rankgpt_") else "",
                        "rankgpt_error": str(prompt_row.get("error") or "") if method.startswith("rankgpt_") else "",
                    }
                )
    return output


def mean_value(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [safe_float(row.get(field)) for row in rows if str(row.get(field, "")) != ""]
    return sum(values) / len(values) if values else 0.0


def summarize_eval_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    datasets = sorted({str(row.get("dataset")) for row in rows})
    methods = sorted({str(row.get("method")) for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        for method in methods:
            method_rows = [row for row in dataset_rows if str(row.get("method")) == method]
            if not method_rows:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "queries": len(method_rows),
                    "support_recall_at_5": mean_value(method_rows, "support_recall"),
                    "support_complete_at_5": mean_value(method_rows, "support_complete"),
                    "parse_ok_rate": mean_value(method_rows, "rankgpt_parse_ok") if method.startswith("rankgpt_") else "",
                    "prompt_no_think_rate": mean_value(method_rows, "rankgpt_prompt_has_no_think") if method.startswith("rankgpt_") else "",
                    "mean_selected_count": mean_value(method_rows, "rankgpt_selected_count") if method.startswith("rankgpt_") else "",
                }
            )
    return output


def build_musique_missing_rows(eval_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    audit_path = Path("reports/repair_candidate_quality_audit_20260507/missing_gold_audit.csv")
    if not audit_path.exists():
        return []
    import csv

    target_rows = []
    with audit_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("dataset")) == "MuSiQue" and str(row.get("bucket")) == "gold_source_only_not_rank_or_dbec":
                target_rows.append(dict(row))
    by_key = {
        (safe_int(row.get("query_index")), str(row.get("method"))): row
        for row in eval_rows
        if str(row.get("dataset")) == "MuSiQue"
    }
    output: list[dict[str, Any]] = []
    for target in target_rows:
        query_index = safe_int(target.get("query_index"))
        gold_title = str(target.get("gold_title") or "")
        source_rank = safe_int(target.get("source_best_gold_rank"), default=999)
        for method in sorted({str(row.get("method")) for row in eval_rows if str(row.get("dataset")) == "MuSiQue"}):
            row = by_key.get((query_index, method))
            if not row:
                continue
            selected_titles = json.loads(str(row.get("selected_titles_json") or "[]"))
            hit = any(normalize_title(title) == normalize_title(gold_title) for title in selected_titles)
            output.append(
                {
                    "dataset": "MuSiQue",
                    "query_index": query_index,
                    "method": method,
                    "gold_title": gold_title,
                    "source_best_gold_rank": source_rank,
                    "hit_at_5": int(hit),
                    "new_hit_at_5": int(source_rank >= TOP_K and hit),
                    "selected_titles_json": row.get("selected_titles_json"),
                }
            )
    return output


def summarize_missing_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in sorted({str(row.get("method")) for row in rows}):
        method_rows = [row for row in rows if str(row.get("method")) == method]
        output.append(
            {
                "dataset": "MuSiQue",
                "method": method,
                "missing_gold_titles": len(method_rows),
                "queries": len({safe_int(row.get("query_index")) for row in method_rows}),
                "hit_at_5": mean_value(method_rows, "hit_at_5"),
                "new_at_5": mean_value(method_rows, "new_hit_at_5"),
            }
        )
    return output


def pct(value: Any) -> str:
    return f"{100.0 * safe_float(value):.1f}%"


def build_markdown(
    *,
    metadata: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    missing_summary: Sequence[Mapping[str, Any]],
    report_dir: Path,
) -> str:
    lines = [
        "# RankGPT Fixed-Pool Baseline",
        "",
        "This report evaluates RankGPT-style no-thinking listwise selection over the fixed PropRAG pool100. It does not retrieve new documents and does not run the reader.",
        "",
        "## Setup",
        "",
        f"- Datasets: `{metadata.get('datasets')}`",
        f"- Limit: `{metadata.get('limit')}`",
        f"- Variants: `{metadata.get('variants')}`",
        f"- Passage id convention: `{'1-based' if metadata.get('one_based_ids') else '0-based'}`",
        f"- Run LLM: `{metadata.get('run_llm')}`",
        f"- LLM base URLs: `{metadata.get('llm_base_urls')}`",
        f"- Workers: `{metadata.get('workers')}`",
        f"- All prompts use `/no_think`: `{metadata.get('all_prompts_no_think')}`",
        "",
        "## Selector Support Metrics",
        "",
        "| Dataset | Method | Q | Support R@5 | Support Complete@5 | Parse ok | selected count |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {method} | {q} | {recall} | {complete} | {parse} | {count} |".format(
                dataset=row.get("dataset"),
                method=row.get("method"),
                q=row.get("queries"),
                recall=pct(row.get("support_recall_at_5")),
                complete=pct(row.get("support_complete_at_5")),
                parse=pct(row.get("parse_ok_rate")) if row.get("parse_ok_rate") != "" else "",
                count=f"{safe_float(row.get('mean_selected_count')):.2f}" if row.get("mean_selected_count") != "" else "",
            )
        )

    if missing_summary:
        lines.extend(
            [
                "",
                "## MuSiQue Source-Visible Missing-Gold Slice",
                "",
                "| Method | Missing titles | Q | Hit@5 | New@5 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in missing_summary:
            lines.append(
                "| {method} | {n} | {q} | {hit} | {new} |".format(
                    method=row.get("method"),
                    n=row.get("missing_gold_titles"),
                    q=row.get("queries"),
                    hit=pct(row.get("hit_at_5")),
                    new=pct(row.get("new_at_5")),
                )
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-query rows: `{report_dir / 'selector_rows.csv'}`",
            f"- Method summary: `{report_dir / 'method_summary.csv'}`",
            f"- Prompt outputs: `{report_dir / 'prompt_outputs.jsonl'}`",
            f"- MuSiQue missing rows: `{report_dir / 'musique_missing_rows.csv'}`",
            f"- Full summary: `{report_dir / 'summary.json'}`",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    datasets = tuple(args.datasets)
    variants = tuple(args.variants)
    base_urls = split_base_urls(args.llm_base_urls)
    llm_probes = [probe_llm(base_url, str(args.llm_model), min(int(args.timeout), 15)) for base_url in base_urls]
    if bool(args.run_llm) and not any(probe.get("available") for probe in llm_probes):
        raise RuntimeError(f"No available LLM endpoints: {dict(zip(base_urls, llm_probes))}")
    active_urls = [url for url, probe in zip(base_urls, llm_probes) if probe.get("available")] or base_urls

    tasks, _records_by_dataset = build_tasks(
        datasets=datasets,
        limit=int(args.limit),
        max_doc_chars=int(args.max_doc_chars),
        llm_model=str(args.llm_model),
        variants=variants,
        one_based_ids=not bool(args.zero_based_ids),
    )
    cache_path = Path(args.cache_path) if args.cache_path else report_dir / "llm_cache.jsonl"
    cache = JsonlCache(cache_path)
    prompt_rows = run_llm_tasks(
        tasks=tasks,
        cache=cache,
        run_llm=bool(args.run_llm),
        base_urls=active_urls,
        model=str(args.llm_model),
        max_tokens=int(args.max_tokens),
        timeout=int(args.timeout),
        workers=int(args.workers),
    )
    selector_rows = build_eval_rows(
        datasets=datasets,
        variants=variants,
        prompt_rows=prompt_rows,
        limit=int(args.limit),
        one_based_ids=not bool(args.zero_based_ids),
    )
    method_summary = summarize_eval_rows(selector_rows)
    missing_rows = build_musique_missing_rows(selector_rows)
    missing_summary = summarize_missing_rows(missing_rows)
    metadata = {
        "datasets": list(datasets),
        "limit": int(args.limit),
        "variants": list(variants),
        "strict_fixed_pool": True,
        "run_llm": bool(args.run_llm),
        "llm_base_urls": active_urls,
        "llm_model": str(args.llm_model),
        "llm_probes": dict(zip(base_urls, llm_probes)),
        "workers": int(args.workers),
        "max_doc_chars": int(args.max_doc_chars),
        "one_based_ids": not bool(args.zero_based_ids),
        "cache_path": str(cache_path),
        "prompt_count": len(prompt_rows),
        "all_prompts_no_think": all(safe_int(row.get("prompt_has_no_think")) for row in prompt_rows) if prompt_rows else True,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {
        "metadata": metadata,
        "method_summary": method_summary,
        "musique_missing_summary": missing_summary,
    }
    write_csv_lf(selector_rows, report_dir / "selector_rows.csv")
    write_csv_lf(method_summary, report_dir / "method_summary.csv")
    write_jsonl(prompt_rows, report_dir / "prompt_outputs.jsonl")
    write_csv_lf(missing_rows, report_dir / "musique_missing_rows.csv")
    write_csv_lf(missing_summary, report_dir / "musique_missing_summary.csv")
    write_json(payload, report_dir / "summary.json")
    (report_dir / "summary.md").write_text(
        build_markdown(
            metadata=metadata,
            summary_rows=method_summary,
            missing_summary=missing_summary,
            report_dir=report_dir,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--run_llm", action="store_true")
    parser.add_argument("--llm_base_urls", nargs="+", default=[DEFAULT_LLM_BASE_URL])
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--cache_path", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_doc_chars", type=int, default=180)
    parser.add_argument("--zero_based_ids", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
