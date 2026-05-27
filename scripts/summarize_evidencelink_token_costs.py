#!/usr/bin/env python3
"""Summarize EvLink generative-token costs for the paper.

The main EvLink run stores exact token metadata for OpenIE calls in the
HippoRAG SQLite LLM cache and for DBEC binding calls in the selector traces.
Question-side requirement decomposition metadata was not retained in the final
stable reports, so this script reconstructs the fixed prompt and estimates that
small online component from the frozen requirement outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASETS: Sequence[tuple[str, str]] = (
    ("hotpotqa", "HotpotQA"),
    ("2wikimultihopqa", "2WikiMultiHopQA"),
    ("musique", "MuSiQue"),
)

DOC_COUNTS = {
    "hotpotqa": 9811,
    "2wikimultihopqa": 6119,
    "musique": 11656,
}

SETWISE_LLM_NO_THINK_PREFIX = "/no_think"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sqlite_token_totals(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("select metadata from cache").fetchall()
    finally:
        conn.close()
    prompt = 0
    completion = 0
    for (metadata_raw,) in rows:
        try:
            metadata = json.loads(metadata_raw or "{}")
        except json.JSONDecodeError:
            metadata = {}
        prompt += int(metadata.get("prompt_tokens") or 0)
        completion += int(metadata.get("completion_tokens") or 0)
    return {
        "calls": int(len(rows)),
        "input_tokens": int(prompt),
        "output_tokens": int(completion),
        "total_tokens": int(prompt + completion),
    }


def build_requirement_messages(query: str) -> list[dict[str, str]]:
    required_keys = "id, subquery, depends_on, expected_answer_type, anchor_mentions, role"
    example_payload = (
        "[{\"id\":\"s1\",\"subquery\":\"Who directed film X?\",\"depends_on\":[],"
        "\"expected_answer_type\":\"person\",\"anchor_mentions\":[\"film X\"],"
        "\"role\":\"bridge\"},"
        "{\"id\":\"s2\",\"subquery\":\"What is the birthplace of that director?\","
        "\"depends_on\":[\"s1\"],\"expected_answer_type\":\"location\","
        "\"anchor_mentions\":[],\"role\":\"answer\"}]"
    )
    return [
        {
            "role": "system",
            "content": (
                "You decompose multi-hop QA questions into evidence requirements for fixed-pool passage selection. "
                "Do not answer the question and do not fill unknown entities from world knowledge. "
                "Return JSON only: an array of 1-4 objects. Each object must have keys: "
                f"{required_keys}. "
                "Use ids like s1, s2. depends_on is a list of previous ids. "
                "A subquery should describe the evidence needed, not the final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{SETWISE_LLM_NO_THINK_PREFIX}\n"
                f"Question: {query}\n\n"
                "Return JSON array only. Example:\n"
                f"{example_payload}"
            ),
        },
    ]


def compact_requirement_output(requirements: Sequence[Mapping[str, Any]]) -> str:
    compact_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(requirements, start=1):
        compact_rows.append(
            {
                "id": str(row.get("id") or row.get("unit_id") or f"s{idx}"),
                "subquery": str(row.get("subquery") or ""),
                "depends_on": list(row.get("depends_on") or []),
                "expected_answer_type": str(row.get("expected_answer_type") or "unknown"),
                "anchor_mentions": list(row.get("anchor_mentions") or []),
                "role": str(row.get("role") or "support"),
            }
        )
    return json.dumps(compact_rows, ensure_ascii=False, separators=(",", ":"))


def make_token_counter() -> tuple[str, Any]:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            return len(encoding.encode(str(text)))

        return "cl100k_base", count
    except Exception:
        def count(text: str) -> int:
            return max(1, round(len(str(text)) / 4))

        return "char4_fallback", count


def count_messages(messages: Sequence[Mapping[str, str]], count_text: Any) -> int:
    # This is intentionally an estimate: the original API usage was not retained
    # for requirement decomposition, so we count the serialized chat content with
    # a small role overhead.
    total = 0
    for message in messages:
        total += 4
        total += count_text(str(message.get("role") or ""))
        total += count_text(str(message.get("content") or ""))
    return int(total + 2)


def summarize_requirement_estimate(report: Mapping[str, Any], count_text: Any) -> dict[str, int]:
    traces = list(report.get("setwise_selector_query_traces") or [])
    input_tokens = 0
    output_tokens = 0
    count = 0
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        question = str(trace.get("question") or "")
        selector_trace = trace.get("selector_trace")
        if not isinstance(selector_trace, Mapping):
            continue
        requirements = [
            row
            for row in list(selector_trace.get("requirements") or [])
            if isinstance(row, Mapping) and str(row.get("subquery") or "").strip()
        ]
        input_tokens += count_messages(build_requirement_messages(question), count_text)
        output_tokens += count_text(compact_requirement_output(requirements))
        count += 1
    return {
        "queries": int(count),
        "input_tokens_est": int(input_tokens),
        "output_tokens_est": int(output_tokens),
        "total_tokens_est": int(input_tokens + output_tokens),
    }


def summarize_binding(report: Mapping[str, Any]) -> dict[str, float | int]:
    traces = list(report.get("setwise_selector_query_traces") or [])
    queries = 0
    calls = 0
    input_tokens = 0
    output_tokens = 0
    latency_s = 0.0
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        selector_trace = trace.get("selector_trace")
        if not isinstance(selector_trace, Mapping):
            continue
        cost = selector_trace.get("llm_binding_cost")
        if not isinstance(cost, Mapping):
            continue
        queries += 1
        calls += int(cost.get("calls") or 0)
        input_tokens += int(cost.get("prompt_tokens") or 0)
        output_tokens += int(cost.get("completion_tokens") or 0)
        latency_s += float(cost.get("latency_s") or 0.0)
    return {
        "queries": int(queries),
        "calls": int(calls),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens + output_tokens),
        "latency_s": round(float(latency_s), 4),
    }


def fmt_millions(value: float | int) -> str:
    return f"{float(value) / 1_000_000:.2f}"


def fmt_per_query(value: float | int, queries: int) -> str:
    return f"{float(value) / max(int(queries), 1):.0f}"


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("_", "\\_")
    )


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidencelink_token_costs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "evidencelink_token_costs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "dataset", "metric", "value"])
        for row in payload["offline"]:
            for key, value in row.items():
                if key != "dataset":
                    writer.writerow(["offline", row["dataset"], key, value])
        for row in payload["online"]:
            for key, value in row.items():
                if key != "dataset":
                    writer.writerow(["online", row["dataset"], key, value])

    tex_lines = [
        "% Auto-generated by scripts/summarize_evidencelink_token_costs.py",
        "\\begin{table}[h]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & Docs & OpenIE calls & Input (M) & Output (M) & Total (M) \\\\",
        "\\midrule",
    ]
    for row in payload["offline"]:
        tex_lines.append(
            f"{latex_escape(row['dataset'])} & {int(row['docs']):,} & {int(row['openie_calls']):,} "
            f"& {fmt_millions(row['openie_input_tokens'])} "
            f"& {fmt_millions(row['openie_output_tokens'])} "
            f"& {fmt_millions(row['openie_total_tokens'])} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Offline generative-token cost for corpus structuring.",
            "Counts are exact API metadata from the Qwen3-32B OpenIE cache.",
            "Graph construction and NV-Embed-v2 embedding are excluded because",
            "they do not consume generative LLM tokens.}",
            "\\label{tab:offline-token-cost}",
            "\\end{table}",
            "",
            "\\begin{table}[h]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Dataset & Req. in/q & Req. out/q & Bind calls/q & Bind in/q & Bind out/q & Aux total/q \\\\",
            "\\midrule",
        ]
    )
    for row in payload["online"]:
        queries = int(row["queries"])
        tex_lines.append(
            f"{latex_escape(row['dataset'])} "
            f"& {fmt_per_query(row['requirement_input_tokens_est'], queries)} "
            f"& {fmt_per_query(row['requirement_output_tokens_est'], queries)} "
            f"& {float(row['binding_calls']) / max(queries, 1):.1f} "
            f"& {fmt_per_query(row['binding_input_tokens'], queries)} "
            f"& {fmt_per_query(row['binding_output_tokens'], queries)} "
            f"& {fmt_per_query(row['aux_total_tokens_est'], queries)} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Online auxiliary generative-token cost per query, excluding",
            "the shared GPT-4o-mini reader. Requirement-decomposition tokens are",
            f"estimated with the \\texttt{{{latex_escape(payload['tokenizer'])}}} tokenizer from the",
            "stored prompts and frozen requirement outputs; binding tokens are exact",
            "metadata from selector traces.}",
            "\\label{tab:online-token-cost}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "evidencelink_token_cost_tables.tex").write_text(
        "\n".join(tex_lines),
        encoding="utf-8",
    )

    md_lines = [
        "# EvLink Token Cost Summary",
        "",
        "## Offline",
        "",
        "| Dataset | Docs | OpenIE calls | Input (M) | Output (M) | Total (M) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["offline"]:
        md_lines.append(
            f"| {row['dataset']} | {int(row['docs']):,} | {int(row['openie_calls']):,} | "
            f"{fmt_millions(row['openie_input_tokens'])} | "
            f"{fmt_millions(row['openie_output_tokens'])} | "
            f"{fmt_millions(row['openie_total_tokens'])} |"
        )
    md_lines.extend(
        [
            "",
            "## Online Auxiliary",
            "",
            "| Dataset | Req. in/q | Req. out/q | Bind calls/q | Bind in/q | Bind out/q | Aux total/q |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["online"]:
        queries = int(row["queries"])
        md_lines.append(
            f"| {row['dataset']} | {fmt_per_query(row['requirement_input_tokens_est'], queries)} | "
            f"{fmt_per_query(row['requirement_output_tokens_est'], queries)} | "
            f"{float(row['binding_calls']) / max(queries, 1):.1f} | "
            f"{fmt_per_query(row['binding_input_tokens'], queries)} | "
            f"{fmt_per_query(row['binding_output_tokens'], queries)} | "
            f"{fmt_per_query(row['aux_total_tokens_est'], queries)} |"
        )
    (output_dir / "evidencelink_token_cost_summary.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    offline_root = root / "run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512"
    dbec_root = (
        root
        / "run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514/dbec_assets/etv4/evals"
    )
    tokenizer_name, count_text = make_token_counter()
    offline_rows: list[dict[str, Any]] = []
    online_rows: list[dict[str, Any]] = []
    for dataset, display_name in DATASETS:
        sqlite_path = offline_root / dataset / "index/llm_cache/qwen3-32b-judge_cache.sqlite"
        openie = sqlite_token_totals(sqlite_path)
        offline_rows.append(
            {
                "dataset": display_name,
                "docs": DOC_COUNTS[dataset],
                "openie_calls": openie["calls"],
                "openie_input_tokens": openie["input_tokens"],
                "openie_output_tokens": openie["output_tokens"],
                "openie_total_tokens": openie["total_tokens"],
                "source": str(sqlite_path.relative_to(root)),
            }
        )

        report_path = dbec_root / f"{dataset}_etv4_pool100_dbec_qwen32b_stable_limit1000.json"
        report = read_json(report_path)
        req = summarize_requirement_estimate(report, count_text)
        binding = summarize_binding(report)
        queries = int(req["queries"] or binding["queries"] or 0)
        online_rows.append(
            {
                "dataset": display_name,
                "queries": queries,
                "requirement_input_tokens_est": req["input_tokens_est"],
                "requirement_output_tokens_est": req["output_tokens_est"],
                "requirement_total_tokens_est": req["total_tokens_est"],
                "binding_calls": binding["calls"],
                "binding_input_tokens": binding["input_tokens"],
                "binding_output_tokens": binding["output_tokens"],
                "binding_total_tokens": binding["total_tokens"],
                "aux_total_tokens_est": int(req["total_tokens_est"] + binding["total_tokens"]),
                "binding_latency_s": binding["latency_s"],
                "source": str(report_path.relative_to(root)),
            }
        )
    return {
        "tokenizer": tokenizer_name,
        "protocol": {
            "offline_counts": "exact OpenAI-compatible usage metadata in Qwen3-32B OpenIE sqlite caches",
            "requirement_counts": "estimated from frozen prompt and stored requirement outputs",
            "binding_counts": "exact selector-trace metadata",
            "reader_policy": "GPT-4o-mini reader excluded because it is shared across methods",
        },
        "offline": offline_rows,
        "online": online_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    args = parser.parse_args()
    payload = build_summary(args)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(args.root).resolve() / output_dir
    write_outputs(payload, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "tokenizer": payload["tokenizer"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
