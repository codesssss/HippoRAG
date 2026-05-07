#!/usr/bin/env python3
"""Build a paper-facing cost/calls table for DAEC reviewer baselines."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_setr_spec = importlib.util.spec_from_file_location(
    "run_setr_style_selector",
    ROOT_DIR / "scripts" / "run_setr_style_selector.py",
)
if _setr_spec is None or _setr_spec.loader is None:
    raise RuntimeError("Unable to load scripts/run_setr_style_selector.py")
_setr_module = importlib.util.module_from_spec(_setr_spec)
_setr_spec.loader.exec_module(_setr_module)
build_prompt = _setr_module.build_prompt


REPORT_DIR = Path("reports/reviewer_baseline_costs_20260506")
SETR_RUN_DIR = Path("run_logs/setr_full1000_20260503")
SETR_REPORT_DIR = Path("reports/setr_full1000_20260503")
LLM_DIRECT_DIR = Path("run_logs/llm_direct_select_proprag_full1000_20260506")
IRCOT_DIR = Path("run_logs/ircot_style_full1000_20260506")
DAEC_DIR = Path("run_logs/daec_llm_wiki_title_proprag_full1000_20260503")
DAEC_SELECTIVE_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")

DATASETS: tuple[tuple[str, str], ...] = (
    ("2Wiki", "2wikimultihopqa"),
    ("HotpotQA", "hotpotqa"),
    ("MuSiQue", "musique"),
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_block(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return data.get("setwise_selector_qa") or {}


def retrieval_r5_from_selector(data: Mapping[str, Any]) -> float:
    block = metric_block(data)
    return safe_float((block.get("selector_retrieval_metrics") or {}).get("Recall@5"))


def retrieval_r5_from_overall(data: Mapping[str, Any]) -> float:
    return safe_float((data.get("overall_recomputed") or {}).get("Recall@5"))


def daec_path(dataset_slug: str) -> Path:
    return DAEC_DIR / f"{dataset_slug}_proprag_wiki_title_daec_llm_full1000.json"


def daec_selective_path(dataset_slug: str) -> Path:
    return DAEC_SELECTIVE_DIR / f"{dataset_slug}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def binding_cost_summary(data: Mapping[str, Any]) -> dict[str, float]:
    totals = {
        "attempts": 0.0,
        "calls": 0.0,
        "cache_hits": 0.0,
        "empty_entity_responses": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "latency_s": 0.0,
    }
    traces = data.get("setwise_selector_query_traces") or []
    for trace in traces:
        cost = ((trace.get("selector_trace") or {}).get("llm_binding_cost") or {})
        for key in totals:
            totals[key] += safe_float(cost.get(key))
    n = max(len(traces), 1)
    return {f"{key}_per_q": value / n for key, value in totals.items()}


def daec_rows(dataset: str, slug: str) -> list[dict[str, Any]]:
    base = read_json(daec_path(slug))
    selective = read_json(daec_selective_path(slug))
    base_metrics = metric_block(base)
    selective_metrics = metric_block(selective)
    base_cost = binding_cost_summary(base)
    selective_recorded_cost = binding_cost_summary(selective)
    logical_calls = 1.0 + base_cost["calls_per_q"]
    return [
        {
            "dataset": dataset,
            "method": "Top5 / original PropRAG pool order",
            "em": safe_float(base_metrics.get("baseline_EM")),
            "f1": safe_float(base_metrics.get("baseline_F1")),
            "support_r5": retrieval_r5_from_overall(base),
            "logical_selector_llm_calls_per_q": 0.0,
            "recorded_api_calls_per_q": 0.0,
            "extra_retrieval_calls_per_q": 0.0,
            "prompt_tokens_per_q": "",
            "completion_tokens_per_q": "",
            "prompt_chars_per_q": "",
            "completion_chars_per_q": "",
            "selector_latency_s_per_q": "",
            "cost_source": "No selector.",
        },
        {
            "dataset": dataset,
            "method": "DAEC-LLM+CTL",
            "em": safe_float(base_metrics.get("selector_EM")),
            "f1": safe_float(base_metrics.get("selector_F1")),
            "support_r5": retrieval_r5_from_selector(base),
            "logical_selector_llm_calls_per_q": logical_calls,
            "recorded_api_calls_per_q": base_cost["calls_per_q"],
            "extra_retrieval_calls_per_q": 0.0,
            "prompt_tokens_per_q": base_cost["prompt_tokens_per_q"],
            "completion_tokens_per_q": base_cost["completion_tokens_per_q"],
            "prompt_chars_per_q": "",
            "completion_chars_per_q": "",
            "selector_latency_s_per_q": base_cost["latency_s_per_q"],
            "cost_source": "Measured binding extraction only; logical calls add one decomposition call/query whose token usage is not instrumented.",
        },
        {
            "dataset": dataset,
            "method": "DAEC-selective titleuniq",
            "em": safe_float(selective_metrics.get("selector_EM")),
            "f1": safe_float(selective_metrics.get("selector_F1")),
            "support_r5": retrieval_r5_from_selector(selective),
            "logical_selector_llm_calls_per_q": logical_calls,
            "recorded_api_calls_per_q": base_cost["calls_per_q"],
            "extra_retrieval_calls_per_q": 0.0,
            "prompt_tokens_per_q": base_cost["prompt_tokens_per_q"],
            "completion_tokens_per_q": base_cost["completion_tokens_per_q"],
            "prompt_chars_per_q": "",
            "completion_chars_per_q": "",
            "selector_latency_s_per_q": base_cost["latency_s_per_q"],
            "cost_source": (
                "Table reports cold-equivalent cost, equal to DAEC-LLM+CTL. "
                f"The fresh run recorded {selective_recorded_cost['calls_per_q']:.2f} binding API calls/query because it used the binding cache."
            ),
        },
    ]


def setr_selection_seconds(label: str) -> float | None:
    log_path = SETR_RUN_DIR / "launcher.log"
    if not log_path.exists():
        return None
    start_re = re.compile(r"^\[(?P<ts>[^]]+)\]\s+START\s+" + re.escape(label) + r"\s*$")
    done_re = re.compile(r"^\[(?P<ts>[^]]+)\]\s+DONE\s+" + re.escape(label) + r"\s*$")
    start: datetime | None = None
    done: datetime | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        start_match = start_re.match(line)
        if start_match:
            start = datetime.fromisoformat(start_match.group("ts"))
        done_match = done_re.match(line)
        if done_match:
            done = datetime.fromisoformat(done_match.group("ts"))
    if not start or not done:
        return None
    return max(0.0, (done - start).total_seconds())


def setr_row(dataset: str, slug: str) -> dict[str, Any]:
    variant = "setr_k20_doc768"
    eval_data = read_json(SETR_REPORT_DIR / f"{slug}_proprag_{variant}.eval.json")
    input_rows = read_jsonl(SETR_RUN_DIR / f"{slug}_proprag_{variant}.inputs.jsonl")
    selection_rows = read_jsonl(SETR_RUN_DIR / f"{slug}_proprag_{variant}.selection.jsonl")
    prompt_chars = [len(build_prompt(row, append_no_think=True)) for row in input_rows]
    completion_chars = [len(str(row.get("raw_output") or "")) for row in selection_rows]
    prompt_chars_per_q = mean(prompt_chars) if prompt_chars else 0.0
    completion_chars_per_q = mean(completion_chars) if completion_chars else 0.0
    seconds = setr_selection_seconds(f"select_{slug}_proprag_{variant}")
    n = max(len(input_rows), 1)
    return {
        "dataset": dataset,
        "method": "SetR-style k20 doc768",
        "em": safe_float((eval_data.get("overall_recomputed") or {}).get("ExactMatch")),
        "f1": safe_float((eval_data.get("overall_recomputed") or {}).get("F1")),
        "support_r5": retrieval_r5_from_overall(eval_data),
        "logical_selector_llm_calls_per_q": 1.0,
        "recorded_api_calls_per_q": 1.0,
        "extra_retrieval_calls_per_q": 0.0,
        "prompt_tokens_per_q": prompt_chars_per_q / 4.0,
        "completion_tokens_per_q": completion_chars_per_q / 4.0,
        "prompt_chars_per_q": prompt_chars_per_q,
        "completion_chars_per_q": completion_chars_per_q,
        "selector_latency_s_per_q": (seconds / n) if seconds is not None else "",
        "cost_source": "Historical run predates token instrumentation; prompt/completion chars reconstructed and token counts estimated as chars/4.",
    }


def usage_summary(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    latencies = []
    parse_successes = 0
    for row in rows:
        usage = row.get("usage") or {}
        prompt = safe_float(usage.get("prompt_tokens"))
        completion = safe_float(usage.get("completion_tokens"))
        total = safe_float(usage.get("total_tokens"), prompt + completion)
        prompt_tokens.append(prompt)
        completion_tokens.append(completion)
        total_tokens.append(total)
        latencies.append(safe_float(row.get("latency_s")))
        parse_successes += int(bool(row.get("parse_success")))
    n = max(len(rows), 1)
    return {
        "prompt_tokens_per_q": mean(prompt_tokens) if prompt_tokens else 0.0,
        "completion_tokens_per_q": mean(completion_tokens) if completion_tokens else 0.0,
        "total_tokens_per_q": mean(total_tokens) if total_tokens else 0.0,
        "latency_s_per_q": mean(latencies) if latencies else 0.0,
        "parse_success_rate": parse_successes / n,
    }


def llm_direct_row(dataset: str, slug: str, variant: str) -> dict[str, Any]:
    eval_data = read_json(LLM_DIRECT_DIR / f"{slug}_{variant}_llm_direct.json")
    cache_rows = read_jsonl(LLM_DIRECT_DIR / "caches" / f"{slug}_{variant}.jsonl")
    usage = usage_summary(cache_rows)
    return {
        "dataset": dataset,
        "method": f"LLM-direct {variant}",
        "em": safe_float((eval_data.get("overall_recomputed") or {}).get("ExactMatch")),
        "f1": safe_float((eval_data.get("overall_recomputed") or {}).get("F1")),
        "support_r5": retrieval_r5_from_overall(eval_data),
        "logical_selector_llm_calls_per_q": 1.0,
        "recorded_api_calls_per_q": 1.0,
        "extra_retrieval_calls_per_q": 0.0,
        "prompt_tokens_per_q": usage["prompt_tokens_per_q"],
        "completion_tokens_per_q": usage["completion_tokens_per_q"],
        "prompt_chars_per_q": "",
        "completion_chars_per_q": "",
        "selector_latency_s_per_q": usage["latency_s_per_q"],
        "cost_source": f"Measured selector usage; parse_success={usage['parse_success_rate']:.3f}.",
    }


def ircot_row(dataset: str, slug: str) -> dict[str, Any]:
    data = read_json(IRCOT_DIR / f"{slug}_ircot_style.json")
    return {
        "dataset": dataset,
        "method": "IRCoT-style local",
        "em": safe_float(data.get("answer_em")),
        "f1": safe_float(data.get("answer_f1")),
        "support_r5": safe_float(data.get("supporting_paragraph_recall")),
        "logical_selector_llm_calls_per_q": safe_float(data.get("llm_calls_per_query")),
        "recorded_api_calls_per_q": safe_float(data.get("llm_calls_per_query")),
        "extra_retrieval_calls_per_q": safe_float(data.get("max_iter")),
        "prompt_tokens_per_q": "",
        "completion_tokens_per_q": "",
        "prompt_chars_per_q": "",
        "completion_chars_per_q": "",
        "selector_latency_s_per_q": safe_float(data.get("latency_per_query")),
        "cost_source": "Local IRCoT-style JSON stores calls, retrieval rounds, and wall-clock latency, not token usage.",
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, slug in DATASETS:
        rows.extend(daec_rows(dataset, slug))
        rows.append(setr_row(dataset, slug))
        rows.append(ircot_row(dataset, slug))
        rows.append(llm_direct_row(dataset, slug, "title"))
        rows.append(llm_direct_row(dataset, slug, "snippet128"))
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value == "":
        return "n/a"
    return f"{safe_float(value):.{digits}f}"


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "# Reviewer Baseline Cost Table - 2026-05-06",
        "",
        "Scope: PropRAG full1000 reviewer baselines. Costs are selector-side only; final reader QA cost is shared by all rows and omitted.",
        "",
        "Important caveats:",
        "",
        "- DAEC token/latency fields measure LLM binding extraction only. Logical calls add one decomposition call/query, but decomposition token usage is not instrumented in these artifacts.",
        "- DAEC-selective cold-equivalent cost equals DAEC because the title-uniqueness gate runs after binding extraction. The fresh run recorded zero binding API calls because it reused the binding cache.",
        "- Historical SetR-style rows predate token instrumentation, so exact token usage is unavailable; token counts are estimated as reconstructed chars/4 and character counts are also shown.",
        "- IRCoT-style local stores LLM call count, retrieval rounds, and wall-clock latency, but not token usage.",
        "",
        "## Main Cost Rows",
        "",
        "| Dataset | Method | EM | F1 | R@5 | Logical LLM calls/q | Measured/equiv API calls/q | Extra retrieval calls/q | Prompt tok/q | Completion tok/q | Prompt chars/q | Latency/q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {fmt(row['em'], 3)} | {fmt(row['f1'], 4)} | "
            f"{fmt(row['support_r5'], 4)} | {fmt(row['logical_selector_llm_calls_per_q'], 2)} | "
            f"{fmt(row['recorded_api_calls_per_q'], 2)} | {fmt(row['extra_retrieval_calls_per_q'], 2)} | "
            f"{fmt(row['prompt_tokens_per_q'], 1)} | {fmt(row['completion_tokens_per_q'], 1)} | "
            f"{fmt(row['prompt_chars_per_q'], 1)} | {fmt(row['selector_latency_s_per_q'], 3)} |"
        )

    lines.extend([
        "",
        "## Cost Sources",
        "",
        "| Dataset | Method | Source / Caveat |",
        "|---|---|---|",
    ])
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['cost_source']} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- DAEC is not lower-call than one-shot LLM selectors: cold logical cost is about one decomposition call plus 6-7 binding calls per query.",
        "- DAEC can still be prompt-token-efficient relative to snippet-heavy selectors: measured binding prompt tokens are about 1.1-1.3k/query, while LLM-direct snippet128 uses about 4.4-4.9k prompt tokens/query.",
        "- SetR-style k20 has one LLM call/query and estimated prompt tokens around 1.9-2.8k/query under the chars/4 heuristic; this is an estimate, not recorded API usage.",
        "- DAEC-selective should be sold as robustness, not cost reduction.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    csv_path = REPORT_DIR / "cost_table.csv"
    json_path = REPORT_DIR / "cost_table.json"
    md_path = REPORT_DIR / "cost_table.md"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps({
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
