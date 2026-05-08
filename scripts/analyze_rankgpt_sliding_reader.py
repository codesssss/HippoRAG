#!/usr/bin/env python3
"""Summarize the corrected RankGPT-style sliding-window reader run."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_reviewer_baseline_paired_ci import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    metric_mean,
    occurrence_keys,
    paired_bootstrap,
    safe_float,
    title_multiset_recall,
    trace_metric_rows,
)
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score  # noqa: E402


OUT_DIR = Path("reports/rankgpt_sliding_reader_full1000_datasetfix_20260508")
SLIDING_SELECTOR_DIR = Path("reports/rankgpt_fixed_pool_baseline_sliding_full1000_20260508")
SINGLE_PASS_SELECTOR_DIR = Path("reports/rankgpt_fixed_pool_baseline_full1000_1based_20260508")
DAEC_SELECTIVE_DIR = Path("run_logs/daec_selective_titleuniq_proprag_full1000_20260506")
SETR_FAITHFUL_DIR = Path("reports/setr_faithful_proprag_full1000_20260507")

RANKGPT_METHOD = "RankGPT-style sliding"
METHODS = ("DAEC-selective", "SetR-faithful", RANKGPT_METHOD)
COMPARISONS = (
    ("DAEC-selective", RANKGPT_METHOD),
    ("SetR-faithful", RANKGPT_METHOD),
    ("DAEC-selective", "SetR-faithful"),
)
METRICS = ("EM", "F1", "R5_TITLE")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def daec_selective_path(slug: str) -> Path:
    return DAEC_SELECTIVE_DIR / f"{slug}_proprag_wiki_title_daec_selective_titleuniq_full1000.json"


def setr_faithful_path(slug: str) -> Path:
    return SETR_FAITHFUL_DIR / f"{slug}_proprag_setr_k20_doc768_faithful.eval.json"


def rankgpt_reader_path(slug: str) -> Path:
    return OUT_DIR / f"{slug}_rankgpt_sliding20_step10.eval.json"


def examples_metric_rows_with_titles(data: Mapping[str, Any], *, reader_top_k: int = 5) -> list[dict[str, Any]]:
    examples = [row for row in data.get("examples", []) if isinstance(row, Mapping)]
    gold_answers = [list(row.get("gold_answers") or []) for row in examples]
    predictions = [str(row.get("answer") or "") for row in examples]
    _, per_query_em = QAExactMatch(global_config=None).calculate_metric_scores(gold_answers, predictions)
    _, per_query_f1 = QAF1Score(global_config=None).calculate_metric_scores(gold_answers, predictions)
    rows: list[dict[str, Any]] = []
    for example, em_row, f1_row in zip(examples, per_query_em, per_query_f1):
        retrieval_trace = example.get("retrieval_trace") or {}
        titles = list(retrieval_trace.get("external_pool_titles") or [])[:reader_top_k]
        rows.append(
            {
                "question": str(example.get("question") or ""),
                "EM": safe_float(em_row.get("ExactMatch")),
                "F1": safe_float(f1_row.get("F1")),
                "gold_titles": list(example.get("gold_titles") or []),
                "top_titles": titles[:5],
            }
        )
    return rows


def daec_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    return trace_metric_rows(data, field="selector_metrics")


def align_methods_local(dataset: str, methods: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    reference_keys = occurrence_keys(methods["DAEC-selective"])
    aligned: dict[str, list[dict[str, Any]]] = {}
    for name, rows in methods.items():
        keys = occurrence_keys(rows)
        if len(keys) != len(reference_keys):
            raise ValueError(f"{dataset}: {name} length={len(keys)}, expected={len(reference_keys)}")
        keyed = {key: row for key, row in zip(keys, rows)}
        missing = [key for key in reference_keys if key not in keyed]
        if missing:
            raise ValueError(f"{dataset}: {name} missing {len(missing)} occurrence-aligned questions")
        aligned[name] = [keyed[key] for key in reference_keys]

    reference_gold = [list(row.get("gold_titles") or []) for row in aligned["DAEC-selective"]]
    for rows in aligned.values():
        for row, gold_titles in zip(rows, reference_gold):
            if not row.get("gold_titles"):
                row["gold_titles"] = list(gold_titles)
            row["R5_TITLE"] = title_multiset_recall(list(gold_titles), list(row.get("top_titles") or []))
    return aligned


def dataset_methods(slug: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "DAEC-selective": daec_rows(read_json(daec_selective_path(slug))),
        "SetR-faithful": examples_metric_rows_with_titles(read_json(setr_faithful_path(slug))),
        RANKGPT_METHOD: examples_metric_rows_with_titles(read_json(rankgpt_reader_path(slug))),
    }


def selector_summary_lookup(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (str(row.get("dataset")), str(row.get("method"))): row
        for row in read_csv_rows(path)
    }


def support_row(
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    dataset: str,
    method: str,
) -> Mapping[str, Any]:
    return lookup.get((dataset, method), {})


def rankgpt_reader_overall(slug: str) -> dict[str, Any]:
    data = read_json(rankgpt_reader_path(slug))
    overall = data.get("overall_recomputed") or {}
    external = data.get("external_pool") or {}
    retrieval = external.get("payload_retrieval") or {}
    proprag_metrics = retrieval.get("proprag_metrics") or {}
    recomputed_title = retrieval.get("recomputed_title_recall") or {}
    return {
        "pipeline_recall_at_5": safe_float(overall.get("Recall@5")),
        "external_proprag_recall_at_5": safe_float(proprag_metrics.get("Recall@5")),
        "external_recomputed_title_recall_at_5": safe_float(recomputed_title.get("Recall@5")),
        "question_mismatch_count": int(external.get("question_mismatch_count") or 0),
        "unmatched_doc_count": int(external.get("unmatched_doc_count") or 0),
    }


def reader_recall_at_5(slug: str, method: str) -> float:
    if method == "DAEC-selective":
        data = read_json(daec_selective_path(slug))
        selector_qa = data.get("setwise_selector_qa") or {}
        retrieval = selector_qa.get("selector_retrieval_metrics") or {}
        return safe_float(retrieval.get("Recall@5"))
    if method == "SetR-faithful":
        data = read_json(setr_faithful_path(slug))
        overall = data.get("overall_recomputed") or {}
        return safe_float(overall.get("Recall@5"))
    if method == RANKGPT_METHOD:
        return safe_float(rankgpt_reader_overall(slug).get("pipeline_recall_at_5"))
    raise KeyError(method)


def build_mean_rows() -> list[dict[str, Any]]:
    sliding_lookup = selector_summary_lookup(SLIDING_SELECTOR_DIR / "method_summary.csv")
    rows: list[dict[str, Any]] = []
    selector_methods = {
        "DAEC-selective": "dbec_selective",
        "SetR-faithful": "setr_faithful",
        RANKGPT_METHOD: "rankgpt_sliding20_step10_no_think",
    }
    for dataset, slug in DATASETS:
        methods = align_methods_local(dataset, dataset_methods(slug))
        reader_overall = rankgpt_reader_overall(slug)
        for method in METHODS:
            selector = support_row(sliding_lookup, dataset=dataset, method=selector_methods[method])
            row = {
                "dataset": dataset,
                "method": method,
                "n": len(methods[method]),
                "EM": metric_mean(methods[method], "EM"),
                "F1": metric_mean(methods[method], "F1"),
                "reader_recall_at_5": reader_recall_at_5(slug, method),
                "R5_TITLE_CI_RECOMPUTED": metric_mean(methods[method], "R5_TITLE"),
                "selector_diagnostic_support_recall_at_5": safe_float(selector.get("support_recall_at_5"), default=""),
                "selector_diagnostic_complete_at_5": safe_float(selector.get("support_complete_at_5"), default=""),
                "parse_ok_rate": safe_float(selector.get("parse_ok_rate"), default=""),
            }
            if method == RANKGPT_METHOD:
                row.update(reader_overall)
                row["llm_calls_per_query"] = 9
                row["sequential_llm_calls_per_query"] = 9
            rows.append(row)
    return rows


def build_selector_variant_rows() -> list[dict[str, Any]]:
    single_lookup = selector_summary_lookup(SINGLE_PASS_SELECTOR_DIR / "method_summary.csv")
    sliding_lookup = selector_summary_lookup(SLIDING_SELECTOR_DIR / "method_summary.csv")
    variants = (
        ("rank5_no_think", single_lookup, "rankgpt_rank5_no_think", 1),
        ("select5_no_think", single_lookup, "rankgpt_select5_no_think", 1),
        ("sliding20_step10_no_think", sliding_lookup, "rankgpt_sliding20_step10_no_think", 9),
    )
    rows: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        for label, lookup, method, calls in variants:
            row = support_row(lookup, dataset=dataset, method=method)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": label,
                    "llm_calls_per_query": calls,
                    "support_recall_at_5": safe_float(row.get("support_recall_at_5")),
                    "support_complete_at_5": safe_float(row.get("support_complete_at_5")),
                    "parse_ok_rate": safe_float(row.get("parse_ok_rate")),
                }
            )
    return rows


def build_musique_missing_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in (
        ("single_pass", SINGLE_PASS_SELECTOR_DIR / "musique_missing_summary.csv"),
        ("sliding", SLIDING_SELECTOR_DIR / "musique_missing_summary.csv"),
    ):
        for row in read_csv_rows(path):
            if not str(row.get("method", "")).startswith("rankgpt_"):
                continue
            rows.append(
                {
                    "source": source,
                    "method": row["method"],
                    "missing_gold_titles": int(row["missing_gold_titles"]),
                    "queries": int(row["queries"]),
                    "hit_at_5": safe_float(row["hit_at_5"]),
                    "new_at_5": safe_float(row["new_at_5"]),
                }
            )
    return rows


def build_paired_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, slug in DATASETS:
        methods = align_methods_local(dataset, dataset_methods(slug))
        for left, right in COMPARISONS:
            for metric in METRICS:
                stats = paired_bootstrap(
                    [safe_float(row.get(metric)) for row in methods[left]],
                    [safe_float(row.get(metric)) for row in methods[right]],
                    seed=BOOTSTRAP_SEED + 707 + len(rows) * 43,
                    samples=BOOTSTRAP_SAMPLES,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{left} - {right}",
                        "metric": metric,
                        "left_mean": metric_mean(methods[left], metric),
                        "right_mean": metric_mean(methods[right], metric),
                        **stats,
                    }
                )
    return rows


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value == "":
        return ""
    return f"{safe_float(value):.{digits}f}"


def sign_fmt(value: Any, digits: int = 4) -> str:
    return f"{safe_float(value):+.{digits}f}"


def pct(value: Any) -> str:
    return f"{safe_float(value) * 100:.1f}%"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def find_ci(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    comparison: str,
    metric: str,
) -> Mapping[str, Any]:
    for row in rows:
        if row["dataset"] == dataset and row["comparison"] == comparison and row["metric"] == metric:
            return row
    raise KeyError((dataset, comparison, metric))


def build_paired_markdown(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "# RankGPT-Style Sliding Reader Paired CI",
        "",
        f"Query-paired percentile bootstrap over `{BOOTSTRAP_SAMPLES}` resamples. Delta is left method minus right method.",
        "",
        "| Dataset | Comparison | Metric | Left | Right | Delta | 95% CI | P(delta > 0) | Excludes 0 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["metric"] not in {"EM", "F1"}:
            continue
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['metric']} | "
            f"{fmt(row['left_mean'])} | {fmt(row['right_mean'])} | {sign_fmt(row['delta_mean'])} | "
            f"[{sign_fmt(row['ci_low'])}, {sign_fmt(row['ci_high'])}] | "
            f"{fmt(row['p_delta_gt_0'])} | {yes_no(row['ci_excludes_zero'])} |"
        )
    return "\n".join(lines) + "\n"


def build_summary_markdown(
    mean_rows: list[Mapping[str, Any]],
    paired_rows: list[Mapping[str, Any]],
    selector_variant_rows: list[Mapping[str, Any]],
    missing_rows: list[Mapping[str, Any]],
) -> str:
    lines = [
        "# RankGPT-Style Sliding Reader Full1000",
        "",
        "Date: 2026-05-08",
        "",
        "Purpose: evaluate a RankGPT-style sliding-window local adaptation under the same controlled Qwen3-8B `/no_think` substrate used by the DBEC/DAEC experiments. This is not a GPT-3.5/4 RankGPT reproduction.",
        "",
        "Important correction: `reports/rankgpt_sliding_reader_full1000_20260508/` is superseded. The corrected materialization filters selector rows by dataset before `query_index`; otherwise later-dataset rows can overwrite earlier datasets. All numbers below use `reports/rankgpt_sliding_reader_full1000_datasetfix_20260508/`.",
        "",
        "## Method Boundary",
        "",
        "| Aspect | Original RankGPT | This run |",
        "|---|---|---|",
        "| LLM | ChatGPT/GPT-4 | Qwen3-8B |",
        "| Candidate pool | BM25 top100 | PropRAG pool100 |",
        "| Ranking algorithm | Sliding-window permutation | Back-to-front sliding-window local adaptation, window=20, step=10 |",
        "| Thinking | Unspecified/default | `/no_think` |",
        "| Metric | IR nDCG | Multi-hop support metrics and QA EM/F1 |",
        "",
        "## Selector Variant Check",
        "",
        "Sliding-window is the most faithful RankGPT-style variant and improves single-pass `rank5_no_think` on overall selector Support R@5 for all three datasets. It remains below DAEC/SetR on selector support metrics.",
        "",
        "| Dataset | Variant | Calls/query | Support R@5 | Complete@5 | Parse ok |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selector_variant_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['llm_calls_per_query']} | "
            f"{pct(row['support_recall_at_5'])} | {pct(row['support_complete_at_5'])} | "
            f"{pct(row['parse_ok_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Reader Results",
            "",
            "EM/F1 are answer-reader metrics. Reader R@5 uses each method's paper-facing reader/eval recall field: DAEC-selective `selector_retrieval_metrics`, SetR-faithful `overall_recomputed`, and RankGPT-style sliding `overall_recomputed`. Selector Support R@5 is reported separately above as a diagnostic, not as the reader main-table support metric.",
            "",
            "| Dataset | Method | EM | F1 | Reader R@5 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dataset, _ in DATASETS:
        for method in METHODS:
            row = next(item for item in mean_rows if item["dataset"] == dataset and item["method"] == method)
            lines.append(
                f"| {dataset} | {method} | {fmt(row['EM'])} | {fmt(row['F1'])} | {pct(row['reader_recall_at_5'])} |"
            )

    lines.extend(
        [
            "",
            "## RankGPT Reader Sanity",
            "",
            "| Dataset | EM | F1 | Pipeline Recall@5 | External title Recall@5 | Question mismatches | Unmatched docs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, _ in DATASETS:
        row = next(item for item in mean_rows if item["dataset"] == dataset and item["method"] == RANKGPT_METHOD)
        lines.append(
            f"| {dataset} | {fmt(row['EM'])} | {fmt(row['F1'])} | "
            f"{fmt(row['pipeline_recall_at_5'])} | {fmt(row['external_recomputed_title_recall_at_5'])} | "
            f"{row['question_mismatch_count']} | {row['unmatched_doc_count']} |"
        )

    lines.extend(
        [
            "",
            "## Paired Bootstrap",
            "",
            "Query-paired percentile bootstrap with 10,000 resamples. Deltas are left method minus right method.",
            "",
            "| Dataset | Comparison | Metric | Left | Right | Delta | 95% CI | P(delta > 0) | Excludes 0 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for dataset, _ in DATASETS:
        for comparison in ("DAEC-selective - RankGPT-style sliding", "SetR-faithful - RankGPT-style sliding"):
            for metric in ("EM", "F1"):
                row = find_ci(paired_rows, dataset=dataset, comparison=comparison, metric=metric)
                lines.append(
                    f"| {dataset} | {comparison} | {metric} | {fmt(row['left_mean'])} | "
                    f"{fmt(row['right_mean'])} | {sign_fmt(row['delta_mean'])} | "
                    f"[{sign_fmt(row['ci_low'])}, {sign_fmt(row['ci_high'])}] | "
                    f"{fmt(row['p_delta_gt_0'])} | {yes_no(row['ci_excludes_zero'])} |"
                )

    lines.extend(
        [
            "",
            "## MuSiQue Missing-Slice Headroom",
            "",
            "The overall result is negative for replacing DAEC/SetR, but listwise selection still recovers a complementary subset of source-visible MuSiQue supports that DBEC/SetR miss.",
            "",
            "| Source | Method | Missing titles | Queries | New@5 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in missing_rows:
        lines.append(
            f"| {row['source']} | {row['method']} | {row['missing_gold_titles']} | "
            f"{row['queries']} | {pct(row['new_at_5'])} |"
        )

    lines.extend(
        [
            "",
            "## Cost Profile",
            "",
            "| Method | LLM calls/query | Sequential calls/query | Operational note |",
            "|---|---:|---:|---|",
            "| DAEC-selective | ~8.4 | partly parallelizable | Existing binding/extraction path; token usage should be reported from the DAEC run logs if needed. |",
            "| SetR-faithful | 1 | 1 | One listwise selection call before selected-only reader context. |",
            "| RankGPT-style single-pass | 1 | 1 | Cheaper but less faithful to RankGPT sliding-window behavior. |",
            "| RankGPT-style sliding | 9 | 9 | Back-to-front windows are sequential; measured run used 27,000 window calls for 3,000 queries. |",
            "",
            "Token usage was not emitted in the local vLLM outputs, so this report does not claim measured input/output token totals. The defensible cost claim here is call count and sequential dependency; exact token accounting should be collected separately if the paper needs a latency/cost table.",
            "",
            "## Interpretation",
            "",
            "- Sliding-window is the correct RankGPT-style local adaptation to report; single-pass is a useful ablation but not the named comparison.",
            "- Under the controlled Qwen3-8B `/no_think` substrate, RankGPT-style sliding is below DAEC-selective and SetR-faithful on reader F1 across 2Wiki, HotpotQA, and MuSiQue.",
            "- The MuSiQue missing-slice New@5 result remains important: listwise full-pool inspection can recover `24.0%` of the source-visible missing supports, so the fixed pool is not theoretically exhausted.",
            "- The paper claim should be bounded: DAEC/DBEC outperforms this RankGPT-style local adaptation under the controlled substrate; do not claim it outperforms original GPT-3.5/4 RankGPT.",
            "- With-thinking variants are not evaluated because the controlled substrate uses `/no_think` for all compared methods; cross-thinking-mode comparison is future work.",
            "",
            "## Files",
            "",
            f"- Comparison table: `{OUT_DIR / 'comparison.csv'}`",
            f"- Paired CI: `{OUT_DIR / 'paired_ci.csv'}` / `{OUT_DIR / 'paired_ci.md'}`",
            f"- Machine summary: `{OUT_DIR / 'summary.json'}`",
            f"- Corrected selected pools: `run_logs/rankgpt_sliding_full1000_20260508/*.selected_pool.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_rows = build_mean_rows()
    selector_variant_rows = build_selector_variant_rows()
    missing_rows = build_musique_missing_rows()
    paired_rows = build_paired_rows()

    write_csv(mean_rows, OUT_DIR / "comparison.csv")
    write_csv(selector_variant_rows, OUT_DIR / "selector_variant_comparison.csv")
    write_csv(missing_rows, OUT_DIR / "musique_missing_headroom.csv")
    write_csv(paired_rows, OUT_DIR / "paired_ci.csv")
    (OUT_DIR / "paired_ci.json").write_text(json.dumps(paired_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "paired_ci.md").write_text(build_paired_markdown(paired_rows), encoding="utf-8")
    payload = {
        "report": "RankGPT-Style Sliding Reader Full1000",
        "date": "2026-05-08",
        "corrected_reader_dir": str(OUT_DIR),
        "supersedes_invalid_reader_dir": "reports/rankgpt_sliding_reader_full1000_20260508",
        "method_boundary": {
            "llm": "Qwen3-8B",
            "thinking": "/no_think",
            "candidate_pool": "PropRAG pool100",
            "sliding_window_size": 20,
            "sliding_step": 10,
            "rank_start": 0,
            "rank_end": 100,
            "rankgpt_reproduction": "local adaptation, not GPT-3.5/4 reproduction",
        },
        "comparison": mean_rows,
        "selector_variants": selector_variant_rows,
        "musique_missing_headroom": missing_rows,
        "paired_ci": paired_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(
        build_summary_markdown(mean_rows, paired_rows, selector_variant_rows, missing_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(OUT_DIR), "datasets": len(DATASETS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
