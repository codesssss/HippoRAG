#!/usr/bin/env python3
"""Analyze why PCEC's K=5,m=4 operating point is robust.

This is an offline, title-level mechanism audit over the existing 8B full1000
frontier artifacts. It intentionally avoids reader reruns.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DEFAULT_FRONTIER_SUMMARY = Path("reports/etv3_dbec_selector_frontier_full1000_20260510/summary.json")
DEFAULT_POOL_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/pools")
DEFAULT_OUTPUT_DIR = Path("reports/pcec_m4_mechanism_8b_20260511")
DEFAULT_VARIANTS = ("top5_max0", "top4_max1", "top3_max2", "top2_max3", "top1_max2")
PRESERVE_M = {
    "top5_max0": 5,
    "top4_max1": 4,
    "top3_max2": 3,
    "top2_max3": 2,
    "top1_max2": 1,
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_title(title: Any) -> str:
    text = str(title or "").strip().lower()
    text = re.sub(r"\s*\([^)]*?\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_counter(titles: Sequence[Any]) -> Counter[str]:
    return Counter(normalize_title(title) for title in titles if normalize_title(title))


def coverage_count(selected_titles: Sequence[Any], gold_titles: Sequence[Any]) -> int:
    selected = title_counter(selected_titles)
    gold = title_counter(gold_titles)
    return int(sum(min(selected.get(title, 0), count) for title, count in gold.items()))


def all_gold(selected_titles: Sequence[Any], gold_titles: Sequence[Any]) -> bool:
    return coverage_count(selected_titles, gold_titles) >= sum(title_counter(gold_titles).values())


def pool_path(pool_root: Path, dataset: str, *, pool_k: int, limit: int) -> Path:
    return pool_root / f"{dataset}_etv3_pool{pool_k}_limit{limit}.json"


def get_query_traces(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    traces = list(payload.get("setwise_selector_query_traces") or [])
    if not traces:
        raise ValueError("payload has no setwise_selector_query_traces")
    return traces


def final_titles_for_variant(trace: Mapping[str, Any], variant: str) -> list[str]:
    key = "baseline_top_titles" if variant == "top5_max0" else "selector_top_titles"
    return [str(title) for title in list(trace.get(key) or [])[:5]]


def safe_swap_steps(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selector_trace = trace.get("selector_trace") or {}
    safe_trace = selector_trace.get("safe_projection_trace") or {}
    steps = list(safe_trace.get("safe_swap_steps") or [])
    if not steps:
        steps = list(selector_trace.get("selection_steps") or [])
    return [step for step in steps if isinstance(step, Mapping)]


def summarize_slot_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slot_gold_counts = Counter()
    query_rank_gold_counts = Counter()
    top4_all = 0
    top5_all = 0
    top4_cov_sum = 0
    top5_cov_sum = 0
    gold_sum = 0
    rank5_gold_top4_incomplete = 0
    rank4_gold_top3_incomplete = 0
    rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        gold_titles = list(record.get("gold_titles") or [])
        pool_titles = list(record.get("pool_titles") or [])
        gold_n = title_counter(gold_titles)
        gold_count = int(sum(gold_n.values()))
        gold_sum += gold_count
        top5 = pool_titles[:5]
        top4 = pool_titles[:4]
        top3 = pool_titles[:3]
        top4_cov = coverage_count(top4, gold_titles)
        top5_cov = coverage_count(top5, gold_titles)
        top4_cov_sum += top4_cov
        top5_cov_sum += top5_cov
        if top4_cov >= gold_count:
            top4_all += 1
        if top5_cov >= gold_count:
            top5_all += 1
        for slot, title in enumerate(top5, start=1):
            if gold_n.get(normalize_title(title), 0) > 0:
                slot_gold_counts[slot] += 1
                query_rank_gold_counts[slot] += 1
        if len(top5) >= 5 and gold_n.get(normalize_title(top5[4]), 0) > 0 and top4_cov < gold_count:
            rank5_gold_top4_incomplete += 1
        if len(top5) >= 4 and gold_n.get(normalize_title(top5[3]), 0) > 0 and coverage_count(top3, gold_titles) < gold_count:
            rank4_gold_top3_incomplete += 1
        rows.append(
            {
                "query_idx": int(record.get("query_idx", idx) or idx),
                "gold_count": gold_count,
                "top4_coverage": top4_cov,
                "top5_coverage": top5_cov,
                "rank4_is_gold": bool(len(top5) >= 4 and gold_n.get(normalize_title(top5[3]), 0) > 0),
                "rank5_is_gold": bool(len(top5) >= 5 and gold_n.get(normalize_title(top5[4]), 0) > 0),
                "top4_all_gold": top4_cov >= gold_count,
                "top5_all_gold": top5_cov >= gold_count,
            }
        )

    count = len(records)
    return {
        "count": count,
        "gold_title_total": int(gold_sum),
        "top4_all_gold_count": int(top4_all),
        "top4_all_gold_rate": round(top4_all / max(count, 1), 6),
        "top5_all_gold_count": int(top5_all),
        "top5_all_gold_rate": round(top5_all / max(count, 1), 6),
        "top4_avg_gold_coverage": round(top4_cov_sum / max(count, 1), 6),
        "top5_avg_gold_coverage": round(top5_cov_sum / max(count, 1), 6),
        "slot_gold_counts": {str(slot): int(slot_gold_counts.get(slot, 0)) for slot in range(1, 6)},
        "slot_gold_query_rates": {
            str(slot): round(slot_gold_counts.get(slot, 0) / max(count, 1), 6)
            for slot in range(1, 6)
        },
        "rank5_gold_top4_incomplete_count": int(rank5_gold_top4_incomplete),
        "rank5_gold_top4_incomplete_rate": round(rank5_gold_top4_incomplete / max(count, 1), 6),
        "rank4_gold_top3_incomplete_count": int(rank4_gold_top3_incomplete),
        "rank4_gold_top3_incomplete_rate": round(rank4_gold_top3_incomplete / max(count, 1), 6),
        "rows": rows,
    }


def summarize_variant_swaps(traces: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    total_swaps = 0
    swap_out_by_slot: dict[str, Counter[str]] = defaultdict(Counter)
    query_gold_out = 0
    query_gold_in = 0
    improved = 0
    worsened = 0
    complete_rescue = 0
    complete_regression = 0
    changed = 0

    for trace in traces:
        gold_titles = list(trace.get("gold_titles") or [])
        baseline_titles = list(trace.get("baseline_top_titles") or [])[:5]
        final_titles = final_titles_for_variant(trace, variant)
        base_cov = coverage_count(baseline_titles, gold_titles)
        final_cov = coverage_count(final_titles, gold_titles)
        base_all = all_gold(baseline_titles, gold_titles)
        final_all = all_gold(final_titles, gold_titles)
        if final_titles != baseline_titles:
            changed += 1
        if final_cov > base_cov:
            improved += 1
        if final_cov < base_cov:
            worsened += 1
        if not base_all and final_all:
            complete_rescue += 1
        if base_all and not final_all:
            complete_regression += 1

        gold_n = title_counter(gold_titles)
        steps = safe_swap_steps(trace)
        any_gold_out = False
        any_gold_in = False
        for step in steps:
            total_swaps += 1
            out_slot = int(step.get("out_position", -1)) + 1
            out_title = step.get("out_title")
            in_title = step.get("in_title")
            out_gold = gold_n.get(normalize_title(out_title), 0) > 0
            in_gold = gold_n.get(normalize_title(in_title), 0) > 0
            slot_key = str(out_slot)
            swap_out_by_slot[slot_key]["total"] += 1
            if out_gold:
                swap_out_by_slot[slot_key]["gold_out"] += 1
                any_gold_out = True
            if in_gold:
                swap_out_by_slot[slot_key]["gold_in"] += 1
                any_gold_in = True
        if any_gold_out:
            query_gold_out += 1
        if any_gold_in:
            query_gold_in += 1

    slot_rows = {}
    for slot in sorted(swap_out_by_slot, key=lambda value: int(value) if value.lstrip("-").isdigit() else 999):
        row = swap_out_by_slot[slot]
        total = int(row.get("total", 0))
        gold_out = int(row.get("gold_out", 0))
        gold_in = int(row.get("gold_in", 0))
        slot_rows[slot] = {
            "total_swaps": total,
            "gold_out_swaps": gold_out,
            "gold_out_rate": round(gold_out / max(total, 1), 6),
            "gold_in_swaps": gold_in,
            "gold_in_rate": round(gold_in / max(total, 1), 6),
        }
    count = len(traces)
    return {
        "variant": variant,
        "preserve_m": int(PRESERVE_M.get(variant, -1)),
        "count": count,
        "changed_count": int(changed),
        "changed_rate": round(changed / max(count, 1), 6),
        "total_swaps": int(total_swaps),
        "avg_swaps": round(total_swaps / max(count, 1), 6),
        "improved_count": int(improved),
        "worsened_count": int(worsened),
        "complete_rescue_count": int(complete_rescue),
        "complete_regression_count": int(complete_regression),
        "queries_swapped_out_gold": int(query_gold_out),
        "queries_swapped_in_gold": int(query_gold_in),
        "swap_out_by_slot": slot_rows,
    }


def summarize_oracle_variants(
    traces_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    variants: Sequence[str],
) -> dict[str, Any]:
    count = len(next(iter(traces_by_variant.values())))
    conservative_best_m = Counter()
    aggressive_best_m = Counter()
    unique_best_m = Counter()
    m4_matches_max = 0
    m4_unique_best = 0
    m4_improves_over_baseline = 0
    m4_worsens_vs_baseline = 0
    m4_matches_or_exceeds_baseline = 0
    best_rows: list[dict[str, Any]] = []

    for idx in range(count):
        variant_cov: dict[str, int] = {}
        gold_titles = list(traces_by_variant[variants[0]][idx].get("gold_titles") or [])
        for variant in variants:
            trace = traces_by_variant[variant][idx]
            variant_cov[variant] = coverage_count(final_titles_for_variant(trace, variant), gold_titles)
        baseline_cov = variant_cov.get("top5_max0", 0)
        m4_cov = variant_cov.get("top4_max1", 0)
        max_cov = max(variant_cov.values()) if variant_cov else 0
        best_variants = [variant for variant, cov in variant_cov.items() if cov == max_cov]
        best_ms = [int(PRESERVE_M[variant]) for variant in best_variants]
        conservative_m = max(best_ms)
        aggressive_m = min(best_ms)
        conservative_best_m[conservative_m] += 1
        aggressive_best_m[aggressive_m] += 1
        unique_variants = [variant for variant, cov in variant_cov.items() if cov == max_cov]
        if len(unique_variants) == 1:
            unique_best_m[int(PRESERVE_M[unique_variants[0]])] += 1
            if unique_variants[0] == "top4_max1":
                m4_unique_best += 1
        if m4_cov == max_cov:
            m4_matches_max += 1
        if m4_cov > baseline_cov:
            m4_improves_over_baseline += 1
        if m4_cov < baseline_cov:
            m4_worsens_vs_baseline += 1
        if m4_cov >= baseline_cov:
            m4_matches_or_exceeds_baseline += 1
        best_rows.append(
            {
                "query_idx": idx,
                "gold_count": int(sum(title_counter(gold_titles).values())),
                "baseline_cov": int(baseline_cov),
                "m4_cov": int(m4_cov),
                "max_cov": int(max_cov),
                "conservative_best_m": int(conservative_m),
                "aggressive_best_m": int(aggressive_m),
                "coverage_by_variant": dict(variant_cov),
            }
        )

    def counter_to_rates(counter: Counter[int]) -> dict[str, dict[str, float | int]]:
        return {
            str(m): {
                "count": int(counter.get(m, 0)),
                "rate": round(counter.get(m, 0) / max(count, 1), 6),
            }
            for m in sorted({1, 2, 3, 4, 5} | set(counter.keys()), reverse=True)
        }

    return {
        "count": int(count),
        "conservative_best_m_distribution": counter_to_rates(conservative_best_m),
        "aggressive_best_m_distribution": counter_to_rates(aggressive_best_m),
        "unique_best_m_distribution": counter_to_rates(unique_best_m),
        "nonbaseline_oracle_need_count": int(count - conservative_best_m.get(5, 0)),
        "nonbaseline_oracle_need_rate": round(
            (count - conservative_best_m.get(5, 0)) / max(count, 1),
            6,
        ),
        "nonbaseline_m4_count": int(conservative_best_m.get(4, 0)),
        "nonbaseline_m4_share": round(
            conservative_best_m.get(4, 0) / max(count - conservative_best_m.get(5, 0), 1),
            6,
        ),
        "nonbaseline_below_m4_count": int(
            sum(conservative_best_m.get(m, 0) for m in (1, 2, 3))
        ),
        "nonbaseline_below_m4_share": round(
            sum(conservative_best_m.get(m, 0) for m in (1, 2, 3))
            / max(count - conservative_best_m.get(5, 0), 1),
            6,
        ),
        "m4_matches_max_coverage_count": int(m4_matches_max),
        "m4_matches_max_coverage_rate": round(m4_matches_max / max(count, 1), 6),
        "m4_unique_best_count": int(m4_unique_best),
        "m4_unique_best_rate": round(m4_unique_best / max(count, 1), 6),
        "m4_improves_over_baseline_count": int(m4_improves_over_baseline),
        "m4_improves_over_baseline_rate": round(m4_improves_over_baseline / max(count, 1), 6),
        "m4_worsens_vs_baseline_count": int(m4_worsens_vs_baseline),
        "m4_worsens_vs_baseline_rate": round(m4_worsens_vs_baseline / max(count, 1), 6),
        "m4_matches_or_exceeds_baseline_count": int(m4_matches_or_exceeds_baseline),
        "m4_matches_or_exceeds_baseline_rate": round(m4_matches_or_exceeds_baseline / max(count, 1), 6),
        "rows": best_rows,
    }


def source_path_for_variant(frontier_summary: Mapping[str, Any], dataset: str, variant: str) -> Path:
    datasets = frontier_summary.get("datasets") or {}
    info = (datasets.get(dataset) or {}).get(variant) or {}
    source_path = info.get("source_path")
    if not source_path:
        raise KeyError(f"missing source_path for {dataset}/{variant}")
    return Path(str(source_path))


def analyze_dataset(
    *,
    dataset: str,
    frontier_summary: Mapping[str, Any],
    pool_root: Path,
    variants: Sequence[str],
    pool_k: int,
    limit: int,
) -> dict[str, Any]:
    pool_payload = read_json(pool_path(pool_root, dataset, pool_k=pool_k, limit=limit))
    pool_records = list(pool_payload.get("records") or [])[:limit]
    slot_distribution = summarize_slot_distribution(pool_records)

    traces_by_variant: dict[str, list[Mapping[str, Any]]] = {}
    variant_sources: dict[str, str] = {}
    swap_summary: dict[str, Any] = {}
    for variant in variants:
        source_path = source_path_for_variant(frontier_summary, dataset, variant)
        payload = read_json(source_path)
        traces = get_query_traces(payload)[:limit]
        traces_by_variant[variant] = traces
        variant_sources[variant] = str(source_path)
        if variant != "top5_max0":
            swap_summary[variant] = summarize_variant_swaps(traces, variant)

    oracle = summarize_oracle_variants(traces_by_variant, variants)
    return {
        "dataset": dataset,
        "pool_json": str(pool_path(pool_root, dataset, pool_k=pool_k, limit=limit)),
        "variant_sources": variant_sources,
        "slot_distribution": {key: value for key, value in slot_distribution.items() if key != "rows"},
        "swap_summary": swap_summary,
        "oracle_variant_summary": {key: value for key, value in oracle.items() if key != "rows"},
    }


def fmt_pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# PCEC m=4 Mechanism Audit (8B Full1000)",
        "",
        "This audit uses existing 8B frontier artifacts. It is title-level unless explicitly stated otherwise.",
        "",
        "Important caveat: `pool_doc_scores` are rank-coded in these artifacts (`200..196`, then `95..`), so a rank-4/5 score-gap plot would be circular and is not used as evidence.",
        "",
        "## Baseline Gold Slot Distribution",
        "",
        "| Dataset | Top4 all-gold | Top5 all-gold | Rank4 gold | Rank5 gold | Rank4 critical | Rank5 critical |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload.get("datasets", []) or []:
        slot = result["slot_distribution"]
        rates = slot["slot_gold_query_rates"]
        lines.append(
            "| {dataset} | {top4} | {top5} | {rank4} | {rank5} | {rank4crit} | {rank5crit} |".format(
                dataset=result["dataset"],
                top4=fmt_pct(slot["top4_all_gold_rate"]),
                top5=fmt_pct(slot["top5_all_gold_rate"]),
                rank4=fmt_pct(rates["4"]),
                rank5=fmt_pct(rates["5"]),
                rank4crit=fmt_pct(slot["rank4_gold_top3_incomplete_rate"]),
                rank5crit=fmt_pct(slot["rank5_gold_top4_incomplete_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "Definitions: `Rank5 critical` means rank 5 is gold and the top-4 prefix is not already all-gold. This is the direct gold-out risk of replacing slot 5. `Rank4 critical` is the analogous risk introduced by allowing replacement of slot 4.",
            "",
            "## Swap-Out Hazard by Variant",
            "",
            "| Dataset | Variant | Swaps | Gold-out queries | Complete rescues | Complete regressions | Slot hazards |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in payload.get("datasets", []) or []:
        for variant, summary in result.get("swap_summary", {}).items():
            hazard_bits = []
            for slot, row in (summary.get("swap_out_by_slot") or {}).items():
                hazard_bits.append(
                    f"r{slot}: {row['gold_out_swaps']}/{row['total_swaps']} ({fmt_pct(row['gold_out_rate'])})"
                )
            lines.append(
                "| {dataset} | {variant} | {swaps} | {goldout} | {rescues} | {regs} | {hazards} |".format(
                    dataset=result["dataset"],
                    variant=variant,
                    swaps=summary["total_swaps"],
                    goldout=summary["queries_swapped_out_gold"],
                    rescues=summary["complete_rescue_count"],
                    regs=summary["complete_regression_count"],
                    hazards="<br>".join(hazard_bits) if hazard_bits else "-",
                )
            )

    lines.extend(
        [
            "",
            "## Oracle Variant View",
            "",
            "`Conservative best m` chooses the largest preservation prefix among variants that achieve the maximum title coverage for that query. It asks: if an oracle picked among the already-run frontier variants, how often would it need to relax below m=4?",
            "",
            "| Dataset | m4 matches max coverage | m4 unique best | m4 improves over baseline | m4 worsens vs baseline | Conservative-best m distribution |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in payload.get("datasets", []) or []:
        oracle = result["oracle_variant_summary"]
        dist = oracle["conservative_best_m_distribution"]
        dist_text = ", ".join(
            f"m={m}: {row['count']} ({fmt_pct(row['rate'])})"
            for m, row in dist.items()
            if int(row["count"]) > 0
        )
        lines.append(
            "| {dataset} | {matches} | {unique} | {improves} | {worsens} | {dist} |".format(
                dataset=result["dataset"],
                matches=f"{oracle['m4_matches_max_coverage_count']} ({fmt_pct(oracle['m4_matches_max_coverage_rate'])})",
                unique=f"{oracle['m4_unique_best_count']} ({fmt_pct(oracle['m4_unique_best_rate'])})",
                improves=f"{oracle['m4_improves_over_baseline_count']} ({fmt_pct(oracle['m4_improves_over_baseline_rate'])})",
                worsens=f"{oracle['m4_worsens_vs_baseline_count']} ({fmt_pct(oracle['m4_worsens_vs_baseline_rate'])})",
                dist=dist_text,
            )
        )

    lines.extend(
        [
            "",
            "## Non-Baseline Oracle Need",
            "",
            "This table removes queries for which the no-op ETv3 top-5 already ties the best title coverage among the frontier variants. It measures where the composition layer actually matters.",
            "",
            "| Dataset | Non-baseline oracle need | m4 share among non-baseline | Below-m4 share among non-baseline |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in payload.get("datasets", []) or []:
        oracle = result["oracle_variant_summary"]
        lines.append(
            "| {dataset} | {need} ({need_rate}) | {m4} ({m4_share}) | {below} ({below_share}) |".format(
                dataset=result["dataset"],
                need=oracle["nonbaseline_oracle_need_count"],
                need_rate=fmt_pct(oracle["nonbaseline_oracle_need_rate"]),
                m4=oracle["nonbaseline_m4_count"],
                m4_share=fmt_pct(oracle["nonbaseline_m4_share"]),
                below=oracle["nonbaseline_below_m4_count"],
                below_share=fmt_pct(oracle["nonbaseline_below_m4_share"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. `m=4` is not explained by a score cliff: the stored pool scores are rank-coded, so score-gap analysis would be circular.",
            "2. The strongest mechanism evidence is slot vulnerability. Rank 4 is more often critical than rank 5 in the baseline top-5, and allowing replacement of earlier slots sharply raises gold-out risk, especially once rank 3 or rank 2 becomes replaceable.",
            "3. `m=4` captures most non-baseline oracle need on 2Wiki and HotpotQA, but it is not a unique oracle optimum. The right claim is conservative: with K=5, `m=4` is the robust one-slot boundary that preserves most selector gains while limiting gold displacement and reader-context risk.",
            "4. MuSiQue remains the warning case: even rank-5 replacement has high gold-out risk, and below-m4 variants add many swaps with limited extra rescues. This supports adaptive retention as future work rather than a stronger fixed cutoff.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_csv(value: str | Sequence[str]) -> list[str]:
    items = value.split(",") if isinstance(value, str) else list(value)
    return [str(item).strip() for item in items if str(item).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--frontier-summary", type=Path, default=DEFAULT_FRONTIER_SUMMARY)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    frontier_summary = read_json(args.frontier_summary)
    datasets = parse_csv(args.datasets)
    variants = parse_csv(args.variants)
    output = {
        "mode": "pcec_m4_mechanism_8b_full1000",
        "frontier_summary": str(args.frontier_summary),
        "pool_root": str(args.pool_root),
        "pool_k": int(args.pool_k),
        "limit": int(args.limit),
        "variants": variants,
        "datasets": [
            analyze_dataset(
                dataset=dataset,
                frontier_summary=frontier_summary,
                pool_root=Path(args.pool_root),
                variants=variants,
                pool_k=int(args.pool_k),
                limit=int(args.limit),
            )
            for dataset in datasets
        ],
    }

    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", output)
    (output_dir / "summary.md").write_text(render_markdown(output), encoding="utf-8")
    print(json.dumps({"output_json": str(output_dir / "summary.json"), "output_md": str(output_dir / "summary.md")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
