#!/usr/bin/env python3
"""Offline policy variants for chain-walking anchor matching.

This script reuses the LLM entity outputs produced by
``probe_chain_walking_binding.py`` and reranks the same fixed MuSiQue pool100
under stricter title-only policies.  It makes no LLM or reader calls.

The goal is to test whether the weak chain-walking probe result was caused by
noisy body/token matching, or by the entity-only anchor itself being too broad.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import normalize_title, safe_float, safe_int  # noqa: E402
from probe_chain_walking_binding import (  # noqa: E402
    CANDIDATE_AUDIT_DIR,
    MUSIQUE_DATASET,
    MUSIQUE_LABEL,
    TARGET_BUCKET,
    doc_entity_score,
    entity_tokens,
    grouped_missing_gold,
    load_artifacts,
    normalize_match_text,
    rank_of_title,
    summarize_probe_rows,
    target_missing_gold_rows,
    write_csv,
    write_json,
)


REPORT_DIR = Path("reports/chain_walking_anchor_variants_20260507")
DEFAULT_PROBE_DIR = Path("reports/chain_walking_binding_probe_20260507")
DEFAULT_PRIMARY_PROBE_DIR = Path("reports/chain_walking_binding_probe_primary_20260507")

POLICIES = (
    "current_all",
    "title_strict",
    "title_only",
    "demand_entity_title",
)

DEMAND_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "with",
}

GENERIC_DEMAND_TOKENS = {
    "answer",
    "area",
    "city",
    "continent",
    "country",
    "demand",
    "entity",
    "find",
    "located",
    "location",
    "name",
    "named",
    "nation",
    "person",
    "place",
    "region",
    "state",
    "subquery",
    "thing",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def content_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_match_text(value).split()
        if len(token) >= 3
        and token not in DEMAND_STOP_TOKENS
        and token not in GENERIC_DEMAND_TOKENS
    }


def title_strict_score(entity: str, title: Any) -> tuple[float, str]:
    entity_norm = normalize_match_text(entity)
    title_norm = normalize_match_text(title)
    if not entity_norm or len(entity_norm) < 2:
        return 0.0, ""
    if entity_norm == title_norm:
        return 100.0, "title_exact"
    if len(entity_norm) >= 4 and entity_norm in title_norm:
        return 85.0, "title_substring"
    return 0.0, ""


def title_only_score(entity: str, title: Any) -> tuple[float, str]:
    score, reason = title_strict_score(entity, title)
    entity_tok = entity_tokens(entity)
    title_tok = entity_tokens(title)
    if entity_tok:
        overlap = len(entity_tok & title_tok) / len(entity_tok)
        overlap_score = 35.0 * overlap
        if overlap_score > score:
            return overlap_score, "title_token_overlap"
    return score, reason


def demand_entity_title_score(entity_row: Mapping[str, Any], title: Any) -> tuple[float, str]:
    entity = str(entity_row.get("entity") or "")
    entity_norm = normalize_match_text(entity)
    title_norm = normalize_match_text(title)
    if not entity_norm or len(entity_norm) < 2:
        return 0.0, ""

    entity_tok = entity_tokens(entity)
    title_tok = entity_tokens(title)
    if not entity_tok or not title_tok:
        return 0.0, ""

    entity_coverage = len(entity_tok & title_tok) / len(entity_tok)
    entity_phrase_match = len(entity_norm) >= 4 and entity_norm in title_norm
    if not entity_phrase_match and entity_coverage <= 0.0:
        return 0.0, ""

    demand_tok = content_tokens(entity_row.get("requirement_subquery"))
    demand_overlap = len(demand_tok & title_tok) / len(demand_tok) if demand_tok else 0.0
    phrase_score = 70.0 if entity_phrase_match else 0.0
    entity_score = max(phrase_score, 50.0 * entity_coverage)
    demand_score = 50.0 * demand_overlap
    conjunct_bonus = 10.0 if entity_score > 0.0 and demand_score > 0.0 else 0.0
    score = entity_score + demand_score + conjunct_bonus
    if demand_score > 0.0 and entity_phrase_match:
        reason = "demand_entity_title_phrase"
    elif demand_score > 0.0:
        reason = "demand_entity_title_tokens"
    elif entity_phrase_match:
        reason = "entity_title_phrase"
    else:
        reason = "entity_title_tokens"
    return score, reason


def score_doc_for_policy(
    *,
    policy: str,
    entity_row: Mapping[str, Any],
    title: Any,
    doc_text: Any,
) -> tuple[float, str]:
    entity = str(entity_row.get("entity") or "")
    if policy == "current_all":
        return doc_entity_score(entity, title, doc_text)
    if policy == "title_strict":
        return title_strict_score(entity, title)
    if policy == "title_only":
        return title_only_score(entity, title)
    if policy == "demand_entity_title":
        return demand_entity_title_score(entity_row, title)
    raise ValueError(f"Unknown policy: {policy}")


def rank_pool_by_policy(
    *,
    policy: str,
    pool_titles: Sequence[Any],
    pool_docs: Sequence[Any],
    entity_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    best_by_position: dict[int, dict[str, Any]] = {}
    for pos, title in enumerate(pool_titles):
        doc = pool_docs[pos] if pos < len(pool_docs) else title
        best: dict[str, Any] = {"score": 0.0, "entity": "", "reason": ""}
        for entity_row in entity_rows:
            score, reason = score_doc_for_policy(
                policy=policy,
                entity_row=entity_row,
                title=title,
                doc_text=doc,
            )
            if score > safe_float(best.get("score")):
                best = {
                    "score": float(score),
                    "entity": str(entity_row.get("entity") or ""),
                    "reason": reason,
                    "requirement_id": str(entity_row.get("requirement_id") or ""),
                    "requirement_subquery": str(entity_row.get("requirement_subquery") or ""),
                    "upstream_position": safe_int(entity_row.get("upstream_position"), default=-1),
                    "upstream_title": str(entity_row.get("upstream_title") or ""),
                    "support_span": str(entity_row.get("support_span") or ""),
                    "why": str(entity_row.get("why") or ""),
                }
        best_by_position[pos] = best
    order = sorted(range(len(pool_titles)), key=lambda pos: (-safe_float(best_by_position[pos].get("score")), pos))
    return order, best_by_position


def entity_rows_by_query_from_outputs(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str, str, int]] = set()
    for row in rows:
        query_index = safe_int(row.get("query_index"))
        try:
            entities = json.loads(str(row.get("entities_json") or "[]"))
        except json.JSONDecodeError:
            entities = []
        for entity in entities if isinstance(entities, list) else []:
            if not isinstance(entity, Mapping):
                continue
            text = str(entity.get("entity") or entity.get("text") or "")
            key = (
                query_index,
                normalize_match_text(text),
                str(row.get("requirement_id") or ""),
                safe_int(row.get("upstream_position"), default=-1),
            )
            if not key[1] or key in seen:
                continue
            seen.add(key)
            output[query_index].append(
                {
                    "query_index": query_index,
                    "entity": text,
                    "support_span": str(entity.get("support_span") or ""),
                    "why": str(entity.get("why") or ""),
                    "requirement_id": str(row.get("requirement_id") or ""),
                    "requirement_subquery": str(row.get("requirement_subquery") or ""),
                    "upstream_position": safe_int(row.get("upstream_position"), default=-1),
                    "upstream_title": str(row.get("upstream_title") or ""),
                }
            )
    return dict(output)


def build_policy_rows(
    *,
    policies: Sequence[str],
    missing_rows: Sequence[Mapping[str, Any]],
    entity_rows_by_query: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    grouped = grouped_missing_gold(missing_rows)
    artifacts = load_artifacts(MUSIQUE_DATASET)
    source_records = artifacts["source_records"]
    output: list[dict[str, Any]] = []

    for policy in policies:
        for query_index in sorted(grouped):
            source_record = source_records[query_index]
            pool_titles = list(source_record.get("pool_titles") or [])
            pool_docs = list(source_record.get("pool_docs") or [])
            order, best_by_position = rank_pool_by_policy(
                policy=policy,
                pool_titles=pool_titles,
                pool_docs=pool_docs,
                entity_rows=entity_rows_by_query.get(query_index, []),
            )
            for missing_row in grouped[query_index]:
                gold_title = str(missing_row.get("gold_title") or "")
                source_rank = safe_int(missing_row.get("source_best_gold_rank"), default=999)
                probe_rank = rank_of_title(gold_title, order, pool_titles)
                if probe_rank is None:
                    probe_rank = source_rank
                gold_positions = [
                    pos
                    for pos, title in enumerate(pool_titles)
                    if normalize_title(title) == normalize_title(gold_title)
                ]
                gold_pos = gold_positions[0] if gold_positions else -1
                best = best_by_position.get(gold_pos, {"score": 0.0})
                output.append(
                    {
                        "policy": policy,
                        "dataset": MUSIQUE_LABEL,
                        "base_dataset": MUSIQUE_DATASET,
                        "query_index": query_index,
                        "question": str(missing_row.get("question") or source_record.get("question") or ""),
                        "gold_title": gold_title,
                        "source_best_gold_rank": source_rank,
                        "best_probe_rank": probe_rank,
                        "rank_improvement": source_rank - probe_rank,
                        "baseline_hit_at_3": int(source_rank < 3),
                        "baseline_hit_at_5": int(source_rank < 5),
                        "baseline_hit_at_10": int(source_rank < 10),
                        "baseline_hit_at_20": int(source_rank < 20),
                        "probe_hit_at_3": int(probe_rank < 3),
                        "probe_hit_at_5": int(probe_rank < 5),
                        "probe_hit_at_10": int(probe_rank < 10),
                        "probe_hit_at_20": int(probe_rank < 20),
                        "new_hit_at_3": int(source_rank >= 3 and probe_rank < 3),
                        "new_hit_at_5": int(source_rank >= 5 and probe_rank < 5),
                        "new_hit_at_10": int(source_rank >= 10 and probe_rank < 10),
                        "new_hit_at_20": int(source_rank >= 20 and probe_rank < 20),
                        "best_probe_score": safe_float(best.get("score")),
                        "best_probe_entity": str(best.get("entity") or ""),
                        "best_probe_reason": str(best.get("reason") or ""),
                        "best_requirement_id": str(best.get("requirement_id") or ""),
                        "best_requirement_subquery": str(best.get("requirement_subquery") or ""),
                        "best_upstream_position": safe_int(best.get("upstream_position"), default=-1),
                        "best_upstream_title": str(best.get("upstream_title") or ""),
                        "best_support_span": str(best.get("support_span") or ""),
                    }
                )
    return output


def summarize_policy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policies: Sequence[str] = POLICIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for policy in policies:
        policy_rows = [row for row in rows if str(row.get("policy")) == policy]
        summaries, metadata = summarize_probe_rows(policy_rows)
        for summary in summaries:
            item = {"policy": policy, **summary}
            summary_rows.append(item)
            if summary.get("bucket") == "overall":
                comparison_rows.append(
                    {
                        "policy": policy,
                        "missing_gold_titles": summary.get("missing_gold_titles", 0),
                        "new_recall_at_5": summary.get("new_recall_at_5", 0.0),
                        "new_recall_at_10": summary.get("new_recall_at_10", 0.0),
                        "new_recall_at_20": summary.get("new_recall_at_20", 0.0),
                        "probe_recall_at_5": summary.get("probe_recall_at_5", 0.0),
                        "mean_source_rank": summary.get("mean_source_rank", 0.0),
                        "mean_probe_rank": summary.get("mean_probe_rank", 0.0),
                        "mean_rank_improvement": summary.get("mean_rank_improvement", 0.0),
                        "positive_score_rate": summary.get("positive_score_rate", 0.0),
                        "decision": metadata.get("decision", ""),
                    }
                )
    return summary_rows, comparison_rows


def add_current_delta_counts(
    comparison_rows: list[dict[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in variant_rows:
        key = (str(row.get("query_index")), str(row.get("gold_title")))
        by_key[key][str(row.get("policy"))] = row

    output: list[dict[str, Any]] = []
    for row in comparison_rows:
        policy = str(row.get("policy") or "")
        unique_gain = 0
        lost = 0
        for policy_rows in by_key.values():
            current = policy_rows.get("current_all")
            candidate = policy_rows.get(policy)
            if current is None or candidate is None:
                continue
            current_hit = safe_int(current.get("new_hit_at_5"))
            candidate_hit = safe_int(candidate.get("new_hit_at_5"))
            if candidate_hit and not current_hit:
                unique_gain += 1
            if current_hit and not candidate_hit:
                lost += 1
        output.append({**row, "unique_new5_vs_current": unique_gain, "lost_new5_vs_current": lost})
    return output


def best_policy(comparison_rows: Sequence[Mapping[str, Any]]) -> str:
    if not comparison_rows:
        return ""
    best = max(comparison_rows, key=lambda row: (safe_float(row.get("new_recall_at_5")), safe_float(row.get("new_recall_at_10"))))
    return str(best.get("policy") or "")


def build_markdown(
    *,
    metadata: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
    report_dir: Path,
) -> str:
    lines = [
        "# Chain-Walking Anchor Variant Probe",
        "",
        "This offline diagnostic reuses the chain-walking LLM entity outputs and reranks the fixed MuSiQue pool100 under stricter title-only policies. It makes no LLM or reader calls.",
        "",
        "## Policy Comparison",
        "",
        "| Policy | New @5 | New @10 | New @20 | unique @5 vs current | lost @5 vs current | mean probe rank | mean improvement | positive-score rate | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {policy} | {r5:.1f}% | {r10:.1f}% | {r20:.1f}% | {gain} | {lost} | {rank:.1f} | {imp:.1f} | {pos:.1f}% | {decision} |".format(
                policy=row.get("policy"),
                r5=100.0 * safe_float(row.get("new_recall_at_5")),
                r10=100.0 * safe_float(row.get("new_recall_at_10")),
                r20=100.0 * safe_float(row.get("new_recall_at_20")),
                gain=row.get("unique_new5_vs_current", 0),
                lost=row.get("lost_new5_vs_current", 0),
                rank=safe_float(row.get("mean_probe_rank")),
                imp=safe_float(row.get("mean_rank_improvement")),
                pos=100.0 * safe_float(row.get("positive_score_rate")),
                decision=row.get("decision", ""),
            )
        )

    lines.extend([
        "",
        "## Source-Rank Buckets",
        "",
        "| Policy | Bucket | Titles | New @5 | New @10 | New @20 | mean improvement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        if row.get("bucket") == "overall":
            continue
        lines.append(
            "| {policy} | {bucket} | {n} | {r5:.1f}% | {r10:.1f}% | {r20:.1f}% | {imp:.1f} |".format(
                policy=row.get("policy"),
                bucket=row.get("bucket"),
                n=row.get("missing_gold_titles"),
                r5=100.0 * safe_float(row.get("new_recall_at_5")),
                r10=100.0 * safe_float(row.get("new_recall_at_10")),
                r20=100.0 * safe_float(row.get("new_recall_at_20")),
                imp=safe_float(row.get("mean_rank_improvement")),
            )
        )

    best = best_policy(comparison_rows)
    current = next((row for row in comparison_rows if row.get("policy") == "current_all"), {})
    demand = next((row for row in comparison_rows if row.get("policy") == "demand_entity_title"), {})
    strict = next((row for row in comparison_rows if row.get("policy") == "title_strict"), {})
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Best New@5 policy: `{best}`.",
        "- If `title_strict` beats `current_all`, the earlier probe was hurt by body/token noise.",
        "- If `demand_entity_title` beats `current_all`, demand-conditioned reformulation has a measurable signal without new LLM calls.",
        "- If all policies stay below 15% New@5, this remains a weak future-work signal rather than a main-method redesign trigger.",
        "",
        "## Key Deltas",
        "",
        "| Comparison | d New@5 | d New@10 | d New@20 |",
        "|---|---:|---:|---:|",
        "| title_strict - current_all | {d5:.1f}% | {d10:.1f}% | {d20:.1f}% |".format(
            d5=100.0 * (safe_float(strict.get("new_recall_at_5")) - safe_float(current.get("new_recall_at_5"))),
            d10=100.0 * (safe_float(strict.get("new_recall_at_10")) - safe_float(current.get("new_recall_at_10"))),
            d20=100.0 * (safe_float(strict.get("new_recall_at_20")) - safe_float(current.get("new_recall_at_20"))),
        ),
        "| demand_entity_title - current_all | {d5:.1f}% | {d10:.1f}% | {d20:.1f}% |".format(
            d5=100.0 * (safe_float(demand.get("new_recall_at_5")) - safe_float(current.get("new_recall_at_5"))),
            d10=100.0 * (safe_float(demand.get("new_recall_at_10")) - safe_float(current.get("new_recall_at_10"))),
            d20=100.0 * (safe_float(demand.get("new_recall_at_20")) - safe_float(current.get("new_recall_at_20"))),
        ),
    ])

    best_rows = sorted(
        [row for row in variant_rows if str(row.get("policy")) == best],
        key=lambda row: safe_float(row.get("rank_improvement")),
        reverse=True,
    )[:12]
    lines.extend(["", "## Top Improvements For Best Policy", ""])
    for row in best_rows:
        lines.append(
            "- q{qid} `{gold}` rank {src} -> {rank} via `{entity}` from `{upstream}` ({reason})".format(
                qid=row.get("query_index"),
                gold=row.get("gold_title"),
                src=row.get("source_best_gold_rank"),
                rank=row.get("best_probe_rank"),
                entity=row.get("best_probe_entity"),
                upstream=row.get("best_upstream_title"),
                reason=row.get("best_probe_reason"),
            )
        )

    lines.extend([
        "",
        "## Files",
        "",
        f"- Variant rows: `{report_dir / 'variant_rows.csv'}`",
        f"- Policy summary: `{report_dir / 'policy_summary.csv'}`",
        f"- Policy comparison: `{report_dir / 'policy_comparison.csv'}`",
        f"- Full summary: `{report_dir / 'summary.json'}`",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe_dir)
    entity_output_path = Path(args.entity_outputs) if args.entity_outputs else probe_dir / "entity_outputs.jsonl"
    prompt_rows = read_jsonl(entity_output_path)
    entity_rows_by_query = entity_rows_by_query_from_outputs(prompt_rows)
    missing_rows = target_missing_gold_rows(
        audit_dir=args.audit_dir,
        dataset_label=MUSIQUE_LABEL,
        bucket=TARGET_BUCKET,
        query_primary_only=bool(args.query_primary_only),
    )
    if int(args.limit_queries) > 0:
        allowed = set(sorted({safe_int(row.get("query_index")) for row in missing_rows})[: int(args.limit_queries)])
        missing_rows = [row for row in missing_rows if safe_int(row.get("query_index")) in allowed]
    policies = [policy for policy in POLICIES if policy in set(args.policies)]
    variant_rows = build_policy_rows(
        policies=policies,
        missing_rows=missing_rows,
        entity_rows_by_query=entity_rows_by_query,
    )
    summary_rows, comparison_rows = summarize_policy_rows(variant_rows, policies=policies)
    comparison_rows = add_current_delta_counts(comparison_rows, variant_rows)
    metadata = {
        "dataset": MUSIQUE_LABEL,
        "base_dataset": MUSIQUE_DATASET,
        "target_bucket": TARGET_BUCKET,
        "query_primary_only": bool(args.query_primary_only),
        "missing_gold_titles": len(missing_rows),
        "queries": len({safe_int(row.get("query_index")) for row in missing_rows}),
        "prompt_rows": len(prompt_rows),
        "entity_outputs": sum(len(rows) for rows in entity_rows_by_query.values()),
        "policies": policies,
        "best_policy": best_policy(comparison_rows),
        "probe_dir": str(probe_dir),
        "entity_outputs_path": str(entity_output_path),
        "audit_dir": str(args.audit_dir),
        "report_dir": str(report_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {
        "metadata": metadata,
        "policy_comparison": comparison_rows,
        "policy_summary": summary_rows,
        "variant_rows": variant_rows,
    }
    write_csv(variant_rows, report_dir / "variant_rows.csv")
    write_csv(summary_rows, report_dir / "policy_summary.csv")
    write_csv(comparison_rows, report_dir / "policy_comparison.csv")
    write_json(payload, report_dir / "summary.json")
    (report_dir / "summary.md").write_text(
        build_markdown(
            metadata=metadata,
            comparison_rows=comparison_rows,
            summary_rows=summary_rows,
            variant_rows=variant_rows,
            report_dir=report_dir,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--probe_dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--entity_outputs", type=Path, default=None)
    parser.add_argument("--audit_dir", type=Path, default=CANDIDATE_AUDIT_DIR)
    parser.add_argument("--query_primary_only", action="store_true")
    parser.add_argument("--limit_queries", type=int, default=0)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
