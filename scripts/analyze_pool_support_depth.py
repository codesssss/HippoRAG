#!/usr/bin/env python3
"""Analyze where gold support appears inside widened external pools."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import string
from typing import Any, Sequence


def normalize_text(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def doc_title(doc: str) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def first_rank_for_support(
    *,
    gold_doc: str,
    gold_title: str,
    pool_docs: Sequence[str],
    pool_titles: Sequence[str],
) -> int | None:
    gold_doc_norm = normalize_text(gold_doc)
    gold_title_norm = normalize_text(gold_title or doc_title(gold_doc))
    for idx, doc in enumerate(pool_docs):
        if gold_doc_norm and normalize_text(doc) == gold_doc_norm:
            return idx + 1
    for idx, title in enumerate(pool_titles):
        if gold_title_norm and normalize_text(title) == gold_title_norm:
            return idx + 1
    for idx, doc in enumerate(pool_docs):
        if gold_title_norm and normalize_text(doc_title(doc)) == gold_title_norm:
            return idx + 1
    return None


def percentile(values: Sequence[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize_ranks(ranks: Sequence[int | None], *, pool_k: int) -> dict[str, Any]:
    present = [int(rank) for rank in ranks if rank is not None]
    missing = len(ranks) - len(present)
    return {
        "count": len(ranks),
        "present_count": len(present),
        "missing_count": missing,
        "present_fraction": len(present) / len(ranks) if ranks else 0.0,
        "median": percentile(present, 0.50),
        "p75": percentile(present, 0.75),
        "p90": percentile(present, 0.90),
        "mean": sum(present) / len(present) if present else None,
        "beyond_top20_fraction": sum(1 for rank in ranks if rank is None or rank > 20) / len(ranks) if ranks else 0.0,
        "within_top20_fraction": sum(1 for rank in ranks if rank is not None and rank <= 20) / len(ranks) if ranks else 0.0,
        "within_top50_fraction": sum(1 for rank in ranks if rank is not None and rank <= 50) / len(ranks) if ranks else 0.0,
        "within_top100_fraction": sum(1 for rank in ranks if rank is not None and rank <= 100) / len(ranks) if ranks else 0.0,
        "within_pool_fraction": sum(1 for rank in ranks if rank is not None and rank <= pool_k) / len(ranks) if ranks else 0.0,
    }


def coverage_by_query(rows: Sequence[dict[str, Any]], k: int) -> dict[str, float]:
    if not rows:
        return {
            "all_support_within_k": 0.0,
            "any_support_beyond_k_or_missing": 0.0,
            "no_support_within_k": 0.0,
        }
    all_within = 0
    any_beyond = 0
    none_within = 0
    for row in rows:
        ranks = list(row.get("support_ranks") or [])
        if ranks and all(rank is not None and int(rank) <= k for rank in ranks):
            all_within += 1
        if any(rank is None or int(rank) > k for rank in ranks):
            any_beyond += 1
        if not any(rank is not None and int(rank) <= k for rank in ranks):
            none_within += 1
    total = len(rows)
    return {
        "all_support_within_k": all_within / total,
        "any_support_beyond_k_or_missing": any_beyond / total,
        "no_support_within_k": none_within / total,
    }


def load_dataset_samples(dataset: str, dataset_dir: str) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / f"{dataset}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def paragraph_title_by_idx(sample: dict[str, Any]) -> dict[str, str]:
    return {
        str(para.get("idx")): str(para.get("title") or "")
        for para in (sample.get("paragraphs") or [])
        if para.get("idx") is not None
    }


def analyze_record(record: dict[str, Any], sample: dict[str, Any] | None = None) -> dict[str, Any]:
    pool_docs = [str(doc or "") for doc in (record.get("pool_docs") or [])]
    pool_titles = [str(title or "") for title in (record.get("pool_titles") or [])]
    gold_docs = [str(doc or "") for doc in (record.get("gold_docs") or [])]
    gold_titles = [str(title or "") for title in (record.get("gold_titles") or [])]
    if len(gold_titles) < len(gold_docs):
        gold_titles.extend(doc_title(doc) for doc in gold_docs[len(gold_titles):])

    ranks = [
        first_rank_for_support(gold_doc=gold_doc, gold_title=gold_titles[idx], pool_docs=pool_docs, pool_titles=pool_titles)
        for idx, gold_doc in enumerate(gold_docs)
    ]
    hop_ranks: list[dict[str, Any]] = []
    if sample:
        title_by_idx = paragraph_title_by_idx(sample)
        for hop_idx, step in enumerate(sample.get("question_decomposition") or [], start=1):
            support_idx = step.get("paragraph_support_idx")
            support_title = title_by_idx.get(str(support_idx), "")
            hop_rank = first_rank_for_support(
                gold_doc="",
                gold_title=support_title,
                pool_docs=pool_docs,
                pool_titles=pool_titles,
            )
            hop_ranks.append(
                {
                    "hop": hop_idx,
                    "paragraph_support_idx": support_idx,
                    "title": support_title,
                    "rank": hop_rank,
                }
            )

    return {
        "query_idx": record.get("query_idx"),
        "question": record.get("question"),
        "support_ranks": ranks,
        "max_support_rank": max([rank for rank in ranks if rank is not None], default=None),
        "missing_support_count": sum(1 for rank in ranks if rank is None),
        "hop_support_ranks": hop_ranks,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Pool Support Depth Analysis",
        "",
        f"- Pool: `{payload['pool_json']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Records: `{payload['n']}`",
        f"- Pool K: `{payload['pool_k']}`",
        "",
        "## Overall Support Rank",
        "",
    ]
    summary = payload["overall_support_rank"]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Query Coverage", "", "| K | All Support Within K | Any Support Beyond/Missing | No Support Within K |", "|---:|---:|---:|---:|"])
    for k, stats in payload["query_coverage"].items():
        lines.append(
            f"| {k} | `{stats['all_support_within_k']:.4f}` | "
            f"`{stats['any_support_beyond_k_or_missing']:.4f}` | `{stats['no_support_within_k']:.4f}` |"
        )
    if payload.get("hop_support_rank"):
        lines.extend(["", "## Hop Support Rank", ""])
        for hop, stats in sorted(payload["hop_support_rank"].items(), key=lambda item: int(item[0])):
            lines.append(f"- hop {hop}: median=`{stats['median']}` p75=`{stats['p75']}` beyond_top20=`{stats['beyond_top20_fraction']:.4f}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    pool_path = Path(args.pool_json)
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    dataset = args.dataset or str(payload.get("dataset") or "")
    records = list(payload.get("records") or [])
    if args.limit > 0:
        records = records[: args.limit]
    samples = load_dataset_samples(dataset, args.dataset_dir) if dataset else []

    rows = []
    for idx, record in enumerate(records):
        sample = samples[idx] if idx < len(samples) else None
        rows.append(analyze_record(record, sample=sample))

    support_ranks = [rank for row in rows for rank in (row.get("support_ranks") or [])]
    hop_rank_by_hop: dict[str, list[int | None]] = defaultdict(list)
    for row in rows:
        for hop_row in row.get("hop_support_ranks") or []:
            hop_rank_by_hop[str(hop_row["hop"])].append(hop_row.get("rank"))

    out_payload = {
        "pool_json": str(pool_path),
        "dataset": dataset,
        "n": len(rows),
        "pool_k": payload.get("pool_k"),
        "overall_support_rank": summarize_ranks(support_ranks, pool_k=int(payload.get("pool_k") or 100)),
        "query_coverage": {str(k): coverage_by_query(rows, k) for k in (20, 50, 100)},
        "hop_support_rank": {
            hop: summarize_ranks(ranks, pool_k=int(payload.get("pool_k") or 100))
            for hop, ranks in hop_rank_by_hop.items()
        },
        "rows": rows,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(out_payload), encoding="utf-8")
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
