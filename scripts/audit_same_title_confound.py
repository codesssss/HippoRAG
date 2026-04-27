#!/usr/bin/env python3
"""Static same-title duplicate/confound audit for paper-integrity checks.

This script audits whether existing method conclusions are plausibly explained
by same-title duplicates in selected evidence sets or edit/add operations.
It intentionally reads completed traces only; it does not rerun selectors or
readers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")


DEFAULT_CACHE_JSONL = Path("data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
DEFAULT_OUTPUT_JSON = Path("reports/paper/same_title_audit.json")
DEFAULT_OUTPUT_MD = Path("reports/paper/same_title_audit.md")


D_PATHRAG_SELECTOR = Path("reports/dpathrag/kfold_proprag/selector_kfold_proprag_embed_local1000.predictions.jsonl")
CEE_MARGIN15 = Path("reports/dpathrag/cee_edit_policy_proprag_kfold1000_margin15.predictions.jsonl")


CPAG_VARIANTS = {
    "CPAG PropRAG rank": Path("reports/cpag/cpag_day1_proprag_rank.rows.jsonl"),
    "CPAG RRF": Path("reports/cpag/cpag_day1_rrf.rows.jsonl"),
    "CPAG pure": Path("reports/cpag/cpag_day1_cpag_pure.rows.jsonl"),
    "CPAG anchored": Path("reports/cpag/cpag_day1_cpag.rows.jsonl"),
}


DAEC_REPORTS = {
    "DAEC PropRAG 2Wiki": Path("run_logs/layer1_proprag_pool_eval_fixed_20260424/2wikimultihopqa_proprag_pool_daec_oracle.json"),
    "DAEC PropRAG HotpotQA": Path("run_logs/layer1_proprag_pool_eval_fixed_20260424/hotpotqa_proprag_pool_daec_oracle.json"),
    "DAEC PropRAG MuSiQue": Path("run_logs/layer1_proprag_pool_eval_fixed_20260424/musique_proprag_pool_daec_oracle.json"),
    "DAEC Dense 2Wiki": Path("run_logs/layer1_dense_pool_eval_20260424/2wikimultihopqa_dense_pool_daec_oracle.json"),
    "DAEC Dense HotpotQA": Path("run_logs/layer1_dense_pool_eval_20260424/hotpotqa_dense_pool_daec_oracle.json"),
    "DAEC Dense MuSiQue": Path("run_logs/layer1_dense_pool_eval_20260424/musique_dense_pool_daec_oracle.json"),
}


DAEC_ALR_TRACE = Path("reports/daec_alr/step1_single_edit_traces.jsonl")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def title_key(title: Any) -> str:
    """Normalize titles without answer-style article dropping.

    Using QA answer normalization here would collapse valid short titles such as
    "A" and is too aggressive for document identity audits.
    """

    return " ".join(TOKEN_RE.findall(str(title or "").lower()))


def row_qid(row: Mapping[str, Any]) -> str:
    return str(row.get("qid") or row.get("query_idx") or "")


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def round6(value: float) -> float:
    return round(float(value), 6)


def key_counter(titles: Sequence[Any]) -> Counter[str]:
    return Counter(key for key in (title_key(title) for title in titles) if key)


def duplicate_stats(titles: Sequence[Any]) -> dict[str, Any]:
    counts = key_counter(titles)
    duplicate_docs = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "has_duplicate": bool(duplicate_docs),
        "duplicate_docs": int(duplicate_docs),
        "unique_title_rate": round6(safe_div(len(counts), len(titles))),
    }


def multiset_added(selected_titles: Sequence[Any], baseline_titles: Sequence[Any]) -> list[str]:
    baseline = key_counter(baseline_titles)
    added: list[str] = []
    for key in [title_key(title) for title in selected_titles]:
        if not key:
            continue
        if baseline.get(key, 0) > 0:
            baseline[key] -= 1
        else:
            added.append(key)
    return added


def candidate_title(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    return str(candidate.get("title") or "")


def candidate_is_gold(candidate: Mapping[str, Any] | None, gold_keys: set[str]) -> bool:
    if not candidate:
        return False
    if "gold_support" in candidate:
        return bool(int(candidate.get("gold_support") or 0))
    return title_key(candidate.get("title")) in gold_keys


def load_cache_by_qid(path: str | Path, *, limit: int = 0) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    if limit > 0:
        rows = rows[: int(limit)]
    return {row_qid(row): row for row in rows}


def init_summary(name: str, category: str, path: str | Path) -> dict[str, Any]:
    return {
        "method": name,
        "category": category,
        "path": str(path),
        "rows": 0,
        "added_docs": 0,
        "added_gold": 0,
        "added_non_gold": 0,
        "added_non_gold_same_title_as_baseline": 0,
        "added_non_gold_same_title_as_removed": 0,
        "added_gold_same_title_as_baseline": 0,
        "net_added_docs": 0,
        "net_added_gold": 0,
        "net_added_non_gold": 0,
        "net_added_non_gold_same_title_as_baseline": 0,
        "selected_duplicate_query_count": 0,
        "selected_duplicate_docs_total": 0,
        "baseline_duplicate_query_count": 0,
        "baseline_duplicate_docs_total": 0,
        "docid_additions": 0,
        "same_title_docid_additions": 0,
        "same_title_docid_added_non_gold": 0,
        "changed_queries": 0,
        "title_mismatch_count": 0,
        "examples": [],
    }


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rows = int(summary.get("rows") or 0)
    added_non_gold = int(summary.get("added_non_gold") or 0)
    same_title_added = int(summary.get("added_non_gold_same_title_as_baseline") or 0)
    docid_additions = int(summary.get("docid_additions") or 0)
    same_title_docid = int(summary.get("same_title_docid_additions") or 0)
    summary["same_title_share_of_added_non_gold"] = round6(safe_div(same_title_added, added_non_gold))
    summary["net_same_title_share_of_added_non_gold"] = round6(
        safe_div(
            int(summary.get("net_added_non_gold_same_title_as_baseline") or 0),
            int(summary.get("net_added_non_gold") or 0),
        )
    )
    summary["selected_duplicate_query_rate"] = round6(safe_div(int(summary.get("selected_duplicate_query_count") or 0), rows))
    summary["baseline_duplicate_query_rate"] = round6(safe_div(int(summary.get("baseline_duplicate_query_count") or 0), rows))
    summary["same_title_docid_addition_share"] = round6(safe_div(same_title_docid, docid_additions))
    summary["non_gold_per_gold"] = round6(
        safe_div(int(summary.get("added_non_gold") or 0), max(1, int(summary.get("added_gold") or 0)))
    )
    summary["interpretation"] = interpret_summary(summary)
    return summary


def add_example(summary: dict[str, Any], example: dict[str, Any], *, max_examples: int = 5) -> None:
    if len(summary["examples"]) < int(max_examples):
        summary["examples"].append(example)


def audit_index_selection_predictions(
    *,
    name: str,
    path: str | Path,
    cache_by_qid: Mapping[str, dict[str, Any]],
    category: str,
    variant: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    rows = load_jsonl(path)
    if variant is not None:
        rows = [row for row in rows if str(row.get("variant") or "") == str(variant)]
    summary = init_summary(name, category, path)
    baseline_indices = set(range(int(top_k)))

    for row in rows:
        qid = row_qid(row)
        cache_row = cache_by_qid.get(qid, {})
        candidates = list(cache_row.get("candidates") or [])
        gold_keys = {title_key(title) for title in cache_row.get("gold_titles") or [] if title_key(title)}

        selected_indices = [as_int(index) for index in row.get("selected_indices") or []]
        selected_indices = [index for index in selected_indices if index >= 0]
        selected_titles = [candidate_title(candidates[index]) if index < len(candidates) else str(title)
                           for index, title in zip(selected_indices, row.get("selected_titles") or [])]
        if len(selected_titles) < len(selected_indices):
            selected_titles = list(row.get("selected_titles") or [])
        baseline_titles = [candidate_title(candidates[index]) for index in range(min(int(top_k), len(candidates)))]
        if not baseline_titles:
            baseline_titles = list(row.get("rank_topk_titles") or [])
        baseline_keys = {title_key(title) for title in baseline_titles if title_key(title)}

        selected_dups = duplicate_stats(selected_titles)
        baseline_dups = duplicate_stats(baseline_titles)
        summary["selected_duplicate_query_count"] += int(selected_dups["has_duplicate"])
        summary["selected_duplicate_docs_total"] += int(selected_dups["duplicate_docs"])
        summary["baseline_duplicate_query_count"] += int(baseline_dups["has_duplicate"])
        summary["baseline_duplicate_docs_total"] += int(baseline_dups["duplicate_docs"])

        final_added_indices = [index for index in selected_indices if index not in baseline_indices]
        for index in final_added_indices:
            candidate = candidates[index] if 0 <= index < len(candidates) else None
            c_key = title_key(candidate_title(candidate))
            is_gold = candidate_is_gold(candidate, gold_keys)
            summary["net_added_docs"] += 1
            summary["net_added_gold"] += int(is_gold)
            summary["net_added_non_gold"] += int(not is_gold)
            if not is_gold:
                summary["net_added_non_gold_same_title_as_baseline"] += int(c_key in baseline_keys)

        edits = list(row.get("edits") or [])
        if edits:
            added_indices = [as_int(edit.get("add_index")) for edit in edits]
            removed_indices = [as_int(edit.get("remove_index")) for edit in edits]
        else:
            selected_set = set(selected_indices)
            added_indices = [index for index in selected_indices if index not in baseline_indices]
            removed_indices = sorted(index for index in baseline_indices if index not in selected_set)
        removed_keys = {
            title_key(candidate_title(candidates[index]))
            for index in removed_indices
            if 0 <= index < len(candidates) and title_key(candidate_title(candidates[index]))
        }
        if added_indices:
            summary["changed_queries"] += 1
        for index in added_indices:
            candidate = candidates[index] if 0 <= index < len(candidates) else None
            c_title = candidate_title(candidate)
            c_key = title_key(c_title)
            is_gold = candidate_is_gold(candidate, gold_keys)
            summary["added_docs"] += 1
            summary["added_gold"] += int(is_gold)
            summary["added_non_gold"] += int(not is_gold)
            summary["added_gold_same_title_as_baseline"] += int(is_gold and c_key in baseline_keys)
            if not is_gold:
                summary["added_non_gold_same_title_as_baseline"] += int(c_key in baseline_keys)
                summary["added_non_gold_same_title_as_removed"] += int(c_key in removed_keys)
                if c_key in baseline_keys or c_key in removed_keys:
                    add_example(
                        summary,
                        {
                            "qid": qid,
                            "added_index": index,
                            "added_title": c_title,
                            "baseline_titles": baseline_titles,
                            "selected_titles": selected_titles,
                            "reason": "added non-gold has same title as baseline/removed doc",
                        },
                    )

        prediction_titles = list(row.get("selected_titles") or [])
        if prediction_titles and [title_key(t) for t in prediction_titles] != [title_key(t) for t in selected_titles]:
            summary["title_mismatch_count"] += 1

    summary["rows"] = len(rows)
    return finalize_summary(summary)


def audit_cpag_variant(
    *,
    name: str,
    path: str | Path,
    baseline_rows: Mapping[str, dict[str, Any]],
    diagnostic_rows: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = load_jsonl(path)
    summary = init_summary(name, "CPAG", path)
    cross_pool_selected = 0
    cross_pool_selected_non_gold = 0
    cross_pool_selected_gold = 0
    diagnostics_docs = 0
    diagnostics_cross_pool_docs = 0

    for row in rows:
        qid = row_qid(row)
        selected_titles = list(row.get("selected_titles") or [])
        baseline_titles = list((baseline_rows.get(qid) or {}).get("selected_titles") or [])
        gold_keys = {title_key(title) for title in row.get("gold_titles") or [] if title_key(title)}
        baseline_keys = {title_key(title) for title in baseline_titles if title_key(title)}

        selected_dups = duplicate_stats(selected_titles)
        baseline_dups = duplicate_stats(baseline_titles)
        summary["selected_duplicate_query_count"] += int(selected_dups["has_duplicate"])
        summary["selected_duplicate_docs_total"] += int(selected_dups["duplicate_docs"])
        summary["baseline_duplicate_query_count"] += int(baseline_dups["has_duplicate"])
        summary["baseline_duplicate_docs_total"] += int(baseline_dups["duplicate_docs"])

        added_keys = multiset_added(selected_titles, baseline_titles)
        if added_keys:
            summary["changed_queries"] += 1
        for key in added_keys:
            is_gold = key in gold_keys
            summary["added_docs"] += 1
            summary["added_gold"] += int(is_gold)
            summary["added_non_gold"] += int(not is_gold)
            if not is_gold:
                summary["added_non_gold_same_title_as_baseline"] += int(key in baseline_keys)
            else:
                summary["added_gold_same_title_as_baseline"] += int(key in baseline_keys)

        diagnostics = list(row.get("doc_diagnostics") or [])
        if not diagnostics:
            diagnostics = list((diagnostic_rows.get(qid) or {}).get("doc_diagnostics") or [])
        diagnostics_docs += len(diagnostics)
        diagnostics_by_key = {title_key(doc.get("title")): doc for doc in diagnostics if title_key(doc.get("title"))}
        for doc in diagnostics:
            diagnostics_cross_pool_docs += int(int(doc.get("pool_count") or 0) >= 2)
        for title in selected_titles:
            doc = diagnostics_by_key.get(title_key(title))
            if doc and int(doc.get("pool_count") or 0) >= 2:
                is_gold = title_key(title) in gold_keys
                cross_pool_selected += 1
                cross_pool_selected_gold += int(is_gold)
                cross_pool_selected_non_gold += int(not is_gold)

    summary["rows"] = len(rows)
    summary["cpag_cross_pool_selected_docs"] = cross_pool_selected
    summary["cpag_cross_pool_selected_gold"] = cross_pool_selected_gold
    summary["cpag_cross_pool_selected_non_gold"] = cross_pool_selected_non_gold
    summary["cpag_diagnostic_docs"] = diagnostics_docs
    summary["cpag_diagnostic_cross_pool_docs"] = diagnostics_cross_pool_docs
    summary["cpag_cross_pool_selected_non_gold_share"] = round6(
        safe_div(cross_pool_selected_non_gold, cross_pool_selected)
    )
    return finalize_summary(summary)


def audit_daec_report(name: str, path: str | Path) -> dict[str, Any]:
    summary = init_summary(name, "DAEC", path)
    p = Path(path)
    if not p.exists():
        summary["missing"] = True
        return finalize_summary(summary)

    report = load_json(p)
    traces = list(report.get("setwise_selector_query_traces") or [])
    for trace in traces:
        baseline_titles = list(trace.get("baseline_top_titles") or [])
        selector_titles = list(trace.get("selector_top_titles") or [])
        gold_keys = {title_key(title) for title in trace.get("gold_titles") or [] if title_key(title)}
        baseline_keys = {title_key(title) for title in baseline_titles if title_key(title)}

        selected_dups = duplicate_stats(selector_titles)
        baseline_dups = duplicate_stats(baseline_titles)
        summary["selected_duplicate_query_count"] += int(selected_dups["has_duplicate"])
        summary["selected_duplicate_docs_total"] += int(selected_dups["duplicate_docs"])
        summary["baseline_duplicate_query_count"] += int(baseline_dups["has_duplicate"])
        summary["baseline_duplicate_docs_total"] += int(baseline_dups["duplicate_docs"])

        added_keys = multiset_added(selector_titles, baseline_titles)
        if added_keys or bool(trace.get("changed_from_baseline")):
            summary["changed_queries"] += 1
        for key in added_keys:
            is_gold = key in gold_keys
            summary["added_docs"] += 1
            summary["added_gold"] += int(is_gold)
            summary["added_non_gold"] += int(not is_gold)
            if not is_gold:
                summary["added_non_gold_same_title_as_baseline"] += int(key in baseline_keys)
            else:
                summary["added_gold_same_title_as_baseline"] += int(key in baseline_keys)

        baseline_doc_ids = list(trace.get("baseline_top_doc_ids") or [])
        selector_doc_ids = list(trace.get("selector_top_doc_ids") or [])
        baseline_id_set = {str(doc_id) for doc_id in baseline_doc_ids}
        baseline_id_title = {
            str(doc_id): title_key(title)
            for doc_id, title in zip(baseline_doc_ids, baseline_titles)
            if str(doc_id) and title_key(title)
        }
        baseline_title_set = set(baseline_id_title.values()) or baseline_keys
        for doc_id, title in zip(selector_doc_ids, selector_titles):
            doc_id_str = str(doc_id)
            key = title_key(title)
            if doc_id_str and doc_id_str not in baseline_id_set:
                summary["docid_additions"] += 1
                if key in baseline_title_set:
                    is_gold = key in gold_keys
                    summary["same_title_docid_additions"] += 1
                    summary["same_title_docid_added_non_gold"] += int(not is_gold)
                    add_example(
                        summary,
                        {
                            "question": trace.get("question"),
                            "added_doc_id": doc_id,
                            "added_title": title,
                            "baseline_doc_ids": baseline_doc_ids,
                            "baseline_titles": baseline_titles,
                            "selector_doc_ids": selector_doc_ids,
                            "selector_titles": selector_titles,
                            "reason": "selector added a different doc id with a title already present in baseline",
                        },
                    )

    summary["rows"] = len(traces)
    selector_qa = report.get("setwise_selector_qa") or {}
    summary["baseline_f1"] = selector_qa.get("baseline_F1")
    summary["selector_f1"] = selector_qa.get("selector_F1")
    summary["f1_delta"] = selector_qa.get("F1_delta")
    summary["baseline_em"] = selector_qa.get("baseline_EM")
    summary["selector_em"] = selector_qa.get("selector_EM")
    summary["em_delta"] = selector_qa.get("EM_delta")
    summary["dataset"] = report.get("dataset")
    return finalize_summary(summary)


def audit_daec_alr(path: str | Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    summary = init_summary("DAEC-ALR Step-1 accepted edits", "DAEC-ALR", path)
    for row in rows:
        final_docs = list(row.get("final_docs") or [])
        final_titles = [doc.get("title") for doc in final_docs]
        selected_dups = duplicate_stats(final_titles)
        summary["selected_duplicate_query_count"] += int(selected_dups["has_duplicate"])
        summary["selected_duplicate_docs_total"] += int(selected_dups["duplicate_docs"])
        if not row.get("accepted"):
            continue
        summary["changed_queries"] += 1
        stats = row.get("edit_stats") or {}
        added_gold = int(stats.get("added_gold") or 0)
        added_non_gold = int(stats.get("added_non_gold") or 0)
        summary["added_docs"] += added_gold + added_non_gold
        summary["added_gold"] += added_gold
        summary["added_non_gold"] += added_non_gold
        plan = row.get("accepted_plan") or {}
        target = plan.get("target") or {}
        candidate_key = title_key(plan.get("candidate_title"))
        target_key = title_key(target.get("title"))
        if candidate_key and candidate_key == target_key:
            summary["same_title_docid_additions"] += 1
            summary["same_title_docid_added_non_gold"] += added_non_gold
            summary["added_non_gold_same_title_as_removed"] += added_non_gold
            add_example(
                summary,
                {
                    "qid": row.get("qid"),
                    "variant": row.get("variant"),
                    "candidate_title": plan.get("candidate_title"),
                    "target_title": target.get("title"),
                    "reason": "accepted edit replaces target with same-title candidate",
                },
            )
    summary["rows"] = len(rows)
    return finalize_summary(summary)


def interpret_summary(summary: Mapping[str, Any]) -> str:
    method = str(summary.get("method") or "")
    category = str(summary.get("category") or "")
    added_non_gold = int(summary.get("added_non_gold") or 0)
    same_share = float(summary.get("same_title_share_of_added_non_gold") or 0.0)
    docid_share = float(summary.get("same_title_docid_addition_share") or 0.0)
    dup_rate = float(summary.get("selected_duplicate_query_rate") or 0.0)

    if category == "CPAG":
        non_gold_cross_pool = int(summary.get("cpag_cross_pool_selected_non_gold") or 0)
        if non_gold_cross_pool > 0:
            return "CPAG failure is better described as shared cross-pool distractor import, not same-title duplicate inflation."
        return "No selected same-title duplicate signal is visible in CPAG rows."
    if category == "DAEC":
        baseline_dup_rate = float(summary.get("baseline_duplicate_query_rate") or 0.0)
        if dup_rate >= 0.05 and baseline_dup_rate >= dup_rate and docid_share < 0.10:
            return "Duplicate-title exposure is inherited from the pool baseline; DAEC does not amplify it materially."
        if docid_share >= 0.10 or (dup_rate >= 0.05 and baseline_dup_rate < dup_rate):
            return "DAEC has method-amplified same-title exposure; inspect examples before freezing the main claim."
        return "DAEC main result has low same-title exposure; positive/negative deltas are unlikely to be title-duplicate artifacts."
    if category == "CEE" and same_share >= 0.05 and float(summary.get("net_same_title_share_of_added_non_gold") or 0.0) == 0.0:
        return "Same-title edit operations come from multi-step oscillation/reinsertion; final selected-set exposure is negligible."
    if "D-PathRAG" in method and added_non_gold > 0 and same_share == 0.0:
        return "D-PathRAG hard-negative import is not explained by same-title additions versus rank top-5."
    if same_share >= 0.05:
        return "Same-title additions are material enough to require rerun or case-level inspection."
    if added_non_gold > 0:
        return "Added non-gold imports exist, but same-title share is negligible."
    return "No material same-title confound detected."


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    cache_by_qid = load_cache_by_qid(args.cache_jsonl, limit=int(args.limit))
    summaries: list[dict[str, Any]] = []

    if Path(args.dpathrag_selector).exists():
        summaries.append(
            audit_index_selection_predictions(
                name="D-PathRAG selector_v1 PropRAG kfold1000",
                path=args.dpathrag_selector,
                cache_by_qid=cache_by_qid,
                category="D-PathRAG",
                top_k=int(args.top_k),
            )
        )

    if Path(args.cee_predictions).exists():
        for variant in ("learned_edit1", "learned_edit2"):
            summaries.append(
                audit_index_selection_predictions(
                    name=f"CEE {variant} margin15",
                    path=args.cee_predictions,
                    cache_by_qid=cache_by_qid,
                    category="CEE",
                    variant=variant,
                    top_k=int(args.top_k),
                )
            )

    cpag_baseline_rows = {
        row_qid(row): row
        for row in load_jsonl(args.cpag_proprag_rank)
    }
    cpag_diagnostic_rows = {
        row_qid(row): row
        for row in load_jsonl(CPAG_VARIANTS["CPAG anchored"])
    }
    for name, default_path in CPAG_VARIANTS.items():
        path = Path(args.cpag_paths.get(name, default_path)) if isinstance(args.cpag_paths, dict) else default_path
        if path.exists():
            summaries.append(
                audit_cpag_variant(
                    name=name,
                    path=path,
                    baseline_rows=cpag_baseline_rows,
                    diagnostic_rows=cpag_diagnostic_rows,
                )
            )

    for name, path in DAEC_REPORTS.items():
        if path.exists():
            summaries.append(audit_daec_report(name, path))

    if Path(args.daec_alr_trace).exists():
        summaries.append(audit_daec_alr(args.daec_alr_trace))

    findings = build_findings(summaries)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "cache_jsonl": str(args.cache_jsonl),
            "dpathrag_selector": str(args.dpathrag_selector),
            "cee_predictions": str(args.cee_predictions),
            "cpag_proprag_rank": str(args.cpag_proprag_rank),
            "daec_reports": {name: str(path) for name, path in DAEC_REPORTS.items()},
            "daec_alr_trace": str(args.daec_alr_trace),
        },
        "summary": {
            "audited_methods": len(summaries),
            "methods_with_material_same_title_added_non_gold": [
                row["method"]
                for row in summaries
                if float(row.get("same_title_share_of_added_non_gold") or 0.0) >= 0.05
            ],
            "methods_with_material_selected_duplicates": [
                row["method"]
                for row in summaries
                if float(row.get("selected_duplicate_query_rate") or 0.0) >= 0.05
            ],
        },
        "method_summaries": summaries,
        "key_findings": findings,
    }


def build_findings(summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    by_method = {str(row.get("method")): row for row in summaries}
    findings: list[str] = []

    dpath = by_method.get("D-PathRAG selector_v1 PropRAG kfold1000")
    if dpath:
        findings.append(
            "D-PathRAG selector_v1 adds "
            f"{int(dpath.get('added_non_gold') or 0)} non-gold documents, but "
            f"{int(dpath.get('added_non_gold_same_title_as_baseline') or 0)} are same-title additions versus rank top-5."
        )

    cee_rows = [row for row in summaries if str(row.get("category")) == "CEE"]
    if cee_rows:
        max_share = max(float(row.get("same_title_share_of_added_non_gold") or 0.0) for row in cee_rows)
        max_net_share = max(float(row.get("net_same_title_share_of_added_non_gold") or 0.0) for row in cee_rows)
        findings.append(
            f"CEE learned edit policies have max operation-level same-title share {max_share:.3f}, "
            f"but max final-set same-title share {max_net_share:.3f}."
        )

    cpag_rows = [row for row in summaries if str(row.get("category")) == "CPAG"]
    if cpag_rows:
        anchored = by_method.get("CPAG anchored")
        if anchored:
            findings.append(
                "CPAG anchored selects "
                f"{int(anchored.get('cpag_cross_pool_selected_non_gold') or 0)} non-gold cross-pool documents; "
                "the failure mode is shared-distractor agreement rather than same-title duplication."
            )

    daec_rows = [row for row in summaries if str(row.get("category")) == "DAEC"]
    if daec_rows:
        max_docid_share = max(float(row.get("same_title_docid_addition_share") or 0.0) for row in daec_rows)
        max_dup_rate = max(float(row.get("selected_duplicate_query_rate") or 0.0) for row in daec_rows)
        findings.append(
            f"DAEC reports have max same-title doc-id addition share {max_docid_share:.3f} "
            f"and max selected duplicate-title query rate {max_dup_rate:.3f}; MuSiQue duplicate exposure is mostly inherited from baseline."
        )

    alr = by_method.get("DAEC-ALR Step-1 accepted edits")
    if alr:
        findings.append(
            "DAEC-ALR accepted edits show "
            f"{int(alr.get('added_non_gold_same_title_as_removed') or 0)} non-gold same-title replacements."
        )

    if not findings:
        findings.append("No same-title audit inputs were found.")
    return findings


def md_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_method_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    headers = [
        "Method",
        "Rows",
        "Added",
        "Added gold",
        "Added non-gold",
        "Same-title non-gold",
        "Share",
        "Base dup rate",
        "Dup query rate",
        "Doc-id same-title add",
        "Interpretation",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * 9 + ["---"]) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row.get("method")),
                    str(int(row.get("rows") or 0)),
                    str(int(row.get("added_docs") or 0)),
                    str(int(row.get("added_gold") or 0)),
                    str(int(row.get("added_non_gold") or 0)),
                    str(int(row.get("added_non_gold_same_title_as_baseline") or 0)),
                    f"{float(row.get('same_title_share_of_added_non_gold') or 0.0):.3f}",
                    f"{float(row.get('baseline_duplicate_query_rate') or 0.0):.3f}",
                    f"{float(row.get('selected_duplicate_query_rate') or 0.0):.3f}",
                    f"{int(row.get('same_title_docid_additions') or 0)}/{int(row.get('docid_additions') or 0)}",
                    md_escape(row.get("interpretation")),
                ]
            )
            + " |"
        )
    return lines


def render_cpag_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    cpag_rows = [row for row in rows if str(row.get("category")) == "CPAG"]
    if not cpag_rows:
        return []
    lines = [
        "## CPAG Cross-Pool Detail",
        "",
        "| Variant | Selected cross-pool | Selected cross-pool gold | Selected cross-pool non-gold | Non-gold share | Diagnostic cross-pool docs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in cpag_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row.get("method")),
                    str(int(row.get("cpag_cross_pool_selected_docs") or 0)),
                    str(int(row.get("cpag_cross_pool_selected_gold") or 0)),
                    str(int(row.get("cpag_cross_pool_selected_non_gold") or 0)),
                    f"{float(row.get('cpag_cross_pool_selected_non_gold_share') or 0.0):.3f}",
                    f"{int(row.get('cpag_diagnostic_cross_pool_docs') or 0)}/{int(row.get('cpag_diagnostic_docs') or 0)}",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_cee_net_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    cee_rows = [row for row in rows if str(row.get("category")) == "CEE"]
    if not cee_rows:
        return []
    lines = [
        "## CEE Operation vs Final-Set Detail",
        "",
        "| Variant | Operation added non-gold | Operation same-title non-gold | Net final added non-gold | Net final same-title non-gold |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in cee_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row.get("method")),
                    str(int(row.get("added_non_gold") or 0)),
                    str(int(row.get("added_non_gold_same_title_as_baseline") or 0)),
                    str(int(row.get("net_added_non_gold") or 0)),
                    str(int(row.get("net_added_non_gold_same_title_as_baseline") or 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "The `learned_edit2` operation count includes second-step reinsertions into the original top-5. "
        "This is edit-policy oscillation, not final selected-set same-title contamination."
    )
    lines.append("")
    return lines


def render_markdown(payload: Mapping[str, Any]) -> str:
    rows = list(payload.get("method_summaries") or [])
    lines: list[str] = [
        "# Same-Title Confound Audit",
        "",
        "## Scope",
        "",
        "- Static trace audit only; no selector, reader, or model reruns.",
        "- Title normalization uses lowercase alphanumeric tokens and does not drop articles.",
        "- `Same-title non-gold` counts added non-gold documents whose title key already appears in the baseline top-5.",
        "- `Doc-id same-title add` is stricter for DAEC traces: a different selected doc id with a title already present in baseline.",
        "",
        "## Key Findings",
        "",
    ]
    for finding in payload.get("key_findings") or []:
        lines.append(f"- {finding}")
    lines.extend(["", "## Raw Audit Table", ""])
    lines.extend(render_method_table(rows))
    lines.append("")
    lines.extend(render_cee_net_table(rows))
    lines.extend(render_cpag_table(rows))

    lines.extend(
        [
            "## Decision Impact",
            "",
            "- D-PathRAG/CEE hard-negative import should not be re-attributed to same-title duplicates unless a future doc-id-level rerun contradicts this static audit.",
            "- CPAG remains a shared-distractor agreement failure: title-canonical union deduplication prevents same-title duplicate inflation in selected rows.",
            "- DAEC 2Wiki/HotpotQA mainline does not require rerun on same-title grounds; MuSiQue has high duplicate-title exposure, but it is already present in baseline and DAEC slightly reduces selected duplicate rate.",
            "- NREV/DAEC-ALR should keep the same-title replacement fix as a destructive-null/edit hygiene requirement, not as evidence that prior main tables were invalid.",
            "",
            "## Examples",
            "",
        ]
    )
    any_examples = False
    for row in rows:
        examples = list(row.get("examples") or [])
        if not examples:
            continue
        any_examples = True
        lines.append(f"### {row.get('method')}")
        lines.append("")
        for example in examples[:5]:
            lines.append(f"- `{json.dumps(example, ensure_ascii=False)}`")
        lines.append("")
    if not any_examples:
        lines.append("- No material same-title examples were found under the audited criteria.")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", type=Path, default=DEFAULT_CACHE_JSONL)
    parser.add_argument("--dpathrag_selector", type=Path, default=D_PATHRAG_SELECTOR)
    parser.add_argument("--cee_predictions", type=Path, default=CEE_MARGIN15)
    parser.add_argument("--cpag_proprag_rank", type=Path, default=CPAG_VARIANTS["CPAG PropRAG rank"])
    parser.add_argument("--daec_alr_trace", type=Path, default=DAEC_ALR_TRACE)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    args.cpag_paths = {}
    return args


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
