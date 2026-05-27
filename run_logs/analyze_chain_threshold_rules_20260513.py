#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DIAG = Path(
    "/mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/"
    "reports_chain_diagnostic_gainloss/etv4_musique_chain_consistency_diagnostic.json"
)
REPORT = Path(
    "/mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/"
    "musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json"
)


def ints(values: Iterable[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def hits(docs: Sequence[int], gold_docs: Sequence[int]) -> int:
    return len(set(ints(docs[:5])) & set(ints(gold_docs)))


def all_gold(docs: Sequence[int], gold_docs: Sequence[int]) -> int:
    gold = set(ints(gold_docs))
    return int(bool(gold) and gold <= set(ints(docs[:5])))


def annotate_rows() -> list[dict[str, Any]]:
    diag = json.loads(DIAG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    by_qid = {
        int(row.get("query_index", idx)): row
        for idx, row in enumerate(report.get("rows", []) or [])
    }
    rows: list[dict[str, Any]] = []
    for row in diag.get("query_rows", []) or []:
        out = dict(row)
        report_row = by_qid[int(row["query_index"])]
        route_trace = report_row.get("route_trace", {}) or {}
        retrieval = route_trace.get("retrieval", {}) or {}
        local_graph = retrieval.get("local_graph", {}) or {}
        candidate_universe = route_trace.get("candidate_universe", {}) or {}
        symbolic = set(
            ints(local_graph.get("symbolic_seed_doc_indices", []) or [])
            or ints(candidate_universe.get("agsto_symbolic_seed_doc_indices", []) or [])
        )
        textual = set(ints(local_graph.get("textual_seed_doc_indices", []) or []))
        candidate_order = ints(candidate_universe.get("candidate_doc_indices", []) or [])
        candidate_pos = {doc: idx + 1 for idx, doc in enumerate(candidate_order)}
        annotated_docs: list[dict[str, Any]] = []
        for info in row.get("inserted_docs", []) or []:
            doc = int(info["doc_index"])
            annotated = dict(info)
            annotated["candidate_pos"] = int(candidate_pos.get(doc, 999))
            if doc in symbolic and doc in textual:
                annotated["origin"] = "symbolic+textual_seed"
            elif doc in symbolic:
                annotated["origin"] = "symbolic_seed_only"
            elif doc in textual:
                annotated["origin"] = "textual_seed_tail"
            else:
                annotated["origin"] = "graph_expanded"
            annotated_docs.append(annotated)
        out["inserted_docs"] = annotated_docs
        rows.append(out)
    return rows


Predicate = Callable[[Mapping[str, Any]], bool]


def cf_top5(row: Mapping[str, Any], pred: Predicate) -> tuple[int, ...]:
    selected = ints(row["etv4_top5"])[:5]
    dense = ints(row["dense_top5"])[:5]
    removed = [doc for doc in dense if doc not in set(selected)]
    by_doc = {int(info["doc_index"]): info for info in row.get("inserted_docs", [])}
    out: list[int] = []
    seen: set[int] = set()
    for doc in selected:
        info = by_doc.get(int(doc))
        if info is not None and pred(info):
            while removed and removed[0] in seen:
                removed.pop(0)
            if removed:
                repl = int(removed.pop(0))
                if repl not in seen:
                    out.append(repl)
                    seen.add(repl)
                    continue
        if int(doc) not in seen:
            out.append(int(doc))
            seen.add(int(doc))
    for doc in dense + selected:
        if len(out) >= 5:
            break
        if int(doc) not in seen:
            out.append(int(doc))
            seen.add(int(doc))
    return tuple(out[:5])


def weak(info: Mapping[str, Any]) -> bool:
    return str(info.get("evidence_class")) in {"weak_graph_connected", "isolated_fill"}


def summarize(name: str, rows: Sequence[Mapping[str, Any]], pred: Predicate) -> dict[str, Any]:
    changed = []
    for row in rows:
        cf = cf_top5(row, pred)
        if cf == tuple(ints(row["etv4_top5"])[:5]):
            continue
        before_hits = hits(row["etv4_top5"], row["gold_doc_indices"])
        after_hits = hits(cf, row["gold_doc_indices"])
        before_all = all_gold(row["etv4_top5"], row["gold_doc_indices"])
        after_all = all_gold(cf, row["gold_doc_indices"])
        changed.append(
            {
                "hop": int(row["hop"]),
                "bucket": str(row["bucket"]),
                "dh": after_hits - before_hits,
                "da": after_all - before_all,
            }
        )
    by_bucket = Counter((row["hop"], row["bucket"]) for row in changed)
    return {
        "rule": name,
        "changed": len(changed),
        "hit_delta": sum(row["dh"] for row in changed),
        "all_delta": sum(row["da"] for row in changed),
        "hit_delta_counts": dict(sorted(Counter(row["dh"] for row in changed).items())),
        "by_bucket": {f"hop{hop}_{bucket}": count for (hop, bucket), count in sorted(by_bucket.items())},
    }


def main() -> int:
    rows = annotate_rows()
    rules: list[tuple[str, Predicate]] = []
    for threshold in (10, 20, 30, 40, 50, 75):
        rules.append((f"candidate_pos>={threshold}", lambda info, t=threshold: int(info["candidate_pos"]) >= t))
        rules.append((f"weak_and_candidate_pos>={threshold}", lambda info, t=threshold: weak(info) and int(info["candidate_pos"]) >= t))
        rules.append(
            (
                f"weak_graph_expanded_candidate_pos>={threshold}",
                lambda info, t=threshold: weak(info)
                and str(info.get("origin")) == "graph_expanded"
                and int(info["candidate_pos"]) >= t,
            )
        )
        rules.append(
            (
                f"weak_no_new_coverage_candidate_pos>={threshold}",
                lambda info, t=threshold: weak(info)
                and not info.get("new_query_coverage")
                and int(info["candidate_pos"]) >= t,
            )
        )

    summaries = [summarize(name, rows, pred) for name, pred in rules]
    summaries.sort(key=lambda item: (item["hit_delta"], item["all_delta"], -item["changed"]), reverse=True)
    print("Rule                                      Changed  HitDelta  AllDelta  HitDeltaCounts  ByBucket")
    print("----------------------------------------  -------  --------  --------  --------------  --------")
    for item in summaries:
        print(
            f"{item['rule'][:40].ljust(40)}  "
            f"{str(item['changed']).rjust(7)}  "
            f"{str(item['hit_delta']).rjust(8)}  "
            f"{str(item['all_delta']).rjust(8)}  "
            f"{item['hit_delta_counts']}  "
            f"{item['by_bucket']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
