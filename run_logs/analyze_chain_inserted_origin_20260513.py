#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


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
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            pass
    return out


def pct(count: int, total: int) -> float:
    return round(float(count) / float(total) * 100.0, 2) if total else 0.0


def main() -> int:
    diag = json.loads(DIAG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    by_qid = {
        int(row.get("query_index", idx)): row
        for idx, row in enumerate(report.get("rows", []) or [])
    }

    origin_counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    evidence_counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    candidate_positions: dict[tuple[int, str], list[int]] = defaultdict(list)
    admission_distances: dict[tuple[int, str], list[int]] = defaultdict(list)

    for row in diag.get("query_rows", []) or []:
        key = (int(row["hop"]), str(row["bucket"]))
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

        for info in row.get("inserted_docs", []) or []:
            doc = int(info["doc_index"])
            if doc in symbolic and doc in textual:
                origin = "symbolic+textual_seed"
            elif doc in symbolic:
                origin = "symbolic_seed_only"
            elif doc in textual:
                origin = "textual_seed_tail"
            else:
                origin = "graph_expanded"
            origin_counts[key][origin] += 1
            evidence_counts[key][str(info.get("evidence_class"))] += 1
            candidate_positions[key].append(int(candidate_pos.get(doc, 999)))
            admission_distances[key].append(int(info.get("admission_distance", 0) or 0))

    print("Hop  Bucket  Inserted  SymOnly%  Sym+Text%  TextTail%  Expanded%  MeanCandPos  MeanAdmDist  EvidenceClasses")
    print("---  ------  --------  --------  ---------  ---------  ---------  -----------  -----------  ---------------")
    for key in sorted(origin_counts):
        counts = origin_counts[key]
        total = sum(counts.values())
        evi = ",".join(f"{name}:{count}" for name, count in sorted(evidence_counts[key].items()))
        print(
            f"{key[0]}  {key[1]}  {total}  "
            f"{pct(counts['symbolic_seed_only'], total)}  "
            f"{pct(counts['symbolic+textual_seed'], total)}  "
            f"{pct(counts['textual_seed_tail'], total)}  "
            f"{pct(counts['graph_expanded'], total)}  "
            f"{round(mean(candidate_positions[key]), 2) if candidate_positions[key] else 0.0}  "
            f"{round(mean(admission_distances[key]), 2) if admission_distances[key] else 0.0}  "
            f"{evi}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
