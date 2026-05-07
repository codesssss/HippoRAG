#!/usr/bin/env python3
"""Pilot Dependency-Bound Iterative Retrieval on MuSiQue audit slices.

This is a fast, train-free pilot.  It does not call the reader or any LLM.  The
retrieval substrate is a local lexical corpus retriever so that the pilot can
separate the structural question from GPU/API availability:

* Does dependency-bound slot control retrieve target gold evidence better than
  same-budget original-query expansion, independent demand retrieval, and a
  generic context-iteration proxy?
* Can retrieval expansion recover any gold titles absent from the fixed
  PropRAG top-100 source pool?

The pilot intentionally evaluates expansion-produced pools only.  It does not
union the initial fixed pool into D-BIR outputs; otherwise Slice A would be
trivially solved because its target gold titles are already source-pool visible.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import (  # noqa: E402
    DBEC_SELECTIVE_DIR,
    SOURCE_POOL_DIR,
    normalize_title,
    read_json,
    safe_float,
    safe_int,
)
from probe_chain_walking_binding import (  # noqa: E402
    CANDIDATE_AUDIT_DIR,
    MUSIQUE_DATASET,
    MUSIQUE_LABEL,
    TARGET_BUCKET,
    dependent_requirements,
    grouped_missing_gold,
    requirement_by_id,
    selector_trace_from_query_trace,
    target_missing_gold_rows,
)


REPORT_DIR = Path("reports/dbir_pilot_musique_20260507")
CORPUS_PATH = Path("reproduce/dataset/musique_corpus.json")
SOURCE_POOL_PATH = SOURCE_POOL_DIR / "musique_pool100.json"
DBEC_TRACE_PATH = DBEC_SELECTIVE_DIR / "musique_proprag_wiki_title_daec_selective_titleuniq_full1000.json"

SLICE_A = "source_visible_not_rank_or_dbec"
SLICE_B = "source_pool_absent"

POLICIES = (
    "fixed_pool_top100",
    "rank_expansion",
    "independent_demand",
    "context_iterative_lite",
    "dbir_det",
)

STOP_TOKENS = {
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

GENERIC_TOKENS = {
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


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value: Any) -> list[str]:
    return [
        token
        for token in normalize_text(value).split()
        if len(token) >= 3 and token not in STOP_TOKENS and token not in GENERIC_TOKENS
    ]


def unique_ordered(values: Iterable[int]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        pos = int(value)
        if pos in seen:
            continue
        seen.add(pos)
        output.append(pos)
    return output


def doc_title(doc: Mapping[str, Any]) -> str:
    return str(doc.get("title") or "")


def doc_text(doc: Mapping[str, Any]) -> str:
    return str(doc.get("text") or "")


class LexicalCorpusRetriever:
    """Small BM25-like lexical retriever over the MuSiQue corpus."""

    def __init__(self, docs: Sequence[Mapping[str, Any]]) -> None:
        self.docs = [dict(doc) for doc in docs]
        self.doc_tokens: list[Counter[str]] = []
        self.title_tokens: list[set[str]] = []
        postings: dict[str, list[int]] = defaultdict(list)
        for idx, doc in enumerate(self.docs):
            tokens = Counter(content_tokens(f"{doc_title(doc)} {doc_text(doc)}"))
            self.doc_tokens.append(tokens)
            self.title_tokens.append(set(content_tokens(doc_title(doc))))
            for token in tokens:
                postings[token].append(idx)
        self.postings = dict(postings)
        total = max(1, len(self.docs))
        self.idf = {
            token: math.log((total + 1) / (len(doc_ids) + 1)) + 1.0
            for token, doc_ids in self.postings.items()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "LexicalCorpusRetriever":
        return cls(list(read_json(path)))

    def retrieve(
        self,
        query: Any,
        *,
        top_k: int,
        exclude: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = Counter(content_tokens(query))
        if not query_tokens:
            return []
        exclude = exclude or set()
        candidates: set[int] = set()
        for token in query_tokens:
            candidates.update(self.postings.get(token, []))
        scored: list[tuple[float, int]] = []
        for doc_id in candidates:
            if doc_id in exclude:
                continue
            score = 0.0
            doc_counter = self.doc_tokens[doc_id]
            title_tokens = self.title_tokens[doc_id]
            for token, qtf in query_tokens.items():
                tf = doc_counter.get(token, 0)
                if not tf:
                    continue
                title_boost = 2.0 if token in title_tokens else 1.0
                score += float(qtf) * (1.0 + math.log(1.0 + tf)) * self.idf.get(token, 1.0) * title_boost
            if score > 0.0:
                scored.append((score, doc_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        output = []
        for rank, (score, doc_id) in enumerate(scored[: int(top_k)]):
            doc = self.docs[doc_id]
            output.append(
                {
                    "doc_id": doc_id,
                    "title": doc_title(doc),
                    "text": doc_text(doc),
                    "score": float(score),
                    "rank": rank,
                }
            )
        return output


def corpus_title_positions(retriever: LexicalCorpusRetriever) -> dict[str, list[int]]:
    output: dict[str, list[int]] = defaultdict(list)
    for idx, doc in enumerate(retriever.docs):
        norm = normalize_title(doc_title(doc))
        if norm:
            output[norm].append(idx)
    return dict(output)


def source_pool_records() -> list[dict[str, Any]]:
    return list(read_json(SOURCE_POOL_PATH).get("records") or [])


def dbec_traces() -> list[dict[str, Any]]:
    return list(read_json(DBEC_TRACE_PATH).get("setwise_selector_query_traces") or [])


def source_positions_for_title(source_record: Mapping[str, Any], title: Any) -> list[int]:
    norm = normalize_title(title)
    return [
        idx
        for idx, candidate in enumerate(source_record.get("pool_titles") or [])
        if normalize_title(candidate) == norm
    ]


def fixed_pool_order_for_record(source_record: Mapping[str, Any], retriever: LexicalCorpusRetriever) -> list[int]:
    title_to_corpus = corpus_title_positions(retriever)
    output: list[int] = []
    for title in source_record.get("pool_titles") or []:
        positions = title_to_corpus.get(normalize_title(title), [])
        if positions:
            output.append(positions[0])
    return unique_ordered(output)


def load_target_rows(*, query_primary_only: bool) -> list[dict[str, Any]]:
    slice_a = target_missing_gold_rows(
        audit_dir=CANDIDATE_AUDIT_DIR,
        dataset_label=MUSIQUE_LABEL,
        bucket=TARGET_BUCKET,
        query_primary_only=query_primary_only,
    )
    for row in slice_a:
        row["pilot_slice"] = SLICE_A
    all_missing = []
    with (CANDIDATE_AUDIT_DIR / "missing_gold_audit.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("dataset")) == MUSIQUE_LABEL and str(row.get("bucket")) == "gold_absent_from_source_pool":
                item = dict(row)
                item["pilot_slice"] = SLICE_B
                all_missing.append(item)
    return sorted(slice_a + all_missing, key=lambda row: (str(row["pilot_slice"]), safe_int(row["query_index"]), str(row["gold_title"])))


def all_requirements(selector_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(req) for req in selector_trace.get("requirements") or [] if isinstance(req, Mapping)]


def topological_requirements(selector_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    requirements = all_requirements(selector_trace)
    by_id = {str(req.get("unit_id")): req for req in requirements}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(req: Mapping[str, Any]) -> None:
        req_id = str(req.get("unit_id") or "")
        if req_id in seen:
            return
        for dep in req.get("depends_on") or []:
            if str(dep) in by_id:
                visit(by_id[str(dep)])
        seen.add(req_id)
        ordered.append(dict(req))

    for req in requirements:
        visit(req)
    return ordered


def snippet_terms(text: Any, *, max_terms: int = 24) -> str:
    tokens = content_tokens(text)
    seen: set[str] = set()
    output: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
        if len(output) >= int(max_terms):
            break
    return " ".join(output)


def doc_snippet(doc: Mapping[str, Any], *, max_chars: int = 300) -> str:
    return f"{doc.get('title', '')} {doc.get('text', '')}"[: int(max_chars)]


def query_text_for_requirement(req: Mapping[str, Any]) -> str:
    anchors = " ".join(str(value) for value in req.get("anchor_mentions") or [])
    return f"{req.get('subquery', '')} {anchors}".strip()


def add_results_to_order(order: list[int], results: Sequence[Mapping[str, Any]], *, limit: int) -> None:
    seen = set(order)
    for result in results:
        doc_id = safe_int(result.get("doc_id"), default=-1)
        if doc_id < 0 or doc_id in seen:
            continue
        order.append(doc_id)
        seen.add(doc_id)
        if len(order) >= int(limit):
            break


def run_rank_expansion(
    *,
    question: str,
    retriever: LexicalCorpusRetriever,
    top_k: int,
) -> tuple[list[int], dict[str, Any]]:
    results = retriever.retrieve(question, top_k=top_k)
    order = [safe_int(result.get("doc_id"), default=-1) for result in results]
    return unique_ordered(pos for pos in order if pos >= 0), {
        "retrieval_calls": 1,
        "slot_count": 0,
        "resolved_slot_count": 0,
        "slot_traces": [],
    }


def run_independent_demand(
    *,
    selector_trace: Mapping[str, Any],
    retriever: LexicalCorpusRetriever,
    top_m: int,
    expanded_k: int,
) -> tuple[list[int], dict[str, Any]]:
    order: list[int] = []
    slot_traces: list[dict[str, Any]] = []
    calls = 0
    for req in topological_requirements(selector_trace):
        query = query_text_for_requirement(req)
        results = retriever.retrieve(query, top_k=top_m)
        calls += 1
        add_results_to_order(order, results, limit=expanded_k)
        slot_traces.append(
            {
                "slot_id": str(req.get("unit_id") or ""),
                "query": query,
                "status": "retrieved" if results else "empty",
                "top_titles": [str(result.get("title") or "") for result in results[:5]],
            }
        )
        if len(order) >= expanded_k:
            break
    return order[:expanded_k], {
        "retrieval_calls": calls,
        "slot_count": len(all_requirements(selector_trace)),
        "resolved_slot_count": sum(1 for trace in slot_traces if trace["status"] == "retrieved"),
        "slot_traces": slot_traces,
    }


def run_context_iterative_lite(
    *,
    question: str,
    retriever: LexicalCorpusRetriever,
    top_m: int,
    iterations: int,
    expanded_k: int,
) -> tuple[list[int], dict[str, Any]]:
    order: list[int] = []
    context = ""
    traces: list[dict[str, Any]] = []
    for step in range(int(iterations)):
        query = f"{question} {context}".strip()
        results = retriever.retrieve(query, top_k=top_m, exclude=set(order))
        add_results_to_order(order, results, limit=expanded_k)
        if results:
            context = snippet_terms(doc_snippet(results[0]), max_terms=24)
        traces.append(
            {
                "step": step + 1,
                "query": query,
                "top_titles": [str(result.get("title") or "") for result in results[:5]],
            }
        )
        if len(order) >= expanded_k or not results:
            break
    return order[:expanded_k], {
        "retrieval_calls": len(traces),
        "slot_count": 0,
        "resolved_slot_count": 0,
        "slot_traces": traces,
    }


def run_dbir_det(
    *,
    selector_trace: Mapping[str, Any],
    retriever: LexicalCorpusRetriever,
    top_m: int,
    max_iterations: int,
    expanded_k: int,
) -> tuple[list[int], dict[str, Any]]:
    requirements = topological_requirements(selector_trace)
    req_by_id = requirement_by_id(selector_trace)
    statuses = {str(req.get("unit_id")): "UNRESOLVED" for req in requirements}
    bindings: dict[str, str] = {}
    snippets: dict[str, str] = {}
    order: list[int] = []
    slot_traces: list[dict[str, Any]] = []
    calls = 0

    for iteration in range(int(max_iterations)):
        progress = False
        for req in requirements:
            req_id = str(req.get("unit_id") or "")
            if statuses.get(req_id) == "RESOLVED":
                continue
            upstream_ids = [str(dep) for dep in req.get("depends_on") or []]
            if any(statuses.get(dep) != "RESOLVED" for dep in upstream_ids):
                continue
            upstream_bindings = [bindings.get(dep, "") for dep in upstream_ids if bindings.get(dep)]
            upstream_context = " ".join(snippets.get(dep, "") for dep in upstream_ids if snippets.get(dep))
            if upstream_bindings or upstream_context:
                query = " ".join(
                    part
                    for part in (
                        query_text_for_requirement(req),
                        " ".join(upstream_bindings),
                        snippet_terms(upstream_context, max_terms=32),
                    )
                    if part
                )
            else:
                query = query_text_for_requirement(req)
            results = retriever.retrieve(query, top_k=top_m, exclude=set(order))
            calls += 1
            top_doc = results[0] if results else None
            if top_doc:
                doc_id = safe_int(top_doc.get("doc_id"), default=-1)
                if doc_id >= 0:
                    order.append(doc_id)
                add_results_to_order(order, results[1:], limit=expanded_k)
                bindings[req_id] = str(top_doc.get("title") or "")
                snippets[req_id] = doc_snippet(top_doc)
                statuses[req_id] = "RESOLVED"
                progress = True
            else:
                statuses[req_id] = "PARTIAL"
            slot_traces.append(
                {
                    "iteration": iteration + 1,
                    "slot_id": req_id,
                    "depends_on": upstream_ids,
                    "query": query,
                    "status": statuses.get(req_id),
                    "binding": bindings.get(req_id, ""),
                    "top_titles": [str(result.get("title") or "") for result in results[:5]],
                }
            )
            if len(order) >= expanded_k:
                break
        if len(order) >= expanded_k or not progress:
            break
    return unique_ordered(order)[:expanded_k], {
        "retrieval_calls": calls,
        "slot_count": len(requirements),
        "resolved_slot_count": sum(1 for status in statuses.values() if status == "RESOLVED"),
        "slot_traces": slot_traces,
        "bindings": bindings,
    }


def rank_of_gold(gold_title: Any, order: Sequence[int], retriever: LexicalCorpusRetriever) -> int | None:
    norm = normalize_title(gold_title)
    if not norm:
        return None
    for rank, doc_id in enumerate(order):
        if 0 <= int(doc_id) < len(retriever.docs) and normalize_title(doc_title(retriever.docs[int(doc_id)])) == norm:
            return rank
    return None


def run_policy(
    *,
    policy: str,
    question: str,
    selector_trace: Mapping[str, Any],
    source_record: Mapping[str, Any],
    retriever: LexicalCorpusRetriever,
    top_m: int,
    expanded_k: int,
    max_iterations: int,
) -> tuple[list[int], dict[str, Any]]:
    if policy == "fixed_pool_top100":
        order = fixed_pool_order_for_record(source_record, retriever)
        return order[:expanded_k], {
            "retrieval_calls": 0,
            "slot_count": len(all_requirements(selector_trace)),
            "resolved_slot_count": 0,
            "slot_traces": [],
        }
    if policy == "rank_expansion":
        return run_rank_expansion(question=question, retriever=retriever, top_k=expanded_k)
    if policy == "independent_demand":
        return run_independent_demand(
            selector_trace=selector_trace,
            retriever=retriever,
            top_m=top_m,
            expanded_k=expanded_k,
        )
    if policy == "context_iterative_lite":
        return run_context_iterative_lite(
            question=question,
            retriever=retriever,
            top_m=top_m,
            iterations=max_iterations,
            expanded_k=expanded_k,
        )
    if policy == "dbir_det":
        return run_dbir_det(
            selector_trace=selector_trace,
            retriever=retriever,
            top_m=top_m,
            max_iterations=max_iterations,
            expanded_k=expanded_k,
        )
    raise ValueError(f"Unknown policy: {policy}")


def build_rows(
    *,
    target_rows: Sequence[Mapping[str, Any]],
    retriever: LexicalCorpusRetriever,
    policies: Sequence[str],
    top_m: int,
    expanded_k: int,
    max_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_records = source_pool_records()
    traces = dbec_traces()
    grouped = grouped_missing_gold(target_rows)
    expanded_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for query_index in sorted(grouped):
        source_record = source_records[query_index]
        selector_trace = selector_trace_from_query_trace(traces[query_index])
        question = str(source_record.get("question") or selector_trace.get("query") or "")
        for policy in policies:
            order, trace = run_policy(
                policy=policy,
                question=question,
                selector_trace=selector_trace,
                source_record=source_record,
                retriever=retriever,
                top_m=top_m,
                expanded_k=expanded_k,
                max_iterations=max_iterations,
            )
            trace_rows.append(
                {
                    "query_index": query_index,
                    "question": question,
                    "policy": policy,
                    "expanded_doc_count": len(order),
                    "expanded_titles": [doc_title(retriever.docs[doc_id]) for doc_id in order[:20]],
                    **trace,
                }
            )
            for target in grouped[query_index]:
                gold_title = str(target.get("gold_title") or "")
                source_positions = source_positions_for_title(source_record, gold_title)
                source_rank = min(source_positions) if source_positions else None
                expanded_rank = rank_of_gold(gold_title, order, retriever)
                if expanded_rank is None:
                    expanded_rank_value = ""
                    rank_improvement = ""
                else:
                    expanded_rank_value = expanded_rank
                    rank_improvement = (
                        safe_int(source_rank, default=expanded_k + 1) - expanded_rank
                        if source_rank is not None
                        else ""
                    )
                expanded_rows.append(
                    {
                        "pilot_slice": str(target.get("pilot_slice") or ""),
                        "policy": policy,
                        "dataset": MUSIQUE_LABEL,
                        "base_dataset": MUSIQUE_DATASET,
                        "query_index": query_index,
                        "question": question,
                        "gold_title": gold_title,
                        "source_has_gold": int(bool(source_positions)),
                        "source_best_gold_rank": source_rank if source_rank is not None else "",
                        "expanded_best_gold_rank": expanded_rank_value,
                        "rank_improvement_vs_source": rank_improvement,
                        "expanded_hit_at_5": int(expanded_rank is not None and expanded_rank < 5),
                        "expanded_hit_at_10": int(expanded_rank is not None and expanded_rank < 10),
                        "expanded_hit_at_20": int(expanded_rank is not None and expanded_rank < 20),
                        "expanded_hit_at_50": int(expanded_rank is not None and expanded_rank < 50),
                        "expanded_hit_at_100": int(expanded_rank is not None and expanded_rank < 100),
                        "final_new_at_5": int((source_rank is None or source_rank >= 5) and expanded_rank is not None and expanded_rank < 5),
                        "pool_absent_recovered_at_100": int(not source_positions and expanded_rank is not None and expanded_rank < 100),
                        "retrieval_calls": safe_int(trace.get("retrieval_calls")),
                        "slot_count": safe_int(trace.get("slot_count")),
                        "resolved_slot_count": safe_int(trace.get("resolved_slot_count")),
                        "slot_coverage": (
                            safe_int(trace.get("resolved_slot_count")) / safe_int(trace.get("slot_count"))
                            if safe_int(trace.get("slot_count")) else ""
                        ),
                        "expanded_top5_titles_json": json.dumps(
                            [doc_title(retriever.docs[doc_id]) for doc_id in order[:5]],
                            ensure_ascii=False,
                        ),
                    }
                )
    return expanded_rows, trace_rows


def mean_value(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    vals = [safe_float(row.get(field)) for row in rows if str(row.get(field, "")) != ""]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    slices = sorted({str(row.get("pilot_slice")) for row in rows})
    policies = [policy for policy in POLICIES if any(str(row.get("policy")) == policy for row in rows)]
    for pilot_slice in slices:
        slice_rows = [row for row in rows if str(row.get("pilot_slice")) == pilot_slice]
        total = len(slice_rows)
        for policy in policies:
            policy_rows = [row for row in slice_rows if str(row.get("policy")) == policy]
            denom = len(policy_rows)
            if not denom:
                continue
            output.append(
                {
                    "pilot_slice": pilot_slice,
                    "policy": policy,
                    "missing_gold_titles": denom,
                    "queries": len({safe_int(row.get("query_index")) for row in policy_rows}),
                    "expanded_recall_at_5": mean_value(policy_rows, "expanded_hit_at_5"),
                    "expanded_recall_at_10": mean_value(policy_rows, "expanded_hit_at_10"),
                    "expanded_recall_at_20": mean_value(policy_rows, "expanded_hit_at_20"),
                    "expanded_recall_at_50": mean_value(policy_rows, "expanded_hit_at_50"),
                    "expanded_recall_at_100": mean_value(policy_rows, "expanded_hit_at_100"),
                    "final_new_at_5": mean_value(policy_rows, "final_new_at_5"),
                    "pool_absent_recovery_at_100": mean_value(policy_rows, "pool_absent_recovered_at_100"),
                    "mean_retrieval_calls": mean_value(policy_rows, "retrieval_calls"),
                    "mean_slot_coverage": mean_value(policy_rows, "slot_coverage"),
                    "mean_rank_improvement_vs_source": mean_value(policy_rows, "rank_improvement_vs_source"),
                }
            )
    return output


def classify_decision(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    def find(slice_name: str, policy: str) -> Mapping[str, Any]:
        return next(
            (
                row
                for row in summary_rows
                if str(row.get("pilot_slice")) == slice_name and str(row.get("policy")) == policy
            ),
            {},
        )

    slice_a_dbir = find(SLICE_A, "dbir_det")
    slice_a_ircot = find(SLICE_A, "context_iterative_lite")
    slice_a_ind = find(SLICE_A, "independent_demand")
    slice_b_dbir = find(SLICE_B, "dbir_det")
    if not slice_a_dbir:
        return "not_run"

    dbir_exp50 = safe_float(slice_a_dbir.get("expanded_recall_at_50"))
    dbir_final5 = safe_float(slice_a_dbir.get("final_new_at_5"))
    dbir_pool_absent = safe_float(slice_b_dbir.get("pool_absent_recovery_at_100"))
    beats_ircot = dbir_exp50 > safe_float(slice_a_ircot.get("expanded_recall_at_50"))
    beats_ind = dbir_exp50 > safe_float(slice_a_ind.get("expanded_recall_at_50"))

    if dbir_exp50 >= 0.40 and dbir_final5 >= 0.20 and dbir_pool_absent >= 0.10 and beats_ircot:
        return "strong_go"
    if dbir_exp50 >= 0.20 and (beats_ircot or beats_ind):
        return "marginal_inspect"
    return "stop_or_pivot"


def pct(value: Any) -> str:
    return f"{100.0 * safe_float(value):.1f}%"


def build_markdown(summary_rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], report_dir: Path) -> str:
    lines = [
        "# D-BIR MuSiQue Pilot",
        "",
        "This pilot uses a local lexical corpus retriever. It is a structural feasibility test, not the final PropRAG-substrate experiment. Expansion metrics are computed on expansion-produced pools only; the original fixed pool is reported as a baseline and is not unioned into D-BIR outputs.",
        "",
        "## Decision",
        "",
        f"- Decision: `{metadata.get('decision')}`",
        f"- Query-primary Slice A: `{metadata.get('query_primary_only')}`",
        f"- Top-m per retrieval call: `{metadata.get('top_m')}`",
        f"- Expanded K: `{metadata.get('expanded_k')}`",
        f"- Max iterations: `{metadata.get('max_iterations')}`",
        "",
        "## Policy Summary",
        "",
        "| Slice | Policy | Titles | Q | Exp@50 | Exp@100 | Final New@5 | Pool-absent@100 | calls/q | slot coverage | rank improvement |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {slice} | {policy} | {titles} | {queries} | {e50} | {e100} | {f5} | {pa} | {calls:.2f} | {slot} | {imp:.1f} |".format(
                slice=row.get("pilot_slice"),
                policy=row.get("policy"),
                titles=row.get("missing_gold_titles"),
                queries=row.get("queries"),
                e50=pct(row.get("expanded_recall_at_50")),
                e100=pct(row.get("expanded_recall_at_100")),
                f5=pct(row.get("final_new_at_5")),
                pa=pct(row.get("pool_absent_recovery_at_100")),
                calls=safe_float(row.get("mean_retrieval_calls")),
                slot=pct(row.get("mean_slot_coverage")),
                imp=safe_float(row.get("mean_rank_improvement_vs_source")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Strong go requires D-BIR to beat the equal-budget context-iteration proxy, reach strong Slice-A expanded recall/final New@5, and recover nontrivial Slice-B pool-absent gold.",
            "- If D-BIR only matches independent demand retrieval, dependency-bound control is not yet contributing enough.",
            "- If expanded recall is high but Final New@5 is weak, the next bottleneck is composition rather than expansion.",
            "",
            "## Files",
            "",
            f"- Expanded rows: `{report_dir / 'expanded_pool_rows.csv'}`",
            f"- Query traces: `{report_dir / 'slot_trace.jsonl'}`",
            f"- Policy summary: `{report_dir / 'policy_summary.csv'}`",
            f"- Full summary: `{report_dir / 'summary.json'}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--corpus_path", type=Path, default=CORPUS_PATH)
    parser.add_argument("--query_primary_only", action="store_true", default=True)
    parser.add_argument("--all_slice_a_titles", action="store_true")
    parser.add_argument("--top_m", type=int, default=10)
    parser.add_argument("--expanded_k", type=int, default=100)
    parser.add_argument("--max_iterations", type=int, default=4)
    parser.add_argument("--limit_queries", type=int, default=0)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    query_primary_only = bool(args.query_primary_only) and not bool(args.all_slice_a_titles)
    target_rows = load_target_rows(query_primary_only=query_primary_only)
    if int(args.limit_queries) > 0:
        allowed = set(sorted({safe_int(row.get("query_index")) for row in target_rows})[: int(args.limit_queries)])
        target_rows = [row for row in target_rows if safe_int(row.get("query_index")) in allowed]
    retriever = LexicalCorpusRetriever.from_json(args.corpus_path)
    rows, trace_rows = build_rows(
        target_rows=target_rows,
        retriever=retriever,
        policies=list(args.policies),
        top_m=int(args.top_m),
        expanded_k=int(args.expanded_k),
        max_iterations=int(args.max_iterations),
    )
    summary_rows = summarize_rows(rows)
    metadata = {
        "dataset": MUSIQUE_LABEL,
        "base_dataset": MUSIQUE_DATASET,
        "retriever": "local_lexical_corpus",
        "corpus_path": str(args.corpus_path),
        "corpus_docs": len(retriever.docs),
        "target_titles": len(target_rows),
        "target_queries": len({safe_int(row.get("query_index")) for row in target_rows}),
        "query_primary_only": query_primary_only,
        "top_m": int(args.top_m),
        "expanded_k": int(args.expanded_k),
        "max_iterations": int(args.max_iterations),
        "policies": list(args.policies),
        "decision": classify_decision(summary_rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {
        "metadata": metadata,
        "policy_summary": summary_rows,
        "expanded_pool_rows": rows,
    }
    write_csv(rows, report_dir / "expanded_pool_rows.csv")
    write_csv(summary_rows, report_dir / "policy_summary.csv")
    write_jsonl(trace_rows, report_dir / "slot_trace.jsonl")
    write_json(payload, report_dir / "summary.json")
    (report_dir / "summary.md").write_text(build_markdown(summary_rows, metadata, report_dir) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
