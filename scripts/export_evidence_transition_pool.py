#!/usr/bin/env python3
"""Export Evidence Transition retrieval candidates as DAEC external-pool JSON.

This utility converts the fresh Evidence Transition retrieval report into the
``--external_pool_json`` format consumed by ``scripts/eval_causal_qwen3.py``.
The exported pool keeps the Evidence Transition top-5 prefix first, then fills
the remaining budget from the method-owned candidate universe.  That makes
``--setwise_selector none`` correspond to the original Evidence Transition
reader context while allowing DAEC/DBEC-style selectors to recombine the same
candidate pool.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_DATA_ROOT = Path("reproduce/dataset")


def unique_ints(values: Iterable[Any]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values or []:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def extract_title(doc: str) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def parse_answer_alias_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                return [cleaned]
            return parse_answer_alias_values(parsed)
        return [cleaned]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        aliases: list[str] = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]


def get_gold_answers(samples: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    gold_answers: list[list[str]] = []
    for sample in samples:
        answers: list[str] = []
        if "answer" in sample or "gold_ans" in sample:
            answers.extend(parse_answer_alias_values(sample.get("answer", sample.get("gold_ans"))))
        elif "reference" in sample:
            answers.extend(parse_answer_alias_values(sample.get("reference")))
        elif "obj" in sample:
            answers.extend(parse_answer_alias_values(sample.get("obj")))
            answers.extend(parse_answer_alias_values(sample.get("possible_answers")))
            answers.extend(parse_answer_alias_values(sample.get("o_wiki_title")))
            answers.extend(parse_answer_alias_values(sample.get("o_aliases")))
        if "answer_aliases" in sample:
            answers.extend(parse_answer_alias_values(sample["answer_aliases"]))
        gold_answers.append(sorted({str(item).strip() for item in answers if str(item).strip()}))
    return gold_answers


def get_gold_docs(samples: Sequence[Mapping[str, Any]], dataset_name: str) -> list[list[str]]:
    gold_docs: list[list[str]] = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            gold_title_and_content = [item for item in sample["context"] if item[0] in gold_titles]
            if str(dataset_name).startswith("hotpotqa"):
                docs = [item[0] + "\n" + "".join(item[1]) for item in gold_title_and_content]
            else:
                docs = [item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content]
        elif "contexts" in sample:
            docs = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item.get("is_supporting")
            ]
        else:
            docs = [
                item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"])
                for item in sample.get("paragraphs", [])
                if item.get("is_supporting") is not False
            ]
        gold_docs.append(sorted(set(docs)))
    return gold_docs


def load_openie_docs(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload.get("docs", []) if isinstance(payload, Mapping) else payload
    if not isinstance(docs, list):
        raise ValueError(f"OpenIE input must be a list or mapping with docs: {path}")
    return [doc for doc in docs if isinstance(doc, Mapping)]


def passage_for_doc(openie_docs: Sequence[Mapping[str, Any]], doc_idx: int) -> str:
    if 0 <= int(doc_idx) < len(openie_docs):
        return str(openie_docs[int(doc_idx)].get("passage") or "")
    return ""


def score_pool_docs(pool_doc_indices: Sequence[int], *, selected_doc_indices: Sequence[int]) -> list[float]:
    selected = set(unique_ints(selected_doc_indices))
    pool_size = max(len(pool_doc_indices), 1)
    scores: list[float] = []
    for rank, doc_idx in enumerate(pool_doc_indices, start=1):
        score = float(pool_size - rank + 1)
        if int(doc_idx) in selected:
            score += float(pool_size)
        scores.append(score)
    return scores


def compute_title_recall(
    *,
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
    k_values: Iterable[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        scores: list[float] = []
        for gold, retrieved in zip(gold_docs, retrieved_docs):
            gold_titles = {extract_title(doc) for doc in gold}
            retrieved_titles = {extract_title(doc) for doc in list(retrieved)[: int(k)]}
            scores.append(0.0 if not gold_titles else len(gold_titles & retrieved_titles) / len(gold_titles))
        metrics[f"Recall@{int(k)}"] = round(float(np.mean(scores)) if scores else 0.0, 4)
    return metrics


def build_records(
    *,
    dataset: str,
    samples: Sequence[Mapping[str, Any]],
    retrieval_rows: Sequence[Mapping[str, Any]],
    openie_docs: Sequence[Mapping[str, Any]],
    pool_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    gold_docs = get_gold_docs(samples, dataset)
    gold_answers = get_gold_answers(samples)
    records: list[dict[str, Any]] = []
    retrieved_doc_lists: list[list[str]] = []

    if len(retrieval_rows) < len(samples):
        raise ValueError(f"Retrieval report has {len(retrieval_rows)} rows, but {len(samples)} samples were requested")

    for query_idx, sample in enumerate(samples):
        row = retrieval_rows[query_idx]
        record_query_idx = int(row.get("query_index", row.get("query_idx", query_idx)))
        if record_query_idx != query_idx:
            raise ValueError(f"Retrieval row index mismatch at {query_idx}: found {record_query_idx}")

        question = str(sample.get("question") or "")
        report_question = str(row.get("question") or "")
        if report_question and report_question != question:
            raise ValueError(f"Question mismatch at {query_idx}: sample={question!r}, report={report_question!r}")

        route_trace = dict(row.get("route_trace") or {})
        candidate_universe = dict(route_trace.get("candidate_universe") or {})
        selected_top5 = unique_ints(row.get("retrieved_doc_indices_top5") or [])
        candidate_indices = unique_ints(candidate_universe.get("candidate_doc_indices") or [])
        if not candidate_indices:
            candidate_indices = unique_ints(candidate_universe.get("admissible_doc_indices") or [])

        pool_doc_indices = [
            idx
            for idx in unique_ints(list(selected_top5) + list(candidate_indices))
            if passage_for_doc(openie_docs, idx)
        ][: int(pool_k)]
        pool_docs = [passage_for_doc(openie_docs, idx) for idx in pool_doc_indices]
        retrieved_doc_lists.append(pool_docs)

        records.append(
            {
                "query_idx": int(query_idx),
                "question": question,
                "gold_answers": list(gold_answers[query_idx]),
                "gold_docs": list(gold_docs[query_idx]),
                "gold_titles": [extract_title(doc) for doc in gold_docs[query_idx]],
                "pool_k": int(pool_k),
                "pool_docs": pool_docs,
                "pool_titles": [extract_title(doc) for doc in pool_docs],
                "pool_doc_scores": score_pool_docs(pool_doc_indices, selected_doc_indices=selected_top5),
                "pool_doc_ids": pool_doc_indices,
                "agsto": {
                    "source": "evidence_transition",
                    "selected_doc_indices": selected_top5,
                    "candidate_doc_indices": candidate_indices[: int(pool_k)],
                    "candidate_count": int(candidate_universe.get("candidate_count", len(candidate_indices)) or 0),
                    "candidate_order_policy": candidate_universe.get("candidate_order_policy"),
                    "candidate_source": candidate_universe.get("candidate_source"),
                    "source_prior_prefix_doc_indices": unique_ints(
                        candidate_universe.get("source_prior_prefix_doc_indices") or []
                    ),
                    "agsto_seed_doc_indices": unique_ints(candidate_universe.get("agsto_seed_doc_indices") or []),
                    "agsto_symbolic_seed_doc_indices": unique_ints(
                        candidate_universe.get("agsto_symbolic_seed_doc_indices") or []
                    ),
                    "agsto_graph_tail_doc_indices": unique_ints(
                        candidate_universe.get("agsto_graph_tail_doc_indices") or []
                    )[: int(pool_k)],
                    "agsto_local_edge_count": int(candidate_universe.get("agsto_local_edge_count", 0) or 0),
                    "agsto_role_graph_edge_count": int(candidate_universe.get("agsto_role_graph_edge_count", 0) or 0),
                },
            }
        )

    return records, compute_title_recall(gold_docs=gold_docs, retrieved_docs=retrieved_doc_lists, k_values=[5, 20, 100])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--retrieval_report", type=Path, required=True)
    parser.add_argument("--openie_results", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    samples = json.loads((Path(args.data_root) / f"{args.dataset}.json").read_text(encoding="utf-8"))
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]

    retrieval_payload = json.loads(Path(args.retrieval_report).read_text(encoding="utf-8"))
    retrieval_rows = list(retrieval_payload.get("rows") or retrieval_payload.get("records") or [])
    openie_docs = load_openie_docs(Path(args.openie_results))
    records, recall = build_records(
        dataset=str(args.dataset),
        samples=samples,
        retrieval_rows=retrieval_rows,
        openie_docs=openie_docs,
        pool_k=int(args.pool_k),
    )

    output = {
        "dataset": str(args.dataset),
        "limit": int(len(samples)),
        "pool_k": int(args.pool_k),
        "source": "evidence_transition_pool_export",
        "retrieval_report": str(args.retrieval_report),
        "openie_path": str(args.openie_results),
        "retrieval": {
            "recomputed_title_recall": recall,
            "input_metrics": retrieval_payload.get("metrics") or {},
            "input_method": retrieval_payload.get("method"),
            "input_source_variant": retrieval_payload.get("source_variant"),
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
        errors="replace",
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "dataset": str(args.dataset),
                "limit": len(samples),
                "pool_k": int(args.pool_k),
                "recomputed_title_recall": recall,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
