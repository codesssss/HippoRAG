#!/usr/bin/env python3
"""Audit whether AG-STO can be reduced to clean STO proposal RRF.

This is retrieval-only: it builds native AG-STO proposal channels, replays
current AG-STO selection, and evaluates clean RRF variants without reader QA or
query-time LLM calls.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agsto_v12 import AGSTOConfig, AGSTORetriever  # noqa: E402
from src.agsto_v12.proposals import build_native_sto_proposals  # noqa: E402
from src.agsto_v12.ranking import rank_sto_proposal_consensus_docs, unique_ranked  # noqa: E402


DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DEFAULT_OPENIE_TEMPLATE = "outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json"
DEFAULT_HIPPORAG_POOL_TEMPLATE = "run_logs/hipporag_pool_exports_full1000_20260503/{dataset}_hipporag_pool100.json"
CHANNEL_NAMES = ("specificity", "hybrid_residual", "neighborhood", "support_set")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_float_csv(value: str | None) -> List[float]:
    values = []
    for item in parse_csv(value):
        values.append(float(item))
    return values


def weight_key(weight: float) -> str:
    return str(float(weight))


def parse_answer_alias_values(value: Any) -> List[str]:
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
        aliases: List[str] = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]


def extract_title(doc: str) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def passage_for_doc(openie_docs: Sequence[Mapping[str, Any]], doc_idx: int) -> str:
    if 0 <= int(doc_idx) < len(openie_docs):
        return str(openie_docs[int(doc_idx)].get("passage") or "")
    return ""


def title_for_doc(openie_docs: Sequence[Mapping[str, Any]], doc_idx: int) -> str:
    return extract_title(passage_for_doc(openie_docs, int(doc_idx)))


def get_gold_docs(samples: Sequence[Mapping[str, Any]], dataset_name: str) -> List[List[str]]:
    gold_docs: List[List[str]] = []
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


def gold_titles_for_sample(gold_docs: Sequence[str]) -> List[str]:
    return sorted({extract_title(doc) for doc in gold_docs if extract_title(doc)})


def title_recall_at_k(gold_titles: Sequence[str], retrieved_titles: Sequence[str], k: int) -> float:
    gold = {str(title) for title in gold_titles if str(title)}
    if not gold:
        return 0.0
    retrieved = {str(title) for title in list(retrieved_titles or [])[: max(int(k), 0)] if str(title)}
    return float(len(gold & retrieved)) / float(len(gold))


def all_gold_at_k(gold_titles: Sequence[str], retrieved_titles: Sequence[str], k: int) -> bool:
    gold = {str(title) for title in gold_titles if str(title)}
    if not gold:
        return False
    retrieved = {str(title) for title in list(retrieved_titles or [])[: max(int(k), 0)] if str(title)}
    return gold.issubset(retrieved)


def jaccard_at_k(left: Sequence[int], right: Sequence[int], k: int) -> float:
    left_set = set(unique_ranked(left)[: max(int(k), 0)])
    right_set = set(unique_ranked(right)[: max(int(k), 0)])
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set)) / float(len(union))


def rrf_rank(
    channels: Mapping[str, Sequence[int]],
    *,
    top_k: int,
    support_weight: float | None,
) -> List[int]:
    proposal_lists = [list(channels.get(name, []) or []) for name in CHANNEL_NAMES]
    if support_weight is None:
        weights = [1.0 for _ in CHANNEL_NAMES]
    else:
        weights = [1.0, 1.0, 1.0, float(support_weight)]
    return rank_sto_proposal_consensus_docs(
        proposal_doc_indices=proposal_lists,
        proposal_weights=weights,
        top_k=max(int(top_k), 1),
    )


def leave_one_out_rrf(
    channels: Mapping[str, Sequence[int]],
    *,
    top_k: int,
    support_weight: float | None,
) -> Dict[str, List[int]]:
    outputs: Dict[str, List[int]] = {}
    base_weights = {
        "specificity": 1.0,
        "hybrid_residual": 1.0,
        "neighborhood": 1.0,
        "support_set": 1.0 if support_weight is None else float(support_weight),
    }
    for missing in CHANNEL_NAMES:
        names = [name for name in CHANNEL_NAMES if name != missing]
        proposal_lists = [list(channels.get(name, []) or []) for name in names]
        weights = [base_weights[name] for name in names]
        outputs[f"minus_{missing}"] = rank_sto_proposal_consensus_docs(
            proposal_doc_indices=proposal_lists,
            proposal_weights=weights,
            top_k=max(int(top_k), 1),
        )
    return outputs


def doc_titles_for_indices(openie_docs: Sequence[Mapping[str, Any]], doc_indices: Sequence[int]) -> List[str]:
    return [title_for_doc(openie_docs, int(doc_idx)) for doc_idx in unique_ranked(doc_indices)]


def low_degree_shared_endpoints(
    *,
    doc_a: int,
    doc_b: int,
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> List[str]:
    doc_to_endpoints = corpus_index.get("doc_to_endpoints", {}) or {}
    endpoint_to_docs = corpus_index.get("endpoint_to_docs", {}) or {}
    left = set(doc_to_endpoints.get(int(doc_a), []) or [])
    right = set(doc_to_endpoints.get(int(doc_b), []) or [])
    shared = []
    for endpoint in sorted(left & right):
        degree = len(endpoint_to_docs.get(endpoint, []) or [])
        if degree <= int(max_endpoint_degree):
            shared.append(endpoint)
    return shared


def connected_in_sto(
    doc_indices: Sequence[int],
    *,
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> bool:
    docs = unique_ranked(doc_indices)
    if len(docs) <= 1:
        return bool(docs)
    adjacency: Dict[int, set[int]] = {int(doc): set() for doc in docs}
    for doc_a, doc_b in itertools.combinations(docs, 2):
        if low_degree_shared_endpoints(
            doc_a=int(doc_a),
            doc_b=int(doc_b),
            corpus_index=corpus_index,
            max_endpoint_degree=int(max_endpoint_degree),
        ):
            adjacency[int(doc_a)].add(int(doc_b))
            adjacency[int(doc_b)].add(int(doc_a))
    seen = {int(docs[0])}
    frontier = [int(docs[0])]
    while frontier:
        current = frontier.pop()
        for nxt in adjacency[current]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return len(seen) == len(docs)


def hub_violation(
    doc_indices: Sequence[int],
    *,
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> bool:
    docs = unique_ranked(doc_indices)
    endpoint_to_docs = corpus_index.get("endpoint_to_docs", {}) or {}
    doc_to_endpoints = corpus_index.get("doc_to_endpoints", {}) or {}
    for doc_a, doc_b in itertools.combinations(docs, 2):
        shared = set(doc_to_endpoints.get(int(doc_a), []) or []) & set(doc_to_endpoints.get(int(doc_b), []) or [])
        for endpoint in shared:
            if len(endpoint_to_docs.get(endpoint, []) or []) > int(max_endpoint_degree):
                return True
    return False


def metric_for_indices(
    *,
    gold_titles: Sequence[str],
    doc_indices: Sequence[int],
    openie_docs: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    titles = doc_titles_for_indices(openie_docs, doc_indices)
    return {
        "r5": title_recall_at_k(gold_titles, titles, 5),
        "r20": title_recall_at_k(gold_titles, titles, 20),
        "r100": title_recall_at_k(gold_titles, titles, 100),
        "all_gold_at5": 1.0 if all_gold_at_k(gold_titles, titles, 5) else 0.0,
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def summarize_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    def metrics_for(row: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = row[key]
        if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping):
            return payload["metrics"]
        return payload

    return {
        "r5": round(mean(metrics_for(row)["r5"] for row in rows), 6),
        "r20": round(mean(metrics_for(row)["r20"] for row in rows), 6),
        "r100": round(mean(metrics_for(row)["r100"] for row in rows), 6),
        "all_gold_at5": round(mean(metrics_for(row).get("all_gold_at5", 0.0) for row in rows), 6),
    }


def load_hipporag_records(path: Path) -> Dict[int, Mapping[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    payload = load_json(path)
    return {int(row.get("query_idx", idx)): row for idx, row in enumerate(payload.get("records", []) or [])}


def load_openie_docs(path: Path) -> List[Mapping[str, Any]]:
    payload = load_json(path)
    docs = payload.get("docs", []) if isinstance(payload, Mapping) else payload
    if not isinstance(docs, list):
        raise ValueError(f"OpenIE input must be a list or mapping with docs: {path}")
    return [doc for doc in docs if isinstance(doc, Mapping)]


def audit_dataset(
    *,
    dataset: str,
    samples: Sequence[Mapping[str, Any]],
    openie_docs: Sequence[Mapping[str, Any]],
    hipporag_records: Mapping[int, Mapping[str, Any]],
    config: AGSTOConfig,
    support_weights: Sequence[float],
    pool_k: int,
    max_endpoint_degree: int,
) -> Dict[str, Any]:
    retriever = AGSTORetriever.from_openie_docs(openie_docs, config=config)
    gold_docs = get_gold_docs(samples, dataset)
    rows: List[Dict[str, Any]] = []

    for query_idx, sample in enumerate(samples):
        if query_idx == 0 or (query_idx + 1) % 10 == 0 or query_idx + 1 == len(samples):
            print(f"[audit] {dataset}: query {query_idx + 1}/{len(samples)}", flush=True)
        question = str(sample.get("question") or sample.get("query") or "")
        gold_titles = gold_titles_for_sample(gold_docs[query_idx])
        proposals = build_native_sto_proposals(
            query=question,
            corpus_index=retriever.corpus_index,
            config=config,
        )
        result = retriever.retrieve(
            query=question,
            anchor_doc_indices=proposals.get("anchor_doc_indices", []),
            native_dense_doc_indices=proposals.get("native_dense_doc_indices", []),
            bm25_doc_indices=proposals.get("bm25_doc_indices", []),
            specificity_doc_indices=proposals.get("specificity_doc_indices", []),
            endpoint_transition_doc_indices=proposals.get("endpoint_transition_doc_indices", []),
            hybrid_residual_doc_indices=proposals.get("hybrid_residual_doc_indices", []),
            query_conditioned_neighborhood=proposals.get("query_conditioned_neighborhood", {}),
            support_set_search=proposals.get("support_set_search", {}),
            proposal_role_records=proposals.get("proposal_role_records", []),
            proposal_role_summary_by_doc=proposals.get("proposal_role_summary_by_doc", {}),
        )
        channels = {
            "specificity": unique_ranked(proposals.get("specificity_doc_indices", [])),
            "hybrid_residual": unique_ranked(proposals.get("hybrid_residual_doc_indices", [])),
            "neighborhood": unique_ranked(
                (proposals.get("query_conditioned_neighborhood", {}) or {}).get("retrieved_doc_indices", []) or []
            ),
            "support_set": unique_ranked(
                (proposals.get("support_set_search", {}) or {}).get("retrieved_doc_indices", []) or []
            ),
        }
        rrf_channels = {
            "specificity": channels["specificity"][: max(int(config.proposal_candidate_depth), 1)],
            "hybrid_residual": channels["hybrid_residual"][: max(int(config.proposal_candidate_depth), 1)],
            "neighborhood": channels["neighborhood"][: max(int(config.proposal_candidate_depth), 1)],
            "support_set": channels["support_set"][: max(int(config.support_proposal_depth), 1)],
        }
        current_top100 = unique_ranked(result.get("retrieved_doc_indices", []) or [])[: int(pool_k)]
        current_top5 = current_top100[:5]
        pre_completion = unique_ranked(result.get("pre_reader_order_doc_indices", []) or [])
        final_selected = unique_ranked((result.get("selected_evidence_set", {}) or {}).get("doc_indices", []) or [])
        completion_trace = result.get("graph_obligated_completion", {}) or {}
        weighted_by_support = {
            weight_key(float(weight)): rrf_rank(rrf_channels, top_k=int(pool_k), support_weight=float(weight))
            for weight in support_weights
        }
        main_weight = 1.75 if 1.75 in {round(float(value), 6) for value in support_weights} else float(support_weights[0])
        main_weight_key = weight_key(main_weight)
        main_rrf = weighted_by_support[main_weight_key]
        unweighted = rrf_rank(rrf_channels, top_k=int(pool_k), support_weight=None)
        leave_one_unweighted = leave_one_out_rrf(rrf_channels, top_k=int(pool_k), support_weight=None)
        leave_one_weighted = leave_one_out_rrf(rrf_channels, top_k=int(pool_k), support_weight=float(main_weight))
        hippo_row = hipporag_records.get(int(query_idx), {})
        hippo_titles = list(hippo_row.get("pool_titles", []) or [])
        hippo_metrics = {
            "r5": title_recall_at_k(gold_titles, hippo_titles, 5),
            "r20": title_recall_at_k(gold_titles, hippo_titles, 20),
            "r100": title_recall_at_k(gold_titles, hippo_titles, 100),
            "all_gold_at5": 1.0 if all_gold_at_k(gold_titles, hippo_titles, 5) else 0.0,
            "all_gold_at20": 1.0 if all_gold_at_k(gold_titles, hippo_titles, 20) else 0.0,
            "all_gold_at100": 1.0 if all_gold_at_k(gold_titles, hippo_titles, 100) else 0.0,
        }
        current_top100_titles = doc_titles_for_indices(openie_docs, current_top100)
        current_titles = current_top100_titles[:5]
        current_metrics_for_diff = {
            "r5": title_recall_at_k(gold_titles, current_top100_titles, 5),
            "r20": title_recall_at_k(gold_titles, current_top100_titles, 20),
            "r100": title_recall_at_k(gold_titles, current_top100_titles, 100),
            "all_gold_at5": 1.0 if all_gold_at_k(gold_titles, current_top100_titles, 5) else 0.0,
            "all_gold_at20": 1.0 if all_gold_at_k(gold_titles, current_top100_titles, 20) else 0.0,
            "all_gold_at100": 1.0 if all_gold_at_k(gold_titles, current_top100_titles, 100) else 0.0,
        }

        row = {
            "query_idx": int(query_idx),
            "question": question,
            "gold_titles": gold_titles,
            "current_agsto": {
                "top5": current_top5,
                "top20": current_top100[:20],
                "top5_titles": current_titles,
                "selection_policy": result.get("selection_policy"),
                "completion_applied": bool(completion_trace.get("applied", False)),
                "final_changed_by_completion": bool(pre_completion and final_selected and pre_completion != final_selected),
                "pre_completion_top5": pre_completion[:5],
                "pre_completion_metrics": metric_for_indices(
                    gold_titles=gold_titles,
                    doc_indices=pre_completion,
                    openie_docs=openie_docs,
                ),
                "metrics": metric_for_indices(gold_titles=gold_titles, doc_indices=current_top100, openie_docs=openie_docs),
            },
            "channels": {name: list(values) for name, values in channels.items()},
            "rrf_channel_depths": {
                name: int(len(values))
                for name, values in rrf_channels.items()
            },
            "channel_metrics": {
                name: metric_for_indices(gold_titles=gold_titles, doc_indices=values, openie_docs=openie_docs)
                for name, values in channels.items()
            },
            "rrf": {
                "unweighted_top100": unweighted,
                "unweighted_metrics": metric_for_indices(gold_titles=gold_titles, doc_indices=unweighted, openie_docs=openie_docs),
                "weighted_top100_by_support_weight": weighted_by_support,
                "weighted_metrics_by_support_weight": {
                    weight_key(float(weight)): metric_for_indices(gold_titles=gold_titles, doc_indices=docs, openie_docs=openie_docs)
                    for weight, docs in weighted_by_support.items()
                },
                "unweighted_leave_one_out": leave_one_unweighted,
                "unweighted_leave_one_out_metrics": {
                    name: metric_for_indices(gold_titles=gold_titles, doc_indices=docs, openie_docs=openie_docs)
                    for name, docs in leave_one_unweighted.items()
                },
                "weighted_leave_one_out": leave_one_weighted,
                "weighted_leave_one_out_metrics": {
                    name: metric_for_indices(gold_titles=gold_titles, doc_indices=docs, openie_docs=openie_docs)
                    for name, docs in leave_one_weighted.items()
                },
            },
            "feasibility": {
                "contains_anchor_top5": bool(set(main_rrf[:5]) & set(unique_ranked(proposals.get("anchor_doc_indices", [])))),
                "connected_top5": connected_in_sto(
                    main_rrf[:5],
                    corpus_index=retriever.corpus_index,
                    max_endpoint_degree=int(max_endpoint_degree),
                ),
                "hub_violation_top5": hub_violation(
                    main_rrf[:5],
                    corpus_index=retriever.corpus_index,
                    max_endpoint_degree=int(max_endpoint_degree),
                ),
            },
            "hipporag": {
                "top5_titles": hippo_titles[:5],
                "metrics": hippo_metrics,
            },
            "agsto_vs_hipporag": {
                "current_r5_minus_hippo_r5": round(float(current_metrics_for_diff["r5"] - hippo_metrics["r5"]), 6),
                "current_r20_minus_hippo_r20": round(float(current_metrics_for_diff["r20"] - hippo_metrics["r20"]), 6),
                "current_r100_minus_hippo_r100": round(float(current_metrics_for_diff["r100"] - hippo_metrics["r100"]), 6),
                "current_all_gold_at5": bool(current_metrics_for_diff["all_gold_at5"]),
                "current_all_gold_at20": bool(current_metrics_for_diff["all_gold_at20"]),
                "current_all_gold_at100": bool(current_metrics_for_diff["all_gold_at100"]),
            },
        }
        rows.append(row)

    return {
        "dataset": dataset,
        "rows": rows,
        "summary": summarize_dataset(rows=rows, support_weights=support_weights),
        "corpus_stats": {
            "doc_count": len(openie_docs),
            "unit_count": len(retriever.corpus_index.get("units", []) or []),
            "endpoint_count": len(retriever.corpus_index.get("endpoint_to_units", {}) or {}),
        },
    }


def summarize_dataset(*, rows: Sequence[Mapping[str, Any]], support_weights: Sequence[float]) -> Dict[str, Any]:
    denom = max(len(rows), 1)
    selection_policy_counts = Counter(str((row.get("current_agsto", {}) or {}).get("selection_policy")) for row in rows)
    channel_pairs = list(itertools.combinations(CHANNEL_NAMES, 2))
    channel_overlap = {
        f"{left}__{right}": {
            "jaccard_at20": round(mean(jaccard_at_k(row["channels"][left], row["channels"][right], 20) for row in rows), 6),
            "jaccard_at100": round(mean(jaccard_at_k(row["channels"][left], row["channels"][right], 100) for row in rows), 6),
        }
        for left, right in channel_pairs
    }
    channel_health = {}
    for channel in CHANNEL_NAMES:
        channel_health[channel] = {
            "nonempty_at20": round(mean(1.0 if row["channels"][channel][:20] else 0.0 for row in rows), 6),
            "nonempty_at100": round(mean(1.0 if row["channels"][channel][:100] else 0.0 for row in rows), 6),
            "mean_list_len": round(mean(len(row["channels"][channel]) for row in rows), 6),
            "r5": round(mean(row["channel_metrics"][channel]["r5"] for row in rows), 6),
            "r20": round(mean(row["channel_metrics"][channel]["r20"] for row in rows), 6),
            "r100": round(mean(row["channel_metrics"][channel]["r100"] for row in rows), 6),
        }
    weight_sweep = {}
    for weight in support_weights:
        row_key = weight_key(float(weight))
        weight_sweep[row_key] = {
            "r5": round(mean(row["rrf"]["weighted_metrics_by_support_weight"][row_key]["r5"] for row in rows), 6),
            "r20": round(mean(row["rrf"]["weighted_metrics_by_support_weight"][row_key]["r20"] for row in rows), 6),
            "r100": round(mean(row["rrf"]["weighted_metrics_by_support_weight"][row_key]["r100"] for row in rows), 6),
        }
    main_weight = 1.75 if 1.75 in {round(float(value), 6) for value in support_weights} else float(support_weights[0])
    main_weight_key = weight_key(main_weight)
    all_weighted = {
        "r5": round(mean(row["rrf"]["weighted_metrics_by_support_weight"][main_weight_key]["r5"] for row in rows), 6),
        "r20": round(mean(row["rrf"]["weighted_metrics_by_support_weight"][main_weight_key]["r20"] for row in rows), 6),
        "r100": round(mean(row["rrf"]["weighted_metrics_by_support_weight"][main_weight_key]["r100"] for row in rows), 6),
    }
    unweighted_all = {
        "r5": round(mean(row["rrf"]["unweighted_metrics"]["r5"] for row in rows), 6),
        "r20": round(mean(row["rrf"]["unweighted_metrics"]["r20"] for row in rows), 6),
        "r100": round(mean(row["rrf"]["unweighted_metrics"]["r100"] for row in rows), 6),
    }
    unweighted_leave_one_out = {}
    weighted_leave_one_out = {}
    for variant in [f"minus_{name}" for name in CHANNEL_NAMES]:
        unweighted_metrics = {
            "r5": round(mean(row["rrf"]["unweighted_leave_one_out_metrics"][variant]["r5"] for row in rows), 6),
            "r20": round(mean(row["rrf"]["unweighted_leave_one_out_metrics"][variant]["r20"] for row in rows), 6),
            "r100": round(mean(row["rrf"]["unweighted_leave_one_out_metrics"][variant]["r100"] for row in rows), 6),
        }
        unweighted_metrics["delta_r5_vs_all"] = round(float(unweighted_metrics["r5"]) - float(unweighted_all["r5"]), 6)
        unweighted_leave_one_out[variant] = unweighted_metrics
        weighted_metrics = {
            "r5": round(mean(row["rrf"]["weighted_leave_one_out_metrics"][variant]["r5"] for row in rows), 6),
            "r20": round(mean(row["rrf"]["weighted_leave_one_out_metrics"][variant]["r20"] for row in rows), 6),
            "r100": round(mean(row["rrf"]["weighted_leave_one_out_metrics"][variant]["r100"] for row in rows), 6),
        }
        weighted_metrics["delta_r5_vs_all"] = round(float(weighted_metrics["r5"]) - float(all_weighted["r5"]), 6)
        weighted_leave_one_out[variant] = weighted_metrics
    hippo_buckets = Counter()
    hippo_summary = {
        "r5": round(mean(row["hipporag"]["metrics"]["r5"] for row in rows), 6),
        "r20": round(mean(row["hipporag"]["metrics"]["r20"] for row in rows), 6),
        "r100": round(mean(row["hipporag"]["metrics"]["r100"] for row in rows), 6),
        "all_gold_at5": round(mean(row["hipporag"]["metrics"]["all_gold_at5"] for row in rows), 6),
        "all_gold_at20": round(mean(row["hipporag"]["metrics"]["all_gold_at20"] for row in rows), 6),
        "all_gold_at100": round(mean(row["hipporag"]["metrics"]["all_gold_at100"] for row in rows), 6),
    }
    for row in rows:
        agsto_hit = bool(row["agsto_vs_hipporag"]["current_all_gold_at5"])
        hippo_hit = bool(row["hipporag"]["metrics"]["all_gold_at5"])
        if agsto_hit and not hippo_hit:
            hippo_buckets["agsto_hit_hippo_miss"] += 1
        elif hippo_hit and not agsto_hit:
            hippo_buckets["hippo_hit_agsto_miss"] += 1
        elif agsto_hit and hippo_hit:
            hippo_buckets["both_hit"] += 1
        else:
            hippo_buckets["both_miss"] += 1
    return {
        "path_usage": {
            "selection_policy_counts": dict(selection_policy_counts),
            "consensus_nonempty_rate": round(mean(1.0 if row["rrf"]["weighted_top100_by_support_weight"][main_weight_key] else 0.0 for row in rows), 6),
            "selection_by_consensus_rate": round(float(selection_policy_counts.get("sto_proposal_consensus", 0)) / float(denom), 6),
            "completion_applied_rate": round(mean(1.0 if row["current_agsto"]["completion_applied"] else 0.0 for row in rows), 6),
            "final_changed_by_completion_rate": round(mean(1.0 if row["current_agsto"]["final_changed_by_completion"] else 0.0 for row in rows), 6),
        },
        "current_agsto": summarize_metrics(rows, "current_agsto"),
        "current_pre_completion": {
            "r5": round(mean(row["current_agsto"]["pre_completion_metrics"]["r5"] for row in rows), 6),
            "r20": round(mean(row["current_agsto"]["pre_completion_metrics"]["r20"] for row in rows), 6),
            "r100": round(mean(row["current_agsto"]["pre_completion_metrics"]["r100"] for row in rows), 6),
            "all_gold_at5": round(mean(row["current_agsto"]["pre_completion_metrics"].get("all_gold_at5", 0.0) for row in rows), 6),
        },
        "hipporag": hippo_summary,
        "unweighted_rrf": unweighted_all,
        "weighted_rrf_main": all_weighted,
        "channel_health": channel_health,
        "channel_overlap": channel_overlap,
        "unweighted_leave_one_out": unweighted_leave_one_out,
        "weighted_leave_one_out": weighted_leave_one_out,
        "leave_one_out": weighted_leave_one_out,
        "weight_sweep": weight_sweep,
        "feasibility": {
            "contains_anchor_at5": round(mean(1.0 if row["feasibility"]["contains_anchor_top5"] else 0.0 for row in rows), 6),
            "connected_at5": round(mean(1.0 if row["feasibility"]["connected_top5"] else 0.0 for row in rows), 6),
            "hub_violation_at5": round(mean(1.0 if row["feasibility"]["hub_violation_top5"] else 0.0 for row in rows), 6),
        },
        "agsto_vs_hipporag": {
            **dict(hippo_buckets),
            "mean_current_r5_minus_hippo_r5": round(mean(row["agsto_vs_hipporag"]["current_r5_minus_hippo_r5"] for row in rows), 6),
            "mean_current_r20_minus_hippo_r20": round(mean(row["agsto_vs_hipporag"]["current_r20_minus_hippo_r20"] for row in rows), 6),
            "mean_current_r100_minus_hippo_r100": round(mean(row["agsto_vs_hipporag"]["current_r100_minus_hippo_r100"] for row in rows), 6),
        },
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        table.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(table)


def write_summary_md(payload: Mapping[str, Any], path: Path) -> None:
    lines = ["# AG-STO Clean-RRF Audit", ""]
    lines.append("## Config")
    lines.append("")
    config_rows = [[key, value] for key, value in (payload.get("config", {}) or {}).items()]
    lines.append(markdown_table(["parameter", "value"], config_rows))
    for dataset in payload.get("datasets", []) or []:
        summary = dataset.get("summary", {}) or {}
        name = dataset.get("dataset")
        lines.extend(["", f"## {name}", ""])
        path_usage = summary.get("path_usage", {}) or {}
        lines.append("### Path Usage")
        lines.append(markdown_table(["metric", "value"], [[key, value] for key, value in path_usage.items()]))
        lines.extend(["", "### Retrieval Metrics", ""])
        metric_rows = [
            ["current_agsto", *[summary.get("current_agsto", {}).get(key, 0.0) for key in ("r5", "r20", "r100")]],
            ["hipporag", *[summary.get("hipporag", {}).get(key, 0.0) for key in ("r5", "r20", "r100")]],
            ["current_pre_completion", *[summary.get("current_pre_completion", {}).get(key, 0.0) for key in ("r5", "r20", "r100")]],
            ["weighted_rrf_main", *[summary.get("weighted_rrf_main", {}).get(key, 0.0) for key in ("r5", "r20", "r100")]],
            ["unweighted_rrf", *[summary.get("unweighted_rrf", {}).get(key, 0.0) for key in ("r5", "r20", "r100")]],
        ]
        lines.append(markdown_table(["variant", "R@5", "R@20", "R@100"], metric_rows))
        lines.extend(["", "### Channel Health", ""])
        channel_rows = []
        for channel, metrics in (summary.get("channel_health", {}) or {}).items():
            channel_rows.append(
                [
                    channel,
                    metrics.get("nonempty_at20"),
                    metrics.get("nonempty_at100"),
                    metrics.get("mean_list_len"),
                    metrics.get("r5"),
                    metrics.get("r20"),
                    metrics.get("r100"),
                ]
            )
        lines.append(markdown_table(["channel", "nonempty@20", "nonempty@100", "mean_len", "R@5", "R@20", "R@100"], channel_rows))
        lines.extend(["", "### Channel Overlap", ""])
        overlap_rows = [
            [pair, values.get("jaccard_at20"), values.get("jaccard_at100")]
            for pair, values in (summary.get("channel_overlap", {}) or {}).items()
        ]
        lines.append(markdown_table(["pair", "Jaccard@20", "Jaccard@100"], overlap_rows))
        lines.extend(["", "### Unweighted Leave-One-Out", ""])
        loo_rows = [
            [variant, metrics.get("r5"), metrics.get("r20"), metrics.get("r100"), metrics.get("delta_r5_vs_all")]
            for variant, metrics in (summary.get("unweighted_leave_one_out", {}) or {}).items()
        ]
        lines.append(markdown_table(["variant", "R@5", "R@20", "R@100", "Delta R@5"], loo_rows))
        lines.extend(["", "### Weighted Leave-One-Out", ""])
        loo_rows = [
            [variant, metrics.get("r5"), metrics.get("r20"), metrics.get("r100"), metrics.get("delta_r5_vs_all")]
            for variant, metrics in (summary.get("weighted_leave_one_out", {}) or {}).items()
        ]
        lines.append(markdown_table(["variant", "R@5", "R@20", "R@100", "Delta R@5"], loo_rows))
        lines.extend(["", "### Support Weight Sweep", ""])
        sweep_rows = [
            [weight, metrics.get("r5"), metrics.get("r20"), metrics.get("r100")]
            for weight, metrics in (summary.get("weight_sweep", {}) or {}).items()
        ]
        lines.append(markdown_table(["support_weight", "R@5", "R@20", "R@100"], sweep_rows))
        lines.extend(["", "### Feasibility", ""])
        lines.append(markdown_table(["metric", "value"], [[key, value] for key, value in (summary.get("feasibility", {}) or {}).items()]))
        lines.extend(["", "### AG-STO vs HippoRAG", ""])
        lines.append(markdown_table(["metric", "value"], [[key, value] for key, value in (summary.get("agsto_vs_hipporag", {}) or {}).items()]))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="2wikimultihopqa,hotpotqa,musique")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--openie_template", default=DEFAULT_OPENIE_TEMPLATE)
    parser.add_argument("--hipporag_pool_template", default=DEFAULT_HIPPORAG_POOL_TEMPLATE)
    parser.add_argument("--support_weights", default="1.0,1.25,1.5,1.75,2.0,2.25")
    parser.add_argument("--candidate_limit", type=int, default=180)
    parser.add_argument("--proposal_candidate_depth", type=int, default=24)
    parser.add_argument("--support_proposal_depth", type=int, default=12)
    parser.add_argument("--beam_size", type=int, default=12)
    parser.add_argument("--stable_anchor_k", type=int, default=2)
    parser.add_argument("--max_endpoint_degree", type=int, default=30)
    parser.add_argument("--output_dir", type=Path, default=Path("run_logs/agsto_clean_audit_limit100_20260505"))
    args = parser.parse_args(argv)

    support_weights = parse_float_csv(args.support_weights)
    if not support_weights:
        raise ValueError("--support_weights must include at least one value")
    config = AGSTOConfig(
        policy="graph",
        retrieval_top_k=max(int(args.pool_k), 5),
        evidence_set_size=5,
        stable_anchor_k=max(int(args.stable_anchor_k), 0),
        proposal_candidate_depth=max(int(args.proposal_candidate_depth), 1),
        support_proposal_depth=max(int(args.support_proposal_depth), 1),
        candidate_limit=max(int(args.candidate_limit), 5),
        beam_size=max(int(args.beam_size), 1),
        set_search_policy="beam",
        max_endpoint_degree=max(int(args.max_endpoint_degree), 2),
    )
    datasets = []
    for dataset in parse_csv(args.datasets):
        print(f"[audit] starting dataset={dataset}", flush=True)
        sample_path = args.data_root / f"{dataset}.json"
        samples = list(load_json(sample_path))
        if int(args.limit) > 0:
            samples = samples[: int(args.limit)]
        openie_path = Path(str(args.openie_template).format(dataset=dataset))
        hipporag_path = Path(str(args.hipporag_pool_template).format(dataset=dataset))
        datasets.append(
            audit_dataset(
                dataset=dataset,
                samples=samples,
                openie_docs=load_openie_docs(openie_path),
                hipporag_records=load_hipporag_records(hipporag_path),
                config=config,
                support_weights=support_weights,
                pool_k=int(args.pool_k),
                max_endpoint_degree=int(args.max_endpoint_degree),
            )
        )
        print(f"[audit] finished dataset={dataset}", flush=True)

    payload = {
        "method": "agsto_clean_rrf_audit",
        "config": {
            "datasets": ",".join(parse_csv(args.datasets)),
            "limit": int(args.limit),
            "pool_k": int(args.pool_k),
            "candidate_limit": int(args.candidate_limit),
            "proposal_candidate_depth": int(args.proposal_candidate_depth),
            "support_proposal_depth": int(args.support_proposal_depth),
            "beam_size": int(args.beam_size),
            "stable_anchor_k": int(args.stable_anchor_k),
            "max_endpoint_degree": int(args.max_endpoint_degree),
            "support_weights": ",".join(str(value) for value in support_weights),
            "openie_template": str(args.openie_template),
            "hipporag_pool_template": str(args.hipporag_pool_template),
        },
        "datasets": datasets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "audit.json"
    output_md = args.output_dir / "summary.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_md(payload, output_md)
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
