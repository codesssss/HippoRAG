#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_requirement_cache import load_dataset, resolve_save_dir  # noqa: E402
from eval_causal_qwen3 import (  # noqa: E402
    build_config,
    build_doc_text_to_chunk_id,
    compute_slice_metrics,
    extract_doc_title,
    get_gold_answers,
    get_gold_docs,
)
from oracle_subset_size_curve import build_protocol_args, compute_per_query_scores  # noqa: E402
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402
from src.hipporag.utils.misc_utils import QuerySolution  # noqa: E402


LOGGER = logging.getLogger(__name__)


def parse_selector_names(value: str) -> list[str]:
    selectors = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    valid = {"baseline", "mmr", "dpp"}
    invalid = [item for item in selectors if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported selectors: {invalid}")
    return selectors or ["baseline", "mmr", "dpp"]


def build_rank_prior_scores(num_docs: int) -> np.ndarray:
    if num_docs <= 0:
        return np.zeros(0, dtype=np.float32)
    ranks = np.arange(1, num_docs + 1, dtype=np.float32)
    scores = 1.0 / ranks
    if num_docs == 1:
        return np.ones(1, dtype=np.float32)
    scores = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-6)
    return scores.astype(np.float32)


def build_psd_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    normalized = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / np.maximum(norms, 1e-6)
    cosine = normalized @ normalized.T
    similarity = 0.5 * (cosine + 1.0)
    similarity = np.clip(similarity, 0.0, 1.0)
    similarity = 0.5 * (similarity + similarity.T)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def finalize_selected_positions(selected_positions: Sequence[int], pool_size: int, top_k: int) -> list[int]:
    chosen: list[int] = []
    seen: set[int] = set()
    for position in selected_positions:
        idx = int(position)
        if idx < 0 or idx >= pool_size or idx in seen:
            continue
        chosen.append(idx)
        seen.add(idx)
        if len(chosen) >= top_k:
            break
    if len(chosen) < top_k:
        for idx in range(pool_size):
            if idx in seen:
                continue
            chosen.append(idx)
            seen.add(idx)
            if len(chosen) >= top_k:
                break
    return chosen[:top_k]


def order_selected_positions(selected_positions: Sequence[int], order_mode: str) -> list[int]:
    if str(order_mode).strip().lower() == "selection_order":
        return [int(position) for position in selected_positions]
    return sorted(int(position) for position in selected_positions)


def select_mmr_positions(relevance_scores: np.ndarray,
                         similarity_matrix: np.ndarray,
                         top_k: int,
                         lambda_weight: float) -> list[int]:
    num_docs = int(len(relevance_scores))
    if num_docs == 0 or top_k <= 0:
        return []
    remaining = list(range(num_docs))
    selected: list[int] = []
    lambda_weight = float(np.clip(lambda_weight, 0.0, 1.0))

    first_idx = max(remaining, key=lambda idx: (float(relevance_scores[idx]), -idx))
    selected.append(first_idx)
    remaining.remove(first_idx)

    while remaining and len(selected) < top_k:
        best_idx = remaining[0]
        best_score = -float("inf")
        for idx in remaining:
            max_similarity = max(float(similarity_matrix[idx, chosen]) for chosen in selected)
            mmr_score = lambda_weight * float(relevance_scores[idx]) - (1.0 - lambda_weight) * max_similarity
            if mmr_score > best_score or (
                np.isclose(mmr_score, best_score) and float(relevance_scores[idx]) > float(relevance_scores[best_idx])
            ):
                best_score = mmr_score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def select_dpp_positions(relevance_scores: np.ndarray,
                         similarity_matrix: np.ndarray,
                         top_k: int,
                         quality_power: float,
                         diagonal_eps: float = 1e-6) -> list[int]:
    num_docs = int(len(relevance_scores))
    if num_docs == 0 or top_k <= 0:
        return []

    quality = np.maximum(np.asarray(relevance_scores, dtype=np.float64), diagonal_eps)
    quality = np.power(quality, float(max(quality_power, 1e-6)))
    kernel = np.outer(quality, quality) * np.asarray(similarity_matrix, dtype=np.float64)
    kernel = 0.5 * (kernel + kernel.T)
    np.fill_diagonal(kernel, np.maximum(np.diag(kernel), diagonal_eps))

    remaining = list(range(num_docs))
    selected: list[int] = []
    while remaining and len(selected) < top_k:
        best_idx = remaining[0]
        best_score = -float("inf")
        for idx in remaining:
            subset = selected + [idx]
            sub_kernel = kernel[np.ix_(subset, subset)] + np.eye(len(subset), dtype=np.float64) * diagonal_eps
            sign, logdet = np.linalg.slogdet(sub_kernel)
            score = float(logdet) if sign > 0 else -float("inf")
            if score > best_score or (
                np.isclose(score, best_score) and float(relevance_scores[idx]) > float(relevance_scores[best_idx])
            ):
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def subset_by_indices(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[idx] for idx in indices]


def compute_bucket_metrics(config: Any,
                           query_solutions: Sequence[QuerySolution],
                           gold_docs: Sequence[Sequence[str]],
                           gold_answers: Sequence[Sequence[str]]) -> Dict[str, Dict[str, Any]]:
    bucket_to_indices: Dict[str, list[int]] = {}
    for idx, docs in enumerate(gold_docs):
        bucket_name = f"{len(docs)}_doc"
        bucket_to_indices.setdefault(bucket_name, []).append(idx)
    bucket_metrics: Dict[str, Dict[str, Any]] = {}
    for bucket_name, indices in sorted(bucket_to_indices.items()):
        subset_metrics = compute_slice_metrics(
            config=config,
            query_solutions=subset_by_indices(query_solutions, indices),
            gold_docs=subset_by_indices(gold_docs, indices),
            gold_answers=subset_by_indices(gold_answers, indices),
        )["overall"]
        bucket_metrics[bucket_name] = subset_metrics
    return bucket_metrics


def load_report_examples(report_payload: Dict[str, Any], limit: int, random_subset_seed: int) -> list[Dict[str, Any]]:
    examples = list(report_payload.get("examples", []) or [])
    if limit and limit > 0 and len(examples) > int(limit):
        if int(random_subset_seed) >= 0:
            rng = np.random.default_rng(int(random_subset_seed))
            indices = sorted(rng.choice(len(examples), size=int(limit), replace=False).tolist())
            examples = [examples[idx] for idx in indices]
        else:
            examples = examples[: int(limit)]
    return examples


def build_context(args: argparse.Namespace,
                  report_payload: Dict[str, Any]) -> Dict[str, Any]:
    corpus, samples = load_dataset(args.dataset, 0)
    sample_by_question = {str(sample["question"]): sample for sample in samples}
    all_gold_docs = get_gold_docs(samples, args.dataset, corpus=corpus)
    all_gold_answers = get_gold_answers(samples)
    gold_docs_by_question = {
        str(samples[idx]["question"]): list(all_gold_docs[idx])
        for idx in range(len(samples))
    }
    gold_answers_by_question = {
        str(samples[idx]["question"]): list(all_gold_answers[idx])
        for idx in range(len(samples))
    }

    resolved_args = argparse.Namespace(**vars(args))
    if not getattr(resolved_args, "save_dir", ""):
        report_config = dict(report_payload.get("config", {}) or {})
        resolved_args.save_dir = str(report_config.get("save_dir", "outputs_step0_general") or "outputs_step0_general")
    if not getattr(resolved_args, "llm_name", ""):
        resolved_args.llm_name = str(report_payload.get("llm_name", "") or "qwen3-8b")
    if not getattr(resolved_args, "llm_request_name", ""):
        resolved_args.llm_request_name = str(report_payload.get("llm_name", "") or resolved_args.llm_name)
    if not getattr(resolved_args, "llm_base_url", ""):
        resolved_args.llm_base_url = str(report_payload.get("llm_base_url", "") or "")
    if not getattr(resolved_args, "embedding_name", ""):
        resolved_args.embedding_name = str(report_payload.get("embedding_name", "") or "VLLM//mnt/nvme/Qwen3-Embedding-8B")
    if not getattr(resolved_args, "embedding_base_url", ""):
        resolved_args.embedding_base_url = str(report_payload.get("embedding_base_url", "") or "")

    config = build_config(build_protocol_args(resolved_args), corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    if not getattr(hipporag, "ready_to_retrieve", False):
        hipporag.prepare_retrieval_objects()
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    chunk_id_to_doc_text = {
        chunk_id: doc_text
        for doc_text, chunk_id in doc_text_to_chunk_id.items()
    }

    report_examples = load_report_examples(report_payload, args.limit, args.random_subset_seed)
    query_items: list[Dict[str, Any]] = []
    baseline_solutions: list[QuerySolution] = []
    gold_docs: list[list[str]] = []
    gold_answers: list[list[str]] = []

    for example in report_examples:
        question = str(example.get("question", ""))
        if question not in sample_by_question or question not in gold_docs_by_question:
            continue
        retrieved_doc_ids = [
            chunk_id for chunk_id in list(example.get("retrieved_doc_ids", []) or [])
            if isinstance(chunk_id, str) and chunk_id in chunk_id_to_doc_text
        ]
        reconstructed_docs = [chunk_id_to_doc_text[chunk_id] for chunk_id in retrieved_doc_ids]
        if not reconstructed_docs:
            continue

        query_items.append(
            {
                "question": question,
                "sample": sample_by_question[question],
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_docs": reconstructed_docs,
                "answer": str(example.get("answer", "") or ""),
            }
        )
        gold_docs.append(list(gold_docs_by_question[question]))
        gold_answers.append(list(gold_answers_by_question[question]))
        baseline_solutions.append(
            QuerySolution(
                question=question,
                docs=list(reconstructed_docs),
                doc_scores=build_rank_prior_scores(len(reconstructed_docs)),
                answer=str(example.get("answer", "") or ""),
                gold_answers=list(gold_answers_by_question[question]),
                gold_docs=list(gold_docs_by_question[question]),
            )
        )

    return {
        "hipporag": hipporag,
        "config": config,
        "query_items": query_items,
        "baseline_solutions": baseline_solutions,
        "gold_docs": gold_docs,
        "gold_answers": gold_answers,
    }


def build_selector_query_solutions(hipporag: HippoRAG,
                                   query_items: Sequence[Dict[str, Any]],
                                   gold_docs: Sequence[Sequence[str]],
                                   pool_k: int,
                                   qa_top_k: int,
                                   selector_name: str,
                                   mmr_lambda: float,
                                   dpp_quality_power: float,
                                   order_mode: str) -> tuple[list[QuerySolution], list[Dict[str, Any]]]:
    selector_solutions: list[QuerySolution] = []
    selector_traces: list[Dict[str, Any]] = []
    passage_node_key_to_doc_idx = dict(getattr(hipporag, "passage_node_key_to_doc_idx", {}) or {})
    if not passage_node_key_to_doc_idx:
        passage_node_keys = list(getattr(hipporag, "passage_node_keys", []) or [])
        passage_node_key_to_doc_idx = {
            str(chunk_id): int(idx)
            for idx, chunk_id in enumerate(passage_node_keys)
        }

    for q_idx, item in enumerate(query_items):
        pool_chunk_ids = list(item["retrieved_doc_ids"][:pool_k])
        pool_docs = list(item["retrieved_docs"][:pool_k])
        available_positions: list[int] = []
        doc_indices: list[int] = []
        for idx, chunk_id in enumerate(pool_chunk_ids):
            mapped = passage_node_key_to_doc_idx.get(chunk_id)
            if mapped is None:
                continue
            available_positions.append(idx)
            doc_indices.append(int(mapped))
        if not available_positions:
            selector_solutions.append(
                QuerySolution(
                    question=item["question"],
                    docs=list(pool_docs[:qa_top_k]),
                    doc_scores=build_rank_prior_scores(min(len(pool_docs), qa_top_k)),
                    gold_docs=list(gold_docs[q_idx]),
                )
            )
            selector_traces.append({"selected_positions": list(range(min(len(pool_docs), qa_top_k)))})
            continue

        pool_embeddings = np.asarray(hipporag.passage_embeddings[doc_indices], dtype=np.float32)
        similarity = build_psd_similarity_matrix(pool_embeddings)
        relevance = build_rank_prior_scores(len(available_positions))

        if selector_name == "mmr":
            local_selected = select_mmr_positions(
                relevance_scores=relevance,
                similarity_matrix=similarity,
                top_k=min(qa_top_k, len(available_positions)),
                lambda_weight=mmr_lambda,
            )
        elif selector_name == "dpp":
            local_selected = select_dpp_positions(
                relevance_scores=relevance,
                similarity_matrix=similarity,
                top_k=min(qa_top_k, len(available_positions)),
                quality_power=dpp_quality_power,
            )
        else:
            raise ValueError(f"Unsupported selector_name: {selector_name}")

        selected_positions = [available_positions[idx] for idx in local_selected]
        selected_positions = finalize_selected_positions(selected_positions, len(pool_docs), qa_top_k)
        selected_positions = order_selected_positions(selected_positions, order_mode=order_mode)
        selected_docs = [pool_docs[idx] for idx in selected_positions]
        selected_scores = np.asarray(
            [build_rank_prior_scores(len(pool_docs))[idx] for idx in selected_positions],
            dtype=np.float32,
        )
        selector_solutions.append(
            QuerySolution(
                question=item["question"],
                docs=selected_docs,
                doc_scores=selected_scores,
                gold_docs=list(gold_docs[q_idx]),
            )
        )
        selector_traces.append(
            {
                "selected_positions": [int(idx) for idx in selected_positions],
                "selected_titles": [extract_doc_title(doc_text) for doc_text in selected_docs],
                "pool_top_titles": [extract_doc_title(doc_text) for doc_text in pool_docs[:qa_top_k]],
            }
        )

    return selector_solutions, selector_traces


def summarize_baseline(config: Any,
                       baseline_solutions: Sequence[QuerySolution],
                       gold_docs: Sequence[Sequence[str]],
                       gold_answers: Sequence[Sequence[str]]) -> Dict[str, Any]:
    overall = compute_slice_metrics(
        config=config,
        query_solutions=list(baseline_solutions),
        gold_docs=list(gold_docs),
        gold_answers=list(gold_answers),
    )["overall"]
    per_query_scores = compute_per_query_scores(
        gold_answers=list(gold_answers),
        query_solutions=list(baseline_solutions),
    )
    bucket_metrics = compute_bucket_metrics(
        config=config,
        query_solutions=baseline_solutions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )
    return {
        "overall": overall,
        "per_query_scores": per_query_scores,
        "per_bucket": bucket_metrics,
    }


def summarize_selected(config: Any,
                       gold_docs: Sequence[Sequence[str]],
                       gold_answers: Sequence[Sequence[str]],
                       query_solutions: Sequence[QuerySolution]) -> Dict[str, Any]:
    overall = compute_slice_metrics(
        config=config,
        query_solutions=list(query_solutions),
        gold_docs=list(gold_docs),
        gold_answers=list(gold_answers),
    )["overall"]
    per_query_scores = compute_per_query_scores(
        gold_answers=list(gold_answers),
        query_solutions=list(query_solutions),
    )
    bucket_metrics = compute_bucket_metrics(
        config=config,
        query_solutions=query_solutions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )
    return {
        "overall": overall,
        "per_query_scores": per_query_scores,
        "per_bucket": bucket_metrics,
    }


def clone_query_solution(query_solution: QuerySolution) -> QuerySolution:
    return QuerySolution(
        question=str(query_solution.question),
        docs=list(query_solution.docs),
        doc_scores=(
            np.asarray(query_solution.doc_scores, dtype=float).copy()
            if query_solution.doc_scores is not None
            else None
        ),
        answer=query_solution.answer,
        gold_answers=list(query_solution.gold_answers) if query_solution.gold_answers is not None else None,
        gold_docs=list(query_solution.gold_docs) if query_solution.gold_docs is not None else None,
        retrieval_trace=query_solution.retrieval_trace,
        qa_trace=query_solution.qa_trace,
    )


def run_fresh_qa(hipporag: HippoRAG,
                 query_solutions: Sequence[QuerySolution],
                 gold_docs: Sequence[Sequence[str]],
                 gold_answers: Sequence[Sequence[str]]) -> list[QuerySolution]:
    evaluated_solutions, _, _, _, _ = hipporag.rag_qa(
        queries=[clone_query_solution(query_solution) for query_solution in query_solutions],
        gold_docs=list(gold_docs),
        gold_answers=list(gold_answers),
    )
    return evaluated_solutions


def compute_delta_summary(baseline_scores: Sequence[Dict[str, float]],
                          method_scores: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    deltas = [
        float(method_scores[idx]["F1"]) - float(baseline_scores[idx]["F1"])
        for idx in range(min(len(baseline_scores), len(method_scores)))
    ]
    improved = sum(delta > 1e-6 for delta in deltas)
    worsened = sum(delta < -1e-6 for delta in deltas)
    return {
        "improved_queries": int(improved),
        "worsened_queries": int(worsened),
        "unchanged_queries": int(len(deltas) - improved - worsened),
        "mean_delta_F1": round(float(np.mean(deltas)) if deltas else 0.0, 4),
    }


def attach_overall_deltas(summary: Dict[str, Any],
                          baseline_summary: Dict[str, Any]) -> None:
    overall = dict(summary.get("overall", {}) or {})
    baseline_overall = dict(baseline_summary.get("overall", {}) or {})
    if overall and baseline_overall:
        overall["delta_em_vs_baseline"] = round(
            float(overall.get("ExactMatch", 0.0)) - float(baseline_overall.get("ExactMatch", 0.0)),
            4,
        )
        overall["delta_f1_vs_baseline"] = round(
            float(overall.get("F1", 0.0)) - float(baseline_overall.get("F1", 0.0)),
            4,
        )
        summary["overall"] = overall


def extract_bridge_report_summary(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    qa_block = dict(payload.get("setwise_selector_qa", {}) or {})
    return {
        "report_path": path,
        "selector": str(qa_block.get("selector", "bridge_beam")),
        "overall": {
            "ExactMatch": qa_block.get("selector_EM"),
            "F1": qa_block.get("selector_F1"),
            "EM_delta": qa_block.get("EM_delta"),
            "F1_delta": qa_block.get("F1_delta"),
        },
        "per_bucket": dict(qa_block.get("per_bucket", {}) or {}),
    }


def extract_oracle_report_summary(path: str, pool_k: int) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    qa_block = dict(payload.get("oracle_select_qa", {}) or {})
    sweep = dict(qa_block.get("sweep", {}) or {})
    key = f"K={int(pool_k)}"
    selected = dict(sweep.get(key, {}) or {})
    return {
        "report_path": path,
        "pool_k": int(pool_k),
        "overall": {
            "ExactMatch": selected.get("oracle_select_EM"),
            "F1": selected.get("oracle_select_F1"),
            "EM_delta": selected.get("EM_delta"),
            "F1_delta": selected.get("F1_delta"),
            "full_support_in_pool_rate": selected.get("full_support_in_pool_rate"),
        },
        "per_bucket": dict(selected.get("bucket_breakdown", {}) or {}),
    }


def build_markdown(report: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Diversity Selector Study: {report['dataset']}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Report: `{report['source_report']}`")
    lines.append(f"- Queries: `{report['num_queries']}`")
    lines.append(f"- Pool K: `{report['pool_k']}`")
    lines.append(f"- QA top-k: `{report['qa_top_k']}`")
    lines.append(f"- Order mode: `{report['order_mode']}`")
    lines.append(f"- MMR lambda: `{report['mmr_lambda']}`")
    lines.append(f"- DPP quality power: `{report['dpp_quality_power']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Method | EM | F1 | ΔEM | ΔF1 |")
    lines.append("|---|---:|---:|---:|---:|")
    baseline = report["baseline"]
    baseline_overall = baseline["overall"]
    lines.append(
        f"| Baseline top-5 | {baseline_overall['ExactMatch']:.4f} | {baseline_overall['F1']:.4f} | — | — |"
    )
    for selector_name in report["selector_order"]:
        selector = report["selectors"][selector_name]
        overall = selector["overall"]
        lines.append(
            f"| {selector_name.upper()} | {overall['ExactMatch']:.4f} | {overall['F1']:.4f} | "
            f"{overall['ExactMatch'] - baseline_overall['ExactMatch']:+.4f} | "
            f"{overall['F1'] - baseline_overall['F1']:+.4f} |"
        )
    if report.get("bridge_report"):
        overall = report["bridge_report"]["overall"]
        lines.append(
            f"| Bridge-Beam | {float(overall['ExactMatch']):.4f} | {float(overall['F1']):.4f} | "
            f"{float(overall['EM_delta']):+.4f} | {float(overall['F1_delta']):+.4f} |"
        )
    if report.get("oracle_report"):
        overall = report["oracle_report"]["overall"]
        lines.append(
            f"| Oracle@{report['pool_k']} | {float(overall['ExactMatch']):.4f} | {float(overall['F1']):.4f} | "
            f"{float(overall['EM_delta']):+.4f} | {float(overall['F1_delta']):+.4f} |"
        )

    lines.append("")
    lines.append("## Query-Level Delta vs Baseline")
    lines.append("")
    lines.append("| Method | Improved | Worsened | Unchanged | Mean ΔF1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for selector_name in report["selector_order"]:
        delta = report["selectors"][selector_name]["delta_vs_baseline"]
        lines.append(
            f"| {selector_name.upper()} | {delta['improved_queries']} | {delta['worsened_queries']} | "
            f"{delta['unchanged_queries']} | {delta['mean_delta_F1']:+.4f} |"
        )

    lines.append("")
    lines.append("## Bucket Breakdown")
    lines.append("")
    bucket_names = sorted(report["baseline"]["per_bucket"].keys())
    for bucket_name in bucket_names:
        lines.append(f"### {bucket_name}")
        lines.append("")
        lines.append("| Method | EM | F1 | ΔEM | ΔF1 |")
        lines.append("|---|---:|---:|---:|---:|")
        bucket = report["baseline"]["per_bucket"][bucket_name]
        lines.append(f"| Baseline top-5 | {bucket['ExactMatch']:.4f} | {bucket['F1']:.4f} | — | — |")
        for selector_name in report["selector_order"]:
            selector_bucket = report["selectors"][selector_name]["per_bucket"][bucket_name]
            lines.append(
                f"| {selector_name.upper()} | {selector_bucket['ExactMatch']:.4f} | {selector_bucket['F1']:.4f} | "
                f"{selector_bucket['ExactMatch'] - bucket['ExactMatch']:+.4f} | "
                f"{selector_bucket['F1'] - bucket['F1']:+.4f} |"
            )
        if report.get("bridge_report") and bucket_name in report["bridge_report"]["per_bucket"]:
            bridge_bucket = report["bridge_report"]["per_bucket"][bucket_name]
            lines.append(
                f"| Bridge-Beam | {float(bridge_bucket['selector_EM']):.4f} | {float(bridge_bucket['selector_F1']):.4f} | "
                f"{float(bridge_bucket['EM_delta']):+.4f} | {float(bridge_bucket['F1_delta']):+.4f} |"
            )
        if report.get("oracle_report"):
            oracle_bucket = report["oracle_report"]["per_bucket"].get(bucket_name.replace("_", "-"))
            if oracle_bucket is not None:
                lines.append(
                    f"| Oracle@{report['pool_k']} | {float(oracle_bucket['EM']):.4f} | {float(oracle_bucket['F1']):.4f} | "
                    f"{float(oracle_bucket['EM']) - float(bucket['ExactMatch']):+.4f} | {float(oracle_bucket['F1']) - float(bucket['F1']):+.4f} |"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_json = Path(args.output_json) if args.output_json else (
        ROOT_DIR / "outputs_smoke" / f"{args.dataset}_diversity_selector_study.json"
    )
    output_md = Path(args.output_md) if args.output_md else (
        ROOT_DIR / "outputs_smoke" / f"{args.dataset}_diversity_selector_study.md"
    )
    return output_json, output_md


def write_report_checkpoint(report: Dict[str, Any],
                            output_json: Path,
                            output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare structure-blind diversity selectors on a frozen fixed pool.")
    parser.add_argument("--dataset", required=True, choices=["2wikimultihopqa", "hotpotqa", "musique"])
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--bridge_report_json", default="")
    parser.add_argument("--oracle_report_json", default="")
    parser.add_argument("--pool_k", type=int, default=100)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random_subset_seed", type=int, default=-1)
    parser.add_argument("--selectors", type=str, default="baseline,mmr,dpp")
    parser.add_argument("--order_mode", choices=["original_rank", "selection_order"], default="original_rank")
    parser.add_argument("--mmr_lambda", type=float, default=0.7)
    parser.add_argument("--dpp_quality_power", type=float, default=1.0)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--output_md", type=str, default="")
    parser.add_argument("--resume_from_output", action="store_true")

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--llm_name", type=str, default="")
    parser.add_argument("--llm_request_name", type=str, default="")
    parser.add_argument("--llm_base_url", type=str, default="")
    parser.add_argument("--embedding_name", type=str, default="")
    parser.add_argument("--embedding_base_url", type=str, default="")
    parser.add_argument("--force_index_from_scratch", action="store_true")
    parser.add_argument("--force_openie_from_scratch", action="store_true")
    parser.add_argument("--max_retry_attempts", type=int, default=3)
    parser.add_argument("--openie_mode", type=str, default="online")
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=1)
    parser.add_argument("--causal_enabled", type=str, default="false")
    parser.add_argument("--causal_query_only", type=str, default="false")
    parser.add_argument("--causal_gate_mode", type=str, default="none")
    parser.add_argument("--causal_seed_top_k", type=int, default=5)
    parser.add_argument("--causal_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--causal_damping", type=float, default=0.1)
    parser.add_argument("--causal_blend_dense_weight", type=float, default=0.2)
    parser.add_argument("--causal_blend_fact_weight", type=float, default=0.2)
    parser.add_argument("--causal_blend_graph_weight", type=float, default=0.6)
    parser.add_argument("--causal_margin_gate_enabled", type=str, default="false")
    parser.add_argument("--causal_margin_threshold", type=float, default=0.0)
    parser.add_argument("--causal_blend_top_k", type=int, default=5)
    parser.add_argument("--causal_engine_version", type=str, default="v2")
    parser.add_argument("--causal_v2_probe_mode", type=str, default="off")
    parser.add_argument("--causal_v2_graph_mode", type=str, default="legacy")
    parser.add_argument("--causal_v2_base_retrieval_mode", type=str, default="legacy_fact_graph")
    parser.add_argument("--general_graph_related_to_weight", type=float, default=0.0)
    parser.add_argument("--general_graph_seed_top_k", type=int, default=5)
    parser.add_argument("--causal_v2_extraction_max_tokens", type=int, default=1024)
    parser.add_argument("--causal_v2_extraction_retry_attempts", type=int, default=3)
    parser.add_argument("--causal_v2_extraction_workers", type=int, default=4)
    parser.add_argument("--causal_event_top_k", type=int, default=10)
    parser.add_argument("--causal_v2_max_hops", type=int, default=2)
    parser.add_argument("--causal_chain_top_k", type=int, default=5)
    parser.add_argument("--causal_context_max_items", type=int, default=8)
    parser.add_argument("--causal_er_similarity_threshold", type=float, default=0.75)
    parser.add_argument("--causal_er_text_threshold", type=float, default=0.0)
    parser.add_argument("--causal_v2_min_edge_confidence", type=float, default=0.0)
    parser.add_argument("--structure_rerank_enabled", type=str, default="false")
    parser.add_argument("--structure_rerank_top_n", type=int, default=20)
    parser.add_argument("--structure_rerank_bonus_weight", type=float, default=0.0)
    parser.add_argument("--structure_rerank_min_edge_support", type=int, default=1)
    parser.add_argument("--structure_rerank_max_top5_swaps", type=int, default=5)
    parser.add_argument("--structure_rerank_seed_top_k", type=int, default=5)
    parser.add_argument("--structure_rerank_max_hops", type=int, default=2)
    parser.add_argument("--structure_rerank_margin_threshold", type=float, default=0.0)
    parser.add_argument("--rerank_require_non_empty", type=str, default="false")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    report_payload = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    selectors = parse_selector_names(args.selectors)
    output_json, output_md = resolve_output_paths(args)
    existing_report: Dict[str, Any] | None = None
    if args.resume_from_output and output_json.exists():
        existing_report = json.loads(output_json.read_text(encoding="utf-8"))
        LOGGER.info("Loaded resume checkpoint from %s", output_json)
    context = build_context(args=args, report_payload=report_payload)
    hipporag: HippoRAG = context["hipporag"]
    config = context["config"]
    query_items = context["query_items"]
    baseline_solutions = context["baseline_solutions"]
    gold_docs = context["gold_docs"]
    gold_answers = context["gold_answers"]

    if existing_report and existing_report.get("baseline") and int(existing_report.get("num_queries", -1)) == len(query_items):
        baseline_summary = dict(existing_report["baseline"])
        LOGGER.info("Reusing baseline from checkpoint for %s (%d queries)", args.dataset, len(query_items))
    else:
        LOGGER.info("Running baseline on %s (%d queries)", args.dataset, len(query_items))
        baseline_eval_solutions = run_fresh_qa(
            hipporag=hipporag,
            query_solutions=baseline_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )
        baseline_summary = summarize_baseline(
            config=config,
            baseline_solutions=baseline_eval_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )
        baseline_summary["reader_mode"] = "fresh_rag_qa"

    selector_reports: Dict[str, Any] = dict(existing_report.get("selectors", {}) if existing_report else {})
    selector_order: list[str] = list(existing_report.get("selector_order", []) if existing_report else [])
    report: Dict[str, Any] = {
        "dataset": args.dataset,
        "source_report": args.report_json,
        "num_queries": len(query_items),
        "pool_k": int(args.pool_k),
        "qa_top_k": int(args.qa_top_k),
        "order_mode": args.order_mode,
        "mmr_lambda": float(args.mmr_lambda),
        "dpp_quality_power": float(args.dpp_quality_power),
        "baseline": baseline_summary,
        "selectors": selector_reports,
        "selector_order": selector_order,
        "bridge_report": extract_bridge_report_summary(args.bridge_report_json) if args.bridge_report_json else None,
        "oracle_report": extract_oracle_report_summary(args.oracle_report_json, pool_k=args.pool_k) if args.oracle_report_json else None,
        "checkpoint_status": "baseline_complete",
    }
    write_report_checkpoint(report, output_json, output_md)

    for selector_name in selectors:
        if selector_name == "baseline":
            continue
        if selector_name in selector_reports:
            LOGGER.info("Skipping %s from checkpoint", selector_name)
            continue
        LOGGER.info("Running %s on %s (%d queries)", selector_name, args.dataset, len(query_items))
        selector_solutions, selector_traces = build_selector_query_solutions(
            hipporag=hipporag,
            query_items=query_items,
            gold_docs=gold_docs,
            pool_k=args.pool_k,
            qa_top_k=args.qa_top_k,
            selector_name=selector_name,
            mmr_lambda=args.mmr_lambda,
            dpp_quality_power=args.dpp_quality_power,
            order_mode=args.order_mode,
        )
        selector_solutions = run_fresh_qa(
            hipporag=hipporag,
            query_solutions=selector_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )
        selector_summary = summarize_selected(
            config=config,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
            query_solutions=selector_solutions,
        )
        selector_summary["reader_mode"] = "fresh_rag_qa"
        attach_overall_deltas(selector_summary, baseline_summary)
        selector_summary["delta_vs_baseline"] = compute_delta_summary(
            baseline_scores=baseline_summary["per_query_scores"],
            method_scores=selector_summary["per_query_scores"],
        )
        selector_summary["selector_traces"] = selector_traces
        selector_reports[selector_name] = selector_summary
        selector_order.append(selector_name)
        report["checkpoint_status"] = f"{selector_name}_complete"
        write_report_checkpoint(report, output_json, output_md)

    report["checkpoint_status"] = "complete"
    write_report_checkpoint(report, output_json, output_md)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
