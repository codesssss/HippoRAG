#!/usr/bin/env python3
"""Run ETv3-pool DBEC safe-selector frontier without reader QA.

This script is intentionally selector-only.  It reuses the frozen ETv3
external-pool JSONs and existing DBEC binding caches, applies
``daec_noisyor_safe_llm`` under different preservation/admission settings, and
writes set-level retrieval/composition metrics.  It does not call the reader and
does not modify ETv3 retrieval.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_causal_qwen3 as eval_causal_module  # noqa: E402
from eval_causal_qwen3 import (  # noqa: E402
    apply_setwise_selector,
    build_doc_text_to_chunk_id,
    extract_doc_title,
    get_gold_answers,
    get_gold_docs,
    install_qwen_disable_thinking,
    load_external_pool_query_solutions,
    resolve_dataset_file_stem,
)
from dtc_embed_utils import DTCRequirement  # noqa: E402
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402
from src.hipporag.utils.config_utils import BaseConfig  # noqa: E402
from src.hipporag.utils.misc_utils import QuerySolution  # noqa: E402


DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DEFAULT_VARIANTS = ("top5_max0", "top4_max1", "top3_max2", "top2_max3", "top1_max2")
DEFAULT_POOL_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/pools")
DEFAULT_BINDING_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/evals")
DEFAULT_REQUIREMENT_REPORT_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/evals")
DEFAULT_OUTPUT_ROOT = Path("run_logs/etv3_dbec_selector_frontier_full1000_20260510")
DEFAULT_SAVE_DIR = "outputs_step0_general_nvembed"
DATASET_PORTS = {
    "musique": 8041,
    "hotpotqa": 8042,
    "2wikimultihopqa": 8043,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_counter(titles: Sequence[Any], *, k: int | None = None) -> Counter[str]:
    selected = list(titles[:k] if k is not None else titles)
    return Counter(title for title in (normalize_title(item) for item in selected) if title)


def title_all_covered(gold_titles: Sequence[Any], candidate_titles: Sequence[Any], *, k: int) -> bool:
    gold = title_counter(gold_titles)
    if not gold:
        return False
    have = title_counter(candidate_titles, k=k)
    return all(have[title] >= count for title, count in gold.items())


def title_recall(gold_titles: Sequence[Any], candidate_titles: Sequence[Any], *, k: int) -> float:
    gold = title_counter(gold_titles)
    if not gold:
        return 0.0
    have = title_counter(candidate_titles, k=k)
    hit = sum(min(count, have[title]) for title, count in gold.items())
    return hit / sum(gold.values())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        output = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(output) or math.isinf(output):
        return default
    return output


def parse_csv(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = list(value)
    return [str(item).strip() for item in raw if str(item).strip()]


def parse_variant(value: str) -> tuple[str, int, int]:
    label = str(value).strip().lower().replace("-", "_").replace("/", "_")
    match = re.fullmatch(r"top(\d+)_max(\d+)", label)
    if not match:
        raise ValueError(f"Variant must look like top4_max1, got: {value}")
    preserve_top_m = int(match.group(1))
    max_swaps = int(match.group(2))
    return label, preserve_top_m, max_swaps


def default_pool_path(dataset: str, pool_k: int, limit: int, pool_root: Path) -> Path:
    return pool_root / f"{dataset}_etv3_pool{pool_k}_limit{limit}.json"


def default_binding_cache_path(dataset: str, pool_k: int, binding_root: Path) -> Path:
    return binding_root / f"{dataset}_etv3_pool{pool_k}_dbec_latest.binding_cache.json"


def default_requirement_report_path(dataset: str, pool_k: int, limit: int, requirement_report_root: Path) -> Path:
    return requirement_report_root / f"{dataset}_etv3_pool{pool_k}_dbec_stable_limit{limit}.json"


def default_output_path(output_root: Path, dataset: str, pool_k: int, variant: str, limit: int) -> Path:
    return output_root / "evals" / f"{dataset}_etv3_pool{pool_k}_dbec_{variant}_selector_limit{limit}.json"


def build_config(args: argparse.Namespace, dataset: str, corpus_len: int) -> BaseConfig:
    return BaseConfig(
        save_dir=str(args.save_dir),
        llm_name=str(args.llm_name),
        llm_request_name=str(args.llm_request_name),
        llm_base_url=str(args.llm_base_url or f"http://localhost:{DATASET_PORTS.get(dataset, 8043)}/v1"),
        embedding_model_name=str(args.embedding_name),
        embedding_base_url=str(args.embedding_base_url),
        max_retry_attempts=int(args.max_retry_attempts),
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        openie_mode="online",
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=int(args.qa_top_k),
        qa_doc_max_chars=int(args.qa_doc_max_chars),
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=int(args.embedding_batch_size),
        corpus_len=int(corpus_len),
        dataset=dataset,
        causal_enabled=False,
        causal_query_only=True,
        causal_engine_version="v2",
        causal_v2_base_retrieval_mode="dense",
        structure_rerank_enabled=False,
    )


def load_dataset(dataset: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[list[str]], list[list[str]]]:
    stem = resolve_dataset_file_stem(dataset)
    corpus_path = ROOT_DIR / "reproduce" / "dataset" / f"{stem}_corpus.json"
    sample_path = ROOT_DIR / "reproduce" / "dataset" / f"{stem}.json"
    corpus = read_json(corpus_path)
    samples = read_json(sample_path)
    if int(limit) > 0:
        samples = samples[: int(limit)]
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset, corpus=corpus)
    return corpus, samples, docs, gold_docs, gold_answers


def clone_solution_with_docs(source: QuerySolution, docs: Sequence[str]) -> QuerySolution:
    return QuerySolution(
        question=source.question,
        docs=list(docs),
        doc_scores=np.asarray(source.doc_scores, dtype=float) if source.doc_scores is not None else None,
        answer=None,
        gold_answers=list(source.gold_answers or []),
        gold_docs=list(source.gold_docs or []),
        retrieval_trace=dict(source.retrieval_trace or {}),
    )


def make_noop_selected_solutions(query_solutions: Sequence[QuerySolution]) -> list[QuerySolution]:
    selected: list[QuerySolution] = []
    for qs in query_solutions:
        trace = dict(qs.retrieval_trace or {})
        trace["setwise_selector_trace"] = {
            "selector": "no_op",
            "selection_steps": [],
            "daec_safe_projection": True,
            "daec_safe_preserve_top_m": 5,
            "daec_safe_max_swaps": 0,
        }
        selected.append(
            QuerySolution(
                question=qs.question,
                docs=list(qs.docs),
                doc_scores=np.asarray(qs.doc_scores, dtype=float) if qs.doc_scores is not None else None,
                answer=None,
                gold_answers=list(qs.gold_answers or []),
                gold_docs=list(qs.gold_docs or []),
                retrieval_trace=trace,
            )
        )
    return selected


def requirement_from_trace(payload: Mapping[str, Any]) -> DTCRequirement:
    return DTCRequirement(
        unit_id=str(payload.get("unit_id") or payload.get("id") or ""),
        subquery=str(payload.get("subquery") or ""),
        depends_on=tuple(str(item) for item in list(payload.get("depends_on") or [])),
        expected_answer_type=str(payload.get("expected_answer_type") or "unknown"),
        anchor_mentions=tuple(str(item) for item in list(payload.get("anchor_mentions") or [])),
        role=str(payload.get("role") or "support"),
        satisfiable_by=str(payload.get("satisfiable_by") or "unknown"),
    )


def load_frozen_requirement_cache(report_path: Path) -> dict[str, list[DTCRequirement]]:
    if not report_path.exists():
        return {}
    payload = read_json(report_path)
    cache: dict[str, list[DTCRequirement]] = {}
    for trace in list(payload.get("setwise_selector_query_traces") or []):
        if not isinstance(trace, Mapping):
            continue
        question = str(trace.get("question") or "").strip()
        selector_trace = dict(trace.get("selector_trace") or {})
        raw_requirements = list(selector_trace.get("requirements") or [])
        requirements = [
            requirement_from_trace(item)
            for item in raw_requirements
            if isinstance(item, Mapping) and str(item.get("subquery") or "").strip()
        ]
        if question and requirements:
            cache[question] = requirements
    return cache


def install_frozen_requirement_cache(cache: Mapping[str, Sequence[DTCRequirement]]) -> int:
    if not cache:
        return 0
    original = eval_causal_module.request_dtc_requirements_from_llm

    def cached_request(
        query: str,
        infer_fn: Any,
        model_name: str,
        max_steps: int = 4,
        max_completion_tokens: int = 512,
        include_satisfiable_by: bool = False,
    ):
        key = str(query or "").strip()
        cached = list(cache.get(key) or [])
        if cached:
            requirements = cached[: max(int(max_steps), 1)]
            return requirements, {
                "mode": "frozen_report",
                "llm_model": str(model_name or ""),
                "max_steps": int(max_steps),
                "include_satisfiable_by": bool(include_satisfiable_by),
                "llm_error": None,
                "fallback_used": False,
                "parse_succeeded": True,
                "active_step_count": int(len(requirements)),
                "raw_output_preview": "",
                "metadata": {"source": "setwise_selector_query_traces.requirements"},
            }
        return original(
            query=query,
            infer_fn=infer_fn,
            model_name=model_name,
            max_steps=max_steps,
            max_completion_tokens=max_completion_tokens,
            include_satisfiable_by=include_satisfiable_by,
        )

    eval_causal_module.request_dtc_requirements_from_llm = cached_request
    return len(cache)


def swap_rows(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selector_trace = dict(trace.get("selector_trace", {}) or {})
    return [row for row in list(selector_trace.get("selection_steps", []) or []) if isinstance(row, Mapping)]


def build_query_rows(
    *,
    dataset: str,
    baseline_solutions: Sequence[QuerySolution],
    selected_solutions: Sequence[QuerySolution],
    gold_docs: Sequence[Sequence[str]],
    qa_top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (baseline_qs, selected_qs) in enumerate(zip(baseline_solutions, selected_solutions)):
        gold_titles = [extract_doc_title(doc) for doc in gold_docs[idx]]
        baseline_titles = [extract_doc_title(doc) for doc in list(baseline_qs.docs)[:qa_top_k]]
        selector_titles = [extract_doc_title(doc) for doc in list(selected_qs.docs)[:qa_top_k]]
        pool_titles = [extract_doc_title(doc) for doc in list(baseline_qs.docs)[:100]]
        selector_trace = dict((selected_qs.retrieval_trace or {}).get("setwise_selector_trace", {}) or {})
        trace = {
            "question": selected_qs.question,
            "gold_doc_count": int(len(set(gold_docs[idx]))),
            "gold_titles": gold_titles,
            "baseline_top_titles": baseline_titles,
            "selector_top_titles": selector_titles,
            "changed_from_baseline": bool(baseline_titles != selector_titles),
            "selector_trace": selector_trace,
        }
        swaps = swap_rows(trace)
        gold_norm = title_counter(gold_titles)
        swapped_in_gold = 0
        swapped_out_gold = 0
        for swap in swaps:
            in_title = normalize_title(swap.get("in_title"))
            out_title = normalize_title(swap.get("out_title"))
            if gold_norm.get(in_title, 0) > 0:
                swapped_in_gold += 1
            if gold_norm.get(out_title, 0) > 0:
                swapped_out_gold += 1
        rows.append(
            {
                "dataset": dataset,
                "query_idx": idx,
                "question": selected_qs.question,
                "gold_doc_count": int(len(set(gold_docs[idx]))),
                "gold_titles": gold_titles,
                "baseline_top_titles": baseline_titles,
                "selector_top_titles": selector_titles,
                "pool_titles_top100": pool_titles,
                "changed": bool(baseline_titles != selector_titles),
                "swap_count": int(len(swaps)),
                "swapped_in_gold_title_count": int(swapped_in_gold),
                "swapped_out_gold_title_count": int(swapped_out_gold),
                "baseline_title_recall_top5": title_recall(gold_titles, baseline_titles, k=qa_top_k),
                "selector_title_recall_top5": title_recall(gold_titles, selector_titles, k=qa_top_k),
                "baseline_title_all_gold_top5": title_all_covered(gold_titles, baseline_titles, k=qa_top_k),
                "selector_title_all_gold_top5": title_all_covered(gold_titles, selector_titles, k=qa_top_k),
                "pool_title_all_gold_top100": title_all_covered(gold_titles, pool_titles, k=100),
                "trace": trace,
            }
        )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    count = len(rows)
    changed = [row for row in rows if bool(row["changed"])]
    rescue = [
        row
        for row in rows
        if (not bool(row["baseline_title_all_gold_top5"])) and bool(row["selector_title_all_gold_top5"])
    ]
    regression = [
        row
        for row in rows
        if bool(row["baseline_title_all_gold_top5"]) and (not bool(row["selector_title_all_gold_top5"]))
    ]
    recall_improved = [row for row in rows if row["selector_title_recall_top5"] > row["baseline_title_recall_top5"] + 1e-9]
    recall_worsened = [row for row in rows if row["selector_title_recall_top5"] < row["baseline_title_recall_top5"] - 1e-9]
    gold_out = [row for row in rows if int(row["swapped_out_gold_title_count"]) > 0]
    return {
        "count": int(count),
        "changed_count": int(len(changed)),
        "changed_rate": round(len(changed) / count, 4),
        "total_swaps": int(sum(int(row["swap_count"]) for row in rows)),
        "avg_swaps": round(mean(float(row["swap_count"]) for row in rows), 4),
        "baseline_title_recall_top5": round(mean(float(row["baseline_title_recall_top5"]) for row in rows), 4),
        "selector_title_recall_top5": round(mean(float(row["selector_title_recall_top5"]) for row in rows), 4),
        "title_recall_delta_top5": round(
            mean(float(row["selector_title_recall_top5"]) - float(row["baseline_title_recall_top5"]) for row in rows),
            4,
        ),
        "baseline_title_all_gold_top5": round(
            sum(bool(row["baseline_title_all_gold_top5"]) for row in rows) / count,
            4,
        ),
        "selector_title_all_gold_top5": round(
            sum(bool(row["selector_title_all_gold_top5"]) for row in rows) / count,
            4,
        ),
        "title_all_gold_delta_top5": round(
            (
                sum(bool(row["selector_title_all_gold_top5"]) for row in rows)
                - sum(bool(row["baseline_title_all_gold_top5"]) for row in rows)
            )
            / count,
            4,
        ),
        "pool_title_all_gold_top100": round(sum(bool(row["pool_title_all_gold_top100"]) for row in rows) / count, 4),
        "title_complete_rescue_count": int(len(rescue)),
        "title_complete_regression_count": int(len(regression)),
        "title_recall_improved_count": int(len(recall_improved)),
        "title_recall_worsened_count": int(len(recall_worsened)),
        "queries_swapped_in_gold": int(sum(int(row["swapped_in_gold_title_count"]) > 0 for row in rows)),
        "queries_swapped_out_gold": int(len(gold_out)),
    }


def summarize_by_depth(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{int(row['gold_doc_count'])}_doc", []).append(row)
    return {key: summarize_rows(grouped[key]) for key in sorted(grouped)}


def build_output_payload(
    *,
    dataset: str,
    variant: str,
    preserve_top_m: int,
    max_swaps: int,
    pool_path: Path,
    binding_cache_path: Path,
    selector_summary: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
    baseline_solutions: Sequence[QuerySolution],
    selected_solutions: Sequence[QuerySolution],
    external_pool_summary: Mapping[str, Any],
    qa_top_k: int,
    limit: int,
) -> dict[str, Any]:
    traces = [dict(row["trace"]) for row in query_rows]
    examples: list[dict[str, Any]] = []
    for baseline_qs, selected_qs in list(zip(baseline_solutions, selected_solutions))[:5]:
        examples.append(
            {
                "question": baseline_qs.question,
                "gold_answers": list(baseline_qs.gold_answers or []),
                "baseline_top_titles": [extract_doc_title(doc) for doc in list(baseline_qs.docs)[:qa_top_k]],
                "selector_top_titles": [extract_doc_title(doc) for doc in list(selected_qs.docs)[:qa_top_k]],
                "retrieval_trace": baseline_qs.retrieval_trace or {},
            }
        )
    return {
        "mode": "selector_only_frontier",
        "dataset": dataset,
        "limit": int(limit),
        "pool_json": str(pool_path),
        "binding_cache_path": str(binding_cache_path),
        "variant": variant,
        "preserve_top_m": int(preserve_top_m),
        "max_swaps": int(max_swaps),
        "qa_top_k": int(qa_top_k),
        "selector": "none" if max_swaps == 0 else "daec_noisyor_safe_llm",
        "external_pool": dict(external_pool_summary),
        "selector_summary": dict(selector_summary),
        "frontier_summary": {
            "overall": summarize_rows(query_rows),
            "by_depth": summarize_by_depth(query_rows),
        },
        "setwise_selector_query_traces": traces,
        "examples": examples,
    }


def run_dataset(args: argparse.Namespace, dataset: str, variants: Sequence[str]) -> None:
    pool_k = int(args.pool_k)
    limit = int(args.limit)
    qa_top_k = int(args.qa_top_k)
    pool_path = default_pool_path(dataset, pool_k, limit, Path(args.pool_root))
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing ETv3 pool JSON: {pool_path}")
    binding_cache_path = default_binding_cache_path(dataset, pool_k, Path(args.binding_root))
    requirement_report_path = default_requirement_report_path(
        dataset,
        pool_k,
        limit,
        Path(args.requirement_report_root),
    )
    frozen_requirement_count = install_frozen_requirement_cache(
        load_frozen_requirement_cache(requirement_report_path)
    )
    if frozen_requirement_count:
        print(
            f"[CACHE] dataset={dataset} frozen_requirements={frozen_requirement_count} source={requirement_report_path}",
            flush=True,
        )
    corpus, samples, docs, gold_docs, gold_answers = load_dataset(dataset, limit)
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    config = build_config(args, dataset, len(corpus))
    hipporag = HippoRAG(global_config=config)
    install_qwen_disable_thinking(getattr(hipporag, "llm_model", None), enabled=bool(args.qwen_disable_thinking))
    hipporag.index(docs)
    if not hasattr(hipporag, "passage_node_key_to_doc_idx") or not bool(getattr(hipporag, "ready_to_retrieve", False)):
        hipporag.prepare_retrieval_objects()
    baseline_solutions, external_pool_summary = load_external_pool_query_solutions(
        pool_path,
        dataset_name=dataset,
        samples=samples,
        corpus=corpus,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
        source_name="etv3_pool100",
        strict_questions=True,
    )

    for variant_value in variants:
        variant, preserve_top_m, max_swaps = parse_variant(variant_value)
        output_path = default_output_path(Path(args.output_root), dataset, pool_k, variant, limit)
        if output_path.exists() and not bool(args.overwrite):
            print(f"[SKIP] {dataset} {variant}: {output_path}", flush=True)
            continue
        print(f"[RUN] dataset={dataset} variant={variant} preserve={preserve_top_m} max_swaps={max_swaps}", flush=True)
        if max_swaps == 0:
            selected_solutions = make_noop_selected_solutions(baseline_solutions)
            selector_summary = {
                "selector": "none",
                "daec_safe_projection": True,
                "daec_safe_preserve_top_m": int(preserve_top_m),
                "daec_safe_max_swaps": int(max_swaps),
                "llm_binding_total_calls": 0,
                "llm_binding_total_cache_hits": 0,
            }
        else:
            selected_solutions, selector_summary = apply_setwise_selector(
                hipporag=hipporag,
                query_solutions=list(baseline_solutions),
                doc_text_to_chunk_id=doc_text_to_chunk_id,
                pool_k=pool_k,
                qa_top_k=qa_top_k,
                selector_name="daec_noisyor_safe_llm",
                score_mode="bridge",
                anchor_count=0,
                reserve_top_m=0,
                max_bridge_slots=0,
                structure_max_hops=2,
                base_weight=1.0,
                structure_weight=1.0,
                novelty_weight=0.1,
                dtc_decomposition_mode="llm",
                dtc_binding_max_candidates=5,
                daec_safe_min_objective_gain=float(args.daec_safe_min_objective_gain),
                daec_safe_min_swap_gain=float(args.daec_safe_min_swap_gain),
                daec_safe_max_swaps=max_swaps,
                daec_safe_preserve_top_m=preserve_top_m,
                llm_binding_url=str(args.llm_binding_url or config.llm_base_url),
                llm_binding_model=str(args.llm_binding_model or args.llm_request_name),
                llm_binding_cache_path=str(binding_cache_path),
                llm_binding_title_match_mode=str(args.llm_binding_title_match_mode),
            )
        query_rows = build_query_rows(
            dataset=dataset,
            baseline_solutions=baseline_solutions,
            selected_solutions=selected_solutions,
            gold_docs=gold_docs,
            qa_top_k=qa_top_k,
        )
        payload = build_output_payload(
            dataset=dataset,
            variant=variant,
            preserve_top_m=preserve_top_m,
            max_swaps=max_swaps,
            pool_path=pool_path,
            binding_cache_path=binding_cache_path,
            selector_summary=selector_summary,
            query_rows=query_rows,
            baseline_solutions=baseline_solutions,
            selected_solutions=selected_solutions,
            external_pool_summary=external_pool_summary,
            qa_top_k=qa_top_k,
            limit=limit,
        )
        payload["frozen_requirement_report_path"] = str(requirement_report_path)
        payload["frozen_requirement_query_count"] = int(frozen_requirement_count)
        write_json(payload, output_path)
        overall = payload["frontier_summary"]["overall"]
        print(
            "[DONE] {dataset} {variant}: title-all@5 {base:.4f}->{sel:.4f} "
            "delta={delta:+.4f} changed={changed} swaps={swaps}".format(
                dataset=dataset,
                variant=variant,
                base=float(overall["baseline_title_all_gold_top5"]),
                sel=float(overall["selector_title_all_gold_top5"]),
                delta=float(overall["title_all_gold_delta_top5"]),
                changed=int(overall["changed_count"]),
                swaps=int(overall["total_swaps"]),
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--qa-doc-max-chars", type=int, default=2048)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--binding-root", type=Path, default=DEFAULT_BINDING_ROOT)
    parser.add_argument("--requirement-report-root", type=Path, default=DEFAULT_REQUIREMENT_REPORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--llm-request-name", default="qwen3-8b-train")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-binding-url", default="")
    parser.add_argument("--llm-binding-model", default="")
    parser.add_argument("--max-retry-attempts", type=int, default=20)
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--embedding-name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--daec-safe-min-objective-gain", type=float, default=0.0)
    parser.add_argument("--daec-safe-min-swap-gain", type=float, default=1e-6)
    parser.add_argument("--llm-binding-title-match-mode", default="wiki_title")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = parse_csv(args.datasets)
    variants = parse_csv(args.variants)
    for dataset in datasets:
        run_dataset(args, dataset, variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
