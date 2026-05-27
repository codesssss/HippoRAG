#!/usr/bin/env python3
"""Run ETv4-composition A0 selector-only dominance probes.

This runner keeps ETv3 frozen and reuses the existing DBEC decomposition /
binding substrate.  It does not call the reader.  Binding extraction is
cache-only by default, so the A0 probe tests the composition rule rather than a
new LLM extraction pass.
"""

from __future__ import annotations

import argparse
import json
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

import run_etv3_dbec_selector_frontier as base  # noqa: E402
from dtc_embed_utils import select_daec_noisyor_positions  # noqa: E402
from eval_causal_qwen3 import (  # noqa: E402
    build_dtc_requirement_embeddings,
    build_dtc_text_embeddings,
    extract_doc_title,
)
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402
from src.hipporag.utils.misc_utils import QuerySolution, compute_mdhash_id  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("run_logs/etv4_composition_a0_selector_full1000_20260510")
DEFAULT_VARIANTS = ("r2_ge", "r2_gt", "r3_ge", "r3_gt")


def parse_variants(value: str | Sequence[str]) -> list[str]:
    items = base.parse_csv(value)
    allowed = set(DEFAULT_VARIANTS)
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown A0 variant(s): {unknown}. Allowed: {sorted(allowed)}")
    return items


def projection_mode_for_variant(variant: str) -> str:
    return f"agreement_{variant}"


def default_output_path(output_root: Path, dataset: str, pool_k: int, variant: str, limit: int) -> Path:
    return output_root / "evals" / f"{dataset}_etv3_pool{pool_k}_etv4_a0_{variant}_selector_limit{limit}.json"


def load_binding_cache(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): [str(item) for item in value]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def make_cache_only_binding_extractor(
    *,
    cache: Mapping[str, Sequence[str]],
    model: str,
    stats: dict[str, int],
):
    think_re = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    def extract(subquery: str, doc_text: str) -> list[str]:
        prompt_payload = {
            "version": "daec_llm_binding_v1",
            "model": str(model),
            "subquery": str(subquery),
            "passage": str(doc_text[:1500]),
        }
        cache_key = compute_mdhash_id(json.dumps(prompt_payload, sort_keys=True, ensure_ascii=False))
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        if cache_key not in cache:
            stats["misses"] = int(stats.get("misses", 0)) + 1
            return []
        stats["hits"] = int(stats.get("hits", 0)) + 1
        # Older caches should already be clean; strip think blocks defensively.
        return [think_re.sub("", str(item)).strip() for item in cache[cache_key] if str(item).strip()]

    return extract


def pool_doc_ids_for_solution(
    *,
    hipporag: HippoRAG,
    doc_text_to_chunk_id: Mapping[str, str],
    pool_docs: Sequence[str],
) -> list[int | None]:
    chunk_text_to_hash = getattr(hipporag, "chunk_text_to_hash", {}) or {}
    pool_doc_ids: list[int | None] = []
    for doc_text in pool_docs:
        chunk_id = doc_text_to_chunk_id.get(doc_text) or chunk_text_to_hash.get(doc_text)
        mapped_doc_id = hipporag.passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
        pool_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)
    return pool_doc_ids


def materialize_selected_solution(
    *,
    source: QuerySolution,
    pool_docs: Sequence[str],
    pool_scores: np.ndarray,
    selected_positions: Sequence[int],
    selector_trace: Mapping[str, Any],
    pool_doc_ids: Sequence[int | None],
    qa_top_k: int,
    pool_limit: int,
) -> QuerySolution:
    front_positions: list[int] = []
    seen_front: set[int] = set()
    for raw_pos in selected_positions:
        pos = int(raw_pos)
        if pos < 0 or pos >= pool_limit or pos in seen_front:
            continue
        front_positions.append(pos)
        seen_front.add(pos)
        if len(front_positions) >= qa_top_k:
            break
    for pos in range(pool_limit):
        if len(front_positions) >= qa_top_k:
            break
        if pos in seen_front:
            continue
        front_positions.append(pos)
        seen_front.add(pos)

    selected_set = set(front_positions)
    reordered_pool_positions = front_positions + [pos for pos in range(pool_limit) if pos not in selected_set]
    tail_docs = list(source.docs[pool_limit:])
    if source.doc_scores is not None and len(source.doc_scores) > pool_limit:
        tail_scores = np.asarray(source.doc_scores[pool_limit:], dtype=float)
    else:
        tail_scores = np.asarray([], dtype=float)
    reordered_docs = [pool_docs[pos] for pos in reordered_pool_positions] + tail_docs
    reordered_scores = np.concatenate(
        [
            np.asarray([float(pool_scores[pos]) for pos in reordered_pool_positions], dtype=float),
            tail_scores,
        ]
    )

    retrieval_trace = dict(source.retrieval_trace or {})
    retrieval_trace["setwise_selector_trace"] = {
        "pool_k": int(pool_limit),
        "selector": "etv4_composition_a0_agreement_dominance",
        "selected_pool_positions": [int(pos) for pos in selected_positions],
        "selected_doc_ids": [
            int(pool_doc_ids[pos]) if pos < len(pool_doc_ids) and pool_doc_ids[pos] is not None else None
            for pos in selected_positions
        ],
        "selected_titles": [extract_doc_title(pool_docs[pos]) for pos in selected_positions if 0 <= int(pos) < pool_limit],
        "final_front_pool_positions": list(front_positions),
        "final_front_doc_ids": [
            int(pool_doc_ids[pos]) if pos < len(pool_doc_ids) and pool_doc_ids[pos] is not None else None
            for pos in front_positions
        ],
        "final_front_titles": [extract_doc_title(pool_docs[pos]) for pos in front_positions],
        **dict(selector_trace),
    }
    return QuerySolution(
        question=source.question,
        docs=reordered_docs,
        doc_scores=reordered_scores,
        answer=None,
        gold_answers=list(source.gold_answers or []),
        gold_docs=list(source.gold_docs or []),
        retrieval_trace=retrieval_trace,
        qa_trace=source.qa_trace,
    )


def summarize_decisions(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = Counter()
    statuses = Counter()
    swap_modes = Counter()
    for trace in traces:
        selector_trace = dict(trace.get("selector_trace", {}) or {})
        statuses[str(selector_trace.get("status", "unknown"))] += 1
        safe_trace = dict(selector_trace.get("safe_projection_trace", {}) or {})
        decisions[str(safe_trace.get("safe_decision", "unknown"))] += 1
        for step in list(safe_trace.get("safe_swap_steps", []) or []):
            if isinstance(step, Mapping):
                swap_modes[str(step.get("mode", "unknown"))] += 1
    return {
        "status_counts": dict(sorted(statuses.items())),
        "safe_decision_counts": dict(sorted(decisions.items())),
        "swap_mode_counts": dict(sorted(swap_modes.items())),
    }


def run_variant(
    *,
    args: argparse.Namespace,
    dataset: str,
    variant: str,
    hipporag: HippoRAG,
    doc_text_to_chunk_id: Mapping[str, str],
    baseline_solutions: Sequence[QuerySolution],
    gold_docs: Sequence[Sequence[str]],
    pool_path: Path,
    binding_cache_path: Path,
    external_pool_summary: Mapping[str, Any],
    requirement_cache: Mapping[str, Sequence[Any]],
    binding_cache: Mapping[str, Sequence[str]],
) -> None:
    pool_k = int(args.pool_k)
    limit = int(args.limit)
    qa_top_k = int(args.qa_top_k)
    output_path = default_output_path(Path(args.output_root), dataset, pool_k, variant, limit)
    if output_path.exists() and not bool(args.overwrite):
        print(f"[SKIP] {dataset} {variant}: {output_path}", flush=True)
        return

    print(f"[RUN] dataset={dataset} variant={variant}", flush=True)
    binding_stats = {"attempts": 0, "hits": 0, "misses": 0}
    llm_extract_fn = make_cache_only_binding_extractor(
        cache=binding_cache,
        model=str(args.llm_binding_model or args.llm_request_name),
        stats=binding_stats,
    )
    selected_solutions: list[QuerySolution] = []
    missing_requirement_count = 0

    for q_idx, qs in enumerate(baseline_solutions):
        pool_limit = min(len(qs.docs), max(pool_k, qa_top_k))
        pool_docs = list(qs.docs[:pool_limit])
        pool_titles = [extract_doc_title(doc_text) for doc_text in pool_docs]
        if qs.doc_scores is not None and len(qs.doc_scores) >= pool_limit:
            pool_scores = np.asarray(qs.doc_scores[:pool_limit], dtype=float)
        else:
            pool_scores = np.linspace(pool_limit, 1, pool_limit, dtype=float)
        pool_doc_ids = pool_doc_ids_for_solution(
            hipporag=hipporag,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
            pool_docs=pool_docs,
        )

        requirements = list(requirement_cache.get(str(qs.question or "").strip()) or [])
        if not requirements:
            missing_requirement_count += 1
        requirement_embeddings = build_dtc_requirement_embeddings(
            hipporag=hipporag,
            requirements=requirements,
        )

        def embed_bound_texts(texts: Sequence[str]) -> dict[str, np.ndarray]:
            return build_dtc_text_embeddings(hipporag=hipporag, texts=texts)

        selected_positions, selector_trace = select_daec_noisyor_positions(
            query=qs.question,
            requirements=requirements,
            requirement_embeddings=requirement_embeddings,
            pool_docs=pool_docs,
            pool_doc_ids=pool_doc_ids,
            pool_doc_scores=pool_scores,
            pool_doc_titles=pool_titles,
            doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            passage_embeddings=np.asarray(getattr(hipporag, "passage_embeddings", np.array([]))),
            qa_top_k=qa_top_k,
            binding_top_m=int(args.dtc_binding_max_candidates),
            embed_texts_fn=embed_bound_texts,
            safe_projection=True,
            safe_min_objective_gain=0.0,
            safe_min_swap_gain=float(args.min_swap_gain),
            safe_max_swaps=qa_top_k,
            safe_preserve_top_m=0,
            safe_projection_mode=projection_mode_for_variant(variant),
            safe_agreement_top_k=int(args.agreement_top_k),
            safe_retriever_margin_threshold=1.01,
            safe_retriever_rank_penalty=0.0,
            llm_extract_fn=llm_extract_fn,
            binding_mode="llm",
            llm_binding_title_match_mode=str(args.llm_binding_title_match_mode),
            agsto_metadata=dict((qs.retrieval_trace or {}).get("external_pool_agsto", {}) or {}),
            selector_label_override="etv4_composition_a0_agreement_dominance",
        )
        selector_trace["a0_variant"] = str(variant)
        selector_trace["a0_projection_mode"] = projection_mode_for_variant(variant)
        selector_trace["a0_query_idx"] = int(q_idx)
        selected_solutions.append(
            materialize_selected_solution(
                source=qs,
                pool_docs=pool_docs,
                pool_scores=pool_scores,
                selected_positions=selected_positions,
                selector_trace=selector_trace,
                pool_doc_ids=pool_doc_ids,
                qa_top_k=qa_top_k,
                pool_limit=pool_limit,
            )
        )

    query_rows = base.build_query_rows(
        dataset=dataset,
        baseline_solutions=baseline_solutions,
        selected_solutions=selected_solutions,
        gold_docs=gold_docs,
        qa_top_k=qa_top_k,
    )
    traces = [dict(row["trace"]) for row in query_rows]
    payload = {
        "mode": "selector_only_etv4_composition_a0",
        "dataset": dataset,
        "limit": int(limit),
        "pool_json": str(pool_path),
        "binding_cache_path": str(binding_cache_path),
        "variant": str(variant),
        "projection_mode": projection_mode_for_variant(variant),
        "retention_signals": ["etv3_top5", "dense_top5"] + (["dbec_top5"] if variant.startswith("r3_") else []),
        "retention_rule": "gt" if variant.endswith("_gt") else "ge",
        "agreement_top_k": int(args.agreement_top_k),
        "min_swap_gain": float(args.min_swap_gain),
        "qa_top_k": int(qa_top_k),
        "selector": "etv4_composition_a0_agreement_dominance",
        "external_pool": dict(external_pool_summary),
        "requirement_cache": {
            "query_count": int(len(requirement_cache)),
            "missing_query_count": int(missing_requirement_count),
        },
        "binding_cache": {
            "path": str(binding_cache_path),
            "entry_count": int(len(binding_cache)),
            **binding_stats,
        },
        "selector_summary": summarize_decisions(traces),
        "frontier_summary": {
            "overall": base.summarize_rows(query_rows),
            "by_depth": base.summarize_by_depth(query_rows),
        },
        "setwise_selector_query_traces": traces,
        "examples": [
            {
                "question": baseline_qs.question,
                "gold_answers": list(baseline_qs.gold_answers or []),
                "baseline_top_titles": [extract_doc_title(doc) for doc in list(baseline_qs.docs)[:qa_top_k]],
                "selector_top_titles": [extract_doc_title(doc) for doc in list(selected_qs.docs)[:qa_top_k]],
                "selector_trace": dict((selected_qs.retrieval_trace or {}).get("setwise_selector_trace", {}) or {}),
            }
            for baseline_qs, selected_qs in list(zip(baseline_solutions, selected_solutions))[:5]
        ],
    }
    base.write_json(payload, output_path)
    overall = payload["frontier_summary"]["overall"]
    print(
        "[DONE] {dataset} {variant}: title-all@5 {base_all:.4f}->{sel_all:.4f} "
        "delta={delta:+.4f} changed={changed} swaps={swaps} gold_out={gold_out} cache_miss={misses}".format(
            dataset=dataset,
            variant=variant,
            base_all=float(overall["baseline_title_all_gold_top5"]),
            sel_all=float(overall["selector_title_all_gold_top5"]),
            delta=float(overall["title_all_gold_delta_top5"]),
            changed=int(overall["changed_count"]),
            swaps=int(overall["total_swaps"]),
            gold_out=int(overall["queries_swapped_out_gold"]),
            misses=int(binding_stats["misses"]),
        ),
        flush=True,
    )


def run_dataset(args: argparse.Namespace, dataset: str, variants: Sequence[str]) -> None:
    pool_k = int(args.pool_k)
    limit = int(args.limit)
    pool_path = base.default_pool_path(dataset, pool_k, limit, Path(args.pool_root))
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing ETv3 pool JSON: {pool_path}")
    binding_cache_path = base.default_binding_cache_path(dataset, pool_k, Path(args.binding_root))
    requirement_report_path = base.default_requirement_report_path(
        dataset,
        pool_k,
        limit,
        Path(args.requirement_report_root),
    )
    requirement_cache = base.load_frozen_requirement_cache(requirement_report_path)
    if not requirement_cache:
        raise RuntimeError(f"Missing frozen requirements in {requirement_report_path}")
    binding_cache = load_binding_cache(binding_cache_path)
    if not binding_cache:
        raise RuntimeError(f"Missing binding cache entries in {binding_cache_path}")

    corpus, samples, docs, gold_docs, gold_answers = base.load_dataset(dataset, limit)
    doc_text_to_chunk_id = base.build_doc_text_to_chunk_id(corpus)
    config = base.build_config(args, dataset, len(corpus))
    hipporag = HippoRAG(global_config=config)
    base.install_qwen_disable_thinking(getattr(hipporag, "llm_model", None), enabled=bool(args.qwen_disable_thinking))
    hipporag.index(docs)
    if not hasattr(hipporag, "passage_node_key_to_doc_idx") or not bool(getattr(hipporag, "ready_to_retrieve", False)):
        hipporag.prepare_retrieval_objects()
    baseline_solutions, external_pool_summary = base.load_external_pool_query_solutions(
        pool_path,
        dataset_name=dataset,
        samples=samples,
        corpus=corpus,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
        source_name="etv3_pool100",
        strict_questions=True,
    )
    max_queries = int(getattr(args, "max_queries", 0) or 0)
    if max_queries > 0:
        baseline_solutions = list(baseline_solutions)[:max_queries]
        gold_docs = list(gold_docs)[:max_queries]
    print(
        f"[DATASET] {dataset}: requirements={len(requirement_cache)} binding_cache={len(binding_cache)} queries={len(baseline_solutions)}",
        flush=True,
    )
    for variant in variants:
        run_variant(
            args=args,
            dataset=dataset,
            variant=variant,
            hipporag=hipporag,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
            baseline_solutions=baseline_solutions,
            gold_docs=gold_docs,
            pool_path=pool_path,
            binding_cache_path=binding_cache_path,
            external_pool_summary=external_pool_summary,
            requirement_cache=requirement_cache,
            binding_cache=binding_cache,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(base.DEFAULT_DATASETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0, help="Debug-only cap after loading the canonical pool file.")
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--qa-doc-max-chars", type=int, default=2048)
    parser.add_argument("--pool-root", type=Path, default=base.DEFAULT_POOL_ROOT)
    parser.add_argument("--binding-root", type=Path, default=base.DEFAULT_BINDING_ROOT)
    parser.add_argument("--requirement-report-root", type=Path, default=base.DEFAULT_REQUIREMENT_REPORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--save-dir", default=base.DEFAULT_SAVE_DIR)
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--llm-request-name", default="qwen3-8b-train")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-binding-model", default="")
    parser.add_argument("--max-retry-attempts", type=int, default=20)
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--embedding-name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--dtc-binding-max-candidates", type=int, default=5)
    parser.add_argument("--llm-binding-title-match-mode", default="wiki_title")
    parser.add_argument("--agreement-top-k", type=int, default=5)
    parser.add_argument("--min-swap-gain", type=float, default=1e-9)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = base.parse_csv(args.datasets)
    variants = parse_variants(args.variants)
    for dataset in datasets:
        run_dataset(args, dataset, variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
