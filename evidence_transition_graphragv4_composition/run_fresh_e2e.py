#!/usr/bin/env python3
"""Run PCEC with fresh frozen-ETv3 expansion and native PrefixResidualReadout.

This Phase-4 runner stops at retrieval/top-5 output. It does not run reader QA.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence_transition_graphragv4_composition.contract import (  # noqa: E402
    DEFAULT_DTC_BINDING_MAX_CANDIDATES,
    DEFAULT_POOL_K,
    DEFAULT_PREFIX_BUDGET_M,
    DEFAULT_READER_BUDGET_K,
    DEFAULT_SAFE_MIN_OBJECTIVE_GAIN,
    DEFAULT_SAFE_MIN_SWAP_GAIN,
    METHOD_CONTRACT,
    METHOD_NAME,
    PAPER_FACING_METHOD_NAME,
)
from evidence_transition_graphragv4_composition.dbec_utility import (  # noqa: E402
    NativeDBECUtilityProvider,
    load_binding_cache,
)
from evidence_transition_graphragv4_composition.expander import (  # noqa: E402
    DEFAULT_FROZEN_ETV3_RUNS_ROOT,
    FrozenETv3Expander,
)
from evidence_transition_graphragv4_composition.native_readout import compose_pcec_readout  # noqa: E402
from evidence_transition_graphragv4_composition.pcec_types import PCECQueryState  # noqa: E402
from evidence_transition_graphragv4_composition.pool_alignment import (  # noqa: E402
    align_pool_record,
    build_doc_text_to_chunk_id,
)
from evidence_transition_graphragv4_composition.readout import write_json  # noqa: E402
from evidence_transition_graphragv4_composition.requirements import FrozenReportRequirementProvider  # noqa: E402
from evidence_transition_graphragv4_composition.run_native_pool import (  # noqa: E402
    DATASET_PORTS,
    DEFAULT_BINDING_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_REQUIREMENT_ROOT,
    build_config,
    build_query_trace,
    build_report_row,
    default_binding_cache,
    default_requirement_report,
    load_corpus_docs,
    summarize_rows,
)
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("run_logs/pcec_fresh_e2e")


def default_output_json(output_root: Path, dataset: str, prefix_budget_m: int, reader_budget_k: int, pool_k: int, limit: int) -> Path:
    residual = int(reader_budget_k) - int(prefix_budget_m)
    return output_root / "evals" / f"{dataset}_pcec_fresh_e2e_prefix{prefix_budget_m}_residual{residual}_pool{pool_k}_limit{limit}.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--artifact-limit", type=int, default=1000)
    parser.add_argument("--candidate-pool-k", type=int, default=DEFAULT_POOL_K)
    parser.add_argument("--et-candidate-pool-k", type=int, default=200)
    parser.add_argument("--reader-budget-k", type=int, default=DEFAULT_READER_BUDGET_K)
    parser.add_argument("--prefix-budget-m", type=int, default=DEFAULT_PREFIX_BUDGET_M)
    parser.add_argument("--requirement-source", choices=["frozen_report"], default="frozen_report")
    parser.add_argument("--requirement-report-path", type=Path, default=None)
    parser.add_argument("--binding-cache-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-runs-root", type=Path, default=DEFAULT_FROZEN_ETV3_RUNS_ROOT)
    parser.add_argument("--save-dir", default="outputs_step0_general_nvembed")
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--llm-request-name", default="qwen3-8b-train")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-binding-model", default="")
    parser.add_argument("--max-retry-attempts", type=int, default=20)
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--embedding-name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--expander-embedding-name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--qa-doc-max-chars", type=int, default=2048)
    parser.add_argument("--dtc-binding-max-candidates", type=int, default=DEFAULT_DTC_BINDING_MAX_CANDIDATES)
    parser.add_argument("--llm-binding-title-match-mode", default="wiki_title")
    parser.add_argument("--daec-safe-min-objective-gain", type=float, default=DEFAULT_SAFE_MIN_OBJECTIVE_GAIN)
    parser.add_argument("--daec-safe-min-swap-gain", type=float, default=DEFAULT_SAFE_MIN_SWAP_GAIN)
    parser.add_argument("--allow-missing-requirements", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def resolve_default_paths(args: argparse.Namespace) -> None:
    limit = int(args.max_queries)
    artifact_limit = int(args.artifact_limit)
    if args.requirement_report_path is None:
        args.requirement_report_path = default_requirement_report(str(args.dataset), int(args.candidate_pool_k), artifact_limit)
    if args.binding_cache_path is None:
        args.binding_cache_path = default_binding_cache(str(args.dataset), int(args.candidate_pool_k))
    if args.output_json is None:
        args.output_json = default_output_json(
            Path(args.output_root),
            str(args.dataset),
            int(args.prefix_budget_m),
            int(args.reader_budget_k),
            int(args.candidate_pool_k),
            limit,
        )


def run_fresh_e2e(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if bool(args.qwen_disable_thinking):
        os.environ["HIPPORAG_RERANK_FORCE_NO_THINK"] = "1"
    dataset = str(args.dataset)
    max_queries = int(args.max_queries)
    corpus, docs = load_corpus_docs(dataset, Path(args.data_root))
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    config = build_config(args, dataset, len(docs))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    if not hasattr(hipporag, "passage_node_key_to_doc_idx") or not bool(getattr(hipporag, "ready_to_retrieve", False)):
        hipporag.prepare_retrieval_objects()

    expander = FrozenETv3Expander(
        dataset=dataset,
        candidate_pool_k=int(args.candidate_pool_k),
        reader_budget_k=int(args.reader_budget_k),
        et_candidate_pool_k=int(args.et_candidate_pool_k),
        data_root=Path(args.data_root),
        run_root=Path(args.frozen_runs_root) / dataset,
        llm_name=str(args.llm_request_name),
        embedding_name=str(args.expander_embedding_name),
        embedding_base_url=str(args.embedding_base_url),
    )
    limit = len(expander.samples) if max_queries <= 0 else min(max_queries, len(expander.samples))
    print(
        f"[READY] dataset={dataset} limit={limit} mode=pcec_fresh_e2e runner={expander.runner}",
        flush=True,
    )
    expander.prepare_queries(max_queries=limit)

    requirement_provider = FrozenReportRequirementProvider(
        args.requirement_report_path,
        allow_missing=bool(args.allow_missing_requirements),
    )
    utility_provider = NativeDBECUtilityProvider(
        hipporag=hipporag,
        binding_cache=load_binding_cache(args.binding_cache_path),
        binding_model=str(args.llm_binding_model or args.llm_request_name),
        binding_top_m=int(args.dtc_binding_max_candidates),
        min_objective_gain=float(args.daec_safe_min_objective_gain),
        min_swap_gain=float(args.daec_safe_min_swap_gain),
        llm_binding_title_match_mode=str(args.llm_binding_title_match_mode),
    )

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    missing_requirement_count = 0
    progress_every = max(int(args.progress_every), 0)
    for query_idx in range(limit):
        if progress_every and (query_idx == 0 or query_idx % progress_every == 0):
            print(
                f"[PROGRESS] dataset={dataset} query={query_idx}/{limit} elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
        record = expander.expand_query_record(query_idx)
        aligned_record = align_pool_record(
            record,
            corpus=corpus,
            hipporag=hipporag,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
        )
        state = PCECQueryState.from_pool_record(
            aligned_record,
            dataset=dataset,
            reader_budget_k=int(args.reader_budget_k),
            prefix_budget_m=int(args.prefix_budget_m),
        )
        requirements = requirement_provider.get_requirements(state)
        if not requirements:
            missing_requirement_count += 1
        result = compose_pcec_readout(
            state,
            requirements=requirements,
            utility_provider=utility_provider,
        )
        source_row: Mapping[str, Any] = {"gold_doc_indices": list(expander.rows[query_idx].get("gold_doc_indices") or [])}
        rows.append(
            build_report_row(
                state=state,
                result=result,
                source_row=source_row,
                requirement_count=len(requirements),
            )
        )
        traces.append(build_query_trace(state=state, result=result))
    if progress_every:
        print(
            f"[PROGRESS] dataset={dataset} query={limit}/{limit} elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )

    payload = {
        "method": METHOD_NAME,
        "paper_facing_method": PAPER_FACING_METHOD_NAME,
        "mode": "pcec_fresh_frozen_etv3_e2e",
        "dataset": dataset,
        "limit": int(limit),
        "pool_json": "",
        "requirement_report": str(args.requirement_report_path),
        "binding_cache_path": str(args.binding_cache_path),
        "retrieval_report": "",
        "openie_path": str(expander.metadata().get("openie_path") or ""),
        "input_report": str(expander.minimal_report_path),
        "fresh_expander": expander.metadata(),
        "pcec_contract": dict(METHOD_CONTRACT),
        "pcec_config": {
            "reader_budget_k": int(args.reader_budget_k),
            "prefix_budget_m": int(args.prefix_budget_m),
            "pool_k": int(args.candidate_pool_k),
            "et_candidate_pool_k": int(args.et_candidate_pool_k),
            "daec_safe_min_objective_gain": float(args.daec_safe_min_objective_gain),
            "daec_safe_min_swap_gain": float(args.daec_safe_min_swap_gain),
        },
        "requirement_provider": {
            **requirement_provider.summary(),
            "missing_requirement_count": int(missing_requirement_count),
        },
        "utility_provider": utility_provider.summary(),
        "summary": summarize_rows(rows, traces),
        "rows": rows,
        "setwise_selector_query_traces": traces,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolve_default_paths(args)
    payload = run_fresh_e2e(args)
    write_json(payload, args.output_json)
    summary = payload.get("summary", {})
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "dataset": payload.get("dataset"),
                "count": summary.get("count"),
                "changed": summary.get("changed_count"),
                "title_all": summary.get("pcec_title_all_gold_top5"),
                "binding_misses": payload.get("utility_provider", {}).get("misses"),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
