#!/usr/bin/env python3
"""Run native PCEC readout over an existing ET candidate pool.

This runner is the parity stage for PCEC native readout.  It consumes the same
ETv3 external-pool JSON and frozen DBEC requirement report used by historical
top4/max1 experiments, but it does not call ``scripts/eval_causal_qwen3.py`` as
a subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidenceflow.contract import (  # noqa: E402
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
from evidenceflow.dbec_utility import (  # noqa: E402
    NativeDBECUtilityProvider,
    load_binding_cache,
)
from evidenceflow.native_readout import compose_pcec_readout  # noqa: E402
from evidenceflow.pool_alignment import (  # noqa: E402
    align_pool_record,
    build_doc_text_to_chunk_id,
)
from evidenceflow.readout import (  # noqa: E402
    title_all_covered,
    title_recall,
    write_json,
)
from evidenceflow.requirements import FrozenReportRequirementProvider  # noqa: E402
from evidenceflow.pcec_types import PCECQueryState  # noqa: E402
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402
from src.hipporag.utils.config_utils import BaseConfig  # noqa: E402


DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DEFAULT_OUTPUT_ROOT = Path("run_logs/evidenceflow_native_pool")
DEFAULT_SAVE_DIR = "outputs_step0_general_nvembed"
DEFAULT_POOL_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/pools")
DEFAULT_REQUIREMENT_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/evals")
DEFAULT_BINDING_ROOT = Path("run_logs/etv3_dbec_latest_full1000_20260510/evals")
DATASET_PORTS = {
    "musique": 8041,
    "hotpotqa": 8042,
    "2wikimultihopqa": 8043,
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_title(doc_text: str) -> str:
    return str(doc_text or "").split("\n", 1)[0].strip()


def default_pool_json(dataset: str, pool_k: int, limit: int) -> Path:
    return DEFAULT_POOL_ROOT / f"{dataset}_etv3_pool{pool_k}_limit{limit}.json"


def default_requirement_report(dataset: str, pool_k: int, limit: int) -> Path:
    return DEFAULT_REQUIREMENT_ROOT / f"{dataset}_etv3_pool{pool_k}_dbec_stable_limit{limit}.json"


def default_binding_cache(dataset: str, pool_k: int) -> Path:
    return DEFAULT_BINDING_ROOT / f"{dataset}_etv3_pool{pool_k}_dbec_latest.binding_cache.json"


def default_output_json(output_root: Path, dataset: str, prefix_budget_m: int, reader_budget_k: int, pool_k: int, limit: int) -> Path:
    residual = int(reader_budget_k) - int(prefix_budget_m)
    return output_root / "evals" / f"{dataset}_pcec_native_pool_prefix{prefix_budget_m}_residual{residual}_pool{pool_k}_limit{limit}.json"


def dataset_llm_base_url(dataset: str) -> str:
    return f"http://localhost:{DATASET_PORTS.get(str(dataset), 8043)}/v1"


def load_corpus_docs(dataset: str, data_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    corpus_path = Path(data_root) / f"{dataset}_corpus.json"
    corpus = read_json(corpus_path)
    docs = [f"{row['title']}\n{row['text']}" for row in corpus]
    return list(corpus), docs


def build_config(args: argparse.Namespace, dataset: str, corpus_len: int) -> BaseConfig:
    return BaseConfig(
        save_dir=str(args.save_dir),
        llm_name=str(args.llm_name),
        llm_request_name=str(args.llm_request_name),
        llm_base_url=str(args.llm_base_url or dataset_llm_base_url(dataset)),
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
        qa_top_k=int(args.reader_budget_k),
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


def source_retrieval_payload(pool_payload: Mapping[str, Any]) -> dict[str, Any]:
    retrieval_report = str(pool_payload.get("retrieval_report") or "").strip()
    if not retrieval_report:
        return {}
    path = Path(retrieval_report)
    if not path.exists():
        return {}
    return dict(read_json(path))


def row_by_query_index(source_payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = {}
    for row in list(source_payload.get("rows") or []):
        if not isinstance(row, Mapping):
            continue
        try:
            query_idx = int(row.get("query_index", row.get("query_idx")))
        except (TypeError, ValueError):
            continue
        rows[query_idx] = row
    return rows


def build_query_trace(
    *,
    state: PCECQueryState,
    result: Any,
) -> dict[str, Any]:
    baseline_titles = list(state.pool_titles[: int(state.reader_budget_k)])
    final_titles = list(result.final_titles)
    return {
        "query_index": int(state.query_idx),
        "question": state.question,
        "gold_answers": list(state.gold_answers),
        "gold_titles": list(state.gold_titles),
        "baseline_top_titles": baseline_titles,
        "selector_top_titles": final_titles,
        "changed_from_baseline": bool(baseline_titles != final_titles),
        "selector_trace": dict(result.legacy_selector_trace),
        "pcec_readout": dict(result.trace),
    }


def build_report_row(
    *,
    state: PCECQueryState,
    result: Any,
    source_row: Mapping[str, Any],
    requirement_count: int,
) -> dict[str, Any]:
    reader_doc_ids_source = list((state.et_trace or {}).get("external_pool_doc_ids") or [])
    if len(reader_doc_ids_source) < len(state.pool_doc_ids):
        reader_doc_ids_source = list(state.pool_doc_ids)
    final_doc_ids = [
        int(reader_doc_ids_source[pos])
        for pos in result.final_positions
        if pos < len(reader_doc_ids_source) and reader_doc_ids_source[pos] is not None
    ]
    baseline_titles = list(state.pool_titles[: int(state.reader_budget_k)])
    final_titles = list(result.final_titles)
    return {
        "query_index": int(state.query_idx),
        "question": state.question,
        "gold_answers": list(state.gold_answers),
        "gold_docs": list(state.gold_docs),
        "gold_titles": list(state.gold_titles),
        "gold_doc_indices": list(source_row.get("gold_doc_indices") or []),
        "retrieved_doc_indices_top5": final_doc_ids,
        "retrieved_titles_top5": final_titles,
        "baseline_titles_top5": baseline_titles,
        "pcec_title_recall_at5": title_recall(list(state.gold_titles), final_titles, k=int(state.reader_budget_k)),
        "pcec_all_gold_at5": title_all_covered(list(state.gold_titles), final_titles, k=int(state.reader_budget_k)),
        "pcec_requirement_count": int(requirement_count),
        "pcec_readout": dict(result.trace),
        "legacy_selector_trace": dict(result.legacy_selector_trace),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]], traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    count = len(rows)
    return {
        "count": int(count),
        "changed_count": int(sum(bool(trace.get("changed_from_baseline")) for trace in traces)),
        "pcec_title_recall_top5": round(mean(float(row.get("pcec_title_recall_at5", 0.0)) for row in rows), 4),
        "pcec_title_all_gold_top5": round(sum(bool(row.get("pcec_all_gold_at5")) for row in rows) / count, 4),
        "prefix_preserved_count": int(
            sum(bool((row.get("pcec_readout") or {}).get("prefix_preserved")) for row in rows)
        ),
        "admit_count": int(
            sum(str((row.get("pcec_readout") or {}).get("decision")) == "admit" for row in rows)
        ),
    }


def run_native_pool(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.qwen_disable_thinking):
        os.environ["HIPPORAG_RERANK_FORCE_NO_THINK"] = "1"
    pool_payload = read_json(args.pool_json)
    records = list(pool_payload.get("records") or [])
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]

    source_payload = source_retrieval_payload(pool_payload)
    source_rows = row_by_query_index(source_payload)
    corpus, docs = load_corpus_docs(str(args.dataset), Path(args.data_root))
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    config = build_config(args, str(args.dataset), len(docs))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    if not hasattr(hipporag, "passage_node_key_to_doc_idx") or not bool(getattr(hipporag, "ready_to_retrieve", False)):
        hipporag.prepare_retrieval_objects()

    requirement_provider = FrozenReportRequirementProvider(
        args.requirement_report,
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
    for record in records:
        aligned_record = align_pool_record(
            record,
            corpus=corpus,
            hipporag=hipporag,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
        )
        state = PCECQueryState.from_pool_record(
            aligned_record,
            dataset=str(args.dataset),
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
        source_row = source_rows.get(int(state.query_idx), {})
        rows.append(
            build_report_row(
                state=state,
                result=result,
                source_row=source_row,
                requirement_count=len(requirements),
            )
        )
        traces.append(build_query_trace(state=state, result=result))

    retrieval_report = str(pool_payload.get("retrieval_report") or "")
    openie_path = str(pool_payload.get("openie_path") or source_payload.get("openie_path") or "")
    input_report = str(source_payload.get("input_report") or "")
    return {
        "method": METHOD_NAME,
        "paper_facing_method": PAPER_FACING_METHOD_NAME,
        "mode": "pcec_native_pool_readout",
        "dataset": str(args.dataset),
        "limit": int(len(records)),
        "pool_json": str(args.pool_json),
        "requirement_report": str(args.requirement_report),
        "binding_cache_path": str(args.binding_cache_path),
        "retrieval_report": retrieval_report,
        "openie_path": openie_path,
        "input_report": input_report,
        "pcec_contract": dict(METHOD_CONTRACT),
        "pcec_config": {
            "reader_budget_k": int(args.reader_budget_k),
            "prefix_budget_m": int(args.prefix_budget_m),
            "pool_k": int(args.pool_k),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--pool-k", type=int, default=DEFAULT_POOL_K)
    parser.add_argument("--reader-budget-k", type=int, default=DEFAULT_READER_BUDGET_K)
    parser.add_argument("--prefix-budget-m", type=int, default=DEFAULT_PREFIX_BUDGET_M)
    parser.add_argument("--pool-json", type=Path, default=None)
    parser.add_argument("--requirement-report", type=Path, default=None)
    parser.add_argument("--binding-cache-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--llm-request-name", default="qwen3-8b-train")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-binding-model", default="")
    parser.add_argument("--max-retry-attempts", type=int, default=20)
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--embedding-name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--qa-doc-max-chars", type=int, default=2048)
    parser.add_argument("--dtc-binding-max-candidates", type=int, default=DEFAULT_DTC_BINDING_MAX_CANDIDATES)
    parser.add_argument("--llm-binding-title-match-mode", default="wiki_title")
    parser.add_argument("--daec-safe-min-objective-gain", type=float, default=DEFAULT_SAFE_MIN_OBJECTIVE_GAIN)
    parser.add_argument("--daec-safe-min-swap-gain", type=float, default=DEFAULT_SAFE_MIN_SWAP_GAIN)
    parser.add_argument("--allow-missing-requirements", action="store_true")
    return parser


def resolve_default_paths(args: argparse.Namespace) -> None:
    if args.pool_json is None:
        args.pool_json = default_pool_json(str(args.dataset), int(args.pool_k), int(args.limit))
    if args.requirement_report is None:
        args.requirement_report = default_requirement_report(str(args.dataset), int(args.pool_k), int(args.limit))
    if args.binding_cache_path is None:
        args.binding_cache_path = default_binding_cache(str(args.dataset), int(args.pool_k))
    if args.output_json is None:
        args.output_json = default_output_json(
            Path(args.output_root),
            str(args.dataset),
            int(args.prefix_budget_m),
            int(args.reader_budget_k),
            int(args.pool_k),
            int(args.limit),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolve_default_paths(args)
    payload = run_native_pool(args)
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
