#!/usr/bin/env python3
"""Run Evidence Transition GraphRAG v3 variable-flow from fresh artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from evidenceflow.frozen_etv3_variable_flow.contract import (
    METHOD_CONTRACT,
    METHOD_NAME,
    QA_DOC_KEY,
)
from run_query_grounded_sto_fresh_e2e import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_NAME,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_NAME,
    DEFAULT_READER_LLM_BASE_URL,
    DEFAULT_READER_LLM_NAME,
    build_fresh_sto_index,
    SUPPORTED_DATASETS,
    embedding_endpoint_model_id,
    fresh_artifact_paths,
    install_openai_client_api_key,
    load_corpus_docs,
    load_current_fresh_sto_index,
    normalize_embedding_name_for_runtime,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.build_minimal_per_query_report import (
    build_minimal_report,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.candidate_generator import (
    AGSTOLocalGraphCandidateGenerator,
    DensePreservingAGSTOLocalGraphCandidateGenerator,
    DenseSeededAGSTOLocalGraphCandidateGenerator,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.e2e_pipeline import (
    evaluate_e2e_rows,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.evaluate_report import (
    load_nodes,
    rows_for_variant,
)
from src.hipporag.information_extraction.openie_openai import OpenIE
from src.hipporag.llm import _get_llm_class
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import compute_mdhash_id


DEFAULT_RUNNER = "query_grounded_sto_source_aligned_v3"
PURE_GRAPH_RUNNER = "query_grounded_sto_graph_native_v3"
TRANSITION_CLOSURE_RUNNER = "query_grounded_sto_transition_closure_v4"
LAYERED_TRANSITION_RUNNER = "query_grounded_sto_layered_transition_v4"
ROOT_BALANCED_TRANSITION_RUNNER = "query_grounded_sto_root_balanced_transition_v4"
DEFAULT_CANDIDATE_GENERATOR_MODE = "dense_seeded_sto"
GRAPH_ONLY_CANDIDATE_GENERATOR_MODE = "graph_only"
FORBIDDEN_FLAGS = frozenset({"--ablation-query-supported-object-handoff"})


@dataclass(frozen=True)
class V2FreshIndexArtifacts:
    save_dir: Path
    openie_path: Path
    llm_cache_dir: Path
    working_dir: Path
    doc_count: int


def _label(value: str) -> str:
    return str(value).replace("/", "_")


def select_runner(*, candidate_mode: str, readout_policy: str) -> str:
    if str(readout_policy) == "root_balanced_transition":
        return ROOT_BALANCED_TRANSITION_RUNNER
    if str(readout_policy) == "layered_transition":
        return LAYERED_TRANSITION_RUNNER
    if str(readout_policy) == "transition_closure":
        return TRANSITION_CLOSURE_RUNNER
    if str(candidate_mode) == GRAPH_ONLY_CANDIDATE_GENERATOR_MODE:
        return PURE_GRAPH_RUNNER
    return DEFAULT_RUNNER


def v2_fresh_artifact_paths(*, save_dir: Path, llm_name: str) -> V2FreshIndexArtifacts:
    llm_label = _label(llm_name)
    working_dir = Path(save_dir) / f"{llm_label}_openie_only"
    return V2FreshIndexArtifacts(
        save_dir=Path(save_dir),
        openie_path=Path(save_dir) / f"openie_results_ner_{llm_label}.json",
        llm_cache_dir=Path(save_dir) / "llm_cache",
        working_dir=working_dir,
        doc_count=0,
    )


def ensure_fresh_output_dir(path: Path, *, overwrite: bool) -> None:
    resolved = path.expanduser()
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to reuse non-empty v3 variable-flow index directory: {resolved}. "
                "Use a new --output-root or pass --overwrite-fresh-index explicitly."
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _v2_chunk_rows(docs: Sequence[str]) -> Dict[str, Mapping[str, Any]]:
    rows: Dict[str, Mapping[str, Any]] = {}
    for doc in docs:
        chunk_key = compute_mdhash_id(str(doc), prefix="chunk-")
        rows[chunk_key] = {
            "content": str(doc),
            "num_tokens": 0,
            "chunk_order": [],
            "full_doc_ids": [],
        }
    return rows


def _save_v2_openie_payload(
    *,
    openie_path: Path,
    all_openie_info: Sequence[Mapping[str, Any]],
) -> None:
    num_phrases = sum(len(row.get("extracted_entities", []) or []) for row in all_openie_info)
    sum_phrase_chars = sum(
        len(str(entity))
        for row in all_openie_info
        for entity in row.get("extracted_entities", []) or []
    )
    sum_phrase_words = sum(
        len(str(entity).split())
        for row in all_openie_info
        for entity in row.get("extracted_entities", []) or []
    )
    payload = {
        "docs": list(all_openie_info),
        "avg_ent_chars": round(sum_phrase_chars / float(num_phrases), 4) if num_phrases else 0,
        "avg_ent_words": round(sum_phrase_words / float(num_phrases), 4) if num_phrases else 0,
        "fresh_index_contract": {
            "source": METHOD_NAME,
            "indexing": "fresh_openie_only",
            "reuses_existing_openie_cache": False,
            "builds_embedding_store": False,
            "retrieval_uses_embeddings": False,
            "uses_sfb_or_support_fusion": False,
            "uses_v13b_source_report": False,
            "uses_legacy_compare_report": False,
        },
    }
    openie_path.parent.mkdir(parents=True, exist_ok=True)
    openie_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_v2_fresh_openie_index(
    *,
    docs: Sequence[str],
    save_dir: Path,
    llm_name: str,
    llm_base_url: str,
    max_new_tokens: int | None,
    max_retry_attempts: int,
    qwen_disable_thinking: bool,
    overwrite: bool,
) -> V2FreshIndexArtifacts:
    install_openai_client_api_key(base_url=str(llm_base_url or ""))
    ensure_fresh_output_dir(save_dir, overwrite=overwrite)
    artifacts = v2_fresh_artifact_paths(save_dir=save_dir, llm_name=llm_name)
    artifacts.working_dir.mkdir(parents=True, exist_ok=True)
    artifacts.llm_cache_dir.mkdir(parents=True, exist_ok=True)

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_base_url=str(llm_base_url or ""),
        llm_name=str(llm_name),
        max_new_tokens=max_new_tokens,
        max_retry_attempts=max(int(max_retry_attempts), 1),
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        qwen_disable_thinking=bool(qwen_disable_thinking),
    )
    llm_model = _get_llm_class(config)
    openie = OpenIE(llm_model=llm_model)
    chunk_rows = _v2_chunk_rows(docs)
    ner_results, triple_results = openie.batch_openie(chunk_rows)

    all_openie_info: List[Dict[str, Any]] = []
    for chunk_key, row in chunk_rows.items():
        try:
            entities = ner_results[chunk_key].unique_entities
            triples = triple_results[chunk_key].triples
        except Exception as exc:
            entities = []
            triples = []
            all_openie_info.append(
                {
                    "idx": chunk_key,
                    "passage": str(row["content"]),
                    "extracted_entities": entities,
                    "extracted_triples": triples,
                    "openie_error": str(exc),
                }
            )
            continue
        all_openie_info.append(
            {
                "idx": chunk_key,
                "passage": str(row["content"]),
                "extracted_entities": list(entities or []),
                "extracted_triples": [list(triple) for triple in triples or []],
            }
        )

    _save_v2_openie_payload(openie_path=artifacts.openie_path, all_openie_info=all_openie_info)
    return V2FreshIndexArtifacts(
        save_dir=artifacts.save_dir,
        openie_path=artifacts.openie_path,
        llm_cache_dir=artifacts.llm_cache_dir,
        working_dir=artifacts.working_dir,
        doc_count=len(all_openie_info),
    )


def load_current_v2_openie_index(
    *,
    docs: Sequence[str],
    save_dir: Path,
    llm_name: str,
) -> V2FreshIndexArtifacts:
    artifacts = v2_fresh_artifact_paths(save_dir=save_dir, llm_name=llm_name)
    if not artifacts.openie_path.exists():
        raise FileNotFoundError(f"fresh v3 variable-flow OpenIE artifact not found: {artifacts.openie_path}")
    payload = json.loads(artifacts.openie_path.read_text(encoding="utf-8"))
    contract = payload.get("fresh_index_contract", {}) or {}
    if contract.get("source") != METHOD_NAME:
        raise ValueError(
            "Refusing to reuse an index without the v3 variable-flow fresh OpenIE contract: "
            f"{artifacts.openie_path}"
        )
    if bool(contract.get("builds_embedding_store", True)):
        raise ValueError(f"Refusing to reuse embedding-backed artifact for v3 variable-flow: {artifacts.openie_path}")
    return V2FreshIndexArtifacts(
        save_dir=artifacts.save_dir,
        openie_path=artifacts.openie_path,
        llm_cache_dir=artifacts.llm_cache_dir,
        working_dir=artifacts.working_dir,
        doc_count=len(docs),
    )


def parse_datasets(value: str) -> List[str]:
    datasets = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = [dataset for dataset in datasets if dataset not in SUPPORTED_DATASETS]
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}; supported={SUPPORTED_DATASETS}")
    return datasets


def write_retrieval_markdown(payload: Mapping[str, Any], output_md: Path) -> None:
    metrics = payload.get("metrics", {}) or {}
    config = payload.get("config", {}) or {}
    index = payload.get("fresh_index", {}) or {}
    lines = [
        "# Evidence Transition GraphRAG v3 Variable-Flow Retrieval",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| method | {payload.get('method', '')} |",
        f"| rows | {payload.get('row_count', 0)} |",
        f"| fresh OpenIE | {index.get('openie_path', '')} |",
        f"| fresh embeddings | {index.get('chunk_embedding_store_path', '')} |",
        f"| candidate source | {config.get('candidate_source', '')} |",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| R@5 | {float(metrics.get('r5', 0.0)):.4f} |",
        f"| all-gold@5 | {float(metrics.get('all_gold_at5', 0.0)):.4f} |",
        f"| mean certified docs@5 | {float(metrics.get('mean_certified_doc_count_top5', 0.0)):.2f} |",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(
    *,
    output_md: Path,
    retrieval_payloads: Sequence[Mapping[str, Any]],
    qa_payload: Mapping[str, Any] | None,
) -> None:
    qa_by_dataset: Dict[str, Mapping[str, Any]] = {}
    if qa_payload:
        for dataset in qa_payload.get("datasets", []) or []:
            methods = dataset.get("methods", {}) or {}
            method_result = methods.get(METHOD_NAME, {})
            qa_by_dataset[str(dataset.get("dataset") or "")] = method_result.get("metrics", {}) or {}

    lines = [
        "# Evidence Transition GraphRAG v3 Variable-Flow Summary",
        "",
        "| dataset | rows | R@5 | EM | F1 | candidate source | fresh OpenIE | fresh embeddings |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for payload in retrieval_payloads:
        dataset = str(payload.get("dataset") or "")
        metrics = payload.get("metrics", {}) or {}
        qa_metrics = qa_by_dataset.get(dataset, {})
        index = payload.get("fresh_index", {}) or {}
        config = payload.get("config", {}) or {}
        r5 = metrics.get("r5")
        em = qa_metrics.get("ExactMatch")
        f1 = qa_metrics.get("F1")
        lines.append(
            f"| {dataset} | {int(payload.get('row_count', 0) or 0)} | "
            f"{'' if r5 is None else f'{float(r5):.4f}'} | "
            f"{'' if em is None else f'{float(em):.4f}'} | "
            f"{'' if f1 is None else f'{float(f1):.4f}'} | "
            f"{config.get('candidate_source', '')} | "
            f"{index.get('openie_path', '')} | "
            f"{index.get('chunk_embedding_store_path', '')} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_v2_qa_report(
    *,
    retrieval_jsons: Sequence[Path],
    output_json: Path,
) -> Mapping[str, Any]:
    datasets = []
    for retrieval_json in retrieval_jsons:
        payload = json.loads(retrieval_json.read_text(encoding="utf-8"))
        rows = []
        for row in payload.get("rows", []) or []:
            copied = dict(row)
            copied[QA_DOC_KEY] = list(row.get("retrieved_doc_indices_top5", []) or [])[:5]
            rows.append(copied)
        datasets.append(
            {
                "dataset": str(payload.get("dataset") or ""),
                "report_path": str(payload.get("input_report") or ""),
                "openie_path": str(payload.get("openie_path") or ""),
                "rows": rows,
                "source_retrieval_json": str(retrieval_json),
                "metrics": dict(payload.get("metrics", {}) or {}),
            }
        )
    output = {
        "method": METHOD_NAME,
        "canonical_method": METHOD_NAME,
        "public_doc_key": QA_DOC_KEY,
        "format": "transition_top5_qa_input",
        "datasets": datasets,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def run_one_dataset(args: argparse.Namespace, dataset: str) -> Mapping[str, Any]:
    data_root = Path(args.data_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    dataset_dir = output_root / dataset
    index_dir = dataset_dir / "index"
    reports_dir = dataset_dir / "reports"
    dataset_path = data_root / f"{dataset}.json"
    corpus_path = data_root / f"{dataset}_corpus.json"
    docs = load_corpus_docs(corpus_path, max_index_docs=int(args.max_index_docs))
    candidate_mode = str(args.candidate_generator_mode)

    if args.dry_run:
        if candidate_mode == GRAPH_ONLY_CANDIDATE_GENERATOR_MODE:
            artifacts = v2_fresh_artifact_paths(
                save_dir=index_dir,
                llm_name=str(args.llm_name),
            )
            chunk_embedding_store_path = ""
        else:
            runtime_embedding_name = normalize_embedding_name_for_runtime(
                embedding_name=str(args.embedding_name),
                embedding_base_url=str(args.embedding_base_url),
            )
            artifacts = fresh_artifact_paths(
                save_dir=index_dir,
                llm_name=str(args.llm_name),
                embedding_name=runtime_embedding_name,
            )
            chunk_embedding_store_path = str(artifacts.chunk_embedding_store_path)
        return {
            "method": METHOD_NAME,
            "runner": select_runner(
                candidate_mode=candidate_mode,
                readout_policy=str(args.readout_policy),
            ),
            "dataset": dataset,
            "row_count": max(int(args.max_queries), 0),
            "metrics": {},
            "dry_run": True,
            "fresh_index": {
                "openie_path": str(artifacts.openie_path),
                "chunk_embedding_store_path": chunk_embedding_store_path,
                "doc_count": len(docs),
            },
            "config": {
                **dict(METHOD_CONTRACT),
                "candidate_generator_mode": candidate_mode,
                "readout_policy": str(args.readout_policy),
            },
        }

    if candidate_mode == GRAPH_ONLY_CANDIDATE_GENERATOR_MODE and bool(args.reuse_current_fresh_index):
        artifacts = load_current_v2_openie_index(
            docs=docs,
            save_dir=index_dir,
            llm_name=str(args.llm_name),
        )
        index_execution = "reuse_current_fresh_index"
    elif candidate_mode == GRAPH_ONLY_CANDIDATE_GENERATOR_MODE:
        artifacts = build_v2_fresh_openie_index(
            docs=docs,
            save_dir=index_dir,
            llm_name=str(args.llm_name),
            llm_base_url=str(args.llm_base_url),
            max_new_tokens=args.max_new_tokens,
            max_retry_attempts=int(args.max_retry_attempts),
            qwen_disable_thinking=bool(args.qwen_disable_thinking),
            overwrite=bool(args.overwrite_fresh_index),
        )
        index_execution = "rebuilt_from_raw_corpus"
    elif bool(args.reuse_current_fresh_index):
        artifacts = load_current_fresh_sto_index(
            docs=docs,
            save_dir=index_dir,
            llm_name=str(args.llm_name),
            embedding_name=str(args.embedding_name),
            embedding_base_url=str(args.embedding_base_url),
        )
        index_execution = "reuse_current_fresh_index"
    else:
        artifacts = build_fresh_sto_index(
            docs=docs,
            save_dir=index_dir,
            llm_name=str(args.llm_name),
            llm_base_url=str(args.llm_base_url),
            embedding_name=str(args.embedding_name),
            embedding_base_url=str(args.embedding_base_url),
            embedding_batch_size=int(args.embedding_batch_size),
            max_new_tokens=args.max_new_tokens,
            max_retry_attempts=int(args.max_retry_attempts),
            qwen_disable_thinking=bool(args.qwen_disable_thinking),
            overwrite=bool(args.overwrite_fresh_index),
        )
        index_execution = "rebuilt_from_raw_corpus"

    report_path = reports_dir / f"{dataset}_minimal_per_query_fresh_v3_variable_flow.json"
    report = build_minimal_report(
        dataset=dataset,
        dataset_path=dataset_path,
        openie_path=artifacts.openie_path,
        output_json=report_path,
    )
    nodes = load_nodes(artifacts.openie_path)
    rows = rows_for_variant(report, str(args.variant))
    if int(args.max_queries) > 0:
        rows = rows[: int(args.max_queries)]

    if candidate_mode == GRAPH_ONLY_CANDIDATE_GENERATOR_MODE:
        candidate_generator = AGSTOLocalGraphCandidateGenerator.build(
            nodes=tuple(nodes),
            textual_seed_top_k=max(int(args.agsto_textual_seed_top_k), 0),
            max_endpoint_degree=max(int(args.agsto_max_endpoint_degree), 1),
            closure_hops=max(int(args.agsto_closure_hops), 0),
            enable_query_supported_same_object_handoff=bool(
                args.ablation_query_supported_object_handoff
            ),
            enable_variable_flow_traversal=bool(args.enable_variable_flow_traversal),
        )
        runner = select_runner(
            candidate_mode=candidate_mode,
            readout_policy=str(args.readout_policy),
        )
    else:
        chunk_embedding_store_path = getattr(artifacts, "chunk_embedding_store_path", None)
        if chunk_embedding_store_path is None:
            raise ValueError(
                "dense-aligned v3 variable-flow candidate generation requires a fresh chunk embedding store"
            )
        embedding_model_id = embedding_endpoint_model_id(
            normalize_embedding_name_for_runtime(
                embedding_name=str(args.embedding_name),
                embedding_base_url=str(args.embedding_base_url),
            )
        )
        if candidate_mode == "dense_preserving_sto":
            candidate_generator = DensePreservingAGSTOLocalGraphCandidateGenerator.from_parquet(
                nodes=tuple(nodes),
                embedding_store_path=Path(chunk_embedding_store_path),
                embedding_base_url=str(args.embedding_base_url),
                embedding_model_name=embedding_model_id,
                preserved_dense_count=max(int(args.preserved_dense_count), 1),
                dense_seed_count=max(int(args.dense_root_count), 1),
                agsto_textual_seed_top_k=max(int(args.agsto_textual_seed_top_k), 0),
                agsto_max_endpoint_degree=max(int(args.agsto_max_endpoint_degree), 1),
                agsto_closure_hops=max(int(args.agsto_closure_hops), 0),
                enable_query_supported_same_object_handoff=bool(
                    args.ablation_query_supported_object_handoff
                ),
                enable_variable_flow_traversal=bool(args.enable_variable_flow_traversal),
            )
        else:
            candidate_generator = DenseSeededAGSTOLocalGraphCandidateGenerator.from_parquet(
                nodes=tuple(nodes),
                embedding_store_path=Path(chunk_embedding_store_path),
                embedding_base_url=str(args.embedding_base_url),
                embedding_model_name=embedding_model_id,
                dense_seed_count=max(int(args.dense_root_count), 1),
                source_prior_prefix_count=max(int(args.top_k), 1),
                agsto_textual_seed_top_k=max(int(args.agsto_textual_seed_top_k), 0),
                agsto_max_endpoint_degree=max(int(args.agsto_max_endpoint_degree), 1),
                agsto_closure_hops=max(int(args.agsto_closure_hops), 0),
                enable_query_supported_same_object_handoff=bool(
                    args.ablation_query_supported_object_handoff
                ),
                enable_variable_flow_traversal=bool(args.enable_variable_flow_traversal),
            )
        runner = select_runner(
            candidate_mode=candidate_mode,
            readout_policy=str(args.readout_policy),
        )
    dataset_result = evaluate_e2e_rows(
        rows=rows,
        nodes=nodes,
        candidate_generator=candidate_generator,
        runner=runner,
        top_k=max(int(args.top_k), 1),
        candidate_pool_k=max(int(args.candidate_pool_k), 1),
        certificate_policy="canonical_source_text",
        enable_pipeline_candidate_expansion=False,
        enable_pipeline_replacement=False,
        allow_frontier_pair_insertion=False,
        assembly_policy="baseline_aligned_graph_entry_no_weighted_fusion",
        source_prior_policy="dense_entry_source_prior" if candidate_mode != GRAPH_ONLY_CANDIDATE_GENERATOR_MODE else "none",
    )
    chunk_embedding_store_path = str(getattr(artifacts, "chunk_embedding_store_path", "") or "")

    payload = {
        "method": METHOD_NAME,
        "runner": runner,
        "dataset": dataset,
        "input_report": str(report_path.resolve()),
        "openie_path": str(artifacts.openie_path.resolve()),
        "source_variant": str(args.variant),
        "fresh_index": {
            "save_dir": str(artifacts.save_dir.resolve()),
            "openie_path": str(artifacts.openie_path.resolve()),
            "llm_cache_dir": str(artifacts.llm_cache_dir.resolve()),
            "working_dir": str(artifacts.working_dir.resolve()),
            "chunk_embedding_store_path": chunk_embedding_store_path,
            "doc_count": int(artifacts.doc_count),
            "index_execution": index_execution,
            "reuses_current_fresh_index": bool(args.reuse_current_fresh_index),
            "builds_embedding_store": candidate_mode != GRAPH_ONLY_CANDIDATE_GENERATOR_MODE,
            "candidate_generation_uses_embeddings": candidate_mode != GRAPH_ONLY_CANDIDATE_GENERATOR_MODE,
            "final_selection_uses_embeddings": False,
            "reuses_legacy_openie_cache": False,
            "reuses_legacy_embedding_cache": False,
        },
        "config": {
            **dict(METHOD_CONTRACT),
            "candidate_source": f"fresh_{candidate_mode}_query_local_sto_graph",
            "candidate_generator_mode": candidate_mode,
            "readout_policy": str(args.readout_policy),
            "readout_runner": runner,
            "source_prior_policy": (
                "entry_only"
                if str(args.readout_policy) == "transition_closure"
                else "structural_guard_only"
                if str(args.readout_policy) in {
                    "layered_transition",
                    "root_balanced_transition",
                }
                else (
                    "dense_entry_source_prior"
                    if candidate_mode != GRAPH_ONLY_CANDIDATE_GENERATOR_MODE
                    else "none"
                )
            ),
            "uses_source_prior_guard": str(args.readout_policy) != "transition_closure",
            "candidate_pool_k": max(int(args.candidate_pool_k), 1),
            "top_k": max(int(args.top_k), 1),
            "max_queries": max(int(args.max_queries), 0),
            "llm_name": str(args.llm_name),
            "llm_base_url": str(args.llm_base_url),
            "index_embedding_name": normalize_embedding_name_for_runtime(
                embedding_name=str(args.embedding_name),
                embedding_base_url=str(args.embedding_base_url),
            ),
            "embedding_endpoint_model_id": embedding_endpoint_model_id(
                normalize_embedding_name_for_runtime(
                    embedding_name=str(args.embedding_name),
                    embedding_base_url=str(args.embedding_base_url),
                )
            ),
            "embedding_base_url": str(args.embedding_base_url),
            "candidate_generation_uses_embeddings": candidate_mode != GRAPH_ONLY_CANDIDATE_GENERATOR_MODE,
            "final_selection_uses_embeddings": False,
            "preserved_dense_count": max(int(args.preserved_dense_count), 1),
            "dense_root_count": max(int(args.dense_root_count), 1),
            "reader_embedding_name": normalize_embedding_name_for_runtime(
                embedding_name=str(args.embedding_name),
                embedding_base_url=str(args.embedding_base_url),
            ),
            "reader_embedding_base_url": str(args.embedding_base_url),
            "agsto_textual_seed_top_k": max(int(args.agsto_textual_seed_top_k), 0),
            "agsto_max_endpoint_degree": max(int(args.agsto_max_endpoint_degree), 1),
            "agsto_closure_hops": max(int(args.agsto_closure_hops), 0),
            "ablation_query_supported_object_handoff": bool(
                args.ablation_query_supported_object_handoff
            ),
            "enable_variable_flow_traversal": bool(args.enable_variable_flow_traversal),
        },
        **dataset_result,
    }
    output_json = reports_dir / f"{dataset}_{METHOD_NAME}_retrieval.json"
    output_md = reports_dir / f"{dataset}_{METHOD_NAME}_retrieval.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_retrieval_markdown(payload, output_md)
    return payload


def run_optional_qa(
    *,
    args: argparse.Namespace,
    retrieval_payloads: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not args.run_qa or args.dry_run:
        return None
    from evidenceflow.frozen_etv3_variable_flow.run_reader_qa import main as qa_main

    install_openai_client_api_key(base_url=str(args.reader_llm_base_url or ""))
    output_root = Path(args.output_root).expanduser()
    retrieval_jsons = [
        Path(str(payload["input_report"])).parent
        / f"{payload['dataset']}_{METHOD_NAME}_retrieval.json"
        for payload in retrieval_payloads
    ]
    qa_json = output_root / "reports" / f"{METHOD_NAME}_qa.json"
    qa_md = output_root / "reports" / f"{METHOD_NAME}_qa.md"
    qa_args: List[str] = [
        "--retrieval-reports",
        ",".join(str(path) for path in retrieval_jsons),
        "--max-queries",
        str(max(int(args.max_queries), 0)),
        "--qa-top-k",
        str(max(int(args.top_k), 1)),
        "--save-dir",
        str(output_root / "reader_runtime"),
        "--llm-name",
        str(args.reader_llm_name),
        "--llm-base-url",
        str(args.reader_llm_base_url),
        "--max-new-tokens",
        str(max(int(args.reader_max_new_tokens), 1)),
        "--embedding-name",
        embedding_endpoint_model_id(
            normalize_embedding_name_for_runtime(
                embedding_name=str(args.embedding_name),
                embedding_base_url=str(args.embedding_base_url),
            )
        ),
        "--embedding-base-url",
        str(args.embedding_base_url),
        "--embedding-batch-size",
        str(max(int(args.embedding_batch_size), 1)),
        "--openie-mode",
        "online",
        "--output-json",
        str(qa_json),
        "--output-md",
        str(qa_md),
    ]
    if bool(args.reader_qwen_disable_thinking):
        qa_args.append("--qwen-disable-thinking")
    qa_main(qa_args)
    return json.loads(qa_json.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="2wikimultihopqa,musique,hotpotqa")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-index-docs", type=int, default=0)
    parser.add_argument("--llm-name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-retry-attempts", type=int, default=5)
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--embedding-name", default=DEFAULT_EMBEDDING_NAME)
    parser.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--variant", default="hipporag_v2")
    parser.add_argument("--candidate-pool-k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--readout-policy",
        choices=(
            "source_aligned",
            "transition_closure",
            "layered_transition",
            "root_balanced_transition",
        ),
        default="source_aligned",
        help=(
            "source_aligned keeps the ETv3 source-prior guarded readout. "
            "transition_closure runs the ETv4 experimental readout: dense/textual "
            "retrieval supplies graph roots only, and final top-k is selected by "
            "query-local STO transition closure. layered_transition keeps the "
            "ETv4 graph readout but orders the frontier by STO edge semantic layer. "
            "root_balanced_transition balances textual/symbolic roots with one "
            "non-root transition witness per root."
        ),
    )
    parser.add_argument(
        "--candidate-generator-mode",
        choices=("dense_seeded_sto", "dense_preserving_sto", GRAPH_ONLY_CANDIDATE_GENERATOR_MODE),
        default=DEFAULT_CANDIDATE_GENERATOR_MODE,
        help=(
            "dense_seeded_sto aligns v3 with HippoRAGv2/PropRAG-style non-graph entry: "
            "dense retrieval seeds the query-local STO graph and source prior, while "
            "final top-k selection remains source-authorized STO. graph_only keeps the "
            "pure OpenIE/STO ablation."
        ),
    )
    parser.add_argument("--preserved-dense-count", type=int, default=200)
    parser.add_argument("--dense-root-count", type=int, default=20)
    parser.add_argument("--agsto-textual-seed-top-k", type=int, default=20)
    parser.add_argument("--agsto-max-endpoint-degree", type=int, default=30)
    parser.add_argument("--agsto-closure-hops", type=int, default=2)
    parser.add_argument(
        "--ablation-query-supported-object-handoff",
        action="store_true",
        help=(
            "Diagnostic ablation only: allow query-relation-supported same_object "
            "handoff during local STO admission/source-prior guarding. Disabled "
            "by default to keep the paper-facing v3 variable-flow path clean."
        ),
    )
    parser.add_argument(
        "--enable-variable-flow-traversal",
        action="store_true",
        help=(
            "Enable query-rooted variable-flow traversal: same-object transitions "
            "are traversable only when their shared endpoint is active in the "
            "current document's fact flow. This is a closed-form graph semantic, "
            "not a relation-trigger ablation."
        ),
    )
    parser.add_argument("--reuse-current-fresh-index", action="store_true")
    parser.add_argument("--overwrite-fresh-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-qa", action="store_true")
    parser.add_argument("--reader-llm-name", default=DEFAULT_READER_LLM_NAME)
    parser.add_argument("--reader-llm-base-url", default=DEFAULT_READER_LLM_BASE_URL)
    parser.add_argument("--reader-max-new-tokens", type=int, default=2048)
    parser.add_argument("--reader-qwen-disable-thinking", action="store_true")
    return parser


def _normalize_args(argv: Sequence[str] | None) -> List[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    forbidden = sorted(flag for flag in FORBIDDEN_FLAGS if flag in args)
    if forbidden:
        raise ValueError(
            "ETv3 variable-flow is the clean line; do not combine it with "
            f"diagnostic object-handoff flags: {', '.join(forbidden)}"
        )
    if "--enable-variable-flow-traversal" not in args:
        args.append("--enable-variable-flow-traversal")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(_normalize_args(argv))
    if bool(args.reuse_current_fresh_index) and bool(args.overwrite_fresh_index):
        raise ValueError("--reuse-current-fresh-index and --overwrite-fresh-index cannot be combined")
    datasets = parse_datasets(str(args.datasets))
    retrieval_payloads = [run_one_dataset(args, dataset) for dataset in datasets]
    qa_payload = run_optional_qa(args=args, retrieval_payloads=retrieval_payloads)
    output_root = Path(args.output_root).expanduser()
    write_summary_markdown(
        output_md=output_root / "reports" / f"{METHOD_NAME}_summary.md",
        retrieval_payloads=retrieval_payloads,
        qa_payload=qa_payload,
    )
    print(
        json.dumps(
            {
                payload["dataset"]: {
                    "method": payload.get("method"),
                    "rows": payload.get("row_count", 0),
                    "R@5": (payload.get("metrics", {}) or {}).get("r5"),
                }
                for payload in retrieval_payloads
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
