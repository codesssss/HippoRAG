#!/usr/bin/env python3
"""Run fresh-index Evidence Transition GraphRAG end to end.

This runner is intentionally stricter than the historical experiment scripts:

raw benchmark corpus
-> fresh passage embeddings
-> fresh OpenIE triples
-> minimal per-query report from raw QA data
-> evidence-transition graph retrieval
-> optional fixed-top5 reader QA

It does not read legacy compare reports, SFB/support_fusion reports, V13B
source reports, or existing OpenIE / embedding caches.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

from source_authorized_vocab_strict_retrieval.build_minimal_per_query_report import (
    build_minimal_report,
)
from source_authorized_vocab_strict_retrieval.candidate_generator import (
    DenseSeededAGSTOLocalGraphCandidateGenerator,
    DensePreservingAGSTOLocalGraphCandidateGenerator,
)
from source_authorized_vocab_strict_retrieval.e2e_pipeline import (
    evaluate_e2e_rows,
    method_name_for_runner,
)
from source_authorized_vocab_strict_retrieval.evaluate_report import (
    load_nodes,
    rows_for_variant,
)
from source_authorized_vocab_strict_retrieval.export_qa_transition_report import (
    QUERY_GROUNDED_STO_QA_METHOD,
    QUERY_GROUNDED_STO_V2_QA_METHOD,
    export_qa_transition_report,
)
from src.hipporag.embedding_model import _get_embedding_model_class
from src.hipporag.embedding_store import EmbeddingStore
from src.hipporag.information_extraction.openie_openai import OpenIE
from src.hipporag.llm import _get_llm_class
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import compute_mdhash_id


logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = ("2wikimultihopqa", "musique", "hotpotqa")
DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DEFAULT_LLM_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_LLM_NAME = "gpt-4o-mini"
DEFAULT_EMBEDDING_NAME = "nvidia/NV-Embed-v2"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8019/v1/embeddings"
DEFAULT_READER_LLM_NAME = "qwen3-8b-train"
DEFAULT_READER_LLM_BASE_URL = "http://localhost:8041/v1"
ETV2_RUNNERS = frozenset(
    {
        "agsto_graph_native_v2",
        "query_grounded_sto_graph_native_v2",
        "evidence_transition_v2",
    }
)


def retrieval_output_stem(*, dataset: str, runner: str) -> str:
    method_suffix = "query_grounded_sto_v2" if str(runner) in ETV2_RUNNERS else "query_grounded_sto"
    return f"{dataset}_fresh_{method_suffix}_retrieval"


def qa_output_stem(*, runner: str) -> str:
    method_suffix = "query_grounded_sto_v2" if str(runner) in ETV2_RUNNERS else "query_grounded_sto"
    return f"fresh_{method_suffix}_qa"


def summary_output_stem(*, runner: str) -> str:
    method_suffix = "query_grounded_sto_v2" if str(runner) in ETV2_RUNNERS else "query_grounded_sto"
    return f"fresh_{method_suffix}_summary"


@dataclass(frozen=True)
class FreshIndexArtifacts:
    """Paths produced by a fresh STO indexing run."""

    save_dir: Path
    openie_path: Path
    chunk_embedding_store_path: Path
    llm_cache_dir: Path
    working_dir: Path
    doc_count: int


def _label(value: str) -> str:
    return str(value).replace("/", "_")


def normalize_embedding_name_for_runtime(
    *,
    embedding_name: str,
    embedding_base_url: str,
) -> str:
    """Select the embedding backend implied by the CLI settings.

    If an OpenAI-compatible embedding endpoint is supplied, local model loading
    is not the intended behavior.  Prefix with ``VLLM/`` unless the caller has
    already selected a concrete backend.
    """

    name = str(embedding_name or "")
    base_url = str(embedding_base_url or "")
    if not base_url:
        return name
    if name.startswith(("VLLM/", "Transformers/")):
        return name
    if "text-embedding" in name or "cohere" in name:
        return name
    return f"VLLM/{name}"


def embedding_endpoint_model_id(embedding_name: str) -> str:
    """Return the model id expected by an OpenAI-compatible embedding server."""

    name = str(embedding_name or "")
    if name.startswith("VLLM/"):
        return name[len("VLLM/") :]
    return name


def fresh_artifact_paths(
    *,
    save_dir: Path,
    llm_name: str,
    embedding_name: str,
) -> FreshIndexArtifacts:
    llm_label = _label(llm_name)
    embedding_label = _label(embedding_name)
    working_dir = Path(save_dir) / f"{llm_label}_{embedding_label}"
    return FreshIndexArtifacts(
        save_dir=Path(save_dir),
        openie_path=Path(save_dir) / f"openie_results_ner_{llm_label}.json",
        chunk_embedding_store_path=working_dir / "chunk_embeddings" / "vdb_chunk.parquet",
        llm_cache_dir=Path(save_dir) / "llm_cache",
        working_dir=working_dir,
        doc_count=0,
    )


def ensure_fresh_output_dir(path: Path, *, overwrite: bool) -> None:
    """Create a fresh output directory without silently reusing artifacts."""

    resolved = path.expanduser()
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to reuse non-empty fresh index directory: {resolved}. "
                "Use a new --output-root or pass --overwrite-fresh-index explicitly."
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _is_local_base_url(base_url: str) -> bool:
    hostname = urlparse(str(base_url or "")).hostname
    return hostname in {"localhost", "127.0.0.1", "::1"}


def install_openai_client_api_key(*, base_url: str) -> None:
    """Install the API key expected by OpenAI-compatible clients.

    Remote endpoints must use a real key supplied through the environment.
    Local vLLM endpoints usually ignore auth, but the OpenAI Python client still
    requires the variable to be present.
    """

    if os.environ.get("OPENAI_API_KEY"):
        return
    yunwu_key = os.environ.get("YUNWU_API_KEY")
    if yunwu_key:
        os.environ["OPENAI_API_KEY"] = yunwu_key
        return
    if _is_local_base_url(base_url):
        os.environ["OPENAI_API_KEY"] = "EMPTY"


def load_corpus_docs(corpus_path: Path, *, max_index_docs: int = 0) -> List[str]:
    corpus = json.loads(corpus_path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(corpus, list):
        raise ValueError(f"corpus must be a list: {corpus_path}")
    docs = [f"{row['title']}\n{row['text']}" for row in corpus]
    if int(max_index_docs) > 0:
        docs = docs[: int(max_index_docs)]
    return docs


def _ordered_chunk_rows(docs: Sequence[str], store: EmbeddingStore) -> Dict[str, Mapping[str, Any]]:
    rows = store.get_all_id_to_rows()
    ordered: Dict[str, Mapping[str, Any]] = {}
    for doc in docs:
        chunk_key = compute_mdhash_id(doc, prefix="chunk-")
        row = rows.get(chunk_key)
        if row is None:
            raise KeyError(f"missing chunk embedding row for {chunk_key}")
        ordered[chunk_key] = row
    return ordered


def _split_openie_batch_results(openie_results: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return NER and triple outputs across HippoRAG OpenIE API versions."""
    if not isinstance(openie_results, tuple) or len(openie_results) < 2:
        raise TypeError(
            "OpenIE.batch_openie must return at least (ner_results, triple_results); "
            f"got {type(openie_results).__name__}"
        )
    return openie_results[0], openie_results[1]


def _save_openie_payload(openie_path: Path, all_openie_info: Sequence[Mapping[str, Any]]) -> None:
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
            "source": "run_query_grounded_sto_fresh_e2e",
            "reuses_existing_openie_cache": False,
            "reuses_existing_embedding_cache": False,
            "builds_legacy_compare_report": False,
            "uses_sfb_or_support_fusion": False,
            "uses_v13b_source_report": False,
        },
    }
    openie_path.parent.mkdir(parents=True, exist_ok=True)
    openie_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_fresh_sto_index(
    *,
    docs: Sequence[str],
    save_dir: Path,
    llm_name: str,
    llm_base_url: str,
    embedding_name: str,
    embedding_base_url: str,
    embedding_batch_size: int,
    max_new_tokens: int | None,
    max_retry_attempts: int,
    qwen_disable_thinking: bool,
    overwrite: bool,
) -> FreshIndexArtifacts:
    """Build only the artifacts consumed by the STO GraphRAG pipeline."""

    runtime_embedding_name = normalize_embedding_name_for_runtime(
        embedding_name=embedding_name,
        embedding_base_url=embedding_base_url,
    )
    install_openai_client_api_key(base_url=str(llm_base_url or ""))
    ensure_fresh_output_dir(save_dir, overwrite=overwrite)
    artifacts = fresh_artifact_paths(
        save_dir=save_dir,
        llm_name=llm_name,
        embedding_name=runtime_embedding_name,
    )

    config_field_names = {field_info.name for field_info in fields(BaseConfig)}
    config_kwargs = {
        "save_dir": str(save_dir),
        "llm_base_url": str(llm_base_url or ""),
        "llm_name": str(llm_name),
        "embedding_base_url": str(embedding_base_url or ""),
        "embedding_model_name": runtime_embedding_name,
        "embedding_batch_size": max(int(embedding_batch_size), 1),
        "max_new_tokens": max_new_tokens,
        "max_retry_attempts": max(int(max_retry_attempts), 1),
        "force_index_from_scratch": True,
        "force_openie_from_scratch": True,
        # STO GraphRAG consumes entities/triples only; causal OpenIE is unused here.
        "causal_enabled": False,
        "qwen_disable_thinking": bool(qwen_disable_thinking),
    }
    config = BaseConfig(**{key: value for key, value in config_kwargs.items() if key in config_field_names})
    if "causal_enabled" not in config_field_names:
        setattr(config, "causal_enabled", False)
    if "qwen_disable_thinking" not in config_field_names:
        setattr(config, "qwen_disable_thinking", bool(qwen_disable_thinking))

    embedding_model = _get_embedding_model_class(runtime_embedding_name)(
        global_config=config,
        embedding_model_name=runtime_embedding_name,
    )
    chunk_store = EmbeddingStore(
        embedding_model,
        str(artifacts.working_dir / "chunk_embeddings"),
        max(int(embedding_batch_size), 1),
        "chunk",
    )
    logger.info("Encoding %d fresh corpus passages.", len(docs))
    chunk_store.insert_strings(list(docs))
    chunk_rows = _ordered_chunk_rows(docs, chunk_store)

    llm_model = _get_llm_class(config)
    openie = OpenIE(llm_model=llm_model)
    logger.info("Running fresh OpenIE for %d corpus passages.", len(chunk_rows))
    ner_results, triple_results = _split_openie_batch_results(openie.batch_openie(chunk_rows))

    all_openie_info: List[Dict[str, Any]] = []
    for chunk_key, row in chunk_rows.items():
        try:
            entities = ner_results[chunk_key].unique_entities
            triples = triple_results[chunk_key].triples
        except Exception as exc:  # defensive: keep doc index stable
            logger.warning("OpenIE failed for chunk %s: %s", chunk_key, exc)
            entities = []
            triples = []
        all_openie_info.append(
            {
                "idx": chunk_key,
                "passage": str(row["content"]),
                "extracted_entities": list(entities or []),
                "extracted_triples": [list(triple) for triple in triples or []],
            }
        )

    _save_openie_payload(artifacts.openie_path, all_openie_info)
    if not artifacts.chunk_embedding_store_path.exists():
        raise FileNotFoundError(
            f"fresh chunk embedding store was not written: {artifacts.chunk_embedding_store_path}"
        )
    if not artifacts.openie_path.exists():
        raise FileNotFoundError(f"fresh OpenIE file was not written: {artifacts.openie_path}")

    return FreshIndexArtifacts(
        save_dir=artifacts.save_dir,
        openie_path=artifacts.openie_path,
        chunk_embedding_store_path=artifacts.chunk_embedding_store_path,
        llm_cache_dir=artifacts.llm_cache_dir,
        working_dir=artifacts.working_dir,
        doc_count=len(all_openie_info),
    )


def load_current_fresh_sto_index(
    *,
    docs: Sequence[str],
    save_dir: Path,
    llm_name: str,
    embedding_name: str,
    embedding_base_url: str,
) -> FreshIndexArtifacts:
    """Load artifacts produced by this runner in the current output tree."""

    runtime_embedding_name = normalize_embedding_name_for_runtime(
        embedding_name=embedding_name,
        embedding_base_url=embedding_base_url,
    )
    artifacts = fresh_artifact_paths(
        save_dir=save_dir,
        llm_name=llm_name,
        embedding_name=runtime_embedding_name,
    )
    if not artifacts.openie_path.exists():
        raise FileNotFoundError(f"fresh OpenIE artifact not found: {artifacts.openie_path}")
    if not artifacts.chunk_embedding_store_path.exists():
        raise FileNotFoundError(
            f"fresh chunk embedding artifact not found: {artifacts.chunk_embedding_store_path}"
        )
    payload = json.loads(artifacts.openie_path.read_text(encoding="utf-8"))
    contract = payload.get("fresh_index_contract", {}) or {}
    if contract.get("source") != "run_query_grounded_sto_fresh_e2e":
        raise ValueError(
            "Refusing to reuse an index without the fresh runner contract: "
            f"{artifacts.openie_path}"
        )
    if bool(contract.get("reuses_existing_openie_cache", True)):
        raise ValueError(f"Refusing to reuse non-fresh OpenIE artifact: {artifacts.openie_path}")
    return FreshIndexArtifacts(
        save_dir=artifacts.save_dir,
        openie_path=artifacts.openie_path,
        chunk_embedding_store_path=artifacts.chunk_embedding_store_path,
        llm_cache_dir=artifacts.llm_cache_dir,
        working_dir=artifacts.working_dir,
        doc_count=len(docs),
    )


def write_retrieval_markdown(payload: Mapping[str, Any], output_md: Path) -> None:
    metrics = payload.get("metrics", {}) or {}
    config = payload.get("config", {}) or {}
    index = payload.get("fresh_index", {}) or {}
    lines = [
        "# Fresh Query-Grounded STO GraphRAG Retrieval",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset | {payload.get('dataset', '')} |",
        f"| method | {payload.get('method', '')} |",
        f"| runner | {payload.get('runner', '')} |",
        f"| rows | {payload.get('row_count', 0)} |",
        f"| fresh OpenIE | {index.get('openie_path', '')} |",
        f"| fresh chunk embeddings | {index.get('chunk_embedding_store_path', '')} |",
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
            method_result = (
                methods.get(QUERY_GROUNDED_STO_V2_QA_METHOD, {})
                or methods.get(QUERY_GROUNDED_STO_QA_METHOD, {})
                or methods.get(
                    "source_authorized_vocab_strict_graphrag",
                    {},
                )
            )
            qa_by_dataset[str(dataset.get("dataset") or "")] = method_result.get("metrics", {}) or {}

    lines = [
        "# Evidence Transition GraphRAG Summary",
        "",
        "| dataset | rows | R@5 | EM | F1 | fresh OpenIE | fresh embeddings |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for payload in retrieval_payloads:
        dataset = str(payload.get("dataset") or "")
        metrics = payload.get("metrics", {}) or {}
        qa_metrics = qa_by_dataset.get(dataset, {})
        index = payload.get("fresh_index", {}) or {}
        r5 = metrics.get("r5")
        em = qa_metrics.get("ExactMatch")
        f1 = qa_metrics.get("F1")
        lines.append(
            f"| {dataset} | {int(payload.get('row_count', 0) or 0)} | "
            f"{'' if r5 is None else f'{float(r5):.4f}'} | "
            f"{'' if em is None else f'{float(em):.4f}'} | "
            f"{'' if f1 is None else f'{float(f1):.4f}'} | "
            f"{index.get('openie_path', '')} | "
            f"{index.get('chunk_embedding_store_path', '')} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_dataset(args: argparse.Namespace, dataset: str) -> Mapping[str, Any]:
    data_root = Path(args.data_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    dataset_dir = output_root / dataset
    index_dir = dataset_dir / "index"
    reports_dir = dataset_dir / "reports"
    dataset_path = data_root / f"{dataset}.json"
    corpus_path = data_root / f"{dataset}_corpus.json"
    docs = load_corpus_docs(corpus_path, max_index_docs=int(args.max_index_docs))

    if args.dry_run:
        runtime_embedding_name = normalize_embedding_name_for_runtime(
            embedding_name=str(args.embedding_name),
            embedding_base_url=str(args.embedding_base_url),
        )
        artifacts = fresh_artifact_paths(
            save_dir=index_dir,
            llm_name=str(args.llm_name),
            embedding_name=runtime_embedding_name,
        )
        return {
            "dataset": dataset,
            "row_count": max(int(args.max_queries), 0),
            "metrics": {},
            "dry_run": True,
            "fresh_index": {
                "openie_path": str(artifacts.openie_path),
                "chunk_embedding_store_path": str(artifacts.chunk_embedding_store_path),
                "doc_count": len(docs),
            },
        }

    if bool(args.reuse_current_fresh_index):
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

    report_path = reports_dir / f"{dataset}_minimal_per_query_fresh.json"
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

    embedding_model_id = embedding_endpoint_model_id(
        normalize_embedding_name_for_runtime(
            embedding_name=str(args.embedding_name),
            embedding_base_url=str(args.embedding_base_url),
        )
    )
    if str(args.candidate_generator_mode) == "dense_preserving_sto":
        candidate_generator = DensePreservingAGSTOLocalGraphCandidateGenerator.from_parquet(
            nodes=tuple(nodes),
            embedding_store_path=artifacts.chunk_embedding_store_path,
            embedding_base_url=str(args.embedding_base_url),
            embedding_model_name=embedding_model_id,
            preserved_dense_count=max(int(args.preserved_dense_count), 1),
            dense_seed_count=max(int(args.dense_root_count), 1),
            agsto_textual_seed_top_k=max(int(args.agsto_textual_seed_top_k), 0),
            agsto_max_endpoint_degree=max(int(args.agsto_max_endpoint_degree), 1),
            agsto_closure_hops=max(int(args.agsto_closure_hops), 0),
        )
    else:
        candidate_generator = DenseSeededAGSTOLocalGraphCandidateGenerator.from_parquet(
            nodes=tuple(nodes),
            embedding_store_path=artifacts.chunk_embedding_store_path,
            embedding_base_url=str(args.embedding_base_url),
            embedding_model_name=embedding_model_id,
            dense_seed_count=max(int(args.dense_root_count), 1),
            source_prior_prefix_count=max(int(args.top_k), 1),
            agsto_textual_seed_top_k=max(int(args.agsto_textual_seed_top_k), 0),
            agsto_max_endpoint_degree=max(int(args.agsto_max_endpoint_degree), 1),
            agsto_closure_hops=max(int(args.agsto_closure_hops), 0),
        )
    dataset_result = evaluate_e2e_rows(
        rows=rows,
        nodes=nodes,
        candidate_generator=candidate_generator,
        runner=str(args.runner),
        top_k=max(int(args.top_k), 1),
        candidate_pool_k=max(int(args.candidate_pool_k), 1),
        certificate_policy=str(args.certificate_policy),
        enable_pipeline_candidate_expansion=False,
        enable_pipeline_replacement=True,
        allow_frontier_pair_insertion=False,
        assembly_policy="source_prior_preserving_tail_insertion",
        source_prior_policy="dense_head",
    )

    output_stem = retrieval_output_stem(dataset=dataset, runner=str(args.runner))
    output_json = reports_dir / f"{output_stem}.json"
    output_md = reports_dir / f"{output_stem}.md"

    payload = {
        "method": method_name_for_runner(str(args.runner)),
        "runner": str(args.runner),
        "dataset": dataset,
        "input_report": str(report_path.resolve()),
        "retrieval_output_json": str(output_json.resolve()),
        "retrieval_output_md": str(output_md.resolve()),
        "openie_path": str(artifacts.openie_path.resolve()),
        "source_variant": str(args.variant),
        "fresh_index": {
            "save_dir": str(artifacts.save_dir.resolve()),
            "openie_path": str(artifacts.openie_path.resolve()),
            "chunk_embedding_store_path": str(artifacts.chunk_embedding_store_path.resolve()),
            "llm_cache_dir": str(artifacts.llm_cache_dir.resolve()),
            "working_dir": str(artifacts.working_dir.resolve()),
            "doc_count": int(artifacts.doc_count),
            "index_execution": index_execution,
            "reuses_current_fresh_index": bool(args.reuse_current_fresh_index),
            "reuses_existing_openie_cache": bool(args.reuse_current_fresh_index),
            "reuses_existing_embedding_cache": bool(args.reuse_current_fresh_index),
            "reuses_legacy_openie_cache": False,
            "reuses_legacy_embedding_cache": False,
        },
        "config": {
            "candidate_source": f"fresh_{args.candidate_generator_mode}_query_grounded_sto",
            "candidate_generator_mode": str(args.candidate_generator_mode),
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
            "preserved_dense_count": max(int(args.preserved_dense_count), 1),
            "dense_root_count": max(int(args.dense_root_count), 1),
            "agsto_textual_seed_top_k": max(int(args.agsto_textual_seed_top_k), 0),
            "agsto_max_endpoint_degree": max(int(args.agsto_max_endpoint_degree), 1),
            "agsto_closure_hops": max(int(args.agsto_closure_hops), 0),
            "uses_sfb_or_support_fusion": False,
            "uses_v13b_source_report": False,
            "uses_legacy_compare_report": False,
            "uses_weighted_score_fusion": False,
            "uses_dataset_routing": False,
        },
        **dataset_result,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_retrieval_markdown(payload, output_md)
    logger.info("Wrote retrieval JSON: %s", output_json)
    return payload


def run_optional_qa(
    *,
    args: argparse.Namespace,
    retrieval_payloads: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not args.run_qa or args.dry_run:
        return None
    from evidence_transition_top5_qa_adapter import main as qa_main

    install_openai_client_api_key(base_url=str(args.reader_llm_base_url or ""))
    output_root = Path(args.output_root).expanduser()
    retrieval_jsons = [
        Path(str(payload.get("retrieval_output_json") or "")).expanduser()
        if payload.get("retrieval_output_json")
        else Path(str(payload["input_report"])).parent
        / f"{payload['dataset']}_fresh_query_grounded_sto_retrieval.json"
        for payload in retrieval_payloads
    ]
    qa_method = (
        QUERY_GROUNDED_STO_V2_QA_METHOD
        if str(args.runner) in ETV2_RUNNERS
        else QUERY_GROUNDED_STO_QA_METHOD
    )
    qa_stem = qa_output_stem(runner=str(args.runner))
    qa_input = output_root / "reports" / f"{qa_stem}_input.json"
    export_qa_transition_report(
        input_jsons=retrieval_jsons,
        output_json=qa_input,
        public_method=qa_method,
    )
    qa_json = output_root / "reports" / f"{qa_stem}.json"
    qa_md = output_root / "reports" / f"{qa_stem}.md"
    qa_args = [
        "--transition-report",
        str(qa_input),
        "--datasets",
        ",".join(str(payload["dataset"]) for payload in retrieval_payloads),
        "--methods",
        qa_method,
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


def parse_datasets(value: str) -> List[str]:
    datasets = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = [dataset for dataset in datasets if dataset not in SUPPORTED_DATASETS]
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}; supported={SUPPORTED_DATASETS}")
    return datasets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="2wikimultihopqa,musique,hotpotqa")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument(
        "--max-index-docs",
        type=int,
        default=0,
        help="Smoke-test only. 0 indexes the full corpus.",
    )
    parser.add_argument("--llm-name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-retry-attempts", type=int, default=5)
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    parser.add_argument("--embedding-name", default=DEFAULT_EMBEDDING_NAME)
    parser.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--variant", default="hipporag_v2")
    parser.add_argument(
        "--runner",
        choices=(
            "query_grounded_sto_graph_native",
            "agsto_graph_native",
            "query_grounded_sto_graph_native_v2",
            "agsto_graph_native_v2",
            "evidence_transition_v2",
        ),
        default="query_grounded_sto_graph_native",
    )
    parser.add_argument(
        "--certificate-policy",
        choices=(
            "canonical_source_text",
            "canonical_transition",
            "canonical_query_transition",
            "canonical_fact_origin_transition",
        ),
        default="canonical_source_text",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-k", type=int, default=200)
    parser.add_argument(
        "--candidate-generator-mode",
        choices=("dense_seeded_sto", "dense_preserving_sto"),
        default="dense_seeded_sto",
        help=(
            "dense_seeded_sto uses dense only as graph entry and admits STO closure into "
            "the candidate universe; dense_preserving_sto keeps the previous dense-first "
            "candidate boundary for ablation."
        ),
    )
    parser.add_argument("--preserved-dense-count", type=int, default=200)
    parser.add_argument("--dense-root-count", type=int, default=20)
    parser.add_argument("--agsto-textual-seed-top-k", type=int, default=20)
    parser.add_argument("--agsto-max-endpoint-degree", type=int, default=30)
    parser.add_argument("--agsto-closure-hops", type=int, default=2)
    parser.add_argument("--run-qa", action="store_true")
    parser.add_argument("--reader-llm-name", default=DEFAULT_READER_LLM_NAME)
    parser.add_argument("--reader-llm-base-url", default=DEFAULT_READER_LLM_BASE_URL)
    parser.add_argument("--reader-max-new-tokens", type=int, default=400)
    parser.add_argument("--reader-qwen-disable-thinking", action="store_true")
    parser.add_argument("--overwrite-fresh-index", action="store_true")
    parser.add_argument(
        "--reuse-current-fresh-index",
        action="store_true",
        help=(
            "Skip indexing and reuse artifacts already produced by this runner in "
            "the same output root. Use this only for retrieval/QA iteration; the "
            "default path still rebuilds from raw corpus."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    args = build_arg_parser().parse_args(argv)
    if bool(args.reuse_current_fresh_index) and bool(args.overwrite_fresh_index):
        raise ValueError("--reuse-current-fresh-index cannot be combined with --overwrite-fresh-index")
    datasets = parse_datasets(args.datasets)
    retrieval_payloads = [run_one_dataset(args, dataset) for dataset in datasets]
    qa_payload = run_optional_qa(args=args, retrieval_payloads=retrieval_payloads)
    summary_md = (
        Path(args.output_root).expanduser()
        / "reports"
        / f"{summary_output_stem(runner=str(args.runner))}.md"
    )
    write_summary_markdown(
        output_md=summary_md,
        retrieval_payloads=retrieval_payloads,
        qa_payload=qa_payload,
    )
    compact = {
        str(payload.get("dataset") or ""): {
            "R@5": (payload.get("metrics", {}) or {}).get("r5"),
            "rows": payload.get("row_count"),
            "fresh_openie": (payload.get("fresh_index", {}) or {}).get("openie_path"),
            "fresh_embeddings": (payload.get("fresh_index", {}) or {}).get(
                "chunk_embedding_store_path"
            ),
        }
        for payload in retrieval_payloads
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    print(f"Wrote {summary_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
