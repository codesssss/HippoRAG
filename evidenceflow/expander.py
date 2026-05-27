"""Legacy frozen-ETv3 expander adapter for PCEC parity runs.

Main EvidenceLink results do not use this adapter as their upstream retriever;
they consume ETv4 fact-witnessed STO pools through ``run_native_pool.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evidenceflow.contract import DEFAULT_POOL_K, DEFAULT_READER_BUDGET_K
from evidenceflow.frozen_etv3_variable_flow.run_fresh_e2e import (
    DEFAULT_CANDIDATE_GENERATOR_MODE,
    DEFAULT_RUNNER,
    embedding_endpoint_model_id,
    load_current_fresh_sto_index,
    normalize_embedding_name_for_runtime,
    select_runner,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.build_minimal_per_query_report import (
    build_minimal_report,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.candidate_generator import (
    AGSTOLocalGraphCandidateGenerator,
    DEFAULT_ROLE_GRAPH_EDGE_POLICY,
    DensePreservingAGSTOLocalGraphCandidateGenerator,
    DenseSeededAGSTOLocalGraphCandidateGenerator,
    NO_FACT_WITNESS_ROLE_GRAPH_EDGE_POLICY,
    PHRASE_SOURCE_ONLY_ROLE_GRAPH_EDGE_POLICY,
    normalize_role_graph_edge_policy,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.e2e_pipeline import (
    retrieve_one_e2e,
)
from evidenceflow.frozen_etv3_variable_flow.source_authorized_vocab_strict_retrieval.evaluate_report import (
    load_nodes,
    rows_for_variant,
)
from evidenceflow.pcec_types import PCECQueryState
from export_evidence_transition_pool import (  # noqa: E402
    extract_title,
    get_gold_answers,
    get_gold_docs,
    score_pool_docs,
    unique_ints,
)


DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DEFAULT_FROZEN_ETV3_RUNS_ROOT = Path("run_logs/evidence_transition_graphragv3_variable_flow_qwen8b_nv2_full_20260510/runs")
DEFAULT_LLM_NAME = "qwen3-8b-train"
DEFAULT_EMBEDDING_NAME = "nvidia/NV-Embed-v2"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8019/v1/embeddings"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def load_dataset_rows(dataset: str, data_root: Path, *, limit: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    dataset_path = Path(data_root) / f"{dataset}.json"
    corpus_path = Path(data_root) / f"{dataset}_corpus.json"
    samples = list(read_json(dataset_path))
    if int(limit) > 0:
        samples = samples[: int(limit)]
    corpus = list(read_json(corpus_path))
    docs = [f"{row['title']}\n{row['text']}" for row in corpus]
    return samples, corpus, docs


def passage_for_doc(nodes: Sequence[Any], doc_idx: int) -> str:
    if 0 <= int(doc_idx) < len(nodes):
        return str(getattr(nodes[int(doc_idx)], "text", "") or "")
    return ""


def build_pool_record_from_retrieval(
    *,
    dataset: str,
    query_idx: int,
    sample: Mapping[str, Any],
    retrieval_result: Any,
    nodes: Sequence[Any],
    gold_docs: Sequence[Sequence[str]],
    gold_answers: Sequence[Sequence[str]],
    pool_k: int,
    top_k: int,
) -> dict[str, Any]:
    question = str(sample.get("question") or "")
    trace = dict(getattr(retrieval_result, "trace", {}) or {})
    candidate_universe = dict(trace.get("candidate_universe") or {})
    selected_topk = unique_ints(getattr(retrieval_result, "doc_indices", ()) or [])[: int(top_k)]
    candidate_indices = unique_ints(candidate_universe.get("candidate_doc_indices") or [])
    if not candidate_indices:
        candidate_indices = unique_ints(candidate_universe.get("admissible_doc_indices") or [])
    pool_doc_indices = [
        idx
        for idx in unique_ints([*selected_topk, *candidate_indices])
        if passage_for_doc(nodes, idx)
    ][: int(pool_k)]
    pool_docs = [passage_for_doc(nodes, idx) for idx in pool_doc_indices]
    agsto_trace = {
        "source": "evidence_transition",
        "selected_doc_indices": selected_topk,
        "candidate_doc_indices": candidate_indices[: int(pool_k)],
        "candidate_count": int(candidate_universe.get("candidate_count", len(candidate_indices)) or 0),
        "candidate_order_policy": candidate_universe.get("candidate_order_policy"),
        "candidate_source": candidate_universe.get("candidate_source"),
        "source_prior_prefix_doc_indices": unique_ints(candidate_universe.get("source_prior_prefix_doc_indices") or []),
        "agsto_seed_doc_indices": unique_ints(candidate_universe.get("agsto_seed_doc_indices") or []),
        "agsto_symbolic_seed_doc_indices": unique_ints(candidate_universe.get("agsto_symbolic_seed_doc_indices") or []),
        "agsto_graph_tail_doc_indices": unique_ints(candidate_universe.get("agsto_graph_tail_doc_indices") or [])[: int(pool_k)],
        "agsto_local_edge_count": int(candidate_universe.get("agsto_local_edge_count", 0) or 0),
        "agsto_role_graph_edge_count": int(candidate_universe.get("agsto_role_graph_edge_count", 0) or 0),
        "external_pool_doc_ids": list(pool_doc_indices),
        "external_pool_titles": [extract_title(doc) for doc in pool_docs],
        "external_pool_scores": score_pool_docs(pool_doc_indices, selected_doc_indices=selected_topk),
        "fresh_expander_trace": trace,
    }
    return {
        "query_idx": int(query_idx),
        "question": question,
        "gold_answers": list(gold_answers[int(query_idx)]),
        "gold_docs": list(gold_docs[int(query_idx)]),
        "gold_titles": [extract_title(doc) for doc in gold_docs[int(query_idx)]],
        "pool_k": int(pool_k),
        "pool_docs": pool_docs,
        "pool_titles": [extract_title(doc) for doc in pool_docs],
        "pool_doc_scores": score_pool_docs(pool_doc_indices, selected_doc_indices=selected_topk),
        "pool_doc_ids": pool_doc_indices,
        "agsto": agsto_trace,
    }


@dataclass
class FrozenETv3Expander:
    """Library adapter around the frozen ETv3 variable-flow expander."""

    dataset: str
    candidate_pool_k: int = DEFAULT_POOL_K
    reader_budget_k: int = DEFAULT_READER_BUDGET_K
    et_candidate_pool_k: int = 200
    data_root: Path = DEFAULT_DATA_ROOT
    run_root: Path | None = None
    llm_name: str = DEFAULT_LLM_NAME
    embedding_name: str = DEFAULT_EMBEDDING_NAME
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    variant: str = "hipporag_v2"
    candidate_generator_mode: str = DEFAULT_CANDIDATE_GENERATOR_MODE
    readout_policy: str = "source_aligned"
    dense_root_count: int = 20
    preserved_dense_count: int = 200
    agsto_textual_seed_top_k: int = 20
    agsto_max_endpoint_degree: int = 30
    agsto_closure_hops: int = 2
    enable_variable_flow_traversal: bool = True
    role_graph_edge_policy: str = DEFAULT_ROLE_GRAPH_EDGE_POLICY

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.run_root = Path(self.run_root) if self.run_root is not None else DEFAULT_FROZEN_ETV3_RUNS_ROOT / str(self.dataset)
        self.dataset_dir = Path(self.run_root) / str(self.dataset)
        self.index_dir = self.dataset_dir / "index"
        self.reports_dir = self.dataset_dir / "reports"
        self.dataset_path = self.data_root / f"{self.dataset}.json"
        self.corpus_path = self.data_root / f"{self.dataset}_corpus.json"
        self.samples, self.corpus, self.docs = load_dataset_rows(str(self.dataset), self.data_root)
        self.gold_docs = get_gold_docs(self.samples, str(self.dataset))
        self.gold_answers = get_gold_answers(self.samples)
        self.artifacts = load_current_fresh_sto_index(
            docs=self.docs,
            save_dir=self.index_dir,
            llm_name=str(self.llm_name),
            embedding_name=str(self.embedding_name),
            embedding_base_url=str(self.embedding_base_url),
        )
        self.nodes = load_nodes(Path(self.artifacts.openie_path))
        self.minimal_report_path = self.reports_dir / f"{self.dataset}_minimal_per_query_fresh_v3_variable_flow.json"
        if self.minimal_report_path.exists():
            self.minimal_report = read_json(self.minimal_report_path)
        else:
            self.minimal_report = build_minimal_report(
                dataset=str(self.dataset),
                dataset_path=self.dataset_path,
                openie_path=Path(self.artifacts.openie_path),
                output_json=self.minimal_report_path,
            )
        self.rows = rows_for_variant(self.minimal_report, str(self.variant))
        self.runner = select_runner(
            candidate_mode=str(self.candidate_generator_mode),
            readout_policy=str(self.readout_policy),
        )
        self.role_graph_edge_policy = normalize_role_graph_edge_policy(self.role_graph_edge_policy)
        self.candidate_generator = self._build_candidate_generator()
        self._prepared_count = 0

    def _build_candidate_generator(self) -> Any:
        mode = str(self.candidate_generator_mode)
        if mode == "graph_only":
            return AGSTOLocalGraphCandidateGenerator.build(
                nodes=tuple(self.nodes),
                textual_seed_top_k=max(int(self.agsto_textual_seed_top_k), 0),
                max_endpoint_degree=max(int(self.agsto_max_endpoint_degree), 1),
                closure_hops=max(int(self.agsto_closure_hops), 0),
                enable_query_supported_same_object_handoff=False,
                enable_variable_flow_traversal=bool(self.enable_variable_flow_traversal),
                role_graph_edge_policy=str(self.role_graph_edge_policy),
            )
        chunk_embedding_store_path = Path(getattr(self.artifacts, "chunk_embedding_store_path", ""))
        embedding_model_id = embedding_endpoint_model_id(
            normalize_embedding_name_for_runtime(
                embedding_name=str(self.embedding_name),
                embedding_base_url=str(self.embedding_base_url),
            )
        )
        if mode == "dense_preserving_sto":
            return DensePreservingAGSTOLocalGraphCandidateGenerator.from_parquet(
                nodes=tuple(self.nodes),
                embedding_store_path=chunk_embedding_store_path,
                embedding_base_url=str(self.embedding_base_url),
                embedding_model_name=embedding_model_id,
                preserved_dense_count=max(int(self.preserved_dense_count), 1),
                dense_seed_count=max(int(self.dense_root_count), 1),
                agsto_textual_seed_top_k=max(int(self.agsto_textual_seed_top_k), 0),
                agsto_max_endpoint_degree=max(int(self.agsto_max_endpoint_degree), 1),
                agsto_closure_hops=max(int(self.agsto_closure_hops), 0),
                enable_query_supported_same_object_handoff=False,
                enable_variable_flow_traversal=bool(self.enable_variable_flow_traversal),
                role_graph_edge_policy=str(self.role_graph_edge_policy),
            )
        return DenseSeededAGSTOLocalGraphCandidateGenerator.from_parquet(
            nodes=tuple(self.nodes),
            embedding_store_path=chunk_embedding_store_path,
            embedding_base_url=str(self.embedding_base_url),
            embedding_model_name=embedding_model_id,
            dense_seed_count=max(int(self.dense_root_count), 1),
            source_prior_prefix_count=max(int(self.reader_budget_k), 1),
            agsto_textual_seed_top_k=max(int(self.agsto_textual_seed_top_k), 0),
            agsto_max_endpoint_degree=max(int(self.agsto_max_endpoint_degree), 1),
            agsto_closure_hops=max(int(self.agsto_closure_hops), 0),
            enable_query_supported_same_object_handoff=False,
            enable_variable_flow_traversal=bool(self.enable_variable_flow_traversal),
            role_graph_edge_policy=str(self.role_graph_edge_policy),
        )

    def prepare_queries(self, *, max_queries: int = 0) -> None:
        rows = self.rows[: int(max_queries)] if int(max_queries) > 0 else self.rows
        prepare = getattr(self.candidate_generator, "prepare_queries", None)
        if callable(prepare):
            prepare([str(row.get("question") or row.get("query") or "") for row in rows])
            self._prepared_count = len(rows)

    def expand_query_record(self, query_idx: int, question: str | None = None) -> dict[str, Any]:
        row = dict(self.rows[int(query_idx)])
        query = str(question if question is not None else row.get("question") or row.get("query") or "")
        result = retrieve_one_e2e(
            query=query,
            query_index=int(query_idx),
            row=row,
            nodes=self.nodes,
            candidate_generator=self.candidate_generator,
            runner=self.runner,
            top_k=int(self.reader_budget_k),
            candidate_pool_k=int(self.et_candidate_pool_k),
            certificate_policy="canonical_source_text",
            enable_pipeline_candidate_expansion=False,
            enable_pipeline_replacement=False,
            allow_frontier_pair_insertion=False,
            assembly_policy="baseline_aligned_graph_entry_no_weighted_fusion",
            source_prior_policy="dense_entry_source_prior" if str(self.candidate_generator_mode) != "graph_only" else "none",
        )
        return build_pool_record_from_retrieval(
            dataset=str(self.dataset),
            query_idx=int(query_idx),
            sample=self.samples[int(query_idx)],
            retrieval_result=result,
            nodes=self.nodes,
            gold_docs=self.gold_docs,
            gold_answers=self.gold_answers,
            pool_k=int(self.candidate_pool_k),
            top_k=int(self.reader_budget_k),
        )

    def expand_query(self, query_idx: int, question: str | None = None) -> PCECQueryState:
        return PCECQueryState.from_pool_record(
            self.expand_query_record(query_idx=query_idx, question=question),
            dataset=str(self.dataset),
            reader_budget_k=int(self.reader_budget_k),
            prefix_budget_m=4,
        )

    def expand_dataset_records(self, *, max_queries: int = 0) -> list[dict[str, Any]]:
        limit = len(self.samples) if int(max_queries) <= 0 else min(int(max_queries), len(self.samples))
        self.prepare_queries(max_queries=limit)
        return [self.expand_query_record(idx) for idx in range(limit)]

    def expand_dataset(self, samples: Sequence[Mapping[str, Any]] | None = None, *, max_queries: int = 0) -> list[PCECQueryState]:
        limit = len(samples) if samples is not None else (int(max_queries) if int(max_queries) > 0 else len(self.samples))
        return [
            PCECQueryState.from_pool_record(
                record,
                dataset=str(self.dataset),
                reader_budget_k=int(self.reader_budget_k),
                prefix_budget_m=4,
            )
            for record in self.expand_dataset_records(max_queries=limit)
        ]

    def metadata(self) -> dict[str, Any]:
        first_sample = self.samples[0] if self.samples else {}
        first_doc = self.docs[0] if self.docs else ""
        return {
            "dataset": str(self.dataset),
            "candidate_pool_k": int(self.candidate_pool_k),
            "et_candidate_pool_k": int(self.et_candidate_pool_k),
            "reader_budget_k": int(self.reader_budget_k),
            "run_root": str(self.run_root),
            "dataset_path": str(self.dataset_path),
            "corpus_path": str(self.corpus_path),
            "sample_count": int(len(self.samples)),
            "corpus_doc_count": int(len(self.docs)),
            "first_sample_hash": stable_hash(first_sample),
            "first_corpus_doc_hash": stable_hash(first_doc),
            "openie_path": str(Path(self.artifacts.openie_path)),
            "chunk_embedding_store_path": str(getattr(self.artifacts, "chunk_embedding_store_path", "") or ""),
            "candidate_generator_mode": str(self.candidate_generator_mode),
            "runner": str(self.runner or DEFAULT_RUNNER),
            "variant": str(self.variant),
            "readout_policy": str(self.readout_policy),
            "enable_variable_flow_traversal": bool(self.enable_variable_flow_traversal),
            "role_graph_edge_policy": str(self.role_graph_edge_policy),
            "ablation_no_fact_witness": str(self.role_graph_edge_policy) == NO_FACT_WITNESS_ROLE_GRAPH_EDGE_POLICY,
            "ablation_phrase_source_only": str(self.role_graph_edge_policy) == PHRASE_SOURCE_ONLY_ROLE_GRAPH_EDGE_POLICY,
            "prepared_query_count": int(self._prepared_count),
        }
