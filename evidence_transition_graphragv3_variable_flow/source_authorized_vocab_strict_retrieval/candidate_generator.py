"""Candidate-universe generators for end-to-end STO GraphRAG.

The generator boundary is intentionally separate from graph retrieval.  A
candidate generator may expose a local universe, but it must not decide the
final reader top-k and must not provide weighted score fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from collections import deque
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .candidate_expansion import (
    SourceTextCandidateExpansionIndex,
    build_source_text_candidate_expansion_index,
)
from .certificate_graph import EvidenceNode
from .normalize import (
    is_specific_title_alias,
    normalize_text,
    phrase_occurs,
    title_aliases,
    tokens,
    unique_ints,
)
from .query_mentions import query_mentioned_doc_indices


CANDIDATE_UNIVERSE_CONTRACT: Mapping[str, bool | str] = {
    "candidate_boundary": "method_owned_candidate_universe",
    "dense_topk_is_final_answer_default": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_external_baseline_frontier": False,
}


def prefix_queries_with_instruction(queries: Sequence[str], instruction: str) -> List[str]:
    if os.environ.get("HIPPO_DISABLE_QUERY_INSTRUCTION_PREFIX") == "1":
        return [str(query) for query in queries]
    if not instruction:
        return [str(query) for query in queries]
    prefix = f"Instruct: {instruction}\nQuery: "
    return [prefix + str(query) for query in queries]


@dataclass(frozen=True)
class CandidateUniverse:
    """A query-local candidate universe before graph construction."""

    doc_indices: Tuple[int, ...]
    trace: Mapping[str, object]
    admissible_doc_indices: Tuple[int, ...] = ()
    graph_payload: Mapping[str, object] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


class CandidateUniverseGenerator(Protocol):
    """Generate a local candidate universe for one query row."""

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        ...


def load_candidate_cache_doc_indices(cache_path: Path) -> Dict[int, List[int]]:
    """Load HippoRAG-style cached candidate doc indices keyed by query index."""

    payload = json.loads(cache_path.expanduser().read_text(encoding="utf-8"))
    query_solutions = payload.get("query_solutions", []) if isinstance(payload, Mapping) else []
    output: Dict[int, List[int]] = {}
    for index, row in enumerate(query_solutions or []):
        if not isinstance(row, Mapping):
            continue
        output[int(index)] = unique_ints(row.get("doc_indices", []) or [])
    return output


@dataclass(frozen=True)
class ReplayCacheCandidateGenerator:
    """Replay a stored candidate universe for parity and ablation runs."""

    candidates_by_query_index: Mapping[int, Sequence[int]]
    name: str = "replay_candidate_cache"

    @classmethod
    def from_path(cls, cache_path: Path) -> "ReplayCacheCandidateGenerator":
        return cls(
            candidates_by_query_index=load_candidate_cache_doc_indices(cache_path),
            name=f"replay_candidate_cache:{cache_path.expanduser()}",
        )

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        docs = unique_ints(self.candidates_by_query_index.get(int(query_index), ()))[: max(int(top_n), 1)]
        return CandidateUniverse(
            doc_indices=tuple(docs),
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": self.name,
                "candidate_source": "replay_cache",
                "query_index": int(query_index),
                "candidate_count": len(docs),
            },
        )


@dataclass
class HippoRAGV2RuntimeCandidateGenerator:
    """Generate the candidate universe by running HippoRAGv2 retrieval.

    This is the replacement for replaying ``baseline_retrieval_cache``.  It uses
    HippoRAGv2 only as the first-stage graph candidate generator; the final
    reader top-k is still produced by the STO pipeline.
    """

    nodes: Tuple[EvidenceNode, ...]
    save_dir: str
    dataset: str
    llm_base_url: str
    llm_name: str
    embedding_base_url: str
    embedding_model_name: str
    retrieval_top_k: int = 200
    linking_top_k: int = 5
    embedding_batch_size: int = 256
    max_new_tokens: Optional[int] = None
    openie_mode: str = "online"
    qwen_disable_thinking: bool = False
    rerank_dspy_file_path: str = "src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json"
    retriever_factory: Optional[Callable[[], Any]] = field(default=None, repr=False, compare=False)
    _retriever: Any = field(default=None, init=False, repr=False, compare=False)
    _candidates_by_query_index: Dict[int, Tuple[int, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _candidates_by_query_text: Dict[str, Tuple[int, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _unresolved_doc_count_by_query_index: Dict[int, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def prepare_queries(self, queries: Sequence[str]) -> None:
        """Batch-run HippoRAGv2 once before row-wise STO retrieval."""

        query_list = [str(query) for query in queries]
        missing: List[Tuple[int, str]] = []
        for index, query in enumerate(query_list):
            if index in self._candidates_by_query_index:
                continue
            if query in self._candidates_by_query_text:
                self._candidates_by_query_index[index] = self._candidates_by_query_text[query]
                continue
            missing.append((index, query))
        if not missing:
            return

        retriever = self._ensure_retriever()
        solutions = retriever.retrieve(
            queries=[query for _index, query in missing],
            num_to_retrieve=max(int(self.retrieval_top_k), 1),
        )
        if isinstance(solutions, tuple):
            solutions = solutions[0]
        if len(solutions) != len(missing):
            raise ValueError(
                "HippoRAGv2 runtime retriever returned unexpected solution count: "
                f"expected={len(missing)}, actual={len(solutions)}"
            )
        for (query_index, query), solution in zip(missing, solutions):
            doc_indices, unresolved_count = self._solution_doc_indices(solution)
            self._candidates_by_query_index[int(query_index)] = tuple(doc_indices)
            self._candidates_by_query_text[str(query)] = tuple(doc_indices)
            self._unresolved_doc_count_by_query_index[int(query_index)] = int(unresolved_count)

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        docs = self._candidates_by_query_index.get(int(query_index))
        resolution = "prepared_query_index"
        if docs is None:
            docs = self._candidates_by_query_text.get(str(query))
            resolution = "prepared_query_text"
        if docs is None:
            self.prepare_queries([str(query)])
            docs = self._candidates_by_query_text.get(str(query), ())
            resolution = "single_query_runtime_retrieve"
        docs = tuple(unique_ints(docs)[: max(int(top_n), 1)])
        unresolved_count = self._unresolved_doc_count_by_query_index.get(int(query_index), 0)
        return CandidateUniverse(
            doc_indices=docs,
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": "hipporag_v2_runtime_candidate_generator",
                "candidate_source": "method_owned_hipporag_v2_runtime_graph_retrieval",
                "query_index": int(query_index),
                "candidate_count": len(docs),
                "candidate_resolution": resolution,
                "hipporag_save_dir": str(self.save_dir),
                "hipporag_dataset": str(self.dataset),
                "hipporag_llm_name": str(self.llm_name),
                "hipporag_embedding_model_name": self._normalized_embedding_model_name(),
                "hipporag_retrieval_top_k": int(self.retrieval_top_k),
                "hipporag_linking_top_k": int(self.linking_top_k),
                "hipporag_openie_mode": str(self.openie_mode),
                "unresolved_runtime_doc_count": int(unresolved_count),
                "uses_external_baseline_frontier": False,
                "uses_replay_candidate_cache": False,
                "uses_hipporag_v2_runtime_graph_retrieval": True,
                "uses_hipporag_v2_internal_ppr": True,
                "dense_topk_is_final_answer_default": False,
            },
        )

    def _ensure_retriever(self) -> Any:
        if self._retriever is not None:
            return self._retriever
        if self.retriever_factory is not None:
            self._retriever = self.retriever_factory()
            return self._retriever

        from src.hipporag.HippoRAG import HippoRAG
        from src.hipporag.utils.config_utils import BaseConfig

        config_field_names = {field_info.name for field_info in fields(BaseConfig)}
        config_kwargs: Dict[str, Any] = {
            "save_dir": str(self.save_dir),
            "llm_base_url": str(self.llm_base_url),
            "llm_name": str(self.llm_name),
            "embedding_model_name": self._normalized_embedding_model_name(),
            "embedding_base_url": str(self.embedding_base_url),
            "max_new_tokens": self.max_new_tokens,
            "force_index_from_scratch": False,
            "force_openie_from_scratch": False,
            "dataset": str(self.dataset),
            "openie_mode": str(self.openie_mode),
            "retrieval_mode": "hipporag_v2",
            "retrieval_top_k": max(int(self.retrieval_top_k), 1),
            "linking_top_k": max(int(self.linking_top_k), 1),
            "qa_top_k": 5,
            "max_qa_steps": 3,
            "graph_type": "facts_and_sim_passage_node_unidirectional",
            "embedding_batch_size": max(int(self.embedding_batch_size), 1),
            "corpus_len": len(self.nodes),
            "qwen_disable_thinking": bool(self.qwen_disable_thinking),
            "rerank_dspy_file_path": str(self.rerank_dspy_file_path),
        }
        config = BaseConfig(
            **{
                key: value
                for key, value in config_kwargs.items()
                if key in config_field_names
            }
        )
        if "qwen_disable_thinking" not in config_field_names:
            setattr(config, "qwen_disable_thinking", bool(self.qwen_disable_thinking))
        self._retriever = HippoRAG(global_config=config)
        return self._retriever

    def _normalized_embedding_model_name(self) -> str:
        name = str(self.embedding_model_name)
        if not self.embedding_base_url:
            return name
        if name.startswith(("VLLM/", "Transformers/")) or "text-embedding" in name:
            return name
        return f"VLLM/{name}"

    def _solution_doc_indices(self, solution: Any) -> Tuple[Tuple[int, ...], int]:
        exact_text_to_doc_index = {str(node.text): int(node.doc_index) for node in self.nodes}
        compact_text_to_doc_index = {
            _compact_doc_text(node.text): int(node.doc_index)
            for node in self.nodes
        }
        doc_indices: List[int] = []
        unresolved_count = 0
        for doc in getattr(solution, "docs", []) or []:
            doc_text = str(doc)
            doc_index = exact_text_to_doc_index.get(doc_text)
            if doc_index is None:
                doc_index = compact_text_to_doc_index.get(_compact_doc_text(doc_text))
            if doc_index is None:
                unresolved_count += 1
                continue
            doc_indices.append(int(doc_index))
        return tuple(unique_ints(doc_indices)), int(unresolved_count)


@dataclass(frozen=True)
class AGSTOLocalGraphCandidateGenerator:
    """Generate candidates by query-local admission over the STO role graph.

    This is the method-owned B candidate boundary: corpus OpenIE/STO units are
    indexed once; each query enters through textual and endpoint seeds, then
    admits a finite local graph by STO role-transition edges.
    """

    nodes: Tuple[EvidenceNode, ...]
    corpus_index: Mapping[str, Any]
    role_graph: Mapping[str, Any]
    textual_seed_top_k: int = 20
    max_endpoint_degree: int = 30
    closure_hops: int = 2
    enable_query_supported_same_object_handoff: bool = False
    enable_variable_flow_traversal: bool = False

    @classmethod
    def build(
        cls,
        nodes: Sequence[EvidenceNode],
        *,
        textual_seed_top_k: int = 20,
        max_endpoint_degree: int = 30,
        closure_hops: int = 2,
        enable_query_supported_same_object_handoff: bool = False,
        enable_variable_flow_traversal: bool = False,
    ) -> "AGSTOLocalGraphCandidateGenerator":
        from evidence_transition_graphragv3_variable_flow.agsto.index import (
            build_corpus_unit_index,
        )
        from evidence_transition_graphragv3_variable_flow.agsto.role_transition import (
            build_role_transition_graph,
        )

        node_tuple = tuple(nodes)
        openie_docs = [
            {
                "idx": str(node.doc_index),
                "passage": str(node.text),
                "title": str(node.display_title),
                "extracted_triples": [list(triple) for triple in node.triples],
            }
            for node in node_tuple
        ]
        corpus_index = build_corpus_unit_index(openie_docs)
        role_graph = build_role_transition_graph(
            corpus_index=corpus_index,
            max_endpoint_degree=max(int(max_endpoint_degree), 1),
            include_title_role_grounding=True,
        )
        return cls(
            nodes=node_tuple,
            corpus_index=corpus_index,
            role_graph=role_graph,
            textual_seed_top_k=max(int(textual_seed_top_k), 0),
            max_endpoint_degree=max(int(max_endpoint_degree), 1),
            closure_hops=max(int(closure_hops), 0),
            enable_query_supported_same_object_handoff=bool(enable_query_supported_same_object_handoff),
            enable_variable_flow_traversal=bool(enable_variable_flow_traversal),
        )

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        from evidence_transition_graphragv3_variable_flow.agsto.local_graph import (
            build_query_local_sto_graph,
        )

        local_graph = build_query_local_sto_graph(
            query=str(query),
            corpus_index=self.corpus_index,
            role_graph=self.role_graph,
            textual_seed_doc_indices=None,
            textual_seed_top_k=max(int(self.textual_seed_top_k), 0),
            max_endpoint_degree=max(int(self.max_endpoint_degree), 1),
            closure_hops=max(int(self.closure_hops), 0),
            candidate_limit=max(int(top_n), 1),
            enable_query_supported_same_object_handoff=bool(self.enable_query_supported_same_object_handoff),
            enable_variable_flow_traversal=bool(self.enable_variable_flow_traversal),
        )
        docs = tuple(unique_ints(local_graph.get("admitted_doc_indices", []) or [])[: max(int(top_n), 1)])
        stats = dict(local_graph.get("stats", {}) or {})
        return CandidateUniverse(
            doc_indices=docs,
            graph_payload={
                "agsto_corpus_index": self.corpus_index,
                "agsto_role_graph": self.role_graph,
                "agsto_local_graph": local_graph,
            },
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": "agsto_local_graph_candidate_generator",
                "candidate_source": "method_owned_agsto_local_graph",
                "query_index": int(query_index),
                "candidate_count": len(docs),
                "textual_seed_top_k": int(self.textual_seed_top_k),
                "max_endpoint_degree": int(self.max_endpoint_degree),
                "closure_hops": int(self.closure_hops),
                "role_graph_edge_count": int(stats.get("role_graph_edge_count", 0) or 0),
                "local_edge_count": int(stats.get("local_edge_count", 0) or 0),
                "seed_doc_indices": tuple(unique_ints(local_graph.get("seed_doc_indices", []) or [])),
                "symbolic_anchor_endpoints": tuple(
                    str(endpoint) for endpoint in local_graph.get("symbolic_anchor_endpoints", []) or []
                ),
                "symbolic_seed_doc_indices": tuple(
                    unique_ints(local_graph.get("symbolic_seed_doc_indices", []) or [])
                ),
                "textual_seed_doc_indices": tuple(
                    unique_ints(local_graph.get("textual_seed_doc_indices", []) or [])
                ),
                "dense_topk_is_final_answer_default": False,
                "uses_weighted_score_fusion": False,
                "uses_external_baseline_frontier": False,
                "uses_replay_candidate_cache": False,
                "uses_hipporag_v2_runtime_graph_retrieval": False,
                "uses_sto_role_graph_admission": True,
                "ablation_query_supported_same_object_handoff": bool(
                    self.enable_query_supported_same_object_handoff
                ),
                "uses_variable_flow_traversal": bool(self.enable_variable_flow_traversal),
            },
        )


@dataclass(frozen=True)
class DenseSeededAGSTOLocalGraphCandidateGenerator:
    """Use dense retrieval only as STO graph-entry seeds.

    Candidate order is deterministic:
    dense reader head -> STO-admitted graph tail -> remaining dense candidates.
    No score fusion is performed after the graph admission step.
    """

    dense_generator: DenseEmbeddingCandidateGenerator
    agsto_generator: AGSTOLocalGraphCandidateGenerator
    dense_seed_count: int = 20
    source_prior_prefix_count: int = 5

    @classmethod
    def from_parquet(
        cls,
        *,
        nodes: Sequence[EvidenceNode],
        embedding_store_path: Path,
        embedding_base_url: str,
        embedding_model_name: str,
        dense_seed_count: int = 20,
        source_prior_prefix_count: int = 5,
        agsto_textual_seed_top_k: int = 20,
        agsto_max_endpoint_degree: int = 30,
        agsto_closure_hops: int = 2,
        enable_query_supported_same_object_handoff: bool = False,
        enable_variable_flow_traversal: bool = False,
    ) -> "DenseSeededAGSTOLocalGraphCandidateGenerator":
        return cls(
            dense_generator=DenseEmbeddingCandidateGenerator.from_parquet(
                nodes=tuple(nodes),
                embedding_store_path=embedding_store_path,
                embedding_base_url=str(embedding_base_url),
                embedding_model_name=str(embedding_model_name),
            ),
            agsto_generator=AGSTOLocalGraphCandidateGenerator.build(
                nodes=tuple(nodes),
                textual_seed_top_k=max(int(agsto_textual_seed_top_k), 0),
                max_endpoint_degree=max(int(agsto_max_endpoint_degree), 1),
                closure_hops=max(int(agsto_closure_hops), 0),
                enable_query_supported_same_object_handoff=bool(
                    enable_query_supported_same_object_handoff
                ),
                enable_variable_flow_traversal=bool(enable_variable_flow_traversal),
            ),
            dense_seed_count=max(int(dense_seed_count), 1),
            source_prior_prefix_count=max(int(source_prior_prefix_count), 1),
        )

    def prepare_queries(self, queries: Sequence[str]) -> None:
        prepare = getattr(self.dense_generator, "prepare_queries", None)
        if callable(prepare):
            prepare([str(query) for query in queries])

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        dense_universe = self.dense_generator.generate(
            query=str(query),
            query_index=int(query_index),
            row=row,
            top_n=max(int(top_n), int(self.dense_seed_count), int(self.source_prior_prefix_count)),
        )
        from evidence_transition_graphragv3_variable_flow.agsto.local_graph import (
            build_query_local_sto_graph,
        )

        local_graph = build_query_local_sto_graph(
            query=str(query),
            corpus_index=self.agsto_generator.corpus_index,
            role_graph=self.agsto_generator.role_graph,
            textual_seed_doc_indices=dense_universe.doc_indices,
            textual_seed_top_k=max(int(self.dense_seed_count), 1),
            max_endpoint_degree=max(int(self.agsto_generator.max_endpoint_degree), 1),
            closure_hops=max(int(self.agsto_generator.closure_hops), 0),
            candidate_limit=max(int(top_n), 1),
            enable_query_supported_same_object_handoff=bool(
                self.agsto_generator.enable_query_supported_same_object_handoff
            ),
            enable_variable_flow_traversal=bool(self.agsto_generator.enable_variable_flow_traversal),
        )
        admitted = tuple(unique_ints(local_graph.get("admitted_doc_indices", []) or []))
        prefix_count = min(max(int(self.source_prior_prefix_count), 1), max(int(top_n), 1))
        source_prior_prefix = tuple(dense_universe.doc_indices[:prefix_count])
        graph_tail = tuple(doc_index for doc_index in admitted if int(doc_index) not in set(source_prior_prefix))
        docs = tuple(
            unique_ints([*source_prior_prefix, *graph_tail, *dense_universe.doc_indices])[: max(int(top_n), 1)]
        )
        stats = dict(local_graph.get("stats", {}) or {})
        return CandidateUniverse(
            doc_indices=docs,
            admissible_doc_indices=tuple(
                doc_index for doc_index in graph_tail if int(doc_index) in set(docs)
            ),
            graph_payload={
                "agsto_corpus_index": self.agsto_generator.corpus_index,
                "agsto_role_graph": self.agsto_generator.role_graph,
                "agsto_local_graph": local_graph,
            },
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": "dense_seeded_agsto_local_graph_candidate_generator",
                "candidate_source": "method_owned_dense_seeded_agsto_local_graph",
                "query_index": int(query_index),
                "candidate_count": len(docs),
                "dense_seed_count": int(self.dense_seed_count),
                "source_prior_prefix_count": int(self.source_prior_prefix_count),
                "source_prior_prefix_doc_indices": source_prior_prefix,
                "agsto_admitted_doc_count": len(admitted),
                "agsto_local_edge_count": int(stats.get("local_edge_count", 0) or 0),
                "agsto_role_graph_edge_count": int(stats.get("role_graph_edge_count", 0) or 0),
                "agsto_seed_doc_indices": tuple(unique_ints(local_graph.get("seed_doc_indices", []) or [])),
                "agsto_symbolic_seed_doc_indices": tuple(
                    unique_ints(local_graph.get("symbolic_seed_doc_indices", []) or [])
                ),
                "agsto_graph_tail_doc_indices": graph_tail[: max(int(top_n), 1)],
                "dense_trace": dict(dense_universe.trace),
                "candidate_order_policy": "dense_head_then_agsto_admitted_tail_then_remaining_dense",
                "dense_topk_is_final_answer_default": False,
                "uses_weighted_score_fusion": False,
                "uses_external_baseline_frontier": False,
                "uses_replay_candidate_cache": False,
                "uses_hipporag_v2_runtime_graph_retrieval": False,
                "uses_dense_only_as_graph_entry": True,
                "uses_sto_role_graph_admission": True,
                "ablation_query_supported_same_object_handoff": bool(
                    self.agsto_generator.enable_query_supported_same_object_handoff
                ),
                "uses_variable_flow_traversal": bool(
                    self.agsto_generator.enable_variable_flow_traversal
                ),
            },
        )


@dataclass(frozen=True)
class DensePreservingAGSTOLocalGraphCandidateGenerator:
    """Preserve the dense candidate universe, then append STO graph expansion.

    This fixes the failure mode where graph admission displaces dense-recalled
    gold documents from the finite candidate pool.  Dense is still only the
    first-stage local universe; the STO graph can add candidates beyond that
    universe when the candidate budget is larger than the preserved dense
    prefix.
    """

    dense_generator: DenseEmbeddingCandidateGenerator
    agsto_generator: AGSTOLocalGraphCandidateGenerator
    preserved_dense_count: int = 200
    dense_seed_count: int = 20

    @classmethod
    def from_parquet(
        cls,
        *,
        nodes: Sequence[EvidenceNode],
        embedding_store_path: Path,
        embedding_base_url: str,
        embedding_model_name: str,
        preserved_dense_count: int = 200,
        dense_seed_count: int = 20,
        agsto_textual_seed_top_k: int = 20,
        agsto_max_endpoint_degree: int = 30,
        agsto_closure_hops: int = 2,
        enable_query_supported_same_object_handoff: bool = False,
        enable_variable_flow_traversal: bool = False,
    ) -> "DensePreservingAGSTOLocalGraphCandidateGenerator":
        return cls(
            dense_generator=DenseEmbeddingCandidateGenerator.from_parquet(
                nodes=tuple(nodes),
                embedding_store_path=embedding_store_path,
                embedding_base_url=str(embedding_base_url),
                embedding_model_name=str(embedding_model_name),
            ),
            agsto_generator=AGSTOLocalGraphCandidateGenerator.build(
                nodes=tuple(nodes),
                textual_seed_top_k=max(int(agsto_textual_seed_top_k), 0),
                max_endpoint_degree=max(int(agsto_max_endpoint_degree), 1),
                closure_hops=max(int(agsto_closure_hops), 0),
                enable_query_supported_same_object_handoff=bool(
                    enable_query_supported_same_object_handoff
                ),
                enable_variable_flow_traversal=bool(enable_variable_flow_traversal),
            ),
            preserved_dense_count=max(int(preserved_dense_count), 1),
            dense_seed_count=max(int(dense_seed_count), 1),
        )

    def prepare_queries(self, queries: Sequence[str]) -> None:
        prepare = getattr(self.dense_generator, "prepare_queries", None)
        if callable(prepare):
            prepare([str(query) for query in queries])

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        preserved_count = min(max(int(self.preserved_dense_count), 1), max(int(top_n), 1))
        dense_universe = self.dense_generator.generate(
            query=str(query),
            query_index=int(query_index),
            row=row,
            top_n=preserved_count,
        )
        from evidence_transition_graphragv3_variable_flow.agsto.local_graph import (
            build_query_local_sto_graph,
        )

        local_graph = build_query_local_sto_graph(
            query=str(query),
            corpus_index=self.agsto_generator.corpus_index,
            role_graph=self.agsto_generator.role_graph,
            textual_seed_doc_indices=dense_universe.doc_indices,
            textual_seed_top_k=max(int(self.dense_seed_count), 1),
            max_endpoint_degree=max(int(self.agsto_generator.max_endpoint_degree), 1),
            closure_hops=max(int(self.agsto_generator.closure_hops), 0),
            candidate_limit=max(int(top_n), 1),
            enable_query_supported_same_object_handoff=bool(
                self.agsto_generator.enable_query_supported_same_object_handoff
            ),
            enable_variable_flow_traversal=bool(self.agsto_generator.enable_variable_flow_traversal),
        )
        admitted = tuple(unique_ints(local_graph.get("admitted_doc_indices", []) or []))
        graph_tail = tuple(doc_index for doc_index in admitted if int(doc_index) not in set(dense_universe.doc_indices))
        docs = tuple(unique_ints([*dense_universe.doc_indices, *graph_tail])[: max(int(top_n), 1)])
        stats = dict(local_graph.get("stats", {}) or {})
        return CandidateUniverse(
            doc_indices=docs,
            admissible_doc_indices=tuple(
                doc_index for doc_index in graph_tail if int(doc_index) in set(docs)
            ),
            graph_payload={
                "agsto_corpus_index": self.agsto_generator.corpus_index,
                "agsto_role_graph": self.agsto_generator.role_graph,
                "agsto_local_graph": local_graph,
            },
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": "dense_preserving_agsto_local_graph_candidate_generator",
                "candidate_source": "method_owned_dense_preserving_agsto_local_graph",
                "query_index": int(query_index),
                "candidate_count": len(docs),
                "preserved_dense_count": int(preserved_count),
                "dense_seed_count": int(self.dense_seed_count),
                "agsto_admitted_doc_count": len(admitted),
                "agsto_appended_doc_count": len([doc for doc in graph_tail if int(doc) in set(docs)]),
                "agsto_local_edge_count": int(stats.get("local_edge_count", 0) or 0),
                "agsto_role_graph_edge_count": int(stats.get("role_graph_edge_count", 0) or 0),
                "agsto_graph_tail_doc_indices": graph_tail[: max(int(top_n), 1)],
                "dense_trace": dict(dense_universe.trace),
                "candidate_order_policy": "preserve_dense_universe_then_append_agsto_admitted_tail",
                "dense_topk_is_final_answer_default": False,
                "uses_weighted_score_fusion": False,
                "uses_external_baseline_frontier": False,
                "uses_replay_candidate_cache": False,
                "uses_hipporag_v2_runtime_graph_retrieval": False,
                "uses_dense_only_as_graph_entry": True,
                "uses_sto_role_graph_admission": True,
                "preserves_dense_candidate_universe": True,
                "ablation_query_supported_same_object_handoff": bool(
                    self.agsto_generator.enable_query_supported_same_object_handoff
                ),
                "uses_variable_flow_traversal": bool(
                    self.agsto_generator.enable_variable_flow_traversal
                ),
            },
        )


@dataclass(frozen=True)
class RowFieldCandidateGenerator:
    """Read candidate universe fields from a query row.

    This is useful for report-backed evaluation when the first-stage retriever
    has already written top-N candidates into the report.
    """

    candidate_fields: Tuple[str, ...]
    field_mode: str = "first"

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        docs: List[int] = []
        if str(self.field_mode) == "union":
            for field in self.candidate_fields:
                docs.extend(unique_ints(_value_at_field(row, field) or []))
            docs = unique_ints(docs)
        else:
            for field in self.candidate_fields:
                docs = unique_ints(_value_at_field(row, field) or [])
                if docs:
                    break
        docs = docs[: max(int(top_n), 1)]
        return CandidateUniverse(
            doc_indices=tuple(docs),
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": "row_field_candidate_generator",
                "candidate_source": "row_fields",
                "candidate_fields": self.candidate_fields,
                "candidate_field_mode": str(self.field_mode),
                "query_index": int(query_index),
                "candidate_count": len(docs),
            },
        )


@dataclass(frozen=True)
class LexicalCandidateGenerator:
    """Dependency-free native candidate generator for clean ablations.

    This generator is deliberately simple and deterministic.  It is not a
    weighted fusion with graph scores; it only exposes a local universe by
    matching query tokens against passage titles/text.
    """

    nodes: Tuple[EvidenceNode, ...]

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        query_tokens = _content_tokens(query)
        ranked: List[Tuple[int, int, int]] = []
        for node in self.nodes:
            title_score = len(query_tokens & _title_token_set(node))
            text_score = len(query_tokens & _content_tokens(node.text))
            if title_score <= 0 and text_score <= 0:
                continue
            ranked.append((-title_score, -text_score, int(node.doc_index)))
        ranked.sort()
        docs = [doc_index for _title_score, _text_score, doc_index in ranked[: max(int(top_n), 1)]]
        return CandidateUniverse(
            doc_indices=tuple(docs),
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": "lexical_candidate_generator",
                "candidate_source": "method_native_lexical",
                "query_index": int(query_index),
                "candidate_count": len(docs),
            },
        )


@dataclass(frozen=True)
class GlobalSTOGraphCandidateGenerator:
    """Generate candidates by query-time search over a corpus STO graph index.

    This is the graph-first candidate boundary.  The corpus-side index is built
    once from passage titles and OpenIE endpoints; each query anchors into that
    index, then expands along source-bound STO transitions.  Lexical roots are
    used only to ground graph entry nodes when the query does not explicitly
    mention a passage title.  They do not decide the final top-k.
    """

    nodes: Tuple[EvidenceNode, ...]
    expansion_index: SourceTextCandidateExpansionIndex
    node_lookup: Mapping[int, EvidenceNode]
    title_token_doc_index: Mapping[str, Tuple[int, ...]]
    text_token_doc_index: Mapping[str, Tuple[int, ...]]
    neighbor_doc_index_cache: Dict[int, Tuple[int, ...]]
    root_candidate_count: int = 5
    max_hops: int = 2

    @classmethod
    def build(
        cls,
        nodes: Sequence[EvidenceNode],
        *,
        root_candidate_count: int = 5,
        max_hops: int = 2,
        max_specific_token_doc_count: int = 32,
    ) -> "GlobalSTOGraphCandidateGenerator":
        node_tuple = tuple(nodes)
        return cls(
            nodes=node_tuple,
            expansion_index=build_source_text_candidate_expansion_index(
                node_tuple,
                max_specific_token_doc_count=max_specific_token_doc_count,
            ),
            node_lookup={int(node.doc_index): node for node in node_tuple},
            title_token_doc_index=_build_title_token_doc_index(node_tuple),
            text_token_doc_index=_build_text_token_doc_index(node_tuple),
            neighbor_doc_index_cache={},
            root_candidate_count=max(int(root_candidate_count), 1),
            max_hops=max(int(max_hops), 0),
        )

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        root_trace = self.root_trace(query=str(query), row=row)
        roots = unique_ints(root_trace["graph_root_doc_indices"])
        title_anchor_docs = unique_ints(root_trace["title_anchor_doc_indices"])
        additional_root_docs = unique_ints(root_trace["additional_root_doc_indices"])
        lexical_root_docs = unique_ints(root_trace["lexical_root_doc_indices"])

        ordered_candidates: List[int] = list(roots)
        visited = set(ordered_candidates)
        frontier = tuple(roots)
        hop_expanded_docs: List[Tuple[int, ...]] = []
        for _hop in range(max(int(self.max_hops), 0)):
            if not frontier:
                break
            next_frontier: List[int] = []
            for source_doc_index in frontier:
                for doc_index in self._global_sto_neighbor_doc_indices(
                    source_doc_index=int(source_doc_index)
                ):
                    doc_index = int(doc_index)
                    if doc_index in visited:
                        continue
                    visited.add(doc_index)
                    ordered_candidates.append(doc_index)
                    next_frontier.append(doc_index)
            hop_expanded_docs.append(tuple(next_frontier))
            frontier = tuple(next_frontier)

        lexical_fill_docs: Tuple[int, ...] = ()
        if len(ordered_candidates) < max(int(top_n), 1):
            lexical_fill_docs = tuple(
                doc_index
                for doc_index in self._lexical_doc_indices(query=str(query), limit=max(int(top_n), 1))
                if int(doc_index) not in visited
            )
            ordered_candidates.extend(lexical_fill_docs)

        docs = tuple(unique_ints(ordered_candidates)[: max(int(top_n), 1)])
        return CandidateUniverse(
            doc_indices=docs,
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": "global_sto_graph_candidate_generator",
                "candidate_source": "method_native_global_sto_graph",
                "query_index": int(query_index),
                "candidate_count": len(docs),
                "root_candidate_count": int(self.root_candidate_count),
                "max_hops": int(self.max_hops),
                **root_trace,
                "hop_expanded_doc_indices": tuple(hop_expanded_docs),
                "lexical_fill_doc_indices": lexical_fill_docs,
                "uses_weighted_score_fusion": False,
                "uses_external_baseline_frontier": False,
            },
        )

    def root_trace(self, *, query: str, row: Mapping[str, Any]) -> Mapping[str, Tuple[int, ...]]:
        query_token_set = _content_tokens(query)
        title_anchor_candidate_docs = unique_ints(
            doc_index
            for token in query_token_set
            for doc_index in self.title_token_doc_index.get(str(token), ())
        )
        title_anchor_docs = query_mentioned_doc_indices(
            query=str(query),
            nodes=self.node_lookup,
            candidate_doc_indices=title_anchor_candidate_docs,
        )
        additional_root_docs = unique_ints(
            row.get("_global_sto_additional_root_doc_indices", ()) or ()
        )
        lexical_root_docs = self._lexical_root_doc_indices(query=str(query))
        roots = unique_ints([*title_anchor_docs, *additional_root_docs, *lexical_root_docs])
        return {
            "title_anchor_doc_indices": title_anchor_docs,
            "additional_root_doc_indices": additional_root_docs,
            "lexical_root_doc_indices": lexical_root_docs,
            "graph_root_doc_indices": roots,
        }

    def _lexical_root_doc_indices(self, *, query: str) -> Tuple[int, ...]:
        return self._lexical_doc_indices(query=query, limit=max(int(self.root_candidate_count), 1))

    def _lexical_doc_indices(self, *, query: str, limit: int) -> Tuple[int, ...]:
        query_tokens = _content_tokens(query)
        doc_scores: Dict[int, List[int]] = {}
        for token in query_tokens:
            for doc_index in self.title_token_doc_index.get(str(token), ()):
                doc_scores.setdefault(int(doc_index), [0, 0])[0] += 1
            for doc_index in self.text_token_doc_index.get(str(token), ()):
                doc_scores.setdefault(int(doc_index), [0, 0])[1] += 1
        ranked = [
            (-int(scores[0]), -int(scores[1]), int(doc_index))
            for doc_index, scores in doc_scores.items()
            if int(scores[0]) > 0 or int(scores[1]) > 0
        ]
        ranked.sort()
        return tuple(
            doc_index
            for _title_score, _text_score, doc_index in ranked[: max(int(limit), 1)]
        )

    def _global_sto_neighbor_doc_indices(self, *, source_doc_index: int) -> Tuple[int, ...]:
        cached = self.neighbor_doc_index_cache.get(int(source_doc_index))
        if cached is not None:
            return cached
        source = self.node_lookup.get(int(source_doc_index))
        if source is None:
            return ()
        neighbors: List[int] = []
        for target_doc_index in self._candidate_docs_mentioned_in_text(source.text):
            if int(target_doc_index) == int(source_doc_index):
                continue
            target = self.node_lookup.get(int(target_doc_index))
            if target is None:
                continue
            if any(
                phrase_occurs(alias, source.text)
                for alias in title_aliases(target.display_title)
                if is_specific_title_alias(alias)
            ):
                neighbors.append(int(target_doc_index))
        for triple in source.triples:
            if len(triple) != 3:
                continue
            subject, _relation, obj = (str(triple[0]), str(triple[1]), str(triple[2]))
            for endpoint in (subject, obj):
                for target_doc_index in self._candidate_docs_for_endpoint(endpoint):
                    if int(target_doc_index) == int(source_doc_index):
                        continue
                    target = self.node_lookup.get(int(target_doc_index))
                    if target is None:
                        continue
                    if _endpoint_matches_title(endpoint=endpoint, target=target):
                        neighbors.append(int(target_doc_index))
        output = tuple(unique_ints(neighbors))
        self.neighbor_doc_index_cache[int(source_doc_index)] = output
        return output

    def _candidate_docs_mentioned_in_text(self, text: object) -> Tuple[int, ...]:
        candidate_docs: List[int] = []
        max_doc_count = max(int(self.expansion_index.max_specific_token_doc_count), 1)
        for token in _rare_tokens(text):
            doc_indices = self.title_token_doc_index.get(str(token), ())
            if not doc_indices or len(doc_indices) > max_doc_count:
                continue
            candidate_docs.extend(int(doc_index) for doc_index in doc_indices)
        return tuple(unique_ints(candidate_docs))

    def _candidate_docs_for_endpoint(self, endpoint: object) -> Tuple[int, ...]:
        candidate_lists = [
            (len(self.title_token_doc_index.get(str(token), ())), str(token), self.title_token_doc_index.get(str(token), ()))
            for token in _rare_tokens(endpoint)
            if self.title_token_doc_index.get(str(token))
        ]
        if not candidate_lists:
            return ()
        candidate_lists.sort(key=lambda item: (int(item[0]), str(item[1])))
        if int(candidate_lists[0][0]) > max(int(self.expansion_index.max_specific_token_doc_count), 1):
            return ()
        return tuple(unique_ints(candidate_lists[0][2]))


@dataclass(frozen=True)
class DenseAnchoredGlobalSTOGraphCandidateGenerator:
    """Use dense retrieval only to anchor graph search, not as final candidates."""

    dense_generator: DenseEmbeddingCandidateGenerator
    graph_generator: GlobalSTOGraphCandidateGenerator
    dense_root_count: int = 20

    @classmethod
    def from_parquet(
        cls,
        *,
        nodes: Sequence[EvidenceNode],
        embedding_store_path: Path,
        embedding_base_url: str,
        embedding_model_name: str,
        dense_root_count: int = 20,
        graph_root_candidate_count: int = 5,
        graph_max_hops: int = 2,
        graph_max_specific_token_doc_count: int = 32,
    ) -> "DenseAnchoredGlobalSTOGraphCandidateGenerator":
        return cls(
            dense_generator=DenseEmbeddingCandidateGenerator.from_parquet(
                nodes=tuple(nodes),
                embedding_store_path=embedding_store_path,
                embedding_base_url=str(embedding_base_url),
                embedding_model_name=str(embedding_model_name),
            ),
            graph_generator=GlobalSTOGraphCandidateGenerator.build(
                nodes=tuple(nodes),
                root_candidate_count=max(int(graph_root_candidate_count), 1),
                max_hops=max(int(graph_max_hops), 0),
                max_specific_token_doc_count=max(int(graph_max_specific_token_doc_count), 1),
            ),
            dense_root_count=max(int(dense_root_count), 1),
        )

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        dense_roots = self.dense_generator.generate(
            query=str(query),
            query_index=int(query_index),
            row=row,
            top_n=max(int(self.dense_root_count), 1),
        )
        graph_row = {
            **dict(row),
            "_global_sto_additional_root_doc_indices": dense_roots.doc_indices,
        }
        graph_universe = self.graph_generator.generate(
            query=str(query),
            query_index=int(query_index),
            row=graph_row,
            top_n=max(int(top_n), 1),
        )
        return CandidateUniverse(
            doc_indices=graph_universe.doc_indices,
            trace={
                **dict(graph_universe.trace),
                "candidate_generator": "dense_anchored_global_sto_graph_candidate_generator",
                "candidate_source": "method_native_dense_anchored_global_sto_graph",
                "dense_root_doc_indices": dense_roots.doc_indices,
                "dense_root_count": int(self.dense_root_count),
                "dense_root_trace": dict(dense_roots.trace),
                "dense_topk_is_final_answer_default": False,
                "uses_weighted_score_fusion": False,
                "uses_external_baseline_frontier": False,
            },
        )

    def prepare_queries(self, queries: Sequence[str]) -> None:
        prepare = getattr(self.dense_generator, "prepare_queries", None)
        if callable(prepare):
            prepare([str(query) for query in queries])


@dataclass(frozen=True)
class DenseEmbeddingCandidateGenerator:
    """Native dense first-stage generator backed by a corpus embedding store.

    Dense retrieval is only used to expose a query-local universe.  It does not
    decide the final reader top-k and is not fused with graph scores.
    """

    doc_indices: Tuple[int, ...]
    embeddings: Tuple[Tuple[float, ...], ...]
    query_encoder: Callable[[str], Sequence[float]]
    name: str = "dense_embedding_candidate_generator"
    embedding_store_path: str = ""

    @classmethod
    def from_parquet(
        cls,
        *,
        nodes: Sequence[EvidenceNode],
        embedding_store_path: Path,
        embedding_base_url: str,
        embedding_model_name: str,
    ) -> "DenseEmbeddingCandidateGenerator":
        """Load passage embeddings and create an OpenAI-compatible query encoder."""

        import numpy as np
        import pandas as pd

        path = embedding_store_path.expanduser()
        frame = pd.read_parquet(path)
        node_by_text = {str(node.text): int(node.doc_index) for node in nodes}
        doc_indices: List[int] = []
        embeddings: List[Tuple[float, ...]] = []
        for row in frame.to_dict(orient="records"):
            content = str(row.get("content", ""))
            if content not in node_by_text:
                continue
            doc_indices.append(int(node_by_text[content]))
            embeddings.append(tuple(float(value) for value in np.asarray(row.get("embedding"), dtype=float)))
        return cls(
            doc_indices=tuple(doc_indices),
            embeddings=tuple(embeddings),
            query_encoder=_openai_compatible_query_encoder(
                base_url=str(embedding_base_url),
                model_name=str(embedding_model_name),
            ),
            name="dense_embedding_store_candidate_generator",
            embedding_store_path=str(path),
        )

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        import numpy as np

        if not self.doc_indices or not self.embeddings:
            docs: Tuple[int, ...] = ()
        else:
            matrix = np.asarray(self.embeddings, dtype=float)
            query_embedding = np.asarray(self.query_encoder(str(query)), dtype=float)
            if query_embedding.ndim != 1:
                query_embedding = np.squeeze(query_embedding)
            if matrix.ndim != 2 or query_embedding.ndim != 1 or matrix.shape[1] != query_embedding.shape[0]:
                raise ValueError(
                    "dense embedding generator dimension mismatch: "
                    f"matrix={matrix.shape}, query={query_embedding.shape}"
                )
            scores = np.dot(matrix, query_embedding.T)
            ranked_positions = np.argsort(scores)[::-1][: max(int(top_n), 1)]
            docs = tuple(int(self.doc_indices[int(position)]) for position in ranked_positions)
        return CandidateUniverse(
            doc_indices=tuple(unique_ints(docs)),
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": self.name,
                "candidate_source": "method_native_dense_embedding_store",
                "embedding_store_path": self.embedding_store_path,
                "query_index": int(query_index),
                "candidate_count": len(docs),
                "embedding_doc_count": len(self.doc_indices),
            },
        )

    def prepare_queries(self, queries: Sequence[str]) -> None:
        prepare = getattr(self.query_encoder, "prepare", None)
        if callable(prepare):
            prepare([str(query) for query in queries])


@dataclass(frozen=True)
class DensePersonalizedGlobalSTOGraphCandidateGenerator:
    """Generate candidates by personalized traversal over the global STO graph.

    Dense retrieval defines reset roots and preserves the source-prior prefix.
    Personalized traversal expands the candidate tail over source-text STO
    edges; it does not replace reader top-k with graph scores.
    """

    dense_generator: DenseEmbeddingCandidateGenerator
    graph_generator: GlobalSTOGraphCandidateGenerator
    dense_root_count: int = 50
    source_prior_prefix_count: int = 5
    restart_probability: float = 0.2
    residual_epsilon: float = 1e-6
    max_pushes: int = 1000

    @classmethod
    def from_parquet(
        cls,
        *,
        nodes: Sequence[EvidenceNode],
        embedding_store_path: Path,
        embedding_base_url: str,
        embedding_model_name: str,
        dense_root_count: int = 50,
        graph_root_candidate_count: int = 5,
        graph_max_hops: int = 2,
        graph_max_specific_token_doc_count: int = 32,
    ) -> "DensePersonalizedGlobalSTOGraphCandidateGenerator":
        return cls(
            dense_generator=DenseEmbeddingCandidateGenerator.from_parquet(
                nodes=tuple(nodes),
                embedding_store_path=embedding_store_path,
                embedding_base_url=str(embedding_base_url),
                embedding_model_name=str(embedding_model_name),
            ),
            graph_generator=GlobalSTOGraphCandidateGenerator.build(
                nodes=tuple(nodes),
                root_candidate_count=max(int(graph_root_candidate_count), 1),
                max_hops=max(int(graph_max_hops), 0),
                max_specific_token_doc_count=max(int(graph_max_specific_token_doc_count), 1),
            ),
            dense_root_count=max(int(dense_root_count), 1),
        )

    def generate(
        self,
        *,
        query: str,
        query_index: int,
        row: Mapping[str, Any],
        top_n: int,
    ) -> CandidateUniverse:
        dense_roots = self.dense_generator.generate(
            query=str(query),
            query_index=int(query_index),
            row=row,
            top_n=max(int(self.dense_root_count), 1),
        )
        graph_row = {
            **dict(row),
            "_global_sto_additional_root_doc_indices": dense_roots.doc_indices,
        }
        root_trace = self.graph_generator.root_trace(query=str(query), row=graph_row)
        roots = unique_ints(root_trace["graph_root_doc_indices"])
        scores, push_count, visited_doc_indices = self._personalized_scores(roots)
        root_rank = {int(doc_index): rank for rank, doc_index in enumerate(roots)}
        ranked_by_graph = sorted(
            scores,
            key=lambda doc_index: (
                -float(scores[int(doc_index)]),
                int(root_rank.get(int(doc_index), 10**9)),
                int(doc_index),
            ),
        )
        source_prior_prefix = tuple(roots[: max(int(self.source_prior_prefix_count), 1)])
        docs = list(source_prior_prefix)
        for doc_index in ranked_by_graph:
            if int(doc_index) not in docs:
                docs.append(int(doc_index))
            if len(docs) >= max(int(top_n), 1):
                break
        if len(docs) < max(int(top_n), 1):
            for doc_index in roots:
                if int(doc_index) not in docs:
                    docs.append(int(doc_index))
                if len(docs) >= max(int(top_n), 1):
                    break
        if len(docs) < max(int(top_n), 1):
            for doc_index in self.graph_generator._lexical_doc_indices(
                query=str(query),
                limit=max(int(top_n), 1),
            ):
                if int(doc_index) not in docs:
                    docs.append(int(doc_index))
                if len(docs) >= max(int(top_n), 1):
                    break
        doc_indices = tuple(unique_ints(docs)[: max(int(top_n), 1)])
        return CandidateUniverse(
            doc_indices=doc_indices,
            trace={
                **dict(CANDIDATE_UNIVERSE_CONTRACT),
                "candidate_generator": "dense_personalized_global_sto_graph_candidate_generator",
                "candidate_source": "method_native_dense_personalized_global_sto_graph",
                "query_index": int(query_index),
                "candidate_count": len(doc_indices),
                "dense_root_doc_indices": dense_roots.doc_indices,
                "dense_root_count": int(self.dense_root_count),
                "source_prior_prefix_count": int(self.source_prior_prefix_count),
                "source_prior_prefix_doc_indices": source_prior_prefix,
                **root_trace,
                "graph_search": "local_push_personalized_sto_traversal",
                "candidate_order_policy": "source_prior_prefix_then_personalized_graph_tail_then_remaining_roots",
                "restart_probability": float(self.restart_probability),
                "residual_epsilon": float(self.residual_epsilon),
                "max_pushes": int(self.max_pushes),
                "push_count": int(push_count),
                "visited_doc_count": len(visited_doc_indices),
                "visited_doc_indices": visited_doc_indices[: max(int(top_n), 1)],
                "dense_topk_is_final_answer_default": False,
                "uses_weighted_score_fusion": False,
                "uses_external_baseline_frontier": False,
            },
        )

    def prepare_queries(self, queries: Sequence[str]) -> None:
        prepare = getattr(self.dense_generator, "prepare_queries", None)
        if callable(prepare):
            prepare([str(query) for query in queries])

    def _personalized_scores(
        self,
        roots: Sequence[int],
    ) -> Tuple[Dict[int, float], int, Tuple[int, ...]]:
        root_tuple = unique_ints(roots)
        if not root_tuple:
            return {}, 0, ()
        restart = min(max(float(self.restart_probability), 0.0), 1.0)
        epsilon = max(float(self.residual_epsilon), 0.0)
        residual: Dict[int, float] = {
            int(doc_index): 1.0 / float(len(root_tuple))
            for doc_index in root_tuple
        }
        scores: Dict[int, float] = {}
        queue = deque(int(doc_index) for doc_index in root_tuple)
        in_queue = set(root_tuple)
        visited: List[int] = []
        push_count = 0
        while queue and push_count < max(int(self.max_pushes), 1):
            doc_index = int(queue.popleft())
            in_queue.discard(doc_index)
            mass = float(residual.pop(doc_index, 0.0))
            if mass <= 0.0:
                continue
            if mass < epsilon:
                scores[doc_index] = scores.get(doc_index, 0.0) + mass
                continue
            push_count += 1
            if doc_index not in visited:
                visited.append(doc_index)
            scores[doc_index] = scores.get(doc_index, 0.0) + restart * mass
            neighbors = self.graph_generator._global_sto_neighbor_doc_indices(
                source_doc_index=doc_index
            )
            if not neighbors:
                scores[doc_index] = scores.get(doc_index, 0.0) + (1.0 - restart) * mass
                continue
            share = ((1.0 - restart) * mass) / float(len(neighbors))
            for neighbor in neighbors:
                neighbor = int(neighbor)
                residual[neighbor] = residual.get(neighbor, 0.0) + share
                if residual[neighbor] >= epsilon and neighbor not in in_queue:
                    queue.append(neighbor)
                    in_queue.add(neighbor)
        for doc_index, mass in residual.items():
            scores[int(doc_index)] = scores.get(int(doc_index), 0.0) + float(mass)
        return scores, push_count, tuple(unique_ints(visited))


def _value_at_field(row: Mapping[str, Any], field: str) -> Any:
    current: Any = row
    for part in str(field).split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _compact_doc_text(text: object) -> str:
    return " ".join(str(text or "").split())


def _content_tokens(text: object) -> set[str]:
    return {
        token
        for token in tokens(text)
        if len(token) >= 3
        and token
        not in {
            "and",
            "are",
            "did",
            "does",
            "for",
            "from",
            "had",
            "has",
            "have",
            "the",
            "was",
            "were",
            "what",
            "when",
            "where",
            "which",
            "who",
            "whose",
            "with",
        }
    }


def _title_token_set(node: EvidenceNode) -> set[str]:
    output: set[str] = set()
    for alias in title_aliases(node.display_title):
        if not is_specific_title_alias(alias):
            continue
        output.update(_content_tokens(alias))
    return output


def _build_title_token_doc_index(
    nodes: Sequence[EvidenceNode],
) -> Mapping[str, Tuple[int, ...]]:
    index: Dict[str, List[int]] = {}
    for node in nodes:
        for token in _title_token_set(node):
            index.setdefault(str(token), []).append(int(node.doc_index))
    return {token: tuple(unique_ints(doc_indices)) for token, doc_indices in index.items()}


def _build_text_token_doc_index(
    nodes: Sequence[EvidenceNode],
) -> Mapping[str, Tuple[int, ...]]:
    index: Dict[str, List[int]] = {}
    for node in nodes:
        for token in _content_tokens(node.text):
            index.setdefault(str(token), []).append(int(node.doc_index))
    return {token: tuple(unique_ints(doc_indices)) for token, doc_indices in index.items()}


def _rare_tokens(text: object) -> Tuple[str, ...]:
    return tuple(token for token in tokens(text) if len(token) >= 4)


def _endpoint_matches_title(*, endpoint: object, target: EvidenceNode) -> bool:
    normalized_endpoint = normalize_text(endpoint)
    if not normalized_endpoint:
        return False
    for alias in title_aliases(target.display_title):
        if not is_specific_title_alias(alias):
            continue
        if normalized_endpoint == alias:
            return True
        if phrase_occurs(alias, normalized_endpoint) or phrase_occurs(normalized_endpoint, alias):
            return True
    return False


@dataclass
class _OpenAICompatibleQueryEncoder:
    base_url: str
    model_name: str
    instruction: str
    batch_size: int = 32

    def __post_init__(self) -> None:
        self._cache: Dict[str, Tuple[float, ...]] = {}

    def prepare(self, queries: Sequence[str]) -> None:
        missing = [str(query) for query in dict.fromkeys(queries) if str(query) not in self._cache]
        if not missing:
            return
        for start in range(0, len(missing), max(int(self.batch_size), 1)):
            batch = missing[start : start + max(int(self.batch_size), 1)]
            embeddings = self._request(batch)
            for query, embedding in zip(batch, embeddings):
                self._cache[str(query)] = tuple(float(value) for value in embedding)

    def __call__(self, query: str) -> Sequence[float]:
        query = str(query)
        self.prepare([query])
        return self._cache[query]

    def _request(self, queries: Sequence[str]) -> Sequence[Sequence[float]]:
        import requests

        payload = {
            "model": self.model_name,
            "input": prefix_queries_with_instruction([str(query) for query in queries], self.instruction),
        }
        response = requests.post(
            str(self.base_url),
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if len(data) != len(queries):
            raise ValueError(
                "embedding endpoint returned unexpected embedding count: "
                f"expected={len(queries)}, actual={len(data)}"
            )
        return [row["embedding"] for row in data]


def _openai_compatible_query_encoder(
    *,
    base_url: str,
    model_name: str,
) -> Callable[[str], Sequence[float]]:
    if not base_url:
        raise ValueError("embedding_base_url is required for dense embedding candidate generation")
    if not model_name:
        raise ValueError("embedding_model_name is required for dense embedding candidate generation")

    from src.hipporag.prompts.linking import get_query_instruction

    return _OpenAICompatibleQueryEncoder(
        base_url=str(base_url),
        model_name=str(model_name),
        instruction=get_query_instruction("query_to_passage"),
    )
