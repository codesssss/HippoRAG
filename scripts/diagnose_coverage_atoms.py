import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import (  # noqa: E402
    build_config,
    build_doc_text_to_chunk_id,
    collect_lexical_query_seed_entities,
    collect_query_seed_entities,
    extract_doc_title,
    normalize_entity_set,
    normalize_structure_text,
)
from src.hipporag.HippoRAG import HippoRAG  # noqa: E402
from src.hipporag.utils.dataset_utils import resolve_dataset_paths  # noqa: E402


LOGGER = logging.getLogger(__name__)


def load_corpus(dataset: str) -> List[dict]:
    corpus_path, _ = resolve_dataset_paths(dataset, ROOT_DIR / "reproduce" / "dataset")
    return json.loads(corpus_path.read_text(encoding="utf-8"))


def build_chunk_id_to_doc_text(corpus: Sequence[dict]) -> Dict[str, str]:
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(list(corpus))
    return {chunk_id: doc_text for doc_text, chunk_id in doc_text_to_chunk_id.items()}


def resolve_save_dir(save_dir: str, dataset: str) -> str:
    if save_dir == "outputs":
        return str(Path(save_dir) / dataset)
    return f"{save_dir}_{dataset}"


def infer_resolved_save_dir(report_path: Path) -> str:
    return str(report_path.parent.parent)


def resolve_runtime_save_dir(report_path: Path, dataset: str, save_dir_arg: str) -> str:
    raw_value = str(save_dir_arg or "").strip()
    if not raw_value:
        return infer_resolved_save_dir(report_path)

    candidate_path = Path(raw_value)
    if candidate_path.exists():
        return str(candidate_path)

    return resolve_save_dir(raw_value, dataset)


def build_runtime_args(report_payload: Dict[str, Any], dataset: str, qa_top_k: int, save_dir: str) -> SimpleNamespace:
    config = dict(report_payload.get("config", {}) or {})
    return SimpleNamespace(
        save_dir=save_dir,
        llm_base_url=str(report_payload.get("llm_base_url", "http://localhost:8043/v1")),
        llm_name=str(report_payload.get("llm_name", "qwen3-8b")),
        llm_request_name=str(report_payload.get("llm_request_name", report_payload.get("llm_name", "qwen3-8b"))),
        embedding_base_url=report_payload.get("embedding_base_url", "http://localhost:8018/v1/embeddings"),
        dataset=dataset,
        embedding_name=str(report_payload.get("embedding_name", "VLLM//mnt/nvme/Qwen3-Embedding-8B")),
        force_index_from_scratch="false",
        force_openie_from_scratch="false",
        max_retry_attempts=int(report_payload.get("max_retry_attempts", 12) or 12),
        openie_mode=str(report_payload.get("openie_mode", "online")),
        planner_enabled="false",
        planner_mode="none",
        planner_max_steps=3,
        retrieval_top_k=int(config.get("retrieval_top_k", 200) or 200),
        linking_top_k=int(config.get("linking_top_k", 5) or 5),
        qa_top_k=int(qa_top_k),
        max_qa_steps=int(config.get("max_qa_steps", 3) or 3),
        embedding_batch_size=int(config.get("embedding_batch_size", 8) or 8),
        causal_enabled="false",
        causal_query_only="true",
        causal_gate_mode="hard",
        causal_seed_top_k=int(config.get("causal_seed_top_k", 20) or 20),
        causal_confidence_threshold=float(config.get("causal_confidence_threshold", 0.5) or 0.5),
        causal_damping=float(config.get("causal_damping", 0.7) or 0.7),
        causal_blend_dense_weight=float(config.get("causal_blend_dense_weight", 0.35) or 0.35),
        causal_blend_fact_weight=float(config.get("causal_blend_fact_weight", 0.15) or 0.15),
        causal_blend_graph_weight=float(config.get("causal_blend_graph_weight", 0.50) or 0.50),
        causal_margin_gate_enabled="false",
        causal_margin_threshold=float(config.get("causal_margin_threshold", 0.02) or 0.02),
        causal_blend_top_k=int(config.get("causal_blend_top_k", 0) or 0),
        causal_engine_version="v2",
        causal_v2_probe_mode=str(config.get("causal_v2_probe_mode", "router")),
        causal_v2_graph_mode=str(config.get("causal_v2_graph_mode", "causal")),
        causal_v2_base_retrieval_mode="legacy_fact_graph",
        general_graph_related_to_weight=float(config.get("general_graph_related_to_weight", 0.3) or 0.3),
        general_graph_seed_top_k=int(config.get("general_graph_seed_top_k", 10) or 10),
        causal_v2_extraction_max_tokens=int(config.get("causal_v2_extraction_max_tokens", 768) or 768),
        causal_v2_extraction_retry_attempts=int(config.get("causal_v2_extraction_retry_attempts", 2) or 2),
        causal_v2_extraction_workers=int(config.get("causal_v2_extraction_workers", 4) or 4),
        causal_event_top_k=int(config.get("causal_event_top_k", 8) or 8),
        causal_v2_max_hops=int(config.get("causal_v2_max_hops", 2) or 2),
        causal_chain_top_k=int(config.get("causal_chain_top_k", 6) or 6),
        causal_context_max_items=int(config.get("causal_context_max_items", 0) or 0),
        causal_er_similarity_threshold=float(config.get("causal_er_similarity_threshold", 0.92) or 0.92),
        causal_er_text_threshold=float(config.get("causal_er_text_threshold", 0.55) or 0.55),
        causal_v2_min_edge_confidence=float(config.get("causal_v2_min_edge_confidence", 0.7) or 0.7),
        structure_rerank_enabled="true",
        structure_rerank_top_n=int(config.get("structure_rerank_top_n", 40) or 40),
        structure_rerank_bonus_weight=float(config.get("structure_rerank_bonus_weight", 0.08) or 0.08),
        structure_rerank_min_edge_support=int(config.get("structure_rerank_min_edge_support", 2) or 2),
        structure_rerank_max_top5_swaps=int(config.get("structure_rerank_max_top5_swaps", 2) or 2),
        structure_rerank_seed_top_k=int(config.get("structure_rerank_seed_top_k", 4) or 4),
        structure_rerank_max_hops=int(config.get("structure_rerank_max_hops", 2) or 2),
        structure_relation_probe_mode=str(config.get("structure_relation_probe_mode", "general_factual")),
        structure_continuity_probe_mode=str(config.get("structure_continuity_probe_mode", "off")),
        structure_seed_target_bridge_mode=str(config.get("structure_seed_target_bridge_mode", "off")),
        structure_rerank_margin_threshold=float(config.get("structure_rerank_margin_threshold", 0.02) or 0.02),
        rerank_require_non_empty="true",
    )


def normalize_candidate_positions(candidate_positions: Sequence[int], pool_size: int) -> List[int]:
    normalized = []
    seen = set()
    for position in candidate_positions:
        try:
            normalized_position = int(position)
        except (TypeError, ValueError):
            continue
        if normalized_position < 0 or normalized_position >= pool_size or normalized_position in seen:
            continue
        seen.add(normalized_position)
        normalized.append(normalized_position)
    return normalized


def build_pool_mapping_from_report(
    retrieved_doc_ids: Sequence[str],
    pool_limit: int,
    chunk_id_to_doc_text: Dict[str, str],
    hipporag: HippoRAG,
) -> Tuple[List[str], np.ndarray, List[int | None]]:
    pool_chunk_ids = list(retrieved_doc_ids[:pool_limit])
    missing_chunk_ids = [chunk_id for chunk_id in pool_chunk_ids if chunk_id not in chunk_id_to_doc_text]
    if missing_chunk_ids:
        missing_preview = ", ".join(str(chunk_id) for chunk_id in missing_chunk_ids[:5])
        raise KeyError(f"Failed to map {len(missing_chunk_ids)} retrieved chunk ids back to corpus text. Examples: {missing_preview}")
    pool_docs = [chunk_id_to_doc_text[chunk_id] for chunk_id in pool_chunk_ids]
    pool_scores = np.linspace(pool_limit, 1, pool_limit, dtype=float)

    pool_doc_ids: List[int | None] = []
    for chunk_id in pool_chunk_ids:
        mapped_doc_id = hipporag.passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
        pool_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)
    return pool_docs, pool_scores, pool_doc_ids


def is_valid_subset(subset: Sequence[int], pool_docs: Sequence[str]) -> bool:
    seen = set()
    for pos in subset:
        title_key = normalize_structure_text(extract_doc_title(pool_docs[pos]))
        if title_key and title_key in seen:
            return False
        if title_key:
            seen.add(title_key)
    return True


def build_doc_cover_maps(
    candidate_positions: Sequence[int],
    pool_doc_ids: Sequence[int | None],
    doc_idx_to_entities: Dict[int, Set[str]],
    doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
    seed_entities: Set[str],
) -> Tuple[Set[str], Set[Tuple[str, str]], Set[str], Dict[int, Set[str]], Dict[int, Set[Tuple[str, str]]], Dict[int, Set[str]]]:
    a_q = set(seed_entities)
    a_e: Set[Tuple[str, str]] = set()
    a_b: Set[str] = set()

    for pos in candidate_positions:
        doc_id = pool_doc_ids[pos]
        if doc_id is None:
            continue
        normalized_entities = normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
        a_b.update(normalized_entities)
        for src, tgt, _, _ in doc_idx_to_edges.get(int(doc_id), []):
            normalized_src = normalize_structure_text(src)
            normalized_tgt = normalize_structure_text(tgt)
            if normalized_src and normalized_tgt:
                a_e.add((normalized_src, normalized_tgt))
    a_b -= a_q

    doc_covers_q: Dict[int, Set[str]] = {}
    doc_covers_e: Dict[int, Set[Tuple[str, str]]] = {}
    doc_covers_b: Dict[int, Set[str]] = {}
    for pos in candidate_positions:
        doc_id = pool_doc_ids[pos]
        if doc_id is None:
            doc_covers_q[pos] = set()
            doc_covers_e[pos] = set()
            doc_covers_b[pos] = set()
            continue

        normalized_entities = normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
        normalized_edges = {
            (normalize_structure_text(src), normalize_structure_text(tgt))
            for src, tgt, _, _ in doc_idx_to_edges.get(int(doc_id), [])
            if normalize_structure_text(src) and normalize_structure_text(tgt)
        }
        doc_covers_q[pos] = normalized_entities & a_q
        doc_covers_e[pos] = normalized_edges & a_e
        doc_covers_b[pos] = (normalized_entities - a_q) & a_b

    return a_q, a_e, a_b, doc_covers_q, doc_covers_e, doc_covers_b


def subset_coverage(
    subset: Sequence[int],
    doc_covers_q: Dict[int, Set[str]],
    doc_covers_e: Dict[int, Set[Tuple[str, str]]],
    doc_covers_b: Dict[int, Set[str]],
) -> Tuple[int, int, int]:
    covered_q: Set[str] = set()
    covered_e: Set[Tuple[str, str]] = set()
    covered_b: Set[str] = set()
    for pos in subset:
        covered_q.update(doc_covers_q.get(pos, set()))
        covered_e.update(doc_covers_e.get(pos, set()))
        covered_b.update(doc_covers_b.get(pos, set()))
    return len(covered_q), len(covered_e), len(covered_b)


def _positions_tiebreak_key(subset: Sequence[int]) -> Tuple[int, ...]:
    return tuple(-int(pos) for pos in subset)


def evaluate_candidate_subsets(
    candidate_positions: Sequence[int],
    qa_top_k: int,
    pool_docs: Sequence[str],
    pool_scores: Sequence[float],
    doc_covers_q: Dict[int, Set[str]],
    doc_covers_e: Dict[int, Set[Tuple[str, str]]],
    doc_covers_b: Dict[int, Set[str]],
) -> Dict[str, Any]:
    cov_q_values: Set[int] = set()
    cov_e_values: Set[int] = set()
    cov_b_values: Set[int] = set()
    valid_subset_count = 0

    best_cove_tuple: Tuple[int, int, int, float, Tuple[int, ...]] | None = None
    best_cove_subset: List[int] = []
    best_coverage_tuple: Tuple[int, int, int, float, Tuple[int, ...]] | None = None
    best_coverage_subset: List[int] = []
    best_score_tuple: Tuple[float, int, int, int, Tuple[int, ...]] | None = None
    best_score_subset: List[int] = []

    for subset_tuple in combinations(candidate_positions, qa_top_k):
        if not is_valid_subset(subset_tuple, pool_docs):
            continue

        valid_subset_count += 1
        cov_q, cov_e, cov_b = subset_coverage(
            subset=subset_tuple,
            doc_covers_q=doc_covers_q,
            doc_covers_e=doc_covers_e,
            doc_covers_b=doc_covers_b,
        )
        score_sum = float(sum(float(pool_scores[pos]) for pos in subset_tuple))
        cov_q_values.add(cov_q)
        cov_e_values.add(cov_e)
        cov_b_values.add(cov_b)

        coverage_tuple = (cov_q, cov_e, cov_b, score_sum, _positions_tiebreak_key(subset_tuple))
        cove_tuple = (cov_e, cov_q, cov_b, score_sum, _positions_tiebreak_key(subset_tuple))
        score_tuple = (score_sum, cov_q, cov_e, cov_b, _positions_tiebreak_key(subset_tuple))

        if best_coverage_tuple is None or coverage_tuple > best_coverage_tuple:
            best_coverage_tuple = coverage_tuple
            best_coverage_subset = list(subset_tuple)
        if best_cove_tuple is None or cove_tuple > best_cove_tuple:
            best_cove_tuple = cove_tuple
            best_cove_subset = list(subset_tuple)
        if best_score_tuple is None or score_tuple > best_score_tuple:
            best_score_tuple = score_tuple
            best_score_subset = list(subset_tuple)

    cov_q_at_max_cove = 0
    if best_cove_subset:
        cov_q_at_max_cove = subset_coverage(
            subset=best_cove_subset,
            doc_covers_q=doc_covers_q,
            doc_covers_e=doc_covers_e,
            doc_covers_b=doc_covers_b,
        )[0]

    return {
        "valid_subset_count": int(valid_subset_count),
        "cov_q_values": sorted(cov_q_values),
        "cov_e_values": sorted(cov_e_values),
        "cov_b_values": sorted(cov_b_values),
        "best_coverage_positions": best_coverage_subset,
        "best_baseline_score_positions": best_score_subset,
        "cov_q_at_max_cov_e": int(cov_q_at_max_cove),
    }


def _subset_titles(subset: Sequence[int], pool_docs: Sequence[str]) -> List[str]:
    return [extract_doc_title(pool_docs[pos]) for pos in subset]


def _titles_symmetric_diff(left_titles: Sequence[str], right_titles: Sequence[str]) -> List[str]:
    left = {normalize_structure_text(title): title for title in left_titles if normalize_structure_text(title)}
    right = {normalize_structure_text(title): title for title in right_titles if normalize_structure_text(title)}
    diff_keys = sorted(set(left) ^ set(right))
    return [left.get(key) or right.get(key) or key for key in diff_keys]


def summarize_query(
    question: str,
    candidate_positions: Sequence[int],
    qa_top_k: int,
    pool_docs: Sequence[str],
    pool_scores: Sequence[float],
    pool_doc_ids: Sequence[int | None],
    seed_entities: Set[str],
    doc_idx_to_entities: Dict[int, Set[str]],
    doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
) -> Dict[str, Any]:
    (
        a_q,
        a_e,
        a_b,
        doc_covers_q,
        doc_covers_e,
        doc_covers_b,
    ) = build_doc_cover_maps(
        candidate_positions=candidate_positions,
        pool_doc_ids=pool_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        seed_entities=seed_entities,
    )

    subset_stats = evaluate_candidate_subsets(
        candidate_positions=candidate_positions,
        qa_top_k=qa_top_k,
        pool_docs=pool_docs,
        pool_scores=pool_scores,
        doc_covers_q=doc_covers_q,
        doc_covers_e=doc_covers_e,
        doc_covers_b=doc_covers_b,
    )

    best_coverage_positions = list(subset_stats["best_coverage_positions"])
    best_baseline_score_positions = list(subset_stats["best_baseline_score_positions"])
    best_coverage_titles = _subset_titles(best_coverage_positions, pool_docs)
    best_baseline_titles = _subset_titles(best_baseline_score_positions, pool_docs)

    cov_q_values = list(subset_stats["cov_q_values"])
    cov_e_values = list(subset_stats["cov_e_values"])
    cov_b_values = list(subset_stats["cov_b_values"])
    return {
        "question": question,
        "candidate_count": int(len(candidate_positions)),
        "valid_subset_count": int(subset_stats["valid_subset_count"]),
        "A_Q_size": int(len(a_q)),
        "A_E_size": int(len(a_e)),
        "A_B_size": int(len(a_b)),
        "covQ_distinct_values": int(len(cov_q_values)),
        "covE_distinct_values": int(len(cov_e_values)),
        "covB_distinct_values": int(len(cov_b_values)),
        "covQ_values": cov_q_values,
        "covE_values": cov_e_values,
        "covB_values": cov_b_values,
        "covQ_max": int(max(cov_q_values, default=0)),
        "covE_max": int(max(cov_e_values, default=0)),
        "covB_max": int(max(cov_b_values, default=0)),
        "covQ_at_max_covE": int(subset_stats["cov_q_at_max_cov_e"]),
        "covQ_is_constant": len(cov_q_values) <= 1,
        "covE_is_constant": len(cov_e_values) <= 1,
        "covB_is_constant": len(cov_b_values) <= 1,
        "best_coverage_positions": best_coverage_positions,
        "best_baseline_score_positions": best_baseline_score_positions,
        "best_coverage_titles": best_coverage_titles,
        "best_baseline_score_titles": best_baseline_titles,
        "coverage_vs_baseline_overlap": int(len(set(best_coverage_positions) & set(best_baseline_score_positions))),
        "coverage_vs_baseline_titles_diff": _titles_symmetric_diff(best_coverage_titles, best_baseline_titles),
    }


def print_query_table(query_rows: Sequence[Dict[str, Any]]) -> None:
    print(
        "idx | cand | valid | AQ | AE | AB | dQ | dE | dB | ovlp | question"
    )
    print(
        "----+------+------+----+----+----+----+----+----+------+---------"
    )
    for idx, row in enumerate(query_rows, start=1):
        question = str(row["question"]).replace("\n", " ").strip()
        if len(question) > 96:
            question = question[:93] + "..."
        print(
            f"{idx:>3} | {row['candidate_count']:>4} | {row['valid_subset_count']:>5} | "
            f"{row['A_Q_size']:>2} | {row['A_E_size']:>2} | {row['A_B_size']:>2} | "
            f"{row['covQ_distinct_values']:>2} | {row['covE_distinct_values']:>2} | {row['covB_distinct_values']:>2} | "
            f"{row['coverage_vs_baseline_overlap']:>4} | {question}"
        )


def aggregate_query_rows(query_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    query_count = len(query_rows)
    if query_count <= 0:
        return {
            "query_count": 0,
            "covQ_constant_pct": 0.0,
            "covE_constant_pct": 0.0,
            "covB_constant_pct": 0.0,
            "avg_covQ_distinct": 0.0,
            "avg_covE_distinct": 0.0,
            "avg_covB_distinct": 0.0,
            "avg_coverage_vs_baseline_overlap": 0.0,
            "queries_where_coverage_differs_from_baseline": 0,
            "queries_with_no_valid_subset": 0,
        }

    def _mean(field: str) -> float:
        return float(sum(float(row[field]) for row in query_rows) / query_count)

    return {
        "query_count": int(query_count),
        "covQ_constant_pct": round(sum(1 for row in query_rows if row["covQ_is_constant"]) / query_count, 4),
        "covE_constant_pct": round(sum(1 for row in query_rows if row["covE_is_constant"]) / query_count, 4),
        "covB_constant_pct": round(sum(1 for row in query_rows if row["covB_is_constant"]) / query_count, 4),
        "avg_covQ_distinct": round(_mean("covQ_distinct_values"), 4),
        "avg_covE_distinct": round(_mean("covE_distinct_values"), 4),
        "avg_covB_distinct": round(_mean("covB_distinct_values"), 4),
        "avg_coverage_vs_baseline_overlap": round(_mean("coverage_vs_baseline_overlap"), 4),
        "queries_where_coverage_differs_from_baseline": int(
            sum(1 for row in query_rows if row["best_coverage_positions"] != row["best_baseline_score_positions"])
        ),
        "queries_with_no_valid_subset": int(sum(1 for row in query_rows if row["valid_subset_count"] <= 0)),
    }


def print_aggregate_summary(aggregate: Dict[str, Any]) -> None:
    print("\nAggregate Summary")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))

    if float(aggregate["covQ_constant_pct"]) > 0.80:
        print("FLAG: CovQ has low discriminability — narrative should be CovE-driven")
    if float(aggregate["covE_constant_pct"]) > 0.50:
        print("FLAG: WARNING: CovE also has low discriminability — coverage assembly may not work, check predicate coverage")
    if float(aggregate["avg_coverage_vs_baseline_overlap"]) > 4.0:
        print("FLAG: Coverage and baseline select nearly identical docs — limited room for improvement")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose coverage atom discriminability on bridge_append candidate sets.")
    parser.add_argument("--report", type=str, required=True, help="Path to a bridge_append eval report JSON.")
    parser.add_argument("--dataset", choices=["musique", "hotpotqa", "2wikimultihopqa"], required=True)
    parser.add_argument("--qa_top_k", type=int, required=True)
    parser.add_argument("--save_dir", type=str, default="", help="Optional save_dir root override (defaults to inference from report path).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report_path = Path(args.report).resolve()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    if str(((report_payload.get("expand_assemble_qa") or {}).get("selector", ""))).strip().lower() != "bridge_append":
        raise ValueError("Report is not from bridge_append.")

    query_traces = list(report_payload.get("expand_assemble_query_traces", []) or [])
    if not query_traces:
        raise ValueError("Report does not contain expand_assemble_query_traces.")

    resolved_save_dir = resolve_runtime_save_dir(
        report_path=report_path,
        dataset=args.dataset,
        save_dir_arg=args.save_dir,
    )
    LOGGER.info("Using resolved save_dir: %s", resolved_save_dir)

    corpus = load_corpus(args.dataset)
    docs = [f"{row['title']}\n{row['text']}" for row in corpus]
    chunk_id_to_doc_text = build_chunk_id_to_doc_text(corpus)

    runtime_args = build_runtime_args(
        report_payload=report_payload,
        dataset=args.dataset,
        qa_top_k=args.qa_top_k,
        save_dir=resolved_save_dir,
    )
    config = build_config(runtime_args, corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    hipporag.prepare_retrieval_objects()

    examples = list(report_payload.get("examples", []) or [])
    if len(examples) != len(query_traces):
        raise RuntimeError(f"Report examples count {len(examples)} does not match trace count {len(query_traces)}.")
    query_rows: List[Dict[str, Any]] = []

    for trace_row, example_row in zip(query_traces, examples):
        question = str(trace_row["question"])
        if str(example_row.get("question", "")) != question:
            raise RuntimeError("Report examples and query traces are misaligned by question.")
        expand_trace = dict(trace_row.get("expand_assemble_trace", {}) or {})
        retrieved_doc_ids = list(example_row.get("retrieved_doc_ids", []) or [])
        if not retrieved_doc_ids:
            raise RuntimeError(f"Missing retrieved_doc_ids for question: {question}")
        requested_positions = normalize_candidate_positions(
            expand_trace.get("candidate_set_positions", []) or [],
            len(retrieved_doc_ids),
        )
        if not requested_positions:
            query_rows.append({
                "question": question,
                "candidate_count": 0,
                "valid_subset_count": 0,
                "A_Q_size": 0,
                "A_E_size": 0,
                "A_B_size": 0,
                "covQ_distinct_values": 0,
                "covE_distinct_values": 0,
                "covB_distinct_values": 0,
                "covQ_values": [],
                "covE_values": [],
                "covB_values": [],
                "covQ_max": 0,
                "covE_max": 0,
                "covB_max": 0,
                "covQ_at_max_covE": 0,
                "covQ_is_constant": True,
                "covE_is_constant": True,
                "covB_is_constant": True,
                "best_coverage_positions": [],
                "best_baseline_score_positions": [],
                "best_coverage_titles": [],
                "best_baseline_score_titles": [],
                "coverage_vs_baseline_overlap": 0,
                "coverage_vs_baseline_titles_diff": [],
            })
            continue

        pool_limit = min(
            len(retrieved_doc_ids),
            max(
                (max(requested_positions) + 1),
                int(expand_trace.get("pool_k", 0) or 0),
                int(args.qa_top_k),
            ),
        )
        pool_docs, pool_scores, pool_doc_ids = build_pool_mapping_from_report(
            retrieved_doc_ids=retrieved_doc_ids,
            pool_limit=pool_limit,
            chunk_id_to_doc_text=chunk_id_to_doc_text,
            hipporag=hipporag,
        )

        trace_seed_entities = normalize_entity_set(expand_trace.get("seed_entities_preview", []) or [])
        seed_entities = set(trace_seed_entities)
        if not seed_entities:
            seed_entities = collect_query_seed_entities(hipporag, question)
        if not seed_entities:
            seed_entities = collect_lexical_query_seed_entities(
                query=question,
                pool_doc_ids=pool_doc_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )

        query_rows.append(
            summarize_query(
                question=question,
                candidate_positions=requested_positions,
                qa_top_k=args.qa_top_k,
                pool_docs=pool_docs,
                pool_scores=pool_scores,
                pool_doc_ids=pool_doc_ids,
                seed_entities=seed_entities,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
            )
        )

    aggregate = aggregate_query_rows(query_rows)
    output_payload = {
        "report": str(report_path),
        "dataset": args.dataset,
        "qa_top_k": int(args.qa_top_k),
        "score_proxy_mode": "retrieval_rank_descending",
        "query_rows": query_rows,
        "aggregate": aggregate,
    }
    output_path = report_path.parent / f"coverage_atom_diagnostic_{args.dataset}_{args.qa_top_k}.json"
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print_query_table(query_rows)
    print_aggregate_summary(aggregate)
    print(f"\nSaved diagnostic JSON to {output_path}")


if __name__ == "__main__":
    main()
