#!/usr/bin/env python3
"""Export document-transition control pools for EvLink ablations.

The exported JSON matches the external-pool schema consumed by
``evidenceflow/run_native_pool.py``.  The controls keep the same evidence
delivery boundary as EvLink but replace fact-certified document
transitions with dense document-neighbor transitions, random transitions, or a
query-level edge-count-matched dense transition graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from export_dense_pool import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SAVE_DIR,
    compute_title_recall,
    embed_queries,
    extract_title,
    get_gold_answers,
    get_gold_docs,
    normalize_rows,
    resolve_chunk_embedding_path,
    resolve_dataset_file_stem,
    string_to_bool,
)


CONTROL_POLICIES = (
    "dense_doc_knn",
    "degree_matched_shuffle",
    "edge_count_matched_dense_doc_knn",
    "same_seed_edge_count_matched_dense_doc_knn",
)


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    clean_k = min(max(int(k), 1), int(scores.shape[0]))
    if clean_k >= len(scores):
        top = np.argsort(scores)[::-1]
    else:
        top = np.argpartition(scores, -clean_k)[-clean_k:]
        top = top[np.argsort(scores[top])[::-1]]
    return np.asarray(top[:clean_k], dtype=np.int64)


def compute_dense_doc_knn(
    doc_embeddings: np.ndarray,
    *,
    neighbor_k: int,
    block_size: int,
) -> List[List[int]]:
    """Return top dense neighbors for every document, excluding self."""

    matrix = np.asarray(doc_embeddings, dtype=np.float32)
    n_docs = int(matrix.shape[0])
    clean_k = min(max(int(neighbor_k), 1), max(n_docs - 1, 1))
    clean_block = max(int(block_size), 1)
    neighbors: List[List[int]] = [[] for _ in range(n_docs)]
    doc_t = matrix.T
    for start in tqdm(range(0, n_docs, clean_block), desc="Dense doc KNN"):
        stop = min(start + clean_block, n_docs)
        scores = matrix[start:stop] @ doc_t
        rows = np.arange(start, stop)
        scores[np.arange(stop - start), rows] = -np.inf
        take_k = min(clean_k, max(n_docs - 1, 1))
        top = np.argpartition(scores, -take_k, axis=1)[:, -take_k:]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(top_scores, axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        for offset, row in enumerate(top):
            neighbors[start + offset] = [int(idx) for idx in row if int(idx) != start + offset][:clean_k]
    return neighbors


def compute_shuffled_doc_neighbors(
    *,
    n_docs: int,
    neighbor_k: int,
    seed: int,
) -> List[List[int]]:
    """Return fixed out-degree random neighbors for every document."""

    clean_k = min(max(int(neighbor_k), 1), max(int(n_docs) - 1, 1))
    rng = np.random.default_rng(int(seed))
    all_indices = np.arange(int(n_docs), dtype=np.int64)
    neighbors: List[List[int]] = []
    for doc_idx in tqdm(range(int(n_docs)), desc="Shuffled doc neighbors"):
        candidates = all_indices[all_indices != int(doc_idx)]
        if clean_k >= len(candidates):
            chosen = rng.permutation(candidates)
        else:
            chosen = rng.choice(candidates, size=clean_k, replace=False)
        neighbors.append([int(idx) for idx in chosen[:clean_k]])
    return neighbors


def compute_edge_count_matched_dense_neighbors(
    doc_embeddings: np.ndarray,
    *,
    local_doc_indices: Sequence[int],
    target_edge_count: int,
) -> tuple[dict[int, list[int]], dict[str, int | bool]]:
    """Build a local dense KNN graph with exactly the requested directed-edge count.

    The reference EvLink trace exposes a query-level local edge count, but
    not the original per-source out-degree distribution.  We therefore match the
    total number of directed local edges and distribute out-degree as evenly as
    possible across the local dense candidate nodes.
    """

    local_nodes = [int(idx) for idx in dict.fromkeys(int(x) for x in local_doc_indices)]
    n_local = len(local_nodes)
    if n_local <= 1:
        return {}, {
            "reference_edge_count": int(target_edge_count),
            "control_edge_count": 0,
            "edge_count_cap": 0,
            "strict_total_edge_count_match": int(target_edge_count) == 0,
        }

    requested_edges = max(int(target_edge_count), 0)
    edge_cap = n_local * (n_local - 1)
    control_edges = min(requested_edges, edge_cap)
    base_degree = control_edges // n_local
    remainder = control_edges % n_local

    matrix = np.asarray(doc_embeddings, dtype=np.float32)
    local_matrix = matrix[np.asarray(local_nodes, dtype=np.int64)]
    scores = local_matrix @ local_matrix.T
    np.fill_diagonal(scores, -np.inf)
    neighbors: dict[int, list[int]] = {}
    for local_pos, doc_idx in enumerate(local_nodes):
        out_degree = base_degree + (1 if local_pos < remainder else 0)
        if out_degree <= 0:
            neighbors[int(doc_idx)] = []
            continue
        top_local = _top_indices(scores[local_pos], out_degree)
        neighbors[int(doc_idx)] = [
            int(local_nodes[int(neighbor_pos)])
            for neighbor_pos in top_local
            if int(local_nodes[int(neighbor_pos)]) != int(doc_idx)
        ][:out_degree]

    actual_edges = sum(len(v) for v in neighbors.values())
    return neighbors, {
        "reference_edge_count": requested_edges,
        "control_edge_count": int(actual_edges),
        "edge_count_cap": int(edge_cap),
        "strict_total_edge_count_match": bool(actual_edges == requested_edges),
    }


def traverse_doc_graph(
    *,
    seed_docs: Sequence[int],
    neighbors_by_doc: Sequence[Sequence[int]] | Mapping[int, Sequence[int]],
    pool_k: int,
    closure_hops: int,
) -> List[int]:
    admitted: List[int] = []
    seen: set[int] = set()
    queue: deque[tuple[int, int]] = deque()

    def admit(doc_idx: int, distance: int) -> None:
        doc = int(doc_idx)
        if doc in seen or doc < 0 or len(admitted) >= int(pool_k):
            return
        seen.add(doc)
        admitted.append(doc)
        queue.append((doc, int(distance)))

    for doc_idx in seed_docs:
        admit(int(doc_idx), 0)

    while queue and len(admitted) < int(pool_k):
        current, distance = queue.popleft()
        if int(distance) >= int(closure_hops):
            continue
        if isinstance(neighbors_by_doc, Mapping):
            neighbors = neighbors_by_doc.get(int(current), ())
        else:
            neighbors = neighbors_by_doc[int(current)]
        for neighbor in neighbors:
            admit(int(neighbor), int(distance) + 1)
            if len(admitted) >= int(pool_k):
                break
    return admitted


def ordered_unique(values: Iterable[int], *, limit: int) -> List[int]:
    output: List[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= int(limit):
            break
    return output


def _reference_int_list(container: Mapping[str, Any], key: str) -> List[int]:
    values = container.get(key) or []
    output: List[int] = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--dense-seed-k", type=int, default=20)
    parser.add_argument("--prefix-k", type=int, default=5)
    parser.add_argument("--neighbor-k", type=int, default=20)
    parser.add_argument("--closure-hops", type=int, default=2)
    parser.add_argument("--policy", choices=CONTROL_POLICIES, required=True)
    parser.add_argument(
        "--reference-pool-json",
        type=Path,
        default=None,
        help="EvLink pool JSON used by edge-count matched control policies.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260520)
    parser.add_argument("--knn-block-size", type=int, default=256)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--llm-name", default="qwen3-32b-judge")
    parser.add_argument("--embedding-name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--normalize", type=string_to_bool, default=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def export_control_pool(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = str(args.dataset)
    dataset_file_stem = resolve_dataset_file_stem(dataset)
    samples_path = Path(args.data_root) / f"{dataset_file_stem}.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]

    queries = [str(sample["question"]) for sample in samples]
    gold_docs = get_gold_docs(samples, dataset)
    gold_answers = get_gold_answers(samples)
    chunk_path = resolve_chunk_embedding_path(
        save_dir=Path(args.save_dir),
        dataset=dataset,
        llm_name=str(args.llm_name),
        embedding_name=str(args.embedding_name),
    )
    chunk_df = pd.read_parquet(chunk_path)
    docs = [str(item) for item in chunk_df["content"].tolist()]
    doc_embeddings = np.stack(chunk_df["embedding"].to_numpy()).astype(np.float32)
    query_embeddings = embed_queries(
        queries,
        model_name=str(args.embedding_name),
        base_url=str(args.embedding_base_url),
        batch_size=int(args.embedding_batch_size),
    )
    if bool(args.normalize):
        doc_embeddings = normalize_rows(doc_embeddings)
        query_embeddings = normalize_rows(query_embeddings)

    reference_records_by_query: dict[int, Mapping[str, Any]] = {}
    edge_count_policy = str(args.policy) in {
        "edge_count_matched_dense_doc_knn",
        "same_seed_edge_count_matched_dense_doc_knn",
    }

    if edge_count_policy:
        if args.reference_pool_json is None:
            raise ValueError(f"--reference-pool-json is required for {args.policy}")
        reference_payload = json.loads(Path(args.reference_pool_json).read_text(encoding="utf-8"))
        reference_records = list(reference_payload.get("records") or [])
        reference_records_by_query = {
            int(row.get("query_idx", row.get("query_index", idx))): row
            for idx, row in enumerate(reference_records)
        }
        if len(reference_records_by_query) < len(samples):
            raise ValueError(
                f"reference pool has {len(reference_records_by_query)} rows, "
                f"but {len(samples)} samples are requested"
            )

    if str(args.policy) == "dense_doc_knn":
        neighbors_by_doc = compute_dense_doc_knn(
            doc_embeddings,
            neighbor_k=int(args.neighbor_k),
            block_size=int(args.knn_block_size),
        )
    elif str(args.policy) == "degree_matched_shuffle":
        neighbors_by_doc = compute_shuffled_doc_neighbors(
            n_docs=len(docs),
            neighbor_k=int(args.neighbor_k),
            seed=int(args.shuffle_seed),
        )
    else:
        neighbors_by_doc = []

    pool_k = int(args.pool_k)
    dense_seed_k = max(int(args.dense_seed_k), int(args.prefix_k), 1)
    dense_rank_k = min(max(pool_k, dense_seed_k), len(docs))
    retrieved_doc_lists: List[List[str]] = []
    records: List[Dict[str, Any]] = []
    for query_idx, query_vec in enumerate(tqdm(query_embeddings, desc="Building control pools")):
        ref_agsto: Mapping[str, Any] = {}
        reference_edge_trace: Mapping[str, Any] = {}
        local_candidate_count = pool_k
        query_dense_seed_k = dense_seed_k
        if edge_count_policy:
            ref_row = reference_records_by_query[int(query_idx)]
            ref_agsto = dict(ref_row.get("agsto") or {})
            local_candidate_count = max(
                int(ref_agsto.get("candidate_count") or 0),
                len(ref_agsto.get("candidate_doc_indices") or []),
                pool_k,
            )
            query_dense_seed_k = max(
                len(ref_agsto.get("agsto_seed_doc_indices") or []),
                int(args.prefix_k),
                1,
            )
            dense_rank_k = min(max(local_candidate_count, query_dense_seed_k, pool_k), len(docs))
        dense_scores = doc_embeddings @ query_vec
        dense_rank = _top_indices(dense_scores, dense_rank_k)
        prefix = [int(idx) for idx in dense_rank[: int(args.prefix_k)]]
        seeds = [int(idx) for idx in dense_rank[:query_dense_seed_k]]
        query_neighbors_by_doc: Sequence[Sequence[int]] | Mapping[int, Sequence[int]] = neighbors_by_doc
        same_seed_policy = str(args.policy) == "same_seed_edge_count_matched_dense_doc_knn"
        reference_seed_docs: List[int] = []
        reference_candidate_docs: List[int] = []
        if edge_count_policy:
            reference_seed_docs = _reference_int_list(ref_agsto, "agsto_seed_doc_indices")
            reference_candidate_docs = _reference_int_list(ref_agsto, "candidate_doc_indices")
        if same_seed_policy and reference_seed_docs:
            seeds = ordered_unique(reference_seed_docs, limit=max(len(reference_seed_docs), 1))
            local_nodes = ordered_unique(
                [*reference_seed_docs, *reference_candidate_docs, *[int(idx) for idx in dense_rank]],
                limit=max(local_candidate_count, len(reference_seed_docs), pool_k),
            )
        elif edge_count_policy:
            local_nodes = [int(idx) for idx in dense_rank[:local_candidate_count]]
        if edge_count_policy:
            query_neighbors_by_doc, reference_edge_trace = compute_edge_count_matched_dense_neighbors(
                doc_embeddings,
                local_doc_indices=local_nodes,
                target_edge_count=int(ref_agsto.get("agsto_local_edge_count") or 0),
            )
        admitted = traverse_doc_graph(
            seed_docs=seeds,
            neighbors_by_doc=query_neighbors_by_doc,
            pool_k=pool_k,
            closure_hops=int(args.closure_hops),
        )
        pool_indices = ordered_unique([*prefix, *[idx for idx in admitted if idx not in set(prefix)], *dense_rank], limit=pool_k)
        pool_docs = [docs[int(idx)] for idx in pool_indices]
        pool_scores = [float(200 - rank) if rank < int(args.prefix_k) else float(pool_k - rank) for rank in range(len(pool_indices))]
        retrieved_doc_lists.append(pool_docs)
        records.append(
            {
                "query_idx": int(query_idx),
                "question": queries[query_idx],
                "gold_answers": list(gold_answers[query_idx]),
                "gold_docs": list(gold_docs[query_idx]),
                "gold_titles": [extract_title(doc) for doc in gold_docs[query_idx]],
                "pool_k": pool_k,
                "pool_docs": pool_docs,
                "pool_titles": [extract_title(doc) for doc in pool_docs],
                "pool_doc_scores": pool_scores,
                "pool_doc_ids": [int(idx) for idx in pool_indices],
                "agsto": {
                    "candidate_source": f"doc_transition_control_{args.policy}",
                    "candidate_order_policy": "dense_prefix_then_control_graph_tail_then_remaining_dense",
                    "source_prior_prefix_doc_indices": prefix,
                    "dense_seed_doc_indices": seeds,
                    "reference_seed_doc_indices": reference_seed_docs if edge_count_policy else None,
                    "reference_candidate_doc_indices": reference_candidate_docs if edge_count_policy else None,
                    "control_admitted_doc_indices": admitted,
                    "control_graph_tail_doc_indices": [int(idx) for idx in admitted if int(idx) not in set(prefix)],
                    "external_pool_doc_ids": [int(idx) for idx in pool_indices],
                    "external_pool_titles": [extract_title(doc) for doc in pool_docs],
                    "external_pool_scores": pool_scores,
                    "control_policy": str(args.policy),
                    "neighbor_k": int(args.neighbor_k),
                    "closure_hops": int(args.closure_hops),
                    "dense_seed_k": int(query_dense_seed_k),
                    "prefix_k": int(args.prefix_k),
                    "shuffle_seed": int(args.shuffle_seed) if str(args.policy) == "degree_matched_shuffle" else None,
                    "reference_pool_json": str(args.reference_pool_json) if args.reference_pool_json else None,
                    "reference_candidate_count": int(ref_agsto.get("candidate_count") or 0) if ref_agsto else None,
                    "reference_seed_count": len(ref_agsto.get("agsto_seed_doc_indices") or []) if ref_agsto else None,
                    "reference_edge_trace": dict(reference_edge_trace),
                    "uses_fact_certified_transitions": False,
                    "uses_dense_doc_knn_transitions": str(args.policy) in {
                        "dense_doc_knn",
                        "edge_count_matched_dense_doc_knn",
                        "same_seed_edge_count_matched_dense_doc_knn",
                    },
                    "uses_degree_matched_random_transitions": str(args.policy) == "degree_matched_shuffle",
                    "uses_query_level_edge_count_match": edge_count_policy,
                    "uses_reference_seed_doc_identities": same_seed_policy,
                },
            }
        )

    recall = compute_title_recall(gold_docs, retrieved_doc_lists, [5, 20, 100])
    return {
        "dataset": dataset,
        "limit": int(len(samples)),
        "pool_k": pool_k,
        "source": "evidencelink_doc_transition_control_pool",
        "chunk_embedding_path": str(chunk_path),
        "embedding_name": str(args.embedding_name),
        "embedding_base_url": str(args.embedding_base_url),
        "normalize": bool(args.normalize),
        "retrieval": {
            "input_method": f"evidencelink_{args.policy}_control_pool",
            "control_policy": str(args.policy),
            "recomputed_title_recall": recall,
            "dense_seed_k": int(dense_seed_k),
            "prefix_k": int(args.prefix_k),
            "neighbor_k": int(args.neighbor_k),
            "closure_hops": int(args.closure_hops),
            "reference_pool_json": str(args.reference_pool_json) if args.reference_pool_json else None,
            "strict_match_scope": (
                (
                    "per-query total local edge count, candidate count, and EvLink seed document identities"
                    if str(args.policy) == "same_seed_edge_count_matched_dense_doc_knn"
                    else "per-query total local edge count, candidate count, and seed count"
                )
                if edge_count_policy
                else None
            ),
        },
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = export_control_pool(args)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "dataset": payload.get("dataset"),
                "limit": payload.get("limit"),
                "pool_k": payload.get("pool_k"),
                "control_policy": payload.get("retrieval", {}).get("control_policy"),
                "retrieval": payload.get("retrieval", {}).get("recomputed_title_recall"),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
