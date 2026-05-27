#!/usr/bin/env python3
"""Phase 2.5 dense-grounding same-substrate mechanism test.

The default backend is deterministic TF-IDF cosine. It is intentionally soft
and dense enough to test whether noisy-OR separates from channel-sum before
investing in external embedding-service runs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Literal, Sequence
import urllib.error
import urllib.request

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.data import normalize_text
from src.dpathrag.daec_dapg.graph import build_absorption_graph
from src.dpathrag.daec_dapg.local import (
    build_local_absorption_inputs,
    documents_from_pool_record,
    extract_demands,
)
from src.dpathrag.daec_dapg.metrics import noise_rate, summarize_numeric_rows
from src.dpathrag.daec_dapg.projection import greedy_noisy_or_select
from src.dpathrag.daec_dapg.propagation import finite_horizon_absorption, support_tensor
from src.dpathrag.io import read_json, write_json, write_jsonl
from src.dpathrag.metrics import recall_at_k, support_complete_at_k


VARIANTS = [
    "dense_query_cosine",
    "dense_demand_union_cosine",
    "query_level_ppr",
    "mixed_source_absorption",
    "channel_cross_talk",
    "demand_source_union",
    "multi_channel_sum",
    "multi_channel_noisy_or",
]

EmbeddingBackend = Literal["tfidf", "api"]
PhiTransform = Literal["linear_clip", "exp"]


def load_pool(path: str, limit: int) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        data = list(data.get("records") or data.get("rows") or data.get("data") or [])
    rows = list(data)
    return rows[: int(limit)] if int(limit) > 0 else rows


def attach_requirements_from_report(records: list[dict[str, Any]], report_path: str) -> list[dict[str, Any]]:
    """Attach DAEC-L1 demand decomposition traces to pool records by index."""

    if not report_path:
        return records
    report = read_json(report_path)
    traces = list(report.get("setwise_selector_query_traces") or [])
    enriched: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        row = dict(record)
        trace = traces[idx] if idx < len(traces) else {}
        selector_trace = trace.get("selector_trace") or {}
        requirements = selector_trace.get("requirements") or []
        if requirements:
            row["requirements"] = requirements
        row["hop_bucket"] = trace.get("gold_doc_count") or row.get("hop_bucket")
        enriched.append(row)
    return enriched


def toks(text: Any) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", normalize_text(text)) if len(tok) > 1]


def tfidf_vectors(texts: Sequence[str]) -> np.ndarray:
    tokenized = [toks(text) for text in texts]
    vocab = sorted({tok for row in tokenized for tok in row})
    if not vocab:
        return np.zeros((len(texts), 1), dtype=float)
    index = {tok: idx for idx, tok in enumerate(vocab)}
    df = Counter(tok for row in tokenized for tok in set(row))
    n = max(1, len(texts))
    matrix = np.zeros((len(texts), len(vocab)), dtype=float)
    for row_idx, row in enumerate(tokenized):
        counts = Counter(row)
        total = max(1, sum(counts.values()))
        for tok, count in counts.items():
            idf = math.log((n + 1.0) / (df[tok] + 1.0)) + 1.0
            matrix[row_idx, index[tok]] = (count / total) * idf
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return matrix / norm


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(float)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return matrix / norm


class ApiEmbeddingClient:
    """Small OpenAI-compatible embedding client with in-process text cache."""

    def __init__(self, *, base_url: str, model: str, batch_size: int, timeout: float) -> None:
        self.base_url = str(base_url)
        self.model = str(model)
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        normalized_texts = [str(text).replace("\n", " ") or " " for text in texts]
        missing = list(dict.fromkeys(text for text in normalized_texts if text not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._request_batch(batch)
            if len(vectors) != len(batch):
                raise ValueError(f"embedding endpoint returned {len(vectors)} vectors for {len(batch)} inputs")
            for text, vector in zip(batch, vectors):
                self.cache[text] = vector
        return l2_normalize(np.vstack([self.cache[text] for text in normalized_texts])) if normalized_texts else np.zeros((0, 0), dtype=float)

    def _request_batch(self, batch: Sequence[str]) -> list[np.ndarray]:
        payload = json.dumps({"model": self.model, "input": list(batch)}).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"embedding endpoint HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"embedding endpoint request failed: {exc}") from exc
        data = json.loads(body)
        rows = sorted(data.get("data") or [], key=lambda item: int(item.get("index", 0)))
        return [np.asarray(row["embedding"], dtype=float) for row in rows]


def encode_texts(texts: Sequence[str], *, embedding_backend: EmbeddingBackend, api_client: ApiEmbeddingClient | None) -> np.ndarray:
    if embedding_backend == "tfidf":
        return tfidf_vectors(texts)
    if api_client is None:
        raise ValueError("api_client is required when embedding_backend='api'")
    return api_client.embed(texts)


def cosine_rows(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.clip(matrix @ query_vec, 0.0, 1.0)


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one numeric value")
    return [float(item) for item in items]


def parse_variant_list(value: str) -> list[str]:
    if not value:
        return list(VARIANTS)
    variants = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; valid variants: {VARIANTS}")
    return variants


def calibrate_phi(phi: np.ndarray, *, scale: float, transform: PhiTransform) -> np.ndarray:
    values = np.clip(np.asarray(phi, dtype=float), 0.0, 1.0)
    if transform == "linear_clip":
        return np.clip(values * float(scale), 0.0, 1.0)
    if transform == "exp":
        return np.clip(1.0 - np.exp(-float(scale) * values), 0.0, 1.0)
    raise ValueError(f"Unsupported phi transform: {transform}")


def dense_phi(
    demand_texts: Sequence[str],
    record: dict[str, Any],
    *,
    alpha: float,
    horizon: int,
    max_docs: int,
    embedding_backend: EmbeddingBackend,
    api_client: ApiEmbeddingClient | None,
) -> tuple[np.ndarray, list[str]]:
    docs = documents_from_pool_record(record, max_docs=int(max_docs))
    if not docs:
        return np.zeros((len(demand_texts), 0), dtype=float), []
    _, _, node_types, edges, node_text = build_local_absorption_inputs(docs)
    graph = build_absorption_graph(node_types=node_types, edges=edges)
    doc_texts = [doc.full_text for doc in docs]
    node_texts = [node_text[node] for node in graph.transient_nodes]
    all_texts = list(demand_texts) + node_texts + doc_texts
    vectors = encode_texts(all_texts, embedding_backend=embedding_backend, api_client=api_client)
    demand_vecs = vectors[: len(demand_texts)]
    node_vecs = vectors[len(demand_texts) : len(demand_texts) + len(node_texts)]
    doc_vecs = vectors[len(demand_texts) + len(node_texts) :]
    rows: list[np.ndarray] = []
    for demand_vec in demand_vecs:
        source_u = cosine_rows(demand_vec, node_vecs) if len(node_texts) else np.zeros(0, dtype=float)
        hit = finite_horizon_absorption(
            graph.q,
            graph.r,
            source_u,
            np.zeros(len(doc_texts), dtype=float),
            alpha=float(alpha),
            horizon=int(horizon),
        )
        kappa = cosine_rows(demand_vec, doc_vecs)
        rows.append(support_tensor(kappa, hit))
    return np.vstack(rows) if rows else np.zeros((0, len(docs)), dtype=float), [doc.title for doc in docs]


def dense_cosine_phi(
    query_texts: Sequence[str],
    record: dict[str, Any],
    *,
    max_docs: int,
    embedding_backend: EmbeddingBackend,
    api_client: ApiEmbeddingClient | None,
    aggregation: Literal["single", "max_union"] = "single",
) -> tuple[np.ndarray, list[str]]:
    """Direct dense retrieval over documents, without graph absorption."""

    docs = documents_from_pool_record(record, max_docs=int(max_docs))
    if not docs:
        return np.zeros((1, 0), dtype=float), []
    doc_texts = [doc.full_text for doc in docs]
    texts = list(query_texts) + doc_texts
    vectors = encode_texts(texts, embedding_backend=embedding_backend, api_client=api_client)
    query_vecs = vectors[: len(query_texts)]
    doc_vecs = vectors[len(query_texts) :]
    if aggregation == "max_union" and len(query_vecs):
        scores = np.vstack([cosine_rows(query_vec, doc_vecs) for query_vec in query_vecs]).max(axis=0)
    else:
        query_vec = query_vecs[0] if len(query_vecs) else np.zeros(doc_vecs.shape[1], dtype=float)
        scores = cosine_rows(query_vec, doc_vecs)
    return scores.reshape(1, -1), [doc.title for doc in docs]


def dense_cross_talk_phi(
    source_texts: Sequence[str],
    gate_text: str,
    record: dict[str, Any],
    *,
    alpha: float,
    horizon: int,
    max_docs: int,
    embedding_backend: EmbeddingBackend,
    api_client: ApiEmbeddingClient | None,
    gate_mode: Literal["joint", "source_union"] = "joint",
) -> tuple[np.ndarray, list[str]]:
    """Single-channel absorption with unioned per-demand source evidence."""

    docs = documents_from_pool_record(record, max_docs=int(max_docs))
    if not docs:
        return np.zeros((1, 0), dtype=float), []
    _, _, node_types, edges, node_text = build_local_absorption_inputs(docs)
    graph = build_absorption_graph(node_types=node_types, edges=edges)
    doc_texts = [doc.full_text for doc in docs]
    node_texts = [node_text[node] for node in graph.transient_nodes]
    texts = list(source_texts) + [gate_text] + node_texts + doc_texts
    vectors = encode_texts(texts, embedding_backend=embedding_backend, api_client=api_client)
    source_vecs = vectors[: len(source_texts)]
    gate_vec = vectors[len(source_texts)]
    node_start = len(source_texts) + 1
    node_vecs = vectors[node_start : node_start + len(node_texts)]
    doc_vecs = vectors[node_start + len(node_texts) :]

    if len(source_vecs) and len(node_texts):
        source_rows = np.vstack([cosine_rows(source_vec, node_vecs) for source_vec in source_vecs])
        source_u = source_rows.max(axis=0)
    else:
        source_rows = np.zeros((0, len(node_texts)), dtype=float)
        source_u = np.zeros(len(node_texts), dtype=float)
    hit = finite_horizon_absorption(
        graph.q,
        graph.r,
        source_u,
        np.zeros(len(doc_texts), dtype=float),
        alpha=float(alpha),
        horizon=int(horizon),
    )
    if gate_mode == "source_union" and len(source_vecs):
        kappa = np.vstack([cosine_rows(source_vec, doc_vecs) for source_vec in source_vecs]).max(axis=0)
    else:
        kappa = cosine_rows(gate_vec, doc_vecs)
    return support_tensor(kappa, hit).reshape(1, -1), [doc.title for doc in docs]


def select_from_phi(phi: np.ndarray, titles: list[str], *, variant: str, top_k: int) -> tuple[list[str], float]:
    if phi.size == 0:
        return [], 0.0
    if variant in {
        "dense_query_cosine",
        "dense_demand_union_cosine",
        "query_level_ppr",
        "mixed_source_absorption",
        "channel_cross_talk",
        "demand_source_union",
        "multi_channel_sum",
    }:
        scores = phi.mean(axis=0)
        indices = list(np.argsort(-scores)[: int(top_k)])
        return [titles[idx] for idx in indices], float(np.sum(scores[indices])) if indices else 0.0
    indices, objective = greedy_noisy_or_select(phi, budget=int(top_k))
    return [titles[idx] for idx in indices], float(objective)


def hop_bucket(record: dict[str, Any]) -> str:
    if record.get("hop_bucket") is not None:
        return str(record["hop_bucket"])
    gold_titles = list(record.get("gold_titles") or [])
    return str(len(gold_titles)) if gold_titles else "unknown"


def run(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    top_k: int,
    max_docs: int,
    alpha: float,
    horizon: int,
    embedding_backend: EmbeddingBackend,
    api_client: ApiEmbeddingClient | None,
    phi_scales: Sequence[float],
    phi_transform: PhiTransform,
    variants: Sequence[str] = VARIANTS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scales = list(phi_scales)
    for idx, record in enumerate(records):
        demands = extract_demands(record)
        demand_sets = {
            "dense_query_cosine": [str(record.get("question") or "")],
            "dense_demand_union_cosine": demands,
            "query_level_ppr": [str(record.get("question") or "")],
            "mixed_source_absorption": [" ".join(demands)],
            "channel_cross_talk": demands,
            "demand_source_union": demands,
            "multi_channel_sum": demands,
            "multi_channel_noisy_or": demands,
        }
        gold_titles = list(record.get("gold_titles") or [])
        phi_cache: dict[tuple[str, ...], tuple[np.ndarray, list[str]]] = {}
        for variant in variants:
            cache_prefix = (
                variant
                if variant in {"dense_query_cosine", "dense_demand_union_cosine", "channel_cross_talk", "demand_source_union"}
                else "standard"
            )
            demand_key = (cache_prefix, *tuple(demand_sets[variant]))
            if demand_key not in phi_cache:
                if variant in {"dense_query_cosine", "dense_demand_union_cosine"}:
                    phi_cache[demand_key] = dense_cosine_phi(
                        demand_sets[variant],
                        record,
                        max_docs=max_docs,
                        embedding_backend=embedding_backend,
                        api_client=api_client,
                        aggregation="max_union" if variant == "dense_demand_union_cosine" else "single",
                    )
                elif variant in {"channel_cross_talk", "demand_source_union"}:
                    phi_cache[demand_key] = dense_cross_talk_phi(
                        demand_sets[variant],
                        " AND ".join(demands),
                        record,
                        alpha=alpha,
                        horizon=horizon,
                        max_docs=max_docs,
                        embedding_backend=embedding_backend,
                        api_client=api_client,
                        gate_mode="source_union" if variant == "demand_source_union" else "joint",
                    )
                else:
                    phi_cache[demand_key] = dense_phi(
                        demand_sets[variant],
                        record,
                        alpha=alpha,
                        horizon=horizon,
                        max_docs=max_docs,
                        embedding_backend=embedding_backend,
                        api_client=api_client,
                    )
            phi, titles = phi_cache[demand_key]
            for scale in scales:
                adjusted_phi = calibrate_phi(phi, scale=float(scale), transform=phi_transform)
                selected_titles, objective = select_from_phi(adjusted_phi, titles, variant=variant, top_k=int(top_k))
                selected_gold = len({normalize_text(title) for title in selected_titles} & {normalize_text(title) for title in gold_titles})
                rows.append(
                    {
                        "qid": str(record.get("qid") or record.get("id") or record.get("query_idx") or idx),
                        "dataset": dataset,
                        "hop_bucket": hop_bucket(record),
                        "variant": variant,
                        "embedding_backend": embedding_backend,
                        "phi_transform": phi_transform,
                        "phi_scale": float(scale),
                        "demand_count": len(demand_sets[variant]),
                        "raw_phi_max": float(np.max(phi)) if phi.size else 0.0,
                        "raw_phi_nonzero_rate": float(np.mean(phi > 1e-12)) if phi.size else 0.0,
                        "phi_max": float(np.max(adjusted_phi)) if adjusted_phi.size else 0.0,
                        "phi_nonzero_rate": float(np.mean(adjusted_phi > 1e-12)) if adjusted_phi.size else 0.0,
                        "selected_titles": selected_titles,
                        "selected_doc_count": len(selected_titles),
                        "support_recall": recall_at_k(gold_titles, selected_titles, int(top_k)),
                        "support_complete": support_complete_at_k(gold_titles, selected_titles, int(top_k)),
                        "selected_gold_count": selected_gold,
                        "noise_rate": noise_rate(selected_gold, len(selected_titles)),
                        "objective": objective,
                        "em": 0.0,
                        "f1": 0.0,
                        "latency_ms": 0.0,
                    }
                )
    return rows


def summarize_sum_vs_noisy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, float, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["variant"] in {"multi_channel_sum", "multi_channel_noisy_or"}:
            by_key[(str(row["hop_bucket"]), float(row.get("phi_scale", 1.0)), str(row["qid"]))][str(row["variant"])] = row

    grouped: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    for (hop, scale, _qid), pair in by_key.items():
        sum_row = pair.get("multi_channel_sum")
        noisy_row = pair.get("multi_channel_noisy_or")
        if not sum_row or not noisy_row:
            continue
        sum_set = {normalize_text(title) for title in sum_row.get("selected_titles") or []}
        noisy_set = {normalize_text(title) for title in noisy_row.get("selected_titles") or []}
        union = sum_set | noisy_set
        grouped[(hop, scale)].append(
            {
                "selection_diff": float(sum_set != noisy_set),
                "selection_jaccard": float(len(sum_set & noisy_set) / len(union)) if union else 1.0,
                "noisy_minus_sum_recall": float(noisy_row["support_recall"]) - float(sum_row["support_recall"]),
                "noisy_minus_sum_complete": float(noisy_row["support_complete"]) - float(sum_row["support_complete"]),
                "noisy_minus_sum_noise": float(noisy_row["noise_rate"]) - float(sum_row["noise_rate"]),
                "noisy_minus_sum_objective": float(noisy_row["objective"]) - float(sum_row["objective"]),
            }
        )

    metrics = [
        "selection_diff",
        "selection_jaccard",
        "noisy_minus_sum_recall",
        "noisy_minus_sum_complete",
        "noisy_minus_sum_noise",
        "noisy_minus_sum_objective",
    ]
    return {
        f"{hop}:scale={scale:g}": summarize_numeric_rows(items, metrics) | {"rows": len(items), "phi_scale": scale}
        for (hop, scale), items in sorted(grouped.items())
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["hop_bucket"]), float(row.get("phi_scale", 1.0)), str(row["variant"]))].append(row)
    metrics = ["support_recall", "support_complete", "noise_rate", "objective", "em", "f1", "latency_ms"]
    return {
        "rows": len(rows),
        "by_hop_variant": {
            f"{hop}:scale={scale:g}:{variant}": summarize_numeric_rows(items, metrics) | {"rows": len(items), "phi_scale": scale}
            for (hop, scale, variant), items in sorted(grouped.items())
        },
    }


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# DAEC-DAPG Phase 2.5 Dense-Grounding Mechanism Test",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Backend: `{payload['embedding_backend']}`",
        f"- Phi transform: `{payload['phi_transform']}`",
        f"- Phi scales: `{payload['phi_scales']}`",
        f"- Rows: `{payload['summary']['rows']}`",
        "",
        "| Hop:Variant | Rows | Support Recall | Support Complete | Noise Rate | Objective |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in payload["summary"]["by_hop_variant"].items():
        lines.append(
            f"| {key} | {value['rows']} | {value['support_recall']:.4f} | {value['support_complete']:.4f} | {value['noise_rate']:.4f} | {value['objective']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Sum vs Noisy-OR",
            "",
            "| Hop:Scale | Rows | Selection Diff Rate | Selection Jaccard | Δ Recall | Δ Complete | Δ Noise |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for hop, value in payload["summary"].get("sum_vs_noisy_or", {}).items():
        lines.append(
            f"| {hop} | {value['rows']} | {value['selection_diff']:.4f} | {value['selection_jaccard']:.4f} | "
            f"{value['noisy_minus_sum_recall']:.4f} | {value['noisy_minus_sum_complete']:.4f} | {value['noisy_minus_sum_noise']:.4f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--daec_report_json", default="")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_docs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--embedding_backend", choices=["tfidf", "api"], default="tfidf")
    parser.add_argument("--embedding_model", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_timeout", type=float, default=120.0)
    parser.add_argument("--phi_transform", choices=["linear_clip", "exp"], default="linear_clip")
    parser.add_argument("--phi_scales", default="1.0")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output_json", default="reports/dpathrag/daec_dapg_phase25_dense_grounding_20260428.json")
    parser.add_argument("--rows_jsonl", default="reports/dpathrag/daec_dapg_phase25_dense_grounding_rows_20260428.jsonl")
    parser.add_argument("--output_md", default="reports/dpathrag/daec_dapg_phase25_dense_grounding_20260428.md")
    args = parser.parse_args()

    records = load_pool(args.pool_json, int(args.limit))
    records = attach_requirements_from_report(records, str(args.daec_report_json))
    phi_scales = parse_float_list(str(args.phi_scales))
    variants = parse_variant_list(str(args.variants))
    api_client = None
    if args.embedding_backend == "api":
        api_client = ApiEmbeddingClient(
            base_url=str(args.embedding_base_url),
            model=str(args.embedding_model),
            batch_size=int(args.embedding_batch_size),
            timeout=float(args.embedding_timeout),
        )
    rows = run(
        records,
        dataset=str(args.dataset),
        top_k=int(args.top_k),
        max_docs=int(args.max_docs),
        alpha=float(args.alpha),
        horizon=int(args.horizon),
        embedding_backend=args.embedding_backend,
        api_client=api_client,
        phi_scales=phi_scales,
        phi_transform=args.phi_transform,
        variants=variants,
    )
    payload = {
        "pool_json": str(args.pool_json),
        "daec_report_json": str(args.daec_report_json),
        "dataset": str(args.dataset),
        "limit": int(args.limit),
        "top_k": int(args.top_k),
        "max_docs": int(args.max_docs),
        "alpha": float(args.alpha),
        "horizon": int(args.horizon),
        "embedding_backend": str(args.embedding_backend),
        "embedding_model": str(args.embedding_model) if args.embedding_backend == "api" else "",
        "embedding_base_url": str(args.embedding_base_url) if args.embedding_backend == "api" else "",
        "phi_transform": str(args.phi_transform),
        "phi_scales": phi_scales,
        "variants": variants,
        "summary": summarize(rows) | {"sum_vs_noisy_or": summarize_sum_vs_noisy(rows)},
    }
    write_jsonl(rows, args.rows_jsonl)
    write_json(payload, args.output_json)
    write_markdown(payload, args.output_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
