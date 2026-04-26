#!/usr/bin/env python3
"""Build compressed q-doc embedding features for D-PathRAG selector training."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import ensure_parent, write_json


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= int(limit):
                    break
    return rows


def embed_texts(
    texts: Sequence[str],
    *,
    model: str,
    base_url: str,
    batch_size: int,
    timeout: int,
) -> "Any":
    import numpy as np
    import requests
    from tqdm import tqdm

    outputs: list[Any] = []
    headers = {"Content-Type": "application/json"}
    model_id = model[len("VLLM/") :] if model.startswith("VLLM/") else model
    for start in tqdm(range(0, len(texts), int(batch_size)), desc="Embedding selector texts"):
        batch = [str(text) if str(text) else " " for text in texts[start : start + int(batch_size)]]
        response = requests.post(
            base_url,
            headers=headers,
            json={"model": model_id, "input": batch},
            timeout=int(timeout),
        )
        response.raise_for_status()
        payload = response.json()
        outputs.append(np.asarray([row["embedding"] for row in payload["data"]], dtype=np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def normalize_rows(array: "Any") -> "Any":
    import numpy as np

    denom = np.linalg.norm(array, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return array / denom


def random_projection_matrix(input_dim: int, output_dim: int, *, seed: int) -> "Any":
    import numpy as np

    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(loc=0.0, scale=1.0 / math.sqrt(float(output_dim)), size=(int(input_dim), int(output_dim)))
    return matrix.astype(np.float32)


def build_projection_features(
    rows: list[dict[str, Any]],
    *,
    embeddings_by_text: dict[str, "Any"],
    projection: "Any",
    max_candidates: int,
) -> tuple["Any", "Any", list[str]]:
    import numpy as np

    projection_dim = int(projection.shape[1])
    semantic_dim = projection_dim * 3 + 1
    candidate_features = np.zeros((len(rows), int(max_candidates), semantic_dim), dtype=np.float32)
    query_features = np.zeros((len(rows), semantic_dim), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        question = str(row.get("question") or "")
        query_text = f"query: {question}"
        q_vec = embeddings_by_text[query_text] @ projection
        q_vec = normalize_rows(q_vec.reshape(1, -1))[0]
        query_features[row_idx, :projection_dim] = q_vec
        query_features[row_idx, projection_dim : projection_dim * 2] = 0.0
        query_features[row_idx, projection_dim * 2 : projection_dim * 3] = q_vec * q_vec
        query_features[row_idx, -1] = 1.0
        for cand_idx, candidate in enumerate(list(row.get("candidates") or [])[: int(max_candidates)]):
            doc_text = f"passage: {candidate.get('title') or ''}\n{candidate.get('text') or ''}"
            d_vec = embeddings_by_text[doc_text] @ projection
            d_vec = normalize_rows(d_vec.reshape(1, -1))[0]
            cosine = float(np.dot(q_vec, d_vec))
            candidate_features[row_idx, cand_idx, :projection_dim] = d_vec
            candidate_features[row_idx, cand_idx, projection_dim : projection_dim * 2] = np.abs(q_vec - d_vec)
            candidate_features[row_idx, cand_idx, projection_dim * 2 : projection_dim * 3] = q_vec * d_vec
            candidate_features[row_idx, cand_idx, -1] = cosine
    feature_names = (
        [f"doc_embed_rp_{idx}" for idx in range(projection_dim)]
        + [f"abs_q_doc_rp_{idx}" for idx in range(projection_dim)]
        + [f"prod_q_doc_rp_{idx}" for idx in range(projection_dim)]
        + ["q_doc_cosine_rp"]
    )
    return candidate_features, query_features, feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--manifest_json", default="")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--embedding_model", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    import numpy as np

    rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    unique_texts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        query_text = f"query: {row.get('question') or ''}"
        if query_text not in seen:
            seen.add(query_text)
            unique_texts.append(query_text)
        for candidate in list(row.get("candidates") or [])[: int(args.max_candidates)]:
            doc_text = f"passage: {candidate.get('title') or ''}\n{candidate.get('text') or ''}"
            if doc_text not in seen:
                seen.add(doc_text)
                unique_texts.append(doc_text)

    embeddings = embed_texts(
        unique_texts,
        model=str(args.embedding_model),
        base_url=str(args.embedding_base_url),
        batch_size=int(args.embedding_batch_size),
        timeout=int(args.timeout),
    )
    embeddings = normalize_rows(embeddings)
    projection = random_projection_matrix(embeddings.shape[1], int(args.projection_dim), seed=int(args.seed))
    embeddings_by_text = {text: embeddings[idx] for idx, text in enumerate(unique_texts)}
    candidate_features, query_features, feature_names = build_projection_features(
        rows,
        embeddings_by_text=embeddings_by_text,
        projection=projection,
        max_candidates=int(args.max_candidates),
    )
    ensure_parent(args.output_npz)
    np.savez_compressed(
        args.output_npz,
        qids=np.asarray([str(row.get("qid") or row.get("query_idx") or "") for row in rows], dtype=object),
        candidate_features=candidate_features,
        query_features=query_features,
        feature_names=np.asarray(feature_names, dtype=object),
        projection_dim=np.asarray([int(args.projection_dim)], dtype=np.int32),
    )
    manifest = {
        "cache_jsonl": str(args.cache_jsonl),
        "output_npz": str(args.output_npz),
        "rows": len(rows),
        "unique_texts": len(unique_texts),
        "max_candidates": int(args.max_candidates),
        "embedding_model": str(args.embedding_model),
        "embedding_base_url": str(args.embedding_base_url),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "projection_dim": int(args.projection_dim),
        "semantic_feature_dim": int(candidate_features.shape[-1]),
    }
    if args.manifest_json:
        write_json(manifest, args.manifest_json)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
