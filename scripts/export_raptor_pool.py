#!/usr/bin/env python3
"""Export RAPTOR retrieval pools under the aligned comparison protocol.

The official RAPTOR implementation retrieves hierarchical summary nodes.  The
paper protocol used in this repository evaluates readers over natural-language
passages, so this adapter builds a RAPTOR tree over corpus passages, retrieves
tree nodes, projects selected summary nodes back to their descendant leaf
passages, and writes the standard external-pool JSON consumed by the shared
GPT-4o-mini reader.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
import tiktoken
from openai import OpenAI
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path("reproduce/dataset")
DEFAULT_SAVE_DIR = Path("outputs_step0_general_nvembed")
DEFAULT_RAPTOR_ROOT = Path("/mnt/nvme/code/RAPTOR")
DATASET_FILE_ALIASES = {
    "nq": "nq_rear",
    "natural_questions": "nq_rear",
}
NO_THINK_PREFIX = "/no_think"


def resolve_dataset_file_stem(dataset_name: str | None) -> str:
    normalized = str(dataset_name or "").strip().lower()
    return DATASET_FILE_ALIASES.get(normalized, str(dataset_name or "").strip())


def display_dataset_name(dataset_name: str | None) -> str:
    normalized = str(dataset_name or "").strip()
    return "nq" if normalized in {"nq", "nq_rear", "natural_questions"} else normalized


def extract_title(doc: str) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def normalize_doc_text(doc: str) -> str:
    return re.sub(r"\s+", " ", str(doc or "")).strip()


def string_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_answer_alias_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                return [cleaned]
            return parse_answer_alias_values(parsed)
        return [cleaned]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        aliases: List[str] = []
        for item in value:
            aliases.extend(parse_answer_alias_values(item))
        return aliases
    return [str(value)]


def get_gold_docs(samples: Sequence[Dict[str, Any]], dataset_name: str) -> List[List[str]]:
    gold_docs: List[List[str]] = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            gold_title_and_content = [item for item in sample["context"] if item[0] in gold_titles]
            if dataset_name.startswith("hotpotqa"):
                docs = [item[0] + "\n" + "".join(item[1]) for item in gold_title_and_content]
            else:
                docs = [item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content]
        elif "contexts" in sample:
            docs = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item.get("is_supporting")
            ]
        else:
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting") is not False]
            docs = [
                item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"])
                for item in paragraphs
            ]
        gold_docs.append(sorted(set(docs)))
    return gold_docs


def get_gold_answers(samples: Sequence[Dict[str, Any]]) -> List[List[str]]:
    gold_answers: List[List[str]] = []
    for sample in samples:
        if "answer" in sample or "gold_ans" in sample:
            answer = sample["answer"] if "answer" in sample else sample["gold_ans"]
            answers = parse_answer_alias_values(answer)
        elif "reference" in sample:
            answers = parse_answer_alias_values(sample["reference"])
        elif "obj" in sample:
            answers = []
            answers.extend(parse_answer_alias_values(sample.get("obj")))
            answers.extend(parse_answer_alias_values(sample.get("possible_answers")))
            answers.extend(parse_answer_alias_values(sample.get("o_wiki_title")))
            answers.extend(parse_answer_alias_values(sample.get("o_aliases")))
        else:
            raise ValueError("Sample has no recognized answer field.")
        if "answer_aliases" in sample:
            answers.extend(parse_answer_alias_values(sample["answer_aliases"]))
        gold_answers.append(sorted({str(item).strip() for item in answers if str(item).strip()}))
    return gold_answers


def compute_title_recall(
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
    k_values: Iterable[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in k_values:
        scores: List[float] = []
        for gold, retrieved in zip(gold_docs, retrieved_docs):
            gold_titles = {extract_title(doc) for doc in gold}
            retrieved_titles = {extract_title(doc) for doc in list(retrieved)[: int(k)]}
            scores.append(0.0 if not gold_titles else len(gold_titles & retrieved_titles) / len(gold_titles))
        metrics[f"Recall@{int(k)}"] = round(float(np.mean(scores)) if scores else 0.0, 4)
    return metrics


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return matrix / norms


def resolve_chunk_embedding_path(save_dir: Path, dataset: str, llm_name: str, embedding_name: str) -> Path:
    suffix = f"{llm_name}_{embedding_name.replace('/', '_')}"
    dataset_file_stem = resolve_dataset_file_stem(dataset)
    candidates = [
        Path(f"{save_dir}_{dataset}") / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
        Path(f"{save_dir}_{dataset_file_stem}") / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
        save_dir / f"{dataset}_{suffix}" / "chunk_embeddings" / "vdb_chunk.parquet",
        save_dir / f"{dataset_file_stem}_{suffix}" / "chunk_embeddings" / "vdb_chunk.parquet",
        save_dir / f"{dataset}" / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
        save_dir / f"{dataset_file_stem}" / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
        Path(f"outputs_step0_general_nvembed_{dataset}") / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
        Path(f"outputs_step0_general_nvembed_{dataset_file_stem}") / suffix / "chunk_embeddings" / "vdb_chunk.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find chunk embedding parquet. Tried:\n" + "\n".join(str(p) for p in candidates))


def load_docs_from_corpus(corpus_path: Path) -> List[str]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    docs: List[str] = []
    for item in corpus:
        if "title" in item and "text" in item:
            docs.append(str(item["title"]) + "\n" + str(item["text"]))
        elif "title" in item and "body_text" in item:
            docs.append(str(item["title"]) + "\n" + str(item["body_text"]))
        else:
            docs.append(str(item.get("text", item.get("body_text", ""))))
    return docs


def load_leaf_docs_and_embeddings(
    *,
    dataset: str,
    corpus_path: Path,
    save_dir: Path,
    llm_name: str,
    embedding_name: str,
    normalize: bool,
) -> tuple[List[str], np.ndarray, str]:
    try:
        chunk_path = resolve_chunk_embedding_path(save_dir, dataset, llm_name, embedding_name)
        chunk_df = pd.read_parquet(chunk_path)
        docs = [str(item) for item in chunk_df["content"].tolist()]
        embeddings = np.stack(chunk_df["embedding"].to_numpy()).astype(np.float32)
        source = str(chunk_path)
    except FileNotFoundError:
        docs = load_docs_from_corpus(corpus_path)
        embeddings = np.zeros((len(docs), 0), dtype=np.float32)
        source = str(corpus_path)
    if normalize and embeddings.size:
        embeddings = normalize_rows(embeddings)
    return docs, embeddings, source


class EmbeddingClient:
    def __init__(self, *, model_name: str, base_url: str, batch_size: int, timeout: float) -> None:
        self.model_name = model_name[len("VLLM/") :] if model_name.startswith("VLLM/") else model_name
        self.base_url = base_url
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)

    def embed(self, texts: Sequence[str], *, desc: str) -> np.ndarray:
        outputs: List[np.ndarray] = []
        headers = {"Content-Type": "application/json"}
        for start in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch = [str(item).replace("\n", " ") for item in texts[start : start + self.batch_size]]
            response = requests.post(
                self.base_url,
                headers=headers,
                json={"model": self.model_name, "input": batch},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            outputs.append(np.asarray([row["embedding"] for row in payload["data"]], dtype=np.float32))
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


class QwenNoThinkSummarizer:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        timeout: float,
        temperature: float,
        cache_path: Path,
    ) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=base_url)
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.cache: Dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("key") or "")
                value = str(row.get("summary") or "")
                if key and value:
                    self.cache[key] = value

    @staticmethod
    def strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", str(text or ""), flags=re.DOTALL).strip()

    def key_for(self, context: str, max_tokens: int) -> str:
        payload = json.dumps(
            {"context": context, "max_tokens": int(max_tokens), "model": self.model_name},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summarize(self, context: str, max_tokens: int) -> str:
        key = self.key_for(context, max_tokens)
        with self.lock:
            cached = self.cache.get(key)
        if cached:
            return cached

        prompt = (
            f"{NO_THINK_PREFIX}\n"
            "Write a concise, information-dense summary of the following passages. "
            "Keep named entities, relations, dates, and facts that may help answer questions. "
            f"Use at most {int(max_tokens)} tokens.\n\n"
            f"{context}"
        )
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You summarize evidence passages faithfully."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=int(max_tokens),
                    timeout=self.timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                summary = self.strip_thinking(response.choices[0].message.content or "")
                if not summary:
                    summary = normalize_doc_text(context)[:1200]
                with self.lock:
                    self.cache[key] = summary
                    with self.cache_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"key": key, "summary": summary}, ensure_ascii=False) + "\n")
                return summary
            except Exception as exc:  # pragma: no cover - endpoint resilience
                last_error = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Qwen summarization failed after retries: {last_error}")


def import_raptor(raptor_root: Path) -> tuple[Any, Any, Any]:
    if not raptor_root.exists():
        raise FileNotFoundError(f"RAPTOR root does not exist: {raptor_root}")
    sys.path.insert(0, str(raptor_root))
    import raptor.cluster_utils as cluster_utils
    from raptor.cluster_utils import RAPTOR_Clustering
    from raptor.tree_structures import Node, Tree
    from raptor.utils import get_text

    def stable_get_optimal_clusters(
        embeddings: np.ndarray,
        max_clusters: int = 50,
        random_state: int = cluster_utils.RANDOM_SEED,
    ) -> int:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        max_clusters = min(int(max_clusters), len(embeddings))
        if max_clusters <= 1:
            return 1
        candidate_counts = np.arange(1, max_clusters)
        bics: List[float] = []
        for n_clusters in candidate_counts:
            try:
                gm = cluster_utils.GaussianMixture(
                    n_components=int(n_clusters),
                    random_state=random_state,
                    reg_covar=1e-5,
                )
                gm.fit(embeddings)
                bics.append(float(gm.bic(embeddings)))
            except ValueError:
                bics.append(float("inf"))
        if not bics or not np.isfinite(bics).any():
            return 1
        return int(candidate_counts[int(np.argmin(bics))])

    def stable_gmm_cluster(
        embeddings: np.ndarray,
        threshold: float,
        random_state: int = 0,
    ) -> tuple[List[np.ndarray], int]:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        n_clusters = stable_get_optimal_clusters(embeddings)
        gm = cluster_utils.GaussianMixture(
            n_components=int(n_clusters),
            random_state=random_state,
            reg_covar=1e-5,
        )
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels: List[np.ndarray] = []
        for prob in probs:
            selected = np.where(prob > threshold)[0]
            if len(selected) == 0:
                selected = np.asarray([int(np.argmax(prob))])
            labels.append(selected)
        return labels, int(n_clusters)

    cluster_utils.get_optimal_clusters = stable_get_optimal_clusters
    cluster_utils.GMM_cluster = stable_gmm_cluster

    return RAPTOR_Clustering, Node, Tree, get_text


def node_layer_map(tree: Any) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for layer, nodes in tree.layer_to_nodes.items():
        for node in nodes:
            mapping[int(node.index)] = int(layer)
    return mapping


def build_tree_from_passages(
    *,
    docs: Sequence[str],
    doc_embeddings: np.ndarray,
    args: argparse.Namespace,
    cache_dir: Path,
) -> Any:
    RAPTOR_Clustering, Node, Tree, get_text = import_raptor(Path(args.raptor_root))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    embedding_key = "NV"
    summarizer = QwenNoThinkSummarizer(
        model_name=str(args.llm_name),
        base_url=str(args.llm_base_url),
        timeout=float(args.llm_timeout),
        temperature=float(args.llm_temperature),
        cache_path=cache_dir / "summary_cache.jsonl",
    )
    embedder = EmbeddingClient(
        model_name=str(args.embedding_name),
        base_url=str(args.embedding_base_url),
        batch_size=int(args.embedding_batch_size),
        timeout=float(args.embedding_timeout),
    )

    if not doc_embeddings.size:
        doc_embeddings = embedder.embed(docs, desc="Embedding RAPTOR leaves")
        doc_embeddings = normalize_rows(doc_embeddings)

    leaf_nodes = {
        int(idx): Node(str(text), int(idx), set(), {embedding_key: doc_embeddings[int(idx)].astype(np.float32)})
        for idx, text in enumerate(docs)
    }
    all_nodes = dict(leaf_nodes)
    layer_to_nodes = {0: list(leaf_nodes.values())}
    current_level_nodes = dict(leaf_nodes)
    next_node_index = len(all_nodes)

    for layer in range(int(args.num_layers)):
        current_nodes = [current_level_nodes[key] for key in sorted(current_level_nodes)]
        if len(current_nodes) <= int(args.reduction_dimension) + 1:
            print(f"Stopping RAPTOR tree at layer {layer}: only {len(current_nodes)} nodes remain.")
            break
        print(f"RAPTOR layer {layer}: clustering {len(current_nodes)} nodes")
        clusters = RAPTOR_Clustering.perform_clustering(
            current_nodes,
            embedding_key,
            max_length_in_cluster=int(args.max_length_in_cluster),
            tokenizer=tokenizer,
            reduction_dimension=int(args.reduction_dimension),
            threshold=float(args.cluster_threshold),
            verbose=bool(args.verbose_clustering),
        )
        print(f"RAPTOR layer {layer}: summarizing {len(clusters)} clusters")

        summaries: List[str] = [""] * len(clusters)
        with ThreadPoolExecutor(max_workers=int(args.summary_workers)) as executor:
            futures = {
                executor.submit(
                    summarizer.summarize,
                    get_text(cluster),
                    int(args.summarization_length),
                ): idx
                for idx, cluster in enumerate(clusters)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Summarizing layer {layer}"):
                idx = futures[future]
                summaries[idx] = str(future.result())

        parent_embeddings = embedder.embed(summaries, desc=f"Embedding layer {layer} summaries")
        parent_embeddings = normalize_rows(parent_embeddings)

        new_level_nodes: Dict[int, Any] = {}
        for cluster_idx, cluster in enumerate(clusters):
            node_idx = next_node_index
            next_node_index += 1
            children = {int(node.index) for node in cluster}
            new_level_nodes[node_idx] = Node(
                summaries[cluster_idx],
                node_idx,
                children,
                {embedding_key: parent_embeddings[cluster_idx].astype(np.float32)},
            )
        layer_to_nodes[layer + 1] = list(new_level_nodes.values())
        current_level_nodes = new_level_nodes
        all_nodes.update(new_level_nodes)
        print(f"RAPTOR layer {layer}: created {len(new_level_nodes)} parent nodes")

    num_layers = max(layer_to_nodes)
    root_nodes = {int(node.index): node for node in layer_to_nodes[num_layers]}
    return Tree(all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes)


def tree_cache_key(*, dataset: str, corpus_count: int, args: argparse.Namespace) -> str:
    payload = {
        "dataset": dataset,
        "corpus_count": int(corpus_count),
        "llm": str(args.llm_name),
        "embedding": str(args.embedding_name),
        "num_layers": int(args.num_layers),
        "reduction_dimension": int(args.reduction_dimension),
        "cluster_threshold": float(args.cluster_threshold),
        "max_length_in_cluster": int(args.max_length_in_cluster),
        "summarization_length": int(args.summarization_length),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{display_dataset_name(dataset)}_{digest}"


def load_or_build_tree(
    *,
    docs: Sequence[str],
    doc_embeddings: np.ndarray,
    args: argparse.Namespace,
    cache_root: Path,
) -> Any:
    cache_root.mkdir(parents=True, exist_ok=True)
    key = tree_cache_key(dataset=str(args.dataset), corpus_count=len(docs), args=args)
    cache_dir = cache_root / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    tree_path = cache_dir / "tree.pkl"
    meta_path = cache_dir / "meta.json"
    if tree_path.exists() and not bool(args.force_rebuild_tree):
        with tree_path.open("rb") as handle:
            tree = pickle.load(handle)
        print(json.dumps({"tree_cache": str(tree_path), "status": "loaded"}, indent=2))
        return tree

    tree = build_tree_from_passages(docs=docs, doc_embeddings=doc_embeddings, args=args, cache_dir=cache_dir)
    with tree_path.open("wb") as handle:
        pickle.dump(tree, handle)
    meta = {
        "dataset": str(args.dataset),
        "doc_count": len(docs),
        "tree_num_layers": int(tree.num_layers),
        "layer_sizes": {str(layer): len(nodes) for layer, nodes in tree.layer_to_nodes.items()},
        "config": {
            "llm_name": str(args.llm_name),
            "llm_base_url": str(args.llm_base_url),
            "embedding_name": str(args.embedding_name),
            "embedding_base_url": str(args.embedding_base_url),
            "num_layers": int(args.num_layers),
            "reduction_dimension": int(args.reduction_dimension),
            "cluster_threshold": float(args.cluster_threshold),
            "max_length_in_cluster": int(args.max_length_in_cluster),
            "summarization_length": int(args.summarization_length),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tree_cache": str(tree_path), "status": "built", "meta": meta}, indent=2))
    return tree


def descendant_leaves(tree: Any, node_index: int, memo: Dict[int, List[int]]) -> List[int]:
    if node_index in memo:
        return memo[node_index]
    node = tree.all_nodes[int(node_index)]
    if not node.children:
        memo[node_index] = [int(node.index)]
        return memo[node_index]
    leaves: List[int] = []
    for child in node.children:
        leaves.extend(descendant_leaves(tree, int(child), memo))
    deduped = sorted(set(leaves))
    memo[node_index] = deduped
    return deduped


def rank_leaf_pool(
    *,
    tree: Any,
    docs: Sequence[str],
    query_embedding: np.ndarray,
    node_embeddings: np.ndarray,
    node_indices: Sequence[int],
    layer_by_node: Mapping[int, int],
    pool_k: int,
    top_nodes: int,
    leaves_per_summary: int,
) -> tuple[List[int], List[float], Dict[str, Any]]:
    node_scores = node_embeddings @ query_embedding.astype(np.float32)
    top_count = min(int(top_nodes), len(node_scores))
    if top_count >= len(node_scores):
        ranked_node_positions = np.argsort(node_scores)[::-1]
    else:
        ranked_node_positions = np.argpartition(node_scores, -top_count)[-top_count:]
        ranked_node_positions = ranked_node_positions[np.argsort(node_scores[ranked_node_positions])[::-1]]

    leaf_embeddings = node_embeddings[: len(docs)]
    leaf_scores = leaf_embeddings @ query_embedding.astype(np.float32)
    leaf_memo: Dict[int, List[int]] = {}
    candidate_scores: Dict[int, float] = {}
    node_trace: List[Dict[str, Any]] = []
    for rank, pos in enumerate(ranked_node_positions):
        node_idx = int(node_indices[int(pos)])
        layer = int(layer_by_node.get(node_idx, 0))
        node_score = float(node_scores[int(pos)])
        leaves = descendant_leaves(tree, node_idx, leaf_memo)
        if len(leaves) > int(leaves_per_summary):
            leaf_array = np.asarray(leaves, dtype=np.int64)
            local_scores = leaf_scores[leaf_array]
            keep = min(int(leaves_per_summary), len(leaf_array))
            local_top = np.argpartition(local_scores, -keep)[-keep:]
            local_top = local_top[np.argsort(local_scores[local_top])[::-1]]
            projected_leaves = [int(leaf_array[i]) for i in local_top]
        else:
            projected_leaves = leaves
        node_trace.append(
            {
                "node_index": node_idx,
                "rank": int(rank + 1),
                "layer": layer,
                "score": node_score,
                "num_descendant_leaves": len(leaves),
                "num_projected_leaves": len(projected_leaves),
            }
        )
        for local_rank, leaf_idx in enumerate(projected_leaves):
            projected_score = node_score + 0.05 * float(leaf_scores[leaf_idx]) - 1e-5 * rank - 1e-6 * local_rank
            previous = candidate_scores.get(int(leaf_idx))
            if previous is None or projected_score > previous:
                candidate_scores[int(leaf_idx)] = float(projected_score)

    if len(candidate_scores) < int(pool_k):
        fill_order = np.argsort(leaf_scores)[::-1]
        for leaf_idx in fill_order:
            if int(leaf_idx) not in candidate_scores:
                candidate_scores[int(leaf_idx)] = float(leaf_scores[int(leaf_idx)]) - 10.0
            if len(candidate_scores) >= int(pool_k):
                break

    ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))[: int(pool_k)]
    return [int(idx) for idx, _ in ranked], [float(score) for _, score in ranked], {"selected_nodes": node_trace[:20]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--corpus-limit", type=int, default=0)
    parser.add_argument("--pool_k", type=int, default=200)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--raptor_root", type=Path, default=DEFAULT_RAPTOR_ROOT)
    parser.add_argument("--tree_cache_root", type=Path, default=Path("run_logs/raptor_tree_cache_32b_no_think_nvembed"))
    parser.add_argument("--llm_name", default="qwen3-32b-judge")
    parser.add_argument("--llm_base_url", default="http://localhost:8046/v1")
    parser.add_argument("--index_llm_name", default="qwen3-32b-judge")
    parser.add_argument("--llm_timeout", type=float, default=180.0)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--embedding_name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--embedding_timeout", type=float, default=180.0)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--reduction_dimension", type=int, default=10)
    parser.add_argument("--cluster_threshold", type=float, default=0.1)
    parser.add_argument("--max_length_in_cluster", type=int, default=3500)
    parser.add_argument("--summarization_length", type=int, default=120)
    parser.add_argument("--summary_workers", type=int, default=2)
    parser.add_argument("--top_nodes", type=int, default=80)
    parser.add_argument("--leaves_per_summary", type=int, default=10)
    parser.add_argument("--normalize", type=string_to_bool, default=True)
    parser.add_argument("--force_rebuild_tree", action="store_true")
    parser.add_argument("--verbose_clustering", action="store_true")
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    dataset_file_stem = resolve_dataset_file_stem(args.dataset)
    samples_path = args.data_root / f"{dataset_file_stem}.json"
    corpus_path = args.data_root / f"{dataset_file_stem}_corpus.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]
    queries = [str(sample["question"]) for sample in samples]
    gold_docs = get_gold_docs(samples, str(args.dataset))
    gold_answers = get_gold_answers(samples)

    docs, doc_embeddings, leaf_source = load_leaf_docs_and_embeddings(
        dataset=str(args.dataset),
        corpus_path=corpus_path,
        save_dir=Path(args.save_dir),
        llm_name=str(args.index_llm_name),
        embedding_name=str(args.embedding_name),
        normalize=bool(args.normalize),
    )
    if int(args.corpus_limit) > 0:
        docs = docs[: int(args.corpus_limit)]
        doc_embeddings = doc_embeddings[: int(args.corpus_limit)] if doc_embeddings.size else doc_embeddings

    tree = load_or_build_tree(docs=docs, doc_embeddings=doc_embeddings, args=args, cache_root=Path(args.tree_cache_root))
    layer_by_node = node_layer_map(tree)
    node_indices = sorted(int(idx) for idx in tree.all_nodes)
    node_embeddings = np.stack(
        [np.asarray(tree.all_nodes[int(idx)].embeddings["NV"], dtype=np.float32) for idx in node_indices]
    )
    if bool(args.normalize):
        node_embeddings = normalize_rows(node_embeddings)

    embedder = EmbeddingClient(
        model_name=str(args.embedding_name),
        base_url=str(args.embedding_base_url),
        batch_size=int(args.embedding_batch_size),
        timeout=float(args.embedding_timeout),
    )
    query_embeddings = embedder.embed(queries, desc="Embedding RAPTOR queries")
    if bool(args.normalize):
        query_embeddings = normalize_rows(query_embeddings)

    records: List[Dict[str, Any]] = []
    retrieved_doc_lists: List[List[str]] = []
    for query_idx, query_vec in enumerate(tqdm(query_embeddings, desc="Ranking RAPTOR pools")):
        pool_doc_ids, pool_scores, trace = rank_leaf_pool(
            tree=tree,
            docs=docs,
            query_embedding=query_vec,
            node_embeddings=node_embeddings,
            node_indices=node_indices,
            layer_by_node=layer_by_node,
            pool_k=int(args.pool_k),
            top_nodes=int(args.top_nodes),
            leaves_per_summary=int(args.leaves_per_summary),
        )
        pool_docs = [docs[int(idx)] for idx in pool_doc_ids]
        retrieved_doc_lists.append(pool_docs)
        records.append(
            {
                "query_idx": int(query_idx),
                "question": queries[query_idx],
                "gold_answers": list(gold_answers[query_idx]),
                "gold_docs": list(gold_docs[query_idx]),
                "gold_titles": [extract_title(doc) for doc in gold_docs[query_idx]],
                "pool_k": len(pool_docs),
                "pool_docs": pool_docs,
                "pool_titles": [extract_title(doc) for doc in pool_docs],
                "pool_doc_scores": [float(score) for score in pool_scores],
                "pool_doc_ids": [int(idx) for idx in pool_doc_ids],
                "retrieval_trace": trace,
            }
        )

    recall = compute_title_recall(gold_docs, retrieved_doc_lists, [5, 20, 100, 200])
    output = {
        "dataset": str(args.dataset),
        "limit": int(len(samples)),
        "pool_k": int(args.pool_k),
        "source": "raptor_aligned_qwen32b_no_think_nvembed_leaf_projection",
        "leaf_source": leaf_source,
        "config": {
            "raptor_root": str(Path(args.raptor_root).resolve()),
            "llm_name": str(args.llm_name),
            "llm_base_url": str(args.llm_base_url),
            "llm_no_think": True,
            "embedding_name": str(args.embedding_name),
            "embedding_base_url": str(args.embedding_base_url),
            "num_layers": int(args.num_layers),
            "actual_tree_num_layers": int(tree.num_layers),
            "layer_sizes": {str(layer): len(nodes) for layer, nodes in tree.layer_to_nodes.items()},
            "reduction_dimension": int(args.reduction_dimension),
            "cluster_threshold": float(args.cluster_threshold),
            "max_length_in_cluster": int(args.max_length_in_cluster),
            "summarization_length": int(args.summarization_length),
            "top_nodes": int(args.top_nodes),
            "leaves_per_summary": int(args.leaves_per_summary),
        },
        "retrieval": {
            "recomputed_title_recall": recall,
            "raptor_metrics": recall,
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "dataset": str(args.dataset),
                "limit": len(samples),
                "pool_k": int(args.pool_k),
                "retrieval": recall,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
