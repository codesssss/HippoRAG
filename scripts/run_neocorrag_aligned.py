#!/usr/bin/env python3
"""Run NeocorRAG under the local limit-N comparison protocol.

This wrapper keeps NeocorRAG native retrieval/reader behavior, but saves a
single compact JSON with retrieval Recall@k and QA EM/F1 so it can be compared
against HippoRAG external-pool reports.
"""

from __future__ import annotations

import argparse
import ast
import atexit
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np


ENTITY_TEXT_KEYS = ("name", "entity", "text", "title", "mention", "value", "label")
NO_THINK = "/no_think"
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _add_neocorrag_to_path(root: Path) -> None:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exists: " + ", ".join(str(p) for p in paths))


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _is_complete_neocorrag_json(path: Path, dataset: str, limit: int) -> bool:
    if not path.exists():
        return False
    try:
        data = _load_json(path)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if data.get("method") != "neocorrag":
        return False
    if data.get("dataset") != dataset:
        return False
    if int(data.get("limit", -1)) != int(limit):
        return False
    return isinstance(data.get("overall_qa_results"), dict)


def _acquire_output_lock(output_json: Path, dataset: str, limit: int) -> Path | None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_json.with_suffix(output_json.suffix + ".lock")

    while True:
        if _is_complete_neocorrag_json(output_json, dataset, limit):
            print(f"[SKIP] complete output already exists: {output_json}", flush=True)
            return None
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner_pid = None
            try:
                lock_info = json.loads(lock_path.read_text(encoding="utf-8"))
                owner_pid = int(lock_info.get("pid", -1))
            except Exception:
                owner_pid = None
            if owner_pid is not None and not _is_process_alive(owner_pid):
                try:
                    lock_path.unlink()
                    print(f"[LOCK] removed stale lock: {lock_path}", flush=True)
                except FileNotFoundError:
                    pass
                continue
            print(f"[LOCK] waiting for active writer: {lock_path}", flush=True)
            time.sleep(60)
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "dataset": dataset,
                    "limit": int(limit),
                    "output_json": str(output_json),
                    "created_at": time.time(),
                },
                handle,
            )

        def cleanup_lock() -> None:
            try:
                info = json.loads(lock_path.read_text(encoding="utf-8"))
                if int(info.get("pid", -1)) == os.getpid():
                    lock_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

        atexit.register(cleanup_lock)
        return lock_path


def _coerce_entity_text(entity: Any) -> List[str]:
    if isinstance(entity, str):
        text = entity.strip()
        return [text] if text else []
    if isinstance(entity, dict):
        for key in ENTITY_TEXT_KEYS:
            value = entity.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        texts: List[str] = []
        for value in entity.values():
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
            elif isinstance(value, (list, tuple)):
                texts.extend(_normalize_entities(value))
        return texts[:1]
    if isinstance(entity, (list, tuple)):
        return _normalize_entities(entity)
    return []


def _normalize_entities(entities: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for entity in entities:
        for text in _coerce_entity_text(entity):
            if text not in seen:
                normalized.append(text)
                seen.add(text)
    return normalized


def _patch_neocorrag_ner_parser() -> None:
    from neocorrag.information_extraction import openie_openai  # type: ignore

    original_extract_ner = openie_openai._extract_ner_from_response

    def extract_ner_as_strings(real_response: Any) -> List[str]:
        extracted = original_extract_ner(real_response)
        if not isinstance(extracted, list):
            extracted = [extracted]
        return _normalize_entities(extracted)

    openie_openai._extract_ner_from_response = extract_ner_as_strings


def _patch_neocorrag_reretrieval_fallback() -> None:
    from neocorrag.constrained_decoding.qa_prompt_builder import (  # type: ignore
        PathGenerationWithAnswerPromptBuilder,
    )

    original_dynamic_search = (
        PathGenerationWithAnswerPromptBuilder.get_graph_index_with_dynamic_search
    )

    def dynamic_search_with_empty_fallback(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_dynamic_search(self, *args, **kwargs)
        if result is None:
            return [], []
        return result

    PathGenerationWithAnswerPromptBuilder.get_graph_index_with_dynamic_search = (
        dynamic_search_with_empty_fallback
    )


def _patch_neocorrag_top_k_weight_filter() -> None:
    from neocorrag.NeocorRAG import NeocorRAG  # type: ignore
    from neocorrag.utils.misc_utils import compute_mdhash_id  # type: ignore

    def get_top_k_weights_without_invalid_phrases(
        self: Any,
        link_top_k: int,
        all_phrase_weights: np.ndarray,
        linking_score_map: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, float]]:
        selected_scores: dict[str, float] = {}
        selected_keys = set()
        sorted_items = sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)

        for phrase, score in sorted_items:
            phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
            phrase_id = self.node_name_to_vertex_idx.get(phrase_key)
            if phrase_id is None:
                continue
            if phrase_id >= len(all_phrase_weights):
                continue
            weight = float(all_phrase_weights[phrase_id])
            if not np.isfinite(weight) or weight == 0.0:
                continue
            selected_scores[phrase] = float(score)
            selected_keys.add(phrase_key)
            if len(selected_scores) >= int(link_top_k):
                break

        for phrase_key, phrase_id in self.node_name_to_vertex_idx.items():
            if phrase_key not in selected_keys and phrase_id is not None:
                all_phrase_weights[phrase_id] = 0.0

        return all_phrase_weights, selected_scores

    NeocorRAG.get_top_k_weights = get_top_k_weights_without_invalid_phrases


def _patch_neocorrag_dspy_filter_template() -> None:
    from neocorrag.utils.rerank_filter import DSPyFilter  # type: ignore

    original_init = DSPyFilter.__init__

    def init_with_template(self: Any, neocorrag: Any) -> None:
        original_init(self, neocorrag)
        if hasattr(self, "message_template"):
            return
        self.one_input_template = (
            "[[ ## question ## ]]\n{question}\n\n"
            "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
            "Respond with the corresponding output fields, starting with the field "
            "`[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python "
            "Fact), and then ending with the marker for `[[ ## completed ## ]]`."
        )
        self.one_output_template = (
            "[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n"
            "[[ ## completed ## ]]"
        )
        self.message_template = self.make_template(
            neocorrag.global_config.rerank_dspy_file_path
        )

    DSPyFilter.__init__ = init_with_template


def _strip_think(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return THINK_RE.sub("", text).strip()


def _with_no_think(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    patched = []
    injected = False
    for message in messages:
        if isinstance(message, dict):
            item = dict(message)
            role = item.get("role")
            content = item.get("content")
            if role == "user" and isinstance(content, str) and not injected:
                if NO_THINK not in content:
                    item["content"] = f"{NO_THINK}\n{content}"
                injected = True
            patched.append(item)
        else:
            patched.append(message)
    if not injected:
        patched.append({"role": "user", "content": NO_THINK})
    return patched


def _patch_neocorrag_qwen_no_think() -> None:
    from neocorrag.llm.openai_gpt import CacheOpenAI  # type: ignore

    original_infer = CacheOpenAI.infer
    original_batch_infer = CacheOpenAI.batch_infer

    def infer_with_no_think(self: Any, messages: Any, *args: Any, **kwargs: Any) -> Any:
        response, metadata, *rest = original_infer(self, _with_no_think(messages), *args, **kwargs)
        response = _strip_think(response)
        if rest:
            return response, metadata, *rest
        return response, metadata

    def batch_infer_with_no_think(self: Any, messages_list: Any, *args: Any, **kwargs: Any) -> Any:
        patched_messages = [_with_no_think(messages) for messages in messages_list]
        responses, metadata = original_batch_infer(self, patched_messages, *args, **kwargs)
        responses = [_strip_think(response) for response in responses]
        return responses, metadata

    CacheOpenAI.infer = infer_with_no_think
    CacheOpenAI.batch_infer = batch_infer_with_no_think


def _with_no_think_text(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    if NO_THINK in text[:128]:
        return text
    return f"{NO_THINK}\n{text}"


def _is_qwen3_hf_model(model: Any) -> bool:
    model_path = str(getattr(getattr(model, "args", None), "model_path", "") or "")
    return "qwen3" in model_path.lower()


def _strip_think_from_generation(output: Any) -> Any:
    if isinstance(output, list):
        return [_strip_think_from_generation(item) for item in output]
    return _strip_think(output)


def _patch_neocorrag_local_reretrieval_no_think() -> None:
    from neocorrag.constrained_decoding.llms.base_hf_causal_model import (  # type: ignore
        HfCausalModel,
    )
    from neocorrag.constrained_decoding.llms.graph_constrained_decoding_model import (  # type: ignore
        GraphConstrainedDecodingModel,
    )

    original_prepare_model_prompt = HfCausalModel.prepare_model_prompt
    original_hf_generate_sentence = HfCausalModel.generate_sentence
    original_gcd_generate_sentence = GraphConstrainedDecodingModel.generate_sentence

    def prepare_model_prompt_with_no_think(self: Any, query: Any) -> Any:
        if _is_qwen3_hf_model(self) and getattr(getattr(self, "args", None), "chat_model", False):
            format_system = """
        Your task is to generate inference paths from candidate documents based on the problem and topic entities.
        Respond is returned in list format!
        An example of the respond format:["The Newcomers - stars -> Chris Evans - known for role -> Steve Rogers Captain America - part of -> Marvel Cinematic Universe"]
        """
            chat_query = [
                {"role": "system", "content": format_system},
                {"role": "user", "content": _with_no_think_text(query)},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    chat_query,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    chat_query,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        if _is_qwen3_hf_model(self):
            query = _with_no_think_text(query)
        return original_prepare_model_prompt(self, query)

    def hf_generate_sentence_strip_think(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _strip_think_from_generation(
            original_hf_generate_sentence(self, *args, **kwargs)
        )

    def gcd_generate_sentence_strip_think(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _strip_think_from_generation(
            original_gcd_generate_sentence(self, *args, **kwargs)
        )

    HfCausalModel.prepare_model_prompt = prepare_model_prompt_with_no_think
    HfCausalModel.generate_sentence = hf_generate_sentence_strip_think
    GraphConstrainedDecodingModel.generate_sentence = gcd_generate_sentence_strip_think


def _as_text_list(items: Any) -> List[str]:
    if items is None:
        return []
    if isinstance(items, str):
        return [items] if items else []
    if isinstance(items, (list, tuple)):
        return [str(item) for item in items if str(item).strip()]
    return [str(items)]


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


def _cap_text_items(items: Any, *, total_chars: int, item_chars: int) -> List[str]:
    capped: List[str] = []
    used = 0
    for text in _as_text_list(items):
        text = text.strip()
        if not text:
            continue
        if item_chars > 0 and len(text) > item_chars:
            text = text[:item_chars].rstrip() + " ..."
        remaining = total_chars - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[: max(0, remaining - 4)].rstrip() + " ..."
        capped.append(text)
        used += len(text) + 1
    return capped


def _patch_neocorrag_qa_prompt_budget() -> None:
    from neocorrag.NeocorRAG import NeocorRAG  # type: ignore

    original_build_qa_prompt = NeocorRAG.build_qa_prompt

    def build_qa_prompt_with_budget(
        self: Any,
        question: str,
        prediction_paths: Any,
        docs: Any,
        prompt_template_type: str = "gcd",
    ) -> str:
        # Qwen3 endpoint here has an 8192-token context window.  Native
        # NeocorRAG can inject tens of thousands of chars of paths into the QA
        # prompt, which turns the run into all API 400 errors.  Keep the native
        # Reretrieval signal but bound it to a reader-feasible budget.
        qa_top_k = int(getattr(self.global_config, "qa_top_k", 5) or 5)
        doc_budget = int(getattr(self.global_config, "aligned_qa_doc_char_budget", 9000))
        path_budget = int(getattr(self.global_config, "aligned_qa_path_char_budget", 4500))
        doc_item_budget = int(getattr(self.global_config, "aligned_qa_doc_item_char_budget", 2200))
        path_item_budget = int(getattr(self.global_config, "aligned_qa_path_item_char_budget", 900))

        capped_docs = _cap_text_items(
            _as_text_list(docs)[:qa_top_k],
            total_chars=doc_budget,
            item_chars=doc_item_budget,
        )
        capped_paths = _cap_text_items(
            prediction_paths,
            total_chars=path_budget,
            item_chars=path_item_budget,
        )
        return original_build_qa_prompt(
            self,
            question,
            capped_paths,
            capped_docs,
            prompt_template_type,
        )

    NeocorRAG.build_qa_prompt = build_qa_prompt_with_budget


def repair_graph_node_coverage(neocorrag: Any) -> dict[str, int]:
    """Add embedding-store nodes missing from the loaded graph.

    NeocorRAG reuses graph.graphml and embedding parquet stores independently.
    Partial or prior failed runs can leave the graph missing entity vertices even
    when their embeddings exist; native retrieval asserts exact coverage before
    it can run.  This repair preserves existing edges and only appends missing
    entity/passage vertices from the stores.
    """

    if "name" in neocorrag.graph.vs.attribute_names():
        graph_nodes = set(neocorrag.graph.vs["name"])
    else:
        graph_nodes = set()

    node_rows = {}
    node_rows.update(neocorrag.entity_embedding_store.get_text_for_all_rows())
    node_rows.update(neocorrag.chunk_embedding_store.get_text_for_all_rows())
    missing = {node_id: row for node_id, row in node_rows.items() if node_id not in graph_nodes}

    if missing:
        new_nodes: dict[str, list[Any]] = {}
        for node_id, row in missing.items():
            node = dict(row)
            node["name"] = node_id
            for key, value in node.items():
                new_nodes.setdefault(key, []).append(value)
        neocorrag.graph.add_vertices(n=len(missing), attributes=new_nodes)
        neocorrag.save_igraph()

    final_graph_nodes = set(neocorrag.graph.vs["name"]) if "name" in neocorrag.graph.vs.attribute_names() else set()
    expected = set(node_rows)
    return {
        "expected_entity_plus_passage_nodes": len(expected),
        "graph_nodes_before": len(graph_nodes),
        "missing_nodes_added": len(missing),
        "graph_nodes_after": neocorrag.graph.vcount(),
        "still_missing_nodes": len(expected - final_graph_nodes),
    }


def get_gold_docs(samples: List[dict], dataset_name: str) -> List[List[str]]:
    gold_docs: List[List[str]] = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_title = {item[0] for item in sample["supporting_facts"]}
            gold_title_and_content = [item for item in sample["context"] if item[0] in gold_title]
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
            paragraphs = []
            for item in sample["paragraphs"]:
                if item.get("is_supporting") is False:
                    continue
                paragraphs.append(item)
            docs = [
                item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"])
                for item in paragraphs
            ]
        gold_docs.append(list(set(docs)))
    return gold_docs


def get_gold_answers(samples: List[dict]) -> List[List[str]]:
    gold_answers: List[List[str]] = []
    for sample in samples:
        answers: List[str] = []
        if "answer" in sample or "gold_ans" in sample:
            answers.extend(parse_answer_alias_values(sample.get("answer", sample.get("gold_ans"))))
        elif "reference" in sample:
            answers.extend(parse_answer_alias_values(sample["reference"]))
        elif "obj" in sample:
            answers.extend(parse_answer_alias_values(sample.get("obj")))
            answers.extend(parse_answer_alias_values(sample.get("possible_answers")))
            answers.extend(parse_answer_alias_values(sample.get("o_wiki_title")))
            answers.extend(parse_answer_alias_values(sample.get("o_aliases")))
        else:
            raise KeyError(f"Cannot find gold answer fields in sample keys={sorted(sample.keys())}")
        if "answer_aliases" in sample:
            answers.extend(parse_answer_alias_values(sample["answer_aliases"]))
        gold_answers.append(sorted({str(item).strip() for item in answers if str(item).strip()}))
    return gold_answers


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, set):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--neocorrag_root", type=Path, default=Path("/mnt/nvme/code/NeocorRAG"))
    parser.add_argument("--hipporag_root", type=Path, default=Path("/mnt/nvme/code/HippoRAG"))
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_method_name", default=None)
    parser.add_argument("--llm_name", default="qwen3-8b-train")
    parser.add_argument("--llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--graph_llm_name", default="qwen3-8b-train")
    parser.add_argument("--graph_llm_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--embedding_name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--embedding_batch_size", type=int, default=4)
    parser.add_argument("--reretrieval_llm_name", default="/mnt/nvme/Qwen3-8B")
    parser.add_argument("--reretrieval_embedding_name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--generation_mode", default="greedy")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_search_paths", type=int, default=50)
    args = parser.parse_args()

    output_lock = _acquire_output_lock(args.output_json, args.dataset, args.limit)
    if output_lock is None:
        return

    _add_neocorrag_to_path(args.neocorrag_root)

    import importlib

    from neocorrag import NeocorRAG  # type: ignore
    from neocorrag.embedding_model.OpenAIEmbedding import OpenAI_Compatible_EmbeddingModel  # type: ignore
    from neocorrag.utils.config_utils import BaseConfig  # type: ignore

    # Qwen-style OpenIE responses sometimes emit named_entities as objects such
    # as {"name": "...", "type": "..."} instead of plain strings.  Native
    # NeocorRAG deduplicates with dict.fromkeys(), which drops those chunks after
    # retrying with "unhashable type: 'dict'".  Normalize at the wrapper boundary
    # so the comparison keeps native behavior without losing valid entities.
    _patch_neocorrag_ner_parser()
    # Native dynamic Reretrieval can return None when a retrieved subgraph has no
    # valid paths, while its caller expects a pair of lists.  Treat no-path cases
    # as empty predictions so NeocorRAG's existing default-QA fallback can handle
    # them instead of crashing after retrieval has completed.
    _patch_neocorrag_reretrieval_fallback()
    # Native retrieval can include phrase strings in linking_score_map whose
    # hashed entity node is absent from the graph, then assert that every kept
    # phrase has a nonzero node weight.  Filter invalid/zero-weight phrases and
    # keep the remaining valid top-k entity weights; passage weights are still
    # added by the original graph_search_with_fact_entities path.
    _patch_neocorrag_top_k_weight_filter()
    # The filter class in this checkout builds prompts from self.message_template
    # but its initializer never creates that attribute.  Add the intended default
    # prompt state at runtime; otherwise every filter call silently falls back.
    _patch_neocorrag_dspy_filter_template()
    # Keep Qwen3 aligned with the rest of the protocol: disable thinking traces
    # for OpenIE/query NER and QA calls, and strip any already-emitted traces.
    _patch_neocorrag_qwen_no_think()
    # Reretrieval is a local HF constrained-decoding model, not an API call.
    # If it is Qwen3, inject /no_think into its chat-template user message too.
    _patch_neocorrag_local_reretrieval_no_think()
    # Native Reretrieval QA prompts can exceed Qwen3's 8192-token window by a
    # large margin.  Bound docs and paths before final reader calls.
    _patch_neocorrag_qa_prompt_budget()

    # NeocorRAG's NVEmbedV2 class hard-codes a lab-local model path in this
    # checkout.  For the aligned protocol we use the same local OpenAI-compatible
    # NV-Embed endpoint as HippoRAG instead of loading a separate embedding model.
    neocorrag_module = importlib.import_module("neocorrag.NeocorRAG")
    neocorrag_module._get_embedding_model_class = (
        lambda *args, **kwargs: OpenAI_Compatible_EmbeddingModel
    )

    neocor_dataset_dir = args.neocorrag_root / "reproduce" / "dataset"
    hippo_dataset_dir = args.hipporag_root / "reproduce" / "dataset"
    sample_path = _first_existing(
        [
            neocor_dataset_dir / f"{args.dataset}.json",
            hippo_dataset_dir / f"{args.dataset}.json",
        ]
    )
    corpus_path = _first_existing(
        [
            neocor_dataset_dir / f"{args.dataset}_corpus.json",
            hippo_dataset_dir / f"{args.dataset}_corpus.json",
        ]
    )

    samples = _load_json(sample_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    corpus = _load_json(corpus_path)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_docs = get_gold_docs(samples, args.dataset)
    gold_answers = get_gold_answers(samples)

    base_save_dir = args.neocorrag_root / "outputs" / args.dataset
    method_name = args.output_method_name or (
        f"aligned_limit{args.limit}_{args.llm_name}_{args.generation_mode}_k{args.k}"
        f"_ret{args.retrieval_top_k}_reremb{args.reretrieval_embedding_name.replace('/', '_')}"
    )
    save_dir = base_save_dir / method_name
    save_dir.mkdir(parents=True, exist_ok=True)

    embedding_base_url = str(args.embedding_base_url)
    if embedding_base_url.rstrip("/").endswith("/embeddings"):
        embedding_base_url = embedding_base_url.rstrip("/")[: -len("/embeddings")]

    config = BaseConfig(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        graph_llm_base_url=args.graph_llm_base_url,
        graph_llm_name=args.graph_llm_name,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        embedding_base_url=embedding_base_url,
        save_dir=str(save_dir),
        base_save_dir=str(base_save_dir),
        corpus_len=len(corpus),
        openie_mode="online",
        generate_mode="online",
        retrieval_top_k=int(args.retrieval_top_k),
        qa_top_k=int(args.qa_top_k),
        embedding_batch_size=int(args.embedding_batch_size),
        max_new_tokens=int(args.max_new_tokens),
        max_search_paths=int(args.max_search_paths),
        reflection_top_n=2,
        reflection_qa_strategy="default_qa",
    )

    neocorrag = NeocorRAG(global_config=config)
    neocorrag.index(docs)
    graph_repair = repair_graph_node_coverage(neocorrag)

    retrieval_solutions, overall_retrieval_result, rerank_log = neocorrag.retrieve(
        queries=queries,
        num_to_retrieve=int(args.retrieval_top_k),
        gold_docs=gold_docs,
    )
    for idx, solution in enumerate(retrieval_solutions):
        solution.gold_docs = gold_docs[idx]
        solution.gold_answers = gold_answers[idx]

    _, _, _, overall_qa_results, qa_save_path = neocorrag.rag_qa_Reretrieval(
        queries=retrieval_solutions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
        Reretrieval_embedding=args.reretrieval_embedding_name,
        Reretrieval_LLM_name=args.reretrieval_llm_name,
        generation_mode=args.generation_mode,
        k=int(args.k),
    )

    records = []
    for idx, solution in enumerate(retrieval_solutions):
        records.append(
            {
                "query_idx": idx,
                "question": solution.question,
                "gold_answers": gold_answers[idx],
                "gold_docs": gold_docs[idx],
                "docs": solution.docs,
                "doc_scores": to_jsonable(solution.doc_scores),
                "answer": solution.answer,
                "reasoning_paths": solution.reasoning_paths,
            }
        )

    output = {
        "method": "neocorrag",
        "dataset": args.dataset,
        "limit": int(args.limit),
        "num_queries": len(queries),
        "sample_path": str(sample_path),
        "corpus_path": str(corpus_path),
        "neocorrag_root": str(args.neocorrag_root),
        "config": {
            "llm_name": args.llm_name,
            "llm_base_url": args.llm_base_url,
            "graph_llm_name": args.graph_llm_name,
            "graph_llm_base_url": args.graph_llm_base_url,
            "embedding_name": args.embedding_name,
            "embedding_base_url": args.embedding_base_url,
            "embedding_base_url_openai_client": embedding_base_url,
            "reretrieval_llm_name": args.reretrieval_llm_name,
            "reretrieval_embedding_name": args.reretrieval_embedding_name,
            "reretrieval_local_no_think": True,
            "generation_mode": args.generation_mode,
            "k": int(args.k),
            "qa_top_k": int(args.qa_top_k),
            "retrieval_top_k": int(args.retrieval_top_k),
            "no_think": True,
            "qa_prompt_budget": {
                "doc_total_chars": 9000,
                "path_total_chars": 4500,
                "doc_item_chars": 2200,
                "path_item_chars": 900,
            },
        },
        "diagnostics": {
            "graph_repair": graph_repair,
        },
        "overall_retrieval_result": to_jsonable(overall_retrieval_result),
        "overall_qa_results": to_jsonable(overall_qa_results),
        "qa_save_path": str(qa_save_path),
        "records": to_jsonable(records),
        "rerank_log_preview": to_jsonable(rerank_log[:5] if isinstance(rerank_log, list) else rerank_log),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset": args.dataset,
        "limit": int(args.limit),
        "overall_retrieval_result": output["overall_retrieval_result"],
        "overall_qa_results": output["overall_qa_results"],
        "output_json": str(args.output_json),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
