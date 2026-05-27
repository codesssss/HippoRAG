#!/usr/bin/env python3
"""Small compatibility shim for the v13b reader-only QA runner.

The source v13b tree imported three helpers from a large graph-retriever
comparison script.  Pulling that whole historical script into this repository
would also pull in unrelated experiments.  This module preserves the exact
helper surface used by ``run_transition_top5_qa.py``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hipporag.HippoRAG import HippoRAG
from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from hipporag.evaluation.retrieval_eval import RetrievalRecall
from hipporag.llm import _get_llm_class
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import QuerySolution, string_to_bool


@dataclass
class BaselineRuntime:
    HippoRAGCls: object
    BaseConfigCls: object
    RetrievalRecallCls: object
    QAExactMatchCls: object
    QAF1ScoreCls: object
    get_llm_class_fn: object
    QuerySolutionCls: object
    source_mode: str
    source_root: str | None = None
    compat_manager_fallback: bool = False
    compat_vllm_base_url_alias: bool = False
    source_fingerprint: dict[str, object] | None = None
    instruction_enhanced_query_text: bool = False
    instruction_enhanced_locations: tuple[str, ...] = ()


def normalize_embedding_name(embedding_name: str, embedding_base_url: str | None) -> str:
    if embedding_base_url is None:
        return embedding_name
    known_prefixes = ("VLLM/", "Transformers/")
    endpoint_native_substrings = ("text-embedding",)
    if str(embedding_name).startswith(known_prefixes) or any(
        token in str(embedding_name) for token in endpoint_native_substrings
    ):
        return embedding_name
    return f"VLLM/{embedding_name}"


def get_worktree_baseline_runtime() -> BaselineRuntime:
    return BaselineRuntime(
        HippoRAGCls=HippoRAG,
        BaseConfigCls=BaseConfig,
        RetrievalRecallCls=RetrievalRecall,
        QAExactMatchCls=QAExactMatch,
        QAF1ScoreCls=QAF1Score,
        get_llm_class_fn=_get_llm_class,
        QuerySolutionCls=QuerySolution,
        source_mode="worktree",
        source_root=str(PROJECT_ROOT),
        source_fingerprint={
            "source_mode": "worktree",
            "source_root": str(PROJECT_ROOT),
        },
    )


def make_config(args: Any, dataset_name: str, save_dir: str, corpus_len: int, config_cls=BaseConfig):
    config_field_names = {field.name for field in fields(config_cls)}
    config_kwargs = {
        "save_dir": save_dir,
        "llm_base_url": args.llm_base_url,
        "llm_name": args.llm_name,
        "embedding_model_name": normalize_embedding_name(args.embedding_name, args.embedding_base_url),
        "embedding_base_url": args.embedding_base_url,
        "max_new_tokens": args.max_new_tokens,
        "force_index_from_scratch": string_to_bool(args.force_index_from_scratch),
        "force_openie_from_scratch": string_to_bool(args.force_openie_from_scratch),
        "dataset": dataset_name,
        "openie_mode": args.openie_mode,
        "retrieval_mode": "hipporag_v2",
        "retrieval_top_k": args.retrieval_top_k,
        "qa_top_k": args.qa_top_k,
        "max_qa_steps": 3,
        "corpus_len": corpus_len,
        "embedding_batch_size": args.embedding_batch_size,
    }
    if hasattr(args, "qwen_disable_thinking"):
        config_kwargs["qwen_disable_thinking"] = bool(args.qwen_disable_thinking)
    for field_name in (
        "graph_use_canonical_triples",
        "graph_drop_self_loop_triples",
    ):
        if hasattr(args, field_name):
            config_kwargs[field_name] = string_to_bool(getattr(args, field_name))
    filtered_kwargs = {key: value for key, value in config_kwargs.items() if key in config_field_names}
    config = config_cls(**filtered_kwargs)
    if "qwen_disable_thinking" not in config_field_names and hasattr(args, "qwen_disable_thinking"):
        setattr(config, "qwen_disable_thinking", bool(args.qwen_disable_thinking))
    return config


def install_qwen_disable_thinking(llm_model: Any, args: Any) -> None:
    if not bool(getattr(args, "qwen_disable_thinking", False)):
        return
    llm_config = getattr(llm_model, "global_config", None)
    model_name = str(
        getattr(llm_config, "llm_name", None)
        or getattr(llm_model, "llm_name", "")
        or ""
    ).lower()
    if "qwen" not in model_name:
        return
    if getattr(llm_model, "_codex_qwen_disable_thinking_installed", False):
        return

    cache_file_name = getattr(llm_model, "cache_file_name", None)
    if isinstance(cache_file_name, str) and cache_file_name:
        if "_no_think_cache" not in os.path.basename(cache_file_name):
            if cache_file_name.endswith("_cache.sqlite"):
                llm_model.cache_file_name = cache_file_name[: -len("_cache.sqlite")] + "_no_think_cache.sqlite"
            else:
                root, ext = os.path.splitext(cache_file_name)
                llm_model.cache_file_name = f"{root}_no_think{ext}"

    original_infer = llm_model.infer

    def no_think_infer(messages, *infer_args, **infer_kwargs):
        extra_body = dict(infer_kwargs.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        infer_kwargs["extra_body"] = extra_body
        return original_infer(messages, *infer_args, **infer_kwargs)

    llm_model.infer = no_think_infer
    llm_model._codex_qwen_disable_thinking_installed = True
