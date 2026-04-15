import argparse
import ast
import copy
from dataclasses import dataclass
import json
import logging
import os
import re
import sys
import types
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import joblib
import numpy as np
import pydantic
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LOCAL_SRC_DIR = ROOT_DIR / "src"
existing_src_module = sys.modules.get("src")
existing_src_paths = list(getattr(existing_src_module, "__path__", [])) if existing_src_module is not None else []
if LOCAL_SRC_DIR.exists() and str(LOCAL_SRC_DIR) not in existing_src_paths:
    local_src_module = types.ModuleType("src")
    local_src_module.__path__ = [str(LOCAL_SRC_DIR)]
    sys.modules["src"] = local_src_module

from requirement_beam_utils import (
    align_requirement_cache_entry_to_pool,
    canonicalize_requirement_positions,
    DEFAULT_REQUIREMENT_ANNOTATION_POOL_K,
    DEFAULT_REQUIREMENT_CF_TAU,
    DEFAULT_REQUIREMENT_SMOOTH_TAU,
    compute_requirement_candidate_feature_rows,
    compute_requirement_state_metrics,
    get_counterfactual_set_score_map,
    get_positive_score_map_with_overrides,
    get_positive_score_map,
    get_positive_units,
    is_pareto_dominated,
    is_need_unit_cache_version,
    load_requirement_cache,
    load_need_unit_atomic_model_bundle,
    load_requirement_model_bundle,
    merge_positive_score_overrides_by_position,
    normalize_requirement_bridge_bonus_mode,
    requirement_feature_rows_to_matrix,
    resolve_requirement_cache_entry,
    validate_requirement_cache_entry,
)
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.utils.causal_utils import (
    expand_directed_entities,
    normalize_structure_text,
    normalize_structure_seed_target_bridge_mode,
    route_query_type,
    score_candidate_docs_by_structure,
)
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.dataset_utils import resolve_dataset_paths
from src.hipporag.utils.misc_utils import QuerySolution, compute_mdhash_id, string_to_bool

LEARNED_SETWISE_FEATURE_NAMES = [
    "base_score",
    "rank_fraction",
    "reciprocal_rank",
    "selected_count_fraction",
    "seed_entity_overlap_ratio",
    "covered_entity_overlap_ratio",
    "selected_entity_overlap_ratio",
    "query_token_overlap_ratio",
    "novelty_score",
    "structure_score_seed",
    "structure_score_covered",
    "reachable_from_seed",
    "reachable_from_covered",
    "entity_count_norm",
    "edge_count_norm",
]

DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG = {
    "path_connectivity": 0.30,
    "reachable_doc_ratio": 0.20,
    "query_reachability": 0.15,
    "support_mean": 0.05,
    "support_min": 0.05,
    "closure_mean": 0.10,
    "suffix_base_mean": 0.05,
    "query_coverage": 0.10,
    "frontier_ratio": 0.05,
    "redundancy_penalty": 0.10,
}
DEFAULT_SET_CLOSURE_EXACT_PATH_MAX_DOCS = 4
DEFAULT_SET_CLOSURE_PROJECTED_SHORTLIST_FACTOR = 1
DEFAULT_REQUIREMENT_BEAM_PROJECTED_SHORTLIST_FACTOR = 3
DEFAULT_REQUIREMENT_BEAM_RESERVE_POLICY = "fixed"
DEFAULT_REQUIREMENT_EXPOSURE_WATCH_TITLES = (
    "Riverside Plaza",
    "Minneapolis",
    "Mississippi River",
    "The Right Stuff Records",
    "Sony Music",
)
DEFAULT_REQUIREMENT_SOURCE_SORT_MODE = "combined"
DEFAULT_REQUIREMENT_SHORTLIST_SORT_MODE = "margin_first"
REPORT_RECALL_CUTOFFS = (1, 2, 5, 10, 20, 100)

SETWISE_LLM_JSON_START_TAG = "<JSON>"
SETWISE_LLM_JSON_END_TAG = "</JSON>"
SETWISE_LLM_LATE_RERANK_MAX_COMPLETION_TOKENS = 256
SETWISE_LLM_LATE_RERANK_REPAIR_MAX_COMPLETION_TOKENS = 96
SETWISE_LLM_NO_THINK_PREFIX = "/no_think"


class SetwiseLateRerankResponseModel(pydantic.BaseModel):
    best_id: int = pydantic.Field(..., description="0-based id of the best candidate evidence set.")
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score between 0 and 1.",
    )


@dataclass
class SetwiseLateRerankJudgeBundle:
    infer_fn: Any
    model_name: str
    backend: str
    base_url: str | None = None
    response_format: Any = None
    reasoning_effort: str | None = None
    api_key_env: str | None = None


class OpenAICompatibleLateRerankJudge:
    def __init__(self,
                 model_name: str,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 backend: str = "responses",
                 timeout_s: float = 120.0,
                 reasoning_effort: str | None = None,
                 client: OpenAI | None = None) -> None:
        normalized_backend = str(backend or "responses").strip().lower()
        if normalized_backend not in {"responses", "chat_completions"}:
            raise ValueError(f"Unsupported late rerank judge backend: {backend}")

        self.model_name = str(model_name).strip()
        if not self.model_name:
            raise ValueError("Late rerank judge model name must be non-empty.")
        self.base_url = str(base_url).strip() if base_url else None
        self.backend = normalized_backend
        self.reasoning_effort = str(reasoning_effort).strip() if reasoning_effort else None
        self.client = client or OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=float(timeout_s),
            max_retries=2,
        )

    def infer(self,
              messages: List[Dict[str, str]],
              **kwargs) -> Tuple[str, Dict[str, Any]]:
        response_format = kwargs.get("response_format")
        model_name = str(kwargs.get("model") or self.model_name)
        temperature = float(kwargs.get("temperature", 0))
        top_p = float(kwargs.get("top_p", 1))
        max_completion_tokens = int(kwargs.get("max_completion_tokens", SETWISE_LLM_LATE_RERANK_MAX_COMPLETION_TOKENS))
        if self.backend == "responses":
            return self._infer_with_responses(
                messages=messages,
                model_name=model_name,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        return self._infer_with_chat_completions(
            messages=messages,
            model_name=model_name,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _infer_with_responses(self,
                              messages: List[Dict[str, str]],
                              model_name: str,
                              response_format: Any,
                              max_completion_tokens: int,
                              temperature: float,
                              top_p: float) -> Tuple[str, Dict[str, Any]]:
        params: Dict[str, Any] = {
            "model": model_name,
            "input": messages,
            "max_output_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}

        response = self.client.responses.create(**params)
        response_text, response_text_source = extract_responses_api_text(response)

        usage = getattr(response, "usage", None)
        metadata = {
            "backend": self.backend,
            "model": model_name,
            "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "finish_reason": str(getattr(response, "status", "") or ""),
            "response_text_source": response_text_source,
            "structured_output_requested": bool(is_pydantic_response_format(response_format)),
        }
        return response_text, metadata

    def _infer_with_chat_completions(self,
                                     messages: List[Dict[str, str]],
                                     model_name: str,
                                     response_format: Any,
                                     max_completion_tokens: int,
                                     temperature: float,
                                     top_p: float) -> Tuple[str, Dict[str, Any]]:
        if not is_pydantic_response_format(response_format):
            return self._infer_with_chat_stream(
                messages=messages,
                model_name=model_name,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

        if is_pydantic_response_format(response_format):
            response = self.client.beta.chat.completions.parse(
                **params,
                response_format=response_format,
            )
            message = response.choices[0].message
            parsed_payload = getattr(message, "parsed", None)
            if parsed_payload is not None:
                response_text = parsed_payload.model_dump_json()
            else:
                response_text = str(message.content or "")
        else:
            response = self.client.chat.completions.create(
                **params,
                response_format=response_format,
            )
            response_text = str(response.choices[0].message.content or "")

        usage = getattr(response, "usage", None)
        finish_reason = None
        if getattr(response, "choices", None):
            finish_reason = getattr(response.choices[0], "finish_reason", None)
        metadata = {
            "backend": self.backend,
            "model": model_name,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "finish_reason": str(finish_reason or ""),
        }
        return response_text, metadata

    def _infer_with_chat_stream(self,
                                messages: List[Dict[str, str]],
                                model_name: str,
                                max_completion_tokens: int,
                                temperature: float,
                                top_p: float) -> Tuple[str, Dict[str, Any]]:
        params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
        }
        stream = self.client.chat.completions.create(**params)
        response_chunks: List[str] = []
        last_chunk = None
        for chunk in stream:
            last_chunk = chunk
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(content, str) and content:
                response_chunks.append(content)

        usage = getattr(last_chunk, "usage", None) if last_chunk is not None else None
        finish_reason = None
        if last_chunk is not None and getattr(last_chunk, "choices", None):
            finish_reason = getattr(last_chunk.choices[0], "finish_reason", None)
        metadata = {
            "backend": self.backend,
            "model": model_name,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "finish_reason": str(finish_reason or ""),
            "streamed": True,
        }
        return "".join(response_chunks), metadata


def get_gold_docs(samples: List, dataset_name: str = None, corpus: List | None = None) -> List:
    gold_docs = []
    corpus = corpus or []
    for sample in samples:
        if 'supporting_facts' in sample:
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        elif 'document' in sample and isinstance(sample['document'], dict) and corpus:
            document_id = str(sample['document'].get('id', '')).strip()
            gold_doc = [
                item['title'] + '\n' + item['text']
                for item in corpus
                if str(item.get('idx', '')).startswith(f"{document_id}_")
            ]
        else:
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample in samples:
        gold_ans = None
        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = list(set([sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']]))
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])
        gold_answers.append(list(gold_ans))
    return gold_answers


def build_doc_text_to_chunk_id(corpus: List[dict]) -> Dict[str, str]:
    doc_text_to_chunk_id: Dict[str, str] = {}
    for row in corpus:
        doc_text = f"{row['title']}\n{row['text']}"
        doc_text_to_chunk_id[doc_text] = compute_mdhash_id(doc_text, prefix="chunk-")
    return doc_text_to_chunk_id


def build_chunk_id_to_doc_text(corpus: List[dict]) -> Dict[str, str]:
    return {
        chunk_id: doc_text
        for doc_text, chunk_id in build_doc_text_to_chunk_id(corpus).items()
    }


def serialize_retrieved_doc_ids(retrieved_docs: List[str], doc_text_to_chunk_id: Dict[str, str]) -> List[str | None]:
    return [doc_text_to_chunk_id.get(doc_text) for doc_text in retrieved_docs]


def subset_by_indices(values: List, indices: List[int]) -> List:
    return [values[idx] for idx in indices]


def min_max_normalize_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=float)

    values = np.asarray(values, dtype=float)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        return np.ones_like(values, dtype=float) if max_value > 0 else np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def extract_doc_title(doc_text: str) -> str:
    return str(doc_text).split("\n", 1)[0].strip()


def unique_ordered_titles(values: Sequence[str] | None) -> List[str]:
    deduped: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        title = str(value or "").strip()
        title_key = normalize_structure_text(title)
        if not title or not title_key or title_key in seen:
            continue
        deduped.append(title)
        seen.add(title_key)
    return deduped


def normalize_title_set(values: Sequence[str] | None) -> Set[str]:
    return {
        normalize_structure_text(value)
        for value in unique_ordered_titles(values)
        if normalize_structure_text(value)
    }


def parse_title_csv(value: str | None,
                    default: Sequence[str] | None = None) -> List[str]:
    raw_value = str(value or "").strip()
    if not raw_value:
        return unique_ordered_titles(default)
    return unique_ordered_titles([
        item.strip()
        for item in raw_value.split(",")
        if str(item).strip()
    ])


def resolve_title_pool_positions(pool_titles: Sequence[str],
                                 target_titles: Sequence[str] | None,
                                 pool_limit: int | None = None) -> List[int]:
    normalized_targets = unique_ordered_titles(target_titles)
    if not normalized_targets:
        return []
    target_keys = {
        normalize_structure_text(title): title
        for title in normalized_targets
    }
    resolved_positions: List[int] = []
    used_keys: Set[str] = set()
    effective_pool_limit = min(len(pool_titles), max(int(pool_limit or len(pool_titles)), 0))
    for pool_position, raw_title in enumerate(pool_titles[:effective_pool_limit]):
        title_key = normalize_structure_text(str(raw_title or "").strip())
        if not title_key or title_key not in target_keys or title_key in used_keys:
            continue
        resolved_positions.append(int(pool_position))
        used_keys.add(title_key)
    return resolved_positions


def resolve_query_pool_gold_titles(pool_titles: Sequence[str],
                                   gold_docs: Sequence[str] | None,
                                   pool_limit: int | None = None) -> Dict[str, object]:
    requested_titles = unique_ordered_titles([
        extract_doc_title(doc_text)
        for doc_text in gold_docs or []
    ])
    resolved_positions = resolve_title_pool_positions(
        pool_titles=pool_titles,
        target_titles=requested_titles,
        pool_limit=pool_limit,
    )
    resolved_titles = unique_ordered_titles([
        str(pool_titles[pos]).strip()
        for pos in resolved_positions
        if 0 <= int(pos) < len(pool_titles)
    ])
    resolved_title_keys = normalize_title_set(resolved_titles)
    missing_titles = [
        title for title in requested_titles
        if normalize_structure_text(title) not in resolved_title_keys
    ]
    return {
        "requested_titles": requested_titles,
        "in_pool_titles": resolved_titles,
        "missing_titles": missing_titles,
        "positions": [int(pos) for pos in resolved_positions],
    }


def build_requirement_title_exposure_summary(pool_titles: Sequence[str],
                                             selector_trace: Dict[str, object] | None,
                                             target_titles: Sequence[str] | None) -> List[Dict[str, object]]:
    normalized_targets = unique_ordered_titles(target_titles)
    if not normalized_targets:
        return []

    title_key_to_label = {
        normalize_structure_text(title): title
        for title in normalized_targets
    }
    pool_positions_by_key: Dict[str, List[int]] = {}
    for pool_position, raw_title in enumerate(pool_titles):
        title = str(raw_title or "").strip()
        title_key = normalize_structure_text(title)
        if not title_key:
            continue
        pool_positions_by_key.setdefault(title_key, []).append(int(pool_position))

    trace_payload = dict(selector_trace or {})
    source_hits: Dict[str, List[Dict[str, int]]] = {}
    shortlist_hits: Dict[str, List[Dict[str, int]]] = {}
    scored_hits: Dict[str, List[Dict[str, object]]] = {}
    forced_source_hits: Dict[str, bool] = {}
    forced_shortlist_hits: Dict[str, bool] = {}
    forced_final_hits = normalize_title_set(trace_payload.get("forced_final_titles_applied", []))
    forced_final_hits.update(normalize_title_set(trace_payload.get("forced_probe_final_titles_applied", [])))
    forced_final_hits.update(normalize_title_set(trace_payload.get("forced_pool_gold_final_titles_applied", [])))
    for step in trace_payload.get("selection_steps", []) or []:
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("step", 0) or 0)
        for watch_row in step.get("watch_title_trace", []) or []:
            if not isinstance(watch_row, dict):
                continue
            title = str(watch_row.get("title", "")).strip()
            title_key = normalize_structure_text(title)
            if title_key not in title_key_to_label:
                continue
            scored_hits.setdefault(title_key, []).append({
                "step": int(step_index),
                "scored_rank": int(watch_row.get("scored_rank", 0) or 0),
                "combined_score": float(watch_row.get("combined_score", 0.0) or 0.0),
                "utility_margin_gain": float(watch_row.get("utility_margin_gain", 0.0) or 0.0),
                "support_completeness_gain": float(watch_row.get("support_completeness_gain", 0.0) or 0.0),
                "forced_into_source": bool(watch_row.get("forced_into_source", False)),
                "forced_into_shortlist": bool(watch_row.get("forced_into_shortlist", False)),
            })
            forced_source_hits[title_key] = forced_source_hits.get(title_key, False) or bool(
                watch_row.get("forced_into_source", False)
            )
            forced_shortlist_hits[title_key] = forced_shortlist_hits.get(title_key, False) or bool(
                watch_row.get("forced_into_shortlist", False)
            )
        for field_name, collector in (
            ("candidate_source_preview", source_hits),
            ("candidate_shortlist_preview", shortlist_hits),
        ):
            preview_rows = step.get(field_name) or []
            if not isinstance(preview_rows, Sequence) or isinstance(preview_rows, (str, bytes)):
                continue
            for preview_row in preview_rows:
                if not isinstance(preview_row, dict):
                    continue
                title = str(preview_row.get("title", "")).strip()
                title_key = normalize_structure_text(title)
                if title_key not in title_key_to_label:
                    continue
                collector.setdefault(title_key, []).append({
                    "step": int(step_index),
                    "preview_rank": int(preview_row.get("preview_rank", 0) or 0),
                    "pool_position": int(preview_row.get("pool_position", -1) or -1),
                })

    heuristic_selected_keys = {
        normalize_structure_text(title)
        for title in trace_payload.get("selected_titles", []) or []
        if str(title or "").strip()
    }
    final_selected_keys = {
        normalize_structure_text(title)
        for title in trace_payload.get("final_front_titles", []) or []
        if str(title or "").strip()
    }

    exposure_rows: List[Dict[str, object]] = []
    for title in normalized_targets:
        title_key = normalize_structure_text(title)
        pool_positions = list(pool_positions_by_key.get(title_key, []))
        source_entries = list(source_hits.get(title_key, []))
        shortlist_entries = list(shortlist_hits.get(title_key, []))
        scored_entries = list(scored_hits.get(title_key, []))
        in_pool = bool(pool_positions)
        in_source = bool(source_entries)
        in_shortlist = bool(shortlist_entries)
        in_selected_set = title_key in heuristic_selected_keys
        in_final_evidence = title_key in final_selected_keys
        if in_final_evidence and not in_selected_set:
            stage = "final_only"
        elif in_final_evidence:
            stage = "selected"
        elif in_shortlist:
            stage = "shortlist"
        elif in_source:
            stage = "source"
        elif in_pool:
            stage = "pool_only"
        else:
            stage = "not_in_pool"
        exposure_rows.append({
            "title": title,
            "stage": stage,
            "present_in_pool": bool(in_pool),
            "pool_positions": pool_positions[:8],
            "appears_in_candidate_source": bool(in_source),
            "appears_in_candidate_shortlist": bool(in_shortlist),
            "appears_in_selected_set": bool(in_selected_set),
            "appears_in_final_evidence": bool(in_final_evidence),
            "source_first_step": int(source_entries[0]["step"]) if source_entries else None,
            "shortlist_first_step": int(shortlist_entries[0]["step"]) if shortlist_entries else None,
            "source_steps": [int(entry["step"]) for entry in source_entries[:8]],
            "shortlist_steps": [int(entry["step"]) for entry in shortlist_entries[:8]],
            "source_preview_ranks": [int(entry["preview_rank"]) for entry in source_entries[:8]],
            "shortlist_preview_ranks": [int(entry["preview_rank"]) for entry in shortlist_entries[:8]],
            "best_scored_rank": min((int(entry["scored_rank"]) for entry in scored_entries), default=None),
            "best_combined_score": round(
                max((float(entry["combined_score"]) for entry in scored_entries), default=0.0),
                4,
            ) if scored_entries else None,
            "best_utility_margin_gain": round(
                max((float(entry["utility_margin_gain"]) for entry in scored_entries), default=0.0),
                4,
            ) if scored_entries else None,
            "best_support_completeness_gain": round(
                max((float(entry["support_completeness_gain"]) for entry in scored_entries), default=0.0),
                4,
            ) if scored_entries else None,
            "forced_into_source": bool(forced_source_hits.get(title_key, False)),
            "forced_into_shortlist": bool(forced_shortlist_hits.get(title_key, False)),
            "forced_into_final": bool(title_key in forced_final_hits and in_final_evidence),
        })
    return exposure_rows


def materialize_reader_top_positions(selected_positions: Sequence[int],
                                     pool_limit: int,
                                     qa_top_k: int,
                                     forced_prefix_positions: Sequence[int] | None = None) -> List[int]:
    normalized_positions: List[int] = []
    seen_positions: Set[int] = set()
    effective_pool_limit = max(int(pool_limit), 0)
    effective_top_k = min(effective_pool_limit, max(int(qa_top_k), 0))
    for raw_pos in forced_prefix_positions or []:
        pos = int(raw_pos)
        if pos < 0 or pos >= effective_pool_limit or pos in seen_positions:
            continue
        normalized_positions.append(pos)
        seen_positions.add(pos)
    for raw_pos in selected_positions:
        pos = int(raw_pos)
        if pos < 0 or pos >= effective_pool_limit or pos in seen_positions:
            continue
        normalized_positions.append(pos)
        seen_positions.add(pos)
    for pos in range(effective_pool_limit):
        if pos in seen_positions:
            continue
        normalized_positions.append(pos)
        if len(normalized_positions) >= effective_top_k:
            break
    return normalized_positions[:effective_top_k]


SETWISE_READER_ORDER_PROBE_MODES = {
    "none",
    "promote_best_bridge_to_slot2",
    "promote_best_bridge_to_slot3",
}


def normalize_setwise_reader_order_probe_mode(mode: str | None) -> str:
    normalized = str(mode or "none").strip().lower()
    if normalized not in SETWISE_READER_ORDER_PROBE_MODES:
        raise ValueError(f"Unsupported setwise reader order probe mode: {mode}")
    return normalized


def maybe_apply_setwise_reader_order_probe(final_front_positions: Sequence[int],
                                           selector_trace: Dict[str, object] | None,
                                           pool_doc_ids: Sequence[int | None],
                                           pool_doc_titles: Sequence[str] | None,
                                           probe_mode: str = "none") -> Tuple[List[int], Dict[str, object]]:
    normalized_mode = normalize_setwise_reader_order_probe_mode(probe_mode)
    target_rank = 2 if normalized_mode == "promote_best_bridge_to_slot2" else 3
    front_positions = [int(pos) for pos in final_front_positions]
    original_titles = [
        str(pool_doc_titles[pos]).strip()
        for pos in front_positions
        if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
    ]
    probe_trace: Dict[str, object] = {
        "enabled": normalized_mode != "none",
        "mode": normalized_mode,
        "applied": False,
        "skip_reason": "disabled" if normalized_mode == "none" else "",
        "target_rank": int(target_rank),
        "promoted_pool_position": None,
        "promoted_doc_id": None,
        "promoted_title": "",
        "promoted_from_rank": None,
        "original_front_pool_positions": list(front_positions),
        "original_front_titles": list(original_titles),
        "probed_front_pool_positions": list(front_positions),
        "probed_front_titles": list(original_titles),
    }
    if normalized_mode == "none":
        return list(front_positions), probe_trace

    if len(front_positions) < int(target_rank):
        probe_trace["skip_reason"] = "front_too_short"
        return list(front_positions), probe_trace

    front_rank_by_position = {
        int(pos): idx + 1
        for idx, pos in enumerate(front_positions)
    }
    bridge_candidates: List[Dict[str, object]] = []
    for step in list((selector_trace or {}).get("selection_steps", []) or []):
        if not isinstance(step, dict):
            continue
        step_mode = str(step.get("mode", "") or "").strip().lower()
        if step_mode not in {"beam", "greedy"}:
            continue
        pool_position = int(step.get("pool_position", -1) or -1)
        if pool_position not in front_rank_by_position:
            continue
        bridge_candidates.append({
            "pool_position": pool_position,
            "doc_id": step.get("doc_id"),
            "title": (
                str(pool_doc_titles[pool_position]).strip()
                if pool_doc_titles is not None and 0 <= pool_position < len(pool_doc_titles)
                else ""
            ),
            "current_rank": int(front_rank_by_position[pool_position]),
            "structure_score": float(step.get("structure_score", 0.0) or 0.0),
            "closure_score": float(step.get("closure_score", 0.0) or 0.0),
            "novelty_score": float(step.get("novelty_score", 0.0) or 0.0),
        })

    if not bridge_candidates:
        probe_trace["skip_reason"] = "no_bridge_doc_in_final_front"
        return list(front_positions), probe_trace

    bridge_candidates.sort(
        key=lambda item: (
            -float(item["structure_score"]),
            -float(item["closure_score"]),
            -float(item["novelty_score"]),
            int(item["pool_position"]),
        )
    )
    best_bridge = bridge_candidates[0]
    probe_trace["promoted_pool_position"] = int(best_bridge["pool_position"])
    probe_trace["promoted_doc_id"] = (
        int(best_bridge["doc_id"]) if best_bridge.get("doc_id") is not None else None
    )
    probe_trace["promoted_title"] = str(best_bridge.get("title", "") or "")
    probe_trace["promoted_from_rank"] = int(best_bridge["current_rank"])

    if int(best_bridge["current_rank"]) <= int(target_rank):
        probe_trace["skip_reason"] = "bridge_already_in_prefix"
        return list(front_positions), probe_trace

    promoted_position = int(best_bridge["pool_position"])
    reordered_front_positions = [pos for pos in front_positions if pos != promoted_position]
    insert_index = max(0, min(int(target_rank) - 1, len(reordered_front_positions)))
    reordered_front_positions.insert(insert_index, promoted_position)
    if reordered_front_positions == front_positions:
        probe_trace["skip_reason"] = "order_unchanged"
        return list(front_positions), probe_trace

    probe_trace["applied"] = True
    probe_trace["skip_reason"] = ""
    probe_trace["probed_front_pool_positions"] = list(reordered_front_positions)
    probe_trace["probed_front_titles"] = [
        str(pool_doc_titles[pos]).strip()
        for pos in reordered_front_positions
        if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
    ]
    return reordered_front_positions, probe_trace


def truncate_prompt_text(value: str | None, max_chars: int) -> str:
    cleaned = " ".join(str(value or "").split())
    if max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[: max_chars - 1] + "…"
    return cleaned


def normalize_llm_result(response: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(response, (tuple, list)) and len(response) >= 2:
        response_text = response[0]
        metadata = response[1]
        if isinstance(metadata, dict):
            return str(response_text), dict(metadata or {})
    return str(response), {}


def is_pydantic_response_format(response_format: Any) -> bool:
    return isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel)


def _lookup_response_value(payload: Any, field_name: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload.get(field_name)
    return getattr(payload, field_name, None)


def extract_responses_api_text(response: Any) -> Tuple[str, str]:
    direct_output_text = str(getattr(response, "output_text", "") or "").strip()
    if direct_output_text:
        return direct_output_text, "output_text"

    collected_fragments: List[str] = []
    output_items = _lookup_response_value(response, "output")
    if isinstance(output_items, Sequence) and not isinstance(output_items, (str, bytes)):
        for output_item in output_items:
            item_content = _lookup_response_value(output_item, "content")
            if not isinstance(item_content, Sequence) or isinstance(item_content, (str, bytes)):
                item_content = [output_item]
            for content_part in item_content:
                for field_name in ("text", "output_text", "content"):
                    value = _lookup_response_value(content_part, field_name)
                    if isinstance(value, str) and value.strip():
                        collected_fragments.append(value.strip())
                        break

    if collected_fragments:
        return "\n".join(collected_fragments), "output"
    return "", "empty"


def resolve_setwise_late_rerank_api_key(api_key: str | None,
                                        api_key_env: str | None,
                                        base_url: str | None) -> str:
    normalized_api_key = str(api_key).strip() if api_key else ""
    if normalized_api_key:
        return normalized_api_key

    normalized_env_name = str(api_key_env).strip() if api_key_env else ""
    if normalized_env_name:
        env_value = str(os.getenv(normalized_env_name, "")).strip()
        if env_value:
            return env_value

    normalized_base_url = str(base_url or "").strip().lower()
    if "localhost" in normalized_base_url or "127.0.0.1" in normalized_base_url:
        return "sk-"

    if not normalized_env_name:
        normalized_env_name = "OPENAI_API_KEY"
    raise ValueError(
        "No API key resolved for late rerank judge. "
        f"Provide --setwise_late_rerank_judge_api_key or set {normalized_env_name}."
    )


def build_setwise_late_rerank_judge_bundle(args: argparse.Namespace,
                                           fallback_model_name: str,
                                           fallback_base_url: str | None) -> SetwiseLateRerankJudgeBundle:
    backend = str(getattr(args, "setwise_late_rerank_judge_backend", "inherit") or "inherit").strip().lower()
    if backend == "inherit":
        return SetwiseLateRerankJudgeBundle(
            infer_fn=None,
            model_name=str(fallback_model_name),
            backend="inherit",
            base_url=str(fallback_base_url).strip() if fallback_base_url else None,
            response_format=None,
        )

    model_name = str(getattr(args, "setwise_late_rerank_judge_model", "") or "").strip() or str(fallback_model_name)
    base_url = str(getattr(args, "setwise_late_rerank_judge_base_url", "") or "").strip() or None
    api_key_env = str(getattr(args, "setwise_late_rerank_judge_api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY").strip()
    api_key = resolve_setwise_late_rerank_api_key(
        api_key=getattr(args, "setwise_late_rerank_judge_api_key", None),
        api_key_env=api_key_env,
        base_url=base_url or fallback_base_url,
    )
    reasoning_effort = str(getattr(args, "setwise_late_rerank_judge_reasoning_effort", "") or "").strip() or None
    timeout_s = float(getattr(args, "setwise_late_rerank_judge_timeout_s", 120.0) or 120.0)
    judge = OpenAICompatibleLateRerankJudge(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        backend=backend,
        timeout_s=timeout_s,
        reasoning_effort=reasoning_effort,
    )
    return SetwiseLateRerankJudgeBundle(
        infer_fn=judge.infer,
        model_name=model_name,
        backend=backend,
        base_url=base_url,
        response_format=None,
        reasoning_effort=reasoning_effort,
        api_key_env=api_key_env,
    )


def extract_setwise_late_rerank_payload(response_text: str) -> str:
    cleaned = str(response_text or "").strip()
    if not cleaned:
        return cleaned

    tag_start = cleaned.find(SETWISE_LLM_JSON_START_TAG)
    if tag_start != -1:
        tag_end = cleaned.find(SETWISE_LLM_JSON_END_TAG, tag_start + len(SETWISE_LLM_JSON_START_TAG))
        tagged_payload = (
            cleaned[tag_start + len(SETWISE_LLM_JSON_START_TAG):tag_end]
            if tag_end != -1 else
            cleaned[tag_start + len(SETWISE_LLM_JSON_START_TAG):]
        ).strip()
        if tagged_payload:
            return tagged_payload

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        return cleaned[object_start:object_end + 1]
    return cleaned


def parse_setwise_late_rerank_response(response_text: str,
                                       num_candidates: int) -> Dict[str, object]:
    payload_text = extract_setwise_late_rerank_payload(response_text)
    if not payload_text:
        return {
            "best_id": None,
            "confidence": None,
            "parse_succeeded": False,
            "parse_error": "empty_response",
            "payload_text": payload_text,
        }

    parsed_payload: Any = None
    parse_error = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed_payload = parser(payload_text)
            parse_error = None
            break
        except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
            parse_error = str(exc)

    if parsed_payload is None:
        return {
            "best_id": None,
            "confidence": None,
            "parse_succeeded": False,
            "parse_error": f"unparseable_payload: {parse_error}",
            "payload_text": payload_text,
        }

    raw_best_id: Any = None
    raw_confidence = None
    if isinstance(parsed_payload, dict):
        raw_best_id = parsed_payload.get("best_id")
        if raw_best_id is None:
            best_ids = parsed_payload.get("best_ids")
            if isinstance(best_ids, list) and best_ids:
                raw_best_id = best_ids[0]
        raw_confidence = parsed_payload.get("confidence")
    elif isinstance(parsed_payload, list) and parsed_payload:
        raw_best_id = parsed_payload[0]

    try:
        if isinstance(raw_best_id, bool):
            raise ValueError("best_id cannot be bool")
        best_id = int(raw_best_id)
    except (TypeError, ValueError):
        best_id = None

    confidence = None
    try:
        if raw_confidence is not None:
            confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = None

    if best_id is None or best_id < 0 or best_id >= max(int(num_candidates), 0):
        return {
            "best_id": None,
            "confidence": confidence,
            "parse_succeeded": False,
            "parse_error": f"invalid_best_id: {raw_best_id}",
            "payload_text": payload_text,
        }

    return {
        "best_id": int(best_id),
        "confidence": confidence,
        "parse_succeeded": True,
        "parse_error": None,
        "payload_text": payload_text,
    }


def build_setwise_late_rerank_candidates(selected_positions: Sequence[int],
                                         selector_trace: Dict[str, object] | None,
                                         pool_limit: int,
                                         qa_top_k: int,
                                         include_baseline: bool = True,
                                         max_candidates: int = 4) -> List[Dict[str, object]]:
    if max_candidates <= 0:
        return []

    candidates: List[Dict[str, object]] = [{
        "source": "heuristic_best",
        "selected_positions": [int(pos) for pos in selected_positions],
        "reader_top_positions": materialize_reader_top_positions(
            selected_positions=selected_positions,
            pool_limit=pool_limit,
            qa_top_k=qa_top_k,
        ),
        "state_score": float((selector_trace or {}).get("beam_best_state_score", 0.0) or 0.0),
        "cumulative_score": float((selector_trace or {}).get("beam_best_cumulative_score", 0.0) or 0.0),
    }]

    for finalist in (selector_trace or {}).get("beam_finalists", []):
        candidates.append({
            "source": str(finalist.get("source", "beam_finalist")),
            "selected_positions": [int(pos) for pos in finalist.get("selected_positions", [])],
            "reader_top_positions": materialize_reader_top_positions(
                selected_positions=finalist.get("selected_positions", []),
                pool_limit=pool_limit,
                qa_top_k=qa_top_k,
            ),
            "state_score": float(finalist.get("state_score", 0.0) or 0.0),
            "cumulative_score": float(finalist.get("cumulative_score", 0.0) or 0.0),
        })

    if include_baseline:
        baseline_positions = list(range(min(max(int(pool_limit), 0), max(int(qa_top_k), 0))))
        candidates.append({
            "source": "baseline_topk",
            "selected_positions": list(baseline_positions),
            "reader_top_positions": list(baseline_positions),
        })

    deduped_candidates: List[Dict[str, object]] = []
    seen_signatures: Set[Tuple[int, ...]] = set()
    for candidate in candidates:
        signature = tuple(int(pos) for pos in candidate.get("reader_top_positions", []))
        if not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_candidates.append(candidate)
        if len(deduped_candidates) >= int(max_candidates):
            break
    return deduped_candidates


def normalize_setwise_late_rerank_policy(policy: str | None) -> str:
    normalized = str(policy or "always").strip().lower()
    if normalized not in {"always", "tiebreak"}:
        raise ValueError(f"Unsupported late rerank policy: {policy}")
    return normalized


def should_apply_setwise_late_rerank_override(heuristic_candidate: Dict[str, object],
                                              chosen_candidate: Dict[str, object],
                                              policy: str = "always",
                                              max_state_score_gap: float = 0.0) -> Tuple[bool, Dict[str, object]]:
    normalized_policy = normalize_setwise_late_rerank_policy(policy)
    heuristic_state_score_raw = heuristic_candidate.get("state_score")
    chosen_state_score_raw = chosen_candidate.get("state_score")
    heuristic_state_score = (
        float(heuristic_state_score_raw)
        if heuristic_state_score_raw is not None
        else None
    )
    chosen_state_score = (
        float(chosen_state_score_raw)
        if chosen_state_score_raw is not None
        else None
    )
    info: Dict[str, object] = {
        "override_policy": normalized_policy,
        "override_max_state_score_gap": float(max_state_score_gap),
        "heuristic_state_score": heuristic_state_score,
        "selected_candidate_state_score": chosen_state_score,
        "override_state_score_gap": None,
        "override_block_reason": None,
    }
    if normalized_policy == "always":
        return True, info

    if heuristic_state_score is None or chosen_state_score is None:
        info["override_block_reason"] = "missing_state_score"
        return False, info

    state_score_gap = abs(heuristic_state_score - chosen_state_score)
    info["override_state_score_gap"] = float(state_score_gap)
    if state_score_gap > float(max_state_score_gap):
        info["override_block_reason"] = "state_score_gap_exceeded"
        return False, info

    return True, info


def rerank_completed_evidence_sets_with_llm(query: str,
                                            pool_docs: Sequence[str],
                                            candidate_sets: Sequence[Dict[str, object]],
                                            llm_infer_fn,
                                            model_name: str,
                                            response_format: Any = None,
                                            max_doc_chars: int = 280) -> Tuple[int | None, Dict[str, object]]:
    trace: Dict[str, object] = {
        "applied": False,
        "candidate_count": int(len(candidate_sets)),
        "parse_succeeded": False,
        "parse_error": None,
        "selected_candidate_id": None,
        "confidence": None,
        "response_preview": None,
        "metadata": {},
        "llm_error": None,
        "repair_applied": False,
        "repair_parse_succeeded": False,
        "repair_response_preview": None,
        "repair_error": None,
        "candidate_sources": [str(candidate.get("source", "unknown")) for candidate in candidate_sets],
        "candidates": [],
    }

    if len(candidate_sets) <= 1:
        return None, trace

    use_structured_output = response_format is not None

    for idx, candidate in enumerate(candidate_sets):
        reader_top_positions = [
            int(pos) for pos in candidate.get("reader_top_positions", [])
            if 0 <= int(pos) < len(pool_docs)
        ]
        trace["candidates"].append({
            "candidate_id": int(idx),
            "source": str(candidate.get("source", "unknown")),
            "reader_top_positions": reader_top_positions,
            "reader_top_titles": [extract_doc_title(pool_docs[pos]) for pos in reader_top_positions],
        })

    candidate_sections: List[str] = []
    for idx, candidate in enumerate(candidate_sets):
        candidate_sections.append(f"Candidate {idx}:")
        for rank, pos in enumerate(candidate.get("reader_top_positions", []), start=1):
            pool_pos = int(pos)
            if pool_pos < 0 or pool_pos >= len(pool_docs):
                continue
            doc_text = str(pool_docs[pool_pos])
            title = extract_doc_title(doc_text)
            body = doc_text.split("\n", 1)[1] if "\n" in doc_text else ""
            candidate_sections.append(f"[{rank}] Title: {title}")
            candidate_sections.append(f"Snippet: {truncate_prompt_text(body, max_doc_chars)}")
        candidate_sections.append("")

    messages = [
        {
            "role": "system",
            "content": (
                "You judge candidate evidence sets for multi-hop question answering. "
                "The JSON object may contain keys \"best_id\" and optional \"confidence\" only. "
                "\"best_id\" must be a single 0-based integer id from the provided candidates. "
                "Choose the evidence set that is most sufficient to answer the question using only the provided documents. "
                "Prefer complete support chains and penalize superficially related bridge documents that replace necessary evidence."
                + (
                    ""
                    if use_structured_output
                    else (
                        f" Return exactly one JSON object wrapped in {SETWISE_LLM_JSON_START_TAG} and {SETWISE_LLM_JSON_END_TAG}."
                        " Do not include markdown, code fences, or explanation."
                    )
                )
            ),
        },
        {
            "role": "user",
            "content": (
                (
                    ""
                    if use_structured_output
                    else f"{SETWISE_LLM_NO_THINK_PREFIX}\n"
                )
                + f"Question: {query}\n\n"
                "Candidate evidence sets:\n"
                + "\n".join(candidate_sections)
                + (
                    "\nReturn a JSON object with fields best_id and optional confidence."
                    if use_structured_output
                    else "\nReturn exactly one line in this shape:\n"
                    f"{SETWISE_LLM_JSON_START_TAG}{{\"best_id\":1,\"confidence\":0.72}}{SETWISE_LLM_JSON_END_TAG}"
                )
            ),
        },
    ]

    try:
        response = llm_infer_fn(
            messages=messages,
            model=model_name,
            response_format=response_format,
            max_completion_tokens=SETWISE_LLM_LATE_RERANK_MAX_COMPLETION_TOKENS,
            temperature=0,
            top_p=1,
        )
        response_text, metadata = normalize_llm_result(response)
    except Exception as exc:
        trace["applied"] = True
        trace["llm_error"] = str(exc)
        return None, trace

    parse_info = parse_setwise_late_rerank_response(
        response_text=response_text,
        num_candidates=len(candidate_sets),
    )
    if not parse_info["parse_succeeded"]:
        trace["repair_applied"] = True
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You repair evidence-set reranker outputs. "
                    "The JSON object may contain keys \"best_id\" and optional \"confidence\" only. "
                    "\"best_id\" must be a single 0-based integer id from the provided candidates."
                    + (
                        ""
                        if use_structured_output
                        else (
                            f" Return exactly one JSON object wrapped in {SETWISE_LLM_JSON_START_TAG} and {SETWISE_LLM_JSON_END_TAG}."
                            " Do not include markdown, code fences, or explanation."
                        )
                    )
                ),
            },
            {
                "role": "user",
                "content": (
                    (
                        ""
                        if use_structured_output
                        else f"{SETWISE_LLM_NO_THINK_PREFIX}\n"
                    )
                    + f"Question: {query}\n"
                    f"Valid candidate ids: 0 to {max(len(candidate_sets) - 1, 0)}\n"
                    f"Previous output preview: {truncate_prompt_text(response_text, 240)}\n"
                    f"Issue: {parse_info['parse_error']}\n"
                    + (
                        "Return a JSON object with fields best_id and optional confidence."
                        if use_structured_output
                        else "Return exactly one line in this shape:\n"
                        f"{SETWISE_LLM_JSON_START_TAG}{{\"best_id\":1,\"confidence\":0.72}}{SETWISE_LLM_JSON_END_TAG}"
                    )
                ),
            },
        ]
        try:
            repair_response = llm_infer_fn(
                messages=repair_messages,
                model=model_name,
                response_format=response_format,
                max_completion_tokens=SETWISE_LLM_LATE_RERANK_REPAIR_MAX_COMPLETION_TOKENS,
                temperature=0,
                top_p=1,
            )
            repair_response_text, repair_metadata = normalize_llm_result(repair_response)
            repair_parse_info = parse_setwise_late_rerank_response(
                response_text=repair_response_text,
                num_candidates=len(candidate_sets),
            )
            trace["repair_parse_succeeded"] = bool(repair_parse_info["parse_succeeded"])
            trace["repair_response_preview"] = truncate_prompt_text(repair_response_text, 240)
            if repair_parse_info["parse_succeeded"]:
                parse_info = repair_parse_info
                response_text = repair_response_text
                metadata = repair_metadata
        except Exception as exc:
            trace["repair_error"] = str(exc)

    trace["applied"] = True
    trace["parse_succeeded"] = bool(parse_info["parse_succeeded"])
    trace["parse_error"] = parse_info["parse_error"]
    trace["selected_candidate_id"] = parse_info["best_id"]
    trace["confidence"] = parse_info["confidence"]
    trace["response_preview"] = truncate_prompt_text(response_text, 240)
    trace["metadata"] = metadata
    return (
        int(parse_info["best_id"]) if parse_info["best_id"] is not None else None,
        trace,
    )


def resolve_reserved_positions(candidate_count: int,
                               target_k: int,
                               anchor_count: int,
                               reserve_top_m: int,
                               pool_doc_titles: Sequence[str] | None = None,
                               dedup_titles: bool = False) -> Tuple[List[int], List[int]]:
    actual_anchor_count = min(max(anchor_count, 0), target_k, candidate_count)
    actual_reserve_count = min(max(actual_anchor_count, max(reserve_top_m, 0)), target_k, candidate_count)
    if not dedup_titles or pool_doc_titles is None:
        anchor_positions = list(range(actual_anchor_count))
        reserved_positions = list(range(actual_reserve_count))
        return anchor_positions, reserved_positions

    deduped_positions: List[int] = []
    seen_titles: Set[str] = set()
    for pos in range(candidate_count):
        title = str(pool_doc_titles[pos]).strip() if pos < len(pool_doc_titles) else ""
        if title and title in seen_titles:
            continue
        deduped_positions.append(pos)
        if title:
            seen_titles.add(title)
        if len(deduped_positions) >= actual_reserve_count:
            break

    anchor_positions = deduped_positions[:actual_anchor_count]
    reserved_positions = deduped_positions[:actual_reserve_count]
    return anchor_positions, reserved_positions


def filter_title_dedup_candidates(scored_candidates: Sequence[Dict[str, object]],
                                  blocked_titles: Set[str],
                                  enabled: bool) -> List[Dict[str, object]]:
    if not enabled or not blocked_titles:
        return list(scored_candidates)

    filtered_candidates = [
        candidate for candidate in scored_candidates
        if str(candidate.get("doc_title", "")).strip() not in blocked_titles
    ]
    return filtered_candidates or list(scored_candidates)


def resolve_selection_target_k(target_k: int,
                               reserved_count: int,
                               max_bridge_slots: int) -> int:
    if max_bridge_slots <= 0:
        return int(target_k)
    return int(min(target_k, max(0, reserved_count) + max(0, max_bridge_slots)))


def resolve_requirement_beam_runtime_reserve_config(anchor_count: int,
                                                    reserve_top_m: int,
                                                    cache_entry: Dict[str, Any],
                                                    policy: str = DEFAULT_REQUIREMENT_BEAM_RESERVE_POLICY) -> Dict[str, int | str]:
    fixed_anchor_count = max(int(anchor_count), 0)
    fixed_reserve_top_m = max(int(reserve_top_m), 0)
    normalized_policy = str(policy or DEFAULT_REQUIREMENT_BEAM_RESERVE_POLICY).strip().lower()
    if normalized_policy not in {"fixed", "adaptive_requirement_count"}:
        raise ValueError(f"Unsupported requirement beam reserve policy: {policy}")

    positive_requirement_count = int(len(cache_entry.get("positive_requirements", []) or []))
    runtime_anchor_count = fixed_anchor_count
    runtime_reserve_top_m = fixed_reserve_top_m
    policy_reason = "fixed"

    if normalized_policy == "adaptive_requirement_count":
        if positive_requirement_count <= 2:
            runtime_anchor_count = min(fixed_anchor_count, 1)
            runtime_reserve_top_m = 1 if (fixed_anchor_count > 0 or fixed_reserve_top_m > 0) else 0
            policy_reason = "positive_requirements<=2"
        elif positive_requirement_count <= 4:
            runtime_anchor_count = min(fixed_anchor_count, 1)
            runtime_reserve_top_m = min(max(fixed_anchor_count, fixed_reserve_top_m), 2)
            policy_reason = "positive_requirements<=4"
        else:
            policy_reason = "positive_requirements>4"

    effective_reserved_count = max(runtime_anchor_count, runtime_reserve_top_m)
    return {
        "policy": normalized_policy,
        "policy_reason": policy_reason,
        "positive_requirement_count": positive_requirement_count,
        "fixed_anchor_count": fixed_anchor_count,
        "fixed_reserve_top_m": fixed_reserve_top_m,
        "anchor_count": int(runtime_anchor_count),
        "reserve_top_m": int(runtime_reserve_top_m),
        "effective_reserved_count": int(effective_reserved_count),
    }


def normalize_setwise_score_mode(score_mode: str | None) -> str:
    normalized = str(score_mode or "bridge").strip().lower()
    if normalized not in {"bridge", "closure_proxy", "set_closure"}:
        raise ValueError(f"Unsupported setwise score mode: {score_mode}")
    return normalized


def normalize_requirement_source_sort_mode(sort_mode: str | None) -> str:
    normalized = str(sort_mode or DEFAULT_REQUIREMENT_SOURCE_SORT_MODE).strip().lower()
    if normalized not in {"combined", "support_bonus"}:
        raise ValueError(f"Unsupported requirement source sort mode: {sort_mode}")
    return normalized


def normalize_requirement_shortlist_sort_mode(sort_mode: str | None) -> str:
    normalized = str(sort_mode or DEFAULT_REQUIREMENT_SHORTLIST_SORT_MODE).strip().lower()
    if normalized not in {"margin_first", "positive_only"}:
        raise ValueError(f"Unsupported requirement shortlist sort mode: {sort_mode}")
    return normalized


def resolve_set_closure_state_weight_config(
    state_weight_config: Dict[str, float] | None = None,
) -> Dict[str, float]:
    resolved = dict(DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG)
    if state_weight_config:
        for key in resolved:
            if key in state_weight_config:
                resolved[key] = float(state_weight_config[key])
    return resolved


def normalize_entity_set(entities: Sequence[str] | Set[str] | None) -> Set[str]:
    normalized_entities: Set[str] = set()
    for entity in entities or []:
        normalized = normalize_structure_text(str(entity))
        if normalized:
            normalized_entities.add(normalized)
    return normalized_entities


def score_doc_path_attachment(doc_entities: Set[str],
                              reachable_entities: Set[str],
                              frontier_scores: Dict[str, float]) -> float:
    if not doc_entities:
        return 0.0

    overlap_support = 1.0 if doc_entities & reachable_entities else 0.0
    frontier_support = max(
        (float(frontier_scores.get(entity, 0.0)) for entity in doc_entities if entity in frontier_scores),
        default=0.0,
    )
    return float(max(overlap_support, frontier_support))


def finalize_state_path_connectivity_metrics(attachment_scores: Sequence[float],
                                             reachable_entities: Set[str],
                                             normalized_query: Set[str],
                                             total_focus: int,
                                             connected_doc_count: int,
                                             mode: str) -> Dict[str, float]:
    query_denominator = max(1, len(normalized_query))
    query_reachability = len(reachable_entities & normalized_query) / float(query_denominator)
    return {
        "path_connectivity": float(sum(attachment_scores) / float(total_focus)),
        "reachable_doc_ratio": float(connected_doc_count / float(total_focus)),
        "query_reachability": float(query_reachability),
        "connected_doc_count": float(connected_doc_count),
        "path_search_mode": str(mode),
    }


def compute_greedy_state_path_connectivity_metrics(selected_doc_entities: Dict[int, Set[str]],
                                                   ordered_focus_positions: Sequence[int],
                                                   initial_reachable_entities: Set[str],
                                                   normalized_query: Set[str],
                                                   adjacency: Dict[str, List[Tuple[str, float, str]]],
                                                   structure_max_hops: int) -> Dict[str, float]:
    reachable_entities = set(initial_reachable_entities)
    remaining_positions = list(ordered_focus_positions)
    connected_doc_count = 0
    attachment_scores: List[float] = []

    while remaining_positions:
        frontier_scores = (
            expand_directed_entities(reachable_entities, adjacency, max_hops=structure_max_hops)
            if reachable_entities
            else {}
        )
        best_index = -1
        best_score = 0.0
        best_query_gain = -1
        best_new_entity_count = -1
        best_pos = 0

        for idx, pos in enumerate(remaining_positions):
            doc_entities = selected_doc_entities.get(pos, set())
            attachment_score = score_doc_path_attachment(
                doc_entities=doc_entities,
                reachable_entities=reachable_entities,
                frontier_scores=frontier_scores,
            )
            if attachment_score <= 0.0:
                continue

            query_gain = len((doc_entities - reachable_entities) & normalized_query)
            new_entity_count = len(doc_entities - reachable_entities)
            if (
                attachment_score > best_score
                or (
                    np.isclose(attachment_score, best_score)
                    and (query_gain, new_entity_count, -pos)
                    > (best_query_gain, best_new_entity_count, -best_pos)
                )
            ):
                best_index = idx
                best_score = attachment_score
                best_query_gain = query_gain
                best_new_entity_count = new_entity_count
                best_pos = pos

        if best_index < 0:
            break

        chosen_pos = remaining_positions.pop(best_index)
        reachable_entities.update(selected_doc_entities.get(chosen_pos, set()))
        connected_doc_count += 1
        attachment_scores.append(float(best_score))

    return finalize_state_path_connectivity_metrics(
        attachment_scores=attachment_scores,
        reachable_entities=reachable_entities,
        normalized_query=normalized_query,
        total_focus=len(ordered_focus_positions),
        connected_doc_count=connected_doc_count,
        mode="greedy",
    )


def compute_exact_state_path_connectivity_metrics(selected_doc_entities: Dict[int, Set[str]],
                                                  ordered_focus_positions: Sequence[int],
                                                  initial_reachable_entities: Set[str],
                                                  normalized_query: Set[str],
                                                  adjacency: Dict[str, List[Tuple[str, float, str]]],
                                                  structure_max_hops: int) -> Dict[str, float]:
    total_focus = len(ordered_focus_positions)
    best_metrics = finalize_state_path_connectivity_metrics(
        attachment_scores=[],
        reachable_entities=set(initial_reachable_entities),
        normalized_query=normalized_query,
        total_focus=total_focus,
        connected_doc_count=0,
        mode="exact",
    )

    def metric_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
        return (
            float(metrics["query_reachability"]),
            float(metrics["reachable_doc_ratio"]),
            float(metrics["path_connectivity"]),
            float(metrics["connected_doc_count"]),
        )

    def dfs(remaining_positions: Tuple[int, ...],
            reachable_entities: Set[str],
            attachment_scores: List[float],
            connected_doc_count: int) -> None:
        nonlocal best_metrics

        current_metrics = finalize_state_path_connectivity_metrics(
            attachment_scores=attachment_scores,
            reachable_entities=reachable_entities,
            normalized_query=normalized_query,
            total_focus=total_focus,
            connected_doc_count=connected_doc_count,
            mode="exact",
        )
        if metric_key(current_metrics) > metric_key(best_metrics):
            best_metrics = current_metrics

        if not remaining_positions:
            return

        frontier_scores = (
            expand_directed_entities(reachable_entities, adjacency, max_hops=structure_max_hops)
            if reachable_entities
            else {}
        )
        candidate_steps: List[Tuple[int, float, int, int]] = []
        for pos in remaining_positions:
            doc_entities = selected_doc_entities.get(pos, set())
            attachment_score = score_doc_path_attachment(
                doc_entities=doc_entities,
                reachable_entities=reachable_entities,
                frontier_scores=frontier_scores,
            )
            if attachment_score <= 0.0:
                continue
            candidate_steps.append((
                int(pos),
                float(attachment_score),
                len((doc_entities - reachable_entities) & normalized_query),
                len(doc_entities - reachable_entities),
            ))

        candidate_steps.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        for pos, attachment_score, _, _ in candidate_steps:
            next_remaining = tuple(candidate for candidate in remaining_positions if candidate != pos)
            next_reachable = set(reachable_entities)
            next_reachable.update(selected_doc_entities.get(pos, set()))
            dfs(
                remaining_positions=next_remaining,
                reachable_entities=next_reachable,
                attachment_scores=list(attachment_scores) + [float(attachment_score)],
                connected_doc_count=connected_doc_count + 1,
            )

    dfs(
        remaining_positions=tuple(int(pos) for pos in ordered_focus_positions),
        reachable_entities=set(initial_reachable_entities),
        attachment_scores=[],
        connected_doc_count=0,
    )
    return best_metrics


def compute_state_path_connectivity_metrics(selected_doc_entities: Dict[int, Set[str]],
                                            focus_positions: Sequence[int],
                                            initial_reachable_entities: Sequence[str] | Set[str] | None,
                                            query_entities: Sequence[str] | Set[str] | None,
                                            adjacency: Dict[str, List[Tuple[str, float, str]]],
                                            structure_max_hops: int) -> Dict[str, float]:
    ordered_focus_positions = [int(pos) for pos in focus_positions]
    total_focus = len(ordered_focus_positions)
    if total_focus <= 0:
        return {
            "path_connectivity": 0.0,
            "reachable_doc_ratio": 0.0,
            "query_reachability": 0.0,
            "connected_doc_count": 0.0,
            "path_search_mode": "none",
        }

    normalized_query = normalize_entity_set(query_entities)
    reachable_entities = normalize_entity_set(initial_reachable_entities)
    if total_focus <= DEFAULT_SET_CLOSURE_EXACT_PATH_MAX_DOCS:
        return compute_exact_state_path_connectivity_metrics(
            selected_doc_entities=selected_doc_entities,
            ordered_focus_positions=ordered_focus_positions,
            initial_reachable_entities=reachable_entities,
            normalized_query=normalized_query,
            adjacency=adjacency,
            structure_max_hops=structure_max_hops,
        )
    return compute_greedy_state_path_connectivity_metrics(
        selected_doc_entities=selected_doc_entities,
        ordered_focus_positions=ordered_focus_positions,
        initial_reachable_entities=reachable_entities,
        normalized_query=normalized_query,
        adjacency=adjacency,
        structure_max_hops=structure_max_hops,
    )


def should_keep_set_closure_expansion(candidate: Dict[str, object],
                                      current_state_metrics: Dict[str, float] | None,
                                      next_state_metrics: Dict[str, float],
                                      eps: float = 1e-9) -> bool:
    current_metrics = current_state_metrics or {}
    candidate_positive_signal = max(
        float(candidate.get("selection_score_raw", 0.0) or 0.0),
        float(candidate.get("closure_score_raw", 0.0) or 0.0),
        float(candidate.get("structure_score", 0.0) or 0.0),
        float(candidate.get("path_coherence_score", 0.0) or 0.0),
        float(candidate.get("query_anchor_score", 0.0) or 0.0),
        float(candidate.get("frontier_gain_score", 0.0) or 0.0),
    ) > float(eps)
    if candidate_positive_signal:
        return True

    for metric_name in ("path_connectivity", "reachable_doc_ratio", "query_reachability", "query_coverage"):
        if float(next_state_metrics.get(metric_name, 0.0) or 0.0) > float(current_metrics.get(metric_name, 0.0) or 0.0) + float(eps):
            return True
    return False


def compute_selector_trace_avg_local_structure(selector_trace: Dict[str, object] | None) -> Tuple[float, int]:
    structure_values: List[float] = []
    for step in (selector_trace or {}).get("selection_steps", []) or []:
        if not isinstance(step, dict) or str(step.get("mode", "")) != "beam":
            continue
        structure_values.append(float(step.get("structure_score", 0.0) or 0.0))
    if not structure_values:
        return 0.0, 0
    return float(np.mean(structure_values)), int(len(structure_values))


def maybe_apply_bridge_saturation_guard(selected_positions: Sequence[int],
                                        selector_trace: Dict[str, object] | None,
                                        gate_decision: Dict[str, object] | None,
                                        pool_doc_titles: Sequence[str] | None,
                                        gate_max_avg_local_structure: float,
                                        gate_min_suffix_base_mean: float) -> Tuple[List[int], Dict[str, object], Dict[str, object]]:
    updated_positions = [int(pos) for pos in selected_positions]
    updated_selector_trace = dict(selector_trace or {})
    updated_gate_decision = dict(gate_decision or {})
    normalized_gate_mode = str(updated_gate_decision.get("gate_mode", "none") or "none").strip().lower()

    avg_local_structure, local_structure_doc_count = compute_selector_trace_avg_local_structure(updated_selector_trace)
    suffix_base_mean = float(updated_selector_trace.get("beam_best_state_suffix_base_mean", 0.0) or 0.0)

    updated_gate_decision["gate_max_avg_local_structure"] = round(float(gate_max_avg_local_structure), 4)
    updated_gate_decision["gate_min_suffix_base_mean"] = round(float(gate_min_suffix_base_mean), 4)
    updated_gate_decision["avg_local_structure"] = round(float(avg_local_structure), 4)
    updated_gate_decision["local_structure_doc_count"] = int(local_structure_doc_count)
    updated_gate_decision["suffix_base_mean"] = round(float(suffix_base_mean), 4)
    updated_gate_decision["saturation_guard_enabled"] = normalized_gate_mode == "suffix_bridge_saturation_guard"
    updated_gate_decision["saturation_guard_triggered"] = False

    updated_selector_trace["avg_local_structure"] = round(float(avg_local_structure), 4)
    updated_selector_trace["local_structure_doc_count"] = int(local_structure_doc_count)
    updated_selector_trace["saturation_guard_suffix_base_mean"] = round(float(suffix_base_mean), 4)
    updated_selector_trace["saturation_guard_enabled"] = normalized_gate_mode == "suffix_bridge_saturation_guard"
    updated_selector_trace["saturation_guard_triggered"] = False

    if normalized_gate_mode != "suffix_bridge_saturation_guard":
        return updated_positions, updated_selector_trace, updated_gate_decision
    if not bool(updated_gate_decision.get("use_selector", False)):
        return updated_positions, updated_selector_trace, updated_gate_decision
    if local_structure_doc_count <= 0:
        return updated_positions, updated_selector_trace, updated_gate_decision

    if float(avg_local_structure) > float(gate_max_avg_local_structure) and float(suffix_base_mean) < float(gate_min_suffix_base_mean):
        updated_gate_decision["pre_saturation_guard_reason"] = str(updated_gate_decision.get("reason", ""))
        updated_gate_decision["use_selector"] = False
        updated_gate_decision["reason"] = "structure_saturated_weak_suffix"
        updated_gate_decision["saturation_guard_triggered"] = True
        updated_selector_trace["saturation_guard_triggered"] = True
        updated_selector_trace["saturation_guard_reason"] = "structure_saturated_weak_suffix"
        updated_selector_trace["saturation_guard_original_selected_positions"] = list(updated_positions)
        updated_selector_trace["saturation_guard_original_selected_titles"] = [
            str(pool_doc_titles[pos]).strip()
            for pos in updated_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ]
        return [], updated_selector_trace, updated_gate_decision

    return updated_positions, updated_selector_trace, updated_gate_decision


def collect_query_seed_entities(hipporag: HippoRAG, query: str) -> Set[str]:
    try:
        query_fact_scores = hipporag.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, _ = hipporag.rerank_facts(query, query_fact_scores)
        if hasattr(hipporag, "_collect_structure_seed_entities"):
            return normalize_entity_set(
                hipporag._collect_structure_seed_entities(
                    query_fact_scores=query_fact_scores,
                    top_k_fact_indices=top_k_fact_indices,
                    top_k_facts=top_k_facts,
                )
            )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to derive structure seed entities for query %r: %s",
            query[:160],
            exc,
        )
    return set()


def collect_lexical_query_seed_entities(query: str,
                                        pool_doc_ids: Sequence[int | None],
                                        doc_idx_to_entities: Dict[int, Set[str]],
                                        max_seed_entities: int = 8) -> Set[str]:
    normalized_query = normalize_structure_text(query)
    if not normalized_query:
        return set()

    query_tokens = {token for token in normalized_query.split() if len(token) > 1}
    scored_entities: List[Tuple[float, int, str]] = []
    seen_entities: Set[str] = set()
    for doc_id in pool_doc_ids:
        if doc_id is None:
            continue
        for entity in doc_idx_to_entities.get(int(doc_id), set()):
            normalized_entity = normalize_structure_text(entity)
            if not normalized_entity or normalized_entity in seen_entities:
                continue
            if len(normalized_entity) < 3:
                continue
            entity_tokens = {token for token in normalized_entity.split() if len(token) > 1}
            if not entity_tokens:
                continue
            overlap = len(entity_tokens & query_tokens)
            substring_match = (
                (len(normalized_entity) >= 4 and normalized_entity in normalized_query)
                or normalized_query in normalized_entity
            )
            if overlap <= 0 and not substring_match:
                continue
            score = 0.0
            if substring_match:
                score += 2.0
            score += overlap / max(1, len(entity_tokens))
            score += overlap / max(1, len(query_tokens))
            scored_entities.append((score, len(normalized_entity), normalized_entity))
            seen_entities.add(normalized_entity)

    scored_entities.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return {entity for _, _, entity in scored_entities[:max_seed_entities]}


def collect_question_query_entities(hipporag: HippoRAG,
                                    query: str,
                                    pool_doc_ids: Sequence[int | None],
                                    doc_idx_to_entities: Dict[int, Set[str]],
                                    max_query_entities: int = 8) -> Set[str]:
    extracted_entities: Set[str] = set()
    causal_engine = getattr(hipporag, "causal_v2_engine", None)
    extract_fn = getattr(causal_engine, "_extract_query_entities", None) if causal_engine is not None else None
    if callable(extract_fn):
        try:
            extracted_entities = normalize_entity_set(extract_fn(query))
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to derive question entities for query %r: %s",
                query[:160],
                exc,
            )

    if extracted_entities:
        return extracted_entities

    return collect_lexical_query_seed_entities(
        query=query,
        pool_doc_ids=pool_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        max_seed_entities=max_query_entities,
    )


def collect_pool_entities(pool_doc_ids: Sequence[int | None],
                          doc_idx_to_entities: Dict[int, Set[str]],
                          top_k: int | None = None) -> Set[str]:
    collected_entities: Set[str] = set()
    limited_doc_ids = list(pool_doc_ids[:max(int(top_k or 0), 0)]) if top_k is not None else list(pool_doc_ids)
    if top_k is not None and int(top_k) <= 0:
        limited_doc_ids = []
    for doc_id in limited_doc_ids:
        if doc_id is None:
            continue
        collected_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))
    return collected_entities


def collect_grounded_question_query_entities(seed_entities: Sequence[str] | Set[str] | None,
                                             question_entities: Sequence[str] | Set[str] | None,
                                             pool_doc_ids: Sequence[int | None],
                                             doc_idx_to_entities: Dict[int, Set[str]],
                                             adjacency: Dict[str, List[Tuple[str, float, str]]],
                                             structure_max_hops: int,
                                             max_query_entities: int = 8,
                                             top_pool_k: int = 20) -> Set[str]:
    normalized_seed = normalize_entity_set(seed_entities)
    normalized_question = normalize_entity_set(question_entities)
    if not normalized_question:
        return set()

    top_pool_entities = collect_pool_entities(
        pool_doc_ids=pool_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        top_k=top_pool_k,
    )
    seed_frontier = (
        expand_directed_entities(normalized_seed, adjacency, max_hops=structure_max_hops)
        if normalized_seed
        else {}
    )

    scored_entities: List[Tuple[float, int, str]] = []
    for entity in normalized_question:
        score = 0.0
        if entity in normalized_seed:
            score += 4.0
        if entity in seed_frontier:
            score += 3.0
        if entity in top_pool_entities:
            score += 2.0
        if score < 2.0:
            continue
        scored_entities.append((score, len(entity), entity))

    if not scored_entities:
        return set()

    scored_entities.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return {entity for _, _, entity in scored_entities[:max(max_query_entities, 1)]}


def resolve_setwise_query_targets(query_entity_source: str,
                                  seed_entities: Sequence[str] | Set[str] | None,
                                  question_entities: Sequence[str] | Set[str] | None,
                                  pool_doc_ids: Sequence[int | None],
                                  doc_idx_to_entities: Dict[int, Set[str]],
                                  adjacency: Dict[str, List[Tuple[str, float, str]]],
                                  structure_max_hops: int) -> Dict[str, Set[str]]:
    normalized_source = str(query_entity_source or "seed").strip().lower()
    normalized_seed = normalize_entity_set(seed_entities)
    normalized_question = normalize_entity_set(question_entities)
    grounded_question = collect_grounded_question_query_entities(
        seed_entities=normalized_seed,
        question_entities=normalized_question,
        pool_doc_ids=pool_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        adjacency=adjacency,
        structure_max_hops=structure_max_hops,
    )

    if normalized_source == "question":
        effective_question = normalized_question or grounded_question or set(normalized_seed)
        return {
            "proposal_query_entities": set(effective_question),
            "state_query_entities": set(effective_question),
            "state_support_query_entities": set(effective_question),
            "grounded_question_entities": set(grounded_question),
        }

    if normalized_source == "hybrid":
        proposal_query_entities = set(normalized_seed or grounded_question or normalized_question)
        state_query_entities = set(grounded_question or normalized_seed or normalized_question)
        state_support_query_entities = set(normalized_seed or state_query_entities)
        return {
            "proposal_query_entities": proposal_query_entities,
            "state_query_entities": state_query_entities,
            "state_support_query_entities": state_support_query_entities,
            "grounded_question_entities": set(grounded_question),
        }

    effective_seed = set(normalized_seed or grounded_question or normalized_question)
    return {
        "proposal_query_entities": set(effective_seed),
        "state_query_entities": set(effective_seed),
        "state_support_query_entities": set(effective_seed),
        "grounded_question_entities": set(grounded_question),
    }


def normalized_token_set(text: str | None) -> Set[str]:
    normalized_text = normalize_structure_text(text or "")
    if not normalized_text:
        return set()
    return {token for token in normalized_text.split() if len(token) > 1}


def compute_candidate_feature_rows(query: str,
                                   pool_docs: Sequence[str],
                                   pool_doc_ids: Sequence[int | None],
                                   pool_doc_scores: np.ndarray,
                                   doc_idx_to_entities: Dict[int, Set[str]],
                                   doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                   adjacency: Dict[str, List[Tuple[str, float, str]]],
                                   qa_top_k: int,
                                   selected_positions: Sequence[int] | None = None,
                                   seed_entities: Sequence[str] | Set[str] | None = None,
                                   structure_max_hops: int = 2,
                                   structure_seed_target_bridge_mode: str = "off",
                                   candidate_positions: Sequence[int] | None = None) -> List[Dict[str, float | int | str | None]]:
    pool_size = len(pool_doc_ids)
    normalized_base_scores = min_max_normalize_array(np.asarray(pool_doc_scores, dtype=float))
    selected_positions = list(selected_positions or [])
    candidate_positions = list(candidate_positions or range(pool_size))
    normalized_seed_entities = normalize_entity_set(seed_entities)
    query_tokens = normalized_token_set(query)

    selected_entity_union: Set[str] = set()
    for pos in selected_positions:
        if pos >= pool_size:
            continue
        doc_id = pool_doc_ids[pos]
        if doc_id is None:
            continue
        selected_entity_union.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))
    covered_entities = normalized_seed_entities | selected_entity_union

    candidate_doc_ids = [
        int(pool_doc_ids[pos])
        for pos in candidate_positions
        if pos < pool_size and pool_doc_ids[pos] is not None
    ]
    structure_scores_seed = (
        score_candidate_docs_by_structure(
            candidate_doc_ids=candidate_doc_ids,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            seed_entities=normalized_seed_entities,
            adjacency=adjacency,
            max_hops=structure_max_hops,
            seed_target_bridge_mode=structure_seed_target_bridge_mode,
        )
        if candidate_doc_ids and normalized_seed_entities
        else {}
    )
    structure_scores_covered = (
        score_candidate_docs_by_structure(
            candidate_doc_ids=candidate_doc_ids,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            seed_entities=covered_entities,
            adjacency=adjacency,
            max_hops=structure_max_hops,
            seed_target_bridge_mode=structure_seed_target_bridge_mode,
        )
        if candidate_doc_ids and covered_entities
        else {}
    )

    rows: List[Dict[str, float | int | str | None]] = []
    for pos in candidate_positions:
        if pos >= pool_size:
            continue
        doc_id = pool_doc_ids[pos]
        doc_entities = (
            normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
            if doc_id is not None
            else set()
        )
        doc_edges = doc_idx_to_edges.get(int(doc_id), []) if doc_id is not None else []
        title = pool_docs[pos].split("\n", 1)[0] if pos < len(pool_docs) else ""
        title_tokens = normalized_token_set(title)
        entity_count = len(doc_entities)
        edge_count = len(doc_edges)

        seed_overlap = len(doc_entities & normalized_seed_entities)
        covered_overlap = len(doc_entities & covered_entities)
        selected_overlap = len(doc_entities & selected_entity_union)
        novelty_score = len(doc_entities - covered_entities) / max(1, entity_count) if doc_entities else 0.0

        row = {
            "pool_position": int(pos),
            "doc_id": int(doc_id) if doc_id is not None else None,
            "title": title,
            "base_score": float(normalized_base_scores[pos]) if pos < len(normalized_base_scores) else 0.0,
            "rank_fraction": 1.0 - (float(pos) / max(1, pool_size - 1)) if pool_size > 1 else 1.0,
            "reciprocal_rank": 1.0 / float(pos + 1),
            "selected_count_fraction": len(selected_positions) / max(1, qa_top_k),
            "seed_entity_overlap_ratio": seed_overlap / max(1, len(normalized_seed_entities)) if normalized_seed_entities else 0.0,
            "covered_entity_overlap_ratio": covered_overlap / max(1, len(covered_entities)) if covered_entities else 0.0,
            "selected_entity_overlap_ratio": selected_overlap / max(1, len(selected_entity_union)) if selected_entity_union else 0.0,
            "query_token_overlap_ratio": len(title_tokens & query_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0,
            "novelty_score": novelty_score,
            "structure_score_seed": float(structure_scores_seed.get(int(doc_id), 0.0)) if doc_id is not None else 0.0,
            "structure_score_covered": float(structure_scores_covered.get(int(doc_id), 0.0)) if doc_id is not None else 0.0,
            "reachable_from_seed": float(structure_scores_seed.get(int(doc_id), 0.0) > 0.0) if doc_id is not None else 0.0,
            "reachable_from_covered": float(structure_scores_covered.get(int(doc_id), 0.0) > 0.0) if doc_id is not None else 0.0,
            "entity_count_norm": min(entity_count, 20) / 20.0,
            "edge_count_norm": min(edge_count, 20) / 20.0,
        }
        rows.append(row)

    return rows


def feature_rows_to_matrix(feature_rows: Sequence[Dict[str, float | int | str | None]]) -> np.ndarray:
    if not feature_rows:
        return np.zeros((0, len(LEARNED_SETWISE_FEATURE_NAMES)), dtype=float)
    matrix = np.asarray([
        [float(row.get(feature_name, 0.0) or 0.0) for feature_name in LEARNED_SETWISE_FEATURE_NAMES]
        for row in feature_rows
    ], dtype=float)
    return matrix


def predict_binary_scores(model_bundle: Dict[str, object], feature_matrix: np.ndarray) -> np.ndarray:
    if feature_matrix.size == 0:
        return np.zeros(0, dtype=float)

    model = model_bundle["model"]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_matrix)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return np.asarray(probabilities[:, 1], dtype=float)
        return np.asarray(probabilities, dtype=float).reshape(-1)

    decision_scores = np.asarray(model.decision_function(feature_matrix), dtype=float).reshape(-1)
    return 1.0 / (1.0 + np.exp(-decision_scores))


def load_learned_model_bundle(model_path: str) -> Dict[str, object]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid learned setwise model bundle at {model_path}")

    feature_names = bundle.get("feature_names")
    if feature_names and list(feature_names) != LEARNED_SETWISE_FEATURE_NAMES:
        raise ValueError(
            "Learned setwise model feature mismatch: "
            f"expected {LEARNED_SETWISE_FEATURE_NAMES}, got {feature_names}"
        )
    bundle.setdefault("feature_names", list(LEARNED_SETWISE_FEATURE_NAMES))
    bundle.setdefault("model_path", model_path)
    return bundle


def prune_requirement_pareto_frontier(states: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    survivors: List[Dict[str, object]] = []
    pruned_count = 0
    for state in states:
        state_metrics = state.get("state_metrics", {}) or {}
        dominated = False
        retained: List[Dict[str, object]] = []
        for survivor in survivors:
            survivor_metrics = survivor.get("state_metrics", {}) or {}
            if is_pareto_dominated(state_metrics, survivor_metrics):
                dominated = True
                pruned_count += 1
                retained.append(survivor)
                continue
            if is_pareto_dominated(survivor_metrics, state_metrics):
                pruned_count += 1
                continue
            retained.append(survivor)
        if dominated:
            survivors = retained
            continue
        retained.append(state)
        survivors = retained
    return survivors, int(pruned_count)


def get_requirement_state_signature(state: Dict[str, object]) -> Tuple[int, ...]:
    signature = state.get("signature")
    if signature is not None:
        return tuple(int(pos) for pos in signature)
    return canonicalize_requirement_positions(state.get("selected_positions", []))


def build_requirement_pre_prune_sort_key(state: Dict[str, object],
                                         normalized_mode: str) -> Tuple[object, ...]:
    signature = get_requirement_state_signature(state)
    state_metrics = state.get("state_metrics", {}) or {}
    return (
        -float(state_metrics.get("support_completeness", 0.0)),
        float(state_metrics.get("counterfactual_leakage", 0.0)),
        float(state_metrics.get("utopia_distance", 0.0)),
        -float(state.get("proposal_bonus", 0.0)) if normalized_mode == "learned" else 0.0,
        -float(state.get("cumulative_score", 0.0)),
        signature,
    )


def build_requirement_final_sort_key(state: Dict[str, object],
                                     normalized_mode: str) -> Tuple[object, ...]:
    signature = get_requirement_state_signature(state)
    state_metrics = state.get("state_metrics", {}) or {}
    return (
        float(state_metrics.get("utopia_distance", 0.0)),
        -float(state_metrics.get("support_completeness", 0.0)),
        float(state_metrics.get("counterfactual_leakage", 0.0)),
        -float(state.get("proposal_bonus", 0.0)) if normalized_mode == "learned" else 0.0,
        -float(state.get("cumulative_score", 0.0)),
        signature,
    )


def dedupe_requirement_states_by_signature(states: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    deduped_states: List[Dict[str, object]] = []
    seen_signatures: Set[Tuple[int, ...]] = set()
    pruned_count = 0
    for state in states:
        signature = get_requirement_state_signature(state)
        if signature in seen_signatures:
            pruned_count += 1
            continue
        state["signature"] = signature
        deduped_states.append(state)
        seen_signatures.add(signature)
    return deduped_states, int(pruned_count)


def select_requirement_beam_positions(pool_doc_ids: Sequence[int | None],
                                      pool_doc_scores: np.ndarray,
                                      pool_doc_titles: Sequence[str] | None,
                                      doc_idx_to_entities: Dict[int, Set[str]],
                                      doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                      adjacency: Dict[str, List[Tuple[str, float, str]]],
                                      qa_top_k: int,
                                      cache_entry: Dict[str, object],
                                      initial_seed_entities: Sequence[str] | Set[str] | None = None,
                                      proposal_query_entities: Sequence[str] | Set[str] | None = None,
                                      anchor_count: int = 2,
                                      reserve_top_m: int = 0,
                                      max_bridge_slots: int = 0,
                                      structure_max_hops: int = 2,
                                      structure_seed_target_bridge_mode: str = "off",
                                      base_weight: float = 0.25,
                                      structure_weight: float = 0.60,
                                      novelty_weight: float = 0.15,
                                      beam_width: int = 4,
                                      beam_expand_per_state: int = 4,
                                      beam_projected_shortlist_factor: int = DEFAULT_REQUIREMENT_BEAM_PROJECTED_SHORTLIST_FACTOR,
                                      beam_candidate_shortlist_limit: int | None = None,
                                      force_source_titles: Sequence[str] | None = None,
                                      force_shortlist_titles: Sequence[str] | None = None,
                                      trace_watch_titles: Sequence[str] | None = None,
                                      source_sort_mode: str = DEFAULT_REQUIREMENT_SOURCE_SORT_MODE,
                                      source_support_gain_weight: float = 0.0,
                                      shortlist_sort_mode: str = DEFAULT_REQUIREMENT_SHORTLIST_SORT_MODE,
                                      bridge_bonus_mode: str = "off",
                                      bridge_bonus_weight: float = 0.0,
                                      non_anchor_title_dedup: bool = False,
                                      requirement_mode: str = "oracle",
                                      requirement_model_bundle: Dict[str, object] | None = None,
                                      requirement_smooth_tau: float = DEFAULT_REQUIREMENT_SMOOTH_TAU,
                                      requirement_counterfactual_tau: float = DEFAULT_REQUIREMENT_CF_TAU) -> Tuple[List[int], Dict[str, object]]:
    def build_candidate_trace_preview(
        candidates: Sequence[Dict[str, object]],
        candidate_rows_by_position: Dict[int, Dict[str, float | int]],
        predicted_scores_by_position: Dict[int, float],
        candidate_requirement_diagnostics: Dict[int, Dict[str, object]],
    ) -> List[Dict[str, object]]:
        preview_rows: List[Dict[str, object]] = []
        for preview_rank, candidate in enumerate(candidates, start=1):
            pool_position = int(candidate["pool_position"])
            candidate_row = candidate_rows_by_position.get(pool_position, {})
            candidate_diag = candidate_requirement_diagnostics.get(pool_position, {})
            preview_rows.append({
                "preview_rank": int(preview_rank),
                "pool_position": int(pool_position),
                "doc_id": int(candidate["doc_id"]) if candidate["doc_id"] is not None else None,
                "title": str(candidate.get("doc_title", "")).strip(),
                "base_score": round(float(candidate.get("base_score", 0.0) or 0.0), 4),
                "combined_score": round(float(candidate.get("combined_score", 0.0) or 0.0), 4),
                "source_sort_score": round(float(candidate.get("source_sort_score_raw", candidate.get("combined_score_raw", 0.0)) or 0.0), 4),
                "structure_score": round(float(candidate.get("structure_score", 0.0) or 0.0), 4),
                "novelty_score": round(float(candidate.get("novelty_score", 0.0) or 0.0), 4),
                "support_completeness_gain": round(float(candidate_row.get("support_completeness_gain", 0.0) or 0.0), 4),
                "counterfactual_leakage_gain": round(float(candidate_row.get("counterfactual_leakage_gain", 0.0) or 0.0), 4),
                "utility_margin_gain": round(float(candidate_row.get("utility_margin_gain", 0.0) or 0.0), 4),
                "support_completeness_after": round(float(candidate_row.get("support_completeness_after", 0.0) or 0.0), 4),
                "counterfactual_leakage_after": round(float(candidate_row.get("counterfactual_leakage_after", 0.0) or 0.0), 4),
                "utility_margin_after": round(float(candidate_row.get("utility_margin_after", 0.0) or 0.0), 4),
                "bridge_bonus_applied": bool(candidate_row.get("bridge_bonus_applied", 0.0) or 0.0),
                "bridge_bonus_mode": str(candidate_row.get("bridge_bonus_mode", "") or ""),
                "bridge_bonus_score": round(float(candidate_row.get("bridge_bonus_score", 0.0) or 0.0), 4),
                "bridge_bonus_base_score": round(float(candidate_row.get("bridge_bonus_base_score", 0.0) or 0.0), 4),
                "bridge_bonus_alignment_score": round(float(candidate_row.get("bridge_bonus_alignment_score", 0.0) or 0.0), 4),
                "bridge_bonus_predecessor_coverage": round(float(candidate_row.get("bridge_bonus_predecessor_coverage", 0.0) or 0.0), 4),
                "bridge_bonus_overlap_entity_count": int(candidate_row.get("bridge_bonus_overlap_entity_count", 0) or 0),
                "bridge_bonus_novel_entity_count": int(candidate_row.get("bridge_bonus_novel_entity_count", 0) or 0),
                "bridge_bonus_unit_id": str(candidate_row.get("bridge_bonus_unit_id", "") or ""),
                "bridge_bonus_unit_type": str(candidate_row.get("bridge_bonus_unit_type", "") or ""),
                "bridge_bonus_unit_predicate": str(candidate_row.get("bridge_bonus_unit_predicate", "") or ""),
                "bridge_bonus_predecessor_id": str(candidate_row.get("bridge_bonus_predecessor_id", "") or ""),
                "doc_positive_mean": round(float(candidate_row.get("doc_positive_mean", 0.0) or 0.0), 4),
                "doc_counterfactual_mean": round(float(candidate_row.get("doc_counterfactual_mean", 0.0) or 0.0), 4),
                "doc_contradiction_mean": round(float(candidate_row.get("doc_contradiction_mean", 0.0) or 0.0), 4),
                "predicted_utility": round(float(predicted_scores_by_position.get(pool_position, 0.0) or 0.0), 4),
                "dominant_positive_unit_id": candidate_diag.get("dominant_positive_unit_id"),
                "dominant_positive_unit_type": candidate_diag.get("dominant_positive_unit_type"),
                "dominant_positive_unit_predicate": candidate_diag.get("dominant_positive_unit_predicate"),
                "dominant_positive_unit_score": candidate_diag.get("dominant_positive_unit_score"),
                "dominant_positive_unit_coverage_before": candidate_diag.get("dominant_positive_unit_coverage_before"),
                "dominant_positive_unit_coverage_after": candidate_diag.get("dominant_positive_unit_coverage_after"),
                "dominant_positive_unit_gain": candidate_diag.get("dominant_positive_unit_gain"),
                "dominant_positive_unit_precovered": candidate_diag.get("dominant_positive_unit_precovered"),
                "dominant_counterfactual_id": candidate_diag.get("dominant_counterfactual_id"),
                "dominant_counterfactual_coverage_before": candidate_diag.get("dominant_counterfactual_coverage_before"),
                "dominant_counterfactual_coverage_after": candidate_diag.get("dominant_counterfactual_coverage_after"),
                "dominant_counterfactual_gain": candidate_diag.get("dominant_counterfactual_gain"),
            })
        return preview_rows

    def inject_forced_title_candidates(
        candidate_list: Sequence[Dict[str, object]],
        fallback_candidates: Sequence[Dict[str, object]],
        forced_title_keys: Set[str],
    ) -> Tuple[List[Dict[str, object]], List[str]]:
        if not forced_title_keys:
            return list(candidate_list), []
        merged = list(candidate_list)
        existing_positions = {
            int(candidate["pool_position"])
            for candidate in merged
        }
        added_titles: List[str] = []
        for candidate in fallback_candidates:
            title = str(candidate.get("doc_title", "")).strip()
            title_key = normalize_structure_text(title)
            pool_position = int(candidate["pool_position"])
            if title_key not in forced_title_keys or pool_position in existing_positions:
                continue
            merged.append(candidate)
            existing_positions.add(pool_position)
            if title:
                added_titles.append(title)
        return merged, unique_ordered_titles(added_titles)

    def build_watch_title_trace(
        scored_candidates: Sequence[Dict[str, object]],
        candidate_rows_by_position: Dict[int, Dict[str, float | int]],
        candidate_requirement_diagnostics: Dict[int, Dict[str, object]],
        candidate_source: Sequence[Dict[str, object]],
        candidate_shortlist: Sequence[Dict[str, object]],
        predicted_scores_by_position: Dict[int, float],
        forced_source_title_keys: Set[str],
        forced_shortlist_title_keys: Set[str],
    ) -> List[Dict[str, object]]:
        watch_title_keys = normalize_title_set(trace_watch_titles)
        if not watch_title_keys:
            return []
        source_rank_by_position = {
            int(candidate["pool_position"]): rank
            for rank, candidate in enumerate(candidate_source, start=1)
        }
        shortlist_rank_by_position = {
            int(candidate["pool_position"]): rank
            for rank, candidate in enumerate(candidate_shortlist, start=1)
        }
        watch_rows: List[Dict[str, object]] = []
        for scored_rank, candidate in enumerate(scored_candidates, start=1):
            title = str(candidate.get("doc_title", "")).strip()
            title_key = normalize_structure_text(title)
            if title_key not in watch_title_keys:
                continue
            pool_position = int(candidate["pool_position"])
            feature_row = candidate_rows_by_position.get(pool_position, {})
            candidate_diag = candidate_requirement_diagnostics.get(pool_position, {})
            watch_rows.append({
                "title": title,
                "pool_position": int(pool_position),
                "scored_rank": int(scored_rank),
                "source_rank": int(source_rank_by_position.get(pool_position, 0) or 0),
                "shortlist_rank": int(shortlist_rank_by_position.get(pool_position, 0) or 0),
                "base_score": round(float(candidate.get("base_score", 0.0) or 0.0), 4),
                "combined_score": round(float(candidate.get("combined_score", 0.0) or 0.0), 4),
                "source_sort_score": round(float(candidate.get("source_sort_score_raw", candidate.get("combined_score_raw", 0.0)) or 0.0), 4),
                "structure_score": round(float(candidate.get("structure_score", 0.0) or 0.0), 4),
                "novelty_score": round(float(candidate.get("novelty_score", 0.0) or 0.0), 4),
                "predicted_utility": round(float(predicted_scores_by_position.get(pool_position, 0.0) or 0.0), 4),
                "utility_margin_gain": round(float(feature_row.get("utility_margin_gain", 0.0) or 0.0), 4),
                "support_completeness_gain": round(float(feature_row.get("support_completeness_gain", 0.0) or 0.0), 4),
                "counterfactual_leakage_gain": round(float(feature_row.get("counterfactual_leakage_gain", 0.0) or 0.0), 4),
                "support_completeness_after": round(float(feature_row.get("support_completeness_after", 0.0) or 0.0), 4),
                "counterfactual_leakage_after": round(float(feature_row.get("counterfactual_leakage_after", 0.0) or 0.0), 4),
                "utility_margin_after": round(float(feature_row.get("utility_margin_after", 0.0) or 0.0), 4),
                "bridge_bonus_applied": bool(feature_row.get("bridge_bonus_applied", 0.0) or 0.0),
                "bridge_bonus_mode": str(feature_row.get("bridge_bonus_mode", "") or ""),
                "bridge_bonus_score": round(float(feature_row.get("bridge_bonus_score", 0.0) or 0.0), 4),
                "bridge_bonus_base_score": round(float(feature_row.get("bridge_bonus_base_score", 0.0) or 0.0), 4),
                "bridge_bonus_alignment_score": round(float(feature_row.get("bridge_bonus_alignment_score", 0.0) or 0.0), 4),
                "bridge_bonus_predecessor_coverage": round(float(feature_row.get("bridge_bonus_predecessor_coverage", 0.0) or 0.0), 4),
                "bridge_bonus_overlap_entity_count": int(feature_row.get("bridge_bonus_overlap_entity_count", 0) or 0),
                "bridge_bonus_novel_entity_count": int(feature_row.get("bridge_bonus_novel_entity_count", 0) or 0),
                "bridge_bonus_unit_id": str(feature_row.get("bridge_bonus_unit_id", "") or ""),
                "bridge_bonus_unit_type": str(feature_row.get("bridge_bonus_unit_type", "") or ""),
                "bridge_bonus_unit_predicate": str(feature_row.get("bridge_bonus_unit_predicate", "") or ""),
                "bridge_bonus_predecessor_id": str(feature_row.get("bridge_bonus_predecessor_id", "") or ""),
                "dominant_positive_unit_id": candidate_diag.get("dominant_positive_unit_id"),
                "dominant_positive_unit_type": candidate_diag.get("dominant_positive_unit_type"),
                "dominant_positive_unit_predicate": candidate_diag.get("dominant_positive_unit_predicate"),
                "dominant_positive_unit_score": candidate_diag.get("dominant_positive_unit_score"),
                "dominant_positive_unit_coverage_before": candidate_diag.get("dominant_positive_unit_coverage_before"),
                "dominant_positive_unit_coverage_after": candidate_diag.get("dominant_positive_unit_coverage_after"),
                "dominant_positive_unit_gain": candidate_diag.get("dominant_positive_unit_gain"),
                "dominant_positive_unit_precovered": candidate_diag.get("dominant_positive_unit_precovered"),
                "dominant_counterfactual_id": candidate_diag.get("dominant_counterfactual_id"),
                "dominant_counterfactual_coverage_before": candidate_diag.get("dominant_counterfactual_coverage_before"),
                "dominant_counterfactual_coverage_after": candidate_diag.get("dominant_counterfactual_coverage_after"),
                "dominant_counterfactual_gain": candidate_diag.get("dominant_counterfactual_gain"),
                "forced_into_source": bool(
                    title_key in forced_source_title_keys and pool_position in source_rank_by_position
                ),
                "forced_into_shortlist": bool(
                    title_key in forced_shortlist_title_keys and pool_position in shortlist_rank_by_position
                ),
            })
        return watch_rows

    effective_candidate_count = len(pool_doc_ids)
    normalized_mode = str(requirement_mode or "oracle").strip().lower()
    if normalized_mode not in {"oracle", "learned"}:
        raise ValueError(f"Unsupported requirement beam mode: {requirement_mode}")
    if normalized_mode == "learned" and requirement_model_bundle is None:
        raise ValueError("learned requirement_beam requires a loaded matcher bundle")
    normalized_source_sort_mode = normalize_requirement_source_sort_mode(source_sort_mode)
    effective_source_support_gain_weight = max(float(source_support_gain_weight), 0.0)
    normalized_shortlist_sort_mode = normalize_requirement_shortlist_sort_mode(shortlist_sort_mode)
    normalized_bridge_bonus_mode = normalize_requirement_bridge_bonus_mode(bridge_bonus_mode)
    normalized_structure_seed_target_bridge_mode = normalize_structure_seed_target_bridge_mode(
        structure_seed_target_bridge_mode
    )
    effective_bridge_bonus_weight = (
        max(float(bridge_bonus_weight), 0.0)
        if normalized_bridge_bonus_mode != "off" else 0.0
    )

    if effective_candidate_count <= 0 or qa_top_k <= 0:
            return [], {
            "selection_steps": [],
            "anchor_positions": [],
            "reserved_positions": [],
            "candidate_count": int(effective_candidate_count),
            "selection_target_k": 0,
            "requirement_mode": normalized_mode,
            "annotation_pool_k": int(cache_entry.get("annotation_pool_k", 0) or 0),
            "beam_width": int(max(beam_width, 1)),
            "beam_expand_per_state": int(max(beam_expand_per_state, 1)),
            "beam_projected_shortlist_factor": int(max(beam_projected_shortlist_factor, 1)),
                "beam_candidate_shortlist_limit": int(max(beam_candidate_shortlist_limit or 0, 0)),
                "source_sort_mode": normalized_source_sort_mode,
                "source_support_gain_weight": round(float(effective_source_support_gain_weight), 4),
                "structure_seed_target_bridge_mode": normalized_structure_seed_target_bridge_mode,
                "beam_avg_frontier_size": 0.0,
                "bridge_bonus_mode": normalized_bridge_bonus_mode,
                "bridge_bonus_weight": round(float(effective_bridge_bonus_weight), 4),
            "beam_pareto_pruned_count": 0,
            "beam_learned_eval_count": 0,
            "beam_best_support_completeness": 0.0,
            "beam_best_counterfactual_leakage": 0.0,
            "beam_best_utility_margin": 0.0,
            "beam_best_utopia_distance": 0.0,
            "beam_finalists": [],
        }

    normalized_base_scores = min_max_normalize_array(np.asarray(pool_doc_scores[:effective_candidate_count], dtype=float))
    target_k = min(effective_candidate_count, qa_top_k)
    proposal_query_entities = normalize_entity_set(proposal_query_entities) or normalize_entity_set(initial_seed_entities)
    seed_entities = normalize_entity_set(initial_seed_entities)
    anchor_positions, reserved_positions = resolve_reserved_positions(
        candidate_count=effective_candidate_count,
        target_k=target_k,
        anchor_count=anchor_count,
        reserve_top_m=reserve_top_m,
        pool_doc_titles=pool_doc_titles[:effective_candidate_count] if pool_doc_titles is not None else None,
        dedup_titles=bool(non_anchor_title_dedup),
    )
    selection_target_k = resolve_selection_target_k(
        target_k=target_k,
        reserved_count=len(reserved_positions),
        max_bridge_slots=max_bridge_slots,
    )
    anchor_position_set = set(anchor_positions)

    initial_covered = set(seed_entities)
    blocked_titles: Set[str] = set()
    initial_steps: List[Dict[str, object]] = []
    for reserved_pos in reserved_positions:
        doc_id = pool_doc_ids[reserved_pos]
        if doc_id is not None:
            initial_covered.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))
        if pool_doc_titles is not None and reserved_pos < len(pool_doc_titles):
            reserved_title = str(pool_doc_titles[reserved_pos]).strip()
            if reserved_title:
                blocked_titles.add(reserved_title)
        initial_steps.append({
            "step": len(initial_steps) + 1,
            "mode": "anchor" if reserved_pos in anchor_position_set else "reserve",
            "pool_position": int(reserved_pos),
            "doc_id": int(doc_id) if doc_id is not None else None,
            "title": str(pool_doc_titles[reserved_pos]).strip() if pool_doc_titles is not None and reserved_pos < len(pool_doc_titles) else "",
        })

    beam_width = max(int(beam_width), 1)
    beam_expand_per_state = max(int(beam_expand_per_state), 1)
    beam_projected_shortlist_factor = max(int(beam_projected_shortlist_factor), 1)
    effective_candidate_shortlist_limit = (
        max(int(beam_candidate_shortlist_limit), 1)
        if beam_candidate_shortlist_limit is not None else None
    )
    normalized_force_source_titles = normalize_title_set(force_source_titles)
    normalized_force_shortlist_titles = normalize_title_set(force_shortlist_titles)
    normalized_trace_watch_titles = normalize_title_set(trace_watch_titles)
    annotations_by_position = {
        int(annotation["pool_position"]): annotation
        for annotation in cache_entry.get("doc_annotations", [])
    }
    active_positive_units = [
        requirement
        for requirement in get_positive_units(cache_entry)
        if (not is_need_unit_cache_version(cache_entry)) or bool(requirement.get("selector_enabled", True))
    ]
    active_counterfactual_ids = [
        str(cf_set.get("cf_id", ""))
        for cf_set in cache_entry.get("counterfactual_sets", [])
        if str(cf_set.get("cf_id", "")).strip()
    ]

    def build_source_candidate_sort_key(
        candidate: Dict[str, object],
        candidate_rows_by_position: Dict[int, Dict[str, float | int]],
    ) -> Tuple[float, float, float, int]:
        pool_position = int(candidate["pool_position"])
        support_gain = float(
            candidate_rows_by_position.get(pool_position, {}).get("support_completeness_gain", 0.0) or 0.0
        )
        combined_score_raw = float(candidate.get("combined_score_raw", 0.0) or 0.0)
        if normalized_source_sort_mode == "support_bonus":
            source_sort_score = combined_score_raw + effective_source_support_gain_weight * support_gain
        else:
            source_sort_score = combined_score_raw
        candidate["source_sort_score_raw"] = float(source_sort_score)
        return (
            -float(source_sort_score),
            -float(support_gain),
            -float(combined_score_raw),
            int(pool_position),
        )

    def _coverage_from_scores(scores: Sequence[float]) -> float:
        residual = 1.0
        for score in scores:
            residual *= max(0.0, 1.0 - float(score))
        return float(1.0 - residual)

    def build_candidate_requirement_diagnostic(
        pool_position: int,
        selected_positions_before: Sequence[int],
        current_positive_score_overrides_by_position: Dict[int, Dict[str, float]] | None = None,
        candidate_positive_score_override_map: Dict[str, float] | None = None,
    ) -> Dict[str, object]:
        annotation = annotations_by_position.get(int(pool_position), {})
        current_positions = list(canonicalize_requirement_positions(selected_positions_before))
        next_positive_score_overrides_by_position = merge_positive_score_overrides_by_position(
            positive_score_overrides_by_position=current_positive_score_overrides_by_position,
            pool_position=int(pool_position),
            override_map=candidate_positive_score_override_map,
        )
        candidate_positive_scores = get_positive_score_map_with_overrides(
            annotation,
            int(pool_position),
            positive_score_overrides_by_position=next_positive_score_overrides_by_position,
        )
        candidate_counterfactual_scores = get_counterfactual_set_score_map(annotation)

        best_positive: Dict[str, object] | None = None
        for requirement in active_positive_units:
            requirement_id = str(requirement.get("unit_id", requirement.get("requirement_id", "")))
            if not requirement_id:
                continue
            before_scores = [
                float(
                    get_positive_score_map_with_overrides(
                        annotations_by_position[pos],
                        pos,
                        positive_score_overrides_by_position=current_positive_score_overrides_by_position,
                    ).get(requirement_id, 0.0)
                )
                for pos in current_positions
                if pos in annotations_by_position
            ]
            before_coverage = _coverage_from_scores(before_scores)
            candidate_score = float(candidate_positive_scores.get(requirement_id, 0.0) or 0.0)
            after_coverage = _coverage_from_scores(before_scores + [candidate_score])
            gain = float(after_coverage - before_coverage)
            row = {
                "dominant_positive_unit_id": requirement_id,
                "dominant_positive_unit_type": str(requirement.get("unit_type", requirement.get("type", "")) or ""),
                "dominant_positive_unit_predicate": str(requirement.get("predicate", "") or ""),
                "dominant_positive_unit_score": round(float(candidate_score), 4),
                "dominant_positive_unit_coverage_before": round(float(before_coverage), 4),
                "dominant_positive_unit_coverage_after": round(float(after_coverage), 4),
                "dominant_positive_unit_gain": round(float(gain), 4),
                "dominant_positive_unit_precovered": bool(before_coverage > 1e-6),
            }
            if best_positive is None or (
                float(gain),
                float(candidate_score),
                float(before_coverage),
                requirement_id,
            ) > (
                float(best_positive["dominant_positive_unit_gain"]),
                float(best_positive["dominant_positive_unit_score"]),
                float(best_positive["dominant_positive_unit_coverage_before"]),
                str(best_positive["dominant_positive_unit_id"]),
            ):
                best_positive = row

        best_counterfactual: Dict[str, object] | None = None
        for cf_id in active_counterfactual_ids:
            before_scores = [
                float(get_counterfactual_set_score_map(annotations_by_position[pos]).get(cf_id, 0.0))
                for pos in current_positions
                if pos in annotations_by_position
            ]
            before_coverage = _coverage_from_scores(before_scores)
            candidate_score = float(candidate_counterfactual_scores.get(cf_id, 0.0) or 0.0)
            after_coverage = _coverage_from_scores(before_scores + [candidate_score])
            gain = float(after_coverage - before_coverage)
            row = {
                "dominant_counterfactual_id": cf_id,
                "dominant_counterfactual_coverage_before": round(float(before_coverage), 4),
                "dominant_counterfactual_coverage_after": round(float(after_coverage), 4),
                "dominant_counterfactual_gain": round(float(gain), 4),
            }
            if best_counterfactual is None or (
                float(gain),
                float(candidate_score),
                cf_id,
            ) > (
                float(best_counterfactual["dominant_counterfactual_gain"]),
                float(best_counterfactual["dominant_counterfactual_coverage_after"] or 0.0),
                str(best_counterfactual["dominant_counterfactual_id"]),
            ):
                best_counterfactual = row

        combined = {}
        if best_positive is not None:
            combined.update(best_positive)
        if best_counterfactual is not None:
            combined.update(best_counterfactual)
        return combined
    initial_state_metrics = compute_requirement_state_metrics(
        cache_entry=cache_entry,
        selected_positions=reserved_positions,
        positive_score_overrides_by_position={},
        smooth_tau=requirement_smooth_tau,
        counterfactual_tau=requirement_counterfactual_tau,
    )
    initial_signature = canonicalize_requirement_positions(reserved_positions)
    beam_states: List[Dict[str, object]] = [{
        "selected_positions": list(reserved_positions),
        "signature": initial_signature,
        "covered_entities": initial_covered,
        "blocked_titles": set(blocked_titles),
        "selection_steps": initial_steps,
        "cumulative_score": 0.0,
        "state_metrics": initial_state_metrics,
        "proposal_bonus": 0.0,
        "positive_score_overrides_by_position": {},
    }]
    seen_signatures = {initial_signature}
    frontier_sizes: List[int] = []
    pareto_pruned_count = 0
    signature_pruned_count = 0
    learned_eval_count = 0

    while beam_states and len(beam_states[0]["selected_positions"]) < selection_target_k:
        expanded_states: List[Dict[str, object]] = []
        for state in beam_states:
            selected_positions = list(state["selected_positions"])
            remaining_positions = [
                pos for pos in range(effective_candidate_count)
                if pos not in set(selected_positions)
            ]
            if not remaining_positions:
                expanded_states.append(state)
                continue

            scored_candidates = score_bridge_candidates(
                pool_doc_ids=pool_doc_ids[:effective_candidate_count],
                normalized_base_scores=normalized_base_scores,
                pool_doc_titles=pool_doc_titles[:effective_candidate_count] if pool_doc_titles is not None else None,
                doc_idx_to_entities=doc_idx_to_entities,
                doc_idx_to_edges=doc_idx_to_edges,
                adjacency=adjacency,
                remaining_positions=remaining_positions,
                covered_entities=state["covered_entities"],
                query_entities=proposal_query_entities or seed_entities,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=normalized_structure_seed_target_bridge_mode,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                score_mode="bridge",
            )
            scored_candidates = filter_title_dedup_candidates(
                scored_candidates=scored_candidates,
                blocked_titles=set(state["blocked_titles"]),
                enabled=non_anchor_title_dedup,
            )
            proposal_source_limit = beam_expand_per_state * beam_projected_shortlist_factor
            if normalized_mode == "learned":
                proposal_source_limit = max(proposal_source_limit, beam_expand_per_state * 2)
            source_feature_positions = (
                [int(candidate["pool_position"]) for candidate in scored_candidates]
                if normalized_source_sort_mode == "support_bonus"
                else [int(candidate["pool_position"]) for candidate in scored_candidates[:proposal_source_limit]]
            )
            candidate_feature_rows = compute_requirement_candidate_feature_rows(
                cache_entry=cache_entry,
                selected_positions=selected_positions,
                candidate_positions=source_feature_positions,
                normalized_base_scores=normalized_base_scores,
                qa_top_k=qa_top_k,
                positive_score_overrides_by_position=state.get("positive_score_overrides_by_position"),
                bridge_bonus_binding_entities=state["covered_entities"],
                bridge_bonus_mode=normalized_bridge_bonus_mode,
                bridge_bonus_weight=effective_bridge_bonus_weight,
                smooth_tau=requirement_smooth_tau,
                counterfactual_tau=requirement_counterfactual_tau,
            )
            candidate_rows_by_position = {
                int(row["pool_position"]): row
                for row in candidate_feature_rows
            }
            candidate_requirement_diagnostics = {
                int(pos): build_candidate_requirement_diagnostic(
                    pool_position=int(pos),
                    selected_positions_before=selected_positions,
                    current_positive_score_overrides_by_position=state.get("positive_score_overrides_by_position"),
                    candidate_positive_score_override_map=dict(
                        candidate_rows_by_position.get(int(pos), {}).get("positive_score_override_map", {}) or {}
                    ),
                )
                for pos in source_feature_positions
            }
            source_sorted_candidates = sorted(
                scored_candidates,
                key=lambda candidate: build_source_candidate_sort_key(
                    candidate,
                    candidate_rows_by_position=candidate_rows_by_position,
                ),
            )
            candidate_source = source_sorted_candidates[:proposal_source_limit]
            candidate_source, forced_source_titles_applied = inject_forced_title_candidates(
                candidate_list=candidate_source,
                fallback_candidates=source_sorted_candidates,
                forced_title_keys=normalized_force_source_titles,
            )
            candidate_source_raw = list(candidate_source)
            candidate_positions = [int(candidate["pool_position"]) for candidate in candidate_source]
            missing_feature_positions = [
                int(pos) for pos in candidate_positions
                if int(pos) not in candidate_rows_by_position
            ]
            if missing_feature_positions:
                extra_feature_rows = compute_requirement_candidate_feature_rows(
                    cache_entry=cache_entry,
                    selected_positions=selected_positions,
                    candidate_positions=missing_feature_positions,
                    normalized_base_scores=normalized_base_scores,
                    qa_top_k=qa_top_k,
                    positive_score_overrides_by_position=state.get("positive_score_overrides_by_position"),
                    bridge_bonus_binding_entities=state["covered_entities"],
                    bridge_bonus_mode=normalized_bridge_bonus_mode,
                    bridge_bonus_weight=effective_bridge_bonus_weight,
                    smooth_tau=requirement_smooth_tau,
                    counterfactual_tau=requirement_counterfactual_tau,
                )
                candidate_feature_rows.extend(extra_feature_rows)
                for row in extra_feature_rows:
                    candidate_rows_by_position[int(row["pool_position"])] = row
                for pos in missing_feature_positions:
                    candidate_requirement_diagnostics[int(pos)] = build_candidate_requirement_diagnostic(
                        pool_position=int(pos),
                        selected_positions_before=selected_positions,
                        current_positive_score_overrides_by_position=state.get("positive_score_overrides_by_position"),
                        candidate_positive_score_override_map=dict(
                            candidate_rows_by_position.get(int(pos), {}).get("positive_score_override_map", {}) or {}
                        ),
                    )
            predicted_scores_by_position: Dict[int, float] = {}
            if normalized_mode == "learned" and candidate_feature_rows:
                feature_matrix = requirement_feature_rows_to_matrix(
                    candidate_feature_rows,
                    feature_names=requirement_model_bundle.get("feature_names") if requirement_model_bundle is not None else None,
                )
                predicted_scores = predict_binary_scores(requirement_model_bundle, feature_matrix)
                learned_eval_count += len(candidate_feature_rows)
                for row, predicted_score in zip(candidate_feature_rows, predicted_scores):
                    predicted_scores_by_position[int(row["pool_position"])] = float(predicted_score)

            if normalized_mode == "learned" and candidate_source:
                candidate_source = sorted(
                    candidate_source,
                    key=lambda candidate: (
                        -float(predicted_scores_by_position.get(int(candidate["pool_position"]), 0.0)),
                        -float(candidate_rows_by_position.get(int(candidate["pool_position"]), {}).get("utility_margin_gain", 0.0) or 0.0),
                        -float(candidate_rows_by_position.get(int(candidate["pool_position"]), {}).get("support_completeness_gain", 0.0) or 0.0),
                        -float(candidate.get("combined_score", 0.0)),
                        int(candidate["pool_position"]),
                    ),
                )
                candidate_shortlist = candidate_source[:beam_expand_per_state]
            elif effective_candidate_shortlist_limit is not None and candidate_source:
                if normalized_shortlist_sort_mode == "positive_only":
                    shortlist_key = lambda candidate: (
                        -float(candidate_rows_by_position.get(int(candidate["pool_position"]), {}).get("support_completeness_gain", 0.0) or 0.0),
                        -float(candidate_rows_by_position.get(int(candidate["pool_position"]), {}).get("utility_margin_gain", 0.0) or 0.0),
                        float(candidate_rows_by_position.get(int(candidate["pool_position"]), {}).get("counterfactual_leakage_gain", 0.0) or 0.0),
                        -float(candidate.get("combined_score", 0.0) or 0.0),
                        int(candidate["pool_position"]),
                    )
                else:
                    shortlist_key = lambda candidate: (
                        -float(candidate_rows_by_position.get(int(candidate["pool_position"]), {}).get("utility_margin_gain", 0.0) or 0.0),
                        -float(candidate_rows_by_position.get(int(candidate["pool_position"]), {}).get("support_completeness_gain", 0.0) or 0.0),
                        float(candidate_rows_by_position.get(int(candidate["pool_position"]), {}).get("counterfactual_leakage_gain", 0.0) or 0.0),
                        -float(candidate.get("combined_score", 0.0) or 0.0),
                        int(candidate["pool_position"]),
                    )
                candidate_shortlist = sorted(candidate_source, key=shortlist_key)[:effective_candidate_shortlist_limit]
            else:
                candidate_shortlist = candidate_source
            candidate_shortlist, forced_shortlist_titles_applied = inject_forced_title_candidates(
                candidate_list=candidate_shortlist,
                fallback_candidates=candidate_source,
                forced_title_keys=normalized_force_shortlist_titles,
            )
            candidate_source_preview = build_candidate_trace_preview(
                candidates=candidate_source_raw,
                candidate_rows_by_position=candidate_rows_by_position,
                predicted_scores_by_position=predicted_scores_by_position,
                candidate_requirement_diagnostics=candidate_requirement_diagnostics,
            )
            candidate_shortlist_preview = build_candidate_trace_preview(
                candidates=candidate_shortlist,
                candidate_rows_by_position=candidate_rows_by_position,
                predicted_scores_by_position=predicted_scores_by_position,
                candidate_requirement_diagnostics=candidate_requirement_diagnostics,
            )
            watch_title_trace = build_watch_title_trace(
                scored_candidates=scored_candidates,
                candidate_rows_by_position=candidate_rows_by_position,
                candidate_requirement_diagnostics=candidate_requirement_diagnostics,
                candidate_source=candidate_source,
                candidate_shortlist=candidate_shortlist,
                predicted_scores_by_position=predicted_scores_by_position,
                forced_source_title_keys=normalized_force_source_titles,
                forced_shortlist_title_keys=normalized_force_shortlist_titles,
            ) if normalized_trace_watch_titles else []

            for proposal_rank, candidate in enumerate(candidate_shortlist, start=1):
                chosen_pos = int(candidate["pool_position"])
                signature = canonicalize_requirement_positions(list(selected_positions) + [chosen_pos])
                if signature in seen_signatures:
                    continue

                next_selected_positions = list(selected_positions) + [chosen_pos]
                next_covered = set(state["covered_entities"])
                next_covered.update(set(candidate["doc_entities"]))
                next_blocked_titles = set(state["blocked_titles"])
                chosen_title = str(candidate.get("doc_title", "")).strip()
                if chosen_title:
                    next_blocked_titles.add(chosen_title)
                candidate_row = candidate_rows_by_position.get(chosen_pos, {})
                next_positive_score_overrides_by_position = merge_positive_score_overrides_by_position(
                    positive_score_overrides_by_position=state.get("positive_score_overrides_by_position"),
                    pool_position=int(chosen_pos),
                    override_map=dict(candidate_row.get("positive_score_override_map", {}) or {}),
                )
                next_state_metrics = compute_requirement_state_metrics(
                    cache_entry=cache_entry,
                    selected_positions=signature,
                    positive_score_overrides_by_position=next_positive_score_overrides_by_position,
                    smooth_tau=requirement_smooth_tau,
                    counterfactual_tau=requirement_counterfactual_tau,
                )
                predicted_score = float(predicted_scores_by_position.get(chosen_pos, 0.0))
                expanded_states.append({
                    "selected_positions": next_selected_positions,
                    "signature": signature,
                    "covered_entities": next_covered,
                    "blocked_titles": next_blocked_titles,
                    "selection_steps": list(state["selection_steps"]) + [{
                        "step": len(state["selection_steps"]) + 1,
                        "mode": "requirement_beam",
                        "pool_position": int(chosen_pos),
                        "doc_id": int(candidate["doc_id"]) if candidate["doc_id"] is not None else None,
                        "title": str(candidate.get("doc_title", "")).strip(),
                        "proposal_rank": int(proposal_rank),
                        "combined_score": float(candidate.get("combined_score", 0.0)),
                        "support_completeness": float(next_state_metrics["support_completeness"]),
                        "counterfactual_leakage": float(next_state_metrics["counterfactual_leakage"]),
                        "utility_margin": float(next_state_metrics["utility_margin"]),
                        "utopia_distance": float(next_state_metrics["utopia_distance"]),
                        "predicted_utility": float(predicted_score),
                        "candidate_source_preview": [dict(row) for row in candidate_source_preview],
                        "candidate_shortlist_preview": [dict(row) for row in candidate_shortlist_preview],
                        "watch_title_trace": [dict(row) for row in watch_title_trace],
                        "source_sort_mode": normalized_source_sort_mode,
                        "source_support_gain_weight": float(effective_source_support_gain_weight),
                        "shortlist_sort_mode": normalized_shortlist_sort_mode,
                        "structure_seed_target_bridge_mode": normalized_structure_seed_target_bridge_mode,
                        "bridge_bonus_mode": normalized_bridge_bonus_mode,
                        "bridge_bonus_weight": float(effective_bridge_bonus_weight),
                        "forced_source_titles_applied": list(forced_source_titles_applied),
                        "forced_shortlist_titles_applied": list(forced_shortlist_titles_applied),
                    }],
                    "cumulative_score": float(state["cumulative_score"]) + float(candidate.get("combined_score_raw", 0.0)),
                    "state_metrics": next_state_metrics,
                    "proposal_bonus": float(state["proposal_bonus"]) + float(predicted_score),
                    "proposal_rank": int(proposal_rank),
                    "candidate_features": candidate_row,
                    "positive_score_overrides_by_position": next_positive_score_overrides_by_position,
                })

        if not expanded_states:
            break

        expanded_states.sort(key=lambda state: build_requirement_pre_prune_sort_key(state, normalized_mode))
        expanded_states, step_signature_pruned_count = dedupe_requirement_states_by_signature(expanded_states)
        signature_pruned_count += int(step_signature_pruned_count)
        expanded_states, step_pruned_count = prune_requirement_pareto_frontier(expanded_states)
        pareto_pruned_count += int(step_pruned_count)
        expanded_states.sort(key=lambda state: build_requirement_final_sort_key(state, normalized_mode))
        beam_states = expanded_states[:beam_width]
        frontier_sizes.append(len(expanded_states))
        for state in beam_states:
            seen_signatures.add(get_requirement_state_signature(state))

    best_state = min(
        beam_states,
        key=lambda state: build_requirement_final_sort_key(state, normalized_mode),
    ) if beam_states else {
        "selected_positions": list(reserved_positions),
        "signature": initial_signature,
        "covered_entities": initial_covered,
        "blocked_titles": set(blocked_titles),
        "selection_steps": initial_steps,
        "cumulative_score": 0.0,
        "state_metrics": initial_state_metrics,
        "proposal_bonus": 0.0,
        "positive_score_overrides_by_position": {},
    }
    ranked_finalists = sorted(
        beam_states or [best_state],
        key=lambda state: build_requirement_final_sort_key(state, normalized_mode),
    )
    best_metrics = best_state["state_metrics"]
    beam_finalists: List[Dict[str, object]] = []
    for finalist in ranked_finalists:
        finalist_positions = [int(pos) for pos in finalist["selected_positions"]]
        finalist_metrics = finalist["state_metrics"]
        beam_finalists.append({
            "source": "beam_finalist",
            "selected_positions": finalist_positions,
            "selected_titles": [
                str(pool_doc_titles[pos]).strip()
                for pos in finalist_positions
                if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
            ],
            "support_completeness": round(float(finalist_metrics["support_completeness"]), 4),
            "counterfactual_leakage": round(float(finalist_metrics["counterfactual_leakage"]), 4),
            "utility_margin": round(float(finalist_metrics["utility_margin"]), 4),
            "utopia_distance": round(float(finalist_metrics["utopia_distance"]), 4),
            "proposal_bonus": round(float(finalist.get("proposal_bonus", 0.0) or 0.0), 4),
            "cumulative_score": round(float(finalist.get("cumulative_score", 0.0) or 0.0), 4),
        })

    beam_watch_title_finalists: List[Dict[str, object]] = []
    best_state_title_keys = normalize_title_set(
        [
            str(pool_doc_titles[pos]).strip()
            for pos in best_state["selected_positions"]
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ]
    )
    if normalized_trace_watch_titles:
        for watch_title in unique_ordered_titles(trace_watch_titles):
            title_key = normalize_structure_text(watch_title)
            if not title_key:
                continue
            containing_rows: List[Dict[str, object]] = []
            for finalist_rank, finalist in enumerate(ranked_finalists, start=1):
                finalist_positions = [int(pos) for pos in finalist["selected_positions"]]
                finalist_titles = [
                    str(pool_doc_titles[pos]).strip()
                    for pos in finalist_positions
                    if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
                ]
                if title_key not in normalize_title_set(finalist_titles):
                    continue
                finalist_metrics = finalist["state_metrics"]
                containing_rows.append({
                    "finalist_rank": int(finalist_rank),
                    "selected_positions": list(finalist_positions),
                    "selected_titles": list(finalist_titles),
                    "support_completeness": round(float(finalist_metrics["support_completeness"]), 4),
                    "counterfactual_leakage": round(float(finalist_metrics["counterfactual_leakage"]), 4),
                    "utility_margin": round(float(finalist_metrics["utility_margin"]), 4),
                    "utopia_distance": round(float(finalist_metrics["utopia_distance"]), 4),
                    "cumulative_score": round(float(finalist.get("cumulative_score", 0.0) or 0.0), 4),
                })
            best_containing = containing_rows[0] if containing_rows else None
            beam_watch_title_finalists.append({
                "title": watch_title,
                "appears_in_best_state": bool(title_key in best_state_title_keys),
                "appears_in_any_finalist": bool(best_containing is not None),
                "best_containing_finalist_rank": int(best_containing["finalist_rank"]) if best_containing is not None else None,
                "best_containing_selected_positions": list(best_containing["selected_positions"]) if best_containing is not None else [],
                "best_containing_selected_titles": list(best_containing["selected_titles"]) if best_containing is not None else [],
                "best_containing_support_completeness": best_containing["support_completeness"] if best_containing is not None else None,
                "best_containing_counterfactual_leakage": best_containing["counterfactual_leakage"] if best_containing is not None else None,
                "best_containing_utility_margin": best_containing["utility_margin"] if best_containing is not None else None,
                "best_containing_utopia_distance": best_containing["utopia_distance"] if best_containing is not None else None,
                "delta_vs_best_state_support_completeness": round(
                    float(best_containing["support_completeness"]) - float(best_metrics["support_completeness"]),
                    4,
                ) if best_containing is not None else None,
                "delta_vs_best_state_counterfactual_leakage": round(
                    float(best_containing["counterfactual_leakage"]) - float(best_metrics["counterfactual_leakage"]),
                    4,
                ) if best_containing is not None else None,
                "delta_vs_best_state_utility_margin": round(
                    float(best_containing["utility_margin"]) - float(best_metrics["utility_margin"]),
                    4,
                ) if best_containing is not None else None,
                "delta_vs_best_state_utopia_distance": round(
                    float(best_containing["utopia_distance"]) - float(best_metrics["utopia_distance"]),
                    4,
                ) if best_containing is not None else None,
            })

    return list(best_state["selected_positions"]), {
        "selection_steps": list(best_state["selection_steps"]),
        "anchor_positions": [int(pos) for pos in anchor_positions],
        "reserved_positions": [int(pos) for pos in reserved_positions],
        "candidate_count": int(effective_candidate_count),
        "selection_target_k": int(selection_target_k),
        "requirement_mode": normalized_mode,
        "annotation_pool_k": int(cache_entry.get("annotation_pool_k", 0) or 0),
        "beam_width": int(beam_width),
        "beam_expand_per_state": int(beam_expand_per_state),
        "beam_projected_shortlist_factor": int(beam_projected_shortlist_factor),
        "beam_candidate_shortlist_limit": int(effective_candidate_shortlist_limit or 0),
        "source_sort_mode": normalized_source_sort_mode,
        "source_support_gain_weight": round(float(effective_source_support_gain_weight), 4),
        "shortlist_sort_mode": normalized_shortlist_sort_mode,
        "structure_seed_target_bridge_mode": normalized_structure_seed_target_bridge_mode,
        "bridge_bonus_mode": normalized_bridge_bonus_mode,
        "bridge_bonus_weight": round(float(effective_bridge_bonus_weight), 4),
        "forced_source_titles": list(unique_ordered_titles(force_source_titles)),
        "forced_shortlist_titles": list(unique_ordered_titles(force_shortlist_titles)),
        "trace_watch_titles": list(unique_ordered_titles(trace_watch_titles)),
        "beam_avg_frontier_size": round(float(np.mean(frontier_sizes)) if frontier_sizes else 0.0, 4),
        "beam_pareto_pruned_count": int(pareto_pruned_count),
        "beam_signature_pruned_count": int(signature_pruned_count),
        "beam_learned_eval_count": int(learned_eval_count),
        "beam_best_support_completeness": round(float(best_metrics["support_completeness"]), 4),
        "beam_best_counterfactual_leakage": round(float(best_metrics["counterfactual_leakage"]), 4),
        "beam_best_utility_margin": round(float(best_metrics["utility_margin"]), 4),
        "beam_best_utopia_distance": round(float(best_metrics["utopia_distance"]), 4),
        "requirement_positive_count": int(len(cache_entry.get("positive_need_units", cache_entry.get("positive_requirements", [])))),
        "requirement_counterfactual_count": int(len(cache_entry.get("counterfactual_sets", []))),
        "beam_finalists": beam_finalists,
        "beam_watch_title_finalists": beam_watch_title_finalists,
    }


def select_learned_greedy_positions(query: str,
                                    pool_docs: Sequence[str],
                                    pool_doc_ids: Sequence[int | None],
                                    pool_doc_scores: np.ndarray,
                                    doc_idx_to_entities: Dict[int, Set[str]],
                                    doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                    adjacency: Dict[str, List[Tuple[str, float, str]]],
                                    qa_top_k: int,
                                    learned_model_bundle: Dict[str, object],
                                    initial_seed_entities: Sequence[str] | Set[str] | None = None,
                                    anchor_count: int = 0,
                                    structure_max_hops: int = 2,
                                    structure_seed_target_bridge_mode: str = "off") -> Tuple[List[int], Dict[str, object]]:
    candidate_count = len(pool_doc_ids)
    if candidate_count == 0 or qa_top_k <= 0:
        return [], {
            "selection_steps": [],
            "anchor_positions": [],
            "seed_entity_count_initial": 0,
            "covered_entity_count_final": 0,
            "candidate_count": candidate_count,
        }

    target_k = min(candidate_count, qa_top_k)
    selected_positions: List[int] = []
    remaining_positions = list(range(candidate_count))
    seed_entities = normalize_entity_set(initial_seed_entities)
    selection_steps: List[Dict[str, object]] = []

    actual_anchor_count = min(max(anchor_count, 0), target_k)
    anchor_positions = remaining_positions[:actual_anchor_count]
    for anchor_pos in anchor_positions:
        selected_positions.append(anchor_pos)
        remaining_positions.remove(anchor_pos)
        selection_steps.append({
            "step": len(selection_steps) + 1,
            "mode": "anchor",
            "pool_position": int(anchor_pos),
            "doc_id": int(pool_doc_ids[anchor_pos]) if pool_doc_ids[anchor_pos] is not None else None,
            "predicted_score": 1.0,
        })

    while len(selected_positions) < target_k and remaining_positions:
        feature_rows = compute_candidate_feature_rows(
            query=query,
            pool_docs=pool_docs,
            pool_doc_ids=pool_doc_ids,
            pool_doc_scores=pool_doc_scores,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            adjacency=adjacency,
            qa_top_k=qa_top_k,
            selected_positions=selected_positions,
            seed_entities=seed_entities,
            structure_max_hops=structure_max_hops,
            structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
            candidate_positions=remaining_positions,
        )
        feature_matrix = feature_rows_to_matrix(feature_rows)
        predicted_scores = predict_binary_scores(learned_model_bundle, feature_matrix)

        best_idx = 0
        best_score = float("-inf")
        for idx, (row, predicted_score) in enumerate(zip(feature_rows, predicted_scores)):
            pool_position = int(row["pool_position"])
            if predicted_score > best_score + 1e-9 or (
                abs(predicted_score - best_score) <= 1e-9 and pool_position < int(feature_rows[best_idx]["pool_position"])
            ):
                best_idx = idx
                best_score = float(predicted_score)

        chosen_row = feature_rows[best_idx]
        chosen_pos = int(chosen_row["pool_position"])
        selected_positions.append(chosen_pos)
        remaining_positions.remove(chosen_pos)
        selection_steps.append({
            "step": len(selection_steps) + 1,
            "mode": "learned_greedy",
            "pool_position": chosen_pos,
            "doc_id": int(chosen_row["doc_id"]) if chosen_row["doc_id"] is not None else None,
            "predicted_score": round(float(best_score), 4),
            "base_score": round(float(chosen_row["base_score"]), 4),
            "structure_score_seed": round(float(chosen_row["structure_score_seed"]), 4),
            "structure_score_covered": round(float(chosen_row["structure_score_covered"]), 4),
            "novelty_score": round(float(chosen_row["novelty_score"]), 4),
        })

    covered_entities = seed_entities.copy()
    for pos in selected_positions:
        doc_id = pool_doc_ids[pos]
        if doc_id is None:
            continue
        covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))

    return selected_positions, {
        "selection_steps": selection_steps,
        "anchor_positions": [int(pos) for pos in anchor_positions],
        "seed_entity_count_initial": len(seed_entities),
        "covered_entity_count_final": len(covered_entities),
        "candidate_count": candidate_count,
    }


def score_bridge_candidates(pool_doc_ids: Sequence[int | None],
                            normalized_base_scores: np.ndarray,
                            pool_doc_titles: Sequence[str] | None,
                            doc_idx_to_entities: Dict[int, Set[str]],
                            doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                            adjacency: Dict[str, List[Tuple[str, float, str]]],
                            remaining_positions: Sequence[int],
                            covered_entities: Sequence[str] | Set[str] | None,
                            structure_max_hops: int,
                            structure_seed_target_bridge_mode: str,
                            base_weight: float,
                            structure_weight: float,
                            novelty_weight: float,
                            query_entities: Sequence[str] | Set[str] | None = None,
                            score_mode: str = "bridge") -> List[Dict[str, object]]:
    normalized_covered = normalize_entity_set(covered_entities)
    normalized_query = normalize_entity_set(query_entities) or set(normalized_covered)
    normalized_score_mode = normalize_setwise_score_mode(score_mode)
    candidate_doc_ids = [
        int(pool_doc_ids[pos])
        for pos in remaining_positions
        if pos < len(pool_doc_ids) and pool_doc_ids[pos] is not None
    ]
    structure_scores = (
        score_candidate_docs_by_structure(
            candidate_doc_ids=candidate_doc_ids,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            seed_entities=normalized_covered,
            adjacency=adjacency,
            max_hops=structure_max_hops,
            seed_target_bridge_mode=structure_seed_target_bridge_mode,
        )
        if candidate_doc_ids and normalized_covered
        else {}
    )
    structure_scores_seed = (
        score_candidate_docs_by_structure(
            candidate_doc_ids=candidate_doc_ids,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            seed_entities=normalized_query,
            adjacency=adjacency,
            max_hops=structure_max_hops,
            seed_target_bridge_mode=structure_seed_target_bridge_mode,
        )
        if candidate_doc_ids and normalized_query
        else {}
    )
    frontier_scores = (
        expand_directed_entities(
            normalized_covered,
            adjacency,
            max_hops=structure_max_hops,
        )
        if normalized_covered
        else {}
    )

    scored_candidates: List[Dict[str, object]] = []
    for pos in remaining_positions:
        doc_id = pool_doc_ids[pos]
        doc_entities = (
            normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
            if doc_id is not None
            else set()
        )
        novelty_score = (
            len(doc_entities - normalized_covered) / max(1, len(doc_entities))
            if doc_entities
            else 0.0
        )
        structure_score = (
            float(structure_scores.get(int(doc_id), 0.0))
            if doc_id is not None
            else 0.0
        )
        structure_seed_score = (
            float(structure_scores_seed.get(int(doc_id), 0.0))
            if doc_id is not None
            else 0.0
        )
        base_score = float(normalized_base_scores[pos]) if pos < len(normalized_base_scores) else 0.0
        doc_title = ""
        if pool_doc_titles is not None and pos < len(pool_doc_titles):
            doc_title = str(pool_doc_titles[pos]).strip()
        doc_edges = doc_idx_to_edges.get(int(doc_id), []) if doc_id is not None else []
        new_entities = doc_entities - normalized_covered
        frontier_gain_score = min(
            1.0,
            float(sum(frontier_scores.get(entity, 0.0) for entity in new_entities)),
        )
        query_overlap_gain = (
            len(new_entities & normalized_query) / max(1.0, min(float(len(normalized_query)), 4.0))
            if normalized_query
            else 0.0
        )
        query_anchor_score = min(
            1.0,
            0.5 * float(query_overlap_gain) + 0.5 * float(structure_seed_score),
        )
        path_coherence_score = 0.0
        for source, target, edge_weight, _ in doc_edges:
            positive_weight = max(float(edge_weight), 0.0)
            if positive_weight <= 0.0:
                continue
            normalized_source = normalize_structure_text(source)
            normalized_target = normalize_structure_text(target)
            if not normalized_source or not normalized_target or normalized_source == normalized_target:
                continue
            support_candidates: List[float] = []
            if normalized_source in normalized_covered and normalized_target in new_entities:
                support_candidates.append(
                    max(float(frontier_scores.get(normalized_target, 0.0)),
                        1.0 if normalized_target in normalized_query else 0.0)
                )
            if normalized_target in normalized_covered and normalized_source in new_entities:
                support_candidates.append(
                    max(float(frontier_scores.get(normalized_source, 0.0)),
                        1.0 if normalized_source in normalized_query else 0.0)
                )
            if normalized_source in normalized_query and normalized_target in new_entities:
                support_candidates.append(max(1.0, float(frontier_scores.get(normalized_target, 0.0))))
            if normalized_target in normalized_query and normalized_source in new_entities:
                support_candidates.append(max(1.0, float(frontier_scores.get(normalized_source, 0.0))))
            if support_candidates:
                path_coherence_score = max(
                    path_coherence_score,
                    min(1.0, positive_weight * max(support_candidates)),
                )
        redundancy_penalty = (
            len(doc_entities & normalized_covered) / max(1, len(doc_entities))
            if doc_entities
            else 0.0
        )
        closure_score = float(np.clip(
            0.35 * float(structure_score)
            + 0.25 * float(frontier_gain_score)
            + 0.20 * float(query_anchor_score)
            + 0.20 * float(path_coherence_score)
            - 0.20 * float(redundancy_penalty),
            0.0,
            1.0,
        ))
        selection_score = (
            0.60 * float(structure_score) + 0.40 * float(closure_score)
            if normalized_score_mode in {"closure_proxy", "set_closure"}
            else float(structure_score)
        )
        combined_score = (
            base_weight * base_score
            + structure_weight * selection_score
            + novelty_weight * novelty_score
        )
        scored_candidates.append({
            "pool_position": int(pos),
            "doc_id": int(doc_id) if doc_id is not None else None,
            "doc_title": doc_title,
            "doc_entities": doc_entities,
            "new_entity_count": len(doc_entities - normalized_covered),
            "base_score": round(base_score, 4),
            "structure_score": round(structure_score, 4),
            "structure_score_seed": round(float(structure_seed_score), 4),
            "novelty_score": round(float(novelty_score), 4),
            "frontier_gain_score": round(float(frontier_gain_score), 4),
            "query_anchor_score": round(float(query_anchor_score), 4),
            "path_coherence_score": round(float(path_coherence_score), 4),
            "redundancy_penalty": round(float(redundancy_penalty), 4),
            "closure_score": round(float(closure_score), 4),
            "closure_score_raw": float(closure_score),
            "selection_score": round(float(selection_score), 4),
            "score_mode": normalized_score_mode,
            "combined_score": round(float(combined_score), 4),
            "selection_score_raw": float(selection_score),
            "combined_score_raw": float(combined_score),
        })

    scored_candidates.sort(
        key=lambda item: (
            -float(item["combined_score_raw"]),
            -float(item["selection_score_raw"]),
            -int(item["new_entity_count"]),
            int(item["pool_position"]),
        )
    )
    return scored_candidates


def _entity_token_set(entities: Sequence[str] | Set[str] | None) -> Set[str]:
    tokens: Set[str] = set()
    for entity in normalize_entity_set(entities):
        tokens.update(normalized_token_set(entity))
    return tokens


def _top_normalized_entities(entities: Sequence[str] | Set[str] | None,
                             limit: int = 2) -> List[str]:
    normalized = sorted(normalize_entity_set(entities), key=lambda item: (-len(item), item))
    return [str(item) for item in normalized[:max(int(limit), 0)]]


def _lexical_overlap_score(lhs_tokens: Sequence[str] | Set[str] | None,
                           rhs_tokens: Sequence[str] | Set[str] | None) -> float:
    lhs = set(lhs_tokens or [])
    rhs = set(rhs_tokens or [])
    if not lhs or not rhs:
        return 0.0
    return float(len(lhs & rhs)) / float(max(1, min(len(lhs), len(rhs))))


_GAP_ROLE_RELATION_RULES: List[Dict[str, object]] = [
    {
        "slot": "director",
        "patterns": (" director ", " directors ", " directed by ", "director of"),
        "slot_cues": {"director", "directed"},
        "query_template": "Find the director of {anchors}. Question: {query}",
    },
    {
        "slot": "actor",
        "patterns": (" actor ", " actors ", " actress ", " starring ", " starred ", " star of "),
        "slot_cues": {"actor", "actors", "actress", "starring", "starred", "star"},
        "query_template": "Find the starring actor or actress of {anchors}. Question: {query}",
    },
    {
        "slot": "author",
        "patterns": (" author ", " wrote ", " writer ", " written by "),
        "slot_cues": {"author", "wrote", "writer", "written"},
        "query_template": "Find the author or writer of {anchors}. Question: {query}",
    },
]

_GAP_TARGET_ATTRIBUTE_RULES: List[Dict[str, object]] = [
    {
        "slot": "education",
        "patterns": (" study", " studied ", " school ", " university ", " college ", " educated "),
        "slot_cues": {"study", "studied", "school", "university", "college", "educated"},
        "query_template": "Find where {anchor} studied. Question: {query}",
    },
    {
        "slot": "birthplace",
        "patterns": (" birthplace ", " born ", " where was ", " where were "),
        "slot_cues": {"born", "birthplace", "where"},
        "query_template": "Find the birthplace of {anchor}. Question: {query}",
    },
    {
        "slot": "death_date",
        "patterns": (" when did ", " died ", " die ", " death "),
        "slot_cues": {"when", "die", "died", "death"},
        "query_template": "Find when {anchor} died. Question: {query}",
    },
    {
        "slot": "nationality",
        "patterns": (" nationality", " same nationality", " what country", " which country"),
        "slot_cues": {"nationality", "country"},
        "query_template": "Find the nationality or country of {anchor}. Question: {query}",
    },
    {
        "slot": "birth_date",
        "patterns": (" when was ", " born "),
        "slot_cues": {"when", "born"},
        "query_template": "Find when {anchor} was born. Question: {query}",
    },
]

_GAP_BRIDGE_ENTITY_RULES: List[Dict[str, object]] = [
    {
        "slot": "spouse",
        "patterns": (" spouse ", " wife ", " husband "),
        "slot_cues": {"spouse", "wife", "husband"},
        "query_template": "Find who is the spouse of {anchor}. Question: {query}",
    },
    {
        "slot": "mother",
        "patterns": (" mother ",),
        "slot_cues": {"mother"},
        "query_template": "Find who is the mother of {anchor}. Question: {query}",
    },
    {
        "slot": "father",
        "patterns": (" father ",),
        "slot_cues": {"father"},
        "query_template": "Find who is the father of {anchor}. Question: {query}",
    },
]


def _find_gap_rule(normalized_query: str,
                   rules: Sequence[Mapping[str, object]]) -> Dict[str, object] | None:
    padded_query = f" {normalized_query} "
    for rule in rules:
        patterns = tuple(str(pattern) for pattern in (rule.get("patterns") or ()))
        if any(str(pattern) in padded_query for pattern in patterns):
            return dict(rule)
    return None


def _resolve_gap_title_aligned_entities(query: str,
                                        query_entities: Sequence[str] | Set[str] | None,
                                        baseline_titles: Sequence[str] | None,
                                        limit: int = 4) -> List[str]:
    ordered_entities = _order_query_entities(query, query_entities, max_entities=max(int(limit), 1) * 2)
    normalized_baseline_titles = [
        normalize_structure_text(title)
        for title in unique_ordered_titles(baseline_titles)
        if normalize_structure_text(title)
    ]
    aligned_entities: List[str] = []
    for entity in ordered_entities:
        entity_key = normalize_structure_text(entity)
        if not entity_key:
            continue
        if any(entity_key == title_key or entity_key in title_key or title_key in entity_key for title_key in normalized_baseline_titles):
            aligned_entities.append(entity)
        if len(aligned_entities) >= max(int(limit), 1):
            break
    return aligned_entities


def _resolve_gap_anchor_entities(query: str,
                                 gap_type: str,
                                 slot: str | None,
                                 baseline_titles: Sequence[str] | None,
                                 query_entities: Sequence[str] | Set[str] | None,
                                 covered_query_entities: Sequence[str] | Set[str] | None,
                                 uncovered_query_entities: Sequence[str] | Set[str] | None) -> List[str]:
    title_aligned_entities = _resolve_gap_title_aligned_entities(
        query=query,
        query_entities=query_entities,
        baseline_titles=baseline_titles,
        limit=4,
    )
    covered_ordered = [
        entity for entity in _order_query_entities(query, covered_query_entities, max_entities=4)
        if normalize_structure_text(entity)
    ]
    uncovered_ordered = [
        entity for entity in _order_query_entities(query, uncovered_query_entities, max_entities=4)
        if normalize_structure_text(entity)
    ]

    if gap_type == "bridge_entity":
        if covered_ordered:
            return covered_ordered[:1]
        if title_aligned_entities:
            return title_aligned_entities[:1]
        return []

    if gap_type == "role_relation":
        if title_aligned_entities:
            return title_aligned_entities[:2]
        if covered_ordered:
            return covered_ordered[:2]
        return []

    if gap_type == "target_attribute":
        possessive_slots = {"spouse", "mother", "father"}
        if slot in possessive_slots and len(title_aligned_entities) >= 2:
            return [title_aligned_entities[-1]]
        if uncovered_ordered:
            return uncovered_ordered[:1]
        if len(title_aligned_entities) == 1:
            return [title_aligned_entities[0]]
        if len(covered_ordered) == 1:
            return [covered_ordered[0]]
        if title_aligned_entities:
            return [title_aligned_entities[-1]]
        return []

    return []


def _build_gap_micro_queries(query: str,
                             gap_type: str,
                             slot_rule: Mapping[str, object] | None,
                             anchors: Sequence[str],
                             bridge_targets: Sequence[str],
                             max_queries: int) -> List[str]:
    query_variants: List[str] = [str(query).strip()]
    slot_template = str((slot_rule or {}).get("query_template", "") or "").strip()
    if gap_type == "bridge_entity" and anchors and slot_template:
        query_variants.append(slot_template.format(anchor=anchors[0], anchors=", ".join(anchors), query=query))
        if bridge_targets:
            query_variants.append(
                f"Connect {anchors[0]} to {bridge_targets[0]}. Question: {query}"
            )
    elif gap_type == "role_relation" and anchors and slot_template:
        query_variants.append(slot_template.format(anchor=anchors[0], anchors=" and ".join(anchors), query=query))
    elif gap_type == "target_attribute" and anchors and slot_template:
        query_variants.append(slot_template.format(anchor=anchors[0], anchors=", ".join(anchors), query=query))

    deduped_queries: List[str] = []
    seen_queries: Set[str] = set()
    for raw_query in query_variants:
        cleaned = str(raw_query or "").strip()
        if not cleaned:
            continue
        normalized = normalize_structure_text(cleaned)
        if not normalized or normalized in seen_queries:
            continue
        deduped_queries.append(cleaned)
        seen_queries.add(normalized)
        if len(deduped_queries) >= max(int(max_queries), 1):
            break
    return deduped_queries


def detect_gap_expand_state(query: str,
                            baseline_prefix_positions: Sequence[int],
                            pool_docs: Sequence[str] | None,
                            pool_doc_ids: Sequence[int | None],
                            doc_idx_to_entities: Mapping[int, Set[str]],
                            query_entities: Sequence[str] | Set[str] | None,
                            covered_entities: Sequence[str] | Set[str] | None,
                            gap_expand_mode: str = "heuristic",
                            gap_expand_max_queries: int | None = None) -> Dict[str, object]:
    normalized_mode = normalize_gap_expand_mode(gap_expand_mode)
    effective_gap_expand_max_queries = max(int(gap_expand_max_queries or DEFAULT_GAP_EXPAND_MAX_QUERIES), 1)
    normalized_query_entities = normalize_entity_set(query_entities)
    normalized_covered_entities = normalize_entity_set(covered_entities)
    covered_query_entities = sorted(normalized_query_entities & normalized_covered_entities)
    uncovered_query_entities = sorted(normalized_query_entities - normalized_covered_entities)
    baseline_titles = [
        extract_doc_title(pool_docs[int(pos)])
        for pos in (baseline_prefix_positions or [])
        if pool_docs is not None and 0 <= int(pos) < len(pool_docs)
    ]
    baseline_doc_entities: Set[str] = set()
    for pos in baseline_prefix_positions or []:
        if int(pos) < 0 or int(pos) >= len(pool_doc_ids):
            continue
        doc_id = pool_doc_ids[int(pos)]
        if doc_id is None:
            continue
        baseline_doc_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))

    query_tokens = normalized_token_set(query)
    entity_tokens = _entity_token_set(normalized_query_entities)
    relation_terms = sorted(
        token for token in query_tokens
        if token not in entity_tokens and token not in _ANSWER_SCENT_ENTITY_STOPWORDS and len(token) > 2
    )
    fallback_used = False
    gap_type = "flat_fallback"
    gap_slot = ""
    gap_rule: Dict[str, object] | None = None
    abstain_reason = ""
    bridge_targets: List[str] = []

    if normalized_mode == "flat_fallback_only":
        gap_type = "abstain"
        fallback_used = True
        abstain_reason = "flat_fallback_only"
    elif normalized_mode in {"typed_abstain", "unit_typed_abstain"}:
        normalized_query = normalize_structure_text(query)
        bridge_rule = _find_gap_rule(normalized_query, _GAP_BRIDGE_ENTITY_RULES)
        role_rule = _find_gap_rule(normalized_query, _GAP_ROLE_RELATION_RULES)
        attribute_rule = _find_gap_rule(normalized_query, _GAP_TARGET_ATTRIBUTE_RULES)
        if normalized_mode == "typed_abstain" and bridge_rule is not None and uncovered_query_entities:
            gap_type = "bridge_entity"
            gap_rule = bridge_rule
            gap_slot = str(bridge_rule.get("slot", "") or "")
            bridge_targets = _top_normalized_entities(uncovered_query_entities, limit=1)
        elif role_rule is not None:
            gap_type = "role_relation"
            gap_rule = role_rule
            gap_slot = str(role_rule.get("slot", "") or "")
        elif attribute_rule is not None:
            gap_type = "target_attribute"
            gap_rule = attribute_rule
            gap_slot = str(attribute_rule.get("slot", "") or "")
        else:
            gap_type = "abstain"
            fallback_used = True
            abstain_reason = "no_typed_slot"
    elif uncovered_query_entities:
        gap_type = "bridge_entity"
    elif len(covered_query_entities) >= 2 and relation_terms:
        gap_type = "linking_relation"
    elif covered_query_entities or baseline_doc_entities:
        gap_type = "target_attribute"
    else:
        gap_type = "flat_fallback"
        fallback_used = True

    anchor_entities: List[str]
    if normalized_mode in {"typed_abstain", "unit_typed_abstain"}:
        anchor_entities = _resolve_gap_anchor_entities(
            query=query,
            gap_type=gap_type,
            slot=gap_slot,
            baseline_titles=baseline_titles,
            query_entities=query_entities,
            covered_query_entities=covered_query_entities,
            uncovered_query_entities=uncovered_query_entities,
        )
        if gap_type != "abstain" and not anchor_entities:
            gap_type = "abstain"
            fallback_used = True
            abstain_reason = "missing_anchor"
        elif normalized_mode == "unit_typed_abstain" and gap_type == "target_attribute":
            stable_anchor_candidates = normalize_entity_set(
                _resolve_gap_title_aligned_entities(
                    query=query,
                    query_entities=query_entities,
                    baseline_titles=baseline_titles,
                    limit=4,
                )
            ) | normalize_entity_set(covered_query_entities)
            if not normalize_entity_set(anchor_entities) or not normalize_entity_set(anchor_entities).issubset(stable_anchor_candidates):
                gap_type = "abstain"
                fallback_used = True
                abstain_reason = "unstable_attribute_anchor"
    elif gap_type == "bridge_entity":
        anchor_entities = _top_normalized_entities(covered_query_entities or baseline_doc_entities, limit=2)
    elif gap_type == "linking_relation":
        anchor_entities = _top_normalized_entities(covered_query_entities or baseline_doc_entities, limit=2)
    elif gap_type == "target_attribute":
        anchor_entities = _top_normalized_entities(covered_query_entities or baseline_doc_entities, limit=1)
    else:
        anchor_entities = _top_normalized_entities(covered_query_entities or normalized_query_entities, limit=2)

    if normalized_mode in {"typed_abstain", "unit_typed_abstain"}:
        deduped_queries = _build_gap_micro_queries(
            query=query,
            gap_type=gap_type,
            slot_rule=gap_rule,
            anchors=anchor_entities,
            bridge_targets=bridge_targets,
            max_queries=effective_gap_expand_max_queries,
        )
    else:
        query_variants: List[str] = [str(query).strip()]
        if gap_type == "bridge_entity":
            missing_text = ", ".join(_top_normalized_entities(uncovered_query_entities, limit=2))
            anchor_text = ", ".join(anchor_entities)
            if anchor_text and missing_text:
                query_variants.append(
                    f"Find the intermediate entity linking {anchor_text} to {missing_text}. Question: {query}"
                )
            elif missing_text:
                query_variants.append(f"Identify the missing bridge entity for: {query}")
        elif gap_type == "target_attribute":
            anchor_text = ", ".join(anchor_entities)
            if anchor_text:
                query_variants.append(
                    f"Find the target attribute or answer-bearing relation for {anchor_text}. Question: {query}"
                )
            else:
                query_variants.append(f"Find the target attribute needed to answer: {query}")
        elif gap_type == "linking_relation":
            if len(anchor_entities) >= 2:
                query_variants.append(
                    f"Find the linking relation between {anchor_entities[0]} and {anchor_entities[1]}. Question: {query}"
                )
            elif anchor_entities:
                query_variants.append(f"Find the linking relation involving {anchor_entities[0]}. Question: {query}")
        deduped_queries = _build_gap_micro_queries(
            query=query,
            gap_type=gap_type,
            slot_rule=None,
            anchors=anchor_entities,
            bridge_targets=[],
            max_queries=effective_gap_expand_max_queries,
        ) if False else []
        if not deduped_queries:
            deduped_queries = []
            seen_queries: Set[str] = set()
            for raw_query in query_variants:
                cleaned = str(raw_query or "").strip()
                if not cleaned:
                    continue
                normalized = normalize_structure_text(cleaned)
                if not normalized or normalized in seen_queries:
                    continue
                deduped_queries.append(cleaned)
                seen_queries.add(normalized)
                if len(deduped_queries) >= effective_gap_expand_max_queries:
                    break

    return {
        "gap_type": str(gap_type),
        "gap_mode": str(normalized_mode),
        "fallback_used": bool(fallback_used),
        "abstain_reason": str(abstain_reason),
        "gap_anchors": list(anchor_entities),
        "gap_slot": str(gap_slot),
        "gap_slot_cues": sorted(str(token) for token in (gap_rule or {}).get("slot_cues", set()) if str(token)),
        "gap_bridge_targets": list(bridge_targets),
        "covered_query_entities": list(covered_query_entities),
        "uncovered_query_entities": list(uncovered_query_entities),
        "covered_entities_snapshot": sorted(normalized_covered_entities)[:16],
        "baseline_titles": list(unique_ordered_titles(baseline_titles)),
        "baseline_entity_count": int(len(baseline_doc_entities)),
        "relation_terms": list(relation_terms[:8]),
        "micro_queries": list(deduped_queries),
        "micro_query_count": int(len(deduped_queries)),
    }


def _score_gap_witness_units(doc_text: str,
                             doc_entity_set: Sequence[str] | Set[str] | None,
                             *,
                             anchor_entity_set: Sequence[str] | Set[str] | None,
                             covered_entity_set: Sequence[str] | Set[str] | None,
                             slot_cue_set: Sequence[str] | Set[str] | None,
                             micro_query_tokens: Sequence[Sequence[str] | Set[str]],
                             bridge_target_set: Sequence[str] | Set[str] | None) -> Dict[str, object]:
    normalized_doc_entities = normalize_entity_set(doc_entity_set)
    normalized_anchor_entities = normalize_entity_set(anchor_entity_set)
    normalized_covered_entities = normalize_entity_set(covered_entity_set)
    normalized_bridge_targets = normalize_entity_set(bridge_target_set)
    anchor_token_set = _entity_token_set(normalized_anchor_entities)
    bridge_target_token_set = _entity_token_set(normalized_bridge_targets)
    non_anchor_entity_set = normalized_doc_entities - normalized_anchor_entities - normalized_covered_entities
    non_anchor_token_set = _entity_token_set(non_anchor_entity_set)

    best_witness_anchor = 0.0
    best_witness_slot = 0.0
    best_witness_query = 0.0
    best_witness_bridge_target = 0.0
    best_role_joint = 0.0
    best_role_anchor_slot = 0.0
    best_role_non_anchor = 0.0
    best_role_query = 0.0
    best_role_unit_type = ""

    for unit in build_witness_units(doc_text, max_sentences=6, max_windows=8, include_full_doc=True):
        unit_text = str(unit.get("unit_text", "") or "")
        unit_tokens = normalized_token_set(unit_text)
        if not unit_tokens:
            continue
        unit_anchor = _lexical_overlap_score(unit_tokens, anchor_token_set)
        unit_slot = _lexical_overlap_score(unit_tokens, slot_cue_set)
        unit_query = max(
            (_lexical_overlap_score(unit_tokens, query_tokens) for query_tokens in micro_query_tokens),
            default=0.0,
        )
        unit_bridge_target = _lexical_overlap_score(unit_tokens, bridge_target_token_set)
        best_witness_anchor = max(best_witness_anchor, unit_anchor)
        best_witness_slot = max(best_witness_slot, unit_slot)
        best_witness_query = max(best_witness_query, unit_query)
        best_witness_bridge_target = max(best_witness_bridge_target, unit_bridge_target)

        if str(unit.get("unit_type", "") or "") == "full_doc":
            continue

        unit_non_anchor = _lexical_overlap_score(unit_tokens, non_anchor_token_set)
        unit_anchor_slot = min(unit_anchor, unit_slot)
        unit_joint = min(unit_anchor, unit_slot, unit_non_anchor)
        current_best_tuple = (
            float(best_role_joint),
            float(best_role_anchor_slot),
            float(best_role_query),
            float(best_role_non_anchor),
        )
        candidate_tuple = (
            float(unit_joint),
            float(unit_anchor_slot),
            float(unit_query),
            float(unit_non_anchor),
        )
        if candidate_tuple > current_best_tuple:
            best_role_joint = unit_joint
            best_role_anchor_slot = unit_anchor_slot
            best_role_query = unit_query
            best_role_non_anchor = unit_non_anchor
            best_role_unit_type = str(unit.get("unit_type", "") or "")

    return {
        "gap_best_witness_anchor": float(best_witness_anchor),
        "gap_best_witness_slot": float(best_witness_slot),
        "gap_best_witness_query": float(best_witness_query),
        "gap_best_witness_bridge_target": float(best_witness_bridge_target),
        "gap_non_anchor_entity_gain": float(len(non_anchor_entity_set)),
        "gap_non_anchor_entities": sorted(non_anchor_entity_set)[:6],
        "gap_best_role_witness_joint": float(best_role_joint),
        "gap_best_role_anchor_slot": float(best_role_anchor_slot),
        "gap_best_role_non_anchor": float(best_role_non_anchor),
        "gap_best_role_query": float(best_role_query),
        "gap_best_role_unit_type": str(best_role_unit_type),
    }


def _role_relation_witness_tuple(payload: Mapping[str, object]) -> Tuple[float, float, float, float, float]:
    return (
        float(payload.get("gap_best_role_witness_joint", 0.0) or 0.0),
        float(payload.get("gap_best_role_anchor_slot", 0.0) or 0.0),
        float(payload.get("gap_best_role_non_anchor", 0.0) or 0.0),
        float(payload.get("gap_best_witness_anchor", 0.0) or 0.0),
        float(payload.get("gap_best_witness_slot", 0.0) or 0.0),
    )


def rerank_gap_expand_candidates(scored_candidates: Sequence[Mapping[str, object]],
                                 pool_docs: Sequence[str] | None,
                                 gap_state: Mapping[str, object],
                                 covered_entities: Sequence[str] | Set[str] | None) -> List[Dict[str, object]]:
    gap_type = str(gap_state.get("gap_type", "flat_fallback") or "flat_fallback")
    gap_mode = str(gap_state.get("gap_mode", "heuristic") or "heuristic")
    micro_query_tokens = [
        normalized_token_set(str(query))
        for query in (gap_state.get("micro_queries", []) or [])
        if str(query).strip()
    ]
    covered_entity_set = normalize_entity_set(covered_entities)
    anchor_entity_set = normalize_entity_set(gap_state.get("gap_anchors", []) or [])
    uncovered_query_entity_set = normalize_entity_set(gap_state.get("uncovered_query_entities", []) or [])
    relation_term_set = {str(token) for token in (gap_state.get("relation_terms", []) or []) if str(token)}
    slot_cue_set = {normalize_structure_text(token) for token in (gap_state.get("gap_slot_cues", []) or []) if normalize_structure_text(token)}
    bridge_target_set = normalize_entity_set(gap_state.get("gap_bridge_targets", []) or [])

    reranked_rows: List[Dict[str, object]] = []
    for raw_row in scored_candidates:
        row = dict(raw_row)
        pool_position = int(row.get("pool_position", -1) or -1)
        doc_text = ""
        if pool_docs is not None and 0 <= pool_position < len(pool_docs):
            doc_text = str(pool_docs[pool_position] or "")
        doc_tokens = normalized_token_set(doc_text)
        doc_entity_set = normalize_entity_set(row.get("doc_entities", set()) or set())
        micro_query_overlap = max(
            (_lexical_overlap_score(doc_tokens, query_tokens) for query_tokens in micro_query_tokens),
            default=0.0,
        )
        covered_overlap = _lexical_overlap_score(doc_entity_set, covered_entity_set)
        uncovered_overlap = _lexical_overlap_score(doc_entity_set, uncovered_query_entity_set)
        anchor_overlap = _lexical_overlap_score(doc_entity_set, anchor_entity_set or covered_entity_set)
        relation_overlap = _lexical_overlap_score(doc_tokens, relation_term_set)
        slot_overlap = _lexical_overlap_score(doc_tokens, slot_cue_set)
        bridge_target_overlap = _lexical_overlap_score(doc_entity_set, bridge_target_set)
        pair_bridge_score = min(
            1.0,
            min(
                max(covered_overlap, anchor_overlap),
                max(uncovered_overlap, micro_query_overlap),
            ),
        ) if (covered_entity_set or anchor_entity_set) and (uncovered_query_entity_set or micro_query_tokens) else 0.0
        multi_anchor_overlap = (
            float(len(doc_entity_set & (anchor_entity_set or covered_entity_set))) / float(max(1, min(2, len(anchor_entity_set or covered_entity_set))))
            if (anchor_entity_set or covered_entity_set)
            else 0.0
        )
        if gap_mode == "typed_abstain":
            witness_stats = _score_gap_witness_units(
                doc_text=doc_text,
                doc_entity_set=doc_entity_set,
                anchor_entity_set=anchor_entity_set,
                covered_entity_set=covered_entity_set,
                slot_cue_set=slot_cue_set,
                micro_query_tokens=micro_query_tokens,
                bridge_target_set=bridge_target_set,
            )
            best_witness_anchor = float(witness_stats.get("gap_best_witness_anchor", 0.0) or 0.0)
            best_witness_slot = float(witness_stats.get("gap_best_witness_slot", 0.0) or 0.0)
            best_witness_query = float(witness_stats.get("gap_best_witness_query", 0.0) or 0.0)
            best_witness_bridge_target = float(witness_stats.get("gap_best_witness_bridge_target", 0.0) or 0.0)
            non_anchor_entity_gain = float(witness_stats.get("gap_non_anchor_entity_gain", 0.0) or 0.0)
            best_role_joint = float(witness_stats.get("gap_best_role_witness_joint", 0.0) or 0.0)
            best_role_anchor_slot = float(witness_stats.get("gap_best_role_anchor_slot", 0.0) or 0.0)
            best_role_non_anchor = float(witness_stats.get("gap_best_role_non_anchor", 0.0) or 0.0)
            best_role_query = float(witness_stats.get("gap_best_role_query", 0.0) or 0.0)

            if gap_type == "bridge_entity":
                gap_filter_passed = bool(best_witness_anchor > 0.0 and best_witness_slot > 0.0 and max(best_witness_bridge_target, uncovered_overlap) > 0.0)
                witness_score = max(
                    min(best_witness_anchor, best_witness_slot, max(best_witness_bridge_target, uncovered_overlap)),
                    min(anchor_overlap, max(best_witness_bridge_target, uncovered_overlap)),
                )
            elif gap_type == "role_relation":
                gap_filter_passed = bool(
                    best_role_joint > 0.0
                    and best_role_anchor_slot > 0.0
                    and best_role_non_anchor > 0.0
                )
                witness_score = max(
                    best_role_joint,
                    min(best_role_anchor_slot, max(best_role_query, best_witness_query)),
                )
            else:
                gap_filter_passed = bool(best_witness_anchor > 0.0 and best_witness_slot > 0.0)
                witness_score = max(
                    min(best_witness_anchor, best_witness_slot),
                    min(anchor_overlap, max(best_witness_slot, slot_overlap)),
                )
            gap_score = float(np.clip(max(witness_score, best_witness_query), 0.0, 1.0))
            gap_gate_score = max(
                float(row.get("structure_score", 0.0) or 0.0),
                float(gap_score),
            )
            row.update({
                "gap_type": gap_type,
                "gap_mode": gap_mode,
                "gap_slot_overlap": round(float(slot_overlap), 4),
                "gap_bridge_target_overlap": round(float(bridge_target_overlap), 4),
                "gap_best_witness_anchor": round(float(best_witness_anchor), 4),
                "gap_best_witness_slot": round(float(best_witness_slot), 4),
                "gap_best_witness_query": round(float(best_witness_query), 4),
                "gap_best_witness_bridge_target": round(float(best_witness_bridge_target), 4),
                "gap_non_anchor_entity_gain": round(float(non_anchor_entity_gain), 4),
                "gap_non_anchor_entities": list(witness_stats.get("gap_non_anchor_entities", []) or []),
                "gap_best_role_witness_joint": round(float(witness_stats.get("gap_best_role_witness_joint", 0.0) or 0.0), 4),
                "gap_best_role_anchor_slot": round(float(witness_stats.get("gap_best_role_anchor_slot", 0.0) or 0.0), 4),
                "gap_best_role_non_anchor": round(float(witness_stats.get("gap_best_role_non_anchor", 0.0) or 0.0), 4),
                "gap_best_role_query": round(float(witness_stats.get("gap_best_role_query", 0.0) or 0.0), 4),
                "gap_best_role_unit_type": str(witness_stats.get("gap_best_role_unit_type", "") or ""),
                "gap_filter_passed": bool(gap_filter_passed),
                "gap_score": round(float(gap_score), 4),
                "gap_score_raw": float(gap_score),
                "gap_gate_score": round(float(gap_gate_score), 4),
                "gap_gate_score_raw": float(gap_gate_score),
                "gap_combined_score": round(float(gap_score), 4),
                "gap_combined_score_raw": float(gap_score),
            })
            reranked_rows.append(row)
            continue

        if gap_type == "bridge_entity":
            gap_score = float(np.clip(
                0.45 * pair_bridge_score
                + 0.25 * uncovered_overlap
                + 0.20 * micro_query_overlap
                + 0.10 * float(row.get("query_anchor_score", 0.0) or 0.0),
                0.0,
                1.0,
            ))
        elif gap_type == "target_attribute":
            gap_score = float(np.clip(
                0.40 * anchor_overlap
                + 0.35 * micro_query_overlap
                + 0.15 * relation_overlap
                + 0.10 * float(row.get("closure_score_raw", 0.0) or 0.0),
                0.0,
                1.0,
            ))
        elif gap_type == "linking_relation":
            gap_score = float(np.clip(
                0.35 * multi_anchor_overlap
                + 0.30 * micro_query_overlap
                + 0.20 * relation_overlap
                + 0.15 * float(row.get("path_coherence_score", 0.0) or 0.0),
                0.0,
                1.0,
            ))
        else:
            gap_score = float(np.clip(
                0.70 * micro_query_overlap
                + 0.30 * float(row.get("selection_score_raw", 0.0) or 0.0),
                0.0,
                1.0,
            ))
        gap_gate_score = max(
            float(row.get("structure_score", 0.0) or 0.0),
            float(gap_score),
        )
        gap_combined_score = (
            0.55 * float(gap_score)
            + 0.25 * float(row.get("selection_score_raw", 0.0) or 0.0)
            + 0.10 * float(row.get("base_score", 0.0) or 0.0)
            + 0.10 * float(row.get("novelty_score", 0.0) or 0.0)
        )
        row.update({
            "gap_type": gap_type,
            "gap_mode": gap_mode,
            "gap_micro_query_overlap": round(float(micro_query_overlap), 4),
            "gap_anchor_overlap": round(float(anchor_overlap), 4),
            "gap_covered_overlap": round(float(covered_overlap), 4),
            "gap_uncovered_overlap": round(float(uncovered_overlap), 4),
            "gap_relation_overlap": round(float(relation_overlap), 4),
            "gap_pair_bridge_score": round(float(pair_bridge_score), 4),
            "gap_score": round(float(gap_score), 4),
            "gap_score_raw": float(gap_score),
            "gap_gate_score": round(float(gap_gate_score), 4),
            "gap_gate_score_raw": float(gap_gate_score),
            "gap_combined_score": round(float(gap_combined_score), 4),
            "gap_combined_score_raw": float(gap_combined_score),
        })
        reranked_rows.append(row)

    reranked_rows.sort(
        key=lambda item: (
            -int(bool(item.get("gap_filter_passed", True))),
            -float(item.get("gap_best_role_witness_joint", 0.0) or 0.0),
            -float(item.get("gap_best_role_anchor_slot", 0.0) or 0.0),
            -float(item.get("gap_best_role_query", item.get("gap_best_witness_query", 0.0)) or 0.0),
            -float(item.get("gap_best_role_non_anchor", 0.0) or 0.0),
            -float(item.get("gap_non_anchor_entity_gain", 0.0) or 0.0),
            -float(item.get("gap_combined_score_raw", 0.0) or 0.0),
            -float(item.get("gap_best_witness_slot", item.get("gap_score_raw", 0.0)) or 0.0),
            -float(item.get("gap_best_witness_anchor", item.get("gap_anchor_overlap", 0.0)) or 0.0),
            -float(item.get("gap_score_raw", 0.0) or 0.0),
            -float(item.get("combined_score_raw", 0.0) or 0.0),
            -int(item.get("new_entity_count", 0) or 0),
            int(item.get("pool_position", 0) or 0),
        )
    )
    return reranked_rows


def _build_gap_candidate_preview(ranked_candidates: Sequence[Mapping[str, object]],
                                 *,
                                 limit: int = 5) -> List[Dict[str, object]]:
    preview_rows: List[Dict[str, object]] = []
    for rank, row in enumerate(list(ranked_candidates)[:max(int(limit), 0)], start=1):
        preview_rows.append({
            "preview_rank": int(rank),
            "pool_position": int(row.get("pool_position", -1) or -1),
            "doc_id": int(row["doc_id"]) if row.get("doc_id") is not None else None,
            "title": str(row.get("doc_title", "") or ""),
            "structure_score": round(float(row.get("structure_score", 0.0) or 0.0), 4),
            "closure_score": round(float(row.get("closure_score", 0.0) or 0.0), 4),
            "novelty_score": round(float(row.get("novelty_score", 0.0) or 0.0), 4),
            "combined_score": round(float(row.get("combined_score", 0.0) or 0.0), 4),
            "gap_score": round(float(row.get("gap_score", 0.0) or 0.0), 4),
            "gap_gate_score": round(float(row.get("gap_gate_score", 0.0) or 0.0), 4),
            "gap_combined_score": round(float(row.get("gap_combined_score", 0.0) or 0.0), 4),
            "gap_filter_passed": bool(row.get("gap_filter_passed", True)),
            "gap_best_witness_anchor": round(float(row.get("gap_best_witness_anchor", 0.0) or 0.0), 4),
            "gap_best_witness_slot": round(float(row.get("gap_best_witness_slot", 0.0) or 0.0), 4),
            "gap_best_role_witness_joint": round(float(row.get("gap_best_role_witness_joint", 0.0) or 0.0), 4),
            "gap_best_role_anchor_slot": round(float(row.get("gap_best_role_anchor_slot", 0.0) or 0.0), 4),
            "gap_best_role_non_anchor": round(float(row.get("gap_best_role_non_anchor", 0.0) or 0.0), 4),
            "gap_best_role_unit_type": str(row.get("gap_best_role_unit_type", "") or ""),
            "gap_unit_eligible": bool(row.get("gap_unit_eligible", False)),
            "gap_unit_anchor_pass": bool(row.get("gap_unit_anchor_pass", False)),
            "gap_unit_slot_pass": bool(row.get("gap_unit_slot_pass", False)),
            "gap_unit_non_anchor_pass": bool(row.get("gap_unit_non_anchor_pass", False)),
            "gap_unit_value_pass": bool(row.get("gap_unit_value_pass", False)),
            "gap_unit_query_score": round(float(row.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
            "gap_unit_type": str(row.get("gap_unit_type", "") or ""),
            "gap_unit_sentence_span": list(row.get("gap_unit_sentence_span", []) or []),
        })
    return preview_rows


def _build_gap_doc_rows_from_positions(positions: Sequence[int],
                                       *,
                                       pool_doc_ids: Sequence[int | None],
                                       pool_doc_titles: Sequence[str] | None,
                                       doc_idx_to_entities: Mapping[int, Set[str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pos in positions:
        pool_position = int(pos)
        if pool_position < 0 or pool_position >= len(pool_doc_ids):
            continue
        doc_id = pool_doc_ids[pool_position]
        doc_entities = normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())) if doc_id is not None else set()
        rows.append({
            "pool_position": int(pool_position),
            "doc_id": int(doc_id) if doc_id is not None else None,
            "doc_title": (
                str(pool_doc_titles[pool_position]).strip()
                if pool_doc_titles is not None and 0 <= pool_position < len(pool_doc_titles)
                else ""
            ),
            "doc_entities": doc_entities,
            "structure_score": 0.0,
            "combined_score_raw": 0.0,
        })
    return rows


def select_gap_expand_positions_unit_typed_abstain(pool_doc_ids: Sequence[int | None],
                                                   normalized_base_scores: np.ndarray,
                                                   pool_doc_titles: Sequence[str] | None,
                                                   doc_idx_to_entities: Dict[int, Set[str]],
                                                   doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                                   adjacency: Dict[str, List[Tuple[str, float, str]]],
                                                   initial_seed_entities: Sequence[str] | Set[str] | None,
                                                   query_entities: Sequence[str] | Set[str] | None,
                                                   pool_limit: int,
                                                   expand_base_k: int,
                                                   append_max_docs: int,
                                                   expand_min_structure_score: float,
                                                   structure_max_hops: int,
                                                   structure_seed_target_bridge_mode: str,
                                                   base_weight: float,
                                                   structure_weight: float,
                                                   novelty_weight: float,
                                                   score_mode: str,
                                                   non_anchor_title_dedup: bool,
                                                   append_random_seed: int,
                                                   query: str | None,
                                                   pool_docs: Sequence[str] | None,
                                                   gap_expand_max_queries: int | None,
                                                   ce_reranker: Any | None = None) -> Tuple[List[int], Dict[str, object]]:
    effective_pool_limit = max(int(pool_limit), 0)
    effective_base_k = min(max(int(expand_base_k), 0), effective_pool_limit)
    effective_append_max_docs = max(int(append_max_docs), 0)
    effective_gap_expand_max_queries = max(int(gap_expand_max_queries or DEFAULT_GAP_EXPAND_MAX_QUERIES), 1)
    baseline_prefix_positions = list(range(effective_base_k))
    baseline_titles = [
        str(pool_doc_titles[pos]).strip()
        for pos in baseline_prefix_positions
        if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
    ]
    covered_entities = normalize_entity_set(initial_seed_entities)
    for pos in baseline_prefix_positions:
        doc_id = pool_doc_ids[pos] if pos < len(pool_doc_ids) else None
        if doc_id is None:
            continue
        covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))

    gap_state = detect_gap_expand_state(
        query=str(query or ""),
        baseline_prefix_positions=baseline_prefix_positions,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        query_entities=query_entities,
        covered_entities=covered_entities,
        gap_expand_mode="unit_typed_abstain",
        gap_expand_max_queries=effective_gap_expand_max_queries,
    )

    unit_trace_defaults = {
        "gap_unit_count": 0,
        "gap_unit_eligible_count": 0,
        "gap_unit_score_source": "cross_encoder" if ce_reranker is not None else "lexical_fallback",
        "gap_unit_top_text": "",
        "gap_unit_top_parent_title": "",
        "gap_unit_top_parent_doc_id": None,
        "gap_unit_selected_parent_title": "",
        "gap_unit_selected_parent_doc_id": None,
        "gap_unit_selected_query_rate": 0.0,
        "gap_unit_selected_query_score": 0.0,
        "gap_unit_anchor_pass": False,
        "gap_unit_slot_pass": False,
        "gap_unit_non_anchor_pass": False,
        "gap_unit_value_pass": False,
        "gap_unit_selected_sentence_span": [],
        "gap_unit_bridge_primary_query_score": 0.0,
        "gap_unit_bridge_primary_parent_title": "",
        "gap_unit_bridge_primary_parent_doc_id": None,
    }

    if str(gap_state.get("gap_type", "")) == "abstain":
        fallback_positions, fallback_trace = select_bridge_append_positions(
            pool_doc_ids=pool_doc_ids,
            normalized_base_scores=normalized_base_scores,
            pool_doc_titles=pool_doc_titles,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            adjacency=adjacency,
            initial_seed_entities=initial_seed_entities,
            query_entities=query_entities,
            pool_limit=pool_limit,
            expand_base_k=expand_base_k,
            append_max_docs=append_max_docs,
            expand_min_structure_score=expand_min_structure_score,
            structure_max_hops=structure_max_hops,
            structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
            base_weight=base_weight,
            structure_weight=structure_weight,
            novelty_weight=novelty_weight,
            score_mode=score_mode,
            non_anchor_title_dedup=non_anchor_title_dedup,
            append_policy="bridge",
            append_random_seed=append_random_seed,
            query=query,
            pool_docs=pool_docs,
            gap_expand_mode="flat_fallback_only",
            gap_expand_max_queries=effective_gap_expand_max_queries,
            ce_reranker=ce_reranker,
        )
        trace = dict(fallback_trace)
        trace.update({
            "append_policy": "gap_expand",
            "gap_expand_enabled": True,
            "gap_expand_mode": "unit_typed_abstain",
            "gap_type": "abstain",
            "gap_mode": "unit_typed_abstain",
            "gap_fallback_used": True,
            "gap_slot": str(gap_state.get("gap_slot", "") or ""),
            "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
            "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
            "gap_micro_queries": list(gap_state.get("micro_queries", []) or []),
            "gap_micro_query_count": int(gap_state.get("micro_query_count", 0) or 0),
            "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
            "gap_candidate_positions": [],
            "gap_candidate_doc_ids": [],
            "gap_candidate_titles": [],
            "gap_abstain_reason": str(gap_state.get("abstain_reason", "") or "abstain"),
            "gap_steps": [{
                "step": 1,
                "gap_type": "abstain",
                "gap_mode": "unit_typed_abstain",
                "fallback_used": True,
                "abstain_reason": str(gap_state.get("abstain_reason", "") or "abstain"),
                "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                "gap_slot": str(gap_state.get("gap_slot", "") or ""),
                "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
                "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
                "micro_queries": list(gap_state.get("micro_queries", []) or []),
                "bridge_fallback_positions": list(fallback_trace.get("appended_positions", []) or []),
                "bridge_fallback_titles": list(fallback_trace.get("appended_titles", []) or []),
            }],
            **unit_trace_defaults,
        })
        return fallback_positions, trace

    bridge_primary_positions, bridge_primary_trace = select_bridge_append_positions(
        pool_doc_ids=pool_doc_ids,
        normalized_base_scores=normalized_base_scores,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        adjacency=adjacency,
        initial_seed_entities=initial_seed_entities,
        query_entities=query_entities,
        pool_limit=pool_limit,
        expand_base_k=expand_base_k,
        append_max_docs=min(effective_append_max_docs, 1),
        expand_min_structure_score=expand_min_structure_score,
        structure_max_hops=structure_max_hops,
        structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
        base_weight=base_weight,
        structure_weight=structure_weight,
        novelty_weight=novelty_weight,
        score_mode=score_mode,
        non_anchor_title_dedup=non_anchor_title_dedup,
        append_policy="bridge",
        append_random_seed=append_random_seed,
        query=query,
        pool_docs=pool_docs,
        gap_expand_mode="flat_fallback_only",
        gap_expand_max_queries=effective_gap_expand_max_queries,
        ce_reranker=ce_reranker,
    )

    candidate_positions = list(bridge_primary_trace.get("candidate_set_positions", baseline_prefix_positions) or baseline_prefix_positions)
    appended_positions = list(bridge_primary_trace.get("appended_positions", []) or [])
    append_steps: List[Dict[str, object]] = []
    gap_steps: List[Dict[str, object]] = [{
        "step": 1,
        "gap_type": str(gap_state.get("gap_type", "abstain") or "abstain"),
        "gap_mode": "unit_typed_abstain",
        "fallback_used": False,
        "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
        "gap_slot": str(gap_state.get("gap_slot", "") or ""),
        "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
        "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
        "micro_queries": list(gap_state.get("micro_queries", []) or []),
        "baseline_titles": list(baseline_titles),
    }]
    if bridge_primary_trace.get("append_steps"):
        primary_step = dict((bridge_primary_trace.get("append_steps") or [])[0])
        primary_step["selection_policy"] = "bridge_primary"
        primary_step["gap_source"] = "bridge_primary"
        append_steps.append(primary_step)

    if effective_append_max_docs <= 0:
        append_stop_reason = "append_cap_zero"
        unit_summary = dict(unit_trace_defaults)
        bridge_primary_unit_summary = {}
        bridge_primary_unit_row: Dict[str, object] = {}
    elif not appended_positions:
        append_stop_reason = str(bridge_primary_trace.get("append_stop_reason", "bridge_primary_unavailable") or "bridge_primary_unavailable")
        unit_summary = dict(unit_trace_defaults)
        bridge_primary_unit_summary = {}
        bridge_primary_unit_row = {}
    elif effective_append_max_docs == 1:
        append_stop_reason = "bridge_primary_only"
        unit_summary = dict(unit_trace_defaults)
        bridge_primary_unit_summary = {}
        bridge_primary_unit_row = {}
    else:
        current_covered_entities = normalize_entity_set(initial_seed_entities)
        for pos in candidate_positions:
            if pos < 0 or pos >= len(pool_doc_ids):
                continue
            doc_id = pool_doc_ids[pos]
            if doc_id is None:
                continue
            current_covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))

        seen_title_keys = normalize_title_set([
            str(pool_doc_titles[pos]).strip()
            for pos in candidate_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ]) if non_anchor_title_dedup else set()

        bridge_primary_rows = _build_gap_doc_rows_from_positions(
            appended_positions[:1],
            pool_doc_ids=pool_doc_ids,
            pool_doc_titles=pool_doc_titles,
            doc_idx_to_entities=doc_idx_to_entities,
        )
        bridge_primary_reference_covered = normalize_entity_set(current_covered_entities)
        for bridge_primary_row in bridge_primary_rows:
            bridge_primary_reference_covered -= normalize_entity_set(bridge_primary_row.get("doc_entities", set()) or set())
        bridge_primary_ranked_rows, bridge_primary_unit_summary = rerank_gap_expand_units(
            bridge_primary_rows,
            pool_docs=pool_docs,
            gap_state=gap_state,
            covered_entities=bridge_primary_reference_covered,
            ce_reranker=ce_reranker,
        )
        bridge_primary_unit_row = dict(bridge_primary_ranked_rows[0]) if bridge_primary_ranked_rows else {}

        remaining_positions = [
            pos for pos in range(effective_base_k, effective_pool_limit)
            if pos not in set(appended_positions)
        ]
        if not remaining_positions:
            append_stop_reason = "no_candidate_remaining"
            unit_summary = dict(unit_trace_defaults)
        else:
            scored_candidates = score_bridge_candidates(
                pool_doc_ids=pool_doc_ids,
                normalized_base_scores=normalized_base_scores,
                pool_doc_titles=pool_doc_titles,
                doc_idx_to_entities=doc_idx_to_entities,
                doc_idx_to_edges=doc_idx_to_edges,
                adjacency=adjacency,
                remaining_positions=remaining_positions,
                covered_entities=current_covered_entities,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                query_entities=query_entities,
                score_mode=score_mode,
            )
            if not scored_candidates:
                append_stop_reason = "no_scored_candidate"
                unit_summary = dict(unit_trace_defaults)
            else:
                ranked_candidates, unit_summary = rerank_gap_expand_units(
                    scored_candidates=scored_candidates,
                    pool_docs=pool_docs,
                    gap_state=gap_state,
                    covered_entities=current_covered_entities,
                    ce_reranker=ce_reranker,
                )
                gap_steps.append({
                    "step": 2,
                    "gap_type": str(gap_state.get("gap_type", "abstain") or "abstain"),
                    "gap_mode": "unit_typed_abstain",
                    "fallback_used": False,
                    "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                    "gap_slot": str(gap_state.get("gap_slot", "") or ""),
                    "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
                    "micro_queries": list(gap_state.get("micro_queries", []) or []),
                    "candidate_preview": _build_gap_candidate_preview(ranked_candidates),
                    **unit_summary,
                })

                selected_row = None
                duplicate_skip_count = 0
                not_better_skip_count = 0
                ineligible_skip_count = 0
                primary_rank_tuple = _gap_unit_rank_tuple(bridge_primary_unit_row, str(gap_state.get("gap_type", "") or "abstain"))
                for row in ranked_candidates:
                    if not bool(row.get("gap_unit_eligible", False)):
                        ineligible_skip_count += 1
                        continue
                    title_key = normalize_structure_text(str(row.get("doc_title", "")).strip())
                    if non_anchor_title_dedup and title_key and title_key in seen_title_keys:
                        duplicate_skip_count += 1
                        continue
                    if _gap_unit_rank_tuple(row, str(gap_state.get("gap_type", "") or "abstain")) <= primary_rank_tuple:
                        not_better_skip_count += 1
                        continue
                    selected_row = dict(row)
                    break

                if selected_row is None:
                    if ineligible_skip_count > 0 and ineligible_skip_count == len(ranked_candidates):
                        append_stop_reason = "no_eligible_gap_units"
                    elif not_better_skip_count > 0:
                        append_stop_reason = "gap_unit_not_better_than_bridge_primary"
                    elif duplicate_skip_count > 0:
                        append_stop_reason = "duplicate_title_only"
                    else:
                        append_stop_reason = "no_unit_alternate"
                else:
                    selected_position = int(selected_row["pool_position"])
                    candidate_positions.append(selected_position)
                    appended_positions.append(selected_position)
                    selected_doc_id = selected_row.get("doc_id")
                    if selected_doc_id is not None:
                        current_covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(selected_doc_id), set())))
                    selected_title_key = normalize_structure_text(str(selected_row.get("doc_title", "")).strip())
                    if non_anchor_title_dedup and selected_title_key:
                        seen_title_keys.add(selected_title_key)
                    append_steps.append({
                        "step": 2,
                        "selection_policy": "gap_unit_alternate",
                        "gap_source": "unit_typed_alternate",
                        "selected_pool_position": int(selected_position),
                        "selected_doc_id": int(selected_doc_id) if selected_doc_id is not None else None,
                        "selected_title": str(selected_row.get("doc_title", "") or ""),
                        "gap_type": str(gap_state.get("gap_type", "") or ""),
                        "gap_slot": str(gap_state.get("gap_slot", "") or ""),
                        "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                        "gap_micro_queries": list(gap_state.get("micro_queries", []) or []),
                        "gap_unit_text": str(selected_row.get("gap_unit_text", "") or ""),
                        "gap_unit_type": str(selected_row.get("gap_unit_type", "") or ""),
                        "gap_unit_sentence_span": list(selected_row.get("gap_unit_sentence_span", []) or []),
                        "gap_unit_query_score": round(float(selected_row.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
                        "gap_unit_anchor_pass": bool(selected_row.get("gap_unit_anchor_pass", False)),
                        "gap_unit_slot_pass": bool(selected_row.get("gap_unit_slot_pass", False)),
                        "gap_unit_non_anchor_pass": bool(selected_row.get("gap_unit_non_anchor_pass", False)),
                        "gap_unit_value_pass": bool(selected_row.get("gap_unit_value_pass", False)),
                        "gap_unit_bridge_primary_query_score": round(float(bridge_primary_unit_row.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
                        "gap_unit_bridge_primary_title": str(bridge_primary_unit_row.get("doc_title", "") or ""),
                        "candidate_preview": _build_gap_candidate_preview(ranked_candidates),
                    })
                    append_stop_reason = "unit_alternate_added"

        covered_entities = current_covered_entities if 'current_covered_entities' in locals() else covered_entities

    gap_candidate_positions = appended_positions[1:] if len(appended_positions) >= 2 else []
    selected_gap_row = None
    if gap_candidate_positions:
        selected_gap_row = next(
            (
                row for row in ([] if 'ranked_candidates' not in locals() else ranked_candidates)
                if int(row.get("pool_position", -1) or -1) == int(gap_candidate_positions[0])
            ),
            None,
        )
    unit_trace = dict(unit_trace_defaults)
    unit_trace.update(dict(unit_summary if 'unit_summary' in locals() else {}))
    unit_trace.update({
        "gap_unit_selected_parent_title": str((selected_gap_row or {}).get("doc_title", "") or ""),
        "gap_unit_selected_parent_doc_id": (
            int((selected_gap_row or {}).get("doc_id"))
            if (selected_gap_row or {}).get("doc_id") is not None else None
        ),
        "gap_unit_selected_query_rate": round(float((selected_gap_row or {}).get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
        "gap_unit_selected_query_score": round(float((selected_gap_row or {}).get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
        "gap_unit_anchor_pass": bool((selected_gap_row or {}).get("gap_unit_anchor_pass", False)),
        "gap_unit_slot_pass": bool((selected_gap_row or {}).get("gap_unit_slot_pass", False)),
        "gap_unit_non_anchor_pass": bool((selected_gap_row or {}).get("gap_unit_non_anchor_pass", False)),
        "gap_unit_value_pass": bool((selected_gap_row or {}).get("gap_unit_value_pass", False)),
        "gap_unit_selected_sentence_span": list((selected_gap_row or {}).get("gap_unit_sentence_span", []) or []),
        "gap_unit_bridge_primary_query_score": round(float((bridge_primary_unit_row or {}).get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
        "gap_unit_bridge_primary_parent_title": str((bridge_primary_unit_row or {}).get("doc_title", "") or ""),
        "gap_unit_bridge_primary_parent_doc_id": (
            int((bridge_primary_unit_row or {}).get("doc_id"))
            if (bridge_primary_unit_row or {}).get("doc_id") is not None else None
        ),
    })

    trace = {
        "selector": "bridge_append",
        "expand_base_k": int(effective_base_k),
        "append_max_docs": int(effective_append_max_docs),
        "append_policy": "gap_expand",
        "append_random_seed": int(append_random_seed),
        "gap_expand_mode": "unit_typed_abstain",
        "gap_expand_max_queries": int(effective_gap_expand_max_queries),
        "expand_min_structure_score": round(float(expand_min_structure_score), 4),
        "score_mode": normalize_setwise_score_mode(score_mode),
        "non_anchor_title_dedup": bool(non_anchor_title_dedup),
        "gap_expand_enabled": True,
        "baseline_prefix_positions": list(baseline_prefix_positions),
        "baseline_prefix_titles": list(baseline_titles),
        "appended_positions": list(appended_positions),
        "appended_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in appended_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "append_count": int(len(appended_positions)),
        "append_stop_reason": str(append_stop_reason),
        "append_steps": append_steps,
        "gap_steps": gap_steps,
        "gap_type": str(gap_state.get("gap_type", "abstain") or "abstain"),
        "gap_mode": "unit_typed_abstain",
        "gap_fallback_used": False,
        "gap_abstain_reason": str(gap_state.get("abstain_reason", "") or ""),
        "gap_slot": str(gap_state.get("gap_slot", "") or ""),
        "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
        "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
        "gap_micro_queries": list(gap_state.get("micro_queries", []) or []),
        "gap_micro_query_count": int(len(gap_state.get("micro_queries", []) or [])),
        "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
        "gap_candidate_positions": list(gap_candidate_positions),
        "gap_candidate_doc_ids": [
            int(pool_doc_ids[pos]) if pos < len(pool_doc_ids) and pool_doc_ids[pos] is not None else None
            for pos in gap_candidate_positions
        ],
        "gap_candidate_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in gap_candidate_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "candidate_set_positions": list(candidate_positions),
        "candidate_set_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in candidate_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "candidate_set_size": int(len(candidate_positions)),
        "covered_entity_count_after_expand": int(len(covered_entities)),
        "bridge_primary_positions": list(appended_positions[:1]),
        "bridge_primary_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in appended_positions[:1]
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        **unit_trace,
    }
    return candidate_positions, trace


def select_gap_expand_positions_typed_abstain(pool_doc_ids: Sequence[int | None],
                                              normalized_base_scores: np.ndarray,
                                              pool_doc_titles: Sequence[str] | None,
                                              doc_idx_to_entities: Dict[int, Set[str]],
                                              doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                              adjacency: Dict[str, List[Tuple[str, float, str]]],
                                              initial_seed_entities: Sequence[str] | Set[str] | None,
                                              query_entities: Sequence[str] | Set[str] | None,
                                              pool_limit: int,
                                              expand_base_k: int,
                                              append_max_docs: int,
                                              expand_min_structure_score: float,
                                              structure_max_hops: int,
                                              structure_seed_target_bridge_mode: str,
                                              base_weight: float,
                                              structure_weight: float,
                                              novelty_weight: float,
                                              score_mode: str,
                                              non_anchor_title_dedup: bool,
                                              append_random_seed: int,
                                              query: str | None,
                                              pool_docs: Sequence[str] | None,
                                              gap_expand_max_queries: int | None,
                                              ce_reranker: Any | None = None) -> Tuple[List[int], Dict[str, object]]:
    effective_pool_limit = max(int(pool_limit), 0)
    effective_base_k = min(max(int(expand_base_k), 0), effective_pool_limit)
    effective_append_max_docs = max(int(append_max_docs), 0)
    effective_gap_expand_max_queries = max(int(gap_expand_max_queries or DEFAULT_GAP_EXPAND_MAX_QUERIES), 1)
    baseline_prefix_positions = list(range(effective_base_k))
    baseline_titles = [
        str(pool_doc_titles[pos]).strip()
        for pos in baseline_prefix_positions
        if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
    ]
    covered_entities = normalize_entity_set(initial_seed_entities)
    for pos in baseline_prefix_positions:
        doc_id = pool_doc_ids[pos] if pos < len(pool_doc_ids) else None
        if doc_id is None:
            continue
        covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))

    gap_state = detect_gap_expand_state(
        query=str(query or ""),
        baseline_prefix_positions=baseline_prefix_positions,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        doc_idx_to_entities=doc_idx_to_entities,
        query_entities=query_entities,
        covered_entities=covered_entities,
        gap_expand_mode="typed_abstain",
        gap_expand_max_queries=effective_gap_expand_max_queries,
    )

    if str(gap_state.get("gap_type", "")) == "abstain":
        fallback_positions, fallback_trace = select_bridge_append_positions(
            pool_doc_ids=pool_doc_ids,
            normalized_base_scores=normalized_base_scores,
            pool_doc_titles=pool_doc_titles,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            adjacency=adjacency,
            initial_seed_entities=initial_seed_entities,
            query_entities=query_entities,
            pool_limit=pool_limit,
            expand_base_k=expand_base_k,
            append_max_docs=append_max_docs,
            expand_min_structure_score=expand_min_structure_score,
            structure_max_hops=structure_max_hops,
            structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
            base_weight=base_weight,
            structure_weight=structure_weight,
            novelty_weight=novelty_weight,
            score_mode=score_mode,
            non_anchor_title_dedup=non_anchor_title_dedup,
            append_policy="bridge",
            append_random_seed=append_random_seed,
            query=query,
            pool_docs=pool_docs,
            gap_expand_mode="flat_fallback_only",
            gap_expand_max_queries=effective_gap_expand_max_queries,
        )
        trace = dict(fallback_trace)
        trace.update({
            "append_policy": "gap_expand",
            "gap_expand_enabled": True,
            "gap_expand_mode": "typed_abstain",
            "gap_type": "abstain",
            "gap_mode": "typed_abstain",
            "gap_fallback_used": True,
            "gap_slot": str(gap_state.get("gap_slot", "") or ""),
            "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
            "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
            "gap_micro_queries": list(gap_state.get("micro_queries", []) or []),
            "gap_micro_query_count": int(gap_state.get("micro_query_count", 0) or 0),
            "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
            "gap_candidate_positions": [],
            "gap_candidate_doc_ids": [],
            "gap_candidate_titles": [],
            "gap_abstain_reason": str(gap_state.get("abstain_reason", "") or "abstain"),
            "gap_steps": [{
                "step": 1,
                "gap_type": "abstain",
                "gap_mode": "typed_abstain",
                "fallback_used": True,
                "abstain_reason": str(gap_state.get("abstain_reason", "") or "abstain"),
                "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                "gap_slot": str(gap_state.get("gap_slot", "") or ""),
                "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
                "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
                "micro_queries": list(gap_state.get("micro_queries", []) or []),
                "bridge_fallback_positions": list(fallback_trace.get("appended_positions", []) or []),
                "bridge_fallback_titles": list(fallback_trace.get("appended_titles", []) or []),
            }],
        })
        return fallback_positions, trace

    bridge_primary_positions, bridge_primary_trace = select_bridge_append_positions(
        pool_doc_ids=pool_doc_ids,
        normalized_base_scores=normalized_base_scores,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        adjacency=adjacency,
        initial_seed_entities=initial_seed_entities,
        query_entities=query_entities,
        pool_limit=pool_limit,
        expand_base_k=expand_base_k,
        append_max_docs=min(effective_append_max_docs, 1),
        expand_min_structure_score=expand_min_structure_score,
        structure_max_hops=structure_max_hops,
        structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
        base_weight=base_weight,
        structure_weight=structure_weight,
        novelty_weight=novelty_weight,
        score_mode=score_mode,
        non_anchor_title_dedup=non_anchor_title_dedup,
        append_policy="bridge",
        append_random_seed=append_random_seed,
        query=query,
        pool_docs=pool_docs,
        gap_expand_mode="flat_fallback_only",
        gap_expand_max_queries=effective_gap_expand_max_queries,
    )

    candidate_positions = list(bridge_primary_trace.get("candidate_set_positions", baseline_prefix_positions) or baseline_prefix_positions)
    appended_positions = list(bridge_primary_trace.get("appended_positions", []) or [])
    append_steps: List[Dict[str, object]] = []
    gap_steps: List[Dict[str, object]] = [{
        "step": 1,
        "gap_type": str(gap_state.get("gap_type", "abstain") or "abstain"),
        "gap_mode": "typed_abstain",
        "fallback_used": False,
        "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
        "gap_slot": str(gap_state.get("gap_slot", "") or ""),
        "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
        "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
        "micro_queries": list(gap_state.get("micro_queries", []) or []),
        "baseline_titles": list(baseline_titles),
    }]
    if bridge_primary_trace.get("append_steps"):
        primary_step = dict((bridge_primary_trace.get("append_steps") or [])[0])
        primary_step["selection_policy"] = "bridge_primary"
        primary_step["gap_source"] = "bridge_primary"
        append_steps.append(primary_step)

    if effective_append_max_docs <= 0:
        append_stop_reason = "append_cap_zero"
    elif not appended_positions:
        append_stop_reason = str(bridge_primary_trace.get("append_stop_reason", "bridge_primary_unavailable") or "bridge_primary_unavailable")
    elif effective_append_max_docs == 1:
        append_stop_reason = "bridge_primary_only"
    else:
        current_covered_entities = normalize_entity_set(initial_seed_entities)
        for pos in candidate_positions:
            if pos < 0 or pos >= len(pool_doc_ids):
                continue
            doc_id = pool_doc_ids[pos]
            if doc_id is None:
                continue
            current_covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))

        seen_title_keys = normalize_title_set([
            str(pool_doc_titles[pos]).strip()
            for pos in candidate_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ]) if non_anchor_title_dedup else set()

        remaining_positions = [
            pos for pos in range(effective_base_k, effective_pool_limit)
            if pos not in set(appended_positions)
        ]
        if not remaining_positions:
            append_stop_reason = "no_candidate_remaining"
        else:
            scored_candidates = score_bridge_candidates(
                pool_doc_ids=pool_doc_ids,
                normalized_base_scores=normalized_base_scores,
                pool_doc_titles=pool_doc_titles,
                doc_idx_to_entities=doc_idx_to_entities,
                doc_idx_to_edges=doc_idx_to_edges,
                adjacency=adjacency,
                remaining_positions=remaining_positions,
                covered_entities=current_covered_entities,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                query_entities=query_entities,
                score_mode=score_mode,
            )
            if not scored_candidates:
                append_stop_reason = "no_scored_candidate"
            else:
                ranked_candidates = rerank_gap_expand_candidates(
                    scored_candidates=scored_candidates,
                    pool_docs=pool_docs,
                    gap_state=gap_state,
                    covered_entities=current_covered_entities,
                )
                gap_steps.append({
                    "step": 2,
                    "gap_type": str(gap_state.get("gap_type", "abstain") or "abstain"),
                    "gap_mode": "typed_abstain",
                    "fallback_used": False,
                    "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                    "gap_slot": str(gap_state.get("gap_slot", "") or ""),
                    "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
                    "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
                    "micro_queries": list(gap_state.get("micro_queries", []) or []),
                    "candidate_preview": _build_gap_candidate_preview(ranked_candidates),
                })

                selected_row = None
                duplicate_skip_count = 0
                filtered_below_threshold = 0
                role_reference_skip_count = 0
                role_reference_trace: Dict[str, object] = {}
                role_reference_tuple: Tuple[float, float, float, float, float] | None = None
                if str(gap_state.get("gap_type", "")) == "role_relation" and pool_docs is not None:
                    role_reference_rows: List[Dict[str, object]] = []
                    for reference_position in candidate_positions:
                        if reference_position < 0 or reference_position >= len(pool_docs) or reference_position >= len(pool_doc_ids):
                            continue
                        reference_doc_id = pool_doc_ids[reference_position]
                        if reference_doc_id is None:
                            continue
                        reference_doc_entities = normalize_entity_set(doc_idx_to_entities.get(int(reference_doc_id), set()))
                        reference_witness_stats = _score_gap_witness_units(
                            doc_text=str(pool_docs[reference_position] or ""),
                            doc_entity_set=reference_doc_entities,
                            anchor_entity_set=gap_state.get("gap_anchors", []) or [],
                            covered_entity_set=(current_covered_entities - reference_doc_entities),
                            slot_cue_set=gap_state.get("gap_slot_cues", []) or [],
                            micro_query_tokens=[
                                normalized_token_set(str(micro_query))
                                for micro_query in (gap_state.get("micro_queries", []) or [])
                                if str(micro_query).strip()
                            ],
                            bridge_target_set=gap_state.get("gap_bridge_targets", []) or [],
                        )
                        role_reference_rows.append({
                            "pool_position": int(reference_position),
                            "doc_id": int(reference_doc_id),
                            "title": extract_doc_title(str(pool_docs[reference_position] or "")),
                            "role_tuple": list(_role_relation_witness_tuple(reference_witness_stats)),
                            "gap_best_role_witness_joint": round(float(reference_witness_stats.get("gap_best_role_witness_joint", 0.0) or 0.0), 4),
                            "gap_best_role_anchor_slot": round(float(reference_witness_stats.get("gap_best_role_anchor_slot", 0.0) or 0.0), 4),
                            "gap_best_role_non_anchor": round(float(reference_witness_stats.get("gap_best_role_non_anchor", 0.0) or 0.0), 4),
                            "gap_best_role_query": round(float(reference_witness_stats.get("gap_best_role_query", 0.0) or 0.0), 4),
                            "gap_best_role_unit_type": str(reference_witness_stats.get("gap_best_role_unit_type", "") or ""),
                        })
                    if role_reference_rows:
                        role_reference_rows.sort(
                            key=lambda item: (
                                -float((item.get("role_tuple") or [0.0])[0]),
                                -float((item.get("role_tuple") or [0.0, 0.0])[1]),
                                -float((item.get("role_tuple") or [0.0, 0.0, 0.0])[2]),
                                -float((item.get("role_tuple") or [0.0, 0.0, 0.0, 0.0])[3]),
                                int(item.get("pool_position", 0) or 0),
                            )
                        )
                        role_reference_trace = dict(role_reference_rows[0])
                        role_reference_tuple = tuple(float(value or 0.0) for value in role_reference_trace.get("role_tuple", []))
                        gap_steps[-1]["role_reference_witness"] = dict(role_reference_trace)

                for row in ranked_candidates:
                    if not bool(row.get("gap_filter_passed", False)):
                        continue
                    if float(row.get("gap_gate_score_raw", 0.0) or 0.0) < float(expand_min_structure_score):
                        filtered_below_threshold += 1
                        continue
                    title_key = normalize_structure_text(str(row.get("doc_title", "")).strip())
                    if non_anchor_title_dedup and title_key and title_key in seen_title_keys:
                        duplicate_skip_count += 1
                        continue
                    if (
                        str(gap_state.get("gap_type", "")) == "role_relation"
                        and role_reference_tuple is not None
                        and _role_relation_witness_tuple(row) <= role_reference_tuple
                    ):
                        role_reference_skip_count += 1
                        continue
                    selected_row = dict(row)
                    break

                if selected_row is None:
                    if filtered_below_threshold > 0:
                        append_stop_reason = "typed_alternate_below_threshold"
                    elif role_reference_skip_count > 0:
                        append_stop_reason = "role_reference_not_better"
                    elif duplicate_skip_count > 0:
                        append_stop_reason = "duplicate_title_only"
                    else:
                        append_stop_reason = "no_typed_alternate"
                else:
                    selected_position = int(selected_row["pool_position"])
                    candidate_positions.append(selected_position)
                    appended_positions.append(selected_position)
                    selected_doc_id = selected_row.get("doc_id")
                    if selected_doc_id is not None:
                        current_covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(selected_doc_id), set())))
                    selected_title_key = normalize_structure_text(str(selected_row.get("doc_title", "")).strip())
                    if non_anchor_title_dedup and selected_title_key:
                        seen_title_keys.add(selected_title_key)
                    append_steps.append({
                        "step": 2,
                        "selection_policy": "gap_alternate",
                        "gap_source": "typed_alternate",
                        "selected_pool_position": int(selected_position),
                        "selected_doc_id": int(selected_doc_id) if selected_doc_id is not None else None,
                        "selected_title": str(selected_row.get("doc_title", "") or ""),
                        "selected_structure_score": round(float(selected_row.get("structure_score", 0.0) or 0.0), 4),
                        "selected_closure_score": round(float(selected_row.get("closure_score", 0.0) or 0.0), 4),
                        "selected_novelty_score": round(float(selected_row.get("novelty_score", 0.0) or 0.0), 4),
                        "selected_combined_score": round(float(selected_row.get("combined_score", 0.0) or 0.0), 4),
                        "selected_gap_score": round(float(selected_row.get("gap_score", 0.0) or 0.0), 4),
                        "selected_gap_gate_score": round(float(selected_row.get("gap_gate_score", 0.0) or 0.0), 4),
                        "selected_gap_combined_score": round(float(selected_row.get("gap_combined_score", 0.0) or 0.0), 4),
                        "gap_type": str(gap_state.get("gap_type", "") or ""),
                        "gap_slot": str(gap_state.get("gap_slot", "") or ""),
                        "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                        "gap_micro_queries": list(gap_state.get("micro_queries", []) or []),
                        "candidate_pool_size": int(len(scored_candidates)),
                        "duplicate_skip_count": int(duplicate_skip_count),
                        "role_reference_skip_count": int(role_reference_skip_count),
                        "role_reference_witness": dict(role_reference_trace),
                        "selected_gap_best_role_witness_joint": round(float(selected_row.get("gap_best_role_witness_joint", 0.0) or 0.0), 4),
                        "selected_gap_best_role_anchor_slot": round(float(selected_row.get("gap_best_role_anchor_slot", 0.0) or 0.0), 4),
                        "selected_gap_best_role_non_anchor": round(float(selected_row.get("gap_best_role_non_anchor", 0.0) or 0.0), 4),
                        "selected_gap_best_role_unit_type": str(selected_row.get("gap_best_role_unit_type", "") or ""),
                        "candidate_preview": _build_gap_candidate_preview(ranked_candidates),
                    })
                    append_stop_reason = "alternate_only_complete"

        covered_entities = current_covered_entities if 'current_covered_entities' in locals() else covered_entities

    gap_candidate_positions = appended_positions[1:] if len(appended_positions) >= 2 else []
    trace = {
        "selector": "bridge_append",
        "expand_base_k": int(effective_base_k),
        "append_max_docs": int(effective_append_max_docs),
        "append_policy": "gap_expand",
        "append_random_seed": int(append_random_seed),
        "gap_expand_mode": "typed_abstain",
        "gap_expand_max_queries": int(effective_gap_expand_max_queries),
        "expand_min_structure_score": round(float(expand_min_structure_score), 4),
        "score_mode": normalize_setwise_score_mode(score_mode),
        "non_anchor_title_dedup": bool(non_anchor_title_dedup),
        "gap_expand_enabled": True,
        "baseline_prefix_positions": list(baseline_prefix_positions),
        "baseline_prefix_titles": list(baseline_titles),
        "appended_positions": list(appended_positions),
        "appended_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in appended_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "append_count": int(len(appended_positions)),
        "append_stop_reason": str(append_stop_reason),
        "append_steps": append_steps,
        "gap_steps": gap_steps,
        "gap_type": str(gap_state.get("gap_type", "abstain") or "abstain"),
        "gap_mode": "typed_abstain",
        "gap_fallback_used": False,
        "gap_abstain_reason": str(gap_state.get("abstain_reason", "") or ""),
        "gap_slot": str(gap_state.get("gap_slot", "") or ""),
        "gap_slot_cues": list(gap_state.get("gap_slot_cues", []) or []),
        "gap_bridge_targets": list(gap_state.get("gap_bridge_targets", []) or []),
        "gap_micro_queries": list(gap_state.get("micro_queries", []) or []),
        "gap_micro_query_count": int(len(gap_state.get("micro_queries", []) or [])),
        "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
        "gap_candidate_positions": list(gap_candidate_positions),
        "gap_candidate_doc_ids": [
            int(pool_doc_ids[pos]) if pos < len(pool_doc_ids) and pool_doc_ids[pos] is not None else None
            for pos in gap_candidate_positions
        ],
        "gap_candidate_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in gap_candidate_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "candidate_set_positions": list(candidate_positions),
        "candidate_set_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in candidate_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "candidate_set_size": int(len(candidate_positions)),
        "covered_entity_count_after_expand": int(len(covered_entities)),
        "bridge_primary_positions": list(appended_positions[:1]),
        "bridge_primary_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in appended_positions[:1]
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
    }
    return candidate_positions, trace


ASSEMBLE_MODES = {
    "none",
    "base_score",
    "embedding_similarity",
    "cross_encoder",
    "coverage",
    "ce_local_repair",
    "action_swap_v0_dryrun",
    "action_swap_v0_judge",
    "action_swap_v0_judge_relaxed",
    "action_swap_noisyor_flat",
    "action_swap_noisyor_dep",
    "action_swap_tiered_witness",
    "action_swap_propose_verify",
}

ASSEMBLE_CE_ACTIVE_MODES = {
    "cross_encoder",
    "coverage",
    "ce_local_repair",
    "action_swap_v0_dryrun",
    "action_swap_v0_judge",
    "action_swap_v0_judge_relaxed",
    "action_swap_noisyor_flat",
    "action_swap_noisyor_dep",
    "action_swap_tiered_witness",
    "action_swap_propose_verify",
}

ACTION_SWAP_V0_MODES = {
    "action_swap_v0_dryrun",
    "action_swap_v0_judge",
    "action_swap_v0_judge_relaxed",
}

ACTION_SWAP_V0_JUDGE_MODES = {
    "action_swap_v0_judge",
    "action_swap_v0_judge_relaxed",
}

ACTION_SWAP_V0_RELAXED_MODES = {
    "action_swap_v0_judge_relaxed",
}

ACTION_SWAP_NOISYOR_MODES = {
    "action_swap_noisyor_flat",
    "action_swap_noisyor_dep",
}

ACTION_SWAP_TIERED_WITNESS_MODES = {
    "action_swap_tiered_witness",
}

ACTION_SWAP_PROPOSE_VERIFY_MODES = {
    "action_swap_propose_verify",
}

ACTION_CONTROLLER_MODES = (
    ACTION_SWAP_V0_MODES
    | ACTION_SWAP_NOISYOR_MODES
    | ACTION_SWAP_TIERED_WITNESS_MODES
    | ACTION_SWAP_PROPOSE_VERIFY_MODES
)
DEFAULT_ACTION_SWAP_NOISYOR_MARGIN = 0.05

COVERAGE_SCORE_VARIANTS = {
    "qe_ce",
    "qeb_ce",
}

COVERAGE_ATOM_SOURCES = {
    "candidate_pool",
    "baseline_prefix",
    "baseline_anchored",
}

COVERAGE_ADMISSIBILITY_MODES = {
    "off",
    "budget_gap",
}

APPEND_POLICIES = {
    "bridge",
    "gap_expand",
    "next_deep",
    "random_deep",
}

GAP_EXPAND_MODES = {
    "heuristic",
    "typed_abstain",
    "unit_typed_abstain",
    "flat_fallback_only",
}
DEFAULT_GAP_EXPAND_MAX_QUERIES = 2


def normalize_assemble_mode(mode: str | None) -> str:
    normalized = str(mode or "none").strip().lower()
    if normalized not in ASSEMBLE_MODES:
        raise ValueError(f"Unsupported assemble mode: {mode}")
    return normalized


def resolve_action_swap_v0_legality_mode(action_mode: str | None) -> str:
    normalized_mode = str(action_mode or "").strip().lower()
    return "relaxed" if normalized_mode in ACTION_SWAP_V0_RELAXED_MODES else "current"


def normalize_coverage_score_variant(variant: str | None) -> str:
    normalized = str(variant or "qe_ce").strip().lower()
    if normalized not in COVERAGE_SCORE_VARIANTS:
        raise ValueError(f"Unsupported coverage score variant: {variant}")
    return normalized


def normalize_coverage_atom_source(atom_source: str | None) -> str:
    normalized = str(atom_source or "candidate_pool").strip().lower()
    if normalized not in COVERAGE_ATOM_SOURCES:
        raise ValueError(f"Unsupported coverage atom source: {atom_source}")
    return normalized


def normalize_coverage_admissibility_mode(mode: str | None) -> str:
    normalized = str(mode or "off").strip().lower()
    if normalized not in COVERAGE_ADMISSIBILITY_MODES:
        raise ValueError(f"Unsupported coverage admissibility mode: {mode}")
    return normalized


def normalize_append_policy(policy: str | None) -> str:
    normalized = str(policy or "bridge").strip().lower()
    if normalized not in APPEND_POLICIES:
        raise ValueError(f"Unsupported append policy: {policy}")
    return normalized


def normalize_gap_expand_mode(mode: str | None) -> str:
    normalized = str(mode or "heuristic").strip().lower()
    if normalized not in GAP_EXPAND_MODES:
        raise ValueError(f"Unsupported gap expand mode: {mode}")
    return normalized


def format_doc_for_assemble_rerank(doc_text: str) -> str:
    cleaned_doc = str(doc_text or "").strip()
    if not cleaned_doc:
        return ""
    title = extract_doc_title(cleaned_doc)
    if not title:
        return cleaned_doc
    first_line, _, remainder = cleaned_doc.partition("\n")
    if normalize_structure_text(first_line) == normalize_structure_text(title):
        return cleaned_doc
    remainder = cleaned_doc if normalize_structure_text(cleaned_doc) != normalize_structure_text(title) else ""
    return f"{title}\n{remainder}".strip()


def _normalize_candidate_positions_for_assemble(candidate_positions: Sequence[int],
                                                pool_docs: Sequence[str]) -> List[int]:
    normalized_positions: List[int] = []
    seen_positions: Set[int] = set()
    for pos in candidate_positions:
        normalized_pos = int(pos)
        if normalized_pos < 0 or normalized_pos >= len(pool_docs) or normalized_pos in seen_positions:
            continue
        seen_positions.add(normalized_pos)
        normalized_positions.append(normalized_pos)
    return normalized_positions


def _compute_assemble_score_rows(query: str,
                                 pool_docs: Sequence[str],
                                 pool_doc_ids: Sequence[int | None],
                                 pool_doc_scores: Sequence[float],
                                 candidate_positions: Sequence[int],
                                 assemble_mode: str,
                                 hipporag: HippoRAG,
                                 ce_reranker: Any = None,
                                 position_sources: Dict[int, str] | None = None) -> Tuple[List[Dict[str, object]], str, str]:
    normalized_mode = normalize_assemble_mode(assemble_mode)
    normalized_positions = _normalize_candidate_positions_for_assemble(candidate_positions, pool_docs)
    rows: List[Dict[str, object]] = []
    fallback_reason = ""

    if normalized_mode == "cross_encoder":
        if ce_reranker is None:
            raise ValueError("cross_encoder assemble_mode requires a loaded ce_reranker")
        pairs = [
            [query, format_doc_for_assemble_rerank(pool_docs[pos])]
            for pos in normalized_positions
        ]
        raw_scores = ce_reranker.compute_score(pairs)
        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]
        score_values = np.asarray(raw_scores, dtype=float)
        score_field = "cross_encoder_score"
    elif normalized_mode == "embedding_similarity":
        query_embedding = None
        query_embedding_store = getattr(hipporag, "query_to_embedding", {}) or {}
        if isinstance(query_embedding_store, dict):
            passage_query_embeddings = query_embedding_store.get("passage", {}) or {}
            if isinstance(passage_query_embeddings, dict):
                query_embedding = passage_query_embeddings.get(query)
        if query_embedding is None and hasattr(hipporag, "_get_passage_query_embeddings"):
            hipporag._get_passage_query_embeddings([query])
            query_embedding = (
                ((getattr(hipporag, "query_to_embedding", {}) or {}).get("passage", {}) or {}).get(query)
            )
        passage_embeddings = np.asarray(getattr(hipporag, "passage_embeddings", np.array([])))
        if query_embedding is None or passage_embeddings.size == 0:
            score_values = np.asarray([
                float(pool_doc_scores[pos]) if pos < len(pool_doc_scores) else 0.0
                for pos in normalized_positions
            ], dtype=float)
            fallback_reason = "missing_query_or_passage_embeddings"
            score_field = "base_score_fallback"
        else:
            query_vector = np.asarray(query_embedding, dtype=float).reshape(-1)
            similarity_scores: List[float] = []
            for pos in normalized_positions:
                doc_id = pool_doc_ids[pos]
                if doc_id is None or int(doc_id) >= len(passage_embeddings):
                    similarity_scores.append(float("-inf"))
                    continue
                passage_vector = np.asarray(passage_embeddings[int(doc_id)], dtype=float).reshape(-1)
                if passage_vector.size == 0 or passage_vector.shape != query_vector.shape:
                    similarity_scores.append(float("-inf"))
                    continue
                similarity_scores.append(float(np.dot(query_vector, passage_vector)))
            score_values = np.asarray(similarity_scores, dtype=float)
            score_field = "embedding_similarity"
    else:
        score_values = np.asarray([
            float(pool_doc_scores[pos]) if pos < len(pool_doc_scores) else 0.0
            for pos in normalized_positions
        ], dtype=float)
        score_field = "base_score"

    for pos, score_value in zip(normalized_positions, score_values.tolist()):
        rows.append({
            "pool_position": int(pos),
            "doc_id": int(pool_doc_ids[pos]) if pool_doc_ids[pos] is not None else None,
            "title": extract_doc_title(pool_docs[pos]),
            "source": str((position_sources or {}).get(int(pos), "candidate")),
            "base_score": float(pool_doc_scores[pos]) if pos < len(pool_doc_scores) else 0.0,
            "assemble_score": float(score_value),
        })
    return rows, str(score_field), str(fallback_reason)


def _sort_assemble_score_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    sorted_rows = list(rows)
    sorted_rows.sort(
        key=lambda row: (
            -float(row.get("assemble_score", float("-inf"))),
            -float(row.get("base_score", 0.0) or 0.0),
            int(row.get("pool_position", 0) or 0),
        )
    )
    return sorted_rows


def _normalize_structure_edges_for_doc(doc_id: int | None,
                                       doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]]) -> Set[Tuple[str, str]]:
    if doc_id is None:
        return set()
    return {
        (normalize_structure_text(src), normalize_structure_text(tgt))
        for src, tgt, _, _ in doc_idx_to_edges.get(int(doc_id), [])
        if normalize_structure_text(src) and normalize_structure_text(tgt)
    }


def _normalize_structure_entities_for_doc(doc_id: int | None,
                                          doc_idx_to_entities: Dict[int, Set[str]]) -> Set[str]:
    if doc_id is None:
        return set()
    return normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))


def extract_baseline_scaffold_positions_from_ranking_rows(
    ranking_rows: Sequence[Mapping[str, object]],
    baseline_prefix_positions: Sequence[int],
    qa_top_k: int,
) -> List[int]:
    baseline_position_set = {
        int(pos)
        for pos in baseline_prefix_positions
    }
    scaffold_positions: List[int] = []
    seen_positions: Set[int] = set()
    for row in ranking_rows:
        raw_pool_position = row.get("pool_position", -1)
        pool_position = int(raw_pool_position) if raw_pool_position is not None else -1
        if pool_position < 0 or pool_position in seen_positions:
            continue
        if baseline_position_set and pool_position not in baseline_position_set:
            continue
        scaffold_positions.append(pool_position)
        seen_positions.add(pool_position)
        if len(scaffold_positions) >= max(int(qa_top_k), 0):
            break
    if scaffold_positions or not ranking_rows:
        return scaffold_positions

    for row in ranking_rows:
        raw_pool_position = row.get("pool_position", -1)
        pool_position = int(raw_pool_position) if raw_pool_position is not None else -1
        if pool_position < 0 or pool_position in seen_positions:
            continue
        scaffold_positions.append(pool_position)
        seen_positions.add(pool_position)
        if len(scaffold_positions) >= max(int(qa_top_k), 0):
            break
    return scaffold_positions


def resolve_pool_position_identity(pool_position: int,
                                   *,
                                   pool_docs: Sequence[str],
                                   pool_doc_ids: Sequence[int | None],
                                   doc_text_to_chunk_id: Mapping[str, str] | None = None) -> Dict[str, object]:
    normalized_position = int(pool_position)
    doc_text = str(pool_docs[normalized_position]) if 0 <= normalized_position < len(pool_docs) else ""
    raw_doc_id = pool_doc_ids[normalized_position] if 0 <= normalized_position < len(pool_doc_ids) else None
    doc_id = int(raw_doc_id) if raw_doc_id is not None else None
    chunk_id = None
    if doc_text and doc_text_to_chunk_id:
        chunk_id = str(doc_text_to_chunk_id.get(doc_text, "") or "").strip() or None
    text_hash = compute_mdhash_id(doc_text) if doc_text else None
    return {
        "pool_position": normalized_position,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text_hash": text_hash,
    }


def _pool_positions_are_duplicate(pool_position_a: int,
                                  pool_position_b: int,
                                  *,
                                  pool_docs: Sequence[str],
                                  pool_doc_ids: Sequence[int | None],
                                  doc_text_to_chunk_id: Mapping[str, str] | None = None) -> bool:
    identity_a = resolve_pool_position_identity(
        pool_position_a,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
    )
    identity_b = resolve_pool_position_identity(
        pool_position_b,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
    )
    doc_id_a = identity_a.get("doc_id")
    doc_id_b = identity_b.get("doc_id")
    if doc_id_a is not None and doc_id_b is not None:
        return int(doc_id_a) == int(doc_id_b)
    chunk_id_a = identity_a.get("chunk_id")
    chunk_id_b = identity_b.get("chunk_id")
    if chunk_id_a and chunk_id_b:
        return str(chunk_id_a) == str(chunk_id_b)
    text_hash_a = identity_a.get("text_hash")
    text_hash_b = identity_b.get("text_hash")
    return bool(text_hash_a and text_hash_b and str(text_hash_a) == str(text_hash_b))


def build_action_swap_jobs(query: str,
                           scaffold_positions: Sequence[int],
                           appended_positions: Sequence[int],
                           ranking_rows: Sequence[Mapping[str, object]],
                           *,
                           pool_docs: Sequence[str],
                           pool_doc_ids: Sequence[int | None],
                           doc_text_to_chunk_id: Mapping[str, str] | None = None,
                           replace_bottom_n: int = 2) -> List[Dict[str, object]]:
    scaffold_list = [int(pos) for pos in scaffold_positions]
    if not scaffold_list or not appended_positions:
        return []

    score_row_by_position = {
        int(row.get("pool_position")): dict(row)
        for row in ranking_rows
        if row.get("pool_position") is not None and int(row.get("pool_position")) >= 0
    }
    replace_candidates = scaffold_list[-max(int(replace_bottom_n), 0):]
    scaffold_docs = [str(pool_docs[pos]) for pos in scaffold_list]
    jobs: List[Dict[str, object]] = []

    for candidate_position in [int(pos) for pos in appended_positions]:
        candidate_duplicate = any(
            _pool_positions_are_duplicate(
                candidate_position,
                scaffold_position,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                doc_text_to_chunk_id=doc_text_to_chunk_id,
            )
            for scaffold_position in scaffold_list
        )
        candidate_identity = resolve_pool_position_identity(
            candidate_position,
            pool_docs=pool_docs,
            pool_doc_ids=pool_doc_ids,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
        )
        candidate_row = score_row_by_position.get(candidate_position, {})
        candidate_score_value = candidate_row.get("assemble_score")
        candidate_score = float(candidate_score_value) if candidate_score_value is not None else float("-inf")
        for replace_position in replace_candidates:
            replace_index = next(
                (idx for idx, pos in enumerate(scaffold_list) if int(pos) == int(replace_position)),
                -1,
            )
            if replace_index < 0:
                continue
            swapped_positions = list(scaffold_list)
            swapped_positions[replace_index] = int(candidate_position)
            replace_identity = resolve_pool_position_identity(
                replace_position,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                doc_text_to_chunk_id=doc_text_to_chunk_id,
            )
            replace_row = score_row_by_position.get(int(replace_position), {})
            replace_score_value = replace_row.get("assemble_score")
            replace_score = float(replace_score_value) if replace_score_value is not None else float("-inf")
            jobs.append({
                "question": str(query),
                "baseline_positions": list(scaffold_list),
                "baseline_docs": list(scaffold_docs),
                "docs": [str(pool_docs[pos]) for pos in swapped_positions],
                "candidate_pool_position": int(candidate_position),
                "candidate_doc_id": candidate_identity.get("doc_id"),
                "candidate_chunk_id": candidate_identity.get("chunk_id"),
                "candidate_title": extract_doc_title(pool_docs[int(candidate_position)]),
                "candidate_assemble_score": float(candidate_score),
                "replace_pool_position": int(replace_position),
                "replace_doc_id": replace_identity.get("doc_id"),
                "replace_chunk_id": replace_identity.get("chunk_id"),
                "replace_incumbent_index": int(replace_index),
                "replace_incumbent_rank": int(replace_index + 1),
                "replace_incumbent_title": extract_doc_title(pool_docs[int(replace_position)]),
                "replace_incumbent_assemble_score": float(replace_score),
                "score_delta": float(candidate_score - replace_score),
                "swapped_positions": list(swapped_positions),
                "is_duplicate_with_scaffold": bool(candidate_duplicate),
            })
    return jobs


def apply_single_slot_preserving_swap(scaffold_positions: Sequence[int],
                                      candidate_position: int,
                                      replace_position: int) -> List[int]:
    replaced_positions: List[int] = []
    replaced = False
    for pool_position in scaffold_positions:
        normalized_position = int(pool_position)
        if not replaced and normalized_position == int(replace_position):
            replaced_positions.append(int(candidate_position))
            replaced = True
        else:
            replaced_positions.append(normalized_position)
    return replaced_positions


def filter_action_swap_legal_jobs(action_jobs: Sequence[Mapping[str, object]],
                                  *,
                                  legality_mode: str = "current") -> List[Dict[str, object]]:
    normalized_legality_mode = str(legality_mode or "current").strip().lower()
    legal_jobs: List[Dict[str, object]] = []
    for job in action_jobs:
        if bool(job.get("is_duplicate_with_scaffold", False)):
            continue
        if normalized_legality_mode == "current":
            if job.get("score_delta") is None or float(job.get("score_delta")) <= 0.0:
                continue
        legal_jobs.append(dict(job))
    return legal_jobs


def select_action_swap_v0_dryrun(action_jobs: Sequence[Mapping[str, object]],
                                 *,
                                 scaffold_positions: Sequence[int],
                                 action_mode: str = "action_swap_v0_dryrun") -> Dict[str, object]:
    before_positions = [int(pos) for pos in scaffold_positions]
    legality_mode = resolve_action_swap_v0_legality_mode(action_mode)
    decision: Dict[str, object] = {
        "action_mode": str(action_mode),
        "action_legality_mode": str(legality_mode),
        "action_executed": False,
        "action_type": "keep",
        "action_candidate_pool_position": None,
        "action_candidate_doc_id": None,
        "action_replace_pool_position": None,
        "action_replace_doc_id": None,
        "action_score_delta": None,
        "action_legal_action_count": 0,
        "action_total_jobs": int(len(action_jobs)),
        "action_skip_reason": "no_legal_actions",
        "final_front_positions_before_action": list(before_positions),
        "final_front_positions_after_action": list(before_positions),
    }

    legal_jobs = filter_action_swap_legal_jobs(action_jobs, legality_mode=legality_mode)
    decision["action_legal_action_count"] = int(len(legal_jobs))
    if not legal_jobs:
        return decision

    best_job = max(
        legal_jobs,
        key=lambda job: (
            float(job.get("score_delta")) if job.get("score_delta") is not None else float("-inf"),
            float(job.get("candidate_assemble_score")) if job.get("candidate_assemble_score") is not None else float("-inf"),
            -int(job.get("candidate_pool_position", 0) or 0),
            -int(job.get("replace_pool_position", 0) or 0),
        ),
    )
    final_positions = apply_single_slot_preserving_swap(
        scaffold_positions=before_positions,
        candidate_position=int(best_job["candidate_pool_position"]),
        replace_position=int(best_job["replace_pool_position"]),
    )
    decision.update({
        "action_executed": True,
        "action_type": "swap",
        "action_candidate_pool_position": int(best_job["candidate_pool_position"]),
        "action_candidate_doc_id": best_job.get("candidate_doc_id"),
        "action_replace_pool_position": int(best_job["replace_pool_position"]),
        "action_replace_doc_id": best_job.get("replace_doc_id"),
        "action_score_delta": round(float(best_job.get("score_delta", 0.0) or 0.0), 4),
        "action_skip_reason": None,
        "final_front_positions_after_action": list(final_positions),
    })
    return decision


def _run_action_swap_v0_judge_jobs(action_jobs: Sequence[Mapping[str, object]],
                                   *,
                                   judge_bundle: SetwiseLateRerankJudgeBundle,
                                   qa_top_k: int,
                                   max_doc_chars: int,
                                   max_completion_tokens: int = SETWISE_LLM_LATE_RERANK_REPAIR_MAX_COMPLETION_TOKENS) -> List[Dict[str, object]]:
    from run_swap_utility_judge import build_swap_utility_messages, parse_swap_utility_judge_response

    judged_rows: List[Dict[str, object]] = []
    for job in action_jobs:
        response_text = ""
        metadata: Dict[str, object] = {}
        judge_error = None
        try:
            response_text, metadata = judge_bundle.infer_fn(
                messages=build_swap_utility_messages(dict(job), qa_top_k=int(qa_top_k), max_doc_chars=int(max_doc_chars)),
                model=judge_bundle.model_name,
                response_format=judge_bundle.response_format,
                max_completion_tokens=int(max_completion_tokens),
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            judge_error = str(exc)
            metadata = {
                "judge_status": "judge_exception",
                "judge_error": judge_error,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "exception",
                "backend": judge_bundle.backend,
                "model": judge_bundle.model_name,
            }
        judged_rows.append({
            **dict(job),
            "judge_response": str(response_text or ""),
            "judge_trace": {
                "judge_status": str(metadata.get("judge_status", "ok")),
                "judge_error": metadata.get("judge_error") or judge_error,
                "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
                "finish_reason": str(metadata.get("finish_reason", "")),
                "backend": metadata.get("backend"),
                "model": metadata.get("model"),
                "response_text_source": metadata.get("response_text_source"),
            },
            "judge_parsed": parse_swap_utility_judge_response(str(response_text or "")),
        })
    return judged_rows


def select_action_swap_v0_judge(action_jobs: Sequence[Mapping[str, object]],
                                *,
                                scaffold_positions: Sequence[int],
                                judge_bundle: SetwiseLateRerankJudgeBundle,
                                qa_top_k: int,
                                max_doc_chars: int,
                                judge_results: Sequence[Mapping[str, object]] | None = None,
                                action_mode: str = "action_swap_v0_judge") -> Dict[str, object]:
    before_positions = [int(pos) for pos in scaffold_positions]
    legality_mode = resolve_action_swap_v0_legality_mode(action_mode)
    decision: Dict[str, object] = {
        "action_mode": str(action_mode),
        "action_legality_mode": str(legality_mode),
        "action_executed": False,
        "action_type": "keep",
        "action_candidate_pool_position": None,
        "action_candidate_doc_id": None,
        "action_replace_pool_position": None,
        "action_replace_doc_id": None,
        "action_score_delta": None,
        "action_legal_action_count": 0,
        "action_total_jobs": int(len(action_jobs)),
        "action_skip_reason": "no_legal_actions",
        "action_gate_verdict": None,
        "action_gate_confidence": None,
        "action_gate_reason": None,
        "final_front_positions_before_action": list(before_positions),
        "final_front_positions_after_action": list(before_positions),
    }

    legal_jobs = filter_action_swap_legal_jobs(action_jobs, legality_mode=legality_mode)
    decision["action_legal_action_count"] = int(len(legal_jobs))
    if not legal_jobs:
        return decision

    evaluated_jobs = [
        dict(row)
        for row in (
            judge_results
            if judge_results is not None else
            _run_action_swap_v0_judge_jobs(
                legal_jobs,
                judge_bundle=judge_bundle,
                qa_top_k=qa_top_k,
                max_doc_chars=max_doc_chars,
            )
        )
    ]
    helpful_jobs = [
        row for row in evaluated_jobs
        if str(((row.get("judge_parsed") or {}).get("verdict") or "")).strip().lower() == "helpful"
    ]
    if not helpful_jobs:
        decision["action_skip_reason"] = "no_helpful_actions"
        return decision

    best_job = max(
        helpful_jobs,
        key=lambda row: (
            (
                float((row.get("judge_parsed") or {}).get("confidence"))
                if (row.get("judge_parsed") or {}).get("confidence") is not None else
                float("-inf")
            ),
            float(row.get("score_delta")) if row.get("score_delta") is not None else float("-inf"),
            float(row.get("candidate_assemble_score")) if row.get("candidate_assemble_score") is not None else float("-inf"),
            -int(row.get("candidate_pool_position", 0) or 0),
            -int(row.get("replace_pool_position", 0) or 0),
        ),
    )
    final_positions = apply_single_slot_preserving_swap(
        scaffold_positions=before_positions,
        candidate_position=int(best_job["candidate_pool_position"]),
        replace_position=int(best_job["replace_pool_position"]),
    )
    judge_parsed = dict(best_job.get("judge_parsed") or {})
    decision.update({
        "action_executed": True,
        "action_type": "swap",
        "action_candidate_pool_position": int(best_job["candidate_pool_position"]),
        "action_candidate_doc_id": best_job.get("candidate_doc_id"),
        "action_replace_pool_position": int(best_job["replace_pool_position"]),
        "action_replace_doc_id": best_job.get("replace_doc_id"),
        "action_score_delta": round(float(best_job.get("score_delta", 0.0) or 0.0), 4),
        "action_skip_reason": None,
        "action_gate_verdict": str(judge_parsed.get("verdict") or "") or None,
        "action_gate_confidence": (
            round(float(judge_parsed.get("confidence", 0.0) or 0.0), 2)
            if judge_parsed.get("confidence") is not None else None
        ),
        "action_gate_reason": str(judge_parsed.get("reason") or "") or None,
        "final_front_positions_after_action": list(final_positions),
    })
    return decision


def _tokenize_support_text(text: str) -> Set[str]:
    normalized = normalize_structure_text(text)
    if not normalized:
        return set()
    return {
        token
        for token in normalized.split()
        if len(token) > 1
    }


def _sigmoid_support_score(value: float) -> float:
    clipped = float(np.clip(float(value), -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _compute_lexical_support_score(facet_text: str, window_text: str) -> float:
    facet_tokens = _tokenize_support_text(facet_text)
    if not facet_tokens:
        return 0.0
    window_tokens = _tokenize_support_text(window_text)
    if not window_tokens:
        return 0.0
    return float(len(facet_tokens & window_tokens) / max(len(facet_tokens), 1))


def _order_query_entities(query: str,
                          query_entities: Sequence[str] | Set[str] | None,
                          *,
                          max_entities: int = 4) -> List[str]:
    normalized_query = normalize_structure_text(query)
    unique_entities: List[str] = []
    seen_keys: Set[str] = set()
    for entity in query_entities or []:
        entity_text = " ".join(str(entity or "").split()).strip()
        entity_key = normalize_structure_text(entity_text)
        if not entity_text or not entity_key or entity_key in seen_keys:
            continue
        unique_entities.append(entity_text)
        seen_keys.add(entity_key)

    def _sort_key(entity_text: str) -> Tuple[int, int, str]:
        entity_key = normalize_structure_text(entity_text)
        index = normalized_query.find(entity_key) if normalized_query and entity_key else -1
        return (0 if index >= 0 else 1, index if index >= 0 else 10**9, entity_key)

    unique_entities.sort(key=_sort_key)
    return unique_entities[:max(int(max_entities), 0)]


def build_query_dependency_graph(query: str,
                                 *,
                                 query_entities: Sequence[str] | Set[str] | None = None,
                                 action_mode: str = "action_swap_noisyor_dep",
                                 max_facets: int = 4) -> Dict[str, object]:
    normalized_mode = str(action_mode or "action_swap_noisyor_dep").strip().lower()
    ordered_entities = _order_query_entities(query, query_entities, max_entities=max_facets)
    nodes: List[Dict[str, object]] = []
    dependency_mode = "flat_fallback"

    if normalized_mode == "action_swap_noisyor_dep" and len(ordered_entities) >= 2:
        head_entity = ordered_entities[0]
        tail_entity = ordered_entities[1]
        nodes = [
            {
                "id": "v1",
                "facet": f"Identify evidence about {head_entity}. Question: {query}",
                "parents": [],
            },
            {
                "id": "v2",
                "facet": f"Find evidence connecting {head_entity} and {tail_entity}. Question: {query}",
                "parents": ["v1"],
            },
            {
                "id": "v3",
                "facet": f"Resolve the part of the question about {tail_entity}. Question: {query}",
                "parents": ["v2"],
            },
        ]
        if len(ordered_entities) >= 3 and int(max_facets) >= 4:
            extra_entity = ordered_entities[2]
            nodes.append({
                "id": "v4",
                "facet": f"Resolve the remaining question detail about {extra_entity}. Question: {query}",
                "parents": ["v2"],
            })
        dependency_mode = "dependency"
    else:
        facet_texts: List[str] = []
        for entity_text in ordered_entities[: max(int(max_facets) - 1, 0)]:
            facet_texts.append(f"Evidence about {entity_text}. Question: {query}")
        if not facet_texts or len(facet_texts) < int(max_facets):
            facet_texts.append(f"Answer the question: {query}")
        facet_texts = facet_texts[:max(int(max_facets), 1)]
        nodes = [
            {
                "id": f"v{index + 1}",
                "facet": facet_text,
                "parents": [],
            }
            for index, facet_text in enumerate(facet_texts)
        ]

    normalized_nodes: List[Dict[str, object]] = []
    valid_ids: Set[str] = set()
    for node in nodes[:max(int(max_facets), 1)]:
        facet_id = str(node.get("id") or "").strip() or f"v{len(normalized_nodes) + 1}"
        if facet_id in valid_ids:
            facet_id = f"v{len(normalized_nodes) + 1}"
        valid_ids.add(facet_id)
        normalized_nodes.append({
            "id": facet_id,
            "facet": " ".join(str(node.get("facet") or "").split()).strip() or f"Answer the question: {query}",
            "parents": [],
        })
    valid_ids = {str(node["id"]) for node in normalized_nodes}
    for node, original_node in zip(normalized_nodes, nodes[:len(normalized_nodes)]):
        parents: List[str] = []
        for raw_parent in original_node.get("parents") or []:
            parent = str(raw_parent or "").strip()
            if parent and parent in valid_ids and parent != str(node["id"]) and parent not in parents:
                parents.append(parent)
        node["parents"] = parents

    return {
        "mode": dependency_mode,
        "nodes": normalized_nodes,
        "ordered_entities": ordered_entities,
    }


def split_doc_body_sentences(doc_text: str,
                             *,
                             max_sentences: int = 8) -> List[str]:
    full_text = str(doc_text or "").strip()
    body = full_text.split("\n", 1)[1] if "\n" in full_text else full_text
    if not body.strip():
        return []

    raw_segments: List[str] = []
    for line in body.splitlines():
        cleaned_line = " ".join(line.split()).strip()
        if not cleaned_line:
            continue
        parts = re.split(r"(?<=[.!?。！？])\s+", cleaned_line)
        raw_segments.extend(parts if parts else [cleaned_line])

    sentences: List[str] = []
    for segment in raw_segments:
        cleaned_segment = " ".join(str(segment or "").split()).strip()
        if not cleaned_segment:
            continue
        sentences.append(cleaned_segment)
        if len(sentences) >= max(int(max_sentences), 1):
            break
    if sentences:
        return sentences
    return [" ".join(body.split()).strip()]


def build_title_prefixed_windows(doc_text: str,
                                 *,
                                 max_sentences: int = 8,
                                 max_windows: int = 16) -> List[str]:
    title = extract_doc_title(doc_text)
    sentences = split_doc_body_sentences(doc_text, max_sentences=max_sentences)
    windows: List[str] = []

    def _append_window(text: str) -> None:
        cleaned_text = " ".join(str(text or "").split()).strip()
        if not cleaned_text:
            return
        window_text = f"{title}\n{cleaned_text}".strip() if title else cleaned_text
        if window_text not in windows:
            windows.append(window_text)

    for sentence in sentences:
        _append_window(sentence)
        if len(windows) >= int(max_windows):
            return windows[:int(max_windows)]

    for first_sentence, second_sentence in zip(sentences, sentences[1:]):
        _append_window(f"{first_sentence} {second_sentence}")
        if len(windows) >= int(max_windows):
            return windows[:int(max_windows)]

    if not windows:
        _append_window(doc_text)
    return windows[:int(max_windows)]


_ANSWER_SCENT_MONTH_TOKENS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
_ANSWER_SCENT_LOCATION_TOKENS = {
    "city", "country", "state", "province", "county", "river", "gulf",
    "lake", "island", "region", "capital", "place", "town", "village",
    "located", "born", "from", "north", "south", "east", "west",
}
_ANSWER_SCENT_PERSON_TOKENS = {
    "person", "actor", "actress", "director", "singer", "player", "author",
    "writer", "president", "minister", "scientist", "artist", "founder",
    "wife", "husband", "father", "mother", "spouse",
}
_ANSWER_SCENT_ENTITY_STOPWORDS = {
    "answer", "question", "evidence", "identify", "find", "resolve", "detail",
    "target", "property", "bridge", "entity", "relation", "needed", "about",
}


def _guess_answer_type_label(query: str) -> str:
    normalized_query = normalize_structure_text(query)
    if not normalized_query:
        return "entity"
    if normalized_query.startswith("how many") or normalized_query.startswith("how much"):
        return "count"
    if normalized_query.startswith("when") or " year " in f" {normalized_query} " or " date " in f" {normalized_query} ":
        return "date"
    if normalized_query.startswith("where") or " place of birth " in f" {normalized_query} ":
        return "location"
    if normalized_query.startswith("who") or normalized_query.startswith("whom"):
        return "person"
    if normalized_query.startswith("which") or normalized_query.startswith("what"):
        if any(token in f" {normalized_query} " for token in (" city ", " country ", " county ", " river ", " gulf ", " state ")):
            return "location"
        if any(token in f" {normalized_query} " for token in (" year ", " date ", " month ", " day ")):
            return "date"
        if any(token in f" {normalized_query} " for token in (" person ", " actor ", " director ", " singer ", " player ", " wife ", " husband ", " father ", " mother ", " spouse ")):
            return "person"
    return "entity"


def _build_flat_query_tiers(query: str,
                            *,
                            query_entities: Sequence[str] | Set[str] | None = None,
                            max_facets: int = 3) -> Dict[str, object]:
    fallback_graph = build_query_dependency_graph(
        query,
        query_entities=query_entities,
        action_mode="action_swap_noisyor_flat",
        max_facets=max(int(max_facets), 1),
    )
    facets = [
        {
            "facet_id": str(node.get("id") or f"f{idx + 1}"),
            "facet_text": str(node.get("facet") or "").strip() or f"Answer the question: {query}",
            "facet_type": "flat",
            "tier_index": 0,
        }
        for idx, node in enumerate(list(fallback_graph.get("nodes") or [])[:max(int(max_facets), 1)])
    ]
    return {
        "mode": "flat_fallback",
        "answer_type": _guess_answer_type_label(query),
        "ordered_entities": list(fallback_graph.get("ordered_entities") or []),
        "tiers": [
            {
                "tier_id": "t1",
                "tier_index": 0,
                "tier_type": "flat",
                "facets": facets,
            },
        ],
    }


def build_query_tiers(query: str,
                      *,
                      query_entities: Sequence[str] | Set[str] | None = None,
                      max_facets: int = 3,
                      max_tiers: int = 2) -> Dict[str, object]:
    ordered_entities = _order_query_entities(query, query_entities, max_entities=max(max_facets, 1))
    normalized_query = normalize_structure_text(query)
    answer_type = _guess_answer_type_label(query)
    tier_limit = max(int(max_tiers), 1)
    facet_limit = max(int(max_facets), 1)

    if tier_limit < 2 or len(ordered_entities) < 2 or len(ordered_entities) > facet_limit:
        return _build_flat_query_tiers(query, query_entities=query_entities, max_facets=facet_limit)

    bridge_markers = (
        " of ", " by ", " after ", " before ", " where ", " when ", " whose ",
        " spouse ", " wife ", " husband ", " father ", " mother ", " director ",
        " publisher ", " performer ", " singer ", " actor ", " team ", " city ",
    )
    if not any(marker in f" {normalized_query} " for marker in bridge_markers):
        return _build_flat_query_tiers(query, query_entities=query_entities, max_facets=facet_limit)

    if len(ordered_entities) > 3:
        return _build_flat_query_tiers(query, query_entities=query_entities, max_facets=facet_limit)

    head_entity = ordered_entities[0]
    tail_entity = ordered_entities[1]
    tier1_facets: List[Dict[str, object]] = [
        {
            "facet_id": "t1_f1",
            "facet_text": f"Identify the bridge entity or relation involving {head_entity}. Question: {query}",
            "facet_type": "bridge_identification",
            "tier_index": 0,
        },
    ]
    if " and " in f" {normalized_query} " and len(ordered_entities) >= 3:
        tier1_facets.append({
            "facet_id": "t1_f2",
            "facet_text": f"Identify the parallel prerequisite involving {tail_entity}. Question: {query}",
            "facet_type": "parallel_prerequisite",
            "tier_index": 0,
        })

    target_entity = ordered_entities[-1]
    tier2_facets = [
        {
            "facet_id": "t2_f1",
            "facet_text": f"Find the {answer_type} answer detail about {target_entity}. Question: {query}",
            "facet_type": "target_property",
            "tier_index": 1,
        },
    ]

    tiers = [
        {
            "tier_id": "t1",
            "tier_index": 0,
            "tier_type": "heuristic_bridge",
            "facets": tier1_facets[:facet_limit],
        },
        {
            "tier_id": "t2",
            "tier_index": 1,
            "tier_type": "answer_slot",
            "facets": tier2_facets[:1],
        },
    ]
    total_facet_count = sum(len(tier.get("facets") or []) for tier in tiers)
    if total_facet_count <= 0 or total_facet_count > facet_limit:
        return _build_flat_query_tiers(query, query_entities=query_entities, max_facets=facet_limit)

    return {
        "mode": "heuristic_tiers",
        "answer_type": answer_type,
        "ordered_entities": ordered_entities,
        "tiers": tiers[:tier_limit],
    }


def _flatten_query_tiers(query_tiers: Mapping[str, object]) -> List[Dict[str, object]]:
    flattened: List[Dict[str, object]] = []
    for tier_index, tier in enumerate(list(query_tiers.get("tiers") or [])):
        for facet in list(tier.get("facets") or []):
            flattened.append({
                "facet_id": str(facet.get("facet_id") or f"t{tier_index + 1}_f{len(flattened) + 1}"),
                "facet_text": str(facet.get("facet_text") or "").strip(),
                "facet_type": str(facet.get("facet_type") or "unknown"),
                "tier_index": int(facet.get("tier_index", tier_index) or tier_index),
            })
    return flattened


def build_witness_units(doc_text: str,
                        *,
                        max_sentences: int = 8,
                        max_windows: int = 16,
                        include_full_doc: bool = True) -> List[Dict[str, str]]:
    units: List[Dict[str, str]] = []
    seen_units: Set[Tuple[str, str]] = set()

    def _append_unit(unit_type: str, unit_text: str) -> None:
        normalized_text = " ".join(str(unit_text or "").split()).strip()
        normalized_key = (str(unit_type), normalized_text)
        if not normalized_text or normalized_key in seen_units:
            return
        seen_units.add(normalized_key)
        units.append({
            "unit_type": str(unit_type),
            "unit_text": str(unit_text).strip(),
        })

    for window_text in build_title_prefixed_windows(
        doc_text,
        max_sentences=max_sentences,
        max_windows=max_windows,
    ):
        sentence_count = len(split_doc_body_sentences(window_text, max_sentences=max_sentences))
        _append_unit("title_plus_2sent" if sentence_count >= 2 else "title_plus_1sent", window_text)

    if include_full_doc:
        _append_unit("full_doc", format_doc_for_assemble_rerank(doc_text))
    if not units:
        _append_unit("full_doc", format_doc_for_assemble_rerank(doc_text))
    return units


_GAP_DATE_VALUE_PATTERN = re.compile(
    r"\b(?:\d{4}|\d{1,2}\s+[a-z]+\s+\d{4}|[a-z]+\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)


def _normalized_text_contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_structure_text(text)
    normalized_phrase = normalize_structure_text(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return normalized_phrase in normalized_text


def _matching_gap_entities_in_text(text: str,
                                   entities: Sequence[str] | Set[str] | None,
                                   *,
                                   exclude_entities: Sequence[str] | Set[str] | None = None) -> List[str]:
    normalized_excluded = normalize_entity_set(exclude_entities)
    matches: List[str] = []
    for entity in sorted(normalize_entity_set(entities)):
        if entity in normalized_excluded:
            continue
        if _normalized_text_contains_phrase(text, entity):
            matches.append(entity)
    return matches


def _gap_target_value_pattern_pass(slot: str, unit_text: str) -> bool:
    normalized_slot = normalize_structure_text(slot)
    normalized_text = normalize_structure_text(unit_text)
    token_set = normalized_token_set(unit_text)
    if not normalized_text:
        return False
    if normalized_slot in {"birth_date", "death_date"}:
        return bool(_GAP_DATE_VALUE_PATTERN.search(normalized_text)) or bool(token_set & _ANSWER_SCENT_MONTH_TOKENS)
    if normalized_slot == "birthplace":
        return (
            " born in " in f" {normalized_text} "
            or " from " in f" {normalized_text} "
            or bool(token_set & _ANSWER_SCENT_LOCATION_TOKENS)
        )
    if normalized_slot == "nationality":
        return (
            " nationality " in f" {normalized_text} "
            or " citizen " in f" {normalized_text} "
            or " country " in f" {normalized_text} "
            or bool(token_set & {"american", "british", "french", "german", "italian", "canadian"})
        )
    if normalized_slot == "education":
        return bool(token_set & {"school", "university", "college", "educated", "studied"})
    return False


def build_gap_evidence_units(doc_text: str,
                             *,
                             parent_doc_id: int | None = None,
                             parent_title: str | None = None,
                             doc_entities: Sequence[str] | Set[str] | None = None,
                             anchor_entities: Sequence[str] | Set[str] | None = None,
                             covered_entities: Sequence[str] | Set[str] | None = None,
                             slot_cues: Sequence[str] | Set[str] | None = None,
                             max_sentences: int = 8,
                             max_windows: int = 16) -> List[Dict[str, object]]:
    title = str(parent_title or extract_doc_title(doc_text) or "").strip()
    sentences = split_doc_body_sentences(doc_text, max_sentences=max_sentences)
    if not sentences:
        sentences = [" ".join(str(doc_text or "").split()).strip()]

    normalized_anchor_entities = normalize_entity_set(anchor_entities)
    normalized_covered_entities = normalize_entity_set(covered_entities)
    normalized_slot_cues = {
        normalize_structure_text(slot_cue)
        for slot_cue in (slot_cues or [])
        if normalize_structure_text(slot_cue)
    }
    units: List[Dict[str, object]] = []
    seen_units: Set[Tuple[str, str]] = set()

    def _append_unit(unit_type: str, unit_body: str, start_index: int, end_index: int) -> None:
        cleaned_body = " ".join(str(unit_body or "").split()).strip()
        if not cleaned_body:
            return
        unit_text = f"{title}\n{cleaned_body}".strip() if title else cleaned_body
        normalized_unit = normalize_structure_text(unit_text)
        unit_key = (str(unit_type), normalized_unit)
        if not normalized_unit or unit_key in seen_units:
            return
        seen_units.add(unit_key)
        matched_anchors = _matching_gap_entities_in_text(unit_text, normalized_anchor_entities)
        matched_non_anchor_entities = _matching_gap_entities_in_text(
            unit_text,
            doc_entities,
            exclude_entities=normalized_anchor_entities | normalized_covered_entities,
        )
        matched_slot_cues = [
            slot_cue
            for slot_cue in sorted(normalized_slot_cues)
            if _normalized_text_contains_phrase(unit_text, slot_cue)
        ]
        units.append({
            "unit_type": str(unit_type),
            "unit_text": unit_text,
            "sentence_span": [int(start_index + 1), int(end_index + 1)],
            "parent_doc_id": int(parent_doc_id) if parent_doc_id is not None else None,
            "parent_title": title,
            "contains_anchor": bool(matched_anchors),
            "contains_slot_cue": bool(matched_slot_cues),
            "contains_non_anchor_entity": bool(matched_non_anchor_entities),
            "matched_anchor_entities": list(matched_anchors),
            "matched_slot_cues": list(matched_slot_cues),
            "matched_non_anchor_entities": list(matched_non_anchor_entities),
        })

    for sentence_index, sentence in enumerate(sentences[:max(int(max_sentences), 1)]):
        _append_unit("title_plus_1sent", sentence, sentence_index, sentence_index)
        if len(units) >= int(max_windows):
            return units[:int(max_windows)]

    for sentence_index in range(max(len(sentences) - 1, 0)):
        _append_unit(
            "title_plus_2sent",
            f"{sentences[sentence_index]} {sentences[sentence_index + 1]}",
            sentence_index,
            sentence_index + 1,
        )
        if len(units) >= int(max_windows):
            return units[:int(max_windows)]

    return units[:int(max_windows)]


def _gap_unit_rank_tuple(payload: Mapping[str, object], gap_type: str) -> Tuple[int, int, int, int, float, int]:
    if str(gap_type or "") == "role_relation":
        return (
            1 if bool(payload.get("gap_unit_eligible", False)) else 0,
            1 if bool(payload.get("gap_unit_anchor_pass", False)) else 0,
            1 if bool(payload.get("gap_unit_slot_pass", False)) else 0,
            1 if bool(payload.get("gap_unit_non_anchor_pass", False)) else 0,
            float(payload.get("gap_unit_query_score_raw", 0.0) or 0.0),
            1 if str(payload.get("gap_unit_type", "") or "") == "title_plus_2sent" else 0,
        )
    return (
        1 if bool(payload.get("gap_unit_eligible", False)) else 0,
        1 if bool(payload.get("gap_unit_anchor_pass", False)) else 0,
        1 if bool(payload.get("gap_unit_slot_or_value_pass", False)) else 0,
        1 if bool(payload.get("gap_unit_value_pass", False)) else 0,
        float(payload.get("gap_unit_query_score_raw", 0.0) or 0.0),
        1 if str(payload.get("gap_unit_type", "") or "") == "title_plus_2sent" else 0,
    )


def rerank_gap_expand_units(scored_candidates: Sequence[Mapping[str, object]],
                            *,
                            pool_docs: Sequence[str] | None,
                            gap_state: Mapping[str, object],
                            covered_entities: Sequence[str] | Set[str] | None,
                            ce_reranker: Any = None,
                            max_sentences: int = 8,
                            max_windows: int = 16) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    gap_type = str(gap_state.get("gap_type", "abstain") or "abstain")
    micro_queries = [
        str(micro_query).strip()
        for micro_query in (gap_state.get("micro_queries", []) or [])
        if str(micro_query).strip()
    ]
    slot_name = str(gap_state.get("gap_slot", "") or "")
    slot_cues = list(gap_state.get("gap_slot_cues", []) or [])
    unit_rows: List[Dict[str, object]] = []

    for raw_row in scored_candidates:
        row = dict(raw_row)
        pool_position = int(row.get("pool_position", -1) or -1)
        if pool_docs is None or pool_position < 0 or pool_position >= len(pool_docs):
            continue
        doc_text = str(pool_docs[pool_position] or "")
        doc_id = int(row["doc_id"]) if row.get("doc_id") is not None else None
        doc_title = str(row.get("doc_title", "") or extract_doc_title(doc_text))
        doc_entities = normalize_entity_set(row.get("doc_entities", set()) or set())
        doc_units = build_gap_evidence_units(
            doc_text,
            parent_doc_id=doc_id,
            parent_title=doc_title,
            doc_entities=doc_entities,
            anchor_entities=gap_state.get("gap_anchors", []) or [],
            covered_entities=covered_entities,
            slot_cues=slot_cues,
            max_sentences=max_sentences,
            max_windows=max_windows,
        )
        for unit in doc_units:
            unit_rows.append({
                "pool_position": int(pool_position),
                "doc_id": doc_id,
                "doc_title": doc_title,
                "structure_score": float(row.get("structure_score", 0.0) or 0.0),
                "combined_score_raw": float(row.get("combined_score_raw", row.get("combined_score", 0.0)) or 0.0),
                "gap_type": gap_type,
                "gap_slot": slot_name,
                **dict(unit),
            })

    if ce_reranker is not None and unit_rows and micro_queries:
        pairs = [
            [micro_query, str(unit_row.get("unit_text", "") or "")]
            for unit_row in unit_rows
            for micro_query in micro_queries
        ]
        raw_scores = ce_reranker.compute_score(pairs) if pairs else []
        if isinstance(raw_scores, (int, float)):
            raw_scores = [float(raw_scores)]
        score_values = [_sigmoid_support_score(float(score)) for score in list(raw_scores)]
    else:
        score_values = []

    score_index = 0
    for unit_row in unit_rows:
        best_query = ""
        best_score = 0.0
        if micro_queries:
            for micro_query in micro_queries:
                if score_values:
                    query_score = float(score_values[score_index]) if score_index < len(score_values) else 0.0
                    score_index += 1
                else:
                    query_score = _compute_lexical_support_score(micro_query, str(unit_row.get("unit_text", "") or ""))
                if query_score > best_score:
                    best_score = float(query_score)
                    best_query = str(micro_query)

        anchor_pass = bool(unit_row.get("contains_anchor", False))
        slot_pass = bool(unit_row.get("contains_slot_cue", False))
        non_anchor_pass = bool(unit_row.get("contains_non_anchor_entity", False))
        value_pass = _gap_target_value_pattern_pass(slot_name, str(unit_row.get("unit_text", "") or ""))
        slot_or_value_pass = bool(slot_pass or value_pass)
        eligible = bool(anchor_pass and slot_or_value_pass and (non_anchor_pass if gap_type == "role_relation" else True))

        unit_row.update({
            "gap_unit_anchor_pass": bool(anchor_pass),
            "gap_unit_slot_pass": bool(slot_pass),
            "gap_unit_non_anchor_pass": bool(non_anchor_pass),
            "gap_unit_value_pass": bool(value_pass),
            "gap_unit_slot_or_value_pass": bool(slot_or_value_pass),
            "gap_unit_eligible": bool(eligible),
            "gap_unit_best_query": str(best_query),
            "gap_unit_query_score": round(float(best_score), 4),
            "gap_unit_query_score_raw": float(best_score),
            "gap_unit_query_rate": round(float(best_score), 4),
            "gap_unit_type": str(unit_row.get("unit_type", "") or ""),
            "gap_unit_text": str(unit_row.get("unit_text", "") or ""),
        })

    best_unit_by_position: Dict[int, Dict[str, object]] = {}
    for unit_row in unit_rows:
        pool_position = int(unit_row.get("pool_position", -1) or -1)
        current_best = best_unit_by_position.get(pool_position)
        if current_best is None or _gap_unit_rank_tuple(unit_row, gap_type) > _gap_unit_rank_tuple(current_best, gap_type):
            best_unit_by_position[pool_position] = dict(unit_row)

    collapsed_rows: List[Dict[str, object]] = []
    for raw_row in scored_candidates:
        row = dict(raw_row)
        pool_position = int(row.get("pool_position", -1) or -1)
        best_unit = dict(best_unit_by_position.get(pool_position, {}))
        collapsed_rows.append({
            **row,
            "gap_type": gap_type,
            "gap_mode": str(gap_state.get("gap_mode", "") or ""),
            "gap_filter_passed": bool(best_unit.get("gap_unit_eligible", False)),
            "gap_score": round(float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
            "gap_score_raw": float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0),
            "gap_gate_score": round(float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
            "gap_gate_score_raw": float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0),
            "gap_combined_score": round(float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
            "gap_combined_score_raw": float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0),
            "gap_unit_anchor_pass": bool(best_unit.get("gap_unit_anchor_pass", False)),
            "gap_unit_slot_pass": bool(best_unit.get("gap_unit_slot_pass", False)),
            "gap_unit_non_anchor_pass": bool(best_unit.get("gap_unit_non_anchor_pass", False)),
            "gap_unit_value_pass": bool(best_unit.get("gap_unit_value_pass", False)),
            "gap_unit_slot_or_value_pass": bool(best_unit.get("gap_unit_slot_or_value_pass", False)),
            "gap_unit_eligible": bool(best_unit.get("gap_unit_eligible", False)),
            "gap_unit_best_query": str(best_unit.get("gap_unit_best_query", "") or ""),
            "gap_unit_query_score": round(float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
            "gap_unit_query_score_raw": float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0),
            "gap_unit_query_rate": round(float(best_unit.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
            "gap_unit_type": str(best_unit.get("gap_unit_type", "") or ""),
            "gap_unit_text": str(best_unit.get("gap_unit_text", "") or ""),
            "gap_unit_sentence_span": list(best_unit.get("sentence_span", []) or []),
            "gap_unit_matched_anchor_entities": list(best_unit.get("matched_anchor_entities", []) or []),
            "gap_unit_matched_slot_cues": list(best_unit.get("matched_slot_cues", []) or []),
            "gap_unit_matched_non_anchor_entities": list(best_unit.get("matched_non_anchor_entities", []) or []),
        })

    collapsed_rows.sort(
        key=lambda item: (
            _gap_unit_rank_tuple(item, gap_type),
            float(item.get("combined_score_raw", 0.0) or 0.0),
            float(item.get("structure_score", 0.0) or 0.0),
            -int(item.get("pool_position", 0) or 0),
        ),
        reverse=True,
    )

    overall_top_unit = None
    for candidate_unit in unit_rows:
        if overall_top_unit is None or _gap_unit_rank_tuple(candidate_unit, gap_type) > _gap_unit_rank_tuple(overall_top_unit, gap_type):
            overall_top_unit = dict(candidate_unit)

    summary = {
        "gap_unit_count": int(len(unit_rows)),
        "gap_unit_eligible_count": int(sum(1 for unit_row in unit_rows if bool(unit_row.get("gap_unit_eligible", False)))),
        "gap_unit_score_source": "cross_encoder" if ce_reranker is not None else "lexical_fallback",
        "gap_unit_top_text": str((overall_top_unit or {}).get("unit_text", "") or ""),
        "gap_unit_top_parent_title": str((overall_top_unit or {}).get("parent_title", "") or ""),
        "gap_unit_top_parent_doc_id": (
            int((overall_top_unit or {}).get("parent_doc_id"))
            if (overall_top_unit or {}).get("parent_doc_id") is not None else None
        ),
        "gap_unit_top_anchor_pass": bool((overall_top_unit or {}).get("gap_unit_anchor_pass", False)),
        "gap_unit_top_slot_pass": bool((overall_top_unit or {}).get("gap_unit_slot_pass", False)),
        "gap_unit_top_non_anchor_pass": bool((overall_top_unit or {}).get("gap_unit_non_anchor_pass", False)),
        "gap_unit_top_query_score": round(float((overall_top_unit or {}).get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
        "gap_unit_preview": [
            {
                "preview_rank": int(rank + 1),
                "pool_position": int(unit_row.get("pool_position", -1) or -1),
                "parent_title": str(unit_row.get("parent_title", "") or ""),
                "unit_type": str(unit_row.get("unit_type", "") or ""),
                "sentence_span": list(unit_row.get("sentence_span", []) or []),
                "gap_unit_anchor_pass": bool(unit_row.get("gap_unit_anchor_pass", False)),
                "gap_unit_slot_pass": bool(unit_row.get("gap_unit_slot_pass", False)),
                "gap_unit_non_anchor_pass": bool(unit_row.get("gap_unit_non_anchor_pass", False)),
                "gap_unit_eligible": bool(unit_row.get("gap_unit_eligible", False)),
                "gap_unit_query_score": round(float(unit_row.get("gap_unit_query_score_raw", 0.0) or 0.0), 4),
                "unit_text": str(unit_row.get("unit_text", "") or ""),
            }
            for rank, unit_row in enumerate(unit_rows[:5])
        ],
    }
    return collapsed_rows, summary


def build_propose_verify_claims(query: str,
                                *,
                                query_entities: Sequence[str] | Set[str] | None = None,
                                max_claims: int = 2,
                                query_tiers_override: Mapping[str, object] | None = None) -> Dict[str, object]:
    claim_limit = max(int(max_claims), 1)
    query_tiers = dict(query_tiers_override or build_query_tiers(
        query,
        query_entities=query_entities,
        max_facets=claim_limit,
        max_tiers=2,
    ))
    claims: List[Dict[str, object]] = []
    for claim_rank, facet in enumerate(_flatten_query_tiers(query_tiers)[:claim_limit], start=1):
        claims.append({
            "claim_id": str(facet.get("facet_id") or f"c{claim_rank}"),
            "claim_text": str(facet.get("facet_text") or "").strip() or f"Answer the question: {query}",
            "claim_rank": int(claim_rank),
            "claim_type": str(facet.get("facet_type") or "unknown"),
            "tier_index": int(facet.get("tier_index", claim_rank - 1) or 0),
        })
    if not claims:
        claims = [{
            "claim_id": "c1",
            "claim_text": f"Answer the question: {query}",
            "claim_rank": 1,
            "claim_type": "fallback",
            "tier_index": 0,
        }]
        query_tiers = {
            "mode": "flat_fallback",
            "tiers": [
                {
                    "tier_id": "t1",
                    "tier_index": 0,
                    "tier_type": "flat",
                    "facets": [
                        {
                            "facet_id": "c1",
                            "facet_text": f"Answer the question: {query}",
                            "facet_type": "fallback",
                            "tier_index": 0,
                        }
                    ],
                }
            ],
        }
    return {
        "claim_mode": str(query_tiers.get("mode") or "flat_fallback"),
        "query_tiers": copy.deepcopy(list(query_tiers.get("tiers") or [])),
        "claims": claims,
    }


def compute_claim_ce_witness_details(claims: Sequence[Mapping[str, object]],
                                     *,
                                     doc_positions: Sequence[int],
                                     pool_docs: Sequence[str],
                                     ce_reranker: Any = None,
                                     max_sentences: int = 8,
                                     max_windows: int = 16,
                                     include_full_doc: bool = True) -> Dict[str, Dict[int, Dict[str, object]]]:
    claim_list = [dict(claim) for claim in claims]
    witness_units_by_doc = {
        int(pos): build_witness_units(
            pool_docs[int(pos)],
            max_sentences=max_sentences,
            max_windows=max_windows,
            include_full_doc=include_full_doc,
        )
        for pos in doc_positions
    }
    witness_details: Dict[str, Dict[int, Dict[str, object]]] = {
        str(claim.get("claim_id") or f"c{idx + 1}"): {
            int(pos): {
                "unit_type": None,
                "unit_text": "",
                "relevance_score": 0.0,
            }
            for pos in doc_positions
        }
        for idx, claim in enumerate(claim_list)
    }

    raw_scores: List[float] = []
    if ce_reranker is not None:
        pairs: List[List[str]] = []
        for claim in claim_list:
            claim_text = str(claim.get("claim_text") or "")
            for pos in doc_positions:
                for unit in witness_units_by_doc.get(int(pos), []):
                    pairs.append([claim_text, str(unit.get("unit_text") or "")])
        raw_scores = ce_reranker.compute_score(pairs) if pairs else []
        if isinstance(raw_scores, (int, float)):
            raw_scores = [float(raw_scores)]
        raw_scores = [
            _sigmoid_support_score(float(score))
            for score in list(raw_scores)
        ]

    score_index = 0
    for claim in claim_list:
        claim_id = str(claim.get("claim_id") or "")
        claim_text = str(claim.get("claim_text") or "")
        for pos in doc_positions:
            for unit in witness_units_by_doc.get(int(pos), []):
                unit_text = str(unit.get("unit_text") or "")
                relevance_score = (
                    float(raw_scores[score_index])
                    if ce_reranker is not None and score_index < len(raw_scores) else
                    _compute_lexical_support_score(claim_text, unit_text)
                )
                if ce_reranker is not None:
                    score_index += 1
                current_best = float(witness_details[claim_id][int(pos)].get("relevance_score", 0.0) or 0.0)
                if relevance_score > current_best:
                    witness_details[claim_id][int(pos)] = {
                        "unit_type": unit.get("unit_type"),
                        "unit_text": unit_text,
                        "relevance_score": round(float(relevance_score), 4),
                    }
    return witness_details


def _select_action_swap_proposal_job(action_jobs: Sequence[Mapping[str, object]]) -> Dict[str, object] | None:
    legal_jobs = filter_action_swap_legal_jobs(action_jobs, legality_mode="relaxed")
    if not legal_jobs:
        return None
    return max(
        legal_jobs,
        key=lambda job: (
            float(job.get("score_delta")) if job.get("score_delta") is not None else float("-inf"),
            float(job.get("candidate_assemble_score")) if job.get("candidate_assemble_score") is not None else float("-inf"),
            -int(job.get("candidate_pool_position", 0) or 0),
            -int(job.get("replace_pool_position", 0) or 0),
        ),
    )


def _run_claim_support_verifier_jobs(
    support_jobs: Sequence[Mapping[str, object]],
    *,
    verifier_bundle: SetwiseLateRerankJudgeBundle | None,
    max_doc_chars: int,
    claim_support_results: Mapping[Tuple[str, int], Mapping[str, object]] | None = None,
    max_completion_tokens: int = SETWISE_LLM_LATE_RERANK_REPAIR_MAX_COMPLETION_TOKENS,
) -> List[Dict[str, object]]:
    from run_swap_utility_judge import build_claim_support_messages, parse_claim_support_verifier_response

    cached_results = {
        (str(claim_id), int(doc_position)): dict(result)
        for (claim_id, doc_position), result in (claim_support_results or {}).items()
    }
    verified_rows: List[Dict[str, object]] = []
    for job in support_jobs:
        raw_doc_position = job.get("doc_position", -1)
        key = (
            str(job.get("claim_id") or ""),
            int(raw_doc_position) if raw_doc_position is not None else -1,
        )
        if key in cached_results:
            parsed = dict(cached_results[key])
            parsed.setdefault("parse_succeeded", parsed.get("verdict") is not None)
            parsed.setdefault("reason", None)
            parsed.setdefault("supported", str(parsed.get("verdict") or "").strip().lower() == "supported")
            verified_rows.append({
                **dict(job),
                "verifier_response": "",
                "verifier_trace": {
                    "judge_status": "cached",
                    "judge_error": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "finish_reason": "cached",
                    "backend": "cached",
                    "model": None,
                },
                "verifier_parsed": parsed,
            })
            continue

        if verifier_bundle is None or verifier_bundle.infer_fn is None:
            verified_rows.append({
                **dict(job),
                "verifier_response": "",
                "verifier_trace": {
                    "judge_status": "missing_verifier_bundle",
                    "judge_error": "missing_verifier_bundle",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "finish_reason": "missing_verifier_bundle",
                    "backend": None,
                    "model": None,
                },
                "verifier_parsed": {
                    "raw_response": "",
                    "parse_succeeded": False,
                    "parse_errors": ["missing_verifier_bundle"],
                    "verdict": None,
                    "reason": None,
                    "supported": False,
                },
            })
            continue

        response_text = ""
        metadata: Dict[str, object] = {}
        verifier_error = None
        try:
            response_text, metadata = verifier_bundle.infer_fn(
                messages=build_claim_support_messages(
                    question=str(job.get("question") or ""),
                    claim_text=str(job.get("claim_text") or ""),
                    doc_title=str(job.get("doc_title") or ""),
                    witness_text=str(job.get("witness_text") or ""),
                    witness_unit_type=str(job.get("witness_unit_type") or ""),
                    max_doc_chars=int(max_doc_chars),
                ),
                model=verifier_bundle.model_name,
                response_format=verifier_bundle.response_format,
                max_completion_tokens=int(max_completion_tokens),
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            verifier_error = str(exc)
            metadata = {
                "judge_status": "verifier_exception",
                "judge_error": verifier_error,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "exception",
                "backend": verifier_bundle.backend,
                "model": verifier_bundle.model_name,
            }
        verified_rows.append({
            **dict(job),
            "verifier_response": str(response_text or ""),
            "verifier_trace": {
                "judge_status": str(metadata.get("judge_status", "ok")),
                "judge_error": metadata.get("judge_error") or verifier_error,
                "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
                "finish_reason": str(metadata.get("finish_reason", "")),
                "backend": metadata.get("backend"),
                "model": metadata.get("model"),
                "response_text_source": metadata.get("response_text_source"),
            },
            "verifier_parsed": parse_claim_support_verifier_response(str(response_text or "")),
        })
    return verified_rows


def score_action_swap_propose_verify_jobs(
    action_jobs: Sequence[Mapping[str, object]],
    *,
    query: str,
    scaffold_positions: Sequence[int],
    pool_docs: Sequence[str],
    query_entities: Sequence[str] | Set[str] | None = None,
    ce_reranker: Any = None,
    verifier_bundle: SetwiseLateRerankJudgeBundle | None = None,
    max_doc_chars: int = 320,
    claim_support_results: Mapping[Tuple[str, int], Mapping[str, object]] | None = None,
    query_tiers_override: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    proposal_job = _select_action_swap_proposal_job(action_jobs)
    claim_bundle = build_propose_verify_claims(
        query,
        query_entities=query_entities,
        max_claims=2,
        query_tiers_override=query_tiers_override,
    )
    claims = [dict(claim) for claim in list(claim_bundle.get("claims") or [])]
    scaffold_list = [int(pos) for pos in scaffold_positions]
    legal_jobs = filter_action_swap_legal_jobs(action_jobs, legality_mode="relaxed")
    candidate_positions = sorted({
        int(job.get("candidate_pool_position", -1))
        for job in legal_jobs
        if job.get("candidate_pool_position") is not None and int(job.get("candidate_pool_position", -1)) >= 0
    })
    doc_positions = sorted(set(scaffold_list) | set(candidate_positions))
    witness_details = compute_claim_ce_witness_details(
        claims,
        doc_positions=doc_positions,
        pool_docs=pool_docs,
        ce_reranker=ce_reranker,
    )

    scaffold_support_jobs: List[Dict[str, object]] = []
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        claim_text = str(claim.get("claim_text") or "")
        for pos in scaffold_list:
            witness = dict(witness_details.get(claim_id, {}).get(int(pos), {}))
            scaffold_support_jobs.append({
                "verification_type": "claim_support",
                "question": str(query),
                "claim_id": claim_id,
                "claim_text": claim_text,
                "doc_position": int(pos),
                "doc_title": extract_doc_title(pool_docs[int(pos)]),
                "witness_text": str(witness.get("unit_text") or ""),
                "witness_unit_type": str(witness.get("unit_type") or ""),
                "witness_relevance_score": float(witness.get("relevance_score", 0.0) or 0.0),
            })
    scaffold_support_rows = _run_claim_support_verifier_jobs(
        scaffold_support_jobs,
        verifier_bundle=verifier_bundle,
        max_doc_chars=max_doc_chars,
        claim_support_results=claim_support_results,
    )
    scaffold_support_lookup: Dict[Tuple[str, int], Dict[str, object]] = {
        (
            str(row.get("claim_id") or ""),
            int(row.get("doc_position")) if row.get("doc_position") is not None else -1,
        ): dict(row)
        for row in scaffold_support_rows
    }

    claim_supports_before: List[Dict[str, object]] = []
    earliest_unsupported_claim: Dict[str, object] | None = None
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        supported_positions = [
            int(pos)
            for pos in scaffold_list
            if bool(((scaffold_support_lookup.get((claim_id, int(pos)), {}).get("verifier_parsed") or {}).get("supported")))
        ]
        claim_supports_before.append({
            "claim_id": claim_id,
            "claim_text": str(claim.get("claim_text") or ""),
            "claim_rank": int(claim.get("claim_rank", 0) or 0),
            "supported_positions": list(supported_positions),
            "supported_titles": [extract_doc_title(pool_docs[int(pos)]) for pos in supported_positions],
        })
        if earliest_unsupported_claim is None and not supported_positions:
            earliest_unsupported_claim = dict(claim)

    candidate_support_lookup: Dict[Tuple[str, int], Dict[str, object]] = {}
    if earliest_unsupported_claim is not None and candidate_positions:
        target_claim_id = str(earliest_unsupported_claim.get("claim_id") or "")
        target_claim_text = str(earliest_unsupported_claim.get("claim_text") or "")
        candidate_support_jobs: List[Dict[str, object]] = []
        for pos in candidate_positions:
            witness = dict(witness_details.get(target_claim_id, {}).get(int(pos), {}))
            candidate_support_jobs.append({
                "verification_type": "claim_support",
                "question": str(query),
                "claim_id": target_claim_id,
                "claim_text": target_claim_text,
                "doc_position": int(pos),
                "doc_title": extract_doc_title(pool_docs[int(pos)]),
                "witness_text": str(witness.get("unit_text") or ""),
                "witness_unit_type": str(witness.get("unit_type") or ""),
                "witness_relevance_score": float(witness.get("relevance_score", 0.0) or 0.0),
            })
        candidate_support_rows = _run_claim_support_verifier_jobs(
            candidate_support_jobs,
            verifier_bundle=verifier_bundle,
            max_doc_chars=max_doc_chars,
            claim_support_results=claim_support_results,
        )
        candidate_support_lookup = {
            (
                str(row.get("claim_id") or ""),
                int(row.get("doc_position")) if row.get("doc_position") is not None else -1,
            ): dict(row)
            for row in candidate_support_rows
        }

    earlier_claim_ids: List[str] = []
    if earliest_unsupported_claim is not None:
        earliest_rank = int(earliest_unsupported_claim.get("claim_rank", 0) or 0)
        earlier_claim_ids = [
            str(claim.get("claim_id") or "")
            for claim in claims
            if int(claim.get("claim_rank", 0) or 0) < earliest_rank
        ]

    scored_jobs: List[Dict[str, object]] = []
    for job in legal_jobs:
        raw_candidate_position = job.get("candidate_pool_position", -1)
        raw_replace_position = job.get("replace_pool_position", -1)
        candidate_position = int(raw_candidate_position) if raw_candidate_position is not None else -1
        replace_position = int(raw_replace_position) if raw_replace_position is not None else -1
        candidate_claim_row = (
            dict(candidate_support_lookup.get((str(earliest_unsupported_claim.get("claim_id") or ""), candidate_position), {}))
            if earliest_unsupported_claim is not None else
            {}
        )
        candidate_verifier_parsed = dict(candidate_claim_row.get("verifier_parsed") or {})
        candidate_supported = bool(candidate_verifier_parsed.get("supported"))
        unique_support_claim_ids: List[str] = []
        for claim_id in earlier_claim_ids:
            replace_row = dict(scaffold_support_lookup.get((claim_id, replace_position), {}))
            replace_supported = bool((replace_row.get("verifier_parsed") or {}).get("supported"))
            if not replace_supported:
                continue
            other_supported = any(
                bool((scaffold_support_lookup.get((claim_id, int(pos)), {}).get("verifier_parsed") or {}).get("supported"))
                for pos in scaffold_list
                if int(pos) != int(replace_position)
            )
            if not other_supported:
                unique_support_claim_ids.append(str(claim_id))

        if earliest_unsupported_claim is None:
            skip_reason = "no_unsupported_claim"
        elif not candidate_supported:
            skip_reason = "gain_verifier_reject"
        elif unique_support_claim_ids:
            skip_reason = "preservation_verifier_reject"
        else:
            skip_reason = None
        scored_jobs.append({
            **dict(job),
            "claim_mode": str(claim_bundle.get("claim_mode") or "flat_fallback"),
            "claims": copy.deepcopy(claims),
            "query_tiers": copy.deepcopy(list(claim_bundle.get("query_tiers") or [])),
            "query_tier_mode": str(claim_bundle.get("claim_mode") or "flat_fallback"),
            "earliest_unsupported_claim_id": (
                str(earliest_unsupported_claim.get("claim_id") or "")
                if earliest_unsupported_claim is not None else None
            ),
            "earliest_unsupported_claim_text": (
                str(earliest_unsupported_claim.get("claim_text") or "")
                if earliest_unsupported_claim is not None else None
            ),
            "claim_supports_before": copy.deepcopy(claim_supports_before),
            "candidate_best_witness": copy.deepcopy(dict(
                witness_details.get(
                    str(earliest_unsupported_claim.get("claim_id") or ""),
                    {},
                ).get(candidate_position, {})
            )) if earliest_unsupported_claim is not None else {},
            "gain_verifier_parsed": candidate_verifier_parsed,
            "gain_verifier_verdict": str(candidate_verifier_parsed.get("verdict") or "") or None,
            "gain_verifier_reason": str(candidate_verifier_parsed.get("reason") or "") or None,
            "preservation_unique_support_claim_ids": list(unique_support_claim_ids),
            "action_should_swap": skip_reason is None,
            "action_skip_reason": skip_reason,
        })

    proposal_key = None
    if proposal_job is not None:
        proposal_key = (
            int(proposal_job.get("candidate_pool_position")) if proposal_job.get("candidate_pool_position") is not None else -1,
            int(proposal_job.get("replace_pool_position")) if proposal_job.get("replace_pool_position") is not None else -1,
        )
    return {
        "claim_mode": str(claim_bundle.get("claim_mode") or "flat_fallback"),
        "claims": claims,
        "query_tiers": copy.deepcopy(list(claim_bundle.get("query_tiers") or [])),
        "proposal_key": proposal_key,
        "scaffold_support_rows": scaffold_support_rows,
        "claim_supports_before": claim_supports_before,
        "scored_jobs": scored_jobs,
    }


def select_action_swap_propose_verify(
    action_jobs: Sequence[Mapping[str, object]],
    *,
    query: str,
    scaffold_positions: Sequence[int],
    pool_docs: Sequence[str],
    query_entities: Sequence[str] | Set[str] | None = None,
    ce_reranker: Any = None,
    verifier_bundle: SetwiseLateRerankJudgeBundle | None = None,
    max_doc_chars: int = 320,
    claim_support_results: Mapping[Tuple[str, int], Mapping[str, object]] | None = None,
    query_tiers_override: Mapping[str, object] | None = None,
    action_mode: str = "action_swap_propose_verify",
) -> Dict[str, object]:
    before_positions = [int(pos) for pos in scaffold_positions]
    decision: Dict[str, object] = {
        "action_mode": str(action_mode),
        "action_legality_mode": "dedup_only",
        "action_executed": False,
        "action_type": "keep",
        "action_candidate_pool_position": None,
        "action_candidate_doc_id": None,
        "action_replace_pool_position": None,
        "action_replace_doc_id": None,
        "action_score_delta": None,
        "action_legal_action_count": 0,
        "action_total_jobs": int(len(action_jobs)),
        "action_skip_reason": "no_legal_actions",
        "action_gate_verdict": None,
        "action_gate_confidence": None,
        "action_gate_reason": None,
        "action_gate_source": "propose_verify_claim_support",
        "action_gate_score": None,
        "action_margin": None,
        "query_dependency_graph": {},
        "query_dependency_mode": None,
        "query_tiers": [],
        "query_tier_mode": "flat_fallback",
        "claims": [],
        "claim_mode": "flat_fallback",
        "proposal_action_present": False,
        "proposal_candidate_pool_position": None,
        "proposal_replace_pool_position": None,
        "earliest_unsupported_claim_id": None,
        "earliest_unsupported_claim_text": None,
        "gain_verifier_verdict": None,
        "gain_verifier_reason": None,
        "preservation_unique_support_claim_ids": [],
        "candidate_best_witness": {},
        "claim_supports_before": [],
        "final_front_positions_before_action": list(before_positions),
        "final_front_positions_after_action": list(before_positions),
    }
    legal_jobs = filter_action_swap_legal_jobs(action_jobs, legality_mode="relaxed")
    decision["action_legal_action_count"] = int(len(legal_jobs))
    if not legal_jobs:
        return decision

    score_bundle = score_action_swap_propose_verify_jobs(
        legal_jobs,
        query=query,
        scaffold_positions=before_positions,
        pool_docs=pool_docs,
        query_entities=query_entities,
        ce_reranker=ce_reranker,
        verifier_bundle=verifier_bundle,
        max_doc_chars=max_doc_chars,
        claim_support_results=claim_support_results,
        query_tiers_override=query_tiers_override,
    )
    decision["claims"] = copy.deepcopy(list(score_bundle.get("claims") or []))
    decision["claim_mode"] = str(score_bundle.get("claim_mode") or "flat_fallback")
    decision["query_tier_mode"] = str(score_bundle.get("claim_mode") or "flat_fallback")
    decision["query_tiers"] = copy.deepcopy(list(score_bundle.get("query_tiers") or []))
    decision["claim_supports_before"] = copy.deepcopy(list(score_bundle.get("claim_supports_before") or []))

    proposal_key = score_bundle.get("proposal_key")
    if proposal_key is None:
        return decision
    decision["proposal_action_present"] = True
    decision["proposal_candidate_pool_position"] = int(proposal_key[0])
    decision["proposal_replace_pool_position"] = int(proposal_key[1])

    scored_jobs = {
        (
            int(job.get("candidate_pool_position")) if job.get("candidate_pool_position") is not None else -1,
            int(job.get("replace_pool_position")) if job.get("replace_pool_position") is not None else -1,
        ): dict(job)
        for job in list(score_bundle.get("scored_jobs") or [])
    }
    proposal_job = dict(scored_jobs.get(tuple(proposal_key), {}))
    if not proposal_job:
        decision["action_skip_reason"] = "proposal_missing"
        return decision

    decision["earliest_unsupported_claim_id"] = proposal_job.get("earliest_unsupported_claim_id")
    decision["earliest_unsupported_claim_text"] = proposal_job.get("earliest_unsupported_claim_text")
    decision["gain_verifier_verdict"] = proposal_job.get("gain_verifier_verdict")
    decision["gain_verifier_reason"] = proposal_job.get("gain_verifier_reason")
    decision["candidate_best_witness"] = copy.deepcopy(dict(proposal_job.get("candidate_best_witness") or {}))
    decision["preservation_unique_support_claim_ids"] = list(proposal_job.get("preservation_unique_support_claim_ids") or [])
    decision["action_gate_verdict"] = (
        str(proposal_job.get("gain_verifier_verdict") or "") or None
        if proposal_job.get("action_skip_reason") != "preservation_verifier_reject" else
        "preservation_reject"
    )
    decision["action_gate_reason"] = (
        proposal_job.get("gain_verifier_reason")
        if proposal_job.get("action_skip_reason") != "preservation_verifier_reject" else
        f"replacee_is_unique_supporter:{','.join(decision['preservation_unique_support_claim_ids'])}"
    )

    if not bool(proposal_job.get("action_should_swap")):
        decision["action_skip_reason"] = str(proposal_job.get("action_skip_reason") or "verifier_reject")
        return decision

    final_positions = apply_single_slot_preserving_swap(
        scaffold_positions=before_positions,
        candidate_position=int(proposal_job["candidate_pool_position"]),
        replace_position=int(proposal_job["replace_pool_position"]),
    )
    decision.update({
        "action_executed": True,
        "action_type": "swap",
        "action_candidate_pool_position": int(proposal_job["candidate_pool_position"]),
        "action_candidate_doc_id": proposal_job.get("candidate_doc_id"),
        "action_replace_pool_position": int(proposal_job["replace_pool_position"]),
        "action_replace_doc_id": proposal_job.get("replace_doc_id"),
        "action_score_delta": round(float(proposal_job.get("score_delta", 0.0) or 0.0), 4),
        "action_skip_reason": None,
        "final_front_positions_after_action": list(final_positions),
    })
    return decision


def _focus_facet_tokens(facet_text: str) -> Set[str]:
    facet_tokens = _tokenize_support_text(facet_text)
    return {
        token
        for token in facet_tokens
        if token not in _ANSWER_SCENT_ENTITY_STOPWORDS
    }


def _compute_answer_type_cue_score(answer_type: str, unit_text: str) -> float:
    normalized_unit = normalize_structure_text(unit_text)
    if not normalized_unit:
        return 0.0
    unit_tokens = _tokenize_support_text(unit_text)
    if answer_type == "count":
        return 1.0 if re.search(r"\b\d+\b", unit_text) else 0.0
    if answer_type == "date":
        if re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", unit_text):
            return 1.0
        if unit_tokens & _ANSWER_SCENT_MONTH_TOKENS:
            return 0.9
        if any(token in unit_tokens for token in {"year", "date", "born", "died", "released", "signed", "founded"}):
            return 0.6
        return 0.0
    if answer_type == "location":
        if " born in " in f" {normalized_unit} " or " located in " in f" {normalized_unit} ":
            return 1.0
        if unit_tokens & _ANSWER_SCENT_LOCATION_TOKENS:
            return 0.75
        return 0.0
    if answer_type == "person":
        if unit_tokens & _ANSWER_SCENT_PERSON_TOKENS:
            return 0.75
        if re.search(r"\b[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)+\b", unit_text):
            return 0.6
        return 0.0
    if re.search(r"\b[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)+\b", unit_text):
        return 0.55
    return 0.2 if unit_tokens else 0.0


def compute_answer_scent_score(facet_text: str, unit_text: str) -> float:
    focus_tokens = _focus_facet_tokens(facet_text)
    unit_tokens = _tokenize_support_text(unit_text)
    overlap = float(len(focus_tokens & unit_tokens) / max(len(focus_tokens), 1)) if focus_tokens else 0.0
    answer_type = _guess_answer_type_label(facet_text)
    cue_score = _compute_answer_type_cue_score(answer_type, unit_text)
    return float(np.clip(0.45 * overlap + 0.55 * cue_score, 0.0, 1.0))


def compute_doc_facet_witness_matrix(query_tiers: Mapping[str, object],
                                     *,
                                     doc_positions: Sequence[int],
                                     pool_docs: Sequence[str],
                                     ce_reranker: Any = None,
                                     max_sentences: int = 8,
                                     max_windows: int = 16,
                                     include_full_doc: bool = True) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, Dict[str, object]]]]:
    facets = _flatten_query_tiers(query_tiers)
    witness_units_by_doc = {
        int(pos): build_witness_units(
            pool_docs[int(pos)],
            max_sentences=max_sentences,
            max_windows=max_windows,
            include_full_doc=include_full_doc,
        )
        for pos in doc_positions
    }
    witness_scores: Dict[str, Dict[int, float]] = {
        str(facet["facet_id"]): {int(pos): 0.0 for pos in doc_positions}
        for facet in facets
    }
    witness_details: Dict[str, Dict[int, Dict[str, object]]] = {
        str(facet["facet_id"]): {
            int(pos): {
                "unit_type": None,
                "unit_text": "",
                "witness_score": 0.0,
                "relevance_score": 0.0,
                "answerability_score": 0.0,
            }
            for pos in doc_positions
        }
        for facet in facets
    }

    relevance_scores_by_pair: List[float] = []
    pair_index: List[Tuple[str, int, int]] = []
    if ce_reranker is not None:
        pairs: List[List[str]] = []
        for facet in facets:
            facet_id = str(facet["facet_id"])
            facet_text = str(facet["facet_text"])
            for pos in doc_positions:
                for unit_index, unit in enumerate(witness_units_by_doc.get(int(pos), [])):
                    pairs.append([facet_text, str(unit.get("unit_text") or "")])
                    pair_index.append((facet_id, int(pos), int(unit_index)))
        raw_scores = ce_reranker.compute_score(pairs) if pairs else []
        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]
        relevance_scores_by_pair = [
            _sigmoid_support_score(float(score))
            for score in list(raw_scores)
        ]

    relevance_index = 0
    for facet in facets:
        facet_id = str(facet["facet_id"])
        facet_text = str(facet["facet_text"])
        for pos in doc_positions:
            for unit in witness_units_by_doc.get(int(pos), []):
                unit_text = str(unit.get("unit_text") or "")
                relevance_score = (
                    float(relevance_scores_by_pair[relevance_index])
                    if ce_reranker is not None and relevance_index < len(relevance_scores_by_pair) else
                    _compute_lexical_support_score(facet_text, unit_text)
                )
                if ce_reranker is not None:
                    relevance_index += 1
                answerability_score = compute_answer_scent_score(facet_text, unit_text)
                witness_score = float(min(relevance_score, answerability_score))
                if witness_score > float(witness_scores[facet_id].get(int(pos), 0.0) or 0.0):
                    witness_scores[facet_id][int(pos)] = witness_score
                    witness_details[facet_id][int(pos)] = {
                        "unit_type": unit.get("unit_type"),
                        "unit_text": unit_text,
                        "witness_score": round(float(witness_score), 4),
                        "relevance_score": round(float(relevance_score), 4),
                        "answerability_score": round(float(answerability_score), 4),
                    }
            if ce_reranker is None:
                continue
    return witness_scores, witness_details


def summarize_query_tier_supports(query_tiers: Mapping[str, object],
                                  *,
                                  positions: Sequence[int],
                                  witness_scores: Mapping[str, Mapping[int, float]],
                                  witness_details: Mapping[str, Mapping[int, Mapping[str, object]]],
                                  pool_docs: Sequence[str]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for facet in _flatten_query_tiers(query_tiers):
        facet_id = str(facet["facet_id"])
        ranked_positions = sorted(
            [int(pos) for pos in positions],
            key=lambda pos: (
                -float(witness_scores.get(facet_id, {}).get(int(pos), 0.0) or 0.0),
                int(pos),
            ),
        )
        top_position = ranked_positions[0] if ranked_positions else None
        second_position = ranked_positions[1] if len(ranked_positions) >= 2 else None
        summaries.append({
            "facet_id": facet_id,
            "facet_text": str(facet["facet_text"]),
            "facet_type": str(facet.get("facet_type") or "unknown"),
            "tier_index": int(facet.get("tier_index", 0) or 0),
            "best_support": round(float(witness_scores.get(facet_id, {}).get(int(top_position), 0.0) or 0.0), 4) if top_position is not None else 0.0,
            "backup_support": round(float(witness_scores.get(facet_id, {}).get(int(second_position), 0.0) or 0.0), 4) if second_position is not None else 0.0,
            "best_position": int(top_position) if top_position is not None else None,
            "backup_position": int(second_position) if second_position is not None else None,
            "best_title": extract_doc_title(pool_docs[int(top_position)]) if top_position is not None else None,
            "backup_title": extract_doc_title(pool_docs[int(second_position)]) if second_position is not None else None,
            "best_witness": dict(witness_details.get(facet_id, {}).get(int(top_position), {})) if top_position is not None else {},
        })
    return summaries


def _compute_facet_backup_values(scaffold_positions: Sequence[int],
                                 *,
                                 query_tiers: Mapping[str, object],
                                 witness_scores: Mapping[str, Mapping[int, float]]) -> Dict[str, Dict[str, object]]:
    backup_values: Dict[str, Dict[str, object]] = {}
    for facet in _flatten_query_tiers(query_tiers):
        facet_id = str(facet["facet_id"])
        ranked = sorted(
            [
                (int(pos), float(witness_scores.get(facet_id, {}).get(int(pos), 0.0) or 0.0))
                for pos in scaffold_positions
            ],
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        best_position = ranked[0][0] if ranked else None
        best_support = ranked[0][1] if ranked else 0.0
        second_position = ranked[1][0] if len(ranked) >= 2 else None
        second_support = ranked[1][1] if len(ranked) >= 2 else 0.0
        backup_values[facet_id] = {
            "best_position": best_position,
            "best_support": float(best_support),
            "backup_position": second_position,
            "backup_support": float(second_support),
        }
    return backup_values


def _compute_facet_incumbent_loss(facet_id: str,
                                  incumbent_position: int,
                                  *,
                                  scaffold_positions: Sequence[int],
                                  witness_scores: Mapping[str, Mapping[int, float]]) -> float:
    ranked = sorted(
        [
            (int(pos), float(witness_scores.get(facet_id, {}).get(int(pos), 0.0) or 0.0))
            for pos in scaffold_positions
        ],
        key=lambda item: (-float(item[1]), int(item[0])),
    )
    if not ranked:
        return 0.0
    best_support = float(ranked[0][1])
    incumbent_support = float(witness_scores.get(facet_id, {}).get(int(incumbent_position), 0.0) or 0.0)
    if incumbent_support + 1e-9 < best_support:
        return 0.0
    best_other = max(
        [
            float(score)
            for pos, score in ranked
            if int(pos) != int(incumbent_position)
        ] or [0.0]
    )
    return float(max(best_support - best_other, 0.0))


def score_action_swap_tiered_witness_jobs(action_jobs: Sequence[Mapping[str, object]],
                                          *,
                                          query: str,
                                          scaffold_positions: Sequence[int],
                                          pool_docs: Sequence[str],
                                          query_entities: Sequence[str] | Set[str] | None = None,
                                          ce_reranker: Any = None,
                                          query_tiers_override: Mapping[str, object] | None = None) -> Dict[str, object]:
    query_tiers = dict(query_tiers_override or build_query_tiers(query, query_entities=query_entities))
    scaffold_list = [int(pos) for pos in scaffold_positions]
    candidate_positions = sorted({
        int(job.get("candidate_pool_position", -1))
        for job in action_jobs
        if job.get("candidate_pool_position") is not None and int(job.get("candidate_pool_position", -1)) >= 0
    })
    doc_positions = sorted(set(scaffold_list) | set(candidate_positions))
    witness_scores, witness_details = compute_doc_facet_witness_matrix(
        query_tiers,
        doc_positions=doc_positions,
        pool_docs=pool_docs,
        ce_reranker=ce_reranker,
    )
    scaffold_backups = _compute_facet_backup_values(
        scaffold_list,
        query_tiers=query_tiers,
        witness_scores=witness_scores,
    )
    before_supports = summarize_query_tier_supports(
        query_tiers,
        positions=scaffold_list,
        witness_scores=witness_scores,
        witness_details=witness_details,
        pool_docs=pool_docs,
    )

    facets = _flatten_query_tiers(query_tiers)
    candidate_best_by_position: Dict[int, Dict[str, object]] = {}
    for candidate_position in candidate_positions:
        chosen_facet: Dict[str, object] | None = None
        for tier in list(query_tiers.get("tiers") or []):
            tier_index = int(tier.get("tier_index", 0) or 0)
            tier_facets = [
                facet for facet in facets
                if int(facet.get("tier_index", 0) or 0) == tier_index
            ]
            tier_rows: List[Dict[str, object]] = []
            for facet in tier_facets:
                facet_id = str(facet["facet_id"])
                best_support = float(scaffold_backups.get(facet_id, {}).get("best_support", 0.0) or 0.0)
                candidate_support = float(witness_scores.get(facet_id, {}).get(int(candidate_position), 0.0) or 0.0)
                gain = float(candidate_support - best_support)
                tier_rows.append({
                    "facet_id": facet_id,
                    "facet_text": str(facet["facet_text"]),
                    "facet_type": str(facet.get("facet_type") or "unknown"),
                    "tier_index": tier_index,
                    "candidate_support": candidate_support,
                    "best_support": best_support,
                    "backup_support": float(scaffold_backups.get(facet_id, {}).get("backup_support", 0.0) or 0.0),
                    "gain": gain,
                    "best_witness": dict(witness_details.get(facet_id, {}).get(int(candidate_position), {})),
                })
            positive_rows = [row for row in tier_rows if float(row["gain"]) > 0.0]
            if not positive_rows:
                continue
            chosen_facet = max(
                positive_rows,
                key=lambda row: (
                    float(row["gain"]),
                    float(row["candidate_support"]),
                    str(row["facet_id"]),
                ),
            )
            break
        if chosen_facet is None:
            candidate_best_by_position[int(candidate_position)] = {
                "candidate_pool_position": int(candidate_position),
                "bottleneck_tier_index": None,
                "target_facet_id": None,
                "target_facet_text": None,
                "target_facet_type": None,
                "target_candidate_gain": 0.0,
                "target_candidate_support": 0.0,
                "target_best_support": 0.0,
                "target_backup_support": 0.0,
                "candidate_best_witness": {},
            }
            continue
        candidate_best_by_position[int(candidate_position)] = {
            "candidate_pool_position": int(candidate_position),
            "bottleneck_tier_index": int(chosen_facet["tier_index"]),
            "target_facet_id": str(chosen_facet["facet_id"]),
            "target_facet_text": str(chosen_facet["facet_text"]),
            "target_facet_type": str(chosen_facet["facet_type"]),
            "target_candidate_gain": round(float(chosen_facet["gain"]), 4),
            "target_candidate_support": round(float(chosen_facet["candidate_support"]), 4),
            "target_best_support": round(float(chosen_facet["best_support"]), 4),
            "target_backup_support": round(float(chosen_facet["backup_support"]), 4),
            "candidate_best_witness": dict(chosen_facet["best_witness"]),
        }

    scored_jobs: List[Dict[str, object]] = []
    for job in action_jobs:
        candidate_position = int(job["candidate_pool_position"])
        replace_position = int(job["replace_pool_position"])
        candidate_summary = dict(candidate_best_by_position.get(candidate_position, {}))
        target_facet_id = candidate_summary.get("target_facet_id")
        bottleneck_tier_index = candidate_summary.get("bottleneck_tier_index")
        earlier_tier_losses = 0.0
        same_tier_collateral = 0.0
        target_loss = 0.0
        replacee_rows: List[Dict[str, object]] = []
        if target_facet_id is not None and bottleneck_tier_index is not None:
            for facet in facets:
                facet_id = str(facet["facet_id"])
                facet_loss = _compute_facet_incumbent_loss(
                    facet_id,
                    replace_position,
                    scaffold_positions=scaffold_list,
                    witness_scores=witness_scores,
                )
                replacee_rows.append({
                    "facet_id": facet_id,
                    "tier_index": int(facet.get("tier_index", 0) or 0),
                    "loss": round(float(facet_loss), 4),
                })
                if int(facet.get("tier_index", 0) or 0) < int(bottleneck_tier_index):
                    earlier_tier_losses += float(facet_loss)
                elif facet_id == str(target_facet_id):
                    target_loss += float(facet_loss)
                elif int(facet.get("tier_index", 0) or 0) == int(bottleneck_tier_index):
                    same_tier_collateral += float(facet_loss)
        gain = float(candidate_summary.get("target_candidate_gain", 0.0) or 0.0)
        should_swap = (
            bool(target_facet_id)
            and gain > float(earlier_tier_losses + target_loss)
        )
        after_positions = list(job.get("swapped_positions") or apply_single_slot_preserving_swap(
            scaffold_list,
            candidate_position,
            replace_position,
        ))
        after_supports = summarize_query_tier_supports(
            query_tiers,
            positions=after_positions,
            witness_scores=witness_scores,
            witness_details=witness_details,
            pool_docs=pool_docs,
        )
        scored_jobs.append({
            **dict(job),
            "query_tier_mode": str(query_tiers.get("mode") or "flat_fallback"),
            "query_tiers": copy.deepcopy(list(query_tiers.get("tiers") or [])),
            "bottleneck_tier_index": bottleneck_tier_index,
            "target_facet_id": target_facet_id,
            "target_facet_text": candidate_summary.get("target_facet_text"),
            "target_facet_type": candidate_summary.get("target_facet_type"),
            "target_candidate_gain": round(float(gain), 4),
            "target_candidate_support": candidate_summary.get("target_candidate_support"),
            "target_best_support": candidate_summary.get("target_best_support"),
            "target_backup_support": candidate_summary.get("target_backup_support"),
            "candidate_best_witness": copy.deepcopy(dict(candidate_summary.get("candidate_best_witness") or {})),
            "replacee_loss_earlier_tiers": round(float(earlier_tier_losses), 4),
            "replacee_loss_same_tier": round(float(same_tier_collateral), 4),
            "replacee_loss_target_facet": round(float(target_loss), 4),
            "replacee_facet_losses": replacee_rows,
            "swap_gain_vs_loss": round(float(gain - (earlier_tier_losses + target_loss)), 4),
            "action_should_swap": bool(should_swap),
            "facet_supports_before": copy.deepcopy(before_supports),
            "facet_supports_after": copy.deepcopy(after_supports),
        })

    return {
        "query_tiers": query_tiers,
        "query_tier_mode": str(query_tiers.get("mode") or "flat_fallback"),
        "scored_jobs": scored_jobs,
        "facet_supports_before": before_supports,
        "witness_scores": witness_scores,
        "witness_details": witness_details,
    }


def select_action_swap_tiered_witness(action_jobs: Sequence[Mapping[str, object]],
                                      *,
                                      query: str,
                                      scaffold_positions: Sequence[int],
                                      pool_docs: Sequence[str],
                                      query_entities: Sequence[str] | Set[str] | None = None,
                                      ce_reranker: Any = None,
                                      action_mode: str = "action_swap_tiered_witness",
                                      query_tiers_override: Mapping[str, object] | None = None) -> Dict[str, object]:
    before_positions = [int(pos) for pos in scaffold_positions]
    decision: Dict[str, object] = {
        "action_mode": str(action_mode),
        "action_legality_mode": "dedup_only",
        "action_executed": False,
        "action_type": "keep",
        "action_candidate_pool_position": None,
        "action_candidate_doc_id": None,
        "action_replace_pool_position": None,
        "action_replace_doc_id": None,
        "action_score_delta": None,
        "action_legal_action_count": 0,
        "action_total_jobs": int(len(action_jobs)),
        "action_skip_reason": "no_legal_actions",
        "action_gate_source": "tier_aware_bottleneck_witness",
        "action_gate_score": None,
        "action_margin": None,
        "query_dependency_graph": {},
        "query_dependency_mode": None,
        "query_tiers": [],
        "query_tier_mode": "flat_fallback",
        "facet_supports_before": [],
        "facet_supports_after": [],
        "incumbent_attributions": [],
        "executed_action_delta": None,
        "bottleneck_tier_index": None,
        "target_facet_id": None,
        "target_facet_text": None,
        "target_candidate_gain": None,
        "replacee_loss_earlier_tiers": None,
        "replacee_loss_same_tier": None,
        "replacee_loss_target_facet": None,
        "swap_gain_vs_loss": None,
        "candidate_best_witness": {},
        "final_front_positions_before_action": list(before_positions),
        "final_front_positions_after_action": list(before_positions),
    }

    legal_jobs = filter_action_swap_legal_jobs(action_jobs, legality_mode="relaxed")
    decision["action_legal_action_count"] = int(len(legal_jobs))
    if not legal_jobs:
        return decision

    score_bundle = score_action_swap_tiered_witness_jobs(
        legal_jobs,
        query=query,
        scaffold_positions=before_positions,
        pool_docs=pool_docs,
        query_entities=query_entities,
        ce_reranker=ce_reranker,
        query_tiers_override=query_tiers_override,
    )
    decision["query_tiers"] = copy.deepcopy(list((score_bundle.get("query_tiers") or {}).get("tiers") or []))
    decision["query_tier_mode"] = str(score_bundle.get("query_tier_mode") or "flat_fallback")
    decision["facet_supports_before"] = copy.deepcopy(list(score_bundle.get("facet_supports_before") or []))

    actionable_jobs = [
        dict(job)
        for job in list(score_bundle.get("scored_jobs") or [])
        if bool(job.get("action_should_swap"))
    ]
    if not actionable_jobs:
        decision["action_skip_reason"] = "no_positive_witness_gain"
        return decision

    best_job = min(
        actionable_jobs,
        key=lambda row: (
            int(row.get("bottleneck_tier_index")) if row.get("bottleneck_tier_index") is not None else 10**6,
            -float(row.get("target_candidate_gain", 0.0) or 0.0),
            float(row.get("replacee_loss_earlier_tiers", 0.0) or 0.0),
            float(row.get("replacee_loss_target_facet", 0.0) or 0.0),
            -float(row.get("candidate_assemble_score", float("-inf")) or float("-inf")),
            int(row.get("candidate_pool_position", 0) or 0),
            int(row.get("replace_pool_position", 0) or 0),
        ),
    )
    decision.update({
        "action_executed": True,
        "action_type": "swap",
        "action_candidate_pool_position": int(best_job["candidate_pool_position"]),
        "action_candidate_doc_id": best_job.get("candidate_doc_id"),
        "action_replace_pool_position": int(best_job["replace_pool_position"]),
        "action_replace_doc_id": best_job.get("replace_doc_id"),
        "action_score_delta": round(float(best_job.get("score_delta", 0.0) or 0.0), 4),
        "action_skip_reason": None,
        "action_gate_score": round(float(best_job.get("swap_gain_vs_loss", 0.0) or 0.0), 4),
        "executed_action_delta": round(float(best_job.get("swap_gain_vs_loss", 0.0) or 0.0), 4),
        "bottleneck_tier_index": best_job.get("bottleneck_tier_index"),
        "target_facet_id": best_job.get("target_facet_id"),
        "target_facet_text": best_job.get("target_facet_text"),
        "target_candidate_gain": best_job.get("target_candidate_gain"),
        "replacee_loss_earlier_tiers": best_job.get("replacee_loss_earlier_tiers"),
        "replacee_loss_same_tier": best_job.get("replacee_loss_same_tier"),
        "replacee_loss_target_facet": best_job.get("replacee_loss_target_facet"),
        "swap_gain_vs_loss": best_job.get("swap_gain_vs_loss"),
        "candidate_best_witness": copy.deepcopy(dict(best_job.get("candidate_best_witness") or {})),
        "facet_supports_after": copy.deepcopy(list(best_job.get("facet_supports_after") or [])),
        "final_front_positions_after_action": list(best_job.get("swapped_positions") or apply_single_slot_preserving_swap(
            before_positions,
            int(best_job["candidate_pool_position"]),
            int(best_job["replace_pool_position"]),
        )),
    })
    decision["incumbent_attributions"] = [
        {
            "pool_position": int(job.get("replace_pool_position", -1)),
            "replacee_loss_earlier_tiers": job.get("replacee_loss_earlier_tiers"),
            "replacee_loss_same_tier": job.get("replacee_loss_same_tier"),
            "replacee_loss_target_facet": job.get("replacee_loss_target_facet"),
        }
        for job in list(score_bundle.get("scored_jobs") or [])
        if int(job.get("candidate_pool_position", -1)) == int(best_job.get("candidate_pool_position", -1))
    ]
    return decision


def compute_truncated_noisyor_support(doc_support_by_position: Mapping[int, float],
                                      set_positions: Sequence[int],
                                      *,
                                      top_n: int = 2) -> float:
    support_values = sorted(
        [
            float(np.clip(float(doc_support_by_position.get(int(pos), 0.0) or 0.0), 0.0, 1.0))
            for pos in set_positions
        ],
        reverse=True,
    )[:max(int(top_n), 1)]
    if not support_values:
        return 0.0
    residual = 1.0
    for support_value in support_values:
        residual *= 1.0 - support_value
    return float(1.0 - residual)


def _collect_query_dependency_ancestors(query_graph: Mapping[str, object]) -> Dict[str, List[str]]:
    parent_map = {
        str(node.get("id")): [str(parent) for parent in (node.get("parents") or []) if str(parent)]
        for node in query_graph.get("nodes", []) or []
    }
    ancestor_map: Dict[str, List[str]] = {}

    def _dfs(node_id: str, trail: Set[str]) -> List[str]:
        if node_id in ancestor_map:
            return list(ancestor_map[node_id])
        ancestors: List[str] = []
        for parent_id in parent_map.get(node_id, []):
            if parent_id in trail:
                continue
            if parent_id not in ancestors:
                ancestors.append(parent_id)
            for ancestor_id in _dfs(parent_id, trail | {parent_id}):
                if ancestor_id not in ancestors:
                    ancestors.append(ancestor_id)
        ancestor_map[node_id] = list(ancestors)
        return list(ancestors)

    for node_id in parent_map:
        _dfs(node_id, {node_id})
    return ancestor_map


def compute_dependency_aware_set_utility(set_positions: Sequence[int],
                                         *,
                                         query_graph: Mapping[str, object],
                                         doc_support_matrix: Mapping[str, Mapping[int, float]],
                                         pool_docs: Sequence[str] | None = None) -> Tuple[float, List[Dict[str, object]]]:
    ancestor_map = _collect_query_dependency_ancestors(query_graph)
    facet_support_rows: List[Dict[str, object]] = []
    truncated_support_by_facet: Dict[str, float] = {}

    for node in query_graph.get("nodes", []) or []:
        facet_id = str(node.get("id") or "")
        support_value = compute_truncated_noisyor_support(
            doc_support_matrix.get(facet_id, {}),
            set_positions,
            top_n=2,
        )
        truncated_support_by_facet[facet_id] = float(support_value)

    total_utility = 0.0
    for node in query_graph.get("nodes", []) or []:
        facet_id = str(node.get("id") or "")
        parent_ids = [str(parent) for parent in (node.get("parents") or []) if str(parent)]
        ancestor_ids = list(ancestor_map.get(facet_id, []))
        effective_support = float(truncated_support_by_facet.get(facet_id, 0.0))
        for ancestor_id in ancestor_ids:
            effective_support *= float(truncated_support_by_facet.get(ancestor_id, 0.0))
        total_utility += float(effective_support)

        ranked_positions = sorted(
            [int(pos) for pos in set_positions],
            key=lambda pos: (
                -float(doc_support_matrix.get(facet_id, {}).get(int(pos), 0.0) or 0.0),
                int(pos),
            ),
        )
        top_positions = ranked_positions[:2]
        facet_support_rows.append({
            "facet_id": facet_id,
            "facet": str(node.get("facet") or ""),
            "parents": list(parent_ids),
            "ancestors": list(ancestor_ids),
            "support": round(float(truncated_support_by_facet.get(facet_id, 0.0)), 4),
            "effective_support": round(float(effective_support), 4),
            "top_support_positions": list(top_positions),
            "top_support_titles": [
                extract_doc_title(pool_docs[int(pos)])
                for pos in top_positions
            ] if pool_docs is not None else [],
        })

    return float(total_utility), facet_support_rows


def compute_incumbent_attributions(scaffold_positions: Sequence[int],
                                   *,
                                   baseline_utility: float,
                                   query_graph: Mapping[str, object],
                                   doc_support_matrix: Mapping[str, Mapping[int, float]],
                                   pool_docs: Sequence[str]) -> List[Dict[str, object]]:
    attribution_rows: List[Dict[str, object]] = []
    scaffold_list = [int(pos) for pos in scaffold_positions]
    for incumbent_position in scaffold_list:
        reduced_positions = [
            int(pos)
            for pos in scaffold_list
            if int(pos) != int(incumbent_position)
        ]
        reduced_utility, _ = compute_dependency_aware_set_utility(
            reduced_positions,
            query_graph=query_graph,
            doc_support_matrix=doc_support_matrix,
            pool_docs=pool_docs,
        )
        attribution_rows.append({
            "pool_position": int(incumbent_position),
            "title": extract_doc_title(pool_docs[int(incumbent_position)]),
            "leave_one_out_delta": round(float(baseline_utility - reduced_utility), 4),
        })
    return attribution_rows


def compute_doc_facet_support_matrix(query_graph: Mapping[str, object],
                                     *,
                                     doc_positions: Sequence[int],
                                     pool_docs: Sequence[str],
                                     ce_reranker: Any = None,
                                     max_sentences: int = 8,
                                     max_windows: int = 16) -> Dict[str, Dict[int, float]]:
    doc_windows: Dict[int, List[str]] = {
        int(pos): build_title_prefixed_windows(
            pool_docs[int(pos)],
            max_sentences=max_sentences,
            max_windows=max_windows,
        )
        for pos in doc_positions
    }
    support_matrix: Dict[str, Dict[int, float]] = {
        str(node.get("id")): {
            int(pos): 0.0
            for pos in doc_positions
        }
        for node in query_graph.get("nodes", []) or []
    }

    if ce_reranker is None:
        for node in query_graph.get("nodes", []) or []:
            facet_id = str(node.get("id") or "")
            facet_text = str(node.get("facet") or "")
            for pos in doc_positions:
                best_score = 0.0
                for window_text in doc_windows.get(int(pos), []) or []:
                    best_score = max(best_score, _compute_lexical_support_score(facet_text, window_text))
                support_matrix[facet_id][int(pos)] = float(np.clip(best_score, 0.0, 1.0))
        return support_matrix

    pairs: List[List[str]] = []
    pair_index: List[Tuple[str, int]] = []
    for node in query_graph.get("nodes", []) or []:
        facet_id = str(node.get("id") or "")
        facet_text = str(node.get("facet") or "")
        for pos in doc_positions:
            windows = doc_windows.get(int(pos), []) or []
            if not windows:
                windows = [pool_docs[int(pos)]]
            for window_text in windows:
                pairs.append([facet_text, window_text])
                pair_index.append((facet_id, int(pos)))

    raw_scores = ce_reranker.compute_score(pairs) if pairs else []
    if isinstance(raw_scores, (int, float)):
        raw_scores = [raw_scores]
    for (facet_id, pos), raw_score in zip(pair_index, list(raw_scores)):
        support_matrix[facet_id][int(pos)] = max(
            float(support_matrix[facet_id].get(int(pos), 0.0) or 0.0),
            _sigmoid_support_score(float(raw_score)),
        )
    return support_matrix


def select_action_swap_noisyor(action_jobs: Sequence[Mapping[str, object]],
                               *,
                               query: str,
                               scaffold_positions: Sequence[int],
                               pool_docs: Sequence[str],
                               query_entities: Sequence[str] | Set[str] | None = None,
                               ce_reranker: Any = None,
                               action_mode: str = "action_swap_noisyor_dep",
                               action_margin: float = DEFAULT_ACTION_SWAP_NOISYOR_MARGIN) -> Dict[str, object]:
    before_positions = [int(pos) for pos in scaffold_positions]
    decision: Dict[str, object] = {
        "action_mode": str(action_mode),
        "action_legality_mode": "dedup_only",
        "action_executed": False,
        "action_type": "keep",
        "action_candidate_pool_position": None,
        "action_candidate_doc_id": None,
        "action_replace_pool_position": None,
        "action_replace_doc_id": None,
        "action_score_delta": None,
        "action_legal_action_count": 0,
        "action_total_jobs": int(len(action_jobs)),
        "action_skip_reason": "no_legal_actions",
        "action_gate_source": "dependency_aware_noisyor",
        "action_gate_score": None,
        "action_margin": round(float(action_margin), 4),
        "query_dependency_graph": {},
        "query_dependency_mode": "flat_fallback",
        "facet_supports_before": [],
        "facet_supports_after": [],
        "incumbent_attributions": [],
        "executed_action_delta": None,
        "final_front_positions_before_action": list(before_positions),
        "final_front_positions_after_action": list(before_positions),
    }

    legal_jobs = filter_action_swap_legal_jobs(action_jobs, legality_mode="relaxed")
    decision["action_legal_action_count"] = int(len(legal_jobs))
    if not legal_jobs:
        return decision

    query_graph = build_query_dependency_graph(
        query,
        query_entities=query_entities,
        action_mode=action_mode,
    )
    decision["query_dependency_graph"] = dict(query_graph)
    decision["query_dependency_mode"] = str(query_graph.get("mode", "flat_fallback"))

    candidate_positions = {
        int(job.get("candidate_pool_position", -1))
        for job in legal_jobs
        if job.get("candidate_pool_position") is not None and int(job.get("candidate_pool_position", -1)) >= 0
    }
    doc_positions = sorted(set(before_positions) | set(candidate_positions))
    doc_support_matrix = compute_doc_facet_support_matrix(
        query_graph,
        doc_positions=doc_positions,
        pool_docs=pool_docs,
        ce_reranker=ce_reranker,
    )
    baseline_utility, facet_supports_before = compute_dependency_aware_set_utility(
        before_positions,
        query_graph=query_graph,
        doc_support_matrix=doc_support_matrix,
        pool_docs=pool_docs,
    )
    decision["facet_supports_before"] = list(facet_supports_before)
    decision["incumbent_attributions"] = compute_incumbent_attributions(
        before_positions,
        baseline_utility=baseline_utility,
        query_graph=query_graph,
        doc_support_matrix=doc_support_matrix,
        pool_docs=pool_docs,
    )

    scored_jobs: List[Dict[str, object]] = []
    for job in legal_jobs:
        swapped_positions = [
            int(pos)
            for pos in (
                job.get("swapped_positions")
                or apply_single_slot_preserving_swap(
                    before_positions,
                    int(job["candidate_pool_position"]),
                    int(job["replace_pool_position"]),
                )
            )
        ]
        swapped_utility, facet_supports_after = compute_dependency_aware_set_utility(
            swapped_positions,
            query_graph=query_graph,
            doc_support_matrix=doc_support_matrix,
            pool_docs=pool_docs,
        )
        scored_jobs.append({
            **dict(job),
            "swapped_positions": list(swapped_positions),
            "utility_before": float(baseline_utility),
            "utility_after": float(swapped_utility),
            "utility_delta": float(swapped_utility - baseline_utility),
            "facet_supports_after": list(facet_supports_after),
        })

    best_job = max(
        scored_jobs,
        key=lambda row: (
            float(row.get("utility_delta", float("-inf"))),
            float(row.get("score_delta")) if row.get("score_delta") is not None else float("-inf"),
            float(row.get("candidate_assemble_score")) if row.get("candidate_assemble_score") is not None else float("-inf"),
            -int(row.get("candidate_pool_position", 0) or 0),
            -int(row.get("replace_pool_position", 0) or 0),
        ),
    )
    best_delta = float(best_job.get("utility_delta", 0.0) or 0.0)
    decision["action_gate_score"] = round(float(best_delta), 4)
    if best_delta <= float(action_margin):
        decision["action_skip_reason"] = "margin_not_met"
        decision["facet_supports_after"] = list(facet_supports_before)
        return decision

    decision.update({
        "action_executed": True,
        "action_type": "swap",
        "action_candidate_pool_position": int(best_job["candidate_pool_position"]),
        "action_candidate_doc_id": best_job.get("candidate_doc_id"),
        "action_replace_pool_position": int(best_job["replace_pool_position"]),
        "action_replace_doc_id": best_job.get("replace_doc_id"),
        "action_score_delta": round(float(best_job.get("score_delta", 0.0) or 0.0), 4),
        "action_skip_reason": None,
        "facet_supports_after": list(best_job.get("facet_supports_after") or []),
        "executed_action_delta": round(float(best_delta), 4),
        "final_front_positions_after_action": list(best_job.get("swapped_positions") or before_positions),
    })
    return decision


def _build_undirected_component_map(nodes: Set[str],
                                    edges: Set[Tuple[str, str]]) -> Dict[str, int]:
    adjacency: Dict[str, Set[str]] = {
        str(node): set()
        for node in nodes
        if str(node).strip()
    }
    for src, tgt in edges:
        normalized_src = normalize_structure_text(src)
        normalized_tgt = normalize_structure_text(tgt)
        if not normalized_src or not normalized_tgt:
            continue
        if normalized_src not in adjacency or normalized_tgt not in adjacency:
            continue
        adjacency[normalized_src].add(normalized_tgt)
        adjacency[normalized_tgt].add(normalized_src)

    component_by_node: Dict[str, int] = {}
    next_component_id = 0
    for node in sorted(adjacency):
        if node in component_by_node:
            continue
        next_component_id += 1
        stack = [node]
        component_by_node[node] = int(next_component_id)
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor in component_by_node:
                    continue
                component_by_node[neighbor] = int(next_component_id)
                stack.append(neighbor)
    return component_by_node


def _build_candidate_component_pairs(candidate_edges: Set[Tuple[str, str]],
                                     scaffold_component_by_node: Mapping[str, int]) -> Set[Tuple[int, int]]:
    adjacency: Dict[str, Set[str]] = {}
    for src, tgt in candidate_edges:
        normalized_src = normalize_structure_text(src)
        normalized_tgt = normalize_structure_text(tgt)
        if not normalized_src or not normalized_tgt:
            continue
        adjacency.setdefault(normalized_src, set()).add(normalized_tgt)
        adjacency.setdefault(normalized_tgt, set()).add(normalized_src)

    bridged_component_pairs: Set[Tuple[int, int]] = set()
    visited: Set[str] = set()
    for node in sorted(adjacency):
        if node in visited:
            continue
        stack = [node]
        component_nodes: Set[str] = set()
        visited.add(node)
        while stack:
            current = stack.pop()
            component_nodes.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        touched_components = sorted({
            int(scaffold_component_by_node[current])
            for current in component_nodes
            if current in scaffold_component_by_node
        })
        if len(touched_components) < 2:
            continue
        for comp_a, comp_b in combinations(touched_components, 2):
            bridged_component_pairs.add((int(comp_a), int(comp_b)))
    return bridged_component_pairs


def assemble_ce_local_repair(query: str,
                             pool_docs: Sequence[str],
                             pool_doc_ids: Sequence[int | None],
                             pool_doc_scores: Sequence[float],
                             candidate_positions: Sequence[int],
                             qa_top_k: int,
                             query_entities: Set[str],
                             seed_entities: Set[str],
                             doc_idx_to_entities: Dict[int, Set[str]],
                             doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                             hipporag: HippoRAG,
                             ce_reranker: Any,
                             position_sources: Dict[int, str] | None = None,
                             replace_bottom_n: int = 2) -> Tuple[List[int], Dict[str, object]]:
    normalized_positions = _normalize_candidate_positions_for_assemble(candidate_positions, pool_docs)
    normalized_query_entities = normalize_entity_set(query_entities) or normalize_entity_set(seed_entities)
    default_trace: Dict[str, object] = {
        "assemble_mode": "ce_local_repair",
        "repair_mode": "ce_local_repair",
        "candidate_pool_positions": list(normalized_positions),
        "candidate_titles": [extract_doc_title(pool_docs[pos]) for pos in normalized_positions],
        "ranked_pool_positions": list(normalized_positions),
        "ranked_titles": [extract_doc_title(pool_docs[pos]) for pos in normalized_positions],
        "ranking_rows": [],
        "score_field": "cross_encoder_score",
        "fallback_reason": "",
        "scaffold_positions": [],
        "scaffold_titles": [],
        "scaffold_ce_sum": 0.0,
        "missing_query_anchor_entities": [],
        "missing_query_anchor_count": 0,
        "scaffold_component_count": 0,
        "repair_candidate_rows": [],
        "best_repair_swap": {},
        "repair_applied": False,
        "final_positions_before_repair": [],
        "final_positions_after_repair": [],
    }
    if not normalized_positions or qa_top_k <= 0:
        default_trace["fallback_reason"] = "empty_candidate_pool"
        return [], default_trace
    if ce_reranker is None:
        raise ValueError("ce_local_repair assemble_mode requires a loaded ce_reranker")

    ce_rows, score_field, fallback_reason = _compute_assemble_score_rows(
        query=query,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=pool_doc_scores,
        candidate_positions=normalized_positions,
        assemble_mode="cross_encoder",
        hipporag=hipporag,
        ce_reranker=ce_reranker,
        position_sources=position_sources,
    )
    sorted_ce_rows = _sort_assemble_score_rows(ce_rows)
    ce_rows_by_position = {
        int(row["pool_position"]): dict(row)
        for row in sorted_ce_rows
    }
    rank_by_position = {
        int(row["pool_position"]): int(rank + 1)
        for rank, row in enumerate(sorted_ce_rows)
    }
    baseline_prefix_positions = [
        int(row["pool_position"])
        for row in sorted_ce_rows
        if str(row.get("source", "")) == "baseline_prefix"
    ]
    scaffold_positions = list(baseline_prefix_positions[:max(int(qa_top_k), 0)])
    if not scaffold_positions:
        scaffold_positions = [
            int(row["pool_position"])
            for row in sorted_ce_rows[:max(int(qa_top_k), 0)]
        ]
    appended_positions = [
        int(pos) for pos in normalized_positions
        if str((position_sources or {}).get(int(pos), "")).startswith("append_")
    ]

    doc_entities_by_position: Dict[int, Set[str]] = {}
    doc_edges_by_position: Dict[int, Set[Tuple[str, str]]] = {}
    for pos in normalized_positions:
        doc_id = pool_doc_ids[int(pos)]
        doc_entities_by_position[int(pos)] = _normalize_structure_entities_for_doc(doc_id, doc_idx_to_entities)
        doc_edges_by_position[int(pos)] = _normalize_structure_edges_for_doc(doc_id, doc_idx_to_edges)

    scaffold_entities: Set[str] = set()
    scaffold_edges: Set[Tuple[str, str]] = set()
    for pos in scaffold_positions:
        scaffold_entities.update(doc_entities_by_position.get(int(pos), set()))
        scaffold_edges.update(doc_edges_by_position.get(int(pos), set()))
    scaffold_node_universe = set(normalized_query_entities) | set(scaffold_entities)
    filtered_scaffold_edges = {
        (src, tgt) for src, tgt in scaffold_edges
        if src in scaffold_node_universe and tgt in scaffold_node_universe
    }
    scaffold_component_by_node = _build_undirected_component_map(
        scaffold_node_universe,
        filtered_scaffold_edges,
    )
    missing_query_anchor_entities = sorted(set(normalized_query_entities) - set(scaffold_entities))
    scaffold_ce_sum = float(sum(
        float(ce_rows_by_position[int(pos)]["assemble_score"])
        for pos in scaffold_positions
        if int(pos) in ce_rows_by_position
    ))

    replace_candidates = list(scaffold_positions[-max(int(replace_bottom_n), 0):])
    repair_candidate_rows: List[Dict[str, object]] = []
    best_swap: Dict[str, object] | None = None
    best_key: Tuple[Any, ...] | None = None

    for appended_pos in appended_positions:
        candidate_entities = doc_entities_by_position.get(int(appended_pos), set())
        candidate_edges = doc_edges_by_position.get(int(appended_pos), set())
        anchor_gain_entities = sorted(candidate_entities & set(missing_query_anchor_entities))
        bridged_component_pairs = sorted(
            list(_build_candidate_component_pairs(candidate_edges, scaffold_component_by_node))
        )
        connector_gain_count = int(len(bridged_component_pairs))
        anchor_gain_count = int(len(anchor_gain_entities))
        candidate_ce_row = ce_rows_by_position.get(int(appended_pos), {})
        candidate_ce_score = float(candidate_ce_row.get("assemble_score", 0.0) or 0.0)
        candidate_ce_rank = int(rank_by_position.get(int(appended_pos), 0))

        for replace_pos in replace_candidates:
            replaced_ce_row = ce_rows_by_position.get(int(replace_pos), {})
            replaced_ce_score = float(replaced_ce_row.get("assemble_score", 0.0) or 0.0)
            ce_sum_after_swap = float(scaffold_ce_sum - replaced_ce_score + candidate_ce_score)
            ce_drop_vs_scaffold = float(scaffold_ce_sum - ce_sum_after_swap)
            row = {
                "appended_pool_position": int(appended_pos),
                "candidate_title": extract_doc_title(pool_docs[int(appended_pos)]),
                "candidate_ce_rank": int(candidate_ce_rank),
                "candidate_ce_score": round(float(candidate_ce_score), 4),
                "replace_pool_position": int(replace_pos),
                "replace_title": extract_doc_title(pool_docs[int(replace_pos)]),
                "connector_gain_count": int(connector_gain_count),
                "anchor_gain_count": int(anchor_gain_count),
                "ce_sum_after_swap": round(float(ce_sum_after_swap), 4),
                "ce_drop_vs_scaffold": round(float(ce_drop_vs_scaffold), 4),
                "accepted_by_req_gain": bool(connector_gain_count > 0 or anchor_gain_count > 0),
            }
            repair_candidate_rows.append(row)
            row_key = (
                int(connector_gain_count),
                int(anchor_gain_count),
                float(ce_sum_after_swap),
                float(candidate_ce_score),
                -int(appended_pos),
                -int(replace_pos),
            )
            if not row["accepted_by_req_gain"]:
                continue
            if best_key is None or row_key > best_key:
                best_key = row_key
                best_swap = {
                    **row,
                    "replaced_incumbent_ce_rank": int(rank_by_position.get(int(replace_pos), 0)),
                    "component_pairs": [
                        [int(comp_a), int(comp_b)]
                        for comp_a, comp_b in bridged_component_pairs
                    ],
                    "anchor_gain_entities": list(anchor_gain_entities),
                }

    final_front_positions = list(scaffold_positions)
    if best_swap is not None:
        replace_pos = int(best_swap["replace_pool_position"])
        appended_pos = int(best_swap["appended_pool_position"])
        final_front_positions = [
            appended_pos if int(pos) == replace_pos else int(pos)
            for pos in scaffold_positions
        ]

    seen_final_positions: Set[int] = set()
    ranked_positions: List[int] = []
    for pos in final_front_positions:
        if int(pos) in seen_final_positions:
            continue
        ranked_positions.append(int(pos))
        seen_final_positions.add(int(pos))
    for row in sorted_ce_rows:
        pos = int(row["pool_position"])
        if pos in seen_final_positions:
            continue
        ranked_positions.append(pos)
        seen_final_positions.add(pos)

    trace = {
        "assemble_mode": "ce_local_repair",
        "repair_mode": "ce_local_repair",
        "candidate_pool_positions": list(normalized_positions),
        "candidate_titles": [extract_doc_title(pool_docs[pos]) for pos in normalized_positions],
        "ranked_pool_positions": list(ranked_positions),
        "ranked_titles": [extract_doc_title(pool_docs[pos]) for pos in ranked_positions],
        "ranking_rows": [
            {
                "rank": int(rank + 1),
                "pool_position": int(row["pool_position"]),
                "doc_id": row["doc_id"],
                "title": str(row["title"]),
                "source": str(row["source"]),
                "base_score": round(float(row["base_score"]), 4),
                "assemble_score": None if not np.isfinite(float(row["assemble_score"])) else round(float(row["assemble_score"]), 4),
            }
            for rank, row in enumerate(sorted_ce_rows)
        ],
        "score_field": str(score_field),
        "fallback_reason": str(fallback_reason),
        "scaffold_positions": list(scaffold_positions),
        "scaffold_titles": [extract_doc_title(pool_docs[pos]) for pos in scaffold_positions],
        "scaffold_ce_sum": round(float(scaffold_ce_sum), 4),
        "missing_query_anchor_entities": list(missing_query_anchor_entities),
        "missing_query_anchor_count": int(len(missing_query_anchor_entities)),
        "scaffold_component_count": int(len(set(scaffold_component_by_node.values()))),
        "repair_candidate_rows": list(repair_candidate_rows),
        "best_repair_swap": dict(best_swap or {}),
        "repair_applied": bool(best_swap is not None),
        "final_positions_before_repair": list(scaffold_positions),
        "final_positions_after_repair": list(final_front_positions),
    }
    return list(ranked_positions), trace


def _build_coverage_atom_maps(candidate_positions: Sequence[int],
                              atom_source_positions: Sequence[int],
                              normalized_atom_source: str,
                              pool_doc_ids: Sequence[int | None],
                              seed_entities: Set[str],
                              doc_idx_to_entities: Dict[int, Set[str]],
                              doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]]) -> Tuple[
                                  Set[str],
                                  Set[Tuple[str, str]],
                                  Set[str],
                                  Dict[int, Set[str]],
                                  Dict[int, Set[Tuple[str, str]]],
                                  Dict[int, Set[str]],
                              ]:
    a_q = normalize_entity_set(seed_entities)
    a_e: Set[Tuple[str, str]] = set()
    a_b: Set[str] = set()
    base_entity_set: Set[str] = set()
    doc_entities_by_position: Dict[int, Set[str]] = {}
    doc_edges_by_position: Dict[int, Set[Tuple[str, str]]] = {}

    for pos in candidate_positions:
        doc_id = pool_doc_ids[pos]
        doc_entities_by_position[int(pos)] = _normalize_structure_entities_for_doc(doc_id, doc_idx_to_entities)
        doc_edges_by_position[int(pos)] = _normalize_structure_edges_for_doc(doc_id, doc_idx_to_edges)

    for pos in atom_source_positions:
        base_entity_set.update(doc_entities_by_position.get(int(pos), set()))

    if normalized_atom_source == "baseline_anchored":
        for pos in candidate_positions:
            anchored_edges = {
                edge for edge in doc_edges_by_position.get(int(pos), set())
                if edge[0] in base_entity_set or edge[1] in base_entity_set
            }
            a_e.update(anchored_edges)
            for src, tgt in anchored_edges:
                a_b.add(src)
                a_b.add(tgt)
    else:
        for pos in atom_source_positions:
            a_b.update(doc_entities_by_position.get(int(pos), set()))
            a_e.update(doc_edges_by_position.get(int(pos), set()))
    a_b -= a_q

    doc_covers_q: Dict[int, Set[str]] = {}
    doc_covers_e: Dict[int, Set[Tuple[str, str]]] = {}
    doc_covers_b: Dict[int, Set[str]] = {}
    for pos in candidate_positions:
        normalized_entities = doc_entities_by_position.get(int(pos), set())
        normalized_edges = doc_edges_by_position.get(int(pos), set())
        doc_covers_q[int(pos)] = normalized_entities & a_q
        doc_covers_e[int(pos)] = normalized_edges & a_e
        if normalized_atom_source == "baseline_anchored":
            anchored_entities = {
                entity
                for src, tgt in doc_covers_e[int(pos)]
                for entity in (src, tgt)
            }
            doc_covers_b[int(pos)] = (anchored_entities - a_q) & a_b
        else:
            doc_covers_b[int(pos)] = (normalized_entities - a_q) & a_b

    return a_q, a_e, a_b, doc_covers_q, doc_covers_e, doc_covers_b


def _numeric_margin(values: Sequence[float], ndigits: int = 4) -> float | None:
    distinct_values = sorted({round(float(value), 8) for value in values}, reverse=True)
    if len(distinct_values) < 2:
        return None
    return round(float(distinct_values[0] - distinct_values[1]), ndigits)


def _safe_mean(values: Sequence[float]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    return round(float(sum(numeric) / len(numeric)), 4)


def _compute_coverage_decision_layer(valid_subset_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    default_payload = {
        "effective_decision_layer": "fallback",
        "max_covQ_subset_count": 0,
        "max_covQ_max_covE_subset_count": 0,
        "max_covQ_max_covE_max_ce_subset_count": 0,
        "covQ_margin": None,
        "covE_margin_within_max_covQ": None,
        "ce_margin_within_max_covQ_covE": None,
    }
    if not valid_subset_rows:
        return default_payload

    max_covq = max(int(row["cov_q"]) for row in valid_subset_rows)
    max_covq_rows = [row for row in valid_subset_rows if int(row["cov_q"]) == max_covq]
    max_cove = max(int(row["cov_e"]) for row in max_covq_rows)
    max_covq_cove_rows = [row for row in max_covq_rows if int(row["cov_e"]) == max_cove]
    max_ce_sum = max(float(row["ce_sum"]) for row in max_covq_cove_rows)
    max_covq_cove_ce_rows = [
        row for row in max_covq_cove_rows
        if abs(float(row["ce_sum"]) - max_ce_sum) <= 1e-9
    ]

    if len(max_covq_rows) == 1:
        effective_decision_layer = "CovQ"
    elif len(max_covq_cove_rows) == 1:
        effective_decision_layer = "CovE"
    elif len(max_covq_cove_ce_rows) == 1:
        effective_decision_layer = "CE"
    else:
        effective_decision_layer = "TIE_AFTER_CE"

    return {
        "effective_decision_layer": str(effective_decision_layer),
        "max_covQ_subset_count": int(len(max_covq_rows)),
        "max_covQ_max_covE_subset_count": int(len(max_covq_cove_rows)),
        "max_covQ_max_covE_max_ce_subset_count": int(len(max_covq_cove_ce_rows)),
        "covQ_margin": _numeric_margin([float(row["cov_q"]) for row in valid_subset_rows]),
        "covE_margin_within_max_covQ": _numeric_margin([float(row["cov_e"]) for row in max_covq_rows]),
        "ce_margin_within_max_covQ_covE": _numeric_margin([float(row["ce_sum"]) for row in max_covq_cove_rows]),
    }


def _compute_coverage_score_tuple(normalized_variant: str,
                                  cov_q: int,
                                  cov_e: int,
                                  cov_b: int,
                                  ce_sum: float,
                                  base_sum: float,
                                  subset: Sequence[int]) -> Tuple[Any, ...]:
    position_tiebreak = tuple(-int(pos) for pos in sorted(subset))
    if normalized_variant == "qeb_ce":
        return (cov_q, cov_e, cov_b, ce_sum, base_sum, position_tiebreak)
    return (cov_q, cov_e, ce_sum, base_sum, position_tiebreak)


def _compute_ce_only_score_tuple(ce_sum: float,
                                 base_sum: float,
                                 subset: Sequence[int]) -> Tuple[Any, ...]:
    position_tiebreak = tuple(-int(pos) for pos in sorted(subset))
    return (ce_sum, base_sum, position_tiebreak)


def _run_coverage_exact_search(candidate_positions: Sequence[int],
                               qa_top_k: int,
                               pool_docs: Sequence[str],
                               sorted_ce_rows: Sequence[Mapping[str, Any]],
                               ce_rows_by_position: Mapping[int, Mapping[str, Any]],
                               ce_scores: Mapping[int, float],
                               pool_doc_scores: Sequence[float],
                               doc_covers_q: Mapping[int, Set[str]],
                               doc_covers_e: Mapping[int, Set[Tuple[str, str]]],
                               doc_covers_b: Mapping[int, Set[str]],
                               normalized_variant: str) -> Dict[str, Any]:
    normalized_positions = [int(pos) for pos in candidate_positions]

    def subset_stats(subset: Sequence[int]) -> Tuple[int, int, int, float, float]:
        covered_q: Set[str] = set()
        covered_e: Set[Tuple[str, str]] = set()
        covered_b: Set[str] = set()
        ce_sum = 0.0
        base_sum = 0.0
        for pos in subset:
            covered_q.update(doc_covers_q.get(int(pos), set()))
            covered_e.update(doc_covers_e.get(int(pos), set()))
            covered_b.update(doc_covers_b.get(int(pos), set()))
            ce_sum += float(ce_scores.get(int(pos), 0.0))
            base_sum += float(pool_doc_scores[int(pos)]) if int(pos) < len(pool_doc_scores) else 0.0
        return len(covered_q), len(covered_e), len(covered_b), float(ce_sum), float(base_sum)

    total_subsets_enumerated = 0
    total_valid_subsets = 0
    total_invalid_subsets_dedup = 0
    best_subset: List[int] = []
    best_score_tuple: Tuple[Any, ...] | None = None
    best_covq = 0
    best_cove = 0
    best_covb = 0
    best_ce_sum = 0.0
    best_ce_subset: List[int] = []
    best_ce_tuple: Tuple[Any, ...] | None = None

    covq_values: Set[float] = set()
    cove_values_within_max_covq: Set[float] = set()
    ce_values_within_max_covq_cove: Set[float] = set()
    max_covq: int | None = None
    max_covq_subset_count = 0
    max_cove_within_max_covq: int | None = None
    max_covq_max_cove_subset_count = 0
    max_ce_within_max_covq_cove: float | None = None
    max_covq_max_cove_max_ce_subset_count = 0

    for subset_tuple in combinations(normalized_positions, int(qa_top_k)):
        total_subsets_enumerated += 1
        if not _is_valid_coverage_subset(subset_tuple, pool_docs):
            total_invalid_subsets_dedup += 1
            continue
        total_valid_subsets += 1
        cov_q, cov_e, cov_b, ce_sum, base_sum = subset_stats(subset_tuple)
        covq_values.add(float(cov_q))

        if max_covq is None or cov_q > max_covq:
            max_covq = int(cov_q)
            max_covq_subset_count = 1
            cove_values_within_max_covq = {float(cov_e)}
            max_cove_within_max_covq = int(cov_e)
            max_covq_max_cove_subset_count = 1
            ce_values_within_max_covq_cove = {float(ce_sum)}
            max_ce_within_max_covq_cove = float(ce_sum)
            max_covq_max_cove_max_ce_subset_count = 1
        elif cov_q == max_covq:
            max_covq_subset_count += 1
            cove_values_within_max_covq.add(float(cov_e))
            if max_cove_within_max_covq is None or cov_e > max_cove_within_max_covq:
                max_cove_within_max_covq = int(cov_e)
                max_covq_max_cove_subset_count = 1
                ce_values_within_max_covq_cove = {float(ce_sum)}
                max_ce_within_max_covq_cove = float(ce_sum)
                max_covq_max_cove_max_ce_subset_count = 1
            elif cov_e == max_cove_within_max_covq:
                max_covq_max_cove_subset_count += 1
                ce_values_within_max_covq_cove.add(float(ce_sum))
                if max_ce_within_max_covq_cove is None or ce_sum > max_ce_within_max_covq_cove + 1e-9:
                    max_ce_within_max_covq_cove = float(ce_sum)
                    max_covq_max_cove_max_ce_subset_count = 1
                elif abs(ce_sum - max_ce_within_max_covq_cove) <= 1e-9:
                    max_covq_max_cove_max_ce_subset_count += 1

        score_tuple = _compute_coverage_score_tuple(
            normalized_variant=normalized_variant,
            cov_q=int(cov_q),
            cov_e=int(cov_e),
            cov_b=int(cov_b),
            ce_sum=float(ce_sum),
            base_sum=float(base_sum),
            subset=subset_tuple,
        )
        ce_tuple = _compute_ce_only_score_tuple(
            ce_sum=float(ce_sum),
            base_sum=float(base_sum),
            subset=subset_tuple,
        )
        if best_score_tuple is None or score_tuple > best_score_tuple:
            best_score_tuple = score_tuple
            best_subset = list(subset_tuple)
            best_covq = int(cov_q)
            best_cove = int(cov_e)
            best_covb = int(cov_b)
            best_ce_sum = float(ce_sum)
        if best_ce_tuple is None or ce_tuple > best_ce_tuple:
            best_ce_tuple = ce_tuple
            best_ce_subset = list(subset_tuple)

    ranked_positions: List[int] = []
    fallback_reason = ""
    effective_decision_layer = "fallback"
    if best_subset:
        ranked_positions = [
            int(row["pool_position"])
            for row in _sort_assemble_score_rows([
                ce_rows_by_position[int(pos)] for pos in best_subset
            ])
        ]
        if max_covq_subset_count == 1:
            effective_decision_layer = "CovQ"
        elif max_covq_max_cove_subset_count == 1:
            effective_decision_layer = "CovE"
        elif max_covq_max_cove_max_ce_subset_count == 1:
            effective_decision_layer = "CE"
        else:
            effective_decision_layer = "TIE_AFTER_CE"
    else:
        fallback_reason = "no_valid_subset_under_title_dedup"
        deduped_positions: List[int] = []
        seen_title_keys: Set[str] = set()
        for row in sorted_ce_rows:
            pos = int(row["pool_position"])
            if pos not in normalized_positions:
                continue
            title_key = normalize_structure_text(str(row["title"]))
            if title_key and title_key in seen_title_keys:
                continue
            deduped_positions.append(pos)
            if title_key:
                seen_title_keys.add(title_key)
            if len(deduped_positions) >= int(qa_top_k):
                break
        if len(deduped_positions) < int(qa_top_k):
            for row in sorted_ce_rows:
                pos = int(row["pool_position"])
                if pos not in normalized_positions or pos in deduped_positions:
                    continue
                deduped_positions.append(pos)
                if len(deduped_positions) >= int(qa_top_k):
                    break
        ranked_positions = list(deduped_positions[:max(int(qa_top_k), 0)])
        if ranked_positions:
            best_subset = list(ranked_positions)
            best_covq, best_cove, best_covb, best_ce_sum, _ = subset_stats(best_subset)

    return {
        "best_subset": list(best_subset),
        "ranked_positions": list(ranked_positions),
        "best_covq": int(best_covq),
        "best_cove": int(best_cove),
        "best_covb": int(best_covb),
        "best_ce_sum": float(best_ce_sum),
        "best_ce_subset": list(best_ce_subset),
        "best_ce_subset_score": float(best_ce_tuple[0]) if best_ce_tuple is not None else 0.0,
        "total_subsets_enumerated": int(total_subsets_enumerated),
        "total_valid_subsets": int(total_valid_subsets),
        "total_invalid_subsets_dedup": int(total_invalid_subsets_dedup),
        "fallback_reason": str(fallback_reason),
        "decision_trace": {
            "effective_decision_layer": str(effective_decision_layer),
            "max_covQ_subset_count": int(max_covq_subset_count),
            "max_covQ_max_covE_subset_count": int(max_covq_max_cove_subset_count),
            "max_covQ_max_covE_max_ce_subset_count": int(max_covq_max_cove_max_ce_subset_count),
            "covQ_margin": _numeric_margin(list(covq_values)),
            "covE_margin_within_max_covQ": _numeric_margin(list(cove_values_within_max_covq)),
            "ce_margin_within_max_covQ_covE": _numeric_margin(list(ce_values_within_max_covq_cove)),
        },
    }


def _build_budget_gap_admissibility_trace(normalized_positions: Sequence[int],
                                          baseline_prefix_positions: Sequence[int],
                                          position_sources: Mapping[int, str],
                                          pool_docs: Sequence[str],
                                          doc_covers_e: Mapping[int, Set[Tuple[str, str]]],
                                          reference_subset: Sequence[int],
                                          frozen_edge_universe: Set[Tuple[str, str]]) -> Tuple[List[int], Dict[str, Any]]:
    appended_positions = [
        int(pos) for pos in normalized_positions
        if str(position_sources.get(int(pos), "")).startswith("append_")
    ]
    covered_0: Set[Tuple[str, str]] = set()
    for pos in reference_subset:
        covered_0.update(doc_covers_e.get(int(pos), set()))
    gap_0 = set(frozen_edge_universe) - covered_0
    edge_support_counter: Counter[Tuple[str, str]] = Counter()
    for pos in normalized_positions:
        for edge in doc_covers_e.get(int(pos), set()):
            edge_support_counter[edge] += 1

    baseline_prefix_set = {int(pos) for pos in baseline_prefix_positions}
    kept_appended_positions: List[int] = []
    rows: List[Dict[str, object]] = []
    for pos in appended_positions:
        marginal_edges = set(doc_covers_e.get(int(pos), set())) & gap_0
        marginal_supports = sorted(int(edge_support_counter.get(edge, 0)) for edge in marginal_edges)
        kept = bool(marginal_edges)
        if kept:
            kept_appended_positions.append(int(pos))
        rows.append({
            "pool_position": int(pos),
            "title": extract_doc_title(pool_docs[int(pos)]),
            "source": str(position_sources.get(int(pos), "")),
            "frozen_edge_count": int(len(doc_covers_e.get(int(pos), set()))),
            "marginal_gap_edge_count": int(len(marginal_edges)),
            "kept": bool(kept),
            "marginal_gap_edges": sorted([list(edge) for edge in marginal_edges]),
            "marginal_gap_edge_support_counts": list(marginal_supports),
            "marginal_gap_edge_support_mean": _safe_mean(marginal_supports),
            "marginal_gap_edge_support_max": max(marginal_supports) if marginal_supports else 0,
        })

    filtered_positions = [
        int(pos) for pos in normalized_positions
        if int(pos) in baseline_prefix_set or int(pos) in set(kept_appended_positions)
    ]
    return filtered_positions, {
        "coverage_admissibility_mode": "budget_gap",
        "reference_subset_mode": "baseline_prefix_coverage",
        "reference_positions": list(reference_subset),
        "reference_titles": [extract_doc_title(pool_docs[int(pos)]) for pos in reference_subset],
        "reference_gap_edge_count": int(len(gap_0)),
        "candidate_positions_before_filter": list(normalized_positions),
        "candidate_positions_after_filter": list(filtered_positions),
        "appended_positions_before_filter": list(appended_positions),
        "appended_positions_after_filter": list(kept_appended_positions),
        "kept_appended_count": int(len(kept_appended_positions)),
        "dropped_appended_count": int(len(appended_positions) - len(kept_appended_positions)),
        "rows": rows,
    }


def _build_selected_appended_trace_rows(selected_positions: Sequence[int],
                                        candidate_positions: Sequence[int],
                                        appended_positions: Sequence[int],
                                        pool_docs: Sequence[str],
                                        doc_covers_e: Dict[int, Set[Tuple[str, str]]]) -> Tuple[
                                            List[Dict[str, object]],
                                            int,
                                            int,
                                        ]:
    appended_position_set = {int(pos) for pos in appended_positions}
    edge_support_counter: Counter[Tuple[str, str]] = Counter()
    for pos in candidate_positions:
        for edge in doc_covers_e.get(int(pos), set()):
            edge_support_counter[edge] += 1

    selected_rows: List[Dict[str, object]] = []
    nonzero_unique_gain_count = 0
    zero_unique_gain_count = 0
    selected_set = {int(pos) for pos in selected_positions}
    for pos in selected_positions:
        if int(pos) not in appended_position_set:
            continue
        covered_edges = set(doc_covers_e.get(int(pos), set()))
        other_edges: Set[Tuple[str, str]] = set()
        for other_pos in selected_set:
            if int(other_pos) == int(pos):
                continue
            other_edges.update(doc_covers_e.get(int(other_pos), set()))
        unique_edges = covered_edges - other_edges
        if unique_edges:
            nonzero_unique_gain_count += 1
        else:
            zero_unique_gain_count += 1
        selected_rows.append({
            "pool_position": int(pos),
            "title": extract_doc_title(pool_docs[int(pos)]),
            "covered_edge_count": int(len(covered_edges)),
            "unique_covE_gain": int(len(unique_edges)),
            "has_nonzero_unique_covE_gain": bool(unique_edges),
            "covered_edge_support_counts": sorted(int(edge_support_counter.get(edge, 0)) for edge in covered_edges),
            "unique_edge_support_counts": sorted(int(edge_support_counter.get(edge, 0)) for edge in unique_edges),
        })
    return selected_rows, int(nonzero_unique_gain_count), int(zero_unique_gain_count)


def _is_valid_coverage_subset(subset: Sequence[int], pool_docs: Sequence[str]) -> bool:
    seen_title_keys: Set[str] = set()
    for pos in subset:
        title_key = normalize_structure_text(extract_doc_title(pool_docs[int(pos)]))
        if title_key and title_key in seen_title_keys:
            return False
        if title_key:
            seen_title_keys.add(title_key)
    return True


def assemble_coverage_exact_search(query: str,
                                   pool_docs: Sequence[str],
                                   pool_doc_ids: Sequence[int | None],
                                   pool_doc_scores: Sequence[float],
                                   candidate_positions: Sequence[int],
                                   qa_top_k: int,
                                   seed_entities: Set[str],
                                   doc_idx_to_entities: Dict[int, Set[str]],
                                   doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                   ce_reranker: Any | None = None,
                                   position_sources: Dict[int, str] | None = None,
                                   coverage_score_variant: str = "qe_ce",
                                   coverage_atom_source: str = "candidate_pool",
                                   coverage_atom_positions: Sequence[int] | None = None,
                                   coverage_admissibility_mode: str = "off") -> Tuple[List[int], Dict[str, object]]:
    normalized_variant = normalize_coverage_score_variant(coverage_score_variant)
    normalized_atom_source = normalize_coverage_atom_source(coverage_atom_source)
    normalized_admissibility_mode = normalize_coverage_admissibility_mode(coverage_admissibility_mode)
    normalized_positions = _normalize_candidate_positions_for_assemble(candidate_positions, pool_docs)
    atom_source_positions = list(normalized_positions)
    if normalized_atom_source in {"baseline_prefix", "baseline_anchored"}:
        if coverage_atom_positions is None:
            atom_source_positions = _normalize_candidate_positions_for_assemble(
                [
                    int(pos) for pos in normalized_positions
                    if str((position_sources or {}).get(int(pos), "")) == "baseline_prefix"
                ],
                pool_docs,
            )
        else:
            atom_source_positions = _normalize_candidate_positions_for_assemble(coverage_atom_positions, pool_docs)
    atom_source_titles = [extract_doc_title(pool_docs[pos]) for pos in atom_source_positions]
    default_trace = {
        "assemble_mode": "coverage",
        "coverage_score_variant": normalized_variant,
        "score_field": "coverage_lexicographic_v1",
        "atom_source_mode": normalized_atom_source,
        "coverage_admissibility_mode": normalized_admissibility_mode,
        "atom_source_positions": list(atom_source_positions),
        "atom_source_titles": list(atom_source_titles),
        "atom_source_counts": {"A_Q": 0, "A_E": 0, "A_B": 0},
        "atom_source_is_frozen": bool(normalized_atom_source == "baseline_prefix"),
        "candidate_pool_positions": list(normalized_positions),
        "candidate_titles": [extract_doc_title(pool_docs[pos]) for pos in normalized_positions],
        "final_candidate_positions": list(normalized_positions),
        "final_candidate_titles": [extract_doc_title(pool_docs[pos]) for pos in normalized_positions],
        "selected_positions": [],
        "ranked_pool_positions": [],
        "ranked_titles": [],
        "ranking_rows": [],
        "atom_counts": {"A_Q": 0, "A_E": 0, "A_B": 0},
        "best_score": {"covQ": 0, "covE": 0, "ce_sum": 0.0},
        "best_covB_trace_only": 0,
        "ce_score_source": "cross_encoder" if ce_reranker is not None else "base_score_fallback",
        "best_ce_subset_by_true_ce": [],
        "best_ce_subset_score": 0.0,
        "coverage_vs_ce_overlap": 0,
        "coverage_vs_baseline_overlap": 0,
        "num_appended_selected": 0,
        "num_baseline_selected": 0,
        "total_subsets_enumerated": 0,
        "total_valid_subsets": 0,
        "total_invalid_subsets_dedup": 0,
        "fallback_reason": "",
        "effective_decision_layer": "fallback",
        "max_covQ_subset_count": 0,
        "max_covQ_max_covE_subset_count": 0,
        "max_covQ_max_covE_max_ce_subset_count": 0,
        "covQ_margin": None,
        "covE_margin_within_max_covQ": None,
        "ce_margin_within_max_covQ_covE": None,
        "selected_appended_rows": [],
        "selected_appended_with_nonzero_unique_gain": 0,
        "selected_appended_with_zero_unique_gain": 0,
        "admissibility_trace": {},
    }
    if not normalized_positions or qa_top_k <= 0:
        default_trace["fallback_reason"] = "empty_candidate_pool"
        return [], default_trace
    if normalized_admissibility_mode != "off" and normalized_atom_source != "baseline_prefix":
        raise ValueError(
            f"coverage_admissibility_mode={normalized_admissibility_mode} requires coverage_atom_source=baseline_prefix"
        )

    ce_rows, ce_score_field, ce_fallback_reason = _compute_assemble_score_rows(
        query=query,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=pool_doc_scores,
        candidate_positions=normalized_positions,
        assemble_mode="cross_encoder" if ce_reranker is not None else "base_score",
        hipporag=types.SimpleNamespace(query_to_embedding={}, passage_embeddings=np.array([])),
        ce_reranker=ce_reranker,
        position_sources=position_sources,
    )
    ce_rows_by_position = {
        int(row["pool_position"]): row
        for row in ce_rows
    }
    ce_scores = {
        pos: float(ce_rows_by_position[pos]["assemble_score"])
        for pos in ce_rows_by_position
    }
    sorted_ce_rows = _sort_assemble_score_rows(ce_rows)

    a_q, a_e, a_b, doc_covers_q, doc_covers_e, doc_covers_b = _build_coverage_atom_maps(
        candidate_positions=normalized_positions,
        atom_source_positions=atom_source_positions,
        normalized_atom_source=normalized_atom_source,
        pool_doc_ids=pool_doc_ids,
        seed_entities=seed_entities,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
    )
    baseline_prefix_positions = [
        int(pos) for pos in atom_source_positions
        if str((position_sources or {}).get(int(pos), "")) == "baseline_prefix"
    ]
    final_positions = list(normalized_positions)
    admissibility_trace: Dict[str, Any] = {}
    if normalized_admissibility_mode == "budget_gap":
        reference_search = _run_coverage_exact_search(
            candidate_positions=baseline_prefix_positions,
            qa_top_k=int(qa_top_k),
            pool_docs=pool_docs,
            sorted_ce_rows=sorted_ce_rows,
            ce_rows_by_position=ce_rows_by_position,
            ce_scores=ce_scores,
            pool_doc_scores=pool_doc_scores,
            doc_covers_q=doc_covers_q,
            doc_covers_e=doc_covers_e,
            doc_covers_b=doc_covers_b,
            normalized_variant=normalized_variant,
        )
        final_positions, admissibility_trace = _build_budget_gap_admissibility_trace(
            normalized_positions=normalized_positions,
            baseline_prefix_positions=baseline_prefix_positions,
            position_sources=(position_sources or {}),
            pool_docs=pool_docs,
            doc_covers_e=doc_covers_e,
            reference_subset=reference_search["best_subset"],
            frozen_edge_universe=a_e,
        )
        admissibility_trace["reference_score"] = {
            "covQ": int(reference_search["best_covq"]),
            "covE": int(reference_search["best_cove"]),
            "ce_sum": round(float(reference_search["best_ce_sum"]), 4),
        }

    search_result = _run_coverage_exact_search(
        candidate_positions=final_positions,
        qa_top_k=int(qa_top_k),
        pool_docs=pool_docs,
        sorted_ce_rows=sorted_ce_rows,
        ce_rows_by_position=ce_rows_by_position,
        ce_scores=ce_scores,
        pool_doc_scores=pool_doc_scores,
        doc_covers_q=doc_covers_q,
        doc_covers_e=doc_covers_e,
        doc_covers_b=doc_covers_b,
        normalized_variant=normalized_variant,
    )
    best_subset = list(search_result["best_subset"])
    ranked_positions = list(search_result["ranked_positions"])
    best_covq = int(search_result["best_covq"])
    best_cove = int(search_result["best_cove"])
    best_covb = int(search_result["best_covb"])
    best_ce_sum = float(search_result["best_ce_sum"])
    best_ce_subset = list(search_result["best_ce_subset"])
    fallback_reason = str(search_result["fallback_reason"])
    coverage_selected_set = set(best_subset)
    best_ce_selected_set = set(best_ce_subset)
    baseline_ranked_positions = list(final_positions[:max(int(qa_top_k), 0)])
    appended_positions = [
        int(pos) for pos in final_positions
        if str((position_sources or {}).get(int(pos), "")).startswith("append_")
    ]
    num_appended_selected = sum(
        1 for pos in best_subset
        if str((position_sources or {}).get(int(pos), "")).startswith("append_")
    )
    num_baseline_selected = sum(
        1 for pos in best_subset
        if str((position_sources or {}).get(int(pos), "")) == "baseline_prefix"
    )
    decision_layer_trace = dict(search_result["decision_trace"])
    selected_appended_rows, selected_appended_nonzero_unique_gain, selected_appended_zero_unique_gain = (
        _build_selected_appended_trace_rows(
            selected_positions=best_subset,
            candidate_positions=final_positions,
            appended_positions=appended_positions,
            pool_docs=pool_docs,
            doc_covers_e=doc_covers_e,
        )
    )
    default_trace.update({
        "selected_positions": list(ranked_positions),
        "ranked_pool_positions": list(ranked_positions),
        "ranked_titles": [extract_doc_title(pool_docs[pos]) for pos in ranked_positions],
        "final_candidate_positions": list(final_positions),
        "final_candidate_titles": [extract_doc_title(pool_docs[pos]) for pos in final_positions],
        "atom_counts": {
            "A_Q": int(len(a_q)),
            "A_E": int(len(a_e)),
            "A_B": int(len(a_b)),
        },
        "atom_source_counts": {
            "A_Q": int(len(a_q)),
            "A_E": int(len(a_e)),
            "A_B": int(len(a_b)),
        },
        "best_score": {
            "covQ": int(best_covq),
            "covE": int(best_cove),
            "ce_sum": round(float(best_ce_sum), 4),
        },
        "best_covB_trace_only": int(best_covb),
        "ce_score_source": "cross_encoder" if ce_reranker is not None else "base_score_fallback",
        "ce_score_field": str(ce_score_field),
        "ce_score_fallback_reason": str(ce_fallback_reason),
        "best_ce_subset_by_true_ce": list(best_ce_subset),
        "best_ce_subset_score": round(float(search_result["best_ce_subset_score"]), 4),
        "coverage_vs_ce_overlap": int(len(coverage_selected_set & best_ce_selected_set)),
        "coverage_vs_baseline_overlap": int(len(coverage_selected_set & set(baseline_ranked_positions))),
        "num_appended_selected": int(num_appended_selected),
        "num_baseline_selected": int(num_baseline_selected),
        "total_subsets_enumerated": int(search_result["total_subsets_enumerated"]),
        "total_valid_subsets": int(search_result["total_valid_subsets"]),
        "total_invalid_subsets_dedup": int(search_result["total_invalid_subsets_dedup"]),
        "fallback_reason": str(fallback_reason),
        "selected_appended_rows": list(selected_appended_rows),
        "selected_appended_with_nonzero_unique_gain": int(selected_appended_nonzero_unique_gain),
        "selected_appended_with_zero_unique_gain": int(selected_appended_zero_unique_gain),
        "admissibility_trace": admissibility_trace,
    })
    default_trace.update(decision_layer_trace)
    default_trace["ranking_rows"] = [
        {
            "rank": int(rank + 1),
            "pool_position": int(pos),
            "doc_id": int(pool_doc_ids[pos]) if pool_doc_ids[pos] is not None else None,
            "title": extract_doc_title(pool_docs[pos]),
            "source": str((position_sources or {}).get(int(pos), "candidate")),
            "covQ_contribution": int(len(doc_covers_q.get(pos, set()))),
            "covE_contribution": int(len(doc_covers_e.get(pos, set()))),
            "covB_contribution": int(len(doc_covers_b.get(pos, set()))),
            "ce_score": round(float(ce_scores.get(pos, 0.0)), 4),
            "base_score": round(float(pool_doc_scores[pos]) if pos < len(pool_doc_scores) else 0.0, 4),
        }
        for rank, pos in enumerate(ranked_positions)
    ]
    return list(ranked_positions), default_trace


def select_bridge_append_positions(pool_doc_ids: Sequence[int | None],
                                   normalized_base_scores: np.ndarray,
                                   pool_doc_titles: Sequence[str] | None,
                                   doc_idx_to_entities: Dict[int, Set[str]],
                                   doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                   adjacency: Dict[str, List[Tuple[str, float, str]]],
                                   initial_seed_entities: Sequence[str] | Set[str] | None,
                                   query_entities: Sequence[str] | Set[str] | None,
                                   pool_limit: int,
                                   expand_base_k: int,
                                   append_max_docs: int,
                                   expand_min_structure_score: float,
                                   structure_max_hops: int,
                                   structure_seed_target_bridge_mode: str,
                                   base_weight: float,
                                   structure_weight: float,
                                   novelty_weight: float,
                                   score_mode: str = "bridge",
                                   non_anchor_title_dedup: bool = True,
                                   append_policy: str = "bridge",
                                   append_random_seed: int = 0,
                                   query: str | None = None,
                                   pool_docs: Sequence[str] | None = None,
                                   gap_expand_mode: str = "heuristic",
                                   gap_expand_max_queries: int | None = None,
                                   ce_reranker: Any | None = None) -> Tuple[List[int], Dict[str, object]]:
    effective_pool_limit = max(int(pool_limit), 0)
    effective_base_k = min(max(int(expand_base_k), 0), effective_pool_limit)
    effective_append_max_docs = max(int(append_max_docs), 0)
    normalized_append_policy = normalize_append_policy(append_policy)
    normalized_gap_expand_mode = normalize_gap_expand_mode(gap_expand_mode)
    effective_gap_expand_max_queries = max(int(gap_expand_max_queries or DEFAULT_GAP_EXPAND_MAX_QUERIES), 1)
    if normalized_append_policy == "gap_expand" and normalized_gap_expand_mode == "unit_typed_abstain":
        return select_gap_expand_positions_unit_typed_abstain(
            pool_doc_ids=pool_doc_ids,
            normalized_base_scores=normalized_base_scores,
            pool_doc_titles=pool_doc_titles,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            adjacency=adjacency,
            initial_seed_entities=initial_seed_entities,
            query_entities=query_entities,
            pool_limit=pool_limit,
            expand_base_k=expand_base_k,
            append_max_docs=append_max_docs,
            expand_min_structure_score=expand_min_structure_score,
            structure_max_hops=structure_max_hops,
            structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
            base_weight=base_weight,
            structure_weight=structure_weight,
            novelty_weight=novelty_weight,
            score_mode=score_mode,
            non_anchor_title_dedup=non_anchor_title_dedup,
            append_random_seed=append_random_seed,
            query=query,
            pool_docs=pool_docs,
            gap_expand_max_queries=effective_gap_expand_max_queries,
            ce_reranker=ce_reranker,
        )
    if normalized_append_policy == "gap_expand" and normalized_gap_expand_mode == "typed_abstain":
        return select_gap_expand_positions_typed_abstain(
            pool_doc_ids=pool_doc_ids,
            normalized_base_scores=normalized_base_scores,
            pool_doc_titles=pool_doc_titles,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            adjacency=adjacency,
            initial_seed_entities=initial_seed_entities,
            query_entities=query_entities,
            pool_limit=pool_limit,
            expand_base_k=expand_base_k,
            append_max_docs=append_max_docs,
            expand_min_structure_score=expand_min_structure_score,
            structure_max_hops=structure_max_hops,
            structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
            base_weight=base_weight,
            structure_weight=structure_weight,
            novelty_weight=novelty_weight,
            score_mode=score_mode,
            non_anchor_title_dedup=non_anchor_title_dedup,
            append_random_seed=append_random_seed,
            query=query,
            pool_docs=pool_docs,
            gap_expand_max_queries=effective_gap_expand_max_queries,
            ce_reranker=ce_reranker,
        )
    baseline_prefix_positions = list(range(effective_base_k))
    candidate_positions = list(baseline_prefix_positions)
    appended_positions: List[int] = []
    append_steps: List[Dict[str, object]] = []
    gap_steps: List[Dict[str, object]] = []
    append_stop_reason = "append_cap_zero" if effective_append_max_docs == 0 else "unknown"
    covered_entities = normalize_entity_set(initial_seed_entities)
    for pos in baseline_prefix_positions:
        doc_id = pool_doc_ids[pos] if pos < len(pool_doc_ids) else None
        if doc_id is None:
            continue
        covered_entities.update(
            normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
        )

    seen_title_keys = set()
    if non_anchor_title_dedup:
        seen_title_keys = normalize_title_set([
            str(pool_doc_titles[pos]).strip()
            for pos in baseline_prefix_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ])

    remaining_positions = list(range(effective_base_k, effective_pool_limit))
    if not remaining_positions and effective_append_max_docs > 0:
        append_stop_reason = "no_deep_pool_candidates"

    if normalized_append_policy == "next_deep":
        for step_index, selected_position in enumerate(remaining_positions[:effective_append_max_docs]):
            selected_title = str(pool_doc_titles[selected_position]).strip() if pool_doc_titles is not None and 0 <= selected_position < len(pool_doc_titles) else ""
            title_key = normalize_structure_text(selected_title)
            if non_anchor_title_dedup and title_key and title_key in seen_title_keys:
                append_stop_reason = "duplicate_title_only"
                break
            candidate_positions.append(selected_position)
            appended_positions.append(selected_position)
            selected_doc_id = pool_doc_ids[selected_position] if selected_position < len(pool_doc_ids) else None
            if selected_doc_id is not None:
                covered_entities.update(
                    normalize_entity_set(doc_idx_to_entities.get(int(selected_doc_id), set()))
                )
            if non_anchor_title_dedup and title_key:
                seen_title_keys.add(title_key)
            append_steps.append({
                "step": int(step_index + 1),
                "selection_policy": normalized_append_policy,
                "selected_pool_position": int(selected_position),
                "selected_doc_id": int(selected_doc_id) if selected_doc_id is not None else None,
                "selected_title": selected_title,
            })
        if append_steps and len(appended_positions) >= effective_append_max_docs:
            append_stop_reason = "append_cap_reached"
        elif not append_steps and append_stop_reason == "unknown":
            append_stop_reason = "no_append"
    elif normalized_append_policy == "random_deep":
        rng = np.random.default_rng(int(append_random_seed))
        random_order = [int(pos) for pos in rng.permutation(remaining_positions).tolist()]
        for step_index, selected_position in enumerate(random_order[:effective_append_max_docs]):
            selected_title = str(pool_doc_titles[selected_position]).strip() if pool_doc_titles is not None and 0 <= selected_position < len(pool_doc_titles) else ""
            title_key = normalize_structure_text(selected_title)
            if non_anchor_title_dedup and title_key and title_key in seen_title_keys:
                append_stop_reason = "duplicate_title_only"
                break
            candidate_positions.append(selected_position)
            appended_positions.append(selected_position)
            selected_doc_id = pool_doc_ids[selected_position] if selected_position < len(pool_doc_ids) else None
            if selected_doc_id is not None:
                covered_entities.update(
                    normalize_entity_set(doc_idx_to_entities.get(int(selected_doc_id), set()))
                )
            if non_anchor_title_dedup and title_key:
                seen_title_keys.add(title_key)
            append_steps.append({
                "step": int(step_index + 1),
                "selection_policy": normalized_append_policy,
                "selected_pool_position": int(selected_position),
                "selected_doc_id": int(selected_doc_id) if selected_doc_id is not None else None,
                "selected_title": selected_title,
            })
        if append_steps and len(appended_positions) >= effective_append_max_docs:
            append_stop_reason = "append_cap_reached"
        elif not append_steps and append_stop_reason == "unknown":
            append_stop_reason = "no_append"
    else:
        for step_index in range(effective_append_max_docs):
            if not remaining_positions:
                append_stop_reason = "no_candidate_remaining"
                break

            scored_candidates = score_bridge_candidates(
                pool_doc_ids=pool_doc_ids,
                normalized_base_scores=normalized_base_scores,
                pool_doc_titles=pool_doc_titles,
                doc_idx_to_entities=doc_idx_to_entities,
                doc_idx_to_edges=doc_idx_to_edges,
                adjacency=adjacency,
                remaining_positions=remaining_positions,
                covered_entities=covered_entities,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                query_entities=query_entities,
                score_mode=score_mode,
            )
            if not scored_candidates:
                append_stop_reason = "no_scored_candidate"
                break

            gap_state = {
                "gap_type": "none",
                "gap_mode": normalized_gap_expand_mode,
                "fallback_used": False,
                "gap_anchors": [],
                "covered_query_entities": [],
                "uncovered_query_entities": [],
                "covered_entities_snapshot": sorted(normalize_entity_set(covered_entities))[:16],
                "baseline_titles": [],
                "baseline_entity_count": 0,
                "relation_terms": [],
                "micro_queries": [str(query or "").strip()] if str(query or "").strip() else [],
                "micro_query_count": 1 if str(query or "").strip() else 0,
            }
            if normalized_append_policy == "gap_expand":
                gap_state = detect_gap_expand_state(
                    query=str(query or ""),
                    baseline_prefix_positions=baseline_prefix_positions,
                    pool_docs=pool_docs,
                    pool_doc_ids=pool_doc_ids,
                    doc_idx_to_entities=doc_idx_to_entities,
                    query_entities=query_entities,
                    covered_entities=covered_entities,
                    gap_expand_mode=normalized_gap_expand_mode,
                    gap_expand_max_queries=effective_gap_expand_max_queries,
                )
                ranked_candidates = rerank_gap_expand_candidates(
                    scored_candidates=scored_candidates,
                    pool_docs=pool_docs,
                    gap_state=gap_state,
                    covered_entities=covered_entities,
                )
            else:
                ranked_candidates = sorted(
                    scored_candidates,
                    key=lambda row: (
                        -float(row.get("structure_score", 0.0) or 0.0),
                        -float(row.get("closure_score", 0.0) or 0.0),
                        -float(row.get("novelty_score", 0.0) or 0.0),
                        int(row.get("pool_position", 0) or 0),
                    ),
                )

            best_threshold_score = float(
                ranked_candidates[0].get(
                    "gap_gate_score_raw" if normalized_append_policy == "gap_expand" else "structure_score",
                    0.0,
                ) or 0.0
            )
            if best_threshold_score < float(expand_min_structure_score):
                if normalized_append_policy == "gap_expand":
                    gap_steps.append({
                        "step": int(step_index + 1),
                        "gap_type": str(gap_state.get("gap_type", "none") or "none"),
                        "gap_mode": str(gap_state.get("gap_mode", normalized_gap_expand_mode) or normalized_gap_expand_mode),
                        "fallback_used": bool(gap_state.get("fallback_used", False)),
                        "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                        "covered_query_entities": list(gap_state.get("covered_query_entities", []) or []),
                        "uncovered_query_entities": list(gap_state.get("uncovered_query_entities", []) or []),
                        "relation_terms": list(gap_state.get("relation_terms", []) or []),
                        "micro_queries": list(gap_state.get("micro_queries", []) or []),
                        "selection_blocked_by_threshold": True,
                        "candidate_preview": [
                            {
                                "preview_rank": int(rank + 1),
                                "pool_position": int(row.get("pool_position", -1) or -1),
                                "doc_id": int(row["doc_id"]) if row.get("doc_id") is not None else None,
                                "title": str(row.get("doc_title", "") or ""),
                                "gap_score": round(float(row.get("gap_score", 0.0) or 0.0), 4),
                                "gap_gate_score": round(float(row.get("gap_gate_score", 0.0) or 0.0), 4),
                                "gap_combined_score": round(float(row.get("gap_combined_score", 0.0) or 0.0), 4),
                                "structure_score": round(float(row.get("structure_score", 0.0) or 0.0), 4),
                            }
                            for rank, row in enumerate(ranked_candidates[:5])
                        ],
                    })
                append_stop_reason = "structure_below_threshold"
                break

            selected_row = None
            duplicate_skip_count = 0
            for row in ranked_candidates:
                title_key = normalize_structure_text(str(row.get("doc_title", "")).strip())
                if non_anchor_title_dedup and title_key and title_key in seen_title_keys:
                    duplicate_skip_count += 1
                    continue
                selected_row = row
                break

            if selected_row is None:
                append_stop_reason = "duplicate_title_only"
                break

            selected_position = int(selected_row["pool_position"])
            candidate_positions.append(selected_position)
            appended_positions.append(selected_position)
            remaining_positions = [pos for pos in remaining_positions if int(pos) != selected_position]

            selected_doc_id = selected_row.get("doc_id")
            if selected_doc_id is not None:
                covered_entities.update(
                    normalize_entity_set(doc_idx_to_entities.get(int(selected_doc_id), set()))
                )
            selected_title_key = normalize_structure_text(str(selected_row.get("doc_title", "")).strip())
            if non_anchor_title_dedup and selected_title_key:
                seen_title_keys.add(selected_title_key)

            gap_steps.append({
                "step": int(step_index + 1),
                "gap_type": str(gap_state.get("gap_type", "none") or "none"),
                "gap_mode": str(gap_state.get("gap_mode", normalized_gap_expand_mode) or normalized_gap_expand_mode),
                "fallback_used": bool(gap_state.get("fallback_used", False)),
                "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                "covered_query_entities": list(gap_state.get("covered_query_entities", []) or []),
                "uncovered_query_entities": list(gap_state.get("uncovered_query_entities", []) or []),
                "relation_terms": list(gap_state.get("relation_terms", []) or []),
                "micro_queries": list(gap_state.get("micro_queries", []) or []),
                "candidate_preview": [
                    {
                        "preview_rank": int(rank + 1),
                        "pool_position": int(row.get("pool_position", -1) or -1),
                        "doc_id": int(row["doc_id"]) if row.get("doc_id") is not None else None,
                        "title": str(row.get("doc_title", "") or ""),
                        "gap_score": round(float(row.get("gap_score", 0.0) or 0.0), 4),
                        "gap_gate_score": round(float(row.get("gap_gate_score", 0.0) or 0.0), 4),
                        "gap_combined_score": round(float(row.get("gap_combined_score", 0.0) or 0.0), 4),
                        "structure_score": round(float(row.get("structure_score", 0.0) or 0.0), 4),
                        "closure_score": round(float(row.get("closure_score", 0.0) or 0.0), 4),
                        "novelty_score": round(float(row.get("novelty_score", 0.0) or 0.0), 4),
                    }
                    for rank, row in enumerate(ranked_candidates[:5])
                ],
            })
            append_steps.append({
                "step": int(step_index + 1),
                "selection_policy": normalized_append_policy,
                "selected_pool_position": selected_position,
                "selected_doc_id": int(selected_doc_id) if selected_doc_id is not None else None,
                "selected_title": str(selected_row.get("doc_title", "") or ""),
                "selected_structure_score": round(float(selected_row.get("structure_score", 0.0) or 0.0), 4),
                "selected_closure_score": round(float(selected_row.get("closure_score", 0.0) or 0.0), 4),
                "selected_novelty_score": round(float(selected_row.get("novelty_score", 0.0) or 0.0), 4),
                "selected_combined_score": round(float(selected_row.get("combined_score", 0.0) or 0.0), 4),
                "selected_gap_score": round(float(selected_row.get("gap_score", 0.0) or 0.0), 4),
                "selected_gap_gate_score": round(float(selected_row.get("gap_gate_score", 0.0) or 0.0), 4),
                "selected_gap_combined_score": round(float(selected_row.get("gap_combined_score", 0.0) or 0.0), 4),
                "gap_type": str(gap_state.get("gap_type", "none") or "none"),
                "gap_mode": str(gap_state.get("gap_mode", normalized_gap_expand_mode) or normalized_gap_expand_mode),
                "gap_fallback_used": bool(gap_state.get("fallback_used", False)),
                "gap_micro_queries": list(gap_state.get("micro_queries", []) or []),
                "gap_anchors": list(gap_state.get("gap_anchors", []) or []),
                "candidate_pool_size": int(len(scored_candidates)),
                "best_structure_score": round(best_threshold_score, 4),
                "duplicate_skip_count": int(duplicate_skip_count),
                "candidate_preview": [
                    {
                        "preview_rank": int(rank + 1),
                        "pool_position": int(row.get("pool_position", -1) or -1),
                        "doc_id": int(row["doc_id"]) if row.get("doc_id") is not None else None,
                        "title": str(row.get("doc_title", "") or ""),
                        "structure_score": round(float(row.get("structure_score", 0.0) or 0.0), 4),
                        "closure_score": round(float(row.get("closure_score", 0.0) or 0.0), 4),
                        "novelty_score": round(float(row.get("novelty_score", 0.0) or 0.0), 4),
                        "combined_score": round(float(row.get("combined_score", 0.0) or 0.0), 4),
                        "gap_score": round(float(row.get("gap_score", 0.0) or 0.0), 4),
                        "gap_gate_score": round(float(row.get("gap_gate_score", 0.0) or 0.0), 4),
                        "gap_combined_score": round(float(row.get("gap_combined_score", 0.0) or 0.0), 4),
                    }
                    for rank, row in enumerate(ranked_candidates[:5])
                ],
            })

    if append_steps and len(appended_positions) >= effective_append_max_docs:
        append_stop_reason = "append_cap_reached"
    elif not append_steps and append_stop_reason == "unknown":
        append_stop_reason = "no_append"

    trace = {
        "selector": "bridge_append",
        "expand_base_k": int(effective_base_k),
        "append_max_docs": int(effective_append_max_docs),
        "append_policy": normalized_append_policy,
        "append_random_seed": int(append_random_seed),
        "gap_expand_mode": normalized_gap_expand_mode,
        "gap_expand_max_queries": int(effective_gap_expand_max_queries),
        "expand_min_structure_score": round(float(expand_min_structure_score), 4),
        "score_mode": normalize_setwise_score_mode(score_mode),
        "non_anchor_title_dedup": bool(non_anchor_title_dedup),
        "gap_expand_enabled": bool(normalized_append_policy == "gap_expand"),
        "baseline_prefix_positions": list(baseline_prefix_positions),
        "baseline_prefix_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in baseline_prefix_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "appended_positions": list(appended_positions),
        "appended_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in appended_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "append_count": int(len(appended_positions)),
        "append_stop_reason": str(append_stop_reason),
        "append_steps": append_steps,
        "gap_steps": gap_steps,
        "gap_type": str(gap_steps[0].get("gap_type", "none")) if gap_steps else "none",
        "gap_mode": str(gap_steps[0].get("gap_mode", normalized_gap_expand_mode)) if gap_steps else str(normalized_gap_expand_mode),
        "gap_fallback_used": bool(any(bool(step.get("fallback_used", False)) for step in gap_steps)),
        "gap_micro_queries": list(gap_steps[0].get("micro_queries", []) or []) if gap_steps else [],
        "gap_micro_query_count": int(len(gap_steps[0].get("micro_queries", []) or [])) if gap_steps else 0,
        "gap_anchors": list(gap_steps[0].get("gap_anchors", []) or []) if gap_steps else [],
        "gap_candidate_positions": list(appended_positions),
        "gap_candidate_doc_ids": [
            int(pool_doc_ids[pos]) if pos < len(pool_doc_ids) and pool_doc_ids[pos] is not None else None
            for pos in appended_positions
        ],
        "gap_candidate_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in appended_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "candidate_set_positions": list(candidate_positions),
        "candidate_set_titles": [
            str(pool_doc_titles[pos]).strip()
            for pos in candidate_positions
            if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
        ],
        "candidate_set_size": int(len(candidate_positions)),
        "covered_entity_count_after_expand": int(len(covered_entities)),
    }
    return candidate_positions, trace


def rerank_candidate_positions_for_assemble(query: str,
                                            pool_docs: Sequence[str],
                                            pool_doc_ids: Sequence[int | None],
                                            pool_doc_scores: Sequence[float],
                                            candidate_positions: Sequence[int],
                                            assemble_mode: str,
                                            hipporag: HippoRAG,
                                            ce_reranker: Any = None,
                                            position_sources: Dict[int, str] | None = None) -> Tuple[List[int], Dict[str, object]]:
    normalized_mode = normalize_assemble_mode(assemble_mode)
    normalized_positions = _normalize_candidate_positions_for_assemble(candidate_positions, pool_docs)
    default_trace = {
        "assemble_mode": normalized_mode,
        "candidate_pool_positions": list(normalized_positions),
        "candidate_titles": [extract_doc_title(pool_docs[pos]) for pos in normalized_positions],
        "ranked_pool_positions": list(normalized_positions),
        "ranked_titles": [extract_doc_title(pool_docs[pos]) for pos in normalized_positions],
        "ranking_rows": [],
        "score_field": "candidate_order",
        "fallback_reason": "",
    }
    if normalized_mode == "none" or not normalized_positions:
        default_trace["ranking_rows"] = [
            {
                "rank": int(rank + 1),
                "pool_position": int(pos),
                "doc_id": int(pool_doc_ids[pos]) if pool_doc_ids[pos] is not None else None,
                "title": extract_doc_title(pool_docs[pos]),
                "source": str((position_sources or {}).get(int(pos), "candidate")),
                "base_score": round(float(pool_doc_scores[pos]), 4) if pos < len(pool_doc_scores) else 0.0,
                "assemble_score": None,
            }
            for rank, pos in enumerate(normalized_positions)
        ]
        return list(normalized_positions), default_trace

    rows, score_field, fallback_reason = _compute_assemble_score_rows(
        query=query,
        pool_docs=pool_docs,
        pool_doc_ids=pool_doc_ids,
        pool_doc_scores=pool_doc_scores,
        candidate_positions=normalized_positions,
        assemble_mode=normalized_mode,
        hipporag=hipporag,
        ce_reranker=ce_reranker,
        position_sources=position_sources,
    )
    rows = _sort_assemble_score_rows(rows)
    ranked_positions = [int(row["pool_position"]) for row in rows]
    trace = {
        "assemble_mode": normalized_mode,
        "candidate_pool_positions": list(normalized_positions),
        "candidate_titles": [extract_doc_title(pool_docs[pos]) for pos in normalized_positions],
        "ranked_pool_positions": list(ranked_positions),
        "ranked_titles": [extract_doc_title(pool_docs[pos]) for pos in ranked_positions],
        "ranking_rows": [
            {
                "rank": int(rank + 1),
                "pool_position": int(row["pool_position"]),
                "doc_id": row["doc_id"],
                "title": str(row["title"]),
                "source": str(row["source"]),
                "base_score": round(float(row["base_score"]), 4),
                "assemble_score": None if not np.isfinite(float(row["assemble_score"])) else round(float(row["assemble_score"]), 4),
            }
            for rank, row in enumerate(rows)
        ],
        "score_field": str(score_field),
        "fallback_reason": str(fallback_reason),
    }
    return ranked_positions, trace


def score_evidence_state(pool_doc_ids: Sequence[int | None],
                         normalized_base_scores: np.ndarray,
                         pool_doc_titles: Sequence[str] | None,
                         doc_idx_to_entities: Dict[int, Set[str]],
                         doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                         adjacency: Dict[str, List[Tuple[str, float, str]]],
                         selected_positions: Sequence[int],
                         fixed_prefix_positions: Sequence[int] | None,
                         seed_entities: Sequence[str] | Set[str] | None,
                         query_entities: Sequence[str] | Set[str] | None,
                         support_query_entities: Sequence[str] | Set[str] | None,
                         structure_max_hops: int,
                         base_weight: float,
                         structure_weight: float,
                         novelty_weight: float,
                         structure_seed_target_bridge_mode: str = "off",
                         state_weight_config: Dict[str, float] | None = None) -> Dict[str, float]:
    normalized_seed = normalize_entity_set(seed_entities)
    normalized_query = normalize_entity_set(query_entities) or set(normalized_seed)
    normalized_support_query = normalize_entity_set(support_query_entities) or set(normalized_query) or set(normalized_seed)
    state_weights = resolve_set_closure_state_weight_config(state_weight_config)
    chosen_positions = [int(pos) for pos in selected_positions]
    fixed_prefix = {int(pos) for pos in (fixed_prefix_positions or [])}
    suffix_positions = [pos for pos in chosen_positions if pos not in fixed_prefix]
    focus_positions = suffix_positions or list(chosen_positions)

    selected_doc_entities: Dict[int, Set[str]] = {}
    selected_titles: List[str] = []
    selected_entity_union: Set[str] = set()
    for pos in chosen_positions:
        if pos >= len(pool_doc_ids):
            continue
        doc_id = pool_doc_ids[pos]
        doc_entities = (
            normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set()))
            if doc_id is not None
            else set()
        )
        selected_doc_entities[pos] = doc_entities
        selected_entity_union.update(doc_entities)
        if pool_doc_titles is not None and pos < len(pool_doc_titles):
            title = str(pool_doc_titles[pos]).strip()
            if title:
                selected_titles.append(title)

    if not focus_positions:
        return {
            "state_score": 0.0,
            "path_connectivity": 0.0,
            "reachable_doc_ratio": 0.0,
            "query_reachability": 0.0,
            "support_mean": 0.0,
            "support_min": 0.0,
            "closure_mean": 0.0,
            "suffix_base_mean": 0.0,
            "query_coverage": 0.0,
            "frontier_ratio": 0.0,
            "redundancy_penalty": 0.0,
            "state_weight_config": dict(state_weights),
        }

    support_scores: List[float] = []
    closure_scores: List[float] = []
    for pos in focus_positions:
        context_entities = set(normalized_seed)
        for other_pos, other_entities in selected_doc_entities.items():
            if other_pos == pos:
                continue
            context_entities.update(other_entities)
        candidate_rows = score_bridge_candidates(
            pool_doc_ids=pool_doc_ids,
            normalized_base_scores=normalized_base_scores,
            pool_doc_titles=pool_doc_titles,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            adjacency=adjacency,
            remaining_positions=[pos],
            covered_entities=context_entities,
            structure_max_hops=structure_max_hops,
            structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
            base_weight=base_weight,
            structure_weight=structure_weight,
            novelty_weight=novelty_weight,
            query_entities=normalized_support_query,
            score_mode="closure_proxy",
        )
        if not candidate_rows:
            continue
        candidate_detail = candidate_rows[0]
        support_scores.append(float(candidate_detail["selection_score_raw"]))
        closure_scores.append(float(candidate_detail["closure_score_raw"]))

    supported_entities = selected_entity_union - normalized_seed
    frontier_scores = (
        expand_directed_entities(normalized_seed, adjacency, max_hops=structure_max_hops)
        if normalized_seed
        else {}
    )
    frontier_values = [
        min(
            1.0,
            max(
                float(frontier_scores.get(entity, 0.0)),
                1.0 if entity in normalized_query else 0.0,
            ),
        )
        for entity in supported_entities
    ]
    frontier_ratio = float(np.mean(frontier_values)) if frontier_values else 0.0

    query_denominator = max(1, len(normalized_query))
    query_coverage = len(selected_entity_union & normalized_query) / float(query_denominator)

    total_entity_mentions = sum(len(selected_doc_entities.get(pos, set())) for pos in chosen_positions)
    entity_redundancy = (
        1.0 - (len(selected_entity_union) / float(total_entity_mentions))
        if total_entity_mentions > 0
        else 0.0
    )
    title_redundancy = (
        (len(selected_titles) - len(set(selected_titles))) / float(len(selected_titles))
        if selected_titles
        else 0.0
    )
    redundancy_penalty = max(entity_redundancy, title_redundancy)

    support_mean = float(np.mean(support_scores)) if support_scores else 0.0
    support_min = float(np.min(support_scores)) if support_scores else 0.0
    closure_mean = float(np.mean(closure_scores)) if closure_scores else 0.0
    suffix_base_scores = [
        float(normalized_base_scores[pos])
        for pos in focus_positions
        if 0 <= pos < len(normalized_base_scores)
    ]
    suffix_base_mean = float(np.mean(suffix_base_scores)) if suffix_base_scores else 0.0
    prefix_reachable_entities = set(normalized_seed)
    for pos in chosen_positions:
        if pos not in fixed_prefix:
            continue
        prefix_reachable_entities.update(selected_doc_entities.get(pos, set()))
    path_metrics = compute_state_path_connectivity_metrics(
        selected_doc_entities=selected_doc_entities,
        focus_positions=focus_positions,
        initial_reachable_entities=prefix_reachable_entities,
        query_entities=normalized_query,
        adjacency=adjacency,
        structure_max_hops=structure_max_hops,
    )
    path_connectivity = float(path_metrics["path_connectivity"])
    reachable_doc_ratio = float(path_metrics["reachable_doc_ratio"])
    query_reachability = float(path_metrics["query_reachability"])
    state_score = (
        state_weights["path_connectivity"] * path_connectivity
        + state_weights["reachable_doc_ratio"] * reachable_doc_ratio
        + state_weights["query_reachability"] * query_reachability
        + state_weights["support_mean"] * support_mean
        + state_weights["support_min"] * support_min
        + state_weights["closure_mean"] * closure_mean
        + state_weights["suffix_base_mean"] * suffix_base_mean
        + state_weights["query_coverage"] * query_coverage
        + state_weights["frontier_ratio"] * frontier_ratio
        - state_weights["redundancy_penalty"] * redundancy_penalty
    )
    return {
        "state_score": float(state_score),
        "path_connectivity": float(path_connectivity),
        "reachable_doc_ratio": float(reachable_doc_ratio),
        "query_reachability": float(query_reachability),
        "support_mean": float(support_mean),
        "support_min": float(support_min),
        "closure_mean": float(closure_mean),
        "suffix_base_mean": float(suffix_base_mean),
        "query_coverage": float(query_coverage),
        "frontier_ratio": float(frontier_ratio),
        "redundancy_penalty": float(redundancy_penalty),
        "state_weight_config": dict(state_weights),
    }


def compute_bridge_gate_decision(pool_doc_ids: Sequence[int | None],
                                 pool_doc_scores: np.ndarray,
                                 pool_doc_titles: Sequence[str] | None,
                                 doc_idx_to_entities: Dict[int, Set[str]],
                                 doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                 adjacency: Dict[str, List[Tuple[str, float, str]]],
                                 qa_top_k: int,
                                 initial_seed_entities: Sequence[str] | Set[str] | None = None,
                                 query_entities: Sequence[str] | Set[str] | None = None,
                                 anchor_count: int = 2,
                                 reserve_top_m: int = 0,
                                 structure_max_hops: int = 2,
                                 structure_seed_target_bridge_mode: str = "off",
                                 base_weight: float = 0.25,
                                 structure_weight: float = 0.60,
                                 novelty_weight: float = 0.15,
                                 score_mode: str = "bridge",
                                 max_bridge_slots: int = 0,
                                 gate_mode: str = "none",
                                 gate_min_structure_score: float = 0.15,
                                 gate_min_combined_margin: float = 0.0,
                                 gate_min_closure_score: float = 0.0,
                                 gate_min_novelty_score: float = 0.0,
                                 gate_min_frontier_gain: float = 0.0,
                                 gate_min_path_coherence: float = 0.0,
                                 non_anchor_title_dedup: bool = False) -> Dict[str, object]:
    normalized_gate_mode = str(gate_mode or "none").strip().lower()
    gate_logic_mode = (
        "suffix_bridge"
        if normalized_gate_mode == "suffix_bridge_saturation_guard"
        else normalized_gate_mode
    )
    candidate_count = len(pool_doc_ids)
    target_k = min(candidate_count, max(int(qa_top_k), 0))
    anchor_positions, reserved_positions = resolve_reserved_positions(
        candidate_count=candidate_count,
        target_k=target_k,
        anchor_count=anchor_count,
        reserve_top_m=reserve_top_m,
    )
    selection_target_k = resolve_selection_target_k(
        target_k=target_k,
        reserved_count=len(reserved_positions),
        max_bridge_slots=max_bridge_slots,
    )
    suffix_budget = max(0, selection_target_k - len(reserved_positions))

    decision = {
        "gate_mode": normalized_gate_mode,
        "gate_logic_mode": gate_logic_mode,
        "gate_enabled": normalized_gate_mode != "none",
        "use_selector": True,
        "reason": "disabled" if normalized_gate_mode == "none" else "apply",
        "target_k": int(target_k),
        "selection_target_k": int(selection_target_k),
        "max_bridge_slots": int(max(0, max_bridge_slots)),
        "reserved_positions": [int(pos) for pos in reserved_positions],
        "suffix_budget": int(suffix_budget),
        "baseline_suffix_positions": [],
        "best_offrank_pool_position": None,
        "best_offrank_structure_score": 0.0,
        "best_offrank_combined_score": 0.0,
        "best_offrank_novelty_score": 0.0,
        "best_offrank_frontier_gain_score": 0.0,
        "best_offrank_path_coherence_score": 0.0,
        "best_offrank_closure_score": 0.0,
        "weakest_baseline_suffix_pool_position": None,
        "weakest_baseline_suffix_structure_score": 0.0,
        "weakest_baseline_suffix_combined_score": 0.0,
        "weakest_baseline_suffix_novelty_score": 0.0,
        "weakest_baseline_suffix_frontier_gain_score": 0.0,
        "weakest_baseline_suffix_path_coherence_score": 0.0,
        "weakest_baseline_suffix_closure_score": 0.0,
        "gate_min_structure_score": round(float(gate_min_structure_score), 4),
        "gate_min_combined_margin": round(float(gate_min_combined_margin), 4),
        "gate_min_closure_score": round(float(gate_min_closure_score), 4),
        "gate_min_novelty_score": round(float(gate_min_novelty_score), 4),
        "gate_min_frontier_gain": round(float(gate_min_frontier_gain), 4),
        "gate_min_path_coherence": round(float(gate_min_path_coherence), 4),
    }
    if normalized_gate_mode == "none":
        return decision

    if suffix_budget <= 0:
        decision["use_selector"] = False
        decision["reason"] = "no_bridge_slots"
        return decision

    normalized_base_scores = min_max_normalize_array(np.asarray(pool_doc_scores, dtype=float))
    covered_entities = normalize_entity_set(initial_seed_entities)
    blocked_titles: Set[str] = set()
    for reserved_pos in reserved_positions:
        doc_id = pool_doc_ids[reserved_pos]
        if doc_id is not None:
            covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))
        if pool_doc_titles is not None and reserved_pos < len(pool_doc_titles):
            reserved_title = str(pool_doc_titles[reserved_pos]).strip()
            if reserved_title:
                blocked_titles.add(reserved_title)

    remaining_positions = [pos for pos in range(candidate_count) if pos not in set(reserved_positions)]
    baseline_suffix_positions = list(remaining_positions[:suffix_budget])
    decision["baseline_suffix_positions"] = [int(pos) for pos in baseline_suffix_positions]
    if not remaining_positions:
        decision["use_selector"] = False
        decision["reason"] = "no_remaining_candidates"
        return decision

    scored_candidates = score_bridge_candidates(
        pool_doc_ids=pool_doc_ids,
        normalized_base_scores=normalized_base_scores,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        adjacency=adjacency,
        remaining_positions=remaining_positions,
        covered_entities=covered_entities,
        query_entities=query_entities or initial_seed_entities,
        structure_max_hops=structure_max_hops,
        structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
        base_weight=base_weight,
        structure_weight=structure_weight,
        novelty_weight=novelty_weight,
        score_mode=score_mode,
    )
    scored_candidates = filter_title_dedup_candidates(
        scored_candidates=scored_candidates,
        blocked_titles=blocked_titles,
        enabled=non_anchor_title_dedup,
    )
    candidates_by_position = {
        int(candidate["pool_position"]): candidate
        for candidate in scored_candidates
    }
    baseline_suffix_candidates = [
        candidates_by_position[pos]
        for pos in baseline_suffix_positions
        if pos in candidates_by_position
    ]
    offrank_candidates = [
        candidate
        for candidate in scored_candidates
        if int(candidate["pool_position"]) not in set(baseline_suffix_positions)
    ]

    if not offrank_candidates:
        decision["use_selector"] = False
        decision["reason"] = "no_offrank_bridge_candidate"
        return decision

    best_offrank = max(
        offrank_candidates,
        key=lambda candidate: (
            float(candidate["structure_score"]),
            float(candidate["combined_score_raw"]),
            -int(candidate["pool_position"]),
        ),
    )
    decision["best_offrank_pool_position"] = int(best_offrank["pool_position"])
    decision["best_offrank_structure_score"] = round(float(best_offrank["structure_score"]), 4)
    decision["best_offrank_combined_score"] = round(float(best_offrank["combined_score"]), 4)
    decision["best_offrank_novelty_score"] = round(float(best_offrank.get("novelty_score", 0.0) or 0.0), 4)
    decision["best_offrank_frontier_gain_score"] = round(float(best_offrank.get("frontier_gain_score", 0.0) or 0.0), 4)
    decision["best_offrank_path_coherence_score"] = round(float(best_offrank.get("path_coherence_score", 0.0) or 0.0), 4)
    decision["best_offrank_closure_score"] = round(float(best_offrank.get("closure_score", 0.0) or 0.0), 4)

    if baseline_suffix_candidates:
        weakest_baseline_suffix = min(
            baseline_suffix_candidates,
            key=lambda candidate: (
                float(candidate["structure_score"]),
                float(candidate["combined_score_raw"]),
                int(candidate["pool_position"]),
            ),
        )
        decision["weakest_baseline_suffix_pool_position"] = int(weakest_baseline_suffix["pool_position"])
        decision["weakest_baseline_suffix_structure_score"] = round(float(weakest_baseline_suffix["structure_score"]), 4)
        decision["weakest_baseline_suffix_combined_score"] = round(float(weakest_baseline_suffix["combined_score"]), 4)
        decision["weakest_baseline_suffix_novelty_score"] = round(float(weakest_baseline_suffix.get("novelty_score", 0.0) or 0.0), 4)
        decision["weakest_baseline_suffix_frontier_gain_score"] = round(float(weakest_baseline_suffix.get("frontier_gain_score", 0.0) or 0.0), 4)
        decision["weakest_baseline_suffix_path_coherence_score"] = round(float(weakest_baseline_suffix.get("path_coherence_score", 0.0) or 0.0), 4)
        decision["weakest_baseline_suffix_closure_score"] = round(float(weakest_baseline_suffix.get("closure_score", 0.0) or 0.0), 4)
    else:
        weakest_baseline_suffix = None

    if float(best_offrank["structure_score"]) < float(gate_min_structure_score):
        decision["use_selector"] = False
        decision["reason"] = "offrank_structure_below_threshold"
        return decision

    if gate_logic_mode == "suffix_bridge_precision":
        if float(best_offrank.get("closure_score_raw", 0.0) or 0.0) < float(gate_min_closure_score):
            decision["use_selector"] = False
            decision["reason"] = "offrank_closure_below_threshold"
            return decision
        has_novelty_signal = (
            float(best_offrank.get("novelty_score", 0.0) or 0.0) >= float(gate_min_novelty_score)
        )
        has_chain_signal = (
            float(best_offrank.get("frontier_gain_score", 0.0) or 0.0) >= float(gate_min_frontier_gain)
            or float(best_offrank.get("path_coherence_score", 0.0) or 0.0) >= float(gate_min_path_coherence)
        )
        if not has_novelty_signal:
            decision["use_selector"] = False
            decision["reason"] = "offrank_novelty_below_threshold"
            return decision
        if not has_chain_signal:
            decision["use_selector"] = False
            decision["reason"] = "offrank_chain_signal_below_threshold"
            return decision

    if (
        weakest_baseline_suffix is not None
        and float(best_offrank["structure_score"]) <= float(weakest_baseline_suffix["structure_score"]) + 1e-9
    ):
        decision["use_selector"] = False
        decision["reason"] = "baseline_suffix_already_has_equal_or_better_bridge"
        return decision

    if (
        weakest_baseline_suffix is not None
        and float(best_offrank["combined_score_raw"])
        <= float(weakest_baseline_suffix["combined_score_raw"]) + float(gate_min_combined_margin)
    ):
        decision["use_selector"] = False
        decision["reason"] = "offrank_combined_margin_too_small"
        return decision

    decision["reason"] = "offrank_bridge_signal_detected"
    return decision


def select_bridge_greedy_positions(pool_doc_ids: Sequence[int | None],
                                   pool_doc_scores: np.ndarray,
                                   pool_doc_titles: Sequence[str] | None,
                                   doc_idx_to_entities: Dict[int, Set[str]],
                                   doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                   adjacency: Dict[str, List[Tuple[str, float, str]]],
                                   qa_top_k: int,
                                   initial_seed_entities: Sequence[str] | Set[str] | None = None,
                                   query_entities: Sequence[str] | Set[str] | None = None,
                                   anchor_count: int = 2,
                                   reserve_top_m: int = 0,
                                   max_bridge_slots: int = 0,
                                   structure_max_hops: int = 2,
                                   structure_seed_target_bridge_mode: str = "off",
                                   base_weight: float = 0.25,
                                   structure_weight: float = 0.60,
                                   novelty_weight: float = 0.15,
                                   score_mode: str = "bridge",
                                   non_anchor_title_dedup: bool = False) -> Tuple[List[int], Dict[str, object]]:
    candidate_count = len(pool_doc_ids)
    if candidate_count == 0 or qa_top_k <= 0:
        return [], {
            "selection_steps": [],
            "anchor_positions": [],
            "reserved_positions": [],
            "seed_entity_count_initial": 0,
            "covered_entity_count_final": 0,
            "candidate_count": candidate_count,
            "reserve_top_m": 0,
            "max_bridge_slots": int(max(0, max_bridge_slots)),
            "selection_target_k": 0,
            "non_anchor_title_dedup": bool(non_anchor_title_dedup),
        }

    normalized_base_scores = min_max_normalize_array(np.asarray(pool_doc_scores, dtype=float))
    target_k = min(candidate_count, qa_top_k)
    selected_positions: List[int] = []
    remaining_positions = list(range(candidate_count))
    covered_entities = normalize_entity_set(initial_seed_entities)
    selection_steps: List[Dict[str, object]] = []

    anchor_positions, reserved_positions = resolve_reserved_positions(
        candidate_count=candidate_count,
        target_k=target_k,
        anchor_count=anchor_count,
        reserve_top_m=reserve_top_m,
    )
    selection_target_k = resolve_selection_target_k(
        target_k=target_k,
        reserved_count=len(reserved_positions),
        max_bridge_slots=max_bridge_slots,
    )
    anchor_position_set = set(anchor_positions)
    blocked_titles: Set[str] = set()
    for reserved_pos in reserved_positions:
        selected_positions.append(reserved_pos)
        doc_id = pool_doc_ids[reserved_pos]
        if doc_id is not None:
            covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))
        if pool_doc_titles is not None and reserved_pos < len(pool_doc_titles):
            reserved_title = str(pool_doc_titles[reserved_pos]).strip()
            if reserved_title:
                blocked_titles.add(reserved_title)
        selection_steps.append({
            "step": len(selection_steps) + 1,
            "mode": "anchor" if reserved_pos in anchor_position_set else "reserve",
            "pool_position": int(reserved_pos),
            "doc_id": int(doc_id) if doc_id is not None else None,
            "base_score": round(float(normalized_base_scores[reserved_pos]), 4),
            "structure_score": 0.0,
            "novelty_score": 0.0,
            "combined_score": round(float(normalized_base_scores[reserved_pos]), 4),
        })

    remaining_positions = [pos for pos in remaining_positions if pos not in set(selected_positions)]

    while len(selected_positions) < selection_target_k and remaining_positions:
        scored_candidates = score_bridge_candidates(
            pool_doc_ids=pool_doc_ids,
            normalized_base_scores=normalized_base_scores,
            pool_doc_titles=pool_doc_titles,
            doc_idx_to_entities=doc_idx_to_entities,
            doc_idx_to_edges=doc_idx_to_edges,
            adjacency=adjacency,
            remaining_positions=remaining_positions,
            covered_entities=covered_entities,
            query_entities=query_entities or initial_seed_entities,
            structure_max_hops=structure_max_hops,
            structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
            base_weight=base_weight,
            structure_weight=structure_weight,
            novelty_weight=novelty_weight,
            score_mode=score_mode,
        )
        scored_candidates = filter_title_dedup_candidates(
            scored_candidates=scored_candidates,
            blocked_titles=blocked_titles,
            enabled=non_anchor_title_dedup,
        )
        best_detail = scored_candidates[0]
        best_pos = int(best_detail["pool_position"])

        selected_positions.append(best_pos)
        remaining_positions.remove(best_pos)
        chosen_doc_id = pool_doc_ids[best_pos]
        if chosen_doc_id is not None:
            covered_entities.update(normalize_entity_set(doc_idx_to_entities.get(int(chosen_doc_id), set())))
        chosen_title = str(best_detail.get("doc_title", "")).strip()
        if chosen_title:
            blocked_titles.add(chosen_title)
        selection_steps.append({
            "step": len(selection_steps) + 1,
            "mode": "greedy",
            "pool_position": int(best_pos),
            "doc_id": int(chosen_doc_id) if chosen_doc_id is not None else None,
            "base_score": float(best_detail["base_score"]),
            "structure_score": float(best_detail["structure_score"]),
            "novelty_score": float(best_detail["novelty_score"]),
            "closure_score": float(best_detail["closure_score"]),
            "selection_score": float(best_detail["selection_score"]),
            "combined_score": float(best_detail["combined_score"]),
        })

    return selected_positions, {
        "selection_steps": selection_steps,
        "anchor_positions": [int(pos) for pos in anchor_positions],
        "reserved_positions": [int(pos) for pos in reserved_positions],
        "seed_entity_count_initial": len(normalize_entity_set(initial_seed_entities)),
        "covered_entity_count_final": len(covered_entities),
        "candidate_count": candidate_count,
        "reserve_top_m": int(max(max(anchor_count, 0), max(reserve_top_m, 0))),
        "max_bridge_slots": int(max(0, max_bridge_slots)),
        "selection_target_k": int(selection_target_k),
        "score_mode": normalize_setwise_score_mode(score_mode),
        "non_anchor_title_dedup": bool(non_anchor_title_dedup),
    }


def select_bridge_beam_positions(pool_doc_ids: Sequence[int | None],
                                 pool_doc_scores: np.ndarray,
                                 pool_doc_titles: Sequence[str] | None,
                                 doc_idx_to_entities: Dict[int, Set[str]],
                                 doc_idx_to_edges: Dict[int, List[Tuple[str, str, float, str]]],
                                 adjacency: Dict[str, List[Tuple[str, float, str]]],
                                 qa_top_k: int,
                                 initial_seed_entities: Sequence[str] | Set[str] | None = None,
                                 query_entities: Sequence[str] | Set[str] | None = None,
                                 proposal_query_entities: Sequence[str] | Set[str] | None = None,
                                 state_query_entities: Sequence[str] | Set[str] | None = None,
                                 state_support_query_entities: Sequence[str] | Set[str] | None = None,
                                 anchor_count: int = 2,
                                 reserve_top_m: int = 0,
                                 max_bridge_slots: int = 0,
                                 structure_max_hops: int = 2,
                                 structure_seed_target_bridge_mode: str = "off",
                                 base_weight: float = 0.25,
                                 structure_weight: float = 0.60,
                                 novelty_weight: float = 0.15,
                                 score_mode: str = "bridge",
                                 beam_width: int = 4,
                                 beam_expand_per_state: int = 4,
                                 beam_projected_shortlist_factor: int = DEFAULT_SET_CLOSURE_PROJECTED_SHORTLIST_FACTOR,
                                 non_anchor_title_dedup: bool = False,
                                 state_weight_config: Dict[str, float] | None = None) -> Tuple[List[int], Dict[str, object]]:
    normalized_score_mode = normalize_setwise_score_mode(score_mode)
    uses_state_level_ranking = normalized_score_mode == "set_closure"
    candidate_count = len(pool_doc_ids)
    if candidate_count == 0 or qa_top_k <= 0:
        return [], {
            "selection_steps": [],
            "anchor_positions": [],
            "reserved_positions": [],
            "seed_entity_count_initial": 0,
            "covered_entity_count_final": 0,
            "candidate_count": candidate_count,
            "reserve_top_m": 0,
            "max_bridge_slots": int(max(0, max_bridge_slots)),
            "selection_target_k": 0,
            "score_mode": normalized_score_mode,
            "non_anchor_title_dedup": bool(non_anchor_title_dedup),
            "beam_width": int(max(beam_width, 1)),
            "beam_expand_per_state": int(max(beam_expand_per_state, 1)),
            "beam_projected_shortlist_factor": int(max(beam_projected_shortlist_factor, 1)),
            "beam_rank_metric": "state_score" if uses_state_level_ranking else "cumulative_score",
            "beam_best_state_score": 0.0,
            "beam_best_state_path_connectivity": 0.0,
            "beam_best_state_reachable_doc_ratio": 0.0,
            "beam_best_state_query_reachability": 0.0,
            "beam_best_state_suffix_base_mean": 0.0,
            "beam_set_closure_guard_skip_count": 0,
            "beam_projection_eval_count": 0,
            "beam_projection_extra_eval_count": 0,
            "beam_projection_rescue_count": 0,
            "beam_projection_changed_state_count": 0,
            "beam_projection_max_selected_rank": 0,
            "beam_finalists": [],
            "state_score_weights": dict(resolve_set_closure_state_weight_config(state_weight_config)),
        }

    normalized_base_scores = min_max_normalize_array(np.asarray(pool_doc_scores, dtype=float))
    target_k = min(candidate_count, qa_top_k)
    seed_entities = normalize_entity_set(initial_seed_entities)
    proposal_query_entities = (
        normalize_entity_set(proposal_query_entities)
        or normalize_entity_set(query_entities)
        or set(seed_entities)
    )
    state_query_entities = (
        normalize_entity_set(state_query_entities)
        or normalize_entity_set(query_entities)
        or set(seed_entities)
    )
    state_support_query_entities = (
        normalize_entity_set(state_support_query_entities)
        or set(proposal_query_entities)
        or set(state_query_entities)
        or set(seed_entities)
    )
    anchor_positions, reserved_positions = resolve_reserved_positions(
        candidate_count=candidate_count,
        target_k=target_k,
        anchor_count=anchor_count,
        reserve_top_m=reserve_top_m,
    )
    selection_target_k = resolve_selection_target_k(
        target_k=target_k,
        reserved_count=len(reserved_positions),
        max_bridge_slots=max_bridge_slots,
    )
    anchor_position_set = set(anchor_positions)

    initial_covered = seed_entities.copy()
    initial_steps: List[Dict[str, object]] = []
    initial_blocked_titles: Set[str] = set()
    for reserved_pos in reserved_positions:
        doc_id = pool_doc_ids[reserved_pos]
        if doc_id is not None:
            initial_covered.update(normalize_entity_set(doc_idx_to_entities.get(int(doc_id), set())))
        if pool_doc_titles is not None and reserved_pos < len(pool_doc_titles):
            reserved_title = str(pool_doc_titles[reserved_pos]).strip()
            if reserved_title:
                initial_blocked_titles.add(reserved_title)
        initial_steps.append({
            "step": len(initial_steps) + 1,
            "mode": "anchor" if reserved_pos in anchor_position_set else "reserve",
            "pool_position": int(reserved_pos),
            "doc_id": int(doc_id) if doc_id is not None else None,
            "base_score": round(float(normalized_base_scores[reserved_pos]), 4),
            "structure_score": 0.0,
            "novelty_score": 0.0,
            "combined_score": round(float(normalized_base_scores[reserved_pos]), 4),
        })

    beam_width = max(int(beam_width), 1)
    beam_expand_per_state = max(int(beam_expand_per_state), 1)
    beam_projected_shortlist_factor = max(int(beam_projected_shortlist_factor), 1)
    initial_state_metrics = score_evidence_state(
        pool_doc_ids=pool_doc_ids,
        normalized_base_scores=normalized_base_scores,
        pool_doc_titles=pool_doc_titles,
        doc_idx_to_entities=doc_idx_to_entities,
        doc_idx_to_edges=doc_idx_to_edges,
        adjacency=adjacency,
        selected_positions=reserved_positions,
        fixed_prefix_positions=reserved_positions,
        seed_entities=seed_entities,
        query_entities=state_query_entities,
        support_query_entities=state_support_query_entities,
        structure_max_hops=structure_max_hops,
        base_weight=base_weight,
        structure_weight=structure_weight,
        novelty_weight=novelty_weight,
        structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
        state_weight_config=state_weight_config,
    )
    beam_states: List[Dict[str, object]] = [{
        "selected_positions": list(reserved_positions),
        "covered_entities": initial_covered,
        "blocked_titles": set(initial_blocked_titles),
        "selection_steps": initial_steps,
        "cumulative_score": 0.0,
        "state_score": float(initial_state_metrics["state_score"]),
        "state_metrics": initial_state_metrics,
    }]
    seen_signatures = {tuple(reserved_positions)}
    set_closure_guard_skip_count = 0
    beam_projection_eval_count = 0
    beam_projection_extra_eval_count = 0
    beam_projection_rescue_count = 0
    beam_projection_changed_state_count = 0
    beam_projection_max_selected_rank = 0

    while beam_states and len(beam_states[0]["selected_positions"]) < selection_target_k:
        expanded_states: List[Dict[str, object]] = []
        for state in beam_states:
            selected_positions = list(state["selected_positions"])
            remaining_positions = [
                pos for pos in range(candidate_count)
                if pos not in set(selected_positions)
            ]
            if not remaining_positions:
                expanded_states.append(state)
                continue

            scored_candidates = score_bridge_candidates(
                pool_doc_ids=pool_doc_ids,
                normalized_base_scores=normalized_base_scores,
                pool_doc_titles=pool_doc_titles,
                doc_idx_to_entities=doc_idx_to_entities,
                doc_idx_to_edges=doc_idx_to_edges,
                adjacency=adjacency,
                remaining_positions=remaining_positions,
                covered_entities=state["covered_entities"],
                query_entities=proposal_query_entities,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                score_mode=score_mode,
            )
            scored_candidates = filter_title_dedup_candidates(
                scored_candidates=scored_candidates,
                blocked_titles=set(state["blocked_titles"]),
                enabled=non_anchor_title_dedup,
            )
            candidate_ranks = {
                int(candidate["pool_position"]): idx + 1
                for idx, candidate in enumerate(scored_candidates)
            }
            candidate_shortlist = (
                scored_candidates[:beam_expand_per_state * beam_projected_shortlist_factor]
                if uses_state_level_ranking
                else scored_candidates[:beam_expand_per_state]
            )
            projected_candidates: List[Dict[str, object]] = []
            for candidate in candidate_shortlist:
                chosen_pos = int(candidate["pool_position"])
                signature = tuple(selected_positions + [chosen_pos])
                if signature in seen_signatures:
                    continue

                next_covered = set(state["covered_entities"])
                next_covered.update(set(candidate["doc_entities"]))
                next_blocked_titles = set(state["blocked_titles"])
                chosen_title = str(candidate.get("doc_title", "")).strip()
                if chosen_title:
                    next_blocked_titles.add(chosen_title)
                next_cumulative_score = float(state["cumulative_score"]) + float(candidate["combined_score_raw"])
                next_state_metrics = score_evidence_state(
                    pool_doc_ids=pool_doc_ids,
                    normalized_base_scores=normalized_base_scores,
                    pool_doc_titles=pool_doc_titles,
                    doc_idx_to_entities=doc_idx_to_entities,
                    doc_idx_to_edges=doc_idx_to_edges,
                    adjacency=adjacency,
                    selected_positions=signature,
                    fixed_prefix_positions=reserved_positions,
                    seed_entities=seed_entities,
                    query_entities=state_query_entities,
                    support_query_entities=state_support_query_entities,
                    structure_max_hops=structure_max_hops,
                    base_weight=base_weight,
                    structure_weight=structure_weight,
                    novelty_weight=novelty_weight,
                    structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                    state_weight_config=state_weight_config,
                )
                if uses_state_level_ranking and not should_keep_set_closure_expansion(
                    candidate=candidate,
                    current_state_metrics=state.get("state_metrics"),
                    next_state_metrics=next_state_metrics,
                ):
                    set_closure_guard_skip_count += 1
                    continue
                projected_candidates.append({
                    "candidate": candidate,
                    "proposal_rank": int(candidate_ranks.get(chosen_pos, 0)),
                    "signature": signature,
                    "next_covered": next_covered,
                    "next_blocked_titles": next_blocked_titles,
                    "next_cumulative_score": next_cumulative_score,
                    "next_state_metrics": next_state_metrics,
                })

            if uses_state_level_ranking:
                projected_candidates.sort(
                    key=lambda item: (
                        -float(item["next_state_metrics"]["state_score"]),
                        -float(item["next_cumulative_score"]),
                        -len(item["next_covered"]),
                        int(item["proposal_rank"]),
                        int(item["candidate"]["pool_position"]),
                    )
                )

            selected_projected_candidates = projected_candidates[:beam_expand_per_state]
            if uses_state_level_ranking and beam_projected_shortlist_factor > 1:
                beam_projection_eval_count += len(projected_candidates)
                beam_projection_extra_eval_count += max(0, len(projected_candidates) - beam_expand_per_state)
                rescued_selected_candidates = [
                    projected for projected in selected_projected_candidates
                    if int(projected["proposal_rank"]) > beam_expand_per_state
                ]
                if rescued_selected_candidates:
                    beam_projection_changed_state_count += 1
                beam_projection_rescue_count += len(rescued_selected_candidates)
                beam_projection_max_selected_rank = max(
                    beam_projection_max_selected_rank,
                    max((int(projected["proposal_rank"]) for projected in selected_projected_candidates), default=0),
                )

            for projected in selected_projected_candidates:
                candidate = projected["candidate"]
                proposal_rank = int(projected["proposal_rank"])
                signature = projected["signature"]
                next_covered = projected["next_covered"]
                next_blocked_titles = projected["next_blocked_titles"]
                next_cumulative_score = float(projected["next_cumulative_score"])
                next_state_metrics = projected["next_state_metrics"]
                chosen_pos = int(candidate["pool_position"])
                seen_signatures.add(signature)
                expanded_states.append({
                    "selected_positions": list(signature),
                    "covered_entities": next_covered,
                    "blocked_titles": next_blocked_titles,
                    "selection_steps": list(state["selection_steps"]) + [{
                        "step": len(state["selection_steps"]) + 1,
                        "mode": "beam",
                        "pool_position": chosen_pos,
                        "doc_id": int(candidate["doc_id"]) if candidate["doc_id"] is not None else None,
                        "base_score": float(candidate["base_score"]),
                        "structure_score": float(candidate["structure_score"]),
                        "novelty_score": float(candidate["novelty_score"]),
                        "closure_score": float(candidate["closure_score"]),
                        "selection_score": float(candidate["selection_score"]),
                        "combined_score": float(candidate["combined_score"]),
                        "proposal_rank": proposal_rank,
                        "state_score": float(next_state_metrics["state_score"]),
                    }],
                    "cumulative_score": next_cumulative_score,
                    "state_score": float(next_state_metrics["state_score"]),
                    "state_metrics": next_state_metrics,
                })

        if not expanded_states:
            break

        expanded_states.sort(
            key=lambda state: (
                -float(state["state_score"] if uses_state_level_ranking else state["cumulative_score"]),
                -float(state["cumulative_score"]),
                -len(state["covered_entities"]),
                tuple(int(pos) for pos in state["selected_positions"]),
            )
        )
        beam_states = expanded_states[:beam_width]

    if beam_states:
        best_state = max(
            beam_states,
            key=lambda state: (
                float(state["state_score"] if uses_state_level_ranking else state["cumulative_score"]),
                float(state["cumulative_score"]),
                len(state["covered_entities"]),
                tuple(-int(pos) for pos in state["selected_positions"]),
            ),
        )
        ranked_finalists = sorted(
            beam_states,
            key=lambda state: (
                -float(state["state_score"] if uses_state_level_ranking else state["cumulative_score"]),
                -float(state["cumulative_score"]),
                -len(state["covered_entities"]),
                tuple(int(pos) for pos in state["selected_positions"]),
            ),
        )
    else:
        best_state = {
            "selected_positions": list(reserved_positions),
            "covered_entities": initial_covered,
            "blocked_titles": set(initial_blocked_titles),
            "selection_steps": initial_steps,
            "cumulative_score": 0.0,
            "state_score": float(initial_state_metrics["state_score"]),
            "state_metrics": initial_state_metrics,
        }
        ranked_finalists = [best_state]

    beam_finalists: List[Dict[str, object]] = []
    for finalist in ranked_finalists:
        finalist_positions = [int(pos) for pos in finalist["selected_positions"]]
        beam_finalists.append({
            "source": "beam_finalist",
            "selected_positions": finalist_positions,
            "selected_doc_ids": [
                int(pool_doc_ids[pos]) if 0 <= pos < len(pool_doc_ids) and pool_doc_ids[pos] is not None else None
                for pos in finalist_positions
            ],
            "selected_titles": [
                str(pool_doc_titles[pos]).strip()
                for pos in finalist_positions
                if pool_doc_titles is not None and 0 <= pos < len(pool_doc_titles)
            ],
            "cumulative_score": round(float(finalist["cumulative_score"]), 4),
            "state_score": round(float(finalist["state_score"]), 4),
            "path_connectivity": round(float(finalist["state_metrics"]["path_connectivity"]), 4),
            "reachable_doc_ratio": round(float(finalist["state_metrics"]["reachable_doc_ratio"]), 4),
            "query_reachability": round(float(finalist["state_metrics"]["query_reachability"]), 4),
            "query_coverage": round(float(finalist["state_metrics"]["query_coverage"]), 4),
            "support_mean": round(float(finalist["state_metrics"]["support_mean"]), 4),
            "suffix_base_mean": round(float(finalist["state_metrics"]["suffix_base_mean"]), 4),
        })

    return list(best_state["selected_positions"]), {
        "selection_steps": list(best_state["selection_steps"]),
        "anchor_positions": [int(pos) for pos in anchor_positions],
        "reserved_positions": [int(pos) for pos in reserved_positions],
        "seed_entity_count_initial": len(seed_entities),
        "covered_entity_count_final": len(set(best_state["covered_entities"])),
        "candidate_count": candidate_count,
        "reserve_top_m": int(max(max(anchor_count, 0), max(reserve_top_m, 0))),
        "max_bridge_slots": int(max(0, max_bridge_slots)),
        "selection_target_k": int(selection_target_k),
        "score_mode": normalized_score_mode,
        "non_anchor_title_dedup": bool(non_anchor_title_dedup),
        "beam_width": beam_width,
        "beam_expand_per_state": beam_expand_per_state,
        "beam_projected_shortlist_factor": beam_projected_shortlist_factor,
        "beam_finalist_count": len(beam_states),
        "beam_rank_metric": "state_score" if uses_state_level_ranking else "cumulative_score",
        "beam_best_cumulative_score": round(float(best_state["cumulative_score"]), 4),
        "beam_best_state_score": round(float(best_state["state_score"]), 4),
        "beam_best_state_path_connectivity": round(float(best_state["state_metrics"]["path_connectivity"]), 4),
        "beam_best_state_reachable_doc_ratio": round(float(best_state["state_metrics"]["reachable_doc_ratio"]), 4),
        "beam_best_state_query_reachability": round(float(best_state["state_metrics"]["query_reachability"]), 4),
        "beam_best_state_support_mean": round(float(best_state["state_metrics"]["support_mean"]), 4),
        "beam_best_state_suffix_base_mean": round(float(best_state["state_metrics"]["suffix_base_mean"]), 4),
        "beam_best_state_query_coverage": round(float(best_state["state_metrics"]["query_coverage"]), 4),
        "beam_set_closure_guard_skip_count": int(set_closure_guard_skip_count),
        "beam_projection_eval_count": int(beam_projection_eval_count),
        "beam_projection_extra_eval_count": int(beam_projection_extra_eval_count),
        "beam_projection_rescue_count": int(beam_projection_rescue_count),
        "beam_projection_changed_state_count": int(beam_projection_changed_state_count),
        "beam_projection_max_selected_rank": int(beam_projection_max_selected_rank),
        "beam_finalists": beam_finalists,
        "state_score_weights": dict(resolve_set_closure_state_weight_config(state_weight_config)),
    }


def apply_setwise_selector(hipporag: HippoRAG,
                           query_solutions: List[QuerySolution],
                           doc_text_to_chunk_id: Dict[str, str],
                           pool_k: int,
                           qa_top_k: int,
                           selector_name: str,
                           score_mode: str,
                           anchor_count: int,
                           reserve_top_m: int,
                           max_bridge_slots: int,
                           structure_max_hops: int,
                           base_weight: float,
                           structure_weight: float,
                           novelty_weight: float,
                           structure_seed_target_bridge_mode: str = "off",
                           learned_model_bundle: Dict[str, object] | None = None,
                           beam_width: int = 4,
                           beam_expand_per_state: int = 4,
                           beam_projected_shortlist_factor: int = DEFAULT_SET_CLOSURE_PROJECTED_SHORTLIST_FACTOR,
                           non_anchor_title_dedup: bool = False,
                           query_entity_source: str = "seed",
                           gate_mode: str = "none",
                           gate_min_structure_score: float = 0.15,
                           gate_min_combined_margin: float = 0.0,
                           gate_min_closure_score: float = 0.0,
                           gate_min_novelty_score: float = 0.0,
                           gate_min_frontier_gain: float = 0.0,
                           gate_min_path_coherence: float = 0.0,
                           gate_max_avg_local_structure: float = 0.95,
                           gate_min_suffix_base_mean: float = 0.15,
                           setwise_reader_order_probe_mode: str = "none",
                           state_weight_config: Dict[str, float] | None = None,
                           late_rerank_enabled: bool = False,
                           late_rerank_candidate_count: int = 4,
                           late_rerank_include_baseline: bool = True,
                           late_rerank_doc_char_limit: int = 280,
                           late_rerank_policy: str = "always",
                           late_rerank_max_state_score_gap: float = 0.0,
                           late_rerank_judge_bundle: SetwiseLateRerankJudgeBundle | None = None,
                           requirement_selector_bundle: Dict[str, object] | None = None,
                           requirement_reserve_policy: str = DEFAULT_REQUIREMENT_BEAM_RESERVE_POLICY,
                           expand_base_k: int = 10,
                           expand_min_structure_score: float = 0.35,
                           assemble_mode: str = "cross_encoder",
                           coverage_score_variant: str = "qe_ce",
                           coverage_atom_source: str = "candidate_pool",
                           coverage_admissibility_mode: str = "off",
                           append_max_docs: int = 3,
                           append_policy: str = "bridge",
                           append_random_seed: int = 0,
                           gap_expand_mode: str = "heuristic",
                           gap_expand_max_queries: int | None = None,
                           ce_model: str = "/mnt/nvme/bge-reranker-v2-m3",
                           ce_device: str = "cuda:1") -> Tuple[List[QuerySolution], Dict[str, object]]:
    logger = logging.getLogger(__name__)
    selector_name = str(selector_name).strip().lower()
    score_mode = normalize_setwise_score_mode(score_mode)
    normalized_assemble_mode = normalize_assemble_mode(assemble_mode)
    normalized_coverage_score_variant = normalize_coverage_score_variant(coverage_score_variant)
    normalized_coverage_atom_source = normalize_coverage_atom_source(coverage_atom_source)
    normalized_coverage_admissibility_mode = normalize_coverage_admissibility_mode(coverage_admissibility_mode)
    normalized_append_policy = normalize_append_policy(append_policy)
    normalized_gap_expand_mode = normalize_gap_expand_mode(gap_expand_mode)
    effective_gap_expand_max_queries = max(int(gap_expand_max_queries or DEFAULT_GAP_EXPAND_MAX_QUERIES), 1)
    if selector_name not in {"bridge_greedy", "bridge_beam", "bridge_append", "learned_greedy", "requirement_beam"}:
        raise ValueError(f"Unsupported setwise selector: {selector_name}")

    selected_solutions: List[QuerySolution] = []
    mapped_pool_doc_counts: List[int] = []
    seed_entity_counts: List[int] = []
    selected_structured_doc_counts: List[int] = []
    selector_examples: List[Dict[str, object]] = []
    gate_reason_counts: Counter[str] = Counter()
    gate_apply_count = 0
    gate_skip_count = 0
    saturation_guard_apply_count = 0
    saturation_guard_skip_count = 0
    late_rerank_apply_count = 0
    late_rerank_override_count = 0
    late_rerank_block_count = 0
    late_rerank_parse_failure_count = 0
    late_rerank_error_count = 0
    action_swap_apply_count = 0
    action_swap_keep_count = 0
    action_swap_judge_count = 0
    reader_order_probe_apply_count = 0
    reader_order_probe_skip_count = 0
    reader_order_probe_reason_counts: Counter[str] = Counter()
    beam_projection_eval_count = 0
    beam_projection_extra_eval_count = 0
    beam_projection_rescue_count = 0
    beam_projection_changed_query_count = 0
    beam_projection_max_selected_rank = 0
    requirement_support_scores: List[float] = []
    requirement_leakage_scores: List[float] = []
    requirement_frontier_sizes: List[float] = []
    requirement_cache_hit_count = 0
    requirement_runtime_anchor_counts: List[int] = []
    requirement_runtime_reserve_counts: List[int] = []
    requirement_reserve_reason_counts: Counter[str] = Counter()
    requirement_live_annotation_apply_count = 0
    requirement_live_source_expand_apply_count = 0
    requirement_probe_force_pool_gold_enabled_count = 0
    requirement_probe_force_pool_gold_in_pool_query_count = 0
    requirement_probe_force_pool_gold_applied_query_count = 0
    requirement_probe_force_pool_gold_applied_title_count = 0
    appended_doc_counts: List[int] = []
    expand_candidate_sizes: List[int] = []
    append_stop_reason_counts: Counter[str] = Counter()
    gap_type_counts: Counter[str] = Counter()
    gap_slot_counts: Counter[str] = Counter()
    gap_abstain_reason_counts: Counter[str] = Counter()
    gap_expand_query_count = 0
    gap_expand_fallback_query_count = 0
    gap_micro_query_counts: List[int] = []
    gap_candidate_counts: List[int] = []
    gap_candidates_selected_into_final_counts: List[int] = []
    normalized_late_rerank_policy = normalize_setwise_late_rerank_policy(late_rerank_policy)
    normalized_reader_order_probe_mode = normalize_setwise_reader_order_probe_mode(
        setwise_reader_order_probe_mode
    )

    chunk_text_to_hash = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}
    assemble_reranker = None
    bridge_append_unit_gap_needs_ce = (
        selector_name == "bridge_append"
        and normalized_append_policy == "gap_expand"
        and normalized_gap_expand_mode == "unit_typed_abstain"
    )
    if selector_name == "bridge_append" and normalized_assemble_mode == "embedding_similarity":
        if hasattr(hipporag, "_get_passage_query_embeddings"):
            hipporag._get_passage_query_embeddings(query_solutions)
    if selector_name == "bridge_append" and (
        normalized_assemble_mode in ASSEMBLE_CE_ACTIVE_MODES
        or bridge_append_unit_gap_needs_ce
    ):
        from FlagEmbedding import FlagReranker

        logger.info("Loading assemble cross-encoder model: %s on %s", ce_model, ce_device)
        assemble_reranker = FlagReranker(
            ce_model,
            use_fp16=True,
            devices=[str(ce_device)],
        )

    for q_idx, qs in enumerate(query_solutions):
        pool_limit = min(len(qs.docs), max(pool_k, qa_top_k))
        pool_docs = list(qs.docs[:pool_limit])
        pool_titles = [extract_doc_title(doc_text) for doc_text in pool_docs]
        if qs.doc_scores is not None and len(qs.doc_scores) >= pool_limit:
            pool_scores = np.asarray(qs.doc_scores[:pool_limit], dtype=float)
            tail_scores = np.asarray(qs.doc_scores[pool_limit:], dtype=float) if len(qs.doc_scores) > pool_limit else np.array([], dtype=float)
        else:
            pool_scores = np.linspace(pool_limit, 1, pool_limit, dtype=float)
            tail_scores = np.array([], dtype=float)

        pool_doc_ids: List[int | None] = []
        for doc_text in pool_docs:
            chunk_id = doc_text_to_chunk_id.get(doc_text) or chunk_text_to_hash.get(doc_text)
            mapped_doc_id = hipporag.passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
            pool_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)

        seed_entities = collect_query_seed_entities(hipporag, qs.question)
        if not seed_entities:
            seed_entities = collect_lexical_query_seed_entities(
                query=qs.question,
                pool_doc_ids=pool_doc_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )
        question_entities = collect_question_query_entities(
            hipporag=hipporag,
            query=qs.question,
            pool_doc_ids=pool_doc_ids,
            doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
        )
        normalized_query_entity_source = str(query_entity_source or "seed").strip().lower()
        query_target_config = resolve_setwise_query_targets(
            query_entity_source=normalized_query_entity_source,
            seed_entities=seed_entities,
            question_entities=question_entities,
            pool_doc_ids=pool_doc_ids,
            doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            adjacency=hipporag.structure_graph_out,
            structure_max_hops=structure_max_hops,
        )
        proposal_query_entities = set(query_target_config["proposal_query_entities"])
        state_query_entities = set(query_target_config["state_query_entities"])
        state_support_query_entities = set(query_target_config["state_support_query_entities"])
        grounded_question_entities = set(query_target_config["grounded_question_entities"])
        gate_decision = {
            "gate_mode": str(gate_mode or "none").strip().lower(),
            "gate_enabled": False,
            "use_selector": True,
            "reason": "disabled",
        }
        if selector_name in {"bridge_greedy", "bridge_beam"}:
            gate_decision = compute_bridge_gate_decision(
                pool_doc_ids=pool_doc_ids,
                pool_doc_scores=pool_scores,
                pool_doc_titles=pool_titles,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                adjacency=hipporag.structure_graph_out,
                qa_top_k=qa_top_k,
                initial_seed_entities=seed_entities,
                query_entities=proposal_query_entities,
                anchor_count=anchor_count,
                reserve_top_m=reserve_top_m,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                score_mode=score_mode,
                max_bridge_slots=max_bridge_slots,
                gate_mode=gate_mode,
                gate_min_structure_score=gate_min_structure_score,
                gate_min_combined_margin=gate_min_combined_margin,
                gate_min_closure_score=gate_min_closure_score,
                gate_min_novelty_score=gate_min_novelty_score,
                gate_min_frontier_gain=gate_min_frontier_gain,
                gate_min_path_coherence=gate_min_path_coherence,
                non_anchor_title_dedup=non_anchor_title_dedup,
            )
            if gate_decision.get("gate_enabled", False):
                if str(gate_mode or "none").strip().lower() != "suffix_bridge_saturation_guard":
                    gate_reason_counts[str(gate_decision.get("reason", "unknown"))] += 1
                    if gate_decision.get("use_selector", False):
                        gate_apply_count += 1
                    else:
                        gate_skip_count += 1
        if selector_name == "bridge_greedy":
            if gate_decision.get("use_selector", True):
                selected_positions, selector_trace = select_bridge_greedy_positions(
                    pool_doc_ids=pool_doc_ids,
                    pool_doc_scores=pool_scores,
                    pool_doc_titles=pool_titles,
                    doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                    doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                    adjacency=hipporag.structure_graph_out,
                    qa_top_k=qa_top_k,
                    initial_seed_entities=seed_entities,
                    query_entities=proposal_query_entities,
                    anchor_count=anchor_count,
                    reserve_top_m=reserve_top_m,
                    max_bridge_slots=max_bridge_slots,
                    structure_max_hops=structure_max_hops,
                    structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                    base_weight=base_weight,
                    structure_weight=structure_weight,
                    novelty_weight=novelty_weight,
                    score_mode=score_mode,
                    non_anchor_title_dedup=non_anchor_title_dedup,
                )
            else:
                selected_positions, selector_trace = [], {
                    "selection_steps": [],
                    "anchor_positions": [],
                    "reserved_positions": [],
                    "seed_entity_count_initial": len(seed_entities),
                    "covered_entity_count_final": len(seed_entities),
                    "candidate_count": len(pool_doc_ids),
                    "reserve_top_m": int(max(max(anchor_count, 0), max(reserve_top_m, 0))),
                    "max_bridge_slots": int(max(0, max_bridge_slots)),
                    "selection_target_k": int(gate_decision.get("selection_target_k", min(pool_limit, qa_top_k))),
                    "non_anchor_title_dedup": bool(non_anchor_title_dedup),
                }
        elif selector_name == "bridge_beam":
            if gate_decision.get("use_selector", True):
                selected_positions, selector_trace = select_bridge_beam_positions(
                    pool_doc_ids=pool_doc_ids,
                    pool_doc_scores=pool_scores,
                    pool_doc_titles=pool_titles,
                    doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                    doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                    adjacency=hipporag.structure_graph_out,
                    qa_top_k=qa_top_k,
                    initial_seed_entities=seed_entities,
                    query_entities=proposal_query_entities,
                    proposal_query_entities=proposal_query_entities,
                    state_query_entities=state_query_entities,
                    state_support_query_entities=state_support_query_entities,
                    anchor_count=anchor_count,
                    reserve_top_m=reserve_top_m,
                    max_bridge_slots=max_bridge_slots,
                    structure_max_hops=structure_max_hops,
                    structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                    base_weight=base_weight,
                    structure_weight=structure_weight,
                    novelty_weight=novelty_weight,
                    score_mode=score_mode,
                    beam_width=beam_width,
                    beam_expand_per_state=beam_expand_per_state,
                    beam_projected_shortlist_factor=beam_projected_shortlist_factor,
                    non_anchor_title_dedup=non_anchor_title_dedup,
                    state_weight_config=state_weight_config,
                )
            else:
                selected_positions, selector_trace = [], {
                    "selection_steps": [],
                    "anchor_positions": [],
                    "reserved_positions": [],
                    "seed_entity_count_initial": len(seed_entities),
                    "covered_entity_count_final": len(seed_entities),
                    "candidate_count": len(pool_doc_ids),
                    "reserve_top_m": int(max(max(anchor_count, 0), max(reserve_top_m, 0))),
                    "max_bridge_slots": int(max(0, max_bridge_slots)),
                    "selection_target_k": int(gate_decision.get("selection_target_k", min(pool_limit, qa_top_k))),
                    "score_mode": score_mode,
                    "non_anchor_title_dedup": bool(non_anchor_title_dedup),
                    "beam_width": int(beam_width),
                    "beam_expand_per_state": int(beam_expand_per_state),
                    "beam_projected_shortlist_factor": int(beam_projected_shortlist_factor),
                    "beam_finalist_count": 0,
                    "beam_rank_metric": "state_score" if score_mode == "set_closure" else "cumulative_score",
                    "beam_best_cumulative_score": 0.0,
                    "beam_best_state_score": 0.0,
                    "beam_best_state_path_connectivity": 0.0,
                    "beam_best_state_reachable_doc_ratio": 0.0,
                    "beam_best_state_query_reachability": 0.0,
                    "beam_best_state_support_mean": 0.0,
                    "beam_best_state_suffix_base_mean": 0.0,
                    "beam_best_state_query_coverage": 0.0,
                    "beam_projection_eval_count": 0,
                    "beam_projection_extra_eval_count": 0,
                    "beam_projection_rescue_count": 0,
                    "beam_projection_changed_state_count": 0,
                    "beam_projection_max_selected_rank": 0,
                    "beam_finalists": [],
                    "state_score_weights": dict(resolve_set_closure_state_weight_config(state_weight_config)),
                }
        elif selector_name == "bridge_append":
            normalized_pool_scores = np.asarray(pool_scores, dtype=float)
            if normalized_pool_scores.size > 0:
                score_range = float(normalized_pool_scores.max() - normalized_pool_scores.min())
                normalized_pool_scores = (
                    (normalized_pool_scores - normalized_pool_scores.min()) / (score_range + 1e-9)
                    if score_range > 0
                    else np.ones_like(normalized_pool_scores)
                )
            position_sources = {
                int(pos): "baseline_prefix"
                for pos in range(min(max(int(expand_base_k), 0), pool_limit))
            }
            selected_positions, selector_trace = select_bridge_append_positions(
                pool_doc_ids=pool_doc_ids,
                normalized_base_scores=normalized_pool_scores,
                query=qs.question,
                pool_docs=pool_docs,
                pool_doc_titles=pool_titles,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                adjacency=hipporag.structure_graph_out,
                initial_seed_entities=seed_entities,
                query_entities=proposal_query_entities,
                pool_limit=pool_limit,
                expand_base_k=expand_base_k,
                append_max_docs=append_max_docs,
                expand_min_structure_score=expand_min_structure_score,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                score_mode=score_mode,
                non_anchor_title_dedup=non_anchor_title_dedup,
                append_policy=normalized_append_policy,
                append_random_seed=append_random_seed,
                gap_expand_mode=normalized_gap_expand_mode,
                gap_expand_max_queries=int(effective_gap_expand_max_queries),
                ce_reranker=assemble_reranker,
            )
            for pos in selector_trace.get("appended_positions", []) or []:
                position_sources[int(pos)] = f"append_{normalized_append_policy}"
            if normalized_assemble_mode == "coverage":
                coverage_atom_positions = None
                if normalized_coverage_atom_source in {"baseline_prefix", "baseline_anchored"}:
                    coverage_atom_positions = selector_trace.get("baseline_prefix_positions", []) or []
                reranked_positions, assemble_trace = assemble_coverage_exact_search(
                    query=qs.question,
                    pool_docs=pool_docs,
                    pool_doc_ids=pool_doc_ids,
                    pool_doc_scores=pool_scores,
                    candidate_positions=selected_positions,
                    qa_top_k=qa_top_k,
                    seed_entities=seed_entities,
                    doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                    doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                    ce_reranker=assemble_reranker,
                    position_sources=position_sources,
                    coverage_score_variant=normalized_coverage_score_variant,
                    coverage_atom_source=normalized_coverage_atom_source,
                    coverage_atom_positions=coverage_atom_positions,
                    coverage_admissibility_mode=normalized_coverage_admissibility_mode,
                )
            elif normalized_assemble_mode == "ce_local_repair":
                reranked_positions, assemble_trace = assemble_ce_local_repair(
                    query=qs.question,
                    pool_docs=pool_docs,
                    pool_doc_ids=pool_doc_ids,
                    pool_doc_scores=pool_scores,
                    candidate_positions=selected_positions,
                    qa_top_k=qa_top_k,
                    query_entities=grounded_question_entities or question_entities or seed_entities,
                    seed_entities=seed_entities,
                    doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                    doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                    hipporag=hipporag,
                    ce_reranker=assemble_reranker,
                    position_sources=position_sources,
                )
            else:
                score_assemble_mode = (
                    "cross_encoder"
                    if normalized_assemble_mode in ACTION_CONTROLLER_MODES else
                    normalized_assemble_mode
                )
                reranked_positions, assemble_trace = rerank_candidate_positions_for_assemble(
                    query=qs.question,
                    pool_docs=pool_docs,
                    pool_doc_ids=pool_doc_ids,
                    pool_doc_scores=pool_scores,
                    candidate_positions=selected_positions,
                    assemble_mode=score_assemble_mode,
                    hipporag=hipporag,
                    ce_reranker=assemble_reranker,
                    position_sources=position_sources,
                )
            selector_trace["assemble_trace"] = assemble_trace
            selector_trace["assemble_mode"] = normalized_assemble_mode
            selector_trace["coverage_score_variant"] = normalized_coverage_score_variant
            selector_trace["coverage_atom_source"] = normalized_coverage_atom_source
            selector_trace["coverage_admissibility_mode"] = normalized_coverage_admissibility_mode
            selector_trace["selected_positions_before_assemble"] = list(selected_positions)
            selected_positions = list(reranked_positions)
        elif selector_name == "learned_greedy":
            if learned_model_bundle is None:
                raise ValueError("learned_greedy selector requires a loaded model bundle")
            selected_positions, selector_trace = select_learned_greedy_positions(
                query=qs.question,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                pool_doc_scores=pool_scores,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                adjacency=hipporag.structure_graph_out,
                qa_top_k=qa_top_k,
                learned_model_bundle=learned_model_bundle,
                initial_seed_entities=seed_entities,
                anchor_count=anchor_count,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
            )
        else:
            if requirement_selector_bundle is None:
                raise ValueError("requirement_beam selector requires a loaded requirement selector bundle")
            requirement_cache = requirement_selector_bundle.get("cache")
            if requirement_cache is None:
                raise ValueError("requirement_beam selector bundle is missing cache payload")
            live_annotation_requested_pool_k = int(
                (requirement_selector_bundle or {}).get("live_annotation_pool_k", 0) or 0
            )
            live_annotation_score_mode = str(
                (requirement_selector_bundle or {}).get("live_annotation_score_mode", "heuristic")
            ).strip().lower() or "heuristic"
            live_atomic_scorer_bundle = requirement_selector_bundle.get("live_atomic_scorer_bundle")
            live_atomic_model_path = str(
                (requirement_selector_bundle or {}).get("live_atomic_model_path", "")
            ).strip()
            live_source_expand_factor = int(
                (requirement_selector_bundle or {}).get("live_source_expand_factor", 0) or 0
            )
            probe_force_source_titles = list((requirement_selector_bundle or {}).get("probe_force_source_titles", []) or [])
            probe_force_shortlist_titles = list((requirement_selector_bundle or {}).get("probe_force_shortlist_titles", []) or [])
            probe_force_final_titles = list((requirement_selector_bundle or {}).get("probe_force_final_titles", []) or [])
            probe_force_pool_gold_into_final = bool(
                (requirement_selector_bundle or {}).get("probe_force_pool_gold_into_final", False)
            )
            probe_source_sort_mode = str(
                (requirement_selector_bundle or {}).get("probe_source_sort_mode", DEFAULT_REQUIREMENT_SOURCE_SORT_MODE)
                or DEFAULT_REQUIREMENT_SOURCE_SORT_MODE
            )
            probe_source_support_gain_weight = float(
                (requirement_selector_bundle or {}).get("probe_source_support_gain_weight", 0.0) or 0.0
            )
            probe_shortlist_sort_mode = str(
                (requirement_selector_bundle or {}).get("probe_shortlist_sort_mode", DEFAULT_REQUIREMENT_SHORTLIST_SORT_MODE)
                or DEFAULT_REQUIREMENT_SHORTLIST_SORT_MODE
            )
            probe_bridge_bonus_mode = str(
                (requirement_selector_bundle or {}).get("probe_bridge_bonus_mode", "off") or "off"
            )
            probe_bridge_bonus_weight = float(
                (requirement_selector_bundle or {}).get("probe_bridge_bonus_weight", 0.0) or 0.0
            )
            trace_watch_titles = list((requirement_selector_bundle or {}).get("exposure_watch_titles", []) or [])
            pool_gold_force_payload = {
                "requested_titles": [],
                "in_pool_titles": [],
                "missing_titles": [],
                "positions": [],
            }
            if probe_force_pool_gold_into_final:
                requirement_probe_force_pool_gold_enabled_count += 1
                pool_gold_force_payload = resolve_query_pool_gold_titles(
                    pool_titles=pool_titles[:pool_limit],
                    gold_docs=qs.gold_docs,
                    pool_limit=pool_limit,
                )
                if pool_gold_force_payload["in_pool_titles"]:
                    requirement_probe_force_pool_gold_in_pool_query_count += 1
            cache_entry = resolve_requirement_cache_entry(
                cache_payload=requirement_cache,
                question=qs.question,
                query_index=q_idx,
            )
            pool_doc_entities = [
                hipporag.doc_idx_to_structure_entities.get(int(doc_id), set())
                if doc_id is not None else set()
                for doc_id in pool_doc_ids
            ]
            cache_annotation_pool_k = int(cache_entry.get("annotation_pool_k", 0) or 0)
            effective_live_annotation_pool_k = 0
            alignment_pool_titles = pool_titles
            alignment_pool_docs = pool_docs
            alignment_pool_doc_entities = pool_doc_entities
            if live_annotation_requested_pool_k > 0:
                effective_live_annotation_pool_k = min(
                    pool_limit,
                    max(cache_annotation_pool_k, live_annotation_requested_pool_k),
                )
                alignment_pool_titles = pool_titles[:effective_live_annotation_pool_k]
                alignment_pool_docs = pool_docs[:effective_live_annotation_pool_k]
                alignment_pool_doc_entities = pool_doc_entities[:effective_live_annotation_pool_k]
            needs_cache_alignment = (
                len(alignment_pool_titles) > cache_annotation_pool_k
            )
            try:
                validate_requirement_cache_entry(
                    cache_entry=cache_entry,
                    pool_titles=alignment_pool_titles,
                )
            except ValueError:
                needs_cache_alignment = True
            if needs_cache_alignment:
                cache_entry = align_requirement_cache_entry_to_pool(
                    cache_entry=cache_entry,
                    pool_titles=alignment_pool_titles,
                    pool_docs=alignment_pool_docs,
                    pool_doc_entities=alignment_pool_doc_entities,
                    atomic_scorer_bundle=live_atomic_scorer_bundle if live_annotation_score_mode == "hybrid" else None,
                    score_mode=live_annotation_score_mode,
                )
            effective_requirement_source_expand_factor = max(
                int(beam_projected_shortlist_factor),
                int(live_source_expand_factor),
            ) if live_source_expand_factor > 0 else int(beam_projected_shortlist_factor)
            effective_requirement_shortlist_limit = (
                int(beam_expand_per_state)
                if live_source_expand_factor > 0 else None
            )
            if needs_cache_alignment and effective_live_annotation_pool_k > 0:
                requirement_live_annotation_apply_count += 1
            if live_source_expand_factor > 0 and effective_requirement_source_expand_factor > int(beam_projected_shortlist_factor):
                requirement_live_source_expand_apply_count += 1
            requirement_cache_hit_count += 1
            runtime_reserve_config = resolve_requirement_beam_runtime_reserve_config(
                anchor_count=anchor_count,
                reserve_top_m=reserve_top_m,
                cache_entry=cache_entry,
                policy=requirement_reserve_policy,
            )
            selected_positions, selector_trace = select_requirement_beam_positions(
                pool_doc_ids=pool_doc_ids,
                pool_doc_scores=pool_scores,
                pool_doc_titles=pool_titles,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                doc_idx_to_edges=hipporag.doc_idx_to_structure_edges,
                adjacency=hipporag.structure_graph_out,
                qa_top_k=qa_top_k,
                cache_entry=cache_entry,
                initial_seed_entities=seed_entities,
                proposal_query_entities=proposal_query_entities,
                anchor_count=int(runtime_reserve_config["anchor_count"]),
                reserve_top_m=int(runtime_reserve_config["reserve_top_m"]),
                max_bridge_slots=max_bridge_slots,
                structure_max_hops=structure_max_hops,
                structure_seed_target_bridge_mode=structure_seed_target_bridge_mode,
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                beam_width=beam_width,
                beam_expand_per_state=beam_expand_per_state,
                beam_projected_shortlist_factor=effective_requirement_source_expand_factor,
                beam_candidate_shortlist_limit=effective_requirement_shortlist_limit,
                force_source_titles=probe_force_source_titles,
                force_shortlist_titles=probe_force_shortlist_titles,
                trace_watch_titles=trace_watch_titles,
                source_sort_mode=probe_source_sort_mode,
                source_support_gain_weight=probe_source_support_gain_weight,
                shortlist_sort_mode=probe_shortlist_sort_mode,
                bridge_bonus_mode=probe_bridge_bonus_mode,
                bridge_bonus_weight=probe_bridge_bonus_weight,
                non_anchor_title_dedup=non_anchor_title_dedup,
                requirement_mode=str(requirement_selector_bundle.get("mode", "oracle")),
                requirement_model_bundle=requirement_selector_bundle.get("model_bundle"),
                requirement_smooth_tau=float(requirement_selector_bundle.get("smooth_tau", DEFAULT_REQUIREMENT_SMOOTH_TAU)),
                requirement_counterfactual_tau=float(requirement_selector_bundle.get("counterfactual_tau", DEFAULT_REQUIREMENT_CF_TAU)),
            )
            selector_trace["cache_annotation_pool_k_before_live"] = int(cache_annotation_pool_k)
            selector_trace["live_annotation_applied"] = bool(needs_cache_alignment and effective_live_annotation_pool_k > 0)
            selector_trace["live_annotation_requested_pool_k"] = int(live_annotation_requested_pool_k)
            selector_trace["live_annotation_effective_pool_k"] = int(cache_entry.get("annotation_pool_k", 0) or 0)
            selector_trace["live_annotation_score_mode"] = str(live_annotation_score_mode)
            selector_trace["live_annotation_model_path"] = live_atomic_model_path or None
            selector_trace["live_source_expand_factor"] = int(live_source_expand_factor)
            selector_trace["effective_beam_projected_shortlist_factor"] = int(
                effective_requirement_source_expand_factor
            )
            selector_trace["effective_beam_candidate_shortlist_limit"] = int(
                effective_requirement_shortlist_limit or 0
            )
            selector_trace["probe_force_source_titles"] = list(unique_ordered_titles(probe_force_source_titles))
            selector_trace["probe_force_shortlist_titles"] = list(unique_ordered_titles(probe_force_shortlist_titles))
            selector_trace["probe_force_final_titles"] = list(unique_ordered_titles(probe_force_final_titles))
            selector_trace["probe_force_pool_gold_into_final"] = bool(probe_force_pool_gold_into_final)
            selector_trace["probe_force_pool_gold_titles_requested"] = list(pool_gold_force_payload["requested_titles"])
            selector_trace["probe_force_pool_gold_titles_in_pool"] = list(pool_gold_force_payload["in_pool_titles"])
            selector_trace["probe_force_pool_gold_titles_missing_from_pool"] = list(pool_gold_force_payload["missing_titles"])
            selector_trace["probe_source_sort_mode"] = normalize_requirement_source_sort_mode(probe_source_sort_mode)
            selector_trace["probe_source_support_gain_weight"] = round(float(probe_source_support_gain_weight), 4)
            selector_trace["probe_shortlist_sort_mode"] = normalize_requirement_shortlist_sort_mode(probe_shortlist_sort_mode)
            selector_trace["probe_bridge_bonus_mode"] = normalize_requirement_bridge_bonus_mode(probe_bridge_bonus_mode)
            selector_trace["probe_bridge_bonus_weight"] = round(float(probe_bridge_bonus_weight), 4)
            selector_trace["requirement_reserve_policy"] = str(runtime_reserve_config["policy"])
            selector_trace["requirement_reserve_policy_reason"] = str(runtime_reserve_config["policy_reason"])
            selector_trace["requirement_positive_count"] = int(runtime_reserve_config["positive_requirement_count"])
            selector_trace["runtime_anchor_count"] = int(runtime_reserve_config["anchor_count"])
            selector_trace["runtime_reserve_top_m"] = int(runtime_reserve_config["reserve_top_m"])
            selector_trace["runtime_effective_reserved_count"] = int(runtime_reserve_config["effective_reserved_count"])
            requirement_support_scores.append(float(selector_trace.get("beam_best_support_completeness", 0.0) or 0.0))
            requirement_leakage_scores.append(float(selector_trace.get("beam_best_counterfactual_leakage", 0.0) or 0.0))
            requirement_frontier_sizes.append(float(selector_trace.get("beam_avg_frontier_size", 0.0) or 0.0))
            requirement_runtime_anchor_counts.append(int(runtime_reserve_config["anchor_count"]))
            requirement_runtime_reserve_counts.append(int(runtime_reserve_config["effective_reserved_count"]))
            requirement_reserve_reason_counts[str(runtime_reserve_config["policy_reason"])] += 1

        if selector_name in {"bridge_greedy", "bridge_beam"}:
            guard_original_use_selector = bool(gate_decision.get("use_selector", False))
            heuristic_selected_positions, selector_trace, gate_decision = maybe_apply_bridge_saturation_guard(
                selected_positions=selected_positions,
                selector_trace=selector_trace,
                gate_decision=gate_decision,
                pool_doc_titles=pool_titles,
                gate_max_avg_local_structure=gate_max_avg_local_structure,
                gate_min_suffix_base_mean=gate_min_suffix_base_mean,
            )
            if bool(gate_decision.get("saturation_guard_triggered", False)):
                saturation_guard_skip_count += 1
            elif str(gate_mode or "none").strip().lower() == "suffix_bridge_saturation_guard" and guard_original_use_selector:
                saturation_guard_apply_count += 1

            if gate_decision.get("gate_enabled", False) and str(gate_mode or "none").strip().lower() == "suffix_bridge_saturation_guard":
                gate_reason_counts[str(gate_decision.get("reason", "unknown"))] += 1
                if gate_decision.get("use_selector", False):
                    gate_apply_count += 1
                else:
                    gate_skip_count += 1
        else:
            heuristic_selected_positions = [int(pos) for pos in selected_positions]

        action_trace: Dict[str, object] = {
            "action_mode": (
                normalized_assemble_mode
                if normalized_assemble_mode in ACTION_CONTROLLER_MODES else
                "off"
            ),
            "action_legality_mode": (
                resolve_action_swap_v0_legality_mode(normalized_assemble_mode)
                if normalized_assemble_mode in ACTION_SWAP_V0_MODES else
                ("dedup_only" if normalized_assemble_mode in ACTION_SWAP_NOISYOR_MODES else "off")
            ),
            "action_executed": False,
            "action_type": "keep" if normalized_assemble_mode in ACTION_CONTROLLER_MODES else "off",
            "action_candidate_pool_position": None,
            "action_candidate_doc_id": None,
            "action_replace_pool_position": None,
            "action_replace_doc_id": None,
            "action_score_delta": None,
            "action_gate_verdict": None,
            "action_gate_confidence": None,
            "action_gate_reason": None,
            "action_gate_source": None,
            "action_gate_score": None,
            "action_margin": None,
            "query_dependency_graph": {},
            "query_dependency_mode": None,
            "query_tiers": [],
            "query_tier_mode": None,
            "claims": [],
            "claim_mode": None,
            "proposal_action_present": False,
            "proposal_candidate_pool_position": None,
            "proposal_replace_pool_position": None,
            "earliest_unsupported_claim_id": None,
            "earliest_unsupported_claim_text": None,
            "gain_verifier_verdict": None,
            "gain_verifier_reason": None,
            "preservation_unique_support_claim_ids": [],
            "claim_supports_before": [],
            "facet_supports_before": [],
            "facet_supports_after": [],
            "incumbent_attributions": [],
            "executed_action_delta": None,
            "bottleneck_tier_index": None,
            "target_facet_id": None,
            "target_facet_text": None,
            "target_candidate_gain": None,
            "replacee_loss_earlier_tiers": None,
            "replacee_loss_same_tier": None,
            "replacee_loss_target_facet": None,
            "swap_gain_vs_loss": None,
            "candidate_best_witness": {},
            "action_legal_action_count": 0,
            "action_total_jobs": 0,
            "action_skip_reason": "disabled" if normalized_assemble_mode not in ACTION_CONTROLLER_MODES else "uninitialized",
            "final_front_positions_before_action": [],
            "final_front_positions_after_action": [],
        }
        if selector_name == "bridge_append" and normalized_assemble_mode in ACTION_CONTROLLER_MODES:
            ranking_rows = list((assemble_trace or {}).get("ranking_rows") or [])
            baseline_scaffold_positions = extract_baseline_scaffold_positions_from_ranking_rows(
                ranking_rows=ranking_rows,
                baseline_prefix_positions=selector_trace.get("baseline_prefix_positions", []) or [],
                qa_top_k=qa_top_k,
            )
            action_jobs = build_action_swap_jobs(
                query=qs.question,
                scaffold_positions=baseline_scaffold_positions,
                appended_positions=selector_trace.get("appended_positions", []) or [],
                ranking_rows=ranking_rows,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                doc_text_to_chunk_id=doc_text_to_chunk_id,
                replace_bottom_n=2,
            )
            if normalized_assemble_mode in ACTION_SWAP_V0_JUDGE_MODES:
                action_swap_judge_count += 1
                if late_rerank_judge_bundle is not None and late_rerank_judge_bundle.infer_fn is not None:
                    judge_bundle = late_rerank_judge_bundle
                else:
                    judge_bundle = SetwiseLateRerankJudgeBundle(
                        infer_fn=hipporag.llm_model.infer,
                        model_name=(
                            getattr(hipporag.global_config, "llm_request_name", None)
                            or hipporag.global_config.llm_name
                        ),
                        backend="inherit",
                        base_url=hipporag.global_config.llm_base_url,
                        response_format=None,
                    )
                action_trace = select_action_swap_v0_judge(
                    action_jobs=action_jobs,
                    scaffold_positions=baseline_scaffold_positions,
                    judge_bundle=judge_bundle,
                    qa_top_k=qa_top_k,
                    max_doc_chars=late_rerank_doc_char_limit,
                    action_mode=normalized_assemble_mode,
                )
            elif normalized_assemble_mode in ACTION_SWAP_PROPOSE_VERIFY_MODES:
                action_swap_judge_count += 1
                if late_rerank_judge_bundle is not None and late_rerank_judge_bundle.infer_fn is not None:
                    judge_bundle = late_rerank_judge_bundle
                else:
                    judge_bundle = SetwiseLateRerankJudgeBundle(
                        infer_fn=hipporag.llm_model.infer,
                        model_name=(
                            getattr(hipporag.global_config, "llm_request_name", None)
                            or hipporag.global_config.llm_name
                        ),
                        backend="inherit",
                        base_url=hipporag.global_config.llm_base_url,
                        response_format=None,
                    )
                action_trace = select_action_swap_propose_verify(
                    action_jobs=action_jobs,
                    query=qs.question,
                    scaffold_positions=baseline_scaffold_positions,
                    pool_docs=pool_docs,
                    query_entities=grounded_question_entities or question_entities or seed_entities,
                    ce_reranker=assemble_reranker,
                    verifier_bundle=judge_bundle,
                    max_doc_chars=late_rerank_doc_char_limit,
                    action_mode=normalized_assemble_mode,
                )
            elif normalized_assemble_mode in ACTION_SWAP_NOISYOR_MODES:
                action_trace = select_action_swap_noisyor(
                    action_jobs=action_jobs,
                    query=qs.question,
                    scaffold_positions=baseline_scaffold_positions,
                    pool_docs=pool_docs,
                    query_entities=grounded_question_entities or question_entities or seed_entities,
                    ce_reranker=assemble_reranker,
                    action_mode=normalized_assemble_mode,
                    action_margin=DEFAULT_ACTION_SWAP_NOISYOR_MARGIN,
                )
            elif normalized_assemble_mode in ACTION_SWAP_TIERED_WITNESS_MODES:
                action_trace = select_action_swap_tiered_witness(
                    action_jobs=action_jobs,
                    query=qs.question,
                    scaffold_positions=baseline_scaffold_positions,
                    pool_docs=pool_docs,
                    query_entities=grounded_question_entities or question_entities or seed_entities,
                    ce_reranker=assemble_reranker,
                    action_mode=normalized_assemble_mode,
                )
            else:
                action_trace = select_action_swap_v0_dryrun(
                    action_jobs=action_jobs,
                    scaffold_positions=baseline_scaffold_positions,
                    action_mode=normalized_assemble_mode,
                )
            selector_trace.update(action_trace)
            final_front_positions = [
                int(pos)
                for pos in action_trace.get("final_front_positions_after_action", [])
            ]
            if bool(action_trace.get("action_executed", False)):
                action_swap_apply_count += 1
            else:
                action_swap_keep_count += 1
        else:
            final_front_positions = materialize_reader_top_positions(
                selected_positions=heuristic_selected_positions,
                pool_limit=pool_limit,
                qa_top_k=qa_top_k,
            )
        late_rerank_trace: Dict[str, object] = {
            "enabled": bool(late_rerank_enabled),
            "applied": False,
            "override_applied": False,
            "candidate_count": 0,
            "override_policy": normalized_late_rerank_policy,
            "override_max_state_score_gap": float(late_rerank_max_state_score_gap),
        }
        if late_rerank_enabled and selector_name == "bridge_beam":
            if late_rerank_judge_bundle is not None and late_rerank_judge_bundle.infer_fn is not None:
                judge_bundle = late_rerank_judge_bundle
            else:
                judge_bundle = SetwiseLateRerankJudgeBundle(
                    infer_fn=hipporag.llm_model.infer,
                    model_name=hipporag.global_config.llm_name,
                    backend="inherit",
                    base_url=hipporag.global_config.llm_base_url,
                    response_format=None,
                )
            late_rerank_candidates = build_setwise_late_rerank_candidates(
                selected_positions=heuristic_selected_positions,
                selector_trace=selector_trace,
                pool_limit=pool_limit,
                qa_top_k=qa_top_k,
                include_baseline=late_rerank_include_baseline,
                max_candidates=late_rerank_candidate_count,
            )
            chosen_candidate_id, late_rerank_trace = rerank_completed_evidence_sets_with_llm(
                query=qs.question,
                pool_docs=pool_docs,
                candidate_sets=late_rerank_candidates,
                llm_infer_fn=judge_bundle.infer_fn,
                model_name=judge_bundle.model_name,
                response_format=judge_bundle.response_format,
                max_doc_chars=late_rerank_doc_char_limit,
            )
            late_rerank_trace["judge_backend"] = judge_bundle.backend
            late_rerank_trace["judge_model"] = judge_bundle.model_name
            late_rerank_trace["judge_base_url"] = judge_bundle.base_url
            late_rerank_trace["judge_reasoning_effort"] = judge_bundle.reasoning_effort
            late_rerank_apply_count += int(bool(late_rerank_trace.get("applied", False)))
            if late_rerank_trace.get("llm_error"):
                late_rerank_error_count += 1
            if late_rerank_trace.get("applied", False) and not late_rerank_trace.get("parse_succeeded", False):
                late_rerank_parse_failure_count += 1
            if chosen_candidate_id is not None and 0 <= chosen_candidate_id < len(late_rerank_candidates):
                chosen_candidate = late_rerank_candidates[chosen_candidate_id]
                chosen_front_positions = [
                    int(pos) for pos in chosen_candidate.get("reader_top_positions", [])
                ]
                late_rerank_trace["selected_candidate_source"] = str(chosen_candidate.get("source", "unknown"))
                late_rerank_trace["selected_reader_top_positions"] = list(chosen_front_positions)
                late_rerank_trace["selected_reader_top_titles"] = [
                    pool_titles[pos] for pos in chosen_front_positions
                ]
                heuristic_front_positions = materialize_reader_top_positions(
                    selected_positions=heuristic_selected_positions,
                    pool_limit=pool_limit,
                    qa_top_k=qa_top_k,
                )
                override_requested = chosen_front_positions != heuristic_front_positions
                late_rerank_trace["override_requested"] = bool(override_requested)
                if not override_requested:
                    final_front_positions = list(heuristic_front_positions)
                else:
                    override_allowed, override_info = should_apply_setwise_late_rerank_override(
                        heuristic_candidate=late_rerank_candidates[0],
                        chosen_candidate=chosen_candidate,
                        policy=normalized_late_rerank_policy,
                        max_state_score_gap=late_rerank_max_state_score_gap,
                    )
                    late_rerank_trace.update(override_info)
                    late_rerank_trace["override_applied"] = bool(override_allowed)
                    if override_allowed:
                        final_front_positions = list(chosen_front_positions)
                        late_rerank_override_count += 1
                    else:
                        final_front_positions = list(heuristic_front_positions)
                        late_rerank_block_count += 1

        forced_final_positions = []
        forced_final_titles_applied: List[str] = []
        forced_probe_final_titles_applied: List[str] = []
        forced_pool_gold_final_titles_applied: List[str] = []
        if selector_name == "requirement_beam":
            probe_force_final_positions = resolve_title_pool_positions(
                pool_titles=pool_titles[:pool_limit],
                target_titles=probe_force_final_titles,
                pool_limit=pool_limit,
            )
            gold_force_final_positions = [
                int(pos)
                for pos in list(pool_gold_force_payload.get("positions", []) or [])
            ]
            seen_forced_final_positions: Set[int] = set()
            for pos in list(probe_force_final_positions) + gold_force_final_positions:
                normalized_pos = int(pos)
                if normalized_pos in seen_forced_final_positions:
                    continue
                forced_final_positions.append(normalized_pos)
                seen_forced_final_positions.add(normalized_pos)
            if forced_final_positions:
                original_front_positions = list(final_front_positions)
                final_front_positions = materialize_reader_top_positions(
                    selected_positions=final_front_positions,
                    pool_limit=pool_limit,
                    qa_top_k=qa_top_k,
                    forced_prefix_positions=forced_final_positions,
                )
                forced_final_titles_applied = unique_ordered_titles(
                    [
                        pool_titles[pos]
                        for pos in final_front_positions
                        if pos in forced_final_positions and pos not in original_front_positions
                    ]
                )
                forced_probe_final_titles_applied = unique_ordered_titles(
                    [
                        pool_titles[pos]
                        for pos in final_front_positions
                        if pos in probe_force_final_positions and pos not in original_front_positions
                    ]
                )
                forced_pool_gold_final_titles_applied = unique_ordered_titles(
                    [
                        pool_titles[pos]
                        for pos in final_front_positions
                        if pos in gold_force_final_positions and pos not in original_front_positions
                    ]
                )
                if forced_pool_gold_final_titles_applied:
                    requirement_probe_force_pool_gold_applied_query_count += 1
                    requirement_probe_force_pool_gold_applied_title_count += len(
                        forced_pool_gold_final_titles_applied
                    )
            selector_trace["forced_final_titles_applied"] = list(forced_final_titles_applied)
            selector_trace["forced_probe_final_titles_applied"] = list(forced_probe_final_titles_applied)
            selector_trace["forced_pool_gold_final_titles_applied"] = list(forced_pool_gold_final_titles_applied)

        reader_order_probe_trace: Dict[str, object] = {
            "enabled": False,
            "mode": normalized_reader_order_probe_mode,
            "applied": False,
            "skip_reason": "disabled" if normalized_reader_order_probe_mode == "none" else "selector_not_bridge_beam",
            "target_rank": 2 if normalized_reader_order_probe_mode == "promote_best_bridge_to_slot2" else 3,
            "promoted_pool_position": None,
            "promoted_doc_id": None,
            "promoted_title": "",
            "promoted_from_rank": None,
            "original_front_pool_positions": list(final_front_positions),
            "original_front_titles": [pool_titles[pos] for pos in final_front_positions],
            "probed_front_pool_positions": list(final_front_positions),
            "probed_front_titles": [pool_titles[pos] for pos in final_front_positions],
        }
        if selector_name == "bridge_beam":
            final_front_positions, reader_order_probe_trace = maybe_apply_setwise_reader_order_probe(
                final_front_positions=final_front_positions,
                selector_trace=selector_trace,
                pool_doc_ids=pool_doc_ids,
                pool_doc_titles=pool_titles,
                probe_mode=normalized_reader_order_probe_mode,
            )
            if bool(reader_order_probe_trace.get("enabled", False)):
                if bool(reader_order_probe_trace.get("applied", False)):
                    reader_order_probe_apply_count += 1
                else:
                    reader_order_probe_skip_count += 1
                reader_order_probe_reason_counts[
                    str(reader_order_probe_trace.get("skip_reason", "") or "applied")
                ] += 1

        selected_position_set = set(final_front_positions)
        if selector_name == "bridge_append" and normalized_append_policy == "gap_expand":
            gap_expand_query_count += 1
            gap_type = str(selector_trace.get("gap_type", "none") or "none")
            gap_type_counts[gap_type] += 1
            gap_slot = str(selector_trace.get("gap_slot", "") or "")
            if gap_slot:
                gap_slot_counts[gap_slot] += 1
            if bool(selector_trace.get("gap_fallback_used", False)):
                gap_expand_fallback_query_count += 1
            gap_abstain_reason = str(selector_trace.get("gap_abstain_reason", "") or "")
            if gap_abstain_reason:
                gap_abstain_reason_counts[gap_abstain_reason] += 1
            gap_micro_query_counts.append(int(selector_trace.get("gap_micro_query_count", 0) or 0))
            gap_candidate_positions = [int(pos) for pos in selector_trace.get("gap_candidate_positions", []) or []]
            gap_candidate_position_set = set(gap_candidate_positions)
            gap_selected_positions = [
                int(pos) for pos in final_front_positions
                if int(pos) in gap_candidate_position_set
            ]
            selector_trace["gap_candidate_selected_into_final"] = list(gap_selected_positions)
            selector_trace["gap_candidate_selected_count"] = int(len(gap_selected_positions))
            selector_trace["gap_candidate_final_front_rate"] = round(
                float(len(gap_selected_positions)) / float(max(len(gap_candidate_positions), 1)),
                4,
            )
            ranking_rows = list((selector_trace.get("assemble_trace") or {}).get("ranking_rows") or [])
            ce_score_by_position = {
                int(row.get("pool_position", -1) or -1): float(row.get("assemble_score", row.get("ce_score", 0.0)) or 0.0)
                for row in ranking_rows
            }
            selector_trace["gap_candidate_ce_scores"] = [
                {
                    "pool_position": int(pos),
                    "ce_score": round(float(ce_score_by_position.get(int(pos), 0.0)), 4),
                }
                for pos in gap_candidate_positions
            ]
            gap_candidate_counts.append(int(len(gap_candidate_positions)))
            gap_candidates_selected_into_final_counts.append(int(len(gap_selected_positions)))
        reordered_pool_positions = final_front_positions + [
            pos for pos in range(pool_limit)
            if pos not in selected_position_set
        ]
        reordered_docs = [pool_docs[pos] for pos in reordered_pool_positions] + list(qs.docs[pool_limit:])
        reordered_scores = np.concatenate(
            [
                np.asarray([pool_scores[pos] for pos in reordered_pool_positions], dtype=float),
                tail_scores,
            ]
        )

        retrieval_trace = dict(qs.retrieval_trace or {})
        retrieval_trace["setwise_selector"] = selector_name
        if selector_name == "bridge_append":
            retrieval_trace["expand_assemble_trace"] = {
                "pool_k": int(pool_limit),
                "score_mode": score_mode,
                "expand_base_k": int(min(max(int(expand_base_k), 0), pool_limit)),
                "append_max_docs": int(max(0, append_max_docs)),
                "append_policy": normalized_append_policy,
                "append_random_seed": int(append_random_seed),
                "gap_expand_mode": normalized_gap_expand_mode,
                "gap_expand_max_queries": int(effective_gap_expand_max_queries),
                "expand_min_structure_score": round(float(expand_min_structure_score), 4),
                "assemble_mode": normalized_assemble_mode,
                "coverage_score_variant": normalized_coverage_score_variant,
                "coverage_atom_source": normalized_coverage_atom_source,
                "coverage_admissibility_mode": normalized_coverage_admissibility_mode,
                "assemble_ce_model": str(ce_model) if normalized_assemble_mode in ASSEMBLE_CE_ACTIVE_MODES else None,
                "assemble_ce_device": str(ce_device) if normalized_assemble_mode in ASSEMBLE_CE_ACTIVE_MODES else None,
                "non_anchor_title_dedup": bool(non_anchor_title_dedup),
                "query_entity_source": normalized_query_entity_source,
                "seed_entities_preview": sorted(seed_entities)[:12],
                "question_entities_preview": sorted(question_entities)[:12],
                "grounded_question_entities_preview": sorted(grounded_question_entities)[:12],
                "proposal_query_entities_preview": sorted(proposal_query_entities)[:12],
                "query_entities_preview": sorted(state_query_entities)[:12],
                "state_support_query_entities_preview": sorted(state_support_query_entities)[:12],
                "selected_pool_positions": list(heuristic_selected_positions),
                "selected_doc_ids": [
                    int(pool_doc_ids[pos]) if pool_doc_ids[pos] is not None else None
                    for pos in heuristic_selected_positions
                ],
                "selected_titles": [pool_titles[pos] for pos in heuristic_selected_positions],
                "final_front_pool_positions": list(final_front_positions),
                "final_front_doc_ids": [
                    int(pool_doc_ids[pos]) if pool_doc_ids[pos] is not None else None
                    for pos in final_front_positions
                ],
                "final_front_titles": [pool_titles[pos] for pos in final_front_positions],
                **selector_trace,
            }
        else:
            retrieval_trace["setwise_selector_trace"] = {
                "pool_k": int(pool_limit),
                "score_mode": score_mode,
                "anchor_count": int(min(max(anchor_count, 0), min(pool_limit, qa_top_k))),
                "reserve_top_m": int(min(max(max(anchor_count, 0), max(reserve_top_m, 0)), min(pool_limit, qa_top_k))),
                "max_bridge_slots": int(max(0, max_bridge_slots)),
                "non_anchor_title_dedup": bool(non_anchor_title_dedup),
                "query_entity_source": normalized_query_entity_source,
                "gate_decision": gate_decision,
                "seed_entities_preview": sorted(seed_entities)[:12],
                "question_entities_preview": sorted(question_entities)[:12],
                "grounded_question_entities_preview": sorted(grounded_question_entities)[:12],
                "proposal_query_entities_preview": sorted(proposal_query_entities)[:12],
                "query_entities_preview": sorted(state_query_entities)[:12],
                "state_support_query_entities_preview": sorted(state_support_query_entities)[:12],
                "selected_pool_positions": list(heuristic_selected_positions),
                "selected_doc_ids": [
                    int(pool_doc_ids[pos]) if pool_doc_ids[pos] is not None else None
                    for pos in heuristic_selected_positions
                ],
                "selected_titles": [pool_titles[pos] for pos in heuristic_selected_positions],
                "final_front_pool_positions": list(final_front_positions),
                "final_front_doc_ids": [
                    int(pool_doc_ids[pos]) if pool_doc_ids[pos] is not None else None
                    for pos in final_front_positions
                ],
                "final_front_titles": [pool_titles[pos] for pos in final_front_positions],
                "reader_order_probe": reader_order_probe_trace,
                "late_rerank_trace": late_rerank_trace,
                **selector_trace,
            }
        if selector_name == "requirement_beam":
            trace_target_titles = unique_ordered_titles(
                list(
                    parse_title_csv(
                        str((requirement_selector_bundle or {}).get("exposure_watch_titles_csv", "")),
                        default=requirement_selector_bundle.get("exposure_watch_titles"),
                    )
                )
                + list((requirement_selector_bundle or {}).get("probe_force_source_titles", []) or [])
                + list((requirement_selector_bundle or {}).get("probe_force_shortlist_titles", []) or [])
                + list((requirement_selector_bundle or {}).get("probe_force_final_titles", []) or [])
                + [
                    extract_doc_title(doc_text)
                    for doc_text in list(qs.gold_docs or [])
                ]
            )
            retrieval_trace["setwise_selector_trace"]["requirement_title_exposure_summary"] = (
                build_requirement_title_exposure_summary(
                    pool_titles=pool_titles,
                    selector_trace=retrieval_trace["setwise_selector_trace"],
                    target_titles=trace_target_titles,
                )
            )

        selected_qs = QuerySolution(
            question=qs.question,
            docs=reordered_docs,
            doc_scores=reordered_scores,
            gold_answers=qs.gold_answers,
            gold_docs=qs.gold_docs,
            retrieval_trace=retrieval_trace,
            qa_trace=qs.qa_trace,
        )
        selected_solutions.append(selected_qs)
        beam_projection_eval_count += int(selector_trace.get("beam_projection_eval_count", 0) or 0)
        beam_projection_extra_eval_count += int(selector_trace.get("beam_projection_extra_eval_count", 0) or 0)
        beam_projection_rescue_count += int(selector_trace.get("beam_projection_rescue_count", 0) or 0)
        if int(selector_trace.get("beam_projection_rescue_count", 0) or 0) > 0:
            beam_projection_changed_query_count += 1
        beam_projection_max_selected_rank = max(
            beam_projection_max_selected_rank,
            int(selector_trace.get("beam_projection_max_selected_rank", 0) or 0),
        )
        if selector_name == "bridge_append":
            appended_doc_count = int(selector_trace.get("append_count", 0) or 0)
            appended_doc_counts.append(appended_doc_count)
            expand_candidate_sizes.append(int(selector_trace.get("candidate_set_size", 0) or 0))
            append_stop_reason_counts[
                str(selector_trace.get("append_stop_reason", "unknown") or "unknown")
            ] += 1

        mapped_pool_doc_counts.append(sum(doc_id is not None for doc_id in pool_doc_ids))
        seed_entity_counts.append(len(seed_entities))
        selected_structured_doc_counts.append(sum(
            1
            for pos in final_front_positions
            if pool_doc_ids[pos] is not None
            and bool(hipporag.doc_idx_to_structure_entities.get(int(pool_doc_ids[pos]), set()))
        ))
        if len(selector_examples) < 5:
            selector_examples.append({
                "question": qs.question,
                "selected_titles": [pool_titles[pos] for pos in final_front_positions],
                "seed_entities_preview": sorted(seed_entities)[:8],
                "question_entities_preview": sorted(question_entities)[:8],
                "grounded_question_entities_preview": sorted(grounded_question_entities)[:8],
                "proposal_query_entities_preview": sorted(proposal_query_entities)[:8],
                "query_entities_preview": sorted(state_query_entities)[:8],
                "selection_steps": selector_trace.get("selection_steps", selector_trace.get("append_steps", [])),
            })

    summary = {
        "selector": selector_name,
        "score_mode": score_mode,
        "avg_mapped_pool_doc_count": round(float(np.mean(mapped_pool_doc_counts)) if mapped_pool_doc_counts else 0.0, 4),
        "avg_seed_entity_count": round(float(np.mean(seed_entity_counts)) if seed_entity_counts else 0.0, 4),
        "avg_selected_structured_doc_count": round(
            float(np.mean(selected_structured_doc_counts)) if selected_structured_doc_counts else 0.0,
            4,
        ),
        "reserve_top_m": int(max(max(anchor_count, 0), max(reserve_top_m, 0))),
        "max_bridge_slots": int(max(0, max_bridge_slots)),
        "non_anchor_title_dedup": bool(non_anchor_title_dedup),
        "query_entity_source": normalized_query_entity_source if query_solutions else str(query_entity_source or "seed").strip().lower(),
        "gate_mode": str(gate_mode or "none").strip().lower(),
        "gate_min_structure_score": round(float(gate_min_structure_score), 4),
        "gate_min_combined_margin": round(float(gate_min_combined_margin), 4),
        "gate_min_closure_score": round(float(gate_min_closure_score), 4),
        "gate_min_novelty_score": round(float(gate_min_novelty_score), 4),
        "gate_min_frontier_gain": round(float(gate_min_frontier_gain), 4),
        "gate_min_path_coherence": round(float(gate_min_path_coherence), 4),
        "gate_max_avg_local_structure": round(float(gate_max_avg_local_structure), 4),
        "gate_min_suffix_base_mean": round(float(gate_min_suffix_base_mean), 4),
        "reader_order_probe_mode": normalized_reader_order_probe_mode,
        "reader_order_probe_apply_count": int(reader_order_probe_apply_count),
        "reader_order_probe_skip_count": int(reader_order_probe_skip_count),
        "reader_order_probe_reason_counts": dict(sorted(reader_order_probe_reason_counts.items())),
        "gate_apply_count": int(gate_apply_count),
        "gate_skip_count": int(gate_skip_count),
        "gate_reason_counts": dict(sorted(gate_reason_counts.items())),
        "saturation_guard_apply_count": int(saturation_guard_apply_count),
        "saturation_guard_skip_count": int(saturation_guard_skip_count),
        "late_rerank_enabled": bool(late_rerank_enabled),
        "late_rerank_apply_count": int(late_rerank_apply_count),
        "late_rerank_override_count": int(late_rerank_override_count),
        "late_rerank_block_count": int(late_rerank_block_count),
        "late_rerank_parse_failure_count": int(late_rerank_parse_failure_count),
        "late_rerank_error_count": int(late_rerank_error_count),
        "late_rerank_policy": normalized_late_rerank_policy,
        "late_rerank_max_state_score_gap": float(late_rerank_max_state_score_gap),
        "beam_width": int(beam_width),
        "beam_expand_per_state": int(beam_expand_per_state),
        "beam_projected_shortlist_factor": int(beam_projected_shortlist_factor),
        "beam_projection_eval_count": int(beam_projection_eval_count),
        "beam_projection_extra_eval_count": int(beam_projection_extra_eval_count),
        "beam_projection_rescue_count": int(beam_projection_rescue_count),
        "beam_projection_changed_query_count": int(beam_projection_changed_query_count),
        "beam_projection_max_selected_rank": int(beam_projection_max_selected_rank),
        "examples_preview": selector_examples,
    }
    if selector_name == "bridge_append":
        append_count_histogram = Counter(appended_doc_counts)
        summary.update({
            "expand_base_k": int(max(int(expand_base_k), 0)),
            "append_max_docs": int(max(int(append_max_docs), 0)),
            "append_policy": normalized_append_policy,
            "append_random_seed": int(append_random_seed),
            "gap_expand_mode": normalized_gap_expand_mode,
            "gap_expand_max_queries": int(effective_gap_expand_max_queries),
            "expand_min_structure_score": round(float(expand_min_structure_score), 4),
            "assemble_mode": normalized_assemble_mode,
            "coverage_score_variant": normalized_coverage_score_variant,
            "coverage_atom_source": normalized_coverage_atom_source,
            "coverage_admissibility_mode": normalized_coverage_admissibility_mode,
            "assemble_ce_model": str(ce_model) if normalized_assemble_mode in ASSEMBLE_CE_ACTIVE_MODES else None,
            "assemble_ce_device": str(ce_device) if normalized_assemble_mode in ASSEMBLE_CE_ACTIVE_MODES else None,
            "action_swap_apply_count": int(action_swap_apply_count),
            "action_swap_keep_count": int(action_swap_keep_count),
            "action_swap_judge_count": int(action_swap_judge_count),
            "avg_appended_doc_count": round(float(np.mean(appended_doc_counts)) if appended_doc_counts else 0.0, 4),
            "avg_candidate_set_size": round(float(np.mean(expand_candidate_sizes)) if expand_candidate_sizes else 0.0, 4),
            "append_count_histogram": {
                str(int(k)): int(v)
                for k, v in sorted(append_count_histogram.items())
            },
            "append_stop_reason_counts": dict(sorted(append_stop_reason_counts.items())),
        })
        if normalized_append_policy == "gap_expand":
            summary.update({
                "gap_expand_query_count": int(gap_expand_query_count),
                "gap_expand_fallback_query_count": int(gap_expand_fallback_query_count),
                "gap_expand_fallback_rate": round(
                    float(gap_expand_fallback_query_count) / float(max(gap_expand_query_count, 1)),
                    4,
                ),
                "gap_type_distribution": dict(sorted(gap_type_counts.items())),
                "gap_slot_distribution": dict(sorted(gap_slot_counts.items())),
                "gap_abstain_reason_counts": dict(sorted(gap_abstain_reason_counts.items())),
                "avg_gap_micro_query_count": round(float(np.mean(gap_micro_query_counts)) if gap_micro_query_counts else 0.0, 4),
                "avg_gap_candidate_count": round(float(np.mean(gap_candidate_counts)) if gap_candidate_counts else 0.0, 4),
                "avg_gap_candidates_selected_into_final": round(float(np.mean(gap_candidates_selected_into_final_counts)) if gap_candidates_selected_into_final_counts else 0.0, 4),
                "gap_candidate_selected_query_rate": round(
                    float(sum(1 for count in gap_candidates_selected_into_final_counts if count > 0)) / float(max(len(gap_candidates_selected_into_final_counts), 1)),
                    4,
                ),
            })
    if selector_name == "requirement_beam":
        summary.update({
            "requirement_mode": str((requirement_selector_bundle or {}).get("mode", "oracle")),
            "requirement_cache_path": str((requirement_selector_bundle or {}).get("cache_path", "")),
            "requirement_model_path": str((requirement_selector_bundle or {}).get("model_path", "")) or None,
            "requirement_reserve_policy": str(requirement_reserve_policy or DEFAULT_REQUIREMENT_BEAM_RESERVE_POLICY),
            "requirement_smooth_tau": round(float((requirement_selector_bundle or {}).get("smooth_tau", DEFAULT_REQUIREMENT_SMOOTH_TAU)), 4),
            "requirement_counterfactual_tau": round(float((requirement_selector_bundle or {}).get("counterfactual_tau", DEFAULT_REQUIREMENT_CF_TAU)), 4),
            "requirement_cache_hit_count": int(requirement_cache_hit_count),
            "avg_requirement_support_completeness": round(float(np.mean(requirement_support_scores)) if requirement_support_scores else 0.0, 4),
            "avg_requirement_counterfactual_leakage": round(float(np.mean(requirement_leakage_scores)) if requirement_leakage_scores else 0.0, 4),
            "avg_requirement_frontier_size": round(float(np.mean(requirement_frontier_sizes)) if requirement_frontier_sizes else 0.0, 4),
            "avg_requirement_runtime_anchor_count": round(float(np.mean(requirement_runtime_anchor_counts)) if requirement_runtime_anchor_counts else 0.0, 4),
            "avg_requirement_runtime_reserved_count": round(float(np.mean(requirement_runtime_reserve_counts)) if requirement_runtime_reserve_counts else 0.0, 4),
            "requirement_live_annotation_apply_count": int(requirement_live_annotation_apply_count),
            "requirement_live_source_expand_apply_count": int(requirement_live_source_expand_apply_count),
            "requirement_live_annotation_pool_k": int((requirement_selector_bundle or {}).get("live_annotation_pool_k", 0) or 0),
            "requirement_live_annotation_score_mode": str((requirement_selector_bundle or {}).get("live_annotation_score_mode", "heuristic")),
            "requirement_live_atomic_model_path": str((requirement_selector_bundle or {}).get("live_atomic_model_path", "")) or None,
            "requirement_live_source_expand_factor": int((requirement_selector_bundle or {}).get("live_source_expand_factor", 0) or 0),
            "requirement_live_source_shortlist_limit": int(beam_expand_per_state if int((requirement_selector_bundle or {}).get("live_source_expand_factor", 0) or 0) > 0 else 0),
            "requirement_exposure_watch_titles": list((requirement_selector_bundle or {}).get("exposure_watch_titles", []) or []),
            "requirement_probe_force_source_titles": list((requirement_selector_bundle or {}).get("probe_force_source_titles", []) or []),
            "requirement_probe_force_shortlist_titles": list((requirement_selector_bundle or {}).get("probe_force_shortlist_titles", []) or []),
            "requirement_probe_force_final_titles": list((requirement_selector_bundle or {}).get("probe_force_final_titles", []) or []),
            "requirement_probe_force_pool_gold_into_final": bool(
                (requirement_selector_bundle or {}).get("probe_force_pool_gold_into_final", False)
            ),
            "requirement_probe_force_pool_gold_enabled_count": int(requirement_probe_force_pool_gold_enabled_count),
            "requirement_probe_force_pool_gold_in_pool_query_count": int(requirement_probe_force_pool_gold_in_pool_query_count),
            "requirement_probe_force_pool_gold_applied_query_count": int(requirement_probe_force_pool_gold_applied_query_count),
            "requirement_probe_force_pool_gold_applied_title_count": int(requirement_probe_force_pool_gold_applied_title_count),
            "requirement_probe_bridge_bonus_mode": str((requirement_selector_bundle or {}).get("probe_bridge_bonus_mode", "off")),
            "requirement_probe_bridge_bonus_weight": round(float((requirement_selector_bundle or {}).get("probe_bridge_bonus_weight", 0.0) or 0.0), 4),
            "requirement_probe_shortlist_sort_mode": str((requirement_selector_bundle or {}).get("probe_shortlist_sort_mode", DEFAULT_REQUIREMENT_SHORTLIST_SORT_MODE)),
            "requirement_reserve_reason_counts": dict(sorted(requirement_reserve_reason_counts.items())),
        })
    logger.info(
        "Applied %s selector over pool@%d for %d queries (avg mapped pool docs=%.2f, avg seed entities=%.2f)",
        selector_name,
        pool_k,
        len(query_solutions),
        float(np.mean(mapped_pool_doc_counts)) if mapped_pool_doc_counts else 0.0,
        float(np.mean(seed_entity_counts)) if seed_entity_counts else 0.0,
    )
    return selected_solutions, summary


def is_causal_query_solution(config: BaseConfig, query_solution: QuerySolution) -> bool:
    if getattr(config, "causal_engine_version", "legacy") == "v2":
        trace = query_solution.retrieval_trace or {}
        return trace.get("router_label") in {"cause", "effect", "prevention", "causal"}
    return route_query_type(query_solution.question) != "non_causal"


def resolve_report_query_type(config: BaseConfig, query_solution: QuerySolution) -> str | None:
    if getattr(config, "causal_engine_version", "legacy") == "v2":
        return (query_solution.retrieval_trace or {}).get("router_label")
    return route_query_type(query_solution.question)


def build_report_examples(config: BaseConfig,
                          query_solutions: Sequence[QuerySolution],
                          doc_text_to_chunk_id: Dict[str, str],
                          retrieval_only: bool,
                          doc_limit: int = 3) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    effective_doc_limit = max(int(doc_limit), 0)
    for query_solution in query_solutions:
        examples.append({
            "question": query_solution.question,
            "query_type": resolve_report_query_type(config, query_solution),
            "answer": query_solution.answer if not retrieval_only else None,
            "gold_answers": query_solution.gold_answers if not retrieval_only else None,
            "docs": list(query_solution.docs[:effective_doc_limit]),
            "retrieved_doc_ids": serialize_retrieved_doc_ids(query_solution.docs, doc_text_to_chunk_id),
            "retrieval_trace": query_solution.retrieval_trace or {},
        })
    return examples


def build_retrieval_cache_examples(query_solutions: Sequence[QuerySolution],
                                   doc_text_to_chunk_id: Dict[str, str]) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for query_solution in query_solutions:
        doc_scores = query_solution.doc_scores
        serialized_scores = (
            [float(score) for score in np.asarray(doc_scores, dtype=float).tolist()]
            if doc_scores is not None else
            []
        )
        examples.append({
            "question": query_solution.question,
            "retrieved_doc_ids": serialize_retrieved_doc_ids(query_solution.docs, doc_text_to_chunk_id),
            "retrieved_doc_scores": serialized_scores,
            "retrieval_trace": query_solution.retrieval_trace or {},
        })
    return examples


def write_retrieval_cache_payload(output_path: str | Path,
                                  query_solutions: Sequence[QuerySolution],
                                  doc_text_to_chunk_id: Dict[str, str],
                                  overall_metrics: Mapping[str, object] | None) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall_metrics": dict(overall_metrics or {}),
        "examples": build_retrieval_cache_examples(
            query_solutions=query_solutions,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_baseline_report_payload(report_path: str | Path) -> Dict[str, object]:
    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = list(payload.get("examples") or [])
    if not examples:
        raise ValueError(f"Baseline report has no examples: {path}")
    overall_metrics = dict(payload.get("overall_recomputed") or payload.get("overall_from_pipeline") or {})
    if not overall_metrics:
        report_metrics = dict(payload.get("report_metrics") or {})
        primary_retrieval_metrics = dict(report_metrics.get("primary_retrieval_metrics") or {})
        primary_qa_metrics = dict(report_metrics.get("primary_qa_metrics") or {})
        if primary_retrieval_metrics or primary_qa_metrics:
            overall_metrics = {
                **primary_retrieval_metrics,
                **primary_qa_metrics,
                **(
                    {"num_queries": int(report_metrics["num_queries"])}
                    if report_metrics.get("num_queries") is not None
                    else {}
                ),
            }
    if not overall_metrics:
        raise ValueError(f"Baseline report has no overall metrics: {path}")
    return {
        "path": str(path),
        "overall_metrics": overall_metrics,
        "examples": examples,
    }


def load_retrieval_cache_payload(cache_path: str | Path) -> Dict[str, object]:
    path = Path(cache_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = list(payload.get("examples") or [])
    if not examples:
        raise ValueError(f"Retrieval cache has no examples: {path}")
    overall_metrics = dict(
        payload.get("overall_metrics")
        or payload.get("overall_recomputed")
        or payload.get("overall_from_pipeline")
        or {}
    )
    return {
        "path": str(path),
        "overall_metrics": overall_metrics,
        "examples": examples,
    }


def ensure_runtime_objects_for_cached_retrieval(hipporag: HippoRAG) -> None:
    """Initialize retrieval-time mappings/structure objects when retrieval is skipped."""
    if getattr(hipporag, "ready_to_retrieve_v2", False) or getattr(hipporag, "ready_to_retrieve", False):
        return
    hipporag.prepare_retrieval_objects()


def compute_retrieval_recall_metrics(
    query_solutions: Sequence[QuerySolution],
    gold_docs: Sequence[Sequence[str]],
    ks: Sequence[int] = REPORT_RECALL_CUTOFFS,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        recalls = []
        for q_idx, qs in enumerate(query_solutions):
            gold_set = set(gold_docs[q_idx])
            top_k_set = set(qs.docs[: int(k)])
            recalls.append(len(gold_set & top_k_set) / max(1, len(gold_set)))
        metrics[f"Recall@{int(k)}"] = round(float(np.mean(recalls)), 4)
    return metrics


def extract_retrieval_metrics(metrics: Mapping[str, object] | None) -> Dict[str, object]:
    return {
        str(key): value
        for key, value in dict(metrics or {}).items()
        if str(key).startswith("Recall@")
    }


def extract_qa_metrics(metrics: Mapping[str, object] | None) -> Dict[str, object]:
    return {
        str(key): value
        for key, value in dict(metrics or {}).items()
        if str(key) in {"ExactMatch", "F1"}
    }


def build_report_metrics_summary(
    overall_metrics: Mapping[str, object] | None,
    *,
    expand_assemble_results: Mapping[str, object] | None = None,
    setwise_selector_results: Mapping[str, object] | None = None,
    cross_encoder_rerank_results: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    baseline_retrieval_metrics = extract_retrieval_metrics(overall_metrics)
    baseline_qa_metrics = extract_qa_metrics(overall_metrics)
    num_queries = (
        int(dict(overall_metrics or {}).get("num_queries", 0))
        if dict(overall_metrics or {}).get("num_queries") is not None
        else None
    )

    primary_run_type = "baseline"
    primary_retrieval_metrics = dict(baseline_retrieval_metrics)
    primary_qa_metrics = dict(baseline_qa_metrics)

    if expand_assemble_results:
        primary_run_type = "expand_assemble"
        primary_retrieval_metrics = dict(expand_assemble_results.get("method_retrieval_metrics") or {})
        primary_qa_metrics = {
            "ExactMatch": expand_assemble_results.get("method_EM"),
            "F1": expand_assemble_results.get("method_F1"),
        }
    elif setwise_selector_results:
        primary_run_type = "setwise_selector"
        primary_retrieval_metrics = dict(setwise_selector_results.get("selector_retrieval_metrics") or {})
        primary_qa_metrics = {
            "ExactMatch": setwise_selector_results.get("selector_EM"),
            "F1": setwise_selector_results.get("selector_F1"),
        }
    elif cross_encoder_rerank_results:
        primary_run_type = "cross_encoder_rerank"
        primary_retrieval_metrics = dict(cross_encoder_rerank_results.get("ce_retrieval_metrics") or {})
        primary_qa_metrics = {
            "ExactMatch": cross_encoder_rerank_results.get("ce_rerank_EM"),
            "F1": cross_encoder_rerank_results.get("ce_rerank_F1"),
        }

    summary: Dict[str, object] = {
        "primary_run_type": primary_run_type,
        "primary_retrieval_metrics": primary_retrieval_metrics,
        "primary_qa_metrics": primary_qa_metrics,
    }
    if num_queries is not None:
        summary["num_queries"] = num_queries
    if primary_run_type != "baseline":
        summary["baseline_retrieval_metrics"] = baseline_retrieval_metrics
        summary["baseline_qa_metrics"] = baseline_qa_metrics
    return summary


def hydrate_query_solutions_from_baseline_report(query_solutions: Sequence[QuerySolution],
                                                 baseline_report_payload: Mapping[str, object],
                                                 gold_docs: Sequence[Sequence[str]] | None = None) -> None:
    examples = list(baseline_report_payload.get("examples") or [])
    if len(examples) != len(query_solutions):
        raise ValueError(
            "Baseline report example count does not match current query count: "
            f"{len(examples)} != {len(query_solutions)}"
        )

    use_index_alignment = True
    for idx, query_solution in enumerate(query_solutions):
        report_question = str((examples[idx] or {}).get("question", "")).strip()
        if report_question != str(query_solution.question).strip():
            use_index_alignment = False
            break

    example_by_question: Dict[str, Dict[str, object]] = {}
    if not use_index_alignment:
        for example in examples:
            question = str((example or {}).get("question", "")).strip()
            if not question:
                raise ValueError("Baseline report example is missing question text.")
            if question in example_by_question:
                raise ValueError(
                    "Baseline report question alignment requires unique question text when order differs."
                )
            example_by_question[question] = dict(example)

    for idx, query_solution in enumerate(query_solutions):
        if use_index_alignment:
            example = dict(examples[idx] or {})
        else:
            example = example_by_question.get(str(query_solution.question).strip())
            if example is None:
                raise ValueError(
                    f"Could not align query_solution question to baseline report example: {query_solution.question!r}"
                )
        query_solution.answer = str(example.get("answer") or "")
        query_solution.gold_answers = list(example.get("gold_answers") or query_solution.gold_answers or [])
        if gold_docs is not None:
            query_solution.gold_docs = list(gold_docs[idx])


def hydrate_query_solutions_from_retrieval_cache(query_solutions: Sequence[QuerySolution],
                                                 retrieval_cache_payload: Mapping[str, object],
                                                 chunk_id_to_doc_text: Mapping[str, str],
                                                 gold_docs: Sequence[Sequence[str]] | None = None) -> None:
    examples = list(retrieval_cache_payload.get("examples") or [])
    if len(examples) != len(query_solutions):
        raise ValueError(
            "Retrieval cache example count does not match current query count: "
            f"{len(examples)} != {len(query_solutions)}"
        )

    use_index_alignment = True
    for idx, query_solution in enumerate(query_solutions):
        cached_question = str((examples[idx] or {}).get("question", "")).strip()
        if cached_question != str(query_solution.question).strip():
            use_index_alignment = False
            break

    example_by_question: Dict[str, Dict[str, object]] = {}
    if not use_index_alignment:
        for example in examples:
            question = str((example or {}).get("question", "")).strip()
            if not question:
                raise ValueError("Retrieval cache example is missing question text.")
            if question in example_by_question:
                raise ValueError(
                    "Retrieval cache question alignment requires unique question text when order differs."
                )
            example_by_question[question] = dict(example)

    for idx, query_solution in enumerate(query_solutions):
        if use_index_alignment:
            example = dict(examples[idx] or {})
        else:
            example = example_by_question.get(str(query_solution.question).strip())
            if example is None:
                raise ValueError(
                    f"Could not align query_solution question to retrieval cache example: {query_solution.question!r}"
                )
        retrieved_doc_ids = list(example.get("retrieved_doc_ids") or [])
        retrieved_doc_scores = list(example.get("retrieved_doc_scores") or [])
        if retrieved_doc_scores and len(retrieved_doc_ids) != len(retrieved_doc_scores):
            raise ValueError(
                "Retrieval cache example has mismatched doc ids and scores lengths "
                f"for question {query_solution.question!r}: "
                f"{len(retrieved_doc_ids)} != {len(retrieved_doc_scores)}"
            )

        docs: List[str] = []
        for chunk_id in retrieved_doc_ids:
            if chunk_id is None:
                raise ValueError(
                    f"Retrieval cache example contains null chunk id for question {query_solution.question!r}"
                )
            doc_text = chunk_id_to_doc_text.get(str(chunk_id))
            if doc_text is None:
                raise ValueError(
                    f"Retrieval cache chunk id {chunk_id!r} is not present in current corpus "
                    f"for question {query_solution.question!r}"
                )
            docs.append(doc_text)

        query_solution.docs = docs
        query_solution.doc_scores = np.asarray(retrieved_doc_scores, dtype=float)
        query_solution.retrieval_trace = dict(example.get("retrieval_trace") or {})
        if gold_docs is not None:
            query_solution.gold_docs = list(gold_docs[idx])


def build_setwise_selector_query_traces(config: BaseConfig,
                                        baseline_solutions: Sequence[QuerySolution],
                                        selected_solutions: Sequence[QuerySolution],
                                        gold_docs: Sequence[Sequence[str]],
                                        gold_answers: Sequence[Sequence[str]],
                                        doc_text_to_chunk_id: Dict[str, str]) -> List[Dict[str, object]]:
    if len(baseline_solutions) != len(selected_solutions):
        raise ValueError("Baseline and selected QuerySolution collections must have the same length.")

    qa_em_metric = QAExactMatch(global_config=None)
    qa_f1_metric = QAF1Score(global_config=None)
    baseline_answers = [query_solution.answer or "" for query_solution in baseline_solutions]
    selector_answers = [query_solution.answer or "" for query_solution in selected_solutions]
    _, baseline_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, baseline_answers)
    _, baseline_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, baseline_answers)
    _, selector_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, selector_answers)
    _, selector_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, selector_answers)

    query_traces: List[Dict[str, object]] = []
    reader_top_k = max(int(getattr(config, "qa_top_k", 5)), 0)
    for q_idx, (baseline_qs, selected_qs) in enumerate(zip(baseline_solutions, selected_solutions)):
        baseline_top_docs = list(baseline_qs.docs[:reader_top_k])
        selector_top_docs = list(selected_qs.docs[:reader_top_k])
        baseline_top_titles = [extract_doc_title(doc_text) for doc_text in baseline_top_docs]
        selector_top_titles = [extract_doc_title(doc_text) for doc_text in selector_top_docs]
        selector_trace = dict((selected_qs.retrieval_trace or {}).get("setwise_selector_trace", {}) or {})
        requirement_title_exposure_summary = list(
            selector_trace.get("requirement_title_exposure_summary", []) or []
        )
        query_traces.append({
            "question": selected_qs.question,
            "query_type": resolve_report_query_type(config, selected_qs),
            "gold_answers": list(selected_qs.gold_answers or []),
            "gold_doc_count": int(len(set(gold_docs[q_idx]))),
            "gold_titles": [extract_doc_title(doc_text) for doc_text in gold_docs[q_idx]],
            "baseline_answer": baseline_qs.answer or "",
            "selector_answer": selected_qs.answer or "",
            "baseline_top_titles": baseline_top_titles,
            "selector_top_titles": selector_top_titles,
            "baseline_top_doc_ids": serialize_retrieved_doc_ids(baseline_top_docs, doc_text_to_chunk_id),
            "selector_top_doc_ids": serialize_retrieved_doc_ids(selector_top_docs, doc_text_to_chunk_id),
            "baseline_title_duplicate_count": int(len(baseline_top_titles) - len(set(baseline_top_titles))),
            "selector_title_duplicate_count": int(len(selector_top_titles) - len(set(selector_top_titles))),
            "changed_from_baseline": bool(
                baseline_top_titles != selector_top_titles
                or (baseline_qs.answer or "") != (selected_qs.answer or "")
            ),
            "baseline_metrics": {
                "ExactMatch": round(float(baseline_per_query_em[q_idx]["ExactMatch"]), 4),
                "F1": round(float(baseline_per_query_f1[q_idx]["F1"]), 4),
            },
            "selector_metrics": {
                "ExactMatch": round(float(selector_per_query_em[q_idx]["ExactMatch"]), 4),
                "F1": round(float(selector_per_query_f1[q_idx]["F1"]), 4),
            },
            "requirement_title_exposure_summary": requirement_title_exposure_summary,
            "selector_trace": selector_trace,
        })
    return query_traces


def build_expand_assemble_query_traces(config: BaseConfig,
                                       baseline_solutions: Sequence[QuerySolution],
                                       method_solutions: Sequence[QuerySolution],
                                       gold_docs: Sequence[Sequence[str]],
                                       gold_answers: Sequence[Sequence[str]],
                                       doc_text_to_chunk_id: Dict[str, str]) -> List[Dict[str, object]]:
    if len(baseline_solutions) != len(method_solutions):
        raise ValueError("Baseline and method QuerySolution collections must have the same length.")

    qa_em_metric = QAExactMatch(global_config=None)
    qa_f1_metric = QAF1Score(global_config=None)
    baseline_answers = [query_solution.answer or "" for query_solution in baseline_solutions]
    method_answers = [query_solution.answer or "" for query_solution in method_solutions]
    _, baseline_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, baseline_answers)
    _, baseline_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, baseline_answers)
    _, method_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, method_answers)
    _, method_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, method_answers)

    query_traces: List[Dict[str, object]] = []
    reader_top_k = max(int(getattr(config, "qa_top_k", 5)), 0)
    for q_idx, (baseline_qs, method_qs) in enumerate(zip(baseline_solutions, method_solutions)):
        baseline_top_docs = list(baseline_qs.docs[:reader_top_k])
        method_top_docs = list(method_qs.docs[:reader_top_k])
        baseline_top_titles = [extract_doc_title(doc_text) for doc_text in baseline_top_docs]
        method_top_titles = [extract_doc_title(doc_text) for doc_text in method_top_docs]
        expand_assemble_trace = dict((method_qs.retrieval_trace or {}).get("expand_assemble_trace", {}) or {})
        query_traces.append({
            "question": method_qs.question,
            "query_type": resolve_report_query_type(config, method_qs),
            "gold_answers": list(method_qs.gold_answers or []),
            "gold_doc_count": int(len(set(gold_docs[q_idx]))),
            "gold_titles": [extract_doc_title(doc_text) for doc_text in gold_docs[q_idx]],
            "baseline_answer": baseline_qs.answer or "",
            "method_answer": method_qs.answer or "",
            "baseline_top_titles": baseline_top_titles,
            "method_top_titles": method_top_titles,
            "baseline_top_doc_ids": serialize_retrieved_doc_ids(baseline_top_docs, doc_text_to_chunk_id),
            "method_top_doc_ids": serialize_retrieved_doc_ids(method_top_docs, doc_text_to_chunk_id),
            "baseline_title_duplicate_count": int(len(baseline_top_titles) - len(set(baseline_top_titles))),
            "method_title_duplicate_count": int(len(method_top_titles) - len(set(method_top_titles))),
            "changed_from_baseline": bool(
                baseline_top_titles != method_top_titles
                or (baseline_qs.answer or "") != (method_qs.answer or "")
            ),
            "baseline_metrics": {
                "ExactMatch": round(float(baseline_per_query_em[q_idx]["ExactMatch"]), 4),
                "F1": round(float(baseline_per_query_f1[q_idx]["F1"]), 4),
            },
            "method_metrics": {
                "ExactMatch": round(float(method_per_query_em[q_idx]["ExactMatch"]), 4),
                "F1": round(float(method_per_query_f1[q_idx]["F1"]), 4),
            },
            "expand_assemble_trace": expand_assemble_trace,
        })
    return query_traces


def has_nonempty_v2_subgraph(query_solution: QuerySolution) -> bool:
    trace = query_solution.retrieval_trace or {}
    return bool(trace.get("subgraph_nonempty", False))


def compute_slice_metrics(config: BaseConfig,
                          query_solutions: List[QuerySolution],
                          gold_docs: List[List[str]],
                          gold_answers: List[List[str]] | None) -> Dict[str, Dict[str, float]]:
    retrieved_docs = [query_solution.docs for query_solution in query_solutions]

    retrieval = RetrievalRecall(global_config=config)

    retrieval_metrics, _ = retrieval.calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=retrieved_docs,
        k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
    )
    em_metrics: Dict[str, float] = {}
    f1_metrics: Dict[str, float] = {}
    predicted_answers = []
    if gold_answers is not None:
        predicted_answers = [query_solution.answer for query_solution in query_solutions]
        qa_em = QAExactMatch(global_config=config)
        qa_f1 = QAF1Score(global_config=config)
        em_metrics, _ = qa_em.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            aggregation_fn=np.max,
        )
        f1_metrics, _ = qa_f1.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            aggregation_fn=np.max,
        )

    causal_indices = [
        idx for idx, query_solution in enumerate(query_solutions)
        if is_causal_query_solution(config, query_solution)
    ]
    if causal_indices:
        causal_retrieval_metrics, _ = retrieval.calculate_metric_scores(
            gold_docs=subset_by_indices(gold_docs, causal_indices),
            retrieved_docs=subset_by_indices(retrieved_docs, causal_indices),
            k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
        )
        if gold_answers is not None:
            qa_em = QAExactMatch(global_config=config)
            qa_f1 = QAF1Score(global_config=config)
            causal_em_metrics, _ = qa_em.calculate_metric_scores(
                gold_answers=subset_by_indices(gold_answers, causal_indices),
                predicted_answers=subset_by_indices(predicted_answers, causal_indices),
                aggregation_fn=np.max,
            )
            causal_f1_metrics, _ = qa_f1.calculate_metric_scores(
                gold_answers=subset_by_indices(gold_answers, causal_indices),
                predicted_answers=subset_by_indices(predicted_answers, causal_indices),
                aggregation_fn=np.max,
            )
        else:
            causal_em_metrics, causal_f1_metrics = {}, {}
    else:
        causal_retrieval_metrics, causal_em_metrics, causal_f1_metrics = {}, {}, {}

    nonempty_subgraph_indices = [
        idx for idx, query_solution in enumerate(query_solutions)
        if getattr(config, "causal_engine_version", "legacy") == "v2"
        and has_nonempty_v2_subgraph(query_solution)
    ]
    if nonempty_subgraph_indices:
        subgraph_retrieval_metrics, _ = retrieval.calculate_metric_scores(
            gold_docs=subset_by_indices(gold_docs, nonempty_subgraph_indices),
            retrieved_docs=subset_by_indices(retrieved_docs, nonempty_subgraph_indices),
            k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
        )
        if gold_answers is not None:
            qa_em = QAExactMatch(global_config=config)
            qa_f1 = QAF1Score(global_config=config)
            subgraph_em_metrics, _ = qa_em.calculate_metric_scores(
                gold_answers=subset_by_indices(gold_answers, nonempty_subgraph_indices),
                predicted_answers=subset_by_indices(predicted_answers, nonempty_subgraph_indices),
                aggregation_fn=np.max,
            )
            subgraph_f1_metrics, _ = qa_f1.calculate_metric_scores(
                gold_answers=subset_by_indices(gold_answers, nonempty_subgraph_indices),
                predicted_answers=subset_by_indices(predicted_answers, nonempty_subgraph_indices),
                aggregation_fn=np.max,
            )
        else:
            subgraph_em_metrics, subgraph_f1_metrics = {}, {}
    else:
        subgraph_retrieval_metrics, subgraph_em_metrics, subgraph_f1_metrics = {}, {}, {}

    return {
        "overall": {
            **retrieval_metrics,
            **em_metrics,
            **f1_metrics,
            "num_queries": len(query_solutions),
        },
        "causal_slice": {
            **causal_retrieval_metrics,
            **causal_em_metrics,
            **causal_f1_metrics,
            "num_queries": len(causal_indices),
        },
        "nonempty_subgraph_slice": {
            **subgraph_retrieval_metrics,
            **subgraph_em_metrics,
            **subgraph_f1_metrics,
            "num_queries": len(nonempty_subgraph_indices),
        },
    }


def summarize_v2_metrics(config: BaseConfig,
                         hipporag: HippoRAG,
                         query_solutions: List[QuerySolution]) -> Dict[str, object]:
    if getattr(config, "causal_engine_version", "legacy") != "v2":
        return {}

    traces = [query_solution.retrieval_trace or {} for query_solution in query_solutions]
    router_label_counts = Counter(str(trace.get("router_label", "unknown")) for trace in traces)
    chain_counts = [int(trace.get("subgraph_chain_count", 0)) for trace in traces]
    selected_chain_counts = [int(trace.get("selected_subgraph_chain_count", 0)) for trace in traces]
    seed_counts = [len(trace.get("event_seed_ids", [])) for trace in traces]
    query_entity_counts = [len(trace.get("query_entities", [])) for trace in traces]
    serialized_counts = [len(trace.get("serialized_causal_context", [])) for trace in traces]
    causal_doc_counts = [int(trace.get("causal_doc_count", 0)) for trace in traces]
    base_retrieval_mode_counts = Counter(str(trace.get("v2_base_retrieval_mode", "dense")) for trace in traces)
    base_retrieval_status_counts = Counter(str(trace.get("v2_base_retrieval_status", "unknown")) for trace in traces)
    base_retrieval_route_counts = Counter(str(trace.get("baseline_route_name", "unknown")) for trace in traces)
    base_retrieval_fact_counts = [int(trace.get("v2_base_retrieval_fact_count", 0)) for trace in traces]
    base_retrieval_dense_fallback_count = sum(1 for trace in traces if trace.get("v2_base_retrieval_used_dense_fallback"))
    use_causal_path_count = sum(1 for trace in traces if trace.get("use_causal_path"))
    probe_attempted_count = sum(1 for trace in traces if trace.get("causal_probe_attempted"))
    forced_probe_count = sum(1 for trace in traces if trace.get("causal_probe_forced"))
    subgraph_nonempty_count = sum(1 for trace in traces if trace.get("subgraph_nonempty"))
    causal_v2_used_count = sum(1 for trace in traces if trace.get("causal_v2_used"))
    generator_used_count = sum(1 for trace in traces if trace.get("generator_used_causal_context"))

    manifest_stats = {}
    engine = getattr(hipporag, "causal_v2_engine", None)
    if engine is not None:
        manifest_stats = engine.read_manifest()

    num_queries = max(1, len(traces))
    causal_examples = []
    for query_solution in query_solutions:
        trace = query_solution.retrieval_trace or {}
        if trace.get("causal_v2_used") or trace.get("causal_probe_attempted"):
            causal_examples.append({
                "question": query_solution.question,
                "router_label": trace.get("router_label"),
                "probe_route_label": trace.get("probe_route_label"),
                "router_score": trace.get("router_score"),
                "router_margin": trace.get("router_margin"),
                "causal_probe_forced": trace.get("causal_probe_forced"),
                "subgraph_nonempty": trace.get("subgraph_nonempty"),
                "subgraph_chain_count": trace.get("subgraph_chain_count"),
                "selected_subgraph_chain_count": trace.get("selected_subgraph_chain_count"),
                "query_entities": trace.get("query_entities", [])[:6],
                "selected_chain_scores": trace.get("selected_chain_scores", []),
                "serialized_causal_context_preview": trace.get("serialized_causal_context_preview", []),
            })
        if len(causal_examples) >= 5:
            break

    return {
        "index_manifest": manifest_stats,
        "router_label_counts": dict(sorted(router_label_counts.items())),
        "base_retrieval_mode_counts": dict(sorted(base_retrieval_mode_counts.items())),
        "base_retrieval_status_counts": dict(sorted(base_retrieval_status_counts.items())),
        "base_retrieval_route_counts": dict(sorted(base_retrieval_route_counts.items())),
        "base_retrieval_dense_fallback_count": int(base_retrieval_dense_fallback_count),
        "base_retrieval_dense_fallback_rate": round(base_retrieval_dense_fallback_count / num_queries, 4),
        "avg_base_retrieval_fact_count": round(float(np.mean(base_retrieval_fact_counts)) if base_retrieval_fact_counts else 0.0, 4),
        "use_causal_path_count": int(use_causal_path_count),
        "use_causal_path_rate": round(use_causal_path_count / num_queries, 4),
        "probe_attempted_count": int(probe_attempted_count),
        "probe_attempted_rate": round(probe_attempted_count / num_queries, 4),
        "forced_probe_count": int(forced_probe_count),
        "forced_probe_rate": round(forced_probe_count / num_queries, 4),
        "subgraph_nonempty_count": int(subgraph_nonempty_count),
        "subgraph_nonempty_rate": round(subgraph_nonempty_count / num_queries, 4),
        "causal_v2_used_count": int(causal_v2_used_count),
        "causal_v2_used_rate": round(causal_v2_used_count / num_queries, 4),
        "generator_used_causal_context_count": int(generator_used_count),
        "generator_used_causal_context_rate": round(generator_used_count / num_queries, 4),
        "avg_subgraph_chain_count": round(float(np.mean(chain_counts)) if chain_counts else 0.0, 4),
        "avg_selected_chain_count": round(float(np.mean(selected_chain_counts)) if selected_chain_counts else 0.0, 4),
        "max_subgraph_chain_count": int(max(chain_counts) if chain_counts else 0),
        "avg_seed_event_count": round(float(np.mean(seed_counts)) if seed_counts else 0.0, 4),
        "avg_query_entity_count": round(float(np.mean(query_entity_counts)) if query_entity_counts else 0.0, 4),
        "avg_serialized_context_count": round(float(np.mean(serialized_counts)) if serialized_counts else 0.0, 4),
        "avg_causal_doc_count": round(float(np.mean(causal_doc_counts)) if causal_doc_counts else 0.0, 4),
        "causal_examples_preview": causal_examples,
    }


def build_config(args, corpus_len: int) -> BaseConfig:
    return BaseConfig(
        save_dir=args.save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        llm_request_name=args.llm_request_name,
        embedding_base_url=args.embedding_base_url,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=args.qa_top_k,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=args.embedding_batch_size,
        max_new_tokens=None,
        max_retry_attempts=args.max_retry_attempts,
        corpus_len=corpus_len,
        openie_mode=args.openie_mode,
        planner_enabled=string_to_bool(args.planner_enabled),
        planner_mode=args.planner_mode,
        planner_max_steps=args.planner_max_steps,
        causal_enabled=string_to_bool(args.causal_enabled),
        causal_query_only=string_to_bool(args.causal_query_only),
        causal_gate_mode=args.causal_gate_mode,
        causal_seed_top_k=args.causal_seed_top_k,
        causal_confidence_threshold=args.causal_confidence_threshold,
        causal_damping=args.causal_damping,
        causal_blend_dense_weight=args.causal_blend_dense_weight,
        causal_blend_fact_weight=args.causal_blend_fact_weight,
        causal_blend_graph_weight=args.causal_blend_graph_weight,
        causal_margin_gate_enabled=string_to_bool(args.causal_margin_gate_enabled),
        causal_margin_threshold=args.causal_margin_threshold,
        causal_blend_top_k=args.causal_blend_top_k,
        causal_engine_version=getattr(args, "causal_engine_version", "legacy"),
        causal_v2_probe_mode=getattr(args, "causal_v2_probe_mode", "router"),
        causal_v2_graph_mode=getattr(args, "causal_v2_graph_mode", "causal"),
        causal_v2_base_retrieval_mode=getattr(args, "causal_v2_base_retrieval_mode", "dense"),
        general_graph_related_to_weight=getattr(args, "general_graph_related_to_weight", 0.3),
        general_graph_seed_top_k=getattr(args, "general_graph_seed_top_k", 10),
        causal_v2_extraction_max_tokens=getattr(args, "causal_v2_extraction_max_tokens", 768),
        causal_v2_extraction_retry_attempts=getattr(args, "causal_v2_extraction_retry_attempts", 2),
        causal_v2_extraction_workers=getattr(args, "causal_v2_extraction_workers", 4),
        causal_event_top_k=getattr(args, "causal_event_top_k", 8),
        causal_v2_max_hops=getattr(args, "causal_v2_max_hops", 2),
        causal_chain_top_k=getattr(args, "causal_chain_top_k", 6),
        causal_context_max_items=getattr(args, "causal_context_max_items", 0),
        causal_er_similarity_threshold=getattr(args, "causal_er_similarity_threshold", 0.92),
        causal_er_text_threshold=getattr(args, "causal_er_text_threshold", 0.55),
        causal_v2_min_edge_confidence=getattr(args, "causal_v2_min_edge_confidence", 0.7),
        structure_rerank_enabled=string_to_bool(args.structure_rerank_enabled),
        structure_rerank_top_n=args.structure_rerank_top_n,
        structure_rerank_bonus_weight=args.structure_rerank_bonus_weight,
        structure_rerank_min_edge_support=args.structure_rerank_min_edge_support,
        structure_rerank_max_top5_swaps=args.structure_rerank_max_top5_swaps,
        structure_rerank_seed_top_k=args.structure_rerank_seed_top_k,
        structure_rerank_max_hops=args.structure_rerank_max_hops,
        structure_relation_probe_mode=getattr(args, "structure_relation_probe_mode", "off"),
        structure_continuity_probe_mode=getattr(args, "structure_continuity_probe_mode", "off"),
        structure_seed_target_bridge_mode=getattr(args, "structure_seed_target_bridge_mode", "off"),
        structure_rerank_margin_threshold=args.structure_rerank_margin_threshold,
        rerank_require_non_empty=string_to_bool(args.rerank_require_non_empty),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate HippoRAG with local Qwen3 causal retrieval.")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--baseline_report_json", type=str, default="")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument(
        "--llm_request_name",
        type=str,
        default=None,
        help="Optional API-side model name. Use this to hit a renamed service while reusing llm_name-keyed artifacts.",
    )
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    parser.add_argument("--planner_enabled", type=str, default="false")
    parser.add_argument("--planner_mode", choices=["none", "myopic"], default="none")
    parser.add_argument("--planner_max_steps", type=int, default=3)
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--max_qa_steps", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--causal_enabled", type=str, default="true")
    parser.add_argument("--causal_query_only", type=str, default="true")
    parser.add_argument("--causal_gate_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--causal_seed_top_k", type=int, default=20)
    parser.add_argument("--causal_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--causal_damping", type=float, default=0.7)
    parser.add_argument("--causal_blend_dense_weight", type=float, default=0.35)
    parser.add_argument("--causal_blend_fact_weight", type=float, default=0.15)
    parser.add_argument("--causal_blend_graph_weight", type=float, default=0.50)
    parser.add_argument("--causal_margin_gate_enabled", type=str, default="false")
    parser.add_argument("--causal_margin_threshold", type=float, default=0.02)
    parser.add_argument("--causal_blend_top_k", type=int, default=0)
    parser.add_argument("--causal_engine_version", choices=["legacy", "v2"], default="legacy")
    parser.add_argument("--causal_v2_probe_mode", choices=["router", "always"], default="router")
    parser.add_argument("--causal_v2_graph_mode", choices=["causal", "general"], default="causal")
    parser.add_argument("--causal_v2_base_retrieval_mode", choices=["dense", "legacy_fact_graph", "general_relation_graph"], default="dense")
    parser.add_argument("--general_graph_related_to_weight", type=float, default=0.3)
    parser.add_argument("--general_graph_seed_top_k", type=int, default=10)
    parser.add_argument("--causal_v2_extraction_max_tokens", type=int, default=768)
    parser.add_argument("--causal_v2_extraction_retry_attempts", type=int, default=2)
    parser.add_argument("--causal_v2_extraction_workers", type=int, default=4)
    parser.add_argument("--causal_event_top_k", type=int, default=8)
    parser.add_argument("--causal_v2_max_hops", type=int, default=2)
    parser.add_argument("--causal_chain_top_k", type=int, default=6)
    parser.add_argument("--causal_context_max_items", type=int, default=0)
    parser.add_argument("--causal_er_similarity_threshold", type=float, default=0.92)
    parser.add_argument("--causal_er_text_threshold", type=float, default=0.55)
    parser.add_argument("--causal_v2_min_edge_confidence", type=float, default=0.7)
    parser.add_argument("--structure_rerank_enabled", type=str, default="true")
    parser.add_argument("--structure_rerank_top_n", type=int, default=40)
    parser.add_argument("--structure_rerank_bonus_weight", type=float, default=0.08)
    parser.add_argument("--structure_rerank_min_edge_support", type=int, default=2)
    parser.add_argument("--structure_rerank_max_top5_swaps", type=int, default=2)
    parser.add_argument("--structure_rerank_seed_top_k", type=int, default=4)
    parser.add_argument("--structure_rerank_max_hops", type=int, default=2)
    parser.add_argument("--structure_relation_probe_mode", choices=["off", "q6_factual", "general_factual", "general_factual_v2"], default="off",
                        help="Eval-only structure-graph predicate coverage probe. `off` preserves the current directed predicate vocabulary; `q6_factual` keeps the existing q6 audit alias; `general_factual` exposes the original shared factual edge family; `general_factual_v2` adds a minimal, default-off coverage patch for audited containment/reference predicates.")
    parser.add_argument("--structure_continuity_probe_mode", choices=["off", "city_state_alias", "location_alias"], default="off",
                        help="Eval-only structure node continuity probe. `city_state_alias` keeps the existing q6 audit alias; `location_alias` exposes the same high-confidence city/state -> bare-city closure under a reusable shared-layer name.")
    parser.add_argument("--structure_seed_target_bridge_mode", choices=["off", "allow_seed_target"], default="off",
                        help="Eval-only structure scorer mode. `off` preserves legacy bridge-edge acceptance; `allow_seed_target` also accepts explicit bridge edges whose target is already in the covered seed set.")
    parser.add_argument("--structure_rerank_margin_threshold", type=float, default=0.02)
    parser.add_argument("--rerank_require_non_empty", type=str, default="true")
    parser.add_argument("--retrieval_only", type=str, default="false")
    parser.add_argument("--gold_doc_reader", type=str, default="false",
                        help="Skip retrieval, feed gold docs directly to reader. Tests reader ceiling.")
    parser.add_argument("--oracle_reorder_k", type=int, default=0,
                        help="Move gold docs found within top-K to front. Tests reranker ceiling. 0=disabled.")
    parser.add_argument("--oracle_select_k", type=str, default="0",
                        help="Oracle select: comma-separated K values (e.g. '20,30,50,100'). From top-K pool, prioritize gold docs in reader's top-5. 0=disabled.")
    parser.add_argument("--cross_encoder_rerank", type=str, default="false",
                        help="Apply cross-encoder rerank on baseline top-K docs. Eval-time only.")
    parser.add_argument("--ce_model", type=str, default="/mnt/nvme/bge-reranker-v2-m3",
                        help="Cross-encoder model path or HF name for FlagEmbedding.")
    parser.add_argument("--ce_alpha", type=float, default=0.7,
                        help="Hybrid weight: alpha * ppr_norm + (1-alpha) * ce_norm. 1.0 = pure PPR.")
    parser.add_argument("--ce_window", type=int, default=20,
                        help="Number of top docs to rerank with cross-encoder.")
    parser.add_argument("--ce_device", type=str, default="cuda:1",
                        help="Device for cross-encoder model.")
    parser.add_argument("--setwise_selector", choices=["none", "bridge_greedy", "bridge_beam", "bridge_append", "learned_greedy", "requirement_beam"], default="none",
                        help="Apply a non-oracle setwise selector over a larger pool before reader top-k truncation.")
    parser.add_argument("--expand_base_k", type=int, default=10,
                        help="For --setwise_selector bridge_append, preserve baseline top-B before appending deep-pool bridge candidates.")
    parser.add_argument("--expand_min_structure_score", type=float, default=0.35,
                        help="For --setwise_selector bridge_append, minimum structure score required before a deep-pool bridge proposal can be appended.")
    parser.add_argument("--append_max_docs", type=int, default=3,
                        help="For --setwise_selector bridge_append, maximum number of deep-pool bridge docs appended to the preserved baseline prefix.")
    parser.add_argument("--append_policy", choices=sorted(APPEND_POLICIES), default="bridge",
                        help="For --setwise_selector bridge_append, how deep-pool docs are proposed before answer-oriented assembly.")
    parser.add_argument("--append_random_seed", type=int, default=0,
                        help="For --append_policy random_deep, deterministic seed used to sample deep-pool docs.")
    parser.add_argument("--gap_expand_mode", choices=sorted(GAP_EXPAND_MODES), default="heuristic",
                        help="For --append_policy gap_expand, heuristic gap detector mode. heuristic infers one gap type from the scaffold; flat_fallback_only always uses the raw query.")
    parser.add_argument("--gap_expand_max_queries", type=int, default=DEFAULT_GAP_EXPAND_MAX_QUERIES,
                        help="For --append_policy gap_expand, maximum number of scaffold-conditioned micro-queries emitted per query.")
    parser.add_argument("--assemble_mode", choices=sorted(ASSEMBLE_MODES), default="cross_encoder",
                        help="For --setwise_selector bridge_append, answer-oriented assembly rerank mode applied over the expanded candidate set.")
    parser.add_argument("--coverage_score_variant", choices=sorted(COVERAGE_SCORE_VARIANTS), default="qe_ce",
                        help="For --assemble_mode coverage, lexicographic score variant. qe_ce uses (CovQ, CovE, CE); qeb_ce adds CovB before CE.")
    parser.add_argument("--coverage_atom_source", choices=sorted(COVERAGE_ATOM_SOURCES), default="candidate_pool",
                        help="For --assemble_mode coverage, atom-universe source. candidate_pool preserves v1 behavior; baseline_prefix freezes A_E/A_B to the preserved baseline prefix; baseline_anchored keeps the baseline entity scaffold but admits candidate edges touching it.")
    parser.add_argument("--coverage_admissibility_mode", choices=sorted(COVERAGE_ADMISSIBILITY_MODES), default="off",
                        help="For --assemble_mode coverage with baseline_prefix atoms, optionally filter appended docs by frozen-structure marginal utility before exact search.")
    parser.add_argument("--setwise_score_mode", choices=["bridge", "closure_proxy", "set_closure"], default="bridge",
                        help="Scoring mode used by bridge_greedy / bridge_beam. bridge preserves the original structure score; closure_proxy uses a frontier-aware evidence-closure proxy; set_closure uses closure-aware proposals and re-ranks beam states with a set-level evidence score centered on explicit path connectivity.")
    parser.add_argument("--setwise_pool_k", type=int, default=20,
                        help="Candidate pool size used by the setwise selector.")
    parser.add_argument("--setwise_anchor_count", type=int, default=2,
                        help="Number of top-ranked anchor docs preserved before greedy bridge completion.")
    parser.add_argument("--setwise_reserve_top_m", type=int, default=0,
                        help="Preserve the top-M pool docs before setwise expansion; effective reserve is max(anchor_count, reserve_top_m).")
    parser.add_argument("--setwise_max_bridge_slots", type=int, default=0,
                        help="Maximum number of non-reserved slots actively filled by the setwise selector. 0 keeps the old unrestricted behavior.")
    parser.add_argument("--setwise_structure_max_hops", type=int, default=2,
                        help="Directed structure expansion depth used by the setwise selector.")
    parser.add_argument("--setwise_base_weight", type=float, default=0.25,
                        help="Weight assigned to baseline retrieval score inside setwise selection.")
    parser.add_argument("--setwise_structure_weight", type=float, default=0.60,
                        help="Weight assigned to structure bridge score inside setwise selection.")
    parser.add_argument("--setwise_novelty_weight", type=float, default=0.15,
                        help="Weight assigned to entity novelty inside setwise selection.")
    parser.add_argument("--setwise_non_anchor_title_dedup", type=string_to_bool, default=False,
                        help="If true, avoid selecting duplicate titles after the reserved prefix unless no alternatives remain.")
    parser.add_argument("--setwise_query_entity_source", choices=["seed", "question", "hybrid"], default="seed",
                        help="Source used for query-side closure features inside the setwise selector. seed keeps the legacy behavior; question uses question-derived entities throughout; hybrid keeps seed entities for bridge proposal but uses grounded question entities for set-level state scoring.")
    parser.add_argument("--setwise_gate_mode", choices=["none", "suffix_bridge", "suffix_bridge_precision", "suffix_bridge_saturation_guard"], default="none",
                        help="Per-query activation gate for bridge selectors. suffix_bridge only fires when an off-prefix candidate shows stronger bridge signal than the baseline suffix; suffix_bridge_precision adds an extra new-information / closure check before activation; suffix_bridge_saturation_guard keeps suffix_bridge activation but can revert saturated high-structure, weak-suffix states after beam selection.")
    parser.add_argument("--setwise_gate_min_structure_score", type=float, default=0.15,
                        help="Minimum structure score required for the adaptive setwise gate to activate on an off-prefix bridge candidate.")
    parser.add_argument("--setwise_gate_min_combined_margin", type=float, default=0.0,
                        help="Minimum combined-score advantage an off-prefix bridge candidate must have over the weakest baseline suffix doc before the gate activates.")
    parser.add_argument("--setwise_gate_min_closure_score", type=float, default=0.0,
                        help="Optional stronger bridge-gate threshold on closure_score. Used by suffix_bridge_precision.")
    parser.add_argument("--setwise_gate_min_novelty_score", type=float, default=0.0,
                        help="Optional stronger bridge-gate threshold on novelty_score. Used by suffix_bridge_precision.")
    parser.add_argument("--setwise_gate_min_frontier_gain", type=float, default=0.0,
                        help="Optional stronger bridge-gate threshold on frontier_gain_score. Used by suffix_bridge_precision.")
    parser.add_argument("--setwise_gate_min_path_coherence", type=float, default=0.0,
                        help="Optional stronger bridge-gate threshold on path_coherence_score. Used by suffix_bridge_precision.")
    parser.add_argument("--setwise_gate_max_avg_local_structure", type=float, default=0.95,
                        help="Eval-only saturation-guard threshold on the average local structure score across selected beam bridge docs. Used by suffix_bridge_saturation_guard.")
    parser.add_argument("--setwise_gate_min_suffix_base_mean", type=float, default=0.15,
                        help="Eval-only saturation-guard threshold on the final state's suffix_base_mean. Used by suffix_bridge_saturation_guard.")
    parser.add_argument("--setwise_reader_order_probe_mode",
                        choices=sorted(SETWISE_READER_ORDER_PROBE_MODES),
                        default="none",
                        help="Eval-only reader-side causal probe. Reorders the final reader top-k without changing the selected evidence set.")
    parser.add_argument("--setwise_beam_width", type=int, default=4,
                        help="Beam width used when --setwise_selector bridge_beam.")
    parser.add_argument("--setwise_beam_expand_per_state", type=int, default=4,
                        help="Number of candidates expanded per beam state for --setwise_selector bridge_beam.")
    parser.add_argument("--setwise_beam_projected_shortlist_factor", type=int, default=DEFAULT_SET_CLOSURE_PROJECTED_SHORTLIST_FACTOR,
                        help="When --setwise_score_mode set_closure, evaluate projected set-level state scores for up to beam_expand_per_state * factor doc-level proposals before keeping the final beam expansions. 1 preserves the legacy behavior.")
    parser.add_argument("--setwise_late_rerank_enabled", type=string_to_bool, default=False,
                        help="If true, run the LLM once per query to rerank a tiny shortlist of completed bridge_beam evidence sets.")
    parser.add_argument("--setwise_late_rerank_candidate_count", type=int, default=4,
                        help="Maximum number of completed evidence-set candidates exposed to the LLM late reranker.")
    parser.add_argument("--setwise_late_rerank_include_baseline", type=string_to_bool, default=True,
                        help="Include the original baseline top-k evidence set in the late LLM rerank shortlist.")
    parser.add_argument("--setwise_late_rerank_doc_char_limit", type=int, default=280,
                        help="Per-document character budget when serializing evidence sets for late LLM reranking.")
    parser.add_argument("--setwise_late_rerank_policy", choices=["always", "tiebreak"], default="always",
                        help="Override policy for the late rerank judge. always lets the judge replace heuristic-best whenever it picks a different candidate; tiebreak only allows overrides when the chosen candidate stays within a small heuristic state-score gap.")
    parser.add_argument("--setwise_late_rerank_max_state_score_gap", type=float, default=0.0,
                        help="Maximum absolute state-score gap allowed when --setwise_late_rerank_policy=tiebreak. Candidates without state scores are blocked from overriding.")
    parser.add_argument("--setwise_late_rerank_judge_backend", choices=["inherit", "responses", "chat_completions"], default="inherit",
                        help="Judge backend used by late evidence-set rerank. inherit uses the main HippoRAG LLM path; responses uses raw-text JSON prompting with local parsing, while chat_completions uses raw JSON prompting and falls back to streaming collection for provider compatibility.")
    parser.add_argument("--setwise_late_rerank_judge_model", type=str, default="",
                        help="Optional model name for the separate late-rerank judge. Defaults to the main --llm_name when omitted.")
    parser.add_argument("--setwise_late_rerank_judge_base_url", type=str, default="",
                        help="Optional base URL for the separate late-rerank judge. Leave empty to use the provider default endpoint.")
    parser.add_argument("--setwise_late_rerank_judge_api_key", type=str, default="",
                        help="Optional API key for the separate late-rerank judge. Prefer env vars for security.")
    parser.add_argument("--setwise_late_rerank_judge_api_key_env", type=str, default="OPENAI_API_KEY",
                        help="Environment variable used to resolve the separate late-rerank judge API key when --setwise_late_rerank_judge_api_key is empty.")
    parser.add_argument("--setwise_late_rerank_judge_reasoning_effort", type=str, default="",
                        help="Optional reasoning effort passed to compatible judge models, e.g. low/medium/high.")
    parser.add_argument("--setwise_late_rerank_judge_timeout_s", type=float, default=120.0,
                        help="Timeout in seconds for the separate late-rerank judge API calls.")
    parser.add_argument("--setwise_state_path_connectivity_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["path_connectivity"],
                        help="Set-level beam score weight for explicit chain/path connectivity under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_reachable_doc_ratio_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["reachable_doc_ratio"],
                        help="Set-level beam score weight for the fraction of suffix docs that attach to the connected chain under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_query_reachability_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["query_reachability"],
                        help="Set-level beam score weight for query entities that are reachable from the connected chain under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_support_mean_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["support_mean"],
                        help="Set-level beam score weight for suffix support mean under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_support_min_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["support_min"],
                        help="Set-level beam score weight for suffix support minimum under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_closure_mean_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["closure_mean"],
                        help="Set-level beam score weight for suffix closure mean under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_suffix_base_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["suffix_base_mean"],
                        help="Set-level beam score weight for suffix-only baseline relevance under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_query_coverage_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["query_coverage"],
                        help="Set-level beam score weight for query-entity coverage under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_frontier_ratio_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["frontier_ratio"],
                        help="Set-level beam score weight for frontier support ratio under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_state_redundancy_penalty_weight", type=float, default=DEFAULT_SET_CLOSURE_STATE_WEIGHT_CONFIG["redundancy_penalty"],
                        help="Set-level beam score penalty weight for redundancy under --setwise_score_mode set_closure.")
    parser.add_argument("--setwise_model_path", type=str, default="",
                        help="Joblib bundle path used by --setwise_selector learned_greedy.")
    parser.add_argument("--setwise_requirement_cache_path", type=str, default="",
                        help="Offline requirement cache JSON used by --setwise_selector requirement_beam.")
    parser.add_argument("--setwise_requirement_mode", choices=["oracle", "learned"], default="oracle",
                        help="Mode used by requirement_beam. oracle uses cached requirement annotations directly; learned adds a lightweight matcher as a proposal prior.")
    parser.add_argument("--setwise_requirement_model_path", type=str, default="",
                        help="Optional requirement matcher joblib bundle used when --setwise_requirement_mode learned.")
    parser.add_argument("--setwise_requirement_reserve_policy", choices=["fixed", "adaptive_requirement_count"], default=DEFAULT_REQUIREMENT_BEAM_RESERVE_POLICY,
                        help="Reserve-prefix policy used by requirement_beam. fixed preserves the configured anchor/reserve counts; adaptive_requirement_count reduces the reserved prefix on queries with very small positive-requirement sets.")
    parser.add_argument("--setwise_requirement_annotation_pool_k", type=int, default=DEFAULT_REQUIREMENT_ANNOTATION_POOL_K,
                        help="Expected annotation pool size stored in the requirement cache. Used for reporting and cache validation only.")
    parser.add_argument("--setwise_requirement_smooth_tau", type=float, default=DEFAULT_REQUIREMENT_SMOOTH_TAU,
                        help="Smooth-min temperature for requirement support completeness.")
    parser.add_argument("--setwise_requirement_counterfactual_tau", type=float, default=DEFAULT_REQUIREMENT_CF_TAU,
                        help="Soft worst-case temperature for counterfactual leakage.")
    parser.add_argument("--setwise_requirement_live_annotation_pool_k", type=int, default=0,
                        help="Optional eval-only annotation frontier override for requirement_beam. When >0, rebuild requirement annotations up to this pool depth during live smoke eval.")
    parser.add_argument("--setwise_requirement_live_annotation_score_mode", choices=["heuristic", "hybrid"], default="heuristic",
                        help="Score mode used for eval-only live requirement annotation rebuilds.")
    parser.add_argument("--setwise_requirement_live_atomic_model_path", type=str, default="",
                        help="Atomic need-unit scorer bundle used when --setwise_requirement_live_annotation_score_mode hybrid.")
    parser.add_argument("--setwise_requirement_live_source_expand_factor", type=int, default=0,
                        help="Optional eval-only override for requirement_beam candidate source width. When >0, replaces beam_projected_shortlist_factor for requirement_beam only.")
    parser.add_argument("--setwise_requirement_exposure_watch_titles", type=str, default=",".join(DEFAULT_REQUIREMENT_EXPOSURE_WATCH_TITLES),
                        help="Comma-separated doc titles to summarize in requirement_beam exposure traces.")
    parser.add_argument("--setwise_requirement_probe_force_source_titles", type=str, default="",
                        help="Eval-only diagnostic: comma-separated titles forced into requirement-beam candidate source when present in the pool.")
    parser.add_argument("--setwise_requirement_probe_force_shortlist_titles", type=str, default="",
                        help="Eval-only diagnostic: comma-separated titles forced into requirement-beam candidate shortlist when already present in candidate source.")
    parser.add_argument("--setwise_requirement_probe_force_final_titles", type=str, default="",
                        help="Eval-only diagnostic: comma-separated titles forced into the final reader evidence set when present in the pool.")
    parser.add_argument("--setwise_requirement_probe_force_pool_gold_into_final", type=string_to_bool, default=False,
                        help="Eval-only diagnostic: for each query, force gold titles that are already inside the current pool into the final reader evidence set.")
    parser.add_argument("--setwise_requirement_probe_source_sort_mode", choices=["combined", "support_bonus"], default=DEFAULT_REQUIREMENT_SOURCE_SORT_MODE,
                        help="Eval-only source ranking mode for requirement_beam. combined preserves legacy source admission; support_bonus adds a small support_completeness gain bonus before the source cutoff.")
    parser.add_argument("--setwise_requirement_probe_source_support_gain_weight", type=float, default=0.0,
                        help="Eval-only support gain weight used when --setwise_requirement_probe_source_sort_mode support_bonus.")
    parser.add_argument("--setwise_requirement_probe_shortlist_sort_mode", choices=["margin_first", "positive_only"], default=DEFAULT_REQUIREMENT_SHORTLIST_SORT_MODE,
                        help="Eval-only shortlist ranking mode for requirement_beam. margin_first uses utility_margin first; positive_only uses support_completeness gain first.")
    parser.add_argument("--setwise_requirement_probe_bridge_bonus_mode", choices=["off", "variable_binding"], default="off",
                        help="Eval-only need-unit bridge bonus mode for requirement_beam. off preserves legacy scoring; variable_binding grants a small relation_hop override when predecessor coverage and entity-binding continuity are present.")
    parser.add_argument("--setwise_requirement_probe_bridge_bonus_weight", type=float, default=0.0,
                        help="Eval-only bridge bonus weight used when --setwise_requirement_probe_bridge_bonus_mode variable_binding.")
    parser.add_argument("--retrieval_cache_json", type=str, default="",
                        help="Optional retrieval cache used to reuse retrieved docs/doc_scores/retrieval traces and skip retrieval.")
    parser.add_argument("--save_retrieval_cache_json", type=str, default="",
                        help="Optional output path for writing a reusable retrieval cache after the baseline retrieval stage.")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    if save_dir == "outputs":
        save_dir = os.path.join(save_dir, dataset_name)
    else:
        save_dir = f"{save_dir}_{dataset_name}"
    args.save_dir = save_dir

    corpus_path, sample_path = resolve_dataset_paths(dataset_name)
    corpus = json.load(corpus_path.open())
    samples = json.load(sample_path.open())

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    chunk_id_to_doc_text = build_chunk_id_to_doc_text(corpus)
    queries = [sample["question"] for sample in samples]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset_name, corpus=corpus)
    retrieval_only = string_to_bool(args.retrieval_only)
    gold_doc_reader = string_to_bool(args.gold_doc_reader)
    oracle_reorder_k = int(args.oracle_reorder_k)
    oracle_select_ks = [int(x) for x in args.oracle_select_k.split(",") if int(x) > 0]

    config = build_config(args, corpus_len=len(corpus))
    logging.basicConfig(level=logging.INFO)
    baseline_report_path = str(args.baseline_report_json or "").strip()
    retrieval_cache_path = str(args.retrieval_cache_json or "").strip()
    save_retrieval_cache_path = str(args.save_retrieval_cache_json or "").strip()

    oracle_reorder_qa_results = None
    setwise_selector_results = None
    setwise_selector_query_traces = None
    expand_assemble_results = None
    expand_assemble_query_traces = None
    learned_model_bundle = None
    state_weight_config = {
        "path_connectivity": float(args.setwise_state_path_connectivity_weight),
        "reachable_doc_ratio": float(args.setwise_state_reachable_doc_ratio_weight),
        "query_reachability": float(args.setwise_state_query_reachability_weight),
        "support_mean": float(args.setwise_state_support_mean_weight),
        "support_min": float(args.setwise_state_support_min_weight),
        "closure_mean": float(args.setwise_state_closure_mean_weight),
        "suffix_base_mean": float(args.setwise_state_suffix_base_weight),
        "query_coverage": float(args.setwise_state_query_coverage_weight),
        "frontier_ratio": float(args.setwise_state_frontier_ratio_weight),
        "redundancy_penalty": float(args.setwise_state_redundancy_penalty_weight),
    }
    setwise_selector = args.setwise_selector.lower() if not gold_doc_reader else "none"
    if setwise_selector == "learned_greedy":
        if not args.setwise_model_path:
            raise ValueError("--setwise_model_path is required when --setwise_selector learned_greedy")
        learned_model_bundle = load_learned_model_bundle(args.setwise_model_path)
    requirement_selector_bundle = None
    if setwise_selector == "requirement_beam":
        if not args.setwise_requirement_cache_path:
            raise ValueError("--setwise_requirement_cache_path is required when --setwise_selector requirement_beam")
        requirement_cache = load_requirement_cache(args.setwise_requirement_cache_path)
        requirement_model_bundle = None
        requirement_model_path = ""
        live_annotation_score_mode = str(args.setwise_requirement_live_annotation_score_mode or "heuristic").strip().lower()
        live_atomic_model_path = str(args.setwise_requirement_live_atomic_model_path or "").strip()
        live_atomic_scorer_bundle = None
        if str(args.setwise_requirement_mode).strip().lower() == "learned":
            requirement_model_path = str(args.setwise_requirement_model_path or "").strip()
            if not requirement_model_path:
                raise ValueError("--setwise_requirement_model_path is required when --setwise_requirement_mode learned")
            requirement_model_bundle = load_requirement_model_bundle(requirement_model_path)
        if live_annotation_score_mode == "hybrid":
            if not live_atomic_model_path:
                raise ValueError(
                    "--setwise_requirement_live_atomic_model_path is required when "
                    "--setwise_requirement_live_annotation_score_mode hybrid"
                )
            live_atomic_scorer_bundle = load_need_unit_atomic_model_bundle(live_atomic_model_path)
        requirement_selector_bundle = {
            "cache": requirement_cache,
            "cache_path": str(args.setwise_requirement_cache_path),
            "mode": str(args.setwise_requirement_mode).strip().lower(),
            "model_bundle": requirement_model_bundle,
            "model_path": requirement_model_path,
            "annotation_pool_k": int(args.setwise_requirement_annotation_pool_k),
            "smooth_tau": float(args.setwise_requirement_smooth_tau),
            "counterfactual_tau": float(args.setwise_requirement_counterfactual_tau),
            "live_annotation_pool_k": int(args.setwise_requirement_live_annotation_pool_k),
            "live_annotation_score_mode": live_annotation_score_mode,
            "live_atomic_model_path": live_atomic_model_path,
            "live_atomic_scorer_bundle": live_atomic_scorer_bundle,
            "live_source_expand_factor": int(args.setwise_requirement_live_source_expand_factor),
            "exposure_watch_titles_csv": str(args.setwise_requirement_exposure_watch_titles or ""),
            "exposure_watch_titles": parse_title_csv(
                args.setwise_requirement_exposure_watch_titles,
                default=DEFAULT_REQUIREMENT_EXPOSURE_WATCH_TITLES,
            ),
            "probe_force_source_titles": parse_title_csv(args.setwise_requirement_probe_force_source_titles),
            "probe_force_shortlist_titles": parse_title_csv(args.setwise_requirement_probe_force_shortlist_titles),
            "probe_force_final_titles": parse_title_csv(args.setwise_requirement_probe_force_final_titles),
            "probe_force_pool_gold_into_final": bool(args.setwise_requirement_probe_force_pool_gold_into_final),
            "probe_source_sort_mode": normalize_requirement_source_sort_mode(args.setwise_requirement_probe_source_sort_mode),
            "probe_source_support_gain_weight": float(args.setwise_requirement_probe_source_support_gain_weight),
            "probe_shortlist_sort_mode": normalize_requirement_shortlist_sort_mode(args.setwise_requirement_probe_shortlist_sort_mode),
            "probe_bridge_bonus_mode": normalize_requirement_bridge_bonus_mode(args.setwise_requirement_probe_bridge_bonus_mode),
            "probe_bridge_bonus_weight": float(args.setwise_requirement_probe_bridge_bonus_weight),
        }
    late_rerank_judge_bundle = SetwiseLateRerankJudgeBundle(
        infer_fn=None,
        model_name=str(args.llm_request_name or args.llm_name),
        backend="inherit",
        base_url=str(args.llm_base_url).strip() if args.llm_base_url else None,
        response_format=None,
    )
    baseline_report_payload = load_baseline_report_payload(baseline_report_path) if baseline_report_path else None
    retrieval_cache_payload = load_retrieval_cache_payload(retrieval_cache_path) if retrieval_cache_path else None

    if gold_doc_reader:
        # Exp2: Gold-doc reader — skip retrieval, feed gold docs to reader
        hipporag = HippoRAG(global_config=config)
        hipporag.index(docs)
        gold_query_solutions = [
            QuerySolution(
                question=query,
                docs=gold_docs[q_idx],
                doc_scores=np.ones(len(gold_docs[q_idx])),
            )
            for q_idx, query in enumerate(queries)
        ]
        # rag_qa accepts QuerySolution list directly — skips retrieve()
        query_solutions, responses, metadata, _, overall_qa_results = hipporag.rag_qa(
            queries=gold_query_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )
        overall_retrieval_result = {"note": "gold_doc_reader mode — retrieval metrics are N/A (oracle-by-construction)"}
        effective_gold_answers = gold_answers
    else:
        hipporag = HippoRAG(global_config=config)
        hipporag.index(docs)
        if retrieval_cache_payload is not None:
            ensure_runtime_objects_for_cached_retrieval(hipporag)
            query_solutions = [
                QuerySolution(
                    question=query,
                    docs=[],
                    doc_scores=np.asarray([], dtype=float),
                )
                for query in queries
            ]
            hydrate_query_solutions_from_retrieval_cache(
                query_solutions=query_solutions,
                retrieval_cache_payload=retrieval_cache_payload,
                chunk_id_to_doc_text=chunk_id_to_doc_text,
                gold_docs=gold_docs,
            )
            overall_retrieval_result = dict(retrieval_cache_payload.get("overall_metrics") or {})
            logging.getLogger(__name__).info(
                "Reused retrieval from cache %s and skipped retrieval stage.",
                retrieval_cache_payload.get("path"),
            )
        else:
            query_solutions = None
            overall_retrieval_result = None

        if retrieval_only:
            if query_solutions is None:
                query_solutions, overall_retrieval_result = hipporag.retrieve(
                    queries=queries,
                    gold_docs=gold_docs,
                )
            responses = []
            metadata = []
            overall_qa_results = {}
            effective_gold_answers = None
        elif baseline_report_payload is not None:
            if query_solutions is None:
                query_solutions, overall_retrieval_result = hipporag.retrieve(
                    queries=queries,
                    gold_docs=gold_docs,
                )
            hydrate_query_solutions_from_baseline_report(
                query_solutions=query_solutions,
                baseline_report_payload=baseline_report_payload,
                gold_docs=gold_docs,
            )
            responses = []
            metadata = []
            overall_qa_results = dict(baseline_report_payload.get("overall_metrics") or {})
            effective_gold_answers = gold_answers
            logging.getLogger(__name__).info(
                "Reused baseline QA from report %s and skipped baseline reader pass.",
                baseline_report_payload.get("path"),
            )
        elif query_solutions is not None:
            query_solutions, responses, metadata, _, overall_qa_results = hipporag.rag_qa(
                queries=query_solutions,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
            effective_gold_answers = gold_answers
        else:
            query_solutions, responses, metadata, overall_retrieval_result, overall_qa_results = hipporag.rag_qa(
                queries=queries,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
            effective_gold_answers = gold_answers

        if save_retrieval_cache_path and query_solutions:
            write_retrieval_cache_payload(
                output_path=save_retrieval_cache_path,
                query_solutions=query_solutions,
                doc_text_to_chunk_id=doc_text_to_chunk_id,
                overall_metrics=overall_retrieval_result,
            )
            logging.getLogger(__name__).info(
                "Wrote retrieval cache to %s",
                save_retrieval_cache_path,
            )

        should_build_external_judge_bundle = (
            (bool(args.setwise_late_rerank_enabled) and setwise_selector == "bridge_beam")
            or (
                setwise_selector == "bridge_append"
                and normalize_assemble_mode(args.assemble_mode) in ACTION_SWAP_V0_JUDGE_MODES
            )
        )
        if should_build_external_judge_bundle:
            late_rerank_judge_bundle = build_setwise_late_rerank_judge_bundle(
                args=args,
                fallback_model_name=(
                    getattr(hipporag.global_config, "llm_request_name", None)
                    or hipporag.global_config.llm_name
                ),
                fallback_base_url=hipporag.global_config.llm_base_url,
            )

        # Exp3: Oracle reorder within top-K
        if oracle_reorder_k > 0 and not retrieval_only:
            qa_em = QAExactMatch(global_config=config)
            qa_f1 = QAF1Score(global_config=config)
            reordered_solutions = []
            full_support_in_topk_count = 0
            for q_idx, qs in enumerate(query_solutions):
                gold_set = set(gold_docs[q_idx])
                top_k_docs = qs.docs[:oracle_reorder_k]
                gold_in_topk = [d for d in top_k_docs if d in gold_set]
                non_gold_in_topk = [d for d in top_k_docs if d not in gold_set]
                rest = qs.docs[oracle_reorder_k:]
                reordered_docs = gold_in_topk + non_gold_in_topk + rest
                if gold_set.issubset(set(top_k_docs)):
                    full_support_in_topk_count += 1
                reordered_qs = QuerySolution(
                    question=qs.question,
                    docs=reordered_docs,
                    doc_scores=qs.doc_scores,
                    gold_docs=gold_docs[q_idx],
                )
                reordered_solutions.append(reordered_qs)
            # Run QA on reordered docs via rag_qa (skips retrieve since input is QuerySolution)
            reordered_solutions, _, _, _, reorder_qa_results = hipporag.rag_qa(
                queries=reordered_solutions,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
            reordered_answers = [qs.answer for qs in reordered_solutions]
            reorder_em = reorder_qa_results.get("ExactMatch", 0.0)
            reorder_f1 = reorder_qa_results.get("F1", 0.0)
            baseline_em = overall_qa_results.get("ExactMatch", 0.0) if overall_qa_results else 0.0
            oracle_reorder_qa_results = {
                "oracle_reorder_k": oracle_reorder_k,
                "oracle_reorder_EM": round(float(reorder_em), 4),
                "oracle_reorder_F1": round(float(reorder_f1), 4),
                "baseline_EM": round(float(baseline_em), 4),
                "EM_delta": round(float(reorder_em) - float(baseline_em), 4),
                "full_support_in_top_k_rate": round(full_support_in_topk_count / max(1, len(queries)), 4),
                "full_support_in_top_k_count": full_support_in_topk_count,
            }

    # Exp4: Oracle select sweep — ceiling curve across multiple K values
    oracle_select_qa_results = None
    if oracle_select_ks and not retrieval_only and not gold_doc_reader:
        qa_top = config.qa_top_k  # typically 5
        logger = logging.getLogger(__name__)
        baseline_em = overall_qa_results.get("ExactMatch", 0.0) if overall_qa_results else 0.0
        baseline_f1 = overall_qa_results.get("F1", 0.0) if overall_qa_results else 0.0

        # --- Minimal full-support depth per query ---
        # Smallest K such that all gold docs are in docs[:K]
        per_query_support_depth = []
        for q_idx, qs in enumerate(query_solutions):
            gold_set = set(gold_docs[q_idx])
            found = set()
            depth = None
            for rank, d in enumerate(qs.docs, 1):
                if d in gold_set:
                    found.add(d)
                if found == gold_set:
                    depth = rank
                    break
            per_query_support_depth.append(depth)  # None = never fully supported

        # Bucket support depth stats
        depth_by_bucket: dict[int, list] = {}
        for q_idx in range(len(query_solutions)):
            n_gold = len(set(gold_docs[q_idx]))
            if n_gold not in depth_by_bucket:
                depth_by_bucket[n_gold] = []
            depth_by_bucket[n_gold].append(per_query_support_depth[q_idx])

        support_depth_summary = {}
        for n_gold in sorted(depth_by_bucket.keys()):
            depths = depth_by_bucket[n_gold]
            finite = [d for d in depths if d is not None]
            support_depth_summary[f"{n_gold}-doc"] = {
                "count": len(depths),
                "fully_supported": len(finite),
                "never_supported": len(depths) - len(finite),
                "median_depth": round(float(np.median(finite)), 1) if finite else None,
                "mean_depth": round(float(np.mean(finite)), 1) if finite else None,
                "p90_depth": round(float(np.percentile(finite, 90)), 1) if finite else None,
                "max_depth": int(max(finite)) if finite else None,
            }
        logger.info("Minimal full-support depth:")
        for bk, bv in support_depth_summary.items():
            logger.info(f"  {bk}: supported={bv['fully_supported']}/{bv['count']}, "
                         f"median={bv['median_depth']}, mean={bv['mean_depth']}, p90={bv['p90_depth']}")

        # --- Oracle select sweep over K values ---
        sweep_results = {}
        for sel_k in sorted(oracle_select_ks):
            logger.info(f"Oracle select: pool={sel_k}, reader sees top-{qa_top}")
            selected_solutions = []
            bucket_stats: dict[int, dict] = {}

            for q_idx, qs in enumerate(query_solutions):
                pool = qs.docs[:sel_k]
                gold_set = set(gold_docs[q_idx])
                n_gold = len(gold_set)

                gold_in_pool = [d for d in pool if d in gold_set]
                non_gold_in_pool = [d for d in pool if d not in gold_set]
                selected_docs = (gold_in_pool + non_gold_in_pool)[:max(sel_k, qa_top)]
                rest = qs.docs[sel_k:]
                all_docs = selected_docs + rest

                selected_qs = QuerySolution(
                    question=qs.question,
                    docs=all_docs,
                    doc_scores=qs.doc_scores,
                    gold_docs=gold_docs[q_idx],
                )
                selected_solutions.append(selected_qs)

                if n_gold not in bucket_stats:
                    bucket_stats[n_gold] = {"count": 0, "gold_found_sum": 0, "fs_sum": 0}
                bucket_stats[n_gold]["count"] += 1
                bucket_stats[n_gold]["gold_found_sum"] += len(gold_in_pool)
                if gold_set.issubset(set(pool)):
                    bucket_stats[n_gold]["fs_sum"] += 1

            # Run QA on oracle-selected docs
            selected_solutions, _, _, _, select_qa_results = hipporag.rag_qa(
                queries=selected_solutions,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
            select_em = select_qa_results.get("ExactMatch", 0.0)
            select_f1 = select_qa_results.get("F1", 0.0)

            # Per-bucket EM/F1 breakdown
            qa_em_metric = QAExactMatch(global_config=config)
            qa_f1_metric = QAF1Score(global_config=config)
            sel_answers = [qs.answer or "" for qs in selected_solutions]
            _, per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, sel_answers)
            _, per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, sel_answers)
            bucket_em: dict[int, list] = {}
            bucket_f1: dict[int, list] = {}
            for q_idx in range(len(selected_solutions)):
                n_gold = len(set(gold_docs[q_idx]))
                if n_gold not in bucket_em:
                    bucket_em[n_gold] = []
                    bucket_f1[n_gold] = []
                bucket_em[n_gold].append(per_query_em[q_idx]["ExactMatch"])
                bucket_f1[n_gold].append(per_query_f1[q_idx]["F1"])

            bucket_breakdown = {}
            for n_gold in sorted(bucket_stats.keys()):
                bs = bucket_stats[n_gold]
                bucket_breakdown[f"{n_gold}-doc"] = {
                    "count": bs["count"],
                    "avg_gold_found_in_pool": round(bs["gold_found_sum"] / max(1, bs["count"]), 3),
                    "full_support_rate": round(bs["fs_sum"] / max(1, bs["count"]), 4),
                    "EM": round(float(np.mean(bucket_em.get(n_gold, [0]))), 4),
                    "F1": round(float(np.mean(bucket_f1.get(n_gold, [0]))), 4),
                }

            total_fs = sum(bs["fs_sum"] for bs in bucket_stats.values())
            sweep_results[f"K={sel_k}"] = {
                "pool_k": sel_k,
                "oracle_select_EM": round(float(select_em), 4),
                "oracle_select_F1": round(float(select_f1), 4),
                "EM_delta": round(float(select_em) - float(baseline_em), 4),
                "F1_delta": round(float(select_f1) - float(baseline_f1), 4),
                "full_support_in_pool_rate": round(total_fs / max(1, len(queries)), 4),
                "bucket_breakdown": bucket_breakdown,
            }
            logger.info(f"Oracle select@{sel_k}: EM={select_em:.4f} (delta={float(select_em)-float(baseline_em):+.4f}), "
                         f"F1={select_f1:.4f}, FS_in_pool={total_fs}/{len(queries)}")
            for bk, bv in bucket_breakdown.items():
                logger.info(f"  {bk}: count={bv['count']}, FS={bv['full_support_rate']}, EM={bv['EM']}, F1={bv['F1']}")

        oracle_select_qa_results = {
            "baseline_EM": round(float(baseline_em), 4),
            "baseline_F1": round(float(baseline_f1), 4),
            "support_depth": support_depth_summary,
            "sweep": sweep_results,
        }

    if setwise_selector != "none" and not retrieval_only and query_solutions:
        logger = logging.getLogger(__name__)
        selected_solutions, selector_summary = apply_setwise_selector(
            hipporag=hipporag,
            query_solutions=query_solutions,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
            pool_k=int(args.setwise_pool_k),
            qa_top_k=int(config.qa_top_k),
            selector_name=setwise_selector,
            score_mode=str(args.setwise_score_mode),
            anchor_count=int(args.setwise_anchor_count),
            reserve_top_m=int(args.setwise_reserve_top_m),
            max_bridge_slots=int(args.setwise_max_bridge_slots),
            structure_max_hops=int(args.setwise_structure_max_hops),
            structure_seed_target_bridge_mode=str(args.structure_seed_target_bridge_mode),
            base_weight=float(args.setwise_base_weight),
            structure_weight=float(args.setwise_structure_weight),
            novelty_weight=float(args.setwise_novelty_weight),
            learned_model_bundle=learned_model_bundle,
            beam_width=int(args.setwise_beam_width),
            beam_expand_per_state=int(args.setwise_beam_expand_per_state),
            beam_projected_shortlist_factor=int(args.setwise_beam_projected_shortlist_factor),
            non_anchor_title_dedup=bool(args.setwise_non_anchor_title_dedup),
            query_entity_source=str(args.setwise_query_entity_source),
            gate_mode=str(args.setwise_gate_mode),
            gate_min_structure_score=float(args.setwise_gate_min_structure_score),
            gate_min_combined_margin=float(args.setwise_gate_min_combined_margin),
            gate_min_closure_score=float(args.setwise_gate_min_closure_score),
            gate_min_novelty_score=float(args.setwise_gate_min_novelty_score),
            gate_min_frontier_gain=float(args.setwise_gate_min_frontier_gain),
            gate_min_path_coherence=float(args.setwise_gate_min_path_coherence),
            gate_max_avg_local_structure=float(args.setwise_gate_max_avg_local_structure),
            gate_min_suffix_base_mean=float(args.setwise_gate_min_suffix_base_mean),
            setwise_reader_order_probe_mode=str(args.setwise_reader_order_probe_mode),
            state_weight_config=state_weight_config,
            late_rerank_enabled=bool(args.setwise_late_rerank_enabled),
            late_rerank_candidate_count=int(args.setwise_late_rerank_candidate_count),
            late_rerank_include_baseline=bool(args.setwise_late_rerank_include_baseline),
            late_rerank_doc_char_limit=int(args.setwise_late_rerank_doc_char_limit),
            late_rerank_policy=str(args.setwise_late_rerank_policy),
            late_rerank_max_state_score_gap=float(args.setwise_late_rerank_max_state_score_gap),
            late_rerank_judge_bundle=late_rerank_judge_bundle,
            requirement_selector_bundle=requirement_selector_bundle,
            requirement_reserve_policy=str(args.setwise_requirement_reserve_policy),
            expand_base_k=int(args.expand_base_k),
            expand_min_structure_score=float(args.expand_min_structure_score),
            assemble_mode=str(args.assemble_mode),
            coverage_score_variant=str(args.coverage_score_variant),
            coverage_atom_source=str(args.coverage_atom_source),
            coverage_admissibility_mode=str(args.coverage_admissibility_mode),
            append_max_docs=int(args.append_max_docs),
            append_policy=str(args.append_policy),
            append_random_seed=int(args.append_random_seed),
            gap_expand_mode=str(args.gap_expand_mode),
            gap_expand_max_queries=int(args.gap_expand_max_queries),
            ce_model=str(args.ce_model),
            ce_device=str(args.ce_device),
        )
        selected_solutions, _, _, _, selector_qa_results = hipporag.rag_qa(
            queries=selected_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )

        qa_em_metric = QAExactMatch(global_config=config)
        qa_f1_metric = QAF1Score(global_config=config)
        bl_answers = [qs.answer or "" for qs in query_solutions]
        selector_answers = [qs.answer or "" for qs in selected_solutions]
        _, bl_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, bl_answers)
        _, bl_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, bl_answers)
        _, selector_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, selector_answers)
        _, selector_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, selector_answers)

        bucket_results = {}
        for q_idx in range(len(queries)):
            n_gold = len(set(gold_docs[q_idx]))
            if n_gold not in bucket_results:
                bucket_results[n_gold] = {
                    "baseline_em": [],
                    "selector_em": [],
                    "baseline_f1": [],
                    "selector_f1": [],
                    "count": 0,
                }
            bucket_results[n_gold]["count"] += 1
            bucket_results[n_gold]["baseline_em"].append(bl_per_query_em[q_idx]["ExactMatch"])
            bucket_results[n_gold]["selector_em"].append(selector_per_query_em[q_idx]["ExactMatch"])
            bucket_results[n_gold]["baseline_f1"].append(bl_per_query_f1[q_idx]["F1"])
            bucket_results[n_gold]["selector_f1"].append(selector_per_query_f1[q_idx]["F1"])

        bucket_summary = {}
        for n_gold, data in sorted(bucket_results.items()):
            bucket_summary[f"{n_gold}_doc"] = {
                "count": data["count"],
                "baseline_EM": round(float(np.mean(data["baseline_em"])), 4),
                "selector_EM": round(float(np.mean(data["selector_em"])), 4),
                "EM_delta": round(float(np.mean(data["selector_em"])) - float(np.mean(data["baseline_em"])), 4),
                "baseline_F1": round(float(np.mean(data["baseline_f1"])), 4),
                "selector_F1": round(float(np.mean(data["selector_f1"])), 4),
                "F1_delta": round(float(np.mean(data["selector_f1"])) - float(np.mean(data["baseline_f1"])), 4),
            }

        selector_retrieval_metrics = compute_retrieval_recall_metrics(
            query_solutions=selected_solutions,
            gold_docs=gold_docs,
        )

        selector_em = selector_qa_results.get("ExactMatch", 0.0)
        selector_f1 = selector_qa_results.get("F1", 0.0)
        baseline_em = overall_qa_results.get("ExactMatch", 0.0) if overall_qa_results else 0.0
        baseline_f1 = overall_qa_results.get("F1", 0.0) if overall_qa_results else 0.0
        if setwise_selector == "bridge_append":
            expand_assemble_results = {
                "selector": setwise_selector,
                "score_mode": str(args.setwise_score_mode),
                "pool_k": int(args.setwise_pool_k),
                "expand_base_k": int(args.expand_base_k),
                "append_max_docs": int(args.append_max_docs),
                "append_policy": normalize_append_policy(args.append_policy),
                "append_random_seed": int(args.append_random_seed),
                "gap_expand_mode": normalize_gap_expand_mode(args.gap_expand_mode),
                "gap_expand_max_queries": int(max(int(args.gap_expand_max_queries), 1)),
                "expand_min_structure_score": round(float(args.expand_min_structure_score), 4),
                "assemble_mode": normalize_assemble_mode(args.assemble_mode),
                "coverage_score_variant": normalize_coverage_score_variant(args.coverage_score_variant),
                "coverage_atom_source": normalize_coverage_atom_source(args.coverage_atom_source),
                "coverage_admissibility_mode": normalize_coverage_admissibility_mode(args.coverage_admissibility_mode),
                "assemble_ce_model": args.ce_model if normalize_assemble_mode(args.assemble_mode) in ASSEMBLE_CE_ACTIVE_MODES else None,
                "assemble_ce_device": args.ce_device if normalize_assemble_mode(args.assemble_mode) in ASSEMBLE_CE_ACTIVE_MODES else None,
                "structure_max_hops": int(args.setwise_structure_max_hops),
                "base_weight": float(args.setwise_base_weight),
                "structure_weight": float(args.setwise_structure_weight),
                "novelty_weight": float(args.setwise_novelty_weight),
                "non_anchor_title_dedup": bool(args.setwise_non_anchor_title_dedup),
                "query_entity_source": str(args.setwise_query_entity_source),
                "method_EM": round(float(selector_em), 4),
                "method_F1": round(float(selector_f1), 4),
                "baseline_EM": round(float(baseline_em), 4),
                "baseline_F1": round(float(baseline_f1), 4),
                "EM_delta": round(float(selector_em) - float(baseline_em), 4),
                "F1_delta": round(float(selector_f1) - float(baseline_f1), 4),
                "method_retrieval_metrics": selector_retrieval_metrics,
                "per_bucket": {
                    key: {
                        "count": value["count"],
                        "baseline_EM": value["baseline_EM"],
                        "method_EM": value["selector_EM"],
                        "EM_delta": value["EM_delta"],
                        "baseline_F1": value["baseline_F1"],
                        "method_F1": value["selector_F1"],
                        "F1_delta": value["F1_delta"],
                    }
                    for key, value in bucket_summary.items()
                },
                "method_summary": selector_summary,
            }
            expand_assemble_query_traces = build_expand_assemble_query_traces(
                config=config,
                baseline_solutions=query_solutions,
                method_solutions=selected_solutions,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
                doc_text_to_chunk_id=doc_text_to_chunk_id,
            )
            logger.info(
                "Expand-assemble %s[%s/%s]@%d: EM=%.4f (delta=%+.4f), F1=%.4f",
                setwise_selector,
                str(args.setwise_score_mode),
                str(args.assemble_mode),
                int(args.setwise_pool_k),
                float(selector_em),
                float(selector_em) - float(baseline_em),
                float(selector_f1),
            )
        else:
            setwise_selector_results = {
                "selector": setwise_selector,
                "score_mode": str(args.setwise_score_mode),
                "pool_k": int(args.setwise_pool_k),
                "anchor_count": int(args.setwise_anchor_count),
                "reserve_top_m": int(max(int(args.setwise_anchor_count), int(args.setwise_reserve_top_m))),
                "max_bridge_slots": int(max(0, int(args.setwise_max_bridge_slots))),
                "structure_max_hops": int(args.setwise_structure_max_hops),
                "base_weight": float(args.setwise_base_weight),
                "structure_weight": float(args.setwise_structure_weight),
                "novelty_weight": float(args.setwise_novelty_weight),
                "non_anchor_title_dedup": bool(args.setwise_non_anchor_title_dedup),
                "query_entity_source": str(args.setwise_query_entity_source),
                "gate_mode": str(args.setwise_gate_mode),
                "gate_min_structure_score": round(float(args.setwise_gate_min_structure_score), 4),
                "gate_min_combined_margin": round(float(args.setwise_gate_min_combined_margin), 4),
                "gate_min_closure_score": round(float(args.setwise_gate_min_closure_score), 4),
                "gate_min_novelty_score": round(float(args.setwise_gate_min_novelty_score), 4),
                "gate_min_frontier_gain": round(float(args.setwise_gate_min_frontier_gain), 4),
                "gate_min_path_coherence": round(float(args.setwise_gate_min_path_coherence), 4),
                "gate_max_avg_local_structure": round(float(args.setwise_gate_max_avg_local_structure), 4),
                "gate_min_suffix_base_mean": round(float(args.setwise_gate_min_suffix_base_mean), 4),
                "reader_order_probe_mode": str(args.setwise_reader_order_probe_mode),
                "beam_width": int(args.setwise_beam_width),
                "beam_expand_per_state": int(args.setwise_beam_expand_per_state),
                "beam_projected_shortlist_factor": int(args.setwise_beam_projected_shortlist_factor),
                "late_rerank_enabled": bool(args.setwise_late_rerank_enabled),
                "late_rerank_candidate_count": int(args.setwise_late_rerank_candidate_count),
                "late_rerank_include_baseline": bool(args.setwise_late_rerank_include_baseline),
                "late_rerank_doc_char_limit": int(args.setwise_late_rerank_doc_char_limit),
                "late_rerank_policy": str(args.setwise_late_rerank_policy),
                "late_rerank_max_state_score_gap": round(float(args.setwise_late_rerank_max_state_score_gap), 4),
                "late_rerank_judge_backend": late_rerank_judge_bundle.backend,
                "late_rerank_judge_model": late_rerank_judge_bundle.model_name,
                "late_rerank_judge_base_url": late_rerank_judge_bundle.base_url,
                "late_rerank_judge_reasoning_effort": late_rerank_judge_bundle.reasoning_effort,
                "state_score_weights": {k: round(float(v), 4) for k, v in resolve_set_closure_state_weight_config(state_weight_config).items()},
                "setwise_model_path": args.setwise_model_path or None,
                "setwise_requirement_cache_path": args.setwise_requirement_cache_path or None,
                "setwise_requirement_mode": str(args.setwise_requirement_mode),
                "setwise_requirement_model_path": args.setwise_requirement_model_path or None,
                "setwise_requirement_reserve_policy": str(args.setwise_requirement_reserve_policy),
                "setwise_requirement_annotation_pool_k": int(args.setwise_requirement_annotation_pool_k),
                "setwise_requirement_smooth_tau": round(float(args.setwise_requirement_smooth_tau), 4),
                "setwise_requirement_counterfactual_tau": round(float(args.setwise_requirement_counterfactual_tau), 4),
                "setwise_requirement_live_annotation_pool_k": int(args.setwise_requirement_live_annotation_pool_k),
                "setwise_requirement_live_annotation_score_mode": str(args.setwise_requirement_live_annotation_score_mode),
                "setwise_requirement_live_atomic_model_path": args.setwise_requirement_live_atomic_model_path or None,
                "setwise_requirement_live_source_expand_factor": int(args.setwise_requirement_live_source_expand_factor),
                "setwise_requirement_exposure_watch_titles": parse_title_csv(
                    args.setwise_requirement_exposure_watch_titles,
                    default=DEFAULT_REQUIREMENT_EXPOSURE_WATCH_TITLES,
                ),
                "setwise_requirement_probe_force_source_titles": parse_title_csv(args.setwise_requirement_probe_force_source_titles),
                "setwise_requirement_probe_force_shortlist_titles": parse_title_csv(args.setwise_requirement_probe_force_shortlist_titles),
                "setwise_requirement_probe_force_final_titles": parse_title_csv(args.setwise_requirement_probe_force_final_titles),
                "setwise_requirement_probe_force_pool_gold_into_final": bool(args.setwise_requirement_probe_force_pool_gold_into_final),
                "setwise_requirement_probe_source_sort_mode": normalize_requirement_source_sort_mode(args.setwise_requirement_probe_source_sort_mode),
                "setwise_requirement_probe_source_support_gain_weight": round(float(args.setwise_requirement_probe_source_support_gain_weight), 4),
                "setwise_requirement_probe_shortlist_sort_mode": normalize_requirement_shortlist_sort_mode(args.setwise_requirement_probe_shortlist_sort_mode),
                "setwise_requirement_probe_bridge_bonus_mode": normalize_requirement_bridge_bonus_mode(args.setwise_requirement_probe_bridge_bonus_mode),
                "setwise_requirement_probe_bridge_bonus_weight": round(float(args.setwise_requirement_probe_bridge_bonus_weight), 4),
                "selector_EM": round(float(selector_em), 4),
                "selector_F1": round(float(selector_f1), 4),
                "baseline_EM": round(float(baseline_em), 4),
                "baseline_F1": round(float(baseline_f1), 4),
                "EM_delta": round(float(selector_em) - float(baseline_em), 4),
                "F1_delta": round(float(selector_f1) - float(baseline_f1), 4),
                "selector_retrieval_metrics": selector_retrieval_metrics,
                "per_bucket": bucket_summary,
                "selector_summary": selector_summary,
            }
            setwise_selector_query_traces = build_setwise_selector_query_traces(
                config=config,
                baseline_solutions=query_solutions,
                selected_solutions=selected_solutions,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
                doc_text_to_chunk_id=doc_text_to_chunk_id,
            )
            logger.info(
                "Setwise selector %s[%s]@%d: EM=%.4f (delta=%+.4f), F1=%.4f",
                setwise_selector,
                str(args.setwise_score_mode),
                int(args.setwise_pool_k),
                float(selector_em),
                float(selector_em) - float(baseline_em),
                float(selector_f1),
            )

    # Cross-encoder rerank on baseline final top-K
    cross_encoder_rerank_results = None
    cross_encoder_rerank = string_to_bool(args.cross_encoder_rerank) if not gold_doc_reader else False
    if cross_encoder_rerank and not retrieval_only and query_solutions:
        from FlagEmbedding import FlagReranker

        ce_window = int(args.ce_window)
        ce_alpha = float(args.ce_alpha)
        ce_model_name = args.ce_model
        ce_device = args.ce_device

        logger = logging.getLogger(__name__)
        logger.info(f"Loading cross-encoder model: {ce_model_name} on {ce_device}")
        ce_reranker = FlagReranker(ce_model_name, use_fp16=True, device=ce_device)

        reranked_solutions = []
        for q_idx, qs in enumerate(query_solutions):
            window = min(ce_window, len(qs.docs))
            window_docs = qs.docs[:window]
            window_scores = qs.doc_scores[:window] if qs.doc_scores is not None and len(qs.doc_scores) >= window else np.ones(window)
            rest_docs = qs.docs[window:]
            rest_scores = qs.doc_scores[window:] if qs.doc_scores is not None and len(qs.doc_scores) > window else np.array([])

            # Cross-encoder scoring
            pairs = [[qs.question, doc] for doc in window_docs]
            ce_scores = ce_reranker.compute_score(pairs)
            if isinstance(ce_scores, (int, float)):
                ce_scores = [ce_scores]
            ce_scores = np.array(ce_scores, dtype=float)

            # Min-max normalize both score arrays within window
            ppr_arr = np.array(window_scores, dtype=float)
            ppr_range = ppr_arr.max() - ppr_arr.min()
            ppr_norm = (ppr_arr - ppr_arr.min()) / (ppr_range + 1e-9) if ppr_range > 0 else np.ones_like(ppr_arr)
            ce_range = ce_scores.max() - ce_scores.min()
            ce_norm = (ce_scores - ce_scores.min()) / (ce_range + 1e-9) if ce_range > 0 else np.ones_like(ce_scores)

            combined = ce_alpha * ppr_norm + (1 - ce_alpha) * ce_norm
            reorder_idx = np.argsort(-combined)

            reranked_docs = [window_docs[i] for i in reorder_idx] + list(rest_docs)
            reranked_scores = np.concatenate([combined[reorder_idx], rest_scores]) if len(rest_scores) > 0 else combined[reorder_idx]

            reranked_qs = QuerySolution(
                question=qs.question,
                docs=reranked_docs,
                doc_scores=reranked_scores,
                gold_docs=gold_docs[q_idx],
            )
            reranked_solutions.append(reranked_qs)

        # Run QA on reranked docs
        logger.info("Running QA on cross-encoder reranked docs...")
        reranked_solutions, _, _, _, ce_qa_results = hipporag.rag_qa(
            queries=reranked_solutions,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )

        # Compute retrieval metrics on reranked order
        retrieval_recall = RetrievalRecall(global_config=config)
        ce_retrieval_metrics = compute_retrieval_recall_metrics(
            query_solutions=reranked_solutions,
            gold_docs=gold_docs,
        )

        ce_em = ce_qa_results.get("ExactMatch", 0.0)
        ce_f1 = ce_qa_results.get("F1", 0.0)
        baseline_em = overall_qa_results.get("ExactMatch", 0.0) if overall_qa_results else 0.0
        baseline_f1 = overall_qa_results.get("F1", 0.0) if overall_qa_results else 0.0

        # Per-bucket breakdown (2-doc vs 4-doc)
        # Compute per-query EM/F1 via the list API
        qa_em_metric = QAExactMatch(global_config=config)
        qa_f1_metric = QAF1Score(global_config=config)
        bl_answers = [qs.answer or "" for qs in query_solutions]
        ce_answers = [qs.answer or "" for qs in reranked_solutions]
        _, bl_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, bl_answers)
        _, bl_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, bl_answers)
        _, ce_per_query_em = qa_em_metric.calculate_metric_scores(gold_answers, ce_answers)
        _, ce_per_query_f1 = qa_f1_metric.calculate_metric_scores(gold_answers, ce_answers)

        bucket_results = {}
        for q_idx in range(len(queries)):
            n_gold = len(set(gold_docs[q_idx]))
            if n_gold not in bucket_results:
                bucket_results[n_gold] = {"baseline_em": [], "ce_em": [], "baseline_f1": [], "ce_f1": [], "count": 0}
            bucket_results[n_gold]["count"] += 1
            bucket_results[n_gold]["baseline_em"].append(bl_per_query_em[q_idx]["ExactMatch"])
            bucket_results[n_gold]["ce_em"].append(ce_per_query_em[q_idx]["ExactMatch"])
            bucket_results[n_gold]["baseline_f1"].append(bl_per_query_f1[q_idx]["F1"])
            bucket_results[n_gold]["ce_f1"].append(ce_per_query_f1[q_idx]["F1"])

        bucket_summary = {}
        for n_gold, data in sorted(bucket_results.items()):
            bucket_summary[f"{n_gold}_doc"] = {
                "count": data["count"],
                "baseline_EM": round(float(np.mean(data["baseline_em"])), 4),
                "ce_rerank_EM": round(float(np.mean(data["ce_em"])), 4),
                "EM_delta": round(float(np.mean(data["ce_em"])) - float(np.mean(data["baseline_em"])), 4),
                "baseline_F1": round(float(np.mean(data["baseline_f1"])), 4),
                "ce_rerank_F1": round(float(np.mean(data["ce_f1"])), 4),
            }

        cross_encoder_rerank_results = {
            "ce_model": ce_model_name,
            "ce_alpha": ce_alpha,
            "ce_window": ce_window,
            "ce_rerank_EM": round(float(ce_em), 4),
            "ce_rerank_F1": round(float(ce_f1), 4),
            "baseline_EM": round(float(baseline_em), 4),
            "baseline_F1": round(float(baseline_f1), 4),
            "EM_delta": round(float(ce_em) - float(baseline_em), 4),
            "F1_delta": round(float(ce_f1) - float(baseline_f1), 4),
            "ce_retrieval_metrics": ce_retrieval_metrics,
            "per_bucket": bucket_summary,
        }

    slice_metrics = compute_slice_metrics(
        config=config,
        query_solutions=query_solutions,
        gold_docs=gold_docs,
        gold_answers=effective_gold_answers,
    )
    v2_metrics = summarize_v2_metrics(
        config=config,
        hipporag=hipporag,
        query_solutions=query_solutions,
    )
    report_metrics = build_report_metrics_summary(
        slice_metrics["overall"],
        expand_assemble_results=expand_assemble_results,
        setwise_selector_results=setwise_selector_results,
        cross_encoder_rerank_results=cross_encoder_rerank_results,
    )
    result = {
        "dataset": dataset_name,
        "limit": len(samples),
        "llm_name": args.llm_name,
        "llm_request_name": args.llm_request_name,
        "llm_base_url": args.llm_base_url,
        "embedding_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "config": {
            "llm_request_name": config.llm_request_name,
            "causal_enabled": config.causal_enabled,
            "causal_query_only": config.causal_query_only,
            "causal_gate_mode": config.causal_gate_mode,
            "causal_seed_top_k": config.causal_seed_top_k,
            "causal_confidence_threshold": config.causal_confidence_threshold,
            "causal_damping": config.causal_damping,
            "causal_blend_dense_weight": config.causal_blend_dense_weight,
            "causal_blend_fact_weight": config.causal_blend_fact_weight,
            "causal_blend_graph_weight": config.causal_blend_graph_weight,
            "causal_margin_gate_enabled": config.causal_margin_gate_enabled,
            "causal_margin_threshold": config.causal_margin_threshold,
            "causal_blend_top_k": config.causal_blend_top_k,
            "causal_engine_version": config.causal_engine_version,
            "causal_v2_probe_mode": config.causal_v2_probe_mode,
            "causal_v2_graph_mode": config.causal_v2_graph_mode,
            "causal_v2_base_retrieval_mode": config.causal_v2_base_retrieval_mode,
            "general_graph_related_to_weight": config.general_graph_related_to_weight,
            "general_graph_seed_top_k": config.general_graph_seed_top_k,
            "retrieval_only": retrieval_only,
            "gold_doc_reader": gold_doc_reader,
            "oracle_reorder_k": oracle_reorder_k,
            "oracle_select_ks": oracle_select_ks,
            "cross_encoder_rerank": cross_encoder_rerank,
            "setwise_selector": setwise_selector,
            "expand_base_k": int(args.expand_base_k),
            "expand_min_structure_score": float(args.expand_min_structure_score),
            "append_max_docs": int(args.append_max_docs),
            "append_policy": normalize_append_policy(args.append_policy),
            "append_random_seed": int(args.append_random_seed),
            "gap_expand_mode": normalize_gap_expand_mode(args.gap_expand_mode),
            "gap_expand_max_queries": int(max(int(args.gap_expand_max_queries), 1)),
            "assemble_mode": normalize_assemble_mode(args.assemble_mode),
            "baseline_report_json": args.baseline_report_json or None,
            "retrieval_cache_json": args.retrieval_cache_json or None,
            "save_retrieval_cache_json": args.save_retrieval_cache_json or None,
            "setwise_score_mode": str(args.setwise_score_mode),
            "setwise_pool_k": int(args.setwise_pool_k),
            "setwise_anchor_count": int(args.setwise_anchor_count),
            "setwise_structure_max_hops": int(args.setwise_structure_max_hops),
            "setwise_base_weight": float(args.setwise_base_weight),
            "setwise_structure_weight": float(args.setwise_structure_weight),
            "setwise_novelty_weight": float(args.setwise_novelty_weight),
            "setwise_beam_width": int(args.setwise_beam_width),
            "setwise_beam_expand_per_state": int(args.setwise_beam_expand_per_state),
            "setwise_beam_projected_shortlist_factor": int(args.setwise_beam_projected_shortlist_factor),
            "setwise_late_rerank_enabled": bool(args.setwise_late_rerank_enabled),
            "setwise_late_rerank_candidate_count": int(args.setwise_late_rerank_candidate_count),
            "setwise_late_rerank_include_baseline": bool(args.setwise_late_rerank_include_baseline),
            "setwise_late_rerank_doc_char_limit": int(args.setwise_late_rerank_doc_char_limit),
            "setwise_late_rerank_policy": str(args.setwise_late_rerank_policy),
            "setwise_late_rerank_max_state_score_gap": float(args.setwise_late_rerank_max_state_score_gap),
            "setwise_late_rerank_judge_backend": late_rerank_judge_bundle.backend,
            "setwise_late_rerank_judge_model": late_rerank_judge_bundle.model_name,
            "setwise_late_rerank_judge_base_url": late_rerank_judge_bundle.base_url,
            "setwise_late_rerank_judge_reasoning_effort": late_rerank_judge_bundle.reasoning_effort,
            "setwise_reader_order_probe_mode": str(args.setwise_reader_order_probe_mode),
            "setwise_model_path": args.setwise_model_path or None,
            "setwise_requirement_reserve_policy": str(args.setwise_requirement_reserve_policy),
            "setwise_requirement_live_annotation_pool_k": int(args.setwise_requirement_live_annotation_pool_k),
            "setwise_requirement_live_annotation_score_mode": str(args.setwise_requirement_live_annotation_score_mode),
            "setwise_requirement_live_atomic_model_path": args.setwise_requirement_live_atomic_model_path or None,
            "setwise_requirement_live_source_expand_factor": int(args.setwise_requirement_live_source_expand_factor),
            "setwise_requirement_exposure_watch_titles": parse_title_csv(
                args.setwise_requirement_exposure_watch_titles,
                default=DEFAULT_REQUIREMENT_EXPOSURE_WATCH_TITLES,
            ),
            "setwise_requirement_probe_force_source_titles": parse_title_csv(args.setwise_requirement_probe_force_source_titles),
            "setwise_requirement_probe_force_shortlist_titles": parse_title_csv(args.setwise_requirement_probe_force_shortlist_titles),
            "setwise_requirement_probe_force_final_titles": parse_title_csv(args.setwise_requirement_probe_force_final_titles),
            "setwise_requirement_probe_force_pool_gold_into_final": bool(args.setwise_requirement_probe_force_pool_gold_into_final),
            "setwise_requirement_probe_source_sort_mode": normalize_requirement_source_sort_mode(args.setwise_requirement_probe_source_sort_mode),
            "setwise_requirement_probe_source_support_gain_weight": round(float(args.setwise_requirement_probe_source_support_gain_weight), 4),
            "setwise_requirement_probe_shortlist_sort_mode": normalize_requirement_shortlist_sort_mode(args.setwise_requirement_probe_shortlist_sort_mode),
            "setwise_requirement_probe_bridge_bonus_mode": normalize_requirement_bridge_bonus_mode(args.setwise_requirement_probe_bridge_bonus_mode),
            "setwise_requirement_probe_bridge_bonus_weight": round(float(args.setwise_requirement_probe_bridge_bonus_weight), 4),
            "causal_v2_extraction_max_tokens": config.causal_v2_extraction_max_tokens,
            "causal_v2_extraction_retry_attempts": config.causal_v2_extraction_retry_attempts,
            "causal_v2_extraction_workers": config.causal_v2_extraction_workers,
            "causal_event_top_k": config.causal_event_top_k,
            "causal_v2_max_hops": config.causal_v2_max_hops,
            "causal_chain_top_k": config.causal_chain_top_k,
            "causal_context_max_items": config.causal_context_max_items,
            "causal_er_similarity_threshold": config.causal_er_similarity_threshold,
            "causal_er_text_threshold": config.causal_er_text_threshold,
            "causal_v2_min_edge_confidence": config.causal_v2_min_edge_confidence,
            "structure_rerank_enabled": config.structure_rerank_enabled,
            "structure_rerank_top_n": config.structure_rerank_top_n,
            "structure_rerank_bonus_weight": config.structure_rerank_bonus_weight,
            "structure_rerank_min_edge_support": config.structure_rerank_min_edge_support,
            "structure_rerank_max_top5_swaps": config.structure_rerank_max_top5_swaps,
            "structure_rerank_seed_top_k": config.structure_rerank_seed_top_k,
            "structure_rerank_max_hops": config.structure_rerank_max_hops,
            "structure_relation_probe_mode": str(config.structure_relation_probe_mode),
            "structure_continuity_probe_mode": str(config.structure_continuity_probe_mode),
            "structure_seed_target_bridge_mode": str(config.structure_seed_target_bridge_mode),
            "structure_rerank_margin_threshold": config.structure_rerank_margin_threshold,
        },
        "overall_from_pipeline": {
            **(overall_retrieval_result or {}),
            **(overall_qa_results or {}),
        },
        "overall_recomputed": slice_metrics["overall"],
        "report_metrics": report_metrics,
        "causal_slice": slice_metrics["causal_slice"],
        "nonempty_subgraph_slice": slice_metrics["nonempty_subgraph_slice"],
        "v2_metrics": v2_metrics,
        **({"oracle_reorder_qa": oracle_reorder_qa_results} if oracle_reorder_qa_results else {}),
        **({"oracle_select_qa": oracle_select_qa_results} if oracle_select_qa_results else {}),
        **({"setwise_selector_qa": setwise_selector_results} if setwise_selector_results else {}),
        **({"setwise_selector_query_traces": setwise_selector_query_traces} if setwise_selector_query_traces else {}),
        **({"expand_assemble_qa": expand_assemble_results} if expand_assemble_results else {}),
        **({"expand_assemble_query_traces": expand_assemble_query_traces} if expand_assemble_query_traces else {}),
        **({"cross_encoder_rerank_qa": cross_encoder_rerank_results} if cross_encoder_rerank_results else {}),
        "examples": build_report_examples(
            config=config,
            query_solutions=query_solutions,
            doc_text_to_chunk_id=doc_text_to_chunk_id,
            retrieval_only=retrieval_only,
            doc_limit=3,
        ),
    }

    output_json = args.output_json
    if output_json is None:
        report_dir = Path(save_dir) / "eval_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_json = report_dir / f"causal_eval_{dataset_name}_{len(samples)}_{args.llm_name.replace('/', '_')}.json"
    else:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print_result = {
        "output_json": str(output_json),
        "overall_recomputed": result["overall_recomputed"],
        "report_metrics": result["report_metrics"],
        "causal_slice": result["causal_slice"],
        "nonempty_subgraph_slice": result["nonempty_subgraph_slice"],
        "v2_metrics": result["v2_metrics"],
    }
    if oracle_reorder_qa_results:
        print_result["oracle_reorder_qa"] = oracle_reorder_qa_results
    if setwise_selector_results:
        print_result["setwise_selector_qa"] = setwise_selector_results
    if expand_assemble_results:
        print_result["expand_assemble_qa"] = expand_assemble_results
    if cross_encoder_rerank_results:
        print_result["cross_encoder_rerank_qa"] = cross_encoder_rerank_results
    if gold_doc_reader:
        print_result["mode"] = "gold_doc_reader"
    print(json.dumps(print_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
