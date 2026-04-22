import argparse
import ast
from dataclasses import dataclass
import json
import logging
import os
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

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
from dtc_embed_utils import (
    DTCRequirement,
    build_fallback_dtc_requirements,
    parse_dtc_decomposition_response,
    select_dtc_embed_positions,
)
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag_ext.strongest.shadow_entry import run_strongest_shadow_for_pool
from src.hipporag_ext.strongest.types import StrongestConfig
from src.hipporag.utils.causal_utils import (
    expand_directed_entities,
    normalize_structure_text,
    normalize_structure_seed_target_bridge_mode,
    route_query_type,
    score_candidate_docs_by_structure,
)
from src.hipporag.utils.config_utils import BaseConfig
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

SETWISE_LLM_JSON_START_TAG = "<JSON>"
SETWISE_LLM_JSON_END_TAG = "</JSON>"
SETWISE_LLM_LATE_RERANK_MAX_COMPLETION_TOKENS = 256
SETWISE_LLM_LATE_RERANK_REPAIR_MAX_COMPLETION_TOKENS = 96
SETWISE_LLM_NO_THINK_PREFIX = "/no_think"
DEFAULT_CE_BATCH_SIZE = 8
DEFAULT_CE_MAX_LENGTH = 1024


_CROSS_ENCODER_RERANKER_CACHE: Dict[Tuple[str, str, bool, int, int], "TransformersCrossEncoderReranker"] = {}


class TransformersCrossEncoderReranker:
    def __init__(self,
                 model_name_or_path: str,
                 device: str = "cpu",
                 use_fp16: bool = True,
                 batch_size: int = DEFAULT_CE_BATCH_SIZE,
                 max_length: int = DEFAULT_CE_MAX_LENGTH,
                 logger: logging.Logger | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Cross-encoder reranking requires `torch` and `transformers`, but one of them is not installed."
            ) from exc

        self._torch = torch
        self.logger = logger or logging.getLogger(__name__)
        self.model_name_or_path = str(model_name_or_path)
        self.requested_device = str(device or "cpu")
        self.device = self._resolve_device(self.requested_device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(16, int(max_length))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        if bool(use_fp16) and self.device.type == "cuda":
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested_device: str):
        torch = self._torch
        normalized = str(requested_device or "cpu").strip() or "cpu"
        if not normalized.startswith("cuda"):
            return torch.device(normalized)
        if not torch.cuda.is_available():
            self.logger.warning(
                "Cross-encoder requested CUDA device %s, but CUDA is unavailable; falling back to CPU.",
                normalized,
            )
            return torch.device("cpu")
        device_index = 0
        if ":" in normalized:
            _, _, suffix = normalized.partition(":")
            try:
                device_index = int(suffix)
            except ValueError:
                self.logger.warning(
                    "Cross-encoder device %s is invalid; falling back to CPU.",
                    normalized,
                )
                return torch.device("cpu")
        if device_index >= torch.cuda.device_count():
            self.logger.warning(
                "Cross-encoder requested CUDA device %s, but only %d device(s) are visible; falling back to CPU.",
                normalized,
                torch.cuda.device_count(),
            )
            return torch.device("cpu")
        return torch.device(normalized)

    def _extract_scores(self, logits):
        if logits.ndim == 0:
            return logits.reshape(1)
        if logits.ndim == 1:
            return logits
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits[..., -1]

    def compute_score(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        torch = self._torch
        if not pairs:
            return []

        all_scores: List[float] = []
        for start_idx in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start_idx:start_idx + self.batch_size]
            queries = [str(pair[0]) for pair in batch_pairs]
            passages = [str(pair[1]) for pair in batch_pairs]
            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            batch_scores = self._extract_scores(logits).detach().float().cpu().tolist()
            all_scores.extend(float(score) for score in batch_scores)
        return all_scores


def load_cross_encoder_reranker(model_name_or_path: str,
                                device: str,
                                use_fp16: bool = True,
                                batch_size: int = DEFAULT_CE_BATCH_SIZE,
                                max_length: int = DEFAULT_CE_MAX_LENGTH,
                                logger: logging.Logger | None = None) -> TransformersCrossEncoderReranker:
    cache_key = (
        str(model_name_or_path),
        str(device or "cpu"),
        bool(use_fp16),
        int(batch_size),
        int(max_length),
    )
    reranker = _CROSS_ENCODER_RERANKER_CACHE.get(cache_key)
    if reranker is None:
        active_logger = logger or logging.getLogger(__name__)
        active_logger.info(
            "Loading transformers cross-encoder model: %s on %s",
            model_name_or_path,
            device,
        )
        reranker = TransformersCrossEncoderReranker(
            model_name_or_path=model_name_or_path,
            device=device,
            use_fp16=use_fp16,
            batch_size=batch_size,
            max_length=max_length,
            logger=active_logger,
        )
        _CROSS_ENCODER_RERANKER_CACHE[cache_key] = reranker
    return reranker


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


def request_dtc_requirements_from_llm(query: str,
                                      infer_fn,
                                      model_name: str,
                                      max_steps: int = 4,
                                      max_completion_tokens: int = 512) -> Tuple[List[DTCRequirement], Dict[str, object]]:
    trace: Dict[str, object] = {
        "mode": "llm",
        "llm_model": str(model_name or ""),
        "max_steps": int(max_steps),
        "llm_error": None,
        "fallback_used": False,
        "raw_output_preview": "",
    }
    if infer_fn is None:
        requirements = build_fallback_dtc_requirements(query)
        trace.update({
            "llm_error": "infer_fn_unavailable",
            "fallback_used": True,
            "parse_succeeded": False,
            "active_step_count": len(requirements),
        })
        return requirements, trace

    messages = [
        {
            "role": "system",
            "content": (
                "You decompose multi-hop QA questions into evidence requirements for fixed-pool passage selection. "
                "Do not answer the question and do not fill unknown entities from world knowledge. "
                "Return JSON only: an array of 1-4 objects. Each object must have keys: "
                "id, subquery, depends_on, expected_answer_type, anchor_mentions, role. "
                "Use ids like s1, s2. depends_on is a list of previous ids. "
                "A subquery should describe the evidence needed, not the final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{SETWISE_LLM_NO_THINK_PREFIX}\n"
                f"Question: {query}\n\n"
                "Return JSON array only. Example:\n"
                "[{\"id\":\"s1\",\"subquery\":\"Who directed film X?\",\"depends_on\":[],"
                "\"expected_answer_type\":\"person\",\"anchor_mentions\":[\"film X\"],\"role\":\"bridge\"},"
                "{\"id\":\"s2\",\"subquery\":\"What is the birthplace of that director?\","
                "\"depends_on\":[\"s1\"],\"expected_answer_type\":\"location\","
                "\"anchor_mentions\":[],\"role\":\"answer\"}]"
            ),
        },
    ]
    try:
        response = infer_fn(
            messages=messages,
            model=model_name,
            max_completion_tokens=int(max_completion_tokens),
            temperature=0,
            top_p=1,
        )
        raw_output, metadata = normalize_llm_result(response)
    except Exception as exc:
        requirements = build_fallback_dtc_requirements(query)
        trace.update({
            "llm_error": f"{type(exc).__name__}: {exc}",
            "fallback_used": True,
            "parse_succeeded": False,
            "active_step_count": len(requirements),
        })
        return requirements, trace

    requirements, parse_trace = parse_dtc_decomposition_response(
        raw_output,
        max_steps=max_steps,
    )
    if not requirements:
        requirements = build_fallback_dtc_requirements(query)
        trace["fallback_used"] = True
    trace.update(parse_trace)
    trace["raw_output_preview"] = truncate_prompt_text(raw_output, 500)
    trace["metadata"] = metadata
    trace["active_step_count"] = int(len(requirements))
    return requirements, trace


def build_dtc_requirement_embeddings(hipporag: HippoRAG,
                                     requirements: Sequence[DTCRequirement]) -> Dict[str, np.ndarray]:
    subqueries = [req.subquery for req in requirements if str(req.subquery or "").strip()]
    return build_dtc_text_embeddings(hipporag=hipporag, texts=subqueries, keys=[req.unit_id for req in requirements if str(req.subquery or "").strip()])


def build_dtc_text_embeddings(hipporag: HippoRAG,
                              texts: Sequence[str],
                              keys: Sequence[str] | None = None) -> Dict[str, np.ndarray]:
    items: List[Tuple[str, str]] = []
    unique_texts: List[str] = []
    seen_texts: Set[str] = set()
    for idx, text in enumerate(texts):
        clean_text = str(text or "").strip()
        if not clean_text:
            continue
        key = str(keys[idx]) if keys is not None and idx < len(keys) else clean_text
        items.append((key, clean_text))
        if clean_text not in seen_texts:
            unique_texts.append(clean_text)
            seen_texts.add(clean_text)
    if unique_texts and hasattr(hipporag, "_get_passage_query_embeddings"):
        hipporag._get_passage_query_embeddings(unique_texts)
    passage_query_embeddings = (
        ((getattr(hipporag, "query_to_embedding", {}) or {}).get("passage", {}) or {})
        if isinstance(getattr(hipporag, "query_to_embedding", {}), dict)
        else {}
    )
    embeddings: Dict[str, np.ndarray] = {}
    for key, text in items:
        vector = passage_query_embeddings.get(text)
        if vector is not None:
            embeddings[str(key)] = np.asarray(vector, dtype=float)
    return embeddings


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


ASSEMBLE_MODES = {
    "none",
    "base_score",
    "embedding_similarity",
    "cross_encoder",
}

APPEND_POLICIES = {
    "bridge",
    "next_deep",
    "random_deep",
}


def normalize_assemble_mode(mode: str | None) -> str:
    normalized = str(mode or "none").strip().lower()
    if normalized not in ASSEMBLE_MODES:
        raise ValueError(f"Unsupported assemble mode: {mode}")
    return normalized


def normalize_append_policy(policy: str | None) -> str:
    normalized = str(policy or "bridge").strip().lower()
    if normalized not in APPEND_POLICIES:
        raise ValueError(f"Unsupported append policy: {policy}")
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
                                   append_random_seed: int = 0) -> Tuple[List[int], Dict[str, object]]:
    effective_pool_limit = max(int(pool_limit), 0)
    effective_base_k = min(max(int(expand_base_k), 0), effective_pool_limit)
    effective_append_max_docs = max(int(append_max_docs), 0)
    normalized_append_policy = normalize_append_policy(append_policy)
    baseline_prefix_positions = list(range(effective_base_k))
    candidate_positions = list(baseline_prefix_positions)
    appended_positions: List[int] = []
    append_steps: List[Dict[str, object]] = []
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

            ranked_by_structure = sorted(
                scored_candidates,
                key=lambda row: (
                    -float(row.get("structure_score", 0.0) or 0.0),
                    -float(row.get("closure_score", 0.0) or 0.0),
                    -float(row.get("novelty_score", 0.0) or 0.0),
                    int(row.get("pool_position", 0) or 0),
                ),
            )
            best_structure_score = float(ranked_by_structure[0].get("structure_score", 0.0) or 0.0)
            if best_structure_score < float(expand_min_structure_score):
                append_stop_reason = "structure_below_threshold"
                break

            selected_row = None
            duplicate_skip_count = 0
            for row in ranked_by_structure:
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
                "candidate_pool_size": int(len(scored_candidates)),
                "best_structure_score": round(best_structure_score, 4),
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
                    }
                    for rank, row in enumerate(ranked_by_structure[:5])
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
        "expand_min_structure_score": round(float(expand_min_structure_score), 4),
        "score_mode": normalize_setwise_score_mode(score_mode),
        "non_anchor_title_dedup": bool(non_anchor_title_dedup),
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
    normalized_positions = [
        int(pos) for pos in candidate_positions
        if 0 <= int(pos) < len(pool_docs)
    ]
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

    rows.sort(
        key=lambda row: (
            -float(row.get("assemble_score", float("-inf"))),
            -float(row.get("base_score", 0.0) or 0.0),
            int(row.get("pool_position", 0) or 0),
        )
    )
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
                           append_max_docs: int = 3,
                           append_policy: str = "bridge",
                           append_random_seed: int = 0,
                           ce_model: str = "/mnt/nvme/bge-reranker-v2-m3",
                           ce_device: str = "cuda:1",
                           strongest_shadow_enabled: bool = False,
                           strongest_shadow_apply_to_pool: bool = False,
                           strongest_candidate_k: int = 20,
                           strongest_final_k: int = 10,
                           strongest_hippo_head_k: int = 10,
                           strongest_smoothed_union_k: int = 10,
                           strongest_gamma: float = 0.15,
                           strongest_rerank_mode: str = "standard",
                           strongest_gbc_protected_anchor_k: int = 2,
                           strongest_gbc_head_coverage_k: int = 5,
                           strongest_gbc_top_passage_pool_k: int = 24,
                           strongest_gbc_frontier_bonus_k: int = 6,
                           strongest_gbc_bonus_weight: float = 1.0,
                           strongest_ras_enabled: bool = False,
                           strongest_ras_prefix_guard_k: int = 3,
                           strongest_ras_requirement_max_units: int = 4,
                           strongest_ras_enable_conflict_veto: bool = True,
                           strongest_ras_core_support_min_eligible: bool = True,
                           strongest_ras_extractor_mode: str = "rule",
                           strongest_ras_support_mode: str = "lexical",
                           strongest_ras_embedding_probe_threshold: float = 0.35,
                           strongest_ras_trace_enabled: bool = True,
                           dtc_max_steps: int = 4,
                           dtc_match_threshold: float = 0.35,
                           dtc_redundancy_weight: float = 0.10,
                           dtc_base_weight: float = 0.05,
                           dtc_anchor_bonus_weight: float = 0.10,
                           dtc_dependency_bonus_weight: float = 0.10,
                           dtc_max_completion_tokens: int = 512,
                           dtc_decomposition_mode: str = "llm",
                           dtc_enforce_dependencies: bool = True,
                           dtc_require_new_crossing: bool = False,
                           dtc_enable_dependency_binding: bool = False,
                           dtc_binding_max_candidates: int = 4,
                           dtc_binding_entity_hit_required: bool = True) -> Tuple[List[QuerySolution], Dict[str, object]]:
    logger = logging.getLogger(__name__)
    selector_name = str(selector_name).strip().lower()
    score_mode = normalize_setwise_score_mode(score_mode)
    normalized_assemble_mode = normalize_assemble_mode(assemble_mode)
    normalized_append_policy = normalize_append_policy(append_policy)
    if selector_name not in {"bridge_greedy", "bridge_beam", "bridge_append", "learned_greedy", "requirement_beam", "dtc_embed"}:
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
    strongest_shadow_apply_count = 0
    strongest_shadow_success_count = 0
    strongest_shadow_error_count = 0
    strongest_shadow_status_counts: Counter[str] = Counter()
    strongest_ras_extractor_fallback_count = 0
    strongest_ras_empty_requirement_query_count = 0
    strongest_ras_queries_with_core_unit_count = 0
    strongest_ras_queries_with_support_unit_count = 0
    strongest_ras_queries_with_nonzero_g_core_count = 0
    strongest_ras_query_level_repair_count = 0
    strongest_ras_slot_family_counts: Counter[str] = Counter()
    strongest_ras_core_requirement_coverage_at5: List[float] = []
    strongest_ras_support_requirement_coverage_at5: List[float] = []
    strongest_ras_unmet_core_mass_reduction: List[float] = []
    strongest_ras_protected_anchor_retention_at3: List[float] = []
    strongest_ras_protected_anchor_retention_at5: List[float] = []
    strongest_ras_prefix_disruption_count_at3: List[float] = []
    strongest_ras_prefix_disruption_count_at5: List[float] = []
    strongest_ras_single_value_conflict_rate_at5: List[float] = []
    strongest_ras_new_conflict_introduced_rate: List[float] = []
    dtc_parse_success_count = 0
    dtc_fallback_count = 0
    dtc_requirement_counts: List[int] = []
    dtc_covered_requirement_rates: List[float] = []
    dtc_embedding_available_requirement_counts: List[int] = []
    normalized_late_rerank_policy = normalize_setwise_late_rerank_policy(late_rerank_policy)
    normalized_reader_order_probe_mode = normalize_setwise_reader_order_probe_mode(
        setwise_reader_order_probe_mode
    )

    chunk_text_to_hash = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}
    assemble_reranker = None
    if selector_name == "bridge_append" and normalized_assemble_mode == "embedding_similarity":
        if hasattr(hipporag, "_get_passage_query_embeddings"):
            hipporag._get_passage_query_embeddings(query_solutions)
    if selector_name == "bridge_append" and normalized_assemble_mode == "cross_encoder":
        assemble_reranker = load_cross_encoder_reranker(
            model_name_or_path=ce_model,
            device=ce_device,
            use_fp16=True,
            logger=logger,
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
            strongest_shadow_trace: Dict[str, object] = {
                "enabled": bool(strongest_shadow_enabled),
                "applied_to_pool": False,
                "shadow_status": "disabled",
                "candidate_indices": [],
                "final_doc_indices": [],
                "final_titles": [],
                "pool_order_indices": [],
                "source_commit_sha": None,
                "error": None,
            }
            if strongest_shadow_enabled:
                strongest_config = StrongestConfig(
                    candidate_k=int(strongest_candidate_k),
                    final_k=int(strongest_final_k),
                    hippo_head_k=int(strongest_hippo_head_k),
                    smoothed_union_k=int(strongest_smoothed_union_k),
                    gamma=float(strongest_gamma),
                    union_mode="standard",
                    suppression_variant="topology",
                    rerank_mode=str(strongest_rerank_mode),
                    gbc_protected_anchor_k=int(strongest_gbc_protected_anchor_k),
                    gbc_head_coverage_k=int(strongest_gbc_head_coverage_k),
                    gbc_top_passage_pool_k=int(strongest_gbc_top_passage_pool_k),
                    gbc_frontier_bonus_k=int(strongest_gbc_frontier_bonus_k),
                    gbc_bonus_weight=float(strongest_gbc_bonus_weight),
                    ras_enabled=bool(strongest_ras_enabled),
                    ras_prefix_guard_k=int(strongest_ras_prefix_guard_k),
                    ras_requirement_max_units=int(strongest_ras_requirement_max_units),
                    ras_enable_conflict_veto=bool(strongest_ras_enable_conflict_veto),
                    ras_core_support_min_eligible=bool(strongest_ras_core_support_min_eligible),
                    ras_extractor_mode=str(strongest_ras_extractor_mode),
                    ras_support_mode=str(strongest_ras_support_mode),
                    ras_embedding_probe_threshold=float(strongest_ras_embedding_probe_threshold),
                    ras_trace_enabled=bool(strongest_ras_trace_enabled),
                )
                try:
                    strongest_result = run_strongest_shadow_for_pool(
                        hipporag=hipporag,
                        query=qs.question,
                        pool_docs=pool_docs,
                        pool_doc_ids=pool_doc_ids,
                        pool_doc_scores=np.asarray(pool_scores, dtype=float),
                        seed_entities=seed_entities,
                        query_entities=proposal_query_entities,
                        config=strongest_config,
                    )
                except Exception as exc:
                    strongest_result = None
                    strongest_shadow_trace.update({
                        "shadow_status": "error",
                        "source_commit_sha": str(strongest_config.source_commit_sha),
                        "error": str(exc),
                    })
                    strongest_shadow_error_count += 1
                else:
                    if strongest_result is None:
                        strongest_shadow_trace.update({
                            "shadow_status": "unavailable",
                            "source_commit_sha": str(strongest_config.source_commit_sha),
                        })
                    else:
                        final_doc_indices = [
                            int(idx) for idx in np.asarray(
                                strongest_result.final_doc_indices,
                                dtype=np.int64,
                            ).tolist()
                        ]
                        pool_order_indices = [
                            int(idx) for idx in np.asarray(
                                strongest_result.pool_order_indices,
                                dtype=np.int64,
                            ).tolist()
                            if 0 <= int(idx) < pool_limit
                        ]
                        strongest_shadow_trace.update({
                            "shadow_status": str(
                                (strongest_result.trace or {}).get("status", "ok")
                            ),
                            "candidate_indices": [
                                int(idx) for idx in np.asarray(
                                    strongest_result.candidate_indices,
                                    dtype=np.int64,
                                ).tolist()
                            ],
                            "final_doc_indices": final_doc_indices,
                            "final_titles": [
                                pool_titles[idx]
                                for idx in final_doc_indices
                                if 0 <= idx < len(pool_titles)
                            ],
                            "pool_order_indices": pool_order_indices,
                            "source_commit_sha": str(strongest_config.source_commit_sha),
                            "trace": dict(strongest_result.trace or {}),
                        })
                        strongest_shadow_success_count += 1
                        if strongest_config.ras_enabled:
                            ras_trace = dict(((strongest_result.trace or {}).get("gbc", {}) or {}).get("ras", {}) or {})
                            extractor_trace = dict(ras_trace.get("extractor_trace", {}) or {})
                            if bool(extractor_trace.get("extractor_fallback", False)):
                                strongest_ras_extractor_fallback_count += 1
                            if bool(extractor_trace.get("empty_requirement_query", False)):
                                strongest_ras_empty_requirement_query_count += 1
                            if int(extractor_trace.get("core_unit_count", 0) or 0) > 0:
                                strongest_ras_queries_with_core_unit_count += 1
                            if int(extractor_trace.get("support_unit_count", 0) or 0) > 0:
                                strongest_ras_queries_with_support_unit_count += 1
                            if bool(ras_trace.get("queries_with_nonzero_g_core", False)):
                                strongest_ras_queries_with_nonzero_g_core_count += 1
                            if bool(ras_trace.get("query_level_repair", False)):
                                strongest_ras_query_level_repair_count += 1
                            for slot_family in extractor_trace.get("slot_families", []) or []:
                                normalized_slot_family = str(slot_family or "").strip().lower()
                                if normalized_slot_family:
                                    strongest_ras_slot_family_counts[normalized_slot_family] += 1
                            for metric_key, metric_values in (
                                ("core_requirement_coverage_at5", strongest_ras_core_requirement_coverage_at5),
                                ("support_requirement_coverage_at5", strongest_ras_support_requirement_coverage_at5),
                                ("unmet_core_mass_reduction", strongest_ras_unmet_core_mass_reduction),
                                ("protected_anchor_retention_at3", strongest_ras_protected_anchor_retention_at3),
                                ("protected_anchor_retention_at5", strongest_ras_protected_anchor_retention_at5),
                                ("prefix_disruption_count_at3", strongest_ras_prefix_disruption_count_at3),
                                ("prefix_disruption_count_at5", strongest_ras_prefix_disruption_count_at5),
                                ("single_value_conflict_rate_at5", strongest_ras_single_value_conflict_rate_at5),
                                ("new_conflict_introduced_rate", strongest_ras_new_conflict_introduced_rate),
                            ):
                                if metric_key in ras_trace:
                                    metric_values.append(float(ras_trace.get(metric_key, 0.0) or 0.0))
                        if strongest_shadow_apply_to_pool and pool_order_indices:
                            reordered_shadow_positions = list(dict.fromkeys(pool_order_indices))
                            reordered_shadow_positions.extend(
                                pos for pos in range(pool_limit)
                                if pos not in set(reordered_shadow_positions)
                            )
                            pool_docs = [pool_docs[pos] for pos in reordered_shadow_positions]
                            pool_titles = [pool_titles[pos] for pos in reordered_shadow_positions]
                            pool_scores = np.asarray(
                                [pool_scores[pos] for pos in reordered_shadow_positions],
                                dtype=float,
                            )
                            pool_doc_ids = [pool_doc_ids[pos] for pos in reordered_shadow_positions]
                            strongest_shadow_trace["applied_to_pool"] = True
                            strongest_shadow_apply_count += 1
                strongest_shadow_status_counts[
                    str(strongest_shadow_trace.get("shadow_status", "unknown"))
                ] += 1
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
            )
            for pos in selector_trace.get("appended_positions", []) or []:
                position_sources[int(pos)] = f"append_{normalized_append_policy}"
            reranked_positions, assemble_trace = rerank_candidate_positions_for_assemble(
                query=qs.question,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                pool_doc_scores=pool_scores,
                candidate_positions=selected_positions,
                assemble_mode=normalized_assemble_mode,
                hipporag=hipporag,
                ce_reranker=assemble_reranker,
                position_sources=position_sources,
            )
            selector_trace["assemble_trace"] = assemble_trace
            selector_trace["assemble_mode"] = normalized_assemble_mode
            selector_trace["selected_positions_before_assemble"] = list(selected_positions)
            selected_positions = list(reranked_positions)
        elif selector_name == "dtc_embed":
            normalized_dtc_decomposition_mode = str(dtc_decomposition_mode or "llm").strip().lower()
            if normalized_dtc_decomposition_mode == "query":
                requirements = build_fallback_dtc_requirements(qs.question)
                decomposition_trace = {
                    "mode": "query",
                    "llm_model": "",
                    "max_steps": int(dtc_max_steps),
                    "llm_error": None,
                    "fallback_used": False,
                    "parse_succeeded": bool(requirements),
                    "parse_error": None,
                    "raw_output_preview": "",
                    "active_step_count": int(len(requirements)),
                }
            else:
                requirements, decomposition_trace = request_dtc_requirements_from_llm(
                    query=qs.question,
                    infer_fn=getattr(getattr(hipporag, "llm_model", None), "infer", None),
                    model_name=str(
                        getattr(getattr(hipporag, "global_config", None), "llm_request_name", None)
                        or getattr(getattr(hipporag, "global_config", None), "llm_name", "")
                        or ""
                    ),
                    max_steps=int(dtc_max_steps),
                    max_completion_tokens=int(dtc_max_completion_tokens),
                )
            requirement_embeddings = build_dtc_requirement_embeddings(
                hipporag=hipporag,
                requirements=requirements,
            )
            def embed_dtc_bound_texts(texts: Sequence[str]) -> Dict[str, np.ndarray]:
                return build_dtc_text_embeddings(hipporag=hipporag, texts=texts)

            effective_dtc_reserve_top_m = max(int(anchor_count), int(reserve_top_m))
            selected_positions, selector_trace = select_dtc_embed_positions(
                query=qs.question,
                requirements=requirements,
                requirement_embeddings=requirement_embeddings,
                pool_docs=pool_docs,
                pool_doc_ids=pool_doc_ids,
                pool_doc_scores=pool_scores,
                pool_doc_titles=pool_titles,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
                passage_embeddings=np.asarray(getattr(hipporag, "passage_embeddings", np.array([]))),
                qa_top_k=qa_top_k,
                reserve_top_m=effective_dtc_reserve_top_m,
                match_threshold=float(dtc_match_threshold),
                redundancy_weight=float(dtc_redundancy_weight),
                base_weight=float(dtc_base_weight),
                anchor_bonus_weight=float(dtc_anchor_bonus_weight),
                dependency_bonus_weight=float(dtc_dependency_bonus_weight),
                non_anchor_title_dedup=bool(non_anchor_title_dedup),
                enable_dependency_binding=bool(dtc_enable_dependency_binding),
                enforce_dependencies=bool(dtc_enforce_dependencies),
                binding_max_candidates=int(dtc_binding_max_candidates),
                binding_entity_hit_required=bool(dtc_binding_entity_hit_required),
                require_new_crossing=bool(dtc_require_new_crossing),
                embed_texts_fn=embed_dtc_bound_texts if bool(dtc_enable_dependency_binding) else None,
            )
            selector_trace["decomposition_trace"] = decomposition_trace
            selector_trace["dtc_max_steps"] = int(dtc_max_steps)
            selector_trace["dtc_max_completion_tokens"] = int(dtc_max_completion_tokens)
            selector_trace["dtc_decomposition_mode"] = normalized_dtc_decomposition_mode
            selector_trace["dtc_enforce_dependencies"] = bool(dtc_enforce_dependencies)
            selector_trace["dtc_require_new_crossing"] = bool(dtc_require_new_crossing)
            selector_trace["dtc_enable_dependency_binding"] = bool(dtc_enable_dependency_binding)
            selector_trace["dtc_binding_max_candidates"] = int(dtc_binding_max_candidates)
            selector_trace["dtc_binding_entity_hit_required"] = bool(dtc_binding_entity_hit_required)
            dtc_parse_success_count += int(bool(decomposition_trace.get("parse_succeeded", False)))
            dtc_fallback_count += int(bool(decomposition_trace.get("fallback_used", False)))
            dtc_requirement_counts.append(int(selector_trace.get("requirement_count", 0) or 0))
            dtc_covered_requirement_rates.append(float(selector_trace.get("covered_requirement_rate", 0.0) or 0.0))
            dtc_embedding_available_requirement_counts.append(
                int(selector_trace.get("embedding_available_requirement_count", 0) or 0)
            )
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
                "expand_min_structure_score": round(float(expand_min_structure_score), 4),
                "assemble_mode": normalized_assemble_mode,
                "assemble_ce_model": str(ce_model) if normalized_assemble_mode == "cross_encoder" else None,
                "assemble_ce_device": str(ce_device) if normalized_assemble_mode == "cross_encoder" else None,
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
                "strongest_shadow": strongest_shadow_trace,
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
        strongest_ras_query_count = max(int(strongest_shadow_success_count), 1)
        summary.update({
            "expand_base_k": int(max(int(expand_base_k), 0)),
            "append_max_docs": int(max(int(append_max_docs), 0)),
            "append_policy": normalized_append_policy,
            "append_random_seed": int(append_random_seed),
            "expand_min_structure_score": round(float(expand_min_structure_score), 4),
            "assemble_mode": normalized_assemble_mode,
            "assemble_ce_model": str(ce_model) if normalized_assemble_mode == "cross_encoder" else None,
            "assemble_ce_device": str(ce_device) if normalized_assemble_mode == "cross_encoder" else None,
            "strongest_shadow_enabled": bool(strongest_shadow_enabled),
            "strongest_shadow_apply_to_pool": bool(strongest_shadow_apply_to_pool),
            "strongest_candidate_k": int(strongest_candidate_k),
            "strongest_final_k": int(strongest_final_k),
            "strongest_hippo_head_k": int(strongest_hippo_head_k),
            "strongest_smoothed_union_k": int(strongest_smoothed_union_k),
            "strongest_gamma": round(float(strongest_gamma), 4),
            "strongest_rerank_mode": str(strongest_rerank_mode),
            "strongest_gbc_protected_anchor_k": int(strongest_gbc_protected_anchor_k),
            "strongest_gbc_head_coverage_k": int(strongest_gbc_head_coverage_k),
            "strongest_gbc_top_passage_pool_k": int(strongest_gbc_top_passage_pool_k),
            "strongest_gbc_frontier_bonus_k": int(strongest_gbc_frontier_bonus_k),
            "strongest_gbc_bonus_weight": round(float(strongest_gbc_bonus_weight), 4),
            "strongest_ras_enabled": bool(strongest_ras_enabled),
            "strongest_ras_prefix_guard_k": int(strongest_ras_prefix_guard_k),
            "strongest_ras_requirement_max_units": int(strongest_ras_requirement_max_units),
            "strongest_ras_enable_conflict_veto": bool(strongest_ras_enable_conflict_veto),
            "strongest_ras_core_support_min_eligible": bool(strongest_ras_core_support_min_eligible),
            "strongest_ras_extractor_mode": str(strongest_ras_extractor_mode),
            "strongest_ras_support_mode": str(strongest_ras_support_mode),
            "strongest_ras_embedding_probe_threshold": float(strongest_ras_embedding_probe_threshold),
            "strongest_ras_trace_enabled": bool(strongest_ras_trace_enabled),
            "strongest_shadow_success_count": int(strongest_shadow_success_count),
            "strongest_shadow_apply_count": int(strongest_shadow_apply_count),
            "strongest_shadow_error_count": int(strongest_shadow_error_count),
            "strongest_shadow_status_counts": dict(sorted(strongest_shadow_status_counts.items())),
            "strongest_ras_extractor_fallback_count": int(strongest_ras_extractor_fallback_count),
            "strongest_ras_extractor_fallback_rate": round(float(strongest_ras_extractor_fallback_count) / float(strongest_ras_query_count), 4),
            "strongest_ras_empty_requirement_query_rate": round(float(strongest_ras_empty_requirement_query_count) / float(strongest_ras_query_count), 4),
            "strongest_ras_queries_with_core_unit_rate": round(float(strongest_ras_queries_with_core_unit_count) / float(strongest_ras_query_count), 4),
            "strongest_ras_queries_with_support_unit_rate": round(float(strongest_ras_queries_with_support_unit_count) / float(strongest_ras_query_count), 4),
            "strongest_ras_queries_with_nonzero_g_core_rate": round(float(strongest_ras_queries_with_nonzero_g_core_count) / float(strongest_ras_query_count), 4),
            "strongest_ras_query_level_repair_count": int(strongest_ras_query_level_repair_count),
            "strongest_ras_slot_family_distribution": dict(sorted(strongest_ras_slot_family_counts.items())),
            "strongest_ras_core_requirement_coverage_at5": round(float(np.mean(strongest_ras_core_requirement_coverage_at5)) if strongest_ras_core_requirement_coverage_at5 else 0.0, 4),
            "strongest_ras_support_requirement_coverage_at5": round(float(np.mean(strongest_ras_support_requirement_coverage_at5)) if strongest_ras_support_requirement_coverage_at5 else 0.0, 4),
            "strongest_ras_unmet_core_mass_reduction": round(float(np.mean(strongest_ras_unmet_core_mass_reduction)) if strongest_ras_unmet_core_mass_reduction else 0.0, 4),
            "strongest_ras_protected_anchor_retention_at3": round(float(np.mean(strongest_ras_protected_anchor_retention_at3)) if strongest_ras_protected_anchor_retention_at3 else 0.0, 4),
            "strongest_ras_protected_anchor_retention_at5": round(float(np.mean(strongest_ras_protected_anchor_retention_at5)) if strongest_ras_protected_anchor_retention_at5 else 0.0, 4),
            "strongest_ras_prefix_disruption_count_at3": round(float(np.mean(strongest_ras_prefix_disruption_count_at3)) if strongest_ras_prefix_disruption_count_at3 else 0.0, 4),
            "strongest_ras_prefix_disruption_count_at5": round(float(np.mean(strongest_ras_prefix_disruption_count_at5)) if strongest_ras_prefix_disruption_count_at5 else 0.0, 4),
            "strongest_ras_single_value_conflict_rate_at5": round(float(np.mean(strongest_ras_single_value_conflict_rate_at5)) if strongest_ras_single_value_conflict_rate_at5 else 0.0, 4),
            "strongest_ras_new_conflict_introduced_rate": round(float(np.mean(strongest_ras_new_conflict_introduced_rate)) if strongest_ras_new_conflict_introduced_rate else 0.0, 4),
            "avg_appended_doc_count": round(float(np.mean(appended_doc_counts)) if appended_doc_counts else 0.0, 4),
            "avg_candidate_set_size": round(float(np.mean(expand_candidate_sizes)) if expand_candidate_sizes else 0.0, 4),
            "append_count_histogram": {
                str(int(k)): int(v)
                for k, v in sorted(append_count_histogram.items())
            },
            "append_stop_reason_counts": dict(sorted(append_stop_reason_counts.items())),
        })
    if selector_name == "dtc_embed":
        summary.update({
            "dtc_max_steps": int(dtc_max_steps),
            "dtc_match_threshold": round(float(dtc_match_threshold), 4),
            "dtc_redundancy_weight": round(float(dtc_redundancy_weight), 4),
            "dtc_base_weight": round(float(dtc_base_weight), 4),
            "dtc_anchor_bonus_weight": round(float(dtc_anchor_bonus_weight), 4),
            "dtc_dependency_bonus_weight": round(float(dtc_dependency_bonus_weight), 4),
            "dtc_max_completion_tokens": int(dtc_max_completion_tokens),
            "dtc_decomposition_mode": str(dtc_decomposition_mode),
            "dtc_enforce_dependencies": bool(dtc_enforce_dependencies),
            "dtc_require_new_crossing": bool(dtc_require_new_crossing),
            "dtc_enable_dependency_binding": bool(dtc_enable_dependency_binding),
            "dtc_binding_max_candidates": int(dtc_binding_max_candidates),
            "dtc_binding_entity_hit_required": bool(dtc_binding_entity_hit_required),
            "dtc_parse_success_count": int(dtc_parse_success_count),
            "dtc_fallback_count": int(dtc_fallback_count),
            "avg_dtc_requirement_count": round(float(np.mean(dtc_requirement_counts)) if dtc_requirement_counts else 0.0, 4),
            "avg_dtc_covered_requirement_rate": round(float(np.mean(dtc_covered_requirement_rates)) if dtc_covered_requirement_rates else 0.0, 4),
            "avg_dtc_embedding_available_requirement_count": round(
                float(np.mean(dtc_embedding_available_requirement_counts))
                if dtc_embedding_available_requirement_counts else 0.0,
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
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument(
        "--llm_request_name",
        type=str,
        default=None,
        help="Optional API-side model name. Use this to hit a renamed service while reusing llm_name-keyed artifacts.",
    )
    parser.add_argument("--embedding_name", type=str, default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8019/v1/embeddings")
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
                        help="Cross-encoder model path or HF name for a transformers sequence-classification reranker.")
    parser.add_argument("--ce_alpha", type=float, default=0.7,
                        help="Hybrid weight: alpha * ppr_norm + (1-alpha) * ce_norm. 1.0 = pure PPR.")
    parser.add_argument("--ce_window", type=int, default=20,
                        help="Number of top docs to rerank with cross-encoder.")
    parser.add_argument("--ce_device", type=str, default="cuda:1",
                        help="Device for cross-encoder model.")
    parser.add_argument("--setwise_selector", choices=["none", "bridge_greedy", "bridge_beam", "bridge_append", "learned_greedy", "requirement_beam", "dtc_embed"], default="none",
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
    parser.add_argument("--assemble_mode", choices=sorted(ASSEMBLE_MODES), default="cross_encoder",
                        help="For --setwise_selector bridge_append, answer-oriented assembly rerank mode applied over the expanded candidate set.")
    parser.add_argument("--strongest_shadow_enabled", type=string_to_bool, default=False,
                        help="Run strongest sidecar in bridge_append shadow mode and record its intermediate outputs without changing baseline behavior by default.")
    parser.add_argument("--strongest_shadow_apply_to_pool", type=string_to_bool, default=False,
                        help="If true, use strongest shadow pool order as the local candidate order before bridge_append expansion.")
    parser.add_argument("--strongest_candidate_k", type=int, default=20,
                        help="Candidate union size used by strongest shadow sidecar.")
    parser.add_argument("--strongest_final_k", type=int, default=10,
                        help="Final top-k size produced by strongest shadow sidecar.")
    parser.add_argument("--strongest_hippo_head_k", type=int, default=10,
                        help="Hippo head prefix width used by strongest shadow sidecar.")
    parser.add_argument("--strongest_smoothed_union_k", type=int, default=10,
                        help="Smoothed rank prefix width used when forming the strongest shadow candidate union.")
    parser.add_argument("--strongest_gamma", type=float, default=0.15,
                        help="Teleport / reset weight used by strongest shadow local PPR.")
    parser.add_argument("--strongest_rerank_mode", choices=["standard", "gbc"], default="standard",
                        help="Strongest shadow rerank mode. `standard` keeps the migrated baseline; `gbc` adds guarded boundary completion on top of strongest local PPR.")
    parser.add_argument("--strongest_gbc_protected_anchor_k", type=int, default=2,
                        help="When --strongest_rerank_mode=gbc, number of early anchors protected during guarded readout.")
    parser.add_argument("--strongest_gbc_head_coverage_k", type=int, default=5,
                        help="When --strongest_rerank_mode=gbc, fallback head width used to estimate covered support if protected anchors are unavailable.")
    parser.add_argument("--strongest_gbc_top_passage_pool_k", type=int, default=24,
                        help="When --strongest_rerank_mode=gbc, strongest-ranked candidate pool size exposed to boundary completion scoring.")
    parser.add_argument("--strongest_gbc_frontier_bonus_k", type=int, default=6,
                        help="When --strongest_rerank_mode=gbc, maximum number of boundary frontier passages eligible for completion bonus.")
    parser.add_argument("--strongest_gbc_bonus_weight", type=float, default=1.0,
                        help="When --strongest_rerank_mode=gbc, multiplicative weight applied to the deficit-scaled completion bonus.")
    parser.add_argument("--strongest_ras_enabled", type=string_to_bool, default=False,
                        help="Enable requirement-aware completion readout on top of strongest GBC reranking.")
    parser.add_argument("--strongest_ras_prefix_guard_k", type=int, default=3,
                        help="Baseline prefix width protected unless a frontier passage has positive core requirement gain.")
    parser.add_argument("--strongest_ras_requirement_max_units", type=int, default=4,
                        help="Maximum number of rule-extracted requirement units used by strongest RAS readout.")
    parser.add_argument("--strongest_ras_enable_conflict_veto", type=string_to_bool, default=True,
                        help="When strongest RAS is enabled, hard-veto frontier passages that conflict with single-valued head fillers.")
    parser.add_argument("--strongest_ras_core_support_min_eligible", type=string_to_bool, default=True,
                        help="When strongest RAS is enabled, require anchor and slot-family support before awarding requirement gain.")
    parser.add_argument("--strongest_ras_extractor_mode", choices=["rule", "llm", "llm_grounded", "llm_closed_grounded"], default="rule",
                        help="Requirement extractor used by strongest RAS. llm_grounded keeps only locally grounded units and bridge refs with resolved support dependencies; llm_closed_grounded further constrains slot_family to the supported closed ontology.")
    parser.add_argument("--strongest_ras_support_mode", choices=["lexical", "embedding_probe"], default="lexical",
                        help="Support matcher used by strongest RAS. lexical preserves current string/triple matching; embedding_probe adds an opt-in semantic probe on top.")
    parser.add_argument("--strongest_ras_embedding_probe_threshold", type=float, default=0.35,
                        help="Minimum cosine-style similarity required before strongest RAS embedding_probe contributes anchor/slot support.")
    parser.add_argument("--strongest_ras_trace_enabled", type=string_to_bool, default=True,
                        help="Persist strongest RAS diagnostic traces in retrieval summaries.")
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
    parser.add_argument("--dtc_max_steps", type=int, default=4,
                        help="For --setwise_selector dtc_embed, maximum LLM-decomposed evidence requirements per query.")
    parser.add_argument("--dtc_match_threshold", type=float, default=0.35,
                        help="For --setwise_selector dtc_embed, normalized requirement coverage score required to mark a requirement covered.")
    parser.add_argument("--dtc_redundancy_weight", type=float, default=0.10,
                        help="For --setwise_selector dtc_embed, penalty on selecting embedding-redundant documents.")
    parser.add_argument("--dtc_base_weight", type=float, default=0.05,
                        help="For --setwise_selector dtc_embed, small baseline-score tie-break weight.")
    parser.add_argument("--dtc_anchor_bonus_weight", type=float, default=0.10,
                        help="For --setwise_selector dtc_embed, bonus for local anchor mention support.")
    parser.add_argument("--dtc_dependency_bonus_weight", type=float, default=0.10,
                        help="For --setwise_selector dtc_embed, bonus for candidates grounded in already covered dependency evidence.")
    parser.add_argument("--dtc_max_completion_tokens", type=int, default=512,
                        help="For --setwise_selector dtc_embed, max tokens for the one-shot decomposition LLM call.")
    parser.add_argument("--dtc_decomposition_mode", choices=["llm", "query"], default="llm",
                        help="For --setwise_selector dtc_embed, use LLM requirements or a query-only single-requirement ablation.")
    parser.add_argument("--dtc_enforce_dependencies", type=string_to_bool, default=True,
                        help="For --setwise_selector dtc_embed, enforce LLM-declared requirement dependencies before downstream coverage.")
    parser.add_argument("--dtc_require_new_crossing", type=string_to_bool, default=False,
                        help="For --setwise_selector dtc_embed, require a candidate to newly cross at least one requirement threshold before replacing baseline fill.")
    parser.add_argument("--dtc_enable_dependency_binding", type=string_to_bool, default=False,
                        help="For --setwise_selector dtc_embed, bind dependent subqueries to pool titles mentioned by upstream evidence.")
    parser.add_argument("--dtc_binding_max_candidates", type=int, default=4,
                        help="For --setwise_selector dtc_embed, max title candidates used to bind each dependent requirement.")
    parser.add_argument("--dtc_binding_entity_hit_required", type=string_to_bool, default=True,
                        help="For --setwise_selector dtc_embed, require a dependent candidate doc to mention the selected binding title.")
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
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    if save_dir == "outputs":
        save_dir = os.path.join(save_dir, dataset_name)
    else:
        save_dir = f"{save_dir}_{dataset_name}"
    args.save_dir = save_dir

    corpus_path = Path(f"reproduce/dataset/{dataset_name}_corpus.json")
    sample_path = Path(f"reproduce/dataset/{dataset_name}.json")
    corpus = json.load(corpus_path.open())
    samples = json.load(sample_path.open())

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    queries = [sample["question"] for sample in samples]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset_name, corpus=corpus)
    retrieval_only = string_to_bool(args.retrieval_only)
    gold_doc_reader = string_to_bool(args.gold_doc_reader)
    oracle_reorder_k = int(args.oracle_reorder_k)
    oracle_select_ks = [int(x) for x in args.oracle_select_k.split(",") if int(x) > 0]

    config = build_config(args, corpus_len=len(corpus))
    logging.basicConfig(level=logging.INFO)

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
        if retrieval_only:
            query_solutions, overall_retrieval_result = hipporag.retrieve(
                queries=queries,
                gold_docs=gold_docs,
            )
            responses = []
            metadata = []
            overall_qa_results = {}
            effective_gold_answers = None
        else:
            query_solutions, responses, metadata, overall_retrieval_result, overall_qa_results = hipporag.rag_qa(
                queries=queries,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )
            effective_gold_answers = gold_answers

        if bool(args.setwise_late_rerank_enabled) and setwise_selector == "bridge_beam":
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
            append_max_docs=int(args.append_max_docs),
            append_policy=str(args.append_policy),
            append_random_seed=int(args.append_random_seed),
            ce_model=str(args.ce_model),
            ce_device=str(args.ce_device),
            strongest_shadow_enabled=bool(args.strongest_shadow_enabled),
            strongest_shadow_apply_to_pool=bool(args.strongest_shadow_apply_to_pool),
            strongest_candidate_k=int(args.strongest_candidate_k),
            strongest_final_k=int(args.strongest_final_k),
            strongest_hippo_head_k=int(args.strongest_hippo_head_k),
            strongest_smoothed_union_k=int(args.strongest_smoothed_union_k),
            strongest_gamma=float(args.strongest_gamma),
            strongest_rerank_mode=str(args.strongest_rerank_mode),
            strongest_gbc_protected_anchor_k=int(args.strongest_gbc_protected_anchor_k),
            strongest_gbc_head_coverage_k=int(args.strongest_gbc_head_coverage_k),
            strongest_gbc_top_passage_pool_k=int(args.strongest_gbc_top_passage_pool_k),
            strongest_gbc_frontier_bonus_k=int(args.strongest_gbc_frontier_bonus_k),
            strongest_gbc_bonus_weight=float(args.strongest_gbc_bonus_weight),
            strongest_ras_enabled=bool(args.strongest_ras_enabled),
            strongest_ras_prefix_guard_k=int(args.strongest_ras_prefix_guard_k),
            strongest_ras_requirement_max_units=int(args.strongest_ras_requirement_max_units),
            strongest_ras_enable_conflict_veto=bool(args.strongest_ras_enable_conflict_veto),
            strongest_ras_core_support_min_eligible=bool(args.strongest_ras_core_support_min_eligible),
            strongest_ras_extractor_mode=str(args.strongest_ras_extractor_mode),
            strongest_ras_support_mode=str(args.strongest_ras_support_mode),
            strongest_ras_embedding_probe_threshold=float(args.strongest_ras_embedding_probe_threshold),
            strongest_ras_trace_enabled=bool(args.strongest_ras_trace_enabled),
            dtc_max_steps=int(args.dtc_max_steps),
            dtc_match_threshold=float(args.dtc_match_threshold),
            dtc_redundancy_weight=float(args.dtc_redundancy_weight),
            dtc_base_weight=float(args.dtc_base_weight),
            dtc_anchor_bonus_weight=float(args.dtc_anchor_bonus_weight),
            dtc_dependency_bonus_weight=float(args.dtc_dependency_bonus_weight),
            dtc_max_completion_tokens=int(args.dtc_max_completion_tokens),
            dtc_decomposition_mode=str(args.dtc_decomposition_mode),
            dtc_enforce_dependencies=bool(args.dtc_enforce_dependencies),
            dtc_require_new_crossing=bool(args.dtc_require_new_crossing),
            dtc_enable_dependency_binding=bool(args.dtc_enable_dependency_binding),
            dtc_binding_max_candidates=int(args.dtc_binding_max_candidates),
            dtc_binding_entity_hit_required=bool(args.dtc_binding_entity_hit_required),
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

        selector_retrieval_metrics = {}
        for k in [1, 2, 5, 10, 20]:
            recalls = []
            for q_idx, qs in enumerate(selected_solutions):
                gold_set = set(gold_docs[q_idx])
                top_k_set = set(qs.docs[:k])
                recalls.append(len(gold_set & top_k_set) / max(1, len(gold_set)))
            selector_retrieval_metrics[f"Recall@{k}"] = round(float(np.mean(recalls)), 4)

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
                "expand_min_structure_score": round(float(args.expand_min_structure_score), 4),
                "assemble_mode": normalize_assemble_mode(args.assemble_mode),
                "assemble_ce_model": args.ce_model if normalize_assemble_mode(args.assemble_mode) == "cross_encoder" else None,
                "assemble_ce_device": args.ce_device if normalize_assemble_mode(args.assemble_mode) == "cross_encoder" else None,
                "strongest_shadow_enabled": bool(args.strongest_shadow_enabled),
                "strongest_shadow_apply_to_pool": bool(args.strongest_shadow_apply_to_pool),
                "strongest_candidate_k": int(args.strongest_candidate_k),
                "strongest_final_k": int(args.strongest_final_k),
                "strongest_hippo_head_k": int(args.strongest_hippo_head_k),
                "strongest_smoothed_union_k": int(args.strongest_smoothed_union_k),
                "strongest_gamma": round(float(args.strongest_gamma), 4),
                "strongest_rerank_mode": str(args.strongest_rerank_mode),
                "strongest_gbc_protected_anchor_k": int(args.strongest_gbc_protected_anchor_k),
                "strongest_gbc_head_coverage_k": int(args.strongest_gbc_head_coverage_k),
                "strongest_gbc_top_passage_pool_k": int(args.strongest_gbc_top_passage_pool_k),
                "strongest_gbc_frontier_bonus_k": int(args.strongest_gbc_frontier_bonus_k),
                "strongest_gbc_bonus_weight": round(float(args.strongest_gbc_bonus_weight), 4),
                "strongest_ras_enabled": bool(args.strongest_ras_enabled),
                "strongest_ras_prefix_guard_k": int(args.strongest_ras_prefix_guard_k),
                "strongest_ras_requirement_max_units": int(args.strongest_ras_requirement_max_units),
                "strongest_ras_enable_conflict_veto": bool(args.strongest_ras_enable_conflict_veto),
                "strongest_ras_core_support_min_eligible": bool(args.strongest_ras_core_support_min_eligible),
                "strongest_ras_extractor_mode": str(args.strongest_ras_extractor_mode),
                "strongest_ras_support_mode": str(args.strongest_ras_support_mode),
                "strongest_ras_embedding_probe_threshold": float(args.strongest_ras_embedding_probe_threshold),
                "strongest_ras_trace_enabled": bool(args.strongest_ras_trace_enabled),
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
        ce_window = int(args.ce_window)
        ce_alpha = float(args.ce_alpha)
        ce_model_name = args.ce_model
        ce_device = args.ce_device

        logger = logging.getLogger(__name__)
        ce_reranker = load_cross_encoder_reranker(
            model_name_or_path=ce_model_name,
            device=ce_device,
            use_fp16=True,
            logger=logger,
        )

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
        ce_retrieval_metrics = {}
        for k in [1, 2, 5, 10, 20]:
            recalls = []
            for q_idx, qs in enumerate(reranked_solutions):
                gold_set = set(gold_docs[q_idx])
                top_k_set = set(qs.docs[:k])
                recalls.append(len(gold_set & top_k_set) / max(1, len(gold_set)))
            ce_retrieval_metrics[f"Recall@{k}"] = round(float(np.mean(recalls)), 4)

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
            "assemble_mode": normalize_assemble_mode(args.assemble_mode),
            "strongest_shadow_enabled": bool(args.strongest_shadow_enabled),
            "strongest_shadow_apply_to_pool": bool(args.strongest_shadow_apply_to_pool),
            "strongest_candidate_k": int(args.strongest_candidate_k),
            "strongest_final_k": int(args.strongest_final_k),
            "strongest_hippo_head_k": int(args.strongest_hippo_head_k),
            "strongest_smoothed_union_k": int(args.strongest_smoothed_union_k),
            "strongest_gamma": float(args.strongest_gamma),
            "strongest_rerank_mode": str(args.strongest_rerank_mode),
            "strongest_gbc_protected_anchor_k": int(args.strongest_gbc_protected_anchor_k),
            "strongest_gbc_head_coverage_k": int(args.strongest_gbc_head_coverage_k),
            "strongest_gbc_top_passage_pool_k": int(args.strongest_gbc_top_passage_pool_k),
            "strongest_gbc_frontier_bonus_k": int(args.strongest_gbc_frontier_bonus_k),
            "strongest_gbc_bonus_weight": float(args.strongest_gbc_bonus_weight),
            "strongest_ras_enabled": bool(args.strongest_ras_enabled),
            "strongest_ras_prefix_guard_k": int(args.strongest_ras_prefix_guard_k),
            "strongest_ras_requirement_max_units": int(args.strongest_ras_requirement_max_units),
            "strongest_ras_enable_conflict_veto": bool(args.strongest_ras_enable_conflict_veto),
            "strongest_ras_core_support_min_eligible": bool(args.strongest_ras_core_support_min_eligible),
            "strongest_ras_extractor_mode": str(args.strongest_ras_extractor_mode),
            "strongest_ras_support_mode": str(args.strongest_ras_support_mode),
            "strongest_ras_embedding_probe_threshold": float(args.strongest_ras_embedding_probe_threshold),
            "strongest_ras_trace_enabled": bool(args.strongest_ras_trace_enabled),
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
            "dtc_max_steps": int(args.dtc_max_steps),
            "dtc_match_threshold": float(args.dtc_match_threshold),
            "dtc_redundancy_weight": float(args.dtc_redundancy_weight),
            "dtc_base_weight": float(args.dtc_base_weight),
            "dtc_anchor_bonus_weight": float(args.dtc_anchor_bonus_weight),
            "dtc_dependency_bonus_weight": float(args.dtc_dependency_bonus_weight),
            "dtc_max_completion_tokens": int(args.dtc_max_completion_tokens),
            "dtc_decomposition_mode": str(args.dtc_decomposition_mode),
            "dtc_enforce_dependencies": bool(args.dtc_enforce_dependencies),
            "dtc_require_new_crossing": bool(args.dtc_require_new_crossing),
            "dtc_enable_dependency_binding": bool(args.dtc_enable_dependency_binding),
            "dtc_binding_max_candidates": int(args.dtc_binding_max_candidates),
            "dtc_binding_entity_hit_required": bool(args.dtc_binding_entity_hit_required),
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
