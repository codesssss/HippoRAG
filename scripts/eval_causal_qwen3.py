import argparse
import ast
from dataclasses import dataclass
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import joblib
import numpy as np
import pydantic
from openai import OpenAI

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.utils.causal_utils import (
    expand_directed_entities,
    normalize_structure_text,
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


def materialize_reader_top_positions(selected_positions: Sequence[int],
                                     pool_limit: int,
                                     qa_top_k: int) -> List[int]:
    normalized_positions: List[int] = []
    seen_positions: Set[int] = set()
    effective_pool_limit = max(int(pool_limit), 0)
    effective_top_k = min(effective_pool_limit, max(int(qa_top_k), 0))
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
                               reserve_top_m: int) -> Tuple[List[int], List[int]]:
    actual_anchor_count = min(max(anchor_count, 0), target_k, candidate_count)
    actual_reserve_count = min(max(actual_anchor_count, max(reserve_top_m, 0)), target_k, candidate_count)
    anchor_positions = list(range(actual_anchor_count))
    reserved_positions = list(range(actual_reserve_count))
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


def normalize_setwise_score_mode(score_mode: str | None) -> str:
    normalized = str(score_mode or "bridge").strip().lower()
    if normalized not in {"bridge", "closure_proxy", "set_closure"}:
        raise ValueError(f"Unsupported setwise score mode: {score_mode}")
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
                                    structure_max_hops: int = 2) -> Tuple[List[int], Dict[str, object]]:
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
                                 base_weight: float = 0.25,
                                 structure_weight: float = 0.60,
                                 novelty_weight: float = 0.15,
                                 score_mode: str = "bridge",
                                 max_bridge_slots: int = 0,
                                 gate_mode: str = "none",
                                 gate_min_structure_score: float = 0.15,
                                 gate_min_combined_margin: float = 0.0,
                                 non_anchor_title_dedup: bool = False) -> Dict[str, object]:
    normalized_gate_mode = str(gate_mode or "none").strip().lower()
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
        "weakest_baseline_suffix_pool_position": None,
        "weakest_baseline_suffix_structure_score": 0.0,
        "weakest_baseline_suffix_combined_score": 0.0,
        "gate_min_structure_score": round(float(gate_min_structure_score), 4),
        "gate_min_combined_margin": round(float(gate_min_combined_margin), 4),
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
    else:
        weakest_baseline_suffix = None

    if float(best_offrank["structure_score"]) < float(gate_min_structure_score):
        decision["use_selector"] = False
        decision["reason"] = "offrank_structure_below_threshold"
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
                           learned_model_bundle: Dict[str, object] | None = None,
                           beam_width: int = 4,
                           beam_expand_per_state: int = 4,
                           beam_projected_shortlist_factor: int = DEFAULT_SET_CLOSURE_PROJECTED_SHORTLIST_FACTOR,
                           non_anchor_title_dedup: bool = False,
                           query_entity_source: str = "seed",
                           gate_mode: str = "none",
                           gate_min_structure_score: float = 0.15,
                           gate_min_combined_margin: float = 0.0,
                           state_weight_config: Dict[str, float] | None = None,
                           late_rerank_enabled: bool = False,
                           late_rerank_candidate_count: int = 4,
                           late_rerank_include_baseline: bool = True,
                           late_rerank_doc_char_limit: int = 280,
                           late_rerank_policy: str = "always",
                           late_rerank_max_state_score_gap: float = 0.0,
                           late_rerank_judge_bundle: SetwiseLateRerankJudgeBundle | None = None) -> Tuple[List[QuerySolution], Dict[str, object]]:
    logger = logging.getLogger(__name__)
    selector_name = str(selector_name).strip().lower()
    score_mode = normalize_setwise_score_mode(score_mode)
    if selector_name not in {"bridge_greedy", "bridge_beam", "learned_greedy"}:
        raise ValueError(f"Unsupported setwise selector: {selector_name}")

    selected_solutions: List[QuerySolution] = []
    mapped_pool_doc_counts: List[int] = []
    seed_entity_counts: List[int] = []
    selected_structured_doc_counts: List[int] = []
    selector_examples: List[Dict[str, object]] = []
    gate_reason_counts: Counter[str] = Counter()
    gate_apply_count = 0
    gate_skip_count = 0
    late_rerank_apply_count = 0
    late_rerank_override_count = 0
    late_rerank_block_count = 0
    late_rerank_parse_failure_count = 0
    late_rerank_error_count = 0
    beam_projection_eval_count = 0
    beam_projection_extra_eval_count = 0
    beam_projection_rescue_count = 0
    beam_projection_changed_query_count = 0
    beam_projection_max_selected_rank = 0
    normalized_late_rerank_policy = normalize_setwise_late_rerank_policy(late_rerank_policy)

    chunk_text_to_hash = getattr(hipporag.chunk_embedding_store, "text_to_hash_id", {}) or {}

    for qs in query_solutions:
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
                base_weight=base_weight,
                structure_weight=structure_weight,
                novelty_weight=novelty_weight,
                score_mode=score_mode,
                max_bridge_slots=max_bridge_slots,
                gate_mode=gate_mode,
                gate_min_structure_score=gate_min_structure_score,
                gate_min_combined_margin=gate_min_combined_margin,
                non_anchor_title_dedup=non_anchor_title_dedup,
            )
            if gate_decision.get("gate_enabled", False):
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
        else:
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
            )

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
            "late_rerank_trace": late_rerank_trace,
            **selector_trace,
        }

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
                "selection_steps": selector_trace["selection_steps"],
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
        "gate_apply_count": int(gate_apply_count),
        "gate_skip_count": int(gate_skip_count),
        "gate_reason_counts": dict(sorted(gate_reason_counts.items())),
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
    parser.add_argument("--setwise_selector", choices=["none", "bridge_greedy", "bridge_beam", "learned_greedy"], default="none",
                        help="Apply a non-oracle setwise selector over a larger pool before reader top-k truncation.")
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
    parser.add_argument("--setwise_gate_mode", choices=["none", "suffix_bridge"], default="none",
                        help="Per-query activation gate for bridge selectors. suffix_bridge only fires when an off-prefix candidate shows stronger bridge signal than the baseline suffix.")
    parser.add_argument("--setwise_gate_min_structure_score", type=float, default=0.15,
                        help="Minimum structure score required for the adaptive setwise gate to activate on an off-prefix bridge candidate.")
    parser.add_argument("--setwise_gate_min_combined_margin", type=float, default=0.0,
                        help="Minimum combined-score advantage an off-prefix bridge candidate must have over the weakest baseline suffix doc before the gate activates.")
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
    late_rerank_judge_bundle = SetwiseLateRerankJudgeBundle(
        infer_fn=None,
        model_name=str(args.llm_name),
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
                fallback_model_name=hipporag.global_config.llm_name,
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
            state_weight_config=state_weight_config,
            late_rerank_enabled=bool(args.setwise_late_rerank_enabled),
            late_rerank_candidate_count=int(args.setwise_late_rerank_candidate_count),
            late_rerank_include_baseline=bool(args.setwise_late_rerank_include_baseline),
            late_rerank_doc_char_limit=int(args.setwise_late_rerank_doc_char_limit),
            late_rerank_policy=str(args.setwise_late_rerank_policy),
            late_rerank_max_state_score_gap=float(args.setwise_late_rerank_max_state_score_gap),
            late_rerank_judge_bundle=late_rerank_judge_bundle,
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
        "llm_base_url": args.llm_base_url,
        "embedding_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "config": {
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
            "setwise_model_path": args.setwise_model_path or None,
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
        **({"cross_encoder_rerank_qa": cross_encoder_rerank_results} if cross_encoder_rerank_results else {}),
        "examples": [
            {
                "question": query_solution.question,
                "query_type": (
                    (query_solution.retrieval_trace or {}).get("router_label")
                    if getattr(config, "causal_engine_version", "legacy") == "v2"
                    else route_query_type(query_solution.question)
                ),
                "answer": query_solution.answer if not retrieval_only else None,
                "gold_answers": query_solution.gold_answers if not retrieval_only else None,
                "docs": query_solution.docs[:3],
                "retrieved_doc_ids": serialize_retrieved_doc_ids(query_solution.docs, doc_text_to_chunk_id),
                "retrieval_trace": query_solution.retrieval_trace or {},
            }
            for query_solution in query_solutions
        ],
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
    if cross_encoder_rerank_results:
        print_result["cross_encoder_rerank_qa"] = cross_encoder_rerank_results
    if gold_doc_reader:
        print_result["mode"] = "gold_doc_reader"
    print(json.dumps(print_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
