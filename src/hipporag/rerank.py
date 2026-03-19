import json
import hashlib
import os
from pydantic import BaseModel, Field
from copy import deepcopy
from typing import Optional, List, Any, Tuple
import re
import ast
from .prompts.filter_default_prompt import best_dspy_prompt
from .utils.llm_utils import fix_broken_generated_json
from .utils.logging_utils import get_logger
from .utils.misc_utils import normalize_triple

logger = get_logger(__name__)
JSON_START_TAG = "<JSON>"
JSON_END_TAG = "</JSON>"
NO_THINK_PREFIX = "/no_think"
MAX_RAW_OUTPUT_PREVIEW_CHARS = 500
MAX_ATTEMPT_SUMMARY_PREVIEW_CHARS = 240
PRIMARY_MAX_COMPLETION_TOKENS = 1024
REPAIR_MAX_COMPLETION_TOKENS = 48

class CandidateSelection(BaseModel):
    best_ids: list[Any] = Field(description="0-based candidate ids selected from the provided snapshot.")
    confidence: Optional[float] = Field(default=None, description="Optional confidence score in [0, 1].")


def _normalize_fact_item(value: Any) -> Optional[Tuple[str, str, str]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    normalized = []
    for item in value:
        if isinstance(item, (str, int, float, bool)) or item is None:
            normalized.append(str(item))
        else:
            return None
    return tuple(normalized)


def _clean_structured_payload(value: str) -> str:
    tag_start = value.find(JSON_START_TAG)
    if tag_start != -1:
        tag_end = value.find(JSON_END_TAG, tag_start + len(JSON_START_TAG))
        tagged_payload = (
            value[tag_start + len(JSON_START_TAG):tag_end]
            if tag_end != -1 else
            value[tag_start + len(JSON_START_TAG):]
        ).strip()
        if tagged_payload:
            return tagged_payload

    cleaned_lines = []
    for line in value.splitlines():
        stripped = line.strip()
        if stripped.startswith("[[ ##"):
            continue
        if stripped.startswith("```"):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        return cleaned[object_start:object_end + 1]

    list_start = cleaned.find("[")
    list_end = cleaned.rfind("]")
    if list_start != -1 and list_end != -1 and list_end > list_start:
        return cleaned[list_start:list_end + 1]

    return cleaned


def _extract_selection_payload_from_text(value: str) -> Optional[dict[str, Any]]:
    best_ids_match = re.search(r'["\']?best_ids["\']?\s*[:=]\s*\[([^\]]*)\]', value, flags=re.IGNORECASE | re.DOTALL)
    if not best_ids_match:
        return None

    best_ids_blob = best_ids_match.group(1)
    raw_items = [item.strip() for item in best_ids_blob.split(",") if item.strip()]
    best_ids: list[Any] = []
    for item in raw_items:
        normalized = item.strip().strip('"').strip("'")
        if re.fullmatch(r"-?\d+", normalized):
            best_ids.append(int(normalized))
        elif normalized:
            best_ids.append(normalized)

    payload: dict[str, Any] = {"best_ids": best_ids}
    confidence_match = re.search(
        r'["\']?confidence["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)',
        value,
        flags=re.IGNORECASE,
    )
    if confidence_match:
        try:
            payload["confidence"] = float(confidence_match.group(1))
        except ValueError:
            pass
    return payload


def _parse_structured_payload(value: Any) -> Any:
    parsed_value = value
    if isinstance(parsed_value, str):
        cleaned_value = _clean_structured_payload(parsed_value)
        candidate_payloads = [cleaned_value]
        fixed_payload = fix_broken_generated_json(cleaned_value)
        if fixed_payload != cleaned_value:
            candidate_payloads.append(fixed_payload)

        last_error = None
        for payload in candidate_payloads:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed_value = parser(payload)
                    break
                except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
                    last_error = exc
                    parsed_value = None
            if parsed_value is not None:
                break

        if parsed_value is None:
            extracted_payload = _extract_selection_payload_from_text(cleaned_value)
            if extracted_payload is not None:
                return extracted_payload
            raise ValueError(f"Unable to parse structured payload: {last_error}")

    return parsed_value


def _coerce_selection_payload(value: Any, num_candidates: int) -> tuple[list[int], list[Any], list[Any], Optional[float]]:
    parsed_value = _parse_structured_payload(value)

    if isinstance(parsed_value, list):
        parsed_value = {"best_ids": parsed_value}

    validated = CandidateSelection.model_validate(parsed_value)
    parsed_best_ids = list(validated.best_ids)

    valid_best_ids: list[int] = []
    invalid_best_ids: list[Any] = []
    for raw_id in parsed_best_ids:
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            invalid_best_ids.append(raw_id)
            continue
        if raw_id < 0 or raw_id >= num_candidates:
            invalid_best_ids.append(raw_id)
            continue
        valid_best_ids.append(raw_id)

    confidence = None
    if isinstance(validated.confidence, (int, float)):
        confidence = float(validated.confidence)

    return valid_best_ids, parsed_best_ids, invalid_best_ids, confidence


def _attempt_status(parse_succeeded: bool,
                    selected_ids: list[int],
                    parsed_best_ids: list[Any],
                    invalid_best_ids: list[Any],
                    parse_error: Optional[str]) -> str:
    if parse_succeeded and invalid_best_ids:
        return "schema_failure" if invalid_best_ids else "empty_output"
    if parse_succeeded and selected_ids:
        return "ok"
    if parse_succeeded and parsed_best_ids:
        return "empty_output"
    if parse_error == "llm_exception":
        return "llm_exception"
    return "parse_failure"


class DSPyFilter:
    def __init__(self, hipporag):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
        hipporag : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = hipporag.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = hipporag.llm_model.infer
        self.model_name = hipporag.global_config.llm_name
        self.require_non_empty = bool(getattr(hipporag.global_config, "rerank_require_non_empty", True))
        self.force_no_think = os.getenv("HIPPORAG_RERANK_FORCE_NO_THINK", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.default_gen_kwargs = {}
        self.max_parse_retries = 2
        self.fallback_top_k = 2
        self.max_repair_retries = 1

    def _make_rerank_log(self, num_candidates: int) -> dict:
        return {
            "confidence": None,
            "used_fallback": False,
            "attempts": [],
            "num_candidates": num_candidates,
            "n_candidates_initial": num_candidates,
            "n_facts_after_rerank_raw": 0,
            "n_facts_after_mapping": 0,
            "n_facts_after_postprocess": 0,
            "n_facts_final": 0,
            "facts_empty_stage": None,
            "final_non_empty_fallback_applied": False,
            "final_non_empty_fallback_reason": None,
            "mapping_issue_detected": False,
            "mapping_exact_match_count": 0,
            "mapping_normalized_match_count": 0,
            "mapping_unmapped_count": 0,
            "mapping_duplicate_drop_count": 0,
            "mapping_partial_recovery_applied": False,
            "mapping_fallback_fill_count": 0,
            "mapping_issue_reason_counts": {},
            "parse_failure_count": 0,
            "schema_failure_count": 0,
            "empty_output_count": 0,
            "model_semantic_empty_count": 0,
            "repair_failure_count": 0,
            "non_empty_fallback_count": 0,
            "final_facts_empty_count": 0,
            "final_failure_reason": None,
            "candidate_id_list": [],
            "candidate_keys_preview": [],
            "valid_id_range": None,
            "parsed_best_ids": [],
            "invalid_best_ids": [],
            "raw_output_preview": None,
            "attempt1_status": None,
            "attempt1_error": None,
            "attempt1_raw_output_preview": None,
            "attempt1_finish_reason": None,
            "attempt1_truncated": False,
            "attempt2_status": None,
            "attempt2_error": None,
            "attempt2_raw_output_preview": None,
            "attempt2_finish_reason": None,
            "attempt2_truncated": False,
            "truncated_response_count": 0,
        }

    @staticmethod
    def _record_primary_attempt_trace(rerank_log: dict,
                                      attempt_number: int,
                                      *,
                                      status: str,
                                      parse_error: Optional[str],
                                      raw_output_preview: Optional[str],
                                      finish_reason: Optional[str],
                                      truncated: bool) -> None:
        if attempt_number not in (1, 2):
            return
        rerank_log[f"attempt{attempt_number}_status"] = status
        rerank_log[f"attempt{attempt_number}_error"] = parse_error
        rerank_log[f"attempt{attempt_number}_raw_output_preview"] = raw_output_preview
        rerank_log[f"attempt{attempt_number}_finish_reason"] = finish_reason
        rerank_log[f"attempt{attempt_number}_truncated"] = bool(truncated)

    @staticmethod
    def _stage_from_reason(reason: str) -> str:
        return {
            "empty_output": "reranker_semantic_empty",
            "parse_failure": "reranker_parse_failure",
            "schema_failure": "reranker_schema_failure",
            "mapping_failure": "mapping_failure_drop",
            "postprocess_filter_drop": "postprocess_filter_drop",
            "candidates_empty": "candidates_empty",
            "rerank_exception": "rerank_exception",
        }.get(reason, reason)

    def _build_failure_result(self,
                              candidate_items: List[Tuple],
                              candidate_indices: List[int],
                              len_after_rerank: Optional[int],
                              rerank_log: dict,
                              reason: str) -> Tuple[List[int], List[Tuple], dict]:
        rerank_log["fallback_reason"] = reason
        rerank_log["final_failure_reason"] = reason
        rerank_log["facts_empty_stage"] = self._stage_from_reason(reason)

        if not self.require_non_empty:
            rerank_log["used_fallback"] = False
            rerank_log["n_facts_final"] = 0
            rerank_log["final_facts_empty_count"] += 1
            return [], [], rerank_log

        fallback_k = len_after_rerank or min(len(candidate_items), self.fallback_top_k)
        fallback_indices = candidate_indices[:fallback_k]
        fallback_items = candidate_items[:fallback_k]
        rerank_log["used_fallback"] = True
        rerank_log["n_facts_final"] = len(fallback_items)
        if fallback_items:
            rerank_log["final_non_empty_fallback_applied"] = True
            rerank_log["final_non_empty_fallback_reason"] = reason
            rerank_log["non_empty_fallback_count"] += 1
        else:
            rerank_log["final_facts_empty_count"] += 1
        return fallback_indices, fallback_items, rerank_log

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"])})
        return message_template

    def parse_filter(self,
                     response: str,
                     num_candidates: int) -> Tuple[List[int], bool, Optional[str], List[Any], List[Any], Optional[float]]:
        try:
            best_ids, parsed_best_ids, invalid_best_ids, confidence = _coerce_selection_payload(
                response,
                num_candidates=num_candidates,
            )
            return best_ids, True, None, parsed_best_ids, invalid_best_ids, confidence
        except Exception as exc:
            return [], False, f"Error parsing raw response as candidate selection: {exc}.", [], [], None

    def _build_candidate_snapshot(self, candidate_items: List[Tuple]) -> dict:
        candidates = []
        for idx, candidate_item in enumerate(candidate_items):
            normalized_item = _normalize_fact_item(candidate_item)
            candidates.append({
                "id": idx,
                "key": " | ".join(normalized_item or tuple(str(part) for part in candidate_item)),
            })
        return {"candidates": candidates}

    @staticmethod
    def _sanitize_candidate_key_for_prompt(value: str, max_len: int = 160) -> str:
        sanitized = value.replace("\n", " ").replace("\r", " ")
        sanitized = sanitized.replace('"', "").replace("'", "").replace("`", "")
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        if len(sanitized) <= max_len:
            return sanitized
        return sanitized[: max_len - 1] + "…"

    def _format_candidate_snapshot(self, candidate_snapshot: dict) -> str:
        lines = ["Candidate ids and facts:"]
        for candidate in candidate_snapshot.get("candidates", []):
            key = self._sanitize_candidate_key_for_prompt(str(candidate.get("key", "")))
            lines.append(f"{candidate.get('id')}: {key}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_llm_result(response: Any) -> tuple[str, dict]:
        if isinstance(response, (tuple, list)) and len(response) >= 2:
            response_text = response[0]
            metadata = response[1]
            if isinstance(metadata, dict):
                return str(response_text), dict(metadata or {})
        return str(response), {}

    @staticmethod
    def _is_truncated_finish_reason(metadata: dict) -> bool:
        finish_reason = metadata.get("finish_reason")
        return finish_reason in {"length", "max_tokens", "max_completion_tokens"}

    def llm_call(self, question, candidate_snapshot_json, allow_think: bool = True):
        if self.require_non_empty:
            instruction = (
                "Candidate ids are non-empty. Return at least one best id; if uncertain, return the single best id."
            )
        else:
            instruction = "Return the best candidate ids only. If none are relevant, return an empty best_ids list."

        messages = [
            {
                "role": "system",
                "content": (
                    "You filter candidate facts for retrieval. "
                    f"Output exactly one single-line JSON object wrapped in {JSON_START_TAG} and {JSON_END_TAG}. "
                    "The JSON object may contain keys \"best_ids\" and optional \"confidence\" only. "
                    "\"best_ids\" must be a list of 0-based integer ids from the provided candidate snapshot. "
                    "Never output candidate text, triples, explanations, markdown, headers, code fences, or extra keys."
                ),
            },
            {
                "role": "user",
                "content": (
                    (f"{NO_THINK_PREFIX}\n" if not allow_think else "")
                    +
                    f"Question: {question}\n"
                    f"{candidate_snapshot_json}\n"
                    f"{instruction}\n"
                    "Use ids only from the candidate snapshot. Do not quote or copy candidate text.\n"
                    "Return exactly one line in this shape:\n"
                    f"{JSON_START_TAG}{{\"best_ids\":[0,3],\"confidence\":0.72}}{JSON_END_TAG}"
                ),
            },
        ]

        gen_kwargs = dict(self.default_gen_kwargs)
        gen_kwargs["max_completion_tokens"] = (
            PRIMARY_MAX_COMPLETION_TOKENS if allow_think else REPAIR_MAX_COMPLETION_TOKENS
        )
        gen_kwargs["temperature"] = 0
        gen_kwargs["top_p"] = 1
        gen_kwargs["stop"] = [JSON_END_TAG]

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            response_format=None,
            **gen_kwargs
        )
        return self._normalize_llm_result(response)

    def repair_llm_call(self,
                        question: str,
                        candidate_snapshot_json: str,
                        raw_response: str,
                        empty_output: bool = False,
                        parse_error: Optional[str] = None):
        repair_reason = (
            "The previous output was valid JSON but returned an empty best_ids list."
            if empty_output else
            "The previous output was not valid for the required JSON schema."
        )
        if self.require_non_empty:
            non_empty_requirement = (
                "Candidate ids are non-empty, so you must return at least one best id from the provided 0-based candidate ids. "
                "If you are uncertain, return the single best candidate id instead of an empty list."
                if empty_output else
                "Return a valid best_ids list using only 0-based candidate ids."
            )
        else:
            non_empty_requirement = (
                "Return a valid best_ids list using only 0-based candidate ids. If none are relevant, return an empty best_ids list."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You repair reranker outputs. "
                    f"Output exactly one single-line JSON object wrapped in {JSON_START_TAG} and {JSON_END_TAG}. "
                    "The JSON object may contain keys \"best_ids\" and optional \"confidence\" only. "
                    "\"best_ids\" must be a list of 0-based integer ids from the provided candidate snapshot. "
                    "Never output candidate text, triples, explanations, markdown, headers, code fences, or extra keys."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{NO_THINK_PREFIX}\n"
                    f"Question: {question}\n"
                    f"{candidate_snapshot_json}\n"
                    f"Previous output preview: {raw_response[:240]}\n"
                    f"Issue: {repair_reason}\n"
                    f"Previous parser error: {parse_error or 'unknown'}\n"
                    f"{non_empty_requirement} "
                    "If the previous output copied candidate text or triple fragments, replace them with ids only. "
                    "Use ids in the inclusive range [0, N-1]. "
                    "Return exactly one line in this shape:\n"
                    f"{JSON_START_TAG}{{\"best_ids\":[0,3],\"confidence\":0.72}}{JSON_END_TAG}"
                ),
            },
        ]
        gen_kwargs = dict(self.default_gen_kwargs)
        gen_kwargs["max_completion_tokens"] = REPAIR_MAX_COMPLETION_TOKENS
        gen_kwargs["temperature"] = 0
        gen_kwargs["top_p"] = 1
        gen_kwargs["stop"] = [JSON_END_TAG]
        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            response_format=None,
            **gen_kwargs,
        )
        return self._normalize_llm_result(response)

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[Tuple],
               candidate_indices: List[int],
               len_after_rerank: int =None) -> Tuple[List[int], List[Tuple], dict]:
        candidate_snapshot = self._build_candidate_snapshot(candidate_items)
        candidate_snapshot_json = self._format_candidate_snapshot(candidate_snapshot)
        selected_candidate_ids: List[int] = []
        rerank_log = self._make_rerank_log(len(candidate_items))
        parse_succeeded = False
        saw_empty_output = False
        saw_schema_failure = False
        candidate_snapshot_hash = hashlib.sha256(
            json.dumps(candidate_snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        rerank_log["candidate_snapshot_hash"] = candidate_snapshot_hash
        rerank_log["candidate_id_list"] = list(range(len(candidate_items)))
        rerank_log["candidate_keys_preview"] = [entry["key"] for entry in candidate_snapshot["candidates"][:5]]
        rerank_log["valid_id_range"] = [0, max(len(candidate_items) - 1, 0)] if candidate_items else []

        for attempt_idx in range(self.max_parse_retries + 1):
            allow_think = (attempt_idx == 0) and not self.force_no_think
            try:
                response_text, response_metadata = self.llm_call(
                    query,
                    candidate_snapshot_json,
                    allow_think=allow_think,
                )
                finish_reason = response_metadata.get("finish_reason")
                truncated = self._is_truncated_finish_reason(response_metadata)
                if truncated:
                    rerank_log["truncated_response_count"] += 1
                rerank_log["raw_output_preview"] = response_text[:MAX_RAW_OUTPUT_PREVIEW_CHARS]
                (
                    selected_candidate_ids,
                    parse_succeeded,
                    parse_error,
                    parsed_best_ids,
                    invalid_best_ids,
                    confidence,
                ) = self.parse_filter(response_text, num_candidates=len(candidate_items))
                rerank_log["parsed_best_ids"] = list(parsed_best_ids)
                rerank_log["invalid_best_ids"] = list(invalid_best_ids)
                if confidence is not None:
                    rerank_log["confidence"] = confidence
                if parse_succeeded and invalid_best_ids:
                    saw_schema_failure = True
                    rerank_log["schema_failure_count"] += 1
                    rerank_log["mapping_issue_detected"] = True
                    rerank_log["mapping_issue_reason_counts"]["invalid_best_id"] = (
                        rerank_log["mapping_issue_reason_counts"].get("invalid_best_id", 0) + len(invalid_best_ids)
                    )
                rerank_log["attempts"].append({
                    "attempt": attempt_idx + 1,
                    "phase": "primary",
                    "allow_think": allow_think,
                    "parse_succeeded": parse_succeeded,
                    "num_generated_facts": len(selected_candidate_ids),
                    "parse_error": parse_error,
                    "parsed_best_ids": list(parsed_best_ids),
                    "invalid_best_ids": list(invalid_best_ids),
                    "finish_reason": finish_reason,
                    "truncated": truncated,
                    "raw_output_preview": response_text[:MAX_ATTEMPT_SUMMARY_PREVIEW_CHARS],
                })
                if parse_succeeded and selected_candidate_ids:
                    rerank_log["n_facts_after_rerank_raw"] = len(selected_candidate_ids)
                    primary_attempt_status = _attempt_status(
                        parse_succeeded=parse_succeeded,
                        selected_ids=selected_candidate_ids,
                        parsed_best_ids=parsed_best_ids,
                        invalid_best_ids=invalid_best_ids,
                        parse_error=parse_error,
                    )
                    self._record_primary_attempt_trace(
                        rerank_log,
                        attempt_idx + 1,
                        status=primary_attempt_status,
                        parse_error=parse_error,
                        raw_output_preview=response_text[:MAX_RAW_OUTPUT_PREVIEW_CHARS],
                        finish_reason=finish_reason,
                        truncated=truncated,
                    )
                    break
                if parse_succeeded and not selected_candidate_ids:
                    if parsed_best_ids and invalid_best_ids:
                        saw_schema_failure = True
                        parse_succeeded = False
                        parse_error = (
                            f"Parsed JSON but best_ids contained no valid 0-based ids in range "
                            f"{rerank_log['valid_id_range']}. Invalid ids: {invalid_best_ids[:10]}"
                        )
                        rerank_log["schema_failure_count"] += 1
                        logger.warning(
                            "Rerank schema failure on attempt %s for query %r: %s",
                            attempt_idx + 1,
                            query[:160],
                            parse_error,
                        )
                    elif invalid_best_ids:
                        saw_schema_failure = True
                        parse_succeeded = False
                        parse_error = (
                            f"Parsed JSON but best_ids contained invalid ids {invalid_best_ids[:10]} "
                            f"for valid range {rerank_log['valid_id_range']}."
                        )
                        rerank_log["schema_failure_count"] += 1
                        logger.warning(
                            "Rerank schema failure on attempt %s for query %r: %s",
                            attempt_idx + 1,
                            query[:160],
                            parse_error,
                        )
                    else:
                        if not self.require_non_empty:
                            rerank_log["empty_output_count"] += 1
                            rerank_log["model_semantic_empty_count"] += 1
                            rerank_log["final_failure_reason"] = "empty_output"
                            rerank_log["facts_empty_stage"] = self._stage_from_reason("empty_output")
                            rerank_log["n_facts_final"] = 0
                            rerank_log["final_facts_empty_count"] += 1
                            return [], [], rerank_log
                        saw_empty_output = True
                        parse_succeeded = False
                        parse_error = "Parsed successfully but returned an empty best_ids list."
                        logger.warning(
                            "Rerank returned empty ids on attempt %s for query %r.",
                            attempt_idx + 1,
                            query[:160],
                        )
                        rerank_log["empty_output_count"] += 1
                        rerank_log["model_semantic_empty_count"] += 1
                else:
                    logger.warning(
                        "Rerank parse failed on attempt %s for query %r: %s",
                        attempt_idx + 1,
                        query[:160],
                        parse_error,
                    )
                    rerank_log["parse_failure_count"] += 1
                primary_attempt_status = _attempt_status(
                    parse_succeeded=parse_succeeded,
                    selected_ids=selected_candidate_ids,
                    parsed_best_ids=parsed_best_ids,
                    invalid_best_ids=invalid_best_ids,
                    parse_error=parse_error,
                )
                self._record_primary_attempt_trace(
                    rerank_log,
                    attempt_idx + 1,
                    status=primary_attempt_status,
                    parse_error=parse_error,
                    raw_output_preview=response_text[:MAX_RAW_OUTPUT_PREVIEW_CHARS],
                    finish_reason=finish_reason,
                    truncated=truncated,
                )
                if parse_succeeded and selected_candidate_ids:
                    break

                for repair_idx in range(self.max_repair_retries):
                    try:
                        repair_response_text, repair_metadata = self.repair_llm_call(
                            query,
                            candidate_snapshot_json,
                            response_text,
                            empty_output=saw_empty_output,
                            parse_error=parse_error,
                        )
                        repair_finish_reason = repair_metadata.get("finish_reason")
                        repair_truncated = self._is_truncated_finish_reason(repair_metadata)
                        if repair_truncated:
                            rerank_log["truncated_response_count"] += 1
                        rerank_log["raw_output_preview"] = repair_response_text[:MAX_RAW_OUTPUT_PREVIEW_CHARS]
                        (
                            repaired_ids,
                            repaired_parse_succeeded,
                            repaired_parse_error,
                            repaired_parsed_best_ids,
                            repaired_invalid_best_ids,
                            repaired_confidence,
                        ) = self.parse_filter(repair_response_text, num_candidates=len(candidate_items))
                        rerank_log["parsed_best_ids"] = list(repaired_parsed_best_ids)
                        rerank_log["invalid_best_ids"] = list(repaired_invalid_best_ids)
                        if repaired_confidence is not None:
                            rerank_log["confidence"] = repaired_confidence
                        if repaired_parse_succeeded and repaired_invalid_best_ids:
                            saw_schema_failure = True
                            rerank_log["schema_failure_count"] += 1
                            rerank_log["mapping_issue_detected"] = True
                            rerank_log["mapping_issue_reason_counts"]["invalid_best_id"] = (
                                rerank_log["mapping_issue_reason_counts"].get("invalid_best_id", 0)
                                + len(repaired_invalid_best_ids)
                            )
                        rerank_log["attempts"].append({
                            "attempt": attempt_idx + 1,
                            "phase": f"repair_{repair_idx + 1}",
                            "allow_think": False,
                            "parse_succeeded": repaired_parse_succeeded and bool(repaired_ids),
                            "num_generated_facts": len(repaired_ids),
                            "parse_error": (
                                repaired_parse_error
                                if not repaired_parse_succeeded else
                                ("empty_output" if not repaired_ids and not repaired_invalid_best_ids else None)
                            ),
                            "parsed_best_ids": list(repaired_parsed_best_ids),
                            "invalid_best_ids": list(repaired_invalid_best_ids),
                            "finish_reason": repair_finish_reason,
                            "truncated": repair_truncated,
                            "raw_output_preview": repair_response_text[:MAX_ATTEMPT_SUMMARY_PREVIEW_CHARS],
                        })
                        repair_attempt_status = _attempt_status(
                            parse_succeeded=repaired_parse_succeeded,
                            selected_ids=repaired_ids,
                            parsed_best_ids=repaired_parsed_best_ids,
                            invalid_best_ids=repaired_invalid_best_ids,
                            parse_error=repaired_parse_error,
                        )
                        self._record_primary_attempt_trace(
                            rerank_log,
                            2,
                            status=repair_attempt_status,
                            parse_error=repaired_parse_error,
                            raw_output_preview=repair_response_text[:MAX_RAW_OUTPUT_PREVIEW_CHARS],
                            finish_reason=repair_finish_reason,
                            truncated=repair_truncated,
                        )
                        if repaired_parse_succeeded and repaired_ids:
                            selected_candidate_ids = repaired_ids
                            parse_succeeded = True
                            rerank_log["n_facts_after_rerank_raw"] = len(selected_candidate_ids)
                            break
                        if repaired_parse_succeeded and not repaired_ids:
                            if repaired_invalid_best_ids or repaired_parsed_best_ids:
                                saw_schema_failure = True
                                rerank_log["schema_failure_count"] += 1
                                repaired_parse_succeeded = False
                                repaired_parse_error = (
                                    f"Parsed JSON but best_ids contained invalid ids {repaired_invalid_best_ids[:10]} "
                                    f"for valid range {rerank_log['valid_id_range']}."
                                )
                            else:
                                saw_empty_output = True
                                rerank_log["empty_output_count"] += 1
                                rerank_log["model_semantic_empty_count"] += 1
                                repaired_parse_succeeded = False
                        else:
                            rerank_log["repair_failure_count"] += 1
                        logger.warning(
                            "Rerank repair failed on attempt %s.%s for query %r: %s",
                            attempt_idx + 1,
                            repair_idx + 1,
                            query[:160],
                            repaired_parse_error or "empty_output",
                        )
                    except Exception as repair_exc:
                        rerank_log["repair_failure_count"] += 1
                        self._record_primary_attempt_trace(
                            rerank_log,
                            2,
                            status="llm_exception",
                            parse_error=str(repair_exc),
                            raw_output_preview=None,
                            finish_reason=None,
                            truncated=False,
                        )
                        rerank_log["attempts"].append({
                            "attempt": attempt_idx + 1,
                            "phase": f"repair_{repair_idx + 1}",
                            "allow_think": False,
                            "parse_succeeded": False,
                            "num_generated_facts": 0,
                            "parse_error": str(repair_exc),
                            "finish_reason": None,
                            "truncated": False,
                            "raw_output_preview": None,
                        })
                        logger.warning(
                            "Rerank repair LLM call failed on attempt %s.%s for query %r: %s",
                            attempt_idx + 1,
                            repair_idx + 1,
                            query[:160],
                            repair_exc,
                        )
                        break

                if parse_succeeded and selected_candidate_ids:
                    break
            except Exception as e:
                self._record_primary_attempt_trace(
                    rerank_log,
                    attempt_idx + 1,
                    status="llm_exception",
                    parse_error=str(e),
                    raw_output_preview=None,
                    finish_reason=None,
                    truncated=False,
                )
                rerank_log["attempts"].append({
                    "attempt": attempt_idx + 1,
                    "phase": "primary",
                    "allow_think": allow_think,
                    "parse_succeeded": False,
                    "num_generated_facts": 0,
                    "parse_error": str(e),
                    "finish_reason": None,
                    "truncated": False,
                    "raw_output_preview": None,
                })
                logger.warning(
                    "Rerank LLM call failed on attempt %s for query %r: %s",
                    attempt_idx + 1,
                    query[:160],
                    e,
                )
                rerank_log["parse_failure_count"] += 1

        if not parse_succeeded or not selected_candidate_ids:
            if saw_empty_output:
                fallback_reason = "empty_output"
            elif saw_schema_failure:
                fallback_reason = "schema_failure"
            else:
                fallback_reason = "parse_failure"
            return self._build_failure_result(
                candidate_items=candidate_items,
                candidate_indices=candidate_indices,
                len_after_rerank=len_after_rerank,
                rerank_log=rerank_log,
                reason=fallback_reason,
            )

        result_indices: list[int] = []
        seen_result_indices = set()
        for selected_idx in selected_candidate_ids:
            if selected_idx in seen_result_indices:
                rerank_log["mapping_duplicate_drop_count"] += 1
                rerank_log["mapping_issue_detected"] = True
                rerank_log["mapping_issue_reason_counts"]["duplicate_selected_id"] = (
                    rerank_log["mapping_issue_reason_counts"].get("duplicate_selected_id", 0) + 1
                )
                continue
            seen_result_indices.add(selected_idx)
            result_indices.append(selected_idx)
            rerank_log["mapping_exact_match_count"] += 1

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        rerank_log["n_facts_after_mapping"] = len(sorted_candidate_items)
        rerank_log["n_facts_after_postprocess"] = len(sorted_candidate_items)
        target_k = len_after_rerank or len(sorted_candidate_items)
        if sorted_candidate_items and len(sorted_candidate_items) < target_k:
            selected_indices = set(result_indices)
            fill_positions = [idx for idx in range(len(candidate_items)) if idx not in selected_indices]
            fill_needed = max(0, target_k - len(sorted_candidate_items))
            fill_positions = fill_positions[:fill_needed]
            if fill_positions:
                rerank_log["mapping_partial_recovery_applied"] = True
                rerank_log["mapping_fallback_fill_count"] = len(fill_positions)
                sorted_candidate_indices.extend(candidate_indices[idx] for idx in fill_positions)
                sorted_candidate_items.extend(candidate_items[idx] for idx in fill_positions)
        final_candidate_indices = sorted_candidate_indices[:len_after_rerank]
        final_candidate_items = sorted_candidate_items[:len_after_rerank]
        if final_candidate_items:
            rerank_log["n_facts_final"] = len(final_candidate_items)
            rerank_log["facts_empty_stage"] = "ok"
            return final_candidate_indices, final_candidate_items, rerank_log

        if sorted_candidate_items:
            return self._build_failure_result(
                candidate_items=sorted_candidate_items,
                candidate_indices=sorted_candidate_indices,
                len_after_rerank=len_after_rerank,
                rerank_log=rerank_log,
                reason="postprocess_filter_drop",
            )

        rerank_log["n_facts_final"] = 0
        rerank_log["facts_empty_stage"] = "postprocess_filter_drop"
        rerank_log["final_failure_reason"] = "postprocess_filter_drop"
        rerank_log["final_facts_empty_count"] += 1
        return final_candidate_indices, final_candidate_items, rerank_log
