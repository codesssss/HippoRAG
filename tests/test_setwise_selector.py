import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.completion = lambda **kwargs: None
    sys.modules["litellm"] = litellm_stub
if "gritlm" not in sys.modules:
    gritlm_stub = types.ModuleType("gritlm")
    gritlm_stub.GritLM = object
    sys.modules["gritlm"] = gritlm_stub
if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers_stub
if "igraph" not in sys.modules:
    igraph_stub = types.ModuleType("igraph")
    igraph_stub.Graph = object
    sys.modules["igraph"] = igraph_stub
if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = object
    vllm_stub.LLM = object
    sys.modules["vllm"] = vllm_stub
if "outlines" not in sys.modules:
    outlines_stub = types.ModuleType("outlines")
    outlines_generate_stub = types.ModuleType("outlines.generate")
    outlines_generate_stub.json = lambda *args, **kwargs: (lambda prompts, **inner_kwargs: [])
    outlines_models_stub = types.ModuleType("outlines.models")
    outlines_models_stub.Transformers = object
    sys.modules["outlines"] = outlines_stub
    sys.modules["outlines.generate"] = outlines_generate_stub
    sys.modules["outlines.models"] = outlines_models_stub

import eval_causal_qwen3 as eval_causal_qwen3_module

from eval_causal_qwen3 import (
    LEARNED_SETWISE_FEATURE_NAMES,
    OpenAICompatibleLateRerankJudge,
    SetwiseLateRerankResponseModel,
    build_expand_assemble_query_traces,
    build_requirement_title_exposure_summary,
    build_setwise_late_rerank_candidates,
    build_setwise_selector_query_traces,
    build_setwise_late_rerank_judge_bundle,
    build_report_examples,
    collect_grounded_question_query_entities,
    collect_lexical_query_seed_entities,
    collect_question_query_entities,
    compute_bridge_gate_decision,
    compute_candidate_feature_rows,
    maybe_apply_bridge_saturation_guard,
    maybe_apply_setwise_reader_order_probe,
    compute_state_path_connectivity_metrics,
    materialize_reader_top_positions,
    normalize_assemble_mode,
    normalize_setwise_late_rerank_policy,
    parse_setwise_late_rerank_response,
    rerank_completed_evidence_sets_with_llm,
    resolve_requirement_beam_runtime_reserve_config,
    resolve_reserved_positions,
    resolve_query_pool_gold_titles,
    resolve_setwise_query_targets,
    rerank_candidate_positions_for_assemble,
    score_evidence_state,
    select_bridge_append_positions,
    select_bridge_beam_positions,
    select_bridge_greedy_positions,
    select_learned_greedy_positions,
    select_requirement_beam_positions,
    should_apply_setwise_late_rerank_override,
)
from requirement_beam_utils import (
    align_requirement_cache_entry_to_pool,
    build_need_unit,
    build_need_unit_doc_annotation,
    build_heuristic_qdmr_step_plan_payload,
    build_need_unit_cache_entry,
    build_requirement_cache_entry,
    compile_qdmr_to_need_units,
    compute_requirement_state_metrics,
    extract_need_unit_atomic_features,
    load_requirement_cache,
    load_need_unit_atomic_model_bundle,
    NEED_UNIT_ATOMIC_FEATURE_NAMES,
    NEED_UNIT_ATOMIC_LABELS,
    NEED_UNIT_CACHE_VERSION,
    NEED_UNIT_MATCHER_FEATURE_NAMES,
    normalize_answer_type_label,
    REQUIREMENT_MATCHER_FEATURE_NAMES,
    requirement_feature_rows_to_matrix,
    save_requirement_cache,
    score_need_unit_support,
)
from annotate_need_unit_support import resolve_atomic_annotation_bundle
from train_need_unit_scorer import build_need_unit_atomic_training_rows
from run_requirement_beam_reserve_ablation import (
    build_requirement_reserve_ablation_jobs,
    resolve_dataset_save_dir,
)
from src.hipporag.utils.misc_utils import QuerySolution


class DummyReachabilityModel:
    def predict_proba(self, feature_matrix):
        structure_seed_idx = LEARNED_SETWISE_FEATURE_NAMES.index("structure_score_seed")
        structure_covered_idx = LEARNED_SETWISE_FEATURE_NAMES.index("structure_score_covered")
        base_score_idx = LEARNED_SETWISE_FEATURE_NAMES.index("base_score")

        positive_score = (
            0.25 * feature_matrix[:, base_score_idx]
            + 0.30 * feature_matrix[:, structure_seed_idx]
            + 0.45 * feature_matrix[:, structure_covered_idx]
        )
        positive_score = np.clip(positive_score, 0.0, 1.0)
        return np.stack([1.0 - positive_score, positive_score], axis=1)


class DummyRequirementPriorModel:
    def predict_proba(self, feature_matrix):
        support_gain_idx = REQUIREMENT_MATCHER_FEATURE_NAMES.index("support_completeness_gain")
        utility_gain_idx = REQUIREMENT_MATCHER_FEATURE_NAMES.index("utility_margin_gain")

        positive_score = (
            0.55 * feature_matrix[:, support_gain_idx]
            + 0.45 * feature_matrix[:, utility_gain_idx]
        )
        positive_score = np.clip(positive_score, 0.0, 1.0)
        return np.stack([1.0 - positive_score, positive_score], axis=1)


class DummyNeedUnitPriorModel:
    def predict_proba(self, feature_matrix):
        support_gain_idx = NEED_UNIT_MATCHER_FEATURE_NAMES.index("support_completeness_gain")
        hop_support_idx = NEED_UNIT_MATCHER_FEATURE_NAMES.index("relation_hop_support_after")

        positive_score = (
            0.60 * feature_matrix[:, support_gain_idx]
            + 0.40 * feature_matrix[:, hop_support_idx]
        )
        positive_score = np.clip(positive_score, 0.0, 1.0)
        return np.stack([1.0 - positive_score, positive_score], axis=1)


class DummyAtomicNeedUnitModel:
    def __init__(self):
        self.classes_ = np.asarray(list(NEED_UNIT_ATOMIC_LABELS), dtype=object)

    def predict_proba(self, feature_matrix):
        title_bridge_idx = NEED_UNIT_ATOMIC_FEATURE_NAMES.index("title_bridge_alignment")
        outputs = []
        for row in feature_matrix:
            if row[title_bridge_idx] > 0.5:
                outputs.append([0.1, 0.7, 0.1, 0.1])
            else:
                outputs.append([0.7, 0.1, 0.1, 0.1])
        return np.asarray(outputs, dtype=float)


class DummyLateRerankModel:
    def __init__(self, response_text):
        if isinstance(response_text, list):
            self.responses = list(response_text)
        else:
            self.responses = [response_text]
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        response_text = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return response_text, {"finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}


class DummyCrossEncoder:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def compute_score(self, pairs):
        self.calls.append(list(pairs))
        return list(self.scores[:len(pairs)])


class DummyParsedResponse:
    def __init__(self, parsed_payload=None, output_text="", usage=None, status="completed", output=None):
        self.output_parsed = parsed_payload
        self.output_text = output_text
        self.output = output or []
        self.usage = usage or type("Usage", (), {"input_tokens": 7, "output_tokens": 3})()
        self.status = status


class DummyResponseContentPart:
    def __init__(self, text=""):
        self.text = text


class DummyResponseOutputItem:
    def __init__(self, *content_parts):
        self.content = list(content_parts)


class DummyChatParsedMessage:
    def __init__(self, parsed_payload=None, content=""):
        self.parsed = parsed_payload
        self.content = content


class DummyChatParsedChoice:
    def __init__(self, parsed_payload=None, content="", finish_reason="stop"):
        self.message = DummyChatParsedMessage(parsed_payload=parsed_payload, content=content)
        self.finish_reason = finish_reason


class DummyChatParsedResponse:
    def __init__(self, parsed_payload=None, content="", finish_reason="stop"):
        self.choices = [DummyChatParsedChoice(parsed_payload=parsed_payload, content=content, finish_reason=finish_reason)]
        self.usage = type("Usage", (), {"prompt_tokens": 5, "completion_tokens": 2})()


class DummyChatStreamDelta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class DummyChatStreamChoice:
    def __init__(self, content=None, reasoning_content=None, finish_reason=None):
        self.delta = DummyChatStreamDelta(content=content, reasoning_content=reasoning_content)
        self.finish_reason = finish_reason


class DummyChatStreamChunk:
    def __init__(self, content=None, reasoning_content=None, finish_reason=None, usage=None):
        self.choices = [DummyChatStreamChoice(content=content, reasoning_content=reasoning_content, finish_reason=finish_reason)]
        self.usage = usage


class DummyResponsesAPI:
    def __init__(self, parsed_payload=None, output_text="", output=None):
        self.parsed_payload = parsed_payload
        self.output_text = output_text
        self.output = output or []
        self.parse_calls = []
        self.create_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return DummyParsedResponse(parsed_payload=self.parsed_payload)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return DummyParsedResponse(output_text=self.output_text, output=self.output)


class DummyBetaChatCompletionsAPI:
    def __init__(self, parsed_payload):
        self.parsed_payload = parsed_payload
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return DummyChatParsedResponse(parsed_payload=self.parsed_payload)


class DummyBetaChatAPI:
    def __init__(self, parsed_payload):
        self.completions = DummyBetaChatCompletionsAPI(parsed_payload=parsed_payload)


class DummyChatCompletionsAPI:
    def __init__(self, stream_chunks=None):
        self.stream_chunks = list(stream_chunks or [])
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.stream_chunks)


class DummyChatAPI:
    def __init__(self, stream_chunks=None):
        self.completions = DummyChatCompletionsAPI(stream_chunks=stream_chunks)


class DummyOpenAIClient:
    def __init__(self, parsed_payload=None, response_output_text="", response_output=None, chat_stream_chunks=None):
        self.responses = DummyResponsesAPI(
            parsed_payload=parsed_payload,
            output_text=response_output_text,
            output=response_output,
        )
        self.beta = type("BetaNamespace", (), {"chat": DummyBetaChatAPI(parsed_payload=parsed_payload)})()
        self.chat = DummyChatAPI(stream_chunks=chat_stream_chunks)


class DummyCausalV2Engine:
    def __init__(self, entities=None):
        self.entities = list(entities or [])

    def _extract_query_entities(self, query):
        return list(self.entities)


class DummyHippoRAGForQueryEntities:
    def __init__(self, entities=None):
        self.causal_v2_engine = DummyCausalV2Engine(entities=entities)


class DummyHippoRAGForAssemble:
    def __init__(self, query_embeddings=None, passage_embeddings=None):
        self.query_to_embedding = {"passage": dict(query_embeddings or {})}
        self.passage_embeddings = np.asarray(passage_embeddings if passage_embeddings is not None else np.zeros((0, 0)), dtype=float)

    def _get_passage_query_embeddings(self, queries):
        return None


def test_materialize_reader_top_positions_preserves_selected_prefix_and_fills_tail():
    top_positions = materialize_reader_top_positions(
        selected_positions=[3, 1, 3],
        pool_limit=6,
        qa_top_k=5,
    )

    assert top_positions == [3, 1, 0, 2, 4]


def test_materialize_reader_top_positions_can_force_prefix_positions():
    top_positions = materialize_reader_top_positions(
        selected_positions=[3, 1, 3],
        pool_limit=6,
        qa_top_k=5,
        forced_prefix_positions=[5, 1],
    )

    assert top_positions == [5, 1, 3, 0, 2]


def test_maybe_apply_setwise_reader_order_probe_promotes_best_bridge_to_slot3():
    reordered, trace = maybe_apply_setwise_reader_order_probe(
        final_front_positions=[0, 1, 2, 4, 5],
        selector_trace={
            "selection_steps": [
                {"mode": "anchor", "pool_position": 0, "doc_id": 10},
                {"mode": "anchor", "pool_position": 1, "doc_id": 11},
                {"mode": "reserve", "pool_position": 2, "doc_id": 12},
                {"mode": "beam", "pool_position": 4, "doc_id": 14, "structure_score": 0.85, "closure_score": 0.40, "novelty_score": 0.30},
                {"mode": "beam", "pool_position": 5, "doc_id": 15, "structure_score": 0.95, "closure_score": 0.60, "novelty_score": 0.50},
            ],
        },
        pool_doc_ids=[10, 11, 12, 13, 14, 15],
        pool_doc_titles=["A", "B", "C", "Noise", "Bridge 1", "Bridge 2"],
        probe_mode="promote_best_bridge_to_slot3",
    )

    assert reordered == [0, 1, 5, 2, 4]
    assert set(reordered) == {0, 1, 2, 4, 5}
    assert trace["applied"] is True
    assert trace["promoted_pool_position"] == 5
    assert trace["promoted_from_rank"] == 5
    assert trace["original_front_titles"] == ["A", "B", "C", "Bridge 1", "Bridge 2"]
    assert trace["probed_front_titles"] == ["A", "B", "Bridge 2", "C", "Bridge 1"]


def test_maybe_apply_setwise_reader_order_probe_promotes_best_bridge_to_slot2():
    reordered, trace = maybe_apply_setwise_reader_order_probe(
        final_front_positions=[0, 1, 2, 4, 5],
        selector_trace={
            "selection_steps": [
                {"mode": "beam", "pool_position": 4, "doc_id": 14, "structure_score": 0.92, "closure_score": 0.55, "novelty_score": 0.45},
            ],
        },
        pool_doc_ids=[10, 11, 12, 13, 14, 15],
        pool_doc_titles=["A", "B", "C", "Noise", "Bridge 1", "Bridge 2"],
        probe_mode="promote_best_bridge_to_slot2",
    )

    assert reordered == [0, 4, 1, 2, 5]
    assert set(reordered) == {0, 1, 2, 4, 5}
    assert trace["applied"] is True
    assert trace["promoted_pool_position"] == 4
    assert trace["promoted_from_rank"] == 4


def test_maybe_apply_setwise_reader_order_probe_is_noop_without_bridge_doc():
    reordered, trace = maybe_apply_setwise_reader_order_probe(
        final_front_positions=[0, 1, 2, 3, 4],
        selector_trace={
            "selection_steps": [
                {"mode": "anchor", "pool_position": 0, "doc_id": 10},
                {"mode": "reserve", "pool_position": 2, "doc_id": 12},
            ],
        },
        pool_doc_ids=[10, 11, 12, 13, 14],
        pool_doc_titles=["A", "B", "C", "D", "E"],
        probe_mode="promote_best_bridge_to_slot3",
    )

    assert reordered == [0, 1, 2, 3, 4]
    assert trace["applied"] is False
    assert trace["skip_reason"] == "no_bridge_doc_in_final_front"


def test_maybe_apply_setwise_reader_order_probe_is_noop_when_bridge_already_in_prefix():
    reordered, trace = maybe_apply_setwise_reader_order_probe(
        final_front_positions=[0, 4, 1, 2, 3],
        selector_trace={
            "selection_steps": [
                {"mode": "beam", "pool_position": 4, "doc_id": 14, "structure_score": 0.92, "closure_score": 0.55, "novelty_score": 0.45},
            ],
        },
        pool_doc_ids=[10, 11, 12, 13, 14],
        pool_doc_titles=["A", "B", "C", "D", "Bridge 1"],
        probe_mode="promote_best_bridge_to_slot2",
    )

    assert reordered == [0, 4, 1, 2, 3]
    assert trace["applied"] is False
    assert trace["skip_reason"] == "bridge_already_in_prefix"


def test_parse_setwise_late_rerank_response_accepts_best_id_schema():
    parsed = parse_setwise_late_rerank_response(
        response_text='<JSON>{"best_id": 2, "confidence": 0.8}</JSON>',
        num_candidates=4,
    )

    assert parsed["parse_succeeded"] is True
    assert parsed["best_id"] == 2
    assert parsed["confidence"] == 0.8


def test_rerank_completed_evidence_sets_with_llm_selects_valid_candidate():
    llm_model = DummyLateRerankModel('<JSON>{"best_id": 1, "confidence": 0.67}</JSON>')
    candidate_sets = [
        {
            "source": "heuristic_best",
            "reader_top_positions": [0, 1],
        },
        {
            "source": "baseline_topk",
            "reader_top_positions": [2, 3],
        },
    ]
    best_id, trace = rerank_completed_evidence_sets_with_llm(
        query="Which city is person A from?",
        pool_docs=[
            "Doc A\nPerson A was born in City X.",
            "Doc B\nPerson A won an award.",
            "Doc C\nCity Y is in Country Z.",
            "Doc D\nCity X is in Country Z.",
        ],
        candidate_sets=candidate_sets,
        llm_infer_fn=llm_model.infer,
        model_name="dummy-model",
        max_doc_chars=80,
    )

    assert best_id == 1
    assert trace["parse_succeeded"] is True
    assert trace["selected_candidate_id"] == 1
    assert len(llm_model.calls) == 1


def test_rerank_completed_evidence_sets_with_llm_repairs_invalid_first_response():
    llm_model = DummyLateRerankModel([
        "best_id: 1",
        '<JSON>{"best_id": 0, "confidence": 0.55}</JSON>',
    ])
    best_id, trace = rerank_completed_evidence_sets_with_llm(
        query="Which city is person A from?",
        pool_docs=[
            "Doc A\nPerson A was born in City X.",
            "Doc B\nPerson A won an award.",
            "Doc C\nCity Y is in Country Z.",
            "Doc D\nCity X is in Country Z.",
        ],
        candidate_sets=[
            {"source": "heuristic_best", "reader_top_positions": [0, 1]},
            {"source": "baseline_topk", "reader_top_positions": [2, 3]},
        ],
        llm_infer_fn=llm_model.infer,
        model_name="dummy-model",
        max_doc_chars=80,
    )

    assert best_id == 0
    assert trace["repair_applied"] is True
    assert trace["repair_parse_succeeded"] is True
    assert len(llm_model.calls) == 2


def test_openai_compatible_late_rerank_judge_responses_backend_uses_create_and_returns_raw_json():
    judge = OpenAICompatibleLateRerankJudge(
        model_name="gpt-5.4",
        base_url="https://example.com/v1",
        api_key="sk-test",
        backend="responses",
        client=DummyOpenAIClient(
            parsed_payload=SetwiseLateRerankResponseModel(best_id=2, confidence=0.81),
            response_output_text='<JSON>{"best_id": 2, "confidence": 0.81}</JSON>',
        ),
    )

    response_text, metadata = judge.infer(
        messages=[{"role": "user", "content": "pick the best set"}],
        model="gpt-5.4",
        response_format=SetwiseLateRerankResponseModel,
        max_completion_tokens=64,
        temperature=0,
        top_p=1,
    )

    parsed = parse_setwise_late_rerank_response(response_text, num_candidates=4)
    assert parsed["best_id"] == 2
    assert parsed["confidence"] == 0.81
    assert metadata["backend"] == "responses"
    assert metadata["prompt_tokens"] == 7
    assert metadata["response_text_source"] == "output_text"
    assert metadata["structured_output_requested"] is True
    assert len(judge.client.responses.create_calls) == 1
    assert len(judge.client.responses.parse_calls) == 0


def test_openai_compatible_late_rerank_judge_responses_backend_extracts_text_from_output_items():
    judge = OpenAICompatibleLateRerankJudge(
        model_name="gpt-5.4",
        base_url="https://example.com/v1",
        api_key="sk-test",
        backend="responses",
        client=DummyOpenAIClient(
            response_output=[
                DummyResponseOutputItem(
                    DummyResponseContentPart('<JSON>{"best_id": 1}</JSON>'),
                ),
            ],
        ),
    )

    response_text, metadata = judge.infer(
        messages=[{"role": "user", "content": "pick the best set"}],
        model="gpt-5.4",
        response_format=None,
        max_completion_tokens=64,
        temperature=0,
        top_p=1,
    )

    parsed = parse_setwise_late_rerank_response(response_text, num_candidates=3)
    assert parsed["best_id"] == 1
    assert metadata["response_text_source"] == "output"


def test_openai_compatible_late_rerank_judge_chat_backend_returns_structured_json():
    judge = OpenAICompatibleLateRerankJudge(
        model_name="gpt-5.4",
        base_url="https://example.com/v1",
        api_key="sk-test",
        backend="chat_completions",
        client=DummyOpenAIClient(
            parsed_payload=SetwiseLateRerankResponseModel(best_id=1, confidence=0.64),
        ),
    )

    response_text, metadata = judge.infer(
        messages=[{"role": "user", "content": "pick the best set"}],
        model="gpt-5.4",
        response_format=SetwiseLateRerankResponseModel,
        max_completion_tokens=64,
        temperature=0,
        top_p=1,
    )

    parsed = SetwiseLateRerankResponseModel.model_validate_json(response_text)
    assert parsed.best_id == 1
    assert parsed.confidence == 0.64
    assert metadata["backend"] == "chat_completions"
    assert metadata["completion_tokens"] == 2


def test_openai_compatible_late_rerank_judge_chat_backend_stream_collects_content_only():
    judge = OpenAICompatibleLateRerankJudge(
        model_name="gpt-5.4",
        base_url="https://example.com/v1",
        api_key="sk-test",
        backend="chat_completions",
        client=DummyOpenAIClient(
            chat_stream_chunks=[
                DummyChatStreamChunk(reasoning_content="thinking"),
                DummyChatStreamChunk(content="<JSON>"),
                DummyChatStreamChunk(content='{"best_id":0,"confidence":0.7}'),
                DummyChatStreamChunk(content="</JSON>", finish_reason="stop", usage=type("Usage", (), {"prompt_tokens": 8, "completion_tokens": 4})()),
            ],
        ),
    )

    response_text, metadata = judge.infer(
        messages=[{"role": "user", "content": "pick the best set"}],
        model="gpt-5.4",
        response_format=None,
        max_completion_tokens=64,
        temperature=0,
        top_p=1,
    )

    parsed = parse_setwise_late_rerank_response(response_text, num_candidates=2)
    assert parsed["best_id"] == 0
    assert parsed["confidence"] == 0.7
    assert metadata["streamed"] is True
    assert metadata["prompt_tokens"] == 8
    assert metadata["completion_tokens"] == 4
    assert judge.client.chat.completions.calls[0]["stream"] is True
    assert judge.client.chat.completions.calls[0]["max_tokens"] == 64


def test_build_setwise_late_rerank_judge_bundle_uses_env_key_and_raw_response_for_responses_backend():
    args = type(
        "Args",
        (),
        {
            "setwise_late_rerank_judge_backend": "responses",
            "setwise_late_rerank_judge_model": "gpt-5.4",
            "setwise_late_rerank_judge_base_url": "https://example.com/v1",
            "setwise_late_rerank_judge_api_key": "",
            "setwise_late_rerank_judge_api_key_env": "OPENAI_API_KEY",
            "setwise_late_rerank_judge_reasoning_effort": "low",
            "setwise_late_rerank_judge_timeout_s": 30.0,
        },
    )()

    old_api_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-from-env"
    try:
        bundle = build_setwise_late_rerank_judge_bundle(
            args=args,
            fallback_model_name="fallback-model",
            fallback_base_url="https://fallback.example/v1",
        )
    finally:
        if old_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_api_key

    assert bundle.backend == "responses"
    assert bundle.model_name == "gpt-5.4"
    assert bundle.base_url == "https://example.com/v1"
    assert bundle.response_format is None
    assert bundle.reasoning_effort == "low"
    assert callable(bundle.infer_fn)


def test_build_setwise_late_rerank_judge_bundle_keeps_structured_response_for_chat_backend():
    args = type(
        "Args",
        (),
        {
            "setwise_late_rerank_judge_backend": "chat_completions",
            "setwise_late_rerank_judge_model": "gpt-5.4",
            "setwise_late_rerank_judge_base_url": "https://example.com/v1",
            "setwise_late_rerank_judge_api_key": "sk-test",
            "setwise_late_rerank_judge_api_key_env": "OPENAI_API_KEY",
            "setwise_late_rerank_judge_reasoning_effort": "",
            "setwise_late_rerank_judge_timeout_s": 30.0,
        },
    )()

    bundle = build_setwise_late_rerank_judge_bundle(
        args=args,
        fallback_model_name="fallback-model",
        fallback_base_url="https://fallback.example/v1",
    )

    assert bundle.backend == "chat_completions"
    assert bundle.response_format is None


def test_build_setwise_late_rerank_candidates_dedups_equivalent_reader_topk():
    candidates = build_setwise_late_rerank_candidates(
        selected_positions=[0, 2],
        selector_trace={
            "beam_best_state_score": 0.91,
            "beam_best_cumulative_score": 1.37,
            "beam_finalists": [
                {"selected_positions": [0, 2]},
                {"selected_positions": [0, 1]},
            ],
        },
        pool_limit=5,
        qa_top_k=3,
        include_baseline=True,
        max_candidates=5,
    )

    assert [candidate["source"] for candidate in candidates] == [
        "heuristic_best",
        "beam_finalist",
    ]
    assert candidates[0]["reader_top_positions"] == [0, 2, 1]
    assert candidates[0]["state_score"] == 0.91
    assert candidates[0]["cumulative_score"] == 1.37
    assert candidates[1]["reader_top_positions"] == [0, 1, 2]


def test_normalize_setwise_late_rerank_policy_accepts_known_modes():
    assert normalize_setwise_late_rerank_policy("always") == "always"
    assert normalize_setwise_late_rerank_policy("TieBreak") == "tiebreak"


def test_should_apply_setwise_late_rerank_override_allows_tight_tiebreak():
    allow_override, info = should_apply_setwise_late_rerank_override(
        heuristic_candidate={"state_score": 0.82},
        chosen_candidate={"state_score": 0.805},
        policy="tiebreak",
        max_state_score_gap=0.02,
    )

    assert allow_override is True
    assert info["override_policy"] == "tiebreak"
    assert info["override_block_reason"] is None
    assert np.isclose(info["override_state_score_gap"], 0.015)


def test_should_apply_setwise_late_rerank_override_blocks_wide_gap():
    allow_override, info = should_apply_setwise_late_rerank_override(
        heuristic_candidate={"state_score": 0.82},
        chosen_candidate={"state_score": 0.73},
        policy="tiebreak",
        max_state_score_gap=0.02,
    )

    assert allow_override is False
    assert info["override_block_reason"] == "state_score_gap_exceeded"
    assert np.isclose(info["override_state_score_gap"], 0.09)


def test_should_apply_setwise_late_rerank_override_blocks_missing_state_score():
    allow_override, info = should_apply_setwise_late_rerank_override(
        heuristic_candidate={"state_score": 0.82},
        chosen_candidate={},
        policy="tiebreak",
        max_state_score_gap=0.02,
    )

    assert allow_override is False
    assert info["override_block_reason"] == "missing_state_score"
    assert info["selected_candidate_state_score"] is None


def test_select_bridge_greedy_positions_prefers_bridge_docs_over_high_rank_distractor():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        pool_doc_titles=["Film X", "Film Y", "Director A", "Director B", "Noise"],
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"film y", "person b"},
            2: {"person a", "birth a"},
            3: {"person b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("person a", "birth a", 1.0, "related_to")],
            3: [("person b", "birth b", 1.0, "related_to")],
            4: [("noise", "other noise", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("birth a", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities={"person a", "person b"},
        anchor_count=2,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["anchor_positions"] == [0, 1]
    assert trace["selection_steps"][2]["doc_id"] == 2
    assert trace["selection_steps"][3]["doc_id"] == 3


def test_select_bridge_greedy_positions_falls_back_to_rank_order_without_structure_signal():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[10, 11, 12, 13],
        pool_doc_scores=np.array([0.80, 0.60, 0.40, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D"],
        doc_idx_to_entities={
            10: {"a"},
            11: {"b"},
            12: {"c"},
            13: {"d"},
        },
        doc_idx_to_edges={
            10: [],
            11: [],
            12: [],
            13: [],
        },
        adjacency={},
        qa_top_k=3,
        initial_seed_entities=set(),
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 2]
    assert trace["selection_steps"][0]["mode"] == "anchor"
    assert trace["selection_steps"][1]["pool_position"] == 1


def test_select_bridge_greedy_positions_can_bootstrap_from_anchor_entities():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        pool_doc_titles=["Film X", "Film Y", "Director A", "Director B", "Noise"],
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"film y", "person b"},
            2: {"person a", "birth a"},
            3: {"person b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("person a", "birth a", 1.0, "related_to")],
            3: [("person b", "birth b", 1.0, "related_to")],
            4: [("noise", "other noise", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("birth a", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities=set(),
        anchor_count=2,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["selection_steps"][2]["doc_id"] == 2


def test_select_bridge_beam_positions_matches_bridge_completion_case():
    selected_positions, trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        pool_doc_titles=["Film X", "Film Y", "Director A", "Director B", "Noise"],
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"film y", "person b"},
            2: {"person a", "birth a"},
            3: {"person b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("person a", "birth a", 1.0, "related_to")],
            3: [("person b", "birth b", 1.0, "related_to")],
            4: [("noise", "other noise", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("birth a", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities=set(),
        anchor_count=2,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        beam_width=4,
        beam_expand_per_state=4,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["beam_width"] == 4
    assert trace["beam_expand_per_state"] == 4
    assert trace["selection_steps"][2]["doc_id"] == 2
    assert trace["selection_steps"][3]["doc_id"] == 3


def test_select_bridge_beam_positions_recovers_two_step_chain_when_greedy_takes_distractor():
    common_kwargs = dict(
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_scores=np.array([1.0, 0.95, 0.30, 0.05], dtype=float),
        pool_doc_titles=["Film X", "Helper H", "Bridge AB", "Birth B"],
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"person a", "helper h"},
            2: {"person a", "helper h", "person b"},
            3: {"person b", "birth b"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h", 1.0, "related_to")],
            2: [("person a", "helper h", 0.40, "related_to")],
            3: [("person b", "birth b", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=3,
        initial_seed_entities=set(),
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    greedy_positions, greedy_trace = select_bridge_greedy_positions(**common_kwargs)
    beam_positions, beam_trace = select_bridge_beam_positions(
        **common_kwargs,
        beam_width=2,
        beam_expand_per_state=2,
    )

    assert greedy_positions == [0, 1, 3]
    assert beam_positions == [0, 2, 3]
    assert greedy_trace["selection_steps"][1]["doc_id"] == 1
    assert beam_trace["selection_steps"][1]["doc_id"] == 2
    assert beam_trace["selection_steps"][2]["doc_id"] == 3
    assert beam_trace["beam_best_cumulative_score"] > 1.0


def test_select_bridge_greedy_positions_closure_proxy_prefers_query_aligned_bridge():
    common_kwargs = dict(
        pool_doc_ids=[0, 1, 2],
        pool_doc_scores=np.array([1.0, 0.85, 0.84], dtype=float),
        pool_doc_titles=["Anchor", "Award", "Birth"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "award a"},
            2: {"person a", "birth a"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "award a", 1.0, "related_to")],
            2: [("person a", "birth a", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("award a", 1.0, "related_to"), ("birth a", 1.0, "related_to")],
        },
        qa_top_k=2,
        initial_seed_entities={"person a"},
        query_entities={"birth a"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    bridge_positions, _ = select_bridge_greedy_positions(
        **common_kwargs,
        score_mode="bridge",
    )
    closure_positions, closure_trace = select_bridge_greedy_positions(
        **common_kwargs,
        score_mode="closure_proxy",
    )

    assert bridge_positions == [0, 1]
    assert closure_positions == [0, 2]
    assert closure_trace["score_mode"] == "closure_proxy"
    assert closure_trace["selection_steps"][1]["doc_id"] == 2
    assert closure_trace["selection_steps"][1]["closure_score"] > 0.0


def test_select_bridge_greedy_positions_reserve_top_m_keeps_prefix_before_bridge_fill():
    selected_positions, trace = select_bridge_greedy_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.85, 0.30, 0.25], dtype=float),
        pool_doc_titles=["Film X", "Film Y", "Distractor", "Director A", "Director B"],
        doc_idx_to_entities={
            0: {"film x", "person a"},
            1: {"film y", "person b"},
            2: {"distractor"},
            3: {"person a", "birth a"},
            4: {"person b", "birth b"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [("person a", "birth a", 1.0, "related_to")],
            4: [("person b", "birth b", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("birth a", 1.0, "related_to")],
            "person b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities={"person a", "person b"},
        anchor_count=2,
        reserve_top_m=3,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["anchor_positions"] == [0, 1]
    assert trace["reserved_positions"] == [0, 1, 2]
    assert trace["selection_steps"][2]["mode"] == "reserve"
    assert trace["selection_steps"][3]["mode"] == "greedy"


def test_select_bridge_beam_positions_non_anchor_title_dedup_skips_duplicate_titles():
    selected_positions, trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_scores=np.array([1.0, 0.9, 0.8, 0.7], dtype=float),
        pool_doc_titles=["Alpha", "Beta", "Beta", "Gamma"],
        doc_idx_to_entities={
            0: {"alpha"},
            1: {"beta"},
            2: {"beta-dup"},
            3: {"gamma"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
        },
        adjacency={},
        qa_top_k=3,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=2,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        beam_width=2,
        beam_expand_per_state=2,
        non_anchor_title_dedup=True,
    )

    assert selected_positions == [0, 1, 3]
    assert trace["non_anchor_title_dedup"] is True
    assert trace["selection_steps"][2]["mode"] == "beam"
    assert trace["selection_steps"][2]["pool_position"] == 3


def test_select_bridge_beam_positions_max_bridge_slots_only_fills_suffix_budget():
    selected_positions, trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.84, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Noise", "Bridge"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"noise"},
            5: {"entity d", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [("entity d", "target", 1.0, "related_to")],
        },
        adjacency={
            "entity d": [("target", 1.0, "related_to")],
        },
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        beam_width=2,
        beam_expand_per_state=2,
    )

    assert selected_positions == [0, 1, 2, 3, 5]
    assert trace["selection_target_k"] == 5
    assert trace["max_bridge_slots"] == 1
    assert trace["selection_steps"][-1]["mode"] == "beam"
    assert trace["selection_steps"][-1]["pool_position"] == 5


def test_score_evidence_state_prefers_connected_chain_over_redundant_helpers():
    pool_doc_ids = [0, 1, 2, 3, 4]
    normalized_base_scores = np.array([1.0, 0.9444, 0.9333, 0.5556, 0.0], dtype=float)
    score_redundant = score_evidence_state(
        pool_doc_ids=pool_doc_ids,
        normalized_base_scores=normalized_base_scores,
        pool_doc_titles=["Anchor", "Helper1", "Helper2", "Bridge", "Answer"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h1"},
            2: {"person a", "helper h2"},
            3: {"person a", "person b"},
            4: {"person b", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h1", 1.0, "related_to")],
            2: [("person a", "helper h2", 1.0, "related_to")],
            3: [("person a", "person b", 0.9, "related_to")],
            4: [("person b", "target", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h1", 1.0, "related_to"), ("helper h2", 1.0, "related_to"), ("person b", 0.9, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        selected_positions=[0, 1, 2],
        fixed_prefix_positions=[0],
        seed_entities={"person a"},
        query_entities={"target"},
        support_query_entities={"target"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )
    score_connected_chain = score_evidence_state(
        pool_doc_ids=pool_doc_ids,
        normalized_base_scores=normalized_base_scores,
        pool_doc_titles=["Anchor", "Helper1", "Helper2", "Bridge", "Answer"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h1"},
            2: {"person a", "helper h2"},
            3: {"person a", "person b"},
            4: {"person b", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h1", 1.0, "related_to")],
            2: [("person a", "helper h2", 1.0, "related_to")],
            3: [("person a", "person b", 0.9, "related_to")],
            4: [("person b", "target", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h1", 1.0, "related_to"), ("helper h2", 1.0, "related_to"), ("person b", 0.9, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        selected_positions=[0, 3, 4],
        fixed_prefix_positions=[0],
        seed_entities={"person a"},
        query_entities={"target"},
        support_query_entities={"target"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert score_connected_chain["path_connectivity"] == 1.0
    assert score_connected_chain["reachable_doc_ratio"] == 1.0
    assert score_connected_chain["query_reachability"] == 1.0
    assert score_connected_chain["state_score"] > score_redundant["state_score"]


def test_score_evidence_state_distinguishes_query_coverage_from_query_reachability():
    score = score_evidence_state(
        pool_doc_ids=[0, 1, 2],
        normalized_base_scores=np.array([1.0, 0.9, 0.1], dtype=float),
        pool_doc_titles=["Anchor", "Helper", "Answer"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h"},
            2: {"person b", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h", 1.0, "related_to")],
            2: [("person b", "target", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h", 1.0, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        selected_positions=[0, 1, 2],
        fixed_prefix_positions=[0],
        seed_entities={"person a"},
        query_entities={"target"},
        support_query_entities={"target"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert score["query_coverage"] == 1.0
    assert score["query_reachability"] == 0.0
    assert score["path_connectivity"] == 0.5
    assert score["reachable_doc_ratio"] == 0.5


def test_score_evidence_state_uses_suffix_base_mean_not_prefix_base_mean():
    score = score_evidence_state(
        pool_doc_ids=[0, 1, 2],
        normalized_base_scores=np.array([1.0, 0.90, 0.20], dtype=float),
        pool_doc_titles=["Anchor", "Prefix", "Suffix"],
        doc_idx_to_entities={
            0: {"anchor"},
            1: {"prefix"},
            2: {"suffix"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
        },
        adjacency={},
        selected_positions=[0, 1, 2],
        fixed_prefix_positions=[0, 1],
        seed_entities={"anchor"},
        query_entities={"suffix"},
        support_query_entities={"suffix"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert score["suffix_base_mean"] == 0.20
    assert score["query_coverage"] == 1.0


def test_score_evidence_state_respects_custom_state_weights():
    score = score_evidence_state(
        pool_doc_ids=[0, 1],
        normalized_base_scores=np.array([1.0, 0.2], dtype=float),
        pool_doc_titles=["Anchor", "Suffix"],
        doc_idx_to_entities={
            0: {"anchor"},
            1: {"target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
        },
        adjacency={},
        selected_positions=[0, 1],
        fixed_prefix_positions=[0],
        seed_entities={"anchor"},
        query_entities={"target"},
        support_query_entities={"target"},
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        state_weight_config={
            "path_connectivity": 0.0,
            "reachable_doc_ratio": 0.0,
            "query_reachability": 0.0,
            "support_mean": 0.0,
            "support_min": 0.0,
            "closure_mean": 0.0,
            "suffix_base_mean": 0.5,
            "query_coverage": 0.5,
            "frontier_ratio": 0.0,
            "redundancy_penalty": 0.0,
        },
    )

    assert score["state_score"] == 0.6
    assert score["state_weight_config"]["suffix_base_mean"] == 0.5


def test_compute_state_path_connectivity_metrics_is_order_invariant_for_small_focus_sets():
    common_kwargs = dict(
        selected_doc_entities={
            0: {"person a", "bridge x"},
            1: {"target"},
            2: {"bridge x", "target"},
        },
        initial_reachable_entities={"person a"},
        query_entities={"target"},
        adjacency={
            "person a": [("bridge x", 1.0, "related_to")],
            "bridge x": [("target", 1.0, "related_to")],
        },
        structure_max_hops=2,
    )

    forward = compute_state_path_connectivity_metrics(
        focus_positions=[0, 2, 1],
        **common_kwargs,
    )
    reverse = compute_state_path_connectivity_metrics(
        focus_positions=[1, 2, 0],
        **common_kwargs,
    )

    assert forward["path_search_mode"] == "exact"
    assert reverse["path_search_mode"] == "exact"
    assert forward["path_connectivity"] == 1.0
    assert reverse["path_connectivity"] == 1.0
    assert forward["reachable_doc_ratio"] == 1.0
    assert reverse["reachable_doc_ratio"] == 1.0
    assert forward["query_reachability"] == 1.0
    assert reverse["query_reachability"] == 1.0


def test_select_bridge_beam_positions_set_closure_reranks_by_state_score():
    common_kwargs = dict(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([1.0, 0.95, 0.94, 0.60, 0.10], dtype=float),
        pool_doc_titles=["Anchor", "Helper1", "Helper2", "Bridge", "Answer"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h1"},
            2: {"person a", "helper h2"},
            3: {"person a", "person b"},
            4: {"person b", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h1", 1.0, "related_to")],
            2: [("person a", "helper h2", 1.0, "related_to")],
            3: [("person a", "person b", 0.9, "related_to")],
            4: [("person b", "target", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h1", 1.0, "related_to"), ("helper h2", 1.0, "related_to"), ("person b", 0.9, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        qa_top_k=3,
        initial_seed_entities={"person a"},
        query_entities={"target"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        beam_width=3,
        beam_expand_per_state=3,
    )

    closure_positions, closure_trace = select_bridge_beam_positions(
        **common_kwargs,
        score_mode="closure_proxy",
    )
    set_positions, set_trace = select_bridge_beam_positions(
        **common_kwargs,
        score_mode="set_closure",
    )

    assert closure_positions == [0, 1, 2]
    assert set_positions == [0, 1, 4]
    assert closure_trace["beam_rank_metric"] == "cumulative_score"
    assert set_trace["beam_rank_metric"] == "state_score"
    assert set_trace["beam_best_state_path_connectivity"] < 1.0
    assert set_trace["beam_best_state_reachable_doc_ratio"] == 1.0
    assert set_trace["beam_best_state_query_reachability"] == 1.0
    assert set_trace["beam_best_state_suffix_base_mean"] > 0.1
    assert set_trace["beam_best_state_score"] > closure_trace["beam_best_state_score"]
    assert set_trace["beam_finalists"][0]["selected_positions"] == [0, 1, 4]
    assert any(finalist["selected_positions"] == [0, 3, 4] for finalist in set_trace["beam_finalists"])


def test_select_bridge_beam_positions_projected_shortlist_can_rescue_low_rank_bridge():
    common_kwargs = dict(
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_scores=np.array([1.0, 0.95, 0.94, 0.30], dtype=float),
        pool_doc_titles=["Anchor", "Helper H1", "Helper H2", "Bridge B"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "helper h1"},
            2: {"person a", "helper h2"},
            3: {"person a", "person b"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "helper h1", 1.0, "related_to")],
            2: [("person a", "helper h2", 1.0, "related_to")],
            3: [("person a", "person b", 1.0, "related_to")],
        },
        adjacency={
            "person a": [("helper h1", 1.0, "related_to"), ("helper h2", 1.0, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        qa_top_k=2,
        initial_seed_entities={"person a"},
        query_entities={"person b"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        score_mode="set_closure",
        beam_width=1,
        beam_expand_per_state=1,
    )

    legacy_positions, legacy_trace = select_bridge_beam_positions(
        **common_kwargs,
        beam_projected_shortlist_factor=1,
    )
    projected_positions, projected_trace = select_bridge_beam_positions(
        **common_kwargs,
        beam_projected_shortlist_factor=3,
    )

    assert legacy_positions == [0, 1]
    assert projected_positions == [0, 3]
    assert legacy_trace["beam_projected_shortlist_factor"] == 1
    assert projected_trace["beam_projected_shortlist_factor"] == 3
    assert legacy_trace["beam_projection_rescue_count"] == 0
    assert projected_trace["beam_projection_eval_count"] == 3
    assert projected_trace["beam_projection_extra_eval_count"] == 2
    assert projected_trace["beam_projection_rescue_count"] == 1
    assert projected_trace["beam_projection_changed_state_count"] == 1
    assert projected_trace["beam_projection_max_selected_rank"] == 3
    assert projected_trace["selection_steps"][1]["proposal_rank"] > 1
    assert projected_trace["beam_best_state_query_reachability"] > legacy_trace["beam_best_state_query_reachability"]


def test_select_bridge_beam_positions_set_closure_stops_before_zero_signal_filler():
    selected_positions, trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([1.0, 0.35, 0.30, 0.95, 0.90], dtype=float),
        pool_doc_titles=["Anchor", "Bridge", "Answer", "Filler1", "Filler2"],
        doc_idx_to_entities={
            0: {"person a"},
            1: {"person a", "person b"},
            2: {"person b", "target"},
            3: {"noise one"},
            4: {"noise two"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("person a", "person b", 1.0, "related_to")],
            2: [("person b", "target", 1.0, "related_to")],
            3: [],
            4: [],
        },
        adjacency={
            "person a": [("person b", 1.0, "related_to")],
            "person b": [("target", 1.0, "related_to")],
        },
        qa_top_k=4,
        initial_seed_entities={"person a"},
        query_entities={"target"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        score_mode="set_closure",
        beam_width=4,
        beam_expand_per_state=4,
    )

    assert selected_positions == [0, 1, 2]
    assert trace["selection_target_k"] == 4
    assert len(selected_positions) < trace["selection_target_k"]
    assert trace["beam_set_closure_guard_skip_count"] >= 1
    assert [step["doc_id"] for step in trace["selection_steps"]] == [0, 1, 2]


def test_compute_bridge_gate_decision_skips_without_strong_offrank_bridge_signal():
    decision = compute_bridge_gate_decision(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.84, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Suffix", "Noise"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"suffix"},
            5: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
        },
        adjacency={},
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        gate_mode="suffix_bridge",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.0,
    )

    assert decision["gate_enabled"] is True
    assert decision["use_selector"] is False
    assert decision["reason"] == "offrank_structure_below_threshold"
    assert decision["baseline_suffix_positions"] == [4]
    assert decision["best_offrank_pool_position"] == 5


def test_compute_bridge_gate_decision_applies_for_stronger_offrank_bridge_signal():
    decision = compute_bridge_gate_decision(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.84, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Suffix", "Bridge"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"suffix"},
            5: {"entity d", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [("entity d", "target", 1.0, "related_to")],
        },
        adjacency={
            "entity d": [("target", 1.0, "related_to")],
        },
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        gate_mode="suffix_bridge",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.0,
    )

    assert decision["gate_enabled"] is True
    assert decision["use_selector"] is True
    assert decision["reason"] == "offrank_bridge_signal_detected"
    assert decision["baseline_suffix_positions"] == [4]
    assert decision["best_offrank_pool_position"] == 5
    assert decision["weakest_baseline_suffix_pool_position"] == 4


def test_compute_bridge_gate_decision_skips_when_combined_margin_is_too_small():
    decision = compute_bridge_gate_decision(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.80, 0.79], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Suffix", "Bridge"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"suffix"},
            5: {"entity d", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [("entity d", "target", 1.0, "related_to")],
        },
        adjacency={
            "entity d": [("target", 1.0, "related_to")],
        },
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.9,
        structure_weight=0.05,
        novelty_weight=0.05,
        gate_mode="suffix_bridge",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.05,
    )

    assert decision["gate_enabled"] is True
    assert decision["use_selector"] is False
    assert decision["reason"] == "offrank_combined_margin_too_small"


def test_compute_bridge_gate_decision_precision_mode_applies_for_bridge_with_newinfo_signal():
    decision = compute_bridge_gate_decision(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.84, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Suffix", "Bridge"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"suffix"},
            5: {"entity d", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [("entity d", "target", 1.0, "related_to")],
        },
        adjacency={
            "entity d": [("target", 1.0, "related_to")],
        },
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        gate_mode="suffix_bridge_precision",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.0,
        gate_min_closure_score=0.2,
        gate_min_novelty_score=0.4,
        gate_min_frontier_gain=0.2,
        gate_min_path_coherence=0.2,
    )

    assert decision["gate_enabled"] is True
    assert decision["use_selector"] is True
    assert decision["reason"] == "offrank_bridge_signal_detected"
    assert decision["best_offrank_closure_score"] >= 0.2
    assert decision["best_offrank_novelty_score"] >= 0.4


def test_compute_bridge_gate_decision_precision_mode_skips_without_enough_newinfo_signal():
    decision = compute_bridge_gate_decision(
        pool_doc_ids=[0, 1, 2, 3, 4, 5],
        pool_doc_scores=np.array([1.0, 0.95, 0.90, 0.85, 0.84, 0.20], dtype=float),
        pool_doc_titles=["A", "B", "C", "D", "Suffix", "Bridge"],
        doc_idx_to_entities={
            0: {"entity a"},
            1: {"entity b"},
            2: {"entity c"},
            3: {"entity d"},
            4: {"suffix"},
            5: {"entity d", "target"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [("entity d", "target", 1.0, "related_to")],
        },
        adjacency={
            "entity d": [("target", 1.0, "related_to")],
        },
        qa_top_k=5,
        initial_seed_entities=set(),
        anchor_count=2,
        reserve_top_m=4,
        max_bridge_slots=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        gate_mode="suffix_bridge_precision",
        gate_min_structure_score=0.15,
        gate_min_combined_margin=0.0,
        gate_min_closure_score=0.2,
        gate_min_novelty_score=1.1,
        gate_min_frontier_gain=1.1,
        gate_min_path_coherence=1.1,
    )

    assert decision["gate_enabled"] is True
    assert decision["use_selector"] is False
    assert decision["reason"] == "offrank_novelty_below_threshold"
    assert decision["best_offrank_pool_position"] == 5
    assert decision["weakest_baseline_suffix_pool_position"] == 4


def test_maybe_apply_bridge_saturation_guard_reverts_saturated_weak_suffix_state():
    selected_positions, selector_trace, gate_decision = maybe_apply_bridge_saturation_guard(
        selected_positions=[0, 1, 2, 26, 57],
        selector_trace={
            "selection_steps": [
                {"step": 4, "mode": "beam", "pool_position": 26, "structure_score": 0.96},
                {"step": 5, "mode": "beam", "pool_position": 57, "structure_score": 1.0},
            ],
            "beam_best_state_suffix_base_mean": 0.12,
        },
        gate_decision={
            "gate_mode": "suffix_bridge_saturation_guard",
            "gate_enabled": True,
            "use_selector": True,
            "reason": "offrank_bridge_signal_detected",
        },
        pool_doc_titles=["A", "B", "C", "D", "E"] * 20,
        gate_max_avg_local_structure=0.95,
        gate_min_suffix_base_mean=0.15,
    )

    assert selected_positions == []
    assert selector_trace["saturation_guard_triggered"] is True
    assert gate_decision["use_selector"] is False
    assert gate_decision["reason"] == "structure_saturated_weak_suffix"
    assert gate_decision["avg_local_structure"] == 0.98
    assert gate_decision["suffix_base_mean"] == 0.12


def test_maybe_apply_bridge_saturation_guard_keeps_high_structure_when_suffix_not_weak():
    selected_positions, selector_trace, gate_decision = maybe_apply_bridge_saturation_guard(
        selected_positions=[0, 1, 2, 12, 4],
        selector_trace={
            "selection_steps": [
                {"step": 4, "mode": "beam", "pool_position": 12, "structure_score": 1.0},
                {"step": 5, "mode": "beam", "pool_position": 4, "structure_score": 0.8961},
            ],
            "beam_best_state_suffix_base_mean": 0.4142,
        },
        gate_decision={
            "gate_mode": "suffix_bridge_saturation_guard",
            "gate_enabled": True,
            "use_selector": True,
            "reason": "offrank_bridge_signal_detected",
        },
        pool_doc_titles=["A", "B", "C", "D", "E"] * 20,
        gate_max_avg_local_structure=0.95,
        gate_min_suffix_base_mean=0.15,
    )

    assert selected_positions == [0, 1, 2, 12, 4]
    assert selector_trace["saturation_guard_triggered"] is False
    assert gate_decision["use_selector"] is True
    assert gate_decision["reason"] == "offrank_bridge_signal_detected"
    assert gate_decision["avg_local_structure"] == 0.9481


def test_collect_lexical_query_seed_entities_matches_query_entity_strings():
    seeds = collect_lexical_query_seed_entities(
        query="When did Lothair II's mother die?",
        pool_doc_ids=[0, 1, 2],
        doc_idx_to_entities={
            0: {"donna summer"},
            1: {"lothair ii", "lotharingia"},
            2: {"ermengarde of tours"},
        },
    )

    assert "lothair ii" in seeds


def test_collect_question_query_entities_prefers_question_extractor_output():
    entities = collect_question_query_entities(
        hipporag=DummyHippoRAGForQueryEntities(entities=["Lothair II", "Ermengarde of Tours"]),
        query="When did Lothair II's mother die?",
        pool_doc_ids=[0, 1],
        doc_idx_to_entities={
            0: {"noise entity"},
            1: {"another noise entity"},
        },
    )

    assert entities == {"lothair ii", "ermengarde of tours"}


def test_collect_question_query_entities_falls_back_to_lexical_matching():
    entities = collect_question_query_entities(
        hipporag=DummyHippoRAGForQueryEntities(entities=[]),
        query="When did Lothair II's mother die?",
        pool_doc_ids=[0, 1, 2],
        doc_idx_to_entities={
            0: {"donna summer"},
            1: {"lothair ii", "lotharingia"},
            2: {"ermengarde of tours"},
        },
    )

    assert "lothair ii" in entities


def test_collect_grounded_question_query_entities_filters_offgraph_noise():
    grounded_entities = collect_grounded_question_query_entities(
        seed_entities={"seed a"},
        question_entities={"seed a", "bridge b", "target c", "offgraph noise"},
        pool_doc_ids=[0, 1, 2],
        doc_idx_to_entities={
            0: {"seed a"},
            1: {"bridge b"},
            2: {"target c"},
        },
        adjacency={
            "seed a": [("bridge b", 1.0, "related_to")],
            "bridge b": [("target c", 1.0, "related_to")],
        },
        structure_max_hops=2,
    )

    assert grounded_entities == {"seed a", "bridge b", "target c"}


def test_resolve_setwise_query_targets_hybrid_falls_back_to_seed_when_grounding_is_empty():
    query_targets = resolve_setwise_query_targets(
        query_entity_source="hybrid",
        seed_entities={"seed a"},
        question_entities={"offgraph noise"},
        pool_doc_ids=[0],
        doc_idx_to_entities={0: {"unrelated entity"}},
        adjacency={},
        structure_max_hops=2,
    )

    assert query_targets["proposal_query_entities"] == {"seed a"}
    assert query_targets["state_query_entities"] == {"seed a"}
    assert query_targets["state_support_query_entities"] == {"seed a"}
    assert query_targets["grounded_question_entities"] == set()


def test_select_bridge_beam_positions_can_use_seed_for_proposal_and_question_for_state():
    legacy_positions, legacy_trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_scores=np.array([1.0, 0.92, 0.35, 0.30], dtype=float),
        pool_doc_titles=["Anchor", "Helper", "Bridge", "Answer"],
        doc_idx_to_entities={
            0: {"seed a"},
            1: {"seed a", "helper h"},
            2: {"seed a", "bridge b"},
            3: {"bridge b", "target c"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("seed a", "helper h", 1.0, "related_to")],
            2: [("seed a", "bridge b", 1.0, "related_to")],
            3: [("bridge b", "target c", 1.0, "related_to")],
        },
        adjacency={
            "seed a": [("helper h", 1.0, "related_to"), ("bridge b", 1.0, "related_to")],
            "bridge b": [("target c", 1.0, "related_to")],
        },
        qa_top_k=3,
        initial_seed_entities={"seed a"},
        query_entities={"seed a"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        score_mode="set_closure",
        beam_width=3,
        beam_expand_per_state=3,
    )

    hybrid_positions, hybrid_trace = select_bridge_beam_positions(
        pool_doc_ids=[0, 1, 2, 3],
        pool_doc_scores=np.array([1.0, 0.92, 0.35, 0.30], dtype=float),
        pool_doc_titles=["Anchor", "Helper", "Bridge", "Answer"],
        doc_idx_to_entities={
            0: {"seed a"},
            1: {"seed a", "helper h"},
            2: {"seed a", "bridge b"},
            3: {"bridge b", "target c"},
        },
        doc_idx_to_edges={
            0: [],
            1: [("seed a", "helper h", 1.0, "related_to")],
            2: [("seed a", "bridge b", 1.0, "related_to")],
            3: [("bridge b", "target c", 1.0, "related_to")],
        },
        adjacency={
            "seed a": [("helper h", 1.0, "related_to"), ("bridge b", 1.0, "related_to")],
            "bridge b": [("target c", 1.0, "related_to")],
        },
        qa_top_k=3,
        initial_seed_entities={"seed a"},
        query_entities={"seed a"},
        proposal_query_entities={"seed a"},
        state_query_entities={"target c"},
        state_support_query_entities={"seed a"},
        anchor_count=1,
        structure_max_hops=2,
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        score_mode="set_closure",
        beam_width=3,
        beam_expand_per_state=3,
    )

    assert legacy_positions == [0, 1, 2]
    assert hybrid_positions == [0, 1, 3]
    assert hybrid_trace["beam_best_state_query_reachability"] == 1.0
    assert hybrid_trace["beam_best_state_query_reachability"] == legacy_trace["beam_best_state_query_reachability"]


def test_compute_candidate_feature_rows_exposes_bridge_structure_signal():
    feature_rows = compute_candidate_feature_rows(
        query="Who was born first, the director of film x or the director of film y?",
        pool_docs=[
            "Film X\nDirector A made film x.",
            "Film Y\nDirector B made film y.",
            "Director A\nDirector A was born in 1970.",
            "Director B\nDirector B was born in 1980.",
            "Noise\nThis is unrelated noise.",
        ],
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        doc_idx_to_entities={
            0: {"film x", "director a"},
            1: {"film y", "director b"},
            2: {"director a", "birth a"},
            3: {"director b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("director a", "birth a", 1.0, "related_to")],
            3: [("director b", "birth b", 1.0, "related_to")],
            4: [],
        },
        adjacency={
            "director a": [("birth a", 1.0, "related_to")],
            "director b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=5,
        selected_positions=[0, 1],
        seed_entities={"director a", "director b"},
        structure_max_hops=2,
        candidate_positions=[2, 3, 4],
    )

    rows_by_doc_id = {int(row["doc_id"]): row for row in feature_rows if row["doc_id"] is not None}
    assert rows_by_doc_id[2]["structure_score_seed"] > 0.0
    assert rows_by_doc_id[3]["structure_score_seed"] > 0.0
    assert rows_by_doc_id[4]["structure_score_seed"] == 0.0


def test_select_learned_greedy_positions_uses_model_scores_to_pick_bridge_docs():
    model_bundle = {"model": DummyReachabilityModel(), "feature_names": LEARNED_SETWISE_FEATURE_NAMES}
    selected_positions, trace = select_learned_greedy_positions(
        query="Who was born first, the director of film x or the director of film y?",
        pool_docs=[
            "Film X\nDirector A made film x.",
            "Film Y\nDirector B made film y.",
            "Director A\nDirector A was born in 1970.",
            "Director B\nDirector B was born in 1980.",
            "Noise\nThis is unrelated noise.",
        ],
        pool_doc_ids=[0, 1, 2, 3, 4],
        pool_doc_scores=np.array([0.95, 0.90, 0.30, 0.25, 0.80], dtype=float),
        doc_idx_to_entities={
            0: {"film x", "director a"},
            1: {"film y", "director b"},
            2: {"director a", "birth a"},
            3: {"director b", "birth b"},
            4: {"noise"},
        },
        doc_idx_to_edges={
            0: [],
            1: [],
            2: [("director a", "birth a", 1.0, "related_to")],
            3: [("director b", "birth b", 1.0, "related_to")],
            4: [],
        },
        adjacency={
            "director a": [("birth a", 1.0, "related_to")],
            "director b": [("birth b", 1.0, "related_to")],
        },
        qa_top_k=4,
        learned_model_bundle=model_bundle,
        initial_seed_entities={"director a", "director b"},
        anchor_count=2,
        structure_max_hops=2,
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["selection_steps"][2]["mode"] == "learned_greedy"
    assert trace["selection_steps"][2]["doc_id"] == 2
    assert trace["selection_steps"][3]["doc_id"] == 3


def test_requirement_cache_roundtrip_and_state_metrics_prefer_support_over_counterfactual(tmp_path):
    cache_entry = build_requirement_cache_entry(
        query_index=0,
        question="Which city is Person A from?",
        pool_docs=[
            "Person A\nPerson A is a musician.",
            "City X\nCity X is the hometown of Person A.",
            "City Y\nCity Y is associated with Person B.",
        ],
        pool_doc_entities=[
            {"person a"},
            {"person a", "city x"},
            {"person b", "city y"},
        ],
        seed_entities={"person a"},
        question_entities={"person a", "city x"},
        annotation_pool_k=3,
    )
    cache_path = tmp_path / "requirement_cache.json"
    save_requirement_cache(cache_path, {
        "version": "pcrs_rag_v1",
        "queries": [cache_entry],
    })
    loaded_cache = load_requirement_cache(cache_path)
    loaded_entry = loaded_cache["entries_by_question"]["Which city is Person A from?"]

    metrics_anchor_only = compute_requirement_state_metrics(
        cache_entry=loaded_entry,
        selected_positions=[0],
    )
    metrics_supported = compute_requirement_state_metrics(
        cache_entry=loaded_entry,
        selected_positions=[0, 1],
    )

    assert loaded_entry["question"] == "Which city is Person A from?"
    assert metrics_supported["support_completeness"] > metrics_anchor_only["support_completeness"]
    assert metrics_supported["selected_annotation_count"] == 2.0


def test_need_unit_cache_roundtrip_drops_wh_subjects_and_tracks_v2_groups(tmp_path):
    step_plan_payload = {
        "question_id": "q0",
        "answer_type": "city",
        "qdmr_steps": [
            {
                "step_id": "s1",
                "operation": "relation_lookup",
                "description": "Find the birth place of Person A.",
                "inputs": ["Person A"],
                "output_variable": "X",
            },
            {
                "step_id": "s2",
                "operation": "answer",
                "description": "Return X as the final answer.",
                "inputs": ["X"],
                "output_variable": "ANSWER",
            },
        ],
    }
    cache_entry = build_need_unit_cache_entry(
        query_index=0,
        question="Which city is Person A from?",
        pool_docs=[
            "Person A\nPerson A is from City X.",
            "City X\nCity X is the hometown of Person A.",
            "City Y\nCity Y is associated with Person B.",
        ],
        pool_doc_entities=[
            {"person a", "city x"},
            {"person a", "city x"},
            {"person b", "city y"},
        ],
        seed_entities={"person a"},
        question_entities={"person a", "city x"},
        annotation_pool_k=3,
        predicted_answer_type="city",
        step_plan_payload=step_plan_payload,
    )
    cache_path = tmp_path / "need_unit_cache.json"
    save_requirement_cache(cache_path, {
        "version": NEED_UNIT_CACHE_VERSION,
        "queries": [cache_entry],
    })
    loaded_cache = load_requirement_cache(cache_path)
    loaded_entry = loaded_cache["entries_by_question"]["Which city is Person A from?"]

    subjects = [str(unit.get("subject", "")).strip().lower() for unit in loaded_entry["positive_need_units"]]
    predicates = [str(unit.get("predicate", "")).strip() for unit in loaded_entry["positive_need_units"]]
    assert loaded_entry["version"] == NEED_UNIT_CACHE_VERSION
    assert "which" not in subjects
    assert loaded_entry["qdmr_steps"]
    assert "birth_place" in predicates

    metrics = compute_requirement_state_metrics(
        cache_entry=loaded_entry,
        selected_positions=[0, 1],
    )
    assert metrics["entity_locator_support"] >= 0.0
    assert metrics["answer_slot_support"] >= 0.0
    assert metrics["selected_annotation_count"] == 2.0


def test_build_heuristic_qdmr_step_plan_payload_extracts_birthplace_then_end_date():
    payload = build_heuristic_qdmr_step_plan_payload(
        question="When was Lady Godiva's birthplace abolished?",
        seed_entities={"lady godiva"},
        question_entities={"lady godiva"},
        predicted_answer_type="date",
    )

    descriptions = [str(step.get("description", "")) for step in payload["qdmr_steps"]]
    assert descriptions[0] == "Find the birthplace of lady godiva."
    assert descriptions[1] == "Find the end date of X."
    assert payload["qdmr_steps"][-1]["operation"] == "answer"


def test_normalize_answer_type_label_drops_lexical_suffix_noise():
    assert normalize_answer_type_label("month tripartite") == "month"
    assert normalize_answer_type_label("county erik") == "county"
    assert normalize_answer_type_label("person name") == "person_or_character"


def test_build_need_unit_cache_entry_preserves_parser_trace_across_fallback():
    cache_entry = build_need_unit_cache_entry(
        query_index=0,
        question="Which city is Person A from?",
        pool_docs=[
            "Person A\nPerson A is from City X.",
            "City X\nCity X is in Country Y.",
        ],
        pool_doc_entities=[
            {"person a", "city x"},
            {"city x", "country y"},
        ],
        seed_entities={"person a"},
        question_entities={"person a"},
        annotation_pool_k=2,
        predicted_answer_type="city",
        step_plan_payload={
            "question_id": "q0",
            "answer_type": "city",
            "qdmr_steps": [
                {
                    "step_id": "s1",
                    "operation": "relation_lookup",
                    "description": "Find the body water of Person A.",
                    "inputs": ["Person A"],
                    "output_variable": "X",
                },
                {
                    "step_id": "s2",
                    "operation": "answer",
                    "description": "Return X as the final answer.",
                    "inputs": ["X"],
                    "output_variable": "ANSWER",
                },
            ],
        },
        parser_trace={
            "raw_response_preview": "{\"answer_type\":\"city\"}",
            "metadata": {"finish_reason": "stop"},
        },
    )

    diagnostics = cache_entry["diagnostics"]
    assert diagnostics["parser_status"] == "fallback"
    assert diagnostics["compiler_status"] == "fallback"
    assert diagnostics["parser_raw_response_preview"] == "{\"answer_type\":\"city\"}"
    assert diagnostics["parser_metadata"]["finish_reason"] == "stop"
    assert diagnostics["pre_fallback_compiler_fallback_reasons"] == [
        "no_selector_enabled_relation_hop",
        "all_relation_hops_low_confidence",
    ]


def test_build_need_unit_cache_entry_supports_offline_hybrid_annotation_mode():
    atomic_bundle = {
        "model": DummyAtomicNeedUnitModel(),
        "feature_names": list(NEED_UNIT_ATOMIC_FEATURE_NAMES),
        "labels": list(NEED_UNIT_ATOMIC_LABELS),
        "bridge_alpha_by_unit_type": {"relation_hop": 0.65},
        "beta_contradiction": 1.0,
        "scorer_version": "need_unit_atomic_v1",
        "model_path": "/tmp/offline-hybrid-atomic.joblib",
    }
    cache_entry = build_need_unit_cache_entry(
        query_index=0,
        question="Where was Person A born?",
        pool_docs=[
            "Person A\nPerson A was born in City X.",
            "City X\nCity X is the birth place of Person A.",
        ],
        pool_doc_entities=[
            {"person a", "city x"},
            {"person a", "city x"},
        ],
        seed_entities={"person a"},
        question_entities={"person a"},
        annotation_pool_k=2,
        atomic_scorer_bundle=atomic_bundle,
        score_mode="hybrid",
    )

    assert cache_entry["diagnostics"]["annotation_score_mode"] == "hybrid"
    assert cache_entry["diagnostics"]["annotation_model_path"] == "/tmp/offline-hybrid-atomic.joblib"
    assert len(cache_entry["doc_annotations"]) == 2
    for row in cache_entry["doc_annotations"]:
        assert row["annotation_metadata"]["score_mode"] == "hybrid"
        assert row["annotation_metadata"]["model_path"] == "/tmp/offline-hybrid-atomic.joblib"


def test_compile_qdmr_to_need_units_relation_hop_cap_is_configurable():
    qdmr_steps = [
        {
            "step_id": "s1",
            "operation": "relation_lookup",
            "description": "Find the designer of Southeast Library.",
            "inputs": ["Southeast Library"],
            "output_variable": "X",
        },
        {
            "step_id": "s2",
            "operation": "relation_lookup",
            "description": "Find the death place of X.",
            "inputs": ["X"],
            "output_variable": "Y",
        },
        {
            "step_id": "s3",
            "operation": "relation_lookup",
            "description": "Find the body of water near Y that empties into the Gulf of Mexico.",
            "inputs": ["Y", "Gulf of Mexico"],
            "output_variable": "ANSWER",
        },
        {
            "step_id": "s4",
            "operation": "answer",
            "description": "Return ANSWER as the final answer.",
            "inputs": ["ANSWER"],
            "output_variable": "ANSWER",
        },
    ]

    compiled_cap2 = compile_qdmr_to_need_units(
        question="Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?",
        qdmr_steps=qdmr_steps,
        answer_type="location",
        question_entities={"southeast library"},
        seed_entities={"southeast library"},
        max_relation_hops=2,
    )
    compiled_cap3 = compile_qdmr_to_need_units(
        question="Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?",
        qdmr_steps=qdmr_steps,
        answer_type="location",
        question_entities={"southeast library"},
        seed_entities={"southeast library"},
        max_relation_hops=3,
    )

    relation_preds_cap2 = [
        unit["predicate"]
        for unit in compiled_cap2["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    ]
    relation_preds_cap3 = [
        unit["predicate"]
        for unit in compiled_cap3["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    ]

    assert relation_preds_cap2 == ["designer_of", "death_place"]
    assert relation_preds_cap3 == ["designer_of", "death_place", "empties_into"]
    assert compiled_cap2["diagnostics"]["relation_hop_cap"] == 2
    assert compiled_cap2["diagnostics"]["pre_cap_relation_hop_count"] == 3
    assert compiled_cap2["diagnostics"]["post_cap_relation_hop_count"] == 2
    assert "empties_into" in compiled_cap2["diagnostics"]["dropped_unit_predicates"]
    assert compiled_cap3["diagnostics"]["relation_hop_cap"] == 3
    assert compiled_cap3["diagnostics"]["post_cap_relation_hop_count"] == 3


def test_compile_qdmr_to_need_units_conditional_mode_keeps_third_hop_for_clean_chain():
    qdmr_steps = [
        {
            "step_id": "s1",
            "operation": "relation_lookup",
            "description": "Find the designer of Southeast Library.",
            "inputs": ["Southeast Library"],
            "output_variable": "X",
        },
        {
            "step_id": "s2",
            "operation": "relation_lookup",
            "description": "Find the death place of X.",
            "inputs": ["X"],
            "output_variable": "Y",
        },
        {
            "step_id": "s3",
            "operation": "relation_lookup",
            "description": "Find the body of water near Y that empties into the Gulf of Mexico.",
            "inputs": ["Y", "Gulf of Mexico"],
            "output_variable": "ANSWER",
        },
        {
            "step_id": "s4",
            "operation": "answer",
            "description": "Return ANSWER as the final answer.",
            "inputs": ["ANSWER"],
            "output_variable": "ANSWER",
        },
    ]

    compiled = compile_qdmr_to_need_units(
        question="Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?",
        qdmr_steps=qdmr_steps,
        answer_type="location",
        question_entities={"southeast library"},
        seed_entities={"southeast library"},
        max_relation_hops=2,
        relation_hop_cap_mode="conditional",
    )

    relation_preds = [
        unit["predicate"]
        for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    ]
    assert relation_preds == ["designer_of", "death_place", "empties_into"]
    assert compiled["diagnostics"]["conditional_third_hop_allowed"] is True
    assert compiled["diagnostics"]["conditional_third_hop_reasons"] == []


def test_compile_qdmr_to_need_units_conditional_mode_blocks_noisy_third_hop():
    qdmr_steps = [
        {
            "step_id": "s1",
            "operation": "relation_lookup",
            "description": "Find the designer of Southeast Library.",
            "inputs": ["Southeast Library"],
            "output_variable": "X",
        },
        {
            "step_id": "s2",
            "operation": "relation_lookup",
            "description": "Find the death place of X.",
            "inputs": ["X"],
            "output_variable": "Y",
        },
        {
            "step_id": "s3",
            "operation": "relation_lookup",
            "description": "Find the body water of Y.",
            "inputs": ["Y"],
            "output_variable": "ANSWER",
        },
        {
            "step_id": "s4",
            "operation": "answer",
            "description": "Return ANSWER as the final answer.",
            "inputs": ["ANSWER"],
            "output_variable": "ANSWER",
        },
    ]

    compiled = compile_qdmr_to_need_units(
        question="Where does the body of water by the city where the Southeast Library designer died go?",
        qdmr_steps=qdmr_steps,
        answer_type="location",
        question_entities={"southeast library"},
        seed_entities={"southeast library"},
        max_relation_hops=2,
        relation_hop_cap_mode="conditional",
    )

    relation_preds = [
        unit["predicate"]
        for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    ]
    assert relation_preds == ["designer_of", "death_place"]
    assert compiled["diagnostics"]["conditional_third_hop_allowed"] is False
    assert "third_relation_chain_not_high_confidence" in compiled["diagnostics"]["conditional_third_hop_reasons"]


def test_compile_qdmr_to_need_units_marks_low_confidence_lexical_predicates_disabled():
    compiled = compile_qdmr_to_need_units(
        question="Which city is Person A from?",
        qdmr_steps=[
            {
                "step_id": "s1",
                "operation": "relation_lookup",
                "description": "Find the body water of Person A.",
                "inputs": ["Person A"],
                "output_variable": "X",
            },
            {
                "step_id": "s2",
                "operation": "answer",
                "description": "Return X as the final answer.",
                "inputs": ["X"],
                "output_variable": "ANSWER",
            },
        ],
        answer_type="city",
        question_entities={"person a"},
        seed_entities={"person a"},
    )

    relation_units = [
        unit for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    ]
    assert relation_units
    assert relation_units[0]["confidence"] == "low"
    assert relation_units[0]["selector_enabled"] is False
    assert compiled["diagnostics"]["low_confidence_unit_count"] >= 1


def test_compile_qdmr_to_need_units_keeps_county_relation_enabled():
    compiled = compile_qdmr_to_need_units(
        question="What county is Erik Hort's birthplace a part of?",
        qdmr_steps=[
            {
                "step_id": "s1",
                "operation": "relation_lookup",
                "description": "Find the birthplace of Erik Hort.",
                "inputs": ["Erik Hort"],
                "output_variable": "X",
            },
            {
                "step_id": "s2",
                "operation": "relation_lookup",
                "description": "Find the county of X.",
                "inputs": ["X"],
                "output_variable": "ANSWER",
            },
            {
                "step_id": "s3",
                "operation": "answer",
                "description": "Return ANSWER as the final answer.",
                "inputs": ["ANSWER"],
                "output_variable": "ANSWER",
            },
        ],
        answer_type="county",
        question_entities={"erik hort"},
        seed_entities={"erik hort"},
    )

    relation_units = [
        unit for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    ]
    assert [unit["predicate"] for unit in relation_units] == ["birth_place", "county"]
    assert relation_units[1]["confidence"] == "high"
    assert relation_units[1]["selector_enabled"] is True


def test_compile_qdmr_to_need_units_maps_end_year_to_end_date():
    compiled = compile_qdmr_to_need_units(
        question="What year did the publisher of Labyrinth end?",
        qdmr_steps=[
            {
                "step_id": "s1",
                "operation": "relation_lookup",
                "description": "Find the publisher of Labyrinth.",
                "inputs": ["Labyrinth"],
                "output_variable": "X",
            },
            {
                "step_id": "s2",
                "operation": "relation_lookup",
                "description": "Find the end year of X.",
                "inputs": ["X"],
                "output_variable": "ANSWER",
            },
            {
                "step_id": "s3",
                "operation": "answer",
                "description": "Return ANSWER as the final answer.",
                "inputs": ["ANSWER"],
                "output_variable": "ANSWER",
            },
        ],
        answer_type="date",
        question_entities={"labyrinth"},
        seed_entities={"labyrinth"},
    )

    relation_units = [
        unit for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    ]
    assert [unit["predicate"] for unit in relation_units] == ["publisher", "end_date"]
    assert relation_units[1]["selector_enabled"] is True


def test_compile_qdmr_to_need_units_maps_city_where_died_to_death_place():
    compiled = compile_qdmr_to_need_units(
        question="Where is the city where Person A died?",
        qdmr_steps=[
            {
                "step_id": "s1",
                "operation": "relation_lookup",
                "description": "Find the city where X died.",
                "inputs": ["X"],
                "output_variable": "ANSWER",
            },
            {
                "step_id": "s2",
                "operation": "answer",
                "description": "Return ANSWER as the final answer.",
                "inputs": ["ANSWER"],
                "output_variable": "ANSWER",
            },
        ],
        answer_type="location",
        question_entities={"person a"},
        seed_entities={"person a"},
    )

    relation_unit = next(
        unit for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    )
    assert relation_unit["predicate"] == "death_place"
    assert relation_unit["selector_enabled"] is True


def test_compile_qdmr_to_need_units_maps_north_of_to_canonical_relation():
    compiled = compile_qdmr_to_need_units(
        question="When was the region immediately north of Israel created?",
        qdmr_steps=[
            {
                "step_id": "s1",
                "operation": "relation_lookup",
                "description": "Find the region immediately north of X.",
                "inputs": ["X"],
                "output_variable": "Y",
            },
            {
                "step_id": "s2",
                "operation": "answer",
                "description": "Return Y as the final answer.",
                "inputs": ["Y"],
                "output_variable": "ANSWER",
            },
        ],
        answer_type="date",
        question_entities={"israel"},
        seed_entities={"israel"},
    )

    relation_unit = next(
        unit for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    )
    assert relation_unit["predicate"] == "north_of"
    assert relation_unit["selector_enabled"] is True


def test_compile_qdmr_to_need_units_generates_role_and_temporal_counterfactuals():
    compiled = compile_qdmr_to_need_units(
        question="Which film directed by Christopher Nolan was released before 2010?",
        qdmr_steps=[
            {
                "step_id": "s1",
                "operation": "relation_lookup",
                "description": "Find films directed by Christopher Nolan.",
                "inputs": ["Christopher Nolan"],
                "output_variable": "X",
            },
            {
                "step_id": "s2",
                "operation": "constraint_check",
                "description": "Keep the film X whose release date is before 2010.",
                "inputs": ["X"],
                "output_variable": "X",
            },
            {
                "step_id": "s3",
                "operation": "answer",
                "description": "Return X as the final answer.",
                "inputs": ["X"],
                "output_variable": "ANSWER",
            },
        ],
        answer_type="film",
        question_entities={"christopher nolan"},
        seed_entities={"christopher nolan"},
    )

    transforms = {str(item.get("transform", "")) for item in compiled["counterfactual_sets"]}
    assert "role_swap" in transforms
    assert "temporal_shift" in transforms or "constraint_flip" in transforms


def test_compile_qdmr_to_need_units_preserves_variable_slots_in_positive_units():
    compiled = compile_qdmr_to_need_units(
        question="Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?",
        qdmr_steps=[
            {
                "step_id": "s1",
                "operation": "relation_lookup",
                "description": "Find the designer of Southeast Library.",
                "inputs": ["Southeast Library"],
                "output_variable": "X",
            },
            {
                "step_id": "s2",
                "operation": "relation_lookup",
                "description": "Find the death place of X.",
                "inputs": ["X"],
                "output_variable": "Y",
            },
            {
                "step_id": "s3",
                "operation": "relation_lookup",
                "description": "Find the body of water near Y that empties into the Gulf of Mexico.",
                "inputs": ["Y", "Gulf of Mexico"],
                "output_variable": "ANSWER",
            },
            {
                "step_id": "s4",
                "operation": "answer",
                "description": "Return ANSWER as the final answer.",
                "inputs": ["ANSWER"],
                "output_variable": "ANSWER",
            },
        ],
        answer_type="location",
        question_entities={"southeast library"},
        seed_entities={"southeast library"},
        max_relation_hops=3,
        relation_hop_cap_mode="conditional",
    )

    relation_units = [
        unit for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "relation_hop"
    ]
    assert [unit["subject"] for unit in relation_units] == [
        "southeast library",
        "?x",
        "?y",
    ]
    assert [unit["object"] for unit in relation_units] == ["?x", "?y", "?ans"]
    assert [unit["target_variable"] for unit in relation_units] == ["?x", "?y", "?ans"]
    answer_unit = next(
        unit for unit in compiled["positive_need_units"]
        if unit.get("unit_type") == "answer_slot"
    )
    assert answer_unit["subject"] == "?ans"
    assert answer_unit["object"] == "?ans"
    assert answer_unit["target_variable"] == "?ans"


def test_build_need_unit_preserves_canonical_variable_slots():
    unit = build_need_unit(
        unit_id="u0",
        unit_type="relation_hop",
        subject="?x",
        predicate="death_place",
        object_value="ANSWER",
        target_variable="Y",
    )

    assert unit["subject"] == "?x"
    assert unit["object"] == "?ans"
    assert unit["target_variable"] == "?y"


def test_extract_need_unit_atomic_features_recovers_legacy_serialized_variables():
    feature_row = extract_need_unit_atomic_features(
        unit={
            "unit_id": "u0",
            "unit_type": "relation_hop",
            "subject": "x",
            "predicate": "death_place",
            "object": "Minneapolis",
            "constraints": [],
        },
        doc_title="Minneapolis",
        doc_body="The death place is Minneapolis.",
        doc_entities={"minneapolis"},
    )

    assert feature_row["subject_is_variable"] == 1.0
    assert feature_row["object_is_variable"] == 0.0
    assert feature_row["alias_or_variable_bridge_alignment"] >= 1.0


def test_compute_requirement_state_metrics_ignores_selector_disabled_need_units():
    cache_entry = {
        "version": NEED_UNIT_CACHE_VERSION,
        "annotation_pool_k": 2,
        "positive_need_units": [
            {"unit_id": "u0", "unit_type": "relation_hop", "selector_enabled": False},
            {"unit_id": "u1", "unit_type": "answer_slot", "selector_enabled": True},
        ],
        "counterfactual_sets": [],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Distractor",
                "positive_need_unit_scores": {"u0": 1.0, "u1": 0.0},
                "positive_requirement_scores": {"u0": 1.0, "u1": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Answer",
                "positive_need_unit_scores": {"u0": 0.0, "u1": 1.0},
                "positive_requirement_scores": {"u0": 0.0, "u1": 1.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    metrics_distractor = compute_requirement_state_metrics(
        cache_entry=cache_entry,
        selected_positions=[0],
    )
    metrics_answer = compute_requirement_state_metrics(
        cache_entry=cache_entry,
        selected_positions=[1],
    )

    assert metrics_distractor["active_group_count"] == 1.0
    assert metrics_distractor["support_completeness"] == 0.0
    assert metrics_answer["support_completeness"] == 1.0


def test_score_need_unit_support_emits_atomic_fields_for_direct_support():
    unit = {
        "unit_id": "u0",
        "unit_type": "relation_hop",
        "subject": "southeast library",
        "predicate": "designer_of",
        "object": "?x",
        "constraints": [],
    }

    scores = score_need_unit_support(
        unit=unit,
        doc_title="Southeast Library",
        doc_body="The designer of Southeast Library is John Smith.",
        doc_entities={"southeast library", "john smith"},
    )

    assert "full_support_prob" in scores
    assert "bridge_support_prob" in scores
    assert scores["full_support_prob"] > scores["bridge_support_prob"]
    assert scores["coverage_score"] > 0.2
    assert scores["score_source"] == "heuristic_fallback"


def test_score_need_unit_support_emits_contradiction_for_opposing_predicate():
    unit = {
        "unit_id": "u0",
        "unit_type": "relation_hop",
        "subject": "the good shepherd",
        "predicate": "directed_by",
        "object": "?x",
        "constraints": [],
    }

    scores = score_need_unit_support(
        unit=unit,
        doc_title="The Good Shepherd",
        doc_body="The Good Shepherd starred in Matt Damon.",
        doc_entities={"the good shepherd", "matt damon"},
    )

    assert scores["contradiction_prob"] >= 0.4
    assert scores["coverage_score"] == 0.0


def test_build_need_unit_doc_annotation_hybrid_adds_atomic_maps_and_metadata():
    unit = {
        "unit_id": "u0",
        "unit_type": "relation_hop",
        "subject": "person a",
        "predicate": "birth_place",
        "object": "city x",
        "constraints": [],
    }
    bundle = {
        "model": DummyAtomicNeedUnitModel(),
        "feature_names": list(NEED_UNIT_ATOMIC_FEATURE_NAMES),
        "labels": list(NEED_UNIT_ATOMIC_LABELS),
        "bridge_alpha_by_unit_type": {
            "relation_hop": 0.65,
        },
        "beta_contradiction": 1.0,
        "scorer_version": "need_unit_atomic_v1",
        "model_path": "/tmp/dummy-atomic.joblib",
    }

    annotation = build_need_unit_doc_annotation(
        pool_position=0,
        doc_text="Person A\nPerson A was from City X.",
        doc_entities={"person a", "city x"},
        positive_need_units=[unit],
        counterfactual_sets=[],
        atomic_scorer_bundle=bundle,
        score_mode="hybrid",
    )

    atomic_scores = annotation["positive_atomic_scores"]["u0"]
    assert annotation["annotation_metadata"]["score_mode"] == "hybrid"
    assert annotation["annotation_metadata"]["model_path"] == "/tmp/dummy-atomic.joblib"
    assert atomic_scores["bridge_support_prob"] == 0.7
    assert annotation["positive_support_probs"]["u0"] == pytest.approx(0.555, abs=1e-6)


def test_resolve_atomic_annotation_bundle_requires_model_path_for_hybrid():
    with pytest.raises(ValueError):
        resolve_atomic_annotation_bundle(score_mode="hybrid", atomic_model_path="")


def test_load_need_unit_atomic_model_bundle_roundtrip(tmp_path):
    bundle_path = tmp_path / "atomic_bundle.joblib"
    import joblib

    joblib.dump({
        "task": "atomic_multiclass",
        "model": DummyAtomicNeedUnitModel(),
        "feature_names": list(NEED_UNIT_ATOMIC_FEATURE_NAMES),
    }, bundle_path)

    loaded = load_need_unit_atomic_model_bundle(bundle_path)

    assert loaded["task"] == "atomic_multiclass"
    assert loaded["feature_names"] == list(NEED_UNIT_ATOMIC_FEATURE_NAMES)
    assert loaded["labels"] == list(NEED_UNIT_ATOMIC_LABELS)


def test_build_need_unit_atomic_training_rows_prefers_label_override_and_schema():
    doc_text = "Person A\nThe birth place of Person A is City X."
    unit = {
        "unit_id": "u0",
        "unit_type": "relation_hop",
        "subject": "person a",
        "predicate": "birth_place",
        "object": "city x",
        "constraints": [],
        "selector_enabled": True,
    }
    cache_entry = {
        "version": NEED_UNIT_CACHE_VERSION,
        "query_index": 0,
        "question": "Which city is Person A from?",
        "question_key": "q0",
        "annotation_pool_k": 1,
        "positive_need_units": [unit],
        "counterfactual_sets": [],
        "doc_annotations": [
            build_need_unit_doc_annotation(
                pool_position=0,
                doc_text=doc_text,
                doc_entities={"person a", "city x"},
                positive_need_units=[unit],
                counterfactual_sets=[],
            ),
        ],
    }
    cache_payload = {
        "version": NEED_UNIT_CACHE_VERSION,
        "queries": [cache_entry],
        "entries_by_question": {cache_entry["question"]: cache_entry},
        "entries_by_index": {0: cache_entry},
    }
    query_solution = QuerySolution(
        question=cache_entry["question"],
        docs=[doc_text],
        gold_docs=[doc_text],
    )

    rows, summary = build_need_unit_atomic_training_rows(
        query_solutions=[query_solution],
        gold_docs=[[doc_text]],
        requirement_cache=cache_payload,
        doc_text_to_entities={doc_text: ["person a", "city x"]},
        label_overrides={"q0:p0:u0": "bridge_support"},
        include_counterfactual_samples=False,
    )

    assert summary["row_count"] == 1
    assert rows[0]["sample_id"] == "q0:p0:u0"
    assert rows[0]["llm_label"] == "bridge_support"
    assert rows[0]["final_label"] == "bridge_support"
    assert rows[0]["label_source"] == "llm"
    assert "features" in rows[0]
    assert rows[0]["features"]["predicate_alignment"] > 0.0


def test_need_unit_feature_matrix_uses_v2_feature_order():
    feature_rows = [{
        "support_completeness_gain": 0.4,
        "relation_hop_support_after": 0.7,
        "doc_contradiction_mean": 0.2,
    }]
    matrix = requirement_feature_rows_to_matrix(
        feature_rows,
        feature_names=NEED_UNIT_MATCHER_FEATURE_NAMES,
    )

    assert matrix.shape == (1, len(NEED_UNIT_MATCHER_FEATURE_NAMES))
    assert matrix[0, NEED_UNIT_MATCHER_FEATURE_NAMES.index("relation_hop_support_after")] == 0.7
    assert matrix[0, NEED_UNIT_MATCHER_FEATURE_NAMES.index("doc_contradiction_mean")] == 0.2


def test_select_requirement_beam_positions_prefers_low_leakage_chain():
    cache_entry = {
        "annotation_pool_k": 4,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
            {"requirement_id": "bridge_0", "type": "bridge"},
            {"requirement_id": "decision_0", "type": "decision"},
        ],
        "counterfactual_sets": [
            {
                "cf_id": "cf_0",
                "requirements": [
                    {"requirement_id": "anchor_0"},
                    {"requirement_id": "bridge_0"},
                    {"requirement_id": "decision_0"},
                ],
            }
        ],
        "pool_titles": ["A", "Bridge", "Leak", "Answer"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "A",
                "positive_requirement_scores": {"anchor_0": 1.0, "bridge_0": 0.0, "decision_0": 0.0},
                "counterfactual_requirement_scores": {"cf_0": {"anchor_0": 0.0, "bridge_0": 0.0, "decision_0": 0.0}},
                "counterfactual_set_scores": {"cf_0": 0.0},
            },
            {
                "pool_position": 1,
                "doc_title": "Bridge",
                "positive_requirement_scores": {"anchor_0": 0.1, "bridge_0": 1.0, "decision_0": 0.2},
                "counterfactual_requirement_scores": {"cf_0": {"anchor_0": 0.0, "bridge_0": 0.0, "decision_0": 0.0}},
                "counterfactual_set_scores": {"cf_0": 0.0},
            },
            {
                "pool_position": 2,
                "doc_title": "Leak",
                "positive_requirement_scores": {"anchor_0": 0.1, "bridge_0": 0.7, "decision_0": 0.1},
                "counterfactual_requirement_scores": {"cf_0": {"anchor_0": 0.9, "bridge_0": 1.0, "decision_0": 0.9}},
                "counterfactual_set_scores": {"cf_0": 0.933333},
            },
            {
                "pool_position": 3,
                "doc_title": "Answer",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.2, "decision_0": 1.0},
                "counterfactual_requirement_scores": {"cf_0": {"anchor_0": 0.0, "bridge_0": 0.0, "decision_0": 0.0}},
                "counterfactual_set_scores": {"cf_0": 0.0},
            },
        ],
    }

    selected_positions, trace = select_requirement_beam_positions(
        pool_doc_ids=[10, 11, 12, 13],
        pool_doc_scores=np.array([0.9, 0.6, 0.8, 0.4], dtype=float),
        pool_doc_titles=["A", "Bridge", "Leak", "Answer"],
        doc_idx_to_entities={
            10: {"person a"},
            11: {"bridge"},
            12: {"fake bridge"},
            13: {"target"},
        },
        doc_idx_to_edges={
            10: [],
            11: [],
            12: [],
            13: [],
        },
        adjacency={
            "person a": [("bridge", 1.0, "related_to"), ("fake bridge", 1.0, "related_to")],
            "bridge": [("target", 1.0, "related_to")],
            "fake bridge": [("target", 1.0, "related_to")],
        },
        qa_top_k=3,
        cache_entry=cache_entry,
        initial_seed_entities={"person a"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=2,
        beam_width=3,
        beam_expand_per_state=3,
        non_anchor_title_dedup=True,
    )

    assert selected_positions == [0, 1, 3]
    assert trace["beam_best_support_completeness"] > 0.5
    assert trace["beam_best_counterfactual_leakage"] < 0.5
    assert trace["beam_finalists"][0]["selected_positions"] == [0, 1, 3]


def test_select_requirement_beam_positions_supports_v2_need_unit_cache_entries():
    cache_entry = {
        "version": NEED_UNIT_CACHE_VERSION,
        "annotation_pool_k": 4,
        "positive_need_units": [
            {"unit_id": "u0", "unit_type": "entity_locator"},
            {"unit_id": "u1", "unit_type": "relation_hop"},
            {"unit_id": "u2", "unit_type": "answer_slot"},
        ],
        "counterfactual_sets": [
            {
                "cf_id": "cf_0",
                "requirements": [
                    {"unit_id": "u0", "unit_type": "entity_locator"},
                    {"unit_id": "u1_cf", "unit_type": "relation_hop"},
                    {"unit_id": "u2", "unit_type": "answer_slot"},
                ],
            }
        ],
        "pool_titles": ["A", "Bridge", "Leak", "Answer"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "A",
                "positive_need_unit_scores": {"u0": 1.0, "u1": 0.0, "u2": 0.0},
                "positive_requirement_scores": {"u0": 1.0, "u1": 0.0, "u2": 0.0},
                "counterfactual_requirement_scores": {"cf_0": {"u0": 0.0, "u1_cf": 0.0, "u2": 0.0}},
                "counterfactual_set_scores": {"cf_0": 0.0},
            },
            {
                "pool_position": 1,
                "doc_title": "Bridge",
                "positive_need_unit_scores": {"u0": 0.1, "u1": 1.0, "u2": 0.2},
                "positive_requirement_scores": {"u0": 0.1, "u1": 1.0, "u2": 0.2},
                "counterfactual_requirement_scores": {"cf_0": {"u0": 0.0, "u1_cf": 0.0, "u2": 0.0}},
                "counterfactual_set_scores": {"cf_0": 0.0},
            },
            {
                "pool_position": 2,
                "doc_title": "Leak",
                "positive_need_unit_scores": {"u0": 0.1, "u1": 0.7, "u2": 0.1},
                "positive_requirement_scores": {"u0": 0.1, "u1": 0.7, "u2": 0.1},
                "counterfactual_requirement_scores": {"cf_0": {"u0": 0.9, "u1_cf": 1.0, "u2": 0.9}},
                "counterfactual_set_scores": {"cf_0": 0.933333},
            },
            {
                "pool_position": 3,
                "doc_title": "Answer",
                "positive_need_unit_scores": {"u0": 0.0, "u1": 0.2, "u2": 1.0},
                "positive_requirement_scores": {"u0": 0.0, "u1": 0.2, "u2": 1.0},
                "counterfactual_requirement_scores": {"cf_0": {"u0": 0.0, "u1_cf": 0.0, "u2": 0.0}},
                "counterfactual_set_scores": {"cf_0": 0.0},
            },
        ],
    }

    selected_positions, trace = select_requirement_beam_positions(
        pool_doc_ids=[10, 11, 12, 13],
        pool_doc_scores=np.array([0.9, 0.6, 0.8, 0.4], dtype=float),
        pool_doc_titles=["A", "Bridge", "Leak", "Answer"],
        doc_idx_to_entities={
            10: {"person a"},
            11: {"bridge"},
            12: {"fake bridge"},
            13: {"target"},
        },
        doc_idx_to_edges={
            10: [],
            11: [],
            12: [],
            13: [],
        },
        adjacency={
            "person a": [("bridge", 1.0, "related_to"), ("fake bridge", 1.0, "related_to")],
            "bridge": [("target", 1.0, "related_to")],
            "fake bridge": [("target", 1.0, "related_to")],
        },
        qa_top_k=3,
        cache_entry=cache_entry,
        initial_seed_entities={"person a"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=2,
        beam_width=3,
        beam_expand_per_state=3,
        non_anchor_title_dedup=True,
    )

    assert selected_positions == [0, 1, 3]
    assert trace["requirement_positive_count"] == 3
    assert trace["beam_best_support_completeness"] > 0.5


def test_requirement_beam_dedupes_permuted_doc_sets_by_canonical_signature():
    cache_entry = {
        "annotation_pool_k": 2,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
        ],
        "counterfactual_sets": [],
        "pool_titles": ["Doc A", "Doc B"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Doc A",
                "positive_requirement_scores": {"anchor_0": 1.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Doc B",
                "positive_requirement_scores": {"anchor_0": 1.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    selected_positions, trace = select_requirement_beam_positions(
        pool_doc_ids=[20, 21],
        pool_doc_scores=np.array([0.9, 0.8], dtype=float),
        pool_doc_titles=["Doc A", "Doc B"],
        doc_idx_to_entities={
            20: {"doc a"},
            21: {"doc b"},
        },
        doc_idx_to_edges={
            20: [],
            21: [],
        },
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"seed"},
        proposal_query_entities={"target"},
        anchor_count=0,
        reserve_top_m=0,
        structure_max_hops=1,
        beam_width=2,
        beam_expand_per_state=2,
        non_anchor_title_dedup=False,
    )

    assert selected_positions == [0, 1]
    assert trace["beam_signature_pruned_count"] == 1
    assert len(trace["beam_finalists"]) == 1
    assert trace["beam_finalists"][0]["selected_positions"] == [0, 1]


def test_requirement_beam_learned_mode_reorders_widened_shortlist_before_expansion():
    cache_entry = {
        "annotation_pool_k": 3,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
            {"requirement_id": "bridge_0", "type": "bridge"},
            {"requirement_id": "decision_0", "type": "decision"},
        ],
        "counterfactual_sets": [],
        "pool_titles": ["Anchor", "Bridge Prior", "Bridge Support"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Anchor",
                "positive_requirement_scores": {"anchor_0": 1.0, "bridge_0": 0.0, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Bridge Prior",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.15, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 2,
                "doc_title": "Bridge Support",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 1.0, "decision_0": 1.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    oracle_positions, oracle_trace = select_requirement_beam_positions(
        pool_doc_ids=[30, 31, 32],
        pool_doc_scores=np.array([0.95, 0.80, 0.20], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Bridge Support"],
        doc_idx_to_entities={
            30: {"anchor"},
            31: {"bridge prior"},
            32: {"bridge support"},
        },
        doc_idx_to_edges={
            30: [],
            31: [],
            32: [],
        },
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )
    learned_positions, learned_trace = select_requirement_beam_positions(
        pool_doc_ids=[30, 31, 32],
        pool_doc_scores=np.array([0.95, 0.80, 0.20], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Bridge Support"],
        doc_idx_to_entities={
            30: {"anchor"},
            31: {"bridge prior"},
            32: {"bridge support"},
        },
        doc_idx_to_edges={
            30: [],
            31: [],
            32: [],
        },
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        non_anchor_title_dedup=False,
        requirement_mode="learned",
        requirement_model_bundle={"model": DummyRequirementPriorModel()},
    )

    assert oracle_positions == [0, 2]
    assert learned_positions == [0, 2]
    assert oracle_positions == learned_positions
    assert oracle_trace["selection_steps"][1]["proposal_rank"] == 2
    assert learned_trace["selection_steps"][1]["proposal_rank"] == 1
    assert learned_trace["beam_learned_eval_count"] == 2
    assert learned_trace["selection_steps"][1]["predicted_utility"] > 0.0
    assert [row["pool_position"] for row in oracle_trace["selection_steps"][1]["candidate_source_preview"]] == [1, 2]
    assert [row["pool_position"] for row in oracle_trace["selection_steps"][1]["candidate_shortlist_preview"]] == [1, 2]
    assert [row["pool_position"] for row in learned_trace["selection_steps"][1]["candidate_source_preview"]] == [1, 2]
    assert [row["pool_position"] for row in learned_trace["selection_steps"][1]["candidate_shortlist_preview"]] == [2]
    assert learned_trace["selection_steps"][1]["candidate_shortlist_preview"][0]["predicted_utility"] > 0.0


def test_requirement_beam_oracle_widened_shortlist_allows_state_rescue():
    cache_entry = {
        "annotation_pool_k": 3,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
            {"requirement_id": "bridge_0", "type": "bridge"},
            {"requirement_id": "decision_0", "type": "decision"},
        ],
        "counterfactual_sets": [],
        "pool_titles": ["Anchor", "Bridge Prior", "Bridge Rescue"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Anchor",
                "positive_requirement_scores": {"anchor_0": 1.0, "bridge_0": 0.0, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Bridge Prior",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.1, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 2,
                "doc_title": "Bridge Rescue",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 1.0, "decision_0": 1.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    legacy_positions, legacy_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.90, 0.05], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Bridge Rescue"],
        doc_idx_to_entities={
            40: {"anchor"},
            41: {"bridge prior"},
            42: {"bridge rescue"},
        },
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=1,
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )
    projected_positions, projected_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.90, 0.05], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Bridge Rescue"],
        doc_idx_to_entities={
            40: {"anchor"},
            41: {"bridge prior"},
            42: {"bridge rescue"},
        },
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=3,
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )

    assert legacy_positions == [0, 1]
    assert projected_positions == [0, 2]
    assert legacy_trace["selection_steps"][1]["proposal_rank"] == 1
    assert projected_trace["selection_steps"][1]["proposal_rank"] == 2
    assert [row["pool_position"] for row in legacy_trace["selection_steps"][1]["candidate_shortlist_preview"]] == [1]
    assert [row["pool_position"] for row in projected_trace["selection_steps"][1]["candidate_shortlist_preview"]] == [1, 2]


def test_requirement_beam_oracle_source_widening_can_keep_shortlist_small():
    cache_entry = {
        "annotation_pool_k": 3,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
            {"requirement_id": "bridge_0", "type": "bridge"},
            {"requirement_id": "decision_0", "type": "decision"},
        ],
        "counterfactual_sets": [],
        "pool_titles": ["Anchor", "Bridge Prior", "Bridge Rescue"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Anchor",
                "positive_requirement_scores": {"anchor_0": 1.0, "bridge_0": 0.0, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Bridge Prior",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.1, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 2,
                "doc_title": "Bridge Rescue",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 1.0, "decision_0": 1.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    widened_positions, widened_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.90, 0.05], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Bridge Rescue"],
        doc_idx_to_entities={
            40: {"anchor"},
            41: {"bridge prior"},
            42: {"bridge rescue"},
        },
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=3,
        beam_candidate_shortlist_limit=1,
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )

    assert widened_positions == [0, 2]
    assert widened_trace["beam_candidate_shortlist_limit"] == 1
    assert [row["pool_position"] for row in widened_trace["selection_steps"][1]["candidate_source_preview"]] == [1, 2]
    assert [row["pool_position"] for row in widened_trace["selection_steps"][1]["candidate_shortlist_preview"]] == [2]
    assert widened_trace["selection_steps"][1]["proposal_rank"] == 1


def test_requirement_beam_probe_force_source_injects_watch_title_without_forcing_shortlist():
    cache_entry = {
        "annotation_pool_k": 3,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
            {"requirement_id": "bridge_0", "type": "bridge"},
            {"requirement_id": "decision_0", "type": "decision"},
        ],
        "counterfactual_sets": [],
        "pool_titles": ["Anchor", "Bridge Prior", "Deep Bridge"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Anchor",
                "positive_requirement_scores": {"anchor_0": 1.0, "bridge_0": 0.0, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Bridge Prior",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.1, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 2,
                "doc_title": "Deep Bridge",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.9, "decision_0": 0.2},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    _, trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.90, 0.05], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Deep Bridge"],
        doc_idx_to_entities={40: {"anchor"}, 41: {"bridge prior"}, 42: {"deep bridge"}},
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=1,
        beam_candidate_shortlist_limit=1,
        force_source_titles=["Deep Bridge"],
        trace_watch_titles=["Deep Bridge"],
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )

    step = trace["selection_steps"][1]
    assert [row["pool_position"] for row in step["candidate_source_preview"]] == [1, 2]
    assert [row["pool_position"] for row in step["candidate_shortlist_preview"]] == [2]
    assert step["forced_source_titles_applied"] == ["Deep Bridge"]
    assert step["forced_shortlist_titles_applied"] == []
    watch_row = step["watch_title_trace"][0]
    assert watch_row["title"] == "Deep Bridge"
    assert watch_row["forced_into_source"] is True
    assert watch_row["forced_into_shortlist"] is False


def test_requirement_beam_probe_force_shortlist_promotes_watch_title_from_source():
    cache_entry = {
        "annotation_pool_k": 3,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
            {"requirement_id": "bridge_0", "type": "bridge"},
            {"requirement_id": "decision_0", "type": "decision"},
        ],
        "counterfactual_sets": [],
        "pool_titles": ["Anchor", "Bridge Prior", "Deep Bridge"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Anchor",
                "positive_requirement_scores": {"anchor_0": 1.0, "bridge_0": 0.0, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Bridge Prior",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.1, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 2,
                "doc_title": "Deep Bridge",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.02, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    positions, trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.90, 0.05], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Deep Bridge"],
        doc_idx_to_entities={40: {"anchor"}, 41: {"bridge prior"}, 42: {"deep bridge"}},
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=3,
        beam_candidate_shortlist_limit=1,
        force_shortlist_titles=["Deep Bridge"],
        trace_watch_titles=["Deep Bridge"],
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )

    step = trace["selection_steps"][1]
    assert [row["pool_position"] for row in step["candidate_source_preview"]] == [1, 2]
    assert [row["pool_position"] for row in step["candidate_shortlist_preview"]] == [1, 2]
    assert step["forced_source_titles_applied"] == []
    assert step["forced_shortlist_titles_applied"] == ["Deep Bridge"]
    watch_row = step["watch_title_trace"][0]
    assert watch_row["forced_into_source"] is False
    assert watch_row["forced_into_shortlist"] is True


def test_requirement_beam_positive_only_shortlist_sort_prefers_support_gain():
    cache_entry = {
        "annotation_pool_k": 3,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
            {"requirement_id": "bridge_0", "type": "bridge"},
        ],
        "counterfactual_sets": [
            {
                "cf_id": "cf_0",
                "requirements": [
                    {"requirement_id": "bridge_cf", "type": "bridge"},
                ],
            }
        ],
        "pool_titles": ["Anchor", "Margin Doc", "Support Doc"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Anchor",
                "positive_requirement_scores": {"anchor_0": 1.0, "bridge_0": 0.0},
                "counterfactual_requirement_scores": {"cf_0": {"bridge_cf": 0.0}},
                "counterfactual_set_scores": {"cf_0": 0.0},
            },
            {
                "pool_position": 1,
                "doc_title": "Margin Doc",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.6},
                "counterfactual_requirement_scores": {"cf_0": {"bridge_cf": 0.0}},
                "counterfactual_set_scores": {"cf_0": 0.0},
            },
            {
                "pool_position": 2,
                "doc_title": "Support Doc",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.9},
                "counterfactual_requirement_scores": {"cf_0": {"bridge_cf": 0.85}},
                "counterfactual_set_scores": {"cf_0": 0.85},
            },
        ],
    }

    margin_positions, margin_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.80, 0.70], dtype=float),
        pool_doc_titles=["Anchor", "Margin Doc", "Support Doc"],
        doc_idx_to_entities={40: {"anchor"}, 41: {"margin"}, 42: {"support"}},
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=2,
        beam_candidate_shortlist_limit=1,
        trace_watch_titles=["Support Doc"],
        shortlist_sort_mode="margin_first",
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )
    positive_positions, positive_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.80, 0.70], dtype=float),
        pool_doc_titles=["Anchor", "Margin Doc", "Support Doc"],
        doc_idx_to_entities={40: {"anchor"}, 41: {"margin"}, 42: {"support"}},
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=2,
        beam_candidate_shortlist_limit=1,
        trace_watch_titles=["Support Doc"],
        shortlist_sort_mode="positive_only",
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )

    assert margin_positions == [0, 1]
    assert positive_positions == [0, 2]
    margin_step = margin_trace["selection_steps"][1]
    positive_step = positive_trace["selection_steps"][1]
    assert margin_step["shortlist_sort_mode"] == "margin_first"
    assert positive_step["shortlist_sort_mode"] == "positive_only"
    assert margin_step["candidate_shortlist_preview"][0]["title"] == "Margin Doc"
    assert positive_step["candidate_shortlist_preview"][0]["title"] == "Support Doc"
    support_watch = positive_step["watch_title_trace"][0]
    assert support_watch["dominant_positive_unit_id"] == "bridge_0"
    assert support_watch["dominant_positive_unit_precovered"] is False
    assert support_watch["support_completeness_after"] > support_watch["counterfactual_leakage_after"]


def test_requirement_beam_support_bonus_source_sort_can_rescue_deep_bridge_into_source():
    cache_entry = {
        "annotation_pool_k": 3,
        "positive_requirements": [
            {"requirement_id": "anchor_0", "type": "anchor"},
            {"requirement_id": "bridge_0", "type": "bridge"},
            {"requirement_id": "decision_0", "type": "decision"},
        ],
        "counterfactual_sets": [],
        "pool_titles": ["Anchor", "Bridge Prior", "Bridge Rescue"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Anchor",
                "positive_requirement_scores": {"anchor_0": 1.0, "bridge_0": 0.0, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Bridge Prior",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 0.1, "decision_0": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 2,
                "doc_title": "Bridge Rescue",
                "positive_requirement_scores": {"anchor_0": 0.0, "bridge_0": 1.0, "decision_0": 1.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    combined_positions, combined_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.90, 0.05], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Bridge Rescue"],
        doc_idx_to_entities={40: {"anchor"}, 41: {"bridge prior"}, 42: {"bridge rescue"}},
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=1,
        beam_candidate_shortlist_limit=1,
        trace_watch_titles=["Bridge Rescue"],
        source_sort_mode="combined",
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )
    support_positions, support_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.90, 0.05], dtype=float),
        pool_doc_titles=["Anchor", "Bridge Prior", "Bridge Rescue"],
        doc_idx_to_entities={40: {"anchor"}, 41: {"bridge prior"}, 42: {"bridge rescue"}},
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"anchor"},
        proposal_query_entities={"target"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=1,
        beam_candidate_shortlist_limit=1,
        trace_watch_titles=["Bridge Rescue"],
        source_sort_mode="support_bonus",
        source_support_gain_weight=0.5,
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )

    assert combined_positions == [0, 1]
    assert support_positions == [0, 2]
    combined_step = combined_trace["selection_steps"][1]
    support_step = support_trace["selection_steps"][1]
    assert combined_trace["source_sort_mode"] == "combined"
    assert support_trace["source_sort_mode"] == "support_bonus"
    assert support_trace["source_support_gain_weight"] == pytest.approx(0.5)
    assert [row["pool_position"] for row in combined_step["candidate_source_preview"]] == [1]
    assert [row["pool_position"] for row in support_step["candidate_source_preview"]] == [2]
    assert [row["pool_position"] for row in support_step["candidate_shortlist_preview"]] == [2]
    support_watch = support_step["watch_title_trace"][0]
    assert support_watch["title"] == "Bridge Rescue"
    assert support_watch["support_completeness_gain"] > 0.0
    assert support_watch["source_sort_score"] > support_watch["combined_score"]
    assert support_watch["base_score"] == pytest.approx(0.0)


def test_requirement_beam_variable_binding_bonus_can_promote_upstream_bridge():
    cache_entry = {
        "version": NEED_UNIT_CACHE_VERSION,
        "annotation_pool_k": 3,
        "positive_need_units": [
            {
                "unit_id": "u0",
                "unit_type": "entity_locator",
                "subject": "southeast library",
                "predicate": "designer_of",
                "object": "ralph rapson",
                "target_variable": "?x",
            },
            {
                "unit_id": "u1",
                "unit_type": "relation_hop",
                "subject": "?x",
                "predicate": "death_place",
                "object": "gulf of mexico",
                "target_variable": "?y",
            },
        ],
        "counterfactual_sets": [],
        "pool_titles": ["Anchor Doc", "Distractor", "Upstream Bridge"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "Anchor Doc",
                "doc_entities": ["ralph rapson"],
                "positive_need_unit_scores": {"u0": 1.0, "u1": 0.0},
                "positive_requirement_scores": {"u0": 1.0, "u1": 0.0},
                "positive_alignment_scores": {"u0": 1.0, "u1": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "Distractor",
                "doc_entities": ["noise"],
                "positive_need_unit_scores": {"u0": 0.0, "u1": 0.0},
                "positive_requirement_scores": {"u0": 0.0, "u1": 0.0},
                "positive_alignment_scores": {"u0": 0.0, "u1": 0.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 2,
                "doc_title": "Upstream Bridge",
                "doc_entities": ["ralph rapson", "minneapolis"],
                "positive_need_unit_scores": {"u0": 0.0, "u1": 0.0},
                "positive_requirement_scores": {"u0": 0.0, "u1": 0.0},
                "positive_alignment_scores": {"u0": 0.0, "u1": 0.8},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
    }

    baseline_positions, baseline_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.80, 0.05], dtype=float),
        pool_doc_titles=["Anchor Doc", "Distractor", "Upstream Bridge"],
        doc_idx_to_entities={
            40: {"ralph rapson"},
            41: {"noise"},
            42: {"ralph rapson", "minneapolis"},
        },
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"southeast library"},
        proposal_query_entities={"gulf of mexico"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=2,
        beam_candidate_shortlist_limit=1,
        trace_watch_titles=["Upstream Bridge"],
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )
    bonus_positions, bonus_trace = select_requirement_beam_positions(
        pool_doc_ids=[40, 41, 42],
        pool_doc_scores=np.array([0.95, 0.80, 0.05], dtype=float),
        pool_doc_titles=["Anchor Doc", "Distractor", "Upstream Bridge"],
        doc_idx_to_entities={
            40: {"ralph rapson"},
            41: {"noise"},
            42: {"ralph rapson", "minneapolis"},
        },
        doc_idx_to_edges={40: [], 41: [], 42: []},
        adjacency={},
        qa_top_k=2,
        cache_entry=cache_entry,
        initial_seed_entities={"southeast library"},
        proposal_query_entities={"gulf of mexico"},
        anchor_count=1,
        reserve_top_m=1,
        structure_max_hops=1,
        beam_width=1,
        beam_expand_per_state=1,
        beam_projected_shortlist_factor=2,
        beam_candidate_shortlist_limit=1,
        trace_watch_titles=["Upstream Bridge"],
        bridge_bonus_mode="variable_binding",
        bridge_bonus_weight=0.6,
        non_anchor_title_dedup=False,
        requirement_mode="oracle",
    )

    assert baseline_positions == [0, 1]
    assert bonus_positions == [0, 2]
    assert baseline_trace["bridge_bonus_mode"] == "off"
    assert bonus_trace["bridge_bonus_mode"] == "variable_binding"
    assert bonus_trace["bridge_bonus_weight"] == pytest.approx(0.6)

    baseline_step = baseline_trace["selection_steps"][1]
    bonus_step = bonus_trace["selection_steps"][1]
    assert [row["pool_position"] for row in baseline_step["candidate_source_preview"]] == [1, 2]
    assert [row["pool_position"] for row in bonus_step["candidate_source_preview"]] == [1, 2]
    assert [row["pool_position"] for row in baseline_step["candidate_shortlist_preview"]] == [1]
    assert [row["pool_position"] for row in bonus_step["candidate_shortlist_preview"]] == [2]

    bonus_watch = bonus_step["watch_title_trace"][0]
    assert bonus_watch["title"] == "Upstream Bridge"
    assert bonus_watch["bridge_bonus_applied"] is True
    assert bonus_watch["bridge_bonus_unit_id"] == "u1"
    assert bonus_watch["bridge_bonus_predecessor_id"] == "u0"
    assert bonus_watch["bridge_bonus_score"] > 0.0
    assert bonus_watch["support_completeness_gain"] > 0.0
    assert bonus_watch["utility_margin_gain"] > 0.0
    assert bonus_watch["dominant_positive_unit_id"] == "u1"
    assert bonus_watch["dominant_positive_unit_score"] > 0.0


def test_resolve_reserved_positions_dedupes_prefix_titles():
    anchor_positions, reserved_positions = resolve_reserved_positions(
        candidate_count=5,
        target_k=3,
        anchor_count=2,
        reserve_top_m=3,
        pool_doc_titles=["A", "A", "B", "C", "D"],
        dedup_titles=True,
    )

    assert anchor_positions == [0, 2]
    assert reserved_positions == [0, 2, 3]


def test_align_requirement_cache_entry_to_pool_reorders_annotations_by_title():
    cache_entry = {
        "annotation_pool_k": 3,
        "pool_titles": ["A", "B", "C"],
        "doc_annotations": [
            {
                "pool_position": 0,
                "doc_title": "A",
                "positive_requirement_scores": {"anchor_0": 1.0},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 1,
                "doc_title": "B",
                "positive_requirement_scores": {"anchor_0": 0.5},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
            {
                "pool_position": 2,
                "doc_title": "C",
                "positive_requirement_scores": {"anchor_0": 0.2},
                "counterfactual_requirement_scores": {},
                "counterfactual_set_scores": {},
            },
        ],
        "diagnostics": {},
    }

    aligned_entry = align_requirement_cache_entry_to_pool(
        cache_entry=cache_entry,
        pool_titles=["A", "C", "B", "D"],
    )

    assert aligned_entry["pool_titles"] == ["A", "C", "B"]
    assert [row["doc_title"] for row in aligned_entry["doc_annotations"]] == ["A", "C", "B"]
    assert [row["pool_position"] for row in aligned_entry["doc_annotations"]] == [0, 1, 2]
    assert aligned_entry["doc_annotations"][1]["positive_requirement_scores"]["anchor_0"] == 0.2
    assert aligned_entry["diagnostics"]["title_aligned_from_cache"] is True


def test_align_requirement_cache_entry_to_pool_rebuilds_missing_titles_from_runtime_docs():
    cache_entry = build_requirement_cache_entry(
        query_index=0,
        question="Where was Alpha born?",
        pool_docs=[
            "Alpha\nAlpha was born in Paris.",
            "Beta\nBeta links Alpha to Paris.",
            "Gamma\nGamma is a distractor.",
        ],
        pool_doc_entities=[
            {"alpha", "paris"},
            {"beta", "alpha", "paris"},
            {"gamma"},
        ],
        seed_entities={"alpha"},
        question_entities={"alpha"},
        annotation_pool_k=3,
    )

    aligned_entry = align_requirement_cache_entry_to_pool(
        cache_entry=cache_entry,
        pool_titles=["Alpha", "Delta", "Gamma"],
        pool_docs=[
            "Alpha\nAlpha was born in Paris.",
            "Delta\nDelta mentions Paris as Alpha's birthplace.",
            "Gamma\nGamma is a distractor.",
        ],
        pool_doc_entities=[
            {"alpha", "paris"},
            {"delta", "alpha", "paris"},
            {"gamma"},
        ],
    )

    assert aligned_entry["pool_titles"] == ["Alpha", "Delta", "Gamma"]
    assert [row["doc_title"] for row in aligned_entry["doc_annotations"]] == ["Alpha", "Delta", "Gamma"]
    rebuilt_row = aligned_entry["doc_annotations"][1]
    assert rebuilt_row["pool_position"] == 1
    assert rebuilt_row["doc_title"] == "Delta"
    assert rebuilt_row["positive_requirement_scores"]
    assert aligned_entry["diagnostics"]["title_alignment_rebuilt_count"] == 1
    assert aligned_entry["diagnostics"]["title_alignment_rebuilt_first_title"] == "Delta"


def test_align_requirement_cache_entry_to_pool_extends_runtime_pool_beyond_cached_annotation_depth():
    cache_entry = build_requirement_cache_entry(
        query_index=0,
        question="Where was Alpha born?",
        pool_docs=[
            "Alpha\nAlpha was born in Paris.",
            "Beta\nBeta links Alpha to Paris.",
        ],
        pool_doc_entities=[
            {"alpha", "paris"},
            {"beta", "alpha", "paris"},
        ],
        seed_entities={"alpha"},
        question_entities={"alpha"},
        annotation_pool_k=2,
    )

    aligned_entry = align_requirement_cache_entry_to_pool(
        cache_entry=cache_entry,
        pool_titles=["Alpha", "Beta", "Gamma", "Delta"],
        pool_docs=[
            "Alpha\nAlpha was born in Paris.",
            "Beta\nBeta links Alpha to Paris.",
            "Gamma\nGamma is located in France.",
            "Delta\nDelta mentions Alpha and Paris together.",
        ],
        pool_doc_entities=[
            {"alpha", "paris"},
            {"beta", "alpha", "paris"},
            {"gamma", "france"},
            {"delta", "alpha", "paris"},
        ],
    )

    assert aligned_entry["annotation_pool_k"] == 4
    assert aligned_entry["pool_titles"] == ["Alpha", "Beta", "Gamma", "Delta"]
    assert [row["pool_position"] for row in aligned_entry["doc_annotations"]] == [0, 1, 2, 3]
    assert [row["doc_title"] for row in aligned_entry["doc_annotations"]] == ["Alpha", "Beta", "Gamma", "Delta"]
    assert aligned_entry["diagnostics"]["title_alignment_rebuilt_count"] == 2
    assert aligned_entry["diagnostics"]["title_alignment_rebuilt_first_title"] == "Gamma"


def test_align_requirement_cache_entry_to_pool_records_hybrid_live_annotation_metadata():
    cache_entry = build_need_unit_cache_entry(
        query_index=0,
        question="Where was Person A born?",
        pool_docs=[
            "Person A\nPerson A was born in City X.",
        ],
        pool_doc_entities=[
            {"person a", "city x"},
        ],
        seed_entities={"person a"},
        question_entities={"person a"},
        annotation_pool_k=1,
    )
    atomic_bundle = {
        "model": DummyAtomicNeedUnitModel(),
        "feature_names": list(NEED_UNIT_ATOMIC_FEATURE_NAMES),
        "labels": list(NEED_UNIT_ATOMIC_LABELS),
        "bridge_alpha_by_unit_type": {"relation_hop": 0.65},
        "beta_contradiction": 1.0,
        "scorer_version": "need_unit_atomic_v1",
        "model_path": "/tmp/live-atomic.joblib",
    }

    aligned_entry = align_requirement_cache_entry_to_pool(
        cache_entry=cache_entry,
        pool_titles=["Person A", "City X"],
        pool_docs=[
            "Person A\nPerson A was born in City X.",
            "City X\nCity X is the birth place of Person A.",
        ],
        pool_doc_entities=[
            {"person a", "city x"},
            {"person a", "city x"},
        ],
        atomic_scorer_bundle=atomic_bundle,
        score_mode="hybrid",
    )

    rebuilt_row = aligned_entry["doc_annotations"][1]
    assert rebuilt_row["doc_title"] == "City X"
    assert rebuilt_row["annotation_metadata"]["score_mode"] == "hybrid"
    assert rebuilt_row["annotation_metadata"]["model_path"] == "/tmp/live-atomic.joblib"
    assert aligned_entry["diagnostics"]["title_alignment_score_mode"] == "hybrid"
    assert aligned_entry["diagnostics"]["title_alignment_atomic_model_path"] == "/tmp/live-atomic.joblib"


def test_resolve_requirement_beam_runtime_reserve_config_supports_fixed_and_adaptive():
    small_cache_entry = {
        "positive_requirements": [
            {"requirement_id": "anchor_0"},
            {"requirement_id": "decision_0"},
        ],
    }
    medium_cache_entry = {
        "positive_requirements": [
            {"requirement_id": "anchor_0"},
            {"requirement_id": "bridge_0"},
            {"requirement_id": "bridge_1"},
            {"requirement_id": "decision_0"},
        ],
    }

    fixed_config = resolve_requirement_beam_runtime_reserve_config(
        anchor_count=2,
        reserve_top_m=3,
        cache_entry=small_cache_entry,
        policy="fixed",
    )
    adaptive_small = resolve_requirement_beam_runtime_reserve_config(
        anchor_count=2,
        reserve_top_m=3,
        cache_entry=small_cache_entry,
        policy="adaptive_requirement_count",
    )
    adaptive_medium = resolve_requirement_beam_runtime_reserve_config(
        anchor_count=2,
        reserve_top_m=3,
        cache_entry=medium_cache_entry,
        policy="adaptive_requirement_count",
    )

    assert fixed_config["anchor_count"] == 2
    assert fixed_config["reserve_top_m"] == 3
    assert fixed_config["effective_reserved_count"] == 3
    assert fixed_config["policy_reason"] == "fixed"

    assert adaptive_small["anchor_count"] == 1
    assert adaptive_small["reserve_top_m"] == 1
    assert adaptive_small["effective_reserved_count"] == 1
    assert adaptive_small["policy_reason"] == "positive_requirements<=2"

    assert adaptive_medium["anchor_count"] == 1
    assert adaptive_medium["reserve_top_m"] == 2
    assert adaptive_medium["effective_reserved_count"] == 2
    assert adaptive_medium["policy_reason"] == "positive_requirements<=4"


def test_build_report_examples_preserves_current_solution_trace():
    config = type("Config", (), {"qa_top_k": 2, "causal_engine_version": "legacy"})()
    query_solution = QuerySolution(
        question="Where was Alpha born?",
        docs=["Alpha\nAlpha was born in Paris.", "Beta\nBeta is unrelated."],
        answer="Paris",
        gold_answers=["Paris"],
        retrieval_trace={"setwise_selector_trace": {"runtime_reserve_top_m": 1}},
    )

    examples = build_report_examples(
        config=config,
        query_solutions=[query_solution],
        doc_text_to_chunk_id={
            "Alpha\nAlpha was born in Paris.": "chunk-alpha",
            "Beta\nBeta is unrelated.": "chunk-beta",
        },
        retrieval_only=False,
        doc_limit=1,
    )

    assert len(examples) == 1
    assert examples[0]["docs"] == ["Alpha\nAlpha was born in Paris."]
    assert examples[0]["retrieval_trace"]["setwise_selector_trace"]["runtime_reserve_top_m"] == 1
    assert examples[0]["retrieved_doc_ids"] == ["chunk-alpha", "chunk-beta"]


def test_build_setwise_selector_query_traces_uses_selected_solutions():
    config = type("Config", (), {"qa_top_k": 2, "causal_engine_version": "legacy"})()
    baseline_solution = QuerySolution(
        question="Where was Alpha born?",
        docs=["Alpha\nAlpha was born in London.", "Beta\nBeta mentions Paris."],
        answer="London",
        gold_answers=["Paris"],
        retrieval_trace={},
    )
    selected_solution = QuerySolution(
        question="Where was Alpha born?",
        docs=["Beta\nBeta mentions Paris.", "Alpha\nAlpha was born in London."],
        answer="Paris",
        gold_answers=["Paris"],
        retrieval_trace={
            "setwise_selector_trace": {
                "selected_titles": ["Beta", "Alpha"],
                "runtime_reserve_top_m": 1,
                "selection_steps": [{"step": 1, "mode": "requirement_beam"}],
            },
        },
    )

    query_traces = build_setwise_selector_query_traces(
        config=config,
        baseline_solutions=[baseline_solution],
        selected_solutions=[selected_solution],
        gold_docs=[["Beta\nBeta mentions Paris."]],
        gold_answers=[["Paris"]],
        doc_text_to_chunk_id={
            "Alpha\nAlpha was born in London.": "chunk-alpha",
            "Beta\nBeta mentions Paris.": "chunk-beta",
        },
    )

    assert len(query_traces) == 1
    trace = query_traces[0]
    assert trace["baseline_top_titles"] == ["Alpha", "Beta"]
    assert trace["selector_top_titles"] == ["Beta", "Alpha"]
    assert trace["changed_from_baseline"] is True
    assert trace["baseline_metrics"]["ExactMatch"] == 0.0
    assert trace["selector_metrics"]["ExactMatch"] == 1.0
    assert trace["selector_trace"]["runtime_reserve_top_m"] == 1
    assert trace["selector_top_doc_ids"] == ["chunk-beta", "chunk-alpha"]


def test_build_requirement_title_exposure_summary_tracks_source_shortlist_and_selected():
    selector_trace = {
        "selected_titles": ["Bridge Doc"],
        "final_front_titles": ["Bridge Doc"],
        "selection_steps": [
            {
                "step": 2,
                "candidate_source_preview": [
                    {"preview_rank": 1, "pool_position": 1, "title": "Bridge Doc"},
                    {"preview_rank": 2, "pool_position": 2, "title": "Deep Gold"},
                ],
                "candidate_shortlist_preview": [
                    {"preview_rank": 1, "pool_position": 2, "title": "Deep Gold"},
                ],
            },
        ],
    }

    summary = build_requirement_title_exposure_summary(
        pool_titles=["Anchor", "Bridge Doc", "Deep Gold", "Distractor"],
        selector_trace=selector_trace,
        target_titles=["Bridge Doc", "Deep Gold", "Missing Gold"],
    )

    by_title = {row["title"]: row for row in summary}
    assert by_title["Bridge Doc"]["stage"] == "selected"
    assert by_title["Bridge Doc"]["appears_in_final_evidence"] is True
    assert by_title["Deep Gold"]["stage"] == "shortlist"
    assert by_title["Deep Gold"]["source_first_step"] == 2
    assert by_title["Deep Gold"]["shortlist_first_step"] == 2
    assert by_title["Missing Gold"]["stage"] == "not_in_pool"


def test_build_requirement_title_exposure_summary_marks_forced_final_only_titles():
    selector_trace = {
        "selected_titles": ["Bridge Doc"],
        "final_front_titles": ["Deep Gold", "Bridge Doc"],
        "forced_final_titles_applied": ["Deep Gold"],
        "selection_steps": [
            {
                "step": 2,
                "candidate_source_preview": [
                    {"preview_rank": 1, "pool_position": 1, "title": "Bridge Doc"},
                ],
                "candidate_shortlist_preview": [
                    {"preview_rank": 1, "pool_position": 1, "title": "Bridge Doc"},
                ],
            },
        ],
    }

    summary = build_requirement_title_exposure_summary(
        pool_titles=["Anchor", "Bridge Doc", "Deep Gold", "Distractor"],
        selector_trace=selector_trace,
        target_titles=["Bridge Doc", "Deep Gold"],
    )

    by_title = {row["title"]: row for row in summary}
    assert by_title["Deep Gold"]["stage"] == "final_only"
    assert by_title["Deep Gold"]["appears_in_final_evidence"] is True
    assert by_title["Deep Gold"]["appears_in_selected_set"] is False
    assert by_title["Deep Gold"]["forced_into_final"] is True
    assert by_title["Bridge Doc"]["stage"] == "selected"


def test_build_requirement_title_exposure_summary_marks_forced_pool_gold_final_titles():
    selector_trace = {
        "selected_titles": ["Bridge Doc"],
        "final_front_titles": ["Deep Gold", "Bridge Doc"],
        "forced_pool_gold_final_titles_applied": ["Deep Gold"],
        "selection_steps": [
            {
                "step": 2,
                "candidate_source_preview": [
                    {"preview_rank": 1, "pool_position": 1, "title": "Bridge Doc"},
                ],
                "candidate_shortlist_preview": [
                    {"preview_rank": 1, "pool_position": 1, "title": "Bridge Doc"},
                ],
            },
        ],
    }

    summary = build_requirement_title_exposure_summary(
        pool_titles=["Anchor", "Bridge Doc", "Deep Gold", "Distractor"],
        selector_trace=selector_trace,
        target_titles=["Bridge Doc", "Deep Gold"],
    )

    by_title = {row["title"]: row for row in summary}
    assert by_title["Deep Gold"]["stage"] == "final_only"
    assert by_title["Deep Gold"]["forced_into_final"] is True


def test_resolve_query_pool_gold_titles_separates_in_pool_from_missing_titles():
    payload = resolve_query_pool_gold_titles(
        pool_titles=[
            "Vilaiyaadu Mankatha",
            "The Right Stuff Records",
            "Love Around",
            "Sony Music",
        ],
        gold_docs=[
            "Vilaiyaadu Mankatha\nA film page.",
            "Sony Music\nA label page.",
            "Santa Monica, California\nA city page.",
        ],
        pool_limit=3,
    )

    assert payload["requested_titles"] == [
        "Vilaiyaadu Mankatha",
        "Sony Music",
        "Santa Monica, California",
    ]
    assert payload["in_pool_titles"] == ["Vilaiyaadu Mankatha"]
    assert payload["missing_titles"] == ["Sony Music", "Santa Monica, California"]
    assert payload["positions"] == [0]


def test_build_setwise_selector_query_traces_surfaces_requirement_exposure_summary():
    config = type("Config", (), {"qa_top_k": 2, "causal_engine_version": "legacy"})()
    baseline_solution = QuerySolution(
        question="Where was Alpha born?",
        docs=["Alpha\nAlpha was born in London.", "Beta\nBeta mentions Paris."],
        answer="London",
        gold_answers=["Paris"],
        retrieval_trace={},
    )
    selected_solution = QuerySolution(
        question="Where was Alpha born?",
        docs=["Beta\nBeta mentions Paris.", "Alpha\nAlpha was born in London."],
        answer="Paris",
        gold_answers=["Paris"],
        retrieval_trace={
            "setwise_selector_trace": {
                "selected_titles": ["Beta", "Alpha"],
                "final_front_titles": ["Beta", "Alpha"],
                "requirement_title_exposure_summary": [
                    {
                        "title": "Beta",
                        "stage": "selected",
                        "appears_in_candidate_source": True,
                        "appears_in_candidate_shortlist": True,
                        "appears_in_final_evidence": True,
                    },
                ],
            },
        },
    )

    query_traces = build_setwise_selector_query_traces(
        config=config,
        baseline_solutions=[baseline_solution],
        selected_solutions=[selected_solution],
        gold_docs=[["Beta\nBeta mentions Paris."]],
        gold_answers=[["Paris"]],
        doc_text_to_chunk_id={
            "Alpha\nAlpha was born in London.": "chunk-alpha",
            "Beta\nBeta mentions Paris.": "chunk-beta",
        },
    )

    assert query_traces[0]["gold_titles"] == ["Beta"]
    assert query_traces[0]["requirement_title_exposure_summary"][0]["title"] == "Beta"
    assert query_traces[0]["requirement_title_exposure_summary"][0]["stage"] == "selected"


def test_select_bridge_append_positions_stops_below_threshold(monkeypatch):
    def fake_score_bridge_candidates(**kwargs):
        return [
            {
                "pool_position": 2,
                "doc_id": 12,
                "doc_title": "Weak Bridge",
                "doc_entities": {"gamma"},
                "structure_score": 0.20,
                "closure_score": 0.15,
                "novelty_score": 0.40,
                "combined_score": 0.25,
            },
        ]

    monkeypatch.setattr(eval_causal_qwen3_module, "score_bridge_candidates", fake_score_bridge_candidates)

    selected_positions, trace = select_bridge_append_positions(
        pool_doc_ids=[10, 11, 12],
        normalized_base_scores=np.asarray([1.0, 0.8, 0.2], dtype=float),
        pool_doc_titles=["Anchor", "Incumbent", "Weak Bridge"],
        doc_idx_to_entities={10: {"alpha"}, 11: {"beta"}, 12: {"gamma"}},
        doc_idx_to_edges={},
        adjacency={},
        initial_seed_entities={"alpha"},
        query_entities={"alpha"},
        pool_limit=3,
        expand_base_k=2,
        append_max_docs=2,
        expand_min_structure_score=0.35,
        structure_max_hops=2,
        structure_seed_target_bridge_mode="off",
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1]
    assert trace["append_count"] == 0
    assert trace["append_stop_reason"] == "structure_below_threshold"
    assert trace["candidate_set_positions"] == [0, 1]


def test_select_bridge_append_positions_skips_duplicate_titles_and_caps(monkeypatch):
    def fake_score_bridge_candidates(**kwargs):
        remaining_positions = list(kwargs["remaining_positions"])
        rows = []
        for pos in remaining_positions:
            if pos == 2:
                rows.append({
                    "pool_position": 2,
                    "doc_id": 12,
                    "doc_title": "Incumbent",
                    "doc_entities": {"gamma"},
                    "structure_score": 0.95,
                    "closure_score": 0.90,
                    "novelty_score": 0.40,
                    "combined_score": 0.70,
                })
            elif pos == 3:
                rows.append({
                    "pool_position": 3,
                    "doc_id": 13,
                    "doc_title": "Bridge A",
                    "doc_entities": {"delta"},
                    "structure_score": 0.90,
                    "closure_score": 0.70,
                    "novelty_score": 0.50,
                    "combined_score": 0.65,
                })
            elif pos == 4:
                rows.append({
                    "pool_position": 4,
                    "doc_id": 14,
                    "doc_title": "Bridge B",
                    "doc_entities": {"epsilon"},
                    "structure_score": 0.88,
                    "closure_score": 0.65,
                    "novelty_score": 0.45,
                    "combined_score": 0.62,
                })
        return rows

    monkeypatch.setattr(eval_causal_qwen3_module, "score_bridge_candidates", fake_score_bridge_candidates)

    selected_positions, trace = select_bridge_append_positions(
        pool_doc_ids=[10, 11, 12, 13, 14],
        normalized_base_scores=np.asarray([1.0, 0.9, 0.4, 0.3, 0.2], dtype=float),
        pool_doc_titles=["Anchor", "Incumbent", "Incumbent", "Bridge A", "Bridge B"],
        doc_idx_to_entities={10: {"alpha"}, 11: {"beta"}, 12: {"gamma"}, 13: {"delta"}, 14: {"epsilon"}},
        doc_idx_to_edges={},
        adjacency={},
        initial_seed_entities={"alpha"},
        query_entities={"alpha"},
        pool_limit=5,
        expand_base_k=2,
        append_max_docs=2,
        expand_min_structure_score=0.35,
        structure_max_hops=2,
        structure_seed_target_bridge_mode="off",
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
    )

    assert selected_positions == [0, 1, 3, 4]
    assert trace["appended_positions"] == [3, 4]
    assert trace["append_count"] == 2
    assert trace["append_stop_reason"] == "append_cap_reached"
    assert trace["append_steps"][0]["duplicate_skip_count"] == 1


def test_select_bridge_append_positions_supports_next_deep_policy():
    selected_positions, trace = select_bridge_append_positions(
        pool_doc_ids=[10, 11, 12, 13, 14],
        normalized_base_scores=np.asarray([1.0, 0.9, 0.4, 0.3, 0.2], dtype=float),
        pool_doc_titles=["Anchor", "Incumbent", "Deep A", "Deep B", "Deep C"],
        doc_idx_to_entities={10: {"alpha"}, 11: {"beta"}, 12: {"gamma"}, 13: {"delta"}, 14: {"epsilon"}},
        doc_idx_to_edges={},
        adjacency={},
        initial_seed_entities={"alpha"},
        query_entities={"alpha"},
        pool_limit=5,
        expand_base_k=2,
        append_max_docs=2,
        expand_min_structure_score=0.35,
        structure_max_hops=2,
        structure_seed_target_bridge_mode="off",
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        append_policy="next_deep",
    )

    assert selected_positions == [0, 1, 2, 3]
    assert trace["append_policy"] == "next_deep"
    assert trace["appended_positions"] == [2, 3]
    assert trace["append_steps"][0]["selection_policy"] == "next_deep"


def test_select_bridge_append_positions_supports_random_deep_policy():
    expected_positions = [
        int(pos)
        for pos in np.random.default_rng(7).permutation([2, 3, 4, 5]).tolist()[:2]
    ]

    selected_positions, trace = select_bridge_append_positions(
        pool_doc_ids=[10, 11, 12, 13, 14, 15],
        normalized_base_scores=np.asarray([1.0, 0.9, 0.5, 0.4, 0.3, 0.2], dtype=float),
        pool_doc_titles=["Anchor", "Incumbent", "Deep A", "Deep B", "Deep C", "Deep D"],
        doc_idx_to_entities={
            10: {"alpha"},
            11: {"beta"},
            12: {"gamma"},
            13: {"delta"},
            14: {"epsilon"},
            15: {"zeta"},
        },
        doc_idx_to_edges={},
        adjacency={},
        initial_seed_entities={"alpha"},
        query_entities={"alpha"},
        pool_limit=6,
        expand_base_k=2,
        append_max_docs=2,
        expand_min_structure_score=0.35,
        structure_max_hops=2,
        structure_seed_target_bridge_mode="off",
        base_weight=0.25,
        structure_weight=0.60,
        novelty_weight=0.15,
        append_policy="random_deep",
        append_random_seed=7,
    )

    assert selected_positions == [0, 1] + expected_positions
    assert trace["append_policy"] == "random_deep"
    assert trace["append_random_seed"] == 7
    assert trace["appended_positions"] == expected_positions
    assert trace["append_steps"][0]["selection_policy"] == "random_deep"


def test_rerank_candidate_positions_for_assemble_supports_similarity_and_ce():
    hipporag = DummyHippoRAGForAssemble(
        query_embeddings={"where is alpha": np.asarray([1.0, 0.0], dtype=float)},
        passage_embeddings=np.asarray([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=float),
    )

    reranked_positions, trace = rerank_candidate_positions_for_assemble(
        query="where is alpha",
        pool_docs=["Doc A\nalpha", "Doc B\nbeta", "Doc C\ngamma"],
        pool_doc_ids=[0, 1, 2],
        pool_doc_scores=np.asarray([0.3, 0.9, 0.6], dtype=float),
        candidate_positions=[0, 1, 2],
        assemble_mode="embedding_similarity",
        hipporag=hipporag,
        position_sources={0: "baseline_prefix", 1: "bridge_append", 2: "bridge_append"},
    )

    assert reranked_positions == [0, 2, 1]
    assert trace["score_field"] == "embedding_similarity"
    assert trace["ranking_rows"][0]["source"] == "baseline_prefix"

    ce_reranker = DummyCrossEncoder([0.2, 0.9, 0.4])
    ce_positions, ce_trace = rerank_candidate_positions_for_assemble(
        query="where is alpha",
        pool_docs=["Doc A\nalpha", "Doc B\nbeta", "Doc C\ngamma"],
        pool_doc_ids=[0, 1, 2],
        pool_doc_scores=np.asarray([0.3, 0.9, 0.6], dtype=float),
        candidate_positions=[0, 1, 2],
        assemble_mode="cross_encoder",
        hipporag=hipporag,
        ce_reranker=ce_reranker,
    )

    assert ce_positions == [1, 2, 0]
    assert ce_trace["score_field"] == "cross_encoder_score"
    assert len(ce_reranker.calls) == 1


def test_normalize_assemble_mode_accepts_daec_noisyor_llm():
    assert normalize_assemble_mode("daec_noisyor_llm") == "daec_noisyor_llm"


def test_build_expand_assemble_query_traces_surfaces_method_trace():
    config = type("Config", (), {"qa_top_k": 2, "causal_engine_version": "legacy"})()
    baseline_solution = QuerySolution(
        question="Where was Alpha born?",
        docs=["Alpha\nAlpha was born in London.", "Beta\nBeta mentions Paris."],
        answer="London",
        gold_answers=["Paris"],
        retrieval_trace={},
    )
    method_solution = QuerySolution(
        question="Where was Alpha born?",
        docs=["Beta\nBeta mentions Paris.", "Gamma\nGamma mentions Rome."],
        answer="Paris",
        gold_answers=["Paris"],
        retrieval_trace={
            "expand_assemble_trace": {
                "candidate_set_titles": ["Alpha", "Beta", "Gamma"],
                "appended_titles": ["Gamma"],
                "assemble_mode": "cross_encoder",
            },
        },
    )

    query_traces = build_expand_assemble_query_traces(
        config=config,
        baseline_solutions=[baseline_solution],
        method_solutions=[method_solution],
        gold_docs=[["Beta\nBeta mentions Paris."]],
        gold_answers=[["Paris"]],
        doc_text_to_chunk_id={
            "Alpha\nAlpha was born in London.": "chunk-alpha",
            "Beta\nBeta mentions Paris.": "chunk-beta",
            "Gamma\nGamma mentions Rome.": "chunk-gamma",
        },
    )

    assert query_traces[0]["gold_titles"] == ["Beta"]
    assert query_traces[0]["method_top_titles"] == ["Beta", "Gamma"]
    assert query_traces[0]["method_metrics"]["ExactMatch"] == 1.0
    assert query_traces[0]["expand_assemble_trace"]["assemble_mode"] == "cross_encoder"


def test_build_requirement_reserve_ablation_jobs_maps_effective_prefix_sizes():
    jobs = build_requirement_reserve_ablation_jobs(
        datasets=["musique"],
        reserve_values=[3, 1, 0],
        limit=40,
        save_dir="outputs_step0_general",
        default_anchor_count=2,
        include_adaptive=True,
        adaptive_base_reserve_top_m=3,
    )

    assert [job.label for job in jobs] == ["reserve3", "reserve1", "reserve0", "adaptive_reqcount"]
    assert resolve_dataset_save_dir("outputs_step0_general", "musique").as_posix().endswith("outputs_step0_general_musique")

    reserve3_job, reserve1_job, reserve0_job, adaptive_job = jobs
    assert reserve3_job.anchor_count == 2 and reserve3_job.reserve_top_m == 3
    assert reserve1_job.anchor_count == 1 and reserve1_job.reserve_top_m == 1
    assert reserve0_job.anchor_count == 0 and reserve0_job.reserve_top_m == 0
    assert adaptive_job.anchor_count == 2 and adaptive_job.reserve_top_m == 3
    assert adaptive_job.reserve_policy == "adaptive_requirement_count"
    assert reserve3_job.output_json.endswith("reserve3.json")
