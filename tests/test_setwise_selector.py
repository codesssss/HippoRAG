import os
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import (
    LEARNED_SETWISE_FEATURE_NAMES,
    OpenAICompatibleLateRerankJudge,
    SetwiseLateRerankResponseModel,
    build_setwise_late_rerank_candidates,
    build_setwise_selector_query_traces,
    build_setwise_late_rerank_judge_bundle,
    build_report_examples,
    collect_grounded_question_query_entities,
    collect_lexical_query_seed_entities,
    collect_question_query_entities,
    compute_bridge_gate_decision,
    compute_candidate_feature_rows,
    compute_state_path_connectivity_metrics,
    materialize_reader_top_positions,
    normalize_setwise_late_rerank_policy,
    parse_setwise_late_rerank_response,
    rerank_completed_evidence_sets_with_llm,
    resolve_requirement_beam_runtime_reserve_config,
    resolve_reserved_positions,
    resolve_setwise_query_targets,
    score_evidence_state,
    select_bridge_beam_positions,
    select_bridge_greedy_positions,
    select_learned_greedy_positions,
    select_requirement_beam_positions,
    should_apply_setwise_late_rerank_override,
)
from requirement_beam_utils import (
    align_requirement_cache_entry_to_pool,
    build_requirement_cache_entry,
    compute_requirement_state_metrics,
    load_requirement_cache,
    REQUIREMENT_MATCHER_FEATURE_NAMES,
    save_requirement_cache,
)
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


def test_materialize_reader_top_positions_preserves_selected_prefix_and_fills_tail():
    top_positions = materialize_reader_top_positions(
        selected_positions=[3, 1, 3],
        pool_limit=6,
        qa_top_k=5,
    )

    assert top_positions == [3, 1, 0, 2, 4]


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
    assert decision["best_offrank_pool_position"] == 5
    assert decision["weakest_baseline_suffix_pool_position"] == 4


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
