import os
import sys
from pathlib import Path
import types
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.modules.setdefault("litellm", types.SimpleNamespace())

from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.rerank import DSPyFilter
from hipporag.utils.config_utils import BaseConfig


def test_base_config_allows_distinct_request_name():
    config = BaseConfig(
        save_dir="outputs_step0_general",
        llm_name="qwen3-8b",
        llm_request_name="qwen3-8b-train",
        llm_base_url="http://localhost:8043/v1",
        embedding_model_name="VLLM//mnt/nvme/Qwen3-Embedding-8B",
        embedding_base_url="http://localhost:8018/v1/embeddings",
    )
    assert config.llm_name == "qwen3-8b"
    assert config.llm_request_name == "qwen3-8b-train"


def test_cache_openai_uses_request_name_for_api_model_but_not_cache_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = BaseConfig(
        save_dir=str(tmp_path),
        llm_name="qwen3-8b",
        llm_request_name="qwen3-8b-train",
        llm_base_url="http://localhost:8043/v1",
        embedding_model_name="VLLM//mnt/nvme/Qwen3-Embedding-8B",
        embedding_base_url="http://localhost:8018/v1/embeddings",
    )
    llm = CacheOpenAI.from_experiment_config(config)

    assert llm.llm_config.generate_params["model"] == "qwen3-8b-train"
    assert llm.cache_file_name.endswith(os.path.join("llm_cache", "qwen3-8b_cache.sqlite"))


def test_dspy_filter_uses_request_name_for_rerank_and_repair_calls():
    config = BaseConfig(
        save_dir="outputs_step0_general",
        llm_name="qwen3-8b",
        llm_request_name="qwen3-8b-train",
        llm_base_url="http://localhost:8043/v1",
        embedding_model_name="VLLM//mnt/nvme/Qwen3-Embedding-8B",
        embedding_base_url="http://localhost:8018/v1/embeddings",
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
    )
    hipporag = SimpleNamespace(
        global_config=config,
        llm_model=SimpleNamespace(infer=lambda *args, **kwargs: ("", {})),
    )

    reranker = DSPyFilter(hipporag)

    assert reranker.model_name == "qwen3-8b-train"


def test_cache_openai_qwen_requests_are_prefixed_with_no_think(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("HIPPORAG_QWEN_FORCE_NO_THINK", raising=False)
    config = BaseConfig(
        save_dir=str(tmp_path),
        llm_name="qwen3-8b",
        llm_request_name="qwen3-8b-train",
        llm_base_url="http://localhost:8043/v1",
        embedding_model_name="VLLM//mnt/nvme/Qwen3-Embedding-8B",
        embedding_base_url="http://localhost:8018/v1/embeddings",
    )
    llm = CacheOpenAI.from_experiment_config(config)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "Example answer."},
        {"role": "user", "content": "Question: Who directed Billy Elliot?\nThought: "},
    ]

    normalized = llm._normalize_messages_for_request(messages, model="qwen3-8b-train")

    assert normalized[-1]["content"].startswith("/no_think\n")
    assert normalized[-1]["content"].count("/no_think") == 1
    assert messages[-1]["content"].startswith("Question:")


def test_cache_openai_no_think_can_be_disabled_for_qwen(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HIPPORAG_QWEN_FORCE_NO_THINK", "0")
    config = BaseConfig(
        save_dir=str(tmp_path),
        llm_name="qwen3-8b",
        llm_request_name="qwen3-8b-train",
        llm_base_url="http://localhost:8043/v1",
        embedding_model_name="VLLM//mnt/nvme/Qwen3-Embedding-8B",
        embedding_base_url="http://localhost:8018/v1/embeddings",
    )
    llm = CacheOpenAI.from_experiment_config(config)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Question: Who directed Billy Elliot?\nThought: "},
    ]

    normalized = llm._normalize_messages_for_request(messages, model="qwen3-8b-train")

    assert normalized[-1]["content"] == messages[-1]["content"]
