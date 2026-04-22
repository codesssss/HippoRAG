from src.hipporag.embedding_model import _get_embedding_model_class
from src.hipporag.embedding_model.NVEmbedV2 import NVEmbedV2EmbeddingModel
from src.hipporag.embedding_model.VLLM import VLLMEmbeddingModel


def test_vllm_prefix_takes_precedence_over_nv_embed_substring():
    assert _get_embedding_model_class("VLLM/nvidia/NV-Embed-v2") is VLLMEmbeddingModel


def test_plain_nv_embed_still_uses_local_nv_embed_class():
    assert _get_embedding_model_class("nvidia/NV-Embed-v2") is NVEmbedV2EmbeddingModel
