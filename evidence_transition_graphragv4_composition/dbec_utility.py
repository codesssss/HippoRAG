"""DBEC frozen-binding utility provider for native PCEC readout."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dtc_embed_utils import DTCRequirement, select_daec_noisyor_positions  # noqa: E402
from src.hipporag.utils.misc_utils import compute_mdhash_id  # noqa: E402

from evidence_transition_graphragv4_composition.contract import (  # noqa: E402
    DEFAULT_DTC_BINDING_MAX_CANDIDATES,
    DEFAULT_SAFE_MIN_OBJECTIVE_GAIN,
    DEFAULT_SAFE_MIN_SWAP_GAIN,
    residual_budget,
)
from evidence_transition_graphragv4_composition.pcec_types import PCECQueryState  # noqa: E402


def load_binding_cache(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): [str(item) for item in value]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def make_cache_only_binding_extractor(
    *,
    cache: Mapping[str, Sequence[str]],
    model: str,
    stats: dict[str, int],
) -> Callable[[str, str], list[str]]:
    think_re = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    def extract(subquery: str, doc_text: str) -> list[str]:
        prompt_payload = {
            "version": "daec_llm_binding_v1",
            "model": str(model),
            "subquery": str(subquery),
            "passage": str(doc_text[:1500]),
        }
        cache_key = compute_mdhash_id(json.dumps(prompt_payload, sort_keys=True, ensure_ascii=False))
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        if cache_key not in cache:
            stats["misses"] = int(stats.get("misses", 0)) + 1
            return []
        stats["hits"] = int(stats.get("hits", 0)) + 1
        return [think_re.sub("", str(item)).strip() for item in cache[cache_key] if str(item).strip()]

    return extract


def build_text_embeddings(hipporag: Any, texts: Sequence[str], keys: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    items: list[tuple[str, str]] = []
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    for idx, text in enumerate(texts):
        clean_text = str(text or "").strip()
        if not clean_text:
            continue
        key = str(keys[idx]) if keys is not None and idx < len(keys) else clean_text
        items.append((key, clean_text))
        if clean_text not in seen_texts:
            seen_texts.add(clean_text)
            unique_texts.append(clean_text)
    if unique_texts and hasattr(hipporag, "_get_passage_query_embeddings"):
        hipporag._get_passage_query_embeddings(unique_texts)
    query_to_embedding = getattr(hipporag, "query_to_embedding", {}) or {}
    passage_query_embeddings = (
        (query_to_embedding.get("passage", {}) or {})
        if isinstance(query_to_embedding, Mapping)
        else {}
    )
    embeddings: dict[str, np.ndarray] = {}
    for key, text in items:
        vector = passage_query_embeddings.get(text)
        if vector is not None:
            embeddings[str(key)] = np.asarray(vector, dtype=float)
    return embeddings


def build_requirement_embeddings(hipporag: Any, requirements: Sequence[DTCRequirement]) -> dict[str, np.ndarray]:
    filtered = [req for req in requirements if str(req.subquery or "").strip()]
    return build_text_embeddings(
        hipporag,
        [req.subquery for req in filtered],
        keys=[req.unit_id for req in filtered],
    )


class NativeDBECUtilityProvider:
    """Library-level adapter over DBEC frozen-binding noisy-OR selection."""

    def __init__(
        self,
        *,
        hipporag: Any,
        binding_cache: Mapping[str, Sequence[str]],
        binding_model: str,
        binding_top_m: int = DEFAULT_DTC_BINDING_MAX_CANDIDATES,
        min_objective_gain: float = DEFAULT_SAFE_MIN_OBJECTIVE_GAIN,
        min_swap_gain: float = DEFAULT_SAFE_MIN_SWAP_GAIN,
        llm_binding_title_match_mode: str = "wiki_title",
    ) -> None:
        self.hipporag = hipporag
        self.binding_cache = dict(binding_cache)
        self.binding_model = str(binding_model)
        self.binding_top_m = int(binding_top_m)
        self.min_objective_gain = float(min_objective_gain)
        self.min_swap_gain = float(min_swap_gain)
        self.llm_binding_title_match_mode = str(llm_binding_title_match_mode)
        self.binding_stats = {"attempts": 0, "hits": 0, "misses": 0}
        self._llm_extract_fn = make_cache_only_binding_extractor(
            cache=self.binding_cache,
            model=self.binding_model,
            stats=self.binding_stats,
        )

    def select_positions(
        self,
        state: PCECQueryState,
        requirements: Sequence[DTCRequirement],
    ) -> tuple[list[int], dict[str, Any]]:
        requirement_embeddings = build_requirement_embeddings(self.hipporag, requirements)

        def embed_bound_texts(texts: Sequence[str]) -> dict[str, np.ndarray]:
            return build_text_embeddings(self.hipporag, texts)

        selected_positions, selector_trace = select_daec_noisyor_positions(
            query=state.question,
            requirements=list(requirements),
            requirement_embeddings=requirement_embeddings,
            pool_docs=list(state.pool_docs),
            pool_doc_ids=list(state.pool_doc_ids),
            pool_doc_scores=np.asarray(list(state.pool_doc_scores), dtype=float),
            pool_doc_titles=list(state.pool_titles),
            doc_idx_to_entities=getattr(self.hipporag, "doc_idx_to_structure_entities", {}) or {},
            passage_embeddings=np.asarray(getattr(self.hipporag, "passage_embeddings", np.array([]))),
            qa_top_k=int(state.reader_budget_k),
            binding_top_m=int(self.binding_top_m),
            embed_texts_fn=embed_bound_texts,
            safe_projection=True,
            safe_min_objective_gain=float(self.min_objective_gain),
            safe_min_swap_gain=float(self.min_swap_gain),
            safe_max_swaps=residual_budget(state.reader_budget_k, state.prefix_budget_m),
            safe_preserve_top_m=int(state.prefix_budget_m),
            safe_projection_mode="rank_cutoff",
            safe_retriever_margin_threshold=1.01,
            safe_retriever_rank_penalty=0.0,
            llm_extract_fn=self._llm_extract_fn,
            binding_mode="llm",
            llm_binding_title_match_mode=self.llm_binding_title_match_mode,
            agsto_metadata=dict(state.et_trace or {}),
            selector_label_override="pcec_native_prefix_residual_readout",
        )
        return [int(pos) for pos in selected_positions], dict(selector_trace)

    def summary(self) -> dict[str, Any]:
        return {
            "provider": "native_dbec_frozen_binding_noisy_or",
            "binding_model": self.binding_model,
            "binding_cache_entry_count": int(len(self.binding_cache)),
            "binding_top_m": int(self.binding_top_m),
            "min_objective_gain": float(self.min_objective_gain),
            "min_swap_gain": float(self.min_swap_gain),
            "llm_binding_title_match_mode": self.llm_binding_title_match_mode,
            **dict(self.binding_stats),
        }
