#!/usr/bin/env python3
"""Run reader QA on fixed top-5 evidence from STO transition variants.

This script is deliberately reader-facing:

- it does not run HippoRAGv2 retrieval;
- it does not run PPR, q-gates, rerankers, or graph expansion;
- it only feeds already-materialized top-5 passages to the QA reader.

The goal is to decide whether the current STO transition line improves actual
QA behavior, not just retrieval-side proxies.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from analyze_top5_reader_facing_audit import (
    DEFAULT_TRANSITION_REPORT,
    aggregate_method_rows,
    evaluate_top5_row,
    passage_for_doc,
    title_for_doc,
    unique_ints,
)
from analyze_upstream_substrate_failures import SFB_VARIANT, load_json
from compare_graph_retrievers import get_worktree_baseline_runtime, install_qwen_disable_thinking, make_config
from hipporag.llm import _get_llm_class
from hipporag.prompts.prompt_template_manager import PromptTemplateManager


METHOD_DOC_KEYS = {
    "sfb_context": "reader_doc_indices_topk",
    "semantic_pair": "semantic_pair_doc_indices_top5",
    "specificity_pair": "specificity_pair_doc_indices_top5",
    "channel_fusion": "channel_fusion",
    "support_fusion": "support_fusion",
    "native_support_fusion": "native_support_fusion",
    "context_support_fusion": "context_support_fusion",
    "bm25_support_fusion": "bm25_support_fusion",
    "hybrid_support_fusion": "hybrid_support_fusion",
    "native_consensus_support_fusion": "native_consensus_support_fusion",
    "native_spec_consensus_support_fusion": "native_spec_consensus_support_fusion",
    "native_anchor_neighborhood_support_fusion": "native_anchor_neighborhood_support_fusion",
    "anchor_guided_evidence": "anchor_guided_evidence_doc_indices_top5",
    "obligation_closed_sto_local_ppr": "obligation_closed_sto_local_ppr_doc_indices_top5",
}

FUSION_METHOD_BASE_CHANNELS = {
    "channel_fusion": "sfb",
    "support_fusion": "sfb",
    "native_support_fusion": "native_dense",
    "context_support_fusion": "context_anchor",
    "bm25_support_fusion": "bm25",
    "hybrid_support_fusion": "hybrid_residual",
    "native_consensus_support_fusion": "native_consensus",
    "native_spec_consensus_support_fusion": "native_spec_consensus",
    "native_anchor_neighborhood_support_fusion": "native_anchor_neighborhood",
}

CONTEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "with",
}

DEFAULT_OUTPUT_JSON = (
    "outputs_anchor_source_audit_limit100_20260423_v2/reports/"
    "transition_top5_qa_v1_specificitypair_w12.json"
)
DEFAULT_OUTPUT_MD = (
    "outputs_anchor_source_audit_limit100_20260423_v2/reports/"
    "transition_top5_qa_v1_specificitypair_w12.md"
)


class ReaderOnlySystem:
    """Minimal system surface needed for HippoRAG's QA prompt path."""

    def __init__(self, global_config, *, qwen_disable_thinking: bool = False):
        self.global_config = global_config
        self.llm_model = _get_llm_class(global_config)
        install_qwen_disable_thinking(
            self.llm_model,
            SimpleNamespace(qwen_disable_thinking=bool(qwen_disable_thinking)),
        )
        self.prompt_template_manager = PromptTemplateManager()

    def qa(self, query_solutions):
        all_messages = []
        qa_top_k = int(getattr(self.global_config, "qa_top_k", 5))
        for query_idx, query_solution in enumerate(query_solutions):
            if query_idx % 10 == 0:
                print(f"[reader] build prompt {query_idx + 1}/{len(query_solutions)}", flush=True)
            prompt_user = ""
            for passage in query_solution.docs[:qa_top_k]:
                prompt_user += f"Wikipedia Title: {passage}\n\n"
            prompt_user += "Question: " + query_solution.question + "\nThought: "
            if self.prompt_template_manager.is_template_name_valid(name=f"rag_qa_{self.global_config.dataset}"):
                prompt_dataset_name = self.global_config.dataset
            else:
                prompt_dataset_name = "musique"
            all_messages.append(
                self.prompt_template_manager.render(
                    name=f"rag_qa_{prompt_dataset_name}",
                    prompt_user=prompt_user,
                )
            )

        responses = []
        for query_idx, messages in enumerate(all_messages):
            if query_idx % 10 == 0:
                print(f"[reader] infer {query_idx + 1}/{len(all_messages)}", flush=True)
            responses.append(self.llm_model.infer(messages))
        response_messages, metadata, _cache_hit = zip(*responses) if responses else ([], [], [])
        answered = []
        for query_solution, response_content in zip(query_solutions, response_messages):
            try:
                pred_ans = str(response_content).split("Answer:", 1)[1].strip()
            except Exception:
                pred_ans = str(response_content)
            query_solution.answer = pred_ans
            answered.append(query_solution)
        return answered, list(response_messages), list(metadata)


def parse_csv(value: str | None, *, allowed: Iterable[str] | None = None) -> List[str]:
    if value is None:
        return []
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if allowed is not None:
        allowed_set = set(allowed)
        unknown = [item for item in items if item not in allowed_set]
        if unknown:
            raise ValueError(f"Unknown values {unknown}; allowed={sorted(allowed_set)}")
    return items


def normalize_context_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def content_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_context_text(value).split()
        if len(token) >= 4 and token not in CONTEXT_STOPWORDS
    }


def split_passage_title_body(passage: str) -> Tuple[str, str]:
    passage = str(passage or "")
    if "\n" not in passage:
        return passage.strip(), ""
    title, body = passage.split("\n", 1)
    return title.strip(), body.strip()


def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def compress_passage_for_query(
    *,
    passage: str,
    question: str,
    max_sentences: int,
    max_chars: int,
) -> str:
    """Extract query-focused sentences while preserving the Wikipedia title."""

    title, body = split_passage_title_body(passage)
    if int(max_sentences) <= 0 and int(max_chars) <= 0:
        return passage
    sentences = sentence_split(body)
    if not sentences:
        return passage[: int(max_chars)] if int(max_chars) > 0 else passage

    query_tokens = content_tokens(question)
    title_tokens = content_tokens(title)
    scored = []
    for idx, sentence in enumerate(sentences):
        sentence_tokens = content_tokens(sentence)
        overlap = len(query_tokens & sentence_tokens)
        title_overlap = len(title_tokens & sentence_tokens)
        score = float(overlap) + 0.25 * float(title_overlap)
        if idx == 0:
            score += 0.1
        scored.append((score, idx, sentence))

    top_k = max(int(max_sentences), 1)
    selected = sorted(sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k], key=lambda item: item[1])
    compressed_body = " ".join(sentence for _score, _idx, sentence in selected).strip()
    compressed = f"{title}\n{compressed_body}".strip() if title else compressed_body
    if int(max_chars) > 0 and len(compressed) > int(max_chars):
        compressed = compressed[: int(max_chars)].rsplit(" ", 1)[0].strip()
    return compressed or passage


def prepare_reader_doc(
    *,
    passage: str,
    question: str,
    context_mode: str,
    context_max_sentences: int,
    context_max_chars: int,
) -> str:
    if context_mode == "full":
        return passage
    if context_mode == "query_sentences":
        return compress_passage_for_query(
            passage=passage,
            question=question,
            max_sentences=context_max_sentences,
            max_chars=context_max_chars,
        )
    raise ValueError(f"Unsupported context_mode={context_mode!r}")


def load_sfb_rows(source_report_path: Path) -> Dict[int, Mapping[str, Any]]:
    source_report = load_json(source_report_path)
    rows = source_report.get("variants", {}).get(SFB_VARIANT)
    if rows is None:
        raise KeyError(f"Missing SFB variant {SFB_VARIANT!r} in {source_report_path}")
    return {int(row["query_index"]): row for row in rows}


def method_doc_indices(
    *,
    method: str,
    transition_row: Mapping[str, Any],
    sfb_row: Mapping[str, Any],
    qa_top_k: int,
    fusion_sfb_weight: float = 2.0,
    fusion_specificity_weight: float = 3.0,
    fusion_sfb_depth: int = 20,
    fusion_specificity_depth: int = 20,
    fusion_stable_prefix_k: int = 0,
    fusion_support_weight: float = 0.0,
    fusion_support_overlap_bonus: float = 0.0,
    fusion_support_depth: int = 10,
) -> List[int]:
    if method == "sfb_context":
        values = (
            sfb_row.get("reader_doc_indices_topk")
            or sfb_row.get("retrieved_doc_indices_top5")
            or []
        )
        return unique_ints(values, limit=qa_top_k)
    if method in FUSION_METHOD_BASE_CHANNELS:
        support_weight = 0.0 if method == "channel_fusion" else fusion_support_weight
        support_overlap_bonus = 0.0 if method == "channel_fusion" else fusion_support_overlap_bonus
        return channel_fusion_doc_indices(
            transition_row=transition_row,
            sfb_row=sfb_row,
            qa_top_k=qa_top_k,
            fusion_sfb_weight=fusion_sfb_weight,
            fusion_specificity_weight=fusion_specificity_weight,
            fusion_sfb_depth=fusion_sfb_depth,
            fusion_specificity_depth=fusion_specificity_depth,
            fusion_stable_prefix_k=fusion_stable_prefix_k,
            fusion_support_weight=support_weight,
            fusion_support_overlap_bonus=support_overlap_bonus,
            fusion_support_depth=fusion_support_depth,
            fusion_base_channel=FUSION_METHOD_BASE_CHANNELS[method],
        )
    if method == "anchor_guided_evidence":
        evidence_payload = transition_row.get("anchor_guided_evidence", {})
        selected_set = (
            evidence_payload.get("selected_evidence_set", {})
            if isinstance(evidence_payload, Mapping)
            else {}
        )
        values = (
            transition_row.get("anchor_guided_evidence_doc_indices_top5")
            or selected_set.get("doc_indices", [])
            or (evidence_payload.get("retrieved_doc_indices", []) if isinstance(evidence_payload, Mapping) else [])
            or []
        )
        return unique_ints(values, limit=qa_top_k)
    key = METHOD_DOC_KEYS[method]
    return unique_ints(transition_row.get(key, []) or [], limit=qa_top_k)


def base_channel_doc_indices(
    *,
    base_channel: str,
    transition_row: Mapping[str, Any],
    sfb_row: Mapping[str, Any],
    depth: int,
    qa_top_k: int,
    stable_prefix_k: int = 0,
) -> List[int]:
    if base_channel == "sfb":
        values = (
            sfb_row.get("retrieved_doc_indices_top20")
            or sfb_row.get("reader_doc_indices_topk")
            or sfb_row.get("retrieved_doc_indices_top5")
            or []
        )
    elif base_channel == "native_dense":
        values = (
            transition_row.get("native_dense_doc_indices_top10")
            or transition_row.get("context_anchor_doc_indices")
            or []
        )
    elif base_channel == "context_anchor":
        values = (
            transition_row.get("context_anchor_doc_indices")
            or transition_row.get("native_dense_doc_indices_top10")
            or []
        )
    elif base_channel == "bm25":
        values = (
            transition_row.get("bm25_doc_indices_top10")
            or transition_row.get("bm25_doc_indices_top5")
            or []
        )
    elif base_channel == "hybrid_residual":
        values = (
            transition_row.get("hybrid_residual_pair_doc_indices_top10")
            or transition_row.get("hybrid_residual_pair_doc_indices_top5")
            or []
        )
    elif base_channel == "support_set":
        values = (
            transition_row.get("support_set_doc_indices_top10")
            or transition_row.get("support_set_doc_indices_top5")
            or []
        )
    elif base_channel == "native_consensus":
        return native_consensus_base_doc_indices(
            transition_row=transition_row,
            sfb_row=sfb_row,
            depth=depth,
            qa_top_k=qa_top_k,
            stable_prefix_k=stable_prefix_k,
            include_support_channel=True,
        )
    elif base_channel == "native_spec_consensus":
        return native_consensus_base_doc_indices(
            transition_row=transition_row,
            sfb_row=sfb_row,
            depth=depth,
            qa_top_k=qa_top_k,
            stable_prefix_k=stable_prefix_k,
            include_support_channel=False,
        )
    elif base_channel == "native_anchor_neighborhood":
        return native_anchor_neighborhood_base_doc_indices(
            transition_row=transition_row,
            sfb_row=sfb_row,
            depth=depth,
            qa_top_k=qa_top_k,
            stable_prefix_k=stable_prefix_k,
        )
    else:
        raise ValueError(f"Unsupported fusion base channel: {base_channel!r}")
    return unique_ints(values, limit=max(int(depth), int(qa_top_k)))


def support_set_ranked_doc_scores(
    transition_row: Mapping[str, Any],
    *,
    support_depth: int,
) -> Dict[int, Tuple[float, int]]:
    support_sets = (
        transition_row.get("support_set_search", {}).get("top_support_sets", [])
        if isinstance(transition_row.get("support_set_search", {}), Mapping)
        else []
    )
    doc_scores: Dict[int, Tuple[float, int]] = {}
    for support_rank, support_set in enumerate(list(support_sets)[: max(0, int(support_depth))], start=1):
        support_docs = unique_ints(support_set.get("doc_indices", []) or [], limit=10)
        if not support_docs:
            continue
        raw_score = max(float(support_set.get("score", 0.0) or 0.0), 0.0)
        normalized_score = min(raw_score / 250.0, 2.0) / math.sqrt(float(support_rank))
        for doc_idx in support_docs:
            current_score, current_rank = doc_scores.get(int(doc_idx), (0.0, 10**9))
            doc_scores[int(doc_idx)] = (
                max(float(current_score), float(normalized_score)),
                min(int(current_rank), int(support_rank)),
            )
    return doc_scores


def native_anchor_neighborhood_base_doc_indices(
    *,
    transition_row: Mapping[str, Any],
    sfb_row: Mapping[str, Any],
    depth: int,
    qa_top_k: int,
    stable_prefix_k: int,
) -> List[int]:
    """Rank a native mid-slate around stable dense anchors.

    This is a native replacement for the SFB base slate, not another SFB patch:
    the prefix comes from dense/context anchors and the tail is scored by
    source/title/OpenIE neighborhood evidence already emitted in the transition
    report. Candidate pages get meaningful credit when they are connected to a
    protected anchor through direct boundary completions. Support-set
    evidence is intentionally left to the later assembly stage; including it
    here double-counts weak support pages and can displace native gold pages.
    """

    effective_depth = max(int(depth), int(qa_top_k))
    native_docs = base_channel_doc_indices(
        base_channel="native_dense",
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=effective_depth,
        qa_top_k=qa_top_k,
    )
    if not native_docs:
        return []

    protected_k = min(max(int(stable_prefix_k), 2), len(native_docs), max(effective_depth, 1))
    protected_prefix = [int(doc_idx) for doc_idx in native_docs[:protected_k]]
    protected_set = set(protected_prefix)

    scores: Dict[int, float] = {}
    best_rank: Dict[int, int] = {}
    candidates: List[int] = []
    seen_candidates: set[int] = set()

    def add_candidate(doc_idx: int, score: float, rank_hint: int) -> None:
        doc_idx = int(doc_idx)
        if doc_idx < 0:
            return
        if doc_idx not in seen_candidates:
            seen_candidates.add(doc_idx)
            candidates.append(doc_idx)
        scores[doc_idx] = scores.get(doc_idx, 0.0) + float(score)
        best_rank[doc_idx] = min(best_rank.get(doc_idx, 10**9), int(rank_hint))

    for rank, doc_idx in enumerate(native_docs, start=1):
        native_weight = 1.1 if rank <= 5 else 0.45
        add_candidate(int(doc_idx), native_weight / math.sqrt(float(rank)), rank)

    neighborhood = transition_row.get("query_conditioned_neighborhood", {})
    if not isinstance(neighborhood, Mapping):
        neighborhood = {}

    for pair_rank, pair in enumerate(neighborhood.get("boundary_readout_pairs", []) or [], start=1):
        if not isinstance(pair, Mapping):
            continue
        anchor_doc = pair.get("anchor_doc")
        completion_doc = pair.get("completion_doc")
        if anchor_doc is None or completion_doc is None:
            continue
        if int(anchor_doc) not in protected_set:
            continue
        direct_score = max(float(pair.get("direct_transition_score", 0.0) or 0.0), 0.0)
        skipped_endpoint_count = max(int(pair.get("direct_skipped_endpoint_count", 0) or 0), 0)
        if direct_score < 5.0 or skipped_endpoint_count > 3:
            continue
        boundary_score = max(float(pair.get("score", 0.0) or 0.0), 0.0)
        endpoint_count = max(int(pair.get("direct_endpoint_count", 0) or 0), 0)
        selected_by_neighborhood = bool(pair.get("selected_by_neighborhood"))
        selected_bonus = 0.35 if selected_by_neighborhood else 0.0
        score = (
            0.75 * math.log1p(direct_score)
            + 0.10 * math.log1p(boundary_score)
            + 0.12 * math.log1p(float(endpoint_count))
            + selected_bonus
        )
        add_candidate(int(completion_doc), score, 40 + pair_rank)

    ranked_tail = [
        doc_idx
        for doc_idx in sorted(
            candidates,
            key=lambda doc_idx: (
                -float(scores.get(int(doc_idx), 0.0)),
                int(best_rank.get(int(doc_idx), 10**9)),
                int(doc_idx),
            ),
        )
        if int(doc_idx) not in protected_set
    ]

    output = unique_ints(protected_prefix + ranked_tail + native_docs, limit=effective_depth)
    return output


def native_consensus_base_doc_indices(
    *,
    transition_row: Mapping[str, Any],
    sfb_row: Mapping[str, Any],
    depth: int,
    qa_top_k: int,
    stable_prefix_k: int,
    include_support_channel: bool = True,
) -> List[int]:
    """Build a native base slate from multi-view agreement, without SFB ranks.

    The prefix remains native dense/context anchors. Ranks after the protected
    prefix are selected by channel consensus across native dense, STO
    specificity, and support-set membership. This targets the observed gap:
    native and SFB agree on top2, but native loses useful ranks 3-10 evidence.
    """

    effective_depth = max(int(depth), int(qa_top_k))
    native_docs = base_channel_doc_indices(
        base_channel="native_dense",
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=effective_depth,
        qa_top_k=qa_top_k,
    )
    specificity_emission = transition_row.get("specificity_pairwise_transition_emission", {}) or {}
    specificity_docs = unique_ints(
        specificity_emission.get("retrieved_doc_indices")
        or transition_row.get("specificity_pair_doc_indices_top10")
        or transition_row.get("specificity_pair_doc_indices_top5")
        or [],
        limit=effective_depth,
    )
    support_scores = (
        support_set_ranked_doc_scores(
            transition_row,
            support_depth=effective_depth,
        )
        if bool(include_support_channel)
        else {}
    )

    candidates: List[int] = []
    seen: set[int] = set()
    for sequence in (native_docs, specificity_docs, list(support_scores.keys())):
        for doc_idx in sequence:
            doc_idx = int(doc_idx)
            if doc_idx in seen:
                continue
            seen.add(doc_idx)
            candidates.append(doc_idx)

    scores: Dict[int, float] = {}
    best_rank: Dict[int, int] = {}

    def add_ranked_channel(docs: Sequence[int], *, weight: float, rank_offset: int = 0) -> None:
        for rank, doc_idx in enumerate(docs, start=1):
            doc_idx = int(doc_idx)
            scores[doc_idx] = scores.get(doc_idx, 0.0) + float(weight) / float(rank + 1)
            best_rank[doc_idx] = min(best_rank.get(doc_idx, 10**9), int(rank_offset) + rank)

    add_ranked_channel(native_docs, weight=1.0, rank_offset=0)
    add_ranked_channel(specificity_docs, weight=1.0, rank_offset=20)
    for doc_idx, (support_score, support_rank) in support_scores.items():
        doc_idx = int(doc_idx)
        scores[doc_idx] = scores.get(doc_idx, 0.0) + float(support_score)
        best_rank[doc_idx] = min(best_rank.get(doc_idx, 10**9), 40 + int(support_rank))

    native_set = set(native_docs)
    specificity_set = set(specificity_docs)
    support_set = set(support_scores)
    for doc_idx in candidates:
        channel_count = int(doc_idx in native_set) + int(doc_idx in specificity_set) + int(doc_idx in support_set)
        if channel_count >= 2:
            scores[int(doc_idx)] = scores.get(int(doc_idx), 0.0) + 0.25 * float(channel_count - 1)

    prefix_count = min(max(int(stable_prefix_k), 0), len(native_docs), int(qa_top_k))
    output: List[int] = []
    output_seen: set[int] = set()
    for doc_idx in native_docs[:prefix_count]:
        output.append(int(doc_idx))
        output_seen.add(int(doc_idx))

    ranked_tail = sorted(
        candidates,
        key=lambda doc_idx: (
            -float(scores.get(int(doc_idx), 0.0)),
            int(best_rank.get(int(doc_idx), 10**9)),
            int(doc_idx),
        ),
    )
    for doc_idx in ranked_tail:
        doc_idx = int(doc_idx)
        if doc_idx in output_seen:
            continue
        output.append(doc_idx)
        output_seen.add(doc_idx)
        if len(output) >= effective_depth:
            break
    return output[:effective_depth]


def channel_fusion_doc_indices(
    *,
    transition_row: Mapping[str, Any],
    sfb_row: Mapping[str, Any],
    qa_top_k: int,
    fusion_sfb_weight: float,
    fusion_specificity_weight: float,
    fusion_sfb_depth: int,
    fusion_specificity_depth: int,
    fusion_stable_prefix_k: int,
    fusion_support_weight: float,
    fusion_support_overlap_bonus: float,
    fusion_support_depth: int,
    fusion_base_channel: str = "sfb",
) -> List[int]:
    """Fuse a base channel and STO-specificity channel without gold/answer access.

    This is a conservative evidence assembly objective, not a new retrieval
    expansion. It scores each candidate by reciprocal-rank support from the
    base channel and the transition-specificity channel, then emits the top
    reader-facing passages. SFB is the default base channel; native variants
    replace it with upstream non-SFB candidates.
    """

    base_docs = base_channel_doc_indices(
        base_channel=fusion_base_channel,
        transition_row=transition_row,
        sfb_row=sfb_row,
        depth=fusion_sfb_depth,
        qa_top_k=qa_top_k,
        stable_prefix_k=fusion_stable_prefix_k,
    )
    specificity_emission = transition_row.get("specificity_pairwise_transition_emission", {}) or {}
    specificity_docs = unique_ints(
        specificity_emission.get("retrieved_doc_indices")
        or transition_row.get("specificity_pair_doc_indices_top10")
        or transition_row.get("specificity_pair_doc_indices_top5")
        or [],
        limit=max(int(fusion_specificity_depth), qa_top_k),
    )

    candidates: List[int] = []
    seen: set[int] = set()
    for doc_idx in list(base_docs) + list(specificity_docs):
        if int(doc_idx) in seen:
            continue
        seen.add(int(doc_idx))
        candidates.append(int(doc_idx))

    scores: Dict[int, float] = {}
    best_channel_rank: Dict[int, int] = {}
    for rank, doc_idx in enumerate(base_docs, start=1):
        scores[int(doc_idx)] = scores.get(int(doc_idx), 0.0) + float(fusion_sfb_weight) / float(rank + 1)
        best_channel_rank[int(doc_idx)] = min(best_channel_rank.get(int(doc_idx), 10**9), rank)
    for rank, doc_idx in enumerate(specificity_docs, start=1):
        scores[int(doc_idx)] = scores.get(int(doc_idx), 0.0) + float(fusion_specificity_weight) / float(rank + 1)
        best_channel_rank[int(doc_idx)] = min(best_channel_rank.get(int(doc_idx), 10**9), rank)

    if float(fusion_support_weight) > 0.0:
        prefix_set = set(base_docs[: max(0, int(fusion_stable_prefix_k))])
        support_sets = (
            transition_row.get("support_set_search", {}).get("top_support_sets", [])
            if isinstance(transition_row.get("support_set_search", {}), Mapping)
            else []
        )
        for support_rank, support_set in enumerate(list(support_sets)[: max(0, int(fusion_support_depth))], start=1):
            support_docs = unique_ints(support_set.get("doc_indices", []) or [], limit=10)
            if not support_docs:
                continue
            raw_support_score = float(support_set.get("score", 0.0) or 0.0)
            support_credit = float(fusion_support_weight) * (raw_support_score / 250.0) / math.sqrt(float(support_rank))
            overlaps_prefix = bool(prefix_set & set(support_docs))
            for doc_idx in support_docs:
                if int(doc_idx) not in seen:
                    seen.add(int(doc_idx))
                    candidates.append(int(doc_idx))
                scores[int(doc_idx)] = scores.get(int(doc_idx), 0.0) + support_credit
                if overlaps_prefix:
                    scores[int(doc_idx)] += float(fusion_support_overlap_bonus)
                best_channel_rank[int(doc_idx)] = min(best_channel_rank.get(int(doc_idx), 10**9), 20 + support_rank)

    ranked = sorted(
        candidates,
        key=lambda doc_idx: (
            -float(scores.get(int(doc_idx), 0.0)),
            int(best_channel_rank.get(int(doc_idx), 10**9)),
            int(doc_idx),
        ),
    )
    stable_prefix: List[int] = []
    seen_prefix: set[int] = set()
    for doc_idx in base_docs[: max(0, int(fusion_stable_prefix_k))]:
        if int(doc_idx) in seen_prefix:
            continue
        stable_prefix.append(int(doc_idx))
        seen_prefix.add(int(doc_idx))
        if len(stable_prefix) >= qa_top_k:
            return stable_prefix[:qa_top_k]

    output = list(stable_prefix)
    seen_output = set(output)
    for doc_idx in ranked:
        if int(doc_idx) in seen_output:
            continue
        output.append(int(doc_idx))
        seen_output.add(int(doc_idx))
        if len(output) >= qa_top_k:
            break
    return output[:qa_top_k]


def safe_gold_indices(row: Mapping[str, Any]) -> List[int]:
    result = []
    for value in row.get("gold_doc_indices", []) or []:
        if value is None:
            continue
        result.append(int(value))
    return result


def top5_failure_bucket(row: Mapping[str, Any], exact_match: float | None) -> str:
    if exact_match is None:
        return "qa_not_run"
    if float(exact_match) >= 1.0:
        if bool(row.get("all_gold_at5")):
            return "qa_exact_all_gold_top5"
        if bool(row.get("any_gold_at5")):
            return "qa_exact_partial_gold_top5"
        return "qa_exact_no_gold_top5"
    if not bool(row.get("any_gold_at5")):
        return "no_gold_top5_qa_wrong"
    if bool(row.get("all_gold_at5")):
        return "all_gold_top5_qa_wrong"
    return "partial_gold_top5_qa_wrong"


def make_reader_config(
    *,
    dataset_name: str,
    source_report_config: Mapping[str, Any],
    corpus_len: int,
    args: argparse.Namespace,
    config_cls,
):
    llm_name = args.llm_name or source_report_config.get("effective_qa_reader_llm_name") or source_report_config.get("llm_name")
    llm_base_url = (
        args.llm_base_url
        or source_report_config.get("effective_qa_reader_llm_base_url")
        or source_report_config.get("llm_base_url")
    )
    embedding_name = args.embedding_name or source_report_config.get("embedding_name") or "/mnt/nvme/Qwen3-Embedding-8B"
    embedding_base_url = args.embedding_base_url or source_report_config.get("embedding_base_url")
    minimal_args = SimpleNamespace(
        save_dir=args.save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        embedding_name=embedding_name,
        embedding_base_url=embedding_base_url,
        force_index_from_scratch="False",
        force_openie_from_scratch="False",
        openie_mode=args.openie_mode,
        retrieval_top_k=args.qa_top_k,
        qa_top_k=args.qa_top_k,
        max_new_tokens=args.max_new_tokens,
        qwen_disable_thinking=bool(args.qwen_disable_thinking),
        embedding_batch_size=args.embedding_batch_size,
    )
    return make_config(
        minimal_args,
        dataset_name,
        args.save_dir,
        corpus_len,
        config_cls=config_cls,
    )


def build_method_query_solutions(
    *,
    dataset_payload: Mapping[str, Any],
    method: str,
    qa_top_k: int,
    max_queries: int,
    fusion_sfb_weight: float,
    fusion_specificity_weight: float,
    fusion_sfb_depth: int,
    fusion_specificity_depth: int,
    fusion_stable_prefix_k: int,
    fusion_support_weight: float,
    fusion_support_overlap_bonus: float,
    fusion_support_depth: int,
    context_mode: str,
    context_max_sentences: int,
    context_max_chars: int,
    query_solution_cls,
) -> Tuple[List[Any], List[List[str]], List[Dict[str, Any]], List[Dict[str, Any]], List[Mapping[str, Any]]]:
    openie_docs = list(load_json(Path(str(dataset_payload["openie_path"]))).get("docs", []) or [])
    sfb_rows = load_sfb_rows(Path(str(dataset_payload["report_path"])))
    transition_rows = list(dataset_payload.get("rows", []) or [])
    if int(max_queries) > 0:
        transition_rows = transition_rows[: int(max_queries)]

    query_solutions: List[Any] = []
    gold_answers: List[List[str]] = []
    evidence_rows: List[Dict[str, Any]] = []
    per_query_base: List[Dict[str, Any]] = []
    source_rows: List[Mapping[str, Any]] = []

    for transition_row in transition_rows:
        query_index = int(transition_row["query_index"])
        sfb_row = sfb_rows[query_index]
        doc_indices = method_doc_indices(
            method=method,
            transition_row=transition_row,
            sfb_row=sfb_row,
            qa_top_k=qa_top_k,
            fusion_sfb_weight=fusion_sfb_weight,
            fusion_specificity_weight=fusion_specificity_weight,
            fusion_sfb_depth=fusion_sfb_depth,
            fusion_specificity_depth=fusion_specificity_depth,
            fusion_stable_prefix_k=fusion_stable_prefix_k,
            fusion_support_weight=fusion_support_weight,
            fusion_support_overlap_bonus=fusion_support_overlap_bonus,
            fusion_support_depth=fusion_support_depth,
        )
        query = str(transition_row.get("question") or sfb_row.get("question") or "")
        selected_docs = [
            prepare_reader_doc(
                passage=passage_for_doc(openie_docs, doc_idx),
                question=query,
                context_mode=context_mode,
                context_max_sentences=context_max_sentences,
                context_max_chars=context_max_chars,
            )
            for doc_idx in doc_indices
        ]
        selected_docs = [doc for doc in selected_docs if doc]
        doc_scores = np.asarray([1.0 / float(rank + 1) for rank in range(len(selected_docs))], dtype=np.float32)
        answers = [str(answer) for answer in sfb_row.get("gold_answers", []) or []]
        merged_row = dict(transition_row)
        merged_row["gold_answers"] = answers
        merged_row["gold_doc_indices"] = safe_gold_indices(transition_row)
        top5_eval = evaluate_top5_row(
            row=merged_row,
            method=method,
            doc_indices=doc_indices,
            openie_docs=openie_docs,
        )
        query_solutions.append(
            query_solution_cls(
                question=query,
                docs=selected_docs,
                doc_scores=doc_scores,
                gold_answers=answers,
                gold_docs=[passage_for_doc(openie_docs, idx) for idx in merged_row["gold_doc_indices"]],
            )
        )
        gold_answers.append(answers)
        evidence_rows.append(top5_eval)
        per_query_base.append(
            {
                "query_index": query_index,
                "question": query,
                "gold_answers": answers,
                "gold_doc_indices": merged_row["gold_doc_indices"],
                "doc_indices_top5": top5_eval["doc_indices_top5"],
                "titles_top5": [title_for_doc(openie_docs, idx) for idx in top5_eval["doc_indices_top5"]],
                "recall_at5": top5_eval["recall_at5"],
                "gold_count_at5": top5_eval["gold_count_at5"],
                "any_gold_at5": top5_eval["any_gold_at5"],
                "all_gold_at5": top5_eval["all_gold_at5"],
                "answer_string_hit_at5": top5_eval["answer_string_hit_at5"],
            }
        )
        source_rows.append(transition_row)
    return query_solutions, gold_answers, evidence_rows, per_query_base, source_rows


def evaluate_method_qa(
    *,
    system: ReaderOnlySystem,
    query_solutions: Sequence[Any],
    gold_answers: Sequence[Sequence[str]],
    qa_em_cls,
    qa_f1_cls,
    skip_qa: bool,
) -> Dict[str, Any]:
    if skip_qa:
        return {
            "predicted_answers": [None for _ in query_solutions],
            "responses": [],
            "metadata": [],
            "qa_metrics": {},
            "example_em": [None for _ in query_solutions],
            "example_f1": [None for _ in query_solutions],
        }
    answered_solutions, responses, metadata = system.qa(list(query_solutions))
    predicted_answers = [str(solution.answer or "") for solution in answered_solutions]
    qa_em = qa_em_cls(global_config=system.global_config)
    qa_f1 = qa_f1_cls(global_config=system.global_config)
    em_results, em_examples = qa_em.calculate_metric_scores(
        gold_answers=[list(items) for items in gold_answers],
        predicted_answers=predicted_answers,
    )
    f1_results, f1_examples = qa_f1.calculate_metric_scores(
        gold_answers=[list(items) for items in gold_answers],
        predicted_answers=predicted_answers,
    )
    return {
        "predicted_answers": predicted_answers,
        "responses": responses,
        "metadata": metadata,
        "qa_metrics": {
            "ExactMatch": round(float(em_results.get("ExactMatch", 0.0)), 4),
            "F1": round(float(f1_results.get("F1", 0.0)), 4),
        },
        "example_em": [float(item["ExactMatch"]) for item in em_examples],
        "example_f1": [float(item["F1"]) for item in f1_examples],
    }


def summarize_method_result(
    *,
    method: str,
    evidence_rows: Sequence[Mapping[str, Any]],
    per_query_base: Sequence[Mapping[str, Any]],
    qa_result: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = dict(aggregate_method_rows(evidence_rows))
    metrics.update(qa_result.get("qa_metrics", {}))
    predicted_answers = list(qa_result.get("predicted_answers", []) or [])
    example_em = list(qa_result.get("example_em", []) or [])
    example_f1 = list(qa_result.get("example_f1", []) or [])
    per_query: List[Dict[str, Any]] = []
    buckets: Counter[str] = Counter()
    for idx, base_row in enumerate(per_query_base):
        em_value = example_em[idx] if idx < len(example_em) else None
        f1_value = example_f1[idx] if idx < len(example_f1) else None
        row = dict(base_row)
        row["predicted_answer"] = predicted_answers[idx] if idx < len(predicted_answers) else None
        row["ExactMatch"] = None if em_value is None else round(float(em_value), 4)
        row["F1"] = None if f1_value is None else round(float(f1_value), 4)
        row["qa_bucket"] = top5_failure_bucket(row, em_value)
        buckets[row["qa_bucket"]] += 1
        per_query.append(row)

    return {
        "method": method,
        "metrics": metrics,
        "qa_buckets": dict(buckets),
        "per_query": per_query,
    }


def qa_deltas(method_results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    if "sfb_context" not in method_results:
        return {}
    base_metrics = method_results["sfb_context"].get("metrics", {})
    deltas: Dict[str, Dict[str, float]] = {}
    for method, result in method_results.items():
        if method == "sfb_context":
            continue
        metrics = result.get("metrics", {})
        deltas[method] = {
            "delta_r5": round(float(metrics.get("r5", 0.0)) - float(base_metrics.get("r5", 0.0)), 6),
            "delta_all_gold_at5": round(
                float(metrics.get("all_gold_at5", 0.0)) - float(base_metrics.get("all_gold_at5", 0.0)),
                6,
            ),
            "delta_answer_string_at5": round(
                float(metrics.get("answer_string_hit_at5", 0.0)) - float(base_metrics.get("answer_string_hit_at5", 0.0)),
                6,
            ),
            "delta_exact_match": round(
                float(metrics.get("ExactMatch", 0.0)) - float(base_metrics.get("ExactMatch", 0.0)),
                6,
            ),
            "delta_f1": round(float(metrics.get("F1", 0.0)) - float(base_metrics.get("F1", 0.0)), 6),
        }
    return deltas


def evaluate_dataset(
    *,
    dataset_payload: Mapping[str, Any],
    methods: Sequence[str],
    args: argparse.Namespace,
    runtime,
) -> Dict[str, Any]:
    dataset_name = str(dataset_payload["dataset"])
    source_report = load_json(Path(str(dataset_payload["report_path"])))
    source_config = source_report.get("config", {}) or {}
    openie_docs = list(load_json(Path(str(dataset_payload["openie_path"]))).get("docs", []) or [])
    config = make_reader_config(
        dataset_name=dataset_name,
        source_report_config=source_config,
        corpus_len=len(openie_docs),
        args=args,
        config_cls=runtime.BaseConfigCls,
    )
    system = None if bool(args.skip_qa) else ReaderOnlySystem(
        global_config=config,
        qwen_disable_thinking=bool(args.qwen_disable_thinking),
    )

    method_results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        print(f"[dataset={dataset_name}] method={method}", flush=True)
        query_solutions, answers, evidence_rows, per_query_base, _source_rows = build_method_query_solutions(
            dataset_payload=dataset_payload,
            method=method,
            qa_top_k=int(args.qa_top_k),
            max_queries=int(args.max_queries),
            fusion_sfb_weight=float(args.fusion_sfb_weight),
            fusion_specificity_weight=float(args.fusion_specificity_weight),
            fusion_sfb_depth=int(args.fusion_sfb_depth),
            fusion_specificity_depth=int(args.fusion_specificity_depth),
            fusion_stable_prefix_k=int(args.fusion_stable_prefix_k),
            fusion_support_weight=float(args.fusion_support_weight),
            fusion_support_overlap_bonus=float(args.fusion_support_overlap_bonus),
            fusion_support_depth=int(args.fusion_support_depth),
            context_mode=str(args.context_mode),
            context_max_sentences=int(args.context_max_sentences),
            context_max_chars=int(args.context_max_chars),
            query_solution_cls=runtime.QuerySolutionCls,
        )
        qa_result = evaluate_method_qa(
            system=system,
            query_solutions=query_solutions,
            gold_answers=answers,
            qa_em_cls=runtime.QAExactMatchCls,
            qa_f1_cls=runtime.QAF1ScoreCls,
            skip_qa=bool(args.skip_qa),
        )
        method_results[method] = summarize_method_result(
            method=method,
            evidence_rows=evidence_rows,
            per_query_base=per_query_base,
            qa_result=qa_result,
        )

    return {
        "dataset": dataset_name,
        "num_queries": len(next(iter(method_results.values()))["per_query"]) if method_results else 0,
        "reader": {
            "llm_name": getattr(config, "llm_name", None),
            "llm_base_url": getattr(config, "llm_base_url", None),
            "qa_top_k": getattr(config, "qa_top_k", None),
        },
        "methods": method_results,
        "deltas_vs_sfb": qa_deltas(method_results),
    }


def write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Transition Top5 QA",
        "",
        "This report evaluates fixed reader-facing top5 evidence. R@10/R@200 are not optimization targets here.",
        "",
        "## Metrics",
        "",
        "| dataset | method | R@5 | all-gold@5 | answer-string@5 | EM | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in payload.get("datasets", []) or []:
        for method, result in dataset.get("methods", {}).items():
            metrics = result.get("metrics", {})
            lines.append(
                f"| {dataset['dataset']} | {method} | "
                f"{float(metrics.get('r5', 0.0)):.4f} | "
                f"{float(metrics.get('all_gold_at5', 0.0)):.4f} | "
                f"{float(metrics.get('answer_string_hit_at5', 0.0)):.4f} | "
                f"{float(metrics.get('ExactMatch', 0.0)):.4f} | "
                f"{float(metrics.get('F1', 0.0)):.4f} |"
            )

    lines.extend(["", "## Delta Vs SFB", ""])
    lines.append("| dataset | method | delta R@5 | delta all-gold@5 | delta answer-string@5 | delta EM | delta F1 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset in payload.get("datasets", []) or []:
        for method, deltas in dataset.get("deltas_vs_sfb", {}).items():
            lines.append(
                f"| {dataset['dataset']} | {method} | "
                f"{float(deltas.get('delta_r5', 0.0)):+.4f} | "
                f"{float(deltas.get('delta_all_gold_at5', 0.0)):+.4f} | "
                f"{float(deltas.get('delta_answer_string_at5', 0.0)):+.4f} | "
                f"{float(deltas.get('delta_exact_match', 0.0)):+.4f} | "
                f"{float(deltas.get('delta_f1', 0.0)):+.4f} |"
            )

    lines.extend(["", "## QA Failure Buckets", ""])
    for dataset in payload.get("datasets", []) or []:
        lines.append(f"### {dataset['dataset']}")
        lines.append("")
        lines.append("| method | bucket | count |")
        lines.append("| --- | --- | ---: |")
        for method, result in dataset.get("methods", {}).items():
            for bucket, count in sorted(result.get("qa_buckets", {}).items()):
                lines.append(f"| {method} | {bucket} | {count} |")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_transition_top5_qa(args: argparse.Namespace) -> Dict[str, Any]:
    transition_report = load_json(Path(args.transition_report))
    selected_datasets = set(parse_csv(args.datasets)) if args.datasets else set()
    methods = parse_csv(args.methods, allowed=METHOD_DOC_KEYS)
    runtime = get_worktree_baseline_runtime()
    datasets = []
    for dataset_payload in transition_report.get("datasets", []) or []:
        dataset_name = str(dataset_payload["dataset"])
        if selected_datasets and dataset_name not in selected_datasets:
            continue
        datasets.append(
            evaluate_dataset(
                dataset_payload=dataset_payload,
                methods=methods,
                args=args,
                runtime=runtime,
            )
        )

    payload = {
        "transition_report_path": str(Path(args.transition_report).resolve()),
        "policy": "top5_reader_qa_only; R@10/R@200 not optimized",
        "skip_qa": bool(args.skip_qa),
        "methods": methods,
        "fusion_config": {
            "sfb_weight": float(args.fusion_sfb_weight),
            "specificity_weight": float(args.fusion_specificity_weight),
            "sfb_depth": int(args.fusion_sfb_depth),
            "specificity_depth": int(args.fusion_specificity_depth),
            "stable_prefix_k": int(args.fusion_stable_prefix_k),
            "support_weight": float(args.fusion_support_weight),
            "support_overlap_bonus": float(args.fusion_support_overlap_bonus),
            "support_depth": int(args.fusion_support_depth),
            "base_channel_by_method": {
                method: FUSION_METHOD_BASE_CHANNELS[method]
                for method in methods
                if method in FUSION_METHOD_BASE_CHANNELS
            },
        },
        "context_config": {
            "mode": str(args.context_mode),
            "max_sentences": int(args.context_max_sentences),
            "max_chars": int(args.context_max_chars),
        },
        "datasets": datasets,
    }
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, output_md)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reader QA on fixed transition top5 evidence.")
    parser.add_argument("--transition-report", default=DEFAULT_TRANSITION_REPORT)
    parser.add_argument("--datasets", default="musique,2wikimultihopqa,hotpotqa")
    parser.add_argument("--methods", default="sfb_context,semantic_pair,specificity_pair")
    parser.add_argument("--fusion-sfb-weight", type=float, default=2.0)
    parser.add_argument("--fusion-specificity-weight", type=float, default=3.0)
    parser.add_argument("--fusion-sfb-depth", type=int, default=20)
    parser.add_argument("--fusion-specificity-depth", type=int, default=20)
    parser.add_argument("--fusion-stable-prefix-k", type=int, default=0)
    parser.add_argument("--fusion-support-weight", type=float, default=0.0)
    parser.add_argument("--fusion-support-overlap-bonus", type=float, default=0.0)
    parser.add_argument("--fusion-support-depth", type=int, default=10)
    parser.add_argument("--context-mode", default="full", choices=["full", "query_sentences"])
    parser.add_argument("--context-max-sentences", type=int, default=3)
    parser.add_argument("--context-max-chars", type=int, default=900)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--save-dir", default="/mnt/nvme/zly/HippoRAG/outputs_activation_fact_object_limit100_source_20260422")
    parser.add_argument("--llm-name", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument(
        "--qwen-disable-thinking",
        action="store_true",
        help="Pass chat_template_kwargs.enable_thinking=false to Qwen OpenAI-compatible reader calls.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--embedding-name", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--openie-mode", default="offline")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)

    payload = run_transition_top5_qa(args)
    compact = {
        dataset["dataset"]: {
            method: result["metrics"]
            for method, result in dataset.get("methods", {}).items()
        }
        for dataset in payload.get("datasets", []) or []
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
