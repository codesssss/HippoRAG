#!/usr/bin/env python3
"""End-to-end QA comparison: clean GBC vs GBC+RAS(LLM) on smoke sets."""
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_causal_qwen3 import (
    build_config,
    build_doc_text_to_chunk_id,
    collect_lexical_query_seed_entities,
    collect_query_seed_entities,
    collect_question_query_entities,
    extract_doc_title,
    get_gold_answers,
)
from run_strongest_shadow_smoke import _resolve_local_llm_runtime
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag_ext.strongest.shadow_entry import run_strongest_shadow_for_pool
from src.hipporag_ext.strongest.types import StrongestConfig
from src.hipporag.utils.misc_utils import extract_answer_from_response, strip_reasoning_content


QA_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, analyze the passages and answer the question "
    "using only the provided evidence. Keep the reasoning brief. Your final line must be exactly "
    'in the format "Answer: <short answer>". Do not output <think> tags or hidden reasoning. /no_think'
)


def _is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    try:
        parsed = urlparse(str(base_url))
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def normalize_answer(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def compute_f1(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(gold: str, predicted: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(predicted))


def score_answer(gold_answers: list, predicted: str) -> dict:
    f1s = [compute_f1(g, predicted) for g in gold_answers]
    ems = [compute_em(g, predicted) for g in gold_answers]
    return {"f1": max(f1s), "em": max(ems)}


def _build_qa_messages(question: str, docs: list, top_k: int = 5) -> list[dict[str, str]]:
    prompt_user = ""
    for passage in docs[:top_k]:
        prompt_user += f"Wikipedia Title: {passage}\n\n"
    prompt_user += (
        f'Question: {question}\n'
        'Thought: Keep the reasoning brief and end with exactly one final line in the format '
        '"Answer: <short answer>". /no_think'
    )
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_user},
    ]


def ask_qa(client, model_name: str, question: str, docs: list, top_k: int = 5) -> tuple[str, dict]:
    messages = _build_qa_messages(question, docs, top_k)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=192,
            temperature=0.0,
        )
        raw = str(response.choices[0].message.content or "").strip()
        parsed_answer, parse_info = extract_answer_from_response(raw)
        cleaned_raw = strip_reasoning_content(raw)
        if parse_info["used_fallback"] and cleaned_raw != raw:
            reparsed_answer, reparsed_info = extract_answer_from_response(cleaned_raw)
            if not reparsed_info["used_fallback"] or cleaned_raw != raw:
                parsed_answer, parse_info = reparsed_answer, reparsed_info
        return parsed_answer, {
            "raw_response_preview": raw[:400],
            "cleaned_response_preview": cleaned_raw[:400],
            "answer_parser_used_fallback": bool(parse_info["used_fallback"]),
            "answer_parser_error_type": parse_info["error_type"],
            "answer_parser_response_type": parse_info["response_type"],
        }
    except Exception as e:
        print(f"  QA error: {e}", file=sys.stderr)
        return "", {
            "raw_response_preview": "",
            "cleaned_response_preview": "",
            "answer_parser_used_fallback": True,
            "answer_parser_error_type": f"qa_exception:{type(e).__name__}",
            "answer_parser_response_type": "exception",
        }


def make_args(dataset, limit, save_dir):
    class Args:
        pass
    a = Args()
    a.dataset = dataset
    a.limit = limit
    a.save_dir = save_dir
    a.llm_base_url = "http://localhost:8039/v1"
    a.llm_name = "qwen3-8b"
    a.llm_request_name = ""
    a.embedding_name = "VLLM//mnt/nvme/Qwen3-Embedding-8B"
    a.embedding_base_url = "http://localhost:8018/v1/embeddings"
    a.openie_mode = "online"
    a.causal_engine_version = "v2"
    a.causal_v2_base_retrieval_mode = "legacy_fact_graph"
    a.qa_top_k = 5
    a.retrieval_top_k = 100
    a.candidate_k = 20
    a.final_k = 10
    a.hippo_head_k = 10
    a.smoothed_union_k = 10
    a.gamma = 0.15
    a.rerank_mode = "gbc"
    a.gbc_protected_anchor_k = 2
    a.gbc_head_coverage_k = 5
    a.gbc_top_passage_pool_k = 24
    a.gbc_frontier_bonus_k = 6
    a.gbc_bonus_weight = 1.0
    for attr, val in {
        "force_index_from_scratch": "false", "force_openie_from_scratch": "false",
        "linking_top_k": 10, "max_qa_steps": 1, "embedding_batch_size": 4,
        "max_retry_attempts": 1, "planner_enabled": "false", "planner_mode": "none",
        "planner_max_steps": 0, "causal_enabled": "false", "causal_query_only": "false",
        "causal_gate_mode": "none", "causal_seed_top_k": 8,
        "causal_confidence_threshold": 0.7, "causal_damping": 0.15,
        "causal_blend_dense_weight": 1.0, "causal_blend_fact_weight": 0.0,
        "causal_blend_graph_weight": 0.0, "causal_margin_gate_enabled": "false",
        "causal_margin_threshold": 0.0, "causal_blend_top_k": 10,
        "structure_rerank_enabled": "false", "structure_rerank_top_n": 0,
        "structure_rerank_bonus_weight": 0.0, "structure_rerank_min_edge_support": 0,
        "structure_rerank_max_top5_swaps": 0, "structure_rerank_seed_top_k": 0,
        "structure_rerank_max_hops": 0, "structure_rerank_margin_threshold": 0.0,
        "rerank_require_non_empty": "true",
    }.items():
        setattr(a, attr, val)
    return a


def run_e2e(
    dataset_name,
    limit,
    qa_top_k=5,
    support_mode="lexical",
    embedding_probe_threshold=0.35,
    extractor_mode="llm_grounded",
):
    os.environ.setdefault("HIPPORAG_RERANK_FORCE_NO_THINK", "1")
    save_dir = f"outputs_step0_general_{dataset_name}"
    args = make_args(dataset_name, limit, save_dir)
    resolved_url, resolved_model = _resolve_local_llm_runtime(args)
    if resolved_url:
        args.llm_base_url = resolved_url
    args.llm_request_name = resolved_model
    print(f"LLM: {args.llm_base_url} / {args.llm_request_name}", flush=True)

    import httpx
    from openai import OpenAI
    qa_base_url = os.environ.get("RAS_LLM_BASE_URL", args.llm_base_url)
    qa_model = os.environ.get("RAS_LLM_MODEL", args.llm_request_name or args.llm_name)
    qa_client = OpenAI(
        base_url=qa_base_url,
        api_key="sk-",
        http_client=httpx.Client(trust_env=not _is_local_base_url(qa_base_url)),
    )
    print(f"QA reader: {qa_base_url} / {qa_model}", flush=True)

    corpus = json.load(open(f"reproduce/dataset/{dataset_name}_corpus.json"))
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json"))[:limit]
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    gold_answers = get_gold_answers(samples)
    queries = [s["question"] for s in samples]

    args.save_dir = save_dir
    config = build_config(args, corpus_len=len(corpus))
    hipporag = HippoRAG(
        global_config=config, save_dir=save_dir,
        llm_model_name=args.llm_name, llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name, embedding_base_url=args.embedding_base_url,
    )
    hipporag.index(docs=docs)
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=args.retrieval_top_k)

    base_cfg = StrongestConfig(
        candidate_k=args.candidate_k, final_k=args.final_k,
        hippo_head_k=args.hippo_head_k, smoothed_union_k=args.smoothed_union_k,
        gamma=args.gamma, rerank_mode="gbc",
        gbc_protected_anchor_k=args.gbc_protected_anchor_k,
        gbc_head_coverage_k=args.gbc_head_coverage_k,
        gbc_top_passage_pool_k=args.gbc_top_passage_pool_k,
        gbc_frontier_bonus_k=args.gbc_frontier_bonus_k,
        gbc_bonus_weight=args.gbc_bonus_weight,
    )
    ras_cfg = StrongestConfig(
        candidate_k=args.candidate_k, final_k=args.final_k,
        hippo_head_k=args.hippo_head_k, smoothed_union_k=args.smoothed_union_k,
        gamma=args.gamma, rerank_mode="gbc",
        gbc_protected_anchor_k=args.gbc_protected_anchor_k,
        gbc_head_coverage_k=args.gbc_head_coverage_k,
        gbc_top_passage_pool_k=args.gbc_top_passage_pool_k,
        gbc_frontier_bonus_k=args.gbc_frontier_bonus_k,
        gbc_bonus_weight=args.gbc_bonus_weight,
        ras_enabled=True, ras_prefix_guard_k=3,
        ras_requirement_max_units=4, ras_enable_conflict_veto=True,
        ras_core_support_min_eligible=True, ras_extractor_mode=str(extractor_mode),
        ras_support_mode=str(support_mode),
        ras_embedding_probe_threshold=float(embedding_probe_threshold),
    )

    base_scores = []
    ras_scores = []
    per_query = []
    base_parse_fallback_count = 0
    ras_parse_fallback_count = 0
    reused_reader_count = 0

    for i, result in enumerate(retrieval_results):
        pool_docs = list(result.docs[:max(args.final_k, args.candidate_k, qa_top_k)])
        pool_scores = np.asarray(result.doc_scores[:len(pool_docs)], dtype=np.float32)
        pool_ids = [
            hipporag.passage_node_key_to_doc_idx.get(doc_text_to_chunk_id.get(doc))
            for doc in pool_docs
        ]
        seed_ents = collect_query_seed_entities(hipporag, result.question)
        if not seed_ents:
            seed_ents = collect_lexical_query_seed_entities(
                query=result.question, pool_doc_ids=pool_ids,
                doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
            )
        query_ents = collect_question_query_entities(
            hipporag=hipporag, query=result.question,
            pool_doc_ids=pool_ids,
            doc_idx_to_entities=hipporag.doc_idx_to_structure_entities,
        )

        base_result = run_strongest_shadow_for_pool(
            hipporag=hipporag, query=result.question,
            pool_docs=pool_docs, pool_doc_ids=pool_ids,
            pool_doc_scores=pool_scores,
            seed_entities=seed_ents, query_entities=query_ents,
            config=base_cfg,
        )
        ras_result = run_strongest_shadow_for_pool(
            hipporag=hipporag, query=result.question,
            pool_docs=pool_docs, pool_doc_ids=pool_ids,
            pool_doc_scores=pool_scores,
            seed_entities=seed_ents, query_entities=query_ents,
            config=ras_cfg,
        )

        base_top = base_result.final_doc_indices.tolist()[:qa_top_k] if base_result else list(range(min(qa_top_k, len(pool_docs))))
        ras_top = ras_result.final_doc_indices.tolist()[:qa_top_k] if ras_result else base_top

        base_docs_qa = [pool_docs[idx] for idx in base_top if 0 <= idx < len(pool_docs)]
        ras_docs_qa = [pool_docs[idx] for idx in ras_top if 0 <= idx < len(pool_docs)]

        base_answer, base_qa_trace = ask_qa(qa_client, qa_model, result.question, base_docs_qa, qa_top_k)
        reused_reader = base_docs_qa == ras_docs_qa
        if reused_reader:
            ras_answer = base_answer
            ras_qa_trace = dict(base_qa_trace)
            ras_qa_trace["reused_from_base"] = True
            reused_reader_count += 1
        else:
            ras_answer, ras_qa_trace = ask_qa(qa_client, qa_model, result.question, ras_docs_qa, qa_top_k)
            ras_qa_trace["reused_from_base"] = False

        gold = gold_answers[i]
        base_s = score_answer(gold, base_answer)
        ras_s = score_answer(gold, ras_answer)
        base_scores.append(base_s)
        ras_scores.append(ras_s)
        base_parse_fallback_count += int(base_qa_trace["answer_parser_used_fallback"])
        ras_parse_fallback_count += int(ras_qa_trace["answer_parser_used_fallback"])

        changed = base_top[:5] != ras_top[:5]
        pq = {
            "query": result.question,
            "gold": gold,
            "base_answer": base_answer,
            "ras_answer": ras_answer,
            "base_raw_response_preview": base_qa_trace["raw_response_preview"],
            "ras_raw_response_preview": ras_qa_trace["raw_response_preview"],
            "base_answer_parser_used_fallback": base_qa_trace["answer_parser_used_fallback"],
            "ras_answer_parser_used_fallback": ras_qa_trace["answer_parser_used_fallback"],
            "base_answer_parser_error_type": base_qa_trace["answer_parser_error_type"],
            "ras_answer_parser_error_type": ras_qa_trace["answer_parser_error_type"],
            "reader_reused_for_ras": reused_reader,
            "base_em": base_s["em"],
            "base_f1": base_s["f1"],
            "ras_em": ras_s["em"],
            "ras_f1": ras_s["f1"],
            "top5_changed": changed,
        }
        per_query.append(pq)

        marker = " [CHANGED]" if changed else ""
        delta_f1 = ras_s["f1"] - base_s["f1"]
        delta_em = ras_s["em"] - base_s["em"]
        sign = "+" if delta_f1 >= 0 else ""
        print(f"  [{i+1}/{limit}]{marker} EM={base_s['em']:.0f}->{ras_s['em']:.0f}  F1={base_s['f1']:.3f}->{ras_s['f1']:.3f} ({sign}{delta_f1:.3f})  Q={result.question[:60]}", flush=True)

    base_em = np.mean([s["em"] for s in base_scores])
    base_f1 = np.mean([s["f1"] for s in base_scores])
    ras_em = np.mean([s["em"] for s in ras_scores])
    ras_f1 = np.mean([s["f1"] for s in ras_scores])

    return {
        "base_em": round(base_em, 4), "base_f1": round(base_f1, 4),
        "ras_em": round(ras_em, 4), "ras_f1": round(ras_f1, 4),
        "delta_em": round(ras_em - base_em, 4), "delta_f1": round(ras_f1 - base_f1, 4),
        "base_answer_parser_fallback_count": int(base_parse_fallback_count),
        "ras_answer_parser_fallback_count": int(ras_parse_fallback_count),
        "reused_reader_count": int(reused_reader_count),
        "n": limit,
    }, per_query


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_limit", type=int, default=10)
    parser.add_argument("--hotpot_limit", type=int, default=5)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--extractor_mode", choices=["rule", "llm", "llm_grounded", "llm_closed_grounded"], default="llm_grounded")
    parser.add_argument("--support_mode", choices=["lexical", "embedding_probe"], default="lexical")
    parser.add_argument("--embedding_probe_threshold", type=float, default=0.35)
    cli = parser.parse_args()

    print(f"{'='*60}", flush=True)
    print(f"End-to-End QA: 2Wiki-{cli.wiki_limit}", flush=True)
    print(f"{'='*60}", flush=True)
    agg_2w, pq_2w = run_e2e(
        "2wikimultihopqa",
        cli.wiki_limit,
        cli.qa_top_k,
        extractor_mode=cli.extractor_mode,
        support_mode=cli.support_mode,
        embedding_probe_threshold=cli.embedding_probe_threshold,
    )
    print(f"\n  2Wiki: base EM={agg_2w['base_em']} F1={agg_2w['base_f1']}  |  RAS EM={agg_2w['ras_em']} F1={agg_2w['ras_f1']}  |  delta EM={agg_2w['delta_em']} F1={agg_2w['delta_f1']}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"End-to-End QA: Hotpot-{cli.hotpot_limit}", flush=True)
    print(f"{'='*60}", flush=True)
    agg_hp, pq_hp = run_e2e(
        "hotpotqa",
        cli.hotpot_limit,
        cli.qa_top_k,
        extractor_mode=cli.extractor_mode,
        support_mode=cli.support_mode,
        embedding_probe_threshold=cli.embedding_probe_threshold,
    )
    print(f"\n  Hotpot: base EM={agg_hp['base_em']} F1={agg_hp['base_f1']}  |  RAS EM={agg_hp['ras_em']} F1={agg_hp['ras_f1']}  |  delta EM={agg_hp['delta_em']} F1={agg_hp['delta_f1']}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'':20s}  {'base EM':>8s}  {'RAS EM':>8s}  {'delta':>8s}  {'base F1':>8s}  {'RAS F1':>8s}  {'delta':>8s}", flush=True)
    print(f"  {'2Wiki':20s}  {agg_2w['base_em']:8.4f}  {agg_2w['ras_em']:8.4f}  {agg_2w['delta_em']:+8.4f}  {agg_2w['base_f1']:8.4f}  {agg_2w['ras_f1']:8.4f}  {agg_2w['delta_f1']:+8.4f}", flush=True)
    print(f"  {'Hotpot':20s}  {agg_hp['base_em']:8.4f}  {agg_hp['ras_em']:8.4f}  {agg_hp['delta_em']:+8.4f}  {agg_hp['base_f1']:8.4f}  {agg_hp['ras_f1']:8.4f}  {agg_hp['delta_f1']:+8.4f}", flush=True)

    out_path = "outputs_smoke/ras_llm_e2e_qa.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({
        "extractor_mode": cli.extractor_mode,
        "support_mode": cli.support_mode,
        "embedding_probe_threshold": float(cli.embedding_probe_threshold),
        "2wiki": {"agg": agg_2w, "per_query": pq_2w},
        "hotpot": {"agg": agg_hp, "per_query": pq_hp},
    },
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nFull trace saved to {out_path}", flush=True)
