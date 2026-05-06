#!/usr/bin/env python3
"""IRCoT-style iterative retrieve/read baseline for BSGS Week 0."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
import urllib.request

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from src.bsgs.metrics import answer_f1, exact_match, recall_at_gold
from src.bsgs.io import read_json, write_json
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import QuerySolution


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def get_gold_docs(samples: list[dict[str, Any]]) -> list[list[str]]:
    gold_docs: list[list[str]] = []
    for sample in samples:
        docs: list[str] = []
        paragraphs = sample.get("paragraphs") or []
        if paragraphs:
            docs = [
                para["title"] + "\n" + (para.get("text") or para.get("paragraph_text") or "")
                for para in paragraphs
                if para.get("is_supporting") is not False
            ]
        elif sample.get("context") and sample.get("supporting_facts"):
            context_by_title = {
                str(title): " ".join(str(sent) for sent in sentences)
                for title, sentences in sample.get("context") or []
            }
            support_titles = [str(row[0]) for row in sample.get("supporting_facts") or [] if row]
            docs = [
                f"{title}\n{context_by_title[title]}"
                for title in support_titles
                if title in context_by_title
            ]
        gold_docs.append(list(dict.fromkeys(docs)))
    return gold_docs


def get_gold_answers(samples: list[dict[str, Any]]) -> list[list[str]]:
    answers: list[list[str]] = []
    for sample in samples:
        values = [sample.get("answer")]
        values.extend(sample.get("answer_aliases") or [])
        answers.append([str(value) for value in values if value is not None])
    return answers


def call_chat(base_url: str, model: str, prompt: str, timeout: int) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage") or {}
    return content, usage


def build_followup_prompt(question: str, evidence_docs: list[str], step: int) -> str:
    snippets = []
    for idx, doc in enumerate(evidence_docs[:5], start=1):
        snippets.append(f"[{idx}] {doc[:900]}")
    evidence = "\n\n".join(snippets)
    return (
        "/no_think\n"
        "You are running IRCoT-style retrieval for multi-hop QA.\n"
        "Given the original question and retrieved evidence, write one short follow-up retrieval query for the next missing hop.\n"
        "Do not answer the original question. Do not include reasoning.\n"
        "Return JSON only: {\"query\": \"...\"}\n\n"
        f"Step: {step}\n"
        f"Question: {question}\n\n"
        f"Retrieved evidence:\n{evidence}\n"
    )


def parse_followup_query(raw: str, fallback: str) -> str:
    match = JSON_RE.search(raw or "")
    if match:
        try:
            payload = json.loads(match.group(0))
            query = str(payload.get("query") or "").strip()
            if query:
                return query
        except json.JSONDecodeError:
            pass
    cleaned = " ".join(str(raw or "").split())
    return cleaned[:240] if cleaned else fallback


def select_final_docs(
    step_docs: list[list[str]],
    score_by_doc: dict[str, float],
    *,
    qa_top_k: int,
    final_doc_order: str,
) -> tuple[list[str], list[float]]:
    selected: list[str] = []
    seen: set[str] = set()
    normalized_order = str(final_doc_order or "append_order").strip().lower()
    if normalized_order == "round_robin":
        max_width = max((len(docs) for docs in step_docs), default=0)
        for rank in range(max_width):
            for docs in step_docs:
                if rank >= len(docs):
                    continue
                doc = docs[rank]
                if doc in seen:
                    continue
                selected.append(doc)
                seen.add(doc)
                if len(selected) >= int(qa_top_k):
                    return selected, [float(score_by_doc.get(doc, 1.0)) for doc in selected]
    else:
        for docs in step_docs:
            for doc in docs:
                if doc in seen:
                    continue
                selected.append(doc)
                seen.add(doc)
                if len(selected) >= int(qa_top_k):
                    return selected, [float(score_by_doc.get(doc, 1.0)) for doc in selected]
    return selected, [float(score_by_doc.get(doc, 1.0)) for doc in selected]


def build_config(args: argparse.Namespace, corpus_len: int) -> BaseConfig:
    save_dir = args.save_dir
    if save_dir == "outputs":
        save_dir = os.path.join(save_dir, args.dataset)
    else:
        save_dir = f"{save_dir}_{args.dataset}"
    return BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        llm_request_name=args.llm_request_name,
        embedding_base_url=args.embedding_base_url,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        max_qa_steps=3,
        qa_top_k=args.qa_top_k,
        qa_doc_max_chars=args.qa_doc_max_chars,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=args.embedding_batch_size,
        max_new_tokens=None,
        max_retry_attempts=args.max_retry_attempts,
        corpus_len=corpus_len,
        openie_mode=args.openie_mode,
        causal_enabled=False,
        planner_enabled=False,
    )


def replay_existing_report(path: str) -> dict[str, Any]:
    payload = read_json(path)
    answer_em = payload.get("qa", {}).get("ExactMatch") or payload.get("ExactMatch")
    answer_f1_value = payload.get("qa", {}).get("F1") or payload.get("F1")
    return {
        "status": "replayed",
        "source_report": path,
        "dataset": payload.get("dataset"),
        "answer_em": answer_em,
        "answer_f1": answer_f1_value,
        "metrics": {
            "answer_em": answer_em,
            "answer_f1": answer_f1_value,
            "supporting_paragraph_recall": payload.get("supporting_paragraph_recall"),
        },
        "retrieval": payload.get("retrieval") or {},
        "note": "This is a replayed report, not a newly executed IRCoT run.",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# IRCoT Baseline",
        "",
        f"- Status: `{report['status']}`",
        f"- Dataset: `{report.get('dataset', 'musique')}`",
        f"- Limit: `{report.get('limit', '')}`",
        f"- Max iter: `{report.get('max_iter', '')}`",
        f"- Top-k per iter: `{report.get('top_k_per_iter', '')}`",
        f"- Answer EM: `{report.get('answer_em', '')}`",
        f"- Answer F1: `{report.get('answer_f1', '')}`",
        f"- LLM calls/query: `{report.get('llm_calls_per_query', '')}`",
        f"- Latency/query: `{report.get('latency_per_query', '')}`",
    ]
    if report.get("reason"):
        lines.extend(["", f"Reason: {report['reason']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ircot(args: argparse.Namespace) -> dict[str, Any]:
    corpus_path = Path(args.dataset_dir) / f"{args.dataset}_corpus.json"
    sample_path = Path(args.dataset_dir) / f"{args.dataset}.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_docs = get_gold_docs(samples)
    gold_answers = get_gold_answers(samples)
    config = build_config(args, corpus_len=len(corpus))
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)

    start_time = perf_counter()
    current_queries = list(queries)
    per_query_docs: list[list[str]] = [[] for _ in queries]
    per_query_scores: list[list[float]] = [[] for _ in queries]
    per_query_step_docs: list[list[list[str]]] = [[] for _ in queries]
    per_query_score_by_doc: list[dict[str, float]] = [{} for _ in queries]
    generated_queries: list[list[str]] = [[] for _ in queries]
    llm_query_calls = 0
    llm_errors: list[str] = []

    for step in range(1, int(args.max_iter) + 1):
        retrieved = hipporag.retrieve(queries=current_queries, num_to_retrieve=args.top_k_per_iter)
        if isinstance(retrieved, tuple):
            query_solutions = retrieved[0]
        else:
            query_solutions = retrieved
        for q_idx, qs in enumerate(query_solutions):
            seen = set(per_query_docs[q_idx])
            scores = qs.doc_scores.tolist() if qs.doc_scores is not None else [1.0] * len(qs.docs)
            step_docs: list[str] = []
            for doc, score in zip(qs.docs, scores):
                per_query_score_by_doc[q_idx].setdefault(doc, float(score))
                if doc not in set(step_docs):
                    step_docs.append(doc)
                if doc not in seen:
                    per_query_docs[q_idx].append(doc)
                    per_query_scores[q_idx].append(float(score))
                    seen.add(doc)
            per_query_step_docs[q_idx].append(step_docs)
        if step >= int(args.max_iter):
            break
        next_queries: list[str] = []
        for q_idx, question in enumerate(queries):
            prompt = build_followup_prompt(question, per_query_docs[q_idx], step=step)
            try:
                raw, _ = call_chat(args.llm_base_url, args.llm_request_name or args.llm_name, prompt, args.timeout)
                llm_query_calls += 1
                next_query = parse_followup_query(raw, fallback=current_queries[q_idx])
            except Exception as exc:
                llm_errors.append(f"{q_idx}: {type(exc).__name__}: {exc}")
                next_query = current_queries[q_idx]
            generated_queries[q_idx].append(next_query)
            next_queries.append(next_query)
        current_queries = next_queries

    final_solutions: list[QuerySolution] = []
    for q_idx, question in enumerate(queries):
        docs_top, scores = select_final_docs(
            per_query_step_docs[q_idx],
            per_query_score_by_doc[q_idx],
            qa_top_k=int(args.qa_top_k),
            final_doc_order=str(args.final_doc_order),
        )
        scores_top = np.asarray(scores[: len(docs_top)], dtype=float)
        final_solutions.append(
            QuerySolution(
                question=question,
                docs=docs_top,
                doc_scores=scores_top,
                gold_docs=gold_docs[q_idx],
            )
        )

    final_solutions, responses, metadata, _, overall_qa_results = hipporag.rag_qa(
        queries=final_solutions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )
    retrieval_metrics, _ = RetrievalRecall(global_config=config).calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=[qs.docs for qs in final_solutions],
        k_list=[1, 2, 5, 10, 20],
    )

    traces: list[dict[str, Any]] = []
    ems: list[float] = []
    f1s: list[float] = []
    support_recalls: list[float] = []
    for q_idx, qs in enumerate(final_solutions):
        answer = qs.answer or (responses[q_idx] if q_idx < len(responses) else "")
        em = exact_match(gold_answers[q_idx], answer)
        f1 = answer_f1(gold_answers[q_idx], answer)
        support_recall = recall_at_gold(qs.docs, gold_docs[q_idx])
        ems.append(em)
        f1s.append(f1)
        support_recalls.append(support_recall)
        traces.append(
            {
                "qid": samples[q_idx].get("id") or samples[q_idx].get("_id"),
                "question": queries[q_idx],
                "generated_queries": generated_queries[q_idx],
                "final_doc_order": str(args.final_doc_order),
                "step_docs": per_query_step_docs[q_idx],
                "answer": answer,
                "gold_answers": gold_answers[q_idx],
                "em": em,
                "f1": f1,
                "supporting_paragraph_recall": support_recall,
                "docs": qs.docs,
                "metadata": metadata[q_idx] if q_idx < len(metadata) else {},
            }
        )

    elapsed = perf_counter() - start_time
    answer_em_mean = float(np.mean(ems)) if ems else 0.0
    answer_f1_mean = float(np.mean(f1s)) if f1s else 0.0
    support_recall_mean = float(np.mean(support_recalls)) if support_recalls else 0.0
    return {
        "status": "completed",
        "dataset": args.dataset,
        "limit": len(samples),
        "max_iter": int(args.max_iter),
        "top_k_per_iter": int(args.top_k_per_iter),
        "qa_top_k": int(args.qa_top_k),
        "final_doc_order": str(args.final_doc_order),
        "answer_em": answer_em_mean,
        "answer_f1": answer_f1_mean,
        "supporting_paragraph_recall": support_recall_mean,
        "metrics": {
            "answer_em": answer_em_mean,
            "answer_f1": answer_f1_mean,
            "supporting_paragraph_recall": support_recall_mean,
        },
        "retrieval": retrieval_metrics,
        "llm_query_calls": llm_query_calls,
        "llm_calls_per_query": (llm_query_calls + len(samples)) / max(1, len(samples)),
        "retrieved_passages_per_query": int(args.max_iter) * int(args.top_k_per_iter),
        "latency_per_query": elapsed / max(1, len(samples)),
        "elapsed_seconds": elapsed,
        "llm_errors": llm_errors[:50],
        "traces": traces,
        "overall_qa_results": overall_qa_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--top_k_per_iter", type=int, default=5)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--final_doc_order", choices=["append_order", "round_robin"], default="round_robin")
    parser.add_argument("--qa_doc_max_chars", type=int, default=2048)
    parser.add_argument("--retrieval_top_k", type=int, default=100)
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--max_retry_attempts", type=int, default=20)
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online")
    parser.add_argument("--save_dir", default="outputs_step0_general")
    parser.add_argument("--llm_base_url", default="")
    parser.add_argument("--llm_name", default="qwen3-8b")
    parser.add_argument("--llm_request_name", default=None)
    parser.add_argument("--embedding_name", default="VLLM/nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", default="http://localhost:8019/v1/embeddings")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--replay_report", default="")
    parser.add_argument("--report_json", default="reports/week0/ircot_musique1000.json")
    parser.add_argument("--report_md", default="reports/week0/ircot_baseline.md")
    args = parser.parse_args()

    if args.replay_report:
        report = replay_existing_report(args.replay_report)
    elif not args.llm_base_url:
        report = {
            "status": "not_run",
            "dataset": args.dataset,
            "limit": args.limit,
            "max_iter": args.max_iter,
            "top_k_per_iter": args.top_k_per_iter,
            "llm_name": args.llm_name,
            "reason": "No --llm_base_url or --replay_report provided. This script will not fabricate IRCoT results.",
        }
    else:
        report = run_ircot(args)
    write_json(report, args.report_json)
    write_markdown(report, Path(args.report_md))
    print(f"Wrote {args.report_json} and {args.report_md}")


if __name__ == "__main__":
    main()
