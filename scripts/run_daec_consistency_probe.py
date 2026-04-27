#!/usr/bin/env python3
"""Probe whether reader answer consistency separates correct vs wrong DAEC contexts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
from string import Template
from typing import Any, Iterable, Sequence
import urllib.request

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import write_json
from src.dpathrag.reader import (
    exact_match,
    format_reader_input,
    gold_answers,
    normalize_answer,
    token_f1,
    token_f1_single,
    write_jsonl,
)
from src.hipporag.utils.misc_utils import extract_answer_from_response
from src.hipporag.prompts.templates.rag_qa_musique import prompt_template as RAG_QA_PROMPT_TEMPLATE


DEFAULT_VARIANTS = ("original", "swap01", "reverse", "rotate_left", "drop_last")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_title_text(doc_text: str, fallback_title: str = "") -> tuple[str, str]:
    text = str(doc_text or "")
    if "\n" in text:
        title, body = text.split("\n", 1)
        return title.strip() or str(fallback_title or ""), body.strip()
    return str(fallback_title or "").strip(), text.strip()


def norm_text(value: str) -> str:
    return normalize_answer(str(value or ""))


def selected_pool_positions(trace: dict[str, Any], top_k: int) -> list[int]:
    selector_trace = trace.get("selector_trace") or {}
    reader_probe = selector_trace.get("reader_order_probe") or {}
    for key in (
        "probed_front_pool_positions" if reader_probe.get("applied") else "",
        "final_front_pool_positions",
        "selected_pool_positions",
    ):
        if not key:
            continue
        values = selector_trace.get(key)
        if isinstance(values, list) and values:
            return [int(value) for value in values[:top_k]]
    return list(range(top_k))


def support_metrics(selected_titles: Sequence[str], gold_titles: Sequence[str]) -> dict[str, float]:
    selected = {norm_text(title) for title in selected_titles if norm_text(title)}
    gold = {norm_text(title) for title in gold_titles if norm_text(title)}
    if not gold:
        return {"support_recall": 0.0, "support_complete": 0.0}
    covered = gold & selected
    return {
        "support_recall": float(len(covered) / len(gold)),
        "support_complete": float(gold.issubset(selected)),
    }


def build_base_records(
    *,
    report_json: str | Path,
    pool_json: str | Path,
    limit: int,
    offset: int,
    top_k: int,
    source: str,
) -> list[dict[str, Any]]:
    report = load_json(report_json)
    pool = load_json(pool_json)
    traces = list(report.get("setwise_selector_query_traces") or [])
    pool_records = list(pool.get("records") or [])
    if not traces:
        raise ValueError(f"No setwise_selector_query_traces found in {report_json}")
    if not pool_records:
        raise ValueError(f"No records found in {pool_json}")

    end = len(traces) if int(limit) <= 0 else min(len(traces), int(offset) + int(limit))
    records: list[dict[str, Any]] = []
    for row_idx in range(int(offset), end):
        trace = traces[row_idx]
        pool_record = pool_records[row_idx]
        trace_question = str(trace.get("question") or "").strip()
        pool_question = str(pool_record.get("question") or "").strip()
        if trace_question != pool_question:
            raise ValueError(
                f"Question mismatch at row {row_idx}: "
                f"trace={trace.get('question')!r} pool={pool_record.get('question')!r}"
            )
        positions = selected_pool_positions(trace, top_k=int(top_k))
        docs: list[dict[str, Any]] = []
        pool_docs = list(pool_record.get("pool_docs") or [])
        pool_titles = list(pool_record.get("pool_titles") or [])
        pool_ids = list(pool_record.get("pool_doc_ids") or [])
        pool_scores = list(pool_record.get("pool_doc_scores") or [])
        gold_titles = [str(title) for title in pool_record.get("gold_titles") or trace.get("gold_titles") or []]
        gold_title_set = {norm_text(title) for title in gold_titles if norm_text(title)}
        for pos in positions:
            if pos < 0 or pos >= len(pool_docs):
                continue
            title, _body = split_title_text(pool_docs[pos], pool_titles[pos] if pos < len(pool_titles) else "")
            if pos < len(pool_titles) and pool_titles[pos]:
                title = str(pool_titles[pos])
            docs.append(
                {
                    "doc_id": pool_ids[pos] if pos < len(pool_ids) else pos,
                    "rank": int(pos) + 1,
                    "title": title,
                    "text": pool_docs[pos],
                    "source": source,
                    "retriever_score": pool_scores[pos] if pos < len(pool_scores) else 0.0,
                    "gold_support": int(norm_text(title) in gold_title_set),
                }
            )
        selected_titles = [str(doc.get("title") or "") for doc in docs]
        metrics = support_metrics(selected_titles, gold_titles)
        answers = pool_record.get("gold_answers") or trace.get("gold_answers") or []
        records.append(
            {
                "qid": str(pool_record.get("qid") or pool_record.get("query_idx") or row_idx),
                "query_idx": int(pool_record.get("query_idx", row_idx)),
                "source": source,
                "question": str(pool_record.get("question") or trace.get("question") or ""),
                "answer": [str(answer) for answer in answers],
                "type": str(trace.get("query_type") or pool_record.get("type") or ""),
                "gold_titles": gold_titles,
                "selected_docs": docs,
                "support_recall": metrics["support_recall"],
                "support_complete": metrics["support_complete"],
                "selector_answer": str(trace.get("selector_answer") or ""),
                "selector_f1_from_report": float((trace.get("selector_metrics") or {}).get("F1") or 0.0),
                "selected_pool_positions": positions,
            }
        )
    return records


def perturb_docs(docs: Sequence[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    values = [dict(doc) for doc in docs]
    if variant == "original":
        return values
    if variant == "swap01":
        if len(values) >= 2:
            values[0], values[1] = values[1], values[0]
        return values
    if variant == "reverse":
        return list(reversed(values))
    if variant == "rotate_left":
        return values[1:] + values[:1] if values else values
    if variant == "drop_last":
        return values[:-1] if len(values) > 1 else values
    raise ValueError(f"Unknown perturbation variant: {variant}")


def make_variant_records(base_records: Sequence[dict[str, Any]], variants: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in base_records:
        for variant in variants:
            item = dict(record)
            item["variant"] = variant
            item["source"] = f"{record.get('source')}_{variant}"
            item["selected_docs"] = perturb_docs(record.get("selected_docs") or [], variant)
            rows.append(item)
    return rows


def mock_prediction(record: dict[str, Any], mode: str) -> str:
    if mode == "oracle":
        return gold_answers(record)[0]
    if mode == "empty":
        return ""
    if mode == "title":
        docs = list(record.get("selected_docs") or [])
        return str((docs[0] or {}).get("title") or "") if docs else ""
    raise ValueError(f"Unsupported mock mode: {mode}")


def generate_predictions_hf(
    records: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    max_docs: int,
    max_doc_chars: int,
    device: str,
) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    resolved_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()

    predictions: list[str] = []
    for start in range(0, len(records), int(batch_size)):
        batch = records[start : start + int(batch_size)]
        inputs = [
            format_reader_input(record, max_docs=int(max_docs), max_doc_chars=int(max_doc_chars))
            for record in batch
        ]
        encoded = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=int(max_input_tokens),
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(**encoded, max_new_tokens=int(max_new_tokens))
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return predictions


def format_llm_reader_messages(
    record: dict[str, Any],
    *,
    max_docs: int,
    max_doc_chars: int,
) -> list[dict[str, str]]:
    docs = list(record.get("selected_docs") or [])
    if max_docs > 0:
        docs = docs[: int(max_docs)]
    prompt_user = ""
    for doc in docs:
        passage = str(doc.get("text") or "")
        if max_doc_chars > 0:
            passage = passage[: int(max_doc_chars)]
        # Match HippoRAG.rag_qa prompt construction: the passage text already starts with the title.
        prompt_user += f"Wikipedia Title: {passage}\n\n"
    prompt_user += "Question: " + str(record.get("question") or "") + "\nThought: "
    return [
        {
            "role": str(item["role"]),
            "content": Template(str(item["content"])).substitute(prompt_user=prompt_user),
        }
        for item in RAG_QA_PROMPT_TEMPLATE
    ]


def call_openai_compatible_reader(
    record: dict[str, Any],
    *,
    base_url: str,
    model: str,
    max_new_tokens: int,
    max_docs: int,
    max_doc_chars: int,
    timeout: int,
    retries: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": format_llm_reader_messages(record, max_docs=max_docs, max_doc_chars=max_doc_chars),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_new_tokens),
    }
    last_error: Exception | None = None
    for _attempt in range(max(1, int(retries) + 1)):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            raw = str(payload["choices"][0]["message"]["content"])
            answer, _meta = extract_answer_from_response(raw)
            return str(answer or raw).strip()
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
    raise RuntimeError(f"LLM reader call failed after retries: {last_error}")


def generate_predictions_openai_compatible(
    records: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    max_new_tokens: int,
    max_docs: int,
    max_doc_chars: int,
    timeout: int,
    retries: int,
    concurrency: int,
) -> list[str]:
    predictions = [""] * len(records)
    workers = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                call_openai_compatible_reader,
                record,
                base_url=base_url,
                model=model,
                max_new_tokens=max_new_tokens,
                max_docs=max_docs,
                max_doc_chars=max_doc_chars,
                timeout=timeout,
                retries=retries,
            ): idx
            for idx, record in enumerate(records)
        }
        for future in as_completed(futures):
            idx = futures[future]
            predictions[idx] = future.result()
    return predictions


def pairwise_prediction_f1(predictions: Sequence[str]) -> float:
    if len(predictions) < 2:
        return 1.0
    scores: list[float] = []
    for left_idx in range(len(predictions)):
        for right_idx in range(left_idx + 1, len(predictions)):
            scores.append(token_f1_single(predictions[left_idx], predictions[right_idx]))
    return sum(scores) / max(1, len(scores))


def consistency_features(predictions: Sequence[str]) -> dict[str, float]:
    normalized = [norm_text(prediction) for prediction in predictions]
    counts = Counter(normalized)
    total = max(1, len(normalized))
    majority = max(counts.values(), default=0)
    probs = [count / total for count in counts.values()]
    if total <= 1 or len(probs) <= 1:
        entropy = 0.0
    else:
        entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0.0) / math.log(total)
    distinct = len(counts)
    if total <= 1:
        inverse_distinct = 1.0
    else:
        inverse_distinct = 1.0 - ((distinct - 1) / (total - 1))
    return {
        "majority_fraction": majority / total,
        "normalized_entropy": entropy,
        "inverse_entropy": 1.0 - entropy,
        "distinct_answer_count": float(distinct),
        "inverse_distinct": inverse_distinct,
        "mean_pairwise_f1": pairwise_prediction_f1(list(predictions)),
    }


def rank_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = [(float(score), int(label)) for label, score in zip(labels, scores)]
    positives = sum(label for _score, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    sorted_pairs = sorted(enumerate(pairs), key=lambda item: item[1][0])
    ranks = [0.0] * len(pairs)
    pos = 0
    while pos < len(sorted_pairs):
        end = pos + 1
        while end < len(sorted_pairs) and sorted_pairs[end][1][0] == sorted_pairs[pos][1][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[sorted_pairs[idx][0]] = avg_rank
        pos = end
    pos_rank_sum = sum(rank for rank, (_score, label) in zip(ranks, pairs) if label == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def bootstrap_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    rounds: int = 1000,
    seed: int = 13,
) -> list[float | None]:
    base = rank_auc(labels, scores)
    if base is None or not labels:
        return [None, None]
    rng = random.Random(seed)
    aucs: list[float] = []
    n = len(labels)
    for _ in range(int(rounds)):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[idx] for idx in indices]
        sample_scores = [scores[idx] for idx in indices]
        auc = rank_auc(sample_labels, sample_scores)
        if auc is not None:
            aucs.append(float(auc))
    if not aucs:
        return [None, None]
    aucs.sort()
    lo = aucs[int(0.025 * (len(aucs) - 1))]
    hi = aucs[int(0.975 * (len(aucs) - 1))]
    return [lo, hi]


def score_original(record: dict[str, Any], prediction: str) -> dict[str, float]:
    answers = gold_answers(record)
    return {
        "original_em": exact_match(answers, prediction),
        "original_f1": token_f1(answers, prediction),
    }


def aggregate_probe(
    base_records: Sequence[dict[str, Any]],
    variant_records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    variants: Sequence[str],
    *,
    bootstrap_rounds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_rows: list[dict[str, Any]] = []
    for record, prediction in zip(variant_records, predictions):
        row = {
            "qid": record.get("qid"),
            "query_idx": record.get("query_idx"),
            "type": record.get("type"),
            "variant": record.get("variant"),
            "question": record.get("question"),
            "prediction": prediction,
            "normalized_prediction": norm_text(prediction),
            "gold_answers": gold_answers(record),
        }
        row.update(score_original(record, prediction))
        prediction_rows.append(row)
        by_qid[str(record.get("qid"))].append(row)

    per_query: list[dict[str, Any]] = []
    base_by_qid = {str(record.get("qid")): record for record in base_records}
    for qid, rows in by_qid.items():
        rows_by_variant = {str(row.get("variant")): row for row in rows}
        ordered_rows = [rows_by_variant[variant] for variant in variants if variant in rows_by_variant]
        preds = [str(row.get("prediction") or "") for row in ordered_rows]
        original = rows_by_variant.get("original") or (ordered_rows[0] if ordered_rows else {})
        features = consistency_features(preds)
        base_record = base_by_qid.get(qid, {})
        item = {
            "qid": qid,
            "query_idx": base_record.get("query_idx"),
            "type": base_record.get("type"),
            "question": base_record.get("question"),
            "gold_answers": gold_answers(base_record),
            "support_recall": float(base_record.get("support_recall") or 0.0),
            "support_complete": float(base_record.get("support_complete") or 0.0),
            "selected_titles": [doc.get("title") for doc in base_record.get("selected_docs") or []],
            "predictions": {str(row.get("variant")): row.get("prediction") for row in ordered_rows},
            "original_prediction": original.get("prediction"),
            "original_em": float(original.get("original_em") or 0.0),
            "original_f1": float(original.get("original_f1") or 0.0),
            "label_em": int(float(original.get("original_em") or 0.0) >= 1.0),
            "label_f1_positive": int(float(original.get("original_f1") or 0.0) > 0.0),
            "label_f1_ge_0_5": int(float(original.get("original_f1") or 0.0) >= 0.5),
        }
        item.update(features)
        per_query.append(item)

    metric_names = ["majority_fraction", "inverse_entropy", "inverse_distinct", "mean_pairwise_f1"]
    label_names = ["label_em", "label_f1_positive", "label_f1_ge_0_5"]
    aucs: dict[str, Any] = {}
    for label_name in label_names:
        labels = [int(row[label_name]) for row in per_query]
        aucs[label_name] = {
            "positives": sum(labels),
            "negatives": len(labels) - sum(labels),
            "metrics": {},
        }
        for metric_name in metric_names:
            scores = [float(row[metric_name]) for row in per_query]
            auc = rank_auc(labels, scores)
            ci = bootstrap_auc_ci(labels, scores, rounds=bootstrap_rounds) if auc is not None else [None, None]
            aucs[label_name]["metrics"][metric_name] = {
                "auc": None if auc is None else round(float(auc), 6),
                "ci95": [None if value is None else round(float(value), 6) for value in ci],
            }

    def mean(rows: Sequence[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return sum(float(row.get(key) or 0.0) for row in rows) / len(rows)

    correct = [row for row in per_query if int(row["label_f1_ge_0_5"]) == 1]
    wrong = [row for row in per_query if int(row["label_f1_ge_0_5"]) == 0]
    summary = {
        "rows": len(per_query),
        "variant_rows": len(variant_records),
        "variants": list(variants),
        "original_em": round(mean(per_query, "original_em"), 6),
        "original_f1": round(mean(per_query, "original_f1"), 6),
        "support_recall": round(mean(per_query, "support_recall"), 6),
        "support_complete": round(mean(per_query, "support_complete"), 6),
        "consistency_means": {
            metric: round(mean(per_query, metric), 6)
            for metric in metric_names + ["normalized_entropy", "distinct_answer_count"]
        },
        "correct_f1_ge_0_5_means": {
            metric: round(mean(correct, metric), 6)
            for metric in metric_names + ["normalized_entropy", "distinct_answer_count"]
        },
        "wrong_f1_lt_0_5_means": {
            metric: round(mean(wrong, metric), 6)
            for metric in metric_names + ["normalized_entropy", "distinct_answer_count"]
        },
        "aucs": aucs,
    }
    return summary, per_query, prediction_rows


def write_markdown_report(output: dict[str, Any], path: str | Path) -> None:
    summary = output["summary"]
    lines = [
        "# DAEC Reader Consistency Probe",
        "",
        "## Purpose",
        "",
        "Test whether reader answer consistency under DAEC evidence perturbations separates correct from wrong original DAEC contexts.",
        "",
        "This is a signal probe only. It does not implement active retrieval or edit admission.",
        "",
        "## Configuration",
        "",
        f"- DAEC report: `{output['inputs']['report_json']}`",
        f"- Pool JSON: `{output['inputs']['pool_json']}`",
        f"- Reader mode: `{output['runtime']['mode']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Variants: `{', '.join(summary['variants'])}`",
        "",
        "## Base Reader Result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Original EM | {summary['original_em']:.4f} |",
        f"| Original F1 | {summary['original_f1']:.4f} |",
        f"| Support Recall | {summary['support_recall']:.4f} |",
        f"| Support Complete | {summary['support_complete']:.4f} |",
        "",
        "## Consistency Means",
        "",
        "| Metric | All | Correct F1>=0.5 | Wrong F1<0.5 |",
        "|---|---:|---:|---:|",
    ]
    for metric in [
        "majority_fraction",
        "inverse_entropy",
        "inverse_distinct",
        "mean_pairwise_f1",
        "normalized_entropy",
        "distinct_answer_count",
    ]:
        lines.append(
            f"| {metric} | "
            f"{summary['consistency_means'][metric]:.4f} | "
            f"{summary['correct_f1_ge_0_5_means'][metric]:.4f} | "
            f"{summary['wrong_f1_lt_0_5_means'][metric]:.4f} |"
        )
    lines.extend(["", "## AUC: Consistency Predicting Correctness", ""])
    for label_name, label_result in summary["aucs"].items():
        lines.extend(
            [
                f"### {label_name}",
                "",
                f"- positives: `{label_result['positives']}`",
                f"- negatives: `{label_result['negatives']}`",
                "",
                "| Metric | AUC | 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for metric_name, metric_result in label_result["metrics"].items():
            ci = metric_result["ci95"]
            ci_text = "n/a" if ci[0] is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            auc = metric_result["auc"]
            auc_text = "n/a" if auc is None else f"{auc:.4f}"
            lines.append(f"| {metric_name} | {auc_text} | {ci_text} |")
        lines.append("")
    primary = summary["aucs"]["label_f1_ge_0_5"]["metrics"]["majority_fraction"]["auc"]
    lines.extend(["## Decision Hint", ""])
    if primary is None:
        decision = "INCONCLUSIVE_NO_CLASS_BALANCE"
    elif primary >= 0.70:
        decision = "PASS_SIGNAL_PROBE"
    elif primary >= 0.65:
        decision = "BORDERLINE_SIGNAL_PROBE"
    else:
        decision = "FAIL_SIGNAL_PROBE"
    lines.extend(
        [
            f"- Primary metric: `majority_fraction` AUC vs `label_f1_ge_0_5` = `{primary}`",
            f"- Decision hint: `{decision}`",
            "",
        ]
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--output_dir", default="reports/daec_alr")
    parser.add_argument("--source", default="daec_proprag")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--llm_base_url", default="")
    parser.add_argument("--llm_model", default="")
    parser.add_argument("--llm_timeout", type=int, default=120)
    parser.add_argument("--llm_retries", type=int, default=1)
    parser.add_argument("--llm_concurrency", type=int, default=8)
    parser.add_argument("--mock_mode", choices=["none", "oracle", "empty", "title"], default="none")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap_rounds", type=int, default=1000)
    args = parser.parse_args()

    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    if "original" not in variants:
        raise ValueError("Variants must include 'original' so correctness can be labeled.")

    base_records = build_base_records(
        report_json=args.report_json,
        pool_json=args.pool_json,
        limit=int(args.limit),
        offset=int(args.offset),
        top_k=int(args.top_k),
        source=str(args.source),
    )
    variant_records = make_variant_records(base_records, variants)

    if args.mock_mode != "none":
        predictions = [mock_prediction(record, str(args.mock_mode)) for record in variant_records]
        runtime = {"mode": f"mock_{args.mock_mode}"}
    elif args.llm_base_url and args.llm_model:
        predictions = generate_predictions_openai_compatible(
            variant_records,
            base_url=str(args.llm_base_url),
            model=str(args.llm_model),
            max_new_tokens=int(args.max_new_tokens),
            max_docs=int(args.max_docs),
            max_doc_chars=int(args.max_doc_chars),
            timeout=int(args.llm_timeout),
            retries=int(args.llm_retries),
            concurrency=int(args.llm_concurrency),
        )
        runtime = {
            "mode": "openai_compatible_chat",
            "base_url": str(args.llm_base_url),
            "model": str(args.llm_model),
            "concurrency": int(args.llm_concurrency),
        }
    elif args.model_name_or_path:
        predictions = generate_predictions_hf(
            variant_records,
            model_name_or_path=str(args.model_name_or_path),
            batch_size=int(args.batch_size),
            max_input_tokens=int(args.max_input_tokens),
            max_new_tokens=int(args.max_new_tokens),
            max_docs=int(args.max_docs),
            max_doc_chars=int(args.max_doc_chars),
            device=str(args.device),
        )
        runtime = {"mode": "hf_seq2seq", "model_name_or_path": str(args.model_name_or_path)}
    else:
        raise ValueError("Provide --model_name_or_path, --llm_base_url/--llm_model, or set --mock_mode.")

    summary, per_query, prediction_rows = aggregate_probe(
        base_records,
        variant_records,
        predictions,
        variants,
        bootstrap_rounds=int(args.bootstrap_rounds),
    )
    output_dir = Path(args.output_dir)
    output = {
        "inputs": {
            "report_json": str(args.report_json),
            "pool_json": str(args.pool_json),
            "limit": int(args.limit),
            "offset": int(args.offset),
            "top_k": int(args.top_k),
        },
        "runtime": runtime,
        "generation": {
            "batch_size": int(args.batch_size),
            "max_input_tokens": int(args.max_input_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "max_docs": int(args.max_docs),
            "max_doc_chars": int(args.max_doc_chars),
        },
        "summary": summary,
    }
    write_json(output, output_dir / "consistency_probe.json")
    write_jsonl(per_query, output_dir / "consistency_probe.rows.jsonl")
    write_jsonl(prediction_rows, output_dir / "consistency_probe.predictions.jsonl")
    write_markdown_report(output, output_dir / "consistency_probe.md")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
