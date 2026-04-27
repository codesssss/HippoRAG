#!/usr/bin/env python3
"""C-CEE Day-1 diagnostics.

This script is deliberately Phase-1 only: it audits assets, validates frozen
reader likelihood mechanics, checks reader semantic capacity, and runs a
decision-level counterfactual edit separability diagnostic on the dev split.
It does not train models and does not run held-out eval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Sequence
from urllib import request

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dpathrag_analyze_hard_negatives import is_gold, load_jsonl, row_qid  # noqa: E402
from dpathrag_cee_edit_policy import load_embedding_features, make_example  # noqa: E402
from dpathrag_cee_pairwise_common import (  # noqa: E402
    action_indices,
    enumerate_edit_summaries,
    mean,
    rank_indices_for,
)
from src.dpathrag.io import write_json  # noqa: E402
from src.dpathrag.reader import format_reader_input, gold_answers, normalize_answer  # noqa: E402
from src.dpathrag.selector_data import support_metrics_for_indices  # noqa: E402


DECISION_VALUES = {
    "PROCEED_PHASE2",
    "STOP_MECHANICAL_FAIL",
    "STOP_READER_SEMANTIC_CAPACITY_FAIL",
    "STOP_DECISION_SEPARABILITY_FAIL",
    "STOP_INSUFFICIENT_DEV_SAMPLE",
}


def finite(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(default)


def selected_docs(record: dict[str, Any], indices: Sequence[int]) -> list[dict[str, Any]]:
    candidates = list(record.get("candidates") or [])
    docs = []
    for index in indices:
        idx = int(index)
        if 0 <= idx < len(candidates):
            candidate = candidates[idx]
            docs.append({"title": str(candidate.get("title") or ""), "text": str(candidate.get("text") or "")})
    return docs


def reader_record(question: str, docs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"question": question, "selected_docs": list(docs)}


def dev_rows(rows: Sequence[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return list(rows[int(start) : int(end)])


def sha1_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class JsonlLoglikCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.values: dict[str, float] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.values[str(row["key"])] = float(row["value"])

    def get(self, key: str) -> float | None:
        return self.values.get(str(key))

    def set(self, key: str, value: float, metadata: dict[str, Any] | None = None) -> None:
        key = str(key)
        value = float(value)
        self.values[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value, "metadata": metadata or {}}, ensure_ascii=False) + "\n")


def masked_length_normalized_loglik(logits: Any, labels: Any, *, ignore_index: int = -100) -> list[float]:
    """Compute length-normalized log likelihood from decoder logits and labels."""
    import torch

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    safe_labels = labels.clone()
    mask = safe_labels.ne(int(ignore_index))
    safe_labels = safe_labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    sums = (token_log_probs * mask.float()).sum(dim=-1)
    lengths = mask.float().sum(dim=-1).clamp_min(1.0)
    return [float(value) for value in (sums / lengths).detach().cpu().tolist()]


class FrozenReaderLoglikProbe:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        cache_path: str | Path,
        max_input_tokens: int,
        max_target_tokens: int,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name_or_path = str(model_name_or_path)
        self.max_input_tokens = int(max_input_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.cache = JsonlLoglikCache(cache_path)

    def cache_key(self, *, qid: str, question: str, docs: Sequence[dict[str, Any]], answer: str) -> str:
        return sha1_json(
            {
                "qid": str(qid),
                "question": str(question),
                "doc_titles": [str(doc.get("title") or "") for doc in docs],
                "doc_text_hashes": [hashlib.sha1(str(doc.get("text") or "").encode("utf-8")).hexdigest() for doc in docs],
                "answer": normalize_answer(answer),
                "raw_answer": str(answer),
                "reader": self.model_name_or_path,
                "max_input_tokens": self.max_input_tokens,
                "max_target_tokens": self.max_target_tokens,
            }
        )

    def length_normalized_loglik(self, qid: str, question: str, docs: Sequence[dict[str, Any]], answer: str) -> float:
        return self.batch_length_normalized_loglik(
            [{"qid": qid, "question": question, "docs": list(docs), "answer": answer}],
            batch_size=1,
        )[0]

    def batch_length_normalized_loglik(self, items: Sequence[dict[str, Any]], *, batch_size: int = 16) -> list[float]:
        import torch

        outputs: list[float | None] = [None for _ in items]
        uncached: list[tuple[int, dict[str, Any], str]] = []
        for offset, item in enumerate(items):
            answer = str(item.get("answer") or "")
            if not normalize_answer(answer):
                raise ValueError("Cannot score an empty normalized answer")
            key = self.cache_key(
                qid=str(item.get("qid") or ""),
                question=str(item.get("question") or ""),
                docs=list(item.get("docs") or []),
                answer=answer,
            )
            cached = self.cache.get(key)
            if cached is None:
                uncached.append((offset, item, key))
            else:
                outputs[offset] = cached

        for start in range(0, len(uncached), int(batch_size)):
            batch = uncached[start : start + int(batch_size)]
            sources = [
                format_reader_input(
                    reader_record(str(item.get("question") or ""), list(item.get("docs") or [])),
                    max_docs=0,
                    max_doc_chars=0,
                )
                for _, item, _ in batch
            ]
            answers = [str(item.get("answer") or "") for _, item, _ in batch]
            encoded = self.tokenizer(
                sources,
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            target = self.tokenizer(
                text_target=answers,
                padding=True,
                truncation=True,
                max_length=self.max_target_tokens,
                return_tensors="pt",
            )
            labels = target["input_ids"]
            labels = labels.masked_fill(labels.eq(self.tokenizer.pad_token_id), -100)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            labels = labels.to(self.device)
            with torch.no_grad():
                result = self.model(**encoded, labels=labels)
            values = masked_length_normalized_loglik(result.logits, labels)
            for (offset, item, key), value in zip(batch, values):
                outputs[offset] = float(value)
                self.cache.set(key, float(value), {"qid": str(item.get("qid") or ""), "answer": str(item.get("answer") or "")})

        return [float(value) for value in outputs if value is not None]

    def generate_answer(self, qid: str, question: str, docs: Sequence[dict[str, Any]], *, max_new_tokens: int = 32) -> str:
        import torch

        source = format_reader_input(reader_record(question, docs), max_docs=0, max_doc_chars=0)
        encoded = self.tokenizer(
            [source],
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**encoded, max_new_tokens=int(max_new_tokens))
        return str(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])


def auc_score(positives: Sequence[float], negatives: Sequence[float]) -> float:
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if float(pos) > float(neg):
                wins += 1.0
            elif math.isclose(float(pos), float(neg)):
                wins += 0.5
    return wins / max(1.0, total)


def pr_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    paired = sorted(zip(scores, labels), key=lambda item: float(item[0]), reverse=True)
    positives = sum(1 for label in labels if int(label) == 1)
    if positives <= 0:
        return 0.0
    tp = 0
    fp = 0
    last_recall = 0.0
    area = 0.0
    for _, label in paired:
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        area += precision * max(0.0, recall - last_recall)
        last_recall = recall
    return area


def bootstrap_auc_ci(
    positives: Sequence[float],
    negatives: Sequence[float],
    *,
    seed: int = 17,
    resamples: int = 1000,
) -> dict[str, float]:
    if not positives or not negatives:
        return {"auc": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(int(seed))
    values = []
    positives = list(float(value) for value in positives)
    negatives = list(float(value) for value in negatives)
    for _ in range(int(resamples)):
        pos = [rng.choice(positives) for _ in positives]
        neg = [rng.choice(negatives) for _ in negatives]
        values.append(auc_score(pos, neg))
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return {"auc": round(auc_score(positives, negatives), 6), "ci_low": round(low, 6), "ci_high": round(high, 6)}


def cluster_answers(hypotheses: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for hyp in hypotheses:
        text = str(hyp.get("text") or "")
        norm = normalize_answer(text)
        if not norm:
            continue
        grouped.setdefault(norm, []).append({**hyp, "normalized": norm})
    clusters = []
    raw_scores = []
    for norm, rows in grouped.items():
        originals = [str(row.get("text") or "") for row in rows if str(row.get("text") or "")]
        canonical = min(originals, key=len) if originals else norm
        logliks = [float(row.get("loglik") or 0.0) for row in rows if row.get("loglik") is not None]
        mean_loglik = mean(logliks) if logliks else 0.0
        raw = len(rows) * math.exp(max(-50.0, min(50.0, mean_loglik)))
        raw_scores.append(raw)
        clusters.append(
            {
                "canonical_text": canonical,
                "normalized": norm,
                "vote_count": len(rows),
                "mean_loglik": mean_loglik,
                "sources": [str(row.get("source") or "") for row in rows],
            }
        )
    total = sum(raw_scores)
    if total <= 0.0 and clusters:
        for cluster in clusters:
            cluster["prior"] = 1.0 / len(clusters)
    else:
        for cluster, raw in zip(clusters, raw_scores):
            cluster["prior"] = float(raw) / total
    return sorted(clusters, key=lambda item: float(item["prior"]), reverse=True)


def posterior_decision(
    edit_scores: Sequence[dict[str, Any]],
    answer_clusters: Sequence[dict[str, Any]],
    *,
    pi_stop: float,
) -> dict[str, Any]:
    if not answer_clusters:
        return {"is_stop": True, "edit": None, "answer": None, "score": float("-inf")}
    edit_count = len(edit_scores)
    log_stop = math.log(float(pi_stop))
    log_edit = math.log(max(1e-12, (1.0 - float(pi_stop)) / max(1, edit_count)))
    best = {"is_stop": True, "edit": None, "answer": answer_clusters[0], "score": float("-inf")}
    for answer in answer_clusters:
        answer_prior = max(1e-12, float(answer.get("prior") or 0.0))
        stop_score = math.log(answer_prior) + log_stop
        if stop_score > float(best["score"]):
            best = {"is_stop": True, "edit": None, "answer": answer, "score": stop_score}
    for row in edit_scores:
        for answer in answer_clusters:
            norm = str(answer["normalized"])
            answer_prior = max(1e-12, float(answer.get("prior") or 0.0))
            score = math.log(answer_prior) + log_edit + float(row.get("deltas", {}).get(norm, 0.0))
            if score > float(best["score"]):
                best = {"is_stop": False, "edit": row["edit"], "answer": answer, "score": score}
    return best


def write_markdown(lines: Sequence[str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def qwen_health(endpoint: str) -> dict[str, Any]:
    try:
        req = request.Request(str(endpoint).rstrip("/") + "/models", method="GET")
        with request.urlopen(req, timeout=2.0) as response:
            return {"available": 200 <= int(response.status) < 500, "status": int(response.status)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def qwen_answer(endpoint: str, question: str, docs: Sequence[dict[str, Any]]) -> str:
    evidence = "\n\n".join(f"[{idx}] {doc.get('title')}\n{doc.get('text')}" for idx, doc in enumerate(docs, start=1))
    prompt = f"Question: {question}\n\nEvidence:\n{evidence}\n\nProvide a concise answer. Output only the answer string, no explanation.\n\nAnswer:"
    payload = {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 32,
    }
    req = request.Request(
        str(endpoint).rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=20.0) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


def load_probe(args: argparse.Namespace) -> FrozenReaderLoglikProbe:
    return FrozenReaderLoglikProbe(
        model_name_or_path=str(args.reader_model),
        cache_path=Path(args.cache_dir) / "loglik_cache.jsonl",
        max_input_tokens=int(args.max_input_tokens),
        max_target_tokens=int(args.max_target_tokens),
        device=str(args.device),
    )


def run_asset_audit(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    reader_dir = Path(args.reader_model)
    reader_files = {name: (reader_dir / name).exists() for name in ("config.json", "tokenizer.json", "spiece.model", "model.safetensors")}
    pool_rows = sum(1 for _ in Path(args.cache_jsonl).open("r", encoding="utf-8")) if Path(args.cache_jsonl).exists() else 0
    dense_rows = sum(1 for _ in Path(args.dense_cache_jsonl).open("r", encoding="utf-8")) if Path(args.dense_cache_jsonl).exists() else 0
    torch_status: dict[str, Any]
    try:
        import torch
        import transformers  # noqa: F401

        torch_status = {"import_ok": True, "cuda_available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count())}
    except Exception as exc:
        torch_status = {"import_ok": False, "error": str(exc)}
    qwen = qwen_health(str(args.llm_endpoint))
    payload = {
        "reader_model": str(args.reader_model),
        "reader_files": reader_files,
        "cache_jsonl": str(args.cache_jsonl),
        "pool_rows": pool_rows,
        "dense_cache_jsonl": str(args.dense_cache_jsonl),
        "dense_rows": dense_rows,
        "reports_dir_created": report_dir.exists(),
        "cache_dir_created": Path(args.cache_dir).exists(),
        "torch": torch_status,
        "qwen": qwen,
        "passed": bool(reader_dir.exists() and all(reader_files.values()) and pool_rows >= 1000 and torch_status.get("import_ok")),
    }
    write_json(payload, report_dir / "asset_audit.json")
    write_markdown(
        [
            "# C-CEE Asset Audit",
            "",
            f"- Passed: `{payload['passed']}`",
            f"- Reader: `{payload['reader_model']}`",
            f"- Reader files: `{payload['reader_files']}`",
            f"- PropRAG rows: `{pool_rows}`",
            f"- Dense rows: `{dense_rows}`",
            f"- Torch: `{torch_status}`",
            f"- Qwen: `{qwen}`",
        ],
        report_dir / "asset_audit.md",
    )
    return payload


def run_likelihood_sanity(args: argparse.Namespace, rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    report_dir = Path(args.output_dir)
    rows = dev_rows(rows or load_jsonl(args.cache_jsonl), int(args.dev_start), int(args.dev_end))[:5]
    probe = load_probe(args)
    max_diffs = {"batch_single": 0.0, "cache_repeat": 0.0}
    token_rows = []
    batch_items = []
    singles = []
    for record in rows:
        qid = row_qid(record)
        base = rank_indices_for(record, top_k=int(args.top_k), max_candidates=int(args.max_candidates))
        docs = selected_docs(record, base)
        answer = gold_answers(record)[0]
        single = probe.length_normalized_loglik(qid, str(record.get("question") or ""), docs, answer)
        repeat = probe.length_normalized_loglik(qid, str(record.get("question") or ""), docs, answer)
        max_diffs["cache_repeat"] = max(max_diffs["cache_repeat"], abs(single - repeat))
        singles.append(single)
        batch_items.append({"qid": qid, "question": str(record.get("question") or ""), "docs": docs, "answer": answer})
        source = format_reader_input(reader_record(str(record.get("question") or ""), docs))
        source_tokens = probe.tokenizer(source, truncation=False, verbose=False)["input_ids"]
        truncated_tokens = probe.tokenizer(source, truncation=True, max_length=int(args.max_input_tokens), verbose=False)["input_ids"]
        target_tokens = probe.tokenizer(text_target=answer, truncation=True, max_length=int(args.max_target_tokens))["input_ids"]
        token_rows.append(
            {
                "qid": qid,
                "source_tokens": len(source_tokens),
                "truncated_source_tokens": len(truncated_tokens),
                "target_tokens": len(target_tokens),
            }
        )
    batched = probe.batch_length_normalized_loglik(batch_items, batch_size=max(1, min(len(batch_items), int(args.batch_size)))) if batch_items else []
    for single, batch_value in zip(singles, batched):
        max_diffs["batch_single"] = max(max_diffs["batch_single"], abs(single - batch_value))
    payload = {
        "rows": len(rows),
        "identity_delta_policy": "defined_exact_zero",
        "batch_single_max_abs_diff": round(max_diffs["batch_single"], 8),
        "cache_repeat_max_abs_diff": round(max_diffs["cache_repeat"], 8),
        "token_rows": token_rows,
        "passed": max_diffs["batch_single"] <= 1e-4 and max_diffs["cache_repeat"] == 0.0 and all(row["target_tokens"] >= 1 for row in token_rows),
    }
    write_json(payload, report_dir / "likelihood_sanity.json")
    write_markdown(
        [
            "# C-CEE Likelihood Sanity",
            "",
            f"- Passed: `{payload['passed']}`",
            "- Identity delta: `0.0 by construction`",
            f"- Batch/single max abs diff: `{payload['batch_single_max_abs_diff']}`",
            f"- Cache repeat max abs diff: `{payload['cache_repeat_max_abs_diff']}`",
            f"- Rows: `{payload['rows']}`",
        ],
        report_dir / "likelihood_sanity.md",
    )
    return payload


def semantic_capacity_rows(rows: Sequence[dict[str, Any]], *, top_k: int, max_candidates: int, limit: int, seed: int) -> list[dict[str, Any]]:
    complete = []
    for record in rows:
        example = make_example(record, max_candidates=max_candidates, top_k=top_k, embedding_features={})
        base = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
        metrics = support_metrics_for_indices(example, base)
        if float(metrics.get("support_complete") or 0.0) >= 1.0 and normalize_answer(gold_answers(record)[0]):
            complete.append(record)
    complete = complete[: int(limit)]
    rng = random.Random(int(seed))
    answers = [gold_answers(row)[0] for row in complete]
    output = []
    for record in complete:
        answer = gold_answers(record)[0]
        distractors = [item for item in answers if normalize_answer(item) != normalize_answer(answer)]
        if not distractors:
            continue
        output.append({"record": record, "gold_answer": answer, "distractor_answer": rng.choice(distractors)})
    return output


def run_reader_semantic_capacity(args: argparse.Namespace, rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    report_dir = Path(args.output_dir)
    rows = dev_rows(rows or load_jsonl(args.cache_jsonl), int(args.dev_start), int(args.dev_end))
    sample = semantic_capacity_rows(
        rows,
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        limit=int(args.semantic_limit),
        seed=int(args.seed),
    )
    probe = load_probe(args)
    pos_scores = []
    neg_scores = []
    for item in sample:
        record = item["record"]
        qid = row_qid(record)
        base = rank_indices_for(record, top_k=int(args.top_k), max_candidates=int(args.max_candidates))
        docs = selected_docs(record, base)
        question = str(record.get("question") or "")
        pos_scores.append(probe.length_normalized_loglik(qid, question, docs, str(item["gold_answer"])))
        neg_scores.append(probe.length_normalized_loglik(qid, question, docs, str(item["distractor_answer"])))
    ci = bootstrap_auc_ci(pos_scores, neg_scores, seed=int(args.seed), resamples=int(args.bootstrap_resamples))
    paired_win = sum(1 for pos, neg in zip(pos_scores, neg_scores) if pos > neg) / max(1, len(pos_scores))
    payload = {
        "sample_size": len(sample),
        "auc": ci["auc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "mean_positive_loglik": round(mean(pos_scores), 6) if pos_scores else 0.0,
        "mean_distractor_loglik": round(mean(neg_scores), 6) if neg_scores else 0.0,
        "mean_gap": round(mean([pos - neg for pos, neg in zip(pos_scores, neg_scores)]), 6) if pos_scores else 0.0,
        "paired_win_rate": round(paired_win, 6),
        "passed": bool(len(sample) > 0 and ci["auc"] >= 0.80),
    }
    write_json(payload, report_dir / "reader_semantic_capacity.json")
    write_markdown(
        [
            "# C-CEE Reader Semantic Capacity",
            "",
            f"- Passed: `{payload['passed']}`",
            f"- Sample size: `{payload['sample_size']}`",
            f"- AUC: `{payload['auc']}`",
            f"- 95% CI: `[{payload['ci_low']}, {payload['ci_high']}]`",
            f"- Paired win rate: `{payload['paired_win_rate']}`",
            f"- Mean gap: `{payload['mean_gap']}`",
        ],
        report_dir / "reader_semantic_capacity.md",
    )
    return payload


def lowest_rank_non_gold_in_s0(record: dict[str, Any], base: Sequence[int]) -> int | None:
    candidates = list(record.get("candidates") or [])
    non_gold = [idx for idx in base if 0 <= int(idx) < len(candidates) and not is_gold(candidates[int(idx)])]
    return max(non_gold) if non_gold else None


def edit_category(edit: dict[str, Any], preferred_remove: int | None) -> str:
    if bool(edit.get("beneficial")) and int(edit.get("added_gold") or 0) == 1 and int(edit.get("removed_non_gold") or 0) == 1:
        return "oracle_beneficial"
    if (
        not bool(edit.get("beneficial"))
        and int(edit.get("added_non_gold") or 0) == 1
        and str(edit.get("hard_negative_type") or "") == "lexical_hard_negative"
        and (preferred_remove is None or int(edit.get("remove_index")) == int(preferred_remove))
    ):
        return "lexical_HN"
    before = float(edit.get("before_metrics", {}).get("support_complete") or 0.0)
    after = float(edit.get("after_metrics", {}).get("support_complete") or 0.0)
    if not bool(edit.get("beneficial")) and after <= before:
        return "random_non_beneficial"
    return ""


def generate_hypotheses_for_record(
    record: dict[str, Any],
    *,
    dense_by_qid: dict[str, dict[str, Any]],
    probe: FrozenReaderLoglikProbe,
    qwen_available: bool,
    llm_endpoint: str,
    top_k: int,
    max_candidates: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    qid = row_qid(record)
    question = str(record.get("question") or "")
    base = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
    prop_docs = selected_docs(record, base)
    evidence_sets = {"prop": prop_docs}
    dense = dense_by_qid.get(qid)
    if dense:
        evidence_sets["dense"] = selected_docs(dense, base)
        seen = {str(doc.get("title") or "") for doc in prop_docs}
        union_docs = list(prop_docs)
        for doc in selected_docs(dense, list(range(min(20, len(dense.get("candidates") or []))))):
            if str(doc.get("title") or "") not in seen:
                union_docs.append(doc)
                seen.add(str(doc.get("title") or ""))
            if len(union_docs) >= top_k:
                break
        evidence_sets["union"] = union_docs[:top_k]
    hypotheses = []
    for source, docs in evidence_sets.items():
        answer = probe.generate_answer(qid, question, docs, max_new_tokens=max_new_tokens)
        if normalize_answer(answer):
            loglik = probe.length_normalized_loglik(qid, question, docs, answer)
            hypotheses.append({"source": source, "text": answer, "loglik": loglik})
    if qwen_available:
        try:
            llm_docs = selected_docs(record, list(range(min(10, len(record.get("candidates") or [])))))
            answer = qwen_answer(llm_endpoint, question, llm_docs)
            if normalize_answer(answer):
                loglik = probe.length_normalized_loglik(qid, question, prop_docs, answer)
                hypotheses.append({"source": "qwen", "text": answer, "loglik": loglik})
        except Exception:
            pass
    return hypotheses


def score_edit_deltas(
    record: dict[str, Any],
    edits: Sequence[dict[str, Any]],
    answers: Sequence[dict[str, Any]],
    *,
    probe: FrozenReaderLoglikProbe,
    top_k: int,
    max_candidates: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    qid = row_qid(record)
    question = str(record.get("question") or "")
    base = rank_indices_for(record, top_k=top_k, max_candidates=max_candidates)
    base_docs = selected_docs(record, base)
    base_items = [
        {"qid": qid, "question": question, "docs": base_docs, "answer": str(answer["canonical_text"])}
        for answer in answers
    ]
    base_scores = {
        str(answer["normalized"]): value
        for answer, value in zip(answers, probe.batch_length_normalized_loglik(base_items, batch_size=batch_size))
    }
    all_items = []
    refs = []
    for edit in edits:
        edited_indices = action_indices(base, edit)
        docs = selected_docs(record, edited_indices)
        for answer in answers:
            all_items.append({"qid": qid, "question": question, "docs": docs, "answer": str(answer["canonical_text"])})
            refs.append((edit, str(answer["normalized"])))
    edit_rows: dict[tuple[int, int], dict[str, Any]] = {}
    values = probe.batch_length_normalized_loglik(all_items, batch_size=batch_size) if all_items else []
    for (edit, norm), value in zip(refs, values):
        key = (int(edit["remove_index"]), int(edit["add_index"]))
        edit_rows.setdefault(key, {"edit": edit, "deltas": {}})
        edit_rows[key]["deltas"][norm] = float(value) - float(base_scores[norm])
    return list(edit_rows.values())


def summarize_decision_predictions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    rank_incomplete = [row for row in rows if not row["rank_complete"]]
    rank_complete = [row for row in rows if row["rank_complete"]]
    edits = [row for row in rows if not row["is_stop"]]
    added_gold = sum(int(row.get("added_gold") or 0) for row in edits)
    added_non_gold = sum(int(row.get("added_non_gold") or 0) for row in edits)
    before_complete = mean([float(row["before_support_complete"]) for row in rows])
    after_complete = mean([float(row["after_support_complete"]) for row in rows])
    before_recall = mean([float(row["before_support_recall"]) for row in rows])
    after_recall = mean([float(row["after_support_recall"]) for row in rows])
    return {
        "queries": len(rows),
        "stop_rate": round(1.0 - len(edits) / max(1.0, float(len(rows))), 6),
        "top1_beneficial_rate_rank_incomplete": round(
            sum(1 for row in rank_incomplete if row.get("chosen_beneficial")) / max(1.0, float(len(rank_incomplete))),
            6,
        ),
        "rank_complete_false_edit_rate": round(
            sum(1 for row in rank_complete if not row["is_stop"]) / max(1.0, float(len(rank_complete))),
            6,
        ),
        "added_gold": added_gold,
        "added_non_gold": added_non_gold,
        "non_gold_per_gold": round(float(added_non_gold) / max(1.0, float(added_gold)), 6),
        "support_complete_before": round(before_complete, 6),
        "support_complete_after": round(after_complete, 6),
        "support_complete_gain": round(after_complete - before_complete, 6),
        "support_recall_before": round(before_recall, 6),
        "support_recall_after": round(after_recall, 6),
        "support_recall_gain": round(after_recall - before_recall, 6),
    }


def run_decision_separability(args: argparse.Namespace, rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    report_dir = Path(args.output_dir)
    rows = dev_rows(rows or load_jsonl(args.cache_jsonl), int(args.dev_start), int(args.dev_end))
    dense_rows = load_jsonl(args.dense_cache_jsonl, limit=int(args.dev_end))
    dense_by_qid = {row_qid(row): row for row in dense_rows}
    embedding_features, embedding_feature_names = load_embedding_features(args.selector_embedding_npz, limit=int(args.dev_end))
    probe = load_probe(args)
    qwen = qwen_health(str(args.llm_endpoint))
    qwen_available = bool(qwen.get("available"))
    rng = random.Random(int(args.seed))
    sampled = {"oracle_beneficial": [], "lexical_HN": [], "random_non_beneficial": []}
    non_oracle_scores = {"oracle_beneficial": [], "lexical_HN": [], "random_non_beneficial": []}
    decision_rows = []
    skipped = 0
    forward_count = 0
    start_time = time.time()
    for record in rows:
        example = make_example(record, max_candidates=int(args.max_candidates), top_k=int(args.top_k), embedding_features=embedding_features)
        base = rank_indices_for(record, top_k=int(args.top_k), max_candidates=int(args.max_candidates))
        before_metrics = support_metrics_for_indices(example, base)
        edits = enumerate_edit_summaries(
            record,
            top_k=int(args.top_k),
            max_candidates=int(args.max_candidates),
            candidate_pool_size=int(args.candidate_pool_size),
            example=example,
            embedding_by_qid=embedding_features,
            embedding_feature_names=embedding_feature_names,
        )
        preferred_remove = lowest_rank_non_gold_in_s0(record, base)
        for edit in edits:
            category = edit_category(edit, preferred_remove)
            if category in sampled and len(sampled[category]) < int(args.sample_per_category):
                sampled[category].append({"qid": row_qid(record), "edit": edit})
        hypotheses = generate_hypotheses_for_record(
            record,
            dense_by_qid=dense_by_qid,
            probe=probe,
            qwen_available=qwen_available,
            llm_endpoint=str(args.llm_endpoint),
            top_k=int(args.top_k),
            max_candidates=int(args.max_candidates),
            max_new_tokens=int(args.max_target_tokens),
        )
        clusters = cluster_answers(hypotheses)
        if not clusters:
            skipped += 1
            continue
        edit_scores = score_edit_deltas(
            record,
            edits,
            clusters,
            probe=probe,
            top_k=int(args.top_k),
            max_candidates=int(args.max_candidates),
            batch_size=int(args.batch_size),
        )
        edit_by_key = {(int(edit["remove_index"]), int(edit["add_index"])): edit for edit in edits}
        for scored in edit_scores:
            edit = edit_by_key[(int(scored["edit"]["remove_index"]), int(scored["edit"]["add_index"]))]
            category = edit_category(edit, preferred_remove)
            if category in non_oracle_scores and len(non_oracle_scores[category]) < int(args.sample_per_category):
                non_oracle_scores[category].append(max(float(value) for value in scored.get("deltas", {}).values()))
        forward_count += len(clusters) * (1 + len(edits))
        decision = posterior_decision(edit_scores, clusters, pi_stop=float(args.pi_stop))
        chosen_edit = decision.get("edit")
        selected = list(base)
        added_gold = added_non_gold = 0
        if chosen_edit is not None:
            selected = action_indices(base, chosen_edit)
            added_gold = int(chosen_edit.get("added_gold") or 0)
            added_non_gold = int(chosen_edit.get("added_non_gold") or 0)
        after_metrics = support_metrics_for_indices(example, selected)
        decision_rows.append(
            {
                "qid": row_qid(record),
                "rank_complete": float(before_metrics.get("support_complete") or 0.0) >= 1.0,
                "is_stop": bool(decision.get("is_stop")),
                "chosen_beneficial": bool(chosen_edit.get("beneficial")) if chosen_edit is not None else False,
                "chosen_hard_negative_type": str(chosen_edit.get("hard_negative_type") or "") if chosen_edit is not None else "",
                "added_gold": added_gold,
                "added_non_gold": added_non_gold,
                "before_support_complete": float(before_metrics.get("support_complete") or 0.0),
                "after_support_complete": float(after_metrics.get("support_complete") or 0.0),
                "before_support_recall": float(before_metrics.get("support_recall") or 0.0),
                "after_support_recall": float(after_metrics.get("support_recall") or 0.0),
                "answer": decision.get("answer", {}).get("canonical_text") if decision.get("answer") else "",
                "score": finite(float(decision.get("score") or 0.0)),
            }
        )
    elapsed = time.time() - start_time

    def sampled_scores(category: str, *, use_gold: bool) -> list[float]:
        output = []
        for row in sampled[category]:
            record = next(item for item in rows if row_qid(item) == row["qid"])
            answer = gold_answers(record)[0]
            if not use_gold:
                answer = gold_answers(record)[0]
            base = rank_indices_for(record, top_k=int(args.top_k), max_candidates=int(args.max_candidates))
            edited = action_indices(base, row["edit"])
            question = str(record.get("question") or "")
            base_score = probe.length_normalized_loglik(row["qid"], question, selected_docs(record, base), answer)
            edit_score = probe.length_normalized_loglik(row["qid"], question, selected_docs(record, edited), answer)
            output.append(edit_score - base_score)
        return output

    oracle_pos = sampled_scores("oracle_beneficial", use_gold=True)
    oracle_lex = sampled_scores("lexical_HN", use_gold=True)
    oracle_rand = sampled_scores("random_non_beneficial", use_gold=True)
    oracle_ci = bootstrap_auc_ci(oracle_pos, oracle_lex, seed=int(args.seed), resamples=int(args.bootstrap_resamples))
    random_ci = bootstrap_auc_ci(oracle_pos, oracle_rand, seed=int(args.seed), resamples=int(args.bootstrap_resamples))
    non_oracle_ci = bootstrap_auc_ci(
        non_oracle_scores["oracle_beneficial"],
        non_oracle_scores["lexical_HN"],
        seed=int(args.seed),
        resamples=int(args.bootstrap_resamples),
    )
    non_oracle_random_ci = bootstrap_auc_ci(
        non_oracle_scores["oracle_beneficial"],
        non_oracle_scores["random_non_beneficial"],
        seed=int(args.seed),
        resamples=int(args.bootstrap_resamples),
    )
    non_oracle_summary = summarize_decision_predictions(decision_rows)
    checks = {
        "auc_ge_0_70": non_oracle_ci["auc"] >= 0.70,
        "auc_ci_low_ge_0_65": non_oracle_ci["ci_low"] >= 0.65,
        "top1_beneficial_rate_ge_0_30": float(non_oracle_summary.get("top1_beneficial_rate_rank_incomplete", 0.0)) >= 0.30,
        "rank_complete_false_edit_rate_le_0_05": float(non_oracle_summary.get("rank_complete_false_edit_rate", 1.0)) <= 0.05,
        "non_gold_per_gold_le_5": float(non_oracle_summary.get("non_gold_per_gold", 999.0)) <= 5.0,
        "support_complete_gain_ge_1pp": float(non_oracle_summary.get("support_complete_gain", 0.0)) >= 0.01,
    }
    payload = {
        "rows": len(rows),
        "skipped_no_answer_clusters": skipped,
        "qwen": qwen,
        "sampled_counts": {key: len(value) for key, value in sampled.items()},
        "oracle_answer_diagnostic": {
            "beneficial_vs_lexical_hn": oracle_ci,
            "beneficial_vs_random_non_beneficial": random_ci,
            "pr_auc_beneficial_vs_lexical_hn": round(pr_auc(oracle_pos + oracle_lex, [1] * len(oracle_pos) + [0] * len(oracle_lex)), 6)
            if oracle_pos and oracle_lex
            else 0.0,
            "mean_delta_beneficial": round(mean(oracle_pos), 6) if oracle_pos else 0.0,
            "mean_delta_lexical_hn": round(mean(oracle_lex), 6) if oracle_lex else 0.0,
            "mean_delta_random_non_beneficial": round(mean(oracle_rand), 6) if oracle_rand else 0.0,
        },
        "non_oracle_auc_diagnostic": {
            "beneficial_vs_lexical_hn": non_oracle_ci,
            "beneficial_vs_random_non_beneficial": non_oracle_random_ci,
            "pr_auc_beneficial_vs_lexical_hn": round(
                pr_auc(
                    non_oracle_scores["oracle_beneficial"] + non_oracle_scores["lexical_HN"],
                    [1] * len(non_oracle_scores["oracle_beneficial"]) + [0] * len(non_oracle_scores["lexical_HN"]),
                ),
                6,
            )
            if non_oracle_scores["oracle_beneficial"] and non_oracle_scores["lexical_HN"]
            else 0.0,
            "mean_delta_beneficial": round(mean(non_oracle_scores["oracle_beneficial"]), 6)
            if non_oracle_scores["oracle_beneficial"]
            else 0.0,
            "mean_delta_lexical_hn": round(mean(non_oracle_scores["lexical_HN"]), 6)
            if non_oracle_scores["lexical_HN"]
            else 0.0,
            "mean_delta_random_non_beneficial": round(mean(non_oracle_scores["random_non_beneficial"]), 6)
            if non_oracle_scores["random_non_beneficial"]
            else 0.0,
        },
        "non_oracle_decision_diagnostic": non_oracle_summary,
        "checks": checks,
        "passed": all(checks.values()),
        "reader_forwards": forward_count,
        "mean_reader_forwards_per_query": round(forward_count / max(1.0, float(len(decision_rows))), 3),
        "elapsed_seconds": round(elapsed, 3),
    }
    write_json(payload, report_dir / "decision_separability_dev.json")
    write_cases_csv(decision_rows, report_dir / "decision_separability_dev_cases.csv")
    write_markdown(
        [
            "# C-CEE Decision-Level Separability Dev",
            "",
            f"- Passed: `{payload['passed']}`",
            f"- Rows: `{payload['rows']}`",
            f"- Skipped no-answer-cluster rows: `{skipped}`",
            f"- Qwen: `{qwen}`",
            f"- Sampled counts: `{payload['sampled_counts']}`",
            "",
            "## Oracle-Answer Diagnostic",
            "",
            f"- AUC beneficial vs lexical HN: `{oracle_ci['auc']}`",
            f"- 95% CI: `[{oracle_ci['ci_low']}, {oracle_ci['ci_high']}]`",
            f"- AUC beneficial vs random: `{random_ci['auc']}`",
            "",
            "## Non-Oracle Decision Diagnostic",
            "",
            f"- AUC beneficial vs lexical HN: `{non_oracle_ci['auc']}`",
            f"- 95% CI: `[{non_oracle_ci['ci_low']}, {non_oracle_ci['ci_high']}]`",
            f"- AUC beneficial vs random: `{non_oracle_random_ci['auc']}`",
            f"- Summary: `{non_oracle_summary}`",
            f"- Checks: `{checks}`",
            f"- Mean reader forwards/query: `{payload['mean_reader_forwards_per_query']}`",
        ],
        report_dir / "decision_separability_dev.md",
    )
    return payload


def write_cases_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qid",
        "rank_complete",
        "is_stop",
        "chosen_beneficial",
        "chosen_hard_negative_type",
        "added_gold",
        "added_non_gold",
        "before_support_complete",
        "after_support_complete",
        "before_support_recall",
        "after_support_recall",
        "answer",
        "score",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_day1_summary(args: argparse.Namespace, decision: str, reports: dict[str, Any]) -> dict[str, Any]:
    if decision not in DECISION_VALUES:
        raise ValueError(f"Unknown Day-1 decision: {decision}")
    payload = {"decision": decision, "reports": reports}
    out_dir = Path(args.output_dir)
    write_json(payload, out_dir / "day1_summary.json")
    lines = ["# C-CEE Day-1 Summary", "", f"- Decision: `{decision}`", ""]
    for name, report in reports.items():
        lines.append(f"- {name}: passed=`{report.get('passed')}`")
    write_markdown(lines, out_dir / "day1_summary.md")
    return payload


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.cache_jsonl, limit=max(int(args.dev_end), 1000))
    reports: dict[str, Any] = {}
    reports["asset_audit"] = run_asset_audit(args)
    reports["likelihood_sanity"] = run_likelihood_sanity(args, rows)
    if not reports["likelihood_sanity"].get("passed"):
        return write_day1_summary(args, "STOP_MECHANICAL_FAIL", reports)
    reports["reader_semantic_capacity"] = run_reader_semantic_capacity(args, rows)
    if reports["reader_semantic_capacity"].get("sample_size", 0) <= 0:
        return write_day1_summary(args, "STOP_INSUFFICIENT_DEV_SAMPLE", reports)
    if not reports["reader_semantic_capacity"].get("passed"):
        return write_day1_summary(args, "STOP_READER_SEMANTIC_CAPACITY_FAIL", reports)
    reports["decision_separability"] = run_decision_separability(args, rows)
    if not reports["decision_separability"].get("passed"):
        return write_day1_summary(args, "STOP_DECISION_SEPARABILITY_FAIL", reports)
    return write_day1_summary(args, "PROCEED_PHASE2", reports)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all", choices=["asset-audit", "likelihood-sanity", "reader-semantic-capacity", "decision-separability", "all"])
    parser.add_argument("--cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--dense_cache_jsonl", default="data/dpathrag/cache/2wiki_dense_pool100_smoke.jsonl")
    parser.add_argument("--reader_model", default="data/dpathrag/models/flan_t5_base_gold_k5_5k")
    parser.add_argument("--selector_embedding_npz", default="data/dpathrag/cache/selector_embedding_proprag_local1000_rp64.npz")
    parser.add_argument("--dev_start", type=int, default=0)
    parser.add_argument("--dev_end", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_target_tokens", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pi_stop", type=float, default=0.9)
    parser.add_argument("--llm_endpoint", default="http://localhost:8039/v1")
    parser.add_argument("--output_dir", default="reports/cee")
    parser.add_argument("--cache_dir", default="data/dpathrag/cache/ccee_day1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--semantic_limit", type=int, default=200)
    parser.add_argument("--sample_per_category", type=int, default=200)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "asset-audit":
        payload = run_asset_audit(args)
    elif args.command == "likelihood-sanity":
        payload = run_likelihood_sanity(args)
    elif args.command == "reader-semantic-capacity":
        payload = run_reader_semantic_capacity(args)
    elif args.command == "decision-separability":
        payload = run_decision_separability(args)
    else:
        payload = run_all(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
