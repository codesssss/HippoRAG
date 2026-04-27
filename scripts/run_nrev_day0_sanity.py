#!/usr/bin/env python3
"""NREV Day-0 gold-vs-plausible-wrong sanity check.

This is deliberately Day-0 only:
- 2Wiki first-N queries.
- Gold answer/evidence pair vs one plausible wrong answer/evidence pair.
- No full fixed-pool prototype.
- No NLI or LLM judge for destructive perturbations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable, Sequence
import urllib.request

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from run_daec_consistency_probe import (  # noqa: E402
    norm_text,
    split_title_text,
)
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import token_f1  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
DATE_WORD_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)

T_PLUS_VARIANTS = ("dependency", "reverse", "rotate_left", "edge_emphasis")
NREV_VARIANTS = ("l_plus", "rev", "nrev_no_alt", "nrev_no_l0", "nrev_full")
ANSWER_TYPE_ORDER = ("date", "number", "yes_no", "person", "place", "organization", "work", "entity")
ANSWER_TYPE_CHOICES = set(ANSWER_TYPE_ORDER)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def tokens(value: Any) -> set[str]:
    return set(TOKEN_RE.findall(norm_text(str(value or ""))))


def title_key(value: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(value or "").lower()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def doc_title(doc: dict[str, Any]) -> str:
    return str(doc.get("title") or "")


def doc_text(doc: dict[str, Any]) -> str:
    return str(doc.get("text") or "")


def make_doc(title: str, text: str, *, doc_id: Any, rank: int, gold_support: bool = False) -> dict[str, Any]:
    return {
        "title": str(title or ""),
        "text": str(text or ""),
        "doc_id": doc_id,
        "rank": int(rank),
        "gold_support": int(bool(gold_support)),
    }


def pool_docs(pool_record: dict[str, Any]) -> list[dict[str, Any]]:
    docs = []
    gold_titles = {norm_text(title) for title in pool_record.get("gold_titles") or [] if norm_text(title)}
    ids = list(pool_record.get("pool_doc_ids") or [])
    titles = list(pool_record.get("pool_titles") or [])
    texts = list(pool_record.get("pool_docs") or [])
    for idx, text in enumerate(texts):
        title, _body = split_title_text(str(text), titles[idx] if idx < len(titles) else "")
        if idx < len(titles) and titles[idx]:
            title = str(titles[idx])
        docs.append(
            make_doc(
                title,
                str(text),
                doc_id=ids[idx] if idx < len(ids) else idx,
                rank=idx + 1,
                gold_support=norm_text(title) in gold_titles,
            )
        )
    return docs


def gold_docs_padded(pool_record: dict[str, Any], *, k: int) -> list[dict[str, Any]]:
    docs = []
    seen_titles: set[str] = set()
    for idx, raw in enumerate(pool_record.get("gold_docs") or []):
        title, _body = split_title_text(str(raw), "")
        key = norm_text(title)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        docs.append(make_doc(title, str(raw), doc_id=f"gold::{idx}", rank=0, gold_support=True))
    for doc in pool_docs(pool_record):
        key = norm_text(doc_title(doc))
        if key and key in seen_titles:
            continue
        docs.append(doc)
        seen_titles.add(key)
        if len(docs) >= int(k):
            break
    return docs[: int(k)]


def choose_wrong_answer(caps_row: dict[str, Any], gold_answers: Sequence[str]) -> str:
    gold_norms = {norm_text(answer) for answer in gold_answers if norm_text(answer)}
    for candidate in caps_row.get("top5") or []:
        answer = str(candidate.get("answer") or "").strip()
        if not answer or bool(candidate.get("is_gold")):
            continue
        if norm_text(answer) in gold_norms:
            continue
        # Drop obvious prompt leakage candidates from CAPS candidate generation.
        if "let's tackle" in answer.lower() or len(answer.split()) > 12:
            continue
        return answer
    answer = str(caps_row.get("best_answer") or "").strip()
    return answer if norm_text(answer) not in gold_norms else ""


def wrong_docs(pool_record: dict[str, Any], wrong_answer: str, *, k: int) -> list[dict[str, Any]]:
    docs = pool_docs(pool_record)
    answer_norm = norm_text(wrong_answer)
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if answer_norm:
        for doc in docs:
            blob = norm_text(f"{doc_title(doc)}\n{doc_text(doc)}")
            if answer_norm and answer_norm in blob:
                selected.append(doc)
                seen_ids.add(str(doc.get("doc_id")))
                if len(selected) >= int(k):
                    return selected[: int(k)]
    for doc in docs:
        if str(doc.get("doc_id")) in seen_ids:
            continue
        selected.append(doc)
        seen_ids.add(str(doc.get("doc_id")))
        if len(selected) >= int(k):
            break
    return selected[: int(k)]


def answer_surface_type(answer: str, question: str = "") -> str:
    raw = str(answer or "").strip()
    norm = norm_text(raw)
    q = norm_text(question)
    if norm in {"yes", "no"}:
        return "yes_no"
    if YEAR_RE.search(raw) or DATE_WORD_RE.search(raw):
        return "date"
    if re.fullmatch(r"[\d,.\s%-]+", raw) and re.search(r"\d", raw):
        return "number"
    if "place of birth" in q or "birthplace" in q or "born in" in q:
        return "place"
    if q.startswith("where") or "what city" in q or "which city" in q or "what country" in q or "which country" in q:
        return "place"
    if q.startswith("when") or "what year" in q or "which year" in q:
        return "date" if re.search(r"\d", raw) else "entity"
    if q.startswith("how many") or "number" in q:
        return "number" if re.search(r"\d", raw) else "entity"
    if q.startswith("who"):
        return "person"
    if "film" in q or "song" in q or "album" in q or "book" in q:
        return "work"
    return "entity"


def heuristic_question_type(question: str) -> str:
    q = norm_text(question)
    if "place of birth" in q or "birthplace" in q or "born in" in q:
        return "place"
    if q.startswith("where") or "what city" in q or "which city" in q or "what country" in q or "which country" in q:
        return "place"
    if q.startswith("when") or "what year" in q or "which year" in q:
        return "date"
    if q.startswith("how many") or "number" in q:
        return "number"
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ")):
        return "yes_no"
    if q.startswith("who"):
        return "person"
    if "film" in q or "song" in q or "album" in q or "book" in q:
        return "work"
    return "entity"


class JsonlValueCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.values: dict[str, Any] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.values[str(row["key"])] = row.get("value")

    def get(self, key: str) -> Any | None:
        return self.values.get(str(key))

    def set_many(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        for row in rows:
            self.values[str(row["key"])] = row.get("value")
        append_jsonl(rows, self.path)

    def set(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        self.set_many([{"key": str(key), "value": value, "metadata": metadata or {}}])


class QwenPromptLogprobScorer:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        cache_path: str | Path,
        timeout: int,
        concurrency: int,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.timeout = int(timeout)
        self.concurrency = int(concurrency)
        self.cache = JsonlValueCache(cache_path)
        self.token_cache: dict[str, list[int]] = {}

    def reader_prefix(self, question: str, docs: Sequence[dict[str, Any]]) -> str:
        evidence = "\n\n".join(
            f"Wikipedia Title: {doc_text(doc)}"
            for doc in docs
        )
        if evidence:
            return f"{evidence}\n\nQuestion: {question}\nAnswer:"
        return f"Question: {question}\nAnswer:"

    def tokenize(self, prompt: str) -> list[int]:
        key = sha1_json({"model": self.model, "prompt": prompt})
        cached = self.token_cache.get(key)
        if cached is not None:
            return cached
        body = {"model": self.model, "prompt": prompt}
        req = urllib.request.Request(
            self.base_url.rsplit("/v1", 1)[0] + "/tokenize",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        tokens = [int(tok) for tok in payload.get("tokens") or []]
        self.token_cache[key] = tokens
        return tokens

    def prompt_logprobs(self, prompt: str) -> list[dict[str, Any] | None]:
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            self.base_url + "/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return list((payload.get("choices") or [{}])[0].get("prompt_logprobs") or [])

    def score_one(self, item: dict[str, Any]) -> float:
        question = str(item.get("question") or "")
        docs = list(item.get("docs") or [])
        answer = str(item.get("answer") or "")
        if not norm_text(answer):
            return float("-inf")
        prefix = self.reader_prefix(question, docs)
        full_prompt = f"{prefix} {answer}"
        cache_key = sha1_json(
            {
                "reader": "qwen_prompt_logprob",
                "model": self.model,
                "question": question,
                "doc_ids": [doc.get("doc_id") for doc in docs],
                "doc_hashes": [hashlib.sha1(doc_text(doc).encode("utf-8")).hexdigest() for doc in docs],
                "answer": answer,
            }
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return float(cached)
        prefix_tokens = self.tokenize(prefix)
        full_tokens = self.tokenize(full_prompt)
        answer_tokens = full_tokens[len(prefix_tokens) :]
        logprob_rows = self.prompt_logprobs(full_prompt)
        values: list[float] = []
        for idx, token_id in enumerate(answer_tokens, start=len(prefix_tokens)):
            if idx >= len(logprob_rows):
                continue
            row = logprob_rows[idx] or {}
            token_payload = row.get(str(token_id))
            if token_payload is None and row:
                # prompt_logprobs=0 should return the actual token only. This fallback
                # keeps the run robust against endpoint formatting differences.
                token_payload = next(iter(row.values()))
            if token_payload is not None:
                values.append(float(token_payload.get("logprob")))
        score = sum(values) / float(max(1, len(values)))
        self.cache.set(cache_key, score, {"answer": answer, "token_count": len(values)})
        return float(score)

    def score_many(self, items: Sequence[dict[str, Any]]) -> list[float]:
        outputs: list[float | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as executor:
            futures = {executor.submit(self.score_one, dict(item)): idx for idx, item in enumerate(items)}
            for future in as_completed(futures):
                outputs[futures[future]] = future.result()
        return [float(value) for value in outputs if value is not None]


def qwen_type_classify(
    *,
    question: str,
    base_url: str,
    model: str,
    cache: JsonlValueCache,
    timeout: int,
) -> str:
    heuristic = heuristic_question_type(question)
    if heuristic != "entity":
        return heuristic
    key = sha1_json({"type_classifier": "qwen", "model": model, "question": question})
    cached = cache.get(key)
    if cached in ANSWER_TYPE_CHOICES:
        return str(cached)
    prompt = (
        "Classify the expected answer type for the question. "
        f"Return exactly one label from: {', '.join(ANSWER_TYPE_ORDER)}.\n"
        f"Question: {question}\nLabel:"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    try:
        req = urllib.request.Request(
            str(base_url).rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        raw = str((payload.get("choices") or [{}])[0].get("message", {}).get("content") or "").lower()
        label = next((choice for choice in ANSWER_TYPE_ORDER if re.search(rf"\b{re.escape(choice)}\b", raw)), "")
        if not label:
            label = heuristic
    except Exception:
        label = heuristic
    cache.set(key, label, {"question": question})
    return label


def extract_final_answer(text: str) -> str:
    raw = str(text or "").strip()
    matches = re.findall(r"(?im)^answer\s*:\s*(.+?)\s*$", raw)
    if matches:
        return matches[-1].strip()
    return raw.splitlines()[-1].strip() if raw else ""


def call_closed_book_reader(
    *,
    question: str,
    base_url: str,
    model: str,
    max_new_tokens: int,
    timeout: int,
) -> str:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the question from your own knowledge. If uncertain, give your best short answer. "
                    "Do not say that passages or evidence are missing. End with exactly one final line: "
                    "Answer: <short answer>."
                ),
            },
            {
                "role": "user",
                "content": f"/no_think\nQuestion: {question}\nKeep reasoning brief, then give the final answer.",
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_new_tokens),
    }
    req = urllib.request.Request(
        str(base_url).rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    raw = str((payload.get("choices") or [{}])[0].get("message", {}).get("content") or "")
    return extract_final_answer(raw)


def doc_blob(doc: dict[str, Any]) -> str:
    return f"{doc_title(doc)}\n{doc_text(doc)}"


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(max(1, len(left | right)))


def matched_replacement(removed: dict[str, Any], current: Sequence[dict[str, Any]], candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    current_ids = {str(doc.get("doc_id")) for doc in current}
    current_titles = {title_key(doc_title(doc)) for doc in current if title_key(doc_title(doc))}
    removed_tokens = tokens(doc_blob(removed))
    removed_title_tokens = tokens(doc_title(removed))
    rows = []
    for cand in candidates:
        if str(cand.get("doc_id")) in current_ids:
            continue
        # A matched replacement is supposed to be a destructive hard negative.
        # Same-title duplicates usually preserve the same evidence cell.
        if title_key(doc_title(cand)) in current_titles:
            continue
        cand_tokens = tokens(doc_blob(cand))
        cand_title_tokens = tokens(doc_title(cand))
        rows.append(
            (
                jaccard(removed_tokens, cand_tokens),
                jaccard(removed_title_tokens, cand_title_tokens),
                -int(cand.get("rank") or 999999),
                cand,
            )
        )
    if not rows:
        return None
    rows.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return dict(rows[0][3])


def t_plus_contexts(docs: Sequence[dict[str, Any]], answer: str) -> list[tuple[str, list[dict[str, Any]]]]:
    values = [dict(doc) for doc in docs]
    if not values:
        return [("dependency", [])]
    rotated = values[1:] + values[:1]
    answer_norm = norm_text(answer)
    key_docs = [doc for doc in values if int(doc.get("gold_support") or 0) or (answer_norm and answer_norm in norm_text(doc_blob(doc)))]
    non_key_docs = [doc for doc in values if doc not in key_docs]
    if len(key_docs) >= 2:
        edge = [key_docs[0]] + non_key_docs + key_docs[1:]
    elif key_docs:
        edge = [key_docs[0]] + non_key_docs
    else:
        edge = values
    return [
        ("dependency", values),
        ("reverse", list(reversed(values))),
        ("rotate_left", rotated),
        ("edge_emphasis", edge[: len(values)]),
    ]


def t_minus_contexts(docs: Sequence[dict[str, Any]], pool: Sequence[dict[str, Any]], *, max_cells: int = 5) -> list[tuple[str, list[dict[str, Any]]]]:
    values = [dict(doc) for doc in docs]
    contexts: list[tuple[str, list[dict[str, Any]]]] = []
    for idx, removed in enumerate(values[: int(max_cells)]):
        contexts.append((f"delete_{idx}", [doc for pos, doc in enumerate(values) if pos != idx]))
        repl = matched_replacement(removed, values, pool)
        if repl is not None:
            edited = [dict(doc) for doc in values]
            edited[idx] = repl
            contexts.append((f"replace_{idx}", edited))
    return contexts


def score_pair_components(
    *,
    scorer: QwenPromptLogprobScorer,
    qid: str,
    question: str,
    answer: str,
    docs: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    alt_answers: Sequence[str],
    answer_type: str,
    alt_types: dict[str, str],
) -> dict[str, Any]:
    t_plus = t_plus_contexts(docs, answer)
    t_minus = t_minus_contexts(docs, pool, max_cells=5)
    items = []
    labels = []
    for name, ctx in t_plus:
        labels.append(("plus", name, answer))
        items.append({"qid": qid, "question": question, "docs": ctx, "answer": answer})
    for name, ctx in t_minus:
        labels.append(("minus", name, answer))
        items.append({"qid": qid, "question": question, "docs": ctx, "answer": answer})
    labels.append(("closed_book", "empty", answer))
    items.append({"qid": qid, "question": question, "docs": [], "answer": answer})
    for alt in alt_answers:
        if alt == answer or alt_types.get(alt) != answer_type:
            continue
        for name, ctx in t_plus:
            labels.append(("alt_plus", name, alt))
            items.append({"qid": qid, "question": question, "docs": ctx, "answer": alt})
    scores = scorer.score_many(items)
    rows = [
        {"component": component, "name": name, "answer": ans, "score": score}
        for (component, name, ans), score in zip(labels, scores)
    ]
    plus = [row["score"] for row in rows if row["component"] == "plus"]
    minus = [row["score"] for row in rows if row["component"] == "minus"]
    closed = [row["score"] for row in rows if row["component"] == "closed_book"]
    alt = [row["score"] for row in rows if row["component"] == "alt_plus"]
    l_plus = sum(plus) / float(max(1, len(plus)))
    l_minus = sum(minus) / float(max(1, len(minus)))
    l0 = closed[0] if closed else float("-inf")
    l_alt = max(alt) if alt else float("-inf")
    return {
        "l_plus": l_plus,
        "l_minus": l_minus,
        "l0": l0,
        "l_alt": l_alt,
        "score_rows": rows,
        "t_plus_count": len(plus),
        "t_minus_count": len(minus),
        "alt_count": len(alt),
    }


def nrev_scores(components: dict[str, Any]) -> dict[str, float]:
    lp = safe_float(components.get("l_plus"), float("-inf"))
    lm = safe_float(components.get("l_minus"), float("-inf"))
    l0 = safe_float(components.get("l0"), float("-inf"))
    la = safe_float(components.get("l_alt"), float("-inf"))
    return {
        "l_plus": lp,
        "rev": lp - lm,
        "nrev_no_alt": lp - max(lm, l0),
        "nrev_no_l0": lp - max(lm, la),
        "nrev_full": lp - max(lm, l0, la),
    }


def auc_score(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif math.isclose(pos, neg):
                wins += 0.5
    return wins / total


def paired_win_rate(rows: Sequence[dict[str, Any]], metric: str) -> float:
    if not rows:
        return 0.0
    wins = 0.0
    for row in rows:
        g = safe_float(row["gold_scores"].get(metric))
        w = safe_float(row["wrong_scores"].get(metric))
        if g > w:
            wins += 1.0
        elif math.isclose(g, w):
            wins += 0.5
    return wins / float(len(rows))


def bootstrap_metric(rows: Sequence[dict[str, Any]], metric: str, *, rounds: int, seed: int = 17) -> dict[str, float | None]:
    if not rows:
        return {"auc": None, "ci_low": None, "ci_high": None, "win_rate": 0.0, "n": 0}
    rng = random.Random(seed)
    aucs = []
    values = list(rows)
    for _ in range(max(1, int(rounds))):
        sample = [rng.choice(values) for _idx in values]
        positives = [safe_float(row["gold_scores"].get(metric)) for row in sample]
        negatives = [safe_float(row["wrong_scores"].get(metric)) for row in sample]
        auc = auc_score(positives, negatives)
        if auc is not None:
            aucs.append(float(auc))
    aucs.sort()
    positives = [safe_float(row["gold_scores"].get(metric)) for row in values]
    negatives = [safe_float(row["wrong_scores"].get(metric)) for row in values]
    auc = auc_score(positives, negatives)
    return {
        "auc": None if auc is None else round(float(auc), 6),
        "ci_low": None if not aucs else round(float(aucs[int(0.025 * (len(aucs) - 1))]), 6),
        "ci_high": None if not aucs else round(float(aucs[int(0.975 * (len(aucs) - 1))]), 6),
        "win_rate": round(paired_win_rate(values, metric), 6),
        "n": len(values),
    }


def summarize(rows: Sequence[dict[str, Any]], *, bootstrap_rounds: int) -> dict[str, Any]:
    subsets = {
        "overall": list(rows),
        "closed_book_correct": [row for row in rows if bool(row.get("closed_book_correct"))],
        "closed_book_wrong": [row for row in rows if not bool(row.get("closed_book_correct"))],
    }
    by_metric: dict[str, Any] = {}
    for subset_name, subset_rows in subsets.items():
        by_metric[subset_name] = {
            metric: bootstrap_metric(subset_rows, metric, rounds=bootstrap_rounds)
            for metric in NREV_VARIANTS
        }
    by_type: dict[str, Any] = {}
    for qtype in sorted({str(row.get("question_type") or "unknown") for row in rows}):
        type_rows = [row for row in rows if str(row.get("question_type") or "unknown") == qtype]
        by_type[qtype] = {
            metric: bootstrap_metric(type_rows, metric, rounds=bootstrap_rounds)
            for metric in NREV_VARIANTS
        }
    full = by_metric["overall"]["nrev_full"]
    cb_wrong = by_metric["closed_book_wrong"]["nrev_full"]
    full_auc = full.get("auc") or 0.0
    full_low = full.get("ci_low") or 0.0
    cb_wrong_auc = cb_wrong.get("auc") or 0.0
    if full_auc >= 0.75 and full_low >= 0.70:
        decision = "STRONG_PROCEED_FULL_NREV"
    elif full_auc < 0.75 and cb_wrong_auc >= 0.75:
        decision = "SCOPE_LIMITED_PROCEED_EVIDENCE_REQUIRED"
    elif full_auc >= 0.65:
        decision = "MARGINAL_REDESIGN_T_MINUS_ONCE"
    else:
        decision = "STOP_NREV_DAY0_FAIL"
    return {
        "rows": len(rows),
        "closed_book_correct": sum(1 for row in rows if bool(row.get("closed_book_correct"))),
        "closed_book_wrong": sum(1 for row in rows if not bool(row.get("closed_book_correct"))),
        "metrics": by_metric,
        "by_question_type": by_type,
        "decision": decision,
    }


def mean_finite(values: Iterable[Any]) -> float | None:
    finite = []
    for value in values:
        parsed = safe_float(value, float("nan"))
        if math.isfinite(parsed):
            finite.append(parsed)
    if not finite:
        return None
    return sum(finite) / float(len(finite))


def dominant_null(components: dict[str, Any]) -> str:
    candidates = {
        "l_minus": safe_float(components.get("l_minus"), float("-inf")),
        "l0": safe_float(components.get("l0"), float("-inf")),
        "l_alt": safe_float(components.get("l_alt"), float("-inf")),
    }
    finite = {key: value for key, value in candidates.items() if math.isfinite(value)}
    if not finite:
        return "none"
    return max(finite, key=finite.get)


def diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paired = {}
    for metric in NREV_VARIANTS:
        wins = sum(1 for row in rows if safe_float(row["gold_scores"].get(metric)) > safe_float(row["wrong_scores"].get(metric)))
        ties = sum(
            1
            for row in rows
            if math.isclose(safe_float(row["gold_scores"].get(metric)), safe_float(row["wrong_scores"].get(metric)))
        )
        paired[metric] = {"wins": wins, "ties": ties, "losses": len(rows) - wins - ties}

    component_means: dict[str, dict[str, Any]] = {}
    dominant: dict[str, Any] = {}
    for side in ("gold", "wrong"):
        component_key = f"{side}_components"
        dominant[side] = dict(Counter(dominant_null(row[component_key]) for row in rows))
        component_means[side] = {}
        for key in ("l_plus", "l_minus", "l0", "l_alt"):
            value = mean_finite(row[component_key].get(key) for row in rows)
            component_means[side][key] = None if value is None else round(value, 6)

    alt_counts = Counter(
        f"gold={row['gold_components'].get('alt_count', 0)},wrong={row['wrong_components'].get('alt_count', 0)}"
        for row in rows
    )
    return {
        "paired_outcomes": paired,
        "dominant_null_counts": dominant,
        "component_means": component_means,
        "alt_count_pairs": dict(alt_counts),
        "question_type_counts": dict(Counter(str(row.get("question_type") or "unknown") for row in rows)),
    }


def write_markdown(output: dict[str, Any], path: str | Path) -> None:
    summary = output["summary"]
    diag = output.get("diagnostics") or {}
    lines = [
        "# NREV Day-0 Sanity",
        "",
        "## Configuration",
        "",
        f"- rows: `{summary['rows']}`",
        f"- scorer: `{output['runtime']['scorer']}`",
        f"- model: `{output['runtime']['model']}`",
        f"- closed-book correct/wrong: `{summary['closed_book_correct']}` / `{summary['closed_book_wrong']}`",
        f"- decision: `{summary['decision']}`",
        "",
        "## Overall Metrics",
        "",
        "| Score | AUC | 95% CI | Paired Win Rate |",
        "|---|---:|---:|---:|",
    ]
    for metric in NREV_VARIANTS:
        row = summary["metrics"]["overall"][metric]
        ci = "n/a" if row["ci_low"] is None else f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        auc = "n/a" if row["auc"] is None else f"{row['auc']:.4f}"
        lines.append(f"| `{metric}` | {auc} | {ci} | {row['win_rate']:.4f} |")
    lines.extend(["", "## Closed-Book Stratification", ""])
    for subset in ("closed_book_correct", "closed_book_wrong"):
        lines.extend([f"### {subset}", "", "| Score | AUC | 95% CI | Paired Win Rate | N |", "|---|---:|---:|---:|---:|"])
        for metric in NREV_VARIANTS:
            row = summary["metrics"][subset][metric]
            ci = "n/a" if row["ci_low"] is None else f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            auc = "n/a" if row["auc"] is None else f"{row['auc']:.4f}"
            lines.append(f"| `{metric}` | {auc} | {ci} | {row['win_rate']:.4f} | {row['n']} |")
        lines.append("")
    lines.extend(
        [
            "## Diagnostics",
            "",
            "### Paired Outcomes",
            "",
            "| Score | Gold Wins | Ties | Wrong Wins |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in NREV_VARIANTS:
        row = (diag.get("paired_outcomes") or {}).get(metric, {})
        lines.append(f"| `{metric}` | {row.get('wins', 0)} | {row.get('ties', 0)} | {row.get('losses', 0)} |")
    lines.extend(["", "### Dominant Null Counts", "", "| Side | l_minus | l0 | l_alt | none |", "|---|---:|---:|---:|---:|"])
    for side in ("gold", "wrong"):
        row = (diag.get("dominant_null_counts") or {}).get(side, {})
        lines.append(
            f"| `{side}` | {row.get('l_minus', 0)} | {row.get('l0', 0)} | {row.get('l_alt', 0)} | {row.get('none', 0)} |"
        )
    lines.extend(["", "### Component Means", "", "| Side | l_plus | l_minus | l0 | l_alt |", "|---|---:|---:|---:|---:|"])
    for side in ("gold", "wrong"):
        row = (diag.get("component_means") or {}).get(side, {})
        lines.append(
            f"| `{side}` | {row.get('l_plus')} | {row.get('l_minus')} | {row.get('l0')} | {row.get('l_alt')} |"
        )
    lines.extend(
        [
            "",
            "## Throughput",
            "",
            f"- elapsed seconds: `{output['throughput']['elapsed_seconds']}`",
            f"- seconds/query: `{output['throughput']['seconds_per_query']}`",
            f"- estimated 100-query seconds: `{output['throughput']['estimated_100_query_seconds']}`",
            f"- estimated 1000-query seconds: `{output['throughput']['estimated_1000_query_seconds']}`",
            "",
            "## Notes",
            "",
            "- Day-0 uses only `(a_gold, S_gold)` and `(a_wrong, S_wrong)` pairs.",
            "- `S_gold` is gold support padded to k=5 with top-ranked pool docs.",
            "- `S_wrong` is built from wrong-answer-containing pool docs plus high-ranked distractors when source evidence is unavailable.",
            "- `l_alt` is type-compatible: incompatible alternative answer types are excluded from the null.",
            "- Closed-book stratification uses a separate own-knowledge prompt, not the evidence-only RAG reader prompt.",
            "- Destructive perturbations use singleton functional-cell deletion/replacement in this Day-0 implementation.",
            "",
        ]
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    pool = load_json(args.pool_json)
    caps = load_json(args.caps_day2_json)
    pool_records = list(pool.get("records") or [])
    caps_rows = list(caps.get("rows") or [])
    rows = []
    end = min(len(pool_records), len(caps_rows), int(args.offset) + int(args.limit))
    for idx in range(int(args.offset), end):
        pool_record = pool_records[idx]
        caps_row = caps_rows[idx]
        if str(pool_record.get("question") or "").strip() != str(caps_row.get("question") or "").strip():
            raise ValueError(f"Question mismatch at row {idx}")
        gold_answers = [str(answer) for answer in pool_record.get("gold_answers") or caps_row.get("gold_answers") or []]
        wrong_answer = choose_wrong_answer(caps_row, gold_answers)
        if not wrong_answer or not gold_answers:
            continue
        pool = pool_docs(pool_record)
        gold_set = gold_docs_padded(pool_record, k=int(args.top_k))
        wrong_set = wrong_docs(pool_record, wrong_answer, k=int(args.top_k))
        rows.append(
            {
                "query_idx": idx,
                "qid": str(caps_row.get("qid") or pool_record.get("query_idx") or idx),
                "question": str(pool_record.get("question") or ""),
                "gold_answer": gold_answers[0],
                "wrong_answer": wrong_answer,
                "gold_docs": gold_set,
                "wrong_docs": wrong_set,
                "pool_docs": pool,
                "caps_type": str(caps_row.get("type") or ""),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir = Path(args.output_dir)
    scorer = QwenPromptLogprobScorer(
        base_url=str(args.llm_base_url),
        model=str(args.llm_model),
        cache_path=output_dir / "day0_logprob_cache.jsonl",
        timeout=int(args.timeout),
        concurrency=int(args.concurrency),
    )
    type_cache = JsonlValueCache(output_dir / "day0_type_cache.jsonl")
    rows = build_rows(args)
    result_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for row in rows:
        question = row["question"]
        if str(args.type_mode) == "qwen":
            question_type = qwen_type_classify(
                question=question,
                base_url=str(args.llm_base_url),
                model=str(args.llm_model),
                cache=type_cache,
                timeout=int(args.timeout),
            )
        else:
            question_type = heuristic_question_type(question)
        answers = [row["gold_answer"], row["wrong_answer"]]
        answer_types = {
            answer: answer_surface_type(answer, question)
            for answer in answers
        }
        # If surface typing is ambiguous, fall back to question type for broad compatibility.
        for answer, atype in list(answer_types.items()):
            if atype == "entity" and question_type in {"person", "place", "organization", "work"}:
                answer_types[answer] = question_type
        closed_book_answer = ""
        closed_book_f1 = 0.0
        if not bool(args.no_closed_book_generation):
            try:
                closed_book_answer = call_closed_book_reader(
                    question=question,
                    base_url=str(args.llm_base_url),
                    model=str(args.llm_model),
                    max_new_tokens=int(args.closed_book_max_new_tokens),
                    timeout=int(args.timeout),
                )
                closed_book_f1 = token_f1([row["gold_answer"]], closed_book_answer)
            except Exception:
                closed_book_answer = ""
                closed_book_f1 = 0.0

        gold_components = score_pair_components(
            scorer=scorer,
            qid=row["qid"],
            question=question,
            answer=row["gold_answer"],
            docs=row["gold_docs"],
            pool=row["pool_docs"],
            alt_answers=[row["wrong_answer"]],
            answer_type=answer_types[row["gold_answer"]],
            alt_types=answer_types,
        )
        wrong_components = score_pair_components(
            scorer=scorer,
            qid=row["qid"],
            question=question,
            answer=row["wrong_answer"],
            docs=row["wrong_docs"],
            pool=row["pool_docs"],
            alt_answers=[row["gold_answer"]],
            answer_type=answer_types[row["wrong_answer"]],
            alt_types=answer_types,
        )
        gold_scores = nrev_scores(gold_components)
        wrong_scores = nrev_scores(wrong_components)
        result = {
            "qid": row["qid"],
            "query_idx": row["query_idx"],
            "question": question,
            "question_type": question_type,
            "gold_answer": row["gold_answer"],
            "wrong_answer": row["wrong_answer"],
            "gold_answer_type": answer_types[row["gold_answer"]],
            "wrong_answer_type": answer_types[row["wrong_answer"]],
            "closed_book_answer": closed_book_answer,
            "closed_book_f1": closed_book_f1,
            "closed_book_correct": bool(closed_book_f1 >= 0.5),
            "gold_scores": gold_scores,
            "wrong_scores": wrong_scores,
            "gold_components": {key: gold_components[key] for key in ("l_plus", "l_minus", "l0", "l_alt", "t_plus_count", "t_minus_count", "alt_count")},
            "wrong_components": {key: wrong_components[key] for key in ("l_plus", "l_minus", "l0", "l_alt", "t_plus_count", "t_minus_count", "alt_count")},
            "gold_doc_titles": [doc_title(doc) for doc in row["gold_docs"]],
            "wrong_doc_titles": [doc_title(doc) for doc in row["wrong_docs"]],
        }
        result_rows.append(result)
        for pair_name, components in (("gold", gold_components), ("wrong", wrong_components)):
            for comp in components["score_rows"]:
                component_rows.append(
                    {
                        "qid": row["qid"],
                        "query_idx": row["query_idx"],
                        "pair": pair_name,
                        **comp,
                    }
                )
    elapsed = time.perf_counter() - start
    summary = summarize(result_rows, bootstrap_rounds=int(args.bootstrap_rounds))
    diag = diagnostics(result_rows)
    throughput = {
        "elapsed_seconds": round(float(elapsed), 4),
        "seconds_per_query": round(float(elapsed) / float(max(1, len(result_rows))), 4),
        "estimated_100_query_seconds": round(float(elapsed) / float(max(1, len(result_rows))) * 100.0, 4),
        "estimated_1000_query_seconds": round(float(elapsed) / float(max(1, len(result_rows))) * 1000.0, 4),
    }
    output = {
        "inputs": {
            "pool_json": str(args.pool_json),
            "caps_day2_json": str(args.caps_day2_json),
            "limit": int(args.limit),
            "offset": int(args.offset),
            "top_k": int(args.top_k),
            "type_mode": str(args.type_mode),
        },
        "runtime": {
            "scorer": "qwen_prompt_logprob",
            "base_url": str(args.llm_base_url),
            "model": str(args.llm_model),
            "concurrency": int(args.concurrency),
        },
        "summary": summary,
        "diagnostics": diag,
        "throughput": throughput,
    }
    write_json(output, output_dir / "day0_sanity.json")
    write_jsonl(result_rows, output_dir / "day0_sanity.rows.jsonl")
    write_jsonl(component_rows, output_dir / "day0_sanity.components.jsonl")
    write_markdown(output, output_dir / "day0_sanity.md")
    write_markdown(output, output_dir / "throughput_benchmark.md")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", default="run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json")
    parser.add_argument("--caps_day2_json", default="reports/caps/caps_day2_proof_separability.json")
    parser.add_argument("--output_dir", default="reports/nrev")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--llm_base_url", default="http://localhost:8043/v1")
    parser.add_argument("--llm_model", default="qwen3-8b-train")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--type_mode", choices=["heuristic", "qwen"], default="qwen")
    parser.add_argument("--no_closed_book_generation", action="store_true")
    parser.add_argument("--closed_book_max_new_tokens", type=int, default=96)
    parser.add_argument("--bootstrap_rounds", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    output = run(parse_args())
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
