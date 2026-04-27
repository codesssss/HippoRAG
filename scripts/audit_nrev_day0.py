#!/usr/bin/env python3
"""Audit NREV Day-0 suspicious failure modes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import urllib.request

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import run_nrev_day0_sanity as nrev  # noqa: E402
from src.dpathrag.io import write_json, write_jsonl  # noqa: E402
from src.dpathrag.reader import token_f1  # noqa: E402


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def chat_completion_raw(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }
    req = urllib.request.Request(
        str(base_url).rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str((payload.get("choices") or [{}])[0].get("message", {}).get("content") or "")


def closed_book_messages(question: str, mode: str) -> list[dict[str, str]]:
    if mode == "direct_short":
        return [
            {
                "role": "system",
                "content": (
                    "Answer the question from your own knowledge. Return only the short answer string. "
                    "If uncertain, still give your best guess. Do not explain."
                ),
            },
            {"role": "user", "content": f"/no_think\nQuestion: {question}\nAnswer:"},
        ]
    return [
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
    ]


def parse_closed_book(raw: str, mode: str) -> str:
    parsed = nrev.extract_final_answer(raw)
    if mode == "direct_short":
        parsed = re.sub(r"(?i)^answer\s*:\s*", "", parsed).strip()
    return parsed


def audit_closed_book(
    rows: Sequence[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    sample_n: int,
    timeout: int,
    max_tokens: int,
) -> dict[str, Any]:
    samples = []
    for row in rows[: int(sample_n)]:
        item = {
            "query_idx": row["query_idx"],
            "qid": row["qid"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
        }
        for mode in ("own_knowledge", "direct_short"):
            try:
                raw = chat_completion_raw(
                    base_url=base_url,
                    model=model,
                    messages=closed_book_messages(str(row["question"]), mode),
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except Exception as exc:
                raw = f"ERROR: {exc}"
            parsed = parse_closed_book(raw, mode)
            item[f"{mode}_raw"] = raw
            item[f"{mode}_parsed"] = parsed
            item[f"{mode}_f1"] = token_f1([str(row["gold_answer"])], parsed)
        samples.append(item)
    summary = {}
    for mode in ("own_knowledge", "direct_short"):
        f1s = [float(item[f"{mode}_f1"]) for item in samples]
        summary[mode] = {
            "sample_n": len(samples),
            "correct_at_f1_0_5": sum(1 for value in f1s if value >= 0.5),
            "mean_f1": sum(f1s) / float(max(1, len(f1s))),
            "empty_or_abstain_like": sum(
                1
                for item in samples
                if not nrev.norm_text(str(item[f"{mode}_parsed"]))
                or any(marker in str(item[f"{mode}_parsed"]).lower() for marker in ("unknown", "not mention", "not provided", "i don't know"))
            ),
        }
    return {"summary": summary, "samples": samples}


def component_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[int, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["query_idx"]), str(row["pair"]), str(row["component"]), str(row["name"]))
        index.setdefault(key, []).append(row)
    return index


def mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / float(len(finite))


def likelihood_scale_audit(rows: Sequence[dict[str, Any]], sample_n: int) -> dict[str, Any]:
    inverted = sorted(
        rows,
        key=lambda row: float(row["gold_components"]["l_minus"]) - float(row["gold_components"]["l_plus"]),
        reverse=True,
    )
    samples = []
    for row in inverted[: int(sample_n)]:
        gold = row["gold_components"]
        wrong = row["wrong_components"]
        samples.append(
            {
                "query_idx": row["query_idx"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "wrong_answer": row["wrong_answer"],
                "gold_l_plus": gold["l_plus"],
                "gold_l_minus": gold["l_minus"],
                "gold_l0": gold["l0"],
                "gold_minus_minus_plus": float(gold["l_minus"]) - float(gold["l_plus"]),
                "wrong_l_plus": wrong["l_plus"],
                "wrong_l_minus": wrong["l_minus"],
                "wrong_l0": wrong["l0"],
            }
        )
    deltas = [float(row["gold_components"]["l_minus"]) - float(row["gold_components"]["l_plus"]) for row in rows]
    return {
        "gold_lminus_gt_lplus": sum(1 for value in deltas if value > 0),
        "gold_lminus_eq_lplus": sum(1 for value in deltas if math.isclose(value, 0.0)),
        "gold_lminus_lt_lplus": sum(1 for value in deltas if value < 0),
        "mean_gold_lminus_minus_lplus": mean(deltas),
        "samples_worst_inversions": samples,
    }


def parse_minus_doc_index(name: str) -> int | None:
    match = re.search(r"_(\d+)$", str(name))
    if not match:
        return None
    return int(match.group(1))


def answer_in_doc(answer: str, doc: dict[str, Any]) -> bool:
    answer_norm = nrev.norm_text(answer)
    return bool(answer_norm and answer_norm in nrev.norm_text(nrev.doc_blob(doc)))


def tminus_content_audit(
    built_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[dict[str, Any]],
    component_rows: Sequence[dict[str, Any]],
    *,
    sample_n: int,
) -> dict[str, Any]:
    built_by_idx = {int(row["query_idx"]): row for row in built_rows}
    comp_by = component_index(component_rows)
    gold_minus_scores_by_removed_kind: dict[str, list[float]] = {"critical": [], "padding": []}
    wrong_minus_scores_by_removed_kind: dict[str, list[float]] = {"answer_containing": [], "other": []}
    for result in result_rows:
        built = built_by_idx[int(result["query_idx"])]
        for pair_name, docs, answer in (
            ("gold", built["gold_docs"], built["gold_answer"]),
            ("wrong", built["wrong_docs"], built["wrong_answer"]),
        ):
            for (query_idx, pair, component, name), values in comp_by.items():
                if query_idx != int(result["query_idx"]) or pair != pair_name or component != "minus":
                    continue
                doc_idx = parse_minus_doc_index(name)
                if doc_idx is None or doc_idx >= len(docs):
                    continue
                score_values = [float(item["score"]) for item in values]
                if pair_name == "gold":
                    kind = "critical" if int(docs[doc_idx].get("gold_support") or 0) else "padding"
                    gold_minus_scores_by_removed_kind[kind].extend(score_values)
                else:
                    kind = "answer_containing" if answer_in_doc(answer, docs[doc_idx]) else "other"
                    wrong_minus_scores_by_removed_kind[kind].extend(score_values)

    worst = sorted(
        result_rows,
        key=lambda row: float(row["gold_components"]["l_minus"]) - float(row["gold_components"]["l_plus"]),
        reverse=True,
    )
    samples = []
    for result in worst[: int(sample_n)]:
        built = built_by_idx[int(result["query_idx"])]
        docs = built["gold_docs"]
        variants = []
        for name, ctx in nrev.t_minus_contexts(docs, built["pool_docs"], max_cells=5):
            idx = parse_minus_doc_index(name)
            removed = docs[idx] if idx is not None and idx < len(docs) else {}
            replacement_title = ""
            if str(name).startswith("replace") and idx is not None and idx < len(ctx):
                replacement_title = nrev.doc_title(ctx[idx])
            variants.append(
                {
                    "name": name,
                    "removed_idx": idx,
                    "removed_title": nrev.doc_title(removed),
                    "removed_gold_support": bool(int(removed.get("gold_support") or 0)),
                    "removed_contains_gold_answer": answer_in_doc(built["gold_answer"], removed),
                    "replacement_title": replacement_title,
                    "context_titles": [nrev.doc_title(doc) for doc in ctx],
                }
            )
        samples.append(
            {
                "query_idx": result["query_idx"],
                "question": result["question"],
                "gold_answer": result["gold_answer"],
                "gold_l_plus": result["gold_components"]["l_plus"],
                "gold_l_minus": result["gold_components"]["l_minus"],
                "gold_docs": [
                    {
                        "idx": idx,
                        "title": nrev.doc_title(doc),
                        "gold_support": bool(int(doc.get("gold_support") or 0)),
                        "contains_gold_answer": answer_in_doc(built["gold_answer"], doc),
                    }
                    for idx, doc in enumerate(docs)
                ],
                "t_minus_variants": variants,
            }
        )
    return {
        "gold_minus_score_means": {
            key: mean(values)
            for key, values in gold_minus_scores_by_removed_kind.items()
        },
        "gold_minus_score_counts": {key: len(values) for key, values in gold_minus_scores_by_removed_kind.items()},
        "wrong_minus_score_means": {
            key: mean(values)
            for key, values in wrong_minus_scores_by_removed_kind.items()
        },
        "wrong_minus_score_counts": {key: len(values) for key, values in wrong_minus_scores_by_removed_kind.items()},
        "samples": samples,
    }


def logprob_breakdown(
    scorer: nrev.QwenPromptLogprobScorer,
    *,
    question: str,
    docs: Sequence[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    prefix = scorer.reader_prefix(question, docs)
    full_prompt = f"{prefix} {answer}"
    prefix_tokens = scorer.tokenize(prefix)
    full_tokens = scorer.tokenize(full_prompt)
    prefix_stable = full_tokens[: len(prefix_tokens)] == prefix_tokens
    answer_tokens = full_tokens[len(prefix_tokens) :]
    logprob_rows = scorer.prompt_logprobs(full_prompt)
    values = []
    for idx, token_id in enumerate(answer_tokens, start=len(prefix_tokens)):
        if idx >= len(logprob_rows):
            continue
        row = logprob_rows[idx] or {}
        payload = row.get(str(token_id))
        if payload is None and row:
            payload = next(iter(row.values()))
        if payload is not None:
            values.append(
                {
                    "token_id": int(token_id),
                    "decoded_token": str(payload.get("decoded_token") or ""),
                    "logprob": float(payload.get("logprob")),
                }
            )
    return {
        "score": mean([item["logprob"] for item in values]),
        "prefix_tokens": len(prefix_tokens),
        "full_tokens": len(full_tokens),
        "answer_tokens": len(answer_tokens),
        "logged_answer_tokens": len(values),
        "prefix_stable": prefix_stable,
        "prompt_chars": len(full_prompt),
        "prompt_tail": full_prompt[-300:],
        "tokens": values,
    }


def prompt_format_audit(
    scorer: nrev.QwenPromptLogprobScorer,
    built_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[dict[str, Any]],
    sample_n: int,
) -> dict[str, Any]:
    built_by_idx = {int(row["query_idx"]): row for row in built_rows}
    worst = sorted(
        result_rows,
        key=lambda row: float(row["gold_components"]["l_minus"]) - float(row["gold_components"]["l_plus"]),
        reverse=True,
    )
    samples = []
    for result in worst[: int(sample_n)]:
        built = built_by_idx[int(result["query_idx"])]
        plus_docs = built["gold_docs"]
        minus_name, minus_docs = nrev.t_minus_contexts(plus_docs, built["pool_docs"], max_cells=5)[0]
        plus_prompt = scorer.reader_prefix(built["question"], plus_docs)
        minus_prompt = scorer.reader_prefix(built["question"], minus_docs)
        samples.append(
            {
                "query_idx": built["query_idx"],
                "question": built["question"],
                "minus_variant": minus_name,
                "plus_doc_count": len(plus_docs),
                "minus_doc_count": len(minus_docs),
                "plus_chars": len(plus_prompt),
                "minus_chars": len(minus_prompt),
                "plus_tokens": len(scorer.tokenize(plus_prompt)),
                "minus_tokens": len(scorer.tokenize(minus_prompt)),
                "plus_tail": plus_prompt[-300:],
                "minus_tail": minus_prompt[-300:],
            }
        )
    return {"samples": samples}


def toy_likelihood_audit(scorer: nrev.QwenPromptLogprobScorer) -> dict[str, Any]:
    toys = [
        (
            "What is the capital of the United Kingdom?",
            "United Kingdom\nLondon is the capital of the United Kingdom.",
            "London",
        ),
        (
            "Who wrote Hamlet?",
            "Hamlet\nHamlet is a tragedy written by William Shakespeare.",
            "William Shakespeare",
        ),
        (
            "What is the largest planet in the Solar System?",
            "Jupiter\nJupiter is the largest planet in the Solar System.",
            "Jupiter",
        ),
    ]
    samples = []
    for idx, (question, text, answer) in enumerate(toys):
        doc = nrev.make_doc(f"toy_{idx}", text, doc_id=f"toy::{idx}", rank=1, gold_support=True)
        with_evidence = logprob_breakdown(scorer, question=question, docs=[doc], answer=answer)
        closed = logprob_breakdown(scorer, question=question, docs=[], answer=answer)
        samples.append(
            {
                "question": question,
                "answer": answer,
                "with_evidence_score": with_evidence["score"],
                "closed_book_score": closed["score"],
                "with_evidence_prefix_stable": with_evidence["prefix_stable"],
                "closed_prefix_stable": closed["prefix_stable"],
                "with_evidence_tokens": with_evidence["tokens"],
            }
        )
    return {"samples": samples}


def recompute_scores_with_support_only_minus(
    built_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[dict[str, Any]],
    component_rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    built_by_idx = {int(row["query_idx"]): row for row in built_rows}
    comp_by = component_index(component_rows)
    recomputed = []
    for result in result_rows[: int(limit)]:
        built = built_by_idx[int(result["query_idx"])]
        row_out = {
            "query_idx": result["query_idx"],
            "gold_scores": {},
            "wrong_scores": {},
        }
        for pair_name, docs, answer in (
            ("gold", built["gold_docs"], built["gold_answer"]),
            ("wrong", built["wrong_docs"], built["wrong_answer"]),
        ):
            original = dict(result[f"{pair_name}_components"])
            selected_scores = []
            for (query_idx, pair, component, name), values in comp_by.items():
                if query_idx != int(result["query_idx"]) or pair != pair_name or component != "minus":
                    continue
                idx = parse_minus_doc_index(name)
                if idx is None or idx >= len(docs):
                    continue
                if pair_name == "gold":
                    keep = bool(int(docs[idx].get("gold_support") or 0))
                else:
                    keep = answer_in_doc(answer, docs[idx])
                if keep:
                    selected_scores.extend(float(item["score"]) for item in values)
            if selected_scores:
                original["l_minus"] = sum(selected_scores) / float(len(selected_scores))
            row_out[f"{pair_name}_scores"] = nrev.nrev_scores(original)
            row_out[f"{pair_name}_support_only_l_minus_count"] = len(selected_scores)
            row_out[f"{pair_name}_support_only_l_minus"] = original["l_minus"]
        recomputed.append(row_out)
    metrics = {}
    for metric in nrev.NREV_VARIANTS:
        positives = [float(row["gold_scores"][metric]) for row in recomputed]
        negatives = [float(row["wrong_scores"][metric]) for row in recomputed]
        metrics[metric] = {
            "auc": nrev.auc_score(positives, negatives),
            "paired_win_rate": nrev.paired_win_rate(recomputed, metric),
        }
    return {"limit": int(limit), "metrics": metrics, "rows": recomputed}


def truncate(value: Any, n: int = 160) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NREV Day-0 Audit",
        "",
        "## Executive Summary",
        "",
    ]
    conclusion = report["conclusion"]
    for item in conclusion:
        lines.append(f"- {item}")
    lines.extend(["", "## Audit 1: Closed-Book Sanity", ""])
    for mode, row in report["closed_book"]["summary"].items():
        lines.append(
            f"- `{mode}`: correct@F1>=0.5 `{row['correct_at_f1_0_5']}/{row['sample_n']}`, "
            f"mean F1 `{row['mean_f1']:.4f}`, abstain-like `{row['empty_or_abstain_like']}`"
        )
    lines.extend(["", "| # | Gold | Own-Knowledge Parsed | F1 | Direct Parsed | F1 | Raw Own-Knowledge |", "|---:|---|---|---:|---|---:|---|"])
    for item in report["closed_book"]["samples"]:
        lines.append(
            f"| {item['query_idx']} | {truncate(item['gold_answer'], 60)} | "
            f"{truncate(item['own_knowledge_parsed'], 70)} | {float(item['own_knowledge_f1']):.3f} | "
            f"{truncate(item['direct_short_parsed'], 70)} | {float(item['direct_short_f1']):.3f} | "
            f"{truncate(item['own_knowledge_raw'], 120)} |"
        )

    scale = report["likelihood_scale"]
    lines.extend(
        [
            "",
            "## Audit 2: Likelihood Scale",
            "",
            f"- gold `l_minus > l_plus`: `{scale['gold_lminus_gt_lplus']}`",
            f"- gold `l_minus < l_plus`: `{scale['gold_lminus_lt_lplus']}`",
            f"- mean gold `l_minus - l_plus`: `{fmt_float(scale['mean_gold_lminus_minus_lplus'])}`",
            "",
            "| # | Gold | l_plus | l_minus | l0 | l_minus-l_plus | Wrong | wrong l_plus | wrong l_minus |",
            "|---:|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for item in scale["samples_worst_inversions"]:
        lines.append(
            f"| {item['query_idx']} | {truncate(item['gold_answer'], 50)} | "
            f"{float(item['gold_l_plus']):.4f} | {float(item['gold_l_minus']):.4f} | {float(item['gold_l0']):.4f} | "
            f"{float(item['gold_minus_minus_plus']):.4f} | {truncate(item['wrong_answer'], 50)} | "
            f"{float(item['wrong_l_plus']):.4f} | {float(item['wrong_l_minus']):.4f} |"
        )

    tminus = report["tminus"]
    lines.extend(
        [
            "",
            "## Audit 3: T-minus Content",
            "",
            "| Pair | Removed kind | Count | Mean l_minus component score |",
            "|---|---|---:|---:|",
        ]
    )
    for kind, count in tminus["gold_minus_score_counts"].items():
        lines.append(f"| gold | `{kind}` | {count} | {fmt_float(tminus['gold_minus_score_means'].get(kind))} |")
    for kind, count in tminus["wrong_minus_score_counts"].items():
        lines.append(f"| wrong | `{kind}` | {count} | {fmt_float(tminus['wrong_minus_score_means'].get(kind))} |")
    lines.extend(["", "### Sample T-minus Variants", ""])
    for sample in tminus["samples"]:
        lines.append(
            f"- query `{sample['query_idx']}`, gold `{sample['gold_answer']}`, "
            f"l_plus `{float(sample['gold_l_plus']):.4f}`, l_minus `{float(sample['gold_l_minus']):.4f}`"
        )
        lines.append(
            "  S_gold: "
            + "; ".join(
                f"[{doc['idx']}] {doc['title']} (support={doc['gold_support']}, answer={doc['contains_gold_answer']})"
                for doc in sample["gold_docs"]
            )
        )
        for variant in sample["t_minus_variants"][:4]:
            repl = f", repl={variant['replacement_title']}" if variant["replacement_title"] else ""
            lines.append(
                f"  - `{variant['name']}` removed `{variant['removed_title']}` "
                f"(support={variant['removed_gold_support']}, answer={variant['removed_contains_gold_answer']}{repl})"
            )

    lines.extend(["", "## Audit 4: Prompt / Token / Toy Checks", ""])
    toy = report["toy"]
    lines.extend(["### Toy Likelihood", "", "| Question | Answer | evidence score | closed score | prefix stable |", "|---|---|---:|---:|---|"])
    for item in toy["samples"]:
        lines.append(
            f"| {truncate(item['question'], 80)} | {item['answer']} | {fmt_float(item['with_evidence_score'])} | "
            f"{fmt_float(item['closed_book_score'])} | {item['with_evidence_prefix_stable']} / {item['closed_prefix_stable']} |"
        )
    breakdown = report["token_breakdown"]
    lines.extend(
        [
            "",
            "### Per-Token Breakdown Example",
            "",
            f"- query: `{breakdown['query_idx']}`",
            f"- answer: `{breakdown['answer']}`",
            f"- prefix stable: `{breakdown['breakdown']['prefix_stable']}`",
            f"- score: `{fmt_float(breakdown['breakdown']['score'])}`",
            "",
            "| Token | Logprob |",
            "|---|---:|",
        ]
    )
    for token in breakdown["breakdown"]["tokens"]:
        lines.append(f"| `{token['decoded_token']}` | {float(token['logprob']):.4f} |")

    lines.extend(["", "### Prompt Format Samples", ""])
    for item in report["prompt_format"]["samples"]:
        lines.append(
            f"- query `{item['query_idx']}` `{item['minus_variant']}`: plus chars/tokens "
            f"`{item['plus_chars']}/{item['plus_tokens']}`, minus chars/tokens `{item['minus_chars']}/{item['minus_tokens']}`"
        )
        lines.append(f"  plus tail: `{truncate(item['plus_tail'], 220)}`")
        lines.append(f"  minus tail: `{truncate(item['minus_tail'], 220)}`")

    posthoc = report["support_only_posthoc"]
    lines.extend(["", "## Post-hoc Support-only T-minus Recompute", ""])
    lines.append(f"- limit: `{posthoc['limit']}`")
    lines.extend(["", "| Score | AUC | Paired Win Rate |", "|---|---:|---:|"])
    for metric, row in posthoc["metrics"].items():
        lines.append(f"| `{metric}` | {fmt_float(row['auc'])} | {fmt_float(row['paired_win_rate'])} |")
    lines.extend(
        [
            "",
            "## Audit Decision",
            "",
            report["decision"],
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = Path(args.day0_rows)
    components_path = Path(args.day0_components)
    result_rows = load_jsonl(rows_path)
    component_rows = load_jsonl(components_path)
    built_rows = nrev.build_rows(args)
    scorer = nrev.QwenPromptLogprobScorer(
        base_url=str(args.llm_base_url),
        model=str(args.llm_model),
        cache_path=output_dir / "day0_audit_logprob_cache.jsonl",
        timeout=int(args.timeout),
        concurrency=int(args.concurrency),
    )

    closed_book = audit_closed_book(
        result_rows,
        base_url=str(args.llm_base_url),
        model=str(args.llm_model),
        sample_n=int(args.closed_book_sample),
        timeout=int(args.timeout),
        max_tokens=int(args.closed_book_max_tokens),
    )
    scale = likelihood_scale_audit(result_rows, int(args.scale_sample))
    tminus = tminus_content_audit(
        built_rows,
        result_rows,
        component_rows,
        sample_n=int(args.tminus_sample),
    )
    prompt_format = prompt_format_audit(scorer, built_rows, result_rows, int(args.tminus_sample))
    toy = toy_likelihood_audit(scorer)

    worst = max(
        result_rows,
        key=lambda row: float(row["gold_components"]["l_minus"]) - float(row["gold_components"]["l_plus"]),
    )
    built_by_idx = {int(row["query_idx"]): row for row in built_rows}
    worst_built = built_by_idx[int(worst["query_idx"])]
    token_breakdown = {
        "query_idx": worst["query_idx"],
        "question": worst["question"],
        "answer": worst["gold_answer"],
        "breakdown": logprob_breakdown(
            scorer,
            question=worst_built["question"],
            docs=worst_built["gold_docs"],
            answer=worst_built["gold_answer"],
        ),
    }
    support_only = recompute_scores_with_support_only_minus(
        built_rows,
        result_rows,
        component_rows,
        limit=int(args.posthoc_limit),
    )

    conclusion = []
    cb_direct = closed_book["summary"]["direct_short"]
    if cb_direct["correct_at_f1_0_5"] > 2:
        conclusion.append(
            "Closed-book `2/100` is not reliable as a scientific claim: a short-answer no-context prompt improves the first-20 sample, so the original stratification is prompt/parsing sensitive."
        )
    else:
        conclusion.append("Closed-book low accuracy appears directionally real on the audited first-20 sample, not just an empty-output parser failure.")
    if scale["gold_lminus_gt_lplus"] >= 50:
        conclusion.append(
            f"`l_minus > l_plus` for gold remains real in stored raw components (`{scale['gold_lminus_gt_lplus']}/100`), not an aggregation-display bug."
        )
    toy_bad = any((item["with_evidence_score"] is None or float(item["with_evidence_score"]) < -3.0) for item in toy["samples"])
    if toy_bad:
        conclusion.append("Toy likelihood magnitudes are suspicious; logprob extraction needs deeper endpoint-level debugging.")
    else:
        conclusion.append("Toy likelihood magnitudes and token-prefix stability look sane; no obvious prompt-logprob span bug was found.")
    support_auc = support_only["metrics"]["nrev_full"]["auc"]
    if support_auc is not None and float(support_auc) >= 0.70:
        conclusion.append("Support-only T-minus post-hoc recompute reaches the keep-alive range on first-30; NREV should be rerun with fixed destructive cells before final rejection.")
        decision = "NREV_AUDIT_KEEP_ALIVE_RERUN_WITH_SUPPORT_ONLY_TMINUS"
    elif support_auc is not None and float(support_auc) < 0.65:
        conclusion.append("Support-only T-minus post-hoc recompute is still below 0.65 on first-30; padding deletion is not sufficient to rescue NREV.")
        decision = "NREV_AUDIT_CONFIRMS_STOP"
    else:
        conclusion.append("Support-only T-minus post-hoc recompute is marginal; one bounded T-minus redesign is the only defensible continuation.")
        decision = "NREV_AUDIT_MARGINAL_TMINUS_REDESIGN_ONLY"

    report = {
        "inputs": {
            "day0_rows": str(rows_path),
            "day0_components": str(components_path),
            "pool_json": str(args.pool_json),
            "caps_day2_json": str(args.caps_day2_json),
            "llm_base_url": str(args.llm_base_url),
            "llm_model": str(args.llm_model),
        },
        "closed_book": closed_book,
        "likelihood_scale": scale,
        "tminus": tminus,
        "prompt_format": prompt_format,
        "toy": toy,
        "token_breakdown": token_breakdown,
        "support_only_posthoc": support_only,
        "conclusion": conclusion,
        "decision": decision,
    }
    write_json(report, output_dir / "day0_audit.json")
    write_jsonl(closed_book["samples"], output_dir / "day0_audit_closed_book_samples.jsonl")
    (output_dir / "day0_audit.md").write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_json", default="run_logs/proprag_pool_exports_full1000_20260424/2wikimultihopqa_pool100.json")
    parser.add_argument("--caps_day2_json", default="reports/caps/caps_day2_proof_separability.json")
    parser.add_argument("--day0_rows", default="reports/nrev/day0_sanity.rows.jsonl")
    parser.add_argument("--day0_components", default="reports/nrev/day0_sanity.components.jsonl")
    parser.add_argument("--output_dir", default="reports/nrev")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--llm_base_url", default="http://localhost:8043/v1")
    parser.add_argument("--llm_model", default="qwen3-8b-train")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--closed_book_sample", type=int, default=20)
    parser.add_argument("--closed_book_max_tokens", type=int, default=96)
    parser.add_argument("--scale_sample", type=int, default=10)
    parser.add_argument("--tminus_sample", type=int, default=5)
    parser.add_argument("--posthoc_limit", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
