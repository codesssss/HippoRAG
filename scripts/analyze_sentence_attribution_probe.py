#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.hipporag.utils.dataset_utils import resolve_dataset_paths
from src.hipporag.utils.misc_utils import compute_mdhash_id


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _tokenize(text: str | None) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(_normalize_text(text))]


def _token_set(text: str | None) -> set[str]:
    return set(_tokenize(text))


def _split_doc_text(doc_text: str) -> tuple[str, str]:
    return (str(doc_text).split("\n", 1) + [""])[:2]


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(str(text or ""))]
    sentences = [part for part in parts if part]
    return sentences or ([_normalize_text(text)] if _normalize_text(text) else [])


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    if total <= 0:
        return [0.0 for _ in exp_values]
    return [value / total for value in exp_values]


def _build_chunk_id_to_doc_text(corpus: list[dict[str, Any]]) -> dict[str, str]:
    chunk_id_to_doc_text: dict[str, str] = {}
    for row in corpus:
        doc_text = f"{row['title']}\n{row['text']}"
        chunk_id = compute_mdhash_id(doc_text, prefix="chunk-")
        chunk_id_to_doc_text[chunk_id] = doc_text
    return chunk_id_to_doc_text


def _resolve_doc_text(
    *,
    doc_id: Any,
    corpus: list[dict[str, Any]],
    chunk_id_to_doc_text: dict[str, str],
) -> str | None:
    if isinstance(doc_id, int):
        if 0 <= int(doc_id) < len(corpus):
            row = corpus[int(doc_id)]
            return f"{row['title']}\n{row['text']}"
        return None
    if isinstance(doc_id, str):
        return chunk_id_to_doc_text.get(doc_id)
    return None


def _case_filter_matches(case_mode: str, delta_em: float, delta_f1: float) -> bool:
    if case_mode == "all_applied":
        return True
    if case_mode == "applied_positive":
        return delta_em > 0.0 or delta_f1 > 0.0
    if case_mode == "applied_nonpositive":
        return delta_em <= 0.0 and delta_f1 <= 0.0
    raise ValueError(f"Unsupported case_mode: {case_mode}")


def extract_probe_cases(
    *,
    repair_payload: dict[str, Any],
    corpus: list[dict[str, Any]],
    case_mode: str,
    max_cases: int,
) -> list[dict[str, Any]]:
    chunk_id_to_doc_text = _build_chunk_id_to_doc_text(corpus)
    cases: list[dict[str, Any]] = []

    for trace in list(repair_payload.get("expand_assemble_query_traces") or []):
        question = str(trace.get("question", "")).strip()
        if not question:
            continue
        expand_trace = dict(trace.get("expand_assemble_trace") or {})
        assemble_trace = dict(expand_trace.get("assemble_trace") or {})
        if not bool(assemble_trace.get("repair_applied", False)):
            continue

        baseline_metrics = dict(trace.get("baseline_metrics") or {})
        method_metrics = dict(trace.get("method_metrics") or {})
        delta_em = float(method_metrics.get("ExactMatch", 0.0) or 0.0) - float(baseline_metrics.get("ExactMatch", 0.0) or 0.0)
        delta_f1 = float(method_metrics.get("F1", 0.0) or 0.0) - float(baseline_metrics.get("F1", 0.0) or 0.0)
        if not _case_filter_matches(case_mode, delta_em, delta_f1):
            continue

        best_swap = dict(assemble_trace.get("best_repair_swap") or {})
        candidate_position = best_swap.get("appended_pool_position")
        replaced_position = best_swap.get("replace_pool_position")
        if candidate_position is None or replaced_position is None:
            continue

        ranking_by_position = {
            int(row.get("pool_position")): dict(row)
            for row in list(assemble_trace.get("ranking_rows") or [])
            if row.get("pool_position") is not None
        }
        candidate_row = ranking_by_position.get(int(candidate_position), {})
        replaced_row = ranking_by_position.get(int(replaced_position), {})

        candidate_doc_text = _resolve_doc_text(
            doc_id=candidate_row.get("doc_id"),
            corpus=corpus,
            chunk_id_to_doc_text=chunk_id_to_doc_text,
        )
        replaced_doc_text = _resolve_doc_text(
            doc_id=replaced_row.get("doc_id"),
            corpus=corpus,
            chunk_id_to_doc_text=chunk_id_to_doc_text,
        )
        if candidate_doc_text is None or replaced_doc_text is None:
            continue

        cases.append({
            "question": question,
            "query_type": str(trace.get("query_type", "")),
            "gold_titles": list(trace.get("gold_titles") or []),
            "gold_answers": list(trace.get("gold_answers") or []),
            "question_entities_preview": list(expand_trace.get("query_entities_preview") or []),
            "candidate_title": str(best_swap.get("candidate_title", "")),
            "replace_title": str(best_swap.get("replace_title", "")),
            "candidate_doc_text": candidate_doc_text,
            "replaced_doc_text": replaced_doc_text,
            "delta_em": delta_em,
            "delta_f1": delta_f1,
            "connector_gain_count": int(best_swap.get("connector_gain_count", 0) or 0),
            "anchor_gain_count": int(best_swap.get("anchor_gain_count", 0) or 0),
            "candidate_ce_rank": int(best_swap.get("candidate_ce_rank", 0) or 0),
            "replaced_ce_rank": int(best_swap.get("replaced_incumbent_ce_rank", 0) or 0),
            "candidate_ce_score": float(candidate_row.get("assemble_score", best_swap.get("candidate_ce_score", 0.0)) or 0.0),
            "replaced_ce_score": float(replaced_row.get("assemble_score", 0.0) or 0.0),
            "ce_score_gap_candidate_minus_replaced": float(candidate_row.get("assemble_score", best_swap.get("candidate_ce_score", 0.0)) or 0.0)
            - float(replaced_row.get("assemble_score", 0.0) or 0.0),
            "scaffold_titles": list(assemble_trace.get("scaffold_titles") or []),
            "final_titles_after_repair": list(trace.get("method_top_titles") or []),
        })
        if max_cases > 0 and len(cases) >= max_cases:
            break

    return cases


def _build_sentence_rows(
    *,
    question: str,
    doc_text: str,
    reranker: Any,
    gold_titles: Iterable[str],
    gold_answers: Iterable[str],
    query_entities_preview: Iterable[str],
) -> dict[str, Any]:
    title, body = _split_doc_text(doc_text)
    sentences = split_sentences(body)
    if not sentences:
        sentences = [title]

    pairs = [[question, sentence] for sentence in sentences]
    raw_scores = reranker.compute_score(pairs)
    if isinstance(raw_scores, (int, float)):
        scores = [float(raw_scores)]
    else:
        scores = [float(score) for score in raw_scores]
    masses = _softmax(scores)

    gold_title_token_sets = [_token_set(title_text) for title_text in gold_titles if _token_set(title_text)]
    gold_answer_token_sets = [_token_set(answer_text) for answer_text in gold_answers if _token_set(answer_text)]
    query_token_set = _token_set(question)
    query_entity_token_set = set()
    for value in query_entities_preview:
        query_entity_token_set |= _token_set(str(value))

    rows: list[dict[str, Any]] = []
    for idx, (sentence, score, mass) in enumerate(zip(sentences, scores, masses), start=1):
        sentence_tokens = _token_set(sentence)
        question_overlap = len(sentence_tokens & query_token_set)
        query_entity_overlap = len(sentence_tokens & query_entity_token_set)
        gold_title_overlap = max((len(sentence_tokens & token_set) for token_set in gold_title_token_sets), default=0)
        gold_answer_overlap = max((len(sentence_tokens & token_set) for token_set in gold_answer_token_sets), default=0)
        rows.append({
            "sentence_rank": idx,
            "sentence_text": sentence,
            "score": float(score),
            "softmax_mass": float(mass),
            "token_count": len(_tokenize(sentence)),
            "question_overlap_tokens": int(question_overlap),
            "query_entity_overlap_tokens": int(query_entity_overlap),
            "gold_title_overlap_tokens": int(gold_title_overlap),
            "gold_answer_overlap_tokens": int(gold_answer_overlap),
            "has_goldish_signal": bool(gold_title_overlap > 0 or gold_answer_overlap > 0),
        })

    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -int(row["gold_title_overlap_tokens"]),
            -int(row["gold_answer_overlap_tokens"]),
            -int(row["query_entity_overlap_tokens"]),
        )
    )
    total_doc_tokens = max(1, len(_tokenize(body)))
    top2 = rows[:2]
    top3 = rows[:3]
    any_goldish = any(bool(row["has_goldish_signal"]) for row in rows)
    top2_goldish = any(bool(row["has_goldish_signal"]) for row in top2)
    goldish_outside_top2 = any(bool(row["has_goldish_signal"]) for row in rows[2:])

    return {
        "title": title,
        "sentence_count": len(sentences),
        "doc_token_count": total_doc_tokens,
        "top1_softmax_mass": _round(top3[0]["softmax_mass"] if top3 else None),
        "top2_softmax_mass": _round(sum(float(row["softmax_mass"]) for row in top2)),
        "top2_token_share": _round(sum(int(row["token_count"]) for row in top2) / float(total_doc_tokens)),
        "any_goldish_sentence": bool(any_goldish),
        "top2_has_goldish_sentence": bool(top2_goldish),
        "goldish_only_outside_top2": bool(any_goldish and not top2_goldish and goldish_outside_top2),
        "top_sentences": [
            {
                **row,
                "score": _round(row["score"]),
                "softmax_mass": _round(row["softmax_mass"]),
            }
            for row in top3
        ],
    }


def analyze_cases(
    *,
    cases: list[dict[str, Any]],
    ce_model: str,
    ce_device: str,
) -> dict[str, Any]:
    from FlagEmbedding import FlagReranker

    reranker = FlagReranker(
        ce_model,
        use_fp16=True,
        devices=[str(ce_device)],
    )

    analyzed_cases: list[dict[str, Any]] = []
    candidate_top2_mass: list[float] = []
    candidate_top2_token_share: list[float] = []
    candidate_top2_goldish_count = 0
    candidate_any_goldish_count = 0
    candidate_goldish_outside_top2_count = 0
    candidate_localized_count = 0

    for case in cases:
        question = str(case["question"])
        candidate_summary = _build_sentence_rows(
            question=question,
            doc_text=str(case["candidate_doc_text"]),
            reranker=reranker,
            gold_titles=list(case.get("gold_titles") or []),
            gold_answers=list(case.get("gold_answers") or []),
            query_entities_preview=list(case.get("question_entities_preview") or []),
        )
        replaced_summary = _build_sentence_rows(
            question=question,
            doc_text=str(case["replaced_doc_text"]),
            reranker=reranker,
            gold_titles=list(case.get("gold_titles") or []),
            gold_answers=list(case.get("gold_answers") or []),
            query_entities_preview=list(case.get("question_entities_preview") or []),
        )

        candidate_top2_mass.append(float(candidate_summary["top2_softmax_mass"] or 0.0))
        candidate_top2_token_share.append(float(candidate_summary["top2_token_share"] or 0.0))
        candidate_top2_goldish_count += int(bool(candidate_summary["top2_has_goldish_sentence"]))
        candidate_any_goldish_count += int(bool(candidate_summary["any_goldish_sentence"]))
        candidate_goldish_outside_top2_count += int(bool(candidate_summary["goldish_only_outside_top2"]))
        candidate_localized_count += int(float(candidate_summary["top2_softmax_mass"] or 0.0) >= 0.6)

        analyzed_cases.append({
            **case,
            "delta_em": _round(case["delta_em"]),
            "delta_f1": _round(case["delta_f1"]),
            "candidate_ce_score": _round(case["candidate_ce_score"]),
            "replaced_ce_score": _round(case["replaced_ce_score"]),
            "ce_score_gap_candidate_minus_replaced": _round(case["ce_score_gap_candidate_minus_replaced"]),
            "candidate_summary": candidate_summary,
            "replaced_summary": replaced_summary,
        })

    summary = {
        "num_cases": len(analyzed_cases),
        "candidate_top2_softmax_mass_mean": _round(statistics.mean(candidate_top2_mass)) if candidate_top2_mass else None,
        "candidate_top2_softmax_mass_median": _round(statistics.median(candidate_top2_mass)) if candidate_top2_mass else None,
        "candidate_top2_token_share_mean": _round(statistics.mean(candidate_top2_token_share)) if candidate_top2_token_share else None,
        "candidate_localized_rate_top2_mass_ge_0_6": _round(100.0 * candidate_localized_count / len(analyzed_cases), 2) if analyzed_cases else None,
        "candidate_any_goldish_sentence_rate": _round(100.0 * candidate_any_goldish_count / len(analyzed_cases), 2) if analyzed_cases else None,
        "candidate_top2_goldish_sentence_rate": _round(100.0 * candidate_top2_goldish_count / len(analyzed_cases), 2) if analyzed_cases else None,
        "candidate_goldish_only_outside_top2_rate": _round(100.0 * candidate_goldish_outside_top2_count / len(analyzed_cases), 2) if analyzed_cases else None,
    }
    return {
        "summary": summary,
        "cases": analyzed_cases,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    meta = dict(payload.get("metadata") or {})
    summary = dict(payload.get("summary") or {})
    lines = [
        f"# Sentence Attribution Probe ({meta.get('dataset', 'unknown')})",
        "",
        f"- repair report: `{meta.get('repair_report', '')}`",
        f"- case mode: `{meta.get('case_mode', '')}`",
        f"- cases analyzed: `{summary.get('num_cases', 0)}`",
        f"- CE model: `{meta.get('ce_model', '')}` on `{meta.get('ce_device', '')}`",
        "",
        "## Aggregate",
        "",
        f"- candidate top2 softmax mass mean / median: `{summary.get('candidate_top2_softmax_mass_mean', '—')}` / `{summary.get('candidate_top2_softmax_mass_median', '—')}`",
        f"- candidate top2 token share mean: `{summary.get('candidate_top2_token_share_mean', '—')}`",
        f"- candidate localized rate (top2 mass >= 0.6): `{summary.get('candidate_localized_rate_top2_mass_ge_0_6', '—')}`",
        f"- candidate any goldish sentence rate: `{summary.get('candidate_any_goldish_sentence_rate', '—')}`",
        f"- candidate top2 goldish sentence rate: `{summary.get('candidate_top2_goldish_sentence_rate', '—')}`",
        f"- candidate goldish only outside top2 rate: `{summary.get('candidate_goldish_only_outside_top2_rate', '—')}`",
        "",
        "## Cases",
        "",
    ]
    for case in list(payload.get("cases") or []):
        candidate_summary = dict(case.get("candidate_summary") or {})
        replaced_summary = dict(case.get("replaced_summary") or {})
        lines.extend([
            f"### {case.get('candidate_title', '')} -> replace {case.get('replace_title', '')}",
            "",
            f"- question: `{case.get('question', '')}`",
            f"- delta EM / F1: `{case.get('delta_em', '—')}` / `{case.get('delta_f1', '—')}`",
            f"- CE gap (candidate - replaced): `{case.get('ce_score_gap_candidate_minus_replaced', '—')}`",
            f"- connector / anchor gain: `{case.get('connector_gain_count', 0)}` / `{case.get('anchor_gain_count', 0)}`",
            f"- candidate top2 mass / token share: `{candidate_summary.get('top2_softmax_mass', '—')}` / `{candidate_summary.get('top2_token_share', '—')}`",
            f"- candidate goldish any/top2/outside-top2-only: `{candidate_summary.get('any_goldish_sentence', False)}` / `{candidate_summary.get('top2_has_goldish_sentence', False)}` / `{candidate_summary.get('goldish_only_outside_top2', False)}`",
            f"- replaced top2 mass / token share: `{replaced_summary.get('top2_softmax_mass', '—')}` / `{replaced_summary.get('top2_token_share', '—')}`",
            "",
            "- candidate top sentences:",
        ])
        for row in list(candidate_summary.get("top_sentences") or []):
            lines.append(
                "  "
                f"[score={row.get('score', '—')}, mass={row.get('softmax_mass', '—')}, "
                f"q={row.get('question_overlap_tokens', 0)}, qe={row.get('query_entity_overlap_tokens', 0)}, "
                f"gt={row.get('gold_title_overlap_tokens', 0)}, ga={row.get('gold_answer_overlap_tokens', 0)}] "
                f"{row.get('sentence_text', '')}"
            )
        lines.append("- replaced top sentences:")
        for row in list(replaced_summary.get("top_sentences") or []):
            lines.append(
                "  "
                f"[score={row.get('score', '—')}, mass={row.get('softmax_mass', '—')}, "
                f"q={row.get('question_overlap_tokens', 0)}, qe={row.get('query_entity_overlap_tokens', 0)}, "
                f"gt={row.get('gold_title_overlap_tokens', 0)}, ga={row.get('gold_answer_overlap_tokens', 0)}] "
                f"{row.get('sentence_text', '')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight sentence-level attribution probe for ce_local_repair cases.")
    parser.add_argument("--repair_report", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--case_mode", choices=["all_applied", "applied_positive", "applied_nonpositive"], required=True)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--ce_model", default="/mnt/nvme/bge-reranker-v2-m3")
    parser.add_argument("--ce_device", default="cuda:7")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    repair_report_path = Path(args.repair_report)
    repair_payload = _load_json(repair_report_path)
    dataset = str(args.dataset or repair_payload.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("Unable to infer dataset; pass --dataset explicitly.")

    corpus_path, _ = resolve_dataset_paths(dataset, ROOT_DIR / "reproduce" / "dataset")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    cases = extract_probe_cases(
        repair_payload=repair_payload,
        corpus=corpus,
        case_mode=str(args.case_mode),
        max_cases=int(args.max_cases or 0),
    )
    analysis = analyze_cases(
        cases=cases,
        ce_model=str(args.ce_model),
        ce_device=str(args.ce_device),
    )

    payload = {
        "metadata": {
            "dataset": dataset,
            "repair_report": str(repair_report_path),
            "case_mode": str(args.case_mode),
            "ce_model": str(args.ce_model),
            "ce_device": str(args.ce_device),
        },
        **analysis,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(output_md),
        "num_cases": int(payload["summary"]["num_cases"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
