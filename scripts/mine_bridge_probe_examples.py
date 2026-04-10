import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_requirement_cache import load_dataset
from eval_causal_qwen3 import get_gold_docs
from requirement_beam_utils import (
    extract_need_unit_atomic_features,
    get_positive_units,
    is_need_unit_cache_version,
    load_requirement_cache,
    normalize_structure_text,
)


def _split_doc_text(doc_text: str) -> tuple[str, str]:
    return (str(doc_text).split("\n", 1) + [""])[:2]


def _build_sample_id(question_key: str, pool_position: int, unit_id: str, cf_id: str = "") -> str:
    prefix = f"{question_key}:p{int(pool_position)}:{unit_id}"
    return f"{prefix}:{cf_id}" if cf_id else prefix


def _load_report_trace_map(report_path: str) -> dict[str, dict[str, Any]]:
    resolved = str(report_path or "").strip()
    if not resolved:
        return {}
    payload = json.loads(Path(resolved).read_text())
    traces = payload.get("setwise_selector_query_traces", [])
    mapping: dict[str, dict[str, Any]] = {}
    for trace in traces:
        question = str(trace.get("question", "")).strip()
        if question:
            mapping[question] = dict(trace)
    return mapping


def _build_corpus_title_map(corpus: list[dict]) -> dict[str, str]:
    title_to_text: dict[str, str] = {}
    for doc in corpus:
        title = str(doc["title"]).strip()
        if title and title not in title_to_text:
            title_to_text[title] = str(doc.get("text", ""))
    return title_to_text


def _safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_unit_score(score_map: dict[str, Any], unit_id: str) -> dict[str, Any]:
    resolved = score_map.get(unit_id, {})
    return dict(resolved) if isinstance(resolved, dict) else {}


def _bridge_signal_summary(feature_row: dict[str, float | int]) -> dict[str, float]:
    return {
        "title_bridge_alignment": _safe_float(feature_row.get("title_bridge_alignment", 0.0)),
        "entity_bridge_alignment": _safe_float(feature_row.get("entity_bridge_alignment", 0.0)),
        "alias_or_variable_bridge_alignment": _safe_float(feature_row.get("alias_or_variable_bridge_alignment", 0.0)),
        "bridge_feature_count": _safe_float(feature_row.get("bridge_feature_count", 0.0)),
        "subject_alignment": _safe_float(feature_row.get("subject_alignment", 0.0)),
        "predicate_alignment": _safe_float(feature_row.get("predicate_alignment", 0.0)),
        "object_alignment": _safe_float(feature_row.get("object_alignment", 0.0)),
        "constraint_alignment": _safe_float(feature_row.get("constraint_alignment", 0.0)),
    }


def _bucket_for_candidate(*,
                          unit_type: str,
                          is_gold_doc: bool,
                          changed_from_baseline: bool,
                          in_baseline_top_titles: bool,
                          in_selector_top_titles: bool,
                          bridge_feature_count: float,
                          alignment_score: float,
                          subject_alignment: float,
                          predicate_alignment: float,
                          object_alignment: float,
                          support_prob: float,
                          full_support_prob: float,
                          bridge_support_prob: float,
                          contradiction_prob: float) -> str | None:
    if unit_type not in {"relation_hop", "constraint_check"}:
        if is_gold_doc and full_support_prob >= 0.9:
            return "full_support_control"
        return None

    bridge_like = bridge_feature_count >= 1.0 and alignment_score >= 0.25
    lost_after_hybrid = changed_from_baseline and in_baseline_top_titles and not in_selector_top_titles
    gold_middle_hop_pattern = (
        is_gold_doc
        and (
            (subject_alignment >= 0.75 and object_alignment >= 0.5)
            or (subject_alignment >= 1.0 and predicate_alignment >= 0.4)
            or bridge_support_prob >= 0.5
        )
    )

    if (
        is_gold_doc
        and (bridge_like or gold_middle_hop_pattern or lost_after_hybrid)
        and support_prob <= 0.8
        and full_support_prob < 0.85
        and contradiction_prob < 0.2
        and (lost_after_hybrid or in_baseline_top_titles or gold_middle_hop_pattern)
    ):
        return "bridge_positive"

    if (
        not is_gold_doc
        and bridge_like
        and support_prob <= 0.2
        and full_support_prob <= 0.2
        and contradiction_prob <= 0.2
    ):
        return "bridge_like_hard_negative"

    if (
        not is_gold_doc
        and bridge_feature_count <= 0.0
        and support_prob <= 0.05
        and contradiction_prob <= 0.1
        and alignment_score <= 0.2
    ):
        return "clean_nei"

    if is_gold_doc and full_support_prob >= 0.9:
        return "full_support_control"

    return None


def _category_priority(bucket: str) -> int:
    order = {
        "bridge_positive": 0,
        "bridge_like_hard_negative": 1,
        "clean_nei": 2,
        "full_support_control": 3,
    }
    return order.get(bucket, 99)


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        -_safe_float(row.get("candidate_priority_score", 0.0)),
        -_safe_float(row.get("bridge_feature_count", 0.0)),
        -_safe_float(row.get("alignment_score", 0.0)),
        str(row.get("sample_id", "")),
    )


def _seed_label_for_candidate(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    bucket = str(row.get("candidate_bucket", ""))
    if bucket == "bridge_positive":
        strong_bridge_continuity = (
            _safe_float(row.get("alignment_score", 0.0)) >= 0.35
            and _safe_float(row.get("subject_alignment", 0.0)) >= 0.75
            and _safe_float(row.get("object_alignment", 0.0)) >= 0.5
        )
        if (
            bool(row.get("is_gold_doc"))
            and _safe_float(row.get("full_support_prob", 0.0)) < 0.85
            and (
                _safe_float(row.get("bridge_support_prob", 0.0)) >= 0.5
                or strong_bridge_continuity
            )
        ):
            return "bridge_support", "high", "gold doc with explicit bridge continuity"
        if (
            bool(row.get("is_gold_doc"))
            and _safe_float(row.get("full_support_prob", 0.0)) < 0.85
            and (
                _safe_float(row.get("bridge_feature_count", 0.0)) >= 1.0
                or _safe_float(row.get("subject_alignment", 0.0)) >= 0.75
                or bool(row.get("changed_from_baseline"))
            )
        ):
            return "bridge_support", "medium", "gold doc with possible bridge continuation pattern"
        return None, None, None

    if bucket == "bridge_like_hard_negative":
        if (
            not bool(row.get("is_gold_doc"))
            and _safe_float(row.get("support_prob", 0.0)) <= 0.1
            and _safe_float(row.get("full_support_prob", 0.0)) <= 0.1
            and _safe_float(row.get("bridge_feature_count", 0.0)) >= 1.0
        ):
            return "nei", "high", "bridge-like overlap without gold support or chain closure"
        return None, None, None

    if bucket == "clean_nei":
        if (
            not bool(row.get("is_gold_doc"))
            and _safe_float(row.get("bridge_feature_count", 0.0)) <= 0.0
            and _safe_float(row.get("support_prob", 0.0)) <= 0.05
        ):
            return "nei", "high", "clean near-negative with no bridge signal"
        return None, None, None

    if bucket == "full_support_control":
        if bool(row.get("is_gold_doc")) and _safe_float(row.get("full_support_prob", 0.0)) >= 0.9:
            return "full_support", "high", "gold control with strong direct support signal"
        return None, None, None

    return None, None, None


def _candidate_priority_score(*,
                              bucket: str,
                              is_gold_doc: bool,
                              changed_from_baseline: bool,
                              in_baseline_top_titles: bool,
                              in_selector_top_titles: bool,
                              bridge_feature_count: float,
                              alignment_score: float,
                              subject_alignment: float,
                              object_alignment: float,
                              support_prob: float,
                              full_support_prob: float,
                              bridge_support_prob: float) -> float:
    score = 0.0
    if bucket == "bridge_positive":
        score += 3.0 * float(is_gold_doc)
        score += 2.0 * float(changed_from_baseline)
        score += 1.5 * float(in_baseline_top_titles)
        score += 1.0 * float(not in_selector_top_titles)
        score += min(2.0, bridge_feature_count)
        score += alignment_score
        score += 0.5 * subject_alignment
        score += 0.5 * object_alignment
        score += bridge_support_prob
        score += max(0.0, 0.7 - support_prob)
    elif bucket == "bridge_like_hard_negative":
        score += min(2.0, bridge_feature_count)
        score += alignment_score
        score += max(0.0, 0.3 - full_support_prob)
        score += float(changed_from_baseline)
    elif bucket == "clean_nei":
        score += max(0.0, 0.1 - support_prob)
        score += max(0.0, 0.2 - alignment_score)
    elif bucket == "full_support_control":
        score += 2.0 * float(is_gold_doc)
        score += full_support_prob
    return round(score, 6)


def _take_bucket_rows(candidates: list[dict[str, Any]],
                      max_per_bucket: int,
                      max_per_query_per_bucket: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_query_counter: Counter[tuple[str, int]] = Counter()
    for row in sorted(candidates, key=_candidate_sort_key):
        query_index = int(row.get("query_index", -1))
        bucket = str(row.get("candidate_bucket", ""))
        query_bucket_key = (bucket, query_index)
        if per_query_counter[query_bucket_key] >= max_per_query_per_bucket:
            continue
        selected.append(row)
        per_query_counter[query_bucket_key] += 1
        if len(selected) >= max_per_bucket:
            break
    return selected


def build_bridge_probe_rows(cache_payload: dict[str, Any],
                            corpus: list[dict],
                            gold_docs: list[list[str]],
                            trace_map: dict[str, dict[str, Any]],
                            max_per_bucket: int,
                            max_per_query_per_bucket: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    title_to_body = _build_corpus_title_map(corpus)
    rows_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_counter: Counter[str] = Counter()

    for entry in cache_payload.get("queries", []):
        query_index = int(entry.get("query_index", 0))
        question = str(entry.get("question", ""))
        question_key = str(entry.get("question_key", f"q{query_index}"))
        trace = trace_map.get(question, {})
        baseline_top_titles = set(trace.get("baseline_top_titles", []) or [])
        selector_top_titles = set(trace.get("selector_top_titles", []) or [])
        changed_from_baseline = bool(trace.get("changed_from_baseline", False))
        gold_titles = {
            str(doc_text).split("\n", 1)[0].strip()
            for doc_text in gold_docs[query_index]
        } if query_index < len(gold_docs) else set()

        units_by_id = {
            str(unit.get("unit_id", unit.get("requirement_id", ""))): dict(unit)
            for unit in get_positive_units(entry)
            if bool(unit.get("selector_enabled", True))
        }

        for ann in entry.get("doc_annotations", []):
            pool_position = int(ann.get("pool_position", -1))
            doc_title = str(ann.get("doc_title", "")).strip()
            doc_body = title_to_body.get(doc_title, "")
            doc_entities = list(ann.get("doc_entities", []) or [])
            positive_atomic_scores = dict(ann.get("positive_atomic_scores", {}) or {})

            in_baseline = doc_title in baseline_top_titles
            in_selector = doc_title in selector_top_titles
            is_gold_doc = doc_title in gold_titles

            for unit_id, unit in units_by_id.items():
                score_payload = _extract_unit_score(positive_atomic_scores, unit_id)
                if not score_payload:
                    continue
                feature_row = extract_need_unit_atomic_features(
                    unit=unit,
                    doc_title=doc_title,
                    doc_body=doc_body,
                    doc_entities=doc_entities,
                )
                bridge_summary = _bridge_signal_summary(feature_row)
                support_prob = _safe_float(score_payload.get("support_prob", 0.0))
                full_support_prob = _safe_float(score_payload.get("full_support_prob", 0.0))
                bridge_support_prob = _safe_float(score_payload.get("bridge_support_prob", 0.0))
                contradiction_prob = _safe_float(score_payload.get("contradiction_prob", 0.0))
                nei_prob = _safe_float(score_payload.get("nei_prob", 0.0))
                alignment_score = _safe_float(score_payload.get("alignment_score", 0.0))
                coverage_score = _safe_float(score_payload.get("coverage_score", 0.0))

                bucket = _bucket_for_candidate(
                    unit_type=str(unit.get("unit_type", unit.get("type", ""))),
                    is_gold_doc=is_gold_doc,
                    changed_from_baseline=changed_from_baseline,
                    in_baseline_top_titles=in_baseline,
                    in_selector_top_titles=in_selector,
                    bridge_feature_count=bridge_summary["bridge_feature_count"],
                    alignment_score=alignment_score,
                    subject_alignment=bridge_summary["subject_alignment"],
                    predicate_alignment=bridge_summary["predicate_alignment"],
                    object_alignment=bridge_summary["object_alignment"],
                    support_prob=support_prob,
                    full_support_prob=full_support_prob,
                    bridge_support_prob=bridge_support_prob,
                    contradiction_prob=contradiction_prob,
                )
                if not bucket:
                    continue

                priority_score = _candidate_priority_score(
                    bucket=bucket,
                    is_gold_doc=is_gold_doc,
                    changed_from_baseline=changed_from_baseline,
                    in_baseline_top_titles=in_baseline,
                    in_selector_top_titles=in_selector,
                    bridge_feature_count=bridge_summary["bridge_feature_count"],
                    alignment_score=alignment_score,
                    subject_alignment=bridge_summary["subject_alignment"],
                    object_alignment=bridge_summary["object_alignment"],
                    support_prob=support_prob,
                    full_support_prob=full_support_prob,
                    bridge_support_prob=bridge_support_prob,
                )
                sample_id = _build_sample_id(
                    question_key=question_key,
                    pool_position=pool_position,
                    unit_id=unit_id,
                )
                label, label_confidence, label_reason = _seed_label_for_candidate({
                    "candidate_bucket": bucket,
                    "is_gold_doc": is_gold_doc,
                    "changed_from_baseline": changed_from_baseline,
                    "in_baseline_top_titles": in_baseline,
                    "in_selector_top_titles": in_selector,
                    "bridge_feature_count": bridge_summary["bridge_feature_count"],
                    "support_prob": support_prob,
                    "full_support_prob": full_support_prob,
                    "bridge_support_prob": bridge_support_prob,
                    "alignment_score": alignment_score,
                    "subject_alignment": bridge_summary["subject_alignment"],
                    "predicate_alignment": bridge_summary["predicate_alignment"],
                    "object_alignment": bridge_summary["object_alignment"],
                })
                row = {
                    "sample_id": sample_id,
                    "question_id": question_key,
                    "query_index": query_index,
                    "question": question,
                    "candidate_bucket": bucket,
                    "candidate_bucket_priority": _category_priority(bucket),
                    "candidate_priority_score": priority_score,
                    "suggested_label": label,
                    "label_confidence": label_confidence,
                    "label_reason": label_reason,
                    "changed_from_baseline": changed_from_baseline,
                    "in_baseline_top_titles": in_baseline,
                    "in_selector_top_titles": in_selector,
                    "is_gold_doc": is_gold_doc,
                    "gold_doc_title_match": is_gold_doc,
                    "unit_id": unit_id,
                    "unit_type": str(unit.get("unit_type", unit.get("type", ""))),
                    "unit_subject": unit.get("subject"),
                    "unit_predicate": unit.get("predicate"),
                    "unit_object": unit.get("object"),
                    "unit_text": unit.get("text", unit.get("normalized_text", "")),
                    "doc_title": doc_title,
                    "doc_body_preview": doc_body[:320],
                    "pool_position": pool_position,
                    "support_prob": support_prob,
                    "full_support_prob": full_support_prob,
                    "bridge_support_prob": bridge_support_prob,
                    "contradiction_prob": contradiction_prob,
                    "nei_prob": nei_prob,
                    "alignment_score": alignment_score,
                    "coverage_score": coverage_score,
                    **bridge_summary,
                }
                rows_by_bucket[bucket].append(row)
                bucket_counter[bucket] += 1

    selected_rows: list[dict[str, Any]] = []
    for bucket in ("bridge_positive", "bridge_like_hard_negative", "clean_nei", "full_support_control"):
        selected_rows.extend(
            _take_bucket_rows(
                rows_by_bucket.get(bucket, []),
                max_per_bucket=max_per_bucket,
                max_per_query_per_bucket=max_per_query_per_bucket,
            )
        )

    selected_rows.sort(key=lambda row: (row["candidate_bucket_priority"],) + _candidate_sort_key(row))
    summary = {
        "candidate_bucket_counts_before_cap": dict(bucket_counter),
        "selected_count": len(selected_rows),
        "selected_bucket_counts": dict(Counter(str(row["candidate_bucket"]) for row in selected_rows)),
        "seed_label_counts": dict(Counter(str(row["suggested_label"]) for row in selected_rows if row.get("suggested_label"))),
    }
    return selected_rows, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_seed_labels_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    seed_rows = []
    for row in rows:
        label = str(row.get("suggested_label", "") or "").strip()
        confidence = str(row.get("label_confidence", "") or "").strip()
        if not label or confidence != "high":
            continue
        seed_rows.append({
            "sample_id": str(row["sample_id"]),
            "label": label,
            "support_subtype": label,
            "label_source": "assistant_seed_bridge_spec_v1",
            "label_confidence": confidence,
            "reason": str(row.get("label_reason", "")),
        })
    _write_jsonl(path, seed_rows)
    return len(seed_rows)


def _write_markdown_preview(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Bridge Probe Preview",
        "",
        f"- selected rows: `{summary.get('selected_count', 0)}`",
        f"- selected bucket counts: `{json.dumps(summary.get('selected_bucket_counts', {}), ensure_ascii=False)}`",
        f"- seed label counts: `{json.dumps(summary.get('seed_label_counts', {}), ensure_ascii=False)}`",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("candidate_bucket", ""))].append(row)
    for bucket in ("bridge_positive", "bridge_like_hard_negative", "clean_nei", "full_support_control"):
        bucket_rows = grouped.get(bucket, [])
        lines.append(f"## {bucket}")
        lines.append("")
        for row in bucket_rows[:10]:
            lines.append(f"- question: `{row['question']}`")
            lines.append(f"  unit: `{row['unit_predicate']}` | doc: `{row['doc_title']}`")
            lines.append(
                "  scores: "
                f"support={row['support_prob']:.4f}, "
                f"full={row['full_support_prob']:.4f}, "
                f"bridge={row['bridge_support_prob']:.4f}, "
                f"align={row['alignment_score']:.4f}, "
                f"bridge_features={row['bridge_feature_count']:.1f}"
            )
            if row.get("suggested_label"):
                lines.append(
                    f"  suggested label: `{row['suggested_label']}` ({row.get('label_confidence', '')})"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine bridge-first probe examples from a need-unit cache and optional smoke report.")
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_md", type=str, default="")
    parser.add_argument("--seed_labels_path", type=str, default="")
    parser.add_argument("--max_per_bucket", type=int, default=20)
    parser.add_argument("--max_per_query_per_bucket", type=int, default=3)
    args = parser.parse_args()

    cache_payload = load_requirement_cache(args.cache_path)
    if not is_need_unit_cache_version(cache_payload):
        raise ValueError(f"Expected a V2 need-unit cache, got {cache_payload.get('version')}")

    corpus, samples = load_dataset(args.dataset, args.limit)
    gold_docs = get_gold_docs(samples, args.dataset, corpus=corpus)
    trace_map = _load_report_trace_map(args.report_path)

    rows, summary = build_bridge_probe_rows(
        cache_payload=cache_payload,
        corpus=corpus,
        gold_docs=gold_docs,
        trace_map=trace_map,
        max_per_bucket=int(args.max_per_bucket),
        max_per_query_per_bucket=int(args.max_per_query_per_bucket),
    )
    _write_jsonl(Path(args.output_path), rows)

    output_md = Path(args.output_md) if str(args.output_md or "").strip() else Path(args.output_path).with_suffix(".md")
    _write_markdown_preview(output_md, rows, summary)

    seed_label_count = 0
    if str(args.seed_labels_path or "").strip():
        seed_label_count = _write_seed_labels_jsonl(Path(args.seed_labels_path), rows)

    console_summary = dict(summary)
    console_summary["cache_path"] = str(args.cache_path)
    console_summary["output_path"] = str(args.output_path)
    console_summary["output_md"] = str(output_md)
    if str(args.seed_labels_path or "").strip():
        console_summary["seed_labels_path"] = str(args.seed_labels_path)
        console_summary["seed_label_row_count"] = int(seed_label_count)
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
