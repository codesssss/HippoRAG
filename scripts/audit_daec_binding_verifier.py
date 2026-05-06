"""Audit a conservative structural binding verifier for DAEC traces.

This is a post-hoc probe. It does not call the reader and does not change the
DAEC selector. The probe simulates a conservative policy:

1. Keep the base DAEC binding unless multiple bindings are objective-tied.
2. In tied cases, keep the base binding if it has cross-document text support.
3. Flip only when the base binding lacks support and a tied alternative has it.

The structural support signal is intentionally simple and train-free: a bound
entity must be mentioned in a non-entity support document, either the upstream
document used by LLM binding extraction or another document selected for that
binding. This avoids using phi as a self-verifying binding score.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(text: Any) -> str:
    raw = unicodedata.normalize("NFKD", str(text or ""))
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = raw.lower()
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def title_variants(title: Any) -> list[str]:
    text = str(title or "").strip()
    variants = [text]
    no_paren = re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()
    if no_paren and no_paren != text:
        variants.append(no_paren)
    if "," in text:
        variants.append(text.split(",", 1)[0].strip())
    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def contains_title(text: Any, title: Any) -> bool:
    haystack = normalize_text(text)
    if not haystack:
        return False
    padded = f" {haystack} "
    for key in title_variants(title):
        if key and f" {key} " in padded:
            return True
    return False


def _records_from_pool(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        rows = payload.get("records") or payload.get("rows") or payload.get("data") or []
    else:
        rows = payload
    return [row for row in rows if isinstance(row, Mapping)]


def _trace_binding_by_id(trace: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for binding in trace.get("bindings") or []:
        if not isinstance(binding, Mapping):
            continue
        binding_id = str(binding.get("binding_id") or "")
        if binding_id:
            out[binding_id] = binding
    return out


def _objective_by_id(trace: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in trace.get("binding_objectives") or []:
        if not isinstance(row, Mapping):
            continue
        binding_id = str(row.get("binding_id") or "")
        if binding_id:
            out[binding_id] = row
    return out


def _entity_positions_from_grounding(trace: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in (trace.get("binding_grounding") or {}).get("scores") or []:
        if not isinstance(row, Mapping):
            continue
        binding_id = str(row.get("binding_id") or "")
        for assignment in row.get("assignments") or []:
            if not isinstance(assignment, Mapping):
                continue
            req_id = str(assignment.get("requirement_id") or "")
            try:
                pos = int(assignment.get("title_pool_position", -1))
            except (TypeError, ValueError):
                pos = -1
            if binding_id and req_id and pos >= 0:
                out[(binding_id, req_id)] = pos
    return out


def _matched_extraction_rows(trace: Mapping[str, Any],
                             req_id: str,
                             title: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    title_keys = set(title_variants(title))
    for extraction in trace.get("llm_binding_extractions") or []:
        if not isinstance(extraction, Mapping):
            continue
        if str(extraction.get("requirement_id") or "") != str(req_id):
            continue
        for entity in extraction.get("matched_entities") or []:
            if not isinstance(entity, Mapping):
                continue
            matched_title = str(entity.get("title") or "")
            if title_keys.intersection(title_variants(matched_title)):
                rows.append(extraction)
                break
    return rows


def _doc_text(pool_record: Mapping[str, Any], pos: int) -> str:
    docs = pool_record.get("pool_docs") or []
    if 0 <= int(pos) < len(docs):
        return str(docs[int(pos)] or "")
    return ""


def _doc_title(pool_record: Mapping[str, Any], pos: int) -> str:
    titles = pool_record.get("pool_titles") or []
    if 0 <= int(pos) < len(titles):
        return str(titles[int(pos)] or "")
    return ""


def _selected_positions(row: Mapping[str, Any]) -> list[int]:
    out: list[int] = []
    for value in row.get("selected_positions") or []:
        try:
            pos = int(value)
        except (TypeError, ValueError):
            continue
        if pos not in out:
            out.append(pos)
    return out


def structural_assignment_support(
    *,
    trace: Mapping[str, Any],
    pool_record: Mapping[str, Any],
    binding_id: str,
    req_id: str,
    title: str,
    binding_objective: Mapping[str, Any],
    entity_pos: int,
    support_mode: str,
) -> dict[str, Any]:
    extraction_hits: list[dict[str, Any]] = []
    companion_hits: list[dict[str, Any]] = []
    selected_position_set = set(_selected_positions(binding_objective))

    for extraction in _matched_extraction_rows(trace, req_id, title):
        try:
            dep_pos = int(extraction.get("dep_position", -1))
        except (TypeError, ValueError):
            dep_pos = -1
        if dep_pos < 0 or dep_pos == int(entity_pos):
            continue
        if support_mode == "selected_companion":
            continue
        if support_mode == "selected_extraction_or_companion" and dep_pos not in selected_position_set:
            continue
        text = _doc_text(pool_record, dep_pos)
        if contains_title(text, title):
            extraction_hits.append({
                "mode": "cross_extraction_mention",
                "position": dep_pos,
                "title": _doc_title(pool_record, dep_pos),
            })

    for pos in _selected_positions(binding_objective):
        if int(pos) == int(entity_pos):
            continue
        text = _doc_text(pool_record, pos)
        if contains_title(text, title):
            companion_hits.append({
                "mode": "selected_companion_mention",
                "position": int(pos),
                "title": _doc_title(pool_record, pos),
            })

    supported = bool(extraction_hits or companion_hits)
    return {
        "requirement_id": str(req_id),
        "title": str(title),
        "entity_position": int(entity_pos),
        "supported": supported,
        "hits": extraction_hits + companion_hits,
    }


def structural_binding_support(
    *,
    trace: Mapping[str, Any],
    pool_record: Mapping[str, Any],
    binding_id: str,
    binding: Mapping[str, Any],
    binding_objective: Mapping[str, Any],
    entity_positions: Mapping[tuple[str, str], int],
    support_mode: str,
) -> dict[str, Any]:
    assignments = binding.get("assignments") or {}
    if not isinstance(assignments, Mapping) or not assignments:
        return {"binding_id": binding_id, "supported": True, "assignments": []}

    assignment_rows: list[dict[str, Any]] = []
    for req_id, title in sorted(assignments.items()):
        pos = int(entity_positions.get((binding_id, str(req_id)), -1))
        row = structural_assignment_support(
            trace=trace,
            pool_record=pool_record,
            binding_id=binding_id,
            req_id=str(req_id),
            title=str(title),
            binding_objective=binding_objective,
            entity_pos=pos,
            support_mode=str(support_mode),
        )
        assignment_rows.append(row)

    return {
        "binding_id": binding_id,
        "supported": bool(assignment_rows) and all(bool(row.get("supported")) for row in assignment_rows),
        "assignments": assignment_rows,
    }


def normalized_title_set(titles: Sequence[Any]) -> set[str]:
    out: set[str] = set()
    for title in titles:
        variants = title_variants(title)
        if variants:
            out.add(variants[0])
    return out


def support_recall(gold_titles: Sequence[Any], selected_titles: Sequence[Any]) -> tuple[float, bool]:
    gold = normalized_title_set(gold_titles)
    if not gold:
        return 0.0, False
    selected = normalized_title_set(selected_titles)
    hits = len(gold.intersection(selected))
    return hits / len(gold), hits == len(gold)


def selected_titles_for_objective(pool_record: Mapping[str, Any],
                                  binding_objective: Mapping[str, Any]) -> list[str]:
    return [_doc_title(pool_record, pos) for pos in _selected_positions(binding_objective)]


def _final_positions_from_trace(trace: Mapping[str, Any], top_k: int) -> list[int]:
    positions: list[int] = []
    for key in ("final_front_pool_positions", "selected_pool_positions", "selected_positions"):
        for value in trace.get(key) or []:
            try:
                pos = int(value)
            except (TypeError, ValueError):
                continue
            if pos not in positions:
                positions.append(pos)
            if len(positions) >= int(top_k):
                return positions[: int(top_k)]
    return positions[: int(top_k)]


def verifier_titles_for_decision(
    *,
    pool_record: Mapping[str, Any],
    trace: Mapping[str, Any],
    verifier_binding_id: str,
    selected_binding_id: str,
    verifier_objective: Mapping[str, Any],
    base_titles: Sequence[Any],
) -> list[str]:
    if str(verifier_binding_id) == str(selected_binding_id):
        return [str(title) for title in base_titles]

    top_k = max(1, len(base_titles))
    positions: list[int] = []
    for pos in _selected_positions(verifier_objective):
        if pos not in positions:
            positions.append(pos)
    for pos in _final_positions_from_trace(trace, top_k):
        if pos not in positions:
            positions.append(pos)
        if len(positions) >= top_k:
            break
    return [_doc_title(pool_record, pos) for pos in positions[:top_k]]


def audit_query(
    *,
    dataset: str,
    query_index: int,
    query_trace: Mapping[str, Any],
    pool_record: Mapping[str, Any],
    objective_epsilon: float,
    support_mode: str,
) -> dict[str, Any]:
    trace = query_trace.get("selector_trace") or {}
    binding_by_id = _trace_binding_by_id(trace)
    objective_by_id = _objective_by_id(trace)
    entity_positions = _entity_positions_from_grounding(trace)
    selected_binding_id = str(trace.get("selected_binding_id") or "")
    selected_obj = objective_by_id.get(selected_binding_id) or {}
    selected_objective = float(selected_obj.get("objective", 0.0) or 0.0)

    tied_ids = [
        binding_id for binding_id, row in objective_by_id.items()
        if float(row.get("objective", 0.0) or 0.0) >= selected_objective - float(objective_epsilon)
    ]
    support_by_id: dict[str, dict[str, Any]] = {}
    for binding_id in tied_ids:
        binding = binding_by_id.get(binding_id) or {}
        obj = objective_by_id.get(binding_id) or {}
        support_by_id[binding_id] = structural_binding_support(
            trace=trace,
            pool_record=pool_record,
            binding_id=binding_id,
            binding=binding,
            binding_objective=obj,
            entity_positions=entity_positions,
            support_mode=str(support_mode),
        )

    base_supported = bool((support_by_id.get(selected_binding_id) or {}).get("supported"))
    decision = "keep_not_tied"
    verifier_binding_id = selected_binding_id
    if len(tied_ids) > 1:
        if base_supported:
            decision = "keep_base_supported"
        else:
            supported_alts = [
                binding_id for binding_id in tied_ids
                if binding_id != selected_binding_id and bool((support_by_id.get(binding_id) or {}).get("supported"))
            ]
            if supported_alts:
                verifier_binding_id = supported_alts[0]
                decision = "flip_alt_supported"
            else:
                decision = "keep_no_supported_alt"

    base_titles = list(query_trace.get("selector_top_titles") or [])
    verifier_obj = objective_by_id.get(verifier_binding_id) or selected_obj
    verifier_titles = verifier_titles_for_decision(
        pool_record=pool_record,
        trace=trace,
        verifier_binding_id=verifier_binding_id,
        selected_binding_id=selected_binding_id,
        verifier_objective=verifier_obj,
        base_titles=base_titles,
    )
    gold_titles = list(query_trace.get("gold_titles") or pool_record.get("gold_titles") or [])
    base_recall, base_complete = support_recall(gold_titles, base_titles)
    verifier_recall, verifier_complete = support_recall(gold_titles, verifier_titles)

    return {
        "dataset": str(dataset),
        "query_index": int(query_index),
        "question": str(query_trace.get("question") or pool_record.get("question") or ""),
        "gold_titles": gold_titles,
        "base_binding_id": selected_binding_id,
        "verifier_binding_id": verifier_binding_id,
        "decision": decision,
        "tied_binding_ids": tied_ids,
        "base_supported": base_supported,
        "support_by_binding": support_by_id,
        "base_titles": base_titles,
        "verifier_titles": verifier_titles,
        "base_support_recall": base_recall,
        "verifier_support_recall": verifier_recall,
        "base_support_complete": base_complete,
        "verifier_support_complete": verifier_complete,
        "base_em": float((query_trace.get("selector_metrics") or {}).get("ExactMatch", 0.0) or 0.0),
        "base_f1": float((query_trace.get("selector_metrics") or {}).get("F1", 0.0) or 0.0),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    flips = [row for row in rows if row.get("decision") == "flip_alt_supported"]
    tied = [row for row in rows if len(row.get("tied_binding_ids") or []) > 1]
    base_supported = [row for row in rows if row.get("base_supported")]
    support_1_to_0 = [
        row for row in rows
        if bool(row.get("base_support_complete")) and not bool(row.get("verifier_support_complete"))
    ]
    support_0_to_1 = [
        row for row in rows
        if not bool(row.get("base_support_complete")) and bool(row.get("verifier_support_complete"))
    ]
    recall_delta = sum(float(row.get("verifier_support_recall", 0.0)) - float(row.get("base_support_recall", 0.0)) for row in rows)
    return {
        "rows": total,
        "tied_rows": len(tied),
        "tie_rate": len(tied) / total if total else 0.0,
        "base_supported_rows": len(base_supported),
        "base_supported_rate": len(base_supported) / total if total else 0.0,
        "flips": len(flips),
        "flip_rate": len(flips) / total if total else 0.0,
        "support_complete_1_to_0": len(support_1_to_0),
        "support_complete_0_to_1": len(support_0_to_1),
        "avg_support_recall_delta": recall_delta / total if total else 0.0,
    }


def build_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# DAEC Saturation-Aware Binding Verifier Probe",
        "",
        "This is a post-hoc audit only. It does not rerun the reader and does not change selector outputs.",
        "",
        f"- Objective epsilon: `{payload['objective_epsilon']}`",
        f"- Support mode: `{payload['support_mode']}`",
        f"- Policy: `{payload['policy']}`",
        "",
        "```text",
        "+----------+------+-------+---------+---------+----------+----------+----------+",
        "| Dataset  | Rows | Tied  | BaseSup | Flips   | SC 1->0  | SC 0->1  | dRecall  |",
        "+----------+------+-------+---------+---------+----------+----------+----------+",
    ]
    for dataset, summary in payload.get("by_dataset", {}).items():
        lines.append(
            "| {dataset:<8} | {rows:>4d} | {tied:>5d} | {base_sup:>7d} | {flips:>7d} | {s10:>8d} | {s01:>8d} | {dr:>+8.4f} |".format(
                dataset=str(dataset),
                rows=int(summary.get("rows", 0)),
                tied=int(summary.get("tied_rows", 0)),
                base_sup=int(summary.get("base_supported_rows", 0)),
                flips=int(summary.get("flips", 0)),
                s10=int(summary.get("support_complete_1_to_0", 0)),
                s01=int(summary.get("support_complete_0_to_1", 0)),
                dr=float(summary.get("avg_support_recall_delta", 0.0)),
            )
        )
    lines.extend([
        "+----------+------+-------+---------+---------+----------+----------+----------+",
        "```",
        "",
        "Decision counts:",
        "",
        "```text",
        "+----------+----------------------+-------+",
        "| Dataset  | Decision             | Count |",
        "+----------+----------------------+-------+",
    ])
    for dataset, counts in payload.get("decision_counts", {}).items():
        for decision, count in sorted(counts.items()):
            lines.append(f"| {dataset:<8} | {decision:<20} | {int(count):>5d} |")
    lines.extend([
        "+----------+----------------------+-------+",
        "```",
        "",
        "Flip examples:",
        "",
    ])
    examples = payload.get("flip_examples") or []
    if not examples:
        lines.append("No flips under the conservative policy.")
    else:
        lines.append("```text")
        lines.append("+----------+-------+----------------------+----------------------+----------+")
        lines.append("| Dataset  | Query | Base                 | Verifier             | dRecall  |")
        lines.append("+----------+-------+----------------------+----------------------+----------+")
        for row in examples[:20]:
            delta = float(row.get("verifier_support_recall", 0.0)) - float(row.get("base_support_recall", 0.0))
            lines.append(
                "| {dataset:<8} | {qid:>5d} | {base:<20.20} | {ver:<20.20} | {delta:>+8.4f} |".format(
                    dataset=str(row.get("dataset", "")),
                    qid=int(row.get("query_index", -1)),
                    base=str(row.get("base_binding_id", "")),
                    ver=str(row.get("verifier_binding_id", "")),
                    delta=delta,
                )
            )
        lines.append("+----------+-------+----------------------+----------------------+----------+")
        lines.append("```")
    return "\n".join(lines) + "\n"


def audit_dataset(dataset: str,
                  *,
                  eval_json: Path,
                  pool_json: Path,
                  objective_epsilon: float,
                  support_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_payload = _read_json(eval_json)
    pool_records = _records_from_pool(_read_json(pool_json))
    traces = list(eval_payload.get("setwise_selector_query_traces") or [])
    rows: list[dict[str, Any]] = []
    for idx, query_trace in enumerate(traces):
        if idx >= len(pool_records):
            break
        rows.append(audit_query(
            dataset=dataset,
            query_index=idx,
            query_trace=query_trace,
            pool_record=pool_records[idx],
            objective_epsilon=objective_epsilon,
            support_mode=str(support_mode),
        ))
    return rows, summarize_rows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="run_logs/daec_binding_grounded_limit100_20260506")
    parser.add_argument("--pool_dir", default="run_logs/hipporag_pool_exports_full1000_20260503")
    parser.add_argument("--output_dir", default="reports/daec_binding_verifier_20260506")
    parser.add_argument("--objective_epsilon", type=float, default=1e-9)
    parser.add_argument("--support_mode", default="extraction_or_companion",
                        choices=["extraction_or_companion", "selected_extraction_or_companion", "selected_companion"])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pool_dir = Path(args.pool_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    by_dataset: dict[str, Any] = {}
    decision_counts: dict[str, dict[str, int]] = {}
    for dataset in args.datasets:
        eval_json = run_dir / "evals" / f"{dataset}_base_qwen8b_hipporag_pool100_limit100.json"
        pool_json = pool_dir / f"{dataset}_hipporag_pool100.json"
        rows, summary = audit_dataset(
            str(dataset),
            eval_json=eval_json,
            pool_json=pool_json,
            objective_epsilon=float(args.objective_epsilon),
            support_mode=str(args.support_mode),
        )
        all_rows.extend(rows)
        label = "2Wiki" if dataset == "2wikimultihopqa" else ("HotpotQA" if dataset == "hotpotqa" else "MuSiQue")
        by_dataset[label] = summary
        counts: dict[str, int] = {}
        for row in rows:
            decision = str(row.get("decision") or "")
            counts[decision] = counts.get(decision, 0) + 1
        decision_counts[label] = counts

    payload = {
        "objective_epsilon": float(args.objective_epsilon),
        "support_mode": str(args.support_mode),
        "policy": "keep base unless objective-tied, base unsupported, and a tied alternative has structural support",
        "run_dir": str(run_dir),
        "pool_dir": str(pool_dir),
        "by_dataset": by_dataset,
        "decision_counts": decision_counts,
        "overall": summarize_rows(all_rows),
        "flip_examples": [row for row in all_rows if row.get("decision") == "flip_alt_supported"][:50],
    }
    _write_json(payload, output_dir / "binding_verifier_probe_summary.json")
    _write_jsonl(all_rows, output_dir / "binding_verifier_probe_rows.jsonl")
    (output_dir / "binding_verifier_probe.md").write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["by_dataset"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
