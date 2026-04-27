#!/usr/bin/env python3
"""Post-hoc diagnostics for Qwen slot generation.

The online slot-quality runner reports a strict variable-grounding score based
on exact output-variable names.  That is useful for schema compliance, but it
is too harsh for Qwen generations that use semantic variable names
(`director`, `birthplace`) instead of oracle placeholders (`x1`, `answer`).

This script recomputes non-LLM, position-aware diagnostics from cached
predictions so Week-0 decisions do not depend on fragile variable names.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.io import load_dataset, read_jsonl, write_json
from src.bsgs.metrics import token_f1
from src.bsgs.slots import Slot, build_oracle_slots_for_sample, parse_predicted_slots


def align_slots(predicted: list[Slot], gold: list[Slot], threshold: float) -> list[tuple[int, int, float]]:
    """Greedy text-F1 alignment, returning `(pred_idx, gold_idx, score)`."""
    matched_gold: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for pred_idx, pred in enumerate(predicted):
        best_idx = -1
        best_score = 0.0
        for gold_idx, gold_slot in enumerate(gold):
            if gold_idx in matched_gold:
                continue
            score = token_f1(gold_slot.slot_text, pred.slot_text)
            if score > best_score:
                best_idx = gold_idx
                best_score = score
        if best_idx >= 0 and best_score >= threshold:
            matched_gold.add(best_idx)
            pairs.append((pred_idx, best_idx, best_score))
    return pairs


def previous_outputs(slots: list[Slot], index: int) -> set[str]:
    return {slot.output_variable for slot in slots[:index] if slot.output_variable}


def depends_on_previous(predicted: list[Slot], index: int) -> bool:
    prev = previous_outputs(predicted, index)
    inputs = {str(value) for value in predicted[index].input_variables}
    return bool(prev & inputs)


def sample_metrics(predicted: list[Slot], gold: list[Slot], threshold: float) -> dict[str, float]:
    pairs = align_slots(predicted, gold, threshold)
    aligned = len(pairs)
    type_matches = 0
    dependency_presence_matches = 0
    dependency_chain_hits = 0
    dependency_chain_total = 0
    for pred_idx, gold_idx, _score in pairs:
        pred = predicted[pred_idx]
        gold_slot = gold[gold_idx]
        if (pred.expected_answer_type or "other") == (gold_slot.expected_answer_type or "other"):
            type_matches += 1
        gold_dep = bool(gold_slot.input_variables)
        pred_dep = bool(pred.input_variables)
        if gold_dep == pred_dep:
            dependency_presence_matches += 1
        if gold_dep:
            dependency_chain_total += 1
            if pred_idx > 0 and depends_on_previous(predicted, pred_idx):
                dependency_chain_hits += 1
    gold_deps = sum(1 for slot in gold if slot.input_variables)
    pred_internal_deps = sum(1 for idx in range(1, len(predicted)) if depends_on_previous(predicted, idx))
    return {
        "slot_count_gold": float(len(gold)),
        "slot_count_predicted": float(len(predicted)),
        "slot_count_exact": float(len(gold) == len(predicted)),
        "slot_count_within_one": float(abs(len(gold) - len(predicted)) <= 1),
        "aligned_slots": float(aligned),
        "aligned_answer_type_accuracy": type_matches / aligned if aligned else 0.0,
        "dependency_presence_accuracy": dependency_presence_matches / aligned if aligned else 0.0,
        "position_chain_accuracy": dependency_chain_hits / dependency_chain_total if dependency_chain_total else 0.0,
        "gold_dependent_slots": float(gold_deps),
        "predicted_internal_dependencies": float(pred_internal_deps),
    }


def mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row.get(key, 0.0) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--predicted_jsonl", default="data/processed/musique_qwen_slots.jsonl")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--json_out", default="reports/week0/qwen_slot_quality_posthoc.json")
    parser.add_argument("--md_out", default="reports/week0/qwen_slot_quality_posthoc.md")
    args = parser.parse_args()

    _corpus, samples = load_dataset(args.dataset, args.dataset_dir)
    by_qid = {str(sample.get("id") or sample.get("_id")): sample for sample in samples}
    prediction_rows = read_jsonl(args.predicted_jsonl)
    metrics: list[dict[str, float]] = []
    examples: list[dict[str, Any]] = []
    for row in prediction_rows:
        qid = str(row.get("qid") or "")
        sample = by_qid.get(qid)
        if not sample:
            continue
        gold = build_oracle_slots_for_sample(sample)
        predicted = parse_predicted_slots(row.get("predicted_slots") or [])
        item = sample_metrics(predicted, gold, args.threshold)
        metrics.append(item)
        if len(examples) < 8 and item["position_chain_accuracy"] < 1.0:
            examples.append(
                {
                    "qid": qid,
                    "question": sample.get("question"),
                    "gold_slots": [slot.to_dict() for slot in gold],
                    "predicted_slots": [slot.to_dict() for slot in predicted],
                    "metrics": item,
                }
            )

    aggregate = {
        "status": "completed" if metrics else "failed",
        "num_examples": len(metrics),
        "threshold": args.threshold,
        "metrics": {
            "avg_gold_slots": mean(metrics, "slot_count_gold"),
            "avg_predicted_slots": mean(metrics, "slot_count_predicted"),
            "slot_count_exact": mean(metrics, "slot_count_exact"),
            "slot_count_within_one": mean(metrics, "slot_count_within_one"),
            "aligned_answer_type_accuracy": mean(metrics, "aligned_answer_type_accuracy"),
            "dependency_presence_accuracy": mean(metrics, "dependency_presence_accuracy"),
            "position_chain_accuracy": mean(metrics, "position_chain_accuracy"),
            "avg_gold_dependent_slots": mean(metrics, "gold_dependent_slots"),
            "avg_predicted_internal_dependencies": mean(metrics, "predicted_internal_dependencies"),
        },
        "interpretation": (
            "position_chain_accuracy checks whether a generated dependent slot refers to a previous "
            "generated output variable, avoiding brittle exact-name matching against oracle x1/x2 labels."
        ),
        "examples": examples,
    }
    write_json(aggregate, args.json_out)

    out = Path(args.md_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen Slot Quality Post-hoc Diagnostics",
        "",
        f"- Status: `{aggregate['status']}`",
        f"- Examples: `{aggregate['num_examples']}`",
        f"- Alignment threshold: `{args.threshold}`",
        "",
        "## Aggregate Metrics",
    ]
    for key, value in aggregate["metrics"].items():
        lines.append(f"- {key}: `{value:.4f}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            aggregate["interpretation"],
            "",
            "The original exact variable-grounding metric remains useful as a schema strictness check, but this post-hoc metric is the better Week-0 decision signal for latent-slot feasibility.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
