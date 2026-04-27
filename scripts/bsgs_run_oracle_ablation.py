#!/usr/bin/env python3
"""Run cheap oracle diagnostics for the failed BSGS operator route.

This script is intentionally additive: it does not change the Week-1 minimal
runner.  The diagnostic isolates whether oracle access to gold support
paragraphs, gold binding rewrites, or oracle likelihoods can rescue the
node-marginal BSGS operator.  If they cannot, the remaining bottleneck is the
graph/state transition itself rather than slot quality or verifier noise.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.extractor import extract_mentions, heuristic_extract_propositions
from src.bsgs.io import load_dataset, read_jsonl, write_json
from src.bsgs.metrics import belief_entropy, recall_at_gold
from src.bsgs.runner_oracle import (
    build_paragraph_propositions,
    build_shared_doc_edges,
    lexical_likelihood,
)
from src.bsgs.slots import Slot
from src.bsgs.state import BeliefStateGraph, PropositionNode
from src.bsgs.transition import semi_absorbing_transition, sparse_transition


VARIANTS = (
    "base",
    "oracle_pool",
    "oracle_pool_binding",
    "oracle_pool_binding_likelihood",
)

REF_RE = re.compile(r"#(\d+)")


def slot_records_by_qid(path: str) -> dict[str, list[Slot]]:
    records = read_jsonl(path)
    by_qid: dict[str, list[Slot]] = {}
    for row in records:
        qid = str(row.get("qid"))
        by_qid[qid] = [Slot(**slot) for slot in row.get("slots") or []]
    return by_qid


def support_idx_by_slot(sample: dict[str, Any]) -> list[Any]:
    decomposition = sample.get("question_decomposition") or []
    return [item.get("paragraph_support_idx") for item in decomposition]


def gold_paragraph_indices(sample: dict[str, Any]) -> list[Any]:
    return [para.get("idx") for para in sample.get("paragraphs") or [] if bool(para.get("is_supporting"))]


def recall_at_gold_idx(selected: Sequence[Any], gold: Sequence[Any]) -> float:
    selected_set = {str(item) for item in selected if item is not None}
    gold_set = {str(item) for item in gold if item is not None}
    if not gold_set:
        return 0.0
    return len(selected_set & gold_set) / len(gold_set)


def rewrite_slot_with_gold_bindings(slot: Slot, prior_gold_answers: Sequence[str]) -> Slot:
    """Substitute MuSiQue #1/#2 references with prior oracle slot answers."""

    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(prior_gold_answers) and prior_gold_answers[idx]:
            return prior_gold_answers[idx]
        return match.group(0)

    rewritten = REF_RE.sub(repl, slot.slot_text)
    if rewritten == slot.slot_text:
        return slot
    return replace(slot, slot_text=rewritten)


def build_diagnostic_propositions(
    sample: dict[str, Any],
    oracle_pool: bool,
    base_max_props: int = 4,
    support_max_props: int = 80,
) -> dict[str, PropositionNode]:
    if not oracle_pool:
        return build_paragraph_propositions(sample, max_props_per_paragraph=base_max_props)

    propositions: dict[str, PropositionNode] = {}
    for para in sample.get("paragraphs") or []:
        title = str(para.get("title") or f"paragraph-{para.get('idx')}")
        doc_id = f"{para.get('idx')}::{title}"
        is_supporting = bool(para.get("is_supporting"))
        metadata = {
            "title": title,
            "paragraph_idx": para.get("idx"),
            "is_supporting": is_supporting,
        }
        passage = str(para.get("paragraph_text") or "")
        max_props = support_max_props if is_supporting else base_max_props
        for prop in heuristic_extract_propositions(
            passage=passage,
            source_doc_id=doc_id,
            max_props=max_props,
            metadata=metadata,
        ):
            propositions[prop.prop_id] = prop

        if is_supporting and passage.strip():
            # Oracle-pool diagnostic: expose the entire gold paragraph so the
            # operator cannot fail merely because the relevant sentence was
            # truncated by the base sentence extractor.
            prop_id = f"{doc_id}::oracle_full"
            propositions[prop_id] = PropositionNode(
                prop_id=prop_id,
                text=passage.strip(),
                source_doc_id=doc_id,
                source_span=passage.strip(),
                mentions=extract_mentions(passage),
                metadata={**metadata, "oracle_full_paragraph": True},
            )
    return propositions


def oracle_likelihood_for_slot(
    prop: PropositionNode,
    slot_support_idx: Any,
    high: float = 0.95,
    low: float = 0.05,
) -> float:
    prop_idx = prop.metadata.get("paragraph_idx")
    if slot_support_idx is not None and str(prop_idx) == str(slot_support_idx):
        return high
    return low


def likelihood_for_variant(
    slot: Slot,
    prop: PropositionNode,
    variant: str,
    slot_support_idx: Any,
) -> float:
    if variant == "oracle_pool_binding_likelihood":
        return oracle_likelihood_for_slot(prop, slot_support_idx)
    return lexical_likelihood(slot.slot_text, prop.text)


def coverage_bin(value: float) -> str:
    if value >= 0.999:
        return "full_support_covered"
    if value <= 0.0:
        return "no_support_covered"
    return "partial_support_covered"


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def duplicate_rate(items: Sequence[str]) -> float:
    clean = [item for item in items if item]
    if not clean:
        return 0.0
    return 1.0 - (len(set(clean)) / len(clean))


def run_sample_variant(
    sample: dict[str, Any],
    slots: Sequence[Slot],
    variant: str,
    top_k: int,
    absorbing_enabled: bool = False,
) -> dict[str, Any]:
    oracle_pool = variant in {
        "oracle_pool",
        "oracle_pool_binding",
        "oracle_pool_binding_likelihood",
    }
    use_binding = variant in {"oracle_pool_binding", "oracle_pool_binding_likelihood"}

    propositions = build_diagnostic_propositions(sample, oracle_pool=oracle_pool)
    edges = build_shared_doc_edges(propositions)
    graph = BeliefStateGraph.uniform(propositions=propositions, edges=edges)
    slot_support_indices = support_idx_by_slot(sample)

    trace: list[dict[str, Any]] = []
    prior_gold_answers: list[str] = []
    for step_idx, slot in enumerate(slots):
        active_slot = rewrite_slot_with_gold_bindings(slot, prior_gold_answers) if use_binding else slot
        slot_support_idx = slot_support_indices[step_idx] if step_idx < len(slot_support_indices) else None
        likelihood = {
            prop_id: likelihood_for_variant(active_slot, prop, variant, slot_support_idx)
            for prop_id, prop in graph.propositions.items()
        }
        transition = sparse_transition(graph.edges, likelihood, nodes=graph.propositions)
        if absorbing_enabled:
            transition = semi_absorbing_transition(transition, likelihood, nodes=graph.propositions)
        graph.belief = graph.predict(transition)
        graph.observe(likelihood)
        trace.append(
            {
                "slot": asdict(active_slot),
                "original_slot_text": slot.slot_text,
                "slot_support_idx": slot_support_idx,
                "entropy": belief_entropy(graph.belief),
                "top_props": graph.topk(3),
            }
        )
        prior_gold_answers.append(str(slot.gold_answer or ""))

    selected = graph.topk(top_k, support_weighted=False)
    selected_props = [graph.propositions[prop_id] for prop_id, _ in selected]
    selected_titles = [str(prop.metadata.get("title") or prop.source_doc_id) for prop in selected_props]
    selected_paragraph_indices = [prop.metadata.get("paragraph_idx") for prop in selected_props]
    gold_titles = [
        str(para.get("title"))
        for para in (sample.get("paragraphs") or [])
        if bool(para.get("is_supporting"))
    ]
    gold_indices = gold_paragraph_indices(sample)
    title_recall = recall_at_gold(selected_titles, gold_titles)
    idx_recall = recall_at_gold_idx(selected_paragraph_indices, gold_indices)
    return {
        "qid": sample.get("id") or sample.get("_id"),
        "question": sample.get("question"),
        "variant": variant,
        "selected_prop_ids": [prop_id for prop_id, _ in selected],
        "selected_titles": selected_titles,
        "selected_paragraph_indices": selected_paragraph_indices,
        "gold_titles": gold_titles,
        "gold_paragraph_indices": gold_indices,
        "supporting_paragraph_recall": title_recall,
        "legacy_title_recall": title_recall,
        "supporting_paragraph_idx_recall": idx_recall,
        "belief_entropy": graph.entropy(),
        "duplicate_title_rate": duplicate_rate(selected_titles),
        "trace": trace,
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    title_recalls = [float(row.get("supporting_paragraph_recall") or 0.0) for row in rows]
    idx_recalls = [float(row.get("supporting_paragraph_idx_recall") or 0.0) for row in rows]
    title_bins = {"full_support_covered": 0, "partial_support_covered": 0, "no_support_covered": 0}
    idx_bins = {"full_support_covered": 0, "partial_support_covered": 0, "no_support_covered": 0}
    for value in title_recalls:
        title_bins[coverage_bin(value)] += 1
    for value in idx_recalls:
        idx_bins[coverage_bin(value)] += 1
    return {
        "n": len(rows),
        "supporting_paragraph_recall": mean(title_recalls),
        "legacy_title_recall": mean(title_recalls),
        "supporting_paragraph_idx_recall": mean(idx_recalls),
        "avg_belief_entropy": mean([float(row.get("belief_entropy") or 0.0) for row in rows]),
        "avg_duplicate_title_rate": mean([float(row.get("duplicate_title_rate") or 0.0) for row in rows]),
        "title_support_coverage_bins": title_bins,
        "paragraph_idx_support_coverage_bins": idx_bins,
    }


def write_markdown_report(payload: dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BSGS Oracle Diagnostic Ablation",
        "",
        "This diagnostic tests whether oracle pool exposure, oracle binding rewrites,",
        "or oracle likelihoods can rescue the node-marginal BSGS operator after the",
        "main Week-1 route failed on MuSiQue.",
        "",
        "## Config",
        f"- dataset: `{payload.get('dataset')}`",
        f"- n: `{payload.get('n')}`",
        f"- top_k: `{payload.get('top_k')}`",
        f"- absorbing_enabled: `{payload.get('absorbing_enabled')}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Title Recall | Paragraph-Idx Recall | Full/Partial/None Title | Full/Partial/None Idx | Entropy | Dup Title |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        summary = payload["variants"].get(variant, {}).get("summary", {})
        title_bins = summary.get("title_support_coverage_bins") or {}
        idx_bins = summary.get("paragraph_idx_support_coverage_bins") or {}
        title_bin_str = (
            f"{title_bins.get('full_support_covered', 0)}/"
            f"{title_bins.get('partial_support_covered', 0)}/"
            f"{title_bins.get('no_support_covered', 0)}"
        )
        idx_bin_str = (
            f"{idx_bins.get('full_support_covered', 0)}/"
            f"{idx_bins.get('partial_support_covered', 0)}/"
            f"{idx_bins.get('no_support_covered', 0)}"
        )
        lines.append(
            f"| `{variant}` | `{summary.get('legacy_title_recall', 0.0):.4f}` | "
            f"`{summary.get('supporting_paragraph_idx_recall', 0.0):.4f}` | "
            f"`{title_bin_str}` | `{idx_bin_str}` | "
            f"`{summary.get('avg_belief_entropy', 0.0):.4f}` | "
            f"`{summary.get('avg_duplicate_title_rate', 0.0):.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "- `oracle_pool` exposes complete gold support paragraphs as candidate propositions.",
            "- `oracle_pool_binding` additionally rewrites `#1/#2` with prior gold slot answers.",
            "- `oracle_pool_binding_likelihood` additionally assigns oracle high/low likelihood by slot support paragraph.",
            "- If the last variant remains weak, the failure is not slot quality, binding, or verifier calibration; it is the graph/state transition.",
            "",
            "## Debug Examples",
            "",
        ]
    )
    for variant in VARIANTS:
        rows = payload["variants"].get(variant, {}).get("rows", [])
        lines.append(f"### {variant}")
        for row in rows[:5]:
            lines.append(
                f"- `{row.get('qid')}` title_recall=`{row.get('legacy_title_recall'):.4f}` "
                f"idx_recall=`{row.get('supporting_paragraph_idx_recall'):.4f}` "
                f"selected={row.get('selected_titles')} gold={row.get('gold_titles')}"
            )
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--oracle_slots", default="data/processed/musique_oracle_slots.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--absorbing", action="store_true")
    parser.add_argument("--json_out", default="reports/week1/bsgs_oracle_diagnostic_musique200.json")
    parser.add_argument("--md_out", default="reports/week1/bsgs_oracle_diagnostic_musique200.md")
    args = parser.parse_args()

    _corpus, samples = load_dataset(args.dataset, dataset_dir=args.dataset_dir)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    slots_by_qid = slot_records_by_qid(args.oracle_slots)

    variants: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        rows: list[dict[str, Any]] = []
        for sample in samples:
            qid = str(sample.get("id") or sample.get("_id"))
            slots = slots_by_qid.get(qid)
            if not slots:
                continue
            rows.append(
                run_sample_variant(
                    sample=sample,
                    slots=slots,
                    variant=variant,
                    top_k=args.top_k,
                    absorbing_enabled=args.absorbing,
                )
            )
        variants[variant] = {
            "summary": summarize_rows(rows),
            "rows": rows,
        }

    payload = {
        "status": "completed",
        "dataset": args.dataset,
        "n": next(iter(variants.values()))["summary"]["n"] if variants else 0,
        "top_k": args.top_k,
        "absorbing_enabled": args.absorbing,
        "variants": variants,
    }
    write_json(payload, args.json_out)
    write_markdown_report(payload, args.md_out)
    print(f"Wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
