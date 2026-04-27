#!/usr/bin/env python3
"""Aggregate Week-0 BSGS audit reports into a decision memo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.io import read_json, write_json


def safe_read(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "missing", "path": path}
    return read_json(p)


def decide(
    split: dict[str, Any],
    nli: dict[str, Any],
    slots: dict[str, Any],
    qwen: dict[str, Any],
    qwen_posthoc: dict[str, Any],
) -> dict[str, Any]:
    components_enabled: list[str] = []
    components_disabled = ["latent-slot main path", "soft binding", "DPP", "LLM verifier"]
    risks: list[str] = []

    if split.get("variant") in {"MuSiQue-Ans", "MuSiQue-Ans-local-subset"}:
        components_enabled.append("MuSiQue answerable-subset diagnostics")
    else:
        risks.append("MuSiQue split is not clearly answerable-only; filter before oracle-slot evaluation.")

    if nli.get("absorbing_status") == "main" and not nli.get("lexical_smoke"):
        components_enabled.append("semi-absorbing transition")
    else:
        components_disabled.append("semi-absorbing transition as main")
        if nli.get("lexical_smoke"):
            risks.append("Formal DeBERTa calibration has not run; lexical smoke is not a main verifier result, so absorbing remains ablation/fallback.")
        else:
            risks.append("NLI calibration did not pass or was unavailable; absorbing remains ablation/fallback.")

    if slots.get("status") == "completed" and slots.get("num_examples", 0) > 0:
        components_enabled.append("oracle-slot Week-1 diagnostic")
    else:
        risks.append("Oracle slots missing; Week 1 cannot run until built.")

    qwen_recall = float((qwen.get("metrics") or {}).get("slot_recall") or 0.0)
    if qwen.get("status") == "completed" and qwen_recall >= 0.70:
        components_enabled.append("latent-slot route as Week-2 candidate")
        if "latent-slot main path" in components_disabled:
            components_disabled.remove("latent-slot main path")
    elif qwen.get("status") == "completed":
        risks.append(
            f"Qwen slot recall is {qwen_recall:.4f}, below the 0.70 Week-2 main-route gate; keep latent slots diagnostic-only."
        )
    else:
        risks.append("Qwen slot quality is missing; latent-slot route cannot be promoted.")

    if qwen_posthoc.get("status") == "completed":
        chain_acc = float((qwen_posthoc.get("metrics") or {}).get("position_chain_accuracy") or 0.0)
        if chain_acc < 0.70:
            risks.append(
                f"Post-hoc position-chain accuracy is {chain_acc:.4f}; Qwen often misses explicit dependency wiring even when slot text matches."
            )

    return {
        "week1_config": {
            "dataset": "musique",
            "slot_mode": "oracle-slot",
            "binding": "hard normalized string / alias match",
            "selector": "posterior top-k",
        },
        "components_enabled": components_enabled,
        "components_disabled": components_disabled,
        "risks": risks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", default="reports/week0/split_audit.json")
    parser.add_argument("--ircot_json", default="reports/week0/ircot_musique1000.json")
    parser.add_argument("--nli_json", default="reports/week0/nli_calibration.json")
    parser.add_argument("--oracle_json", default="reports/week0/oracle_slot_pipeline.json")
    parser.add_argument("--qwen_json", default="reports/week0/qwen_slot_quality.json")
    parser.add_argument("--qwen_posthoc_json", default="reports/week0/qwen_slot_quality_posthoc.json")
    parser.add_argument("--json_out", default="reports/week0/decision.json")
    parser.add_argument("--md_out", default="reports/week0/decision.md")
    args = parser.parse_args()

    split = safe_read(args.split_json)
    ircot = safe_read(args.ircot_json)
    nli = safe_read(args.nli_json)
    oracle = safe_read(args.oracle_json)
    qwen = safe_read(args.qwen_json)
    qwen_posthoc = safe_read(args.qwen_posthoc_json)
    decision = decide(split, nli, oracle, qwen, qwen_posthoc)
    payload = {
        "split_audit": split,
        "ircot_baseline": ircot,
        "nli_calibration": nli,
        "oracle_slot_pipeline": oracle,
        "qwen_slot_quality": qwen,
        "qwen_slot_quality_posthoc": qwen_posthoc,
        "decision": decision,
    }
    write_json(payload, args.json_out)

    lines = [
        "# Week 0 Decision Report",
        "",
        "## Split audit",
        f"- Variant: `{split.get('variant', 'missing')}`",
        f"- Dev/local size: `{split.get('dev_size', split.get('local_sample_size', ''))}`",
        f"- Current MuSiQue-1000 source: `{split.get('current_layer1_musique_1000_source', '')}`",
        f"- Recommended evaluation split: `{split.get('recommended_eval_split', '')}`",
        "",
        "## IRCoT baseline",
        f"- Status: `{ircot.get('status', 'missing')}`",
        f"- MuSiQue EM/F1: `{ircot.get('answer_em', '')}` / `{ircot.get('answer_f1', '')}`",
        f"- Latency: `{ircot.get('latency_per_query', '')}`",
        f"- LLM calls/query: `{ircot.get('llm_calls_per_query', '')}`",
        "",
        "## NLI calibration",
        f"- Status: `{nli.get('status', 'missing')}`",
        f"- ECE: `{nli.get('ece', '')}`",
        f"- Brier: `{nli.get('brier', '')}`",
        f"- Absorbing status: `{nli.get('absorbing_status', '')}`",
        "",
        "## Oracle slot pipeline",
        f"- Status: `{oracle.get('status', 'missing')}`",
        f"- Built examples: `{oracle.get('num_examples', '')}`",
        f"- Issues: `{oracle.get('contamination_warning', '')}`",
        "",
        "## Qwen slot quality",
        f"- Status: `{qwen.get('status', 'missing')}`",
        f"- Slot recall: `{(qwen.get('metrics') or {}).get('slot_recall', '')}`",
        f"- Slot precision: `{(qwen.get('metrics') or {}).get('slot_precision', '')}`",
        f"- Variable grounding accuracy: `{(qwen.get('metrics') or {}).get('variable_grounding_accuracy', '')}`",
        "",
        "## Qwen slot quality post-hoc",
        f"- Status: `{qwen_posthoc.get('status', 'missing')}`",
        f"- Position-chain accuracy: `{(qwen_posthoc.get('metrics') or {}).get('position_chain_accuracy', '')}`",
        f"- Dependency presence accuracy: `{(qwen_posthoc.get('metrics') or {}).get('dependency_presence_accuracy', '')}`",
        f"- Slot count within one: `{(qwen_posthoc.get('metrics') or {}).get('slot_count_within_one', '')}`",
        "",
        "## Decision",
        f"- Week 1 config: `{decision['week1_config']}`",
        f"- Components enabled: `{', '.join(decision['components_enabled'])}`",
        f"- Components disabled: `{', '.join(decision['components_disabled'])}`",
        "",
        "## Risks",
    ]
    lines.extend(f"- {risk}" for risk in decision["risks"])
    out = Path(args.md_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
