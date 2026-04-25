#!/usr/bin/env python3
"""Audit the local MuSiQue split before BSGS experiments."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bsgs.io import load_dataset, write_json


KNOWN_MUSIQUE_ANS_DEV_SIZE = 2417
KNOWN_MUSIQUE_ANS_TOTAL = 24814


def has_supporting_paragraphs(sample: dict[str, Any]) -> bool:
    paragraphs = sample.get("paragraphs") or []
    if any("is_supporting" in para for para in paragraphs):
        return True
    return any(item.get("paragraph_support_idx") is not None for item in sample.get("question_decomposition") or [])


def infer_variant(samples: list[dict[str, Any]]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not samples:
        return "unknown", ["No local samples found."]
    answerable_values = [sample.get("answerable") for sample in samples]
    answerable_counter = Counter(answerable_values)
    all_answerable = all(value is True for value in answerable_values if value is not None)
    any_unanswerable = any(value is False for value in answerable_values)
    if len(samples) == KNOWN_MUSIQUE_ANS_DEV_SIZE and all_answerable:
        return "MuSiQue-Ans", ["Local size matches official MuSiQue-Ans dev size."]
    if all_answerable and len(samples) < KNOWN_MUSIQUE_ANS_DEV_SIZE:
        notes.append(
            "Local file is answerable-only and smaller than official MuSiQue-Ans dev; treat it as a local MuSiQue-Ans-derived subset."
        )
        return "MuSiQue-Ans-local-subset", notes
    if any_unanswerable:
        notes.append(f"Found answerable distribution: {dict(answerable_counter)}")
        return "MuSiQue-Full-or-mixed", notes
    notes.append(f"Could not infer from answerable distribution: {dict(answerable_counter)}")
    return "unknown", notes


def find_musique_files(dataset_dir: Path) -> list[str]:
    return [str(path) for path in sorted(dataset_dir.glob("*musique*")) if path.is_file()]


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    corpus, samples = load_dataset(args.dataset, dataset_dir=dataset_dir)
    variant, notes = infer_variant(samples)
    has_decomposition = any(bool(sample.get("question_decomposition")) for sample in samples)
    support_present = any(has_supporting_paragraphs(sample) for sample in samples)
    answerable_counter = Counter(sample.get("answerable") for sample in samples)
    source_file = dataset_dir / f"{args.dataset}.json"
    corpus_file = dataset_dir / f"{args.dataset}_corpus.json"
    recommended = str(source_file)
    if variant == "MuSiQue-Full-or-mixed":
        recommended += " filtered to answerable examples for oracle-slot diagnostic"
    notes.extend(
        [
            f"Official reference: MuSiQue-Ans total={KNOWN_MUSIQUE_ANS_TOTAL}, dev={KNOWN_MUSIQUE_ANS_DEV_SIZE}.",
            "Do not tune latent-slot prompts on the same examples used for oracle-slot diagnostics.",
        ]
    )
    return {
        "dataset_name": "musique",
        "variant": variant,
        "train_size": 0,
        "dev_size": len(samples),
        "test_size": 0,
        "local_sample_size": len(samples),
        "local_corpus_size": len(corpus),
        "has_unanswerable": any(value is False for value in answerable_counter),
        "answerable_distribution": {str(k): int(v) for k, v in answerable_counter.items()},
        "has_decomposition": bool(has_decomposition),
        "has_supporting_paragraphs": bool(support_present),
        "current_layer1_musique_1000_source": str(source_file),
        "current_layer1_musique_1000_corpus": str(corpus_file) if corpus_file.exists() else "",
        "recommended_eval_split": recommended,
        "all_local_musique_files": find_musique_files(dataset_dir),
        "notes": notes,
    }


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# MuSiQue Split Audit",
        "",
        f"- Variant: `{audit['variant']}`",
        f"- Local sample size: `{audit['local_sample_size']}`",
        f"- Local corpus size: `{audit['local_corpus_size']}`",
        f"- Has unanswerable: `{audit['has_unanswerable']}`",
        f"- Has decomposition: `{audit['has_decomposition']}`",
        f"- Has supporting paragraphs: `{audit['has_supporting_paragraphs']}`",
        f"- Current Layer-1 MuSiQue-1000 source: `{audit['current_layer1_musique_1000_source']}`",
        f"- Recommended eval split: `{audit['recommended_eval_split']}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in audit.get("notes") or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset_dir", default="reproduce/dataset")
    parser.add_argument("--json_out", default="reports/week0/split_audit.json")
    parser.add_argument("--md_out", default="reports/week0/split_audit.md")
    args = parser.parse_args()

    audit = build_audit(args)
    write_json(audit, args.json_out)
    write_markdown(audit, Path(args.md_out))
    print(f"Wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
