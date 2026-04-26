"""Dataset parsing helpers for D-PathRAG.

The first D-PathRAG pilot uses 2WikiMultiHopQA because it exposes evidence
triples and supporting-fact annotations that can initialize an autoregressive
evidence-path selector.  These helpers keep the schema handling explicit so the
training code does not silently mix local 1000-example eval subsets with full
train/dev/test releases.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.dpathrag.io import read_json


@dataclass(frozen=True)
class DatasetBundle:
    """Local dataset files used by a D-PathRAG audit/cache build."""

    samples_path: Path
    corpus_path: Path
    samples: list[dict[str, Any]]
    corpus: list[dict[str, Any]]


def load_2wiki_bundle(data_root: str | Path, dataset: str = "2wikimultihopqa") -> DatasetBundle:
    root = Path(data_root)
    samples_path = root / f"{dataset}.json"
    corpus_path = root / f"{dataset}_corpus.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing 2Wiki samples file: {samples_path}")
    samples = read_json(samples_path)
    corpus = read_json(corpus_path) if corpus_path.exists() else []
    if not isinstance(samples, list):
        raise ValueError(f"Expected list in {samples_path}, got {type(samples).__name__}")
    if not isinstance(corpus, list):
        raise ValueError(f"Expected list in {corpus_path}, got {type(corpus).__name__}")
    return DatasetBundle(samples_path=samples_path, corpus_path=corpus_path, samples=samples, corpus=corpus)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def extract_title(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("title") or "").strip()
    return str(doc or "").split("\n", 1)[0].strip()


def support_titles(sample: dict[str, Any]) -> list[str]:
    facts = sample.get("supporting_facts") or []
    titles = []
    for fact in facts:
        if isinstance(fact, (list, tuple)) and fact:
            titles.append(str(fact[0]))
        elif isinstance(fact, dict) and "title" in fact:
            titles.append(str(fact["title"]))
    return sorted(set(titles))


def evidence_path_entities(sample: dict[str, Any]) -> list[str]:
    """Return the entity sequence exposed by 2Wiki evidence triples.

    A path like [A, relation, B], [B, relation, C] becomes [A, B, C].  The
    result is diagnostic metadata, not a training target by itself.
    """

    entities: list[str] = []
    for triple in sample.get("evidences") or []:
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        head = str(triple[0])
        tail = str(triple[2])
        if not entities:
            entities.append(head)
        if not entities or normalize_text(entities[-1]) != normalize_text(tail):
            entities.append(tail)
    return entities


def context_titles(sample: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for item in sample.get("context") or []:
        if isinstance(item, (list, tuple)) and item:
            titles.append(str(item[0]))
        elif isinstance(item, dict) and "title" in item:
            titles.append(str(item["title"]))
    return titles


def field_coverage(samples: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, float]:
    if not samples:
        return {field: 0.0 for field in fields}
    return {
        field: round(sum(1 for sample in samples if field in sample and sample.get(field) is not None) / len(samples), 4)
        for field in fields
    }


def summarize_2wiki_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    support_counts = [len(sample.get("supporting_facts") or []) for sample in samples]
    support_title_counts = [len(support_titles(sample)) for sample in samples]
    evidence_lens = [len(sample.get("evidences") or []) for sample in samples]
    context_counts = [len(sample.get("context") or []) for sample in samples]
    type_counts = Counter(str(sample.get("type", "unknown")) for sample in samples)
    fields = ["_id", "type", "question", "context", "supporting_facts", "evidences", "evidences_id", "answer"]
    examples: list[dict[str, Any]] = []
    for sample in list(samples)[:3]:
        examples.append(
            {
                "id": sample.get("_id"),
                "type": sample.get("type"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "supporting_facts": sample.get("supporting_facts"),
                "evidences": sample.get("evidences"),
                "evidence_path_entities": evidence_path_entities(sample),
            }
        )
    return {
        "field_coverage": field_coverage(samples, fields),
        "type_distribution": dict(sorted(type_counts.items())),
        "avg_supporting_facts": round(float(mean(support_counts)) if support_counts else 0.0, 4),
        "avg_support_titles": round(float(mean(support_title_counts)) if support_title_counts else 0.0, 4),
        "avg_evidence_path_len": round(float(mean(evidence_lens)) if evidence_lens else 0.0, 4),
        "avg_context_docs": round(float(mean(context_counts)) if context_counts else 0.0, 4),
        "examples": examples,
    }


def infer_split_status(data_root: str | Path, dataset: str, sample_count: int) -> dict[str, Any]:
    root = Path(data_root)
    split_candidates = sorted(
        str(path)
        for path in root.glob("*2wiki*")
        if path.is_file() and any(token in path.name.lower() for token in ("train", "dev", "valid", "test"))
    )
    has_train = any("train" in Path(path).name.lower() for path in split_candidates)
    has_dev = any(("dev" in Path(path).name.lower() or "valid" in Path(path).name.lower()) for path in split_candidates)
    has_test = any("test" in Path(path).name.lower() for path in split_candidates)
    if has_train and has_dev:
        status = "train_dev_available"
    elif sample_count == 1000 and not split_candidates:
        status = "local_eval_subset_only"
    else:
        status = "unknown_local_release"
    return {
        "status": status,
        "has_train_dev_test": bool(has_train and has_dev and has_test),
        "split_files": split_candidates,
        "notes": [
            "The local 1000-example file is suitable for smoke/cache audit, not for full E2E selector training."
            if status == "local_eval_subset_only"
            else "Verify split provenance before training D-PathRAG."
        ],
    }

