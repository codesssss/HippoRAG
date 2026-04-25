"""I/O helpers for BSGS scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Any, path: str | Path) -> None:
    out = Path(path)
    ensure_parent(out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    ensure_parent(out)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset(dataset: str, dataset_dir: str | Path = "reproduce/dataset") -> tuple[list[dict], list[dict]]:
    root = Path(dataset_dir)
    sample_path = root / f"{dataset}.json"
    corpus_path = root / f"{dataset}_corpus.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {sample_path}")
    samples = read_json(sample_path)
    corpus = read_json(corpus_path) if corpus_path.exists() else []
    return corpus, samples
