from pathlib import Path
from typing import Tuple


DATASET_FILE_STEM_ALIASES = {
    "nq": "nq_rear",
    "naturalquestions": "nq_rear",
}


def resolve_dataset_file_stem(dataset_name: str) -> str:
    normalized = str(dataset_name).strip()
    if not normalized:
        raise ValueError("dataset_name must be non-empty")
    return DATASET_FILE_STEM_ALIASES.get(normalized, normalized)


def resolve_dataset_paths(dataset_name: str, dataset_root: str | Path = "reproduce/dataset") -> Tuple[Path, Path]:
    root = Path(dataset_root)
    file_stem = resolve_dataset_file_stem(dataset_name)
    sample_path = root / f"{file_stem}.json"
    corpus_path = root / f"{file_stem}_corpus.json"
    return corpus_path, sample_path
