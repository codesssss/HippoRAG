from __future__ import annotations

from typing import List, Sequence

import numpy as np


def select_union_candidate_indices(
    primary_indices: Sequence[int],
    secondary_indices: Sequence[int],
    fallback_indices: Sequence[int],
    candidate_k: int,
) -> np.ndarray:
    selected: List[int] = []
    seen = set()
    max_candidates = max(int(candidate_k), 0)
    if max_candidates <= 0:
        return np.zeros(0, dtype=np.int64)

    for sequence in (primary_indices, secondary_indices, fallback_indices):
        for idx in sequence:
            normalized_idx = int(idx)
            if normalized_idx in seen:
                continue
            seen.add(normalized_idx)
            selected.append(normalized_idx)
            if len(selected) >= max_candidates:
                return np.asarray(selected, dtype=np.int64)

    return np.asarray(selected, dtype=np.int64)
