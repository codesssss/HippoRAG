from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("audit_agsto_clean_rrf", ROOT / "scripts/audit_agsto_clean_rrf.py")
assert _SPEC is not None and _SPEC.loader is not None
audit_agsto_clean_rrf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_agsto_clean_rrf)

connected_in_sto = audit_agsto_clean_rrf.connected_in_sto
hub_violation = audit_agsto_clean_rrf.hub_violation
jaccard_at_k = audit_agsto_clean_rrf.jaccard_at_k
leave_one_out_rrf = audit_agsto_clean_rrf.leave_one_out_rrf
rrf_rank = audit_agsto_clean_rrf.rrf_rank
title_recall_at_k = audit_agsto_clean_rrf.title_recall_at_k
weight_key = audit_agsto_clean_rrf.weight_key


def toy_corpus_index() -> dict[str, object]:
    return {
        "doc_to_endpoints": {
            0: ["a", "hub"],
            1: ["a", "hub"],
            2: ["hub"],
            3: ["hub"],
        },
        "endpoint_to_docs": {
            "a": [0, 1],
            "hub": [0, 1, 2, 3],
        },
    }


def test_rrf_rank_uses_support_weight_and_stable_tie_breaking() -> None:
    channels = {
        "specificity": [1, 2],
        "hybrid_residual": [2, 3],
        "neighborhood": [3, 1],
        "support_set": [4, 1],
    }

    assert rrf_rank(channels, top_k=4, support_weight=2.0) == [1, 4, 2, 3]
    assert rrf_rank(channels, top_k=4, support_weight=None) == [1, 2, 3, 4]


def test_leave_one_out_rrf_removes_exactly_one_channel() -> None:
    channels = {
        "specificity": [1, 2],
        "hybrid_residual": [2, 3],
        "neighborhood": [3, 1],
        "support_set": [4, 1],
    }

    outputs = leave_one_out_rrf(channels, top_k=4, support_weight=1.75)
    unweighted_outputs = leave_one_out_rrf(channels, top_k=4, support_weight=None)

    assert sorted(outputs) == [
        "minus_hybrid_residual",
        "minus_neighborhood",
        "minus_specificity",
        "minus_support_set",
    ]
    assert 4 not in outputs["minus_support_set"]
    assert outputs["minus_specificity"][0] == 1
    assert unweighted_outputs["minus_support_set"] == [1, 2, 3]


def test_overlap_recall_and_weight_key_helpers() -> None:
    assert jaccard_at_k([1, 2, 3], [2, 3, 4], 3) == 0.5
    assert jaccard_at_k([], [], 10) == 0.0
    assert title_recall_at_k(["A", "B"], ["B", "C"], 5) == 0.5
    assert title_recall_at_k([], ["B"], 5) == 0.0
    assert weight_key(1) == "1.0"
    assert weight_key(1.75) == "1.75"


def test_sto_connectivity_uses_low_degree_shared_endpoints() -> None:
    corpus_index = toy_corpus_index()

    assert connected_in_sto([0, 1], corpus_index=corpus_index, max_endpoint_degree=2)
    assert not connected_in_sto([0, 2], corpus_index=corpus_index, max_endpoint_degree=2)
    assert connected_in_sto([0, 2], corpus_index=corpus_index, max_endpoint_degree=4)


def test_hub_violation_flags_high_degree_shared_endpoint() -> None:
    corpus_index = toy_corpus_index()

    assert hub_violation([0, 2], corpus_index=corpus_index, max_endpoint_degree=2)
    assert hub_violation([0, 1], corpus_index=corpus_index, max_endpoint_degree=2)
    assert not hub_violation([0, 1], corpus_index=corpus_index, max_endpoint_degree=4)
