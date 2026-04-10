import sys
import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODULE_PATH = ROOT_DIR / "scripts" / "diagnose_coverage_atoms.py"
SPEC = importlib.util.spec_from_file_location("diagnose_coverage_atoms", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_doc_cover_maps = MODULE.build_doc_cover_maps
evaluate_candidate_subsets = MODULE.evaluate_candidate_subsets
is_valid_subset = MODULE.is_valid_subset


def test_is_valid_subset_rejects_duplicate_titles():
    pool_docs = [
        "Alpha\nbody",
        "Alpha\nanother body",
        "Beta\nbody",
    ]
    assert not is_valid_subset((0, 1), pool_docs)
    assert is_valid_subset((0, 2), pool_docs)


def test_evaluate_candidate_subsets_prefers_higher_coverage_then_score():
    pool_docs = [
        "Alpha\nbody",
        "Beta\nbody",
        "Gamma\nbody",
    ]
    pool_scores = [0.9, 0.8, 0.2]
    doc_covers_q = {
        0: {"q1"},
        1: {"q2"},
        2: set(),
    }
    doc_covers_e = {
        0: {("a", "b")},
        1: set(),
        2: {("b", "c"), ("c", "d")},
    }
    doc_covers_b = {
        0: {"b1"},
        1: {"b2"},
        2: {"b3", "b4"},
    }

    stats = evaluate_candidate_subsets(
        candidate_positions=[0, 1, 2],
        qa_top_k=2,
        pool_docs=pool_docs,
        pool_scores=pool_scores,
        doc_covers_q=doc_covers_q,
        doc_covers_e=doc_covers_e,
        doc_covers_b=doc_covers_b,
    )

    assert stats["valid_subset_count"] == 3
    assert stats["cov_q_values"] == [1, 2]
    assert stats["cov_e_values"] == [1, 2, 3]
    assert stats["best_coverage_positions"] == [0, 1]
    assert stats["best_baseline_score_positions"] == [0, 1]
    assert stats["cov_q_at_max_cov_e"] == 1


def test_build_doc_cover_maps_constructs_atom_universes():
    a_q, a_e, a_b, doc_covers_q, doc_covers_e, doc_covers_b = build_doc_cover_maps(
        candidate_positions=[0, 1],
        pool_doc_ids=[10, 11],
        doc_idx_to_entities={
            10: {"alpha", "beta"},
            11: {"beta", "gamma"},
        },
        doc_idx_to_edges={
            10: [("alpha", "beta", 1.0, "rel")],
            11: [("beta", "gamma", 1.0, "rel")],
        },
        seed_entities={"alpha"},
    )

    assert a_q == {"alpha"}
    assert a_e == {("alpha", "beta"), ("beta", "gamma")}
    assert a_b == {"beta", "gamma"}
    assert doc_covers_q[0] == {"alpha"}
    assert doc_covers_q[1] == set()
    assert doc_covers_e[0] == {("alpha", "beta")}
    assert doc_covers_b[1] == {"beta", "gamma"}
