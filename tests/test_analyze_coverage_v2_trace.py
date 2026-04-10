import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODULE_PATH = ROOT_DIR / "scripts" / "analyze_coverage_v2_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_coverage_v2_trace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_doc_cover_maps_for_atom_source = MODULE.build_doc_cover_maps_for_atom_source
compute_edge_support_counts = MODULE.compute_edge_support_counts
compute_selected_appended_doc_rows = MODULE.compute_selected_appended_doc_rows
classify_route = MODULE.classify_route
summarize_admissibility_metrics = MODULE.summarize_admissibility_metrics


def test_build_doc_cover_maps_for_atom_source_freezes_edges_to_baseline_prefix():
    _, a_e, _, _, doc_covers_e, _ = build_doc_cover_maps_for_atom_source(
        candidate_positions=[0, 1, 2],
        atom_source_positions=[0, 1],
        pool_doc_ids=[10, 11, 12],
        doc_idx_to_entities={
            10: {"alpha", "base"},
            11: {"base", "beta"},
            12: {"poison", "target"},
        },
        doc_idx_to_edges={
            10: [("alpha", "base", 1.0, "rel")],
            11: [("base", "beta", 1.0, "rel")],
            12: [("poison", "target", 1.0, "rel")],
        },
        seed_entities={"alpha"},
    )

    assert a_e == {("alpha", "base"), ("base", "beta")}
    assert doc_covers_e[0] == {("alpha", "base")}
    assert doc_covers_e[1] == {("base", "beta")}
    assert doc_covers_e[2] == set()


def test_build_doc_cover_maps_for_atom_source_supports_baseline_anchored_edges():
    _, a_e, a_b, _, doc_covers_e, doc_covers_b = build_doc_cover_maps_for_atom_source(
        candidate_positions=[0, 1, 2, 3],
        atom_source_positions=[0, 1],
        pool_doc_ids=[10, 11, 12, 13],
        doc_idx_to_entities={
            10: {"alpha", "base"},
            11: {"base", "beta"},
            12: {"beta", "gamma"},
            13: {"poison", "target"},
        },
        doc_idx_to_edges={
            10: [("alpha", "base", 1.0, "rel")],
            11: [("base", "beta", 1.0, "rel")],
            12: [("beta", "gamma", 1.0, "rel")],
            13: [("poison", "target", 1.0, "rel")],
        },
        seed_entities={"alpha"},
        atom_source_mode="baseline_anchored",
    )

    assert a_e == {("alpha", "base"), ("base", "beta"), ("beta", "gamma")}
    assert a_b == {"base", "beta", "gamma"}
    assert doc_covers_e[2] == {("beta", "gamma")}
    assert doc_covers_e[3] == set()
    assert doc_covers_b[2] == {"beta", "gamma"}


def test_compute_selected_appended_doc_rows_uses_unique_frozen_edge_gain():
    pool_docs = [
        "Base A\nbody",
        "Base B\nbody",
        "Append Good\nbody",
        "Append Redundant\nbody",
    ]
    doc_covers_e = {
        0: {("a", "b")},
        1: {("b", "c")},
        2: {("c", "d")},
        3: {("b", "c")},
    }
    edge_support_counts = compute_edge_support_counts([0, 1, 2, 3], doc_covers_e)

    rows = compute_selected_appended_doc_rows(
        selected_positions=[0, 1, 2, 3],
        appended_positions=[2, 3],
        pool_docs=pool_docs,
        doc_covers_e=doc_covers_e,
        edge_support_counts=edge_support_counts,
    )

    assert [row["title"] for row in rows] == ["Append Good", "Append Redundant"]
    assert rows[0]["unique_covE_gain"] == 1
    assert rows[0]["has_nonzero_unique_covE_gain"] is True
    assert rows[0]["unique_edge_support_counts"] == [1]
    assert rows[0]["unique_frozen_covE_gain"] == 1
    assert rows[0]["has_nonzero_unique_frozen_covE_gain"] is True
    assert rows[0]["unique_frozen_edge_support_counts"] == [1]
    assert rows[1]["unique_covE_gain"] == 0
    assert rows[1]["has_nonzero_unique_covE_gain"] is False
    assert rows[1]["covered_edge_support_counts"] == [2]
    assert rows[1]["unique_frozen_covE_gain"] == 0
    assert rows[1]["has_nonzero_unique_frozen_covE_gain"] is False
    assert rows[1]["covered_frozen_edge_support_counts"] == [2]


def test_classify_route_prefers_continuation_only_when_both_signals_are_strong():
    route_a = classify_route(selected_appended_query_rate=0.35, nonzero_gain_doc_rate=0.70)
    route_b = classify_route(selected_appended_query_rate=0.10, nonzero_gain_doc_rate=0.90)

    assert route_a["route"] == "A"
    assert "continuing" in route_a["decision"]
    assert route_b["route"] == "B"
    assert "interface" in route_b["decision"]


def test_summarize_admissibility_metrics_aggregates_budget_gap_rows():
    metrics = summarize_admissibility_metrics(
        [
            {
                "coverage_admissibility_mode": "budget_gap",
                "admissible_appended_count": 2,
                "num_appended_selected_v2": 1,
                "selected_appended_from_admissible_count": 1,
                "admissibility_reference_gap_edge_count": 3,
                "admissibility_rows": [
                    {
                        "pool_position": 3,
                        "kept": True,
                        "marginal_gap_edge_count": 2,
                        "marginal_gap_edge_support_mean": 1.5,
                        "marginal_gap_edge_support_max": 2,
                    },
                    {
                        "pool_position": 4,
                        "kept": False,
                        "marginal_gap_edge_count": 0,
                        "marginal_gap_edge_support_mean": 0.0,
                        "marginal_gap_edge_support_max": 0,
                    },
                ],
            },
            {
                "coverage_admissibility_mode": "budget_gap",
                "admissible_appended_count": 0,
                "num_appended_selected_v2": 0,
                "selected_appended_from_admissible_count": 0,
                "admissibility_reference_gap_edge_count": 1,
                "admissibility_rows": [
                    {
                        "pool_position": 5,
                        "kept": True,
                        "marginal_gap_edge_count": 1,
                        "marginal_gap_edge_support_mean": 3.0,
                        "marginal_gap_edge_support_max": 3,
                    }
                ],
            },
        ]
    )

    assert metrics["admissible_appended_query_rate_target"] == 0.5
    assert metrics["avg_num_admissible_appended_target"] == 1.0
    assert metrics["selected_from_admissible_doc_rate_target"] == 1.0
    assert metrics["avg_marginal_gap_edge_count_kept"] == 1.5
    assert metrics["avg_marginal_gap_edge_support_mean_kept"] == 2.25
    assert metrics["avg_marginal_gap_edge_support_max_kept"] == 2.5
    assert metrics["avg_reference_gap_edge_count_target"] == 2.0


def test_summarize_admissibility_metrics_is_backward_compatible_without_rows():
    metrics = summarize_admissibility_metrics(
        [
            {
                "coverage_admissibility_mode": "off",
                "num_appended_selected_v2": 2,
            }
        ]
    )

    assert metrics["admissible_appended_query_rate_target"] == 0.0
    assert metrics["avg_num_admissible_appended_target"] == 0.0
    assert metrics["selected_from_admissible_doc_rate_target"] == 0.0
    assert metrics["avg_marginal_gap_edge_count_kept"] == 0.0
    assert metrics["avg_reference_gap_edge_count_target"] == 0.0
