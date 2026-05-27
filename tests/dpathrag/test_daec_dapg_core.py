from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dpathrag.daec_dapg.binding import enumerate_binding_priors, normalize_slot_priors
from src.dpathrag.daec_dapg.graph import build_absorption_graph
from src.dpathrag.daec_dapg.metrics import noise_rate, phi_auprc
from src.dpathrag.daec_dapg.projection import (
    greedy_noisy_or_select,
    noisy_or_coverage,
    select_best_binding,
)
from src.dpathrag.daec_dapg.propagation import finite_horizon_absorption, support_tensor


def test_binding_priors_truncate_and_renormalize() -> None:
    bindings = enumerate_binding_priors(
        {
            "s1": {"A": 3.0, "B": 1.0},
            "s2": {"X": 4.0, "Y": 2.0},
            "s3": {"M": 5.0, "N": 1.0},
        },
        bmax=3,
        epsilon=0.0,
    )
    assert len(bindings) == 3
    assert abs(sum(binding.prior for binding in bindings) - 1.0) < 1e-9
    assert bindings[0].raw_score >= bindings[1].raw_score >= bindings[2].raw_score


def test_slot_priors_do_not_require_retriever_score() -> None:
    priors = normalize_slot_priors({"entity_a": 0.0, "entity_b": 2.0}, epsilon=1e-6)
    assert set(priors) == {"entity_a", "entity_b"}
    assert abs(sum(priors.values()) - 1.0) < 1e-9
    assert priors["entity_b"] > priors["entity_a"]


def test_absorption_graph_uses_document_sinks_and_excludes_entity_to_doc_by_default() -> None:
    graph = build_absorption_graph(
        node_types={"p": "P", "e": "E", "d": "D"},
        edges=[
            ("p", "e", 1.0),
            ("e", "p", 1.0),
            ("p", "d", 1.0),
            ("e", "d", 100.0),
            ("d", "p", 100.0),
        ],
    )
    assert graph.transient_nodes == ["e", "p"]
    assert graph.document_nodes == ["d"]
    e_row = graph.transient_nodes.index("e")
    p_row = graph.transient_nodes.index("p")
    assert graph.r[e_row, 0] == 0.0
    assert graph.r[p_row, 0] > 0.0
    assert np.allclose(graph.q.sum(axis=1) + graph.r.sum(axis=1), np.ones(2))


def test_absorption_can_enable_entity_to_doc_as_ablation() -> None:
    graph = build_absorption_graph(
        node_types={"e": "E", "d": "D"},
        edges=[("e", "d", 2.0)],
        allow_entity_to_doc=True,
    )
    assert graph.r[0, 0] == 1.0


def test_finite_horizon_absorption_and_phi_bounds() -> None:
    q = np.array([[0.0]])
    r = np.array([[1.0]])
    hit = finite_horizon_absorption(q, r, np.array([1.0]), np.array([0.0]), alpha=0.0, horizon=1)
    assert np.allclose(hit, np.array([1.0]))
    phi = support_tensor(np.array([0.25]), hit)
    assert np.allclose(phi, np.array([0.25]))
    assert np.all((0.0 <= phi) & (phi <= 1.0))


def test_noisy_or_is_monotone_and_has_diminishing_returns() -> None:
    phi = np.array(
        [
            [0.5, 0.4, 0.1],
            [0.0, 0.6, 0.2],
        ]
    )
    empty = noisy_or_coverage(phi, [])
    first = noisy_or_coverage(phi, [0])
    first_second = noisy_or_coverage(phi, [0, 1])
    first_second_third = noisy_or_coverage(phi, [0, 1, 2])
    assert empty <= first <= first_second <= first_second_third
    gain_to_small = noisy_or_coverage(phi, [2], None) - noisy_or_coverage(phi, [], None)
    gain_to_large = noisy_or_coverage(phi, [0, 1, 2], None) - noisy_or_coverage(phi, [0, 1], None)
    assert gain_to_small >= gain_to_large


def test_greedy_noisy_or_select_returns_budgeted_indices() -> None:
    phi = np.array([[0.9, 0.1, 0.8]])
    selected, objective = greedy_noisy_or_select(phi, budget=2)
    assert selected == [0, 2]
    assert objective > 0.9


def test_best_binding_uses_explicit_prior_not_phi_scaling() -> None:
    phi = np.array(
        [
            [
                [0.5, 0.0],
                [0.9, 0.0],
            ]
        ]
    )
    result = select_best_binding(
        phi,
        {"low_coverage_high_prior": 0.9, "high_coverage_low_prior": 0.1},
        ["low_coverage_high_prior", "high_coverage_low_prior"],
        budget=1,
    )
    best = result["best"]
    assert best["binding_key"] == "low_coverage_high_prior"
    assert best["objective"] == 0.5
    assert best["score"] == 0.45


def test_metrics_noise_rate_and_auprc() -> None:
    assert noise_rate(2, 5) == 0.6
    assert phi_auprc([0.9, 0.1, 0.8], [1, 0, 1]) == 1.0
