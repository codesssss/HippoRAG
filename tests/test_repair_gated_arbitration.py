from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyze_repair_gated_arbitration_offline import (  # noqa: E402
    BranchContext,
    RepairCandidate,
    VariantConfig,
    repair_gated_order,
    support_match,
)
from build_repair_gated_pools import select_variants  # noqa: E402


def candidate(position: int, title: str, gain: float) -> RepairCandidate:
    return RepairCandidate(
        position=position,
        title=title,
        gain=gain,
        step=position,
        mode="test",
        coverage_by_requirement={},
    )


def test_repair_gated_order_keeps_rank_quota_after_accepting_repair() -> None:
    pool_titles = ["Seed", "DBEC Gold", "Rank Gold", "Noise"]
    order, trace = repair_gated_order(
        seed_positions=[0],
        pool_titles=pool_titles,
        dbec_candidates=[candidate(1, "DBEC Gold", 0.2)],
        config=VariantConfig(rank_quota=1, max_repairs=1, min_gain=0.05),
        top_k=3,
    )

    assert order == [0, 1, 2]
    assert trace["accepted_repair_count"] == 1
    assert trace["rank_fill_count"] == 1
    assert trace["accepted_repairs"][0]["title"] == "DBEC Gold"


def test_repair_gated_order_forces_rank_when_rank_quota_consumes_fill_budget() -> None:
    pool_titles = ["Seed", "Rank A", "Rank B", "DBEC Candidate"]
    order, trace = repair_gated_order(
        seed_positions=[0],
        pool_titles=pool_titles,
        dbec_candidates=[candidate(3, "DBEC Candidate", 1.0)],
        config=VariantConfig(rank_quota=2, max_repairs=1, min_gain=0.0),
        top_k=3,
    )

    assert order == [0, 1, 2]
    assert trace["accepted_repair_count"] == 0
    assert trace["rank_fill_count"] == 2


def test_repair_gated_order_rejects_duplicate_title_repair() -> None:
    pool_titles = ["Anchor", "Anchor", "Fallback"]
    order, trace = repair_gated_order(
        seed_positions=[0],
        pool_titles=pool_titles,
        dbec_candidates=[candidate(1, "Anchor", 1.0)],
        config=VariantConfig(rank_quota=0, max_repairs=1, min_gain=0.0),
        top_k=2,
    )

    assert order == [0, 1]
    assert trace["accepted_repair_count"] == 0
    assert trace["rejection_counts"] == {"duplicate_title": 1}


def test_branch_strict_rejects_sibling_and_accepts_positive_anchor() -> None:
    pool_titles = [
        "Seed",
        "Royal Navy destroyer classes",
        "United States Navy destroyer classes",
    ]
    order, trace = repair_gated_order(
        seed_positions=[0],
        pool_titles=pool_titles,
        dbec_candidates=[
            candidate(1, "Royal Navy destroyer classes", 1.0),
            candidate(2, "United States Navy destroyer classes", 1.0),
        ],
        config=VariantConfig(rank_quota=0, max_repairs=1, min_gain=0.0, branch_policy="strict"),
        branch_context=BranchContext(
            positive_anchors=("united states navy",),
            sibling_anchors=("royal navy",),
        ),
        top_k=2,
    )

    assert order == [0, 2]
    assert trace["accepted_repair_count"] == 1
    assert trace["branch_conflict_reject_count"] == 1
    assert trace["accepted_repairs"][0]["branch_status"] == "support"


def test_support_match_counts_complete_recall_with_normalized_titles() -> None:
    metrics = support_match(
        ["Capital punishment in New Zealand", "Death penalty"],
        ["Capital Punishment in New Zealand (article)", "Other"],
    )

    assert metrics["hit_count"] == 1
    assert metrics["missing_count"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["complete"] == 0


def test_select_variants_prefers_musique_2wiki_and_cross_dataset() -> None:
    rows = [
        {
            "dataset": "MuSiQue",
            "variant": "v_m",
            "delta_support_complete_vs_rank": "0.2",
            "delta_support_recall_vs_rank": "0.1",
            "harmful_replacement_rate": "0.0",
            "accepted_repair_count": "1.0",
            "min_gain": "0.05",
            "rank_quota": "1",
        },
        {
            "dataset": "2Wiki",
            "variant": "v_w",
            "delta_support_complete_vs_rank": "0.3",
            "delta_support_recall_vs_rank": "0.1",
            "harmful_replacement_rate": "0.0",
            "accepted_repair_count": "1.0",
            "min_gain": "0.05",
            "rank_quota": "1",
        },
        {
            "dataset": "MuSiQue",
            "variant": "v_cross",
            "delta_support_complete_vs_rank": "0.1",
            "delta_support_recall_vs_rank": "0.1",
            "harmful_replacement_rate": "0.0",
            "accepted_repair_count": "1.0",
            "min_gain": "0.10",
            "rank_quota": "1",
        },
        {
            "dataset": "2Wiki",
            "variant": "v_cross",
            "delta_support_complete_vs_rank": "0.1",
            "delta_support_recall_vs_rank": "0.1",
            "harmful_replacement_rate": "0.0",
            "accepted_repair_count": "1.0",
            "min_gain": "0.10",
            "rank_quota": "1",
        },
    ]

    selected = select_variants(rows, max_variants=3)

    assert [row["variant"] for row in selected] == ["v_m", "v_w", "v_cross"]
    assert [row["reason"] for row in selected] == [
        "best_musique",
        "best_2wiki",
        "best_cross_dataset_average",
    ]
