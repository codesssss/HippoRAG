#!/usr/bin/env python3
"""Plot dense-missed fact-linked support recovery diagnostics.

The figure visualizes the mechanism claim behind EvLink:
fine-grained graph states are not sufficient by themselves, and generic
document links are not enough without reliable source-grounded evidence links.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    ROOT
    / "run_logs"
    / "evidencelink_fcrg_hard_dense_miss_top5_20260520"
    / "fcrg_aggregate.csv"
)
OUT_DIR = ROOT / "paper" / "figures"


METHOD_LABELS = {
    "HippoRAG2 pool": "HippoRAG2\npool",
    "PropRAG pool": "PropRAG\npool",
    "Degree-matched shuffled transitions delivered": "Shuffled\nlinks",
    "Edge-count matched dense transitions delivered": "Edge-count\nDense-doc",
    "Dense-doc KNN transitions delivered": "Dense-doc\nKNN",
    "w/o evidence-linked transitions delivered": "w/o evidence\nlinks",
    "EvidenceLink delivered": "EvLink",
}


def load_recovery() -> dict[str, float]:
    rows: dict[str, float] = {}
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = str(row["method"])
            rows[method] = 100.0 * float(row["method_at5_pct"])
    if "EvidenceLink delivered" not in rows and "EvLink delivered" in rows:
        rows["EvidenceLink delivered"] = rows["EvLink delivered"]
    return rows


def add_values(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        value = float(bar.get_width())
        ax.text(
            value + 1.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=7.8,
        )


def draw_panel(
    ax: plt.Axes,
    methods: list[str],
    values: dict[str, float],
    colors: list[str],
    hatches: list[str],
    ylabel: str | None = None,
) -> None:
    y = [values[m] for m in methods]
    bars = ax.barh(
        range(len(methods)),
        y,
        color=colors,
        edgecolor="#333333",
        linewidth=0.7,
        height=0.58,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    add_values(ax, list(bars))
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[m].replace("\n", " ") for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 76)
    ax.set_xticks([0, 20, 40, 60])
    if ylabel:
        ax.set_xlabel(ylabel)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    values = load_recovery()
    evidence_color = "#0072B2"
    fine_color = "#BDBDBD"
    dense_color = "#D9D9D9"
    weak_color = "#F0F0F0"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)

    left_methods = [
        "HippoRAG2 pool",
        "PropRAG pool",
        "EvidenceLink delivered",
    ]
    draw_panel(
        axes[0],
        left_methods,
        values,
        colors=[fine_color, fine_color, evidence_color],
        hatches=["///", "///", ""],
        ylabel="Recovered dense-missed fact-linked gold docs (%)",
    )
    axes[0].set_title("(a) Fine-grained graph-state baselines", loc="left", fontsize=8.5)

    right_methods = [
        "Degree-matched shuffled transitions delivered",
        "Edge-count matched dense transitions delivered",
        "Dense-doc KNN transitions delivered",
        "w/o evidence-linked transitions delivered",
        "EvidenceLink delivered",
    ]
    draw_panel(
        axes[1],
        right_methods,
        values,
        colors=[weak_color, dense_color, dense_color, "#BDBDBD", evidence_color],
        hatches=["xx", "\\\\", "\\\\", "...", ""],
    )
    axes[1].set_title("(b) Document links without reliable evidence", loc="left", fontsize=8.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=fine_color, edgecolor="#333333", hatch="///", label="Fine-grained graph baseline"),
        plt.Rectangle((0, 0), 1, 1, facecolor=dense_color, edgecolor="#333333", hatch="\\\\", label="Document graph without evidence links"),
        plt.Rectangle((0, 0), 1, 1, facecolor=evidence_color, edgecolor="#333333", label="EvLink"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=2.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUT_DIR / f"fact_linked_bridge_recovery.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print(path)


if __name__ == "__main__":
    main()
