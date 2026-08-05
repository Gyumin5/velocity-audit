#!/usr/bin/env python3
"""Figure 1: what each stage of the audit is allowed to consult.

The point of the two-stage design is a barrier, so the figure is built around
one: everything that fixes the probe sits above the line, every published
velocity sits below it, and nothing crosses upward. A reader who takes only the
figure should come away knowing that the probe was not tuned on the thing it
later measures.

The release count is read from the committed outputs rather than typed, so the
figure cannot drift out of step with the tables the way its predecessor did.
Protocol constants (W, p, window duration) are definitions, not measurements,
and are stated directly.

Writes paper/figures/fig_two_stage.pdf.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "paper" / "figures" / "fig_two_stage.pdf"

STAGE1_FILL, STAGE1_EDGE = "#eef4fb", "#3b6ea5"
STAGE2_FILL, STAGE2_EDGE = "#fdf2e9", "#c1682a"
PROBE_FILL, PROBE_EDGE = "#ffffff", "#2f2f2f"


def box(ax, x, y, w, h, text, fill, edge, size=6.4, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=0.9, facecolor=fill, edgecolor=edge,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, xy_from, xy_to, color="#2f2f2f"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.9,
                                 color=color, zorder=4,
                                 shrinkA=0, shrinkB=0))


def main() -> None:
    n_releases = len(pd.read_csv(RESULTS / "crossds_recomputed.csv"))

    fig = plt.figure(figsize=(3.45, 3.7))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Stage 1: truth known -------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.015, 0.615), 0.97, 0.365,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=0, facecolor=STAGE1_FILL, zorder=0))
    ax.text(0.035, 0.945, "STAGE 1", fontsize=6.6, weight="bold",
            color=STAGE1_EDGE, va="center")
    ax.text(0.185, 0.945, "\u00b7  truth known", fontsize=6.0,
            color=STAGE1_EDGE, va="center")

    box(ax, 0.045, 0.775, 0.43, 0.125,
        "closed-form\nvariance ratio\n(window geometry)", "#ffffff", STAGE1_EDGE)
    box(ax, 0.525, 0.775, 0.43, 0.125,
        "synthetic sweep\nagainst an\nanalytic speed", "#ffffff", STAGE1_EDGE)

    arrow(ax, (0.26, 0.775), (0.36, 0.725))
    arrow(ax, (0.74, 0.775), (0.64, 0.725))

    box(ax, 0.075, 0.628, 0.85, 0.095,
        "fixed probe: local LSQ polynomial\n$W{=}5$,  $p{=}3$,  1.0 s window",
        PROBE_FILL, PROBE_EDGE, size=6.0, weight="bold")

    # --- the barrier ----------------------------------------------------------
    ax.plot([0.015, 0.985], [0.588, 0.588], linestyle=(0, (4, 3)),
            linewidth=1.0, color="#8a8a8a", zorder=1)
    ax.text(0.5, 0.588, "  no published velocity consulted above this line  ",
            fontsize=6.0, style="italic", color="#5a5a5a",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none"), zorder=5)

    # --- Stage 2: truth unknown ----------------------------------------------
    ax.add_patch(FancyBboxPatch((0.015, 0.02), 0.97, 0.545,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=0, facecolor=STAGE2_FILL, zorder=0))
    ax.text(0.035, 0.532, "STAGE 2", fontsize=6.6, weight="bold",
            color=STAGE2_EDGE, va="center")
    ax.text(0.185, 0.532, "\u00b7  truth unknown", fontsize=6.0,
            color=STAGE2_EDGE, va="center")

    arrow(ax, (0.5, 0.628), (0.5, 0.489), color="#2f2f2f")

    box(ax, 0.075, 0.372, 0.85, 0.105,
        f"apply unchanged, with no per-dataset retuning,\nto {n_releases} public releases",
        "#ffffff", STAGE2_EDGE, size=6.0)
    arrow(ax, (0.5, 0.372), (0.5, 0.325))

    box(ax, 0.075, 0.218, 0.85, 0.105,
        "residual against each published velocity\nread as a measurement "
        "$\\it{of\\ the\\ release}$", "#ffffff", STAGE2_EDGE, size=6.0)
    arrow(ax, (0.29, 0.218), (0.24, 0.168))
    arrow(ax, (0.71, 0.218), (0.76, 0.168))

    box(ax, 0.045, 0.048, 0.43, 0.12,
        "coupling axis\nseparated / weak\n/ batch-joint", "#ffffff", "#7a7a7a")
    box(ax, 0.525, 0.048, 0.43, 0.12,
        "reference physics\ninertial /\nnon-inertial", "#ffffff", "#7a7a7a")

    ax.text(0.5, 0.012, "both assigned from documentation, before the probe runs",
            fontsize=5.9, style="italic", color="#5a5a5a", ha="center", va="center")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}  ({n_releases} releases from CSV)")


if __name__ == "__main__":
    main()
