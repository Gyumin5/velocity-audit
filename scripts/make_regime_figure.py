#!/usr/bin/env python3
"""Regime overview figure (horizontal-bar, no legend overlap).

Layout rules (per ai-debate consensus):
  - generous left padding so dataset names never touch bars
  - value labels INSIDE the bar (right-aligned for positive, left-aligned for negative)
  - short bars get the label outside the bar tip (clamped within xlim)
  - observed-range band label sits at the top of panel (b), not in the bar grid
  - regime meaning is documented in the caption, not in an inline legend
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt


# Every number plotted here is read from the committed audit outputs; nothing is
# transcribed by hand, so the figure cannot drift out of step with the tables:
#   seven main releases  <- results/crossds_recomputed.csv  (delta_pct)
#                           results/multimetric_recomputed.csv (M3_c / M3_f)
#   KAIST CU boundary    <- results/kaist_cu/per_sequence.csv
#   pose-only checks     <- results/{av2,durlar,ncd}/per_sequence.csv
# The smoothness convention is the ratio of per-dataset medians, matching the
# paper's tables; the shaded band is the observed range over the seven releases.
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

REGIME_OF = {
    "HeLiPR": "separated", "Oxford": "alg. separated",
    "nuScenes": "weak", "KITTI": "online", "KITTI-360": "online",
    "Boreas": "batch", "Pit30M": "batch",
}
DISPLAY_NAME = {"KITTI": "KITTI raw"}


def _ratio_of_medians(path: Path, num: str, den: str) -> float:
    d = pd.read_csv(path)
    return float(d[num].median() / d[den].median())


def _build_datasets():
    cross = pd.read_csv(RESULTS / "crossds_recomputed.csv").set_index("dataset")
    multi = pd.read_csv(RESULTS / "multimetric_recomputed.csv").set_index("dataset")
    rows = []
    for name, regime in REGIME_OF.items():
        rows.append((DISPLAY_NAME.get(name, name), regime,
                     float(cross.loc[name, "delta_pct"]),
                     float(multi.loc[name, "M3_c"] / multi.loc[name, "M3_f"])))
    main_ratios = [r[3] for r in rows]

    kaist = RESULTS / "kaist_cu" / "per_sequence.csv"
    rows.append(("KAIST CU", "wheel-encoder",
                 (_ratio_of_medians(kaist, "family_a_W5_M4", "central_M4") - 1.0) * 100.0,
                 _ratio_of_medians(kaist, "central_M3sm", "family_a_W5_M3sm")))
    for label, sub in (("AV2", "av2"), ("DurLAR", "durlar"), ("NCD", "ncd")):
        rows.append((label, "pose-only", None,
                     _ratio_of_medians(RESULTS / sub / "per_sequence.csv",
                                       "central_M3sm", "family_a_W5_M3sm")))
    return rows, (min(main_ratios), max(main_ratios))


DATASETS, SMOOTH_BAND = _build_datasets()

REGIME_COLOR = {
    "separated":      "#2a8a3a",
    "alg. separated": "#3aa050",
    "weak":           "#4c78c8",
    "online":         "#7a8acb",
    "batch":          "#c84a4a",
    "wheel-encoder":  "#d09020",
    "pose-only":      "#8a8a8a",
}


def _label_right_of_bar(ax, value, y, text, xlim, font=8.0):
    """Place every value label to the right side of its bar:
    - positive bars: just right of the bar tip
    - negative bars: just right of the zero line (where the bar ends going right)
    This keeps all labels visually grouped in the right-of-zero region and
    avoids the leftmost negative labels colliding with the y-axis frame.
    """
    xmin, xmax = xlim
    pad = 0.02 * (xmax - xmin)
    x = max(value, 0.0) + pad
    ax.text(x, y, text, va="center", ha="left",
            fontsize=font, fontweight="bold", color="black")


def main():
    out = Path("/home/gmoh/av-ros/test/velocity/paper/figures/fig_regime_overview.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Wide two-column figure: more horizontal room → labels outside bars stay readable
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.5, 4.5),
                                    gridspec_kw={"height_ratios": [0.8, 1.0]})

    # ---- (a) M_4 change vs central ----
    # Sign-color encoding (cycle 24): blue=Δ<-6% (aligns closer), gray=|Δ|≤6% (floor), orange=Δ>+6% (anti-align)
    # KAIST CU (wheel-encoder, reference-physics boundary) and near-floor entries also get hatch for grayscale separation.
    SIGN_BLUE  = "#3a78c8"  # alignment direction
    SIGN_GRAY  = "#9a9a9a"  # near-floor / no method effect
    SIGN_ORANGE= "#e07a2e"  # anti-alignment direction
    def _sign_color(regime, delta):
        if regime == "wheel-encoder":
            return (SIGN_GRAY, "///")          # reference-physics boundary: hatched
        if delta is None: return (SIGN_GRAY, None)
        if abs(delta) <= 6: return (SIGN_GRAY, "....")   # near-floor: dotted hatch
        if delta < 0: return (SIGN_BLUE, None)
        return (SIGN_ORANGE, None)
    a_data = [d for d in DATASETS if d[2] is not None]
    y_a = list(range(len(a_data)))
    xlim_a = (-80, 175)
    for i, (_, regime, delta, _) in enumerate(a_data):
        color, hatch = _sign_color(regime, delta)
        axA.barh(i, delta, color=color, edgecolor="k", linewidth=0.4, height=0.66,
                 hatch=hatch)
        lbl = f"{round(delta):+.0f}%" if round(delta) != 0 else "0%"
        _label_right_of_bar(axA, delta, i, lbl, xlim_a, font=9.0)
    axA.axvline(0, color="0.15", lw=1.2, zorder=3)
    # Coupling-band separator lines + direct group labels (left of bars)
    # Group boundaries (after each contiguous regime cluster in a_data):
    #   0-1: separated/alg.sep    (HeLiPR, Oxford)
    #   2-4: weak/online          (nuScenes, KITTI, KITTI-360)
    #   5-6: batch joint          (Boreas, Pit30M)
    #   7:   wheel-encoder        (KAIST CU)
    for ysep in (1.5, 4.5, 6.5):
        axA.axhline(ysep, color="0.4", lw=0.7, ls=(0, (3, 2)), zorder=1)
    group_labels = [
        (0.5,  "sep."),
        (3.0,  "online"),
        (5.5,  "batch"),
        (7.0,  "wheel"),
    ]
    for ypos, txt in group_labels:
        axA.text(-78, ypos, txt, fontsize=7.5, color="0.25",
                 ha="left", va="center", style="italic", zorder=4)
    axA.set_yticks(y_a)
    axA.set_yticklabels([d[0] for d in a_data], fontsize=9.0)
    axA.invert_yaxis()
    axA.set_xlim(*xlim_a)
    axA.set_xlabel(r"$M_4$ change vs central (%)", fontsize=9.0, labelpad=2)
    axA.tick_params(axis="x", labelsize=8.0)
    axA.tick_params(axis="y", labelsize=9.0)
    axA.grid(True, axis="x", lw=0.3, ls=":", color="0.8")
    axA.set_title("(a) Alignment vs published velocity", fontsize=9.5, pad=4)

    # ---- (b) M_3^sm ratio ----
    y_b = list(range(len(DATASETS)))
    xlim_b = (0, 6.0)
    band_lo, band_hi = (round(v, 1) for v in SMOOTH_BAND)
    axB.axvspan(band_lo, band_hi, color=(0.90, 0.95, 0.90), zorder=0)
    for i, (_, regime, _, ratio) in enumerate(DATASETS):
        color = REGIME_COLOR[regime]
        axB.barh(i, ratio, color=color, edgecolor="k", linewidth=0.4, height=0.66,
                 zorder=2)
        _label_right_of_bar(axB, ratio, i, f"{ratio:.1f}", xlim_b, font=9.0)
    axB.text(2.95, -0.85, rf"${band_lo}$–${band_hi}\times$ observed range", fontsize=8.0, color="0.25",
             ha="center", va="center", fontweight="bold")
    axB.set_yticks(y_b)
    axB.set_yticklabels([d[0] for d in DATASETS], fontsize=9.0)
    axB.invert_yaxis()
    axB.set_xlim(*xlim_b)
    axB.set_ylim(len(DATASETS) - 0.5, -1.5)
    axB.set_xlabel(r"$M_3^{\mathrm{sm}}$ ratio (central / Family A)",
                   fontsize=9.0, labelpad=2)
    axB.tick_params(axis="x", labelsize=8.0)
    axB.tick_params(axis="y", labelsize=9.0)
    axB.grid(True, axis="x", lw=0.3, ls=":", color="0.8")
    axB.set_title("(b) Smoothness gain", fontsize=9.5, pad=4)

    fig.subplots_adjust(left=0.22, right=0.96, top=0.94, bottom=0.10, hspace=0.55)
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
