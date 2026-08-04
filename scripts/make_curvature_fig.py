#!/usr/bin/env python3
"""Fig 1: cross-dataset curvature-binned RMSE reduction (cycle 25 redesign).

Two side-by-side panels (no broken axis, no grouped bars):
  Left  (wide): 6 driving datasets as line+marker. y-range close to data.
  Right (narrow): Boreas only. Visually preserved as the 7th audit point.

Plot-speaks-for-itself: lines show how the residual change behaves across curvature bins.
The flat-or-decreasing shape in the left panel makes the "no high-curvature
concentration" observation visible without reading the caption.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


BINS = ["low", "med-low", "med", "med-high", "high"]
BIN_LABELS = ["0e+00-1e-03", "1e-03-5e-03", "5e-03-2e-02", "2e-02-1e-01", "1e-01-1e+00"]
ORDER = ["HeLiPR", "Oxford", "nuScenes", "KITTI raw", "KITTI-360", "Pit30M", "Boreas"]


def load_data(csv_path):
    """Per-bin residual reduction (%) of Family A W=5 vs central differencing.

    Read from results/curvature_bins_all_datasets.csv, which is produced by
    scripts/curvature_bins_all_datasets.py from the committed per-frame files.
    Aggregation matches the rest of the paper: median over sequences per
    method, then the reduction of those medians.
    """
    import csv as _csv
    from statistics import median
    acc = {}
    with open(csv_path) as fh:
        for r in _csv.DictReader(fh):
            acc.setdefault((r["dataset"], r["kappa_bin"], r["method"]), []).append(
                float(r["rmse_vs_ref"]))
    out = {}
    for ds in ORDER:
        row = []
        for b in BIN_LABELS:
            c = acc.get((ds, b, "central"))
            f = acc.get((ds, b, "family_a_W5"))
            row.append(float("nan") if not c or not f
                       else (median(c) - median(f)) / median(c) * 100.0)
        out[ds] = row
    return out


# Grayscale-safe: every dataset has a distinct (marker, linestyle).
STYLE = {
    "HeLiPR":    dict(color="#1f77b4", marker="o", linestyle="-",  linewidth=1.4, markersize=5.0),
    "Oxford":    dict(color="#2ca02c", marker="s", linestyle="--", linewidth=1.2, markersize=4.8),
    "nuScenes":  dict(color="#9467bd", marker="^", linestyle="-.", linewidth=1.1, markersize=5.2),
    "KITTI raw": dict(color="#8c564b", marker="D", linestyle=":",  linewidth=1.1, markersize=4.4),
    "KITTI-360": dict(color="#e377c2", marker="v", linestyle="-",  linewidth=1.1, markersize=4.8),
    "Pit30M":    dict(color="#7f7f7f", marker="x", linestyle="--", linewidth=1.1, markersize=5.0),
    "Boreas":    dict(color="#d62728", marker="o", linestyle="-",  linewidth=1.4, markersize=5.2),
}


def main():
    repo = REPO_ROOT
    DATA = load_data(repo / "results" / "curvature_bins_all_datasets.csv")
    out = repo / "paper" / "figures" / "fig_curvature_bins.pdf"
    out_png = out.with_suffix(".png")

    fig, (ax_main, ax_bor) = plt.subplots(
        1, 2, figsize=(6.0, 3.0),
        gridspec_kw={"width_ratios": [4.5, 1.2], "wspace": 0.32},
    )

    xs = np.arange(len(BINS))

    main_keys = ["HeLiPR", "Oxford", "nuScenes", "KITTI raw", "KITTI-360", "Pit30M"]
    for ds in main_keys:
        ax_main.plot(xs, DATA[ds], label=ds, **STYLE[ds])

    ax_main.axhline(0, color="0.45", lw=0.6)
    ax_main.set_ylim(-10, 50)
    ax_main.set_xticks(xs)
    ax_main.set_xticklabels(BINS, fontsize=8)
    ax_main.set_xlabel(r"curvature $|\kappa|$ bin", fontsize=9)
    ax_main.set_ylabel("RMSE reduction vs central (%)\n(positive = closer to published)", fontsize=8)
    ax_main.tick_params(axis="y", labelsize=8)
    ax_main.grid(True, axis="y", lw=0.3, ls=":", color="0.7")
    ax_main.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.92,
                   handlelength=2.2, columnspacing=0.9)
    ax_main.set_title("6 audited datasets", fontsize=9, pad=2)

    # Boreas only on the right panel
    ax_bor.plot(xs, DATA["Boreas"], **STYLE["Boreas"])
    ax_bor.axhline(0, color="0.45", lw=0.6)
    ax_bor.set_ylim(-130, 10)
    ax_bor.set_xticks(xs)
    ax_bor.set_xticklabels(BINS, fontsize=7, rotation=45)
    ax_bor.tick_params(axis="y", labelsize=8)
    ax_bor.grid(True, axis="y", lw=0.3, ls=":", color="0.7")
    ax_bor.set_title("Boreas (batch)", fontsize=9, pad=2)
    ax_bor.set_xlabel(r"$|\kappa|$ bin", fontsize=8)

    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out} and {out_png}")


if __name__ == "__main__":
    main()
