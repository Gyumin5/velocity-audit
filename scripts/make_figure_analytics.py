#!/usr/bin/env python3
"""Figures for the extended HeLiPR analytics (sensitivity / curvature / L1.8 / M9)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    "central": "#1f77b4",
    "savgol_w7p3": "#ff7f0e",
    "family_a_W5": "#d62728",
    "family_a_W7": "#8c564b",
    "smoothing_spline_default": "#9467bd",
    "smoothing_spline_tuned": "#17becf",
}


def fig_sensitivity_heatmap(df, outdir):
    piv = df.pivot(index="W", columns="degree", values="rmse_vs_ins")
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("half-window size W")
    ax.set_title("Family A sensitivity on HeLiPR roundabout01\nRMSE vs INS [m/s]")
    fig.colorbar(im, ax=ax, label="RMSE [m/s]")
    # Annotate cells.
    for (i, j), v in np.ndenumerate(piv.values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > np.nanmedian(piv.values) else "black",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig_sensitivity_heatmap.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_sensitivity_heatmap.png", dpi=160, bbox_inches="tight")


def fig_curvature_bins(df, outdir):
    # Sort bin label by lower bound.
    def lower_edge(lbl):
        # labels look like '1e-03-5e-03' — find the central '-' separator (after the e-XX of lower bound).
        # Simpler: parse with regex.
        import re
        m = re.match(r"^([\d.eE+\-]+?)-([\d.eE+\-]+)$", lbl.replace("e-", "eN").replace("e+", "eP"))
        if m:
            return float(m.group(1).replace("eN", "e-").replace("eP", "e+"))
        return 0.0
    bins = sorted(df.kappa_bin.unique(), key=lower_edge)
    methods = ["central", "savgol_w7p3", "family_a_W5"]
    med = df.groupby(["kappa_bin", "method"])["rmse_vs_ins"].median().unstack().reindex(bins)
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    xs = np.arange(len(bins))
    w = 0.26
    for i, m in enumerate(methods):
        if m in med.columns:
            ax.bar(xs + (i - 1) * w, med[m].values, w, label=m, color=COLORS.get(m))
    ax.set_xticks(xs)
    ax.set_xticklabels(bins, rotation=20, ha="right", fontsize=8)
    ax.set_xlabel(r"$|\kappa|$ bin [1/m]")
    ax.set_ylabel("RMSE vs INS [m/s] (median over 6 seqs)")
    ax.set_title("HeLiPR: curvature-binned speed reconstruction error")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_curvature_bins.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_curvature_bins.png", dpi=160, bbox_inches="tight")


def fig_l18_degradation(df, outdir):
    # Fix: only noise=0, jitter=0 subset. Show RMSE vs effective rate (ds_factor).
    sub = df[(df.noise_sigma == 0.0) & (df.jitter_sigma == 0.0)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    # Left: RMSE vs ds_factor at sigma=0.
    for m, g in sub.groupby("method"):
        g = g.sort_values("ds_factor")
        axes[0].plot(g["ds_factor"], g["rmse"], marker="o", label=m, color=COLORS.get(m))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("downsample factor (native 10 Hz / factor)")
    axes[0].set_ylabel("RMSE vs INS [m/s]")
    axes[0].set_title(r"L1.8: clean pose, no jitter")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    # Right: RMSE vs noise_sigma at ds_factor=1, jitter=0.
    sub2 = df[(df.ds_factor == 1) & (df.jitter_sigma == 0.0)]
    for m, g in sub2.groupby("method"):
        g = g.sort_values("noise_sigma")
        axes[1].plot(g["noise_sigma"], g["rmse"], marker="o", label=m, color=COLORS.get(m))
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"injected position noise $\sigma_p$ [m]")
    axes[1].set_ylabel("RMSE vs INS [m/s]")
    axes[1].set_title("L1.8: native rate, additive noise")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle("Semi-controlled degradation of HeLiPR roundabout01 pose → reconstructed speed", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig_l18_degradation.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_l18_degradation.png", dpi=200, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/helipr"))
    ap.add_argument("--figout", type=Path, default=Path("results/helipr/figures"))
    args = ap.parse_args()
    args.figout.mkdir(parents=True, exist_ok=True)

    if (args.results / "sensitivity.csv").exists():
        fig_sensitivity_heatmap(pd.read_csv(args.results / "sensitivity.csv"), args.figout)
    if (args.results / "curvature_bins_all.csv").exists():
        fig_curvature_bins(pd.read_csv(args.results / "curvature_bins_all.csv"), args.figout)
    if (args.results / "l18_degradation.parquet").exists():
        fig_l18_degradation(pd.read_parquet(args.results / "l18_degradation.parquet"), args.figout)
    print(f"Figures → {args.figout}")


if __name__ == "__main__":
    main()
