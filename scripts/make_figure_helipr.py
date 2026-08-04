#!/usr/bin/env python3
"""Figures for HeLiPR Layer 2/3 results."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS = {
    "forward": "#888888",
    "central": "#1f77b4",
    "cubic_global": "#2ca02c",
    "smoothing_spline": "#9467bd",
    "savgol_w7p3": "#ff7f0e",
    "family_a_W3": "#ffbb78",
    "family_a_W5": "#d62728",
    "family_a_W7": "#8c564b",
    "family_a_midspan_W5": "#e377c2",
}

MARKERS = {m: o for m, o in zip(COLORS.keys(), "oPvs*Dd^h")}


def fig_bar_m8(df, outdir):
    seqs = sorted(df.sequence.unique())
    piv = df.pivot(index="method", columns="sequence", values="m8_rmse_vs_ins")
    piv = piv.loc[sorted(piv.index, key=lambda m: piv.loc[m].median())]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    xs = np.arange(len(piv.index))
    width = 0.13
    for i, sq in enumerate(seqs):
        ax.bar(xs + (i - (len(seqs) - 1) / 2) * width, piv[sq].values, width, label=sq)
    ax.set_xticks(xs)
    ax.set_xticklabels(piv.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("RMSE vs INS horizontal speed [m/s]")
    ax.set_yscale("log")
    ax.set_title("HeLiPR Layer 3: alignment with independent INS velocity (lower is better)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig_helipr_m8_bar.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_helipr_m8_bar.png", dpi=160, bbox_inches="tight")


def fig_pareto_smoothness(df, outdir):
    # Median over sequences per method.
    agg = df.groupby("method").agg(
        smooth=("m4_smooth", "median"),
        m8=("m8_rmse_vs_ins", "median"),
        m3=("m3_low_speed_rms", "median"),
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for m, row in agg.iterrows():
        ax.scatter(row["smooth"], row["m8"], color=COLORS.get(m, "k"), marker=MARKERS.get(m, "o"),
                   s=85, edgecolor="k", linewidth=0.5, label=m)
    for m, row in agg.iterrows():
        ax.annotate(m, (row["smooth"], row["m8"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(r"smoothness (RMS of $\Delta^2 \hat v$)")
    ax.set_ylabel("RMSE vs INS [m/s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("HeLiPR: smoothness vs INS-alignment (bottom-left = better)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_helipr_pareto.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_helipr_pareto.png", dpi=160, bbox_inches="tight")


def fig_timeseries(outdir, per_frame_path, t0=60.0, span=30.0):
    df = pd.read_parquet(per_frame_path)
    mask = (df.t >= t0) & (df.t <= t0 + span)
    if mask.sum() < 10:
        print("insufficient data for timeseries window")
        return
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df.t[mask], df.v_ins[mask], "k-", lw=1.2, label="INS (inspva, independent)", alpha=0.8)
    for m in ["v_central", "v_smoothing_spline", "v_family_a_W5"]:
        if m in df.columns:
            ax.plot(df.t[mask], df[m][mask], lw=1.0, label=m.replace("v_", ""), alpha=0.85,
                    color=COLORS.get(m.replace("v_", ""), None))
    ax.set_xlabel("time [s] (since pose start)")
    ax.set_ylabel("speed [m/s]")
    ax.set_title(f"HeLiPR {per_frame_path.stem.replace('per_frame_', '')}: "
                 f"pose-derived speed vs INS reference (window {t0:.0f}s..+{span:.0f}s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    name = per_frame_path.stem.replace("per_frame_", "")
    fig.savefig(outdir / f"fig_helipr_ts_{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"fig_helipr_ts_{name}.png", dpi=160, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/helipr"))
    ap.add_argument("--figout", type=Path, default=Path("results/helipr/figures"))
    args = ap.parse_args()
    args.figout.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results / "summary.csv")
    fig_bar_m8(df, args.figout)
    fig_pareto_smoothness(df, args.figout)
    for pf in sorted(args.results.glob("per_frame_*.parquet")):
        fig_timeseries(args.figout, pf, t0=60.0, span=30.0)
    print(f"Figures → {args.figout}")


if __name__ == "__main__":
    main()
