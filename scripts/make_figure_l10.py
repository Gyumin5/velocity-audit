#!/usr/bin/env python3
"""Plot L1.0 synthetic sweep results."""
from __future__ import annotations
import argparse
from pathlib import Path
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("results/l10/results.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/l10/figures"))
    args = ap.parse_args()
    df = pd.read_parquet(args.inp)
    args.out.mkdir(parents=True, exist_ok=True)

    # F-A: RMSE vs sampling rate at sigma=0.05, jitter=0.0, per scenario.
    cond = (df.noise_sigma == 0.05) & (df.jitter_sigma == 0.0)
    sub = df[cond]
    scenarios = sorted(sub.scenario.unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.4), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, sc in zip(axes, scenarios):
        ssc = sub[sub.scenario == sc]
        med = ssc.groupby(["method", "fs"])["rmse"].median().unstack(-1)
        for m, row in med.iterrows():
            ax.plot(row.index, row.values, marker="o", label=m, color=COLORS.get(m, None))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("sampling rate fs [Hz]")
        ax.set_title(sc)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("RMSE of speed [m/s] (interior)")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.suptitle(r"L1.0 synthetic: RMSE vs fs ($\sigma_p = 0.05$ m, no jitter)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out / "fA_rmse_vs_fs.pdf", bbox_inches="tight")
    fig.savefig(args.out / "fA_rmse_vs_fs.png", dpi=160, bbox_inches="tight")

    # F-B: RMSE vs position noise at fs=10, jitter=0.
    cond2 = (df.fs == 10.0) & (df.jitter_sigma == 0.0)
    sub2 = df[cond2]
    fig2, axes2 = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.4), sharey=False)
    if len(scenarios) == 1:
        axes2 = [axes2]
    for ax, sc in zip(axes2, scenarios):
        ssc = sub2[sub2.scenario == sc]
        med = ssc.groupby(["method", "noise_sigma"])["rmse"].median().unstack(-1)
        for m, row in med.iterrows():
            ax.plot(row.index, row.values, marker="o", label=m, color=COLORS.get(m, None))
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma_p$ [m]")
        ax.set_title(sc)
        ax.grid(True, alpha=0.3)
    axes2[0].set_ylabel("RMSE of speed [m/s] (interior)")
    axes2[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig2.suptitle("L1.0 synthetic: RMSE vs position noise (fs=10 Hz)", y=1.02)
    fig2.tight_layout()
    fig2.savefig(args.out / "fB_rmse_vs_noise.pdf", bbox_inches="tight")
    fig2.savefig(args.out / "fB_rmse_vs_noise.png", dpi=160, bbox_inches="tight")
    print(f"Figures → {args.out}")


if __name__ == "__main__":
    main()
