#!/usr/bin/env python3
"""Spectral / acceleration analysis across all 6 HeLiPR sequences.

Reports per-sequence values plus median+IQR aggregate.
roundabout01 is kept as the illustrative excerpt for narrative figures.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch


def accel_stats(v: np.ndarray, dt: float) -> dict:
    a = np.diff(v) / dt
    return {
        "rms_accel": float(np.sqrt(np.mean(a**2))),
        "p95_accel": float(np.percentile(np.abs(a), 95)),
        "p99_accel": float(np.percentile(np.abs(a), 99)),
    }


def psd_bands(v: np.ndarray, fs: float, nperseg: int) -> dict:
    f, P = welch(v, fs=fs, nperseg=nperseg)
    low = float(np.trapezoid(P[f < 0.5], f[f < 0.5]))
    mid = float(np.trapezoid(P[(f >= 0.5) & (f < 2.0)], f[(f >= 0.5) & (f < 2.0)]))
    high = float(np.trapezoid(P[f >= 2.0], f[f >= 2.0]))
    return {"psd_low": low, "psd_mid": mid, "psd_high": high}


METHODS = {
    "INS_ref": ["v_ins", "v_ins_horiz"],
    "central": ["v_central"],
    "famA_W5": ["v_family_a_W5"],
    "famA_W7": ["v_family_a_W7"],
}


def pick(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None


def main():
    rows = []
    seqs = sorted(Path("results/helipr").glob("per_frame_*.parquet"))
    for pf in seqs:
        seq = pf.stem.replace("per_frame_", "")
        df = pd.read_parquet(pf)
        t = df["t"].values
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt
        nperseg = min(256, len(t) // 4)
        for method, keys in METHODS.items():
            col = pick(df, keys)
            if col is None:
                continue
            v = df[col].values
            s = accel_stats(v, dt)
            b = psd_bands(v, fs, nperseg)
            rows.append({"sequence": seq, "method": method, **s, **b})
    out = pd.DataFrame(rows)
    out.to_csv("results/helipr/spectral_per_sequence.csv", index=False)
    print(f"Wrote results/helipr/spectral_per_sequence.csv ({len(out)} rows)")

    print("\n=== Per-method aggregate (median, IQR over 6 sequences) ===")
    agg = out.groupby("method").agg(
        rms_med=("rms_accel", "median"),
        rms_q1=("rms_accel", lambda s: s.quantile(0.25)),
        rms_q3=("rms_accel", lambda s: s.quantile(0.75)),
        p95_med=("p95_accel", "median"),
        p99_med=("p99_accel", "median"),
        psd_low_med=("psd_low", "median"),
        psd_mid_med=("psd_mid", "median"),
        psd_high_med=("psd_high", "median"),
    )
    print(agg.to_string())
    agg.to_csv("results/helipr/spectral_aggregate.csv")
    print("\nWrote results/helipr/spectral_aggregate.csv")


if __name__ == "__main__":
    main()
