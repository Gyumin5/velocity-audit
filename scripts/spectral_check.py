#!/usr/bin/env python3
"""Spectral analysis: does Family A preserve physical acceleration band?

Plots / prints PSD of v(t) for central diff, Family A W=5, and INS reference.
Reports acceleration RMS and 95th percentile for each method on roundabout01.
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


def main():
    pf = Path("results/helipr/per_frame_roundabout01.parquet")
    if not pf.exists():
        # Fallback: try any per_frame file
        candidates = list(Path("results/helipr").glob("per_frame_*.parquet"))
        if not candidates:
            raise SystemExit("No per_frame parquet found")
        pf = candidates[0]
    df = pd.read_parquet(pf)
    print(f"Using {pf}; columns: {list(df.columns)}")

    t = df["t"].values
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    print(f"fs = {fs:.2f} Hz, n = {len(t)}")

    cols = {
        "INS_ref": "v_ins" if "v_ins" in df.columns else "v_ins_horiz",
        "central": "v_central",
        "famA_W5": "v_family_a_W5",
        "famA_W7": "v_family_a_W7",
    }
    nperseg = min(256, len(t) // 4)

    print(f"\n=== Acceleration stats (m/s^2) ===")
    print(f"{'method':<12} {'RMS':>8} {'p95':>8} {'p99':>8}")
    accel_rows = []
    for name, col in cols.items():
        if col not in df.columns:
            continue
        v = df[col].values
        s = accel_stats(v, dt)
        print(f"{name:<12} {s['rms_accel']:>8.3f} {s['p95_accel']:>8.3f} {s['p99_accel']:>8.3f}")
        accel_rows.append({"method": name, **s})

    print(f"\n=== Velocity PSD power in frequency bands (low / acceleration / noise) ===")
    print(f"{'method':<12} {'<0.5Hz':>10} {'0.5-2Hz':>10} {'>2Hz':>10}")
    for name, col in cols.items():
        if col not in df.columns:
            continue
        v = df[col].values
        f, P = welch(v, fs=fs, nperseg=nperseg)
        low = float(np.trapezoid(P[f < 0.5], f[f < 0.5]))
        mid = float(np.trapezoid(P[(f >= 0.5) & (f < 2.0)], f[(f >= 0.5) & (f < 2.0)]))
        high = float(np.trapezoid(P[f >= 2.0], f[f >= 2.0]))
        print(f"{name:<12} {low:>10.4f} {mid:>10.4f} {high:>10.4f}")

    pd.DataFrame(accel_rows).to_csv("results/spectral_check.csv", index=False)
    print(f"\nWrote results/spectral_check.csv")


if __name__ == "__main__":
    main()
