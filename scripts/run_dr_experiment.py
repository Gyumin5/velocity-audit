#!/usr/bin/env python3
"""Dead-reckoning experiment: integrate per-method estimated speed and compare
the resulting cumulative distance against the integrated INSPVA reference on
HeLiPR. Each method's drift after T seconds is a direct functional metric of
how its speed quality propagates downstream.

Output: results/dr/dr_helipr.csv with cumulative distance error per method,
per sequence, at fixed checkpoints (100 s, 500 s, end).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SEQS = ["bridge01", "dcc05", "kaist05", "riverside05", "roundabout01", "town01"]
METHODS = [
    "v_forward",
    "v_central",
    "v_savgol_w7p3",
    "v_family_a_W3",
    "v_family_a_W5",
    "v_family_a_W7",
]
CHECKPOINTS = [100.0, 500.0, None]  # None = final


def cumulative_dist(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    return np.cumsum(v * dt)


def main():
    out = Path("results/dr")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seq in SEQS:
        df = pd.read_parquet(f"results/helipr/per_frame_{seq}.parquet")
        t = df["t"].to_numpy()
        v_ref = df["v_ins"].to_numpy()
        s_ref = cumulative_dist(t, v_ref)
        for m in METHODS:
            v = df[m].to_numpy()
            s = cumulative_dist(t, v)
            err = s - s_ref
            for cp in CHECKPOINTS:
                if cp is None:
                    idx = -1
                    label = "final"
                else:
                    above = np.where(t >= cp)[0]
                    if above.size == 0:
                        continue
                    idx = int(above[0])
                    label = f"t{int(cp)}s"
                rows.append({
                    "sequence": seq,
                    "method": m,
                    "checkpoint": label,
                    "elapsed_s": float(t[idx] - t[0]),
                    "dist_traveled_m": float(s_ref[idx]),
                    "drift_m": float(err[idx]),
                    "drift_abs_m": float(abs(err[idx])),
                    "drift_pct": float(100 * err[idx] / s_ref[idx]) if s_ref[idx] != 0 else float("nan"),
                })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out / "dr_helipr.csv", index=False)

    # Aggregate: median |drift| per method per checkpoint across 6 seqs.
    agg = (
        df_out.groupby(["method", "checkpoint"])
        .agg(median_drift_abs_m=("drift_abs_m", "median"),
             median_drift_pct=("drift_pct", "median"),
             max_drift_abs_m=("drift_abs_m", "max"))
        .reset_index()
    )
    agg.to_csv(out / "dr_helipr_summary.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
