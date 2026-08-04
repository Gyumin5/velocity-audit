#!/usr/bin/env python3
"""DR (dead-reckoning) experiment across all 7 datasets.

For each dataset and method, integrate v_hat(t) over time -> cumulative distance,
compare to integrated reference -> drift. Report median |drift| per method
per dataset at the final timestamp.
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import pandas as pd

REF_COL = {
    "helipr":   "v_ins",
    "oxford":   "v_ins",
    "nuscenes": "v_can",
    "kitti":    "v_oxts",
    "kitti360": "v_ref",
    "boreas":   "v_ref",
    "pit30m":   "v_ref",
}
DATASETS = list(REF_COL.keys())
METHODS = ["v_forward", "v_central", "v_savgol_w7p3",
           "v_family_a_W3", "v_family_a_W5", "v_family_a_W7"]


def cumdist(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    return np.cumsum(v * dt)


def main():
    out = Path("results/dr")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in DATASETS:
        ref_col = REF_COL[d]
        files = sorted(glob.glob(f"results/{d}/per_frame_*.parquet"))
        if not files:
            print(f"  {d}: no per_frame files")
            continue
        for fp in files:
            df = pd.read_parquet(fp)
            t = df["t"].to_numpy()
            if ref_col not in df.columns:
                continue
            v_ref = df[ref_col].to_numpy()
            s_ref = cumdist(t, v_ref)
            elapsed = float(t[-1] - t[0])
            for m in METHODS:
                if m not in df.columns:
                    continue
                v = df[m].to_numpy()
                if np.any(~np.isfinite(v)):
                    continue
                s = cumdist(t, v)
                drift = s[-1] - s_ref[-1]
                rows.append({
                    "dataset": d,
                    "sequence": Path(fp).stem.replace("per_frame_", ""),
                    "method": m,
                    "elapsed_s": elapsed,
                    "dist_traveled_m": float(s_ref[-1]),
                    "drift_m": float(drift),
                    "drift_abs_m": float(abs(drift)),
                    "drift_pct": float(100 * drift / s_ref[-1]) if s_ref[-1] != 0 else float("nan"),
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out / "dr_all.csv", index=False)

    agg = (
        df_out.groupby(["dataset", "method"])
        .agg(n=("sequence", "count"),
             median_dist_m=("dist_traveled_m", "median"),
             median_drift_abs_m=("drift_abs_m", "median"),
             median_drift_pct=("drift_pct", "median"))
        .reset_index()
    )
    agg.to_csv(out / "dr_all_summary.csv", index=False)

    # Pivot for readability: median |drift| % per (dataset, method).
    piv = agg.pivot_table(index="dataset", columns="method",
                          values="median_drift_pct").reindex(DATASETS)
    print("\n=== Median drift % per dataset per method (final position) ===")
    print(piv[METHODS].round(4).to_string())
    print("\n=== Median |drift| m per dataset per method (final position) ===")
    piv_m = agg.pivot_table(index="dataset", columns="method",
                            values="median_drift_abs_m").reindex(DATASETS)
    print(piv_m[METHODS].round(3).to_string())
    print("\n=== Median distance traveled (m) ===")
    piv_d = agg.pivot_table(index="dataset", columns="method",
                            values="median_dist_m").reindex(DATASETS)
    print(piv_d[[METHODS[0]]].round(0).to_string())


if __name__ == "__main__":
    main()
