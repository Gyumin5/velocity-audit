#!/usr/bin/env python3
"""Run Layer 2 / Layer 3 on Boreas sequences (10 Hz downsampled)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from velref.io.boreas import DEFAULT_SEQUENCES, load_sequence
from velref.methods.baselines import (
    forward_diff, central_diff, cubic_spline_deriv,
    smoothing_spline_deriv, savgol_deriv,
)
from velref.methods.family_a import family_a_pointwise, family_a_midspan
from velref.metrics.layer2 import m1_path_length_ratio, m3_low_speed_rms, m4_smooth_lag
from velref.metrics.core_metrics import rmse


def make_methods(pose):
    return {
        "forward": forward_diff(pose),
        "central": central_diff(pose),
        "cubic_global": cubic_spline_deriv(pose),
        "savgol_w7p3": savgol_deriv(pose, window=7, polyorder=3),
        "smoothing_spline": smoothing_spline_deriv(pose),
        "family_a_W3": family_a_pointwise(pose, W=3),
        "family_a_W5": family_a_pointwise(pose, W=5),
        "family_a_W7": family_a_pointwise(pose, W=7),
        "family_a_midspan_W5": family_a_midspan(pose, W=5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/Data/velref/boreas"))
    ap.add_argument("--sequences", nargs="*", default=list(DEFAULT_SEQUENCES))
    ap.add_argument("--target-hz", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("results/boreas"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seq_name in args.sequences:
        seq_path = args.root / seq_name
        if not seq_path.exists():
            print(f"  skip {seq_name}: {seq_path} not found")
            continue
        print(f"Loading {seq_name} ...", flush=True)
        s = load_sequence(seq_path, target_hz=args.target_hz)
        n = len(s.pose)
        print(f"  pose samples = {n}, span = {s.pose.t[-1]-s.pose.t[0]:.1f}s, "
              f"native {s.native_rate_hz:.0f} Hz -> effective {n/(s.pose.t[-1]-s.pose.t[0]):.1f} Hz")

        v_ref = s.v_horiz
        dt_med = float(np.median(np.diff(s.pose.t)))

        methods = make_methods(s.pose)
        interior = slice(max(10, n // 50), -max(10, n // 50))

        for mname, v_hat in methods.items():
            if v_hat.shape != v_ref.shape:
                continue
            m1 = m1_path_length_ratio(s.pose, v_hat)
            m3 = m3_low_speed_rms(v_hat[interior], v_ref[interior], thr=0.3)
            m8_rmse = rmse(v_hat[interior], v_ref[interior])
            smooth, lag = m4_smooth_lag(v_hat[interior], v_ref[interior], dt_med)
            rows.append({
                "sequence": seq_name,
                "method": mname,
                "n": int(v_hat.size),
                "dt_med": dt_med,
                "m1_path_len_err": m1,
                "m3_low_speed_rms": m3,
                "m4_smooth": smooth,
                "m4_lag_s": lag,
                "m8_rmse_vs_pospac": m8_rmse,
                "m8_bias": float(np.mean(v_hat[interior] - v_ref[interior])),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "summary.csv", index=False)
    print(f"\nWrote {len(df)} rows → {args.out}/summary.csv\n")

    if not df.empty:
        for m in ["m1_path_len_err", "m3_low_speed_rms", "m4_smooth", "m4_lag_s",
                  "m8_rmse_vs_pospac"]:
            piv = df.pivot(index="method", columns="sequence", values=m)
            piv["median"] = piv.median(axis=1)
            piv = piv.sort_values("median")
            print(f"\n=== {m} ===")
            print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
