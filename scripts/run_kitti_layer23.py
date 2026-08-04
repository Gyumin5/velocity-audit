#!/usr/bin/env python3
"""Run Layer 2 (self-consistency) and Layer 3 (OXTS vf alignment) on KITTI raw drives.

Mirrors scripts/run_helipr_layer23.py but uses KITTI's OXTS vf (forward velocity)
as the independent speed reference. Pose is reconstructed from OXTS lat/lon via
a local equirectangular projection (see src/velref/io/kitti.py).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from velref.io.kitti import DEFAULT_DRIVES, load_sequence
from velref.methods.baselines import (
    forward_diff,
    central_diff,
    cubic_spline_deriv,
    smoothing_spline_deriv,
    savgol_deriv,
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
    ap.add_argument("--root", type=Path, default=Path("/mnt/Data/velref/kitti"))
    ap.add_argument("--drives", nargs="*", default=list(DEFAULT_DRIVES))
    ap.add_argument("--ref", default="vf", choices=["vf", "horiz"],
                    help="independent reference: OXTS forward (vf) or horizontal (sqrt(vn^2+ve^2))")
    ap.add_argument("--out", type=Path, default=Path("results/kitti"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for rel in args.drives:
        seq_path = args.root / rel
        if not seq_path.exists():
            print(f"  skip {rel}: {seq_path} not found")
            continue
        print(f"Loading {rel} ...", flush=True)
        s = load_sequence(seq_path)
        n = len(s.pose)
        print(f"  pose samples = {n}, span = {s.pose.t[-1]-s.pose.t[0]:.1f}s, "
              f"pos_acc med = {np.median(s.pos_accuracy):.2f} m, "
              f"vel_acc med = {np.median(s.vel_accuracy):.3f} m/s")

        v_ref = s.vf.copy() if args.ref == "vf" else s.v_horiz
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
                "drive": s.name,
                "method": mname,
                "n": int(v_hat.size),
                "dt_med": dt_med,
                "m1_path_len_err": m1,
                "m3_low_speed_rms": m3,
                "m4_smooth": smooth,
                "m4_lag_s": lag,
                "m8_rmse_vs_oxts": m8_rmse,
                "m8_bias_vs_oxts": float(np.mean(v_hat[interior] - v_ref[interior])),
            })

        per_frame = {
            "t": s.pose.t,
            "x": s.pose.x,
            "y": s.pose.y,
            "v_oxts": v_ref,
        }
        for mname, v_hat in methods.items():
            per_frame[f"v_{mname}"] = v_hat
        pd.DataFrame(per_frame).to_parquet(
            args.out / f"per_frame_{s.name}.parquet", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "summary.csv", index=False)
    df.to_parquet(args.out / "summary.parquet", index=False)
    print(f"\nWrote {len(df)} rows → {args.out}/summary.csv (ref = {args.ref})\n")

    if not df.empty:
        for m in ["m1_path_len_err", "m3_low_speed_rms", "m4_smooth", "m4_lag_s",
                  "m8_rmse_vs_oxts"]:
            piv = df.pivot(index="method", columns="drive", values=m)
            piv["median"] = piv.median(axis=1)
            piv = piv.sort_values("median")
            print(f"\n=== {m} ===")
            print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
