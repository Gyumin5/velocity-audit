#!/usr/bin/env python3
"""Layer 2/3 evaluation on Oxford RobotCar runs.

Default mode 'rtk_ins' takes pose from rtk.csv and velocity reference from
ins.csv (separated provenance). Pass --mode ins to use INS-only (coupled
provenance) for a controlled internal contrast.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from velref.io.oxford_robotcar import DEFAULT_RUNS, load_sequence
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
    ap.add_argument("--root", type=Path, default=Path("/mnt/Data/velref/oxford_robotcar"))
    ap.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    ap.add_argument("--mode", default="rtk", choices=["rtk_ins", "ins", "rtk"])
    ap.add_argument("--out", type=Path, default=Path("results/oxford"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in args.runs:
        seq_path = args.root / run
        if not seq_path.exists():
            print(f"  skip {run}: {seq_path} not found")
            continue
        print(f"Loading {run} (mode={args.mode}) ...", flush=True)
        try:
            s = load_sequence(seq_path, mode=args.mode)
        except FileNotFoundError as e:
            print(f"  skip {run}: missing file {e}")
            continue
        n = len(s.pose)
        if n < 100:
            print(f"  skip {run}: only {n} pose samples")
            continue
        dt_med = float(np.median(np.diff(s.pose.t)))
        print(f"  pose samples = {n}, span = {s.pose.t[-1]-s.pose.t[0]:.1f}s, "
              f"dt_med = {dt_med*1000:.1f} ms, "
              f"pose_source = {s.pose_source}, dropped_ins/rtk = {s.n_dropped_ins}/{s.n_dropped_rtk}")

        v_ref = s.v_horiz
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
                "run": run,
                "mode": args.mode,
                "method": mname,
                "n": int(v_hat.size),
                "dt_med": dt_med,
                "m1_path_len_err": m1,
                "m3_low_speed_rms": m3,
                "m4_smooth": smooth,
                "m4_lag_s": lag,
                "m8_rmse_vs_ins": m8_rmse,
                "m8_bias": float(np.mean(v_hat[interior] - v_ref[interior])),
            })

        per_frame = {"t": s.pose.t, "x": s.pose.x, "y": s.pose.y, "v_ins": v_ref}
        for mname, v_hat in methods.items():
            per_frame[f"v_{mname}"] = v_hat
        pd.DataFrame(per_frame).to_parquet(
            args.out / f"per_frame_{run}_{args.mode}.parquet", index=False)

    df = pd.DataFrame(rows)
    out_csv = args.out / f"summary_{args.mode}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows → {out_csv}\n")

    if not df.empty:
        for m in ["m1_path_len_err", "m3_low_speed_rms", "m4_smooth", "m4_lag_s",
                  "m8_rmse_vs_ins"]:
            piv = df.pivot(index="method", columns="run", values=m)
            piv["median"] = piv.median(axis=1)
            piv = piv.sort_values("median")
            print(f"\n=== {m} ===")
            print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
