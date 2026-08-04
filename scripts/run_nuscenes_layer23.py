#!/usr/bin/env python3
"""Layer 2/3 evaluation on nuScenes CAN bus pose streams (10 Hz decimated)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from velref.io.nuscenes_can import DEFAULT_SCENES, load_scene
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
    ap.add_argument("--root", type=Path, default=Path("/mnt/Data/velref/nuscenes/can_bus"))
    ap.add_argument("--scenes", nargs="*", default=list(DEFAULT_SCENES))
    ap.add_argument("--target-hz", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("results/nuscenes"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in args.scenes:
        try:
            s = load_scene(args.root, sid, target_hz=args.target_hz)
        except FileNotFoundError as e:
            print(f"  skip {sid}: {e}")
            continue
        n = len(s.pose)
        if n < 50:
            print(f"  skip {sid}: only {n} samples after decimation")
            continue
        dt_med = float(np.median(np.diff(s.pose.t)))
        print(f"{s.name}: n={n}, dur={s.pose.t[-1]:.1f}s, native {s.native_rate_hz:.0f}Hz "
              f"-> {1/dt_med:.1f}Hz, vel range [{s.v_horiz.min():.2f}, {s.v_horiz.max():.2f}]")

        v_ref = s.v_horiz
        methods = make_methods(s.pose)
        # nuScenes scenes are short (~20 s, ~200 samples at 10 Hz), so a smaller
        # interior trim avoids wiping out the experiment.
        trim = max(5, n // 30)
        interior = slice(trim, -trim)

        for mname, v_hat in methods.items():
            if v_hat.shape != v_ref.shape:
                continue
            m1 = m1_path_length_ratio(s.pose, v_hat)
            m3 = m3_low_speed_rms(v_hat[interior], v_ref[interior], thr=0.3)
            m8 = rmse(v_hat[interior], v_ref[interior])
            smooth, lag = m4_smooth_lag(v_hat[interior], v_ref[interior], dt_med)
            rows.append({
                "scene": s.name, "method": mname,
                "n": int(v_hat.size), "dt_med": dt_med,
                "m1_path_len_err": m1,
                "m3_low_speed_rms": m3,
                "m4_smooth": smooth,
                "m4_lag_s": lag,
                "m8_rmse_vs_can": m8,
                "m8_bias": float(np.mean(v_hat[interior] - v_ref[interior])),
            })

        per_frame = {"t": s.pose.t, "x": s.pose.x, "y": s.pose.y, "v_can": v_ref}
        for mname, v_hat in methods.items():
            per_frame[f"v_{mname}"] = v_hat
        pd.DataFrame(per_frame).to_parquet(
            args.out / f"per_frame_{s.name}.parquet", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "summary.csv", index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}/summary.csv\n")

    if not df.empty:
        for m in ["m1_path_len_err", "m3_low_speed_rms", "m4_smooth", "m4_lag_s",
                  "m8_rmse_vs_can"]:
            piv = df.pivot(index="method", columns="scene", values=m)
            piv["median"] = piv.median(axis=1)
            piv = piv.sort_values("median")
            print(f"\n=== {m} ===")
            print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
