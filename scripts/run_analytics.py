#!/usr/bin/env python3
"""Additional Layer 2/3 analytics: L1.8 degradation, sensitivity, curvature, dead-reckoning, dcc05 diag.

Consumes per_frame_<seq>.parquet produced by run_helipr_layer23.py and re-runs targeted
analyses without re-loading big raw files.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from velref.io.helipr import load_sequence, interpolate_ins_to_pose, DEFAULT_SEQUENCES
from velref.core.trajectory import Pose2D
from velref.core.curvature import estimate_curvature
from velref.methods.baselines import (
    forward_diff,
    central_diff,
    cubic_spline_deriv,
    smoothing_spline_deriv,
    smoothing_spline_tuned_deriv,
    savgol_deriv,
)
from velref.methods.family_a import family_a_pointwise


# ---------- Experiment 1: L1.8 semi-controlled degradation ----------
def experiment_degradation(seq, out_dir: Path):
    rng = np.random.default_rng(0)
    rows = []
    # Grid
    ds_factors = [1, 2, 5, 10]            # subsample factor (10 Hz → {10, 5, 2, 1} Hz effective)
    noise_sigmas = [0.0, 0.05, 0.2, 0.5]  # position noise [m]
    jitter_sigmas = [0.0, 0.005, 0.02]    # timestamp jitter [s]
    full_ins = interpolate_ins_to_pose(seq)

    methods = {
        "central": lambda p: central_diff(p),
        "savgol_w7p3": lambda p: savgol_deriv(p, 7, 3),
        "smoothing_spline_default": smoothing_spline_deriv,
        "smoothing_spline_tuned": smoothing_spline_tuned_deriv,
        "family_a_W5": lambda p: family_a_pointwise(p, W=5),
        "family_a_W7": lambda p: family_a_pointwise(p, W=7),
    }
    for df_factor, sigma, jitter in itertools.product(ds_factors, noise_sigmas, jitter_sigmas):
        idx = np.arange(0, len(seq.pose), df_factor)
        t = seq.pose.t[idx].copy()
        x = seq.pose.x[idx].copy()
        y = seq.pose.y[idx].copy()
        if jitter > 0:
            t = t + rng.normal(0, jitter, t.size)
            keep_order = np.argsort(t)
            t = t[keep_order]
            x = x[keep_order]
            y = y[keep_order]
            strict = np.concatenate([[True], np.diff(t) > 1e-6])
            t, x, y = t[strict], x[strict], y[strict]
            idx = idx[np.isin(idx, idx[keep_order])][strict[: idx.size]]
            idx = np.arange(t.size)  # idx now refers to degraded indices, not original.
        if sigma > 0:
            x = x + rng.normal(0, sigma, x.size)
            y = y + rng.normal(0, sigma, y.size)
        if t.size < 30:
            continue
        p_deg = Pose2D(t, x, y)
        # Reference: INS at original rate, interpolated to degraded timestamps.
        v_ref_deg = np.interp(t, seq.t_ins, seq.v_ins_horiz)
        interior = slice(max(10, t.size // 50), -max(10, t.size // 50))
        for mname, fn in methods.items():
            try:
                v_hat = fn(p_deg)
            except Exception:
                continue
            if v_hat.shape != v_ref_deg.shape:
                continue
            rmse = float(np.sqrt(np.mean((v_hat[interior] - v_ref_deg[interior]) ** 2)))
            rows.append({
                "ds_factor": df_factor,
                "effective_hz": 1.0 / (np.median(np.diff(t)) if t.size > 1 else 1.0),
                "noise_sigma": sigma,
                "jitter_sigma": jitter,
                "method": mname,
                "rmse": rmse,
                "n": int(t.size),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "l18_degradation.parquet", index=False)
    df.to_csv(out_dir / "l18_degradation.csv", index=False)
    print(f"L1.8 degradation: {len(df)} rows")
    return df


# ---------- Experiment 2: W × degree sensitivity heatmap ----------
def experiment_sensitivity(seq, out_dir: Path):
    v_ref = interpolate_ins_to_pose(seq)
    n = len(seq.pose)
    interior = slice(max(10, n // 50), -max(10, n // 50))
    rows = []
    for W in [2, 3, 4, 5, 6, 7, 9, 12]:
        for degree in [2, 3, 4, 5]:
            try:
                v_hat = family_a_pointwise(seq.pose, W=W, degree=degree)
            except Exception:
                continue
            rmse = float(np.sqrt(np.mean((v_hat[interior] - v_ref[interior]) ** 2)))
            rows.append({"W": W, "degree": degree, "rmse_vs_ins": rmse})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensitivity.csv", index=False)
    print(f"Sensitivity: {len(df)} rows")
    return df


# ---------- Experiment 3: curvature-bin breakdown ----------
def experiment_curvature_bins(seq, out_dir: Path):
    kappa = estimate_curvature(seq.pose, window=9, polyorder=3)
    v_ref = interpolate_ins_to_pose(seq)
    methods = {
        "central": central_diff(seq.pose),
        "savgol_w7p3": savgol_deriv(seq.pose, 7, 3),
        "family_a_W5": family_a_pointwise(seq.pose, W=5),
    }
    # Bins in |kappa| [1/m]
    bin_edges = np.array([0.0, 1e-3, 5e-3, 2e-2, 0.1, 1.0])
    bin_labels = [f"{bin_edges[i]:.0e}-{bin_edges[i+1]:.0e}" for i in range(len(bin_edges) - 1)]
    abs_kappa = np.abs(kappa)
    bins = np.digitize(abs_kappa, bin_edges) - 1
    bins = np.clip(bins, 0, len(bin_labels) - 1)
    rows = []
    for mname, v_hat in methods.items():
        for b, lbl in enumerate(bin_labels):
            mask = bins == b
            if mask.sum() < 20:
                continue
            rmse = float(np.sqrt(np.mean((v_hat[mask] - v_ref[mask]) ** 2)))
            rows.append({"method": mname, "kappa_bin": lbl, "n": int(mask.sum()), "rmse_vs_ins": rmse})
    df = pd.DataFrame(rows)
    print(f"Curvature bins: {len(df)} rows")
    return df


# ---------- Experiment 4: dead-reckoning residual (M9) ----------
def experiment_deadreckoning(seq, out_dir: Path):
    """Replace speed channel with each method, keep yaw from pose, integrate, compare endpoint."""
    # Heading from atan2 of diff of pose (non-parametric).
    dx = np.diff(seq.pose.x)
    dy = np.diff(seq.pose.y)
    heading = np.arctan2(dy, dx)
    heading = np.concatenate([heading, heading[-1:]])
    dt = np.diff(seq.pose.t)

    methods = {
        "raw_poly_len": np.hypot(dx, dy) / dt,
        "central": central_diff(seq.pose)[:-1],
        "savgol_w7p3": savgol_deriv(seq.pose, 7, 3)[:-1],
        "smoothing_spline_tuned": smoothing_spline_tuned_deriv(seq.pose)[:-1],
        "family_a_W5": family_a_pointwise(seq.pose, W=5)[:-1],
        "family_a_W7": family_a_pointwise(seq.pose, W=7)[:-1],
    }
    rows = []
    # Use 100m sub-segments (sliding) and report median endpoint error.
    # Simpler: full-sequence endpoint vs pose endpoint.
    for mname, v_seg in methods.items():
        if v_seg.size != dt.size:
            continue
        # Integrate position using midpoint heading.
        dx_hat = v_seg * np.cos(heading[:-1]) * dt
        dy_hat = v_seg * np.sin(heading[:-1]) * dt
        x_hat = seq.pose.x[0] + np.concatenate([[0.0], np.cumsum(dx_hat)])
        y_hat = seq.pose.y[0] + np.concatenate([[0.0], np.cumsum(dy_hat)])
        err = np.hypot(x_hat[-1] - seq.pose.x[-1], y_hat[-1] - seq.pose.y[-1])
        total_len = float(np.sum(np.hypot(dx, dy)))
        rows.append({
            "method": mname,
            "endpoint_err_m": float(err),
            "relative_err": float(err / max(total_len, 1e-6)),
            "total_polyline_len_m": total_len,
        })
    df = pd.DataFrame(rows)
    print(f"Dead-reckoning: {len(df)} rows")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/helipr"))
    ap.add_argument("--out", type=Path, default=Path("results/helipr"))
    ap.add_argument("--sensitivity-seq", default="roundabout01")
    ap.add_argument("--l18-seq", default="roundabout01")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Run curvature bins + dead-reckoning across all sequences.
    all_curv_rows = []
    all_dr_rows = []
    for seq_name in DEFAULT_SEQUENCES:
        seq_path = args.root / seq_name
        if not seq_path.exists():
            continue
        print(f"Loading {seq_name} for curvature + dead-reckoning ...", flush=True)
        s = load_sequence(seq_path, lidar="Ouster", use_global=True)
        curv_df = experiment_curvature_bins(s, args.out)
        curv_df["sequence"] = seq_name
        all_curv_rows.append(curv_df)
        dr_df = experiment_deadreckoning(s, args.out)
        dr_df["sequence"] = seq_name
        all_dr_rows.append(dr_df)
    if all_curv_rows:
        curv_all = pd.concat(all_curv_rows, ignore_index=True)
        curv_all.to_csv(args.out / "curvature_bins_all.csv", index=False)
        print(f"Combined curvature bins: {len(curv_all)} rows")
    if all_dr_rows:
        dr_all = pd.concat(all_dr_rows, ignore_index=True)
        dr_all.to_csv(args.out / "dead_reckoning_all.csv", index=False)
        print(f"Combined dead-reckoning: {len(dr_all)} rows")

    # Sensitivity and L1.8 only on the selected sequence.
    seq_path = args.root / args.sensitivity_seq
    if seq_path.exists():
        print(f"Loading {args.sensitivity_seq} for sensitivity + L1.8 ...", flush=True)
        s = load_sequence(seq_path, lidar="Ouster", use_global=True)
        experiment_sensitivity(s, args.out)
        experiment_degradation(s, args.out)

    print("Done.")


if __name__ == "__main__":
    main()
