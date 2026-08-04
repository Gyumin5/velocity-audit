#!/usr/bin/env python3
"""Artifact verification for KAIST CU urban08 +8% surprise:
1. Latency shift sweep (encoder vs pose timestamps)
2. Window size sweep
3. Pose PSD vs HeLiPR comparison
4. Per-state error analysis (stationary / accelerating / steady)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from velref.core.trajectory import Pose2D  # noqa: E402
from velref.methods.family_a import family_a_pointwise  # noqa: E402
from velref.methods.baselines import central_diff  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


ENC_RES = 4096
D_LEFT = 0.623803
D_RIGHT = 0.623095
M_PER_PULSE_L = np.pi * D_LEFT / ENC_RES
M_PER_PULSE_R = np.pi * D_RIGHT / ENC_RES


def rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def load():
    base = Path("/mnt/Data/velref/kaist_cu/urban08")
    pose_csv = base / "urban08" / "global_pose.csv"
    enc_csv = base / "sensor_data" / "encoder.csv"
    pose_cols = ["ts", "R11", "R12", "R13", "tx", "R21", "R22", "R23", "ty",
                 "R31", "R32", "R33", "tz"]
    pose_df = pd.read_csv(pose_csv, header=None, names=pose_cols)
    t_pose = pose_df["ts"].to_numpy() / 1e9
    x_pose = pose_df["tx"].to_numpy()
    y_pose = pose_df["ty"].to_numpy()
    order = np.argsort(t_pose)
    t_pose, x_pose, y_pose = t_pose[order], x_pose[order], y_pose[order]
    keep = np.concatenate(([True], np.diff(t_pose) > 1e-6))
    t_pose, x_pose, y_pose = t_pose[keep], x_pose[keep], y_pose[keep]

    enc_df = pd.read_csv(enc_csv, header=None, names=["ts", "left", "right"])
    t_enc = enc_df["ts"].to_numpy() / 1e9
    L = enc_df["left"].to_numpy().astype(np.float64)
    R = enc_df["right"].to_numpy().astype(np.float64)
    order = np.argsort(t_enc)
    t_enc, L, R = t_enc[order], L[order], R[order]
    dt = np.diff(t_enc)
    valid = dt > 1e-3
    vL = np.where(valid, np.diff(L) * M_PER_PULSE_L / dt, 0.0)
    vR = np.where(valid, np.diff(R) * M_PER_PULSE_R / dt, 0.0)
    v_enc = 0.5 * (vL + vR)
    t_enc_v = t_enc[1:]
    return t_pose, x_pose, y_pose, t_enc_v, v_enc


def downsample10hz(t, x, y):
    target_dt = 0.1
    last = t[0]; keep = [0]
    for i in range(1, len(t)):
        if t[i] - last >= target_dt - 1e-3:
            keep.append(i); last = t[i]
    keep = np.array(keep)
    return t[keep], x[keep], y[keep]


def main():
    t_pose, x_pose, y_pose, t_enc_v, v_enc = load()
    t_grid, x_grid, y_grid = downsample10hz(t_pose, x_pose, y_pose)
    pose = Pose2D(t_grid, x_grid, y_grid)
    v_central = np.asarray(central_diff(pose), dtype=float)
    if v_central.ndim > 1:
        v_central = np.linalg.norm(v_central, axis=-1)
    v_fa5 = np.asarray(family_a_pointwise(pose, W=5), dtype=float)
    if v_fa5.ndim > 1:
        v_fa5 = np.linalg.norm(v_fa5, axis=-1)

    print("=" * 60)
    print("1. LATENCY SHIFT SWEEP (encoder ts shifted by tau)")
    print("=" * 60)
    print(f"{'tau (ms)':>10} {'cent M4':>10} {'FA W5 M4':>10} {'FA delta %':>12}")
    best = (0, 1e9, 1e9, 0)
    for tau_ms in range(-500, 501, 50):
        tau = tau_ms / 1000.0
        v_ref = np.interp(t_grid, t_enc_v + tau, v_enc)
        rc = rmse(v_central, v_ref)
        rf = rmse(v_fa5, v_ref)
        delta = (rf / rc - 1) * 100 if rc > 0 else float("nan")
        if rc < best[1]:
            best = (tau_ms, rc, rf, delta)
        print(f"{tau_ms:>10d} {rc:>10.4f} {rf:>10.4f} {delta:>+11.1f}%")
    print(f"\nbest cent M4 at tau={best[0]} ms: cent={best[1]:.4f} FA={best[2]:.4f} delta={best[3]:+.1f}%")

    print("\n" + "=" * 60)
    print("2. WINDOW SWEEP (Family A W=3/5/7/9, encoder no shift)")
    print("=" * 60)
    v_ref = np.interp(t_grid, t_enc_v, v_enc)
    rc = rmse(v_central, v_ref)
    print(f"central diff M4: {rc:.4f}")
    for W in [3, 5, 7, 9]:
        v_w = np.asarray(family_a_pointwise(pose, W=W), dtype=float)
        if v_w.ndim > 1:
            v_w = np.linalg.norm(v_w, axis=-1)
        r = rmse(v_w, v_ref)
        delta = (r / rc - 1) * 100
        print(f"  Family A W={W}: M4={r:.4f}  delta={delta:+.1f}%")

    print("\n" + "=" * 60)
    print("3. PER-STATE ERROR ANALYSIS")
    print("=" * 60)
    states = {
        "stationary (|v|<0.5)": np.abs(v_ref) < 0.5,
        "low (0.5-3 m/s)": (np.abs(v_ref) >= 0.5) & (np.abs(v_ref) < 3),
        "mid (3-10 m/s)": (np.abs(v_ref) >= 3) & (np.abs(v_ref) < 10),
        "high (>10 m/s)": np.abs(v_ref) >= 10,
    }
    a_ref = np.diff(v_ref) / 0.1
    accel_state = np.concatenate(([0], np.abs(a_ref)))
    for label, mask in states.items():
        if mask.sum() < 30:
            continue
        n = min(len(v_central), len(v_ref))
        m = mask[:n]
        rc = rmse(v_central[:n][m], v_ref[:n][m])
        rf = rmse(v_fa5[:n][m], v_ref[:n][m])
        delta = (rf / rc - 1) * 100 if rc > 0 else float("nan")
        print(f"  {label:<28} n={mask.sum():>5}  cent={rc:.4f}  FA={rf:.4f}  delta={delta:+.1f}%")
    # Accel / steady split
    n = min(len(v_central), len(v_ref), len(accel_state))
    accel_mask = accel_state[:n] > 0.5
    steady_mask = ~accel_mask
    for label, mask in [("steady (|a|<=0.5)", steady_mask), ("accel/decel (|a|>0.5)", accel_mask)]:
        if mask.sum() < 30:
            continue
        rc = rmse(v_central[:n][mask], v_ref[:n][mask])
        rf = rmse(v_fa5[:n][mask], v_ref[:n][mask])
        delta = (rf / rc - 1) * 100 if rc > 0 else float("nan")
        print(f"  {label:<28} n={mask.sum():>5}  cent={rc:.4f}  FA={rf:.4f}  delta={delta:+.1f}%")

    print("\n" + "=" * 60)
    print("4. POSE QUALITY: jerk and PSD high-band of central-diff velocity")
    print("=" * 60)
    jerk = np.diff(v_central, n=2)
    print(f"central-diff velocity 2nd-diff RMS: {np.sqrt(np.mean(jerk**2)):.4f}")
    print(f"central-diff velocity stdev      : {np.std(v_central):.3f}")
    # Compare with HeLiPR pose if available
    helipr_paths = list(REPO_ROOT / "results" / "helipr".glob("per_frame_*.parquet"))
    if helipr_paths:
        df = pd.read_parquet(helipr_paths[0])
        if "v_central" in df.columns:
            jh = np.diff(df["v_central"].to_numpy(), n=2)
            print(f"HeLiPR ({helipr_paths[0].stem}) central-diff 2nd-diff RMS: {np.sqrt(np.mean(jh**2)):.4f}")
    print("(Lower 2nd-diff RMS in central-diff velocity ⇒ pose is already smoother → less room for Family A)")


if __name__ == "__main__":
    main()
