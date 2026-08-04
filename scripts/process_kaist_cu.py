#!/usr/bin/env python3
"""Process KAIST Complex Urban (urban08): pose from global_pose.csv,
velocity from wheel encoder.csv (independent sensor stack — SEPARATED regime).

global_pose.csv columns: ts_ns, R11, R12, R13, tx, R21, R22, R23, ty, R31, R32, R33, tz
encoder.csv columns: ts_ns, left_pulse, right_pulse  (cumulative counts; differentiate)

EncoderParameter: 4096 pulses/rev, wheel diameter ~0.624 m → 0.478e-3 m/pulse.
Linear vehicle velocity ≈ mean(v_left, v_right).
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


ENC_RES = 4096
D_LEFT = 0.623803  # m
D_RIGHT = 0.623095
M_PER_PULSE_L = np.pi * D_LEFT / ENC_RES
M_PER_PULSE_R = np.pi * D_RIGHT / ENC_RES


def low_speed_rms(v, threshold=0.5):
    mask = np.abs(v) < threshold
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean(v[mask] ** 2)))


def smoothness_rms2nd(v):
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


def rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def main():
    base = Path("/mnt/Data/velref/kaist_cu/urban08")
    pose_csv = base / "urban08" / "global_pose.csv"
    enc_csv = base / "sensor_data" / "encoder.csv"
    out_dir = _root / "results" / "kaist_cu"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pose
    pose_cols = ["ts", "R11", "R12", "R13", "tx",
                 "R21", "R22", "R23", "ty",
                 "R31", "R32", "R33", "tz"]
    pose_df = pd.read_csv(pose_csv, header=None, names=pose_cols)
    t_pose = pose_df["ts"].to_numpy() / 1e9
    x_pose = pose_df["tx"].to_numpy()
    y_pose = pose_df["ty"].to_numpy()
    # Sort + dedup
    order = np.argsort(t_pose)
    t_pose = t_pose[order]; x_pose = x_pose[order]; y_pose = y_pose[order]
    keep = np.concatenate(([True], np.diff(t_pose) > 1e-6))
    t_pose = t_pose[keep]; x_pose = x_pose[keep]; y_pose = y_pose[keep]
    print(f"pose: {len(t_pose)} samples, "
          f"{(t_pose[-1] - t_pose[0]):.0f} s, dt median {np.median(np.diff(t_pose))*1000:.1f} ms")

    # Load encoder, compute wheel velocity
    enc_df = pd.read_csv(enc_csv, header=None, names=["ts", "left", "right"])
    t_enc = enc_df["ts"].to_numpy() / 1e9
    L = enc_df["left"].to_numpy().astype(np.float64)
    R = enc_df["right"].to_numpy().astype(np.float64)
    order = np.argsort(t_enc)
    t_enc = t_enc[order]; L = L[order]; R = R[order]
    # Compute speed (m/s) at midpoints between samples, then assign to right-side ts
    dt = np.diff(t_enc)
    valid = dt > 1e-3
    dL = np.diff(L) * M_PER_PULSE_L
    dR = np.diff(R) * M_PER_PULSE_R
    vL = np.where(valid, dL / dt, 0.0)
    vR = np.where(valid, dR / dt, 0.0)
    v_enc = 0.5 * (vL + vR)
    # Use right-side ts (i.e., t_enc[1:]) as the timestamp for v_enc[i]
    t_enc_v = t_enc[1:]
    print(f"encoder: {len(v_enc)} samples after diff, dt median {np.median(np.diff(t_enc_v))*1000:.1f} ms, "
          f"speed median {np.median(np.abs(v_enc)):.2f} m/s")

    # Resample both to a common 10 Hz grid (pose timestamps downsampled to 10 Hz)
    target_dt = 0.1
    last = t_pose[0]; keep_idx = [0]
    for i in range(1, len(t_pose)):
        if t_pose[i] - last >= target_dt - 1e-3:
            keep_idx.append(i); last = t_pose[i]
    keep_idx = np.array(keep_idx)
    t_grid = t_pose[keep_idx]
    x_grid = x_pose[keep_idx]; y_grid = y_pose[keep_idx]
    if len(t_grid) < 50:
        print("too few pose samples")
        return

    # Encoder velocity interpolated to t_grid
    v_enc_at_grid = np.interp(t_grid, t_enc_v, v_enc)

    # Build Pose2D
    pose = Pose2D(t_grid, x_grid, y_grid)

    methods = {
        "central": central_diff(pose),
        "family_a_W5": family_a_pointwise(pose, W=5),
        "family_a_W7": family_a_pointwise(pose, W=7),
    }
    rows = []
    out_row = {"sequence": "urban08", "n_frames": len(t_grid)}
    for name, v in methods.items():
        v = np.asarray(v, dtype=float)
        if v.ndim > 1:
            v = np.linalg.norm(v, axis=-1) if v.shape[-1] in (2, 3) else v.squeeze()
        out_row[f"{name}_M2"] = low_speed_rms(v)
        out_row[f"{name}_M3sm"] = smoothness_rms2nd(v)
        out_row[f"{name}_M4"] = rmse(v, v_enc_at_grid)
    rows.append(out_row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sequence.csv", index=False)
    print(f"\n[KAIST CU urban08] -> {out_dir}")
    print(df[["sequence", "n_frames", "central_M4", "family_a_W5_M4",
              "central_M3sm", "family_a_W5_M3sm", "central_M2", "family_a_W5_M2"]].to_string())
    if df.central_M4.iloc[0] > 0:
        delta = (df.family_a_W5_M4.iloc[0] / df.central_M4.iloc[0] - 1) * 100
        print(f"\nFamily A W5 delta vs central: {delta:+.1f}%")
    print(f"smoothness ratio: {df.central_M3sm.iloc[0] / df.family_a_W5_M3sm.iloc[0]:.2f}x")


if __name__ == "__main__":
    main()
