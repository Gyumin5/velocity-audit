#!/usr/bin/env python3
"""Process all available KAIST CU sequences and aggregate."""
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
M_PER_PULSE_L = np.pi * 0.623803 / ENC_RES
M_PER_PULSE_R = np.pi * 0.623095 / ENC_RES


def low_speed_rms(v, threshold=0.5):
    mask = np.abs(v) < threshold
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean(v[mask] ** 2)))


def smoothness_rms2nd(v):
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


def rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def process(seq: str):
    base = Path(f"/mnt/Data/velref/kaist_cu/{seq}")
    pose_csv = base / seq / "global_pose.csv"
    # Encoder CSV may be under base/sensor_data/ or base/seq/sensor_data/
    enc_csv = base / "sensor_data" / "encoder.csv"
    if not enc_csv.exists():
        enc_csv = base / seq / "sensor_data" / "encoder.csv"
    if not pose_csv.exists() or not enc_csv.exists():
        return None
    pose_df = pd.read_csv(pose_csv, header=None)
    t_pose = pose_df.iloc[:, 0].to_numpy() / 1e9
    x_pose = pose_df.iloc[:, 4].to_numpy()
    y_pose = pose_df.iloc[:, 8].to_numpy()
    order = np.argsort(t_pose)
    t_pose = t_pose[order]; x_pose = x_pose[order]; y_pose = y_pose[order]
    keep = np.concatenate(([True], np.diff(t_pose) > 1e-6))
    t_pose, x_pose, y_pose = t_pose[keep], x_pose[keep], y_pose[keep]

    enc = pd.read_csv(enc_csv, header=None, names=["ts", "left", "right"])
    t_enc = enc["ts"].to_numpy() / 1e9
    L = enc["left"].to_numpy().astype(np.float64)
    R = enc["right"].to_numpy().astype(np.float64)
    order = np.argsort(t_enc)
    t_enc, L, R = t_enc[order], L[order], R[order]
    dt_e = np.diff(t_enc)
    valid = dt_e > 1e-3
    vL = np.where(valid, np.diff(L) * M_PER_PULSE_L / dt_e, 0.0)
    vR = np.where(valid, np.diff(R) * M_PER_PULSE_R / dt_e, 0.0)
    v_enc = 0.5 * (vL + vR)
    t_enc_v = t_enc[1:]

    target_dt = 0.1
    last = t_pose[0]; keep = [0]
    for i in range(1, len(t_pose)):
        if t_pose[i] - last >= target_dt - 1e-3:
            keep.append(i); last = t_pose[i]
    keep = np.array(keep)
    t_grid = t_pose[keep]; x_grid = x_pose[keep]; y_grid = y_pose[keep]
    if len(t_grid) < 50:
        return None
    v_ref = np.interp(t_grid, t_enc_v, v_enc)
    pose = Pose2D(t_grid, x_grid, y_grid)
    out = {"sequence": seq, "n_frames": len(t_grid), "duration_s": t_grid[-1] - t_grid[0]}
    for name, fn in [("central", lambda: central_diff(pose)),
                     ("family_a_W5", lambda: family_a_pointwise(pose, W=5)),
                     ("family_a_W7", lambda: family_a_pointwise(pose, W=7))]:
        v = np.asarray(fn(), dtype=float)
        if v.ndim > 1:
            v = np.linalg.norm(v, axis=-1) if v.shape[-1] in (2, 3) else v.squeeze()
        out[f"{name}_M2"] = low_speed_rms(v)
        out[f"{name}_M3sm"] = smoothness_rms2nd(v)
        out[f"{name}_M4"] = rmse(v, v_ref)
    # pose smoothness diagnostic
    v_central = np.asarray(central_diff(pose), dtype=float)
    if v_central.ndim > 1:
        v_central = np.linalg.norm(v_central, axis=-1)
    out["central_diff_jerk_RMS"] = smoothness_rms2nd(v_central)
    return out


def main():
    out_dir = _root / "results" / "kaist_cu"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    available = sorted({p.parts[-3] for p in Path("/mnt/Data/velref/kaist_cu").glob("urban*/urban*/global_pose.csv")})
    print(f"available: {available}")
    for seq in available:
        r = process(seq)
        if r is None:
            print(f"[skip] {seq} (missing or short)")
            continue
        delta = (r["family_a_W5_M4"] / r["central_M4"] - 1) * 100 if r["central_M4"] > 0 else float("nan")
        ratio = r["central_M3sm"] / r["family_a_W5_M3sm"] if r["family_a_W5_M3sm"] > 0 else float("nan")
        rows.append(r)
        print(f"  {seq}: n={r['n_frames']:5d} ({r['duration_s']:.0f}s)  "
              f"cent M4={r['central_M4']:.4f}  FA W5={r['family_a_W5_M4']:.4f}  "
              f"Δ={delta:+.1f}%  smooth ratio={ratio:.2f}x  pose-jerk-RMS={r['central_diff_jerk_RMS']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sequence.csv", index=False)
    print()
    if len(df) > 0:
        print(f"Median over {len(df)} sequences:")
        print(f"  cent M4: {df.central_M4.median():.4f}")
        print(f"  FA W5 M4: {df.family_a_W5_M4.median():.4f}")
        print(f"  Δ: {(df.family_a_W5_M4.median()/df.central_M4.median()-1)*100:+.1f}%")
        print(f"  M3sm ratio: {df.central_M3sm.median()/df.family_a_W5_M3sm.median():.2f}x")
        if len(df) > 1:
            rng = np.random.default_rng(42)
            pairs = list(zip(df.central_M4, df.family_a_W5_M4))
            n = len(pairs)
            boot = []
            for _ in range(5000):
                idx = rng.integers(0, n, n)
                a = np.array([pairs[i][0] for i in idx]); b = np.array([pairs[i][1] for i in idx])
                ma = np.median(a)
                if ma > 0:
                    boot.append((np.median(b)/ma - 1) * 100)
            print(f"  95% CI: [{np.percentile(boot,2.5):+.1f}, {np.percentile(boot,97.5):+.1f}]")


if __name__ == "__main__":
    main()
