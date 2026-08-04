#!/usr/bin/env python3
"""Process Pit30M all_poses.npz: split each log at large dt gaps into segments,
process each segment separately. Pose from continuous.x,y; velocity from vx,vy.
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


def low_speed_rms(v, threshold=0.5):
    mask = np.abs(v) < threshold
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean(v[mask] ** 2)))


def smoothness_rms2nd(v):
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


GAP_THR = 0.05  # 50 ms gap → split (data is 100 Hz)
MIN_SEG_FRAMES = 1000  # require at least 10 s of contiguous frames at 100 Hz


def process_segment(t, x, y, v_horiz, seq_id):
    # Sort by t
    order = np.argsort(t)
    t = t[order]; x = x[order]; y = y[order]; v_horiz = v_horiz[order]
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t = t[keep]; x = x[keep]; y = y[keep]; v_horiz = v_horiz[keep]
    # Downsample 100 Hz → 10 Hz
    target_dt = 0.1
    last = t[0]
    keep_idx = [0]
    for i in range(1, len(t)):
        if t[i] - last >= target_dt - 1e-3:
            keep_idx.append(i)
            last = t[i]
    keep_idx = np.array(keep_idx)
    t = t[keep_idx]; x = x[keep_idx]; y = y[keep_idx]; v_horiz = v_horiz[keep_idx]
    if len(t) < 30:
        return None
    pose = Pose2D(t, x, y)
    methods = {
        "central": central_diff(pose),
        "family_a_W5": family_a_pointwise(pose, W=5),
        "family_a_W7": family_a_pointwise(pose, W=7),
    }
    out = {"sequence": seq_id, "n_frames": len(t)}
    for name, v in methods.items():
        v = np.asarray(v, dtype=float)
        if v.ndim > 1:
            v = np.linalg.norm(v, axis=-1) if v.shape[-1] in (2, 3) else v.squeeze()
        out[f"{name}_M2"] = low_speed_rms(v)
        out[f"{name}_M3sm"] = smoothness_rms2nd(v)
        out[f"{name}_M4"] = rmse(v, v_horiz)
    return out


def split_log(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)["data"]
    valid = d["continuous"]["valid"] & d["poses_and_differentials_valid"]
    d = d[valid]
    if len(d) < MIN_SEG_FRAMES:
        return []
    t = d["capture_time"]
    cont = d["continuous"]
    x = cont["x"]; y = cont["y"]
    v_horiz = np.hypot(cont["vx"], cont["vy"])
    order = np.argsort(t)
    t = t[order]; x = x[order]; y = y[order]; v_horiz = v_horiz[order]
    # Find gap boundaries
    dt = np.diff(t)
    gaps = np.where(dt > GAP_THR)[0]
    boundaries = [0] + (gaps + 1).tolist() + [len(t)]
    segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= MIN_SEG_FRAMES:
            segments.append((t[s:e], x[s:e], y[s:e], v_horiz[s:e]))
    return segments


def main():
    base = Path("/mnt/Data/velref/pit30m")
    out_dir = _root / "results" / "pit30m"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fp in sorted(base.glob("*.npz")):
        try:
            segs = split_log(fp)
            for j, (t, x, y, v_horiz) in enumerate(segs):
                r = process_segment(t, x, y, v_horiz, f"{fp.stem[:8]}_{j:02d}")
                if r:
                    rows.append(r)
        except Exception as e:
            print(f"  {fp.stem[:8]}: FAIL {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sequence.csv", index=False)
    print(f"[Pit30M] {len(df)} clean segments -> {out_dir}")
    if len(df) == 0:
        return
    print(f"medians: cent M4={df.central_M4.median():.4f}  FA W5={df.family_a_W5_M4.median():.4f}  "
          f"Δ={(df.family_a_W5_M4.median()/df.central_M4.median()-1)*100:+.1f}%")
    print(f"         cent M3sm={df.central_M3sm.median():.4f}  FA W5={df.family_a_W5_M3sm.median():.4f}  "
          f"ratio={df.central_M3sm.median()/df.family_a_W5_M3sm.median():.2f}x")
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
    print(f"95% CI: [{np.percentile(boot,2.5):.1f}, {np.percentile(boot,97.5):.1f}]")
    print(f"\nFirst 10 segments:")
    print(df.head(10)[['sequence','n_frames','central_M4','family_a_W5_M4','central_M3sm','family_a_W5_M3sm']].to_string())


if __name__ == "__main__":
    main()
