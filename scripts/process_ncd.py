#!/usr/bin/env python3
"""Process Newer College Dataset GT poses (TUM format + registered_poses.csv).

NCD ships ICP-to-prior-map poses only; no published velocity. Provenance regime
is 'BATCH GT, handheld' — different from any vehicle dataset. Used as smoothness-
only external check.
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


def load_tum(fp: Path):
    """TUM format: time(sec) tx ty tz qx qy qz qw"""
    arr = np.loadtxt(fp)
    t = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]
    return t, x, y


def load_registered(fp: Path):
    """NCD registered_poses.csv: #sec,nsec,x,y,z,qx,qy,qz,qw"""
    df = pd.read_csv(fp, comment="@")  # header is "#sec,nsec,..."
    if df.columns[0].startswith("#"):
        df.columns = [c.lstrip("#") for c in df.columns]
    t = df["sec"].to_numpy() + df["nsec"].to_numpy() / 1e9
    return t, df["x"].to_numpy(), df["y"].to_numpy()


def process_one(fp: Path):
    if fp.name.endswith("_tum.csv"):
        t, x, y = load_tum(fp)
    else:
        t, x, y = load_registered(fp)
    order = np.argsort(t)
    t = t[order]; x = x[order]; y = y[order]
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t = t[keep]; x = x[keep]; y = y[keep]
    if len(t) < 30:
        return None
    # NCD pose at ~10 Hz already; no need to downsample
    pose = Pose2D(t, x, y)
    methods = {
        "central": central_diff(pose),
        "family_a_W5": family_a_pointwise(pose, W=5),
        "family_a_W7": family_a_pointwise(pose, W=7),
    }
    out = {"sequence": fp.stem, "n_frames": len(t)}
    for name, v in methods.items():
        v = np.asarray(v, dtype=float)
        if v.ndim > 1:
            v = np.linalg.norm(v, axis=-1) if v.shape[-1] in (2, 3) else v.squeeze()
        out[f"{name}_M2"] = low_speed_rms(v)
        out[f"{name}_M3sm"] = smoothness_rms2nd(v)
    return out


def main():
    base = Path("/mnt/Data/velref/ncd/gt")
    out_dir = _root / "results" / "ncd"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fp in sorted(base.glob("*.csv")):
        try:
            r = process_one(fp)
            if r:
                ratio = r['central_M3sm'] / r['family_a_W5_M3sm'] if r['family_a_W5_M3sm'] > 0 else float('nan')
                rows.append(r)
                print(f"  {fp.stem[:35]:<35}: n={r['n_frames']:5d}  M3sm cent={r['central_M3sm']:.4f}  FA={r['family_a_W5_M3sm']:.4f}  ratio={ratio:.2f}x")
        except Exception as e:
            print(f"  {fp.stem}: FAIL {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sequence.csv", index=False)
    print(f"\n[NCD] {len(df)} sequences -> {out_dir}")
    if len(df) > 0:
        print(f"medians: cent M3sm={df.central_M3sm.median():.4f}  FA W5={df.family_a_W5_M3sm.median():.4f}  "
              f"ratio={df.central_M3sm.median()/df.family_a_W5_M3sm.median():.2f}x")


if __name__ == "__main__":
    main()
