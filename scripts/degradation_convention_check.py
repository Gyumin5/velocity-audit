#!/usr/bin/env python3
"""Compare the two evaluation conventions for the pose-degradation sweep.

The sweep in scripts/run_analytics.py::experiment_degradation scores on the
interior of each sequence and re-interpolates the reference onto the jittered
timestamps; scripts/degradation_stress_all.py scores on the full series and
leaves the reference at its own timestamps. Both are defensible, and this script
runs them side by side so the reported claim can be checked against each rather
than resting on the choice.

Prints the probe's residual reduction relative to central differencing, in
percent, at each downsampling factor under both conventions.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from velref.core.trajectory import Pose2D  # noqa: E402
from velref.methods.family_a import family_a_pointwise  # noqa: E402
from velref.methods.baselines import central_diff  # noqa: E402

DATASETS = [
    ("HeLiPR", "helipr", "v_ins"), ("Oxford", "oxford_x11", "v_ins"),
    ("nuScenes", "nuscenes_x20", "v_can"), ("KITTI raw", "kitti", "v_oxts"),
    ("KITTI-360", "kitti360", "v_ref"), ("Boreas", "boreas", "v_ref"),
    ("Pit30M", "pit30m_10hz", "v_ref"),
]
DS_FACTORS = (1, 2, 5, 10)
SIGMA_P, SIGMA_T, SEED = 0.2, 0.005, 0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def reduction(name: str, subdir: str, ref_col: str, ds: int, interior: bool) -> float:
    rng = np.random.default_rng(SEED)
    red = []
    for p in sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
        d = pd.read_parquet(p)
        t0 = d["t"].to_numpy()[::ds]
        x = d["x"].to_numpy()[::ds] + rng.normal(0.0, SIGMA_P, len(t0))
        y = d["y"].to_numpy()[::ds] + rng.normal(0.0, SIGMA_P, len(t0))
        r0 = d[ref_col].to_numpy()[::ds]
        t = np.maximum.accumulate(t0 + rng.normal(0.0, SIGMA_T, len(t0)))
        if len(t) < (30 if interior else 15):
            continue
        # interior convention re-interpolates the reference onto the jittered
        # clock, matching run_analytics; the other leaves it where it was read
        r = np.interp(t, t0, r0) if interior else r0
        pose = Pose2D(t=t, x=x, y=y)
        c, f = central_diff(pose), family_a_pointwise(pose, W=5)
        if interior:
            m = max(10, len(t) // 50)
            sl = slice(m, -m)
            c, f, r = c[sl], f[sl], r[sl]
        rc, rf = rmse(c, r), rmse(f, r)
        if np.isfinite(rc) and rc > 0:
            red.append((1.0 - rf / rc) * 100.0)
    return float(np.median(red)) if red else float("nan")


def main() -> None:
    print(f"{'dataset':<11} {'ds':>3}   full-series   interior")
    rows = []
    for name, subdir, ref_col in DATASETS:
        for ds in DS_FACTORS:
            a = reduction(name, subdir, ref_col, ds, interior=False)
            b = reduction(name, subdir, ref_col, ds, interior=True)
            rows.append({"dataset": name, "ds": ds, "full_series": a, "interior": b})
            print(f"{name:<11} {ds:>3}   {a:>10.0f}   {b:>9.0f}")
    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "degradation_conventions.csv",
                              index=False)
    print("\nwrote results/degradation_conventions.csv")


if __name__ == "__main__":
    main()
