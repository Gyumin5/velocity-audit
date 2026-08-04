#!/usr/bin/env python3
"""Pose-degradation stress sweep across the seven audited releases.

For each release the pose stream is subsampled by a downsampling factor,
position noise and timestamp jitter are injected, speed is reconstructed by
central differencing and by the audited probe, and both are compared against the
release's published velocity. The reported quantity is the residual reduction of
the probe relative to central differencing, in percent, at each factor.

Pit30M is read from results/pit30m_10hz/, the 10 Hz analysis series, so its
probe window spans the same 1.0 s as on the releases published at 10 Hz.

Writes results/degradation_stress.csv.
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
    ("HeLiPR",    "helipr",       "v_ins"),
    ("Oxford",    "oxford_x11",   "v_ins"),
    ("nuScenes",  "nuscenes_x20", "v_can"),
    ("KITTI raw", "kitti",        "v_oxts"),
    ("KITTI-360", "kitti360",     "v_ref"),
    ("Boreas",    "boreas",       "v_ref"),
    ("Pit30M",    "pit30m_10hz",  "v_ref"),
]

DS_FACTORS = (1, 2, 5, 10)
SIGMA_P = 0.2    # position noise [m]
SIGMA_T = 0.005  # timestamp jitter [s]
SEED = 0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def main() -> None:
    rows = []
    for name, subdir, ref_col in DATASETS:
        files = sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet"))
        if not files:
            print(f"{name}: no per-frame streams")
            continue
        for ds in DS_FACTORS:
            rng = np.random.default_rng(SEED)
            red = []
            for p in files:
                d = pd.read_parquet(p)
                t = d["t"].to_numpy()[::ds]
                x = d["x"].to_numpy()[::ds]
                y = d["y"].to_numpy()[::ds]
                r = d[ref_col].to_numpy()[::ds]
                # W=5 needs 11 samples; keep a margin so the interior of a
                # subsampled scene still carries a usable median.
                if len(t) < 15:
                    continue
                t = t + rng.normal(0.0, SIGMA_T, len(t))
                t = np.maximum.accumulate(t)
                x = x + rng.normal(0.0, SIGMA_P, len(x))
                y = y + rng.normal(0.0, SIGMA_P, len(y))
                pose = Pose2D(t=t, x=x, y=y)
                c = rmse(central_diff(pose), r)
                f = rmse(family_a_pointwise(pose, W=5), r)
                if np.isfinite(c) and c > 0:
                    red.append((1.0 - f / c) * 100.0)
            if red:
                rows.append({"dataset": name, "ds": ds, "n_seq": len(red),
                             "reduction_pct": float(np.median(red))})
        vals = [r["reduction_pct"] for r in rows if r["dataset"] == name]
        print(f"{name:<10} " + "/".join(f"{v:.0f}" for v in vals))

    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "degradation_stress.csv", index=False)
    print("\nwrote results/degradation_stress.csv")


if __name__ == "__main__":
    main()
