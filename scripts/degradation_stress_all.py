#!/usr/bin/env python3
"""Pose-degradation stress sweep across the seven audited releases.

For each release the pose stream is subsampled by a downsampling factor,
position noise and timestamp jitter are injected, speed is reconstructed by
central differencing and by the audited probe, and both are compared against the
release's published velocity. The reported quantity is the residual reduction of
the probe relative to central differencing, in percent, at each factor.

The injected degradation is random, so a single draw is a single realization
rather than a property of the release: on the shorter scenes a lone seed moves
the reduction by several points. Each cell is therefore the median over
N_SEEDS independent realizations, and the spread across those realizations is
written alongside it so the stability of each number is visible rather than
implied. Scoring uses the full series, matching the convention in
build_crossds_tables.py; scripts/degradation_convention_check.py reports the
interior-scored variant for comparison.

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
N_SEEDS = 20     # independent degradation realizations per cell
MIN_LEN = 15     # W=5 needs 11 samples; keep a margin for the median


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def load(subdir: str, ref_col: str) -> list[tuple[np.ndarray, ...]]:
    out = []
    for p in sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
        d = pd.read_parquet(p, columns=["t", "x", "y", ref_col])
        out.append(tuple(d[c].to_numpy() for c in ("t", "x", "y", ref_col)))
    return out


def realization(seqs, ds: int, seed: int) -> float:
    """Median per-sequence residual reduction for one degradation draw."""
    rng = np.random.default_rng(seed)
    red = []
    for t0, x0, y0, r0 in seqs:
        t, x, y, r = t0[::ds], x0[::ds], y0[::ds], r0[::ds]
        if len(t) < MIN_LEN:
            continue
        t = np.maximum.accumulate(t + rng.normal(0.0, SIGMA_T, len(t)))
        pose = Pose2D(t=t, x=x + rng.normal(0.0, SIGMA_P, len(x)),
                      y=y + rng.normal(0.0, SIGMA_P, len(y)))
        c = rmse(central_diff(pose), r)
        f = rmse(family_a_pointwise(pose, W=5), r)
        if np.isfinite(c) and c > 0:
            red.append((1.0 - f / c) * 100.0)
    return float(np.median(red)) if red else float("nan")


def main() -> None:
    rows = []
    for name, subdir, ref_col in DATASETS:
        seqs = load(subdir, ref_col)
        if not seqs:
            print(f"{name}: no per-frame streams")
            continue
        cells = []
        for ds in DS_FACTORS:
            draws = np.array([realization(seqs, ds, s) for s in range(N_SEEDS)])
            med = float(np.nanmedian(draws))
            lo, hi = (float(v) for v in np.nanpercentile(draws, [5, 95]))
            rows.append({"dataset": name, "ds": ds, "n_seq": len(seqs),
                         "n_seeds": N_SEEDS, "reduction_pct": med,
                         "reduction_p5": lo, "reduction_p95": hi})
            cells.append(f"{med:.0f}")
        print(f"{name:<10} " + "/".join(cells) + "   (median of "
              f"{N_SEEDS} realizations)")

    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "degradation_stress.csv", index=False)
    print("\nwrote results/degradation_stress.csv")


if __name__ == "__main__":
    main()
