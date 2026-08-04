#!/usr/bin/env python3
"""Stride-offset sensitivity for the Pit30M 10 Hz analysis series.

Decimating the archived 100 Hz per-frame streams to the 10 Hz analysis cadence
requires choosing which of the ten phases to keep. The phase must be fixed by a
rule rather than by the result it produces, so this sweeps all ten offsets and
reports the spread. If the spread is wide, a single number cannot be reported
without a sensitivity band.

Read-only with respect to the published results. Writes
results/pit30m_stride_offset.csv.
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

STRIDE = 10  # 100 Hz -> 10 Hz


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def smooth(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


def main() -> None:
    files = sorted((REPO_ROOT / "results" / "pit30m").glob("per_frame_*.parquet"))
    rows = []
    for off in range(STRIDE):
        per_seq = []
        for f in files:
            d = pd.read_parquet(f)
            t = d["t"].to_numpy()[off::STRIDE]
            x = d["x"].to_numpy()[off::STRIDE]
            y = d["y"].to_numpy()[off::STRIDE]
            vr = d["v_ref"].to_numpy()[off::STRIDE]
            if len(t) < 60:
                continue
            pose = Pose2D(t=t, x=x, y=y)
            vc, vf = central_diff(pose), family_a_pointwise(pose, W=5)
            per_seq.append((rmse(vc, vr), rmse(vf, vr), smooth(vc), smooth(vf)))
        a = np.array(per_seq)
        c, fa = float(np.median(a[:, 0])), float(np.median(a[:, 1]))
        sc, sf = float(np.median(a[:, 2])), float(np.median(a[:, 3]))
        rows.append({"offset": off, "n_seq": len(a), "central_M4": c,
                     "family_a_M4": fa, "delta_pct": (fa / c - 1) * 100,
                     "smooth_ratio": sc / sf})
        print(f"offset {off}: n={len(a):3d}  central {c:.4f}  FA {fa:.4f}  "
              f"Delta {(fa/c-1)*100:+7.2f}%   smooth {sc/sf:.2f}x")

    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / "results" / "pit30m_stride_offset.csv", index=False)
    d = out.delta_pct
    print(f"\nDelta across offsets: min {d.min():+.2f}%  max {d.max():+.2f}%  "
          f"median {d.median():+.2f}%  spread {d.max()-d.min():.2f} pp")
    r = out.smooth_ratio
    print(f"smooth ratio: {r.min():.2f}x -- {r.max():.2f}x (median {r.median():.2f}x)")


if __name__ == "__main__":
    main()
