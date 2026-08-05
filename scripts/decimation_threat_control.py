#!/usr/bin/env python3
"""Push the aliasing threat further on Boreas and nuScenes.

Three of the seven releases are published above the 10 Hz analysis cadence and
are decimated to it by stride subsampling with no anti-aliasing prefilter, so
power above the 5 Hz analysis Nyquist limit folds back into both operators'
outputs. The direct control for that is to remove the decimation instead of
adding to it, which scripts/native_window_control.py and
scripts/pit30m_native_window_control.py do for all three releases; those are
what the paper reports.

This script probes the same threat from the opposite side. Decimating the 10 Hz
series again -- to 5 Hz, and to 3.33 Hz -- folds strictly more power back, at
every available phase, while the window duration is held near 1.0 s by widening
W. If stride aliasing were producing the reported sign, adding more of it should
not leave the reading intact.

One caveat is visible in the output rather than assumed: at p=3 a setting with
W<3 leaves a single degree of freedom, so the "probe" interpolates instead of
smoothing. Those rows are identifiable by a smoothness ratio below one and are
not evidence about the release.

Writes results/decimation_threat_control.csv.
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

DATASETS = [("Boreas", "boreas", "v_ref"), ("nuScenes", "nuscenes_x20", "v_can")]
# (stride, W): W chosen so (2W+1)/rate stays as close to 1.0 s as an integer allows
SETTINGS = [(1, 5), (1, 7), (2, 2), (2, 3), (2, 5), (3, 2), (3, 3)]
SHIFTS = (-0.5, -0.2, -0.1, 0.1, 0.2, 0.5)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def smoothness(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2))) if len(v) > 3 else float("nan")


def main() -> None:
    rows = []
    for name, subdir, ref_col in DATASETS:
        files = sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet"))
        raw = [pd.read_parquet(f, columns=["t", "x", "y", ref_col]) for f in files]
        print(f"\n{name}  ({len(raw)} sequences)")
        for stride, W in SETTINGS:
            for phase in range(stride):
                cs, fs, m3c, m3f, lat = [], [], [], [], {s: [] for s in SHIFTS}
                span = None
                for d in raw:
                    t = d["t"].to_numpy()[phase::stride]
                    x = d["x"].to_numpy()[phase::stride]
                    y = d["y"].to_numpy()[phase::stride]
                    r = d[ref_col].to_numpy()[phase::stride]
                    if len(t) < 2 * W + 11:
                        continue
                    span = 2 * W * float(np.median(np.diff(t)))  # first sample to last
                    pose = Pose2D(t=t, x=x, y=y)
                    c, f = central_diff(pose), family_a_pointwise(pose, W=W)
                    cs.append(rmse(c, r)); fs.append(rmse(f, r))
                    m3c.append(smoothness(c)); m3f.append(smoothness(f))
                    for s in SHIFTS:
                        rs = np.interp(t, t + s, r, left=np.nan, right=np.nan)
                        lat[s].append((rmse(c, rs), rmse(f, rs)))
                if not cs:
                    continue
                mc, mf = float(np.median(cs)), float(np.median(fs))
                delta = (mf / mc - 1) * 100
                worst = max(abs(float(np.median([b for _, b in v]))
                                / float(np.median([a for a, _ in v])) - 1) * 100
                            for v in lat.values())
                rows.append({"dataset": name, "stride": stride, "phase": phase, "W": W,
                             "rate_hz": round(10.0 / stride, 2), "span_s": round(span, 2),
                             "delta_pct": delta, "smooth_x": np.median(m3c) / np.median(m3f),
                             "max_abs_delta_any_shift_pct": worst})
                print(f"  {10.0/stride:5.2f} Hz  W={W}  span {span:.2f}s   "
                      f"Delta {delta:+7.2f}%   smooth {np.median(m3c)/np.median(m3f):.2f}x   "
                      f"worst non-zero shift {worst:+6.2f}%")

    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "decimation_threat_control.csv",
                              index=False)
    print("\nwrote results/decimation_threat_control.csv")


if __name__ == "__main__":
    main()
