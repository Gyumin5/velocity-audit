#!/usr/bin/env python3
"""Control experiment: does the Pit30M result survive without decimation?

The audit fixes the probe's window by duration, not by sample count. Pit30M
publishes at 100 Hz, so reaching the 1.0 s window used on the 10 Hz releases can
be done two ways, and the paper's tables use the first:

  decimated  stride-10 to 10 Hz, then W=5   (results/pit30m_10hz/)
  native     keep 100 Hz, widen to W=50     (results/pit30m/)

If the reported behaviour were an artifact of throwing away 90% of the samples,
the two would disagree. This script recomputes all three qualitative claims --
the sign of the M_4 change, the collapse of that change under a latency shift,
and the direction of the per-curvature-bin residual change -- on the native-rate
series, so the decimated result can be checked against a variant that decimates
nothing.

Conventions match scripts/build_crossds_tables.py (median per segment, then
ratio of medians) and scripts/curvature_bins_all_datasets.py (same bin edges).

Writes results/pit30m_native_control.csv.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from velref.core.curvature import estimate_curvature  # noqa: E402
from velref.core.trajectory import Pose2D  # noqa: E402
from velref.methods.family_a import family_a_pointwise  # noqa: E402
from velref.methods.baselines import central_diff  # noqa: E402

SHIFTS = (-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5)
BIN_EDGES = np.array([0.0, 1e-3, 5e-3, 2e-2, 0.1, 1.0])
BIN_LABELS = [f"{BIN_EDGES[i]:.0e}-{BIN_EDGES[i + 1]:.0e}" for i in range(len(BIN_EDGES) - 1)]
MIN_SAMPLES = 20
W_NATIVE = 50  # 100 Hz x (2*50+1) samples = 1.01 s, the analysis window duration


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def smoothness(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


def main() -> None:
    files = sorted((REPO_ROOT / "results" / "pit30m").glob("per_frame_*.parquet"))
    if not files:
        print("no native-rate Pit30M streams under results/pit30m")
        return

    seqs = []
    for p in files:
        d = pd.read_parquet(p)
        t, x, y = (d[c].to_numpy() for c in ("t", "x", "y"))
        if len(t) < 2 * W_NATIVE + 11:
            continue
        pose = Pose2D(t=t, x=x, y=y)
        seqs.append((p.stem.replace("per_frame_", ""), t, x, y,
                     d["v_ref"].to_numpy(), central_diff(pose),
                     family_a_pointwise(pose, W=W_NATIVE)))

    rows = []

    # --- claim 1: sign of the M_4 change ---
    m4c = np.array([rmse(c, r) for _, _, _, _, r, c, _ in seqs])
    m4f = np.array([rmse(f, r) for _, _, _, _, r, _, f in seqs])
    med_c, med_f = float(np.median(m4c)), float(np.median(m4f))
    delta0 = (med_f / med_c - 1) * 100
    m3c = float(np.median([smoothness(c) for _, _, _, _, _, c, _ in seqs]))
    m3f = float(np.median([smoothness(f) for _, _, _, _, _, _, f in seqs]))
    rows.append({"check": "alignment", "key": "delta_pct", "value": delta0})
    rows.append({"check": "alignment", "key": "smooth_x", "value": m3c / m3f})
    print(f"{len(seqs)} segments at native 100 Hz, W={W_NATIVE}")
    print(f"  M4 {med_c:.5f} -> {med_f:.5f}   Delta {delta0:+.2f}%   "
          f"smooth {m3c / m3f:.2f}x")

    # --- claim 2: the change collapses under any non-zero latency shift ---
    print("  latency sweep:")
    for s in SHIFTS:
        sc, sf = [], []
        for _, t, _, _, r, c, f in seqs:
            r_shift = np.interp(t, t + s, r, left=np.nan, right=np.nan)
            sc.append(rmse(c, r_shift))
            sf.append(rmse(f, r_shift))
        mc, mf = float(np.median(sc)), float(np.median(sf))
        d = (mf / mc - 1) * 100
        rows.append({"check": "latency", "key": f"shift_{s:+.1f}s", "value": d})
        print(f"    {s:+.1f}s  Delta {d:+7.2f}%")

    # --- claim 3: the residual is larger in every curvature bin ---
    print("  curvature bins:")
    per_bin: dict[str, list[tuple[float, float]]] = {b: [] for b in BIN_LABELS}
    for _, t, x, y, r, c, f in seqs:
        kappa = np.abs(estimate_curvature(Pose2D(t=t, x=x, y=y), window=9, polyorder=3))
        bins = np.clip(np.digitize(kappa, BIN_EDGES) - 1, 0, len(BIN_LABELS) - 1)
        ok = np.isfinite(c) & np.isfinite(f) & np.isfinite(r)
        for b, label in enumerate(BIN_LABELS):
            mask = (bins == b) & ok
            if mask.sum() < MIN_SAMPLES:
                continue
            per_bin[label].append((rmse(c[mask], r[mask]), rmse(f[mask], r[mask])))
    for label in BIN_LABELS:
        vals = per_bin[label]
        if not vals:
            continue
        mc = float(np.median([v[0] for v in vals]))
        mf = float(np.median([v[1] for v in vals]))
        d = (mf / mc - 1) * 100
        rows.append({"check": "curvature", "key": label, "value": d})
        print(f"    kappa {label:<12} n_seq {len(vals):3d}  Delta {d:+7.2f}%")

    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "pit30m_native_control.csv",
                              index=False)
    print("\nwrote results/pit30m_native_control.csv")


if __name__ == "__main__":
    main()
