#!/usr/bin/env python3
"""Re-run the Pit30M audit at a window duration matched to the other releases.

Pit30M's per-frame stream is native 100 Hz, so the fixed setting W=5 spans 0.1 s
there while it spans 1.0 s on the 10 Hz releases. This script recomputes the
Pit30M row under two settings that restore the 1.0 s duration:

  * decimated: stride-10 subsampling to 10 Hz, then W=5   (matches Boreas/nuScenes)
  * native:    100 Hz retained, W=50                      (same duration, no decimation)

and prints the resulting median Delta and smoothness ratio next to the published
W=5-at-100 Hz numbers. Writes results/pit30m_window_duration.csv.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from velref.core.trajectory import Pose2D
from velref.methods.family_a import family_a_pointwise
from velref.methods.baselines import central_diff


def interior(n: int) -> slice:
    m = max(10, n // 50)
    return slice(m, -m)


def metrics(v_hat: np.ndarray, v_ref: np.ndarray) -> tuple[float, float]:
    sl = interior(len(v_hat))
    a, b = v_hat[sl], v_ref[sl]
    m4 = float(np.sqrt(np.mean((a - b) ** 2)))
    d2 = np.diff(np.diff(a))
    return m4, float(np.sqrt(np.mean(d2 ** 2)))


def main() -> None:
    rows = []
    for f in sorted((REPO_ROOT / "results" / "pit30m").glob("per_frame_*.parquet")):
        df = pd.read_parquet(f)
        if len(df) < 600:
            continue
        t, x, y = (df[c].to_numpy() for c in ("t", "x", "y"))
        v_ref = df["v_ref"].to_numpy()
        row = {"sequence": f.stem.replace("per_frame_", ""), "n": len(df)}

        # published setting: native 100 Hz, W=5 (0.1 s)
        row["m4_c_pub"], row["m3_c_pub"] = metrics(central_diff(Pose2D(t=t, x=x, y=y)), v_ref)
        row["m4_f_pub"], row["m3_f_pub"] = metrics(
            family_a_pointwise(Pose2D(t=t, x=x, y=y), W=5), v_ref)

        # matched duration, native rate: W=50 (1.0 s)
        row["m4_f_w50"], row["m3_f_w50"] = metrics(
            family_a_pointwise(Pose2D(t=t, x=x, y=y), W=50), v_ref)

        # matched duration, decimated to 10 Hz: W=5 (1.0 s)
        td, xd, yd, vd = t[::10], x[::10], y[::10], v_ref[::10]
        pd10 = Pose2D(t=td, x=xd, y=yd)
        row["m4_c_dec"], row["m3_c_dec"] = metrics(central_diff(pd10), vd)
        row["m4_f_dec"], row["m3_f_dec"] = metrics(family_a_pointwise(pd10, W=5), vd)
        rows.append(row)

    d = pd.DataFrame(rows)
    d.to_csv(REPO_ROOT / "results" / "pit30m_window_duration.csv", index=False)
    med = d.median(numeric_only=True)

    def line(tag, mc, mf, sc, sf):
        delta = (med[mf] / med[mc] - 1) * 100
        print(f"{tag:<34} central M4 {med[mc]:.5f}  FA M4 {med[mf]:.5f}  "
              f"Delta {delta:+6.2f}%   smooth ratio {med[sc]/med[sf]:.2f}x")

    print(f"{len(d)} segments\n")
    line("published (100 Hz, W=5, 0.1 s)", "m4_c_pub", "m4_f_pub", "m3_c_pub", "m3_f_pub")
    line("native   (100 Hz, W=50, 1.0 s)", "m4_c_pub", "m4_f_w50", "m3_c_pub", "m3_f_w50")
    line("decimated (10 Hz, W=5, 1.0 s)", "m4_c_dec", "m4_f_dec", "m3_c_dec", "m3_f_dec")


if __name__ == "__main__":
    main()
