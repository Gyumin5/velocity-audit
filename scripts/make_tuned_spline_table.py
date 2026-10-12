#!/usr/bin/env python3
"""Publish the tuned smoothing spline's per-sequence HeLiPR metrics.

The tuned spline is the one operator in tab:helipr-main with no column in the
published per-frame parquets, so a reader cannot read its row off them the way
the other rows are read.  This rebuilds it from those same parquets, on the
same full stored interval the caption states, and writes the per-sequence
M1-M4 so the printed row is a median a reader can recompute.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from velref.core.trajectory import Pose2D  # noqa: E402
from velref.methods.baselines import smoothing_spline_tuned_deriv  # noqa: E402

SEQS = ["bridge01", "dcc05", "kaist05", "riverside05", "roundabout01", "town01"]
OUT = ROOT / "results" / "helipr" / "tuned_spline_full_interval.csv"


def main() -> int:
    rows = []
    for seq in SEQS:
        d = pd.read_parquet(ROOT / "results" / "helipr" / f"per_frame_{seq}.parquet")
        t, x, y = d["t"].to_numpy(), d["x"].to_numpy(), d["y"].to_numpy()
        vr = d["v_ins"].to_numpy()
        vh = smoothing_spline_tuned_deriv(Pose2D(t=t, x=x, y=y))
        v_int = float(np.sum(0.5 * (vh[1:] + vh[:-1]) * np.diff(t)))
        poly = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        mask = np.abs(vr) < 0.3
        rows.append({
            "sequence": seq,
            "method": "smoothing_spline_tuned",
            "n": len(t),
            "m1_path_len_err": abs(v_int / poly - 1.0),
            "m3_low_speed_rms": float(np.sqrt(np.mean(vh[mask] ** 2))),
            "m4_smooth": float(np.sqrt(np.mean(np.diff(np.diff(vh)) ** 2))),
            "m8_rmse_vs_ins": float(np.sqrt(np.mean((vh - vr) ** 2))),
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    med = df[["m1_path_len_err", "m3_low_speed_rms", "m4_smooth", "m8_rmse_vs_ins"]].median()
    print(df.to_string(index=False))
    print("\nmedians:", {k: float(f"{v:.6g}") for k, v in med.items()})
    print(f"M4 range: {df.m8_rmse_vs_ins.min():.6f} - {df.m8_rmse_vs_ins.max():.6f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
