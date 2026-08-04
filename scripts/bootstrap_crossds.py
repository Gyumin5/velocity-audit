#!/usr/bin/env python3
"""Paired bootstrap 95% CI on Family A W=5 vs central differencing per dataset."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(0)


def paired_bootstrap_pct(rmse_central: np.ndarray, rmse_fa: np.ndarray, B: int = 5000):
    """Return median(fa-central)/median(central)*100 with 95% CI via paired resampling."""
    n = len(rmse_central)
    pct = []
    for _ in range(B):
        idx = RNG.integers(0, n, size=n)
        m_c = np.median(rmse_central[idx])
        m_f = np.median(rmse_fa[idx])
        if m_c > 0:
            pct.append((m_f - m_c) / m_c * 100.0)
    pct = np.asarray(pct)
    point = (np.median(rmse_fa) - np.median(rmse_central)) / np.median(rmse_central) * 100.0
    return point, np.percentile(pct, 2.5), np.percentile(pct, 97.5)


def per_unit_pivot(csv: Path, unit_col: str, central_label: str = "central",
                    fa_label: str = "family_a_W5", rmse_col: str | None = None):
    df = pd.read_csv(csv)
    if rmse_col is None:
        for c in ["m8_rmse_vs_ins", "m8_rmse_vs_can", "m8_rmse_vs_oxts",
                  "m8_rmse_vs_pospac", "m8_rmse"]:
            if c in df.columns:
                rmse_col = c
                break
    if rmse_col is None:
        raise RuntimeError(f"No rmse column in {csv}: {list(df.columns)}")
    pv = df.pivot_table(index=unit_col, columns="method", values=rmse_col)
    return pv[central_label].values, pv[fa_label].values


CONFIG = [
    ("HeLiPR",     "results/helipr/summary.csv",        "sequence"),
    ("Oxford RTK", "results/oxford/summary_rtk_x11.csv", "run"),
    ("nuScenes",   "results/nuscenes/summary_x20.csv",   "scene"),
    ("KITTI",      "results/kitti/summary.csv",          "drive"),
    ("Boreas",     "results/boreas/summary.csv",         "sequence"),
]


def main():
    print(f"{'dataset':<12} {'n':>3} {'central':>9} {'famA W=5':>9} "
          f"{'point %':>8} {'lo95':>7} {'hi95':>7}")
    rows = []
    for name, path, unit_col in CONFIG:
        p = Path(path)
        if not p.exists():
            print(f"{name:<12} (missing {path})")
            continue
        try:
            c, f = per_unit_pivot(p, unit_col)
        except Exception as e:
            print(f"{name:<12} error: {e}")
            continue
        c = c[~np.isnan(c)]; f = f[~np.isnan(f)]
        if len(c) < 2:
            print(f"{name:<12} too few samples ({len(c)})")
            continue
        point, lo, hi = paired_bootstrap_pct(c, f)
        med_c = float(np.median(c))
        med_f = float(np.median(f))
        print(f"{name:<12} {len(c):>3d} {med_c:>9.4f} {med_f:>9.4f} "
              f"{point:>+7.1f}% {lo:>+6.1f}% {hi:>+6.1f}%")
        rows.append((name, len(c), med_c, med_f, point, lo, hi))

    out = pd.DataFrame(rows, columns=["dataset", "n", "central", "famA_W5",
                                       "delta_pct", "ci_lo", "ci_hi"])
    out.to_csv("results/cross_dataset_bootstrap_ci.csv", index=False)
    print(f"\nWrote results/cross_dataset_bootstrap_ci.csv")


if __name__ == "__main__":
    main()
