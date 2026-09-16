#!/usr/bin/env python3
"""Does the near-zero norm bias carry the cross-dataset reading?

A two-component speed norm has a positive expectation at a true speed of zero
that scales with whatever noise the operator leaves, so an operator that leaves
less noise looks better on a stopped segment for a reason that has nothing to do
with the release. M_2 was excluded from the coupling reading for that reason,
but M_4 is computed over the whole evaluation interval and therefore still
contains those samples.

This control recomputes M_4 and Delta with the low-speed samples removed, under
exactly the conventions of build_crossds_tables.py, so the contribution can be
read rather than assumed. The mask is the same one M_2 uses: |v_ref| < 0.3 m/s.

Writes results/lowspeed_exclusion_control.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("HeLiPR",    "helipr",       "v_ins"),
    ("Oxford",    "oxford_x11",   "v_ins"),
    ("nuScenes",  "nuscenes_x20", "v_can"),
    ("KITTI",     "kitti",        "v_oxts"),
    ("KITTI-360", "kitti360",     "v_ref"),
    ("Boreas",    "boreas",       "v_ref"),
    ("Pit30M",    "pit30m_10hz",  "v_ref"),
]

LOW_SPEED_THR = 0.3


def rmse(a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if keep is not None:
        ok &= keep
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def main() -> None:
    rows = []
    for name, subdir, ref_col in DATASETS:
        paths = sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet"))
        if not paths:
            print(f"{name}: no per-frame streams under results/{subdir}")
            continue
        full_c, full_f, cut_c, cut_f = [], [], [], []
        n_low = n_tot = 0
        for p in paths:
            d = pd.read_parquet(p)
            r = d[ref_col].to_numpy()
            c = d["v_central"].to_numpy()
            f = d["v_family_a_W5"].to_numpy()
            keep = np.abs(r) >= LOW_SPEED_THR
            full_c.append(rmse(c, r))
            full_f.append(rmse(f, r))
            cut_c.append(rmse(c, r, keep))
            cut_f.append(rmse(f, r, keep))
            finite = np.isfinite(r)
            n_low += int((~keep & finite).sum())
            n_tot += int(finite.sum())
        mfc, mff = np.median(full_c), np.median(full_f)
        mcc, mcf = np.nanmedian(cut_c), np.nanmedian(cut_f)
        rows.append({
            "dataset": name,
            "N": len(paths),
            "low_speed_frac": n_low / max(1, n_tot),
            "M4_c_full": mfc, "M4_f_full": mff,
            "delta_full_pct": (mff / mfc - 1) * 100,
            "M4_c_cut": mcc, "M4_f_cut": mcf,
            "delta_cut_pct": (mcf / mcc - 1) * 100,
        })
        print(f"{name:<10} N={len(paths):3d}  low-speed {rows[-1]['low_speed_frac']*100:5.1f}%  "
              f"delta {rows[-1]['delta_full_pct']:+7.1f}% -> {rows[-1]['delta_cut_pct']:+7.1f}%")
    out = REPO_ROOT / "results" / "lowspeed_exclusion_control.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
