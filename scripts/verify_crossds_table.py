#!/usr/bin/env python3
"""Recompute every row of Table crossds from the surviving per-frame streams.

The Pit30M row turned out to disagree with the repository's own processing
script, so this checks whether the other six rows reproduce. It reads only the
committed per-frame parquets and applies the paper's stated aggregation:
median M_4 per sequence, then the ratio of medians, with the interior slice on
the five releases that use it and the full series on KITTI-360 and Pit30M.

Read-only. Writes results/crossds_verify.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# (label, results subdir, reference column, uses interior slice, paper's Delta [%])
DATASETS = [
    ("HeLiPR",    "helipr",       "v_ins",   True,  -46.0),
    ("Oxford",    "oxford_x11",   "v_ins",   True,  -40.0),
    ("nuScenes",  "nuscenes_x20", "v_can",   True,   -6.0),
    ("KITTI raw", "kitti",        "v_oxts",  True,   -2.0),
    ("KITTI-360", "kitti360",     "v_ref",   False,  -0.0),
    ("Boreas",    "boreas",       "v_ref",   True,  102.0),
    ("Pit30M",    "pit30m",       "v_ref",   False,   0.0),
]


def interior(n: int) -> slice:
    m = max(10, n // 50)
    return slice(m, -m)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def main() -> None:
    rows = []
    for label, subdir, ref_col, use_interior, paper in DATASETS:
        per_seq = []
        for f in sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
            d = pd.read_parquet(f)
            sl = interior(len(d)) if use_interior else slice(None)
            v_ref = d[ref_col].to_numpy()[sl]
            per_seq.append((
                rmse(d["v_central"].to_numpy()[sl], v_ref),
                rmse(d["v_family_a_W5"].to_numpy()[sl], v_ref),
            ))
        if not per_seq:
            print(f"{label:<10} no per-frame files")
            continue
        arr = np.array(per_seq)
        c, fa = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        delta = (fa / c - 1) * 100
        rows.append({"dataset": label, "n_seq": len(arr), "central_M4": c,
                     "family_a_M4": fa, "delta_pct": delta, "paper_pct": paper,
                     "diff_pp": delta - paper})
        flag = "OK" if abs(delta - paper) <= 1.5 else "MISMATCH"
        print(f"{label:<10} n={len(arr):3d}  central {c:.4f}  FA {fa:.4f}  "
              f"Delta {delta:+7.2f}%   paper {paper:+6.1f}%   {flag}")

    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / "results" / "crossds_verify.csv", index=False)


if __name__ == "__main__":
    main()
