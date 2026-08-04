#!/usr/bin/env python3
"""Regenerate the cross-dataset tables and the latency sweep from the per-frame streams.

This is the generator behind results/crossds_recomputed.csv,
results/multimetric_recomputed.csv and results/latency_sweep.csv, which are the
sources for the cross-dataset table, the multi-metric table, and the latency
paragraph. It reads only the per-frame streams under results/<release>/ and
applies the aggregation the paper states: a median per sequence, then the ratio
of medians across a release, with a paired sequence-level bootstrap for the
spread column.

Pit30M is read from results/pit30m_10hz/, the 10 Hz analysis series built by
scripts/pit30m_build_10hz.py, so that its probe window spans the same 1.0 s as
the six releases published at 10 Hz.

Run with --check to verify the regenerated numbers against the committed CSVs
instead of overwriting them.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# display name -> (results subdirectory, published-velocity column, regime label)
DATASETS = [
    ("HeLiPR",    "helipr",       "v_ins",  "separated"),
    ("Oxford",    "oxford_x11",   "v_ins",  "alg.\\ sep."),
    ("nuScenes",  "nuscenes_x20", "v_can",  "weak.\\ coup."),
    ("KITTI",     "kitti",        "v_oxts", "online"),
    ("KITTI-360", "kitti360",     "v_ref",  "online"),
    ("Boreas",    "boreas",       "v_ref",  "batch joint"),
    ("Pit30M",    "pit30m_10hz",  "v_ref",  "batch joint"),
]

SHIFTS = (-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5)
LOW_SPEED_THR = 0.3
BOOT = 5000


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def low_speed_rms(v_hat: np.ndarray, v_ref: np.ndarray) -> float:
    mask = np.isfinite(v_hat) & (np.abs(v_ref) < LOW_SPEED_THR)
    return float(np.sqrt(np.mean(v_hat[mask] ** 2))) if mask.sum() >= 5 else float("nan")


def smoothness(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


def paired_bootstrap(c: np.ndarray, f: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(c)
    pct = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        mc = np.median(c[idx])
        if mc > 0:
            pct.append((np.median(f[idx]) - mc) / mc * 100.0)
    return float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5))


def load(subdir: str, ref_col: str):
    """Yield (t, v_ref, v_central, v_family_a_W5) per sequence of a release."""
    for p in sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
        d = pd.read_parquet(p)
        yield (d["t"].to_numpy(), d[ref_col].to_numpy(),
               d["v_central"].to_numpy(), d["v_family_a_W5"].to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed CSVs instead of writing")
    args = ap.parse_args()

    cross, multi, lat = [], [], []
    for name, subdir, ref_col, regime in DATASETS:
        seqs = list(load(subdir, ref_col))
        if not seqs:
            print(f"{name}: no per-frame streams under results/{subdir}")
            continue

        m4c = np.array([rmse(c, r) for _, r, c, _ in seqs])
        m4f = np.array([rmse(f, r) for _, r, _, f in seqs])
        lo, hi = paired_bootstrap(m4c, m4f)
        med_c, med_f = float(np.median(m4c)), float(np.median(m4f))
        cross.append({"dataset": name, "N": len(seqs), "central": med_c,
                      "fa_W5": med_f, "delta_pct": (med_f / med_c - 1) * 100,
                      "ci_lo": lo, "ci_hi": hi, "regime": regime})

        # A sequence with no samples below the low-speed threshold contributes no
        # M_2 value; those are skipped rather than propagated as NaN.
        m2c = np.nanmedian([low_speed_rms(c, r) for _, r, c, _ in seqs])
        m2f = np.nanmedian([low_speed_rms(f, r) for _, r, _, f in seqs])
        m3c = np.median([smoothness(c) for _, _, c, _ in seqs])
        m3f = np.median([smoothness(f) for _, _, _, f in seqs])
        multi.append({"dataset": name, "N": len(seqs), "M2_c": m2c, "M2_f": m2f,
                      "M3_c": m3c, "M3_f": m3f, "M4_c": med_c, "M4_f": med_f,
                      "smooth_x": m3c / m3f})

        for s in SHIFTS:
            sc, sf = [], []
            for t, r, c, f in seqs:
                r_shift = np.interp(t, t + s, r, left=np.nan, right=np.nan)
                sc.append(rmse(c, r_shift))
                sf.append(rmse(f, r_shift))
            mc, mf = float(np.median(sc)), float(np.median(sf))
            lat.append({"dataset": subdir.split("_")[0], "shift_s": s,
                        "median_M4_central": mc, "median_M4_FA": mf,
                        "delta_pct": (mf / mc - 1) * 100, "N": len(seqs)})

        print(f"{name:<10} N={len(seqs):3d}  M4 {med_c:.4f} -> {med_f:.4f}  "
              f"Delta {(med_f/med_c-1)*100:+7.2f}%   smooth {m3c/m3f:.2f}x")

    out = {
        "crossds_recomputed.csv": pd.DataFrame(cross),
        "multimetric_recomputed.csv": pd.DataFrame(multi),
        "latency_sweep.csv": pd.DataFrame(lat),
    }
    if args.check:
        for fn, new in out.items():
            old = pd.read_csv(REPO_ROOT / "results" / fn)
            print(f"\n--- {fn} ---")
            key = "dataset"
            for _, row in new.iterrows():
                m = old[old[key].str.lower() == str(row[key]).lower()]
                if m.empty:
                    print(f"  {row[key]}: not in committed file")
        return

    for fn, df in out.items():
        df.to_csv(REPO_ROOT / "results" / fn, index=False)
    print(f"\nwrote {', '.join(out)}")


if __name__ == "__main__":
    main()
