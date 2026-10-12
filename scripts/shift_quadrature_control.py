#!/usr/bin/env python3
"""Controlled test of the alternative explanation for the time-shift sweep.

Reviewer 1 (comment 1) and reviewer 2 (comment 1) of the second IEEE Access
round raise the same objection: shifting the published velocity in time injects
an error that both differentiation operators inherit, so if that common error
grows past the original residuals the relative difference Delta collapses toward
zero whether or not the two channels share a smoother. Under that reading the
collapse is arithmetic, not evidence of shared estimation.

This script tests the objection two ways and writes one table per test.

1. Quadrature prediction on the released streams.  If the shift injects an error
   e(s) that is independent of the operator, then each operator's residual at
   shift s should be sqrt(r(0)^2 + e(s)^2).  We take e(s) from the central
   operator alone --- e(s)^2 = r_c(s)^2 - r_c(0)^2 --- and then *predict* the
   Family A residual and Delta at that shift with no further fitting.  Comparing
   the predicted Delta to the measured one says how much of the sweep the
   objection accounts for.

2. Synthetic control with no shared smoother.  Pose and published velocity are
   generated from the same truth but corrupted by independent noise, so by
   construction nothing is shared between the two chains.  Running the same
   sweep on that pair shows whether the collapse appears anyway.

Both tables are written under results/ and are read by the manuscript.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from velref.core.trajectory import Pose2D                      # noqa: E402
from velref.io import synthetic                                # noqa: E402
from velref.methods.baselines import central_diff              # noqa: E402
from velref.methods.family_a import family_a_pointwise         # noqa: E402

# Same releases, columns and shift grid as scripts/build_crossds_tables.py.
DATASETS = [
    ("HeLiPR",    "helipr",       "v_ins"),
    ("Oxford",    "oxford_x11",   "v_ins"),
    ("nuScenes",  "nuscenes_x20", "v_can"),
    ("KITTI",     "kitti",        "v_oxts"),
    ("KITTI-360", "kitti360",     "v_ref"),
    ("Boreas",    "boreas",       "v_ref"),
    ("Pit30M",    "pit30m_10hz",  "v_ref"),
]
SHIFTS = (-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def load(subdir: str, ref_col: str):
    for p in sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
        d = pd.read_parquet(p)
        yield (d["t"].to_numpy(), d[ref_col].to_numpy(),
               d["v_central"].to_numpy(), d["v_family_a_W5"].to_numpy())


def per_sequence_sweep(seqs, shifts=SHIFTS) -> np.ndarray:
    """Residual of each operator, per sequence, per shift, on one common interval.

    A shift of $s$ pushes the resampled reference off the ends of the track, so
    every shift keeps a slightly different set of frames.  Comparing residuals
    across shifts then compares slightly different intervals.  We intersect the
    valid masks of all shifts first and evaluate every shift on that one
    interval, which is the whole sweep's common support.

    Returns an array of shape (n_sequences, n_shifts, 2): central, Family A.
    """
    out = np.full((len(seqs), len(shifts), 2), np.nan)
    for i, (t, r, c, f) in enumerate(seqs):
        shifted = [np.interp(t, t + s, r, left=np.nan, right=np.nan) for s in shifts]
        common = np.isfinite(c) & np.isfinite(f)
        for rs in shifted:
            common &= np.isfinite(rs)
        for j, rs in enumerate(shifted):
            out[i, j, 0] = rmse(c[common], rs[common])
            out[i, j, 1] = rmse(f[common], rs[common])
    return out


def quadrature_rows(name: str, res: np.ndarray) -> list[dict]:
    """Predict the Family A residual at each shift from the central channel alone.

    The prediction is made per sequence and only then aggregated, because the
    median of a prediction is not the prediction from a median.  Where the
    central residual at a shift is no larger than at zero, the implied common
    error is imaginary; we clamp it to zero and count how often that happens so
    the clamp is visible rather than buried.
    """
    z = SHIFTS.index(0.0)
    rows = []
    for j, s in enumerate(SHIFTS):
        rc, rf = res[:, j, 0], res[:, j, 1]
        rc0, rf0 = res[:, z, 0], res[:, z, 1]
        e2 = rc ** 2 - rc0 ** 2
        clamped = int(np.sum(e2 < 0))
        e = np.sqrt(np.clip(e2, 0.0, None))
        rf_pred = np.sqrt(rf0 ** 2 + e ** 2)
        med_c, med_f = float(np.median(rc)), float(np.median(rf))
        med_pred = float(np.median(rf_pred))
        rows.append({
            "dataset": name,
            "shift_s": s,
            "n_sequences": int(res.shape[0]),
            "central_obs": med_c,
            "fa_obs": med_f,
            "delta_obs_pct": (med_f / med_c - 1) * 100.0,
            "common_error_e_median": float(np.median(e)),
            "fa_pred_quadrature": med_pred,
            "delta_pred_pct": (med_pred / med_c - 1) * 100.0,
            "delta_gap_pp": (med_f / med_c - 1) * 100.0 - (med_pred / med_c - 1) * 100.0,
            "sequences_with_clamped_error": clamped,
            "sequences_below_zero_shift_residual": int(np.sum(rc < rc0)),
        })
    return rows


def synthetic_control(seed: int = 7) -> list[dict]:
    """Two chains that share no smoother, swept the same way.

    The pose carries position noise; the published velocity is the truth plus its
    own independent noise.  No estimator, filter or smoother is shared, so any
    collapse of Delta under the shift cannot come from shared estimation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    cases = [
        # label, position noise (m), published-velocity noise (m/s)
        ("independent, published velocity noisy", 0.03, 0.30),
        ("independent, published velocity clean", 0.03, 0.02),
    ]
    for label, pos_sigma, vel_sigma in cases:
        seqs = []
        for k in range(8):
            smp = synthetic.stop_go(duration=60.0, fs=10.0, seed=100 + k)
            t, x, y = smp.pose.t, smp.pose.x, smp.pose.y
            x = x + rng.normal(0.0, pos_sigma, x.size)
            y = y + rng.normal(0.0, pos_sigma, y.size)
            p = Pose2D(t, x, y)
            v_pub = smp.v_true + rng.normal(0.0, vel_sigma, t.size)
            seqs.append((t, v_pub, central_diff(p), family_a_pointwise(p, W=5)))
        res = per_sequence_sweep(seqs)
        for j, s in enumerate(SHIFTS):
            rc = float(np.median(res[:, j, 0]))
            rf = float(np.median(res[:, j, 1]))
            rows.append({
                "case": label,
                "pos_sigma_m": pos_sigma,
                "vel_sigma_mps": vel_sigma,
                "shift_s": s,
                "central": rc,
                "fa_W5": rf,
                "delta_pct": (rf / rc - 1) * 100.0,
                "shared_smoother": "no (independent by construction)",
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args()

    quad = []
    print("released streams: measured vs quadrature-predicted Delta")
    for name, subdir, ref_col in DATASETS:
        seqs = list(load(subdir, ref_col))
        if not seqs:
            print(f"  {name}: no per-frame streams under results/{subdir}")
            continue
        res = per_sequence_sweep(seqs)
        rows = quadrature_rows(name, res)
        quad.extend(rows)
        worst = max(rows, key=lambda r: abs(r["delta_gap_pp"]))
        for r in rows:
            mark = "  <- worst gap" if r is worst else ""
            print(f"  {name:<10} s={r['shift_s']:+.1f}  "
                  f"central {r['central_obs']:.4f}  FamilyA {r['fa_obs']:.4f}  "
                  f"measured {r['delta_obs_pct']:+8.2f}%   "
                  f"quadrature {r['delta_pred_pct']:+8.2f}%   "
                  f"gap {r['delta_gap_pp']:+7.2f} pp  "
                  f"clamped {r['sequences_with_clamped_error']}/{r['n_sequences']}{mark}")

    syn = synthetic_control()
    print("\nsynthetic control (no shared smoother):")
    for r in syn:
        if r["shift_s"] in (0.0, 0.5):
            print(f"  {r['case']:<40} s={r['shift_s']:+.1f}  Delta {r['delta_pct']:+7.2f}%")

    pd.DataFrame(quad).to_csv(args.out / "shift_quadrature_control.csv", index=False)
    pd.DataFrame(syn).to_csv(args.out / "shift_synthetic_control.csv", index=False)
    print(f"\nwrote {args.out/'shift_quadrature_control.csv'} and "
          f"{args.out/'shift_synthetic_control.csv'}")


if __name__ == "__main__":
    main()
