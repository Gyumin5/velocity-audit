#!/usr/bin/env python3
"""Measurements behind the four detail checks in reviewer 2's comment 6.

The reviewer asks us to check (a) the definition of Delta t in the chord--arc
discussion, (b) the valid comparison interval and the boundary handling during
the time shifts, (c) the numerical precision used to report M1, and (d) the
operator count and column descriptions in Table 8.  (d) is a caption fix with
nothing to measure; the other three are measured here from the per-frame
streams and written to one CSV each.

(a) Chord vs arc.  Sec.~\\ref{sec:problem} defines $\\Delta t_k = t_{k+1} - t_k$
    and then writes the central estimator over $t_{k+1} - t_{k-1}$, which is
    $2\\Delta t$.  The subtended angle is therefore $\\theta = 2\\omega\\Delta t$,
    not $\\omega\\Delta t$, so the relative chord-to-arc bias
    $1 - \\mathrm{sinc}(\\theta/2) \\approx \\theta^2/24 = (\\omega\\Delta t)^2/6$
    is four times what the current text implies.  We evaluate it frame by frame
    from the measured curvature and the published speed instead of asserting an
    order of magnitude, and we put the resulting bias next to the residual the
    audit actually reports.

(b) Shift boundaries.  The sweep resamples the published channel with
    `np.interp(t, t + s, r, left=nan, right=nan)`, so a shift of $s$ drops the
    first $s$ seconds (or the last $|s|$ for $s < 0$) and the residual is taken
    over the surviving frames only.  Both operators are evaluated on the same
    surviving frames.  We measure the retained fraction and check the mask
    identity rather than describing them.

(c) M1 precision.  Table 4 prints several M1 entries as "$\\sim$0".  We print
    the values that are being rounded so the manuscript can report them at a
    fixed, stated precision.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from velref.core.curvature import estimate_curvature    # noqa: E402
from velref.core.trajectory import Pose2D               # noqa: E402

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


def streams(subdir: str):
    for p in sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
        yield p.stem[len("per_frame_"):], pd.read_parquet(p)


# Frames where the pose-derived curvature is itself meaningful.  Below the
# low-speed scale the curvature estimator divides by a speed that is dominated
# by position noise (Sec. "Low-speed degeneracy"), and a passenger vehicle does
# not turn inside a 5 m radius, so values past these bounds measure the
# curvature estimator rather than the trajectory.
V_MIN_MPS = 1.0
KAPPA_MAX = 0.2


def chord_arc(name: str, subdir: str, ref_col: str) -> dict:
    """Relative and absolute chord-to-arc bias over the central baseline."""
    rel, absol, resid, kept = [], [], [], []
    for _, d in streams(subdir):
        t = d["t"].to_numpy()
        v = d[ref_col].to_numpy()
        kappa = estimate_curvature(Pose2D(t, d["x"].to_numpy(), d["y"].to_numpy()))
        # Baseline of the central estimator, t_{k+1} - t_{k-1} = 2 dt.
        base = np.full_like(t, np.nan)
        base[1:-1] = t[2:] - t[:-2]
        theta = np.abs(kappa) * v * base                       # omega = |kappa| v
        ok = (np.isfinite(theta) & np.isfinite(v) & (v >= V_MIN_MPS)
              & (np.abs(kappa) <= KAPPA_MAX))
        kept.append(ok.sum() / t.size)
        half = theta[ok] / 2.0
        bias = np.where(half > 0, 1.0 - np.sin(half) / np.where(half > 0, half, 1.0), 0.0)
        rel.append(bias)
        absol.append(bias * v[ok])
        resid.append(rmse(d["v_central"].to_numpy(), v))
    rel = np.concatenate(rel)
    absol = np.concatenate(absol)
    med_resid = float(np.median(resid))
    return {
        "dataset": name,
        "frames_used": int(rel.size),
        "frames_used_fraction": float(np.mean(kept)),
        "rel_bias_median": float(np.median(rel)),
        "rel_bias_p99": float(np.percentile(rel, 99)),
        "rel_bias_max": float(rel.max()),
        "abs_bias_p99_mps": float(np.percentile(absol, 99)),
        "abs_bias_max_mps": float(absol.max()),
        "abs_bias_rms_mps": float(np.sqrt(np.mean(absol ** 2))),
        "central_residual_median_mps": med_resid,
        # The residual is an RMSE, so the RMS of the bias is the comparable scale;
        # the per-frame maximum is reported beside it as the worst case.
        "rms_bias_over_residual": float(np.sqrt(np.mean(absol ** 2)) / med_resid),
        "max_bias_over_residual": float(absol.max() / med_resid),
    }


def shift_interval(name: str, subdir: str, ref_col: str) -> list[dict]:
    """Retained fraction and mask identity for each shift."""
    rows = []
    for s in SHIFTS:
        keep, dropped, same_mask = [], [], True
        for _, d in streams(subdir):
            t = d["t"].to_numpy()
            r = np.interp(t, t + s, d[ref_col].to_numpy(), left=np.nan, right=np.nan)
            c, f = d["v_central"].to_numpy(), d["v_family_a_W5"].to_numpy()
            mc = np.isfinite(c) & np.isfinite(r)
            mf = np.isfinite(f) & np.isfinite(r)
            same_mask &= bool(np.array_equal(mc, mf))
            keep.append(mc.sum() / t.size)
            dropped.append(t.size - mc.sum())
        rows.append({
            "dataset": name,
            "shift_s": s,
            "retained_fraction_min": float(np.min(keep)),
            "retained_fraction_median": float(np.median(keep)),
            "frames_dropped_median": float(np.median(dropped)),
            "operators_share_mask": same_mask,
        })
    return rows


def m1_precision() -> list[dict]:
    """M1 of every operator in tab:helipr-main, unrounded.

    Same construction as the table: the trapezoidal integral of the estimate
    over the polyline length of the published pose track, median over the six
    sequences.  The table prints most of these as a magnitude symbol.
    """
    methods = ["forward", "central", "cubic_global", "savgol_w7p3", "smoothing_spline",
               "family_a_W3", "family_a_W5", "family_a_W7"]
    rows = []
    for meth in methods:
        vals = []
        for _, d in streams("helipr"):
            t, v = d["t"].to_numpy(), d[f"v_{meth}"].to_numpy()
            v_int = float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(t)))
            poly = float(np.sum(np.hypot(np.diff(d["x"].to_numpy()), np.diff(d["y"].to_numpy()))))
            vals.append(abs(v_int / poly - 1.0))
        rows.append({"method": meth, "m1": float(np.median(vals)),
                     "m1_min": float(np.min(vals)), "m1_max": float(np.max(vals))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args()

    chord, interval = [], []
    print("chord-to-arc bias over the central baseline (2 dt), measured per frame")
    for name, subdir, ref_col in DATASETS:
        if not list((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
            print(f"  {name}: no per-frame streams under results/{subdir}")
            continue
        row = chord_arc(name, subdir, ref_col)
        chord.append(row)
        print(f"  {name:<10} frames {row['frames_used_fraction']*100:5.1f}%  "
              f"median {row['rel_bias_median']:.2e}  p99 {row['rel_bias_p99']:.2e}  "
              f"max {row['rel_bias_max']:.2e}   rms abs {row['abs_bias_rms_mps']:.2e} m/s "
              f"= {row['rms_bias_over_residual']*100:.3f}% of the central residual "
              f"(worst frame {row['max_bias_over_residual']*100:.2f}%)")
        interval.extend(shift_interval(name, subdir, ref_col))

    print("\nvalid comparison interval under the shift")
    for r in interval:
        if r["shift_s"] in (0.0, 0.5):
            print(f"  {r['dataset']:<10} s={r['shift_s']:+.1f}  retained "
                  f"{r['retained_fraction_median']*100:.2f}% (worst {r['retained_fraction_min']*100:.2f}%)  "
                  f"dropped {r['frames_dropped_median']:.0f} frames  "
                  f"same mask for both operators: {r['operators_share_mask']}")

    m1 = m1_precision()
    print("\nM1 of tab:helipr-main at full precision (median over six sequences)")
    for r in m1:
        print(f"  {r['method']:<28} M1 = {r['m1']:.3e}   "
              f"range over sequences [{r['m1_min']:.3e}, {r['m1_max']:.3e}]")

    pd.DataFrame(chord).to_csv(args.out / "chord_arc_bias.csv", index=False)
    pd.DataFrame(interval).to_csv(args.out / "shift_valid_interval.csv", index=False)
    pd.DataFrame(m1).to_csv(args.out / "m1_precision.csv", index=False)
    print(f"\nwrote {args.out/'chord_arc_bias.csv'}, {args.out/'shift_valid_interval.csv'} "
          f"and {args.out/'m1_precision.csv'}")


if __name__ == "__main__":
    main()
