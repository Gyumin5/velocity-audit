#!/usr/bin/env python3
"""Does stride decimation without an anti-aliasing filter change the audit reading?

Three of the seven releases publish above the 10 Hz analysis cadence and are
brought to it by taking every k-th sample:

  release   native     stride
  nuScenes   ~50 Hz      5
  Pit30M     100 Hz     10
  Boreas    ~200 Hz     20

Reviewer 3 objects that plain stride subsampling folds everything above 5 Hz
back into the analysis band. This script answers that with two measurements
rather than an argument.

(1) How much power is up there. Welch spectra of the published velocity and of
    the detrended position channels at the native rate, integrated above the
    5 Hz analysis Nyquist and expressed as a fraction of total power. If that
    fraction is small there is little to fold.

(2) Whether removing it changes the reading. The same 10 Hz series is built a
    second time through a zero-phase low-pass filter, and the audit quantities
    are recomputed on both routes under identical treatment:

      stride    x[::k]                      -- what the manuscript tables use
      filtered  filtfilt(b, a, x)[::k]      -- anti-aliased before subsampling

    Butterworth, order 4, applied with scipy.signal.filtfilt so the filter has
    exactly zero phase and no group delay to correct for. Two cutoffs are run
    (4.0 Hz = 0.8x the analysis Nyquist, 2.5 Hz = 0.5x) so the result does not
    rest on one knob setting. Both pose channels and the published velocity go
    through the same filter, so the two streams are compared over the same band.
    filtfilt pads by 3*max(len(a), len(b)) samples at each end; on top of that we
    drop one probe window (10 samples at 10 Hz) from each end of both routes, so
    the comparison never reads a filter edge transient, and the trim is identical
    on the two routes.

Non-uniform native clocks are put on a uniform grid at the native median
interval before filtering (filtfilt assumes uniform sampling); the stride route
is built from the same grid, so the two routes differ only by the filter.

Writes results/antialias_control.csv.

Usage:  python scripts/antialias_control.py [--root /mnt/Data/velref]
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from velref.core.trajectory import Pose2D  # noqa: E402
from velref.methods.family_a import family_a_pointwise  # noqa: E402
from velref.methods.baselines import central_diff  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_window_control import read_boreas, read_nuscenes  # noqa: E402

ANALYSIS_HZ = 10.0
ANALYSIS_NYQUIST = ANALYSIS_HZ / 2
CUTOFFS_HZ = (4.0, 2.5)
FILTER_ORDER = 4
W = 5           # 1.0 s at the 10 Hz analysis cadence
EDGE_TRIM = 10  # one probe window dropped from each end, both routes alike
SHIFTS = (-0.5, -0.2, -0.1, 0.1, 0.2, 0.5)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def smoothness(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


def read_pit30m():
    for p in sorted((REPO_ROOT / "results" / "pit30m").glob("per_frame_*.parquet")):
        d = pd.read_parquet(p, columns=["t", "x", "y", "v_ref"])
        t = d["t"].to_numpy(np.float64)
        keep = np.concatenate([[True], np.diff(t) > 0])
        yield (p.stem.replace("per_frame_", ""), t[keep] - t[keep][0],
               d["x"].to_numpy()[keep], d["y"].to_numpy()[keep],
               d["v_ref"].to_numpy()[keep])


RELEASES = [
    ("nuScenes", read_nuscenes, True),
    ("Pit30M", lambda root: read_pit30m(), False),
    ("Boreas", read_boreas, True),
]


def uniform_grid(t, *channels):
    """Resample onto a regular grid at the native median interval."""
    dt = float(np.median(np.diff(t)))
    tu = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / dt)) + 1) * dt
    return dt, tu, [np.interp(tu, t, c) for c in channels]


def high_band_fraction(sig: np.ndarray, fs: float) -> float:
    """Share of Welch power above the 5 Hz analysis Nyquist."""
    sig = sig - np.mean(sig)
    nper = min(len(sig), 4096)
    f, p = welch(sig, fs=fs, nperseg=nper)
    tot = np.trapz(p, f)
    if tot <= 0:
        return float("nan")
    hi = np.trapz(p[f > ANALYSIS_NYQUIST], f[f > ANALYSIS_NYQUIST])
    return float(hi / tot)


def detrend_poly(t: np.ndarray, v: np.ndarray, deg: int = 3) -> np.ndarray:
    return v - np.polyval(np.polyfit(t - t[0], v, deg), t - t[0])


def build_route(t, x, y, v, step, sos=None):
    if sos is not None:
        b, a = sos
        x, y, v = (filtfilt(b, a, c) for c in (x, y, v))
    sl = slice(EDGE_TRIM, -EDGE_TRIM if EDGE_TRIM else None)
    return t[::step][sl], x[::step][sl], y[::step][sl], v[::step][sl]


def audit(t, x, y, v):
    pose = Pose2D(t=t, x=x, y=y)
    c, f = central_diff(pose), family_a_pointwise(pose, W=W)
    return c, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/Data/velref"))
    args = ap.parse_args()

    rows = []
    for name, reader, needs_root in RELEASES:
        print(f"\n=== {name}")
        seqs = list(reader(args.root) if needs_root else reader(args.root))
        if not seqs:
            print("  no source sequences found; skipping")
            continue

        hi_v, hi_x, per_route = [], [], {}
        for sid, t, x, y, v in seqs:
            dt, tu, (xu, yu, vu) = uniform_grid(t, x, y, v)
            fs = 1.0 / dt
            step = int(round(fs / ANALYSIS_HZ))
            if step < 2 or len(tu) < 2 * (step * EDGE_TRIM + 2 * W + 3):
                print(f"  {sid}: too short or already at the analysis cadence")
                continue
            hi_v.append(high_band_fraction(vu, fs))
            hi_x.append(np.mean([high_band_fraction(detrend_poly(tu, c), fs)
                                 for c in (xu, yu)]))

            routes = {"stride": None}
            for fc in CUTOFFS_HZ:
                routes[f"filtered_{fc:g}Hz"] = butter(FILTER_ORDER, fc / (fs / 2), btype="low")
            for label, coef in routes.items():
                td, xd, yd, vd = build_route(tu, xu, yu, vu, step, coef)
                c, f = audit(td, xd, yd, vd)
                rec = per_route.setdefault(label, {"m4c": [], "m4f": [], "m3c": [], "m3f": [],
                                                   "shift": {s: ([], []) for s in SHIFTS}})
                rec["m4c"].append(rmse(c, vd))
                rec["m4f"].append(rmse(f, vd))
                rec["m3c"].append(smoothness(c))
                rec["m3f"].append(smoothness(f))
                for s in SHIFTS:
                    vs = np.interp(td, td + s, vd, left=np.nan, right=np.nan)
                    rec["shift"][s][0].append(rmse(c, vs))
                    rec["shift"][s][1].append(rmse(f, vs))

        if not per_route:
            continue
        # A sequence whose published velocity is constant (a parked scene) has no
        # spectral power to apportion and drops out of the velocity column only.
        fx, fv = float(np.nanmedian(hi_x)), float(np.nanmedian(hi_v))
        n_v = int(np.isfinite(hi_v).sum())
        print(f"  power above {ANALYSIS_NYQUIST:g} Hz at native rate: "
              f"position {fx:.2e} ({len(hi_x)} seq), "
              f"published velocity {fv:.2e} ({n_v} seq)")

        base = None
        for label, rec in per_route.items():
            mc, mf = float(np.median(rec["m4c"])), float(np.median(rec["m4f"]))
            delta = (mf / mc - 1) * 100
            ratio = float(np.median(rec["m3c"])) / float(np.median(rec["m3f"]))
            if base is None:
                base = delta
            shifts = {s: (float(np.median(rec["shift"][s][1]))
                          / float(np.median(rec["shift"][s][0])) - 1) * 100 for s in SHIFTS}
            rows.append({
                "dataset": name, "route": label, "n_seq": len(rec["m4c"]),
                "hi_band_frac_position": fx, "hi_band_frac_velocity": fv,
                "n_seq_velocity_spectrum": n_v,
                "M4_central": mc, "M4_fa_W5": mf, "delta_pct": delta,
                "delta_minus_stride_pp": delta - base,
                "smooth_ratio": ratio,
                **{f"delta_shift_{s:+.1f}s_pct": shifts[s] for s in SHIFTS},
            })
            print(f"  {label:<16} M4 {mc:.5f} -> {mf:.5f}  Delta {delta:+7.2f}%  "
                  f"({delta - base:+.2f} pp vs stride)  smooth {ratio:.2f}x  "
                  f"max|Delta| over shifts {max(abs(v) for v in shifts.values()):.2f}%")

    if not rows:
        print("nothing computed")
        return
    out = REPO_ROOT / "results" / "antialias_control.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
