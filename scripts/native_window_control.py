#!/usr/bin/env python3
"""Control experiment: does the reading survive without decimation?

Three of the seven audited releases publish above the 10 Hz analysis cadence and
are brought to it by stride subsampling. The audit fixes the probe's window by
duration rather than sample count, so the same 1.0 s window can be reached the
other way -- keep the native rate and widen W -- and if the reported behaviour
were an artifact of discarding samples the two routes would disagree.

  release    native      decimated route        native route
  Boreas     ~200 Hz     stride-20, W=5         W=100
  nuScenes    ~50 Hz     stride-5,  W=5         W=25
  Pit30M      100 Hz     stride-10, W=5         W=50   (scripts/pit30m_native_window_control.py)

This script runs the Boreas and nuScenes halves from each release's own source
files. It first rebuilds the decimated series and checks it against the committed
per-frame stream, so a disagreement in the control cannot be blamed on a
different extraction; then it recomputes the three qualitative claims -- the sign
of the M_4 change, the collapse of that change under a latency shift, and the
direction of the per-curvature-bin change -- at the native rate.

Conventions follow scripts/pit30m_native_window_control.py so the three controls
are directly comparable: median per sequence then ratio of medians, the same bin
edges, and the full series rather than an interior slice.

Writes results/<release>_native_control.csv.

Usage:  python scripts/native_window_control.py [--root /mnt/Data/velref]
"""
from __future__ import annotations
from pathlib import Path
import argparse
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
ANALYSIS_HZ = 10.0
WINDOW_S = 1.0  # the probe's fixed window duration: W=5 spans 1.0 s at 10 Hz


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def smoothness(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


# --- release readers, each returning native-rate (t, x, y, v_ref) per sequence ---

def read_boreas(root: Path):
    from velref.io.boreas import DEFAULT_SEQUENCES
    for name in DEFAULT_SEQUENCES:
        seq_root = root / "boreas" / name
        csv = seq_root / "applanix" / "gps_post_process.csv"
        if not csv.exists():
            print(f"  missing {csv}")
            continue
        d = pd.read_csv(csv)
        t = d["GPSTime"].to_numpy(np.float64)
        keep = np.concatenate([[True], np.diff(t) > 0])
        d, t = d.loc[keep].reset_index(drop=True), t[keep]
        yield (name, t - t[0], d["easting"].to_numpy(np.float64),
               d["northing"].to_numpy(np.float64),
               np.hypot(d["vel_east"].to_numpy(np.float64),
                        d["vel_north"].to_numpy(np.float64)))


def read_nuscenes(root: Path):
    import json
    ids = sorted(pd.read_csv(REPO_ROOT / "results" / "audited_sequences.csv")
                 .query("dataset == 'nuScenes'").id)
    for sid in ids:
        p = root / "nuscenes" / "can_bus" / f"{sid}_pose.json"
        if not p.exists():
            print(f"  missing {p}")
            continue
        rec = json.loads(p.read_text())
        t = np.asarray([r["utime"] for r in rec], np.int64)
        pos = np.asarray([r["pos"] for r in rec], np.float64)
        vel = np.asarray([r["vel"] for r in rec], np.float64)
        t = (t - t[0]) / 1e6
        keep = np.concatenate([[True], np.diff(t) > 0])
        yield (sid, t[keep], pos[keep, 0], pos[keep, 1],
               np.hypot(vel[keep, 0], vel[keep, 1]))


RELEASES = [
    ("Boreas", "boreas", "v_ref", read_boreas),
    ("nuScenes", "nuscenes_x20", "v_can", read_nuscenes),
]


def check_decimation(subdir: str, ref_col: str, seqs) -> None:
    """Confirm the freshly read source reproduces the committed 10 Hz stream.

    Without this the control would only show that two of our own runs agree. The
    comparison is on the inputs -- timestamps, position, published velocity --
    since those are what the extraction stage is responsible for.
    """
    print(f"  decimation check against results/{subdir}/")
    for sid, t, x, y, v in seqs:
        p = REPO_ROOT / "results" / subdir / f"per_frame_{sid}.parquet"
        if not p.exists():
            print(f"    {sid}: no committed stream")
            continue
        step = max(1, int(round((1.0 / np.median(np.diff(t))) / ANALYSIS_HZ)))
        got = {"t": t[::step], "x": x[::step], "y": y[::step], ref_col: v[::step]}
        ref = pd.read_parquet(p, columns=list(got))
        n = min(len(ref), len(got["t"]))
        worst = max(float(np.nanmax(np.abs(got[c][:n] - ref[c].to_numpy()[:n])))
                    for c in got)
        flag = "ok" if worst < 1e-6 and abs(len(ref) - len(got["t"])) <= 1 else "MISMATCH"
        print(f"    {sid:<28} n {len(got['t']):6d} vs {len(ref):6d}  "
              f"max abs diff {worst:.3e}  {flag}")


def run_control(seqs) -> pd.DataFrame:
    dt = float(np.median(np.diff(seqs[0][1])))
    W = int(round((WINDOW_S / dt) / 2))
    built = []
    for sid, t, x, y, v in seqs:
        if len(t) < 2 * W + 11:
            print(f"    {sid}: too short for W={W}")
            continue
        pose = Pose2D(t=t, x=x, y=y)
        built.append((sid, t, x, y, v, central_diff(pose),
                      family_a_pointwise(pose, W=W)))
    if not built:
        return pd.DataFrame()

    rows = []
    m4c = [rmse(c, r) for *_, r, c, _ in built]
    m4f = [rmse(f, r) for *_, r, _, f in built]
    med_c, med_f = float(np.median(m4c)), float(np.median(m4f))
    delta0 = (med_f / med_c - 1) * 100
    m3c = float(np.median([smoothness(c) for *_, c, _ in built]))
    m3f = float(np.median([smoothness(f) for *_, f in built]))
    rows.append({"check": "alignment", "key": "delta_pct", "value": delta0})
    rows.append({"check": "alignment", "key": "smooth_x", "value": m3c / m3f})
    rows.append({"check": "alignment", "key": "W_native", "value": W})
    rows.append({"check": "alignment", "key": "rate_hz", "value": 1.0 / dt})
    print(f"  {len(built)} sequences at native {1.0 / dt:.1f} Hz, W={W} "
          f"({2 * W * dt:.2f} s span)")
    print(f"    M4 {med_c:.5f} -> {med_f:.5f}   Delta {delta0:+.2f}%   "
          f"smooth {m3c / m3f:.2f}x")

    print("    latency sweep:")
    for s in SHIFTS:
        sc, sf = [], []
        for _, t, _, _, r, c, f in built:
            rs = np.interp(t, t + s, r, left=np.nan, right=np.nan)
            sc.append(rmse(c, rs))
            sf.append(rmse(f, rs))
        d = (float(np.median(sf)) / float(np.median(sc)) - 1) * 100
        rows.append({"check": "latency", "key": f"shift_{s:+.1f}s", "value": d})
        print(f"      {s:+.1f}s  Delta {d:+8.2f}%")

    print("    curvature bins:")
    per_bin: dict[str, list[tuple[float, float]]] = {b: [] for b in BIN_LABELS}
    for _, t, x, y, r, c, f in built:
        kappa = np.abs(estimate_curvature(Pose2D(t=t, x=x, y=y), window=9, polyorder=3))
        bins = np.clip(np.digitize(kappa, BIN_EDGES) - 1, 0, len(BIN_LABELS) - 1)
        ok = np.isfinite(c) & np.isfinite(f) & np.isfinite(r)
        for b, label in enumerate(BIN_LABELS):
            mask = (bins == b) & ok
            if mask.sum() >= MIN_SAMPLES:
                per_bin[label].append((rmse(c[mask], r[mask]), rmse(f[mask], r[mask])))
    for label in BIN_LABELS:
        vals = per_bin[label]
        if not vals:
            continue
        d = (float(np.median([v[1] for v in vals]))
             / float(np.median([v[0] for v in vals])) - 1) * 100
        rows.append({"check": "curvature", "key": label, "value": d})
        print(f"      kappa {label:<12} n_seq {len(vals):3d}  Delta {d:+8.2f}%")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/Data/velref"))
    args = ap.parse_args()

    for name, subdir, ref_col, reader in RELEASES:
        print(f"\n=== {name}")
        seqs = list(reader(args.root))
        if not seqs:
            print("  no source sequences found; skipping")
            continue
        check_decimation(subdir, ref_col, seqs)
        df = run_control(seqs)
        if df.empty:
            continue
        out = REPO_ROOT / "results" / f"{subdir}_native_control.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
