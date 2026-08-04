#!/usr/bin/env python3
"""Process KITTI-360 OXTS data: pose from lat/lon, published velocity from vn/ve.

Provenance regime: ONLINE (single OXTS RT3003 INS produces both pose and velocity),
distinct from KITTI raw only by sequence/location, useful as a same-regime repeat.

Outputs:
  results/kitti360/per_sequence.csv
  results/kitti360/aggregate.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from velref.core.trajectory import Pose2D  # noqa: E402
from velref.methods.family_a import family_a_pointwise  # noqa: E402
from velref.methods.baselines import central_diff  # noqa: E402


EARTH_R = 6378137.0


def latlon_to_xy(lat_deg, lon_deg, lat0, lon0):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)
    x = EARTH_R * (lon - lon0r) * np.cos(lat0r)
    y = EARTH_R * (lat - lat0r)
    return x, y


def load_oxts_drive(drive_dir: Path):
    ts_file = drive_dir / "oxts" / "timestamps.txt"
    data_dir = drive_dir / "oxts" / "data"
    ts = []
    with open(ts_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # e.g., "2013-05-28 08:46:02.901385659"
            t = pd.Timestamp(line)
            if pd.isna(t):
                continue
            ts.append(t.timestamp())
    ts = np.asarray(ts, dtype=np.float64)
    data_files = sorted(data_dir.glob("*.txt"))
    n = len(data_files)
    if len(ts) != n:
        # Some sequences have a few-frame mismatch; align to min
        m = min(n, len(ts))
        ts = ts[:m]
        data_files = data_files[:m]
        n = m
    arr = np.zeros((n, 30), dtype=np.float64)
    for i, fp in enumerate(data_files):
        vals = np.fromstring(open(fp).read().strip(), sep=" ")
        arr[i, :min(30, len(vals))] = vals[:30]
    # cols: 0:lat 1:lon 2:alt 3:roll 4:pitch 5:yaw 6:vn 7:ve 8:vf 9:vl 10:vu
    return ts, arr


def low_speed_rms(v, threshold=0.5):
    mask = np.abs(v) < threshold
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean(v[mask] ** 2)))


def smoothness_rms2nd(v):
    return float(np.sqrt(np.mean(np.diff(v, n=2) ** 2)))


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def process_drive(drive_dir: Path):
    ts, oxts = load_oxts_drive(drive_dir)
    if len(ts) < 50:
        return None
    lat, lon = oxts[:, 0], oxts[:, 1]
    vn, ve, vf = oxts[:, 6], oxts[:, 7], oxts[:, 8]
    v_horiz = np.hypot(vn, ve)
    # Drop duplicate timestamps
    keep = np.concatenate(([True], np.diff(ts) > 1e-6))
    ts = ts[keep]
    lat = lat[keep]
    lon = lon[keep]
    vn = vn[keep]
    ve = ve[keep]
    vf = vf[keep]
    v_horiz = v_horiz[keep]
    if len(ts) < 50:
        return None
    x, y = latlon_to_xy(lat, lon, lat[0], lon[0])
    pose = Pose2D(ts, x, y)
    methods = {
        "central": central_diff(pose),
        "family_a_W5": family_a_pointwise(pose, W=5),
        "family_a_W7": family_a_pointwise(pose, W=7),
    }
    out = {"sequence": drive_dir.name, "n_frames": len(ts)}
    for name, v in methods.items():
        v = np.asarray(v, dtype=float)
        if v.ndim > 1:
            v = np.linalg.norm(v, axis=-1) if v.shape[-1] in (2, 3) else v.squeeze()
        out[f"{name}_M2"] = low_speed_rms(v)
        out[f"{name}_M3sm"] = smoothness_rms2nd(v)
        out[f"{name}_M4_vs_vhoriz"] = rmse(v, v_horiz)
        out[f"{name}_M4_vs_vf"] = rmse(v, vf)
    return out


def main():
    base = Path("/mnt/Data/velref/kitti360/data_poses")
    out_dir = _root / "results" / "kitti360"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for drv in sorted(base.glob("2013_05_28_drive_*_sync")):
        try:
            r = process_drive(drv)
            if r:
                rows.append(r)
                print(f"  {drv.name}: n={r['n_frames']}  "
                      f"central M4 vs vhoriz={r['central_M4_vs_vhoriz']:.3f}  "
                      f"FA W5 M4 vs vhoriz={r['family_a_W5_M4_vs_vhoriz']:.3f}  "
                      f"delta={(r['family_a_W5_M4_vs_vhoriz']/r['central_M4_vs_vhoriz']-1)*100:+.1f}%")
        except Exception as e:
            print(f"  {drv.name}: FAIL {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sequence.csv", index=False)

    agg = {}
    for col in df.columns:
        if col in ("sequence", "n_frames") or col.endswith("_err"):
            continue
        agg[col + "_median"] = df[col].median()
        agg[col + "_q1"] = df[col].quantile(0.25)
        agg[col + "_q3"] = df[col].quantile(0.75)
    pd.DataFrame([agg]).to_csv(out_dir / "aggregate.csv", index=False)
    print(f"\n[KITTI-360] {len(df)} drives -> {out_dir}")
    cols = [c for c in df.columns if c not in ("sequence",)]
    print(df[cols].describe().to_string())


if __name__ == "__main__":
    main()
