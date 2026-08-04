#!/usr/bin/env python3
"""Process AV2 sensor val pose feathers and DurLAR_S exemplar GPS CSVs.

Both datasets ship pose only (no published velocity channel), so we report
M_2 (low-speed RMS) and M_3_sm (smoothness, RMS of 2nd difference). M_4 is
marked N/A. This is the "no published velocity" sub-regime.

Outputs:
  results/av2/per_sequence.csv
  results/av2/aggregate.csv
  results/durlar/per_sequence.csv
  results/durlar/aggregate.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from velref.core.trajectory import Pose2D  # noqa: E402
from velref.methods.family_a import family_a_pointwise  # noqa: E402
from velref.methods.baselines import central_diff  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


EARTH_R = 6378137.0


def latlon_to_xy(lat_deg, lon_deg, lat0, lon0):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)
    x = EARTH_R * (lon - lon0r) * np.cos(lat0r)
    y = EARTH_R * (lat - lat0r)
    return x, y


def low_speed_rms(v: np.ndarray, threshold: float = 0.5) -> float:
    """RMS of velocity at frames where |v| < threshold (low-speed sanity)."""
    mask = np.abs(v) < threshold
    if mask.sum() < 5:
        return float("nan")
    return float(np.sqrt(np.mean(v[mask] ** 2)))


def smoothness_rms2nd(v: np.ndarray) -> float:
    """RMS of 2nd difference (M_3_sm proxy)."""
    d2 = np.diff(v, n=2)
    return float(np.sqrt(np.mean(d2 ** 2)))


def per_method_metrics(pose: Pose2D) -> dict:
    out = {}
    methods = {
        "central": lambda: central_diff(pose),
        "family_a_W5": lambda: family_a_pointwise(pose, W=5),
        "family_a_W7": lambda: family_a_pointwise(pose, W=7),
    }
    for name, fn in methods.items():
        try:
            v = fn()
            v = np.asarray(v, dtype=float)
            if v.ndim > 1:
                v = np.linalg.norm(v, axis=-1) if v.shape[-1] in (2, 3) else v.squeeze()
            out[f"{name}_M2"] = low_speed_rms(v)
            out[f"{name}_M3sm"] = smoothness_rms2nd(v)
        except Exception as e:
            out[f"{name}_err"] = str(e)
    return out


def process_av2(feather_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fp in sorted(feather_dir.glob("*.feather")):
        df = feather.read_feather(fp)
        if len(df) < 50:
            continue
        t = df["timestamp_ns"].to_numpy() / 1e9
        # AV2 pose at ~200 Hz; downsample to 10 Hz to match other datasets
        target_dt = 0.1
        keep = [0]
        last = t[0]
        for i in range(1, len(t)):
            if t[i] - last >= target_dt - 1e-3:
                keep.append(i)
                last = t[i]
        keep = np.array(keep)
        ts = t[keep]
        x = df["tx_m"].to_numpy()[keep]
        y = df["ty_m"].to_numpy()[keep]
        if len(ts) < 30:
            continue
        pose = Pose2D(ts, x, y)
        m = per_method_metrics(pose)
        m["sequence"] = fp.stem
        m["n_frames"] = len(ts)
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sequence.csv", index=False)
    # aggregate
    agg = {}
    for col in df.columns:
        if col in ("sequence", "n_frames") or col.endswith("_err"):
            continue
        agg[col + "_median"] = df[col].median()
        agg[col + "_q1"] = df[col].quantile(0.25)
        agg[col + "_q3"] = df[col].quantile(0.75)
    pd.DataFrame([agg]).to_csv(out_dir / "aggregate.csv", index=False)
    print(f"[AV2] {len(df)} sequences -> {out_dir}")
    print(df[[c for c in df.columns if c not in ("sequence",)]].describe().to_string())
    return df


def process_durlar(durlar_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq_dir in sorted(durlar_dir.glob("DurLAR_*_S")):
        gps_csv = seq_dir / "gps" / "data.csv"
        if not gps_csv.exists():
            continue
        # GPS CSV columns: ts_ns, frame_idx, header_ts_ns, frame_id, status, service, lat, lon, alt, ... (no header)
        cols = ["ts_ns", "frame_idx", "header_ts_ns", "frame_id", "status", "service",
                "lat", "lon", "alt", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "pos_cov_type"]
        df = pd.read_csv(gps_csv, header=None, names=cols)
        ts = df["header_ts_ns"].to_numpy() / 1e9
        # Sort by ts (sometimes out of order)
        order = np.argsort(ts)
        ts = ts[order]
        lat = df["lat"].to_numpy()[order]
        lon = df["lon"].to_numpy()[order]
        # Drop duplicate timestamps
        keep = np.concatenate(([True], np.diff(ts) > 1e-6))
        ts = ts[keep]
        lat = lat[keep]
        lon = lon[keep]
        if len(ts) < 30:
            continue
        x, y = latlon_to_xy(lat, lon, lat[0], lon[0])
        # DurLAR GPS at ~100 Hz; downsample to 10 Hz
        target_dt = 0.1
        last = ts[0]
        keep_idx = [0]
        for i in range(1, len(ts)):
            if ts[i] - last >= target_dt - 1e-3:
                keep_idx.append(i)
                last = ts[i]
        keep_idx = np.array(keep_idx)
        ts = ts[keep_idx]
        x = x[keep_idx]
        y = y[keep_idx]
        if len(ts) < 30:
            continue
        pose = Pose2D(ts, x, y)
        m = per_method_metrics(pose)
        m["sequence"] = seq_dir.name
        m["n_frames"] = len(ts)
        rows.append(m)
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
    print(f"[DurLAR_S] {len(df)} sequences -> {out_dir}")
    print(df[[c for c in df.columns if c not in ("sequence",)]].describe().to_string())
    return df


def main():
    av2_dir = Path("/mnt/Data/velref/av2_poses")
    durlar_dir = Path("/mnt/Data/velref/durlar_S_full")
    out_root = REPO_ROOT / "results"
    process_av2(av2_dir, out_root / "av2")
    process_durlar(durlar_dir, out_root / "durlar")


if __name__ == "__main__":
    main()
