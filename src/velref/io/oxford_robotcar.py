"""Oxford RobotCar loader.

Supports two pose-reference pairings:

- ``mode='ins'``: pose (northing, easting) and velocity components both come
  from ``ins.csv`` (the NovAtel SPAN-CPT GPS/INS solution). Pose and reference
  share the INS Kalman filter, so this is a *coupled* provenance regime
  (similar to KITTI OXTS).
- ``mode='rtk_ins'``: pose comes from ``rtk.csv`` (post-processed RTK-GPS
  positioning) while the velocity reference comes from ``ins.csv``. Although
  both streams use the same NovAtel sensor suite, the position and velocity
  estimates are produced by *different algorithms* with different smoothing
  characteristics, giving a HeLiPR-like *separated* provenance regime.

Both ``ins.csv`` and ``rtk.csv`` follow the columns documented in the
RobotCar SDK::

    timestamp, ins_status, latitude, longitude, altitude,
    northing, easting, down, utm_zone,
    velocity_north, velocity_east, velocity_down,
    roll, pitch, yaw

Timestamps are microseconds since the Unix epoch.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from velref.core.trajectory import Pose2D


@dataclass
class OxfordSequence:
    name: str
    pose: Pose2D
    vel_north: np.ndarray   # m/s, INS-fused
    vel_east: np.ndarray    # m/s, INS-fused
    vel_down: np.ndarray    # m/s, INS-fused
    yaw: np.ndarray         # rad
    pose_source: str        # 'ins' or 'rtk'
    n_dropped_ins: int = 0
    n_dropped_rtk: int = 0

    @property
    def v_horiz(self) -> np.ndarray:
        return np.hypot(self.vel_east, self.vel_north)

    @property
    def v_ins_horiz(self) -> np.ndarray:
        return self.v_horiz


def _read_oxford_csv(path: Path) -> pd.DataFrame:
    """Read ins.csv or rtk.csv. Drops invalid rows (status != 'INS_SOLUTION_GOOD' / etc)."""
    df = pd.read_csv(path)
    keep_cols = [
        "timestamp", "northing", "easting",
        "velocity_north", "velocity_east", "velocity_down",
        "roll", "pitch", "yaw",
    ]
    if "ins_status" in df.columns:
        # Drop obviously unusable rows. Accept GOOD/FIXED/FLOAT/COMPLETE/CONVERGED.
        status = df["ins_status"].astype(str).str.upper()
        bad = status.str.contains("INACTIVE|ALIGNING|HIGH_VARIANCE|FREE|NONE", na=False)
        df = df.loc[~bad]
    cols = [c for c in keep_cols if c in df.columns]
    sub: pd.DataFrame = df.loc[:, cols].dropna()
    return sub.sort_values("timestamp").reset_index(drop=True)


def load_sequence(seq_root: Path | str, mode: str = "rtk") -> OxfordSequence:
    """Load one Oxford RobotCar run.

    seq_root example: /mnt/Data/oxford_robotcar/2014-05-06-12-54-54

    Modes:
      - 'rtk': pose and velocity both from rtk.csv (post-processed RTK GT, coupled regime)
      - 'ins': pose and velocity both from ins.csv (online INS, coupled regime)
      - 'rtk_ins': pose from rtk.csv, velocity from ins.csv (separated regime)
    """
    seq_root = Path(seq_root)
    ins_path = seq_root / "ins.csv"
    rtk_path = seq_root / "rtk.csv"

    if mode not in {"ins", "rtk_ins", "rtk"}:
        raise ValueError(f"unknown mode '{mode}', expected 'ins', 'rtk', or 'rtk_ins'")
    if mode in {"ins", "rtk_ins"} and not ins_path.exists():
        raise FileNotFoundError(ins_path)
    if mode in {"rtk", "rtk_ins"} and not rtk_path.exists():
        raise FileNotFoundError(rtk_path)
    ins_df = _read_oxford_csv(ins_path) if ins_path.exists() else None

    if mode == "rtk":
        rtk_df = _read_oxford_csv(rtk_path)
        t_us = rtk_df["timestamp"].values.astype(np.int64)
        t_s = (t_us - t_us[0]) / 1e6
        x = rtk_df["easting"].values.astype(np.float64)
        y = rtk_df["northing"].values.astype(np.float64)
        vn = rtk_df["velocity_north"].values.astype(np.float64)
        ve = rtk_df["velocity_east"].values.astype(np.float64)
        vd = rtk_df["velocity_down"].values.astype(np.float64) \
            if "velocity_down" in rtk_df.columns else np.zeros_like(t_s)
        yaw = rtk_df["yaw"].values.astype(np.float64) if "yaw" in rtk_df.columns \
            else np.zeros_like(t_s)
        pose_source = "rtk"
        n_dropped_ins = 0
        n_dropped_rtk = max(0, len(pd.read_csv(rtk_path)) - len(rtk_df))
    elif mode == "rtk_ins":
        assert ins_df is not None
        if not rtk_path.exists():
            raise FileNotFoundError(rtk_path)
        rtk_df = _read_oxford_csv(rtk_path)
        # Resample INS onto RTK timestamps via linear interpolation.
        t_us = rtk_df["timestamp"].values.astype(np.int64)
        t_s = (t_us - t_us[0]) / 1e6
        x = rtk_df["easting"].values.astype(np.float64)
        y = rtk_df["northing"].values.astype(np.float64)
        # Interpolate INS velocity at RTK timestamps.
        ins_t_us = ins_df["timestamp"].values.astype(np.int64)
        ins_t_s_at_rtk0 = (ins_t_us - t_us[0]) / 1e6
        vn = np.interp(t_s, ins_t_s_at_rtk0, ins_df["velocity_north"].values.astype(np.float64))
        ve = np.interp(t_s, ins_t_s_at_rtk0, ins_df["velocity_east"].values.astype(np.float64))
        vd = np.interp(t_s, ins_t_s_at_rtk0, ins_df["velocity_down"].values.astype(np.float64))
        yaw = rtk_df["yaw"].values.astype(np.float64) if "yaw" in rtk_df.columns \
            else np.zeros_like(t_s)
        pose_source = "rtk"
        n_dropped_ins = max(0, len(pd.read_csv(ins_path)) - len(ins_df))
        n_dropped_rtk = max(0, len(pd.read_csv(rtk_path)) - len(rtk_df))
    else:
        assert ins_df is not None
        t_us = ins_df["timestamp"].values.astype(np.int64)
        t_s = (t_us - t_us[0]) / 1e6
        x = ins_df["easting"].values.astype(np.float64)
        y = ins_df["northing"].values.astype(np.float64)
        vn = ins_df["velocity_north"].values.astype(np.float64)
        ve = ins_df["velocity_east"].values.astype(np.float64)
        vd = ins_df["velocity_down"].values.astype(np.float64)
        yaw = ins_df["yaw"].values.astype(np.float64) if "yaw" in ins_df.columns \
            else np.zeros_like(t_s)
        pose_source = "ins"
        n_dropped_ins = max(0, len(pd.read_csv(ins_path)) - len(ins_df))
        n_dropped_rtk = 0

    # Strict monotonic time.
    keep = np.concatenate([[True], np.diff(t_s) > 0])
    if not keep.all():
        t_s = t_s[keep]
        x = x[keep]; y = y[keep]
        vn = vn[keep]; ve = ve[keep]; vd = vd[keep]
        yaw = yaw[keep]

    return OxfordSequence(
        name=seq_root.name,
        pose=Pose2D(t_s, x, y),
        vel_north=vn,
        vel_east=ve,
        vel_down=vd,
        yaw=yaw,
        pose_source=pose_source,
        n_dropped_ins=n_dropped_ins,
        n_dropped_rtk=n_dropped_rtk,
    )


def interpolate_ref_to_pose(seq: OxfordSequence) -> np.ndarray:
    """Reference horizontal speed at pose timestamps (already aligned)."""
    return seq.v_horiz


DEFAULT_RUNS = (
    "2014-11-25-09-18-32",
    "2014-12-09-13-21-02",
    "2015-02-13-09-16-26",
)
