"""Boreas dataset loader (applanix/gps_post_process.csv).

Boreas provides post-processed GNSS/INS (Applanix POSPac) at ~200 Hz with
easting, northing, altitude and vel_east, vel_north, vel_up in a common frame.
Pose and velocity are produced by the same batch smoother, so this dataset
serves as a *coupled-pipeline* cross-dataset check (contrast with HeLiPR's
LiDAR-pose vs INSPVA-velocity split).

Default behaviour downsamples to 10 Hz to match KITTI/HeLiPR for a fair
apples-to-apples comparison with the main paper results.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from velref.core.trajectory import Pose2D


@dataclass
class BoreasSequence:
    name: str
    pose: Pose2D
    vel_east: np.ndarray   # POSPac velocity east [m/s]
    vel_north: np.ndarray  # POSPac velocity north [m/s]
    heading: np.ndarray    # rad
    native_rate_hz: float  # before downsampling

    @property
    def v_horiz(self) -> np.ndarray:
        return np.hypot(self.vel_east, self.vel_north)

    @property
    def v_ins_horiz(self) -> np.ndarray:
        return self.v_horiz


def load_sequence(seq_root: Path | str, target_hz: float = 10.0) -> BoreasSequence:
    """Load one Boreas sequence (expects applanix/gps_post_process.csv).

    seq_root example: /mnt/Data/boreas/boreas-2020-11-26-13-58
    target_hz: downsampled output rate. Boreas is ~200 Hz natively; 10 Hz
      matches HeLiPR/KITTI. Use 0 or native rate to skip decimation.
    """
    seq_root = Path(seq_root)
    csv_path = seq_root / "applanix" / "gps_post_process.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    t = df["GPSTime"].values.astype(np.float64)
    native_rate = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 0.0

    if target_hz > 0 and native_rate > target_hz * 1.5:
        step = int(round(native_rate / target_hz))
        idx = np.arange(0, len(df), step)
        df = df.iloc[idx].reset_index(drop=True)
        t = df["GPSTime"].values.astype(np.float64)

    # Relative time starting at 0.
    t_rel = t - t[0]

    # Enforce strict monotonic.
    keep = np.concatenate([[True], np.diff(t_rel) > 0])
    if not keep.all():
        df = df.loc[keep].reset_index(drop=True)
        t_rel = t_rel[keep]

    x = df["easting"].values.astype(np.float64)
    y = df["northing"].values.astype(np.float64)
    ve = df["vel_east"].values.astype(np.float64)
    vn = df["vel_north"].values.astype(np.float64)
    hd = df["heading"].values.astype(np.float64)

    return BoreasSequence(
        name=seq_root.name,
        pose=Pose2D(t_rel, x, y),
        vel_east=ve,
        vel_north=vn,
        heading=hd,
        native_rate_hz=native_rate,
    )


def interpolate_ref_to_pose(seq: BoreasSequence) -> np.ndarray:
    """POSPac velocity is at pose timestamps (same row), so just return horizontal speed."""
    return seq.v_horiz


DEFAULT_SEQUENCES = (
    "boreas-2020-11-26-13-58",
    "boreas-2021-06-17-17-52",
    "boreas-2021-09-02-11-42",
)
