"""KITTI raw dataset loader (pose from OXTS lat/lon, velocity from OXTS vf)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import datetime
import numpy as np

from velref.core.trajectory import Pose2D


EARTH_RADIUS = 6378137.0


@dataclass
class KittiSequence:
    name: str
    pose: Pose2D
    yaw: np.ndarray          # heading from OXTS [rad]
    vf: np.ndarray           # forward velocity [m/s] at pose timestamps (independent ref)
    vn: np.ndarray           # north velocity
    ve: np.ndarray           # east velocity
    pos_accuracy: np.ndarray # [m]
    vel_accuracy: np.ndarray # [m/s]

    @property
    def v_horiz(self) -> np.ndarray:
        """Horizontal speed (magnitude of vn, ve)."""
        return np.hypot(self.vn, self.ve)

    @property
    def v_ins_horiz(self) -> np.ndarray:
        """Alias for compatibility with HeLiPR pipeline."""
        return self.v_horiz


def _mercator_xy(lat_deg: np.ndarray, lon_deg: np.ndarray,
                 lat0_deg: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular projection around (lat0, lon0). Meters.

    Accurate to < 0.1 m drift over 5 km at city latitudes.
    """
    lat0 = np.deg2rad(lat0_deg)
    x = EARTH_RADIUS * np.deg2rad(lon_deg - lon0_deg) * np.cos(lat0)
    y = EARTH_RADIUS * np.deg2rad(lat_deg - lat0_deg)
    return x, y


def _parse_timestamps(ts_path: Path) -> np.ndarray:
    """KITTI timestamps.txt -> seconds since first frame."""
    rows = []
    with open(ts_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Format: "2011-09-30 12:34:56.789012345"
            date_part, time_part = s.split(" ")
            hh, mm, ss = time_part.split(":")
            sec_int = int(float(ss))
            frac_ns = int(round((float(ss) - sec_int) * 1e9))
            dt = datetime.datetime.fromisoformat(f"{date_part} {hh}:{mm}:{sec_int:02d}")
            rows.append(dt.timestamp() * 1e9 + frac_ns)
    t_ns = np.asarray(rows, dtype=np.int64)
    return t_ns


def load_sequence(seq_root: Path | str) -> KittiSequence:
    """Load one KITTI raw drive (expects oxts/data/*.txt + oxts/timestamps.txt).

    seq_root example: /mnt/Data/kitti/2011_09_30/2011_09_30_drive_0028_sync
    """
    seq_root = Path(seq_root)
    oxts_dir = seq_root / "oxts" / "data"
    ts_path = seq_root / "oxts" / "timestamps.txt"
    if not oxts_dir.is_dir():
        raise FileNotFoundError(oxts_dir)
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)

    t_ns = _parse_timestamps(ts_path)
    n = t_ns.size
    files = sorted(oxts_dir.glob("*.txt"))
    if len(files) != n:
        raise ValueError(f"oxts count ({len(files)}) != timestamps ({n}) in {seq_root}")

    rows = np.empty((n, 30), dtype=np.float64)
    rows.fill(np.nan)
    for i, fp in enumerate(files):
        with open(fp, "r") as f:
            parts = f.readline().strip().split()
        arr = np.asarray([float(p) for p in parts[:30]], dtype=np.float64)
        rows[i, : arr.size] = arr

    lat = rows[:, 0]
    lon = rows[:, 1]
    yaw = rows[:, 5]
    vn = rows[:, 6]
    ve = rows[:, 7]
    vf = rows[:, 8]
    pos_accuracy = rows[:, 23]
    vel_accuracy = rows[:, 24]

    # Local planar frame anchored at first pose.
    lat0, lon0 = float(lat[0]), float(lon[0])
    x, y = _mercator_xy(lat, lon, lat0, lon0)

    t_s = (t_ns - t_ns[0]) / 1e9

    # Enforce strict monotonic time.
    keep = np.concatenate([[True], np.diff(t_s) > 0])
    if not keep.all():
        t_s = t_s[keep]
        x = x[keep]; y = y[keep]; yaw = yaw[keep]
        vn = vn[keep]; ve = ve[keep]; vf = vf[keep]
        pos_accuracy = pos_accuracy[keep]; vel_accuracy = vel_accuracy[keep]

    return KittiSequence(
        name=seq_root.name,
        pose=Pose2D(t_s, x, y),
        yaw=yaw,
        vf=vf,
        vn=vn,
        ve=ve,
        pos_accuracy=pos_accuracy,
        vel_accuracy=vel_accuracy,
    )


def interpolate_ref_to_pose(seq: KittiSequence, which: str = "vf") -> np.ndarray:
    """Return the chosen reference speed at pose timestamps.

    KITTI OXTS is sampled at pose cadence (10 Hz), so no interpolation is needed;
    we keep the same function name as HeLiPR for pipeline compatibility.
    """
    if which == "vf":
        return seq.vf.copy()
    if which == "horiz":
        return seq.v_horiz
    raise ValueError(f"unknown reference '{which}'")


DEFAULT_DRIVES = (
    "2011_09_30/2011_09_30_drive_0028_sync",
    "2011_10_03/2011_10_03_drive_0027_sync",
    "2011_10_03/2011_10_03_drive_0034_sync",
)
