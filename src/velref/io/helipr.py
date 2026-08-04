"""HeLiPR dataset loader (pose from LiDAR_GT, velocity from inspva.csv)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from velref.core.trajectory import Pose2D


@dataclass
class HelipSequence:
    name: str
    pose: Pose2D
    yaw: np.ndarray  # from quaternion (heading in radians)
    t_ins: np.ndarray  # INS timestamps [s from pose_start]
    vn: np.ndarray  # north velocity [m/s]
    ve: np.ndarray  # east velocity [m/s]
    vu: np.ndarray  # up velocity [m/s]

    @property
    def v_ins_horiz(self) -> np.ndarray:
        return np.hypot(self.vn, self.ve)


def _quat_to_yaw(qx, qy, qz, qw):
    # ZYX Tait-Bryan yaw.
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny, cosy)


def load_sequence(
    seq_root: Path | str,
    lidar: str = "Ouster",
    use_global: bool = True,
) -> HelipSequence:
    """Load one HeLiPR sequence.

    seq_root e.g. /mnt/Data/mulran/roundabout01.
    lidar e.g. 'Ouster', 'Aeva', 'Avia', 'Velodyne'.
    use_global = True uses global_<lidar>_gt.txt (UTM frame).
    """
    seq_root = Path(seq_root)
    prefix = "global_" if use_global else ""
    pose_path = seq_root / "LiDAR_GT" / f"{prefix}{lidar}_gt.txt"
    ins_path = seq_root / "Inertial_data" / "inspva.csv"
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    if not ins_path.exists():
        raise FileNotFoundError(ins_path)

    pose = np.loadtxt(pose_path)
    t_pose_ns = pose[:, 0].astype(np.int64)
    x = pose[:, 1]
    y = pose[:, 2]
    qx, qy, qz, qw = pose[:, 4], pose[:, 5], pose[:, 6], pose[:, 7]
    yaw = _quat_to_yaw(qx, qy, qz, qw)

    # Time origin at pose[0].
    t0 = t_pose_ns[0]
    t_pose_s = (t_pose_ns - t0) / 1e9

    # Some sequences have duplicate timestamps; keep the first of each.
    _, uniq = np.unique(t_pose_s, return_index=True)
    uniq = np.sort(uniq)
    t_pose_s = t_pose_s[uniq]
    x = x[uniq]
    y = y[uniq]
    yaw = yaw[uniq]

    # Enforce strict increase defensively.
    keep = np.concatenate([[True], np.diff(t_pose_s) > 0])
    t_pose_s = t_pose_s[keep]
    x = x[keep]
    y = y[keep]
    yaw = yaw[keep]

    # Parse inspva.csv: timestamp_ns, lat, lon, alt, vn, ve, vu, roll, pitch, azim, "status: N"
    ins_rows = []
    with open(ins_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            try:
                ts = int(parts[0])
                vn = float(parts[4])
                ve = float(parts[5])
                vu = float(parts[6])
            except (ValueError, IndexError):
                continue
            ins_rows.append((ts, vn, ve, vu))
    ins = np.asarray(ins_rows, dtype=np.float64)
    t_ins_s = (ins[:, 0].astype(np.int64) - t0) / 1e9
    vn = ins[:, 1]
    ve = ins[:, 2]
    vu = ins[:, 3]

    # Trim INS to pose span.
    span_mask = (t_ins_s >= t_pose_s[0] - 1.0) & (t_ins_s <= t_pose_s[-1] + 1.0)
    t_ins_s = t_ins_s[span_mask]
    vn = vn[span_mask]
    ve = ve[span_mask]
    vu = vu[span_mask]

    return HelipSequence(
        name=seq_root.name,
        pose=Pose2D(t_pose_s, x, y),
        yaw=yaw,
        t_ins=t_ins_s,
        vn=vn,
        ve=ve,
        vu=vu,
    )


def interpolate_ins_to_pose(seq: HelipSequence) -> np.ndarray:
    """Resample INS horizontal speed onto pose timestamps."""
    v_h = seq.v_ins_horiz
    return np.interp(seq.pose.t, seq.t_ins, v_h)


DEFAULT_SEQUENCES = (
    "bridge01",
    "dcc05",
    "kaist05",
    "riverside05",
    "roundabout01",
    "town01",
)
