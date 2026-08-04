"""nuScenes CAN bus expansion loader.

Each scene-XXXX_pose.json contains ~20 s at ~50 Hz of records with keys:
  utime (microseconds), pos [x, y, z], vel [vx, vy, vz] (body frame),
  accel, orientation, rotation_rate.

The pose stream comes from nuScenes' SLAM-aligned global pose pipeline
(offline keyframe matching against pre-built HD maps), while the velocity
stream is the CAN/IMU-fused ego-velocity. The two are produced by different
algorithms operating on different sensors -- a *separated* provenance regime
analogous to HeLiPR's LiDAR-pose vs. INSPVA pairing.

Default behaviour decimates to 10 Hz to match HeLiPR/KITTI/Boreas.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from velref.core.trajectory import Pose2D


@dataclass
class NuscenesScene:
    name: str
    pose: Pose2D
    vx_body: np.ndarray   # forward velocity, body frame [m/s]
    vy_body: np.ndarray   # lateral velocity, body frame [m/s]
    yaw: np.ndarray       # heading from quaternion-derived yaw [rad]
    native_rate_hz: float

    @property
    def v_horiz(self) -> np.ndarray:
        """Horizontal speed magnitude from CAN/IMU fusion."""
        return np.hypot(self.vx_body, self.vy_body)

    @property
    def v_ins_horiz(self) -> np.ndarray:
        return self.v_horiz


def load_scene(can_root: Path | str, scene_id: str,
               target_hz: float = 10.0) -> NuscenesScene:
    """Load one nuScenes scene's CAN-bus pose stream.

    can_root example: /mnt/Data/nuscenes/can_bus
    scene_id example: '0001'  (loads scene-0001_pose.json)
    target_hz: 10 by default; pass 0 to keep native rate.
    """
    can_root = Path(can_root)
    pose_path = can_root / f"scene-{scene_id}_pose.json"
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)

    with open(pose_path) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"empty scene file: {pose_path}")

    t_us = np.asarray([r["utime"] for r in records], dtype=np.int64)
    pos = np.asarray([r["pos"] for r in records], dtype=np.float64)
    vel = np.asarray([r["vel"] for r in records], dtype=np.float64)
    ori = np.asarray([r["orientation"] for r in records], dtype=np.float64)

    native_rate = 1e6 / float(np.median(np.diff(t_us))) if t_us.size > 1 else 0.0

    # Decimate by integer step.
    if target_hz > 0 and native_rate > target_hz * 1.5:
        step = max(1, int(round(native_rate / target_hz)))
        idx = np.arange(0, len(records), step)
        t_us = t_us[idx]
        pos = pos[idx]
        vel = vel[idx]
        ori = ori[idx]

    t_s = (t_us - t_us[0]) / 1e6

    # Strict monotonic.
    keep = np.concatenate([[True], np.diff(t_s) > 0])
    if not keep.all():
        t_s = t_s[keep]; pos = pos[keep]; vel = vel[keep]; ori = ori[keep]

    return NuscenesScene(
        name=f"scene-{scene_id}",
        pose=Pose2D(t_s, pos[:, 0], pos[:, 1]),
        vx_body=vel[:, 0],
        vy_body=vel[:, 1],
        yaw=ori[:, 0],
        native_rate_hz=native_rate,
    )


def interpolate_ref_to_pose(seq: NuscenesScene) -> np.ndarray:
    return seq.v_horiz


# Six scenes spanning urban / highway / stop-and-go from the trainval split.
DEFAULT_SCENES = (
    "0001",
    "0061",
    "0103",
    "0500",
    "0750",
    "1000",
)
