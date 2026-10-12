#!/usr/bin/env python3
"""Count the zero components of the nuScenes CAN pose records.

The matched-horizontal recompute of Table 7 cannot be done on nuScenes, and the
reason is a property of the published records rather than a choice: the velocity
carries no lateral or vertical component and the orientation carries no pitch.
This counts both over the whole published CAN pose set, so the manuscript's
statement rests on the records and not on a sample of them.

Usage: python3 scripts/nuscenes_vel_components.py [can_root] [out_csv]
  can_root defaults to /mnt/Data/velref/nuscenes/can_bus
  out_csv  defaults to results/nuscenes_vel_components.csv
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

DEFAULT_ROOT = pathlib.Path("/mnt/Data/velref/nuscenes/can_bus")
DEFAULT_OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "nuscenes_vel_components.csv"


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    files = sorted(root.glob("scene-*_pose.json"))
    if not files:
        sys.exit(f"no scene pose files under {root}")

    rows = 0
    vy_nz = vz_nz = qx_nz = qy_nz = 0
    max_vy = max_vz = max_qx = max_qy = 0.0

    for f in files:
        with open(f) as fh:
            recs = json.load(fh)
        vel = np.asarray([r["vel"] for r in recs], dtype=np.float64)
        ori = np.asarray([r["orientation"] for r in recs], dtype=np.float64)
        rows += len(vel)
        # vel is [vx, vy, vz] in the body frame; orientation is [w, x, y, z].
        vy_nz += int((vel[:, 1] != 0).sum())
        vz_nz += int((vel[:, 2] != 0).sum())
        qx_nz += int((ori[:, 1] != 0).sum())
        qy_nz += int((ori[:, 2] != 0).sum())
        max_vy = max(max_vy, float(np.abs(vel[:, 1]).max()))
        max_vz = max(max_vz, float(np.abs(vel[:, 2]).max()))
        max_qx = max(max_qx, float(np.abs(ori[:, 1]).max()))
        max_qy = max(max_qy, float(np.abs(ori[:, 2]).max()))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_scenes", "n_rows", "vy_nonzero", "vz_nonzero",
                    "max_abs_vy", "max_abs_vz", "quat_x_nonzero", "quat_y_nonzero",
                    "max_abs_quat_x", "max_abs_quat_y"])
        w.writerow([len(files), rows, vy_nz, vz_nz, max_vy, max_vz,
                    qx_nz, qy_nz, max_qx, max_qy])

    print(f"{len(files)} scenes, {rows} records; "
          f"vy/vz nonzero {vy_nz}/{vz_nz}, quat x/y nonzero {qx_nz}/{qy_nz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
