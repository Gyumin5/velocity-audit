#!/usr/bin/env python3
"""Integrated-speed drift per release, from the committed per-frame files.

For each sequence, trapezoidally integrates each operator's speed over the pose
timestamps and compares the total against the polyline length L_poly of the same
pose stream. The reported quantity is the relative final drift

    |int v_hat dt - L_poly| / L_poly,

median over the sequences of a release. This is the quantity behind the
"residual versus integrated drift" paragraph: it is computed from the pose
stream alone and does not consult the published velocity channel.

Writes results/integrated_drift.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("HeLiPR", "helipr"),
    ("Oxford", "oxford_x11"),
    ("nuScenes", "nuscenes_x20"),
    ("KITTI raw", "kitti"),
    ("KITTI-360", "kitti360"),
    ("Boreas", "boreas"),
    ("Pit30M", "pit30m_10hz"),
]

METHODS = ("v_central", "v_family_a_W5")


def main() -> None:
    rows = []
    for dataset, subdir in DATASETS:
        for f in sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
            df = pd.read_parquet(f)
            t = df["t"].to_numpy()
            x, y = df["x"].to_numpy(), df["y"].to_numpy()
            l_poly = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
            if l_poly <= 0:
                continue
            row = {"dataset": dataset, "sequence": f.stem.replace("per_frame_", ""),
                   "L_poly": l_poly}
            for m in METHODS:
                v = df[m].to_numpy()
                ok = np.isfinite(v)
                if ok.sum() < 3:
                    continue
                row[m[2:]] = abs(float(np.trapz(v[ok], t[ok])) - l_poly) / l_poly * 100.0
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / "results" / "integrated_drift.csv", index=False)
    med = out.groupby("dataset")[["central", "family_a_W5"]].median()
    print("relative final drift [% of L_poly], median over sequences\n")
    print(med.round(3).to_string())


if __name__ == "__main__":
    main()
