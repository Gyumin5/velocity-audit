#!/usr/bin/env python3
"""Curvature-binned residual change for every audited release.

Recomputes the panel behind Fig. 3 from the committed per-frame parquet files
in results/<dataset>/per_frame_*.parquet, so the figure no longer depends on a
hardcoded table. Only the pose stream and the published velocity are used --
the same inputs the audit itself consumes -- and the binning matches
scripts/run_analytics.py::experiment_curvature_bins.

Writes results/curvature_bins_all_datasets.csv (one row per dataset x bin x
method) and prints the per-bin residual reduction of Family A W=5 relative to
central differencing.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from velref.core.curvature import estimate_curvature
from velref.core.trajectory import Pose2D

# display name -> (results subdirectory, published-velocity column)
DATASETS = [
    ("HeLiPR",    "helipr",       "v_ins"),
    ("Oxford",    "oxford_x11",   "v_ins"),
    ("nuScenes",  "nuscenes_x20", "v_can"),
    ("KITTI raw", "kitti",        "v_oxts"),
    ("KITTI-360", "kitti360",     "v_ref"),
    ("Boreas",    "boreas",       "v_ref"),
    ("Pit30M",    "pit30m",       "v_ref"),
]

BIN_EDGES = np.array([0.0, 1e-3, 5e-3, 2e-2, 0.1, 1.0])
BIN_LABELS = [f"{BIN_EDGES[i]:.0e}-{BIN_EDGES[i + 1]:.0e}" for i in range(len(BIN_EDGES) - 1)]
MIN_SAMPLES = 20


def per_sequence_rows(path: Path, dataset: str, ref_col: str) -> list[dict]:
    df = pd.read_parquet(path)
    kappa = np.abs(estimate_curvature(Pose2D(t=df["t"].to_numpy(),
                                             x=df["x"].to_numpy(),
                                             y=df["y"].to_numpy()),
                                      window=9, polyorder=3))
    bins = np.clip(np.digitize(kappa, BIN_EDGES) - 1, 0, len(BIN_LABELS) - 1)
    v_ref = df[ref_col].to_numpy()
    rows = []
    for method in ("v_central", "v_family_a_W5"):
        v_hat = df[method].to_numpy()
        ok = np.isfinite(v_hat) & np.isfinite(v_ref)
        for b, label in enumerate(BIN_LABELS):
            mask = (bins == b) & ok
            if mask.sum() < MIN_SAMPLES:
                continue
            rows.append({
                "dataset": dataset,
                "sequence": path.stem.replace("per_frame_", ""),
                "method": method[2:],
                "kappa_bin": label,
                "n": int(mask.sum()),
                "rmse_vs_ref": float(np.sqrt(np.mean((v_hat[mask] - v_ref[mask]) ** 2))),
            })
    return rows


def main() -> None:
    rows: list[dict] = []
    for dataset, subdir, ref_col in DATASETS:
        files = sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet"))
        if not files:
            print(f"{dataset}: no per-frame files, skipped")
            continue
        for f in files:
            rows.extend(per_sequence_rows(f, dataset, ref_col))
        print(f"{dataset}: {len(files)} sequences")

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "curvature_bins_all_datasets.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows)\n")

    # Reduction of the medians, matching the aggregation used elsewhere in the paper.
    med = df.groupby(["dataset", "kappa_bin", "method"])["rmse_vs_ref"].median().unstack("method")
    med["reduction_pct"] = (med["central"] - med["family_a_W5"]) / med["central"] * 100.0
    print(f"{'dataset':<10}" + "".join(f"{b:>14}" for b in BIN_LABELS))
    for dataset, _, _ in DATASETS:
        if dataset not in med.index.get_level_values(0):
            continue
        cells = []
        for b in BIN_LABELS:
            try:
                cells.append(f"{med.loc[(dataset, b), 'reduction_pct']:>13.0f}%")
            except KeyError:
                cells.append(f"{'--':>14}")
        print(f"{dataset:<10}" + "".join(cells))


if __name__ == "__main__":
    main()
