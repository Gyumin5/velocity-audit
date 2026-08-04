#!/usr/bin/env python3
"""Run window/degree sensitivity and L1.8 degradation across all 6 HeLiPR sequences.

Replaces the single-sequence (roundabout01) analysis with a 6-sequence sweep
so the paper can report median+IQR aggregates rather than a single excerpt.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from velref.io.helipr import load_sequence  # type: ignore  # noqa: E402
from scripts.run_analytics import (  # type: ignore  # noqa: E402
    DEFAULT_SEQUENCES,
    experiment_sensitivity,
    experiment_degradation,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/helipr"))
    ap.add_argument("--out", type=Path, default=Path("results/helipr"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sens_all, deg_all = [], []
    for seq_name in DEFAULT_SEQUENCES:
        seq_path = args.root / seq_name
        if not seq_path.exists():
            print(f"SKIP {seq_name} (missing)")
            continue
        print(f"Loading {seq_name} ...", flush=True)
        s = load_sequence(seq_path, lidar="Ouster", use_global=True)
        sens = experiment_sensitivity(s, args.out)
        sens["sequence"] = seq_name
        sens_all.append(sens)
        deg = experiment_degradation(s, args.out)
        deg["sequence"] = seq_name
        deg_all.append(deg)

    if sens_all:
        out = pd.concat(sens_all, ignore_index=True)
        out.to_csv(args.out / "sensitivity_all.csv", index=False)
        print(f"\nSensitivity all: {len(out)} rows -> sensitivity_all.csv")
        agg = out.groupby(["W", "degree"])["rmse_vs_ins"].agg(
            ["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
        )
        agg.columns = ["median", "q1", "q3"]
        agg.to_csv(args.out / "sensitivity_aggregate.csv")
        print(f"Aggregated sensitivity (W,degree) -> sensitivity_aggregate.csv")

    if deg_all:
        out = pd.concat(deg_all, ignore_index=True)
        out.to_csv(args.out / "l18_degradation_all.csv", index=False)
        print(f"\nDegradation all: {len(out)} rows -> l18_degradation_all.csv")
        agg = out.groupby(["ds_factor", "noise_sigma", "jitter_sigma", "method"])[
            "rmse"
        ].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        agg.columns = ["median", "q1", "q3"]
        agg.to_csv(args.out / "l18_degradation_aggregate.csv")
        print(f"Aggregated degradation -> l18_degradation_aggregate.csv")


if __name__ == "__main__":
    main()
