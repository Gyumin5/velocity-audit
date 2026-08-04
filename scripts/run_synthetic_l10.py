#!/usr/bin/env python3
"""L1.0 synthetic sweep: methods x scenarios x sampling x noise."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from velref.io.synthetic import circle, straight_variable_speed, stop_go
from velref.methods.baselines import (
    forward_diff,
    central_diff,
    cubic_spline_deriv,
    smoothing_spline_deriv,
    savgol_deriv,
)
from velref.methods.family_a import family_a_pointwise, family_a_midspan
from velref.metrics.core_metrics import rmse, bias


SCENARIOS = {
    "circle_r15_v5": lambda fs, sigma, jitter, seed: circle(
        duration=20.0, fs=fs, radius=15.0, v=5.0,
        noise_sigma=sigma, jitter_sigma=jitter, seed=seed,
    ),
    "circle_r5_v2": lambda fs, sigma, jitter, seed: circle(
        duration=20.0, fs=fs, radius=5.0, v=2.0,
        noise_sigma=sigma, jitter_sigma=jitter, seed=seed,
    ),
    "straight_var": lambda fs, sigma, jitter, seed: straight_variable_speed(
        duration=20.0, fs=fs, v0=0.5, v1=10.0,
        noise_sigma=sigma, jitter_sigma=jitter, seed=seed,
    ),
    "stop_go": lambda fs, sigma, jitter, seed: stop_go(
        duration=30.0, fs=fs, cruise=5.0,
        noise_sigma=sigma, jitter_sigma=jitter, seed=seed,
    ),
}


def run_methods(sample):
    p = sample.pose
    return {
        "forward": forward_diff(p),
        "central": central_diff(p),
        "cubic_global": cubic_spline_deriv(p),
        "smoothing_spline": smoothing_spline_deriv(p),
        "savgol_w5p2": savgol_deriv(p, window=5, polyorder=2),
        "savgol_w7p3": savgol_deriv(p, window=7, polyorder=3),
        "savgol_w9p3": savgol_deriv(p, window=9, polyorder=3),
        "savgol_w11p3": savgol_deriv(p, window=11, polyorder=3),
        "family_a_W3": family_a_pointwise(p, W=3),
        "family_a_W5": family_a_pointwise(p, W=5),
        "family_a_W7": family_a_pointwise(p, W=7),
        "family_a_midspan_W5": family_a_midspan(p, W=5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/l10"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    fs_grid = [2.0, 5.0, 10.0, 50.0]
    sigma_grid = [0.0, 0.01, 0.05, 0.20]
    jitter_grid = [0.0, 0.005]

    rows = []
    for name, make in SCENARIOS.items():
        for fs, sigma, jitter, seed in itertools.product(fs_grid, sigma_grid, jitter_grid, range(args.seeds)):
            sample = make(fs, sigma, jitter, seed)
            methods = run_methods(sample)
            interior = slice(max(5, int(0.05 * sample.pose.t.size)), -max(5, int(0.05 * sample.pose.t.size)))
            for mname, v_hat in methods.items():
                if v_hat.shape != sample.v_true.shape:
                    continue
                rows.append({
                    "scenario": name,
                    "fs": fs,
                    "noise_sigma": sigma,
                    "jitter_sigma": jitter,
                    "seed": seed,
                    "method": mname,
                    "rmse": rmse(v_hat[interior], sample.v_true[interior]),
                    "bias": bias(v_hat[interior], sample.v_true[interior]),
                    "n": int(sample.pose.t.size),
                })
    df = pd.DataFrame(rows)
    df.to_parquet(args.out / "results.parquet", index=False)
    df.to_csv(args.out / "results.csv", index=False)

    # Aggregate.
    agg = (
        df.groupby(["scenario", "method", "fs", "noise_sigma"])
        .agg(rmse_med=("rmse", "median"), rmse_iqr=("rmse", lambda x: np.subtract(*np.percentile(x, [75, 25]))),
             bias_med=("bias", "median"), n_seeds=("seed", "count"))
        .reset_index()
    )
    agg.to_csv(args.out / "summary.csv", index=False)
    print(f"Wrote {len(df)} rows to {args.out}/results.parquet")
    print(f"Summary rows: {len(agg)}")
    print("\nTop-line: median RMSE per method per scenario at fs=10, sigma=0.05, jitter=0:")
    slice_df = df[(df.fs == 10.0) & (df.noise_sigma == 0.05) & (df.jitter_sigma == 0.0)]
    top = slice_df.groupby(["scenario", "method"])["rmse"].median().unstack(0)
    print(top.round(4).to_string())


if __name__ == "__main__":
    main()
