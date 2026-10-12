#!/usr/bin/env python3
"""What is left of the HeLiPR residual once pose and velocity share one INS.

Reviewer 1 (comment 2) and reviewer 2 (comment 3) ask for the documentation
that supports each coupling label.  Reading HeLiPR's again (Jung et al. 2024,
Sec. 4.1.4 and 4.3) the pose we audit is not an independent LiDAR-inertial
product: the LiDAR ground truth is the INS position B-spline-interpolated to
each LiDAR's acquisition time and expressed at that LiDAR through the extrinsic
calibration, and the published velocity is the INSPVA stream of the same
SPAN-CPT7 unit.  One source, two exports.

That raises a question the manuscript has to answer: if both streams come from
one INS, why is the residual 0.0486 m/s rather than Boreas-like 0.011 m/s?  Two
documented transforms stand between the two exports, and they leave different
signatures:

  lever arm   the pose is at the LiDAR, the velocity at the INS reference
              point, so a rotation adds omega x r --- the residual then grows
              with yaw rate;
  resampling  the pose is a spline sampled at the LiDAR cadence while the
              velocity is published at the INS cadence, so the residual then
              grows with how fast the speed itself is changing.

We bin the per-frame residual by both and report the two profiles.  Nothing
here identifies the pipeline; it separates which documented transform the
residual tracks.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]      # quantile edges


def unwrap_yaw_rate(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Heading rate from the pose track itself [rad/s]."""
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    yaw = np.unwrap(np.arctan2(dy, dx))
    return np.gradient(yaw, t)


def profile(key: np.ndarray, resid: np.ndarray) -> list[float]:
    edges = np.quantile(key, BINS)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key <= hi) if hi == edges[-1] else (key >= lo) & (key < hi)
        out.append(float(np.sqrt(np.mean(resid[m] ** 2))) if m.sum() >= 20 else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args()

    rows = []
    for p in sorted((REPO_ROOT / "results" / "helipr").glob("per_frame_*.parquet")):
        d = pd.read_parquet(p)
        t, x, y = d["t"].to_numpy(), d["x"].to_numpy(), d["y"].to_numpy()
        v_ins = d["v_ins"].to_numpy()
        yaw_rate = np.abs(unwrap_yaw_rate(t, x, y))
        accel = np.abs(np.gradient(v_ins, t))
        for op in ("v_central", "v_family_a_W5"):
            r = d[op].to_numpy() - v_ins
            ok = np.isfinite(r) & np.isfinite(yaw_rate) & np.isfinite(accel)
            rows.append({
                "sequence": p.stem[len("per_frame_"):], "operator": op,
                "rmse_all": float(np.sqrt(np.mean(r[ok] ** 2))),
                **{f"yawrate_q{i+1}": v for i, v in
                   enumerate(profile(yaw_rate[ok], r[ok]))},
                **{f"accel_q{i+1}": v for i, v in
                   enumerate(profile(accel[ok], r[ok]))},
            })

    df = pd.DataFrame(rows)
    for op in ("v_central", "v_family_a_W5"):
        s = df[df.operator == op]
        yw = [s[f"yawrate_q{i+1}"].median() for i in range(len(BINS) - 1)]
        ac = [s[f"accel_q{i+1}"].median() for i in range(len(BINS) - 1)]
        print(f"{op}  overall {s.rmse_all.median():.4f} m/s")
        print("   by |yaw rate| quintile    " + "  ".join(f"{v:.4f}" for v in yw)
              + f"   ratio top/bottom {yw[-1]/yw[0]:.2f}")
        print("   by |dv/dt| quintile       " + "  ".join(f"{v:.4f}" for v in ac)
              + f"   ratio top/bottom {ac[-1]/ac[0]:.2f}")

    df.to_csv(args.out / "helipr_shared_source_check.csv", index=False)
    print(f"\nwrote {args.out/'helipr_shared_source_check.csv'}")


if __name__ == "__main__":
    main()
