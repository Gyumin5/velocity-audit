#!/usr/bin/env python3
"""Build the Pit30M 10 Hz analysis series from the archived 100 Hz per-frame streams.

The audit fixes the probe's window by duration, not by sample count: W=5 spans
1.0 s on the six 10 Hz releases. Pit30M's archived per-frame streams are at the
platform's native 100 Hz, where the same W=5 would span only 0.1 s, so they are
decimated to the 10 Hz analysis cadence first, exactly as Boreas and nuScenes
are. The phase is fixed by rule -- keep every tenth sample starting from the
first -- and not chosen from the result; scripts/pit30m_stride_offset_sensitivity.py
reports the spread over all ten phases.

Writes results/pit30m_10hz/per_frame_*.parquet with the same schema as the
input, so every downstream aggregation and figure script consumes it unchanged.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from velref.core.trajectory import Pose2D  # noqa: E402
from velref.methods.family_a import family_a_pointwise  # noqa: E402
from velref.methods.baselines import central_diff  # noqa: E402

STRIDE = 10   # 100 Hz -> 10 Hz
OFFSET = 0    # first sample, same rule as the other decimated releases


def main() -> None:
    src = REPO_ROOT / "results" / "pit30m"
    dst = REPO_ROOT / "results" / "pit30m_10hz"
    dst.mkdir(parents=True, exist_ok=True)

    n_out = 0
    for f in sorted(src.glob("per_frame_*.parquet")):
        d = pd.read_parquet(f)
        sl = slice(OFFSET, None, STRIDE)
        t, x, y = (d[c].to_numpy()[sl] for c in ("t", "x", "y"))
        v_ref = d["v_ref"].to_numpy()[sl]
        if len(t) < 60:
            continue
        pose = Pose2D(t=t, x=x, y=y)
        out = pd.DataFrame({
            "t": t, "x": x, "y": y, "v_ref": v_ref,
            "v_central": central_diff(pose),
            "v_family_a_W3": family_a_pointwise(pose, W=3),
            "v_family_a_W5": family_a_pointwise(pose, W=5),
            "v_family_a_W7": family_a_pointwise(pose, W=7),
        })
        out.to_parquet(dst / f.name, index=False)
        n_out += 1

    print(f"[Pit30M] wrote {n_out} decimated segments to {dst}")
    dts = []
    for f in sorted(dst.glob("per_frame_*.parquet")):
        dts.append(np.median(np.diff(pd.read_parquet(f)["t"].to_numpy())))
    print(f"median sample interval: {np.median(dts):.4f} s "
          f"-> W=5 window spans {10 * np.median(dts):.2f} s")


if __name__ == "__main__":
    main()
