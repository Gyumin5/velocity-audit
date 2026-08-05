#!/usr/bin/env python3
"""Content digests for the per-frame streams the analysis runs on.

The per-frame streams are not redistributed here (they carry each source
release's own pose and velocity content), so this manifest is what makes the
reproduction path checkable: regenerate a stream from your own copy of the
release, run this script, and compare the digest against the committed row.

Two digests are recorded per sequence:

  input_sha256   the source-derived content -- t, x, y, and the published
                 velocity -- which anyone with the release should reproduce
  derived_sha256 the estimator columns this repository computes from that input,
                 which anyone running the code on a matching input should reproduce

The digest is taken over the float64 little-endian bytes of each column in a
fixed order, not over the file bytes, so it does not depend on the parquet
writer or its version. Values are rounded to 1e-9 first: reordered floating-point
summation in a different BLAS build can move the last bit or two, and that is not
a reproduction failure.

Writes results/per_frame_manifest.csv.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# results subdirectory -> published-velocity column
SERIES = {
    "helipr": "v_ins", "oxford_x11": "v_ins", "nuscenes_x20": "v_can",
    "kitti": "v_oxts", "kitti360": "v_ref", "boreas": "v_ref",
    "pit30m_10hz": "v_ref", "pit30m": "v_ref",
}
INPUT_COLS = ("t", "x", "y")
ROUND_DECIMALS = 9


def digest(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha256()
    for c in cols:
        a = np.ascontiguousarray(df[c].to_numpy(dtype=np.float64))
        h.update(c.encode())
        h.update(np.round(a, ROUND_DECIMALS).astype("<f8").tobytes())
    return h.hexdigest()


def main() -> None:
    rows = []
    for subdir, ref_col in SERIES.items():
        files = sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet"))
        if not files:
            print(f"{subdir}: no per-frame streams, skipped")
            continue
        for p in files:
            d = pd.read_parquet(p)
            derived = [c for c in d.columns if c.startswith("v_") and c != ref_col]
            rows.append({
                "series": subdir,
                "sequence": p.stem.replace("per_frame_", ""),
                "n_rows": len(d),
                "t_start": float(d["t"].iloc[0]),
                "t_end": float(d["t"].iloc[-1]),
                "ref_col": ref_col,
                "input_sha256": digest(d, [*INPUT_COLS, ref_col]),
                "derived_cols": ";".join(derived),
                "derived_sha256": digest(d, derived),
            })
        print(f"{subdir:<14} {len(files):3d} sequences")

    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / "results" / "per_frame_manifest.csv", index=False)
    print(f"\nwrote results/per_frame_manifest.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
