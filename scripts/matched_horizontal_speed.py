#!/usr/bin/env python3
"""Recompute the OXTS releases against a matched horizontal-speed reference.

Reviewer 1 (comment 3) and reviewer 2 (comment 4): the probe differentiates the
planar pose and so produces a horizontal ground speed, whereas KITTI raw and
KITTI-360 are audited against the OXTS forward component ``vf``.  Those are the
only two releases in the corpus whose reference is not already a horizontal
speed.  The OXTS record also publishes the north and east components, and
``dataformat.txt`` defines both ``vf`` and ``vl`` as parallel to the earth
surface, so

    sqrt(vn^2 + ve^2) == sqrt(vf^2 + vl^2)

is the matched quantity and the published ``vf`` differs from it by the lateral
component alone.  Vehicle pitch does not enter either side: the OXTS components
used here are already earth-surface parallel, and the probe never sees z.

We keep the audited per-frame streams exactly as they are --- same frames, same
operators, same setting --- and change only the comparison target, so the two
columns of the output differ in the reference definition and in nothing else.
Reverse motion is reported as well, since a signed forward component and an
unsigned speed part company there.

The audited streams carry time relative to the first frame, so the OXTS record
is put on the same origin before matching.  KITTI-360's extract OXTS runs at
100~Hz against a 10~Hz audited stream, so only the matched records are read off
disk rather than all of them.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ("v_central", "v_family_a_W5")


def oxts_dir(drive_dir: Path) -> Path:
    """The one directory holding timestamps.txt and data/ for this drive."""
    return next(drive_dir.rglob("oxts/timestamps.txt")).parent


def read_times(oxts: Path) -> np.ndarray:
    """OXTS record times, relative to the first record, in seconds."""
    t = np.asarray([float(pd.Timestamp(line.strip()).value) / 1e9
                    for line in (oxts / "timestamps.txt").read_text().splitlines()
                    if line.strip()])
    return t - t[0]


def read_records(oxts: Path, idx: np.ndarray) -> np.ndarray:
    """Columns vn, ve, vf, vl for the requested record indices."""
    files = sorted((oxts / "data").glob("*.txt"))
    out = np.empty((len(idx), 4), dtype=float)
    for k, i in enumerate(idx):
        v = np.fromstring(files[i].read_text().strip(), sep=" ")
        out[k] = v[6], v[7], v[8], v[9]          # vn ve vf vl
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def match_frames(stream_t: np.ndarray, oxts_t: np.ndarray) -> np.ndarray:
    """Index into the OXTS record for each audited frame, by nearest time.

    Both clocks are relative to their own first sample, which is the same frame.
    A frame landing further than half a sample interval away means the two are
    not the same recording, so we refuse the sequence rather than guess.
    """
    j = np.searchsorted(oxts_t, stream_t).clip(1, len(oxts_t) - 1)
    pick = np.where(np.abs(stream_t - oxts_t[j - 1]) <= np.abs(stream_t - oxts_t[j]),
                    j - 1, j)
    gap = float(np.max(np.abs(stream_t - oxts_t[pick])))
    if gap > 0.05:
        raise ValueError(f"audited frames do not line up with the OXTS record "
                         f"(worst gap {gap:.3f} s)")
    return pick


def process(release: str, stream_dir: Path, oxts_root: Path,
            ref_col: str) -> list[dict]:
    rows = []
    for parquet in sorted(stream_dir.glob("per_frame_*.parquet")):
        seq = parquet.stem[len("per_frame_"):]
        drive = seq.replace("_extract", "").replace("_sync", "")
        cand = [d for d in oxts_root.iterdir() if d.is_dir() and drive in d.name]
        if not cand:
            print(f"  {release} {seq}: no OXTS directory found, skipped")
            continue
        d = pd.read_parquet(parquet)
        oxts = oxts_dir(cand[0])
        try:
            pick = match_frames(d["t"].to_numpy(), read_times(oxts))
        except ValueError as e:
            print(f"  {release} {seq}: {e}, skipped")
            continue
        rec = read_records(oxts, pick)
        vn, ve, vf, vl = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
        v_horiz = np.hypot(vn, ve)
        published = d[ref_col].to_numpy()
        row = {
            "dataset": release, "sequence": seq, "n_frames": len(d),
            "published_matches_vf_rms": rmse(published, vf),
            "identity_residual_rms": rmse(np.hypot(vf, vl), v_horiz),
            "horiz_minus_vf_rms": rmse(v_horiz, vf),
            "horiz_minus_vf_median": float(np.median(v_horiz - vf)),
            "lateral_abs_median": float(np.median(np.abs(vl))),
            "reverse_frame_fraction": float(np.mean(vf < -0.1)),
        }
        # Where the two reference definitions part company.  The forward
        # component and an unsigned speed differ by 2|v| in reverse and by a
        # lateral term that only matters when the forward speed is small.
        for lo, hi, tag in ((-np.inf, 0.0, "reverse"),
                            (0.0, 0.5, "slow"), (0.5, np.inf, "cruise")):
            m = (vf >= lo) & (vf < hi)
            row[f"n_{tag}"] = int(m.sum())
            row[f"horiz_minus_vf_rms_{tag}"] = (
                rmse(v_horiz[m], vf[m]) if m.sum() else float("nan"))
            row[f"lateral_abs_median_{tag}"] = (
                float(np.median(np.abs(vl[m]))) if m.sum() else float("nan"))
        for op in OPERATORS:
            v = d[op].to_numpy()
            row[f"{op}_M4_vs_published"] = rmse(v, published)
            row[f"{op}_M4_vs_horizontal"] = rmse(v, v_horiz)
        for tag in ("published", "horizontal"):
            c = row[f"v_central_M4_vs_{tag}"]
            f = row[f"v_family_a_W5_M4_vs_{tag}"]
            row[f"delta_pct_vs_{tag}"] = (f / c - 1) * 100.0
        rows.append(row)
        print(f"  {release} {seq}: central {row['v_central_M4_vs_published']:.4f} "
              f"-> {row['v_central_M4_vs_horizontal']:.4f} m/s, "
              f"delta {row['delta_pct_vs_published']:+.2f}% -> "
              f"{row['delta_pct_vs_horizontal']:+.2f}%", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path,
                    default=REPO_ROOT / "state" / "r2-horizontal")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args()

    rows = []
    rows += process("KITTI", REPO_ROOT / "results" / "kitti",
                    args.state / "kitti_raw", "v_oxts")
    k360 = args.state / "kitti360_oxts_extract" / "unpacked"
    if k360.exists():
        rows += process("KITTI-360", REPO_ROOT / "results" / "kitti360",
                        k360, "v_ref")
    else:
        print(f"  KITTI-360: {k360} not present, skipped")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "matched_horizontal_speed.csv", index=False)
    print()
    for name, g in df.groupby("dataset"):
        print(f"{name}  n={len(g)}  "
              f"central {g.v_central_M4_vs_published.median():.4f} -> "
              f"{g.v_central_M4_vs_horizontal.median():.4f} m/s   "
              f"FamilyA {g.v_family_a_W5_M4_vs_published.median():.4f} -> "
              f"{g.v_family_a_W5_M4_vs_horizontal.median():.4f}   "
              f"delta {g.delta_pct_vs_published.median():+.2f}% -> "
              f"{g.delta_pct_vs_horizontal.median():+.2f}%   "
              f"|horiz-vf| rms {g.horiz_minus_vf_rms.median():.4f} m/s   "
              f"reverse {g.reverse_frame_fraction.median()*100:.2f}% of frames")
        print(f"          cruise-only |horiz-vf| rms "
              f"{g.horiz_minus_vf_rms_cruise.min():.5f}-"
              f"{g.horiz_minus_vf_rms_cruise.max():.5f} m/s over sequences; "
              f"identity |hypot(vf,vl)-hypot(vn,ve)| rms "
              f"max {g.identity_residual_rms.max():.5f} m/s")
    print(f"\nwrote {args.out/'matched_horizontal_speed.csv'}")


if __name__ == "__main__":
    main()
