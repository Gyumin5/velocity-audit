#!/usr/bin/env python3
"""Resampling unit, aggregation and grouping for the cross-dataset spread column.

Reviewer 1 (comment 5) of the second IEEE Access round asks three things about
the paired bootstrap behind Table~\\ref{tab:crossds}: what the resampling unit is,
how sequence-level results are aggregated and weighted, and whether Pit30M
segments cut from the same source log are grouped.  The reviewer also asks for
individual sequence values where a release has only three sequences.

This script answers all three from the per-frame streams.

Published procedure (scripts/build_crossds_tables.py): the unit is a sequence.
Each sequence contributes one median residual per operator; the release value is
the median across sequences, unweighted, so a long sequence counts the same as a
short one; the spread column is the 2.5--97.5 percentile of the ratio of medians
over 5000 resamples of the sequences with replacement.

Two things that procedure does not do are measured here:

1. Pit30M grouping.  Its 53 analysis segments come from 12 source logs, so the
   published interval treats correlated units as independent.  We recompute it as
   a cluster bootstrap that resamples whole logs.
2. Small-sample behaviour.  With three sequences the median of a resample is
   almost always one of the three values, so the interval is not centred on the
   point estimate.  We write the individual sequence values instead.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("HeLiPR",    "helipr",       "v_ins"),
    ("Oxford",    "oxford_x11",   "v_ins"),
    ("nuScenes",  "nuscenes_x20", "v_can"),
    ("KITTI",     "kitti",        "v_oxts"),
    ("KITTI-360", "kitti360",     "v_ref"),
    ("Boreas",    "boreas",       "v_ref"),
    ("Pit30M",    "pit30m_10hz",  "v_ref"),
]
BOOT = 5000


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def per_sequence(subdir: str, ref_col: str) -> pd.DataFrame:
    rows = []
    for p in sorted((REPO_ROOT / "results" / subdir).glob("per_frame_*.parquet")):
        d = pd.read_parquet(p)
        name = p.stem[len("per_frame_"):]
        # Pit30M segments are named <logid>_<segment>; everything else is one unit.
        m = re.match(r"^([0-9a-f]{8})_\d+$", name)
        rows.append({
            "sequence": name,
            "group": m.group(1) if m else name,
            "n": len(d),
            "central": rmse(d["v_central"].to_numpy(), d[ref_col].to_numpy()),
            "fa_W5": rmse(d["v_family_a_W5"].to_numpy(), d[ref_col].to_numpy()),
        })
    return pd.DataFrame(rows)


def boot_interval(c: np.ndarray, f: np.ndarray, groups: np.ndarray | None = None,
                  seed: int = 0) -> tuple[float, float]:
    """Percentile interval of the ratio of medians.

    groups=None resamples sequences (the published unit).  Otherwise whole groups
    are drawn with replacement and every sequence of a drawn group is kept, which
    is the cluster bootstrap the reviewer asks about.
    """
    rng = np.random.default_rng(seed)
    pct = []
    if groups is None:
        n = len(c)
        for _ in range(BOOT):
            idx = rng.integers(0, n, n)
            mc = np.median(c[idx])
            if mc > 0:
                pct.append((np.median(f[idx]) - mc) / mc * 100.0)
    else:
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        for _ in range(BOOT):
            drawn = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([members[uniq[d]] for d in drawn])
            mc = np.median(c[idx])
            if mc > 0:
                pct.append((np.median(f[idx]) - mc) / mc * 100.0)
    return float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args()

    summary, per_seq_all = [], []
    for name, subdir, ref_col in DATASETS:
        df = per_sequence(subdir, ref_col)
        if df.empty:
            print(f"  {name}: no per-frame streams under results/{subdir}")
            continue
        df.insert(0, "dataset", name)
        per_seq_all.append(df)

        c, f = df["central"].to_numpy(), df["fa_W5"].to_numpy()
        med_c, med_f = float(np.median(c)), float(np.median(f))
        delta = (med_f / med_c - 1) * 100.0
        lo, hi = boot_interval(c, f)
        n_groups = df["group"].nunique()
        row = {
            "dataset": name, "n_sequences": len(df), "n_groups": n_groups,
            "central": med_c, "fa_W5": med_f, "delta_pct": delta,
            "seq_boot_lo": lo, "seq_boot_hi": hi,
        }
        if n_groups < len(df):
            glo, ghi = boot_interval(c, f, df["group"].to_numpy())
            row["cluster_boot_lo"], row["cluster_boot_hi"] = glo, ghi
            row["cluster_width_ratio"] = (ghi - glo) / (hi - lo)
        summary.append(row)

        msg = (f"  {name:<10} n={len(df):3d} (logs {n_groups:2d})  "
               f"Delta {delta:+7.2f}%  sequence boot [{lo:+7.2f},{hi:+7.2f}]")
        if n_groups < len(df):
            msg += f"  cluster boot [{row['cluster_boot_lo']:+7.2f},{row['cluster_boot_hi']:+7.2f}]"
        print(msg)

    print("\nreleases with three or four sequences, individual values [m/s]:")
    for df in per_seq_all:
        if len(df) <= 4:
            for _, r in df.iterrows():
                print(f"  {r['dataset']:<10} {r['sequence']:<32} n={r['n']:6d} "
                      f"central {r['central']:.4f}  FamilyA {r['fa_W5']:.4f}  "
                      f"Delta {(r['fa_W5']/r['central']-1)*100:+7.2f}%")

    pd.DataFrame(summary).to_csv(args.out / "bootstrap_unit_control.csv", index=False)
    pd.concat(per_seq_all).to_csv(args.out / "per_sequence_m4.csv", index=False)
    print(f"\nwrote {args.out/'bootstrap_unit_control.csv'} and {args.out/'per_sequence_m4.csv'}")


if __name__ == "__main__":
    main()
