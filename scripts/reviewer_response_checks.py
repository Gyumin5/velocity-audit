#!/usr/bin/env python3
"""Checks requested by the IEEE Access first-round reviewers (Access-2026-39049).

Three questions, all answered from the per-frame streams already under results/:

A) Duration-matched Savitzky-Golay baseline on the field data, on all seven
   releases rather than the five-release subset the manuscript reports.
   The manuscript reports SG w7p3, which spans 0.6 s at the 10 Hz analysis
   cadence, against a probe that spans 1.0 s.  Reviewer 1 asks for SG w11p3,
   the window that matches the probe's duration.  The same pass runs the probe
   a second time on a synthetic uniform time vector (t_k = t_0 + k*median dt,
   real positions), which isolates how much of any probe-vs-SG gap is the
   timestamp handling and how much is anything else.

B) The latency sweep read from central differencing alone.
   Reviewer 1 asks whether the collapse that identifies a shared time base
   needs the characterized probe at all.  The existing sweep already carries
   the central-only column, so this recomputes it as a ratio against the
   unshifted value and records the shape per release.

C) Whether the documentation-assigned coupling label orders Delta better than
   release-level covariates that are already in the multi-metric table.

Writes results/reviewer_r1_checks/.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from velref.core.trajectory import Pose2D
from velref.methods.baselines import savgol_deriv, smoothing_spline_deriv
from velref.methods.family_a import family_a_pointwise

# display name -> (results subdirectory, published-velocity column, coupling label)
DATASETS = [
    ("HeLiPR",    "helipr",       "v_ins",  "separated"),
    ("Oxford",    "oxford_x11",   "v_ins",  "alg. separated"),
    ("nuScenes",  "nuscenes_x20", "v_can",  "weak/online"),
    ("KITTI",     "kitti",        "v_oxts", "weak/online"),
    ("KITTI-360", "kitti360",     "v_ref",  "weak/online"),
    ("Boreas",    "boreas",       "v_ref",  "batch joint"),
    ("Pit30M",    "pit30m_10hz",  "v_ref",  "batch joint"),
]

# The paper's interior rule: drop the first and last max(10, n/50) samples on
# five releases, use the full series on KITTI-360 and Pit30M.
# build_crossds_tables.py aggregates over the whole stream: the per-frame files are
# already written with the interior rule applied, so no further trimming here.
FULL_SERIES = {"helipr", "oxford_x11", "nuscenes_x20", "kitti", "kitti360", "boreas", "pit30m_10hz"}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.sum() >= 5 else float("nan")


def interior(n: int, subdir: str) -> slice:
    if subdir in FULL_SERIES:
        return slice(None)
    m = max(10, n // 50)
    return slice(m, -m)


def check_a() -> pd.DataFrame:
    rows = []
    for name, subdir, ref_col, label in DATASETS:
        keys = ("central", "sg_w7p3", "sg_w11p3", "spline", "fa_W5", "fa_W5_uniform_t")
        per_seq = {k: [] for k in keys}
        dt_cv = []
        files = sorted((REPO / "results" / subdir).glob("per_frame_*.parquet"))
        for p in files:
            d = pd.read_parquet(p)
            t, x, y = d["t"].to_numpy(), d["x"].to_numpy(), d["y"].to_numpy()
            ref = d[ref_col].to_numpy()
            sl = interior(len(ref), subdir)
            pose = Pose2D(t, x, y)
            dt = np.diff(t)
            dt_cv.append(float(np.std(dt) / np.mean(dt)))
            # Same positions, timestamps replaced by a perfectly regular grid at
            # the sequence's median interval.  Any difference from the real-clock
            # run is attributable to timestamp handling alone.
            pose_uniform = Pose2D(t[0] + np.arange(t.size) * float(np.median(dt)), x, y)
            est = {
                "central": d["v_central"].to_numpy(),
                "sg_w7p3": savgol_deriv(pose, 7, 3),
                "sg_w11p3": savgol_deriv(pose, 11, 3),
                "spline": smoothing_spline_deriv(pose),
                "fa_W5": d["v_family_a_W5"].to_numpy(),
                "fa_W5_uniform_t": family_a_pointwise(pose_uniform, W=5),
            }
            for k, v in est.items():
                per_seq[k].append(rmse(v[sl], ref[sl]))
        med = {k: float(np.median(v)) for k, v in per_seq.items()}
        rows.append({
            "dataset": name, "N": len(files), "coupling": label,
            "dt_cv_median": float(np.median(dt_cv)),
            "M4_central": med["central"], "M4_sg_w7p3": med["sg_w7p3"],
            "M4_sg_w11p3": med["sg_w11p3"], "M4_spline": med["spline"],
            "M4_fa_W5": med["fa_W5"], "M4_fa_W5_uniform_t": med["fa_W5_uniform_t"],
            "delta_fa_pct": (med["fa_W5"] / med["central"] - 1) * 100,
            "delta_sg11_pct": (med["sg_w11p3"] / med["central"] - 1) * 100,
            "delta_sg7_pct": (med["sg_w7p3"] / med["central"] - 1) * 100,
        })
        r = rows[-1]
        print(f"{name:<10} N={r['N']:3d} dtCV {r['dt_cv_median']:.4f}  "
              f"central {r['M4_central']:.4f}  SG7 {r['M4_sg_w7p3']:.4f}  "
              f"SG11 {r['M4_sg_w11p3']:.4f}  spline {r['M4_spline']:.4f}  "
              f"FA5 {r['M4_fa_W5']:.4f}  FA5/uniform-t {r['M4_fa_W5_uniform_t']:.4f}")
    df = pd.DataFrame(rows)
    gap = (df.M4_fa_W5_uniform_t - df.M4_sg_w11p3).abs().max()
    print(f"\n  max |Family A on a uniform clock  -  SG w11p3| across releases: {gap:.5f} m/s")
    return df


def check_b() -> pd.DataFrame:
    lat = pd.read_csv(REPO / "results" / "latency_sweep.csv")
    out = []
    for ds, g in lat.groupby("dataset", sort=False):
        base = float(g.loc[g.shift_s == 0.0, "median_M4_central"].iloc[0])
        for _, r in g.iterrows():
            out.append({
                "dataset": ds, "shift_s": r.shift_s,
                "M4_central": r.median_M4_central,
                "central_ratio_vs_zero": r.median_M4_central / base,
                "M4_FA": r.median_M4_FA,
                "delta_pct": r.delta_pct,
            })
    df = pd.DataFrame(out)
    print("\nCentral-only latency shape (M4_central at shift / M4_central at 0):")
    piv = df.pivot(index="dataset", columns="shift_s", values="central_ratio_vs_zero")
    print(piv.round(2).to_string())
    return df


def check_c(a: pd.DataFrame) -> pd.DataFrame:
    multi = pd.read_csv(REPO / "results" / "multimetric_recomputed.csv")
    m = a.merge(multi[["dataset", "M3_c", "M4_c"]], on="dataset")
    order = {"separated": 0, "alg. separated": 0, "weak/online": 1, "batch joint": 2}
    m["coupling_rank"] = m["coupling"].map(order)
    m = m.reset_index(drop=True)
    # Reviewer 1 notes that KITTI raw and KITTI-360 share the OXTS platform and
    # export path, so the online band is one sensing chain seen twice. Each
    # subset below drops one or both of them; if the ordering only held because
    # that chain was counted twice, it would not survive here.
    subsets = {
        "all seven": m,
        "drop KITTI raw": m[m.dataset != "KITTI"],
        "drop KITTI-360": m[m.dataset != "KITTI-360"],
        "drop both KITTI": m[~m.dataset.isin(["KITTI", "KITTI-360"])],
    }
    rows = []
    for subset, mm in subsets.items():
        line = []
        for col in ("coupling_rank", "M4_c", "M3_c"):
            rho = mm[[col, "delta_fa_pct"]].corr(method="spearman").iloc[0, 1]
            rows.append({"subset": subset, "n": len(mm), "predictor": col,
                         "spearman_rho_with_delta": rho})
            line.append(f"{col} {rho:+.3f}")
        print(f"  {subset:<16} n={len(mm)}  " + "   ".join(line))
    print("\n  Near-matched smoothness pairs (same central M3, different delta):")
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if abs(m.M3_c[i] - m.M3_c[j]) / max(m.M3_c[i], m.M3_c[j]) < 0.10:
                print(f"    {m.dataset[i]:<10} M3_c={m.M3_c[i]:.3f} delta={m.delta_fa_pct[i]:+6.1f}%"
                      f"  vs  {m.dataset[j]:<10} M3_c={m.M3_c[j]:.3f} delta={m.delta_fa_pct[j]:+6.1f}%")
    return pd.DataFrame(rows)


TAU = 0.15
SHIFTS = (-0.5, -0.2, -0.1, 0.1, 0.2, 0.5)
WINDOWS = {3: 0.6, 5: 1.0, 7: 1.4}  # W -> span in seconds at the 10 Hz analysis cadence


def algorithm_output(coupling: str, delta_frac: float, sign_survives: bool) -> str:
    """Algorithm 1's returned interpretation, evaluated as written."""
    if coupling == "batch joint":
        return "no independent comparison; report delta and its latency fragility"
    if coupling == "weak/online":
        return ("consistent with shared filter state" if abs(delta_frac) < TAU
                else "residual response departs from the setting's expectation")
    if coupling in ("separated", "alg. separated"):
        if delta_frac < 0 and sign_survives:
            return "response not attributable to a shared clock alone"
        return "inconclusive"
    return "unclassified"


def check_d() -> pd.DataFrame:
    """Window-duration sensitivity on all seven releases at the same 10 Hz cadence.

    The manuscript reports this sweep on a five-release subset and in prose. Reviewer 1
    asks what the free settings buy, so every release is swept here and each row carries
    what the audit would actually do at that setting: the residual change, whether it
    clears the pre-set reporting margin, what Algorithm 1 returns, and how far the
    reading moved from the adopted 1.0 s setting.
    """
    rows = []
    for name, subdir, ref_col, label in DATASETS:
        files = sorted((REPO / "results" / subdir).glob("per_frame_*.parquet"))
        per = {W: {"c": [], "f": [], "shift": {s: ([], []) for s in SHIFTS}} for W in WINDOWS}
        for p in files:
            d = pd.read_parquet(p)
            t = d["t"].to_numpy()
            ref = d[ref_col].to_numpy()
            cen = d["v_central"].to_numpy()
            for W in WINDOWS:
                fa = d[f"v_family_a_W{W}"].to_numpy()
                per[W]["c"].append(rmse(cen, ref))
                per[W]["f"].append(rmse(fa, ref))
                for s in SHIFTS:
                    rs = np.interp(t, t + s, ref, left=np.nan, right=np.nan)
                    per[W]["shift"][s][0].append(rmse(cen, rs))
                    per[W]["shift"][s][1].append(rmse(fa, rs))
        base = None
        for W, span in WINDOWS.items():
            mc = float(np.median(per[W]["c"]))
            mf = float(np.median(per[W]["f"]))
            delta = mf / mc - 1
            shifted = {s: float(np.median(per[W]["shift"][s][1]))
                       / float(np.median(per[W]["shift"][s][0])) - 1 for s in SHIFTS}
            survives = all(np.sign(v) == np.sign(delta) for v in shifted.values())
            if W == 5:
                base = delta
            rows.append({
                "dataset": name, "coupling": label, "W": W, "span_s": span,
                "M4_central": mc, "M4_fa": mf, "delta_pct": delta * 100,
                "clears_margin": abs(delta) >= TAU,
                "latency_sign_survives": survives,
                "algorithm_output": algorithm_output(label, delta, survives),
                **{f"delta_shift_{s:+.1f}s_pct": shifted[s] * 100 for s in SHIFTS},
            })
        for r in rows[-len(WINDOWS):]:
            r["delta_minus_default_pp"] = r["delta_pct"] - base * 100
    df = pd.DataFrame(rows)
    piv = df.pivot(index="dataset", columns="span_s", values="delta_pct")
    print("\n  Delta (%) vs central, by window span at 10 Hz:")
    print(piv.round(1).to_string())
    print("\n  clears the tau=0.15 reporting margin:")
    print(df.pivot(index="dataset", columns="span_s", values="clears_margin").to_string())
    flips = df.groupby("dataset").algorithm_output.nunique()
    changed = list(flips[flips > 1].index)
    print(f"\n  releases whose Algorithm 1 output changes across the three spans: "
          f"{changed if changed else 'none'}")
    return df


def main() -> None:
    out_dir = REPO / "results" / "reviewer_r1_checks"
    out_dir.mkdir(exist_ok=True)
    print("=== A: duration-matched SG baseline on all seven releases ===")
    a = check_a()
    a.to_csv(out_dir / "sg_duration_matched.csv", index=False)
    print("\n=== B: latency sweep read from central differencing alone ===")
    b = check_b()
    b.to_csv(out_dir / "latency_central_only.csv", index=False)
    print("\n=== C: does the coupling label order delta better than covariates? ===")
    c = check_c(a)
    c.to_csv(out_dir / "label_vs_covariates.csv", index=False)
    print("\n=== D: window-duration sensitivity on all seven releases ===")
    d = check_d()
    d.to_csv(out_dir / "window_sensitivity.csv", index=False)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
