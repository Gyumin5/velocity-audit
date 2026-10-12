"""Publish the interior-slice control of Table 1 as a CSV.

The cross-release tables are computed on the full stored series. This records
what the alternative convention---dropping the first and last max(10, n/50)
samples, so a fixed window is never evaluated where it is truncated---would
change, per release.
"""
import pandas as pd, numpy as np, pathlib

R = pathlib.Path("results")
NAME = {"helipr": "HeLiPR", "oxford_x11": "Oxford", "nuscenes_x20": "nuScenes",
        "kitti": "KITTI raw", "kitti360": "KITTI-360", "boreas": "Boreas",
        "pit30m_10hz": "Pit30M"}
sets = [("helipr", "v_ins"), ("oxford_x11", "v_ins"), ("nuscenes_x20", "v_can"),
        ("kitti", "v_oxts"), ("kitti360", "v_ref"), ("boreas", "v_ref"),
        ("pit30m_10hz", "v_ref")]


def stats(sub, ref, interior):
    c, f = [], []
    for path in sorted((R / sub).glob("per_frame_*.parquet")):
        d = pd.read_parquet(path)
        m = max(10, len(d) // 50)
        sl = slice(m, -m) if interior else slice(None)
        r = d[ref].to_numpy()[sl]
        for col, acc in (("v_central", c), ("v_family_a_W5", f)):
            a = d[col].to_numpy()[sl]
            ok = np.isfinite(a) & np.isfinite(r)
            acc.append(float(np.sqrt(np.mean((a[ok] - r[ok]) ** 2))))
    mc, mf = float(np.median(c)), float(np.median(f))
    return mc, mf, (mf / mc - 1) * 100


rows = []
for sub, ref in sets:
    fc, ff, fd = stats(sub, ref, False)
    ic, i_f, idd = stats(sub, ref, True)
    rows.append(dict(dataset=NAME[sub], central_full=fc, famA_W5_full=ff,
                     delta_pct_full=fd, central_interior=ic,
                     famA_W5_interior=i_f, delta_pct_interior=idd,
                     delta_move_pp=abs(fd - idd)))
out = pd.DataFrame(rows)
out.to_csv(R / "interior_slice_control.csv", index=False)
print(out.to_string(index=False))
