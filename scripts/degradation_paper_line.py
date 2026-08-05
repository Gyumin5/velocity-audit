#!/usr/bin/env python3
"""Emit the manuscript's degradation list straight from the committed CSV.

The paragraph in Sec. "Semi-controlled degradation" quotes seven four-number
sequences. Typing those by hand is how a figure in this project drifted out of
step with its own table, so they are generated here instead and pasted as a
block. Also reports which datasets clear the 30% expectation at each factor, so
the sentence that follows the list states what the data says rather than what it
said last time.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDER = ["HeLiPR", "Oxford", "nuScenes", "KITTI raw", "KITTI-360", "Boreas", "Pit30M"]
THRESHOLD = 30.0


def tex(v: float) -> str:
    """LaTeX-safe integer with a math-mode minus."""
    n = round(v)
    return f"$-{abs(n):.0f}$" if n < 0 else f"{n:.0f}"


def main() -> None:
    d = pd.read_csv(REPO_ROOT / "results" / "degradation_stress.csv")
    piv = d.pivot_table(index="dataset", columns="ds", values="reduction_pct")
    print("--- manuscript list ---")
    print("; ".join(f"{n} " + "/".join(tex(piv.loc[n, ds]) for ds in (1, 2, 5, 10))
                    for n in ORDER if n in piv.index))
    print("\n--- clears the 30% expectation ---")
    for ds in (1, 2, 5, 10):
        ok = [n for n in ORDER if n in piv.index and piv.loc[n, ds] >= THRESHOLD]
        miss = [(n, piv.loc[n, ds]) for n in ORDER
                if n in piv.index and piv.loc[n, ds] < THRESHOLD]
        note = "  below: " + ", ".join(f"{n} {v:.0f}%" for n, v in miss) if miss else ""
        print(f"  ds={ds:<3} {len(ok)}/7{note}")

    if "reduction_p5" in d.columns:
        print("\n--- realization spread (p5-p95) at ds=5 ---")
        for r in d[d.ds == 5].itertuples():
            print(f"  {r.dataset:<10} {r.reduction_pct:5.0f}%  "
                  f"[{r.reduction_p5:.0f}, {r.reduction_p95:.0f}]")


if __name__ == "__main__":
    main()
