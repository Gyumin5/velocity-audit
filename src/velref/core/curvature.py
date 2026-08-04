"""Common curvature estimator from a 2D pose sequence."""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

from velref.core.trajectory import Pose2D


def estimate_curvature(p: Pose2D, window: int = 9, polyorder: int = 3) -> np.ndarray:
    """Signed 2D curvature via smoothed finite differences.

    kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

    Uses Savitzky–Golay derivatives for stability on discrete pose samples.
    Assumes approximately uniform sampling (median dt).
    """
    n = int(p.t.size)
    if n < window:
        return np.zeros(n)
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 2)
    window = min(window, (n // 2) * 2 + 1)
    dt = float(np.median(np.diff(p.t)))
    x = p.x
    y = p.y
    xp = savgol_filter(x, window, polyorder, deriv=1, delta=dt)
    yp = savgol_filter(y, window, polyorder, deriv=1, delta=dt)
    xpp = savgol_filter(x, window, polyorder, deriv=2, delta=dt)
    ypp = savgol_filter(y, window, polyorder, deriv=2, delta=dt)
    num = xp * ypp - yp * xpp
    den = (xp * xp + yp * yp) ** 1.5 + 1e-9
    return num / den
