"""Baselines for speed estimation from pose sequences."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from velref.core.trajectory import Pose2D


def forward_diff(p: Pose2D) -> np.ndarray:
    dt = np.diff(p.t)
    dx = np.diff(p.x)
    dy = np.diff(p.y)
    v = np.hypot(dx, dy) / dt
    return np.concatenate([v, v[-1:]])


def central_diff(p: Pose2D) -> np.ndarray:
    n = len(p)
    v = np.empty(n)
    v[0] = np.hypot(p.x[1] - p.x[0], p.y[1] - p.y[0]) / (p.t[1] - p.t[0])
    v[-1] = np.hypot(p.x[-1] - p.x[-2], p.y[-1] - p.y[-2]) / (p.t[-1] - p.t[-2])
    dt = p.t[2:] - p.t[:-2]
    dx = p.x[2:] - p.x[:-2]
    dy = p.y[2:] - p.y[:-2]
    v[1:-1] = np.hypot(dx, dy) / dt
    return v


def cubic_spline_deriv(p: Pose2D) -> np.ndarray:
    cs_x = CubicSpline(p.t, p.x, bc_type="natural")
    cs_y = CubicSpline(p.t, p.y, bc_type="natural")
    vx = cs_x(p.t, 1)
    vy = cs_y(p.t, 1)
    return np.hypot(vx, vy)


def smoothing_spline_deriv(p: Pose2D, s: float | None = None) -> np.ndarray:
    """Reinsch smoothing spline with scipy defaults (GCV-like parameter if s=None).

    scipy.interpolate.UnivariateSpline requires strictly increasing x.
    s is the smoothing factor; None selects by len(x) default.
    """
    s_val = s if s is not None else float(p.t.size)
    us_x = UnivariateSpline(p.t, p.x, k=5, s=s_val)
    us_y = UnivariateSpline(p.t, p.y, k=5, s=s_val)
    return np.hypot(us_x.derivative()(p.t), us_y.derivative()(p.t))


def _estimate_position_noise(signal: np.ndarray) -> float:
    """Robust estimate of per-sample position noise from the second difference MAD.

    Under white-noise position error with variance sigma^2, the second difference
    has variance 6*sigma^2, so sigma ≈ MAD(d2)/sqrt(6)/0.6745.
    """
    d2 = np.diff(np.diff(signal))
    mad = np.median(np.abs(d2 - np.median(d2)))
    if mad <= 0:
        return 0.0
    return float(mad / 0.6745 / np.sqrt(6.0))


def smoothing_spline_tuned_deriv(p: Pose2D) -> np.ndarray:
    """Smoothing spline with sigma-aware s = n * sigma_p^2 (noise-scaled Reinsch).

    Addresses the RA-L reviewer concern that the default UnivariateSpline with s=n
    is an unfair strawman: here s is set from a robust position-noise estimate.
    """
    sigma_x = _estimate_position_noise(p.x)
    sigma_y = _estimate_position_noise(p.y)
    sigma_p = max(sigma_x, sigma_y, 1e-3)
    n = int(p.t.size)
    s_val = max(1e-3, n * sigma_p * sigma_p)
    # Order 5 degrades numerically for very tight s; fall back to order 3.
    for k in (5, 3):
        try:
            us_x = UnivariateSpline(p.t, p.x, k=k, s=s_val)
            us_y = UnivariateSpline(p.t, p.y, k=k, s=s_val)
            return np.hypot(us_x.derivative()(p.t), us_y.derivative()(p.t))
        except Exception:
            continue
    # Last-resort: return central-diff to avoid NaNs.
    return central_diff(p)


def savgol_deriv(p: Pose2D, window: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky–Golay derivative. Assumes near-uniform sampling.

    For strictly nonuniform timestamps, prefer smoothing_spline_deriv.
    """
    from scipy.signal import savgol_filter
    if window % 2 == 0:
        window += 1
    window = min(window, (p.t.size // 2) * 2 + 1)
    window = max(window, polyorder + 2)
    dt = float(np.median(np.diff(p.t)))
    vx = savgol_filter(p.x, window, polyorder, deriv=1, delta=dt)
    vy = savgol_filter(p.y, window, polyorder, deriv=1, delta=dt)
    return np.hypot(vx, vy)
