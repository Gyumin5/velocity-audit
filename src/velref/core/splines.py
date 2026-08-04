"""Local cubic spline fit + arc-length integration."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


# 7-node Gauss-Legendre on [-1, 1].
_GL7_NODES = np.array([
    -0.9491079123427585,
    -0.7415311855993945,
    -0.4058451513773972,
     0.0,
     0.4058451513773972,
     0.7415311855993945,
     0.9491079123427585,
])
_GL7_WEIGHTS = np.array([
    0.1294849661688697,
    0.2797053914892767,
    0.3818300505051189,
    0.4179591836811853,
    0.3818300505051189,
    0.2797053914892767,
    0.1294849661688697,
])


def _gauss_legendre_7(f, a: float, b: float) -> float:
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    xs = mid + half * _GL7_NODES
    return float(half * np.sum(_GL7_WEIGHTS * f(xs)))


def fit_local_cubic(t: np.ndarray, x: np.ndarray) -> CubicSpline:
    """Natural cubic spline through (t_i, x_i). Assumes t strictly increasing."""
    return CubicSpline(t, x, bc_type="natural")


def local_speed_arclength(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    W: int,
) -> float:
    """Arc-length speed at index k using a window of half-size W.

    Fits cubic splines x(t), y(t) on [k-W .. k+W]; returns sqrt(x'(t_k)^2 + y'(t_k)^2).

    For small noise this is equivalent to option A1 in METHOD_CANDIDATES.
    """
    n = t.size
    lo = max(0, k - W)
    hi = min(n, k + W + 1)
    if hi - lo < 4:
        # Fallback: central/forward difference.
        if 0 < k < n - 1:
            dt = t[k + 1] - t[k - 1]
            dx = x[k + 1] - x[k - 1]
            dy = y[k + 1] - y[k - 1]
        elif k == 0:
            dt = t[1] - t[0]
            dx = x[1] - x[0]
            dy = y[1] - y[0]
        else:
            dt = t[k] - t[k - 1]
            dx = x[k] - x[k - 1]
            dy = y[k] - y[k - 1]
        return float(np.hypot(dx, dy) / dt)
    cs_x = fit_local_cubic(t[lo:hi], x[lo:hi])
    cs_y = fit_local_cubic(t[lo:hi], y[lo:hi])
    vx = float(cs_x(t[k], 1))
    vy = float(cs_y(t[k], 1))
    return float(np.hypot(vx, vy))


def local_speed_arclength_midspan(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    W: int,
) -> float:
    """Option A2: v_k = arc-length over [t_{k-m}..t_{k+m}] divided by span, m = W.

    Uses Gauss–Legendre-7 quadrature of ||p'(t)|| over the local spline.
    """
    n = t.size
    lo = max(0, k - W)
    hi = min(n, k + W + 1)
    if hi - lo < 4 or hi <= lo + 1:
        return local_speed_arclength(t, x, y, k, W)
    cs_x = fit_local_cubic(t[lo:hi], x[lo:hi])
    cs_y = fit_local_cubic(t[lo:hi], y[lo:hi])
    a = float(t[max(lo, k - W)])
    b = float(t[min(hi - 1, k + W)])
    if b <= a:
        return local_speed_arclength(t, x, y, k, W)
    def speed_fn(ts):
        vx = cs_x(ts, 1)
        vy = cs_y(ts, 1)
        return np.hypot(vx, vy)
    arc = _gauss_legendre_7(speed_fn, a, b)
    return float(arc / (b - a))
