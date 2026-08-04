"""Family A: local polynomial LS fit + arc-length speed.

Two variants:
  A1 (pointwise) : v_k = ||p'(t_k)|| from a local degree-p polynomial LS fit on
                   [k-W .. k+W]. Robust to noise, unlike an interpolating spline.
  A2 (midspan)   : v_k = arc-length over [t_{k-W}..t_{k+W}] / span, using the
                   same local LS polynomial, integrated with Gauss-Legendre-7.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import fixed_quad

from velref.core.trajectory import Pose2D


def _local_ls_poly_deriv_at(
    t: np.ndarray,
    y: np.ndarray,
    k: int,
    W: int,
    degree: int,
) -> tuple[float, np.ndarray, float, float]:
    """Fit degree-p polynomial (centered at t_k) to (t_i, y_i) by LS on the window.

    Returns (y'(t_k), coeffs_in_centered_basis, t_lo, t_hi) where coeffs are in the
    centered-and-scaled monomial basis (see below). The derivative at the window
    centre is coeff[1] / span_scale (since u = (t - t_k) / span_scale).
    """
    n = t.size
    lo = max(0, k - W)
    hi = min(n, k + W + 1)
    width = t[hi - 1] - t[lo]
    if width <= 0:
        return 0.0, np.zeros(degree + 1), float(t[lo]), float(t[hi - 1])
    # Center + scale for numerical conditioning.
    u = (t[lo:hi] - t[k]) / width
    y_win = y[lo:hi]
    # Fit y = sum_j c_j u^j  via np.polyfit (which expects descending order).
    eff_deg = min(degree, hi - lo - 1)
    coeffs_desc = np.polyfit(u, y_win, eff_deg)
    # Reorder to ascending: c[0] + c[1] u + ... (so c[1] is the derivative wrt u at u=0).
    c = coeffs_desc[::-1]
    dy_du = float(c[1]) if c.size > 1 else 0.0
    dy_dt = dy_du / width
    return dy_dt, c, float(t[lo]), float(t[hi - 1])


def _poly_value_from_coeffs(c: np.ndarray, u):
    # Horner in ascending order.
    acc = np.zeros_like(np.asarray(u, dtype=float))
    for ci in c[::-1]:
        acc = acc * u + ci
    return acc


def family_a_pointwise(p: Pose2D, W: int = 5, degree: int = 3) -> np.ndarray:
    """Option A1: local LS polynomial derivative at t_k."""
    n = len(p)
    out = np.empty(n)
    for k in range(n):
        dxdt, _, _, _ = _local_ls_poly_deriv_at(p.t, p.x, k, W, degree)
        dydt, _, _, _ = _local_ls_poly_deriv_at(p.t, p.y, k, W, degree)
        out[k] = float(np.hypot(dxdt, dydt))
    return out


def family_a_midspan(p: Pose2D, W: int = 5, degree: int = 3) -> np.ndarray:
    """Option A2: arc-length over window / span, via Gauss-Legendre on the LS polynomial."""
    n = len(p)
    out = np.empty(n)
    for k in range(n):
        _, cx, t_lo_x, t_hi_x = _local_ls_poly_deriv_at(p.t, p.x, k, W, degree)
        _, cy, t_lo_y, t_hi_y = _local_ls_poly_deriv_at(p.t, p.y, k, W, degree)
        # Use the common window endpoints.
        t_lo = max(t_lo_x, t_lo_y)
        t_hi = min(t_hi_x, t_hi_y)
        width_x = t_hi_x - t_lo_x if t_hi_x > t_lo_x else 1.0
        width_y = t_hi_y - t_lo_y if t_hi_y > t_lo_y else 1.0
        span = t_hi - t_lo
        if span <= 0:
            # Fallback: pointwise.
            out[k] = family_a_pointwise(p, W, degree)[k]
            continue

        def speed_fn(ts):
            ts = np.asarray(ts, dtype=float)
            ux = (ts - p.t[k]) / width_x
            uy = (ts - p.t[k]) / width_y
            # dx/dt = (dc/du * du/dt); for poly in u, derivative polynomial in u:
            dcx = np.array([j * cx[j] for j in range(1, cx.size)])
            dcy = np.array([j * cy[j] for j in range(1, cy.size)])
            vx = _poly_value_from_coeffs(dcx, ux) / width_x if dcx.size else np.zeros_like(ts)
            vy = _poly_value_from_coeffs(dcy, uy) / width_y if dcy.size else np.zeros_like(ts)
            return np.hypot(vx, vy)

        # 7-node Gauss-Legendre via scipy.
        arc, _ = fixed_quad(speed_fn, t_lo, t_hi, n=7)
        out[k] = float(arc / span)
    return out
