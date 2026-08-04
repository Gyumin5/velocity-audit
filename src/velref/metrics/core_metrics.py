"""Metrics M1 (integrated displacement), M3 (stop-go index), plus RMSE for synthetic."""
from __future__ import annotations
import numpy as np
from velref.core.trajectory import Pose2D


def rmse(v_hat: np.ndarray, v_ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((v_hat - v_ref) ** 2)))


def bias(v_hat: np.ndarray, v_ref: np.ndarray) -> float:
    return float(np.mean(v_hat - v_ref))


def m1_integrated_displacement(p: Pose2D, v_hat: np.ndarray) -> float:
    """|∫ v_hat dt − ||p(T) − p(0)|| | / ||p(T) − p(0)|| (safe on short straight seg)."""
    dt = np.diff(p.t)
    # Trapezoid.
    v_int = float(np.sum(0.5 * (v_hat[1:] + v_hat[:-1]) * dt))
    endpoint_dist = float(np.hypot(p.x[-1] - p.x[0], p.y[-1] - p.y[0]))
    # For straight paths endpoint_dist == true path length; for curved it is a lower bound.
    # Report signed log-ratio vs endpoint_dist as a proxy. NOT suitable for loops.
    if endpoint_dist < 1e-6:
        return float("nan")
    return float(abs(v_int - endpoint_dist) / endpoint_dist)


def m1_path_length(p: Pose2D, v_hat: np.ndarray) -> float:
    """Ratio ∫ v_hat dt to the polyline path length of the pose sequence.

    Returns |ratio − 1|; closer to 0 is better.
    """
    dt = np.diff(p.t)
    v_int = float(np.sum(0.5 * (v_hat[1:] + v_hat[:-1]) * dt))
    poly_len = float(np.sum(np.hypot(np.diff(p.x), np.diff(p.y))))
    if poly_len < 1e-6:
        return float("nan")
    return float(abs(v_int / poly_len - 1.0))


def m3_stop_go_oscillation(v_hat: np.ndarray, v_ref: np.ndarray | None, thr: float = 0.3) -> float:
    """RMS of v_hat on the low-speed mask (|v_ref|<thr if given, else |v_hat|<thr)."""
    ref = v_ref if v_ref is not None else v_hat
    mask = np.abs(ref) < thr
    if mask.sum() < 2:
        return float("nan")
    return float(np.sqrt(np.mean(v_hat[mask] ** 2)))
