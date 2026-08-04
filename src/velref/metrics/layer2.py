"""Layer-2 reference-free metrics on real sequences."""
from __future__ import annotations
import numpy as np
from velref.core.trajectory import Pose2D


def m1_path_length_ratio(p: Pose2D, v_hat: np.ndarray) -> float:
    """|∫ v̂ dt / polyline_length − 1|. Smaller is better."""
    dt = np.diff(p.t)
    v_int = float(np.sum(0.5 * (v_hat[1:] + v_hat[:-1]) * dt))
    poly_len = float(np.sum(np.hypot(np.diff(p.x), np.diff(p.y))))
    if poly_len < 1e-6:
        return float("nan")
    return float(abs(v_int / poly_len - 1.0))


def m3_low_speed_rms(v_hat: np.ndarray, v_ref: np.ndarray, thr: float = 0.3) -> float:
    """RMS of v_hat on low-speed mask (|v_ref|<thr). Lower is better (reduced oscillation)."""
    mask = np.abs(v_ref) < thr
    if mask.sum() < 3:
        return float("nan")
    return float(np.sqrt(np.mean(v_hat[mask] ** 2)))


def m4_smooth_lag(v_hat: np.ndarray, v_ref: np.ndarray, dt: float) -> tuple[float, float]:
    """Return (smoothness_rms_second_diff, lag_seconds).

    smoothness = RMS of second difference of v_hat (in units of m/s per Δt^2 ~ jerk-ish).
    lag = argmax of cross-correlation between v_hat and v_ref, in seconds.
    """
    if v_hat.size < 3:
        return float("nan"), float("nan")
    d2 = np.diff(np.diff(v_hat))
    smooth = float(np.sqrt(np.mean(d2 ** 2)))
    n = v_hat.size
    # Lag via brute correlation over +-30 samples.
    lags = np.arange(-30, 31)
    xs = v_hat - v_hat.mean()
    ys = v_ref - v_ref.mean()
    denom = np.sqrt(np.sum(xs * xs) * np.sum(ys * ys)) + 1e-12
    best = 0
    best_score = -np.inf
    for L in lags:
        if L >= 0:
            a = xs[: n - L]
            b = ys[L:]
        else:
            a = xs[-L:]
            b = ys[: n + L]
        if a.size < 10:
            continue
        s = float(np.sum(a * b) / denom)
        if s > best_score:
            best_score = s
            best = L
    return smooth, float(best * dt)
