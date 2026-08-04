"""Synthetic 2D trajectories with closed-form arc length and speed."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from velref.core.trajectory import Pose2D


@dataclass
class SyntheticSample:
    pose: Pose2D
    v_true: np.ndarray        # scalar speed ground truth at each t
    kappa_true: np.ndarray    # curvature at each t (signed)
    name: str


def circle(
    duration: float = 20.0,
    fs: float = 10.0,
    radius: float = 15.0,
    v: float = 5.0,
    noise_sigma: float = 0.0,
    jitter_sigma: float = 0.0,
    seed: int = 0,
) -> SyntheticSample:
    rng = np.random.default_rng(seed)
    t_regular = np.arange(0, duration + 1.0 / fs, 1.0 / fs)
    if jitter_sigma > 0:
        t = t_regular + rng.normal(0.0, jitter_sigma, t_regular.size)
        t = np.clip(t, 0, duration + 1.0 / fs)
        t.sort()
    else:
        t = t_regular
    omega = v / radius
    theta = omega * t
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    if noise_sigma > 0:
        x = x + rng.normal(0.0, noise_sigma, x.size)
        y = y + rng.normal(0.0, noise_sigma, y.size)
    v_true = np.full_like(t, v)
    kappa = np.full_like(t, 1.0 / radius)
    return SyntheticSample(Pose2D(t, x, y), v_true, kappa, "circle")


def straight_variable_speed(
    duration: float = 20.0,
    fs: float = 10.0,
    v0: float = 0.5,
    v1: float = 10.0,
    noise_sigma: float = 0.0,
    jitter_sigma: float = 0.0,
    seed: int = 1,
) -> SyntheticSample:
    rng = np.random.default_rng(seed)
    t_reg = np.arange(0, duration + 1.0 / fs, 1.0 / fs)
    if jitter_sigma > 0:
        t = t_reg + rng.normal(0.0, jitter_sigma, t_reg.size)
        t = np.clip(t, 0, duration + 1.0 / fs)
        t.sort()
    else:
        t = t_reg
    # Linearly time-varying speed → position is quadratic in t.
    a = (v1 - v0) / duration
    # s(t) = v0*t + 0.5*a*t^2 ; x = s, y = 0
    s = v0 * t + 0.5 * a * t * t
    x = s
    y = np.zeros_like(t)
    if noise_sigma > 0:
        x = x + rng.normal(0.0, noise_sigma, x.size)
        y = y + rng.normal(0.0, noise_sigma, y.size)
    v_true = v0 + a * t
    kappa = np.zeros_like(t)
    return SyntheticSample(Pose2D(t, x, y), v_true, kappa, "straight_variable")


def stop_go(
    duration: float = 30.0,
    fs: float = 10.0,
    cruise: float = 5.0,
    stop_intervals=((8.0, 12.0), (18.0, 22.0)),
    noise_sigma: float = 0.0,
    jitter_sigma: float = 0.0,
    seed: int = 2,
) -> SyntheticSample:
    """Smooth stop-go: speed follows a product of logistic ramps."""
    rng = np.random.default_rng(seed)
    t_reg = np.arange(0, duration + 1.0 / fs, 1.0 / fs)
    if jitter_sigma > 0:
        t = t_reg + rng.normal(0.0, jitter_sigma, t_reg.size)
        t = np.clip(t, 0, duration + 1.0 / fs)
        t.sort()
    else:
        t = t_reg

    def gate(t, a, b, k=4.0):
        # 1 outside [a,b], smoothly 0 inside.
        in_left = 1.0 / (1.0 + np.exp(-k * (t - a)))
        in_right = 1.0 / (1.0 + np.exp(-k * (t - b)))
        return 1.0 - (in_left - in_right)

    g = np.ones_like(t)
    for a, b in stop_intervals:
        g = g * gate(t, a, b, k=4.0)
    v_true = cruise * g

    # Integrate v_true to get s(t) via trapezoidal rule on a dense grid for accuracy.
    dense_t = np.linspace(t[0], t[-1], 20 * t.size)
    g_dense = np.ones_like(dense_t)
    for a, b in stop_intervals:
        g_dense = g_dense * gate(dense_t, a, b, k=4.0)
    v_dense = cruise * g_dense
    s_dense = np.concatenate([[0.0], np.cumsum(0.5 * (v_dense[1:] + v_dense[:-1]) * np.diff(dense_t))])
    s = np.interp(t, dense_t, s_dense)
    x = s
    y = np.zeros_like(t)
    if noise_sigma > 0:
        x = x + rng.normal(0.0, noise_sigma, x.size)
        y = y + rng.normal(0.0, noise_sigma, y.size)
    kappa = np.zeros_like(t)
    return SyntheticSample(Pose2D(t, x, y), v_true, kappa, "stop_go")
