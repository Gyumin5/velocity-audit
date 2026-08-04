import numpy as np
from velref.io.synthetic import circle, straight_variable_speed, stop_go
from velref.methods.baselines import (
    forward_diff,
    central_diff,
    cubic_spline_deriv,
    smoothing_spline_deriv,
    savgol_deriv,
)
from velref.methods.family_a import family_a_pointwise, family_a_midspan
from velref.metrics.core_metrics import rmse


def test_family_a_beats_central_on_noisy_circle():
    s = circle(duration=20.0, fs=10.0, radius=15.0, v=5.0, noise_sigma=0.05, seed=42)
    v_central = central_diff(s.pose)
    v_a = family_a_pointwise(s.pose, W=5)
    # Compare on the interior to ignore boundary effects.
    interior = slice(5, -5)
    r_central = rmse(v_central[interior], s.v_true[interior])
    r_a = rmse(v_a[interior], s.v_true[interior])
    assert r_a < r_central, f"Family A {r_a:.4f} not < central {r_central:.4f}"


def test_family_a_handles_low_speed_stop_go():
    s = stop_go(duration=30.0, fs=10.0, cruise=5.0, noise_sigma=0.01, seed=7)
    v_central = central_diff(s.pose)
    v_a = family_a_pointwise(s.pose, W=5)
    # Low-speed mask based on truth.
    mask = s.v_true < 0.3
    if mask.sum() < 5:
        return
    osc_central = float(np.sqrt(np.mean(v_central[mask] ** 2)))
    osc_a = float(np.sqrt(np.mean(v_a[mask] ** 2)))
    assert osc_a <= osc_central + 0.1, f"Family A low-speed osc {osc_a} vs central {osc_central}"


def test_straight_variable_speed_recovery():
    s = straight_variable_speed(duration=20.0, fs=10.0, v0=0.5, v1=10.0, noise_sigma=0.02, seed=3)
    v_a = family_a_pointwise(s.pose, W=5)
    interior = slice(5, -5)
    r = rmse(v_a[interior], s.v_true[interior])
    # Should be well below 0.5 m/s (which is the slow-end speed).
    assert r < 0.5


def test_smoothing_spline_runs():
    s = circle(duration=10.0, fs=10.0, noise_sigma=0.02, seed=11)
    v = smoothing_spline_deriv(s.pose)
    assert v.shape == s.v_true.shape
    assert np.all(np.isfinite(v))


def test_savgol_runs():
    s = straight_variable_speed(duration=10.0, fs=10.0, noise_sigma=0.01, seed=13)
    v = savgol_deriv(s.pose, window=7, polyorder=3)
    assert v.shape == s.v_true.shape
