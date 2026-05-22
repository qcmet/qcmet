"""test_rb_fitting.py.

Unit tests for shared randomized benchmarking fitting utilities.
"""

import numpy as np
import pandas as pd
import pytest

from qcmet.benchmarks.gate_execution_quality_metrics.rb_fitting import (
    RBFitConfig,
    add_survival_probabilities,
    average_survival_by_m,
    clip_p0_to_bounds,
    compute_bootstrap_fit_band,
    covariance_parameter_errors,
    fit_curve,
    fit_log_linear,
    get_joint_log_initial_guess,
    get_joint_rb_fit_model,
    get_single_log_initial_guess,
    get_single_rb_fit_model,
    joint_irb_fit_errors,
    single_rb_fit_errors,
    survival_probability_from_counts,
)


def test_rb_fit_config_b0_ideal():
    """Verify ideal baseline is 1 / 2**num_qubits."""
    assert RBFitConfig(num_qubits=1).b0_ideal == 0.5
    assert RBFitConfig(num_qubits=2).b0_ideal == 0.25


def test_survival_probability_from_counts():
    """Verify survival probability is computed from all-zero counts."""
    counts = {"0": 80, "1": 20}
    assert survival_probability_from_counts(counts, num_qubits=1) == 0.8


def test_survival_probability_empty_counts_raises():
    """Verify empty counts raise ValueError."""
    with pytest.raises(ValueError, match="empty counts"):
        survival_probability_from_counts({}, num_qubits=1)


def test_add_survival_probabilities():
    """Verify p_survival column is added."""
    df = pd.DataFrame(
        {
            "m": [0, 1],
            "circuit_measurements": [{"0": 10, "1": 0}, {"0": 5, "1": 5}],
        }
    )

    out = add_survival_probabilities(df, num_qubits=1)

    assert "p_survival" in out.columns
    assert np.allclose(out["p_survival"], [1.0, 0.5])


def test_average_survival_by_m():
    """Verify survival probabilities are averaged by m."""
    df = pd.DataFrame(
        {
            "m": [0, 0, 1, 1],
            "circuit_measurements": [
                {"0": 10, "1": 0},
                {"0": 8, "1": 2},
                {"0": 5, "1": 5},
                {"0": 3, "1": 7},
            ],
        }
    )

    av_df, out_df = average_survival_by_m(df, num_qubits=1)

    assert "p_survival" in out_df.columns
    assert np.allclose(av_df["m"], [0, 1])
    assert np.allclose(av_df["p_survival"], [0.9, 0.4])


@pytest.mark.parametrize(
    "fixed_a0,fixed_b0,expected_names",
    [
        (True, True, ["alpha"]),
        (True, False, ["alpha", "b0"]),
        (False, True, ["alpha", "a0"]),
        (False, False, ["alpha", "a0", "b0"]),
    ],
)
def test_get_single_rb_fit_model_parameter_names(
    fixed_a0,
    fixed_b0,
    expected_names,
):
    """Verify single RB model exposes expected free parameters."""
    config = RBFitConfig(
        num_qubits=1,
        fixed_a0=fixed_a0,
        fixed_b0=fixed_b0,
    )

    fit_func, param_names, bounds, p0 = get_single_rb_fit_model(config)

    assert param_names == expected_names
    assert len(p0) == len(expected_names)
    assert len(bounds[0]) == len(expected_names)
    assert len(bounds[1]) == len(expected_names)


@pytest.mark.parametrize(
    "fixed_a0,fixed_b0,expected_names",
    [
        (True, True, ["alpha", "alpha_g"]),
        (True, False, ["alpha", "alpha_g", "b0"]),
        (False, True, ["alpha", "alpha_g", "a0"]),
        (False, False, ["alpha", "alpha_g", "a0", "b0"]),
    ],
)
def test_get_joint_rb_fit_model_parameter_names(
    fixed_a0,
    fixed_b0,
    expected_names,
):
    """Verify joint RB/IRB model exposes expected free parameters."""
    config = RBFitConfig(
        num_qubits=1,
        fixed_a0=fixed_a0,
        fixed_b0=fixed_b0,
    )

    fit_func, param_names, bounds, p0 = get_joint_rb_fit_model(config)

    assert param_names == expected_names
    assert len(p0) == len(expected_names)
    assert len(bounds[0]) == len(expected_names)
    assert len(bounds[1]) == len(expected_names)


def test_fit_log_linear_fixed_b0():
    """Verify log-linear fit recovers alpha and a0 for synthetic data."""
    config = RBFitConfig(
        num_qubits=1,
        fixed_a0=False,
        fixed_b0=True,
    )

    m = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    alpha_true = 0.95
    a0_true = 0.45
    b0 = config.b0_ideal

    p = a0_true * alpha_true**m + b0

    popt, pcov, names, params = fit_log_linear(m, p, config=config)

    assert names == ["alpha", "a0"]
    assert np.isclose(params["alpha"], alpha_true)
    assert np.isclose(params["a0"], a0_true)


def test_fit_log_linear_requires_known_b0():
    """Verify log fit raises if b0 is not fixed and not explicitly passed."""
    config = RBFitConfig(
        num_qubits=1,
        fixed_b0=False,
    )

    m = np.array([0, 1, 2], dtype=float)
    p = np.array([1.0, 0.9, 0.8], dtype=float)

    with pytest.raises(ValueError, match="known b0"):
        fit_log_linear(m, p, config=config)


def test_clip_p0_to_bounds():
    """Verify initial guesses are clipped into bounds."""
    p0 = [-1, 0.5, 2]
    bounds = ([0, 0, 0], [1, 1, 1])

    clipped = clip_p0_to_bounds(p0, bounds)

    assert clipped == [0.0, 0.5, 1.0]


def test_fit_curve_direct_curve_fit():
    """Verify fit_curve performs nonlinear curve fitting."""
    config = RBFitConfig(
        num_qubits=1,
        fixed_a0=False,
        fixed_b0=True,
        fit_method="curve_fit",
    )

    fit_func, param_names, bounds, p0 = get_single_rb_fit_model(config)

    m = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    alpha_true = 0.95
    a0_true = 0.45
    p = fit_func(m, alpha_true, a0_true)

    popt, pcov, params, log_result = fit_curve(
        xdata=m,
        ydata=p,
        fit_func=fit_func,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        config=config,
    )

    assert log_result is None
    assert np.isclose(params["alpha"], alpha_true)
    assert np.isclose(params["a0"], a0_true)


def test_get_single_log_initial_guess():
    """Verify log initial guess maps to curve_fit parameter order."""
    config = RBFitConfig(
        num_qubits=1,
        fixed_a0=False,
        fixed_b0=False,
    )

    fit_func, param_names, bounds, p0 = get_single_rb_fit_model(config)

    m = np.array([0, 1, 2, 3, 4], dtype=float)
    alpha_true = 0.95
    a0_true = 0.45
    b0_for_log = config.b0_ideal

    p = a0_true * alpha_true**m + b0_for_log

    p0_new, log_result = get_single_log_initial_guess(
        m,
        p,
        config,
        param_names,
        bounds,
        p0,
    )

    assert log_result["status"] == "success"
    assert np.isclose(p0_new[param_names.index("alpha")], alpha_true)
    assert np.isclose(p0_new[param_names.index("a0")], a0_true)
    assert np.isclose(p0_new[param_names.index("b0")], b0_for_log)


def test_get_joint_log_initial_guess():
    """Verify joint log initialisation maps RB and IRB decay rates correctly."""
    config = RBFitConfig(
        num_qubits=1,
        fixed_a0=False,
        fixed_b0=False,
    )

    fit_func, param_names, bounds, p0 = get_joint_rb_fit_model(config)

    m = np.array([0, 1, 2, 3, 4], dtype=float)
    b0_for_log = config.b0_ideal

    alpha_true = 0.96
    alpha_g_true = 0.93
    a0_true = 0.45

    p_rb = a0_true * alpha_true**m + b0_for_log
    p_irb = a0_true * alpha_g_true**m + b0_for_log

    p0_new, log_result = get_joint_log_initial_guess(
        m,
        p_rb,
        m,
        p_irb,
        config,
        param_names,
        bounds,
        p0,
    )

    assert log_result["status"] == "success"
    assert np.isclose(p0_new[param_names.index("alpha")], alpha_true)
    assert np.isclose(p0_new[param_names.index("alpha_g")], alpha_g_true)


def test_covariance_parameter_errors_success():
    """Verify standard errors are extracted from covariance matrix."""
    pcov = np.diag([0.01, 0.04])
    param_names = ["alpha", "a0"]

    result = covariance_parameter_errors(pcov, param_names)

    assert result["status"] == "success"
    assert np.isclose(result["param_stderr"]["alpha"], 0.1)
    assert np.isclose(result["param_stderr"]["a0"], 0.2)


def test_covariance_parameter_errors_invalid_shape():
    """Verify invalid covariance shape fails gracefully."""
    pcov = np.ones((2, 3))
    result = covariance_parameter_errors(pcov, ["alpha", "a0"])

    assert result["status"] == "failed"


def test_single_rb_fit_errors():
    """Verify single RB fit uncertainty propagates to AverageGateError."""
    config = RBFitConfig(num_qubits=1)
    popt = np.array([0.95, 0.5])
    pcov = np.diag([0.0001, 0.0004])
    param_names = ["alpha", "a0"]

    result = single_rb_fit_errors(popt, pcov, param_names, config)

    assert result["status"] == "success"
    assert np.isclose(result["alpha_stderr"], 0.01)
    assert np.isclose(result["AverageGateError_stderr"], 0.005)


def test_joint_irb_fit_errors():
    """Verify joint IRB covariance uncertainty includes interleaved error."""
    config = RBFitConfig(num_qubits=1)

    popt = np.array([0.96, 0.93, 0.45, 0.5])
    pcov = np.diag([0.0001, 0.0001, 0.0004, 0.0004])
    param_names = ["alpha", "alpha_g", "a0", "b0"]

    result = joint_irb_fit_errors(popt, pcov, param_names, config)

    assert result["status"] == "success"
    assert "alpha_stderr" in result
    assert "alpha_g_stderr" in result
    assert "InterleavedGateError_stderr" in result


def test_compute_bootstrap_fit_band():
    """Verify bootstrap fit band returns lower and upper curves."""
    def fit_func(x, alpha):
        return alpha**x

    fit_xs = np.array([0, 1, 2], dtype=float)
    popt_samples = [
        np.array([0.9]),
        np.array([0.95]),
        np.array([0.99]),
    ]

    band = compute_bootstrap_fit_band(
        fit_func=fit_func,
        fit_xs=fit_xs,
        popt_samples=popt_samples,
        confidence_level=0.68,
    )

    assert "lower" in band
    assert "upper" in band
    assert band["lower"].shape == fit_xs.shape
    assert band["upper"].shape == fit_xs.shape
    assert np.all(band["lower"] <= band["upper"])