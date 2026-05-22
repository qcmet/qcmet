"""Shared fitting utilities for randomized benchmarking analyses.

This module provides fitting and uncertainty-estimation routines for Clifford
Randomized Benchmarking and Interleaved Clifford Randomized Benchmarking.

It supports single RB fits of the form

    p_survival(m) = a0 * alpha**m + b0

and linked interleaved RB fits in which the standard and interleaved decay curves
share the same amplitude and baseline parameters.

Fitting can be performed directly using ``scipy.optimize.curve_fit`` or using a
two-step procedure in which a log-linear fit provides initial guesses for a final
nonlinear curve fit. Uncertainties can be estimated either from the final fit
covariance matrix or by bootstrapping survival probabilities within each
sequence length.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class RBFitConfig:
    """Configuration options for randomized benchmarking fits.

    Attributes:
        num_qubits: Number of qubits in the randomized benchmarking experiment.
            This determines the Hilbert-space dimension and the ideal
            depolarized survival probability.
        fixed_a0: If True, fix the RB amplitude parameter ``a0`` to
            ``1 - b0`` during fitting.
        fixed_b0: If True, fix the RB baseline parameter ``b0`` to the ideal
            depolarized-state value ``1 / 2**num_qubits``.
        fit_method: Fitting strategy to use. Supported values are expected to include
            ``"direct"``, ``"two_step"``, and model-specific handling of
            ``"log"`` by callers.
        errorbar_method: Method used to estimate uncertainty. Expected values include
            ``"fit"`` for covariance-based errors and, where supported,
            ``"bootstrap"``.
        bootstrap_samples: Number of bootstrap resamples to draw when using bootstrap-based
            uncertainty estimation.
        bootstrap_seed: Optional random seed used for reproducible bootstrap resampling.
        bootstrap_confidence_level: Confidence level used when summarising bootstrap intervals.
        maxfev: Maximum number of function evaluations passed to
            ``scipy.optimize.curve_fit``.
    
    """

    num_qubits: int
    fixed_a0: bool = False
    fixed_b0: bool = False
    fit_method: str = "two_step"
    errorbar_method: str = "fit"
    bootstrap_samples: int = 1000
    bootstrap_seed: int | None = None
    bootstrap_confidence_level: float = 0.68
    maxfev: int = 200000

    @property
    def b0_ideal(self) -> float:
        """Returns the probablity of measuring |0> in a fully depolarized state."""
        return 1 / 2**self.num_qubits


def survival_probability_from_counts(counts, num_qubits):
    """Return probability of measuring the all-zero state."""
    ground_state = "0" * num_qubits
    total_counts = sum(counts.values())

    if total_counts == 0:
        raise ValueError("Cannot compute survival probability from empty counts.")

    return counts.get(ground_state, 0) / total_counts


def add_survival_probabilities(experiment_data, num_qubits):
    """Add a p_survival column to an experiment dataframe."""
    experiment_data = experiment_data.copy()

    experiment_data["p_survival"] = experiment_data["circuit_measurements"].apply(
        lambda counts: survival_probability_from_counts(counts, num_qubits)
    )

    return experiment_data


def average_survival_by_m(experiment_data, num_qubits):
    """Average survival probabilities by sequence length.

    Computes per-circuit survival probabilities and then groups them by the
    RB sequence length ``m``. The returned averaged dataframe contains one row
    per sequence length.

    Args:
        experiment_data:
            DataFrame containing columns ``m`` and ``circuit_measurements``.
            The ``m`` column gives the sequence length, and
            ``circuit_measurements`` contains shot-count dictionaries.
        num_qubits:
            Number of qubits measured in the experiment.

    Returns:
        A tuple ``(av_p_surv_df, experiment_data_with_survival)`` where:

        - ``av_p_surv_df`` is a dataframe with columns ``m`` and
        ``p_survival``, containing the mean survival probability for each
        sequence length.
        - ``experiment_data_with_survival`` is a copy of the original dataframe
        with an added ``p_survival`` column.

    """
    experiment_data = add_survival_probabilities(experiment_data, num_qubits)

    av_p_surv_df = (
        experiment_data.groupby("m", as_index=False)["p_survival"]
        .mean()
        .sort_values("m")
    )

    return av_p_surv_df, experiment_data


def get_single_rb_fit_model(config: RBFitConfig):
    """Return model, parameter names, bounds and p0 for single RB."""
    fixed_a0 = config.fixed_a0
    fixed_b0 = config.fixed_b0

    b0_fixed = config.b0_ideal

    alpha_guess = 0.999
    a0_guess = 1 - b0_fixed
    b0_guess = b0_fixed

    if fixed_a0 and fixed_b0:

        def fit_func(m, alpha):
            return (1 - b0_fixed) * alpha**m + b0_fixed

        param_names = ["alpha"]
        bounds = ([0.0], [1.0])
        p0 = [alpha_guess]

    elif fixed_a0 and not fixed_b0:

        def fit_func(m, alpha, b0):
            return (1 - b0) * alpha**m + b0

        param_names = ["alpha", "b0"]
        bounds = ([0.0, 0.0], [1.0, 1.0])
        p0 = [alpha_guess, b0_guess]

    elif not fixed_a0 and fixed_b0:

        def fit_func(m, alpha, a0):
            return a0 * alpha**m + b0_fixed

        param_names = ["alpha", "a0"]
        bounds = ([0.0, 0.0], [1.0, 1.0])
        p0 = [alpha_guess, a0_guess]

    else:

        def fit_func(m, alpha, a0, b0):
            return a0 * alpha**m + b0

        param_names = ["alpha", "a0", "b0"]
        bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        p0 = [alpha_guess, a0_guess, b0_guess]

    return fit_func, param_names, bounds, p0


def get_joint_rb_fit_model(config: RBFitConfig):
    """Return joint RB/IRB model with shared a0 and b0.

    RB:
        p(m) = a0 * alpha**m + b0

    IRB:
        p(m) = a0 * alpha_g**m + b0
    """
    fixed_a0 = config.fixed_a0
    fixed_b0 = config.fixed_b0

    b0_fixed = config.b0_ideal

    alpha_guess = 0.999
    alpha_g_guess = 0.999
    a0_guess = 1 - b0_fixed
    b0_guess = b0_fixed

    if fixed_a0 and fixed_b0:

        def fit_func(xdata, alpha, alpha_g):
            m, is_irb = xdata
            decay = np.where(is_irb, alpha_g, alpha)
            return (1 - b0_fixed) * decay**m + b0_fixed

        param_names = ["alpha", "alpha_g"]
        bounds = ([0.0, 0.0], [1.0, 1.0])
        p0 = [alpha_guess, alpha_g_guess]

    elif fixed_a0 and not fixed_b0:

        def fit_func(xdata, alpha, alpha_g, b0):
            m, is_irb = xdata
            decay = np.where(is_irb, alpha_g, alpha)
            return (1 - b0) * decay**m + b0

        param_names = ["alpha", "alpha_g", "b0"]
        bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        p0 = [alpha_guess, alpha_g_guess, b0_guess]

    elif not fixed_a0 and fixed_b0:

        def fit_func(xdata, alpha, alpha_g, a0):
            m, is_irb = xdata
            decay = np.where(is_irb, alpha_g, alpha)
            return a0 * decay**m + b0_fixed

        param_names = ["alpha", "alpha_g", "a0"]
        bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        p0 = [alpha_guess, alpha_g_guess, a0_guess]

    else:

        def fit_func(xdata, alpha, alpha_g, a0, b0):
            m, is_irb = xdata
            decay = np.where(is_irb, alpha_g, alpha)
            return a0 * decay**m + b0

        param_names = ["alpha", "alpha_g", "a0", "b0"]
        bounds = ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        p0 = [alpha_guess, alpha_g_guess, a0_guess, b0_guess]

    return fit_func, param_names, bounds, p0


def fit_log_linear(m_values, p_surv, config: RBFitConfig, b0_for_log=None):
    """Fit RB decay using log-linear transform.

    Uses:

        log(p_survival - b0) = log(a0) + m log(alpha)
    """
    if b0_for_log is None:
        if not config.fixed_b0:
            raise ValueError(
                "Log-linear fitting requires a known b0. Either set fixed_b0=True "
                "or pass b0_for_log explicitly."
            )
        b0_for_log = config.b0_ideal

    m_values = np.asarray(m_values, dtype=float)
    p_surv = np.asarray(p_surv, dtype=float)

    shifted_p = p_surv - b0_for_log
    valid = shifted_p > 0

    if np.count_nonzero(valid) < 2:
        raise ValueError(
            "Not enough valid data points for log-linear fitting. "
            "Need at least two points with p_survival > assumed b0."
        )

    m_fit = m_values[valid]
    log_y = np.log(shifted_p[valid])

    if config.fixed_a0:
        a0_for_log = 1 - b0_for_log
        log_a0 = np.log(a0_for_log)

        x = m_fit
        y = log_y - log_a0

        if np.sum(x**2) == 0:
            raise ValueError(
                "Cannot perform log-linear fit with fixed_a0=True because "
                "all valid sequence lengths are zero."
            )

        log_alpha = np.sum(x * y) / np.sum(x**2)
        alpha = np.exp(log_alpha)

        residuals = y - log_alpha * x
        dof = max(len(x) - 1, 1)
        residual_variance = np.sum(residuals**2) / dof
        var_log_alpha = residual_variance / np.sum(x**2)
        var_alpha = alpha**2 * var_log_alpha

        popt = np.array([alpha])
        pcov = np.array([[var_alpha]])
        param_names = ["alpha"]

    else:
        if len(m_fit) < 3:
            raise ValueError(
                "Need at least three valid data points to estimate alpha, a0, "
                "and covariance with log-linear fitting."
            )

        coeffs, cov = np.polyfit(m_fit, log_y, deg=1, cov=True)

        log_alpha = coeffs[0]
        log_a0 = coeffs[1]

        alpha = np.exp(log_alpha)
        a0 = np.exp(log_a0)

        jacobian = np.diag([alpha, a0])
        pcov = jacobian @ cov @ jacobian.T

        popt = np.array([alpha, a0])
        param_names = ["alpha", "a0"]

    fit_params = dict(zip(param_names, popt))

    return popt, pcov, param_names, fit_params


def clip_p0_to_bounds(p0, bounds):
    """Clip initial guesses so they lie inside curve_fit bounds."""
    lower, upper = bounds

    p0 = np.asarray(p0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    return np.clip(p0, lower, upper).tolist()


def get_single_log_initial_guess(m_values, p_surv, config, param_names, bounds, p0):
    """Use log-linear fit to initialise a single RB curve_fit."""
    b0_for_log = config.b0_ideal

    log_popt, log_pcov, log_names, log_params = fit_log_linear(
        m_values,
        p_surv,
        config=config,
        b0_for_log=b0_for_log,
    )

    p0 = list(p0)

    for name, value in zip(log_names, log_popt):
        if name in param_names:
            p0[param_names.index(name)] = value

    if "b0" in param_names:
        p0[param_names.index("b0")] = b0_for_log

    p0 = clip_p0_to_bounds(p0, bounds)

    log_fit_result = {
        "status": "success",
        "assumed_b0": b0_for_log,
        "popt": log_popt,
        "pcov": log_pcov,
        "param_names": log_names,
        "params": log_params,
    }

    return p0, log_fit_result


def get_joint_log_initial_guess(
    m_rb,
    p_rb,
    m_irb,
    p_irb,
    config,
    param_names,
    bounds,
    p0,
):
    """Use separate log-linear fits to initialise a joint RB/IRB fit."""
    b0_for_log = config.b0_ideal

    rb_popt, rb_pcov, rb_names, rb_params = fit_log_linear(
        m_rb,
        p_rb,
        config=config,
        b0_for_log=b0_for_log,
    )

    irb_popt, irb_pcov, irb_names, irb_params = fit_log_linear(
        m_irb,
        p_irb,
        config=config,
        b0_for_log=b0_for_log,
    )

    p0 = list(p0)

    if "alpha" in param_names:
        p0[param_names.index("alpha")] = rb_params["alpha"]

    if "alpha_g" in param_names:
        p0[param_names.index("alpha_g")] = irb_params["alpha"]

    if "a0" in param_names:
        a0_values = []

        if "a0" in rb_params:
            a0_values.append(rb_params["a0"])

        if "a0" in irb_params:
            a0_values.append(irb_params["a0"])

        if a0_values:
            p0[param_names.index("a0")] = float(np.mean(a0_values))

    if "b0" in param_names:
        p0[param_names.index("b0")] = b0_for_log

    p0 = clip_p0_to_bounds(p0, bounds)

    log_fit_result = {
        "status": "success",
        "assumed_b0": b0_for_log,
        "RB_initial_fit": {
            "popt": rb_popt,
            "pcov": rb_pcov,
            "param_names": rb_names,
            "params": rb_params,
        },
        "IRB_initial_fit": {
            "popt": irb_popt,
            "pcov": irb_pcov,
            "param_names": irb_names,
            "params": irb_params,
        },
    }

    return p0, log_fit_result


def fit_curve(
    xdata,
    ydata,
    fit_func,
    param_names,
    bounds,
    p0,
    config: RBFitConfig,
    log_initial_guess_func=None,
):
    """Fit a curve using log initialisation, direct curve_fit, or log only."""
    fit_method = config.fit_method
    log_fit_result = None

    if fit_method == "log":
        raise ValueError(
            "fit_method='log' should be handled by the caller because log fitting "
            "depends on the model type."
        )

    p0_for_fit = p0

    if fit_method == "two_step" and log_initial_guess_func is not None:
        try:
            p0_for_fit, log_fit_result = log_initial_guess_func()
        except ValueError as exc:
            warnings.warn(
                "Log-linear initialisation failed during two-step fitting. "
                f"Falling back to default p0. Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            log_fit_result = {
                "status": "failed",
                "reason": str(exc),
                "assumed_b0": config.b0_ideal,
            }
            p0_for_fit = p0

    popt, pcov = curve_fit(
        fit_func,
        xdata,
        ydata,
        p0=p0_for_fit,
        bounds=bounds,
        maxfev=config.maxfev,
    )

    fit_params = dict(zip(param_names, popt))

    return popt, pcov, fit_params, log_fit_result


def single_rb_fit_errors(popt, pcov, param_names, config: RBFitConfig):
    """""Compute covariance-based uncertainties for a single RB fit."""
    result = covariance_parameter_errors(pcov, param_names)

    if result["status"] != "success":
        return {"method": "fit", **result}

    param_stderr = result["param_stderr"]
    d = 2**config.num_qubits

    alpha_stderr = param_stderr["alpha"]
    avg_gate_err_stderr = ((d - 1) / d) * alpha_stderr

    return {
        "method": "fit",
        "status": "success",
        "param_stderr": param_stderr,
        "alpha_stderr": float(alpha_stderr),
        "AverageGateError_stderr": float(avg_gate_err_stderr),
    }


def joint_irb_fit_errors(popt, pcov, param_names, config: RBFitConfig):
    """Compute covariance-based uncertainties for a joint RB/IRB fit."""
    result = covariance_parameter_errors(pcov, param_names)

    if result["status"] != "success":
        return {"method": "fit", **result}

    param_stderr = result["param_stderr"]
    fit_params = dict(zip(param_names, popt))

    alpha = fit_params["alpha"]
    alpha_g = fit_params["alpha_g"]

    alpha_index = param_names.index("alpha")
    alpha_g_index = param_names.index("alpha_g")

    d = 2**config.num_qubits
    prefactor = (d - 1) / d

    avg_gate_err_stderr = prefactor * param_stderr["alpha"]

    grad = np.zeros(len(param_names))
    grad[alpha_index] = prefactor * alpha_g / alpha**2
    grad[alpha_g_index] = -prefactor / alpha

    int_gate_err_var = grad @ pcov @ grad
    int_gate_err_stderr = np.sqrt(max(int_gate_err_var, 0.0))

    return {
        "method": "fit",
        "status": "success",
        "param_stderr": param_stderr,
        "alpha_stderr": float(param_stderr["alpha"]),
        "alpha_g_stderr": float(param_stderr["alpha_g"]),
        "AverageGateError_stderr": float(avg_gate_err_stderr),
        "InterleavedGateError_stderr": float(int_gate_err_stderr),
    }


def bootstrap_grouped_survival(
    grouped_values,
    m_values,
    fit_sample_func,
    config: RBFitConfig,
):
    """Do bootstrapping for grouped survival probabilities.

    Args:
        grouped_values:
            Dict mapping m -> array of per-circuit survival probabilities.
        m_values:
            Sequence lengths to bootstrap.
        fit_sample_func:
            Callable taking bootstrapped p_surv and returning a result dict.
        config:
            Fit configuration.

    Returns:
        dict with bootstrap samples and counts.

    """
    rng = np.random.default_rng(config.bootstrap_seed)

    samples = []
    failed = 0

    for _ in range(config.bootstrap_samples):
        try:
            boot_p = []

            for m in m_values:
                values = grouped_values[m]
                resampled = rng.choice(values, size=len(values), replace=True)
                boot_p.append(np.mean(resampled))

            boot_p = np.asarray(boot_p, dtype=float)
            samples.append(fit_sample_func(boot_p))

        except Exception:
            failed += 1

    return samples, failed


def covariance_parameter_errors(pcov, param_names):
    """Return standard errors from covariance matrix."""
    if pcov is None:
        return {
            "status": "failed",
            "reason": "Fit covariance matrix is None.",
        }

    pcov = np.asarray(pcov, dtype=float)

    if pcov.ndim != 2 or pcov.shape[0] != pcov.shape[1]:
        return {
            "status": "failed",
            "reason": "Fit covariance matrix has invalid shape.",
        }

    variances = np.diag(pcov)

    if np.any(variances < 0):
        return {
            "status": "failed",
            "reason": "Fit covariance matrix contains negative variances.",
        }

    stderr = np.sqrt(variances)

    return {
        "status": "success",
        "param_stderr": dict(zip(param_names, stderr)),
    }


def compute_bootstrap_fit_band(
    fit_func,
    fit_xs,
    popt_samples,
    confidence_level,
):
    """Compute a bootstrap confidence band for a fitted function.

    Args:
        fit_func:
            Model function.
        fit_xs:
            x points to evaluate.
        popt_samples:
            list/array of fitted parameter sets from bootstrap
        confidence_level:
            central CI (e.g. 0.68 or 0.95)

    Returns:
        dict:
            {
                "lower": array,
                "upper": array,
            }

    """
    curves = []

    for popt in popt_samples:
        curves.append(fit_func(fit_xs, *popt))

    curves = np.asarray(curves)

    lower_percentile = 100 * (1 - confidence_level) / 2
    upper_percentile = 100 * (1 + confidence_level) / 2

    lower = np.percentile(curves, lower_percentile, axis=0)
    upper = np.percentile(curves, upper_percentile, axis=0)

    return {
        "lower": lower,
        "upper": upper,
    }